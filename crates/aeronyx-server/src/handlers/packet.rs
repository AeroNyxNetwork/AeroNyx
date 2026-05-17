// ============================================
// File: crates/aeronyx-server/src/handlers/packet.rs
// ============================================
//! # Packet Handler
//!
//! ## Creation Reason
//! Handles data packet processing including encryption, decryption,
//! and forwarding between UDP and TUN interfaces.
//!
//! ## Modification Reason
//! Added MemChain 1st-byte multiplexing support. After decryption,
//! the plaintext's first byte is inspected:
//!   - `0x4X` (IPv4) → normal VPN path (validate source IP, forward to TUN)
//!   - `0xAE` disc 0-17 → MemChain message → MemPool
//!   - `0xAE` disc 31-33 → Voice signal → forward to target session by wallet
//!   - `0xAF` → Voice audio frame → forward to target session by dst_ip
//!
//! ## Traffic Accounting Fix (v1.0.0-TrafficFix)
//! record_rx() is called ONLY inside VPN arms (IPv4/IPv6).
//! MemChain and Voice control messages do NOT increment bytes_rx.
//!
//! ## Voice Signal Routing (v1.0.0-VoiceSignal)
//! disc=31 (Offer), disc=32 (Answer), disc=33 (End) inside 0xAE packets
//! are voice call signals. They carry a JSON payload with a "pubkey" field
//! identifying the target wallet. The server extracts the pubkey, looks up
//! the target session, re-encrypts and forwards the full payload.
//! This fixes the error:
//!   "invalid value: integer `31`, expected variant index 0 <= i < 18"
//! which occurred because the server tried to decode disc=31 as a MemChain
//! variant (only 0-17 are valid).
//!
//! ## ⚠️ Important Note for Next Developer
//! - Performance critical — minimize allocations
//! - Counter validation prevents replay attacks
//! - Source IP validation is now advisory (debug log only) for iOS/macOS NE
//!   compatibility — Apple's NetworkExtension framework sends packets with the
//!   device's real LAN IP as src, not the VPN virtual IP
//! - MemChain path deliberately skips IP validation (no IP header)
//! - MemChain messages and Voice signals are NOT counted in bytes_rx
//! - record_rx uses plaintext_len (decrypted bytes), not raw UDP wire length
//! - record_tx uses ip_packet.len() (plaintext), not output.len() (encrypted)
//! - MEMCHAIN_MAGIC (0xAE) must stay in sync with aeronyx_core
//! - VOICE_MAGIC (0xAF) must stay in sync with client-side magic bytes
//! - Voice signal discriminants 31/32/33 must match client VoiceFrameBuilder
//!
//! ## Last Modified
//! v0.1.0 - Initial packet handler
//! v0.2.0 - Added MemChain 1st-byte multiplexing (DecryptedPayload enum)
//! v1.0.0-TrafficFix - record_rx() moved inside VPN arms only
//! v1.0.0-VoiceSignal - disc=31/32/33 routed as voice signals, not MemChain

use std::net::Ipv4Addr;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tracing::{debug, trace, warn};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::{decode_data_packet, encode_data_packet};
use aeronyx_core::protocol::memchain::{decode_memchain, MEMCHAIN_MAGIC};
use aeronyx_core::protocol::messages::DATA_PACKET_HEADER_SIZE;
use aeronyx_core::protocol::{DataPacket, MemChainMessage};

use crate::error::{Result, ServerError};
use crate::services::{RoutingService, Session, SessionManager};

// ============================================
// Constants
// ============================================

/// Minimum IPv4 header size.
const IPV4_HEADER_MIN_SIZE: usize = 20;

/// Offset of destination IP in IPv4 header.
const IPV4_DST_OFFSET: usize = 16;

/// Offset of source IP in IPv4 header.
const IPV4_SRC_OFFSET: usize = 12;

/// Magic byte identifying a Voice audio frame packet (0xAF).
///
/// High nibble = 0xA (10), distinct from IPv4 (0x4), IPv6 (0x6), MemChain (0xAE).
///
/// Wire format after stripping magic byte:
///   [dst_virtual_ip: 4 bytes][payload: variable]
pub const VOICE_MAGIC: u8 = 0xAF;

/// Minimum size of a voice audio packet: magic(1) + dst_ip(4) = 5 bytes.
const VOICE_PACKET_MIN_SIZE: usize = 5;

/// Offset of dst IP inside a voice audio packet plaintext.
const VOICE_DST_IP_OFFSET: usize = 1;

/// Voice signal discriminant: CallOffer.
pub const DISC_VOICE_OFFER: u32 = 31;
/// Voice signal discriminant: CallAnswer.
pub const DISC_VOICE_ANSWER: u32 = 32;
/// Voice signal discriminant: CallEnd.
pub const DISC_VOICE_END: u32 = 33;

// ============================================
// DecryptedPayload
// ============================================

/// Result of decrypting an incoming DataPacket.
///
/// Multiplexer dispatch table:
/// | plaintext[0] | discriminant | Variant       |
/// |--------------|--------------|---------------|
/// | 0x4X (IPv4)  | —            | Vpn           |
/// | 0x6X (IPv6)  | —            | Vpn           |
/// | 0xAE         | 0-17         | MemChain      |
/// | 0xAE         | 31-33        | VoiceSignal   |
/// | 0xAF         | —            | Voice         |
///
/// ## Traffic accounting
/// Only `Vpn` increments `session.stats.bytes_rx`.
/// All other variants are control/signal traffic — not billed.
#[derive(Debug)]
pub enum DecryptedPayload {
    /// Standard VPN IP packet — write to TUN.
    /// `bytes_rx` already incremented before returning.
    Vpn(Vec<u8>),

    /// MemChain application message (disc 0-17) — route to MemPool.
    MemChain(MemChainMessage),

    /// Voice call signal (disc 31=Offer, 32=Answer, 33=End).
    /// Carried inside 0xAE packets with discriminant outside MemChain range.
    /// Server extracts `target_wallet` from JSON payload and forwards `payload`
    /// to that wallet's session (re-encrypted with the target's session key).
    VoiceSignal {
        /// Discriminant (31, 32, or 33).
        discriminant: u32,
        /// Target wallet pubkey hex from JSON "pubkey" field.
        /// None if JSON parsing failed — caller should drop.
        target_wallet: Option<String>,
        /// Full plaintext (including 0xAE magic + discriminant).
        /// Re-encrypted and forwarded to target session unchanged.
        payload: Vec<u8>,
    },

    /// Voice audio frame (0xAF) — forward to target session by dst_ip.
    Voice {
        /// Destination virtual IP (100.64.0.x) from plaintext[1..5].
        dst_ip: Ipv4Addr,
        /// Full plaintext including magic byte and dst_ip header.
        payload: Vec<u8>,
    },
}

// ============================================
// PacketHandler
// ============================================

/// Handles data packet processing.
///
/// # Thread Safety
/// All operations are thread-safe and can be called concurrently.
pub struct PacketHandler {
    sessions: Arc<SessionManager>,
    routing:  Arc<RoutingService>,
    crypto:   DefaultTransportCrypto,
}

impl PacketHandler {
    pub fn new(sessions: Arc<SessionManager>, routing: Arc<RoutingService>) -> Self {
        Self { sessions, routing, crypto: DefaultTransportCrypto::new() }
    }

    /// Processes an incoming UDP data packet.
    ///
    /// Returns `(session, DecryptedPayload)` on success.
    /// Returns `Err(ServerError::SessionNotFound)` when the session ID is
    /// unknown — caller should send a 0xFF RESET to the client.
    pub fn handle_udp_packet(&self, data: &[u8]) -> Result<(Arc<Session>, DecryptedPayload)> {
        // ── Minimum size check ────────────────────────────────────────────
        if data.len() < DATA_PACKET_HEADER_SIZE + ENCRYPTION_OVERHEAD {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "Packet too short",
            ));
        }

        // ── Debug logging ─────────────────────────────────────────────────
        let raw_session_id = &data[0..16];
        debug!(
            "[PACKET_HANDLER] Processing packet, raw SessionID bytes: {:02X?}",
            raw_session_id
        );
        debug!(
            "[PACKET_HANDLER] SessionID (base64): {}",
            BASE64.encode(raw_session_id)
        );

        // ── Decode packet header ──────────────────────────────────────────
        let packet = decode_data_packet(data)?;

        debug!(
            "[PACKET_HANDLER] Decoded packet.session_id (base64): {}",
            BASE64.encode(&packet.session_id)
        );

        // ── Session lookup ────────────────────────────────────────────────
        let session_id = SessionId::from_bytes(&packet.session_id)
            .ok_or_else(|| {
                warn!(
                    "[PACKET_HANDLER] Invalid session ID format: {:02X?}",
                    &packet.session_id
                );
                ServerError::invalid_packet(
                    "0.0.0.0:0".parse().unwrap(),
                    "Invalid session ID",
                )
            })?;

        debug!("[PACKET_HANDLER] Looking up SessionID: {}", session_id);
        debug!("[PACKET_HANDLER] Active sessions count: {}", self.sessions.count());

        let session = match self.sessions.get(&session_id) {
            Some(s) => {
                debug!(
                    "[PACKET_HANDLER] Session FOUND: {}, virtual_ip={}, endpoint={}",
                    session_id, s.virtual_ip, s.client_endpoint
                );
                s
            }
            None => {
                warn!(
                    "[PACKET_HANDLER] Session NOT FOUND: {} (base64: {})",
                    session_id,
                    BASE64.encode(session_id.as_bytes())
                );
                return Err(self.sessions.get_or_error(&session_id).unwrap_err());
            }
        };

        // ── Replay counter validation ─────────────────────────────────────
        if !session.validate_rx_counter(packet.counter) {
            warn!(
                session_id = %session_id,
                counter = packet.counter,
                "Replay attack detected"
            );
            return Err(ServerError::Core(aeronyx_core::error::CoreError::replay(
                packet.counter,
                session.rx_counter.load(std::sync::atomic::Ordering::SeqCst),
            )));
        }

        // ── Decrypt ───────────────────────────────────────────────────────
        let mut plaintext = vec![0u8; packet.encrypted_payload.len()];
        let plaintext_len = self.crypto.decrypt(
            &session.session_key,
            packet.counter,
            &packet.session_id,
            &packet.encrypted_payload,
            &mut plaintext,
        )?;
        plaintext.truncate(plaintext_len);

        // ── Update session activity (all paths) ───────────────────────────
        session.touch();

        // ====================================================================
        // 1st-Byte Multiplexing + Traffic Accounting
        // ====================================================================
        let payload = match plaintext.first().copied() {

            // ── IPv4 VPN (most common) ────────────────────────────────────
            Some(b) if b >> 4 == 4 => {
                if plaintext_len < IPV4_HEADER_MIN_SIZE {
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "IP packet too short",
                    ));
                }

                // Source IP validation is advisory only.
                // Apple NetworkExtension sends packets with the device's real
                // LAN IP as src (e.g. 10.x.x.x), not the VPN virtual IP.
                // Session-key authentication already proves identity.
                let src_ip = extract_ipv4_src(&plaintext)?;
                if src_ip != session.virtual_ip {
                    debug!(
                        session_id = %session_id,
                        expected = %session.virtual_ip,
                        actual = %src_ip,
                        "src IP mismatch (iOS/macOS NE — packet accepted)"
                    );
                }

                // ✅ Count inbound VPN user-data bytes.
                session.stats.record_rx(plaintext_len as u64);

                trace!(session_id = %session_id, len = plaintext_len, "VPN IPv4 decrypted");
                DecryptedPayload::Vpn(plaintext)
            }

            // ── IPv6 VPN (pass-through) ───────────────────────────────────
            Some(b) if b >> 4 == 6 => {
                // ✅ Count inbound VPN user-data bytes.
                session.stats.record_rx(plaintext_len as u64);
                trace!(session_id = %session_id, len = plaintext_len, "VPN IPv6 decrypted");
                DecryptedPayload::Vpn(plaintext)
            }

            // ── 0xAE: MemChain or Voice Signal ───────────────────────────
            Some(MEMCHAIN_MAGIC) => {
                // Need at least magic(1) + discriminant(4) = 5 bytes.
                if plaintext_len < 5 {
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "0xAE packet too short for discriminant",
                    ));
                }

                // Read discriminant (u32 LE) from plaintext[1..5].
                let disc = u32::from_le_bytes([
                    plaintext[1], plaintext[2], plaintext[3], plaintext[4],
                ]);

                // Voice signal discriminants: 31=Offer, 32=Answer, 33=End.
                // These are NOT MemChain variants — route by target wallet pubkey.
                if matches!(disc, DISC_VOICE_OFFER | DISC_VOICE_ANSWER | DISC_VOICE_END) {
                    // JSON payload starts at plaintext[5].
                    // Format: {"type":"offer","call_id":"...","pubkey":"<64-char hex>"}
                    let target_wallet = extract_pubkey_from_json(&plaintext[5..]);

                    trace!(
                        session_id = %session_id,
                        disc,
                        target = %target_wallet.as_deref().unwrap_or("unknown"),
                        "[VOICE_SIGNAL] Signal received"
                    );

                    return Ok((session, DecryptedPayload::VoiceSignal {
                        discriminant: disc,
                        target_wallet,
                        payload: plaintext,
                    }));
                }

                // MemChain variants (disc 0-17): decode normally.
                // ⚠️ Do NOT call record_rx() — control traffic is not billed.
                let memchain_payload = &plaintext[1..];

                match decode_memchain(memchain_payload) {
                    Ok(msg) => {
                        debug!(
                            session_id = %session_id,
                            msg_type = ?std::mem::discriminant(&msg),
                            "[MEMCHAIN] Message decoded"
                        );
                        DecryptedPayload::MemChain(msg)
                    }
                    Err(e) => {
                        let dump_len = plaintext.len().min(16);
                        warn!(
                            session_id = %session_id,
                            error = %e,
                            plaintext_total_len = plaintext.len(),
                            header_hex = %hex::encode(&plaintext[..dump_len]),
                            discriminant_bytes = %hex::encode(&plaintext[1..plaintext.len().min(5)]),
                            "[MEMCHAIN] Failed to decode payload"
                        );
                        return Err(ServerError::invalid_packet(
                            session.client_endpoint,
                            format!("MemChain decode error: {}", e),
                        ));
                    }
                }
            }

            // ── 0xAF: Voice audio frame ───────────────────────────────────
            Some(VOICE_MAGIC) => {
                if plaintext_len < VOICE_PACKET_MIN_SIZE {
                    warn!(
                        session_id = %session_id,
                        len = plaintext_len,
                        "[VOICE] Packet too short (need {} bytes for dst_ip)",
                        VOICE_PACKET_MIN_SIZE
                    );
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "Voice packet too short",
                    ));
                }

                let mut dst_octets = [0u8; 4];
                dst_octets.copy_from_slice(
                    &plaintext[VOICE_DST_IP_OFFSET..VOICE_DST_IP_OFFSET + 4]
                );
                let dst_ip = Ipv4Addr::from(dst_octets);

                trace!(
                    session_id = %session_id,
                    dst_ip = %dst_ip,
                    len = plaintext_len,
                    "[VOICE] Audio frame decoded"
                );

                DecryptedPayload::Voice { dst_ip, payload: plaintext }
            }

            // ── Unknown ───────────────────────────────────────────────────
            Some(b) => {
                warn!(
                    session_id = %session_id,
                    first_byte = format_args!("0x{:02X}", b),
                    "Unknown plaintext type"
                );
                return Err(ServerError::invalid_packet(
                    session.client_endpoint,
                    format!("Unknown plaintext first byte: 0x{:02X}", b),
                ));
            }

            None => {
                return Err(ServerError::invalid_packet(
                    session.client_endpoint,
                    "Empty plaintext after decryption",
                ));
            }
        };

        Ok((session, payload))
    }

    /// Processes an IP packet from the TUN device (outbound: server → client).
    ///
    /// ## Traffic accounting
    /// Uses ip_packet.len() (plaintext bytes), not output.len() (encrypted).
    /// Billing should reflect user data transferred, not wire overhead.
    pub fn handle_tun_packet(
        &self,
        ip_packet: &[u8],
    ) -> Result<(Vec<u8>, std::net::SocketAddr)> {
        if ip_packet.len() < IPV4_HEADER_MIN_SIZE {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "TUN packet too short",
            ));
        }

        let dst_ip    = extract_ipv4_dst(ip_packet)?;
        let session_id = self.routing.lookup_or_error(dst_ip)?;
        let session   = self.sessions.get_or_error(&session_id)?;
        let counter   = session.next_tx_counter();

        let mut encrypted = vec![0u8; ip_packet.len() + ENCRYPTION_OVERHEAD];
        let actual_len = self.crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            ip_packet,
            &mut encrypted,
        )?;
        encrypted.truncate(actual_len);

        let data_packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let output = encode_data_packet(&data_packet).to_vec();

        // ✅ Count outbound VPN user-data bytes (plaintext length).
        session.touch();
        session.stats.record_tx(ip_packet.len() as u64);

        trace!(
            session_id = %session_id,
            dst_ip = %dst_ip,
            plaintext_len = ip_packet.len(),
            wire_len = output.len(),
            "TUN packet encrypted and sent"
        );

        Ok((output, session.client_endpoint))
    }
}

impl std::fmt::Debug for PacketHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacketHandler").finish()
    }
}

// ============================================
// Helper Functions
// ============================================

fn extract_ipv4_src(packet: &[u8]) -> Result<Ipv4Addr> {
    if packet.len() < IPV4_SRC_OFFSET + 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            "Packet too short for IPv4 src",
        ));
    }
    if packet[0] >> 4 != 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            format!("Expected IPv4, got version {}", packet[0] >> 4),
        ));
    }
    let mut octets = [0u8; 4];
    octets.copy_from_slice(&packet[IPV4_SRC_OFFSET..IPV4_SRC_OFFSET + 4]);
    Ok(Ipv4Addr::from(octets))
}

fn extract_ipv4_dst(packet: &[u8]) -> Result<Ipv4Addr> {
    if packet.len() < IPV4_DST_OFFSET + 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            "Packet too short for IPv4 dst",
        ));
    }
    if packet[0] >> 4 != 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            format!("Expected IPv4, got version {}", packet[0] >> 4),
        ));
    }
    let mut octets = [0u8; 4];
    octets.copy_from_slice(&packet[IPV4_DST_OFFSET..IPV4_DST_OFFSET + 4]);
    Ok(Ipv4Addr::from(octets))
}

/// Extracts the routing target from a voice signal JSON payload.
///
/// Reads the `"to"` field (target wallet pubkey hex) for routing.
/// The `"pubkey"` field identifies the sender — kept for the receiver
/// to display caller identity, but NOT used for routing.
///
/// Expected format:
/// ```json
/// {
///   "type":    "offer",
///   "call_id": "<base64>",
///   "pubkey":  "<sender 64-char hex>",
///   "to":      "<target 64-char hex>"
/// }
/// ```
///
/// Returns `Some(hex_string)` if a valid 64-char hex `"to"` value is found.
fn extract_pubkey_from_json(json_bytes: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(json_bytes).ok()?;

    // Look for the "to" field (routing target).
    // Falls back to "pubkey" for backward compatibility during transition.
    extract_json_hex_field(s, "\"to\"")
        .or_else(|| extract_json_hex_field(s, "\"pubkey\""))
}

/// Extracts a 64-char hex string value for the given JSON key.
/// Simple byte scanning — no JSON parser dependency on the hot path.
fn extract_json_hex_field<'a>(s: &'a str, key: &str) -> Option<String> {
    let key_pos = s.find(key)?;
    let after_key = &s[key_pos + key.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let value_start = &after_colon[1..];
    let end = value_start.find('"')?;
    let hex = &value_start[..end];
    if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(hex.to_string())
    } else {
        None
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ipv4(src: Ipv4Addr, dst: Ipv4Addr) -> Vec<u8> {
        let mut p = vec![0u8; 20];
        p[0] = 0x45;
        p[2] = 0x00; p[3] = 0x14;
        p[IPV4_SRC_OFFSET..IPV4_SRC_OFFSET + 4].copy_from_slice(&src.octets());
        p[IPV4_DST_OFFSET..IPV4_DST_OFFSET + 4].copy_from_slice(&dst.octets());
        p
    }

    #[test]
    fn test_extract_ipv4_src() {
        let src = Ipv4Addr::new(192, 168, 1, 100);
        let dst = Ipv4Addr::new(8, 8, 8, 8);
        assert_eq!(extract_ipv4_src(&make_ipv4(src, dst)).unwrap(), src);
    }

    #[test]
    fn test_extract_ipv4_dst() {
        let src = Ipv4Addr::new(192, 168, 1, 100);
        let dst = Ipv4Addr::new(8, 8, 8, 8);
        assert_eq!(extract_ipv4_dst(&make_ipv4(src, dst)).unwrap(), dst);
    }

    #[test]
    fn test_extract_from_short_packet() {
        let short = vec![0x45, 0x00];
        assert!(extract_ipv4_src(&short).is_err());
        assert!(extract_ipv4_dst(&short).is_err());
    }

    #[test]
    fn test_extract_wrong_version() {
        let mut p = vec![0u8; 20];
        p[0] = 0x60; // IPv6
        assert!(extract_ipv4_src(&p).is_err());
    }

    #[test]
    fn test_magic_bytes_no_collision() {
        // MEMCHAIN_MAGIC = 0xAE, VOICE_MAGIC = 0xAF
        // Neither collides with IPv4 (0x4X) or IPv6 (0x6X)
        assert_ne!(MEMCHAIN_MAGIC >> 4, 4u8);
        assert_ne!(MEMCHAIN_MAGIC >> 4, 6u8);
        assert_ne!(VOICE_MAGIC >> 4,    4u8);
        assert_ne!(VOICE_MAGIC >> 4,    6u8);
        assert_ne!(VOICE_MAGIC, MEMCHAIN_MAGIC);
    }

    #[test]
    fn test_first_byte_routing_invariants() {
        assert_eq!(0x45u8 >> 4, 4u8); // IPv4
        assert_eq!(0x60u8 >> 4, 6u8); // IPv6
        assert_eq!(MEMCHAIN_MAGIC >> 4, 10u8);
        assert_eq!(VOICE_MAGIC,         0xAFu8);
    }

    #[test]
    fn test_voice_packet_dst_ip_extraction() {
        let dst = Ipv4Addr::new(100, 64, 0, 5);
        let mut p = vec![VOICE_MAGIC];
        p.extend_from_slice(&dst.octets());
        p.extend_from_slice(b"audio");
        assert!(p.len() >= VOICE_PACKET_MIN_SIZE);
        let mut octets = [0u8; 4];
        octets.copy_from_slice(&p[VOICE_DST_IP_OFFSET..VOICE_DST_IP_OFFSET + 4]);
        assert_eq!(Ipv4Addr::from(octets), dst);
    }

    #[test]
    fn test_voice_signal_discriminants() {
        assert_eq!(DISC_VOICE_OFFER,  31u32);
        assert_eq!(DISC_VOICE_ANSWER, 32u32);
        assert_eq!(DISC_VOICE_END,    33u32);
        // All outside MemChain range 0-17
        assert!(DISC_VOICE_OFFER  >= 18);
        assert!(DISC_VOICE_ANSWER >= 18);
        assert!(DISC_VOICE_END    >= 18);
    }

    #[test]
    fn test_extract_pubkey_from_json() {
        let sender = "437059fc0f5403365c69ef9ba6df93cdb0dc1f0058acee6af647f0b4152a7010";
        let target = "99e2b4602033e7b8c744232a1a74a3ec8b6c82c9dfedf66406f09b779071f753";

        // New format: "to" field is used for routing (target wallet)
        let json_new = format!(
            r#"{{"type":"offer","call_id":"abc","pubkey":"{}","to":"{}"}}"#,
            sender, target
        );
        // "to" takes priority over "pubkey"
        assert_eq!(
            extract_pubkey_from_json(json_new.as_bytes()).as_deref(),
            Some(target),
            "should extract 'to' field for routing"
        );

        // Old format (no "to" field): fall back to "pubkey" for backward compat
        let json_old = format!(
            r#"{{"type":"offer","call_id":"abc","pubkey":"{}"}}"#,
            target
        );
        assert_eq!(
            extract_pubkey_from_json(json_old.as_bytes()).as_deref(),
            Some(target),
            "should fall back to 'pubkey' when 'to' is absent"
        );

        // Missing both fields
        assert!(extract_pubkey_from_json(br#"{"type":"offer"}"#).is_none());

        // "to" present but too short
        let bad = format!(r#"{{"to":"deadbeef","pubkey":"{}"}}"#, target);
        assert_eq!(
            extract_pubkey_from_json(bad.as_bytes()).as_deref(),
            Some(target),
            "should fall back to valid 'pubkey' when 'to' is invalid"
        );
    }
}
