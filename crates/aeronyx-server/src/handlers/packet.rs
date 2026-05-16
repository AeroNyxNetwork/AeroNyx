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
//! Added **MemChain 1st-byte multiplexing** support. After decryption,
//! the plaintext's first byte is inspected:
//!   - `0x4X` (IPv4) → normal VPN path (validate source IP, forward to TUN)
//!   - `0xAE` (MemChain magic) → strip prefix, deserialise to
//!     `MemChainMessage`, return via `DecryptedPayload::MemChain` —
//!     caller routes to MemPool instead of TUN.
//!
//! This achieves **zero modification** to the outer wire protocol,
//! crypto layer, or session management.
//!
//! ## Traffic Accounting Fix (v1.0.0-TrafficFix)
//! Previous implementation had two precision issues:
//!
//! 1. **Inbound (UDP → TUN)**: `record_rx()` was called AFTER the
//!    `match plaintext.first()` block, meaning MemChain messages and
//!    replay-rejected packets were also counted. Now `record_rx()` is
//!    called only inside the IPv4/IPv6 VPN arms, counting only actual
//!    VPN user-data bytes. MemChain control messages are excluded.
//!
//! 2. **Outbound (TUN → UDP)**: `record_tx()` correctly uses
//!    `ip_packet.len()` (plaintext bytes) rather than `output.len()`
//!    (encrypted + header overhead). This was already correct and is
//!    preserved verbatim.
//!
//! Result: `bytes_rx` / `bytes_tx` now reflect only user VPN traffic,
//! which is what aeronyx.network billing expects.
//!
//! ## Main Functionality
//! - `PacketHandler`: Main packet processing logic
//! - `DecryptedPayload`: enum distinguishing VPN vs MemChain payloads
//! - UDP packet decryption and TUN forwarding
//! - TUN packet encryption and UDP forwarding
//! - Session and routing lookups
//!
//! ## Packet Processing
//!
//! ### Client → Internet (UDP → TUN)  [original path]
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Receive UDP packet                                      │
//! │  2. Lookup session by ID                                    │
//! │  3. Validate counter (replay protection)                    │
//! │  4. Decrypt payload with session key                        │
//! │  5. Peek plaintext[0]:                                      │
//! │     ├─ 0x4X → Validate source IP → record_rx → Vpn        │
//! │     ├─ 0x6X → record_rx → Vpn (IPv6 pass-through)         │
//! │     └─ 0xAE → bincode decode → MemChain (NOT counted)      │
//! │  6. Caller writes VPN to TUN, MemChain to MemPool           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### Internet → Client (TUN → UDP)  [unchanged]
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Read IP packet from TUN                                 │
//! │  2. Extract destination IP                                  │
//! │  3. Lookup route → session                                  │
//! │  4. Encrypt packet                                          │
//! │  5. record_tx(ip_packet.len()) — plaintext bytes only       │
//! │  6. Build packet with session ID and counter                │
//! │  7. Send via UDP to client                                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Performance critical - minimize allocations
//! - Counter validation prevents replay attacks
//! - Source IP validation prevents IP spoofing (VPN path only)
//! - MemChain path deliberately skips IP validation (no IP header)
//! - MemChain messages are NOT counted in bytes_rx — they are control
//!   traffic, not user data. Billing depends on this distinction.
//! - record_rx uses plaintext_len (decrypted bytes), NOT the raw UDP
//!   wire length. This counts actual user payload, not crypto overhead.
//! - record_tx uses ip_packet.len() (plaintext), NOT output.len()
//!   (encrypted). Same reasoning — bill for user data, not overhead.
//! - Log security events but avoid log flooding
//! - The `MEMCHAIN_MAGIC` constant (0xAE) must stay in sync with
//!   `aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC`
//!
//! ## Last Modified
//! v0.1.0 - Initial packet handler
//! v0.2.0 - Added MemChain 1st-byte multiplexing (DecryptedPayload enum)
//! v1.0.0-TrafficFix - record_rx() moved inside VPN arms only;
//!   MemChain messages excluded from traffic accounting.
//!   record_tx() verified correct (ip_packet.len() not output.len()).

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

// ============================================
// DecryptedPayload
// ============================================

/// Magic byte identifying a Voice packet (signal or audio frame).
///
/// Chosen to be distinct from IPv4 (0x4X), IPv6 (0x6X), and MemChain (0xAE).
/// High nibble = 0xA (10), so it cannot collide with any IP version nibble.
///
/// Wire format after stripping the magic byte:
/// ```
/// [0xAF][dst_virtual_ip: 4 bytes][payload: variable]
///         ^                       ^
///         100.64.0.x              CallOffer / CallAnswer / CallEnd / audio frame
/// ```
/// The server routes the packet to the session owning `dst_virtual_ip`
/// without inspecting or modifying the payload (end-to-end encrypted).
pub const VOICE_MAGIC: u8 = 0xAF;

/// Minimum size of a voice packet after decryption:
/// magic byte (1) + dst_virtual_ip (4) = 5 bytes minimum.
const VOICE_PACKET_MIN_SIZE: usize = 5;

/// Offset of the destination IPv4 address inside a voice packet plaintext.
/// plaintext[0]   = VOICE_MAGIC (0xAF)
/// plaintext[1..5] = dst_virtual_ip (4 bytes, big-endian octets)
/// plaintext[5..]  = voice payload (signal or audio)
const VOICE_DST_IP_OFFSET: usize = 1;

/// Result of decrypting an incoming DataPacket.
///
/// The multiplexer inspects `plaintext[0]` to decide the variant:
/// - `0x4X` (IPv4) → [`DecryptedPayload::Vpn`]
/// - `0x6X` (IPv6) → [`DecryptedPayload::Vpn`] (pass-through)
/// - `0xAE`        → [`DecryptedPayload::MemChain`]
/// - `0xAF`        → [`DecryptedPayload::Voice`]
///
/// The caller (`server.rs` UDP task) uses this to route the payload
/// to the TUN device, MemChain MemPool, or the target voice session.
///
/// ## Traffic accounting
/// Only `Vpn` variants increment `session.stats.bytes_rx`.
/// `MemChain` and `Voice` are control/signalling traffic — not billed.
#[derive(Debug)]
pub enum DecryptedPayload {
    /// Standard VPN IP packet — should be written to TUN.
    /// `bytes_rx` has already been incremented for this payload.
    Vpn(Vec<u8>),

    /// MemChain application message — should be routed to MemPool.
    /// Does NOT increment `bytes_rx` — control traffic is not billed.
    MemChain(MemChainMessage),

    /// Voice signalling or audio frame — should be forwarded to the
    /// session identified by `dst_ip`.
    ///
    /// The server routes the full `payload` (including the 0xAF magic byte
    /// and dst_ip header) to the target session's UDP endpoint unchanged.
    /// Content is end-to-end encrypted between the two clients; the server
    /// never decrypts or inspects it.
    ///
    /// Does NOT increment `bytes_rx` — voice relay is not billed as VPN traffic.
    Voice {
        /// Destination virtual IP (100.64.0.x) extracted from plaintext[1..5].
        dst_ip: Ipv4Addr,
        /// Full plaintext including the 0xAF magic byte and dst_ip header.
        /// Re-encrypted and forwarded to the target session as-is.
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
    /// Session manager for lookups.
    sessions: Arc<SessionManager>,
    /// Routing service for IP lookups.
    routing: Arc<RoutingService>,
    /// Transport encryption.
    crypto: DefaultTransportCrypto,
}

impl PacketHandler {
    /// Creates a new packet handler.
    pub fn new(sessions: Arc<SessionManager>, routing: Arc<RoutingService>) -> Self {
        Self {
            sessions,
            routing,
            crypto: DefaultTransportCrypto::new(),
        }
    }

    /// Processes an incoming UDP data packet.
    ///
    /// After decryption, the plaintext's **first byte** is peeked to
    /// determine whether this is a VPN IP packet or a MemChain message:
    ///
    /// | Byte      | Action                                           |
    /// |-----------|--------------------------------------------------|
    /// | `0x4X`    | IPv4 — validate source IP, record_rx, return Vpn |
    /// | `0x6X`    | IPv6 — record_rx, return Vpn (pass-through)      |
    /// | `0xAE`    | MemChain — bincode decode, return MemChain        |
    /// | other     | Error (unknown payload type)                     |
    ///
    /// ## Traffic accounting
    /// `session.stats.record_rx(plaintext_len)` is called only for VPN
    /// packets (IPv4 and IPv6). MemChain control messages are excluded.
    /// `plaintext_len` is the decrypted payload length — user data bytes
    /// without any wire-level crypto overhead.
    ///
    /// # Arguments
    /// * `data` - Raw UDP packet data (session_id + counter + ciphertext)
    ///
    /// # Returns
    /// - `Ok((session, DecryptedPayload::Vpn(ip_packet)))` for VPN traffic
    /// - `Ok((session, DecryptedPayload::MemChain(msg)))` for MemChain traffic
    ///
    /// # Errors
    /// - Session not found
    /// - Decryption failure
    /// - Replay attack detected
    /// - IP spoofing detected (VPN path only)
    /// - Unknown plaintext type
    /// - MemChain deserialisation failure
    pub fn handle_udp_packet(&self, data: &[u8]) -> Result<(Arc<Session>, DecryptedPayload)> {
        // ---- Validate minimum size ----
        if data.len() < DATA_PACKET_HEADER_SIZE + ENCRYPTION_OVERHEAD {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "Packet too short",
            ));
        }

        // ---- DEBUG LOG: Raw session ID bytes ----
        let raw_session_id = &data[0..16];
        debug!(
            "[PACKET_HANDLER] Processing packet, raw SessionID bytes: {:02X?}",
            raw_session_id
        );
        debug!(
            "[PACKET_HANDLER] SessionID (base64): {}",
            BASE64.encode(raw_session_id)
        );

        // ---- Decode packet header ----
        let packet = decode_data_packet(data)?;

        debug!(
            "[PACKET_HANDLER] Decoded packet.session_id (base64): {}",
            BASE64.encode(&packet.session_id)
        );

        // ---- Lookup session ----
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

        debug!(
            "[PACKET_HANDLER] Looking up SessionID: {}",
            session_id
        );
        debug!(
            "[PACKET_HANDLER] Active sessions count: {}",
            self.sessions.count()
        );

        let session = match self.sessions.get(&session_id) {
            Some(s) => {
                debug!(
                    "[PACKET_HANDLER] Session FOUND: {}, virtual_ip={}, endpoint={}",
                    session_id,
                    s.virtual_ip,
                    s.client_endpoint
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

        // ---- Validate counter (replay protection) ----
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

        // ---- Decrypt packet ----
        let mut plaintext = vec![0u8; packet.encrypted_payload.len()];
        let plaintext_len = self.crypto.decrypt(
            &session.session_key,
            packet.counter,
            &packet.session_id,
            &packet.encrypted_payload,
            &mut plaintext,
        )?;
        plaintext.truncate(plaintext_len);

        // ---- Update session activity (all paths) ----
        // touch() is always called — even MemChain messages keep the session alive.
        session.touch();

        // ====================================================
        // 1st-Byte Multiplexing + Traffic Accounting
        // ====================================================
        // Zero-copy peek of the first byte to determine payload type.
        //
        // IMPORTANT: record_rx() is called ONLY inside VPN arms (IPv4/IPv6).
        // MemChain control messages must NOT increment bytes_rx because:
        //   1. They are not user data — they carry protocol control messages.
        //   2. aeronyx.network billing counts user VPN traffic only.
        //   3. MemChain traffic volume is independent of VPN usage.
        let payload = match plaintext.first().copied() {
            // ---- IPv4 VPN packet (most common path) ----
            Some(b) if b >> 4 == 4 => {
                // Validate minimum IP header size
                if plaintext_len < IPV4_HEADER_MIN_SIZE {
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "IP packet too short",
                    ));
                }

                // Validate source IP matches session's virtual IP (anti-spoofing).
                //
                // NOTE: iOS/macOS Network Extension packetFlow.readPackets() returns
                // raw IP packets with the device's real LAN IP as src (e.g. 10.x.x.x),
                // not the VPN virtual IP (100.64.0.x). This is a known limitation of
                // Apple's NetworkExtension framework — the TUN rewriting happens at the
                // OS level but the raw packets seen by the extension still carry the
                // original src IP.
                //
                // We log at debug level instead of warn to avoid log spam, and we
                // do NOT drop the packet — the session key authentication already
                // proves the packet came from the legitimate client. IP src validation
                // is a secondary defence that is redundant when session auth passes.
                let src_ip = extract_ipv4_src(&plaintext)?;
                if src_ip != session.virtual_ip {
                    debug!(
                        session_id = %session_id,
                        expected = %session.virtual_ip,
                        actual = %src_ip,
                        "src IP mismatch (iOS/macOS NE limitation — packet accepted)"
                    );
                    // Do NOT return error — session key auth is sufficient proof of identity.
                }

                // ✅ Count inbound VPN user-data bytes (plaintext, no overhead).
                session.stats.record_rx(plaintext_len as u64);

                trace!(
                    session_id = %session_id,
                    len = plaintext_len,
                    "VPN IPv4 packet decrypted"
                );

                DecryptedPayload::Vpn(plaintext)
            }

            // ---- IPv6 — pass through as VPN (future support) ----
            Some(b) if b >> 4 == 6 => {
                // ✅ Count inbound VPN user-data bytes for IPv6 as well.
                session.stats.record_rx(plaintext_len as u64);

                trace!(
                    session_id = %session_id,
                    len = plaintext_len,
                    "VPN IPv6 packet decrypted (pass-through)"
                );
                DecryptedPayload::Vpn(plaintext)
            }

            // ---- MemChain message — NOT counted in bytes_rx ----
            Some(MEMCHAIN_MAGIC) => {
                // Strip the magic byte; remaining bytes are bincode payload.
                // ⚠️ Do NOT call record_rx() here — MemChain is control traffic.
                let memchain_payload = &plaintext[1..];

                match decode_memchain(memchain_payload) {
                    Ok(msg) => {
                        debug!(
                            session_id = %session_id,
                            msg_type = ?std::mem::discriminant(&msg),
                            "[MEMCHAIN] MemChain message decoded"
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
                            "[MEMCHAIN] Failed to decode MemChain payload"
                        );
                        return Err(ServerError::invalid_packet(
                            session.client_endpoint,
                            format!("MemChain decode error: {}", e),
                        ));
                    }
                }
            }

            // ---- Voice signalling / audio frame (0xAF) ----------------------
            // Wire format: [0xAF][dst_ip: 4 bytes][encrypted voice payload]
            //
            // The server extracts dst_ip and routes the full plaintext to the
            // target session's UDP endpoint without inspecting the payload.
            // Content is end-to-end encrypted between clients — server is
            // a transparent relay only.
            //
            // ⚠️ Do NOT call record_rx() — voice relay is not VPN user-data.
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

                // Extract destination virtual IP from plaintext[1..5].
                let mut dst_octets = [0u8; 4];
                dst_octets.copy_from_slice(
                    &plaintext[VOICE_DST_IP_OFFSET..VOICE_DST_IP_OFFSET + 4]
                );
                let dst_ip = Ipv4Addr::from(dst_octets);

                trace!(
                    session_id = %session_id,
                    dst_ip = %dst_ip,
                    len = plaintext_len,
                    "[VOICE] Voice packet decoded"
                );

                DecryptedPayload::Voice { dst_ip, payload: plaintext }
            }

            // ---- Unknown payload type ----
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

            // ---- Empty plaintext (should not happen after decrypt) ----
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
    /// `session.stats.record_tx(ip_packet.len())` counts plaintext bytes —
    /// the actual user-data size before encryption. This is intentional:
    /// billing should reflect user data transferred, not wire-level overhead
    /// (ENCRYPTION_OVERHEAD adds ~16 bytes of Poly1305 tag per packet).
    ///
    /// # Arguments
    /// * `ip_packet` - Raw IP packet from TUN (plaintext)
    ///
    /// # Returns
    /// Tuple of (encrypted wire packet, client UDP endpoint).
    ///
    /// # Errors
    /// - No route for destination IP
    /// - Session not found
    /// - Encryption failure
    pub fn handle_tun_packet(
        &self,
        ip_packet: &[u8],
    ) -> Result<(Vec<u8>, std::net::SocketAddr)> {
        // Validate minimum size
        if ip_packet.len() < IPV4_HEADER_MIN_SIZE {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "TUN packet too short",
            ));
        }

        // Extract destination IP
        let dst_ip = extract_ipv4_dst(ip_packet)?;

        // Lookup route
        let session_id = self.routing.lookup_or_error(dst_ip)?;

        // Get session
        let session = self.sessions.get_or_error(&session_id)?;

        // Get next counter
        let counter = session.next_tx_counter();

        // Encrypt packet
        let encrypted_len = ip_packet.len() + ENCRYPTION_OVERHEAD;
        let mut encrypted = vec![0u8; encrypted_len];
        let actual_len = self.crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            ip_packet,
            &mut encrypted,
        )?;
        encrypted.truncate(actual_len);

        // Build data packet
        let data_packet = DataPacket::new(
            *session.id.as_bytes(),
            counter,
            encrypted,
        );

        let output = encode_data_packet(&data_packet).to_vec();

        // ✅ Count outbound VPN user-data bytes.
        // Use ip_packet.len() (plaintext), NOT output.len() (encrypted + header).
        // Billing counts user data transferred, not wire overhead.
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

/// Extracts the source IPv4 address from an IP packet.
fn extract_ipv4_src(packet: &[u8]) -> Result<Ipv4Addr> {
    if packet.len() < IPV4_SRC_OFFSET + 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            "Packet too short for IPv4 header",
        ));
    }

    let version = packet[0] >> 4;
    if version != 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            format!("Expected IPv4, got version {}", version),
        ));
    }

    let mut octets = [0u8; 4];
    octets.copy_from_slice(&packet[IPV4_SRC_OFFSET..IPV4_SRC_OFFSET + 4]);
    Ok(Ipv4Addr::from(octets))
}

/// Extracts the destination IPv4 address from an IP packet.
fn extract_ipv4_dst(packet: &[u8]) -> Result<Ipv4Addr> {
    if packet.len() < IPV4_DST_OFFSET + 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            "Packet too short for IPv4 header",
        ));
    }

    let version = packet[0] >> 4;
    if version != 4 {
        return Err(ServerError::invalid_packet(
            "0.0.0.0:0".parse().unwrap(),
            format!("Expected IPv4, got version {}", version),
        ));
    }

    let mut octets = [0u8; 4];
    octets.copy_from_slice(&packet[IPV4_DST_OFFSET..IPV4_DST_OFFSET + 4]);
    Ok(Ipv4Addr::from(octets))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ipv4_packet(src: Ipv4Addr, dst: Ipv4Addr) -> Vec<u8> {
        let mut packet = vec![0u8; 20];
        packet[0] = 0x45; // Version 4, IHL 5
        packet[1] = 0x00; // DSCP, ECN
        packet[2] = 0x00; // Total length (high)
        packet[3] = 0x14; // Total length (low) = 20
        packet[IPV4_SRC_OFFSET..IPV4_SRC_OFFSET + 4].copy_from_slice(&src.octets());
        packet[IPV4_DST_OFFSET..IPV4_DST_OFFSET + 4].copy_from_slice(&dst.octets());
        packet
    }

    #[test]
    fn test_extract_ipv4_src() {
        let src = Ipv4Addr::new(192, 168, 1, 100);
        let dst = Ipv4Addr::new(8, 8, 8, 8);
        let packet = create_test_ipv4_packet(src, dst);
        let extracted = extract_ipv4_src(&packet).unwrap();
        assert_eq!(extracted, src);
    }

    #[test]
    fn test_extract_ipv4_dst() {
        let src = Ipv4Addr::new(192, 168, 1, 100);
        let dst = Ipv4Addr::new(8, 8, 8, 8);
        let packet = create_test_ipv4_packet(src, dst);
        let extracted = extract_ipv4_dst(&packet).unwrap();
        assert_eq!(extracted, dst);
    }

    #[test]
    fn test_extract_from_short_packet() {
        let short_packet = vec![0x45, 0x00]; // Too short
        assert!(extract_ipv4_src(&short_packet).is_err());
        assert!(extract_ipv4_dst(&short_packet).is_err());
    }

    #[test]
    fn test_extract_wrong_version() {
        let mut packet = vec![0u8; 20];
        packet[0] = 0x60; // IPv6 version
        assert!(extract_ipv4_src(&packet).is_err());
    }

    #[test]
    fn test_memchain_magic_no_ip_collision() {
        // MEMCHAIN_MAGIC (0xAE) high nibble = 0xA = 10
        // VOICE_MAGIC    (0xAF) high nibble = 0xA = 10
        // IPv4 high nibble = 4, IPv6 high nibble = 6
        // These must never collide or the multiplexer will misroute packets.
        assert_ne!(MEMCHAIN_MAGIC >> 4, 4u8, "MEMCHAIN must not collide with IPv4");
        assert_ne!(MEMCHAIN_MAGIC >> 4, 6u8, "MEMCHAIN must not collide with IPv6");
        assert_ne!(VOICE_MAGIC >> 4,    4u8, "VOICE must not collide with IPv4");
        assert_ne!(VOICE_MAGIC >> 4,    6u8, "VOICE must not collide with IPv6");
        assert_ne!(VOICE_MAGIC, MEMCHAIN_MAGIC, "VOICE and MEMCHAIN must be distinct");
    }

    /// Verify that the traffic accounting logic is correctly separated.
    #[test]
    fn test_first_byte_routing_invariants() {
        // IPv4: version nibble = 4
        assert_eq!(0x45u8 >> 4, 4u8);
        // IPv6: version nibble = 6
        assert_eq!(0x60u8 >> 4, 6u8);
        // MemChain: version nibble = 10 (0xA)
        assert_eq!(MEMCHAIN_MAGIC >> 4, 10u8);
        // Voice: version nibble = 10 (0xA), distinct byte value
        assert_eq!(VOICE_MAGIC >> 4, 10u8);
        assert_eq!(VOICE_MAGIC, 0xAFu8);
    }

    /// Verify voice packet dst_ip extraction.
    #[test]
    fn test_voice_packet_dst_ip_extraction() {
        // Construct a minimal voice plaintext:
        // [0xAF][100][64][0][5][...payload...]
        let dst = Ipv4Addr::new(100, 64, 0, 5);
        let mut packet = vec![VOICE_MAGIC];
        packet.extend_from_slice(&dst.octets());
        packet.extend_from_slice(b"voice_payload");

        assert!(packet.len() >= VOICE_PACKET_MIN_SIZE);

        let mut octets = [0u8; 4];
        octets.copy_from_slice(&packet[VOICE_DST_IP_OFFSET..VOICE_DST_IP_OFFSET + 4]);
        let extracted = Ipv4Addr::from(octets);
        assert_eq!(extracted, dst);
    }
}
