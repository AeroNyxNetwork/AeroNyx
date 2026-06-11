// ============================================
// File: crates/aeronyx-server/src/handlers/packet.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Injected Arc<TrafficTracker> into PacketHandler.
//   VPN hot path (IPv4/IPv6) now dual-writes bytes to both
//   SessionStats (per-session cumulative) and TrafficTracker
//   (per-wallet delta, drained each heartbeat).
//   Uses session.wallet_hex cache — no allocation per packet.
//
// Main Functionality:
//   - PacketHandler: handles UDP and TUN packet processing
//   - handle_udp_packet(): decrypt, replay-check, 1st-byte multiplex
//   - handle_tun_packet(): encrypt outbound IP packets for VPN clients
//
// Dependencies:
//   - services/session.rs: Session, SessionManager
//   - services/routing.rs: RoutingService
//   - services/traffic_tracker.rs: TrafficTracker (v1.0.0-Membership)
//   - aeronyx-core: crypto, protocol
//
// ⚠️ Important Notes for Next Developer:
//   - Only VPN arms (0x4X / 0x6X) write to TrafficTracker.
//     MemChain / Voice signals / Voice audio do NOT — not billed.
//   - wallet_hex is read from session.wallet_hex (cached at Session::new).
//     Never call hex::encode in this hot path.
//   - MEMCHAIN_MAGIC (0xAE) and VOICE_MAGIC (0xAF) must stay in sync
//     with aeronyx_core and client-side magic bytes.
//   - Source IP validation is advisory only (iOS/macOS NE compatibility).
//
// Last Modified:
//   v1.0.0-TrafficFix     — record_rx() moved inside VPN arms only
//   v1.0.0-VoiceSignal    — disc=31/32/33 routed as voice signals
//   v1.0.0-Membership     — TrafficTracker dual-write on VPN path
// ============================================

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
use crate::services::{NodePolicyRuntime, RoutingService, Session, SessionManager};
use crate::services::traffic_tracker::TrafficTracker;

// ============================================
// Constants
// ============================================

const IPV4_HEADER_MIN_SIZE: usize = 20;
const IPV4_DST_OFFSET:      usize = 16;
const IPV4_SRC_OFFSET:      usize = 12;
const IPV4_PROTOCOL_OFFSET: usize = 9;
const ICMP_PROTOCOL:        u8 = 1;
const ICMP_ECHO_REPLY:      u8 = 0;
const ICMP_ECHO_REQUEST:    u8 = 8;
const ICMP_HEADER_SIZE:     usize = 8;

/// Magic byte identifying a Voice audio frame packet (0xAF).
pub const VOICE_MAGIC: u8 = 0xAF;

const VOICE_PACKET_MIN_SIZE: usize = 5;
const VOICE_DST_IP_OFFSET:   usize = 1;

pub const DISC_VOICE_OFFER:  u32 = 31;
pub const DISC_VOICE_ANSWER: u32 = 32;
pub const DISC_VOICE_END:    u32 = 33;

// ============================================
// DecryptedPayload
// ============================================

/// Result of decrypting an incoming DataPacket.
///
/// Multiplex dispatch table:
/// | plaintext[0] | discriminant | Variant      |
/// |--------------|--------------|--------------|
/// | 0x4X (IPv4)  | —            | Vpn          |
/// | 0x6X (IPv6)  | —            | Vpn          |
/// | 0xAE         | 0-17         | MemChain     |
/// | 0xAE         | 31-33        | VoiceSignal  |
/// | 0xAF         | —            | Voice        |
///
/// Only Vpn increments bytes_rx / TrafficTracker.
#[derive(Debug)]
pub enum DecryptedPayload {
    Vpn(Vec<u8>),
    KeepaliveAck { rtt_ms: f64 },
    MemChain(MemChainMessage),
    VoiceSignal {
        discriminant:  u32,
        target_wallet: Option<String>,
        payload:       Vec<u8>,
    },
    Voice {
        dst_ip:  Ipv4Addr,
        payload: Vec<u8>,
    },
}

// ============================================
// PacketHandler
// ============================================

pub struct PacketHandler {
    sessions: Arc<SessionManager>,
    routing:  Arc<RoutingService>,
    crypto:   DefaultTransportCrypto,
    /// Per-wallet traffic delta tracker for membership quota enforcement.
    /// Drained by HeartbeatReporter every ~30s and sent to CMS.
    traffic:  Arc<TrafficTracker>,
    /// Operator policy from nodeboard Settings.
    policy:   Arc<NodePolicyRuntime>,
}

impl PacketHandler {
    pub fn new(
        sessions: Arc<SessionManager>,
        routing:  Arc<RoutingService>,
        traffic:  Arc<TrafficTracker>,
        policy:   Arc<NodePolicyRuntime>,
    ) -> Self {
        Self {
            sessions,
            routing,
            crypto: DefaultTransportCrypto::new(),
            traffic,
            policy,
        }
    }

    /// Processes an incoming UDP data packet.
    ///
    /// Returns (session, DecryptedPayload) on success.
    /// Returns Err(SessionNotFound) when session ID is unknown —
    /// caller should send 0xFF RESET to client.
    pub fn handle_udp_packet(&self, data: &[u8]) -> Result<(Arc<Session>, DecryptedPayload)> {
        if data.len() < DATA_PACKET_HEADER_SIZE + ENCRYPTION_OVERHEAD {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "Packet too short",
            ));
        }

        let raw_session_id = &data[0..16];
        debug!(
            "[PACKET_HANDLER] Processing packet, raw SessionID bytes: {:02X?}",
            raw_session_id
        );
        debug!(
            "[PACKET_HANDLER] SessionID (base64): {}",
            BASE64.encode(raw_session_id)
        );

        let packet = decode_data_packet(data)?;

        debug!(
            "[PACKET_HANDLER] Decoded packet.session_id (base64): {}",
            BASE64.encode(&packet.session_id)
        );

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

        if !session.validate_rx_counter(packet.counter) {
            warn!(
                session_id = %session_id,
                counter    = packet.counter,
                "Replay attack detected"
            );
            return Err(ServerError::Core(aeronyx_core::error::CoreError::replay(
                packet.counter,
                session.rx_counter.load(std::sync::atomic::Ordering::SeqCst),
            )));
        }

        let mut plaintext = vec![0u8; packet.encrypted_payload.len()];
        let plaintext_len = self.crypto.decrypt(
            &session.session_key,
            packet.counter,
            &packet.session_id,
            &packet.encrypted_payload,
            &mut plaintext,
        )?;
        plaintext.truncate(plaintext_len);

        session.touch();

        let payload = match plaintext.first().copied() {

            // ── IPv4 VPN ──────────────────────────────────────────────
            Some(b) if b >> 4 == 4 => {
                if plaintext_len < IPV4_HEADER_MIN_SIZE {
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "IP packet too short",
                    ));
                }

                let src_ip = extract_ipv4_src(&plaintext)?;
                if src_ip != session.virtual_ip {
                    debug!(
                        session_id = %session_id,
                        expected   = %session.virtual_ip,
                        actual     = %src_ip,
                        "src IP mismatch (iOS/macOS NE — packet accepted)"
                    );
                }

                if let Some(rtt_ms) = record_keepalive_echo_reply(&session, &plaintext) {
                    trace!(
                        session_id = %session_id,
                        rtt_ms,
                        "[KEEPALIVE] ICMP echo reply received"
                    );
                    return Ok((session, DecryptedPayload::KeepaliveAck { rtt_ms }));
                }

                // Session-level cumulative counter (billing audit).
                if !self.policy.allow_traffic_bytes(plaintext_len) {
                    return Err(ServerError::node_policy_rejected("bandwidth_limit_mbps"));
                }
                session.stats.record_rx(plaintext_len as u64);
                // Wallet-level delta counter (heartbeat quota enforcement).
                // Uses cached wallet_hex — no allocation on hot path.
                self.traffic.record_rx(&session.wallet_hex, plaintext_len as u64);

                trace!(session_id = %session_id, len = plaintext_len, "VPN IPv4 decrypted");
                DecryptedPayload::Vpn(plaintext)
            }

            // ── IPv6 VPN ──────────────────────────────────────────────
            Some(b) if b >> 4 == 6 => {
                // Session-level cumulative counter.
                if !self.policy.allow_traffic_bytes(plaintext_len) {
                    return Err(ServerError::node_policy_rejected("bandwidth_limit_mbps"));
                }
                session.stats.record_rx(plaintext_len as u64);
                // Wallet-level delta counter.
                self.traffic.record_rx(&session.wallet_hex, plaintext_len as u64);

                trace!(session_id = %session_id, len = plaintext_len, "VPN IPv6 decrypted");
                DecryptedPayload::Vpn(plaintext)
            }

            // ── 0xAE: MemChain or Voice Signal ────────────────────────
            Some(MEMCHAIN_MAGIC) => {
                if plaintext_len < 5 {
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "0xAE packet too short for discriminant",
                    ));
                }

                let disc = u32::from_le_bytes([
                    plaintext[1], plaintext[2], plaintext[3], plaintext[4],
                ]);

                if matches!(disc, DISC_VOICE_OFFER | DISC_VOICE_ANSWER | DISC_VOICE_END) {
                    let target_wallet = extract_pubkey_from_json(&plaintext[5..]);
                    trace!(
                        session_id = %session_id,
                        disc,
                        target = %target_wallet.as_deref().unwrap_or("unknown"),
                        "[VOICE_SIGNAL] Signal received"
                    );
                    return Ok((session, DecryptedPayload::VoiceSignal {
                        discriminant:  disc,
                        target_wallet,
                        payload:       plaintext,
                    }));
                }

                let memchain_payload = &plaintext[1..];
                match decode_memchain(memchain_payload) {
                    Ok(msg) => {
                        debug!(
                            session_id = %session_id,
                            msg_type   = ?std::mem::discriminant(&msg),
                            "[MEMCHAIN] Message decoded"
                        );
                        DecryptedPayload::MemChain(msg)
                    }
                    Err(e) => {
                        let dump_len = plaintext.len().min(16);
                        warn!(
                            session_id            = %session_id,
                            error                 = %e,
                            plaintext_total_len   = plaintext.len(),
                            header_hex            = %hex::encode(&plaintext[..dump_len]),
                            discriminant_bytes    = %hex::encode(&plaintext[1..plaintext.len().min(5)]),
                            "[MEMCHAIN] Failed to decode payload"
                        );
                        return Err(ServerError::invalid_packet(
                            session.client_endpoint,
                            format!("MemChain decode error: {}", e),
                        ));
                    }
                }
            }

            // ── 0xAF: Voice audio frame ───────────────────────────────
            Some(VOICE_MAGIC) => {
                if plaintext_len < VOICE_PACKET_MIN_SIZE {
                    warn!(
                        session_id = %session_id,
                        len        = plaintext_len,
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
                    dst_ip     = %dst_ip,
                    len        = plaintext_len,
                    "[VOICE] Audio frame decoded"
                );

                DecryptedPayload::Voice { dst_ip, payload: plaintext }
            }

            // ── Unknown ───────────────────────────────────────────────
            Some(b) => {
                warn!(
                    session_id  = %session_id,
                    first_byte  = format_args!("0x{:02X}", b),
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

        let dst_ip     = extract_ipv4_dst(ip_packet)?;
        let session_id = self.routing.lookup_or_error(dst_ip)?;
        let session    = self.sessions.get_or_error(&session_id)?;
        if !self.policy.allow_traffic_bytes(ip_packet.len()) {
            return Err(ServerError::node_policy_rejected("bandwidth_limit_mbps"));
        }
        let counter    = session.next_tx_counter();

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
        let output      = encode_data_packet(&data_packet).to_vec();

        session.touch();

        // Session-level cumulative counter (billing audit).
        session.stats.record_tx(ip_packet.len() as u64);
        // Wallet-level delta counter (heartbeat quota enforcement).
        self.traffic.record_tx(&session.wallet_hex, ip_packet.len() as u64);

        trace!(
            session_id    = %session_id,
            dst_ip        = %dst_ip,
            plaintext_len = ip_packet.len(),
            wire_len      = output.len(),
            "TUN packet encrypted and sent"
        );

        Ok((output, session.client_endpoint))
    }

    /// Builds an encrypted in-tunnel ICMP Echo Request for RTT measurement.
    ///
    /// Source path:
    ///   /root/a/AeroNyx/crates/aeronyx-server/src/handlers/packet.rs
    ///
    /// This is an operational keepalive probe, not user traffic. It is sent only
    /// to the session's assigned virtual IP and therefore does not reveal user
    /// destinations, DNS contents, packet payloads, or browsing history.
    pub fn build_keepalive_probe(
        &self,
        session: &Arc<Session>,
        gateway_ip: Ipv4Addr,
    ) -> Result<(Vec<u8>, std::net::SocketAddr)> {
        let (identifier, sequence) = session.begin_keepalive_probe();
        let ip_packet = build_icmp_echo_request(
            gateway_ip,
            session.virtual_ip,
            identifier,
            sequence,
        );
        let counter = session.next_tx_counter();

        let mut encrypted = vec![0u8; ip_packet.len() + ENCRYPTION_OVERHEAD];
        let actual_len = self.crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            &ip_packet,
            &mut encrypted,
        )?;
        encrypted.truncate(actual_len);

        let data_packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let output = encode_data_packet(&data_packet).to_vec();
        session.stats.record_control_tx();

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

fn record_keepalive_echo_reply(session: &Arc<Session>, packet: &[u8]) -> Option<f64> {
    if packet.len() < IPV4_HEADER_MIN_SIZE {
        return None;
    }
    if packet[0] >> 4 != 4 || packet[IPV4_PROTOCOL_OFFSET] != ICMP_PROTOCOL {
        return None;
    }

    let ihl = ((packet[0] & 0x0f) as usize) * 4;
    if ihl < IPV4_HEADER_MIN_SIZE || packet.len() < ihl + ICMP_HEADER_SIZE {
        return None;
    }

    let icmp = &packet[ihl..];
    if icmp[0] != ICMP_ECHO_REPLY || icmp[1] != 0 {
        return None;
    }

    let identifier = u16::from_be_bytes([icmp[4], icmp[5]]);
    let sequence = u16::from_be_bytes([icmp[6], icmp[7]]);
    session.complete_keepalive_probe(identifier, sequence)
}

fn build_icmp_echo_request(
    src: Ipv4Addr,
    dst: Ipv4Addr,
    identifier: u16,
    sequence: u16,
) -> Vec<u8> {
    let payload = b"AERONYX-KEEPALIVE";
    let total_len = IPV4_HEADER_MIN_SIZE + ICMP_HEADER_SIZE + payload.len();
    let mut packet = vec![0u8; total_len];

    packet[0] = 0x45;
    packet[1] = 0;
    packet[2..4].copy_from_slice(&(total_len as u16).to_be_bytes());
    packet[4..6].copy_from_slice(&sequence.to_be_bytes());
    packet[6..8].copy_from_slice(&0u16.to_be_bytes());
    packet[8] = 64;
    packet[9] = ICMP_PROTOCOL;
    packet[12..16].copy_from_slice(&src.octets());
    packet[16..20].copy_from_slice(&dst.octets());
    let ip_sum = internet_checksum(&packet[..IPV4_HEADER_MIN_SIZE]);
    packet[10..12].copy_from_slice(&ip_sum.to_be_bytes());

    let icmp_start = IPV4_HEADER_MIN_SIZE;
    packet[icmp_start] = ICMP_ECHO_REQUEST;
    packet[icmp_start + 1] = 0;
    packet[icmp_start + 4..icmp_start + 6].copy_from_slice(&identifier.to_be_bytes());
    packet[icmp_start + 6..icmp_start + 8].copy_from_slice(&sequence.to_be_bytes());
    packet[icmp_start + ICMP_HEADER_SIZE..].copy_from_slice(payload);
    let icmp_sum = internet_checksum(&packet[icmp_start..]);
    packet[icmp_start + 2..icmp_start + 4].copy_from_slice(&icmp_sum.to_be_bytes());

    packet
}

fn internet_checksum(bytes: &[u8]) -> u16 {
    let mut sum = 0u32;
    let mut chunks = bytes.chunks_exact(2);
    for chunk in &mut chunks {
        sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
    }
    if let Some(&last) = chunks.remainder().first() {
        sum += (last as u32) << 8;
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
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
/// Reads "to" field first; falls back to "pubkey" for backward compatibility.
fn extract_pubkey_from_json(json_bytes: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(json_bytes).ok()?;
    extract_json_hex_field(s, "\"to\"")
        .or_else(|| extract_json_hex_field(s, "\"pubkey\""))
}

fn extract_json_hex_field<'a>(s: &'a str, key: &str) -> Option<String> {
    let key_pos     = s.find(key)?;
    let after_key   = &s[key_pos + key.len()..];
    let colon_pos   = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') { return None; }
    let value_start = &after_colon[1..];
    let end         = value_start.find('"')?;
    let hex         = &value_start[..end];
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
        p[0] = 0x60;
        assert!(extract_ipv4_src(&p).is_err());
    }

    #[test]
    fn test_magic_bytes_no_collision() {
        assert_ne!(MEMCHAIN_MAGIC >> 4, 4u8);
        assert_ne!(MEMCHAIN_MAGIC >> 4, 6u8);
        assert_ne!(VOICE_MAGIC >> 4,    4u8);
        assert_ne!(VOICE_MAGIC >> 4,    6u8);
        assert_ne!(VOICE_MAGIC, MEMCHAIN_MAGIC);
    }

    #[test]
    fn test_first_byte_routing_invariants() {
        assert_eq!(0x45u8 >> 4, 4u8);
        assert_eq!(0x60u8 >> 4, 6u8);
        assert_eq!(MEMCHAIN_MAGIC >> 4, 10u8);
        assert_eq!(VOICE_MAGIC, 0xAFu8);
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
        assert!(DISC_VOICE_OFFER  >= 18);
        assert!(DISC_VOICE_ANSWER >= 18);
        assert!(DISC_VOICE_END    >= 18);
    }

    #[test]
    fn test_extract_pubkey_from_json() {
        let sender = "437059fc0f5403365c69ef9ba6df93cdb0dc1f0058acee6af647f0b4152a7010";
        let target = "99e2b4602033e7b8c744232a1a74a3ec8b6c82c9dfedf66406f09b779071f753";

        let json_new = format!(
            r#"{{"type":"offer","call_id":"abc","pubkey":"{}","to":"{}"}}"#,
            sender, target
        );
        assert_eq!(
            extract_pubkey_from_json(json_new.as_bytes()).as_deref(),
            Some(target)
        );

        let json_old = format!(
            r#"{{"type":"offer","call_id":"abc","pubkey":"{}"}}"#,
            target
        );
        assert_eq!(
            extract_pubkey_from_json(json_old.as_bytes()).as_deref(),
            Some(target)
        );

        assert!(extract_pubkey_from_json(br#"{"type":"offer"}"#).is_none());
    }
}
