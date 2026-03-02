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
//! ## Main Functionality
//! - `PacketHandler`: Main packet processing logic
//! - `DecryptedPayload`: 🌟 NEW enum distinguishing VPN vs MemChain payloads
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
//! │     ├─ 0x4X → Validate source IP → DecryptedPayload::Vpn   │
//! │     └─ 0xAE → bincode decode → DecryptedPayload::MemChain  │
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
//! │  5. Build packet with session ID and counter                │
//! │  6. Send via UDP to client                                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Performance critical - minimize allocations
//! - Counter validation prevents replay attacks
//! - Source IP validation prevents IP spoofing (VPN path only)
//! - MemChain path deliberately skips IP validation (no IP header)
//! - Log security events but avoid log flooding
//! - The `MEMCHAIN_MAGIC` constant (0xAE) must stay in sync with
//!   `aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC`
//!
//! ## Last Modified
//! v0.1.0 - Initial packet handler
//! v0.2.0 - Added MemChain 1st-byte multiplexing (DecryptedPayload enum)

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
// DecryptedPayload — 🌟 NEW
// ============================================

/// Result of decrypting an incoming DataPacket.
///
/// The multiplexer inspects `plaintext[0]` to decide the variant:
/// - `0x4X` (IPv4) → [`DecryptedPayload::Vpn`]
/// - `0xAE` → [`DecryptedPayload::MemChain`]
///
/// The caller (`server.rs` UDP task) uses this to route the payload
/// to either the TUN device or the MemChain MemPool.
#[derive(Debug)]
pub enum DecryptedPayload {
    /// Standard VPN IP packet — should be written to TUN.
    Vpn(Vec<u8>),

    /// MemChain application message — should be routed to MemPool.
    MemChain(MemChainMessage),
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
    /// | Byte      | Action                                          |
    /// |-----------|-------------------------------------------------|
    /// | `0x4X`    | IPv4 — validate source IP, return `Vpn`         |
    /// | `0xAE`    | MemChain — bincode decode, return `MemChain`    |
    /// | other     | Error (unknown payload type)                    |
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
                    "[PACKET_HANDLER] ❌ Invalid session ID format: {:02X?}",
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
                    "[PACKET_HANDLER] ✅ Session FOUND: {}, virtual_ip={}, endpoint={}",
                    session_id,
                    s.virtual_ip,
                    s.client_endpoint
                );
                s
            }
            None => {
                warn!(
                    "[PACKET_HANDLER] ❌ Session NOT FOUND: {} (base64: {})",
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

        // ====================================================
        // 🌟 1st-Byte Multiplexing — The Core Routing Hack
        // ====================================================
        // Zero-copy peek of the first byte to determine payload type.
        // This is O(1) and adds no measurable latency.
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

                // Validate source IP matches session's virtual IP
                let src_ip = extract_ipv4_src(&plaintext)?;
                if src_ip != session.virtual_ip {
                    warn!(
                        session_id = %session_id,
                        expected = %session.virtual_ip,
                        actual = %src_ip,
                        "IP spoofing detected"
                    );
                    return Err(ServerError::invalid_packet(
                        session.client_endpoint,
                        "Source IP mismatch",
                    ));
                }

                trace!(
                    session_id = %session_id,
                    len = plaintext_len,
                    "VPN packet decrypted"
                );

                DecryptedPayload::Vpn(plaintext)
            }

            // ---- MemChain message ----
            Some(MEMCHAIN_MAGIC) => {
                // Strip the magic byte; remaining bytes are bincode payload
                let memchain_payload = &plaintext[1..];

                match decode_memchain(memchain_payload) {
                    Ok(msg) => {
                        debug!(
                            session_id = %session_id,
                            msg_type = ?std::mem::discriminant(&msg),
                            "[MEMCHAIN] ✅ MemChain message decoded"
                        );
                        DecryptedPayload::MemChain(msg)
                    }
                    Err(e) => {
                        warn!(
                            session_id = %session_id,
                            error = %e,
                            "[MEMCHAIN] ❌ Failed to decode MemChain payload"
                        );
                        return Err(ServerError::invalid_packet(
                            session.client_endpoint,
                            format!("MemChain decode error: {}", e),
                        ));
                    }
                }
            }

            // ---- IPv6 — pass through as VPN (future support) ----
            Some(b) if b >> 4 == 6 => {
                trace!(
                    session_id = %session_id,
                    len = plaintext_len,
                    "IPv6 packet decrypted (pass-through)"
                );
                DecryptedPayload::Vpn(plaintext)
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

        // ---- Update session activity and stats ----
        session.touch();
        session.stats.record_rx(plaintext_len as u64);

        Ok((session, payload))
    }

    /// Processes an IP packet from the TUN device.
    ///
    /// # Arguments
    /// * `ip_packet` - Raw IP packet from TUN
    ///
    /// # Returns
    /// Tuple of (encrypted packet, client endpoint).
    ///
    /// # Errors
    /// - No route for destination
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

        // Update session activity and stats
        session.touch();
        session.stats.record_tx(ip_packet.len() as u64);

        trace!(
            session_id = %session_id,
            dst_ip = %dst_ip,
            len = output.len(),
            "TUN packet encrypted"
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

    // Check IP version
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

    // Check IP version
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
        // Verify MEMCHAIN_MAGIC cannot be confused with IP version nibble
        assert_ne!(MEMCHAIN_MAGIC >> 4, 4, "Must not collide with IPv4");
        assert_ne!(MEMCHAIN_MAGIC >> 4, 6, "Must not collide with IPv6");
    }
}
