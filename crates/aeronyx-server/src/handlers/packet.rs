// ============================================
// File: crates/aeronyx-server/src/handlers/packet.rs
// ============================================
//! # Packet Handler
//!
//! ## Creation Reason
//! Handles data packet processing including encryption, decryption,
//! and forwarding between UDP and TUN interfaces.
//!
//! ## Main Functionality
//! - `PacketHandler`: Main packet processing logic
//! - UDP packet decryption and TUN forwarding
//! - TUN packet encryption and UDP forwarding
//! - Session and routing lookups
//!
//! ## Packet Processing
//!
//! ### Client → Internet (UDP → TUN)
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Receive UDP packet                                      │
//! │     ┌────────────┬─────────┬──────────────────────────┐    │
//! │     │ Session ID │ Counter │ Encrypted Payload        │    │
//! │     └────────────┴─────────┴──────────────────────────┘    │
//! │                                                             │
//! │  2. Lookup session by ID                                    │
//! │                                                             │
//! │  3. Validate counter (replay protection)                    │
//! │                                                             │
//! │  4. Decrypt payload with session key                        │
//! │     ┌──────────────────────────────────────────────────┐   │
//! │     │              IP Packet                            │   │
//! │     └──────────────────────────────────────────────────┘   │
//! │                                                             │
//! │  5. Validate source IP matches session                      │
//! │                                                             │
//! │  6. Write to TUN device                                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ### Internet → Client (TUN → UDP)
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Read IP packet from TUN                                 │
//! │     ┌──────────────────────────────────────────────────┐   │
//! │     │              IP Packet (dest: 100.64.0.x)        │   │
//! │     └──────────────────────────────────────────────────┘   │
//! │                                                             │
//! │  2. Extract destination IP                                  │
//! │                                                             │
//! │  3. Lookup route → session                                  │
//! │                                                             │
//! │  4. Encrypt with session key                                │
//! │                                                             │
//! │  5. Build packet with session ID and counter                │
//! │     ┌────────────┬─────────┬──────────────────────────┐    │
//! │     │ Session ID │ Counter │ Encrypted Payload        │    │
//! │     └────────────┴─────────┴──────────────────────────┘    │
//! │                                                             │
//! │  6. Send via UDP to client endpoint                         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Performance critical - minimize allocations
//! - Counter validation prevents replay attacks
//! - Source IP validation prevents IP spoofing
//! - Log security events but avoid log flooding
//!
//! ## Last Modified
//! v0.1.0 - Initial packet handler

use std::net::Ipv4Addr;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::{Bytes, BytesMut};
use tracing::{debug, trace, warn, info};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::{decode_data_packet, encode_data_packet};
use aeronyx_core::protocol::messages::DATA_PACKET_HEADER_SIZE;
use aeronyx_core::protocol::DataPacket;

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
    /// # Arguments
    /// * `data` - Raw UDP packet data
    ///
    /// # Returns
    /// Decrypted IP packet ready for TUN device.
    ///
    /// # Errors
    /// - Session not found
    /// - Decryption failure
    /// - Replay attack detected
    /// - IP spoofing detected
    pub fn handle_udp_packet(&self, data: &[u8]) -> Result<(Arc<Session>, Vec<u8>)> {
        // Validate minimum size
        if data.len() < DATA_PACKET_HEADER_SIZE + ENCRYPTION_OVERHEAD {
            return Err(ServerError::invalid_packet(
                "0.0.0.0:0".parse().unwrap(),
                "Packet too short",
            ));
        }

        // ========== DEBUG LOG: Raw session ID bytes ==========
        let raw_session_id = &data[0..16];
        debug!(
            "[PACKET_HANDLER] Processing packet, raw SessionID bytes: {:02X?}",
            raw_session_id
        );
        debug!(
            "[PACKET_HANDLER] SessionID (base64): {}",
            BASE64.encode(raw_session_id)
        );

        // Decode packet header
        let packet = decode_data_packet(data)?;

        // ========== DEBUG LOG: Decoded session ID ==========
        debug!(
            "[PACKET_HANDLER] Decoded packet.session_id (base64): {}",
            BASE64.encode(&packet.session_id)
        );

        // Lookup session
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

        // ========== DEBUG LOG: Looking up session ==========
        debug!(
            "[PACKET_HANDLER] Looking up SessionID: {}",
            session_id
        );

        // ========== DEBUG LOG: Session count ==========
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

        // Validate counter (replay protection)
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

        // Decrypt packet
        let mut plaintext = vec![0u8; packet.encrypted_payload.len()];
        let plaintext_len = self.crypto.decrypt(
            &session.session_key,
            packet.counter,
            &packet.session_id,
            &packet.encrypted_payload,
            &mut plaintext,
        )?;
        plaintext.truncate(plaintext_len);

        // Validate IP packet
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

        // Update session activity and stats
        session.touch();
        session.stats.record_rx(plaintext_len as u64);

        trace!(
            session_id = %session_id,
            len = plaintext_len,
            "UDP packet decrypted"
        );

        Ok((session, plaintext))
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
        // ... other fields ...
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
}
