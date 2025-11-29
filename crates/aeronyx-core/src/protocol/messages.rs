// ============================================
// File: crates/aeronyx-core/src/protocol/messages.rs
// ============================================
//! # Protocol Message Definitions
//!
//! ## Creation Reason
//! Defines the structure of all protocol messages exchanged between
//! AeroNyx clients and servers.
//!
//! ## Main Functionality
//! - `MessageType`: Enum for message type identification
//! - `ClientHello`: Client's initial handshake message
//! - `ServerHello`: Server's handshake response
//! - `DataPacket`: Encrypted tunnel data header
//!
//! ## Message Sizes
//! | Message | Size (bytes) |
//! |---------|--------------|
//! | ClientHello | 138 |
//! | ServerHello | 150 |
//! | DataPacket header | 24 |
//!
//! ## Wire Format (Little Endian)
//! All multi-byte integers are encoded in little-endian byte order.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Field order is critical - DO NOT reorder without version bump
//! - All sizes are fixed and validated on parse
//! - Add new message types at end of enum to maintain compatibility
//!
//! ## Last Modified
//! v0.1.0 - Initial message definitions

use serde::{Deserialize, Serialize};

// ============================================
// Message Type Constants
// ============================================

/// Size of ClientHello message in bytes.
pub const CLIENT_HELLO_SIZE: usize = 138;

/// Size of ServerHello message in bytes.
pub const SERVER_HELLO_SIZE: usize = 150;

/// Size of DataPacket header in bytes (excluding encrypted payload).
pub const DATA_PACKET_HEADER_SIZE: usize = 24;

// ============================================
// MessageType
// ============================================

/// Protocol message type identifier.
///
/// # Wire Format
/// Single byte at the start of every message identifying its type.
///
/// # Values
/// | Value | Type |
/// |-------|------|
/// | 0x01 | ClientHello |
/// | 0x02 | ServerHello |
/// | 0x03 | Data |
/// | 0x04 | Keepalive |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Client's initial handshake message.
    ClientHello = 0x01,
    /// Server's handshake response.
    ServerHello = 0x02,
    /// Encrypted data packet.
    Data = 0x03,
    /// Keep-alive ping (no payload).
    Keepalive = 0x04,
}

impl MessageType {
    /// Converts a byte to a MessageType.
    ///
    /// # Returns
    /// - `Some(MessageType)` if the byte is a valid message type
    /// - `None` if the byte is unknown
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(Self::ClientHello),
            0x02 => Some(Self::ServerHello),
            0x03 => Some(Self::Data),
            0x04 => Some(Self::Keepalive),
            _ => None,
        }
    }

    /// Converts the MessageType to its byte representation.
    #[must_use]
    pub const fn as_byte(&self) -> u8 {
        *self as u8
    }

    /// Checks if this is a handshake message.
    #[must_use]
    pub const fn is_handshake(&self) -> bool {
        matches!(self, Self::ClientHello | Self::ServerHello)
    }

    /// Checks if this is a data message.
    #[must_use]
    pub const fn is_data(&self) -> bool {
        matches!(self, Self::Data)
    }
}

impl TryFrom<u8> for MessageType {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::from_byte(value).ok_or(value)
    }
}

impl From<MessageType> for u8 {
    fn from(msg_type: MessageType) -> Self {
        msg_type.as_byte()
    }
}

// ============================================
// ClientHello
// ============================================

/// Client's initial handshake message.
///
/// # Purpose
/// Initiates a session by providing the client's identity and
/// ephemeral key for key exchange.
///
/// # Wire Format (138 bytes)
/// ```text
/// ┌────────────────────────────────────────────┐
/// │ message_type (1 byte)         │ 0x01       │
/// ├────────────────────────────────────────────┤
/// │ version (1 byte)              │ Protocol   │
/// ├────────────────────────────────────────────┤
/// │ client_public_key (32 bytes)  │ Ed25519    │
/// ├────────────────────────────────────────────┤
/// │ client_ephemeral_key (32 bytes)│ X25519    │
/// ├────────────────────────────────────────────┤
/// │ timestamp (8 bytes)           │ Unix secs  │
/// ├────────────────────────────────────────────┤
/// │ signature (64 bytes)          │ Ed25519    │
/// └────────────────────────────────────────────┘
/// ```
///
/// # Signature Covers
/// All fields except the signature itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientHello {
    /// Message type (always 0x01).
    pub message_type: u8,
    /// Protocol version.
    pub version: u8,
    /// Client's Ed25519 public key (identity).
    pub client_public_key: [u8; 32],
    /// Client's X25519 ephemeral public key (for key exchange).
    pub client_ephemeral_key: [u8; 32],
    /// Unix timestamp in seconds (for replay protection).
    pub timestamp: i64,
    /// Ed25519 signature over the above fields.
    pub signature: [u8; 64],
}

impl ClientHello {
    /// Creates a new ClientHello with the given parameters.
    ///
    /// Note: The signature field is set to zeros and must be
    /// filled in using `sign()` or the handshake crypto module.
    #[must_use]
    pub fn new(
        version: u8,
        client_public_key: [u8; 32],
        client_ephemeral_key: [u8; 32],
        timestamp: i64,
    ) -> Self {
        Self {
            message_type: MessageType::ClientHello.as_byte(),
            version,
            client_public_key,
            client_ephemeral_key,
            timestamp,
            signature: [0u8; 64],
        }
    }

    /// Returns the expected size of a serialized ClientHello.
    #[must_use]
    pub const fn wire_size() -> usize {
        CLIENT_HELLO_SIZE
    }
}

// ============================================
// ServerHello
// ============================================

/// Server's handshake response message.
///
/// # Purpose
/// Completes the handshake by providing the server's ephemeral key
/// and session parameters.
///
/// # Wire Format (150 bytes)
/// ```text
/// ┌────────────────────────────────────────────┐
/// │ message_type (1 byte)         │ 0x02       │
/// ├────────────────────────────────────────────┤
/// │ version (1 byte)              │ Protocol   │
/// ├────────────────────────────────────────────┤
/// │ server_public_key (32 bytes)  │ Ed25519    │
/// ├────────────────────────────────────────────┤
/// │ server_ephemeral_key (32 bytes)│ X25519    │
/// ├────────────────────────────────────────────┤
/// │ assigned_ip (4 bytes)         │ IPv4       │
/// ├────────────────────────────────────────────┤
/// │ session_id (16 bytes)         │ Random     │
/// ├────────────────────────────────────────────┤
/// │ signature (64 bytes)          │ Ed25519    │
/// └────────────────────────────────────────────┘
/// ```
///
/// # Signature Covers
/// All fields except signature, plus the client's public key
/// (for binding to the specific client).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    /// Message type (always 0x02).
    pub message_type: u8,
    /// Protocol version.
    pub version: u8,
    /// Server's Ed25519 public key (identity).
    pub server_public_key: [u8; 32],
    /// Server's X25519 ephemeral public key (for key exchange).
    pub server_ephemeral_key: [u8; 32],
    /// Virtual IP address assigned to the client.
    pub assigned_ip: [u8; 4],
    /// Unique session identifier.
    pub session_id: [u8; 16],
    /// Ed25519 signature over the above fields + client_public_key.
    pub signature: [u8; 64],
}

impl ServerHello {
    /// Creates a new ServerHello with the given parameters.
    ///
    /// Note: The signature field is set to zeros and must be
    /// filled in using the handshake crypto module.
    #[must_use]
    pub fn new(
        version: u8,
        server_public_key: [u8; 32],
        server_ephemeral_key: [u8; 32],
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Self {
        Self {
            message_type: MessageType::ServerHello.as_byte(),
            version,
            server_public_key,
            server_ephemeral_key,
            assigned_ip,
            session_id,
            signature: [0u8; 64],
        }
    }

    /// Returns the expected size of a serialized ServerHello.
    #[must_use]
    pub const fn wire_size() -> usize {
        SERVER_HELLO_SIZE
    }
}

// ============================================
// DataPacket
// ============================================

/// Encrypted data packet header.
///
/// # Purpose
/// Contains metadata for routing and decryption of encrypted
/// tunnel data.
///
/// # Wire Format (24 bytes header + variable payload)
/// ```text
/// ┌────────────────────────────────────────────┐
/// │ session_id (16 bytes)         │ Routing    │
/// ├────────────────────────────────────────────┤
/// │ counter (8 bytes)             │ Nonce/seq  │
/// ├────────────────────────────────────────────┤
/// │ encrypted_payload (variable)  │ AEAD       │
/// │ └─ includes 16-byte auth tag  │            │
/// └────────────────────────────────────────────┘
/// ```
///
/// # Security
/// - `session_id`: Used as Additional Authenticated Data (AAD)
/// - `counter`: Combined with zeros to form the nonce; must be unique per packet
/// - Payload is encrypted with ChaCha20-Poly1305
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPacket {
    /// Session identifier for routing.
    pub session_id: [u8; 16],
    /// Packet counter (used for nonce construction).
    pub counter: u64,
    /// Encrypted payload (IP packet + auth tag).
    pub encrypted_payload: Vec<u8>,
}

impl DataPacket {
    /// Creates a new DataPacket.
    #[must_use]
    pub fn new(session_id: [u8; 16], counter: u64, encrypted_payload: Vec<u8>) -> Self {
        Self {
            session_id,
            counter,
            encrypted_payload,
        }
    }

    /// Returns the header size (excluding encrypted payload).
    #[must_use]
    pub const fn header_size() -> usize {
        DATA_PACKET_HEADER_SIZE
    }

    /// Returns the total wire size of this packet.
    #[must_use]
    pub fn wire_size(&self) -> usize {
        DATA_PACKET_HEADER_SIZE + self.encrypted_payload.len()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_roundtrip() {
        for msg_type in [
            MessageType::ClientHello,
            MessageType::ServerHello,
            MessageType::Data,
            MessageType::Keepalive,
        ] {
            let byte = msg_type.as_byte();
            let restored = MessageType::from_byte(byte).unwrap();
            assert_eq!(msg_type, restored);
        }
    }

    #[test]
    fn test_message_type_unknown() {
        assert!(MessageType::from_byte(0x00).is_none());
        assert!(MessageType::from_byte(0xFF).is_none());
    }

    #[test]
    fn test_message_type_classification() {
        assert!(MessageType::ClientHello.is_handshake());
        assert!(MessageType::ServerHello.is_handshake());
        assert!(!MessageType::Data.is_handshake());
        
        assert!(MessageType::Data.is_data());
        assert!(!MessageType::ClientHello.is_data());
    }

    #[test]
    fn test_client_hello_size() {
        let hello = ClientHello::new(
            1,
            [0u8; 32],
            [0u8; 32],
            12345,
        );
        
        // 1 + 1 + 32 + 32 + 8 + 64 = 138
        assert_eq!(ClientHello::wire_size(), CLIENT_HELLO_SIZE);
        assert_eq!(hello.message_type, MessageType::ClientHello.as_byte());
    }

    #[test]
    fn test_server_hello_size() {
        let hello = ServerHello::new(
            1,
            [0u8; 32],
            [0u8; 32],
            [100, 64, 0, 2],
            [0u8; 16],
        );
        
        // 1 + 1 + 32 + 32 + 4 + 16 + 64 = 150
        assert_eq!(ServerHello::wire_size(), SERVER_HELLO_SIZE);
        assert_eq!(hello.message_type, MessageType::ServerHello.as_byte());
    }

    #[test]
    fn test_data_packet_size() {
        let packet = DataPacket::new(
            [0u8; 16],
            1,
            vec![0u8; 100],
        );
        
        // 16 + 8 = 24 header + 100 payload
        assert_eq!(DataPacket::header_size(), DATA_PACKET_HEADER_SIZE);
        assert_eq!(packet.wire_size(), 124);
    }
}
