// ============================================
// File: crates/aeronyx-core/src/protocol/codec.rs
// ============================================
//! # Protocol Codec
//!
//! ## Creation Reason
//! Provides binary serialization and deserialization for protocol
//! messages, enabling efficient wire-format encoding.
//!
//! ## Main Functionality
//! - `Codec` trait: Generic encode/decode interface
//! - `ProtocolCodec`: Implementation for all message types
//! - Zero-copy parsing where possible
//!
//! ## Wire Format
//! All messages use little-endian byte order for multi-byte integers.
//!
//! ## Parsing Strategy
//! 1. Check minimum message length
//! 2. Read message type byte
//! 3. Dispatch to type-specific parser
//! 4. Validate all fields
//!
//! ## ⚠️ Important Note for Next Developer
//! - Always validate buffer lengths before reading
//! - Use checked arithmetic to prevent overflows
//! - Keep parsing zero-allocation where possible
//!
//! ## Last Modified
//! v0.1.0 - Initial codec implementation

use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::error::{CoreError, Result};
use crate::protocol::messages::{
    ClientHello, DataPacket, MessageType, ServerHello,
    CLIENT_HELLO_SIZE, DATA_PACKET_HEADER_SIZE, SERVER_HELLO_SIZE,
};

// ============================================
// Codec Trait
// ============================================

/// Trait for encoding and decoding protocol messages.
///
/// # Type Parameters
/// * `T` - The message type to encode/decode
pub trait Codec<T> {
    /// Encodes a message into a byte buffer.
    ///
    /// # Arguments
    /// * `msg` - The message to encode
    /// * `buf` - Buffer to write encoded bytes
    fn encode(&self, msg: &T, buf: &mut BytesMut);

    /// Decodes a message from bytes.
    ///
    /// # Arguments
    /// * `buf` - Bytes to decode
    ///
    /// # Returns
    /// The decoded message, or an error if decoding fails.
    fn decode(&self, buf: &mut Bytes) -> Result<T>;
}

// ============================================
// ProtocolCodec
// ============================================

/// Codec implementation for all protocol messages.
#[derive(Debug, Default, Clone)]
pub struct ProtocolCodec;

impl ProtocolCodec {
    /// Creates a new protocol codec.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Identifies the message type from a buffer without consuming it.
    ///
    /// # Arguments
    /// * `buf` - Buffer containing at least 1 byte
    ///
    /// # Returns
    /// The message type, or an error if unknown.
    pub fn peek_message_type(buf: &[u8]) -> Result<MessageType> {
        if buf.is_empty() {
            return Err(CoreError::too_short(1, 0));
        }
        MessageType::from_byte(buf[0])
            .ok_or(CoreError::UnknownMessageType(buf[0]))
    }

    /// Checks if the buffer contains a complete message.
    ///
    /// # Arguments
    /// * `buf` - Buffer to check
    ///
    /// # Returns
    /// - `Ok(Some(len))` - Complete message of `len` bytes
    /// - `Ok(None)` - Incomplete message, need more data
    /// - `Err(_)` - Invalid/unknown message type
    pub fn check_complete(buf: &[u8]) -> Result<Option<usize>> {
        if buf.is_empty() {
            return Ok(None);
        }

        let msg_type = Self::peek_message_type(buf)?;
        
        let required = match msg_type {
            MessageType::ClientHello => CLIENT_HELLO_SIZE,
            MessageType::ServerHello => SERVER_HELLO_SIZE,
            MessageType::Data => {
                if buf.len() < DATA_PACKET_HEADER_SIZE {
                    return Ok(None);
                }
                // Data packets are variable length
                // For now, we'll handle this in the decode method
                return Ok(Some(buf.len()));
            }
            MessageType::Keepalive => 1, // Just the type byte
        };

        if buf.len() >= required {
            Ok(Some(required))
        } else {
            Ok(None)
        }
    }
}

// ============================================
// ClientHello Codec
// ============================================

impl Codec<ClientHello> for ProtocolCodec {
    fn encode(&self, msg: &ClientHello, buf: &mut BytesMut) {
        buf.reserve(CLIENT_HELLO_SIZE);
        buf.put_u8(msg.message_type);
        buf.put_u8(msg.version);
        buf.put_slice(&msg.client_public_key);
        buf.put_slice(&msg.client_ephemeral_key);
        buf.put_i64_le(msg.timestamp);
        buf.put_slice(&msg.signature);
    }

    fn decode(&self, buf: &mut Bytes) -> Result<ClientHello> {
        if buf.len() < CLIENT_HELLO_SIZE {
            return Err(CoreError::too_short(CLIENT_HELLO_SIZE, buf.len()));
        }

        let message_type = buf.get_u8();
        if message_type != MessageType::ClientHello.as_byte() {
            return Err(CoreError::malformed(format!(
                "Expected ClientHello (0x01), got 0x{:02x}",
                message_type
            )));
        }

        let version = buf.get_u8();

        let mut client_public_key = [0u8; 32];
        buf.copy_to_slice(&mut client_public_key);

        let mut client_ephemeral_key = [0u8; 32];
        buf.copy_to_slice(&mut client_ephemeral_key);

        let timestamp = buf.get_i64_le();

        let mut signature = [0u8; 64];
        buf.copy_to_slice(&mut signature);

        Ok(ClientHello {
            message_type,
            version,
            client_public_key,
            client_ephemeral_key,
            timestamp,
            signature,
        })
    }
}

// ============================================
// ServerHello Codec
// ============================================

impl Codec<ServerHello> for ProtocolCodec {
    fn encode(&self, msg: &ServerHello, buf: &mut BytesMut) {
        buf.reserve(SERVER_HELLO_SIZE);
        buf.put_u8(msg.message_type);
        buf.put_u8(msg.version);
        buf.put_slice(&msg.server_public_key);
        buf.put_slice(&msg.server_ephemeral_key);
        buf.put_slice(&msg.assigned_ip);
        buf.put_slice(&msg.session_id);
        buf.put_slice(&msg.signature);
    }

    fn decode(&self, buf: &mut Bytes) -> Result<ServerHello> {
        if buf.len() < SERVER_HELLO_SIZE {
            return Err(CoreError::too_short(SERVER_HELLO_SIZE, buf.len()));
        }

        let message_type = buf.get_u8();
        if message_type != MessageType::ServerHello.as_byte() {
            return Err(CoreError::malformed(format!(
                "Expected ServerHello (0x02), got 0x{:02x}",
                message_type
            )));
        }

        let version = buf.get_u8();

        let mut server_public_key = [0u8; 32];
        buf.copy_to_slice(&mut server_public_key);

        let mut server_ephemeral_key = [0u8; 32];
        buf.copy_to_slice(&mut server_ephemeral_key);

        let mut assigned_ip = [0u8; 4];
        buf.copy_to_slice(&mut assigned_ip);

        let mut session_id = [0u8; 16];
        buf.copy_to_slice(&mut session_id);

        let mut signature = [0u8; 64];
        buf.copy_to_slice(&mut signature);

        Ok(ServerHello {
            message_type,
            version,
            server_public_key,
            server_ephemeral_key,
            assigned_ip,
            session_id,
            signature,
        })
    }
}

// ============================================
// DataPacket Codec
// ============================================

impl Codec<DataPacket> for ProtocolCodec {
    fn encode(&self, msg: &DataPacket, buf: &mut BytesMut) {
        buf.reserve(DATA_PACKET_HEADER_SIZE + msg.encrypted_payload.len());
        buf.put_slice(&msg.session_id);
        buf.put_u64_le(msg.counter);
        buf.put_slice(&msg.encrypted_payload);
    }

    fn decode(&self, buf: &mut Bytes) -> Result<DataPacket> {
        if buf.len() < DATA_PACKET_HEADER_SIZE {
            return Err(CoreError::too_short(DATA_PACKET_HEADER_SIZE, buf.len()));
        }

        let mut session_id = [0u8; 16];
        buf.copy_to_slice(&mut session_id);

        let counter = buf.get_u64_le();

        // Remaining bytes are the encrypted payload
        let encrypted_payload = buf.to_vec();

        Ok(DataPacket {
            session_id,
            counter,
            encrypted_payload,
        })
    }
}

// ============================================
// Convenience Functions
// ============================================

/// Encodes a ClientHello message to bytes.
#[must_use]
pub fn encode_client_hello(msg: &ClientHello) -> BytesMut {
    let mut buf = BytesMut::with_capacity(CLIENT_HELLO_SIZE);
    ProtocolCodec.encode(msg, &mut buf);
    buf
}

/// Decodes a ClientHello message from bytes.
pub fn decode_client_hello(buf: &[u8]) -> Result<ClientHello> {
    let mut bytes = Bytes::copy_from_slice(buf);
    ProtocolCodec.decode(&mut bytes)
}

/// Encodes a ServerHello message to bytes.
#[must_use]
pub fn encode_server_hello(msg: &ServerHello) -> BytesMut {
    let mut buf = BytesMut::with_capacity(SERVER_HELLO_SIZE);
    ProtocolCodec.encode(msg, &mut buf);
    buf
}

/// Decodes a ServerHello message from bytes.
pub fn decode_server_hello(buf: &[u8]) -> Result<ServerHello> {
    let mut bytes = Bytes::copy_from_slice(buf);
    ProtocolCodec.decode(&mut bytes)
}

/// Encodes a DataPacket to bytes.
#[must_use]
pub fn encode_data_packet(msg: &DataPacket) -> BytesMut {
    let mut buf = BytesMut::with_capacity(DATA_PACKET_HEADER_SIZE + msg.encrypted_payload.len());
    ProtocolCodec.encode(msg, &mut buf);
    buf
}

/// Decodes a DataPacket from bytes.
pub fn decode_data_packet(buf: &[u8]) -> Result<DataPacket> {
    let mut bytes = Bytes::copy_from_slice(buf);
    ProtocolCodec.decode(&mut bytes)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_hello_roundtrip() {
        let original = ClientHello {
            message_type: MessageType::ClientHello.as_byte(),
            version: 1,
            client_public_key: [0x01u8; 32],
            client_ephemeral_key: [0x02u8; 32],
            timestamp: 1234567890,
            signature: [0x03u8; 64],
        };

        let encoded = encode_client_hello(&original);
        assert_eq!(encoded.len(), CLIENT_HELLO_SIZE);

        let decoded = decode_client_hello(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_server_hello_roundtrip() {
        let original = ServerHello {
            message_type: MessageType::ServerHello.as_byte(),
            version: 1,
            server_public_key: [0x01u8; 32],
            server_ephemeral_key: [0x02u8; 32],
            assigned_ip: [100, 64, 0, 2],
            session_id: [0x04u8; 16],
            signature: [0x05u8; 64],
        };

        let encoded = encode_server_hello(&original);
        assert_eq!(encoded.len(), SERVER_HELLO_SIZE);

        let decoded = decode_server_hello(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_data_packet_roundtrip() {
        let original = DataPacket {
            session_id: [0x01u8; 16],
            counter: 42,
            encrypted_payload: vec![0x02u8; 100],
        };

        let encoded = encode_data_packet(&original);
        assert_eq!(encoded.len(), DATA_PACKET_HEADER_SIZE + 100);

        let decoded = decode_data_packet(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_peek_message_type() {
        let client_hello = [0x01u8; 138];
        assert_eq!(
            ProtocolCodec::peek_message_type(&client_hello).unwrap(),
            MessageType::ClientHello
        );

        let server_hello = [0x02u8; 150];
        assert_eq!(
            ProtocolCodec::peek_message_type(&server_hello).unwrap(),
            MessageType::ServerHello
        );

        let unknown = [0xFFu8; 10];
        assert!(ProtocolCodec::peek_message_type(&unknown).is_err());
    }

    #[test]
    fn test_check_complete() {
        // Empty buffer
        assert_eq!(ProtocolCodec::check_complete(&[]).unwrap(), None);

        // Incomplete ClientHello
        let partial = [0x01u8; 50];
        assert_eq!(ProtocolCodec::check_complete(&partial).unwrap(), None);

        // Complete ClientHello
        let complete = [0x01u8; CLIENT_HELLO_SIZE];
        assert_eq!(
            ProtocolCodec::check_complete(&complete).unwrap(),
            Some(CLIENT_HELLO_SIZE)
        );
    }

    #[test]
    fn test_decode_wrong_message_type() {
        let server_hello_bytes = [0x02u8; SERVER_HELLO_SIZE];
        let result = decode_client_hello(&server_hello_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_too_short() {
        let short = [0x01u8; 50];
        let result = decode_client_hello(&short);
        assert!(matches!(result, Err(CoreError::MessageTooShort { .. })));
    }

    #[test]
    fn test_timestamp_encoding() {
        // Test that timestamp is correctly encoded as little-endian
        let hello = ClientHello {
            message_type: MessageType::ClientHello.as_byte(),
            version: 1,
            client_public_key: [0u8; 32],
            client_ephemeral_key: [0u8; 32],
            timestamp: 0x0102030405060708i64,
            signature: [0u8; 64],
        };

        let encoded = encode_client_hello(&hello);
        
        // Timestamp is at offset 1 + 1 + 32 + 32 = 66
        let timestamp_bytes = &encoded[66..74];
        assert_eq!(timestamp_bytes, &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }
}
