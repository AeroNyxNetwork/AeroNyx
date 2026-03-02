// ============================================
// File: crates/aeronyx-core/src/protocol/memchain.rs
// ============================================
//! # MemChain Protocol Messages
//!
//! ## Creation Reason
//! Defines the application-layer messages that travel **inside** the
//! existing AeroNyx encrypted DataPacket. These messages are invisible
//! to any network observer — they share the exact same session
//! encryption, counter, and wire format as normal VPN traffic.
//!
//! ## Multiplexing Design (The 1st-Byte Hack)
//! After decryption, the plaintext's first byte determines the payload type:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ plaintext[0]  │  Meaning                                  │
//! ├───────────────┼───────────────────────────────────────────┤
//! │  0x40..0x4F   │  IPv4 (0x45) / IPv6 (0x60) — VPN traffic │
//! │  0xAE         │  MemChain message (our custom prefix)     │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! `0xAE` was chosen because:
//! - It does NOT collide with any IP version nibble (4 or 6).
//! - It is the first byte of "AEronyx" — easy to remember.
//! - It falls outside the valid IP version range, so no legitimate
//!   VPN packet will ever start with this byte.
//!
//! ## Wire Format
//! ```text
//! ┌──────┬──────────────────────────────────────────────┐
//! │ 0xAE │  bincode-serialised MemChainMessage          │
//! └──────┴──────────────────────────────────────────────┘
//! ```
//!
//! ## Main Functionality
//! - `MEMCHAIN_MAGIC`: The `0xAE` prefix constant.
//! - `MemChainMessage` enum: All MemChain P2P operations.
//! - `encode_memchain()` / `decode_memchain()`: Helpers that prepend /
//!   strip the magic byte and (de)serialise with `bincode`.
//!
//! ## Dependencies
//! - `serde` / `bincode` (workspace)
//! - `aeronyx_core::ledger::Fact`
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER change `MEMCHAIN_MAGIC` — it would break all in-flight
//!   MemChain traffic and the multiplexing router in `packet.rs`.
//! - Adding new variants to `MemChainMessage` is safe (bincode handles
//!   enum discriminants), but NEVER reorder or remove existing variants.
//! - The `encode_memchain` output is what gets encrypted by the existing
//!   `TransportCrypto::encrypt` — no additional encryption is needed.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain protocol messages for P2P memory sync

use serde::{Deserialize, Serialize};

use crate::ledger::Fact;

// ============================================
// Constants
// ============================================

/// Magic byte prepended to every MemChain plaintext payload.
///
/// Chosen to be outside the valid IP version nibble range (4/6)
/// so that the multiplexer in `packet.rs` can distinguish MemChain
/// traffic from normal VPN IP packets with a single byte peek.
///
/// `0xAE` = first byte of "**AE**ronyx".
pub const MEMCHAIN_MAGIC: u8 = 0xAE;

// ============================================
// MemChainMessage
// ============================================

/// Application-layer messages for MemChain P2P memory synchronisation.
///
/// These are serialised with `bincode`, prefixed with [`MEMCHAIN_MAGIC`],
/// and then encrypted inside a standard `DataPacket`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemChainMessage {
    /// Broadcast a newly created Fact to peers.
    BroadcastFact(Fact),

    /// Request synchronisation: "send me all facts after this hash".
    SyncRequest {
        /// Hash of the last known fact on the requester's chain.
        last_known_hash: [u8; 32],
    },

    /// Response to a sync request with a batch of facts.
    SyncResponse {
        /// Ordered batch of facts the requester is missing.
        facts: Vec<Fact>,
    },

    /// Query: ask a peer whether it has a specific fact.
    QueryRequest {
        /// The `fact_id` to look up.
        fact_id: [u8; 32],
    },

    /// Query response.
    QueryResponse {
        /// `None` if the fact is not found.
        fact: Option<Fact>,
    },

    /// Lightweight ping to verify MemChain layer is alive.
    Ping {
        /// Opaque nonce echoed back in `Pong`.
        nonce: u64,
    },

    /// Response to a `Ping`.
    Pong {
        /// Echoed nonce from the `Ping`.
        nonce: u64,
    },
}

// ============================================
// Encode / Decode helpers
// ============================================

/// Encodes a `MemChainMessage` into a byte vector with the `0xAE` prefix.
///
/// The returned `Vec<u8>` is the **plaintext** that should be handed to
/// `TransportCrypto::encrypt` in place of a normal IP packet.
///
/// # Errors
/// Returns a bincode serialisation error on failure (should never happen
/// for well-formed messages).
pub fn encode_memchain(msg: &MemChainMessage) -> std::result::Result<Vec<u8>, bincode::Error> {
    let payload = bincode::serialize(msg)?;
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(MEMCHAIN_MAGIC);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decodes a `MemChainMessage` from a plaintext slice whose first byte
/// (`MEMCHAIN_MAGIC`) has **already been verified and stripped** by the
/// caller.
///
/// # Arguments
/// * `payload` - The bytes **after** the `0xAE` prefix.
///
/// # Errors
/// Returns a bincode deserialisation error if the payload is malformed.
pub fn decode_memchain(payload: &[u8]) -> std::result::Result<MemChainMessage, bincode::Error> {
    bincode::deserialize(payload)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_does_not_collide_with_ip() {
        // IPv4 packets start with 0x4X (typically 0x45)
        // IPv6 packets start with 0x6X (typically 0x60)
        assert_ne!(MEMCHAIN_MAGIC >> 4, 4, "Must not collide with IPv4");
        assert_ne!(MEMCHAIN_MAGIC >> 4, 6, "Must not collide with IPv6");
    }

    #[test]
    fn test_broadcast_fact_roundtrip() {
        let fact = Fact::new(1_700_000_000, "s".into(), "p".into(), "o".into());
        let msg = MemChainMessage::BroadcastFact(fact.clone());

        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::BroadcastFact(f) => assert_eq!(f, fact),
            other => panic!("Expected BroadcastFact, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_request_roundtrip() {
        let msg = MemChainMessage::SyncRequest {
            last_known_hash: [0xAB; 32],
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::SyncRequest { last_known_hash } => {
                assert_eq!(last_known_hash, [0xAB; 32]);
            }
            other => panic!("Expected SyncRequest, got {:?}", other),
        }
    }

    #[test]
    fn test_ping_pong_roundtrip() {
        let msg = MemChainMessage::Ping { nonce: 42 };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::Ping { nonce } => assert_eq!(nonce, 42),
            other => panic!("Expected Ping, got {:?}", other),
        }
    }
}
