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
//! ## Modification Reason
//! - 🌟 v0.5.0: Added `BlockAnnounce(BlockHeader)` variant at the END
//!   of the enum to preserve bincode discriminant compatibility.
//!   This allows the Miner to broadcast lightweight block headers
//!   (<100 bytes) over UDP without hitting MTU limits.
//! - 🌟 v1.0.0: Added `BroadcastRecord(MemoryRecord)` variant at the
//!   END of the enum for broadcasting MRS-1 MemoryRecord entries.
//!   Also added `SyncRecordRequest` and `SyncRecordResponse` for
//!   record-based P2P synchronisation.
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
//! ## Wire Format
//! ```text
//! ┌──────┬──────────────────────────────────────────────┐
//! │ 0xAE │  bincode-serialised MemChainMessage          │
//! └──────┴──────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER change `MEMCHAIN_MAGIC` — it would break all in-flight
//!   MemChain traffic and the multiplexing router in `packet.rs`.
//! - Adding new variants to `MemChainMessage` is safe ONLY at the end.
//!   NEVER reorder or remove existing variants (bincode discriminants).
//! - `BlockAnnounce` carries only the header (~100 bytes), NOT the full
//!   Block. Full block retrieval uses SyncRequest/SyncResponse.
//! - `BroadcastRecord` carries a full MemoryRecord including encrypted
//!   content. Ensure MTU safety for large records (may need chunking
//!   in future phases).
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain protocol messages for P2P memory sync
//! v0.5.0 - 🌟 Added BlockAnnounce variant for Miner block broadcast
//! v1.0.0 - 🌟 Added BroadcastRecord, SyncRecordRequest, SyncRecordResponse

use serde::{Deserialize, Serialize};

#[allow(deprecated)]
use crate::ledger::Fact;
use crate::ledger::{BlockHeader, MemoryRecord};

// ============================================
// Constants
// ============================================

/// Magic byte prepended to every MemChain plaintext payload.
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
///
/// ## Variant Ordering — STABLE CONTRACT
/// bincode serialises enum discriminants by index. The order below
/// MUST NOT change. New variants MUST be appended at the end.
///
/// | Index | Variant              | Added in |
/// |-------|----------------------|----------|
/// | 0     | BroadcastFact        | v0.2.0   |
/// | 1     | SyncRequest          | v0.2.0   |
/// | 2     | SyncResponse         | v0.2.0   |
/// | 3     | QueryRequest         | v0.2.0   |
/// | 4     | QueryResponse        | v0.2.0   |
/// | 5     | Ping                 | v0.2.0   |
/// | 6     | Pong                 | v0.2.0   |
/// | 7     | BlockAnnounce        | v0.5.0   |
/// | 8     | BroadcastRecord      | v1.0.0   |
/// | 9     | SyncRecordRequest    | v1.0.0   |
/// | 10    | SyncRecordResponse   | v1.0.0   |
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(deprecated)] // BroadcastFact/SyncResponse use deprecated Fact type for P2P compat
pub enum MemChainMessage {
    /// Broadcast a newly created Fact to peers (legacy).
    BroadcastFact(Fact),

    /// Request synchronisation: "send me all facts after this hash" (legacy).
    SyncRequest {
        /// Hash of the last known fact on the requester's chain.
        last_known_hash: [u8; 32],
    },

    /// Response to a sync request with a batch of facts (legacy).
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

    /// 🌟 Announce a newly mined block (header only, <100 bytes).
    ///
    /// The full block is NOT broadcast over UDP (MTU risk).
    /// Peers that want the full block content can use `SyncRequest`.
    BlockAnnounce(BlockHeader),

    // ====================================================
    // 🌟 v1.0.0: MemoryRecord-based messages (MRS-1)
    // ====================================================

    /// 🌟 Broadcast a newly created MemoryRecord to peers.
    ///
    /// This is the MRS-1 equivalent of `BroadcastFact`. Contains
    /// the full record including encrypted content and embedding.
    ///
    /// # MTU Warning
    /// MemoryRecords with large encrypted content may exceed UDP
    /// MTU limits. The sender should check size before broadcasting.
    /// For records > 1200 bytes, consider using SyncRecordRequest
    /// for on-demand retrieval instead.
    BroadcastRecord(MemoryRecord),

    /// 🌟 Request record synchronisation.
    ///
    /// "Send me all MemoryRecords for this owner after this timestamp."
    /// This is more targeted than the legacy `SyncRequest` which sends
    /// all facts — record sync is owner-scoped and time-bounded.
    SyncRecordRequest {
        /// Owner wallet public key to sync records for.
        owner: [u8; 32],
        /// Only return records with timestamp > this value.
        /// Use 0 to request all records for this owner.
        after_timestamp: u64,
    },

    /// 🌟 Response to a record sync request.
    SyncRecordResponse {
        /// Ordered batch of MemoryRecords the requester is missing.
        records: Vec<MemoryRecord>,
    },
}

// ============================================
// Encode / Decode helpers
// ============================================

/// Encodes a `MemChainMessage` into a byte vector with the `0xAE` prefix.
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
pub fn decode_memchain(payload: &[u8]) -> std::result::Result<MemChainMessage, bincode::Error> {
    bincode::deserialize(payload)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(deprecated)]
    use crate::ledger::{BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH, MemoryLayer};

    #[test]
    fn test_magic_does_not_collide_with_ip() {
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

    #[test]
    fn test_block_announce_roundtrip() {
        let header = BlockHeader {
            height: 42,
            timestamp: 1_700_000_000,
            prev_block_hash: GENESIS_PREV_HASH,
            merkle_root: [0xBB; 32],
            block_type: BLOCK_TYPE_NORMAL,
        };
        let msg = MemChainMessage::BlockAnnounce(header.clone());

        let encoded = encode_memchain(&msg).expect("encode");
        // Verify it's well within MTU: 1 (magic) + bincode overhead + ~81 bytes
        assert!(encoded.len() < 200, "BlockAnnounce must be <200 bytes, got {}", encoded.len());

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::BlockAnnounce(h) => {
                assert_eq!(h.height, 42);
                assert_eq!(h, header);
            }
            other => panic!("Expected BlockAnnounce, got {:?}", other),
        }
    }

    #[test]
    fn test_broadcast_record_roundtrip() {
        let record = MemoryRecord::new(
            [0xAA; 32],
            1_700_000_000,
            MemoryLayer::Episode,
            vec!["test".into(), "memory".into()],
            "openclaw-v1".into(),
            b"encrypted_content".to_vec(),
            b"encrypted_embedding".to_vec(),
        );
        let msg = MemChainMessage::BroadcastRecord(record.clone());

        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::BroadcastRecord(r) => {
                assert_eq!(r.record_id, record.record_id);
                assert_eq!(r.layer, MemoryLayer::Episode);
                assert_eq!(r.source_ai, "openclaw-v1");
                assert_eq!(r.topic_tags, vec!["test", "memory"]);
            }
            other => panic!("Expected BroadcastRecord, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_record_request_roundtrip() {
        let msg = MemChainMessage::SyncRecordRequest {
            owner: [0xCC; 32],
            after_timestamp: 1_700_000_000,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::SyncRecordRequest { owner, after_timestamp } => {
                assert_eq!(owner, [0xCC; 32]);
                assert_eq!(after_timestamp, 1_700_000_000);
            }
            other => panic!("Expected SyncRecordRequest, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_record_response_roundtrip() {
        let record = MemoryRecord::new(
            [0xAA; 32],
            1_700_000_000,
            MemoryLayer::Knowledge,
            vec![],
            "test".into(),
            b"data".to_vec(),
            vec![],
        );
        let msg = MemChainMessage::SyncRecordResponse {
            records: vec![record.clone()],
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::SyncRecordResponse { records } => {
                assert_eq!(records.len(), 1);
                assert_eq!(records[0], record);
            }
            other => panic!("Expected SyncRecordResponse, got {:?}", other),
        }
    }

    /// Verify that legacy discriminant indices are preserved.
    /// This test would catch accidental reordering of variants.
    #[test]
    fn test_discriminant_stability() {
        // BroadcastFact = index 0
        let fact_msg = MemChainMessage::BroadcastFact(
            Fact::new(0, "s".into(), "p".into(), "o".into()),
        );
        let fact_bytes = bincode::serialize(&fact_msg).expect("ser");
        // First 4 bytes of bincode enum are the discriminant (u32 LE)
        let disc = u32::from_le_bytes([fact_bytes[0], fact_bytes[1], fact_bytes[2], fact_bytes[3]]);
        assert_eq!(disc, 0, "BroadcastFact must be discriminant 0");

        // BlockAnnounce = index 7
        let block_msg = MemChainMessage::BlockAnnounce(BlockHeader {
            height: 0, timestamp: 0,
            prev_block_hash: [0; 32], merkle_root: [0; 32],
            block_type: 0x01,
        });
        let block_bytes = bincode::serialize(&block_msg).expect("ser");
        let disc = u32::from_le_bytes([block_bytes[0], block_bytes[1], block_bytes[2], block_bytes[3]]);
        assert_eq!(disc, 7, "BlockAnnounce must be discriminant 7");

        // BroadcastRecord = index 8
        let record_msg = MemChainMessage::BroadcastRecord(MemoryRecord::new(
            [0; 32], 0, MemoryLayer::Episode, vec![], "".into(), vec![], vec![],
        ));
        let record_bytes = bincode::serialize(&record_msg).expect("ser");
        let disc = u32::from_le_bytes([record_bytes[0], record_bytes[1], record_bytes[2], record_bytes[3]]);
        assert_eq!(disc, 8, "BroadcastRecord must be discriminant 8");
    }
}
