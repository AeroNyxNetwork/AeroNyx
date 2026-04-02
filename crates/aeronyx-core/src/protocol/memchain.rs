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
//! - 🌟 v1.1.0-ChatRelay: Added 5 Chat Relay variants at the END:
//!   `ChatRelay`, `ChatPull`, `ChatPullResponse`, `ChatAck`, `ChatExpired`.
//!   These enable zero-knowledge P2P messaging: the node forwards
//!   encrypted envelopes without being able to read message content.
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
//! - Chat Relay variants (11-15): the node validates the Ed25519 signature
//!   in `ChatRelay` but cannot decrypt `ChatEnvelope.ciphertext`.
//!   `ChatPull` is owner-authenticated by the session (wallet == session key).
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain protocol messages for P2P memory sync
//! v0.5.0 - 🌟 Added BlockAnnounce variant for Miner block broadcast
//! v1.0.0 - 🌟 Added BroadcastRecord, SyncRecordRequest, SyncRecordResponse
//! v1.1.0-ChatRelay - 🌟 Added ChatRelay, ChatPull, ChatPullResponse,
//!                        ChatAck, ChatExpired for zero-knowledge P2P messaging

use serde::{Deserialize, Serialize};
use bincode::Options;

#[allow(deprecated)]
use crate::ledger::Fact;
use crate::ledger::{BlockHeader, MemoryRecord};
use crate::protocol::chat::ChatEnvelope;

// ============================================
// Deserialisation size limits
// ============================================

/// Maximum accepted size for a single MemChain message payload (excluding magic byte).
/// Protects against malicious length-prefix attacks that cause multi-GB allocations.
/// - SyncRecordResponse with large records: ~2 MB upper bound
/// - BroadcastRecord with embedding: ~64 KB typical
/// - ChatPullResponse (100 envelopes × ~1 KB): ~100 KB typical
const MAX_MEMCHAIN_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024; // 2 MB

/// Maximum accepted size for a single `ChatEnvelope` when decoded standalone.
/// Text ciphertext ≤ 64 KB + fixed fields ≤ 1 KB overhead.
const MAX_ENVELOPE_BYTES: u64 = 128 * 1024; // 128 KB

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

/// Application-layer messages for MemChain P2P memory synchronisation
/// and zero-knowledge chat relay.
///
/// These are serialised with `bincode`, prefixed with [`MEMCHAIN_MAGIC`],
/// and then encrypted inside a standard `DataPacket`.
///
/// ## Variant Ordering — STABLE CONTRACT
/// bincode serialises enum discriminants by index. The order below
/// MUST NOT change. New variants MUST be appended at the end.
///
/// | Index | Variant              | Added in               |
/// |-------|----------------------|------------------------|
/// | 0     | BroadcastFact        | v0.2.0                 |
/// | 1     | SyncRequest          | v0.2.0                 |
/// | 2     | SyncResponse         | v0.2.0                 |
/// | 3     | QueryRequest         | v0.2.0                 |
/// | 4     | QueryResponse        | v0.2.0                 |
/// | 5     | Ping                 | v0.2.0                 |
/// | 6     | Pong                 | v0.2.0                 |
/// | 7     | BlockAnnounce        | v0.5.0                 |
/// | 8     | BroadcastRecord      | v1.0.0                 |
/// | 9     | SyncRecordRequest    | v1.0.0                 |
/// | 10    | SyncRecordResponse   | v1.0.0                 |
/// | 11    | ChatRelay            | v1.1.0-ChatRelay       |
/// | 12    | ChatPull             | v1.1.0-ChatRelay       |
/// | 13    | ChatPullResponse     | v1.1.0-ChatRelay       |
/// | 14    | ChatAck              | v1.1.0-ChatRelay       |
/// | 15    | ChatExpired          | v1.1.0-ChatRelay       |
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

    // ====================================================
    // 🌟 v1.1.0-ChatRelay: Zero-knowledge P2P chat (indices 11-15)
    // ====================================================

    /// 🌟 [index 11] Deliver an E2E-encrypted chat message to the target wallet.
    ///
    /// The node validates the Ed25519 signature in the envelope, then either:
    /// - Forwards immediately if the receiver has an active session.
    /// - Stores in `chat_pending.db` for up to `offline_ttl_secs` (default 72h).
    ///
    /// The node CANNOT decrypt `ChatEnvelope.ciphertext` — it only reads
    /// `sender`, `receiver`, `timestamp`, and `content_type` for routing.
    ///
    /// # Deduplication
    /// - **Offline path**: SQLite PRIMARY KEY on `message_id` rejects duplicates.
    /// - **Online path**: In-memory LRU cache (capacity 10 000) rejects duplicates.
    ///
    /// # Size Limits
    /// - Text (content_type=0): ciphertext ≤ 64 KB
    /// - Media (content_type=1): ciphertext is a small encrypted MediaPointer;
    ///   the actual file travels via `POST /api/chat/blob`.
    ChatRelay(ChatEnvelope),

    /// 🌟 [index 12] Pull pending offline messages for the authenticated wallet.
    ///
    /// Sent by the client immediately after establishing a session.
    /// The node verifies that `wallet == session.client_public_key` to prevent
    /// one client from pulling another wallet's messages.
    ///
    /// # Pagination
    /// Use `cursor = [0u8; 16]` for the first request.
    /// If the response has `has_more = true`, send another `ChatPull` with
    /// `cursor = last received envelope's message_id`.
    ///
    /// # Incremental Sync
    /// Set `after_timestamp` to the last-seen message timestamp to avoid
    /// re-downloading messages already stored on the device.
    ChatPull {
        /// The wallet requesting its offline messages (must equal session key).
        wallet: [u8; 32],
        /// Only return messages with timestamp > this value (0 = all messages).
        after_timestamp: u64,
        /// Pagination cursor: last message_id from previous response.
        /// Use `[0u8; 16]` for the first request.
        cursor: [u8; 16],
        /// Maximum messages per page (capped at 100 by the server).
        limit: u32,
    },

    /// 🌟 [index 13] Response to `ChatPull` with a page of pending messages.
    ///
    /// The client should:
    /// 1. Decrypt and persist all envelopes locally.
    /// 2. Send `ChatAck` with all received `message_id`s.
    /// 3. If `has_more == true`, send another `ChatPull` with the last
    ///    envelope's `message_id` as the new `cursor`.
    ChatPullResponse {
        /// Ordered batch of pending `ChatEnvelope`s (oldest first).
        envelopes: Vec<ChatEnvelope>,
        /// When `true`, there are more pages to fetch.
        /// Client should continue pulling with an updated cursor.
        has_more: bool,
    },

    /// 🌟 [index 14] Acknowledge successful receipt of one or more messages.
    ///
    /// Sent by the receiver (Bob) after decrypting and persisting each batch.
    /// The node deletes acknowledged messages from `chat_pending.db`.
    ///
    /// # Security
    /// The node validates that `session.wallet == receiver` of each acked
    /// message before deleting, preventing one user from deleting another's
    /// pending messages.
    ChatAck {
        /// Message IDs that have been successfully received and persisted.
        message_ids: Vec<[u8; 16]>,
    },

    /// 🌟 [index 15] Notify sender that messages expired before delivery.
    ///
    /// Sent by the node → Alice when pending messages exceed `offline_ttl_secs`
    /// (default 72 h) without being acknowledged by Bob.
    ///
    /// If Alice is offline when the TTL fires, the notification is queued in
    /// `expired_notifications` table and delivered when Alice next connects.
    /// Notifications are discarded after 7 days if Alice never reconnects.
    ///
    /// Alice's Flutter client should update the local message status to
    /// "undelivered" upon receiving this notification.
    ChatExpired {
        /// IDs of expired messages that were never delivered to `receiver`.
        message_ids: Vec<[u8; 16]>,
        /// The intended receiver wallet (helps Alice identify which conversation
        /// the expired messages belong to).
        receiver: [u8; 32],
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
///
/// # Size limit
/// Rejects payloads that would require allocating more than
/// `MAX_MEMCHAIN_PAYLOAD_BYTES` (2 MB). This prevents a malicious
/// length-prefix from triggering a multi-GB allocation (OOM DoS).
pub fn decode_memchain(payload: &[u8]) -> std::result::Result<MemChainMessage, bincode::Error> {
    bincode::options()
        .with_limit(MAX_MEMCHAIN_PAYLOAD_BYTES)
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .deserialize(payload)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(deprecated)]
    use crate::ledger::{BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH, MemoryLayer};
    use crate::crypto::IdentityKeyPair;
    use crate::protocol::chat::{ChatContentType, encode_envelope};

    // ── Existing tests (preserved verbatim) ─────────────────────────────

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

    // ── Discriminant stability (extended to cover chat variants) ────────

    /// Verify that legacy discriminant indices are preserved.
    /// Extended in v1.1.0-ChatRelay to also verify indices 11-15.
    /// This test would catch accidental reordering of variants.
    #[test]
    fn test_discriminant_stability() {
        // BroadcastFact = index 0
        let fact_msg = MemChainMessage::BroadcastFact(
            Fact::new(0, "s".into(), "p".into(), "o".into()),
        );
        let fact_bytes = bincode::serialize(&fact_msg).expect("ser");
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

        // ChatRelay = index 11
        let kp = IdentityKeyPair::generate();
        let mut chat_env = ChatEnvelope {
            message_id: [0x01; 16],
            sender: kp.public_key_bytes(),
            receiver: [0xBB; 32],
            timestamp: 1_700_000_000,
            ciphertext: b"test".to_vec(),
            nonce: [0x02; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        let data = chat_env.sign_data();
        chat_env.signature = kp.sign(&data);

        let chat_relay_msg = MemChainMessage::ChatRelay(chat_env);
        let chat_bytes = bincode::serialize(&chat_relay_msg).expect("ser");
        let disc = u32::from_le_bytes([chat_bytes[0], chat_bytes[1], chat_bytes[2], chat_bytes[3]]);
        assert_eq!(disc, 11, "ChatRelay must be discriminant 11");

        // ChatPull = index 12
        let pull_msg = MemChainMessage::ChatPull {
            wallet: [0xAA; 32],
            after_timestamp: 0,
            cursor: [0u8; 16],
            limit: 50,
        };
        let pull_bytes = bincode::serialize(&pull_msg).expect("ser");
        let disc = u32::from_le_bytes([pull_bytes[0], pull_bytes[1], pull_bytes[2], pull_bytes[3]]);
        assert_eq!(disc, 12, "ChatPull must be discriminant 12");

        // ChatPullResponse = index 13
        let pull_resp_msg = MemChainMessage::ChatPullResponse {
            envelopes: vec![],
            has_more: false,
        };
        let pull_resp_bytes = bincode::serialize(&pull_resp_msg).expect("ser");
        let disc = u32::from_le_bytes([pull_resp_bytes[0], pull_resp_bytes[1], pull_resp_bytes[2], pull_resp_bytes[3]]);
        assert_eq!(disc, 13, "ChatPullResponse must be discriminant 13");

        // ChatAck = index 14
        let ack_msg = MemChainMessage::ChatAck {
            message_ids: vec![[0u8; 16]],
        };
        let ack_bytes = bincode::serialize(&ack_msg).expect("ser");
        let disc = u32::from_le_bytes([ack_bytes[0], ack_bytes[1], ack_bytes[2], ack_bytes[3]]);
        assert_eq!(disc, 14, "ChatAck must be discriminant 14");

        // ChatExpired = index 15
        let expired_msg = MemChainMessage::ChatExpired {
            message_ids: vec![[0u8; 16]],
            receiver: [0xCC; 32],
        };
        let expired_bytes = bincode::serialize(&expired_msg).expect("ser");
        let disc = u32::from_le_bytes([expired_bytes[0], expired_bytes[1], expired_bytes[2], expired_bytes[3]]);
        assert_eq!(disc, 15, "ChatExpired must be discriminant 15");
    }

    // ── Chat Relay variant roundtrip tests ───────────────────────────────

    #[test]
    fn test_chat_relay_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let mut env = ChatEnvelope {
            message_id: [0xDE; 16],
            sender: kp.public_key_bytes(),
            receiver: [0xBE; 32],
            timestamp: 1_700_000_001,
            ciphertext: b"hello encrypted world".to_vec(),
            nonce: [0x05; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        let data = env.sign_data();
        env.signature = kp.sign(&data);

        let msg = MemChainMessage::ChatRelay(env.clone());
        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatRelay(e) => {
                assert_eq!(e.message_id, env.message_id);
                assert_eq!(e.sender, env.sender);
                assert_eq!(e.receiver, env.receiver);
                assert_eq!(e.ciphertext, env.ciphertext);
                assert_eq!(e.content_type, ChatContentType::Text);
                // Signature must still verify after encode/decode
                assert!(e.verify_signature().is_ok(), "Signature must survive roundtrip");
            }
            other => panic!("Expected ChatRelay, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_pull_roundtrip() {
        let msg = MemChainMessage::ChatPull {
            wallet: [0xAA; 32],
            after_timestamp: 1_700_000_000,
            cursor: [0x01; 16],
            limit: 50,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatPull { wallet, after_timestamp, cursor, limit } => {
                assert_eq!(wallet, [0xAA; 32]);
                assert_eq!(after_timestamp, 1_700_000_000);
                assert_eq!(cursor, [0x01; 16]);
                assert_eq!(limit, 50);
            }
            other => panic!("Expected ChatPull, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_pull_response_roundtrip() {
        let msg = MemChainMessage::ChatPullResponse {
            envelopes: vec![],
            has_more: true,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatPullResponse { envelopes, has_more } => {
                assert!(envelopes.is_empty());
                assert!(has_more);
            }
            other => panic!("Expected ChatPullResponse, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_ack_roundtrip() {
        let ids: Vec<[u8; 16]> = vec![[0xAA; 16], [0xBB; 16]];
        let msg = MemChainMessage::ChatAck { message_ids: ids.clone() };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatAck { message_ids } => {
                assert_eq!(message_ids, ids);
            }
            other => panic!("Expected ChatAck, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_expired_roundtrip() {
        let ids: Vec<[u8; 16]> = vec![[0xCC; 16]];
        let receiver = [0xDD; 32];
        let msg = MemChainMessage::ChatExpired {
            message_ids: ids.clone(),
            receiver,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatExpired { message_ids, receiver: r } => {
                assert_eq!(message_ids, ids);
                assert_eq!(r, receiver);
            }
            other => panic!("Expected ChatExpired, got {:?}", other),
        }
    }

    /// Verify that v1.0.0 messages still decode correctly after v1.1.0 additions.
    /// Simulates a v1.0.0 node message being decoded by a v1.1.0 node.
    #[test]
    fn test_backward_compat_v100_messages_still_decode() {
        // SyncRecordRequest (index 9) must still decode unchanged
        let msg = MemChainMessage::SyncRecordRequest {
            owner: [0xEE; 32],
            after_timestamp: 9_999_999,
        };
        let bytes = bincode::serialize(&msg).expect("serialize");
        let disc = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(disc, 9, "SyncRecordRequest discriminant must remain 9");

        let decoded: MemChainMessage = bincode::deserialize(&bytes).expect("deserialize");
        match decoded {
            MemChainMessage::SyncRecordRequest { owner, after_timestamp } => {
                assert_eq!(owner, [0xEE; 32]);
                assert_eq!(after_timestamp, 9_999_999);
            }
            other => panic!("Backward compat failed: got {:?}", other),
        }
    }
}
