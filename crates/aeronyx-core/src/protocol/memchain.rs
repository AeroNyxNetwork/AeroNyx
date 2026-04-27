// ============================================================================
// File: crates/aeronyx-core/src/protocol/memchain.rs
// ============================================================================
// Version: 1.3.0-Sovereign
//
// Modification Reason:
//   v1.3.0-Sovereign — Breaking protocol upgrade. Wallet identity is no longer
//   derived from the session key. Each sensitive message now carries an explicit
//   wallet_pubkey + timestamp + Ed25519 signature, allowing the server to verify
//   wallet ownership per-message without trusting the session binding.
//
// Main Functionality:
//   Defines all application-layer messages that travel inside the existing
//   AeroNyx encrypted DataPacket, multiplexed by the 0xAE magic byte.
//
// Dependencies:
//   - crates/aeronyx-core/src/ledger: Fact, BlockHeader, MemoryRecord
//   - crates/aeronyx-core/src/protocol/chat: ChatEnvelope
//   - bincode: serialization (positional — field order is wire format)
//
// Main Logical Flow:
//   1. Caller constructs a MemChainMessage variant
//   2. encode_memchain() serializes and prepends 0xAE
//   3. Outer layer encrypts the whole buffer as a DataPacket
//   4. On receipt: magic byte stripped, decode_memchain() deserializes
//   5. Server dispatches on variant, verifies signature via auth::verify_signed_message
//
// ⚠️ Important Notes for Next Developer:
//   - NEVER change MEMCHAIN_MAGIC (0xAE) — breaks all in-flight traffic
//   - NEVER reorder or remove existing enum variants (bincode discriminants)
//   - New variants MUST be appended at the END only
//   - v1.3.0 is a BREAKING CHANGE: DeviceRegister, ChatPull, ChatAck wire
//     format changed — old clients cannot talk to new servers and vice versa
//   - WalletPresence (17) is a lightweight heartbeat — node never replies
//   - serde_bytes64 is defined in chat.rs; the [u8;64] signature fields here
//     use the same two-[u8;32] trick for bincode compatibility
//
// Last Modified:
//   v1.2.0-MultiDevice — Added DeviceRegister (index 16)
//   v1.3.0-Sovereign   — Added wallet_pubkey/timestamp/signature to
//                        DeviceRegister, ChatPull, ChatAck; added WalletPresence (17)
// ============================================================================

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
const MAX_MEMCHAIN_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024; // 2 MB

/// Maximum accepted size for a single `ChatEnvelope` when decoded standalone.
const MAX_ENVELOPE_BYTES: u64 = 128 * 1024; // 128 KB

// ============================================
// Constants
// ============================================

/// Magic byte prepended to every MemChain plaintext payload.
/// `0xAE` = first byte of "**AE**ronyx".
pub const MEMCHAIN_MAGIC: u8 = 0xAE;

// ============================================
// Internal serde helper for [u8; 64]
// ============================================
// Identical to the one in chat.rs. Duplicated here to avoid a cross-module
// dependency for a single serde helper. Both produce the same wire bytes.
// DO NOT change the serialisation logic — it must stay wire-compatible.

mod serde_bytes64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        let (lo, hi) = v.split_at(32);
        let lo: [u8; 32] = lo.try_into().unwrap();
        let hi: [u8; 32] = hi.try_into().unwrap();
        (lo, hi).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let (lo, hi): ([u8; 32], [u8; 32]) = Deserialize::deserialize(d)?;
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        Ok(out)
    }
}

// ============================================
// MemChainMessage
// ============================================

/// Application-layer messages for MemChain P2P memory synchronisation
/// and zero-knowledge chat relay.
///
/// Serialised with `bincode`, prefixed with [`MEMCHAIN_MAGIC`], then
/// encrypted inside a standard `DataPacket`.
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
/// | 16    | DeviceRegister       | v1.2.0-MultiDevice     |
/// | 17    | WalletPresence       | v1.3.0-Sovereign       |
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(deprecated)]
pub enum MemChainMessage {
    /// Broadcast a newly created Fact to peers (legacy).
    BroadcastFact(Fact),

    /// Request synchronisation: "send me all facts after this hash" (legacy).
    SyncRequest {
        last_known_hash: [u8; 32],
    },

    /// Response to a sync request with a batch of facts (legacy).
    SyncResponse {
        facts: Vec<Fact>,
    },

    /// Query: ask a peer whether it has a specific fact.
    QueryRequest {
        fact_id: [u8; 32],
    },

    /// Query response.
    QueryResponse {
        fact: Option<Fact>,
    },

    /// Lightweight ping to verify MemChain layer is alive.
    Ping {
        nonce: u64,
    },

    /// Response to a `Ping`.
    Pong {
        nonce: u64,
    },

    /// Announce a newly mined block (header only, <100 bytes).
    BlockAnnounce(BlockHeader),

    // ── v1.0.0: MemoryRecord-based messages ─────────────────────────────

    /// Broadcast a newly created MemoryRecord to peers.
    BroadcastRecord(MemoryRecord),

    /// Request record synchronisation for an owner after a timestamp.
    SyncRecordRequest {
        owner: [u8; 32],
        after_timestamp: u64,
    },

    /// Response to a record sync request.
    SyncRecordResponse {
        records: Vec<MemoryRecord>,
    },

    // ── v1.1.0-ChatRelay: Zero-knowledge P2P chat (indices 11-15) ────────

    /// [index 11] Deliver an E2E-encrypted chat message to the target wallet.
    ///
    /// The node validates the Ed25519 signature in the envelope, then either
    /// forwards immediately (receiver online) or stores in chat_pending.db.
    ChatRelay(ChatEnvelope),

    /// [index 12] Pull pending offline messages for the authenticated wallet.
    ///
    /// ## v1.3.0-Sovereign Breaking Change
    /// Added `request_timestamp` and `signature` fields. The node now verifies
    /// the signature before serving messages. Old clients missing these fields
    /// will fail to deserialize this variant on the server side.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-ChatPull-v1" ||
    /// wallet(32) || after_timestamp(8,LE) || cursor(16) || limit(4,LE) ||
    /// request_timestamp(8,LE)
    ChatPull {
        /// The wallet requesting its offline messages.
        wallet: [u8; 32],
        /// Only return messages with timestamp > this value (0 = all).
        after_timestamp: u64,
        /// Pagination cursor: last message_id from previous response.
        /// Use `[0u8; 16]` for the first request.
        cursor: [u8; 16],
        /// Maximum messages per page (capped at 100 by the server).
        limit: u32,
        /// Unix epoch seconds when this request was constructed.
        /// Must be within ±60 s of server clock.
        request_timestamp: u64,
        /// Ed25519 signature over the canonical input described above.
        /// Proves the requester holds the private key for `wallet`.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 13] Response to `ChatPull` with a page of pending messages.
    ChatPullResponse {
        envelopes: Vec<ChatEnvelope>,
        has_more: bool,
    },

    /// [index 14] Acknowledge successful receipt of one or more messages.
    ///
    /// ## v1.3.0-Sovereign Breaking Change
    /// Added `wallet`, `ack_timestamp`, and `signature` fields.
    /// The node verifies the signature and enforces `receiver = wallet`
    /// before deleting — preventing cross-wallet ACK attacks.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-ChatAck-v1" ||
    /// wallet(32) || ack_timestamp(8,LE) || SHA256(message_ids_concatenated)(32)
    ///
    /// message_ids_concatenated = id[0](16) || id[1](16) || … (in list order)
    ChatAck {
        /// Message IDs that have been successfully received and persisted.
        message_ids: Vec<[u8; 16]>,
        /// The wallet that owns these messages (= receiver of each message).
        wallet: [u8; 32],
        /// Unix epoch seconds when this ACK was constructed.
        ack_timestamp: u64,
        /// Ed25519 signature over the canonical input described above.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 15] Notify sender that messages expired before delivery.
    ///
    /// Server → client only. Client should mark messages as "undelivered".
    ChatExpired {
        message_ids: Vec<[u8; 16]>,
        receiver: [u8; 32],
    },

    // ── v1.2.0-MultiDevice: Device registration (index 16) ───────────────

    /// [index 16] Register a device under the authenticated wallet.
    ///
    /// ## v1.3.0-Sovereign Breaking Change
    /// Added `wallet_pubkey`, `timestamp`, and `signature` fields.
    /// Session key is no longer trusted as wallet proof; the client must
    /// sign to prove ownership of the wallet private key.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-DeviceRegister-v1" ||
    /// session_id(16) || device_id(16) || wallet_pubkey(32) || timestamp(8,LE)
    DeviceRegister {
        /// Stable random ID generated once on device install.
        device_id: [u8; 16],
        /// Human-readable device label, max 64 bytes UTF-8.
        device_name: String,
        /// The wallet this device is registering under.
        /// Must match the signing key used to produce `signature`.
        wallet_pubkey: [u8; 32],
        /// Unix epoch seconds when this message was constructed.
        /// Must be within ±60 s of server clock.
        timestamp: u64,
        /// Ed25519 signature over the canonical input described above.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v1.3.0-Sovereign: Wallet presence heartbeat (index 17) ───────────

    /// [index 17] Lightweight heartbeat proving wallet ownership.
    ///
    /// Sent by the client periodically (recommended: every 60–120 s) to keep
    /// the in-memory wallet route table alive. The server never replies.
    ///
    /// ## Why this exists
    /// Without explicit heartbeats the route table entry for this wallet
    /// would be cleaned up after the stale TTL (default 300 s). Sending
    /// WalletPresence refreshes the `last_active` timestamp without the
    /// overhead of a full DeviceRegister.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-WalletPresence-v1" ||
    /// session_id(16) || wallet_pubkey(32) || timestamp(8,LE)
    WalletPresence {
        /// The wallet asserting its presence.
        wallet_pubkey: [u8; 32],
        /// Unix epoch seconds when this heartbeat was constructed.
        timestamp: u64,
        /// Ed25519 signature over the canonical input described above.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
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
/// (`MEMCHAIN_MAGIC`) has **already been verified and stripped** by the caller.
///
/// Rejects payloads that would require allocating more than
/// `MAX_MEMCHAIN_PAYLOAD_BYTES` (2 MB).
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
        let msg = MemChainMessage::SyncRequest { last_known_hash: [0xAB; 32] };
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
            [0xAA; 32], 1_700_000_000, MemoryLayer::Episode,
            vec!["test".into(), "memory".into()], "openclaw-v1".into(),
            b"encrypted_content".to_vec(), b"encrypted_embedding".to_vec(),
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
            [0xAA; 32], 1_700_000_000, MemoryLayer::Knowledge,
            vec![], "test".into(), b"data".to_vec(), vec![],
        );
        let msg = MemChainMessage::SyncRecordResponse { records: vec![record.clone()] };
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

    // ── Discriminant stability ───────────────────────────────────────────

    /// Verify that all discriminant indices are preserved.
    /// Extended in v1.3.0-Sovereign to include WalletPresence = 17.
    /// This test catches accidental reordering of variants.
    #[test]
    fn test_discriminant_stability() {
        fn disc(bytes: &[u8]) -> u32 {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }

        // BroadcastFact = 0
        let b = bincode::serialize(&MemChainMessage::BroadcastFact(
            Fact::new(0, "s".into(), "p".into(), "o".into()),
        )).unwrap();
        assert_eq!(disc(&b), 0, "BroadcastFact must be discriminant 0");

        // BlockAnnounce = 7
        let b = bincode::serialize(&MemChainMessage::BlockAnnounce(BlockHeader {
            height: 0, timestamp: 0,
            prev_block_hash: [0; 32], merkle_root: [0; 32],
            block_type: 0x01,
        })).unwrap();
        assert_eq!(disc(&b), 7, "BlockAnnounce must be discriminant 7");

        // BroadcastRecord = 8
        let b = bincode::serialize(&MemChainMessage::BroadcastRecord(
            MemoryRecord::new([0; 32], 0, MemoryLayer::Episode, vec![], "".into(), vec![], vec![]),
        )).unwrap();
        assert_eq!(disc(&b), 8, "BroadcastRecord must be discriminant 8");

        // ChatRelay = 11
        let kp = IdentityKeyPair::generate();
        let mut env = crate::protocol::chat::ChatEnvelope {
            message_id: [0x01; 16],
            sender: kp.public_key_bytes(),
            receiver: [0xBB; 32],
            timestamp: 1_700_000_000,
            ciphertext: b"test".to_vec(),
            nonce: [0x02; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        let data = env.sign_data();
        env.signature = kp.sign(&data);
        let b = bincode::serialize(&MemChainMessage::ChatRelay(env)).unwrap();
        assert_eq!(disc(&b), 11, "ChatRelay must be discriminant 11");

        // ChatPull = 12
        let b = bincode::serialize(&MemChainMessage::ChatPull {
            wallet: [0xAA; 32],
            after_timestamp: 0,
            cursor: [0u8; 16],
            limit: 50,
            request_timestamp: 1_700_000_000,
            signature: [0u8; 64],
        }).unwrap();
        assert_eq!(disc(&b), 12, "ChatPull must be discriminant 12");

        // ChatPullResponse = 13
        let b = bincode::serialize(&MemChainMessage::ChatPullResponse {
            envelopes: vec![], has_more: false,
        }).unwrap();
        assert_eq!(disc(&b), 13, "ChatPullResponse must be discriminant 13");

        // ChatAck = 14
        let b = bincode::serialize(&MemChainMessage::ChatAck {
            message_ids: vec![[0u8; 16]],
            wallet: [0xAA; 32],
            ack_timestamp: 1_700_000_000,
            signature: [0u8; 64],
        }).unwrap();
        assert_eq!(disc(&b), 14, "ChatAck must be discriminant 14");

        // ChatExpired = 15
        let b = bincode::serialize(&MemChainMessage::ChatExpired {
            message_ids: vec![[0u8; 16]],
            receiver: [0xCC; 32],
        }).unwrap();
        assert_eq!(disc(&b), 15, "ChatExpired must be discriminant 15");

        // DeviceRegister = 16
        let b = bincode::serialize(&MemChainMessage::DeviceRegister {
            device_id: [0x01u8; 16],
            device_name: "test-device".to_string(),
            wallet_pubkey: [0xAA; 32],
            timestamp: 1_700_000_000,
            signature: [0u8; 64],
        }).unwrap();
        assert_eq!(disc(&b), 16, "DeviceRegister must be discriminant 16");

        // WalletPresence = 17
        let b = bincode::serialize(&MemChainMessage::WalletPresence {
            wallet_pubkey: [0xBB; 32],
            timestamp: 1_700_000_000,
            signature: [0u8; 64],
        }).unwrap();
        assert_eq!(disc(&b), 17, "WalletPresence must be discriminant 17");
    }

    // ── Chat Relay variant roundtrip tests ───────────────────────────────

    #[test]
    fn test_chat_relay_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let mut env = crate::protocol::chat::ChatEnvelope {
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
            request_timestamp: 1_700_000_001,
            signature: [0xBB; 64],
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatPull {
                wallet, after_timestamp, cursor, limit,
                request_timestamp, signature,
            } => {
                assert_eq!(wallet, [0xAA; 32]);
                assert_eq!(after_timestamp, 1_700_000_000);
                assert_eq!(cursor, [0x01; 16]);
                assert_eq!(limit, 50);
                assert_eq!(request_timestamp, 1_700_000_001);
                assert_eq!(signature, [0xBB; 64]);
            }
            other => panic!("Expected ChatPull, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_pull_response_roundtrip() {
        let msg = MemChainMessage::ChatPullResponse { envelopes: vec![], has_more: true };
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
        let msg = MemChainMessage::ChatAck {
            message_ids: ids.clone(),
            wallet: [0xCC; 32],
            ack_timestamp: 1_700_000_000,
            signature: [0xDD; 64],
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatAck { message_ids, wallet, ack_timestamp, signature } => {
                assert_eq!(message_ids, ids);
                assert_eq!(wallet, [0xCC; 32]);
                assert_eq!(ack_timestamp, 1_700_000_000);
                assert_eq!(signature, [0xDD; 64]);
            }
            other => panic!("Expected ChatAck, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_expired_roundtrip() {
        let ids: Vec<[u8; 16]> = vec![[0xCC; 16]];
        let receiver = [0xDD; 32];
        let msg = MemChainMessage::ChatExpired { message_ids: ids.clone(), receiver };
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

    // ── v1.3.0-Sovereign: new variant roundtrip tests ────────────────────

    #[test]
    fn test_device_register_roundtrip_v130() {
        let kp = IdentityKeyPair::generate();
        let msg = MemChainMessage::DeviceRegister {
            device_id: [0xABu8; 16],
            device_name: "iPhone 14 Pro".to_string(),
            wallet_pubkey: kp.public_key_bytes(),
            timestamp: 1_700_000_000,
            signature: kp.sign(b"dummy_sign_data_for_roundtrip"),
        };
        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::DeviceRegister {
                device_id, device_name, wallet_pubkey, timestamp, signature,
            } => {
                assert_eq!(device_id, [0xABu8; 16]);
                assert_eq!(device_name, "iPhone 14 Pro");
                assert_eq!(wallet_pubkey, kp.public_key_bytes());
                assert_eq!(timestamp, 1_700_000_000);
                // Signature bytes must survive roundtrip intact
                assert_eq!(
                    signature,
                    kp.sign(b"dummy_sign_data_for_roundtrip")
                );
            }
            other => panic!("Expected DeviceRegister, got {:?}", other),
        }
    }

    #[test]
    fn test_wallet_presence_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let ts = 1_700_000_042u64;
        let sig = kp.sign(b"dummy_presence_sign_data");

        let msg = MemChainMessage::WalletPresence {
            wallet_pubkey: kp.public_key_bytes(),
            timestamp: ts,
            signature: sig,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);

        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::WalletPresence { wallet_pubkey, timestamp, signature } => {
                assert_eq!(wallet_pubkey, kp.public_key_bytes());
                assert_eq!(timestamp, ts);
                assert_eq!(signature, sig);
            }
            other => panic!("Expected WalletPresence, got {:?}", other),
        }
    }

    #[test]
    fn test_wallet_presence_size_reasonable() {
        let kp = IdentityKeyPair::generate();
        let msg = MemChainMessage::WalletPresence {
            wallet_pubkey: kp.public_key_bytes(),
            timestamp: 1_700_000_000,
            signature: [0u8; 64],
        };
        let encoded = encode_memchain(&msg).expect("encode");
        // 1 (magic) + 4 (discriminant) + 32 (pubkey) + 8 (ts) + 64 (sig) = 109 bytes
        assert!(
            encoded.len() < 200,
            "WalletPresence must be <200 bytes for UDP friendliness, got {}",
            encoded.len()
        );
    }

    /// Regression: v1.0.0 messages still decode correctly after v1.3.0 additions.
    #[test]
    fn test_backward_compat_v100_messages_still_decode() {
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
