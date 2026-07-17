// ============================================================================
// File: crates/aeronyx-core/src/protocol/memchain.rs
// ============================================================================
// Version: 2.8.12-VerifiedDeliveryAnchorWitness
//
// Modification Reason:
//   v1.3.0-Sovereign — Breaking protocol upgrade. Wallet identity is no longer
//   derived from the session key. Each sensitive message now carries an explicit
//   wallet_pubkey + timestamp + Ed25519 signature, allowing the server to verify
//   wallet ownership per-message without trusting the session binding.
//   v2.7.0-BlockSync — Appended signed commitment announcement and bounded
//   peer range request/response variants. Existing discriminants are unchanged.
//   v2.7.5-CheckpointProof — Appended signed chain-checkpoint reconciliation
//   variants. Existing discriminants remain unchanged.
//   v2.8.0-ChatPullV2 — Appended authenticated opaque-cursor request/response
//   variants. Existing v1 chat pull discriminants and wire bytes are unchanged.
//   v2.8.7-CheckpointCertificateExchange — Appended fixed-size, signed
//   certificate request/response variants for admitted node peers.
//   v2.8.10-CoordinatorLease — Appended signed short-lived coordinator lease
//   request/response variants. Existing discriminants remain unchanged.
//   v2.8.11-CoordinatorLeaseRelease — Appended signed graceful lease release
//   request/response variants without changing existing wire indices.
//   v2.8.12-VerifiedDeliveryAnchorWitness — Appended fixed-size signed node
//   delivery-anchor witness frames. They carry only generation and digest,
//   never relay routes, message identifiers, payloads, or delivery counts.
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
//   - Record block/checkpoint/certificate/lease and delivery-anchor witness
//     frames (19-22, 25-32) are node-peer control messages and MUST NOT be
//     accepted from ordinary client tunnels
//   - Commitment blocks contain opaque record IDs only; sealed memory payload
//     replication requires a separate owner-authorised protocol
//   - serde_bytes64 is defined in chat.rs; the [u8;64] signature fields here
//     use the same two-[u8;32] trick for bincode compatibility
//
// Last Modified:
//   v2.8.12-VerifiedDeliveryAnchorWitness — Appended variants 31-32 and
//                        canonical aggregate-only witness signing contracts
//   v2.8.11-CoordinatorLeaseRelease — Appended variants 29-30 for authenticated
//                        graceful handover without weakening crash fencing
//   v2.8.10-CoordinatorLease — Appended variants 27-28 and canonical lease
//                        signing contracts for cross-host writer fencing
//   v2.8.7-CheckpointCertificateExchange — Appended variants 25-26 and fixed
//                        certificate member/signing contracts
//   v2.8.0-ChatPullV2 — Appended variants 23-24 for monotonic mailbox paging
//   v1.2.0-MultiDevice — Added DeviceRegister (index 16)
//   v1.3.0-Sovereign   — Added wallet_pubkey/timestamp/signature to
//                        DeviceRegister, ChatPull, ChatAck; added WalletPresence (17)
//   v2.7.0-BlockSync   — Appended variants 18-20 and canonical request/response
//                        signing bytes for node-blind commitment sync
//   v2.7.5-CheckpointProof — Appended variants 21-22 and canonical signed
//                        checkpoint reconciliation bytes
// ============================================================================

use bincode::Options;
use serde::{de, Deserialize, Deserializer, Serialize};

#[allow(deprecated)]
use crate::ledger::Fact;
use crate::ledger::{BlockHeader, MemoryRecord, RecordCommitmentBlockV1, RecordCommitmentHeaderV1};
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

/// Maximum accepted opaque cursor bytes in a `ChatPullV2` request.
///
/// The current server token is smaller; this protocol ceiling leaves room for
/// versioned authenticated-encryption formats without permitting large
/// attacker-controlled allocations before wallet signature verification.
pub const MAX_CHAT_PULL_CURSOR_V2_BYTES: usize = 128;

/// Maximum independently signed witness frames carried by one certificate.
///
/// The fixed three-slot wire representation prevents attacker-controlled
/// collection lengths from allocating memory before peer authentication.
pub const MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1: usize = 3;

/// Smallest coordinator lease accepted by an audited witness.
pub const MIN_COORDINATOR_LEASE_TTL_SECS_V1: u32 = 60;

/// Largest coordinator lease accepted by an audited witness.
///
/// Short leases bound failover delay and the exposure window after a lost
/// renewal. Coordinators renew well before this deadline.
pub const MAX_COORDINATOR_LEASE_TTL_SECS_V1: u32 = 300;

/// Witness atomically advanced from an older generation to this request.
pub const VERIFIED_DELIVERY_WITNESS_ADVANCED_V1: u8 = 0;
/// Witness already retained the same generation and digest.
pub const VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1: u8 = 1;
/// Request generation was below the witness's durable high-water mark.
pub const VERIFIED_DELIVERY_WITNESS_STALE_V1: u8 = 2;
/// Request reused the witness generation with a different digest.
pub const VERIFIED_DELIVERY_WITNESS_CONFLICT_V1: u8 = 3;
/// Request skipped one or more generations after this witness was established.
///
/// A witness must not advance on this outcome. Requiring contiguous updates
/// prevents a later, still correctly signed host snapshot from silently
/// replacing an unwitnessed intermediate generation.
pub const VERIFIED_DELIVERY_WITNESS_GAP_V1: u8 = 4;

// ============================================
// Internal serde helper for [u8; 64]
// ============================================
// Identical to the one in chat.rs. Duplicated here to avoid a cross-module
// dependency for a single serde helper. Both produce the same wire bytes.
// DO NOT change the serialisation logic — it must stay wire-compatible.

mod serde_bytes64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

/// Fixed-size representation of one historical signed checkpoint response.
///
/// The certificate response supplies the common `chain_id`. Reconstructing a
/// [`MemChainMessage::RecordChainCheckpointResponseV1`] from these fields
/// yields the exact frame whose SHA-256 digest is committed by the certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordCheckpointCertificateMemberV1 {
    /// Original request identifier signed by the witness.
    pub request_id: [u8; 16],
    /// Witness Ed25519 public key.
    pub responder: [u8; 32],
    /// Original response timestamp signed by the witness.
    pub response_timestamp: u64,
    /// Certified shared-prefix height.
    pub checkpoint_height: u64,
    /// Witness block hash at `checkpoint_height`.
    pub checkpoint_hash: [u8; 32],
    /// Witness tip height when it signed the response.
    pub tip_height: u64,
    /// Witness tip hash when it signed the response.
    pub tip_hash: [u8; 32],
    /// Original witness signature over the v1 checkpoint response fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
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
/// | 18    | RecordBlockAnnounceV1| v2.7.0-BlockSync       |
/// | 19    | RecordBlockRangeRequestV1 | v2.7.0-BlockSync  |
/// | 20    | RecordBlockRangeResponseV1| v2.7.0-BlockSync  |
/// | 21    | RecordChainCheckpointRequestV1 | v2.7.5-CheckpointProof |
/// | 22    | RecordChainCheckpointResponseV1| v2.7.5-CheckpointProof |
/// | 23    | ChatPullV2          | v2.8.0-ChatPullV2       |
/// | 24    | ChatPullResponseV2  | v2.8.0-ChatPullV2       |
/// | 25    | RecordCheckpointCertificateRequestV1 | v2.8.7-CertificateExchange |
/// | 26    | RecordCheckpointCertificateResponseV1| v2.8.7-CertificateExchange |
/// | 27    | RecordCoordinatorLeaseRequestV1 | v2.8.10-CoordinatorLease |
/// | 28    | RecordCoordinatorLeaseResponseV1| v2.8.10-CoordinatorLease |
/// | 29    | RecordCoordinatorLeaseReleaseRequestV1 | v2.8.11-LeaseRelease |
/// | 30    | RecordCoordinatorLeaseReleaseResponseV1| v2.8.11-LeaseRelease |
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(deprecated)]
pub enum MemChainMessage {
    /// Broadcast a newly created Fact to peers (legacy).
    BroadcastFact(Fact),

    /// Request synchronisation: "send me all facts after this hash" (legacy).
    SyncRequest { last_known_hash: [u8; 32] },

    /// Response to a sync request with a batch of facts (legacy).
    SyncResponse { facts: Vec<Fact> },

    /// Query: ask a peer whether it has a specific fact.
    QueryRequest { fact_id: [u8; 32] },

    /// Query response.
    QueryResponse { fact: Option<Fact> },

    /// Lightweight ping to verify MemChain layer is alive.
    Ping { nonce: u64 },

    /// Response to a `Ping`.
    Pong { nonce: u64 },

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
    SyncRecordResponse { records: Vec<MemoryRecord> },

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

    // ── v2.7.0-BlockSync: Node-blind commitment chain (indices 18-20) ──
    /// [index 18] Announces a signed commitment header without memory payload.
    RecordBlockAnnounceV1 {
        /// Versioned node-blind commitment header.
        header: RecordCommitmentHeaderV1,
        /// Proposer signature over `header.hash()`.
        #[serde(with = "serde_bytes64")]
        proposer_signature: [u8; 64],
    },

    /// [index 19] Requests a bounded contiguous range of commitment blocks.
    ///
    /// Signature coverage is returned by
    /// [`record_block_range_request_signing_bytes`]. The requester must be a
    /// currently valid signed discovery peer; arbitrary clients cannot use
    /// this message to enumerate ledger commitments.
    RecordBlockRangeRequestV1 {
        /// Expected production/private chain identifier.
        chain_id: [u8; 32],
        /// First one-based height requested.
        from_height: u64,
        /// Requested block count; servers enforce their own lower cap.
        limit: u16,
        /// Per-request random identifier used to bind the response.
        request_id: [u8; 16],
        /// Requesting node's Ed25519 public key.
        requester: [u8; 32],
        /// Unix epoch seconds; peers reject stale requests.
        request_timestamp: u64,
        /// Requester signature over the canonical request fields.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 20] Returns a bounded, contiguous page of commitment blocks.
    ///
    /// Each block is independently proposer-signed. The response signature
    /// additionally binds page ordering, pagination state, and request id.
    RecordBlockRangeResponseV1 {
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Responding node's Ed25519 public key.
        responder: [u8; 32],
        /// Unix epoch seconds when this page was constructed.
        response_timestamp: u64,
        /// Contiguous commitment blocks beginning at the requested height.
        blocks: Vec<RecordCommitmentBlockV1>,
        /// Whether another page exists after this response.
        has_more: bool,
        /// Responder's current verified tip height.
        tip_height: u64,
        /// Responder's current verified tip hash, or zero at height zero.
        tip_hash: [u8; 32],
        /// Responder signature over canonical response fields and block hashes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 21] Requests a signed comparison checkpoint from a node peer.
    ///
    /// The requester attests to its current fully verified local tip. The
    /// responder returns its own tip plus the block hash at the shorter tip,
    /// allowing both sides to distinguish lag from an actual fork without
    /// exposing record commitments or memory payloads.
    RecordChainCheckpointRequestV1 {
        /// Expected production/private chain identifier.
        chain_id: [u8; 32],
        /// Requester's current fully verified local tip height.
        known_tip_height: u64,
        /// Requester's current tip hash, or genesis previous hash at height 0.
        known_tip_hash: [u8; 32],
        /// Per-request random identifier used to bind the response.
        request_id: [u8; 16],
        /// Requesting node's Ed25519 public key.
        requester: [u8; 32],
        /// Unix epoch seconds; peers reject stale requests.
        request_timestamp: u64,
        /// Requester signature over the canonical request fields.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 22] Returns a signed chain tip and shared-prefix checkpoint.
    RecordChainCheckpointResponseV1 {
        /// Chain identifier copied into the signed response.
        chain_id: [u8; 32],
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Responding node's Ed25519 public key.
        responder: [u8; 32],
        /// Unix epoch seconds when this checkpoint was constructed.
        response_timestamp: u64,
        /// Height selected as `min(requester_tip, responder_tip)`.
        checkpoint_height: u64,
        /// Responder's verified block hash at `checkpoint_height`.
        checkpoint_hash: [u8; 32],
        /// Responder's current fully verified tip height.
        tip_height: u64,
        /// Responder's current fully verified tip hash.
        tip_hash: [u8; 32],
        /// Responder signature over all canonical response fields.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.0-ChatPullV2: monotonic opaque mailbox cursor (indices 23-24) ──
    /// [index 23] Pulls one stable snapshot page using a server-issued cursor.
    ///
    /// An empty cursor begins a new snapshot. A non-empty cursor is an opaque,
    /// AEAD-protected token bound to this wallet and `after_timestamp`; clients
    /// must not parse or modify it. Existing clients continue using
    /// [`Self::ChatPull`] and remain wire-compatible.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-ChatPull-v2" || wallet(32) ||
    /// after_timestamp(8,LE) || cursor_len(2,LE) || cursor || limit(4,LE) ||
    /// request_timestamp(8,LE)
    ChatPullV2 {
        /// Wallet requesting its own offline mailbox.
        wallet: [u8; 32],
        /// Optional client timestamp floor; use zero to include all pending rows.
        after_timestamp: u64,
        /// Empty for a new snapshot, otherwise the exact token from the prior response.
        #[serde(deserialize_with = "deserialize_chat_pull_cursor_v2")]
        cursor: Vec<u8>,
        /// Maximum envelopes per page; the server clamps this to 1..=100.
        limit: u32,
        /// Unix epoch seconds when this request was constructed.
        request_timestamp: u64,
        /// Wallet signature over the canonical fields documented above.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 24] Stable snapshot page returned for [`Self::ChatPullV2`].
    ChatPullResponseV2 {
        /// Valid signed E2E envelopes in monotonic durable queue order.
        envelopes: Vec<ChatEnvelope>,
        /// Opaque cursor to echo in the next v2 request.
        next_cursor: Vec<u8>,
        /// Whether the current snapshot still contains another page.
        has_more: bool,
    },

    // ── v2.8.7: admitted checkpoint-certificate exchange (indices 25-26) ──
    /// [index 25] Requests a certificate for the requester's exact audited tip.
    ///
    /// This frame cannot request arbitrary history. The serving peer returns a
    /// bundle only when its latest retained certificate matches both height
    /// and hash, preventing certificate enumeration by admitted peers.
    RecordCheckpointCertificateRequestV1 {
        /// Expected production/private chain identifier.
        chain_id: [u8; 32],
        /// Requester's fully audited local tip height.
        known_tip_height: u64,
        /// Requester's fully audited local tip hash.
        known_tip_hash: [u8; 32],
        /// Per-request random identifier used for replay protection.
        request_id: [u8; 16],
        /// Requesting node's Ed25519 public key.
        requester: [u8; 32],
        /// Unix epoch seconds; peers reject stale requests.
        request_timestamp: u64,
        /// Requester signature over the canonical certificate request fields.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 26] Returns one bounded, independently verifiable certificate.
    ///
    /// Slots are packed from index zero and sorted by witness identity. The
    /// outer responder signature proves fresh transport; every member keeps
    /// its own historical witness signature and remains independently valid.
    RecordCheckpointCertificateResponseV1 {
        /// Production/private chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Serving node's Ed25519 public key.
        responder: [u8; 32],
        /// Unix epoch seconds when this bundle was served.
        response_timestamp: u64,
        /// Exact certified local tip height.
        checkpoint_height: u64,
        /// Exact certified local tip hash.
        checkpoint_hash: [u8; 32],
        /// Domain-separated digest of the certificate metadata and members.
        certificate_digest: [u8; 32],
        /// Operator threshold used when the certificate was created.
        required_signers: u8,
        /// Fixed bounded member slots; unused trailing entries are `None`.
        members:
            [Option<RecordCheckpointCertificateMemberV1>; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1],
        /// Serving node signature over the fresh response metadata.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.10: witness-backed coordinator lease (indices 27-28) ───────
    /// [index 27] Requests one short-lived exclusive coordinator lease.
    ///
    /// The long-term coordinator identity signs a random process instance id,
    /// exact audited tip, bounded TTL, and replay-resistant request id. A
    /// follower witness grants only its configured pinned coordinator and only
    /// when no different unexpired instance already owns the lease.
    RecordCoordinatorLeaseRequestV1 {
        /// Production/private chain identifier.
        chain_id: [u8; 32],
        /// Configured long-term coordinator Ed25519 identity.
        coordinator: [u8; 32],
        /// Random identifier generated once for this coordinator process.
        instance_id: [u8; 32],
        /// Coordinator's fully audited local tip height.
        known_tip_height: u64,
        /// Coordinator's fully audited local tip hash.
        known_tip_hash: [u8; 32],
        /// Requested bounded lease lifetime.
        requested_ttl_secs: u32,
        /// Per-request random identifier used for replay protection.
        request_id: [u8; 16],
        /// Unix epoch seconds; witnesses reject stale requests.
        request_timestamp: u64,
        /// Coordinator signature over every canonical request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 28] Returns one signed exclusive lease grant.
    ///
    /// A refusal is represented by a stable HTTP error and never discloses the
    /// current holder. Successful responses bind the exact request, witness,
    /// persisted lease epoch, expiry, and witness tip.
    RecordCoordinatorLeaseResponseV1 {
        /// Production/private chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Long-term coordinator identity copied from the request.
        coordinator: [u8; 32],
        /// Process instance identifier copied from the request.
        instance_id: [u8; 32],
        /// Granting witness Ed25519 identity.
        witness: [u8; 32],
        /// Unix epoch seconds when the grant was signed.
        response_timestamp: u64,
        /// Monotonic per-coordinator witness lease generation.
        lease_epoch: u64,
        /// Witness wall-clock expiry for durable restart recovery.
        lease_expires_at: u64,
        /// Witness's exact fully audited tip height at grant time.
        witness_tip_height: u64,
        /// Witness's exact fully audited tip hash at grant time.
        witness_tip_hash: [u8; 32],
        /// Witness signature over every canonical response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.11: authenticated graceful lease release (indices 29-30) ──
    /// [index 29] Releases only the calling process instance's current lease.
    ///
    /// Crash recovery does not send this frame and therefore remains bounded
    /// by the signed lease expiry. A copied identity cannot release a different
    /// active instance because the process id is covered by the signature and
    /// checked against the witness's durable row.
    RecordCoordinatorLeaseReleaseRequestV1 {
        /// Production/private chain identifier.
        chain_id: [u8; 32],
        /// Configured long-term coordinator Ed25519 identity.
        coordinator: [u8; 32],
        /// Exact process instance that previously acquired the lease.
        instance_id: [u8; 32],
        /// Per-request random identifier used for replay protection.
        request_id: [u8; 16],
        /// Unix epoch seconds; witnesses reject stale requests.
        request_timestamp: u64,
        /// Coordinator signature over every canonical release field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 30] Confirms one durable instance-matched release.
    RecordCoordinatorLeaseReleaseResponseV1 {
        /// Production/private chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Long-term coordinator identity copied from the request.
        coordinator: [u8; 32],
        /// Released process instance copied from the request.
        instance_id: [u8; 32],
        /// Releasing witness Ed25519 identity.
        witness: [u8; 32],
        /// Unix epoch seconds when the release was durably applied.
        released_at: u64,
        /// Lease generation released by this witness.
        lease_epoch: u64,
        /// Witness signature over every canonical response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.12: aggregate delivery-anchor witnessing (indices 31-32) ──
    /// [index 31] Asks an admitted node to witness one signed local cache generation.
    ///
    /// `anchor_digest` commits the requester's independently signed local
    /// aggregate anchor. The anchor and its delivery count never leave the
    /// requester; the witness receives only this fixed-size opaque digest.
    VerifiedDeliveryAnchorWitnessRequestV1 {
        /// Requesting node's Ed25519 identity.
        requester: [u8; 32],
        /// Positive monotonic local cache generation.
        generation: u64,
        /// Domain-separated digest of the requester's signed local anchor.
        anchor_digest: [u8; 32],
        /// Per-request random identifier used for replay protection.
        request_id: [u8; 16],
        /// Unix epoch seconds; witnesses reject stale requests.
        request_timestamp: u64,
        /// Requester signature over every canonical request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 32] Returns the witness's signed durable high-water decision.
    ///
    /// Outcomes are the fixed `VERIFIED_DELIVERY_WITNESS_*_V1` constants.
    /// Stale and conflicting requests still receive an authenticated response,
    /// allowing the requester to fail closed without trusting HTTP text.
    VerifiedDeliveryAnchorWitnessResponseV1 {
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Requesting node identity copied from the request.
        requester: [u8; 32],
        /// Generation copied from the request.
        requested_generation: u64,
        /// Anchor digest copied from the request.
        requested_anchor_digest: [u8; 32],
        /// Witness Ed25519 identity.
        witness: [u8; 32],
        /// Unix epoch seconds when the durable decision was signed.
        response_timestamp: u64,
        /// Witness's durable generation after evaluating the request.
        witness_generation: u64,
        /// Witness's durable digest after evaluating the request.
        witness_anchor_digest: [u8; 32],
        /// Fixed decision bucket; see `VERIFIED_DELIVERY_WITNESS_*_V1`.
        outcome: u8,
        /// Witness signature over every canonical response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
}

fn deserialize_chat_pull_cursor_v2<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedCursorVisitor;

    impl<'de> de::Visitor<'de> for BoundedCursorVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "an opaque ChatPullV2 cursor no larger than {} bytes",
                MAX_CHAT_PULL_CURSOR_V2_BYTES
            )
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            if sequence.size_hint().unwrap_or(0) > MAX_CHAT_PULL_CURSOR_V2_BYTES {
                return Err(de::Error::custom(
                    "ChatPullV2 cursor exceeds protocol limit",
                ));
            }
            let mut cursor = Vec::with_capacity(
                sequence
                    .size_hint()
                    .unwrap_or(0)
                    .min(MAX_CHAT_PULL_CURSOR_V2_BYTES),
            );
            while let Some(byte) = sequence.next_element::<u8>()? {
                if cursor.len() == MAX_CHAT_PULL_CURSOR_V2_BYTES {
                    return Err(de::Error::custom(
                        "ChatPullV2 cursor exceeds protocol limit",
                    ));
                }
                cursor.push(byte);
            }
            Ok(cursor)
        }
    }

    deserializer.deserialize_seq(BoundedCursorVisitor)
}

/// Canonical bytes signed by `RecordBlockRangeRequestV1.requester`.
#[must_use]
pub fn record_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(140);
    bytes.extend_from_slice(b"AeroNyx-RecordBlockRangeRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&from_height.to_le_bytes());
    bytes.extend_from_slice(&limit.to_le_bytes());
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordBlockRangeResponseV1.responder`.
#[must_use]
pub fn record_block_range_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    blocks: &[RecordCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Vec<u8> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    for block in blocks {
        hasher.update(block.hash());
    }
    let block_hashes_digest: [u8; 32] = hasher.finalize().into();

    let mut bytes = Vec::with_capacity(170);
    bytes.extend_from_slice(b"AeroNyx-RecordBlockRangeResponse-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&block_hashes_digest);
    bytes.push(u8::from(has_more));
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes
}

/// Canonical bytes signed by `RecordChainCheckpointRequestV1.requester`.
#[must_use]
pub fn record_chain_checkpoint_request_signing_bytes(
    chain_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(166);
    bytes.extend_from_slice(b"AeroNyx-RecordChainCheckpointRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordChainCheckpointResponseV1.responder`.
#[must_use]
pub fn record_chain_checkpoint_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(214);
    bytes.extend_from_slice(b"AeroNyx-RecordChainCheckpointResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&checkpoint_height.to_le_bytes());
    bytes.extend_from_slice(checkpoint_hash);
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes
}

/// Canonical bytes signed by
/// `RecordCheckpointCertificateRequestV1.requester`.
#[must_use]
pub fn record_checkpoint_certificate_request_signing_bytes(
    chain_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(178);
    bytes.extend_from_slice(b"AeroNyx-RecordCheckpointCertificateRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `RecordCheckpointCertificateResponseV1.responder`.
///
/// `certificate_digest` already commits the exact independently signed member
/// frames, so the transport signature binds that digest rather than repeating
/// historical witness material in the signing preimage.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_checkpoint_certificate_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    certificate_digest: &[u8; 32],
    required_signers: u8,
    signer_count: u8,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(240);
    bytes.extend_from_slice(b"AeroNyx-RecordCheckpointCertificateResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&checkpoint_height.to_le_bytes());
    bytes.extend_from_slice(checkpoint_hash);
    bytes.extend_from_slice(certificate_digest);
    bytes.push(required_signers);
    bytes.push(signer_count);
    bytes
}

/// Computes the canonical v1 checkpoint-certificate digest.
///
/// Each tuple is `(witness_identity, SHA-256(exact signed checkpoint frame))`.
/// Sorting makes the digest independent of network arrival order while still
/// requiring distinct identities at the verifier and storage layers.
#[must_use]
pub fn record_checkpoint_certificate_digest_v1(
    chain_id: &[u8; 32],
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    required_signers: usize,
    members: &[([u8; 32], [u8; 32])],
) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut ordered = members.to_vec();
    ordered.sort_unstable_by_key(|member| member.0);
    let mut digest = Sha256::new();
    digest.update(b"AERONYX_RECORD_CHECKPOINT_CERTIFICATE_V1");
    digest.update(chain_id);
    digest.update(checkpoint_height.to_be_bytes());
    digest.update(checkpoint_hash);
    digest.update((required_signers as u64).to_be_bytes());
    digest.update((ordered.len() as u64).to_be_bytes());
    for (responder, evidence_digest) in ordered {
        digest.update(responder);
        digest.update(evidence_digest);
    }
    digest.finalize().into()
}

/// Canonical bytes signed by `RecordCoordinatorLeaseRequestV1.coordinator`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_request_signing_bytes(
    chain_id: &[u8; 32],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    requested_ttl_secs: u32,
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(224);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(&requested_ttl_secs.to_le_bytes());
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    witness: &[u8; 32],
    response_timestamp: u64,
    lease_epoch: u64,
    lease_expires_at: u64,
    witness_tip_height: u64,
    witness_tip_hash: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(260);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&lease_epoch.to_le_bytes());
    bytes.extend_from_slice(&lease_expires_at.to_le_bytes());
    bytes.extend_from_slice(&witness_tip_height.to_le_bytes());
    bytes.extend_from_slice(witness_tip_hash);
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseReleaseRequestV1.coordinator`.
#[must_use]
pub fn record_coordinator_lease_release_request_signing_bytes(
    chain_id: &[u8; 32],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseReleaseRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseReleaseResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_release_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    witness: &[u8; 32],
    released_at: u64,
    lease_epoch: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(224);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseReleaseResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&released_at.to_le_bytes());
    bytes.extend_from_slice(&lease_epoch.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `VerifiedDeliveryAnchorWitnessRequestV1.requester`.
#[must_use]
pub fn verified_delivery_anchor_witness_request_signing_bytes(
    requester: &[u8; 32],
    generation: u64,
    anchor_digest: &[u8; 32],
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"AeroNyx-VerifiedDeliveryAnchorWitnessRequest-v1");
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(anchor_digest);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `VerifiedDeliveryAnchorWitnessResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn verified_delivery_anchor_witness_response_signing_bytes(
    request_id: &[u8; 16],
    requester: &[u8; 32],
    requested_generation: u64,
    requested_anchor_digest: &[u8; 32],
    witness: &[u8; 32],
    response_timestamp: u64,
    witness_generation: u64,
    witness_anchor_digest: &[u8; 32],
    outcome: u8,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(b"AeroNyx-VerifiedDeliveryAnchorWitnessResponse-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&requested_generation.to_le_bytes());
    bytes.extend_from_slice(requested_anchor_digest);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&witness_generation.to_le_bytes());
    bytes.extend_from_slice(witness_anchor_digest);
    bytes.push(outcome);
    bytes
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
    use crate::crypto::IdentityKeyPair;
    #[allow(deprecated)]
    use crate::ledger::{
        MemoryLayer, RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, BLOCK_TYPE_NORMAL,
        GENESIS_PREV_HASH,
    };
    use crate::protocol::chat::{encode_envelope, ChatContentType};

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
        assert!(
            encoded.len() < 200,
            "BlockAnnounce must be <200 bytes, got {}",
            encoded.len()
        );
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
            "aeronyx-memory-v1".into(),
            b"encrypted_content".to_vec(),
            vec![0.1, 0.2, 0.3],
        );
        let msg = MemChainMessage::BroadcastRecord(record.clone());
        let encoded = encode_memchain(&msg).expect("encode");
        assert_eq!(encoded[0], MEMCHAIN_MAGIC);
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::BroadcastRecord(r) => {
                assert_eq!(r.record_id, record.record_id);
                assert_eq!(r.layer, MemoryLayer::Episode);
                assert_eq!(r.source_ai, "aeronyx-memory-v1");
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
            MemChainMessage::SyncRecordRequest {
                owner,
                after_timestamp,
            } => {
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
        let b = bincode::serialize(&MemChainMessage::BroadcastFact(Fact::new(
            0,
            "s".into(),
            "p".into(),
            "o".into(),
        )))
        .unwrap();
        assert_eq!(disc(&b), 0, "BroadcastFact must be discriminant 0");

        // BlockAnnounce = 7
        let b = bincode::serialize(&MemChainMessage::BlockAnnounce(BlockHeader {
            height: 0,
            timestamp: 0,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            block_type: 0x01,
        }))
        .unwrap();
        assert_eq!(disc(&b), 7, "BlockAnnounce must be discriminant 7");

        // BroadcastRecord = 8
        let b = bincode::serialize(&MemChainMessage::BroadcastRecord(MemoryRecord::new(
            [0; 32],
            0,
            MemoryLayer::Episode,
            vec![],
            "".into(),
            vec![],
            vec![],
        )))
        .unwrap();
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
        })
        .unwrap();
        assert_eq!(disc(&b), 12, "ChatPull must be discriminant 12");

        // ChatPullResponse = 13
        let b = bincode::serialize(&MemChainMessage::ChatPullResponse {
            envelopes: vec![],
            has_more: false,
        })
        .unwrap();
        assert_eq!(disc(&b), 13, "ChatPullResponse must be discriminant 13");

        // ChatAck = 14
        let b = bincode::serialize(&MemChainMessage::ChatAck {
            message_ids: vec![[0u8; 16]],
            wallet: [0xAA; 32],
            ack_timestamp: 1_700_000_000,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(disc(&b), 14, "ChatAck must be discriminant 14");

        // ChatExpired = 15
        let b = bincode::serialize(&MemChainMessage::ChatExpired {
            message_ids: vec![[0u8; 16]],
            receiver: [0xCC; 32],
        })
        .unwrap();
        assert_eq!(disc(&b), 15, "ChatExpired must be discriminant 15");

        // DeviceRegister = 16
        let b = bincode::serialize(&MemChainMessage::DeviceRegister {
            device_id: [0x01u8; 16],
            device_name: "test-device".to_string(),
            wallet_pubkey: [0xAA; 32],
            timestamp: 1_700_000_000,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(disc(&b), 16, "DeviceRegister must be discriminant 16");

        // WalletPresence = 17
        let b = bincode::serialize(&MemChainMessage::WalletPresence {
            wallet_pubkey: [0xBB; 32],
            timestamp: 1_700_000_000,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(disc(&b), 17, "WalletPresence must be discriminant 17");

        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_000_001,
            GENESIS_PREV_HASH,
            vec![[0x11; 32]],
            &identity,
        );
        let b = bincode::serialize(&MemChainMessage::RecordBlockAnnounceV1 {
            header: block.header.clone(),
            proposer_signature: block.proposer_signature,
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            18,
            "RecordBlockAnnounceV1 must be discriminant 18"
        );

        let b = bincode::serialize(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: 16,
            request_id: [0x22; 16],
            requester: identity.public_key_bytes(),
            request_timestamp: 1_700_000_002,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            19,
            "RecordBlockRangeRequestV1 must be discriminant 19"
        );

        let b = bincode::serialize(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id: [0x22; 16],
            responder: identity.public_key_bytes(),
            response_timestamp: 1_700_000_003,
            blocks: vec![block],
            has_more: false,
            tip_height: 1,
            tip_hash: [0x33; 32],
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            20,
            "RecordBlockRangeResponseV1 must be discriminant 20"
        );

        let b = bincode::serialize(&MemChainMessage::RecordChainCheckpointRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            known_tip_height: 1,
            known_tip_hash: [0x33; 32],
            request_id: [0x44; 16],
            requester: identity.public_key_bytes(),
            request_timestamp: 1_700_000_004,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            21,
            "RecordChainCheckpointRequestV1 must be discriminant 21"
        );

        let b = bincode::serialize(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id: [0x44; 16],
            responder: identity.public_key_bytes(),
            response_timestamp: 1_700_000_005,
            checkpoint_height: 1,
            checkpoint_hash: [0x33; 32],
            tip_height: 1,
            tip_hash: [0x33; 32],
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            22,
            "RecordChainCheckpointResponseV1 must be discriminant 22"
        );

        let b = bincode::serialize(&MemChainMessage::ChatPullV2 {
            wallet: [0x55; 32],
            after_timestamp: 0,
            cursor: vec![],
            limit: 100,
            request_timestamp: 1_700_000_006,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(disc(&b), 23, "ChatPullV2 must be discriminant 23");

        let b = bincode::serialize(&MemChainMessage::ChatPullResponseV2 {
            envelopes: vec![],
            next_cursor: vec![0x77; 57],
            has_more: true,
        })
        .unwrap();
        assert_eq!(disc(&b), 24, "ChatPullResponseV2 must be discriminant 24");

        let b = bincode::serialize(&MemChainMessage::RecordCheckpointCertificateRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            known_tip_height: 1,
            known_tip_hash: [0x33; 32],
            request_id: [0x88; 16],
            requester: identity.public_key_bytes(),
            request_timestamp: 1_700_000_007,
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            25,
            "RecordCheckpointCertificateRequestV1 must be discriminant 25"
        );

        let member = RecordCheckpointCertificateMemberV1 {
            request_id: [0x44; 16],
            responder: identity.public_key_bytes(),
            response_timestamp: 1_700_000_005,
            checkpoint_height: 1,
            checkpoint_hash: [0x33; 32],
            tip_height: 1,
            tip_hash: [0x33; 32],
            signature: [0u8; 64],
        };
        let b = bincode::serialize(&MemChainMessage::RecordCheckpointCertificateResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id: [0x88; 16],
            responder: identity.public_key_bytes(),
            response_timestamp: 1_700_000_008,
            checkpoint_height: 1,
            checkpoint_hash: [0x33; 32],
            certificate_digest: [0x99; 32],
            required_signers: 2,
            members: [Some(member), Some(member), None],
            signature: [0u8; 64],
        })
        .unwrap();
        assert_eq!(
            disc(&b),
            26,
            "RecordCheckpointCertificateResponseV1 must be discriminant 26"
        );
    }

    #[test]
    fn test_chat_pull_v2_roundtrip_and_cursor_bound() {
        let request = MemChainMessage::ChatPullV2 {
            wallet: [0x51; 32],
            after_timestamp: 1_700_100_000,
            cursor: vec![0xA5; 57],
            limit: 25,
            request_timestamp: 1_700_100_001,
            signature: [0x7C; 64],
        };
        let encoded = encode_memchain(&request).expect("encode bounded ChatPullV2");
        let decoded = decode_memchain(&encoded[1..]).expect("decode bounded ChatPullV2");
        match decoded {
            MemChainMessage::ChatPullV2 {
                wallet,
                after_timestamp,
                cursor,
                limit,
                request_timestamp,
                signature,
            } => {
                assert_eq!(wallet, [0x51; 32]);
                assert_eq!(after_timestamp, 1_700_100_000);
                assert_eq!(cursor, vec![0xA5; 57]);
                assert_eq!(limit, 25);
                assert_eq!(request_timestamp, 1_700_100_001);
                assert_eq!(signature, [0x7C; 64]);
            }
            other => panic!("Expected ChatPullV2, got {:?}", other),
        }

        let oversized = MemChainMessage::ChatPullV2 {
            wallet: [0x51; 32],
            after_timestamp: 0,
            cursor: vec![0xA5; MAX_CHAT_PULL_CURSOR_V2_BYTES + 1],
            limit: 25,
            request_timestamp: 1_700_100_001,
            signature: [0x7C; 64],
        };
        let encoded = encode_memchain(&oversized).expect("serialization remains infallible");
        assert!(
            decode_memchain(&encoded[1..]).is_err(),
            "oversized ChatPullV2 cursor must be rejected during decode"
        );

        let response = MemChainMessage::ChatPullResponseV2 {
            envelopes: vec![],
            next_cursor: vec![0x42; 57],
            has_more: false,
        };
        let encoded = encode_memchain(&response).expect("encode ChatPullResponseV2");
        match decode_memchain(&encoded[1..]).expect("decode ChatPullResponseV2") {
            MemChainMessage::ChatPullResponseV2 {
                envelopes,
                next_cursor,
                has_more,
            } => {
                assert!(envelopes.is_empty());
                assert_eq!(next_cursor, vec![0x42; 57]);
                assert!(!has_more);
            }
            other => panic!("Expected ChatPullResponseV2, got {:?}", other),
        }
    }

    #[test]
    fn test_record_block_range_messages_roundtrip_and_signatures() {
        let requester = IdentityKeyPair::generate();
        let request_id = [0xA1; 16];
        let request_timestamp = 1_700_100_000;
        let request_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            8,
            &request_id,
            &requester.public_key_bytes(),
            request_timestamp,
        );
        let request = MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: 8,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp,
            signature: requester.sign(&request_bytes),
        };
        let encoded = encode_memchain(&request).expect("encode block range request");
        let decoded = decode_memchain(&encoded[1..]).expect("decode block range request");
        match decoded {
            MemChainMessage::RecordBlockRangeRequestV1 {
                chain_id,
                from_height,
                limit,
                request_id,
                requester: requester_key,
                request_timestamp,
                signature,
            } => {
                let signed = record_block_range_request_signing_bytes(
                    &chain_id,
                    from_height,
                    limit,
                    &request_id,
                    &requester_key,
                    request_timestamp,
                );
                requester
                    .verify(&signed, &signature)
                    .expect("request signature");
            }
            other => panic!("Expected RecordBlockRangeRequestV1, got {other:?}"),
        }

        let responder = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            request_timestamp,
            GENESIS_PREV_HASH,
            vec![[0x42; 32]],
            &responder,
        );
        let response_timestamp = request_timestamp + 1;
        let response_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder.public_key_bytes(),
            response_timestamp,
            std::slice::from_ref(&block),
            false,
            1,
            &block.hash(),
        );
        let response = MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder: responder.public_key_bytes(),
            response_timestamp,
            blocks: vec![block.clone()],
            has_more: false,
            tip_height: 1,
            tip_hash: block.hash(),
            signature: responder.sign(&response_bytes),
        };
        let encoded = encode_memchain(&response).expect("encode block range response");
        let decoded = decode_memchain(&encoded[1..]).expect("decode block range response");
        match decoded {
            MemChainMessage::RecordBlockRangeResponseV1 {
                request_id,
                responder: responder_key,
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
                signature,
            } => {
                let signed = record_block_range_response_signing_bytes(
                    &request_id,
                    &responder_key,
                    response_timestamp,
                    &blocks,
                    has_more,
                    tip_height,
                    &tip_hash,
                );
                responder
                    .verify(&signed, &signature)
                    .expect("response signature");
                assert_eq!(blocks, vec![block]);
            }
            other => panic!("Expected RecordBlockRangeResponseV1, got {other:?}"),
        }
    }

    #[test]
    fn test_record_chain_checkpoint_messages_roundtrip_and_signatures() {
        let requester = IdentityKeyPair::generate();
        let responder = IdentityKeyPair::generate();
        let request_id = [0xB1; 16];
        let known_tip_hash = [0xB2; 32];
        let request_timestamp = 1_700_200_000;
        let request_bytes = record_chain_checkpoint_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            7,
            &known_tip_hash,
            &request_id,
            &requester.public_key_bytes(),
            request_timestamp,
        );
        let request = MemChainMessage::RecordChainCheckpointRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            known_tip_height: 7,
            known_tip_hash,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp,
            signature: requester.sign(&request_bytes),
        };
        let encoded = encode_memchain(&request).expect("encode checkpoint request");
        let decoded = decode_memchain(&encoded[1..]).expect("decode checkpoint request");
        let MemChainMessage::RecordChainCheckpointRequestV1 {
            chain_id,
            known_tip_height,
            known_tip_hash,
            request_id,
            requester: requester_key,
            request_timestamp,
            signature,
        } = decoded
        else {
            panic!("expected checkpoint request");
        };
        let signed = record_chain_checkpoint_request_signing_bytes(
            &chain_id,
            known_tip_height,
            &known_tip_hash,
            &request_id,
            &requester_key,
            request_timestamp,
        );
        requester
            .verify(&signed, &signature)
            .expect("checkpoint request signature");

        let response_timestamp = request_timestamp + 1;
        let checkpoint_hash = [0xC1; 32];
        let tip_hash = [0xC2; 32];
        let response_bytes = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &request_id,
            &responder.public_key_bytes(),
            response_timestamp,
            7,
            &checkpoint_hash,
            9,
            &tip_hash,
        );
        let response = MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id,
            request_id,
            responder: responder.public_key_bytes(),
            response_timestamp,
            checkpoint_height: 7,
            checkpoint_hash,
            tip_height: 9,
            tip_hash,
            signature: responder.sign(&response_bytes),
        };
        let encoded = encode_memchain(&response).expect("encode checkpoint response");
        let decoded = decode_memchain(&encoded[1..]).expect("decode checkpoint response");
        let MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id,
            request_id,
            responder: responder_key,
            response_timestamp,
            checkpoint_height,
            checkpoint_hash,
            tip_height,
            tip_hash,
            signature,
        } = decoded
        else {
            panic!("expected checkpoint response");
        };
        let signed = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &request_id,
            &responder_key,
            response_timestamp,
            checkpoint_height,
            &checkpoint_hash,
            tip_height,
            &tip_hash,
        );
        responder
            .verify(&signed, &signature)
            .expect("checkpoint response signature");
    }

    #[test]
    fn test_checkpoint_certificate_messages_roundtrip_and_digest() {
        use sha2::{Digest, Sha256};

        let requester = IdentityKeyPair::generate();
        let serving_node = IdentityKeyPair::generate();
        let witnesses = [IdentityKeyPair::generate(), IdentityKeyPair::generate()];
        let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
        let height = 7;
        let hash = [0xD1; 32];
        let request_id = [0xD2; 16];
        let request_timestamp = 1_700_300_000;
        let request_bytes = record_checkpoint_certificate_request_signing_bytes(
            &chain_id,
            height,
            &hash,
            &request_id,
            &requester.public_key_bytes(),
            request_timestamp,
        );
        let request = MemChainMessage::RecordCheckpointCertificateRequestV1 {
            chain_id,
            known_tip_height: height,
            known_tip_hash: hash,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp,
            signature: requester.sign(&request_bytes),
        };
        let encoded = encode_memchain(&request).expect("encode certificate request");
        let decoded = decode_memchain(&encoded[1..]).expect("decode certificate request");
        let MemChainMessage::RecordCheckpointCertificateRequestV1 { signature, .. } = decoded
        else {
            panic!("expected certificate request");
        };
        requester
            .verify(&request_bytes, &signature)
            .expect("certificate request signature");

        let mut member_slots = [None; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1];
        let mut digest_members = Vec::new();
        for (index, witness) in witnesses.iter().enumerate() {
            let member_request_id = [0xE0 + index as u8; 16];
            let member_timestamp = request_timestamp.saturating_sub(60 - index as u64);
            let signing_bytes = record_chain_checkpoint_response_signing_bytes(
                &chain_id,
                &member_request_id,
                &witness.public_key_bytes(),
                member_timestamp,
                height,
                &hash,
                height,
                &hash,
            );
            let member = RecordCheckpointCertificateMemberV1 {
                request_id: member_request_id,
                responder: witness.public_key_bytes(),
                response_timestamp: member_timestamp,
                checkpoint_height: height,
                checkpoint_hash: hash,
                tip_height: height,
                tip_hash: hash,
                signature: witness.sign(&signing_bytes),
            };
            let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
                chain_id,
                request_id: member.request_id,
                responder: member.responder,
                response_timestamp: member.response_timestamp,
                checkpoint_height: member.checkpoint_height,
                checkpoint_hash: member.checkpoint_hash,
                tip_height: member.tip_height,
                tip_hash: member.tip_hash,
                signature: member.signature,
            })
            .expect("encode certificate member");
            digest_members.push((member.responder, Sha256::digest(frame).into()));
            member_slots[index] = Some(member);
        }
        member_slots
            .sort_unstable_by_key(|member| member.map_or([0xFF; 32], |present| present.responder));
        digest_members.sort_unstable_by_key(|member| member.0);
        let certificate_digest =
            record_checkpoint_certificate_digest_v1(&chain_id, height, &hash, 2, &digest_members);
        let response_timestamp = request_timestamp + 1;
        let response_bytes = record_checkpoint_certificate_response_signing_bytes(
            &chain_id,
            &request_id,
            &serving_node.public_key_bytes(),
            response_timestamp,
            height,
            &hash,
            &certificate_digest,
            2,
            2,
        );
        let response = MemChainMessage::RecordCheckpointCertificateResponseV1 {
            chain_id,
            request_id,
            responder: serving_node.public_key_bytes(),
            response_timestamp,
            checkpoint_height: height,
            checkpoint_hash: hash,
            certificate_digest,
            required_signers: 2,
            members: member_slots,
            signature: serving_node.sign(&response_bytes),
        };
        let encoded = encode_memchain(&response).expect("encode certificate response");
        let decoded = decode_memchain(&encoded[1..]).expect("decode certificate response");
        let MemChainMessage::RecordCheckpointCertificateResponseV1 {
            members, signature, ..
        } = decoded
        else {
            panic!("expected certificate response");
        };
        assert_eq!(members.iter().flatten().count(), 2);
        serving_node
            .verify(&response_bytes, &signature)
            .expect("certificate response signature");

        let tampered = record_checkpoint_certificate_digest_v1(
            &chain_id,
            height,
            &[0xD3; 32],
            2,
            &digest_members,
        );
        assert_ne!(tampered, certificate_digest);
    }

    #[test]
    fn test_coordinator_lease_messages_roundtrip_and_signatures() {
        let coordinator = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
        let instance_id = [0xA1; 32];
        let tip_height = 41;
        let tip_hash = [0xA2; 32];
        let request_id = [0xA3; 16];
        let request_timestamp = 1_700_400_000;
        let requested_ttl_secs = 120;
        let request_bytes = record_coordinator_lease_request_signing_bytes(
            &chain_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            tip_height,
            &tip_hash,
            requested_ttl_secs,
            &request_id,
            request_timestamp,
        );
        let request = MemChainMessage::RecordCoordinatorLeaseRequestV1 {
            chain_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            known_tip_height: tip_height,
            known_tip_hash: tip_hash,
            requested_ttl_secs,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&request_bytes),
        };
        let encoded = encode_memchain(&request).expect("encode lease request");
        let decoded = decode_memchain(&encoded[1..]).expect("decode lease request");
        let MemChainMessage::RecordCoordinatorLeaseRequestV1 { signature, .. } = decoded else {
            panic!("expected coordinator lease request");
        };
        coordinator
            .verify(&request_bytes, &signature)
            .expect("lease request signature");
        let tampered_request = record_coordinator_lease_request_signing_bytes(
            &chain_id,
            &coordinator.public_key_bytes(),
            &[0xFF; 32],
            tip_height,
            &tip_hash,
            requested_ttl_secs,
            &request_id,
            request_timestamp,
        );
        assert!(coordinator.verify(&tampered_request, &signature).is_err());

        let response_timestamp = request_timestamp + 1;
        let lease_epoch = 3;
        let lease_expires_at = response_timestamp + u64::from(requested_ttl_secs);
        let response_bytes = record_coordinator_lease_response_signing_bytes(
            &chain_id,
            &request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            response_timestamp,
            lease_epoch,
            lease_expires_at,
            tip_height,
            &tip_hash,
        );
        let response = MemChainMessage::RecordCoordinatorLeaseResponseV1 {
            chain_id,
            request_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            witness: witness.public_key_bytes(),
            response_timestamp,
            lease_epoch,
            lease_expires_at,
            witness_tip_height: tip_height,
            witness_tip_hash: tip_hash,
            signature: witness.sign(&response_bytes),
        };
        let encoded = encode_memchain(&response).expect("encode lease response");
        let decoded = decode_memchain(&encoded[1..]).expect("decode lease response");
        let MemChainMessage::RecordCoordinatorLeaseResponseV1 { signature, .. } = decoded else {
            panic!("expected coordinator lease response");
        };
        witness
            .verify(&response_bytes, &signature)
            .expect("lease response signature");

        let release_request_id = [0xA4; 16];
        let release_timestamp = response_timestamp + 2;
        let release_request_bytes = record_coordinator_lease_release_request_signing_bytes(
            &chain_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &release_request_id,
            release_timestamp,
        );
        let release_request = MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
            chain_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            request_id: release_request_id,
            request_timestamp: release_timestamp,
            signature: coordinator.sign(&release_request_bytes),
        };
        let encoded = encode_memchain(&release_request).expect("encode lease release request");
        assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 29);
        let decoded = decode_memchain(&encoded[1..]).expect("decode lease release request");
        let MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 { signature, .. } = decoded
        else {
            panic!("expected coordinator lease release request");
        };
        coordinator
            .verify(&release_request_bytes, &signature)
            .expect("lease release request signature");

        let release_response_bytes = record_coordinator_lease_release_response_signing_bytes(
            &chain_id,
            &release_request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            release_timestamp,
            lease_epoch,
        );
        let release_response = MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
            chain_id,
            request_id: release_request_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            witness: witness.public_key_bytes(),
            released_at: release_timestamp,
            lease_epoch,
            signature: witness.sign(&release_response_bytes),
        };
        let encoded = encode_memchain(&release_response).expect("encode lease release response");
        assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 30);
        let decoded = decode_memchain(&encoded[1..]).expect("decode lease release response");
        let MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 { signature, .. } = decoded
        else {
            panic!("expected coordinator lease release response");
        };
        witness
            .verify(&release_response_bytes, &signature)
            .expect("lease release response signature");
    }

    #[test]
    fn test_verified_delivery_anchor_witness_messages_roundtrip_and_signatures() {
        let requester = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let generation = 42;
        let anchor_digest = [0xA5; 32];
        let request_id = [0xB6; 16];
        let request_timestamp = 1_700_900_000;
        let request_signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
            &requester.public_key_bytes(),
            generation,
            &anchor_digest,
            &request_id,
            request_timestamp,
        );
        let request = MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            requester: requester.public_key_bytes(),
            generation,
            anchor_digest,
            request_id,
            request_timestamp,
            signature: requester.sign(&request_signing_bytes),
        };
        let encoded = encode_memchain(&request).expect("encode delivery witness request");
        assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 31);
        let decoded = decode_memchain(&encoded[1..]).expect("decode delivery witness request");
        let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            requester: requester_key,
            generation: decoded_generation,
            anchor_digest: decoded_digest,
            request_id: decoded_request_id,
            request_timestamp: decoded_timestamp,
            signature,
        } = decoded
        else {
            panic!("expected delivery witness request");
        };
        assert_eq!(decoded_generation, generation);
        assert_eq!(decoded_digest, anchor_digest);
        let decoded_signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
            &requester_key,
            decoded_generation,
            &decoded_digest,
            &decoded_request_id,
            decoded_timestamp,
        );
        requester
            .verify(&decoded_signing_bytes, &signature)
            .expect("delivery witness request signature");

        let response_timestamp = request_timestamp + 1;
        let response_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
            &request_id,
            &requester.public_key_bytes(),
            generation,
            &anchor_digest,
            &witness.public_key_bytes(),
            response_timestamp,
            generation,
            &anchor_digest,
            VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
        );
        let response = MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
            request_id,
            requester: requester.public_key_bytes(),
            requested_generation: generation,
            requested_anchor_digest: anchor_digest,
            witness: witness.public_key_bytes(),
            response_timestamp,
            witness_generation: generation,
            witness_anchor_digest: anchor_digest,
            outcome: VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
            signature: witness.sign(&response_signing_bytes),
        };
        let encoded = encode_memchain(&response).expect("encode delivery witness response");
        assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 32);
        let decoded = decode_memchain(&encoded[1..]).expect("decode delivery witness response");
        let MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
            request_id: decoded_request_id,
            requester: requester_key,
            requested_generation,
            requested_anchor_digest,
            witness: witness_key,
            response_timestamp: decoded_timestamp,
            witness_generation,
            witness_anchor_digest,
            outcome,
            signature,
        } = decoded
        else {
            panic!("expected delivery witness response");
        };
        let decoded_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
            &decoded_request_id,
            &requester_key,
            requested_generation,
            &requested_anchor_digest,
            &witness_key,
            decoded_timestamp,
            witness_generation,
            &witness_anchor_digest,
            outcome,
        );
        witness
            .verify(&decoded_signing_bytes, &signature)
            .expect("delivery witness response signature");

        assert_ne!(
            verified_delivery_anchor_witness_request_signing_bytes(
                &requester.public_key_bytes(),
                generation + 1,
                &anchor_digest,
                &request_id,
                request_timestamp,
            ),
            request_signing_bytes,
            "generation must be covered by the request signature"
        );
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
                assert!(
                    e.verify_signature().is_ok(),
                    "Signature must survive roundtrip"
                );
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
                wallet,
                after_timestamp,
                cursor,
                limit,
                request_timestamp,
                signature,
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
        let msg = MemChainMessage::ChatPullResponse {
            envelopes: vec![],
            has_more: true,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatPullResponse {
                envelopes,
                has_more,
            } => {
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
            MemChainMessage::ChatAck {
                message_ids,
                wallet,
                ack_timestamp,
                signature,
            } => {
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
        let msg = MemChainMessage::ChatExpired {
            message_ids: ids.clone(),
            receiver,
        };
        let encoded = encode_memchain(&msg).expect("encode");
        let decoded = decode_memchain(&encoded[1..]).expect("decode");
        match decoded {
            MemChainMessage::ChatExpired {
                message_ids,
                receiver: r,
            } => {
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
                device_id,
                device_name,
                wallet_pubkey,
                timestamp,
                signature,
            } => {
                assert_eq!(device_id, [0xABu8; 16]);
                assert_eq!(device_name, "iPhone 14 Pro");
                assert_eq!(wallet_pubkey, kp.public_key_bytes());
                assert_eq!(timestamp, 1_700_000_000);
                // Signature bytes must survive roundtrip intact
                assert_eq!(signature, kp.sign(b"dummy_sign_data_for_roundtrip"));
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
            MemChainMessage::WalletPresence {
                wallet_pubkey,
                timestamp,
                signature,
            } => {
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
            MemChainMessage::SyncRecordRequest {
                owner,
                after_timestamp,
            } => {
                assert_eq!(owner, [0xEE; 32]);
                assert_eq!(after_timestamp, 9_999_999);
            }
            other => panic!("Backward compat failed: got {:?}", other),
        }
    }
}
