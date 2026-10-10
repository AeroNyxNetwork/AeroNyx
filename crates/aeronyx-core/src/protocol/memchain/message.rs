// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/message.rs
// ============================================
//! # Wire message enum
//!
//! Owns `MemChainMessage`, the positional bincode enum whose variant order is
//! the stable wire contract, and the bounded `ChatPullV2` cursor deserializer
//! that rejects oversized cursors before signature verification.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use serde::{de, Deserialize, Deserializer, Serialize};

#[allow(deprecated)]
use crate::ledger::Fact;
use crate::ledger::{
    BlockHeader, MemoryRecord, RecordCommitmentBlockV1, RecordCommitmentHeaderV1,
    RecordCoordinatorHandoverV1,
};
use crate::protocol::anonymous_mailbox::{
    AnonymousMailboxRouteRequestV1, AnonymousMailboxRouteResponseV1,
};
use crate::protocol::chat::{ChatEnvelope, CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1};

#[cfg(doc)]
use super::peer_control::record_block_range_request_signing_bytes;
use super::peer_control::RecordCheckpointCertificateMemberV1;
use super::verified_submit::{ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1};
#[cfg(doc)]
use super::MEMCHAIN_MAGIC;
use super::{serde_bytes64, MAX_CHAT_PULL_CURSOR_V2_BYTES, MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1};

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
/// | 31    | VerifiedDeliveryAnchorWitnessRequestV1 | v2.8.12-DeliveryWitness |
/// | 32    | VerifiedDeliveryAnchorWitnessResponseV1| v2.8.12-DeliveryWitness |
/// | 33    | RecordCoordinatorHandoverRequestV1 | v2.8.14-HandoverExchange |
/// | 34    | RecordCoordinatorHandoverResponseV1| v2.8.14-HandoverExchange |
/// | 35    | SessionCloseV1       | v2.8.15-AuthenticatedSessionClose |
/// | 36    | CustodyAuditAnchorWitnessRequestV1 | v2.8.16-CustodyWitnessNetwork |
/// | 37    | CustodyAuditAnchorWitnessResponseV1| v2.8.16-CustodyWitnessNetwork |
/// | 38    | ChatRelayVerifiedSubmitV1 | v2.8.17-VerifiedChatSubmit |
/// | 39    | ChatRelayVerifiedSubmitResponseV1 | v2.8.17-VerifiedChatSubmit |
/// | 40    | AnonymousMailboxRouteV1 | v2.8.18-AnonymousMailboxV1 |
/// | 41    | AnonymousMailboxRouteResponseV1 | v2.8.18-AnonymousMailboxV1 |
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

    // ── v2.8.14: bounded coordinator authority history (indices 33-34) ──
    /// [index 33] Requests the exact next durable coordinator handover proof.
    ///
    /// [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] One-proof paging is
    /// deliberate: a cold follower must interleave block-prefix verification
    /// with each exact-next authority transition instead of accepting a future
    /// unanchored history batch.
    RecordCoordinatorHandoverRequestV1 {
        /// Expected production/private chain identifier.
        chain_id: [u8; 32],
        /// Highest contiguous authority epoch already durable locally.
        after_authority_epoch: u64,
        /// Per-request random identifier used for replay protection.
        request_id: [u8; 16],
        /// Requesting node's Ed25519 identity.
        requester: [u8; 32],
        /// Unix epoch seconds; peers reject stale requests.
        request_timestamp: u64,
        /// Requester signature over every canonical request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 34] Returns at most the exact next dual-signed handover proof.
    RecordCoordinatorHandoverResponseV1 {
        /// Chain identifier copied into the signed response.
        chain_id: [u8; 32],
        /// Request identifier copied from the request.
        request_id: [u8; 16],
        /// Responding node's Ed25519 identity.
        responder: [u8; 32],
        /// Unix epoch seconds when this snapshot was signed.
        response_timestamp: u64,
        /// Exact `after_authority_epoch + 1` proof, or `None` at current head.
        handover: Option<RecordCoordinatorHandoverV1>,
        /// Latest contiguous authority epoch in the responder's audited store.
        latest_authority_epoch: u64,
        /// Responder signature over the request binding and proof digest.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.15: authenticated graceful tunnel close (index 35) ─────────
    /// [index 35] Requests immediate cleanup of the current encrypted session.
    ///
    /// [SESSION-TERMINATION 2026-08-15 by Codex] UDP has no connection-close
    /// handshake. This fixed-size request lets short-lived tools and clients
    /// release server capacity without waiting for the liveness timeout. The
    /// server accepts it only inside the exact encrypted session named here and
    /// only when the handshake identity verifies the signature.
    ///
    /// ## Signature Coverage
    /// domain="AeroNyx-SessionClose-v1" ||
    /// session_id(16) || close_timestamp(8,LE)
    SessionCloseV1 {
        /// Exact outer encrypted transport session being closed.
        session_id: [u8; 16],
        /// Unix epoch seconds; must be inside the standard authentication window.
        close_timestamp: u64,
        /// Signature by the Ed25519 identity authenticated in ClientHello.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.16: independent custody-audit witness exchange (36-37) ─────
    /// [index 36] Requests an independent signed decision for one exact anchor.
    ///
    /// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] The nested anchor proves
    /// producer authorship. This outer signature separately binds the anchor
    /// to a fresh request id so a captured anchor cannot be replayed as a new
    /// witness request by another peer.
    CustodyAuditAnchorWitnessRequestV1 {
        /// Per-request random identifier used by the witness replay guard.
        request_id: [u8; 16],
        /// Requesting producer identity; must equal `anchor.producer_node_id`.
        requester: [u8; 32],
        /// Unix epoch seconds; witnesses reject stale requests.
        request_timestamp: u64,
        /// Exact producer-signed aggregate custody anchor being witnessed.
        anchor: CustodyAuditAnchorV1,
        /// Requester signature over the canonical anchor digest and request.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    /// [index 37] Returns one portable receipt and request-bound signature.
    ///
    /// The nested receipt is independently portable evidence. The outer
    /// signature prevents a valid receipt from being substituted into a
    /// different in-flight request. Neither layer carries archive contents.
    CustodyAuditAnchorWitnessResponseV1 {
        /// Exact request identifier copied from the admitted request.
        request_id: [u8; 16],
        /// Producer identity copied from the admitted request.
        requester: [u8; 32],
        /// Responding witness identity and receipt signer.
        witness: [u8; 32],
        /// Unix epoch seconds; must equal `receipt.observed_at`.
        response_timestamp: u64,
        /// Signed monotonic accept/stale/conflict/gap evidence.
        receipt: CustodyAuditWitnessReceiptV1,
        /// Witness signature binding the portable receipt to this request.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },

    // ── v2.8.17: opt-in verified client onion delivery (38-39) ─────────
    /// [index 38] Requests terminal-verifiable onion delivery.
    ChatRelayVerifiedSubmitV1(ChatRelayVerifiedSubmitRequestV1),

    /// [index 39] Returns entry custody and exact terminal receipt evidence.
    ChatRelayVerifiedSubmitResponseV1(ChatRelayVerifiedSubmitResponseV1),

    // [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] Additive one-target route
    // carriers. The opaque terminal bytes contain no relay-visible chat peers.
    /// [index 40] Carries one source-signed sealed anonymous mailbox request.
    AnonymousMailboxRouteV1(AnonymousMailboxRouteRequestV1),

    /// [index 41] Carries one target-signed sealed anonymous mailbox response.
    AnonymousMailboxRouteResponseV1(AnonymousMailboxRouteResponseV1),
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

#[cfg(test)]
mod tests;
