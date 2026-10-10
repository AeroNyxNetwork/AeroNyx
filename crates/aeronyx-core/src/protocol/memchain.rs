// ============================================================================
// File: crates/aeronyx-core/src/protocol/memchain.rs
// ============================================================================
// Version: 2.8.26-AnonymousMailboxCanonicalOuter
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
//   v2.8.13-BoundedWireCodec — Unified outbound and inbound byte ceilings
//   without changing message discriminants, fields, or valid wire bytes.
//   v2.8.14-AuthorityHandoverExchange — Appended fixed-size authenticated
//   request/response frames carrying at most one dual-signed authority proof.
//   v2.8.15-AuthenticatedSessionClose — Appended a signed, fixed-size graceful
//   UDP session close request. Existing discriminants and wire bytes remain
//   unchanged.
//   v2.8.16-CustodyAuditWitnessNetwork — Appended fixed-size signed request and
//   response frames for independent custody-audit anchor witnesses. Existing
//   discriminants and wire bytes remain unchanged.
//   v2.8.17-VerifiedChatSubmit — Appended an opt-in authenticated client frame
//   that requires terminal-signed onion delivery evidence plus its response.
//   v2.8.18-VerifiedSubmitResultLabels — Centralized privacy-safe labels for
//   the fixed verified-submit result vocabulary. Existing wire bytes remain
//   unchanged.
//   v2.8.19-VerifiedSubmitOutcomeMapping — Centralized the boolean evidence
//   to result-code table used by verified chat submit responses. Existing wire
//   bytes remain unchanged.
//   v2.8.20-VerifiedSubmitResponseEvidence — Added a response constructor
//   that derives result code and receipt retention from closed evidence inputs.
//   Existing wire bytes remain unchanged.
//   v2.8.21-VerifiedSubmitRouteId — Centralized retry-stable verified submit
//   route id derivation. Existing wire bytes remain unchanged.
//   v2.8.22-VerifiedSubmitRejectedResponse — Added a canonical rejected
//   response constructor. Existing wire bytes remain unchanged.
//   v2.8.23-ChatSessionSenderBinding — Reused the core-owned envelope sender
//   binding contract during verified-submit construction. Existing wire bytes
//   remain unchanged.
//   v2.8.24-VerifiedSubmitRequestBinding — Added response verification against
//   the exact client request id and envelope. Existing wire bytes remain
//   unchanged.
//   v2.8.25-VerifiedSubmitResponseCorrelation — Added request correlation for
//   every result state and centralized terminal receipt verification internals.
//   Existing wire bytes remain unchanged.
//   v2.8.26-AnonymousMailboxCanonicalOuter — New anonymous-mailbox route
//   variants reject trailing or otherwise non-canonical outer bytes while
//   preserving the legacy trailing-byte policy for every older variant.
//   [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Added signature-only
//   verification for bounded, read-only replay of an existing durable result;
//   callers must still enforce freshness before admitting any new effect.
//
// Main Functionality:
//   Defines all application-layer messages that travel inside the existing
//   AeroNyx encrypted DataPacket, multiplexed by the 0xAE magic byte.
//
// Module Layout:
//   [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.
//   Every public item is re-exported from this module root, so all existing
//   `protocol::memchain::*` paths are unchanged.
//   - memchain/message.rs: the `MemChainMessage` wire enum and its bounded
//     ChatPullV2 cursor deserializer
//   - memchain/verified_submit.rs: opt-in verified chat submit request and
//     response, result codes and labels, outcome mapping, and route id derivation
//   - memchain/peer_control.rs: node-peer control-frame contracts (checkpoint
//     certificate members, delivery-witness outcome codes, canonical signing
//     bytes and digests)
//   - memchain/codec.rs: bounded `encode_memchain` / `decode_memchain` framing
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
//   - Record block/checkpoint/certificate/lease, delivery-anchor witness,
//     authority-handover, and custody-witness frames (19-22, 25-34, 36-37) are
//     node-peer control messages and MUST NOT be accepted from ordinary client
//     tunnels
//   - SessionCloseV1 (35) is a client-tunnel control frame. It must match both
//     the outer encrypted session ID and its handshake Ed25519 identity.
//   - ChatRelayVerifiedSubmitV1 (38) is opt-in. Legacy ChatRelay (11) remains
//     unchanged and must never be silently upgraded to stronger semantics.
//   - Commitment blocks contain opaque record IDs only; sealed memory payload
//     replication requires a separate owner-authorised protocol
//   - serde_bytes64 is defined in chat.rs; the [u8;64] signature fields here
//     use the same two-[u8;32] trick for bincode compatibility
//   - [BOUNDED-WIRE-CODEC 2026-07-23 by Codex] Encode and decode must use the
//     same 2 MiB payload ceiling. Keep the complete-slice length preflight in
//     the shared codec so ignored legacy trailing bytes cannot bypass the cap.
//
// Last Modified:
//   [ARCH-SPLIT 2026-10-10 by Claude] Split into focused child modules;
//                        bodies unchanged
//   v2.8.26-AnonymousMailboxCanonicalOuter — Added a variant-selective
//                        canonical decode gate for variants 40-41
//   v2.8.25-VerifiedSubmitResponseCorrelation — Added all-result correlation
//   v2.8.24-VerifiedSubmitRequestBinding — Added exact-request response verifier
//   v2.8.23-ChatSessionSenderBinding — Reused core envelope identity binding
//   v2.8.22-VerifiedSubmitRejectedResponse — Added rejected response builder
//   v2.8.21-VerifiedSubmitRouteId — Added core-owned route id derivation
//   v2.8.20-VerifiedSubmitResponseEvidence — Added fail-closed response builder
//   v2.8.19-VerifiedSubmitOutcomeMapping — Added core-owned outcome mapping
//   v2.8.18-VerifiedSubmitResultLabels — Added helper labels for result codes
//   v2.8.17-VerifiedChatSubmit — Appended variants 38-39 without changing any
//                        existing discriminant or legacy ChatRelay behavior
//   v2.8.16-CustodyAuditWitnessNetwork — Appended variants 36-37 and canonical
//                        exact-anchor/receipt request bindings
//   v2.8.15-AuthenticatedSessionClose — Appended variant 35 for bounded,
//                        signed graceful UDP tunnel cleanup
//   v2.8.14-AuthorityHandoverExchange — Appended variants 33-34 and canonical
//                        fixed-size authority-history signing contracts
//   v2.8.13-BoundedWireCodec — Symmetric frame limits, wire compatibility
//                        tests, and padded-input rejection
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

mod codec;
mod message;
mod peer_control;
mod verified_submit;

pub use codec::{decode_memchain, encode_memchain};
pub use message::MemChainMessage;
pub use peer_control::{
    custody_audit_anchor_witness_request_signing_bytes,
    custody_audit_anchor_witness_response_signing_bytes, record_block_range_request_signing_bytes,
    record_block_range_response_signing_bytes, record_chain_checkpoint_request_signing_bytes,
    record_chain_checkpoint_response_signing_bytes, record_checkpoint_certificate_digest_v1,
    record_checkpoint_certificate_request_signing_bytes,
    record_checkpoint_certificate_response_signing_bytes,
    record_coordinator_handover_request_signing_bytes,
    record_coordinator_handover_response_signing_bytes,
    record_coordinator_lease_release_request_signing_bytes,
    record_coordinator_lease_release_response_signing_bytes,
    record_coordinator_lease_request_signing_bytes,
    record_coordinator_lease_response_signing_bytes,
    verified_delivery_anchor_witness_request_signing_bytes,
    verified_delivery_anchor_witness_response_signing_bytes, RecordCheckpointCertificateMemberV1,
    VERIFIED_DELIVERY_WITNESS_ADVANCED_V1, VERIFIED_DELIVERY_WITNESS_CONFLICT_V1,
    VERIFIED_DELIVERY_WITNESS_GAP_V1, VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
    VERIFIED_DELIVERY_WITNESS_STALE_V1,
};
pub use verified_submit::{
    chat_verified_submit_result_for_outcomes, chat_verified_submit_result_label,
    chat_verified_submit_route_id, ChatRelayVerifiedSubmitRequestV1,
    ChatRelayVerifiedSubmitResponseV1, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
    CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1, CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1,
    CHAT_VERIFIED_SUBMIT_REJECTED_V1,
};

// ============================================
// Deserialisation size limits
// ============================================

/// Maximum accepted size for a single MemChain message payload (excluding magic byte).
const MAX_MEMCHAIN_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024; // 2 MB

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
