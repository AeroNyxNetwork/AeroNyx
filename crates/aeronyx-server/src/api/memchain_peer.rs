// ============================================
// File: crates/aeronyx-server/src/api/memchain_peer.rs
// ============================================
//! # MemChain Node Peer API — Commitment Block Synchronisation
//!
//! ## Creation Reason
//! Block Sync v1 needs a node-to-node transport that is separate from the VPN
//! client tunnel and from public discovery metadata. Reusing either surface
//! would let ordinary clients enumerate commitments or couple ledger catch-up
//! to unrelated descriptor gossip.
//!
//! ## Main Functionality
//! - `POST /api/memchain/peer/block-range`
//! - `POST /api/memchain/peer/block-announce`
//! - `POST /api/memchain/peer/checkpoint`
//! - `POST /api/memchain/peer/checkpoint-certificate`
//! - `POST /api/memchain/peer/coordinator-lease`
//! - `POST /api/memchain/peer/coordinator-lease/release`
//! - `POST /api/memchain/peer/coordinator-handover`
//! - `POST /api/memchain/peer/custody-audit-anchor-witness`
//! - `POST /api/discovery/peer/verified-delivery-anchor-witness`
//! - Bincode `MemChainMessage` request/response with the existing magic byte.
//! - Signed discovery-peer admission, timestamp freshness, stateful-request
//!   replay protection, shared per-peer rate limiting, and bounded pagination.
//! - Monotonic per-peer abuse windows remain stable across NTP and wall-clock
//!   corrections while signed-frame freshness continues to use Unix time.
//! - Idempotent signed tip hints may be retried within that shared rate limit;
//!   they only coalesce a follower wake-up and never mutate canonical state.
//! - Coordinator delivery uses a three-peer, three-attempt in-memory retry
//!   queue with bounded exponential backoff for transport and transient HTTP
//!   failures; every retry revalidates the latest signed peer endpoint.
//! - Response signing that binds request id, block order, pagination, and tip.
//! - Default-off follower pull from one configured coordinator identity.
//! - Best-effort signed tip announcements that only wake the existing verified
//!   follower pull; an announcement can never append or select a chain.
//! - Whole-page signature, proposer, continuity, fork, and rollback validation
//!   followed by one atomic SQLite page append.
//! - Signed tip/checkpoint comparison that distinguishes lag from a fork.
//! - Durable bounded storage of the exact verified checkpoint response before
//!   follower convergence can be reported.
//! - Bounded coordinator witness rounds that collect signed peer checkpoints
//!   as evidence without treating peer count as consensus or fork choice.
//! - Operator-pinned divergent checkpoints become durable storage incidents;
//!   the verified relation still reaches startup/runtime policy unchanged.
//! - Strict startup witness reconciliation may contact an operator-pinned
//!   identity through an authentic expired cache descriptor, preventing an
//!   outage/TTL boot loop while retaining signed-response verification.
//! - Coordinator startup republishes its current signed discovery descriptor
//!   to operator-pinned witnesses before strict checkpoint and lease gates,
//!   allowing endpoint rotation to recover older compatible witness nodes.
//! - Direction-isolated checkpoint telemetry: serving a requester updates only
//!   service counters and cannot manufacture local convergence or divergence.
//! - Audit-gated block pages assembled from one SQLite snapshot and
//!   canonically reverified before the node signs a response.
//! - Fixed-size certificate exchange between admitted peers. Imported members
//!   must still belong to the receiver's operator-pinned witness set.
//! - Followers refresh current-tip checkpoint certificates only after signed
//!   chain convergence; mixed-version absence never rolls back a verified tip.
//! - [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Followers may recover
//!   coordinator-signed pages from bounded operator-pinned carriers after a
//!   classified coordinator availability failure. The carrier signs only the
//!   transport envelope; every block must still be signed by the configured
//!   coordinator and terminal recovery requires the local witness threshold.
//! - [BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex] Repeated availability
//!   failures open a short process-only cooldown for the corresponding fixed
//!   operator-pin slot. Half-open recovery probes remain coordinator-first,
//!   bounded, and fail closed on every observed security error.
//! - [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Local status and
//!   heartbeat report only aggregate cooling-slot, skipped-attempt, and
//!   half-open-probe counts; circuit slots and source details remain private.
//! - [TYPED-CARRIER-CIRCUIT 2026-07-29 by Codex] Authority-proof, block-page,
//!   and certificate recovery share one zero-cost generic circuit while domain
//!   markers prevent any path from receiving another's mutable state.
//! - [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Follower and
//!   coordinator certificate recovery share one fail-closed carrier primitive:
//!   only availability faults advance, every verified response stops the
//!   round, and security faults cannot be hidden by a later source.
//! - Followers report identity-blind current-policy readiness only after exact
//!   local tip, pin-set, threshold, and durable-certificate validation.
//! - Last-hop public-IP validation on every outbound commitment request so a
//!   rotated signed descriptor cannot redirect the node into private services.
//! - Default-off, signed short-lived coordinator leases persisted by followers
//!   for cross-host duplicate-writer fencing.
//! - Default-off external witness transport for signed aggregate-only verified-
//!   delivery cache anchors, with contiguous generation enforcement.
//! - [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] Explicit custody
//!   witness collection runs the hard-bounded pin set concurrently so one
//!   unavailable witness cannot multiply the maintenance-lock hold time.
//! - [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] Fixed one-proof
//!   authority-history exchange interleaved with exact-prefix block catch-up.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` on the public node peer listener and local operator
//!   listener when Local-mode `MemoryStorage` is available.
//! - Reads peer pages through
//!   `MemoryStorage::get_verified_record_commitment_block_page`.
//! - Uses `PeerStore::get_valid` as the node admission boundary.
//! - Uses canonical signing bytes from `aeronyx_core::protocol::memchain`.
//! - `server.rs` runs the optional low-frequency follower and coordinator
//!   witness schedulers.
//! - Coordinator startup may restrict reconciliation to explicit operator-
//!   pinned witness identities before opening transport/API listeners.
//!
//! ## Privacy Invariant
//! This API returns only signed commitment blocks. It never returns memory
//! records, ciphertext, owners, tags, embeddings, client IPs, destinations,
//! routes, endpoints, or social graph metadata.
//! The verified-delivery witness endpoint stores only a requester node id,
//! generation, opaque anchor digest, and observation time. Delivery counts,
//! delivery timestamps, routes, message ids, payloads, and client metadata
//! never cross the node-to-node witness boundary.
//! The custody witness endpoint stores only a producer node id, monotonic
//! checkpoint generation, canonical aggregate-anchor digest, and observation
//! time. It never receives archive content, record ids, owners, users, routes,
//! messages, endpoints, destinations, or plaintext.
//!
//! ## Important Note for Next Developer
//! - Do not mount this handler without `PeerStore` admission.
//! - Do not add a JSON/debug response containing raw commitments.
//! - Do not increase body/page limits without memory and abuse testing.
//! - Sealed payload replication requires a separate owner-authorised protocol.
//! - Never fall back from the pinned coordinator to an arbitrary discovered
//!   peer. A block carrier must be an explicit witness pin, is bounded to the
//!   existing witness fan-out limit, and gains no proposer, checkpoint,
//!   consensus, finality, or fork-choice authority.
//! - A block announcement is an untrusted scheduling hint even after its
//!   signature is verified. It must never bypass page/checkpoint validation,
//!   failure backoff, rollback protection, or the pinned coordinator policy.
//! - Do not put the deterministic block-header hash into the stateful replay
//!   cache. A follower must be able to retry the exact signed hint after a
//!   transient pull failure; rate limiting and the capacity-one notifier bound
//!   that idempotent wake-up without weakening other anti-replay checks.
//! - Never retry permanent `4xx` or protocol-incompatible receipts. Retry work
//!   must remain process-local, bounded to pinned peers, cancellable on task
//!   shutdown, and unable to delay or roll back canonical block production.
//! - Checkpoint proof establishes what a peer signed; it is not a majority,
//!   finality, leader-election, or longest-chain consensus rule.
//! - The latest bounded round summary is aggregate operational evidence only;
//!   its counts must never become voting weight or a fork-choice input.
//! - Coordinator witness failures or divergence evidence must never mutate the
//!   canonical chain; they are operator evidence until consensus is designed.
//! - Only explicit operator pins may turn signed checkpoint evidence into a
//!   startup gate. Permissionless discovery peers remain evidence-only.
//! - Expired descriptors are never live peers. Their endpoints may be used
//!   only as transport hints for operator-pinned witness reconciliation, where
//!   the response is independently bound to the pinned Ed25519 identity.
//! - Descriptor preflight sends only this node's already-public signed
//!   descriptor to exact operator pins. A successful POST is not authority:
//!   the subsequent signed checkpoint and all-witness lease gates remain the
//!   only paths that permit coordinator production.
//! - A trusted divergent-prefix incident must not be converted into a generic
//!   transport failure: callers need the verified divergence to fail closed.
//! - Never derive outbound checkpoint state from an inbound request. The peer
//!   controls its requested height/hash, so those values are not local evidence.
//! - Never sign a range assembled from separate block/tip reads or from a
//!   missing/stale process audit baseline.
//! - Imported certificates are post-startup evidence only. Never let a replayed
//!   bundle satisfy the live startup witness threshold.
//! - Certificate-policy readiness is a local operations signal, not consensus
//!   or finality. `ready` requires the current audited tip to satisfy the
//!   current local pin set and threshold; transport success alone is not enough.
//! - Revalidate the resolved signed endpoint inside every pull helper. Candidate
//!   filtering alone is vulnerable to concurrent descriptor replacement.
//! - Coordinator leases require every configured witness grant. Do not describe
//!   them as permissionless consensus, Byzantine finality, or fork choice.
//! - A delivery witness accepts any positive generation only for first contact;
//!   every later advance must be exactly one generation. Never auto-heal a gap
//!   by overwriting the witness high-water mark.
//! - The localhost endpoint override below is compiled only for crate tests.
//!   Never expose it in production or bypass final-hop SSRF validation.
//! - A handover response is transport only. Accept authority exclusively by
//!   persisting the exact-next dual-signed proof against the configured root
//!   and audited predecessor; never trust responder identity as authority.
//!
//! ## Last Modified
//! v2.8.65-CustodyWitnessConcurrentRound - Bounded explicit custody witness
//! transport to one concurrent request per distinct configured pin while
//! preserving durable-before-counting and adverse-evidence fail-closed rules.
//! v2.8.64-CustodyWitnessReceiptVault - Added fail-closed producer receipt
//! persistence and restart-safe exact-anchor policy reconstruction.
//! v2.8.63-CustodyWitnessTransport - Added explicit pinned witness transport
//! and adverse-evidence-aware bounded quorum rounds without a scheduler.
//! v2.8.62-CustodyWitnessPlanner - Added local aggregate-only eligibility
//! planning and authenticated witness admission before private pin checks.
//! v2.8.61-CustodyWitnessNetwork - Added independently pinned canonical
//! custody-anchor admission and portable positive/adverse receipt responses.
//! v2.8.60-AuthorityHandoverCarrier - Recovered exact dual-signed authority
//! proofs through bounded operator-pinned transport carriers.
//! v2.8.59-AuthorityHandoverExchange - Added bounded authenticated next-proof
//! transport and height-aware follower authority synchronization.
//! v2.8.58-MonotonicPeerRateLimit - Detached node-to-node abuse windows from
//!   wall-clock minutes and added deterministic rollback/boundary coverage.
//! v2.8.57-CertificatePersistenceTruth - Report verified-but-unpersisted follower evidence honestly.
//! v2.8.54-CertificateCarrierRecovery - Unified fail-closed certificate carrier recovery.
//! v2.8.53-TypedCarrierCircuit - Isolated block and certificate circuit domains.
//! v2.8.52-BlockCarrierCircuitTelemetry - Added source-blind circuit health aggregates.
//! v2.8.51-BlockCarrierCircuitBreaker - Added anonymous cross-round carrier cooldown and half-open recovery.
//! v2.8.50-CertifiedBlockCarrier - Recovered coordinator-signed pages through bounded pinned carriers.
//! v2.8.49-FollowerCertificateTipBinding - Bound every applicable policy outcome to its audited tip.
//! v2.8.48-FollowerCertificateReadiness - Reported exact current-policy readiness without witness identities.
//! v2.8.45-FollowerCertificateTelemetry - Reported source-blind certificate recovery outcomes.
//! v2.8.32-FollowerCertificateCarrier - Recover audited certificates from pinned witness carriers.
//! v2.8.31-FollowerCertificateSync - Refresh audited checkpoint certificates after follower convergence.
//! v2.8.30-WitnessDescriptorPreflight - Republish the current coordinator descriptor before strict startup gates.
//! v2.8.29-VerifiedDeliveryWitnessAdmission - Require bilateral requester pinning before witness writes.
//! v2.8.28-VerifiedDeliveryAnchorWitness - Added authenticated contiguous external cache witnesses.
//! v2.8.19-TipSupersessionIntegration - Added a test-only real HTTP delivery seam.
//! v2.8.17-TipRetryQueue - Added bounded transient-failure delivery retries.
//! v2.8.16-IdempotentTipRetry - Allowed bounded retry of signed follower wake-ups.
//! v2.8.15-AnnouncementReceipts - Classified exact accepted, stale, and failed receipts.
//! v2.8.14-SyncObservability - Added privacy-safe authenticated announcement dispositions.
//! v2.8.13-EventDrivenFollower - Added authenticated coalesced tip wake-ups.
//! v2.8.11-CoordinatorLeaseRelease - Added authenticated graceful lease handover.
//! v2.8.10-CoordinatorLease - Added durable follower lease grants and verified client.
//! v2.8.8-EndpointSSRFGuard - Enforced final-hop public endpoint validation.
//! v2.8.7-CertificateExchange - Added admitted fixed-size certificate exchange.
//! v2.8.6-CheckpointCertificate - Require distinct pinned witnesses for certificate rounds.
//! v2.8.5-TrustedDivergenceHalt - Preserve verified divergence after sticky incident creation.
//! v2.8.3-WitnessDivergence - Exposed crate-local reconciliation for startup tests.
//! v2.8.4-WitnessEquivocation - Retain and reject conflicting pinned-witness claims.
//! v2.8.2-AdversarialFollower - Added signed malicious-page regression coverage.
//! v2.7.18-VerifiedRangeSnapshot - Sign only snapshot-consistent audited pages.
//! v2.7.17-AtomicBlockPage - Commit each verified follower page atomically.
//! v2.7.15-ExternalWitnessGuard - Added identity-pinned reconciliation.
//! v2.7.5-CheckpointProof - Signed cross-node checkpoint reconciliation.
//! v2.7.6-EvidenceVault - Fail-closed durable verified checkpoint evidence.
//! v2.7.8-CoordinatorWitness - Bounded non-consensus witness reconciliation.
//! v2.7.10-CheckpointDirectionIsolation - Isolated inbound service telemetry.
//! v2.7.12-WitnessRoundEvidence - Persist aggregate bounded-round runtime state.
//! v2.7.1-BlockFollower - Pinned coordinator pull and fail-closed page verification.
//! v2.7.0-BlockSync - Initial signed node-blind block range protocol.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use futures::StreamExt;
use rand::RngCore;
use reqwest::Url;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{
    RecordCommitmentBlockV1, RecordCoordinatorHandoverV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
    GENESIS_PREV_HASH, MAX_RECORD_COMMITMENTS_PER_BLOCK, RECORD_COMMITMENT_BLOCK_VERSION_V1,
};
use aeronyx_core::protocol::chat::{
    custody_audit_anchor_frame_sha256, CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1,
    CUSTODY_AUDIT_WITNESS_ADVANCED_V1, CUSTODY_AUDIT_WITNESS_CONFLICT_V1,
    CUSTODY_AUDIT_WITNESS_GAP_V1, CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
    CUSTODY_AUDIT_WITNESS_STALE_V1,
};
use aeronyx_core::protocol::memchain::{
    custody_audit_anchor_witness_request_signing_bytes, decode_memchain, encode_memchain,
    record_block_range_request_signing_bytes, record_block_range_response_signing_bytes,
    record_chain_checkpoint_request_signing_bytes, record_chain_checkpoint_response_signing_bytes,
    record_checkpoint_certificate_digest_v1, record_checkpoint_certificate_request_signing_bytes,
    record_checkpoint_certificate_response_signing_bytes,
    record_coordinator_handover_request_signing_bytes,
    record_coordinator_handover_response_signing_bytes,
    record_coordinator_lease_release_request_signing_bytes,
    record_coordinator_lease_release_response_signing_bytes,
    record_coordinator_lease_request_signing_bytes,
    record_coordinator_lease_response_signing_bytes,
    verified_delivery_anchor_witness_request_signing_bytes,
    verified_delivery_anchor_witness_response_signing_bytes, MemChainMessage,
    RecordCheckpointCertificateMemberV1, MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1,
    MAX_COORDINATOR_LEASE_TTL_SECS_V1, MEMCHAIN_MAGIC, MIN_COORDINATOR_LEASE_TTL_SECS_V1,
    VERIFIED_DELIVERY_WITNESS_ADVANCED_V1, VERIFIED_DELIVERY_WITNESS_CONFLICT_V1,
    VERIFIED_DELIVERY_WITNESS_GAP_V1, VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
    VERIFIED_DELIVERY_WITNESS_STALE_V1,
};
use aeronyx_core::protocol::{NodeCapability, NodeDiscoveryMessage, SignedNodeDescriptor};
use sha2::{Digest, Sha256};

use super::{
    canonical_peer_http_url, peer_endpoint_is_public_ip, read_bounded_http_response,
    PeerEndpointUrlError,
};
use crate::api::discovery::GossipResponse;
use crate::services::memchain::storage_ops::{
    RecordCommitmentAuthorityState, RecordCommitmentCheckpointEvidencePersistOutcome,
    RecordCoordinatorHandoverPersistOutcome,
};
use crate::services::memchain::{
    MemoryStorage, RecordCommitmentAnnouncementDisposition,
    RecordCommitmentAuthoritySyncDisposition, RecordCommitmentBlockPagePullDisposition,
    RecordCommitmentCertificatePolicyReadiness, RecordCommitmentCertificateSyncDisposition,
};
use crate::services::PeerStore;

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod block_page;
mod carrier_telemetry;
mod certificate_sync;
mod checkpoint_sync;
mod coordinator_lease;
mod custody_witness;
mod delivery_witness;
mod descriptor_publish;
mod handover;
mod peer_http;
mod tip_announce;
mod witness_reconcile;

use block_page::block_range_handler;
use block_page::commitment_block_source_failure_class;
use block_page::eligible_pinned_commitment_carriers;
pub use block_page::pull_record_commitment_page;
use block_page::pull_record_commitment_page_from_source_with_endpoint_policy;
pub(crate) use block_page::pull_record_commitment_page_with_carrier_cursor;
use block_page::pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy;
pub use block_page::pull_record_commitment_page_with_carrier_recovery;
#[cfg(test)]
use block_page::pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy;
use block_page::pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy;
pub(crate) use block_page::pull_record_commitment_page_with_carrier_runtime_bounded;
use block_page::pull_record_commitment_page_with_endpoint_policy;
use block_page::verify_record_commitment_page;
use carrier_telemetry::record_commitment_authority_carrier_circuit_telemetry;
use carrier_telemetry::record_commitment_block_carrier_circuit_telemetry;
use carrier_telemetry::record_commitment_certificate_carrier_circuit_telemetry;
use certificate_sync::checkpoint_certificate_handler;
use certificate_sync::checkpoint_certificate_member_from_frame;
use certificate_sync::commitment_certificate_source_failure_class;
use certificate_sync::follower_certificate_sync_disposition;
use certificate_sync::normalized_commitment_certificate_carriers;
pub use certificate_sync::pull_record_commitment_checkpoint_certificate;
use certificate_sync::pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy;
use certificate_sync::pull_record_commitment_checkpoint_certificate_with_endpoint_policy;
pub(crate) use certificate_sync::recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime;
use certificate_sync::recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy;
pub use certificate_sync::sync_follower_record_commitment_checkpoint_certificate;
pub(crate) use certificate_sync::sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime;
use certificate_sync::sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy;
#[cfg(test)]
use certificate_sync::sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy;
use certificate_sync::verify_checkpoint_certificate_response;
use checkpoint_sync::checkpoint_handler;
pub use checkpoint_sync::pull_record_commitment_checkpoint;
use checkpoint_sync::pull_record_commitment_checkpoint_with_endpoint_policy;
use checkpoint_sync::verify_record_commitment_checkpoint;
pub use coordinator_lease::release_record_commitment_coordinator_lease;
use coordinator_lease::release_record_commitment_coordinator_lease_with_endpoint_policy;
pub use coordinator_lease::request_record_commitment_coordinator_lease;
use coordinator_lease::request_record_commitment_coordinator_lease_with_endpoint_policy;
use coordinator_lease::verify_record_commitment_coordinator_lease_release_response;
use coordinator_lease::verify_record_commitment_coordinator_lease_response;
pub use custody_witness::plan_custody_audit_witnesses;
use custody_witness::plan_custody_audit_witnesses_with_endpoint_policy;
use custody_witness::validate_custody_anchor_for_producer;
pub use custody_witness::witness_custody_audit_anchor;
pub use custody_witness::witness_custody_audit_anchor_round;
pub use custody_witness::witness_custody_audit_anchor_round_durable;
use custody_witness::witness_custody_audit_anchor_round_with_endpoint_policy;
use custody_witness::witness_custody_audit_anchor_with_endpoint_policy;
pub use delivery_witness::witness_verified_delivery_anchor;
use delivery_witness::witness_verified_delivery_anchor_with_endpoint_policy;
use descriptor_publish::commitment_peer_descriptor;
use descriptor_publish::coordinator_control_requester_is_admitted;
pub use descriptor_publish::publish_current_descriptor_to_commitment_witnesses;
use descriptor_publish::publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy;
use handover::coordinator_handover_source_failure_class;
pub use handover::sync_next_record_coordinator_handover;
pub(crate) use handover::sync_next_record_coordinator_handover_with_carrier_runtime;
use handover::sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy;
#[cfg(test)]
use handover::sync_next_record_coordinator_handover_with_endpoint_policy;
use handover::sync_record_coordinator_handover_from_source_with_endpoint_policy;
use peer_http::classify_http_error;
use peer_http::commitment_block_announce_url;
use peer_http::commitment_block_range_url;
use peer_http::commitment_checkpoint_certificate_url;
use peer_http::commitment_checkpoint_url;
use peer_http::commitment_coordinator_handover_url;
use peer_http::commitment_coordinator_lease_release_url;
use peer_http::commitment_coordinator_lease_url;
pub(crate) use peer_http::commitment_peer_endpoint_is_public;
pub(crate) use peer_http::commitment_peer_url;
use peer_http::custody_audit_anchor_witness_url;
use peer_http::now_secs;
use peer_http::protocol_error;
use peer_http::read_bounded_response;
use peer_http::verified_delivery_anchor_witness_url;
pub use tip_announce::announce_current_record_commitment_tip;
#[cfg(test)]
pub(crate) use tip_announce::announce_current_record_commitment_tip_for_test;
use tip_announce::announce_current_record_commitment_tip_with_endpoint_policy;
use tip_announce::announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy;
use tip_announce::block_announce_handler;
use tip_announce::classify_commitment_tip_announcement_status;
use tip_announce::deliver_commitment_tip_announcement;
use tip_announce::runtime_authorized_coordinator_for_height;
use tip_announce::runtime_authorized_coordinator_for_next_height;
use tip_announce::verified_local_commitment_tip;
use witness_reconcile::checkpoint_relation_priority;
use witness_reconcile::reconcile_record_commitment_candidate_ids;
pub use witness_reconcile::reconcile_record_commitment_pinned_witnesses;
pub use witness_reconcile::reconcile_record_commitment_pinned_witnesses_with_certificate_threshold;
pub(crate) use witness_reconcile::reconcile_record_commitment_pinned_witnesses_with_endpoint_policy;
pub use witness_reconcile::reconcile_record_commitment_witnesses;
use witness_reconcile::reconcile_record_commitment_witnesses_with_endpoint_policy;

#[path = "memchain_peer/control_plane.rs"]
mod control_plane;

const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 512 * 1024;
const MAX_BLOCKS_PER_RESPONSE: usize = 16;
pub(crate) const MAX_BLOCKS_PER_RESPONSE_WIRE: u16 = 16;
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
const PEER_RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);
const PEER_RATE_LIMIT_RETENTION: Duration = Duration::from_secs(120);
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
const REPLAY_RETENTION_SECS: u64 = 120;
const MAX_PINNED_WITNESSES_PER_ROUND: usize = 3;
const PINNED_CARRIER_FAILURES_BEFORE_COOLDOWN: u32 = 2;
const PINNED_CARRIER_RECOVERY_COOLDOWN: Duration = Duration::from_secs(60);
const MAX_DESCRIPTOR_PREFLIGHT_RESPONSE_BYTES: usize = 16 * 1024;
const MAX_CUSTODY_WITNESS_RESPONSE_BYTES: usize = 1024;
const TIP_ANNOUNCEMENT_MAX_ATTEMPTS: usize = 3;
const TIP_ANNOUNCEMENT_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);

/// Aggregate result of one bounded coordinator-descriptor preflight.
///
/// [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] The report deliberately
/// excludes witness identities, endpoints, HTTP statuses, and descriptor
/// fields. It is safe for startup logs and cannot be interpreted as lease,
/// checkpoint, quorum, or consensus evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitmentWitnessDescriptorPublishRound {
    /// Distinct operator-pinned identities considered under the hard cap.
    pub configured: usize,
    /// Requests sent after a signed endpoint passed transport policy.
    pub attempted: usize,
    /// Witness discovery endpoints that accepted the signed descriptor.
    pub accepted: usize,
    /// Missing, unsafe, unreachable, or rejecting witness endpoints.
    pub failed: usize,
}

/// Aggregate result of one bounded follower pull.
///
/// No record ids, block hashes, peer endpoint, or memory metadata are exposed
/// so callers can log this structure without widening the privacy boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentSyncPageOutcome {
    /// Newly persisted blocks from this response page.
    pub inserted: usize,
    /// Blocks already present because another valid catch-up won the race.
    pub already_present: usize,
    /// Whether the signed coordinator tip extends beyond this page.
    pub has_more: bool,
    /// Privacy-safe height of the coordinator's signed chain tip.
    pub remote_tip_height: u64,
}

/// Result of one authenticated exact-next authority synchronization step.
///
/// Coordinator identities stay process-local and must never enter public
/// status, logs, heartbeat fields, or peer reputation. The follower uses this
/// value only to select the next authenticated control-plane request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentAuthoritySyncOutcome {
    /// Highest authority epoch now durable locally.
    pub authority_epoch: u64,
    /// Coordinator authorised for `next_block_height`.
    pub active_coordinator: [u8; 32],
    /// Exact height the next block page must start from.
    pub next_block_height: u64,
    /// Future transition boundary not yet anchored by the local block prefix.
    pub pending_activation_height: Option<u64>,
    /// Whether this step durably inserted the exact-next proof.
    pub handover_inserted: bool,
    /// Identity-blind transport class that supplied the verified snapshot.
    pub source: CommitmentAuthoritySyncSource,
    /// Number of pinned carriers contacted after direct unavailability.
    pub carrier_attempts: usize,
}

/// Privacy-safe transport class for one authority-history synchronization.
///
/// [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] A carrier never becomes
/// authority: it signs only the response envelope around a proof whose
/// predecessor and successor signatures are verified independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentAuthoritySyncSource {
    /// The currently audited coordinator served its own history snapshot.
    Coordinator,
    /// An operator-pinned peer transported the coordinator-signed proof.
    PinnedCarrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedCoordinatorHandoverResponse {
    handover: Option<RecordCoordinatorHandoverV1>,
    latest_authority_epoch: u64,
}

/// Privacy-safe transport class for one verified commitment page.
///
/// [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] This enum deliberately does
/// not retain the responding identity, endpoint, route, request id, block
/// hashes, or certificate material. It describes availability only and must
/// never be used as proposer authority, reputation, consensus, or fork choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentSyncPageSource {
    /// The configured coordinator signed both the envelope and every block.
    Coordinator,
    /// An operator-pinned witness signed the envelope around coordinator blocks.
    PinnedCarrier,
}

/// Result of one direct-first, bounded page retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentFollowerPagePullOutcome {
    /// Fully verified and atomically appended page result.
    pub page: CommitmentSyncPageOutcome,
    /// Identity-blind transport class.
    pub source: CommitmentSyncPageSource,
    /// Number of pinned carriers contacted after the direct availability fault.
    pub carrier_attempts: usize,
}

/// Round-local preference for one typed bounded carrier domain.
///
/// [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] The zero-sized domain keeps
/// authority-proof and block-page preferences separate while sharing the same
/// scheduling algorithm. The cursor stores only an index into the caller's
/// validated pin order; it is neither persisted nor reported and cannot name
/// a node. Direct-first and fail-closed behavior remain mandatory.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CommitmentCarrierCursor<Domain> {
    next_index: usize,
    domain: PhantomData<fn() -> Domain>,
}

impl<Domain> Default for CommitmentCarrierCursor<Domain> {
    fn default() -> Self {
        Self {
            next_index: 0,
            domain: PhantomData,
        }
    }
}

impl<Domain> CommitmentCarrierCursor<Domain> {
    fn reset(&mut self) {
        self.next_index = 0;
    }

    fn start_index(&self, carrier_count: usize) -> usize {
        if carrier_count == 0 {
            0
        } else {
            self.next_index % carrier_count
        }
    }

    fn prefer(&mut self, carrier_index: usize, carrier_count: usize) {
        self.next_index = if carrier_count == 0 {
            0
        } else {
            carrier_index % carrier_count
        };
    }

    fn advance_after_availability_failure(&mut self, carrier_index: usize, carrier_count: usize) {
        self.next_index = if carrier_count == 0 {
            0
        } else {
            carrier_index.saturating_add(1) % carrier_count
        };
    }
}

/// Marker isolating commitment block-page carrier state.
#[derive(Debug)]
pub(crate) enum CommitmentBlockCarrierCircuitDomain {}

/// Marker isolating coordinator-handover carrier scheduling state.
#[derive(Debug)]
pub(crate) enum CommitmentAuthorityCarrierCircuitDomain {}

/// Marker isolating checkpoint-certificate carrier state.
#[derive(Debug)]
pub(crate) enum CommitmentCertificateCarrierCircuitDomain {}

/// Round-local preference for coordinator-handover evidence carriers.
pub(crate) type CommitmentAuthorityCarrierCursor =
    CommitmentCarrierCursor<CommitmentAuthorityCarrierCircuitDomain>;

/// Round-local preference for commitment block-page carriers.
pub(crate) type CommitmentBlockCarrierCursor =
    CommitmentCarrierCursor<CommitmentBlockCarrierCircuitDomain>;

/// Process-only availability circuit for fixed operator-pin positions.
///
/// [TYPED-CARRIER-CIRCUIT 2026-07-29 by Codex] The domain parameter is a
/// zero-sized compile-time boundary: authority-proof, block-page, and
/// certificate recovery share scheduling mechanics without sharing mutable
/// failure state. Slots contain
/// no node id, endpoint, error text, or wall-clock timestamp. Their position is
/// meaningful only inside one normalized pin order, and a pin-count change
/// clears every slot so state cannot be reassigned silently.
#[derive(Debug)]
pub(crate) struct CommitmentCarrierCircuitBreaker<Domain> {
    slots: Vec<CommitmentCarrierCircuitSlot>,
    domain: PhantomData<fn() -> Domain>,
}

/// Block-page availability circuit domain.
pub(crate) type CommitmentBlockCarrierCircuitBreaker =
    CommitmentCarrierCircuitBreaker<CommitmentBlockCarrierCircuitDomain>;

/// Coordinator-handover availability circuit domain.
pub(crate) type CommitmentAuthorityCarrierCircuitBreaker =
    CommitmentCarrierCircuitBreaker<CommitmentAuthorityCarrierCircuitDomain>;

/// Checkpoint-certificate availability circuit domain.
pub(crate) type CommitmentCertificateCarrierCircuitBreaker =
    CommitmentCarrierCircuitBreaker<CommitmentCertificateCarrierCircuitDomain>;

#[derive(Debug, Default)]
struct CommitmentCarrierCircuitSlot {
    consecutive_availability_failures: u32,
    retry_after: Option<Instant>,
}

/// Scheduling state for one anonymous fixed circuit slot.
///
/// [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] This enum is local
/// control flow only. It deliberately carries no identity, endpoint, error,
/// status code, route, payload, or timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentCarrierCircuitDecision {
    Closed,
    Cooling,
    HalfOpen,
}

impl<Domain> Default for CommitmentCarrierCircuitBreaker<Domain> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            domain: PhantomData,
        }
    }
}

impl<Domain> CommitmentCarrierCircuitBreaker<Domain> {
    fn align_slots(&mut self, carrier_count: usize) {
        if self.slots.len() != carrier_count {
            self.slots.clear();
            self.slots
                .resize_with(carrier_count, CommitmentCarrierCircuitSlot::default);
        }
    }

    fn decision(&self, carrier_index: usize, now: Instant) -> CommitmentCarrierCircuitDecision {
        let Some(slot) = self.slots.get(carrier_index) else {
            debug_assert!(
                false,
                "carrier circuit slot must be aligned before selection"
            );
            return CommitmentCarrierCircuitDecision::Cooling;
        };
        match slot.retry_after {
            None => CommitmentCarrierCircuitDecision::Closed,
            Some(retry_after) if now < retry_after => CommitmentCarrierCircuitDecision::Cooling,
            Some(_) => CommitmentCarrierCircuitDecision::HalfOpen,
        }
    }

    fn cooling_slots(&self, now: Instant) -> usize {
        self.slots
            .iter()
            .filter(|slot| {
                slot.retry_after
                    .is_some_and(|retry_after| now < retry_after)
            })
            .count()
    }

    fn record_success(&mut self, carrier_index: usize) {
        if let Some(slot) = self.slots.get_mut(carrier_index) {
            *slot = CommitmentCarrierCircuitSlot::default();
        }
    }

    fn record_availability_failure(&mut self, carrier_index: usize, now: Instant) {
        let Some(slot) = self.slots.get_mut(carrier_index) else {
            return;
        };

        if slot
            .retry_after
            .is_some_and(|retry_after| now >= retry_after)
        {
            // One failed half-open probe immediately reopens the circuit.
            slot.consecutive_availability_failures = 0;
            slot.retry_after = Some(now + PINNED_CARRIER_RECOVERY_COOLDOWN);
            return;
        }

        if slot.retry_after.is_some() {
            // A cooling slot is never scheduled, but keep this method robust
            // against an accidental duplicate outcome.
            return;
        }

        slot.consecutive_availability_failures =
            slot.consecutive_availability_failures.saturating_add(1);
        if slot.consecutive_availability_failures >= PINNED_CARRIER_FAILURES_BEFORE_COOLDOWN {
            slot.consecutive_availability_failures = 0;
            slot.retry_after = Some(now + PINNED_CARRIER_RECOVERY_COOLDOWN);
        }
    }
}

/// One independently verified coordinator lease grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCoordinatorLeaseGrant {
    /// Signed witness lease epoch.
    pub lease_epoch: u64,
    /// Signed witness wall-clock expiry.
    pub lease_expires_at: u64,
    /// Conservative duration between signed response time and expiry.
    pub valid_for_secs: u64,
}

/// One independently verified graceful lease release acknowledgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCoordinatorLeaseRelease {
    /// Released witness lease generation.
    pub lease_epoch: u64,
    /// Signed witness release timestamp.
    pub released_at: u64,
}

/// Privacy-safe aggregate result of one bounded external witness round.
///
/// No node ids, endpoints, anchor digests, delivery counts, message ids, or
/// client metadata are retained in this structure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifiedDeliveryAnchorWitnessRound {
    /// Distinct configured witness identities after hard bounding.
    pub configured: usize,
    /// Witnesses for which transport was attempted.
    pub attempted: usize,
    /// Cryptographically valid signed responses.
    pub verified: usize,
    /// Witnesses that durably advanced.
    pub advanced: usize,
    /// Witnesses already holding the exact anchor.
    pub idempotent: usize,
    /// Witnesses proving the requester rolled back below their high-water.
    pub stale: usize,
    /// Witnesses proving a different digest reused the same generation.
    pub conflicts: usize,
    /// Witnesses refusing a discontinuous generation advance.
    pub gaps: usize,
    /// Admission, endpoint, transport, decoding, or signature failures.
    pub failed: usize,
}

/// Privacy-safe dry-run plan for independent custody witnesses.
///
/// [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] This value contains only
/// aggregate policy counts. Planning never signs, serializes, or transmits a
/// custody anchor and cannot reveal node identities or endpoints to callers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustodyAuditWitnessPlan {
    /// Distinct non-self operator pins considered after validation.
    pub configured: usize,
    /// Pins with a fresh descriptor, storage capability, and safe endpoint.
    pub eligible: usize,
    /// Configured pins that are currently unavailable or ineligible.
    pub unavailable: usize,
    /// Duplicate identities ignored defensively by the runtime planner.
    pub duplicates_ignored: usize,
    /// Local identity pins excluded because self-witnessing is invalid.
    pub self_excluded: usize,
    /// Independent eligible witnesses required by operator policy.
    pub minimum_verified: usize,
    /// Whether the current local peer view can satisfy the policy threshold.
    pub quorum_ready: bool,
}

/// Privacy-safe aggregate result of one bounded custody witness round.
///
/// [CUSTODY-WITNESS-TRANSPORT 2026-08-16 by Codex] The round intentionally
/// retains no witness ids, endpoints, frame hashes, receipt signatures, or
/// custody counters. A verified adverse receipt always prevents quorum from
/// being reported, even when enough other witnesses accept the same anchor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustodyAuditWitnessRound {
    /// Distinct non-self operator pins considered by this round.
    pub configured: usize,
    /// Cryptographically valid request-bound witness receipts.
    pub verified: usize,
    /// Valid receipts proving the exact requested anchor was retained.
    pub accepted: usize,
    /// Witnesses that durably advanced to the requested generation.
    pub advanced: usize,
    /// Witnesses that already retained the exact requested anchor.
    pub idempotent: usize,
    /// Witnesses proving the producer requested an older generation.
    pub stale: usize,
    /// Witnesses proving same-generation anchor equivocation.
    pub conflicts: usize,
    /// Witnesses refusing a discontinuous generation advance.
    pub gaps: usize,
    /// Admission, endpoint, transport, decoding, or signature failures.
    pub failed: usize,
    /// Duplicate identities ignored defensively by the runtime round.
    pub duplicates_ignored: usize,
    /// Local identity pins excluded because self-witnessing is invalid.
    pub self_excluded: usize,
    /// Independent accepted receipts required by local policy.
    pub minimum_verified: usize,
    /// Whether any authentic stale, conflict, or gap evidence was observed.
    pub adverse_evidence: bool,
    /// Whether enough receipts accepted and no adverse evidence was observed.
    pub quorum_satisfied: bool,
}

/// Relationship proven by one valid signed checkpoint response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentCheckpointRelation {
    /// Both peers signed the same tip height and hash.
    Converged,
    /// The responder extends the requester's verified chain prefix.
    RemoteAhead,
    /// The responder is behind but shares its full verified prefix.
    RemoteBehind,
    /// The signed chains disagree at the shorter peer's tip.
    Diverged,
}

impl CommitmentCheckpointRelation {
    /// Stable privacy-safe status value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::RemoteAhead => "remote_ahead",
            Self::RemoteBehind => "remote_behind",
            Self::Diverged => "diverged",
        }
    }
}

/// Aggregate result of a cryptographically verified checkpoint response.
///
/// The evidence digest identifies the exact signed response for an operator
/// evidence vault without putting peer identities, hashes, or signatures into
/// logs, status APIs, or heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCheckpointOutcome {
    /// Proven relationship between the two verified chains.
    pub relation: CommitmentCheckpointRelation,
    /// Requester's tip height at proof construction.
    pub local_tip_height: u64,
    /// Responder's signed tip height.
    pub remote_tip_height: u64,
    /// Height at which the shared-prefix comparison was made.
    pub checkpoint_height: u64,
    /// SHA-256 digest of the complete signed response frame.
    pub evidence_digest: [u8; 32],
}

/// Privacy-safe aggregate result of one bounded coordinator witness round.
///
/// Counts establish only how many signed observations were collected. They do
/// not represent votes, quorum, finality, peer trust weight, or fork choice.
/// Peer identities, endpoints, hashes, signatures, and request ids remain in
/// the local evidence vault and are deliberately absent from this structure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitmentReconciliationOutcome {
    /// Valid discovered peers eligible for checkpoint observation.
    pub eligible_witnesses: usize,
    /// Peers contacted after applying the per-round bound.
    pub attempted: usize,
    /// Responses that passed identity, freshness, signature, and chain checks.
    pub verified: usize,
    /// Verified peers at the same height and hash.
    pub converged: usize,
    /// Verified peers extending the local chain prefix.
    pub remote_ahead: usize,
    /// Verified peers behind the local tip on the same prefix.
    pub remote_behind: usize,
    /// Verified peers signing a different hash at the shared height.
    pub diverged: usize,
    /// Attempts that did not establish durable signed evidence.
    pub failed: usize,
    /// Distinct certifiable pinned-witness frames in this exact round.
    pub certificate_signers: usize,
    /// Threshold requested for an immutable certificate; zero when disabled.
    pub certificate_required_signers: usize,
    /// Whether the current local tip has a re-audited immutable certificate.
    pub certificate_persisted: bool,
    /// Whether certificate persistence or its full re-audit failed.
    pub certificate_persistence_failed: bool,
}

/// Privacy-safe result of importing one independently verified certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCertificateImportOutcome {
    /// Certified local height; hashes and witness identities remain private.
    pub checkpoint_height: u64,
    /// Distinct pinned witnesses represented by exact signed frames.
    pub signer_count: usize,
    /// Threshold embedded in the immutable certificate.
    pub required_signers: usize,
    /// Whether storage contains a fully re-audited certificate afterward.
    pub persisted: bool,
}

/// Transport class used only to classify a verified follower certificate result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentFollowerCertificateSource {
    Coordinator,
    PinnedCarrier,
}

/// Source-blind terminal state of one coordinator certificate recovery round.
///
/// [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] This contract exposes
/// only bounded aggregate control state to the server runtime. A caller cannot
/// log or persist source identity, endpoint, signature material, request ids,
/// response bytes, or the underlying security error through this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitmentCertificateCarrierRecoveryDisposition {
    /// A fully verified certificate is now durable under current local policy.
    Persisted,
    /// The response verified, but a concurrent local state change deferred
    /// persistence.
    VerifiedUnpersisted,
    /// Every eligible non-cooling carrier ended in an availability failure.
    AvailabilityExhausted,
    /// Policy, authentication, canonicalization, or evidence validation failed.
    SecurityStopped,
}

/// Privacy-safe aggregate result of one coordinator certificate recovery round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommitmentCertificateCarrierRecoveryRound {
    /// Terminal source-blind disposition.
    pub disposition: CommitmentCertificateCarrierRecoveryDisposition,
    /// Verified certificate height, or zero when no response verified.
    pub checkpoint_height: u64,
    /// Verified distinct signer count, or zero when no response verified.
    pub signer_count: usize,
    /// Receiver-enforced signer threshold, or zero when no response verified.
    pub required_signers: usize,
    /// Carrier HTTP requests actually started after circuit filtering.
    pub carrier_attempts: usize,
    /// Cooling carrier slots skipped without transport.
    pub cooldown_skips: usize,
    /// Expired-cooldown carrier probes attempted in this round.
    pub half_open_attempts: usize,
    /// Anonymous carrier slots still cooling after this round.
    pub cooling_slots: usize,
}

/// Result of one policy-bounded follower checkpoint-certificate refresh.
///
/// [FOLLOWER-CERTIFICATE-SYNC 2026-07-29 by Codex] This result intentionally
/// contains no source identity, endpoint, witness identity, hash, signature,
/// request id, or frame. A refresh is post-convergence evidence replication;
/// it grants no startup authority and cannot select or mutate the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentFollowerCertificateSyncOutcome {
    /// The local operator policy does not require threshold certificates.
    PolicyDisabled,
    /// The audited local vault already certifies the exact converged tip.
    AlreadyCurrent,
    /// A source response passed local policy and durable re-audit.
    Refreshed(CommitmentCertificateImportOutcome),
}

/// Aggregate delivery result for one best-effort commitment tip announcement.
///
/// Peer identities, endpoints, hashes, and timing remain intentionally absent
/// so the result is safe for operational logs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitmentTipAnnouncementOutcome {
    /// Audited local tip height actually encoded in the outbound frame.
    pub announced_height: u64,
    /// Distinct operator-pinned peers considered in this bounded round.
    pub attempted: usize,
    /// Peers that accepted or coalesced the wake-up hint.
    pub accepted: usize,
    /// Peers already at or above the announced height.
    pub stale: usize,
    /// Missing, unsafe, unreachable, or incompatible peers.
    pub failed: usize,
    /// Additional HTTP attempts after an initial transient delivery failure.
    pub retries_attempted: usize,
    /// Peers that returned a terminal accepted/stale receipt after a retry.
    pub retries_succeeded: usize,
    /// Peers still transiently failing after the bounded retry budget.
    pub retries_exhausted: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentTipAnnouncementDelivery {
    Accepted,
    Stale,
    RetryableFailure,
    PermanentFailure,
}

#[derive(Debug, Clone, Copy)]
struct CommitmentTipAnnouncementRetryPolicy {
    max_attempts: usize,
    base_delay: Duration,
}

const TIP_ANNOUNCEMENT_RETRY_POLICY: CommitmentTipAnnouncementRetryPolicy =
    CommitmentTipAnnouncementRetryPolicy {
        max_attempts: TIP_ANNOUNCEMENT_MAX_ATTEMPTS,
        base_delay: TIP_ANNOUNCEMENT_RETRY_BASE_DELAY,
    };

#[derive(Debug)]
struct VerifiedCommitmentPage {
    blocks: Vec<RecordCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
}

#[derive(Debug)]
struct VerifiedCertificateMember {
    observed_at: u64,
    remote_tip_height: u64,
    evidence_digest: [u8; 32],
    frame: Vec<u8>,
}

#[derive(Debug)]
struct VerifiedCheckpointCertificate {
    checkpoint_height: u64,
    required_signers: usize,
    members: Vec<VerifiedCertificateMember>,
}

#[derive(Clone)]
struct MemChainPeerState {
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    guard: Arc<Mutex<PeerRequestGuard>>,
    lease_authorized_coordinator: Option<[u8; 32]>,
    block_announce_notifier: Option<mpsc::Sender<u64>>,
}

#[derive(Debug, Default)]
struct PeerRequestGuard {
    rate_windows: HashMap<[u8; 32], PeerRateWindow>,
    seen_requests: HashMap<([u8; 32], [u8; 16]), u64>,
}

#[derive(Debug, Clone, Copy)]
struct PeerRateWindow {
    started_at: Instant,
    used: u32,
}

impl PeerRateWindow {
    const fn new(started_at: Instant) -> Self {
        Self {
            started_at,
            used: 0,
        }
    }
}

impl PeerRequestGuard {
    fn prune_replay_requests(&mut self, now: u64) {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
    }

    // [MEMCHAIN-PEER-MONOTONIC-RATE 2026-08-12 by Codex] Authentication
    // freshness intentionally uses Unix time, but elapsed abuse-control
    // windows must use Instant so NTP corrections cannot reset peer budgets.
    fn admit_rate_limited_at(&mut self, requester: [u8; 32], now: Instant) -> bool {
        self.rate_windows.retain(|_, window| {
            now.checked_duration_since(window.started_at)
                .is_none_or(|elapsed| elapsed < PEER_RATE_LIMIT_RETENTION)
        });
        let window = self
            .rate_windows
            .entry(requester)
            .or_insert_with(|| PeerRateWindow::new(now));
        let elapsed = now.checked_duration_since(window.started_at);
        if elapsed.is_some_and(|elapsed| elapsed >= PEER_RATE_LIMIT_WINDOW) {
            *window = PeerRateWindow::new(now);
        }
        if window.used >= MAX_REQUESTS_PER_PEER_PER_MINUTE {
            return false;
        }
        window.used += 1;
        true
    }

    /// Admits a stateful request exactly once inside the replay-retention window.
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.admit_at(requester, request_id, now, Instant::now())
    }

    fn admit_at(
        &mut self,
        requester: [u8; 32],
        request_id: [u8; 16],
        wall_now: u64,
        monotonic_now: Instant,
    ) -> bool {
        self.prune_replay_requests(wall_now);
        if !self.admit_rate_limited_at(requester, monotonic_now) {
            return false;
        }
        // Rejected replay attempts consume the same abuse budget as valid
        // requests; otherwise one signed frame could bypass the rate cap.
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }
        self.seen_requests.insert((requester, request_id), wall_now);
        true
    }

    /// Admits an authenticated, idempotent scheduling hint within the shared cap.
    fn admit_idempotent_hint(&mut self, requester: [u8; 32], now: u64) -> bool {
        self.admit_idempotent_hint_at(requester, now, Instant::now())
    }

    fn admit_idempotent_hint_at(
        &mut self,
        requester: [u8; 32],
        wall_now: u64,
        monotonic_now: Instant,
    ) -> bool {
        self.prune_replay_requests(wall_now);
        self.admit_rate_limited_at(requester, monotonic_now)
    }
}

/// Builds the signed node-to-node commitment block sync router.
#[must_use]
pub fn build_memchain_peer_router(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
) -> Router {
    build_memchain_peer_router_with_runtime(storage, peer_store, identity, None, None)
}

/// Builds the peer router with an optional follower-side lease trust root.
///
/// `lease_authorized_coordinator` must be the follower's explicitly pinned
/// Block Sync coordinator. `None` keeps the new endpoint fail-closed while all
/// existing block/checkpoint routes remain wire-compatible.
#[must_use]
pub fn build_memchain_peer_router_with_coordinator_lease(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    lease_authorized_coordinator: Option<[u8; 32]>,
) -> Router {
    build_memchain_peer_router_with_runtime(
        storage,
        peer_store,
        identity,
        lease_authorized_coordinator,
        None,
    )
}

/// Builds the peer router with follower lease and event-driven sync runtime.
///
/// The same explicitly pinned coordinator identity authorizes lease requests
/// and block announcements. The notifier is bounded by the caller; this
/// handler uses only `try_send`, so public traffic cannot stall the HTTP task.
#[must_use]
pub fn build_memchain_peer_router_with_runtime(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    lease_authorized_coordinator: Option<[u8; 32]>,
    block_announce_notifier: Option<mpsc::Sender<u64>>,
) -> Router {
    let state = MemChainPeerState {
        storage,
        peer_store,
        identity,
        guard: Arc::new(Mutex::new(PeerRequestGuard::default())),
        lease_authorized_coordinator,
        block_announce_notifier,
    };
    let router = Router::new()
        .route(
            "/api/memchain/peer/block-announce",
            post(block_announce_handler),
        )
        .route("/api/memchain/peer/block-range", post(block_range_handler))
        .route("/api/memchain/peer/checkpoint", post(checkpoint_handler))
        .route(
            "/api/memchain/peer/checkpoint-certificate",
            post(checkpoint_certificate_handler),
        );
    let router = control_plane::mount_routes(router);
    router
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

// [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] Descriptor freshness is an
// explicit trust-boundary choice. Only the operator-pinned witness path may use
// an authentic expired cache record as a transport hint; every permissionless
// and route-bearing path remains current-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitmentPeerDescriptorPolicy {
    CurrentOnly,
    AllowExpiredForPinnedWitness,
}

#[derive(Debug)]
enum CommitmentCertificateCarrierPullTerminal {
    Imported(CommitmentCertificateImportOutcome),
    AvailabilityExhausted,
    SecurityStopped(String),
}

#[derive(Debug)]
struct CommitmentCertificateCarrierPullRound {
    terminal: CommitmentCertificateCarrierPullTerminal,
    carrier_attempts: usize,
    cooldown_skips: usize,
    half_open_attempts: usize,
}

/// Determines whether trying another already-pinned evidence carrier is safe.
///
/// This allowlist is intentionally narrow. Decode, signature, identity,
/// policy, tip, canonicalization, size, and persistence failures are security
/// failures even when another source might return a valid-looking response.
///
/// [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentCertificateSourceFailureClass {
    Availability,
    Security,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentAuthoritySourceFailureClass {
    Availability,
    Security,
}

/// Narrow retry policy for alternate pinned block carriers.
///
/// Decode, signature, responder, proposer, continuity, pagination, size,
/// endpoint-policy, and storage failures are always security failures. A stale
/// carrier tip is availability-only because it cannot mutate local state and a
/// later exact pin may hold a newer verified prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentBlockSourceFailureClass {
    Availability,
    Security,
}

#[cfg(test)]
mod tests {
    mod admission;
    mod carrier;
    mod commitment;
    mod coordinator;
    mod handover;
    mod remaining;
    mod verification;
    use super::*;

    use aeronyx_core::protocol::{
        NodeBootstrapSnapshot, NodeDescriptor, NodeDiscoveryMessage, SignedNodeDescriptor,
    };
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn admit_peer(
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        endpoint: Option<String>,
        now: u64,
    ) {
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        descriptor.public_endpoint = endpoint;
        descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
        let descriptor = SignedNodeDescriptor::sign(descriptor, identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);
    }

    fn allow_test_endpoint(_endpoint: &str) -> bool {
        true
    }

    fn custody_witness_request_frame(
        producer: &IdentityKeyPair,
        anchor: &CustodyAuditAnchorV1,
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let producer_id = producer.public_key_bytes();
        let anchor_sha256 =
            custody_audit_anchor_frame_sha256(anchor).expect("hash custody audit anchor");
        let signing_bytes = custody_audit_anchor_witness_request_signing_bytes(
            &request_id,
            &producer_id,
            request_timestamp,
            &anchor_sha256,
        );
        encode_memchain(&MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
            request_id,
            requester: producer_id,
            request_timestamp,
            anchor: anchor.clone(),
            signature: producer.sign(&signing_bytes),
        })
        .expect("encode custody witness request")
    }

    fn delivery_witness_request_frame(
        requester: &IdentityKeyPair,
        generation: u64,
        anchor_digest: [u8; 32],
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let requester_id = requester.public_key_bytes();
        let signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
            &requester_id,
            generation,
            &anchor_digest,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            requester: requester_id,
            generation,
            anchor_digest,
            request_id,
            request_timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .expect("encode delivery witness request")
    }

    async fn post_custody_witness(router: &Router, frame: Vec<u8>) -> Response {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/custody-audit-anchor-witness")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .expect("custody witness HTTP request"),
            )
            .await
            .expect("custody witness HTTP response")
    }

    async fn post_delivery_witness(router: &Router, frame: Vec<u8>) -> Response {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/discovery/peer/verified-delivery-anchor-witness")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .expect("delivery witness HTTP request"),
            )
            .await
            .expect("delivery witness HTTP response")
    }

    fn signed_handover_response_frame(
        responder: &IdentityKeyPair,
        request_id: [u8; 16],
        response_timestamp: u64,
        handover: Option<RecordCoordinatorHandoverV1>,
        latest_authority_epoch: u64,
    ) -> Vec<u8> {
        let responder_id = responder.public_key_bytes();
        let signing_bytes = record_coordinator_handover_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder_id,
            response_timestamp,
            handover.as_ref(),
            latest_authority_epoch,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorHandoverResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder: responder_id,
            response_timestamp,
            handover,
            latest_authority_epoch,
            signature: responder.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn block_announce_frame(block: &RecordCommitmentBlockV1) -> Vec<u8> {
        encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
            header: block.header.clone(),
            proposer_signature: block.proposer_signature,
        })
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn coordinator_lease_request_frame(
        coordinator: &IdentityKeyPair,
        instance_id: [u8; 32],
        tip_height: u64,
        tip_hash: [u8; 32],
        ttl_secs: u32,
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let coordinator_id = coordinator.public_key_bytes();
        let signing_bytes = record_coordinator_lease_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &coordinator_id,
            &instance_id,
            tip_height,
            &tip_hash,
            ttl_secs,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorLeaseRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            coordinator: coordinator_id,
            instance_id,
            known_tip_height: tip_height,
            known_tip_hash: tip_hash,
            requested_ttl_secs: ttl_secs,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn coordinator_lease_release_request_frame(
        coordinator: &IdentityKeyPair,
        instance_id: [u8; 32],
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let coordinator_id = coordinator.public_key_bytes();
        let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &coordinator_id,
            &instance_id,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            coordinator: coordinator_id,
            instance_id,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn signed_block_page_frame(
        signer: &IdentityKeyPair,
        request_id: [u8; 16],
        response_timestamp: u64,
        blocks: Vec<RecordCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
    ) -> Vec<u8> {
        let responder = signer.public_key_bytes();
        let signing_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            response_timestamp,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: signer.sign(&signing_bytes),
        })
        .unwrap()
    }
}
