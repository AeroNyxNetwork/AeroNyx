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
    custody_audit_anchor_frame_sha256, custody_audit_witness_receipt_frame_sha256,
    CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1, CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
    CUSTODY_AUDIT_WITNESS_CONFLICT_V1, CUSTODY_AUDIT_WITNESS_GAP_V1,
    CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1, CUSTODY_AUDIT_WITNESS_STALE_V1,
};
use aeronyx_core::protocol::memchain::{
    custody_audit_anchor_witness_request_signing_bytes,
    custody_audit_anchor_witness_response_signing_bytes, decode_memchain, encode_memchain,
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
    CustodyAuditAnchorWitnessOutcome, RecordCommitmentAuthorityState,
    RecordCommitmentCheckpointEvidencePersistOutcome, RecordCoordinatorHandoverPersistOutcome,
    RecordCoordinatorLeaseGrantOutcome, RecordCoordinatorLeaseReleaseOutcome,
    VerifiedDeliveryAnchorWitnessOutcome,
};
use crate::services::memchain::{
    MemoryStorage, RecordCommitmentAnnouncementDisposition,
    RecordCommitmentAuthoritySyncDisposition, RecordCommitmentBlockPagePullDisposition,
    RecordCommitmentCertificatePolicyReadiness, RecordCommitmentCertificateSyncDisposition,
};
use crate::services::PeerStore;

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

    fn advance_after_availability_failure(
        &mut self,
        carrier_index: usize,
        carrier_count: usize,
    ) {
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

fn record_commitment_block_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentBlockCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Observe the
    // monotonic circuit at one instant, then discard every per-slot detail.
    // Storage receives only bounded aggregate counts and cannot reconstruct a
    // source identity or endpoint from this call.
    storage.record_commitment_block_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
}

fn record_commitment_authority_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentAuthorityCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Authority transport
    // has a distinct typed circuit. Collapse it to source-blind aggregates at
    // the storage boundary so telemetry cannot reconstruct a carrier or route.
    storage.record_commitment_authority_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
}

fn record_commitment_certificate_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentCertificateCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] Preserve the same
    // source-blind aggregate contract as block-page recovery while keeping an
    // independent typed circuit and independent counters.
    storage.record_commitment_certificate_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
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

/// Maps certificate source and durable outcome into one telemetry disposition.
///
/// [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] Verification proves that
/// a response is authentic; it does not prove recovery until the exact current
/// policy certificate is durable locally. Keeping this mapping in one pure
/// function prevents direct and carrier paths from drifting apart.
const fn follower_certificate_sync_disposition(
    source: CommitmentFollowerCertificateSource,
    persisted: bool,
) -> RecordCommitmentCertificateSyncDisposition {
    match (source, persisted) {
        (CommitmentFollowerCertificateSource::Coordinator, true) => {
            RecordCommitmentCertificateSyncDisposition::Coordinator
        }
        (CommitmentFollowerCertificateSource::PinnedCarrier, true) => {
            RecordCommitmentCertificateSyncDisposition::CarrierRecovered
        }
        (_, false) => RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted,
    }
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

// Keep the receipt contract at the HTTP wire boundary. Axum and Reqwest may
// resolve different `http` crate versions, but the protocol status codes are
// stable and must not depend on either transport library's Rust type.
fn classify_commitment_tip_announcement_status(status: u16) -> CommitmentTipAnnouncementDelivery {
    match status {
        202 => CommitmentTipAnnouncementDelivery::Accepted,
        204 => CommitmentTipAnnouncementDelivery::Stale,
        408 | 425 | 500..=599 => CommitmentTipAnnouncementDelivery::RetryableFailure,
        _ => CommitmentTipAnnouncementDelivery::PermanentFailure,
    }
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

// [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] A follower may use an
// authentic expired cache descriptor only for its explicitly pinned
// coordinator's signed checkpoint/lease control traffic. The request timestamp
// and payload signature are still verified by each handler. This breaks the
// reverse half of the cold-start deadlock without admitting stale
// permissionless peers to block sync, routing, gossip, or public membership.
fn coordinator_control_requester_is_admitted(
    state: &MemChainPeerState,
    requester: &[u8; 32],
    now: u64,
) -> bool {
    state.peer_store.get_valid(requester, now).is_some()
        || (state.lease_authorized_coordinator == Some(*requester)
            && state
                .peer_store
                .get_signature_verified_cached(requester)
                .is_some())
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
        self.seen_requests
            .insert((requester, request_id), wall_now);
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
    Router::new()
        .route(
            "/api/memchain/peer/block-announce",
            post(block_announce_handler),
        )
        .route("/api/memchain/peer/block-range", post(block_range_handler))
        .route("/api/memchain/peer/checkpoint", post(checkpoint_handler))
        .route(
            "/api/memchain/peer/checkpoint-certificate",
            post(checkpoint_certificate_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-lease",
            post(coordinator_lease_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-lease/release",
            post(coordinator_lease_release_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-handover",
            post(coordinator_handover_handler),
        )
        .route(
            "/api/discovery/peer/verified-delivery-anchor-witness",
            post(verified_delivery_anchor_witness_handler),
        )
        .route(
            "/api/memchain/peer/custody-audit-anchor-witness",
            post(custody_audit_anchor_witness_handler),
        )
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn verified_local_commitment_tip(storage: &MemoryStorage) -> Result<(u64, [u8; 32]), String> {
    let (_, checkpoint_hash, tip_height, tip_hash) = storage
        .record_commitment_chain_checkpoint(u64::MAX)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if checkpoint_hash != tip_hash {
        return Err("local_checkpoint_tip_mismatch".to_string());
    }
    Ok((tip_height, tip_hash))
}

/// Announces the current audited tip to a bounded set of operator-pinned peers.
///
/// Delivery is advisory and best effort. Followers independently authenticate
/// the pinned coordinator and then run the ordinary signed page/checkpoint
/// pull, so accepting this frame never changes their canonical chain.
///
/// # Errors
///
/// Returns an error only when the local audited tip cannot be loaded or encoded.
/// Individual peer failures are represented in the privacy-safe aggregate.
pub async fn announce_current_record_commitment_tip(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
) -> Result<CommitmentTipAnnouncementOutcome, String> {
    announce_current_record_commitment_tip_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Runs the production announcement encoder, peer lookup, HTTP transport, and
/// retry queue with a localhost-capable endpoint policy for integration tests.
///
/// This seam does not exist in non-test builds. Production callers must use
/// [`announce_current_record_commitment_tip`], which enforces final-hop public
/// endpoint validation on every attempt.
#[cfg(test)]
pub(crate) async fn announce_current_record_commitment_tip_for_test(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    max_attempts: usize,
    base_delay: Duration,
) -> Result<CommitmentTipAnnouncementOutcome, String> {
    let endpoint_allowed = |_endpoint: &str| true;
    announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        &endpoint_allowed,
        CommitmentTipAnnouncementRetryPolicy {
            max_attempts,
            base_delay,
        },
    )
    .await
}

async fn announce_current_record_commitment_tip_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        endpoint_allowed,
        TIP_ANNOUNCEMENT_RETRY_POLICY,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
    retry_policy: CommitmentTipAnnouncementRetryPolicy,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let (tip_height, _) = verified_local_commitment_tip(storage).await?;
    if tip_height == 0 {
        return Err("local_commitment_tip_empty".to_string());
    }
    let page = storage
        .get_verified_record_commitment_block_page(tip_height, 1)
        .await
        .map_err(|_| "local_commitment_tip_unavailable".to_string())?;
    let block = page
        .blocks
        .into_iter()
        .next()
        .filter(|block| block.header.height == tip_height)
        .ok_or_else(|| "local_commitment_tip_unavailable".to_string())?;
    if block.header.proposer != identity.public_key_bytes() {
        return Err("local_commitment_tip_not_self_proposed".to_string());
    }
    let frame = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
        header: block.header,
        proposer_signature: block.proposer_signature,
    })
    .map_err(|_| "tip_announcement_encode_failed".to_string())?;

    let self_node_id = identity.public_key_bytes();
    let mut distinct = HashSet::new();
    let mut outcome = CommitmentTipAnnouncementOutcome {
        announced_height: tip_height,
        ..CommitmentTipAnnouncementOutcome::default()
    };
    let mut pending = pinned_peer_ids
        .iter()
        .copied()
        .filter(|peer_id| *peer_id != self_node_id && distinct.insert(*peer_id))
        .take(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>();
    outcome.attempted = pending.len();

    // Keep the queue hard-bounded even if a future internal caller supplies a
    // malformed policy. Production uses exactly three attempts.
    let max_attempts = retry_policy.max_attempts.clamp(1, 8);
    let mut attempt_number = 1usize;
    while !pending.is_empty() {
        if attempt_number > 1 {
            let shift = u32::try_from(attempt_number.saturating_sub(2))
                .unwrap_or(u32::MAX)
                .min(7);
            let multiplier = 1u32 << shift;
            tokio::time::sleep(retry_policy.base_delay.saturating_mul(multiplier)).await;
            outcome.retries_attempted = outcome.retries_attempted.saturating_add(pending.len());
        }

        let deliveries = futures::stream::iter(pending.into_iter())
            .map(|peer_id| {
                let frame = frame.clone();
                async move {
                    let delivery = deliver_commitment_tip_announcement(
                        peer_store,
                        client,
                        peer_id,
                        frame,
                        endpoint_allowed,
                    )
                    .await;
                    (peer_id, delivery)
                }
            })
            .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
            .collect::<Vec<_>>()
            .await;
        let mut retry_queue = Vec::with_capacity(deliveries.len());
        for (peer_id, delivery) in deliveries {
            match delivery {
                CommitmentTipAnnouncementDelivery::Accepted => {
                    outcome.accepted = outcome.accepted.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::Stale => {
                    outcome.stale = outcome.stale.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure
                    if attempt_number < max_attempts =>
                {
                    retry_queue.push(peer_id);
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                    outcome.retries_exhausted = outcome.retries_exhausted.saturating_add(1);
                }
                CommitmentTipAnnouncementDelivery::PermanentFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                }
            }
        }
        pending = retry_queue;
        attempt_number = attempt_number.saturating_add(1);
    }
    Ok(outcome)
}

async fn deliver_commitment_tip_announcement<F>(
    peer_store: &PeerStore,
    client: &reqwest::Client,
    peer_id: [u8; 32],
    frame: Vec<u8>,
    endpoint_allowed: &F,
) -> CommitmentTipAnnouncementDelivery
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let Some(peer) = peer_store.get_valid(&peer_id, now_secs()) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    if !endpoint_allowed(endpoint) {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    }
    let Ok(url) = commitment_block_announce_url(endpoint) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    match client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
    {
        Ok(response) => classify_commitment_tip_announcement_status(response.status().as_u16()),
        Err(_) => CommitmentTipAnnouncementDelivery::RetryableFailure,
    }
}

async fn block_announce_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordBlockAnnounceV1 {
        header,
        proposer_signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if header.protocol_version != RECORD_COMMITMENT_BLOCK_VERSION_V1
        || header.chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || header.height == 0
        || header.timestamp == 0
        || header.timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS)
        || header.record_count == 0
        || header.record_count as usize > MAX_RECORD_COMMITMENTS_PER_BLOCK
        || (header.height == 1 && header.prev_block_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_block_announcement");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Authenticate before
    // consulting authority or PeerStore state. This avoids membership and
    // authority probes while keeping unauthenticated traffic away from the
    // storage-backed authority audit.
    let header_hash = header.hash();
    let signature_valid = IdentityPublicKey::from_bytes(&header.proposer)
        .and_then(|key| key.verify(&header_hash, &proposer_signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    let authorized_coordinator = match runtime_authorized_coordinator_for_height(
        &state.storage,
        state.lease_authorized_coordinator,
        header.height,
    )
    .await
    {
        Ok(Some(coordinator)) => coordinator,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited announcement authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if header.proposer != authorized_coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "coordinator_not_authorized");
    }
    if state.peer_store.get_valid(&header.proposer, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state
        .guard
        .lock()
        .await
        .admit_idempotent_hint(header.proposer, now)
    {
        // Keep the established wire error for older coordinators even though
        // authenticated tip hints are now rejected only by the shared rate cap.
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (local_tip_height, _) = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified"),
    };
    if header.height <= local_tip_height {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Stale,
        );
        return StatusCode::NO_CONTENT.into_response();
    }
    let Some(notifier) = state.block_announce_notifier.as_ref() else {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Unavailable,
        );
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable");
    };
    match notifier.try_send(header.height) {
        Ok(()) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Accepted,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up accepted"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Coalesced,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up coalesced"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Unavailable,
            );
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable")
        }
    }
}

async fn runtime_authorized_coordinator_for_height(
    storage: &MemoryStorage,
    legacy_coordinator: Option<[u8; 32]>,
    height: u64,
) -> Result<Option<[u8; 32]>, String> {
    // [AUTHORITY-SCHEDULE-RUNTIME 2026-08-14 by Codex] Preserve the legacy
    // static pin when no authority root is configured. Once enabled, only the
    // fully audited append-only schedule may authorise a proposer.
    if storage.record_commitment_authority_enforced() {
        storage.record_commitment_authority_for_height(height).await
    } else {
        Ok(legacy_coordinator)
    }
}

async fn runtime_authorized_coordinator_for_next_height(
    storage: &MemoryStorage,
    legacy_coordinator: Option<[u8; 32]>,
) -> Result<Option<[u8; 32]>, String> {
    if storage.record_commitment_authority_enforced() {
        Ok(storage
            .record_commitment_authority_state()
            .await?
            .map(|authority| authority.coordinator))
    } else {
        Ok(legacy_coordinator)
    }
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

fn commitment_peer_descriptor(
    peer_store: &PeerStore,
    node_id: &[u8; 32],
    now: u64,
    policy: CommitmentPeerDescriptorPolicy,
) -> Option<SignedNodeDescriptor> {
    peer_store.get_valid(node_id, now).or_else(|| {
        (policy == CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness)
            .then(|| peer_store.get_signature_verified_cached(node_id))
            .flatten()
    })
}

/// Republishes this coordinator's current signed descriptor to pinned witnesses.
///
/// This bounded compatibility preflight runs before strict startup checkpoint
/// and coordinator-lease gates. It may use an authentic expired descriptor as
/// a transport hint for an exact operator-pinned witness, but it sends only the
/// coordinator's public signed discovery descriptor. Witness acceptance grants
/// no authority and cannot bypass the subsequent signed protocol gates.
pub async fn publish_current_descriptor_to_commitment_witnesses(
    peer_store: &PeerStore,
    self_descriptor: &SignedNodeDescriptor,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
) -> CommitmentWitnessDescriptorPublishRound {
    publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
        peer_store,
        self_descriptor,
        client,
        witness_node_ids,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    self_descriptor: &SignedNodeDescriptor,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    endpoint_allowed: &F,
) -> CommitmentWitnessDescriptorPublishRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    // [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] Preserve first-seen
    // operator order while deduplicating and enforcing the existing protocol
    // fan-out cap. A malformed configuration cannot amplify startup traffic.
    let mut distinct_witnesses = Vec::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    let mut seen = HashSet::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    let self_node_id = self_descriptor.node_id();
    for witness in witness_node_ids {
        if *witness == self_node_id || !seen.insert(*witness) {
            continue;
        }
        distinct_witnesses.push(*witness);
        if distinct_witnesses.len() == MAX_PINNED_WITNESSES_PER_ROUND {
            break;
        }
    }

    let now = now_secs();
    let configured = distinct_witnesses.len();
    let mut urls = Vec::with_capacity(configured);
    for witness in distinct_witnesses {
        let Some(descriptor) = commitment_peer_descriptor(
            peer_store,
            &witness,
            now,
            CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        ) else {
            continue;
        };
        let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
            continue;
        };
        if !endpoint_allowed(endpoint) {
            continue;
        }
        let Ok(url) = canonical_peer_http_url(endpoint, "/api/discovery/gossip") else {
            continue;
        };
        urls.push(url);
    }

    let attempted = urls.len();
    let accepted = futures::stream::iter(urls)
        .map(|url| {
            let message = NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: self_descriptor.clone(),
            };
            async move {
                let Ok(response) = client.post(url).json(&message).send().await else {
                    return false;
                };
                if !response.status().is_success() {
                    return false;
                }
                let Ok(body) =
                    read_bounded_http_response(response, MAX_DESCRIPTOR_PREFLIGHT_RESPONSE_BYTES)
                        .await
                else {
                    return false;
                };
                let Ok(receipt) = serde_json::from_slice::<GossipResponse>(&body) else {
                    return false;
                };
                receipt.applied.total == 1
                    && receipt.applied.stale == 0
                    && receipt.applied.rejected == 0
                    && receipt
                        .applied
                        .inserted
                        .saturating_add(receipt.applied.unchanged)
                        == 1
            }
        })
        .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
        .filter(|accepted| std::future::ready(*accepted))
        .count()
        .await;

    CommitmentWitnessDescriptorPublishRound {
        configured,
        attempted,
        accepted,
        failed: configured.saturating_sub(accepted),
    }
}

/// Obtains and verifies one signed chain-checkpoint comparison from the pinned
/// coordinator. The response proves peer attestation, not network consensus.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, its response is invalid,
/// or durable checkpoint evidence cannot be stored.
pub async fn pull_record_commitment_checkpoint(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCheckpointOutcome, String> {
    pull_record_commitment_checkpoint_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        false,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn pull_record_commitment_checkpoint_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    track_trusted_witness_incidents: bool,
    descriptor_policy: CommitmentPeerDescriptorPolicy,
    endpoint_allowed: &F,
) -> Result<CommitmentCheckpointOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let coordinator = commitment_peer_descriptor(
        peer_store,
        coordinator_node_id,
        request_timestamp,
        descriptor_policy,
    )
    .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_url(endpoint)?;

    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("checkpoint_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "checkpoint_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let observed_at = now_secs();
    let outcome = verify_record_commitment_checkpoint(
        storage,
        &body,
        &request_id,
        coordinator_node_id,
        (known_tip_height, known_tip_hash),
        observed_at,
    )
    .await?;
    let persist_outcome = storage
        .persist_record_commitment_checkpoint_evidence_with_witness_policy(
            observed_at,
            outcome.relation.as_str(),
            outcome.local_tip_height,
            outcome.remote_tip_height,
            outcome.checkpoint_height,
            &outcome.evidence_digest,
            &body,
            track_trusted_witness_incidents,
        )
        .await
        .map_err(|_| "checkpoint_evidence_persist_failed".to_string())?;
    if persist_outcome == RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected {
        return Err("checkpoint_witness_equivocation".to_string());
    }
    Ok(outcome)
}

/// Requests and verifies one short-lived lease from an operator-pinned witness.
///
/// The response authorizes only `instance_id` and the exact audited local tip.
/// It does not expose the current holder when contended and does not establish
/// permissionless consensus, fork choice, or finality.
///
/// # Errors
///
/// Returns a stable privacy-safe code for invalid policy, peer admission,
/// unsafe endpoint, contention, transport failure, stale response, tip
/// mismatch, or signature failure.
pub async fn request_record_commitment_coordinator_lease(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    request_record_commitment_coordinator_lease_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        requested_ttl_secs,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Releases one previously acquired witness lease during graceful shutdown.
///
/// A failed or partial release is safe: the unreleased witnesses retain their
/// short expiry and the next process remains fail-closed until it can acquire
/// every configured grant.
pub async fn release_record_commitment_coordinator_lease(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    release_record_commitment_coordinator_lease_with_endpoint_policy(
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Builds a bounded local eligibility plan for custody-audit witnesses.
///
/// This function performs no network I/O and does not construct an anchor.
/// It evaluates only operator pins against the current authenticated PeerStore
/// view and the production public-endpoint policy.
///
/// # Errors
///
/// Returns a stable error when the local threshold is zero, exceeds the hard
/// witness bound, or the caller bypasses configuration validation with too
/// many pins.
pub fn plan_custody_audit_witnesses(
    peer_store: &PeerStore,
    producer_node_id: &[u8; 32],
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    now: u64,
) -> Result<CustodyAuditWitnessPlan, String> {
    plan_custody_audit_witnesses_with_endpoint_policy(
        peer_store,
        producer_node_id,
        witness_node_ids,
        minimum_verified,
        now,
        &commitment_peer_endpoint_is_public,
    )
}

fn plan_custody_audit_witnesses_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    producer_node_id: &[u8; 32],
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    now: u64,
    endpoint_allowed: &F,
) -> Result<CustodyAuditWitnessPlan, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if minimum_verified == 0 || minimum_verified > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_minimum_invalid".to_string());
    }
    if witness_node_ids.len() > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_pin_limit_exceeded".to_string());
    }

    let mut plan = CustodyAuditWitnessPlan {
        minimum_verified,
        ..CustodyAuditWitnessPlan::default()
    };
    let mut distinct = HashSet::with_capacity(witness_node_ids.len());
    for witness_node_id in witness_node_ids {
        if witness_node_id == producer_node_id {
            plan.self_excluded = plan.self_excluded.saturating_add(1);
            continue;
        }
        if !distinct.insert(*witness_node_id) {
            plan.duplicates_ignored = plan.duplicates_ignored.saturating_add(1);
            continue;
        }
        plan.configured = plan.configured.saturating_add(1);

        let eligible = peer_store
            .get_valid(witness_node_id, now)
            .is_some_and(|peer| {
                peer.descriptor
                    .capabilities
                    .contains(&NodeCapability::EncryptedStorage)
                    && peer
                        .descriptor
                        .public_endpoint
                        .as_deref()
                        .is_some_and(|endpoint| {
                            endpoint_allowed(endpoint)
                                && custody_audit_anchor_witness_url(endpoint).is_ok()
                        })
            });
        if eligible {
            plan.eligible = plan.eligible.saturating_add(1);
        } else {
            plan.unavailable = plan.unavailable.saturating_add(1);
        }
    }
    plan.quorum_ready = plan.configured >= minimum_verified && plan.eligible >= minimum_verified;
    Ok(plan)
}

/// Sends one signed aggregate-only cache anchor to operator-pinned witnesses.
///
/// Witnesses are discovery-admitted and endpoint-pinned on every request. The
/// returned counters are operational evidence only; they are not consensus,
/// finality, voting weight, or a source of user-visible delivery statistics.
///
/// # Errors
///
/// Returns an error only when the local anchor input is structurally invalid.
/// Individual witness failures are counted in the bounded round result.
pub async fn witness_verified_delivery_anchor(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    generation: u64,
    anchor_digest: &[u8; 32],
) -> Result<VerifiedDeliveryAnchorWitnessRound, String> {
    witness_verified_delivery_anchor_with_endpoint_policy(
        peer_store,
        identity,
        client,
        witness_node_ids,
        generation,
        anchor_digest,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn witness_verified_delivery_anchor_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    generation: u64,
    anchor_digest: &[u8; 32],
    endpoint_allowed: &F,
) -> Result<VerifiedDeliveryAnchorWitnessRound, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if generation == 0 || generation > i64::MAX as u64 {
        return Err("verified_delivery_witness_generation_invalid".to_string());
    }
    if anchor_digest == &[0u8; 32] {
        return Err("verified_delivery_witness_digest_invalid".to_string());
    }

    let self_node_id = identity.public_key_bytes();
    let mut distinct = HashSet::new();
    let witnesses = witness_node_ids
        .iter()
        .copied()
        .filter(|node_id| *node_id != self_node_id && distinct.insert(*node_id))
        .take(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>();
    let mut round = VerifiedDeliveryAnchorWitnessRound {
        configured: witnesses.len(),
        ..VerifiedDeliveryAnchorWitnessRound::default()
    };

    for witness_node_id in witnesses {
        let request_timestamp = now_secs();
        let Some(peer) = peer_store.get_valid(&witness_node_id, request_timestamp) else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };
        let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };
        if !endpoint_allowed(endpoint) {
            round.failed = round.failed.saturating_add(1);
            continue;
        }
        let Ok(url) = verified_delivery_anchor_witness_url(endpoint) else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };

        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
            &self_node_id,
            generation,
            anchor_digest,
            &request_id,
            request_timestamp,
        );
        let request = MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            requester: self_node_id,
            generation,
            anchor_digest: *anchor_digest,
            request_id,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = match encode_memchain(&request) {
            Ok(frame) => frame,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        round.attempted = round.attempted.saturating_add(1);
        let response = match client
            .post(url)
            .header("content-type", "application/octet-stream")
            .body(frame)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => response,
            Ok(_) | Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        let body = match read_bounded_response(response).await {
            Ok(body) => body,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        let outcome = match verify_delivery_anchor_witness_response(
            &body,
            &request_id,
            &self_node_id,
            generation,
            anchor_digest,
            &witness_node_id,
            now_secs(),
        ) {
            Ok(outcome) => outcome,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        round.verified = round.verified.saturating_add(1);
        match outcome {
            VERIFIED_DELIVERY_WITNESS_ADVANCED_V1 => {
                round.advanced = round.advanced.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1 => {
                round.idempotent = round.idempotent.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_STALE_V1 => round.stale = round.stale.saturating_add(1),
            VERIFIED_DELIVERY_WITNESS_CONFLICT_V1 => {
                round.conflicts = round.conflicts.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_GAP_V1 => round.gaps = round.gaps.saturating_add(1),
            _ => unreachable!("verified response outcome was validated"),
        }
    }
    Ok(round)
}

async fn release_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseRelease, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    // [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] The caller supplies an
    // operator-pinned witness identity and verifies the signed response against
    // that exact key, so an authentic expired descriptor is only an endpoint
    // recovery hint during cold start.
    let witness = commitment_peer_descriptor(
        peer_store,
        witness_node_id,
        request_timestamp,
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
    )
    .ok_or_else(|| "lease_release_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_release_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_release_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_release_url(endpoint)?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_release_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_release", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_release_not_holder".to_string());
    }
    if !response.status().is_success() {
        return Err(format!(
            "lease_release_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    verify_record_commitment_coordinator_lease_release_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        now_secs(),
    )
}

#[allow(clippy::too_many_arguments)]
async fn request_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseGrant, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
        .contains(&requested_ttl_secs)
    {
        return Err("lease_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    // [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] Lease acquisition has the
    // same explicit identity pin and signed-response boundary as startup
    // checkpoint reconciliation. Descriptor expiry cannot grant authority.
    let witness = commitment_peer_descriptor(
        peer_store,
        witness_node_id,
        request_timestamp,
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
    )
    .ok_or_else(|| "lease_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_request_encode_failed".to_string())?;
    let request_started = Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_request", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_contended".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("lease_http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let mut grant = verify_record_commitment_coordinator_lease_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        (known_tip_height, known_tip_hash),
        requested_ttl_secs,
        now_secs(),
    )?;
    grant.valid_for_secs = grant
        .valid_for_secs
        .saturating_sub(request_started.elapsed().as_secs());
    if grant.valid_for_secs == 0 {
        return Err("lease_expired_in_transit".to_string());
    }
    Ok(grant)
}

/// Pulls and imports one current-tip certificate from an admitted peer.
///
/// The serving peer is transport only. Every historical member must verify as
/// an exact signed checkpoint frame from a distinct identity in
/// `allowed_witnesses`. `minimum_required_signers` is the receiver's current
/// operator policy and cannot be downgraded by the serving peer. Callers must
/// use this after startup; a replayed bundle never replaces the live startup
/// witness round.
///
/// # Errors
///
/// Returns a stable privacy-safe code when peer admission, transport, outer
/// freshness, member signatures, operator pinning, digest, or durable storage
/// verification fails.
pub async fn pull_record_commitment_checkpoint_certificate(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
) -> Result<CommitmentCertificateImportOutcome, String> {
    pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
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

fn normalized_commitment_certificate_carriers(
    local_node_id: &[u8; 32],
    excluded_primary: Option<&[u8; 32]>,
    allowed_witnesses: &[[u8; 32]],
    max_carriers: usize,
) -> Vec<[u8; 32]> {
    let carrier_limit = max_carriers.min(MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1);
    let mut carriers = Vec::with_capacity(carrier_limit);
    for witness in allowed_witnesses
        .iter()
        .take(MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1)
    {
        if carriers.len() >= carrier_limit {
            break;
        }
        if witness == local_node_id
            || excluded_primary.is_some_and(|excluded| witness == excluded)
            || carriers.contains(witness)
        {
            continue;
        }
        carriers.push(*witness);
    }
    carriers
}

/// Runs one bounded, fail-closed certificate carrier sequence.
///
/// Availability is the only failure class that may advance to another exact
/// operator pin. Any verified response, including one made non-durable by a
/// concurrent local state change, stops the round. Security failures retain
/// their private stable code only long enough for the follower adapter to
/// preserve its existing API; the coordinator adapter collapses them before
/// returning to runtime logging.
///
/// [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex]
#[allow(clippy::too_many_arguments)]
async fn pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carriers: &[[u8; 32]],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierPullRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    circuit_breaker.align_slots(carriers.len());
    let mut carrier_attempts = 0usize;
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;

    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&minimum_required_signers)
        || allowed_witnesses.len() < minimum_required_signers
    {
        return CommitmentCertificateCarrierPullRound {
            terminal: CommitmentCertificateCarrierPullTerminal::SecurityStopped(
                "certificate_policy_invalid".to_string(),
            ),
            carrier_attempts,
            cooldown_skips,
            half_open_attempts,
        };
    }

    for (carrier_index, candidate) in carriers.iter().enumerate() {
        match circuit_breaker.decision(carrier_index, Instant::now()) {
            CommitmentCarrierCircuitDecision::Closed => {}
            CommitmentCarrierCircuitDecision::Cooling => {
                cooldown_skips = cooldown_skips.saturating_add(1);
                continue;
            }
            CommitmentCarrierCircuitDecision::HalfOpen => {
                half_open_attempts = half_open_attempts.saturating_add(1);
            }
        }
        carrier_attempts = carrier_attempts.saturating_add(1);

        match pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            candidate,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(imported) => {
                circuit_breaker.record_success(carrier_index);
                return CommitmentCertificateCarrierPullRound {
                    terminal: CommitmentCertificateCarrierPullTerminal::Imported(imported),
                    carrier_attempts,
                    cooldown_skips,
                    half_open_attempts,
                };
            }
            Err(error)
                if commitment_certificate_source_failure_class(&error)
                    == CommitmentCertificateSourceFailureClass::Availability =>
            {
                circuit_breaker.record_availability_failure(carrier_index, Instant::now());
            }
            Err(error) => {
                return CommitmentCertificateCarrierPullRound {
                    terminal: CommitmentCertificateCarrierPullTerminal::SecurityStopped(error),
                    carrier_attempts,
                    cooldown_skips,
                    half_open_attempts,
                };
            }
        }
    }

    CommitmentCertificateCarrierPullRound {
        terminal: CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted,
        carrier_attempts,
        cooldown_skips,
        half_open_attempts,
    }
}

/// Recovers post-startup certificate evidence from exact operator pins.
///
/// This coordinator-side path never grants startup authority, selects a chain,
/// or treats peer count as consensus. It returns only source-blind aggregates
/// suitable for runtime logs.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    max_carriers: usize,
    client: &reqwest::Client,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierRecoveryRound {
    recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        allowed_witnesses,
        minimum_required_signers,
        max_carriers,
        client,
        &commitment_peer_endpoint_is_public,
        circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    max_carriers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierRecoveryRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let carriers = normalized_commitment_certificate_carriers(
        &identity.public_key_bytes(),
        None,
        allowed_witnesses,
        max_carriers,
    );
    let pull_round =
        pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carriers,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
            circuit_breaker,
        )
        .await;
    let (disposition, checkpoint_height, signer_count, required_signers) =
        match pull_round.terminal {
            CommitmentCertificateCarrierPullTerminal::Imported(imported) if imported.persisted => (
                CommitmentCertificateCarrierRecoveryDisposition::Persisted,
                imported.checkpoint_height,
                imported.signer_count,
                imported.required_signers,
            ),
            CommitmentCertificateCarrierPullTerminal::Imported(imported) => (
                CommitmentCertificateCarrierRecoveryDisposition::VerifiedUnpersisted,
                imported.checkpoint_height,
                imported.signer_count,
                imported.required_signers,
            ),
            CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted => (
                CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted,
                0,
                0,
                0,
            ),
            CommitmentCertificateCarrierPullTerminal::SecurityStopped(_) => (
                CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped,
                0,
                0,
                0,
            ),
        };

    CommitmentCertificateCarrierRecoveryRound {
        disposition,
        checkpoint_height,
        signer_count,
        required_signers,
        carrier_attempts: pull_round.carrier_attempts,
        cooldown_skips: pull_round.cooldown_skips,
        half_open_attempts: pull_round.half_open_attempts,
        cooling_slots: circuit_breaker.cooling_slots(Instant::now()),
    }
}

/// Refreshes follower certificate evidence after signed tip convergence.
///
/// The follower's configured coordinator is transport only. The response must
/// still satisfy the receiver's witness allowlist and minimum threshold. A
/// threshold below two disables certificate replication for backward
/// compatibility; malformed enabled policy fails closed. If the coordinator is
/// unavailable, exact operator-pinned witnesses may carry the same immutable
/// certificate. They gain no authority over chain state or certificate policy.
///
/// # Errors
///
/// Returns a stable privacy-safe code when local policy, peer transport,
/// certificate verification, or durable storage validation fails.
pub async fn sync_follower_record_commitment_checkpoint_certificate(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String> {
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        &mut circuit_breaker,
    )
    .await
}

/// Refreshes follower certificate evidence with process-lifetime carrier state.
///
/// The caller must retain this circuit only for this exact follower policy
/// domain. The typed marker prevents block-page circuit state from being
/// passed here accidentally.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String> {
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        &commitment_peer_endpoint_is_public,
        circuit_breaker,
    )
    .await
}

#[cfg(test)]
async fn sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        endpoint_allowed,
        &mut circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if minimum_required_signers < 2 {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::Disabled,
        );
        return Ok(CommitmentFollowerCertificateSyncOutcome::PolicyDisabled);
    }
    if minimum_required_signers > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
        || allowed_witnesses.len() < minimum_required_signers
    {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::ConfigurationError,
        );
        return Err("certificate_policy_invalid".to_string());
    }
    if converged_tip_height == 0 {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::WaitingForConvergence,
        );
        return Err("certificate_local_tip_unavailable".to_string());
    }
    let (_, _, local_tip_height, _) = match storage
        .record_commitment_chain_checkpoint(converged_tip_height)
        .await
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            return Err(error);
        }
    };
    if local_tip_height != converged_tip_height {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::WaitingForConvergence,
        );
        return Err("certificate_converged_tip_changed".to_string());
    }

    let local_node_id = identity.public_key_bytes();
    let carriers = normalized_commitment_certificate_carriers(
        &local_node_id,
        Some(source_node_id),
        allowed_witnesses,
        MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1,
    );
    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Align before the
    // already-current check so a pin-count change cannot retain positional
    // circuit state even when no transport request is needed.
    circuit_breaker.align_slots(carriers.len());

    let certificate_already_current = match storage
        .record_commitment_checkpoint_certificate_satisfies_policy(
            converged_tip_height,
            allowed_witnesses,
            minimum_required_signers,
        )
        .await
    {
        Ok(current) => current,
        Err(error) => {
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            return Err(error);
        }
    };
    if certificate_already_current {
        record_commitment_certificate_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            0,
            0,
        );
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::Ready {
                tip_height: converged_tip_height,
            },
        );
        return Ok(CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent);
    }

    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Coordinator transport
    // remains first. Only its narrow availability class may enter the shared
    // carrier primitive; every other coordinator error stops immediately.
    let coordinator_availability_failure =
        match pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            source_node_id,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(imported) => {
                if imported.checkpoint_height != converged_tip_height {
                    record_commitment_certificate_carrier_circuit_telemetry(
                        storage,
                        circuit_breaker,
                        0,
                        0,
                    );
                    storage.record_commitment_certificate_sync_outcome(
                        now_secs(),
                        RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                        0,
                    );
                    storage.record_commitment_certificate_policy_evaluation(
                        now_secs(),
                        RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                    );
                    return Err("certificate_converged_tip_changed".to_string());
                }
                record_commitment_certificate_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    0,
                    0,
                );
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    follower_certificate_sync_disposition(
                        CommitmentFollowerCertificateSource::Coordinator,
                        imported.persisted,
                    ),
                    0,
                );
                let readiness = if imported.persisted {
                    RecordCommitmentCertificatePolicyReadiness::Ready {
                        tip_height: converged_tip_height,
                    }
                } else {
                    RecordCommitmentCertificatePolicyReadiness::WaitingForCertificate {
                        tip_height: converged_tip_height,
                    }
                };
                storage.record_commitment_certificate_policy_evaluation(now_secs(), readiness);
                return Ok(CommitmentFollowerCertificateSyncOutcome::Refreshed(
                    imported,
                ));
            }
            Err(error)
                if commitment_certificate_source_failure_class(&error)
                    == CommitmentCertificateSourceFailureClass::Availability =>
            {
                error
            }
            Err(error) => {
                record_commitment_certificate_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    0,
                    0,
                );
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                    0,
                );
                storage.record_commitment_certificate_policy_evaluation(
                    now_secs(),
                    RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                );
                return Err(error);
            }
        };

    let carrier_round =
        pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carriers,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
            circuit_breaker,
        )
        .await;
    record_commitment_certificate_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        carrier_round.cooldown_skips,
        carrier_round.half_open_attempts,
    );
    match carrier_round.terminal {
        CommitmentCertificateCarrierPullTerminal::Imported(imported) => {
            if imported.checkpoint_height != converged_tip_height {
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                    carrier_round.carrier_attempts,
                );
                storage.record_commitment_certificate_policy_evaluation(
                    now_secs(),
                    RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                );
                return Err("certificate_converged_tip_changed".to_string());
            }
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                follower_certificate_sync_disposition(
                    CommitmentFollowerCertificateSource::PinnedCarrier,
                    imported.persisted,
                ),
                carrier_round.carrier_attempts,
            );
            let readiness = if imported.persisted {
                RecordCommitmentCertificatePolicyReadiness::Ready {
                    tip_height: converged_tip_height,
                }
            } else {
                RecordCommitmentCertificatePolicyReadiness::WaitingForCertificate {
                    tip_height: converged_tip_height,
                }
            };
            storage.record_commitment_certificate_policy_evaluation(now_secs(), readiness);
            Ok(CommitmentFollowerCertificateSyncOutcome::Refreshed(
                imported,
            ))
        }
        CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted => {
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                RecordCommitmentCertificateSyncDisposition::AvailabilityExhausted,
                carrier_round.carrier_attempts,
            );
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SourceUnavailable {
                    tip_height: converged_tip_height,
                },
            );
            Err(coordinator_availability_failure)
        }
        CommitmentCertificateCarrierPullTerminal::SecurityStopped(error) => {
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                carrier_round.carrier_attempts,
            );
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            Err(error)
        }
    }
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

fn commitment_certificate_source_failure_class(
    error: &str,
) -> CommitmentCertificateSourceFailureClass {
    let retryable_status = error
        .strip_prefix("certificate_http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "certificate_source_unavailable"
                | "certificate_source_missing_endpoint"
                | "certificate_request_timeout"
                | "certificate_request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
        )
    {
        CommitmentCertificateSourceFailureClass::Availability
    } else {
        CommitmentCertificateSourceFailureClass::Security
    }
}

async fn pull_record_commitment_checkpoint_certificate_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCertificateImportOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&minimum_required_signers)
        || allowed_witnesses.len() < minimum_required_signers
    {
        return Err("certificate_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    let source = peer_store
        .get_valid(source_node_id, request_timestamp)
        .ok_or_else(|| "certificate_source_unavailable".to_string())?;
    let endpoint = source
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "certificate_source_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("certificate_source_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_certificate_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    if known_tip_height == 0 {
        return Err("certificate_local_tip_unavailable".to_string());
    }

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("certificate_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "certificate_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let verified = verify_checkpoint_certificate_response(
        &body,
        &request_id,
        source_node_id,
        (known_tip_height, known_tip_hash),
        allowed_witnesses,
        minimum_required_signers,
        now_secs(),
    )?;

    let mut evidence_digests = Vec::with_capacity(verified.members.len());
    for member in &verified.members {
        let relation = if member.remote_tip_height == verified.checkpoint_height {
            "converged"
        } else {
            "remote_ahead"
        };
        let persist_outcome = storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                member.observed_at,
                relation,
                verified.checkpoint_height,
                member.remote_tip_height,
                verified.checkpoint_height,
                &member.evidence_digest,
                &member.frame,
                true,
            )
            .await
            .map_err(|_| "certificate_member_persist_failed".to_string())?;
        if persist_outcome != RecordCommitmentCheckpointEvidencePersistOutcome::Stored {
            return Err("certificate_member_security_incident".to_string());
        }
        evidence_digests.push(member.evidence_digest);
    }
    let persisted = storage
        .persist_record_commitment_checkpoint_certificate(
            now_secs(),
            verified.required_signers,
            allowed_witnesses,
            &evidence_digests,
        )
        .await
        .map_err(|_| "certificate_persist_failed".to_string())?;
    Ok(CommitmentCertificateImportOutcome {
        checkpoint_height: verified.checkpoint_height,
        signer_count: verified.members.len(),
        required_signers: verified.required_signers,
        persisted,
    })
}

/// Collects a bounded set of signed checkpoint observations from discovered
/// encrypted-storage peers.
///
/// This is evidence collection for the current single-writer Block Sync v1
/// architecture, not distributed consensus. The function never adopts a
/// remote chain, changes the coordinator, selects a longest chain, or derives
/// truth from peer count. Every accepted response is independently verified
/// and durably stored by `pull_record_commitment_checkpoint`.
pub async fn reconcile_record_commitment_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        max_witnesses,
        commitment_peer_endpoint_is_public,
    )
    .await
}

/// Collects checkpoints only from explicit operator-pinned identities.
///
/// This is the trust boundary used by the coordinator startup guard. Signed
/// discovery still resolves endpoint rotation, but no unpinned permissionless
/// peer can become startup authority merely by advertising a capability.
pub async fn reconcile_record_commitment_pinned_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        2,
    )
    .await
}

/// Collects pinned witness proofs and attempts one immutable certificate using
/// the operator's configured minimum. Values below two preserve legacy
/// one-witness startup behavior but cannot be represented as a multi-witness
/// certificate.
pub async fn reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        minimum_certificate_signers,
        commitment_peer_endpoint_is_public,
    )
    .await
}

async fn reconcile_record_commitment_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidates: Vec<_> = peer_store
        .peers_with_capability(NodeCapability::EncryptedStorage, now)
        .into_iter()
        .filter(|candidate| candidate.descriptor.node_id != self_node_id)
        .filter(|candidate| {
            candidate
                .descriptor
                .public_endpoint
                .as_deref()
                .is_some_and(&endpoint_allowed)
        })
        .collect();
    candidates.sort_by_key(|candidate| candidate.descriptor.node_id);
    if !candidates.is_empty() {
        // Rotate the deterministic signed-descriptor order so a larger network
        // does not permanently starve peers beyond the per-round fan-out cap.
        // The selector is local scheduling state only and is never reported.
        let node_selector = u64::from_be_bytes(
            self_node_id[..8]
                .try_into()
                .expect("fixed identity prefix length"),
        );
        let offset =
            usize::try_from((node_selector ^ (now / 300)) % candidates.len() as u64).unwrap_or(0);
        candidates.rotate_left(offset);
    }

    let candidate_ids = candidates
        .into_iter()
        .map(|candidate| candidate.descriptor.node_id)
        .collect();
    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        max_witnesses,
        false,
        None,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &endpoint_allowed,
    )
    .await
}

pub(crate) async fn reconcile_record_commitment_pinned_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidate_ids = Vec::with_capacity(witness_node_ids.len());
    for node_id in witness_node_ids {
        if *node_id == self_node_id || candidate_ids.contains(node_id) {
            continue;
        }
        let Some(peer) = commitment_peer_descriptor(
            peer_store,
            node_id,
            now,
            CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        ) else {
            continue;
        };
        if peer
            .descriptor
            .public_endpoint
            .as_deref()
            .is_some_and(&endpoint_allowed)
        {
            candidate_ids.push(*node_id);
        }
    }

    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        witness_node_ids.len().min(MAX_PINNED_WITNESSES_PER_ROUND),
        true,
        Some(minimum_certificate_signers),
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        &endpoint_allowed,
    )
    .await
}

async fn reconcile_record_commitment_candidate_ids<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    mut candidate_ids: Vec<[u8; 32]>,
    max_witnesses: usize,
    track_trusted_witness_incidents: bool,
    minimum_certificate_signers: Option<usize>,
    descriptor_policy: CommitmentPeerDescriptorPolicy,
    endpoint_allowed: &F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    // Preserve operator order while ensuring one identity can contribute at
    // most one request, one result, and one certificate member. This remains
    // defense in depth even when config parsing already rejects duplicate IDs.
    let mut distinct_candidates = HashSet::with_capacity(candidate_ids.len());
    candidate_ids.retain(|candidate| distinct_candidates.insert(*candidate));
    let eligible_witnesses = candidate_ids.len();
    candidate_ids.truncate(max_witnesses);
    let attempted = candidate_ids.len();
    let mut outcome = CommitmentReconciliationOutcome {
        eligible_witnesses,
        attempted,
        ..CommitmentReconciliationOutcome::default()
    };
    let certificate_witnesses = candidate_ids.clone();
    let mut verified = Vec::with_capacity(attempted);
    let mut certificate_evidence = Vec::with_capacity(attempted);

    for candidate_node_id in candidate_ids {
        match pull_record_commitment_checkpoint_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &candidate_node_id,
            client,
            track_trusted_witness_incidents,
            descriptor_policy,
            endpoint_allowed,
        )
        .await
        {
            Ok(proof) => {
                outcome.verified = outcome.verified.saturating_add(1);
                match proof.relation {
                    CommitmentCheckpointRelation::Converged => {
                        outcome.converged = outcome.converged.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteAhead => {
                        outcome.remote_ahead = outcome.remote_ahead.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteBehind => {
                        outcome.remote_behind = outcome.remote_behind.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::Diverged => {
                        outcome.diverged = outcome.diverged.saturating_add(1);
                    }
                }
                if matches!(
                    proof.relation,
                    CommitmentCheckpointRelation::Converged
                        | CommitmentCheckpointRelation::RemoteAhead
                ) && proof.checkpoint_height == proof.local_tip_height
                {
                    certificate_evidence.push(proof.evidence_digest);
                }
                verified.push(proof);
            }
            Err(_) => {
                outcome.failed = outcome.failed.saturating_add(1);
            }
        }
    }

    let completed_at = now_secs();
    if let Some(configured_threshold) = minimum_certificate_signers {
        let threshold = configured_threshold
            .max(2)
            .min(MAX_PINNED_WITNESSES_PER_ROUND);
        outcome.certificate_signers = certificate_evidence.len();
        outcome.certificate_required_signers = threshold;
        if certificate_evidence.len() >= threshold {
            match storage
                .persist_record_commitment_checkpoint_certificate(
                    completed_at,
                    threshold,
                    &certificate_witnesses,
                    &certificate_evidence,
                )
                .await
            {
                Ok(persisted) => outcome.certificate_persisted = persisted,
                Err(_) => outcome.certificate_persistence_failed = true,
            }
        }
    }
    for _ in 0..outcome.failed {
        storage.record_commitment_checkpoint_failure(completed_at);
    }
    // Record valid proofs after failures and from least to most severe. A
    // partial transport failure must not hide valid evidence, while a signed
    // divergence must remain the final operator-visible state for the round.
    verified.sort_by_key(|proof| checkpoint_relation_priority(proof.relation));
    for proof in verified {
        storage.record_commitment_checkpoint_verified(
            completed_at,
            proof.relation.as_str(),
            proof.local_tip_height,
            proof.remote_tip_height,
        );
    }
    storage.record_commitment_checkpoint_witness_round(
        completed_at,
        outcome.eligible_witnesses,
        outcome.attempted,
        outcome.verified,
        outcome.failed,
        outcome.converged,
        outcome.remote_ahead,
        outcome.remote_behind,
        outcome.diverged,
    );

    outcome
}

/// Accepts only public IP literals for permissionless witness traffic.
///
/// A signed descriptor proves who advertised an endpoint, not that the target
/// is safe for this host to contact. Domain names are deliberately excluded to
/// prevent DNS rebinding; loopback, private, link-local, CGNAT, benchmark,
/// documentation, multicast, and reserved ranges are also rejected.
pub(crate) fn commitment_peer_endpoint_is_public(endpoint: &str) -> bool {
    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] MemChain and discovery
    // share one public-host policy so a future range update cannot leave one
    // permissionless transport less protected than another.
    peer_endpoint_is_public_ip(endpoint)
}

const fn checkpoint_relation_priority(relation: CommitmentCheckpointRelation) -> u8 {
    match relation {
        CommitmentCheckpointRelation::RemoteBehind => 0,
        CommitmentCheckpointRelation::Converged => 1,
        CommitmentCheckpointRelation::RemoteAhead => 2,
        CommitmentCheckpointRelation::Diverged => 3,
    }
}

/// Pulls, verifies, and atomically appends one bounded commitment-block page.
///
/// The coordinator identity is supplied by validated operator configuration.
/// Discovery is used only to resolve that exact identity's current signed
/// endpoint; this function never selects or falls back to another peer.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, the signed page is invalid,
/// or the atomic local append fails closed.
pub async fn pull_record_commitment_page(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentSyncPageOutcome, String> {
    pull_record_commitment_page_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Pulls at most one exact-next coordinator handover proof from the currently
/// authorised coordinator and persists it only at its activation boundary.
///
/// [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] The responder is merely a
/// transport source. Authority comes exclusively from the dual-signed proof,
/// the immutable local authority root, and the audited local block prefix.
/// A future proof is returned only as a page boundary so the caller can catch
/// up without accepting coordinator authority early.
pub async fn sync_next_record_coordinator_handover(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
) -> Result<CommitmentAuthoritySyncOutcome, String> {
    let mut cursor = CommitmentAuthorityCarrierCursor::default();
    let mut circuit_breaker = CommitmentAuthorityCarrierCircuitBreaker::default();
    sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        &[],
        client,
        &commitment_peer_endpoint_is_public,
        &mut cursor,
        &mut circuit_breaker,
    )
    .await
}

/// Synchronizes one authority proof with bounded operator-pinned recovery.
///
/// Direct coordinator transport is always attempted first. Only explicit
/// availability failures may advance to a carrier, and carriers transport but
/// never authorise the independently dual-signed transition.
pub(crate) async fn sync_next_record_coordinator_handover_with_carrier_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carrier_node_ids: &[[u8; 32]],
    client: &reqwest::Client,
    cursor: &mut CommitmentAuthorityCarrierCursor,
    circuit_breaker: &mut CommitmentAuthorityCarrierCircuitBreaker,
) -> Result<CommitmentAuthoritySyncOutcome, String> {
    sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        carrier_node_ids,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
        circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn sync_record_coordinator_handover_from_source_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    expected_authority: &RecordCommitmentAuthorityState,
    response_signer: &[u8; 32],
    source: CommitmentAuthoritySyncSource,
    carrier_attempts: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    if authority != *expected_authority {
        return Err("handover_local_authority_changed".to_string());
    }
    if authority.coordinator == identity.public_key_bytes() {
        return Err("active_coordinator_cannot_follow_itself".to_string());
    }

    let request_timestamp = now_secs();
    let (unavailable_error, missing_endpoint_error, unsafe_endpoint_error) = match source {
        CommitmentAuthoritySyncSource::Coordinator => (
            "active_coordinator_unavailable",
            "active_coordinator_missing_endpoint",
            "active_coordinator_unsafe_endpoint",
        ),
        CommitmentAuthoritySyncSource::PinnedCarrier => (
            "handover_carrier_unavailable",
            "handover_carrier_missing_endpoint",
            "handover_carrier_unsafe_endpoint",
        ),
    };
    let responder = peer_store
        .get_valid(response_signer, request_timestamp)
        .ok_or_else(|| unavailable_error.to_string())?;
    let endpoint = responder
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| missing_endpoint_error.to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err(unsafe_endpoint_error.to_string());
    }
    let url = commitment_coordinator_handover_url(endpoint)?;

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_coordinator_handover_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        authority.authority_epoch,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        after_authority_epoch: authority.authority_epoch,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame =
        encode_memchain(&request).map_err(|_| "handover_request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("handover_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "handover_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let verified = verify_record_coordinator_handover_response(
        &body,
        &request_id,
        response_signer,
        &authority.coordinator,
        authority.authority_epoch,
        authority.next_block_height,
        now_secs(),
    )?;

    if source == CommitmentAuthoritySyncSource::PinnedCarrier && verified.handover.is_none() {
        // A carrier's empty local head cannot prove the active coordinator
        // made no transition; another exact operator pin may be less stale.
        return Err("handover_carrier_behind".to_string());
    }

    let mut handover_inserted = false;
    let mut pending_activation_height = None;
    if let Some(handover) = verified.handover {
        if handover.header.activation_height == authority.next_block_height {
            handover_inserted = matches!(
                storage
                    .persist_configured_record_coordinator_handover(&handover, now_secs())
                    .await?,
                RecordCoordinatorHandoverPersistOutcome::Inserted
            );
        } else {
            pending_activation_height = Some(handover.header.activation_height);
        }
    }

    let refreshed = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    Ok(CommitmentAuthoritySyncOutcome {
        authority_epoch: refreshed.authority_epoch,
        active_coordinator: refreshed.coordinator,
        next_block_height: refreshed.next_block_height,
        pending_activation_height,
        handover_inserted,
        source,
        carrier_attempts,
    })
}

#[cfg(test)]
async fn sync_next_record_coordinator_handover_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    sync_record_coordinator_handover_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        &authority,
        &authority.coordinator,
        CommitmentAuthoritySyncSource::Coordinator,
        0,
        client,
        endpoint_allowed,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carrier_node_ids: &[[u8; 32]],
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentAuthorityCarrierCursor,
    circuit_breaker: &mut CommitmentAuthorityCarrierCircuitBreaker,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    if authority.coordinator == identity.public_key_bytes() {
        return Err("active_coordinator_cannot_follow_itself".to_string());
    }

    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Recovery membership is
    // exactly the operator pin set. Discovery resolves fresh endpoints only;
    // it cannot nominate a transport source or alter proof authority.
    let carriers = eligible_pinned_commitment_carriers(
        identity.public_key_bytes(),
        &authority.coordinator,
        carrier_node_ids,
    );
    circuit_breaker.align_slots(carriers.len());
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;
    let direct = sync_record_coordinator_handover_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        &authority,
        &authority.coordinator,
        CommitmentAuthoritySyncSource::Coordinator,
        0,
        client,
        endpoint_allowed,
    )
    .await;
    let direct_error = match direct {
        Ok(outcome) => {
            cursor.reset();
            record_commitment_authority_carrier_circuit_telemetry(
                storage,
                circuit_breaker,
                cooldown_skips,
                half_open_attempts,
            );
            storage.record_commitment_authority_sync_outcome(
                now_secs(),
                RecordCommitmentAuthoritySyncDisposition::Coordinator,
                0,
            );
            return Ok(outcome);
        }
        Err(error) => error,
    };
    if coordinator_handover_source_failure_class(&direct_error)
        == CommitmentAuthoritySourceFailureClass::Security
    {
        record_commitment_authority_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_authority_sync_outcome(
            now_secs(),
            RecordCommitmentAuthoritySyncDisposition::SecurityStopped,
            0,
        );
        return Err(direct_error);
    }

    let carrier_count = carriers.len();
    let start_index = cursor.start_index(carrier_count);
    let mut carrier_attempts = 0usize;
    for offset in 0..carrier_count {
        let carrier_index = start_index.saturating_add(offset) % carrier_count;
        match circuit_breaker.decision(carrier_index, Instant::now()) {
            CommitmentCarrierCircuitDecision::Closed => {}
            CommitmentCarrierCircuitDecision::Cooling => {
                cooldown_skips = cooldown_skips.saturating_add(1);
                continue;
            }
            CommitmentCarrierCircuitDecision::HalfOpen => {
                half_open_attempts = half_open_attempts.saturating_add(1);
            }
        }
        let carrier = carriers[carrier_index];
        carrier_attempts = carrier_attempts.saturating_add(1);
        match sync_record_coordinator_handover_from_source_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &authority,
            &carrier,
            CommitmentAuthoritySyncSource::PinnedCarrier,
            carrier_attempts,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(outcome) => {
                circuit_breaker.record_success(carrier_index);
                cursor.prefer(carrier_index, carrier_count);
                record_commitment_authority_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_authority_sync_outcome(
                    now_secs(),
                    RecordCommitmentAuthoritySyncDisposition::CarrierRecovered,
                    carrier_attempts,
                );
                return Ok(outcome);
            }
            Err(error)
                if coordinator_handover_source_failure_class(&error)
                    == CommitmentAuthoritySourceFailureClass::Availability =>
            {
                circuit_breaker.record_availability_failure(carrier_index, Instant::now());
                cursor.advance_after_availability_failure(carrier_index, carrier_count);
            }
            Err(error) => {
                record_commitment_authority_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_authority_sync_outcome(
                    now_secs(),
                    RecordCommitmentAuthoritySyncDisposition::SecurityStopped,
                    carrier_attempts,
                );
                return Err(error);
            }
        }
    }

    // Preserve the direct source's stable availability code for existing
    // follower alerting when every bounded carrier is unavailable or behind.
    record_commitment_authority_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        cooldown_skips,
        half_open_attempts,
    );
    storage.record_commitment_authority_sync_outcome(
        now_secs(),
        RecordCommitmentAuthoritySyncDisposition::AvailabilityExhausted,
        carrier_attempts,
    );
    Err(direct_error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentAuthoritySourceFailureClass {
    Availability,
    Security,
}

fn coordinator_handover_source_failure_class(error: &str) -> CommitmentAuthoritySourceFailureClass {
    let retryable_status = error
        .strip_prefix("handover_http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "active_coordinator_unavailable"
                | "active_coordinator_missing_endpoint"
                | "handover_carrier_unavailable"
                | "handover_carrier_missing_endpoint"
                | "handover_carrier_behind"
                | "handover_request_timeout"
                | "handover_request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
        )
    {
        CommitmentAuthoritySourceFailureClass::Availability
    } else {
        CommitmentAuthoritySourceFailureClass::Security
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_record_coordinator_handover_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    expected_previous_coordinator: &[u8; 32],
    expected_authority_epoch: u64,
    expected_next_block_height: u64,
    now: u64,
) -> Result<VerifiedCoordinatorHandoverResponse, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_handover_response_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_handover_response_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_handover_response_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_handover_response".to_string());
    }
    let MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        handover,
        latest_authority_epoch,
        signature,
    } = response
    else {
        return Err("unexpected_handover_response".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("handover_response_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("handover_response_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("handover_response_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_handover_response".to_string());
    }
    if latest_authority_epoch < expected_authority_epoch {
        return Err("handover_history_rollback".to_string());
    }
    let signing_bytes = record_coordinator_handover_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        handover.as_ref(),
        latest_authority_epoch,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_handover_response_signature".to_string())?;

    match handover.as_ref() {
        Some(proof) => {
            proof
                .verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID)
                .map_err(|_| "invalid_handover_proof".to_string())?;
            let expected_epoch = expected_authority_epoch
                .checked_add(1)
                .ok_or_else(|| "authority_epoch_exhausted".to_string())?;
            if proof.header.authority_epoch != expected_epoch {
                return Err("handover_epoch_discontinuity".to_string());
            }
            if proof.header.previous_coordinator != *expected_previous_coordinator {
                return Err("handover_previous_coordinator_mismatch".to_string());
            }
            if proof.header.activation_height < expected_next_block_height {
                return Err("handover_activation_rollback".to_string());
            }
            if latest_authority_epoch < proof.header.authority_epoch {
                return Err("handover_history_head_mismatch".to_string());
            }
        }
        None if latest_authority_epoch != expected_authority_epoch => {
            return Err("handover_proof_omitted".to_string());
        }
        None => {}
    }

    Ok(VerifiedCoordinatorHandoverResponse {
        handover,
        latest_authority_epoch,
    })
}

/// Pulls one coordinator-authored page with bounded pinned-carrier recovery.
///
/// The coordinator is always attempted first. Carrier recovery is enabled only
/// when the local follower requires at least two witness signatures and has
/// enough configured witness pins to satisfy that policy. A carrier signs the
/// page envelope but every block must remain signed by `coordinator_node_id`.
/// Only classified availability failures may advance to another source.
///
/// # Errors
///
/// Returns the coordinator's stable availability code when every bounded
/// source is unavailable. Any endpoint-policy, decoding, identity, signature,
/// proposer, continuity, rollback, pagination, or storage failure stops closed
/// before another source can mask the incident.
pub async fn pull_record_commitment_page_with_carrier_recovery(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    let mut cursor = CommitmentBlockCarrierCursor::default();
    pull_record_commitment_page_with_carrier_cursor(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &mut cursor,
    )
    .await
}

/// Pulls one page while preserving a successful carrier preference within one
/// caller-owned multi-page synchronization round.
///
/// The cursor never weakens coordinator-first behavior or source validation.
/// It only avoids retrying earlier availability failures before a carrier that
/// already delivered a fully verified page in the same round.
pub(crate) async fn pull_record_commitment_page_with_carrier_cursor(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    cursor: &mut CommitmentBlockCarrierCursor,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
    )
    .await
}

/// Pulls one verified page with an exact caller-selected hard limit.
///
/// The follower uses this only to stop immediately before a signed authority
/// transition. Existing callers retain the full protocol page size.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pull_record_commitment_page_with_carrier_runtime_bounded(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    cursor: &mut CommitmentBlockCarrierCursor,
    circuit_breaker: &mut CommitmentBlockCarrierCircuitBreaker,
    max_blocks: u16,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
        circuit_breaker,
        max_blocks,
    )
    .await
}

async fn pull_record_commitment_page_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentSyncPageOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    pull_record_commitment_page_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        coordinator_node_id,
        client,
        endpoint_allowed,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
async fn pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut cursor = CommitmentBlockCarrierCursor::default();
    pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        endpoint_allowed,
        &mut cursor,
    )
    .await
}

fn eligible_pinned_commitment_carriers(
    local_node_id: [u8; 32],
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
) -> Vec<[u8; 32]> {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Block and handover
    // recovery share one immutable pin normalizer. A typed cursor may change
    // only the bounded attempt start inside this exact list; it cannot import
    // a discovery peer, include self/primary, or alter membership.
    let mut carriers = Vec::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    for carrier in carrier_node_ids {
        if *carrier == local_node_id || carrier == coordinator_node_id || carriers.contains(carrier)
        {
            continue;
        }
        carriers.push(*carrier);
        if carriers.len() == MAX_PINNED_WITNESSES_PER_ROUND {
            break;
        }
    }
    carriers
}

#[allow(clippy::too_many_arguments)]
async fn pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentBlockCarrierCursor,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();
    pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        endpoint_allowed,
        cursor,
        &mut circuit_breaker,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentBlockCarrierCursor,
    circuit_breaker: &mut CommitmentBlockCarrierCircuitBreaker,
    max_blocks: u16,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    // [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Align before the
    // coordinator request so an operator pin-count change clears positional
    // state even when the direct path succeeds and no carrier is contacted.
    let carriers = eligible_pinned_commitment_carriers(
        identity.public_key_bytes(),
        coordinator_node_id,
        carrier_node_ids,
    );
    circuit_breaker.align_slots(carriers.len());
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;

    // [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] Every terminal
    // path records one typed aggregate disposition. Recording remains inside
    // this direct-first primitive so future callers cannot omit or reinterpret
    // the source budget; the storage contract discards all source details.
    let direct = pull_record_commitment_page_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        coordinator_node_id,
        client,
        endpoint_allowed,
        max_blocks,
    )
    .await;
    let direct_error = match direct {
        Ok(page) => {
            cursor.reset();
            record_commitment_block_carrier_circuit_telemetry(
                storage,
                circuit_breaker,
                cooldown_skips,
                half_open_attempts,
            );
            storage.record_commitment_block_page_pull_outcome(
                now_secs(),
                RecordCommitmentBlockPagePullDisposition::Coordinator,
                0,
            );
            return Ok(CommitmentFollowerPagePullOutcome {
                page,
                source: CommitmentSyncPageSource::Coordinator,
                carrier_attempts: 0,
            });
        }
        Err(error) => error,
    };

    if commitment_block_source_failure_class(&direct_error)
        == CommitmentBlockSourceFailureClass::Security
    {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err(direct_error);
    }
    if minimum_required_signers < 2 {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted,
            0,
        );
        return Err(direct_error);
    }
    if minimum_required_signers > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1 {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err("block_carrier_policy_invalid".to_string());
    }

    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Preserve operator order,
    // exclude self/coordinator, deduplicate, and enforce the same hard fan-out
    // cap as witness operations. Discovery never chooses a recovery source.
    if carriers.len() < minimum_required_signers {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err("block_carrier_policy_invalid".to_string());
    }

    let mut carrier_attempts = 0usize;
    let carrier_count = carriers.len();
    let start_index = cursor.start_index(carrier_count);
    for offset in 0..carrier_count {
        let carrier_index = start_index.saturating_add(offset) % carrier_count;
        match circuit_breaker.decision(carrier_index, Instant::now()) {
            CommitmentCarrierCircuitDecision::Closed => {}
            CommitmentCarrierCircuitDecision::Cooling => {
                cooldown_skips = cooldown_skips.saturating_add(1);
                continue;
            }
            CommitmentCarrierCircuitDecision::HalfOpen => {
                half_open_attempts = half_open_attempts.saturating_add(1);
            }
        }
        let carrier = carriers[carrier_index];
        carrier_attempts = carrier_attempts.saturating_add(1);
        match pull_record_commitment_page_from_source_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carrier,
            coordinator_node_id,
            client,
            endpoint_allowed,
            max_blocks,
        )
        .await
        {
            Ok(page) => {
                circuit_breaker.record_success(carrier_index);
                cursor.prefer(carrier_index, carrier_count);
                record_commitment_block_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_block_page_pull_outcome(
                    now_secs(),
                    RecordCommitmentBlockPagePullDisposition::CarrierRecovered,
                    carrier_attempts,
                );
                return Ok(CommitmentFollowerPagePullOutcome {
                    page,
                    source: CommitmentSyncPageSource::PinnedCarrier,
                    carrier_attempts,
                });
            }
            Err(error)
                if commitment_block_source_failure_class(&error)
                    == CommitmentBlockSourceFailureClass::Availability =>
            {
                circuit_breaker
                    .record_availability_failure(carrier_index, Instant::now());
                cursor.advance_after_availability_failure(carrier_index, carrier_count);
            }
            Err(error) => {
                record_commitment_block_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_block_page_pull_outcome(
                    now_secs(),
                    RecordCommitmentBlockPagePullDisposition::SecurityStopped,
                    carrier_attempts,
                );
                return Err(error);
            }
        }
    }

    // Preserve the coordinator's established privacy-safe code so existing
    // operations and alerting remain backward compatible.
    record_commitment_block_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        cooldown_skips,
        half_open_attempts,
    );
    storage.record_commitment_block_page_pull_outcome(
        now_secs(),
        RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted,
        carrier_attempts,
    );
    Err(direct_error)
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

fn commitment_block_source_failure_class(error: &str) -> CommitmentBlockSourceFailureClass {
    let retryable_status = error
        .strip_prefix("http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "pinned_coordinator_unavailable"
                | "pinned_coordinator_missing_endpoint"
                | "request_timeout"
                | "request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
                | "carrier_tip_behind"
        )
    {
        CommitmentBlockSourceFailureClass::Availability
    } else {
        CommitmentBlockSourceFailureClass::Security
    }
}

#[allow(clippy::too_many_arguments)]
async fn pull_record_commitment_page_from_source_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    response_signer_node_id: &[u8; 32],
    expected_proposer_node_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
    max_blocks: u16,
) -> Result<CommitmentSyncPageOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let source = peer_store
        .get_valid(response_signer_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = source
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_block_range_url(endpoint)?;

    let local_tip = verified_local_commitment_tip(storage).await?;
    let from_height = local_tip.0.saturating_add(1).max(1);
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    let limit = max_blocks;
    let signing_bytes = record_block_range_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("request", &error))?;
    if !response.status().is_success() {
        return Err(format!("http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let page = verify_record_commitment_page(
        &body,
        &request_id,
        response_signer_node_id,
        expected_proposer_node_id,
        local_tip,
        now_secs(),
    )?;
    if page.blocks.len() > usize::from(max_blocks) {
        return Err("response_page_exceeds_request".to_string());
    }

    let append = storage
        .append_record_commitment_blocks_atomic(&page.blocks, Some(expected_proposer_node_id))
        .await
        .map_err(|_| "storage_append_rejected".to_string())?;

    Ok(CommitmentSyncPageOutcome {
        inserted: append.inserted,
        already_present: append.already_present,
        has_more: page.has_more,
        remote_tip_height: page.tip_height,
    })
}

fn commitment_block_range_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-range")
}

fn commitment_block_announce_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-announce")
}

fn commitment_checkpoint_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint")
}

fn commitment_checkpoint_certificate_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint-certificate")
}

fn commitment_coordinator_handover_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-handover")
}

fn commitment_coordinator_lease_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease")
}

fn commitment_coordinator_lease_release_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease/release")
}

fn verified_delivery_anchor_witness_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(
        endpoint,
        "/api/discovery/peer/verified-delivery-anchor-witness",
    )
}

fn custody_audit_anchor_witness_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/custody-audit-anchor-witness")
}

pub(crate) fn commitment_peer_url(endpoint: &str, path: &str) -> Result<Url, String> {
    canonical_peer_http_url(endpoint, path).map_err(|error| match error {
        PeerEndpointUrlError::Missing => "pinned_coordinator_missing_endpoint".to_string(),
        PeerEndpointUrlError::Invalid => "pinned_coordinator_invalid_endpoint".to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_record_commitment_coordinator_lease_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    expected_tip: (u64, [u8; 32]),
    requested_ttl_secs: u32,
    now: u64,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_lease_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_lease_frame")?;
    if canonical != body {
        return Err("noncanonical_lease_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        witness_tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_lease_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_witness_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_lease_response".to_string());
    }
    if (witness_tip_height, witness_tip_hash) != expected_tip {
        return Err("lease_tip_mismatch".to_string());
    }
    if lease_epoch == 0 || lease_expires_at <= now {
        return Err("lease_expiry_invalid".to_string());
    }
    let valid_for_secs = lease_expires_at
        .checked_sub(response_timestamp)
        .ok_or_else(|| "lease_expiry_invalid".to_string())?;
    // The signed remainder can be slightly shorter than the minimum request
    // TTL when persistence and response signing cross a second boundary.
    if valid_for_secs == 0 || valid_for_secs > u64::from(requested_ttl_secs) {
        return Err("lease_duration_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        &witness_tip_hash,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseGrant {
        lease_epoch,
        lease_expires_at,
        valid_for_secs,
    })
}

fn verify_record_commitment_coordinator_lease_release_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    now: u64,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_release_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_lease_release_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_lease_release_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_lease_release_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature,
    } = response
    else {
        return Err("unexpected_lease_release_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_release_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_release_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_release_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_release_witness_mismatch".to_string());
    }
    if lease_epoch == 0 || now.abs_diff(released_at) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("lease_release_timestamp_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_release_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseRelease {
        lease_epoch,
        released_at,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_delivery_anchor_witness_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_generation: u64,
    expected_anchor_digest: &[u8; 32],
    expected_witness: &[u8; 32],
    now: u64,
) -> Result<u8, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_delivery_witness_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_delivery_witness_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_delivery_witness_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_delivery_witness_frame".to_string());
    }
    let MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester,
        requested_generation,
        requested_anchor_digest,
        witness,
        response_timestamp,
        witness_generation,
        witness_anchor_digest,
        outcome,
        signature,
    } = response
    else {
        return Err("unexpected_delivery_witness_message".to_string());
    };
    if request_id != *expected_request_id
        || requester != *expected_requester
        || requested_generation != expected_generation
        || requested_anchor_digest != *expected_anchor_digest
    {
        return Err("delivery_witness_request_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("delivery_witness_identity_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || witness_generation == 0
        || witness_generation > i64::MAX as u64
        || witness_anchor_digest == [0u8; 32]
    {
        return Err("delivery_witness_state_invalid".to_string());
    }

    let relation_valid = match outcome {
        VERIFIED_DELIVERY_WITNESS_ADVANCED_V1 | VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1 => {
            witness_generation == expected_generation
                && witness_anchor_digest == *expected_anchor_digest
        }
        VERIFIED_DELIVERY_WITNESS_STALE_V1 => witness_generation > expected_generation,
        VERIFIED_DELIVERY_WITNESS_CONFLICT_V1 => {
            witness_generation == expected_generation
                && witness_anchor_digest != *expected_anchor_digest
        }
        VERIFIED_DELIVERY_WITNESS_GAP_V1 => witness_generation
            .checked_add(1)
            .is_some_and(|next| expected_generation > next),
        _ => false,
    };
    if !relation_valid {
        return Err("delivery_witness_outcome_invalid".to_string());
    }

    let signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        requested_generation,
        &requested_anchor_digest,
        &witness,
        response_timestamp,
        witness_generation,
        &witness_anchor_digest,
        outcome,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_delivery_witness_signature".to_string())?;
    Ok(outcome)
}

/// Verifies one request-bound portable custody witness receipt.
///
/// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Negative receipts are valid
/// evidence, so this verifier authenticates their exact relation rather than
/// collapsing them into transport errors. The outer signature binds the
/// portable receipt to one request; the nested signature remains independently
/// verifiable by later auditors.
#[allow(clippy::too_many_arguments)]
fn verify_custody_audit_anchor_witness_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_witness: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
    anchor_frame_sha256: &[u8; 32],
    now: u64,
) -> Result<CustodyAuditWitnessReceiptV1, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_custody_witness_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_custody_witness_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_custody_witness_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_custody_witness_frame".to_string());
    }
    let MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id,
        requester,
        witness,
        response_timestamp,
        receipt,
        signature,
    } = response
    else {
        return Err("unexpected_custody_witness_message".to_string());
    };
    if request_id != *expected_request_id || requester != *expected_requester {
        return Err("custody_witness_request_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("custody_witness_identity_mismatch".to_string());
    }
    if response_timestamp != receipt.observed_at
        || now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
    {
        return Err("custody_witness_timestamp_invalid".to_string());
    }

    let receipt_sha256 = custody_audit_witness_receipt_frame_sha256(&receipt)
        .map_err(|_| "invalid_custody_witness_receipt".to_string())?;
    let signing_bytes = custody_audit_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        &witness,
        response_timestamp,
        &receipt_sha256,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_custody_witness_response_signature".to_string())?;
    receipt
        .verify_for_anchor(
            anchor,
            anchor_frame_sha256,
            expected_requester,
            expected_witness,
            1,
        )
        .map_err(|_| "invalid_custody_witness_receipt".to_string())?;
    Ok(receipt)
}

fn verify_checkpoint_certificate_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    now: u64,
) -> Result<VerifiedCheckpointCertificate, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_certificate_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_certificate_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_certificate_frame")?;
    if canonical != body {
        return Err("noncanonical_certificate_frame".to_string());
    }
    let MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        certificate_digest,
        required_signers,
        members,
        signature,
    } = response
    else {
        return Err("unexpected_certificate_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("certificate_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("certificate_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("certificate_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_certificate_response".to_string());
    }
    if checkpoint_height == 0 || checkpoint_height != local_tip.0 || checkpoint_hash != local_tip.1
    {
        return Err("certificate_local_tip_mismatch".to_string());
    }
    let signer_count = members.iter().flatten().count();
    let required_signers = usize::from(required_signers);
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&required_signers)
        || signer_count < required_signers
        || signer_count > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return Err("certificate_threshold_invalid".to_string());
    }
    if required_signers < minimum_required_signers {
        return Err("certificate_threshold_below_policy".to_string());
    }
    let signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        &certificate_digest,
        required_signers as u8,
        signer_count as u8,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_certificate_response_signature".to_string())?;

    let mut saw_empty_slot = false;
    let mut previous_responder = None;
    let mut digest_members = Vec::with_capacity(signer_count);
    let mut verified_members = Vec::with_capacity(signer_count);
    for slot in members {
        let Some(member) = slot else {
            saw_empty_slot = true;
            continue;
        };
        if saw_empty_slot {
            return Err("certificate_members_not_packed".to_string());
        }
        if previous_responder.is_some_and(|previous| previous >= member.responder) {
            return Err("certificate_members_not_distinct_sorted".to_string());
        }
        previous_responder = Some(member.responder);
        if !allowed_witnesses.contains(&member.responder) {
            return Err("certificate_member_not_pinned".to_string());
        }
        if member.response_timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS) {
            return Err("certificate_member_timestamp_invalid".to_string());
        }
        if member.checkpoint_height != checkpoint_height
            || member.checkpoint_hash != checkpoint_hash
            || member.tip_height < checkpoint_height
            || (member.tip_height == checkpoint_height && member.tip_hash != checkpoint_hash)
        {
            return Err("certificate_member_claim_invalid".to_string());
        }
        let member_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &member.request_id,
            &member.responder,
            member.response_timestamp,
            member.checkpoint_height,
            &member.checkpoint_hash,
            member.tip_height,
            &member.tip_hash,
        );
        IdentityPublicKey::from_bytes(&member.responder)
            .and_then(|key| key.verify(&member_signing_bytes, &member.signature))
            .map_err(|_| "invalid_certificate_member_signature".to_string())?;
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
        .map_err(|_| "certificate_member_encode_failed".to_string())?;
        let evidence_digest: [u8; 32] = Sha256::digest(&frame).into();
        digest_members.push((member.responder, evidence_digest));
        verified_members.push(VerifiedCertificateMember {
            observed_at: member.response_timestamp,
            remote_tip_height: member.tip_height,
            evidence_digest,
            frame,
        });
    }
    let computed_digest = record_checkpoint_certificate_digest_v1(
        &chain_id,
        checkpoint_height,
        &checkpoint_hash,
        required_signers,
        &digest_members,
    );
    if computed_digest != certificate_digest {
        return Err("certificate_digest_mismatch".to_string());
    }
    Ok(VerifiedCheckpointCertificate {
        checkpoint_height,
        required_signers,
        members: verified_members,
    })
}

async fn verify_record_commitment_checkpoint(
    storage: &MemoryStorage,
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<CommitmentCheckpointOutcome, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_checkpoint_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_checkpoint_frame")?;
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_checkpoint_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("checkpoint_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("checkpoint_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("checkpoint_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_checkpoint_response".to_string());
    }
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_checkpoint_signature".to_string())?;

    if tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
        return Err("invalid_checkpoint_genesis".to_string());
    }
    let expected_checkpoint_height = local_tip.0.min(tip_height);
    if checkpoint_height != expected_checkpoint_height {
        return Err("checkpoint_height_mismatch".to_string());
    }
    if checkpoint_height == tip_height && checkpoint_hash != tip_hash {
        return Err("checkpoint_tip_inconsistent".to_string());
    }
    let (resolved_height, local_checkpoint_hash, _, _) = storage
        .record_commitment_chain_checkpoint(checkpoint_height)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if resolved_height != checkpoint_height {
        return Err("local_checkpoint_height_mismatch".to_string());
    }

    let relation = if local_checkpoint_hash != checkpoint_hash {
        CommitmentCheckpointRelation::Diverged
    } else if local_tip.0 == tip_height {
        CommitmentCheckpointRelation::Converged
    } else if local_tip.0 < tip_height {
        CommitmentCheckpointRelation::RemoteAhead
    } else {
        CommitmentCheckpointRelation::RemoteBehind
    };
    let evidence_digest: [u8; 32] = Sha256::digest(body).into();
    Ok(CommitmentCheckpointOutcome {
        relation,
        local_tip_height: local_tip.0,
        remote_tip_height: tip_height,
        checkpoint_height,
        evidence_digest,
    })
}

async fn read_bounded_response(response: reqwest::Response) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BODY_BYTES as u64)
    {
        return Err("response_too_large".to_string());
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| classify_http_error("response_body", &error))?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BODY_BYTES {
            return Err("response_too_large".to_string());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn verify_record_commitment_page(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    expected_proposer: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<VerifiedCommitmentPage, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_response_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_response_frame")?;
    let MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_response_message".to_string());
    };

    if request_id != *expected_request_id {
        return Err("response_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("response_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_response".to_string());
    }
    if blocks.len() > MAX_BLOCKS_PER_RESPONSE {
        return Err("response_page_too_large".to_string());
    }
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_response_signature".to_string())?;

    let (local_height, local_hash) = local_tip;
    if local_height == 0 && local_hash != GENESIS_PREV_HASH {
        return Err("invalid_local_genesis".to_string());
    }
    if tip_height < local_height {
        return Err(if expected_responder == expected_proposer {
            "coordinator_rollback_detected"
        } else {
            "carrier_tip_behind"
        }
        .to_string());
    }
    if blocks.is_empty() {
        if has_more || tip_height != local_height || tip_hash != local_hash {
            return Err("empty_page_tip_mismatch".to_string());
        }
        return Ok(VerifiedCommitmentPage {
            blocks,
            has_more,
            tip_height,
        });
    }
    if tip_height <= local_height {
        return Err("unexpected_blocks_at_current_tip".to_string());
    }

    let mut expected_height = local_height.saturating_add(1);
    let mut expected_prev_hash = local_hash;
    for block in &blocks {
        if block.header.proposer != *expected_proposer {
            return Err("unexpected_block_proposer".to_string());
        }
        block
            .verify(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                expected_height,
                &expected_prev_hash,
            )
            .map_err(|_| "commitment_chain_verification_failed".to_string())?;
        expected_height = expected_height.saturating_add(1);
        expected_prev_hash = block.hash();
    }

    let page_tip_height = expected_height.saturating_sub(1);
    let expected_has_more = page_tip_height < tip_height;
    if has_more != expected_has_more {
        return Err("pagination_state_mismatch".to_string());
    }
    if !has_more && (tip_height != page_tip_height || tip_hash != expected_prev_hash) {
        return Err("terminal_tip_mismatch".to_string());
    }

    Ok(VerifiedCommitmentPage {
        blocks,
        has_more,
        tip_height,
    })
}

fn classify_http_error(phase: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_body() {
        "body"
    } else if error.is_decode() {
        "decode"
    } else if error.is_request() {
        "request"
    } else {
        "unknown"
    };
    format!("{phase}_{kind}")
}

fn checkpoint_certificate_member_from_frame(
    frame: &[u8],
    expected_chain_id: &[u8; 32],
) -> Result<RecordCheckpointCertificateMemberV1, String> {
    if frame.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("certificate_member_frame_invalid".to_string());
    }
    let message =
        decode_memchain(&frame[1..]).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    let canonical =
        encode_memchain(&message).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    if canonical != frame {
        return Err("certificate_member_frame_noncanonical".to_string());
    }
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("certificate_member_frame_unexpected".to_string());
    };
    if chain_id != *expected_chain_id {
        return Err("certificate_member_chain_mismatch".to_string());
    }
    Ok(RecordCheckpointCertificateMemberV1 {
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    })
}

async fn checkpoint_certificate_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || known_tip_height == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_certificate_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let bundle = match state
        .storage
        .record_commitment_checkpoint_certificate_bundle(known_tip_height, &known_tip_hash)
        .await
    {
        Ok(Some(bundle)) => bundle,
        Ok(None) => return protocol_error(StatusCode::NOT_FOUND, "certificate_unavailable"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited certificate export");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
        }
    };
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&bundle.required_signers)
        || bundle.member_frames.len() < bundle.required_signers
        || bundle.member_frames.len() > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
    }
    let mut members = [None; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1];
    for (slot, frame) in members.iter_mut().zip(bundle.member_frames.iter()) {
        *slot = match checkpoint_certificate_member_from_frame(frame, &chain_id) {
            Ok(member) => Some(member),
            Err(error) => {
                warn!(error = %error, "[MEMCHAIN_BLOCK] Refused invalid certificate member");
                return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
            }
        };
    }
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        bundle.checkpoint_height,
        &bundle.checkpoint_hash,
        &bundle.certificate_digest,
        bundle.required_signers as u8,
        bundle.member_frames.len() as u8,
    );
    let response = MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height: bundle.checkpoint_height,
        checkpoint_hash: bundle.checkpoint_hash,
        certificate_digest: bundle.certificate_digest,
        required_signers: bundle.required_signers as u8,
        members,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode certificate response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        checkpoint_height = bundle.checkpoint_height,
        signer_count = bundle.member_frames.len(),
        "[MEMCHAIN_BLOCK] Served authenticated checkpoint certificate"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_handover_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id,
        after_authority_epoch,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || after_authority_epoch == u64::MAX {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_handover_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_handover_request_signing_bytes(
        &chain_id,
        after_authority_epoch,
        &request_id,
        &requester,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [AUTHORITY-HANDOVER-ADMISSION 2026-08-14 by Codex] Authenticate before
    // consulting PeerStore. Otherwise a forged request can distinguish a
    // known public key from an unknown one by comparing HTTP error classes.
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let page = match state
        .storage
        .next_record_coordinator_handover_page(after_authority_epoch)
        .await
    {
        Ok(page) => page,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused authority handover snapshot");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "handover_history_unavailable",
            );
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let has_handover = page.handover.is_some();
    let response_signing_bytes = record_coordinator_handover_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        page.handover.as_ref(),
        page.latest_authority_epoch,
    );
    let response = MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        handover: page.handover,
        latest_authority_epoch: page.latest_authority_epoch,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        has_handover,
        latest_authority_epoch = page.latest_authority_epoch,
        "[MEMCHAIN_BLOCK] Served authenticated authority handover snapshot"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || instance_id.iter().all(|byte| *byte == 0)
        || !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
            .contains(&requested_ttl_secs)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Only an
    // authenticated coordinator may trigger the storage-backed authority
    // lookup or learn its peer-admission result.
    let Some(next_height) = known_tip_height.checked_add(1) else {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_request");
    };
    let authorized_coordinator = match runtime_authorized_coordinator_for_height(
        &state.storage,
        state.lease_authorized_coordinator,
        next_height,
    )
    .await
    {
        Ok(Some(authorized)) => authorized,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited lease authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if authorized_coordinator != coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if !coordinator_control_requester_is_admitted(&state, &coordinator, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let witness_tip = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => {
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "witness_tip_unavailable");
        }
    };
    if witness_tip != (known_tip_height, known_tip_hash) {
        return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
    }
    let grant = match state
        .storage
        .grant_record_commitment_coordinator_lease(
            &chain_id,
            &coordinator,
            &instance_id,
            known_tip_height,
            &known_tip_hash,
            now,
            requested_ttl_secs,
        )
        .await
    {
        Ok(RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch,
            lease_expires_at,
        }) => (lease_epoch, lease_expires_at),
        Ok(RecordCoordinatorLeaseGrantOutcome::TipMismatch) => {
            return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
        }
        Ok(RecordCoordinatorLeaseGrantOutcome::Contended) => {
            return protocol_error(StatusCode::CONFLICT, "lease_contended");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease persistence failed");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "lease_persist_failed");
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        grant.0,
        grant.1,
        witness_tip.0,
        &witness_tip.1,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch: grant.0,
        lease_expires_at: grant.1,
        witness_tip_height: witness_tip.0,
        witness_tip_hash: witness_tip.1,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch = grant.0,
        lease_ttl_secs = requested_ttl_secs,
        "[MEMCHAIN_BLOCK] Granted authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_release_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || instance_id.iter().all(|byte| *byte == 0) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_release_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Release requests
    // use the same authenticate-before-authorize ordering as acquisition.
    let authorized_coordinator = match runtime_authorized_coordinator_for_next_height(
        &state.storage,
        state.lease_authorized_coordinator,
    )
    .await
    {
        Ok(Some(authorized)) => authorized,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited lease release authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if authorized_coordinator != coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if !coordinator_control_requester_is_admitted(&state, &coordinator, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (lease_epoch, released_at) = match state
        .storage
        .release_record_commitment_coordinator_lease(&chain_id, &coordinator, &instance_id, now)
        .await
    {
        Ok(RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch,
            released_at,
        }) => (lease_epoch, released_at),
        Ok(RecordCoordinatorLeaseReleaseOutcome::NotHolder) => {
            return protocol_error(StatusCode::CONFLICT, "lease_release_not_holder");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease release persistence failed");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "lease_release_persist_failed",
            );
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch,
        "[MEMCHAIN_BLOCK] Released authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn verified_delivery_anchor_witness_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let canonical = match encode_memchain(&message) {
        Ok(canonical) => canonical,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    if canonical.as_slice() != body.as_ref() {
        // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] Witness writes are
        // security evidence. Reject alternate encodings instead of allowing
        // one signed request to acquire multiple replay identities.
        return protocol_error(StatusCode::BAD_REQUEST, "noncanonical_frame");
    }
    let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
        requester,
        generation,
        anchor_digest,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if generation == 0 || generation > i64::MAX as u64 || anchor_digest == [0u8; 32] {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_delivery_witness_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
        &requester,
        generation,
        &anchor_digest,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] Authenticate before
    // consulting private operator pins, then collapse pin and discovery
    // failures into one response. An unauthenticated caller cannot use status
    // differences to enumerate a witness node's trust relationships.
    if !state
        .peer_store
        .verified_delivery_witness_requester_allowed(&requester)
        || state.peer_store.get_valid(&requester, now).is_none()
    {
        return protocol_error(StatusCode::FORBIDDEN, "witness_requester_not_authorized");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (outcome, witness_generation, witness_anchor_digest) = match state
        .storage
        .witness_verified_delivery_anchor(&requester, generation, &anchor_digest, now)
        .await
    {
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_STALE_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_CONFLICT_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        }) => (VERIFIED_DELIVERY_WITNESS_GAP_V1, generation, anchor_digest),
        Err(error) => {
            warn!(error = %error, "[DISCOVERY] Delivery-anchor witness persistence failed");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "delivery_witness_persist_failed",
            );
        }
    };

    let witness = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        generation,
        &anchor_digest,
        &witness,
        response_timestamp,
        witness_generation,
        &witness_anchor_digest,
        outcome,
    );
    let response = MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester,
        requested_generation: generation,
        requested_anchor_digest: anchor_digest,
        witness,
        response_timestamp,
        witness_generation,
        witness_anchor_digest,
        outcome,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        generation,
        outcome, "[DISCOVERY] Served authenticated delivery-anchor witness decision"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn custody_audit_anchor_witness_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let canonical = match encode_memchain(&message) {
        Ok(canonical) => canonical,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    if canonical.as_slice() != body.as_ref() {
        return protocol_error(StatusCode::BAD_REQUEST, "noncanonical_frame");
    }
    let MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id,
        requester,
        request_timestamp,
        anchor,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if anchor.checkpoint_generation == 0
        || anchor.checkpoint_generation > i64::MAX as u64
        || requester != anchor.producer_node_id
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_custody_witness_request");
    }
    // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Keep freshness failures
    // distinct from structural failures so operators can identify replay or
    // clock-skew incidents without exposing request identities in logs.
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let witness = state.identity.public_key_bytes();
    if anchor
        .verify_expected(&requester, anchor.checkpoint_generation)
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_anchor_signature");
    }
    let anchor_sha256 = match custody_audit_anchor_frame_sha256(&anchor) {
        Ok(digest) if digest != [0u8; 32] => digest,
        Ok(_) | Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_custody_anchor"),
    };
    let signing_bytes = custody_audit_anchor_witness_request_signing_bytes(
        &request_id,
        &requester,
        request_timestamp,
        &anchor_sha256,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if requester == witness {
        // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Reject only after
        // authentication and before monotonic state changes. Self-witness
        // evidence remains invalid without becoming a signature oracle.
        return protocol_error(StatusCode::FORBIDDEN, "independent_witness_required");
    }
    if !state
        .peer_store
        .custody_audit_witness_requester_allowed(&requester)
        || state.peer_store.get_valid(&requester, now).is_none()
    {
        return protocol_error(StatusCode::FORBIDDEN, "custody_witness_not_authorized");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (outcome, witness_generation, witness_anchor_sha256) = match state
        .storage
        .witness_custody_audit_anchor(
            &requester,
            anchor.checkpoint_generation,
            &anchor_sha256,
            now,
        )
        .await
    {
        Ok(CustodyAuditAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_ADVANCED_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        }) => (
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        Ok(CustodyAuditAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_STALE_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_CONFLICT_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_GAP_V1, generation, anchor_digest),
        Err(_) => {
            warn!(
                generation = anchor.checkpoint_generation,
                "[MEMCHAIN] Custody-anchor witness persistence failed"
            );
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "custody_witness_persist_failed",
            );
        }
    };

    let receipt = match CustodyAuditWitnessReceiptV1::signed(
        requester,
        anchor.checkpoint_generation,
        anchor_sha256,
        now,
        witness_generation,
        witness_anchor_sha256,
        outcome,
        &state.identity,
    ) {
        Ok(receipt) => receipt,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "receipt_sign_failed"),
    };
    let receipt_sha256 = match custody_audit_witness_receipt_frame_sha256(&receipt) {
        Ok(digest) => digest,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    let response_signing_bytes = custody_audit_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        &witness,
        receipt.observed_at,
        &receipt_sha256,
    );
    let response = MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id,
        requester,
        witness,
        response_timestamp: receipt.observed_at,
        receipt,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    if verify_custody_audit_anchor_witness_response(
        &encoded,
        &request_id,
        &requester,
        &witness,
        &anchor,
        &anchor_sha256,
        now,
    )
    .is_err()
    {
        // A locally generated frame that fails the public verification
        // contract must never leave the process or become apparent evidence.
        return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "receipt_verify_failed");
    }
    debug!(
        generation = anchor.checkpoint_generation,
        outcome, "[MEMCHAIN] Served authenticated custody-anchor witness decision"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn checkpoint_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || (known_tip_height == 0 && known_tip_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_checkpoint_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if !coordinator_control_requester_is_admitted(&state, &requester, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (checkpoint_height, checkpoint_hash, tip_height, tip_hash) = match state
        .storage
        .record_commitment_chain_checkpoint(known_tip_height)
        .await
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited checkpoint proof");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let relation = if known_tip_height > tip_height {
        "served"
    } else if known_tip_hash != checkpoint_hash {
        "diverged"
    } else if known_tip_height == tip_height {
        "converged"
    } else {
        "remote_behind"
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode checkpoint response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    // Count only a response that was successfully constructed. The inbound
    // request's relation/heights remain requester-controlled debug context and
    // cannot overwrite this node's outbound checkpoint evidence.
    state.storage.record_commitment_checkpoint_served(now);
    debug!(
        relation,
        checkpoint_height, tip_height, "[MEMCHAIN_BLOCK] Served authenticated chain checkpoint"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn block_range_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || from_height == 0 || limit == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let page_limit = usize::from(limit).min(MAX_BLOCKS_PER_RESPONSE);
    let page = match state
        .storage
        .get_verified_record_commitment_block_page(from_height, page_limit)
        .await
    {
        Ok(page) => page,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unverified block range");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let blocks = page.blocks;
    let tip_height = page.tip_height;
    let tip_hash = page.tip_hash;
    let page_tip = blocks.last().map_or_else(
        || from_height.saturating_sub(1),
        |block| block.header.height,
    );
    let has_more = page_tip < tip_height;
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode block range response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        blocks = match &response {
            MemChainMessage::RecordBlockRangeResponseV1 { blocks, .. } => blocks.len(),
            _ => 0,
        },
        has_more, tip_height, "[MEMCHAIN_BLOCK] Served authenticated commitment range"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, axum::Json(serde_json::json!({ "error": code }))).into_response()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn handover_response_rejects_omission_and_wrong_predecessor() {
        // [AUTHORITY-HANDOVER-ADVERSARIAL 2026-08-14 by Codex] A responder
        // cannot advertise a newer history head while withholding the exact
        // next proof, nor wrap somebody else's valid transition in its own
        // authenticated transport response.
        let now = 50_000;
        let request_id = [0x41; 16];
        let active = IdentityKeyPair::generate();
        let next = IdentityKeyPair::generate();
        let carrier = IdentityKeyPair::generate();
        let omitted = signed_handover_response_frame(&active, request_id, now, None, 1);
        assert_eq!(
            verify_record_coordinator_handover_response(
                &omitted,
                &request_id,
                &active.public_key_bytes(),
                &active.public_key_bytes(),
                0,
                1,
                now,
            )
            .unwrap_err(),
            "handover_proof_omitted"
        );

        let unrelated_previous = IdentityKeyPair::generate();
        let wrong_predecessor = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            [0x42; 32],
            [0x43; 16],
            now.saturating_sub(1),
            &unrelated_previous,
            &next,
        );
        let wrapped =
            signed_handover_response_frame(&carrier, request_id, now, Some(wrong_predecessor), 1);
        assert_eq!(
            verify_record_coordinator_handover_response(
                &wrapped,
                &request_id,
                &carrier.public_key_bytes(),
                &active.public_key_bytes(),
                0,
                1,
                now,
            )
            .unwrap_err(),
            "handover_previous_coordinator_mismatch"
        );

        let valid_transition = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            [0x44; 32],
            [0x45; 16],
            now.saturating_sub(1),
            &active,
            &next,
        );
        let carried = signed_handover_response_frame(
            &carrier,
            request_id,
            now,
            Some(valid_transition),
            1,
        );
        assert!(verify_record_coordinator_handover_response(
            &carried,
            &request_id,
            &carrier.public_key_bytes(),
            &active.public_key_bytes(),
            0,
            1,
            now,
        )
        .is_ok());
    }

    #[test]
    fn handover_carrier_retries_only_explicit_availability_failures() {
        // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Alternate pins may
        // improve availability, never mask authenticated protocol failures.
        for error in [
            "active_coordinator_unavailable",
            "active_coordinator_missing_endpoint",
            "handover_carrier_unavailable",
            "handover_carrier_missing_endpoint",
            "handover_carrier_behind",
            "handover_request_timeout",
            "handover_http_status_503",
        ] {
            assert_eq!(
                coordinator_handover_source_failure_class(error),
                CommitmentAuthoritySourceFailureClass::Availability,
                "{error}"
            );
        }
        for error in [
            "active_coordinator_unsafe_endpoint",
            "handover_carrier_unsafe_endpoint",
            "handover_http_status_401",
            "invalid_handover_response_signature",
            "handover_previous_coordinator_mismatch",
            "handover_local_authority_changed",
            "storage_append_rejected",
        ] {
            assert_eq!(
                coordinator_handover_source_failure_class(error),
                CommitmentAuthoritySourceFailureClass::Security,
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn handover_endpoint_authenticates_before_peer_membership_admission() {
        // [AUTHORITY-HANDOVER-ADMISSION 2026-08-14 by Codex] A forged request
        // must not reveal whether its claimed requester is in PeerStore. A
        // genuinely signed but unadmitted requester remains forbidden.
        let now = now_secs();
        let responder = Arc::new(IdentityKeyPair::generate());
        let known = IdentityKeyPair::generate();
        let unknown = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &known, None, now);
        let router = build_memchain_peer_router(storage, peer_store, responder);

        for (requester, request_id) in [
            (known.public_key_bytes(), [0x44; 16]),
            (unknown.public_key_bytes(), [0x45; 16]),
        ] {
            let forged = encode_memchain(
                &MemChainMessage::RecordCoordinatorHandoverRequestV1 {
                    chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                    after_authority_epoch: 0,
                    request_id,
                    requester,
                    request_timestamp: now,
                    signature: [0u8; 64],
                },
            )
            .unwrap();
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/memchain/peer/coordinator-handover")
                        .body(Body::from(forged))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }

        let request_id = [0x46; 16];
        let requester = unknown.public_key_bytes();
        let signing_bytes = record_coordinator_handover_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            0,
            &request_id,
            &requester,
            now,
        );
        let signed_unknown = encode_memchain(
            &MemChainMessage::RecordCoordinatorHandoverRequestV1 {
                chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                after_authority_epoch: 0,
                request_id,
                requester,
                request_timestamp: now,
                signature: unknown.sign(&signing_bytes),
            },
        )
        .unwrap();
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-handover")
                    .body(Body::from(signed_unknown))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn cold_follower_interleaves_prefix_and_exact_handover() {
        // [AUTHORITY-HANDOVER-FOLLOWER 2026-08-14 by Codex] The follower first
        // learns the transition as a future boundary, pulls only block one,
        // then accepts the same dual-signed proof and switches authority for
        // height two. No responder assertion alone can rotate authority.
        let now = now_secs();
        let coordinator = Arc::new(IdentityKeyPair::generate());
        let next = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        source
            .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();
        let first_block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x51; 32]],
            coordinator.as_ref(),
        );
        source
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        let proof = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            first_block.hash(),
            [0x52; 16],
            now.saturating_sub(1),
            coordinator.as_ref(),
            &next,
        );
        source
            .persist_configured_record_coordinator_handover(&proof, now)
            .await
            .unwrap();
        let second_block = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first_block.hash(),
            vec![[0x53; 32]],
            &next,
        );
        source
            .append_record_commitment_block(&second_block, None)
            .await
            .unwrap();

        let source_peers = Arc::new(PeerStore::new());
        admit_peer(&source_peers, &follower, None, now);
        let router =
            build_memchain_peer_router(Arc::clone(&source), source_peers, Arc::clone(&coordinator));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let destination = MemoryStorage::open(":memory:", None).unwrap();
        destination
            .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_sync(false, true);
        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            coordinator.as_ref(),
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();

        let pending = sync_next_record_coordinator_handover_with_endpoint_policy(
            &destination,
            &destination_peers,
            &follower,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(pending.authority_epoch, 0);
        assert_eq!(pending.active_coordinator, coordinator.public_key_bytes());
        assert_eq!(pending.next_block_height, 1);
        assert_eq!(pending.pending_activation_height, Some(2));
        assert!(!pending.handover_inserted);
        assert_eq!(pending.source, CommitmentAuthoritySyncSource::Coordinator);
        assert_eq!(pending.carrier_attempts, 0);

        let page = pull_record_commitment_page_from_source_with_endpoint_policy(
            &destination,
            &destination_peers,
            &follower,
            &coordinator.public_key_bytes(),
            &coordinator.public_key_bytes(),
            &client,
            &allow_test_endpoint,
            1,
        )
        .await
        .unwrap();
        assert_eq!(page.inserted, 1);
        assert!(page.has_more);

        let activated = sync_next_record_coordinator_handover_with_endpoint_policy(
            &destination,
            &destination_peers,
            &follower,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(activated.authority_epoch, 1);
        assert_eq!(activated.active_coordinator, next.public_key_bytes());
        assert_eq!(activated.next_block_height, 2);
        assert_eq!(activated.pending_activation_height, None);
        assert!(activated.handover_inserted);
        assert_eq!(
            activated.source,
            CommitmentAuthoritySyncSource::Coordinator
        );
        assert_eq!(activated.carrier_attempts, 0);

        server.abort();
    }

    #[tokio::test]
    async fn pinned_carrier_recovers_exact_handover_when_coordinator_is_unavailable() {
        // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] The carrier signs
        // only its response envelope. The accepted authority transition must
        // still be the root coordinator's exact-next dual-signed proof bound
        // to the follower's already-audited block-one prefix.
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let next = IdentityKeyPair::generate();
        let stale_carrier = Arc::new(IdentityKeyPair::generate());
        let carrier = Arc::new(IdentityKeyPair::generate());
        let follower = IdentityKeyPair::generate();
        let first_block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x54; 32]],
            &coordinator,
        );
        let proof = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            first_block.hash(),
            [0x55; 16],
            now.saturating_sub(1),
            &coordinator,
            &next,
        );

        let stale_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        stale_storage
            .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
            .unwrap();
        stale_storage.audit_record_commitment_chain().await.unwrap();
        stale_storage
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        let stale_peers = Arc::new(PeerStore::new());
        admit_peer(&stale_peers, &follower, None, now);
        let stale_router = build_memchain_peer_router(
            Arc::clone(&stale_storage),
            stale_peers,
            Arc::clone(&stale_carrier),
        );
        let stale_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let stale_address = stale_listener.local_addr().unwrap();
        let stale_server = tokio::spawn(async move {
            axum::serve(stale_listener, stale_router).await.unwrap();
        });

        let carrier_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        carrier_storage
            .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
            .unwrap();
        carrier_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        carrier_storage
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        carrier_storage
            .persist_configured_record_coordinator_handover(&proof, now)
            .await
            .unwrap();
        let carrier_peers = Arc::new(PeerStore::new());
        admit_peer(&carrier_peers, &follower, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&carrier_storage),
            carrier_peers,
            Arc::clone(&carrier),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let destination = MemoryStorage::open(":memory:", None).unwrap();
        destination
            .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        destination.configure_record_commitment_sync(false, true);
        let destination_peers = PeerStore::new();
        // The active coordinator is authenticated but has no reachable
        // endpoint, forcing only the narrow availability fallback.
        admit_peer(&destination_peers, &coordinator, None, now);
        admit_peer(
            &destination_peers,
            stale_carrier.as_ref(),
            Some(format!("http://{stale_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            carrier.as_ref(),
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let mut cursor = CommitmentAuthorityCarrierCursor::default();
        let mut circuit_breaker = CommitmentAuthorityCarrierCircuitBreaker::default();
        let recovered =
            sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
                &destination,
                &destination_peers,
                &follower,
                &[stale_carrier.public_key_bytes(), carrier.public_key_bytes()],
                &client,
                &allow_test_endpoint,
                &mut cursor,
                &mut circuit_breaker,
            )
            .await
            .unwrap();

        assert_eq!(
            recovered.source,
            CommitmentAuthoritySyncSource::PinnedCarrier
        );
        assert_eq!(recovered.carrier_attempts, 2);
        assert!(recovered.handover_inserted);
        assert_eq!(recovered.authority_epoch, 1);
        assert_eq!(recovered.active_coordinator, next.public_key_bytes());
        assert_eq!(recovered.next_block_height, 2);
        assert_eq!(recovered.pending_activation_height, None);
        let status = destination.record_commitment_sync_status();
        assert_eq!(status.authority_sync_rounds_total, 1);
        assert_eq!(status.authority_coordinator_success_total, 0);
        assert_eq!(status.authority_carrier_attempts_total, 2);
        assert_eq!(status.authority_carrier_recoveries_total, 1);
        assert_eq!(status.authority_availability_exhausted_total, 0);
        assert_eq!(status.authority_security_stops_total, 0);
        assert_eq!(
            status.last_authority_sync_result.as_deref(),
            Some("carrier_recovered")
        );
        assert!(status.last_authority_carrier_recovered_at.is_some());

        stale_server.abort();
        server.abort();
    }

    #[tokio::test]
    async fn audited_authority_schedule_supersedes_legacy_runtime_pin() {
        // [AUTHORITY-SCHEDULE-RUNTIME 2026-08-14 by Codex] Announcement and
        // lease handlers share this resolver. Once a handover is durable, the
        // legacy bootstrap pin cannot continue authorising later heights.
        let now = now_secs();
        let root = IdentityKeyPair::generate();
        let next = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage
            .configure_record_commitment_authority_root(Some(root.public_key_bytes()))
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        let first_block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x61; 32]],
            &root,
        );
        storage
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        let handover = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            first_block.hash(),
            [0x62; 16],
            now.saturating_sub(1),
            &root,
            &next,
        );
        storage
            .persist_configured_record_coordinator_handover(&handover, now)
            .await
            .unwrap();

        assert_eq!(
            runtime_authorized_coordinator_for_height(&storage, Some(root.public_key_bytes()), 1)
                .await
                .unwrap(),
            Some(root.public_key_bytes())
        );
        assert_eq!(
            runtime_authorized_coordinator_for_height(&storage, Some(root.public_key_bytes()), 2)
                .await
                .unwrap(),
            Some(next.public_key_bytes())
        );
        assert_eq!(
            runtime_authorized_coordinator_for_next_height(
                &storage,
                Some(root.public_key_bytes()),
            )
            .await
            .unwrap(),
            Some(next.public_key_bytes())
        );
    }

    #[test]
    fn follower_certificate_telemetry_requires_durable_persistence_for_success() {
        // [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] Both transport
        // paths must report an authenticated-but-deferred import identically;
        // only durable persistence may count as direct or carrier recovery.
        assert_eq!(
            follower_certificate_sync_disposition(
                CommitmentFollowerCertificateSource::Coordinator,
                true,
            ),
            RecordCommitmentCertificateSyncDisposition::Coordinator
        );
        assert_eq!(
            follower_certificate_sync_disposition(
                CommitmentFollowerCertificateSource::PinnedCarrier,
                true,
            ),
            RecordCommitmentCertificateSyncDisposition::CarrierRecovered
        );
        for source in [
            CommitmentFollowerCertificateSource::Coordinator,
            CommitmentFollowerCertificateSource::PinnedCarrier,
        ] {
            assert_eq!(
                follower_certificate_sync_disposition(source, false),
                RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted
            );
        }
    }

    #[test]
    fn certificate_carrier_fallback_classifies_only_availability_failures() {
        // [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex] Admission and
        // transport outages may advance to the next exact operator pin.
        // Authentication and evidence-integrity failures must remain terminal.
        for error in [
            "certificate_source_unavailable",
            "certificate_source_missing_endpoint",
            "certificate_request_timeout",
            "certificate_request_connect",
            "response_body_body",
            "certificate_http_status_403",
            "certificate_http_status_404",
            "certificate_http_status_429",
            "certificate_http_status_503",
        ] {
            assert_eq!(
                commitment_certificate_source_failure_class(error),
                CommitmentCertificateSourceFailureClass::Availability,
                "{error} must permit only bounded pinned-carrier recovery"
            );
        }
        for error in [
            "certificate_source_unsafe_endpoint",
            "certificate_http_status_400",
            "certificate_http_status_401",
            "invalid_certificate_frame",
            "invalid_certificate_response_signature",
            "certificate_local_tip_mismatch",
            "certificate_member_not_pinned",
            "certificate_digest_mismatch",
            "certificate_persist_failed",
        ] {
            assert_eq!(
                commitment_certificate_source_failure_class(error),
                CommitmentCertificateSourceFailureClass::Security,
                "{error} must stop before any fallback"
            );
        }
    }

    #[test]
    fn block_carrier_fallback_classifies_only_availability_failures() {
        // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Only source absence,
        // transient transport/status failure, or a safely ignored stale
        // carrier may advance to another exact pin. Evidence failures stop.
        for error in [
            "pinned_coordinator_unavailable",
            "pinned_coordinator_missing_endpoint",
            "request_timeout",
            "request_connect",
            "response_body_body",
            "http_status_403",
            "http_status_404",
            "http_status_429",
            "http_status_503",
            "carrier_tip_behind",
        ] {
            assert_eq!(
                commitment_block_source_failure_class(error),
                CommitmentBlockSourceFailureClass::Availability,
                "{error} must permit only bounded pinned-carrier recovery"
            );
        }
        for error in [
            "pinned_coordinator_unsafe_endpoint",
            "http_status_400",
            "http_status_401",
            "invalid_response_frame",
            "invalid_response_signature",
            "response_responder_mismatch",
            "unexpected_block_proposer",
            "commitment_chain_verification_failed",
            "storage_append_rejected",
        ] {
            assert_eq!(
                commitment_block_source_failure_class(error),
                CommitmentBlockSourceFailureClass::Security,
                "{error} must stop before another carrier"
            );
        }
    }

    #[tokio::test]
    async fn block_carrier_policy_requires_distinct_external_pins() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage.configure_record_commitment_sync(false, true);
        let peers = PeerStore::new();
        let follower = IdentityKeyPair::generate();
        let coordinator = IdentityKeyPair::generate().public_key_bytes();
        let witness = IdentityKeyPair::generate().public_key_bytes();
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();

        // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Repeating one pin or
        // including self cannot satisfy a two-witness recovery policy.
        let error = pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
            &storage,
            &peers,
            &follower,
            &coordinator,
            &[witness, witness, follower.public_key_bytes()],
            2,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "block_carrier_policy_invalid");

        // Backward-compatible policy keeps the historical direct-only failure
        // instead of silently enabling carrier transport.
        let error = pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
            &storage,
            &peers,
            &follower,
            &coordinator,
            &[witness],
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "pinned_coordinator_unavailable");
        let status = storage.record_commitment_sync_status();
        assert_eq!(status.block_page_pulls_total, 2);
        assert_eq!(status.block_page_coordinator_success_total, 0);
        assert_eq!(status.block_carrier_attempts_total, 0);
        assert_eq!(status.block_carrier_recoveries_total, 0);
        assert_eq!(status.block_page_availability_exhausted_total, 1);
        assert_eq!(status.block_page_security_stops_total, 1);
        assert_eq!(
            status.last_block_page_pull_result.as_deref(),
            Some("availability_exhausted")
        );
    }

    #[tokio::test]
    async fn multi_page_carrier_cursor_prefers_verified_source_and_hands_off() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let unavailable_carrier = IdentityKeyPair::generate();
        let first_live_carrier = Arc::new(IdentityKeyPair::generate());
        let second_live_carrier = Arc::new(IdentityKeyPair::generate());
        let follower = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

        let mut previous_hash = GENESIS_PREV_HASH;
        for height in 1..=u64::try_from(MAX_BLOCKS_PER_RESPONSE + 1).unwrap() {
            let marker = u8::try_from(height).unwrap();
            let block = RecordCommitmentBlockV1::new_signed(
                height,
                now.saturating_sub(32).saturating_add(height),
                previous_hash,
                vec![[marker; 32]],
                &coordinator,
            );
            previous_hash = block.hash();
            source
                .append_record_commitment_block(&block, None)
                .await
                .unwrap();
        }
        source.audit_record_commitment_chain().await.unwrap();

        let first_peers = Arc::new(PeerStore::new());
        admit_peer(&first_peers, &follower, None, now);
        let first_router = build_memchain_peer_router(
            Arc::clone(&source),
            first_peers,
            Arc::clone(&first_live_carrier),
        );
        let first_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let first_address = first_listener.local_addr().unwrap();
        let first_server = tokio::spawn(async move {
            axum::serve(first_listener, first_router).await.unwrap();
        });

        let second_peers = Arc::new(PeerStore::new());
        admit_peer(&second_peers, &follower, None, now);
        let second_router = build_memchain_peer_router(
            Arc::clone(&source),
            second_peers,
            Arc::clone(&second_live_carrier),
        );
        let second_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let second_address = second_listener.local_addr().unwrap();
        let second_server = tokio::spawn(async move {
            axum::serve(second_listener, second_router).await.unwrap();
        });

        // Allocate dedicated closed ports after both live endpoints are fixed
        // so neither unavailable descriptor aliases a live test server.
        let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let coordinator_address = coordinator_listener.local_addr().unwrap();
        let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let unavailable_address = unavailable_listener.local_addr().unwrap();
        drop(coordinator_listener);
        drop(unavailable_listener);

        let destination = MemoryStorage::open(":memory:", None).unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_sync(false, true);
        destination.configure_record_commitment_certificate_policy(3, 2);
        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            &coordinator,
            Some(format!("http://{coordinator_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &unavailable_carrier,
            Some(format!("http://{unavailable_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &first_live_carrier,
            Some(format!("http://{first_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &second_live_carrier,
            Some(format!("http://{second_address}")),
            now,
        );
        let carrier_ids = [
            unavailable_carrier.public_key_bytes(),
            first_live_carrier.public_key_bytes(),
            second_live_carrier.public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let mut cursor = CommitmentBlockCarrierCursor::default();

        // [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex] Page one
        // bypasses an unavailable first pin and establishes the second pin as
        // the round-local preferred carrier.
        let first_page =
            pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
                &destination,
                &destination_peers,
                &follower,
                &coordinator.public_key_bytes(),
                &carrier_ids,
                2,
                &client,
                &allow_test_endpoint,
                &mut cursor,
            )
            .await
            .unwrap();
        assert_eq!(first_page.source, CommitmentSyncPageSource::PinnedCarrier);
        assert_eq!(first_page.carrier_attempts, 2);
        assert_eq!(first_page.page.inserted, MAX_BLOCKS_PER_RESPONSE);
        assert!(first_page.page.has_more);
        assert_eq!(cursor.next_index, 1);

        first_server.abort();
        let _ = first_server.await;
        // Axum may leave an accepted keep-alive connection alive after the
        // listener task is aborted. A fresh pool models a process/network
        // outage by requiring a new connection to the now-closed endpoint.
        let failover_client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();

        // The preferred carrier disappears between pages. The cursor starts
        // there, then hands off directly to the next exact pin without
        // retrying the earlier unavailable pin.
        let second_page =
            pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
                &destination,
                &destination_peers,
                &follower,
                &coordinator.public_key_bytes(),
                &carrier_ids,
                2,
                &failover_client,
                &allow_test_endpoint,
                &mut cursor,
            )
            .await
            .unwrap();
        assert_eq!(
            second_page.source,
            CommitmentSyncPageSource::PinnedCarrier
        );
        assert_eq!(second_page.carrier_attempts, 2);
        assert_eq!(second_page.page.inserted, 1);
        assert!(!second_page.page.has_more);
        assert_eq!(cursor.next_index, 2);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        destination.audit_record_commitment_chain().await.unwrap();

        let status = destination.record_commitment_sync_status();
        assert_eq!(status.block_page_pulls_total, 2);
        assert_eq!(status.block_carrier_attempts_total, 4);
        assert_eq!(status.block_carrier_recoveries_total, 2);
        assert_eq!(status.block_page_security_stops_total, 0);

        second_server.abort();
        let _ = second_server.await;
    }

    #[tokio::test]
    async fn carrier_cursor_never_masks_a_security_failure_with_the_next_pin() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let malformed_carrier = IdentityKeyPair::generate();
        let valid_carrier = Arc::new(IdentityKeyPair::generate());
        let follower = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x73; 32]],
            &coordinator,
        );
        source
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();

        let malformed_router = Router::new().route(
            "/api/memchain/peer/block-range",
            post(|| async { (StatusCode::OK, vec![0u8]) }),
        );
        let malformed_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let malformed_address = malformed_listener.local_addr().unwrap();
        let malformed_server = tokio::spawn(async move {
            axum::serve(malformed_listener, malformed_router)
                .await
                .unwrap();
        });

        let valid_peers = Arc::new(PeerStore::new());
        admit_peer(&valid_peers, &follower, None, now);
        let valid_router = build_memchain_peer_router(
            Arc::clone(&source),
            valid_peers,
            Arc::clone(&valid_carrier),
        );
        let valid_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let valid_address = valid_listener.local_addr().unwrap();
        let valid_server = tokio::spawn(async move {
            axum::serve(valid_listener, valid_router).await.unwrap();
        });

        let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let coordinator_address = coordinator_listener.local_addr().unwrap();
        drop(coordinator_listener);

        let destination = MemoryStorage::open(":memory:", None).unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_sync(false, true);
        destination.configure_record_commitment_certificate_policy(2, 2);
        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            &coordinator,
            Some(format!("http://{coordinator_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &malformed_carrier,
            Some(format!("http://{malformed_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &valid_carrier,
            Some(format!("http://{valid_address}")),
            now,
        );
        let carrier_ids = [
            malformed_carrier.public_key_bytes(),
            valid_carrier.public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let mut cursor = CommitmentBlockCarrierCursor::default();

        // [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex] Rotation is
        // availability-only. A malformed preferred carrier must stop before a
        // valid later pin can hide the security incident.
        let error = pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
            &destination,
            &destination_peers,
            &follower,
            &coordinator.public_key_bytes(),
            &carrier_ids,
            2,
            &client,
            &allow_test_endpoint,
            &mut cursor,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "invalid_response_frame");
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            (0, GENESIS_PREV_HASH)
        );
        let status = destination.record_commitment_sync_status();
        assert_eq!(status.block_page_pulls_total, 1);
        assert_eq!(status.block_carrier_attempts_total, 1);
        assert_eq!(status.block_carrier_recoveries_total, 0);
        assert_eq!(status.block_page_security_stops_total, 1);

        malformed_server.abort();
        let _ = malformed_server.await;
        valid_server.abort();
        let _ = valid_server.await;
    }

    #[test]
    fn carrier_circuit_breaker_uses_half_open_recovery_without_identity_state() {
        let started_at = Instant::now();
        let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();
        circuit_breaker.align_slots(2);

        // [BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex] Two consecutive
        // availability failures open the fixed slot. The first retry after the
        // monotonic cooldown is half-open; another availability failure
        // immediately reopens it, while a verified success fully resets it.
        assert_eq!(
            circuit_breaker.decision(0, started_at),
            CommitmentCarrierCircuitDecision::Closed
        );
        circuit_breaker.record_availability_failure(0, started_at);
        assert_eq!(
            circuit_breaker.decision(0, started_at),
            CommitmentCarrierCircuitDecision::Closed
        );
        circuit_breaker.record_availability_failure(0, started_at);
        assert_eq!(
            circuit_breaker.decision(
                0,
                started_at + PINNED_CARRIER_RECOVERY_COOLDOWN - Duration::from_secs(1)
            ),
            CommitmentCarrierCircuitDecision::Cooling
        );

        let half_open_at = started_at + PINNED_CARRIER_RECOVERY_COOLDOWN;
        assert_eq!(
            circuit_breaker.decision(0, half_open_at),
            CommitmentCarrierCircuitDecision::HalfOpen
        );
        circuit_breaker.record_availability_failure(0, half_open_at);
        assert_eq!(
            circuit_breaker.decision(0, half_open_at),
            CommitmentCarrierCircuitDecision::Cooling
        );
        assert_eq!(
            circuit_breaker.decision(0, half_open_at + PINNED_CARRIER_RECOVERY_COOLDOWN),
            CommitmentCarrierCircuitDecision::HalfOpen
        );

        circuit_breaker.record_success(0);
        assert_eq!(
            circuit_breaker.decision(0, half_open_at),
            CommitmentCarrierCircuitDecision::Closed
        );

        circuit_breaker.record_availability_failure(0, half_open_at);
        circuit_breaker.record_availability_failure(0, half_open_at);
        assert_eq!(
            circuit_breaker.decision(0, half_open_at),
            CommitmentCarrierCircuitDecision::Cooling
        );
        circuit_breaker.align_slots(1);
        assert_eq!(
            circuit_breaker.decision(0, half_open_at),
            CommitmentCarrierCircuitDecision::Closed
        );
    }

    #[tokio::test]
    async fn certificate_carrier_circuit_skips_repeated_outages_across_rounds() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let first_carrier = IdentityKeyPair::generate();
        let second_carrier = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let destination = MemoryStorage::open(":memory:", None).unwrap();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x75; 32]],
            &coordinator,
        );
        destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_sync(false, true);
        destination.configure_record_commitment_certificate_policy(2, 2);

        let peer_store = PeerStore::new();
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test client");
        let carrier_ids = [
            first_carrier.public_key_bytes(),
            second_carrier.public_key_bytes(),
        ];
        let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

        // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] The coordinator is
        // still attempted every round. Each missing pinned carrier is attempted
        // twice, then its anonymous slot cools and the third round avoids both
        // requests without changing certificate policy or trust.
        for _ in 0..3 {
            let error =
                sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
                    &destination,
                    &peer_store,
                    &follower,
                    &coordinator.public_key_bytes(),
                    &carrier_ids,
                    2,
                    1,
                    &client,
                    &allow_test_endpoint,
                    &mut circuit_breaker,
                )
                .await
                .unwrap_err();
            assert_eq!(error, "certificate_source_unavailable");
        }

        assert_eq!(
            circuit_breaker.decision(0, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );
        assert_eq!(
            circuit_breaker.decision(1, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );
        let status = destination.record_commitment_sync_status();
        assert_eq!(status.certificate_sync_rounds_total, 3);
        assert_eq!(status.certificate_carrier_attempts_total, 4);
        assert_eq!(status.certificate_availability_exhausted_total, 3);
        assert_eq!(status.certificate_security_stops_total, 0);
        assert_eq!(status.certificate_carrier_cooling_slots, 2);
        assert_eq!(status.certificate_carrier_cooldown_skips_total, 2);
        assert_eq!(status.certificate_carrier_half_open_attempts_total, 0);
    }

    #[tokio::test]
    async fn coordinator_certificate_recovery_stops_before_later_carrier_on_security_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let malformed_carrier = IdentityKeyPair::generate();
        let later_carrier = IdentityKeyPair::generate();
        let destination = MemoryStorage::open(":memory:", None).unwrap();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x76; 32]],
            &coordinator,
        );
        destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_certificate_policy(2, 2);

        let malformed_router = Router::new().route(
            "/api/memchain/peer/checkpoint-certificate",
            post(|| async { (StatusCode::OK, vec![0u8]) }),
        );
        let malformed_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let malformed_address = malformed_listener.local_addr().unwrap();
        let malformed_server = tokio::spawn(async move {
            axum::serve(malformed_listener, malformed_router)
                .await
                .unwrap();
        });

        let later_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&later_hits);
        let later_router = Router::new().route(
            "/api/memchain/peer/checkpoint-certificate",
            post(move || {
                let hits = Arc::clone(&handler_hits);
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }),
        );
        let later_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let later_address = later_listener.local_addr().unwrap();
        let later_server = tokio::spawn(async move {
            axum::serve(later_listener, later_router).await.unwrap();
        });

        let peer_store = PeerStore::new();
        admit_peer(
            &peer_store,
            &malformed_carrier,
            Some(format!("http://{malformed_address}")),
            now,
        );
        admit_peer(
            &peer_store,
            &later_carrier,
            Some(format!("http://{later_address}")),
            now,
        );
        let carrier_ids = [
            malformed_carrier.public_key_bytes(),
            later_carrier.public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("test client");
        let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

        // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] A later healthy
        // or merely responsive carrier must never hide an earlier malformed
        // signed-protocol response from an exact operator pin.
        let recovery =
            recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
                &destination,
                &peer_store,
                &coordinator,
                &carrier_ids,
                2,
                2,
                &client,
                &allow_test_endpoint,
                &mut circuit_breaker,
            )
            .await;
        assert_eq!(
            recovery.disposition,
            CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped
        );
        assert_eq!(recovery.carrier_attempts, 1);
        assert_eq!(recovery.cooldown_skips, 0);
        assert_eq!(recovery.half_open_attempts, 0);
        assert_eq!(recovery.cooling_slots, 0);
        assert_eq!(later_hits.load(Ordering::SeqCst), 0);

        malformed_server.abort();
        let _ = malformed_server.await;
        later_server.abort();
        let _ = later_server.await;
    }

    #[tokio::test]
    async fn coordinator_certificate_recovery_cools_repeatedly_unavailable_carriers() {
        let coordinator = IdentityKeyPair::generate();
        let first_carrier = IdentityKeyPair::generate();
        let second_carrier = IdentityKeyPair::generate();
        let destination = MemoryStorage::open(":memory:", None).unwrap();
        let peer_store = PeerStore::new();
        let carrier_ids = [
            first_carrier.public_key_bytes(),
            second_carrier.public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test client");
        let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

        // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Coordinator
        // backfill retains only anonymous slot health. Two failed rounds open
        // both circuits; the third performs no transport attempt.
        for expected_attempts in [2, 2, 0] {
            let recovery =
                recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
                    &destination,
                    &peer_store,
                    &coordinator,
                    &carrier_ids,
                    2,
                    2,
                    &client,
                    &allow_test_endpoint,
                    &mut circuit_breaker,
                )
                .await;
            assert_eq!(
                recovery.disposition,
                CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted
            );
            assert_eq!(recovery.carrier_attempts, expected_attempts);
        }
        assert_eq!(
            circuit_breaker.decision(0, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );
        assert_eq!(
            circuit_breaker.decision(1, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );
        let final_round =
            recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
                &destination,
                &peer_store,
                &coordinator,
                &carrier_ids,
                2,
                2,
                &client,
                &allow_test_endpoint,
                &mut circuit_breaker,
            )
            .await;
        assert_eq!(final_round.carrier_attempts, 0);
        assert_eq!(final_round.cooldown_skips, 2);
        assert_eq!(final_round.cooling_slots, 2);
    }

    #[tokio::test]
    async fn carrier_circuit_breaker_skips_repeated_outage_across_sync_rounds() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let unavailable_carrier = IdentityKeyPair::generate();
        let live_carrier = Arc::new(IdentityKeyPair::generate());
        let follower = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x74; 32]],
            &coordinator,
        );
        source
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();

        let live_peers = Arc::new(PeerStore::new());
        admit_peer(&live_peers, &follower, None, now);
        let live_router = build_memchain_peer_router(
            Arc::clone(&source),
            live_peers,
            Arc::clone(&live_carrier),
        );
        let live_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let live_address = live_listener.local_addr().unwrap();
        let live_server = tokio::spawn(async move {
            axum::serve(live_listener, live_router).await.unwrap();
        });

        let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let coordinator_address = coordinator_listener.local_addr().unwrap();
        let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let unavailable_address = unavailable_listener.local_addr().unwrap();
        drop(coordinator_listener);
        drop(unavailable_listener);

        let destination = MemoryStorage::open(":memory:", None).unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination.configure_record_commitment_sync(false, true);
        destination.configure_record_commitment_certificate_policy(2, 2);
        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            &coordinator,
            Some(format!("http://{coordinator_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &unavailable_carrier,
            Some(format!("http://{unavailable_address}")),
            now,
        );
        admit_peer(
            &destination_peers,
            &live_carrier,
            Some(format!("http://{live_address}")),
            now,
        );
        let carrier_ids = [
            unavailable_carrier.public_key_bytes(),
            live_carrier.public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();

        // Each new cursor models a new follower round. The unavailable first
        // pin is contacted in rounds one and two, then its process-only slot
        // cools down while the exact second pin continues verified delivery.
        for expected_attempts in [2, 2, 1] {
            let mut cursor = CommitmentBlockCarrierCursor::default();
            let outcome = pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
                &destination,
                &destination_peers,
                &follower,
                &coordinator.public_key_bytes(),
                &carrier_ids,
                2,
                &client,
                &allow_test_endpoint,
                &mut cursor,
                &mut circuit_breaker,
                MAX_BLOCKS_PER_RESPONSE_WIRE,
            )
            .await
            .unwrap();
            assert_eq!(outcome.source, CommitmentSyncPageSource::PinnedCarrier);
            assert_eq!(outcome.carrier_attempts, expected_attempts);
            assert_eq!(outcome.page.remote_tip_height, 1);
            assert!(!outcome.page.has_more);
        }

        assert_eq!(
            circuit_breaker.decision(0, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );

        // Force only the monotonic deadline to expire. The next real request
        // is counted as half-open, fails availability, and reopens the same
        // anonymous slot before the verified second carrier recovers the page.
        circuit_breaker.slots[0].retry_after = Some(Instant::now());
        let mut cursor = CommitmentBlockCarrierCursor::default();
        let half_open_outcome =
            pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
                &destination,
                &destination_peers,
                &follower,
                &coordinator.public_key_bytes(),
                &carrier_ids,
                2,
                &client,
                &allow_test_endpoint,
                &mut cursor,
                &mut circuit_breaker,
                MAX_BLOCKS_PER_RESPONSE_WIRE,
            )
            .await
            .unwrap();
        assert_eq!(
            half_open_outcome.source,
            CommitmentSyncPageSource::PinnedCarrier
        );
        assert_eq!(half_open_outcome.carrier_attempts, 2);
        assert_eq!(
            circuit_breaker.decision(0, Instant::now()),
            CommitmentCarrierCircuitDecision::Cooling
        );
        assert_eq!(destination.record_commitment_chain_tip().await.0, 1);
        destination.audit_record_commitment_chain().await.unwrap();
        let status = destination.record_commitment_sync_status();
        assert_eq!(status.block_page_pulls_total, 4);
        assert_eq!(status.block_carrier_attempts_total, 7);
        assert_eq!(status.block_carrier_recoveries_total, 4);
        assert_eq!(status.block_page_security_stops_total, 0);
        assert_eq!(status.block_carrier_cooling_slots, 1);
        assert_eq!(status.block_carrier_cooldown_skips_total, 1);
        assert_eq!(status.block_carrier_half_open_attempts_total, 1);

        live_server.abort();
        let _ = live_server.await;
    }

    #[tokio::test]
    async fn verified_delivery_anchor_witness_round_is_signed_contiguous_and_bounded() {
        let now = now_secs();
        let requester = IdentityKeyPair::generate();
        let witness = Arc::new(IdentityKeyPair::generate());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_peers = Arc::new(PeerStore::new());
        admit_peer(&witness_peers, &requester, None, now);
        let router = build_memchain_peer_router(
            witness_storage,
            Arc::clone(&witness_peers),
            Arc::clone(&witness),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let requester_peers = PeerStore::new();
        admit_peer(
            &requester_peers,
            &witness,
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let witness_id = witness.public_key_bytes();
        let digest_10 = [0x81; 32];

        let denied = witness_verified_delivery_anchor_with_endpoint_policy(
            &requester_peers,
            &requester,
            &client,
            &[witness_id],
            9,
            &[0x80; 32],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(denied.attempted, 1);
        assert_eq!(denied.verified, 0);
        assert_eq!(denied.failed, 1);

        witness_peers
            .configure_verified_delivery_witness_requesters(&[requester.public_key_bytes()]);

        let advanced = witness_verified_delivery_anchor_with_endpoint_policy(
            &requester_peers,
            &requester,
            &client,
            &[witness_id, witness_id],
            10,
            &digest_10,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(
            advanced,
            VerifiedDeliveryAnchorWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                advanced: 1,
                ..VerifiedDeliveryAnchorWitnessRound::default()
            }
        );

        let idempotent = witness_verified_delivery_anchor_with_endpoint_policy(
            &requester_peers,
            &requester,
            &client,
            &[witness_id],
            10,
            &digest_10,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(idempotent.verified, 1);
        assert_eq!(idempotent.idempotent, 1);

        let gap = witness_verified_delivery_anchor_with_endpoint_policy(
            &requester_peers,
            &requester,
            &client,
            &[witness_id],
            12,
            &[0x82; 32],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(gap.verified, 1);
        assert_eq!(gap.gaps, 1);

        let advanced_next = witness_verified_delivery_anchor_with_endpoint_policy(
            &requester_peers,
            &requester,
            &client,
            &[witness_id],
            11,
            &[0x83; 32],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(advanced_next.verified, 1);
        assert_eq!(advanced_next.advanced, 1);
        server.abort();
    }

    #[tokio::test]
    async fn delivery_witness_authenticates_before_admission_and_rejects_padding() {
        // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] The same forged
        // identity receives the same signature failure before and after local
        // admission. Alternate encodings never reach durable state.
        let now = now_secs();
        let requester = IdentityKeyPair::from_bytes(&[0x81; 32]).expect("requester identity");
        let witness = Arc::new(IdentityKeyPair::from_bytes(&[0x82; 32]).expect("witness identity"));
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peers = Arc::new(PeerStore::new());
        admit_peer(&peers, &requester, None, now);
        let router = build_memchain_peer_router(storage, Arc::clone(&peers), witness);
        let valid_frame =
            delivery_witness_request_frame(&requester, 1, [0x83; 32], [0x84; 16], now);

        let mut forged_message =
            decode_memchain(&valid_frame[1..]).expect("decode delivery request for tamper");
        let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            ref mut signature, ..
        } = forged_message
        else {
            panic!("expected delivery witness request");
        };
        signature[0] ^= 0x01;
        let forged_frame =
            encode_memchain(&forged_message).expect("encode forged delivery request");
        assert_eq!(
            post_delivery_witness(&router, forged_frame.clone())
                .await
                .status(),
            StatusCode::UNAUTHORIZED
        );

        let mut padded_frame = valid_frame.clone();
        padded_frame.push(0);
        assert_eq!(
            post_delivery_witness(&router, padded_frame).await.status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            post_delivery_witness(&router, valid_frame.clone())
                .await
                .status(),
            StatusCode::FORBIDDEN
        );

        peers.configure_verified_delivery_witness_requesters(&[requester.public_key_bytes()]);
        assert_eq!(
            post_delivery_witness(&router, forged_frame).await.status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            post_delivery_witness(&router, valid_frame).await.status(),
            StatusCode::OK
        );
    }

    #[test]
    fn custody_witness_planner_is_bounded_independent_and_non_transmitting() {
        // [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] A pure plan is safe
        // to run in unit tests without an HTTP client or custody anchor.
        let now = now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0x91; 32]).expect("producer identity");
        let eligible = IdentityKeyPair::from_bytes(&[0x92; 32]).expect("eligible witness");
        let unavailable = IdentityKeyPair::from_bytes(&[0x93; 32]).expect("unavailable witness");
        let wrong_capability =
            IdentityKeyPair::from_bytes(&[0x94; 32]).expect("wrong-capability witness");
        let peers = PeerStore::new();
        admit_peer(
            &peers,
            &eligible,
            Some("http://127.0.0.1:8422".to_string()),
            now,
        );
        admit_peer(&peers, &unavailable, None, now);
        let mut wrong_capability_descriptor = NodeDescriptor::new(
            wrong_capability.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        wrong_capability_descriptor.public_endpoint = Some("http://127.0.0.1:8422".to_string());
        wrong_capability_descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let wrong_capability_descriptor =
            SignedNodeDescriptor::sign(wrong_capability_descriptor, &wrong_capability)
                .expect("sign wrong-capability descriptor");
        let import = peers.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: wrong_capability_descriptor,
            },
            now,
        );
        assert_eq!(import.inserted, 1);

        let deduplicated = plan_custody_audit_witnesses_with_endpoint_policy(
            &peers,
            &producer.public_key_bytes(),
            &[
                eligible.public_key_bytes(),
                eligible.public_key_bytes(),
                producer.public_key_bytes(),
            ],
            1,
            now,
            &allow_test_endpoint,
        )
        .expect("plan duplicate and self pins");
        assert_eq!(
            deduplicated,
            CustodyAuditWitnessPlan {
                configured: 1,
                eligible: 1,
                unavailable: 0,
                duplicates_ignored: 1,
                self_excluded: 1,
                minimum_verified: 1,
                quorum_ready: true,
            }
        );

        let insufficient = plan_custody_audit_witnesses_with_endpoint_policy(
            &peers,
            &producer.public_key_bytes(),
            &[eligible.public_key_bytes(), unavailable.public_key_bytes()],
            2,
            now,
            &allow_test_endpoint,
        )
        .expect("plan insufficient current eligibility");
        assert_eq!(insufficient.configured, 2);
        assert_eq!(insufficient.eligible, 1);
        assert_eq!(insufficient.unavailable, 1);
        assert!(!insufficient.quorum_ready);

        let wrong_capability_plan = plan_custody_audit_witnesses_with_endpoint_policy(
            &peers,
            &producer.public_key_bytes(),
            &[wrong_capability.public_key_bytes()],
            1,
            now,
            &allow_test_endpoint,
        )
        .expect("plan rejects a witness without encrypted-storage capability");
        assert_eq!(wrong_capability_plan.eligible, 0);
        assert_eq!(wrong_capability_plan.unavailable, 1);
        assert!(!wrong_capability_plan.quorum_ready);

        assert!(plan_custody_audit_witnesses_with_endpoint_policy(
            &peers,
            &producer.public_key_bytes(),
            &[],
            0,
            now,
            &allow_test_endpoint,
        )
        .is_err());
        assert!(plan_custody_audit_witnesses_with_endpoint_policy(
            &peers,
            &producer.public_key_bytes(),
            &[[0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32]],
            1,
            now,
            &allow_test_endpoint,
        )
        .is_err());
    }

    #[tokio::test]
    async fn custody_audit_witness_endpoint_is_pinned_signed_and_contiguous() {
        // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Exercise the public
        // handler in process: no external endpoint or custody metadata leaves
        // this test while admission, durable outcomes, and signatures remain
        // identical to production routing.
        let now = now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0xA1; 32]).expect("producer identity");
        let witness = Arc::new(IdentityKeyPair::from_bytes(&[0xA2; 32]).expect("witness identity"));
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peers = Arc::new(PeerStore::new());
        admit_peer(&peers, &producer, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&storage),
            Arc::clone(&peers),
            Arc::clone(&witness),
        );
        let anchor_10 = CustodyAuditAnchorV1::signed(10, 100, 10_000, [0xA3; 32], &producer)
            .expect("sign generation 10 anchor");
        let anchor_10_sha =
            custody_audit_anchor_frame_sha256(&anchor_10).expect("hash generation 10 anchor");

        let mut forged_message = decode_memchain(
            &custody_witness_request_frame(&producer, &anchor_10, [0xC1; 16], now)[1..],
        )
        .expect("decode request for signature tamper");
        let MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
            ref mut signature, ..
        } = forged_message
        else {
            panic!("expected custody witness request");
        };
        signature[0] ^= 0x01;
        let forged_frame =
            encode_memchain(&forged_message).expect("encode signature-tampered request");
        let forged_unpinned = post_custody_witness(&router, forged_frame.clone()).await;
        assert_eq!(forged_unpinned.status(), StatusCode::UNAUTHORIZED);

        let denied = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &anchor_10, [0xA4; 16], now),
        )
        .await;
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);

        peers.configure_custody_audit_witness_requesters(&[producer.public_key_bytes()]);
        let forged = post_custody_witness(&router, forged_frame).await;
        assert_eq!(forged.status(), StatusCode::UNAUTHORIZED);

        let stale = post_custody_witness(
            &router,
            custody_witness_request_frame(
                &producer,
                &anchor_10,
                [0xC2; 16],
                now.saturating_sub(REQUEST_TIMESTAMP_SKEW_SECS + 1),
            ),
        )
        .await;
        assert_eq!(stale.status(), StatusCode::UNAUTHORIZED);

        let advanced_request_id = [0xA4; 16];
        let advanced = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &anchor_10, advanced_request_id, now),
        )
        .await;
        assert_eq!(advanced.status(), StatusCode::OK);
        let advanced_body = axum::body::to_bytes(advanced.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let advanced_receipt = verify_custody_audit_anchor_witness_response(
            &advanced_body,
            &advanced_request_id,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            &anchor_10,
            &anchor_10_sha,
            now,
        )
        .expect("verify advanced custody receipt");
        assert_eq!(advanced_receipt.outcome, CUSTODY_AUDIT_WITNESS_ADVANCED_V1);

        let replayed = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &anchor_10, advanced_request_id, now),
        )
        .await;
        assert_eq!(replayed.status(), StatusCode::TOO_MANY_REQUESTS);

        let idempotent_request_id = [0xA5; 16];
        let idempotent = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &anchor_10, idempotent_request_id, now),
        )
        .await;
        assert_eq!(idempotent.status(), StatusCode::OK);
        let idempotent_body = axum::body::to_bytes(idempotent.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let idempotent_receipt = verify_custody_audit_anchor_witness_response(
            &idempotent_body,
            &idempotent_request_id,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            &anchor_10,
            &anchor_10_sha,
            now,
        )
        .expect("verify idempotent custody receipt");
        assert_eq!(
            idempotent_receipt.outcome,
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1
        );

        let conflicting_anchor =
            CustodyAuditAnchorV1::signed(10, 101, 10_001, [0xA6; 32], &producer)
                .expect("sign conflicting anchor");
        let conflict_request_id = [0xA7; 16];
        let conflict = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &conflicting_anchor, conflict_request_id, now),
        )
        .await;
        assert_eq!(conflict.status(), StatusCode::OK);
        let conflict_body = axum::body::to_bytes(conflict.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let conflict_receipt = verify_custody_audit_anchor_witness_response(
            &conflict_body,
            &conflict_request_id,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            &conflicting_anchor,
            &custody_audit_anchor_frame_sha256(&conflicting_anchor)
                .expect("hash conflicting anchor"),
            now,
        )
        .expect("verify conflict custody receipt");
        assert_eq!(conflict_receipt.outcome, CUSTODY_AUDIT_WITNESS_CONFLICT_V1);

        let gap_anchor = CustodyAuditAnchorV1::signed(12, 120, 12_000, [0xA8; 32], &producer)
            .expect("sign gap anchor");
        let gap_request_id = [0xA9; 16];
        let gap = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &gap_anchor, gap_request_id, now),
        )
        .await;
        assert_eq!(gap.status(), StatusCode::OK);
        let gap_body = axum::body::to_bytes(gap.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let gap_receipt = verify_custody_audit_anchor_witness_response(
            &gap_body,
            &gap_request_id,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            &gap_anchor,
            &custody_audit_anchor_frame_sha256(&gap_anchor).expect("hash gap anchor"),
            now,
        )
        .expect("verify gap custody receipt");
        assert_eq!(gap_receipt.outcome, CUSTODY_AUDIT_WITNESS_GAP_V1);

        let anchor_11 = CustodyAuditAnchorV1::signed(11, 110, 11_000, [0xAA; 32], &producer)
            .expect("sign generation 11 anchor");
        let next_request_id = [0xAB; 16];
        let next = post_custody_witness(
            &router,
            custody_witness_request_frame(&producer, &anchor_11, next_request_id, now),
        )
        .await;
        assert_eq!(next.status(), StatusCode::OK);
        let next_body = axum::body::to_bytes(next.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let next_receipt = verify_custody_audit_anchor_witness_response(
            &next_body,
            &next_request_id,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            &anchor_11,
            &custody_audit_anchor_frame_sha256(&anchor_11).expect("hash generation 11 anchor"),
            now,
        )
        .expect("verify generation 11 custody receipt");
        assert_eq!(next_receipt.outcome, CUSTODY_AUDIT_WITNESS_ADVANCED_V1);
    }

    #[tokio::test]
    async fn custody_audit_witness_endpoint_rejects_self_witness_before_persistence() {
        let now = now_secs();
        let identity =
            Arc::new(IdentityKeyPair::from_bytes(&[0xB1; 32]).expect("self witness identity"));
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peers = Arc::new(PeerStore::new());
        admit_peer(&peers, &identity, None, now);
        peers.configure_custody_audit_witness_requesters(&[identity.public_key_bytes()]);
        let router = build_memchain_peer_router(storage, peers, Arc::clone(&identity));
        let anchor = CustodyAuditAnchorV1::signed(1, 1, 1, [0xB2; 32], &identity)
            .expect("sign self custody anchor");
        let response = post_custody_witness(
            &router,
            custody_witness_request_frame(&identity, &anchor, [0xB3; 16], now),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn verified_delivery_anchor_response_rejects_forged_outcome_relation() {
        let requester = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let request_id = [0x91; 16];
        let digest = [0x92; 32];
        let now = now_secs();
        let signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
            &request_id,
            &requester.public_key_bytes(),
            10,
            &digest,
            &witness.public_key_bytes(),
            now,
            9,
            &[0x93; 32],
            VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
        );
        let frame = encode_memchain(&MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
            request_id,
            requester: requester.public_key_bytes(),
            requested_generation: 10,
            requested_anchor_digest: digest,
            witness: witness.public_key_bytes(),
            response_timestamp: now,
            witness_generation: 9,
            witness_anchor_digest: [0x93; 32],
            outcome: VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
            signature: witness.sign(&signing_bytes),
        })
        .unwrap();

        assert_eq!(
            verify_delivery_anchor_witness_response(
                &frame,
                &request_id,
                &requester.public_key_bytes(),
                10,
                &digest,
                &witness.public_key_bytes(),
                now,
            )
            .unwrap_err(),
            "delivery_witness_outcome_invalid"
        );
    }

    fn block_announce_frame(block: &RecordCommitmentBlockV1) -> Vec<u8> {
        encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
            header: block.header.clone(),
            proposer_signature: block.proposer_signature,
        })
        .unwrap()
    }

    #[test]
    fn tip_announcement_status_contract_is_exact() {
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::ACCEPTED.as_u16()),
            CommitmentTipAnnouncementDelivery::Accepted
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::NO_CONTENT.as_u16()),
            CommitmentTipAnnouncementDelivery::Stale
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::OK.as_u16()),
            CommitmentTipAnnouncementDelivery::PermanentFailure
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::SERVICE_UNAVAILABLE.as_u16()),
            CommitmentTipAnnouncementDelivery::RetryableFailure
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::TOO_MANY_REQUESTS.as_u16()),
            CommitmentTipAnnouncementDelivery::PermanentFailure
        );
    }

    #[tokio::test]
    async fn pinned_block_announcement_wakes_follower_without_mutating_chain() {
        let now = now_secs();
        let follower = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        storage.configure_record_commitment_sync(false, true);
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            Arc::clone(&storage),
            peer_store,
            follower,
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x31; 32]],
            &coordinator,
        );
        let frame = block_announce_frame(&block);

        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let accepted = storage.record_commitment_sync_status();
        assert_eq!(
            accepted.last_announcement_result.as_deref(),
            Some("accepted")
        );
        assert_eq!(accepted.announcements_accepted_total, 1);

        let next_block = RecordCommitmentBlockV1::new_signed(
            2,
            now,
            block.header.hash(),
            vec![[0x32; 32]],
            &coordinator,
        );
        let coalesced = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(block_announce_frame(&next_block)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(coalesced.status(), StatusCode::ACCEPTED);
        let coalesced_status = storage.record_commitment_sync_status();
        assert_eq!(
            coalesced_status.last_announcement_result.as_deref(),
            Some("coalesced")
        );
        assert_eq!(coalesced_status.last_announced_height, Some(2));
        assert_eq!(coalesced_status.announcements_accepted_total, 1);
        assert_eq!(coalesced_status.announcements_coalesced_total, 1);
        assert_eq!(notifications.recv().await, Some(1));
        assert_eq!(storage.record_commitment_chain_tip().await.0, 0);

        let retry = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::ACCEPTED);
        assert_eq!(notifications.recv().await, Some(1));
        let retry_status = storage.record_commitment_sync_status();
        assert_eq!(
            retry_status.last_announcement_result.as_deref(),
            Some("accepted")
        );
        assert_eq!(retry_status.last_announced_height, Some(2));
        assert_eq!(retry_status.announcements_accepted_total, 2);
        assert_eq!(retry_status.announcements_coalesced_total, 1);
        assert_eq!(storage.record_commitment_chain_tip().await.0, 0);
    }

    #[tokio::test]
    async fn block_announcement_rejects_unpinned_or_invalid_proposer() {
        let now = now_secs();
        let follower = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let unpinned = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        admit_peer(&peer_store, &unpinned, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            storage,
            peer_store,
            follower,
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let unpinned_block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x32; 32]],
            &unpinned,
        );
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(block_announce_frame(&unpinned_block)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(notifications.try_recv().is_err());

        let coordinator_block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x34; 32]],
            &coordinator,
        );
        let coordinator_block_hash = coordinator_block.hash();
        let invalid_signature = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
            header: coordinator_block.header,
            proposer_signature: unpinned.sign(&coordinator_block_hash),
        })
        .unwrap();
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(invalid_signature))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(notifications.try_recv().is_err());
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_reaches_pinned_follower_runtime() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = Arc::new(IdentityKeyPair::generate());
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x33; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        follower_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let follower_peers = Arc::new(PeerStore::new());
        admit_peer(&follower_peers, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            follower_storage,
            follower_peers,
            Arc::clone(&follower),
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(outcome.announced_height, 1);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.stale, 0);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 0);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 0);
        assert_eq!(notifications.recv().await, Some(1));

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_reports_current_follower_as_stale() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = Arc::new(IdentityKeyPair::generate());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x34; 32]],
            &coordinator,
        );

        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        follower_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        follower_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let follower_peers = Arc::new(PeerStore::new());
        admit_peer(&follower_peers, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            follower_storage,
            follower_peers,
            Arc::clone(&follower),
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(outcome.announced_height, 1);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.stale, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 0);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 0);
        assert!(notifications.try_recv().is_err());

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_retries_transient_failure_to_success() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x35; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let handler_attempts = Arc::clone(&attempts);
        let endpoint_checks = AtomicUsize::new(0);
        let endpoint_policy = |_endpoint: &str| {
            endpoint_checks.fetch_add(1, Ordering::SeqCst);
            true
        };
        let router = Router::new().route(
            "/api/memchain/peer/block-announce",
            post(move || {
                let attempts = Arc::clone(&handler_attempts);
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        StatusCode::ACCEPTED
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &endpoint_policy,
            CommitmentTipAnnouncementRetryPolicy {
                max_attempts: 3,
                base_delay: Duration::ZERO,
            },
        )
        .await
        .unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(endpoint_checks.load(Ordering::SeqCst), 2);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 1);
        assert_eq!(outcome.retries_succeeded, 1);
        assert_eq!(outcome.retries_exhausted, 0);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_stops_after_retry_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x36; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let handler_attempts = Arc::clone(&attempts);
        let router = Router::new().route(
            "/api/memchain/peer/block-announce",
            post(move || {
                let attempts = Arc::clone(&handler_attempts);
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
            CommitmentTipAnnouncementRetryPolicy {
                max_attempts: 3,
                base_delay: Duration::ZERO,
            },
        )
        .await
        .unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.failed, 1);
        assert_eq!(outcome.retries_attempted, 2);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 1);

        server.abort();
        let _ = server.await;
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

    #[tokio::test]
    async fn coordinator_lease_endpoint_grants_renews_and_rejects_competing_instance() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            Arc::clone(&storage),
            peer_store,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let first_instance = [0x71; 32];
        let first_request_id = [0x72; 16];
        let first_frame = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            first_request_id,
            now,
        );
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(first_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let grant = verify_record_commitment_coordinator_lease_response(
            &body,
            &first_request_id,
            &coordinator.public_key_bytes(),
            &first_instance,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            now,
        )
        .unwrap();
        assert_eq!(grant.lease_epoch, 1);

        let renewal = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x73; 16],
            now,
        );
        let renewed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(renewal))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(renewed.status(), StatusCode::OK);

        let competing = coordinator_lease_request_frame(
            &coordinator,
            [0x74; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x75; 16],
            now,
        );
        let rejected = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(competing))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn coordinator_lease_endpoint_releases_exact_holder_and_hands_over_immediately() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            storage,
            peer_store,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let first_instance = [0x76; 32];
        let second_instance = [0x77; 32];
        let acquire = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x78; 16],
            now,
        );
        let acquired = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(acquire))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(acquired.status(), StatusCode::OK);

        let wrong_release =
            coordinator_lease_release_request_frame(&coordinator, second_instance, [0x79; 16], now);
        let wrong_release_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(wrong_release))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_release_response.status(), StatusCode::CONFLICT);

        let release_request_id = [0x7A; 16];
        let release_frame = coordinator_lease_release_request_frame(
            &coordinator,
            first_instance,
            release_request_id,
            now,
        );
        let released = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(release_frame.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(released.status(), StatusCode::OK);
        let body = axum::body::to_bytes(released.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let release_ack = verify_record_commitment_coordinator_lease_release_response(
            &body,
            &release_request_id,
            &coordinator.public_key_bytes(),
            &first_instance,
            &witness.public_key_bytes(),
            now,
        )
        .unwrap();
        assert_eq!(release_ack.lease_epoch, 1);

        let replay = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(release_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);

        let delayed_renewal = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x7B; 16],
            now,
        );
        let delayed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(delayed_renewal))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delayed.status(), StatusCode::CONFLICT);

        let takeover = coordinator_lease_request_frame(
            &coordinator,
            second_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x7C; 16],
            now,
        );
        let takeover_response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(takeover))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(takeover_response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(takeover_response.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let grant = verify_record_commitment_coordinator_lease_response(
            &body,
            &[0x7C; 16],
            &coordinator.public_key_bytes(),
            &second_instance,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            now,
        )
        .unwrap();
        assert_eq!(grant.lease_epoch, 2);
    }

    #[tokio::test]
    async fn coordinator_lease_endpoint_rejects_unpinned_invalid_and_wrong_tip_requests() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let unpinned = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        admit_peer(&peer_store, &unpinned, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            storage,
            peer_store,
            witness,
            Some(coordinator.public_key_bytes()),
        );

        let unpinned_frame = coordinator_lease_request_frame(
            &unpinned,
            [0x81; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x82; 16],
            now,
        );
        let unpinned_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(unpinned_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unpinned_response.status(), StatusCode::FORBIDDEN);

        let mut invalid_signature = coordinator_lease_request_frame(
            &coordinator,
            [0x83; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x84; 16],
            now,
        );
        let last = invalid_signature.len() - 1;
        invalid_signature[last] ^= 0x01;
        let signature_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(invalid_signature))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(signature_response.status(), StatusCode::UNAUTHORIZED);

        let wrong_tip = coordinator_lease_request_frame(
            &coordinator,
            [0x85; 32],
            1,
            [0x86; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x87; 16],
            now,
        );
        let tip_response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(wrong_tip))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tip_response.status(), StatusCode::CONFLICT);
    }

    #[test]
    fn coordinator_lease_response_accepts_processing_time_remainder() {
        let coordinator = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
        let request_id = [0x88; 16];
        let instance_id = [0x89; 32];
        let response_timestamp = 10_001;
        let lease_expires_at = 10_060;
        let signing_bytes = record_coordinator_lease_response_signing_bytes(
            &chain_id,
            &request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            response_timestamp,
            1,
            lease_expires_at,
            0,
            &GENESIS_PREV_HASH,
        );
        let frame = encode_memchain(&MemChainMessage::RecordCoordinatorLeaseResponseV1 {
            chain_id,
            request_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            witness: witness.public_key_bytes(),
            response_timestamp,
            lease_epoch: 1,
            lease_expires_at,
            witness_tip_height: 0,
            witness_tip_hash: GENESIS_PREV_HASH,
            signature: witness.sign(&signing_bytes),
        })
        .unwrap();

        let grant = verify_record_commitment_coordinator_lease_response(
            &frame,
            &request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            response_timestamp,
        )
        .unwrap();
        assert_eq!(grant.valid_for_secs, 59);
    }

    #[tokio::test]
    async fn coordinator_lease_client_verifies_grant_release_and_immediate_handover() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witness = Arc::new(IdentityKeyPair::generate());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        witness_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let witness_peers = Arc::new(PeerStore::new());
        admit_peer(&witness_peers, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            witness_storage,
            witness_peers,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let coordinator_storage = MemoryStorage::open(":memory:", None).unwrap();
        coordinator_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &witness,
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap();
        let grant = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x91; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(grant.lease_epoch, 1);
        assert!(grant.valid_for_secs > 0);

        let error = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x92; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "lease_contended");

        let release = release_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x91; 32],
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(release.lease_epoch, 1);

        let takeover = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x92; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(takeover.lease_epoch, 2);
        server.abort();
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

    #[test]
    fn peer_guard_rejects_replay_and_enforces_rate_limit() {
        let peer = [0x11; 32];
        let wall_now = 1_700_000_000;
        let monotonic_now = Instant::now();
        let mut guard = PeerRequestGuard::default();
        assert!(guard.admit_at(peer, [0x01; 16], wall_now, monotonic_now));
        assert!(!guard.admit_at(peer, [0x01; 16], wall_now, monotonic_now));
        for value in 2..MAX_REQUESTS_PER_PEER_PER_MINUTE {
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&value.to_le_bytes());
            assert!(guard.admit_at(peer, request_id, wall_now, monotonic_now));
        }
        assert!(!guard.admit_at(peer, [0xFF; 16], wall_now, monotonic_now));
        assert!(guard.admit_at(
            peer,
            [0xFF; 16],
            wall_now + 61,
            monotonic_now + PEER_RATE_LIMIT_WINDOW,
        ));
    }

    #[test]
    fn peer_guard_wall_clock_corrections_cannot_reset_rate_budget() {
        let peer = [0x12; 32];
        let wall_now = 1_700_000_000;
        let monotonic_now = Instant::now();
        let mut guard = PeerRequestGuard::default();

        for value in 0..MAX_REQUESTS_PER_PEER_PER_MINUTE {
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&value.to_le_bytes());
            assert!(guard.admit_at(peer, request_id, wall_now, monotonic_now));
        }

        assert!(!guard.admit_at(
            peer,
            [0xFE; 16],
            wall_now.saturating_sub(3_600),
            monotonic_now,
        ));
        assert!(!guard.admit_at(
            peer,
            [0xFD; 16],
            wall_now + 3_600,
            monotonic_now,
        ));
        assert!(guard.admit_at(
            peer,
            [0xFC; 16],
            wall_now + 3_600,
            monotonic_now + PEER_RATE_LIMIT_WINDOW,
        ));
    }

    #[test]
    fn peer_guard_allows_idempotent_hint_retries_within_shared_rate_limit() {
        let peer = [0x22; 32];
        let wall_now = 1_700_000_000;
        let monotonic_now = Instant::now();
        let mut guard = PeerRequestGuard::default();

        for _ in 0..MAX_REQUESTS_PER_PEER_PER_MINUTE {
            assert!(guard.admit_idempotent_hint_at(peer, wall_now, monotonic_now));
        }
        assert!(!guard.admit_idempotent_hint_at(peer, wall_now, monotonic_now));
        assert!(guard.admit_idempotent_hint_at(
            peer,
            wall_now,
            monotonic_now + PEER_RATE_LIMIT_WINDOW,
        ));
    }

    #[test]
    fn commitment_range_url_is_bounded_to_the_peer_api_path() {
        let url = commitment_block_range_url("https://node.example/ignored?secret=no").unwrap();
        assert_eq!(
            url.as_str(),
            "https://node.example/api/memchain/peer/block-range"
        );
        assert!(commitment_block_range_url("ftp://node.example").is_err());
        assert!(commitment_block_range_url("https://user@node.example").is_err());
        assert_eq!(
            commitment_checkpoint_url("node.example:9281/path")
                .unwrap()
                .as_str(),
            "http://node.example:9281/api/memchain/peer/checkpoint"
        );
        assert_eq!(
            commitment_checkpoint_certificate_url("node.example:9281/path")
                .unwrap()
                .as_str(),
            "http://node.example:9281/api/memchain/peer/checkpoint-certificate"
        );
    }

    #[test]
    fn commitment_peer_endpoint_rejects_ssrf_targets() {
        assert!(commitment_peer_endpoint_is_public("http://8.8.8.8:8422"));
        assert!(commitment_peer_endpoint_is_public(
            "https://[2606:4700:4700::1111]:8422"
        ));
        for endpoint in [
            "http://127.0.0.1:8422",
            "http://127.1:8422",
            "http://2130706433:8422",
            "http://0x7f000001:8422",
            "http://017700000001:8422",
            "http://10.0.0.1:8422",
            "http://100.64.0.1:8422",
            "http://169.254.1.1:8422",
            "http://172.16.0.1:8422",
            "http://192.168.1.1:8422",
            "http://198.18.0.1:8422",
            "http://203.0.113.1:8422",
            "http://node.example:8422",
            "http://[::1]:8422",
            "http://[::ffff:127.0.0.1]:8422",
            "http://[fc00::1]:8422",
            "http://[fe80::1]:8422",
            "http://[2001:db8::1]:8422",
        ] {
            assert!(
                !commitment_peer_endpoint_is_public(endpoint),
                "unexpectedly accepted {endpoint}"
            );
        }
    }

    #[tokio::test]
    async fn outbound_commitment_pulls_reject_private_descriptor_targets() {
        let now = now_secs();
        let local_identity = IdentityKeyPair::generate();
        let remote_identity = IdentityKeyPair::generate();
        let peer_store = PeerStore::new();
        admit_peer(
            &peer_store,
            &remote_identity,
            Some("http://169.254.169.254/latest/meta-data".to_string()),
            now,
        );
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let remote_id = remote_identity.public_key_bytes();

        let checkpoint_error = pull_record_commitment_checkpoint(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(checkpoint_error, "pinned_coordinator_unsafe_endpoint");

        let page_error = pull_record_commitment_page(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(page_error, "pinned_coordinator_unsafe_endpoint");

        let certificate_error = pull_record_commitment_checkpoint_certificate(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &[remote_id, IdentityKeyPair::generate().public_key_bytes()],
            2,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(certificate_error, "certificate_source_unsafe_endpoint");
    }

    #[tokio::test]
    async fn witness_reconciliation_rechecks_endpoint_after_selection() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let local_identity = IdentityKeyPair::generate();
        let remote_identity = IdentityKeyPair::generate();
        let peer_store = PeerStore::new();
        admit_peer(
            &peer_store,
            &remote_identity,
            Some("http://8.8.8.8:8422".to_string()),
            now,
        );
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let checks = AtomicUsize::new(0);
        let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
            &storage,
            &peer_store,
            &local_identity,
            &client,
            1,
            |_endpoint| checks.fetch_add(1, Ordering::SeqCst) == 0,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 0);
        assert_eq!(round.failed, 1);
        assert!(checks.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn checkpoint_endpoint_refuses_to_sign_an_unaudited_chain() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &requester_identity, None, now);

        let request_id = [0x91; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_chain_checkpoint_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            0,
            &GENESIS_PREV_HASH,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            known_tip_height: 0,
            known_tip_hash: GENESIS_PREV_HASH,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let response = build_memchain_peer_router(storage, peer_store, responder_identity)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/checkpoint")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn block_range_endpoint_refuses_to_sign_an_unaudited_chain() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &requester_identity, None, now);

        let request_id = [0x92; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            MAX_BLOCKS_PER_RESPONSE_WIRE,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE_WIRE,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let response = build_memchain_peer_router(storage, peer_store, responder_identity)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-range")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn response_verification_rejects_blocks_from_an_unpinned_proposer() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let other_writer = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x44; 32]],
            &other_writer,
        );
        let request_id = [0x33; 16];
        let responder = coordinator.public_key_bytes();
        let blocks = vec![block.clone()];
        let signing_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &blocks,
            false,
            1,
            &block.hash(),
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks,
            has_more: false,
            tip_height: 1,
            tip_hash: block.hash(),
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap();

        let error = verify_record_commitment_page(
            &frame,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err();
        assert_eq!(error, "unexpected_block_proposer");

        // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] A valid carrier
        // envelope cannot promote carrier-authored blocks into the configured
        // coordinator namespace.
        let carrier = other_writer.public_key_bytes();
        let carrier_frame = signed_block_page_frame(
            &other_writer,
            request_id,
            now,
            vec![block.clone()],
            false,
            1,
            block.hash(),
        );
        let error = verify_record_commitment_page(
            &carrier_frame,
            &request_id,
            &carrier,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err();
        assert_eq!(error, "unexpected_block_proposer");
    }

    #[test]
    fn response_verification_rejects_coordinator_rollback_and_fork() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let responder = coordinator.public_key_bytes();
        let request_id = [0x61; 16];
        let local_tip = (1, [0x71; 32]);

        let rollback_signing = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &[],
            false,
            0,
            &GENESIS_PREV_HASH,
        );
        let rollback_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks: Vec::new(),
            has_more: false,
            tip_height: 0,
            tip_hash: GENESIS_PREV_HASH,
            signature: coordinator.sign(&rollback_signing),
        })
        .unwrap();
        assert_eq!(
            verify_record_commitment_page(
                &rollback_frame,
                &request_id,
                &responder,
                &responder,
                local_tip,
                now,
            )
            .unwrap_err(),
            "coordinator_rollback_detected"
        );

        let forked =
            RecordCommitmentBlockV1::new_signed(2, now, [0x72; 32], vec![[0x73; 32]], &coordinator);
        let fork_blocks = vec![forked.clone()];
        let fork_signing = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &fork_blocks,
            false,
            2,
            &forked.hash(),
        );
        let fork_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks: fork_blocks,
            has_more: false,
            tip_height: 2,
            tip_hash: forked.hash(),
            signature: coordinator.sign(&fork_signing),
        })
        .unwrap();
        assert_eq!(
            verify_record_commitment_page(
                &fork_frame,
                &request_id,
                &responder,
                &responder,
                local_tip,
                now,
            )
            .unwrap_err(),
            "commitment_chain_verification_failed"
        );
    }

    #[test]
    fn response_verification_binds_signature_request_and_pagination_metadata() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let responder = coordinator.public_key_bytes();
        let request_id = [0x81; 16];
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x82; 32]],
            &coordinator,
        );
        let valid = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block.clone()],
            false,
            1,
            block.hash(),
        );
        verify_record_commitment_page(
            &valid,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap();

        let invalid_signature_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            std::slice::from_ref(&block),
            false,
            1,
            &block.hash(),
        );
        let mut invalid_signature = coordinator.sign(&invalid_signature_bytes);
        invalid_signature[0] ^= 0x01;
        let invalid_signature_frame =
            encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
                request_id,
                responder,
                response_timestamp: now,
                blocks: vec![block.clone()],
                has_more: false,
                tip_height: 1,
                tip_hash: block.hash(),
                signature: invalid_signature,
            })
            .unwrap();
        assert_eq!(
            verify_record_commitment_page(
                &invalid_signature_frame,
                &request_id,
                &responder,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "invalid_response_signature"
        );
        assert_eq!(
            verify_record_commitment_page(
                &valid,
                &[0x83; 16],
                &responder,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "response_request_mismatch"
        );
        assert_eq!(
            verify_record_commitment_page(
                &valid,
                &request_id,
                &responder,
                &responder,
                (0, GENESIS_PREV_HASH),
                now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS + 1),
            )
            .unwrap_err(),
            "stale_response"
        );

        let inconsistent_tip = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block.clone()],
            false,
            1,
            [0x84; 32],
        );
        assert_eq!(
            verify_record_commitment_page(
                &inconsistent_tip,
                &request_id,
                &responder,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "terminal_tip_mismatch"
        );

        let inconsistent_pagination = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block],
            true,
            1,
            [0x85; 32],
        );
        assert_eq!(
            verify_record_commitment_page(
                &inconsistent_pagination,
                &request_id,
                &responder,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "pagination_state_mismatch"
        );
    }

    #[tokio::test]
    async fn signed_checkpoint_distinguishes_remote_lag_from_divergence() {
        let now = now_secs();
        let responder = IdentityKeyPair::generate();
        let local_writer = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x31; 32]],
            &local_writer,
        );
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x32; 32]],
            &local_writer,
        );
        storage
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();

        let request_id = [0xA2; 16];
        let responder_key = responder.public_key_bytes();
        let lagging_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder_key,
            now,
            1,
            &first.hash(),
            1,
            &first.hash(),
        );
        let lagging_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder: responder_key,
            response_timestamp: now,
            checkpoint_height: 1,
            checkpoint_hash: first.hash(),
            tip_height: 1,
            tip_hash: first.hash(),
            signature: responder.sign(&lagging_signing_bytes),
        })
        .unwrap();
        let lagging = verify_record_commitment_checkpoint(
            &storage,
            &lagging_frame,
            &request_id,
            &responder_key,
            (2, second.hash()),
            now,
        )
        .await
        .unwrap();
        assert_eq!(lagging.relation, CommitmentCheckpointRelation::RemoteBehind);

        let fork_hash = [0xF1; 32];
        let fork_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder_key,
            now,
            2,
            &fork_hash,
            2,
            &fork_hash,
        );
        let fork_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder: responder_key,
            response_timestamp: now,
            checkpoint_height: 2,
            checkpoint_hash: fork_hash,
            tip_height: 2,
            tip_hash: fork_hash,
            signature: responder.sign(&fork_signing_bytes),
        })
        .unwrap();
        let diverged = verify_record_commitment_checkpoint(
            &storage,
            &fork_frame,
            &request_id,
            &responder_key,
            (2, second.hash()),
            now,
        )
        .await
        .unwrap();
        assert_eq!(diverged.relation, CommitmentCheckpointRelation::Diverged);
    }

    #[tokio::test]
    async fn authenticated_range_sync_converges_two_commitment_ledgers() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let destination = MemoryStorage::open(":memory:", None).unwrap();

        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x11; 32], [0x22; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x33; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();
        destination.audit_record_commitment_chain().await.unwrap();

        let peer_store = Arc::new(PeerStore::new());
        let descriptor = NodeDescriptor::new(
            requester_identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        let descriptor = SignedNodeDescriptor::sign(descriptor, &requester_identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);

        let request_id = [0xA7; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            MAX_BLOCKS_PER_RESPONSE_WIRE,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE_WIRE,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let router = build_memchain_peer_router(
            Arc::clone(&source),
            peer_store,
            Arc::clone(&responder_identity),
        );
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-range")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
        let response = decode_memchain(&body[1..]).unwrap();
        let MemChainMessage::RecordBlockRangeResponseV1 {
            request_id: response_request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } = response
        else {
            panic!("expected record block range response");
        };
        assert_eq!(response_request_id, request_id);
        assert_eq!(responder, responder_identity.public_key_bytes());
        assert!(!has_more);
        assert_eq!(tip_height, 2);
        assert_eq!(tip_hash, second.hash());
        let response_signing_bytes = record_block_range_response_signing_bytes(
            &response_request_id,
            &responder,
            response_timestamp,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&response_signing_bytes, &signature)
            .unwrap();

        for block in &blocks {
            destination
                .append_record_commitment_block(block, Some(&responder))
                .await
                .unwrap();
        }
        assert_eq!(blocks, vec![first, second]);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        let status = destination.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 2);
        assert_eq!(status.commitment_count, 3);
    }

    #[tokio::test]
    async fn live_http_follower_pull_converges_with_pinned_coordinator() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x51; 32], [0x52; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x53; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();
        destination.audit_record_commitment_chain().await.unwrap();

        let source_peers = Arc::new(PeerStore::new());
        admit_peer(&source_peers, &requester_identity, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&source),
            source_peers,
            Arc::clone(&responder_identity),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let follower_peers = PeerStore::new();
        admit_peer(
            &follower_peers,
            &responder_identity,
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let before_pull = pull_record_commitment_checkpoint_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            false,
            CommitmentPeerDescriptorPolicy::CurrentOnly,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(
            before_pull.relation,
            CommitmentCheckpointRelation::RemoteAhead
        );
        assert_eq!(before_pull.local_tip_height, 0);
        assert_eq!(before_pull.remote_tip_height, 2);
        let outcome = pull_record_commitment_page_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();

        assert_eq!(outcome.inserted, 2);
        assert_eq!(outcome.already_present, 0);
        assert!(!outcome.has_more);
        assert_eq!(outcome.remote_tip_height, 2);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        let checkpoint = pull_record_commitment_checkpoint_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            false,
            CommitmentPeerDescriptorPolicy::CurrentOnly,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(checkpoint.relation, CommitmentCheckpointRelation::Converged);
        assert_eq!(checkpoint.local_tip_height, 2);
        assert_eq!(checkpoint.remote_tip_height, 2);
        assert_eq!(checkpoint.checkpoint_height, 2);
        assert_ne!(checkpoint.evidence_digest, [0u8; 32]);
        let served = source.record_commitment_checkpoint_status();
        assert_eq!(served.requests_served_total, 2);
        assert!(served.last_served_at.is_some());
        assert_eq!(served.state, "not_checked");
        assert_eq!(served.last_checked_at, None);
        assert_eq!(served.last_divergence_at, None);
        assert_eq!(served.proofs_verified_total, 0);
        assert_eq!(served.divergences_total, 0);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn live_http_follower_rejects_signed_malicious_page_without_mutation() {
        let now = now_secs();
        let coordinator = Arc::new(IdentityKeyPair::generate());
        let requester = IdentityKeyPair::generate();
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        destination.audit_record_commitment_chain().await.unwrap();

        // The pinned coordinator signs both layers, but the block deliberately
        // forks before genesis. Envelope authenticity must never replace chain
        // continuity verification.
        let malicious_block =
            RecordCommitmentBlockV1::new_signed(1, now, [0xF1; 32], vec![[0xF2; 32]], &coordinator);
        let router = Router::new().route(
            "/api/memchain/peer/block-range",
            post({
                let coordinator = Arc::clone(&coordinator);
                move |body: Bytes| {
                    let coordinator = Arc::clone(&coordinator);
                    let malicious_block = malicious_block.clone();
                    async move {
                        assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
                        let request = decode_memchain(&body[1..]).unwrap();
                        let MemChainMessage::RecordBlockRangeRequestV1 { request_id, .. } = request
                        else {
                            panic!("expected commitment block range request");
                        };
                        let tip_hash = malicious_block.hash();
                        let frame = signed_block_page_frame(
                            &coordinator,
                            request_id,
                            now_secs(),
                            vec![malicious_block],
                            false,
                            1,
                            tip_hash,
                        );
                        (
                            StatusCode::OK,
                            [(header::CONTENT_TYPE, "application/octet-stream")],
                            frame,
                        )
                            .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let peers = PeerStore::new();
        admit_peer(&peers, &coordinator, Some(format!("http://{address}")), now);
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let error = pull_record_commitment_page_with_endpoint_policy(
            &destination,
            &peers,
            &requester,
            &coordinator.public_key_bytes(),
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "commitment_chain_verification_failed");
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            (0, GENESIS_PREV_HASH)
        );
        let status = destination.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 0);
        assert_eq!(status.commitment_count, 0);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_witness_round_keeps_valid_proof_over_partial_failure() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let converged_witness = Arc::new(IdentityKeyPair::generate());
        let lagging_witness = Arc::new(IdentityKeyPair::generate());
        let unavailable_witness = IdentityKeyPair::generate();
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let converged = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let lagging = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x71; 32]],
            &coordinator,
        );
        local
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        converged
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        converged.audit_record_commitment_chain().await.unwrap();
        lagging.audit_record_commitment_chain().await.unwrap();

        let converged_peers = Arc::new(PeerStore::new());
        admit_peer(&converged_peers, &coordinator, None, now);
        let converged_router = build_memchain_peer_router(
            Arc::clone(&converged),
            converged_peers,
            Arc::clone(&converged_witness),
        );
        let converged_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let converged_address = converged_listener.local_addr().unwrap();
        let converged_server = tokio::spawn(async move {
            axum::serve(converged_listener, converged_router)
                .await
                .unwrap();
        });

        let lagging_peers = Arc::new(PeerStore::new());
        admit_peer(&lagging_peers, &coordinator, None, now);
        let lagging_router = build_memchain_peer_router(
            Arc::clone(&lagging),
            lagging_peers,
            Arc::clone(&lagging_witness),
        );
        let lagging_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let lagging_address = lagging_listener.local_addr().unwrap();
        let lagging_server = tokio::spawn(async move {
            axum::serve(lagging_listener, lagging_router).await.unwrap();
        });

        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &converged_witness,
            Some(format!("http://{converged_address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &lagging_witness,
            Some(format!("http://{lagging_address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &unavailable_witness,
            Some("https://[invalid".to_string()),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            3,
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 3);
        assert_eq!(round.attempted, 3);
        assert_eq!(round.verified, 2);
        assert_eq!(round.converged, 1);
        assert_eq!(round.remote_behind, 1);
        assert_eq!(round.remote_ahead, 0);
        assert_eq!(round.diverged, 0);
        assert_eq!(round.failed, 1);
        let status = local.record_commitment_checkpoint_status();
        assert_eq!(status.state, "converged");
        assert_eq!(status.proofs_verified_total, 2);
        assert_eq!(status.proofs_failed_total, 1);
        assert_eq!(status.evidence_records, 2);
        assert_eq!(status.evidence_state, "verified");
        assert_eq!(status.last_round_state, "partial");
        assert_eq!(status.last_round_eligible, 3);
        assert_eq!(status.last_round_attempted, 3);
        assert_eq!(status.last_round_verified, 2);
        assert_eq!(status.last_round_failed, 1);
        assert_eq!(status.last_round_converged, 1);
        assert_eq!(status.last_round_remote_ahead, 0);
        assert_eq!(status.last_round_remote_behind, 1);
        assert_eq!(status.last_round_diverged, 0);
        assert!(status.last_round_at.is_some());

        converged_server.abort();
        lagging_server.abort();
        let _ = converged_server.await;
        let _ = lagging_server.await;
    }

    #[tokio::test]
    async fn pinned_witness_round_excludes_unpinned_permissionless_peers() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let pinned_witness = Arc::new(IdentityKeyPair::generate());
        let unpinned_peer = IdentityKeyPair::generate();
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        witness_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let witness_peers = Arc::new(PeerStore::new());
        admit_peer(&witness_peers, &coordinator, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&witness_storage),
            witness_peers,
            Arc::clone(&pinned_witness),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &pinned_witness,
            Some(format!("http://{address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &unpinned_peer,
            Some("https://[invalid".to_string()),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            &[
                pinned_witness.public_key_bytes(),
                pinned_witness.public_key_bytes(),
            ],
            2,
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 1);
        assert_eq!(round.converged, 1);
        assert_eq!(round.failed, 0);
        assert_eq!(round.certificate_signers, 1);
        assert!(!round.certificate_persisted);
        assert_eq!(
            local.record_commitment_checkpoint_status().evidence_records,
            1
        );

        server.abort();
        let _ = server.await;
    }

    // [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] Reproduces a real
    // production outage: strict witness verification and all-witness lease
    // acquisition run before gossip can refresh either side after a long
    // reboot. Expired descriptors are transport/admission hints only; every
    // request and response still verifies against the exact pinned keys.
    #[tokio::test]
    async fn descriptor_preflight_refreshes_legacy_witness_before_strict_gate() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();

        let witness_peers = Arc::new(PeerStore::new());
        let expired_coordinator = NodeDescriptor::new(
            coordinator.public_key_bytes(),
            1,
            now.saturating_sub(1_200),
            now.saturating_sub(600),
            "expired-coordinator-before-preflight",
        );
        let expired_coordinator =
            SignedNodeDescriptor::sign(expired_coordinator, &coordinator).unwrap();
        let imported = witness_peers.load_peer_cache_snapshot_from_source(
            &NodeBootstrapSnapshot::new(now, vec![expired_coordinator]),
            now,
            "test_expired_coordinator_cache",
        );
        assert_eq!(imported.inserted, 1);
        assert!(witness_peers
            .get_valid(&coordinator.public_key_bytes(), now)
            .is_none());

        let router = crate::api::discovery::build_discovery_router(
            Arc::clone(&witness_peers),
            crate::api::discovery::DiscoveryApiPolicy::default(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let mut expired_witness = NodeDescriptor::new(
            witness.public_key_bytes(),
            1,
            now.saturating_sub(1_200),
            now.saturating_sub(600),
            "expired-witness-endpoint-hint",
        );
        expired_witness.public_endpoint = Some(format!("http://{address}"));
        let expired_witness = SignedNodeDescriptor::sign(expired_witness, &witness).unwrap();
        let coordinator_peers = PeerStore::new();
        let imported = coordinator_peers.load_peer_cache_snapshot_from_source(
            &NodeBootstrapSnapshot::new(now, vec![expired_witness]),
            now,
            "test_expired_witness_cache",
        );
        assert_eq!(imported.inserted, 1);

        let mut current_coordinator = NodeDescriptor::new(
            coordinator.public_key_bytes(),
            2,
            now,
            now.saturating_add(600),
            "current-coordinator-after-endpoint-rotation",
        );
        current_coordinator.public_endpoint = Some("http://8.8.8.8:8422".to_string());
        let current_coordinator =
            SignedNodeDescriptor::sign(current_coordinator, &coordinator).unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();

        let round = publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
            &coordinator_peers,
            &current_coordinator,
            &client,
            &[
                witness.public_key_bytes(),
                witness.public_key_bytes(),
                coordinator.public_key_bytes(),
            ],
            &allow_test_endpoint,
        )
        .await;

        assert_eq!(
            round,
            CommitmentWitnessDescriptorPublishRound {
                configured: 1,
                attempted: 1,
                accepted: 1,
                failed: 0,
            }
        );
        let refreshed = witness_peers
            .get_valid(&coordinator.public_key_bytes(), now)
            .unwrap();
        assert_eq!(refreshed.descriptor.sequence, 2);
        assert_eq!(
            refreshed.descriptor.public_endpoint.as_deref(),
            Some("http://8.8.8.8:8422")
        );

        // [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] The discovery API
        // intentionally returns a structured 200 response for stale sequence
        // input. Preflight must parse that receipt instead of misreporting the
        // transport-level success as a descriptor refresh.
        let stale_coordinator = SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                coordinator.public_key_bytes(),
                1,
                now,
                now.saturating_add(600),
                "stale-coordinator-after-endpoint-rotation",
            ),
            &coordinator,
        )
        .unwrap();
        let stale_round = publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
            &coordinator_peers,
            &stale_coordinator,
            &client,
            &[witness.public_key_bytes()],
            &allow_test_endpoint,
        )
        .await;
        assert_eq!(stale_round.attempted, 1);
        assert_eq!(stale_round.accepted, 0);
        assert_eq!(stale_round.failed, 1);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn descriptor_preflight_rejects_unsafe_witness_endpoint_without_request() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &witness,
            Some("http://127.0.0.1:8422".to_string()),
            now,
        );
        let current_coordinator = SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                coordinator.public_key_bytes(),
                1,
                now,
                now.saturating_add(600),
                "unsafe-preflight-target-test",
            ),
            &coordinator,
        )
        .unwrap();
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();

        let round = publish_current_descriptor_to_commitment_witnesses(
            &coordinator_peers,
            &current_coordinator,
            &client,
            &[witness.public_key_bytes()],
        )
        .await;

        assert_eq!(
            round,
            CommitmentWitnessDescriptorPublishRound {
                configured: 1,
                attempted: 0,
                accepted: 0,
                failed: 1,
            }
        );
    }

    #[tokio::test]
    async fn pinned_witness_round_recovers_through_authentic_expired_cache_descriptor() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let pinned_witness = Arc::new(IdentityKeyPair::generate());
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        witness_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let witness_peers = Arc::new(PeerStore::new());
        let expired_coordinator = NodeDescriptor::new(
            coordinator.public_key_bytes(),
            1,
            now.saturating_sub(1_200),
            now.saturating_sub(600),
            "expired-pinned-coordinator-test",
        );
        let expired_coordinator =
            SignedNodeDescriptor::sign(expired_coordinator, &coordinator).unwrap();
        let imported = witness_peers.load_peer_cache_snapshot_from_source(
            &NodeBootstrapSnapshot::new(now, vec![expired_coordinator]),
            now,
            "test_expired_coordinator_cache",
        );
        assert_eq!(imported.inserted, 1);
        assert!(witness_peers
            .get_valid(&coordinator.public_key_bytes(), now)
            .is_none());
        let router = build_memchain_peer_router_with_coordinator_lease(
            Arc::clone(&witness_storage),
            witness_peers,
            Arc::clone(&pinned_witness),
            Some(coordinator.public_key_bytes()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let mut expired = NodeDescriptor::new(
            pinned_witness.public_key_bytes(),
            1,
            now.saturating_sub(1_200),
            now.saturating_sub(600),
            "expired-pinned-witness-test",
        );
        expired.public_endpoint = Some(format!("http://{address}"));
        expired.capabilities = vec![NodeCapability::EncryptedStorage];
        let expired = SignedNodeDescriptor::sign(expired, &pinned_witness).unwrap();
        let coordinator_peers = PeerStore::new();
        let imported = coordinator_peers.load_peer_cache_snapshot_from_source(
            &NodeBootstrapSnapshot::new(now, vec![expired]),
            now,
            "test_expired_cache",
        );
        assert_eq!(imported.inserted, 1);
        assert!(coordinator_peers
            .get_valid(&pinned_witness.public_key_bytes(), now)
            .is_none());
        assert!(commitment_peer_descriptor(
            &coordinator_peers,
            &pinned_witness.public_key_bytes(),
            now,
            CommitmentPeerDescriptorPolicy::CurrentOnly,
        )
        .is_none());
        assert!(commitment_peer_descriptor(
            &coordinator_peers,
            &pinned_witness.public_key_bytes(),
            now,
            CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        )
        .is_some());

        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            &[pinned_witness.public_key_bytes()],
            1,
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 1);
        assert_eq!(round.converged, 1);
        assert_eq!(round.failed, 0);
        assert_eq!(
            local.record_commitment_checkpoint_status().evidence_records,
            1
        );

        let instance_id = [0x6a; 32];
        let lease = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &pinned_witness.public_key_bytes(),
            &instance_id,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert!(lease.lease_epoch > 0);
        assert!(lease.valid_for_secs > 0);

        let released = release_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_peers,
            &coordinator,
            &pinned_witness.public_key_bytes(),
            &instance_id,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(released.lease_epoch, lease.lease_epoch);
        assert!(released.released_at >= now);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn pinned_witness_round_persists_two_signer_checkpoint_certificate() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witnesses = [
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(IdentityKeyPair::generate()),
        ];
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_storages = [
            Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
            Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
        ];
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x91; 32]],
            &coordinator,
        );
        local
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();

        let mut addresses = Vec::new();
        let mut servers = Vec::new();
        for (storage, witness) in witness_storages.iter().zip(witnesses.iter()) {
            storage
                .append_record_commitment_block(&block, None)
                .await
                .unwrap();
            storage.audit_record_commitment_chain().await.unwrap();
            let peers = Arc::new(PeerStore::new());
            admit_peer(&peers, &coordinator, None, now);
            let router =
                build_memchain_peer_router(Arc::clone(storage), peers, Arc::clone(witness));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            addresses.push(listener.local_addr().unwrap());
            servers.push(tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap();
            }));
        }

        let coordinator_peers = PeerStore::new();
        for (witness, address) in witnesses.iter().zip(addresses.iter()) {
            admit_peer(
                &coordinator_peers,
                witness,
                Some(format!("http://{address}")),
                now,
            );
        }
        let witness_ids = [
            witnesses[0].public_key_bytes(),
            witnesses[1].public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            &witness_ids,
            2,
            |_| true,
        )
        .await;

        assert_eq!(round.verified, 2);
        assert_eq!(round.converged, 2);
        assert_eq!(round.certificate_signers, 2);
        assert_eq!(round.certificate_required_signers, 2);
        assert!(round.certificate_persisted);
        assert!(!round.certificate_persistence_failed);
        let status = local.record_commitment_checkpoint_status();
        assert_eq!(status.checkpoint_certificates, 1);
        assert_eq!(status.latest_certified_height, Some(1));
        assert_eq!(status.latest_certificate_signers, 2);

        let destination_identity = IdentityKeyPair::generate();
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        destination.configure_record_commitment_sync(false, true);
        destination.configure_record_commitment_certificate_policy(0, 1);

        let source_identity = Arc::new(coordinator);
        let source_peers = Arc::new(PeerStore::new());
        admit_peer(&source_peers, &destination_identity, None, now);
        let source_router = build_memchain_peer_router(
            Arc::clone(&local),
            source_peers,
            Arc::clone(&source_identity),
        );
        let source_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let source_address = source_listener.local_addr().unwrap();
        let source_server = tokio::spawn(async move {
            axum::serve(source_listener, source_router).await.unwrap();
        });

        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            &source_identity,
            Some(format!("http://{source_address}")),
            now,
        );
        let disabled = sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &witness_ids,
            1,
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(
            disabled,
            CommitmentFollowerCertificateSyncOutcome::PolicyDisabled
        );
        let disabled_status = destination.record_commitment_sync_status();
        assert_eq!(disabled_status.certificate_policy_state, "disabled");
        assert!(!disabled_status.certificate_policy_ready);
        assert!(disabled_status
            .certificate_policy_last_evaluated_at
            .is_some());

        destination.configure_record_commitment_certificate_policy(2, 2);
        let imported = sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &witness_ids,
            2,
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        let CommitmentFollowerCertificateSyncOutcome::Refreshed(imported) = imported else {
            panic!("uncertified converged follower must import current certificate");
        };
        assert_eq!(imported.checkpoint_height, 1);
        assert_eq!(imported.signer_count, 2);
        assert_eq!(imported.required_signers, 2);
        assert!(imported.persisted);
        assert_eq!(
            destination
                .record_commitment_checkpoint_status()
                .checkpoint_certificates,
            1
        );
        let imported_status = destination.record_commitment_sync_status();
        assert_eq!(imported_status.certificate_policy_state, "ready");
        assert!(imported_status.certificate_policy_ready);
        assert_eq!(imported_status.certificate_witnesses_configured, 2);
        assert_eq!(imported_status.certificate_minimum_signers, 2);
        assert_eq!(imported_status.certificate_sync_rounds_total, 1);
        assert_eq!(imported_status.certificate_coordinator_success_total, 1);
        assert_eq!(
            imported_status.certificate_verified_unpersisted_total,
            0
        );
        assert_eq!(
            imported_status.certificate_policy_evaluated_tip_height,
            Some(1)
        );
        assert!(imported_status
            .certificate_policy_last_evaluated_at
            .is_some());

        let replacement_witness = IdentityKeyPair::generate().public_key_bytes();
        destination.configure_record_commitment_certificate_policy(2, 2);
        let rotated_policy_error =
            sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
                &destination,
                &destination_peers,
                &destination_identity,
                &source_identity.public_key_bytes(),
                &[witness_ids[0], replacement_witness],
                2,
                1,
                &client,
                &allow_test_endpoint,
            )
            .await
            .unwrap_err();
        assert_eq!(
            rotated_policy_error, "certificate_member_not_pinned",
            "a same-height certificate under retired pins must not be current"
        );
        let rotated_status = destination.record_commitment_sync_status();
        assert_eq!(rotated_status.certificate_policy_state, "security_stopped");
        assert!(!rotated_status.certificate_policy_ready);

        let third_witness = IdentityKeyPair::generate().public_key_bytes();
        let strict_witness_ids = [witness_ids[0], witness_ids[1], third_witness];
        let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &strict_witness_ids,
            3,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "certificate_threshold_below_policy");

        let unpinned_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        unpinned_destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        unpinned_destination
            .audit_record_commitment_chain()
            .await
            .unwrap();
        unpinned_destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &unpinned_destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &[witness_ids[0], replacement_witness],
            2,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "certificate_member_not_pinned");

        for server in servers {
            server.abort();
            let _ = server.await;
        }
        source_server.abort();
        let _ = source_server.await;

        // [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex] A witness serves
        // only as a read-only carrier for an already audited certificate. The
        // receiver still validates the embedded coordinator checkpoint,
        // distinct witness frames, local pins, threshold, and exact local tip.
        let carrier_destination_identity = IdentityKeyPair::generate();
        let guarded_destination_identity = IdentityKeyPair::generate();
        let carrier_identity = Arc::clone(&witnesses[0]);
        let carrier_peers = Arc::new(PeerStore::new());
        admit_peer(&carrier_peers, &carrier_destination_identity, None, now);
        admit_peer(&carrier_peers, &guarded_destination_identity, None, now);
        assert!(carrier_peers
            .get_valid(&carrier_destination_identity.public_key_bytes(), now_secs())
            .is_some());
        let carrier_router = build_memchain_peer_router(
            Arc::clone(&local),
            Arc::clone(&carrier_peers),
            Arc::clone(&carrier_identity),
        );
        let carrier_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let carrier_address = carrier_listener.local_addr().unwrap();
        let carrier_server = tokio::spawn(async move {
            axum::serve(carrier_listener, carrier_router).await.unwrap();
        });

        let carrier_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        carrier_destination
            .audit_record_commitment_chain()
            .await
            .unwrap();
        carrier_destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        carrier_destination.configure_record_commitment_sync(false, true);
        carrier_destination.configure_record_commitment_certificate_policy(2, 2);
        let carrier_destination_peers = PeerStore::new();
        let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let unavailable_address = unavailable_listener.local_addr().unwrap();
        drop(unavailable_listener);
        admit_peer(
            &carrier_destination_peers,
            &source_identity,
            Some(format!("http://{unavailable_address}")),
            now,
        );
        admit_peer(
            &carrier_destination_peers,
            &carrier_identity,
            Some(format!("http://{carrier_address}")),
            now,
        );
        // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] The coordinator is
        // unreachable and the destination starts at genesis. The pinned
        // witness signs only the page envelope; the imported block must retain
        // the unavailable coordinator as proposer.
        let recovered_page =
            pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
                &carrier_destination,
                &carrier_destination_peers,
                &carrier_destination_identity,
                &source_identity.public_key_bytes(),
                &witness_ids,
                2,
                &client,
                &allow_test_endpoint,
            )
            .await
            .unwrap();
        assert_eq!(
            recovered_page.source,
            CommitmentSyncPageSource::PinnedCarrier
        );
        assert_eq!(recovered_page.carrier_attempts, 1);
        assert_eq!(recovered_page.page.inserted, 1);
        assert!(!recovered_page.page.has_more);
        assert_eq!(recovered_page.page.remote_tip_height, 1);
        assert_eq!(
            carrier_destination.record_commitment_chain_tip().await,
            local.record_commitment_chain_tip().await
        );
        let block_carrier_status = carrier_destination.record_commitment_sync_status();
        assert_eq!(block_carrier_status.block_page_pulls_total, 1);
        assert_eq!(
            block_carrier_status.block_page_coordinator_success_total,
            0
        );
        assert_eq!(block_carrier_status.block_carrier_attempts_total, 1);
        assert_eq!(block_carrier_status.block_carrier_recoveries_total, 1);
        assert_eq!(
            block_carrier_status.block_page_availability_exhausted_total,
            0
        );
        assert_eq!(block_carrier_status.block_page_security_stops_total, 0);
        assert_eq!(
            block_carrier_status.last_block_page_pull_result.as_deref(),
            Some("carrier_recovered")
        );
        assert!(block_carrier_status
            .last_block_carrier_recovered_at
            .is_some());
        let recovered =
            sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
                &carrier_destination,
                &carrier_destination_peers,
                &carrier_destination_identity,
                &source_identity.public_key_bytes(),
                &witness_ids,
                2,
                1,
                &client,
                &allow_test_endpoint,
            )
            .await
            .unwrap();
        let CommitmentFollowerCertificateSyncOutcome::Refreshed(recovered) = recovered else {
            panic!("pinned witness carrier must recover coordinator certificate availability");
        };
        assert!(recovered.persisted);
        assert_eq!(recovered.checkpoint_height, 1);
        assert_eq!(recovered.signer_count, 2);
        let carrier_status = carrier_destination.record_commitment_sync_status();
        assert_eq!(carrier_status.certificate_sync_rounds_total, 1);
        assert_eq!(carrier_status.certificate_coordinator_success_total, 0);
        assert_eq!(carrier_status.certificate_carrier_attempts_total, 1);
        assert_eq!(carrier_status.certificate_carrier_recoveries_total, 1);
        assert_eq!(carrier_status.certificate_verified_unpersisted_total, 0);
        assert_eq!(carrier_status.certificate_availability_exhausted_total, 0);
        assert_eq!(carrier_status.certificate_security_stops_total, 0);
        assert_eq!(
            carrier_status.last_certificate_sync_result.as_deref(),
            Some("carrier_recovered")
        );
        assert!(carrier_status
            .last_certificate_carrier_recovered_at
            .is_some());
        assert_eq!(carrier_status.certificate_policy_state, "ready");
        assert!(carrier_status.certificate_policy_ready);
        assert_eq!(
            carrier_status.certificate_policy_evaluated_tip_height,
            Some(1)
        );

        // A malformed primary response is a security failure, not an
        // availability event. The valid carrier must not mask it.
        let malformed_router = Router::new().route(
            "/api/memchain/peer/checkpoint-certificate",
            post(|| async { (StatusCode::OK, vec![0u8]) }),
        );
        let malformed_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let malformed_address = malformed_listener.local_addr().unwrap();
        let malformed_server = tokio::spawn(async move {
            axum::serve(malformed_listener, malformed_router)
                .await
                .unwrap();
        });
        let guarded_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        guarded_destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        guarded_destination
            .audit_record_commitment_chain()
            .await
            .unwrap();
        guarded_destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        guarded_destination.configure_record_commitment_sync(false, true);
        guarded_destination.configure_record_commitment_certificate_policy(2, 2);
        let guarded_destination_peers = PeerStore::new();
        admit_peer(
            &guarded_destination_peers,
            &source_identity,
            Some(format!("http://{malformed_address}")),
            now,
        );
        admit_peer(
            &guarded_destination_peers,
            &carrier_identity,
            Some(format!("http://{carrier_address}")),
            now,
        );
        let guarded_error =
            sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
                &guarded_destination,
                &guarded_destination_peers,
                &guarded_destination_identity,
                &source_identity.public_key_bytes(),
                &witness_ids,
                2,
                1,
                &client,
                &allow_test_endpoint,
            )
            .await
            .unwrap_err();
        assert_eq!(guarded_error, "invalid_certificate_frame");
        let guarded_status = guarded_destination.record_commitment_sync_status();
        assert_eq!(guarded_status.certificate_sync_rounds_total, 1);
        assert_eq!(guarded_status.certificate_carrier_attempts_total, 0);
        assert_eq!(guarded_status.certificate_carrier_recoveries_total, 0);
        assert_eq!(guarded_status.certificate_verified_unpersisted_total, 0);
        assert_eq!(guarded_status.certificate_security_stops_total, 1);
        assert_eq!(
            guarded_status.last_certificate_sync_result.as_deref(),
            Some("security_stopped")
        );
        assert_eq!(guarded_status.certificate_policy_state, "security_stopped");
        assert!(!guarded_status.certificate_policy_ready);

        malformed_server.abort();
        let _ = malformed_server.await;
        carrier_server.abort();
        let _ = carrier_server.await;

        destination.configure_record_commitment_certificate_policy(2, 2);
        let already_current =
            sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
                &destination,
                &destination_peers,
                &destination_identity,
                &source_identity.public_key_bytes(),
                &witness_ids,
                2,
                1,
                &client,
                &allow_test_endpoint,
            )
            .await
            .unwrap();
        assert_eq!(
            already_current,
            CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent
        );
        let already_current_status = destination.record_commitment_sync_status();
        assert_eq!(already_current_status.certificate_policy_state, "ready");
        assert!(already_current_status.certificate_policy_ready);
        assert_eq!(
            already_current_status.certificate_policy_evaluated_tip_height,
            Some(1)
        );
    }
}
