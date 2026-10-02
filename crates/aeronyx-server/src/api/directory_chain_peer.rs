// ============================================
// File: crates/aeronyx-server/src/api/directory_chain_peer.rs
// ============================================
//! # Directory Chain Peer API
//!
//! ## Creation Reason
//! A durable local Directory Chain cannot become independently verifiable by
//! other nodes until it has a narrow authenticated transport. Discovery gossip
//! is permissionless and optimized for current descriptors, so it must not be
//! reused as an unbounded historical ledger endpoint.
//!
//! ## Main Functionality
//! - `POST /api/discovery/peer/directory/tip`
//! - `POST /api/discovery/peer/directory/block-range`
//! - `POST /api/discovery/peer/directory/descriptor-objects`
//! - `POST /api/discovery/peer/directory/descriptor-inclusion-proof`
//! - `POST /api/discovery/peer/directory/replica-block-range`
//! - `POST /api/discovery/peer/directory/replica-descriptor-objects`
//! - `POST /api/discovery/peer/directory/replica-descriptor-inclusion-proof`
//! - `POST /api/discovery/peer/directory/observation-checkpoint-witness`
//! - `POST /api/discovery/peer/directory/observation-checkpoint-witness-carrier`
//! - `POST /api/discovery/peer/directory/observation-policy-anchor`
//! - `POST /api/discovery/peer/directory/observation-certificate`
//! - Tiered admission: verified public peers may mirror this node's own signed
//!   producer history and recover retained non-authoritative mirror evidence;
//!   witness and policy-anchor routes remain restricted to operator-pinned peers.
//! - Ed25519 request/response authentication, timestamp freshness, replay
//!   rejection, per-peer rate limits, body limits, and audit-gated reads.
//! - Exact content-addressed descriptor batches; no partial object response.
//! - Pinned-peer-only, exact-block descriptor inclusion proofs for light
//!   verifiers that already trust a producer and selected block hash.
//! - Independent checkpoint root recomputation before a signed witness decision.
//! - Monotonic opaque policy-head retention before a signed anchor decision.
//! - Pinned-peer-only export of the latest locally verified portable
//!   observation certificate with exact frame digest binding.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Pinned-authority-only, one-hop
//!   transport of an exact observer-signed witness request to an exact pinned
//!   witness. The carrier verifies both inner frames but never signs evidence.
//! - [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Process-only mutually
//!   exclusive carrier outcomes with no identity, route, or frame retention.
//! - [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Fail-fast outbound
//!   concurrency admission and descriptor-sequence-bound target cooldowns.
//! - Audited producer-replica export with a separate carrier signature layer.
//! - [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] Exact original
//!   producer-signed descriptor proofs through audited carriers, without
//!   granting the carrier producer or chain-selection authority.
//! - Explicit lagging-carrier responses that let a requester continue to the
//!   next verified carrier without retrying malformed protocol requests.
//! - Multi-block catch-up pages capped to one block's maximum aggregate
//!   commitment budget and stopped before repeated descriptor objects.
//! - [DIRECTORY-BLOCKING-BOUNDARY 2026-07-30 by Codex] One fail-closed
//!   blocking-worker boundary preserves the existing protocol response while
//!   recording only a static operation role and fixed join-failure category.
//! - [DIRECTORY-AUDIT-ADMISSION 2026-08-12 by Codex] Expensive authenticated
//!   chain audits share one runtime-owned fail-fast permit across public and
//!   local listeners. Concurrent peers receive a retryable fixed bucket instead
//!   of exhausting Tokio's blocking pool.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` only when `DirectoryChainStore` is configured.
//! - Uses protocol contracts from `aeronyx-core/src/protocol/discovery.rs`.
//! - Reads only through `DirectoryChainStore::audited_*` methods.
//! - Uses `PeerStore::get_valid` as a second, live descriptor admission gate.
//!
//! ## Main Logical Flow
//! 1. Axum rejects an oversized body before protocol deserialization.
//! 2. The handler decodes one canonical Directory Sync V1 frame.
//! 3. Chain id, request bounds, route-specific admission, live peer descriptor,
//!    timestamp, signature, replay id, and rate budgets are verified.
//! 4. A blocking worker performs the complete local chain audit and bounded read.
//! 5. The local producer signs a response binding request id, ordered hashes,
//!    returned block identities, and the audited tip.
//! 6. A witness response is accepted only after the local replica store
//!    independently reproduces every exact prefix and observation root.
//! 7. Carrier routes audit one producer namespace before reading it. Public
//!    recovery can export configured producer history or a durable mirror
//!    namespace. The carrier signs transport but never producer history.
//! 8. Policy-anchor requests disclose only an observer epoch and opaque digest;
//!    rollback, same-epoch conflict, and non-contiguous progression fail closed.
//! 9. Certificate export re-audits the latest local certificate against the
//!    current witness policy and binds the exact frame into a signed response.
//! 10. Witness carrier requests bind one exact inner frame and target. The
//!     carrier resolves only a current signed public-IP endpoint, disables HTTP
//!     proxies and redirects, forwards once, verifies the witness signature,
//!     and returns a carrier-signed transport envelope.
//! 11. Inclusion-proof requests bind one exact descriptor and independently
//!     selected block hash; the server never substitutes another block.
//! 12. Replica proof recovery audits one producer namespace transactionally,
//!     preserves the original producer proof, then adds a distinct carrier
//!     transport signature for independent verification.
//!
//! ## Privacy Invariant
//! This API serves signed public node-directory commitments and the public
//! signed descriptors they already bind. It never serves client identities,
//! IPs, routes, selected hops, message ids, packet/chat payloads, Memory Chain
//! records, DNS contents, destinations, private keys, or wallet traffic.
//!
//! ## Important Note for Next Developer
//! - Permissionless carrier admission applies only when public mirror reads are
//!   enabled and only to configured producers or namespaces in the durable
//!   mirror registry. Witness and policy-anchor routes always require a pin.
//! - Never return a descriptor not committed by the audited local chain.
//! - Keep request/response limits synchronized with `aeronyx-core` constants.
//! - Replica import, fork quarantine, and fork choice are separate layers. A
//!   valid response proves what one producer signed; it is not consensus.
//! - Never sign an accepted witness response from the observer signature alone.
//!   Missing local prefixes must remain unavailable, never trusted by fallback.
//! - Never export a non-configured replica unless it remains in the audited
//!   mirror registry. A carrier signature does not replace any producer block or
//!   descriptor signature and grants no authority.
//! - [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] Observation certificates expose
//!   public observer/witness identities needed for verification. Keep this
//!   endpoint POST-only and pinned-peer-only; do not mount a public GET alias.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Never allow recursive forwarding,
//!   caller-supplied URLs, redirects, an unpinned target, or a carrier-generated
//!   witness outcome. Availability transport must not expand authority.
//! - [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] Keep proof export
//!   pinned-peer-only. The proof establishes producer-signed inclusion, not
//!   canonical-chain selection, consensus, finality, or user activity.
//! - Replica proof recovery may follow verified-public mirror admission, but
//!   only for a producer still in the durable mirror registry. Carrier
//!   availability is never producer, checkpoint, witness, or policy authority.
//! - Never format a blocking worker's raw `JoinError`; Tokio panic payloads may
//!   contain filesystem, descriptor, or other internal process material.
//! - Keep the audit admission fail-fast. Queueing permissionless history audits
//!   would retain unbounded request futures and turn peer traffic into memory
//!   pressure even though the blocking worker count remains bounded.
//!
//! ## Last Modified
//! v0.17.1-AuditOwnership - Bound public and local listener admission to the
//! same shared `DirectoryReplicaSyncRuntime` resource.
//! v0.17.0-AuditAdmission - Added process-wide fail-fast admission for
//! expensive authenticated Directory audit workers.
//! v0.16.0-BlockingWorkerBoundary - Centralized all Directory peer blocking
//! worker joins behind privacy-safe, fail-closed diagnostics.
//! v0.15.0-ReplicaDescriptorInclusionProof - Added registry-gated carrier
//! recovery for exact original producer descriptor inclusion proofs.
//! v0.14.0-DirectoryInclusionProof - Added exact-block, audit-gated,
//! pinned-peer-only descriptor proof export.
//! v0.13.0-WitnessCarrierAdmission - Added bounded outbound concurrency,
//! descriptor-bound failure cooldowns, and deterministic recovery coverage.
//! v0.12.1-WitnessCarrierResultMatrix - Reused one bounded transport client and
//! added deterministic handler-level coverage for every terminal outcome.
//! v0.12.0-WitnessCarrierServiceTelemetry - Added shared privacy-safe carrier
//! runtime observations for public and local Directory status.
//! v0.11.0-BoundedWitnessCarrier - Added pinned, single-hop, exact-frame
//! checkpoint-witness transport with independent inner-frame verification.
//! v0.10.0-AuthenticatedCertificateExchange - Added pinned-peer-only portable
//! observation-certificate export with exact frame digest binding.
//! v0.9.1-MirrorCarrierRangeAvailability - Distinguished valid unavailable
//! replica ranges from malformed requests for bounded carrier failover.
//! v0.9.0-MirrorRecovery - Added bounded verified-public recovery for audited mirror namespaces.
//! v0.8.0-FullNodeMirror - Added verified-public read admission with pinned authority routes.
//! v0.7.0-DirectoryPolicyHeadAnchor - Added durable opaque policy-head anchor route.
//! v0.6.1-DirectoryBoundedMultiBlockCatchUp - Added commitment-bounded multi-block pages.
//! v0.6.0-DirectoryEvidenceCarrier - Added audited pinned replica-carrier routes.
//! v0.5.0-DirectoryObservationWitness - Added independently recomputed pinned-peer witness route.
//! v0.4.0-DirectoryReplicaModuleSplit - Moved status, outbound transport, and
//! scheduling into dedicated modules without changing Directory Sync V1.
//! v0.3.0-DirectoryReplicaStatus - Added privacy-tiered status and bounded
//! multi-page request-budget primitives.
//! v0.2.0-DirectorySyncPull - Added verified bounded replica page download.
//! v0.1.0-DirectorySyncServing - Initial authenticated bounded peer transport.
// ============================================

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_request_signing_bytes,
    directory_block_range_response_signing_bytes,
    directory_descriptor_inclusion_proof_request_signing_bytes,
    directory_descriptor_inclusion_proof_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes,
    directory_observation_certificate_request_signing_bytes,
    directory_observation_certificate_response_signing_bytes,
    directory_observation_witness_carrier_request_signing_bytes,
    directory_observation_witness_carrier_response_signing_bytes,
    directory_observation_witness_request_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_request_signing_bytes,
    directory_replica_block_range_response_signing_bytes,
    directory_replica_descriptor_inclusion_proof_request_signing_bytes,
    directory_replica_descriptor_inclusion_proof_response_signing_bytes,
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes,
    directory_tip_request_signing_bytes, directory_tip_response_signing_bytes,
    encode_directory_observation_certificate, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectoryDescriptorInclusionProofV1,
    DirectoryObservationCheckpointV1, DirectorySyncMessage, SignedNodeDescriptor,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES, MAX_DIRECTORY_SYNC_BLOCKS_V1,
    MAX_DIRECTORY_SYNC_OBJECTS_V1,
};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
use crate::error::RuntimeTaskJoinFailureKind;
use crate::services::directory_replica::{
    DirectoryObservationWitnessCarrierOutcome, DirectoryObservationWitnessPolicyAnchorDecision,
    DirectoryReplicaEvidencePage, DirectoryReplicaSyncRuntime,
};
use crate::services::{
    DirectoryChainStore, DirectoryChainStoreError, DirectoryObservationWitnessDecision,
    DirectoryReplicaStore, DirectoryReplicaStoreError, PeerStore,
};

/// A request contains at most sixteen hashes plus fixed signatures and fields.
const MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES: usize = 16 * 1024;
/// A carried witness response is always smaller than the shared request bound.
const MAX_WITNESS_CARRIER_RESPONSE_BODY_BYTES: usize = MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES;
/// One carrier may make only one direct target request under this deadline.
const WITNESS_CARRIER_REQUEST_TIMEOUT_SECS: u64 = 10;
/// Carrier-side target requests are rejected instead of queued past this bound.
const MAX_WITNESS_CARRIER_REQUESTS_IN_FLIGHT: usize = 8;
/// Process-only cooldown state cannot grow beyond the operator's practical pin set.
const MAX_WITNESS_CARRIER_COOLDOWN_TARGETS: usize = 256;
/// Availability failures receive a short retry pause.
const WITNESS_CARRIER_AVAILABILITY_COOLDOWN_SECS: u64 = 30;
/// Invalid successful responses receive a longer safety pause.
const WITNESS_CARRIER_INVALID_RESPONSE_COOLDOWN_SECS: u64 = 60;
/// Missing target capability is descriptor-bound and changes less frequently.
const WITNESS_CARRIER_CAPABILITY_COOLDOWN_SECS: u64 = 300;
/// Shared inbound budget for each pinned peer identity.
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
/// Global budget bounds aggregate pressure from permissionless verified peers.
const MAX_DIRECTORY_REQUESTS_GLOBAL_PER_MINUTE: u32 = 512;
/// Accepted signed request clock skew in either direction.
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
/// Stateful request ids remain rejected for this duration.
const REPLAY_RETENTION_SECS: u64 = 120;

/// Complete bounded result of one witness-target request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WitnessCarrierTransportResponse {
    status: u16,
    body: Vec<u8>,
}

/// Transport failures are intentionally coarser than the underlying HTTP error.
///
/// This boundary prevents endpoint, route, and lower-level connection details
/// from entering carrier telemetry or protocol responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WitnessCarrierTransportError {
    LocalUnavailable,
    TargetUnavailable,
    ResponseTooLarge,
}

/// One-hop witness transport isolated from authority and frame verification.
#[async_trait::async_trait]
trait WitnessCarrierTransport: Send + Sync {
    async fn send(
        &self,
        url: reqwest::Url,
        request_frame: Vec<u8>,
    ) -> Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError>;
}

/// Production no-proxy transport shared by all carrier requests in one router.
struct ReqwestWitnessCarrierTransport {
    client: Result<reqwest::Client, ()>,
}

impl ReqwestWitnessCarrierTransport {
    fn new() -> Self {
        let client = reqwest::Client::builder()
            // [WITNESS-CARRIER-MATRIX 2026-07-27 by Codex] Build once per
            // router while preserving the existing SSRF and timeout boundary.
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(WITNESS_CARRIER_REQUEST_TIMEOUT_SECS))
            .timeout(Duration::from_secs(WITNESS_CARRIER_REQUEST_TIMEOUT_SECS))
            .build()
            .map_err(|_| ());
        Self { client }
    }
}

#[async_trait::async_trait]
impl WitnessCarrierTransport for ReqwestWitnessCarrierTransport {
    async fn send(
        &self,
        url: reqwest::Url,
        request_frame: Vec<u8>,
    ) -> Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError> {
        let client = self
            .client
            .as_ref()
            .map_err(|_| WitnessCarrierTransportError::LocalUnavailable)?;
        let response = client
            .post(url)
            .header("content-type", "application/octet-stream")
            .body(request_frame)
            .send()
            .await
            .map_err(|_| WitnessCarrierTransportError::TargetUnavailable)?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Ok(WitnessCarrierTransportResponse {
                status,
                body: Vec::new(),
            });
        }
        if response.content_length().is_some_and(|length| {
            length > u64::try_from(MAX_WITNESS_CARRIER_RESPONSE_BODY_BYTES).unwrap_or(u64::MAX)
        }) {
            return Err(WitnessCarrierTransportError::ResponseTooLarge);
        }
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| WitnessCarrierTransportError::TargetUnavailable)?;
            if body.len().saturating_add(chunk.len()) > MAX_WITNESS_CARRIER_RESPONSE_BODY_BYTES {
                return Err(WitnessCarrierTransportError::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        Ok(WitnessCarrierTransportResponse { status, body })
    }
}

/// Process-only availability guard for witness-carrier outbound work.
///
/// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Target identities are used
/// only as private map keys and are never emitted through status or logs. A
/// cooldown is bound to the signed descriptor sequence, so a fresh descriptor
/// immediately receives a new availability attempt.
struct WitnessCarrierRuntime {
    transport: Arc<dyn WitnessCarrierTransport>,
    in_flight: Arc<Semaphore>,
    target_cooldowns: Mutex<HashMap<[u8; 32], WitnessCarrierTargetCooldown>>,
}

#[derive(Debug, Clone, Copy)]
struct WitnessCarrierTargetCooldown {
    descriptor_sequence: u64,
    retry_at: u64,
}

impl WitnessCarrierRuntime {
    fn new(transport: Arc<dyn WitnessCarrierTransport>, max_in_flight: usize) -> Self {
        Self {
            transport,
            in_flight: Arc::new(Semaphore::new(max_in_flight.max(1))),
            target_cooldowns: Mutex::new(HashMap::new()),
        }
    }

    async fn target_is_cooling_down(
        &self,
        target: &[u8; 32],
        descriptor_sequence: u64,
        observed_at: u64,
    ) -> bool {
        let mut cooldowns = self.target_cooldowns.lock().await;
        cooldowns.retain(|_, cooldown| cooldown.retry_at > observed_at);
        cooldowns.get(target).is_some_and(|cooldown| {
            cooldown.descriptor_sequence == descriptor_sequence && cooldown.retry_at > observed_at
        })
    }

    async fn cool_down_target(
        &self,
        target: [u8; 32],
        descriptor_sequence: u64,
        observed_at: u64,
        duration_secs: u64,
    ) {
        let mut cooldowns = self.target_cooldowns.lock().await;
        cooldowns.retain(|_, cooldown| cooldown.retry_at > observed_at);
        if !cooldowns.contains_key(&target)
            && cooldowns.len() >= MAX_WITNESS_CARRIER_COOLDOWN_TARGETS
        {
            return;
        }
        cooldowns.insert(
            target,
            WitnessCarrierTargetCooldown {
                descriptor_sequence,
                retry_at: observed_at.saturating_add(duration_secs),
            },
        );
    }

    async fn clear_target_cooldown(&self, target: &[u8; 32]) {
        self.target_cooldowns.lock().await.remove(target);
    }
}

#[derive(Clone)]
struct DirectoryChainPeerState {
    store: Arc<DirectoryChainStore>,
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peers: Arc<HashSet<[u8; 32]>>,
    allow_public_mirror_reads: bool,
    guard: Arc<Mutex<DirectoryPeerRequestGuard>>,
    audit_admission: Arc<Semaphore>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    witness_carrier: Arc<WitnessCarrierRuntime>,
}

#[derive(Debug, Default)]
struct DirectoryPeerRequestGuard {
    global_window: PeerRateWindow,
    rate_windows: HashMap<[u8; 32], PeerRateWindow>,
    seen_requests: HashMap<([u8; 32], [u8; 16]), u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PeerRateWindow {
    minute: u64,
    used: u32,
}

impl DirectoryPeerRequestGuard {
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
        let minute = now / 60;
        self.rate_windows
            .retain(|_, window| window.minute >= minute.saturating_sub(1));
        if self.global_window.minute != minute {
            self.global_window = PeerRateWindow { minute, used: 0 };
        }
        if self.global_window.used >= MAX_DIRECTORY_REQUESTS_GLOBAL_PER_MINUTE {
            return false;
        }
        let window = self.rate_windows.entry(requester).or_default();
        if window.minute != minute {
            *window = PeerRateWindow { minute, used: 0 };
        }
        if window.used >= MAX_REQUESTS_PER_PEER_PER_MINUTE {
            return false;
        }
        self.global_window.used = self.global_window.used.saturating_add(1);
        window.used += 1;
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }
        self.seen_requests.insert((requester, request_id), now);
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryPeerAdmission {
    /// Read-only mirroring of this node's own public signed producer history.
    VerifiedPublicMirror,
    /// Read-only recovery of another producer's retained public signed history.
    VerifiedPublicRecovery,
    /// Authority-sensitive carrier, witness, and policy-anchor operations.
    PinnedAuthority,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod observation_exchange;
mod peer_admission;
mod producer_history;
mod replica_pages;

use observation_exchange::independently_evaluate_checkpoint;
use observation_exchange::observation_certificate_handler;
use observation_exchange::observation_checkpoint_witness_carrier_handler;
use observation_exchange::observation_checkpoint_witness_handler;
use observation_exchange::observation_policy_anchor_handler;
use observation_exchange::verify_carried_observation_witness_request;
use observation_exchange::verify_carried_observation_witness_response;
use observation_exchange::witness_carrier_outcome_response;
use peer_admission::authenticate_request;
use peer_admission::bounded_directory_transport_blocks;
use peer_admission::decode_request;
use peer_admission::encoded_response;
use peer_admission::now_secs;
use peer_admission::protocol_error;
use peer_admission::replica_store_error_response;
use peer_admission::run_directory_chain_blocking;
use peer_admission::run_directory_chain_blocking_with_admission;
use peer_admission::store_error_response;
use producer_history::block_range_handler;
use producer_history::descriptor_inclusion_proof_handler;
use producer_history::descriptor_objects_handler;
use producer_history::tip_handler;
use replica_pages::audited_replica_descriptor_proof_for_request;
use replica_pages::audited_replica_objects_for_request;
use replica_pages::audited_replica_page_for_request;
use replica_pages::replica_block_range_handler;
use replica_pages::replica_descriptor_inclusion_proof_handler;
use replica_pages::replica_descriptor_objects_handler;

/// Builds the fail-closed Directory Chain peer router.
#[must_use]
pub fn build_directory_chain_peer_router(
    store: Arc<DirectoryChainStore>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peer_ids: Vec<[u8; 32]>,
) -> Router {
    build_directory_chain_peer_router_with_replica(
        store,
        None,
        peer_store,
        identity,
        pinned_peer_ids,
        false,
    )
}

/// Builds the peer router with independent observation-checkpoint witnessing.
///
/// Passing `None` preserves the pre-witness route surface. The witness route is
/// mounted only when a startup-audited producer-isolated replica store exists.
pub fn build_directory_chain_peer_router_with_replica(
    store: Arc<DirectoryChainStore>,
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peer_ids: Vec<[u8; 32]>,
    allow_public_mirror_reads: bool,
) -> Router {
    build_directory_chain_peer_router_with_replica_and_runtime(
        store,
        replica_store,
        peer_store,
        identity,
        pinned_peer_ids,
        allow_public_mirror_reads,
        Arc::new(DirectoryReplicaSyncRuntime::default()),
    )
}

/// Builds the peer router with the synchronization runtime shared by status.
///
/// Existing builders allocate an isolated default runtime for compatibility.
/// Production listeners must use this function so carrier-side outcomes and
/// observer-side scheduling remain visible through one process snapshot.
pub fn build_directory_chain_peer_router_with_replica_and_runtime(
    store: Arc<DirectoryChainStore>,
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peer_ids: Vec<[u8; 32]>,
    allow_public_mirror_reads: bool,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
) -> Router {
    build_directory_chain_peer_router_with_replica_runtime_and_transport(
        store,
        replica_store,
        peer_store,
        identity,
        pinned_peer_ids,
        allow_public_mirror_reads,
        runtime,
        Arc::new(WitnessCarrierRuntime::new(
            Arc::new(ReqwestWitnessCarrierTransport::new()),
            MAX_WITNESS_CARRIER_REQUESTS_IN_FLIGHT,
        )),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_directory_chain_peer_router_with_replica_runtime_and_transport(
    store: Arc<DirectoryChainStore>,
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peer_ids: Vec<[u8; 32]>,
    allow_public_mirror_reads: bool,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    witness_carrier: Arc<WitnessCarrierRuntime>,
) -> Router {
    let audit_admission = runtime.directory_audit_admission();
    let state = DirectoryChainPeerState {
        store,
        replica_store,
        peer_store,
        identity,
        pinned_peers: Arc::new(pinned_peer_ids.into_iter().collect()),
        allow_public_mirror_reads,
        guard: Arc::new(Mutex::new(DirectoryPeerRequestGuard::default())),
        audit_admission,
        runtime,
        witness_carrier,
    };
    let mut router = Router::new()
        .route("/api/discovery/peer/directory/tip", post(tip_handler))
        .route(
            "/api/discovery/peer/directory/block-range",
            post(block_range_handler),
        )
        .route(
            "/api/discovery/peer/directory/descriptor-objects",
            post(descriptor_objects_handler),
        )
        .route(
            "/api/discovery/peer/directory/descriptor-inclusion-proof",
            post(descriptor_inclusion_proof_handler),
        );
    if state.replica_store.is_some() {
        router = router
            .route(
                "/api/discovery/peer/directory/replica-block-range",
                post(replica_block_range_handler),
            )
            .route(
                "/api/discovery/peer/directory/replica-descriptor-objects",
                post(replica_descriptor_objects_handler),
            )
            .route(
                "/api/discovery/peer/directory/replica-descriptor-inclusion-proof",
                post(replica_descriptor_inclusion_proof_handler),
            )
            .route(
                "/api/discovery/peer/directory/observation-checkpoint-witness",
                post(observation_checkpoint_witness_handler),
            )
            .route(
                "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                post(observation_checkpoint_witness_carrier_handler),
            )
            .route(
                "/api/discovery/peer/directory/observation-policy-anchor",
                post(observation_policy_anchor_handler),
            )
            .route(
                "/api/discovery/peer/directory/observation-certificate",
                post(observation_certificate_handler),
            );
    }
    router
        .layer(DefaultBodyLimit::max(MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES))
        .with_state(state)
}

#[derive(Debug, Clone, Copy)]
struct CarriedObservationWitnessRequestContext {
    request_id: [u8; 16],
    requester: [u8; 32],
    request_timestamp: u64,
    checkpoint_sequence: u64,
    checkpoint_hash: [u8; 32],
}

#[cfg(test)]
mod tests {
    mod history;
    mod observation;
    mod other;

    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tokio::sync::{Barrier, Notify};
    use tower::ServiceExt;

    use crate::api::directory_replica_sync::{
        verify_block_range_response, verify_descriptor_objects_response,
        verify_observation_certificate_response, verify_replica_block_range_response,
        verify_replica_descriptor_objects_response,
    };
    use aeronyx_core::protocol::discovery::{
        decode_directory_observation_certificate, directory_tip_response_signing_bytes,
        DirectoryCommitmentBlockV1, DirectoryDescriptorCommitmentV1,
        DirectoryObservationCheckpointV1, DirectoryObservationTipV1, NodeDescriptor,
        SignedNodeDescriptor,
    };
    use tempfile::TempDir;

    fn signed_descriptor(identity: &IdentityKeyPair, now: u64) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                identity.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "directory-sync-test",
            ),
            identity,
        )
        .unwrap()
    }

    fn test_router(
        pinned: bool,
        public_discovery: bool,
        allow_public_mirror_reads: bool,
    ) -> (
        Router,
        Arc<IdentityKeyPair>,
        IdentityKeyPair,
        SignedNodeDescriptor,
    ) {
        let now = now_secs();
        let producer = Arc::new(IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap());
        let requester = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let observed = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let observed_descriptor = signed_descriptor(&observed, now);
        let mut requester_node_descriptor = NodeDescriptor::new(
            requester.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 600,
            "directory-sync-test",
        );
        requester_node_descriptor.policy.public_discovery = public_discovery;
        let requester_descriptor =
            SignedNodeDescriptor::sign(requester_node_descriptor, &requester).unwrap();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(requester_descriptor, now, "directory_sync_test")
            .unwrap();
        let temp = TempDir::new().unwrap();
        let path = temp.keep().join("directory.db");
        let (store, _) = DirectoryChainStore::open(path, producer.public_key_bytes(), now).unwrap();
        store
            .append_descriptors(
                std::slice::from_ref(&observed_descriptor),
                now,
                producer.as_ref(),
            )
            .unwrap();
        let pins = pinned
            .then_some(requester.public_key_bytes())
            .into_iter()
            .collect();
        (
            build_directory_chain_peer_router_with_replica(
                Arc::new(store),
                None,
                peer_store,
                Arc::clone(&producer),
                pins,
                allow_public_mirror_reads,
            ),
            producer,
            requester,
            observed_descriptor,
        )
    }

    fn tip_request(requester: &IdentityKeyPair, request_id: [u8; 16]) -> Vec<u8> {
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let signing_bytes = directory_tip_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester_id,
            timestamp,
        );
        encode_directory_sync_message(&DirectorySyncMessage::TipRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn observation_certificate_request(
        requester: &IdentityKeyPair,
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let signing_bytes = directory_observation_certificate_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester_id,
            timestamp,
        );
        encode_directory_sync_message(&DirectorySyncMessage::ObservationCertificateRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn replica_range_request(
        requester: &IdentityKeyPair,
        producer: &[u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        replica_range_request_from_height(requester, producer, 1, request_id)
    }

    fn replica_range_request_from_height(
        requester: &IdentityKeyPair,
        producer: &[u8; 32],
        from_height: u64,
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let signing_bytes = directory_replica_block_range_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer,
            from_height,
            1,
            &request_id,
            &requester_id,
            timestamp,
        );
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: *producer,
            from_height,
            limit: 1,
            request_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn replica_descriptor_proof_request(
        requester: &IdentityKeyPair,
        producer: &[u8; 32],
        block_hash: &[u8; 32],
        descriptor_hash: &[u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let signing_bytes = directory_replica_descriptor_inclusion_proof_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer,
            block_hash,
            descriptor_hash,
            &request_id,
            &requester_id,
            timestamp,
        );
        encode_directory_sync_message(
            &DirectorySyncMessage::ReplicaDescriptorInclusionProofRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                producer: *producer,
                block_hash: *block_hash,
                descriptor_hash: *descriptor_hash,
                request_id,
                requester: requester_id,
                request_timestamp: timestamp,
                signature: requester.sign(&signing_bytes),
            },
        )
        .unwrap()
    }

    fn verify_replica_descriptor_proof_response(
        body: &[u8],
        expected_request_id: &[u8; 16],
        expected_producer: &[u8; 32],
        expected_carrier: &[u8; 32],
        expected_block_hash: &[u8; 32],
        expected_descriptor_hash: &[u8; 32],
    ) -> DirectoryDescriptorInclusionProofV1 {
        let DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
            chain_id,
            request_id,
            producer,
            carrier,
            response_timestamp,
            block_hash,
            descriptor_hash,
            proof,
            signature,
        } = decode_directory_sync_message(body).unwrap()
        else {
            panic!("unexpected replica descriptor proof response");
        };
        assert_eq!(&request_id, expected_request_id);
        assert_eq!(&producer, expected_producer);
        assert_eq!(&carrier, expected_carrier);
        assert_eq!(&block_hash, expected_block_hash);
        assert_eq!(&descriptor_hash, expected_descriptor_hash);
        let signing_bytes = directory_replica_descriptor_inclusion_proof_response_signing_bytes(
            &chain_id,
            &request_id,
            &producer,
            &carrier,
            response_timestamp,
            &block_hash,
            &descriptor_hash,
            &proof,
        );
        IdentityPublicKey::from_bytes(&carrier)
            .unwrap()
            .verify(&signing_bytes, &signature)
            .unwrap();
        proof
            .verify_at(&chain_id, &producer, &block_hash, now_secs())
            .unwrap();
        assert_eq!(proof.commitment.descriptor_hash, descriptor_hash);
        proof
    }

    fn witness_test_router() -> (
        Router,
        Arc<IdentityKeyPair>,
        IdentityKeyPair,
        Arc<DirectoryReplicaSyncRuntime>,
    ) {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap());
        let observer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_descriptor(&observer, now),
                now,
                "directory_witness_test",
            )
            .unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.keep();
        let (chain_store, _) = DirectoryChainStore::open(
            root.join("directory-chain.db"),
            witness.public_key_bytes(),
            now,
        )
        .unwrap();
        let (replica_store, _) = DirectoryReplicaStore::open(
            root.join("directory-replica.db"),
            witness.public_key_bytes(),
            now,
        )
        .unwrap();
        let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        (
            build_directory_chain_peer_router_with_replica_and_runtime(
                Arc::new(chain_store),
                Some(Arc::new(replica_store)),
                peer_store,
                Arc::clone(&witness),
                vec![observer.public_key_bytes()],
                false,
                Arc::clone(&runtime),
            ),
            witness,
            observer,
            runtime,
        )
    }

    #[derive(Clone)]
    struct FixedWitnessCarrierTransport {
        result: Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError>,
        calls: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl WitnessCarrierTransport for FixedWitnessCarrierTransport {
        async fn send(
            &self,
            _url: reqwest::Url,
            _request_frame: Vec<u8>,
        ) -> Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.result.clone()
        }
    }

    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Deterministic transport
    /// that fails once and then signs each distinct carried request as the
    /// target witness, preserving realistic replay semantics in recovery tests.
    struct RecoveringWitnessCarrierTransport {
        observer: IdentityKeyPair,
        witness: IdentityKeyPair,
        calls: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl WitnessCarrierTransport for RecoveringWitnessCarrierTransport {
        async fn send(
            &self,
            _url: reqwest::Url,
            request_frame: Vec<u8>,
        ) -> Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                Err(WitnessCarrierTransportError::TargetUnavailable)
            } else {
                Ok(WitnessCarrierTransportResponse {
                    status: 200,
                    body: witness_carrier_target_response(
                        &self.observer,
                        &self.witness,
                        &request_frame,
                    ),
                })
            }
        }
    }

    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Holds one outbound slot
    /// until the admission test explicitly releases it.
    struct BlockingWitnessCarrierTransport {
        response: WitnessCarrierTransportResponse,
        calls: Arc<AtomicU64>,
        entered: Arc<Barrier>,
        release: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl WitnessCarrierTransport for BlockingWitnessCarrierTransport {
        async fn send(
            &self,
            _url: reqwest::Url,
            _request_frame: Vec<u8>,
        ) -> Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.entered.wait().await;
            self.release.notified().await;
            Ok(self.response.clone())
        }
    }

    /// Builds a carrier with an independently pinned observer and target.
    ///
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] The advertised endpoint
    /// remains a syntactically public literal so the production SSRF gate runs.
    /// Only bounded outbound execution is replaced by a deterministic test
    /// runtime, and the peer store is returned for descriptor-rotation tests.
    fn witness_carrier_test_router_with_transport(
        transport: Arc<dyn WitnessCarrierTransport>,
        target_pinned: bool,
        advertise_target: bool,
        max_in_flight: usize,
    ) -> (
        Router,
        IdentityKeyPair,
        IdentityKeyPair,
        Arc<DirectoryReplicaSyncRuntime>,
        Arc<PeerStore>,
    ) {
        let now = now_secs();
        let carrier = Arc::new(IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap());
        let observer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xd9; 32]).unwrap();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_descriptor(&observer, now),
                now,
                "directory_witness_carrier_test",
            )
            .unwrap();
        if advertise_target {
            let mut descriptor = NodeDescriptor::new(
                witness.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "directory-sync-test",
            );
            descriptor.public_endpoint = Some("1.1.1.1:8422".to_string());
            peer_store
                .upsert_verified_from_source(
                    SignedNodeDescriptor::sign(descriptor, &witness).unwrap(),
                    now,
                    "directory_witness_carrier_test",
                )
                .unwrap();
        }
        let temp = TempDir::new().unwrap();
        let root = temp.keep();
        let (chain_store, _) = DirectoryChainStore::open(
            root.join("directory-chain.db"),
            carrier.public_key_bytes(),
            now,
        )
        .unwrap();
        let (replica_store, _) = DirectoryReplicaStore::open(
            root.join("directory-replica.db"),
            carrier.public_key_bytes(),
            now,
        )
        .unwrap();
        let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        let mut pins = vec![observer.public_key_bytes()];
        if target_pinned {
            pins.push(witness.public_key_bytes());
        }
        (
            build_directory_chain_peer_router_with_replica_runtime_and_transport(
                Arc::new(chain_store),
                Some(Arc::new(replica_store)),
                Arc::clone(&peer_store),
                carrier,
                pins,
                false,
                Arc::clone(&runtime),
                Arc::new(WitnessCarrierRuntime::new(transport, max_in_flight)),
            ),
            observer,
            witness,
            runtime,
            peer_store,
        )
    }

    fn witness_carrier_test_router(
        transport_result: Result<WitnessCarrierTransportResponse, WitnessCarrierTransportError>,
        target_pinned: bool,
        advertise_target: bool,
    ) -> (
        Router,
        IdentityKeyPair,
        IdentityKeyPair,
        Arc<DirectoryReplicaSyncRuntime>,
        Arc<AtomicU64>,
    ) {
        let calls = Arc::new(AtomicU64::new(0));
        let transport = Arc::new(FixedWitnessCarrierTransport {
            result: transport_result,
            calls: Arc::clone(&calls),
        });
        let (router, observer, witness, runtime, _) = witness_carrier_test_router_with_transport(
            transport,
            target_pinned,
            advertise_target,
            MAX_WITNESS_CARRIER_REQUESTS_IN_FLIGHT,
        );
        (router, observer, witness, runtime, calls)
    }

    fn witness_carrier_inner_request(observer: &IdentityKeyPair) -> Vec<u8> {
        witness_carrier_inner_request_with_id(observer, [0xdf; 16])
    }

    fn witness_carrier_inner_request_with_id(
        observer: &IdentityKeyPair,
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let producer_a = IdentityKeyPair::from_bytes(&[0xda; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xdb; 32]).unwrap();
        let now = now_secs();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            1,
            now,
            [0u8; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 1,
                    tip_hash: [0xdc; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 1,
                    tip_hash: [0xdd; 32],
                },
            ],
            [0xde; 32],
            observer,
        )
        .unwrap();
        let checkpoint_hash = checkpoint.hash();
        let signing_bytes = directory_observation_witness_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            now,
            &checkpoint_hash,
        );
        encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                requester: observer.public_key_bytes(),
                request_timestamp: now,
                checkpoint,
                signature: observer.sign(&signing_bytes),
            },
        )
        .unwrap()
    }

    fn witness_carrier_outer_request(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        inner_frame: Vec<u8>,
    ) -> Vec<u8> {
        witness_carrier_outer_request_with_id(observer, witness, inner_frame, [0xe0; 16])
    }

    fn witness_carrier_outer_request_with_id(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        inner_frame: Vec<u8>,
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let now = now_secs();
        let inner_sha256: [u8; 32] = Sha256::digest(&inner_frame).into();
        let signing_bytes = directory_observation_witness_carrier_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            now,
            &witness.public_key_bytes(),
            &inner_sha256,
            u64::try_from(inner_frame.len()).unwrap(),
        );
        encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                requester: observer.public_key_bytes(),
                request_timestamp: now,
                witness: witness.public_key_bytes(),
                witness_request_sha256: inner_sha256,
                witness_request_frame: inner_frame,
                signature: observer.sign(&signing_bytes),
            },
        )
        .unwrap()
    }

    fn witness_carrier_target_response(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        inner_frame: &[u8],
    ) -> Vec<u8> {
        let now = now_secs();
        let request = verify_carried_observation_witness_request(
            inner_frame,
            &observer.public_key_bytes(),
            now,
        )
        .unwrap();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request.request_id,
            &request.requester,
            request.checkpoint_sequence,
            &request.checkpoint_hash,
            &witness.public_key_bytes(),
            now,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id: request.request_id,
                observer: request.requester,
                checkpoint_sequence: request.checkpoint_sequence,
                checkpoint_hash: request.checkpoint_hash,
                responder: witness.public_key_bytes(),
                response_timestamp: now,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&signing_bytes),
            },
        )
        .unwrap()
    }

    fn assert_single_witness_carrier_outcome(
        snapshot: crate::services::directory_replica::DirectoryObservationWitnessCarrierSnapshot,
        expected: DirectoryObservationWitnessCarrierOutcome,
    ) {
        assert_eq!(snapshot.requests, 1);
        assert_eq!(
            snapshot.forwarded
                + snapshot.policy_rejected
                + snapshot.invalid_requests
                + snapshot.target_unavailable
                + snapshot.target_capability_unavailable
                + snapshot.target_rejected
                + snapshot.target_invalid_response
                + snapshot.target_cooling_down
                + snapshot.local_overloaded
                + snapshot.local_failures,
            1,
            "one authenticated request must have exactly one terminal outcome"
        );
        let actual = match expected {
            DirectoryObservationWitnessCarrierOutcome::Forwarded => snapshot.forwarded,
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected => snapshot.policy_rejected,
            DirectoryObservationWitnessCarrierOutcome::InvalidRequest => snapshot.invalid_requests,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable => {
                snapshot.target_unavailable
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable => {
                snapshot.target_capability_unavailable
            }
            DirectoryObservationWitnessCarrierOutcome::TargetRejected => snapshot.target_rejected,
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse => {
                snapshot.target_invalid_response
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCoolingDown => {
                snapshot.target_cooling_down
            }
            DirectoryObservationWitnessCarrierOutcome::LocalOverloaded => snapshot.local_overloaded,
            DirectoryObservationWitnessCarrierOutcome::LocalFailure => snapshot.local_failures,
        };
        assert_eq!(actual, 1);
    }

    fn import_certificate_test_producer(
        store: &DirectoryReplicaStore,
        producer: &IdentityKeyPair,
        object: &SignedNodeDescriptor,
        block: &DirectoryCommitmentBlockV1,
        request_id: [u8; 16],
        now: u64,
    ) {
        let responder = producer.public_key_bytes();
        let response_signing = directory_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            std::slice::from_ref(block),
            false,
            1,
            &block.hash(),
        );
        let response_frame =
            encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                responder,
                response_timestamp: now,
                blocks: vec![block.clone()],
                has_more: false,
                tip_height: 1,
                tip_hash: block.hash(),
                signature: producer.sign(&response_signing),
            })
            .unwrap();
        store
            .import_verified_page(
                responder,
                std::slice::from_ref(block),
                std::slice::from_ref(object),
                1,
                block.hash(),
                &response_frame,
                now,
            )
            .unwrap();
    }

    fn positive_certificate_test_router() -> (Router, Arc<IdentityKeyPair>, IdentityKeyPair, u64) {
        let now = now_secs();
        let observer = Arc::new(IdentityKeyPair::from_bytes(&[0xcd; 32]).unwrap());
        let witness_a = IdentityKeyPair::from_bytes(&[0xce; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0xcf; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0xd0; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
        let object = signed_descriptor(&subject, now);
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object).unwrap();
        let block_a = DirectoryCommitmentBlockV1::new_signed(
            1,
            now,
            [0u8; 32],
            vec![commitment],
            &producer_a,
        )
        .unwrap();
        let block_b = DirectoryCommitmentBlockV1::new_signed(
            1,
            now,
            [0u8; 32],
            vec![commitment],
            &producer_b,
        )
        .unwrap();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_descriptor(&witness_a, now),
                now,
                "directory_certificate_test",
            )
            .unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.keep();
        let (chain_store, _) = DirectoryChainStore::open(
            root.join("directory-chain.db"),
            observer.public_key_bytes(),
            now,
        )
        .unwrap();
        let (replica_store, _) = DirectoryReplicaStore::open(
            root.join("directory-replica.db"),
            observer.public_key_bytes(),
            now,
        )
        .unwrap();
        import_certificate_test_producer(
            &replica_store,
            &producer_a,
            &object,
            &block_a,
            [0xd3; 16],
            now,
        );
        import_certificate_test_producer(
            &replica_store,
            &producer_b,
            &object,
            &block_b,
            [0xd4; 16],
            now,
        );
        let producers = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        replica_store
            .append_observation_checkpoint(&producers, &observer, now)
            .unwrap();
        let checkpoint = replica_store
            .latest_audited_observation_checkpoint(now)
            .unwrap()
            .unwrap();
        let witness_ids = [witness_a.public_key_bytes(), witness_b.public_key_bytes()];
        replica_store
            .reconcile_observation_witness_policy(&observer, &witness_ids, 2, now)
            .unwrap();
        for (witness, request_id) in [(&witness_a, [0xd5; 16]), (&witness_b, [0xd6; 16])] {
            let checkpoint_hash = checkpoint.hash();
            let signing = directory_observation_witness_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &observer.public_key_bytes(),
                checkpoint.sequence,
                &checkpoint_hash,
                &witness.public_key_bytes(),
                now,
                DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            );
            let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                observer: observer.public_key_bytes(),
                checkpoint_sequence: checkpoint.sequence,
                checkpoint_hash,
                responder: witness.public_key_bytes(),
                response_timestamp: now,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&signing),
            };
            assert!(replica_store
                .persist_observation_checkpoint_witness(&response, now)
                .unwrap());
        }
        (
            build_directory_chain_peer_router_with_replica(
                Arc::new(chain_store),
                Some(Arc::new(replica_store)),
                peer_store,
                Arc::clone(&observer),
                witness_ids.to_vec(),
                false,
            ),
            observer,
            witness_a,
            checkpoint.sequence,
        )
    }

    #[derive(Clone, Copy)]
    enum CarrierTestPolicy {
        PinnedAuthority,
        PublicMirror,
        PublicWithoutMirror,
        PublicMirrorDisabled,
    }

    #[allow(clippy::too_many_lines)]
    fn carrier_test_router_with_access(
        policy: CarrierTestPolicy,
    ) -> (
        Router,
        Arc<IdentityKeyPair>,
        IdentityKeyPair,
        IdentityKeyPair,
        SignedNodeDescriptor,
    ) {
        let (mirror_namespace, requester_pinned, producer_pinned, allow_public_mirror_reads) =
            match policy {
                CarrierTestPolicy::PinnedAuthority => (false, true, true, false),
                CarrierTestPolicy::PublicMirror => (true, false, false, true),
                CarrierTestPolicy::PublicWithoutMirror => (false, false, false, true),
                CarrierTestPolicy::PublicMirrorDisabled => (true, false, false, false),
            };
        let now = now_secs();
        let carrier = Arc::new(IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap());
        let requester = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0xd3; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xd4; 32]).unwrap();
        let object = signed_descriptor(&subject, now);
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object).unwrap();
        let block =
            DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
                .unwrap();
        let request_id = [0xd5; 16];
        let response_signing = directory_block_range_response_signing_bytes(
            &request_id,
            &producer.public_key_bytes(),
            now,
            std::slice::from_ref(&block),
            false,
            1,
            &block.hash(),
        );
        let response_frame =
            encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                responder: producer.public_key_bytes(),
                response_timestamp: now,
                blocks: vec![block.clone()],
                has_more: false,
                tip_height: 1,
                tip_hash: block.hash(),
                signature: producer.sign(&response_signing),
            })
            .unwrap();
        let peer_store = Arc::new(PeerStore::new());
        let mut requester_descriptor = signed_descriptor(&requester, now);
        requester_descriptor.descriptor.policy.public_discovery = true;
        requester_descriptor =
            SignedNodeDescriptor::sign(requester_descriptor.descriptor, &requester).unwrap();
        peer_store
            .upsert_verified_from_source(requester_descriptor, now, "directory_carrier_test")
            .unwrap();
        peer_store
            .upsert_verified_from_source(
                signed_descriptor(&producer, now),
                now,
                "directory_carrier_test",
            )
            .unwrap();
        let temp = TempDir::new().unwrap();
        let root = temp.keep();
        let path = root.join("directory.db");
        let (chain_store, _) =
            DirectoryChainStore::open(&path, carrier.public_key_bytes(), now).unwrap();
        let (replica_store, _) =
            DirectoryReplicaStore::open(&path, carrier.public_key_bytes(), now).unwrap();
        if mirror_namespace {
            replica_store
                .import_verified_mirror_page(
                    producer.public_key_bytes(),
                    1,
                    4,
                    std::slice::from_ref(&block),
                    std::slice::from_ref(&object),
                    1,
                    block.hash(),
                    &response_frame,
                    now,
                )
                .unwrap();
        } else {
            replica_store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(&block),
                    std::slice::from_ref(&object),
                    1,
                    block.hash(),
                    &response_frame,
                    now,
                )
                .unwrap();
        }
        let mut pins = Vec::new();
        if requester_pinned {
            pins.push(requester.public_key_bytes());
        }
        if producer_pinned {
            pins.push(producer.public_key_bytes());
        }
        (
            build_directory_chain_peer_router_with_replica(
                Arc::new(chain_store),
                Some(Arc::new(replica_store)),
                peer_store,
                Arc::clone(&carrier),
                pins,
                allow_public_mirror_reads,
            ),
            carrier,
            requester,
            producer,
            object,
        )
    }

    fn carrier_test_router() -> (
        Router,
        Arc<IdentityKeyPair>,
        IdentityKeyPair,
        IdentityKeyPair,
        SignedNodeDescriptor,
    ) {
        carrier_test_router_with_access(CarrierTestPolicy::PinnedAuthority)
    }
}
