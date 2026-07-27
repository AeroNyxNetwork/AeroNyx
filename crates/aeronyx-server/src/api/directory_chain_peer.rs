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
//! - `POST /api/discovery/peer/directory/replica-block-range`
//! - `POST /api/discovery/peer/directory/replica-descriptor-objects`
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
//! - Independent checkpoint root recomputation before a signed witness decision.
//! - Monotonic opaque policy-head retention before a signed anchor decision.
//! - Pinned-peer-only export of the latest locally verified portable
//!   observation certificate with exact frame digest binding.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Pinned-authority-only, one-hop
//!   transport of an exact observer-signed witness request to an exact pinned
//!   witness. The carrier verifies both inner frames but never signs evidence.
//! - [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Process-only mutually
//!   exclusive carrier outcomes with no identity, route, or frame retention.
//! - Audited producer-replica export with a separate carrier signature layer.
//! - Explicit lagging-carrier responses that let a requester continue to the
//!   next verified carrier without retrying malformed protocol requests.
//! - Multi-block catch-up pages capped to one block's maximum aggregate
//!   commitment budget and stopped before repeated descriptor objects.
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
//!
//! ## Last Modified
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
use tokio::sync::Mutex;
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_request_signing_bytes,
    directory_block_range_response_signing_bytes,
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
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes,
    directory_tip_request_signing_bytes, directory_tip_response_signing_bytes,
    encode_directory_observation_certificate, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectoryObservationCheckpointV1, DirectorySyncMessage,
    SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1, DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES, MAX_DIRECTORY_SYNC_BLOCKS_V1,
    MAX_DIRECTORY_SYNC_OBJECTS_V1,
};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
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

#[derive(Clone)]
struct DirectoryChainPeerState {
    store: Arc<DirectoryChainStore>,
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peers: Arc<HashSet<[u8; 32]>>,
    allow_public_mirror_reads: bool,
    guard: Arc<Mutex<DirectoryPeerRequestGuard>>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    witness_carrier_transport: Arc<dyn WitnessCarrierTransport>,
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
        Arc::new(ReqwestWitnessCarrierTransport::new()),
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
    witness_carrier_transport: Arc<dyn WitnessCarrierTransport>,
) -> Router {
    let state = DirectoryChainPeerState {
        store,
        replica_store,
        peer_store,
        identity,
        pinned_peers: Arc::new(pinned_peer_ids.into_iter().collect()),
        allow_public_mirror_reads,
        guard: Arc::new(Mutex::new(DirectoryPeerRequestGuard::default())),
        runtime,
        witness_carrier_transport,
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

async fn authenticate_request(
    state: &DirectoryChainPeerState,
    admission: DirectoryPeerAdmission,
    requester: [u8; 32],
    request_id: [u8; 16],
    request_timestamp: u64,
    signing_bytes: &[u8],
    signature: &[u8; 64],
    now: u64,
) -> Result<(), Response> {
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err(protocol_error(StatusCode::UNAUTHORIZED, "stale_request"));
    }
    if admission == DirectoryPeerAdmission::PinnedAuthority
        && !state.pinned_peers.contains(&requester)
    {
        return Err(protocol_error(StatusCode::FORBIDDEN, "peer_not_pinned"));
    }
    let requester_is_pinned = state.pinned_peers.contains(&requester);
    let permissionless_read = matches!(
        admission,
        DirectoryPeerAdmission::VerifiedPublicMirror
            | DirectoryPeerAdmission::VerifiedPublicRecovery
    );
    if permissionless_read && !state.allow_public_mirror_reads && !requester_is_pinned {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(descriptor) = state.peer_store.get_valid(&requester, now) else {
        return Err(protocol_error(StatusCode::FORBIDDEN, "unknown_peer"));
    };
    let public_descriptor_required = admission == DirectoryPeerAdmission::VerifiedPublicMirror
        || (admission == DirectoryPeerAdmission::VerifiedPublicRecovery && !requester_is_pinned);
    if public_descriptor_required && !descriptor.descriptor.policy.public_discovery {
        return Err(protocol_error(StatusCode::FORBIDDEN, "peer_not_public"));
    }
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(signing_bytes, signature))
        .is_err()
    {
        return Err(protocol_error(
            StatusCode::UNAUTHORIZED,
            "invalid_signature",
        ));
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return Err(protocol_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_or_replay_limited",
        ));
    }
    Ok(())
}

fn decode_request(body: &[u8]) -> Result<DirectorySyncMessage, Response> {
    let message = decode_directory_sync_message(body)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    if canonical != body {
        return Err(protocol_error(
            StatusCode::BAD_REQUEST,
            "noncanonical_frame",
        ));
    }
    Ok(message)
}

fn bounded_directory_transport_blocks(
    blocks: Vec<DirectoryCommitmentBlockV1>,
) -> Vec<DirectoryCommitmentBlockV1> {
    let mut commitment_count = 0usize;
    let mut descriptor_hashes = HashSet::new();
    let mut accepted = 0usize;
    for block in &blocks {
        let Some(next_count) = commitment_count.checked_add(block.commitments.len()) else {
            break;
        };
        if next_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK
            || block
                .commitments
                .iter()
                .any(|commitment| descriptor_hashes.contains(&commitment.descriptor_hash))
        {
            break;
        }
        descriptor_hashes.extend(
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash),
        );
        commitment_count = next_count;
        accepted = accepted.saturating_add(1);
    }
    blocks.into_iter().take(accepted).collect()
}

async fn tip_handler(State(state): State<DirectoryChainPeerState>, body: Bytes) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::TipRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID {
        return protocol_error(StatusCode::BAD_REQUEST, "wrong_chain");
    }
    let now = now_secs();
    let signing_bytes =
        directory_tip_request_signing_bytes(&chain_id, &request_id, &requester, request_timestamp);
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let audit = match tokio::task::spawn_blocking(move || store.audited_tip(now)).await {
        Ok(Ok(audit)) => audit,
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_tip_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        audit.tip_height,
        &audit.tip_hash,
        audit.tip_timestamp,
    );
    encoded_response(DirectorySyncMessage::TipResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        tip_height: audit.tip_height,
        tip_hash: audit.tip_hash,
        tip_timestamp: audit.tip_timestamp,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn block_range_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::BlockRangeRequestV1 {
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
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || from_height == 0
        || limit == 0
        || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    let now = now_secs();
    let signing_bytes = directory_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let page = match tokio::task::spawn_blocking(move || {
        store.audited_block_page(from_height, limit, now)
    })
    .await
    {
        Ok(Ok(page)) => page,
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let blocks = bounded_directory_transport_blocks(page.blocks);
    let has_more = blocks
        .last()
        .is_some_and(|block| block.header.height < page.tip_height);
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        page.tip_height,
        &page.tip_hash,
    );
    debug!(
        blocks = blocks.len(),
        has_more,
        tip_height = page.tip_height,
        "[DIRECTORY_CHAIN] Served authenticated bounded block page"
    );
    encoded_response(DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height: page.tip_height,
        tip_hash: page.tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn descriptor_objects_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::DescriptorObjectsRequestV1 {
        chain_id,
        descriptor_hashes,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let mut unique_hashes = descriptor_hashes.clone();
    unique_hashes.sort_unstable();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || descriptor_hashes.is_empty()
        || descriptor_hashes.len() > MAX_DIRECTORY_SYNC_OBJECTS_V1
        || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
        || unique_hashes.windows(2).any(|pair| pair[0] == pair[1])
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_object_request");
    }
    let now = now_secs();
    let signing_bytes = directory_descriptor_objects_request_signing_bytes(
        &chain_id,
        &descriptor_hashes,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let requested_hashes = descriptor_hashes.clone();
    let objects = match tokio::task::spawn_blocking(move || {
        store.audited_descriptor_objects(&requested_hashes, now)
    })
    .await
    {
        Ok(Ok(Some(objects))) => objects,
        Ok(Ok(None)) => return protocol_error(StatusCode::NOT_FOUND, "object_not_found"),
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_descriptor_objects_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &descriptor_hashes,
    );
    debug!(
        objects = objects.len(),
        "[DIRECTORY_CHAIN] Served authenticated descriptor objects"
    );
    encoded_response(DirectorySyncMessage::DescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn audited_replica_page_for_request(
    state: &DirectoryChainPeerState,
    producer: [u8; 32],
    from_height: u64,
    limit: u16,
    observed_at: u64,
) -> Result<DirectoryReplicaEvidencePage, Response> {
    let producer_is_pinned = state.pinned_peers.contains(&producer);
    if !producer_is_pinned && !state.allow_public_mirror_reads {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    match tokio::task::spawn_blocking(move || {
        if producer_is_pinned {
            store.audited_evidence_page(&producer, from_height, limit, observed_at)
        } else {
            store.audited_mirror_evidence_page(&producer, from_height, limit, observed_at)
        }
    })
    .await
    {
        Ok(Ok(page)) if page.tip_height > 0 => Ok(page),
        Ok(Ok(_)) => Err(protocol_error(StatusCode::NOT_FOUND, "replica_not_found")),
        Ok(Err(error)) => Err(replica_store_error_response(&error)),
        Err(_) => Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "audit_task_failed",
        )),
    }
}

async fn audited_replica_objects_for_request(
    state: &DirectoryChainPeerState,
    producer: [u8; 32],
    descriptor_hashes: Vec<[u8; 32]>,
    observed_at: u64,
) -> Result<Vec<SignedNodeDescriptor>, Response> {
    let producer_is_pinned = state.pinned_peers.contains(&producer);
    if !producer_is_pinned && !state.allow_public_mirror_reads {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    match tokio::task::spawn_blocking(move || {
        if producer_is_pinned {
            store.audited_evidence_descriptor_objects(&producer, &descriptor_hashes, observed_at)
        } else {
            store.audited_mirror_evidence_descriptor_objects(
                &producer,
                &descriptor_hashes,
                observed_at,
            )
        }
    })
    .await
    {
        Ok(Ok(Some(objects))) => Ok(objects),
        Ok(Ok(None)) => Err(protocol_error(
            StatusCode::NOT_FOUND,
            "replica_object_not_found",
        )),
        Ok(Err(error)) => Err(replica_store_error_response(&error)),
        Err(_) => Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "audit_task_failed",
        )),
    }
}

async fn replica_block_range_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
        chain_id,
        producer,
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
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || producer == [0u8; 32]
        || producer == state.identity.public_key_bytes()
        || from_height == 0
        || limit == 0
        || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_range");
    }
    let now = now_secs();
    let signing_bytes = directory_replica_block_range_request_signing_bytes(
        &chain_id,
        &producer,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicRecovery,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let page =
        match audited_replica_page_for_request(&state, producer, from_height, limit, now).await {
            Ok(page) => page,
            Err(response) => return response,
        };
    let blocks = bounded_directory_transport_blocks(page.blocks);
    let has_more = blocks
        .last()
        .is_some_and(|block| block.header.height < page.tip_height);
    let carrier = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_replica_block_range_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &blocks,
        has_more,
        page.tip_height,
        &page.tip_hash,
    );
    debug!(
        blocks = blocks.len(),
        has_more,
        tip_height = page.tip_height,
        "[DIRECTORY_CHAIN] Served audited replica evidence page"
    );
    encoded_response(DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        blocks,
        has_more,
        tip_height: page.tip_height,
        tip_hash: page.tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn replica_descriptor_objects_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
        chain_id,
        producer,
        descriptor_hashes,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let unique_hashes = descriptor_hashes.iter().copied().collect::<HashSet<_>>();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || producer == [0u8; 32]
        || producer == state.identity.public_key_bytes()
        || descriptor_hashes.is_empty()
        || descriptor_hashes.len() > MAX_DIRECTORY_SYNC_OBJECTS_V1
        || unique_hashes.len() != descriptor_hashes.len()
        || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_object_request");
    }
    let now = now_secs();
    let signing_bytes = directory_replica_descriptor_objects_request_signing_bytes(
        &chain_id,
        &producer,
        &descriptor_hashes,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicRecovery,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let objects =
        match audited_replica_objects_for_request(&state, producer, descriptor_hashes.clone(), now)
            .await
        {
            Ok(objects) => objects,
            Err(response) => return response,
        };
    let carrier = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_replica_descriptor_objects_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &descriptor_hashes,
    );
    debug!(
        objects = objects.len(),
        "[DIRECTORY_CHAIN] Served audited replica descriptor objects"
    );
    encoded_response(DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn observation_checkpoint_witness_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester != checkpoint.observer
        || requester == state.identity.public_key_bytes()
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_witness_request");
    }
    let checkpoint_hash = checkpoint.hash();
    let now = now_secs();
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let checkpoint_sequence = checkpoint.sequence;
    let decision = match independently_evaluate_checkpoint(&state, &checkpoint, now).await {
        Ok(decision) => decision,
        Err(response) => return response,
    };
    let outcome = match decision {
        DirectoryObservationWitnessDecision::Accepted => DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        DirectoryObservationWitnessDecision::EvidenceUnavailable => {
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1
        }
        DirectoryObservationWitnessDecision::EvidenceConflict => {
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_observation_witness_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        checkpoint_sequence,
        &checkpoint_hash,
        &responder,
        response_timestamp,
        outcome,
    );
    debug!(
        accepted = outcome == DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        checkpoint_sequence,
        "[DIRECTORY_CHAIN] Evaluated authenticated observation checkpoint witness"
    );
    encoded_response(
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer: requester,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature: state.identity.sign(&response_signing_bytes),
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct CarriedObservationWitnessRequestContext {
    request_id: [u8; 16],
    requester: [u8; 32],
    request_timestamp: u64,
    checkpoint_sequence: u64,
    checkpoint_hash: [u8; 32],
}

fn verify_carried_observation_witness_request(
    frame: &[u8],
    expected_requester: &[u8; 32],
    observed_at: u64,
) -> Result<CarriedObservationWitnessRequestContext, &'static str> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "carried_witness_request_decode_failed")?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "carried_witness_request_encode_failed")?;
    if canonical != frame {
        return Err("carried_witness_request_noncanonical");
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature,
    } = message
    else {
        return Err("carried_witness_request_unexpected_message");
    };
    let checkpoint_hash = checkpoint.hash();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester != *expected_requester
        || requester != checkpoint.observer
        || observed_at.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .is_err()
    {
        return Err("carried_witness_request_contract_mismatch");
    }
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
    );
    IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "carried_witness_request_invalid_signature")?;
    Ok(CarriedObservationWitnessRequestContext {
        request_id,
        requester,
        request_timestamp,
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
    })
}

fn verify_carried_observation_witness_response(
    frame: &[u8],
    request: &CarriedObservationWitnessRequestContext,
    expected_witness: &[u8; 32],
    observed_at: u64,
) -> Result<(), &'static str> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "carried_witness_response_decode_failed")?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "carried_witness_response_encode_failed")?;
    if canonical != frame {
        return Err("carried_witness_response_noncanonical");
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id,
        request_id,
        observer,
        checkpoint_sequence,
        checkpoint_hash,
        responder,
        response_timestamp,
        outcome,
        signature,
    } = message
    else {
        return Err("carried_witness_response_unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != request.request_id
        || observer != request.requester
        || checkpoint_sequence != request.checkpoint_sequence
        || checkpoint_hash != request.checkpoint_hash
        || responder != *expected_witness
        || observed_at.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS)
            < request.request_timestamp
        || ![
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
        ]
        .contains(&outcome)
    {
        return Err("carried_witness_response_contract_mismatch");
    }
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &chain_id,
        &request_id,
        &observer,
        checkpoint_sequence,
        &checkpoint_hash,
        &responder,
        response_timestamp,
        outcome,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "carried_witness_response_invalid_signature")
}

async fn observation_checkpoint_witness_carrier_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        witness,
        witness_request_sha256,
        witness_request_frame,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let carrier = state.identity.public_key_bytes();
    let actual_request_sha256: [u8; 32] = Sha256::digest(&witness_request_frame).into();
    let request_frame_bytes = u64::try_from(witness_request_frame.len()).unwrap_or(u64::MAX);
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester == carrier
        || requester == witness
        || witness == carrier
        || witness_request_frame.is_empty()
        || witness_request_frame.len() > MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES
        || witness_request_sha256 == [0u8; 32]
        || witness_request_sha256 != actual_request_sha256
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_witness_carrier_request");
    }
    let now = now_secs();
    let signing_bytes = directory_observation_witness_carrier_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &witness,
        &witness_request_sha256,
        request_frame_bytes,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    if !state.pinned_peers.contains(&witness) {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected,
            protocol_error(StatusCode::FORBIDDEN, "witness_target_not_pinned"),
        );
    }
    let carried_request =
        match verify_carried_observation_witness_request(&witness_request_frame, &requester, now) {
            Ok(request) => request,
            Err(_) => {
                return witness_carrier_outcome_response(
                    &state,
                    DirectoryObservationWitnessCarrierOutcome::InvalidRequest,
                    protocol_error(StatusCode::BAD_REQUEST, "invalid_inner_witness_request"),
                );
            }
        };
    let Some(descriptor) = state.peer_store.get_valid(&witness, now) else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    if !commitment_peer_endpoint_is_public(endpoint) {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    }
    let Ok(url) = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-checkpoint-witness",
    ) else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    let target_response = match state
        .witness_carrier_transport
        .send(url, witness_request_frame)
        .await
    {
        Ok(response) => response,
        Err(WitnessCarrierTransportError::LocalUnavailable) => {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::LocalFailure,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_carrier_transport_unavailable",
                ),
            );
        }
        Err(WitnessCarrierTransportError::TargetUnavailable) => {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_target_unavailable",
                ),
            );
        }
        Err(WitnessCarrierTransportError::ResponseTooLarge) => {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
                protocol_error(StatusCode::BAD_GATEWAY, "witness_target_invalid_response"),
            );
        }
    };
    let status = target_response.status;
    if !(200..300).contains(&status) {
        if matches!(status, 404 | 405 | 501) {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable,
                protocol_error(
                    StatusCode::FAILED_DEPENDENCY,
                    "witness_target_capability_unavailable",
                ),
            );
        }
        if matches!(status, 408 | 429) || (500..600).contains(&status) {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_target_unavailable",
                ),
            );
        }
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetRejected,
            protocol_error(StatusCode::BAD_GATEWAY, "witness_target_rejected"),
        );
    }
    let witness_response_frame = target_response.body;
    if witness_response_frame.is_empty()
        || verify_carried_observation_witness_response(
            &witness_response_frame,
            &carried_request,
            &witness,
            now_secs(),
        )
        .is_err()
    {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
            protocol_error(StatusCode::BAD_GATEWAY, "witness_target_invalid_response"),
        );
    }
    let response_timestamp = now_secs();
    let witness_response_sha256: [u8; 32] = Sha256::digest(&witness_response_frame).into();
    let response_frame_bytes = u64::try_from(witness_response_frame.len()).unwrap_or(u64::MAX);
    let response_signing_bytes = directory_observation_witness_carrier_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &witness,
        &carrier,
        response_timestamp,
        &witness_request_sha256,
        &witness_response_sha256,
        response_frame_bytes,
    );
    // [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Do not emit a per-request
    // carrier success event. Even an otherwise public checkpoint sequence plus
    // log time would create a cross-node correlation handle. The aggregate,
    // process-only outcome below is the complete operational signal.
    witness_carrier_outcome_response(
        &state,
        DirectoryObservationWitnessCarrierOutcome::Forwarded,
        encoded_response(
            DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
                chain_id,
                request_id,
                requester,
                witness,
                carrier,
                response_timestamp,
                witness_request_sha256,
                witness_response_sha256,
                witness_response_frame,
                signature: state.identity.sign(&response_signing_bytes),
            },
        ),
    )
}

/// Records one terminal carrier outcome without accepting identity-bearing data.
fn witness_carrier_outcome_response(
    state: &DirectoryChainPeerState,
    outcome: DirectoryObservationWitnessCarrierOutcome,
    response: Response,
) -> Response {
    state
        .runtime
        .record_observation_witness_carrier_outcome(outcome, now_secs());
    response
}

async fn observation_policy_anchor_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        policy_epoch,
        previous_policy_digest,
        policy_digest,
        signature,
    } = &message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let position_valid = (*policy_epoch == 1 && *previous_policy_digest == [0u8; 32])
        || (*policy_epoch > 1 && *previous_policy_digest != [0u8; 32]);
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || *requester == state.identity.public_key_bytes()
        || *policy_digest == [0u8; 32]
        || !position_valid
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_policy_anchor_request");
    }
    let now = now_secs();
    let signing_bytes = directory_policy_anchor_request_signing_bytes(
        chain_id,
        request_id,
        requester,
        *request_timestamp,
        *policy_epoch,
        previous_policy_digest,
        policy_digest,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
        *requester,
        *request_id,
        *request_timestamp,
        &signing_bytes,
        signature,
        now,
    )
    .await
    {
        return response;
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_store_disabled");
    };
    let anchor_request = message.clone();
    let decision = match tokio::task::spawn_blocking(move || {
        store.persist_remote_observation_witness_policy_anchor(&anchor_request, now)
    })
    .await
    {
        Ok(Ok(decision)) => decision,
        Ok(Err(error)) => return replica_store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let outcome = decision.outcome();
    let response_signing_bytes = directory_policy_anchor_response_signing_bytes(
        chain_id,
        request_id,
        requester,
        *policy_epoch,
        policy_digest,
        &responder,
        response_timestamp,
        outcome,
    );
    debug!(
        accepted = decision == DirectoryObservationWitnessPolicyAnchorDecision::Accepted,
        policy_epoch, "[DIRECTORY_CHAIN] Evaluated authenticated opaque policy-head anchor"
    );
    encoded_response(
        DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: *chain_id,
            request_id: *request_id,
            observer: *requester,
            policy_epoch: *policy_epoch,
            policy_digest: *policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature: state.identity.sign(&response_signing_bytes),
        },
    )
}

async fn observation_certificate_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCertificateRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester == state.identity.public_key_bytes()
    {
        return protocol_error(
            StatusCode::BAD_REQUEST,
            "invalid_observation_certificate_request",
        );
    }
    let now = now_secs();
    let signing_bytes = directory_observation_certificate_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_store_disabled");
    };
    let mut eligible_witnesses = state.pinned_peers.iter().copied().collect::<Vec<_>>();
    eligible_witnesses.sort_unstable();
    let certificate = match tokio::task::spawn_blocking(move || {
        let snapshot = store.status_snapshot()?;
        let minimum_witnesses =
            usize::try_from(snapshot.observation_witness_policy_threshold).unwrap_or(usize::MAX);
        if minimum_witnesses == 0 || minimum_witnesses > eligible_witnesses.len() {
            return Ok(None);
        }
        store.latest_observation_certificate_for_pins(&eligible_witnesses, minimum_witnesses, now)
    })
    .await
    {
        Ok(Ok(Some(certificate))) => certificate,
        Ok(Ok(None)) => {
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "observation_certificate_unavailable",
            )
        }
        Ok(Err(error)) => return replica_store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let certificate_frame = match encode_directory_observation_certificate(&certificate) {
        Ok(frame) if frame.len() <= MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES => frame,
        Ok(_) => {
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_oversized",
            )
        }
        Err(error) => {
            warn!(
                error = %error,
                "[DIRECTORY_CHAIN] Failed to encode verified observation certificate"
            );
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_encode_failed",
            );
        }
    };
    let certificate_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
    let certificate_frame_bytes = match u64::try_from(certificate_frame.len()) {
        Ok(bytes) => bytes,
        Err(_) => {
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_oversized",
            )
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_observation_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &responder,
        response_timestamp,
        &certificate_sha256,
        certificate_frame_bytes,
    );
    debug!(
        certificate_sequence = certificate.checkpoint.sequence,
        certificate_frame_bytes,
        "[DIRECTORY_CHAIN] Served authenticated portable observation certificate"
    );
    encoded_response(DirectorySyncMessage::ObservationCertificateResponseV1 {
        chain_id,
        request_id,
        requester,
        responder,
        response_timestamp,
        certificate_sha256,
        certificate_frame,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn independently_evaluate_checkpoint(
    state: &DirectoryChainPeerState,
    checkpoint: &DirectoryObservationCheckpointV1,
    now: u64,
) -> Result<DirectoryObservationWitnessDecision, Response> {
    let Some(replica_store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    let chain_store = Arc::clone(&state.store);
    match tokio::task::spawn_blocking(move || chain_store.audit(now)).await {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Local producer audit failed closed");
            return Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "chain_not_verified",
            ));
        }
        Err(_) => {
            return Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "audit_task_failed",
            ))
        }
    }
    let checkpoint = checkpoint.clone();
    match tokio::task::spawn_blocking(move || {
        replica_store.evaluate_observation_checkpoint_witness(&checkpoint, now)
    })
    .await
    {
        Ok(Ok(decision)) => Ok(decision),
        Ok(Err(DirectoryReplicaStoreError::Request(_))) => Err(protocol_error(
            StatusCode::BAD_REQUEST,
            "invalid_checkpoint",
        )),
        Ok(Err(error)) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Witness recomputation failed closed");
            Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "replica_not_verified",
            ))
        }
        Err(_) => Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "audit_task_failed",
        )),
    }
}

fn encoded_response(message: DirectorySyncMessage) -> Response {
    match encode_directory_sync_message(&message) {
        Ok(encoded) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            encoded,
        )
            .into_response(),
        Err(error) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Failed to encode peer response");
            protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error")
        }
    }
}

fn store_error_response(error: &DirectoryChainStoreError) -> Response {
    match error {
        DirectoryChainStoreError::Request(_) => {
            protocol_error(StatusCode::BAD_REQUEST, "invalid_request")
        }
        _ => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Refused unaudited peer export");
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified")
        }
    }
}

fn replica_store_error_response(error: &DirectoryReplicaStoreError) -> Response {
    match error {
        DirectoryReplicaStoreError::Request(_) => {
            protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_request")
        }
        // [MIRROR-CARRIER 2026-07-24 by Codex] A lagging carrier is an
        // availability miss, not evidence that the signed request is invalid.
        // A distinct 404 keeps true 400 contract failures fail-closed while the
        // requester advances to another bounded verified carrier.
        DirectoryReplicaStoreError::RangeNotRetained { .. } => {
            protocol_error(StatusCode::NOT_FOUND, "replica_range_not_retained")
        }
        DirectoryReplicaStoreError::Quarantined(_) => {
            protocol_error(StatusCode::CONFLICT, "producer_quarantined")
        }
        DirectoryReplicaStoreError::MirrorNotRetained => {
            protocol_error(StatusCode::NOT_FOUND, "mirror_replica_not_retained")
        }
        _ => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Refused unaudited replica export");
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_not_verified")
        }
    }
}

fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, [(header::CONTENT_TYPE, "text/plain")], code).into_response()
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
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
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

    #[test]
    fn request_guard_enforces_global_permissionless_budget() {
        let mut guard = DirectoryPeerRequestGuard::default();
        let now = 1_700_000_000;
        for index in 0..MAX_DIRECTORY_REQUESTS_GLOBAL_PER_MINUTE {
            let mut requester = [0u8; 32];
            requester[..4].copy_from_slice(&index.to_le_bytes());
            requester[31] = 1;
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&index.to_le_bytes());
            assert!(guard.admit(requester, request_id, now));
        }
        assert!(!guard.admit([0xff; 32], [0xff; 16], now));
        assert!(guard.admit([0xfe; 32], [0xfe; 16], now + 60));
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

    /// Builds a carrier with an independently pinned observer and target.
    ///
    /// [WITNESS-CARRIER-MATRIX 2026-07-27 by Codex] The advertised endpoint is
    /// a syntactically public literal so the production SSRF gate still runs;
    /// only the outbound transport is replaced by a deterministic test double.
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
        let calls = Arc::new(AtomicU64::new(0));
        let transport = Arc::new(FixedWitnessCarrierTransport {
            result: transport_result,
            calls: Arc::clone(&calls),
        });
        let mut pins = vec![observer.public_key_bytes()];
        if target_pinned {
            pins.push(witness.public_key_bytes());
        }
        (
            build_directory_chain_peer_router_with_replica_runtime_and_transport(
                Arc::new(chain_store),
                Some(Arc::new(replica_store)),
                peer_store,
                carrier,
                pins,
                false,
                Arc::clone(&runtime),
                transport,
            ),
            observer,
            witness,
            runtime,
            calls,
        )
    }

    fn witness_carrier_inner_request(observer: &IdentityKeyPair) -> Vec<u8> {
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
        let request_id = [0xdf; 16];
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
        let now = now_secs();
        let request_id = [0xe0; 16];
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

    #[tokio::test]
    async fn pinned_live_peer_receives_signed_audited_tip_and_replay_is_rejected() {
        let (router, producer, requester, _) = test_router(true, true, false);
        let request_id = [0xb1; 16];
        let request = tip_request(&requester, request_id);
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(request.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let DirectorySyncMessage::TipResponseV1 {
            chain_id,
            request_id: response_request_id,
            responder,
            response_timestamp,
            tip_height,
            tip_hash,
            tip_timestamp,
            signature,
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected response");
        };
        assert_eq!(chain_id, AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        assert_eq!(response_request_id, request_id);
        assert_eq!(responder, producer.public_key_bytes());
        assert_eq!(tip_height, 1);
        let signing_bytes = directory_tip_response_signing_bytes(
            &chain_id,
            &response_request_id,
            &responder,
            response_timestamp,
            tip_height,
            &tip_hash,
            tip_timestamp,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&signing_bytes, &signature)
            .unwrap();

        let replay = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn unpinned_public_peer_can_read_signed_local_producer_tip() {
        let (router, _, requester, _) = test_router(false, true, true);
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(tip_request(&requester, [0xb2; 16])))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn mirror_mode_disabled_preserves_pinned_only_read_admission() {
        let (router, _, requester, _) = test_router(false, true, false);
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(tip_request(&requester, [0xba; 16])))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn unpinned_private_peer_cannot_read_signed_local_producer_tip() {
        let (router, _, requester, _) = test_router(false, false, true);
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(tip_request(&requester, [0xb9; 16])))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn block_and_descriptor_routes_return_exact_committed_objects() {
        let (router, producer, requester, expected_descriptor) = test_router(true, true, false);
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let range_id = [0xb3; 16];
        let range_signing = directory_block_range_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            1,
            &range_id,
            &requester_id,
            timestamp,
        );
        let range = encode_directory_sync_message(&DirectorySyncMessage::BlockRangeRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: 1,
            request_id: range_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&range_signing),
        })
        .unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/block-range")
                    .body(Body::from(range))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let verified_range = verify_block_range_response(
            &body,
            &range_id,
            &producer.public_key_bytes(),
            1,
            timestamp,
        )
        .unwrap();
        assert_eq!(verified_range.0.len(), 1);
        let mut tampered_range = body.to_vec();
        *tampered_range.last_mut().unwrap() ^= 0x01;
        assert_eq!(
            verify_block_range_response(
                &tampered_range,
                &range_id,
                &producer.public_key_bytes(),
                1,
                timestamp,
            )
            .unwrap_err(),
            "directory_range_response_invalid_signature"
        );
        let DirectorySyncMessage::BlockRangeResponseV1 {
            blocks, responder, ..
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected range response");
        };
        assert_eq!(responder, producer.public_key_bytes());
        assert_eq!(blocks.len(), 1);
        let descriptor_hash = blocks[0].commitments[0].descriptor_hash;
        let expected_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&expected_descriptor).unwrap();
        assert_eq!(descriptor_hash, expected_commitment.descriptor_hash);

        let object_id = [0xb4; 16];
        let object_signing = directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &[descriptor_hash],
            &object_id,
            &requester_id,
            timestamp,
        );
        let object_request =
            encode_directory_sync_message(&DirectorySyncMessage::DescriptorObjectsRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                descriptor_hashes: vec![descriptor_hash],
                request_id: object_id,
                requester: requester_id,
                request_timestamp: timestamp,
                signature: requester.sign(&object_signing),
            })
            .unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/descriptor-objects")
                    .body(Body::from(object_request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let verified_objects = verify_descriptor_objects_response(
            &body,
            &object_id,
            &producer.public_key_bytes(),
            &[descriptor_hash],
            timestamp,
        )
        .unwrap();
        assert_eq!(verified_objects, vec![expected_descriptor.clone()]);
        let DirectorySyncMessage::DescriptorObjectsResponseV1 {
            descriptor_hashes,
            objects,
            ..
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected object response");
        };
        assert_eq!(descriptor_hashes, vec![descriptor_hash]);
        assert_eq!(objects, vec![expected_descriptor]);
    }

    #[tokio::test]
    async fn verified_public_peer_can_recover_only_registered_mirror_evidence() {
        let (router, carrier, requester, producer, expected_object) =
            carrier_test_router_with_access(CarrierTestPolicy::PublicMirror);
        let request_id = [0xce; 16];
        let producer_id = producer.public_key_bytes();
        let request = replica_range_request(&requester, &producer_id, request_id);
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-block-range")
                    .body(Body::from(request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let (blocks, has_more, tip_height, _) = verify_replica_block_range_response(
            &body,
            &request_id,
            &producer_id,
            &carrier.public_key_bytes(),
            1,
            now_secs(),
        )
        .unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(!has_more);
        assert_eq!(tip_height, 1);

        // [MIRROR-CARRIER 2026-07-24 by Codex] A valid request beyond this
        // carrier's retained producer tip is retryable availability, not a
        // malformed-frame response that would abort the bounded carrier list.
        let unavailable_response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-block-range")
                    .body(Body::from(replica_range_request_from_height(
                        &requester,
                        &producer_id,
                        3,
                        [0xcc; 16],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unavailable_response.status(), StatusCode::NOT_FOUND);
        let unavailable_body = to_bytes(unavailable_response.into_body(), 128)
            .await
            .unwrap();
        assert_eq!(&unavailable_body[..], b"replica_range_not_retained");

        let descriptor_hash = blocks[0].commitments[0].descriptor_hash;
        let object_id = [0xcd; 16];
        let object_timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let object_signing = directory_replica_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer_id,
            &[descriptor_hash],
            &object_id,
            &requester_id,
            object_timestamp,
        );
        let object_request = encode_directory_sync_message(
            &DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                producer: producer_id,
                descriptor_hashes: vec![descriptor_hash],
                request_id: object_id,
                requester: requester_id,
                request_timestamp: object_timestamp,
                signature: requester.sign(&object_signing),
            },
        )
        .unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-descriptor-objects")
                    .body(Body::from(object_request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        assert_eq!(
            verify_replica_descriptor_objects_response(
                &body,
                &object_id,
                &producer_id,
                &carrier.public_key_bytes(),
                &[descriptor_hash],
                object_timestamp,
            )
            .unwrap(),
            vec![expected_object]
        );
    }

    #[tokio::test]
    async fn verified_public_recovery_refuses_unregistered_or_disabled_namespaces() {
        let (unregistered_router, _, requester, producer, _) =
            carrier_test_router_with_access(CarrierTestPolicy::PublicWithoutMirror);
        let producer_id = producer.public_key_bytes();
        let response = unregistered_router
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-block-range")
                    .body(Body::from(replica_range_request(
                        &requester,
                        &producer_id,
                        [0xcf; 16],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let (disabled_router, _, requester, producer, _) =
            carrier_test_router_with_access(CarrierTestPolicy::PublicMirrorDisabled);
        let response = disabled_router
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-block-range")
                    .body(Body::from(replica_range_request(
                        &requester,
                        &producer.public_key_bytes(),
                        [0xd0; 16],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn carrier_routes_export_only_audited_producer_bound_evidence() {
        let (router, carrier, requester, producer, expected_object) = carrier_test_router();
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let producer_id = producer.public_key_bytes();
        let range_id = [0xd6; 16];
        let range_signing = directory_replica_block_range_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer_id,
            1,
            1,
            &range_id,
            &requester_id,
            timestamp,
        );
        let range_request =
            encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                producer: producer_id,
                from_height: 1,
                limit: 1,
                request_id: range_id,
                requester: requester_id,
                request_timestamp: timestamp,
                signature: requester.sign(&range_signing),
            })
            .unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-block-range")
                    .body(Body::from(range_request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let (blocks, has_more, tip_height, _) = verify_replica_block_range_response(
            &body,
            &range_id,
            &producer_id,
            &carrier.public_key_bytes(),
            1,
            timestamp,
        )
        .unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(!has_more);
        assert_eq!(tip_height, 1);
        let descriptor_hash = blocks[0].commitments[0].descriptor_hash;

        let object_id = [0xd7; 16];
        let object_signing = directory_replica_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer_id,
            &[descriptor_hash],
            &object_id,
            &requester_id,
            timestamp,
        );
        let object_request = encode_directory_sync_message(
            &DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                producer: producer_id,
                descriptor_hashes: vec![descriptor_hash],
                request_id: object_id,
                requester: requester_id,
                request_timestamp: timestamp,
                signature: requester.sign(&object_signing),
            },
        )
        .unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/replica-descriptor-objects")
                    .body(Body::from(object_request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let objects = verify_replica_descriptor_objects_response(
            &body,
            &object_id,
            &producer_id,
            &carrier.public_key_bytes(),
            &[descriptor_hash],
            timestamp,
        )
        .unwrap();
        assert_eq!(objects, vec![expected_object]);
    }

    #[test]
    fn transport_page_accepts_eight_unique_blocks_and_stops_before_duplicate_objects() {
        let now = now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let mut blocks = Vec::new();
        let mut previous_hash = [0u8; 32];
        for offset in 1u8..=8 {
            let subject = IdentityKeyPair::from_bytes(&[offset; 32]).unwrap();
            let object = signed_descriptor(&subject, now);
            let commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object).unwrap();
            let block = DirectoryCommitmentBlockV1::new_signed(
                u64::from(offset),
                now + u64::from(offset),
                previous_hash,
                vec![commitment],
                &producer,
            )
            .unwrap();
            previous_hash = block.hash();
            blocks.push(block);
        }
        assert_eq!(bounded_directory_transport_blocks(blocks.clone()).len(), 8);

        let repeated_commitment = blocks[0].commitments[0];
        let duplicate = DirectoryCommitmentBlockV1::new_signed(
            2,
            blocks[0].header.timestamp + 1,
            blocks[0].hash(),
            vec![repeated_commitment],
            &producer,
        )
        .unwrap();
        assert_eq!(
            bounded_directory_transport_blocks(vec![blocks[0].clone(), duplicate]).len(),
            1
        );
    }

    #[tokio::test]
    async fn witness_route_signs_unavailable_instead_of_trusting_observer() {
        let (router, witness, observer, _) = witness_test_router();
        let now = now_secs();
        let producer_a = IdentityKeyPair::from_bytes(&[0xc3; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xc4; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            1,
            now,
            [0u8; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 1,
                    tip_hash: [0xc5; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 1,
                    tip_hash: [0xc6; 32],
                },
            ],
            [0xc7; 32],
            &observer,
        )
        .unwrap();
        let request_id = [0xc8; 16];
        let checkpoint_hash = checkpoint.hash();
        let signing_bytes = directory_observation_witness_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            now,
            &checkpoint_hash,
        );
        let request = encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                requester: observer.public_key_bytes(),
                request_timestamp: now,
                checkpoint,
                signature: observer.sign(&signing_bytes),
            },
        )
        .unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/observation-checkpoint-witness")
                    .body(Body::from(request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let message = decode_directory_sync_message(&body).unwrap();
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            observer: response_observer,
            checkpoint_sequence,
            checkpoint_hash: response_checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
            ..
        } = message
        else {
            panic!("unexpected response");
        };
        assert_eq!(response_observer, observer.public_key_bytes());
        assert_eq!(checkpoint_sequence, 1);
        assert_eq!(response_checkpoint_hash, checkpoint_hash);
        assert_eq!(responder, witness.public_key_bytes());
        assert_eq!(
            outcome,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1
        );
        let response_signing = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &response_observer,
            checkpoint_sequence,
            &response_checkpoint_hash,
            &responder,
            response_timestamp,
            outcome,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&response_signing, &signature)
            .unwrap();
    }

    #[tokio::test]
    async fn witness_carrier_rejects_unpinned_target_before_transport() {
        let (router, observer, witness, runtime, calls) = witness_carrier_test_router(
            Err(WitnessCarrierTransportError::TargetUnavailable),
            false,
            true,
        );
        let request = witness_carrier_outer_request(
            &observer,
            &witness,
            witness_carrier_inner_request(&observer),
        );
        let response = router
            .oneshot(
                Request::post(
                    "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                )
                .body(Body::from(request))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_single_witness_carrier_outcome(
            runtime.observation_witness_carrier_snapshot(),
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected,
        );
    }

    #[tokio::test]
    async fn witness_carrier_handler_maps_each_transport_result_to_one_terminal_outcome() {
        let observer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xd9; 32]).unwrap();
        let inner_frame = witness_carrier_inner_request(&observer);
        let valid_response = witness_carrier_target_response(&observer, &witness, &inner_frame);
        let cases = vec![
            (
                "forwarded",
                Ok(WitnessCarrierTransportResponse {
                    status: 200,
                    body: valid_response,
                }),
                StatusCode::OK,
                DirectoryObservationWitnessCarrierOutcome::Forwarded,
            ),
            (
                "route_not_found",
                Ok(WitnessCarrierTransportResponse {
                    status: 404,
                    body: Vec::new(),
                }),
                StatusCode::FAILED_DEPENDENCY,
                DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable,
            ),
            (
                "target_rate_limited",
                Ok(WitnessCarrierTransportResponse {
                    status: 429,
                    body: Vec::new(),
                }),
                StatusCode::SERVICE_UNAVAILABLE,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            ),
            (
                "target_server_failure",
                Ok(WitnessCarrierTransportResponse {
                    status: 503,
                    body: Vec::new(),
                }),
                StatusCode::SERVICE_UNAVAILABLE,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            ),
            (
                "target_rejected",
                Ok(WitnessCarrierTransportResponse {
                    status: 403,
                    body: Vec::new(),
                }),
                StatusCode::BAD_GATEWAY,
                DirectoryObservationWitnessCarrierOutcome::TargetRejected,
            ),
            (
                "malformed_success_body",
                Ok(WitnessCarrierTransportResponse {
                    status: 200,
                    body: vec![0xff],
                }),
                StatusCode::BAD_GATEWAY,
                DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
            ),
            (
                "response_body_too_large",
                Err(WitnessCarrierTransportError::ResponseTooLarge),
                StatusCode::BAD_GATEWAY,
                DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
            ),
            (
                "target_timeout_or_stream_failure",
                Err(WitnessCarrierTransportError::TargetUnavailable),
                StatusCode::SERVICE_UNAVAILABLE,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            ),
            (
                "local_client_unavailable",
                Err(WitnessCarrierTransportError::LocalUnavailable),
                StatusCode::SERVICE_UNAVAILABLE,
                DirectoryObservationWitnessCarrierOutcome::LocalFailure,
            ),
        ];

        for (name, transport_result, expected_status, expected_outcome) in cases {
            let (router, observer, witness, runtime, calls) =
                witness_carrier_test_router(transport_result, true, true);
            let request = witness_carrier_outer_request(&observer, &witness, inner_frame.clone());
            let response = router
                .oneshot(
                    Request::post(
                        "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                    )
                    .body(Body::from(request))
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status, "{name}");
            assert_eq!(calls.load(Ordering::Relaxed), 1, "{name}");
            assert_single_witness_carrier_outcome(
                runtime.observation_witness_carrier_snapshot(),
                expected_outcome,
            );
        }
    }

    #[tokio::test]
    async fn witness_carrier_handler_fails_closed_before_transport_for_inner_and_descriptor_errors()
    {
        let cases = [
            (
                "invalid_inner_frame",
                true,
                vec![0xff],
                StatusCode::BAD_REQUEST,
                DirectoryObservationWitnessCarrierOutcome::InvalidRequest,
            ),
            (
                "missing_target_descriptor",
                false,
                Vec::new(),
                StatusCode::SERVICE_UNAVAILABLE,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            ),
        ];

        for (name, advertise_target, invalid_inner, expected_status, expected_outcome) in cases {
            let (router, observer, witness, runtime, calls) = witness_carrier_test_router(
                Err(WitnessCarrierTransportError::TargetUnavailable),
                true,
                advertise_target,
            );
            let inner_frame = if invalid_inner.is_empty() {
                witness_carrier_inner_request(&observer)
            } else {
                invalid_inner
            };
            let request = witness_carrier_outer_request(&observer, &witness, inner_frame);
            let response = router
                .oneshot(
                    Request::post(
                        "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                    )
                    .body(Body::from(request))
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status, "{name}");
            assert_eq!(calls.load(Ordering::Relaxed), 0, "{name}");
            assert_single_witness_carrier_outcome(
                runtime.observation_witness_carrier_snapshot(),
                expected_outcome,
            );
        }
    }

    #[test]
    fn carried_witness_response_requires_exact_target_signature() {
        let observer = IdentityKeyPair::from_bytes(&[0xdf; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xe0; 32]).unwrap();
        let other = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xe3; 32]).unwrap();
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
                    tip_hash: [0xe4; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 1,
                    tip_hash: [0xe5; 32],
                },
            ],
            [0xe6; 32],
            &observer,
        )
        .unwrap();
        let request_id = [0xe7; 16];
        let checkpoint_hash = checkpoint.hash();
        let request_signing = directory_observation_witness_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            now,
            &checkpoint_hash,
        );
        let request_frame = encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                requester: observer.public_key_bytes(),
                request_timestamp: now,
                checkpoint,
                signature: observer.sign(&request_signing),
            },
        )
        .unwrap();
        let context = verify_carried_observation_witness_request(
            &request_frame,
            &observer.public_key_bytes(),
            now,
        )
        .unwrap();
        let response_signing = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            1,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            now,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response_frame = encode_directory_sync_message(
            &DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                observer: observer.public_key_bytes(),
                checkpoint_sequence: 1,
                checkpoint_hash,
                responder: witness.public_key_bytes(),
                response_timestamp: now,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&response_signing),
            },
        )
        .unwrap();
        assert!(verify_carried_observation_witness_response(
            &response_frame,
            &context,
            &witness.public_key_bytes(),
            now
        )
        .is_ok());
        assert_eq!(
            verify_carried_observation_witness_response(
                &response_frame,
                &context,
                &other.public_key_bytes(),
                now
            ),
            Err("carried_witness_response_contract_mismatch")
        );
    }

    #[tokio::test]
    async fn observation_certificate_route_is_pinned_and_fails_closed_without_evidence() {
        let (router, _, observer, _) = witness_test_router();
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/observation-certificate")
                    .body(Body::from(observation_certificate_request(
                        &observer, [0xca; 16],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let unpinned = IdentityKeyPair::from_bytes(&[0xcb; 32]).unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/observation-certificate")
                    .body(Body::from(observation_certificate_request(
                        &unpinned, [0xcc; 16],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn observation_certificate_route_serves_exact_current_policy_evidence() {
        let (router, observer, requester, expected_sequence) = positive_certificate_test_router();
        let request_id = [0xdd; 16];
        let request_timestamp = now_secs();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/observation-certificate")
                    .body(Body::from(observation_certificate_request(
                        &requester, request_id,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let observed_at = now_secs();
        let authenticated = verify_observation_certificate_response(
            &body,
            &request_id,
            &requester.public_key_bytes(),
            &observer.public_key_bytes(),
            request_timestamp,
            observed_at,
        )
        .unwrap();
        let certificate = decode_directory_observation_certificate(&authenticated.frame).unwrap();
        certificate
            .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .unwrap();
        assert_eq!(certificate.checkpoint.observer, observer.public_key_bytes());
        assert_eq!(certificate.checkpoint.sequence, expected_sequence);
        assert_eq!(certificate.minimum_witnesses, 2);
        assert_eq!(certificate.receipts.len(), 2);
    }
}
