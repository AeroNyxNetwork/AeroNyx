// ============================================================================
// File: crates/aeronyx-server/src/api/public_node_router.rs
// ============================================================================
//! Public node-listener composition and descriptor-authenticated endpoint proof.
//!
//! This module is the only composition boundary that may turn an ADET V2
//! request into [`VerifiedEndpointProofPeerContext`]. It has no promotion
//! authority and never consults the mutable peer store for candidate identity.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use aeronyx_core::protocol::discovery_endpoint_proof::{
    canonical_public_endpoint_commitment, DiscoveryEndpointAuthenticatedTransportV2,
    DiscoveryEndpointChallengeV1, DiscoveryEndpointTransportOperationV1,
    DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V2,
};
use aeronyx_transport::UdpTransport;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tower::ServiceExt;

use crate::api::blind_vault::{
    build_blind_vault_router_with_admission_runtime, BlindVaultApiAdmissionRuntime,
};
use crate::api::chat_peer::build_chat_peer_router_with_anonymous_mailbox;
use crate::api::directory_chain_peer::build_directory_chain_peer_router_with_replica_and_runtime;
use crate::api::directory_replica_status::{
    build_directory_replica_status_router_with_witness_carrier, DirectoryReplicaStatusScope,
};
use crate::api::discovery::{
    build_discovery_router_with_local_entry_and_attestation_inbox, DiscoveryApiPolicy,
    DiscoveryLocalCapabilityStatus,
};
use crate::api::discovery_endpoint_verification::{
    build_discovery_endpoint_verification_router_with_evidence, VerifiedEndpointProofPeerContext,
};
use crate::api::memchain_peer::build_memchain_peer_router_with_runtime;
use crate::services::chat_relay::ChatRelayService;
use crate::services::chat_relay_mailbox::AnonymousMailboxCustodyRepository;
use crate::services::memchain::MemoryStorage;
use crate::services::{
    BlindVaultService, DirectoryChainStore, DirectoryReplicaStore, DirectoryReplicaSyncRuntime,
    DiscoveryEndpointVerificationConfig, DiscoveryEndpointVerificationService, PeerStore,
    SessionManager, SqliteDiscoveryEndpointAttestationInbox, SqliteDiscoveryEndpointEvidenceStore,
};

const ADEA_MAGIC: [u8; 4] = *b"ADEA";
const ADEA_VERSION_V1: u8 = 1;
const ADEA_HEADER_BYTES: usize = 8;
const ADEA_ISSUE_FIXED_BODY_BYTES: usize = 32 + 32 + 1;
const PUBLIC_ENDPOINT_FLOW_CONTEXT_DOMAIN: &[u8] = b"AeroNyx/PublicEndpointProofFlowContextV1\0";

// [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-24 by Codex] This module owns the
// sole V2-to-verified-context transition. It intentionally has no PeerStore
// lookup or descriptor-promotion capability.

/// Immutable dependencies for the Internet-facing public node router.
pub(crate) struct PublicNodeRouterDependencies {
    pub(crate) peer_store: Arc<PeerStore>,
    pub(crate) discovery_api_policy: DiscoveryApiPolicy,
    pub(crate) chat_relay: Option<Arc<ChatRelayService>>,
    pub(crate) sessions: Arc<SessionManager>,
    pub(crate) udp: Arc<UdpTransport>,
    pub(crate) node_identity: Arc<IdentityKeyPair>,
    pub(crate) peer_http_client: Arc<reqwest::Client>,
    pub(crate) local_capability_status: DiscoveryLocalCapabilityStatus,
    pub(crate) directory_chain_store: Option<Arc<DirectoryChainStore>>,
    pub(crate) directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    pub(crate) directory_replica_sync_runtime: Arc<DirectoryReplicaSyncRuntime>,
    pub(crate) directory_chain_sync_peer_ids: Vec<[u8; 32]>,
    pub(crate) directory_observation_witness_min_verified: usize,
    pub(crate) directory_observation_witness_maturity_delay_secs: u64,
    pub(crate) directory_full_node_mirror_enabled: bool,
    pub(crate) directory_full_node_mirror_max_producers: usize,
    pub(crate) commitment_storage: Option<Arc<MemoryStorage>>,
    pub(crate) commitment_lease_authorized_coordinator: Option<[u8; 32]>,
    pub(crate) commitment_sync_tip_notifier: Option<mpsc::Sender<u64>>,
    pub(crate) blind_vault: Option<Arc<BlindVaultService>>,
    pub(crate) blind_vault_public_api_enabled: bool,
    pub(crate) blind_vault_admission: Arc<BlindVaultApiAdmissionRuntime>,
    pub(crate) anonymous_mailbox: Option<Arc<dyn AnonymousMailboxCustodyRepository>>,
    pub(crate) endpoint_proof_enabled: bool,
    pub(crate) endpoint_proof_max_entries: usize,
    pub(crate) endpoint_proof_ttl_secs: u64,
    pub(crate) endpoint_evidence: Option<Arc<SqliteDiscoveryEndpointEvidenceStore>>,
    /// Shared durable quarantine; it grants no routing or promotion authority.
    pub(crate) endpoint_attestation_inbox: Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>,
}

/// Builds the complete public node router.
///
/// Endpoint proof transport is absent when its independent rollout gate is
/// false, so no verifier service or replay state is allocated in that mode.
pub(crate) fn build_public_node_router(deps: PublicNodeRouterDependencies) -> Router {
    let block_peer_store = Arc::clone(&deps.peer_store);
    let block_identity = Arc::clone(&deps.node_identity);
    let directory_peer_store = Arc::clone(&deps.peer_store);
    let directory_identity = Arc::clone(&deps.node_identity);
    let witness_carrier_route_enabled =
        deps.directory_chain_store.is_some() && deps.directory_replica_store.is_some();
    let blind_vault_for_onion = deps
        .blind_vault_public_api_enabled
        .then(|| deps.blind_vault.clone())
        .flatten();

    let mut app = build_discovery_router_with_local_entry_and_attestation_inbox(
        Arc::clone(&deps.peer_store),
        deps.discovery_api_policy,
        deps.local_capability_status,
        deps.directory_replica_store.clone(),
        deps.node_identity.public_key_bytes(),
        deps.endpoint_attestation_inbox,
    )
    .merge(build_chat_peer_router_with_anonymous_mailbox(
        deps.chat_relay,
        deps.sessions,
        deps.udp,
        Arc::clone(&deps.peer_store),
        Arc::clone(&deps.node_identity),
        deps.peer_http_client,
        blind_vault_for_onion,
        deps.anonymous_mailbox,
    ))
    .merge(build_directory_replica_status_router_with_witness_carrier(
        deps.directory_replica_store.clone(),
        Arc::clone(&deps.directory_replica_sync_runtime),
        deps.directory_chain_sync_peer_ids.clone(),
        deps.directory_observation_witness_min_verified,
        deps.directory_observation_witness_maturity_delay_secs,
        deps.directory_full_node_mirror_enabled,
        deps.directory_full_node_mirror_max_producers,
        witness_carrier_route_enabled,
        DirectoryReplicaStatusScope::PublicAggregate,
    ));

    if let Some(endpoint_proof_router) = build_optional_public_endpoint_transport_router(
        deps.endpoint_proof_enabled,
        deps.node_identity.as_ref().clone(),
        deps.endpoint_proof_max_entries,
        deps.endpoint_proof_ttl_secs,
        deps.endpoint_evidence,
    ) {
        app = app.merge(endpoint_proof_router);
    }

    if let Some(store) = deps.directory_chain_store {
        app = app.merge(build_directory_chain_peer_router_with_replica_and_runtime(
            store,
            deps.directory_replica_store,
            directory_peer_store,
            directory_identity,
            deps.directory_chain_sync_peer_ids,
            deps.directory_full_node_mirror_enabled,
            deps.directory_replica_sync_runtime,
        ));
    }
    if let (true, Some(vault)) = (deps.blind_vault_public_api_enabled, deps.blind_vault) {
        app = app.merge(build_blind_vault_router_with_admission_runtime(
            vault,
            deps.node_identity,
            deps.blind_vault_admission,
        ));
    }
    if let Some(storage) = deps.commitment_storage {
        app = app.merge(build_memchain_peer_router_with_runtime(
            storage,
            block_peer_store,
            block_identity,
            deps.commitment_lease_authorized_coordinator,
            deps.commitment_sync_tip_notifier,
        ));
    }
    app
}

fn build_optional_public_endpoint_transport_router(
    enabled: bool,
    challenger_identity: IdentityKeyPair,
    max_entries: usize,
    challenge_ttl_secs: u64,
    evidence: Option<Arc<SqliteDiscoveryEndpointEvidenceStore>>,
) -> Option<Router> {
    if !enabled {
        return None;
    }
    let challenger_node_id = challenger_identity.public_key_bytes();
    let service = DiscoveryEndpointVerificationService::new(
        challenger_identity,
        DiscoveryEndpointVerificationConfig {
            max_entries,
            challenge_ttl_secs,
        },
    )
    .ok()?;
    Some(build_public_endpoint_transport_router(
        Arc::new(service),
        challenger_node_id,
        evidence,
    ))
}

#[derive(Clone)]
struct PublicEndpointTransportState {
    stage_c: Router,
    flow_context: [u8; 32],
}

fn build_public_endpoint_transport_router(
    service: Arc<DiscoveryEndpointVerificationService>,
    challenger_node_id: [u8; 32],
    evidence: Option<Arc<SqliteDiscoveryEndpointEvidenceStore>>,
) -> Router {
    let flow_context = public_endpoint_flow_context(challenger_node_id);
    let stage_c = build_discovery_endpoint_verification_router_with_evidence(service, evidence);
    Router::new()
        .route(
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            post(issue_endpoint_proof),
        )
        .route(
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            post(verify_endpoint_proof),
        )
        .layer(DefaultBodyLimit::max(
            DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V2,
        ))
        .with_state(PublicEndpointTransportState {
            stage_c,
            flow_context,
        })
}

async fn issue_endpoint_proof(
    State(state): State<PublicEndpointTransportState>,
    body: Bytes,
) -> Response {
    dispatch_authenticated_transport(state, DiscoveryEndpointTransportOperationV1::Issue, body)
        .await
}

async fn verify_endpoint_proof(
    State(state): State<PublicEndpointTransportState>,
    body: Bytes,
) -> Response {
    dispatch_authenticated_transport(state, DiscoveryEndpointTransportOperationV1::Verify, body)
        .await
}

async fn dispatch_authenticated_transport(
    state: PublicEndpointTransportState,
    operation: DiscoveryEndpointTransportOperationV1,
    body: Bytes,
) -> Response {
    let now = unix_now_secs();
    let Ok(transport) = DiscoveryEndpointAuthenticatedTransportV2::decode(&body) else {
        return coarse_rejection();
    };
    if transport
        .verify_at(
            now,
            operation,
            &transport.target_node_id(),
            &state.flow_context,
        )
        .is_err()
        || validate_inner_binding(&transport, operation).is_err()
    {
        return coarse_rejection();
    }
    let Some(mut context) = VerifiedEndpointProofPeerContext::from_authenticated_peer(
        transport.target_node_id(),
        state.flow_context,
    ) else {
        return coarse_rejection();
    };
    if operation == DiscoveryEndpointTransportOperationV1::Verify {
        // [PERMISSIONLESS-ENDPOINT-EVIDENCE 2026-09-24 by Codex] Preserve the
        // canonical authenticated bytes only for the private persistence boundary.
        context = context.with_authenticated_transport_frame(Arc::from(body.as_ref()));
    }
    let Ok(request) = Request::builder()
        .method("POST")
        .uri(operation.path())
        .body(Body::from(transport.inner_frame().to_vec()))
    else {
        return coarse_rejection();
    };
    state
        .stage_c
        .clone()
        .layer(Extension(context))
        .oneshot(request)
        .await
        .unwrap_or_else(|_| coarse_unavailable())
}

fn validate_inner_binding(
    transport: &DiscoveryEndpointAuthenticatedTransportV2,
    operation: DiscoveryEndpointTransportOperationV1,
) -> Result<(), ()> {
    let inner = transport.inner_frame();
    if inner.len() < ADEA_HEADER_BYTES + 32
        || inner[..4] != ADEA_MAGIC
        || inner[4] != ADEA_VERSION_V1
        || inner[5] != operation as u8
        || usize::from(u16::from_be_bytes([inner[6], inner[7]])) + ADEA_HEADER_BYTES != inner.len()
    {
        return Err(());
    }
    match operation {
        DiscoveryEndpointTransportOperationV1::Issue => validate_issue_binding(transport, inner),
        DiscoveryEndpointTransportOperationV1::Verify => validate_verify_binding(transport, inner),
    }
}

fn validate_issue_binding(
    transport: &DiscoveryEndpointAuthenticatedTransportV2,
    inner: &[u8],
) -> Result<(), ()> {
    let body = &inner[ADEA_HEADER_BYTES..];
    if body.len() < ADEA_ISSUE_FIXED_BODY_BYTES {
        return Err(());
    }
    let request_id: [u8; 32] = body[..32].try_into().map_err(|_| ())?;
    let descriptor_commitment: [u8; 32] = body[32..64].try_into().map_err(|_| ())?;
    let endpoint_len = usize::from(body[64]);
    if endpoint_len == 0 || body.len() != ADEA_ISSUE_FIXED_BODY_BYTES + endpoint_len {
        return Err(());
    }
    let endpoint = std::str::from_utf8(&body[65..]).map_err(|_| ())?;
    let endpoint_commitment = canonical_public_endpoint_commitment(endpoint).map_err(|_| ())?;
    if request_id != transport.request_id()
        || descriptor_commitment != transport.descriptor_commitment()
        || endpoint_commitment != transport.endpoint_commitment()
    {
        return Err(());
    }
    let descriptor =
        SignedNodeDescriptor::decode_canonical(transport.descriptor_bytes()).map_err(|_| ())?;
    if descriptor.descriptor.public_endpoint.as_deref() != Some(endpoint) {
        return Err(());
    }
    Ok(())
}

fn validate_verify_binding(
    transport: &DiscoveryEndpointAuthenticatedTransportV2,
    inner: &[u8],
) -> Result<(), ()> {
    let body = &inner[ADEA_HEADER_BYTES..];
    if body.len() < 34 {
        return Err(());
    }
    let request_id: [u8; 32] = body[..32].try_into().map_err(|_| ())?;
    let challenge_len = usize::from(u16::from_be_bytes([body[32], body[33]]));
    let challenge_end = 34usize.checked_add(challenge_len).ok_or(())?;
    let challenge = DiscoveryEndpointChallengeV1::decode(body.get(34..challenge_end).ok_or(())?)
        .map_err(|_| ())?;
    if request_id != transport.request_id()
        || challenge.target_node_id() != transport.target_node_id()
        || challenge.descriptor_commitment() != transport.descriptor_commitment()
        || challenge.endpoint_commitment() != transport.endpoint_commitment()
        || challenge.challenger_context() != transport.flow_context()
    {
        return Err(());
    }
    Ok(())
}

pub(crate) fn public_endpoint_flow_context(challenger_node_id: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PUBLIC_ENDPOINT_FLOW_CONTEXT_DOMAIN);
    hasher.update(challenger_node_id);
    hasher.finalize().into()
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn coarse_rejection() -> Response {
    (StatusCode::BAD_REQUEST, "request_rejected").into_response()
}

fn coarse_unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable").into_response()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    use crate::services::DiscoveryEndpointEvidenceStoreConfig;
    use aeronyx_core::ledger::{AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH};
    use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, NodeDescriptor};
    use aeronyx_core::protocol::discovery::{NodeCapability, NodeDiscoveryMessage};
    use aeronyx_core::protocol::discovery_endpoint_proof::{
        DiscoveryEndpointAuthenticatedTransportV1, DiscoveryEndpointProofV1,
    };
    use aeronyx_core::protocol::memchain::{
        encode_memchain, record_coordinator_lease_request_signing_bytes, MemChainMessage,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
    };
    use axum::body::to_bytes;
    use axum::http::header;
    use std::time::Duration;

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed key")
    }

    fn admit_control_peer(peer_store: &PeerStore, identity: &IdentityKeyPair, now: u64) {
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "public-router-control-test",
        );
        descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
        let descriptor = SignedNodeDescriptor::sign(descriptor, identity).expect("descriptor");
        let outcome = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(outcome.inserted, 1);
    }

    fn coordinator_lease_frame(
        coordinator: &IdentityKeyPair,
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let instance_id = [0x41; 32];
        let coordinator_id = coordinator.public_key_bytes();
        let signing_bytes = record_coordinator_lease_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &coordinator_id,
            &instance_id,
            0,
            &GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorLeaseRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            coordinator: coordinator_id,
            instance_id,
            known_tip_height: 0,
            known_tip_hash: GENESIS_PREV_HASH,
            requested_ttl_secs: MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&signing_bytes),
        })
        .expect("lease frame")
    }

    fn descriptor(
        target: &IdentityKeyPair,
        sequence: u64,
        now: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            sequence,
            now.saturating_sub(10),
            now + 600,
            "1.0.0",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        SignedNodeDescriptor::sign(descriptor, target).expect("signed descriptor")
    }

    fn descriptor_commitment(descriptor: &SignedNodeDescriptor) -> [u8; 32] {
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .expect("descriptor commitment")
            .descriptor_hash
    }

    fn issue_inner(
        request_id: [u8; 32],
        descriptor_commitment: [u8; 32],
        endpoint: &str,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&request_id);
        body.extend_from_slice(&descriptor_commitment);
        body.push(u8::try_from(endpoint.len()).expect("bounded endpoint"));
        body.extend_from_slice(endpoint.as_bytes());
        adapter_frame(DiscoveryEndpointTransportOperationV1::Issue as u8, &body)
    }

    fn verify_inner(request_id: [u8; 32], challenge: &[u8], proof: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&request_id);
        body.extend_from_slice(
            &u16::try_from(challenge.len())
                .expect("challenge length")
                .to_be_bytes(),
        );
        body.extend_from_slice(challenge);
        body.extend_from_slice(
            &u16::try_from(proof.len())
                .expect("proof length")
                .to_be_bytes(),
        );
        body.extend_from_slice(proof);
        adapter_frame(DiscoveryEndpointTransportOperationV1::Verify as u8, &body)
    }

    fn adapter_frame(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&ADEA_MAGIC);
        frame.push(ADEA_VERSION_V1);
        frame.push(kind);
        frame.extend_from_slice(
            &u16::try_from(body.len())
                .expect("bounded adapter body")
                .to_be_bytes(),
        );
        frame.extend_from_slice(body);
        frame
    }

    fn v2_transport(
        operation: DiscoveryEndpointTransportOperationV1,
        request_id: [u8; 32],
        flow_context: [u8; 32],
        now: u64,
        inner: &[u8],
        descriptor: &SignedNodeDescriptor,
        target: &IdentityKeyPair,
    ) -> Vec<u8> {
        DiscoveryEndpointAuthenticatedTransportV2::sign(
            operation,
            request_id,
            flow_context,
            now,
            now + 60,
            inner,
            descriptor,
            target,
        )
        .expect("V2 transport")
        .encode()
    }

    fn enabled_router(challenger: &IdentityKeyPair, max_entries: usize) -> Router {
        build_optional_public_endpoint_transport_router(
            true,
            challenger.clone(),
            max_entries,
            60,
            None,
        )
        .expect("enabled router")
    }

    fn enabled_router_with_evidence(
        challenger: &IdentityKeyPair,
        max_entries: usize,
        evidence: Arc<SqliteDiscoveryEndpointEvidenceStore>,
    ) -> Router {
        build_optional_public_endpoint_transport_router(
            true,
            challenger.clone(),
            max_entries,
            60,
            Some(evidence),
        )
        .expect("enabled router")
    }

    fn evidence_store(
        path: &std::path::Path,
        challenger: &IdentityKeyPair,
        max_entries: usize,
    ) -> Arc<SqliteDiscoveryEndpointEvidenceStore> {
        Arc::new(
            SqliteDiscoveryEndpointEvidenceStore::open(
                DiscoveryEndpointEvidenceStoreConfig {
                    db_path: path.to_path_buf(),
                    max_entries,
                    retention_ttl_secs: 3600,
                    cleanup_batch_size: 16,
                },
                challenger.public_key_bytes(),
                public_endpoint_flow_context(challenger.public_key_bytes()),
            )
            .expect("evidence store"),
        )
    }

    async fn post(router: Router, path: &str, body: Vec<u8>) -> (StatusCode, Vec<u8>) {
        let response = router
            .oneshot(Request::post(path).body(Body::from(body)).expect("request"))
            .await
            .expect("response");
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body")
            .to_vec();
        (status, body)
    }

    fn challenge_from_issue_response(response: &[u8]) -> Vec<u8> {
        assert_eq!(&response[..4], &ADEA_MAGIC);
        assert_eq!(response[4], ADEA_VERSION_V1);
        assert_eq!(response[5], 129);
        assert!(matches!(response[8], 1 | 2));
        let len = usize::from(u16::from_be_bytes([response[9], response[10]]));
        assert_eq!(response.len(), 11 + len);
        response[11..].to_vec()
    }

    #[tokio::test]
    async fn disabled_gate_has_no_router_or_verifier_state() {
        let challenger = key(3);
        assert!(
            build_optional_public_endpoint_transport_router(false, challenger, 0, 0, None)
                .is_none()
        );
        let (status, _) = post(
            Router::new(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            vec![0; 8],
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn endpoint_transport_composition_is_public_listener_only() {
        let server = include_str!("../server.rs");
        assert_eq!(server.matches("build_public_node_router(").count(), 1);
        let public_branch = server
            .find("if let Some((public_addr, public_listener)) = public_api_listener")
            .expect("public listener branch");
        let builder = server
            .find("build_public_node_router(")
            .expect("public router builder");
        let local_branch = server
            .find("let chat_blob_router =")
            .expect("local listener composition");
        assert!(public_branch < builder && builder < local_branch);
    }

    #[tokio::test]
    async fn public_router_control_route_auth_is_coarse_and_rejects_without_mutation() {
        // [CONTROL-ROUTE-BOUNDARY-TEST 2026-09-25 by Codex] Exercise the
        // actual public composition root, not only the peer router. An
        // authenticated but unknown coordinator must be rejected before the
        // lease write; the same exact signed frame succeeds after admission.
        let now = unix_now_secs();
        let coordinator = key(0x31);
        let witness = Arc::new(key(0x32));
        let peer_store = Arc::new(PeerStore::new());
        let storage = Arc::new(MemoryStorage::open(":memory:", None).expect("storage"));
        storage
            .audit_record_commitment_chain()
            .await
            .expect("genesis audit");
        let udp = Arc::new(
            UdpTransport::bind("127.0.0.1:0")
                .await
                .expect("loopback udp"),
        );
        let deps = PublicNodeRouterDependencies {
            peer_store: Arc::clone(&peer_store),
            discovery_api_policy: DiscoveryApiPolicy::default(),
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(8, Duration::from_secs(30))),
            udp,
            node_identity: Arc::clone(&witness),
            peer_http_client: Arc::new(reqwest::Client::new()),
            local_capability_status: DiscoveryLocalCapabilityStatus::default(),
            directory_chain_store: None,
            directory_replica_store: None,
            directory_replica_sync_runtime: Arc::new(DirectoryReplicaSyncRuntime::default()),
            directory_chain_sync_peer_ids: Vec::new(),
            directory_observation_witness_min_verified: 0,
            directory_observation_witness_maturity_delay_secs: 0,
            directory_full_node_mirror_enabled: false,
            directory_full_node_mirror_max_producers: 0,
            commitment_storage: Some(Arc::clone(&storage)),
            commitment_lease_authorized_coordinator: Some(coordinator.public_key_bytes()),
            commitment_sync_tip_notifier: None,
            blind_vault: None,
            blind_vault_public_api_enabled: false,
            blind_vault_admission: Arc::new(BlindVaultApiAdmissionRuntime::default()),
            anonymous_mailbox: None,
            endpoint_proof_enabled: false,
            endpoint_proof_max_entries: 1,
            endpoint_proof_ttl_secs: 1,
            endpoint_evidence: None,
            endpoint_attestation_inbox: None,
        };
        let router = build_public_node_router(deps);
        let malformed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(vec![0u8; 1]))
                    .expect("malformed request"),
            )
            .await
            .expect("malformed response");
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        let malformed_body = to_bytes(malformed.into_body(), 1024)
            .await
            .expect("malformed coarse body");
        assert!(malformed_body.len() <= 128);

        let frame = coordinator_lease_frame(&coordinator, [0x51; 16], now);
        let rejected = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame.clone()))
                    .expect("unknown coordinator request"),
            )
            .await
            .expect("unknown coordinator response");
        assert_eq!(rejected.status(), StatusCode::FORBIDDEN);
        let rejected_body = to_bytes(rejected.into_body(), 1024)
            .await
            .expect("coarse body");
        assert!(rejected_body.len() <= 128);
        assert!(!String::from_utf8_lossy(&rejected_body).contains("31"));

        // The exact signed request must still be usable after the peer is
        // admitted, proving the rejection above did not consume lease state.
        admit_control_peer(&peer_store, &coordinator, now);
        let admitted = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .expect("admitted coordinator request"),
            )
            .await
            .expect("admitted coordinator response");
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn v2_issue_verify_and_exact_replay_succeed() {
        let now = unix_now_secs();
        let challenger = key(3);
        let target = key(9);
        let endpoint = "8.8.8.8:51820";
        let signed_descriptor = descriptor(&target, 7, now, endpoint);
        let request_id = [0x11; 32];
        let flow_context = public_endpoint_flow_context(challenger.public_key_bytes());
        let inner = issue_inner(
            request_id,
            descriptor_commitment(&signed_descriptor),
            endpoint,
        );
        let transport = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            flow_context,
            now,
            &inner,
            &signed_descriptor,
            &target,
        );
        let router = enabled_router(&challenger, 4);
        let (status, issued) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            transport.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let challenge_frame = challenge_from_issue_response(&issued);

        let (_, replayed) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            transport,
        )
        .await;
        assert_eq!(challenge_from_issue_response(&replayed), challenge_frame);

        let challenge = DiscoveryEndpointChallengeV1::decode(&challenge_frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &flow_context,
            challenge.issued_at().saturating_add(1),
            &target,
        )
        .expect("proof")
        .encode();
        let verify_inner = verify_inner(request_id, &challenge_frame, &proof);
        let verify_transport = v2_transport(
            DiscoveryEndpointTransportOperationV1::Verify,
            request_id,
            flow_context,
            now,
            &verify_inner,
            &signed_descriptor,
            &target,
        );
        let (verify_status, response) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            verify_transport.clone(),
        )
        .await;
        assert_eq!(verify_status, StatusCode::OK);
        assert_eq!(response[8], 1);
        let (_, replay_response) = post(
            router,
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            verify_transport,
        )
        .await;
        assert_eq!(replay_response[8], 2);
    }

    #[tokio::test]
    async fn accepted_v2_proof_is_durable_and_exact_replay_is_idempotent() {
        std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
        let directory =
            tempfile::TempDir::new_in("target/test-temp").expect("private evidence directory");
        let path = directory.path().join("endpoint-evidence.sqlite3");
        let now = unix_now_secs();
        let challenger = key(3);
        let target = key(9);
        let endpoint = "8.8.8.8:51820";
        let signed_descriptor = descriptor(&target, 7, now, endpoint);
        let request_id = [0x31; 32];
        let flow_context = public_endpoint_flow_context(challenger.public_key_bytes());
        let inner = issue_inner(
            request_id,
            descriptor_commitment(&signed_descriptor),
            endpoint,
        );
        let store = evidence_store(&path, &challenger, 1);
        let router = enabled_router_with_evidence(&challenger, 4, Arc::clone(&store));
        let (_, issued) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            v2_transport(
                DiscoveryEndpointTransportOperationV1::Issue,
                request_id,
                flow_context,
                now,
                &inner,
                &signed_descriptor,
                &target,
            ),
        )
        .await;
        let challenge_frame = challenge_from_issue_response(&issued);
        let challenge = DiscoveryEndpointChallengeV1::decode(&challenge_frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &flow_context,
            challenge.issued_at().saturating_add(1),
            &target,
        )
        .expect("proof")
        .encode();
        let verify = v2_transport(
            DiscoveryEndpointTransportOperationV1::Verify,
            request_id,
            flow_context,
            now,
            &verify_inner(request_id, &challenge_frame, &proof),
            &signed_descriptor,
            &target,
        );
        let mut tampered = verify.clone();
        let final_byte = tampered.last_mut().expect("transport signature byte");
        *final_byte ^= 0x01;
        let (rejected, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            tampered,
        )
        .await;
        assert_eq!(rejected, StatusCode::BAD_REQUEST);
        assert_eq!(store.snapshot().expect("rejected snapshot").retained, 0);
        let (accepted, accepted_body) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            verify.clone(),
        )
        .await;
        assert_eq!(accepted, StatusCode::OK);
        assert_eq!(accepted_body[8], 1);
        let (existing, existing_body) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            verify,
        )
        .await;
        assert_eq!(existing, StatusCode::OK);
        assert_eq!(existing_body[8], 2);
        assert_eq!(store.snapshot().expect("snapshot").retained, 1);

        let second_target = key(10);
        let second_request_id = [0x32; 32];
        let second_descriptor = descriptor(&second_target, 8, now, "8.8.4.4:51820");
        let second_inner = issue_inner(
            second_request_id,
            descriptor_commitment(&second_descriptor),
            "8.8.4.4:51820",
        );
        let (_, second_issued) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            v2_transport(
                DiscoveryEndpointTransportOperationV1::Issue,
                second_request_id,
                flow_context,
                now,
                &second_inner,
                &second_descriptor,
                &second_target,
            ),
        )
        .await;
        let second_challenge_frame = challenge_from_issue_response(&second_issued);
        let second_challenge = DiscoveryEndpointChallengeV1::decode(&second_challenge_frame)
            .expect("second challenge");
        let second_proof = DiscoveryEndpointProofV1::respond(
            &second_challenge,
            &flow_context,
            second_challenge.issued_at().saturating_add(1),
            &second_target,
        )
        .expect("second proof")
        .encode();
        let second_verify = v2_transport(
            DiscoveryEndpointTransportOperationV1::Verify,
            second_request_id,
            flow_context,
            now,
            &verify_inner(second_request_id, &second_challenge_frame, &second_proof),
            &second_descriptor,
            &second_target,
        );
        let (at_capacity, at_capacity_body) = post(
            router,
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            second_verify,
        )
        .await;
        assert_eq!(at_capacity, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(at_capacity_body, b"service_unavailable");
        assert_eq!(store.snapshot().expect("capacity snapshot").retained, 1);
        drop(store);

        let reopened = evidence_store(&path, &challenger, 1);
        assert_eq!(reopened.snapshot().expect("restart snapshot").retained, 1);
    }

    #[tokio::test]
    async fn v1_and_wrong_context_time_path_or_inner_binding_fail_closed() {
        let now = unix_now_secs();
        let challenger = key(3);
        let target = key(9);
        let endpoint = "8.8.8.8:51820";
        let signed_descriptor = descriptor(&target, 7, now, endpoint);
        let descriptor_hash = descriptor_commitment(&signed_descriptor);
        let request_id = [0x21; 32];
        let flow_context = public_endpoint_flow_context(challenger.public_key_bytes());
        let inner = issue_inner(request_id, descriptor_hash, endpoint);
        let router = enabled_router(&challenger, 8);

        let v1 = DiscoveryEndpointAuthenticatedTransportV1::sign(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            descriptor_hash,
            canonical_public_endpoint_commitment(endpoint).expect("endpoint"),
            flow_context,
            now,
            now + 60,
            &inner,
            &target,
        )
        .expect("V1 transport")
        .encode();
        let (v1_status, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            v1,
        )
        .await;
        assert_eq!(v1_status, StatusCode::BAD_REQUEST);

        let wrong_context = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            [0x77; 32],
            now,
            &inner,
            &signed_descriptor,
            &target,
        );
        let (context_status, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            wrong_context,
        )
        .await;
        assert_eq!(context_status, StatusCode::BAD_REQUEST);

        let stale_descriptor = descriptor(&target, 6, now - 180, endpoint);
        let stale_inner = issue_inner(
            request_id,
            descriptor_commitment(&stale_descriptor),
            endpoint,
        );
        let stale = DiscoveryEndpointAuthenticatedTransportV2::sign(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            flow_context,
            now - 120,
            now - 60,
            &stale_inner,
            &stale_descriptor,
            &target,
        )
        .expect("stale signed transport")
        .encode();
        let (stale_status, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            stale,
        )
        .await;
        assert_eq!(stale_status, StatusCode::BAD_REQUEST);

        let valid = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            flow_context,
            now,
            &inner,
            &signed_descriptor,
            &target,
        );
        let (path_status, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Verify.path(),
            valid,
        )
        .await;
        assert_eq!(path_status, StatusCode::BAD_REQUEST);

        let mismatched_inner = issue_inner(request_id, descriptor_hash, "8.8.4.4:51820");
        let mismatch = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            flow_context,
            now,
            &mismatched_inner,
            &signed_descriptor,
            &target,
        );
        let (mismatch_status, _) = post(
            router,
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            mismatch,
        )
        .await;
        assert_eq!(mismatch_status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn descriptor_tamper_conflict_and_capacity_fail_closed() {
        let now = unix_now_secs();
        let challenger = key(3);
        let target = key(9);
        let endpoint = "8.8.8.8:51820";
        let signed_descriptor = descriptor(&target, 7, now, endpoint);
        let flow_context = public_endpoint_flow_context(challenger.public_key_bytes());
        let first_id = [0x31; 32];
        let first_inner = issue_inner(
            first_id,
            descriptor_commitment(&signed_descriptor),
            endpoint,
        );
        let mut tampered = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            first_id,
            flow_context,
            now,
            &first_inner,
            &signed_descriptor,
            &target,
        );
        tampered[200] ^= 1;
        let router = enabled_router(&challenger, 1);
        let (tamper_status, _) = post(
            router.clone(),
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            tampered,
        )
        .await;
        assert_eq!(tamper_status, StatusCode::BAD_REQUEST);

        let first = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            first_id,
            flow_context,
            now,
            &first_inner,
            &signed_descriptor,
            &target,
        );
        assert_eq!(
            post(
                router.clone(),
                DiscoveryEndpointTransportOperationV1::Issue.path(),
                first,
            )
            .await
            .0,
            StatusCode::OK
        );

        let rotated = descriptor(&target, 8, now, endpoint);
        let conflict_inner = issue_inner(first_id, descriptor_commitment(&rotated), endpoint);
        let conflict = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            first_id,
            flow_context,
            now,
            &conflict_inner,
            &rotated,
            &target,
        );
        assert_eq!(
            post(
                router.clone(),
                DiscoveryEndpointTransportOperationV1::Issue.path(),
                conflict,
            )
            .await
            .0,
            StatusCode::CONFLICT
        );

        let second_id = [0x32; 32];
        let second_inner = issue_inner(
            second_id,
            descriptor_commitment(&signed_descriptor),
            endpoint,
        );
        let second = v2_transport(
            DiscoveryEndpointTransportOperationV1::Issue,
            second_id,
            flow_context,
            now,
            &second_inner,
            &signed_descriptor,
            &target,
        );
        assert_eq!(
            post(
                router,
                DiscoveryEndpointTransportOperationV1::Issue.path(),
                second,
            )
            .await
            .0,
            StatusCode::TOO_MANY_REQUESTS
        );
    }
}
