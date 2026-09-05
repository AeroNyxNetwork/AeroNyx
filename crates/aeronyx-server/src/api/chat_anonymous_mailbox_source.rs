// ============================================
// File: crates/aeronyx-server/src/api/chat_anonymous_mailbox_source.rs
// ============================================
//! VPN-only composition for exact-target anonymous-mailbox source requests.
//!
//! This route is deliberately not a peer ingress API: it accepts only a
//! descriptor-pinned canonical terminal frame, reloads that exact descriptor
//! before every outbound attempt, and persists retry/one-shot state in the
//! source journal. The caller supplies neither an endpoint nor a raw peer
//! carrier, so it cannot steer transport or cause fanout.

use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::protocol::anonymous_mailbox::{
    AnonymousMailboxSourceSealedResponseV1, MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS,
    MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES, MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES,
};
use aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Extension, Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::api::mpi::AuthenticatedOwner;
use crate::api::{decode_bounded_json_response, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES};
use crate::config_chat_relay::AnonymousMailboxSourceConfig;
use crate::services::chat_relay_anonymous_mailbox_source::{
    AnonymousMailboxSourceCoordinator, AnonymousMailboxSourceError, AnonymousMailboxSourceResult,
    ExactAnonymousMailboxTargetPin,
};

const SOURCE_SUBMIT_VERSION: u8 = 1;
/// The only source composition route. `mpi.rs` uses this exact constant to
/// retain authentication while suppressing unrelated owner-storage effects.
pub(crate) const ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH: &str =
    "/api/chat/anonymous-mailbox/source/submit";
const MAX_TERMINAL_FRAME_B64_BYTES: usize =
    ((MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES + 2) / 3) * 4;
const MAX_SEALED_RESPONSE_B64_BYTES: usize =
    ((MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES + 2) / 3) * 4;
const SOURCE_SUBMIT_BODY_MAX_BYTES: usize = MAX_TERMINAL_FRAME_B64_BYTES + 1024;

#[derive(Clone)]
struct SourceApiState {
    coordinator: Arc<AnonymousMailboxSourceCoordinator>,
    client: Arc<reqwest::Client>,
    admission: Arc<Semaphore>,
    timeout: Duration,
}

/// Builds the VPN-only source composition router. `server.rs` mounts this
/// result exclusively on `vpn_client_api`, never on node-peer/public routers.
pub(crate) fn build_chat_anonymous_mailbox_source_router(
    coordinator: Arc<AnonymousMailboxSourceCoordinator>,
    client: Arc<reqwest::Client>,
    config: &AnonymousMailboxSourceConfig,
) -> Router {
    let state = SourceApiState {
        coordinator,
        client,
        admission: Arc::new(Semaphore::new(config.max_in_flight)),
        timeout: Duration::from_secs(config.request_timeout_secs),
    };
    Router::new()
        .route(ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH, post(submit))
        .layer(DefaultBodyLimit::max(SOURCE_SUBMIT_BODY_MAX_BYTES))
        .with_state(state)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSubmitV1 {
    version: u8,
    route_id_b64: String,
    target: SourceTargetCommitmentV1,
    terminal_frame_b64: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceTargetCommitmentV1 {
    node_id_b64: String,
    sequence: u64,
    descriptor_hash_b64: String,
}

#[derive(Serialize)]
struct SourceSubmitResultV1 {
    version: u8,
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    terminal_response_b64: Option<String>,
}

#[derive(Serialize)]
struct SourceSubmitError {
    success: bool,
    error: &'static str,
}

struct SourceApiFailure {
    status: StatusCode,
    reason: &'static str,
}

impl SourceApiFailure {
    const fn invalid() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            reason: "anonymous_mailbox_source_rejected",
        }
    }

    const fn conflict() -> Self {
        Self {
            status: StatusCode::CONFLICT,
            reason: "anonymous_mailbox_source_conflict",
        }
    }

    const fn busy() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            reason: "anonymous_mailbox_source_busy",
        }
    }

    const fn unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            reason: "anonymous_mailbox_source_unavailable",
        }
    }

    const fn ambiguous() -> Self {
        Self {
            status: StatusCode::CONFLICT,
            reason: "anonymous_mailbox_source_ambiguous",
        }
    }
}

impl IntoResponse for SourceApiFailure {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(SourceSubmitError {
                success: false,
                error: self.reason,
            }),
        )
            .into_response()
    }
}

async fn submit(
    State(state): State<SourceApiState>,
    Extension(_owner): Extension<AuthenticatedOwner>,
    Json(input): Json<SourceSubmitV1>,
) -> Result<Json<SourceSubmitResultV1>, SourceApiFailure> {
    // The unified MPI middleware admitted this caller. Ownership does not
    // participate in the anonymous mailbox protocol and is dropped before any
    // journal, descriptor, route or network action to avoid an identity link.
    drop(_owner);
    let _permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| SourceApiFailure::busy())?;
    let (route_id, pin, terminal_frame) = parse_submit(input)?;
    let now = unix_now_secs();
    let coordinator = Arc::clone(&state.coordinator);
    let prepared =
        run_blocking(move || coordinator.prepare(pin, route_id, terminal_frame, now)).await?;
    let route_id = prepared.route_id();

    let coordinator = Arc::clone(&state.coordinator);
    match run_blocking(move || coordinator.result(route_id)).await? {
        AnonymousMailboxSourceResult::Completed(response) => return Ok(completed(response)),
        AnonymousMailboxSourceResult::Ambiguous => return Err(SourceApiFailure::ambiguous()),
        AnonymousMailboxSourceResult::Rejected => return Err(SourceApiFailure::conflict()),
        AnonymousMailboxSourceResult::Prepared | AnonymousMailboxSourceResult::Armed => {}
    }

    let coordinator = Arc::clone(&state.coordinator);
    let outbound =
        run_blocking(move || coordinator.begin_dispatch(route_id, unix_now_secs())).await?;
    let response = state
        .client
        .post(outbound.url().clone())
        .header("content-type", "application/json")
        .timeout(state.timeout)
        .body(outbound.body().to_vec())
        .send()
        .await
        .map_err(|_| SourceApiFailure::unavailable())?;
    if !response.status().is_success() {
        return Err(SourceApiFailure::unavailable());
    }
    let peer_response = decode_bounded_json_response(response, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES)
        .await
        .map_err(|_| SourceApiFailure::unavailable())?;
    let sealed_response =
        match validate_terminal_response(&peer_response, &outbound, unix_now_secs()) {
            Ok(response) => response,
            Err(()) => {
                let coordinator = Arc::clone(&state.coordinator);
                let _ = run_blocking(move || coordinator.mark_ambiguous(route_id)).await;
                return Err(SourceApiFailure::ambiguous());
            }
        };
    let coordinator = Arc::clone(&state.coordinator);
    run_blocking(move || coordinator.open_response(route_id, &sealed_response)).await?;
    let coordinator = Arc::clone(&state.coordinator);
    match run_blocking(move || coordinator.result(route_id)).await? {
        AnonymousMailboxSourceResult::Completed(response) => Ok(completed(response)),
        AnonymousMailboxSourceResult::Ambiguous => Err(SourceApiFailure::ambiguous()),
        _ => Err(SourceApiFailure::unavailable()),
    }
}

fn completed(response: Vec<u8>) -> Json<SourceSubmitResultV1> {
    Json(SourceSubmitResultV1 {
        version: SOURCE_SUBMIT_VERSION,
        state: "completed",
        terminal_response_b64: Some(STANDARD.encode(response)),
    })
}

fn parse_submit(
    input: SourceSubmitV1,
) -> Result<([u8; 16], ExactAnonymousMailboxTargetPin, Vec<u8>), SourceApiFailure> {
    if input.version != SOURCE_SUBMIT_VERSION
        || input.terminal_frame_b64.len() > MAX_TERMINAL_FRAME_B64_BYTES
    {
        return Err(SourceApiFailure::invalid());
    }
    let route_id = decode_fixed_b64::<16>(&input.route_id_b64)?;
    let node_id = decode_fixed_b64::<32>(&input.target.node_id_b64)?;
    let descriptor_hash = decode_fixed_b64::<32>(&input.target.descriptor_hash_b64)?;
    if route_id == [0; 16]
        || node_id == [0; 32]
        || descriptor_hash == [0; 32]
        || input.target.sequence == 0
    {
        return Err(SourceApiFailure::invalid());
    }
    let terminal_frame = STANDARD
        .decode(input.terminal_frame_b64)
        .map_err(|_| SourceApiFailure::invalid())?;
    if terminal_frame.is_empty()
        || terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES
    {
        return Err(SourceApiFailure::invalid());
    }
    Ok((
        route_id,
        ExactAnonymousMailboxTargetPin::new(
            node_id,
            DirectoryDescriptorCommitmentV1 {
                node_id,
                sequence: input.target.sequence,
                descriptor_hash,
            },
        ),
        terminal_frame,
    ))
}

fn decode_fixed_b64<const N: usize>(value: &str) -> Result<[u8; N], SourceApiFailure> {
    let expected = ((N + 2) / 3) * 4;
    if value.len() != expected {
        return Err(SourceApiFailure::invalid());
    }
    STANDARD
        .decode(value)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(SourceApiFailure::invalid)
}

fn validate_terminal_response(
    response: &crate::api::chat_peer::PeerBlindRelayResponse,
    outbound: &crate::services::chat_relay_anonymous_mailbox_source::AnonymousMailboxSourceOutbound,
    now: u64,
) -> Result<Vec<u8>, ()> {
    if !response.accepted
        || !response.terminal
        || response.forwarded
        || response.delivery_receipt.is_some()
        || response.failure_receipt.is_some()
        || response.success_receipt.is_none()
    {
        return Err(());
    }
    let receipt = response.success_receipt.as_ref().ok_or(())?;
    if receipt.accepted_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        || now.saturating_sub(receipt.accepted_at) > MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS
    {
        return Err(());
    }
    let encoded = response.opaque_terminal_response_b64.as_deref().ok_or(())?;
    if encoded.is_empty() || encoded.len() > MAX_SEALED_RESPONSE_B64_BYTES {
        return Err(());
    }
    receipt
        .verify_expected(
            &outbound.request().envelope,
            response.terminal,
            response.forwarded,
            response.ttl_remaining,
            response.reason.as_deref(),
            None,
            Some(encoded.as_bytes()),
            outbound.target_node_id(),
        )
        .map_err(|_| ())?;
    let sealed = STANDARD.decode(encoded).map_err(|_| ())?;
    if sealed.is_empty() || sealed.len() > MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES {
        return Err(());
    }
    if STANDARD.encode(&sealed) != encoded {
        return Err(());
    }
    AnonymousMailboxSourceSealedResponseV1::decode(&sealed).map_err(|_| ())?;
    Ok(sealed)
}

async fn run_blocking<T>(
    operation: impl FnOnce() -> Result<T, AnonymousMailboxSourceError> + Send + 'static,
) -> Result<T, SourceApiFailure>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|_| SourceApiFailure::unavailable())?
        .map_err(map_source_error)
}

fn map_source_error(error: AnonymousMailboxSourceError) -> SourceApiFailure {
    match error {
        AnonymousMailboxSourceError::Conflict => SourceApiFailure::conflict(),
        AnonymousMailboxSourceError::Ambiguous => SourceApiFailure::ambiguous(),
        AnonymousMailboxSourceError::Disabled
        | AnonymousMailboxSourceError::Unavailable
        | AnonymousMailboxSourceError::Corrupt => SourceApiFailure::unavailable(),
        AnonymousMailboxSourceError::Rejected => SourceApiFailure::invalid(),
    }
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::anonymous_mailbox::{
        decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
        AnonymousMailboxOutcomeV1, AnonymousMailboxSourceTerminalCarrierV1,
        AnonymousMailboxTerminalFrameV1, AnonymousMailboxTicketIssueResponseV1,
        AnonymousMailboxTicketIssueV1,
    };
    use aeronyx_core::protocol::chat::BlindRelaySuccessReceipt;
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeDescriptor, NodeProtocolFeature, SignedNodeDescriptor,
    };
    use aeronyx_core::protocol::memchain::{decode_memchain, MemChainMessage, MEMCHAIN_MAGIC};
    use aeronyx_core::protocol::onion::open_onion_layer;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use parking_lot::{Mutex, RwLock};
    use rusqlite::Connection;
    use sha2::{Digest, Sha256};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tower::ServiceExt;

    use super::*;
    use crate::api::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};
    use crate::api::mpi::{
        build_mpi_router, build_mpi_router_with_source, Mode, MpiState, SessionEmbeddingCache,
    };
    use crate::services::chat_relay_anonymous_mailbox_source::{
        AnonymousMailboxSourcePhase, ExactAnonymousMailboxTargetResolver,
        SqliteAnonymousMailboxSourceJournal,
    };

    const NOW: u64 = 1_800_000_000;

    struct TestResolver {
        descriptor: SignedNodeDescriptor,
        calls: Mutex<usize>,
    }

    impl ExactAnonymousMailboxTargetResolver for TestResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
            *self.calls.lock() += 1;
            (node_id == &self.descriptor.descriptor.node_id).then(|| self.descriptor.clone())
        }
    }

    fn source_outbound_for_receipt_test() -> (
        crate::services::chat_relay_anonymous_mailbox_source::AnonymousMailboxSourceOutbound,
        IdentityKeyPair,
    ) {
        let target = IdentityKeyPair::generate();
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            7,
            NOW.saturating_sub(1),
            NOW.saturating_add(60),
            "test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        descriptor.public_endpoint = Some("https://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let resolver = Arc::new(TestResolver {
            descriptor,
            calls: Mutex::new(0),
        });
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            max_journal_entries: 4,
            max_journal_bytes: 512 * 1024,
            ..AnonymousMailboxSourceConfig::default()
        };
        let journal = SqliteAnonymousMailboxSourceJournal::new(
            Connection::open_in_memory().expect("sqlite"),
            [0xA1; 32],
            &config,
        )
        .expect("journal");
        let source = Arc::new(IdentityKeyPair::generate());
        let coordinator =
            AnonymousMailboxSourceCoordinator::new(source, resolver, Arc::new(journal));
        let route_id = [0xA2; 16];
        let request = AnonymousMailboxTicketIssueV1::new(
            [0xA3; 16],
            [0xA4; 16],
            target.public_key_bytes(),
            [0xA5; 32],
            NOW,
            NOW + 30,
            0,
        )
        .expect("ticket issue");
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(request),
        )
        .expect("terminal frame");
        coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
                route_id,
                terminal_frame,
                NOW,
            )
            .expect("prepare");
        (
            coordinator
                .begin_dispatch(route_id, NOW)
                .expect("pinned outbound"),
            target,
        )
    }

    fn valid_submit_json() -> serde_json::Value {
        serde_json::json!({
            "version": SOURCE_SUBMIT_VERSION,
            "route_id_b64": STANDARD.encode([0x11; 16]),
            "target": {
                "node_id_b64": STANDARD.encode([0x12; 32]),
                "sequence": 7,
                "descriptor_hash_b64": STANDARD.encode([0x13; 32]),
            },
            "terminal_frame_b64": STANDARD.encode([0x14]),
        })
    }

    #[test]
    fn submit_schema_rejects_identity_or_endpoint_fields() {
        let mut with_identity = valid_submit_json();
        with_identity["sender"] = serde_json::json!("not admitted");
        assert!(serde_json::from_value::<SourceSubmitV1>(with_identity).is_err());

        let mut with_endpoint = valid_submit_json();
        with_endpoint["target"]["endpoint"] = serde_json::json!("https://127.0.0.1");
        assert!(serde_json::from_value::<SourceSubmitV1>(with_endpoint).is_err());
    }

    #[test]
    fn submit_parser_bounds_target_fields_before_coordinator_materialization() {
        let input = serde_json::from_value::<SourceSubmitV1>(valid_submit_json())
            .expect("closed schema input");
        assert!(parse_submit(input).is_ok());

        let mut oversized = valid_submit_json();
        oversized["terminal_frame_b64"] =
            serde_json::json!("A".repeat(MAX_TERMINAL_FRAME_B64_BYTES.saturating_add(1),));
        let input = serde_json::from_value::<SourceSubmitV1>(oversized).expect("shape");
        assert!(parse_submit(input).is_err());
    }

    #[test]
    fn durable_corruption_is_coarse_unavailable_not_client_invalid() {
        assert_eq!(
            map_source_error(AnonymousMailboxSourceError::Corrupt).status,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn source_response_requires_exact_receipt_and_canonical_amsr() {
        let (outbound, target) = source_outbound_for_receipt_test();
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            outbound.route_id(),
            [0xA6; 32],
            target.public_key_bytes(),
            IdentityKeyPair::generate().x25519_public_key_bytes(),
            &[0xA7],
            &target,
        )
        .and_then(|value| value.encode())
        .expect("sealed shape");
        let encoded = STANDARD.encode(&sealed);
        let receipt = BlindRelaySuccessReceipt::terminal(
            &outbound.request().envelope,
            1,
            None,
            None,
            Some(encoded.as_bytes()),
            NOW,
            &target,
        );
        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
            success_receipt: Some(receipt),
            failure_receipt: None,
            opaque_terminal_response_b64: Some(encoded.clone()),
        };
        assert_eq!(
            validate_terminal_response(&response, &outbound, NOW).expect("exact response"),
            sealed
        );

        let mut tampered = response.clone();
        let mut altered = sealed;
        altered[0] ^= 1;
        tampered.opaque_terminal_response_b64 = Some(STANDARD.encode(altered));
        assert!(validate_terminal_response(&tampered, &outbound, NOW).is_err());
    }

    // [M13J 2026-09-05 by Codex] Router admission uses the production MPI
    // middleware and real body-bound Ed25519 signatures. Durable retry and
    // terminal-proof tests use the coordinator's existing typed I/O boundary,
    // so they remain deterministic and never open a socket or contact a peer.
    struct RecordingResolver {
        descriptor: SignedNodeDescriptor,
        exact_calls: AtomicUsize,
        unexpected_calls: AtomicUsize,
    }

    impl ExactAnonymousMailboxTargetResolver for RecordingResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
            if node_id == &self.descriptor.descriptor.node_id {
                self.exact_calls.fetch_add(1, Ordering::Relaxed);
                Some(self.descriptor.clone())
            } else {
                self.unexpected_calls.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    struct SourceRouterFixture {
        coordinator: Arc<AnonymousMailboxSourceCoordinator>,
        resolver: Arc<RecordingResolver>,
        config: AnonymousMailboxSourceConfig,
        target: IdentityKeyPair,
        route_id: [u8; 16],
        commitment: DirectoryDescriptorCommitmentV1,
        terminal_frame: Vec<u8>,
        _source_store: tempfile::TempDir,
    }

    fn source_mpi_state() -> Arc<MpiState> {
        let identity = IdentityKeyPair::from_bytes(&[0xB1; 32]).expect("test MPI identity");
        let owner_key = identity.public_key_bytes();
        Arc::new(MpiState {
            mode: Mode::Local,
            storage: None,
            vector_index: None,
            identity,
            identity_cache: RwLock::new(HashMap::new()),
            index_ready: AtomicBool::new(true),
            user_weights: Arc::new(RwLock::new(HashMap::new())),
            mvf_alpha: 0.0,
            mvf_enabled: false,
            session_embeddings: RwLock::new(SessionEmbeddingCache::default()),
            mvf_baseline: RwLock::new(None),
            owner_key,
            api_secret: Some("m13j-router-test-secret".into()),
            embed_engine: None,
            // Remote body-bound auth is the production admission path for the
            // source route; it must not allocate regular MPI storage.
            allow_remote_storage: true,
            blind_storage_enabled: false,
            max_remote_owners: 0,
            ner_engine: None,
            graph_enabled: false,
            entropy_filter_enabled: false,
            reranker_engine: None,
            rawlog_key: Some([0xB2; 32]),
            llm_router: None,
            storage_pool: None,
            vector_pool: None,
            volume_router: None,
            system_db: None,
            jwt_secret: None,
            token_ttl_secs: 86_400,
            pool_max_connections: 0,
            pool_idle_timeout_secs: 0,
        })
    }

    fn canonical_ticket_terminal(
        target: &IdentityKeyPair,
        request_id: [u8; 16],
        now: u64,
    ) -> Vec<u8> {
        let request = AnonymousMailboxTicketIssueV1::new(
            request_id,
            [0xB4; 16],
            target.public_key_bytes(),
            [0xB5; 32],
            now,
            now.saturating_add(30),
            0,
        )
        .expect("canonical ticket request");
        encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::TicketIssue(
            request,
        ))
        .expect("canonical terminal frame")
    }

    fn source_router_fixture() -> SourceRouterFixture {
        let target = IdentityKeyPair::from_bytes(&[0xB3; 32]).expect("test terminal identity");
        let now = unix_now_secs();
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            11,
            now.saturating_sub(1),
            now.saturating_add(60),
            "m13j-router-test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        // This is an unroutable test target. The deterministic coordinator
        // seam below never opens a connection to it.
        descriptor.public_endpoint = Some("http://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let resolver = Arc::new(RecordingResolver {
            descriptor,
            exact_calls: AtomicUsize::new(0),
            unexpected_calls: AtomicUsize::new(0),
        });
        let source_store = tempfile::tempdir().expect("private source store");
        let source_db = std::fs::canonicalize(source_store.path())
            .expect("canonical private source store")
            .join("source.sqlite");
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: source_db.to_string_lossy().into_owned(),
            max_journal_entries: 8,
            max_journal_bytes: 512 * 1024,
            max_in_flight: 2,
            request_timeout_secs: 2,
        };
        let journal = SqliteAnonymousMailboxSourceJournal::open(config.clone(), [0xB6; 32])
            .expect("private source journal");
        let coordinator = Arc::new(AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[0xB7; 32]).expect("source identity")),
            resolver.clone(),
            Arc::new(journal),
        ));
        SourceRouterFixture {
            coordinator,
            resolver,
            config,
            target: target.clone(),
            route_id: [0xB8; 16],
            commitment,
            terminal_frame: canonical_ticket_terminal(&target, [0xB9; 16], now),
            _source_store: source_store,
        }
    }

    fn submit_body(
        route_id: [u8; 16],
        commitment: DirectoryDescriptorCommitmentV1,
        terminal_frame: &[u8],
    ) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "version": SOURCE_SUBMIT_VERSION,
            "route_id_b64": STANDARD.encode(route_id),
            "target": {
                "node_id_b64": STANDARD.encode(commitment.node_id),
                "sequence": commitment.sequence,
                "descriptor_hash_b64": STANDARD.encode(commitment.descriptor_hash),
            },
            "terminal_frame_b64": STANDARD.encode(terminal_frame),
        }))
        .expect("source submit JSON")
    }

    fn signed_remote_request(
        method: Method,
        path: &str,
        signed_body: &[u8],
        body: Vec<u8>,
        signer: &IdentityKeyPair,
    ) -> Request<Body> {
        let timestamp = unix_now_secs().to_string();
        let body_hash = Sha256::digest(signed_body);
        let mut digest = Sha256::new();
        digest.update(timestamp.as_bytes());
        digest.update(method.as_str().as_bytes());
        digest.update(path.as_bytes());
        digest.update(body_hash);
        let signature = signer.sign(&digest.finalize());
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .header(
                "x-memchain-publickey",
                hex::encode(signer.public_key_bytes()),
            )
            .header("x-memchain-timestamp", timestamp)
            .header("x-memchain-signature", hex::encode(signature))
            .body(Body::from(body))
            .expect("authenticated source request")
    }

    fn source_app(
        state: Arc<MpiState>,
        fixture: &SourceRouterFixture,
        client: reqwest::Client,
    ) -> Router {
        let source = build_chat_anonymous_mailbox_source_router(
            Arc::clone(&fixture.coordinator),
            Arc::new(client),
            &fixture.config,
        );
        build_mpi_router_with_source(state, source)
    }

    fn direct_test_client() -> reqwest::Client {
        // Avoid the host proxy discovery provider: source transport never
        // inherits it in production, and router tests need no network.
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("direct test client")
    }

    fn terminal_peer_response(peer_body: &[u8], target: &IdentityKeyPair) -> (Vec<u8>, Vec<u8>) {
        let request: PeerBlindRelayRequest =
            serde_json::from_slice(peer_body).expect("canonical peer request");
        let (terminal_kem_secret, _) = target.to_x25519();
        let peeled = open_onion_layer(&request.envelope.encrypted_blob, &terminal_kem_secret)
            .expect("target peels source onion");
        assert!(peeled.next_hop.is_none(), "source route has one exact hop");
        assert_eq!(peeled.inner.first().copied(), Some(MEMCHAIN_MAGIC));
        let MemChainMessage::AnonymousMailboxRouteV1(route) =
            decode_memchain(&peeled.inner[1..]).expect("source route payload")
        else {
            panic!("source payload must be an anonymous-mailbox route");
        };
        let carrier = AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
            &route.sealed_terminal_frame,
            request.envelope.route_id,
            target.public_key_bytes(),
        )
        .expect("canonical terminal carrier");
        let AnonymousMailboxTerminalFrameV1::TicketIssue(ticket) =
            decode_anonymous_mailbox_terminal_frame(carrier.terminal_frame())
                .expect("canonical ticket terminal")
        else {
            panic!("fixture only sends ticket issue");
        };
        let now = unix_now_secs();
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(
                AnonymousMailboxTicketIssueResponseV1::signed(
                    &ticket,
                    AnonymousMailboxOutcomeV1::Rejected,
                    None,
                    now,
                    target,
                )
                .expect("signed terminal response"),
            ),
        )
        .expect("canonical terminal response");
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            request.envelope.route_id,
            carrier.context_commitment(),
            target.public_key_bytes(),
            carrier.reply_public_key(),
            &terminal_frame,
            target,
        )
        .and_then(|value| value.encode())
        .expect("source sealed response");
        let encoded = STANDARD.encode(&sealed);
        let receipt = BlindRelaySuccessReceipt::terminal(
            &request.envelope,
            1,
            None,
            None,
            Some(encoded.as_bytes()),
            now,
            target,
        );
        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
            success_receipt: Some(receipt),
            failure_receipt: None,
            opaque_terminal_response_b64: Some(encoded),
        };
        (
            serde_json::to_vec(&response).expect("peer response JSON"),
            terminal_frame,
        )
    }

    // [M13J-E2 2026-09-05 by Codex] This is a synthetic host-local HTTP proxy
    // fixture. It intercepts the descriptor-pinned public test URL at an
    // ephemeral loopback port and never models an onion hop, a fleet node, or
    // an externally reachable endpoint.
    async fn read_proxy_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
        let mut bytes = Vec::new();
        loop {
            let mut chunk = [0u8; 2048];
            let read = stream.read(&mut chunk).await.expect("proxy read");
            assert_ne!(read, 0, "HTTP client closed before sending a request");
            bytes.extend_from_slice(&chunk[..read]);
            let Some(headers_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let body_start = headers_end + 4;
            let headers = std::str::from_utf8(&bytes[..headers_end]).expect("ASCII headers");
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .expect("content length");
            if bytes.len() < body_start + content_length {
                continue;
            }
            let request_line = headers.lines().next().expect("request line").to_owned();
            return (
                request_line,
                bytes[body_start..body_start + content_length].to_vec(),
            );
        }
    }

    fn loopback_proxy_client(listener: &TcpListener) -> reqwest::Client {
        let address = listener.local_addr().expect("loopback proxy address");
        reqwest::Client::builder()
            .no_proxy()
            .proxy(reqwest::Proxy::all(format!("http://{address}")).expect("proxy URL"))
            .build()
            .expect("loopback proxy client")
    }

    async fn assert_no_proxy_connection(listener: &TcpListener) {
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err(),
            "source request unexpectedly reached the local proxy"
        );
    }

    #[tokio::test]
    async fn source_router_rejects_before_journal_or_exact_target_io() {
        let fixture = source_router_fixture();
        let state = source_mpi_state();
        let app = source_app(state.clone(), &fixture, direct_test_client());
        let signer = IdentityKeyPair::from_bytes(&[0xBA; 32]).expect("remote signer");
        let valid = submit_body(
            fixture.route_id,
            fixture.commitment,
            &fixture.terminal_frame,
        );

        let missing_auth = Request::builder()
            .method(Method::POST)
            .uri(ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH)
            .header("content-type", "application/json")
            .body(Body::from(valid.clone()))
            .expect("request");
        assert_eq!(
            app.clone()
                .oneshot(missing_auth)
                .await
                .expect("response")
                .status(),
            StatusCode::UNAUTHORIZED
        );

        let mut tampered = valid.clone();
        *tampered.last_mut().expect("non-empty body") ^= 1;
        assert_eq!(
            app.clone()
                .oneshot(signed_remote_request(
                    Method::POST,
                    ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                    &valid,
                    tampered,
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::UNAUTHORIZED
        );

        assert_eq!(
            app.clone()
                .oneshot(signed_remote_request(
                    Method::GET,
                    ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                    &valid,
                    valid.clone(),
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::METHOD_NOT_ALLOWED
        );

        let extra_path = "/api/chat/anonymous-mailbox/source/submit/extra";
        assert_eq!(
            app.clone()
                .oneshot(signed_remote_request(
                    Method::POST,
                    extra_path,
                    &valid,
                    valid.clone(),
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::NOT_FOUND
        );

        let oversized = vec![b'x'; SOURCE_SUBMIT_BODY_MAX_BYTES.saturating_add(1)];
        assert_eq!(
            app.clone()
                .oneshot(signed_remote_request(
                    Method::POST,
                    ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                    &oversized,
                    oversized.clone(),
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );

        let malformed = submit_body(fixture.route_id, fixture.commitment, &[0x01]);
        assert_eq!(
            app.clone()
                .oneshot(signed_remote_request(
                    Method::POST,
                    ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                    &malformed,
                    malformed.clone(),
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::BAD_REQUEST
        );

        // An ordinary peer path is never a source ingress, even with a valid
        // source-style signature and body.
        assert_eq!(
            app.oneshot(signed_remote_request(
                Method::POST,
                "/api/chat/peer/blind-relay",
                &valid,
                valid.clone(),
                &signer,
            ))
            .await
            .expect("response")
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fixture.resolver.unexpected_calls.load(Ordering::Relaxed), 0);

        // The disabled composition is the base MPI router: it owns no source
        // coordinator, journal, key, route or outbound client.
        let disabled = build_mpi_router(state);
        let disabled_body = submit_body(
            fixture.route_id,
            fixture.commitment,
            &fixture.terminal_frame,
        );
        assert_eq!(
            disabled
                .oneshot(signed_remote_request(
                    Method::POST,
                    ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                    &disabled_body,
                    disabled_body.clone(),
                    &signer,
                ))
                .await
                .expect("response")
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn source_router_local_socket_retries_exact_body_and_completes_once() {
        let fixture = source_router_fixture();
        let state = source_mpi_state();
        let signer = IdentityKeyPair::from_bytes(&[0xBD; 32]).expect("remote signer");
        let body = submit_body(
            fixture.route_id,
            fixture.commitment,
            &fixture.terminal_frame,
        );

        // Authentication fails in MPI middleware before a source record or
        // transport attempt can exist.
        let rejected_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback proxy");
        let rejected = source_app(
            state.clone(),
            &fixture,
            loopback_proxy_client(&rejected_listener),
        )
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH)
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .expect("unsigned request"),
        )
        .await
        .expect("middleware rejection");
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
        assert_no_proxy_connection(&rejected_listener).await;
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 0);

        // A proxy that reads and drops the connection models a response loss
        // after the exact body has left the coordinator, without any external
        // connection. The request must remain Armed for one exact retry.
        let loss_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback loss proxy");
        let loss_client = loopback_proxy_client(&loss_listener);
        let lost_request = tokio::spawn(async move {
            let (mut stream, _) = loss_listener.accept().await.expect("one proxy connection");
            read_proxy_request(&mut stream).await
        });
        let first = source_app(state.clone(), &fixture, loss_client)
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                &signer,
            ))
            .await
            .expect("lost response result");
        assert_eq!(first.status(), StatusCode::SERVICE_UNAVAILABLE);
        let (first_line, first_peer_body) = lost_request.await.expect("loss proxy task");
        assert_eq!(
            first_line,
            "POST http://8.8.8.8/api/chat/peer/blind-relay HTTP/1.1"
        );
        assert_eq!(
            fixture
                .coordinator
                .resume(fixture.route_id)
                .expect("armed durable request")
                .body(),
            first_peer_body
        );

        let reply_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback reply proxy");
        let reply_client = loopback_proxy_client(&reply_listener);
        let target = fixture.target.clone();
        let delivered_request = tokio::spawn(async move {
            let (mut stream, _) = reply_listener.accept().await.expect("one proxy connection");
            let (request_line, peer_body) = read_proxy_request(&mut stream).await;
            let (response, terminal_frame) = terminal_peer_response(&peer_body, &target);
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream
                .write_all(headers.as_bytes())
                .await
                .expect("response headers");
            stream.write_all(&response).await.expect("response body");
            (request_line, peer_body, terminal_frame)
        });
        let second = source_app(state.clone(), &fixture, reply_client)
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                &signer,
            ))
            .await
            .expect("terminal response result");
        assert_eq!(second.status(), StatusCode::OK);
        let completed_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(second.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
                .await
                .expect("bounded completion body"),
        )
        .expect("completion JSON");
        assert_eq!(completed_json["state"], "completed");
        let terminal_b64 = completed_json["terminal_response_b64"]
            .as_str()
            .expect("terminal response");
        let (second_line, second_peer_body, expected_terminal) =
            delivered_request.await.expect("reply proxy task");
        assert_eq!(
            second_line,
            "POST http://8.8.8.8/api/chat/peer/blind-relay HTTP/1.1"
        );
        assert_eq!(second_peer_body, first_peer_body, "retry sends exact bytes");
        assert_eq!(
            STANDARD
                .decode(terminal_b64)
                .expect("base64 terminal response"),
            expected_terminal
        );

        // Completion and same-route conflicts never open the proxy again.
        let completed_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback completion proxy");
        let completed_retry = source_app(
            state.clone(),
            &fixture,
            loopback_proxy_client(&completed_listener),
        )
        .oneshot(signed_remote_request(
            Method::POST,
            ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
            &body,
            body.clone(),
            &signer,
        ))
        .await
        .expect("completed retry result");
        assert_eq!(completed_retry.status(), StatusCode::OK);
        let completed_retry_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(completed_retry.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
                .await
                .expect("bounded completed retry body"),
        )
        .expect("completed retry JSON");
        assert_eq!(completed_retry_json, completed_json);
        assert_no_proxy_connection(&completed_listener).await;

        let conflicting_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback conflict proxy");
        let conflicting_terminal =
            canonical_ticket_terminal(&fixture.target, [0xBE; 16], unix_now_secs());
        let conflicting = submit_body(fixture.route_id, fixture.commitment, &conflicting_terminal);
        let conflict = source_app(
            state,
            &fixture,
            loopback_proxy_client(&conflicting_listener),
        )
        .oneshot(signed_remote_request(
            Method::POST,
            ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
            &conflicting,
            conflicting.clone(),
            &signer,
        ))
        .await
        .expect("conflict result");
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_no_proxy_connection(&conflicting_listener).await;
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 3);
        assert_eq!(fixture.resolver.unexpected_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn source_coordinator_preserves_exact_body_across_response_loss_retry_and_completion() {
        let fixture = source_router_fixture();
        let now = unix_now_secs();
        let prepared = fixture
            .coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(
                    fixture.target.public_key_bytes(),
                    fixture.commitment,
                ),
                fixture.route_id,
                fixture.terminal_frame.clone(),
                now,
            )
            .expect("durable source preparation");
        let first_outbound = fixture
            .coordinator
            .begin_dispatch(fixture.route_id, now)
            .expect("first exact target dispatch");
        assert_eq!(prepared.body(), first_outbound.body());

        // Model a response lost after the exact body was released: no terminal
        // response is opened, so the retained armed record is the only retry.
        let stored_body = fixture
            .coordinator
            .resume(fixture.route_id)
            .expect("armed source retry")
            .body()
            .to_vec();
        assert_eq!(stored_body, first_outbound.body());
        let retry_outbound = fixture
            .coordinator
            .begin_dispatch(fixture.route_id, now)
            .expect("exact durable retry dispatch");
        assert_eq!(
            retry_outbound.body(),
            first_outbound.body(),
            "retry sends exact durable bytes"
        );

        let (peer_response, expected_terminal) =
            terminal_peer_response(retry_outbound.body(), &fixture.target);
        let peer_response: PeerBlindRelayResponse =
            serde_json::from_slice(&peer_response).expect("canonical terminal peer response");
        let sealed_response =
            validate_terminal_response(&peer_response, &retry_outbound, unix_now_secs())
                .expect("receipt and envelope-bound terminal response");
        fixture
            .coordinator
            .open_response(fixture.route_id, &sealed_response)
            .expect("complete exact terminal response");
        let completed = match fixture
            .coordinator
            .result(fixture.route_id)
            .expect("result")
        {
            AnonymousMailboxSourceResult::Completed(response) => response,
            _ => panic!("source request must complete"),
        };
        assert_eq!(
            completed, expected_terminal,
            "completion retains canonical terminal bytes"
        );
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 3);
        assert_eq!(fixture.resolver.unexpected_calls.load(Ordering::Relaxed), 0);

        // Completion is terminal: the matching submit yields the same durable
        // body/result without descriptor resolution or another dispatch.
        let completed_retry = fixture
            .coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(
                    fixture.target.public_key_bytes(),
                    fixture.commitment,
                ),
                fixture.route_id,
                fixture.terminal_frame.clone(),
                now,
            )
            .expect("matching completed retry");
        assert_eq!(
            completed_retry.phase(),
            AnonymousMailboxSourcePhase::Completed
        );
        assert_eq!(completed_retry.body(), first_outbound.body());
        assert!(matches!(
            fixture.coordinator.result(fixture.route_id),
            Ok(AnonymousMailboxSourceResult::Completed(response)) if response == completed
        ));
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 3);

        let conflicting_terminal =
            canonical_ticket_terminal(&fixture.target, [0xBC; 16], unix_now_secs());
        assert!(matches!(
            fixture.coordinator.prepare(
                ExactAnonymousMailboxTargetPin::new(
                    fixture.target.public_key_bytes(),
                    fixture.commitment,
                ),
                fixture.route_id,
                conflicting_terminal,
                now,
            ),
            Err(AnonymousMailboxSourceError::Conflict)
        ));
        assert_eq!(fixture.resolver.exact_calls.load(Ordering::Relaxed), 3);
        assert_eq!(fixture.resolver.unexpected_calls.load(Ordering::Relaxed), 0);
    }
}
