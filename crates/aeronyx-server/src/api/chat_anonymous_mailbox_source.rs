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
    use std::net::TcpListener as StdTcpListener;
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::anonymous_mailbox::{
        decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
        AnonymousMailboxAckV1, AnonymousMailboxLeaseCreateV1, AnonymousMailboxOutcomeV1,
        AnonymousMailboxPullOneV1, AnonymousMailboxPullResultV1, AnonymousMailboxPutV1,
        AnonymousMailboxSourceTerminalCarrierV1, AnonymousMailboxTerminalFrameV1,
        AnonymousMailboxTicketIssueResponseV1, AnonymousMailboxTicketIssueV1,
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
    use tokio::process::Child;
    use tower::ServiceExt;

    use super::*;
    use crate::api::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};
    use crate::api::chat_peer_anonymous_mailbox::PreparedAnonymousMailboxTerminal;
    use crate::api::mpi::{
        build_mpi_router, build_mpi_router_with_source, Mode, MpiState, SessionEmbeddingCache,
    };
    use crate::config_chat_relay::AnonymousMailboxStoreConfig;
    use crate::services::chat_relay_anonymous_mailbox_source::{
        AnonymousMailboxSourcePhase, ExactAnonymousMailboxTargetResolver,
        SqliteAnonymousMailboxSourceJournal,
    };
    use crate::services::chat_relay_mailbox::{
        AnonymousMailboxCustodyRepository, AnonymousMailboxTicketIssueOutcome,
        SqliteAnonymousMailboxStore,
    };

    const NOW: u64 = 1_800_000_000;
    const PROCESS_CRASH_WORKER: &str = concat!(
        "api::chat_anonymous_mailbox_source::tests::",
        "source_router_process_crash_subprocess_worker"
    );
    const PROCESS_CRASH_MODE_ENV: &str = "AERONYX_TEST_SOURCE_PROCESS_CRASH_MODE";
    const PROCESS_CRASH_DB_ENV: &str = "AERONYX_TEST_SOURCE_PROCESS_CRASH_DB";
    const PROCESS_CRASH_PROXY_PORT_ENV: &str = "AERONYX_TEST_SOURCE_PROCESS_CRASH_PROXY_PORT";
    const PROCESS_CRASH_NOW_ENV: &str = "AERONYX_TEST_SOURCE_PROCESS_CRASH_NOW";
    const PROCESS_CRASH_DEADLINE: Duration = Duration::from_secs(20);

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

    struct PinnedTargetResolver {
        exact_node_id: [u8; 32],
        descriptor: RwLock<Option<SignedNodeDescriptor>>,
        alternate_node_id: [u8; 32],
        exact_calls: AtomicUsize,
        alternate_calls: AtomicUsize,
        unknown_calls: AtomicUsize,
    }

    impl ExactAnonymousMailboxTargetResolver for PinnedTargetResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
            if node_id == &self.exact_node_id {
                self.exact_calls.fetch_add(1, Ordering::Relaxed);
                return self.descriptor.read().clone();
            }
            if node_id == &self.alternate_node_id {
                self.alternate_calls.fetch_add(1, Ordering::Relaxed);
            } else {
                self.unknown_calls.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }

    impl PinnedTargetResolver {
        fn replace_exact_descriptor(&self, descriptor: Option<SignedNodeDescriptor>) {
            *self.descriptor.write() = descriptor;
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

    struct RealSourceEntry {
        coordinator: Arc<AnonymousMailboxSourceCoordinator>,
        resolver: Arc<PinnedTargetResolver>,
        config: AnonymousMailboxSourceConfig,
        source_seed: u8,
        journal_key: u8,
        alternate_node_id: [u8; 32],
        _source_store: tempfile::TempDir,
    }

    struct ProcessCrashFixture {
        target: IdentityKeyPair,
        descriptor: SignedNodeDescriptor,
        ticket_request: AnonymousMailboxTicketIssueV1,
        body: Vec<u8>,
    }

    struct ProcessCrashSourceEntry {
        coordinator: Arc<AnonymousMailboxSourceCoordinator>,
        resolver: Arc<PinnedTargetResolver>,
        config: AnonymousMailboxSourceConfig,
    }

    fn real_source_entry(
        descriptor: SignedNodeDescriptor,
        source_seed: u8,
        journal_key: u8,
        alternate_node_id: [u8; 32],
    ) -> RealSourceEntry {
        let source_store = tempfile::tempdir().expect("private source store");
        let source_db = std::fs::canonicalize(source_store.path())
            .expect("canonical private source store")
            .join("source.sqlite");
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: source_db.to_string_lossy().into_owned(),
            max_journal_entries: 16,
            max_journal_bytes: 512 * 1024,
            max_in_flight: 2,
            request_timeout_secs: 2,
            ..AnonymousMailboxSourceConfig::default()
        };
        let resolver = Arc::new(PinnedTargetResolver {
            exact_node_id: descriptor.descriptor.node_id,
            descriptor: RwLock::new(Some(descriptor)),
            alternate_node_id,
            exact_calls: AtomicUsize::new(0),
            alternate_calls: AtomicUsize::new(0),
            unknown_calls: AtomicUsize::new(0),
        });
        let journal = SqliteAnonymousMailboxSourceJournal::open(config.clone(), [journal_key; 32])
            .expect("private source journal");
        let coordinator = Arc::new(AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[source_seed; 32]).expect("source identity")),
            resolver.clone(),
            Arc::new(journal),
        ));
        RealSourceEntry {
            coordinator,
            resolver,
            config,
            source_seed,
            journal_key,
            alternate_node_id,
            _source_store: source_store,
        }
    }

    // [E6-EXACT-TARGET-RECOVERY-GUARD 2026-09-05 by Codex] Transfer only the
    // test-owned private journal directory into a fresh coordinator. This
    // models source-process journal reopening without changing the production
    // recovery path or adding a test-only route/session bypass.
    fn reopen_real_source_entry(
        entry: RealSourceEntry,
        descriptor: SignedNodeDescriptor,
    ) -> RealSourceEntry {
        let RealSourceEntry {
            config,
            source_seed,
            journal_key,
            alternate_node_id,
            _source_store,
            ..
        } = entry;
        let resolver = Arc::new(PinnedTargetResolver {
            exact_node_id: descriptor.descriptor.node_id,
            descriptor: RwLock::new(Some(descriptor)),
            alternate_node_id,
            exact_calls: AtomicUsize::new(0),
            alternate_calls: AtomicUsize::new(0),
            unknown_calls: AtomicUsize::new(0),
        });
        let journal = SqliteAnonymousMailboxSourceJournal::open(config.clone(), [journal_key; 32])
            .expect("reopen private source journal");
        let coordinator = Arc::new(AnonymousMailboxSourceCoordinator::new(
            Arc::new(
                IdentityKeyPair::from_bytes(&[source_seed; 32]).expect("reopened source identity"),
            ),
            resolver.clone(),
            Arc::new(journal),
        ));
        RealSourceEntry {
            coordinator,
            resolver,
            config,
            source_seed,
            journal_key,
            alternate_node_id,
            _source_store,
        }
    }

    // [E7-PROCESS-CRASH-EVIDENCE 2026-09-06 by Codex] The replacement child
    // gets only a private SQLite path, an ephemeral loopback proxy port, a
    // clock value and a fixture mode. It reconstructs all keys and the signed
    // source request locally; no key, route or payload crosses argv/env.
    fn process_crash_fixture(now: u64) -> ProcessCrashFixture {
        let target = IdentityKeyPair::from_bytes(&[0xC1; 32]).expect("process-crash target");
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            37,
            now.saturating_sub(1),
            now.saturating_add(60),
            "e7-process-crash-test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        // The ordinary source client uses only the test-owned loopback proxy;
        // this public-shaped descriptor remains unreachable from the host.
        descriptor.public_endpoint = Some("http://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let ticket_request =
            ticket_request_with_one_bit_proof(&target, [0xC5; 16], [0xC6; 16], [0xC7; 32], now);
        let terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        )
        .expect("ticket terminal");
        ProcessCrashFixture {
            target,
            descriptor,
            ticket_request,
            body: submit_body([0xC8; 16], commitment, &terminal),
        }
    }

    fn process_crash_source_config(db_path: &Path) -> AnonymousMailboxSourceConfig {
        AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: db_path.to_string_lossy().into_owned(),
            max_journal_entries: 16,
            max_journal_bytes: 512 * 1024,
            max_in_flight: 2,
            request_timeout_secs: 15,
            ..AnonymousMailboxSourceConfig::default()
        }
    }

    fn process_crash_source_entry(
        db_path: &Path,
        fixture: &ProcessCrashFixture,
        mode: &str,
    ) -> ProcessCrashSourceEntry {
        let descriptor = match mode {
            "crash" | "restored" => Some(fixture.descriptor.clone()),
            "rotated" => {
                let mut rotated = fixture.descriptor.descriptor.clone();
                rotated.sequence = rotated.sequence.saturating_add(1);
                Some(
                    SignedNodeDescriptor::sign(rotated, &fixture.target)
                        .expect("rotated process-crash descriptor"),
                )
            }
            "missing" => None,
            _ => panic!("unknown process-crash worker mode"),
        };
        let config = process_crash_source_config(db_path);
        let resolver = Arc::new(PinnedTargetResolver {
            exact_node_id: fixture.target.public_key_bytes(),
            descriptor: RwLock::new(descriptor),
            alternate_node_id: [0xC9; 32],
            exact_calls: AtomicUsize::new(0),
            alternate_calls: AtomicUsize::new(0),
            unknown_calls: AtomicUsize::new(0),
        });
        let journal = SqliteAnonymousMailboxSourceJournal::open(config.clone(), [0xCA; 32])
            .expect("process-crash source journal");
        let coordinator = Arc::new(AnonymousMailboxSourceCoordinator::new(
            Arc::new(
                IdentityKeyPair::from_bytes(&[0xCB; 32]).expect("process-crash source identity"),
            ),
            resolver.clone(),
            Arc::new(journal),
        ));
        ProcessCrashSourceEntry {
            coordinator,
            resolver,
            config,
        }
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

    fn ticket_request_with_one_bit_proof(
        target: &IdentityKeyPair,
        request_id: [u8; 16],
        ticket_id: [u8; 16],
        claims_commitment: [u8; 32],
        now: u64,
    ) -> AnonymousMailboxTicketIssueV1 {
        (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    request_id,
                    ticket_id,
                    target.public_key_bytes(),
                    claims_commitment,
                    now,
                    now.saturating_add(300),
                    proof_nonce,
                )
                .expect("ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit ticket proof")
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
            ..AnonymousMailboxSourceConfig::default()
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
        source_app_for(
            state,
            Arc::clone(&fixture.coordinator),
            &fixture.config,
            client,
        )
    }

    fn source_app_for(
        state: Arc<MpiState>,
        coordinator: Arc<AnonymousMailboxSourceCoordinator>,
        config: &AnonymousMailboxSourceConfig,
        client: reqwest::Client,
    ) -> Router {
        let source =
            build_chat_anonymous_mailbox_source_router(coordinator, Arc::new(client), config);
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

    // [M13J-E3 2026-09-05 by Codex] The loopback carrier below is only a
    // synthetic HTTP transport.  Its response is produced by the production
    // terminal adapter and a real private SQLite custody store; it is not an
    // onion middle hop, a fleet node, or an externally reachable peer.
    fn real_terminal_peer_response(
        peer_body: &[u8],
        target: &IdentityKeyPair,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
    ) -> Vec<u8> {
        let request: PeerBlindRelayRequest =
            serde_json::from_slice(peer_body).expect("canonical peer request");
        let (terminal_kem_secret, _) = target.to_x25519();
        let peeled = open_onion_layer(&request.envelope.encrypted_blob, &terminal_kem_secret)
            .expect("target peels source onion");
        assert!(peeled.next_hop.is_none(), "source route has one exact hop");
        let encoded = PreparedAnonymousMailboxTerminal::decode(
            &peeled.inner,
            request.envelope.route_id,
            target.public_key_bytes(),
        )
        .expect("canonical terminal request")
        .execute(repository, Arc::new(target.clone()), unix_now_secs())
        .expect("real terminal response");
        let receipt = BlindRelaySuccessReceipt::terminal(
            &request.envelope,
            1,
            None,
            None,
            Some(encoded.as_bytes()),
            unix_now_secs(),
            target,
        );
        serde_json::to_vec(&PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
            success_receipt: Some(receipt),
            failure_receipt: None,
            opaque_terminal_response_b64: Some(encoded),
        })
        .expect("real peer response JSON")
    }

    async fn write_peer_response(stream: &mut TcpStream, response: &[u8]) {
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            response.len()
        );
        stream
            .write_all(headers.as_bytes())
            .await
            .expect("response headers");
        stream.write_all(response).await.expect("response body");
    }

    // [M13J-E4 2026-09-05 by Codex] This barrier owns only test-created
    // loopback tasks. It signals after production terminal execution has
    // durably completed, holds the first HTTP response, and releases no
    // response until the test cancels its own source future. It is neither a
    // server shutdown/drain model nor a live relay peer.
    async fn held_real_terminal_proxy(
        target: IdentityKeyPair,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        accepted_connections: Arc<AtomicUsize>,
    ) -> (
        reqwest::Client,
        tokio::sync::oneshot::Receiver<Vec<u8>>,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Vec<Vec<u8>>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback held real-terminal proxy");
        let client = loopback_proxy_client(&listener);
        let (committed_tx, committed_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first_stream, _) = listener.accept().await.expect("first proxy connection");
            accepted_connections.fetch_add(1, Ordering::Relaxed);
            let (_, first_body) = read_proxy_request(&mut first_stream).await;
            let _ = real_terminal_peer_response(&first_body, &target, repository.clone());
            committed_tx
                .send(first_body.clone())
                .expect("first durable effect signal");
            release_rx
                .await
                .expect("test-owned source cancellation release");
            drop(first_stream);

            let mut bodies = vec![first_body];
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.expect("follow-up proxy connection");
                accepted_connections.fetch_add(1, Ordering::Relaxed);
                let (_, peer_body) = read_proxy_request(&mut stream).await;
                let response = real_terminal_peer_response(&peer_body, &target, repository.clone());
                write_peer_response(&mut stream, &response).await;
                bodies.push(peer_body);
            }
            bodies
        });
        (client, committed_rx, release_tx, server)
    }

    async fn source_request_with_real_terminal(
        state: Arc<MpiState>,
        entry: &RealSourceEntry,
        target: IdentityKeyPair,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        body: Vec<u8>,
        signer: &IdentityKeyPair,
    ) -> (StatusCode, Vec<u8>, Vec<u8>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback real-terminal proxy");
        let client = loopback_proxy_client(&listener);
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("one proxy connection");
            let (_, peer_body) = read_proxy_request(&mut stream).await;
            let response = real_terminal_peer_response(&peer_body, &target, repository);
            write_peer_response(&mut stream, &response).await;
            peer_body
        });
        let response = source_app_for(state, Arc::clone(&entry.coordinator), &entry.config, client)
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                signer,
            ))
            .await
            .expect("source terminal response");
        let status = response.status();
        let source_body = axum::body::to_bytes(response.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
            .await
            .expect("bounded source response")
            .to_vec();
        (
            status,
            source_body,
            server.await.expect("real-terminal proxy task"),
        )
    }

    async fn source_request_lost_after_real_terminal(
        state: Arc<MpiState>,
        entry: &RealSourceEntry,
        target: IdentityKeyPair,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        body: Vec<u8>,
        signer: &IdentityKeyPair,
    ) -> (StatusCode, Vec<u8>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback response-loss proxy");
        let client = loopback_proxy_client(&listener);
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("one proxy connection");
            let (_, peer_body) = read_proxy_request(&mut stream).await;
            let _ = real_terminal_peer_response(&peer_body, &target, repository);
            // Drop after the durable terminal effect and before any HTTP
            // response reaches the source.  The next attempt must reuse the
            // source journal's exact armed body.
            peer_body
        });
        let response = source_app_for(state, Arc::clone(&entry.coordinator), &entry.config, client)
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                signer,
            ))
            .await
            .expect("response-loss result");
        (
            response.status(),
            server.await.expect("response-loss proxy task"),
        )
    }

    fn completed_terminal_frame(source_body: &[u8]) -> AnonymousMailboxTerminalFrameV1 {
        let result: serde_json::Value =
            serde_json::from_slice(source_body).expect("completed source JSON");
        assert_eq!(result["state"], "completed");
        let response = result["terminal_response_b64"]
            .as_str()
            .expect("terminal response");
        decode_anonymous_mailbox_terminal_frame(
            &STANDARD.decode(response).expect("terminal response base64"),
        )
        .expect("terminal response frame")
    }

    fn assert_outer_peer_carrier_is_opaque(
        peer_body: &[u8],
        terminal_frame: &[u8],
        reader: &IdentityKeyPair,
    ) {
        let request: PeerBlindRelayRequest =
            serde_json::from_slice(peer_body).expect("canonical peer request");
        assert!(request.onward_envelope.is_none());
        assert!(request.onward_descriptor_hint.is_none());
        let raw = std::str::from_utf8(peer_body).expect("JSON peer carrier");
        assert!(
            !raw.contains(&STANDARD.encode(terminal_frame)),
            "outer peer carrier must not project the canonical terminal frame"
        );
        assert!(
            !raw.contains(&STANDARD.encode(reader.public_key_bytes())),
            "outer peer carrier must not project the receiver verifier"
        );
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
        loopback_proxy_client_at(address.port())
    }

    fn loopback_proxy_client_at(port: u16) -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .proxy(
                reqwest::Proxy::all(format!("http://127.0.0.1:{port}"))
                    .expect("loopback proxy URL"),
            )
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

    enum ProcessCrashTerminalControl {
        AssertQuiet(tokio::sync::oneshot::Sender<Result<(), String>>),
        ServeRestored(tokio::sync::oneshot::Sender<Vec<u8>>),
    }

    // [E7-PROCESS-CRASH-EVIDENCE 2026-09-06 by Codex] The terminal side is
    // parent-owned and local. It signals only after the real SQLite terminal
    // adapter returns, then withholds the first response until the parent has
    // reaped its own child. The control channel makes no-network assertions
    // deterministic with try_accept rather than time-based sleeps.
    fn process_crash_terminal_proxy(
        listener: TcpListener,
        inspection_listener: StdTcpListener,
        target: IdentityKeyPair,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
    ) -> (
        tokio::sync::oneshot::Receiver<Vec<u8>>,
        tokio::sync::oneshot::Sender<()>,
        tokio::sync::mpsc::Sender<ProcessCrashTerminalControl>,
        tokio::task::JoinHandle<()>,
    ) {
        let (committed_tx, committed_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let (control_tx, mut control_rx) = tokio::sync::mpsc::channel(3);
        let server = tokio::spawn(async move {
            let (mut first_stream, _) = listener
                .accept()
                .await
                .expect("process-crash first proxy connection");
            let (_, first_body) = read_proxy_request(&mut first_stream).await;
            let _ = real_terminal_peer_response(&first_body, &target, repository.clone());
            committed_tx
                .send(first_body)
                .expect("process-crash durable terminal event");
            release_rx.await.expect("process-crash parent release");
            drop(first_stream);

            while let Some(control) = control_rx.recv().await {
                match control {
                    ProcessCrashTerminalControl::AssertQuiet(reply) => {
                        let result = match inspection_listener.accept() {
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
                            Ok((stream, _)) => {
                                drop(stream);
                                Err("drifted replacement unexpectedly dispatched".into())
                            }
                            Err(_) => Err("process-crash listener inspection failed".into()),
                        };
                        let _ = reply.send(result);
                    }
                    ProcessCrashTerminalControl::ServeRestored(reply) => {
                        let (mut stream, _) = listener
                            .accept()
                            .await
                            .expect("restored process-crash proxy connection");
                        let (_, body) = read_proxy_request(&mut stream).await;
                        let response = real_terminal_peer_response(&body, &target, repository);
                        write_peer_response(&mut stream, &response).await;
                        let _ = reply.send(body);
                        return;
                    }
                }
            }
            panic!("process-crash parent dropped terminal control before restore");
        });
        (committed_rx, release_tx, control_tx, server)
    }

    fn spawn_process_crash_child(mode: &str, db_path: &Path, proxy_port: u16, now: u64) -> Child {
        let executable = std::env::current_exe().expect("resolve process-crash test binary");
        let mut child = crate::isolated_child_command(executable);
        child
            .arg(PROCESS_CRASH_WORKER)
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(PROCESS_CRASH_MODE_ENV, mode)
            .env(PROCESS_CRASH_DB_ENV, db_path)
            .env(PROCESS_CRASH_PROXY_PORT_ENV, proxy_port.to_string())
            .env(PROCESS_CRASH_NOW_ENV, now.to_string())
            .env_remove("AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_PHASE")
            .env_remove("AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_BARRIER")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        child.spawn().expect("spawn isolated process-crash child")
    }

    async fn terminate_and_reap_process_crash_child(child: &mut Child) {
        if matches!(child.try_wait(), Ok(None)) {
            let _ = child.kill().await;
        }
        let _ = tokio::time::timeout(PROCESS_CRASH_DEADLINE, child.wait()).await;
    }

    async fn require_process_crash_child_success(child: &mut Child) -> Result<(), String> {
        match tokio::time::timeout(PROCESS_CRASH_DEADLINE, child.wait()).await {
            Ok(Ok(status)) if status.success() => Ok(()),
            Ok(Ok(_)) => Err("replacement child exited unsuccessfully".into()),
            Ok(Err(_)) => Err("replacement child could not be reaped".into()),
            Err(_) => {
                terminate_and_reap_process_crash_child(child).await;
                Err("replacement child exceeded bounded deadline".into())
            }
        }
    }

    async fn kill_and_reap_process_crash_child(child: &mut Child) -> Result<(), String> {
        match child.try_wait() {
            Ok(None) => {}
            Ok(Some(_)) => return Err("crash child exited before parent kill".into()),
            Err(_) => return Err("could not inspect crash child".into()),
        }
        if child.kill().await.is_err() {
            terminate_and_reap_process_crash_child(child).await;
            return Err("could not terminate owned crash child".into());
        }
        match tokio::time::timeout(PROCESS_CRASH_DEADLINE, child.wait()).await {
            Ok(Ok(status)) if !status.success() => Ok(()),
            Ok(Ok(_)) => Err("owned crash child unexpectedly exited successfully".into()),
            Ok(Err(_)) => Err("owned crash child could not be reaped".into()),
            Err(_) => {
                terminate_and_reap_process_crash_child(child).await;
                Err("owned crash child exceeded bounded reap deadline".into())
            }
        }
    }

    async fn abort_and_reap_process_crash_server(server: &mut Option<tokio::task::JoinHandle<()>>) {
        if let Some(server) = server.take() {
            server.abort();
            let _ = tokio::time::timeout(PROCESS_CRASH_DEADLINE, server).await;
        }
    }

    async fn require_process_crash_server_complete(
        server: &mut Option<tokio::task::JoinHandle<()>>,
    ) -> Result<(), String> {
        let Some(mut server) = server.take() else {
            return Err("process-crash server completion was consumed".into());
        };
        match tokio::time::timeout(PROCESS_CRASH_DEADLINE, &mut server).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err("process-crash terminal fixture failed".into()),
            Err(_) => {
                server.abort();
                let _ = tokio::time::timeout(PROCESS_CRASH_DEADLINE, server).await;
                Err("process-crash terminal fixture exceeded bounded deadline".into())
            }
        }
    }

    async fn request_process_crash_quiet(
        control: &tokio::sync::mpsc::Sender<ProcessCrashTerminalControl>,
    ) -> Result<(), String> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        control
            .send(ProcessCrashTerminalControl::AssertQuiet(reply_tx))
            .await
            .map_err(|_| "process-crash terminal control unavailable".to_string())?;
        match tokio::time::timeout(PROCESS_CRASH_DEADLINE, reply_rx).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(_))) => Err("drifted replacement dispatched to terminal".into()),
            Ok(Err(_)) => Err("process-crash terminal dropped quiet acknowledgement".into()),
            Err(_) => Err("process-crash terminal quiet acknowledgement timed out".into()),
        }
    }

    #[test]
    #[ignore = "spawned only by the bounded process-crash source acceptance"]
    fn source_router_process_crash_subprocess_worker() {
        let mode = std::env::var(PROCESS_CRASH_MODE_ENV).expect("process-crash worker mode");
        let db_path = PathBuf::from(
            std::env::var_os(PROCESS_CRASH_DB_ENV).expect("process-crash worker database"),
        );
        let proxy_port = std::env::var(PROCESS_CRASH_PROXY_PORT_ENV)
            .expect("process-crash proxy port")
            .parse::<u16>()
            .expect("numeric process-crash proxy port");
        let now = std::env::var(PROCESS_CRASH_NOW_ENV)
            .expect("process-crash fixture clock")
            .parse::<u64>()
            .expect("numeric process-crash fixture clock");
        let fixture = process_crash_fixture(now);
        let entry = process_crash_source_entry(&db_path, &fixture, &mode);
        let signer = IdentityKeyPair::from_bytes(&[0xCC; 32]).expect("process-crash signer");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("process-crash worker runtime");
        runtime.block_on(async move {
            let response = source_app_for(
                source_mpi_state(),
                Arc::clone(&entry.coordinator),
                &entry.config,
                loopback_proxy_client_at(proxy_port),
            )
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &fixture.body,
                fixture.body.clone(),
                &signer,
            ))
            .await
            .expect("process-crash source response");

            match mode.as_str() {
                "crash" => panic!("process-crash worker returned before parent termination"),
                "rotated" | "missing" => {
                    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                    assert!(matches!(
                        entry.coordinator.result([0xC8; 16]),
                        Ok(AnonymousMailboxSourceResult::Armed)
                    ));
                    assert_eq!(entry.resolver.alternate_calls.load(Ordering::Relaxed), 0);
                    assert_eq!(entry.resolver.unknown_calls.load(Ordering::Relaxed), 0);
                }
                "restored" => {
                    assert_eq!(response.status(), StatusCode::OK);
                    let body =
                        axum::body::to_bytes(response.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
                            .await
                            .expect("bounded restored source response");
                    let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket) =
                        completed_terminal_frame(&body)
                    else {
                        panic!("restored process-crash response kind");
                    };
                    ticket
                        .verify_for_request(
                            &fixture.ticket_request,
                            &fixture.target.public_key_bytes(),
                        )
                        .expect("request-bound restored ticket response");
                    assert_eq!(entry.resolver.alternate_calls.load(Ordering::Relaxed), 0);
                    assert_eq!(entry.resolver.unknown_calls.load(Ordering::Relaxed), 0);
                }
                _ => panic!("unknown process-crash worker mode"),
            }
        });
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

    #[tokio::test]
    async fn source_router_wrong_target_receipt_marks_ambiguous_without_completion_or_replay() {
        // [R7-RECEIPT-ZERO-EFFECT 2026-09-07 by Codex] A syntactically valid
        // terminal-shaped response signed by another key must not become a
        // source completion.  The only socket is this test-owned loopback
        // proxy; the response is fabricated before any custody adapter runs.
        let fixture = source_router_fixture();
        let state = source_mpi_state();
        let signer = IdentityKeyPair::from_bytes(&[0xBF; 32]).expect("remote signer");
        let body = submit_body(
            fixture.route_id,
            fixture.commitment,
            &fixture.terminal_frame,
        );
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("wrong-target loopback proxy");
        let client = loopback_proxy_client(&listener);
        let target = fixture.target.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("one proxy connection");
            let (_, peer_body) = read_proxy_request(&mut stream).await;
            let request: PeerBlindRelayRequest =
                serde_json::from_slice(&peer_body).expect("canonical peer request");
            let (response, _) = terminal_peer_response(&peer_body, &target);
            let mut response: PeerBlindRelayResponse =
                serde_json::from_slice(&response).expect("canonical peer response");
            let encoded = response
                .opaque_terminal_response_b64
                .as_deref()
                .expect("sealed terminal response");
            let wrong_target =
                IdentityKeyPair::from_bytes(&[0xC0; 32]).expect("wrong receipt signer");
            response.success_receipt = Some(BlindRelaySuccessReceipt::terminal(
                &request.envelope,
                1,
                None,
                None,
                Some(encoded.as_bytes()),
                unix_now_secs(),
                &wrong_target,
            ));
            write_peer_response(
                &mut stream,
                &serde_json::to_vec(&response).expect("wrong-target peer response"),
            )
            .await;
        });

        let rejected = source_app(state.clone(), &fixture, client)
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                &signer,
            ))
            .await
            .expect("wrong-target source response");
        assert_eq!(rejected.status(), StatusCode::CONFLICT);
        server.await.expect("wrong-target proxy task");
        assert!(matches!(
            fixture.coordinator.result(fixture.route_id),
            Ok(AnonymousMailboxSourceResult::Ambiguous)
        ));
        let exact_calls_before_retry = fixture.resolver.exact_calls.load(Ordering::Relaxed);

        let retry_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("ambiguous retry proxy");
        let retry = source_app(state, &fixture, loopback_proxy_client(&retry_listener))
            .oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &body,
                body.clone(),
                &signer,
            ))
            .await
            .expect("ambiguous retry response");
        assert_eq!(retry.status(), StatusCode::CONFLICT);
        assert_no_proxy_connection(&retry_listener).await;
        assert_eq!(
            fixture.resolver.exact_calls.load(Ordering::Relaxed),
            exact_calls_before_retry,
            "ambiguous retry must not re-resolve or replay"
        );
        assert_eq!(fixture.resolver.unexpected_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn source_router_real_terminal_store_survives_response_loss_and_receiver_entry() {
        // [M13J-E3 2026-09-05 by Codex] This is the source HTTP/real terminal
        // adapter acceptance boundary.  M and R are distinct source journals;
        // each can resolve only the same descriptor-pinned T.  The only socket
        // is the synthetic local proxy carrying the production peer body.
        let now = unix_now_secs();
        let target = IdentityKeyPair::from_bytes(&[0xD1; 32]).expect("target identity");
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            19,
            now.saturating_sub(1),
            now.saturating_add(60),
            "m13j-e3-local-test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        descriptor.public_endpoint = Some("http://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let alternate_node_id = [0xD2; 32];
        let entry_m = real_source_entry(descriptor.clone(), 0xD3, 0xD4, alternate_node_id);
        let entry_r = real_source_entry(descriptor, 0xD5, 0xD6, alternate_node_id);
        let state = source_mpi_state();
        let signer = IdentityKeyPair::from_bytes(&[0xD7; 32]).expect("remote signer");
        let private_directory = tempfile::tempdir().expect("private target store");
        let store_path = std::fs::canonicalize(private_directory.path())
            .expect("canonical target store")
            .join("terminal.sqlite");
        let store_config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: store_path.to_string_lossy().into_owned(),
            max_leases_total: 4,
            max_items_total: 1_024,
            max_bytes_total: 1024 * 1024,
            max_in_flight: 4,
            cleanup_batch_size: 8,
            max_outstanding_tickets: 4,
            max_ticket_issues_per_window: 2,
            ticket_issuance_window_secs: 60,
            ticket_issue_work_bits: 1,
        };
        let depositor = IdentityKeyPair::from_bytes(&[0xD8; 32]).expect("deposit capability");
        let reader = IdentityKeyPair::from_bytes(&[0xD9; 32]).expect("read capability");
        let mailbox_id = [0xDA; 32];
        let lease_expires_at = now.saturating_add(3_600);
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            4,
            16 * 1024,
            now,
            lease_expires_at,
        );
        let ticket_request = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    [0xDB; 16],
                    [0xDC; 16],
                    target.public_key_bytes(),
                    claims,
                    now,
                    now.saturating_add(300),
                    proof_nonce,
                )
                .expect("ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit ticket proof");
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config.clone(),
                target.clone(),
                [0xDD; 32],
            )
            .expect("real SQLite target store"),
        );

        let ticket_route = [0xDE; 16];
        let ticket_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        )
        .expect("ticket terminal");
        let (status, source_body, ticket_peer_body) = source_request_with_real_terminal(
            state.clone(),
            &entry_m,
            target.clone(),
            store.clone(),
            submit_body(ticket_route, commitment, &ticket_terminal),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("ticket response kind");
        };
        ticket_response
            .verify_for_request(&ticket_request, &target.public_key_bytes())
            .expect("request-bound ticket response");
        assert_outer_peer_carrier_is_opaque(&ticket_peer_body, &ticket_terminal, &reader);
        let ticket = ticket_response.ticket.expect("issued ticket");

        let lease = AnonymousMailboxLeaseCreateV1::new(
            mailbox_id,
            depositor.public_key_bytes(),
            4,
            16 * 1024,
            now,
            lease_expires_at,
            ticket,
            &reader,
        )
        .expect("lease request");
        let lease_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
        )
        .expect("lease terminal");
        let (status, source_body, lease_peer_body) = source_request_with_real_terminal(
            state.clone(),
            &entry_m,
            target.clone(),
            store.clone(),
            submit_body([0xDF; 16], commitment, &lease_terminal),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(lease_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("lease response kind");
        };
        assert_eq!(lease_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_outer_peer_carrier_is_opaque(&lease_peer_body, &lease_terminal, &reader);

        let opaque_item = vec![0xE0; 4096];
        let put = AnonymousMailboxPutV1::new(
            mailbox_id,
            [0xE1; 16],
            opaque_item.clone(),
            now,
            now.saturating_add(600),
            &depositor,
        )
        .expect("put request");
        let put_route = [0xE2; 16];
        let put_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Put(put.clone()),
        )
        .expect("put terminal");
        let put_body = submit_body(put_route, commitment, &put_terminal);
        let (status, first_peer_body) = source_request_lost_after_real_terminal(
            state.clone(),
            &entry_m,
            target.clone(),
            store.clone(),
            put_body.clone(),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let armed = entry_m
            .coordinator
            .begin_dispatch(put_route, unix_now_secs())
            .expect("durable exact retry");
        assert_eq!(armed.body(), first_peer_body);
        assert_outer_peer_carrier_is_opaque(&first_peer_body, &put_terminal, &reader);

        // The target committed the Put before the source observed a response.
        // Reopening only durable SQLite state proves the retry remains an exact
        // source body and has one terminal custody effect.
        drop(store);
        let restarted = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config.clone(),
                target.clone(),
                [0xDD; 32],
            )
            .expect("restart real SQLite target store"),
        );
        let (status, source_body, second_peer_body) = source_request_with_real_terminal(
            state.clone(),
            &entry_m,
            target.clone(),
            restarted.clone(),
            put_body,
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            second_peer_body, first_peer_body,
            "retry sends exact peer body"
        );
        let AnonymousMailboxTerminalFrameV1::PutResponse(put_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("put response kind");
        };
        assert_eq!(put_response.outcome, AnonymousMailboxOutcomeV1::Accepted);

        let pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xE3; 16],
            Vec::new(),
            unix_now_secs(),
            &reader,
        )
        .expect("pull request");
        let pull_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(pull),
        )
        .expect("pull terminal");
        let (status, source_body, pull_peer_body) = source_request_with_real_terminal(
            state.clone(),
            &entry_r,
            target.clone(),
            restarted.clone(),
            submit_body([0xE4; 16], commitment, &pull_terminal),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(pull_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("pull response kind");
        };
        assert_eq!(pull_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let pulled = AnonymousMailboxPullResultV1::decode(&pull_response.sealed_payload)
            .expect("real pull result");
        assert_eq!(pulled.item_id, put.item_id);
        assert_eq!(pulled.sealed_commitment, put.sealed_commitment());
        assert_eq!(pulled.sealed_item, opaque_item);
        assert_outer_peer_carrier_is_opaque(&pull_peer_body, &pull_terminal, &reader);

        let ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xE5; 16],
            pulled.item_id,
            pulled.sealed_commitment,
            unix_now_secs(),
            &reader,
        )
        .expect("ack request");
        let ack_terminal =
            encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::Ack(ack))
                .expect("ack terminal");
        let (status, source_body, ack_peer_body) = source_request_with_real_terminal(
            state.clone(),
            &entry_r,
            target.clone(),
            restarted.clone(),
            submit_body([0xE6; 16], commitment, &ack_terminal),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let AnonymousMailboxTerminalFrameV1::AckResponse(ack_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("ack response kind");
        };
        assert_eq!(ack_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_outer_peer_carrier_is_opaque(&ack_peer_body, &ack_terminal, &reader);

        let empty_pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xE7; 16],
            Vec::new(),
            unix_now_secs(),
            &reader,
        )
        .expect("empty pull request");
        let empty_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(empty_pull),
        )
        .expect("empty pull terminal");
        let (status, source_body, empty_peer_body) = source_request_with_real_terminal(
            state,
            &entry_r,
            target,
            restarted,
            submit_body([0xE8; 16], commitment, &empty_terminal),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(empty_response) =
            completed_terminal_frame(&source_body)
        else {
            panic!("empty pull response kind");
        };
        assert_eq!(empty_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(empty_response.sealed_payload.is_empty());
        assert_outer_peer_carrier_is_opaque(&empty_peer_body, &empty_terminal, &reader);

        assert_eq!(entry_m.resolver.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry_r.resolver.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry_m.resolver.unknown_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry_r.resolver.unknown_calls.load(Ordering::Relaxed), 0);
        assert!(entry_m.resolver.exact_calls.load(Ordering::Relaxed) >= 7);
        assert!(entry_r.resolver.exact_calls.load(Ordering::Relaxed) >= 6);
    }

    #[tokio::test]
    async fn source_router_reopened_armed_retry_never_falls_back_after_exact_target_drift() {
        // [E6-EXACT-TARGET-RECOVERY-GUARD 2026-09-05 by Codex] Model the
        // bounded loss boundary after one real terminal mutation, then reopen
        // only S's private journal. A rotated or missing exact descriptor must
        // leave its immutable Armed request off-network; only the unchanged
        // descriptor control may release the already-stored peer body again.
        let now = unix_now_secs();
        let target = IdentityKeyPair::from_bytes(&[0xF7; 32]).expect("target identity");
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            29,
            now.saturating_sub(1),
            now.saturating_add(60),
            "e6-local-test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        descriptor.public_endpoint = Some("http://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let entry = real_source_entry(descriptor.clone(), 0xF8, 0xF9, [0xFA; 32]);
        let state = source_mpi_state();
        let signer = IdentityKeyPair::from_bytes(&[0xFB; 32]).expect("remote signer");
        let private_directory = tempfile::tempdir().expect("private target store");
        let store_path = std::fs::canonicalize(private_directory.path())
            .expect("canonical target store")
            .join("terminal.sqlite");
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                AnonymousMailboxStoreConfig {
                    enabled: true,
                    db_path: store_path.to_string_lossy().into_owned(),
                    max_leases_total: 4,
                    max_items_total: 1_024,
                    max_bytes_total: 1024 * 1024,
                    max_in_flight: 2,
                    cleanup_batch_size: 8,
                    max_outstanding_tickets: 4,
                    max_ticket_issues_per_window: 4,
                    ticket_issuance_window_secs: 60,
                    ticket_issue_work_bits: 1,
                },
                target.clone(),
                [0xFC; 32],
            )
            .expect("real SQLite target store"),
        );
        let ticket_request =
            ticket_request_with_one_bit_proof(&target, [0xFD; 16], [0xFE; 16], [0xFF; 32], now);
        let terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        )
        .expect("ticket terminal");
        let route_id = [0xA0; 16];
        let body = submit_body(route_id, commitment, &terminal);

        let (status, first_peer_body) = source_request_lost_after_real_terminal(
            state.clone(),
            &entry,
            target.clone(),
            store.clone(),
            body.clone(),
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(
            entry.coordinator.result(route_id),
            Ok(AnonymousMailboxSourceResult::Armed)
        ));
        let issued_ticket = match store
            .issue_ticket(&ticket_request, unix_now_secs())
            .expect("read first terminal mutation")
        {
            AnonymousMailboxTicketIssueOutcome::Existing(ticket) => ticket,
            _ => panic!("response loss must not mint a second ticket"),
        };

        let entry = reopen_real_source_entry(entry, descriptor.clone());
        assert_eq!(
            entry
                .coordinator
                .resume(route_id)
                .expect("reopened Armed request")
                .body(),
            first_peer_body
        );

        let mut rotated_body = descriptor.descriptor.clone();
        rotated_body.sequence = rotated_body.sequence.saturating_add(1);
        let rotated =
            SignedNodeDescriptor::sign(rotated_body, &target).expect("rotated descriptor");
        entry.resolver.replace_exact_descriptor(Some(rotated));
        let rotated_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback drift observer");
        let rotated = source_app_for(
            state.clone(),
            Arc::clone(&entry.coordinator),
            &entry.config,
            loopback_proxy_client(&rotated_listener),
        )
        .oneshot(signed_remote_request(
            Method::POST,
            ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
            &body,
            body.clone(),
            &signer,
        ))
        .await
        .expect("drift response");
        assert_eq!(rotated.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_no_proxy_connection(&rotated_listener).await;
        assert!(matches!(
            entry.coordinator.result(route_id),
            Ok(AnonymousMailboxSourceResult::Armed)
        ));

        entry.resolver.replace_exact_descriptor(None);
        let missing_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback missing observer");
        let missing = source_app_for(
            state.clone(),
            Arc::clone(&entry.coordinator),
            &entry.config,
            loopback_proxy_client(&missing_listener),
        )
        .oneshot(signed_remote_request(
            Method::POST,
            ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
            &body,
            body.clone(),
            &signer,
        ))
        .await
        .expect("missing response");
        assert_eq!(missing.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_no_proxy_connection(&missing_listener).await;
        assert!(matches!(
            entry.coordinator.result(route_id),
            Ok(AnonymousMailboxSourceResult::Armed)
        ));
        assert_eq!(entry.resolver.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry.resolver.unknown_calls.load(Ordering::Relaxed), 0);

        entry
            .resolver
            .replace_exact_descriptor(Some(descriptor.clone()));
        let (status, source_body, control_peer_body) = source_request_with_real_terminal(
            state,
            &entry,
            target.clone(),
            store.clone(),
            body,
            &signer,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            control_peer_body, first_peer_body,
            "control retries exact bytes"
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(control_ticket) =
            completed_terminal_frame(&source_body)
        else {
            panic!("control ticket response kind");
        };
        control_ticket
            .verify_for_request(&ticket_request, &target.public_key_bytes())
            .expect("request-bound control ticket response");
        assert!(
            control_ticket.ticket.as_ref() == Some(&issued_ticket),
            "control retry must retain the first terminal ticket"
        );
        assert!(matches!(
            store
                .issue_ticket(&ticket_request, unix_now_secs())
                .expect("read retained terminal ticket"),
            AnonymousMailboxTicketIssueOutcome::Existing(ticket) if ticket == issued_ticket
        ));
        assert!(matches!(
            entry.coordinator.result(route_id),
            Ok(AnonymousMailboxSourceResult::Completed(_))
        ));
        assert_eq!(entry.resolver.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry.resolver.unknown_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn source_router_process_kill_reopens_exact_target_without_fallback() {
        // [E7-PROCESS-CRASH-EVIDENCE 2026-09-06 by Codex] This test kills only
        // its own libtest child after a real terminal SQLite mutation and
        // before that child receives an HTTP response. Replacement libtest
        // children use the same source identity/journal; no host service,
        // fleet node, production configuration or process discovery is used.
        let source_directory = tempfile::tempdir().expect("process-crash source directory");
        let source_root = std::fs::canonicalize(source_directory.path())
            .expect("canonical process-crash source directory");
        let source_db = source_root.join("source.sqlite");
        let target_directory = tempfile::tempdir().expect("process-crash target directory");
        let target_root = std::fs::canonicalize(target_directory.path())
            .expect("canonical process-crash target directory");
        let target_db = target_root.join("terminal.sqlite");
        let std_listener =
            StdTcpListener::bind("127.0.0.1:0").expect("process-crash loopback listener");
        std_listener
            .set_nonblocking(true)
            .expect("nonblocking process-crash listener");
        let inspection_listener = std_listener
            .try_clone()
            .expect("clone process-crash inspection listener");
        let listener =
            TcpListener::from_std(std_listener).expect("tokio process-crash loopback listener");
        let fixture_now = unix_now_secs();
        let fixture = process_crash_fixture(fixture_now);
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                AnonymousMailboxStoreConfig {
                    enabled: true,
                    db_path: target_db.to_string_lossy().into_owned(),
                    max_leases_total: 4,
                    max_items_total: 1_024,
                    max_bytes_total: 1024 * 1024,
                    max_in_flight: 2,
                    cleanup_batch_size: 8,
                    max_outstanding_tickets: 4,
                    max_ticket_issues_per_window: 4,
                    ticket_issuance_window_secs: 60,
                    ticket_issue_work_bits: 1,
                },
                fixture.target.clone(),
                [0xCD; 32],
            )
            .expect("process-crash real SQLite target store"),
        );
        let proxy_port = listener
            .local_addr()
            .expect("process-crash loopback address")
            .port();
        let (committed_rx, release_tx, control_tx, server) = process_crash_terminal_proxy(
            listener,
            inspection_listener,
            fixture.target.clone(),
            store.clone(),
        );
        let mut server = Some(server);

        let scenario = async {
            let mut crashed =
                spawn_process_crash_child("crash", &source_db, proxy_port, fixture_now);
            let first_peer_body =
                match tokio::time::timeout(PROCESS_CRASH_DEADLINE, committed_rx).await {
                    Ok(Ok(body)) => body,
                    Ok(Err(_)) => {
                        terminate_and_reap_process_crash_child(&mut crashed).await;
                        return Err("terminal did not report its durable mutation".to_string());
                    }
                    Err(_) => {
                        terminate_and_reap_process_crash_child(&mut crashed).await;
                        return Err(
                            "terminal durable-mutation event exceeded bounded deadline".to_string()
                        );
                    }
                };
            kill_and_reap_process_crash_child(&mut crashed).await?;
            release_tx
                .send(())
                .map_err(|_| "process-crash terminal release failed".to_string())?;

            for mode in ["rotated", "missing"] {
                let mut replacement =
                    spawn_process_crash_child(mode, &source_db, proxy_port, fixture_now);
                require_process_crash_child_success(&mut replacement).await?;
                request_process_crash_quiet(&control_tx).await?;
            }

            let mut restored =
                spawn_process_crash_child("restored", &source_db, proxy_port, fixture_now);
            let (restored_tx, restored_rx) = tokio::sync::oneshot::channel();
            if control_tx
                .send(ProcessCrashTerminalControl::ServeRestored(restored_tx))
                .await
                .is_err()
            {
                terminate_and_reap_process_crash_child(&mut restored).await;
                return Err("process-crash terminal restore control failed".into());
            }
            let restored_peer_body =
                match tokio::time::timeout(PROCESS_CRASH_DEADLINE, restored_rx).await {
                    Ok(Ok(body)) => body,
                    Ok(Err(_)) => {
                        terminate_and_reap_process_crash_child(&mut restored).await;
                        return Err("process-crash terminal dropped restored body".into());
                    }
                    Err(_) => {
                        terminate_and_reap_process_crash_child(&mut restored).await;
                        return Err("restored terminal request exceeded bounded deadline".into());
                    }
                };
            require_process_crash_child_success(&mut restored).await?;
            if restored_peer_body != first_peer_body {
                return Err("replacement source body was not byte-identical".into());
            }
            if !matches!(
                store
                    .issue_ticket(&fixture.ticket_request, unix_now_secs())
                    .expect("read process-crash terminal ticket"),
                AnonymousMailboxTicketIssueOutcome::Existing(_)
            ) {
                return Err("restored terminal retry minted a second ticket".into());
            }
            Ok::<(), String>(())
        }
        .await;

        match scenario {
            Ok(()) => {
                if let Err(error) = require_process_crash_server_complete(&mut server).await {
                    abort_and_reap_process_crash_server(&mut server).await;
                    panic!("{error}");
                }
            }
            Err(error) => {
                abort_and_reap_process_crash_server(&mut server).await;
                panic!("{error}");
            }
        }
    }

    #[tokio::test]
    async fn source_router_cancellation_releases_bounded_admission_without_replaying_effect() {
        // [M13J-E4 2026-09-05 by Codex] Exercise only request-future
        // cancellation at the source handler boundary. This is deliberately
        // not a process shutdown, service drain, onion-middle, or fleet test.
        let now = unix_now_secs();
        let target = IdentityKeyPair::from_bytes(&[0xE9; 32]).expect("target identity");
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            23,
            now.saturating_sub(1),
            now.saturating_add(60),
            "m13j-e4-local-test",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        descriptor.public_endpoint = Some("http://8.8.8.8".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let entry = real_source_entry(descriptor, 0xEA, 0xEB, [0xEC; 32]);
        let mut constrained_config = entry.config.clone();
        constrained_config.max_in_flight = 1;
        constrained_config.request_timeout_secs = 10;
        let state = source_mpi_state();
        let signer = IdentityKeyPair::from_bytes(&[0xED; 32]).expect("remote signer");
        let private_directory = tempfile::tempdir().expect("private target store");
        let store_path = std::fs::canonicalize(private_directory.path())
            .expect("canonical target store")
            .join("terminal.sqlite");
        let store_config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: store_path.to_string_lossy().into_owned(),
            max_leases_total: 4,
            max_items_total: 1_024,
            max_bytes_total: 1024 * 1024,
            max_in_flight: 4,
            cleanup_batch_size: 8,
            max_outstanding_tickets: 4,
            max_ticket_issues_per_window: 4,
            ticket_issuance_window_secs: 60,
            ticket_issue_work_bits: 1,
        };
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config,
                target.clone(),
                [0xEE; 32],
            )
            .expect("real SQLite target store"),
        );
        let first_ticket =
            ticket_request_with_one_bit_proof(&target, [0xEF; 16], [0xF0; 16], [0xF1; 32], now);
        let first_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(first_ticket.clone()),
        )
        .expect("first ticket terminal");
        let first_route = [0xF2; 16];
        let first_body = submit_body(first_route, commitment, &first_terminal);
        let later_ticket =
            ticket_request_with_one_bit_proof(&target, [0xF3; 16], [0xF4; 16], [0xF5; 32], now);
        let later_terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(later_ticket.clone()),
        )
        .expect("later ticket terminal");
        let later_route = [0xF6; 16];
        let later_body = submit_body(later_route, commitment, &later_terminal);
        let accepted_connections = Arc::new(AtomicUsize::new(0));
        let (client, committed, release, server) = held_real_terminal_proxy(
            target.clone(),
            store.clone(),
            Arc::clone(&accepted_connections),
        )
        .await;
        let app = source_app_for(
            state,
            Arc::clone(&entry.coordinator),
            &constrained_config,
            client,
        );
        let first_request = signed_remote_request(
            Method::POST,
            ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
            &first_body,
            first_body.clone(),
            &signer,
        );
        let first_app = app.clone();
        let first_source = tokio::spawn(async move { first_app.oneshot(first_request).await });

        let first_peer_body = tokio::time::timeout(Duration::from_secs(2), committed)
            .await
            .expect("first terminal effect supervision")
            .expect("first terminal effect signal");
        assert!(matches!(
            entry.coordinator.result(first_route),
            Ok(AnonymousMailboxSourceResult::Armed)
        ));
        assert_eq!(accepted_connections.load(Ordering::Relaxed), 1);

        let exact_before_busy = entry.resolver.exact_calls.load(Ordering::Relaxed);
        let busy = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &later_body,
                later_body.clone(),
                &signer,
            )),
        )
        .await
        .expect("busy admission supervision")
        .expect("busy admission response");
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(matches!(
            entry.coordinator.result(later_route),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
        assert_eq!(
            entry.resolver.exact_calls.load(Ordering::Relaxed),
            exact_before_busy
        );
        assert_eq!(accepted_connections.load(Ordering::Relaxed), 1);

        first_source.abort();
        assert!(tokio::time::timeout(Duration::from_secs(2), first_source)
            .await
            .expect("source cancellation supervision")
            .expect_err("test-owned source future must cancel")
            .is_cancelled());
        release
            .send(())
            .expect("release held test-owned proxy after cancellation");
        assert!(matches!(
            entry.coordinator.result(first_route),
            Ok(AnonymousMailboxSourceResult::Armed)
        ));

        let retry = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &first_body,
                first_body.clone(),
                &signer,
            )),
        )
        .await
        .expect("exact retry supervision")
        .expect("exact retry response");
        assert_eq!(retry.status(), StatusCode::OK);
        let retry_body = axum::body::to_bytes(retry.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
            .await
            .expect("bounded exact retry response");
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(retry_ticket) =
            completed_terminal_frame(&retry_body)
        else {
            panic!("exact retry ticket response kind");
        };
        retry_ticket
            .verify_for_request(&first_ticket, &target.public_key_bytes())
            .expect("request-bound exact retry receipt");
        let expected_first_ticket = match store
            .issue_ticket(&first_ticket, unix_now_secs())
            .expect("read existing first ticket")
        {
            AnonymousMailboxTicketIssueOutcome::Existing(ticket) => ticket,
            _ => panic!("cancelled request must not issue a second ticket"),
        };
        assert!(retry_ticket.ticket.as_ref() == Some(&expected_first_ticket));
        assert!(matches!(
            entry.coordinator.result(first_route),
            Ok(AnonymousMailboxSourceResult::Completed(_))
        ));

        let later = tokio::time::timeout(
            Duration::from_secs(2),
            app.oneshot(signed_remote_request(
                Method::POST,
                ANONYMOUS_MAILBOX_SOURCE_SUBMIT_PATH,
                &later_body,
                later_body.clone(),
                &signer,
            )),
        )
        .await
        .expect("post-cancellation admission supervision")
        .expect("post-cancellation admission response");
        assert_eq!(later.status(), StatusCode::OK);
        let later_response = axum::body::to_bytes(later.into_body(), SOURCE_SUBMIT_BODY_MAX_BYTES)
            .await
            .expect("bounded later response");
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(later_ticket_response) =
            completed_terminal_frame(&later_response)
        else {
            panic!("later ticket response kind");
        };
        later_ticket_response
            .verify_for_request(&later_ticket, &target.public_key_bytes())
            .expect("request-bound later receipt");
        assert!(later_ticket_response.ticket.is_some());
        assert!(matches!(
            entry.coordinator.result(later_route),
            Ok(AnonymousMailboxSourceResult::Completed(_))
        ));

        let peer_bodies = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .expect("proxy completion supervision")
            .expect("proxy task");
        assert_eq!(peer_bodies.len(), 3);
        assert_eq!(peer_bodies[0], first_peer_body);
        assert_eq!(peer_bodies[0], peer_bodies[1]);
        assert_ne!(peer_bodies[1], peer_bodies[2]);
        assert_eq!(accepted_connections.load(Ordering::Relaxed), 3);
        assert_eq!(entry.resolver.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(entry.resolver.unknown_calls.load(Ordering::Relaxed), 0);
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
