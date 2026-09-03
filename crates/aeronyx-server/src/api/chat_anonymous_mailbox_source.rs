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
    use std::sync::Arc;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxTerminalFrameV1,
        AnonymousMailboxTicketIssueV1,
    };
    use aeronyx_core::protocol::chat::BlindRelaySuccessReceipt;
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeDescriptor, NodeProtocolFeature, SignedNodeDescriptor,
    };
    use parking_lot::Mutex;
    use rusqlite::Connection;

    use super::*;
    use crate::api::chat_peer::PeerBlindRelayResponse;
    use crate::services::chat_relay_anonymous_mailbox_source::{
        ExactAnonymousMailboxTargetResolver, SqliteAnonymousMailboxSourceJournal,
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
}
