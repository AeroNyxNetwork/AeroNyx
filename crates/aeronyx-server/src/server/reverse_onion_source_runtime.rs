// ============================================
// File: crates/aeronyx-server/src/server/reverse_onion_source_runtime.rs
// ============================================
//! Dormant source-side carrier for the fixed-class private Blind Vault Pull.
//!
//! [REVERSE-ONION-SOURCE-RUNTIME 2026-10-04 by Codex] This module is an
//! explicitly constructed capability. It is not registered by this change:
//! router, startup, configuration, and PeerStore composition remain owned by
//! the integration caller. The carrier keeps the source journal as the only
//! effect authority, never accepts a caller-selected endpoint, and never
//! retries a POST after its Prepared->Armed CAS.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::blind_vault::BlindVaultOnionPullSession;
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, SignedNodeDescriptor,
    SignedPrivateOnionRecipientAuthorizationV1,
};
use aeronyx_core::protocol::onion::OnionRoutePurpose;
use aeronyx_core::protocol::onion_reply::decode_onion_sealed_response;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionSourceEvidenceV1, ReverseOnionSourceQueryV1, SourceEvidencePartV1,
    SourceEvidenceStateV1, VerifiedSourceEvidenceChain, REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS,
    MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES,
};
use async_trait::async_trait;
use axum::body::Bytes;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::api::{
    canonical_peer_http_url, peer_endpoint_is_public_ip, privacy_safe_peer_http_client_builder,
    read_bounded_http_response, BoundedHttpResponseError, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES,
};
use crate::api::chat_peer::{
    blind_relay_authenticated_request_commitment, PeerBlindRelayRequest, PeerBlindRelayResponse,
};
use crate::services::reverse_onion_source::{
    ExpectedRetainedEnvelope, ReverseOnionSourceJournal, SourceJournalError, SourcePhase,
    SourcePreparedPull, SourceRecoveryMetadata,
};

const BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";
const SOURCE_QUERY_PATH: &str = "/api/chat/peer/reverse-onion/source-query";
const MAX_SOURCE_DISPATCH_BYTES: usize = 64 * 1024;
const MAX_SOURCE_TERMINAL_BYTES: usize = 512;
const MAX_SOURCE_IN_FLIGHT: usize = 64;
const DEFAULT_SOURCE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_SOURCE_TIMEOUT: Duration = Duration::from_secs(60);

/// Coarse runtime failures. No variant carries an endpoint, node id, body, or
/// peer-controlled diagnostic string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SourceRuntimeError {
    #[error("source runtime is stopped")]
    Stopped,
    #[error("source runtime is busy")]
    Busy,
    #[error("source runtime is unavailable")]
    Unavailable,
    #[error("source runtime is ambiguous")]
    Ambiguous,
    #[error("source runtime rejected")]
    Rejected,
    #[error("source runtime conflict")]
    Conflict,
    #[error("source runtime expired")]
    Expired,
}

impl From<SourceJournalError> for SourceRuntimeError {
    fn from(error: SourceJournalError) -> Self {
        match error {
            SourceJournalError::Busy => Self::Busy,
            SourceJournalError::Ambiguous => Self::Ambiguous,
            SourceJournalError::Conflict => Self::Conflict,
            SourceJournalError::Expired => Self::Expired,
            SourceJournalError::Unavailable | SourceJournalError::Corrupt
            | SourceJournalError::ClockRollback | SourceJournalError::Capacity
            | SourceJournalError::MigrationRequired => Self::Unavailable,
            SourceJournalError::Rejected | SourceJournalError::ReplyRejected => Self::Rejected,
        }
    }
}

/// Fixed source-side timeout and in-flight policy. Zero or oversized values
/// are rejected before a client, task, or semaphore is constructed.
#[derive(Clone, Copy)]
pub(crate) struct SourceRuntimeConfig {
    pub(crate) max_in_flight: usize,
    pub(crate) timeout: Duration,
}

impl SourceRuntimeConfig {
    pub(crate) fn new(max_in_flight: usize, timeout: Duration) -> Result<Self, SourceRuntimeError> {
        if max_in_flight == 0
            || max_in_flight > MAX_SOURCE_IN_FLIGHT
            || timeout.is_zero()
            || timeout > MAX_SOURCE_TIMEOUT
        {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self { max_in_flight, timeout })
    }
}

/// Signed, pinned R/P authority and its exact source identity. No Debug is
/// implemented: descriptors contain public endpoints and must not be logged.
pub(crate) struct SourcePinnedRelayPolicy {
    source: [u8; 32],
    relay: SignedNodeDescriptor,
    recipient: SignedNodeDescriptor,
    authorization: SignedPrivateOnionRecipientAuthorizationV1,
}

impl SourcePinnedRelayPolicy {
    pub(crate) fn new(
        source: [u8; 32],
        relay: SignedNodeDescriptor,
        recipient: SignedNodeDescriptor,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<Self, SourceRuntimeError> {
        if source == [0; 32]
            || relay.node_id() == [0; 32]
            || recipient.node_id() == [0; 32]
            || source == relay.node_id()
            || source == recipient.node_id()
            || relay.node_id() == recipient.node_id()
            || authorization
                .verify_at(
                    &relay,
                    &recipient,
                    OnionRoutePurpose::BlindVaultPull.as_str(),
                    now,
                )
                .is_err()
            || relay.verify_at(now).is_err()
            || recipient.verify_at(now).is_err()
            || !relay
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            || !relay
                .descriptor
                .capabilities
                .contains(&NodeCapability::BlindVaultReplica)
            || !recipient
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            || !recipient
                .descriptor
                .capabilities
                .contains(&NodeCapability::BlindVaultReplica)
            || !has_required_features(&relay)
            || !has_required_features(&recipient)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let endpoint = relay
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        if !peer_endpoint_is_public_ip(endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self {
            source,
            relay,
            recipient,
            authorization,
        })
    }

    fn validate_at(&self, now: u64) -> Result<(), SourceRuntimeError> {
        self.relay
            .verify_at(now)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        self.recipient
            .verify_at(now)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        self.authorization
            .verify_at(
                &self.relay,
                &self.recipient,
                OnionRoutePurpose::BlindVaultPull.as_str(),
                now,
            )
            .map_err(|_| SourceRuntimeError::Rejected)?;
        let endpoint = self
            .relay
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        if !peer_endpoint_is_public_ip(endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(())
    }

    fn relay_id(&self) -> [u8; 32] {
        self.relay.node_id()
    }

    fn recipient_id(&self) -> [u8; 32] {
        self.recipient.node_id()
    }

    fn descriptor_commitment(&self) -> Result<[u8; 32], SourceRuntimeError> {
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&self.recipient)
            .map(|commitment| commitment.hash())
            .map_err(|_| SourceRuntimeError::Rejected)
    }

    fn relay_descriptor_commitment(&self) -> Result<[u8; 32], SourceRuntimeError> {
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&self.relay)
            .map(|commitment| commitment.hash())
            .map_err(|_| SourceRuntimeError::Rejected)
    }

    fn relay_url(&self, path: &str) -> Result<reqwest::Url, SourceRuntimeError> {
        let endpoint = self
            .relay
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        canonical_peer_http_url(endpoint, path).map_err(|_| SourceRuntimeError::Rejected)
    }

    fn authority(
        &self,
    ) -> Result<crate::services::reverse_onion_source::SourceRouteAuthority, SourceRuntimeError> {
        crate::services::reverse_onion_source::SourceRouteAuthority::from_signed(
            &self.relay,
            &self.recipient,
            &self.authorization,
            OnionRoutePurpose::BlindVaultPull.as_str(),
        )
        .map_err(SourceRuntimeError::from)
    }
}

fn has_required_features(descriptor: &SignedNodeDescriptor) -> bool {
    let terminal = OnionRoutePurpose::BlindVaultPull.required_terminal_protocol_features();
    let path = OnionRoutePurpose::BlindVaultPull.required_path_protocol_features();
    terminal
        .iter()
        .chain(path.iter())
        .all(|feature| descriptor.descriptor.advertises_protocol_feature(*feature))
}

pub(crate) enum SourceTransportOutcome {
    Response { status: u16, body: Vec<u8> },
    Ambiguous,
}

#[async_trait]
pub(crate) trait SourceTransport: Send + Sync {
    async fn post(&self, url: reqwest::Url, body: Bytes) -> SourceTransportOutcome;
    async fn query(&self, url: reqwest::Url, body: Bytes) -> SourceTransportOutcome;
}

/// Production transport. It receives typed canonical URLs only and never
/// follows redirects or inherits an environment proxy.
pub(crate) struct ReqwestSourceTransport {
    client: reqwest::Client,
}

impl ReqwestSourceTransport {
    pub(crate) fn new(timeout: Duration) -> Result<Self, SourceRuntimeError> {
        let client = privacy_safe_peer_http_client_builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|_| SourceRuntimeError::Unavailable)?;
        Ok(Self { client })
    }

    async fn send(
        &self,
        url: reqwest::Url,
        body: Bytes,
        limit: usize,
        content_type: &'static str,
    ) -> SourceTransportOutcome {
        let response = match self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => return SourceTransportOutcome::Ambiguous,
        };
        let status = response.status().as_u16();
        match read_bounded_http_response(response, limit).await {
            Ok(body) => SourceTransportOutcome::Response { status, body },
            Err(BoundedHttpResponseError::TooLarge | BoundedHttpResponseError::BodyRead) => {
                SourceTransportOutcome::Ambiguous
            }
            Err(BoundedHttpResponseError::JsonDecode) => SourceTransportOutcome::Ambiguous,
        }
    }
}

#[async_trait]
impl SourceTransport for ReqwestSourceTransport {
    async fn post(&self, url: reqwest::Url, body: Bytes) -> SourceTransportOutcome {
        self.send(url, body, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES, "application/json")
            .await
    }

    async fn query(&self, url: reqwest::Url, body: Bytes) -> SourceTransportOutcome {
        self.send(
            url,
            body,
            MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES,
            "application/octet-stream",
        )
        .await
    }
}

/// Source carrier with one bounded in-flight class and an explicit stop bit.
/// The journal is opened and owned by composition; this object never opens a
/// second database or creates an async task that outlives its caller.
pub(crate) struct ReverseOnionSourceRuntime {
    journal: Arc<ReverseOnionSourceJournal>,
    identity: Arc<IdentityKeyPair>,
    policy: Arc<SourcePinnedRelayPolicy>,
    transport: Arc<dyn SourceTransport>,
    permits: Arc<Semaphore>,
    timeout: Duration,
    stopped: Arc<AtomicBool>,
}

impl ReverseOnionSourceRuntime {
    pub(crate) fn new(
        journal: Arc<ReverseOnionSourceJournal>,
        identity: Arc<IdentityKeyPair>,
        policy: Arc<SourcePinnedRelayPolicy>,
        transport: Arc<dyn SourceTransport>,
        config: SourceRuntimeConfig,
    ) -> Result<Self, SourceRuntimeError> {
        if identity.public_key_bytes() != policy.source {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self {
            journal,
            identity,
            policy,
            transport,
            permits: Arc::new(Semaphore::new(config.max_in_flight)),
            timeout: config.timeout,
            stopped: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Arms once, sends the exact durable JSON body once, then recovers the
    /// complete three-part evidence chain. Any transport/semantic uncertainty
    /// remains Ambiguous and never creates a second POST or new route.
    pub(crate) async fn dispatch(
        &self,
        request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope,
        session: BlindVaultOnionPullSession,
        admitted_deadline: u64,
        terminal_request: Vec<u8>,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        if terminal_request.is_empty() || terminal_request.len() > MAX_SOURCE_TERMINAL_BYTES {
            return Err(SourceRuntimeError::Rejected);
        }
        let permit = Arc::new(
            self.permits
                .clone()
                .try_acquire_owned()
                .map_err(|_| SourceRuntimeError::Busy)?,
        );
        let admission = self.prepare_and_arm(
            request,
            expected,
            session,
            admitted_deadline,
            terminal_request,
            now,
            permit,
        ).await?;
        let (request, route, deadline, url, exact_bytes) = match admission {
            SourceAdmission::Send(admission) => (
                admission.request,
                admission.route,
                admission.deadline,
                admission.url,
                admission.exact_bytes,
            ),
            SourceAdmission::Recover { route } => {
                return self.collect_and_open(route, now).await;
            }
            SourceAdmission::Complete { route } => return self.read_verified(route, now).await,
        };
        if exact_bytes.len() > MAX_SOURCE_DISPATCH_BYTES {
            self.mark_ambiguous(route, now).await;
            return Err(SourceRuntimeError::Rejected);
        }
        let response = match tokio::time::timeout(
            self.timeout,
            self.transport.post(url, Bytes::from(exact_bytes)),
        )
        .await
        {
            Ok(response) => response,
            Err(_) => SourceTransportOutcome::Ambiguous,
        };
        match response {
            SourceTransportOutcome::Ambiguous => {
                self.mark_ambiguous(route, now).await;
                Err(SourceRuntimeError::Ambiguous)
            }
            SourceTransportOutcome::Response { status, body } => {
                let request_for_verify = request;
                let relay = self.policy.relay_id();
                let response = match tokio::task::spawn_blocking(move || {
                    let response = decode_peer_response(status, &body)?;
                    verify_success_response(&request_for_verify, &response, relay)?;
                    Ok::<_, SourceRuntimeError>(response)
                })
                .await
                .map_err(|_| SourceRuntimeError::Unavailable)?
                {
                    Ok(response) => response,
                    Err(error) => {
                        self.mark_ambiguous(route, now).await;
                        return Err(error);
                    }
                };
                let _ = response;
                self.collect_and_open(route, now).await
            }
        }
    }

    async fn prepare_and_arm(
        &self,
        request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope,
        session: BlindVaultOnionPullSession,
        admitted_deadline: u64,
        terminal_request: Vec<u8>,
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<SourceAdmission, SourceRuntimeError> {
        let policy = Arc::clone(&self.policy);
        let journal = Arc::clone(&self.journal);
        let identity = Arc::clone(&self.identity);
        let request_for_plan = request.clone();
        let route = request_for_plan.envelope.route_id;
        let blocking_permit = Arc::clone(&permit);
        let result = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            policy.validate_at(now)?;
            if request_for_plan.previous_hop_node_id != identity.public_key_bytes()
                || request_for_plan.envelope.next_hop != policy.relay_id()
            {
                return Err(SourceRuntimeError::Rejected);
            }
            let authority = policy.authority()?;
            let descriptor = policy.descriptor_commitment()?;
            let relay_descriptor = policy.relay_descriptor_commitment()?;
            let request_commitment = blind_relay_authenticated_request_commitment(&request_for_plan)
                .map_err(|_| SourceRuntimeError::Rejected)?;
            let plan = SourcePreparedPull::from_runtime_admission(
                &identity,
                request_for_plan.clone(),
                expected,
                policy.recipient_id(),
                descriptor,
                relay_descriptor,
                descriptor,
                request_commitment,
                authority,
                admitted_deadline,
                terminal_request,
            )
            .map_err(SourceRuntimeError::from)?;
            let phase = journal
                .prepare(plan, session, now)
                .map_err(SourceRuntimeError::from)?;
            match phase {
                SourcePhase::Prepared => {
                    let dispatch = journal.arm(request_for_plan.envelope.route_id, now)
                        .map_err(SourceRuntimeError::from)?;
                    let url = policy.relay_url(BLIND_RELAY_PATH)?;
                    Ok(SourceAdmission::Send(PreparedDispatch {
                        exact_bytes: dispatch.exact_bytes,
                        request: request_for_plan,
                        route,
                        deadline: admitted_deadline,
                        url,
                    }))
                }
                SourcePhase::Armed | SourcePhase::DispatchAmbiguous | SourcePhase::ResultReady => {
                    Ok(SourceAdmission::Recover { route })
                }
                SourcePhase::Verified => Ok(SourceAdmission::Complete { route }),
                SourcePhase::Rejected | SourcePhase::Opening | SourcePhase::OpenAmbiguous => {
                    Err(SourceRuntimeError::Ambiguous)
                }
            }
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)??;
        Ok(result)
    }

    async fn mark_ambiguous(&self, route: [u8; 16], now: u64) {
        let journal = Arc::clone(&self.journal);
        let _ = tokio::task::spawn_blocking(move || journal.mark_dispatch_ambiguous(route, now)).await;
    }

    async fn collect_and_open(
        &self,
        route: [u8; 16],
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let metadata = self.metadata(route, now).await?;
        let deadline = metadata.deadline();
        if !matches!(metadata.phase(), SourcePhase::Armed | SourcePhase::DispatchAmbiguous) {
            return match metadata.phase() {
                SourcePhase::Verified => self.read_verified(route, now).await,
                SourcePhase::ResultReady => self.open_result(route, now).await,
                SourcePhase::Rejected => Err(SourceRuntimeError::Rejected),
                _ => Err(SourceRuntimeError::Ambiguous),
            };
        }
        let mut parts = Vec::with_capacity(3);
        for part in [SourceEvidencePartV1::Claim, SourceEvidencePartV1::Lease, SourceEvidencePartV1::Result] {
            let query = self.sign_query(route, metadata.request_commitment(), part, now).await?;
            let body = query.encode();
            let url = self.policy.relay_url(SOURCE_QUERY_PATH)?;
            let response = match tokio::time::timeout(
                self.timeout,
                self.transport.query(url, Bytes::from(body)),
            )
            .await
            {
                Ok(response) => response,
                Err(_) => SourceTransportOutcome::Ambiguous,
            };
            let bytes = match response {
                SourceTransportOutcome::Response { status: 200, body } => body,
                SourceTransportOutcome::Response { .. } | SourceTransportOutcome::Ambiguous => {
                    self.mark_ambiguous(route, now).await;
                    return Err(SourceRuntimeError::Ambiguous);
                }
            };
            if bytes.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                self.mark_ambiguous(route, now).await;
                return Err(SourceRuntimeError::Ambiguous);
            }
            let verified = tokio::task::spawn_blocking(move || {
                let evidence = ReverseOnionSourceEvidenceV1::decode(&bytes)
                    .map_err(|_| SourceRuntimeError::Ambiguous)?;
                let evidence_state = evidence
                    .verify_for_query(&query, now)
                    .map_err(|_| SourceRuntimeError::Ambiguous)?;
                if evidence_state.state() != SourceEvidenceStateV1::Available {
                    return Err(SourceRuntimeError::Ambiguous);
                }
                Ok::<_, SourceRuntimeError>((query, evidence))
            })
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)?;
            let (query, evidence) = match verified {
                Ok(pair) => pair,
                Err(_) => {
                    self.mark_ambiguous(route, now).await;
                    return Err(SourceRuntimeError::Ambiguous);
                }
            };
            if evidence.encode().len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                self.mark_ambiguous(route, now).await;
                return Err(SourceRuntimeError::Ambiguous);
            }
            parts.push((query, evidence));
        }
        let chain = tokio::task::spawn_blocking(move || {
            let verified = [
                parts[0].1.verify_for_query(&parts[0].0, now).map_err(|_| SourceRuntimeError::Ambiguous)?,
                parts[1].1.verify_for_query(&parts[1].0, now).map_err(|_| SourceRuntimeError::Ambiguous)?,
                parts[2].1.verify_for_query(&parts[2].0, now).map_err(|_| SourceRuntimeError::Ambiguous)?,
            ];
            VerifiedSourceEvidenceChain::verify([&verified[0], &verified[1], &verified[2]], deadline, now)
                .map_err(|_| SourceRuntimeError::Ambiguous)
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)??;
        let journal = Arc::clone(&self.journal);
        let opened = tokio::task::spawn_blocking(move || {
            journal.record_result(route, chain.claim(), chain.lease(), chain.result(), now)
                .map_err(SourceRuntimeError::from)?;
            journal.open_result(route, now).map_err(SourceRuntimeError::from)
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)??;
        Ok(BlindVaultPullResult { response: opened })
    }

    async fn sign_query(
        &self,
        route: [u8; 16],
        request_commitment: [u8; 32],
        part: SourceEvidencePartV1,
        now: u64,
    ) -> Result<ReverseOnionSourceQueryV1, SourceRuntimeError> {
        let identity = Arc::clone(&self.identity);
        let relay = self.policy.relay_id();
        tokio::task::spawn_blocking(move || {
            let mut nonce = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut nonce);
            if nonce == [0; 32] {
                return Err(SourceRuntimeError::Unavailable);
            }
            ReverseOnionSourceQueryV1::sign(
                &identity,
                relay,
                route,
                request_commitment,
                part,
                nonce,
                now,
                now.checked_add(REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS)
                    .ok_or(SourceRuntimeError::Rejected)?,
            )
            .map_err(|_| SourceRuntimeError::Rejected)
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)?
    }

    async fn metadata(&self, route: [u8; 16], now: u64) -> Result<SourceRecoveryMetadata, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let page = tokio::task::spawn_blocking(move || {
            let mut after = None;
            for _ in 0..16 {
                let page = journal.recover_metadata(after, 64, now).map_err(SourceRuntimeError::from)?;
                for item in page.items {
                    if item.route() == route {
                        return Ok(item);
                    }
                }
                let Some(next) = page.next_after else { break; };
                after = Some(next);
            }
            Err(SourceRuntimeError::Rejected)
        })
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)??;
        Ok(page)
    }

    async fn read_verified(&self, route: [u8; 16], now: u64) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let response = tokio::task::spawn_blocking(move || journal.read_verified(route, now))
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)?
            .map_err(SourceRuntimeError::from)?;
        Ok(BlindVaultPullResult { response })
    }

    async fn open_result(&self, route: [u8; 16], now: u64) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let response = tokio::task::spawn_blocking(move || journal.open_result(route, now))
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)?
            .map_err(SourceRuntimeError::from)?;
        Ok(BlindVaultPullResult { response })
    }
}

enum SourceAdmission {
    Send(PreparedDispatch),
    Recover { route: [u8; 16] },
    Complete { route: [u8; 16] },
}

struct PreparedDispatch {
    exact_bytes: Vec<u8>,
    request: PeerBlindRelayRequest,
    route: [u8; 16],
    deadline: u64,
    url: reqwest::Url,
}

/// The successful source-side result is intentionally typed; callers cannot
/// inspect or replay the encrypted terminal bytes outside the journal's
/// one-shot opening transition.
pub(crate) struct BlindVaultPullResult {
    response: aeronyx_core::protocol::blind_vault::BlindVaultPullResponse,
}

impl BlindVaultPullResult {
    pub(crate) fn response(&self) -> &aeronyx_core::protocol::blind_vault::BlindVaultPullResponse {
        &self.response
    }
}

fn decode_peer_response(status: u16, body: &[u8]) -> Result<PeerBlindRelayResponse, SourceRuntimeError> {
    if status != 200 || body.len() > BLIND_RELAY_ACK_RESPONSE_MAX_BYTES {
        return Err(SourceRuntimeError::Ambiguous);
    }
    serde_json::from_slice(body).map_err(|_| SourceRuntimeError::Ambiguous)
}

fn verify_success_response(
    request: &PeerBlindRelayRequest,
    response: &PeerBlindRelayResponse,
    relay: [u8; 32],
) -> Result<(), SourceRuntimeError> {
    if !response.accepted || !response.terminal || response.forwarded || response.delivery_receipt.is_some() {
        return Err(SourceRuntimeError::Ambiguous);
    }
    if response.opaque_terminal_response_b64.is_none() {
        return Err(SourceRuntimeError::Ambiguous);
    }
    let opaque = response
        .opaque_terminal_response_b64
        .as_deref()
        .ok_or(SourceRuntimeError::Ambiguous)?;
    if opaque.len() > aeronyx_core::protocol::MAX_ONION_SEALED_RESPONSE_BASE64_BYTES {
        return Err(SourceRuntimeError::Ambiguous);
    }
    let opaque = BASE64.decode(opaque).map_err(|_| SourceRuntimeError::Ambiguous)?;
    if opaque.is_empty() || opaque.len() > aeronyx_core::protocol::MAX_ONION_SEALED_RESPONSE_BYTES {
        return Err(SourceRuntimeError::Ambiguous);
    }
    decode_onion_sealed_response(&opaque).map_err(|_| SourceRuntimeError::Ambiguous)?;
    let receipt = response
        .success_receipt
        .as_ref()
        .ok_or(SourceRuntimeError::Ambiguous)?;
    receipt
        .verify_expected(
            &request.envelope,
            true,
            false,
            response.ttl_remaining,
            response.reason.as_deref(),
            None,
            Some(&opaque),
            &relay,
        )
        .map_err(|_| SourceRuntimeError::Ambiguous)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_timeout_is_bounded_without_constructing_network_state() {
        let config = SourceRuntimeConfig::new(1, DEFAULT_SOURCE_TIMEOUT).unwrap();
        assert_eq!(config.max_in_flight, 1);
        assert!(SourceRuntimeConfig::new(0, DEFAULT_SOURCE_TIMEOUT).is_err());
        assert!(SourceRuntimeConfig::new(1, MAX_SOURCE_TIMEOUT + Duration::from_secs(1)).is_err());
    }

    #[test]
    fn malformed_peer_success_receipt_is_ambiguous_without_effects() {
        let request = PeerBlindRelayRequest {
            envelope: aeronyx_core::protocol::chat::BlindRelayEnvelope {
                route_id: [1; 16],
                next_hop: [2; 32],
                ttl: 1,
                timestamp: 1,
                encrypted_blob: vec![],
                signature: [0; 64],
            },
            previous_hop_node_id: [3; 32],
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 0,
            reason: None,
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };
        assert_eq!(verify_success_response(&request, &response, [2; 32]), Err(SourceRuntimeError::Ambiguous));
    }
}
