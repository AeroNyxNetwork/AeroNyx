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
    Arc, Mutex,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::blind_vault::BlindVaultOnionPullSession;
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, SignedNodeDescriptor,
    SignedPrivateOnionRecipientAuthorizationV1,
};
use aeronyx_core::protocol::onion::OnionRoutePurpose;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionSourceEvidenceV1, ReverseOnionSourceQueryV1, SourceEvidencePartV1,
    SourceEvidenceStateV1, VerifiedSourceEvidenceChain, REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS,
    MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES,
};
use async_trait::async_trait;
use axum::body::Bytes;
use rand::RngCore;
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
        let policy = Self::validate_shape(source, relay, recipient, authorization)?;
        policy.validate_at(now)?;
        Ok(policy)
    }

    /// Constructs the same pinned authority for a durable grace/restart read.
    /// The journal has already authenticated the retained authorization at its
    /// historical envelope timestamp; this constructor only repeats signature,
    /// role, feature, endpoint, and identity checks without requiring the
    /// authorization to remain within its original wall-clock window. The
    /// caller must use this only after journal metadata recovery, which
    /// authenticates the retained authorization at its historical anchor.
    pub(crate) fn new_for_recovery(
        source: [u8; 32],
        relay: SignedNodeDescriptor,
        recipient: SignedNodeDescriptor,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
    ) -> Result<Self, SourceRuntimeError> {
        Self::validate_shape(source, relay, recipient, authorization)
    }

    fn validate_shape(
        source: [u8; 32],
        relay: SignedNodeDescriptor,
        recipient: SignedNodeDescriptor,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
    ) -> Result<Self, SourceRuntimeError> {
        if source == [0; 32]
            || relay.node_id() == [0; 32]
            || recipient.node_id() == [0; 32]
            || source == relay.node_id()
            || source == recipient.node_id()
            || relay.node_id() == recipient.node_id()
            || relay.verify_signature().is_err()
            || recipient.verify_signature().is_err()
            || !relay
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            || !relay
                .descriptor
                .capabilities
                .contains(&NodeCapability::OnionMiddle)
            || !recipient
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            || !recipient
                .descriptor
                .capabilities
                .contains(&NodeCapability::BlindVaultReplica)
            || !has_required_path_features(&relay)
            || !has_required_terminal_features(&recipient)
            || recipient.descriptor.public_endpoint.is_some()
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

    fn validate_relay_at(&self, now: u64) -> Result<(), SourceRuntimeError> {
        self.relay
            .verify_at(now)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        if !self.relay.descriptor.capabilities.contains(&NodeCapability::ChatRelay)
            || !self.relay.descriptor.capabilities.contains(&NodeCapability::OnionMiddle)
            || !has_required_path_features(&self.relay)
        {
            return Err(SourceRuntimeError::Rejected);
        }
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

    fn current_validity_bounds(&self) -> (u64, u64) {
        (
            self.relay
                .descriptor
                .issued_at
                .max(self.recipient.descriptor.issued_at),
            self.relay
                .descriptor
                .expires_at
                .min(self.recipient.descriptor.expires_at)
                .min(self.authorization.expires_at()),
        )
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

fn has_required_path_features(descriptor: &SignedNodeDescriptor) -> bool {
    let path = OnionRoutePurpose::BlindVaultPull.required_path_protocol_features();
    path.iter().all(|feature| descriptor.descriptor.advertises_protocol_feature(*feature))
}

fn has_required_terminal_features(descriptor: &SignedNodeDescriptor) -> bool {
    OnionRoutePurpose::BlindVaultPull
        .required_terminal_protocol_features()
        .iter()
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
    max_in_flight: u32,
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
        if identity.public_key_bytes() != policy.source
            || config.max_in_flight == 0
            || config.max_in_flight > MAX_SOURCE_IN_FLIGHT
            || config.timeout.is_zero()
            || config.timeout > MAX_SOURCE_TIMEOUT
        {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self {
            journal,
            identity,
            policy,
            transport,
            permits: Arc::new(Semaphore::new(config.max_in_flight)),
            max_in_flight: config.max_in_flight as u32,
            timeout: config.timeout,
            stopped: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Stops admission and waits until every accepted dispatch, including any
    /// cancellation-surviving blocking child holding a permit clone, has
    /// released its permit. Composition must call this before closing the
    /// journal or removing the authenticated route.
    pub(crate) async fn shutdown_and_drain(&self) {
        self.request_stop();
        let _drain = self
            .permits
            .clone()
            .acquire_many_owned(self.max_in_flight)
            .await;
    }

    /// Evidence-only restart/grace recovery. It never prepares a session,
    /// mutates a Prepared row, or emits another blind-relay POST. The durable
    /// row remains the authority for the historical P authorization; only the
    /// current exact relay descriptor is required before evidence queries.
    pub(crate) async fn resume(
        &self,
        route: [u8; 16],
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let permit = Arc::new(self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| SourceRuntimeError::Busy)?);
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let observed_at = observed_now(now)?;
        let metadata = self.metadata(route, observed_at, Arc::clone(&permit)).await?;
        if metadata.source() != self.identity.public_key_bytes()
            || metadata.relay() != self.policy.relay_id()
            || metadata.recipient() != self.policy.recipient_id()
            || metadata.target() != self.policy.recipient_id()
        {
            return Err(SourceRuntimeError::Unavailable);
        }
        if observed_at >= metadata.retain_until() {
            return Err(SourceRuntimeError::Expired);
        }
        match metadata.phase() {
            SourcePhase::ResultReady => self.open_result(route, observed_at, Arc::clone(&permit)).await,
            SourcePhase::Verified => self.read_verified(route, observed_at, Arc::clone(&permit)).await,
            SourcePhase::Armed | SourcePhase::DispatchAmbiguous => {
                if metadata.recipient_descriptor_commitment() != self.policy.descriptor_commitment()?
                    || metadata.relay_descriptor_commitment() != self.policy.relay_descriptor_commitment()?
                {
                    return Err(SourceRuntimeError::Unavailable);
                }
                self.policy.validate_relay_at(observed_at)?;
                self.collect_and_open(route, observed_at, Arc::clone(&permit)).await
            }
            SourcePhase::Prepared => Err(SourceRuntimeError::Ambiguous),
            SourcePhase::Rejected | SourcePhase::Opening | SourcePhase::OpenAmbiguous => {
                Err(SourceRuntimeError::Ambiguous)
            }
        }
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
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let admission = self.prepare_and_arm(
            request,
            expected,
            session,
            admitted_deadline,
            terminal_request,
            now,
            Arc::clone(&permit),
        ).await?;
        let (request, route, _deadline, url, exact_bytes) = match admission {
            SourceAdmission::Send(admission) => (
                admission.request,
                admission.route,
                admission.deadline,
                admission.url,
                admission.exact_bytes,
            ),
            SourceAdmission::Recover { route } => {
                return self.collect_and_open(route, now, Arc::clone(&permit)).await;
            }
            SourceAdmission::Complete { route } => {
                return self.read_verified(route, now, Arc::clone(&permit)).await;
            }
        };
        if exact_bytes.len() > MAX_SOURCE_DISPATCH_BYTES {
            let observed_at = observed_now(now)?;
            self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
            return Err(SourceRuntimeError::Rejected);
        }
        if self.stopped.load(Ordering::Acquire) {
            let observed_at = observed_now(now)?;
            self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
            return Err(SourceRuntimeError::Stopped);
        }
        let policy = Arc::clone(&self.policy);
        let blocking_permit = Arc::clone(&permit);
        let post_gate = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            let post_now = observed_now(0)?;
            if post_now >= _deadline {
                return Err(SourceRuntimeError::Expired);
            }
            policy.validate_at(post_now)?;
            let (policy_from, policy_until) = policy.current_validity_bounds();
            // The successful current validation also proves the private
            // authorization's issued-at bound, which is intentionally not
            // exposed as a public field. Keep that exact worker observation
            // as the rollback floor for the post-await cheap gate.
            Ok((post_now.max(policy_from), policy_until))
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)?;
        let (valid_from, valid_until) = match post_gate {
            Ok(value) => value,
            Err(error) => {
                let observed_at = observed_now(now)?;
                self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                return Err(error);
            }
        };
        let post_now = observed_now(now)?;
        if self.stopped.load(Ordering::Acquire)
            || post_now < valid_from
            || post_now >= _deadline
            || post_now >= valid_until
        {
            self.mark_ambiguous(route, post_now, Arc::clone(&permit)).await;
            return Err(if self.stopped.load(Ordering::Acquire) {
                SourceRuntimeError::Stopped
            } else {
                SourceRuntimeError::Expired
            });
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
                let observed_at = observed_now(now)?;
                self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                Err(SourceRuntimeError::Ambiguous)
            }
            SourceTransportOutcome::Response { status, body } => {
                let observed_at = observed_now(now)?;
                let request_for_verify = request;
                let relay = self.policy.relay_id();
                let blocking_permit = Arc::clone(&permit);
                let response = match tokio::task::spawn_blocking(move || {
                    let _permit = blocking_permit;
                    let response = decode_peer_response(status, &body)?;
                    verify_success_response(&request_for_verify, &response, relay, observed_at)?;
                    Ok::<_, SourceRuntimeError>(response)
                })
                .await
                .map_err(|_| SourceRuntimeError::Unavailable)?
                {
                    Ok(response) => response,
                    Err(error) => {
                        self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                        return Err(error);
                    }
                };
                let _ = response;
                self.collect_and_open(route, observed_at, Arc::clone(&permit)).await
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
            let admission_now = observed_now(now)?;
            policy.validate_at(admission_now)?;
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
                .prepare(plan, session, admission_now)
                .map_err(SourceRuntimeError::from)?;
            match phase {
                SourcePhase::Prepared => {
                    let dispatch = journal.arm(request_for_plan.envelope.route_id, admission_now)
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

    async fn mark_ambiguous(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) {
        let journal = Arc::clone(&self.journal);
        let _ = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            journal.mark_dispatch_ambiguous(route, now)
        })
        .await;
    }

    async fn collect_and_open(
        &self,
        route: [u8; 16],
        fallback_now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let now = observed_now(fallback_now)?;
        let metadata = self.metadata(route, now, Arc::clone(&permit)).await?;
        let deadline = metadata.deadline();
        if !matches!(metadata.phase(), SourcePhase::Armed | SourcePhase::DispatchAmbiguous) {
            return match metadata.phase() {
                SourcePhase::Verified => self.read_verified(route, now, Arc::clone(&permit)).await,
                SourcePhase::ResultReady => self.open_result(route, now, Arc::clone(&permit)).await,
                SourcePhase::Rejected => Err(SourceRuntimeError::Rejected),
                _ => Err(SourceRuntimeError::Ambiguous),
            };
        }
        let mut parts = Vec::with_capacity(3);
        for part in [SourceEvidencePartV1::Claim, SourceEvidencePartV1::Lease, SourceEvidencePartV1::Result] {
            if self.stopped.load(Ordering::Acquire) {
                let observed_at = observed_now(now)?;
                self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                return Err(SourceRuntimeError::Stopped);
            }
            let query = self
                .sign_query(route, metadata.request_commitment(), part, now, Arc::clone(&permit))
                .await?;
            let (query, signed_at) = query;
            let body = query.encode();
            let url = self.policy.relay_url(SOURCE_QUERY_PATH)?;
            let send_now = observed_now(now)?;
            if self.stopped.load(Ordering::Acquire) {
                self.mark_ambiguous(route, send_now, Arc::clone(&permit)).await;
                return Err(SourceRuntimeError::Stopped);
            }
            if send_now < signed_at
                || send_now
                    >= signed_at
                        .checked_add(REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS)
                        .ok_or(SourceRuntimeError::Rejected)?
            {
                self.mark_ambiguous(route, send_now, Arc::clone(&permit)).await;
                return Err(SourceRuntimeError::Expired);
            }
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
                    let observed_at = observed_now(now)?;
                    self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                    return Err(SourceRuntimeError::Ambiguous);
                }
            };
            if bytes.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                let observed_at = observed_now(now)?;
                self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                return Err(SourceRuntimeError::Ambiguous);
            }
            let blocking_permit = Arc::clone(&permit);
            let verified = tokio::task::spawn_blocking(move || {
                let _permit = blocking_permit;
                let observed_at = observed_now(0)?;
                let evidence = ReverseOnionSourceEvidenceV1::decode(&bytes)
                    .map_err(|_| SourceRuntimeError::Ambiguous)?;
                let evidence_state = evidence
                    .verify_for_query(&query, observed_at)
                    .map_err(|_| SourceRuntimeError::Ambiguous)?;
                if evidence_state.state() != SourceEvidenceStateV1::Available {
                    return Err(SourceRuntimeError::Ambiguous);
                }
                Ok::<_, SourceRuntimeError>((query, evidence, observed_at))
            })
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)?;
            let (query, evidence, observed_at) = match verified {
                Ok(pair) => pair,
                Err(_) => {
                    let observed_at = observed_now(now)?;
                    self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                    return Err(SourceRuntimeError::Ambiguous);
                }
            };
            if evidence.encode().len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await;
                return Err(SourceRuntimeError::Ambiguous);
            }
            parts.push((query, evidence));
        }
        let blocking_permit = Arc::clone(&permit);
        let chain = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            let chain_now = observed_now(0)?;
            let verified = [
                parts[0].1.verify_for_query(&parts[0].0, chain_now).map_err(|_| SourceRuntimeError::Ambiguous)?,
                parts[1].1.verify_for_query(&parts[1].0, chain_now).map_err(|_| SourceRuntimeError::Ambiguous)?,
                parts[2].1.verify_for_query(&parts[2].0, chain_now).map_err(|_| SourceRuntimeError::Ambiguous)?,
            ];
            VerifiedSourceEvidenceChain::verify([&verified[0], &verified[1], &verified[2]], deadline, chain_now)
                .map(|chain| (chain, chain_now))
                .map_err(|_| SourceRuntimeError::Ambiguous)
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)??;
        let journal = Arc::clone(&self.journal);
        let blocking_permit = Arc::clone(&permit);
        let opened = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            let (chain, _chain_now) = chain;
            let record_now = observed_now(0)?;
            journal.record_result(route, chain.claim(), chain.lease(), chain.result(), record_now)
                .map_err(SourceRuntimeError::from)?;
            let open_now = observed_now(0)?;
            journal.open_result(route, open_now).map_err(SourceRuntimeError::from)
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
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<(ReverseOnionSourceQueryV1, u64), SourceRuntimeError> {
        let identity = Arc::clone(&self.identity);
        let relay = self.policy.relay_id();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let signed_at = observed_now(now)?;
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
                signed_at,
                signed_at.checked_add(REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS)
                    .ok_or(SourceRuntimeError::Rejected)?,
            )
            .map(|query| (query, signed_at))
            .map_err(|_| SourceRuntimeError::Rejected)
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)?
    }

    async fn metadata(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<SourceRecoveryMetadata, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let page = tokio::task::spawn_blocking(move || {
            let _permit = permit;
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

    async fn read_verified(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let response = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let observed_at = observed_now(now)?;
            journal
                .read_verified(route, observed_at)
                .map_err(SourceRuntimeError::from)
        })
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)??;
        Ok(BlindVaultPullResult { response })
    }

    async fn open_result(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let journal = Arc::clone(&self.journal);
        let response = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let observed_at = observed_now(now)?;
            journal
                .open_result(route, observed_at)
                .map_err(SourceRuntimeError::from)
        })
            .await
            .map_err(|_| SourceRuntimeError::Unavailable)??;
        Ok(BlindVaultPullResult { response })
    }
}

// [REVERSE-ONION-SOURCE-LIFECYCLE 2026-10-04 by Codex] Composition injects
// one already-authenticated runtime; this owner adds only admission/ownership
// sequencing. It never creates a key, opens a journal, selects an endpoint,
// or exposes a transport route. The runtime's semaphore remains the sole
// accounting source for blocking children and in-flight network work.
pub(crate) struct ReverseOnionSourceLifecycle {
    runtime: Mutex<Option<Arc<ReverseOnionSourceRuntime>>>,
    stopped: AtomicBool,
    drain_waiter: tokio::sync::Mutex<()>,
}

impl ReverseOnionSourceLifecycle {
    pub(crate) fn new() -> Self {
        Self {
            runtime: Mutex::new(None),
            stopped: AtomicBool::new(false),
            drain_waiter: tokio::sync::Mutex::new(()),
        }
    }

    /// Installs a fully preflighted runtime. No filesystem or network work is
    /// performed here; startup owns those effects before calling this method.
    pub(crate) fn install(
        &self,
        runtime: Arc<ReverseOnionSourceRuntime>,
    ) -> Result<(), SourceRuntimeError> {
        let mut slot = self
            .runtime
            .lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?;
        if self.stopped.load(Ordering::Acquire) || slot.is_some() {
            return Err(SourceRuntimeError::Rejected);
        }
        *slot = Some(runtime);
        Ok(())
    }

    fn runtime_for_work(&self) -> Result<Arc<ReverseOnionSourceRuntime>, SourceRuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?
            .clone()
            .ok_or(SourceRuntimeError::Unavailable)?;
        // request_stop races are closed by the runtime's own admission gate;
        // this second owner check only avoids starting a new call after stop.
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        Ok(runtime)
    }

    pub(crate) fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Ok(runtime) = self.runtime.lock() {
            if let Some(runtime) = runtime.as_ref() {
                runtime.request_stop();
            }
        }
    }

    /// Internal caller path; no caller-selected endpoint or alternate
    /// transport can bypass ReverseOnionSourceRuntime's policy/journal gates.
    pub(crate) async fn dispatch(
        &self,
        request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope,
        session: BlindVaultOnionPullSession,
        admitted_deadline: u64,
        terminal_request: Vec<u8>,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let runtime = self.runtime_for_work()?;
        runtime
            .dispatch(
                request,
                expected,
                session,
                admitted_deadline,
                terminal_request,
                now,
            )
            .await
    }

    /// Internal restart path; historical journal semantics remain owned by
    /// ReverseOnionSourceRuntime::resume and are not replaced by lifecycle
    /// policy or a new live authorization check.
    pub(crate) async fn resume(
        &self,
        route: [u8; 16],
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let runtime = self.runtime_for_work()?;
        runtime.resume(route, now).await
    }

    /// Closes new work first, then retains the injected runtime/journal until
    /// its semaphore reaches zero. Only after that point is the owner slot
    /// released for shutdown teardown.
    pub(crate) async fn shutdown_and_drain(&self) -> Result<(), SourceRuntimeError> {
        let _waiter = self.drain_waiter.lock().await;
        self.request_stop();
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?
            .clone();
        if let Some(runtime) = runtime {
            runtime.shutdown_and_drain().await;
        }
        let mut slot = self
            .runtime
            .lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?;
        *slot = None;
        Ok(())
    }
}

impl Drop for ReverseOnionSourceLifecycle {
    fn drop(&mut self) {
        self.request_stop();
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

fn observed_now(_fallback: u64) -> Result<u64, SourceRuntimeError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| SourceRuntimeError::Unavailable)
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
    observed_at: u64,
) -> Result<(), SourceRuntimeError> {
    if !response.accepted
        || response.terminal
        || !response.forwarded
        || request.envelope.ttl.checked_sub(1) != Some(response.ttl_remaining)
        || response.reason.is_some()
        || response.delivery_receipt.is_some()
        || response.failure_receipt.is_some()
        || response.opaque_terminal_response_b64.is_some()
    {
        return Err(SourceRuntimeError::Ambiguous);
    }
    let receipt = response
        .success_receipt
        .as_ref()
        .ok_or(SourceRuntimeError::Ambiguous)?;
    if receipt.accepted_at > observed_at.saturating_add(30)
        || observed_at.saturating_sub(receipt.accepted_at) > 120
    {
        return Err(SourceRuntimeError::Ambiguous);
    }
    receipt
        .verify_expected(
            &request.envelope,
            false,
            true,
            response.ttl_remaining,
            response.reason.as_deref(),
            None,
            None,
            &relay,
        )
        .map_err(|_| SourceRuntimeError::Ambiguous)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    #[cfg(unix)]
    use std::future::Future;
    #[cfg(unix)]
    use std::task::{Context, Poll, Wake, Waker};

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
        assert_eq!(verify_success_response(&request, &response, [2; 32], 1_700_000_000), Err(SourceRuntimeError::Ambiguous));
    }

    #[test]
    fn forwarded_custody_receipt_is_fresh_and_terminal_payload_is_not_opened_here() {
        // [REVERSE-ONION-SOURCE-ACK-GATE 2026-10-04 by Codex] The source
        // accepts only the immediate relay's forwarded custody ACK; AXRE
        // evidence remains the sole terminal-response opening authority.
        let relay = IdentityKeyPair::generate();
        let envelope = aeronyx_core::protocol::chat::BlindRelayEnvelope {
            route_id: [1; 16],
            next_hop: relay.public_key_bytes(),
            ttl: 2,
            timestamp: 1_700_000_000,
            encrypted_blob: vec![7; 4],
            signature: [0; 64],
        };
        let request = PeerBlindRelayRequest {
            envelope: envelope.clone(),
            previous_hop_node_id: [3; 32],
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let receipt = aeronyx_core::protocol::chat::BlindRelaySuccessReceipt::forwarded(
            &envelope,
            1,
            None,
            None,
            None,
            1_700_000_001,
            &relay,
        );
        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
            success_receipt: Some(receipt),
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };
        assert!(verify_success_response(
            &request,
            &response,
            relay.public_key_bytes(),
            1_700_000_002,
        )
        .is_ok());
        let mut wrong_ttl = response.clone();
        wrong_ttl.ttl_remaining = 0;
        wrong_ttl.success_receipt = Some(
            aeronyx_core::protocol::chat::BlindRelaySuccessReceipt::forwarded(
                &envelope,
                0,
                None,
                None,
                None,
                1_700_000_001,
                &relay,
            ),
        );
        assert_eq!(
            verify_success_response(&request, &wrong_ttl, relay.public_key_bytes(), 1_700_000_002),
            Err(SourceRuntimeError::Ambiguous)
        );
        let mut stale = response;
        stale.success_receipt = Some(
            aeronyx_core::protocol::chat::BlindRelaySuccessReceipt::forwarded(
                &envelope,
                1,
                None,
                None,
                None,
                1_700_000_002 - 121,
                &relay,
            ),
        );
        assert_eq!(
            verify_success_response(&request, &stale, relay.public_key_bytes(), 1_700_000_002),
            Err(SourceRuntimeError::Ambiguous)
        );
    }

    // [REVERSE-ONION-SOURCE-RUNTIME-TESTS 2026-10-04 by Codex] These tests
    // exercise only the existing private constructor and source journal
    // fixture. No production route, endpoint, or transport is registered.
    #[cfg(unix)]
    struct CountingTransport {
        posts: Arc<AtomicUsize>,
        queries: Arc<AtomicUsize>,
    }

    #[cfg(unix)]
    #[async_trait::async_trait]
    impl SourceTransport for CountingTransport {
        async fn post(&self, _url: reqwest::Url, _body: Bytes) -> SourceTransportOutcome {
            self.posts.fetch_add(1, AtomicOrdering::SeqCst);
            SourceTransportOutcome::Ambiguous
        }

        async fn query(&self, _url: reqwest::Url, _body: Bytes) -> SourceTransportOutcome {
            self.queries.fetch_add(1, AtomicOrdering::SeqCst);
            SourceTransportOutcome::Ambiguous
        }
    }

    #[cfg(unix)]
    fn fixture_runtime(
        fixture: &crate::services::reverse_onion_source::tests::Fixture,
        transport: Arc<dyn SourceTransport>,
        journal: Arc<ReverseOnionSourceJournal>,
    ) -> Arc<ReverseOnionSourceRuntime> {
        let (relay, recipient, authorization) = fixture.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy {
            source: fixture.source_identity().public_key_bytes(),
            relay,
            recipient,
            authorization,
        });
        Arc::new(
            ReverseOnionSourceRuntime::new(
                journal,
                fixture.source_identity(),
                policy,
                transport,
                SourceRuntimeConfig::new(1, Duration::from_secs(1)).unwrap(),
            )
            .unwrap(),
        )
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_sets_stop_and_waits_for_all_accepted_permits() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let journal = Arc::new(fixture.open(1_800_000_000));
        let transport = Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)),
            queries: Arc::new(AtomicUsize::new(0)),
        });
        let runtime = fixture_runtime(&fixture, transport, journal);
        let held = runtime.permits.clone().acquire_owned().await.unwrap();
        let mut draining = Box::pin(runtime.shutdown_and_drain());
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(draining.as_mut().poll(&mut context), Poll::Pending));
        assert!(runtime.stopped.load(Ordering::Acquire));
        drop(held);
        tokio::time::timeout(Duration::from_secs(1), draining)
            .await
            .unwrap();
        assert!(runtime.stopped.load(Ordering::Acquire));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reopened_journal_helper_opens_result_without_post_or_query() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.ready_with_maximum(&journal);
        let expected_body = fixture.outbound_bytes();
        assert!(!expected_body.is_empty());
        let before = journal
            .recover_metadata(None, 1, fixture.now() + 3)
            .unwrap()
            .items
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(before.phase(), SourcePhase::ResultReady);
        assert_eq!(before.body_commitment(), fixture.outbound_body_commitment());
        drop(journal);
        let reopened = Arc::new(fixture.open(fixture.now() + 3));
        let reopened_item = reopened
            .recover_metadata(None, 1, fixture.now() + 3)
            .unwrap()
            .items
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(reopened_item.phase(), SourcePhase::ResultReady);
        assert_eq!(reopened_item.body_commitment(), before.body_commitment());
        let posts = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(CountingTransport {
            posts: Arc::clone(&posts),
            queries: Arc::clone(&queries),
        });
        let runtime = fixture_runtime(&fixture, transport, reopened);
        let permit = Arc::new(runtime.permits.clone().try_acquire_owned().unwrap());
        let result = runtime
            .open_result(fixture.route(), fixture.now(), permit)
            .await
            .unwrap();
        assert_eq!(result.response().lease_id, [7; 32]);
        assert_eq!(result.response().objects.len(), 1);
        assert_eq!(result.response().objects[0].object_id, [6; 32]);
        assert_eq!(
            result.response().objects[0].ciphertext,
            fixture.result_object_bytes()
        );
        assert_eq!(
            result.response().objects[0].ciphertext_commitment,
            fixture.result_object_commitment()
        );
        let observed = observed_now(0).unwrap();
        let verified = runtime
            .journal
            .recover_metadata(None, 1, observed)
            .unwrap()
            .items
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(verified.phase(), SourcePhase::Verified);
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(queries.load(AtomicOrdering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn lifecycle_stop_gate_rejects_new_resume_and_drains_injected_runtime() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let journal = Arc::new(fixture.open(1_800_000_000));
        let transport = Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)),
            queries: Arc::new(AtomicUsize::new(0)),
        });
        let runtime = fixture_runtime(&fixture, transport, journal);
        let lifecycle = ReverseOnionSourceLifecycle::new();
        lifecycle.install(Arc::clone(&runtime)).unwrap();
        lifecycle.request_stop();
        assert!(matches!(
            lifecycle.resume(fixture.route(), 1_800_000_000).await,
            Err(SourceRuntimeError::Stopped)
        ));
        lifecycle.shutdown_and_drain().await.unwrap();
        assert!(matches!(
            lifecycle.resume(fixture.route(), 1_800_000_000).await,
            Err(SourceRuntimeError::Stopped)
        ));
        assert_eq!(
            lifecycle.install(runtime).err(),
            Some(SourceRuntimeError::Rejected)
        );
    }

}
