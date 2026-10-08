// ============================================
// File: crates/aeronyx-server/src/server/reverse_onion_source_runtime.rs
// ============================================
//! Source-side carrier for the fixed-class private Blind Vault Pull.
//!
//! [REVERSE-ONION-SOURCE-RUNTIME 2026-10-04 by Codex] This module is an
//! explicitly constructed capability. Startup now composes this default-off
//! internal source with a durable journal and current PeerStore pins, without
//! exposing a source-control HTTP endpoint. The carrier keeps the journal as the only
//! effect authority, never accepts a caller-selected endpoint, and never
//! retries a POST after its Prepared->Armed CAS.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::task::Poll;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::blind_vault::{
    encode_blind_vault_frame, BlindVaultFrame, BlindVaultOnionPullSession,
    BlindVaultPullRequest,
};
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, SignedNodeDescriptor,
    SignedPrivateOnionRecipientAuthorizationV1,
};
use aeronyx_core::protocol::onion::{OnionRoutePurpose, VerifiedOnionRoute};
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionSourceEvidenceV1, ReverseOnionSourceQueryV1, SourceEvidencePartV1,
    SourceEvidenceStateV1, VerifiedSourceEvidenceChain, REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS,
    MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES,
};
use async_trait::async_trait;
use axum::body::{Body, Bytes};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::FutureExt;
#[cfg(test)]
use futures::Stream;
use rand::RngCore;
use tokio::sync::{watch, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use crate::api::{
    canonical_peer_http_url, resolve_pinned_peer_http_target, reverse_onion_endpoint_supported,
    PinnedPeerHttpTarget,
    read_bounded_http_response, BoundedHttpResponseError, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES,
};
use crate::api::chat_peer::{
    blind_relay_authenticated_request_commitment, PeerBlindRelayRequest, PeerBlindRelayResponse,
};
use crate::services::reverse_onion_source::{
    ExpectedRetainedEnvelope, ReverseOnionSourceJournal, SourceJournalError, SourcePhase,
    SourcePreparedPull, SourceRecoveryDispatch, SourceRecoveryMetadata,
};
use crate::services::peer_store::PeerStore;
use super::reverse_onion_runtime::ReverseOnionLocalObservation;

const BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";
const SOURCE_QUERY_PATH: &str = "/api/chat/peer/reverse-onion/source-query";
const MAX_SOURCE_DISPATCH_BYTES: usize = 64 * 1024;
const MAX_SOURCE_TERMINAL_BYTES: usize = 512;
const MAX_SOURCE_IN_FLIGHT: usize = 64;
const DEFAULT_SOURCE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_SOURCE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_SOURCE_RESULT_WAIT: Duration = Duration::from_secs(30);
use aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_ENVELOPE_LIFETIME_SECS;

// [SOURCE-STARTUP-OWNER 2026-10-05 by Codex] Opening the journal audits and
// recovers durable state without network effects. A later authenticated source
// retry may replay exact armed bytes to the same relay under live authority.
// Expired P authority still permits evidence/local recovery, never POST replay.
pub(super) async fn open_configured_source(
    config: &crate::config_reverse_onion::ReverseOnionSourceConfig,
    identity: Arc<IdentityKeyPair>,
    peers: Arc<PeerStore>,
) -> Result<Arc<ReverseOnionSourceRuntime>, SourceRuntimeError> {
    let (relay_id, recipient_id, relay_endpoint) = config.identity_pins()
        .map_err(|_| SourceRuntimeError::Rejected)?;
    peers.pin_private_onion_route_identities(relay_id, recipient_id)
        .map_err(|_| SourceRuntimeError::Rejected)?;
    if let Some((relay, recipient, authorization)) = config.optional_authority_material()
        .map_err(|_| SourceRuntimeError::Rejected)?
    {
        if relay.node_id() != relay_id || recipient.node_id() != recipient_id
            || authorization.relay_node_id() != relay_id
            || authorization.recipient_node_id() != recipient_id
            || !relay.descriptor.public_endpoint.as_deref().is_some_and(|endpoint|
                SourcePinnedRelayPolicy::same_reverse_onion_endpoint(endpoint, &relay_endpoint))
            || relay.verify_signature().is_err() || recipient.verify_signature().is_err()
            || authorization.verify_signature().is_err()
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let pin_now = observed_now(0)?;
        if relay.verify_at(pin_now).is_ok() && recipient.verify_at(pin_now).is_ok() {
            // [REVERSE-ONION-STALE-SEED 2026-10-05 by Codex] Optional seed
            // data only warms the authenticated PeerStore; it is not required
            // to start the authority-inert source runtime.
            peers.seed_private_onion_route_descriptor(
                &relay_id, &recipient_id, relay.clone(), pin_now, "reverse_onion_source_seed",
            ).map_err(|_| SourceRuntimeError::Rejected)?;
            peers.seed_private_onion_route_descriptor(
                &relay_id, &recipient_id, recipient.clone(), pin_now, "reverse_onion_source_seed",
            ).map_err(|_| SourceRuntimeError::Rejected)?;
        let current_seed_pair = peers.get_valid(&relay_id, pin_now)
                .zip(peers.get_valid(&recipient_id, pin_now))
                .is_some_and(|(current_relay, current_recipient)| {
                    matches!((current_relay.encode_canonical(), relay.encode_canonical()),
                        (Ok(current), Ok(seed)) if current == seed)
                        && matches!((current_recipient.encode_canonical(), recipient.encode_canonical()),
                            (Ok(current), Ok(seed)) if current == seed)
                });
            if current_seed_pair && authorization.verify_at(
                &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), pin_now,
            ).is_ok() {
                peers.import_private_onion_authorization_bundle(
                    authorization, relay.clone(), recipient.clone(), identity.public_key_bytes(), pin_now,
                ).map_err(|_| SourceRuntimeError::Rejected)?;
            }
        }
    }
    let runtime_config = SourceRuntimeConfig::new(
        config.max_in_flight, Duration::from_secs(config.request_timeout_secs),
        config.lease_max_secs,
        Duration::from_secs(config.result_wait_secs),
        Duration::from_millis(config.result_poll_interval_ms),
    )?;
    let transport = Arc::new(ReqwestSourceTransport::new(runtime_config.timeout)?);
    let path = std::path::PathBuf::from(&config.state_db_path);
    let limits = crate::services::reverse_onion_source::SourceJournalLimits {
        max_entries: config.max_pending_items,
        max_bytes: config.max_bytes,
    };
    let journal_identity = Arc::clone(&identity);
    let recovery_only = config.recovery_only;
    let journal = tokio::task::spawn_blocking(move || {
        let open = if recovery_only { ReverseOnionSourceJournal::open_existing }
            else { ReverseOnionSourceJournal::open };
        open(&path, journal_identity, limits, observed_now(0)?)
            .map_err(SourceRuntimeError::from)
    }).await.map_err(|_| SourceRuntimeError::Unavailable)??;
    let mut runtime = ReverseOnionSourceRuntime::new_identity_pinned(
        Arc::new(journal), identity, relay_id, recipient_id, relay_endpoint,
        peers, transport, runtime_config,
    )?;
    runtime.recovery_only = recovery_only;
    Ok(Arc::new(runtime))
}

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
    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Signed Pending
    // evidence is retryable and must not be conflated with uncertain POST acceptance.
    #[error("source result is still pending")]
    Pending,
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
            // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] Quota
            // exhaustion is retryable backpressure, not a failed local owner.
            SourceJournalError::Busy | SourceJournalError::Capacity => Self::Busy,
            SourceJournalError::Ambiguous => Self::Ambiguous,
            SourceJournalError::Conflict => Self::Conflict,
            SourceJournalError::Expired => Self::Expired,
            SourceJournalError::Unavailable | SourceJournalError::Corrupt
            | SourceJournalError::ClockRollback
            | SourceJournalError::MigrationRequired => Self::Unavailable,
            SourceJournalError::Rejected | SourceJournalError::ReplyRejected => Self::Rejected,
        }
    }
}

/// Fixed source-side timeout and in-flight policy. Zero or oversized values
/// are rejected before a client, task, or semaphore is constructed.
// [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex]
#[derive(Clone, Copy)]
pub(crate) struct SourceRuntimeConfig {
    pub(crate) max_in_flight: usize,
    pub(crate) timeout: Duration,
    pub(crate) lease_max_secs: u64,
    pub(crate) result_wait: Duration,
    pub(crate) result_poll_interval: Duration,
}

impl SourceRuntimeConfig {
    pub(crate) fn new(
        max_in_flight: usize,
        timeout: Duration,
        lease_max_secs: u64,
        result_wait: Duration,
        result_poll_interval: Duration,
    ) -> Result<Self, SourceRuntimeError> {
        if max_in_flight == 0
            || max_in_flight > MAX_SOURCE_IN_FLIGHT
            || timeout.is_zero()
            || timeout > MAX_SOURCE_TIMEOUT
            || !(1..=600).contains(&lease_max_secs)
            || result_wait > MAX_SOURCE_RESULT_WAIT
            || result_poll_interval < Duration::from_millis(100)
            || result_poll_interval > Duration::from_secs(5)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self { max_in_flight, timeout, lease_max_secs, result_wait, result_poll_interval })
    }
}

/// Signed, pinned R/P authority and its exact source identity. No Debug is
/// implemented: descriptors contain public endpoints and must not be logged.
pub(crate) struct SourcePinnedRelayPolicy {
    source: [u8; 32],
    relay: SignedNodeDescriptor,
    recipient: SignedNodeDescriptor,
    authorization: SignedPrivateOnionRecipientAuthorizationV1,
    current_peers: Option<Arc<PeerStore>>,
    relay_endpoint_pin: Option<String>,
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
            // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] P is private and
            // operation admission is proven by its signed terminal feature;
            // it need not expose a public ChatRelay or replica API capability.
            || !has_required_path_features(&relay)
            || !has_required_terminal_features(&recipient)
            || recipient.descriptor.public_endpoint.is_some()
            // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] The
            // signed terminal policy must affirm private discovery too.
            || recipient.descriptor.policy.public_discovery
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let endpoint = relay
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        if !reverse_onion_endpoint_supported(endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self {
            source,
            relay,
            recipient,
            authorization,
            current_peers: None,
            relay_endpoint_pin: None,
        })
    }

    // [SOURCE-CURRENT-PINS 2026-10-05 by Codex] New effects require both
    // current pins; historical evidence recovery requires only its relay.
    fn with_current_peers(mut self, peers: Arc<PeerStore>) -> Self {
        self.current_peers = Some(peers);
        self
    }

    fn with_relay_endpoint_pin(mut self, endpoint: String) -> Self {
        self.relay_endpoint_pin = Some(endpoint);
        self
    }

    // [PHALA-SOURCE-AUTHORITY-SNAPSHOT 2026-10-06 by Codex] Reject a grant
    // unless it is exactly the current authenticated snapshot and fixed origin.
    fn from_current_snapshot(
        source: [u8; 32], relay_id: [u8; 32], recipient_id: [u8; 32],
        endpoint_pin: &str, peers: Arc<PeerStore>,
        authorization: SignedPrivateOnionRecipientAuthorizationV1, now: u64,
    ) -> Result<Self, SourceRuntimeError> {
        if authorization.relay_node_id() != relay_id
            || authorization.recipient_node_id() != recipient_id
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let (relay, recipient, current_authorization) = peers
            .current_private_onion_authority_snapshot(&relay_id, &recipient_id, now)
            .ok_or(SourceRuntimeError::Rejected)?;
        if current_authorization != authorization
            || !relay.descriptor.public_endpoint.as_deref().is_some_and(|endpoint|
                Self::same_reverse_onion_endpoint(endpoint_pin, endpoint))
        {
            return Err(SourceRuntimeError::Rejected);
        }
        Self::validate_shape(source, relay, recipient, authorization)
            .map(|policy| policy.with_current_peers(peers)
                .with_relay_endpoint_pin(endpoint_pin.to_owned()))
    }

    // [REVERSE-SOURCE-HISTORIC-AUTH 2026-10-05 by Codex] The journal supplies
    // the route's historical public authority; reuse the same live pin source.
    fn current_peers(&self) -> Option<Arc<PeerStore>> {
        self.current_peers.as_ref().map(Arc::clone)
    }

    fn check_current_pin(&self, expected: &SignedNodeDescriptor, now: u64)
        -> Result<(), SourceRuntimeError>
    {
        if let Some(peers) = &self.current_peers {
            let current = peers.get_valid(&expected.node_id(), now)
                .ok_or(SourceRuntimeError::Rejected)?;
            let current = current.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?;
            let expected = expected.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?;
            if current != expected { return Err(SourceRuntimeError::Rejected); }
        }
        Ok(())
    }

    fn validate_at(&self, now: u64) -> Result<(), SourceRuntimeError> {
        if let Some(peers) = &self.current_peers {
            let _authority_epoch = peers.private_onion_authority_read_guard();
            return self.validate_at_under_authority_guard(now);
        }
        self.validate_pinned_at(now)
    }

    // [REVERSE-ONION-SOURCE-AUTHORITY-FENCE 2026-10-05 by Codex] The caller
    // holds PeerStore's read epoch through source journal prepare+arm, making
    // the exact authority check and durable one-shot dispatch boundary atomic
    // with respect to descriptor/grant rotation.
    fn validate_at_under_authority_guard(&self, now: u64) -> Result<(), SourceRuntimeError> {
        if let Some(peers) = &self.current_peers {
            let (relay, recipient, current_authorization) = peers
                .current_private_onion_authority_snapshot_under_guard(
                    &self.relay_id(),
                    &self.recipient_id(),
                    now,
                )
                .ok_or(SourceRuntimeError::Rejected)?;
            if relay.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?
                    != self.relay.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?
                || recipient.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?
                    != self.recipient.encode_canonical().map_err(|_| SourceRuntimeError::Rejected)?
                || current_authorization != self.authorization
            {
                return Err(SourceRuntimeError::Rejected);
            }
        }
        self.validate_pinned_at(now)
    }

    fn validate_pinned_at(&self, now: u64) -> Result<(), SourceRuntimeError> {
        self.relay
            .verify_at(now)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        self.require_phala_route_appraisal(&self.relay, now)?;
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
        if !reverse_onion_endpoint_supported(endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(())
    }

    // [PHALA-PRIVATE-SOURCE-ROUTE-GATE 2026-10-06 by Codex] The private
    // source-pull path uses an operator-pinned relay instead of route scoring,
    // so apply the same opt-in appraisal policy at its effect boundary.
    fn require_phala_route_appraisal(
        &self,
        relay: &SignedNodeDescriptor,
        now: u64,
    ) -> Result<(), SourceRuntimeError> {
        if self
            .current_peers
            .as_ref()
            .is_some_and(|peers| !peers.phala_peer_route_is_eligible(relay, now))
        {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(())
    }

    // [REVERSE-ONION-AUTHORITY-RENEWAL 2026-10-05 by Codex] New effects require
    // the exact current signed R/P pair and latest cached P authorization.
    // Recheck together before preparing durable work and immediately pre-POST.
    fn with_current_authorization(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<Self, SourceRuntimeError> {
        let peers = self.current_peers.as_ref().ok_or(SourceRuntimeError::Rejected)?;
        if authorization.relay_node_id() != self.relay_id()
            || authorization.recipient_node_id() != self.recipient_id()
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let (relay, recipient, current_authorization) = peers
            .current_private_onion_authority_snapshot(
                &self.relay_id(),
                &self.recipient_id(),
                now,
            )
            .ok_or(SourceRuntimeError::Rejected)?;
        let updated = Self::validate_shape(self.source, relay, recipient, authorization)?
            .with_current_peers(Arc::clone(peers));
        if updated.authorization != current_authorization {
            return Err(SourceRuntimeError::Rejected);
        }
        updated.validate_at(now)?;
        Ok(updated)
    }

    fn validate_relay_at(&self, now: u64) -> Result<(), SourceRuntimeError> {
        self.recovery_relay_descriptor_at(now).map(|_| ())
    }

    // [REVERSE-SOURCE-RECOVERY-SNAPSHOT 2026-10-07 by Codex] Addressing
    // and appraisal must consume the same signed descriptor. A second lookup
    // after validation could otherwise select an unvalidated rotated origin.
    fn recovery_relay_descriptor_at(
        &self,
        now: u64,
    ) -> Result<SignedNodeDescriptor, SourceRuntimeError> {
        // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Descriptor
        // and appraisal lookup use the same epoch, including evidence recovery.
        let _authority_epoch = self.current_peers.as_ref()
            .map(|peers| peers.private_onion_authority_read_guard());
        self.recovery_relay_descriptor_at_under_authority_guard(now)
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Caller already
    // owns the epoch; reacquiring a read lock could deadlock behind a writer.
    fn recovery_relay_descriptor_at_under_authority_guard(
        &self, now: u64,
    ) -> Result<SignedNodeDescriptor, SourceRuntimeError> {
        // [REVERSE-RECOVERY-ROUTE-ROTATION 2026-10-05 by Codex] Evidence
        // recovery authenticates the stable relay identity and original
        // journal commitments. A newer descriptor may renew metadata, but its
        // canonical transport origin must remain the one authorized by the
        // original route. New dispatch still needs the exact current R/P pair
        // and grant.
        let relay = match &self.current_peers {
            Some(peers) => peers.get_valid(&self.relay_id(), now)
                .ok_or(SourceRuntimeError::Rejected)?,
            None => self.relay.clone(),
        };
        if relay.node_id() != self.relay_id() { return Err(SourceRuntimeError::Rejected); }
        let historical_endpoint = self.relay.descriptor.public_endpoint.as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        let current_endpoint = relay.descriptor.public_endpoint.as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        if self.relay_endpoint_pin.as_deref().is_some_and(|pinned|
            !Self::same_reverse_onion_endpoint(pinned, current_endpoint))
        {
            return Err(SourceRuntimeError::Rejected);
        }
        if !Self::same_reverse_onion_endpoint(historical_endpoint, current_endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        relay
            .verify_at(now)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        self.require_phala_route_appraisal(&relay, now)?;
        if !relay.descriptor.capabilities.contains(&NodeCapability::ChatRelay)
            || !relay.descriptor.capabilities.contains(&NodeCapability::OnionMiddle)
            || !has_required_path_features(&relay)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let endpoint = relay
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        if !reverse_onion_endpoint_supported(endpoint) {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(relay)
    }

    // [REVERSE-RECOVERY-ROUTE-ROTATION 2026-10-05 by Codex] The historical
    // authority remains attached to the journal; only transport addressing
    // follows the currently verified descriptor for the same relay identity.
    fn recovery_relay_snapshot(
        &self,
        path: &str,
        now: u64,
    ) -> Result<(reqwest::Url, [u8; 32]), SourceRuntimeError> {
        let relay = self.recovery_relay_descriptor_at(now)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&relay)
            .map_err(|_| SourceRuntimeError::Rejected)?.hash();
        let endpoint = relay.descriptor.public_endpoint.as_deref()
            .ok_or(SourceRuntimeError::Rejected)?;
        let url = canonical_peer_http_url(endpoint, path)
            .map_err(|_| SourceRuntimeError::Rejected)?;
        Ok((url, commitment))
    }

    // [REVERSE-SOURCE-RECOVERY-SNAPSHOT 2026-10-07 by Codex] DNS can await
    // across descriptor or appraisal rotation. Revalidate and compare the
    // exact descriptor used to resolve the target before disclosing a query.
    fn validate_recovery_relay_snapshot(
        &self,
        expected_commitment: [u8; 32],
        now: u64,
    ) -> Result<(), SourceRuntimeError> {
        // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex]
        let _authority_epoch = self.current_peers.as_ref()
            .map(|peers| peers.private_onion_authority_read_guard());
        self.validate_recovery_relay_snapshot_under_authority_guard(expected_commitment, now)
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex]
    fn validate_recovery_relay_snapshot_under_authority_guard(
        &self, expected_commitment: [u8; 32], now: u64,
    ) -> Result<(), SourceRuntimeError> {
        let relay = self.recovery_relay_descriptor_at_under_authority_guard(now)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&relay)
            .map_err(|_| SourceRuntimeError::Rejected)?.hash();
        if commitment != expected_commitment {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(())
    }

    // [REVERSE-RECOVERY-ENDPOINT-PIN 2026-10-06 by Codex] Recovery may follow
    // a signed renewal only at the previously authorized HTTPS origin.
    // Canonical parsing normalizes host casing and default ports while
    // preventing a rotated descriptor from redirecting historical evidence.
    fn same_reverse_onion_endpoint(left: &str, right: &str) -> bool {
        // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Reuse the
        // recipient/discovery gate, including rejection of non-public origins.
        crate::api::reverse_onion_same_origin(left, right)
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
    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Only a local gate or
    // request-build failure before HTTP entry may produce this proof.
    NotSent(SourceNoSend),
}

// [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Carry the exact source
// owner stop flag and immutable time bounds into the transport future. Its
// first poll and request construction may occur after the caller's last check.
pub(crate) struct SourceSendAdmission {
    stopped: Arc<AtomicBool>,
    // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Shared only
    // with this attempt's owner so HTTP cancellation cannot lose its sample.
    floor: Arc<ReverseOnionLocalObservation>,
    deadline: u64,
    poll_deadline: Option<std::time::Instant>,
    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Retain the
    // selected policy, not permission to independently select another route.
    authority: SourceSendAuthority,
}

// [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Fresh dispatch needs
// the exact R/P/grant. Historical evidence needs only its selected same-origin
// relay snapshot, never an unrelated current grant or replacement endpoint.
enum SourceSendAuthority {
    Dispatch(Arc<SourcePinnedRelayPolicy>),
    Evidence { policy: Arc<SourcePinnedRelayPolicy>, relay_snapshot: [u8; 32] },
    // Clock-only fixtures cannot create an unguarded production admission.
    #[cfg(test)]
    TimeOnly,
}

impl SourceSendAuthority {
    fn current_peers(&self) -> Option<&PeerStore> {
        match self {
            Self::Dispatch(policy) | Self::Evidence { policy, .. } => policy.current_peers.as_deref(),
            #[cfg(test)]
            Self::TimeOnly => None,
        }
    }

    fn validate_under_authority_guard(&self, now: u64) -> Result<(), SourceRuntimeError> {
        match self {
            Self::Dispatch(policy) => policy.validate_at_under_authority_guard(now),
            Self::Evidence { policy, relay_snapshot } => {
                policy.validate_recovery_relay_snapshot_under_authority_guard(*relay_snapshot, now)
            }
            #[cfg(test)]
            Self::TimeOnly => Ok(()),
        }
    }
}

// [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Private fields keep this
// local admission evidence distinct from an arbitrary HTTP response/status.
// It proves only the current attempt did not enter HTTP, never prior attempts.
pub(crate) struct SourceNoSend {
    error: SourceRuntimeError,
}

impl SourceSendAdmission {
    pub(crate) fn check(&self) -> Result<(), SourceNoSend> {
        self.check_with_clock(|| observed_now(0))
    }

    #[cfg(test)]
    pub(crate) fn check_at(&self, now: Result<u64, SourceRuntimeError>) -> Result<(), SourceNoSend> {
        self.check_with_clock(|| now)
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Never block the
    // HTTP executor behind a writer. Sample time only after admission to the
    // epoch; no read guard or successful check escapes to a different attempt.
    fn check_with_clock(&self, clock: impl FnOnce() -> Result<u64, SourceRuntimeError>)
        -> Result<(), SourceNoSend>
    {
        let _authority_epoch = match self.authority.current_peers() {
            Some(peers) => Some(peers.try_private_onion_authority_read_guard()
                .ok_or(SourceNoSend { error: SourceRuntimeError::Busy })?),
            None => None,
        };
        self.check_under_authority_guard(clock())
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Both the actual
    // transport and injected-clock test path validate under the held epoch.
    fn check_under_authority_guard(&self, now: Result<u64, SourceRuntimeError>) -> Result<(), SourceNoSend> {
        let now = self.floor.observe(now.map_err(|_| ()))
            .map_err(|_| SourceNoSend { error: SourceRuntimeError::Unavailable })?;
        let error = if self.stopped.load(Ordering::Acquire) {
            Some(SourceRuntimeError::Stopped)
        } else if now >= self.deadline {
            Some(SourceRuntimeError::Expired)
        } else if self.poll_deadline.is_some_and(|until| std::time::Instant::now() >= until) {
            Some(SourceRuntimeError::Pending)
        } else {
            None
        };
        if let Some(error) = error { return Err(SourceNoSend { error }); }
        self.authority.validate_under_authority_guard(now).map_err(|error| SourceNoSend { error })
    }
}

// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] Source transport accepts
// only pre-resolved targets so the durable arm cannot be followed by a fresh
// unvalidated hostname lookup.
#[async_trait]
pub(crate) trait SourceTransport: Send + Sync {
    async fn post(
        &self,
        target: PinnedPeerHttpTarget,
        body: Bytes,
        authorization: Bytes,
        admission: SourceSendAdmission,
    ) -> SourceTransportOutcome;
    async fn query(&self, target: PinnedPeerHttpTarget, body: Bytes,
        admission: SourceSendAdmission) -> SourceTransportOutcome;
}

/// Production transport. It receives a DNS-pinned route client and canonical
/// URL; hostname TLS verification remains enabled and no proxy is inherited.
pub(crate) struct ReqwestSourceTransport {
    timeout: Duration,
}

impl ReqwestSourceTransport {
    pub(crate) fn new(timeout: Duration) -> Result<Self, SourceRuntimeError> {
        if timeout.is_zero() || timeout > MAX_SOURCE_TIMEOUT {
            return Err(SourceRuntimeError::Rejected);
        }
        Ok(Self { timeout })
    }

    async fn send(
        &self,
        target: PinnedPeerHttpTarget,
        body: Bytes,
        authorization: Option<Bytes>,
        limit: usize,
        content_type: &'static str,
        admission: SourceSendAdmission,
    ) -> SourceTransportOutcome {
        let mut request = target.client.post(target.url)
            .header(reqwest::header::CONTENT_TYPE, content_type);
        if let Some(authorization) = authorization {
            request = request.header(
                "x-aeronyx-private-recipient-authorization",
                STANDARD.encode(authorization),
            );
        }
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Build all headers
        // and bytes first. No await separates the final gate from HTTP entry;
        // once execute starts, every timeout/error remains ambiguous.
        let request = match request.body(body).build() {
            Ok(request) => request,
            Err(_) => return SourceTransportOutcome::NotSent(SourceNoSend { error: SourceRuntimeError::Rejected }),
        };
        if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
        let response = match target.client.execute(request).await {
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
    async fn post(&self, target: PinnedPeerHttpTarget, body: Bytes, authorization: Bytes,
        admission: SourceSendAdmission) -> SourceTransportOutcome {
        match tokio::time::timeout(self.timeout, self.send(
            target, body, Some(authorization), BLIND_RELAY_ACK_RESPONSE_MAX_BYTES, "application/json", admission,
        )).await {
            Ok(outcome) => outcome,
            Err(_) => SourceTransportOutcome::Ambiguous,
        }
    }

    async fn query(&self, target: PinnedPeerHttpTarget, body: Bytes,
        admission: SourceSendAdmission) -> SourceTransportOutcome {
        match tokio::time::timeout(self.timeout, self.send(
            target,
            body,
            None,
            MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES,
            "application/octet-stream",
            admission,
        )).await {
            Ok(outcome) => outcome,
            Err(_) => SourceTransportOutcome::Ambiguous,
        }
    }
}

/// Source carrier with one bounded in-flight class and an explicit stop bit.
/// The journal is opened and owned by composition; this object never opens a
/// second database or creates an async task that outlives its caller.
pub(crate) struct ReverseOnionSourceRuntime {
    // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Immutable per-owner mode.
    recovery_only: bool,
    journal: Arc<ReverseOnionSourceJournal>,
    identity: Arc<IdentityKeyPair>,
    policy: Option<Arc<SourcePinnedRelayPolicy>>,
    authority_pins: Option<SourceAuthorityPins>,
    transport: Arc<dyn SourceTransport>,
    permits: Arc<Semaphore>,
    max_in_flight: u32,
    timeout: Duration,
    stopped: Arc<AtomicBool>,
    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] A sticky,
    // source-blind failure signal survives notification before subscription.
    // Graceful stop only closes intake and never publishes this fault.
    failure: watch::Sender<bool>,
    // [REVERSE-SOURCE-ROUTE-ADMISSION 2026-10-05 by Codex] Prevent concurrent
    // identical client retries from preparing two randomized onion requests.
    active_routes: Arc<Mutex<HashSet<[u8; 16]>>>,
    lease_max_secs: u64,
    result_wait: Duration,
    result_poll_interval: Duration,
}

// [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Stable operator
// topology pins are separate from replaceable signed authority snapshots.
struct SourceAuthorityPins {
    relay_id: [u8; 32],
    recipient_id: [u8; 32],
    relay_endpoint: String,
    peers: Arc<PeerStore>,
}

// [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Keep the source permit
// while waiting, then own journal + lane + source permit inside blocking work.
// Async cancellation after spawn cannot release any of them before the DB
// action/fence ends. All production callers already hold bounded admission;
// no HTTP/DNS await or recursively acquired lane is allowed in the action.
pub(super) async fn source_db<T: Send + 'static>(
    journal: &Arc<ReverseOnionSourceJournal>,
    permit: Arc<OwnedSemaphorePermit>,
    now: u64,
    action: impl FnOnce(&ReverseOnionSourceJournal, u64) -> Result<T, SourceRuntimeError> + Send + 'static,
) -> Result<T, SourceRuntimeError> {
    let lane = journal.blocking_operation_lane().acquire_owned().await
        .map_err(|_| SourceRuntimeError::Unavailable)?;
    let journal = Arc::clone(journal);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _lane = lane;
        action(&journal, observed_now(now)?)
    }).await.map_err(|_| SourceRuntimeError::Unavailable)?
}

// [REVERSE-SOURCE-API-ADMISSION 2026-10-05 by Codex] The VPN route's body
// parser and the source runtime share one lifecycle-owned admission boundary.
// It rejects new bodies after stop and drains requests that crossed admission.
pub(crate) struct ReverseOnionSourceRequestAdmission {
    permits: Arc<Semaphore>,
    max_permits: u32,
    stopped: AtomicBool,
    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Install binds
    // the exact runtime stop bit once; a failed owner cannot leave body intake
    // open while the process supervisor is being scheduled.
    runtime_stop: Mutex<Option<Arc<AtomicBool>>>,
    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Weak entries
    // cannot retain abandoned bodies. Poll, new admission and drain all expire
    // the same buffer owner; no detached timer or extra capacity pool exists.
    responses: crate::api::ReverseOnionResponseRegistry,
}

// [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] An unforgeable local
// extension shares one HTTP slot from pre-auth buffering through the source
// response. It grants no authenticated-owner or durable-work authority.
#[derive(Clone)]
pub(crate) struct ReverseOnionSourceRequestPermit {
    admission: Arc<ReverseOnionSourceRequestAdmission>,
    _permit: Arc<OwnedSemaphorePermit>,
}

// [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] This is a delivery
// deadline, not a task/lease deadline. A truncated HTTP reply remains recoverable
// from the existing authenticated Verified row, without another onion POST.
// [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Preserve the source's
// delivery contract while sharing its bounded body owner with queue HTTP.
#[cfg(test)]
pub(crate) const SOURCE_RESPONSE_TIMEOUT: Duration = crate::api::REVERSE_ONION_RESPONSE_TIMEOUT;
#[cfg(test)]
pub(crate) const SOURCE_RESPONSE_CHUNK_BYTES: usize = crate::api::REVERSE_ONION_RESPONSE_CHUNK_BYTES;

impl ReverseOnionSourceRequestAdmission {
    fn new(max_in_flight: usize) -> Self {
        let bounded = max_in_flight.clamp(1, MAX_SOURCE_IN_FLIGHT);
        Self {
            permits: Arc::new(Semaphore::new(bounded)),
            max_permits: bounded as u32,
            stopped: AtomicBool::new(false),
            runtime_stop: Mutex::new(None),
            responses: crate::api::ReverseOnionResponseRegistry::new(bounded),
        }
    }

    pub(crate) fn try_acquire(self: &Arc<Self>) -> Option<OwnedSemaphorePermit> {
        // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Reclaim
        // expired idle bodies before deciding the shared pool is saturated.
        self.responses.expire();
        if self.is_stopped() { return None; }
        let permit = Arc::clone(&self.permits).try_acquire_owned().ok()?;
        if self.is_stopped() { return None; }
        Some(permit)
    }

    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Clones retain the
    // same permit, never multiply capacity or transfer it to another owner.
    pub(crate) fn try_acquire_request(self: &Arc<Self>) -> Option<ReverseOnionSourceRequestPermit> {
        self.try_acquire().map(|permit| ReverseOnionSourceRequestPermit {
            admission: Arc::clone(self), _permit: Arc::new(permit),
        })
    }

    pub(crate) fn recognizes_request(self: &Arc<Self>, permit: &ReverseOnionSourceRequestPermit) -> bool {
        Arc::ptr_eq(self, &permit.admission) && !self.is_stopped()
    }

    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] An admitted
    // successful reply may finish after stop. Identity, not renewed intake
    // eligibility, binds its existing permit to this exact lifecycle.
    pub(crate) fn bound_response_body(self: &Arc<Self>, body: Body,
        permit: ReverseOnionSourceRequestPermit) -> Result<Body, SourceRuntimeError> {
        if !Arc::ptr_eq(self, &permit.admission) { return Err(SourceRuntimeError::Rejected); }
        self.responses.bound_body(body, permit,
            aeronyx_core::protocol::onion::reverse_delivery::MAX_REVERSE_ONION_SOURCE_PULL_RESPONSE_BYTES,
        ).map_err(|error| match error {
            crate::api::ReverseOnionResponseError::Busy => SourceRuntimeError::Busy,
            crate::api::ReverseOnionResponseError::Unavailable => SourceRuntimeError::Unavailable,
        })
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] A stop can arrive
    // during bounded body buffering. Reject before JSON/auth crypto extraction
    // rather than handing already-stopped work into the source lifecycle.
    pub(crate) fn is_stopped(&self) -> bool {
        if self.stopped.load(Ordering::Acquire) { return true; }
        self.runtime_stop.lock().map(|stop| {
            stop.as_ref().is_some_and(|stop| stop.load(Ordering::Acquire))
        }).unwrap_or(true)
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] No generation
    // replacement/reopening: shutdown drains the existing bound owner.
    fn bind_runtime_stop(&self, stopped: Arc<AtomicBool>) -> Result<(), SourceRuntimeError> {
        let mut slot = self.runtime_stop.lock().map_err(|_| SourceRuntimeError::Unavailable)?;
        if self.stopped.load(Ordering::Acquire) || slot.is_some() || stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Rejected);
        }
        *slot = Some(stopped);
        Ok(())
    }

    fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    async fn shutdown_and_drain(&self) {
        self.request_stop();
        // [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Expire even
        // unpolled bodies without cancelling accepted durable operations.
        self.responses.drain(Arc::clone(&self.permits), self.max_permits).await;
    }
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
            || config.result_wait > MAX_SOURCE_RESULT_WAIT
            || config.result_poll_interval < Duration::from_millis(100)
            || config.result_poll_interval > Duration::from_secs(5)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        // [PHALA-SOURCE-JOURNAL-FAULT 2026-10-07 by Codex] Adopt the DB
        // owner's exact stop/fault pair, including faults after caller loss.
        // A stopped journal is not a new runtime generation; reopen/audit it.
        let stopped = journal.intake_stop_flag();
        let failure = journal.failure_signal();
        if stopped.load(Ordering::Acquire) { return Err(SourceRuntimeError::Stopped); }
        Ok(Self {
            journal,
            identity,
            policy: Some(policy),
            authority_pins: None,
            transport,
            permits: Arc::new(Semaphore::new(config.max_in_flight)),
            max_in_flight: config.max_in_flight as u32,
            timeout: config.timeout,
            stopped,
            failure,
            recovery_only: false,
            active_routes: Arc::new(Mutex::new(HashSet::new())),
            lease_max_secs: config.lease_max_secs,
            result_wait: config.result_wait,
            result_poll_interval: config.result_poll_interval,
        })
    }

    // [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Start inertly;
    // no signed route bundle is required until a caller attempts new work.
    fn new_identity_pinned(
        journal: Arc<ReverseOnionSourceJournal>, identity: Arc<IdentityKeyPair>,
        relay_id: [u8; 32], recipient_id: [u8; 32], relay_endpoint: String,
        peers: Arc<PeerStore>, transport: Arc<dyn SourceTransport>, config: SourceRuntimeConfig,
    ) -> Result<Self, SourceRuntimeError> {
        if relay_id == [0; 32] || recipient_id == [0; 32] || relay_id == recipient_id
            || identity.public_key_bytes() == relay_id || identity.public_key_bytes() == recipient_id
            || !reverse_onion_endpoint_supported(&relay_endpoint)
            || config.max_in_flight == 0 || config.max_in_flight > MAX_SOURCE_IN_FLIGHT
            || config.timeout.is_zero() || config.timeout > MAX_SOURCE_TIMEOUT
            || config.result_wait > MAX_SOURCE_RESULT_WAIT
            || config.result_poll_interval < Duration::from_millis(100)
            || config.result_poll_interval > Duration::from_secs(5)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        // [PHALA-SOURCE-JOURNAL-FAULT 2026-10-07 by Codex] Identity-only
        // bootstrap shares the same DB failure boundary as seeded startup.
        let stopped = journal.intake_stop_flag();
        let failure = journal.failure_signal();
        if stopped.load(Ordering::Acquire) { return Err(SourceRuntimeError::Stopped); }
        Ok(Self {
            journal, identity, policy: None,
            authority_pins: Some(SourceAuthorityPins { relay_id, recipient_id, relay_endpoint, peers }),
            transport, permits: Arc::new(Semaphore::new(config.max_in_flight)),
            max_in_flight: config.max_in_flight as u32, timeout: config.timeout,
            stopped, failure,
            recovery_only: false,
            active_routes: Arc::new(Mutex::new(HashSet::new())),
            lease_max_secs: config.lease_max_secs, result_wait: config.result_wait,
            result_poll_interval: config.result_poll_interval,
        })
    }

    // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] The connected HTTP
    // fixture must use the production identity-pinned constructor and current
    // PeerStore authority, not a static-policy bypass. No production API added.
    #[cfg(test)]
    // [PHALA-SOURCE-API-FIXTURE-AUTHORITY 2026-10-08 by Codex] The API
    // sibling uses this test-only entry too; production visibility is unchanged.
    pub(crate) fn new_identity_pinned_for_test(
        journal: Arc<ReverseOnionSourceJournal>, identity: Arc<IdentityKeyPair>,
        relay_id: [u8; 32], recipient_id: [u8; 32], relay_endpoint: String,
        peers: Arc<PeerStore>, transport: Arc<dyn SourceTransport>, config: SourceRuntimeConfig,
    ) -> Result<Self, SourceRuntimeError> {
        Self::new_identity_pinned(journal, identity, relay_id, recipient_id,
            relay_endpoint, peers, transport, config)
    }

    fn relay_id(&self) -> [u8; 32] {
        self.authority_pins.as_ref().map(|pins| pins.relay_id)
            .or_else(|| self.policy.as_ref().map(|policy| policy.relay_id()))
            .unwrap_or([0; 32])
    }

    fn recipient_id(&self) -> [u8; 32] {
        self.authority_pins.as_ref().map(|pins| pins.recipient_id)
            .or_else(|| self.policy.as_ref().map(|policy| policy.recipient_id()))
            .unwrap_or([0; 32])
    }

    fn current_peers(&self) -> Option<Arc<PeerStore>> {
        self.authority_pins.as_ref().map(|pins| Arc::clone(&pins.peers))
            .or_else(|| self.policy.as_ref().and_then(|policy| policy.current_peers()))
    }

    // [PHALA-SOURCE-AUTHORITY-SNAPSHOT 2026-10-06 by Codex] Materialize one
    // immutable dispatch policy from the current verified R/P/grant tuple.
    fn policy_for_current_authority(
        &self, authorization: SignedPrivateOnionRecipientAuthorizationV1, now: u64,
    ) -> Result<SourcePinnedRelayPolicy, SourceRuntimeError> {
        if let Some(pins) = &self.authority_pins {
            return SourcePinnedRelayPolicy::from_current_snapshot(
                self.identity.public_key_bytes(), pins.relay_id, pins.recipient_id,
                &pins.relay_endpoint, Arc::clone(&pins.peers), authorization, now,
            );
        }
        self.policy.as_ref().ok_or(SourceRuntimeError::Rejected)?
            .with_current_authorization(authorization, now)
    }

    pub(crate) fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Publish no
    // endpoint, route, error details, or payload. send_replace retains failure
    // even when startup has not installed its observer yet.
    fn report_failure(&self) {
        self.request_stop();
        self.failure.send_replace(true);
    }

    /// [REVERSE-ONION-PREPARED-ROTATION 2026-10-05 by Codex] Builds one
    /// fixed-class source Pull from currently pinned signed authority.
    /// Existing armed routes recover only; Prepared may rotate its signed
    /// envelope/session only after the journal proves no relay POST escaped.
    // [REVERSE-SOURCE-TYPED-SUBMIT 2026-10-05 by Codex]
    pub(crate) async fn submit_pull(
        &self,
        route_id: [u8; 16],
        pull: BlindVaultPullRequest,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let _route = ActiveSourceRoute::reserve(Arc::clone(&self.active_routes), route_id)?;
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let observed = observed_now(now)?;
        let phase = self.lookup_phase(route_id, observed).await?;
        if phase.is_some_and(|phase| phase != SourcePhase::Prepared) {
            return self.resume_reserved(route_id, observed).await;
        }
        if self.recovery_only {
            return Err(SourceRuntimeError::Rejected);
        }
        pull.validate().map_err(|_| SourceRuntimeError::Rejected)?;
        if pull.limit != 1 {
            return Err(SourceRuntimeError::Rejected);
        }
        let policy = Arc::new(self.policy_for_current_authority(authorization, observed)?);
        let route = VerifiedOnionRoute::from_signed_private_recipient_descriptors(
            self.identity.public_key_bytes(),
            &policy.relay,
            &policy.recipient,
            &policy.authorization,
            OnionRoutePurpose::BlindVaultPull,
            observed,
        ).map_err(|_| SourceRuntimeError::Rejected)?;
        let (terminal, session) = BlindVaultOnionPullSession::prepare(
            route_id, policy.recipient_id(), pull,
        ).map_err(|_| SourceRuntimeError::Rejected)?;
        let (envelope, expectation) = route.build_envelope_with_forward_expectation(
            &terminal, route_id, observed, &self.identity,
        ).map_err(|_| SourceRuntimeError::Rejected)?;
        let expectation = expectation.ok_or(SourceRuntimeError::Rejected)?;
        let expected = ExpectedRetainedEnvelope::from_verified_forward_expectation(&expectation)
            .map_err(SourceRuntimeError::from)?;
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: self.identity.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let (valid_from, valid_until) = policy.current_validity_bounds();
        let deadline = route.valid_until()
            .min(valid_until)
            .min(observed.checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
                .ok_or(SourceRuntimeError::Rejected)?)
            .min(observed.checked_add(self.lease_max_secs)
                .ok_or(SourceRuntimeError::Rejected)?);
        if observed < valid_from || deadline <= observed {
            return Err(SourceRuntimeError::Rejected);
        }
        // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] The typed
        // path already holds the route; do not recursively reserve it.
        self.dispatch_reserved(
            request,
            expected,
            session,
            deadline,
            terminal,
            observed,
            policy,
        ).await
    }

    /// Resolves a fresh P-signed grant from the process-local verified gossip
    /// cache, then enters the same immutable route/deadline snapshot path.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub(crate) async fn submit_pull_with_live_authority(
        &self,
        route_id: [u8; 16],
        pull: BlindVaultPullRequest,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        // [REVERSE-ONION-RECOVERY-BEFORE-REAUTH 2026-10-05 by Codex] Armed
        // durable work is evidence-only recovery; an expired current grant
        // must not block resuming its historically authorized route. Prepared
        // is the sole existing state eligible for same-route zero-send retry.
        let observed = observed_now(now)?;
        if self.lookup_phase(route_id, observed).await?
            .is_some_and(|phase| phase != SourcePhase::Prepared)
        {
            return self.resume(route_id, observed).await;
        }
        let peers = self.current_peers().ok_or(SourceRuntimeError::Rejected)?;
        let (_, _, authorization) = peers.current_private_onion_authority_snapshot(
                &self.relay_id(), &self.recipient_id(), observed,
            )
            .ok_or(SourceRuntimeError::Rejected)?;
        self.submit_pull(route_id, pull, authorization, observed).await
    }

    async fn lookup_phase(
        &self,
        route: [u8; 16],
        now: u64,
    ) -> Result<Option<SourcePhase>, SourceRuntimeError> {
        let permit = Arc::new(self.permits.clone().try_acquire_owned()
            .map_err(|_| SourceRuntimeError::Busy)?);
        source_db(&self.journal, permit, now, move |journal, now| {
            journal.lookup_phase(route, now).map_err(SourceRuntimeError::from)
        }).await
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

    /// Restart recovery may replay only the exact durable POST to the same
    /// still-authorized relay before the immutable route deadline. It never
    /// creates a new request or changes relay, recipient, or authorization.
    pub(crate) async fn resume(
        &self,
        route: [u8; 16],
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let _route = ActiveSourceRoute::reserve(Arc::clone(&self.active_routes), route)?;
        self.resume_reserved(route, now).await
    }

    // [REVERSE-ONION-RESUME-SINGLE-FLIGHT 2026-10-05 by Codex] The caller
    // owns the route reservation, so submit and recovery share one exclusion
    // boundary without recursively acquiring the same route lock.
    async fn resume_reserved(
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
        let (metadata, observed_at) = self.metadata(route, observed_at, Arc::clone(&permit)).await?;
        if metadata.source() != self.identity.public_key_bytes()
            || metadata.relay() != self.relay_id()
            || metadata.recipient() != self.recipient_id()
            || metadata.target() != self.recipient_id()
        {
            return Err(SourceRuntimeError::Unavailable);
        }
        if observed_at >= metadata.retain_until() {
            return Err(SourceRuntimeError::Expired);
        }
        match metadata.phase() {
            // [SOURCE-OPEN-RECOVERY 2026-10-05 by Codex] Resume a retained
            // local result after crash without another relay POST.
            SourcePhase::ResultReady | SourcePhase::Opening => {
                self.open_result(route, observed_at, Arc::clone(&permit)).await
            }
            SourcePhase::Verified => self.read_verified(route, observed_at, Arc::clone(&permit)).await,
            SourcePhase::Armed => {
                let observed_at = self.replay_armed_once(route, observed_at, Arc::clone(&permit)).await?;
                self.collect_and_open(route, observed_at, Arc::clone(&permit)).await
            }
            // An observed ambiguous transport outcome never gets another POST.
            SourcePhase::DispatchAmbiguous => {
                self.collect_and_open(route, observed_at, Arc::clone(&permit)).await
            }
            // Prepared proves the one-shot arm/send boundary was never crossed.
            SourcePhase::Prepared => Err(SourceRuntimeError::Rejected),
            SourcePhase::Rejected | SourcePhase::OpenAmbiguous => {
                Err(SourceRuntimeError::Ambiguous)
            }
        }
    }

    /// Arms once and sends the exact durable JSON body. An observed outcome,
    /// including a custody ACK, permits evidence recovery only. Exact replay
    /// remains reserved for the unobserved Armed crash window at the same relay.
    pub(crate) async fn dispatch(
        &self,
        request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope,
        session: BlindVaultOnionPullSession,
        admitted_deadline: u64,
        terminal_request: Vec<u8>,
        now: u64,
        policy: Arc<SourcePinnedRelayPolicy>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Direct
        // internal callers must share typed submit/resume's route exclusion.
        let _route = ActiveSourceRoute::reserve(
            Arc::clone(&self.active_routes), request.envelope.route_id,
        )?;
        self.dispatch_reserved(request, expected, session, admitted_deadline,
            terminal_request, now, policy).await
    }

    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] The caller holds
    // the route through POST observation, evidence recovery and reply opening.
    async fn dispatch_reserved(
        &self,
        request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope,
        session: BlindVaultOnionPullSession,
        admitted_deadline: u64,
        terminal_request: Vec<u8>,
        now: u64,
        policy: Arc<SourcePinnedRelayPolicy>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        if terminal_request.is_empty() || terminal_request.len() > MAX_SOURCE_TERMINAL_BYTES {
            return Err(SourceRuntimeError::Rejected);
        }
        // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Reject before permits,
        // preparation, durable arming, or transport. Historical work uses resume.
        if self.recovery_only { return Err(SourceRuntimeError::Rejected); }
        let permit = Arc::new(
            self.permits
                .clone()
                .try_acquire_owned()
                .map_err(|_| SourceRuntimeError::Busy)?,
        );
        if self.stopped.load(Ordering::Acquire) {
            return Err(SourceRuntimeError::Stopped);
        }
        let relay_url = policy.relay_url(BLIND_RELAY_PATH)?;
        // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Check
        // the local observation even when resolution fails before any POST.
        let resolved = resolve_pinned_peer_http_target(relay_url, self.timeout).await;
        let preflight_at = observed_now(now)?;
        let target = resolved.map_err(|_| SourceRuntimeError::Rejected)?;
        let admission = self.prepare_and_arm(
            request,
            expected,
            session,
            admitted_deadline,
            terminal_request,
            preflight_at,
            Arc::clone(&permit),
            Arc::clone(&policy),
            target,
        ).await?;
        let (request, route, _deadline, armed_at, target, exact_bytes, authorization_bytes) = match admission {
            SourceAdmission::Send(admission) => (
                admission.request,
                admission.route,
                admission.deadline,
                admission.observed_at,
                admission.target,
                admission.exact_bytes,
                admission.authorization_bytes,
            ),
            SourceAdmission::Recover { route, observed_at } => {
                let observed_at = self.replay_armed_once(route, observed_at, Arc::clone(&permit)).await?;
                return self.collect_and_open(route, observed_at, Arc::clone(&permit)).await;
            }
            SourceAdmission::Complete { route, observed_at } => {
                return self.read_verified(route, observed_at, Arc::clone(&permit)).await;
            }
        };
        if exact_bytes.len() > MAX_SOURCE_DISPATCH_BYTES {
            let observed_at = observed_now(armed_at)?;
            self.restore_prepared_after_no_send(route, observed_at, Arc::clone(&permit)).await?;
            return Err(SourceRuntimeError::Rejected);
        }
        if self.stopped.load(Ordering::Acquire) {
            let observed_at = observed_now(armed_at)?;
            self.restore_prepared_after_no_send(route, observed_at, Arc::clone(&permit)).await?;
            return Err(SourceRuntimeError::Stopped);
        }
        let policy_for_post = Arc::clone(&policy);
        let blocking_permit = Arc::clone(&permit);
        let post_gate = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
            // Sample after authority ownership, not before its lock wait.
            let peers = policy_for_post.current_peers();
            let _authority_epoch = peers.as_ref()
                .map(|store| store.private_onion_authority_read_guard());
            let post_now = observed_now(armed_at)?;
            let gate = (|| {
                if post_now >= _deadline {
                    return Err(SourceRuntimeError::Expired);
                }
                policy_for_post.validate_at_under_authority_guard(post_now)?;
                let (policy_from, policy_until) = policy_for_post.current_validity_bounds();
                // Successful validation proves the private authorization's
                // issued-at bound without exposing it as a public field.
                Ok((post_now.max(policy_from), policy_until))
            })();
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
            // A known no-send rejection also retains the checked local floor.
            Ok::<_, SourceRuntimeError>((gate, post_now))
        })
        .await;
        let post_gate = match post_gate {
            Ok(gate) => gate,
            Err(_) => {
                // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
                // A failed local owner is not a fresh Prepared retry permit.
                self.report_failure();
                return Err(SourceRuntimeError::Unavailable);
            }
        };
        let (post_gate, checked_at) = match post_gate {
            Ok(value) => value,
            Err(error) => {
                self.report_failure();
                return Err(error);
            }
        };
        let (valid_from, valid_until) = match post_gate {
            Ok(value) => value,
            Err(SourceRuntimeError::Unavailable) => {
                self.report_failure();
                return Err(SourceRuntimeError::Unavailable);
            }
            // The network call has not started. Do not label a known local
            // preflight rejection as uncertain remote acceptance.
            Err(error) => {
                let observed_at = observed_now(checked_at)?;
                self.restore_prepared_after_no_send(route, observed_at, Arc::clone(&permit)).await?;
                return Err(error);
            }
        };
        let post_now = observed_now(valid_from)?;
        if self.stopped.load(Ordering::Acquire)
            || post_now >= _deadline
            || post_now >= valid_until
        {
            self.restore_prepared_after_no_send(route, post_now, Arc::clone(&permit)).await?;
            return Err(if self.stopped.load(Ordering::Acquire) {
                SourceRuntimeError::Stopped
            } else {
                SourceRuntimeError::Expired
            });
        }
        let transport_observation = Arc::new(ReverseOnionLocalObservation::new(post_now));
        let response = match tokio::time::timeout(
            self.timeout,
            self.transport.post(target, Bytes::from(exact_bytes), Bytes::from(authorization_bytes),
                SourceSendAdmission { stopped: Arc::clone(&self.stopped), floor: Arc::clone(&transport_observation),
                    deadline: _deadline.min(valid_until), poll_deadline: None,
                    authority: SourceSendAuthority::Dispatch(Arc::clone(&policy)) }),
        )
        .await
        {
            Ok(response) => response,
            Err(_) => SourceTransportOutcome::Ambiguous,
        };
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] A timeout
        // drops the carrier future, not its latest floor or sticky clock fault.
        let post_now = source_transport_completion_now(&transport_observation)
            .map_err(|error| { self.report_failure(); error })?;
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] This is the first
        // attempt of a newly armed request, so proven no-send may restore
        // Prepared. A clock fault instead closes intake and retains Armed.
        if let SourceTransportOutcome::NotSent(ref proof) = response {
            if proof.error == SourceRuntimeError::Unavailable {
                self.report_failure();
            } else {
                let observed_at = observed_now(post_now)?;
                self.restore_prepared_after_no_send(route, observed_at, Arc::clone(&permit)).await?;
            }
            return Err(proof.error);
        }
        // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Record the
        // observation BEFORE decoding an ACK or polling evidence. Pending,
        // verification failure, or cancellation cannot leave observed custody
        // replay-eligible as an unobserved Armed crash window.
        let observed_at = self.record_post_observation(route, post_now, Arc::clone(&permit)).await?;
        match response {
            SourceTransportOutcome::NotSent(proof) => Err(proof.error),
            SourceTransportOutcome::Ambiguous => {
                Err(SourceRuntimeError::Ambiguous)
            }
            SourceTransportOutcome::Response { status, body } => {
                let request_for_verify = request;
                let relay = policy.relay_id();
                let blocking_permit = Arc::clone(&permit);
                let (response, verified_at) = match tokio::task::spawn_blocking(move || {
                    let _permit = blocking_permit;
                    // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
                    // ACK verification can wait after durable observation.
                    let verified_at = observed_now(observed_at)?;
                    let response = decode_peer_response(status, &body)?;
                    verify_success_response(&request_for_verify, &response, relay, verified_at)?;
                    Ok::<_, SourceRuntimeError>((response, verified_at))
                })
                .await
                // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex]
                // Observation is already durable before ACK verification,
                // including when this verifier task fails.
                .unwrap_or(Err(SourceRuntimeError::Unavailable))
                {
                    Ok(response) => response,
                    Err(error) => return Err(error),
                };
                let _ = response;
                self.collect_and_open(route, verified_at, Arc::clone(&permit)).await
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
        policy: Arc<SourcePinnedRelayPolicy>,
        target: PinnedPeerHttpTarget,
    ) -> Result<SourceAdmission, SourceRuntimeError> {
        let authorization_bytes = policy.authorization.encode_canonical()
            .map_err(|_| SourceRuntimeError::Rejected)?;
        let identity = Arc::clone(&self.identity);
        let stopped = Arc::clone(&self.stopped);
        let request_for_plan = request.clone();
        let route = request_for_plan.envelope.route_id;
        let result = source_db(&self.journal, permit, now, move |journal, admission_now| {
            // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] A stop can
            // arrive while waiting for completion writers or the blocking pool.
            if stopped.load(Ordering::Acquire) { return Err(SourceRuntimeError::Stopped); }
            // [REVERSE-ONION-SOURCE-AUTHORITY-FENCE 2026-10-05 by Codex]
            // Hold the same signed-authority epoch from final validation
            // through durable Prepared->Armed. The guard ends before network
            // I/O; once Armed, recovery preserves that exact request.
            let _authority_epoch = policy.current_peers.as_ref()
                .map(|peers| peers.private_onion_authority_read_guard());
            let admission_now = observed_now(admission_now)?;
            policy.validate_at_under_authority_guard(admission_now)?;
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
            if stopped.load(Ordering::Acquire) { return Err(SourceRuntimeError::Stopped); }
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
            // Existing-route recovery must also inherit Prepared's SQL sample.
            let mut prepared_at = admission_now;
            let phase = journal
                .prepare_at(plan, session, || {
                    let checked_at = source_admission_now(&stopped, admission_now,
                        || observed_now(0))?;
                    prepared_at = checked_at;
                    Ok(checked_at)
                })
                .map_err(|error| source_admission_error(error, &stopped))?;
            match phase {
                SourcePhase::Prepared => {
                    if stopped.load(Ordering::Acquire) { return Err(SourceRuntimeError::Stopped); }
                    // [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex]
                    // Recheck after Prepared's sealing/commit/fence, under the
                    // Armed transaction. Failure preserves zero-send Prepared.
                    let dispatch = journal.arm_at(request_for_plan.envelope.route_id,
                        || source_admission_now(&stopped, prepared_at, || observed_now(0)))
                        .map_err(|error| source_admission_error(error, &stopped))?;
                    let persisted_request: PeerBlindRelayRequest =
                        serde_json::from_slice(&dispatch.exact_bytes)
                            .map_err(|_| SourceRuntimeError::Unavailable)?;
                    if persisted_request.envelope.route_id != route {
                        return Err(SourceRuntimeError::Unavailable);
                    }
                    Ok(SourceAdmission::Send(PreparedDispatch {
                        exact_bytes: dispatch.exact_bytes,
                        request: persisted_request,
                        route,
                        deadline: dispatch.deadline,
                        observed_at: dispatch.observed_at,
                        target,
                        authorization_bytes,
                    }))
                }
                SourcePhase::Armed | SourcePhase::DispatchAmbiguous
                | SourcePhase::ResultReady | SourcePhase::Opening => {
                    Ok(SourceAdmission::Recover { route, observed_at: prepared_at })
                }
                SourcePhase::Verified => Ok(SourceAdmission::Complete { route, observed_at: prepared_at }),
                SourcePhase::Rejected | SourcePhase::OpenAmbiguous => {
                    Err(SourceRuntimeError::Ambiguous)
                }
            }
        })
        .await?;
        Ok(result)
    }

    async fn mark_ambiguous(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<u64, SourceRuntimeError> {
        // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Observed remote
        // uncertainty must not silently remain replay-eligible Armed in this
        // live owner. Stop further work if the durable transition cannot finish.
        let result = source_db(&self.journal, permit, now, move |journal, now| {
            journal.mark_dispatch_ambiguous(route, now).map_err(SourceRuntimeError::from)?;
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
            // Preserve the completion lane's observation after its SQL fence.
            Ok(now)
        }).await;
        if result.is_err() { self.report_failure(); }
        result
    }

    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] The existing
    // phase means execution unresolved, not that custody was rejected. Keep
    // schema/wire compatibility; no observation may authorize another POST.
    async fn record_post_observation(
        &self, route: [u8; 16], now: u64, permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<u64, SourceRuntimeError> {
        let observed_at = match observed_now(now) {
            Ok(observed_at) => observed_at,
            Err(error) => {
                self.report_failure();
                return Err(error);
            }
        };
        self.mark_ambiguous(route, observed_at, permit).await
    }

    // [REVERSE-SOURCE-SAME-RELAY-RECOVERY 2026-10-05 by Codex] Resolve the
    // crash window after durable arming by replaying only identical canonical
    // bytes to the original relay. Relay queue admission is keyed by the
    // authenticated source/route/request tuple and returns existing state;
    // the result is still established only through signed evidence queries.
    async fn replay_armed_once(
        &self,
        route: [u8; 16],
        fallback_now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<u64, SourceRuntimeError> {
        let now = observed_now(fallback_now)?;
        let (metadata, now) = self.metadata(route, now, Arc::clone(&permit)).await?;
        if metadata.phase() != SourcePhase::Armed
            || now >= metadata.deadline()
            || self.stopped.load(Ordering::Acquire)
        {
            return Ok(now);
        }

        let (authority, now) = source_db(&self.journal, Arc::clone(&permit), now, move |journal, now| {
            journal.recover_route_authority(route, now)
                .map(|authority| (authority, now))
                .map_err(SourceRuntimeError::from)
        })
        .await?;
        let (relay, recipient, authorization) = authority
            .signed_parts()
            .map_err(SourceRuntimeError::from)?;
        let mut policy = SourcePinnedRelayPolicy::new_for_recovery(
            self.identity.public_key_bytes(), relay, recipient, authorization,
        )?;
        if let Some(peers) = self.current_peers() {
            policy = policy.with_current_peers(peers);
        }
        if let Some(pins) = &self.authority_pins {
            policy = policy.with_relay_endpoint_pin(pins.relay_endpoint.clone());
        }
        let policy = Arc::new(policy);

        // Rotated, revoked, or expired authority permits evidence recovery,
        // but never a fresh POST. The current R/P/grant must exactly match.
        let gate_policy = Arc::clone(&policy);
        let gate_permit = Arc::clone(&permit);
        let (authorized, authorized_at) = tokio::task::spawn_blocking(move || {
            let _permit = gate_permit;
            let peers = gate_policy.current_peers();
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] The
            // authority lock wait must not reuse the pre-wait observation.
            let _authority_epoch = peers.as_ref()
                .map(|store| store.private_onion_authority_read_guard());
            let checked_at = observed_now(now)?;
            let authorized = gate_policy.validate_at_under_authority_guard(checked_at).is_ok();
            Ok::<_, SourceRuntimeError>((authorized, checked_at))
        })
        .await
        .map_err(|_| SourceRuntimeError::Unavailable)??;
        if !authorized { return Ok(authorized_at); }

        let dispatch = source_db(&self.journal, Arc::clone(&permit), authorized_at, move |journal, now| {
            journal.recover_dispatch_at(route, || observed_now(now)
                .map_err(|_| SourceJournalError::Unavailable))
                .map_err(SourceRuntimeError::from)
        })
        .await?;
        if dispatch.deadline != metadata.deadline()
            || dispatch.request_commitment != metadata.request_commitment()
        {
            return Err(SourceRuntimeError::Unavailable);
        }
        let request: PeerBlindRelayRequest = serde_json::from_slice(dispatch.exact_bytes.as_slice())
            .map_err(|_| SourceRuntimeError::Unavailable)?;
        if request.envelope.route_id != route
            || request.previous_hop_node_id != self.identity.public_key_bytes()
            || request.envelope.next_hop != policy.relay_id()
            || blind_relay_authenticated_request_commitment(&request)
                .map_err(|_| SourceRuntimeError::Unavailable)? != dispatch.request_commitment
        {
            return Err(SourceRuntimeError::Unavailable);
        }
        let canonical_authorization = policy.authorization.encode_canonical()
            .map_err(|_| SourceRuntimeError::Unavailable)?;
        if canonical_authorization.as_slice() != dispatch.authorization_bytes.as_slice() {
            return Err(SourceRuntimeError::Unavailable);
        }

        // DNS pinning and TLS hostname validation are identical to first send;
        // no endpoint supplied by the recovered row is trusted independently.
        let url = policy.relay_url(BLIND_RELAY_PATH)?;
        let resolved = resolve_pinned_peer_http_target(url, self.timeout).await;
        let post_now = observed_now(dispatch.observed_at)?;
        let target = match resolved {
            Ok(target) => target,
            Err(_) => return Ok(post_now),
        };
        if self.stopped.load(Ordering::Acquire)
            || post_now >= dispatch.deadline
            || policy.validate_at(post_now).is_err()
        {
            return Ok(post_now);
        }

        let exact_body = Bytes::copy_from_slice(dispatch.exact_bytes.as_slice());
        let exact_authorization = Bytes::copy_from_slice(dispatch.authorization_bytes.as_slice());
        // The response is deliberately not treated as task completion. Even a
        // verified custody ACK is followed by the source's signed evidence poll.
        let transport_observation = Arc::new(ReverseOnionLocalObservation::new(post_now));
        let response = tokio::time::timeout(
            self.timeout,
            self.transport.post(target, exact_body, exact_authorization,
                SourceSendAdmission { stopped: Arc::clone(&self.stopped), floor: Arc::clone(&transport_observation),
                    deadline: dispatch.deadline.min(policy.current_validity_bounds().1), poll_deadline: None,
                    authority: SourceSendAuthority::Dispatch(Arc::clone(&policy)) }),
        )
        .await;
        let post_now = source_transport_completion_now(&transport_observation)
            .map_err(|error| { self.report_failure(); error })?;
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] A no-send proof for
        // this retry says nothing about the original crash window. Preserve
        // Armed; never restore a historical attempt to fresh Prepared work.
        if let Ok(SourceTransportOutcome::NotSent(proof)) = response {
            if proof.error == SourceRuntimeError::Unavailable {
                self.report_failure();
                return Err(proof.error);
            }
            if matches!(proof.error, SourceRuntimeError::Stopped | SourceRuntimeError::Busy) {
                return Err(proof.error);
            }
            return Ok(observed_now(post_now)?);
        }
        self.record_post_observation(route, post_now, Arc::clone(&permit)).await
    }

    // [REVERSE-SOURCE-ZERO-SEND 2026-10-05 by Codex] Roll back only before
    // entering HTTP, including an explicit transport no-send proof on the
    // first attempt only. A failed rollback is conservative: the
    // durable Armed row becomes ambiguous on restart, never replayed.
    async fn restore_prepared_after_no_send(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<(), SourceRuntimeError> {
        source_db(&self.journal, permit, now, move |journal, now| {
            journal.restore_prepared_after_no_send(route, now).map_err(SourceRuntimeError::from)
        })
        .await
    }

    async fn collect_and_open(
        &self,
        route: [u8; 16],
        fallback_now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Advance
        // one trusted local floor through all three evidence parts and retries.
        let mut now = observed_now(fallback_now)?;
        let (metadata, metadata_at) = self.metadata(route, now, Arc::clone(&permit)).await?;
        now = metadata_at;
        let deadline = metadata.deadline();
        if !matches!(metadata.phase(), SourcePhase::Armed | SourcePhase::DispatchAmbiguous) {
            return match metadata.phase() {
                SourcePhase::Verified => self.read_verified(route, now, Arc::clone(&permit)).await,
                // [SOURCE-OPEN-RECOVERY 2026-10-05 by Codex] Opening already
                // has durable result evidence; resume only the local open.
                SourcePhase::ResultReady | SourcePhase::Opening => {
                    self.open_result(route, now, Arc::clone(&permit)).await
                }
                SourcePhase::Rejected => Err(SourceRuntimeError::Rejected),
                _ => Err(SourceRuntimeError::Ambiguous),
            };
        }
        // [REVERSE-SOURCE-HISTORIC-AUTH 2026-10-05 by Codex] Query the exact
        // authority sealed with this armed route, not a newer global config.
        // It is usable only for signed evidence reads and only while retained.
        let (authority, authority_at) = source_db(&self.journal, Arc::clone(&permit), now, move |journal, now| {
            journal.recover_route_authority(route, now)
                .map(|authority| (authority, now))
                .map_err(SourceRuntimeError::from)
        })
        .await?;
        now = authority_at;
        let (relay, recipient, authorization) = authority
            .signed_parts()
            .map_err(SourceRuntimeError::from)?;
        let mut route_policy = SourcePinnedRelayPolicy::new_for_recovery(
            self.identity.public_key_bytes(), relay, recipient, authorization,
        )?;
        if let Some(peers) = self.current_peers() {
            route_policy = route_policy.with_current_peers(peers);
        }
        if let Some(pins) = &self.authority_pins {
            route_policy = route_policy.with_relay_endpoint_pin(pins.relay_endpoint.clone());
        }
        let route_policy = Arc::new(route_policy);
        if metadata.source() != self.identity.public_key_bytes()
            || metadata.relay() != route_policy.relay_id()
            || metadata.recipient() != route_policy.recipient_id()
            || metadata.target() != route_policy.recipient_id()
            || metadata.relay_descriptor_commitment() != route_policy.relay_descriptor_commitment()?
            || metadata.recipient_descriptor_commitment() != route_policy.descriptor_commitment()?
        {
            return Err(SourceRuntimeError::Unavailable);
        }
        route_policy.validate_relay_at(now)?;
        // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Poll only inside
        // the trusted result grace, never the longer recovery-retention horizon.
        let evidence_deadline = metadata.result_deadline().min(metadata.retain_until());
        let poll_deadline = (!self.result_wait.is_zero())
            .then(|| std::time::Instant::now() + self.result_wait);
        let mut parts = Vec::with_capacity(3);
        'poll: loop {
            parts.clear();
            let mut pending = false;
            for part in [SourceEvidencePartV1::Claim, SourceEvidencePartV1::Lease, SourceEvidencePartV1::Result] {
                now = observed_now(now)?;
                if now >= evidence_deadline { return Err(SourceRuntimeError::Expired); }
                if poll_deadline.is_some_and(|until| std::time::Instant::now() >= until) {
                    return Err(SourceRuntimeError::Pending);
                }
                if self.stopped.load(Ordering::Acquire) {
                    self.mark_ambiguous(route, now, Arc::clone(&permit)).await?;
                    return Err(SourceRuntimeError::Stopped);
                }
                let query = self
                    .sign_query(
                        route,
                        metadata.request_commitment(),
                        part,
                        now,
                        Arc::clone(&permit),
                        Arc::clone(&route_policy),
                    )
                    .await?;
                let (query, signed_at) = query;
                let body = query.encode();
                let route_now = observed_now(signed_at)?;
                let (url, relay_snapshot) = route_policy
                    .recovery_relay_snapshot(SOURCE_QUERY_PATH, route_now)?;
                let query_timeout = source_query_timeout(
                    route_now,
                    evidence_deadline,
                    poll_deadline,
                    self.timeout,
                )?;
                let resolved = resolve_pinned_peer_http_target(url, query_timeout).await;
                let send_now = observed_now(route_now)?;
                now = send_now;
                let target = match resolved {
                    Ok(target) => target,
                    Err(_) => {
                        if now >= evidence_deadline {
                            return Err(SourceRuntimeError::Expired);
                        }
                        if poll_deadline.is_some_and(|until| std::time::Instant::now() >= until) {
                            return Err(SourceRuntimeError::Pending);
                        }
                        // [REVERSE-ONION-SOURCE-READ-RETRY 2026-10-05 by Codex]
                        // DNS/connect failure occurred before this idempotent
                        // GET. Retain the same route and retry within its
                        // already bounded result-poll window.
                        pending = true;
                        break;
                    }
                };
                if send_now >= evidence_deadline { return Err(SourceRuntimeError::Expired); }
                // [PHALA-PRIVATE-SOURCE-ROUTE-GATE 2026-10-06 by Codex]
                // DNS pinning can await past appraisal expiry; recheck the
                // same current relay immediately before the read-only GET.
                route_policy.validate_recovery_relay_snapshot(relay_snapshot, send_now)?;
                if self.stopped.load(Ordering::Acquire) {
                    self.mark_ambiguous(route, send_now, Arc::clone(&permit)).await?;
                    return Err(SourceRuntimeError::Stopped);
                }
                if send_now >= signed_at
                    .checked_add(REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS)
                    .ok_or(SourceRuntimeError::Rejected)?
                {
                    self.mark_ambiguous(route, send_now, Arc::clone(&permit)).await?;
                    return Err(SourceRuntimeError::Expired);
                }
                let query_timeout = source_query_timeout(
                    send_now,
                    evidence_deadline,
                    poll_deadline,
                    self.timeout,
                )?;
                let transport_observation = Arc::new(ReverseOnionLocalObservation::new(send_now));
                let response = match tokio::time::timeout(
                    query_timeout,
                    self.transport.query(target, Bytes::from(body), SourceSendAdmission {
                        stopped: Arc::clone(&self.stopped), floor: Arc::clone(&transport_observation),
                        deadline: evidence_deadline.min(signed_at.checked_add(
                            REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS).ok_or(SourceRuntimeError::Rejected)?),
                        poll_deadline,
                        authority: SourceSendAuthority::Evidence {
                            policy: Arc::clone(&route_policy), relay_snapshot,
                        },
                    }),
                )
                .await
                {
                    Ok(response) => response,
                    Err(_) => {
                        now = source_transport_completion_now(&transport_observation)
                            .map_err(|error| { self.report_failure(); error })?;
                        if now >= evidence_deadline {
                            return Err(SourceRuntimeError::Expired);
                        }
                        if poll_deadline.is_some_and(|until| std::time::Instant::now() >= until) {
                            return Err(SourceRuntimeError::Pending);
                        }
                        SourceTransportOutcome::Ambiguous
                    }
                };
                now = source_transport_completion_now(&transport_observation)
                    .map_err(|error| { self.report_failure(); error })?;
                let bytes = match response {
                    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] A
                    // read-only query that never entered HTTP cannot change
                    // custody state or authorize another task POST.
                    SourceTransportOutcome::NotSent(proof) => {
                        if proof.error == SourceRuntimeError::Unavailable { self.report_failure(); }
                        return Err(proof.error);
                    }
                    SourceTransportOutcome::Response { status: 200, body } => body,
                    outcome => {
                        if source_query_outcome_is_retryable(&outcome) {
                            // Evidence queries are read-only. A lost response
                            // cannot make the already-armed task ambiguous anew.
                            pending = true;
                            break;
                        } else {
                            let observed_at = observed_now(now)?;
                            self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await?;
                            return Err(SourceRuntimeError::Ambiguous);
                        }
                    }
                };
                if bytes.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                    let observed_at = observed_now(now)?;
                    self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await?;
                    return Err(SourceRuntimeError::Ambiguous);
                }
                let blocking_permit = Arc::clone(&permit);
                let (verified, verified_at) = tokio::task::spawn_blocking(move || {
                    let _permit = blocking_permit;
                    let observed_at = observed_now(now)?;
                    // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
                    // Pending and malformed remote evidence still return the
                    // checked local floor; neither may erase an observation.
                    let verified = (|| {
                        let evidence = ReverseOnionSourceEvidenceV1::decode(&bytes)
                            .map_err(|_| SourceRuntimeError::Ambiguous)?;
                        let evidence_state = evidence.verify_for_query(&query, observed_at)
                            .map_err(|_| SourceRuntimeError::Ambiguous)?;
                        let state = evidence_state.state();
                        drop(evidence_state);
                        match state {
                            SourceEvidenceStateV1::Pending => Ok(None),
                            SourceEvidenceStateV1::Available => Ok(Some((query, evidence))),
                            SourceEvidenceStateV1::Unavailable => Err(SourceRuntimeError::Ambiguous),
                        }
                    })();
                    Ok::<_, SourceRuntimeError>((verified, observed_at))
                })
                .await
                .map_err(|_| SourceRuntimeError::Unavailable)??;
                now = verified_at;
                let (query, evidence) = match verified {
                    Ok(Some(pair)) => pair,
                    Ok(None) => {
                        pending = true;
                        break;
                    }
                    // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex]
                    // Only untrusted evidence becomes protocol ambiguity. A
                    // local clock failure must reach the owned fault boundary.
                    Err(SourceRuntimeError::Ambiguous) => {
                        let observed_at = observed_now(now)?;
                        self.mark_ambiguous(route, observed_at, Arc::clone(&permit)).await?;
                        return Err(SourceRuntimeError::Ambiguous);
                    }
                    Err(error) => return Err(error),
                };
                if evidence.encode().len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
                    self.mark_ambiguous(route, now, Arc::clone(&permit)).await?;
                    return Err(SourceRuntimeError::Ambiguous);
                }
                parts.push((query, evidence));
            }
            if !pending { break; }
            now = observed_now(now)?;
            if now >= evidence_deadline {
                return Err(SourceRuntimeError::Expired);
            }
            let Some(wait_until) = poll_deadline else {
                return Err(SourceRuntimeError::Pending);
            };
            let wait_remaining = wait_until.saturating_duration_since(std::time::Instant::now());
            if wait_remaining.is_zero() { return Err(SourceRuntimeError::Pending); }
            tokio::time::sleep(self.result_poll_interval.min(wait_remaining)).await;
            if self.stopped.load(Ordering::Acquire) {
                return Err(SourceRuntimeError::Stopped);
            }
            continue 'poll;
        }
        let blocking_permit = Arc::clone(&permit);
        let chain = tokio::task::spawn_blocking(move || {
            let _permit = blocking_permit;
            let chain_now = observed_now(now)?;
            // [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] Verification
            // scheduling cannot extend the journal's immutable recovery bound.
            if chain_now >= evidence_deadline {
                return Err(SourceRuntimeError::Expired);
            }
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
        let (chain, chain_now) = chain;
        let opened = source_db(&self.journal, Arc::clone(&permit), chain_now, move |journal, record_now| {
            // [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] The journal
            // lane can wait after chain verification; recheck before mutation.
            if record_now >= evidence_deadline {
                return Err(SourceRuntimeError::Expired);
            }
            // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] Recheck the
            // immutable source bound after SQL lock/audit wait as well as lane
            // admission. A clock failure retains journal fault supervision.
            let mut last_checked_at = record_now;
            journal.record_result_at(route, chain.claim(), chain.lease(), chain.result(), || {
                let checked_at = observed_now(record_now)
                    .map_err(|_| SourceJournalError::Unavailable)?;
                if checked_at >= evidence_deadline {
                    return Err(SourceJournalError::Expired);
                }
                last_checked_at = checked_at;
                Ok(checked_at)
            })
                .map_err(SourceRuntimeError::from)?;
            let open_now = observed_now(last_checked_at)?;
            if open_now >= evidence_deadline {
                return Err(SourceRuntimeError::Expired);
            }
            // [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] The same
            // lane/permit owns both crypto fences; sample again under SQL
            // ownership instead of extrapolating this pre-lock observation.
            journal.open_result_at(route, open_now, source_result_clock)
                .map_err(SourceRuntimeError::from)
        })
        .await?;
        Ok(BlindVaultPullResult { response: opened })
    }

    async fn sign_query(
        &self,
        route: [u8; 16],
        request_commitment: [u8; 32],
        part: SourceEvidencePartV1,
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
        policy: Arc<SourcePinnedRelayPolicy>,
    ) -> Result<(ReverseOnionSourceQueryV1, u64), SourceRuntimeError> {
        let identity = Arc::clone(&self.identity);
        let relay = policy.relay_id();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let signed_at = observed_now(now)?;
            policy.validate_relay_at(signed_at)?;
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
    ) -> Result<(SourceRecoveryMetadata, u64), SourceRuntimeError> {
        let page = source_db(&self.journal, permit, now, move |journal, now| {
            let mut after = None;
            for _ in 0..16 {
                let page = journal.recover_metadata(after, 64, now).map_err(SourceRuntimeError::from)?;
                for item in page.items {
                    if item.route() == route {
                        // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
                        // Recovery carries the journal lane's checked sample.
                        return Ok((item, now));
                    }
                }
                let Some(next) = page.next_after else { break; };
                after = Some(next);
            }
            Err(SourceRuntimeError::Rejected)
        })
            .await?;
        Ok(page)
    }

    async fn read_verified(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let response = source_db(&self.journal, permit, now, move |journal, observed_at| {
            journal
                // [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] A
                // retained Verified row does not bypass live lock-time checks.
                .read_verified_at(route, observed_at, source_result_clock)
                .map_err(SourceRuntimeError::from)
        })
            .await?;
        Ok(BlindVaultPullResult { response })
    }

    async fn open_result(
        &self,
        route: [u8; 16],
        now: u64,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let response = source_db(&self.journal, permit, now, move |journal, observed_at| {
            journal
                // [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] Restart
                // Opening repeats only local crypto with both live fences.
                .open_result_at(route, observed_at, source_result_clock)
                .map_err(SourceRuntimeError::from)
        })
            .await?;
        Ok(BlindVaultPullResult { response })
    }
}

// [REVERSE-ONION-SOURCE-LIFECYCLE 2026-10-04 by Codex] Composition injects
// one already-authenticated runtime; this owner adds only admission/ownership
// sequencing. It never creates a key, opens a journal, selects an endpoint,
// or exposes a transport route. The runtime's semaphore accounts for blocking
// children/network work; its request gate bounds pre-body API admission.
pub(crate) struct ReverseOnionSourceLifecycle {
    runtime: Mutex<Option<Arc<ReverseOnionSourceRuntime>>>,
    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Retain the
    // signal, not the journal, after drain; graceful release cannot turn a
    // pending observer into a false missing-owner fault or erase a real fault.
    failure: Mutex<Option<watch::Sender<bool>>>,
    request_admission: Arc<ReverseOnionSourceRequestAdmission>,
    owned_operations: Mutex<OwnedSourceOperations>,
    max_owned_operations: usize,
    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] An owned panic
    // closes this exact lifecycle gate, including after its HTTP waiter left.
    stopped: Arc<AtomicBool>,
    drain_waiter: tokio::sync::Mutex<()>,
}

// [REVERSE-ONION-SOURCE-OWNED-OPERATIONS 2026-10-05 by Codex] Once an
// authenticated request begins source work, its HTTP waiter is not the task
// owner. Retain task handles until completion or server drain.
struct OwnedSourceOperations {
    closed: bool,
    tasks: Vec<JoinHandle<()>>,
    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Preserve both
    // caught unwinds and join failures across reaping/cancelled drain waiters.
    failed: Arc<AtomicBool>,
}

impl ReverseOnionSourceLifecycle {
    pub(crate) fn new() -> Self {
        Self::with_request_limit(MAX_SOURCE_IN_FLIGHT)
    }

    pub(crate) fn with_request_limit(max_in_flight: usize) -> Self {
        // [PHALA-SOURCE-CONSTRUCTOR-BOUND 2026-10-08 by Codex] Server::new
        // precedes config validation. Derive both ownership pools from the
        // same bounded admission; Server::run still rejects invalid input.
        let request_admission = Arc::new(ReverseOnionSourceRequestAdmission::new(max_in_flight));
        let max_owned_operations = request_admission.max_permits as usize;
        Self {
            runtime: Mutex::new(None),
            failure: Mutex::new(None),
            request_admission,
            owned_operations: Mutex::new(OwnedSourceOperations {
                closed: false,
                tasks: Vec::new(),
                failed: Arc::new(AtomicBool::new(false)),
            }),
            max_owned_operations,
            stopped: Arc::new(AtomicBool::new(false)),
            drain_waiter: tokio::sync::Mutex::new(()),
        }
    }

    pub(crate) fn request_admission(&self) -> Arc<ReverseOnionSourceRequestAdmission> {
        Arc::clone(&self.request_admission)
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
        if self.stopped.load(Ordering::Acquire) || slot.is_some()
            || runtime.stopped.load(Ordering::Acquire)
        {
            return Err(SourceRuntimeError::Rejected);
        }
        let mut failure = self.failure.lock().map_err(|_| SourceRuntimeError::Unavailable)?;
        if failure.is_some() { return Err(SourceRuntimeError::Rejected); }
        // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Link the
        // actual pre-body gate before publishing runtime readiness.
        self.request_admission.bind_runtime_stop(Arc::clone(&runtime.stopped))?;
        *failure = Some(runtime.failure.clone());
        *slot = Some(runtime);
        Ok(())
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Missing or
    // poisoned installed ownership cannot satisfy the server's READY gate.
    pub(crate) fn has_failed(&self) -> bool {
        let owned_failed = self.owned_operations.lock()
            .map(|owner| owner.failed.load(Ordering::Acquire)).unwrap_or(true);
        owned_failed || self.failure.lock()
            .map(|failure| failure.as_ref().map_or(true, |failure| *failure.borrow()))
            .unwrap_or(true)
    }

    // [PHALA-READY-PUBLICATION 2026-10-07 by Codex] A normally stopped
    // source is not failed, but cannot advertise readiness. Check installed
    // ownership and the actual intake gates, not current route authority:
    // discovery may legitimately renew that authority after startup.
    pub(super) fn verify_ready_now(&self) -> Result<(), SourceRuntimeError> {
        let readiness = self.runtime_for_work().and_then(|runtime| {
            if runtime.stopped.load(Ordering::Acquire) || self.request_admission.is_stopped() {
                Err(SourceRuntimeError::Stopped)
            } else {
                Ok(())
            }
        });
        // Sticky journal/owned-task faults take priority over normal stop,
        // including faults reported while the readiness snapshot was taken.
        if self.has_failed() { Err(SourceRuntimeError::Unavailable) } else { readiness }
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Wait for an
    // actual failure, not a normal stop. No spawned observer or detached task;
    // the server selects this future alongside its existing shutdown waits.
    pub(crate) async fn wait_for_failure(&self) {
        let receiver = self.failure.lock().ok()
            .and_then(|failure| failure.as_ref().map(|failure| failure.subscribe()));
        if let Some(mut failure) = receiver {
            loop {
                if *failure.borrow_and_update() { break; }
                if failure.changed().await.is_err() { break; }
            }
        }
        self.request_stop();
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

    // [REVERSE-ONION-SOURCE-OWNED-OPERATIONS 2026-10-05 by Codex] API
    // cancellation drops only this result receiver. The spawned operation
    // retains the runtime's durable journal and its own admission permit.
    async fn run_owned<T, F>(&self, operation: F) -> Result<T, SourceRuntimeError>
    where
        T: Send + 'static,
        F: Future<Output = Result<T, SourceRuntimeError>> + Send + 'static,
    {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Snapshot the
        // installed owner; replacement is forbidden until stop and full drain.
        let operation_runtime = self.runtime.lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?.clone();
        let operation_stopped = Arc::clone(&self.stopped);
        let operation_admission = Arc::clone(&self.request_admission);
        {
            let mut owner = self
                .owned_operations
                .lock()
                .map_err(|_| SourceRuntimeError::Unavailable)?;
            owner.tasks.retain(|task| !task.is_finished());
            if owner.closed || self.stopped.load(Ordering::Acquire) {
                return Err(SourceRuntimeError::Stopped);
            }
            if owner.tasks.len() >= self.max_owned_operations {
                return Err(SourceRuntimeError::Busy);
            }
            let panic_failed = Arc::clone(&owner.failed);
            owner.tasks.push(tokio::spawn(async move {
                let result = match std::panic::AssertUnwindSafe(operation).catch_unwind().await {
                    Ok(result) => result,
                    Err(_) => {
                        // The unwind cannot prove transport never started.
                        // Preserve Armed for authenticated restart recovery,
                        // but never let this live owner attempt it again.
                        panic_failed.store(true, Ordering::SeqCst);
                        Err(SourceRuntimeError::Unavailable)
                    }
                };
                // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] A
                // normally returned local fault is just as fatal to intake as
                // an unwind, even after the HTTP waiter disconnects. Publish
                // before returning; preserve journal phases and completion
                // lanes for ordered drain/authenticated restart recovery.
                if matches!(&result, Err(SourceRuntimeError::Unavailable)) {
                    operation_stopped.store(true, Ordering::SeqCst);
                    operation_admission.request_stop();
                    if let Some(runtime) = operation_runtime.as_ref() { runtime.report_failure(); }
                }
                let _ = result_tx.send(result);
            }));
        }
        result_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?
    }

    async fn drain_owned_operations(&self) -> Result<(), SourceRuntimeError> {
        self.owned_operations
            .lock()
            .map_err(|_| SourceRuntimeError::Unavailable)?
            .closed = true;
        // [REVERSE-ONION-SOURCE-DRAIN-OWNERSHIP 2026-10-05 by Codex] Poll
        // handles in-place and remove only completed tasks. If this drain
        // future is cancelled, its handles and failure state remain owned by
        // the lifecycle for the next drain attempt.
        futures::future::poll_fn(|cx| {
            let mut owner = match self.owned_operations.lock() {
                Ok(owner) => owner,
                Err(_) => return Poll::Ready(Err(SourceRuntimeError::Unavailable)),
            };
            let mut index = 0;
            while index < owner.tasks.len() {
                match Pin::new(&mut owner.tasks[index]).poll(cx) {
                    Poll::Pending => index += 1,
                    Poll::Ready(result) => {
                        let _completed = owner.tasks.swap_remove(index);
                        if result.is_err() { owner.failed.store(true, Ordering::SeqCst); }
                    }
                }
            }
            if owner.tasks.is_empty() {
                Poll::Ready(if owner.failed.load(Ordering::Acquire) {
                    Err(SourceRuntimeError::Unavailable)
                } else {
                    Ok(())
                })
            } else {
                Poll::Pending
            }
        })
        .await
    }

    pub(crate) fn request_stop(&self) {
        self.request_admission.request_stop();
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
        policy: Arc<SourcePinnedRelayPolicy>,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let runtime = self.runtime_for_work()?;
        self.run_owned(async move {
            runtime.dispatch(
                request,
                expected,
                session,
                admitted_deadline,
                terminal_request,
                now,
                policy,
            )
            .await
        })
        .await
    }

    pub(crate) async fn submit_pull(
        &self,
        route: [u8; 16],
        pull: BlindVaultPullRequest,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let runtime = self.runtime_for_work()?;
        self.run_owned(async move {
            runtime.submit_pull(route, pull, authorization, now).await
        })
        .await
    }

    /// Resolves current P-signed authority from the live PeerStore for callers
    /// that do not transport public authorization bytes themselves.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub(crate) async fn submit_pull_with_live_authority(
        &self,
        route: [u8; 16],
        pull: BlindVaultPullRequest,
        now: u64,
    ) -> Result<BlindVaultPullResult, SourceRuntimeError> {
        let runtime = self.runtime_for_work()?;
        self.run_owned(async move {
            runtime.submit_pull_with_live_authority(route, pull, now).await
        })
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
        self.run_owned(async move { runtime.resume(route, now).await }).await
    }

    /// Closes new work first, then retains the injected runtime/journal until
    /// its semaphore reaches zero. Only after that point is the owner slot
    /// released for shutdown teardown.
    pub(crate) async fn shutdown_and_drain(&self) -> Result<(), SourceRuntimeError> {
        let _waiter = self.drain_waiter.lock().await;
        self.request_stop();
        self.request_admission.shutdown_and_drain().await;
        let operations = self.drain_owned_operations().await;
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
        // [REVERSE-ONION-SOURCE-DRAIN-FAILURE 2026-10-05 by Codex] A child
        // join failure remains reportable, but must not skip the runtime's
        // semaphore drain or release its journal before those children end.
        operations
    }
}

struct ActiveSourceRoute {
    routes: Arc<Mutex<HashSet<[u8; 16]>>>,
    route: [u8; 16],
}

impl ActiveSourceRoute {
    fn reserve(
        routes: Arc<Mutex<HashSet<[u8; 16]>>>,
        route: [u8; 16],
    ) -> Result<Self, SourceRuntimeError> {
        let mut active = routes.lock().map_err(|_| SourceRuntimeError::Unavailable)?;
        if !active.insert(route) { return Err(SourceRuntimeError::Busy); }
        Ok(Self { routes: Arc::clone(&routes), route })
    }
}

impl Drop for ActiveSourceRoute {
    fn drop(&mut self) {
        if let Ok(mut active) = self.routes.lock() { active.remove(&self.route); }
    }
}

impl Drop for ReverseOnionSourceLifecycle {
    fn drop(&mut self) {
        self.request_stop();
    }
}

enum SourceAdmission {
    Send(PreparedDispatch),
    Recover { route: [u8; 16], observed_at: u64 },
    Complete { route: [u8; 16], observed_at: u64 },
}

struct PreparedDispatch {
    exact_bytes: Vec<u8>,
    request: PeerBlindRelayRequest,
    route: [u8; 16],
    deadline: u64,
    // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Preserve
    // Prepared->Armed's checked SQL time across post-commit scheduling.
    observed_at: u64,
    target: PinnedPeerHttpTarget,
    authorization_bytes: Vec<u8>,
}

// [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] These floors
// are trusted local observations, never remote request timestamps. Do not
// substitute a fallback or clamp rollback into apparent forward progress.
fn observed_now(floor: u64) -> Result<u64, SourceRuntimeError> {
    checked_source_observation(floor, SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| SourceRuntimeError::Unavailable))
}

// [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] Journal compares this
// raw local sample with the lane/previous/durable floors before advancing its
// separate monotonic expiry bound. Never replace an OS failure with an anchor.
fn source_result_clock() -> Result<u64, SourceJournalError> {
    observed_now(0).map_err(|_| SourceJournalError::Unavailable)
}

// [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Only trusted local
// observations enter this attempt; remote receipt timestamps never set it.
fn source_transport_completion_now(observation: &ReverseOnionLocalObservation)
    -> Result<u64, SourceRuntimeError> {
    observation.observe(observed_now(0).map_err(|_| ()))
        .map_err(|_| SourceRuntimeError::Unavailable)
}

fn checked_source_observation(floor: u64, observation: Result<u64, SourceRuntimeError>)
    -> Result<u64, SourceRuntimeError> {
    let now = observation?;
    if now < floor { return Err(SourceRuntimeError::Unavailable); }
    Ok(now)
}

// [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] This callback runs
// inside Prepared/Armed SQL ownership, while the route authority epoch is held.
// Stop is a zero-send admission rejection, not a journal fault. Clock failure
// or regression still reaches the journal's sticky failure supervision.
pub(super) fn source_admission_now(
    stopped: &AtomicBool, floor: u64,
    clock: impl FnOnce() -> Result<u64, SourceRuntimeError>,
) -> Result<u64, SourceJournalError> {
    let now = clock().map_err(|_| SourceJournalError::Unavailable)?;
    if now < floor { return Err(SourceJournalError::ClockRollback); }
    if stopped.load(Ordering::Acquire) { return Err(SourceJournalError::Rejected); }
    Ok(now)
}

// [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] Do not mask an
// integrity/clock fault just because shutdown raced its publication.
fn source_admission_error(error: SourceJournalError, stopped: &AtomicBool) -> SourceRuntimeError {
    if error == SourceJournalError::Rejected && stopped.load(Ordering::Acquire) {
        SourceRuntimeError::Stopped
    } else {
        SourceRuntimeError::from(error)
    }
}

// [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Each resolution or
// evidence request is clipped to both the caller's bounded poll window and
// the authenticated result-grace deadline. The request timeout is only a cap.
fn source_query_timeout(
    now: u64,
    evidence_deadline: u64,
    poll_deadline: Option<std::time::Instant>,
    request_timeout: Duration,
) -> Result<Duration, SourceRuntimeError> {
    if now >= evidence_deadline {
        return Err(SourceRuntimeError::Expired);
    }
    let evidence_remaining = Duration::from_secs(evidence_deadline - now);
    let mut timeout = request_timeout.min(evidence_remaining);
    if let Some(poll_deadline) = poll_deadline {
        let poll_remaining = poll_deadline.saturating_duration_since(std::time::Instant::now());
        if poll_remaining.is_zero() {
            return Err(SourceRuntimeError::Pending);
        }
        timeout = timeout.min(poll_remaining);
    }
    if timeout.is_zero() {
        return Err(SourceRuntimeError::Expired);
    }
    Ok(timeout)
}

// [REVERSE-ONION-SOURCE-READ-RETRY 2026-10-05 by Codex] Only explicit
// overload, timeout, and server-failure statuses are retried; protocol/auth
// rejections remain fail-closed. The outer loop enforces both time horizons.
fn source_query_status_is_retryable(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}

fn source_query_outcome_is_retryable(outcome: &SourceTransportOutcome) -> bool {
    match outcome {
        SourceTransportOutcome::NotSent(_) => false,
        SourceTransportOutcome::Ambiguous => true,
        SourceTransportOutcome::Response { status, .. } => {
            source_query_status_is_retryable(*status)
        }
    }
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

    // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Authored only:
    // production composition cannot publish a runtime for another journal
    // owner or a future/noncanonical durable clock. No exchange is invoked.
    #[cfg(unix)]
    #[tokio::test]
    async fn configured_source_open_rejects_custody_metadata_without_replacement() {
        use rusqlite::{params, types::Value, Connection};
        use crate::config_reverse_onion::ReverseOnionSourceConfig;
        use crate::services::reverse_onion_source::SourceJournalLimits;
        for scenario in 0..4 {
            let directory = tempfile::Builder::new().prefix("phala-source-open-owner-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("source.sqlite");
            let identity = Arc::new(IdentityKeyPair::from_bytes(&[91; 32]).unwrap());
            let relay = IdentityKeyPair::from_bytes(&[92; 32]).unwrap().public_key_bytes();
            let recipient = IdentityKeyPair::from_bytes(&[93; 32]).unwrap().public_key_bytes();
            let now = observed_now(0).unwrap();
            drop(ReverseOnionSourceJournal::open(&path, identity.clone(),
                SourceJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now).unwrap());
            let snapshot = {
                let connection = Connection::open(&path).unwrap();
                if scenario == 2 {
                    connection.execute("UPDATE source_meta SET clock=?1", params![i64::MAX - 1]).unwrap();
                } else if scenario == 3 {
                    connection.execute("UPDATE source_meta SET source=?1",
                        params![hex::encode(identity.public_key_bytes())]).unwrap();
                }
                connection.query_row("SELECT source,clock FROM source_meta", [],
                    |r| Ok((r.get::<_, Value>(0)?, r.get::<_, Value>(1)?))).unwrap()
            };
            let mut config = ReverseOnionSourceConfig::default();
            config.enabled = true;
            config.recovery_only = true;
            config.state_db_path = path.to_string_lossy().into_owned();
            config.relay_node_id = hex::encode(relay);
            config.recipient_node_id = hex::encode(recipient);
            config.relay_endpoint = "https://1.1.1.1".into();
            config.max_pending_items = 4;
            let running_identity = if scenario == 1 {
                Arc::new(IdentityKeyPair::from_bytes(&[94; 32]).unwrap())
            } else { identity };
            let peers = Arc::new(PeerStore::new());
            let opened = open_configured_source(&config, running_identity, peers.clone()).await;
            if scenario == 0 {
                opened.unwrap().shutdown_and_drain().await;
            } else {
                assert_eq!(opened.err(), Some(SourceRuntimeError::Unavailable));
                let connection = Connection::open(&path).unwrap();
                let after = connection.query_row("SELECT source,clock FROM source_meta", [],
                    |r| Ok((r.get::<_, Value>(0)?, r.get::<_, Value>(1)?))).unwrap();
                assert_eq!(after, snapshot);
                let count: i64 = connection.query_row("SELECT count(*) FROM source_jobs", [], |r| r.get(0)).unwrap();
                assert_eq!(count, 0);
            }
            assert!(peers.is_empty(), "identity pins do not create route authority");
        }
    }

    // [PHALA-SOURCE-RETENTION-ADMISSION 2026-10-08 by Codex] Authored,
    // not run: the actual prepare/arm producer reclaims a full expired slot
    // after restart; stopped admission cannot remove or replace that custody.
    #[cfg(unix)]
    #[tokio::test]
    async fn fresh_runtime_admission_reclaims_expired_capacity_but_stop_preserves_it() {
        use crate::services::reverse_onion_source::tests::Fixture;
        for stopped in [false, true] {
            let fresh = Fixture::new_for_runtime();
            // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex] Expire
            // the complete custody horizon, not merely the execution grace.
            let old = Fixture::for_retention_admission(17, fresh.now() - 600
                - aeronyx_core::protocol::onion::reverse_delivery::MAX_REVERSE_ONION_RECOVERY_RETENTION_SECS);
            let journal = old.open_one_slot(old.now());
            old.prepare(&journal, old.now());
            journal.arm(old.route(), old.now() + 1).unwrap();
            drop(journal);
            let journal = Arc::new(old.open_one_slot(fresh.now()));
            assert_eq!(journal.lookup_phase(old.route(), fresh.now()).unwrap(), Some(SourcePhase::Armed));
            let posts = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let runtime = fixture_runtime(&fresh, Arc::new(CountingTransport {
                posts: posts.clone(), queries: queries.clone(),
            }), journal.clone(), false);
            let policy = runtime.policy.as_ref().unwrap().clone();
            // No DNS/socket request is needed to exercise this durable producer.
            let target = PinnedPeerHttpTarget {
                client: crate::api::privacy_safe_peer_http_client_builder().build().unwrap(),
                url: policy.relay_url(BLIND_RELAY_PATH).unwrap(),
            };
            let (request, expected, session, terminal) = fresh.runtime_admission_parts();
            let permit = Arc::new(runtime.permits.clone().acquire_owned().await.unwrap());
            if stopped { runtime.request_stop(); }
            let result = runtime.prepare_and_arm(request, expected, session, fresh.now() + 600,
                terminal, fresh.now(), permit, policy, target).await;
            if stopped {
                assert!(matches!(result, Err(SourceRuntimeError::Stopped)));
            } else {
                assert!(matches!(result, Ok(SourceAdmission::Send(dispatch))
                    if dispatch.route == fresh.route() && dispatch.exact_bytes == fresh.outbound_bytes()));
            }
            let now = observed_now(fresh.now()).unwrap();
            assert_eq!(journal.lookup_phase(old.route(), now).unwrap(),
                if stopped { Some(SourcePhase::Armed) } else { None });
            assert_eq!(journal.lookup_phase(fresh.route(), now).unwrap(),
                if stopped { None } else { Some(SourcePhase::Armed) });
            assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(queries.load(AtomicOrdering::SeqCst), 0);
            runtime.shutdown_and_drain().await;
        }
    }

    // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Authored,
    // not run: reject rollback below the latest observation, even if a later
    // sample remains above the original operation's starting time.
    #[test]
    fn source_observation_uses_the_latest_floor_without_fallback_or_clamping() {
        let first = checked_source_observation(100, Ok(100)).unwrap();
        let next = checked_source_observation(first, Ok(110)).unwrap();
        assert_eq!(checked_source_observation(next, Ok(110)), Ok(110));
        assert_eq!(checked_source_observation(next, Ok(105)), Err(SourceRuntimeError::Unavailable));
        assert_eq!(checked_source_observation(next, Err(SourceRuntimeError::Unavailable)),
            Err(SourceRuntimeError::Unavailable));
        assert_eq!(observed_now(u64::MAX), Err(SourceRuntimeError::Unavailable));
    }

    // [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] Authored, not
    // run: shutdown rejects new writes but cannot downgrade a clock failure.
    #[test]
    fn source_admission_clock_preserves_faults_and_samples_stop_after_clock() {
        let stopped = AtomicBool::new(false);
        assert_eq!(source_admission_now(&stopped, 10, || Ok(10)), Ok(10));
        assert_eq!(source_admission_now(&stopped, 10, || {
            stopped.store(true, Ordering::Release);
            Ok(11)
        }), Err(SourceJournalError::Rejected));
        assert_eq!(source_admission_error(SourceJournalError::Rejected, &stopped), SourceRuntimeError::Stopped);
        assert_eq!(source_admission_now(&stopped, 10, || Ok(9)), Err(SourceJournalError::ClockRollback));
        assert_eq!(source_admission_now(&stopped, 10, || Err(SourceRuntimeError::Unavailable)),
            Err(SourceJournalError::Unavailable));
        for fault in [SourceJournalError::ClockRollback, SourceJournalError::Corrupt, SourceJournalError::Unavailable] {
            assert_eq!(source_admission_error(fault, &stopped), SourceRuntimeError::Unavailable);
        }
    }

    // [PHALA-SOURCE-CONSTRUCTOR-BOUND 2026-10-08 by Codex] Authored, not
    // run: invalid direct construction cannot expand owned-task capacity
    // beyond the HTTP/response pool, including usize::MAX before validation.
    #[test]
    fn source_constructor_uses_one_bounded_capacity_for_both_owners() {
        for requested in [0, 1, 4, MAX_SOURCE_IN_FLIGHT, MAX_SOURCE_IN_FLIGHT + 1, usize::MAX] {
            let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(requested);
            let expected = requested.clamp(1, MAX_SOURCE_IN_FLIGHT);
            let admission = lifecycle.request_admission();
            assert_eq!(lifecycle.max_owned_operations, expected);
            assert_eq!(admission.max_permits as usize, expected);
            let permits: Vec<_> = (0..expected)
                .map(|_| admission.try_acquire().unwrap()).collect();
            assert!(admission.try_acquire().is_none());
            drop(permits);
            assert_eq!(admission.permits.available_permits(), expected);
            lifecycle.request_stop();
            assert!(admission.try_acquire().is_none());
        }
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
    // not run: runtime binding is once-only and fail-closed after stop, without
    // holding a runtime/journal Arc in the pre-body admission gate.
    #[test]
    fn request_gate_cannot_rebind_or_reopen_a_stopped_runtime() {
        let gate = Arc::new(ReverseOnionSourceRequestAdmission::new(1));
        let stopped = Arc::new(AtomicBool::new(true));
        assert_eq!(gate.bind_runtime_stop(stopped.clone()), Err(SourceRuntimeError::Rejected));
        let live = Arc::new(AtomicBool::new(false));
        gate.bind_runtime_stop(live.clone()).unwrap();
        assert_eq!(gate.bind_runtime_stop(Arc::new(AtomicBool::new(false))), Err(SourceRuntimeError::Rejected));
        let permit = gate.try_acquire().unwrap();
        live.store(true, Ordering::SeqCst);
        assert!(gate.is_stopped());
        drop(permit);
        assert!(gate.try_acquire().is_none());
        assert_eq!(gate.bind_runtime_stop(Arc::new(AtomicBool::new(false))), Err(SourceRuntimeError::Rejected));
        let closed = ReverseOnionSourceRequestAdmission::new(1);
        closed.request_stop();
        assert_eq!(closed.bind_runtime_stop(Arc::new(AtomicBool::new(false))), Err(SourceRuntimeError::Rejected));
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
    // not run: a fault before any subscriber remains visible; a graceful stop
    // alone never completes the failure observer or permits replacement.
    #[cfg(unix)]
    #[tokio::test]
    async fn sticky_source_fault_survives_late_observer_but_normal_stop_is_not_a_fault() {
        for fault in [false, true] {
            let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
            let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
                posts: Arc::new(AtomicUsize::new(0)), queries: Arc::new(AtomicUsize::new(0)),
            }), Arc::new(fixture.open(fixture.now())), false);
            let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
            lifecycle.install(runtime.clone()).unwrap();
            if fault { runtime.report_failure(); } else { lifecycle.request_stop(); }
            assert_eq!(lifecycle.has_failed(), fault);
            assert!(lifecycle.request_admission().is_stopped());
            assert_eq!(lifecycle.install(runtime.clone()), Err(SourceRuntimeError::Rejected));
            let mut observer = Box::pin(lifecycle.wait_for_failure());
            if fault {
                assert!(futures::poll!(observer.as_mut()).is_ready());
            } else {
                assert!(futures::poll!(observer.as_mut()).is_pending());
            }
            drop(observer);
            let fresh = ReverseOnionSourceLifecycle::with_request_limit(1);
            assert_eq!(fresh.install(runtime), Err(SourceRuntimeError::Rejected));
            lifecycle.shutdown_and_drain().await.unwrap();
            assert_eq!(lifecycle.has_failed(), fault);
        }
    }

    // [PHALA-SOURCE-AUTHORITY-SNAPSHOT 2026-10-06 by Codex] Authored, not
    // executed: fixed identity pins alone cannot authorize dispatch; a
    // current signed R/P/grant tuple and the exact operator origin are needed.
    #[cfg(unix)]
    #[test]
    fn identity_only_source_requires_current_gossiped_authority_snapshot() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, authorization) = fixture.policy_parts();
        let relay_id = relay.node_id();
        let recipient_id = recipient.node_id();
        let source = fixture.source_identity().public_key_bytes();
        let peers = Arc::new(PeerStore::new());
        peers.pin_private_onion_route_identities(relay_id, recipient_id).unwrap();
        assert!(SourcePinnedRelayPolicy::from_current_snapshot(
            source, relay_id, recipient_id,
            relay.descriptor.public_endpoint.as_deref().unwrap(),
            Arc::clone(&peers), authorization.clone(), fixture.now(),
        ).is_err());

        peers.import_private_onion_authorization_bundle(
            authorization.clone(), relay.clone(), recipient.clone(), source, fixture.now(),
        ).unwrap();
        assert!(SourcePinnedRelayPolicy::from_current_snapshot(
            source, relay_id, recipient_id,
            relay.descriptor.public_endpoint.as_deref().unwrap(),
            Arc::clone(&peers), authorization.clone(), fixture.now(),
        ).is_ok());
        assert!(SourcePinnedRelayPolicy::from_current_snapshot(
            source, relay_id, recipient_id, "https://different.example.net",
            peers, authorization, fixture.now(),
        ).is_err());
    }

    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] Source-side
    // authority reconstruction must reject a signed public-discovery policy.
    #[test]
    fn recovery_policy_rejects_publicly_discoverable_private_recipient() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, authorization) = fixture.policy_parts();
        let recipient_identity = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
        let mut public_policy = recipient.descriptor.clone();
        public_policy.policy.public_discovery = true;
        let public_policy = SignedNodeDescriptor::sign(public_policy, &recipient_identity).unwrap();
        assert!(SourcePinnedRelayPolicy::new_for_recovery(
            fixture.source_identity().public_key_bytes(), relay, public_policy, authorization,
        )
        .is_err());
    }

    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Authored, not run:
    // the configured window is a wall-clock bound, not a fresh allowance after
    // the first Pending response, and the immutable result grace wins expiry.
    #[test]
    fn source_query_timeout_uses_the_earliest_trusted_bound() {
        let now = 1_800_000_000;
        assert_eq!(
            source_query_timeout(now, now + 2, None, Duration::from_secs(5)),
            Ok(Duration::from_secs(2)),
        );
        let timeout = source_query_timeout(
            now,
            now + 20,
            Some(std::time::Instant::now() + Duration::from_secs(1)),
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(timeout > Duration::ZERO && timeout <= Duration::from_secs(1));
        assert_eq!(
            source_query_timeout(now + 20, now + 20, None, Duration::from_secs(5)),
            Err(SourceRuntimeError::Expired),
        );
        assert_eq!(
            source_query_timeout(
                now,
                now + 20,
                Some(std::time::Instant::now()),
                Duration::from_secs(5),
            ),
            Err(SourceRuntimeError::Pending),
        );
    }

    // [REVERSE-ONION-SOURCE-READ-RETRY 2026-10-05 by Codex] Authored, not run:
    // retry only bounded transient read failures, never peer protocol rejects.
    #[test]
    fn source_evidence_retry_statuses_are_narrow_and_explicit() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(source_query_status_is_retryable(status));
        }
        for status in [200, 301, 400, 401, 404, 422, 501, 505, 599] {
            assert!(!source_query_status_is_retryable(status));
        }
        assert!(source_query_outcome_is_retryable(&SourceTransportOutcome::Ambiguous));
        assert!(!source_query_outcome_is_retryable(&SourceTransportOutcome::Response {
            status: 401,
            body: Vec::new(),
        }));
    }

    // [REVERSE-SOURCE-STARTUP-SEED 2026-10-05 by Codex] Authored, not run.
    #[cfg(unix)]
    #[test]
    fn older_config_seed_never_replaces_or_blocks_a_newer_peer_descriptor() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, _) = fixture.policy_parts();
        let relay_identity = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let mut current = relay.descriptor.clone();
        current.sequence += 1;
        let current = SignedNodeDescriptor::sign(current, &relay_identity).unwrap();
        let peers = PeerStore::new();
        peers.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
        peers.upsert_verified(current.clone(), fixture.now()).unwrap();

        peers.seed_private_onion_route_descriptor(
            &relay.node_id(), &recipient.node_id(), relay, fixture.now(), "test_seed",
        ).unwrap();

        assert_eq!(peers.get_valid(&current.node_id(), fixture.now()), Some(current));
    }

    // [REVERSE-RECOVERY-ENDPOINT-PIN 2026-10-06 by Codex] Authored, not
    // executed: same-origin signed renewals are accepted for evidence reads;
    // endpoint changes cannot redirect historical evidence.
    #[cfg(unix)]
    #[test]
    fn recovery_accepts_same_origin_renewal_and_rejects_endpoint_rotation() {
        assert!(SourcePinnedRelayPolicy::same_reverse_onion_endpoint(
            "https://RELAY.example.net:443/old-path",
            "https://relay.example.net/",
        ));
        assert!(!SourcePinnedRelayPolicy::same_reverse_onion_endpoint(
            "https://relay.example.net:443",
            "https://relay.example.net:8443",
        ));
        // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Matching two
        // unsafe origins must not satisfy historical recovery's endpoint pin.
        for endpoint in ["http://relay.example.net", "https://relay.internal",
            "https://127.0.0.1", "https://user@relay.example.net"] {
            assert!(!SourcePinnedRelayPolicy::same_reverse_onion_endpoint(endpoint, endpoint));
        }
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, authorization) = fixture.policy_parts();
        let peers = Arc::new(PeerStore::new());
        let policy = SourcePinnedRelayPolicy {
            source: fixture.source_identity().public_key_bytes(), relay: relay.clone(),
            recipient, authorization, current_peers: Some(Arc::clone(&peers)),
            relay_endpoint_pin: None,
        };
        let relay_identity = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let mut refreshed = relay.descriptor.clone();
        refreshed.sequence += 1;
        let refreshed = SignedNodeDescriptor::sign(refreshed, &relay_identity).unwrap();
        peers.upsert_verified(refreshed.clone(), fixture.now()).unwrap();
        assert!(policy.validate_relay_at(fixture.now()).is_ok());
        assert!(policy.validate_at(fixture.now()).is_err());
        let (_, initial_snapshot) = policy
            .recovery_relay_snapshot(SOURCE_QUERY_PATH, fixture.now()).unwrap();
        assert!(policy.validate_recovery_relay_snapshot(initial_snapshot, fixture.now()).is_ok());
        // [REVERSE-SOURCE-RECOVERY-SNAPSHOT 2026-10-07 by Codex] Authored,
        // not executed: even same-origin renewal during DNS invalidates the
        // old target snapshot; a later retry can obtain the renewed snapshot.
        let mut renewed = refreshed.descriptor.clone();
        renewed.sequence += 1;
        let renewed = SignedNodeDescriptor::sign(renewed, &relay_identity).unwrap();
        peers.upsert_verified(renewed.clone(), fixture.now()).unwrap();
        assert!(policy.validate_relay_at(fixture.now()).is_ok());
        assert_eq!(policy.validate_recovery_relay_snapshot(initial_snapshot, fixture.now()),
            Err(SourceRuntimeError::Rejected));
        let (_, renewed_snapshot) = policy
            .recovery_relay_snapshot(SOURCE_QUERY_PATH, fixture.now()).unwrap();
        assert_ne!(initial_snapshot, renewed_snapshot);
        assert!(policy.validate_recovery_relay_snapshot(renewed_snapshot, fixture.now()).is_ok());
        let mut moved = renewed.descriptor.clone();
        moved.sequence += 1;
        moved.public_endpoint = Some("https://relay.example.net:443".to_owned());
        let moved = SignedNodeDescriptor::sign(moved, &relay_identity).unwrap();
        peers.upsert_verified(moved, fixture.now()).unwrap();
        assert!(policy.validate_relay_at(fixture.now()).is_err());
        assert!(policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, fixture.now()).is_err());
        // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] A signed public-IP
        // endpoint is still rejected over HTTP: relay custody receipts and
        // source replies require authenticated transport, not just identity.
        let mut cleartext = peers.get_valid(&relay.node_id(), fixture.now()).unwrap().descriptor;
        cleartext.sequence += 1;
        cleartext.public_endpoint = Some("http://8.8.8.8:8422".to_owned());
        peers.upsert_verified(
            SignedNodeDescriptor::sign(cleartext, &relay_identity).unwrap(),
            fixture.now(),
        ).unwrap();
        assert!(policy.validate_relay_at(fixture.now()).is_err());
        let mut private = peers.get_valid(&relay.node_id(), fixture.now()).unwrap().descriptor;
        private.sequence += 1;
        private.public_endpoint = Some("https://127.0.0.1:443".to_owned());
        peers.upsert_verified(SignedNodeDescriptor::sign(private, &relay_identity).unwrap(), fixture.now()).unwrap();
        assert!(policy.validate_relay_at(fixture.now()).is_err());
    }

    // [PHALA-PRIVATE-SOURCE-ROUTE-GATE 2026-10-06 by Codex] Authored, not
    // executed: direct pinned source-pull POST and recovery GET obey the same
    // strict appraisal freshness as scored blind-relay candidates.
    #[cfg(unix)]
    #[test]
    fn private_source_dispatch_and_recovery_require_fresh_phala_appraisal() {
        use aeronyx_core::protocol::discovery::NodeProtocolFeature;

        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, _) = fixture.policy_parts();
        let relay_key = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let recipient_key = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
        let now = fixture.now();
        let relay = SignedNodeDescriptor::sign(
            relay.descriptor
                .with_protocol_features([NodeProtocolFeature::PhalaNodeAttestationV1]),
            &relay_key,
        )
        .unwrap();
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay,
            &recipient,
            OnionRoutePurpose::BlindVaultPull.as_str(),
            now,
            now + 600,
            &recipient_key,
        )
        .unwrap();
        let peers = Arc::new(PeerStore::new());
        peers.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
        // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Install the
        // real signed fresh-dispatch authority, not only an isolated R pin.
        peers.upsert_verified_from_source(recipient.clone(), now, "test_pin").unwrap();
        peers.remember_issued_private_onion_authorization(
            authorization.clone(), recipient.node_id(), now,
        ).unwrap();
        peers.configure_phala_attested_peer_routes(true, 300);
        let policy = Arc::new(SourcePinnedRelayPolicy {
            source: fixture.source_identity().public_key_bytes(),
            relay: relay.clone(),
            recipient,
            authorization,
            current_peers: Some(Arc::clone(&peers)),
            relay_endpoint_pin: relay.descriptor.public_endpoint.clone(),
        });

        assert!(policy.validate_pinned_at(now).is_err());
        assert!(policy.validate_relay_at(now).is_err());
        assert!(policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, now).is_err());

        assert!(peers.record_phala_peer_attestation(&relay, now));
        assert!(policy.validate_pinned_at(now).is_ok());
        assert!(policy.validate_relay_at(now).is_ok());
        assert!(policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, now).is_ok());

        assert!(policy.validate_pinned_at(now + 301).is_err());
        assert!(policy.validate_relay_at(now + 301).is_err());
        assert!(policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, now + 301).is_err());

        // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Authored,
        // not run: use the transport's actual final gate with both authority
        // modes. The clock is injected; no DNS or HTTP operation is performed.
        assert!(peers.record_phala_peer_attestation(&relay, now + 301));
        let (_, selected_relay) = policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, now + 301).unwrap();
        let fresh = SourceSendAdmission {
            stopped: Arc::new(AtomicBool::new(false)), floor: Arc::new(ReverseOnionLocalObservation::new(now)), deadline: now + 1_200,
            poll_deadline: None, authority: SourceSendAuthority::Dispatch(Arc::clone(&policy)),
        };
        let evidence = SourceSendAdmission {
            stopped: Arc::new(AtomicBool::new(false)), floor: Arc::new(ReverseOnionLocalObservation::new(now)), deadline: now + 1_200,
            poll_deadline: None, authority: SourceSendAuthority::Evidence {
                policy: Arc::clone(&policy), relay_snapshot: selected_relay,
            },
        };
        assert!(fresh.check_at(Ok(now + 301)).is_ok());
        assert!(evidence.check_at(Ok(now + 301)).is_ok());
        // [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex]
        // A different carrier has already observed a later cache time. The
        // final POST/GET gate must reject an earlier request sample even when
        // that request's own clock floor has not advanced. No network I/O.
        assert!(peers.phala_peer_route_is_eligible(&relay, now + 302));
        assert_eq!(fresh.check_at(Ok(now + 301)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        assert_eq!(evidence.check_at(Ok(now + 301)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        assert!(fresh.check_at(Ok(now + 302)).is_ok());
        assert!(evidence.check_at(Ok(now + 302)).is_ok());
        // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] A zero
        // monotonic budget expires instantly; use the smallest valid config.
        peers.configure_phala_attested_peer_routes(true, 1);
        assert_eq!(fresh.check_at(Ok(now + 303)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        assert_eq!(evidence.check_at(Ok(now + 303)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        assert!(peers.record_phala_peer_attestation(&relay, now + 303));
        assert!(fresh.check_at(Ok(now + 303)).is_ok());
        assert!(evidence.check_at(Ok(now + 303)).is_ok());

        // Expiring a fresh grant must not strand historical evidence reads.
        assert!(peers.record_phala_peer_attestation(&relay, now + 601));
        assert_eq!(fresh.check_at(Ok(now + 601)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        assert!(evidence.check_at(Ok(now + 601)).is_ok());

        let mut renewed_body = relay.descriptor.clone();
        renewed_body.sequence += 1;
        renewed_body.issued_at = now + 602;
        let renewed = SignedNodeDescriptor::sign(renewed_body, &relay_key).unwrap();
        peers.upsert_verified_from_source(renewed.clone(), now + 602, "test_pin").unwrap();
        assert!(!peers.record_phala_peer_attestation(&relay, now + 602));
        assert!(peers.record_phala_peer_attestation(&renewed, now + 602));
        assert_eq!(evidence.check_at(Ok(now + 602)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Rejected));
        let (_, renewed_snapshot) = policy.recovery_relay_snapshot(SOURCE_QUERY_PATH, now + 602).unwrap();
        let renewed_evidence = SourceSendAdmission {
            stopped: Arc::new(AtomicBool::new(false)), floor: Arc::new(ReverseOnionLocalObservation::new(now + 602)), deadline: now + 1_200,
            poll_deadline: None, authority: SourceSendAuthority::Evidence {
                policy, relay_snapshot: renewed_snapshot,
            },
        };
        assert!(renewed_evidence.check_at(Ok(now + 602)).is_ok());
    }

    // [SOURCE-CURRENT-PINS 2026-10-05 by Codex] Authored, not executed.
    // A signed config blob is not a substitute for the current peer-store pin.
    #[cfg(unix)]
    #[test]
    fn configured_source_requires_the_exact_current_peer_pin() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, authorization) = fixture.policy_parts();
        let peers = Arc::new(PeerStore::new());
        let policy = SourcePinnedRelayPolicy {
            source: fixture.source_identity().public_key_bytes(),
            relay: relay.clone(),
            recipient: recipient.clone(),
            authorization,
            current_peers: Some(Arc::clone(&peers)),
            relay_endpoint_pin: None,
        };
        assert_eq!(policy.check_current_pin(&relay, fixture.now()), Err(SourceRuntimeError::Rejected));
        peers.upsert_verified(relay.clone(), fixture.now()).unwrap();
        assert!(policy.check_current_pin(&relay, fixture.now()).is_ok());
        assert_eq!(policy.check_current_pin(&recipient, fixture.now()), Err(SourceRuntimeError::Rejected));
        peers.upsert_verified(recipient.clone(), fixture.now()).unwrap();
        assert!(policy.check_current_pin(&recipient, fixture.now()).is_ok());
    }

    // [REVERSE-ONION-AUTHORITY-RENEWAL 2026-10-05 by Codex] Authored, not
    // executed. New work may use a fresh P signature only for the exact
    // current descriptor pair; a rotated P descriptor invalidates the old
    // token before the replacement token is accepted.
    #[cfg(unix)]
    #[test]
    fn fresh_authorization_tracks_current_descriptor_pair() {
        use aeronyx_core::protocol::discovery::SignedPrivateOnionRecipientAuthorizationV1;

        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, initial_authorization) = fixture.policy_parts();
        let source = fixture.source_identity().public_key_bytes();
        let recipient_identity = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
        let now = fixture.now() + 10;
        let peers = Arc::new(PeerStore::new());
        peers.upsert_verified(relay.clone(), now).unwrap();
        peers.upsert_verified(recipient.clone(), now).unwrap();
        let policy = SourcePinnedRelayPolicy::new(
            source,
            relay.clone(),
            recipient.clone(),
            initial_authorization.clone(),
            now,
        )
        .unwrap()
        .with_current_peers(Arc::clone(&peers));

        let fresh_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay,
            &recipient,
            OnionRoutePurpose::BlindVaultPull.as_str(),
            now + 1,
            now + 8_000,
            &recipient_identity,
        )
        .unwrap();
        peers.cache_verified_private_onion_authorization(
            fresh_authorization.clone(), now + 2,
        ).unwrap();
        // [REVERSE-ONION-SOURCE-AUTHORITY-SNAPSHOT 2026-10-05 by Codex]
        let (snapshot_relay, snapshot_recipient, snapshot_grant) = peers
            .current_private_onion_authority_snapshot(
                &relay.node_id(),
                &recipient.node_id(),
                now + 2,
            )
            .unwrap();
        assert_eq!(snapshot_relay, relay);
        assert_eq!(snapshot_recipient, recipient);
        assert_eq!(snapshot_grant, fresh_authorization);
        let fresh_policy = policy
            .with_current_authorization(fresh_authorization.clone(), now + 2)
            .unwrap();
        fresh_policy.validate_at(now + 2).unwrap();
        assert_eq!(
            policy.with_current_authorization(initial_authorization.clone(), now + 2).err(),
            Some(SourceRuntimeError::Rejected),
        );

        let superseding_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay,
            &recipient,
            OnionRoutePurpose::BlindVaultPull.as_str(),
            now + 2,
            now + 8_000,
            &recipient_identity,
        )
        .unwrap();
        peers.cache_verified_private_onion_authorization(
            superseding_authorization, now + 3,
        ).unwrap();
        assert_eq!(
            fresh_policy.validate_at(now + 3),
            Err(SourceRuntimeError::Rejected),
        );

        let mut rotated_descriptor = recipient.descriptor.clone();
        rotated_descriptor.sequence += 1;
        let rotated_recipient = SignedNodeDescriptor::sign(
            rotated_descriptor,
            &recipient_identity,
        )
        .unwrap();
        peers.upsert_verified(rotated_recipient.clone(), now + 4).unwrap();
        assert!(peers.current_private_onion_authority_snapshot(
            &relay.node_id(),
            &recipient.node_id(),
            now + 4,
        ).is_none());
        assert_eq!(
            policy.with_current_authorization(fresh_authorization, now + 4).err(),
            Some(SourceRuntimeError::Rejected),
        );

        let rotated_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay,
            &rotated_recipient,
            OnionRoutePurpose::BlindVaultPull.as_str(),
            now + 4,
            now + 8_000,
            &recipient_identity,
        )
        .unwrap();
        peers.cache_verified_private_onion_authorization(
            rotated_authorization.clone(), now + 5,
        ).unwrap();
        let (_, snapshot_recipient, snapshot_grant) = peers
            .current_private_onion_authority_snapshot(
                &relay.node_id(),
                &recipient.node_id(),
                now + 5,
            )
            .unwrap();
        assert_eq!(snapshot_recipient, rotated_recipient);
        assert_eq!(snapshot_grant, rotated_authorization);
        let rotated_policy = policy
            .with_current_authorization(rotated_authorization, now + 5)
            .unwrap();
        rotated_policy.validate_at(now + 5).unwrap();
    }

    #[cfg(unix)]
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    #[cfg(unix)]
    use std::future::Future;
    #[cfg(unix)]
    use std::task::{Context, Poll, Wake, Waker};

    #[test]
    fn default_timeout_is_bounded_without_constructing_network_state() {
        let config = SourceRuntimeConfig::new(
            1, DEFAULT_SOURCE_TIMEOUT, 120, Duration::from_secs(5), Duration::from_millis(250),
        ).unwrap();
        assert_eq!(config.max_in_flight, 1);
        assert!(SourceRuntimeConfig::new(
            0, DEFAULT_SOURCE_TIMEOUT, 120, Duration::from_secs(5), Duration::from_millis(250),
        ).is_err());
        assert!(SourceRuntimeConfig::new(
            1, MAX_SOURCE_TIMEOUT + Duration::from_secs(1), 120,
            Duration::from_secs(5), Duration::from_millis(250),
        ).is_err());
        assert!(SourceRuntimeConfig::new(
            1, DEFAULT_SOURCE_TIMEOUT, 0, Duration::from_secs(5), Duration::from_millis(250),
        ).is_err());
        assert!(SourceRuntimeConfig::new(
            1, DEFAULT_SOURCE_TIMEOUT, 120, MAX_SOURCE_RESULT_WAIT + Duration::from_secs(1),
            Duration::from_millis(250),
        ).is_err());
        assert!(SourceRuntimeConfig::new(
            1, DEFAULT_SOURCE_TIMEOUT, 120, Duration::from_secs(5), Duration::from_millis(99),
        ).is_err());
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
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Fixture effects
        // obey the same final admission contract as the production transport.
        async fn post(&self, _target: PinnedPeerHttpTarget, _body: Bytes, _authorization: Bytes,
            admission: SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            self.posts.fetch_add(1, AtomicOrdering::SeqCst);
            SourceTransportOutcome::Ambiguous
        }

        async fn query(&self, _target: PinnedPeerHttpTarget, _body: Bytes,
            admission: SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            self.queries.fetch_add(1, AtomicOrdering::SeqCst);
            SourceTransportOutcome::Ambiguous
        }
    }

    #[cfg(unix)]
    fn fixture_runtime(
        fixture: &crate::services::reverse_onion_source::tests::Fixture,
        transport: Arc<dyn SourceTransport>,
        journal: Arc<ReverseOnionSourceJournal>,
        recovery_only: bool,
    ) -> Arc<ReverseOnionSourceRuntime> {
        let (relay, recipient, authorization) = fixture.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy {
            source: fixture.source_identity().public_key_bytes(),
            relay,
            recipient,
            authorization,
            current_peers: None,
            relay_endpoint_pin: None,
        });
        let mut runtime = ReverseOnionSourceRuntime::new(
            journal,
            fixture.source_identity(),
            policy,
            transport,
            SourceRuntimeConfig::new(
                1, Duration::from_secs(1), 120,
                Duration::from_secs(5), Duration::from_millis(250),
            ).unwrap(),
        )
        .unwrap();
        runtime.recovery_only = recovery_only;
        Arc::new(runtime)
    }

    // [PHALA-SOURCE-HTTP-FAULTS 2026-10-08 by Codex] Real loopback sockets
    // exercise the production transport after HTTP entry. Test-only HTTP and
    // TimeOnly admission do not establish public DNS, TLS or route authority.
    #[tokio::test]
    async fn production_transport_bounds_post_entry_failures_without_redirects() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let client = crate::api::privacy_safe_peer_http_client_builder().build().unwrap();
        for query in [false, true] {
            let limit = if query { MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES }
                else { BLIND_RELAY_ACK_RESPONSE_MAX_BYTES };
            // Complete, declared overflow, chunked overflow, truncated,
            // stalled body, redirect and an exactly full bounded response.
            for scenario in 0..7 {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let redirect = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let redirect_address = redirect.local_addr().unwrap();
                let (release_stall, stalled) = tokio::sync::oneshot::channel::<()>();
                let server = tokio::spawn(async move {
                    let (mut socket, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
                        .await.unwrap().unwrap();
                    let mut request = Vec::new();
                    let header_end = loop {
                        let mut chunk = [0u8; 1024];
                        let count = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
                            .await.unwrap().unwrap();
                        assert_ne!(count, 0, "complete request headers required");
                        request.extend_from_slice(&chunk[..count]);
                        assert!(request.len() <= 4096);
                        if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                            break end + 4;
                        }
                    };
                    let raw_headers = std::str::from_utf8(&request[..header_end]).unwrap();
                    let headers = raw_headers.to_ascii_lowercase();
                    let path = if query { SOURCE_QUERY_PATH } else { BLIND_RELAY_PATH };
                    assert!(headers.starts_with(&format!("post {path} http/1.1\r\n")));
                    assert!(headers.contains("content-length: 4\r\n"));
                    if query {
                        assert!(headers.contains("content-type: application/octet-stream\r\n"));
                        assert!(!headers.contains("x-aeronyx-private-recipient-authorization:"));
                    } else {
                        assert!(headers.contains("content-type: application/json\r\n"));
                        let authorization = raw_headers.lines().filter_map(|line| line.split_once(':'))
                            .find(|(name, _)| name.eq_ignore_ascii_case("x-aeronyx-private-recipient-authorization"))
                            .map(|(_, value)| value.trim());
                        assert_eq!(authorization, Some(STANDARD.encode(b"grant").as_str()));
                    }
                    while request.len() < header_end + 4 {
                        let mut chunk = [0u8; 4];
                        let count = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
                            .await.unwrap().unwrap();
                        assert_ne!(count, 0, "complete request body required");
                        request.extend_from_slice(&chunk[..count]);
                    }
                    assert_eq!(&request[header_end..], b"task");
                    let response = match scenario {
                        0 => b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_vec(),
                        1 => format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", limit + 1).into_bytes(),
                        2 => {
                            let mut response = format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n", limit + 1).into_bytes();
                            response.extend(std::iter::repeat_n(b'x', limit + 1));
                            response.extend_from_slice(b"\r\n0\r\n\r\n");
                            response
                        },
                        3 | 4 => b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nx".to_vec(),
                        5 => format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{redirect_address}/unexpected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes(),
                        6 => {
                            let mut response = format!("HTTP/1.1 200 OK\r\nContent-Length: {limit}\r\nConnection: close\r\n\r\n").into_bytes();
                            response.extend(std::iter::repeat_n(b'x', limit));
                            response
                        },
                        _ => unreachable!(),
                    };
                    // Oversize/timeout rejection may close the peer early.
                    let _ = tokio::time::timeout(Duration::from_secs(5), socket.write_all(&response)).await;
                    // Keep the incomplete body open until the client has
                    // returned: peer disconnect must not impersonate timeout.
                    if scenario == 4 { stalled.await.expect("client must release the stalled peer"); }
                });
                let timeout = if scenario == 4 { Duration::from_millis(100) } else { Duration::from_secs(2) };
                let transport = ReqwestSourceTransport::new(timeout).unwrap();
                let target = PinnedPeerHttpTarget { client: client.clone(),
                    url: reqwest::Url::parse(&format!("http://{address}{}",
                        if query { SOURCE_QUERY_PATH } else { BLIND_RELAY_PATH })).unwrap() };
                let at = observed_now(0).unwrap();
                let admission = SourceSendAdmission { stopped: Arc::new(AtomicBool::new(false)),
                    floor: Arc::new(ReverseOnionLocalObservation::new(at)), deadline: at + 30,
                    poll_deadline: None, authority: SourceSendAuthority::TimeOnly };
                let outcome = tokio::time::timeout(Duration::from_secs(3), async {
                    if query {
                        transport.query(target, Bytes::from_static(b"task"), admission).await
                    } else {
                        transport.post(target, Bytes::from_static(b"task"), Bytes::from_static(b"grant"), admission).await
                    }
                }).await.expect("production transport must complete without peer disconnect");
                match (scenario, outcome) {
                    (0, SourceTransportOutcome::Response { status: 200, body }) => assert_eq!(body, b"ok"),
                    (5, SourceTransportOutcome::Response { status: 307, body }) => assert!(body.is_empty()),
                    (6, SourceTransportOutcome::Response { status: 200, body }) => assert_eq!(body, vec![b'x'; limit]),
                    (1..=4, SourceTransportOutcome::Ambiguous) => {},
                    _ => panic!("unexpected post-entry outcome for query={query}, scenario={scenario}"),
                }
                if scenario == 4 {
                    assert!(!server.is_finished(), "the peer must still own its incomplete response");
                    release_stall.send(()).unwrap();
                }
                tokio::time::timeout(Duration::from_secs(6), server).await.unwrap().unwrap();
                assert!(tokio::time::timeout(Duration::from_millis(30), redirect.accept()).await.is_err(),
                    "neither task nor source evidence may follow a redirect");
            }
        }
    }

    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
    // exercise the actual transport gate with known failing clock inputs.
    #[test]
    fn send_admission_faults_are_not_hidden_by_stop_or_poll_expiry() {
        let stopped = Arc::new(AtomicBool::new(false));
        let mut admission = SourceSendAdmission {
            stopped: stopped.clone(), floor: Arc::new(ReverseOnionLocalObservation::new(100)), deadline: 110, poll_deadline: None,
            authority: SourceSendAuthority::TimeOnly,
        };
        assert!(admission.check_at(Ok(101)).is_ok());
        assert_eq!(admission.check_at(Ok(110)).err().map(|proof| proof.error), Some(SourceRuntimeError::Expired));
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] These are
        // independent attempts; a rejected attempt still retains its 110 floor.
        assert_eq!(admission.floor.floor(), Ok(110));
        admission.floor = Arc::new(ReverseOnionLocalObservation::new(100));
        stopped.store(true, AtomicOrdering::SeqCst);
        assert_eq!(admission.check_at(Ok(101)).err().map(|proof| proof.error), Some(SourceRuntimeError::Stopped));
        assert_eq!(admission.check_at(Ok(99)).err().map(|proof| proof.error), Some(SourceRuntimeError::Unavailable));
        assert_eq!(admission.check_at(Err(SourceRuntimeError::Unavailable)).err().map(|proof| proof.error),
            Some(SourceRuntimeError::Unavailable));
        stopped.store(false, AtomicOrdering::SeqCst);
        admission.floor = Arc::new(ReverseOnionLocalObservation::new(100));
        admission.poll_deadline = Some(std::time::Instant::now());
        assert_eq!(admission.check_at(Ok(101)).err().map(|proof| proof.error), Some(SourceRuntimeError::Pending));
        assert_eq!(admission.check_at(Ok(99)).err().map(|proof| proof.error), Some(SourceRuntimeError::Unavailable));
    }

    // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Authored,
    // not run: cancellation drops a controlled carrier future but not the
    // successful final sample or failed observation shared with its owner.
    // No socket, OS clock change or remote timestamp is used.
    #[tokio::test]
    async fn cancelled_transport_future_cannot_erase_its_clock_observation() {
        for failed_sample in [false, true] {
            let now = observed_now(0).unwrap();
            let observation = Arc::new(ReverseOnionLocalObservation::new(now));
            let admission = SourceSendAdmission {
                stopped: Arc::new(AtomicBool::new(false)), floor: observation.clone(),
                deadline: now + 2_000, poll_deadline: None, authority: SourceSendAuthority::TimeOnly,
            };
            let mut attempt = Box::pin(async move {
                if failed_sample {
                    assert!(admission.check_at(Err(SourceRuntimeError::Unavailable)).is_err());
                } else {
                    assert!(admission.check_at(Ok(now + 1_000)).is_ok());
                }
                std::future::pending::<()>().await;
            });
            assert!(futures::poll!(attempt.as_mut()).is_pending());
            assert!(tokio::time::timeout(Duration::from_millis(1), attempt).await.is_err());
            if failed_sample { assert_eq!(observation.floor(), Err(())); }
            else { assert_eq!(observation.floor(), Ok(now + 1_000)); }
            assert_eq!(source_transport_completion_now(&observation), Err(SourceRuntimeError::Unavailable));
            assert_eq!(observation.floor(), Err(()));
        }
    }

    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
    // the production request builder must reject before execute when first
    // polled after stop/expiry. This test-only target pins a reserved hostname
    // to loopback so a broken gate cannot contact an external service. It does
    // not validate production route admission, DNS/TLS, or socket observations.
    #[tokio::test]
    async fn production_transport_checks_admission_inside_its_first_poll() {
        let transport = ReqwestSourceTransport::new(Duration::from_secs(1)).unwrap();
        for query in [false, true] {
            for stopped_before_poll in [false, true] {
                let at = observed_now(0).unwrap();
                let stopped = Arc::new(AtomicBool::new(false));
                let target = PinnedPeerHttpTarget {
                    client: crate::api::privacy_safe_peer_http_client_builder()
                        .resolve("send-fence.invalid", "127.0.0.1:9".parse().unwrap())
                        .build().unwrap(),
                    url: reqwest::Url::parse(
                        "https://send-fence.invalid:9/api/chat/peer/reverse-onion/source-query",
                    ).unwrap(),
                };
                let admission = SourceSendAdmission {
                    stopped: stopped.clone(), floor: Arc::new(ReverseOnionLocalObservation::new(at)),
                    deadline: if stopped_before_poll { at + 60 } else { at }, poll_deadline: None,
                    authority: SourceSendAuthority::TimeOnly,
                };
                let attempt = if query {
                    transport.query(target, Bytes::from_static(b"query"), admission)
                } else {
                    transport.post(target, Bytes::from_static(b"task"), Bytes::from_static(b"grant"), admission)
                };
                if stopped_before_poll { stopped.store(true, AtomicOrdering::SeqCst); }
                match attempt.await {
                    SourceTransportOutcome::NotSent(proof) => assert_eq!(proof.error,
                        if stopped_before_poll { SourceRuntimeError::Stopped } else { SourceRuntimeError::Expired }),
                    _ => panic!("closed pre-entry admission must not enter HTTP"),
                }
            }
        }
    }

    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
    // a final transport clock fault wins over concurrent normal stop and
    // cannot restore a fresh Armed barrier to Prepared.
    #[cfg(unix)]
    #[tokio::test]
    async fn final_send_clock_fault_preserves_armed_and_fails_source_owner() {
        struct ClockFault;
        #[async_trait]
        impl SourceTransport for ClockFault {
            async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
                admission: SourceSendAdmission) -> SourceTransportOutcome {
                admission.stopped.store(true, AtomicOrdering::SeqCst);
                match admission.check_at(Ok(0)) {
                    Err(proof) => SourceTransportOutcome::NotSent(proof),
                    Ok(()) => panic!("known backward clock must fail transport admission"),
                }
            }
            async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
                _: SourceSendAdmission) -> SourceTransportOutcome {
                panic!("failed clock owner must not start evidence HTTP");
            }
        }
        let f = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(f.open(f.now()));
        let runtime = fixture_runtime(&f, Arc::new(ClockFault), journal.clone(), false);
        let policy = runtime.policy.as_ref().unwrap().clone();
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime.clone()).unwrap();
        let (request, expected, session, terminal) = f.runtime_admission_parts();
        let outcome = lifecycle.dispatch(request, expected, session, f.now() + 600,
            terminal, f.now(), policy).await;
        assert_eq!(outcome.err(), Some(SourceRuntimeError::Unavailable));
        assert!(lifecycle.request_admission().is_stopped());
        assert!(lifecycle.has_failed());
        let at = observed_now(0).unwrap();
        assert_eq!(journal.lookup_phase(f.route(), at).unwrap(), Some(SourcePhase::Armed));
        lifecycle.shutdown_and_drain().await.unwrap();
        drop(lifecycle);
        drop(runtime);
        drop(journal);
        let reopened = f.open(at);
        assert_eq!(reopened.lookup_phase(f.route(), at).unwrap(), Some(SourcePhase::Armed));
    }

    // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
    // the real ambiguity writer waits for DB ownership instead of discarding
    // Busy. Stop closes new work but must not cancel this completion write.
    #[cfg(unix)]
    #[tokio::test]
    async fn ambiguity_completion_waits_for_shared_journal_lane_after_stop() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        journal.arm(fixture.route(), fixture.now() + 1).unwrap();
        let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)), queries: Arc::new(AtomicUsize::new(0)),
        }), Arc::clone(&journal), false);
        let permit = Arc::new(Arc::clone(&runtime.permits).acquire_owned().await.unwrap());
        let lane = journal.blocking_operation_lane();
        let held = Arc::clone(&lane).acquire_owned().await.unwrap();
        let mut marking = Box::pin(runtime.mark_ambiguous(fixture.route(), fixture.now(), permit));
        assert!(futures::poll!(marking.as_mut()).is_pending());
        runtime.request_stop();
        assert_eq!(runtime.permits.available_permits(), 0);
        drop(held);
        // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
        // Completion returns its post-wait DB observation, not fixture.now().
        let completion_floor = observed_now(0).unwrap();
        assert!(marking.await.unwrap() >= completion_floor);
        assert_eq!(journal.lookup_phase(fixture.route(), observed_now(0).unwrap()).unwrap(),
            Some(SourcePhase::DispatchAmbiguous));
        runtime.shutdown_and_drain().await;
        assert_eq!(runtime.permits.available_permits(), 1);
        assert_eq!(lane.available_permits(), 1);
        assert!(!lane.is_closed());
    }

    // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
    // a post-commit fence failure is not permission to continue/replay Armed.
    #[cfg(unix)]
    #[tokio::test]
    async fn failed_ambiguity_fence_stops_the_live_source_owner() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        journal.arm(fixture.route(), fixture.now() + 1).unwrap();
        let posts = Arc::new(AtomicUsize::new(0));
        let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
            posts: Arc::clone(&posts), queries: Arc::new(AtomicUsize::new(0)),
        }), Arc::clone(&journal), false);
        let permit = Arc::new(Arc::clone(&runtime.permits).acquire_owned().await.unwrap());
        journal.fail_next_commit_fence();
        assert_eq!(runtime.mark_ambiguous(fixture.route(), fixture.now(), permit).await,
            Err(SourceRuntimeError::Unavailable));
        assert!(runtime.stopped.load(Ordering::SeqCst));
        assert_eq!(runtime.resume(fixture.route(), fixture.now()).await.err(), Some(SourceRuntimeError::Stopped));
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
        runtime.shutdown_and_drain().await;
    }

    // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Authored,
    // not run: all admission variants export the latest SQL observation.
    // Exact duplicates preserve existing work and never enter HTTP here.
    #[cfg(unix)]
    #[tokio::test]
    async fn source_admission_carries_current_floor_for_fresh_recovery_and_verified_routes() {
        use crate::services::reverse_onion_source::tests::Fixture;
        for existing in [None, Some(SourcePhase::Armed), Some(SourcePhase::Verified)] {
            let f = Fixture::new_for_runtime();
            let journal = Arc::new(f.open(f.now()));
            match existing {
                Some(SourcePhase::Armed) => {
                    f.prepare(&journal, f.now());
                    journal.arm(f.route(), f.now() + 1).unwrap();
                }
                Some(SourcePhase::Verified) => {
                    f.ready(&journal);
                    journal.open_result(f.route(), f.now() + 3).unwrap();
                }
                _ => {}
            }
            let posts = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let runtime = fixture_runtime(&f, Arc::new(CountingTransport {
                posts: posts.clone(), queries: queries.clone(),
            }), journal.clone(), false);
            let policy = runtime.policy.as_ref().unwrap().clone();
            let target = resolve_pinned_peer_http_target(policy.relay_url(BLIND_RELAY_PATH).unwrap(),
                Duration::from_secs(1)).await.unwrap();
            let (request, expected, session, terminal) = f.runtime_admission_parts();
            let permit = Arc::new(runtime.permits.clone().acquire_owned().await.unwrap());
            let floor = observed_now(0).unwrap();
            let admission = runtime.prepare_and_arm(request, expected, session, f.now() + 600,
                terminal, f.now(), permit, policy, target).await.unwrap();
            let checked_at = match (existing, admission) {
                (None, SourceAdmission::Send(dispatch)) => {
                    assert_eq!(dispatch.route, f.route());
                    assert_eq!(dispatch.exact_bytes, f.outbound_bytes());
                    dispatch.observed_at
                }
                (Some(SourcePhase::Armed), SourceAdmission::Recover { route, observed_at })
                | (Some(SourcePhase::Verified), SourceAdmission::Complete { route, observed_at }) => {
                    assert_eq!(route, f.route());
                    observed_at
                }
                _ => panic!("admission must preserve the exact existing phase"),
            };
            assert!(checked_at >= floor);
            let (metadata, metadata_at) = runtime.metadata(f.route(), checked_at,
                Arc::new(runtime.permits.clone().acquire_owned().await.unwrap())).await.unwrap();
            assert!(metadata_at >= checked_at);
            assert_eq!(metadata.phase(), existing.unwrap_or(SourcePhase::Armed));
            assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(queries.load(AtomicOrdering::SeqCst), 0);
            runtime.shutdown_and_drain().await;
        }
    }

    // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
    // exercise the actual source preparation/arming boundary, not a separate
    // stop predicate. No route row or transport effect may escape this wait.
    #[cfg(unix)]
    #[tokio::test]
    async fn source_admission_observes_stop_after_journal_lane_wait() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        let posts = Arc::new(AtomicUsize::new(0));
        let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
            posts: Arc::clone(&posts), queries: Arc::new(AtomicUsize::new(0)),
        }), Arc::clone(&journal), false);
        let policy = Arc::clone(runtime.policy.as_ref().unwrap());
        // Fixture endpoint is the public IP literal 1.1.1.1. Resolving the
        // pinned target here neither sends HTTP nor performs a hostname lookup.
        let target = resolve_pinned_peer_http_target(policy.relay_url(BLIND_RELAY_PATH).unwrap(),
            Duration::from_secs(1)).await.unwrap();
        let (request, expected, session, terminal) = fixture.runtime_admission_parts();
        let permit = Arc::new(Arc::clone(&runtime.permits).acquire_owned().await.unwrap());
        let held = journal.blocking_operation_lane().acquire_owned().await.unwrap();
        let mut preparing = Box::pin(runtime.prepare_and_arm(
            request, expected, session, fixture.now() + 600, terminal, fixture.now(), permit, policy, target,
        ));
        assert!(futures::poll!(preparing.as_mut()).is_pending());
        runtime.request_stop();
        drop(held);
        assert_eq!(preparing.await.err(), Some(SourceRuntimeError::Stopped));
        assert_eq!(journal.lookup_phase(fixture.route(), observed_now(0).unwrap()).unwrap(), None);
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
        runtime.shutdown_and_drain().await;
    }

    // [REVERSE-ONION-SAME-RELAY-RECOVERY 2026-10-05 by Codex] Authored, not
    // executed: an armed route replays one exact POST to its same pinned relay
    // while the original signed authority remains current, then polls proof.
    #[cfg(unix)]
    #[tokio::test]
    async fn live_submit_recovers_armed_route_only_through_same_relay() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        journal.arm(fixture.route(), fixture.now() + 1).unwrap();
        let posts = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(AtomicUsize::new(0));
        let mut runtime = fixture_runtime(
            &fixture,
            Arc::new(CountingTransport {
                posts: Arc::clone(&posts),
                queries: Arc::clone(&queries),
            }),
            journal,
            true,
        );
        // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex] Select the
        // supported one-query mode; the normal five-second window retries.
        Arc::get_mut(&mut runtime).unwrap().result_wait = Duration::ZERO;

        let result = runtime
            .submit_pull_with_live_authority(
                fixture.route(),
                BlindVaultPullRequest {
                    version: aeronyx_core::protocol::blind_vault::BLIND_VAULT_PROTOCOL_VERSION,
                    lease_id: [7; 32],
                    read_capability: [8; 32],
                    continuation_cursor: Vec::new(),
                    limit: 1,
                },
                fixture.now() + 2,
            )
            .await;

        assert!(result.is_err());
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(queries.load(AtomicOrdering::SeqCst), 1);
    }

    // [PHALA-ROTATED-CUSTODY-RECOVERY 2026-10-08 by Codex] Authored,
    // not run: use the production identity-pinned owner after actual journal
    // reopen and signed R/P renewal. A new live grant cannot re-POST old work.
    #[cfg(unix)]
    #[tokio::test]
    async fn renewed_live_authority_keeps_restarted_custody_evidence_only() {
        for ambiguous in [false, true] {
            let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
            let now = fixture.now();
            let journal = fixture.open(now);
            fixture.prepare(&journal, now);
            journal.arm(fixture.route(), now + 1).unwrap();
            if ambiguous {
                journal.mark_dispatch_ambiguous(fixture.route(), now + 2).unwrap();
            }
            drop(journal);
            let journal = Arc::new(fixture.open_existing(now + 3));
            let (old_relay, old_recipient, old_grant) = fixture.policy_parts();
            let (relay, recipient, grant) = fixture.renewed_policy_parts(now + 3);
            assert!(old_grant.verify_at(&relay, &recipient,
                aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull.as_str(),
                now + 3).is_err());
            let peers = Arc::new(PeerStore::new());
            peers.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
            peers.upsert_verified_from_source(relay.clone(), now + 3, "test_pin").unwrap();
            peers.upsert_verified_from_source(recipient.clone(), now + 3, "test_pin").unwrap();
            peers.remember_issued_private_onion_authorization(grant.clone(), recipient.node_id(), now + 3).unwrap();
            let posts = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let runtime = ReverseOnionSourceRuntime::new_identity_pinned(
                Arc::clone(&journal), fixture.source_identity(), relay.node_id(), recipient.node_id(),
                relay.descriptor.public_endpoint.clone().unwrap(), Arc::clone(&peers),
                Arc::new(CountingTransport { posts: Arc::clone(&posts), queries: Arc::clone(&queries) }),
                SourceRuntimeConfig::new(1, Duration::from_secs(1), 120,
                    Duration::ZERO, Duration::from_millis(250)).unwrap(),
            ).unwrap();
            // Positive control: the replacement is usable for NEW admission;
            // this must not substitute it into the old sealed route record.
            runtime.policy_for_current_authority(grant, now + 3).unwrap()
                .validate_at(now + 3).unwrap();
            let recovery_at = observed_now(now + 3).unwrap();
            old_grant.verify_at(&old_relay, &old_recipient,
                aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull.as_str(),
                recovery_at).unwrap();
            assert_eq!(runtime.resume(fixture.route(), recovery_at).await.err(),
                Some(SourceRuntimeError::Pending));
            assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(queries.load(AtomicOrdering::SeqCst), 1);
            let restored = journal.recover_route_authority(
                fixture.route(), observed_now(now + 3).unwrap(),
            ).unwrap().signed_parts().unwrap();
            assert_eq!(restored.0.encode_canonical().unwrap(), old_relay.encode_canonical().unwrap());
            assert_eq!(restored.1.encode_canonical().unwrap(), old_recipient.encode_canonical().unwrap());
            assert_eq!(restored.2.encode_canonical().unwrap(), old_grant.encode_canonical().unwrap());
            runtime.shutdown_and_drain().await;
        }
    }

    // [REVERSE-SOURCE-AMBIGUOUS-STOP 2026-10-05 by Codex] Authored, not
    // executed: once transport ambiguity was observed, recovery is evidence
    // only and must not repeat the POST even to the same relay.
    #[cfg(unix)]
    #[tokio::test]
    async fn observed_ambiguous_dispatch_only_polls_evidence() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        journal.arm(fixture.route(), fixture.now() + 1).unwrap();
        journal.mark_dispatch_ambiguous(fixture.route(), fixture.now() + 2).unwrap();
        let posts = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(AtomicUsize::new(0));
        let mut runtime = fixture_runtime(
            &fixture,
            Arc::new(CountingTransport {
                posts: Arc::clone(&posts),
                queries: Arc::clone(&queries),
            }),
            journal,
            true,
        );
        // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex]
        Arc::get_mut(&mut runtime).unwrap().result_wait = Duration::ZERO;

        let result = runtime.resume(fixture.route(), fixture.now() + 3).await;

        assert!(result.is_err());
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(queries.load(AtomicOrdering::SeqCst), 1);
    }

    // [REVERSE-ONION-RESUME-SINGLE-FLIGHT 2026-10-05 by Codex] Authored,
    // not executed: a retry/recovery request cannot race another operation
    // already holding the same durable route through evidence collection/open.
    #[cfg(unix)]
    #[tokio::test]
    async fn resume_rejects_a_route_already_owned_by_submit_or_recovery() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        journal.arm(fixture.route(), fixture.now() + 1).unwrap();
        let transport = Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)),
            queries: Arc::new(AtomicUsize::new(0)),
        });
        let runtime = fixture_runtime(&fixture, transport, journal, false);
        let _owner = ActiveSourceRoute::reserve(
            Arc::clone(&runtime.active_routes),
            fixture.route(),
        )
        .unwrap();

        assert_eq!(
            runtime.resume(fixture.route(), fixture.now() + 2).await.err(),
            Some(SourceRuntimeError::Busy),
        );
    }

    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Authored, not run:
    // direct dispatch cannot bypass the same route reservation used by resume.
    #[cfg(unix)]
    #[tokio::test]
    async fn direct_dispatch_rejects_an_already_owned_route_before_transport() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        let posts = Arc::new(AtomicUsize::new(0));
        let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
            posts: posts.clone(), queries: Arc::new(AtomicUsize::new(0)),
        }), journal.clone(), false);
        let _owner = ActiveSourceRoute::reserve(runtime.active_routes.clone(), fixture.route()).unwrap();
        let (request, expected, session, terminal) = fixture.runtime_admission_parts();
        assert_eq!(runtime.dispatch(request, expected, session, fixture.now() + 600,
            terminal, fixture.now(), runtime.policy.as_ref().unwrap().clone()).await.err(),
            Some(SourceRuntimeError::Busy));
        assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(journal.lookup_phase(fixture.route(), observed_now(0).unwrap()).unwrap(), None);
    }

    // [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Authored, not run:
    // a disconnected HTTP waiter cannot hide a later owned-operation panic.
    #[tokio::test]
    async fn cancelled_waiter_does_not_hide_owned_panic_or_reopen_admission() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let caller_lifecycle = lifecycle.clone();
        let caller = tokio::spawn(async move {
            caller_lifecycle.run_owned::<(), _>(async move {
                let _ = entered_tx.send(());
                release_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?;
                panic!("synthetic owned unwind after caller cancellation");
            }).await
        });
        entered_rx.await.unwrap();
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !admission.is_stopped() { tokio::task::yield_now().await; }
        }).await.unwrap();
        assert_eq!(lifecycle.run_owned(async { Ok(()) }).await, Err(SourceRuntimeError::Stopped));
        assert_eq!(lifecycle.shutdown_and_drain().await, Err(SourceRuntimeError::Unavailable));
        assert_eq!(lifecycle.shutdown_and_drain().await, Err(SourceRuntimeError::Unavailable));
    }

    // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] Authored, not
    // run: exercise the real installed owner with a normal local error,
    // including after waiter cancellation. This is not an OS clock probe.
    #[cfg(unix)]
    #[tokio::test]
    async fn returned_local_fault_survives_cancellation_and_reaches_supervision() {
        use super::super::runtime_supervision;
        for cancel_waiter in [false, true] {
            let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
            let journal = Arc::new(fixture.open(fixture.now()));
            fixture.prepare(&journal, fixture.now());
            let posts = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
                posts: posts.clone(), queries: queries.clone(),
            }), journal.clone(), false);
            let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
            lifecycle.install(runtime.clone()).unwrap();
            assert_eq!(lifecycle.verify_ready_now(), Ok(()));
            let mut observer = Box::pin(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)));
            assert!(futures::poll!(observer.as_mut()).is_pending());
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let caller_owner = lifecycle.clone();
            // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex]
            // Reach the actual source observer, not a fabricated local error.
            let operation_runtime = runtime.clone();
            let route = fixture.route();
            let caller = tokio::spawn(async move {
                caller_owner.run_owned::<(), _>(async move {
                    let _ = entered_tx.send(());
                    release_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?;
                    operation_runtime.resume(route, u64::MAX).await.map(|_| ())
                }).await
            });
            entered_rx.await.unwrap();
            if cancel_waiter {
                caller.abort();
            }
            release_tx.send(()).unwrap();
            // The operation publishes the sticky fault before result send;
            // neither a live nor a disconnected HTTP waiter owns publication.
            tokio::time::timeout(Duration::from_secs(1), async {
                while !lifecycle.request_admission().is_stopped() { tokio::task::yield_now().await; }
                while !lifecycle.has_failed() { tokio::task::yield_now().await; }
            }).await.unwrap();
            assert_eq!(lifecycle.verify_ready_now(), Err(SourceRuntimeError::Unavailable));
            let expected = runtime_supervision::reverse_onion_source_failed();
            assert_eq!(tokio::time::timeout(Duration::from_secs(1), observer.as_mut()).await.unwrap(), expected);
            drop(observer);
            if cancel_waiter {
                assert!(caller.await.unwrap_err().is_cancelled());
            } else {
                assert_eq!(caller.await.unwrap(), Err(SourceRuntimeError::Unavailable));
            }
            assert!(lifecycle.has_failed());
            assert!(lifecycle.request_admission().is_stopped());
            assert_eq!(lifecycle.verify_ready_now(), Err(SourceRuntimeError::Unavailable));
            assert_eq!(runtime.resume(fixture.route(), observed_now(0).unwrap()).await.err(), Some(SourceRuntimeError::Stopped));
            assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(queries.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(journal.lookup_phase(fixture.route(), observed_now(0).unwrap()).unwrap(), Some(SourcePhase::Prepared));
            // Normal returned faults close intake but are not child join
            // failures. Drain still succeeds and cannot erase supervision.
            lifecycle.shutdown_and_drain().await.unwrap();
            lifecycle.shutdown_and_drain().await.unwrap();
            assert_eq!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)), Some(expected));
        }
    }

    // [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex] Authored,
    // not run: accepted work can fail after stop and caller cancellation.
    // A successful lifetime drain must retain that separately sticky fault.
    #[cfg(unix)]
    #[tokio::test]
    async fn source_completion_fault_after_stop_survives_cancelled_caller_and_successful_drain() {
        use super::super::runtime_supervision;
        // [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] Returned
        // startup errors and unwinds each cover healthy/faulty completion.
        for (server_unwinds, fail_completion) in [(false, false), (false, true), (true, false), (true, true)] {
            let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
            let journal = Arc::new(fixture.open(fixture.now()));
            fixture.prepare(&journal, fixture.now());
            let posts = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(AtomicUsize::new(0));
            let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
                posts: posts.clone(), queries: queries.clone(),
            }), journal.clone(), false);
            let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
            lifecycle.install(runtime.clone()).unwrap();
            let caller_owner = lifecycle.clone();
            let operation_runtime = runtime.clone();
            let route = fixture.route();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let caller = tokio::spawn(async move {
                caller_owner.run_owned(async move {
                    let _ = entered_tx.send(());
                    release_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?;
                    // Continue an already accepted local journal read, not
                    // a fresh dispatch through the closed intake gate.
                    let permit = Arc::new(operation_runtime.permits.clone().acquire_owned().await
                        .map_err(|_| SourceRuntimeError::Unavailable)?);
                    operation_runtime.metadata(route, observed_now(0)?, permit).await.map(|_| ())
                }).await
            });
            entered_rx.await.unwrap();
            // [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex]
            // Model the same outer dependency owner used by Server::run_owned.
            // Its generic task must remain alive until the real DB read ends.
            struct Dropped(Arc<AtomicBool>);
            impl Drop for Dropped {
                fn drop(&mut self) { self.0.store(true, AtomicOrdering::SeqCst); }
            }
            let dropped = Arc::new(AtomicBool::new(false));
            let marker = Dropped(dropped.clone());
            let (dependency_entered, dependency_started) = tokio::sync::oneshot::channel();
            let (dependency_release, dependency_exit) = tokio::sync::oneshot::channel();
            let dependency = tokio::spawn(async move {
                let _marker = marker;
                dependency_entered.send(()).unwrap();
                let _ = dependency_exit.await;
            });
            let dependency_abort = dependency.abort_handle();
            let mut dependencies = runtime_supervision::ReverseRuntimeDependencies::default();
            let server_result = runtime_supervision::catch_reverse_runtime_unwind(async {
                dependencies.tasks.push(("test-accepted-work-dependency", dependency));
                dependency_started.await.unwrap();
                if server_unwinds { panic!("synthetic server unwind payload"); }
                Err(crate::error::ServerError::startup_failed("synthetic startup error"))
            }).await;
            if server_unwinds {
                assert!(matches!(server_result, Err(crate::error::ServerError::RuntimeFailed { task, reason })
                    if task == "reverse-owned-server" && reason == "required reverse server owner unwound"));
            } else {
                assert!(matches!(server_result, Err(crate::error::ServerError::StartupFailed { reason })
                    if reason == "synthetic startup error"));
            }
            assert!(!dependency_abort.is_finished());
            assert!(!dropped.load(AtomicOrdering::SeqCst));
            let mut draining = Box::pin(dependencies.drain_reverse_owners(Some(&lifecycle), None));
            assert!(futures::poll!(draining.as_mut()).is_pending());
            assert!(lifecycle.request_admission().is_stopped());
            // Losing a drain waiter must not release the outer registry.
            drop(draining);
            let mut draining = Box::pin(dependencies.drain_reverse_owners(Some(&lifecycle), None));
            assert!(futures::poll!(draining.as_mut()).is_pending());
            assert!(!dependency_abort.is_finished());
            assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
                None, Some(&lifecycle), None,
            ), None);
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            if fail_completion { journal.fail_next_commit_fence(); }
            release_tx.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(1), draining).await.unwrap().unwrap();
            assert!(!dropped.load(AtomicOrdering::SeqCst));
            dependency_release.send(()).unwrap();
            let reports = super::super::Server::shutdown_runtime_tasks(dependencies.tasks.take_for_shutdown()).await;
            assert_eq!(reports[0].outcome, runtime_supervision::RuntimeTaskShutdownOutcome::Completed);
            assert!(dropped.load(AtomicOrdering::SeqCst));
            lifecycle.shutdown_and_drain().await.unwrap();
            assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
                None, Some(&lifecycle), None,
            ), fail_completion.then(runtime_supervision::reverse_onion_source_failed));
            let selected = runtime_supervision::CriticalRuntimeFailure {
                task: "synthetic-required-listener", reason: "synthetic listener failed".into(),
            };
            assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
                Some(selected.clone()), Some(&lifecycle), None,
            ), Some(selected));
            assert_eq!(posts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(queries.load(AtomicOrdering::SeqCst), 0);
        }
    }

    // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] Authored, not
    // run: client quota/protocol failures must not become an owner-kill switch.
    #[cfg(unix)]
    #[tokio::test]
    async fn returned_backpressure_and_protocol_errors_keep_source_owner_live() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        let runtime = fixture_runtime(&fixture, Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)), queries: Arc::new(AtomicUsize::new(0)),
        }), journal, false);
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime).unwrap();
        assert_eq!(SourceRuntimeError::from(SourceJournalError::Capacity), SourceRuntimeError::Busy);
        for error in [SourceRuntimeError::from(SourceJournalError::Capacity), SourceRuntimeError::Busy,
            SourceRuntimeError::Rejected, SourceRuntimeError::Expired, SourceRuntimeError::Conflict,
            SourceRuntimeError::Ambiguous, SourceRuntimeError::Pending]
        {
            assert_eq!(lifecycle.run_owned::<(), _>(async move { Err(error) }).await, Err(error));
            assert!(!lifecycle.has_failed());
            assert!(!lifecycle.request_admission().is_stopped());
            assert_eq!(lifecycle.verify_ready_now(), Ok(()));
        }
        lifecycle.shutdown_and_drain().await.unwrap();
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
        let runtime = fixture_runtime(&fixture, transport, journal, false);
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
    async fn resumed_opening_journal_finishes_local_open_without_post_or_query() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.ready_with_maximum(&journal);
        fixture.mark_opening(&journal);
        let expected_body = fixture.outbound_bytes();
        assert!(!expected_body.is_empty());
        let before = journal
            .recover_metadata(None, 1, fixture.now() + 3)
            .unwrap()
            .items
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(before.phase(), SourcePhase::Opening);
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
        assert_eq!(reopened_item.phase(), SourcePhase::Opening);
        assert_eq!(reopened_item.body_commitment(), before.body_commitment());
        let posts = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(CountingTransport {
            posts: Arc::clone(&posts),
            queries: Arc::clone(&queries),
        });
        let runtime = fixture_runtime(&fixture, transport, reopened, false);
        let result = runtime
            .resume(fixture.route(), fixture.now())
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
        let runtime = fixture_runtime(&fixture, transport, journal, false);
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

    // [REVERSE-SOURCE-API-ADMISSION 2026-10-05 by Codex] Authored, not run:
    // source shutdown must wait for requests admitted before the stop boundary.
    #[tokio::test]
    async fn lifecycle_drain_waits_for_pre_body_admission_permit() {
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        let admission = lifecycle.request_admission();
        let permit = admission.try_acquire().unwrap();
        lifecycle.request_stop();
        assert!(admission.try_acquire().is_none());

        let mut drain = Box::pin(lifecycle.shutdown_and_drain());
        assert!(matches!(futures::poll!(drain.as_mut()), std::task::Poll::Pending));
        drop(drain);
        drop(permit);

        lifecycle.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Authored, not run:
    // a cloned extension is one retained permit, bound to exactly one owner.
    #[tokio::test]
    async fn shared_http_permit_cannot_transfer_or_outlive_drain_accounting() {
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        let admission = lifecycle.request_admission();
        let other = ReverseOnionSourceLifecycle::with_request_limit(1);
        let permit = admission.try_acquire_request().unwrap();
        let retained = permit.clone();
        assert!(admission.recognizes_request(&permit));
        assert!(!other.request_admission().recognizes_request(&permit));
        assert!(admission.try_acquire_request().is_none());
        lifecycle.request_stop();
        assert!(!admission.recognizes_request(&permit));
        let mut drain = Box::pin(lifecycle.shutdown_and_drain());
        assert!(futures::poll!(drain.as_mut()).is_pending());
        drop(permit);
        assert!(futures::poll!(drain.as_mut()).is_pending());
        drop(drain);
        drop(retained);
        lifecycle.shutdown_and_drain().await.unwrap();
        other.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Authored, not
    // run: an unpolled or partially delivered body owns the same HTTP slot.
    // Expiry is driven by polling, new admission or a cancellable drain waiter.
    #[tokio::test(start_paused = true)]
    async fn response_buffers_share_capacity_deadline_and_cancellation_safe_drain() {
        use futures::StreamExt;
        struct TrackedBody {
            bytes: Option<Bytes>,
            dropped: Arc<AtomicUsize>,
        }
        impl Stream for TrackedBody {
            type Item = Result<Bytes, std::io::Error>;
            fn poll_next(mut self: Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Option<Self::Item>> {
                Poll::Ready(self.bytes.take().map(Ok))
            }
        }
        impl Drop for TrackedBody {
            fn drop(&mut self) { self.dropped.fetch_add(1, AtomicOrdering::SeqCst); }
        }
        for mode in 0..5 {
            let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
            // [PHALA-SOURCE-FIXTURE-REPAIR 2026-10-08 by Codex] Missing
            // runtime is already a fault; body expiry must not change it.
            let failure_before_expiry = lifecycle.has_failed();
            let admission = lifecycle.request_admission();
            let dropped = Arc::new(AtomicUsize::new(0));
            let payload = vec![93; SOURCE_RESPONSE_CHUNK_BYTES * 3];
            let allocation = payload.as_ptr() as usize;
            let allocation_end = allocation + payload.len();
            let body = Body::from_stream(TrackedBody {
                bytes: Some(Bytes::from(payload)), dropped: dropped.clone(),
            });
            let body = admission.bound_response_body(body, admission.try_acquire_request().unwrap()).unwrap();
            let mut stream = body.into_data_stream();
            assert!(admission.try_acquire().is_none());
            if mode == 1 {
                let chunk = stream.next().await.unwrap().unwrap();
                assert_eq!(chunk.len(), SOURCE_RESPONSE_CHUNK_BYTES);
                let pointer = chunk.as_ptr() as usize;
                assert!(pointer < allocation || pointer >= allocation_end,
                    "the socket chunk must not keep the original allocation alive");
                assert!(admission.try_acquire().is_none());
            }
            if mode == 3 {
                drop(stream);
                assert_eq!(dropped.load(AtomicOrdering::SeqCst), 1);
                assert!(admission.try_acquire().is_some());
                lifecycle.shutdown_and_drain().await.unwrap();
                continue;
            }
            if mode == 4 {
                tokio::time::advance(SOURCE_RESPONSE_TIMEOUT).await;
                assert!(stream.next().await.unwrap().is_err(), "the first late poll cannot renew the handoff deadline");
                assert!(stream.next().await.is_none());
                assert_eq!(dropped.load(AtomicOrdering::SeqCst), 1);
                assert!(admission.try_acquire().is_some());
                lifecycle.shutdown_and_drain().await.unwrap();
                continue;
            }
            if mode == 2 {
                let mut first = Box::pin(lifecycle.shutdown_and_drain());
                assert!(futures::poll!(first.as_mut()).is_pending());
                drop(first);
                let mut retry = Box::pin(lifecycle.shutdown_and_drain());
                assert!(futures::poll!(retry.as_mut()).is_pending());
                tokio::time::advance(SOURCE_RESPONSE_TIMEOUT).await;
                retry.await.unwrap();
            } else {
                tokio::time::advance(SOURCE_RESPONSE_TIMEOUT).await;
                assert!(admission.try_acquire().is_some(), "new intake expires idle large buffers");
            }
            assert_eq!(dropped.load(AtomicOrdering::SeqCst), 1);
            assert!(stream.next().await.unwrap().is_err(), "expiry is not a successful empty body");
            assert!(stream.next().await.is_none());
            assert_eq!(lifecycle.has_failed(), failure_before_expiry,
                "delivery timeout must not change existing owner failure state");
            lifecycle.shutdown_and_drain().await.unwrap();
        }
    }

    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Authored, not
    // run: reject foreign local proofs and oversized response streams, while
    // normal EOS returns its exact bytes and releases the slot without a timer.
    #[tokio::test(start_paused = true)]
    async fn response_body_requires_own_permit_and_enforces_wire_limit() {
        use futures::StreamExt;
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let foreign = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        assert!(matches!(admission.bound_response_body(Body::from("x"),
            foreign.request_admission().try_acquire_request().unwrap()), Err(SourceRuntimeError::Rejected)));
        for oversized in [false, true] {
            let payload = if oversized { vec![0;
                aeronyx_core::protocol::onion::reverse_delivery::MAX_REVERSE_ONION_SOURCE_PULL_RESPONSE_BYTES + 1]
            } else { vec![87; SOURCE_RESPONSE_CHUNK_BYTES * 2 + 1] };
            let body = admission.bound_response_body(Body::from(payload.clone()),
                admission.try_acquire_request().unwrap()).unwrap();
            if oversized {
                assert!(body.into_data_stream().next().await.unwrap().is_err());
            } else {
                let mut stream = body.into_data_stream();
                let mut received = Vec::new();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    assert!(chunk.len() <= SOURCE_RESPONSE_CHUNK_BYTES);
                    received.extend_from_slice(&chunk);
                }
                assert_eq!(received, payload);
                tokio::time::advance(SOURCE_RESPONSE_TIMEOUT).await;
                assert!(stream.next().await.is_none(), "normal EOS remains EOS after the old deadline");
            }
            assert!(admission.try_acquire().is_some());
        }
        lifecycle.shutdown_and_drain().await.unwrap();
        foreign.shutdown_and_drain().await.unwrap();
    }

    // [REVERSE-ONION-SOURCE-OWNED-OPERATIONS 2026-10-05 by Codex] Authored,
    // not run: aborting the HTTP-side waiter must not abort accepted source
    // work, and shutdown must retain its owner until that work completes.
    #[tokio::test]
    async fn lifecycle_owns_source_operation_after_waiter_cancellation() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let task_lifecycle = Arc::clone(&lifecycle);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move {
            task_lifecycle
                .run_owned(async move {
                    let _ = started_tx.send(());
                    release_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?;
                    Ok(())
                })
                .await
        });

        started_rx.await.unwrap();
        waiter.abort();
        let _ = waiter.await;

        let drain_lifecycle = Arc::clone(&lifecycle);
        let drain = tokio::spawn(async move { drain_lifecycle.shutdown_and_drain().await });
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());
        release_tx.send(()).unwrap();
        drain.await.unwrap().unwrap();
        assert!(lifecycle.owned_operations.lock().unwrap().tasks.is_empty());
    }

    // [REVERSE-ONION-SOURCE-DRAIN-OWNERSHIP 2026-10-05 by Codex] Authored,
    // not run: cancelling a shutdown waiter must leave the same JoinHandle
    // available to a later drain, rather than detaching accepted work.
    #[tokio::test]
    async fn cancelled_source_drain_retains_owned_operation_handles() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let task_lifecycle = Arc::clone(&lifecycle);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move {
            task_lifecycle
                .run_owned(async move {
                    let _ = started_tx.send(());
                    release_rx.await.map_err(|_| SourceRuntimeError::Unavailable)?;
                    Ok(())
                })
                .await
        });
        started_rx.await.unwrap();
        waiter.abort();
        let _ = waiter.await;

        let mut first_drain = Box::pin(lifecycle.shutdown_and_drain());
        assert!(matches!(
            futures::poll!(first_drain.as_mut()),
            std::task::Poll::Pending
        ));
        assert_eq!(lifecycle.owned_operations.lock().unwrap().tasks.len(), 1);
        drop(first_drain);

        let second_lifecycle = Arc::clone(&lifecycle);
        let second_drain = tokio::spawn(async move { second_lifecycle.shutdown_and_drain().await });
        tokio::task::yield_now().await;
        release_tx.send(()).unwrap();
        second_drain.await.unwrap().unwrap();
        assert!(lifecycle.owned_operations.lock().unwrap().tasks.is_empty());
    }

    // [REVERSE-ONION-SOURCE-DRAIN-FAILURE 2026-10-05 by Codex] Authored,
    // not run: join failure must still drain and release the runtime owner.
    #[cfg(unix)]
    #[tokio::test]
    async fn source_drain_finishes_runtime_after_owned_task_join_error() {
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let journal = Arc::new(fixture.open(1_800_000_000));
        let transport = Arc::new(CountingTransport {
            posts: Arc::new(AtomicUsize::new(0)),
            queries: Arc::new(AtomicUsize::new(0)),
        });
        let runtime = fixture_runtime(&fixture, transport, journal, false);
        let runtime_permit = Arc::clone(&runtime.permits).acquire_owned().await.unwrap();
        let lifecycle = ReverseOnionSourceLifecycle::new();
        lifecycle.install(Arc::clone(&runtime)).unwrap();

        let operation = tokio::spawn(std::future::pending::<()>());
        operation.abort();
        tokio::task::yield_now().await;
        lifecycle.owned_operations.lock().unwrap().tasks.push(operation);

        let mut drain = Box::pin(lifecycle.shutdown_and_drain());
        assert!(matches!(
            futures::poll!(drain.as_mut()),
            std::task::Poll::Pending
        ));
        assert!(lifecycle.runtime.lock().unwrap().is_some());
        drop(drain);
        drop(runtime_permit);

        assert_eq!(
            lifecycle.shutdown_and_drain().await,
            Err(SourceRuntimeError::Unavailable)
        );
        assert!(lifecycle.runtime.lock().unwrap().is_none());
    }

}
