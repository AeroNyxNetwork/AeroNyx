//! Source-local exact-replay transport boundary for Blind Vault replication.
//!
//! # File Creation Notes
//! - Creation reason: keep one committed replica effect bound to the exact
//!   opaque peer blind-relay request that may be replayed after a restart.
//! - Main functionality: immutable request artifacts, core send-context
//!   matching, freshness enforcement, and replaceable asynchronous I/O.
//! - Dependencies: consumes existing Blind Vault replica workflow context,
//!   onion-reply framing, and peer blind-relay JSON without changing them.
//!
//! # Main Logical Flow
//! 1. Build an effect binding from one durable workflow work item.
//! 2. Validate and freeze the exact reply-capable payload and outer request.
//! 3. Rebuild only the expected source-local context at send time.
//! 4. Reject every mismatch before handing exact bytes to asynchronous I/O.
//!
//! # Important Note For The Next Developer
//! - This module never builds an onion envelope or chooses an endpoint.
//! - Persist the complete artifact before publishing a committed send marker.
//! - A committed retry must reuse this artifact; never reconstruct JSON.
//! - The encrypted payload is opaque and must never enter diagnostics.
//!
//! Last Modified: v1.0.0-ExactReplicaReplay - Added immutable committed
//! artifacts and fail-closed asynchronous replay composition.

use crate::api::chat_peer::PeerBlindRelayRequest;
use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::protocol::blind_vault_replica_workflow::{
    BlindVaultReplicaTerminalSendContext, BlindVaultReplicaWorkId,
};
use aeronyx_core::protocol::onion::OnionRoutePurpose;
use aeronyx_core::protocol::{
    decode_onion_reply_request, encode_onion_reply_request, MAX_ONION_SEALED_RESPONSE_BYTES,
};
use sha2::{Digest, Sha256};
use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use zeroize::Zeroize;

const MAX_EXACT_OUTER_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const PAYLOAD_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaTransportPayload-v1";
const BODY_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaTransportOuterBody-v1";
const ARTIFACT_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaTransportArtifact-v1";

/// Source-local identity of one committed workflow attempt.
///
/// This type intentionally omits `Debug` and `Display`: its job and workflow
/// identifiers are private correlation material, not telemetry.
pub(crate) struct BlindVaultReplicaAttemptBindingV1 {
    job_id: [u8; 16],
    workflow_id: [u8; 16],
    work_sequence: u16,
    attempt: u8,
}

impl BlindVaultReplicaAttemptBindingV1 {
    /// Binds a durable job to the exact core work item and attempt.
    pub(crate) fn from_committed_work(
        job_id: [u8; 16],
        work_id: BlindVaultReplicaWorkId,
        attempt: u8,
    ) -> Result<Self, BlindVaultReplicaTransportBuildError> {
        let workflow_id = work_id.workflow_id();
        if job_id == [0; 16] || workflow_id == [0; 16] || job_id != workflow_id || attempt == 0 {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        Ok(Self {
            job_id,
            workflow_id,
            work_sequence: work_id.sequence(),
            attempt,
        })
    }
}

/// Exact ordered effect identity inside one committed attempt.
///
/// [BLIND-VAULT-REPLICA-TRANSPORT 2026-09-01 by Codex] The outer request
/// artifact consumes this complete binding. No later API accepts a parallel
/// caller-selected target, route, request id, purpose, or ordinal.
pub(crate) struct BlindVaultReplicaEffectBindingV1 {
    attempt: BlindVaultReplicaAttemptBindingV1,
    effect_ordinal: u16,
    effect_count: u16,
    purpose: OnionRoutePurpose,
    source_node_id: [u8; 32],
    target_node_id: [u8; 32],
    entry_node_id: [u8; 32],
    route_id: [u8; 16],
    request_id: [u8; 16],
}

impl BlindVaultReplicaEffectBindingV1 {
    /// Extends one committed attempt with its exact effect and routing context.
    pub(crate) fn new(
        attempt: BlindVaultReplicaAttemptBindingV1,
        effect_ordinal: u16,
        effect_count: u16,
        purpose: OnionRoutePurpose,
        source_node_id: [u8; 32],
        target_node_id: [u8; 32],
        entry_node_id: [u8; 32],
        route_id: [u8; 16],
        request_id: [u8; 16],
    ) -> Result<Self, BlindVaultReplicaTransportBuildError> {
        if !is_m12_effect_context(effect_ordinal, effect_count, purpose)
            || IdentityPublicKey::from_bytes(&source_node_id).is_err()
            || IdentityPublicKey::from_bytes(&target_node_id).is_err()
            || IdentityPublicKey::from_bytes(&entry_node_id).is_err()
            || source_node_id == target_node_id
            || source_node_id == entry_node_id
            || route_id == [0; 16]
            || request_id == [0; 16]
        {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        Ok(Self {
            attempt,
            effect_ordinal,
            effect_count,
            purpose,
            source_node_id,
            target_node_id,
            entry_node_id,
            route_id,
            request_id,
        })
    }

    fn matches_expectation(&self, expected: &BlindVaultReplicaReplayExpectationV1) -> bool {
        self.attempt.job_id == expected.job_id
            && self.attempt.workflow_id == expected.workflow_id
            && self.attempt.work_sequence == expected.work_sequence
            && self.attempt.attempt == expected.attempt
            && self.effect_ordinal == expected.effect_ordinal
            && self.effect_count == expected.effect_count
            && self.purpose == expected.purpose
            && self.source_node_id == expected.source_node_id
            && self.target_node_id == expected.target_node_id
            && self.entry_node_id == expected.entry_node_id
            && self.route_id == expected.route_id
            && self.request_id == expected.request_id
    }
}

/// Immutable source-private artifact for one exact committed outbound effect.
///
/// Fields are private and the type is not `Clone`. The only outbound view
/// borrows the frozen body after every binding and commitment check succeeds.
pub(crate) struct BlindVaultReplicaExactRequestArtifactV1 {
    binding: BlindVaultReplicaEffectBindingV1,
    exact_onion_reply_request_bytes: Box<[u8]>,
    exact_outer_request_bytes: Box<[u8]>,
    payload_commitment: [u8; 32],
    body_commitment: [u8; 32],
    expires_at_ms: u64,
    artifact_commitment: [u8; 32],
}

impl BlindVaultReplicaExactRequestArtifactV1 {
    /// Validates and freezes one exact peer blind-relay carrier.
    ///
    /// The onion reply wrapper is decoded only to enforce its bounded framing;
    /// its Blind Vault payload remains opaque and is never interpreted here.
    pub(crate) fn new(
        binding: BlindVaultReplicaEffectBindingV1,
        encoded_onion_reply_request: &[u8],
        exact_outer_request_bytes: Vec<u8>,
        expires_at_ms: u64,
    ) -> Result<Self, BlindVaultReplicaTransportBuildError> {
        if expires_at_ms == 0
            || encoded_onion_reply_request.is_empty()
            || exact_outer_request_bytes.is_empty()
            || exact_outer_request_bytes.len() > MAX_EXACT_OUTER_REQUEST_BYTES
        {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        let onion_reply_request = decode_onion_reply_request(encoded_onion_reply_request)
            .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;
        let canonical_onion_reply_request = encode_onion_reply_request(&onion_reply_request)
            .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;
        if canonical_onion_reply_request != encoded_onion_reply_request {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }

        // [BLIND-VAULT-REPLICA-TRANSPORT 2026-09-01 by Codex] Parsing stops
        // at the existing peer control envelope. It validates route binding
        // and previous-hop authentication without decrypting encrypted_blob.
        let peer_request: PeerBlindRelayRequest =
            serde_json::from_slice(&exact_outer_request_bytes)
                .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;
        let canonical_outer_request = serde_json::to_vec(&peer_request)
            .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;
        if canonical_outer_request != exact_outer_request_bytes {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        if peer_request.envelope.route_id != binding.route_id
            || peer_request.previous_hop_node_id != binding.source_node_id
            || peer_request.envelope.next_hop != binding.entry_node_id
        {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        let previous_hop = IdentityPublicKey::from_bytes(&peer_request.previous_hop_node_id)
            .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;
        peer_request
            .envelope
            .verify_signature_from(&previous_hop)
            .map_err(|_| BlindVaultReplicaTransportBuildError::Rejected)?;

        let payload_commitment = commitment(PAYLOAD_COMMITMENT_DOMAIN, encoded_onion_reply_request);
        let body_commitment = commitment(BODY_COMMITMENT_DOMAIN, &exact_outer_request_bytes);
        let artifact_commitment = artifact_commitment(
            &binding,
            &payload_commitment,
            &body_commitment,
            expires_at_ms,
        );
        Ok(Self {
            binding,
            exact_onion_reply_request_bytes: canonical_onion_reply_request.into_boxed_slice(),
            exact_outer_request_bytes: exact_outer_request_bytes.into_boxed_slice(),
            payload_commitment,
            body_commitment,
            expires_at_ms,
            artifact_commitment,
        })
    }

    /// Sends only the exact frozen body after complete committed-context checks.
    pub(crate) async fn replay_exact<Io>(
        &self,
        expected: &BlindVaultReplicaReplayExpectationV1,
        now_ms: u64,
        io: &Io,
    ) -> Result<Vec<u8>, BlindVaultReplicaTransportError<Io::Error>>
    where
        Io: BlindVaultReplicaOutboundIo,
    {
        if now_ms == 0
            || now_ms >= self.expires_at_ms
            || !self.binding.matches_expectation(expected)
        {
            return Err(if now_ms >= self.expires_at_ms {
                BlindVaultReplicaTransportError::Expired
            } else {
                BlindVaultReplicaTransportError::Rejected
            });
        }

        let current_payload_commitment = commitment(
            PAYLOAD_COMMITMENT_DOMAIN,
            &self.exact_onion_reply_request_bytes,
        );
        let current_body_commitment =
            commitment(BODY_COMMITMENT_DOMAIN, &self.exact_outer_request_bytes);
        let current_artifact_commitment = artifact_commitment(
            &self.binding,
            &self.payload_commitment,
            &self.body_commitment,
            self.expires_at_ms,
        );
        if !commitment_eq(&self.payload_commitment, &current_payload_commitment)
            || !commitment_eq(&self.body_commitment, &current_body_commitment)
            || !commitment_eq(&self.artifact_commitment, &current_artifact_commitment)
        {
            return Err(BlindVaultReplicaTransportError::Rejected);
        }

        let response = io
            .send_exact(BlindVaultReplicaExactOutboundRequest {
                entry_node_id: &self.binding.entry_node_id,
                target_node_id: &self.binding.target_node_id,
                route_id: &self.binding.route_id,
                request_id: &self.binding.request_id,
                exact_outer_request_bytes: &self.exact_outer_request_bytes,
            })
            .await
            .map_err(BlindVaultReplicaTransportError::Unavailable)?;
        if response.is_empty() || response.len() > MAX_ONION_SEALED_RESPONSE_BYTES {
            return Err(BlindVaultReplicaTransportError::Rejected);
        }
        Ok(response)
    }
}

impl Drop for BlindVaultReplicaExactRequestArtifactV1 {
    fn drop(&mut self) {
        // [BLIND-VAULT-REPLICA-TRANSPORT 2026-09-01 by Codex] These buffers
        // contain opaque encrypted workflow material. They are never plaintext,
        // but retaining stale copies still weakens the node-blind lifecycle.
        self.exact_onion_reply_request_bytes.zeroize();
        self.exact_outer_request_bytes.zeroize();
        self.payload_commitment.zeroize();
        self.body_commitment.zeroize();
        self.artifact_commitment.zeroize();
        self.binding.attempt.job_id.zeroize();
        self.binding.attempt.workflow_id.zeroize();
        self.binding.source_node_id.zeroize();
        self.binding.target_node_id.zeroize();
        self.binding.entry_node_id.zeroize();
        self.binding.route_id.zeroize();
        self.binding.request_id.zeroize();
        self.expires_at_ms.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaExactRequestArtifactV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaExactRequestArtifactV1")
            .field("state", &"committed")
            .finish_non_exhaustive()
    }
}

impl fmt::Display for BlindVaultReplicaExactRequestArtifactV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("blind vault replica exact request artifact")
    }
}

/// Expected workflow context reconstructed at the trusted send boundary.
///
/// [BLIND-VAULT-REPLICA-TRANSPORT 2026-09-01 by Codex] Construction consumes
/// the core-selected effect index. A caller cannot substitute another ordinal
/// after the durable sequence has selected the next effect.
pub(crate) struct BlindVaultReplicaReplayExpectationV1 {
    job_id: [u8; 16],
    workflow_id: [u8; 16],
    work_sequence: u16,
    attempt: u8,
    effect_ordinal: u16,
    effect_count: u16,
    purpose: OnionRoutePurpose,
    source_node_id: [u8; 32],
    target_node_id: [u8; 32],
    entry_node_id: [u8; 32],
    route_id: [u8; 16],
    request_id: [u8; 16],
}

impl BlindVaultReplicaReplayExpectationV1 {
    /// Reconstructs exact expected context from the core ordered-send token.
    pub(crate) fn from_terminal_send_context(
        job_id: [u8; 16],
        context: BlindVaultReplicaTerminalSendContext,
        purpose: OnionRoutePurpose,
        source_node_id: [u8; 32],
        target_node_id: [u8; 32],
        entry_node_id: [u8; 32],
        route_id: [u8; 16],
        request_id: [u8; 16],
    ) -> Result<Self, BlindVaultReplicaTransportBuildError> {
        let work_id = context.work_id();
        let expectation = Self {
            job_id,
            workflow_id: work_id.workflow_id(),
            work_sequence: work_id.sequence(),
            attempt: context.attempt(),
            effect_ordinal: context.effect_index(),
            effect_count: context.effect_count(),
            purpose,
            source_node_id,
            target_node_id,
            entry_node_id,
            route_id,
            request_id,
        };
        if expectation.job_id == [0; 16]
            || expectation.job_id != expectation.workflow_id
            || expectation.attempt == 0
            || !is_m12_effect_context(
                expectation.effect_ordinal,
                expectation.effect_count,
                expectation.purpose,
            )
            || IdentityPublicKey::from_bytes(&expectation.source_node_id).is_err()
            || IdentityPublicKey::from_bytes(&expectation.target_node_id).is_err()
            || IdentityPublicKey::from_bytes(&expectation.entry_node_id).is_err()
            || expectation.source_node_id == expectation.target_node_id
            || expectation.source_node_id == expectation.entry_node_id
            || expectation.route_id == [0; 16]
            || expectation.request_id == [0; 16]
            || context
                .authorized_terminal_node_id()
                .is_some_and(|node_id| node_id != expectation.target_node_id)
        {
            return Err(BlindVaultReplicaTransportBuildError::Rejected);
        }
        Ok(expectation)
    }
}

/// Borrowed exact request supplied to one replaceable asynchronous I/O adapter.
///
/// No endpoint is stored here. The adapter resolves only the already-bound
/// entry node; it must not replace that hop with the terminal target.
pub(crate) struct BlindVaultReplicaExactOutboundRequest<'a> {
    entry_node_id: &'a [u8; 32],
    target_node_id: &'a [u8; 32],
    route_id: &'a [u8; 16],
    request_id: &'a [u8; 16],
    exact_outer_request_bytes: &'a [u8],
}

impl BlindVaultReplicaExactOutboundRequest<'_> {
    #[must_use]
    pub(crate) const fn entry_node_id(&self) -> &[u8; 32] {
        self.entry_node_id
    }

    #[must_use]
    pub(crate) const fn target_node_id(&self) -> &[u8; 32] {
        self.target_node_id
    }

    #[must_use]
    pub(crate) const fn route_id(&self) -> &[u8; 16] {
        self.route_id
    }

    #[must_use]
    pub(crate) const fn request_id(&self) -> &[u8; 16] {
        self.request_id
    }

    #[must_use]
    pub(crate) const fn exact_outer_request_bytes(&self) -> &[u8] {
        self.exact_outer_request_bytes
    }
}

impl fmt::Debug for BlindVaultReplicaExactOutboundRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaExactOutboundRequest")
            .field("state", &"exact")
            .finish_non_exhaustive()
    }
}

/// Replaceable non-blocking I/O boundary for one already-prepared request.
///
/// Implementations must return only bounded, decoded opaque onion response
/// bytes. HTTP URL selection, JSON response parsing, and receipt verification
/// remain outside this contract until their production composition is wired.
pub(crate) trait BlindVaultReplicaOutboundIo: Send + Sync {
    type Error: Send + Sync + 'static;

    fn send_exact<'a>(
        &'a self,
        request: BlindVaultReplicaExactOutboundRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, Self::Error>> + Send + 'a>>;
}

/// Coarse construction failure with no private context.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindVaultReplicaTransportBuildError {
    Rejected,
}

impl fmt::Debug for BlindVaultReplicaTransportBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Rejected")
    }
}

impl fmt::Display for BlindVaultReplicaTransportBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("blind vault replica transport input rejected")
    }
}

impl StdError for BlindVaultReplicaTransportBuildError {}

/// Coarse send result; adapter errors never enter standard diagnostics.
pub(crate) enum BlindVaultReplicaTransportError<IoError> {
    Rejected,
    Expired,
    Unavailable(IoError),
}

impl<IoError> fmt::Debug for BlindVaultReplicaTransportError<IoError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected => formatter.write_str("Rejected"),
            Self::Expired => formatter.write_str("Expired"),
            Self::Unavailable(_) => formatter.write_str("Unavailable(<redacted>)"),
        }
    }
}

impl<IoError> fmt::Display for BlindVaultReplicaTransportError<IoError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected => formatter.write_str("blind vault replica transport rejected"),
            Self::Expired => formatter.write_str("blind vault replica transport expired"),
            Self::Unavailable(_) => {
                formatter.write_str("blind vault replica transport unavailable")
            }
        }
    }
}

impl<IoError: 'static> StdError for BlindVaultReplicaTransportError<IoError> {}

fn is_m12_effect_context(
    effect_ordinal: u16,
    effect_count: u16,
    purpose: OnionRoutePurpose,
) -> bool {
    if effect_count != 3 {
        return false;
    }
    matches!(
        (effect_ordinal, purpose),
        (0, OnionRoutePurpose::BlindVaultLeaseAdmission)
            | (1, OnionRoutePurpose::BlindVaultPutReceipt)
            | (2, OnionRoutePurpose::BlindVaultLeaseInventory)
    )
}

fn commitment(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

fn artifact_commitment(
    binding: &BlindVaultReplicaEffectBindingV1,
    payload_commitment: &[u8; 32],
    body_commitment: &[u8; 32],
    expires_at_ms: u64,
) -> [u8; 32] {
    let purpose = binding.purpose.as_str().as_bytes();
    let mut hasher = Sha256::new();
    hasher.update(ARTIFACT_COMMITMENT_DOMAIN);
    hasher.update(binding.attempt.job_id);
    hasher.update(binding.attempt.workflow_id);
    hasher.update(binding.attempt.work_sequence.to_be_bytes());
    hasher.update([binding.attempt.attempt]);
    hasher.update(binding.effect_ordinal.to_be_bytes());
    hasher.update(binding.effect_count.to_be_bytes());
    hasher.update((purpose.len() as u16).to_be_bytes());
    hasher.update(purpose);
    hasher.update(binding.source_node_id);
    hasher.update(binding.target_node_id);
    hasher.update(binding.entry_node_id);
    hasher.update(binding.route_id);
    hasher.update(binding.request_id);
    hasher.update(payload_commitment);
    hasher.update(body_commitment);
    hasher.update(expires_at_ms.to_be_bytes());
    hasher.finalize().into()
}

fn commitment_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right.iter())
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::BlindRelayEnvelope;
    use aeronyx_core::protocol::{encode_onion_reply_request, OnionReplyRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const JOB_ID: [u8; 16] = [0x11; 16];
    const ROUTE_ID: [u8; 16] = [0x22; 16];
    const REQUEST_ID: [u8; 16] = [0x33; 16];
    const NOW_MS: u64 = 1_000_000;
    const EXPIRES_AT_MS: u64 = NOW_MS + 60_000;

    struct RecordingIo {
        calls: AtomicUsize,
        bodies: Mutex<Vec<Vec<u8>>>,
        response: Vec<u8>,
    }

    impl RecordingIo {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                bodies: Mutex::new(Vec::new()),
                response: vec![0xA5; 64],
            }
        }
    }

    impl BlindVaultReplicaOutboundIo for RecordingIo {
        type Error = SecretIoError;

        fn send_exact<'a>(
            &'a self,
            request: BlindVaultReplicaExactOutboundRequest<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, Self::Error>> + Send + 'a>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.bodies
                .lock()
                .expect("recording lock")
                .push(request.exact_outer_request_bytes().to_vec());
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    struct SecretIoError;

    impl fmt::Debug for SecretIoError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("endpoint=https://secret.invalid payload=secret")
        }
    }

    impl fmt::Display for SecretIoError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("node=secret commitment=secret signature=secret")
        }
    }

    fn target_identity() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[0x41; 32]).expect("target identity")
    }

    fn source_identity() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[0x42; 32]).expect("source identity")
    }

    fn encoded_onion_reply_request() -> Vec<u8> {
        let request = OnionReplyRequest::new_source_sealed([0x51; 32], 8 * 1024, vec![0x61; 96])
            .expect("reply request");
        encode_onion_reply_request(&request).expect("encode reply request")
    }

    fn exact_outer_request_bytes(target_node_id: [u8; 32]) -> Vec<u8> {
        let source = source_identity();
        let envelope = BlindRelayEnvelope {
            route_id: ROUTE_ID,
            next_hop: target_node_id,
            ttl: 1,
            encrypted_blob: vec![0x71; 128],
            timestamp: 1_000,
            signature: [0; 64],
        }
        .sign_with(&source);
        serde_json::to_vec(&PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        })
        .expect("encode peer request")
    }

    fn test_binding(target_node_id: [u8; 32]) -> BlindVaultReplicaEffectBindingV1 {
        let attempt = BlindVaultReplicaAttemptBindingV1 {
            job_id: JOB_ID,
            workflow_id: JOB_ID,
            work_sequence: 0,
            attempt: 1,
        };
        BlindVaultReplicaEffectBindingV1::new(
            attempt,
            0,
            3,
            OnionRoutePurpose::BlindVaultLeaseAdmission,
            source_identity().public_key_bytes(),
            target_node_id,
            target_node_id,
            ROUTE_ID,
            REQUEST_ID,
        )
        .expect("effect binding")
    }

    fn test_expectation(target_node_id: [u8; 32]) -> BlindVaultReplicaReplayExpectationV1 {
        BlindVaultReplicaReplayExpectationV1 {
            job_id: JOB_ID,
            workflow_id: JOB_ID,
            work_sequence: 0,
            attempt: 1,
            effect_ordinal: 0,
            effect_count: 3,
            purpose: OnionRoutePurpose::BlindVaultLeaseAdmission,
            source_node_id: source_identity().public_key_bytes(),
            target_node_id,
            entry_node_id: target_node_id,
            route_id: ROUTE_ID,
            request_id: REQUEST_ID,
        }
    }

    fn test_artifact() -> (
        BlindVaultReplicaExactRequestArtifactV1,
        Vec<u8>,
        BlindVaultReplicaReplayExpectationV1,
    ) {
        let target_node_id = target_identity().public_key_bytes();
        let payload = encoded_onion_reply_request();
        let artifact = BlindVaultReplicaExactRequestArtifactV1::new(
            test_binding(target_node_id),
            &payload,
            exact_outer_request_bytes(target_node_id),
            EXPIRES_AT_MS,
        )
        .expect("artifact");
        (artifact, payload, test_expectation(target_node_id))
    }

    #[tokio::test]
    async fn exact_replay_reuses_identical_outer_bytes() {
        let (artifact, _payload, expected) = test_artifact();
        let exact_body = artifact.exact_outer_request_bytes.to_vec();
        let io = RecordingIo::new();

        artifact
            .replay_exact(&expected, NOW_MS, &io)
            .await
            .expect("first replay");
        artifact
            .replay_exact(&expected, NOW_MS + 1, &io)
            .await
            .expect("second replay");

        assert_eq!(io.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            io.bodies.lock().expect("recording lock").as_slice(),
            &[exact_body.clone(), exact_body]
        );
    }

    #[tokio::test]
    async fn context_substitution_fails_before_io() {
        let (artifact, _payload, mut expected) = test_artifact();
        let io = RecordingIo::new();

        expected.effect_ordinal = 1;
        assert!(matches!(
            artifact.replay_exact(&expected, NOW_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Rejected)
        ));
        expected.effect_ordinal = 0;
        expected.target_node_id = IdentityKeyPair::from_bytes(&[0x43; 32])
            .expect("alternate target")
            .public_key_bytes();
        assert!(matches!(
            artifact.replay_exact(&expected, NOW_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Rejected)
        ));
        expected.target_node_id = target_identity().public_key_bytes();
        expected.request_id[0] ^= 0xFF;
        assert!(matches!(
            artifact.replay_exact(&expected, NOW_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Rejected)
        ));
        assert_eq!(io.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn expired_artifact_fails_before_io() {
        let (artifact, _payload, expected) = test_artifact();
        let io = RecordingIo::new();

        assert!(matches!(
            artifact.replay_exact(&expected, EXPIRES_AT_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Expired)
        ));
        assert_eq!(io.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn payload_or_body_mutation_fails_before_io() {
        let (mut artifact, _payload, expected) = test_artifact();
        let io = RecordingIo::new();
        let last = artifact.exact_onion_reply_request_bytes.len() - 1;
        artifact.exact_onion_reply_request_bytes[last] ^= 0x01;

        assert!(matches!(
            artifact.replay_exact(&expected, NOW_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Rejected)
        ));
        artifact.exact_onion_reply_request_bytes[last] ^= 0x01;
        artifact.exact_outer_request_bytes[0] ^= 0x01;
        assert!(matches!(
            artifact.replay_exact(&expected, NOW_MS, &io).await,
            Err(BlindVaultReplicaTransportError::Rejected)
        ));
        assert_eq!(io.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn diagnostics_are_coarse_and_redacted() {
        let (artifact, _, _) = test_artifact();
        let diagnostics = [
            format!("{artifact:?}"),
            artifact.to_string(),
            format!(
                "{:?}",
                BlindVaultReplicaTransportError::Unavailable(SecretIoError)
            ),
            BlindVaultReplicaTransportError::Unavailable(SecretIoError).to_string(),
        ]
        .join(" ");
        for forbidden in [
            "secret.invalid",
            "node=",
            "payload=",
            "commitment=",
            "signature=",
            "https://",
        ] {
            assert!(!diagnostics.contains(forbidden), "leaked {forbidden}");
        }
    }

    #[test]
    fn body_route_or_signature_mismatch_is_rejected() {
        let target_node_id = target_identity().public_key_bytes();
        let payload = encoded_onion_reply_request();
        let mut wrong_route = exact_outer_request_bytes(target_node_id);
        let mut decoded: PeerBlindRelayRequest =
            serde_json::from_slice(&wrong_route).expect("decode request");
        decoded.envelope.route_id[0] ^= 0x01;
        wrong_route = serde_json::to_vec(&decoded).expect("encode request");
        assert!(BlindVaultReplicaExactRequestArtifactV1::new(
            test_binding(target_node_id),
            &payload,
            wrong_route,
            EXPIRES_AT_MS,
        )
        .is_err());

        let mut bad_signature: PeerBlindRelayRequest =
            serde_json::from_slice(&exact_outer_request_bytes(target_node_id))
                .expect("decode request");
        bad_signature.envelope.signature[0] ^= 0x01;
        assert!(BlindVaultReplicaExactRequestArtifactV1::new(
            test_binding(target_node_id),
            &payload,
            serde_json::to_vec(&bad_signature).expect("encode request"),
            EXPIRES_AT_MS,
        )
        .is_err());
    }

    #[test]
    fn m12_effect_order_and_hop_identity_are_fail_closed() {
        assert!(is_m12_effect_context(
            0,
            3,
            OnionRoutePurpose::BlindVaultLeaseAdmission
        ));
        assert!(is_m12_effect_context(
            1,
            3,
            OnionRoutePurpose::BlindVaultPutReceipt
        ));
        assert!(is_m12_effect_context(
            2,
            3,
            OnionRoutePurpose::BlindVaultLeaseInventory
        ));
        assert!(!is_m12_effect_context(
            1,
            3,
            OnionRoutePurpose::BlindVaultPut
        ));
        assert!(!is_m12_effect_context(
            0,
            4,
            OnionRoutePurpose::BlindVaultLeaseAdmission
        ));

        let source_node_id = source_identity().public_key_bytes();
        let target_node_id = target_identity().public_key_bytes();
        let alternate_entry_node_id = IdentityKeyPair::from_bytes(&[0x43; 32])
            .expect("alternate entry")
            .public_key_bytes();
        let attempt = BlindVaultReplicaAttemptBindingV1 {
            job_id: JOB_ID,
            workflow_id: JOB_ID,
            work_sequence: 0,
            attempt: 1,
        };
        let binding = BlindVaultReplicaEffectBindingV1::new(
            attempt,
            0,
            3,
            OnionRoutePurpose::BlindVaultLeaseAdmission,
            source_node_id,
            target_node_id,
            alternate_entry_node_id,
            ROUTE_ID,
            REQUEST_ID,
        )
        .expect("binding");
        assert!(BlindVaultReplicaExactRequestArtifactV1::new(
            binding,
            &encoded_onion_reply_request(),
            exact_outer_request_bytes(target_node_id),
            EXPIRES_AT_MS,
        )
        .is_err());
    }

    #[test]
    fn outer_request_must_use_exact_canonical_json() {
        let target_node_id = target_identity().public_key_bytes();
        let mut body = exact_outer_request_bytes(target_node_id);
        body.push(b' ');
        assert!(BlindVaultReplicaExactRequestArtifactV1::new(
            test_binding(target_node_id),
            &encoded_onion_reply_request(),
            body,
            EXPIRES_AT_MS,
        )
        .is_err());
    }
}
