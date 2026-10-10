// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/coordinator.rs
// ============================================
//! # Exact-target source coordinator
//!
//! Owns the default-off exact-target composition: prepare, result, dispatch
//! arming, response opening, ambiguity marking and restart resume over the
//! source journal, plus acquisition of per-route execution permits.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use std::sync::Arc;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxRouteRequestV1, AnonymousMailboxSourceTerminalCarrierV1,
};
use aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{OnionRoutePurpose, VerifiedOnionRoute};

use crate::api::chat_peer::{prepare_exact_peer_blind_relay_http_request, PeerBlindRelayRequest};
use crate::api::{peer_endpoint_is_permitted, peer_transport_url};

use super::execution::SourceExecutionRegistry;
use super::frame_validation::{ensure_ticket_target, verify_response};
use super::journal::SourceJournalRecord;
use super::state_codec::{
    decode_state, encode_state, source_body_commitment, source_request_commitment,
};
use super::{
    AnonymousMailboxSourceError, AnonymousMailboxSourceOutbound, AnonymousMailboxSourcePhase,
    AnonymousMailboxSourcePrepared, AnonymousMailboxSourceResult, ExactAnonymousMailboxTargetPin,
    ExactAnonymousMailboxTargetResolver, SourceExecutionPermit,
    SqliteAnonymousMailboxSourceJournal,
};

const PEER_BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";

// [MAILBOX-SOURCE-COALESCING 2026-10-01 by Codex] Independent hard ceiling
// for active AND waiting executions across router clones. Normal requests
// also retain the existing configured API admission limit. No identifiers
// survive idle-lane reclamation or enter diagnostics.
const MAX_SOURCE_EXECUTIONS: usize = 1024;

/// Default-off exact-target source composition. It has no network ownership;
/// callers obtain a non-debug typed outbound through `begin_dispatch` and use
/// a separately injected bounded transport.
pub struct AnonymousMailboxSourceCoordinator {
    source_identity: Arc<IdentityKeyPair>,
    resolver: Arc<dyn ExactAnonymousMailboxTargetResolver>,
    journal: Arc<SqliteAnonymousMailboxSourceJournal>,
    executions: SourceExecutionRegistry,
}

impl AnonymousMailboxSourceCoordinator {
    // [MAILBOX-SOURCE-COALESCING 2026-10-01 by Codex] Test-only observation
    // makes duplicate-arrival ordering explicit without a timer-based race.
    #[cfg(test)]
    pub(crate) fn execution_waiters(&self) -> usize {
        self.executions
            .waiters
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Acquire before prepare/result/dispatch and retain through completion.
    /// Busy admission has no journal/network effect. This process-local guard
    /// does not replace durable CAS or promise exactly-once network delivery
    /// across cancellation, timeout or restart.
    pub(crate) async fn acquire_execution(
        &self,
        route_id: [u8; 16],
    ) -> Option<SourceExecutionPermit> {
        self.executions.acquire(route_id).await
    }

    #[must_use]
    pub(crate) fn new(
        source_identity: Arc<IdentityKeyPair>,
        resolver: Arc<dyn ExactAnonymousMailboxTargetResolver>,
        journal: Arc<SqliteAnonymousMailboxSourceJournal>,
    ) -> Self {
        Self {
            source_identity,
            resolver,
            journal,
            executions: SourceExecutionRegistry::new(MAX_SOURCE_EXECUTIONS),
        }
    }

    /// Plans exactly one descriptor-pinned one-hop onion request and commits it
    /// to the encrypted journal before a caller can dispatch its body.
    pub(crate) fn prepare(
        &self,
        pin: ExactAnonymousMailboxTargetPin,
        route_id: [u8; 16],
        terminal_frame: Vec<u8>,
        now: u64,
    ) -> Result<AnonymousMailboxSourcePrepared, AnonymousMailboxSourceError> {
        // [ANONYMOUS-MAILBOX-TICKET-TARGET 2026-09-24 by Codex] A canonical
        // TicketIssue must name the exact pinned responder before even reading
        // the journal. A different inner target cannot receive a signed
        // rejection from this node and would otherwise strand an Armed row.
        ensure_ticket_target(&terminal_frame, &pin.target_node_id)?;
        let request_commitment = source_request_commitment(
            &route_id,
            &pin.target_node_id,
            &pin.descriptor_commitment,
            &terminal_frame,
        );
        if let Some(existing) = self.journal.load(&route_id)? {
            if existing.request_commitment != request_commitment
                || existing.target_node_id != pin.target_node_id
                || existing.descriptor_commitment != pin.descriptor_commitment
            {
                return Err(AnonymousMailboxSourceError::Conflict);
            }
            return Ok(AnonymousMailboxSourcePrepared {
                route_id,
                body: existing.body,
                phase: existing.phase,
            });
        }
        let descriptor = self
            .resolver
            .get_valid_exact(&pin.target_node_id, now)
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        let observed = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        if observed != pin.descriptor_commitment || observed.node_id != pin.target_node_id {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let route = VerifiedOnionRoute::from_signed_descriptors(
            self.source_identity.public_key_bytes(),
            [&descriptor],
            OnionRoutePurpose::AnonymousMailboxV1,
            now,
        )
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        if route.terminal_node_id() != pin.target_node_id
            || route.entry_node_id() != pin.target_node_id
        {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let (carrier, session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            pin.target_node_id,
            terminal_frame.clone(),
        )
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let route_request = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            pin.target_node_id,
            carrier
                .encode()
                .map_err(|_| AnonymousMailboxSourceError::Rejected)?,
            now,
            &self.source_identity,
        )
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let payload = encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(route_request))
            .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let envelope = route
            .build_envelope(&payload, route_id, now, &self.source_identity)
            .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: self.source_identity.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let prepared = prepare_exact_peer_blind_relay_http_request(&request)
            .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        if prepared.route_id() != &route_id {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let body = prepared.body().to_vec();
        let state = encode_state(&body, &terminal_frame, Some(&session), None)?;
        let record = SourceJournalRecord {
            route_id,
            request_commitment,
            target_node_id: pin.target_node_id,
            descriptor_commitment: pin.descriptor_commitment,
            body,
            phase: AnonymousMailboxSourcePhase::Prepared,
            retain_until: None,
            state,
        };
        let record = self.journal.insert_or_exact(&record)?;
        Ok(AnonymousMailboxSourcePrepared {
            route_id,
            body: record.body,
            phase: record.phase,
        })
    }

    /// Returns a durable lifecycle result without exposing encrypted source
    /// state. Completed retries return the identical canonical terminal frame.
    pub(crate) fn result(
        &self,
        route_id: [u8; 16],
    ) -> Result<AnonymousMailboxSourceResult, AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        match record.phase {
            AnonymousMailboxSourcePhase::Prepared => Ok(AnonymousMailboxSourceResult::Prepared),
            AnonymousMailboxSourcePhase::Armed => Ok(AnonymousMailboxSourceResult::Armed),
            AnonymousMailboxSourcePhase::Completed => {
                let decoded = decode_state(&record.state)?;
                Ok(AnonymousMailboxSourceResult::Completed(
                    decoded
                        .completed
                        .ok_or(AnonymousMailboxSourceError::Corrupt)?,
                ))
            }
            AnonymousMailboxSourcePhase::Ambiguous => Ok(AnonymousMailboxSourceResult::Ambiguous),
            AnonymousMailboxSourcePhase::Rejected => Ok(AnonymousMailboxSourceResult::Rejected),
        }
    }

    /// Performs the last exact-target check immediately before I/O, then arms
    /// the immutable journaled body. A stale descriptor, altered body, or
    /// unexpected lifecycle state releases no network request.
    pub(crate) fn begin_dispatch(
        &self,
        route_id: [u8; 16],
        now: u64,
    ) -> Result<AnonymousMailboxSourceOutbound, AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        if matches!(
            record.phase,
            AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
        ) {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        if record.phase == AnonymousMailboxSourcePhase::Ambiguous {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        // Historic Armed rows could have passed the old source admission.
        // Keep the journal readable, but never release such a row to I/O.
        let terminal_frame = decode_state(&record.state)?.terminal_frame;
        ensure_ticket_target(&terminal_frame, &record.target_node_id)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;

        let descriptor = self
            .resolver
            .get_valid_exact(&record.target_node_id, now)
            .ok_or(AnonymousMailboxSourceError::Unavailable)?;
        let observed = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if observed != record.descriptor_commitment || observed.node_id != record.target_node_id {
            return Err(AnonymousMailboxSourceError::Unavailable);
        }
        let route = VerifiedOnionRoute::from_signed_descriptors(
            self.source_identity.public_key_bytes(),
            [&descriptor],
            OnionRoutePurpose::AnonymousMailboxV1,
            now,
        )
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if route.entry_node_id() != record.target_node_id
            || route.terminal_node_id() != record.target_node_id
        {
            return Err(AnonymousMailboxSourceError::Unavailable);
        }
        let endpoint = descriptor
            .descriptor
            .public_endpoint
            .clone()
            .filter(|value| !value.trim().is_empty())
            .ok_or(AnonymousMailboxSourceError::Unavailable)?;
        if !peer_endpoint_is_permitted(&endpoint) {
            return Err(AnonymousMailboxSourceError::Unavailable);
        }
        let url = peer_transport_url(&endpoint, PEER_BLIND_RELAY_PATH)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let request: PeerBlindRelayRequest = serde_json::from_slice(&record.body)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        if request.envelope.route_id != record.route_id
            || request.envelope.next_hop != record.target_node_id
            || request.previous_hop_node_id != self.source_identity.public_key_bytes()
            || request.onward_envelope.is_some()
            || request.onward_descriptor_hint.is_some()
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let source_public = IdentityPublicKey::from_bytes(&self.source_identity.public_key_bytes())
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        request
            .envelope
            .verify_signature_from(&source_public)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        if record.phase == AnonymousMailboxSourcePhase::Prepared {
            self.journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                record.state.clone(),
            )?;
        }
        Ok(AnonymousMailboxSourceOutbound {
            route_id: record.route_id,
            target_node_id: record.target_node_id,
            url,
            body: record.body,
            request,
        })
    }

    /// Opens and verifies one source-sealed terminal response. Any malformed,
    /// mismatched, or unauthenticated response consumes the persisted one-shot
    /// session and durably transitions to Ambiguous before reporting failure.
    pub(crate) fn open_response(
        &self,
        route_id: [u8; 16],
        encoded_response: &[u8],
    ) -> Result<(), AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        if record.phase == AnonymousMailboxSourcePhase::Completed
            || record.phase == AnonymousMailboxSourcePhase::Rejected
        {
            return Ok(());
        }
        if record.phase == AnonymousMailboxSourcePhase::Prepared {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        if record.phase == AnonymousMailboxSourcePhase::Ambiguous {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        let decoded = decode_state(&record.state)?;
        if decoded.body_commitment != source_body_commitment(&record.body) {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let terminal_frame = decoded.terminal_frame;
        let mut session = decoded
            .restart
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        let opened = session.open(encoded_response);
        let response_frame =
            match opened.and_then(|bytes| decode_anonymous_mailbox_terminal_frame(&bytes)) {
                Ok(frame) => frame,
                Err(_) => {
                    self.journal.transition(
                        &record,
                        AnonymousMailboxSourcePhase::Armed,
                        AnonymousMailboxSourcePhase::Ambiguous,
                        encode_state(&record.body, &terminal_frame, None, None)?,
                    )?;
                    return Err(AnonymousMailboxSourceError::Ambiguous);
                }
            };
        if verify_response(&terminal_frame, &response_frame, &record.target_node_id).is_err() {
            self.journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Armed,
                AnonymousMailboxSourcePhase::Ambiguous,
                encode_state(&record.body, &terminal_frame, None, None)?,
            )?;
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        let completed = encode_anonymous_mailbox_terminal_frame(&response_frame)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        self.journal.transition(
            &record,
            AnonymousMailboxSourcePhase::Armed,
            AnonymousMailboxSourcePhase::Completed,
            encode_state(&record.body, &terminal_frame, None, Some(&completed))?,
        )
    }

    /// Consumes an armed request after an authenticated transport response
    /// violates the terminal/receipt contract before it can enter the sealed
    /// response opener. This is intentionally irreversible: a caller cannot
    /// turn a potentially observed remote outcome into a resend.
    pub(crate) fn mark_ambiguous(
        &self,
        route_id: [u8; 16],
    ) -> Result<(), AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        match record.phase {
            AnonymousMailboxSourcePhase::Armed => {
                let decoded = decode_state(&record.state)?;
                if decoded.body_commitment != source_body_commitment(&record.body) {
                    return Err(AnonymousMailboxSourceError::Corrupt);
                }
                self.journal.transition(
                    &record,
                    AnonymousMailboxSourcePhase::Armed,
                    AnonymousMailboxSourcePhase::Ambiguous,
                    encode_state(&record.body, &decoded.terminal_frame, None, None)?,
                )
            }
            AnonymousMailboxSourcePhase::Ambiguous => Err(AnonymousMailboxSourceError::Ambiguous),
            AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected => {
                Ok(())
            }
            AnonymousMailboxSourcePhase::Prepared => Err(AnonymousMailboxSourceError::Rejected),
        }
    }

    /// Reopens only the retained immutable request after process restart.
    pub(crate) fn resume(
        &self,
        route_id: [u8; 16],
    ) -> Result<AnonymousMailboxSourcePrepared, AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        Ok(AnonymousMailboxSourcePrepared {
            route_id,
            body: record.body,
            phase: record.phase,
        })
    }
}

#[cfg(test)]
mod tests;
