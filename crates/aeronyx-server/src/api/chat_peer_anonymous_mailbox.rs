// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_anonymous_mailbox.rs
// ============================================
//! Anonymous mailbox terminal dispatch for the authenticated blind-relay path.
//!
//! The node sees only signed opaque capabilities and sealed bytes.  This module
//! deliberately does not expose participant identity, contact, route, wallet,
//! content-key, or plaintext fields.
//!
//! ## Last Modified
//! v1.0.2-CrossEntryVertical — M13G source-to-custody cross-entry proof.

use std::sync::Arc;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxAdmissionTicketV1, AnonymousMailboxOperationV1, AnonymousMailboxOutcomeV1,
    AnonymousMailboxPullResultV1, AnonymousMailboxRouteRequestV1,
    AnonymousMailboxSourceSealedResponseV1, AnonymousMailboxSourceTerminalCarrierV1,
    AnonymousMailboxTerminalFrameV1, AnonymousMailboxTerminalResponseV1,
    AnonymousMailboxTicketIssueResponseV1,
};
use aeronyx_core::protocol::memchain::{decode_memchain, MemChainMessage, MEMCHAIN_MAGIC};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

use crate::services::chat_relay_mailbox::{
    AnonymousMailboxAckOutcome, AnonymousMailboxCreateOutcome, AnonymousMailboxCustodyRepository,
    AnonymousMailboxPullOutcome, AnonymousMailboxPutOutcome, AnonymousMailboxStoreError,
    AnonymousMailboxTicketIssueOutcome,
};

/// [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-02 by Codex] A prepared request owns
/// the decoded source carrier and exact route request.  It has no Debug output
/// because it contains opaque capability and sealed-byte material.
pub(super) struct PreparedAnonymousMailboxTerminal {
    route_request: AnonymousMailboxRouteRequestV1,
    carrier: AnonymousMailboxSourceTerminalCarrierV1,
    request: AnonymousMailboxTerminalFrameV1,
}

/// Coarse terminal execution failure; it intentionally has no source detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AnonymousMailboxTerminalFailure {
    Rejected,
    Unavailable,
}

impl PreparedAnonymousMailboxTerminal {
    /// Decodes the sole canonical mailbox terminal carrier after onion peel.
    /// No route signature is verified against the previous hop: each inner
    /// ticket/deposit/read signature remains the mutation authority.
    pub(super) fn decode(
        payload: &[u8],
        actual_route_id: [u8; 16],
        local_target_node_id: [u8; 32],
    ) -> Result<Self, AnonymousMailboxTerminalFailure> {
        if payload.first().copied() != Some(MEMCHAIN_MAGIC) {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        let MemChainMessage::AnonymousMailboxRouteV1(route_request) =
            decode_memchain(&payload[1..])
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?
        else {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        };
        if route_request.request_id != actual_route_id
            || route_request.target_node_id != local_target_node_id
            || route_request.signing_bytes().is_err()
        {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        let carrier = AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
            &route_request.sealed_terminal_frame,
            actual_route_id,
            local_target_node_id,
        )
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        let request = decode_anonymous_mailbox_terminal_frame(carrier.terminal_frame())
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        if !matches!(
            request,
            AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
                | AnonymousMailboxTerminalFrameV1::Put(_)
                | AnonymousMailboxTerminalFrameV1::PullOne(_)
                | AnonymousMailboxTerminalFrameV1::Ack(_)
                | AnonymousMailboxTerminalFrameV1::TicketIssue(_)
        ) {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        Ok(Self {
            route_request,
            carrier,
            request,
        })
    }

    pub(super) fn execute(
        self,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        terminal_identity: Arc<IdentityKeyPair>,
        now: u64,
    ) -> Result<String, AnonymousMailboxTerminalFailure> {
        if let AnonymousMailboxTerminalFrameV1::TicketIssue(request) = &self.request {
            // [BLIND-RELAY-ANONYMOUS-MAILBOX-TICKET 2026-09-03 by Codex]
            // Ticket issuance has a dedicated target-signed response: it
            // carries the exact durable ticket only for Accepted and must not
            // be projected through the generic terminal response payload.
            let (outcome, ticket) = map_ticket_issue(repository.issue_ticket(request, now))?;
            let response = AnonymousMailboxTicketIssueResponseV1::signed(
                request,
                outcome,
                ticket,
                now,
                &terminal_identity,
            )
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            let response_frame = encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
            )
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            return self.seal_response_frame(response_frame, &terminal_identity);
        }
        let (operation, request_id, request_commitment, outcome, payload) = match &self.request {
            AnonymousMailboxTerminalFrameV1::LeaseCreate(request) => {
                request
                    .verify_for_target(&terminal_identity.public_key_bytes(), now)
                    .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
                let outcome = map_create(repository.create(request, now))?;
                (
                    AnonymousMailboxOperationV1::LeaseCreate,
                    request.admission.ticket_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            AnonymousMailboxTerminalFrameV1::Put(request) => {
                let outcome = map_put(repository.put(request, now))?;
                (
                    AnonymousMailboxOperationV1::Put,
                    request.item_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            AnonymousMailboxTerminalFrameV1::PullOne(request) => {
                let (outcome, payload) = map_pull(repository.pull_one(request, now))?;
                (
                    AnonymousMailboxOperationV1::PullOne,
                    request.request_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    payload,
                )
            }
            AnonymousMailboxTerminalFrameV1::Ack(request) => {
                let outcome = map_ack(repository.ack(request, now))?;
                (
                    AnonymousMailboxOperationV1::Ack,
                    request.request_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            _ => return Err(AnonymousMailboxTerminalFailure::Rejected),
        };
        let response = AnonymousMailboxTerminalResponseV1::signed(
            operation,
            request_id,
            request_commitment,
            outcome,
            payload,
            now,
            &terminal_identity,
        )
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        let response_frame = match operation {
            AnonymousMailboxOperationV1::LeaseCreate => {
                AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response)
            }
            AnonymousMailboxOperationV1::Put => {
                AnonymousMailboxTerminalFrameV1::PutResponse(response)
            }
            AnonymousMailboxOperationV1::PullOne => {
                AnonymousMailboxTerminalFrameV1::PullOneResponse(response)
            }
            AnonymousMailboxOperationV1::Ack => {
                AnonymousMailboxTerminalFrameV1::AckResponse(response)
            }
            AnonymousMailboxOperationV1::TicketIssue => {
                return Err(AnonymousMailboxTerminalFailure::Rejected)
            }
        };
        let response_frame = encode_anonymous_mailbox_terminal_frame(&response_frame)
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        self.seal_response_frame(response_frame, &terminal_identity)
    }

    fn seal_response_frame(
        &self,
        response_frame: Vec<u8>,
        terminal_identity: &IdentityKeyPair,
    ) -> Result<String, AnonymousMailboxTerminalFailure> {
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            self.route_request.request_id,
            self.carrier.context_commitment(),
            terminal_identity.public_key_bytes(),
            self.carrier.reply_public_key(),
            &response_frame,
            &terminal_identity,
        )
        .and_then(|value| value.encode())
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        // [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-03 by Codex] The onion ACK
        // carries only the compact source-sealed response. The signed route
        // response remains a compatible core type, but wrapping AMSR in it
        // would force middle hops to classify a terminal workload.
        Ok(BASE64.encode(sealed))
    }
}

fn map_store_error(error: AnonymousMailboxStoreError) -> AnonymousMailboxTerminalFailure {
    match error {
        AnonymousMailboxStoreError::Rejected => AnonymousMailboxTerminalFailure::Rejected,
        AnonymousMailboxStoreError::Disabled
        | AnonymousMailboxStoreError::Busy
        | AnonymousMailboxStoreError::UnsupportedSchema
        | AnonymousMailboxStoreError::Corrupt
        | AnonymousMailboxStoreError::Unavailable => AnonymousMailboxTerminalFailure::Unavailable,
    }
}

fn map_create(
    value: Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxCreateOutcome::Created(_) | AnonymousMailboxCreateOutcome::Existing(_) => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxCreateOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxCreateOutcome::AtCapacity => Ok(AnonymousMailboxOutcomeV1::AtCapacity),
    }
}

fn map_put(
    value: Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxPutOutcome::Stored(_) | AnonymousMailboxPutOutcome::Existing(_) => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxPutOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxPutOutcome::AtCapacity => Ok(AnonymousMailboxOutcomeV1::AtCapacity),
        AnonymousMailboxPutOutcome::LeaseExpired => Ok(AnonymousMailboxOutcomeV1::Expired),
        AnonymousMailboxPutOutcome::LeaseNotFound => Ok(AnonymousMailboxOutcomeV1::Rejected),
    }
}

fn map_pull(
    value: Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError>,
) -> Result<(AnonymousMailboxOutcomeV1, Vec<u8>), AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxPullOutcome::Item(item) => {
            let sealed_len = usize::try_from(item.sealed_length)
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            let sealed_item = item
                .padded_sealed_envelope
                .get(..sealed_len)
                .ok_or(AnonymousMailboxTerminalFailure::Rejected)?
                .to_vec();
            let result = AnonymousMailboxPullResultV1::new(item.item_id, item.cursor, sealed_item)
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            if result.sealed_commitment != item.sealed_commitment {
                return Err(AnonymousMailboxTerminalFailure::Rejected);
            }
            let result = result
                .encode()
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            Ok((AnonymousMailboxOutcomeV1::Accepted, result))
        }
        AnonymousMailboxPullOutcome::Empty => Ok((AnonymousMailboxOutcomeV1::Accepted, Vec::new())),
        AnonymousMailboxPullOutcome::LeaseExpired => {
            Ok((AnonymousMailboxOutcomeV1::Expired, Vec::new()))
        }
        AnonymousMailboxPullOutcome::LeaseNotFound => {
            Ok((AnonymousMailboxOutcomeV1::Rejected, Vec::new()))
        }
    }
}

fn map_ack(
    value: Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxAckOutcome::Acknowledged
        | AnonymousMailboxAckOutcome::AlreadyAcknowledged => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxAckOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxAckOutcome::NotFound => Ok(AnonymousMailboxOutcomeV1::Rejected),
    }
}

fn map_ticket_issue(
    value: Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError>,
) -> Result<
    (
        AnonymousMailboxOutcomeV1,
        Option<AnonymousMailboxAdmissionTicketV1>,
    ),
    AnonymousMailboxTerminalFailure,
> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxTicketIssueOutcome::Issued(ticket)
        | AnonymousMailboxTicketIssueOutcome::Existing(ticket) => {
            Ok((AnonymousMailboxOutcomeV1::Accepted, Some(ticket)))
        }
        AnonymousMailboxTicketIssueOutcome::Conflict => {
            Ok((AnonymousMailboxOutcomeV1::Conflict, None))
        }
        AnonymousMailboxTicketIssueOutcome::AtCapacity => {
            Ok((AnonymousMailboxOutcomeV1::AtCapacity, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::config_chat_relay::{AnonymousMailboxSourceConfig, AnonymousMailboxStoreConfig};
    use crate::services::chat_relay_anonymous_mailbox_source::{
        AnonymousMailboxSourceCoordinator, AnonymousMailboxSourceResult,
        ExactAnonymousMailboxTargetPin, ExactAnonymousMailboxTargetResolver,
        SqliteAnonymousMailboxSourceJournal,
    };
    use crate::services::chat_relay_mailbox::AnonymousMailboxCleanupReport;
    use crate::services::chat_relay_mailbox::SqliteAnonymousMailboxStore;
    use aeronyx_core::protocol::anonymous_mailbox::{
        decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
        AnonymousMailboxAckV1, AnonymousMailboxAdmissionTicketV1, AnonymousMailboxLeaseCreateV1,
        AnonymousMailboxPullOneV1, AnonymousMailboxPullResultV1, AnonymousMailboxPutV1,
        AnonymousMailboxSourceSealSessionV1, AnonymousMailboxTicketIssueV1,
    };
    use aeronyx_core::protocol::discovery::{
        DirectoryDescriptorCommitmentV1, NodeCapability, NodeDescriptor, NodeProtocolFeature,
        SignedNodeDescriptor,
    };
    use aeronyx_core::protocol::memchain::encode_memchain;
    use aeronyx_core::protocol::onion::open_onion_layer;
    use rusqlite::Connection;

    struct TicketRepository {
        outcomes:
            Mutex<VecDeque<Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError>>>,
        issue_calls: AtomicUsize,
    }

    struct CrossEntryExactResolver {
        descriptor: SignedNodeDescriptor,
        exact_calls: AtomicUsize,
        wrong_target_calls: AtomicUsize,
    }

    impl ExactAnonymousMailboxTargetResolver for CrossEntryExactResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _now: u64) -> Option<SignedNodeDescriptor> {
            if node_id != &self.descriptor.descriptor.node_id {
                self.wrong_target_calls.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.exact_calls.fetch_add(1, Ordering::Relaxed);
            Some(self.descriptor.clone())
        }
    }

    impl TicketRepository {
        fn new(
            outcomes: impl IntoIterator<
                Item = Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError>,
            >,
        ) -> Self {
            Self {
                outcomes: Mutex::new(outcomes.into_iter().collect()),
                issue_calls: AtomicUsize::new(0),
            }
        }
    }

    impl AnonymousMailboxCustodyRepository for TicketRepository {
        fn issue_ticket(
            &self,
            _request: &AnonymousMailboxTicketIssueV1,
            _now: u64,
        ) -> Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError> {
            self.issue_calls.fetch_add(1, Ordering::Relaxed);
            self.outcomes
                .lock()
                .expect("ticket outcomes")
                .pop_front()
                .expect("one configured ticket outcome")
        }

        fn create(
            &self,
            _request: &aeronyx_core::protocol::anonymous_mailbox::AnonymousMailboxLeaseCreateV1,
            _now: u64,
        ) -> Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError> {
            Err(AnonymousMailboxStoreError::Unavailable)
        }

        fn put(
            &self,
            _request: &aeronyx_core::protocol::anonymous_mailbox::AnonymousMailboxPutV1,
            _now: u64,
        ) -> Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError> {
            Err(AnonymousMailboxStoreError::Unavailable)
        }

        fn pull_one(
            &self,
            _request: &AnonymousMailboxPullOneV1,
            _now: u64,
        ) -> Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError> {
            Err(AnonymousMailboxStoreError::Unavailable)
        }

        fn ack(
            &self,
            _request: &aeronyx_core::protocol::anonymous_mailbox::AnonymousMailboxAckV1,
            _now: u64,
        ) -> Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError> {
            Err(AnonymousMailboxStoreError::Unavailable)
        }

        fn cleanup(
            &self,
            _now: u64,
        ) -> Result<AnonymousMailboxCleanupReport, AnonymousMailboxStoreError> {
            Err(AnonymousMailboxStoreError::Unavailable)
        }
    }

    fn ticket_issue(
        target: &IdentityKeyPair,
        request_id: [u8; 16],
    ) -> AnonymousMailboxTicketIssueV1 {
        AnonymousMailboxTicketIssueV1::new(
            request_id,
            [0x72; 16],
            target.public_key_bytes(),
            [0x73; 32],
            1_800_000_000,
            1_800_000_300,
            0,
        )
        .expect("ticket request")
    }

    fn routed_ticket_issue(
        route_id: [u8; 16],
        target: &IdentityKeyPair,
        request: AnonymousMailboxTicketIssueV1,
    ) -> (Vec<u8>, AnonymousMailboxSourceSealSessionV1) {
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(request),
        )
        .expect("terminal frame");
        routed_terminal_frame(route_id, target, terminal_frame)
    }

    fn routed_terminal_frame(
        route_id: [u8; 16],
        target: &IdentityKeyPair,
        terminal_frame: Vec<u8>,
    ) -> (Vec<u8>, AnonymousMailboxSourceSealSessionV1) {
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).expect("source");
        let (carrier, session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("carrier");
        let route = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            target.public_key_bytes(),
            carrier.encode().expect("carrier bytes"),
            1_800_000_000,
            &source,
        )
        .expect("route");
        (
            encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(route)).expect("outer"),
            session,
        )
    }

    fn execute_terminal_frame(
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        target: &IdentityKeyPair,
        route_id: [u8; 16],
        request: AnonymousMailboxTerminalFrameV1,
    ) -> AnonymousMailboxTerminalFrameV1 {
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&request).expect("terminal frame");
        let (encoded, mut session) = routed_terminal_frame(route_id, target, terminal_frame);
        let sealed =
            PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, target.public_key_bytes())
                .expect("canonical terminal request")
                .execute(repository, Arc::new(target.clone()), 1_800_000_000)
                .expect("terminal response");
        let sealed = BASE64.decode(sealed).expect("base64 AMSR");
        decode_anonymous_mailbox_terminal_frame(&session.open(&sealed).expect("one-shot AMSR open"))
            .expect("canonical response frame")
    }

    fn execute_ticket(
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        target: &IdentityKeyPair,
        route_id: [u8; 16],
        request: AnonymousMailboxTicketIssueV1,
    ) -> AnonymousMailboxTicketIssueResponseV1 {
        let (encoded, mut session) = routed_ticket_issue(route_id, target, request.clone());
        let sealed =
            PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, target.public_key_bytes())
                .expect("canonical ticket request")
                .execute(repository, Arc::new(target.clone()), 1_800_000_000)
                .expect("ticket terminal response");
        let sealed = BASE64.decode(sealed).expect("base64 AMSR");
        let frame = decode_anonymous_mailbox_terminal_frame(
            &session.open(&sealed).expect("one-shot AMSR open"),
        )
        .expect("canonical response frame");
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response) = frame else {
            panic!("ticket issuance must use its dedicated response frame");
        };
        response
            .verify_for_request(&request, &target.public_key_bytes())
            .expect("request-bound target response");
        response
    }

    fn cross_entry_descriptor(target: &IdentityKeyPair) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            17,
            1_799_999_999,
            1_800_000_120,
            "cross-entry-test",
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
        SignedNodeDescriptor::sign(descriptor, target).expect("signed cross-entry descriptor")
    }

    fn cross_entry_coordinator(
        source_seed: u8,
        resolver: Arc<CrossEntryExactResolver>,
        journal_key: u8,
    ) -> AnonymousMailboxSourceCoordinator {
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            max_journal_entries: 16,
            max_journal_bytes: 8 * 1024 * 1024,
            ..AnonymousMailboxSourceConfig::default()
        };
        let journal = SqliteAnonymousMailboxSourceJournal::new(
            Connection::open_in_memory().expect("source journal"),
            [journal_key; 32],
            &config,
        )
        .expect("source journal schema");
        AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[source_seed; 32]).expect("source identity")),
            resolver,
            Arc::new(journal),
        )
    }

    fn dispatch_cross_entry_terminal(
        coordinator: &AnonymousMailboxSourceCoordinator,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        target: &IdentityKeyPair,
        descriptor_commitment: DirectoryDescriptorCommitmentV1,
        route_id: [u8; 16],
        request: AnonymousMailboxTerminalFrameV1,
    ) -> (Vec<u8>, Vec<u8>) {
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&request).expect("terminal frame");
        let prepared = coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(
                    target.public_key_bytes(),
                    descriptor_commitment,
                ),
                route_id,
                terminal_frame,
                1_800_000_000,
            )
            .expect("source prepare");
        let outbound = coordinator
            .begin_dispatch(route_id, 1_800_000_000)
            .expect("source exact-target dispatch");
        assert_eq!(prepared.body(), outbound.body());
        assert_eq!(outbound.target_node_id(), &target.public_key_bytes());

        let (target_kem_secret, _) = target.to_x25519();
        let peel = open_onion_layer(
            &outbound.request().envelope.encrypted_blob,
            &target_kem_secret,
        )
        .expect("target peels one-hop source route");
        assert!(peel.next_hop.is_none());
        let sealed = PreparedAnonymousMailboxTerminal::decode(
            &peel.inner,
            route_id,
            target.public_key_bytes(),
        )
        .expect("target decodes source carrier")
        .execute(repository, Arc::new(target.clone()), 1_800_000_000)
        .expect("target executes terminal request");
        (
            outbound.body().to_vec(),
            BASE64.decode(sealed).expect("source-sealed response"),
        )
    }

    fn complete_cross_entry_response(
        coordinator: &AnonymousMailboxSourceCoordinator,
        route_id: [u8; 16],
        sealed_response: &[u8],
    ) -> AnonymousMailboxTerminalFrameV1 {
        coordinator
            .open_response(route_id, sealed_response)
            .expect("source verifies terminal response");
        let AnonymousMailboxSourceResult::Completed(response) = coordinator
            .result(route_id)
            .expect("source terminal result")
        else {
            panic!("source request must be completed");
        };
        decode_anonymous_mailbox_terminal_frame(&response).expect("canonical completed response")
    }

    fn routed_pull(route_id: [u8; 16], target: &IdentityKeyPair) -> Vec<u8> {
        let source = IdentityKeyPair::from_bytes(&[0x61; 32]).expect("source");
        let reader = IdentityKeyPair::from_bytes(&[0x62; 32]).expect("reader");
        let pull = AnonymousMailboxPullOneV1::new(
            [0x63; 32],
            [0x64; 16],
            Vec::new(),
            1_800_000_000,
            &reader,
        )
        .expect("pull");
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(pull),
        )
        .expect("terminal frame");
        let (carrier, _session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("carrier");
        let route = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            target.public_key_bytes(),
            carrier.encode().expect("carrier bytes"),
            1_800_000_000,
            &source,
        )
        .expect("route");
        encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(route)).expect("outer")
    }

    #[test]
    fn canonical_carrier_rejects_route_or_target_substitution_before_repository_access() {
        let target = IdentityKeyPair::from_bytes(&[0x65; 32]).expect("target");
        let route_id = [0x66; 16];
        let encoded = routed_pull(route_id, &target);
        assert!(PreparedAnonymousMailboxTerminal::decode(
            &encoded,
            route_id,
            target.public_key_bytes(),
        )
        .is_ok());
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &encoded,
                [0x67; 16],
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, [0x68; 32]),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
    }

    #[test]
    fn ticket_issue_is_source_sealed_and_returns_the_durable_ticket() {
        let target = IdentityKeyPair::from_bytes(&[0x74; 32]).expect("target");
        let request = ticket_issue(&target, [0x75; 16]);
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            request.ticket_id,
            request.lease_claims_commitment,
            request.issued_at,
            request.expires_at,
            &target,
        )
        .expect("durable ticket");
        let repository = Arc::new(TicketRepository::new([Ok(
            AnonymousMailboxTicketIssueOutcome::Issued(ticket.clone()),
        )]));
        let response = execute_ticket(repository.clone(), &target, [0x76; 16], request);
        assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(response.ticket.as_ref() == Some(&ticket));
        assert_eq!(repository.issue_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ticket_issue_exact_replay_returns_the_original_ticket() {
        let target = IdentityKeyPair::from_bytes(&[0x77; 32]).expect("target");
        let request = ticket_issue(&target, [0x78; 16]);
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            request.ticket_id,
            request.lease_claims_commitment,
            request.issued_at,
            request.expires_at,
            &target,
        )
        .expect("durable ticket");
        let repository = Arc::new(TicketRepository::new([
            Ok(AnonymousMailboxTicketIssueOutcome::Issued(ticket.clone())),
            Ok(AnonymousMailboxTicketIssueOutcome::Existing(ticket.clone())),
        ]));
        let issued = execute_ticket(repository.clone(), &target, [0x79; 16], request.clone());
        let replay = execute_ticket(repository.clone(), &target, [0x7a; 16], request);
        assert!(issued.ticket.as_ref() == Some(&ticket));
        assert!(replay.ticket.as_ref() == Some(&ticket));
        assert_eq!(repository.issue_calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn ticket_issue_conflict_and_capacity_remain_coarse_without_ticket() {
        let target = IdentityKeyPair::from_bytes(&[0x7b; 32]).expect("target");
        let repository = Arc::new(TicketRepository::new([
            Ok(AnonymousMailboxTicketIssueOutcome::Conflict),
            Ok(AnonymousMailboxTicketIssueOutcome::AtCapacity),
        ]));
        let conflict = execute_ticket(
            repository.clone(),
            &target,
            [0x7c; 16],
            ticket_issue(&target, [0x7d; 16]),
        );
        let capacity = execute_ticket(
            repository.clone(),
            &target,
            [0x7e; 16],
            ticket_issue(&target, [0x7f; 16]),
        );
        assert_eq!(conflict.outcome, AnonymousMailboxOutcomeV1::Conflict);
        assert!(conflict.ticket.is_none());
        assert_eq!(capacity.outcome, AnonymousMailboxOutcomeV1::AtCapacity);
        assert!(capacity.ticket.is_none());
        assert_eq!(repository.issue_calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn ticket_issue_policy_rejection_never_builds_a_ticket_response() {
        let target = IdentityKeyPair::from_bytes(&[0x80; 32]).expect("target");
        let request = ticket_issue(&target, [0x81; 16]);
        let (encoded, _session) = routed_ticket_issue([0x82; 16], &target, request);
        let repository = Arc::new(TicketRepository::new([Err(
            AnonymousMailboxStoreError::Rejected,
        )]));
        assert_eq!(
            PreparedAnonymousMailboxTerminal::decode(
                &encoded,
                [0x82; 16],
                target.public_key_bytes(),
            )
            .expect("canonical ticket request")
            .execute(repository.clone(), Arc::new(target), 1_800_000_000),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        );
        assert_eq!(repository.issue_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn real_store_terminal_lifecycle_survives_restart_and_exact_retries() {
        // [BLIND-RELAY-ANONYMOUS-MAILBOX-VERTICAL 2026-09-03 by Codex] This
        // exercises the production adapter and SQLite repository together;
        // no socket, participant identity, clear receipt, or fallback peer is
        // involved.
        let directory = tempfile::tempdir().expect("private temporary directory");
        let private_directory = std::fs::canonicalize(directory.path()).expect("canonical path");
        let config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: private_directory
                .join("anonymous-mailbox.sqlite")
                .display()
                .to_string(),
            max_leases_total: 4,
            max_items_total: 1_024,
            max_bytes_total: 1024 * 1024,
            max_in_flight: 4,
            cleanup_batch_size: 8,
            max_outstanding_tickets: 4,
            max_ticket_issues_per_window: 1,
            ticket_issuance_window_secs: 60,
            ticket_issue_work_bits: 1,
        };
        let target = IdentityKeyPair::from_bytes(&[0x91; 32]).expect("target");
        let depositor = IdentityKeyPair::from_bytes(&[0x92; 32]).expect("depositor");
        let reader = IdentityKeyPair::from_bytes(&[0x93; 32]).expect("reader");
        let mailbox_id = [0x94; 32];
        let ticket_id = [0x95; 16];
        let lease_expires_at = 1_800_001_000;
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            4,
            16 * 1024,
            1_800_000_000,
            lease_expires_at,
        );
        let ticket_request = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    [0x96; 16],
                    ticket_id,
                    target.public_key_bytes(),
                    claims,
                    1_800_000_000,
                    1_800_000_300,
                    proof_nonce,
                )
                .expect("ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit proof");

        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                [0x97; 32],
            )
            .expect("open real store"),
        );

        let (encoded, _) = routed_ticket_issue([0x98; 16], &target, ticket_request.clone());
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &encoded,
                [0x99; 16],
                target.public_key_bytes()
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(&encoded, [0x98; 16], [0x9a; 32]),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));

        let issued = execute_terminal_frame(
            store.clone(),
            &target,
            [0x98; 16],
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(issued) = issued else {
            panic!("ticket response kind");
        };
        issued
            .verify_for_request(&ticket_request, &target.public_key_bytes())
            .expect("target-bound ticket response");
        assert_eq!(issued.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let ticket = issued.ticket.expect("durable ticket");

        let replayed = execute_terminal_frame(
            store.clone(),
            &target,
            [0x9b; 16],
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(replayed) = replayed else {
            panic!("ticket replay response kind");
        };
        assert_eq!(replayed.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(replayed.ticket.as_ref() == Some(&ticket));

        let conflicting_ticket = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    ticket_request.request_id,
                    [0x9c; 16],
                    target.public_key_bytes(),
                    claims,
                    1_800_000_000,
                    1_800_000_300,
                    proof_nonce,
                )
                .expect("conflicting ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit conflict proof");
        let conflict = execute_terminal_frame(
            store.clone(),
            &target,
            [0x9d; 16],
            AnonymousMailboxTerminalFrameV1::TicketIssue(conflicting_ticket),
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(conflict) = conflict else {
            panic!("ticket conflict response kind");
        };
        assert_eq!(conflict.outcome, AnonymousMailboxOutcomeV1::Conflict);
        assert!(conflict.ticket.is_none());

        let lease = AnonymousMailboxLeaseCreateV1::new(
            mailbox_id,
            depositor.public_key_bytes(),
            4,
            16 * 1024,
            1_800_000_000,
            lease_expires_at,
            ticket,
            &reader,
        )
        .expect("lease request");
        for route_id in [[0xa0; 16], [0xa1; 16]] {
            let response = execute_terminal_frame(
                store.clone(),
                &target,
                route_id,
                AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            );
            let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response) = response else {
                panic!("lease response kind");
            };
            assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
            assert!(response.sealed_payload.is_empty());
        }

        let opaque_item = vec![0xa2; 4096];
        let put = AnonymousMailboxPutV1::new(
            mailbox_id,
            [0xa3; 16],
            opaque_item.clone(),
            1_800_000_000,
            1_800_000_600,
            &depositor,
        )
        .expect("put request");
        for route_id in [[0xa4; 16], [0xa5; 16]] {
            let response = execute_terminal_frame(
                store.clone(),
                &target,
                route_id,
                AnonymousMailboxTerminalFrameV1::Put(put.clone()),
            );
            let AnonymousMailboxTerminalFrameV1::PutResponse(response) = response else {
                panic!("put response kind");
            };
            assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        }
        let changed = AnonymousMailboxPutV1::new(
            mailbox_id,
            put.item_id,
            vec![0xa6; 4096],
            put.issued_at,
            put.expires_at,
            &depositor,
        )
        .expect("conflicting put");
        let response = execute_terminal_frame(
            store.clone(),
            &target,
            [0xa7; 16],
            AnonymousMailboxTerminalFrameV1::Put(changed),
        );
        let AnonymousMailboxTerminalFrameV1::PutResponse(response) = response else {
            panic!("put conflict response kind");
        };
        assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Conflict);
        drop(store);

        let reopened = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                [0x97; 32],
            )
            .expect("restart store"),
        );
        let pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xa8; 16],
            Vec::new(),
            1_800_000_001,
            &reader,
        )
        .expect("pull request");
        let response = execute_terminal_frame(
            reopened.clone(),
            &target,
            [0xa9; 16],
            AnonymousMailboxTerminalFrameV1::PullOne(pull.clone()),
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(response) = response else {
            panic!("pull response kind");
        };
        response
            .verify_for_request(
                AnonymousMailboxOperationV1::PullOne,
                &pull.request_id,
                &pull.request_commitment().expect("pull commitment"),
                &target.public_key_bytes(),
            )
            .expect("pull response binding");
        assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let pulled = AnonymousMailboxPullResultV1::decode(&response.sealed_payload)
            .expect("canonical pull result");
        assert_eq!(pulled.item_id, put.item_id);
        assert_eq!(pulled.sealed_item, opaque_item);
        assert_eq!(pulled.sealed_commitment, put.sealed_commitment());

        let ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xaa; 16],
            pulled.item_id,
            pulled.sealed_commitment,
            1_800_000_002,
            &reader,
        )
        .expect("ack request");
        for route_id in [[0xab; 16], [0xac; 16]] {
            let response = execute_terminal_frame(
                reopened.clone(),
                &target,
                route_id,
                AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
            );
            let AnonymousMailboxTerminalFrameV1::AckResponse(response) = response else {
                panic!("ack response kind");
            };
            assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        }
        drop(reopened);

        let restarted = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config,
                target.clone(),
                [0x97; 32],
            )
            .expect("second restart"),
        );
        let empty_pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xad; 16],
            Vec::new(),
            1_800_000_003,
            &reader,
        )
        .expect("empty pull request");
        let response = execute_terminal_frame(
            restarted,
            &target,
            [0xae; 16],
            AnonymousMailboxTerminalFrameV1::PullOne(empty_pull),
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(response) = response else {
            panic!("empty pull response kind");
        };
        assert_eq!(response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(response.sealed_payload.is_empty());
    }

    #[test]
    fn cross_entry_source_terminal_store_pull_and_ack_are_exact_targeted() {
        // [ANONYMOUS-MAILBOX-CROSS-ENTRY 2026-09-03 by Codex] Model sender
        // entry M and later receiver entry R as distinct source coordinators.
        // Both receive only the same receiver-provided T pin. The generated
        // onion is peeled by T, whose real SQLite store survives between Put
        // and Pull/Ack. No socket, alternate candidate, wallet or identity
        // locator participates in the proof.
        let directory = tempfile::tempdir().expect("private temporary directory");
        let private_directory = std::fs::canonicalize(directory.path()).expect("canonical path");
        let store_config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: private_directory
                .join("cross-entry-mailbox.sqlite")
                .display()
                .to_string(),
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
        let target = IdentityKeyPair::from_bytes(&[0xb1; 32]).expect("target T");
        let descriptor = cross_entry_descriptor(&target);
        let descriptor_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                .expect("descriptor commitment");
        let m_resolver = Arc::new(CrossEntryExactResolver {
            descriptor: descriptor.clone(),
            exact_calls: AtomicUsize::new(0),
            wrong_target_calls: AtomicUsize::new(0),
        });
        let r_resolver = Arc::new(CrossEntryExactResolver {
            descriptor,
            exact_calls: AtomicUsize::new(0),
            wrong_target_calls: AtomicUsize::new(0),
        });
        let entry_m = cross_entry_coordinator(0xb2, Arc::clone(&m_resolver), 0xb3);
        let entry_r = cross_entry_coordinator(0xb4, Arc::clone(&r_resolver), 0xb5);
        let depositor = IdentityKeyPair::from_bytes(&[0xb6; 32]).expect("deposit capability");
        let reader = IdentityKeyPair::from_bytes(&[0xb7; 32]).expect("read capability");
        let mailbox_id = [0xb8; 32];
        let ticket_id = [0xb9; 16];
        let lease_expires_at = 1_800_001_000;
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            4,
            16 * 1024,
            1_800_000_000,
            lease_expires_at,
        );
        let ticket_request = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    [0xba; 16],
                    ticket_id,
                    target.public_key_bytes(),
                    claims,
                    1_800_000_000,
                    1_800_000_300,
                    proof_nonce,
                )
                .expect("ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit proof");
        let cursor_secret = [0xbb; 32];
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("open target store"),
        );

        let ticket_route = [0xbc; 16];
        let (_, sealed_ticket) = dispatch_cross_entry_terminal(
            &entry_m,
            store.clone(),
            &target,
            descriptor_commitment,
            ticket_route,
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket_response) =
            complete_cross_entry_response(&entry_m, ticket_route, &sealed_ticket)
        else {
            panic!("ticket response kind");
        };
        ticket_response
            .verify_for_request(&ticket_request, &target.public_key_bytes())
            .expect("target-bound ticket response");
        let ticket = ticket_response.ticket.expect("issued ticket");

        let lease = AnonymousMailboxLeaseCreateV1::new(
            mailbox_id,
            depositor.public_key_bytes(),
            4,
            16 * 1024,
            1_800_000_000,
            lease_expires_at,
            ticket,
            &reader,
        )
        .expect("lease request");
        let lease_route = [0xbd; 16];
        let (_, sealed_lease) = dispatch_cross_entry_terminal(
            &entry_m,
            store.clone(),
            &target,
            descriptor_commitment,
            lease_route,
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(lease_response) =
            complete_cross_entry_response(&entry_m, lease_route, &sealed_lease)
        else {
            panic!("lease response kind");
        };
        assert_eq!(lease_response.outcome, AnonymousMailboxOutcomeV1::Accepted);

        let opaque_item = vec![0xbe; 4096];
        let put = AnonymousMailboxPutV1::new(
            mailbox_id,
            [0xbf; 16],
            opaque_item.clone(),
            1_800_000_000,
            1_800_000_600,
            &depositor,
        )
        .expect("put request");
        let put_route = [0xc0; 16];
        let (put_body, sealed_put) = dispatch_cross_entry_terminal(
            &entry_m,
            store.clone(),
            &target,
            descriptor_commitment,
            put_route,
            AnonymousMailboxTerminalFrameV1::Put(put.clone()),
        );
        let replay = entry_m
            .begin_dispatch(put_route, 1_800_000_001)
            .expect("exact armed replay");
        assert_eq!(replay.body(), put_body, "lost response replays exact bytes");
        let AnonymousMailboxTerminalFrameV1::PutResponse(put_response) =
            complete_cross_entry_response(&entry_m, put_route, &sealed_put)
        else {
            panic!("put response kind");
        };
        assert_eq!(put_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        drop(store);

        let restarted = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config,
                target.clone(),
                cursor_secret,
            )
            .expect("restart target store"),
        );
        let pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xc1; 16],
            Vec::new(),
            1_800_000_001,
            &reader,
        )
        .expect("pull request");
        let pull_route = [0xc2; 16];
        let (_, sealed_pull) = dispatch_cross_entry_terminal(
            &entry_r,
            restarted.clone(),
            &target,
            descriptor_commitment,
            pull_route,
            AnonymousMailboxTerminalFrameV1::PullOne(pull),
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(pull_response) =
            complete_cross_entry_response(&entry_r, pull_route, &sealed_pull)
        else {
            panic!("pull response kind");
        };
        let pulled = AnonymousMailboxPullResultV1::decode(&pull_response.sealed_payload)
            .expect("pull result");
        assert_eq!(pulled.item_id, put.item_id);
        assert_eq!(pulled.sealed_commitment, put.sealed_commitment());
        assert_eq!(pulled.sealed_item, opaque_item);

        let ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xc3; 16],
            pulled.item_id,
            pulled.sealed_commitment,
            1_800_000_002,
            &reader,
        )
        .expect("ack request");
        let ack_route = [0xc4; 16];
        let (_, sealed_ack) = dispatch_cross_entry_terminal(
            &entry_r,
            restarted.clone(),
            &target,
            descriptor_commitment,
            ack_route,
            AnonymousMailboxTerminalFrameV1::Ack(ack),
        );
        let AnonymousMailboxTerminalFrameV1::AckResponse(ack_response) =
            complete_cross_entry_response(&entry_r, ack_route, &sealed_ack)
        else {
            panic!("ack response kind");
        };
        assert_eq!(ack_response.outcome, AnonymousMailboxOutcomeV1::Accepted);

        let empty_pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xc5; 16],
            Vec::new(),
            1_800_000_003,
            &reader,
        )
        .expect("empty pull request");
        let empty_route = [0xc6; 16];
        let (_, sealed_empty) = dispatch_cross_entry_terminal(
            &entry_r,
            restarted,
            &target,
            descriptor_commitment,
            empty_route,
            AnonymousMailboxTerminalFrameV1::PullOne(empty_pull),
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(empty_response) =
            complete_cross_entry_response(&entry_r, empty_route, &sealed_empty)
        else {
            panic!("empty response kind");
        };
        assert_eq!(empty_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(empty_response.sealed_payload.is_empty());

        assert_eq!(m_resolver.wrong_target_calls.load(Ordering::Relaxed), 0);
        assert_eq!(r_resolver.wrong_target_calls.load(Ordering::Relaxed), 0);
        assert!(m_resolver.exact_calls.load(Ordering::Relaxed) >= 6);
        assert!(r_resolver.exact_calls.load(Ordering::Relaxed) >= 6);
    }
}
