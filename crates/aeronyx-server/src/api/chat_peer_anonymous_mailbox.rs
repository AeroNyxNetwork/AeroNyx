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
//! v1.0.4-NoSocketSmtr — Prove exact pinned-target S/M/T/R retries and restart.
//! v1.0.3-LeaseReplay — Preserve durable exact lease-create replay after
//! admission-ticket freshness expires.
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
                // [M13J 2026-09-05 by Codex] The repository owns the typed
                // exact-replay boundary: it checks the durable full request
                // commitment before freshness, while a miss is fully target,
                // signature, claims, and time verified before any mutation.
                // Repeating freshness here would reject a durable Accepted
                // lease after its one-time admission ticket expires.
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
        AnonymousMailboxSourceCoordinator, AnonymousMailboxSourceError,
        AnonymousMailboxSourceOutbound, AnonymousMailboxSourceResult,
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
    use aeronyx_core::protocol::anonymous_mailbox_deposit_invitation::{
        AnonymousMailboxDepositInvitationV2, AnonymousMailboxDepositTargetPinV1,
        DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
    };
    use aeronyx_core::protocol::anonymous_mailbox_recipient_seal::{
        AnonymousMailboxRecipientSealKeyHandleV1, AnonymousMailboxRecipientSealPublicV1,
    };
    use aeronyx_core::protocol::chat::{ChatContentType, ChatEnvelope};
    use aeronyx_core::protocol::discovery::{
        DirectoryDescriptorCommitmentV1, NodeCapability, NodeDescriptor, NodeProtocolFeature,
        SignedNodeDescriptor,
    };
    use aeronyx_core::protocol::memchain::encode_memchain;
    use aeronyx_core::protocol::onion::open_onion_layer;
    use rusqlite::Connection;
    use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
    use zeroize::Zeroizing;

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

    // [ANONYMOUS-MAILBOX-SMTR-HARNESS 2026-09-14 by Codex] A deterministic
    // no-socket transport accepts exactly one pinned terminal. Counters prove
    // the harness neither fans out nor attempts an alternate on rejected
    // descriptor, target, commitment, or carrier inputs.
    struct CrossEntryNoSocketTransport {
        target: IdentityKeyPair,
        exact_calls: AtomicUsize,
        alternate_calls: AtomicUsize,
    }

    impl CrossEntryNoSocketTransport {
        fn new(target: &IdentityKeyPair) -> Self {
            Self {
                target: target.clone(),
                exact_calls: AtomicUsize::new(0),
                alternate_calls: AtomicUsize::new(0),
            }
        }

        fn deliver(
            &self,
            outbound: &AnonymousMailboxSourceOutbound,
            repository: Arc<dyn AnonymousMailboxCustodyRepository>,
            now: u64,
        ) -> Vec<u8> {
            if outbound.target_node_id() != &self.target.public_key_bytes() {
                self.alternate_calls.fetch_add(1, Ordering::Relaxed);
                panic!("no-socket transport received a non-pinned terminal");
            }
            self.exact_calls.fetch_add(1, Ordering::Relaxed);
            let (target_kem_secret, _) = self.target.to_x25519();
            let peel = open_onion_layer(
                &outbound.request().envelope.encrypted_blob,
                &target_kem_secret,
            )
            .expect("target peels one-hop source route");
            assert!(peel.next_hop.is_none());
            let sealed = PreparedAnonymousMailboxTerminal::decode(
                &peel.inner,
                outbound.route_id(),
                self.target.public_key_bytes(),
            )
            .expect("target decodes source carrier")
            .execute(repository, Arc::new(self.target.clone()), now)
            .expect("target executes terminal request");
            BASE64.decode(sealed).expect("source-sealed response")
        }
    }

    struct TestRecipientKeyHandle {
        key_id: [u8; 16],
        secret: StaticSecret,
        public: [u8; 32],
    }

    impl TestRecipientKeyHandle {
        fn new(key_id: [u8; 16], secret_bytes: [u8; 32]) -> Self {
            let secret = StaticSecret::from(secret_bytes);
            let public = X25519PublicKey::from(&secret).to_bytes();
            Self {
                key_id,
                secret,
                public,
            }
        }
    }

    impl AnonymousMailboxRecipientSealKeyHandleV1 for TestRecipientKeyHandle {
        fn key_id(&self) -> [u8; 16] {
            self.key_id
        }

        fn public_key(&self) -> [u8; 32] {
            self.public
        }

        fn derive_shared_secret(
            &self,
            peer_public_key: [u8; 32],
        ) -> Result<
            Zeroizing<[u8; 32]>,
            aeronyx_core::protocol::anonymous_mailbox_recipient_seal::AnonymousMailboxRecipientSealError,
        >{
            Ok(Zeroizing::new(
                *self
                    .secret
                    .diffie_hellman(&X25519PublicKey::from(peer_public_key))
                    .as_bytes(),
            ))
        }
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
        execute_terminal_frame_at(repository, target, route_id, request, 1_800_000_000)
    }

    fn execute_terminal_frame_at(
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        target: &IdentityKeyPair,
        route_id: [u8; 16],
        request: AnonymousMailboxTerminalFrameV1,
        now: u64,
    ) -> AnonymousMailboxTerminalFrameV1 {
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&request).expect("terminal frame");
        let (encoded, mut session) = routed_terminal_frame(route_id, target, terminal_frame);
        let sealed =
            PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, target.public_key_bytes())
                .expect("canonical terminal request")
                .execute(repository, Arc::new(target.clone()), now)
                .expect("terminal response");
        let sealed = BASE64.decode(sealed).expect("base64 AMSR");
        decode_anonymous_mailbox_terminal_frame(&session.open(&sealed).expect("one-shot AMSR open"))
            .expect("canonical response frame")
    }

    fn durable_admission_state(db_path: &str) -> (i64, i64, i64, i64, i64) {
        let connection = Connection::open(db_path).expect("open durable state for audit");
        let (leases, outstanding, issues): (i64, i64, i64) = connection
            .query_row(
                "SELECT total_leases, outstanding_tickets, issues_in_window
                   FROM anonymous_mailbox_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("durable counters");
        let consumed: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM anonymous_mailbox_tickets",
                [],
                |row| row.get(0),
            )
            .expect("consumed ticket count");
        let issued_consumed: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM anonymous_mailbox_issued_tickets
                  WHERE consumed_at IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .expect("issuer token count");
        (leases, outstanding, issues, consumed, issued_consumed)
    }

    fn durable_item_state(db_path: &str) -> (i64, i64, i64, Vec<Vec<u8>>) {
        let connection = Connection::open(db_path).expect("open durable item state for audit");
        let (total_items, total_bytes): (i64, i64) = connection
            .query_row(
                "SELECT total_items, total_bytes
                 FROM anonymous_mailbox_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("durable item counters");
        let row_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM anonymous_mailbox_items", [], |row| {
                row.get(0)
            })
            .expect("durable item row count");
        let mut statement = connection
            .prepare("SELECT sealed_envelope FROM anonymous_mailbox_items ORDER BY sequence")
            .expect("prepare opaque item audit");
        let items = statement
            .query_map([], |row| row.get(0))
            .expect("query opaque items")
            .collect::<Result<Vec<Vec<u8>>, _>>()
            .expect("materialize bounded test rows");
        (total_items, total_bytes, row_count, items)
    }

    fn execute_terminal_result_at(
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        target: &IdentityKeyPair,
        route_id: [u8; 16],
        request: AnonymousMailboxTerminalFrameV1,
        now: u64,
    ) -> Result<String, AnonymousMailboxTerminalFailure> {
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&request).expect("terminal frame");
        let (encoded, _session) = routed_terminal_frame(route_id, target, terminal_frame);
        PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, target.public_key_bytes())?
            .execute(repository, Arc::new(target.clone()), now)
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
        transport: &CrossEntryNoSocketTransport,
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
        let sealed = transport.deliver(&outbound, repository, 1_800_000_000);
        (outbound.body().to_vec(), sealed)
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
    fn lease_exact_replay_after_ticket_expiry_survives_restart_without_mutation() {
        // [M13J 2026-09-05 by Codex] A lost Accepted response must not turn a
        // durable lease into a terminal rejection merely because its one-time
        // admission ticket expired before the source retried.  The fresh
        // response time attests the replay lookup; it is not the original
        // acceptance time.
        const NOW: u64 = 1_800_100_000;
        const RETRY_NOW: u64 = NOW + 301;
        let directory = tempfile::tempdir().expect("private temporary directory");
        let private_directory = std::fs::canonicalize(directory.path()).expect("canonical path");
        let config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: private_directory
                .join("lease-replay.sqlite")
                .display()
                .to_string(),
            max_leases_total: 4,
            max_items_total: 1_024,
            max_bytes_total: 64 * 1024,
            max_in_flight: 4,
            cleanup_batch_size: 8,
            max_outstanding_tickets: 4,
            max_ticket_issues_per_window: 4,
            ticket_issuance_window_secs: 60,
            ticket_issue_work_bits: 1,
        };
        let target = IdentityKeyPair::from_bytes(&[0xc1; 32]).expect("target");
        let depositor = IdentityKeyPair::from_bytes(&[0xc2; 32]).expect("depositor");
        let reader = IdentityKeyPair::from_bytes(&[0xc3; 32]).expect("reader");
        let mailbox_id = [0xc4; 32];
        let lease_expires_at = NOW + 3_600;
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            4,
            16 * 1024,
            NOW,
            lease_expires_at,
        );
        let ticket_request = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    [0xc5; 16],
                    [0xc6; 16],
                    target.public_key_bytes(),
                    claims,
                    NOW,
                    NOW + 300,
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
                [0xc7; 32],
            )
            .expect("open real store"),
        );
        let ticket_response = execute_terminal_frame_at(
            store.clone(),
            &target,
            [0xc8; 16],
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
            NOW,
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket_response) = ticket_response
        else {
            panic!("ticket response kind");
        };
        let ticket = ticket_response.ticket.expect("issued ticket");
        let lease = AnonymousMailboxLeaseCreateV1::new(
            mailbox_id,
            depositor.public_key_bytes(),
            4,
            16 * 1024,
            NOW,
            lease_expires_at,
            ticket,
            &reader,
        )
        .expect("lease request");
        let first = execute_terminal_frame_at(
            store.clone(),
            &target,
            [0xc9; 16],
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            NOW,
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(first) = first else {
            panic!("lease response kind");
        };
        assert_eq!(first.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_eq!(first.responded_at, NOW);
        drop(store); // The source lost `first`; restart from only durable state.

        let accepted_state = durable_admission_state(&config.db_path);
        assert_eq!(accepted_state, (1, 0, 1, 1, 1));
        let reopened = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                [0xc7; 32],
            )
            .expect("restart real store"),
        );
        let replay = execute_terminal_frame_at(
            reopened.clone(),
            &target,
            [0xca; 16],
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            RETRY_NOW,
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(replay) = replay else {
            panic!("lease replay response kind");
        };
        assert_eq!(replay.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_eq!(replay.responded_at, RETRY_NOW);

        let mut changed_bytes = lease.clone();
        changed_bytes.signature[0] ^= 0x80;
        let changed = execute_terminal_frame_at(
            reopened.clone(),
            &target,
            [0xcb; 16],
            AnonymousMailboxTerminalFrameV1::LeaseCreate(changed_bytes),
            RETRY_NOW,
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(changed) = changed else {
            panic!("changed lease response kind");
        };
        assert_eq!(changed.outcome, AnonymousMailboxOutcomeV1::Conflict);

        let wrong_reader = IdentityKeyPair::from_bytes(&[0xcc; 32]).expect("wrong reader");
        let mut wrong_key = lease.clone();
        wrong_key.read_verifier = wrong_reader.public_key_bytes();
        wrong_key.signature =
            wrong_reader.sign(&wrong_key.signing_bytes().expect("wrong-key body"));
        let wrong_key_response = execute_terminal_frame_at(
            reopened.clone(),
            &target,
            [0xcd; 16],
            AnonymousMailboxTerminalFrameV1::LeaseCreate(wrong_key),
            RETRY_NOW,
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(wrong_key_response) =
            wrong_key_response
        else {
            panic!("wrong-key response kind");
        };
        assert_eq!(
            wrong_key_response.outcome,
            AnonymousMailboxOutcomeV1::Conflict
        );

        let stale_mailbox = [0xce; 32];
        let stale_claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &stale_mailbox,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            1,
            1024,
            NOW,
            lease_expires_at,
        );
        let stale_ticket = AnonymousMailboxAdmissionTicketV1::issue(
            [0xcf; 16],
            stale_claims,
            NOW,
            NOW + 300,
            &target,
        )
        .expect("stale ticket fixture");
        let stale_miss = AnonymousMailboxLeaseCreateV1::new(
            stale_mailbox,
            depositor.public_key_bytes(),
            1,
            1024,
            NOW,
            lease_expires_at,
            stale_ticket,
            &reader,
        )
        .expect("stale miss fixture");
        assert_eq!(
            execute_terminal_result_at(
                reopened.clone(),
                &target,
                [0xd0; 16],
                AnonymousMailboxTerminalFrameV1::LeaseCreate(stale_miss),
                RETRY_NOW,
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        );

        let foreign_target = IdentityKeyPair::from_bytes(&[0xd1; 32]).expect("foreign target");
        let foreign_mailbox = [0xd2; 32];
        let foreign_claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &foreign_mailbox,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            1,
            1024,
            NOW,
            lease_expires_at,
        );
        let foreign_ticket = AnonymousMailboxAdmissionTicketV1::issue(
            [0xd3; 16],
            foreign_claims,
            NOW,
            NOW + 300,
            &foreign_target,
        )
        .expect("foreign ticket fixture");
        let wrong_target = AnonymousMailboxLeaseCreateV1::new(
            foreign_mailbox,
            depositor.public_key_bytes(),
            1,
            1024,
            NOW,
            lease_expires_at,
            foreign_ticket,
            &reader,
        )
        .expect("wrong-target lease fixture");
        assert_eq!(
            execute_terminal_result_at(
                reopened.clone(),
                &target,
                [0xd4; 16],
                AnonymousMailboxTerminalFrameV1::LeaseCreate(wrong_target),
                NOW + 1,
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        );
        drop(reopened);
        assert_eq!(durable_admission_state(&config.db_path), accepted_state);
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
        let transport = CrossEntryNoSocketTransport::new(&target);
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

        let empty_admission_state = durable_admission_state(&store_config.db_path);
        let empty_item_state = durable_item_state(&store_config.db_path);
        let ticket_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
        )
        .expect("canonical ticket terminal frame");
        let mut wrong_descriptor = descriptor_commitment;
        wrong_descriptor.descriptor_hash[0] ^= 0x80;
        assert!(matches!(
            entry_m.prepare(
                ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), wrong_descriptor,),
                [0x21; 16],
                ticket_frame.clone(),
                1_800_000_000,
            ),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
        let foreign_target = IdentityKeyPair::from_bytes(&[0x22; 32]).expect("non-pinned terminal");
        assert!(matches!(
            entry_m.prepare(
                ExactAnonymousMailboxTargetPin::new(
                    foreign_target.public_key_bytes(),
                    descriptor_commitment,
                ),
                [0x23; 16],
                ticket_frame.clone(),
                1_800_000_000,
            ),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
        let (canonical_ticket_route, _unused_session) =
            routed_terminal_frame([0x24; 16], &target, ticket_frame);
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &canonical_ticket_route,
                [0x24; 16],
                foreign_target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        let mut tampered_ticket_route = canonical_ticket_route;
        tampered_ticket_route.push(0xff);
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &tampered_ticket_route,
                [0x24; 16],
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert_eq!(
            durable_admission_state(&store_config.db_path),
            empty_admission_state
        );
        assert_eq!(durable_item_state(&store_config.db_path), empty_item_state);
        assert_eq!(transport.exact_calls.load(Ordering::Relaxed), 0);
        assert_eq!(transport.alternate_calls.load(Ordering::Relaxed), 0);

        let ticket_route = [0xbc; 16];
        let (_, sealed_ticket) = dispatch_cross_entry_terminal(
            &entry_m,
            &transport,
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
            &transport,
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
        let (put_body, _lost_sealed_put) = dispatch_cross_entry_terminal(
            &entry_m,
            &transport,
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
        let sealed_put = transport.deliver(&replay, store.clone(), 1_800_000_001);
        let AnonymousMailboxTerminalFrameV1::PutResponse(put_response) =
            complete_cross_entry_response(&entry_m, put_route, &sealed_put)
        else {
            panic!("put response kind");
        };
        assert_eq!(put_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let stored_once = (
            1,
            i64::try_from(opaque_item.len()).expect("opaque item length"),
            1,
            vec![opaque_item.clone()],
        );
        assert_eq!(durable_item_state(&store_config.db_path), stored_once);

        let changed_put = AnonymousMailboxPutV1::new(
            mailbox_id,
            put.item_id,
            vec![0x25; 4096],
            1_800_000_000,
            1_800_000_600,
            &depositor,
        )
        .expect("same-id different-content Put");
        let conflict_route = [0x26; 16];
        let (_, sealed_conflict) = dispatch_cross_entry_terminal(
            &entry_m,
            &transport,
            store.clone(),
            &target,
            descriptor_commitment,
            conflict_route,
            AnonymousMailboxTerminalFrameV1::Put(changed_put.clone()),
        );
        let AnonymousMailboxTerminalFrameV1::PutResponse(conflict_response) =
            complete_cross_entry_response(&entry_m, conflict_route, &sealed_conflict)
        else {
            panic!("conflict response kind");
        };
        assert_eq!(
            conflict_response.outcome,
            AnonymousMailboxOutcomeV1::Conflict
        );
        let changed_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Put(changed_put),
        )
        .expect("changed Put terminal frame");
        let deliveries_before_commitment_conflict = transport.exact_calls.load(Ordering::Relaxed);
        assert!(matches!(
            entry_m.prepare(
                ExactAnonymousMailboxTargetPin::new(
                    target.public_key_bytes(),
                    descriptor_commitment,
                ),
                put_route,
                changed_frame,
                1_800_000_001,
            ),
            Err(AnonymousMailboxSourceError::Conflict)
        ));
        assert_eq!(
            transport.exact_calls.load(Ordering::Relaxed),
            deliveries_before_commitment_conflict,
            "source commitment conflict cannot reach transport"
        );
        assert_eq!(durable_item_state(&store_config.db_path), stored_once);
        drop(store);

        let restarted = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config.clone(),
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
            &transport,
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
        let (_, _lost_sealed_ack) = dispatch_cross_entry_terminal(
            &entry_r,
            &transport,
            restarted.clone(),
            &target,
            descriptor_commitment,
            ack_route,
            AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
        );
        assert_eq!(
            durable_item_state(&store_config.db_path),
            (0, 0, 0, Vec::new())
        );
        let ack_replay = entry_r
            .begin_dispatch(ack_route, 1_800_000_003)
            .expect("exact Ack replay after lost response");
        let sealed_ack = transport.deliver(&ack_replay, restarted.clone(), 1_800_000_003);
        let AnonymousMailboxTerminalFrameV1::AckResponse(ack_response) =
            complete_cross_entry_response(&entry_r, ack_route, &sealed_ack)
        else {
            panic!("ack response kind");
        };
        assert_eq!(ack_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_eq!(
            durable_item_state(&store_config.db_path),
            (0, 0, 0, Vec::new())
        );
        drop(restarted);

        let restarted_after_ack = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                store_config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("restart target after exact Ack replay"),
        );

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
            &transport,
            restarted_after_ack,
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
        assert_eq!(
            durable_item_state(&store_config.db_path),
            (0, 0, 0, Vec::new())
        );

        assert_eq!(m_resolver.wrong_target_calls.load(Ordering::Relaxed), 1);
        assert_eq!(r_resolver.wrong_target_calls.load(Ordering::Relaxed), 0);
        assert_eq!(transport.exact_calls.load(Ordering::Relaxed), 9);
        assert_eq!(transport.alternate_calls.load(Ordering::Relaxed), 0);
        assert_eq!(m_resolver.exact_calls.load(Ordering::Relaxed), 10);
        assert_eq!(r_resolver.exact_calls.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn recipient_sealed_chat_survives_real_terminal_store_restart_and_expired_invitation() {
        // [ANONYMOUS-MAILBOX-RECIPIENT-SEAL-VERTICAL 2026-09-08 by Codex]
        // Exercise the production terminal adapter and SQLite custody store
        // with opaque AMSI bytes. The in-memory key handle exists only on the
        // simulated recipient side; T never receives a chat identity or key.
        const NOW: u64 = 1_800_010_000;
        const INVITATION_EXPIRES: u64 = NOW + 300;
        const LEASE_EXPIRES: u64 = NOW + 1_800;
        const ITEM_EXPIRES: u64 = NOW + 1_200;

        let directory = tempfile::tempdir().expect("private temporary directory");
        let private_directory = std::fs::canonicalize(directory.path()).expect("canonical path");
        let config = AnonymousMailboxStoreConfig {
            enabled: true,
            db_path: private_directory
                .join("recipient-sealed-mailbox.sqlite")
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
        let target = IdentityKeyPair::from_bytes(&[0xd1; 32]).expect("target T");
        let depositor = IdentityKeyPair::from_bytes(&[0xd2; 32]).expect("deposit capability");
        let reader = IdentityKeyPair::from_bytes(&[0xd3; 32]).expect("mailbox reader");
        let chat_receiver =
            IdentityKeyPair::from_bytes(&[0xd4; 32]).expect("authenticated chat receiver");
        let chat_sender = IdentityKeyPair::from_bytes(&[0xd5; 32]).expect("chat sender");
        let recipient_secret = [0xd6; 32];
        let recipient_key_id = [0xd7; 16];
        let recipient_handle = TestRecipientKeyHandle::new(recipient_key_id, recipient_secret);
        let recipient_public =
            AnonymousMailboxRecipientSealPublicV1::new(recipient_key_id, recipient_handle.public)
                .expect("recipient public seal capability");
        assert_ne!(reader.public_key_bytes(), chat_receiver.public_key_bytes());
        assert_ne!(reader.public_key_bytes(), recipient_handle.public);
        assert_ne!(chat_receiver.public_key_bytes(), recipient_handle.public);

        let mailbox_id = [0xd8; 32];
        let ticket_id = [0xd9; 16];
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            4,
            512 * 1024,
            NOW,
            LEASE_EXPIRES,
        );
        let ticket_request = (0..u64::MAX)
            .find_map(|proof_nonce| {
                let request = AnonymousMailboxTicketIssueV1::new(
                    [0xda; 16],
                    ticket_id,
                    target.public_key_bytes(),
                    claims,
                    NOW,
                    NOW + 300,
                    proof_nonce,
                )
                .expect("ticket request");
                (request.proof_digest().expect("proof digest")[0] & 0x80 == 0).then_some(request)
            })
            .expect("one-bit proof");
        let cursor_secret = [0xdb; 32];
        let store = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("open real target store"),
        );
        let ticket_response = execute_terminal_frame_at(
            store.clone(),
            &target,
            [0xdc; 16],
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()),
            NOW,
        );
        let AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket_response) = ticket_response
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
            512 * 1024,
            NOW,
            LEASE_EXPIRES,
            ticket,
            &reader,
        )
        .expect("lease request");
        let lease_response = execute_terminal_frame_at(
            store.clone(),
            &target,
            [0xdd; 16],
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            NOW,
        );
        let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(lease_response) = lease_response
        else {
            panic!("lease response kind");
        };
        assert_eq!(lease_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let lease_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
        )
        .expect("canonical lease request frame");
        let lease_response_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(lease_response),
        )
        .expect("canonical lease response frame");
        let target_pin =
            AnonymousMailboxDepositTargetPinV1::new(target.public_key_bytes(), 17, [0xde; 32])
                .expect("exact target pin");
        let invitation_a = AnonymousMailboxDepositInvitationV2::issue(
            [0xdf; 16],
            NOW,
            INVITATION_EXPIRES,
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
            target_pin,
            depositor.to_bytes(),
            chat_receiver.public_key_bytes(),
            recipient_public,
            lease_frame.clone(),
            lease_response_frame.clone(),
            &reader,
        )
        .expect("signed invitation A");
        let invitation_b = AnonymousMailboxDepositInvitationV2::issue(
            [0xe0; 16],
            NOW,
            INVITATION_EXPIRES,
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
            target_pin,
            depositor.to_bytes(),
            chat_receiver.public_key_bytes(),
            recipient_public,
            lease_frame,
            lease_response_frame,
            &reader,
        )
        .expect("signed invitation B");
        let invitation_a_bytes = invitation_a.encode().expect("durable invitation A");
        let active_a = invitation_a
            .verify_for_new_put_at(NOW)
            .expect("active invitation A");
        let active_b = invitation_b
            .verify_for_new_put_at(NOW)
            .expect("active invitation B");
        let mut chat = ChatEnvelope {
            message_id: [0xe1; 16],
            sender: chat_sender.public_key_bytes(),
            receiver: chat_receiver.public_key_bytes(),
            timestamp: NOW,
            ciphertext: b"recipient-only signed chat ciphertext".to_vec(),
            nonce: [0xe2; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        chat.signature = chat_sender.sign(&chat.sign_data());

        let sealed_for_wrong_invitation = active_a
            .seal_chat_envelope_at(&chat, NOW)
            .expect("A seals signed chat");
        assert!(active_b
            .prepare_put(
                [0xe3; 16],
                sealed_for_wrong_invitation,
                NOW,
                ITEM_EXPIRES,
                NOW,
            )
            .is_err());
        assert_eq!(durable_item_state(&config.db_path), (0, 0, 0, Vec::new()));

        let sealed = active_a
            .seal_chat_envelope_at(&chat, NOW)
            .expect("A seals signed chat for Put");
        let sealed_bytes = sealed.as_bytes().to_vec();
        let put = active_a
            .prepare_put([0xe4; 16], sealed, NOW, ITEM_EXPIRES, NOW)
            .expect("A prepares bound Put");
        let put_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Put(put.clone()),
        )
        .expect("canonical Put terminal frame");
        let route_id = [0xe5; 16];
        let (prepared_bytes, mut source_session) =
            routed_terminal_frame(route_id, &target, put_frame);
        let wrong_target =
            IdentityKeyPair::from_bytes(&[0xe6; 32]).expect("different terminal target");
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &prepared_bytes,
                [0xe7; 16],
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &prepared_bytes,
                route_id,
                wrong_target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert_eq!(durable_item_state(&config.db_path), (0, 0, 0, Vec::new()));

        let _lost_response = PreparedAnonymousMailboxTerminal::decode(
            &prepared_bytes,
            route_id,
            target.public_key_bytes(),
        )
        .expect("first exact prepared request")
        .execute(store.clone(), Arc::new(target.clone()), NOW)
        .expect("first target Put");
        let exact_replay_response = PreparedAnonymousMailboxTerminal::decode(
            &prepared_bytes,
            route_id,
            target.public_key_bytes(),
        )
        .expect("byte-exact prepared retry")
        .execute(store.clone(), Arc::new(target.clone()), NOW + 1)
        .expect("idempotent target Put retry");
        let exact_replay_response = BASE64
            .decode(exact_replay_response)
            .expect("base64 exact-retry response");
        let exact_replay_response = decode_anonymous_mailbox_terminal_frame(
            &source_session
                .open(&exact_replay_response)
                .expect("open exact-retry response once"),
        )
        .expect("canonical Put response");
        let AnonymousMailboxTerminalFrameV1::PutResponse(exact_replay_response) =
            exact_replay_response
        else {
            panic!("Put response kind");
        };
        assert_eq!(
            exact_replay_response.outcome,
            AnonymousMailboxOutcomeV1::Accepted
        );
        assert_eq!(
            durable_item_state(&config.db_path),
            (
                1,
                i64::try_from(sealed_bytes.len()).expect("opaque byte count"),
                1,
                vec![sealed_bytes.clone()]
            ),
            "T persists one opaque AMSI row for an exact prepared retry"
        );

        assert!(active_a
            .seal_chat_envelope_at(&chat, INVITATION_EXPIRES + 1)
            .is_err());
        let sealed_before_expiry = active_a
            .seal_chat_envelope_at(&chat, NOW)
            .expect("seal while invitation is current");
        assert!(active_a
            .prepare_put(
                [0xe8; 16],
                sealed_before_expiry,
                NOW,
                ITEM_EXPIRES,
                INVITATION_EXPIRES + 1,
            )
            .is_err());
        drop(store);

        let restarted = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("restart target store"),
        );
        let pull_now = INVITATION_EXPIRES + 1;
        let pull =
            AnonymousMailboxPullOneV1::new(mailbox_id, [0xe9; 16], Vec::new(), pull_now, &reader)
                .expect("pull after invitation expiry");
        let pull_response = execute_terminal_frame_at(
            restarted.clone(),
            &target,
            [0xea; 16],
            AnonymousMailboxTerminalFrameV1::PullOne(pull),
            pull_now,
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(pull_response) = pull_response else {
            panic!("Pull response kind");
        };
        assert_eq!(pull_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        let pulled = AnonymousMailboxPullResultV1::decode(&pull_response.sealed_payload)
            .expect("canonical opaque pull result");
        assert_eq!(pulled.item_id, put.item_id);
        assert_eq!(pulled.sealed_commitment, put.sealed_commitment());
        assert_eq!(pulled.sealed_item, sealed_bytes);

        // Simulate recipient restart: rebuild only the native key handle and
        // historical open context. Invitation expiry cannot revoke retained
        // ciphertext that remains inside the lease/item lifetime.
        let historical = AnonymousMailboxDepositInvitationV2::decode_historical_recipient_context(
            &invitation_a_bytes,
        )
        .expect("historical open context");
        let restarted_handle = TestRecipientKeyHandle::new(recipient_key_id, recipient_secret);
        let opened = historical
            .open_chat_envelope(&restarted_handle, &pulled.sealed_item)
            .expect("open retained signed chat after invitation expiry");
        assert_eq!(opened.message_id, chat.message_id);
        assert_eq!(opened.sender, chat_sender.public_key_bytes());
        assert_eq!(opened.receiver, chat_receiver.public_key_bytes());
        assert_eq!(opened.timestamp, chat.timestamp);
        assert_eq!(opened.ciphertext, chat.ciphertext);
        assert_eq!(opened.nonce, chat.nonce);
        assert_eq!(opened.content_type, chat.content_type);
        assert_eq!(opened.signature, chat.signature);

        // [ANONYMOUS-MAILBOX-RECIPIENT-ACK-FAULT 2026-09-08 by Codex]
        // A canonical ACK signed by a different reader and a correctly signed
        // ACK for a different sealed commitment must both preserve the exact
        // opaque AMSI row and its byte counters. T never opens the item.
        let retained_state = (
            1,
            i64::try_from(sealed_bytes.len()).expect("opaque retained byte count"),
            1,
            vec![sealed_bytes.clone()],
        );
        let wrong_reader = IdentityKeyPair::from_bytes(&[0xef; 32]).expect("wrong read key");
        let wrong_reader_ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xf0; 16],
            pulled.item_id,
            pulled.sealed_commitment,
            pull_now + 1,
            &wrong_reader,
        )
        .expect("canonical wrong-reader Ack");
        assert!(matches!(
            execute_terminal_result_at(
                restarted.clone(),
                &target,
                [0xf1; 16],
                AnonymousMailboxTerminalFrameV1::Ack(wrong_reader_ack),
                pull_now + 1,
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert_eq!(durable_item_state(&config.db_path), retained_state);

        let mut wrong_commitment = pulled.sealed_commitment;
        wrong_commitment[0] ^= 0x80;
        let wrong_commitment_ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xf2; 16],
            pulled.item_id,
            wrong_commitment,
            pull_now + 1,
            &reader,
        )
        .expect("canonical wrong-commitment Ack");
        let wrong_commitment_response = execute_terminal_frame_at(
            restarted.clone(),
            &target,
            [0xf3; 16],
            AnonymousMailboxTerminalFrameV1::Ack(wrong_commitment_ack),
            pull_now + 1,
        );
        let AnonymousMailboxTerminalFrameV1::AckResponse(wrong_commitment_response) =
            wrong_commitment_response
        else {
            panic!("wrong-commitment Ack response kind");
        };
        assert_eq!(
            wrong_commitment_response.outcome,
            AnonymousMailboxOutcomeV1::Conflict
        );
        assert_eq!(durable_item_state(&config.db_path), retained_state);
        drop(restarted);

        let restarted_after_rejection = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("restart after rejected Acks"),
        );
        let preserved_pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xf4; 16],
            Vec::new(),
            pull_now + 2,
            &reader,
        )
        .expect("pull exact AMSI after rejected Acks");
        let preserved_pull_response = execute_terminal_frame_at(
            restarted_after_rejection.clone(),
            &target,
            [0xf5; 16],
            AnonymousMailboxTerminalFrameV1::PullOne(preserved_pull),
            pull_now + 2,
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(preserved_pull_response) =
            preserved_pull_response
        else {
            panic!("preserved Pull response kind");
        };
        assert_eq!(
            preserved_pull_response.outcome,
            AnonymousMailboxOutcomeV1::Accepted
        );
        let preserved =
            AnonymousMailboxPullResultV1::decode(&preserved_pull_response.sealed_payload)
                .expect("preserved canonical opaque Pull result");
        assert_eq!(preserved.item_id, pulled.item_id);
        assert_eq!(preserved.sealed_commitment, pulled.sealed_commitment);
        assert_eq!(preserved.sealed_item, sealed_bytes);

        let ack_now = pull_now + 3;
        let ack = AnonymousMailboxAckV1::new(
            mailbox_id,
            [0xeb; 16],
            preserved.item_id,
            preserved.sealed_commitment,
            ack_now,
            &reader,
        )
        .expect("ack retained AMSI");
        let ack_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
        )
        .expect("canonical Ack terminal frame");
        let ack_route = [0xec; 16];
        let (ack_bytes, lost_ack_session) = routed_terminal_frame(ack_route, &target, ack_frame);
        let lost_ack_response = PreparedAnonymousMailboxTerminal::decode(
            &ack_bytes,
            ack_route,
            target.public_key_bytes(),
        )
        .expect("canonical Ack before response loss")
        .execute(
            restarted_after_rejection.clone(),
            Arc::new(target.clone()),
            ack_now,
        )
        .expect("durable Ack before response loss");
        assert!(!lost_ack_response.is_empty());
        drop(lost_ack_session); // The sealed AMSR is intentionally never opened.
        assert_eq!(durable_item_state(&config.db_path), (0, 0, 0, Vec::new()));
        drop(restarted_after_rejection);

        let restarted_after_lost_ack = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config.clone(),
                target.clone(),
                cursor_secret,
            )
            .expect("restart after lost Ack response"),
        );
        let ack_retry_now = ack_now + 301;
        let ack_response = execute_terminal_frame_at(
            restarted_after_lost_ack.clone(),
            &target,
            [0xf6; 16],
            AnonymousMailboxTerminalFrameV1::Ack(ack),
            ack_retry_now,
        );
        let AnonymousMailboxTerminalFrameV1::AckResponse(ack_response) = ack_response else {
            panic!("exact-retry Ack response kind");
        };
        assert_eq!(ack_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert_eq!(
            durable_item_state(&config.db_path),
            (0, 0, 0, Vec::new()),
            "an exact Ack replay after restart cannot revive the item"
        );
        drop(restarted_after_lost_ack);

        let restarted_empty = Arc::new(
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config,
                target.clone(),
                cursor_secret,
            )
            .expect("restart empty target store"),
        );
        let empty = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            [0xed; 16],
            Vec::new(),
            ack_retry_now + 1,
            &reader,
        )
        .expect("empty pull after Ack restart");
        let empty_response = execute_terminal_frame_at(
            restarted_empty,
            &target,
            [0xee; 16],
            AnonymousMailboxTerminalFrameV1::PullOne(empty),
            ack_retry_now + 1,
        );
        let AnonymousMailboxTerminalFrameV1::PullOneResponse(empty_response) = empty_response
        else {
            panic!("empty Pull response kind");
        };
        assert_eq!(empty_response.outcome, AnonymousMailboxOutcomeV1::Accepted);
        assert!(empty_response.sealed_payload.is_empty());
        assert_eq!(
            durable_item_state(
                &private_directory
                    .join("recipient-sealed-mailbox.sqlite")
                    .display()
                    .to_string()
            ),
            (0, 0, 0, Vec::new())
        );
    }
}
