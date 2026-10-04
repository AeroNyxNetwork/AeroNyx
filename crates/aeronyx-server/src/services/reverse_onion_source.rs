// ============================================
// File: crates/aeronyx-server/src/services/reverse_onion_source.rs
// ============================================
//! Private, durable source-side fixed-class Blind Vault Pull journal.
//!
//! [REVERSE-ONION-SOURCE-JOURNAL 2026-10-04 by Codex] No network, startup,
//! route admission or registration. All methods are synchronous; callers own
//! spawn_blocking and shutdown/drain. A cancelled waiter does not undo a CAS.
//! Exact S->R dispatch and expected R->P signing data are DIFFERENT artifacts.
//! Only trusted route construction may supply the latter before dispatch.
//! A deadline parameter is a runtime admission assertion, not a signature.
//!
//! Prepared->Armed commits before dispatch bytes escape. Opening commits before
//! restoring a one-use key. Crashed Opening is never reopened. Authenticated
//! cached pages are readable without reopening a reply session. Disk rollback
//! by a hostile same-euid actor is NOT prevented without an external anchor.
//! Evidence is bounded, not eternal: identifiers are protected through their
//! admitted freshness horizon, not forever after all tombstones are deleted.
//! Last Modified: v1.2.0 — Preflight and post-operation observed physical bound.
//! [REVERSE-ONION-SOURCE-DB-BOUNDARY 2026-10-04 by Codex] Not an OS hard quota.
//! v1.1.0 — Typed same-pass forward expectation intake.
//! [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] No raw production builder.

use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame,
    BlindVaultOnionPullSession, BlindVaultPullResponse,
    BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS, MAX_BLIND_VAULT_ONION_PULL_RESTART_BYTES,
};
use aeronyx_core::protocol::blind_vault_replica_workflow::{
    open_blind_vault_source_pull_journal, seal_blind_vault_source_pull_journal,
    MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES,
    MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES,
};
use aeronyx_core::protocol::chat::{encode_blind_relay_envelope, BlindRelayEnvelope};
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, SignedNodeDescriptor,
    SignedPrivateOnionRecipientAuthorizationV1,
    MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES,
};
use aeronyx_core::protocol::onion::{is_onion_blob, VerifiedOnionForwardExpectation, reverse_delivery::{
    ReverseOnionFrameV1, MAX_REVERSE_ONION_FRAME_BYTES,
    REVERSE_ONION_ENVELOPE_LIFETIME_SECS, REVERSE_ONION_RESULT_RETENTION_SECS,
}};
use aeronyx_core::protocol::onion_reply::decode_onion_reply_request;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use crate::api::chat_peer::{
    blind_relay_authenticated_request_commitment, PeerBlindRelayRequest,
};

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::{fs::{FileExt, MetadataExt, OpenOptionsExt}, io::AsRawFd};

const MAX_ENTRIES: usize = 1024;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const PAGE_LIMIT: usize = 64;
// Narrow source Pull admission, not a change to general relay wire ceilings.
const MAX_DISPATCH_BYTES: usize = 64 * 1024;
const MAX_TERMINAL_BYTES: usize = 512;
const MAX_CLAIM_BYTES: usize = 234;
const MAX_LEASE_BYTES: usize = 234 + 256 * 1024;
const MAX_PAGE_BYTES: usize = BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS;
// [REVERSE-ONION-SOURCE-JOURNAL-V2 2026-10-04 by Codex] Sealed v2 rows bind
// the canonical dispatch/body, route authority bytes, and restart metadata;
// historical v1 rows remain readable only as an explicit migration failure.
const MAX_SIGNED_DESCRIPTOR_BYTES: usize = 16 * 1024;
const MAX_AUTHORIZATION_BYTES: usize = MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES;
const SOURCE_RECORD_VERSION_V2: u16 = 2;
const RECORD_FIXED_BYTES: usize = 4 + 2 + 1 + 8 + 8 + 8 + 32 + 16 + 32 * 4
    + 8 + 8 + 1 + 8 + 32 * 2 + 32 * 4 + 11 * 4;
const SEALED_OVERHEAD: usize = MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES
    - MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES;
const ROW_ACCOUNTING_BYTES: usize = 256;
const RESERVED_PER_JOB: u64 =
    (MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES + ROW_ACCOUNTING_BYTES) as u64;
const APPLICATION_ID: i64 = 0x4158504a;
const META_SQL: &str = "CREATE TABLE source_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), source BLOB NOT NULL, clock INTEGER NOT NULL)";
const ROW_SQL: &str = "CREATE TABLE source_jobs (route BLOB PRIMARY KEY NOT NULL, phase INTEGER NOT NULL, generation INTEGER NOT NULL, reserved INTEGER NOT NULL, retain_until INTEGER NOT NULL, sealed BLOB NOT NULL)";

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SourceJournalError {
    #[error("source journal rejected")] Rejected,
    #[error("source journal busy")] Busy,
    #[error("source journal conflict")] Conflict,
    #[error("source journal capacity")] Capacity,
    #[error("source journal expired")] Expired,
    #[error("source journal ambiguous")] Ambiguous,
    #[error("source journal corrupt")] Corrupt,
    #[error("source journal unavailable")] Unavailable,
    #[error("source journal clock rejected")] ClockRollback,
    #[error("source reply rejected")] ReplyRejected,
    #[error("source journal migration required")] MigrationRequired,
}
type Result<T> = std::result::Result<T, SourceJournalError>;

pub(crate) struct SourceJournalLimits {
    pub(crate) max_entries: usize,
    pub(crate) max_bytes: u64,
}

/// Authenticated R/P route evidence retained inside the sealed source row.
/// The purpose is caller-supplied and must be the separately reviewed private
/// BlindVaultPull authorization; this journal does not admit AMST/AMSR.
/// The endpoint is never accepted as an independent caller argument: a
/// recovered transport must revalidate these canonical descriptors against its
/// current PeerStore view before opening a socket.
pub(crate) struct SourceRouteAuthority {
    relay_descriptor: Zeroizing<Vec<u8>>,
    recipient_descriptor: Zeroizing<Vec<u8>>,
    authorization: Zeroizing<Vec<u8>>,
    purpose: Zeroizing<Vec<u8>>,
}

impl SourceRouteAuthority {
    pub(crate) fn from_signed(
        relay: &SignedNodeDescriptor,
        recipient: &SignedNodeDescriptor,
        authorization: &SignedPrivateOnionRecipientAuthorizationV1,
        purpose: &str,
    ) -> Result<Self> {
        let relay_descriptor = relay
            .encode_canonical()
            .map_err(|_| SourceJournalError::Rejected)?;
        let recipient_descriptor = recipient
            .encode_canonical()
            .map_err(|_| SourceJournalError::Rejected)?;
        let authorization = authorization
            .encode_canonical()
            .map_err(|_| SourceJournalError::Rejected)?;
        if purpose != aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull.as_str() {
            return Err(SourceJournalError::Rejected);
        }
        let authority = Self {
            relay_descriptor: Zeroizing::new(relay_descriptor),
            recipient_descriptor: Zeroizing::new(recipient_descriptor),
            authorization: Zeroizing::new(authorization),
            purpose: Zeroizing::new(purpose.as_bytes().to_vec()),
        };
        authority.validate_shape()?;
        Ok(authority)
    }

    fn validate_shape(&self) -> Result<()> {
        if self.relay_descriptor.is_empty()
            || self.relay_descriptor.len() > MAX_SIGNED_DESCRIPTOR_BYTES
            || self.recipient_descriptor.is_empty()
            || self.recipient_descriptor.len() > MAX_SIGNED_DESCRIPTOR_BYTES
            || self.authorization.is_empty()
            || self.authorization.len() > MAX_AUTHORIZATION_BYTES
        {
            return Err(SourceJournalError::Capacity);
        }
        let relay = SignedNodeDescriptor::decode_canonical(&self.relay_descriptor)
            .map_err(|_| SourceJournalError::Rejected)?;
        let recipient = SignedNodeDescriptor::decode_canonical(&self.recipient_descriptor)
            .map_err(|_| SourceJournalError::Rejected)?;
        let _authorization = SignedPrivateOnionRecipientAuthorizationV1::decode_canonical(
            &self.authorization,
        )
        .map_err(|_| SourceJournalError::Rejected)?;
        let purpose = std::str::from_utf8(&self.purpose).map_err(|_| SourceJournalError::Rejected)?;
        if purpose != aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull.as_str() {
            return Err(SourceJournalError::Rejected);
        }
        relay
            .verify_signature()
            .map_err(|_| SourceJournalError::Rejected)?;
        recipient
            .verify_signature()
            .map_err(|_| SourceJournalError::Rejected)?;
        let relay_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&relay)
            .map_err(|_| SourceJournalError::Rejected)?;
        let recipient_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&recipient)
                .map_err(|_| SourceJournalError::Rejected)?;
        if relay_commitment.node_id == [0; 32]
            || relay_commitment.sequence == 0
            || relay_commitment.descriptor_hash == [0; 32]
            || recipient_commitment.node_id == [0; 32]
            || recipient_commitment.sequence == 0
            || recipient_commitment.descriptor_hash == [0; 32]
        {
            return Err(SourceJournalError::Rejected);
        }
        Ok(())
    }

    fn validate_at(
        &self,
        expected_relay: [u8; 32],
        expected_recipient: [u8; 32],
        relay_commitment: [u8; 32],
        recipient_commitment: [u8; 32],
        now: u64,
    ) -> Result<()> {
        self.validate_shape()?;
        let relay = SignedNodeDescriptor::decode_canonical(&self.relay_descriptor)
            .map_err(|_| SourceJournalError::Rejected)?;
        let recipient = SignedNodeDescriptor::decode_canonical(&self.recipient_descriptor)
            .map_err(|_| SourceJournalError::Rejected)?;
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::decode_canonical(
            &self.authorization,
        )
        .map_err(|_| SourceJournalError::Rejected)?;
        let purpose = std::str::from_utf8(&self.purpose).map_err(|_| SourceJournalError::Rejected)?;
        let relay_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&relay)
            .map_err(|_| SourceJournalError::Rejected)?;
        let recipient_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&recipient)
            .map_err(|_| SourceJournalError::Rejected)?;
        if relay.node_id() != expected_relay
            || recipient.node_id() != expected_recipient
            || relay_pin.hash() != relay_commitment
            || recipient_pin.hash() != recipient_commitment
        {
            return Err(SourceJournalError::Rejected);
        }
        authorization
            .verify_at(&relay, &recipient, purpose, now)
            .map_err(|_| SourceJournalError::Rejected)
    }
}

impl std::fmt::Debug for SourceRouteAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourceRouteAuthority")
            .field("relay_descriptor_bytes", &self.relay_descriptor.len())
            .field("recipient_descriptor_bytes", &self.recipient_descriptor.len())
            .field("authorization_bytes", &self.authorization.len())
            .finish_non_exhaustive()
    }
}

/// Private route-construction projection. Never supplied by a response handler.
/// Signature is deliberately absent; the source cannot manufacture R's signature.
pub(crate) struct ExpectedRetainedEnvelope {
    relay: [u8; 32],
    recipient: [u8; 32],
    route: [u8; 16],
    ttl: u8,
    timestamp: u64,
    blob_hash: [u8; 32],
    signing_commitment: [u8; 32],
}

impl ExpectedRetainedEnvelope {
    /// Consumes the projection captured by the verified builder during the
    /// SAME encryption pass as the outbound envelope. No relay secret, raw
    /// inner frame, or post-response expectation derivation is accepted here.
    /// Descriptor/deadline authority still belongs to runtime admission.
    // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex]
    pub(crate) fn from_verified_forward_expectation(
        expectation: &VerifiedOnionForwardExpectation,
    ) -> Result<Self> {
        let value = Self {
            relay: expectation.first_relay_node_id(), recipient: expectation.next_hop_node_id(),
            route: expectation.route_id(), ttl: expectation.ttl(), timestamp: expectation.timestamp(),
            blob_hash: expectation.encrypted_blob_hash(), signing_commitment: expectation.signing_data_commitment(),
        };
        valid_key(value.relay)?; valid_key(value.recipient)?;
        if value.relay == value.recipient || value.relay == [0; 32] || value.recipient == [0; 32]
            || value.route == [0; 16] || value.ttl == 0 || value.timestamp == 0
            || value.timestamp.checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS).is_none()
            || value.blob_hash == [0; 32] || value.signing_commitment == [0; 32]
        { return Err(SourceJournalError::Rejected); }
        Ok(value)
    }

    /// Inputs must be the deterministic R->P projection from trusted source
    /// route construction, not guessed or learned after the source send.
    /// Test-only parity control; not available to any production caller.
    #[cfg(test)]
    fn from_route_construction(
        relay: [u8; 32], recipient: [u8; 32], route: [u8; 16], ttl: u8,
        timestamp: u64, exact_inner_onion: &[u8],
    ) -> Result<Self> {
        valid_key(relay)?;
        valid_key(recipient)?;
        if relay == recipient || route == [0; 16] || ttl == 0
            || exact_inner_onion.len() > 256 * 1024 || !is_onion_blob(exact_inner_onion)
        { return Err(SourceJournalError::Rejected); }
        let projection = BlindRelayEnvelope {
            route_id: route, next_hop: recipient, ttl, timestamp,
            encrypted_blob: exact_inner_onion.to_vec(), signature: [0; 64],
        };
        // This is canonical SIGNING DATA only, never a purported signed frame.
        let signing_commitment = hash(&projection.signing_data());
        Ok(Self { relay, recipient, route, ttl, timestamp,
            blob_hash: hash(exact_inner_onion), signing_commitment })
    }

    fn matches(&self, envelope: &BlindRelayEnvelope) -> bool {
        envelope.route_id == self.route && envelope.next_hop == self.recipient
            && envelope.ttl == self.ttl && envelope.timestamp == self.timestamp
            && hash(&envelope.encrypted_blob) == self.blob_hash
            && hash(&envelope.signing_data()) == self.signing_commitment
    }
}

/// Source-private expectations are encrypted at rest, never placed in relay
/// plaintext. This constructor does not establish descriptor trust or deadline
/// authority; the authenticated runtime admission boundary remains mandatory.
pub(crate) struct SourcePreparedPull {
    source: [u8; 32],
    target: [u8; 32],
    descriptor: [u8; 32],
    relay_descriptor_commitment: [u8; 32],
    recipient_descriptor_commitment: [u8; 32],
    request_commitment: [u8; 32],
    body_commitment: [u8; 32],
    authority: SourceRouteAuthority,
    expected: ExpectedRetainedEnvelope,
    deadline: u64,
    original_expiry: u64,
    dispatch: Vec<u8>,
    terminal: Zeroizing<Vec<u8>>,
}

impl SourcePreparedPull {
    /// `request_commitment` must be the existing authenticated blind-relay
    /// commitment, supplied by the admission owner; this journal never
    /// substitutes a new hash domain for that protocol commitment.
    pub(crate) fn from_runtime_admission(
        source: &IdentityKeyPair, request: PeerBlindRelayRequest,
        expected: ExpectedRetainedEnvelope, target: [u8; 32], descriptor: [u8; 32],
        relay_descriptor_commitment: [u8; 32], recipient_descriptor_commitment: [u8; 32],
        request_commitment: [u8; 32], authority: SourceRouteAuthority,
        admitted_deadline: u64, exact_terminal_request: Vec<u8>,
    ) -> Result<Self> {
        let terminal = Zeroizing::new(exact_terminal_request);
        if terminal.is_empty() || terminal.len() > MAX_TERMINAL_BYTES
            || request.onward_envelope.is_some() || request.onward_descriptor_hint.is_some()
        { return Err(SourceJournalError::Rejected); }
        // Bound the input before JSON allocates its worst-case numeric array.
        if request.envelope.encrypted_blob.len() > MAX_DISPATCH_BYTES / 4 {
            return Err(SourceJournalError::Capacity);
        }
        let original_expiry = request.envelope.timestamp
            .checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS).ok_or(SourceJournalError::Rejected)?;
        let dispatch = serde_json::to_vec(&request).map_err(|_| SourceJournalError::Rejected)?;
        let body_commitment = hash(&dispatch);
        let value = Self { source: source.public_key_bytes(), target, descriptor,
            relay_descriptor_commitment, recipient_descriptor_commitment,
            request_commitment, body_commitment, authority, expected,
            deadline: admitted_deadline, original_expiry, dispatch, terminal };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        valid_key(self.source)?; valid_key(self.target)?;
        valid_key(self.expected.relay)?; valid_key(self.expected.recipient)?;
        if self.dispatch.is_empty() || self.dispatch.len() > MAX_DISPATCH_BYTES
            || self.terminal.is_empty() || self.terminal.len() > MAX_TERMINAL_BYTES
            || self.descriptor == [0; 32]
            || self.relay_descriptor_commitment == [0; 32]
            || self.recipient_descriptor_commitment == [0; 32]
            || self.request_commitment == [0; 32]
            || self.body_commitment == [0; 32]
            || self.body_commitment != hash(&self.dispatch)
            || self.expected.route == [0; 16]
            || self.expected.timestamp == 0 || self.expected.blob_hash == [0; 32]
            || self.expected.signing_commitment == [0; 32]
            || self.expected.ttl == 0 || self.expected.relay == self.expected.recipient
            || self.deadline <= self.expected.timestamp
            || self.deadline > self.expected.timestamp.checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
                .ok_or(SourceJournalError::Rejected)?
            || self.deadline > self.original_expiry
        { return Err(SourceJournalError::Rejected); }
        let request: PeerBlindRelayRequest = serde_json::from_slice(&self.dispatch)
            .map_err(|_| SourceJournalError::Rejected)?;
        let request_commitment = blind_relay_authenticated_request_commitment(&request)
            .map_err(|_| SourceJournalError::Rejected)?;
        if request_commitment != self.request_commitment {
            return Err(SourceJournalError::Rejected);
        }
        if request.previous_hop_node_id != self.source || request.onward_envelope.is_some()
            || request.onward_descriptor_hint.is_some()
            || request.envelope.route_id != self.expected.route
            || request.envelope.next_hop != self.expected.relay || request.envelope.ttl == 0
            || request.envelope.timestamp == 0
            || request.envelope.timestamp != self.expected.timestamp
            || request.envelope.ttl.checked_sub(1) != Some(self.expected.ttl)
            || !is_onion_blob(&request.envelope.encrypted_blob)
            || request.envelope.timestamp >= self.deadline
            || request.envelope.timestamp.checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
                != Some(self.original_expiry)
            || serde_json::to_vec(&request).map_err(|_| SourceJournalError::Rejected)? != self.dispatch
        { return Err(SourceJournalError::Rejected); }
        encode_blind_relay_envelope(&request.envelope).map_err(|_| SourceJournalError::Rejected)?;
        self.authority.validate_shape()?;
        request.envelope.verify_signature_from(&IdentityPublicKey::from_bytes(&self.source)
            .map_err(|_| SourceJournalError::Rejected)?).map_err(|_| SourceJournalError::Rejected)?;
        Ok(())
    }

    fn validate_at(&self, now: u64) -> Result<()> {
        self.validate()?;
        self.authority.validate_at(
            self.expected.relay,
            self.expected.recipient,
            self.relay_descriptor_commitment,
            self.recipient_descriptor_commitment,
            now,
        )?;
        if self.descriptor != self.recipient_descriptor_commitment
            || self.target != self.expected.recipient
        {
            return Err(SourceJournalError::Rejected);
        }
        Ok(())
    }

    fn fresh(&self, now: u64) -> Result<()> {
        // [REVERSE-ONION-SOURCE-HISTORICAL-AUTH 2026-10-04 by Codex] The
        // immutable envelope timestamp is the admission anchor: an
        // authorization issued after that signed envelope must never be
        // admitted only to fail closed on restart. Current-time validation is
        // still required for a new prepare/arm, while restart/grace reads use
        // the historical anchor above without extending the lease.
        self.validate_at(self.expected.timestamp)?;
        self.validate_at(now)?;
        let request: PeerBlindRelayRequest = serde_json::from_slice(&self.dispatch)
            .map_err(|_| SourceJournalError::Corrupt)?;
        if now < self.expected.timestamp || now < request.envelope.timestamp || now >= self.deadline {
            return Err(SourceJournalError::Expired);
        }
        Ok(())
    }

    fn retention(&self) -> Result<u64> {
        let retained_envelope_expiry = self.expected.timestamp
            .checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS).ok_or(SourceJournalError::Rejected)?;
        let result_grace = self.deadline.checked_add(REVERSE_ONION_RESULT_RETENTION_SECS)
            .ok_or(SourceJournalError::Rejected)?;
        Ok(self.original_expiry.max(retained_envelope_expiry).max(result_grace))
    }

    fn same(&self, other: &Self) -> bool {
        self.source == other.source && self.target == other.target && self.descriptor == other.descriptor
            && self.relay_descriptor_commitment == other.relay_descriptor_commitment
            && self.recipient_descriptor_commitment == other.recipient_descriptor_commitment
            && self.request_commitment == other.request_commitment
            && self.body_commitment == other.body_commitment
            && self.authority.relay_descriptor == other.authority.relay_descriptor
            && self.authority.recipient_descriptor == other.authority.recipient_descriptor
            && self.authority.authorization == other.authority.authorization
            && self.authority.purpose == other.authority.purpose
            && self.expected.relay == other.expected.relay && self.expected.recipient == other.expected.recipient
            && self.expected.route == other.expected.route && self.expected.ttl == other.expected.ttl
            && self.expected.timestamp == other.expected.timestamp
            && self.expected.blob_hash == other.expected.blob_hash
            && self.expected.signing_commitment == other.expected.signing_commitment
            && self.deadline == other.deadline && self.original_expiry == other.original_expiry
            && self.dispatch == other.dispatch && self.terminal.as_slice() == other.terminal.as_slice()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SourcePhase {
    Prepared = 1, Armed = 2, DispatchAmbiguous = 3, ResultReady = 4,
    Opening = 5, Verified = 6, Rejected = 7, OpenAmbiguous = 8,
}

impl SourcePhase {
    fn decode(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Prepared), 2 => Ok(Self::Armed), 3 => Ok(Self::DispatchAmbiguous),
            4 => Ok(Self::ResultReady), 5 => Ok(Self::Opening), 6 => Ok(Self::Verified),
            7 => Ok(Self::Rejected), 8 => Ok(Self::OpenAmbiguous),
            _ => Err(SourceJournalError::Corrupt),
        }
    }
    fn permits(self, next: Self) -> bool {
        matches!((self, next),
            (Self::Prepared, Self::Armed) | (Self::Armed, Self::DispatchAmbiguous)
            | (Self::Armed | Self::DispatchAmbiguous, Self::ResultReady)
            | (Self::ResultReady, Self::Opening)
            | (Self::Opening, Self::Verified | Self::Rejected | Self::OpenAmbiguous))
    }
}

/// Exact bytes returned ONLY by the successful first Prepared->Armed CAS.
/// No Clone/Debug; runtime must not duplicate or reconstruct this authority.
pub(crate) struct SourceDispatch { pub(crate) exact_bytes: Vec<u8> }
pub(crate) struct SourceRecoveryPage {
    pub(crate) items: Vec<([u8; 16], SourcePhase)>,
    pub(crate) next_after: Option<[u8; 16]>,
}

/// Immutable restart/query metadata. It deliberately omits dispatch and
/// terminal bytes; after Armed, recovery may only issue read-only evidence
/// queries and must not obtain a second effectful request body.
pub(crate) struct SourceRecoveryMetadata {
    route: [u8; 16],
    phase: SourcePhase,
    source: [u8; 32],
    relay: [u8; 32],
    recipient: [u8; 32],
    target: [u8; 32],
    request_commitment: [u8; 32],
    body_commitment: [u8; 32],
    relay_descriptor_commitment: [u8; 32],
    recipient_descriptor_commitment: [u8; 32],
    deadline: u64,
    retain_until: u64,
}

impl SourceRecoveryMetadata {
    pub(crate) fn route(&self) -> [u8; 16] { self.route }
    pub(crate) fn phase(&self) -> SourcePhase { self.phase }
    pub(crate) fn source(&self) -> [u8; 32] { self.source }
    pub(crate) fn relay(&self) -> [u8; 32] { self.relay }
    pub(crate) fn recipient(&self) -> [u8; 32] { self.recipient }
    pub(crate) fn target(&self) -> [u8; 32] { self.target }
    pub(crate) fn request_commitment(&self) -> [u8; 32] { self.request_commitment }
    pub(crate) fn body_commitment(&self) -> [u8; 32] { self.body_commitment }
    pub(crate) fn relay_descriptor_commitment(&self) -> [u8; 32] {
        self.relay_descriptor_commitment
    }
    pub(crate) fn recipient_descriptor_commitment(&self) -> [u8; 32] {
        self.recipient_descriptor_commitment
    }
    pub(crate) fn deadline(&self) -> u64 { self.deadline }
    pub(crate) fn retain_until(&self) -> u64 { self.retain_until }
}

pub(crate) struct SourceRecoveryMetadataPage {
    pub(crate) items: Vec<SourceRecoveryMetadata>,
    pub(crate) next_after: Option<[u8; 16]>,
}

struct Record {
    plan: SourcePreparedPull,
    phase: SourcePhase,
    generation: u64,
    reserved: u64,
    retain_until: u64,
    restart: Zeroizing<Vec<u8>>,
    claim: Vec<u8>,
    lease: Vec<u8>,
    result: Vec<u8>,
    verified: Zeroizing<Vec<u8>>,
}

impl Record {
    fn reservation(plan: &SourcePreparedPull) -> Result<u64> {
        let maximum_body = [RECORD_FIXED_BYTES, plan.dispatch.len(), plan.terminal.len(),
            MAX_BLIND_VAULT_ONION_PULL_RESTART_BYTES, MAX_CLAIM_BYTES, MAX_LEASE_BYTES,
            MAX_REVERSE_ONION_FRAME_BYTES, MAX_PAGE_BYTES, MAX_SIGNED_DESCRIPTOR_BYTES,
            MAX_SIGNED_DESCRIPTOR_BYTES, MAX_AUTHORIZATION_BYTES, 128].into_iter()
            .try_fold(0usize, |sum, n| sum.checked_add(n)).ok_or(SourceJournalError::Capacity)?;
        if maximum_body > MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES {
            return Err(SourceJournalError::Capacity);
        }
        // Reserve the complete fixed record ceiling, not only current bytes.
        // Plain SQLite counters cannot understate a row's future response cost.
        // This is conservative: aggregate capacity may bind before max_entries.
        Ok(RESERVED_PER_JOB)
    }

    fn evidence(&self, now: Option<u64>) -> Result<Vec<u8>> {
        let claim = ReverseOnionFrameV1::decode_for_recovery(&self.claim).map_err(rejected)?;
        let lease = ReverseOnionFrameV1::decode_for_recovery(&self.lease).map_err(rejected)?;
        let result = ReverseOnionFrameV1::decode_for_recovery(&self.result).map_err(rejected)?;
        claim.verify_claim(self.plan.expected.relay, self.plan.expected.recipient, lease.issued_at())
            .map_err(rejected)?;
        let envelope = lease.verify_lease(&claim, self.plan.deadline, lease.issued_at()).map_err(rejected)?;
        if !self.plan.expected.matches(&envelope) { return Err(SourceJournalError::Rejected); }
        // Core already checks R's envelope signature and exact canonical bytes.
        let opaque = result.verify_result(&claim, &lease, self.plan.deadline,
            now.unwrap_or(result.issued_at())).map_err(rejected)?;
        Ok(opaque.to_vec())
    }

    fn cached_page(&self) -> Result<BlindVaultPullResponse> {
        let frame = decode_blind_vault_frame(&self.verified).map_err(|_| SourceJournalError::Corrupt)?;
        let BlindVaultFrame::PullResponse(page) = frame else { return Err(SourceJournalError::Corrupt); };
        page.validate_and_verify(&IdentityPublicKey::from_bytes(&self.plan.target)
            .map_err(|_| SourceJournalError::Corrupt)?).map_err(|_| SourceJournalError::Corrupt)?;
        if page.lease_id != terminal_lease(&self.plan.terminal)? { return Err(SourceJournalError::Corrupt); }
        Ok(page)
    }

    fn validate(&self) -> Result<()> {
        self.plan.validate().map_err(|_| SourceJournalError::Corrupt)?;
        if self.reserved != Self::reservation(&self.plan)? || self.generation == 0
            || self.retain_until != self.plan.retention()?
        { return Err(SourceJournalError::Corrupt); }
        let has_evidence = !self.result.is_empty();
        let needs_evidence = matches!(self.phase, SourcePhase::ResultReady | SourcePhase::Opening
            | SourcePhase::Verified | SourcePhase::Rejected | SourcePhase::OpenAmbiguous);
        let needs_session = matches!(self.phase, SourcePhase::Prepared | SourcePhase::Armed
            | SourcePhase::DispatchAmbiguous | SourcePhase::ResultReady | SourcePhase::Opening);
        if has_evidence != needs_evidence || self.claim.is_empty() != self.result.is_empty()
            || self.lease.is_empty() != self.result.is_empty()
            || self.restart.is_empty() == needs_session
            || self.verified.is_empty() == (self.phase == SourcePhase::Verified)
        { return Err(SourceJournalError::Corrupt); }
        if has_evidence { self.evidence(None).map_err(|_| SourceJournalError::Corrupt)?; }
        if self.phase == SourcePhase::Verified { self.cached_page()?; }
        Ok(())
    }

    fn encode(&self) -> Result<Zeroizing<Vec<u8>>> {
        let fields: [(&[u8], usize); 11] = [
            (&self.plan.dispatch, MAX_DISPATCH_BYTES), (&self.plan.terminal, MAX_TERMINAL_BYTES),
            (&self.restart, MAX_BLIND_VAULT_ONION_PULL_RESTART_BYTES), (&self.claim, MAX_CLAIM_BYTES),
            (&self.lease, MAX_LEASE_BYTES), (&self.result, MAX_REVERSE_ONION_FRAME_BYTES),
            (&self.verified, MAX_PAGE_BYTES),
            (&self.plan.authority.relay_descriptor, MAX_SIGNED_DESCRIPTOR_BYTES),
            (&self.plan.authority.recipient_descriptor, MAX_SIGNED_DESCRIPTOR_BYTES),
            (&self.plan.authority.authorization, MAX_AUTHORIZATION_BYTES),
            (&self.plan.authority.purpose, 128),
        ];
        let mut length = RECORD_FIXED_BYTES;
        for (field, bound) in fields {
            if field.len() > bound { return Err(SourceJournalError::Capacity); }
            length = length.checked_add(field.len()).ok_or(SourceJournalError::Capacity)?;
        }
        if length > MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES { return Err(SourceJournalError::Capacity); }
        let mut bytes = Zeroizing::new(Vec::with_capacity(length));
        bytes.extend_from_slice(b"AXSJ"); bytes.extend_from_slice(&SOURCE_RECORD_VERSION_V2.to_be_bytes());
        bytes.push(self.phase as u8);
        for value in [self.generation, self.reserved, self.retain_until] { bytes.extend_from_slice(&value.to_be_bytes()); }
        bytes.extend_from_slice(&self.plan.source); bytes.extend_from_slice(&self.plan.expected.route);
        for value in [self.plan.target, self.plan.descriptor, self.plan.expected.relay, self.plan.expected.recipient] {
            bytes.extend_from_slice(&value);
        }
        bytes.extend_from_slice(&self.plan.deadline.to_be_bytes());
        bytes.extend_from_slice(&self.plan.original_expiry.to_be_bytes()); bytes.push(self.plan.expected.ttl);
        bytes.extend_from_slice(&self.plan.expected.timestamp.to_be_bytes());
        bytes.extend_from_slice(&self.plan.expected.blob_hash);
        bytes.extend_from_slice(&self.plan.expected.signing_commitment);
        bytes.extend_from_slice(&self.plan.relay_descriptor_commitment);
        bytes.extend_from_slice(&self.plan.recipient_descriptor_commitment);
        bytes.extend_from_slice(&self.plan.request_commitment);
        bytes.extend_from_slice(&self.plan.body_commitment);
        for (field, _) in fields {
            bytes.extend_from_slice(&(field.len() as u32).to_be_bytes()); bytes.extend_from_slice(field);
        }
        if bytes.len() != length { return Err(SourceJournalError::Corrupt); }
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES { return Err(SourceJournalError::Corrupt); }
        let mut cursor = Cursor(bytes);
        if cursor.array::<4>()? != *b"AXSJ" {
            return Err(SourceJournalError::Corrupt);
        }
        let version = u16::from_be_bytes(cursor.array()?);
        if version == 1 { return Err(SourceJournalError::MigrationRequired); }
        if version != SOURCE_RECORD_VERSION_V2 { return Err(SourceJournalError::Corrupt); }
        let phase = SourcePhase::decode(cursor.array::<1>()?[0])?;
        let generation = cursor.number()?; let reserved = cursor.number()?; let retain_until = cursor.number()?;
        let source = cursor.array()?; let route = cursor.array()?; let target = cursor.array()?;
        let descriptor = cursor.array()?; let relay = cursor.array()?; let recipient = cursor.array()?;
        let deadline = cursor.number()?; let original_expiry = cursor.number()?;
        let ttl = cursor.array::<1>()?[0]; let timestamp = cursor.number()?;
        let blob_hash = cursor.array()?; let signing_commitment = cursor.array()?;
        let relay_descriptor_commitment = cursor.array()?;
        let recipient_descriptor_commitment = cursor.array()?;
        let request_commitment = cursor.array()?;
        let body_commitment = cursor.array()?;
        let dispatch = cursor.field(MAX_DISPATCH_BYTES)?.to_vec();
        let terminal = Zeroizing::new(cursor.field(MAX_TERMINAL_BYTES)?.to_vec());
        let restart = Zeroizing::new(cursor.field(MAX_BLIND_VAULT_ONION_PULL_RESTART_BYTES)?.to_vec());
        let claim = cursor.field(MAX_CLAIM_BYTES)?.to_vec(); let lease = cursor.field(MAX_LEASE_BYTES)?.to_vec();
        let result = cursor.field(MAX_REVERSE_ONION_FRAME_BYTES)?.to_vec();
        let verified = Zeroizing::new(cursor.field(MAX_PAGE_BYTES)?.to_vec());
        let relay_descriptor = Zeroizing::new(cursor.field(MAX_SIGNED_DESCRIPTOR_BYTES)?.to_vec());
        let recipient_descriptor = Zeroizing::new(cursor.field(MAX_SIGNED_DESCRIPTOR_BYTES)?.to_vec());
        let authorization = Zeroizing::new(cursor.field(MAX_AUTHORIZATION_BYTES)?.to_vec());
        let purpose = Zeroizing::new(cursor.field(128)?.to_vec());
        if !cursor.0.is_empty() { return Err(SourceJournalError::Corrupt); }
        let record = Self { plan: SourcePreparedPull { source, target, descriptor,
            relay_descriptor_commitment, recipient_descriptor_commitment, request_commitment,
            body_commitment,
            authority: SourceRouteAuthority { relay_descriptor, recipient_descriptor, authorization, purpose },
            deadline, original_expiry,
            expected: ExpectedRetainedEnvelope { relay, recipient, route, ttl, timestamp, blob_hash, signing_commitment },
            dispatch, terminal }, phase, generation, reserved, retain_until, restart, claim, lease, result, verified };
        record.validate()?;
        Ok(record)
    }
}

struct Inner { connection: Connection, poisoned: bool }

/// Exclusive private repository; not an OS hard quota or anti-rollback anchor.
pub(crate) struct ReverseOnionSourceJournal {
    inner: Mutex<Inner>, identity: Arc<IdentityKeyPair>, limits: SourceJournalLimits,
    #[cfg(unix)] _inode_lock: File,
    #[cfg(unix)] _parent: File,
    #[cfg(unix)] db_path: std::path::PathBuf,
    #[cfg(unix)] physical_limit: u64,
    #[cfg(unix)] inode_identity: (u64, u64),
    #[cfg(unix)] parent_identity: (u64, u64),
    #[cfg(test)] commits_until_fence_error: std::sync::atomic::AtomicUsize,
}

impl ReverseOnionSourceJournal {
    #[cfg(unix)]
    pub(crate) fn open(path: &Path, identity: Arc<IdentityKeyPair>, limits: SourceJournalLimits, now: u64) -> Result<Self> {
        use super::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};
        if limits.max_entries == 0 || limits.max_entries > MAX_ENTRIES
            || limits.max_bytes == 0 || limits.max_bytes > MAX_BYTES || path == Path::new(":memory:")
        { return Err(SourceJournalError::Rejected); }
        sql(now)?;
        // [REVERSE-ONION-SOURCE-DB-BOUNDARY 2026-10-04 by Codex] Do not
        // create/chmod the primary or its parent before existing sidecars and
        // their aggregate are admitted. This preflight has no write effects.
        let physical_limit = source_physical_limit(&limits)?;
        preflight_source_files(path, physical_limit)?;
        let target = prepare_private_sqlite_target(path).map_err(|_| SourceJournalError::Unavailable)?;
        verify_private_file(&target.resolved_path, true).map_err(|_| SourceJournalError::Rejected)?;
        let inode = std::fs::OpenOptions::new().read(true).write(true)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC | nix::libc::O_NONBLOCK)
            .open(&target.resolved_path).map_err(|_| SourceJournalError::Unavailable)?;
        let metadata = inode.metadata().map_err(|_| SourceJournalError::Unavailable)?;
        // SAFETY: geteuid takes no pointers; inode stays owned for journal lifetime.
        if !metadata.is_file() || metadata.uid() != unsafe { nix::libc::geteuid() }
            || metadata.nlink() != 1 || metadata.mode() & 0o777 != 0o600
        { return Err(SourceJournalError::Rejected); }
        if metadata.len() > physical_limit { return Err(SourceJournalError::Capacity); }
        // SAFETY: valid owned fd; advisory cooperation, not hostile same-euid defense.
        if unsafe { nix::libc::flock(inode.as_raw_fd(), nix::libc::LOCK_EX | nix::libc::LOCK_NB) } != 0 {
            return Err(SourceJournalError::Busy);
        }
        // [REVERSE-ONION-SOURCE-JOURNAL 2026-10-04 by Codex] An existing
        // nonempty DB must declare this owner before writable SQLite recovery
        // or journal-mode changes. SQLite header offsets are its documented
        // file ABI; full schema/integrity checks still follow recovery. A crash
        // during unacknowledged first initialization may require operator care.
        if metadata.len() != 0 {
            let mut header = [0u8; 100];
            inode.read_exact_at(&mut header, 0).map_err(|_| SourceJournalError::Corrupt)?;
            if &header[..16] != b"SQLite format 3\0"
                || header[60..64] != 1u32.to_be_bytes()
                || header[68..72] != (APPLICATION_ID as u32).to_be_bytes()
            { return Err(SourceJournalError::Corrupt); }
        }
        audit_source_sidecars(&target.resolved_path, physical_limit, metadata.len())?;
        let parent_metadata = target.parent.metadata().map_err(|_| SourceJournalError::Unavailable)?;
        validate_source_parent(&parent_metadata)?;
        let mut connection = Connection::open_with_flags(&target.resolved_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW).map_err(unavailable)?;
        let after = std::fs::symlink_metadata(&target.resolved_path).map_err(|_| SourceJournalError::Unavailable)?;
        if metadata.dev() != after.dev() || metadata.ino() != after.ino() { return Err(SourceJournalError::Rejected); }
        verify_private_file(&target.resolved_path, true).map_err(|_| SourceJournalError::Rejected)?;
        connection.execute_batch("PRAGMA busy_timeout=0; PRAGMA trusted_schema=OFF; PRAGMA temp_store=MEMORY; PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=DELETE; PRAGMA synchronous=EXTRA; PRAGMA fullfsync=ON; PRAGMA foreign_keys=ON;").map_err(unavailable)?;
        let page_size: i64 = connection.query_row("PRAGMA page_size", [], |r| r.get(0)).map_err(unavailable)?;
        if !(512..=65536).contains(&page_size) || !(page_size as u64).is_power_of_two() { return Err(SourceJournalError::Corrupt); }
        connection.pragma_update(None, "max_page_count", (physical_limit / page_size as u64) as i64).map_err(unavailable)?;
        audit_source_pragmas(&connection, physical_limit)?;
        let integrity: String = connection.query_row("PRAGMA quick_check", [], |r| r.get(0)).map_err(unavailable)?;
        if integrity != "ok" { return Err(SourceJournalError::Corrupt); }
        initialize_schema(&mut connection, identity.public_key_bytes(), now)?;
        let journal = Self { inner: Mutex::new(Inner { connection, poisoned: false }), identity, limits,
            _inode_lock: inode, _parent: target.parent,
            db_path: target.resolved_path, physical_limit,
            inode_identity: (metadata.dev(), metadata.ino()),
            parent_identity: (parent_metadata.dev(), parent_metadata.ino()),
            #[cfg(test)] commits_until_fence_error: std::sync::atomic::AtomicUsize::new(0),
        };
        journal.with_inner(|inner| journal.transaction(inner, now, |tx| {
            for id in ids(tx, None, MAX_ENTRIES + 1)? {
                let mut row = journal.load(tx, id)?.ok_or(SourceJournalError::Corrupt)?;
                // Historical descriptor/auth evidence proves the original
                // admission; current execution/recovery freshness is checked
                // separately by the phase and retention deadlines.
                row.plan.validate_at(row.plan.expected.timestamp)?;
                let next = match row.phase {
                    SourcePhase::Armed => Some(SourcePhase::DispatchAmbiguous),
                    SourcePhase::Opening => Some(SourcePhase::OpenAmbiguous), _ => None,
                };
                if let Some(next) = next {
                    if next == SourcePhase::OpenAmbiguous { row.restart.zeroize(); }
                    journal.replace(tx, &mut row, next)?;
                }
            }
            Ok(())
        }))?;
        Ok(journal)
    }

    #[cfg(not(unix))]
    pub(crate) fn open(_path: &Path, _identity: Arc<IdentityKeyPair>, _limits: SourceJournalLimits, _now: u64) -> Result<Self> {
        Err(SourceJournalError::Rejected)
    }

    /// Exact retries compare immutable admitted claims, not randomized AEAD bytes.
    pub(crate) fn prepare(&self, plan: SourcePreparedPull, session: BlindVaultOnionPullSession, now: u64) -> Result<SourcePhase> {
        plan.validate()?; plan.fresh(now)?;
        if plan.source != self.identity.public_key_bytes() { return Err(SourceJournalError::Rejected); }
        let restart = Zeroizing::new(session.seal_restart(&self.identity, plan.expected.route, plan.target, &plan.terminal)
            .map_err(|_| SourceJournalError::Rejected)?);
        let reserved = Record::reservation(&plan)?;
        let retain_until = plan.retention()?;
        let row = Record { plan, restart, reserved, retain_until, generation: 1, phase: SourcePhase::Prepared,
            claim: vec![], lease: vec![], result: vec![], verified: Zeroizing::new(vec![]) };
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            if let Some(existing) = self.load(tx, row.plan.expected.route)? {
                if !existing.plan.same(&row.plan) { return Err(SourceJournalError::Conflict); }
                return Ok(existing.phase);
            }
            let sealed = self.seal(&row)?;
            one(tx.execute("INSERT INTO source_jobs VALUES(?1,?2,?3,?4,?5,?6)", params![
                row.plan.expected.route.as_slice(), row.phase as u8, sql(row.generation)?,
                sql(row.reserved)?, sql(row.retain_until)?, sealed]).map_err(unavailable)?)?;
            Ok(SourcePhase::Prepared)
        }))
    }

    pub(crate) fn arm(&self, route: [u8; 16], now: u64) -> Result<SourceDispatch> {
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let mut row = self.load(tx, route)?.ok_or(SourceJournalError::Rejected)?;
            if row.phase != SourcePhase::Prepared { return Err(SourceJournalError::Ambiguous); }
            row.plan.fresh(now)?;
            self.replace(tx, &mut row, SourcePhase::Armed)?;
            Ok(SourceDispatch { exact_bytes: row.plan.dispatch })
        }))
    }

    pub(crate) fn mark_dispatch_ambiguous(&self, route: [u8; 16], now: u64) -> Result<()> {
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let mut row = self.load(tx, route)?.ok_or(SourceJournalError::Rejected)?;
            if row.phase == SourcePhase::DispatchAmbiguous { return Ok(()); }
            self.replace(tx, &mut row, SourcePhase::DispatchAmbiguous)
        }))
    }

    /// No caller-supplied 'verified' bit: re-check the complete adjacent chain.
    pub(crate) fn record_result(&self, route: [u8; 16], claim: &ReverseOnionFrameV1,
        lease: &ReverseOnionFrameV1, result: &ReverseOnionFrameV1, now: u64) -> Result<()> {
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let mut row = self.load(tx, route)?.ok_or(SourceJournalError::Rejected)?;
            let claim_bytes = claim.encode(); let lease_bytes = lease.encode(); let result_bytes = result.encode();
            if claim_bytes.len() > MAX_CLAIM_BYTES || lease_bytes.len() > MAX_LEASE_BYTES
                || result_bytes.len() > MAX_REVERSE_ONION_FRAME_BYTES { return Err(SourceJournalError::Rejected); }
            if !row.result.is_empty() {
                if row.claim == claim_bytes && row.lease == lease_bytes && row.result == result_bytes { return Ok(()); }
                return Err(SourceJournalError::Conflict);
            }
            if !matches!(row.phase, SourcePhase::Armed | SourcePhase::DispatchAmbiguous) {
                return Err(SourceJournalError::Conflict);
            }
            if now >= row.retain_until { return Err(SourceJournalError::Expired); }
            row.claim = claim_bytes; row.lease = lease_bytes; row.result = result_bytes;
            row.evidence(Some(now))?;
            self.replace(tx, &mut row, SourcePhase::ResultReady)
        }))
    }

    /// Owns the mutex across both durable transitions and synchronous crypto.
    /// No async await or recursively acquired lock. Panic poisons the mutex;
    /// restart retires Opening without ever restoring its reply key again.
    pub(crate) fn open_result(&self, route: [u8; 16], now: u64) -> Result<BlindVaultPullResponse> {
        self.with_inner(|inner| {
            let mut row = self.transaction(inner, now, |tx| {
                let mut row = self.load(tx, route)?.ok_or(SourceJournalError::Rejected)?;
                if row.phase != SourcePhase::ResultReady { return Err(SourceJournalError::Ambiguous); }
                row.evidence(Some(now))?;
                self.replace(tx, &mut row, SourcePhase::Opening)?;
                Ok(row)
            })?;
            // Only this post-commit point may restore the session. Neither load
            // nor recover nor read_verified calls this API.
            let session = BlindVaultOnionPullSession::restore_restart(&self.identity, &row.restart,
                row.plan.expected.route, row.plan.target, &row.plan.terminal)
                .map_err(|_| SourceJournalError::Corrupt)?;
            let opaque = row.evidence(None).map_err(|_| SourceJournalError::Corrupt)?;
            let verified = session.open(&opaque);
            row.restart.zeroize();
            match verified {
                Ok(page) => {
                    row.verified = Zeroizing::new(encode_blind_vault_frame(&BlindVaultFrame::PullResponse(page))
                        .map_err(|_| SourceJournalError::Corrupt)?);
                    if row.verified.len() > MAX_PAGE_BYTES { return Err(SourceJournalError::Corrupt); }
                    let page = row.cached_page()?;
                    self.transaction(inner, now, |tx| self.replace(tx, &mut row, SourcePhase::Verified))?;
                    Ok(page)
                }
                Err(_) => {
                    self.transaction(inner, now, |tx| self.replace(tx, &mut row, SourcePhase::Rejected))?;
                    Err(SourceJournalError::ReplyRejected)
                }
            }
        })
    }

    pub(crate) fn read_verified(&self, route: [u8; 16], now: u64) -> Result<BlindVaultPullResponse> {
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let row = self.load(tx, route)?.ok_or(SourceJournalError::Rejected)?;
            if now >= row.retain_until { return Err(SourceJournalError::Expired); }
            if row.phase != SourcePhase::Verified { return Err(SourceJournalError::Rejected); }
            row.cached_page()
        }))
    }

    pub(crate) fn recover(&self, after: Option<[u8; 16]>, limit: usize, now: u64) -> Result<SourceRecoveryPage> {
        if limit == 0 || limit > PAGE_LIMIT { return Err(SourceJournalError::Rejected); }
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let selected = ids(tx, after, limit + 1)?;
            let more = selected.len() > limit;
            let mut items = Vec::with_capacity(limit);
            for id in selected.into_iter().take(limit) {
                let row = self.load(tx, id)?.ok_or(SourceJournalError::Corrupt)?;
                items.push((id, row.phase));
            }
            let next_after = if more { items.last().map(|(id, _)| *id) } else { None };
            Ok(SourceRecoveryPage { items, next_after })
        }))
    }

    /// Returns only immutable evidence-query metadata. No persisted effectful
    /// dispatch or terminal bytes cross this recovery boundary.
    pub(crate) fn recover_metadata(
        &self,
        after: Option<[u8; 16]>,
        limit: usize,
        now: u64,
    ) -> Result<SourceRecoveryMetadataPage> {
        if limit == 0 || limit > PAGE_LIMIT { return Err(SourceJournalError::Rejected); }
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let selected = ids(tx, after, limit + 1)?;
            let more = selected.len() > limit;
            let mut items = Vec::with_capacity(limit);
            for id in selected.into_iter().take(limit) {
                let row = self.load(tx, id)?.ok_or(SourceJournalError::Corrupt)?;
                // Descriptor/auth verification is anchored to the original
                // admitted timestamp, not to a later result-grace read.
                row.plan.validate_at(row.plan.expected.timestamp)?;
                items.push(SourceRecoveryMetadata {
                    route: id,
                    phase: row.phase,
                    source: row.plan.source,
                    relay: row.plan.expected.relay,
                    recipient: row.plan.expected.recipient,
                    target: row.plan.target,
                    request_commitment: row.plan.request_commitment,
                    body_commitment: row.plan.body_commitment,
                    relay_descriptor_commitment: row.plan.relay_descriptor_commitment,
                    recipient_descriptor_commitment: row.plan.recipient_descriptor_commitment,
                    deadline: row.plan.deadline,
                    retain_until: row.retain_until,
                });
            }
            let next_after = if more { items.last().map(|item| item.route) } else { None };
            Ok(SourceRecoveryMetadataPage { items, next_after })
        }))
    }

    pub(crate) fn cleanup(&self, limit: usize, now: u64) -> Result<usize> {
        if limit == 0 || limit > PAGE_LIMIT { return Err(SourceJournalError::Rejected); }
        self.with_inner(|inner| self.transaction(inner, now, |tx| {
            let mut removed = 0;
            // Scan at most the fixed row ceiling in SQLite, but decrypt only
            // this cleanup batch, not every unrelated live result/large page.
            let selected = {
                let mut statement = tx.prepare("SELECT CASE WHEN typeof(route)='blob' AND length(route)=16 THEN route ELSE NULL END FROM source_jobs WHERE retain_until<=?1 ORDER BY route LIMIT ?2").map_err(unavailable)?;
                let rows = statement.query_map(params![sql(now)?, limit as i64], |r| r.get::<_, Option<Vec<u8>>>(0)).map_err(unavailable)?;
                let mut selected: Vec<[u8; 16]> = Vec::new();
                for row in rows {
                    selected.push(row.map_err(|_| SourceJournalError::Corrupt)?.ok_or(SourceJournalError::Corrupt)?
                        .try_into().map_err(|_| SourceJournalError::Corrupt)?);
                }
                selected
            };
            for id in selected {
                let row = self.load(tx, id)?.ok_or(SourceJournalError::Corrupt)?;
                if now >= row.retain_until {
                    one(tx.execute("DELETE FROM source_jobs WHERE route=?1 AND generation=?2 AND retain_until<=?3",
                        params![id.as_slice(), sql(row.generation)?, sql(now)?]).map_err(unavailable)?)?;
                    removed += 1;
                    if removed == limit { break; }
                }
            }
            Ok(removed)
        }))
    }

    fn with_inner<T>(&self, action: impl FnOnce(&mut Inner) -> Result<T>) -> Result<T> {
        let mut inner = match self.inner.try_lock() {
            Ok(inner) => inner, Err(TryLockError::WouldBlock) => return Err(SourceJournalError::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(SourceJournalError::Unavailable),
        };
        if inner.poisoned { return Err(SourceJournalError::Unavailable); }
        let result = action(&mut inner);
        if matches!(&result, Err(SourceJournalError::Corrupt | SourceJournalError::Unavailable | SourceJournalError::ClockRollback)) {
            inner.poisoned = true;
        }
        result
    }

    fn transaction<T>(&self, inner: &mut Inner, now: u64, action: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<T> {
        let tx = inner.connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(unavailable)?;
        let outcome = (|| {
            let (source, clock): (Vec<u8>, i64) = tx.query_row(
                "SELECT source,clock FROM source_meta WHERE singleton=1 AND typeof(source)='blob' AND length(source)=32 AND typeof(clock)='integer'",
                [], |r| Ok((r.get(0)?, r.get(1)?))).map_err(|_| SourceJournalError::Corrupt)?;
            if source.as_slice() != self.identity.public_key_bytes().as_slice() || clock < 0 { return Err(SourceJournalError::Corrupt); }
            if sql(now)? < clock { return Err(SourceJournalError::ClockRollback); }
            self.audit_bounds(&tx)?;
            let value = action(&tx)?;
            self.audit_bounds(&tx)?;
            one(tx.execute("UPDATE source_meta SET clock=?1 WHERE singleton=1 AND clock=?2", params![sql(now)?, clock]).map_err(unavailable)?)?;
            Ok(value)
        })();
        match outcome {
            Ok(value) => {
                tx.commit().map_err(unavailable)?;
                // [REVERSE-ONION-SOURCE-JOURNAL 2026-10-04 by Codex] A
                // deterministic test-only post-commit uncertainty seam. There
                // is no production bypass constructor or weakened sync mode.
                #[cfg(test)] {
                    use std::sync::atomic::Ordering;
                    let remaining = self.commits_until_fence_error.load(Ordering::SeqCst);
                    if remaining > 0 {
                        self.commits_until_fence_error.store(remaining - 1, Ordering::SeqCst);
                        if remaining == 1 { return Err(SourceJournalError::Unavailable); }
                    }
                }
                // Keep the caller's mutex until every observed file/pragma
                // fence succeeds; a failed post-commit bound is ambiguous and
                // poisons via Unavailable, NEVER a successful dispatch/result.
                self.post_operation_fence(&inner.connection)
                    .map_err(|_| SourceJournalError::Unavailable)?;
                Ok(value)
            }
            Err(error) => {
                tx.rollback().map_err(unavailable)?;
                self.post_operation_fence(&inner.connection)
                    .map_err(|_| SourceJournalError::Unavailable)?;
                Err(error)
            }
        }
    }

    // [REVERSE-ONION-SOURCE-DB-BOUNDARY 2026-10-04 by Codex] Observes the
    // aggregate at this fence, not transient filesystem peak usage. Same-euid
    // hostile pathname races require external isolation/a trusted SQLite VFS.
    #[cfg(unix)]
    fn post_operation_fence(&self, connection: &Connection) -> Result<()> {
        let held = self._inode_lock.metadata().map_err(|_| SourceJournalError::Unavailable)?;
        validate_source_file(&held)?;
        if (held.dev(), held.ino()) != self.inode_identity { return Err(SourceJournalError::Rejected); }
        let parent = self._parent.metadata().map_err(|_| SourceJournalError::Unavailable)?;
        validate_source_parent(&parent)?;
        if (parent.dev(), parent.ino()) != self.parent_identity { return Err(SourceJournalError::Rejected); }
        let parent_path = self.db_path.parent().ok_or(SourceJournalError::Rejected)?;
        let observed_parent = std::fs::symlink_metadata(parent_path).map_err(|_| SourceJournalError::Unavailable)?;
        validate_source_parent(&observed_parent)?;
        if (observed_parent.dev(), observed_parent.ino()) != self.parent_identity {
            return Err(SourceJournalError::Rejected);
        }
        let observed = std::fs::symlink_metadata(&self.db_path).map_err(|_| SourceJournalError::Unavailable)?;
        validate_source_file(&observed)?;
        if (observed.dev(), observed.ino()) != self.inode_identity || observed.len() != held.len() {
            return Err(SourceJournalError::Rejected);
        }
        audit_source_sidecars(&self.db_path, self.physical_limit, observed.len())?;
        audit_source_pragmas(connection, self.physical_limit)?;
        self._parent.sync_all().map_err(|_| SourceJournalError::Unavailable)
    }

    #[cfg(not(unix))]
    fn post_operation_fence(&self, _connection: &Connection) -> Result<()> { Err(SourceJournalError::Rejected) }

    fn audit_bounds(&self, tx: &Transaction<'_>) -> Result<()> {
        let mut statement = tx.prepare("SELECT reserved,length(sealed),typeof(sealed),typeof(reserved) FROM source_jobs LIMIT ?1").map_err(unavailable)?;
        let rows = statement.query_map(params![(MAX_ENTRIES + 1) as i64], |r| Ok((r.get::<_, i64>(0)?,
            r.get::<_, i64>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))).map_err(unavailable)?;
        let mut count = 0usize; let mut reserved_total = 0u64;
        for row in rows {
            let (reserved, actual, blob_type, number_type) = row.map_err(|_| SourceJournalError::Corrupt)?;
            if reserved != RESERVED_PER_JOB as i64 || actual < SEALED_OVERHEAD as i64 || blob_type != "blob" || number_type != "integer"
                || actual > MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES as i64
                || (actual as u64).checked_add(ROW_ACCOUNTING_BYTES as u64).map_or(true, |n| n > reserved as u64)
            { return Err(SourceJournalError::Corrupt); }
            reserved_total = reserved_total.checked_add(reserved as u64).ok_or(SourceJournalError::Corrupt)?;
            count += 1;
        }
        if count > self.limits.max_entries || reserved_total > self.limits.max_bytes { return Err(SourceJournalError::Capacity); }
        Ok(())
    }

    fn seal(&self, row: &Record) -> Result<Vec<u8>> {
        row.validate()?;
        seal_blind_vault_source_pull_journal(&self.identity, &row.encode()?).map_err(|_| SourceJournalError::Corrupt)
    }

    fn replace(&self, tx: &Transaction<'_>, row: &mut Record, next: SourcePhase) -> Result<()> {
        let old = row.phase; let generation = row.generation;
        if !old.permits(next) { return Err(SourceJournalError::Conflict); }
        row.phase = next; row.generation = generation.checked_add(1).ok_or(SourceJournalError::Corrupt)?;
        let sealed = self.seal(row)?;
        one(tx.execute("UPDATE source_jobs SET phase=?1,generation=?2,sealed=?3 WHERE route=?4 AND phase=?5 AND generation=?6 AND reserved=?7 AND retain_until=?8",
            params![next as u8, sql(row.generation)?, sealed, row.plan.expected.route.as_slice(), old as u8,
                sql(generation)?, sql(row.reserved)?, sql(row.retain_until)?]).map_err(unavailable)?)
    }

    fn load(&self, tx: &Transaction<'_>, route: [u8; 16]) -> Result<Option<Record>> {
        let length: Option<i64> = tx.query_row("SELECT CASE WHEN typeof(sealed)='blob' THEN length(sealed) ELSE -1 END FROM source_jobs WHERE route=?1",
            params![route.as_slice()], |r| r.get(0)).optional().map_err(|_| SourceJournalError::Corrupt)?;
        let Some(length) = length else { return Ok(None); };
        if length < SEALED_OVERHEAD as i64 || length > MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES as i64 {
            return Err(SourceJournalError::Corrupt);
        }
        let (phase, generation, reserved, retain, sealed): (i64, i64, i64, i64, Vec<u8>) = tx.query_row(
            "SELECT phase,generation,reserved,retain_until,sealed FROM source_jobs WHERE route=?1 AND typeof(phase)='integer' AND typeof(generation)='integer' AND typeof(reserved)='integer' AND typeof(retain_until)='integer'",
            params![route.as_slice()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).map_err(|_| SourceJournalError::Corrupt)?;
        let clear = open_blind_vault_source_pull_journal(&self.identity, &sealed).map_err(|_| SourceJournalError::Corrupt)?;
        let row = Record::decode(&clear)?;
        if row.plan.source != self.identity.public_key_bytes() || row.plan.expected.route != route
            || row.phase as i64 != phase || sql(row.generation)? != generation || sql(row.reserved)? != reserved
            || sql(row.retain_until)? != retain { return Err(SourceJournalError::Corrupt); }
        Ok(Some(row))
    }
}

// [REVERSE-ONION-SOURCE-DB-BOUNDARY 2026-10-04 by Codex] Local composition
// of the reviewed queue boundary policy; no shared helper or owner changes.
fn source_physical_limit(limits: &SourceJournalLimits) -> Result<u64> {
    limits.max_bytes.checked_mul(2)
        .and_then(|n| n.checked_add((MAX_ENTRIES as u64 + 256).checked_mul(4096)?))
        .ok_or(SourceJournalError::Rejected)
}

#[cfg(unix)]
fn validate_source_file(metadata: &std::fs::Metadata) -> Result<()> {
    // SAFETY: geteuid has no arguments or pointer preconditions.
    if !metadata.is_file() || metadata.uid() != unsafe { nix::libc::geteuid() }
        || metadata.nlink() != 1 || metadata.mode() & 0o777 != 0o600
    { return Err(SourceJournalError::Rejected); }
    Ok(())
}

#[cfg(unix)]
fn validate_source_parent(metadata: &std::fs::Metadata) -> Result<()> {
    // SAFETY: geteuid has no arguments or pointer preconditions.
    if !metadata.is_dir() || metadata.uid() != unsafe { nix::libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    { return Err(SourceJournalError::Rejected); }
    Ok(())
}

#[cfg(unix)]
fn preflight_source_files(path: &Path, physical_limit: u64) -> Result<()> {
    let parent_path = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    match std::fs::symlink_metadata(parent_path) {
        Ok(parent) => validate_source_parent(&parent)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(SourceJournalError::Unavailable),
    }
    let primary = match std::fs::symlink_metadata(path) {
        Ok(metadata) => { validate_source_file(&metadata)?; metadata.len() }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(_) => return Err(SourceJournalError::Unavailable),
    };
    audit_source_sidecars(path, physical_limit, primary)
}

fn source_checked_aggregate(primary: u64, rollback: u64, limit: u64) -> Result<u64> {
    let total = primary.checked_add(rollback).ok_or(SourceJournalError::Capacity)?;
    if total > limit { return Err(SourceJournalError::Capacity); }
    Ok(total)
}

#[cfg(unix)]
fn source_sidecar(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string(); name.push(suffix); name.into()
}

#[cfg(unix)]
fn audit_source_sidecars(path: &Path, physical_limit: u64, primary: u64) -> Result<()> {
    let mut total = source_checked_aggregate(primary, 0, physical_limit)?;
    for suffix in ["-journal", "-wal", "-shm"] {
        match std::fs::symlink_metadata(source_sidecar(path, suffix)) {
            Ok(_) if suffix != "-journal" => return Err(SourceJournalError::Corrupt),
            Ok(metadata) => {
                validate_source_file(&metadata)?;
                total = source_checked_aggregate(total, metadata.len(), physical_limit)?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(SourceJournalError::Unavailable),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn audit_source_pragmas(connection: &Connection, physical_limit: u64) -> Result<()> {
    for (query, expected) in [
        ("PRAGMA busy_timeout", 0i64), ("PRAGMA trusted_schema", 0),
        ("PRAGMA temp_store", 2), ("PRAGMA synchronous", 3),
        ("PRAGMA fullfsync", 1), ("PRAGMA foreign_keys", 1),
    ] {
        let actual: i64 = connection.query_row(query, [], |r| r.get(0)).map_err(unavailable)?;
        if actual != expected { return Err(SourceJournalError::Unavailable); }
    }
    for (query, expected) in [("PRAGMA journal_mode", "delete"), ("PRAGMA locking_mode", "exclusive")] {
        let actual: String = connection.query_row(query, [], |r| r.get(0)).map_err(unavailable)?;
        if !actual.eq_ignore_ascii_case(expected) { return Err(SourceJournalError::Unavailable); }
    }
    let page_size: i64 = connection.query_row("PRAGMA page_size", [], |r| r.get(0)).map_err(unavailable)?;
    let pages: i64 = connection.query_row("PRAGMA page_count", [], |r| r.get(0)).map_err(unavailable)?;
    let maximum: i64 = connection.query_row("PRAGMA max_page_count", [], |r| r.get(0)).map_err(unavailable)?;
    if !(512..=65536).contains(&page_size) || !(page_size as u64).is_power_of_two() {
        return Err(SourceJournalError::Corrupt);
    }
    if pages < 0 || maximum <= 0 || pages > maximum || maximum as u64 > physical_limit / page_size as u64
        || (pages as u64).checked_mul(page_size as u64).map_or(true, |n| n > physical_limit)
    { return Err(SourceJournalError::Capacity); }
    Ok(())
}

fn initialize_schema(connection: &mut Connection, source: [u8; 32], now: u64) -> Result<()> {
    let tx = connection.transaction_with_behavior(TransactionBehavior::Exclusive).map_err(unavailable)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(unavailable)?;
    let app: i64 = tx.query_row("PRAGMA application_id", [], |r| r.get(0)).map_err(unavailable)?;
    let count: i64 = tx.query_row("SELECT count(*) FROM (SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 3)", [], |r| r.get(0)).map_err(unavailable)?;
    if version == 0 && app == 0 && count == 0 {
        tx.execute_batch(META_SQL).map_err(unavailable)?; tx.execute_batch(ROW_SQL).map_err(unavailable)?;
        tx.execute("INSERT INTO source_meta VALUES(1,?1,?2)", params![source.as_slice(), sql(now)?]).map_err(unavailable)?;
        tx.pragma_update(None, "application_id", APPLICATION_ID).map_err(unavailable)?;
        tx.pragma_update(None, "user_version", 1).map_err(unavailable)?;
    } else if version != 1 || app != APPLICATION_ID || count != 2 { return Err(SourceJournalError::Corrupt); }
    for (name, expected) in [("source_meta", META_SQL), ("source_jobs", ROW_SQL)] {
        let actual: String = tx.query_row("SELECT CASE WHEN length(sql)=?2 THEN sql ELSE NULL END FROM sqlite_master WHERE type='table' AND name=?1",
            params![name, expected.len() as i64], |r| r.get(0)).map_err(|_| SourceJournalError::Corrupt)?;
        if actual != expected { return Err(SourceJournalError::Corrupt); }
    }
    tx.commit().map_err(unavailable)
}

fn ids(tx: &Transaction<'_>, after: Option<[u8; 16]>, limit: usize) -> Result<Vec<[u8; 16]>> {
    let mut statement = tx.prepare("SELECT CASE WHEN typeof(route)='blob' AND length(route)=16 THEN route ELSE NULL END FROM source_jobs WHERE (?1 IS NULL OR route>?1) ORDER BY route LIMIT ?2").map_err(unavailable)?;
    let rows = statement.query_map(params![after.as_ref().map(|id| id.as_slice()), limit as i64], |r| r.get::<_, Option<Vec<u8>>>(0)).map_err(unavailable)?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row.map_err(|_| SourceJournalError::Corrupt)?.ok_or(SourceJournalError::Corrupt)?
            .try_into().map_err(|_| SourceJournalError::Corrupt)?);
        if result.len() > MAX_ENTRIES { return Err(SourceJournalError::Capacity); }
    }
    Ok(result)
}

fn terminal_lease(encoded: &[u8]) -> Result<[u8; 32]> {
    let mut request = decode_onion_reply_request(encoded).map_err(|_| SourceJournalError::Corrupt)?;
    let decoded = decode_blind_vault_frame(&request.payload);
    request.payload.zeroize();
    let BlindVaultFrame::PullRequest(mut pull) = decoded.map_err(|_| SourceJournalError::Corrupt)? else {
        return Err(SourceJournalError::Corrupt);
    };
    let valid = pull.validate().is_ok() && pull.limit == 1;
    let lease = pull.lease_id;
    pull.read_capability.zeroize(); pull.continuation_cursor.zeroize();
    if !valid { return Err(SourceJournalError::Corrupt); }
    Ok(lease)
}

struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() { return Err(SourceJournalError::Corrupt); }
        let (value, rest) = self.0.split_at(n);
        self.0 = rest; Ok(value)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> { self.take(N)?.try_into().map_err(|_| SourceJournalError::Corrupt) }
    fn number(&mut self) -> Result<u64> { Ok(u64::from_be_bytes(self.array()?)) }
    fn field(&mut self, limit: usize) -> Result<&'a [u8]> {
        let n = u32::from_be_bytes(self.array()?) as usize;
        if n > limit { return Err(SourceJournalError::Corrupt); }
        self.take(n)
    }
}
fn hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }
fn valid_key(bytes: [u8; 32]) -> Result<()> {
    IdentityPublicKey::from_bytes(&bytes).map(|_| ()).map_err(|_| SourceJournalError::Rejected)
}
fn sql(n: u64) -> Result<i64> { i64::try_from(n).map_err(|_| SourceJournalError::Rejected) }
fn rejected<T>(_: T) -> SourceJournalError { SourceJournalError::Rejected }
fn unavailable(_: rusqlite::Error) -> SourceJournalError { SourceJournalError::Unavailable }
fn one(n: usize) -> Result<()> { if n == 1 { Ok(()) } else { Err(SourceJournalError::Corrupt) } }

// [REVERSE-ONION-SOURCE-JOURNAL 2026-10-04 by Codex] Authored only. These
// tests use R's private key only to simulate actual relay delivery; expected
// bytes now come from the production typed, same-pass verified builder.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultPullRequest, BlindVaultRecoveredObject, BLIND_VAULT_PROTOCOL_VERSION,
        BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES,
    };
    use aeronyx_core::protocol::onion::{open_onion_layer, OnionRoutePurpose, VerifiedOnionRoute};
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeDescriptor, NodeProtocolFeature, SignedNodeDescriptor,
        SignedPrivateOnionRecipientAuthorizationV1,
    };
    use aeronyx_core::protocol::onion_reply::{encode_onion_sealed_response, seal_onion_reply};

    const NOW: u64 = 1_800_000_000;

    struct Fixture {
        directory: tempfile::TempDir,
        source: Arc<IdentityKeyPair>, relay: IdentityKeyPair, recipient: IdentityKeyPair,
        outbound: PeerBlindRelayRequest, retained: BlindRelayEnvelope,
        expectation: VerifiedOnionForwardExpectation,
        terminal: Zeroizing<Vec<u8>>, snapshot: Zeroizing<Vec<u8>>,
        claim: ReverseOnionFrameV1, lease: ReverseOnionFrameV1,
        authority: SourceRouteAuthority,
        relay_descriptor_commitment: [u8; 32],
        recipient_descriptor_commitment: [u8; 32],
    }

    impl Fixture {
        // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] All route
        // descriptors are signed; no server fixture constructs the core type.
        fn descriptor(identity: &IdentityKeyPair) -> SignedNodeDescriptor {
            let purpose = OnionRoutePurpose::BlindVaultPull;
            let features = purpose.required_terminal_protocol_features().iter()
                .chain(purpose.required_path_protocol_features()).copied()
                .chain(std::iter::once(NodeProtocolFeature::AnonymousMailboxV1));
            let mut descriptor = NodeDescriptor::new(identity.public_key_bytes(), 1, NOW - 1, NOW + 10_000, "test")
                .with_x25519_kem(identity.x25519_public_key_bytes()).with_protocol_features(features);
            descriptor.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle, NodeCapability::BlindVaultReplica];
            descriptor.public_endpoint = Some("https://1.1.1.1:443".into());
            SignedNodeDescriptor::sign(descriptor, identity).unwrap()
        }
        fn new() -> Self {
            Self::new_with_route(11)
        }
        fn new_with_route(route_byte: u8) -> Self {
            let directory = tempfile::Builder::new().prefix("r1-source-pull-test-")
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let source = Arc::new(IdentityKeyPair::from_bytes(&[41; 32]).unwrap());
            let relay = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
            let recipient = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
            let route = [route_byte; 16];
            let (request, session) = BlindVaultOnionPullSession::prepare(route, recipient.public_key_bytes(),
                BlindVaultPullRequest { version: BLIND_VAULT_PROTOCOL_VERSION, lease_id: [7; 32],
                    read_capability: [8; 32], continuation_cursor: vec![], limit: 1 }).unwrap();
            let terminal = Zeroizing::new(request);
            let snapshot = Zeroizing::new(session.seal_restart(&source, route, recipient.public_key_bytes(), &terminal).unwrap());
            let descriptors = [Self::descriptor(&relay), Self::descriptor(&recipient)];
            let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
                &descriptors[0], &descriptors[1], OnionRoutePurpose::BlindVaultPull.as_str(),
                NOW, NOW + 9_000, &recipient,
            ).unwrap();
            let authority = SourceRouteAuthority::from_signed(
                &descriptors[0], &descriptors[1], &authorization,
                OnionRoutePurpose::BlindVaultPull.as_str(),
            ).unwrap();
            let relay_descriptor_commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptors[0]).unwrap().hash();
            let recipient_descriptor_commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptors[1]).unwrap().hash();
            let verified_route = VerifiedOnionRoute::from_signed_descriptors(
                source.public_key_bytes(), descriptors.iter(), OnionRoutePurpose::BlindVaultPull, NOW,
            ).unwrap();
            let (envelope, expectation) = verified_route.build_envelope_with_forward_expectation(
                &terminal, route, NOW, &source,
            ).unwrap();
            let expectation = expectation.unwrap();
            // R's secret is used ONLY after source construction to emulate R.
            let (relay_secret, _) = relay.to_x25519();
            let peeled = open_onion_layer(&envelope.encrypted_blob, &relay_secret).unwrap();
            let retained = BlindRelayEnvelope { route_id: route, next_hop: recipient.public_key_bytes(), ttl: 1,
                timestamp: NOW, encrypted_blob: peeled.inner, signature: [0; 64] }.sign_with(&relay);
            let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [12; 16], NOW, NOW + 30, &recipient).unwrap();
            let lease = ReverseOnionFrameV1::lease(&claim, &retained, [13; 16], NOW + 600, NOW, &relay).unwrap();
            let outbound = PeerBlindRelayRequest { envelope, previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None, onward_descriptor_hint: None };
            Self { directory, source, relay, recipient, outbound, retained, expectation, terminal, snapshot,
                claim, lease, authority, relay_descriptor_commitment, recipient_descriptor_commitment }
        }
        fn path(&self) -> std::path::PathBuf { self.directory.path().join("source.sqlite") }
        fn route(&self) -> [u8; 16] { self.retained.route_id }
        fn expected(&self) -> ExpectedRetainedEnvelope {
            ExpectedRetainedEnvelope::from_verified_forward_expectation(&self.expectation).unwrap()
        }
        fn plan(&self) -> SourcePreparedPull {
            let request_commitment =
                blind_relay_authenticated_request_commitment(&self.outbound).unwrap();
            SourcePreparedPull::from_runtime_admission(&self.source, self.outbound.clone(), self.expected(),
                self.recipient.public_key_bytes(), self.recipient_descriptor_commitment,
                self.relay_descriptor_commitment, self.recipient_descriptor_commitment,
                request_commitment, SourceRouteAuthority {
                    relay_descriptor: self.authority.relay_descriptor.clone(),
                    recipient_descriptor: self.authority.recipient_descriptor.clone(),
                    authorization: self.authority.authorization.clone(),
                    purpose: self.authority.purpose.clone(),
                }, NOW + 600, self.terminal.to_vec()).unwrap()
        }
        fn session(&self) -> BlindVaultOnionPullSession {
            // Fixture construction only. Production restore lives inside Opening.
            BlindVaultOnionPullSession::restore_restart(&self.source, &self.snapshot,
                self.route(), self.recipient.public_key_bytes(), &self.terminal).unwrap()
        }
        fn open(&self, now: u64) -> ReverseOnionSourceJournal {
            self.open_limits(now, 8, RESERVED_PER_JOB * 8).unwrap()
        }
        fn open_limits(&self, now: u64, max_entries: usize, max_bytes: u64) -> Result<ReverseOnionSourceJournal> {
            ReverseOnionSourceJournal::open(&self.path(), self.source.clone(), SourceJournalLimits { max_entries, max_bytes }, now)
        }
        fn prepare(&self, journal: &ReverseOnionSourceJournal, now: u64) {
            assert_eq!(journal.prepare(self.plan(), self.session(), now).unwrap(), SourcePhase::Prepared);
        }
        fn response(&self, lease: &ReverseOnionFrameV1, lease_id: [u8; 32], maximum: bool) -> ReverseOnionFrameV1 {
            let request = decode_onion_reply_request(&self.terminal).unwrap();
            let objects = if maximum {
                let bytes = vec![0x55; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES.len() - 1]];
                vec![BlindVaultRecoveredObject { object_id: [6; 32], ciphertext_commitment: hash(&bytes),
                    ciphertext: bytes, expires_at_ms: (NOW + 3600) * 1000 }]
            } else { vec![] };
            let mut page = BlindVaultPullResponse::new(lease_id, objects, vec![], NOW * 1000, self.recipient.public_key_bytes());
            page.sign(&self.recipient).unwrap();
            let encoded = encode_blind_vault_frame(&BlindVaultFrame::PullResponse(page)).unwrap();
            let sealed = seal_onion_reply(self.route(), &request, &encoded, &self.recipient).unwrap();
            ReverseOnionFrameV1::result(&self.claim, lease, &encode_onion_sealed_response(&sealed).unwrap(),
                NOW + 600, NOW + 2, &self.recipient).unwrap()
        }
        fn ready(&self, journal: &ReverseOnionSourceJournal) -> ReverseOnionFrameV1 {
            self.prepare(journal, NOW);
            journal.arm(self.route(), NOW + 1).unwrap();
            let result = self.response(&self.lease, [7; 32], false);
            journal.record_result(self.route(), &self.claim, &self.lease, &result, NOW + 2).unwrap();
            result
        }
    }

    #[test]
    fn prepared_restart_arms_exact_bytes_once_and_uncertain_send_never_rearms() {
        let f = Fixture::new();
        let journal = f.open(NOW); f.prepare(&journal, NOW);
        let exact = serde_json::to_vec(&f.outbound).unwrap();
        drop(journal);
        let journal = f.open(NOW + 1);
        assert_eq!(journal.arm(f.route(), NOW + 1).unwrap().exact_bytes, exact);
        assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Ambiguous));
        drop(journal);
        let journal = f.open(NOW + 2);
        assert_eq!(journal.recover(None, 1, NOW + 2).unwrap().items, vec![(f.route(), SourcePhase::DispatchAmbiguous)]);
        assert_eq!(journal.arm(f.route(), NOW + 2).err(), Some(SourceJournalError::Ambiguous));
        let result = f.response(&f.lease, [7; 32], false);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 2).unwrap();
        assert_eq!(journal.open_result(f.route(), NOW + 2).unwrap().lease_id, [7; 32]);
    }

    // [REVERSE-ONION-SOURCE-JOURNAL-V2 2026-10-04 by Codex] Recovery exposes
    // only immutable evidence-query metadata; the exact effectful body remains
    // available solely through the one-shot Prepared->Armed CAS.
    #[test]
    fn recovery_metadata_binds_exact_body_and_descriptor_authority() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        f.prepare(&journal, NOW);
        let page = journal.recover_metadata(None, 1, NOW + 1).unwrap();
        let item = &page.items[0];
        assert_eq!(item.route(), f.route());
        assert_eq!(item.phase(), SourcePhase::Prepared);
        assert_eq!(item.source(), f.source.public_key_bytes());
        assert_eq!(item.relay(), f.relay.public_key_bytes());
        assert_eq!(item.recipient(), f.recipient.public_key_bytes());
        assert_eq!(item.target(), f.recipient.public_key_bytes());
        assert_eq!(item.recipient_descriptor_commitment(), f.recipient_descriptor_commitment);
        assert_eq!(item.relay_descriptor_commitment(), f.relay_descriptor_commitment);
        assert_eq!(item.body_commitment(), hash(&serde_json::to_vec(&f.outbound).unwrap()));
        assert_ne!(item.request_commitment(), [0; 32]);
        assert!(item.deadline() <= item.retain_until());
        assert!(page.next_after.is_none());
    }

    #[test]
    fn historical_authority_survives_execution_expiry_during_result_grace() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        f.prepare(&journal, NOW);
        journal.arm(f.route(), NOW + 1).unwrap();
        let page = journal.recover_metadata(None, 1, NOW + 601).unwrap();
        assert_eq!(page.items[0].phase(), SourcePhase::Armed);
        assert!(page.items[0].deadline() < NOW + 601);
        assert!(page.items[0].retain_until() > NOW + 601);
    }

    #[test]
    fn request_commitment_and_purpose_drift_fail_before_prepare() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        let mut request_drift = f.plan();
        request_drift.request_commitment[0] ^= 1;
        assert_eq!(journal.prepare(request_drift, f.session(), NOW).err(), Some(SourceJournalError::Rejected));
        let mut purpose_drift = f.plan();
        purpose_drift.authority.purpose[0] ^= 1;
        assert_eq!(journal.prepare(purpose_drift, f.session(), NOW).err(), Some(SourceJournalError::Rejected));
    }

    fn authorization_after_envelope(f: &Fixture) -> SignedPrivateOnionRecipientAuthorizationV1 {
        let relay = Fixture::descriptor(&f.relay);
        let recipient = Fixture::descriptor(&f.recipient);
        SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay,
            &recipient,
            OnionRoutePurpose::BlindVaultPull.as_str(),
            NOW + 1,
            NOW + 9_001,
            &f.recipient,
        )
        .unwrap()
    }

    #[test]
    fn authorization_after_envelope_is_currently_valid_but_prepare_is_zero_mutation() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        let authorization = authorization_after_envelope(&f);
        authorization
            .verify_at(
                &Fixture::descriptor(&f.relay),
                &Fixture::descriptor(&f.recipient),
                OnionRoutePurpose::BlindVaultPull.as_str(),
                NOW + 2,
            )
            .unwrap();
        let mut plan = f.plan();
        plan.authority = SourceRouteAuthority::from_signed(
            &Fixture::descriptor(&f.relay),
            &Fixture::descriptor(&f.recipient),
            &authorization,
            OnionRoutePurpose::BlindVaultPull.as_str(),
        )
        .unwrap();
        assert_eq!(journal.prepare(plan, f.session(), NOW + 2).err(), Some(SourceJournalError::Rejected));
        let connection = Connection::open(&f.path()).unwrap();
        let count: i64 = connection.query_row("SELECT count(*) FROM source_jobs", [], |row| row.get(0)).unwrap();
        let clock: i64 = connection.query_row("SELECT clock FROM source_meta WHERE singleton=1", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 0);
        assert_eq!(clock, NOW as i64);
    }

    #[test]
    fn prepared_row_with_late_authorization_cannot_arm_or_rewrite() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        f.prepare(&journal, NOW);
        let authorization = authorization_after_envelope(&f);
        journal.with_inner(|inner| journal.transaction(inner, NOW + 1, |tx| {
            let mut row = journal.load(tx, f.route())?.ok_or(SourceJournalError::Corrupt)?;
            row.plan.authority.authorization = Zeroizing::new(authorization.encode_canonical().unwrap());
            let sealed = journal.seal(&row)?;
            one(tx.execute("UPDATE source_jobs SET sealed=?1 WHERE route=?2", params![sealed, f.route().as_slice()]).map_err(unavailable)?)
        })).unwrap();
        let (snapshot, before_clock) = journal.with_inner(|inner| {
            let snapshot = inner.connection.query_row(
                "SELECT phase,generation,reserved,retain_until,sealed FROM source_jobs WHERE route=?1",
                params![f.route().as_slice()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, Vec<u8>>(4)?)),
            ).map_err(unavailable)?;
            let clock = inner.connection.query_row(
                "SELECT clock FROM source_meta WHERE singleton=1", [], |row| row.get::<_, i64>(0),
            ).map_err(unavailable)?;
            Ok((snapshot, clock))
        }).unwrap();
        assert_eq!(journal.arm(f.route(), NOW + 2).err(), Some(SourceJournalError::Rejected));
        let (after, after_clock) = journal.with_inner(|inner| {
            let after = inner.connection.query_row(
                "SELECT phase,generation,reserved,retain_until,sealed FROM source_jobs WHERE route=?1",
                params![f.route().as_slice()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, Vec<u8>>(4)?)),
            ).map_err(unavailable)?;
            let clock = inner.connection.query_row(
                "SELECT clock FROM source_meta WHERE singleton=1", [], |row| row.get::<_, i64>(0),
            ).map_err(unavailable)?;
            Ok((after, clock))
        }).unwrap();
        assert_eq!(after, snapshot);
        assert_eq!(after_clock, before_clock);
    }

    // [REVERSE-ONION-SOURCE-V1-MIGRATION-FIXTURE 2026-10-04 by Codex]
    // Encode the genuine historical body shape, rather than only changing a
    // v2 tag. This proves the migration gate is reached before v2 validation.
    fn encode_v1_fixture(row: &Record) -> Zeroizing<Vec<u8>> {
        const V1_FIXED: usize = 4 + 2 + 1 + 8 + 8 + 8 + 32 + 16 + 32 * 4
            + 8 + 8 + 1 + 8 + 32 * 2 + 7 * 4;
        let fields: [&[u8]; 7] = [
            &row.plan.dispatch, &row.plan.terminal, &row.restart, &row.claim,
            &row.lease, &row.result, &row.verified,
        ];
        let mut bytes = Vec::with_capacity(V1_FIXED + fields.iter().map(|field| field.len()).sum::<usize>());
        bytes.extend_from_slice(b"AXSJ");
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.push(row.phase as u8);
        for value in [row.generation, row.reserved, row.retain_until] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&row.plan.source);
        bytes.extend_from_slice(&row.plan.expected.route);
        for value in [row.plan.target, row.plan.descriptor, row.plan.expected.relay, row.plan.expected.recipient] {
            bytes.extend_from_slice(&value);
        }
        bytes.extend_from_slice(&row.plan.deadline.to_be_bytes());
        bytes.extend_from_slice(&row.plan.original_expiry.to_be_bytes());
        bytes.push(row.plan.expected.ttl);
        bytes.extend_from_slice(&row.plan.expected.timestamp.to_be_bytes());
        bytes.extend_from_slice(&row.plan.expected.blob_hash);
        bytes.extend_from_slice(&row.plan.expected.signing_commitment);
        for field in fields {
            bytes.extend_from_slice(&(field.len() as u32).to_be_bytes());
            bytes.extend_from_slice(field);
        }
        Zeroizing::new(bytes)
    }

    #[test]
    fn historical_v1_payload_is_migration_required_without_rewrite() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        f.prepare(&journal, NOW);
        journal.with_inner(|inner| journal.transaction(inner, NOW + 1, |tx| {
            let row = journal.load(tx, f.route())?.ok_or(SourceJournalError::Corrupt)?;
            let clear = encode_v1_fixture(&row);
            let sealed = seal_blind_vault_source_pull_journal(&journal.identity, &clear)
                .map_err(|_| SourceJournalError::Corrupt)?;
            one(tx.execute("UPDATE source_jobs SET sealed=?1 WHERE route=?2",
                params![sealed, f.route().as_slice()]).map_err(unavailable)?)
        })).unwrap();
        drop(journal);
        assert_eq!(ReverseOnionSourceJournal::open(&f.path(), f.source.clone(),
            SourceJournalLimits { max_entries: 8, max_bytes: RESERVED_PER_JOB * 8 }, NOW + 2).err(),
            Some(SourceJournalError::MigrationRequired));
    }

    #[test]
    fn mixed_v2_and_v1_migration_failure_preserves_v2_phase_and_clock() {
        let f = Fixture::new();
        let journal = f.open(NOW);
        f.prepare(&journal, NOW);
        journal.arm(f.route(), NOW + 1).unwrap();
        let f2 = Fixture::new_with_route(12);
        journal.prepare(f2.plan(), f2.session(), NOW + 1).unwrap();
        journal.arm(f2.route(), NOW + 1).unwrap();
        let (sealed, phase, generation, reserved, retain_until) = journal.with_inner(|inner| {
            journal.transaction(inner, NOW + 1, |tx| {
                let row = journal.load(tx, f2.route())?.ok_or(SourceJournalError::Corrupt)?;
                let clear = encode_v1_fixture(&row);
                let sealed = seal_blind_vault_source_pull_journal(&journal.identity, &clear)
                    .map_err(|_| SourceJournalError::Corrupt)?;
                Ok((sealed, row.phase as u8, row.generation, row.reserved, row.retain_until))
            })
        }).unwrap();
        journal.with_inner(|inner| journal.transaction(inner, NOW + 1, |tx| {
            one(tx.execute(
                "UPDATE source_jobs SET sealed=?1 WHERE route=?2 AND phase=?3 AND generation=?4 AND reserved=?5 AND retain_until=?6",
                params![sealed, f2.route().as_slice(), phase, generation as i64, reserved as i64, retain_until as i64],
            ).map_err(unavailable)?)
        })).unwrap();
        drop(journal);
        let snapshot = |connection: &Connection| {
            let mut statement = connection.prepare(
                "SELECT route,phase,generation,reserved,retain_until,sealed FROM source_jobs ORDER BY route",
            ).unwrap();
            let rows = statement.query_map([], |row| Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Vec<u8>>(5)?,
            ) )).unwrap();
            rows.collect::<std::result::Result<Vec<_>, _>>().unwrap()
        };
        let (before_rows, before_clock) = {
            let before_connection = Connection::open(&f.path()).unwrap();
            let before_rows = snapshot(&before_connection);
            let before_clock: i64 = before_connection.query_row(
                "SELECT clock FROM source_meta WHERE singleton=1", [], |row| row.get(0),
            ).unwrap();
            (before_rows, before_clock)
        };
        assert_eq!(ReverseOnionSourceJournal::open(&f.path(), f.source.clone(),
            SourceJournalLimits { max_entries: 8, max_bytes: RESERVED_PER_JOB * 8 }, NOW + 2).err(),
            Some(SourceJournalError::MigrationRequired));
        let after_connection = Connection::open(&f.path()).unwrap();
        let after_rows = snapshot(&after_connection);
        let after_clock: i64 = after_connection.query_row(
            "SELECT clock FROM source_meta WHERE singleton=1", [], |row| row.get(0),
        ).unwrap();
        assert_eq!(after_rows, before_rows);
        assert_eq!(after_clock, before_clock);
    }

    #[test]
    fn result_ready_restart_verifies_and_reads_authenticated_cache_without_reopen() {
        let f = Fixture::new(); let journal = f.open(NOW); let result = f.ready(&journal);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 3).unwrap();
        drop(journal);
        let journal = f.open(NOW + 4);
        assert_eq!(journal.open_result(f.route(), NOW + 4).unwrap().lease_id, [7; 32]);
        assert_eq!(journal.open_result(f.route(), NOW + 4).err(), Some(SourceJournalError::Ambiguous));
        drop(journal);
        let journal = f.open(NOW + 5);
        assert_eq!(journal.read_verified(f.route(), NOW + 5).unwrap().lease_id, [7; 32]);
        assert_eq!(journal.open_result(f.route(), NOW + 5).err(), Some(SourceJournalError::Ambiguous));
    }

    #[test]
    fn crash_after_opening_commit_never_restores_session_again() {
        let f = Fixture::new(); let journal = f.open(NOW); let result = f.ready(&journal);
        journal.with_inner(|inner| journal.transaction(inner, NOW + 3, |tx| {
            let mut row = journal.load(tx, f.route())?.unwrap();
            journal.replace(tx, &mut row, SourcePhase::Opening)
        })).unwrap();
        drop(journal);
        let journal = f.open(NOW + 4);
        assert_eq!(journal.recover(None, 64, NOW + 4).unwrap().items, vec![(f.route(), SourcePhase::OpenAmbiguous)]);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 4).unwrap();
        assert_eq!(journal.open_result(f.route(), NOW + 4).err(), Some(SourceJournalError::Ambiguous));
        journal.with_inner(|inner| journal.transaction(inner, NOW + 4, |tx| {
            assert!(journal.load(tx, f.route())?.unwrap().restart.is_empty()); Ok(())
        })).unwrap();
    }

    #[test]
    fn wrong_terminal_page_is_terminal_rejection_not_retry_authority() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.arm(f.route(), NOW + 1).unwrap();
        let result = f.response(&f.lease, [99; 32], false);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 2).unwrap();
        assert_eq!(journal.open_result(f.route(), NOW + 2).err(), Some(SourceJournalError::ReplyRejected));
        drop(journal);
        let journal = f.open(NOW + 3);
        assert_eq!(journal.recover(None, 1, NOW + 3).unwrap().items, vec![(f.route(), SourcePhase::Rejected)]);
        assert_eq!(journal.open_result(f.route(), NOW + 3).err(), Some(SourceJournalError::Ambiguous));
    }

    #[test]
    fn relay_signed_ttl_timestamp_or_blob_substitution_has_zero_result_mutation() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.arm(f.route(), NOW + 1).unwrap();
        for change in 0..3 {
            let mut envelope = f.retained.clone();
            match change { 0 => envelope.ttl += 1, 1 => envelope.timestamp += 1,
                _ => { let last = envelope.encrypted_blob.len() - 1; envelope.encrypted_blob[last] ^= 1; } }
            let envelope = envelope.sign_with(&f.relay);
            let lease = ReverseOnionFrameV1::lease(&f.claim, &envelope, [13; 16], NOW + 600, NOW + 1, &f.relay).unwrap();
            let result = f.response(&lease, [7; 32], false);
            assert_eq!(journal.record_result(f.route(), &f.claim, &lease, &result, NOW + 2).err(), Some(SourceJournalError::Rejected));
            assert_eq!(journal.recover(None, 1, NOW + 2).unwrap().items, vec![(f.route(), SourcePhase::Armed)]);
        }
        let result = f.response(&f.lease, [7; 32], false);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 2).unwrap();
        let different = f.response(&f.lease, [7; 32], false);
        assert_eq!(journal.record_result(f.route(), &f.claim, &f.lease, &different, NOW + 2).err(), Some(SourceJournalError::Conflict));
    }

    #[test]
    fn exact_prepare_retry_precedes_full_quota_but_changed_claims_conflict() {
        let f = Fixture::new(); let journal = f.open_limits(NOW, 1, RESERVED_PER_JOB).unwrap();
        f.prepare(&journal, NOW); f.prepare(&journal, NOW + 1);
        let mut changed = f.plan(); changed.descriptor = [15; 32];
        assert_eq!(journal.prepare(changed, f.session(), NOW + 1).err(), Some(SourceJournalError::Conflict));
        let other = Fixture::new_with_route(15);
        assert_eq!(journal.prepare(other.plan(), other.session(), NOW + 1).err(), Some(SourceJournalError::Capacity));
        assert_eq!(journal.recover(None, 1, NOW + 1).unwrap().items.len(), 1);
        drop(journal);
        assert_eq!(f.open_limits(NOW + 2, 1, RESERVED_PER_JOB - 1).err(), Some(SourceJournalError::Capacity));
    }

    #[test]
    fn full_reservation_allows_maximum_fixed_class_page_and_restart() {
        let f = Fixture::new(); let journal = f.open_limits(NOW, 1, RESERVED_PER_JOB).unwrap();
        f.prepare(&journal, NOW); journal.arm(f.route(), NOW + 1).unwrap();
        let result = f.response(&f.lease, [7; 32], true);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 2).unwrap();
        let page = journal.open_result(f.route(), NOW + 2).unwrap();
        assert_eq!(page.objects.len(), 1);
        journal.with_inner(|inner| journal.transaction(inner, NOW + 2, |tx| {
            let row = journal.load(tx, f.route())?.unwrap();
            let actual = row.encode()?.len() + SEALED_OVERHEAD + ROW_ACCOUNTING_BYTES;
            assert!(actual as u64 <= row.reserved);
            assert!(row.encode()?.len() <= MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_BODY_BYTES);
            Ok(())
        })).unwrap();
        drop(journal);
        let journal = f.open_limits(NOW + 3, 1, RESERVED_PER_JOB).unwrap();
        assert_eq!(journal.read_verified(f.route(), NOW + 3).unwrap().objects.len(), 1);
    }

    #[test]
    fn concurrent_arm_and_open_each_have_one_successful_owner() {
        let f = Fixture::new(); let journal = Arc::new(f.open(NOW)); f.prepare(&journal, NOW);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut workers = vec![];
        for _ in 0..2 {
            let journal = journal.clone(); let barrier = barrier.clone(); let route = f.route();
            workers.push(std::thread::spawn(move || { barrier.wait(); journal.arm(route, NOW + 1).is_ok() }));
        }
        assert_eq!(workers.into_iter().map(|w| usize::from(w.join().unwrap())).sum::<usize>(), 1);
        let result = f.response(&f.lease, [7; 32], false);
        journal.record_result(f.route(), &f.claim, &f.lease, &result, NOW + 2).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2)); let mut workers = vec![];
        for _ in 0..2 {
            let journal = journal.clone(); let barrier = barrier.clone(); let route = f.route();
            workers.push(std::thread::spawn(move || { barrier.wait(); journal.open_result(route, NOW + 3).is_ok() }));
        }
        assert_eq!(workers.into_iter().map(|w| usize::from(w.join().unwrap())).sum::<usize>(), 1);
    }

    #[test]
    fn corruption_clock_rollback_and_reserved_counter_drift_fail_closed() {
        for mutation in 0..4 {
            let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
            if mutation == 0 {
                assert_eq!(journal.recover(None, 1, NOW - 1).err(), Some(SourceJournalError::ClockRollback));
            } else {
                journal.with_inner(|inner| {
                    match mutation {
                        1 => { inner.connection.execute("UPDATE source_jobs SET sealed=zeroblob(?1)",
                            params![(MAX_BLIND_VAULT_SOURCE_PULL_JOURNAL_SEALED_BYTES + 1) as i64]).map_err(unavailable)?; }
                        2 => { inner.connection.execute("UPDATE source_jobs SET reserved=reserved-1", []).map_err(unavailable)?; }
                        _ => { inner.connection.execute("UPDATE source_jobs SET generation=generation+1", []).map_err(unavailable)?; }
                    }
                    Ok(())
                }).unwrap();
                assert_eq!(journal.recover(None, 1, NOW).err(), Some(SourceJournalError::Corrupt));
            }
            assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Unavailable));
        }
    }

    #[test]
    fn private_database_excludes_second_handle_and_rejects_link_aliases() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        assert!(f.open_limits(NOW, 8, RESERVED_PER_JOB * 8).is_err());
        drop(journal);
        std::fs::set_permissions(f.path(), std::fs::Permissions::from_mode(0o640)).unwrap();
        let alias = f.directory.path().join("alias.sqlite");
        std::fs::hard_link(f.path(), &alias).unwrap();
        assert!(ReverseOnionSourceJournal::open(&alias, f.source.clone(),
            SourceJournalLimits { max_entries: 8, max_bytes: RESERVED_PER_JOB * 8 }, NOW).is_err());
        assert_eq!(std::fs::metadata(f.path()).unwrap().permissions().mode() & 0o777, 0o640);
        let link = f.directory.path().join("parent-link");
        symlink(f.directory.path(), &link).unwrap();
        assert!(ReverseOnionSourceJournal::open(&link.join("new.sqlite"), f.source.clone(),
            SourceJournalLimits { max_entries: 8, max_bytes: RESERVED_PER_JOB * 8 }, NOW).is_err());
        assert!(!f.directory.path().join("new.sqlite").exists());
    }

    #[test]
    fn bounded_cleanup_keeps_evidence_until_horizon_and_rejects_stale_prepare() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        assert_eq!(journal.cleanup(1, NOW + 899).unwrap(), 0);
        assert_eq!(journal.cleanup(1, NOW + 900).unwrap(), 1);
        assert_eq!(journal.prepare(f.plan(), f.session(), NOW + 900).err(), Some(SourceJournalError::Expired));
        assert_eq!(journal.recover(None, 1, NOW + 900).unwrap().items.len(), 0);
        assert_eq!(journal.recover(None, 65, NOW + 900).err(), Some(SourceJournalError::Rejected));
        // This does not promise permanent route-id exclusion after deletion.
    }

    #[test]
    fn sealed_record_binding_rejects_swapped_route_and_unknown_schema() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.with_inner(|inner| {
            inner.connection.execute("UPDATE source_jobs SET route=?1", params![[90u8; 16].as_slice()]).map_err(unavailable)?;
            Ok(())
        }).unwrap();
        assert_eq!(journal.recover(None, 1, NOW).err(), Some(SourceJournalError::Corrupt));
        drop(journal);
        let connection = Connection::open(f.path()).unwrap();
        connection.pragma_update(None, "user_version", 2).unwrap(); drop(connection);
        assert_eq!(f.open_limits(NOW, 8, RESERVED_PER_JOB * 8).err(), Some(SourceJournalError::Corrupt));
    }

    #[test]
    fn uncertain_arm_commit_returns_no_dispatch_and_restart_is_ambiguous() {
        use std::sync::atomic::Ordering;
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.commits_until_fence_error.store(1, Ordering::SeqCst);
        assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Unavailable));
        assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Unavailable));
        drop(journal);
        let journal = f.open(NOW + 2);
        assert_eq!(journal.recover(None, 1, NOW + 2).unwrap().items[0].1, SourcePhase::DispatchAmbiguous);
        assert_eq!(journal.arm(f.route(), NOW + 2).err(), Some(SourceJournalError::Ambiguous));
    }

    #[test]
    fn uncertainty_at_opening_or_final_commit_never_grants_second_open() {
        use std::sync::atomic::Ordering;
        for failure_commit in [1, 2] {
            let f = Fixture::new(); let journal = f.open(NOW); f.ready(&journal);
            journal.commits_until_fence_error.store(failure_commit, Ordering::SeqCst);
            assert_eq!(journal.open_result(f.route(), NOW + 3).err(), Some(SourceJournalError::Unavailable));
            drop(journal);
            let journal = f.open(NOW + 4);
            let phase = journal.recover(None, 1, NOW + 4).unwrap().items[0].1;
            assert_eq!(phase, if failure_commit == 1 { SourcePhase::OpenAmbiguous } else { SourcePhase::Verified });
            assert_eq!(journal.open_result(f.route(), NOW + 4).err(), Some(SourceJournalError::Ambiguous));
            if phase == SourcePhase::Verified {
                assert_eq!(journal.read_verified(f.route(), NOW + 4).unwrap().lease_id, [7; 32]);
            }
        }
    }

    #[test]
    fn bounded_cursor_progresses_past_older_rows() {
        let first = Fixture::new_with_route(1); let second = Fixture::new_with_route(2);
        let third = Fixture::new_with_route(3); let journal = first.open(NOW);
        for fixture in [&first, &second, &third] { fixture.prepare(&journal, NOW); }
        let page = journal.recover(None, 1, NOW).unwrap();
        assert_eq!(page.items[0].0, first.route());
        let page = journal.recover(page.next_after, 1, NOW).unwrap();
        assert_eq!(page.items[0].0, second.route());
        let page = journal.recover(page.next_after, 1, NOW).unwrap();
        assert_eq!(page.items[0].0, third.route()); assert!(page.next_after.is_none());
        assert_eq!(journal.cleanup(1, NOW + 900).unwrap(), 1);
        assert_eq!(journal.recover(None, 64, NOW + 900).unwrap().items.len(), 2);
    }

    // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] Authored only.
    #[test]
    fn typed_projection_matches_same_pass_relay_bytes_and_preserves_record_schema() {
        let f = Fixture::new();
        f.expectation.verify_relay_produced_envelope(&f.retained).unwrap();
        let typed = f.expected();
        assert!(typed.matches(&f.retained));
        let raw_control = ExpectedRetainedEnvelope::from_route_construction(
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), f.route(), f.retained.ttl,
            f.retained.timestamp, &f.retained.encrypted_blob,
        ).unwrap();
        let typed_plan = f.plan(); let mut control_plan = f.plan(); control_plan.expected = raw_control;
        assert!(typed_plan.same(&control_plan));
        assert_eq!(typed.blob_hash, hash(&f.retained.encrypted_blob));
        let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.with_inner(|inner| journal.transaction(inner, NOW, |tx| {
            let row = journal.load(tx, f.route())?.unwrap();
            let encoded = row.encode()?;
            assert_eq!(&encoded[..6], b"AXSJ\x00\x01");
            let restored = Record::decode(&encoded)?;
            assert!(restored.plan.same(&typed_plan));
            assert!(restored.plan.expected.matches(&f.retained));
            Ok(())
        })).unwrap();
    }

    #[test]
    fn source_rejects_zero_or_inconsistent_projection_context_before_dispatch() {
        let f = Fixture::new(); let journal = f.open(NOW);
        for change in 0..5 {
            let mut plan = f.plan();
            match change {
                0 => plan.expected.timestamp = 0,
                1 => plan.expected.timestamp = u64::MAX,
                2 => plan.expected.route = [0; 16],
                3 => plan.expected.ttl += 1,
                _ => plan.expected.blob_hash = [0; 32],
            }
            assert_eq!(journal.prepare(plan, f.session(), NOW).err(), Some(SourceJournalError::Rejected));
        }
        assert!(journal.recover(None, 1, NOW).unwrap().items.is_empty());
    }

    // [REVERSE-ONION-SOURCE-DB-BOUNDARY 2026-10-04 by Codex] Authored only;
    // sparse fixture files exercise metadata bounds without large allocations.
    fn private_sparse(path: &Path, bytes: u64) {
        std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600)
            .open(path).unwrap().set_len(bytes).unwrap();
    }

    #[test]
    fn source_physical_preflight_rejects_individual_under_but_aggregate_over() {
        let f = Fixture::new();
        let limits = SourceJournalLimits { max_entries: 1, max_bytes: RESERVED_PER_JOB };
        let bound = source_physical_limit(&limits).unwrap();
        let primary_bytes = bound * 3 / 4; let rollback_bytes = bound / 2;
        let rollback = source_sidecar(&f.path(), "-journal");
        private_sparse(&f.path(), primary_bytes); private_sparse(&rollback, rollback_bytes);
        let parent_before = std::fs::metadata(f.directory.path()).unwrap();
        assert!(primary_bytes < bound && rollback_bytes < bound);
        assert_eq!(f.open_limits(NOW, 1, RESERVED_PER_JOB).err(), Some(SourceJournalError::Capacity));
        assert_eq!(std::fs::metadata(f.path()).unwrap().len(), primary_bytes);
        assert_eq!(std::fs::metadata(&rollback).unwrap().len(), rollback_bytes);
        assert_eq!(std::fs::metadata(f.directory.path()).unwrap().mode(), parent_before.mode());
        assert_eq!(std::fs::metadata(f.path()).unwrap().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(rollback).unwrap().mode() & 0o777, 0o600);
    }

    #[test]
    fn source_absent_primary_oversized_rollback_refusal_has_no_dirent_or_mode_effect() {
        let f = Fixture::new();
        let bound = source_physical_limit(&SourceJournalLimits { max_entries: 1, max_bytes: RESERVED_PER_JOB }).unwrap();
        let rollback = source_sidecar(&f.path(), "-journal");
        private_sparse(&rollback, bound + 1);
        let parent_before = std::fs::metadata(f.directory.path()).unwrap();
        let journal_before = std::fs::metadata(&rollback).unwrap();
        let names_before: Vec<_> = std::fs::read_dir(f.directory.path()).unwrap()
            .map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(f.open_limits(NOW, 1, RESERVED_PER_JOB).err(), Some(SourceJournalError::Capacity));
        assert!(!f.path().exists());
        let names_after: Vec<_> = std::fs::read_dir(f.directory.path()).unwrap()
            .map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(names_before, names_after);
        assert_eq!(std::fs::metadata(f.directory.path()).unwrap().mode(), parent_before.mode());
        let after = std::fs::metadata(&rollback).unwrap();
        assert_eq!(after.mode(), journal_before.mode()); assert_eq!(after.len(), journal_before.len());
    }

    #[test]
    fn source_post_fence_observed_aggregate_failure_poisons_before_publication() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        let primary = std::fs::metadata(f.path()).unwrap().len();
        assert!(primary > 1 && primary < journal.physical_limit);
        let rollback = source_sidecar(&f.path(), "-journal");
        private_sparse(&rollback, journal.physical_limit - 1);
        let publication = journal.with_inner(|inner| {
            journal.post_operation_fence(&inner.connection).map_err(|_| SourceJournalError::Unavailable)?;
            Ok(())
        });
        assert_eq!(publication.err(), Some(SourceJournalError::Unavailable));
        assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Unavailable));
        assert!(rollback.exists());
    }

    #[test]
    fn source_critical_pragma_readback_detects_fullfsync_drift_before_dispatch_publication() {
        let f = Fixture::new(); let journal = f.open(NOW); f.prepare(&journal, NOW);
        journal.with_inner(|inner| {
            audit_source_pragmas(&inner.connection, journal.physical_limit)?;
            inner.connection.pragma_update(None, "fullfsync", 0).map_err(unavailable)
        }).unwrap();
        assert_eq!(journal.arm(f.route(), NOW + 1).err(), Some(SourceJournalError::Unavailable));
        assert_eq!(journal.recover(None, 1, NOW + 1).err(), Some(SourceJournalError::Unavailable));
        drop(journal);
        let journal = f.open(NOW + 2);
        assert_eq!(journal.recover(None, 1, NOW + 2).unwrap().items[0].1, SourcePhase::DispatchAmbiguous);
    }

    #[test]
    fn source_aggregate_overflow_and_forbidden_sidecars_fail_closed_without_creation() {
        assert_eq!(source_checked_aggregate(u64::MAX, 1, u64::MAX).err(), Some(SourceJournalError::Capacity));
        assert_eq!(source_checked_aggregate(4, 5, 9).unwrap(), 9);
        for suffix in ["-wal", "-shm"] {
            let f = Fixture::new(); let sidecar = source_sidecar(&f.path(), suffix);
            private_sparse(&sidecar, 1);
            assert_eq!(f.open_limits(NOW, 1, RESERVED_PER_JOB).err(), Some(SourceJournalError::Corrupt));
            assert!(!f.path().exists()); assert_eq!(std::fs::metadata(sidecar).unwrap().len(), 1);
        }
    }
}
