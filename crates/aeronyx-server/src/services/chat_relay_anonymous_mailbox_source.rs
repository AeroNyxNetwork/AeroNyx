// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source.rs
// ============================================
//! # Exact-target anonymous-mailbox source coordinator
//!
//! This default-off building block prepares one immutable blind-relay request
//! for one receiver-provided, descriptor-pinned custody node. It does not own
//! HTTP, server startup, client APIs, discovery, or any participant identity.
//! All methods are synchronous deliberately: a future composition root must
//! invoke the SQLite-backed journal through an explicit blocking boundary.

use std::sync::Arc;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxOperationV1, AnonymousMailboxPullResultV1, AnonymousMailboxRouteRequestV1,
    AnonymousMailboxSourceSealSessionV1, AnonymousMailboxSourceTerminalCarrierV1,
    AnonymousMailboxTerminalFrameV1, AnonymousMailboxTerminalResponseV1,
};
use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{OnionRoutePurpose, VerifiedOnionRoute};
use chacha20poly1305::aead::{Aead, NewAead, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use sha2::{Digest, Sha256};

use crate::api::chat_peer::{prepare_exact_peer_blind_relay_http_request, PeerBlindRelayRequest};
use crate::config_chat_relay::AnonymousMailboxSourceConfig;

use super::peer_store::PeerStore;

const JOURNAL_STATE_VERSION: u8 = 1;
const JOURNAL_NONCE_BYTES: usize = 24;
const JOURNAL_AEAD_TAG_BYTES: usize = 16;
const MAX_JOURNAL_DESCRIPTOR_BYTES: usize = 1024;
const MAX_JOURNAL_BODY_BYTES: usize = 2 * 1024 * 1024;
const JOURNAL_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-journal/v1\0";
const REQUEST_COMMITMENT_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-request/v1\0";

/// Coarse source coordinator failure. No variant carries an opaque request,
/// descriptor, identity, path, or capability value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AnonymousMailboxSourceError {
    #[error("anonymous mailbox source disabled")]
    Disabled,
    #[error("anonymous mailbox source rejected")]
    Rejected,
    #[error("anonymous mailbox source conflict")]
    Conflict,
    #[error("anonymous mailbox source journal unavailable")]
    Unavailable,
    #[error("anonymous mailbox source journal corrupt")]
    Corrupt,
    #[error("anonymous mailbox source result ambiguous")]
    Ambiguous,
}

/// Durable request lifecycle. Armed and ambiguous records may only retry their
/// byte-identical prepared carrier; callers never synthesize a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnonymousMailboxSourcePhase {
    Prepared,
    Armed,
    Completed,
    Ambiguous,
    Rejected,
}

impl AnonymousMailboxSourcePhase {
    fn code(self) -> i64 {
        match self {
            Self::Prepared => 1,
            Self::Armed => 2,
            Self::Completed => 3,
            Self::Ambiguous => 4,
            Self::Rejected => 5,
        }
    }

    fn decode(code: i64) -> Result<Self, AnonymousMailboxSourceError> {
        match code {
            1 => Ok(Self::Prepared),
            2 => Ok(Self::Armed),
            3 => Ok(Self::Completed),
            4 => Ok(Self::Ambiguous),
            5 => Ok(Self::Rejected),
            _ => Err(AnonymousMailboxSourceError::Corrupt),
        }
    }
}

/// Receiver-shared exact custody pin. The full descriptor commitment prevents
/// a source from silently selecting a fresher or otherwise different peer.
///
/// [ANONYMOUS-MAILBOX-SOURCE 2026-09-03 by Codex] This type intentionally has
/// no Debug implementation: callers must not accidentally log target metadata.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ExactAnonymousMailboxTargetPin {
    target_node_id: [u8; 32],
    descriptor_commitment: DirectoryDescriptorCommitmentV1,
}

impl ExactAnonymousMailboxTargetPin {
    #[must_use]
    pub(crate) const fn new(
        target_node_id: [u8; 32],
        descriptor_commitment: DirectoryDescriptorCommitmentV1,
    ) -> Self {
        Self {
            target_node_id,
            descriptor_commitment,
        }
    }
}

/// Narrow exact-node lookup boundary. The production adapter calls only
/// `PeerStore::get_valid`; no iterator or fallback candidate API is exposed.
pub(crate) trait ExactAnonymousMailboxTargetResolver: Send + Sync {
    fn get_valid_exact(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor>;
}

impl ExactAnonymousMailboxTargetResolver for PeerStore {
    fn get_valid_exact(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor> {
        self.get_valid(node_id, now)
    }
}

/// Future HTTP composition boundary. The coordinator gives it exactly the
/// pre-journaled JSON bytes and cannot ask it to construct a second request.
pub(crate) trait AnonymousMailboxSourceDispatch: Send + Sync {
    fn dispatch_exact(&self, body: &[u8]) -> Result<Vec<u8>, AnonymousMailboxSourceError>;
}

/// Opaque prepared result exposed to a future transport composition root.
/// It intentionally has no Debug implementation because it retains ciphertext.
pub(crate) struct AnonymousMailboxSourcePrepared {
    route_id: [u8; 16],
    body: Vec<u8>,
    phase: AnonymousMailboxSourcePhase,
}

impl AnonymousMailboxSourcePrepared {
    #[must_use]
    pub(crate) const fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    #[must_use]
    pub(crate) fn body(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub(crate) const fn phase(&self) -> AnonymousMailboxSourcePhase {
        self.phase
    }
}

struct SourceJournalRecord {
    route_id: [u8; 16],
    request_commitment: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_commitment: DirectoryDescriptorCommitmentV1,
    body: Vec<u8>,
    phase: AnonymousMailboxSourcePhase,
    state: Vec<u8>,
}

/// SQLite implementation of the source journal. Construction takes an
/// already-private/opened connection; this slice neither chooses a disk path
/// nor performs filesystem work on a Tokio runtime.
pub(crate) struct SqliteAnonymousMailboxSourceJournal {
    connection: Mutex<Connection>,
    journal_key: [u8; 32],
    max_entries: usize,
    max_bytes: u64,
}

impl SqliteAnonymousMailboxSourceJournal {
    pub(crate) fn new(
        connection: Connection,
        journal_key: [u8; 32],
        config: &AnonymousMailboxSourceConfig,
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !config.enabled || config.max_journal_entries == 0 || config.max_journal_bytes == 0 {
            return Err(AnonymousMailboxSourceError::Disabled);
        }
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS anonymous_mailbox_source_journal (
                    route_id BLOB PRIMARY KEY NOT NULL,
                    request_commitment BLOB NOT NULL,
                    target_node_id BLOB NOT NULL,
                    descriptor_commitment BLOB NOT NULL,
                    body BLOB NOT NULL,
                    phase INTEGER NOT NULL,
                    state_nonce BLOB NOT NULL,
                    protected_state BLOB NOT NULL
                );",
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        Ok(Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
        })
    }

    fn load(
        &self,
        route_id: &[u8; 16],
    ) -> Result<Option<SourceJournalRecord>, AnonymousMailboxSourceError> {
        let connection = self.connection.lock();
        let lengths: Option<(i64, i64, i64, i64, i64, i64)> = connection
            .query_row(
                "SELECT length(request_commitment), length(target_node_id),
                        length(descriptor_commitment), length(body), length(state_nonce),
                        length(protected_state)
                 FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![route_id.as_slice()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let Some((request_len, target_len, descriptor_len, body_len, nonce_len, protected_len)) =
            lengths
        else {
            return Ok(None);
        };
        let max_protected = usize::try_from(self.max_bytes)
            .unwrap_or(usize::MAX)
            .saturating_add(JOURNAL_AEAD_TAG_BYTES);
        if request_len != 32
            || target_len != 32
            || descriptor_len < 1
            || usize::try_from(descriptor_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_DESCRIPTOR_BYTES)
                .is_none()
            || body_len < 1
            || usize::try_from(body_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
                .is_none()
            || nonce_len != JOURNAL_NONCE_BYTES as i64
            || protected_len < JOURNAL_AEAD_TAG_BYTES as i64
            || usize::try_from(protected_len)
                .ok()
                .filter(|value| *value <= max_protected)
                .is_none()
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        connection
            .query_row(
                "SELECT request_commitment, target_node_id, descriptor_commitment, body, phase,
                        state_nonce, protected_state
                 FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![route_id.as_slice()],
                |row| {
                    let request_commitment: Vec<u8> = row.get(0)?;
                    let target_node_id: Vec<u8> = row.get(1)?;
                    let descriptor: Vec<u8> = row.get(2)?;
                    let body: Vec<u8> = row.get(3)?;
                    let phase: i64 = row.get(4)?;
                    let nonce: Vec<u8> = row.get(5)?;
                    let protected: Vec<u8> = row.get(6)?;
                    Ok((
                        request_commitment,
                        target_node_id,
                        descriptor,
                        body,
                        phase,
                        nonce,
                        protected,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?
            .map(
                |(request, target, descriptor, body, phase, nonce, protected)| {
                    let request_commitment = fixed::<32>(&request)?;
                    let target_node_id = fixed::<32>(&target)?;
                    let descriptor_commitment = bincode::deserialize(&descriptor)
                        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
                    let phase = AnonymousMailboxSourcePhase::decode(phase)?;
                    let state = self.open_state(
                        route_id,
                        &request_commitment,
                        &target_node_id,
                        &nonce,
                        &protected,
                    )?;
                    Ok(SourceJournalRecord {
                        route_id: *route_id,
                        request_commitment,
                        target_node_id,
                        descriptor_commitment,
                        body,
                        phase,
                        state,
                    })
                },
            )
            .transpose()
    }

    fn insert_or_exact(
        &self,
        record: &SourceJournalRecord,
    ) -> Result<SourceJournalRecord, AnonymousMailboxSourceError> {
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT request_commitment FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![record.route_id.as_slice()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if let Some(existing) = existing {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            if fixed::<32>(&existing)? != record.request_commitment {
                return Err(AnonymousMailboxSourceError::Conflict);
            }
            drop(connection);
            let loaded = self
                .load(&record.route_id)?
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            if loaded.target_node_id != record.target_node_id
                || loaded.descriptor_commitment != record.descriptor_commitment
            {
                return Err(AnonymousMailboxSourceError::Conflict);
            }
            return Ok(loaded);
        }
        let entries: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM anonymous_mailbox_source_journal",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let used: i64 = transaction
            .query_row(
                "SELECT COALESCE(SUM(length(body) + length(protected_state)), 0)
                 FROM anonymous_mailbox_source_journal",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let (nonce, protected) = self.seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &record.state,
        )?;
        let incoming = record
            .body
            .len()
            .checked_add(protected.len())
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        if entries < 0
            || usize::try_from(entries).map_err(|_| AnonymousMailboxSourceError::Corrupt)?
                >= self.max_entries
            || u64::try_from(used)
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?
                .checked_add(
                    u64::try_from(incoming).map_err(|_| AnonymousMailboxSourceError::Rejected)?,
                )
                .ok_or(AnonymousMailboxSourceError::Rejected)?
                > self.max_bytes
        {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let descriptor = bincode::serialize(&record.descriptor_commitment)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body, phase,
                    state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    record.route_id.as_slice(), record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(), descriptor, record.body,
                    record.phase.code(), nonce, protected
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        Ok(SourceJournalRecord {
            route_id: record.route_id,
            request_commitment: record.request_commitment,
            target_node_id: record.target_node_id,
            descriptor_commitment: record.descriptor_commitment,
            body: record.body.clone(),
            phase: record.phase,
            state: record.state.clone(),
        })
    }

    fn transition(
        &self,
        record: &SourceJournalRecord,
        expected: AnonymousMailboxSourcePhase,
        phase: AnonymousMailboxSourcePhase,
        state: Vec<u8>,
    ) -> Result<(), AnonymousMailboxSourceError> {
        let (nonce, protected) = self.seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &state,
        )?;
        let connection = self.connection.lock();
        let updated = connection
            .execute(
                "UPDATE anonymous_mailbox_source_journal
                 SET phase = ?1, state_nonce = ?2, protected_state = ?3
                 WHERE route_id = ?4 AND request_commitment = ?5 AND phase = ?6",
                params![
                    phase.code(),
                    nonce,
                    protected,
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    expected.code()
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if updated == 1 {
            Ok(())
        } else {
            Err(AnonymousMailboxSourceError::Ambiguous)
        }
    }

    fn seal_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        state: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), AnonymousMailboxSourceError> {
        let mut nonce = [0u8; JOURNAL_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.journal_key));
        let protected = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: state,
                    aad: &journal_aad(route_id, request_commitment, target),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        Ok((nonce.to_vec(), protected))
    }

    fn open_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        nonce: &[u8],
        protected: &[u8],
    ) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
        let nonce = fixed::<JOURNAL_NONCE_BYTES>(nonce)?;
        XChaCha20Poly1305::new(Key::from_slice(&self.journal_key))
            .decrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: protected,
                    aad: &journal_aad(route_id, request_commitment, target),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)
    }
}

/// Default-off exact-target source composition. It has no network ownership;
/// callers use `dispatch` with a separately injected bounded transport.
pub(crate) struct AnonymousMailboxSourceCoordinator {
    source_identity: Arc<IdentityKeyPair>,
    resolver: Arc<dyn ExactAnonymousMailboxTargetResolver>,
    journal: Arc<SqliteAnonymousMailboxSourceJournal>,
}

impl AnonymousMailboxSourceCoordinator {
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
        ensure_request_frame(&terminal_frame)?;
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
        let state = encode_state(&terminal_frame, Some(&session), None)?;
        let record = SourceJournalRecord {
            route_id,
            request_commitment,
            target_node_id: pin.target_node_id,
            descriptor_commitment: pin.descriptor_commitment,
            body,
            phase: AnonymousMailboxSourcePhase::Prepared,
            state,
        };
        let record = self.journal.insert_or_exact(&record)?;
        Ok(AnonymousMailboxSourcePrepared {
            route_id,
            body: record.body,
            phase: record.phase,
        })
    }

    /// Arms the exact retained request before transport. A transport failure
    /// leaves it armed and retryable only with its stored bytes.
    pub(crate) fn dispatch(
        &self,
        route_id: [u8; 16],
        transport: &dyn AnonymousMailboxSourceDispatch,
    ) -> Result<(), AnonymousMailboxSourceError> {
        let record = self
            .journal
            .load(&route_id)?
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        match record.phase {
            AnonymousMailboxSourcePhase::Prepared => self.journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                record.state.clone(),
            )?,
            AnonymousMailboxSourcePhase::Armed => {}
            AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected => {
                return Ok(())
            }
            AnonymousMailboxSourcePhase::Ambiguous => {
                return Err(AnonymousMailboxSourceError::Ambiguous)
            }
        }
        let response = transport.dispatch_exact(&record.body)?;
        self.open_response(route_id, &response)
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
        let (terminal_frame, restart, _) = decode_state(&record.state)?;
        let mut session = restart.ok_or(AnonymousMailboxSourceError::Corrupt)?;
        let opened = session.open(encoded_response);
        let response_frame =
            match opened.and_then(|bytes| decode_anonymous_mailbox_terminal_frame(&bytes)) {
                Ok(frame) => frame,
                Err(_) => {
                    self.journal.transition(
                        &record,
                        AnonymousMailboxSourcePhase::Armed,
                        AnonymousMailboxSourcePhase::Ambiguous,
                        encode_state(&terminal_frame, None, None)?,
                    )?;
                    return Err(AnonymousMailboxSourceError::Ambiguous);
                }
            };
        if verify_response(&terminal_frame, &response_frame, &record.target_node_id).is_err() {
            self.journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Armed,
                AnonymousMailboxSourcePhase::Ambiguous,
                encode_state(&terminal_frame, None, None)?,
            )?;
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        let completed = encode_anonymous_mailbox_terminal_frame(&response_frame)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        self.journal.transition(
            &record,
            AnonymousMailboxSourcePhase::Armed,
            AnonymousMailboxSourcePhase::Completed,
            encode_state(&terminal_frame, None, Some(&completed))?,
        )
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

fn ensure_request_frame(frame: &[u8]) -> Result<(), AnonymousMailboxSourceError> {
    match decode_anonymous_mailbox_terminal_frame(frame)
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?
    {
        AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
        | AnonymousMailboxTerminalFrameV1::Put(_)
        | AnonymousMailboxTerminalFrameV1::PullOne(_)
        | AnonymousMailboxTerminalFrameV1::Ack(_)
        | AnonymousMailboxTerminalFrameV1::TicketIssue(_) => Ok(()),
        _ => Err(AnonymousMailboxSourceError::Rejected),
    }
}

fn verify_response(
    request: &[u8],
    response: &AnonymousMailboxTerminalFrameV1,
    target: &[u8; 32],
) -> Result<(), AnonymousMailboxSourceError> {
    let request = decode_anonymous_mailbox_terminal_frame(request)
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    match (request, response) {
        (
            AnonymousMailboxTerminalFrameV1::LeaseCreate(request),
            AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::LeaseCreate,
            request.admission.ticket_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::Put(request),
            AnonymousMailboxTerminalFrameV1::PutResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::Put,
            request.item_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::PullOne(request),
            AnonymousMailboxTerminalFrameV1::PullOneResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::PullOne,
            request.request_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::Ack(request),
            AnonymousMailboxTerminalFrameV1::AckResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::Ack,
            request.request_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::TicketIssue(request),
            AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
        ) => response
            .verify_for_request(&request, target)
            .map_err(|_| AnonymousMailboxSourceError::Rejected),
        _ => Err(AnonymousMailboxSourceError::Rejected),
    }
}

fn verify_standard_response(
    operation: AnonymousMailboxOperationV1,
    request_id: [u8; 16],
    commitment: [u8; 32],
    response: &AnonymousMailboxTerminalResponseV1,
    target: &[u8; 32],
) -> Result<(), AnonymousMailboxSourceError> {
    response
        .verify_for_request(operation, &request_id, &commitment, target)
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    match operation {
        AnonymousMailboxOperationV1::LeaseCreate
        | AnonymousMailboxOperationV1::Put
        | AnonymousMailboxOperationV1::Ack => {
            if response.sealed_payload.is_empty() {
                Ok(())
            } else {
                Err(AnonymousMailboxSourceError::Rejected)
            }
        }
        AnonymousMailboxOperationV1::PullOne => {
            if response.sealed_payload.is_empty() {
                Ok(())
            } else {
                AnonymousMailboxPullResultV1::decode(&response.sealed_payload)
                    .map(|_| ())
                    .map_err(|_| AnonymousMailboxSourceError::Rejected)
            }
        }
        AnonymousMailboxOperationV1::TicketIssue => Err(AnonymousMailboxSourceError::Rejected),
    }
}

fn source_request_commitment(
    route_id: &[u8; 16],
    target: &[u8; 32],
    descriptor: &DirectoryDescriptorCommitmentV1,
    terminal_frame: &[u8],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(REQUEST_COMMITMENT_DOMAIN);
    hash.update(route_id);
    hash.update(target);
    hash.update(descriptor.hash());
    hash.update((terminal_frame.len() as u64).to_be_bytes());
    hash.update(terminal_frame);
    hash.finalize().into()
}

fn journal_aad(route_id: &[u8; 16], commitment: &[u8; 32], target: &[u8; 32]) -> Vec<u8> {
    let mut aad = Vec::with_capacity(JOURNAL_DOMAIN.len() + 80);
    aad.extend_from_slice(JOURNAL_DOMAIN);
    aad.extend_from_slice(route_id);
    aad.extend_from_slice(commitment);
    aad.extend_from_slice(target);
    aad
}

fn encode_state(
    terminal_frame: &[u8],
    session: Option<&AnonymousMailboxSourceSealSessionV1>,
    completed: Option<&[u8]>,
) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
    let restart = session
        .map(|value| {
            value
                .encode_restart_state()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)
        })
        .transpose()?
        .map(|value| value.as_bytes().to_vec())
        .unwrap_or_default();
    let terminal_len =
        u32::try_from(terminal_frame.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    let restart_len =
        u16::try_from(restart.len()).map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let completed = completed.unwrap_or_default();
    let completed_len =
        u32::try_from(completed.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    let mut bytes =
        Vec::with_capacity(1 + 4 + terminal_frame.len() + 2 + restart.len() + 4 + completed.len());
    bytes.push(JOURNAL_STATE_VERSION);
    bytes.extend_from_slice(&terminal_len.to_le_bytes());
    bytes.extend_from_slice(terminal_frame);
    bytes.extend_from_slice(&restart_len.to_le_bytes());
    bytes.extend_from_slice(&restart);
    bytes.extend_from_slice(&completed_len.to_le_bytes());
    bytes.extend_from_slice(completed);
    Ok(bytes)
}

fn decode_state(
    bytes: &[u8],
) -> Result<
    (
        Vec<u8>,
        Option<AnonymousMailboxSourceSealSessionV1>,
        Option<Vec<u8>>,
    ),
    AnonymousMailboxSourceError,
> {
    if bytes.first().copied() != Some(JOURNAL_STATE_VERSION) || bytes.len() < 1 + 4 + 2 + 4 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let mut offset = 1;
    let terminal_len = u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    offset += 4;
    let terminal_end = offset
        .checked_add(terminal_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if terminal_end > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let terminal = bytes[offset..terminal_end].to_vec();
    ensure_request_frame(&terminal)?;
    offset = terminal_end;
    if offset + 2 > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let restart_len = u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    offset += 2;
    let restart_end = offset
        .checked_add(restart_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if restart_end + 4 > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let session = if restart_len == 0 {
        None
    } else {
        Some(
            AnonymousMailboxSourceSealSessionV1::decode_restart_state(&bytes[offset..restart_end])
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
        )
    };
    offset = restart_end;
    let completed_len = u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    offset += 4;
    let end = offset
        .checked_add(completed_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if end != bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let completed = if completed_len == 0 {
        None
    } else {
        Some(bytes[offset..end].to_vec())
    };
    Ok((terminal, session, completed))
}

fn fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N], AnonymousMailboxSourceError> {
    bytes
        .try_into()
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aeronyx_core::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxOutcomeV1,
        AnonymousMailboxSourceSealedResponseV1, AnonymousMailboxTicketIssueResponseV1,
        AnonymousMailboxTicketIssueV1,
    };
    use aeronyx_core::protocol::discovery::{NodeCapability, NodeDescriptor, NodeProtocolFeature};

    const NOW: u64 = 1_800_000_000;

    fn target() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[0x71; 32]).expect("valid identity")
    }

    fn ticket_request(target: &IdentityKeyPair) -> Vec<u8> {
        let request = AnonymousMailboxTicketIssueV1::new(
            [0x11; 16],
            [0x12; 16],
            target.public_key_bytes(),
            [0x13; 32],
            NOW,
            NOW + 30,
            0,
        )
        .expect("ticket request");
        encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::TicketIssue(
            request,
        ))
        .expect("terminal frame")
    }

    fn journal() -> SqliteAnonymousMailboxSourceJournal {
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            max_journal_entries: 4,
            max_journal_bytes: 4096,
        };
        SqliteAnonymousMailboxSourceJournal::new(
            Connection::open_in_memory().expect("memory sqlite"),
            [0x22; 32],
            &config,
        )
        .expect("journal")
    }

    struct ExactOnlyResolver {
        descriptor: SignedNodeDescriptor,
        calls: AtomicUsize,
    }

    impl ExactAnonymousMailboxTargetResolver for ExactOnlyResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            (node_id == &self.descriptor.descriptor.node_id).then(|| self.descriptor.clone())
        }
    }

    fn mailbox_descriptor(target: &IdentityKeyPair) -> SignedNodeDescriptor {
        let mut descriptor =
            NodeDescriptor::new(target.public_key_bytes(), 9, NOW - 1, NOW + 60, "test")
                .with_x25519_kem(target.x25519_public_key_bytes())
                .with_protocol_features([
                    NodeProtocolFeature::AnonymousMailboxV1,
                    NodeProtocolFeature::OnionReplyV1,
                    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
                ]);
        descriptor.public_endpoint = Some("https://node.invalid".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, target).expect("signed descriptor")
    }

    #[test]
    fn journal_exact_replay_is_stable_and_conflict_does_not_overwrite() {
        let target = target();
        let terminal = ticket_request(&target);
        let record = SourceJournalRecord {
            route_id: [0x31; 16],
            request_commitment: [0x32; 32],
            target_node_id: target.public_key_bytes(),
            descriptor_commitment: DirectoryDescriptorCommitmentV1 {
                node_id: target.public_key_bytes(),
                sequence: 7,
                descriptor_hash: [0x33; 32],
            },
            body: vec![0x34; 73],
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&terminal, None, None).expect("state"),
        };
        let journal = journal();
        let first = journal.insert_or_exact(&record).expect("insert");
        let replay = journal.insert_or_exact(&record).expect("exact replay");
        assert_eq!(first.body, replay.body);
        let mut conflict = record;
        conflict.request_commitment = [0x35; 32];
        conflict.body = vec![0x36; 73];
        assert!(matches!(
            journal.insert_or_exact(&conflict),
            Err(AnonymousMailboxSourceError::Conflict)
        ));
        assert_eq!(
            journal
                .load(&[0x31; 16])
                .expect("load")
                .expect("record")
                .body,
            vec![0x34; 73]
        );
    }

    #[test]
    fn journal_phase_cas_keeps_terminal_outcomes_irreversible() {
        let target = target();
        let terminal = ticket_request(&target);
        let record = SourceJournalRecord {
            route_id: [0x37; 16],
            request_commitment: [0x38; 32],
            target_node_id: target.public_key_bytes(),
            descriptor_commitment: DirectoryDescriptorCommitmentV1 {
                node_id: target.public_key_bytes(),
                sequence: 8,
                descriptor_hash: [0x39; 32],
            },
            body: vec![0x3a; 64],
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&terminal, None, None).expect("state"),
        };
        let journal = journal();
        journal.insert_or_exact(&record).expect("insert");
        journal
            .transition(
                &record,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                record.state.clone(),
            )
            .expect("arm");
        journal
            .transition(
                &record,
                AnonymousMailboxSourcePhase::Armed,
                AnonymousMailboxSourcePhase::Completed,
                encode_state(&terminal, None, Some(&terminal)).expect("completed state"),
            )
            .expect("complete");
        assert!(matches!(
            journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Armed,
                AnonymousMailboxSourcePhase::Ambiguous,
                record.state.clone(),
            ),
            Err(AnonymousMailboxSourceError::Ambiguous)
        ));
        assert_eq!(
            journal
                .load(&record.route_id)
                .expect("load")
                .expect("record")
                .phase,
            AnonymousMailboxSourcePhase::Completed
        );
    }

    #[test]
    fn source_sealed_ticket_response_is_one_shot_and_request_bound() {
        let target = target();
        let request = AnonymousMailboxTicketIssueV1::new(
            [0x41; 16],
            [0x42; 16],
            target.public_key_bytes(),
            [0x43; 32],
            NOW,
            NOW + 30,
            0,
        )
        .expect("ticket request");
        let request_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(request.clone()),
        )
        .expect("request frame");
        let route_id = [0x44; 16];
        let (carrier, mut session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            request_frame.clone(),
        )
        .expect("carrier");
        let response = AnonymousMailboxTicketIssueResponseV1::signed(
            &request,
            AnonymousMailboxOutcomeV1::Rejected,
            None,
            NOW,
            &target,
        )
        .expect("response");
        let response_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
        )
        .expect("response frame");
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            route_id,
            carrier.context_commitment(),
            target.public_key_bytes(),
            carrier.reply_public_key(),
            &response_frame,
            &target,
        )
        .and_then(|value| value.encode())
        .expect("seal");
        let opened = session.open(&sealed).expect("open once");
        let decoded = decode_anonymous_mailbox_terminal_frame(&opened).expect("decode");
        assert!(verify_response(&request_frame, &decoded, &target.public_key_bytes()).is_ok());
        assert!(
            session.open(&sealed).is_err(),
            "response key is consumed after the first open"
        );
    }

    #[test]
    fn coordinator_resolves_one_pinned_target_without_fallback() {
        let target = target();
        let descriptor = mailbox_descriptor(&target);
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("commitment");
        let resolver = Arc::new(ExactOnlyResolver {
            descriptor,
            calls: AtomicUsize::new(0),
        });
        let coordinator = AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source")),
            resolver.clone(),
            Arc::new(journal()),
        );
        let prepared = coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
                [0x51; 16],
                ticket_request(&target),
                NOW,
            )
            .expect("one target request");
        let replay = coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(
                    target.public_key_bytes(),
                    DirectoryDescriptorCommitmentV1::from_signed_descriptor(&resolver.descriptor)
                        .expect("same commitment"),
                ),
                [0x51; 16],
                ticket_request(&target),
                NOW + 1,
            )
            .expect("exact retry");
        assert!(!prepared.body().is_empty());
        assert_eq!(prepared.body(), replay.body(), "retry retains exact bytes");
        assert_eq!(
            resolver.calls.load(Ordering::Relaxed),
            1,
            "no candidate fallback"
        );
    }
}
