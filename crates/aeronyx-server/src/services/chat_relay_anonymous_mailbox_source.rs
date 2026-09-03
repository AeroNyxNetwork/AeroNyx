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

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[cfg(unix)]
use std::fs::File;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxOperationV1, AnonymousMailboxPullResultV1, AnonymousMailboxRouteRequestV1,
    AnonymousMailboxSourceSealSessionV1, AnonymousMailboxSourceTerminalCarrierV1,
    AnonymousMailboxTerminalFrameV1, AnonymousMailboxTerminalResponseV1,
    MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES,
};
use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{OnionRoutePurpose, VerifiedOnionRoute};
use chacha20poly1305::aead::{Aead, NewAead, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use sha2::{Digest, Sha256};

use crate::api::chat_peer::{prepare_exact_peer_blind_relay_http_request, PeerBlindRelayRequest};
use crate::api::{canonical_peer_http_url, peer_endpoint_is_public_ip};
use crate::config_chat_relay::AnonymousMailboxSourceConfig;

use super::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use super::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use super::chat_relay_mailbox::{
    prepare_private_sqlite_target, verify_private_file, AnonymousMailboxStoreError,
};
use super::peer_store::PeerStore;

const JOURNAL_STATE_VERSION: u8 = 2;
const JOURNAL_NONCE_BYTES: usize = 24;
const JOURNAL_AEAD_TAG_BYTES: usize = 16;
const MAX_JOURNAL_DESCRIPTOR_BYTES: usize = 1024;
const MAX_JOURNAL_BODY_BYTES: usize = 2 * 1024 * 1024;
// [ANONYMOUS-MAILBOX-SOURCE-BOUNDS 2026-09-03 by Codex] Keep persisted
// clear/protected state bounded by protocol frames, not operator-configurable
// aggregate storage. The restart allowance intentionally requires an explicit
// source-journal update if the core restart ABI ever grows beyond 256 bytes.
const MAX_JOURNAL_RESTART_STATE_BYTES: usize = 256;
const JOURNAL_STATE_BODY_COMMITMENT_BYTES: usize = 32;
const JOURNAL_STATE_ENVELOPE_BYTES: usize = 1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES + 4 + 2 + 4;
const MAX_JOURNAL_CLEAR_STATE_BYTES: usize = JOURNAL_STATE_ENVELOPE_BYTES
    + (2 * MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES)
    + MAX_JOURNAL_RESTART_STATE_BYTES;
const MAX_JOURNAL_PROTECTED_STATE_BYTES: usize =
    MAX_JOURNAL_CLEAR_STATE_BYTES + JOURNAL_AEAD_TAG_BYTES;
const JOURNAL_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-journal/v1\0";
const REQUEST_COMMITMENT_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-request/v1\0";
const BODY_COMMITMENT_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-body/v1\0";
const SOURCE_JOURNAL_SCHEMA_VERSION: i64 = 1;
const PEER_BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";

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

/// Durable terminal result. The bytes are canonical source-sealed terminal
/// response bytes, never a peer's outer HTTP response or receipt.
pub(crate) enum AnonymousMailboxSourceResult {
    Prepared,
    Armed,
    Completed(Vec<u8>),
    Ambiguous,
    Rejected,
}

/// Exact outbound work released only after the durable record has been
/// revalidated against the current descriptor. This type intentionally omits
/// Debug so a target endpoint or opaque carrier cannot reach diagnostics.
pub(crate) struct AnonymousMailboxSourceOutbound {
    route_id: [u8; 16],
    target_node_id: [u8; 32],
    url: reqwest::Url,
    body: Vec<u8>,
    request: PeerBlindRelayRequest,
}

impl AnonymousMailboxSourceOutbound {
    #[must_use]
    pub(crate) const fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    #[must_use]
    pub(crate) fn target_node_id(&self) -> &[u8; 32] {
        &self.target_node_id
    }

    #[must_use]
    pub(crate) fn url(&self) -> &reqwest::Url {
        &self.url
    }

    #[must_use]
    pub(crate) fn body(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub(crate) fn request(&self) -> &PeerBlindRelayRequest {
        &self.request
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

struct DecodedSourceState {
    body_commitment: [u8; 32],
    terminal_frame: Vec<u8>,
    restart: Option<AnonymousMailboxSourceSealSessionV1>,
    completed: Option<Vec<u8>>,
}

/// SQLite implementation of the source journal. `open` is deliberately
/// synchronous: server startup invokes it through `spawn_blocking` before any
/// source route becomes reachable.
pub(crate) struct SqliteAnonymousMailboxSourceJournal {
    connection: Mutex<Connection>,
    journal_key: [u8; 32],
    max_entries: usize,
    max_bytes: u64,
    #[cfg(unix)]
    _database_parent: Option<File>,
}

impl SqliteAnonymousMailboxSourceJournal {
    pub(crate) fn new(
        mut connection: Connection,
        journal_key: [u8; 32],
        config: &AnonymousMailboxSourceConfig,
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !config.enabled || config.max_journal_entries == 0 || config.max_journal_bytes == 0 {
            return Err(AnonymousMailboxSourceError::Disabled);
        }
        initialize_or_verify_source_schema(&mut connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            #[cfg(unix)]
            _database_parent: None,
        })
    }

    /// Opens the production journal only after descriptor-relative ownership,
    /// link-count and mode validation. The source DB is never shared with
    /// receiver custody or chat relay state.
    pub(crate) fn open(
        config: AnonymousMailboxSourceConfig,
        journal_key: [u8; 32],
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !config.enabled
            || config.db_path.is_empty()
            || config.db_path == ":memory:"
            || config.max_journal_entries == 0
            || i64::try_from(config.max_journal_entries).is_err()
            || config.max_journal_bytes == 0
            || config.max_journal_bytes > i64::MAX as u64
            || journal_key == [0; 32]
        {
            return Err(AnonymousMailboxSourceError::Rejected);
        }

        let target = prepare_private_sqlite_target(Path::new(&config.db_path))
            .map_err(map_private_sqlite_error)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        verify_private_file(&target.resolved_path, false).map_err(map_private_sqlite_error)?;
        restrict_private_source_permissions(&target.resolved_path)?;
        verify_private_file(&target.resolved_path, true).map_err(map_private_sqlite_error)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        verify_source_sqlite_integrity(&connection)?;
        configure_source_full_durability(&connection)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        initialize_or_verify_source_schema(&mut connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            #[cfg(unix)]
            _database_parent: Some(target.parent),
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
        let stored_bytes = u64::try_from(body_len).ok().and_then(|body| {
            u64::try_from(protected_len)
                .ok()
                .and_then(|protected| body.checked_add(protected))
        });
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
                .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
                .is_none()
            || stored_bytes
                .filter(|value| *value <= self.max_bytes)
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
                        &body,
                        &nonce,
                        &protected,
                    )?;
                    // [ANONYMOUS-MAILBOX-SOURCE-WIRING 2026-09-03 by Codex]
                    // The encrypted state binds every durable projection to
                    // the exact serialized carrier. Old rows lacking this
                    // commitment, body swaps, or descriptor/frame drift fail
                    // before a restart can release a network dispatch.
                    let decoded = decode_state(&state)?;
                    if decoded.body_commitment != source_body_commitment(&body)
                        || source_request_commitment(
                            route_id,
                            &target_node_id,
                            &descriptor_commitment,
                            &decoded.terminal_frame,
                        ) != request_commitment
                    {
                        return Err(AnonymousMailboxSourceError::Corrupt);
                    }
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
        validate_source_record_projection(record)?;
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
        if record.body.is_empty() || record.body.len() > MAX_JOURNAL_BODY_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let incoming = record
            .body
            .len()
            .checked_add(protected.len())
            .ok_or(AnonymousMailboxSourceError::Unavailable)?;
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
        if descriptor.is_empty() || descriptor.len() > MAX_JOURNAL_DESCRIPTOR_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
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
        validate_source_record_projection(record)?;
        let desired_decoded = decode_state(&state)?;
        if desired_decoded.body_commitment != source_body_commitment(&record.body)
            || desired_decoded.terminal_frame != decode_state(&record.state)?.terminal_frame
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let desired = self.seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &state,
        )?;
        let compact_ambiguous = if phase == AnonymousMailboxSourcePhase::Completed {
            let decoded = decode_state(&state)?;
            if decoded.body_commitment != source_body_commitment(&record.body) {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let compact_state = encode_state(&record.body, &decoded.terminal_frame, None, None)?;
            Some(self.seal_state(
                &record.route_id,
                &record.request_commitment,
                &record.target_node_id,
                &compact_state,
            )?)
        } else {
            None
        };

        // [ANONYMOUS-MAILBOX-SOURCE-BOUNDS 2026-09-03 by Codex] The phase
        // check, replacement accounting, fallback selection, and update share
        // one write transaction. A valid but unretainable response consumes
        // the one-shot session into a compact Ambiguous state rather than
        // creating a row that load() must later reject as corrupt.
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let current: Option<(i64, i64, i64)> = transaction
            .query_row(
                "SELECT phase, length(body), length(protected_state)
                 FROM anonymous_mailbox_source_journal
                 WHERE route_id = ?1 AND request_commitment = ?2 AND target_node_id = ?3",
                params![
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let Some((current_phase, body_len, old_protected_len)) = current else {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        };
        if AnonymousMailboxSourcePhase::decode(current_phase)? != expected {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        if body_len < 1
            || usize::try_from(body_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
                .is_none()
            || old_protected_len < JOURNAL_AEAD_TAG_BYTES as i64
            || usize::try_from(old_protected_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
                .is_none()
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let used: i64 = transaction
            .query_row(
                "SELECT COALESCE(SUM(length(body) + length(protected_state)), 0)
                 FROM anonymous_mailbox_source_journal",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let used = u64::try_from(used).map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        let old_row_bytes = u64::try_from(body_len)
            .ok()
            .and_then(|body| {
                u64::try_from(old_protected_len)
                    .ok()
                    .and_then(|protected| body.checked_add(protected))
            })
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        let retained_without_row = used
            .checked_sub(old_row_bytes)
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;

        let desired_row_bytes = u64::try_from(body_len)
            .ok()
            .and_then(|body| {
                u64::try_from(desired.1.len())
                    .ok()
                    .and_then(|protected| body.checked_add(protected))
            })
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        let desired_used = retained_without_row
            .checked_add(desired_row_bytes)
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        let (stored_phase, nonce, protected, fell_back) = if desired_used <= self.max_bytes {
            (phase, desired.0, desired.1, false)
        } else if let Some((nonce, protected)) = compact_ambiguous {
            let fallback_row_bytes = u64::try_from(body_len)
                .ok()
                .and_then(|body| {
                    u64::try_from(protected.len())
                        .ok()
                        .and_then(|protected| body.checked_add(protected))
                })
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            if retained_without_row
                .checked_add(fallback_row_bytes)
                .filter(|value| *value <= self.max_bytes)
                .is_none()
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            (
                AnonymousMailboxSourcePhase::Ambiguous,
                nonce,
                protected,
                true,
            )
        } else {
            return Err(AnonymousMailboxSourceError::Rejected);
        };

        let updated = transaction
            .execute(
                "UPDATE anonymous_mailbox_source_journal
                 SET phase = ?1, state_nonce = ?2, protected_state = ?3
                 WHERE route_id = ?4 AND request_commitment = ?5 AND target_node_id = ?6
                       AND phase = ?7",
                params![
                    stored_phase.code(),
                    nonce,
                    protected,
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(),
                    expected.code()
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if updated != 1 {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if fell_back {
            Err(AnonymousMailboxSourceError::Ambiguous)
        } else {
            Ok(())
        }
    }

    fn seal_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        state: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), AnonymousMailboxSourceError> {
        if state.len() > MAX_JOURNAL_CLEAR_STATE_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let decoded = decode_state(state)?;
        if decoded.body_commitment == [0; JOURNAL_STATE_BODY_COMMITMENT_BYTES] {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let mut nonce = [0u8; JOURNAL_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.journal_key));
        let protected = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: state,
                    aad: &journal_aad(
                        route_id,
                        request_commitment,
                        target,
                        &decoded.body_commitment,
                    ),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if protected.len() > MAX_JOURNAL_PROTECTED_STATE_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        Ok((nonce.to_vec(), protected))
    }

    fn open_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        body: &[u8],
        nonce: &[u8],
        protected: &[u8],
    ) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
        if protected.len() < JOURNAL_AEAD_TAG_BYTES
            || protected.len() > MAX_JOURNAL_PROTECTED_STATE_BYTES
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let nonce = fixed::<JOURNAL_NONCE_BYTES>(nonce)?;
        XChaCha20Poly1305::new(Key::from_slice(&self.journal_key))
            .decrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: protected,
                    aad: &journal_aad(
                        route_id,
                        request_commitment,
                        target,
                        &source_body_commitment(body),
                    ),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)
    }
}

/// Default-off exact-target source composition. It has no network ownership;
/// callers obtain a non-debug typed outbound through `begin_dispatch` and use
/// a separately injected bounded transport.
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
        let state = encode_state(&body, &terminal_frame, Some(&session), None)?;
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
        if !peer_endpoint_is_public_ip(&endpoint) {
            return Err(AnonymousMailboxSourceError::Unavailable);
        }
        let url = canonical_peer_http_url(&endpoint, PEER_BLIND_RELAY_PATH)
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

fn initialize_or_verify_source_schema(
    connection: &mut Connection,
) -> Result<(), AnonymousMailboxSourceError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    let version: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if version == 0 {
        let foreign: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type IN ('table', 'index', 'view', 'trigger')
                   AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        if foreign != 0 {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        transaction
            .execute_batch(
                "CREATE TABLE anonymous_mailbox_source_journal (
                    route_id BLOB PRIMARY KEY NOT NULL CHECK (length(route_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    descriptor_commitment BLOB NOT NULL CHECK (length(descriptor_commitment) > 0),
                    body BLOB NOT NULL CHECK (length(body) > 0),
                    phase INTEGER NOT NULL CHECK (phase BETWEEN 1 AND 5),
                    state_nonce BLOB NOT NULL CHECK (length(state_nonce) = 24),
                    protected_state BLOB NOT NULL CHECK (length(protected_state) >= 16)
                 );
                 PRAGMA user_version = 1;",
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    } else if version != SOURCE_JOURNAL_SCHEMA_VERSION {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let owned: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type IN ('table', 'index', 'view', 'trigger')
               AND name NOT LIKE 'sqlite_%'
               AND name != 'anonymous_mailbox_source_journal'",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if owned != 0 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let table_exists: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'anonymous_mailbox_source_journal'",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if table_exists != 1 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

fn verify_source_sqlite_integrity(
    connection: &Connection,
) -> Result<(), AnonymousMailboxSourceError> {
    verify_sqlite_physical_integrity(connection, "anonymous_mailbox_source_startup_integrity")
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
}

fn configure_source_full_durability(
    connection: &Connection,
) -> Result<(), AnonymousMailboxSourceError> {
    configure_full_durability(connection, 2)
        .map(|_| ())
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

fn restrict_private_source_permissions(path: &Path) -> Result<(), AnonymousMailboxSourceError> {
    restrict_private_sqlite_permissions(path).map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

fn map_private_sqlite_error(error: AnonymousMailboxStoreError) -> AnonymousMailboxSourceError {
    match error {
        AnonymousMailboxStoreError::Rejected => AnonymousMailboxSourceError::Rejected,
        AnonymousMailboxStoreError::Corrupt | AnonymousMailboxStoreError::UnsupportedSchema => {
            AnonymousMailboxSourceError::Corrupt
        }
        AnonymousMailboxStoreError::Disabled
        | AnonymousMailboxStoreError::Busy
        | AnonymousMailboxStoreError::Unavailable => AnonymousMailboxSourceError::Unavailable,
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

fn source_body_commitment(body: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(BODY_COMMITMENT_DOMAIN);
    hash.update((body.len() as u64).to_be_bytes());
    hash.update(body);
    hash.finalize().into()
}

fn validate_source_record_projection(
    record: &SourceJournalRecord,
) -> Result<(), AnonymousMailboxSourceError> {
    let decoded = decode_state(&record.state)?;
    if decoded.body_commitment != source_body_commitment(&record.body)
        || source_request_commitment(
            &record.route_id,
            &record.target_node_id,
            &record.descriptor_commitment,
            &decoded.terminal_frame,
        ) != record.request_commitment
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    Ok(())
}

fn journal_aad(
    route_id: &[u8; 16],
    commitment: &[u8; 32],
    target: &[u8; 32],
    body_commitment: &[u8; 32],
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(JOURNAL_DOMAIN.len() + 112);
    aad.extend_from_slice(JOURNAL_DOMAIN);
    aad.extend_from_slice(route_id);
    aad.extend_from_slice(commitment);
    aad.extend_from_slice(target);
    aad.extend_from_slice(body_commitment);
    aad
}

fn encode_state(
    body: &[u8],
    terminal_frame: &[u8],
    session: Option<&AnonymousMailboxSourceSealSessionV1>,
    completed: Option<&[u8]>,
) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
    if terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let restart = session
        .map(|value| {
            value
                .encode_restart_state()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)
        })
        .transpose()?
        .map(|value| value.as_bytes().to_vec())
        .unwrap_or_default();
    if restart.len() > MAX_JOURNAL_RESTART_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let terminal_len =
        u32::try_from(terminal_frame.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    let restart_len =
        u16::try_from(restart.len()).map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let completed = completed.unwrap_or_default();
    if completed.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let completed_len =
        u32::try_from(completed.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    if body.is_empty() || body.len() > MAX_JOURNAL_BODY_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let mut bytes = Vec::with_capacity(
        JOURNAL_STATE_ENVELOPE_BYTES + terminal_frame.len() + restart.len() + completed.len(),
    );
    bytes.push(JOURNAL_STATE_VERSION);
    bytes.extend_from_slice(&source_body_commitment(body));
    bytes.extend_from_slice(&terminal_len.to_le_bytes());
    bytes.extend_from_slice(terminal_frame);
    bytes.extend_from_slice(&restart_len.to_le_bytes());
    bytes.extend_from_slice(&restart);
    bytes.extend_from_slice(&completed_len.to_le_bytes());
    bytes.extend_from_slice(completed);
    if bytes.len() > MAX_JOURNAL_CLEAR_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    Ok(bytes)
}

fn decode_state(bytes: &[u8]) -> Result<DecodedSourceState, AnonymousMailboxSourceError> {
    if bytes.len() > MAX_JOURNAL_CLEAR_STATE_BYTES
        || bytes.first().copied() != Some(JOURNAL_STATE_VERSION)
        || bytes.len() < JOURNAL_STATE_ENVELOPE_BYTES
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let body_commitment = fixed::<JOURNAL_STATE_BODY_COMMITMENT_BYTES>(
        &bytes[1..1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES],
    )?;
    let mut offset = 1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES;
    let terminal_len = u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    if terminal_len > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
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
    if restart_len > MAX_JOURNAL_RESTART_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
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
    if completed_len > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
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
    Ok(DecodedSourceState {
        body_commitment,
        terminal_frame: terminal,
        restart: session,
        completed,
    })
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
        AnonymousMailboxPullOneV1, AnonymousMailboxSourceSealedResponseV1,
        AnonymousMailboxTicketIssueResponseV1, AnonymousMailboxTicketIssueV1,
        MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
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

    fn journal_with_max_bytes(max_journal_bytes: u64) -> SqliteAnonymousMailboxSourceJournal {
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            max_journal_entries: 4,
            max_journal_bytes,
            ..AnonymousMailboxSourceConfig::default()
        };
        SqliteAnonymousMailboxSourceJournal::new(
            Connection::open_in_memory().expect("memory sqlite"),
            [0x22; 32],
            &config,
        )
        .expect("journal")
    }

    fn journal() -> SqliteAnonymousMailboxSourceJournal {
        journal_with_max_bytes(4096)
    }

    struct ExactOnlyResolver {
        descriptor: SignedNodeDescriptor,
        calls: AtomicUsize,
    }

    struct MutableExactResolver {
        descriptor: Mutex<SignedNodeDescriptor>,
    }

    impl ExactAnonymousMailboxTargetResolver for MutableExactResolver {
        fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
            let descriptor = self.descriptor.lock().clone();
            (node_id == &descriptor.descriptor.node_id).then_some(descriptor)
        }
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
        let body = vec![0x34; 73];
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: 7,
            descriptor_hash: [0x33; 32],
        };
        let record = SourceJournalRecord {
            route_id: [0x31; 16],
            request_commitment: source_request_commitment(
                &[0x31; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&body, &terminal, None, None).expect("state"),
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
        let body = vec![0x3a; 64];
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: 8,
            descriptor_hash: [0x39; 32],
        };
        let record = SourceJournalRecord {
            route_id: [0x37; 16],
            request_commitment: source_request_commitment(
                &[0x37; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&body, &terminal, None, None).expect("state"),
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
                encode_state(&body, &terminal, None, Some(&terminal)).expect("completed state"),
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
    fn completed_pull_over_budget_becomes_restart_safe_ambiguous() {
        let target = target();
        let reader = IdentityKeyPair::from_bytes(&[0x3b; 32]).expect("reader");
        let request =
            AnonymousMailboxPullOneV1::new([0x3c; 32], [0x3d; 16], Vec::new(), NOW, &reader)
                .expect("pull request");
        let terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(request.clone()),
        )
        .expect("pull frame");
        let pulled = AnonymousMailboxPullResultV1::new(
            [0x3e; 16],
            Vec::new(),
            vec![0x3f; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES],
        )
        .and_then(|value| value.encode())
        .expect("maximum pull result");
        let response = AnonymousMailboxTerminalResponseV1::signed(
            AnonymousMailboxOperationV1::PullOne,
            request.request_id,
            request.request_commitment().expect("request commitment"),
            AnonymousMailboxOutcomeV1::Accepted,
            pulled,
            NOW,
            &target,
        )
        .expect("pull response");
        let completed = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOneResponse(response),
        )
        .expect("response frame");
        assert!(completed.len() <= MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES);

        let body = vec![0x40; 64];
        let prepared_state = encode_state(&body, &terminal, None, None).expect("prepared state");
        let exact_prepared_budget =
            u64::try_from(body.len() + prepared_state.len() + JOURNAL_AEAD_TAG_BYTES)
                .expect("budget");
        let journal = journal_with_max_bytes(exact_prepared_budget);
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: 9,
            descriptor_hash: [0x43; 32],
        };
        let record = SourceJournalRecord {
            route_id: [0x41; 16],
            request_commitment: source_request_commitment(
                &[0x41; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: prepared_state.clone(),
        };
        journal.insert_or_exact(&record).expect("insert");
        journal
            .transition(
                &record,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                prepared_state,
            )
            .expect("arm");
        let completed_state =
            encode_state(&body, &terminal, None, Some(&completed)).expect("completed state");
        assert!(matches!(
            journal.transition(
                &record,
                AnonymousMailboxSourcePhase::Armed,
                AnonymousMailboxSourcePhase::Completed,
                completed_state,
            ),
            Err(AnonymousMailboxSourceError::Ambiguous)
        ));

        let loaded = journal
            .load(&record.route_id)
            .expect("restart load")
            .expect("retained record");
        assert_eq!(loaded.phase, AnonymousMailboxSourcePhase::Ambiguous);
        let compact = decode_state(&loaded.state).expect("compact state");
        assert_eq!(compact.terminal_frame, terminal);
        assert!(compact.restart.is_none());
        assert!(compact.completed.is_none());
        let used: i64 = journal
            .connection
            .lock()
            .query_row(
                "SELECT SUM(length(body) + length(protected_state))
                 FROM anonymous_mailbox_source_journal",
                [],
                |row| row.get(0),
            )
            .expect("aggregate bytes");
        assert!(u64::try_from(used).expect("non-negative") <= exact_prepared_budget);
    }

    #[test]
    fn fixed_state_bounds_reject_oversize_before_state_materialization() {
        let journal = journal_with_max_bytes(
            u64::try_from(MAX_JOURNAL_PROTECTED_STATE_BYTES * 2).expect("large config"),
        );
        assert!(matches!(
            journal.seal_state(
                &[0x44; 16],
                &[0x45; 32],
                &[0x46; 32],
                &vec![0; MAX_JOURNAL_CLEAR_STATE_BYTES + 1],
            ),
            Err(AnonymousMailboxSourceError::Rejected)
        ));

        let descriptor = bincode::serialize(&DirectoryDescriptorCommitmentV1 {
            node_id: [0x46; 32],
            sequence: 10,
            descriptor_hash: [0x47; 32],
        })
        .expect("descriptor");
        journal
            .connection
            .lock()
            .execute(
                "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body,
                    phase, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    [0x44_u8; 16].as_slice(),
                    [0x45_u8; 32].as_slice(),
                    [0x46_u8; 32].as_slice(),
                    descriptor,
                    [0x48_u8].as_slice(),
                    AnonymousMailboxSourcePhase::Prepared.code(),
                    [0x49_u8; JOURNAL_NONCE_BYTES].as_slice(),
                    vec![0x4a_u8; MAX_JOURNAL_PROTECTED_STATE_BYTES + 1],
                ],
            )
            .expect("inject oversized protected state");
        assert!(matches!(
            journal.load(&[0x44; 16]),
            Err(AnonymousMailboxSourceError::Corrupt)
        ));
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

    #[test]
    fn journal_body_tamper_fails_before_restart_dispatch() {
        let target = target();
        let terminal = ticket_request(&target);
        let body = vec![0x63; 96];
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: 11,
            descriptor_hash: [0x64; 32],
        };
        let record = SourceJournalRecord {
            route_id: [0x65; 16],
            request_commitment: source_request_commitment(
                &[0x65; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&body, &terminal, None, None).expect("state"),
        };
        let journal = journal();
        journal.insert_or_exact(&record).expect("insert");
        journal
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_source_journal SET body = ?1 WHERE route_id = ?2",
                params![vec![0x66u8; 96], record.route_id.as_slice()],
            )
            .expect("tamper row");
        assert!(matches!(
            journal.load(&record.route_id),
            Err(AnonymousMailboxSourceError::Corrupt)
        ));
    }

    #[test]
    fn descriptor_drift_keeps_prepared_record_off_network() {
        let target = target();
        let descriptor = mailbox_descriptor(&target);
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("commitment");
        let resolver = Arc::new(MutableExactResolver {
            descriptor: Mutex::new(descriptor),
        });
        let coordinator = AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[0x73; 32]).expect("source")),
            resolver.clone(),
            Arc::new(journal()),
        );
        let route_id = [0x74; 16];
        coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
                route_id,
                ticket_request(&target),
                NOW,
            )
            .expect("prepare");
        let mut drifted = mailbox_descriptor(&target).descriptor;
        drifted.sequence = drifted.sequence.saturating_add(1);
        *resolver.descriptor.lock() = SignedNodeDescriptor::sign(drifted, &target).expect("drift");
        assert!(matches!(
            coordinator.begin_dispatch(route_id, NOW),
            Err(AnonymousMailboxSourceError::Unavailable)
        ));
        assert!(matches!(
            coordinator.result(route_id).expect("result"),
            AnonymousMailboxSourceResult::Prepared
        ));
    }

    #[test]
    fn source_journal_private_open_restarts_only_from_one_owner_private_inode() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = std::fs::canonicalize(directory.path())
            .expect("canonical private directory")
            .join("source.db");
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: path.to_string_lossy().into_owned(),
            ..AnonymousMailboxSourceConfig::default()
        };
        let target = target();
        let terminal = ticket_request(&target);
        let body = vec![0x67; 64];
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: 12,
            descriptor_hash: [0x68; 32],
        };
        let record = SourceJournalRecord {
            route_id: [0x69; 16],
            request_commitment: source_request_commitment(
                &[0x69; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase: AnonymousMailboxSourcePhase::Prepared,
            state: encode_state(&body, &terminal, None, None).expect("state"),
        };
        let journal = SqliteAnonymousMailboxSourceJournal::open(config.clone(), [0x67; 32])
            .expect("first open");
        journal.insert_or_exact(&record).expect("insert");
        journal
            .transition(
                &record,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                record.state.clone(),
            )
            .expect("arm");
        drop(journal);
        let reopened =
            SqliteAnonymousMailboxSourceJournal::open(config, [0x67; 32]).expect("restart open");
        let resumed = reopened
            .load(&record.route_id)
            .expect("restart load")
            .expect("record");
        assert_eq!(resumed.phase, AnonymousMailboxSourcePhase::Armed);
        assert_eq!(resumed.body, body);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                std::fs::metadata(path).expect("metadata").mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn source_journal_rejects_symlink_and_hardlink_before_sqlite_mutation() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("tempdir");
        let private_directory =
            std::fs::canonicalize(directory.path()).expect("canonical private directory");
        let target = private_directory.join("target.db");
        std::fs::File::create(&target).expect("target");
        let symlink_path = private_directory.join("symlink.db");
        symlink(&target, &symlink_path).expect("symlink");
        let symlink_config = AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: symlink_path.to_string_lossy().into_owned(),
            ..AnonymousMailboxSourceConfig::default()
        };
        assert!(matches!(
            SqliteAnonymousMailboxSourceJournal::open(symlink_config, [0x68; 32]),
            Err(AnonymousMailboxSourceError::Rejected | AnonymousMailboxSourceError::Unavailable)
        ));

        let hardlink_path = private_directory.join("hardlink.db");
        std::fs::hard_link(&target, &hardlink_path).expect("hardlink");
        let hardlink_config = AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: hardlink_path.to_string_lossy().into_owned(),
            ..AnonymousMailboxSourceConfig::default()
        };
        assert!(matches!(
            SqliteAnonymousMailboxSourceJournal::open(hardlink_config, [0x69; 32]),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
    }
}
