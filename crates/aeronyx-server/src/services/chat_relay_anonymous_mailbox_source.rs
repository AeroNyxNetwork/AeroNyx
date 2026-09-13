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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use crate::config_chat_relay::{
    AnonymousMailboxSourceConfig, MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_RETENTION_SECS,
};

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
// [ANONYMOUS-MAILBOX-SOURCE-RETENTION 2026-09-13 by Codex] Schema v2 gives
// terminal rows a durable replay deadline and exact aggregate accounting.
// Unresolved phases deliberately retain NULL forever and cannot be age-cleaned.
const SOURCE_JOURNAL_SCHEMA_VERSION: i64 = 2;
const PEER_BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";

#[cfg(test)]
const SOURCE_JOURNAL_CRASH_PHASE_ENV: &str = "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_PHASE";
#[cfg(test)]
const SOURCE_JOURNAL_CRASH_BARRIER_ENV: &str =
    "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_BARRIER";
#[cfg(test)]
const SOURCE_JOURNAL_CRASH_EXIT_CODE: i32 = 79;

#[cfg(test)]
fn crash_after_source_journal_commit(phase: AnonymousMailboxSourcePhase) {
    use std::io::Write;

    if std::env::var(SOURCE_JOURNAL_CRASH_PHASE_ENV)
        .ok()
        .as_deref()
        != Some(match phase {
            AnonymousMailboxSourcePhase::Prepared => "prepared",
            AnonymousMailboxSourcePhase::Armed => "armed",
            AnonymousMailboxSourcePhase::Completed => "completed",
            AnonymousMailboxSourcePhase::Ambiguous => "ambiguous",
            AnonymousMailboxSourcePhase::Rejected => "rejected",
        })
    {
        return;
    }
    let barrier = std::env::var_os(SOURCE_JOURNAL_CRASH_BARRIER_ENV)
        .expect("source crash drill barrier path");
    let mut marker = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(barrier)
        .expect("create source crash drill barrier");
    marker
        .write_all(b"phase-commit-observed")
        .expect("write source crash drill barrier");
    marker.sync_all().expect("sync source crash drill barrier");

    // [ANONYMOUS-MAILBOX-SOURCE-CRASH-DRILL 2026-09-05 by Codex] This hook
    // exists only in the libtest build. It crosses a real process boundary
    // after SQLite commit but before the journal method can return, without
    // signalling or otherwise interacting with any production process.
    std::process::exit(SOURCE_JOURNAL_CRASH_EXIT_CODE);
}

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
    retain_until: Option<u64>,
    state: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AnonymousMailboxSourceCleanupReport {
    pub(crate) rows_removed: u64,
    pub(crate) bytes_removed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceJournalMeta {
    entries: u64,
    bytes: u64,
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
    terminal_retention_secs: u64,
    cleanup_batch_size: usize,
    #[cfg(unix)]
    _database_parent: Option<File>,
}

impl SqliteAnonymousMailboxSourceJournal {
    pub(crate) fn new(
        mut connection: Connection,
        journal_key: [u8; 32],
        config: &AnonymousMailboxSourceConfig,
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !source_config_is_valid(config) || journal_key == [0; 32] {
            return Err(AnonymousMailboxSourceError::Disabled);
        }
        initialize_or_verify_source_schema(
            &mut connection,
            source_now_secs()?,
            config.terminal_retention_secs,
        )?;
        let journal = Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            terminal_retention_secs: config.terminal_retention_secs,
            cleanup_batch_size: config.cleanup_batch_size,
            #[cfg(unix)]
            _database_parent: None,
        };
        journal.audit_startup()?;
        Ok(journal)
    }

    /// Opens the production journal only after descriptor-relative ownership,
    /// link-count and mode validation. The source DB is never shared with
    /// receiver custody or chat relay state.
    pub(crate) fn open(
        config: AnonymousMailboxSourceConfig,
        journal_key: [u8; 32],
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !source_config_is_valid(&config)
            || config.db_path.is_empty()
            || config.db_path == ":memory:"
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
        initialize_or_verify_source_schema(
            &mut connection,
            source_now_secs()?,
            config.terminal_retention_secs,
        )?;
        let journal = Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            terminal_retention_secs: config.terminal_retention_secs,
            cleanup_batch_size: config.cleanup_batch_size,
            #[cfg(unix)]
            _database_parent: Some(target.parent),
        };
        journal.audit_startup()?;
        Ok(journal)
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
                        retain_until, state_nonce, protected_state
                 FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![route_id.as_slice()],
                |row| {
                    let request_commitment: Vec<u8> = row.get(0)?;
                    let target_node_id: Vec<u8> = row.get(1)?;
                    let descriptor: Vec<u8> = row.get(2)?;
                    let body: Vec<u8> = row.get(3)?;
                    let phase: i64 = row.get(4)?;
                    let retain_until: Option<i64> = row.get(5)?;
                    let nonce: Vec<u8> = row.get(6)?;
                    let protected: Vec<u8> = row.get(7)?;
                    Ok((
                        request_commitment,
                        target_node_id,
                        descriptor,
                        body,
                        phase,
                        retain_until,
                        nonce,
                        protected,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?
            .map(
                |(request, target, descriptor, body, phase, retain_until, nonce, protected)| {
                    let request_commitment = fixed::<32>(&request)?;
                    let target_node_id = fixed::<32>(&target)?;
                    let descriptor_commitment = bincode::deserialize(&descriptor)
                        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
                    let phase = AnonymousMailboxSourcePhase::decode(phase)?;
                    let retain_until = retain_until.map(source_u64).transpose()?;
                    validate_phase_retention(phase, retain_until)?;
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
                        retain_until,
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
        let meta = load_source_meta(&transaction)?;
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
        let incoming =
            u64::try_from(incoming).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let updated_meta = SourceJournalMeta {
            entries: meta
                .entries
                .checked_add(1)
                .ok_or(AnonymousMailboxSourceError::Rejected)?,
            bytes: meta
                .bytes
                .checked_add(incoming)
                .ok_or(AnonymousMailboxSourceError::Rejected)?,
        };
        if updated_meta.entries
            > u64::try_from(self.max_entries).map_err(|_| AnonymousMailboxSourceError::Corrupt)?
            || updated_meta.bytes > self.max_bytes
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
                    retain_until, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    record.route_id.as_slice(), record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(), descriptor, record.body,
                    record.phase.code(), record.retain_until.map(source_i64).transpose()?, nonce,
                    protected
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        update_source_meta_exact(&transaction, meta, updated_meta)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        #[cfg(test)]
        crash_after_source_journal_commit(record.phase);
        Ok(SourceJournalRecord {
            route_id: record.route_id,
            request_commitment: record.request_commitment,
            target_node_id: record.target_node_id,
            descriptor_commitment: record.descriptor_commitment,
            body: record.body.clone(),
            phase: record.phase,
            retain_until: record.retain_until,
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
        self.transition_at(record, expected, phase, state, source_now_secs()?)
    }

    fn transition_at(
        &self,
        record: &SourceJournalRecord,
        expected: AnonymousMailboxSourcePhase,
        phase: AnonymousMailboxSourcePhase,
        state: Vec<u8>,
        transitioned_at: u64,
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
        let current: Option<(i64, i64, i64, Option<i64>)> = transaction
            .query_row(
                "SELECT phase, length(body), length(protected_state), retain_until
                 FROM anonymous_mailbox_source_journal
                 WHERE route_id = ?1 AND request_commitment = ?2 AND target_node_id = ?3",
                params![
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let Some((current_phase, body_len, old_protected_len, current_retain_until)) = current
        else {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        };
        let current_phase = AnonymousMailboxSourcePhase::decode(current_phase)?;
        let current_retain_until = current_retain_until.map(source_u64).transpose()?;
        validate_phase_retention(current_phase, current_retain_until)?;
        if current_phase != expected {
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
        let meta = load_source_meta(&transaction)?;
        let old_row_bytes = u64::try_from(body_len)
            .ok()
            .and_then(|body| {
                u64::try_from(old_protected_len)
                    .ok()
                    .and_then(|protected| body.checked_add(protected))
            })
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        let retained_without_row = meta
            .bytes
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
        let retain_until = if matches!(
            stored_phase,
            AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
        ) {
            Some(
                transitioned_at
                    .checked_add(self.terminal_retention_secs)
                    .filter(|value| *value <= i64::MAX as u64)
                    .ok_or(AnonymousMailboxSourceError::Rejected)?,
            )
        } else {
            None
        };
        validate_phase_retention(stored_phase, retain_until)?;
        let updated_meta = SourceJournalMeta {
            entries: meta.entries,
            bytes: retained_without_row
                .checked_add(
                    u64::try_from(body_len)
                        .ok()
                        .and_then(|body| {
                            u64::try_from(protected.len())
                                .ok()
                                .and_then(|protected| body.checked_add(protected))
                        })
                        .ok_or(AnonymousMailboxSourceError::Corrupt)?,
                )
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
        };

        let updated = transaction
            .execute(
                "UPDATE anonymous_mailbox_source_journal
                 SET phase = ?1, retain_until = ?2, state_nonce = ?3, protected_state = ?4
                 WHERE route_id = ?5 AND request_commitment = ?6 AND target_node_id = ?7
                       AND phase = ?8",
                params![
                    stored_phase.code(),
                    retain_until.map(source_i64).transpose()?,
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
        update_source_meta_exact(&transaction, meta, updated_meta)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        #[cfg(test)]
        crash_after_source_journal_commit(stored_phase);
        if fell_back {
            Err(AnonymousMailboxSourceError::Ambiguous)
        } else {
            Ok(())
        }
    }

    /// Reclaims only terminal source rows whose byte-identical replay window ended.
    ///
    /// Prepared, armed, and ambiguous rows are durable safety fences and are
    /// never selected, regardless of age. The aggregate report contains no
    /// route, target, commitment, path, or ciphertext information.
    pub(crate) fn cleanup_terminal_records(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxSourceCleanupReport, AnonymousMailboxSourceError> {
        let now_sql = source_i64(now)?;
        let limit = source_i64(
            u64::try_from(self.cleanup_batch_size)
                .map_err(|_| AnonymousMailboxSourceError::Rejected)?,
        )?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let before = load_source_meta(&transaction)?;
        let rows = {
            let mut statement = transaction
                .prepare(
                    "SELECT route_id, phase, retain_until, length(body), length(protected_state)
                     FROM anonymous_mailbox_source_journal
                     WHERE phase IN (3, 5) AND retain_until < ?1
                     ORDER BY retain_until, route_id LIMIT ?2",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            let mapped = statement
                .query_map(params![now_sql, limit], |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                })
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            mapped
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?
        };
        let mut report = AnonymousMailboxSourceCleanupReport::default();
        for (route_id, phase, retain_until, body_len, protected_len) in rows {
            let route_id = fixed::<16>(&route_id)?;
            let phase = AnonymousMailboxSourcePhase::decode(phase)?;
            let retain_until = retain_until.map(source_u64).transpose()?;
            validate_phase_retention(phase, retain_until)?;
            if !matches!(
                phase,
                AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
            ) || retain_until.filter(|deadline| *deadline < now).is_none()
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let row_bytes = source_row_bytes(body_len, protected_len)?;
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_source_journal
                     WHERE route_id = ?1 AND phase = ?2 AND retain_until = ?3",
                    params![
                        route_id.as_slice(),
                        phase.code(),
                        retain_until.map(source_i64).transpose()?
                    ],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            if removed != 1 {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            report.rows_removed = report
                .rows_removed
                .checked_add(1)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            report.bytes_removed = report
                .bytes_removed
                .checked_add(row_bytes)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        }
        let after = SourceJournalMeta {
            entries: before
                .entries
                .checked_sub(report.rows_removed)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
            bytes: before
                .bytes
                .checked_sub(report.bytes_removed)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
        };
        if after != before {
            update_source_meta_exact(&transaction, before, after)?;
        }
        if compute_source_meta(&transaction)? != after {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        Ok(report)
    }

    fn audit_startup(&self) -> Result<(), AnonymousMailboxSourceError> {
        let route_ids = {
            let connection = self.connection.lock();
            let stored = load_source_meta(&connection)?;
            let observed = compute_source_meta(&connection)?;
            if stored != observed
                || stored.entries
                    > u64::try_from(self.max_entries)
                        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?
                || stored.bytes > self.max_bytes
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let mut statement = connection
                .prepare("SELECT route_id FROM anonymous_mailbox_source_journal ORDER BY route_id")
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            let rows = statement
                .query_map([], |row| row.get::<_, Vec<u8>>(0))
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            rows.map(|row| {
                row.map_err(|_| AnonymousMailboxSourceError::Corrupt)
                    .and_then(|bytes| fixed::<16>(&bytes))
            })
            .collect::<Result<Vec<_>, _>>()?
        };
        for route_id in route_ids {
            self.load(&route_id)?
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        }
        Ok(())
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
    migration_now: u64,
    terminal_retention_secs: u64,
) -> Result<(), AnonymousMailboxSourceError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    let version: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let foreign_before: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type IN ('table', 'index', 'view', 'trigger')
               AND name NOT LIKE 'sqlite_%'
               AND name NOT IN ('anonymous_mailbox_source_journal',
                                'anonymous_mailbox_source_meta')",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if foreign_before != 0 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    match version {
        0 => {
            let owned_before: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type IN ('table', 'index', 'view', 'trigger')
                       AND name NOT LIKE 'sqlite_%'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            if owned_before != 0 {
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
                        retain_until INTEGER CHECK (retain_until IS NULL OR retain_until >= 0),
                        state_nonce BLOB NOT NULL CHECK (length(state_nonce) = 24),
                        protected_state BLOB NOT NULL CHECK (length(protected_state) >= 16)
                     );
                     CREATE TABLE anonymous_mailbox_source_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        schema_version INTEGER NOT NULL CHECK (schema_version = 2),
                        total_entries INTEGER NOT NULL CHECK (total_entries >= 0),
                        total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0)
                     );
                     INSERT INTO anonymous_mailbox_source_meta VALUES (1, 2, 0, 0);
                     PRAGMA user_version = 2;",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        }
        1 => {
            let legacy_tables: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_source_journal'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            let unexpected_meta: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_source_meta'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            if legacy_tables != 1 || unexpected_meta != 0 {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let retain_until = migration_now
                .checked_add(terminal_retention_secs)
                .filter(|value| *value <= i64::MAX as u64)
                .ok_or(AnonymousMailboxSourceError::Rejected)?;
            transaction
                .execute_batch(
                    "ALTER TABLE anonymous_mailbox_source_journal
                         ADD COLUMN retain_until INTEGER
                         CHECK (retain_until IS NULL OR retain_until >= 0);
                     CREATE TABLE anonymous_mailbox_source_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        schema_version INTEGER NOT NULL CHECK (schema_version = 2),
                        total_entries INTEGER NOT NULL CHECK (total_entries >= 0),
                        total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0)
                     );",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_source_journal SET retain_until = ?1
                     WHERE phase IN (3, 5)",
                    params![source_i64(retain_until)?],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            let observed = compute_source_meta(&transaction)?;
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_source_meta
                     (singleton, schema_version, total_entries, total_bytes)
                     VALUES (1, 2, ?1, ?2)",
                    params![source_i64(observed.entries)?, source_i64(observed.bytes)?],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            transaction
                .execute_batch("PRAGMA user_version = 2;")
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        }
        SOURCE_JOURNAL_SCHEMA_VERSION => {}
        _ => return Err(AnonymousMailboxSourceError::Corrupt),
    }
    let table_count: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name IN ('anonymous_mailbox_source_journal',
                            'anonymous_mailbox_source_meta')",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let owned_after: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type IN ('table', 'index', 'view', 'trigger')
               AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if table_count != 2 || owned_after != 2 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    load_source_meta(&transaction)?;
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

fn source_config_is_valid(config: &AnonymousMailboxSourceConfig) -> bool {
    config.enabled
        && config.max_journal_entries > 0
        && i64::try_from(config.max_journal_entries).is_ok()
        && config.max_journal_bytes > 0
        && config.max_journal_bytes <= i64::MAX as u64
        && config.max_in_flight > 0
        && config.request_timeout_secs > 0
        && config.terminal_retention_secs > 0
        && config.terminal_retention_secs <= MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_RETENTION_SECS
        && config.cleanup_batch_size > 0
        && config.cleanup_batch_size <= 4_096
}

fn source_now_secs() -> Result<u64, AnonymousMailboxSourceError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

fn source_i64(value: u64) -> Result<i64, AnonymousMailboxSourceError> {
    i64::try_from(value).map_err(|_| AnonymousMailboxSourceError::Rejected)
}

fn source_u64(value: i64) -> Result<u64, AnonymousMailboxSourceError> {
    u64::try_from(value).map_err(|_| AnonymousMailboxSourceError::Corrupt)
}

fn source_row_bytes(body_len: i64, protected_len: i64) -> Result<u64, AnonymousMailboxSourceError> {
    if body_len < 1
        || usize::try_from(body_len)
            .ok()
            .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
            .is_none()
        || protected_len < JOURNAL_AEAD_TAG_BYTES as i64
        || usize::try_from(protected_len)
            .ok()
            .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
            .is_none()
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    source_u64(body_len)?
        .checked_add(source_u64(protected_len)?)
        .ok_or(AnonymousMailboxSourceError::Corrupt)
}

fn validate_phase_retention(
    phase: AnonymousMailboxSourcePhase,
    retain_until: Option<u64>,
) -> Result<(), AnonymousMailboxSourceError> {
    let terminal = matches!(
        phase,
        AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
    );
    if terminal == retain_until.is_some() {
        Ok(())
    } else {
        Err(AnonymousMailboxSourceError::Corrupt)
    }
}

fn load_source_meta(
    connection: &Connection,
) -> Result<SourceJournalMeta, AnonymousMailboxSourceError> {
    connection
        .query_row(
            "SELECT schema_version, total_entries, total_bytes
             FROM anonymous_mailbox_source_meta WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
        .and_then(|(version, entries, bytes)| {
            if version != SOURCE_JOURNAL_SCHEMA_VERSION {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            Ok(SourceJournalMeta {
                entries: source_u64(entries)?,
                bytes: source_u64(bytes)?,
            })
        })
}

fn compute_source_meta(
    connection: &Connection,
) -> Result<SourceJournalMeta, AnonymousMailboxSourceError> {
    let (entries, bytes): (i64, i64) = connection
        .query_row(
            "SELECT COUNT(*), COALESCE(SUM(length(body) + length(protected_state)), 0)
             FROM anonymous_mailbox_source_journal",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    Ok(SourceJournalMeta {
        entries: source_u64(entries)?,
        bytes: source_u64(bytes)?,
    })
}

fn update_source_meta_exact(
    transaction: &rusqlite::Transaction<'_>,
    before: SourceJournalMeta,
    after: SourceJournalMeta,
) -> Result<(), AnonymousMailboxSourceError> {
    let updated = transaction
        .execute(
            "UPDATE anonymous_mailbox_source_meta
             SET total_entries = ?1, total_bytes = ?2
             WHERE singleton = 1 AND schema_version = 2
               AND total_entries = ?3 AND total_bytes = ?4",
            params![
                source_i64(after.entries)?,
                source_i64(after.bytes)?,
                source_i64(before.entries)?,
                source_i64(before.bytes)?
            ],
        )
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    if updated == 1 {
        Ok(())
    } else {
        Err(AnonymousMailboxSourceError::Corrupt)
    }
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
    validate_phase_retention(record.phase, record.retain_until)?;
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
    use std::path::{Path, PathBuf};
    use std::process::Output;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aeronyx_core::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxOutcomeV1,
        AnonymousMailboxPullOneV1, AnonymousMailboxSourceSealedResponseV1,
        AnonymousMailboxTicketIssueResponseV1, AnonymousMailboxTicketIssueV1,
        MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
    };
    use aeronyx_core::protocol::discovery::{NodeCapability, NodeDescriptor, NodeProtocolFeature};

    const NOW: u64 = 1_800_000_000;
    const SOURCE_CRASH_STAGE_ENV: &str = "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_STAGE";
    const SOURCE_CRASH_DB_ENV: &str = "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_DB";
    const SOURCE_CRASH_WORKER: &str = concat!(
        "services::chat_relay_anonymous_mailbox_source::tests::",
        "source_journal_crash_drill_subprocess_worker"
    );

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
        journal_with_limits(4, max_journal_bytes, 7 * 24 * 60 * 60, 256)
    }

    fn journal_with_limits(
        max_journal_entries: usize,
        max_journal_bytes: u64,
        terminal_retention_secs: u64,
        cleanup_batch_size: usize,
    ) -> SqliteAnonymousMailboxSourceJournal {
        let config = AnonymousMailboxSourceConfig {
            enabled: true,
            max_journal_entries,
            max_journal_bytes,
            terminal_retention_secs,
            cleanup_batch_size,
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

    fn journal_record(
        route_byte: u8,
        body_byte: u8,
        phase: AnonymousMailboxSourcePhase,
    ) -> SourceJournalRecord {
        let target = target();
        let terminal = ticket_request(&target);
        let body = vec![body_byte; 64];
        let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
            node_id: target.public_key_bytes(),
            sequence: u64::from(route_byte),
            descriptor_hash: [route_byte.wrapping_add(1); 32],
        };
        SourceJournalRecord {
            route_id: [route_byte; 16],
            request_commitment: source_request_commitment(
                &[route_byte; 16],
                &target.public_key_bytes(),
                &descriptor_commitment,
                &terminal,
            ),
            target_node_id: target.public_key_bytes(),
            descriptor_commitment,
            body: body.clone(),
            phase,
            retain_until: None,
            state: encode_state(&body, &terminal, None, None).expect("state"),
        }
    }

    fn retain_terminal_record(
        journal: &SqliteAnonymousMailboxSourceJournal,
        record: &SourceJournalRecord,
        phase: AnonymousMailboxSourcePhase,
        transitioned_at: u64,
        completed: Option<&[u8]>,
    ) -> SourceJournalRecord {
        journal.insert_or_exact(record).expect("insert source row");
        let terminal = decode_state(&record.state)
            .expect("decode prepared state")
            .terminal_frame;
        journal
            .transition_at(
                record,
                AnonymousMailboxSourcePhase::Prepared,
                phase,
                encode_state(&record.body, &terminal, None, completed)
                    .expect("encode terminal state"),
                transitioned_at,
            )
            .expect("terminal transition");
        journal
            .load(&record.route_id)
            .expect("load terminal row")
            .expect("terminal row")
    }

    fn create_legacy_source_schema(connection: &Connection) {
        connection
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
            .expect("legacy source schema");
    }

    fn insert_legacy_source_record(
        connection: &Connection,
        sealer: &SqliteAnonymousMailboxSourceJournal,
        record: &SourceJournalRecord,
    ) {
        let descriptor =
            bincode::serialize(&record.descriptor_commitment).expect("legacy descriptor");
        let (nonce, protected) = sealer
            .seal_state(
                &record.route_id,
                &record.request_commitment,
                &record.target_node_id,
                &record.state,
            )
            .expect("legacy protected state");
        connection
            .execute(
                "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body,
                    phase, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(),
                    descriptor,
                    &record.body,
                    record.phase.code(),
                    nonce,
                    protected
                ],
            )
            .expect("legacy source row");
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

    fn source_crash_descriptor(target: &IdentityKeyPair) -> SignedNodeDescriptor {
        let mut descriptor =
            NodeDescriptor::new(target.public_key_bytes(), 9, NOW - 1, NOW + 60, "test")
                .with_x25519_kem(target.x25519_public_key_bytes())
                .with_protocol_features([
                    NodeProtocolFeature::AnonymousMailboxV1,
                    NodeProtocolFeature::OnionReplyV1,
                    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
                ]);
        descriptor.public_endpoint = Some("https://1.1.1.1:443".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, target).expect("signed crash-drill descriptor")
    }

    fn source_crash_config(path: &Path) -> AnonymousMailboxSourceConfig {
        AnonymousMailboxSourceConfig {
            enabled: true,
            db_path: path.to_string_lossy().into_owned(),
            ..AnonymousMailboxSourceConfig::default()
        }
    }

    fn source_crash_coordinator(
        journal: Arc<SqliteAnonymousMailboxSourceJournal>,
    ) -> (AnonymousMailboxSourceCoordinator, Arc<ExactOnlyResolver>) {
        let target = target();
        let resolver = Arc::new(ExactOnlyResolver {
            descriptor: source_crash_descriptor(&target),
            calls: AtomicUsize::new(0),
        });
        let coordinator = AnonymousMailboxSourceCoordinator::new(
            Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source identity")),
            resolver.clone(),
            journal,
        );
        (coordinator, resolver)
    }

    async fn run_source_crash_child(
        stage: &str,
        crash_phase: &str,
        db_path: &Path,
        barrier_path: &Path,
    ) -> Output {
        let executable = std::env::current_exe().expect("resolve source crash test binary");
        let mut child = crate::isolated_child_command(executable);
        child
            .arg(SOURCE_CRASH_WORKER)
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(SOURCE_CRASH_STAGE_ENV, stage)
            .env(SOURCE_CRASH_DB_ENV, db_path)
            .env(SOURCE_JOURNAL_CRASH_PHASE_ENV, crash_phase)
            .env(SOURCE_JOURNAL_CRASH_BARRIER_ENV, barrier_path)
            .kill_on_drop(true);
        tokio::time::timeout(Duration::from_secs(20), child.output())
            .await
            .expect("source crash child exceeded bounded deadline")
            .expect("start source crash child")
    }

    fn assert_source_crash_child(output: &Output, barrier_path: &Path) {
        assert_eq!(
            output.status.code(),
            Some(SOURCE_JOURNAL_CRASH_EXIT_CODE),
            "source crash worker missed the commit boundary; status={:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            std::fs::read(barrier_path).expect("read source crash barrier"),
            b"phase-commit-observed"
        );
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
            retain_until: None,
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
            retain_until: None,
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
    fn terminal_retention_preserves_exact_replay_until_strict_deadline() {
        // [ANONYMOUS-MAILBOX-SOURCE-RETENTION 2026-09-13 by Codex] A
        // terminal row is the durable idempotency result until strictly after
        // its deadline. Equality remains retained to avoid a clock-boundary
        // retry racing cleanup.
        let journal = journal_with_limits(2, 4096, 10, 2);
        let record = journal_record(0x81, 0x82, AnonymousMailboxSourcePhase::Prepared);
        let completed = vec![0x83; 37];
        let retained = retain_terminal_record(
            &journal,
            &record,
            AnonymousMailboxSourcePhase::Completed,
            NOW,
            Some(&completed),
        );
        assert_eq!(retained.retain_until, Some(NOW + 10));

        let replay = journal.insert_or_exact(&record).expect("exact retry");
        assert_eq!(replay.phase, AnonymousMailboxSourcePhase::Completed);
        assert_eq!(
            decode_state(&replay.state)
                .expect("decode replay")
                .completed,
            Some(completed)
        );
        let mut conflict = journal_record(0x81, 0x84, AnonymousMailboxSourcePhase::Prepared);
        conflict.request_commitment = [0x85; 32];
        assert!(matches!(
            journal.insert_or_exact(&conflict),
            Err(AnonymousMailboxSourceError::Conflict)
        ));

        assert_eq!(
            journal
                .cleanup_terminal_records(NOW + 10)
                .expect("deadline cleanup"),
            AnonymousMailboxSourceCleanupReport::default()
        );
        assert!(journal
            .load(&record.route_id)
            .expect("load at deadline")
            .is_some());
        let reclaimed = journal
            .cleanup_terminal_records(NOW + 11)
            .expect("expired cleanup");
        assert_eq!(reclaimed.rows_removed, 1);
        assert!(reclaimed.bytes_removed > 0);
        assert!(journal
            .load(&record.route_id)
            .expect("load reclaimed")
            .is_none());
    }

    #[test]
    fn cleanup_never_reclaims_unresolved_rows_and_releases_quota() {
        let journal = journal_with_limits(3, 8192, 10, 3);
        let prepared = journal_record(0x86, 0x87, AnonymousMailboxSourcePhase::Prepared);
        let armed = journal_record(0x88, 0x89, AnonymousMailboxSourcePhase::Prepared);
        let ambiguous = journal_record(0x8a, 0x8b, AnonymousMailboxSourcePhase::Prepared);
        journal.insert_or_exact(&prepared).expect("prepared");
        journal.insert_or_exact(&armed).expect("armed insert");
        journal
            .transition_at(
                &armed,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Armed,
                armed.state.clone(),
                NOW,
            )
            .expect("arm");
        journal
            .insert_or_exact(&ambiguous)
            .expect("ambiguous insert");
        journal
            .transition_at(
                &ambiguous,
                AnonymousMailboxSourcePhase::Prepared,
                AnonymousMailboxSourcePhase::Ambiguous,
                ambiguous.state.clone(),
                NOW,
            )
            .expect("mark ambiguous");
        assert_eq!(
            journal
                .cleanup_terminal_records(i64::MAX as u64)
                .expect("unresolved cleanup"),
            AnonymousMailboxSourceCleanupReport::default()
        );
        for route_id in [prepared.route_id, armed.route_id, ambiguous.route_id] {
            let loaded = journal.load(&route_id).expect("load unresolved");
            assert!(loaded.is_some());
            assert_eq!(loaded.expect("unresolved row").retain_until, None);
        }

        let quota = journal_with_limits(1, 4096, 10, 1);
        let expired = journal_record(0x8c, 0x8d, AnonymousMailboxSourcePhase::Prepared);
        retain_terminal_record(
            &quota,
            &expired,
            AnonymousMailboxSourcePhase::Rejected,
            NOW,
            None,
        );
        let replacement = journal_record(0x8e, 0x8f, AnonymousMailboxSourcePhase::Prepared);
        assert!(matches!(
            quota.insert_or_exact(&replacement),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
        assert_eq!(
            quota
                .cleanup_terminal_records(NOW + 11)
                .expect("quota cleanup")
                .rows_removed,
            1
        );
        quota
            .insert_or_exact(&replacement)
            .expect("quota released after atomic cleanup");
    }

    #[test]
    fn cleanup_is_bounded_and_rolls_back_row_and_meta_together() {
        let journal = journal_with_limits(3, 8192, 10, 1);
        let first = journal_record(0x90, 0x91, AnonymousMailboxSourcePhase::Prepared);
        let second = journal_record(0x92, 0x93, AnonymousMailboxSourcePhase::Prepared);
        retain_terminal_record(
            &journal,
            &first,
            AnonymousMailboxSourcePhase::Completed,
            NOW,
            Some(&[0x94; 8]),
        );
        retain_terminal_record(
            &journal,
            &second,
            AnonymousMailboxSourcePhase::Rejected,
            NOW,
            None,
        );
        let before = load_source_meta(&journal.connection.lock()).expect("meta before cleanup");
        journal
            .connection
            .lock()
            .execute_batch(
                "CREATE TRIGGER abort_source_cleanup BEFORE DELETE
                 ON anonymous_mailbox_source_journal
                 BEGIN SELECT RAISE(ABORT, 'bounded rollback fixture'); END;",
            )
            .expect("install rollback trigger");
        assert!(matches!(
            journal.cleanup_terminal_records(NOW + 11),
            Err(AnonymousMailboxSourceError::Unavailable)
        ));
        assert_eq!(
            load_source_meta(&journal.connection.lock()).expect("meta after rollback"),
            before
        );
        assert!(journal
            .load(&first.route_id)
            .expect("first after rollback")
            .is_some());
        assert!(journal
            .load(&second.route_id)
            .expect("second after rollback")
            .is_some());
        journal
            .connection
            .lock()
            .execute_batch("DROP TRIGGER abort_source_cleanup;")
            .expect("drop rollback trigger");

        assert_eq!(
            journal
                .cleanup_terminal_records(NOW + 11)
                .expect("first bounded batch")
                .rows_removed,
            1
        );
        assert_eq!(
            journal
                .cleanup_terminal_records(NOW + 11)
                .expect("batch plus one")
                .rows_removed,
            1
        );
        assert_eq!(
            journal
                .cleanup_terminal_records(NOW + 11)
                .expect("empty cleanup")
                .rows_removed,
            0
        );
    }

    #[test]
    fn legacy_v1_migration_grants_full_window_and_preserves_unresolved_rows() {
        let sealer = journal();
        let mut completed = journal_record(0x95, 0x96, AnonymousMailboxSourcePhase::Completed);
        completed.state = encode_state(
            &completed.body,
            &decode_state(&completed.state)
                .expect("decode completed fixture")
                .terminal_frame,
            None,
            Some(&[0x97; 19]),
        )
        .expect("completed fixture state");
        let armed = journal_record(0x98, 0x99, AnonymousMailboxSourcePhase::Armed);
        let mut connection = Connection::open_in_memory().expect("legacy sqlite");
        create_legacy_source_schema(&connection);
        insert_legacy_source_record(&connection, &sealer, &completed);
        insert_legacy_source_record(&connection, &sealer, &armed);

        initialize_or_verify_source_schema(&mut connection, NOW, 10).expect("migrate v1");
        let version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("schema version");
        assert_eq!(version, SOURCE_JOURNAL_SCHEMA_VERSION);
        let terminal_deadline: Option<i64> = connection
            .query_row(
                "SELECT retain_until FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![completed.route_id.as_slice()],
                |row| row.get(0),
            )
            .expect("terminal deadline");
        assert_eq!(
            terminal_deadline,
            Some(source_i64(NOW + 10).expect("deadline"))
        );
        let unresolved_deadline: Option<i64> = connection
            .query_row(
                "SELECT retain_until FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![armed.route_id.as_slice()],
                |row| row.get(0),
            )
            .expect("unresolved deadline");
        assert_eq!(unresolved_deadline, None);

        let migrated = SqliteAnonymousMailboxSourceJournal {
            connection: Mutex::new(connection),
            journal_key: [0x22; 32],
            max_entries: 4,
            max_bytes: 4096,
            terminal_retention_secs: 10,
            cleanup_batch_size: 2,
            #[cfg(unix)]
            _database_parent: None,
        };
        migrated.audit_startup().expect("migrated startup audit");
        let replay = migrated
            .insert_or_exact(&journal_record(
                0x95,
                0x96,
                AnonymousMailboxSourcePhase::Prepared,
            ))
            .expect("legacy exact retry");
        assert_eq!(replay.phase, AnonymousMailboxSourcePhase::Completed);
        assert_eq!(replay.retain_until, Some(NOW + 10));
        assert_eq!(
            decode_state(&replay.state)
                .expect("legacy replay state")
                .completed,
            Some(vec![0x97; 19])
        );
        assert_eq!(
            migrated
                .cleanup_terminal_records(NOW + 10)
                .expect("migration boundary")
                .rows_removed,
            0
        );
        assert_eq!(
            migrated
                .cleanup_terminal_records(NOW + 11)
                .expect("migration expiry")
                .rows_removed,
            1
        );
        assert_eq!(
            migrated
                .load(&armed.route_id)
                .expect("legacy unresolved load")
                .expect("legacy unresolved row")
                .phase,
            AnonymousMailboxSourcePhase::Armed
        );
    }

    #[test]
    fn startup_audit_rejects_retention_and_meta_corruption() {
        let journal = journal_with_limits(2, 4096, 10, 2);
        let terminal = journal_record(0x9a, 0x9b, AnonymousMailboxSourcePhase::Prepared);
        retain_terminal_record(
            &journal,
            &terminal,
            AnonymousMailboxSourcePhase::Completed,
            NOW,
            Some(&[0x9c; 7]),
        );
        journal
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_source_journal SET retain_until = NULL
                 WHERE route_id = ?1",
                params![terminal.route_id.as_slice()],
            )
            .expect("corrupt terminal retention");
        assert!(matches!(
            journal.audit_startup(),
            Err(AnonymousMailboxSourceError::Corrupt)
        ));
        journal
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_source_journal SET retain_until = ?1
                 WHERE route_id = ?2",
                params![
                    source_i64(NOW + 10).expect("deadline"),
                    terminal.route_id.as_slice()
                ],
            )
            .expect("restore terminal retention");
        journal
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_source_meta SET total_entries = total_entries + 1
                 WHERE singleton = 1",
                [],
            )
            .expect("corrupt source meta");
        assert!(matches!(
            journal.audit_startup(),
            Err(AnonymousMailboxSourceError::Corrupt)
        ));
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
            retain_until: None,
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
            retain_until: None,
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
    #[ignore = "spawned only by the bounded source-journal crash drill"]
    fn source_journal_crash_drill_subprocess_worker() {
        let stage = std::env::var(SOURCE_CRASH_STAGE_ENV).expect("source crash drill stage");
        let db_path = PathBuf::from(
            std::env::var_os(SOURCE_CRASH_DB_ENV).expect("source crash drill database"),
        );
        let config = source_crash_config(&db_path);
        let route_id = [0x7a; 16];
        let journal = Arc::new(
            SqliteAnonymousMailboxSourceJournal::open(config, [0x7b; 32])
                .expect("open source crash journal"),
        );

        match stage.as_str() {
            "prepared" => {
                let target = target();
                let descriptor = source_crash_descriptor(&target);
                let commitment =
                    DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                        .expect("source crash descriptor commitment");
                let resolver = Arc::new(ExactOnlyResolver {
                    descriptor,
                    calls: AtomicUsize::new(0),
                });
                let coordinator = AnonymousMailboxSourceCoordinator::new(
                    Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source identity")),
                    resolver,
                    journal,
                );
                let _ = coordinator
                    .prepare(
                        ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
                        route_id,
                        ticket_request(&target),
                        NOW,
                    )
                    .expect("prepare should reach crash hook");
            }
            "armed" => {
                let (coordinator, _) = source_crash_coordinator(journal);
                let _ = coordinator
                    .begin_dispatch(route_id, NOW)
                    .expect("arming should reach crash hook");
            }
            "completed" => {
                let record = journal
                    .load(&route_id)
                    .expect("load armed source record")
                    .expect("armed source record");
                assert_eq!(record.phase, AnonymousMailboxSourcePhase::Armed);
                let decoded = decode_state(&record.state).expect("decode armed source state");
                let request = match decode_anonymous_mailbox_terminal_frame(&decoded.terminal_frame)
                    .expect("decode source request")
                {
                    AnonymousMailboxTerminalFrameV1::TicketIssue(request) => request,
                    _ => panic!("source crash fixture must retain ticket request"),
                };
                let target = target();
                let response = AnonymousMailboxTicketIssueResponseV1::signed(
                    &request,
                    AnonymousMailboxOutcomeV1::Rejected,
                    None,
                    NOW,
                    &target,
                )
                .expect("sign source crash response");
                let completed = encode_anonymous_mailbox_terminal_frame(
                    &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
                )
                .expect("encode source crash response");
                journal
                    .transition(
                        &record,
                        AnonymousMailboxSourcePhase::Armed,
                        AnonymousMailboxSourcePhase::Completed,
                        encode_state(
                            &record.body,
                            &decoded.terminal_frame,
                            None,
                            Some(&completed),
                        )
                        .expect("encode completed source state"),
                    )
                    .expect("completion should reach crash hook");
            }
            "tampered" => {
                drop(journal);
                let connection = Connection::open(&db_path).expect("open source row for tamper");
                connection
                    .execute_batch("PRAGMA synchronous=FULL; BEGIN IMMEDIATE;")
                    .expect("begin durable source tamper");
                let updated = connection
                    .execute(
                        "UPDATE anonymous_mailbox_source_journal
                         SET body = zeroblob(length(body)) WHERE route_id = ?1",
                        params![route_id.as_slice()],
                    )
                    .expect("tamper source body");
                assert_eq!(updated, 1);
                connection
                    .execute_batch("COMMIT;")
                    .expect("commit source tamper");
                crash_after_source_journal_commit(AnonymousMailboxSourcePhase::Prepared);
            }
            _ => panic!("unknown source crash drill stage"),
        }
        panic!("source crash drill missed its post-commit exit hook");
    }

    #[tokio::test]
    async fn source_journal_phases_survive_abrupt_exit_and_completed_never_redispatches() {
        // [ANONYMOUS-MAILBOX-SOURCE-CRASH-DRILL 2026-09-05 by Codex] Each
        // child exits immediately after a FULL-durability phase commit and
        // before the journal method returns. Reopening in the parent therefore
        // cannot inherit any process-local SQLite or source-session state.
        let directory = tempfile::tempdir().expect("source crash directory");
        let private_directory =
            std::fs::canonicalize(directory.path()).expect("canonical source crash directory");
        let db_path = private_directory.join("source-crash.sqlite3");

        let prepared_barrier = private_directory.join("prepared.barrier");
        let prepared =
            run_source_crash_child("prepared", "prepared", &db_path, &prepared_barrier).await;
        assert_source_crash_child(&prepared, &prepared_barrier);

        let prepared_journal =
            SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
                .expect("reopen prepared source journal");
        let prepared_record = prepared_journal
            .load(&[0x7a; 16])
            .expect("load prepared after crash")
            .expect("durable prepared record");
        assert_eq!(prepared_record.phase, AnonymousMailboxSourcePhase::Prepared);
        let prepared_state = decode_state(&prepared_record.state).expect("decode prepared state");
        let prepared_restart = prepared_state
            .restart
            .as_ref()
            .expect("prepared restart session")
            .encode_restart_state()
            .expect("encode prepared restart session")
            .as_bytes()
            .to_vec();
        let exact_body = prepared_record.body.clone();
        drop(prepared_journal);

        let armed_barrier = private_directory.join("armed.barrier");
        let armed = run_source_crash_child("armed", "armed", &db_path, &armed_barrier).await;
        assert_source_crash_child(&armed, &armed_barrier);

        let armed_journal =
            SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
                .expect("reopen armed source journal");
        let armed_record = armed_journal
            .load(&[0x7a; 16])
            .expect("load armed after crash")
            .expect("durable armed record");
        assert_eq!(armed_record.phase, AnonymousMailboxSourcePhase::Armed);
        assert_eq!(armed_record.body, exact_body, "armed body must be exact");
        let armed_state = decode_state(&armed_record.state).expect("decode armed state");
        assert_eq!(
            armed_state
                .restart
                .as_ref()
                .expect("armed restart session")
                .encode_restart_state()
                .expect("encode armed restart session")
                .as_bytes(),
            prepared_restart,
            "arming must retain the exact one-shot source session"
        );
        drop(armed_journal);

        let completed_barrier = private_directory.join("completed.barrier");
        let completed =
            run_source_crash_child("completed", "completed", &db_path, &completed_barrier).await;
        assert_source_crash_child(&completed, &completed_barrier);

        let completed_journal = Arc::new(
            SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
                .expect("reopen completed source journal"),
        );
        let completed_record = completed_journal
            .load(&[0x7a; 16])
            .expect("load completed after crash")
            .expect("durable completed record");
        assert_eq!(
            completed_record.phase,
            AnonymousMailboxSourcePhase::Completed
        );
        assert_eq!(completed_record.body, exact_body);
        let completed_state =
            decode_state(&completed_record.state).expect("decode completed state");
        assert!(completed_state.restart.is_none());
        let exact_completed = completed_state.completed.expect("completed response");
        let (coordinator, resolver) = source_crash_coordinator(completed_journal);
        assert!(matches!(
            coordinator.result([0x7a; 16]).expect("completed result"),
            AnonymousMailboxSourceResult::Completed(bytes) if bytes == exact_completed
        ));
        assert!(matches!(
            coordinator.begin_dispatch([0x7a; 16], NOW),
            Err(AnonymousMailboxSourceError::Rejected)
        ));
        assert_eq!(
            resolver.calls.load(Ordering::Relaxed),
            0,
            "completed restart must stop before target resolution or outbound release"
        );
    }

    #[tokio::test]
    async fn source_journal_crash_reopen_rejects_tampered_projection_before_outbound() {
        let directory = tempfile::tempdir().expect("source tamper crash directory");
        let private_directory = std::fs::canonicalize(directory.path())
            .expect("canonical source tamper crash directory");
        let db_path = private_directory.join("source-tamper-crash.sqlite3");
        let prepared_barrier = private_directory.join("tamper-prepared.barrier");
        let prepared =
            run_source_crash_child("prepared", "prepared", &db_path, &prepared_barrier).await;
        assert_source_crash_child(&prepared, &prepared_barrier);

        let tampered_barrier = private_directory.join("tampered.barrier");
        let tampered =
            run_source_crash_child("tampered", "prepared", &db_path, &tampered_barrier).await;
        assert_source_crash_child(&tampered, &tampered_barrier);

        let journal = match SqliteAnonymousMailboxSourceJournal::open(
            source_crash_config(&db_path),
            [0x7b; 32],
        ) {
            Err(AnonymousMailboxSourceError::Corrupt) => return,
            Err(error) => panic!("tampered startup returned unexpected error: {error}"),
            Ok(journal) => Arc::new(journal),
        };
        let (coordinator, resolver) = source_crash_coordinator(journal);
        assert!(matches!(
            coordinator.begin_dispatch([0x7a; 16], NOW),
            Err(AnonymousMailboxSourceError::Corrupt)
        ));
        assert_eq!(
            resolver.calls.load(Ordering::Relaxed),
            0,
            "tampered durable projections must fail before outbound resolution"
        );
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
            retain_until: None,
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
