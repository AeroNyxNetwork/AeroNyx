// ============================================
// File: crates/aeronyx-server/src/services/reverse_onion_queue.rs
// ============================================
//! Durable public-relay queue for adjacent-hop reverse-onion delivery.
//!
//! The relay persists one exact signed envelope for one immediate recipient.
//! That recipient later submits a core-authenticated Claim. The queue selects
//! only rows for the authenticated recipient, invokes the core Lease builder
//! inside one SQLite transaction, and commits the exact Lease bytes as Armed
//! before returning them. A Result is then accepted only for that immutable
//! Lease and retained as exact replay evidence. This is not the source-local
//! recipient journal and never invents a terminal route or wire codec.
//!
//! [REVERSE-ONION-QUEUE 2026-10-04 by Codex] Added the standalone additive
//! SQLite adapter with durable-before-claim admission, atomic signed-lease
//! issuance, exact route/body conflict checks, bounded quotas, and fail-closed
//! damaged-row handling. It remains unregistered until runtime ownership is
//! explicitly wired.
//!
//! [REVERSE-ONION-QUEUE-HARDENING 2026-10-04 by Codex] Deadline caps,
//! transactional schema/index ownership checks, combined replay quotas, and
//! bounded row-key validation are enforced before new admission.
//!
//! [REVERSE-ONION-QUEUE-OWNERSHIP 2026-10-04 by Codex] A fixed metadata
//! ownership/version marker is required before an existing database can be
//! opened; unmarked legacy data is preserved and returned as migration-needed.
//!
//! [REVERSE-ONION-QUEUE-RESULT-CONTEXT 2026-10-04 by Codex] Result recovery
//! callbacks receive the original bounded envelope and route binding loaded
//! from the durable row, never caller-supplied substitutions.
//!
//! [REVERSE-ONION-SOURCE-BINDING 2026-10-04 by Codex] Schema v3 carries an
//! optional authenticated source identity. Legacy rows remain readable but
//! are never backfilled or eligible for source-bound lookup.
//!
//! [REVERSE-ONION-QUEUE-DURABLE-CLOCK 2026-10-04 by Codex] The owned metadata
//! clock is a persisted local high-water. Only semantically observed
//! timestamps establish legacy history; future expiry fields never do.
//!
//! [REVERSE-ONION-RESULT-CONTEXT-LOOKUP 2026-10-04 by Codex] Result recovery
//! locates only an authenticated recipient/claim/lease/route tuple. This
//! read-only outcome never issues a lease, inserts no-work, or authorizes
//! completion without the core Result verifier.
//!
//! [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Schema v4 adds an exact
//! source-tuple covering index. Source reads validate only their bounded row;
//! startup and maintenance retain the full audit. No source read writes SQL.
//! Last Modified: v1.3.0-SourceIndex - Explicit v3/v4 index migration.

use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::protocol::onion::reverse_delivery::{
    MAX_REVERSE_ONION_CLAIM_BYTES, MAX_REVERSE_ONION_ENVELOPE_BYTES,
    MAX_REVERSE_ONION_FRAME_BYTES,
};
use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use thiserror::Error;

const TABLE: &str = "reverse_onion_delivery_queue_v1";
const NO_WORK_TABLE: &str = "reverse_onion_delivery_queue_v1_no_work";
const META_TABLE: &str = "reverse_onion_delivery_queue_v1_meta";
const SCHEMA_VERSION: i64 = 4;
const SOURCE_BINDING_SCHEMA_VERSION: i64 = 3;
const CLOCK_SCHEMA_VERSION: i64 = 2;
const LEGACY_SCHEMA_VERSION: i64 = 1;
const OWNERSHIP_TAG: &[u8] = b"AeroNyx-ReverseOnionQueue-v1";
const LEGACY_MAX_APPLICATION_OBJECTS: usize = 7;
const MAX_APPLICATION_OBJECTS: usize = 8;
const SOURCE_ROUTE_INDEX: &str = "idx_reverse_onion_delivery_queue_v1_source_route";
const MAX_APPLICATION_OBJECT_NAME_BYTES: i64 = 128;
const PENDING: i64 = 0;
const LEASED: i64 = 1; // Reserved for fail-closed recovery; never issued externally.
const ARMED: i64 = 2;
const RESULT: i64 = 3;
const TOMBSTONE: i64 = 4;
const ID_BYTES: usize = 16;
const COMMITMENT_BYTES: usize = 32;
// [REVERSE-ONION-QUEUE-CORE-BOUNDS 2026-10-04 by Codex] Queue limits consume
// the core's canonical frame/envelope bounds. Claim remains its exact fixed
// frame size; Lease/Result remain full frame bounds; aggregate reservation
// continues to distinguish per-item bytes from the envelope carrier bound.
const MAX_CLAIM_BYTES: usize = MAX_REVERSE_ONION_CLAIM_BYTES;
const MAX_LEASE_BYTES: usize = MAX_REVERSE_ONION_FRAME_BYTES;
const MAX_RESULT_BYTES: usize = MAX_REVERSE_ONION_FRAME_BYTES;
pub(crate) const MAX_REVERSE_ONION_QUEUE_ITEM_BYTES: usize =
    MAX_REVERSE_ONION_ENVELOPE_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ReverseOnionQueueError {
    #[error("reverse onion queue rejected input")]
    Rejected,
    #[error("reverse onion queue context conflicts")]
    Conflict,
    #[error("reverse onion queue is at capacity")]
    Capacity,
    #[error("reverse onion queue has no eligible work")]
    NoWork,
    #[error("reverse onion queue exposure is ambiguous")]
    Ambiguous,
    #[error("reverse onion queue lease was lost")]
    LeaseLost,
    #[error("reverse onion queue result is already complete")]
    AlreadyComplete,
    #[error("reverse onion queue storage is unavailable")]
    Unavailable,
    #[error("reverse onion queue contains corrupt state")]
    Corrupt,
    #[error("reverse onion queue requires explicit schema migration")]
    MigrationRequired,
}

impl From<rusqlite::Error> for ReverseOnionQueueError {
    fn from(_: rusqlite::Error) -> Self {
        Self::Unavailable
    }
}

/// Runtime limits mapped from `ReverseOnionQueueConfig` without duplicating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReverseOnionQueueLimits {
    pub(crate) max_items: u64,
    pub(crate) max_bytes: u64,
    pub(crate) max_items_per_recipient: u64,
    pub(crate) route_max_secs: u64,
    pub(crate) lease_max_secs: u64,
    pub(crate) recovery_retention_secs: u64,
}

impl ReverseOnionQueueLimits {
    pub(crate) fn new(
        max_items: u64,
        max_bytes: u64,
        max_items_per_recipient: u64,
        lease_max_secs: u64,
        recovery_retention_secs: u64,
    ) -> Result<Self, ReverseOnionQueueError> {
        if max_items == 0
            || max_items > i64::MAX as u64
            || max_bytes == 0
            || max_items_per_recipient == 0
            || max_items_per_recipient > i64::MAX as u64
            || lease_max_secs == 0
            || recovery_retention_secs == 0
            || max_bytes < (MAX_CLAIM_BYTES + MAX_LEASE_BYTES + MAX_RESULT_BYTES) as u64
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        Ok(Self {
            max_items,
            max_bytes,
            max_items_per_recipient,
            route_max_secs: lease_max_secs,
            lease_max_secs,
            recovery_retention_secs,
        })
    }

    /// Keeps the original constructor compatible while allowing route
    /// validity and execution lifetime to be configured independently.
    pub(crate) fn with_route_max_secs(
        mut self,
        route_max_secs: u64,
    ) -> Result<Self, ReverseOnionQueueError> {
        if route_max_secs == 0 {
            return Err(ReverseOnionQueueError::Rejected);
        }
        self.route_max_secs = route_max_secs;
        Ok(self)
    }
}

/// Exact source-enqueued envelope projection. `route_deadline` is computed by
/// authenticated route admission and is never supplied by a polling claimant.
pub(crate) struct ReverseOnionQueueItem {
    queue_key: [u8; COMMITMENT_BYTES],
    route_id: [u8; ID_BYTES],
    request_commitment: [u8; COMMITMENT_BYTES],
    source_node_id: [u8; COMMITMENT_BYTES],
    route_body_commitment: [u8; COMMITMENT_BYTES],
    immediate_recipient: [u8; COMMITMENT_BYTES],
    envelope_commitment: [u8; COMMITMENT_BYTES],
    envelope: Vec<u8>,
    route_deadline: u64,
}

impl ReverseOnionQueueItem {
    #[allow(clippy::too_many_arguments)]
    /// `source_node_id` is the authenticated original source for the
    /// explicitly supported direct source-to-recipient topology. Callers
    /// handling multihop routes must not substitute the previous hop here.
    pub(crate) fn new(
        queue_key: [u8; COMMITMENT_BYTES],
        route_id: [u8; ID_BYTES],
        request_commitment: [u8; COMMITMENT_BYTES],
        source_node_id: [u8; COMMITMENT_BYTES],
        route_body_commitment: [u8; COMMITMENT_BYTES],
        immediate_recipient: [u8; COMMITMENT_BYTES],
        envelope_commitment: [u8; COMMITMENT_BYTES],
        envelope: Vec<u8>,
        route_deadline: u64,
    ) -> Result<Self, ReverseOnionQueueError> {
        if is_zero(&queue_key)
            || is_zero(&route_id)
            || is_zero(&request_commitment)
            || !valid_source_node_id(&source_node_id)
            || is_zero(&route_body_commitment)
            || is_zero(&immediate_recipient)
            || is_zero(&envelope_commitment)
            || envelope.is_empty()
            || envelope.len() > MAX_REVERSE_ONION_QUEUE_ITEM_BYTES
            || route_deadline == 0
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        Ok(Self {
            queue_key,
            route_id,
            request_commitment,
            source_node_id,
            route_body_commitment,
            immediate_recipient,
            envelope_commitment,
            envelope,
            route_deadline,
        })
    }
}

/// Core-created Lease bytes. The queue stores and replays these bytes; it does
/// not sign, decode, or regenerate them.
pub(crate) struct ReverseOnionQueueLeaseMaterial {
    lease_id: [u8; ID_BYTES],
    lease_commitment: [u8; COMMITMENT_BYTES],
    frame: Vec<u8>,
    execution_deadline: u64,
    replay_evidence_deadline: u64,
}

impl ReverseOnionQueueLeaseMaterial {
    pub(crate) fn new(
        lease_id: [u8; ID_BYTES],
        lease_commitment: [u8; COMMITMENT_BYTES],
        frame: Vec<u8>,
        execution_deadline: u64,
        replay_evidence_deadline: u64,
    ) -> Result<Self, ReverseOnionQueueError> {
        if is_zero(&lease_id)
            || is_zero(&lease_commitment)
            || frame.is_empty()
            || frame.len() > MAX_LEASE_BYTES
            || execution_deadline == 0
            || replay_evidence_deadline < execution_deadline
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        Ok(Self {
            lease_id,
            lease_commitment,
            frame,
            execution_deadline,
            replay_evidence_deadline,
        })
    }
}

pub(crate) struct ReverseOnionQueueIssuedLease {
    queue_key: [u8; COMMITMENT_BYTES],
    envelope_commitment: [u8; COMMITMENT_BYTES],
    lease_id: [u8; ID_BYTES],
    lease_commitment: [u8; COMMITMENT_BYTES],
    frame: Vec<u8>,
}

impl ReverseOnionQueueIssuedLease {
    pub(crate) fn frame(&self) -> &[u8] {
        &self.frame
    }
}

pub(crate) struct ReverseOnionQueueResult {
    frame: Vec<u8>,
    commitment: [u8; COMMITMENT_BYTES],
}

impl ReverseOnionQueueResult {
    pub(crate) fn frame(&self) -> &[u8] {
        &self.frame
    }

    pub(crate) fn commitment(&self) -> [u8; COMMITMENT_BYTES] {
        self.commitment
    }
}

/// Opaque stored item passed to the core Lease builder. No raw route parsing.
pub(crate) struct ReverseOnionQueueStoredItem {
    queue_key: [u8; COMMITMENT_BYTES],
    route_id: [u8; ID_BYTES],
    request_commitment: [u8; COMMITMENT_BYTES],
    source_node_id: Option<[u8; COMMITMENT_BYTES]>,
    route_body_commitment: [u8; COMMITMENT_BYTES],
    immediate_recipient: [u8; COMMITMENT_BYTES],
    envelope_commitment: [u8; COMMITMENT_BYTES],
    envelope: Vec<u8>,
    route_deadline: u64,
}

impl ReverseOnionQueueStoredItem {
    pub(crate) fn envelope(&self) -> &[u8] {
        &self.envelope
    }
    pub(crate) fn queue_key(&self) -> [u8; COMMITMENT_BYTES] {
        self.queue_key
    }
    pub(crate) fn route_id(&self) -> [u8; ID_BYTES] {
        self.route_id
    }
    pub(crate) fn request_commitment(&self) -> [u8; COMMITMENT_BYTES] {
        self.request_commitment
    }
    pub(crate) fn source_node_id(&self) -> Option<[u8; COMMITMENT_BYTES]> {
        self.source_node_id
    }
    pub(crate) fn route_body_commitment(&self) -> [u8; COMMITMENT_BYTES] {
        self.route_body_commitment
    }
    pub(crate) fn immediate_recipient(&self) -> [u8; COMMITMENT_BYTES] {
        self.immediate_recipient
    }
    pub(crate) fn envelope_commitment(&self) -> [u8; COMMITMENT_BYTES] {
        self.envelope_commitment
    }
    pub(crate) fn route_deadline(&self) -> u64 {
        self.route_deadline
    }
}

/// Bounded immutable evidence selected by an exact authenticated source,
/// route, and original request tuple. The envelope is deliberately omitted.
pub(crate) struct ReverseOnionQueueSourceSnapshot {
    source_node_id: [u8; COMMITMENT_BYTES],
    route_id: [u8; ID_BYTES],
    request_commitment: [u8; COMMITMENT_BYTES],
    immediate_recipient: [u8; COMMITMENT_BYTES],
    route_deadline: u64,
    claim_frame: Option<Vec<u8>>,
    lease_frame: Option<Vec<u8>>,
    result_frame: Option<Vec<u8>>,
}

impl ReverseOnionQueueSourceSnapshot {
    pub(crate) fn source_node_id(&self) -> [u8; COMMITMENT_BYTES] { self.source_node_id }
    pub(crate) fn route_id(&self) -> [u8; ID_BYTES] { self.route_id }
    pub(crate) fn request_commitment(&self) -> [u8; COMMITMENT_BYTES] { self.request_commitment }
    pub(crate) fn immediate_recipient(&self) -> [u8; COMMITMENT_BYTES] { self.immediate_recipient }
    pub(crate) fn route_deadline(&self) -> u64 { self.route_deadline }
    pub(crate) fn claim_frame(&self) -> Option<&[u8]> { self.claim_frame.as_deref() }
    pub(crate) fn lease_frame(&self) -> Option<&[u8]> { self.lease_frame.as_deref() }
    pub(crate) fn result_frame(&self) -> Option<&[u8]> { self.result_frame.as_deref() }
}

/// Exact persisted Claim/Lease context supplied to the core Result verifier.
/// This adapter does not decode either frame or derive a replacement deadline.
pub(crate) struct ReverseOnionQueueStoredResultContext {
    claim_frame: Vec<u8>,
    lease_frame: Vec<u8>,
    envelope: Vec<u8>,
    envelope_commitment: [u8; COMMITMENT_BYTES],
    route_id: [u8; ID_BYTES],
    route_deadline: u64,
}

impl ReverseOnionQueueStoredResultContext {
    fn from_row(row: &StoredRow) -> Result<Self, ReverseOnionQueueError> {
        Ok(Self {
            claim_frame: row
                .claim_frame
                .clone()
                .ok_or(ReverseOnionQueueError::Corrupt)?,
            lease_frame: row
                .lease_frame
                .clone()
                .ok_or(ReverseOnionQueueError::Corrupt)?,
            envelope: row.envelope.clone(),
            envelope_commitment: row
                .envelope_commitment
                .as_slice()
                .try_into()
                .map_err(|_| ReverseOnionQueueError::Corrupt)?,
            route_id: row
                .route_id
                .as_slice()
                .try_into()
                .map_err(|_| ReverseOnionQueueError::Corrupt)?,
            route_deadline: u64::try_from(row.route_deadline)
                .map_err(|_| ReverseOnionQueueError::Corrupt)?,
        })
    }

    pub(crate) fn claim_frame(&self) -> &[u8] {
        &self.claim_frame
    }

    pub(crate) fn lease_frame(&self) -> &[u8] {
        &self.lease_frame
    }

    pub(crate) fn envelope(&self) -> &[u8] {
        &self.envelope
    }

    pub(crate) fn envelope_commitment(&self) -> [u8; COMMITMENT_BYTES] {
        self.envelope_commitment
    }

    pub(crate) fn route_id(&self) -> [u8; ID_BYTES] {
        self.route_id
    }

    pub(crate) fn route_deadline(&self) -> u64 {
        self.route_deadline
    }
}

pub(crate) enum ReverseOnionQueueIssue {
    Issued(ReverseOnionQueueIssuedLease),
    Existing(ReverseOnionQueueIssuedLease),
    Result(ReverseOnionQueueResult),
    Ambiguous,
    NoWork,
}

/// Read-only result recovery context selected by authenticated Result fields.
/// `Armed` and `Result` carry durable material; neither authorizes completion
/// without the core verifier and the queue's own CAS transaction.
pub(crate) enum ReverseOnionQueueResultContext {
    Armed(ReverseOnionQueueIssuedLease),
    Result(ReverseOnionQueueResult),
    NoWork,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionQueueAdmission {
    Created,
    Pending,
    Armed,
    Result,
    Tombstone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionQueueCompletion {
    Stored,
    AlreadyComplete,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SqliteReverseOnionQueue {
    limits: ReverseOnionQueueLimits,
}

impl SqliteReverseOnionQueue {
    pub(crate) const fn new(limits: ReverseOnionQueueLimits) -> Self {
        Self { limits }
    }

    pub(crate) fn initialize(
        &self,
        connection: &Mutex<Connection>,
    ) -> Result<(), ReverseOnionQueueError> {
        self.initialize_inner(connection, None)
    }

    pub(crate) fn initialize_at(
        &self,
        connection: &Mutex<Connection>,
        trusted_now: u64,
    ) -> Result<(), ReverseOnionQueueError> {
        validate_now(trusted_now)?;
        self.initialize_inner(connection, Some(trusted_now))
    }

    fn initialize_inner(
        &self,
        connection: &Mutex<Connection>,
        trusted_now: Option<u64>,
    ) -> Result<(), ReverseOnionQueueError> {
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let has_objects = has_application_objects(&tx)?;
        let meta_object = named_object_type(&tx, META_TABLE)?;
        if meta_object.is_none() {
            if has_objects {
                return Err(ReverseOnionQueueError::MigrationRequired);
            }
            create_meta_schema(&tx)?;
            create_main_schema(&tx)?;
            create_no_work_schema(&tx)?;
            create_source_route_index(&tx)?;
        } else {
            if meta_object.as_deref() != Some("table") {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            let objects = application_objects(&tx)?;
            validate_owned_objects(&objects)?;
            let version = metadata_schema_version(&tx)?;
            if version < SCHEMA_VERSION
                && (objects.len() > LEGACY_MAX_APPLICATION_OBJECTS
                    || objects.iter().any(|name| name == SOURCE_ROUTE_INDEX))
            {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            if named_object_type(&tx, TABLE)?.as_deref() != Some("table")
                || named_object_type(&tx, NO_WORK_TABLE)?.as_deref() != Some("table")
            {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            match version {
                LEGACY_SCHEMA_VERSION => {
                    let Some(trusted_now) = trusted_now else {
                        return Err(ReverseOnionQueueError::MigrationRequired);
                    };
                    validate_meta_schema_v1(&tx)?;
                    validate_schema_legacy(&tx)?;
                    validate_no_work_schema(&tx)?;
                    migrate_meta_v1_to_v2(&tx, trusted_now, self.limits.max_items)?;
                    migrate_source_binding_v2_to_v3(&tx)?;
                    validate_all_rows(&tx, &self.limits)?;
                    validate_all_no_work_rows(&tx, &self.limits)?;
                }
                CLOCK_SCHEMA_VERSION => {
                    validate_meta_schema_version(&tx, CLOCK_SCHEMA_VERSION)?;
                    validate_schema_legacy(&tx)?;
                    validate_no_work_schema(&tx)?;
                    migrate_source_binding_v2_to_v3(&tx)?;
                    validate_all_rows(&tx, &self.limits)?;
                    validate_all_no_work_rows(&tx, &self.limits)?;
                }
                SOURCE_BINDING_SCHEMA_VERSION => {
                    validate_meta_schema_version(&tx, SOURCE_BINDING_SCHEMA_VERSION)?;
                    validate_schema(&tx)?;
                    validate_no_work_schema(&tx)?;
                }
                SCHEMA_VERSION => {
                    validate_meta_schema(&tx)?;
                    // A v4 database cannot silently recreate a missing index.
                    if named_object_type(&tx, SOURCE_ROUTE_INDEX)?.as_deref() != Some("index") {
                        return Err(ReverseOnionQueueError::Corrupt);
                    }
                }
                _ => return Err(ReverseOnionQueueError::Corrupt),
            }
            if version < SCHEMA_VERSION {
                validate_all_rows(&tx, &self.limits)?;
                validate_all_no_work_rows(&tx, &self.limits)?;
                migrate_source_index_v3_to_v4(&tx)?;
            }
        }
        let existing_main_indexes = validate_schema(&tx)?;
        if !existing_main_indexes {
            create_main_indexes(&tx)?;
        }
        let existing_no_work_indexes = validate_no_work_schema(&tx)?;
        if !existing_no_work_indexes {
            create_no_work_indexes(&tx)?;
        }
        validate_owned_objects(&application_objects(&tx)?)?;
        validate_meta_schema(&tx)?;
        validate_schema(&tx)?;
        validate_no_work_schema(&tx)?;
        validate_all_rows(&tx, &self.limits)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        tx.commit()?;
        Ok(())
    }

    /// Source admission. The authenticated route deadline is persisted here;
    /// a claimant cannot extend it later.
    pub(crate) fn enqueue(
        &self,
        connection: &Mutex<Connection>,
        item: &ReverseOnionQueueItem,
        now: u64,
    ) -> Result<ReverseOnionQueueAdmission, ReverseOnionQueueError> {
        validate_now(now)?;
        let route_limit = now
            .checked_add(self.limits.route_max_secs)
            .ok_or(ReverseOnionQueueError::Rejected)?;
        let now = sqlite_integer(now)?;
        let route_deadline = sqlite_integer(item.route_deadline)?;
        if route_deadline <= now {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, now)?;
        validate_all_rows(&tx, &self.limits)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        cleanup_expired(&tx, now)?;
        if let Some(row) = load_row(&tx, &item.queue_key)? {
            validate_row(&row)?;
            if row.state == TOMBSTONE {
                ensure_item_identity(&row, item)?;
            } else {
                ensure_item_context(&row, item)?;
            }
            let admission = admission_for_state(row.state);
            tx.commit()?;
            return Ok(admission);
        }
        if item.route_deadline > route_limit {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let incoming = u64::try_from(item.envelope.len())
            .ok()
            .and_then(|value| value.checked_add(MAX_CLAIM_BYTES as u64))
            .and_then(|value| value.checked_add(MAX_LEASE_BYTES as u64))
            .and_then(|value| value.checked_add(MAX_RESULT_BYTES as u64))
            .ok_or(ReverseOnionQueueError::Rejected)?;
        enforce_quotas(&tx, &self.limits, &item.immediate_recipient, incoming)?;
        tx.execute(
            &format!(
                "INSERT INTO {TABLE} (
                    queue_key, route_id, request_commitment, route_body_commitment,
                    immediate_recipient, envelope_commitment, envelope, state,
                    claim_id, claim_commitment, claim_frame, lease_id,
                    lease_commitment, lease_frame, execution_deadline, route_deadline,
                    result_frame, result_commitment, completed_at, retained_until,
                    source_node_id
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                           NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?9,
                           NULL, NULL, NULL, ?9, ?10)"
            ),
            params![
                item.queue_key.as_slice(),
                item.route_id.as_slice(),
                item.request_commitment.as_slice(),
                item.route_body_commitment.as_slice(),
                item.immediate_recipient.as_slice(),
                item.envelope_commitment.as_slice(),
                item.envelope.as_slice(),
                PENDING,
                route_deadline,
                item.source_node_id.as_slice(),
            ],
        )?;
        tx.commit()?;
        Ok(ReverseOnionQueueAdmission::Created)
    }

    /// Recipient-authenticated poll and atomic Lease issue. The closure must
    /// call the core `ReverseOnionFrameV1::lease`; its exact bytes, lease
    /// commitment, and deadlines are committed before the method returns.
    pub(crate) fn issue_lease<V, F>(
        &self,
        connection: &Mutex<Connection>,
        recipient: [u8; COMMITMENT_BYTES],
        claim_id: [u8; ID_BYTES],
        claim_commitment: [u8; COMMITMENT_BYTES],
        claim_frame: Vec<u8>,
        now: u64,
        verify_claim: V,
        build_lease: F,
    ) -> Result<ReverseOnionQueueIssue, ReverseOnionQueueError>
    where
        V: Fn(&[u8]) -> Result<[u8; COMMITMENT_BYTES], ReverseOnionQueueError>,
        F: FnOnce(&ReverseOnionQueueStoredItem, &[u8])
            -> Result<ReverseOnionQueueLeaseMaterial, ReverseOnionQueueError>,
    {
        validate_now(now)?;
        let now_u64 = now;
        if is_zero(&recipient)
            || is_zero(&claim_id)
            || is_zero(&claim_commitment)
            || claim_frame.is_empty()
            || claim_frame.len() > MAX_CLAIM_BYTES
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let now = sqlite_integer(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, now)?;
        validate_all_rows(&tx, &self.limits)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        cleanup_expired(&tx, now)?;
        if let Some(marker) = load_no_work(&tx, &claim_id)? {
            if load_by_claim_id(&tx, &claim_id)?.is_some() {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            if marker.immediate_recipient.as_slice() != recipient.as_slice()
                || marker.claim_commitment.as_slice() != claim_commitment.as_slice()
                || marker.claim_frame.as_slice() != claim_frame.as_slice()
            {
                return Err(ReverseOnionQueueError::Conflict);
            }
            tx.commit()?;
            return Ok(ReverseOnionQueueIssue::NoWork);
        }
        if let Some(existing) = load_by_claim_id(&tx, &claim_id)? {
            validate_row(&existing)?;
            if existing.immediate_recipient.as_slice() != recipient.as_slice()
                || existing.claim_commitment.as_deref() != Some(claim_commitment.as_slice())
                || (existing.state != TOMBSTONE
                    && existing.claim_frame.as_deref() != Some(claim_frame.as_slice()))
            {
                return Err(ReverseOnionQueueError::Conflict);
            }
            let issue = match existing.state {
                ARMED => ReverseOnionQueueIssue::Existing(issued_lease_from_row(&existing)?),
                RESULT => ReverseOnionQueueIssue::Result(result_from_row(&existing)?),
                LEASED | TOMBSTONE => ReverseOnionQueueIssue::Ambiguous,
                PENDING => return Err(ReverseOnionQueueError::Corrupt),
                _ => return Err(ReverseOnionQueueError::Corrupt),
            };
            tx.commit()?;
            return Ok(issue);
        }
        let Some(row) = select_pending(&tx, &recipient)? else {
            let verified = verify_claim(&claim_frame)?;
            if verified != claim_commitment || is_zero(&verified) {
                return Err(ReverseOnionQueueError::Rejected);
            }
            insert_no_work_marker(
                &tx,
                &self.limits,
                &recipient,
                &claim_id,
                &claim_commitment,
                &claim_frame,
                now,
            )?;
            tx.commit()?;
            return Ok(ReverseOnionQueueIssue::NoWork);
        };
        validate_row(&row)?;
        if row.route_deadline <= now {
            let verified = verify_claim(&claim_frame)?;
            if verified != claim_commitment || is_zero(&verified) {
                return Err(ReverseOnionQueueError::Rejected);
            }
            insert_no_work_marker(
                &tx,
                &self.limits,
                &recipient,
                &claim_id,
                &claim_commitment,
                &claim_frame,
                now,
            )?;
            tx.commit()?;
            return Ok(ReverseOnionQueueIssue::NoWork);
        }
        let verified = verify_claim(&claim_frame)?;
        if verified != claim_commitment || is_zero(&verified) {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let stored = stored_item_from_row(&row)?;
        let material = build_lease(&stored, &claim_frame)?;
        let execution_deadline = sqlite_integer(material.execution_deadline)?;
        let retained_until = sqlite_integer(material.replay_evidence_deadline)?;
        let execution_limit = now_u64
            .checked_add(self.limits.lease_max_secs)
            .ok_or(ReverseOnionQueueError::Rejected)?;
        let mandatory_retention = now_u64
            .checked_add(self.limits.recovery_retention_secs)
            .ok_or(ReverseOnionQueueError::Rejected)?;
        if execution_deadline <= now
            || retained_until < execution_deadline
            || execution_deadline > row.route_deadline
            || material.execution_deadline > execution_limit
            || material.replay_evidence_deadline < mandatory_retention
            || material.replay_evidence_deadline < material.execution_deadline
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        if tx.execute(
            &format!(
                "UPDATE {TABLE}
                 SET state = ?1, claim_id = ?2, claim_commitment = ?3,
                     claim_frame = ?4, lease_id = ?5, lease_commitment = ?6,
                     lease_frame = ?7, execution_deadline = ?8,
                     retained_until = ?9
                 WHERE queue_key = ?10 AND state = ?11
                   AND immediate_recipient = ?12"
            ),
            params![
                ARMED,
                claim_id.as_slice(),
                claim_commitment.as_slice(),
                claim_frame.as_slice(),
                material.lease_id.as_slice(),
                material.lease_commitment.as_slice(),
                material.frame.as_slice(),
                execution_deadline,
                retained_until,
                row.queue_key.as_slice(),
                PENDING,
                recipient.as_slice(),
            ],
        )? != 1
        {
            return Err(ReverseOnionQueueError::LeaseLost);
        }
        tx.commit()?;
        Ok(ReverseOnionQueueIssue::Issued(ReverseOnionQueueIssuedLease {
            queue_key: row.queue_key.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?,
            envelope_commitment: row
                .envelope_commitment
                .as_slice()
                .try_into()
                .map_err(|_| ReverseOnionQueueError::Corrupt)?,
            lease_id: material.lease_id,
            lease_commitment: material.lease_commitment,
            frame: material.frame,
        }))
    }

    /// Reconstructs the exact persisted lease handle after restart. No new
    /// lease is issued and no claim freshness check is repeated here; the
    /// later core Result verifier uses the persisted claim/lease frames and
    /// route deadline supplied by `complete`'s context callback.
    pub(crate) fn lookup_armed(
        &self,
        connection: &Mutex<Connection>,
        queue_key: [u8; COMMITMENT_BYTES],
        envelope_commitment: [u8; COMMITMENT_BYTES],
        recipient: [u8; COMMITMENT_BYTES],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueIssuedLease>, ReverseOnionQueueError> {
        validate_now(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, sqlite_integer(now)?)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        let result = if let Some(row) = load_row(&tx, &queue_key)? {
            validate_row(&row)?;
            if row.envelope_commitment.as_slice() != envelope_commitment.as_slice()
                || row.immediate_recipient.as_slice() != recipient.as_slice()
            {
                return Err(ReverseOnionQueueError::Conflict);
            }
            if row.state != ARMED || row.retained_until <= sqlite_integer(now)? {
                None
            } else {
                Some(issued_lease_from_row(&row)?)
            }
        } else {
            None
        };
        tx.commit()?;
        Ok(result)
    }

    /// Locates an existing result context without issuing, inserting, or
    /// reassigning any row. Unknown or mismatched identity is intentionally
    /// indistinguishable from NoWork so another recipient's state cannot be
    /// probed. The returned Armed/Result material is not completion authority.
    pub(crate) fn lookup_result_context(
        &self,
        connection: &Mutex<Connection>,
        recipient: [u8; COMMITMENT_BYTES],
        claim_id: [u8; ID_BYTES],
        lease_id: [u8; ID_BYTES],
        route_id: [u8; ID_BYTES],
        now: u64,
    ) -> Result<ReverseOnionQueueResultContext, ReverseOnionQueueError> {
        validate_now(now)?;
        if is_zero(&recipient) || is_zero(&claim_id) || is_zero(&lease_id) || is_zero(&route_id) {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let now = sqlite_integer(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, now)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        let outcome = if let Some(row) = load_by_claim_id(&tx, &claim_id)? {
            let identity_matches = row.immediate_recipient.as_slice() == recipient.as_slice()
                && row.route_id.as_slice() == route_id.as_slice()
                && row.lease_id.as_deref() == Some(lease_id.as_slice());
            if !identity_matches {
                ReverseOnionQueueResultContext::NoWork
            } else {
                validate_row(&row)?;
                if row.state == ARMED {
                    if row.retained_until <= now {
                        ReverseOnionQueueResultContext::Ambiguous
                    } else {
                        ReverseOnionQueueResultContext::Armed(issued_lease_from_row(&row)?)
                    }
                } else if row.state == RESULT {
                    if row.retained_until <= now {
                        ReverseOnionQueueResultContext::NoWork
                    } else {
                        ReverseOnionQueueResultContext::Result(result_from_row(&row)?)
                    }
                } else if matches!(row.state, LEASED | TOMBSTONE) {
                    ReverseOnionQueueResultContext::Ambiguous
                } else {
                    ReverseOnionQueueResultContext::NoWork
                }
            }
        } else {
            ReverseOnionQueueResultContext::NoWork
        };
        tx.commit()?;
        Ok(outcome)
    }

    /// Stores a core-verified exact Result for the immutable Armed lease.
    pub(crate) fn complete(
        &self,
        connection: &Mutex<Connection>,
        lease: &ReverseOnionQueueIssuedLease,
        result_frame: &[u8],
        now: u64,
        verify_result: impl FnOnce(
            &ReverseOnionQueueStoredResultContext,
            &[u8],
        ) -> Result<[u8; COMMITMENT_BYTES], ReverseOnionQueueError>,
    ) -> Result<ReverseOnionQueueCompletion, ReverseOnionQueueError> {
        validate_now(now)?;
        if result_frame.is_empty() || result_frame.len() > MAX_RESULT_BYTES {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let now = sqlite_integer(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, now)?;
        validate_all_rows(&tx, &self.limits)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        let row = load_row(&tx, &lease.queue_key)?.ok_or(ReverseOnionQueueError::LeaseLost)?;
        validate_row(&row)?;
        if row.envelope_commitment.as_slice() != lease.envelope_commitment.as_slice()
            || row.lease_id.as_deref() != Some(lease.lease_id.as_slice())
            || row.lease_commitment.as_deref() != Some(lease.lease_commitment.as_slice())
        {
            return Err(ReverseOnionQueueError::LeaseLost);
        }
        if row.state == RESULT {
            if row.retained_until <= now {
                // [REVERSE-ONION-QUEUE-DURABLE-ERROR-FENCE 2026-10-04 by Codex]
                // The authenticated retained row has been observed as expired;
                // commit only the clock and publish NoWork after the DB wrapper
                // performs its durable filesystem fence.
                tx.commit()?;
                return Err(ReverseOnionQueueError::NoWork);
            }
            let stored = result_from_row(&row)?;
            return if stored.frame == result_frame {
                tx.commit()?;
                Ok(ReverseOnionQueueCompletion::AlreadyComplete)
            } else {
                Err(ReverseOnionQueueError::Conflict)
            };
        }
        if row.state != ARMED {
            return Err(ReverseOnionQueueError::LeaseLost);
        }
        if row.retained_until <= now {
            // [REVERSE-ONION-QUEUE-DURABLE-ERROR-FENCE 2026-10-04 by Codex]
            // Preserve the trusted expiry observation without mutating the
            // still-armed row; the DB wrapper fences before returning Ambiguous.
            tx.commit()?;
            return Err(ReverseOnionQueueError::Ambiguous);
        }
        let result_commitment = verify_result(
            &ReverseOnionQueueStoredResultContext::from_row(&row)?,
            result_frame,
        )?;
        if is_zero(&result_commitment) {
            return Err(ReverseOnionQueueError::Rejected);
        }
        if tx.execute(
            &format!(
                "UPDATE {TABLE}
                 SET state = ?1, result_frame = ?2, result_commitment = ?3,
                     completed_at = ?4
                 WHERE queue_key = ?5 AND state = ?6
                   AND lease_id = ?7 AND lease_commitment = ?8"
            ),
            params![
                RESULT,
                result_frame,
                result_commitment.as_slice(),
                now,
                lease.queue_key.as_slice(),
                ARMED,
                lease.lease_id.as_slice(),
                lease.lease_commitment.as_slice(),
            ],
        )? != 1
        {
            return Err(ReverseOnionQueueError::LeaseLost);
        }
        tx.commit()?;
        Ok(ReverseOnionQueueCompletion::Stored)
    }

    pub(crate) fn lookup_result(
        &self,
        connection: &Mutex<Connection>,
        queue_key: [u8; COMMITMENT_BYTES],
        envelope_commitment: [u8; COMMITMENT_BYTES],
        recipient: [u8; COMMITMENT_BYTES],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueResult>, ReverseOnionQueueError> {
        validate_now(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, sqlite_integer(now)?)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        let result = if let Some(row) = load_row(&tx, &queue_key)? {
            validate_row(&row)?;
            if row.envelope_commitment.as_slice() != envelope_commitment.as_slice()
                || row.immediate_recipient.as_slice() != recipient.as_slice()
            {
                return Err(ReverseOnionQueueError::Conflict);
            }
            if row.state != RESULT || row.retained_until <= sqlite_integer(now)? {
                None
            } else {
                Some(result_from_row(&row)?)
            }
        } else {
            None
        };
        tx.commit()?;
        Ok(result)
    }

    /// Returns only exact source-bound signed parts. Absence is an internal
    /// Option and is never serialized as an unsigned proof.
    pub(crate) fn lookup_source(
        &self,
        connection: &Mutex<Connection>,
        source_node_id: [u8; COMMITMENT_BYTES],
        route_id: [u8; ID_BYTES],
        request_commitment: [u8; COMMITMENT_BYTES],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueSourceSnapshot>, ReverseOnionQueueError> {
        validate_now(now)?;
        if !valid_source_node_id(&source_node_id)
            || is_zero(&route_id)
            || is_zero(&request_commitment)
        {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let now = sqlite_integer(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let clock_high_water: i64 = tx
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .map_err(|_| ReverseOnionQueueError::Corrupt)?;
        if clock_high_water < 0 || now < clock_high_water {
            return Err(ReverseOnionQueueError::Rejected);
        }
        let row = load_by_source_route(&tx, &source_node_id, &route_id, &request_commitment)?;
        let snapshot = if let Some(row) = row {
            validate_row(&row)?;
            let expired = match row.state {
                PENDING => row.route_deadline <= now,
                LEASED | ARMED | RESULT | TOMBSTONE => row.retained_until <= now,
                _ => true,
            };
            if expired {
                None
            } else {
                Some(ReverseOnionQueueSourceSnapshot {
                    source_node_id: <[u8; COMMITMENT_BYTES]>::try_from(
                        row.source_node_id.as_deref()
                            .ok_or(ReverseOnionQueueError::Rejected)?,
                    )
                    .map_err(|_| ReverseOnionQueueError::Corrupt)?,
                    route_id: row.route_id.as_slice().try_into()
                        .map_err(|_| ReverseOnionQueueError::Corrupt)?,
                    request_commitment: row.request_commitment.as_slice().try_into()
                        .map_err(|_| ReverseOnionQueueError::Corrupt)?,
                    immediate_recipient: row.immediate_recipient.as_slice().try_into()
                        .map_err(|_| ReverseOnionQueueError::Corrupt)?,
                    route_deadline: u64::try_from(row.route_deadline)
                        .map_err(|_| ReverseOnionQueueError::Corrupt)?,
                    claim_frame: row.claim_frame,
                    lease_frame: row.lease_frame,
                    result_frame: row.result_frame,
                })
            }
        } else {
            None
        };
        tx.commit()?;
        Ok(snapshot)
    }

    pub(crate) fn cleanup(
        &self,
        connection: &Mutex<Connection>,
        now: u64,
    ) -> Result<u64, ReverseOnionQueueError> {
        validate_now(now)?;
        let now = sqlite_integer(now)?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        advance_clock(&tx, now)?;
        validate_all_rows(&tx, &self.limits)?;
        validate_all_no_work_rows(&tx, &self.limits)?;
        let removed = cleanup_expired(&tx, now)?;
        tx.commit()?;
        Ok(removed)
    }
}

struct StoredRow {
    queue_key: Vec<u8>,
    route_id: Vec<u8>,
    request_commitment: Vec<u8>,
    source_node_id: Option<Vec<u8>>,
    route_body_commitment: Vec<u8>,
    immediate_recipient: Vec<u8>,
    envelope_commitment: Vec<u8>,
    envelope: Vec<u8>,
    state: i64,
    claim_id: Option<Vec<u8>>,
    claim_commitment: Option<Vec<u8>>,
    claim_frame: Option<Vec<u8>>,
    lease_id: Option<Vec<u8>>,
    lease_commitment: Option<Vec<u8>>,
    lease_frame: Option<Vec<u8>>,
    execution_deadline: Option<i64>,
    route_deadline: i64,
    result_frame: Option<Vec<u8>>,
    result_commitment: Option<Vec<u8>>,
    completed_at: Option<i64>,
    retained_until: i64,
}

struct NoWorkRow {
    claim_id: Vec<u8>,
    immediate_recipient: Vec<u8>,
    claim_commitment: Vec<u8>,
    claim_frame: Vec<u8>,
    recorded_at: i64,
    retained_until: i64,
}

fn load_no_work(
    connection: &Connection,
    claim_id: &[u8; ID_BYTES],
) -> Result<Option<NoWorkRow>, ReverseOnionQueueError> {
    let shape: Option<(String, Option<i64>, String, Option<i64>, String, Option<i64>, String, Option<i64>, i64, i64)> = connection
        .query_row(
            &format!(
                "SELECT typeof(claim_id), length(claim_id),
                        typeof(immediate_recipient), length(immediate_recipient),
                        typeof(claim_commitment), length(claim_commitment),
                        typeof(claim_frame), length(claim_frame),
                        recorded_at, retained_until
                 FROM {NO_WORK_TABLE} WHERE claim_id = ?1"
            ),
            params![claim_id.as_slice()],
            |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?, row.get(9)?,
                ))
            },
        )
        .optional()?;
    let Some((claim_class, claim_len, recipient_class, recipient_len, commitment_class, commitment_len, frame_class, frame_len, recorded_at, retained_until)) = shape else {
        return Ok(None);
    };
    if claim_class != "blob"
        || claim_len != Some(ID_BYTES as i64)
        || recipient_class != "blob"
        || recipient_len != Some(COMMITMENT_BYTES as i64)
        || commitment_class != "blob"
        || commitment_len != Some(COMMITMENT_BYTES as i64)
        || frame_class != "blob"
        || frame_len.is_none_or(|length| length <= 0 || usize::try_from(length).map_or(true, |value| value > MAX_CLAIM_BYTES))
        || recorded_at <= 0
        || retained_until < recorded_at
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    let row = connection.query_row(
        &format!(
            "SELECT claim_id, immediate_recipient, claim_commitment,
                    claim_frame, recorded_at, retained_until
             FROM {NO_WORK_TABLE} WHERE claim_id = ?1"
        ),
        params![claim_id.as_slice()],
        |row| {
            Ok(NoWorkRow {
                claim_id: row.get(0)?,
                immediate_recipient: row.get(1)?,
                claim_commitment: row.get(2)?,
                claim_frame: row.get(3)?,
                recorded_at: row.get(4)?,
                retained_until: row.get(5)?,
            })
        },
    )?;
    if row.claim_id.len() != ID_BYTES
        || row.immediate_recipient.len() != COMMITMENT_BYTES
        || row.claim_commitment.len() != COMMITMENT_BYTES
        || row.claim_frame.is_empty()
        || row.claim_frame.len() > MAX_CLAIM_BYTES
        || is_zero(&row.claim_id)
        || is_zero(&row.immediate_recipient)
        || is_zero(&row.claim_commitment)
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(Some(row))
}

fn insert_no_work_marker(
    connection: &Connection,
    limits: &ReverseOnionQueueLimits,
    recipient: &[u8; COMMITMENT_BYTES],
    claim_id: &[u8; ID_BYTES],
    claim_commitment: &[u8; COMMITMENT_BYTES],
    claim_frame: &[u8],
    now: i64,
) -> Result<(), ReverseOnionQueueError> {
    let retained_until = u64::try_from(now)
        .ok()
        .and_then(|value| value.checked_add(limits.recovery_retention_secs))
        .ok_or(ReverseOnionQueueError::Rejected)
        .and_then(sqlite_integer)?;
    let (main_count, main_bytes): (i64, i64) = connection.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(envelope)
                + COALESCE(LENGTH(claim_frame), 0)
                + COALESCE(LENGTH(lease_frame), 0)
                + COALESCE(LENGTH(result_frame), 0)), 0)
             FROM {TABLE}"
        ),
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let (marker_count, marker_bytes): (i64, i64) = connection.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(claim_frame) + 80), 0)
             FROM {NO_WORK_TABLE}"
        ),
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let count = main_count
        .checked_add(marker_count)
        .ok_or(ReverseOnionQueueError::Corrupt)?;
    let bytes = main_bytes
        .checked_add(marker_bytes)
        .ok_or(ReverseOnionQueueError::Corrupt)?;
    let count = u64::try_from(count).map_err(|_| ReverseOnionQueueError::Corrupt)?;
    let bytes = u64::try_from(bytes).map_err(|_| ReverseOnionQueueError::Corrupt)?;
    let incoming = u64::try_from(claim_frame.len())
        .ok()
        .and_then(|value| value.checked_add(80))
        .ok_or(ReverseOnionQueueError::Rejected)?;
    if count >= limits.max_items
        || bytes
            .checked_add(incoming)
            .is_none_or(|value| value > limits.max_bytes)
    {
        return Err(ReverseOnionQueueError::Capacity);
    }
    connection.execute(
        &format!(
            "INSERT INTO {NO_WORK_TABLE}
                (claim_id, immediate_recipient, claim_commitment, claim_frame,
                 recorded_at, retained_until)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        ),
        params![
            claim_id.as_slice(),
            recipient.as_slice(),
            claim_commitment.as_slice(),
            claim_frame,
            now,
            retained_until,
        ],
    )?;
    Ok(())
}

// [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Test-only per-thread seam
// counts bounded row projections, without timing or shared-test interference.
#[cfg(test)]
thread_local! {
    static SOURCE_TEST_ROW_LOADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn load_row(connection: &Connection, key: &[u8; COMMITMENT_BYTES]) -> Result<Option<StoredRow>, ReverseOnionQueueError> {
    #[cfg(test)]
    SOURCE_TEST_ROW_LOADS.with(|count| count.set(count.get() + 1));
    let Some(shape) = connection
        .query_row(
            &format!(
                "SELECT
                    typeof(queue_key), length(queue_key),
                    typeof(route_id), length(route_id),
                    typeof(request_commitment), length(request_commitment),
                    typeof(source_node_id), length(source_node_id),
                    typeof(route_body_commitment), length(route_body_commitment),
                    typeof(immediate_recipient), length(immediate_recipient),
                    typeof(envelope_commitment), length(envelope_commitment),
                    typeof(envelope), length(envelope),
                    typeof(claim_id), length(claim_id),
                    typeof(claim_commitment), length(claim_commitment),
                    typeof(claim_frame), length(claim_frame),
                    typeof(lease_id), length(lease_id),
                    typeof(lease_commitment), length(lease_commitment),
                    typeof(lease_frame), length(lease_frame),
                    typeof(result_frame), length(result_frame),
                    typeof(result_commitment), length(result_commitment),
                    state, route_deadline, execution_deadline,
                    completed_at, retained_until
                 FROM {TABLE} WHERE queue_key = ?1"
            ),
            params![key.as_slice()],
            |row| {
                let mut shapes = Vec::with_capacity(16);
                for index in 0..16 {
                    shapes.push((
                        row.get::<_, String>(index * 2)?,
                        row.get::<_, Option<i64>>(index * 2 + 1)?,
                    ));
                }
                Ok((
                    shapes,
                    row.get::<_, i64>(32)?,
                    row.get::<_, i64>(33)?,
                    row.get::<_, Option<i64>>(34)?,
                    row.get::<_, Option<i64>>(35)?,
                    row.get::<_, i64>(36)?,
                ))
            },
        )
        .optional()?
    else {
        return Ok(None);
    };
    let (shapes, state, route_deadline, execution_deadline, completed_at, retained_until) = shape;
    for (index, expected) in [
        (0, 32),
        (1, 16),
        (2, 32),
        (4, 32),
        (5, 32),
        (6, 32),
        (7, MAX_REVERSE_ONION_QUEUE_ITEM_BYTES),
    ] {
        if shapes[index].0 != "blob"
            || shapes[index].1.is_none_or(|length| {
                length < 0 || usize::try_from(length).map_or(true, |value| value > expected)
            })
            || index != 3 && index != 7
                && shapes[index].1 != Some(i64::try_from(expected).unwrap_or(i64::MAX))
        {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    if !optional_blob_shape(&shapes[3].0, shapes[3].1, COMMITMENT_BYTES) {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    for (index, maximum) in [
        (8, ID_BYTES),
        (9, COMMITMENT_BYTES),
        (10, MAX_CLAIM_BYTES),
        (11, ID_BYTES),
        (12, COMMITMENT_BYTES),
        (13, MAX_LEASE_BYTES),
        (14, MAX_RESULT_BYTES),
        (15, COMMITMENT_BYTES),
    ] {
        if !optional_blob_shape(&shapes[index].0, shapes[index].1, maximum) {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    if state < PENDING
        || state > TOMBSTONE
        || route_deadline < 0
        || execution_deadline.is_some_and(|value| value < 0)
        || completed_at.is_some_and(|value| value < 0)
        || retained_until < 0
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    connection.query_row(&format!("SELECT queue_key, route_id, request_commitment, source_node_id, route_body_commitment, immediate_recipient, envelope_commitment, envelope, state, claim_id, claim_commitment, claim_frame, lease_id, lease_commitment, lease_frame, execution_deadline, route_deadline, result_frame, result_commitment, completed_at, retained_until FROM {TABLE} WHERE queue_key = ?1"), params![key.as_slice()], |row| {
        Ok(StoredRow {
            queue_key: row.get(0)?, route_id: row.get(1)?, request_commitment: row.get(2)?, source_node_id: row.get(3)?, route_body_commitment: row.get(4)?, immediate_recipient: row.get(5)?, envelope_commitment: row.get(6)?, envelope: row.get(7)?, state: row.get(8)?, claim_id: row.get(9)?, claim_commitment: row.get(10)?, claim_frame: row.get(11)?, lease_id: row.get(12)?, lease_commitment: row.get(13)?, lease_frame: row.get(14)?, execution_deadline: row.get(15)?, route_deadline: row.get(16)?, result_frame: row.get(17)?, result_commitment: row.get(18)?, completed_at: row.get(19)?, retained_until: row.get(20)?,
        })
    }).optional().map_err(Into::into)
}

fn optional_blob_shape(class: &str, length: Option<i64>, maximum: usize) -> bool {
    match (class, length) {
        ("null", None) => true,
        ("blob", Some(length)) => {
            length >= 0
                && usize::try_from(length)
                    .map_or(false, |value| value <= maximum)
        }
        _ => false,
    }
}

fn load_by_claim_id(connection: &Connection, claim_id: &[u8; ID_BYTES]) -> Result<Option<StoredRow>, ReverseOnionQueueError> {
    let key_shape: Option<(String, Option<i64>)> = connection.query_row(&format!("SELECT typeof(queue_key), length(queue_key) FROM {TABLE} WHERE claim_id = ?1"), params![claim_id.as_slice()], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
    if let Some((class, length)) = key_shape {
        if class != "blob" || length != Some(COMMITMENT_BYTES as i64) { return Err(ReverseOnionQueueError::Corrupt); }
    }
    let key: Option<Vec<u8>> = connection.query_row(&format!("SELECT queue_key FROM {TABLE} WHERE claim_id = ?1"), params![claim_id.as_slice()], |row| row.get(0)).optional()?;
    let Some(key) = key else { return Ok(None); };
    let key: [u8; COMMITMENT_BYTES] = key.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?;
    load_row(connection, &key)
}

fn load_by_source_route(
    connection: &Connection,
    source_node_id: &[u8; COMMITMENT_BYTES],
    route_id: &[u8; ID_BYTES],
    request_commitment: &[u8; COMMITMENT_BYTES],
) -> Result<Option<StoredRow>, ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!(
        "SELECT typeof(queue_key), length(queue_key),
                CASE WHEN typeof(queue_key) = 'blob' AND length(queue_key) = 32
                     THEN queue_key ELSE NULL END
         FROM {TABLE} INDEXED BY {SOURCE_ROUTE_INDEX}
         WHERE source_node_id = ?1 AND route_id = ?2 AND request_commitment = ?3
         ORDER BY queue_key ASC LIMIT 2"
    ))?;
    let keys = statement
        .query_map(
            params![source_node_id.as_slice(), route_id.as_slice(), request_commitment.as_slice()],
            |row| {
                let class: String = row.get(0)?;
                let length: Option<i64> = row.get(1)?;
                if class != "blob" || length != Some(COMMITMENT_BYTES as i64) {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                row.get::<_, Vec<u8>>(2)
            },
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    if keys.len() > 1 {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    let Some(key) = keys.into_iter().next() else {
        return Ok(None);
    };
    let key: [u8; COMMITMENT_BYTES] = key.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?;
    load_row(connection, &key)
}

fn select_pending(connection: &Connection, recipient: &[u8; COMMITMENT_BYTES]) -> Result<Option<StoredRow>, ReverseOnionQueueError> {
    let key_shape: Option<(String, Option<i64>)> = connection.query_row(&format!("SELECT typeof(queue_key), length(queue_key) FROM {TABLE} WHERE immediate_recipient = ?1 AND state = ?2 ORDER BY route_deadline ASC, queue_key ASC LIMIT 1"), params![recipient.as_slice(), PENDING], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
    if let Some((class, length)) = key_shape {
        if class != "blob" || length != Some(COMMITMENT_BYTES as i64) { return Err(ReverseOnionQueueError::Corrupt); }
    }
    let key: Option<Vec<u8>> = connection.query_row(&format!("SELECT queue_key FROM {TABLE} WHERE immediate_recipient = ?1 AND state = ?2 ORDER BY route_deadline ASC, queue_key ASC LIMIT 1"), params![recipient.as_slice(), PENDING], |row| row.get(0)).optional()?;
    let Some(key) = key else { return Ok(None); };
    let key: [u8; COMMITMENT_BYTES] = key.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?;
    load_row(connection, &key)
}

fn stored_item_from_row(row: &StoredRow) -> Result<ReverseOnionQueueStoredItem, ReverseOnionQueueError> {
    Ok(ReverseOnionQueueStoredItem {
        queue_key: row.queue_key.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, route_id: row.route_id.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, request_commitment: row.request_commitment.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, source_node_id: row.source_node_id.as_deref().map(|bytes| <[u8; COMMITMENT_BYTES]>::try_from(bytes)).transpose().map_err(|_| ReverseOnionQueueError::Corrupt)?, route_body_commitment: row.route_body_commitment.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, immediate_recipient: row.immediate_recipient.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, envelope_commitment: row.envelope_commitment.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, envelope: row.envelope.clone(), route_deadline: u64::try_from(row.route_deadline).map_err(|_| ReverseOnionQueueError::Corrupt)?,
    })
}

fn issued_lease_from_row(row: &StoredRow) -> Result<ReverseOnionQueueIssuedLease, ReverseOnionQueueError> {
    Ok(ReverseOnionQueueIssuedLease {
        queue_key: row.queue_key.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, envelope_commitment: row.envelope_commitment.as_slice().try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, lease_id: row.lease_id.as_deref().ok_or(ReverseOnionQueueError::Corrupt)?.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, lease_commitment: row.lease_commitment.as_deref().ok_or(ReverseOnionQueueError::Corrupt)?.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?, frame: row.lease_frame.clone().ok_or(ReverseOnionQueueError::Corrupt)?,
    })
}

fn result_from_row(row: &StoredRow) -> Result<ReverseOnionQueueResult, ReverseOnionQueueError> {
    Ok(ReverseOnionQueueResult { frame: row.result_frame.clone().ok_or(ReverseOnionQueueError::Corrupt)?, commitment: row.result_commitment.as_deref().ok_or(ReverseOnionQueueError::Corrupt)?.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)? })
}

fn ensure_item_context(row: &StoredRow, item: &ReverseOnionQueueItem) -> Result<(), ReverseOnionQueueError> {
    ensure_item_identity(row, item)?;
    if row.envelope.as_slice() != item.envelope.as_slice() {
        return Err(ReverseOnionQueueError::Conflict);
    }
    Ok(())
}

fn ensure_item_identity(row: &StoredRow, item: &ReverseOnionQueueItem) -> Result<(), ReverseOnionQueueError> {
    if row.source_node_id.as_deref() != Some(item.source_node_id.as_slice())
        || row.route_id.as_slice() != item.route_id.as_slice()
        || row.request_commitment.as_slice() != item.request_commitment.as_slice()
        || row.route_body_commitment.as_slice() != item.route_body_commitment.as_slice()
        || row.immediate_recipient.as_slice() != item.immediate_recipient.as_slice()
        || row.envelope_commitment.as_slice() != item.envelope_commitment.as_slice()
        || row.route_deadline != i64::try_from(item.route_deadline).unwrap_or(i64::MIN)
    {
        return Err(ReverseOnionQueueError::Conflict);
    }
    Ok(())
}

fn validate_row(row: &StoredRow) -> Result<(), ReverseOnionQueueError> {
    if let Some(source_node_id) = row.source_node_id.as_deref() {
        if !valid_source_node_slice(source_node_id) {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    if row.queue_key.len() != COMMITMENT_BYTES || row.route_id.len() != ID_BYTES || row.request_commitment.len() != COMMITMENT_BYTES || row.route_body_commitment.len() != COMMITMENT_BYTES || row.immediate_recipient.len() != COMMITMENT_BYTES || row.envelope_commitment.len() != COMMITMENT_BYTES || row.envelope.len() > MAX_REVERSE_ONION_QUEUE_ITEM_BYTES || row.route_deadline < 0 || row.retained_until < row.route_deadline || !matches!(row.state, PENDING | LEASED | ARMED | RESULT | TOMBSTONE) || is_zero(&row.queue_key) || is_zero(&row.route_id) || is_zero(&row.request_commitment) || is_zero(&row.route_body_commitment) || is_zero(&row.immediate_recipient) || is_zero(&row.envelope_commitment) { return Err(ReverseOnionQueueError::Corrupt); }
    match row.state {
        PENDING => {
            if row.envelope.is_empty() || row.claim_id.is_some() || row.claim_commitment.is_some() || row.claim_frame.is_some() || row.lease_id.is_some() || row.lease_commitment.is_some() || row.lease_frame.is_some() || row.execution_deadline.is_some() || row.result_frame.is_some() || row.result_commitment.is_some() || row.completed_at.is_some() { return Err(ReverseOnionQueueError::Corrupt); }
        }
        LEASED | ARMED => {
            if row.envelope.is_empty() || row.claim_id.as_ref().is_none_or(|v| v.len() != ID_BYTES || is_zero(v)) || row.claim_commitment.as_ref().is_none_or(|v| v.len() != COMMITMENT_BYTES || is_zero(v)) || row.claim_frame.as_ref().is_none_or(|v| v.is_empty() || v.len() > MAX_CLAIM_BYTES) || row.lease_id.as_ref().is_none_or(|v| v.len() != ID_BYTES || is_zero(v)) || row.lease_commitment.as_ref().is_none_or(|v| v.len() != COMMITMENT_BYTES || is_zero(v)) || row.lease_frame.as_ref().is_none_or(|v| v.is_empty() || v.len() > MAX_LEASE_BYTES) || row.execution_deadline.is_none_or(|v| v <= 0 || v > row.route_deadline) || row.result_frame.is_some() || row.result_commitment.is_some() || row.completed_at.is_some() { return Err(ReverseOnionQueueError::Corrupt); }
        }
        RESULT => {
            if row.envelope.is_empty()
                || row.claim_id.as_ref().is_none_or(|v| v.len() != ID_BYTES || is_zero(v))
                || row.claim_commitment.as_ref().is_none_or(|v| v.len() != COMMITMENT_BYTES || is_zero(v))
                || row.claim_frame.as_ref().is_none_or(|v| v.is_empty() || v.len() > MAX_CLAIM_BYTES)
                || row.lease_id.as_ref().is_none_or(|v| v.len() != ID_BYTES || is_zero(v))
                || row.lease_commitment.as_ref().is_none_or(|v| v.len() != COMMITMENT_BYTES || is_zero(v))
                || row.lease_frame.as_ref().is_none_or(|v| v.is_empty() || v.len() > MAX_LEASE_BYTES)
                || row.execution_deadline.is_none_or(|v| v <= 0 || v > row.route_deadline)
                || row.result_frame.as_ref().is_none_or(|v| v.is_empty() || v.len() > MAX_RESULT_BYTES)
                || row.result_commitment.as_ref().is_none_or(|v| v.len() != COMMITMENT_BYTES || is_zero(v))
                || row.completed_at.is_none_or(|v| v <= 0)
            {
                return Err(ReverseOnionQueueError::Corrupt);
            }
        }
        TOMBSTONE => {
            if !row.envelope.is_empty() || row.claim_frame.is_some() || row.lease_frame.is_some() || row.result_frame.is_some() || row.result_commitment.is_some() || row.completed_at.is_some() { return Err(ReverseOnionQueueError::Corrupt); }
        }
        _ => return Err(ReverseOnionQueueError::Corrupt),
    }
    Ok(())
}

fn validate_all_rows(
    connection: &Connection,
    limits: &ReverseOnionQueueLimits,
) -> Result<(), ReverseOnionQueueError> {
    let row_limit = sqlite_integer(
        limits
            .max_items
            .checked_add(1)
            .ok_or(ReverseOnionQueueError::Rejected)?,
    )?;
    let keys = {
        let mut statement = connection.prepare(&format!(
            "SELECT typeof(queue_key), length(queue_key), queue_key
             FROM {TABLE} LIMIT ?1"
        ))?;
        let rows = statement.query_map(params![row_limit], |row| {
            let class = row.get::<_, String>(0)?;
            let length = row.get::<_, Option<i64>>(1)?;
            if class != "blob" || length != Some(COMMITMENT_BYTES as i64) {
                return Err(rusqlite::Error::InvalidQuery);
            }
            row.get::<_, Vec<u8>>(2)
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    if keys.len() as u64 > limits.max_items {
        return Err(ReverseOnionQueueError::Capacity);
    }
    for key in keys { let key: [u8; COMMITMENT_BYTES] = key.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?; validate_row(&load_row(connection, &key)?.ok_or(ReverseOnionQueueError::Corrupt)?)?; }
    Ok(())
}

fn validate_all_no_work_rows(
    connection: &Connection,
    limits: &ReverseOnionQueueLimits,
) -> Result<(), ReverseOnionQueueError> {
    let row_limit = sqlite_integer(
        limits
            .max_items
            .checked_add(1)
            .ok_or(ReverseOnionQueueError::Rejected)?,
    )?;
    let mut statement = connection.prepare(&format!(
        "SELECT typeof(claim_id), length(claim_id), claim_id
         FROM {NO_WORK_TABLE} LIMIT ?1"
    ))?;
    let rows = statement.query_map(params![row_limit], |row| {
        let class = row.get::<_, String>(0)?;
        let length = row.get::<_, Option<i64>>(1)?;
        if class != "blob" || length != Some(ID_BYTES as i64) {
            return Err(rusqlite::Error::InvalidQuery);
        }
        row.get::<_, Vec<u8>>(2)
    })?;
    let keys = rows.collect::<Result<Vec<_>, _>>()?;
    if keys.len() as u64 > limits.max_items {
        return Err(ReverseOnionQueueError::Capacity);
    }
    for key in keys {
        let key: [u8; ID_BYTES] = key.try_into().map_err(|_| ReverseOnionQueueError::Corrupt)?;
        let row = load_no_work(connection, &key)?.ok_or(ReverseOnionQueueError::Corrupt)?;
        if row.retained_until < row.recorded_at {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    Ok(())
}

fn cleanup_expired(connection: &Connection, now: i64) -> Result<u64, ReverseOnionQueueError> {
    let mut removed = u64::try_from(connection.execute(&format!("DELETE FROM {TABLE} WHERE state = ?1 AND route_deadline <= ?2"), params![PENDING, now])?).map_err(|_| ReverseOnionQueueError::Corrupt)?;
    removed = removed.saturating_add(u64::try_from(connection.execute(&format!("DELETE FROM {TABLE} WHERE state IN (?1, ?2) AND retained_until <= ?3"), params![RESULT, TOMBSTONE, now])?).map_err(|_| ReverseOnionQueueError::Corrupt)?);
    removed = removed.saturating_add(u64::try_from(connection.execute(&format!("DELETE FROM {NO_WORK_TABLE} WHERE retained_until <= ?1"), params![now])?).map_err(|_| ReverseOnionQueueError::Corrupt)?);
    connection.execute(&format!("UPDATE {TABLE} SET state = ?1, envelope = zeroblob(0), claim_frame = NULL, lease_frame = NULL WHERE state = ?2 AND retained_until <= ?3"), params![TOMBSTONE, ARMED, now])?;
    Ok(removed)
}

fn enforce_quotas(connection: &Connection, limits: &ReverseOnionQueueLimits, recipient: &[u8; COMMITMENT_BYTES], incoming: u64) -> Result<(), ReverseOnionQueueError> {
    let (main_count, main_bytes): (i64, i64) = connection.query_row(&format!("SELECT COUNT(*), COALESCE(SUM(LENGTH(envelope) + COALESCE(LENGTH(claim_frame), 0) + COALESCE(LENGTH(lease_frame), 0) + COALESCE(LENGTH(result_frame), 0)), 0) FROM {TABLE}"), [], |row| Ok((row.get(0)?, row.get(1)?)))?;
    let (marker_count, marker_bytes): (i64, i64) = connection.query_row(&format!("SELECT COUNT(*), COALESCE(SUM(LENGTH(claim_frame) + 80), 0) FROM {NO_WORK_TABLE}"), [], |row| Ok((row.get(0)?, row.get(1)?)))?;
    let count = main_count.checked_add(marker_count).ok_or(ReverseOnionQueueError::Corrupt)?;
    let bytes = main_bytes.checked_add(marker_bytes).ok_or(ReverseOnionQueueError::Corrupt)?;
    let count = u64::try_from(count).map_err(|_| ReverseOnionQueueError::Corrupt)?; let bytes = u64::try_from(bytes).map_err(|_| ReverseOnionQueueError::Corrupt)?;
    if count >= limits.max_items || bytes.checked_add(incoming).is_none_or(|v| v > limits.max_bytes) { return Err(ReverseOnionQueueError::Capacity); }
    let main_recipient_count: i64 = connection.query_row(&format!("SELECT COUNT(*) FROM {TABLE} WHERE immediate_recipient = ?1"), params![recipient.as_slice()], |row| row.get(0))?;
    let marker_recipient_count: i64 = connection.query_row(&format!("SELECT COUNT(*) FROM {NO_WORK_TABLE} WHERE immediate_recipient = ?1"), params![recipient.as_slice()], |row| row.get(0))?;
    let recipient_count = main_recipient_count.checked_add(marker_recipient_count).ok_or(ReverseOnionQueueError::Corrupt)?;
    if u64::try_from(recipient_count).map_err(|_| ReverseOnionQueueError::Corrupt)? >= limits.max_items_per_recipient { return Err(ReverseOnionQueueError::Capacity); }
    Ok(())
}

fn admission_for_state(state: i64) -> ReverseOnionQueueAdmission { match state { PENDING => ReverseOnionQueueAdmission::Pending, LEASED | ARMED => ReverseOnionQueueAdmission::Armed, RESULT => ReverseOnionQueueAdmission::Result, TOMBSTONE => ReverseOnionQueueAdmission::Tombstone, _ => ReverseOnionQueueAdmission::Tombstone } }

fn has_application_objects(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
            )",
            [],
            |row| row.get(0),
        )
        .map_err(Into::into)
}

fn application_objects(connection: &Connection) -> Result<Vec<String>, ReverseOnionQueueError> {
    let mut statement = connection.prepare(
        "SELECT CASE
                    WHEN length(CAST(name AS BLOB)) <= ?1 THEN name
                    ELSE NULL
                END
         FROM sqlite_master
         WHERE name NOT LIKE 'sqlite_%'
         LIMIT ?2",
    )?;
    let object_limit = i64::try_from(MAX_APPLICATION_OBJECTS + 1)
        .map_err(|_| ReverseOnionQueueError::Rejected)?;
    let rows = statement.query_map(
        params![MAX_APPLICATION_OBJECT_NAME_BYTES, object_limit],
        |row| row.get::<_, Option<String>>(0),
    )?;
    let mut objects = Vec::new();
    for row in rows {
        let object = row?.ok_or(ReverseOnionQueueError::Corrupt)?;
        if object.len() > usize::try_from(MAX_APPLICATION_OBJECT_NAME_BYTES).unwrap_or(0) {
            return Err(ReverseOnionQueueError::Corrupt);
        }
        objects.push(object);
    }
    if objects.len() > MAX_APPLICATION_OBJECTS {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(objects)
}

fn named_object_type(
    connection: &Connection,
    name: &str,
) -> Result<Option<String>, ReverseOnionQueueError> {
    connection
        .query_row(
            "SELECT type FROM sqlite_master WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )
        .optional()
        .map_err(Into::into)
}

fn validate_owned_objects(objects: &[String]) -> Result<(), ReverseOnionQueueError> {
    let allowed = [
        META_TABLE,
        TABLE,
        NO_WORK_TABLE,
        "idx_reverse_onion_delivery_queue_v1_claim_id",
        "idx_reverse_onion_delivery_queue_v1_recipient",
        "idx_reverse_onion_delivery_queue_v1_retention",
        "idx_reverse_onion_delivery_queue_v1_no_work_retention",
        SOURCE_ROUTE_INDEX,
    ];
    if objects
        .iter()
        .any(|object| !allowed.iter().any(|allowed| object.as_str() == *allowed))
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(())
}

fn create_meta_schema(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE TABLE {META_TABLE} (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            ownership_tag BLOB NOT NULL,
            clock_high_water INTEGER NOT NULL
        );
        INSERT INTO {META_TABLE}(id, schema_version, ownership_tag, clock_high_water)
            VALUES (1, {SCHEMA_VERSION}, X'4165726F4E79782D526576657273654F6E696F6E51756575652D7631', 0);"
    ))?;
    Ok(())
}

fn create_main_schema(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE TABLE {TABLE} (
            queue_key BLOB PRIMARY KEY,
            route_id BLOB NOT NULL,
            request_commitment BLOB NOT NULL,
            route_body_commitment BLOB NOT NULL,
            immediate_recipient BLOB NOT NULL,
            envelope_commitment BLOB NOT NULL,
            envelope BLOB NOT NULL,
            state INTEGER NOT NULL,
            claim_id BLOB,
            claim_commitment BLOB,
            claim_frame BLOB,
            lease_id BLOB,
            lease_commitment BLOB,
            lease_frame BLOB,
            execution_deadline INTEGER,
            route_deadline INTEGER NOT NULL,
            result_frame BLOB,
            result_commitment BLOB,
            completed_at INTEGER,
            retained_until INTEGER NOT NULL,
            source_node_id BLOB
        );"
    ))?;
    create_main_indexes(connection)
}

fn create_no_work_schema(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE TABLE {NO_WORK_TABLE} (
            claim_id BLOB PRIMARY KEY,
            immediate_recipient BLOB NOT NULL,
            claim_commitment BLOB NOT NULL,
            claim_frame BLOB NOT NULL,
            recorded_at INTEGER NOT NULL,
            retained_until INTEGER NOT NULL
        );"
    ))?;
    create_no_work_indexes(connection)
}

fn validate_meta_schema(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    validate_meta_schema_version(connection, SCHEMA_VERSION)
}

fn validate_meta_schema_version(
    connection: &Connection,
    expected_version: i64,
) -> Result<(), ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({META_TABLE})"))?;
    let columns: Vec<(String, String, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(5)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let expected = [
        ("id", "INTEGER", 0, 1),
        ("schema_version", "INTEGER", 1, 0),
        ("ownership_tag", "BLOB", 1, 0),
        ("clock_high_water", "INTEGER", 1, 0),
    ];
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(|(actual, expected)| {
            actual.0 != expected.0
                || actual.1.to_ascii_uppercase() != expected.1
                || actual.2 != expected.2
                || actual.3 != expected.3
        })
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    let marker: Option<(i64, Vec<u8>, i64)> = connection
        .query_row(
            &format!("SELECT schema_version, ownership_tag, clock_high_water FROM {META_TABLE} WHERE id = 1"),
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    let Some((version, tag, clock_high_water)) = marker else {
        return Err(ReverseOnionQueueError::Corrupt);
    };
    let row_count: i64 = connection
        .query_row(
            &format!("SELECT COUNT(*) FROM {META_TABLE}"),
            [],
            |row| row.get(0),
        )
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    if row_count != 1
        || version != expected_version
        || tag.as_slice() != OWNERSHIP_TAG
        || clock_high_water < 0
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(())
}

fn metadata_schema_version(connection: &Connection) -> Result<i64, ReverseOnionQueueError> {
    connection
        .query_row(
            &format!("SELECT schema_version FROM {META_TABLE} WHERE id = 1"),
            [],
            |row| row.get(0),
        )
        .map_err(|_| ReverseOnionQueueError::Corrupt)
}

fn validate_meta_schema_v1(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({META_TABLE})"))?;
    let columns: Vec<(String, String, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(5)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let expected = [
        ("id", "INTEGER", 0, 1),
        ("schema_version", "INTEGER", 1, 0),
        ("ownership_tag", "BLOB", 1, 0),
    ];
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(|(actual, expected)| {
            actual.0 != expected.0
                || actual.1.to_ascii_uppercase() != expected.1
                || actual.2 != expected.2
                || actual.3 != expected.3
        })
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    let marker: Option<(i64, Vec<u8>)> = connection
        .query_row(
            &format!("SELECT schema_version, ownership_tag FROM {META_TABLE} WHERE id = 1"),
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    let Some((version, tag)) = marker else {
        return Err(ReverseOnionQueueError::Corrupt);
    };
    let row_count: i64 = connection
        .query_row(&format!("SELECT COUNT(*) FROM {META_TABLE}"), [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    if row_count != 1 || version != LEGACY_SCHEMA_VERSION || tag.as_slice() != OWNERSHIP_TAG {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(())
}

fn migrate_meta_v1_to_v2(
    connection: &Connection,
    trusted_now: u64,
    max_items: u64,
) -> Result<(), ReverseOnionQueueError> {
    let observed = observed_clock_high_water(connection, max_items)?;
    let trusted_now = sqlite_integer(trusted_now)?;
    if observed > trusted_now {
        return Err(ReverseOnionQueueError::Rejected);
    }
    connection.execute(
        &format!("ALTER TABLE {META_TABLE} ADD COLUMN clock_high_water INTEGER NOT NULL DEFAULT 0"),
        [],
    )?;
    connection.execute(
        &format!("UPDATE {META_TABLE} SET schema_version = ?1, clock_high_water = ?2 WHERE id = 1"),
        params![CLOCK_SCHEMA_VERSION, trusted_now],
    )?;
    Ok(())
}

fn migrate_source_binding_v2_to_v3(
    connection: &Connection,
) -> Result<(), ReverseOnionQueueError> {
    connection.execute(
        &format!("ALTER TABLE {TABLE} ADD COLUMN source_node_id BLOB"),
        [],
    )?;
    connection.execute(
        &format!("UPDATE {META_TABLE} SET schema_version = ?1 WHERE id = 1"),
        params![SOURCE_BINDING_SCHEMA_VERSION],
    )?;
    Ok(())
}

// [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Metadata and index change
// in the caller's immediate transaction; no row or NULL source is rewritten.
fn create_source_route_index(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE INDEX {SOURCE_ROUTE_INDEX} ON {TABLE}
         (source_node_id, route_id, request_commitment, queue_key);"
    ))?;
    Ok(())
}

fn migrate_source_index_v3_to_v4(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    validate_meta_schema_version(connection, SOURCE_BINDING_SCHEMA_VERSION)?;
    create_source_route_index(connection)?;
    if connection.execute(
        &format!("UPDATE {META_TABLE} SET schema_version = ?1 WHERE id = 1 AND schema_version = ?2"),
        params![SCHEMA_VERSION, SOURCE_BINDING_SCHEMA_VERSION],
    )? != 1 {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(())
}

fn observed_clock_high_water(
    connection: &Connection,
    max_items: u64,
) -> Result<i64, ReverseOnionQueueError> {
    let mut high_water = 0i64;
    for (table, column) in [(TABLE, "completed_at"), (NO_WORK_TABLE, "recorded_at")] {
        let count: i64 = connection.query_row(
            &format!("SELECT COUNT(*) FROM {table}"),
            [],
            |row| row.get(0),
        )?;
        if count < 0 || u64::try_from(count).map_err(|_| ReverseOnionQueueError::Corrupt)? > max_items {
            return Err(ReverseOnionQueueError::Capacity);
        }
        let invalid: i64 = connection.query_row(
            &format!(
                "SELECT COUNT(*) FROM {table}
                 WHERE {column} IS NOT NULL
                   AND (typeof({column}) <> 'integer' OR {column} < 0)"
            ),
            [],
            |row| row.get(0),
        )?;
        if invalid != 0 {
            return Err(ReverseOnionQueueError::Corrupt);
        }
        let observed: Option<i64> = connection.query_row(
            &format!("SELECT MAX({column}) FROM {table}"),
            [],
            |row| row.get(0),
        )?;
        if let Some(observed) = observed {
            if observed < 0 {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            high_water = high_water.max(observed);
        }
    }
    if max_items == 0 {
        return Err(ReverseOnionQueueError::Rejected);
    }
    Ok(high_water)
}

fn validate_schema(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    let expected = [
        ("queue_key", "BLOB", 0, 1),
        ("route_id", "BLOB", 1, 0),
        ("request_commitment", "BLOB", 1, 0),
        ("route_body_commitment", "BLOB", 1, 0),
        ("immediate_recipient", "BLOB", 1, 0),
        ("envelope_commitment", "BLOB", 1, 0),
        ("envelope", "BLOB", 1, 0),
        ("state", "INTEGER", 1, 0),
        ("claim_id", "BLOB", 0, 0),
        ("claim_commitment", "BLOB", 0, 0),
        ("claim_frame", "BLOB", 0, 0),
        ("lease_id", "BLOB", 0, 0),
        ("lease_commitment", "BLOB", 0, 0),
        ("lease_frame", "BLOB", 0, 0),
        ("execution_deadline", "INTEGER", 0, 0),
        ("route_deadline", "INTEGER", 1, 0),
        ("result_frame", "BLOB", 0, 0),
        ("result_commitment", "BLOB", 0, 0),
        ("completed_at", "INTEGER", 0, 0),
        ("retained_until", "INTEGER", 1, 0),
        ("source_node_id", "BLOB", 0, 0),
    ];
    let mut statement = connection.prepare(&format!("PRAGMA table_info({TABLE})"))?;
    let columns: Vec<(String, String, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(5)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(|(actual, expected)| {
            actual.0 != expected.0
                || actual.1.to_ascii_uppercase() != expected.1
                || actual.2 != expected.2
                || actual.3 != expected.3
        })
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    validate_main_indexes(connection)
}

fn validate_schema_legacy(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    let expected = [
        ("queue_key", "BLOB", 0, 1),
        ("route_id", "BLOB", 1, 0),
        ("request_commitment", "BLOB", 1, 0),
        ("route_body_commitment", "BLOB", 1, 0),
        ("immediate_recipient", "BLOB", 1, 0),
        ("envelope_commitment", "BLOB", 1, 0),
        ("envelope", "BLOB", 1, 0),
        ("state", "INTEGER", 1, 0),
        ("claim_id", "BLOB", 0, 0),
        ("claim_commitment", "BLOB", 0, 0),
        ("claim_frame", "BLOB", 0, 0),
        ("lease_id", "BLOB", 0, 0),
        ("lease_commitment", "BLOB", 0, 0),
        ("lease_frame", "BLOB", 0, 0),
        ("execution_deadline", "INTEGER", 0, 0),
        ("route_deadline", "INTEGER", 1, 0),
        ("result_frame", "BLOB", 0, 0),
        ("result_commitment", "BLOB", 0, 0),
        ("completed_at", "INTEGER", 0, 0),
        ("retained_until", "INTEGER", 1, 0),
    ];
    let mut statement = connection.prepare(&format!("PRAGMA table_info({TABLE})"))?;
    let columns: Vec<(String, String, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(5)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(|(actual, expected)| {
            actual.0 != expected.0
                || actual.1.to_ascii_uppercase() != expected.1
                || actual.2 != expected.2
                || actual.3 != expected.3
        })
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    validate_main_indexes(connection)
}

fn validate_no_work_schema(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    let expected = [
        ("claim_id", "BLOB", 0, 1),
        ("immediate_recipient", "BLOB", 1, 0),
        ("claim_commitment", "BLOB", 1, 0),
        ("claim_frame", "BLOB", 1, 0),
        ("recorded_at", "INTEGER", 1, 0),
        ("retained_until", "INTEGER", 1, 0),
    ];
    let mut statement = connection.prepare(&format!("PRAGMA table_info({NO_WORK_TABLE})"))?;
    let columns: Vec<(String, String, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(5)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(|(actual, expected)| {
            actual.0 != expected.0
                || actual.1.to_ascii_uppercase() != expected.1
                || actual.2 != expected.2
                || actual.3 != expected.3
        })
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    validate_no_work_indexes(connection)
}

fn create_main_indexes(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_claim_id
            ON {TABLE}(claim_id) WHERE claim_id IS NOT NULL;
         CREATE INDEX IF NOT EXISTS idx_{TABLE}_recipient
            ON {TABLE}(immediate_recipient, state, route_deadline);
         CREATE INDEX IF NOT EXISTS idx_{TABLE}_retention
            ON {TABLE}(state, route_deadline, retained_until);"
    ))?;
    Ok(())
}

fn create_no_work_indexes(connection: &Connection) -> Result<(), ReverseOnionQueueError> {
    connection.execute_batch(&format!(
        "CREATE INDEX IF NOT EXISTS idx_{NO_WORK_TABLE}_retention
            ON {NO_WORK_TABLE}(retained_until);"
    ))?;
    Ok(())
}

fn validate_main_indexes(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!("PRAGMA index_list({TABLE})"))?;
    let indexes: Vec<(String, i64, String, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let auto = format!("sqlite_autoindex_{TABLE}_1");
    let mut expected = vec![
        (format!("idx_{TABLE}_claim_id"), true, true, vec!["claim_id".to_owned()]),
        (format!("idx_{TABLE}_recipient"), false, false, vec!["immediate_recipient".to_owned(), "state".to_owned(), "route_deadline".to_owned()]),
        (format!("idx_{TABLE}_retention"), false, false, vec!["state".to_owned(), "route_deadline".to_owned(), "retained_until".to_owned()]),
    ];
    if metadata_schema_version(connection)? == SCHEMA_VERSION {
        expected.push((SOURCE_ROUTE_INDEX.to_owned(), false, false,
            vec!["source_node_id".to_owned(), "route_id".to_owned(),
                 "request_commitment".to_owned(), "queue_key".to_owned()]));
    }
    let mut complete = true;
    for (name, unique, _origin, partial) in &indexes {
        if name == &auto {
            if *unique != 1 || *partial != 0 {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            continue;
        }
        let Some((_, expected_unique, expected_partial, columns)) = expected
            .iter()
            .find(|item| item.0.as_str() == name.as_str())
        else {
            return Err(ReverseOnionQueueError::Corrupt);
        };
        if (*unique == 1) != *expected_unique
            || (*partial == 1) != *expected_partial
            || index_columns(connection, name)? != columns.to_vec()
        {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    for (name, _, _, _) in &expected {
        if !indexes.iter().any(|item| item.0.as_str() == name.as_str()) {
            complete = false;
        }
    }
    Ok(complete)
}

fn validate_no_work_indexes(connection: &Connection) -> Result<bool, ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!("PRAGMA index_list({NO_WORK_TABLE})"))?;
    let indexes: Vec<(String, i64, String, i64)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let auto = format!("sqlite_autoindex_{NO_WORK_TABLE}_1");
    let retention = format!("idx_{NO_WORK_TABLE}_retention");
    let mut complete = false;
    for (name, unique, _origin, partial) in indexes {
        if name.as_str() == auto.as_str() {
            if unique != 1 || partial != 0 {
                return Err(ReverseOnionQueueError::Corrupt);
            }
        } else if name.as_str() == retention.as_str() {
            if unique != 0
                || partial != 0
                || index_columns(connection, name)? != vec!["retained_until".to_owned()]
            {
                return Err(ReverseOnionQueueError::Corrupt);
            }
            complete = true;
        } else {
            return Err(ReverseOnionQueueError::Corrupt);
        }
    }
    Ok(complete)
}

fn index_columns(connection: &Connection, index: &str) -> Result<Vec<String>, ReverseOnionQueueError> {
    let mut statement = connection.prepare(&format!("PRAGMA index_info({index})"))?;
    Ok(statement
        .query_map([], |row| row.get::<_, String>(2))?
        .collect::<Result<Vec<_>, _>>()?)
}

fn sqlite_integer(value: u64) -> Result<i64, ReverseOnionQueueError> { i64::try_from(value).map_err(|_| ReverseOnionQueueError::Rejected) }

fn advance_clock(connection: &Connection, now: i64) -> Result<(), ReverseOnionQueueError> {
    if now <= 0 {
        return Err(ReverseOnionQueueError::Rejected);
    }
    let clock: Option<i64> = connection
        .query_row(
            &format!(
                "SELECT CASE WHEN typeof(clock_high_water) = 'integer'
                              AND clock_high_water >= 0
                            THEN clock_high_water ELSE NULL END
                   FROM {META_TABLE} WHERE id = 1"
            ),
            [],
            |row| row.get(0),
        )
        .map_err(|_| ReverseOnionQueueError::Corrupt)?;
    let clock = clock.ok_or(ReverseOnionQueueError::Corrupt)?;
    if now < clock {
        return Err(ReverseOnionQueueError::Rejected);
    }
    if now > clock
        && connection.execute(
            &format!(
                "UPDATE {META_TABLE}
                    SET clock_high_water = ?1
                  WHERE id = 1 AND clock_high_water = ?2"
            ),
            params![now, clock],
        )? != 1
    {
        return Err(ReverseOnionQueueError::Corrupt);
    }
    Ok(())
}

fn validate_now(now: u64) -> Result<(), ReverseOnionQueueError> { if now == 0 { Err(ReverseOnionQueueError::Rejected) } else { Ok(()) } }
fn is_zero(bytes: &[u8]) -> bool { bytes.iter().all(|byte| *byte == 0) }
fn valid_source_node_id(bytes: &[u8; COMMITMENT_BYTES]) -> bool {
    !is_zero(bytes) && IdentityPublicKey::from_bytes(bytes).is_ok()
}
fn valid_source_node_slice(bytes: &[u8]) -> bool {
    let Ok(bytes) = <[u8; COMMITMENT_BYTES]>::try_from(bytes) else {
        return false;
    };
    valid_source_node_id(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Authored, unexecuted.
    fn downgrade_fixture_to_v3(connection: &Mutex<Connection>) {
        connection.lock().execute_batch(&format!(
            "DROP INDEX {SOURCE_ROUTE_INDEX}; UPDATE {META_TABLE} SET schema_version = 3;"
        )).unwrap();
    }

    #[test]
    fn source_index_v3_migration_preserves_rows_and_rolls_back_damage() {
        let (queue, connection) = initialized_queue(4, 600, 120);
        let row = item(81, 200);
        queue.enqueue(&connection, &row, 100).unwrap();
        downgrade_fixture_to_v3(&connection);
        queue.initialize(&connection).unwrap();
        assert_eq!(metadata_schema_version(&connection.lock()).unwrap(), 4);
        assert_eq!(application_objects(&connection.lock()).unwrap().len(), 8);
        let snapshot = queue.lookup_source(&connection, source_node_id(), row.route_id,
            row.request_commitment, 101).unwrap().unwrap();
        assert_eq!(snapshot.route_deadline(), 200);
        queue.initialize(&connection).unwrap();
        downgrade_fixture_to_v3(&connection);
        connection.lock().execute(&format!("UPDATE {TABLE} SET envelope = X''"), []).unwrap();
        assert!(matches!(queue.initialize(&connection), Err(ReverseOnionQueueError::Corrupt)));
        assert_eq!(metadata_schema_version(&connection.lock()).unwrap(), 3);
        assert!(named_object_type(&connection.lock(), SOURCE_ROUTE_INDEX).unwrap().is_none());
        let count: i64 = connection.lock().query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1, "migration must not delete damaged data");
    }

    #[test]
    fn source_index_v4_missing_wrong_or_unknown_schema_is_not_repaired() {
        for mode in 0..3 {
            let (queue, connection) = initialized_queue(4, 600, 120);
            match mode {
                0 => connection.lock().execute_batch(&format!("DROP INDEX {SOURCE_ROUTE_INDEX}")),
                1 => connection.lock().execute_batch(&format!(
                    "DROP INDEX {SOURCE_ROUTE_INDEX}; CREATE INDEX {SOURCE_ROUTE_INDEX} ON {TABLE}(route_id);")),
                _ => connection.lock().execute_batch(&format!("UPDATE {META_TABLE} SET schema_version = 99")),
            }.unwrap();
            assert!(matches!(queue.initialize(&connection), Err(ReverseOnionQueueError::Corrupt)));
        }
        let (queue, connection) = initialized_queue(4, 600, 120);
        // A v4 index cannot be smuggled into the v3 seven-object inventory.
        connection.lock().execute(&format!("UPDATE {META_TABLE} SET schema_version = 3"), []).unwrap();
        assert!(matches!(queue.initialize(&connection), Err(ReverseOnionQueueError::Corrupt)));
        assert_eq!(metadata_schema_version(&connection.lock()).unwrap(), 3);
    }

    #[test]
    fn source_index_reads_only_target_projection_and_bounds_key_before_load() {
        let (queue, connection) = initialized_queue(128, 600, 120);
        for seed in 1..65 { queue.enqueue(&connection, &item(seed, 200), 100).unwrap(); }
        let target = item(32, 200);
        let changes_before: i64 = connection.lock().query_row("SELECT total_changes()", [], |r| r.get(0)).unwrap();
        SOURCE_TEST_ROW_LOADS.with(|count| count.set(0));
        assert!(queue.lookup_source(&connection, source_node_id(), [100; 16], [101; 32], 101).unwrap().is_none());
        assert_eq!(SOURCE_TEST_ROW_LOADS.with(Cell::get), 0);
        assert!(queue.lookup_source(&connection, source_node_id(), target.route_id, target.request_commitment, 101).unwrap().is_some());
        assert_eq!(SOURCE_TEST_ROW_LOADS.with(Cell::get), 1);
        let changes_after: i64 = connection.lock().query_row("SELECT total_changes()", [], |r| r.get(0)).unwrap();
        assert_eq!(changes_after, changes_before, "lookup must not advance SQL clock or clean rows");
        // Corrupt local data must not be copied into a Vec before admission.
        connection.lock().execute(&format!("UPDATE {TABLE} SET queue_key = zeroblob(1048576) WHERE queue_key = ?1"),
            params![target.queue_key.as_slice()]).unwrap();
        SOURCE_TEST_ROW_LOADS.with(|count| count.set(0));
        assert!(matches!(queue.lookup_source(&connection, source_node_id(), target.route_id,
            target.request_commitment, 101), Err(ReverseOnionQueueError::Corrupt)));
        assert_eq!(SOURCE_TEST_ROW_LOADS.with(Cell::get), 0);
    }

    #[test]
    fn source_snapshot_real_core_signed_chain_interoperates_with_axre() {
        use aeronyx_core::crypto::IdentityKeyPair;
        use aeronyx_core::protocol::chat::{decode_blind_relay_envelope, encode_blind_relay_envelope};
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
        use aeronyx_core::protocol::onion::reverse_delivery::{ReverseOnionFrameV1,
            ReverseOnionSourceQueryV1, ReverseOnionSourceEvidenceV1,
            SourceEvidencePartV1, VerifiedSourceEvidenceChain};
        use aeronyx_core::protocol::onion_reply::{OnionReplySession, seal_onion_reply,
            encode_onion_sealed_response, ONION_REPLY_RESPONSE_SIZE_CLASSES};
        use sha2::{Digest, Sha256};

        const NOW: u64 = 1_800_000_000;
        let relay = IdentityKeyPair::from_bytes(&[11; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[22; 32]).unwrap();
        let source = IdentityKeyPair::from_bytes(&[33; 32]).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [1; 16],
            NOW, NOW + 30, &recipient).unwrap();
        let (_, kem) = recipient.to_x25519();
        let envelope = build_onion_envelope(&[OnionHop { node_id: recipient.public_key_bytes(),
            kem_pub: kem.to_bytes() }], b"opaque request", [2; 16], 1, NOW, &relay).unwrap();
        let envelope_bytes = encode_blind_relay_envelope(&envelope).unwrap();
        let envelope_hash: [u8; 32] = Sha256::digest(&envelope_bytes).into();
        let row = ReverseOnionQueueItem::new([3; 32], [2; 16], [4; 32],
            source.public_key_bytes(), [5; 32], recipient.public_key_bytes(),
            envelope_hash, envelope_bytes, NOW + 600).unwrap();
        let (queue, connection) = initialized_queue(4, 600, 120);
        queue.enqueue(&connection, &row, NOW).unwrap();
        let verify_claim = |bytes: &[u8]| {
            let frame = ReverseOnionFrameV1::decode(bytes, NOW)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            frame.verify_claim(relay.public_key_bytes(), recipient.public_key_bytes(), NOW)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            Ok(frame.commitment())
        };
        let mut damaged = claim.encode();
        *damaged.last_mut().unwrap() ^= 1;
        assert!(verify_claim(&damaged).is_err());
        assert!(matches!(queue.issue_lease(&connection, recipient.public_key_bytes(),
            claim.claim_id(), claim.commitment(), damaged, NOW, &verify_claim,
            |_, _| panic!("invalid signature must not reach lease constructor")),
            Err(ReverseOnionQueueError::Rejected)));
        let pending = queue.lookup_source(&connection, source.public_key_bytes(), [2; 16],
            [4; 32], NOW).unwrap().unwrap();
        assert!(pending.claim_frame().is_none(), "bad signature must leave durable state unchanged");
        let issued = queue.issue_lease(&connection, recipient.public_key_bytes(),
            claim.claim_id(), claim.commitment(), claim.encode(), NOW, &verify_claim,
            |stored, bytes| {
                let checked_claim = ReverseOnionFrameV1::decode(bytes, NOW)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let checked_envelope = decode_blind_relay_envelope(stored.envelope())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let lease = ReverseOnionFrameV1::lease(&checked_claim, &checked_envelope,
                    [6; 16], stored.route_deadline(), NOW, &relay)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                ReverseOnionQueueLeaseMaterial::new(lease.lease_id(), lease.commitment(),
                    lease.encode(), lease.expires_at(), lease.result_retention_deadline()
                        .map_err(|_| ReverseOnionQueueError::Rejected)?)
            }).unwrap();
        let issued = match issued { ReverseOnionQueueIssue::Issued(value) => value,
            _ => panic!("expected real signed lease") };
        let lease = ReverseOnionFrameV1::decode_for_recovery(issued.frame()).unwrap();
        let (reply_request, _session) = OnionReplySession::prepare_source_sealed([2; 16],
            recipient.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"operation".to_vec()).unwrap();
        let reply = seal_onion_reply([2; 16], &reply_request, b"opaque result", &recipient).unwrap();
        let result = ReverseOnionFrameV1::result(&claim, &lease,
            &encode_onion_sealed_response(&reply).unwrap(), NOW + 600, NOW + 1, &recipient).unwrap();
        queue.complete(&connection, &issued, &result.encode(), NOW + 1, |context, bytes| {
            let claim = ReverseOnionFrameV1::decode_for_recovery(context.claim_frame())
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            let lease = ReverseOnionFrameV1::decode_for_recovery(context.lease_frame())
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            let result = ReverseOnionFrameV1::decode_for_recovery(bytes)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            result.verify_result(&claim, &lease, context.route_deadline(), NOW + 1)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            Ok(result.commitment())
        }).unwrap();
        let snapshot = queue.lookup_source(&connection, source.public_key_bytes(), [2; 16],
            [4; 32], NOW + 2).unwrap().unwrap();
        let stored_claim = ReverseOnionFrameV1::decode_for_recovery(snapshot.claim_frame().unwrap()).unwrap();
        let stored_lease = ReverseOnionFrameV1::decode_for_recovery(snapshot.lease_frame().unwrap()).unwrap();
        let stored_result = ReverseOnionFrameV1::decode_for_recovery(snapshot.result_frame().unwrap()).unwrap();
        assert_eq!(stored_lease.immediate_recipient(), snapshot.immediate_recipient());
        let queries: Vec<_> = [SourceEvidencePartV1::Claim, SourceEvidencePartV1::Lease,
            SourceEvidencePartV1::Result].into_iter().map(|part|
                ReverseOnionSourceQueryV1::sign(&source, relay.public_key_bytes(), snapshot.route_id(),
                    snapshot.request_commitment(), part, [7; 32], NOW + 2, NOW + 20).unwrap()).collect();
        let responses: Vec<_> = queries.iter().map(|query| {
            let authority = query.verify_binding(snapshot.source_node_id(), relay.public_key_bytes(),
                snapshot.route_id(), snapshot.request_commitment(), NOW + 2).unwrap();
            let response = ReverseOnionSourceEvidenceV1::available(&authority, &stored_claim,
                &stored_lease, &stored_result, snapshot.route_deadline(), NOW + 2, NOW + 20, &relay).unwrap();
            ReverseOnionSourceEvidenceV1::decode(&response.encode()).unwrap()
        }).collect();
        let parts: Vec<_> = responses.iter().zip(&queries).map(|(response, query)|
            response.verify_for_query(query, NOW + 2).unwrap()).collect();
        let verified = VerifiedSourceEvidenceChain::verify([&parts[0], &parts[1], &parts[2]],
            snapshot.route_deadline(), NOW + 2).unwrap();
        assert_eq!(verified.result().encode(), result.encode());
    }

    // [REVERSE-ONION-QUEUE-REGRESSIONS 2026-10-04 by Codex] These fixtures
    // below use opaque structural placeholders. The separate source snapshot
    // interoperability fixture above uses actual core-signed frames.
    fn limits(max_items: u64, lease_max_secs: u64, recovery_retention_secs: u64) -> ReverseOnionQueueLimits {
        ReverseOnionQueueLimits::new(
            max_items,
            8 * 1024 * 1024,
            max_items,
            lease_max_secs,
            recovery_retention_secs,
        )
        .unwrap()
    }

    fn connection() -> Mutex<Connection> {
        Mutex::new(Connection::open_in_memory().unwrap())
    }

    fn create_legacy_main_schema(connection: &Connection) {
        connection
            .execute_batch(&format!(
                "CREATE TABLE {TABLE} (
                    queue_key BLOB PRIMARY KEY,
                    route_id BLOB NOT NULL,
                    request_commitment BLOB NOT NULL,
                    route_body_commitment BLOB NOT NULL,
                    immediate_recipient BLOB NOT NULL,
                    envelope_commitment BLOB NOT NULL,
                    envelope BLOB NOT NULL,
                    state INTEGER NOT NULL,
                    claim_id BLOB,
                    claim_commitment BLOB,
                    claim_frame BLOB,
                    lease_id BLOB,
                    lease_commitment BLOB,
                    lease_frame BLOB,
                    execution_deadline INTEGER,
                    route_deadline INTEGER NOT NULL,
                    result_frame BLOB,
                    result_commitment BLOB,
                    completed_at INTEGER,
                    retained_until INTEGER NOT NULL
                );"
            ))
            .unwrap();
        create_main_indexes(connection).unwrap();
    }

    fn initialized_queue(
        max_items: u64,
        lease_max_secs: u64,
        recovery_retention_secs: u64,
    ) -> (SqliteReverseOnionQueue, Mutex<Connection>) {
        let queue = SqliteReverseOnionQueue::new(limits(
            max_items,
            lease_max_secs,
            recovery_retention_secs,
        ));
        let connection = connection();
        queue.initialize(&connection).unwrap();
        (queue, connection)
    }

    fn legacy_connection(observed_at: i64) -> Mutex<Connection> {
        let connection = connection();
        {
            let connection = connection.lock();
            connection
                .execute_batch(&format!(
                    "CREATE TABLE {META_TABLE} (
                        id INTEGER PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        ownership_tag BLOB NOT NULL
                    );
                     INSERT INTO {META_TABLE}(id, schema_version, ownership_tag)
                        VALUES (1, 1, X'4165726F4E79782D526576657273654F6E696F6E51756575652D7631');"
                ))
                .unwrap();
            create_legacy_main_schema(&connection);
            create_no_work_schema(&connection).unwrap();
            connection
                .execute(
                    &format!(
                        "INSERT INTO {NO_WORK_TABLE}
                         (claim_id, immediate_recipient, claim_commitment, claim_frame,
                          recorded_at, retained_until)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
                    ),
                    params![
                        [1; 16].as_slice(),
                        [2; 32].as_slice(),
                        [3; 32].as_slice(),
                        [4; 5].as_slice(),
                        observed_at,
                        observed_at + 10_000,
                    ],
                )
                .unwrap();
        }
        connection
    }

    fn legacy_connection_with_main_row(state: i64) -> Mutex<Connection> {
        let connection = legacy_connection(120);
        let connection_guard = connection.lock();
        connection_guard
            .execute(
                &format!(
                    "INSERT INTO {TABLE} (
                        queue_key, route_id, request_commitment, route_body_commitment,
                        immediate_recipient, envelope_commitment, envelope, state,
                        claim_id, claim_commitment, claim_frame, lease_id,
                        lease_commitment, lease_frame, execution_deadline, route_deadline,
                        result_frame, result_commitment, completed_at, retained_until
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                               ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                               ?17, ?18, ?19, ?20)"
                ),
                params![
                    [40; 32].as_slice(),
                    [41; 16].as_slice(),
                    [42; 32].as_slice(),
                    [43; 32].as_slice(),
                    [44; 32].as_slice(),
                    [45; 32].as_slice(),
                    [46; 3].as_slice(),
                    state,
                    [47; 16].as_slice(),
                    [48; 32].as_slice(),
                    [49; 5].as_slice(),
                    [50; 16].as_slice(),
                    [51; 32].as_slice(),
                    [52; 5].as_slice(),
                    150i64,
                    200i64,
                    if state == RESULT { Some([53; 6].as_slice()) } else { None },
                    if state == RESULT { Some([54; 32].as_slice()) } else { None },
                    if state == RESULT { Some(160i64) } else { None },
                    300i64,
                ],
            )
            .unwrap();
        drop(connection_guard);
        connection
    }

    fn v2_connection_with_main_row(state: i64) -> Mutex<Connection> {
        let connection = legacy_connection_with_main_row(state);
        let connection_guard = connection.lock();
        connection_guard
            .execute(
                &format!(
                    "ALTER TABLE {META_TABLE} ADD COLUMN clock_high_water INTEGER NOT NULL DEFAULT 0"
                ),
                [],
            )
            .unwrap();
        connection_guard
            .execute(
                &format!(
                    "UPDATE {META_TABLE} SET schema_version = 2, clock_high_water = 120 WHERE id = 1"
                ),
                [],
            )
            .unwrap();
        drop(connection_guard);
        connection
    }

    fn item(seed: u8, route_deadline: u64) -> ReverseOnionQueueItem {
        let source_node_id = source_node_id();
        ReverseOnionQueueItem::new(
            [seed; COMMITMENT_BYTES],
            [seed.wrapping_add(1); ID_BYTES],
            [seed.wrapping_add(2); COMMITMENT_BYTES],
            source_node_id,
            [seed.wrapping_add(3); COMMITMENT_BYTES],
            [seed.wrapping_add(4); COMMITMENT_BYTES],
            [seed.wrapping_add(5); COMMITMENT_BYTES],
            vec![seed; 3],
            route_deadline,
        )
        .unwrap()
    }

    fn source_node_id() -> [u8; COMMITMENT_BYTES] {
        aeronyx_core::crypto::IdentityKeyPair::from_bytes(&[0x71; 32])
            .expect("valid source identity")
            .public_key_bytes()
    }

    fn no_work_issue(
        queue: &SqliteReverseOnionQueue,
        connection: &Mutex<Connection>,
        recipient: [u8; COMMITMENT_BYTES],
        claim_id: [u8; ID_BYTES],
        claim_commitment: [u8; COMMITMENT_BYTES],
        claim_frame: Vec<u8>,
        now: u64,
        verify_calls: &Cell<u8>,
    ) -> Result<ReverseOnionQueueIssue, ReverseOnionQueueError> {
        queue.issue_lease(
            connection,
            recipient,
            claim_id,
            claim_commitment,
            claim_frame,
            now,
            |_| {
                verify_calls.set(verify_calls.get().saturating_add(1));
                Ok(claim_commitment)
            },
            |_, _| Err(ReverseOnionQueueError::Rejected),
        )
    }

    #[test]
    fn v1_migration_preserves_legacy_null_and_recipient_replay() {
        let connection = legacy_connection_with_main_row(ARMED);
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        queue.initialize_at(&connection, 150).unwrap();
        let lease = queue
            .lookup_armed(&connection, [40; 32], [45; 32], [44; 32], 170)
            .unwrap()
            .expect("legacy armed lease remains replayable");
        assert_eq!(lease.frame(), [52; 5].as_slice());
        assert!(queue
            .lookup_source(&connection, source_node_id(), [41; 16], [42; 32], 170)
            .unwrap()
            .is_none());
        let connection = connection.lock();
        let source: Option<Vec<u8>> = connection
            .query_row(
                &format!("SELECT source_node_id FROM {TABLE} WHERE queue_key = ?1"),
                params![[40; 32].as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(source.is_none(), "legacy NULL source must not be backfilled");
    }

    #[test]
    fn v2_migration_preserves_legacy_result_replay_and_null_source() {
        let connection = v2_connection_with_main_row(RESULT);
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        queue.initialize(&connection).unwrap();
        let result = queue
            .lookup_result(&connection, [40; 32], [45; 32], [44; 32], 170)
            .unwrap()
            .expect("legacy result remains replayable");
        assert_eq!(result.frame(), [53; 6].as_slice());
        assert!(queue
            .lookup_source(&connection, source_node_id(), [41; 16], [42; 32], 170)
            .unwrap()
            .is_none());
    }

    #[test]
    fn source_snapshot_is_exact_and_distinguishes_partial_from_complete() {
        // [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] The admitted
        // deadline must fit the fixture's actual configured route window.
        let (queue, connection) = initialized_queue(4, 600, 120);
        let item = item(60, 200);
        queue.enqueue(&connection, &item, 100).unwrap();
        let partial = queue
            .lookup_source(&connection, source_node_id(), item.route_id, item.request_commitment, 110)
            .unwrap()
            .expect("pending source row");
        assert_eq!(partial.source_node_id(), source_node_id());
        assert_eq!(partial.route_id(), item.route_id);
        assert_eq!(partial.request_commitment(), item.request_commitment);
        assert_eq!(partial.immediate_recipient(), item.immediate_recipient);
        assert_eq!(partial.route_deadline(), item.route_deadline);
        assert!(partial.claim_frame().is_none());
        assert!(partial.lease_frame().is_none());
        assert!(partial.result_frame().is_none());
        let claim_commitment = [63; 32];
        let lease = match queue
            .issue_lease(
                &connection,
                item.immediate_recipient,
                [61; 16],
                claim_commitment,
                vec![62; 5],
                110,
                |_| Ok(claim_commitment),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [64; 16],
                        [65; 32],
                        vec![66; 5],
                        150,
                        300,
                    )
                },
            )
            .unwrap()
        {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        queue
            .complete(&connection, &lease, &[67; 6], 120, |_, _| Ok([68; 32]))
            .unwrap();
        let complete = queue
            .lookup_source(&connection, source_node_id(), item.route_id, item.request_commitment, 130)
            .unwrap()
            .expect("complete source row");
        assert_eq!(complete.claim_frame(), Some([62; 5].as_slice()));
        assert_eq!(complete.lease_frame(), Some([66; 5].as_slice()));
        assert_eq!(complete.result_frame(), Some([67; 6].as_slice()));
    }

    #[test]
    fn source_lookup_rejects_wrong_source_and_expiry_without_mutation() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        let item = item(70, 110);
        queue.enqueue(&connection, &item, 100).unwrap();
        let wrong_source = aeronyx_core::crypto::IdentityKeyPair::from_bytes(&[0x72; 32])
            .unwrap()
            .public_key_bytes();
        assert!(queue
            .lookup_source(&connection, wrong_source, item.route_id, item.request_commitment, 105)
            .unwrap()
            .is_none());
        assert!(queue
            .lookup_source(&connection, source_node_id(), item.route_id, item.request_commitment, 110)
            .unwrap()
            .is_none());
        let remaining: i64 = connection
            .lock()
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 1, "read-only expiry lookup must not delete");
    }

    #[test]
    fn source_lookup_rejects_duplicate_source_tuple_fail_closed() {
        let (queue, connection) = initialized_queue(4, 600, 120);
        let item = item(71, 300);
        queue.enqueue(&connection, &item, 100).unwrap();
        let duplicate = ReverseOnionQueueItem::new(
            [72; 32],
            item.route_id,
            item.request_commitment,
            item.source_node_id,
            item.route_body_commitment,
            item.immediate_recipient,
            item.envelope_commitment,
            item.envelope.clone(),
            item.route_deadline,
        )
        .unwrap();
        queue.enqueue(&connection, &duplicate, 105).unwrap();
        assert!(matches!(
            queue.lookup_source(
                &connection,
                source_node_id(),
                item.route_id,
                item.request_commitment,
                106,
            ),
            Err(ReverseOnionQueueError::Corrupt)
        ));
    }

    #[test]
    fn source_lookup_rejects_clock_rollback_and_startup_audits_unmatched_corruption() {
        let (queue, connection) = initialized_queue(4, 600, 120);
        let item = item(80, 300);
        queue.enqueue(&connection, &item, 100).unwrap();
        queue.cleanup(&connection, 200).unwrap();
        let before: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(matches!(
            queue.lookup_source(&connection, source_node_id(), item.route_id, item.request_commitment, 150),
            Err(ReverseOnionQueueError::Rejected)
        ));
        let after: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(before, after);
        connection
            .lock()
            .execute(
                &format!("UPDATE {TABLE} SET source_node_id = ?1 WHERE queue_key = ?2"),
                params![vec![1u8; 31], item.queue_key.as_slice()],
            )
            .unwrap();
        // A malformed source no longer matches this exact tuple. Hot reads
        // return no data; startup/maintenance, not unrelated reads, audit it.
        assert!(queue.lookup_source(&connection, source_node_id(), item.route_id,
            item.request_commitment, 200).unwrap().is_none());
        assert!(matches!(queue.initialize(&connection), Err(ReverseOnionQueueError::Corrupt)));
        connection
            .lock()
            .execute(
                &format!("UPDATE {TABLE} SET source_node_id = ?1 WHERE queue_key = ?2"),
                params![vec![0u8; 32], item.queue_key.as_slice()],
            )
            .unwrap();
        assert!(queue.lookup_source(&connection, source_node_id(), item.route_id,
            item.request_commitment, 200).unwrap().is_none());
        assert!(matches!(queue.initialize(&connection), Err(ReverseOnionQueueError::Corrupt)));
    }

    #[test]
    fn exact_no_work_retry_is_durable_and_conflict_fenced() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        let calls = Cell::new(0);
        let recipient = [7; COMMITMENT_BYTES];
        let claim_id = [8; ID_BYTES];
        let claim_commitment = [9; COMMITMENT_BYTES];
        let claim_frame = vec![10; 5];
        assert!(matches!(
            no_work_issue(
                &queue,
                &connection,
                recipient,
                claim_id,
                claim_commitment,
                claim_frame.clone(),
                100,
                &calls,
            )
            .unwrap(),
            ReverseOnionQueueIssue::NoWork
        ));
        assert!(matches!(
            {
                let reopened = SqliteReverseOnionQueue::new(limits(4, 60, 120));
                reopened.initialize(&connection).unwrap();
                no_work_issue(
                    &reopened,
                    &connection,
                    recipient,
                    claim_id,
                    claim_commitment,
                    claim_frame,
                    101,
                    &calls,
                )
                .unwrap()
            },
            ReverseOnionQueueIssue::NoWork
        ));
        assert_eq!(calls.get(), 1);
        assert!(matches!(
            no_work_issue(
                &queue,
                &connection,
                recipient,
                claim_id,
                [11; COMMITMENT_BYTES],
                vec![10; 5],
                102,
                &calls,
            ),
            Err(ReverseOnionQueueError::Conflict)
        ));
    }

    #[test]
    fn no_work_and_main_rows_share_global_and_recipient_quota() {
        let (queue, connection) = initialized_queue(1, 60, 120);
        let calls = Cell::new(0);
        assert!(matches!(
            no_work_issue(
                &queue,
                &connection,
                [12; COMMITMENT_BYTES],
                [13; ID_BYTES],
                [14; COMMITMENT_BYTES],
                vec![15; 4],
                100,
                &calls,
            )
            .unwrap(),
            ReverseOnionQueueIssue::NoWork
        ));
        assert!(matches!(
            queue.enqueue(&connection, &item(16, 150), 100),
            Err(ReverseOnionQueueError::Capacity)
        ));
    }

    #[test]
    fn new_admission_caps_do_not_truncate_existing_evidence() {
        let queue = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(200)
            .unwrap();
        let connection = connection();
        queue.initialize(&connection).unwrap();
        let now = 100;
        let queued = item(20, 200);
        assert_eq!(
            queue.enqueue(&connection, &queued, now).unwrap(),
            ReverseOnionQueueAdmission::Created
        );
        assert!(matches!(
            queue.enqueue(&connection, &item(21, 301), now),
            Err(ReverseOnionQueueError::Rejected)
        ));
        let claim_id = [22; ID_BYTES];
        let claim_commitment = [23; COMMITMENT_BYTES];
        assert!(matches!(
            queue.issue_lease(
                &connection,
                queued.immediate_recipient,
                claim_id,
                claim_commitment,
                vec![24; 6],
                now,
                |_| Ok(claim_commitment),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [25; ID_BYTES],
                        [26; COMMITMENT_BYTES],
                        vec![27; 6],
                        now + 101,
                        now + 250,
                    )
                },
            ),
            Err(ReverseOnionQueueError::Rejected)
        ));
        let lease = queue.issue_lease(
            &connection,
            queued.immediate_recipient,
            claim_id,
            claim_commitment,
            vec![24; 6],
            now,
            |_| Ok(claim_commitment),
            |_, _| {
                ReverseOnionQueueLeaseMaterial::new(
                    [25; ID_BYTES],
                    [26; COMMITMENT_BYTES],
                    vec![27; 6],
                    now + 50,
                    now + 250,
                )
            },
        );
        let lease = match lease.unwrap() {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        let lowered = SqliteReverseOnionQueue::new(limits(4, 10, 10));
        let recovered = lowered
            .lookup_armed(
                &connection,
                lease.queue_key,
                lease.envelope_commitment,
                queued.immediate_recipient,
                now + 60,
            )
            .unwrap()
            .unwrap();
        assert_eq!(recovered.frame(), lease.frame());
        let result_frame = vec![28; 7];
        assert_eq!(
            lowered
                .complete(&connection, &recovered, &result_frame, now + 60, |context, frame| {
                    assert_eq!(context.claim_frame(), [24; 6].as_slice());
                    assert_eq!(context.lease_frame(), [27; 6].as_slice());
                    assert_eq!(context.envelope(), [20; 3].as_slice());
                    assert_eq!(context.envelope_commitment(), [25; COMMITMENT_BYTES]);
                    assert_eq!(context.route_id(), [21; ID_BYTES]);
                    assert_eq!(context.route_deadline(), now + 100);
                    assert_eq!(frame, result_frame.as_slice());
                    Ok([29; COMMITMENT_BYTES])
                })
                .unwrap(),
            ReverseOnionQueueCompletion::Stored
        );
        assert_eq!(
            lowered
                .lookup_result(
                    &connection,
                    recovered.queue_key,
                    recovered.envelope_commitment,
                    queued.immediate_recipient,
                    now + 60,
                )
                .unwrap()
                .unwrap()
                .frame(),
            result_frame.as_slice()
        );
    }

    fn insert_raw_pending(connection: &Mutex<Connection>, seed: u8, now: i64) {
        let connection = connection.lock();
        connection
            .execute(
                &format!(
                    "INSERT INTO {TABLE} (
                        queue_key, route_id, request_commitment, route_body_commitment,
                        immediate_recipient, envelope_commitment, envelope, state,
                        claim_id, claim_commitment, claim_frame, lease_id,
                        lease_commitment, lease_frame, execution_deadline, route_deadline,
                        result_frame, result_commitment, completed_at, retained_until
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                              NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?9,
                              NULL, NULL, NULL, ?9)"
                ),
                params![
                    [seed; COMMITMENT_BYTES].as_slice(),
                    [seed.wrapping_add(1); ID_BYTES].as_slice(),
                    [seed.wrapping_add(2); COMMITMENT_BYTES].as_slice(),
                    [seed.wrapping_add(3); COMMITMENT_BYTES].as_slice(),
                    [seed.wrapping_add(4); COMMITMENT_BYTES].as_slice(),
                    [seed.wrapping_add(5); COMMITMENT_BYTES].as_slice(),
                    [seed.wrapping_add(6); 3].as_slice(),
                    PENDING,
                    now + 100,
                ],
            )
            .unwrap();
    }

    #[test]
    fn corrupt_row_scan_is_bounded_by_configured_item_limit() {
        let (queue, connection) = initialized_queue(1, 60, 120);
        insert_raw_pending(&connection, 30, 100);
        insert_raw_pending(&connection, 40, 100);
        assert!(matches!(
            queue.cleanup(&connection, 100),
            Err(ReverseOnionQueueError::Capacity)
        ));
    }

    #[test]
    fn foreign_schema_is_rejected_without_create_side_effect() {
        let connection = connection();
        {
            let connection = connection.lock();
            connection
                .execute(&format!("CREATE TABLE {TABLE} (foreign_value BLOB)"), [])
                .unwrap();
        }
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::MigrationRequired)
        );
        let connection = connection.lock();
        let objects: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(objects, 1);
    }

    #[test]
    fn exact_shape_foreign_schema_requires_explicit_migration() {
        let connection = connection();
        {
            let connection = connection.lock();
            create_main_schema(&connection).unwrap();
            create_no_work_schema(&connection).unwrap();
        }
        let before = application_objects(&connection.lock()).unwrap();
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::MigrationRequired)
        );
        assert_eq!(application_objects(&connection.lock()).unwrap(), before);
    }

    #[test]
    fn foreign_extra_object_is_rejected_without_schema_mutation() {
        let connection = connection();
        {
            let connection = connection.lock();
            create_main_schema(&connection).unwrap();
            create_no_work_schema(&connection).unwrap();
            connection
                .execute("CREATE TABLE foreign_extra (opaque BLOB)", [])
                .unwrap();
        }
        let before = application_objects(&connection.lock()).unwrap();
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::MigrationRequired)
        );
        assert_eq!(application_objects(&connection.lock()).unwrap(), before);
    }

    #[test]
    fn marker_version_mismatch_is_rejected_without_mutation() {
        let connection = connection();
        {
            let connection = connection.lock();
            create_meta_schema(&connection).unwrap();
            connection
                .execute(
                    &format!("UPDATE {META_TABLE} SET schema_version = 99 WHERE id = 1"),
                    [],
                )
                .unwrap();
        }
        let before = application_objects(&connection.lock()).unwrap();
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::Corrupt)
        );
        assert_eq!(application_objects(&connection.lock()).unwrap(), before);
    }

    #[test]
    fn owned_unknown_object_is_rejected_without_mutation() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        // [REVERSE-ONION-INVENTORY-FIXTURE 2026-10-04 by Codex] Snapshot
        // the deliberately over-cap fixture without invoking the production
        // inventory gate that initialize is expected to reject below.
        let snapshot = || {
            let connection = connection.lock();
            let mut statement = connection
                .prepare("SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master ORDER BY type, name")
                .unwrap();
            let objects = statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            objects
        };
        {
            let connection = connection.lock();
            connection
                .execute("CREATE TABLE foreign_extra (opaque BLOB)", [])
                .unwrap();
        }
        let before = snapshot();
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::Corrupt)
        );
        assert_eq!(snapshot(), before);
    }

    #[test]
    fn owned_oversized_object_name_is_rejected_without_mutation() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        let object_name = "x".repeat(256);
        {
            let connection = connection.lock();
            connection
                .execute(
                    &format!("CREATE TABLE \"{object_name}\" (opaque BLOB)"),
                    [],
                )
                .unwrap();
        }
        let before: i64 = connection
            .lock()
            .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::Corrupt)
        );
        let after: i64 = connection
            .lock()
            .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn unmarked_object_inventory_is_bounded_before_migration_rejection() {
        let connection = connection();
        {
            let connection = connection.lock();
            for index in 0..(MAX_APPLICATION_OBJECTS + 1) {
                connection
                    .execute(
                        &format!("CREATE TABLE foreign_inventory_{index} (opaque BLOB)"),
                        [],
                    )
                    .unwrap();
            }
        }
        let before: i64 = connection
            .lock()
            .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap();
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection),
            Err(ReverseOnionQueueError::MigrationRequired)
        );
        let after: i64 = connection
            .lock()
            .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn owned_schema_reopen_is_idempotent() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        let before = application_objects(&connection.lock()).unwrap();
        queue.initialize(&connection).unwrap();
        assert_eq!(application_objects(&connection.lock()).unwrap(), before);
    }

    #[test]
    fn restart_completion_recovery_fences_identity_and_reuses_exact_context() {
        let queue = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(200)
            .unwrap();
        let connection = connection();
        queue.initialize(&connection).unwrap();
        let now = 100;
        let queued = item(50, 200);
        queue.enqueue(&connection, &queued, now).unwrap();
        let lease = match queue
            .issue_lease(
                &connection,
                queued.immediate_recipient,
                [51; ID_BYTES],
                [52; COMMITMENT_BYTES],
                vec![53; 5],
                now,
                |_| Ok([52; COMMITMENT_BYTES]),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [54; ID_BYTES],
                        [55; COMMITMENT_BYTES],
                        vec![56; 5],
                        now + 20,
                        now + 220,
                    )
                },
            )
            .unwrap()
        {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        let restarted = SqliteReverseOnionQueue::new(limits(4, 10, 10));
        assert!(matches!(
            restarted.lookup_armed(
                &connection,
                lease.queue_key,
                [57; COMMITMENT_BYTES],
                queued.immediate_recipient,
                now + 1,
            ),
            Err(ReverseOnionQueueError::Conflict)
        ));
        let recovered = restarted
            .lookup_armed(
                &connection,
                lease.queue_key,
                lease.envelope_commitment,
                queued.immediate_recipient,
                now + 1,
            )
            .unwrap()
            .unwrap();
        let mut wrong = ReverseOnionQueueIssuedLease {
            queue_key: recovered.queue_key,
            envelope_commitment: recovered.envelope_commitment,
            lease_id: [58; ID_BYTES],
            lease_commitment: recovered.lease_commitment,
            frame: recovered.frame.clone(),
        };
        let wrong_callback_calls = Cell::new(0);
        assert!(matches!(
            restarted.complete(&connection, &wrong, &[59; 4], now + 1, |_, _| {
                wrong_callback_calls.set(wrong_callback_calls.get().saturating_add(1));
                Ok([60; COMMITMENT_BYTES])
            }),
            Err(ReverseOnionQueueError::LeaseLost)
        ));
        assert_eq!(wrong_callback_calls.get(), 0);
        wrong.lease_id = recovered.lease_id;
        assert_eq!(
            restarted
                .complete(&connection, &recovered, &[61; 4], now + 1, |context, _| {
                    assert_eq!(context.claim_frame(), [53; 5].as_slice());
                    assert_eq!(context.lease_frame(), [56; 5].as_slice());
                    assert_eq!(context.envelope(), [50; 3].as_slice());
                    assert_eq!(context.envelope_commitment(), [55; COMMITMENT_BYTES]);
                    assert_eq!(context.route_id(), [51; ID_BYTES]);
                    Ok([62; COMMITMENT_BYTES])
                })
                .unwrap(),
            ReverseOnionQueueCompletion::Stored
        );
    }

    #[test]
    fn owned_v1_migration_uses_observed_time_and_trusted_now_only() {
        let connection = legacy_connection(120);
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        queue.initialize_at(&connection, 150).unwrap();
        let connection = connection.lock();
        let (version, clock, retained): (i64, i64, i64) = connection
            .query_row(
                &format!(
                    "SELECT m.schema_version, m.clock_high_water, n.retained_until
                       FROM {META_TABLE} m CROSS JOIN {NO_WORK_TABLE} n
                      WHERE m.id = 1 AND n.claim_id = ?1"
                ),
                params![[1; 16].as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        assert_eq!(clock, 150);
        assert_eq!(retained, 10_120);
    }

    #[test]
    fn legacy_initialize_requires_explicit_trusted_migration_time() {
        let connection = legacy_connection(120);
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize(&connection).unwrap_err(),
            ReverseOnionQueueError::MigrationRequired
        );
        let connection = connection.lock();
        let version: i64 = connection
            .query_row(
                &format!("SELECT schema_version FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, LEGACY_SCHEMA_VERSION);
    }

    #[test]
    fn legacy_migration_rejects_observed_time_after_trusted_now_without_schema_change() {
        let connection = legacy_connection(151);
        let queue = SqliteReverseOnionQueue::new(limits(4, 60, 120));
        assert_eq!(
            queue.initialize_at(&connection, 150),
            Err(ReverseOnionQueueError::Rejected)
        );
        let connection = connection.lock();
        let columns: Vec<String> = connection
            .prepare(&format!("PRAGMA table_info({META_TABLE})"))
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            columns,
            vec![
                "id".to_owned(),
                "schema_version".to_owned(),
                "ownership_tag".to_owned(),
            ]
        );
    }

    #[test]
    fn rollback_lookup_rejects_without_mutation_and_valid_lookup_advances_clock() {
        let (queue, connection) = initialized_queue(4, 60, 120);
        let queued = item(71, 300);
        queue.enqueue(&connection, &queued, 100).unwrap();
        assert_eq!(
            queue.enqueue(&connection, &queued, 101).unwrap(),
            ReverseOnionQueueAdmission::Existing
        );
        let clock: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(clock, 101);
        assert!(matches!(
            queue.lookup_result(
                &connection,
                queued.queue_key,
                queued.envelope_commitment,
                queued.immediate_recipient,
                99,
            ),
            Err(ReverseOnionQueueError::Rejected)
        ));
        let clock: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(clock, 101);
        assert_eq!(
            queue
                .lookup_result(
                    &connection,
                    queued.queue_key,
                    queued.envelope_commitment,
                    queued.immediate_recipient,
                    101,
                )
                .unwrap(),
            None
        );
        let clock: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(clock, 101);
    }

    #[test]
    fn expired_complete_outcomes_commit_observed_clock_without_partial_result() {
        let queue = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(400)
            .unwrap();
        let connection = connection();
        queue.initialize(&connection).unwrap();
        let queued = item(72, 400);
        queue.enqueue(&connection, &queued, 100).unwrap();
        let lease = match queue
            .issue_lease(
                &connection,
                queued.immediate_recipient,
                [73; ID_BYTES],
                [74; COMMITMENT_BYTES],
                vec![75; 4],
                100,
                |_| Ok([74; COMMITMENT_BYTES]),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [76; ID_BYTES],
                        [77; COMMITMENT_BYTES],
                        vec![78; 4],
                        120,
                        300,
                    )
                },
            )
            .unwrap()
        {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        assert_eq!(
            queue
                .complete(&connection, &lease, &[79; 4], 120, |_, _| {
                    Ok([80; COMMITMENT_BYTES])
                })
                .unwrap(),
            ReverseOnionQueueCompletion::Stored
        );
        assert_eq!(
            queue
                .complete(&connection, &lease, &[79; 4], 301, |_, _| {
                    Err(ReverseOnionQueueError::Rejected)
                })
                .unwrap_err(),
            ReverseOnionQueueError::NoWork
        );
        let clock: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(clock, 301);
        let connection = connection.lock();
        let (state, stored): (i64, Vec<u8>) = connection
            .query_row(
                &format!("SELECT state, result_frame FROM {TABLE} WHERE queue_key = ?1"),
                params![lease.queue_key.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, RESULT);
        assert_eq!(stored, [79; 4]);
    }

    #[test]
    fn expired_armed_complete_commits_clock_but_keeps_row_armed() {
        let queue = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(400)
            .unwrap();
        let connection = connection();
        queue.initialize(&connection).unwrap();
        let queued = item(81, 400);
        queue.enqueue(&connection, &queued, 100).unwrap();
        let lease = match queue
            .issue_lease(
                &connection,
                queued.immediate_recipient,
                [82; ID_BYTES],
                [83; COMMITMENT_BYTES],
                vec![84; 4],
                100,
                |_| Ok([83; COMMITMENT_BYTES]),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [85; ID_BYTES],
                        [86; COMMITMENT_BYTES],
                        vec![87; 4],
                        120,
                        300,
                    )
                },
            )
            .unwrap()
        {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        assert_eq!(
            queue
                .complete(&connection, &lease, &[88; 4], 301, |_, _| {
                    Err(ReverseOnionQueueError::Rejected)
                })
                .unwrap_err(),
            ReverseOnionQueueError::Ambiguous
        );
        let clock: i64 = connection
            .lock()
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(clock, 301);
        let connection = connection.lock();
        let (state, result_frame): (i64, Option<Vec<u8>>) = connection
            .query_row(
                &format!("SELECT state, result_frame FROM {TABLE} WHERE queue_key = ?1"),
                params![lease.queue_key.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, ARMED);
        assert_eq!(result_frame, None);
    }

    #[test]
    fn result_context_lookup_is_authenticated_read_only_and_restart_safe() {
        let queue = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(400)
            .unwrap();
        let connection = connection();
        queue.initialize(&connection).unwrap();
        let queued = item(91, 400);
        queue.enqueue(&connection, &queued, 100).unwrap();
        let claim_id = [92; ID_BYTES];
        let claim_commitment = [93; COMMITMENT_BYTES];
        let claim_frame = vec![94; 4];
        let lease = match queue
            .issue_lease(
                &connection,
                queued.immediate_recipient,
                claim_id,
                claim_commitment,
                claim_frame,
                100,
                |_| Ok(claim_commitment),
                |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        [95; ID_BYTES],
                        [96; COMMITMENT_BYTES],
                        vec![97; 4],
                        120,
                        300,
                    )
                },
            )
            .unwrap()
        {
            ReverseOnionQueueIssue::Issued(lease) => lease,
            _ => panic!("expected issued lease"),
        };
        let wrong = queue
            .lookup_result_context(
                &connection,
                [99; COMMITMENT_BYTES],
                claim_id,
                lease_id(&lease),
                queued.route_id,
                101,
            )
            .unwrap();
        assert!(matches!(wrong, ReverseOnionQueueResultContext::NoWork));
        let connection_guard = connection.lock();
        let row_count: i64 = connection_guard
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |row| row.get(0))
            .unwrap();
        let clock: i64 = connection_guard
            .query_row(
                &format!("SELECT clock_high_water FROM {META_TABLE} WHERE id = 1"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(row_count, 1);
        assert_eq!(clock, 101);
        drop(connection_guard);
        for (claim, lease_id_value, route) in [
            ([98; ID_BYTES], lease_id(&lease), queued.route_id),
            (claim_id, [97; ID_BYTES], queued.route_id),
            (claim_id, lease_id(&lease), [98; ID_BYTES]),
        ] {
            let outcome = queue
                .lookup_result_context(
                    &connection,
                    queued.immediate_recipient,
                    claim,
                    lease_id_value,
                    route,
                    102,
                )
                .unwrap();
            assert!(matches!(outcome, ReverseOnionQueueResultContext::NoWork));
        }
        let reopened = SqliteReverseOnionQueue::new(limits(4, 100, 200))
            .with_route_max_secs(400)
            .unwrap();
        reopened.initialize(&connection).unwrap();
        let expected_lease_frame = lease.frame().to_vec();
        let armed = reopened
            .lookup_result_context(
                &connection,
                queued.immediate_recipient,
                claim_id,
                lease_id(&lease),
                queued.route_id,
                103,
            )
            .unwrap();
        let recovered = match armed {
            ReverseOnionQueueResultContext::Armed(lease) => {
                assert_eq!(lease.frame(), expected_lease_frame.as_slice());
                lease
            }
            _ => panic!("expected armed context"),
        };
        assert_eq!(
            reopened
                .complete(&connection, &recovered, &[101; 4], 110, |_, _| {
                    Ok([102; COMMITMENT_BYTES])
                })
                .unwrap(),
            ReverseOnionQueueCompletion::Stored
        );
        let stored = reopened
            .lookup_result_context(
                &connection,
                queued.immediate_recipient,
                claim_id,
                lease_id(&lease),
                queued.route_id,
                111,
            )
            .unwrap();
        match stored {
            ReverseOnionQueueResultContext::Result(result) => {
                assert_eq!(result.frame(), &[101; 4]);
            }
            _ => panic!("expected stored result context"),
        }
        let expired = reopened
            .lookup_result_context(
                &connection,
                queued.immediate_recipient,
                claim_id,
                lease_id(&lease),
                queued.route_id,
                301,
            )
            .unwrap();
        assert!(matches!(expired, ReverseOnionQueueResultContext::NoWork));
    }

    fn lease_id(lease: &ReverseOnionQueueIssuedLease) -> [u8; ID_BYTES] {
        lease.lease_id
    }
}
