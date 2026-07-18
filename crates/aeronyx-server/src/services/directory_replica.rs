// ============================================================================
// File: crates/aeronyx-server/src/services/directory_replica.rs
// ============================================================================
//! # Directory Chain Replica Store
//!
//! ## Creation Reason
//! Directory Sync responses are producer attestations, not global consensus.
//! A node therefore needs a durable producer-scoped replica namespace instead
//! of inserting remote blocks into its own locally produced Directory Chain.
//!
//! ## Main Functionality
//! - Stores each remote producer chain under an independent `SQLite` namespace.
//! - Re-verifies signed response evidence, blocks, commitments, and descriptor
//!   objects before one atomic import transaction.
//! - Makes exact repeated pages idempotent.
//! - Persists signed fork/rollback evidence and permanently quarantines only
//!   the producer that authored conflicting chain claims.
//! - Records authenticated descriptor equivocation without blaming an honest
//!   chain producer that merely observed the conflicting public descriptors.
//! - Audits every replica chain and index before the node may synchronize.
//! - Persists bounded producer retry state across process restarts and clears
//!   it atomically with the next authenticated successful page.
//! - Computes a bounded recent-window intersection of exact descriptor
//!   commitments across non-quarantined configured producer replicas.
//! - Exposes low-cost aggregate snapshots and privacy-safe synchronization
//!   observations, including bounded retry state, without re-running a full
//!   cryptographic audit per API read.
//!
//! ## Calling Relationships
//! - `server.rs` opens this store beside `DirectoryChainStore` at startup.
//! - `api/directory_replica_sync.rs` verifies and downloads bounded peer pages,
//!   then calls `import_verified_page` from a blocking worker.
//! - `api/directory_replica_status.rs` reads only low-cost audited snapshots.
//! - The local producer store remains the only source served by peer routes.
//!
//! ## Main Logical Flow
//! 1. Open the existing Directory Chain `SQLite` file and initialize only the
//!    `directory_replica_*` tables.
//! 2. Pin schema, chain id, and the local node identity in replica metadata.
//! 3. Audit every accepted producer prefix and all durable incident evidence.
//! 4. Re-verify the signed range-response frame and exact descriptor objects.
//! 5. Atomically append a contiguous producer prefix, clear its retry state,
//!    or persist quarantine without mutating another producer namespace.
//! 6. Derive recent multi-source observation evidence without choosing a fork,
//!    producer, quorum, or globally finalized height.
//!
//! ## Privacy Invariant
//! Replica tables contain only public signed node descriptors, public
//! descriptor commitments, signed Directory Chain blocks, and signed incident
//! evidence. They must never contain client identities, IPs, routes, selected
//! hops, message ids, payloads, ciphertext, Memory Chain records, DNS contents,
//! destinations, private keys, or wallet traffic.
//!
//! ## Important Note for Next Developer
//! - Never merge remote blocks into `directory_chain_blocks`.
//! - Never auto-delete, auto-rewind, or auto-select through a quarantined fork.
//! - A producer-signed block fork/rollback quarantines that producer. A signed
//!   descriptor equivocation is evidence about the descriptor owner and does
//!   not automatically quarantine the observing producer.
//! - Keep all limits synchronized with the core Directory Sync V1 contract.
//! - Observation convergence is a local digest over independently verified
//!   producer evidence. It is never consensus, voting, fork choice, or finality.
//!
//! ## Last Modified
//! v0.4.0-DirectoryReplicaObservationConvergence - Added bounded recent-window
//! multi-source commitment overlap and a deterministic local observation root.
//! v0.3.0-DirectoryReplicaDurableRetry - Added an atomic schema v1-to-v2
//! migration and audited restart-durable producer retry state.
//! v0.2.2-DirectoryReplicaRetryRuntime - Added producer-local retry boundaries
//! and backoff skip counters to process-lifetime synchronization telemetry.
//! v0.2.1-DirectoryReplicaModuleSplit - Updated transport and status ownership.
//! v0.2.0-DirectoryReplicaStatus - Added aggregate status snapshots and shared
//! synchronization observations for bounded catch-up visibility.
//! v0.1.0-DirectoryReplicaStore - Initial producer-isolated replica persistence.
// ============================================================================

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_response_signing_bytes,
    encode_directory_sync_message, DirectoryCommitmentBlockV1, DirectoryCommitmentValidationError,
    DirectoryDescriptorCommitmentV1, DirectorySyncMessage, SignedNodeDescriptor,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_SYNC_BLOCKS_V1,
};
use bincode::Options;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};

const DIRECTORY_REPLICA_SCHEMA_VERSION: i64 = 2;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V1: i64 = 1;
const MAX_DIRECTORY_BLOCK_BYTES: u64 = 64 * 1024;
const MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES: u64 = 32 * 1024;
const MAX_DIRECTORY_SYNC_EVIDENCE_BYTES: usize = 512 * 1024;
const DIRECTORY_REPLICA_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;
const MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES: usize = 96;
const DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS: u64 = 32;
const MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS: usize = 16;
/// Maximum producer failure streak retained in memory and audited `SQLite`.
pub(crate) const DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES: u64 = 64;
/// Maximum durable retry delay accepted by the replica store and scheduler.
pub(crate) const DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS: u64 = 30 * 60;

/// Failures returned by the producer-isolated replica store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryReplicaStoreError {
    /// Filesystem setup failed.
    #[error("directory replica filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// `SQLite` rejected a schema, query, or transaction operation.
    #[error("directory replica sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A protocol object could not be encoded or decoded safely.
    #[error("directory replica codec error: {0}")]
    Codec(String),
    /// A descriptor object did not reproduce its signed commitment.
    #[error("directory replica descriptor error: {0}")]
    Descriptor(String),
    /// A block failed the canonical Directory Chain V1 contract.
    #[error("directory replica block validation error: {0}")]
    Block(#[from] DirectoryCommitmentValidationError),
    /// Durable metadata, chain, index, or evidence is inconsistent.
    #[error("directory replica integrity error: {0}")]
    Integrity(String),
    /// A bounded import request violates the V1 transport contract.
    #[error("directory replica request error: {0}")]
    Request(String),
    /// The producer is durably isolated pending operator review.
    #[error("directory producer is quarantined: {0}")]
    Quarantined(String),
}

/// Aggregate result of a complete replica startup audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryReplicaAudit {
    /// Number of producer namespaces.
    pub producers: u64,
    /// Number of producer namespaces currently quarantined.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks.
    pub blocks: u64,
    /// Number of commitments exactly matched to block payloads.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Number of audited producer-local retry rows.
    pub retry_states: u64,
}

/// Low-cost aggregate view of the already audited replica namespace.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaStoreSnapshot {
    /// Number of producer namespaces currently persisted.
    pub producers: u64,
    /// Number of producer namespaces blocked by durable quarantine.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks retained across all producers.
    pub blocks: u64,
    /// Number of verified descriptor commitments retained across all producers.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Per-producer accepted-prefix summaries for local operator presentation.
    pub producer_snapshots: Vec<DirectoryReplicaProducerSnapshot>,
}

/// Bounded, locally recomputable overlap across verified producer replicas.
///
/// This snapshot compares exact commitment hashes from each eligible
/// producer's most recent block window. It does not assign voting weight,
/// choose a chain, or create a globally finalized checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaObservationConvergenceSnapshot {
    /// Unique producer pins supplied by the validated node configuration.
    pub configured_producers: u64,
    /// Configured producers with a non-empty, non-quarantined accepted prefix.
    pub eligible_producers: u64,
    /// Configured producers that have not supplied an accepted block yet.
    pub pending_producers: u64,
    /// Configured producers excluded because signed evidence quarantined them.
    pub excluded_quarantined_producers: u64,
    /// Maximum number of recent blocks inspected per eligible producer.
    pub window_blocks: u64,
    /// Commitment observations across all eligible producer windows.
    pub recent_commitments: u64,
    /// Unique commitment hashes across all eligible producer windows.
    pub distinct_recent_commitments: u64,
    /// Commitments observed by at least two eligible producer chains.
    pub multi_source_recent_commitments: u64,
    /// Commitments observed by every eligible producer when at least two exist.
    pub all_eligible_source_recent_commitments: u64,
    /// Deterministic digest of eligible tips and their exact common commitments.
    pub observation_root: Option<[u8; 32]>,
}

/// Persisted aggregate state for one producer namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaProducerSnapshot {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Timestamp signed into the accepted tip block.
    pub tip_timestamp: u64,
    /// Whether imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable authenticated incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Last time this namespace metadata changed locally.
    pub updated_at: u64,
    /// Verified blocks retained for this producer.
    pub blocks: u64,
    /// Verified commitments retained for this producer.
    pub commitments: u64,
    /// Durable incidents attributed to this producer response stream.
    pub incidents: u64,
}

/// Current accepted prefix and isolation state for one producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaTip {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Accepted tip hash, or zero for an empty prefix.
    pub tip_hash: [u8; 32],
    /// Accepted tip timestamp, or zero for an empty prefix.
    pub tip_timestamp: u64,
    /// Whether further imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable incident kind when quarantined.
    pub quarantine_kind: Option<String>,
}

impl DirectoryReplicaTip {
    fn empty(producer: [u8; 32]) -> Self {
        Self {
            producer,
            tip_height: 0,
            tip_hash: [0u8; 32],
            tip_timestamp: 0,
            quarantined: false,
            quarantine_kind: None,
        }
    }
}

/// Result of one verified, atomic bounded-page import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaImportReport {
    /// New blocks committed by this transaction.
    pub blocks_inserted: u64,
    /// Exact existing blocks accepted idempotently.
    pub blocks_already_present: u64,
    /// New descriptor commitments committed by this transaction.
    pub commitments_inserted: u64,
    /// Newly recorded same-node/same-sequence descriptor conflicts.
    pub descriptor_equivocations: u64,
    /// Accepted producer prefix height after import.
    pub tip_height: u64,
    /// Accepted producer prefix hash after import.
    pub tip_hash: [u8; 32],
}

/// Restart-durable producer-local synchronization failure state.
///
/// The state contains bounded control-plane scheduling metadata only. It never
/// contains endpoints, response bodies, descriptors, routes, payloads, client
/// identifiers, private keys, wallet traffic, or social graph data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaRetryState {
    /// Remote producer identity used as the local scheduling key.
    pub producer: [u8; 32],
    /// Consecutive failures since the last authenticated successful page.
    pub consecutive_failures: u64,
    /// Earliest Unix timestamp at which another pull may begin.
    pub retry_not_before: Option<u64>,
    /// Timestamp of the most recent failed pull.
    pub last_failure_at: u64,
    /// Stable bounded internal failure bucket.
    pub last_failure_reason: String,
    /// Number of timer rounds skipped while durable backoff was active.
    pub backoff_skips: u64,
}

/// Runtime-only synchronization observation for one pinned producer.
///
/// These fields intentionally contain no endpoint, full response, descriptor,
/// route, payload, client, or wallet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaSyncObservation {
    /// Pinned producer identity used only for internal status correlation.
    pub producer: [u8; 32],
    /// Most recent bounded pull attempt.
    pub last_attempt_at: Option<u64>,
    /// Most recent authenticated successful page.
    pub last_success_at: Option<u64>,
    /// Most recent rejected or failed page.
    pub last_failure_at: Option<u64>,
    /// Stable privacy-safe reason code from the most recent failure.
    pub last_failure_reason: Option<String>,
    /// Earliest Unix timestamp at which this producer may be attempted again.
    pub retry_not_before: Option<u64>,
    /// Signed remote tip height most recently observed.
    pub remote_tip_height: Option<u64>,
    /// Accepted local replica height after the most recent success.
    pub local_tip_height: u64,
    /// Whether the most recent signed response indicated additional pages.
    pub has_more: bool,
    /// Consecutive failed attempts since the last successful page.
    pub consecutive_failures: u64,
    /// Total bounded attempts during this process lifetime.
    pub total_attempts: u64,
    /// Total authenticated pages accepted during this process lifetime.
    pub successful_pages: u64,
    /// Total failed attempts during this process lifetime.
    pub failed_attempts: u64,
    /// Total scheduled rounds skipped while this producer was in backoff.
    pub backoff_skips: u64,
    /// Total new blocks committed during this process lifetime.
    pub blocks_inserted: u64,
    /// Total new commitments committed during this process lifetime.
    pub commitments_inserted: u64,
    /// Total HTTP requests consumed by authenticated successful pages.
    pub requests_sent: u64,
}

impl DirectoryReplicaSyncObservation {
    fn new(producer: [u8; 32]) -> Self {
        Self {
            producer,
            last_attempt_at: None,
            last_success_at: None,
            last_failure_at: None,
            last_failure_reason: None,
            retry_not_before: None,
            remote_tip_height: None,
            local_tip_height: 0,
            has_more: false,
            consecutive_failures: 0,
            total_attempts: 0,
            successful_pages: 0,
            failed_attempts: 0,
            backoff_skips: 0,
            blocks_inserted: 0,
            commitments_inserted: 0,
            requests_sent: 0,
        }
    }
}

/// Shared process-lifetime synchronization telemetry.
#[derive(Debug, Default)]
pub struct DirectoryReplicaSyncRuntime {
    observations: Mutex<HashMap<[u8; 32], DirectoryReplicaSyncObservation>>,
}

impl DirectoryReplicaSyncRuntime {
    /// Registers configured pins so status reports can distinguish pending from
    /// disabled before the first low-frequency synchronization round.
    pub fn register_producers(&self, producers: &[[u8; 32]]) {
        let mut observations = self.observations.lock();
        for producer in producers {
            if *producer != [0u8; 32] {
                observations
                    .entry(*producer)
                    .or_insert_with(|| DirectoryReplicaSyncObservation::new(*producer));
            }
        }
    }

    /// Restores audited retry boundaries before the coordinator starts.
    ///
    /// Process-lifetime attempt/page counters remain zero after restart; only
    /// the active failure streak, retry boundary, reason, and skip count are
    /// restored because those fields control request pressure.
    pub fn restore_retry_states(&self, states: &[DirectoryReplicaRetryState]) {
        let mut observations = self.observations.lock();
        for state in states {
            let observation = observations
                .entry(state.producer)
                .or_insert_with(|| DirectoryReplicaSyncObservation::new(state.producer));
            observation.last_attempt_at = Some(state.last_failure_at);
            observation.last_failure_at = Some(state.last_failure_at);
            observation.last_failure_reason = Some(state.last_failure_reason.clone());
            observation.retry_not_before = state.retry_not_before;
            observation.consecutive_failures = state.consecutive_failures;
            observation.backoff_skips = state.backoff_skips;
        }
        drop(observations);
    }

    /// Records the beginning of one bounded page request.
    pub fn record_attempt(&self, producer: [u8; 32], attempted_at: u64) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(attempted_at);
        observation.total_attempts = observation.total_attempts.saturating_add(1);
    }

    /// Records one authenticated page after its atomic import completes.
    #[allow(clippy::too_many_arguments)]
    pub fn record_success(
        &self,
        producer: [u8; 32],
        succeeded_at: u64,
        local_tip_height: u64,
        remote_tip_height: u64,
        has_more: bool,
        blocks_inserted: u64,
        commitments_inserted: u64,
        requests_sent: u32,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(succeeded_at);
        observation.last_success_at = Some(succeeded_at);
        observation.remote_tip_height = Some(remote_tip_height);
        observation.local_tip_height = local_tip_height;
        observation.has_more = has_more;
        observation.consecutive_failures = 0;
        observation.retry_not_before = None;
        observation.successful_pages = observation.successful_pages.saturating_add(1);
        observation.blocks_inserted = observation.blocks_inserted.saturating_add(blocks_inserted);
        observation.commitments_inserted = observation
            .commitments_inserted
            .saturating_add(commitments_inserted);
        observation.requests_sent = observation
            .requests_sent
            .saturating_add(u64::from(requests_sent));
    }

    /// Records one stable failure code without retaining peer endpoints,
    /// response bodies, or underlying transport error strings.
    pub fn record_failure(
        &self,
        producer: [u8; 32],
        failed_at: u64,
        reason: &str,
        retry_not_before: Option<u64>,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(failed_at);
        observation.last_failure_at = Some(failed_at);
        observation.last_failure_reason = Some(reason.chars().take(96).collect());
        observation.retry_not_before = retry_not_before.map(|value| value.max(failed_at));
        observation.consecutive_failures = observation
            .consecutive_failures
            .saturating_add(1)
            .min(DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES);
        observation.failed_attempts = observation.failed_attempts.saturating_add(1);
        drop(observations);
    }

    /// Returns the future retry boundary for one producer, if backoff is active.
    #[must_use]
    pub fn deferred_retry_until(&self, producer: &[u8; 32], now: u64) -> Option<u64> {
        self.observations
            .lock()
            .get(producer)
            .and_then(|observation| observation.retry_not_before)
            .filter(|retry_at| *retry_at > now)
    }

    /// Returns the current consecutive failure count for backoff calculation.
    #[must_use]
    pub fn consecutive_failures(&self, producer: &[u8; 32]) -> u64 {
        self.observations
            .lock()
            .get(producer)
            .map_or(0, |observation| observation.consecutive_failures)
    }

    /// Records one timer tick intentionally skipped by producer-local backoff.
    pub fn record_backoff_skip(&self, producer: [u8; 32]) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.backoff_skips = observation.backoff_skips.saturating_add(1);
        drop(observations);
    }

    /// Returns producer observations in deterministic identity order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DirectoryReplicaSyncObservation> {
        let mut observations = self
            .observations
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        observations.sort_by_key(|observation| observation.producer);
        observations
    }
}

#[derive(Debug)]
struct StoredReplicaBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    prev_block_hash: Vec<u8>,
    produced_at: i64,
    commitment_count: i64,
    block_blob: Vec<u8>,
}

#[derive(Debug)]
struct QuarantineIncident<'a> {
    kind: &'a str,
    height: u64,
    local_hash: [u8; 32],
    remote_hash: [u8; 32],
    evidence_frame: &'a [u8],
}

/// Durable producer-scoped replica namespace.
pub struct DirectoryReplicaStore {
    connection: Mutex<Connection>,
    path: PathBuf,
    local_node_id: [u8; 32],
}

impl DirectoryReplicaStore {
    /// Opens or creates replica tables and audits every accepted producer prefix.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for filesystem/SQLite failures,
    /// incompatible metadata, invalid signed blocks, malformed indexes, or
    /// invalid durable incident evidence.
    pub fn open(
        path: impl AsRef<Path>,
        local_node_id: [u8; 32],
        observed_at: u64,
    ) -> Result<(Self, DirectoryReplicaAudit), DirectoryReplicaStoreError> {
        if local_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "local node identity must not be the zero sentinel".to_string(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let mut connection = Connection::open(&path)?;
        connection.busy_timeout(DIRECTORY_REPLICA_BUSY_TIMEOUT)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        Self::initialize_schema(&mut connection, &local_node_id)?;
        let store = Self {
            connection: Mutex::new(connection),
            path,
            local_node_id,
        };
        let audit = store.audit(observed_at)?;
        Ok((store, audit))
    }

    /// Returns the shared Directory Chain `SQLite` path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns one producer's accepted prefix and quarantine state.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when a persisted row is malformed.
    pub fn producer_tip(
        &self,
        producer: &[u8; 32],
    ) -> Result<DirectoryReplicaTip, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::load_tip(&connection, producer)
    }

    /// Returns a low-cost aggregate snapshot of persisted, already-audited
    /// replica indexes.
    ///
    /// This is an observability read, not a replacement for [`Self::audit`].
    /// Startup still performs the full signature, linkage, object, index, and
    /// incident audit before synchronization or API serving begins.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when a persisted status row is
    /// malformed or `SQLite` cannot complete the bounded aggregate query.
    pub fn status_snapshot(
        &self,
    ) -> Result<DirectoryReplicaStoreSnapshot, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let mut statement = connection.prepare(
            "SELECT c.producer, c.tip_height, c.tip_timestamp, c.quarantined,
                    c.quarantine_kind, c.updated_at,
                    (SELECT COUNT(*) FROM directory_replica_blocks b
                     WHERE b.producer = c.producer),
                    (SELECT COUNT(*) FROM directory_replica_commitments m
                     WHERE m.producer = c.producer),
                    (SELECT COUNT(*) FROM directory_replica_incidents i
                     WHERE i.producer = c.producer)
             FROM directory_replica_chains c
             ORDER BY c.producer ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut snapshot = DirectoryReplicaStoreSnapshot::default();
        for (
            producer,
            tip_height,
            tip_timestamp,
            quarantined,
            quarantine_kind,
            updated_at,
            blocks,
            commitments,
            incidents,
        ) in rows
        {
            if quarantined != 0 && quarantined != 1 {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica status quarantine flag is invalid".to_string(),
                ));
            }
            let producer_snapshot = DirectoryReplicaProducerSnapshot {
                producer: bytes32(&producer, "replica status producer")?,
                tip_height: nonnegative_i64_to_u64(tip_height, "replica status tip height")?,
                tip_timestamp: nonnegative_i64_to_u64(
                    tip_timestamp,
                    "replica status tip timestamp",
                )?,
                quarantined: quarantined == 1,
                quarantine_kind,
                updated_at: nonnegative_i64_to_u64(updated_at, "replica status updated at")?,
                blocks: nonnegative_i64_to_u64(blocks, "replica status blocks")?,
                commitments: nonnegative_i64_to_u64(commitments, "replica status commitments")?,
                incidents: nonnegative_i64_to_u64(incidents, "replica status incidents")?,
            };
            snapshot.producers = snapshot.producers.saturating_add(1);
            snapshot.quarantined_producers = snapshot
                .quarantined_producers
                .saturating_add(u64::from(producer_snapshot.quarantined));
            snapshot.blocks = snapshot.blocks.saturating_add(producer_snapshot.blocks);
            snapshot.commitments = snapshot
                .commitments
                .saturating_add(producer_snapshot.commitments);
            snapshot.incidents = snapshot
                .incidents
                .saturating_add(producer_snapshot.incidents);
            snapshot.producer_snapshots.push(producer_snapshot);
        }
        Ok(snapshot)
    }

    /// Computes bounded recent commitment overlap for configured producers.
    ///
    /// Only non-empty, non-quarantined producer prefixes are eligible. The
    /// returned root binds the exact eligible producer tips and commitment
    /// hashes observed by every eligible source inside the recent block
    /// window. It is a local evidence digest, not a signature, vote, quorum,
    /// fork choice, consensus result, or finalized checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the producer set exceeds
    /// the configured protocol bound, contains invalid identities, persisted
    /// rows are malformed, or `SQLite` cannot complete a bounded query.
    pub fn observation_convergence(
        &self,
        configured_producers: &[[u8; 32]],
    ) -> Result<DirectoryReplicaObservationConvergenceSnapshot, DirectoryReplicaStoreError> {
        let configured = self.validate_convergence_producers(configured_producers)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let snapshot = Self::observation_convergence_from_connection(&connection, &configured);
        drop(connection);
        snapshot
    }

    fn validate_convergence_producers(
        &self,
        configured_producers: &[[u8; 32]],
    ) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        if configured_producers.len() > MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS {
            return Err(DirectoryReplicaStoreError::Request(format!(
                "observation convergence supports at most {MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS} producers"
            )));
        }
        let mut configured = configured_producers.to_vec();
        configured.sort_unstable();
        if configured.windows(2).any(|values| values[0] == values[1]) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation convergence producer identities must be unique".to_string(),
            ));
        }
        if configured
            .iter()
            .any(|producer| *producer == [0u8; 32] || producer == &self.local_node_id)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation convergence producer identity is invalid".to_string(),
            ));
        }
        Ok(configured)
    }

    fn observation_convergence_from_connection(
        connection: &Connection,
        configured: &[[u8; 32]],
    ) -> Result<DirectoryReplicaObservationConvergenceSnapshot, DirectoryReplicaStoreError> {
        let mut snapshot = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::try_from(configured.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation convergence producer count exceeds u64".to_string(),
                )
            })?,
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let eligible_tips = Self::eligible_convergence_tips(connection, configured, &mut snapshot)?;
        let occurrences =
            Self::recent_commitment_occurrences(connection, &eligible_tips, &mut snapshot)?;
        Self::complete_observation_convergence(&mut snapshot, &eligible_tips, &occurrences)?;
        Ok(snapshot)
    }

    fn eligible_convergence_tips(
        connection: &Connection,
        configured: &[[u8; 32]],
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
    ) -> Result<Vec<DirectoryReplicaTip>, DirectoryReplicaStoreError> {
        let mut eligible_tips = Vec::with_capacity(configured.len());
        for producer in configured {
            let tip = Self::load_tip(connection, producer)?;
            if tip.quarantined {
                snapshot.excluded_quarantined_producers =
                    snapshot.excluded_quarantined_producers.saturating_add(1);
            } else if tip.tip_height == 0 {
                snapshot.pending_producers = snapshot.pending_producers.saturating_add(1);
            } else {
                eligible_tips.push(tip);
            }
        }
        snapshot.eligible_producers = u64::try_from(eligible_tips.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation convergence eligible count exceeds u64".to_string(),
            )
        })?;
        Ok(eligible_tips)
    }

    fn recent_commitment_occurrences(
        connection: &Connection,
        eligible_tips: &[DirectoryReplicaTip],
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
    ) -> Result<BTreeMap<[u8; 32], u64>, DirectoryReplicaStoreError> {
        let mut occurrence_by_commitment = BTreeMap::<[u8; 32], u64>::new();
        for tip in eligible_tips {
            let first_height = tip
                .tip_height
                .saturating_sub(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.saturating_sub(1))
                .max(1);
            let first_height =
                u64_to_i64(first_height, "observation convergence first block height")?;
            let tip_height =
                u64_to_i64(tip.tip_height, "observation convergence tip block height")?;
            let mut statement = connection.prepare(
                "SELECT commitment_hash FROM directory_replica_commitments
                 WHERE producer = ?1 AND block_height BETWEEN ?2 AND ?3
                 ORDER BY commitment_hash ASC",
            )?;
            let hashes = statement
                .query_map(
                    params![tip.producer.as_slice(), first_height, tip_height],
                    |row| row.get::<_, Vec<u8>>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            let mut seen_for_producer = HashSet::with_capacity(hashes.len());
            for hash in hashes {
                let hash = bytes32(&hash, "observation convergence commitment hash")?;
                if !seen_for_producer.insert(hash) {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "observation convergence found a duplicate producer commitment".to_string(),
                    ));
                }
                snapshot.recent_commitments = snapshot.recent_commitments.saturating_add(1);
                let occurrence = occurrence_by_commitment.entry(hash).or_default();
                *occurrence = occurrence.saturating_add(1);
            }
        }
        Ok(occurrence_by_commitment)
    }

    fn complete_observation_convergence(
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
        eligible_tips: &[DirectoryReplicaTip],
        occurrence_by_commitment: &BTreeMap<[u8; 32], u64>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        snapshot.distinct_recent_commitments = u64::try_from(occurrence_by_commitment.len())
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation convergence commitment count exceeds u64".to_string(),
                )
            })?;
        if snapshot.eligible_producers >= 2 {
            snapshot.multi_source_recent_commitments = occurrence_by_commitment
                .values()
                .filter(|occurrence| **occurrence >= 2)
                .count()
                .try_into()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation convergence multi-source count exceeds u64".to_string(),
                    )
                })?;
            snapshot.all_eligible_source_recent_commitments = occurrence_by_commitment
                .values()
                .filter(|occurrence| **occurrence == snapshot.eligible_producers)
                .count()
                .try_into()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation convergence all-source count exceeds u64".to_string(),
                    )
                })?;
            snapshot.observation_root = Some(observation_convergence_root(
                eligible_tips,
                occurrence_by_commitment,
            ));
        }
        Ok(())
    }

    /// Returns all audited restart-durable producer retry states.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when metadata or any bounded
    /// retry-state row is malformed.
    pub fn retry_states(
        &self,
    ) -> Result<Vec<DirectoryReplicaRetryState>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        Self::load_retry_states(&connection, &self.local_node_id)
    }

    /// Persists one producer failure before exposing its retry boundary.
    ///
    /// The failure reason must be a stable ASCII bucket. Peer-controlled error
    /// text, endpoints, response bodies, and payloads are rejected.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid bounded state or a
    /// failed atomic `SQLite` transaction.
    pub fn persist_retry_failure(
        &self,
        producer: [u8; 32],
        consecutive_failures: u64,
        retry_not_before: Option<u64>,
        last_failure_at: u64,
        last_failure_reason: &str,
    ) -> Result<(), DirectoryReplicaStoreError> {
        validate_retry_state_fields(
            &producer,
            &self.local_node_id,
            consecutive_failures,
            retry_not_before,
            last_failure_at,
            last_failure_reason,
        )
        .map_err(|reason| DirectoryReplicaStoreError::Request(reason.to_string()))?;
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::ensure_producer_row(&transaction, &producer, last_failure_at)?;
        transaction.execute(
            "INSERT INTO directory_replica_retry_state
                (producer, consecutive_failures, retry_not_before,
                 last_failure_at, last_failure_reason, backoff_skips, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?4)
             ON CONFLICT(producer) DO UPDATE SET
                 consecutive_failures = excluded.consecutive_failures,
                 retry_not_before = CASE
                     WHEN excluded.last_failure_at >= directory_replica_retry_state.last_failure_at
                     THEN excluded.retry_not_before
                     ELSE directory_replica_retry_state.retry_not_before
                 END,
                 last_failure_at = MAX(
                     directory_replica_retry_state.last_failure_at,
                     excluded.last_failure_at
                 ),
                 last_failure_reason = CASE
                     WHEN excluded.last_failure_at >= directory_replica_retry_state.last_failure_at
                     THEN excluded.last_failure_reason
                     ELSE directory_replica_retry_state.last_failure_reason
                 END,
                 updated_at = MAX(
                     directory_replica_retry_state.updated_at,
                     excluded.updated_at
                 )",
            params![
                producer.as_slice(),
                u64_to_i64(consecutive_failures, "replica retry consecutive failures")?,
                retry_not_before
                    .map(|value| u64_to_i64(value, "replica retry boundary"))
                    .transpose()?,
                u64_to_i64(last_failure_at, "replica retry failure timestamp")?,
                last_failure_reason,
            ],
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(())
    }

    /// Persists one scheduled round skipped by an active retry boundary.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the expected active durable
    /// retry row is missing or `SQLite` cannot commit the update.
    pub fn persist_retry_skip(
        &self,
        producer: [u8; 32],
        skipped_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if producer == [0u8; 32] || producer == self.local_node_id || skipped_at == 0 {
            return Err(DirectoryReplicaStoreError::Request(
                "replica retry skip fields are invalid".to_string(),
            ));
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed = transaction.execute(
            "UPDATE directory_replica_retry_state
             SET backoff_skips = CASE
                     WHEN backoff_skips < 9223372036854775807
                     THEN backoff_skips + 1
                     ELSE backoff_skips
                 END,
                 updated_at = MAX(updated_at, ?2)
             WHERE producer = ?1
               AND retry_not_before IS NOT NULL
               AND retry_not_before > ?2",
            params![
                producer.as_slice(),
                u64_to_i64(skipped_at, "replica retry skip timestamp")?
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "active replica retry state is missing during skip".to_string(),
            ));
        }
        transaction.commit()?;
        drop(connection);
        Ok(())
    }

    /// Re-verifies and atomically imports one signed bounded producer page.
    ///
    /// The exact encoded `BlockRangeResponseV1` is required as durable fork
    /// evidence. Descriptor objects must exactly cover every commitment in the
    /// supplied page, without extras or duplicates.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid evidence, blocks,
    /// descriptor objects, chain gaps, storage errors, or durable quarantine.
    #[allow(clippy::too_many_arguments)]
    pub fn import_verified_page(
        &self,
        producer: [u8; 32],
        blocks: &[DirectoryCommitmentBlockV1],
        objects: &[SignedNodeDescriptor],
        advertised_tip_height: u64,
        advertised_tip_hash: [u8; 32],
        signed_response_frame: &[u8],
        observed_at: u64,
    ) -> Result<DirectoryReplicaImportReport, DirectoryReplicaStoreError> {
        if producer == [0u8; 32] || producer == self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "remote producer must be non-zero and differ from the local node".to_string(),
            ));
        }
        if blocks.len() > usize::from(MAX_DIRECTORY_SYNC_BLOCKS_V1) {
            return Err(DirectoryReplicaStoreError::Request(
                "block page exceeds the Directory Sync V1 bound".to_string(),
            ));
        }
        if signed_response_frame.is_empty()
            || signed_response_frame.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES
        {
            return Err(DirectoryReplicaStoreError::Request(
                "signed response evidence is empty or oversized".to_string(),
            ));
        }
        let has_more = verify_range_response_evidence(
            signed_response_frame,
            &producer,
            blocks,
            advertised_tip_height,
            &advertised_tip_hash,
            observed_at,
        )?;
        validate_page_tip_contract(
            blocks,
            has_more,
            advertised_tip_height,
            &advertised_tip_hash,
        )?;
        let descriptors = validate_exact_descriptor_objects(blocks, objects)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::ensure_producer_row(&transaction, &producer, observed_at)?;
        let mut tip = Self::load_tip(&transaction, &producer)?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }

        if advertised_tip_height < tip.tip_height {
            let incident = QuarantineIncident {
                kind: "signed_tip_rollback",
                height: advertised_tip_height,
                local_hash: tip.tip_hash,
                remote_hash: advertised_tip_hash,
                evidence_frame: signed_response_frame,
            };
            Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
            transaction.commit()?;
            return Err(DirectoryReplicaStoreError::Quarantined(
                incident.kind.to_string(),
            ));
        }
        if advertised_tip_height == tip.tip_height && advertised_tip_hash != tip.tip_hash {
            let incident = QuarantineIncident {
                kind: "signed_tip_fork",
                height: advertised_tip_height,
                local_hash: tip.tip_hash,
                remote_hash: advertised_tip_hash,
                evidence_frame: signed_response_frame,
            };
            Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
            transaction.commit()?;
            return Err(DirectoryReplicaStoreError::Quarantined(
                incident.kind.to_string(),
            ));
        }
        if blocks.is_empty() {
            if advertised_tip_height > tip.tip_height {
                let incident = QuarantineIncident {
                    kind: "signed_empty_range_gap",
                    height: tip.tip_height.saturating_add(1),
                    local_hash: tip.tip_hash,
                    remote_hash: advertised_tip_hash,
                    evidence_frame: signed_response_frame,
                };
                Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                transaction.commit()?;
                return Err(DirectoryReplicaStoreError::Quarantined(
                    incident.kind.to_string(),
                ));
            }
            Self::clear_retry_state(&transaction, &producer)?;
            transaction.commit()?;
            return Ok(DirectoryReplicaImportReport {
                blocks_inserted: 0,
                blocks_already_present: 0,
                commitments_inserted: 0,
                descriptor_equivocations: 0,
                tip_height: tip.tip_height,
                tip_hash: tip.tip_hash,
            });
        }

        let mut blocks_inserted = 0u64;
        let mut blocks_already_present = 0u64;
        let mut commitments_inserted = 0u64;
        let mut descriptor_equivocations = 0u64;
        for block in blocks {
            if block.header.producer != producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range contains a block signed for another producer".to_string(),
                ));
            }
            let existing_hash = Self::block_hash_at(&transaction, &producer, block.header.height)?;
            if let Some(existing_hash) = existing_hash {
                if existing_hash != block.hash() {
                    let incident = QuarantineIncident {
                        kind: "signed_block_fork",
                        height: block.header.height,
                        local_hash: existing_hash,
                        remote_hash: block.hash(),
                        evidence_frame: signed_response_frame,
                    };
                    Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                    transaction.commit()?;
                    return Err(DirectoryReplicaStoreError::Quarantined(
                        incident.kind.to_string(),
                    ));
                }
                blocks_already_present = blocks_already_present.saturating_add(1);
                continue;
            }
            let expected_height = tip.tip_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity("replica chain height exhausted".to_string())
            })?;
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &tip.tip_hash,
                tip.tip_timestamp,
                observed_at,
            )?;
            let block_objects = block
                .commitments
                .iter()
                .map(|commitment| {
                    descriptors
                        .get(&commitment.descriptor_hash)
                        .copied()
                        .ok_or_else(|| {
                            DirectoryReplicaStoreError::Integrity(
                                "validated descriptor object map became incomplete".to_string(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (inserted, equivocations) = Self::insert_block(
                &transaction,
                &producer,
                block,
                &block_objects,
                signed_response_frame,
                observed_at,
            )?;
            commitments_inserted = commitments_inserted.saturating_add(inserted);
            descriptor_equivocations = descriptor_equivocations.saturating_add(equivocations);
            blocks_inserted = blocks_inserted.saturating_add(1);
            tip.tip_height = block.header.height;
            tip.tip_hash = block.hash();
            tip.tip_timestamp = block.header.timestamp;
        }
        transaction.execute(
            "UPDATE directory_replica_chains
             SET tip_height = ?2, tip_hash = ?3, tip_timestamp = ?4, updated_at = ?5
             WHERE producer = ?1",
            params![
                producer.as_slice(),
                u64_to_i64(tip.tip_height, "replica tip height")?,
                tip.tip_hash.as_slice(),
                u64_to_i64(tip.tip_timestamp, "replica tip timestamp")?,
                u64_to_i64(observed_at, "replica update timestamp")?
            ],
        )?;
        Self::clear_retry_state(&transaction, &producer)?;
        transaction.commit()?;
        Ok(DirectoryReplicaImportReport {
            blocks_inserted,
            blocks_already_present,
            commitments_inserted,
            descriptor_equivocations,
            tip_height: tip.tip_height,
            tip_hash: tip.tip_hash,
        })
    }

    /// Audits metadata, producer prefixes, indexes, incidents, and retry rows.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] on the first malformed row,
    /// invalid signature/link/root, missing object/index, or invalid incident.
    pub fn audit(
        &self,
        observed_at: u64,
    ) -> Result<DirectoryReplicaAudit, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let producers = Self::load_all_tips(&connection)?;
        let mut report = DirectoryReplicaAudit::default();
        for tip in producers {
            Self::audit_producer(&connection, &tip, observed_at, &mut report)?;
        }
        report.retry_states =
            u64::try_from(Self::load_retry_states(&connection, &self.local_node_id)?.len())
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "replica retry state count exceeds platform bounds".to_string(),
                    )
                })?;
        let incidents = Self::audit_incidents(&connection)?;
        report.incidents = incidents;
        Ok(report)
    }

    fn initialize_schema(
        connection: &mut Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::create_schema_tables(&transaction)?;
        Self::migrate_schema_metadata(&transaction, local_node_id)?;
        transaction.commit()?;
        Self::validate_metadata(connection, local_node_id)
    }

    fn create_schema_tables(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute_batch(
            "CREATE TABLE IF NOT EXISTS directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32)
             );
             CREATE TABLE IF NOT EXISTS directory_replica_chains (
                 producer BLOB PRIMARY KEY CHECK (length(producer) = 32),
                 tip_height INTEGER NOT NULL CHECK (tip_height >= 0),
                 tip_hash BLOB NOT NULL CHECK (length(tip_hash) = 32),
                 tip_timestamp INTEGER NOT NULL CHECK (tip_timestamp >= 0),
                 quarantined INTEGER NOT NULL CHECK (quarantined IN (0, 1)),
                 quarantine_kind TEXT,
                 updated_at INTEGER NOT NULL CHECK (updated_at > 0)
             );
             CREATE TABLE IF NOT EXISTS directory_replica_blocks (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 height INTEGER NOT NULL CHECK (height > 0),
                 block_hash BLOB NOT NULL CHECK (length(block_hash) = 32),
                 prev_block_hash BLOB NOT NULL CHECK (length(prev_block_hash) = 32),
                 produced_at INTEGER NOT NULL CHECK (produced_at > 0),
                 commitment_count INTEGER NOT NULL CHECK (commitment_count > 0),
                 block_blob BLOB NOT NULL,
                 PRIMARY KEY (producer, height),
                 UNIQUE (producer, block_hash),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_descriptor_objects (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 descriptor_hash BLOB NOT NULL CHECK (length(descriptor_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_blob BLOB NOT NULL,
                 PRIMARY KEY (producer, descriptor_hash),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_commitments (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 commitment_hash BLOB NOT NULL CHECK (length(commitment_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_hash BLOB NOT NULL CHECK (length(descriptor_hash) = 32),
                 block_height INTEGER NOT NULL CHECK (block_height > 0),
                 PRIMARY KEY (producer, commitment_hash),
                 UNIQUE (producer, descriptor_hash),
                 FOREIGN KEY (producer, block_height)
                     REFERENCES directory_replica_blocks(producer, height)
                     ON UPDATE RESTRICT ON DELETE RESTRICT,
                 FOREIGN KEY (producer, descriptor_hash)
                     REFERENCES directory_replica_descriptor_objects(producer, descriptor_hash)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_replica_commitments_by_block
                 ON directory_replica_commitments(producer, block_height, commitment_hash);
             CREATE INDEX IF NOT EXISTS directory_replica_commitments_by_subject
                 ON directory_replica_commitments(producer, node_id, sequence_le);
             CREATE TABLE IF NOT EXISTS directory_replica_incidents (
                 incident_digest BLOB PRIMARY KEY CHECK (length(incident_digest) = 32),
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 subject_node_id BLOB NOT NULL CHECK (length(subject_node_id) = 32),
                 kind TEXT NOT NULL,
                 height INTEGER NOT NULL CHECK (height >= 0),
                 local_hash BLOB NOT NULL CHECK (length(local_hash) = 32),
                 remote_hash BLOB NOT NULL CHECK (length(remote_hash) = 32),
                 evidence_frame BLOB NOT NULL,
                 observed_at INTEGER NOT NULL CHECK (observed_at > 0),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_retry_state (
                 producer BLOB PRIMARY KEY CHECK (length(producer) = 32),
                 consecutive_failures INTEGER NOT NULL
                     CHECK (consecutive_failures > 0 AND consecutive_failures <= 64),
                 retry_not_before INTEGER
                     CHECK (retry_not_before IS NULL OR retry_not_before > 0),
                 last_failure_at INTEGER NOT NULL CHECK (last_failure_at > 0),
                 last_failure_reason TEXT NOT NULL
                     CHECK (length(last_failure_reason) BETWEEN 1 AND 96),
                 backoff_skips INTEGER NOT NULL CHECK (backoff_skips >= 0),
                 updated_at INTEGER NOT NULL CHECK (updated_at > 0),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );",
        )?;
        Ok(())
    }

    fn migrate_schema_metadata(
        transaction: &Transaction<'_>,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let existing: Option<(i64, Vec<u8>, Vec<u8>)> = transaction
            .query_row(
                "SELECT schema_version, chain_id, local_node_id
                 FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        match existing {
            None => {
                transaction.execute(
                    "INSERT INTO directory_replica_meta
                        (singleton, schema_version, chain_id, local_node_id)
                     VALUES (1, ?1, ?2, ?3)",
                    params![
                        DIRECTORY_REPLICA_SCHEMA_VERSION,
                        AERONYX_DIRECTORY_MAINNET_CHAIN_ID.as_slice(),
                        local_node_id.as_slice()
                    ],
                )?;
            }
            Some((version, chain_id, stored_local_node_id)) => {
                if bytes32(&chain_id, "replica metadata chain id")?
                    != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                    || bytes32(&stored_local_node_id, "replica metadata local node id")?
                        != *local_node_id
                {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "directory replica metadata identity is incompatible".to_string(),
                    ));
                }
                match version {
                    DIRECTORY_REPLICA_SCHEMA_VERSION => {}
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V1 => {
                        transaction.execute(
                            "UPDATE directory_replica_meta
                             SET schema_version = ?1
                             WHERE singleton = 1 AND schema_version = ?2",
                            params![
                                DIRECTORY_REPLICA_SCHEMA_VERSION,
                                DIRECTORY_REPLICA_SCHEMA_VERSION_V1
                            ],
                        )?;
                    }
                    _ => {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "directory replica schema version is unsupported".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_metadata(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let metadata = connection
            .query_row(
                "SELECT schema_version, chain_id, local_node_id
                 FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica metadata row is missing".to_string(),
                )
            })?;
        if metadata.0 != DIRECTORY_REPLICA_SCHEMA_VERSION
            || bytes32(&metadata.1, "replica metadata chain id")?
                != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || bytes32(&metadata.2, "replica metadata local node id")? != *local_node_id
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica metadata does not match this node and Directory Sync V1"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn ensure_producer_row(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "INSERT OR IGNORE INTO directory_replica_chains
                (producer, tip_height, tip_hash, tip_timestamp,
                 quarantined, quarantine_kind, updated_at)
             VALUES (?1, 0, ?2, 0, 0, NULL, ?3)",
            params![
                producer.as_slice(),
                [0u8; 32].as_slice(),
                u64_to_i64(observed_at, "replica observed timestamp")?
            ],
        )?;
        Ok(())
    }

    fn clear_retry_state(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "DELETE FROM directory_replica_retry_state WHERE producer = ?1",
            params![producer.as_slice()],
        )?;
        Ok(())
    }

    fn load_tip(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<DirectoryReplicaTip, DirectoryReplicaStoreError> {
        let row = connection
            .query_row(
                "SELECT tip_height, tip_hash, tip_timestamp, quarantined, quarantine_kind
                 FROM directory_replica_chains WHERE producer = ?1",
                params![producer.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((height, hash, timestamp, quarantined, quarantine_kind)) = row else {
            return Ok(DirectoryReplicaTip::empty(*producer));
        };
        if quarantined != 0 && quarantined != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica quarantine flag is invalid".to_string(),
            ));
        }
        if quarantined == 1 && quarantine_kind.as_deref().unwrap_or_default().is_empty() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "quarantined producer is missing its incident kind".to_string(),
            ));
        }
        Ok(DirectoryReplicaTip {
            producer: *producer,
            tip_height: nonnegative_i64_to_u64(height, "replica tip height")?,
            tip_hash: bytes32(&hash, "replica tip hash")?,
            tip_timestamp: nonnegative_i64_to_u64(timestamp, "replica tip timestamp")?,
            quarantined: quarantined == 1,
            quarantine_kind,
        })
    }

    fn load_all_tips(
        connection: &Connection,
    ) -> Result<Vec<DirectoryReplicaTip>, DirectoryReplicaStoreError> {
        let mut statement = connection
            .prepare("SELECT producer FROM directory_replica_chains ORDER BY producer ASC")?;
        let producers = statement
            .query_map([], |row| row.get::<_, Vec<u8>>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        producers
            .into_iter()
            .map(|producer| {
                let producer = bytes32(&producer, "replica producer")?;
                Self::load_tip(connection, &producer)
            })
            .collect()
    }

    fn load_retry_states(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<Vec<DirectoryReplicaRetryState>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT producer, consecutive_failures, retry_not_before,
                    last_failure_at, last_failure_reason, backoff_skips, updated_at
             FROM directory_replica_retry_state
             ORDER BY producer ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(
                |(
                    producer,
                    consecutive_failures,
                    retry_not_before,
                    last_failure_at,
                    last_failure_reason,
                    backoff_skips,
                    updated_at,
                )| {
                    let producer = bytes32(&producer, "replica retry producer")?;
                    let consecutive_failures = nonnegative_i64_to_u64(
                        consecutive_failures,
                        "replica retry consecutive failures",
                    )?;
                    let retry_not_before = retry_not_before
                        .map(|value| nonnegative_i64_to_u64(value, "replica retry boundary"))
                        .transpose()?;
                    let last_failure_at =
                        nonnegative_i64_to_u64(last_failure_at, "replica retry failure timestamp")?;
                    validate_retry_state_fields(
                        &producer,
                        local_node_id,
                        consecutive_failures,
                        retry_not_before,
                        last_failure_at,
                        &last_failure_reason,
                    )
                    .map_err(|reason| DirectoryReplicaStoreError::Integrity(reason.to_string()))?;
                    let updated_at =
                        nonnegative_i64_to_u64(updated_at, "replica retry update timestamp")?;
                    if updated_at < last_failure_at {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "replica retry update timestamp predates failure".to_string(),
                        ));
                    }
                    Ok(DirectoryReplicaRetryState {
                        producer,
                        consecutive_failures,
                        retry_not_before,
                        last_failure_at,
                        last_failure_reason,
                        backoff_skips: nonnegative_i64_to_u64(
                            backoff_skips,
                            "replica retry skip count",
                        )?,
                    })
                },
            )
            .collect()
    }

    fn block_hash_at(
        connection: &Connection,
        producer: &[u8; 32],
        height: u64,
    ) -> Result<Option<[u8; 32]>, DirectoryReplicaStoreError> {
        let value = connection
            .query_row(
                "SELECT block_hash FROM directory_replica_blocks
                 WHERE producer = ?1 AND height = ?2",
                params![
                    producer.as_slice(),
                    u64_to_i64(height, "replica block height")?
                ],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        value
            .as_deref()
            .map(|bytes| bytes32(bytes, "replica block hash"))
            .transpose()
    }

    fn insert_block(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        block: &DirectoryCommitmentBlockV1,
        descriptors: &[&SignedNodeDescriptor],
        evidence_frame: &[u8],
        observed_at: u64,
    ) -> Result<(u64, u64), DirectoryReplicaStoreError> {
        if descriptors.len() != block.commitments.len() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "descriptor count does not match block commitments".to_string(),
            ));
        }
        let height = u64_to_i64(block.header.height, "replica block height")?;
        transaction.execute(
            "INSERT INTO directory_replica_blocks
                (producer, height, block_hash, prev_block_hash, produced_at,
                 commitment_count, block_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                producer.as_slice(),
                height,
                block.hash().as_slice(),
                block.header.prev_block_hash.as_slice(),
                u64_to_i64(block.header.timestamp, "replica block timestamp")?,
                i64::from(block.header.commitment_count),
                encode_block(block)?
            ],
        )?;
        let mut inserted = 0u64;
        let mut equivocations = 0u64;
        for (commitment, descriptor) in block.commitments.iter().zip(descriptors) {
            let derived = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            if derived != *commitment {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "descriptor object does not match its block commitment".to_string(),
                ));
            }
            let conflicting = transaction
                .query_row(
                    "SELECT descriptor_hash FROM directory_replica_commitments
                     WHERE producer = ?1 AND node_id = ?2 AND sequence_le = ?3
                       AND descriptor_hash != ?4 LIMIT 1",
                    params![
                        producer.as_slice(),
                        commitment.node_id.as_slice(),
                        commitment.sequence.to_le_bytes().as_slice(),
                        commitment.descriptor_hash.as_slice()
                    ],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .optional()?;
            if let Some(conflicting) = conflicting {
                let conflicting = bytes32(&conflicting, "equivocation descriptor hash")?;
                let incident = QuarantineIncident {
                    kind: "descriptor_sequence_equivocation",
                    height: block.header.height,
                    local_hash: conflicting,
                    remote_hash: commitment.descriptor_hash,
                    evidence_frame,
                };
                if Self::insert_incident(
                    transaction,
                    producer,
                    &commitment.node_id,
                    &incident,
                    observed_at,
                )? {
                    equivocations = equivocations.saturating_add(1);
                }
            }
            transaction.execute(
                "INSERT INTO directory_replica_descriptor_objects
                    (producer, descriptor_hash, node_id, sequence_le, descriptor_blob)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    producer.as_slice(),
                    commitment.descriptor_hash.as_slice(),
                    commitment.node_id.as_slice(),
                    commitment.sequence.to_le_bytes().as_slice(),
                    encode_descriptor_object(descriptor)?
                ],
            )?;
            transaction.execute(
                "INSERT INTO directory_replica_commitments
                    (producer, commitment_hash, node_id, sequence_le,
                     descriptor_hash, block_height)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    producer.as_slice(),
                    commitment.hash().as_slice(),
                    commitment.node_id.as_slice(),
                    commitment.sequence.to_le_bytes().as_slice(),
                    commitment.descriptor_hash.as_slice(),
                    height
                ],
            )?;
            inserted = inserted.saturating_add(1);
        }
        Ok((inserted, equivocations))
    }

    fn persist_quarantine(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        incident: &QuarantineIncident<'_>,
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        Self::insert_incident(transaction, producer, producer, incident, observed_at)?;
        transaction.execute(
            "UPDATE directory_replica_chains
             SET quarantined = 1, quarantine_kind = ?2, updated_at = ?3
             WHERE producer = ?1",
            params![
                producer.as_slice(),
                incident.kind,
                u64_to_i64(observed_at, "quarantine timestamp")?
            ],
        )?;
        Ok(())
    }

    fn insert_incident(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        subject_node_id: &[u8; 32],
        incident: &QuarantineIncident<'_>,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let digest = incident_digest(producer, subject_node_id, incident);
        let changed = transaction.execute(
            "INSERT OR IGNORE INTO directory_replica_incidents
                (incident_digest, producer, subject_node_id, kind, height,
                 local_hash, remote_hash, evidence_frame, observed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                digest.as_slice(),
                producer.as_slice(),
                subject_node_id.as_slice(),
                incident.kind,
                u64_to_i64(incident.height, "incident height")?,
                incident.local_hash.as_slice(),
                incident.remote_hash.as_slice(),
                incident.evidence_frame,
                u64_to_i64(observed_at, "incident timestamp")?
            ],
        )?;
        Ok(changed == 1)
    }

    fn audit_producer(
        connection: &Connection,
        tip: &DirectoryReplicaTip,
        observed_at: u64,
        report: &mut DirectoryReplicaAudit,
    ) -> Result<(), DirectoryReplicaStoreError> {
        report.producers = report.producers.saturating_add(1);
        if tip.quarantined {
            report.quarantined_producers = report.quarantined_producers.saturating_add(1);
            let incident_exists = connection
                .query_row(
                    "SELECT 1 FROM directory_replica_incidents
                     WHERE producer = ?1 AND subject_node_id = ?1 AND kind = ?2 LIMIT 1",
                    params![
                        tip.producer.as_slice(),
                        tip.quarantine_kind.as_deref().unwrap_or_default()
                    ],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if !incident_exists {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "quarantined producer is missing matching signed incident evidence".to_string(),
                ));
            }
        }
        let rows = Self::load_block_rows(connection, &tip.producer)?;
        let mut commitments = Self::load_commitment_index(connection, &tip.producer)?;
        let mut objects = Self::load_descriptor_objects(connection, &tip.producer)?;
        let mut expected_height = 1u64;
        let mut previous_hash = [0u8; 32];
        let mut previous_timestamp = 0u64;
        for row in rows {
            let block = decode_block(&row.block_blob)?;
            let height = positive_i64_to_u64(row.height, "replica block height")?;
            if block.header.producer != tip.producer
                || height != expected_height
                || height != block.header.height
                || bytes32(&row.block_hash, "stored replica block hash")? != block.hash()
                || bytes32(&row.prev_block_hash, "stored replica previous hash")?
                    != block.header.prev_block_hash
                || positive_i64_to_u64(row.produced_at, "replica produced timestamp")?
                    != block.header.timestamp
                || nonnegative_i64_to_u64(row.commitment_count, "replica commitment count")?
                    != u64::from(block.header.commitment_count)
            {
                return Err(DirectoryReplicaStoreError::Integrity(format!(
                    "replica block {height} columns do not match its signed object"
                )));
            }
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &previous_hash,
                previous_timestamp,
                observed_at,
            )?;
            let mut actual = commitments.remove(&row.height).unwrap_or_default();
            actual.sort_unstable();
            if actual != block.commitments {
                return Err(DirectoryReplicaStoreError::Integrity(format!(
                    "replica block {height} commitment index mismatch"
                )));
            }
            for commitment in &block.commitments {
                let actual = objects.remove(&commitment.descriptor_hash).ok_or_else(|| {
                    DirectoryReplicaStoreError::Integrity(format!(
                        "replica block {height} is missing a descriptor object"
                    ))
                })?;
                if actual != *commitment {
                    return Err(DirectoryReplicaStoreError::Integrity(format!(
                        "replica block {height} descriptor object mismatch"
                    )));
                }
            }
            report.blocks = report.blocks.saturating_add(1);
            report.commitments = report
                .commitments
                .saturating_add(u64::from(block.header.commitment_count));
            previous_hash = block.hash();
            previous_timestamp = block.header.timestamp;
            expected_height = expected_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity("replica height exhausted".to_string())
            })?;
        }
        if !commitments.is_empty() || !objects.is_empty() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica contains orphaned commitment or descriptor indexes".to_string(),
            ));
        }
        let audited_height = expected_height.saturating_sub(1);
        if tip.tip_height != audited_height
            || tip.tip_hash != previous_hash
            || tip.tip_timestamp != previous_timestamp
            || (tip.tip_height == 0 && tip.tip_hash != [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica producer tip does not match its accepted block prefix".to_string(),
            ));
        }
        Ok(())
    }

    fn load_block_rows(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<Vec<StoredReplicaBlockRow>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT height, block_hash, prev_block_hash, produced_at,
                    commitment_count, block_blob
             FROM directory_replica_blocks WHERE producer = ?1 ORDER BY height ASC",
        )?;
        let rows = statement
            .query_map(params![producer.as_slice()], |row| {
                Ok(StoredReplicaBlockRow {
                    height: row.get(0)?,
                    block_hash: row.get(1)?,
                    prev_block_hash: row.get(2)?,
                    produced_at: row.get(3)?,
                    commitment_count: row.get(4)?,
                    block_blob: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DirectoryReplicaStoreError::from)?;
        Ok(rows)
    }

    fn load_commitment_index(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<BTreeMap<i64, Vec<DirectoryDescriptorCommitmentV1>>, DirectoryReplicaStoreError>
    {
        let mut statement = connection.prepare(
            "SELECT block_height, commitment_hash, node_id, sequence_le, descriptor_hash
             FROM directory_replica_commitments WHERE producer = ?1
             ORDER BY block_height ASC, commitment_hash ASC",
        )?;
        let rows = statement.query_map(params![producer.as_slice()], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;
        let mut index = BTreeMap::new();
        for row in rows {
            let (height, hash, node_id, sequence, descriptor_hash) = row?;
            let sequence: [u8; 8] = sequence.try_into().map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "replica commitment sequence must contain 8 bytes".to_string(),
                )
            })?;
            let commitment = DirectoryDescriptorCommitmentV1 {
                node_id: bytes32(&node_id, "replica commitment node id")?,
                sequence: u64::from_le_bytes(sequence),
                descriptor_hash: bytes32(&descriptor_hash, "replica commitment descriptor hash")?,
            };
            if bytes32(&hash, "replica commitment hash")? != commitment.hash() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica commitment content hash mismatch".to_string(),
                ));
            }
            index
                .entry(height)
                .or_insert_with(Vec::new)
                .push(commitment);
        }
        Ok(index)
    }

    fn load_descriptor_objects(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<HashMap<[u8; 32], DirectoryDescriptorCommitmentV1>, DirectoryReplicaStoreError>
    {
        let mut statement = connection.prepare(
            "SELECT descriptor_hash, node_id, sequence_le, descriptor_blob
             FROM directory_replica_descriptor_objects WHERE producer = ?1",
        )?;
        let rows = statement.query_map(params![producer.as_slice()], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })?;
        let mut objects = HashMap::new();
        for row in rows {
            let (hash, node_id, sequence, blob) = row?;
            let descriptor = decode_descriptor_object(&blob)?;
            let commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                    .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            let sequence: [u8; 8] = sequence.try_into().map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "replica descriptor sequence must contain 8 bytes".to_string(),
                )
            })?;
            let hash = bytes32(&hash, "replica descriptor hash")?;
            if hash != commitment.descriptor_hash
                || bytes32(&node_id, "replica descriptor node id")? != commitment.node_id
                || u64::from_le_bytes(sequence) != commitment.sequence
                || objects.insert(hash, commitment).is_some()
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica descriptor object index mismatch".to_string(),
                ));
            }
        }
        Ok(objects)
    }

    fn audit_incidents(connection: &Connection) -> Result<u64, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT incident_digest, producer, subject_node_id, kind, height,
                    local_hash, remote_hash, evidence_frame
             FROM directory_replica_incidents ORDER BY incident_digest ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Vec<u8>>(5)?,
                row.get::<_, Vec<u8>>(6)?,
                row.get::<_, Vec<u8>>(7)?,
            ))
        })?;
        let mut count = 0u64;
        for row in rows {
            let (digest, producer, subject, kind, height, local, remote, evidence) = row?;
            if kind.is_empty()
                || evidence.is_empty()
                || evidence.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica incident metadata or evidence is invalid".to_string(),
                ));
            }
            let incident = QuarantineIncident {
                kind: &kind,
                height: nonnegative_i64_to_u64(height, "incident height")?,
                local_hash: bytes32(&local, "incident local hash")?,
                remote_hash: bytes32(&remote, "incident remote hash")?,
                evidence_frame: &evidence,
            };
            let producer = bytes32(&producer, "incident producer")?;
            let subject = bytes32(&subject, "incident subject")?;
            if bytes32(&digest, "incident digest")?
                != incident_digest(&producer, &subject, &incident)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica incident digest mismatch".to_string(),
                ));
            }
            verify_incident_response_evidence(&evidence, &producer)?;
            count = count.saturating_add(1);
        }
        Ok(count)
    }
}

fn observation_convergence_root(
    eligible_tips: &[DirectoryReplicaTip],
    occurrence_by_commitment: &BTreeMap<[u8; 32], u64>,
) -> [u8; 32] {
    debug_assert!(eligible_tips.len() >= 2);
    let eligible_count = eligible_tips.len() as u64;
    let common_count = occurrence_by_commitment
        .values()
        .filter(|occurrence| **occurrence == eligible_count)
        .count() as u64;
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaObservationConvergence-v1");
    hasher.update(AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
    hasher.update(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.to_le_bytes());
    hasher.update(eligible_count.to_le_bytes());
    for tip in eligible_tips {
        hasher.update(tip.producer);
        hasher.update(tip.tip_height.to_le_bytes());
        hasher.update(tip.tip_hash);
    }
    hasher.update(common_count.to_le_bytes());
    for (commitment, occurrence) in occurrence_by_commitment {
        if *occurrence == eligible_count {
            hasher.update(commitment);
        }
    }
    hasher.finalize().into()
}

fn verify_incident_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    let message = decode_directory_sync_message(frame)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
    if encode_directory_sync_message(&message)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
        != frame
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence frame is not canonical".to_string(),
        ));
    }
    let DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence is not a block-range response".to_string(),
        ));
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID || responder != *expected_producer {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence belongs to another chain or producer".to_string(),
        ));
    }
    let signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "incident evidence producer identity is invalid".to_string(),
            )
        })?
        .verify(&signing_bytes, &signature)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "incident evidence producer signature is invalid".to_string(),
            )
        })
}

fn verify_range_response_evidence(
    frame: &[u8],
    producer: &[u8; 32],
    expected_blocks: &[DirectoryCommitmentBlockV1],
    expected_tip_height: u64,
    expected_tip_hash: &[u8; 32],
    observed_at: u64,
) -> Result<bool, DirectoryReplicaStoreError> {
    let message = decode_directory_sync_message(frame)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
    let DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err(DirectoryReplicaStoreError::Request(
            "evidence is not a block-range response".to_string(),
        ));
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || responder != *producer
        || blocks != expected_blocks
        || tip_height != expected_tip_height
        || tip_hash != *expected_tip_hash
        || response_timestamp.abs_diff(observed_at) > RESPONSE_TIMESTAMP_SKEW_SECS
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "signed range evidence does not match the import".to_string(),
        ));
    }
    let signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .map_err(|_| DirectoryReplicaStoreError::Integrity("invalid response signer".to_string()))?
        .verify(&signing_bytes, &signature)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "invalid block-range response signature".to_string(),
            )
        })?;
    Ok(has_more)
}

fn validate_page_tip_contract(
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    if tip_height == 0 && *tip_hash != [0u8; 32] {
        return Err(DirectoryReplicaStoreError::Integrity(
            "empty advertised tip must use the zero hash".to_string(),
        ));
    }
    let Some(last) = blocks.last() else {
        if has_more {
            return Err(DirectoryReplicaStoreError::Integrity(
                "an empty response cannot advertise more pages".to_string(),
            ));
        }
        return Ok(());
    };
    if last.header.height > tip_height
        || (has_more && last.header.height >= tip_height)
        || (!has_more && (last.header.height != tip_height || last.hash() != *tip_hash))
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "range pagination fields contradict the signed tip".to_string(),
        ));
    }
    Ok(())
}

fn validate_exact_descriptor_objects<'a>(
    blocks: &[DirectoryCommitmentBlockV1],
    objects: &'a [SignedNodeDescriptor],
) -> Result<HashMap<[u8; 32], &'a SignedNodeDescriptor>, DirectoryReplicaStoreError> {
    let required = blocks
        .iter()
        .flat_map(|block| block.commitments.iter().map(|entry| entry.descriptor_hash))
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();
    if required_set.len() != required.len() || objects.len() != required.len() {
        return Err(DirectoryReplicaStoreError::Request(
            "descriptor objects must exactly cover unique page commitments".to_string(),
        ));
    }
    let mut mapped = HashMap::with_capacity(objects.len());
    for descriptor in objects {
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
        if !required_set.contains(&commitment.descriptor_hash)
            || mapped
                .insert(commitment.descriptor_hash, descriptor)
                .is_some()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "descriptor response contains an extra or duplicate object".to_string(),
            ));
        }
    }
    Ok(mapped)
}

fn incident_digest(
    producer: &[u8; 32],
    subject_node_id: &[u8; 32],
    incident: &QuarantineIncident<'_>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaIncident-v1");
    hasher.update(producer);
    hasher.update(subject_node_id);
    hasher.update((incident.kind.len() as u64).to_le_bytes());
    hasher.update(incident.kind.as_bytes());
    hasher.update(incident.height.to_le_bytes());
    hasher.update(incident.local_hash);
    hasher.update(incident.remote_hash);
    hasher.update((incident.evidence_frame.len() as u64).to_le_bytes());
    hasher.update(incident.evidence_frame);
    hasher.finalize().into()
}

fn encode_block(block: &DirectoryCommitmentBlockV1) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .serialize(block)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_block(bytes: &[u8]) -> Result<DirectoryCommitmentBlockV1, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_DIRECTORY_BLOCK_BYTES) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica block exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn encode_descriptor_object(
    descriptor: &SignedNodeDescriptor,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .serialize(descriptor)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_descriptor_object(
    bytes: &[u8],
) -> Result<SignedNodeDescriptor, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length > MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES
    }) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica descriptor object exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn validate_retry_state_fields(
    producer: &[u8; 32],
    local_node_id: &[u8; 32],
    consecutive_failures: u64,
    retry_not_before: Option<u64>,
    last_failure_at: u64,
    last_failure_reason: &str,
) -> Result<(), &'static str> {
    if *producer == [0u8; 32] || producer == local_node_id {
        return Err("replica retry producer is invalid");
    }
    if consecutive_failures == 0
        || consecutive_failures > DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
    {
        return Err("replica retry failure count is invalid");
    }
    if last_failure_at == 0 {
        return Err("replica retry failure timestamp is invalid");
    }
    if retry_not_before.is_some_and(|retry_at| {
        retry_at < last_failure_at
            || retry_at.saturating_sub(last_failure_at) > DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS
    }) {
        return Err("replica retry boundary is invalid");
    }
    if last_failure_reason.is_empty()
        || last_failure_reason.len() > MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES
        || !last_failure_reason
            .bytes()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == b'_')
    {
        return Err("replica retry failure reason is invalid");
    }
    Ok(())
}

fn bytes32(bytes: &[u8], field: &str) -> Result<[u8; 32], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 32 bytes"))
    })
}

fn u64_to_i64(value: u64, field: &str) -> Result<i64, DirectoryReplicaStoreError> {
    i64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} exceeds SQLite integer range"))
    })
}

fn positive_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value <= 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must be positive"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

fn nonnegative_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value < 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must not be negative"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        directory_block_range_response_signing_bytes, encode_directory_sync_message, NodeDescriptor,
    };
    use tempfile::TempDir;

    const NOW: u64 = 1_700_000_100;

    fn descriptor(identity: &IdentityKeyPair, sequence: u64) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                identity.public_key_bytes(),
                sequence,
                NOW - 10,
                NOW + 3_600,
                "replica-test",
            ),
            identity,
        )
        .unwrap()
    }

    fn block(
        producer: &IdentityKeyPair,
        height: u64,
        previous: [u8; 32],
        object: &SignedNodeDescriptor,
    ) -> DirectoryCommitmentBlockV1 {
        DirectoryCommitmentBlockV1::new_signed(
            height,
            NOW + height,
            previous,
            vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(object).unwrap()],
            producer,
        )
        .unwrap()
    }

    fn response_frame(
        producer: &IdentityKeyPair,
        blocks: Vec<DirectoryCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let responder = producer.public_key_bytes();
        let signing = directory_block_range_response_signing_bytes(
            &request_id,
            &responder,
            NOW + 20,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            responder,
            response_timestamp: NOW + 20,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: producer.sign(&signing),
        })
        .unwrap()
    }

    #[test]
    fn producer_replicas_are_isolated_idempotent_and_reopen_cleanly() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x41; 16],
        );
        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        assert_eq!(audit, DirectoryReplicaAudit::default());
        let imported = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(imported.blocks_inserted, 1);
        let repeated = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(repeated.blocks_already_present, 1);
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.producers, 1);
        assert_eq!(snapshot.quarantined_producers, 0);
        assert_eq!(snapshot.blocks, 1);
        assert_eq!(snapshot.commitments, 1);
        assert_eq!(snapshot.incidents, 0);
        assert_eq!(snapshot.producer_snapshots.len(), 1);
        assert_eq!(
            snapshot.producer_snapshots[0].producer,
            producer.public_key_bytes()
        );
        assert_eq!(snapshot.producer_snapshots[0].tip_height, 1);
        drop(store);
        let (_, reopened) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(reopened.producers, 1);
        assert_eq!(reopened.blocks, 1);
        assert_eq!(reopened.commitments, 1);
    }

    #[test]
    fn recent_observation_convergence_is_multi_source_and_order_independent() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x21; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x23; 32]).unwrap();
        let pending = IdentityKeyPair::from_bytes(&[0x24; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x25; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let frame_a = response_frame(
            &producer_a,
            vec![block_a.clone()],
            false,
            1,
            block_a.hash(),
            [0x26; 16],
        );
        let frame_b = response_frame(
            &producer_b,
            vec![block_b.clone()],
            false,
            1,
            block_b.hash(),
            [0x27; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        for (producer, replica_block, frame) in [
            (&producer_a, &block_a, &frame_a),
            (&producer_b, &block_b, &frame_b),
        ] {
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(replica_block),
                    std::slice::from_ref(&object),
                    1,
                    replica_block.hash(),
                    frame,
                    NOW + 20,
                )
                .unwrap();
        }

        let configured = [
            producer_a.public_key_bytes(),
            producer_b.public_key_bytes(),
            pending.public_key_bytes(),
        ];
        let snapshot = store.observation_convergence(&configured).unwrap();
        assert_eq!(snapshot.configured_producers, 3);
        assert_eq!(snapshot.eligible_producers, 2);
        assert_eq!(snapshot.pending_producers, 1);
        assert_eq!(snapshot.excluded_quarantined_producers, 0);
        assert_eq!(snapshot.window_blocks, 32);
        assert_eq!(snapshot.recent_commitments, 2);
        assert_eq!(snapshot.distinct_recent_commitments, 1);
        assert_eq!(snapshot.multi_source_recent_commitments, 1);
        assert_eq!(snapshot.all_eligible_source_recent_commitments, 1);
        assert!(snapshot.observation_root.is_some());

        let reversed = [configured[2], configured[1], configured[0]];
        assert_eq!(store.observation_convergence(&reversed).unwrap(), snapshot);
        assert!(store
            .observation_convergence(&[configured[0], configured[0]])
            .is_err());
    }

    #[test]
    fn observation_convergence_reads_only_the_bounded_recent_block_window() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x70; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        let mut previous = [0u8; 32];
        for height in 1..=DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS + 1 {
            let object = descriptor(&subject, height);
            let replica_block = block(&producer, height, previous, &object);
            let frame = response_frame(
                &producer,
                vec![replica_block.clone()],
                false,
                height,
                replica_block.hash(),
                [u8::try_from(height).unwrap(); 16],
            );
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(&replica_block),
                    std::slice::from_ref(&object),
                    height,
                    replica_block.hash(),
                    &frame,
                    NOW + 20,
                )
                .unwrap();
            previous = replica_block.hash();
        }

        let snapshot = store
            .observation_convergence(&[producer.public_key_bytes()])
            .unwrap();
        assert_eq!(snapshot.eligible_producers, 1);
        assert_eq!(snapshot.window_blocks, 32);
        assert_eq!(snapshot.recent_commitments, 32);
        assert_eq!(snapshot.distinct_recent_commitments, 32);
        assert_eq!(snapshot.multi_source_recent_commitments, 0);
        assert_eq!(snapshot.observation_root, None);
    }

    #[test]
    fn quarantined_producer_is_excluded_from_observation_convergence() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x28; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x29; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x2a; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x2b; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x2c; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first_a = block(&producer_a, 1, [0u8; 32], &object_a);
        let first_b = block(&producer_b, 1, [0u8; 32], &object_a);
        let fork_b = block(&producer_b, 1, [0u8; 32], &object_b);
        let frame_a = response_frame(
            &producer_a,
            vec![first_a.clone()],
            false,
            1,
            first_a.hash(),
            [0x2d; 16],
        );
        let frame_b = response_frame(
            &producer_b,
            vec![first_b.clone()],
            false,
            1,
            first_b.hash(),
            [0x2e; 16],
        );
        let fork_frame = response_frame(
            &producer_b,
            vec![fork_b.clone()],
            false,
            1,
            fork_b.hash(),
            [0x2f; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        store
            .import_verified_page(
                producer_a.public_key_bytes(),
                std::slice::from_ref(&first_a),
                std::slice::from_ref(&object_a),
                1,
                first_a.hash(),
                &frame_a,
                NOW + 20,
            )
            .unwrap();
        store
            .import_verified_page(
                producer_b.public_key_bytes(),
                std::slice::from_ref(&first_b),
                std::slice::from_ref(&object_a),
                1,
                first_b.hash(),
                &frame_b,
                NOW + 20,
            )
            .unwrap();
        assert!(matches!(
            store.import_verified_page(
                producer_b.public_key_bytes(),
                std::slice::from_ref(&fork_b),
                std::slice::from_ref(&object_b),
                1,
                fork_b.hash(),
                &fork_frame,
                NOW + 20,
            ),
            Err(DirectoryReplicaStoreError::Quarantined(_))
        ));

        let snapshot = store
            .observation_convergence(&[
                producer_a.public_key_bytes(),
                producer_b.public_key_bytes(),
            ])
            .unwrap();
        assert_eq!(snapshot.configured_producers, 2);
        assert_eq!(snapshot.eligible_producers, 1);
        assert_eq!(snapshot.pending_producers, 0);
        assert_eq!(snapshot.excluded_quarantined_producers, 1);
        assert_eq!(snapshot.recent_commitments, 1);
        assert_eq!(snapshot.multi_source_recent_commitments, 0);
        assert_eq!(snapshot.all_eligible_source_recent_commitments, 0);
        assert_eq!(snapshot.observation_root, None);
    }

    #[test]
    fn schema_v1_is_atomically_migrated_to_v2() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V1],
            )
            .unwrap();
        connection
            .execute_batch("DROP TABLE directory_replica_retry_state;")
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.retry_states, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let retry_table: String = connection
            .query_row(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name = 'directory_replica_retry_state'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(retry_table, "directory_replica_retry_state");
    }

    #[test]
    fn retry_state_survives_reopen_and_is_fully_audited() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        store
            .persist_retry_failure(
                producer_id,
                2,
                Some(NOW + 120),
                NOW,
                "directory_range_transport_failed",
            )
            .unwrap();
        store.persist_retry_skip(producer_id, NOW + 1).unwrap();
        let states = store.retry_states().unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].consecutive_failures, 2);
        assert_eq!(states[0].retry_not_before, Some(NOW + 120));
        assert_eq!(states[0].backoff_skips, 1);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 2).unwrap();
        assert_eq!(audit.producers, 1);
        assert_eq!(audit.retry_states, 1);
        assert_eq!(reopened.retry_states().unwrap(), states);
    }

    #[test]
    fn older_failure_timestamp_cannot_shorten_retry_boundary() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x3a; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x3b; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW,
        )
        .unwrap();
        store
            .persist_retry_failure(
                producer_id,
                2,
                Some(NOW + 300),
                NOW,
                "directory_range_transport_failed",
            )
            .unwrap();
        store
            .persist_retry_failure(
                producer_id,
                3,
                Some(NOW + 100),
                NOW - 100,
                "directory_object_transport_failed",
            )
            .unwrap();

        let state = &store.retry_states().unwrap()[0];
        assert_eq!(state.consecutive_failures, 3);
        assert_eq!(state.last_failure_at, NOW);
        assert_eq!(state.retry_not_before, Some(NOW + 300));
        assert_eq!(
            state.last_failure_reason,
            "directory_range_transport_failed"
        );
    }

    #[test]
    fn authenticated_import_atomically_clears_durable_retry_state() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x35; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x36; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x37; 16],
        );
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        store
            .persist_retry_failure(
                producer_id,
                3,
                Some(NOW + 300),
                NOW,
                "directory_object_transport_failed",
            )
            .unwrap();
        assert_eq!(store.retry_states().unwrap().len(), 1);

        store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert!(store.retry_states().unwrap().is_empty());
        drop(store);

        let (_, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(audit.retry_states, 0);
        assert_eq!(audit.blocks, 1);
    }

    #[test]
    fn retry_state_rejects_peer_controlled_or_unbounded_fields() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x38; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x39; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW,
        )
        .unwrap();
        assert!(store
            .persist_retry_failure(
                producer.public_key_bytes(),
                1,
                None,
                NOW,
                "https://peer.example/private",
            )
            .is_err());
        assert!(store
            .persist_retry_failure(
                producer.public_key_bytes(),
                DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES + 1,
                None,
                NOW,
                "directory_range_transport_failed",
            )
            .is_err());
        assert!(store.retry_states().unwrap().is_empty());
    }

    #[test]
    fn sync_runtime_tracks_bounded_success_and_stable_failure_without_endpoint_data() {
        let producer = [0x44; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.register_producers(&[producer]);
        runtime.record_attempt(producer, NOW);
        runtime.record_success(producer, NOW + 1, 3, 7, true, 1, 4, 2);
        runtime.record_attempt(producer, NOW + 2);
        runtime.record_failure(
            producer,
            NOW + 3,
            "pinned_directory_peer_unavailable_and_reason_is_bounded",
            Some(NOW + 123),
        );
        runtime.record_backoff_skip(producer);

        let observations = runtime.snapshot();
        assert_eq!(observations.len(), 1);
        let observation = &observations[0];
        assert_eq!(observation.producer, producer);
        assert_eq!(observation.last_success_at, Some(NOW + 1));
        assert_eq!(observation.last_failure_at, Some(NOW + 3));
        assert_eq!(observation.remote_tip_height, Some(7));
        assert_eq!(observation.local_tip_height, 3);
        assert!(observation.has_more);
        assert_eq!(observation.consecutive_failures, 1);
        assert_eq!(observation.total_attempts, 2);
        assert_eq!(observation.successful_pages, 1);
        assert_eq!(observation.failed_attempts, 1);
        assert_eq!(observation.retry_not_before, Some(NOW + 123));
        assert_eq!(observation.backoff_skips, 1);
        assert_eq!(observation.blocks_inserted, 1);
        assert_eq!(observation.commitments_inserted, 4);
        assert_eq!(observation.requests_sent, 2);
        assert_eq!(
            observation.last_failure_reason.as_deref(),
            Some("pinned_directory_peer_unavailable_and_reason_is_bounded")
        );
        assert_eq!(
            runtime.deferred_retry_until(&producer, NOW + 20),
            Some(NOW + 123)
        );
        assert_eq!(runtime.deferred_retry_until(&producer, NOW + 123), None);
        assert_eq!(runtime.consecutive_failures(&producer), 1);
    }

    #[test]
    fn sync_runtime_success_clears_backoff_without_erasing_history() {
        let producer = [0x45; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.record_failure(
            producer,
            NOW,
            "directory_range_transport_failed",
            Some(NOW + 60),
        );
        runtime.record_backoff_skip(producer);
        runtime.record_success(producer, NOW + 61, 4, 4, false, 1, 2, 1);

        let observation = &runtime.snapshot()[0];
        assert_eq!(observation.consecutive_failures, 0);
        assert_eq!(observation.retry_not_before, None);
        assert_eq!(observation.failed_attempts, 1);
        assert_eq!(observation.backoff_skips, 1);
        assert_eq!(observation.successful_pages, 1);
    }

    #[test]
    fn sync_runtime_restores_only_bounded_scheduler_state() {
        let producer = [0x46; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.register_producers(&[producer]);
        runtime.restore_retry_states(&[DirectoryReplicaRetryState {
            producer,
            consecutive_failures: DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES,
            retry_not_before: Some(NOW + 600),
            last_failure_at: NOW,
            last_failure_reason: "directory_range_transport_failed".to_string(),
            backoff_skips: 7,
        }]);

        let observation = &runtime.snapshot()[0];
        assert_eq!(
            observation.consecutive_failures,
            DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
        );
        assert_eq!(observation.retry_not_before, Some(NOW + 600));
        assert_eq!(observation.last_attempt_at, Some(NOW));
        assert_eq!(observation.last_failure_at, Some(NOW));
        assert_eq!(observation.backoff_skips, 7);
        assert_eq!(observation.total_attempts, 0);
        assert_eq!(observation.failed_attempts, 0);
        assert_eq!(observation.successful_pages, 0);

        runtime.record_failure(
            producer,
            NOW + 601,
            "directory_range_transport_failed",
            Some(NOW + 1_200),
        );
        assert_eq!(
            runtime.snapshot()[0].consecutive_failures,
            DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
        );
    }

    #[test]
    fn signed_block_fork_is_durably_quarantined_without_rewriting_prefix() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first = block(&producer, 1, [0u8; 32], &object_a);
        let fork = block(&producer, 1, [0u8; 32], &object_b);
        let first_frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x55; 16],
        );
        let fork_frame = response_frame(
            &producer,
            vec![fork.clone()],
            false,
            1,
            fork.hash(),
            [0x56; 16],
        );
        let mut invalid_evidence = fork_frame.clone();
        *invalid_evidence.last_mut().unwrap() ^= 0x01;
        assert!(
            verify_incident_response_evidence(&invalid_evidence, &producer.public_key_bytes())
                .is_err()
        );
        let path = temp.path().join("directory.db");
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object_a),
                1,
                first.hash(),
                &first_frame,
                NOW + 20,
            )
            .unwrap();
        let error = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 20,
            )
            .unwrap_err();
        assert!(matches!(error, DirectoryReplicaStoreError::Quarantined(_)));
        let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
        assert!(tip.quarantined);
        assert_eq!(tip.tip_hash, first.hash());
        drop(store);
        let (_, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(audit.quarantined_producers, 1);
        assert_eq!(audit.incidents, 1);
    }

    #[test]
    fn malformed_or_unrelated_objects_are_rejected_before_sqlite_changes() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
        let unrelated = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let wrong = descriptor(&unrelated, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x65; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        assert!(store
            .import_verified_page(
                producer.public_key_bytes(),
                &[first],
                &[wrong],
                1,
                frame_tip_hash(&frame),
                &frame,
                NOW + 20,
            )
            .is_err());
        assert_eq!(
            store
                .producer_tip(&producer.public_key_bytes())
                .unwrap()
                .tip_height,
            0
        );
    }

    fn frame_tip_hash(frame: &[u8]) -> [u8; 32] {
        let DirectorySyncMessage::BlockRangeResponseV1 { tip_hash, .. } =
            decode_directory_sync_message(frame).unwrap()
        else {
            unreachable!()
        };
        tip_hash
    }
}
