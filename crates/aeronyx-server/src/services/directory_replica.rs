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
//! - Stores each remote producer chain under an independent SQLite namespace.
//! - Re-verifies signed response evidence, blocks, commitments, and descriptor
//!   objects before one atomic import transaction.
//! - Makes exact repeated pages idempotent.
//! - Persists signed fork/rollback evidence and permanently quarantines only
//!   the producer that authored conflicting chain claims.
//! - Records authenticated descriptor equivocation without blaming an honest
//!   chain producer that merely observed the conflicting public descriptors.
//! - Audits every replica chain and index before the node may synchronize.
//! - Exposes low-cost aggregate snapshots and privacy-safe synchronization
//!   observations without re-running a full cryptographic audit per API read.
//!
//! ## Calling Relationships
//! - `server.rs` opens this store beside `DirectoryChainStore` at startup.
//! - `api/directory_chain_peer.rs` verifies and downloads bounded peer pages,
//!   then calls `import_verified_page` from a blocking worker.
//! - The local producer store remains the only source served by peer routes.
//!
//! ## Main Logical Flow
//! 1. Open the existing Directory Chain SQLite file and initialize only the
//!    `directory_replica_*` tables.
//! 2. Pin schema, chain id, and the local node identity in replica metadata.
//! 3. Audit every accepted producer prefix and all durable incident evidence.
//! 4. Re-verify the signed range-response frame and exact descriptor objects.
//! 5. Atomically append a contiguous producer prefix or persist quarantine.
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
//!
//! ## Last Modified
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

const DIRECTORY_REPLICA_SCHEMA_VERSION: i64 = 1;
const MAX_DIRECTORY_BLOCK_BYTES: u64 = 64 * 1024;
const MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES: u64 = 32 * 1024;
const MAX_DIRECTORY_SYNC_EVIDENCE_BYTES: usize = 512 * 1024;
const DIRECTORY_REPLICA_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;

/// Failures returned by the producer-isolated replica store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryReplicaStoreError {
    /// Filesystem setup failed.
    #[error("directory replica filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// SQLite rejected a schema, query, or transaction operation.
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
            remote_tip_height: None,
            local_tip_height: 0,
            has_more: false,
            consecutive_failures: 0,
            total_attempts: 0,
            successful_pages: 0,
            failed_attempts: 0,
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
    pub fn record_failure(&self, producer: [u8; 32], failed_at: u64, reason: &str) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(failed_at);
        observation.last_failure_at = Some(failed_at);
        observation.last_failure_reason = Some(reason.chars().take(96).collect());
        observation.consecutive_failures = observation.consecutive_failures.saturating_add(1);
        observation.failed_attempts = observation.failed_attempts.saturating_add(1);
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

    /// Returns the shared Directory Chain SQLite path.
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
    /// This is an observability read, not a replacement for Self::audit.
    /// Startup still performs the full signature, linkage, object, index, and
    /// incident audit before synchronization or API serving begins.
    ///
    /// # Errors
    /// Returns DirectoryReplicaStoreError when a persisted status row is
    /// malformed or SQLite cannot complete the bounded aggregate query.
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

    /// Audits metadata, producer prefixes, all indexes, and incident digests.
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
        let incidents = Self::audit_incidents(&connection)?;
        report.incidents = incidents;
        Ok(report)
    }

    fn initialize_schema(
        connection: &mut Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        connection.execute_batch(
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
             );",
        )?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<i64> = transaction
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if existing.is_none() {
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
        transaction.commit()?;
        Self::validate_metadata(connection, local_node_id)
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
                "directory replica metadata does not match this node and V1 production".to_string(),
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
        );

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
        assert_eq!(observation.blocks_inserted, 1);
        assert_eq!(observation.commitments_inserted, 4);
        assert_eq!(observation.requests_sent, 2);
        assert_eq!(
            observation.last_failure_reason.as_deref(),
            Some("pinned_directory_peer_unavailable_and_reason_is_bounded")
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
