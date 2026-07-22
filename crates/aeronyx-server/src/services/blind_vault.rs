// ============================================
// File: crates/aeronyx-server/src/services/blind_vault.rs
// ============================================
//! # Blind Vault Service
//!
//! ## Creation Reason
//! Implements durable, node-blind storage for encrypted contact-vault and
//! optional message-archive segments without reusing identity-indexed MemChain
//! records or receiver-indexed legacy chat queues.
//!
//! ## Main Functionality
//! - Self-authenticating anonymous lease provisioning.
//! - Immutable, idempotent ciphertext object persistence.
//! - Capability-gated bounded recovery pages.
//! - Administration-key object deletion with signed node receipts.
//! - Transactional per-lease count/byte quotas and bounded expiry cleanup.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::blind_vault`: stable signed wire contracts.
//! - `config_blind_vault.rs`: storage and retention policy.
//! - `rusqlite`: dedicated WAL database; never shares a MemChain/chat path.
//!
//! ## Main Logical Flow
//! 1. Provision a random replica/epoch lease after external admission succeeds.
//! 2. Validate put signatures with the lease write key and store ciphertext
//!    verbatim in one immediate transaction.
//! 3. Return a descriptor-identity-signed `BlindVaultStoredReceipt`.
//! 4. Authenticate recovery with a random bearer capability and return only a
//!    bounded page of still-live opaque objects.
//! 5. Validate deletion with the separate administration key, retain a bounded
//!    commitment-only tombstone, and sign a deletion receipt.
//!
//! ## Privacy Invariant
//! This service has no account, wallet, sender, receiver, conversation,
//! namespace, content-type, vector, search-token, or social-edge columns. Logs
//! remain aggregate-only and must never include IDs, capabilities, ciphertext,
//! commitments, keys, paths selected by clients, or per-lease usage.
//!
//! ## Important Note For The Next Developer
//! - Lease provisioning is a storage primitive, not public admission. A future
//!   anonymous token gate must run before `provision_lease` is exposed publicly.
//! - Never accept mutable replacement under an existing object ID.
//! - Never publish object commitments or receipts into the Directory Chain.
//! - API handlers must run synchronous SQLite methods in `spawn_blocking`.
//! - Pull continuation sequence is node-internal and must be authenticated or
//!   encrypted before it becomes a client cursor.
//!
//! Last Modified: v1.0.0-BlindVaultService - Initial transactional service.
//! ============================================

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::blind_vault::{
    BlindVaultDeleteRequest, BlindVaultDeletedReceipt, BlindVaultError,
    BlindVaultLeaseCreateRequest, BlindVaultPutRequest, BlindVaultStoredReceipt,
};
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::config::BlindVaultConfig;

type HmacSha256 = Hmac<Sha256>;

const READ_AUTH_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Key-v1";
const READ_AUTH_TAG_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Tag-v1";
const CLEANUP_OBJECT_BATCH: usize = 512;
const CLEANUP_LEASE_BATCH: usize = 128;
const CLEANUP_TOMBSTONE_BATCH: usize = 512;

/// Result of idempotent anonymous lease provisioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseProvisionOutcome {
    /// A new lease row was committed.
    Created,
    /// The exact same signed lease was already present.
    Existing,
}

/// One opaque object returned to an authorised client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindVaultStoredObject {
    /// Replica-local object identifier.
    pub object_id: [u8; 32],
    /// Exact client ciphertext stored by the node.
    pub ciphertext: Vec<u8>,
    /// SHA-256 commitment bound by storage receipts.
    pub ciphertext_commitment: [u8; 32],
    /// Object retention deadline in Unix milliseconds.
    pub expires_at_ms: u64,
}

/// Bounded internal recovery page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindVaultPullPage {
    /// Still-live opaque objects in insertion order.
    pub objects: Vec<BlindVaultStoredObject>,
    /// Internal continuation sequence; API layers must protect it before use.
    pub continuation_sequence: Option<u64>,
}

/// Aggregate-only bounded maintenance result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultCleanupReport {
    /// Expired objects removed in this bounded run.
    pub objects_removed: u64,
    /// Expired leases removed in this bounded run.
    pub leases_removed: u64,
    /// Expired commitment-only tombstones removed in this bounded run.
    pub tombstones_removed: u64,
}

/// Aggregate-only local service health snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultStatus {
    /// Whether the configured service is enabled.
    pub enabled: bool,
    /// Number of live anonymous replica leases.
    pub live_leases: u64,
    /// Number of live opaque objects.
    pub live_objects: u64,
    /// Total live ciphertext bytes.
    pub live_ciphertext_bytes: u64,
    /// Number of retained commitment-only deletion tombstones.
    pub tombstones: u64,
}

/// Fail-closed service errors. API handlers should map these to coarse public
/// buckets instead of returning database or capability details.
#[derive(Debug, thiserror::Error)]
pub enum BlindVaultServiceError {
    /// Service was not explicitly enabled.
    #[error("blind vault service is disabled")]
    Disabled,
    /// Dedicated SQLite operation failed.
    #[error("blind vault database error")]
    Sqlite(#[from] rusqlite::Error),
    /// Core protocol validation or signature verification failed.
    #[error("blind vault protocol validation failed")]
    Protocol(#[from] BlindVaultError),
    /// Database directory could not be created.
    #[error("blind vault database directory could not be created")]
    Filesystem,
    /// Lease does not exist.
    #[error("blind vault lease not found")]
    LeaseNotFound,
    /// Lease is no longer live.
    #[error("blind vault lease expired")]
    LeaseExpired,
    /// Existing lease ID or provisioning request has different authority data.
    #[error("blind vault lease conflicts with existing state")]
    LeaseConflict,
    /// Existing object ID has different immutable content.
    #[error("blind vault object conflicts with existing state")]
    ObjectConflict,
    /// Idempotency request ID was reused for a different object.
    #[error("blind vault request conflicts with existing state")]
    RequestConflict,
    /// Object was already deleted and its identifier cannot be reused.
    #[error("blind vault object was deleted")]
    ObjectDeleted,
    /// Per-lease object or byte quota would be exceeded.
    #[error("blind vault lease quota exceeded")]
    QuotaExceeded,
    /// Read capability did not authorise this lease.
    #[error("blind vault read capability rejected")]
    ReadUnauthorized,
    /// Requested object is absent and has no retained tombstone.
    #[error("blind vault object not found")]
    ObjectNotFound,
    /// Durable row violated an internal fixed-size invariant.
    #[error("blind vault durable state is corrupt")]
    CorruptState,
    /// Timestamp could not be represented safely by SQLite.
    #[error("blind vault timestamp is outside the supported range")]
    TimestampOutOfRange,
}

/// Dedicated anonymous encrypted-object storage service.
pub struct BlindVaultService {
    config: BlindVaultConfig,
    connection: Mutex<Connection>,
    node_identity: IdentityKeyPair,
    read_auth_key: Zeroizing<[u8; 32]>,
}

impl BlindVaultService {
    /// Opens the dedicated database, applies WAL safety settings, and audits the
    /// schema. The caller must only construct this service when config is valid.
    pub fn new(
        config: BlindVaultConfig,
        node_identity: IdentityKeyPair,
    ) -> Result<Self, BlindVaultServiceError> {
        if !config.enabled {
            return Err(BlindVaultServiceError::Disabled);
        }
        if let Some(parent) = Path::new(&config.db_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|_| BlindVaultServiceError::Filesystem)?;
            }
        }
        let connection = Connection::open(&config.db_path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        init_schema(&connection)?;

        let seed = Zeroizing::new(node_identity.to_bytes());
        let mut key_deriver = HmacSha256::new_from_slice(seed.as_ref())
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        key_deriver.update(READ_AUTH_KEY_DOMAIN);
        let read_auth_key = Zeroizing::new(key_deriver.finalize().into_bytes().into());

        Ok(Self {
            config,
            connection: Mutex::new(connection),
            node_identity,
            read_auth_key,
        })
    }

    /// Inserts one exact anonymous lease or accepts an idempotent retry.
    ///
    /// A caller must perform external anonymous admission before invoking this
    /// method from a public route.
    pub fn provision_lease(
        &self,
        request: &BlindVaultLeaseCreateRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        request.validate_and_verify(now_ms, self.config.max_lease_ttl_ms())?;
        let now = sqlite_i64(now_ms)?;
        let expires_at = sqlite_i64(request.expires_at_ms)?;
        let read_tag = self.read_capability_tag(&request.lease_id, &request.read_capability_hash);

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing = load_lease_provisioning(&transaction, &request.lease_id)?;
        if let Some(existing) = existing {
            let exact = existing.request_id == request.request_id
                && existing.write_verifying_key == request.write_verifying_key
                && existing.admin_verifying_key == request.admin_verifying_key
                && existing.read_capability_tag == read_tag
                && existing.expires_at_ms == request.expires_at_ms;
            return if exact {
                Ok(BlindVaultLeaseProvisionOutcome::Existing)
            } else {
                Err(BlindVaultServiceError::LeaseConflict)
            };
        }

        let request_conflict: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM blind_vault_leases WHERE request_id = ?1)",
            params![&request.request_id[..]],
            |row| row.get(0),
        )?;
        if request_conflict {
            return Err(BlindVaultServiceError::RequestConflict);
        }

        transaction.execute(
            "INSERT INTO blind_vault_leases
             (lease_id, request_id, write_verifying_key, admin_verifying_key,
              read_capability_tag, created_at_ms, expires_at_ms, object_count, byte_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 0)",
            params![
                &request.lease_id[..],
                &request.request_id[..],
                &request.write_verifying_key[..],
                &request.admin_verifying_key[..],
                &read_tag[..],
                now,
                expires_at,
            ],
        )?;
        transaction.commit()?;
        Ok(BlindVaultLeaseProvisionOutcome::Created)
    }

    /// Stores one immutable ciphertext and returns a node-signed acceptance
    /// receipt. Exact retries return a new signature over the original
    /// acceptance time without increasing usage.
    pub fn put(
        &self,
        request: &BlindVaultPutRequest,
        now_ms: u64,
    ) -> Result<BlindVaultStoredReceipt, BlindVaultServiceError> {
        request.validate(now_ms, self.config.max_object_ttl_ms())?;
        let now = sqlite_i64(now_ms)?;
        let expires_at = sqlite_i64(request.expires_at_ms)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let lease = load_lease_runtime(&transaction, &request.lease_id)?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        if lease.expires_at_ms <= now_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        if request.expires_at_ms > lease.expires_at_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        let write_key = IdentityPublicKey::from_bytes(&lease.write_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        request.validate_and_verify(now_ms, self.config.max_object_ttl_ms(), &write_key)?;

        let tombstoned: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_tombstones
                WHERE lease_id = ?1 AND object_id = ?2
             )",
            params![&request.lease_id[..], &request.object_id[..]],
            |row| row.get(0),
        )?;
        if tombstoned {
            return Err(BlindVaultServiceError::ObjectDeleted);
        }

        if let Some(existing) = load_existing_object(&transaction, request)? {
            if existing.request_id != request.request_id
                || existing.ciphertext_commitment != request.ciphertext_commitment
                || existing.expires_at_ms != request.expires_at_ms
            {
                return Err(BlindVaultServiceError::ObjectConflict);
            }
            transaction.commit()?;
            return self.sign_store_receipt(request, existing.created_at_ms);
        }

        let request_conflict: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_objects
                WHERE lease_id = ?1 AND request_id = ?2
             )",
            params![&request.lease_id[..], &request.request_id[..]],
            |row| row.get(0),
        )?;
        if request_conflict {
            return Err(BlindVaultServiceError::RequestConflict);
        }

        let object_bytes = u64::try_from(request.ciphertext.len())
            .map_err(|_| BlindVaultServiceError::QuotaExceeded)?;
        let next_count = lease
            .object_count
            .checked_add(1)
            .ok_or(BlindVaultServiceError::QuotaExceeded)?;
        let next_bytes = lease
            .byte_count
            .checked_add(object_bytes)
            .ok_or(BlindVaultServiceError::QuotaExceeded)?;
        if next_count > self.config.max_objects_per_lease
            || next_bytes > self.config.max_bytes_per_lease
        {
            return Err(BlindVaultServiceError::QuotaExceeded);
        }

        transaction.execute(
            "INSERT INTO blind_vault_objects
             (lease_id, object_id, request_id, ciphertext, ciphertext_commitment,
              created_at_ms, expires_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &request.lease_id[..],
                &request.object_id[..],
                &request.request_id[..],
                &request.ciphertext,
                &request.ciphertext_commitment[..],
                now,
                expires_at,
            ],
        )?;
        transaction.execute(
            "UPDATE blind_vault_leases
             SET object_count = ?2, byte_count = ?3
             WHERE lease_id = ?1",
            params![
                &request.lease_id[..],
                sqlite_i64(next_count)?,
                sqlite_i64(next_bytes)?
            ],
        )?;
        transaction.commit()?;
        self.sign_store_receipt(request, now_ms)
    }

    /// Pulls one bounded page after authenticating a random read capability.
    pub fn pull_page(
        &self,
        lease_id: &[u8; 32],
        read_capability: &[u8; 32],
        after_sequence: Option<u64>,
        requested_limit: usize,
        now_ms: u64,
    ) -> Result<BlindVaultPullPage, BlindVaultServiceError> {
        if requested_limit == 0 {
            return Err(BlindVaultServiceError::ReadUnauthorized);
        }
        let now = sqlite_i64(now_ms)?;
        let after = sqlite_i64(after_sequence.unwrap_or(0))?;
        let limit = requested_limit.min(self.config.max_pull_objects);
        let query_limit = i64::try_from(limit.saturating_add(1))
            .map_err(|_| BlindVaultServiceError::CorruptState)?;

        let connection = self.connection.lock();
        let (read_tag, lease_expiry): (Vec<u8>, i64) = connection
            .query_row(
                "SELECT read_capability_tag, expires_at_ms
                 FROM blind_vault_leases WHERE lease_id = ?1",
                params![&lease_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        if lease_expiry <= now {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        self.verify_read_capability(lease_id, read_capability, &read_tag)?;

        let mut statement = connection.prepare(
            "SELECT sequence, object_id, ciphertext, ciphertext_commitment, expires_at_ms
             FROM blind_vault_objects
             WHERE lease_id = ?1 AND sequence > ?2 AND expires_at_ms > ?3
             ORDER BY sequence ASC LIMIT ?4",
        )?;
        let rows = statement.query_map(params![&lease_id[..], after, now, query_limit], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })?;

        let mut decoded = Vec::new();
        for row in rows {
            let (sequence, object_id, ciphertext, commitment, expires_at_ms) = row?;
            decoded.push((
                u64::try_from(sequence).map_err(|_| BlindVaultServiceError::CorruptState)?,
                BlindVaultStoredObject {
                    object_id: fixed_array(&object_id)?,
                    ciphertext,
                    ciphertext_commitment: fixed_array(&commitment)?,
                    expires_at_ms: u64::try_from(expires_at_ms)
                        .map_err(|_| BlindVaultServiceError::CorruptState)?,
                },
            ));
        }
        let has_more = decoded.len() > limit;
        if has_more {
            decoded.truncate(limit);
        }
        let continuation_sequence = has_more
            .then(|| decoded.last().map(|(sequence, _)| *sequence))
            .flatten();
        Ok(BlindVaultPullPage {
            objects: decoded.into_iter().map(|(_, object)| object).collect(),
            continuation_sequence,
        })
    }

    /// Deletes one object under the separate administration key and returns a
    /// signed receipt. A retained tombstone makes retries idempotent.
    pub fn delete(
        &self,
        request: &BlindVaultDeleteRequest,
        now_ms: u64,
    ) -> Result<BlindVaultDeletedReceipt, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let tombstone_expiry = now_ms
            .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
            .ok_or(BlindVaultServiceError::TimestampOutOfRange)?;
        let tombstone_expiry = sqlite_i64(tombstone_expiry)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let lease = load_lease_runtime(&transaction, &request.lease_id)?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        let admin_key = IdentityPublicKey::from_bytes(&lease.admin_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        request.validate_and_verify(now_ms, self.config.mutation_clock_skew_ms(), &admin_key)?;

        let existing: Option<(Vec<u8>, i64)> = transaction
            .query_row(
                "SELECT ciphertext_commitment, length(ciphertext)
                 FROM blind_vault_objects
                 WHERE lease_id = ?1 AND object_id = ?2",
                params![&request.lease_id[..], &request.object_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let (commitment, deleted_at_ms) = if let Some((commitment, object_bytes)) = existing {
            transaction.execute(
                "DELETE FROM blind_vault_objects
                 WHERE lease_id = ?1 AND object_id = ?2",
                params![&request.lease_id[..], &request.object_id[..]],
            )?;
            transaction.execute(
                "UPDATE blind_vault_leases
                 SET object_count = MAX(object_count - 1, 0),
                     byte_count = MAX(byte_count - ?2, 0)
                 WHERE lease_id = ?1",
                params![&request.lease_id[..], object_bytes],
            )?;
            transaction.execute(
                "INSERT INTO blind_vault_tombstones
                 (lease_id, object_id, ciphertext_commitment, deleted_at_ms, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(lease_id, object_id) DO UPDATE SET
                   ciphertext_commitment = excluded.ciphertext_commitment,
                   deleted_at_ms = excluded.deleted_at_ms,
                   expires_at_ms = excluded.expires_at_ms",
                params![
                    &request.lease_id[..],
                    &request.object_id[..],
                    &commitment,
                    now,
                    tombstone_expiry,
                ],
            )?;
            (fixed_array(&commitment)?, now_ms)
        } else {
            let tombstone: Option<(Vec<u8>, i64)> = transaction
                .query_row(
                    "SELECT ciphertext_commitment, deleted_at_ms
                     FROM blind_vault_tombstones
                     WHERE lease_id = ?1 AND object_id = ?2 AND expires_at_ms > ?3",
                    params![&request.lease_id[..], &request.object_id[..], now],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let (commitment, deleted_at) =
                tombstone.ok_or(BlindVaultServiceError::ObjectNotFound)?;
            (
                fixed_array(&commitment)?,
                u64::try_from(deleted_at).map_err(|_| BlindVaultServiceError::CorruptState)?,
            )
        };
        transaction.commit()?;

        let mut receipt = BlindVaultDeletedReceipt::new(
            request,
            commitment,
            deleted_at_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    /// Removes a bounded number of expired rows and repairs lease usage inside
    /// one immediate transaction.
    pub fn run_cleanup(
        &self,
        now_ms: u64,
    ) -> Result<BlindVaultCleanupReport, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let tombstone_expiry = sqlite_i64(
            now_ms
                .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
                .ok_or(BlindVaultServiceError::TimestampOutOfRange)?,
        )?;
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let expired_objects = select_expired_objects(&transaction, now)?;
        for object in &expired_objects {
            transaction.execute(
                "DELETE FROM blind_vault_objects WHERE sequence = ?1",
                params![object.sequence],
            )?;
            transaction.execute(
                "UPDATE blind_vault_leases
                 SET object_count = MAX(object_count - 1, 0),
                     byte_count = MAX(byte_count - ?2, 0)
                 WHERE lease_id = ?1",
                params![&object.lease_id, object.ciphertext_bytes],
            )?;
            transaction.execute(
                "INSERT OR IGNORE INTO blind_vault_tombstones
                 (lease_id, object_id, ciphertext_commitment, deleted_at_ms, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &object.lease_id,
                    &object.object_id,
                    &object.ciphertext_commitment,
                    now,
                    tombstone_expiry,
                ],
            )?;
        }

        let expired_leases = select_expired_leases(&transaction, now)?;
        for lease_id in &expired_leases {
            transaction.execute(
                "DELETE FROM blind_vault_leases WHERE lease_id = ?1",
                params![lease_id],
            )?;
        }

        let expired_tombstones = transaction.execute(
            "DELETE FROM blind_vault_tombstones
             WHERE (lease_id, object_id) IN (
                SELECT lease_id, object_id FROM blind_vault_tombstones
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_TOMBSTONE_BATCH as i64],
        )?;
        transaction.commit()?;

        Ok(BlindVaultCleanupReport {
            objects_removed: expired_objects.len() as u64,
            leases_removed: expired_leases.len() as u64,
            tombstones_removed: expired_tombstones as u64,
        })
    }

    /// Returns only node-wide aggregate health information.
    pub fn status(&self, now_ms: u64) -> Result<BlindVaultStatus, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let connection = self.connection.lock();
        let live_leases: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_leases WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let (live_objects, live_bytes): (i64, i64) = connection.query_row(
            "SELECT COUNT(*), COALESCE(SUM(length(ciphertext)), 0)
             FROM blind_vault_objects WHERE expires_at_ms > ?1",
            params![now],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let tombstones: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_tombstones WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        Ok(BlindVaultStatus {
            enabled: true,
            live_leases: non_negative_u64(live_leases)?,
            live_objects: non_negative_u64(live_objects)?,
            live_ciphertext_bytes: non_negative_u64(live_bytes)?,
            tombstones: non_negative_u64(tombstones)?,
        })
    }

    fn sign_store_receipt(
        &self,
        request: &BlindVaultPutRequest,
        accepted_at_ms: u64,
    ) -> Result<BlindVaultStoredReceipt, BlindVaultServiceError> {
        let mut receipt = BlindVaultStoredReceipt::from_put(
            request,
            accepted_at_ms,
            request.expires_at_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    fn read_capability_tag(&self, lease_id: &[u8; 32], capability_hash: &[u8; 32]) -> [u8; 32] {
        let mut mac = HmacSha256::new_from_slice(self.read_auth_key.as_ref())
            .expect("HMAC accepts a 32-byte key");
        mac.update(READ_AUTH_TAG_DOMAIN);
        mac.update(lease_id);
        mac.update(capability_hash);
        mac.finalize().into_bytes().into()
    }

    fn verify_read_capability(
        &self,
        lease_id: &[u8; 32],
        capability: &[u8; 32],
        stored_tag: &[u8],
    ) -> Result<(), BlindVaultServiceError> {
        let capability_hash: [u8; 32] = Sha256::digest(capability).into();
        let mut mac = HmacSha256::new_from_slice(self.read_auth_key.as_ref())
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        mac.update(READ_AUTH_TAG_DOMAIN);
        mac.update(lease_id);
        mac.update(&capability_hash);
        mac.verify_slice(stored_tag)
            .map_err(|_| BlindVaultServiceError::ReadUnauthorized)
    }
}

#[derive(Debug)]
struct LeaseProvisioningRow {
    request_id: [u8; 16],
    write_verifying_key: [u8; 32],
    admin_verifying_key: [u8; 32],
    read_capability_tag: [u8; 32],
    expires_at_ms: u64,
}

#[derive(Debug)]
struct LeaseRuntimeRow {
    write_verifying_key: [u8; 32],
    admin_verifying_key: [u8; 32],
    expires_at_ms: u64,
    object_count: u64,
    byte_count: u64,
}

#[derive(Debug)]
struct ExistingObjectRow {
    request_id: [u8; 16],
    ciphertext_commitment: [u8; 32],
    created_at_ms: u64,
    expires_at_ms: u64,
}

#[derive(Debug)]
struct ExpiredObjectRow {
    sequence: i64,
    lease_id: Vec<u8>,
    object_id: Vec<u8>,
    ciphertext_commitment: Vec<u8>,
    ciphertext_bytes: i64,
}

fn init_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS blind_vault_leases (
            lease_id              BLOB PRIMARY KEY CHECK(length(lease_id) = 32),
            request_id            BLOB NOT NULL UNIQUE CHECK(length(request_id) = 16),
            write_verifying_key   BLOB NOT NULL CHECK(length(write_verifying_key) = 32),
            admin_verifying_key   BLOB NOT NULL CHECK(length(admin_verifying_key) = 32),
            read_capability_tag   BLOB NOT NULL CHECK(length(read_capability_tag) = 32),
            created_at_ms         INTEGER NOT NULL CHECK(created_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > created_at_ms),
            object_count          INTEGER NOT NULL DEFAULT 0 CHECK(object_count >= 0),
            byte_count            INTEGER NOT NULL DEFAULT 0 CHECK(byte_count >= 0)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS blind_vault_objects (
            sequence              INTEGER PRIMARY KEY AUTOINCREMENT,
            lease_id              BLOB NOT NULL CHECK(length(lease_id) = 32),
            object_id             BLOB NOT NULL CHECK(length(object_id) = 32),
            request_id            BLOB NOT NULL CHECK(length(request_id) = 16),
            ciphertext            BLOB NOT NULL,
            ciphertext_commitment BLOB NOT NULL CHECK(length(ciphertext_commitment) = 32),
            created_at_ms         INTEGER NOT NULL CHECK(created_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > created_at_ms),
            UNIQUE(lease_id, object_id),
            UNIQUE(lease_id, request_id),
            FOREIGN KEY(lease_id) REFERENCES blind_vault_leases(lease_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_blind_vault_objects_pull
          ON blind_vault_objects(lease_id, sequence, expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_blind_vault_objects_expiry
          ON blind_vault_objects(expires_at_ms, sequence);

        CREATE TABLE IF NOT EXISTS blind_vault_tombstones (
            lease_id              BLOB NOT NULL CHECK(length(lease_id) = 32),
            object_id             BLOB NOT NULL CHECK(length(object_id) = 32),
            ciphertext_commitment BLOB NOT NULL CHECK(length(ciphertext_commitment) = 32),
            deleted_at_ms         INTEGER NOT NULL CHECK(deleted_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > deleted_at_ms),
            PRIMARY KEY(lease_id, object_id),
            FOREIGN KEY(lease_id) REFERENCES blind_vault_leases(lease_id) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_tombstones_expiry
          ON blind_vault_tombstones(expires_at_ms);",
    )
}

fn load_lease_provisioning(
    transaction: &Transaction<'_>,
    lease_id: &[u8; 32],
) -> Result<Option<LeaseProvisioningRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, i64)> = transaction
        .query_row(
            "SELECT request_id, write_verifying_key, admin_verifying_key,
                    read_capability_tag, expires_at_ms
             FROM blind_vault_leases WHERE lease_id = ?1",
            params![&lease_id[..]],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    row.map(|(request_id, write, admin, read_tag, expires)| {
        Ok(LeaseProvisioningRow {
            request_id: fixed_array(&request_id)?,
            write_verifying_key: fixed_array(&write)?,
            admin_verifying_key: fixed_array(&admin)?,
            read_capability_tag: fixed_array(&read_tag)?,
            expires_at_ms: u64::try_from(expires)
                .map_err(|_| BlindVaultServiceError::CorruptState)?,
        })
    })
    .transpose()
}

fn load_lease_runtime(
    transaction: &Transaction<'_>,
    lease_id: &[u8; 32],
) -> Result<Option<LeaseRuntimeRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, i64, i64, i64)> = transaction
        .query_row(
            "SELECT write_verifying_key, admin_verifying_key, expires_at_ms,
                    object_count, byte_count
             FROM blind_vault_leases WHERE lease_id = ?1",
            params![&lease_id[..]],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    row.map(|(write, admin, expires, count, bytes)| {
        Ok(LeaseRuntimeRow {
            write_verifying_key: fixed_array(&write)?,
            admin_verifying_key: fixed_array(&admin)?,
            expires_at_ms: non_negative_u64(expires)?,
            object_count: non_negative_u64(count)?,
            byte_count: non_negative_u64(bytes)?,
        })
    })
    .transpose()
}

fn load_existing_object(
    transaction: &Transaction<'_>,
    request: &BlindVaultPutRequest,
) -> Result<Option<ExistingObjectRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, i64, i64)> = transaction
        .query_row(
            "SELECT request_id, ciphertext_commitment, created_at_ms, expires_at_ms
             FROM blind_vault_objects
             WHERE lease_id = ?1 AND object_id = ?2",
            params![&request.lease_id[..], &request.object_id[..]],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    row.map(|(request_id, commitment, created, expires)| {
        Ok(ExistingObjectRow {
            request_id: fixed_array(&request_id)?,
            ciphertext_commitment: fixed_array(&commitment)?,
            created_at_ms: non_negative_u64(created)?,
            expires_at_ms: non_negative_u64(expires)?,
        })
    })
    .transpose()
}

fn select_expired_objects(
    transaction: &Transaction<'_>,
    now: i64,
) -> Result<Vec<ExpiredObjectRow>, BlindVaultServiceError> {
    let mut statement = transaction.prepare(
        "SELECT sequence, lease_id, object_id, ciphertext_commitment, length(ciphertext)
         FROM blind_vault_objects WHERE expires_at_ms <= ?1
         ORDER BY expires_at_ms ASC, sequence ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![now, CLEANUP_OBJECT_BATCH as i64], |row| {
        Ok(ExpiredObjectRow {
            sequence: row.get(0)?,
            lease_id: row.get(1)?,
            object_id: row.get(2)?,
            ciphertext_commitment: row.get(3)?,
            ciphertext_bytes: row.get(4)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn select_expired_leases(
    transaction: &Transaction<'_>,
    now: i64,
) -> Result<Vec<Vec<u8>>, BlindVaultServiceError> {
    let mut statement = transaction.prepare(
        "SELECT lease_id FROM blind_vault_leases
         WHERE expires_at_ms <= ?1 ORDER BY expires_at_ms ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![now, CLEANUP_LEASE_BATCH as i64], |row| row.get(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn fixed_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], BlindVaultServiceError> {
    bytes
        .try_into()
        .map_err(|_| BlindVaultServiceError::CorruptState)
}

fn sqlite_i64(value: u64) -> Result<i64, BlindVaultServiceError> {
    i64::try_from(value).map_err(|_| BlindVaultServiceError::TimestampOutOfRange)
}

fn non_negative_u64(value: i64) -> Result<u64, BlindVaultServiceError> {
    u64::try_from(value).map_err(|_| BlindVaultServiceError::CorruptState)
}

/// Shared service pointer used by future Axum handlers and maintenance tasks.
pub type SharedBlindVaultService = Arc<BlindVaultService>;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const NOW_MS: u64 = 1_800_000_000_000;

    struct Fixture {
        _directory: TempDir,
        service: BlindVaultService,
        write_key: IdentityKeyPair,
        admin_key: IdentityKeyPair,
        read_capability: [u8; 32],
        lease: BlindVaultLeaseCreateRequest,
    }

    impl Fixture {
        fn new(max_objects: u64, max_bytes: u64) -> Self {
            let directory = tempfile::tempdir().expect("temp directory");
            let config = BlindVaultConfig {
                enabled: true,
                db_path: directory.path().join("vault.db").display().to_string(),
                max_objects_per_lease: max_objects,
                max_bytes_per_lease: max_bytes,
                ..BlindVaultConfig::default()
            };
            let write_key = IdentityKeyPair::from_bytes(&[7; 32]).expect("write key");
            let admin_key = IdentityKeyPair::from_bytes(&[8; 32]).expect("admin key");
            let read_capability = [9; 32];
            let mut lease = BlindVaultLeaseCreateRequest::new(
                [1; 32],
                [2; 16],
                write_key.public_key_bytes(),
                admin_key.public_key_bytes(),
                Sha256::digest(read_capability).into(),
                NOW_MS + 7 * 24 * 60 * 60 * 1_000,
            );
            lease.sign(&admin_key).expect("sign lease");
            let node_key = IdentityKeyPair::from_bytes(&[10; 32]).expect("node key");
            let service = BlindVaultService::new(config, node_key).expect("service");
            Self {
                _directory: directory,
                service,
                write_key,
                admin_key,
                read_capability,
                lease,
            }
        }

        fn provision(&self) {
            assert_eq!(
                self.service
                    .provision_lease(&self.lease, NOW_MS)
                    .expect("provision lease"),
                BlindVaultLeaseProvisionOutcome::Created
            );
        }

        fn put(&self, object_byte: u8, request_byte: u8) -> BlindVaultPutRequest {
            let mut put = BlindVaultPutRequest::new(
                [1; 32],
                [object_byte; 32],
                [request_byte; 16],
                vec![object_byte; 4 * 1024],
                NOW_MS + 24 * 60 * 60 * 1_000,
            );
            put.sign(&self.write_key);
            put
        }
    }

    #[test]
    fn lease_provisioning_is_idempotent_but_not_mutable() {
        let fixture = Fixture::new(10, 1024 * 1024);
        fixture.provision();
        assert_eq!(
            fixture
                .service
                .provision_lease(&fixture.lease, NOW_MS)
                .expect("idempotent lease"),
            BlindVaultLeaseProvisionOutcome::Existing
        );

        let mut conflict = fixture.lease.clone();
        conflict.expires_at_ms += 1;
        conflict.sign(&fixture.admin_key).expect("sign conflict");
        assert!(matches!(
            fixture.service.provision_lease(&conflict, NOW_MS),
            Err(BlindVaultServiceError::LeaseConflict)
        ));
    }

    #[test]
    fn immutable_put_retry_and_capability_pull_preserve_ciphertext() {
        let fixture = Fixture::new(10, 1024 * 1024);
        fixture.provision();
        let put = fixture.put(3, 4);
        let first = fixture.service.put(&put, NOW_MS + 10).expect("put");
        let retry = fixture.service.put(&put, NOW_MS + 20).expect("retry");
        assert_eq!(first.accepted_at_ms, retry.accepted_at_ms);
        assert!(first.matches_put(&put));

        let page = fixture
            .service
            .pull_page(&[1; 32], &fixture.read_capability, None, 10, NOW_MS + 30)
            .expect("authorised pull");
        assert_eq!(page.objects.len(), 1);
        assert_eq!(page.objects[0].ciphertext, put.ciphertext);
        assert_eq!(
            page.objects[0].ciphertext_commitment,
            put.ciphertext_commitment
        );
        assert!(page.continuation_sequence.is_none());

        assert!(matches!(
            fixture
                .service
                .pull_page(&[1; 32], &[0; 32], None, 10, NOW_MS + 30),
            Err(BlindVaultServiceError::ReadUnauthorized)
        ));
    }

    #[test]
    fn per_lease_quota_is_transactional() {
        let fixture = Fixture::new(1, 4 * 1024);
        fixture.provision();
        fixture
            .service
            .put(&fixture.put(3, 4), NOW_MS + 10)
            .expect("first object");
        assert!(matches!(
            fixture.service.put(&fixture.put(5, 6), NOW_MS + 20),
            Err(BlindVaultServiceError::QuotaExceeded)
        ));
        let status = fixture.service.status(NOW_MS + 30).expect("status");
        assert_eq!(status.live_objects, 1);
        assert_eq!(status.live_ciphertext_bytes, 4 * 1024);
    }

    #[test]
    fn admin_delete_is_idempotent_and_prevents_object_id_reuse() {
        let fixture = Fixture::new(10, 1024 * 1024);
        fixture.provision();
        let put = fixture.put(3, 4);
        fixture.service.put(&put, NOW_MS + 10).expect("put");

        let mut delete = BlindVaultDeleteRequest::new([1; 32], [3; 32], [5; 16], NOW_MS + 20);
        delete.sign(&fixture.admin_key);
        let first = fixture
            .service
            .delete(&delete, NOW_MS + 20)
            .expect("delete");
        let retry = fixture
            .service
            .delete(&delete, NOW_MS + 30)
            .expect("delete retry");
        assert_eq!(
            first.previous_ciphertext_commitment,
            put.ciphertext_commitment
        );
        assert_eq!(first.deleted_at_ms, retry.deleted_at_ms);
        assert!(first.matches_delete(&delete));
        assert!(matches!(
            fixture.service.put(&put, NOW_MS + 40),
            Err(BlindVaultServiceError::ObjectDeleted)
        ));
    }

    #[test]
    fn pull_pages_are_bounded_and_cleanup_is_batched() {
        let fixture = Fixture::new(10, 1024 * 1024);
        fixture.provision();
        let first = fixture.put(3, 4);
        let second = fixture.put(5, 6);
        fixture.service.put(&first, NOW_MS + 10).expect("first");
        fixture.service.put(&second, NOW_MS + 20).expect("second");

        let page = fixture
            .service
            .pull_page(&[1; 32], &fixture.read_capability, None, 1, NOW_MS + 30)
            .expect("first page");
        assert_eq!(page.objects.len(), 1);
        let cursor = page.continuation_sequence.expect("continuation");
        let page = fixture
            .service
            .pull_page(
                &[1; 32],
                &fixture.read_capability,
                Some(cursor),
                1,
                NOW_MS + 30,
            )
            .expect("second page");
        assert_eq!(page.objects.len(), 1);

        let cleanup = fixture
            .service
            .run_cleanup(NOW_MS + 2 * 24 * 60 * 60 * 1_000)
            .expect("cleanup");
        assert_eq!(cleanup.objects_removed, 2);
        assert_eq!(
            fixture
                .service
                .status(NOW_MS + 3)
                .expect("status")
                .live_objects,
            0
        );
    }
}
