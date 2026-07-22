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
//! - Atomic one-time bearer admission spend + lease creation.
//! - RFC 9474 blind-admission verification under rotating public epoch keys.
//! - Deterministic public issuer-epoch snapshots for node-signed discovery.
//! - Immutable, idempotent ciphertext object persistence.
//! - Capability-gated bounded recovery pages with encrypted snapshot cursors.
//! - Administration-key object deletion with signed node receipts.
//! - Transactional per-lease count/byte quotas and bounded expiry cleanup.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::blind_vault`: stable signed wire contracts.
//! - `config_blind_vault.rs`: storage and retention policy.
//! - `rusqlite`: dedicated WAL database; never shares a MemChain/chat path.
//!
//! ## Main Logical Flow
//! 1. Verify an operator-pinned anonymous admission issuer, atomically consume
//!    the one-time bearer ticket, and provision a random replica/epoch lease.
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
//! - Every lease, including tests, must enter through V1 or V2 admission and
//!   the one-time spend table. Do not restore a direct provisioning bypass.
//! - V1 tickets remain linkable bearer credentials for compatibility. Prefer
//!   V2 blind admission for new integrations and never reuse issuance logs as
//!   storage-node policy input.
//! - Never accept mutable replacement under an existing object ID.
//! - Never publish object commitments or receipts into the Directory Chain.
//! - API handlers must run synchronous SQLite methods in `spawn_blocking`.
//! - Pull cursors must remain lease-bound AEAD ciphertext; never expose the
//!   internal SQLite sequence or accept a caller-provided raw sequence.
//!
//! Last Modified: v1.5.0-BlindVaultAdmissionOnly - Removed the unused direct
//! lease-provisioning bypass and moved tests onto the production admission path.
//! v1.4.0-BlindVaultIssuerDirectory - Added deterministic
//! active/future issuer snapshots for authenticated key discovery.
//! v1.3.0-BlindVaultBlindAdmission - Added RFC 9474 V2
//! verification with key-bound policy and scheme-separated replay markers.
//! v1.2.0-BlindVaultAdmission - Added pinned issuer validation,
//! atomic one-time redemption, and bounded spend-marker cleanup.
//! v1.1.0-BlindVaultSnapshotCursor - Added lease-bound encrypted
//! snapshot cursors for stable recovery pagination.
//! v1.0.0-BlindVaultService - Initial transactional service.
//! ============================================

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindIssuerEpoch, BlindVaultBlindLeaseAdmissionRequest, BlindVaultDeleteRequest,
    BlindVaultDeletedReceipt, BlindVaultError, BlindVaultLeaseAdmissionRequest,
    BlindVaultLeaseCreateRequest, BlindVaultPutRequest, BlindVaultStoredReceipt,
};
use blind_rsa_signatures::{MessageRandomizer, PublicKeySha384PSSRandomized, Signature};
use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::config::BlindVaultConfig;

type HmacSha256 = Hmac<Sha256>;

const READ_AUTH_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Key-v1";
const READ_AUTH_TAG_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Tag-v1";
const PULL_CURSOR_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindVault-PullCursor-Key-v1";
const PULL_CURSOR_AAD_DOMAIN: &[u8] = b"AeroNyx-BlindVault-PullCursor-AAD-v1";
const PULL_CURSOR_VERSION: u8 = 1;
const PULL_CURSOR_NONCE_BYTES: usize = 24;
const PULL_CURSOR_PLAINTEXT_BYTES: usize = 16;
const PULL_CURSOR_TAG_BYTES: usize = 16;
const PULL_CURSOR_BYTES: usize =
    1 + PULL_CURSOR_NONCE_BYTES + PULL_CURSOR_PLAINTEXT_BYTES + PULL_CURSOR_TAG_BYTES;
const CLEANUP_OBJECT_BATCH: usize = 512;
const CLEANUP_LEASE_BATCH: usize = 128;
const CLEANUP_TOMBSTONE_BATCH: usize = 512;
const CLEANUP_ADMISSION_SPEND_BATCH: usize = 512;

struct BlindAdmissionIssuer {
    public_key: PublicKeySha384PSSRandomized,
    not_before_ms: u64,
    expires_at_ms: u64,
    max_lease_ttl_ms: u64,
}

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
    /// Lease-bound encrypted cursor for the same stable recovery snapshot.
    pub continuation_cursor: Option<Vec<u8>>,
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
    /// Expired one-time admission spend markers removed in this bounded run.
    pub admission_spends_removed: u64,
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
    /// Unexpired one-time admission spend markers retained for replay defence.
    pub retained_admission_spends: u64,
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
    /// Pull cursor could not be authenticated for this lease.
    #[error("blind vault pull cursor rejected")]
    InvalidPullCursor,
    /// Pull cursor could not be encrypted.
    #[error("blind vault pull cursor encryption failed")]
    PullCursorEncryptionFailed,
    /// Public lease admission is not explicitly enabled.
    #[error("blind vault public admission is disabled")]
    AdmissionUnavailable,
    /// Admission issuer is not pinned by this node operator.
    #[error("blind vault admission issuer rejected")]
    AdmissionIssuerRejected,
    /// Blind admission proof did not verify under its pinned epoch key.
    #[error("blind vault admission proof rejected")]
    AdmissionProofRejected,
    /// Admission ticket was already consumed by another lease.
    #[error("blind vault admission ticket already spent")]
    AdmissionSpent,
    /// Admission issuer configuration could not be parsed safely.
    #[error("blind vault admission issuer configuration is invalid")]
    AdmissionConfigurationInvalid,
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
    pull_cursor_key: Zeroizing<[u8; 32]>,
    admission_issuers: HashMap<[u8; 32], IdentityPublicKey>,
    blind_admission_issuers: HashMap<[u8; 32], BlindAdmissionIssuer>,
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
        let read_auth_key = derive_node_key(seed.as_ref(), READ_AUTH_KEY_DOMAIN)?;
        let pull_cursor_key = derive_node_key(seed.as_ref(), PULL_CURSOR_KEY_DOMAIN)?;
        let admission_issuers = config
            .admission_issuer_key_bytes()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?
            .into_iter()
            .map(|bytes| {
                IdentityPublicKey::from_bytes(&bytes)
                    .map(|key| (bytes, key))
                    .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let blind_admission_issuers = config
            .blind_admission_issuer_materials()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?
            .into_iter()
            .map(|material| {
                PublicKeySha384PSSRandomized::from_der(&material.public_key_der)
                    .map(|public_key| {
                        (
                            material.key_id,
                            BlindAdmissionIssuer {
                                public_key,
                                not_before_ms: material.not_before_ms,
                                expires_at_ms: material.expires_at_ms,
                                max_lease_ttl_ms: material.max_lease_ttl_ms,
                            },
                        )
                    })
                    .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        Ok(Self {
            config,
            connection: Mutex::new(connection),
            node_identity,
            read_auth_key,
            pull_cursor_key,
            admission_issuers,
            blind_admission_issuers,
        })
    }

    /// Returns deterministic, public, non-expired V2 issuer epochs.
    ///
    /// Future keys are intentionally included so clients can prefetch rotation
    /// material. Expired keys and all issuer-private state are omitted.
    ///
    /// # Errors
    /// Returns `AdmissionConfigurationInvalid` if canonical key material no
    /// longer matches the validated configuration snapshot.
    pub fn blind_admission_issuer_epochs(
        &self,
        now_ms: u64,
    ) -> Result<Vec<BlindVaultBlindIssuerEpoch>, BlindVaultServiceError> {
        let mut epochs = self
            .blind_admission_issuers
            .iter()
            .filter(|(_, issuer)| issuer.expires_at_ms > now_ms)
            .map(|(configured_key_id, issuer)| {
                let public_key_der = issuer
                    .public_key
                    .to_der()
                    .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
                let epoch = BlindVaultBlindIssuerEpoch::new(
                    public_key_der,
                    issuer.not_before_ms,
                    issuer.expires_at_ms,
                    issuer.max_lease_ttl_ms,
                );
                if &epoch.issuer_key_id != configured_key_id {
                    return Err(BlindVaultServiceError::AdmissionConfigurationInvalid);
                }
                Ok(epoch)
            })
            .collect::<Result<Vec<_>, BlindVaultServiceError>>()?;
        // [BLIND-VAULT-ISSUER-DIRECTORY 2026-07-23 by Codex] HashMap order is
        // process-randomized; canonical sorting keeps signatures reproducible.
        epochs.sort_by_key(|epoch| epoch.issuer_key_id);
        Ok(epochs)
    }

    /// Atomically consumes one operator-approved anonymous bearer ticket and
    /// provisions its self-authenticating lease. Exact retries remain
    /// idempotent; the same ticket cannot create a second lease.
    pub fn provision_lease_with_admission(
        &self,
        request: &BlindVaultLeaseAdmissionRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        let issuer_key = self
            .admission_issuers
            .get(&request.admission.issuer_id)
            .ok_or(BlindVaultServiceError::AdmissionIssuerRejected)?;
        request.validate_and_verify(
            now_ms,
            self.config.max_lease_ttl_ms(),
            self.config.max_admission_ticket_ttl_ms(),
            issuer_key,
        )?;

        self.provision_validated_admission(
            &request.lease,
            &request.admission.token_id,
            request.admission.expires_at_ms,
            now_ms,
        )
    }

    /// Verifies and atomically spends an unlinkable RFC 9474 credential.
    pub fn provision_lease_with_blind_admission(
        &self,
        request: &BlindVaultBlindLeaseAdmissionRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        request.admission.validate_shape()?;
        let issuer = self
            .blind_admission_issuers
            .get(&request.admission.issuer_key_id)
            .ok_or(BlindVaultServiceError::AdmissionIssuerRejected)?;
        if now_ms < issuer.not_before_ms || now_ms >= issuer.expires_at_ms {
            return Err(BlindVaultServiceError::AdmissionIssuerRejected);
        }
        let signature = Signature::new(request.admission.signature.clone());
        let randomizer = MessageRandomizer::new(request.admission.message_randomizer);
        issuer
            .public_key
            .verify(
                &signature,
                Some(randomizer),
                request.admission.message_bytes(),
            )
            .map_err(|_| BlindVaultServiceError::AdmissionProofRejected)?;
        request.lease.validate_and_verify(
            now_ms,
            self.config.max_lease_ttl_ms().min(issuer.max_lease_ttl_ms),
        )?;

        self.provision_validated_admission(
            &request.lease,
            &request.admission.spend_id(),
            issuer.expires_at_ms,
            now_ms,
        )
    }

    fn provision_validated_admission(
        &self,
        lease: &BlindVaultLeaseCreateRequest,
        spend_id: &[u8; 32],
        spend_expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let lease_expires_at = sqlite_i64(lease.expires_at_ms)?;
        let spend_expires_at = sqlite_i64(spend_expires_at_ms)?;
        let read_tag = self.read_capability_tag(&lease.lease_id, &lease.read_capability_hash);

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(outcome) = existing_lease_outcome(&transaction, lease, &read_tag)? {
            return Ok(outcome);
        }
        let spent: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_admission_spends WHERE token_id = ?1
             )",
            params![&spend_id[..]],
            |row| row.get(0),
        )?;
        if spent {
            return Err(BlindVaultServiceError::AdmissionSpent);
        }
        ensure_lease_request_available(&transaction, &lease.request_id)?;

        // [BLIND-VAULT-ADMISSION 2026-07-23 by Codex] V1 raw token IDs and
        // V2 domain-separated spend IDs share one transaction and replay table.
        // A crash commits both spend and lease or neither.
        transaction.execute(
            "INSERT INTO blind_vault_admission_spends
             (token_id, consumed_at_ms, expires_at_ms) VALUES (?1, ?2, ?3)",
            params![&spend_id[..], now, spend_expires_at],
        )?;
        insert_lease_row(&transaction, lease, &read_tag, now, lease_expires_at)?;
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
        continuation_cursor: Option<&[u8]>,
        requested_limit: usize,
        now_ms: u64,
    ) -> Result<BlindVaultPullPage, BlindVaultServiceError> {
        if requested_limit == 0 {
            return Err(BlindVaultServiceError::ReadUnauthorized);
        }
        let now = sqlite_i64(now_ms)?;
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

        // [BLIND-VAULT-CURSOR 2026-07-23 by Codex] The first page freezes a
        // sequence ceiling. Subsequent pages can neither reveal raw SQLite
        // positions nor drift into objects written after recovery began.
        let (after_sequence, ceiling_sequence) = if let Some(cursor) = continuation_cursor {
            self.decode_pull_cursor(lease_id, cursor)?
        } else {
            let ceiling: i64 = connection.query_row(
                "SELECT COALESCE(MAX(sequence), 0)
                 FROM blind_vault_objects
                 WHERE lease_id = ?1 AND expires_at_ms > ?2",
                params![&lease_id[..], now],
                |row| row.get(0),
            )?;
            (
                0,
                u64::try_from(ceiling).map_err(|_| BlindVaultServiceError::CorruptState)?,
            )
        };
        let after = sqlite_i64(after_sequence)?;
        let ceiling = sqlite_i64(ceiling_sequence)?;

        let mut statement = connection.prepare(
            "SELECT sequence, object_id, ciphertext, ciphertext_commitment, expires_at_ms
             FROM blind_vault_objects
             WHERE lease_id = ?1 AND sequence > ?2 AND sequence <= ?3
               AND expires_at_ms > ?4
             ORDER BY sequence ASC LIMIT ?5",
        )?;
        let rows = statement.query_map(
            params![&lease_id[..], after, ceiling, now, query_limit],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )?;

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
        let continuation_cursor = if has_more {
            let position = decoded
                .last()
                .map(|(sequence, _)| *sequence)
                .ok_or(BlindVaultServiceError::CorruptState)?;
            Some(self.encode_pull_cursor(lease_id, position, ceiling_sequence)?)
        } else {
            None
        };
        Ok(BlindVaultPullPage {
            objects: decoded.into_iter().map(|(_, object)| object).collect(),
            continuation_cursor,
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
        let expired_admission_spends = transaction.execute(
            "DELETE FROM blind_vault_admission_spends
             WHERE token_id IN (
                SELECT token_id FROM blind_vault_admission_spends
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_ADMISSION_SPEND_BATCH as i64],
        )?;
        transaction.commit()?;

        Ok(BlindVaultCleanupReport {
            objects_removed: expired_objects.len() as u64,
            leases_removed: expired_leases.len() as u64,
            tombstones_removed: expired_tombstones as u64,
            admission_spends_removed: expired_admission_spends as u64,
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
        let retained_admission_spends: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_admission_spends WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        Ok(BlindVaultStatus {
            enabled: true,
            live_leases: non_negative_u64(live_leases)?,
            live_objects: non_negative_u64(live_objects)?,
            live_ciphertext_bytes: non_negative_u64(live_bytes)?,
            tombstones: non_negative_u64(tombstones)?,
            retained_admission_spends: non_negative_u64(retained_admission_spends)?,
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

    fn pull_cursor_aad(lease_id: &[u8; 32]) -> Vec<u8> {
        let mut aad = Vec::with_capacity(PULL_CURSOR_AAD_DOMAIN.len() + lease_id.len());
        aad.extend_from_slice(PULL_CURSOR_AAD_DOMAIN);
        aad.extend_from_slice(lease_id);
        aad
    }

    fn encode_pull_cursor(
        &self,
        lease_id: &[u8; 32],
        position: u64,
        ceiling: u64,
    ) -> Result<Vec<u8>, BlindVaultServiceError> {
        if position > ceiling || ceiling > i64::MAX as u64 {
            return Err(BlindVaultServiceError::PullCursorEncryptionFailed);
        }
        let mut plaintext = [0_u8; PULL_CURSOR_PLAINTEXT_BYTES];
        plaintext[..8].copy_from_slice(&position.to_le_bytes());
        plaintext[8..].copy_from_slice(&ceiling.to_le_bytes());

        let mut nonce = [0_u8; PULL_CURSOR_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(self.pull_cursor_key.as_ref()));
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &plaintext,
                    aad: &Self::pull_cursor_aad(lease_id),
                },
            )
            .map_err(|_| BlindVaultServiceError::PullCursorEncryptionFailed)?;
        if ciphertext.len() != PULL_CURSOR_PLAINTEXT_BYTES + PULL_CURSOR_TAG_BYTES {
            return Err(BlindVaultServiceError::PullCursorEncryptionFailed);
        }

        let mut encoded = Vec::with_capacity(PULL_CURSOR_BYTES);
        encoded.push(PULL_CURSOR_VERSION);
        encoded.extend_from_slice(&nonce);
        encoded.extend_from_slice(&ciphertext);
        Ok(encoded)
    }

    fn decode_pull_cursor(
        &self,
        lease_id: &[u8; 32],
        encoded: &[u8],
    ) -> Result<(u64, u64), BlindVaultServiceError> {
        if encoded.len() != PULL_CURSOR_BYTES
            || encoded.first().copied() != Some(PULL_CURSOR_VERSION)
        {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }
        let nonce_start = 1;
        let ciphertext_start = nonce_start + PULL_CURSOR_NONCE_BYTES;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(self.pull_cursor_key.as_ref()));
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(&encoded[nonce_start..ciphertext_start]),
                Payload {
                    msg: &encoded[ciphertext_start..],
                    aad: &Self::pull_cursor_aad(lease_id),
                },
            )
            .map_err(|_| BlindVaultServiceError::InvalidPullCursor)?;
        if plaintext.len() != PULL_CURSOR_PLAINTEXT_BYTES {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }

        let mut position = [0_u8; 8];
        position.copy_from_slice(&plaintext[..8]);
        let mut ceiling = [0_u8; 8];
        ceiling.copy_from_slice(&plaintext[8..]);
        let position = u64::from_le_bytes(position);
        let ceiling = u64::from_le_bytes(ceiling);
        if position > ceiling || ceiling > i64::MAX as u64 {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }
        Ok((position, ceiling))
    }
}

fn derive_node_key(
    seed: &[u8],
    domain: &[u8],
) -> Result<Zeroizing<[u8; 32]>, BlindVaultServiceError> {
    let mut key_deriver =
        HmacSha256::new_from_slice(seed).map_err(|_| BlindVaultServiceError::CorruptState)?;
    key_deriver.update(domain);
    Ok(Zeroizing::new(key_deriver.finalize().into_bytes().into()))
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
          ON blind_vault_tombstones(expires_at_ms);

        CREATE TABLE IF NOT EXISTS blind_vault_admission_spends (
            token_id       BLOB PRIMARY KEY CHECK(length(token_id) = 32),
            consumed_at_ms INTEGER NOT NULL CHECK(consumed_at_ms >= 0),
            expires_at_ms  INTEGER NOT NULL CHECK(expires_at_ms > consumed_at_ms)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_admission_spends_expiry
          ON blind_vault_admission_spends(expires_at_ms);",
    )
}

fn existing_lease_outcome(
    transaction: &Transaction<'_>,
    request: &BlindVaultLeaseCreateRequest,
    read_capability_tag: &[u8; 32],
) -> Result<Option<BlindVaultLeaseProvisionOutcome>, BlindVaultServiceError> {
    let Some(existing) = load_lease_provisioning(transaction, &request.lease_id)? else {
        return Ok(None);
    };
    let exact = existing.request_id == request.request_id
        && existing.write_verifying_key == request.write_verifying_key
        && existing.admin_verifying_key == request.admin_verifying_key
        && existing.read_capability_tag == *read_capability_tag
        && existing.expires_at_ms == request.expires_at_ms;
    if exact {
        Ok(Some(BlindVaultLeaseProvisionOutcome::Existing))
    } else {
        Err(BlindVaultServiceError::LeaseConflict)
    }
}

fn ensure_lease_request_available(
    transaction: &Transaction<'_>,
    request_id: &[u8; 16],
) -> Result<(), BlindVaultServiceError> {
    let conflict: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM blind_vault_leases WHERE request_id = ?1)",
        params![&request_id[..]],
        |row| row.get(0),
    )?;
    if conflict {
        Err(BlindVaultServiceError::RequestConflict)
    } else {
        Ok(())
    }
}

fn insert_lease_row(
    transaction: &Transaction<'_>,
    request: &BlindVaultLeaseCreateRequest,
    read_capability_tag: &[u8; 32],
    created_at_ms: i64,
    expires_at_ms: i64,
) -> Result<(), BlindVaultServiceError> {
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
            &read_capability_tag[..],
            created_at_ms,
            expires_at_ms,
        ],
    )?;
    Ok(())
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
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultBlindAdmissionToken,
        BlindVaultBlindLeaseAdmissionRequest,
    };
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};
    use tempfile::TempDir;

    use crate::config_blind_vault::BlindVaultBlindAdmissionIssuerConfig;

    const NOW_MS: u64 = 1_800_000_000_000;

    struct Fixture {
        _directory: TempDir,
        service: BlindVaultService,
        write_key: IdentityKeyPair,
        admin_key: IdentityKeyPair,
        read_capability: [u8; 32],
        admission: BlindVaultAdmissionTicket,
        lease: BlindVaultLeaseCreateRequest,
    }

    impl Fixture {
        fn new(max_objects: u64, max_bytes: u64) -> Self {
            let directory = tempfile::tempdir().expect("temp directory");
            let issuer_key = IdentityKeyPair::from_bytes(&[6; 32]).expect("issuer key");
            let config = BlindVaultConfig {
                enabled: true,
                public_api_enabled: true,
                admission_issuer_public_keys: vec![hex::encode(issuer_key.public_key_bytes())],
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
            let mut admission = BlindVaultAdmissionTicket::new(
                [6; 32],
                issuer_key.public_key_bytes(),
                NOW_MS - 1_000,
                NOW_MS + 60 * 60 * 1_000,
                14 * 24 * 60 * 60 * 1_000,
            );
            admission.sign(&issuer_key).expect("sign admission");
            let node_key = IdentityKeyPair::from_bytes(&[10; 32]).expect("node key");
            let service = BlindVaultService::new(config, node_key).expect("service");
            Self {
                _directory: directory,
                service,
                write_key,
                admin_key,
                read_capability,
                admission,
                lease,
            }
        }

        fn provision(&self) {
            assert_eq!(
                self.service
                    .provision_lease_with_admission(
                        &self.admission_request(self.lease.clone()),
                        NOW_MS
                    )
                    .expect("provision lease"),
                BlindVaultLeaseProvisionOutcome::Created
            );
        }

        fn admission_request(
            &self,
            lease: BlindVaultLeaseCreateRequest,
        ) -> BlindVaultLeaseAdmissionRequest {
            BlindVaultLeaseAdmissionRequest {
                admission: self.admission.clone(),
                lease,
            }
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
                .provision_lease_with_admission(
                    &fixture.admission_request(fixture.lease.clone()),
                    NOW_MS,
                )
                .expect("idempotent lease"),
            BlindVaultLeaseProvisionOutcome::Existing
        );

        let mut conflict = fixture.lease.clone();
        conflict.expires_at_ms += 1;
        conflict.sign(&fixture.admin_key).expect("sign conflict");
        assert!(matches!(
            fixture
                .service
                .provision_lease_with_admission(&fixture.admission_request(conflict), NOW_MS,),
            Err(BlindVaultServiceError::LeaseConflict)
        ));
    }

    #[test]
    fn admission_ticket_is_spent_atomically_and_retry_is_idempotent() {
        let directory = tempfile::tempdir().expect("temp directory");
        let issuer = IdentityKeyPair::from_bytes(&[21; 32]).expect("issuer key");
        let config = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            admission_issuer_public_keys: vec![hex::encode(issuer.public_key_bytes())],
            db_path: directory.path().join("vault.db").display().to_string(),
            ..BlindVaultConfig::default()
        };
        let node_key = IdentityKeyPair::from_bytes(&[22; 32]).expect("node key");
        let service = BlindVaultService::new(config, node_key).expect("service");

        let write_key = IdentityKeyPair::from_bytes(&[23; 32]).expect("write key");
        let admin_key = IdentityKeyPair::from_bytes(&[24; 32]).expect("admin key");
        let mut lease = BlindVaultLeaseCreateRequest::new(
            [25; 32],
            [26; 16],
            write_key.public_key_bytes(),
            admin_key.public_key_bytes(),
            Sha256::digest([27; 32]).into(),
            NOW_MS + 7 * 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key).expect("sign lease");
        let mut admission = BlindVaultAdmissionTicket::new(
            [28; 32],
            issuer.public_key_bytes(),
            NOW_MS - 1_000,
            NOW_MS + 60 * 60 * 1_000,
            14 * 24 * 60 * 60 * 1_000,
        );
        admission.sign(&issuer).expect("sign admission");
        let request = BlindVaultLeaseAdmissionRequest { admission, lease };

        assert_eq!(
            service
                .provision_lease_with_admission(&request, NOW_MS)
                .expect("first admission"),
            BlindVaultLeaseProvisionOutcome::Created
        );
        assert_eq!(
            service
                .provision_lease_with_admission(&request, NOW_MS + 1)
                .expect("idempotent admission retry"),
            BlindVaultLeaseProvisionOutcome::Existing
        );

        let mut second_lease = request.lease.clone();
        second_lease.lease_id = [29; 32];
        second_lease.request_id = [30; 16];
        second_lease.sign(&admin_key).expect("sign second lease");
        let replay = BlindVaultLeaseAdmissionRequest {
            admission: request.admission.clone(),
            lease: second_lease,
        };
        assert!(matches!(
            service.provision_lease_with_admission(&replay, NOW_MS + 2),
            Err(BlindVaultServiceError::AdmissionSpent)
        ));
        assert_eq!(
            service
                .status(NOW_MS + 3)
                .expect("status")
                .retained_admission_spends,
            1
        );

        let cleanup = service
            .run_cleanup(NOW_MS + 60 * 60 * 1_000 + 1)
            .expect("cleanup spent ticket");
        assert_eq!(cleanup.admission_spends_removed, 1);
    }

    // [BLIND-VAULT-BLIND-REDEMPTION 2026-07-23 by Codex] Simulate the full
    // RFC 9474 lifecycle. The storage service receives only the finalized
    // token and cannot recover or correlate the issuer's blind transcript.
    #[test]
    fn blind_admission_is_unlinkable_one_time_and_policy_bounded() {
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        let public_der = key_pair.pk.to_der().expect("public key DER");
        let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
        let directory = tempfile::tempdir().expect("temp directory");
        let config = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
                public_key_der_base64: BASE64.encode(&public_der),
                not_before_unix_secs: NOW_MS / 1_000 - 60,
                expires_at_unix_secs: NOW_MS / 1_000 + 24 * 60 * 60,
                max_lease_ttl_secs: 14 * 24 * 60 * 60,
            }],
            db_path: directory.path().join("vault.db").display().to_string(),
            ..BlindVaultConfig::default()
        };
        let node_key = IdentityKeyPair::from_bytes(&[51; 32]).expect("node key");
        let service = BlindVaultService::new(config, node_key).expect("service");
        let advertised_epochs = service
            .blind_admission_issuer_epochs(NOW_MS)
            .expect("issuer epochs");
        assert_eq!(advertised_epochs.len(), 1);
        assert_eq!(advertised_epochs[0].issuer_key_id, issuer_key_id);
        assert_eq!(advertised_epochs[0].public_key_der, public_der);
        assert!(service
            .blind_admission_issuer_epochs(NOW_MS + 24 * 60 * 60 * 1_000)
            .expect("expired issuer filter")
            .is_empty());

        let write_key = IdentityKeyPair::from_bytes(&[52; 32]).expect("write key");
        let admin_key = IdentityKeyPair::from_bytes(&[53; 32]).expect("admin key");
        let mut lease = BlindVaultLeaseCreateRequest::new(
            [54; 32],
            [55; 16],
            write_key.public_key_bytes(),
            admin_key.public_key_bytes(),
            Sha256::digest([56; 32]).into(),
            NOW_MS + 7 * 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key).expect("sign lease");

        let token_id = [57; 32];
        let unsigned =
            BlindVaultBlindAdmissionToken::new(issuer_key_id, token_id, [1; 32], vec![0; 256]);
        let message = unsigned.message_bytes();
        let blinding = key_pair
            .pk
            .blind(&mut DefaultRng, &message)
            .expect("blind admission message");
        let blind_signature = key_pair
            .sk
            .blind_sign(&blinding.blind_message)
            .expect("blind sign admission");
        let signature = key_pair
            .pk
            .finalize(&blind_signature, &blinding, &message)
            .expect("finalize admission");
        let randomizer = blinding.msg_randomizer.expect("randomized-message mode").0;
        let admission =
            BlindVaultBlindAdmissionToken::new(issuer_key_id, token_id, randomizer, signature.0);
        let request = BlindVaultBlindLeaseAdmissionRequest { admission, lease };

        assert_eq!(
            service
                .provision_lease_with_blind_admission(&request, NOW_MS)
                .expect("first blind redemption"),
            BlindVaultLeaseProvisionOutcome::Created
        );
        assert_eq!(
            service
                .provision_lease_with_blind_admission(&request, NOW_MS + 1)
                .expect("idempotent blind retry"),
            BlindVaultLeaseProvisionOutcome::Existing
        );

        let mut second_lease = request.lease.clone();
        second_lease.lease_id = [58; 32];
        second_lease.request_id = [59; 16];
        second_lease.sign(&admin_key).expect("sign second lease");
        let replay = BlindVaultBlindLeaseAdmissionRequest {
            admission: request.admission.clone(),
            lease: second_lease,
        };
        assert!(matches!(
            service.provision_lease_with_blind_admission(&replay, NOW_MS + 2),
            Err(BlindVaultServiceError::AdmissionSpent)
        ));

        let mut forged = request.clone();
        forged.admission.token_id[0] ^= 1;
        assert!(matches!(
            service.provision_lease_with_blind_admission(&forged, NOW_MS + 3),
            Err(BlindVaultServiceError::AdmissionProofRejected)
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
        assert!(page.continuation_cursor.is_none());

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
        let cursor = page.continuation_cursor.expect("continuation");
        assert_eq!(cursor.len(), PULL_CURSOR_BYTES);
        let mut tampered_cursor = cursor.clone();
        tampered_cursor[10] ^= 1;
        assert!(matches!(
            fixture
                .service
                .decode_pull_cursor(&[1; 32], &tampered_cursor),
            Err(BlindVaultServiceError::InvalidPullCursor)
        ));
        assert!(matches!(
            fixture.service.decode_pull_cursor(&[2; 32], &cursor),
            Err(BlindVaultServiceError::InvalidPullCursor)
        ));

        // Objects appended after the first page belong to the next snapshot,
        // never to the in-progress recovery page sequence.
        let third = fixture.put(7, 8);
        fixture.service.put(&third, NOW_MS + 25).expect("third");
        let page = fixture
            .service
            .pull_page(
                &[1; 32],
                &fixture.read_capability,
                Some(&cursor),
                1,
                NOW_MS + 30,
            )
            .expect("second page");
        assert_eq!(page.objects.len(), 1);
        assert_eq!(page.objects[0].object_id, second.object_id);
        assert!(page.continuation_cursor.is_none());

        let cleanup = fixture
            .service
            .run_cleanup(NOW_MS + 2 * 24 * 60 * 60 * 1_000)
            .expect("cleanup");
        assert_eq!(cleanup.objects_removed, 3);
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
