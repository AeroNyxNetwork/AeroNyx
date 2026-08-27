// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_support.rs
// ============================================
// Version: 1.0.0-BackupSupportBoundary
//
// Creation Reason:
//   [CHAT-BACKUP-SUPPORT-DOMAIN 2026-08-28 by Codex] Move private backup
//   composition helpers and deterministic test seams out of the relay root
//   without making them part of the crate-wide service API.
//
// Main Functionality:
//   - Composes private backup namespaces, inventory, audit, and certification.
//   - Maps private filesystem and namespace failures to stable relay errors.
//   - Executes verified backup creation and explicitly admitted retention.
//   - Preserves deterministic in-module seams for backup safety tests.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns constants and the composed service fields.
//   - Backup domain modules own policy, filesystem, audit, and SQLite mechanics.
//   - `chat_relay_backup_facade.rs` owns the public operator API surface.
//
// Main Logical Flow:
//   1. Resolve only the node-owned private backup boundary.
//   2. Compose bounded verification, retention, audit, and restore policies.
//   3. Acquire cross-process locks before any durable mutation.
//   4. Return aggregate receipts or stable path-free failures.
//
// Important Note for Next Developer:
//   - `pub(super)` recreates the original relay-module visibility; do not widen.
//   - Preserve validation-before-path-resolution and established lock ordering.
//   - Never expose paths, artifact HMACs, node secrets, or audit authenticators.
//   - Keep every filesystem open no-follow and every mutation fail-closed.
//
// Last Modified:
//   v1.0.0-BackupSupportBoundary - Initial private support extraction
// ============================================

#[cfg(test)]
use std::fs::File;
use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::config::ChatRelayConfig;
use crate::services::chat_relay_backup_audit::{
    BackupAuditPhase, ChatRelayBackupMaintenanceAuditCounts,
};
use crate::services::chat_relay_backup_audit_anchor::{
    derive_backup_audit_anchor_digest, BackupAuditAnchorDigestError,
};
use crate::services::chat_relay_backup_audit_chain::ChatRelayBackupAuditChainVerification;
use crate::services::chat_relay_backup_audit_maintenance::{
    BackupAuditMaintenance, BackupAuditMaintenanceLimits,
};
#[cfg(test)]
use crate::services::chat_relay_backup_audit_rotation::ChatRelayBackupAuditSegmentRange;
use crate::services::chat_relay_backup_audit_verification::ChatRelayBackupAuditVerificationState;
use crate::services::chat_relay_backup_certification::{
    BackupRecoveryImageCertification, RecoveryImageSchemaRequirement,
    SqliteBackupRecoveryImageCertifier,
};
use crate::services::chat_relay_backup_contract::{
    ChatRelayBackupPruneReceipt, ChatRelayBackupPruneRequest, ChatRelayBackupReceipt,
};
use crate::services::chat_relay_backup_create::{
    verify_existing_backup_artifact as verify_existing_backup_creation_artifact,
    VerifiedBackupCreationCommand, VerifiedBackupCreationRequest,
};
use crate::services::chat_relay_backup_inventory::{
    BackupInventory, BackupInventoryLimits, ChatRelayBackupRetentionInspection,
    VerifiedBackupInventory,
};
#[cfg(test)]
use crate::services::chat_relay_backup_io::PrivateBackupControlFileMode;
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem,
};
use crate::services::chat_relay_backup_namespace::{
    BackupArtifactNamespace, BackupNamespaceError, HmacBackupArtifactNamespace,
};
use crate::services::chat_relay_backup_prune::{
    admit_backup_prune_request, AuditedBackupPruneExecutor, BackupPruneExecutor,
    LocalBackupArtifactRemoval,
};
use crate::services::chat_relay_backup_retention::{
    BackupRetentionLimits, BoundedBackupRetentionPlanner,
};
use crate::services::chat_relay_backup_sqlite::{
    restrict_private_sqlite_permissions, SqliteRelayBackupDatabase,
};
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};
use crate::services::chat_relay_restore_command::{local_restore_plan_command, RestorePlanCommand};

use super::{
    now_secs, ChatRelayService, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION, CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
    CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS, CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES,
    CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS, CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES,
    CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY, CHAT_RELAY_BACKUP_BUSY_TIMEOUT,
    CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT, CHAT_RELAY_BACKUP_LOCK_FILE_NAME,
    CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES, CHAT_RELAY_BACKUP_PAGES_PER_STEP,
    VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE, VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
};

impl ChatRelayService {
    #[cfg(test)]
    pub(super) fn restrict_sqlite_file_permissions(path: &Path) -> ChatRelayResult<()> {
        // [CHAT-BACKUP-SQLITE-DOMAIN 2026-08-28 by Codex] Preserve the
        // established in-crate fixture entry point while production ownership
        // remains entirely inside the SQLite backup adapter.
        restrict_private_sqlite_permissions(path)
    }

    pub(super) fn backup_io_error(code: i32, message: &'static str) -> ChatRelayError {
        backup_io_error(code, message)
    }

    pub(super) fn backup_audit_maintenance() -> BackupAuditMaintenance {
        // [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Build
        // verification, rotation, and append from one immutable limit set so
        // no service wrapper can compose mismatched safety policies.
        BackupAuditMaintenance::new(BackupAuditMaintenanceLimits {
            max_record_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES,
            max_records_per_segment: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS)
                .unwrap_or(u64::MAX),
            max_segment_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
            max_segments: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS).unwrap_or(u64::MAX),
            max_total_bytes: CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES,
        })
    }

    #[cfg(test)]
    pub(super) fn reserve_private_backup_file(path: &Path) -> ChatRelayResult<()> {
        // Test fixtures use the production no-follow/private reservation
        // boundary when simulating an interrupted maintenance artifact.
        LocalBackupFilesystem.reserve_private_file(path)
    }

    pub(super) fn private_backup_directory_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<PathBuf> {
        LocalBackupFilesystem.private_directory_for_database(&config.db_path)
    }

    pub(super) fn private_backup_directory(&self) -> ChatRelayResult<PathBuf> {
        Self::private_backup_directory_for_config(&self.config)
    }

    pub(super) fn backup_artifact_namespace() -> HmacBackupArtifactNamespace {
        HmacBackupArtifactNamespace::new(CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES)
    }

    pub(super) fn backup_recovery_image_certifier() -> SqliteBackupRecoveryImageCertifier {
        // [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex] Keep
        // runtime schema installation and recovery certification bound to the
        // same exact versions without moving service migration ownership.
        SqliteBackupRecoveryImageCertifier::new(
            RecoveryImageSchemaRequirement::new(
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            ),
            RecoveryImageSchemaRequirement::new(
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
            ),
        )
    }

    pub(super) fn verify_existing_backup_artifact(path: &Path) -> ChatRelayResult<u64> {
        verify_existing_backup_creation_artifact(
            &LocalBackupFilesystem,
            path,
            Self::verify_sqlite_backup,
        )
    }

    pub(super) fn backup_inventory() -> VerifiedBackupInventory<
        HmacBackupArtifactNamespace,
        BoundedBackupRetentionPlanner,
        fn(&Path) -> ChatRelayResult<u64>,
    > {
        // [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] Compose
        // trusted namespace and retention policy around the full SQLite
        // verifier; no private path or mutable state enters the policies.
        VerifiedBackupInventory::new(
            Self::backup_artifact_namespace(),
            BoundedBackupRetentionPlanner,
            Self::verify_existing_backup_artifact,
        )
    }

    pub(super) fn backup_inventory_limits(config: &ChatRelayConfig) -> BackupInventoryLimits {
        BackupInventoryLimits::new(
            CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT,
            BackupRetentionLimits::new(
                config.custody_backup_retention_target_artifacts,
                config.custody_backup_retention_target_bytes,
                config.custody_backup_partial_grace_secs,
            ),
        )
    }

    pub(super) fn restore_plan_command() -> impl RestorePlanCommand {
        // [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Compose the
        // verified private inventory with metadata-only active inspection and
        // the v1 authenticator at the service edge.
        local_restore_plan_command(Self::backup_inventory())
    }

    pub(super) fn inspect_verified_backup_retention(
        config: &ChatRelayConfig,
        backup_directory: &Path,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupRetentionInspection> {
        Self::backup_inventory().inspect(
            backup_directory,
            now_unix_secs,
            Self::backup_inventory_limits(config),
        )
    }

    pub(super) fn map_backup_namespace_error(error: BackupNamespaceError) -> ChatRelayError {
        match error {
            BackupNamespaceError::EmptyOperationId
            | BackupNamespaceError::OperationIdTooLarge
            | BackupNamespaceError::OperationIdLengthOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_MISUSE,
                "invalid relay backup operation identifier",
            ),
            BackupNamespaceError::SecretRejected => Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to derive private relay backup artifact identity",
            ),
        }
    }

    #[cfg(test)]
    pub(super) fn open_private_backup_control_file(
        path: &Path,
        append: bool,
    ) -> ChatRelayResult<File> {
        let mode = if append {
            PrivateBackupControlFileMode::Append
        } else {
            PrivateBackupControlFileMode::ReadWrite
        };
        LocalBackupFilesystem.open_control_file(path, mode)
    }

    pub(super) fn acquire_backup_filesystem_lock(
        backup_directory: &Path,
    ) -> ChatRelayResult<Connection> {
        LocalBackupFilesystem
            .acquire_maintenance_lock(backup_directory, CHAT_RELAY_BACKUP_LOCK_FILE_NAME)
    }

    #[cfg(test)]
    pub(super) fn backup_audit_segment_file_name(
        range: ChatRelayBackupAuditSegmentRange,
    ) -> String {
        Self::backup_audit_maintenance().segment_file_name(range)
    }

    #[cfg(test)]
    pub(super) fn backup_audit_checkpoint_file_name(
        range: ChatRelayBackupAuditSegmentRange,
    ) -> String {
        Self::backup_audit_maintenance().checkpoint_file_name(range)
    }

    pub(super) fn backup_audit_anchor_digest(
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<[u8; 32]> {
        derive_backup_audit_anchor_digest(state).map_err(|error| match error {
            BackupAuditAnchorDigestError::MissingImmutableCheckpoint => Self::backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay backup maintenance audit has no immutable checkpoint to anchor",
            ),
            BackupAuditAnchorDigestError::InvalidCheckpointAuthenticator => Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint anchor is invalid",
            ),
        })
    }

    #[cfg(test)]
    pub(super) fn verify_backup_audit_log(
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        Self::backup_audit_maintenance().verify_log(file, node_secret)
    }

    pub(super) fn verify_backup_audit_chain(
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification> {
        Self::backup_audit_maintenance().verify_chain(parent, node_secret)
    }

    #[cfg(test)]
    pub(super) fn rotate_backup_audit_segment(
        parent: &Path,
        node_secret: &[u8; 32],
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_maintenance().rotate_segment(parent, node_secret, state)
    }

    #[cfg(test)]
    pub(super) fn backup_audit_segment_needs_rotation(
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> ChatRelayResult<bool> {
        Self::backup_audit_maintenance().segment_needs_rotation(
            active_record_count,
            active_bytes,
            next_record_bytes,
        )
    }

    pub(super) fn append_backup_maintenance_audit(
        backup_directory: &Path,
        node_secret: &[u8; 32],
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_maintenance().append(
            backup_directory,
            node_secret,
            phase,
            timestamp,
            counts,
        )
    }

    #[cfg(test)]
    pub(super) fn is_lower_hex(value: &str, expected_len: usize) -> bool {
        // [CHAT-RELAY-BACKUP-AUDIT-IO-DOMAIN 2026-08-27 by Codex] Preserve
        // the shared private test helper without reintroducing production I/O.
        value.len() == expected_len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    pub(super) fn prune_verified_backup_retention_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        // [CHAT-RELAY-BACKUP-PRUNE-DOMAIN 2026-08-27 by Codex] Admission
        // remains before path resolution so an invalid command has no storage
        // side effect. The service keeps lock ownership for the full command.
        let admission = admit_backup_prune_request(request)?;
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let audit = |phase, counts| {
            Self::append_backup_maintenance_audit(
                &backup_directory,
                node_secret,
                phase,
                now_unix_secs,
                counts,
            )
        };
        AuditedBackupPruneExecutor::new(Self::backup_inventory(), LocalBackupArtifactRemoval, audit)
            .execute(
                &backup_directory,
                admission,
                now_unix_secs,
                Self::backup_inventory_limits(config),
            )
    }

    pub(super) fn create_verified_backup_artifact(
        &self,
        backup_directory: &Path,
        destination: &Path,
        reuse_existing: bool,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        let temporary_nonce = rand::random::<u64>();
        let temporary_name = Self::backup_artifact_namespace()
            .temporary_recovery_image_name(now_secs(), temporary_nonce);
        let temporary = backup_directory.join(temporary_name.as_str());
        VerifiedBackupCreationCommand::new(
            LocalBackupFilesystem,
            SqliteRelayBackupDatabase::new(
                &self.conn,
                Self::backup_recovery_image_certifier(),
                CHAT_RELAY_BACKUP_PAGES_PER_STEP,
                CHAT_RELAY_BACKUP_BUSY_TIMEOUT,
                CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY,
            ),
        )
        .execute(VerifiedBackupCreationRequest {
            backup_directory,
            destination,
            temporary: &temporary,
            reuse_existing,
        })
    }

    pub(super) fn verify_sqlite_backup(conn: &Connection) -> ChatRelayResult<()> {
        Self::backup_recovery_image_certifier().verify(conn, now_secs())
    }
}
