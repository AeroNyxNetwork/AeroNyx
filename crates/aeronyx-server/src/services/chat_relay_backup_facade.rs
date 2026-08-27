// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_facade.rs
// ============================================
// Version: 1.0.0-BackupManagementFacade
//
// Creation Reason:
//   [CHAT-BACKUP-FACADE-DOMAIN 2026-08-28 by Codex] Move the host-local backup,
//   retention, audit, anchor, and restore-plan API surface out of the oversized
//   relay orchestration file without widening service field visibility.
//
// Main Functionality:
//   - Creates unique or operation-idempotent verified recovery images.
//   - Audits bounded retention without mutation.
//   - Verifies maintenance audit history and issues signed public anchors.
//   - Audits restore readiness and issues/re-verifies short-lived plans.
//   - Executes explicitly admitted retention prune commands.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns service fields and compatibility helpers.
//   - Focused backup domains own filesystem, integrity, policy, and audit logic.
//   - `aeronyx-core` owns identity keys and public custody-anchor contracts.
//
// Main Logical Flow:
//   1. Validate public command data before resolving private filesystem state.
//   2. Acquire service-local and cross-process locks in the established order.
//   3. Delegate integrity, retention, audit, or restore work to focused domains.
//   4. Return aggregate receipts or opaque signed anchors without private paths.
//
// Important Note for Next Developer:
//   - Keep validation-before-path-resolution and lock ordering unchanged.
//   - Synchronous operations must remain on blocking workers at async edges.
//   - Never return or log private artifact paths, HMACs, operation ids, or keys.
//   - Restore plans prove readiness only; they are not execution authority.
//   - `*_at` methods are `pub(super)` only for deterministic sibling tests.
//
// Last Modified:
//   v1.0.0-BackupManagementFacade - Initial nested facade extraction
// ============================================

use std::path::PathBuf;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::CustodyAuditAnchorV1;

use crate::config::ChatRelayConfig;

use super::{
    derive_node_secret, now_secs, ChatRelayBackupAuditVerificationReceipt,
    ChatRelayBackupPruneReceipt, ChatRelayBackupPruneRequest, ChatRelayBackupReceipt,
    ChatRelayBackupRetentionReceipt, ChatRelayCustodyAuditAnchorGuard, ChatRelayRestorePlanReceipt,
    ChatRelayRestoreReadinessReceipt, ChatRelayResult, ChatRelayService,
};

impl ChatRelayService {
    /// Creates an owner-private, verified recovery image of relay custody.
    ///
    /// The artifact is confined to `.aeronyx-relay-backups` beside the active
    /// database; callers cannot redirect encrypted custody to an arbitrary
    /// path. `SQLite`'s online-backup API includes committed WAL pages. Before
    /// publication, the isolated image must pass physical integrity, schema
    /// sentinel, canonical usage, queue sequence, and anonymous circuit-state
    /// validation. Existing artifacts are never replaced.
    ///
    /// [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] The source connection
    /// mutex remains held only during page copying. Validation and publication
    /// operate on the isolated artifact and never inspect E2E ciphertext,
    /// wallet identities, routes, or message identifiers. This is synchronous
    /// storage work; async operator surfaces must invoke it through
    /// `spawn_blocking`, never from an Axum request executor directly.
    ///
    /// # Errors
    ///
    /// Returns a path-free SQLite/storage error if the service uses an
    /// in-memory database, the private backup area cannot be secured, the
    /// source remains locked, validation fails, or durable publication fails.
    pub fn create_verified_backup(&self) -> ChatRelayResult<PathBuf> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let created_at = now_secs();
        let nonce = rand::random::<u64>();
        let artifact_name =
            Self::backup_artifact_namespace().unique_recovery_image_name(created_at, nonce);
        let destination = backup_directory.join(artifact_name.as_str());
        self.create_verified_backup_artifact(&backup_directory, &destination, false)?;
        Ok(destination)
    }

    /// Creates or reuses one verified backup for an audited operation ID.
    ///
    /// The raw ID is never persisted. A domain-separated HMAC under the stable
    /// node secret selects one private destination, so CMS replay after ACK
    /// loss or process restart cannot create a second recovery image. An
    /// existing artifact is opened read-only with `NOFOLLOW` and fully verified
    /// before it is accepted. Corrupt or non-private artifacts fail closed.
    ///
    /// [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] This method returns
    /// aggregate receipt data only; callers cannot learn the artifact key or
    /// path. It is synchronous and must run through `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the operation ID is empty or too
    /// large, the private boundary is unavailable, snapshot certification or
    /// publication fails, or a replay artifact cannot be re-verified.
    pub(crate) fn create_verified_backup_for_operation(
        &self,
        operation_id: &str,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        let artifact_name = Self::backup_artifact_namespace()
            .idempotent_recovery_image_name(&self.node_secret, operation_id)
            .map_err(Self::map_backup_namespace_error)?;

        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let destination = backup_directory.join(artifact_name.as_str());
        self.create_verified_backup_artifact(&backup_directory, &destination, true)
    }

    /// Audits the configured retention policy without creating or deleting
    /// recovery images.
    ///
    /// Every managed recovery image is fully re-verified before the first
    /// capacity calculation. Incomplete private SQLite files left by an
    /// interrupted backup are reported, while unknown entries, symlinks,
    /// permission drift, or corrupt recovery images fail closed. At least the
    /// newest verified image is counted as retained even if it alone exceeds
    /// the configured byte budget.
    ///
    /// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] This operation shares
    /// one lock with publication and replay, so the inspection cannot race a
    /// backup that has not yet produced its audit receipt. This method has no
    /// filesystem deletion path.
    /// This is synchronous filesystem/SQLite work; async callers must use a
    /// blocking worker rather than an application request executor.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary cannot be
    /// inspected, its entry count is unbounded, an entry is unmanaged or not
    /// owner-private, or a recovery image fails full SQLite verification.
    pub fn audit_verified_backup_retention(
        &self,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(
            Self::inspect_verified_backup_retention(&self.config, &backup_directory, now_secs())?
                .receipt,
        )
    }

    /// Audits one configured private backup boundary without opening the live
    /// relay database.
    ///
    /// This host-local entry point exists for the CLI. It takes the same
    /// cross-process lock as online backup creation and returns aggregate-only
    /// state without writing an audit record or deleting an artifact.
    pub fn audit_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(Self::inspect_verified_backup_retention(config, &backup_directory, now_secs())?.receipt)
    }

    /// Verifies the private relay-custody maintenance audit from genesis.
    ///
    /// [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] The verifier authenticates
    /// every bounded record under the node identity secret, checks sequence and
    /// hash-chain continuity across immutable segments, authenticates each
    /// SHA-256 segment checkpoint, rejects schema drift, and detects file length
    /// changes during reads. It returns aggregate phase/segment counts only.
    /// Audit MACs, checkpoint MACs, hashes, operation identifiers, paths,
    /// payloads, and custody metadata never cross this service boundary. An
    /// absent audit is a valid empty history; this method does not create,
    /// rotate, recover, or repair audit files.
    ///
    /// The host-local CLI is the intended caller. This synchronous filesystem
    /// operation must run on a blocking worker when invoked from async code.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error if the private control boundary is
    /// unsafe, the audit exceeds its fixed limits, a record/checkpoint is
    /// malformed or unauthenticated, chain continuity fails, or any retained
    /// segment changes while it is being verified.
    pub fn verify_backup_maintenance_audit_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        Ok(Self::verify_backup_audit_chain(parent, node_secret)?
            .state
            .into_receipt())
    }

    /// Creates a portable identity-signed anchor for the latest immutable
    /// relay-custody audit checkpoint.
    ///
    /// [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] The returned protocol frame
    /// commits to the private checkpoint through a domain-separated opaque
    /// digest. It exposes only the producer identity, monotonic checkpoint
    /// generation, aggregate covered records/bytes, and signature. The frame is
    /// deterministic for a checkpoint; independent witnesses own time evidence.
    /// Active records are deliberately excluded until their segment is
    /// checkpointed. An interrupted rotation must be recovered by the next
    /// locked maintenance append before an anchor can be issued.
    ///
    /// # Errors
    /// Returns a path-free storage error when the private chain is unsafe,
    /// unauthenticated, has no immutable checkpoint, or rotation publication is
    /// incomplete.
    pub fn create_backup_maintenance_audit_anchor_for_config(
        config: &ChatRelayConfig,
        identity: &IdentityKeyPair,
    ) -> ChatRelayResult<CustodyAuditAnchorV1> {
        Ok(Self::hold_backup_maintenance_audit_anchor_for_config(config, identity)?.into_anchor())
    }

    /// Holds the cross-process custody maintenance lock while exposing the
    /// exact current producer-signed checkpoint anchor.
    ///
    /// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] This is the
    /// transactional boundary for host-local receipt import. Keeping the guard
    /// alive prevents a concurrent backup/prune operation from publishing a
    /// newer immutable checkpoint between exact-anchor validation and evidence
    /// persistence. It performs no network I/O and exposes no private audit
    /// material.
    ///
    /// # Errors
    /// Returns a path-free storage error under the same conditions as
    /// [`Self::create_backup_maintenance_audit_anchor_for_config`], or when the
    /// cross-process maintenance lock is already held.
    pub fn hold_backup_maintenance_audit_anchor_for_config(
        config: &ChatRelayConfig,
        identity: &IdentityKeyPair,
    ) -> ChatRelayResult<ChatRelayCustodyAuditAnchorGuard> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        let node_secret = derive_node_secret(&identity.to_bytes());
        let chain = Self::verify_backup_audit_chain(parent, &node_secret)?;
        if chain.pending_rotation.is_some() {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay backup maintenance audit rotation must complete before anchoring",
            ));
        }
        let anchor_digest = Self::backup_audit_anchor_digest(&chain.state)?;
        let receipt = chain.state.receipt();
        let anchor = CustodyAuditAnchorV1::signed(
            receipt.checkpoint_count,
            receipt.archived_record_count,
            receipt.archived_bytes,
            anchor_digest,
            identity,
        )
        .map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to sign relay backup maintenance audit anchor",
            )
        })?;
        Ok(ChatRelayCustodyAuditAnchorGuard::new(
            filesystem_lock,
            anchor,
        ))
    }

    /// Verifies whether the newest private recovery image is ready for a
    /// separately approved host-local restore operation.
    ///
    /// The command fully verifies every managed backup under the same
    /// cross-process lock used by publication and prune, then inspects the
    /// configured active main file and sidecars through metadata only. It never
    /// opens active storage, writes a maintenance audit record, or changes
    /// custody/backup artifacts. The shared control lock may be initialized.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary is unsafe,
    /// any managed image is corrupt, or active-file metadata cannot be trusted.
    pub fn audit_latest_restore_readiness_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayRestoreReadinessReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Self::restore_plan_command().audit_readiness(
            &backup_directory,
            &config.db_path,
            now_secs(),
            Self::backup_inventory_limits(config),
        )
    }

    /// Creates a short-lived authenticated plan for the newest verified image.
    ///
    /// The plan commits to private artifact identity, active database identity,
    /// aggregate recovery state, a random nonce, and a fixed ten-minute expiry.
    /// It does not copy, replace, remove, or open active custody. The plan is
    /// not sufficient authorization for a future restore operation.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when no verified image exists, active
    /// `SQLite` sidecars remain, private metadata is unsafe, or the commitment
    /// cannot be produced.
    pub fn create_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        Self::create_latest_restore_plan_at(config, node_secret, now_secs())
    }

    pub(super) fn create_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        issued_at: u64,
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Self::restore_plan_command().issue(
            &backup_directory,
            &config.db_path,
            node_secret,
            issued_at,
            Self::backup_inventory_limits(config),
        )
    }

    /// Re-verifies that an authenticated restore plan remains current.
    ///
    /// Verification repeats full backup integrity checks and active-boundary
    /// metadata inspection under the shared filesystem lock. Any expiry,
    /// tampering, backup rotation, config drift, or active database change
    /// invalidates the plan. This remains a read-only check and does not grant
    /// permission to restore. The shared lock is released on return; a future
    /// executor must re-verify and replace storage inside one uninterrupted
    /// lock scope rather than treating this method as TOCTOU-safe authority.
    ///
    /// # Errors
    ///
    /// Returns a generic path-free authentication error for an invalid,
    /// expired, or stale plan, and a storage error when the private boundary
    /// itself cannot be inspected safely.
    pub fn verify_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
    ) -> ChatRelayResult<()> {
        Self::verify_latest_restore_plan_at(config, node_secret, plan, now_secs())
    }

    pub(super) fn verify_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> ChatRelayResult<()> {
        let command = Self::restore_plan_command();
        // [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Preserve
        // validation before path resolution so malformed plans cannot create
        // the private maintenance directory or lock file.
        command.validate_public_contract(plan, now_unix_secs)?;
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        command.verify(
            &backup_directory,
            &config.db_path,
            node_secret,
            plan,
            now_unix_secs,
            Self::backup_inventory_limits(config),
        )
    }

    /// Runs a host-local retention dry-run or explicitly-confirmed prune.
    ///
    /// Dry-run is the default request state. Deletion requires the exact
    /// confirmation phrase and an operator assertion that the node is stopped;
    /// every mutation is bracketed by a private HMAC-chained aggregate audit.
    pub fn prune_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        Self::prune_verified_backup_retention_at(config, node_secret, request, now_secs())
    }

    /// Runs the same prune contract through an initialized relay service.
    pub fn prune_verified_backup_retention(
        &self,
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        let _operation = self.backup_operations.lock();
        Self::prune_verified_backup_retention_at(
            &self.config,
            &self.node_secret,
            request,
            now_secs(),
        )
    }
}
