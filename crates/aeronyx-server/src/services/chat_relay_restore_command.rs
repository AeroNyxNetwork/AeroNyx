// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_restore_command.rs
// ============================================
// Version: 1.0.0-ComposedRestoreCommandDomain
//
// Creation Reason:
//   [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Extract restore
//   readiness, authenticated-plan issuance, and current-state verification
//   from the oversized relay orchestration service.
//
// Main Functionality:
//   - Builds path-free restore-readiness receipts from verified custody state.
//   - Issues short-lived plans bound to private backup and active-file identity.
//   - Revalidates public contracts and current private state fail-closed.
//   - Maps closed restore-plan failures to stable relay storage errors.
//
// Dependencies:
//   - `chat_relay_backup_inventory` supplies verified images and active metadata.
//   - `chat_relay_restore_plan` supplies side-effect-free HMAC policy.
//   - `chat_relay_backup_contract` supplies the public readiness receipt.
//   - `rand::rngs::OsRng` supplies unique per-plan nonce material.
//
// Main Logical Flow:
//   1. Inspect the fully verified private backup inventory.
//   2. Inspect active custody metadata without opening SQLite state.
//   3. Derive aggregate readiness or a private identity-bound plan snapshot.
//   4. Issue or verify the path-free plan through the authenticator trait.
//
// Important Note for Next Developer:
//   - The caller must hold the cross-process maintenance lock during commands.
//   - Public-plan validation must happen before caller-owned path resolution.
//   - Keep blocker codes, error text, and v1 aggregate semantics stable.
//   - Never return paths, filenames, timestamps, device IDs, or inodes.
//
// Last Modified:
//   v1.0.0-ComposedRestoreCommandDomain - Initial command extraction
// ============================================

use std::path::Path;

use rand::{rngs::OsRng, RngCore};

use crate::services::chat_relay_backup_artifact::BackupArtifactSnapshot;
use crate::services::chat_relay_backup_contract::ChatRelayRestoreReadinessReceipt;
use crate::services::chat_relay_backup_inventory::{
    inspect_active_restore_boundary, verified_restore_backup_count, BackupInventory,
    BackupInventoryLimits, ChatRelayActiveRestoreBoundary, ChatRelayBackupRetentionInspection,
};
use crate::services::chat_relay_backup_io::backup_io_error;
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};
use crate::services::chat_relay_restore_plan::{
    ChatRelayRestorePlanReceipt, HmacRestorePlanAuthenticator, RestorePlanAggregate,
    RestorePlanAuthenticator, RestorePlanError, RestorePlanPrivateBoundary,
    RESTORE_PLAN_NONCE_BYTES,
};

/// Replaceable metadata-only inspection of active relay custody.
pub(super) trait ActiveRestoreBoundaryInspector {
    fn inspect(&self, database_path: &str) -> ChatRelayResult<ChatRelayActiveRestoreBoundary>;
}

impl<F> ActiveRestoreBoundaryInspector for F
where
    F: Fn(&str) -> ChatRelayResult<ChatRelayActiveRestoreBoundary>,
{
    fn inspect(&self, database_path: &str) -> ChatRelayResult<ChatRelayActiveRestoreBoundary> {
        self(database_path)
    }
}

/// Host filesystem boundary inspector preserving metadata-only semantics.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LocalActiveRestoreBoundaryInspector;

impl ActiveRestoreBoundaryInspector for LocalActiveRestoreBoundaryInspector {
    fn inspect(&self, database_path: &str) -> ChatRelayResult<ChatRelayActiveRestoreBoundary> {
        inspect_active_restore_boundary(database_path)
    }
}

/// Replaceable source for per-plan uniqueness material.
pub(super) trait RestorePlanNonceSource {
    fn generate(&self) -> [u8; RESTORE_PLAN_NONCE_BYTES];
}

/// Operating-system random nonce source used by production planning.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct OsRestorePlanNonceSource;

impl RestorePlanNonceSource for OsRestorePlanNonceSource {
    fn generate(&self) -> [u8; RESTORE_PLAN_NONCE_BYTES] {
        let mut nonce = [0u8; RESTORE_PLAN_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        nonce
    }
}

/// Lock-protected restore-readiness and plan command capability.
pub(super) trait RestorePlanCommand {
    fn audit_readiness(
        &self,
        backup_directory: &Path,
        database_path: &str,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayRestoreReadinessReceipt>;

    fn validate_public_contract(
        &self,
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> ChatRelayResult<()>;

    fn issue(
        &self,
        backup_directory: &Path,
        database_path: &str,
        node_secret: &[u8; 32],
        issued_at: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt>;

    fn verify(
        &self,
        backup_directory: &Path,
        database_path: &str,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<()>;
}

/// Composed command over inventory, active-boundary, and authentication traits.
#[derive(Debug)]
pub(super) struct ComposedRestorePlanCommand<Inventory, Boundary, Authenticator, Nonce> {
    inventory: Inventory,
    boundary: Boundary,
    authenticator: Authenticator,
    nonce: Nonce,
}

impl<Inventory, Boundary, Authenticator, Nonce>
    ComposedRestorePlanCommand<Inventory, Boundary, Authenticator, Nonce>
{
    pub(super) const fn new(
        inventory: Inventory,
        boundary: Boundary,
        authenticator: Authenticator,
        nonce: Nonce,
    ) -> Self {
        Self {
            inventory,
            boundary,
            authenticator,
            nonce,
        }
    }
}

/// Composes the production restore command around a caller-provided inventory.
pub(super) fn local_restore_plan_command<Inventory>(
    inventory: Inventory,
) -> ComposedRestorePlanCommand<
    Inventory,
    LocalActiveRestoreBoundaryInspector,
    HmacRestorePlanAuthenticator,
    OsRestorePlanNonceSource,
> {
    ComposedRestorePlanCommand::new(
        inventory,
        LocalActiveRestoreBoundaryInspector,
        HmacRestorePlanAuthenticator,
        OsRestorePlanNonceSource,
    )
}

impl<Inventory, Boundary, Authenticator, Nonce> RestorePlanCommand
    for ComposedRestorePlanCommand<Inventory, Boundary, Authenticator, Nonce>
where
    Inventory: BackupInventory,
    Boundary: ActiveRestoreBoundaryInspector,
    Authenticator: RestorePlanAuthenticator,
    Nonce: RestorePlanNonceSource,
{
    fn audit_readiness(
        &self,
        backup_directory: &Path,
        database_path: &str,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayRestoreReadinessReceipt> {
        let inspection = self
            .inventory
            .inspect(backup_directory, now_unix_secs, limits)?;
        let verified_backup_count = verified_restore_backup_count(&inspection)?;
        let selected_backup_bytes = inspection
            .newest_backup
            .as_ref()
            .map(BackupArtifactSnapshot::size_bytes)
            .unwrap_or_default();
        let active = self.boundary.inspect(database_path)?;
        let blocker = if inspection.newest_backup.is_none() {
            Some("no_verified_backup")
        } else if active.sidecars_present {
            Some("active_sqlite_sidecars_present")
        } else {
            None
        };

        Ok(ChatRelayRestoreReadinessReceipt {
            ready: blocker.is_none(),
            verified_backup_count,
            selected_backup_bytes,
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
            active_sidecars_present: active.sidecars_present,
            blocker,
        })
    }

    fn validate_public_contract(
        &self,
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> ChatRelayResult<()> {
        self.authenticator
            .validate_public_contract(plan, now_unix_secs)
            .map_err(map_restore_plan_error)
    }

    fn issue(
        &self,
        backup_directory: &Path,
        database_path: &str,
        node_secret: &[u8; 32],
        issued_at: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        let inspection = self
            .inventory
            .inspect(backup_directory, issued_at, limits)?;
        let backup = inspection.newest_backup.as_ref().ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay restore plan requires a verified backup",
            )
        })?;
        let active = self.boundary.inspect(database_path)?;
        if active.sidecars_present {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay restore plan requires an inactive SQLite boundary",
            ));
        }
        let aggregate = issue_aggregate(&inspection, backup, &active)?;
        let nonce = self.nonce.generate();
        self.authenticator
            .issue(
                node_secret,
                issued_at,
                aggregate,
                private_boundary(database_path, backup, &active),
                nonce,
            )
            .map_err(map_restore_plan_error)
    }

    fn verify(
        &self,
        backup_directory: &Path,
        database_path: &str,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<()> {
        // [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Validate
        // again inside the lock-protected command. The service also validates
        // before path resolution so malformed input cannot create storage.
        self.validate_public_contract(plan, now_unix_secs)?;
        let inspection = self
            .inventory
            .inspect(backup_directory, now_unix_secs, limits)?;
        let backup = inspection
            .newest_backup
            .as_ref()
            .ok_or_else(invalid_or_stale_restore_plan)?;
        let active = self.boundary.inspect(database_path)?;
        if active.sidecars_present {
            return Err(invalid_or_stale_restore_plan());
        }
        let aggregate = verification_aggregate(&inspection, backup, &active)?;
        self.authenticator
            .verify(
                node_secret,
                plan,
                now_unix_secs,
                aggregate,
                private_boundary(database_path, backup, &active),
            )
            .map_err(map_restore_plan_error)
    }
}

fn issue_aggregate(
    inspection: &ChatRelayBackupRetentionInspection,
    backup: &BackupArtifactSnapshot,
    active: &ChatRelayActiveRestoreBoundary,
) -> ChatRelayResult<RestorePlanAggregate> {
    let verified_backup_count =
        u64::try_from(verified_restore_backup_count(inspection)?).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay restore-plan backup count exceeds wire format",
            )
        })?;
    Ok(RestorePlanAggregate {
        verified_backup_count,
        selected_backup_bytes: backup.size_bytes(),
        active_database_present: active.present,
        active_database_bytes: active.size_bytes,
    })
}

fn verification_aggregate(
    inspection: &ChatRelayBackupRetentionInspection,
    backup: &BackupArtifactSnapshot,
    active: &ChatRelayActiveRestoreBoundary,
) -> ChatRelayResult<RestorePlanAggregate> {
    let verified_backup_count = u64::try_from(verified_restore_backup_count(inspection)?)
        .map_err(|_| invalid_or_stale_restore_plan())?;
    Ok(RestorePlanAggregate {
        verified_backup_count,
        selected_backup_bytes: backup.size_bytes(),
        active_database_present: active.present,
        active_database_bytes: active.size_bytes,
    })
}

fn private_boundary<'a>(
    database_path: &'a str,
    backup: &'a BackupArtifactSnapshot,
    active: &ChatRelayActiveRestoreBoundary,
) -> RestorePlanPrivateBoundary<'a> {
    RestorePlanPrivateBoundary {
        configured_database_path: database_path,
        selected_backup_name: backup.file_name(),
        selected_backup_modified_at: backup.modified_at(),
        active_database_modified_at: active.modified_at,
        selected_backup_device_id: backup.device_id(),
        selected_backup_inode: backup.inode(),
        active_database_device_id: active.device_id,
        active_database_inode: active.inode,
    }
}

fn invalid_or_stale_restore_plan() -> ChatRelayError {
    map_restore_plan_error(RestorePlanError::InvalidOrStale)
}

fn map_restore_plan_error(error: RestorePlanError) -> ChatRelayError {
    match error {
        RestorePlanError::ExpiryOutOfRange => backup_io_error(
            rusqlite::ffi::SQLITE_RANGE,
            "relay restore-plan expiry is out of range",
        ),
        RestorePlanError::FilesystemTimeOutOfRange => backup_io_error(
            rusqlite::ffi::SQLITE_RANGE,
            "relay restore-plan filesystem time is out of range",
        ),
        RestorePlanError::EncodingFailed => backup_io_error(
            rusqlite::ffi::SQLITE_FORMAT,
            "unable to encode relay restore plan",
        ),
        RestorePlanError::AuthenticatorInitFailed => backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "unable to initialize relay restore plan",
        ),
        RestorePlanError::InvalidOrStale => backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "relay restore plan is invalid, expired, or stale",
        ),
    }
}
