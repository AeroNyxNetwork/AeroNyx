// ============================================
// File: crates/aeronyx-server/src/services/memchain/volume_router.rs
// ============================================
//! # VolumeRouter — Per-User Disk Volume Routing
//!
//! ## Creation Reason
//! Part of the MemChain Multi-Tenant Architecture (v1.0).
//! Decides which physical disk volume stores each user's SQLite DB file.
//! Supports hot-reload of volumes.toml for zero-downtime disk expansion.
//!
//! ## Main Functionality
//! - Loads and validates volumes.toml configuration
//! - Routes owner pubkeys to their assigned volume
//! - Selects least-loaded writable volume for new users
//! - Generates deterministic DB and vec file paths per owner
//! - Supports hot-reload (SIGHUP) to add/change volumes at runtime
//! - Auto-generates default volumes.toml on first SaaS startup
//! - Measures managed DB/sidecar/vector bytes without following symlinks
//!
//! ## Dependencies
//! - Depends on SystemDb (Task 1a) for persistent assignment storage
//! - Used by StoragePool (Task 1b) to open per-user MemoryStorage
//! - Used by VectorIndexPool (Task 1c) for .vec file paths
//! - Used by Admin endpoints (Task 5) for volume health stats
//!
//! ## File Naming
//! DB path: `{volume.path}/{hex(owner)[..16]}.db`
//! Vec path: `{volume.path}/{hex(owner)[..16]}.vec`
//!
//! Collision probability at 10,000 users/volume: ~0.0003%
//! If collision detected at open time, fallback to full 64-char hex.
//!
//! ## Volume State Machine
//! ```text
//! read-write ──(capacity)──→ read-only ──(future draining)──→ removed
//! ```
//! Only read-write and read-only are implemented in this version.
//! Draining support is reserved for future migration tooling.
//!
//! ⚠️ Important Note for Next Developer:
//! - volumes uses parking_lot::RwLock because every critical section is
//!   synchronous and short. Do not hold its guard across an await.
//! - assignment_gate serializes new-owner placement and config reload. This
//!   prevents a reload from removing a volume after assign() selected it and
//!   keeps max_users decisions consistent within this process.
//! - SystemDb.assign_volume() still uses INSERT OR IGNORE + AlreadyAssigned as
//!   defense in depth against another process assigning the same owner.
//! - reload_config(): path changes are intentionally ignored (the old
//!   path is the canonical location for existing user DBs).
//! - A hot reload cannot remove a volume with active assignments. Migrate its
//!   users first; otherwise route() and db_path() could disagree.
//! - ensure_volumes_config() is called from Server::new() in SaaS mode,
//!   not from VolumeRouter::new() — keep them separate for testability.
//! - Volume usage probing is synchronous filesystem work and must stay behind
//!   the explicit `spawn_blocking` boundary in `volume_stats()`.
//!
//! ## Last Modified
//! v2.8.56-VolumeUsageObservability - Report real managed-file volume usage.
//! v2.8.55-VolumeRouterIntegrity - Made hot reload fail closed and recoverable.
//! v2.7.14-RustdocQuality - Classified the volume lifecycle diagram as text.
//! v1.0.0-MultiTenant - Initial implementation (Task 1a)
// ============================================

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use super::system_db::{SystemDb, SystemDbError};

// ============================================
// Public Types
// ============================================

/// Status of a storage volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VolumeStatus {
    /// Accepts new users and writes.
    ReadWrite,
    /// Existing users can read/write but no new users are assigned here.
    ReadOnly,
    /// Reserved for future migration support — parsed but not acted on.
    Draining,
}

/// Configuration for a single storage volume.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct VolumeConfig {
    pub id: String,
    pub path: PathBuf,
    #[serde(default = "default_status")]
    pub status: VolumeStatus,
    #[serde(default = "default_max_users")]
    pub max_users: usize,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: u64,
}

fn default_status() -> VolumeStatus {
    VolumeStatus::ReadWrite
}
fn default_max_users() -> usize {
    10_000
}
fn default_max_bytes() -> u64 {
    500_000_000_000
} // 500 GB

/// Top-level structure of volumes.toml.
#[derive(serde::Deserialize)]
struct VolumesFile {
    volumes: Vec<VolumeConfig>,
}

/// Runtime statistics for a single volume (Admin API).
#[derive(Debug, Clone, serde::Serialize)]
pub struct VolumeStats {
    pub volume_id: String,
    pub status: VolumeStatus,
    pub path: PathBuf,
    pub user_count: usize,
    pub max_users: usize,
    pub max_bytes: u64,
    /// Actual bytes occupied by AeroNyx-managed regular files in this volume.
    pub disk_usage_bytes: u64,
}

/// Typed failures produced while measuring one volume's managed files.
#[derive(Debug, thiserror::Error)]
pub enum VolumeUsageProbeError {
    #[error("Failed to {operation} at '{path}': {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Volume root must not be a symbolic link: {0}")]
    SymlinkRoot(PathBuf),

    #[error("Volume root is not a directory: {0}")]
    InvalidRoot(PathBuf),

    #[error("Managed-file byte accounting overflow at '{path}'")]
    Overflow { path: PathBuf },
}

/// Replaceable boundary for synchronous volume-usage measurement.
///
/// Implementations may perform blocking filesystem I/O. Callers from async
/// code must execute this trait behind a blocking-worker boundary.
pub trait VolumeUsageProbe: Send + Sync {
    fn usage_bytes(&self, volume_root: &Path) -> Result<u64, VolumeUsageProbeError>;
}

/// Production probe for the flat per-owner volume layout.
#[derive(Debug, Default)]
pub struct FilesystemVolumeUsageProbe;

impl VolumeUsageProbe for FilesystemVolumeUsageProbe {
    fn usage_bytes(&self, volume_root: &Path) -> Result<u64, VolumeUsageProbeError> {
        // [VOLUME-USAGE-OBSERVABILITY 2026-08-31 by Codex] Reject a symlinked
        // root and inspect only its direct entries. No recursive traversal can
        // escape the configured volume, and symlink metadata never follows an
        // entry to an operator-uncontrolled target.
        let root_metadata =
            std::fs::symlink_metadata(volume_root).map_err(|source| VolumeUsageProbeError::Io {
                operation: "inspect volume root",
                path: volume_root.to_path_buf(),
                source,
            })?;
        if root_metadata.file_type().is_symlink() {
            return Err(VolumeUsageProbeError::SymlinkRoot(
                volume_root.to_path_buf(),
            ));
        }
        if !root_metadata.is_dir() {
            return Err(VolumeUsageProbeError::InvalidRoot(
                volume_root.to_path_buf(),
            ));
        }

        let entries =
            std::fs::read_dir(volume_root).map_err(|source| VolumeUsageProbeError::Io {
                operation: "read volume directory",
                path: volume_root.to_path_buf(),
                source,
            })?;
        let mut total = 0u64;
        for entry in entries {
            let entry = entry.map_err(|source| VolumeUsageProbeError::Io {
                operation: "read volume directory entry",
                path: volume_root.to_path_buf(),
                source,
            })?;
            if !is_managed_volume_filename(&entry.file_name()) {
                continue;
            }

            let path = entry.path();
            let metadata =
                std::fs::symlink_metadata(&path).map_err(|source| VolumeUsageProbeError::Io {
                    operation: "inspect managed volume file",
                    path: path.clone(),
                    source,
                })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                continue;
            }
            total = checked_add_usage(total, metadata.len(), &path)?;
        }
        Ok(total)
    }
}

/// Errors from VolumeRouter operations.
#[derive(Debug, thiserror::Error)]
pub enum VolumeRouterError {
    #[error("No writable volume available — all volumes are full or read-only")]
    NoWritableVolume,

    #[error("Volume config file not found: {0}")]
    ConfigNotFound(PathBuf),

    #[error("Volume config parse error: {0}")]
    ConfigParse(String),

    #[error("Volume path does not exist and could not be created: {0}")]
    PathNotFound(PathBuf),

    #[error("Duplicate volume id: {0}")]
    DuplicateId(String),

    #[error("Volume '{0}' not found in config")]
    VolumeNotFound(String),

    /// Durable assignments reference a volume absent from active configuration.
    #[error(
        "Volume '{volume_id}' has {assigned_users} assigned users but is absent from active configuration"
    )]
    AssignedVolumeUnavailable {
        volume_id: String,
        assigned_users: usize,
    },

    #[error("SystemDb error: {0}")]
    SystemDb(#[from] SystemDbError),

    #[error("Volume usage probe failed for '{volume_id}': {source}")]
    VolumeUsage {
        volume_id: String,
        #[source]
        source: VolumeUsageProbeError,
    },

    #[error("Volume usage blocking worker failed")]
    VolumeUsageWorkerFailed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ============================================
// VolumeRouter
// ============================================

/// Routes owner pubkeys to their assigned storage volume.
///
/// Keeps an in-memory DashMap cache of owner → volume_id.
/// The DashMap is populated from SystemDb on startup and kept in sync
/// as new users are assigned.
///
/// ## Thread Safety
/// `DashMap` is concurrent-safe for all read/write operations.
/// `RwLock<Vec<VolumeConfig>>` protects the volume config list —
/// reads are cheap (shared), reload takes a write lock briefly.
pub struct VolumeRouter {
    /// Volume configuration list (hot-reloadable).
    volumes: RwLock<Vec<VolumeConfig>>,

    /// Serializes new-owner placement with configuration replacement.
    ///
    /// [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Existing-owner lookups
    /// remain lock-free through DashMap; only the rare placement/reload control
    /// path is serialized across its async SystemDb boundary.
    assignment_gate: TokioMutex<()>,

    /// In-memory owner → volume_id cache (populated at startup from SystemDb).
    assignments: DashMap<[u8; 32], String>,

    /// SystemDb reference for persistent assignment writes.
    system_db: Arc<SystemDb>,

    /// Path to volumes.toml for hot-reload support.
    config_path: PathBuf,

    /// Synchronous managed-file measurement boundary.
    usage_probe: Arc<dyn VolumeUsageProbe>,
}

impl VolumeRouter {
    // ============================================
    // Construction
    // ============================================

    /// Initialize the VolumeRouter from volumes.toml and SystemDb state.
    ///
    /// Steps:
    /// 1. Load and validate volumes.toml
    /// 2. Create any missing volume directories
    /// 3. Check for duplicate volume IDs
    /// 4. Load all existing owner → volume_id assignments from SystemDb
    /// 5. Warn about assignments pointing to now-unconfigured volumes
    pub async fn new(
        config_path: &Path,
        system_db: Arc<SystemDb>,
    ) -> Result<Arc<Self>, VolumeRouterError> {
        Self::new_with_usage_probe(config_path, system_db, Arc::new(FilesystemVolumeUsageProbe))
            .await
    }

    /// Initialize with an explicit usage probe.
    ///
    /// This preserves the production constructor while making filesystem
    /// measurement replaceable for alternate platforms and deterministic tests.
    pub async fn new_with_usage_probe(
        config_path: &Path,
        system_db: Arc<SystemDb>,
        usage_probe: Arc<dyn VolumeUsageProbe>,
    ) -> Result<Arc<Self>, VolumeRouterError> {
        let config_path = config_path.to_path_buf();

        // Load and validate config.
        let volumes = load_volumes_config(&config_path)?;

        // Populate in-memory cache from persistent state.
        let assignments: DashMap<[u8; 32], String> = DashMap::new();
        let all = system_db.load_all_assignments().await?;
        let volume_ids: std::collections::HashSet<&str> =
            volumes.iter().map(|v| v.id.as_str()).collect();
        let mut ghost_assignments = std::collections::BTreeMap::<String, usize>::new();

        for (owner, vol_id) in &all {
            if !volume_ids.contains(vol_id.as_str()) {
                let count = ghost_assignments.entry(vol_id.clone()).or_default();
                *count = count.saturating_add(1);
            }
            assignments.insert(*owner, vol_id.clone());
        }
        for (volume_id, assigned_users) in &ghost_assignments {
            // [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Volume health is
            // operator metadata; owner public keys must never enter node logs.
            warn!(
                volume_id,
                assigned_users,
                "[VOLUME_ROUTER] Assignments reference an unconfigured volume; \
                 restore the volume configuration before opening user storage"
            );
        }
        if let Some((volume_id, assigned_users)) = ghost_assignments.into_iter().next() {
            return Err(VolumeRouterError::AssignedVolumeUnavailable {
                volume_id,
                assigned_users,
            });
        }

        info!(
            config = %config_path.display(),
            volumes = volumes.len(),
            assignments = all.len(),
            "[VOLUME_ROUTER] ✅ Initialized"
        );

        Ok(Arc::new(Self {
            volumes: RwLock::new(volumes),
            assignment_gate: TokioMutex::new(()),
            assignments,
            system_db,
            config_path,
            usage_probe,
        }))
    }

    // ============================================
    // Routing
    // ============================================

    /// Look up the volume assigned to an owner. O(1) memory lookup.
    ///
    /// Returns None for new users who have not been assigned yet.
    pub fn route(&self, owner: &[u8; 32]) -> Option<String> {
        self.assignments.get(owner).map(|v| v.clone())
    }

    #[cfg(test)]
    pub(crate) fn cache_assignment_for_test(&self, owner: [u8; 32], volume_id: &str) {
        self.assignments.insert(owner, volume_id.to_string());
    }

    /// Assign a new user to the least-loaded writable volume.
    ///
    /// ## Algorithm
    /// 1. Get per-volume user counts from SystemDb
    /// 2. Filter to ReadWrite volumes that haven't reached max_users
    /// 3. Choose the volume with the fewest current users (load balance)
    /// 4. Persist to SystemDb (INSERT OR IGNORE + AlreadyAssigned handling)
    /// 5. Update in-memory cache
    ///
    /// ## TOCTOU Mitigation
    /// The count → assign window can have a race: two concurrent requests
    /// for new users may both pick the same "least loaded" volume.
    /// This is acceptable — both will be assigned there, and the extra
    /// user just slightly over-counts. If SystemDb.assign_volume returns
    /// AlreadyAssigned, the assignment already exists (race was won by
    /// the other request) and we use that existing volume.
    ///
    /// # SECURITY
    /// This function must only be called after the owner has been
    /// authenticated. Never call with an unverified owner pubkey.
    pub async fn assign(&self, owner: &[u8; 32]) -> Result<String, VolumeRouterError> {
        // [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Hold one async control
        // gate across selection and persistence so reload cannot invalidate the
        // selected volume before the durable assignment commits.
        let _assignment_guard = self.assignment_gate.lock().await;

        // Get current per-volume user counts.
        let counts: std::collections::HashMap<String, usize> = self
            .system_db
            .count_users_per_volume()
            .await?
            .into_iter()
            .collect();

        // Select the target volume under a read lock.
        let target_id = {
            let vols = self.volumes.read();

            vols.iter()
                .filter(|v| v.status == VolumeStatus::ReadWrite)
                .filter(|v| {
                    let current = counts.get(&v.id).copied().unwrap_or(0);
                    current < v.max_users
                })
                .min_by_key(|v| counts.get(&v.id).copied().unwrap_or(0))
                .map(|v| v.id.clone())
                .ok_or(VolumeRouterError::NoWritableVolume)?
        };

        // Persist the assignment. Handle the race case gracefully.
        match self.system_db.assign_volume(owner, &target_id).await {
            Ok(()) => {
                // New assignment — update cache.
                self.assignments.insert(*owner, target_id.clone());
                info!(
                    volume_id = target_id,
                    "[VOLUME_ROUTER] Assigned new user to volume"
                );
                Ok(target_id)
            }
            Err(SystemDbError::AlreadyAssigned(existing)) => {
                // Race: another request assigned this owner first.
                // Update our cache and return the winner's volume.
                self.assignments.insert(*owner, existing.clone());
                Ok(existing)
            }
            Err(e) => Err(VolumeRouterError::SystemDb(e)),
        }
    }

    // ============================================
    // Path Generation
    // ============================================

    /// Get the SQLite DB file path for a given owner on a given volume.
    ///
    /// Format: `{volume.path}/{hex(owner)[..16]}.db`
    ///
    /// # Errors
    /// Returns `VolumeNotFound` if volume_id is not in the current config.
    pub fn db_path(&self, volume_id: &str, owner: &[u8; 32]) -> Result<PathBuf, VolumeRouterError> {
        let vol_path = self.volume_path(volume_id)?;
        let filename = format!("{}.db", owner_to_filename(owner));
        Ok(vol_path.join(filename))
    }

    /// Get the vector index file path for a given owner on a given volume.
    ///
    /// Format: `{volume.path}/{hex(owner)[..16]}.vec`
    pub fn vec_path(
        &self,
        volume_id: &str,
        owner: &[u8; 32],
    ) -> Result<PathBuf, VolumeRouterError> {
        let vol_path = self.volume_path(volume_id)?;
        let filename = format!("{}.vec", owner_to_filename(owner));
        Ok(vol_path.join(filename))
    }

    /// Retrieve the filesystem path for a volume by ID.
    fn volume_path(&self, volume_id: &str) -> Result<PathBuf, VolumeRouterError> {
        let vols = self.volumes.read();
        vols.iter()
            .find(|v| v.id == volume_id)
            .map(|v| v.path.clone())
            .ok_or_else(|| VolumeRouterError::VolumeNotFound(volume_id.to_string()))
    }

    // ============================================
    // Hot Reload
    // ============================================

    /// Reload volumes.toml at runtime (triggered by SIGHUP or Admin API).
    ///
    /// ## Reload Rules
    /// - New volumes: added to the list immediately
    /// - Status changes: take effect immediately for new assignments
    /// - Path changes: warned and ignored (existing DBs stay at canonical paths)
    /// - Removed unassigned volumes: removed immediately
    /// - Removed assigned volumes: reject the complete reload without mutation
    ///
    /// Existing in-memory assignments are NOT invalidated on reload.
    pub async fn reload_config(&self) -> Result<(), VolumeRouterError> {
        let mut new_volumes = load_volumes_config(&self.config_path)?;
        let _assignment_guard = self.assignment_gate.lock().await;
        let mut vols = self.volumes.write();

        let assigned_users_by_volume = self.assignments.iter().fold(
            std::collections::HashMap::<String, usize>::new(),
            |mut counts, assignment| {
                let count = counts.entry(assignment.value().clone()).or_default();
                *count = count.saturating_add(1);
                counts
            },
        );

        // [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Reject the complete
        // reload before mutating live state when it would orphan any durable
        // assignment. Unassigned volumes remain removable.
        for old_volume in vols.iter() {
            if new_volumes
                .iter()
                .any(|candidate| candidate.id == old_volume.id)
            {
                continue;
            }
            let assigned_users = assigned_users_by_volume
                .get(&old_volume.id)
                .copied()
                .unwrap_or(0);
            if assigned_users > 0 {
                return Err(VolumeRouterError::AssignedVolumeUnavailable {
                    volume_id: old_volume.id.clone(),
                    assigned_users,
                });
            }
        }

        // Preserve canonical paths while applying mutable capacity/status
        // fields. A config typo must not redirect existing DB filenames.
        for new_vol in &mut new_volumes {
            if let Some(old_vol) = vols.iter().find(|v| v.id == new_vol.id) {
                if old_vol.path != new_vol.path {
                    warn!(
                        volume_id = new_vol.id,
                        old_path = %old_vol.path.display(),
                        new_path = %new_vol.path.display(),
                        "[VOLUME_ROUTER] Path change ignored on reload — \
                         existing user DBs remain at old path"
                    );
                    new_vol.path.clone_from(&old_vol.path);
                }
            }
        }

        let new_count = new_volumes.len();
        let old_count = vols.len();
        *vols = new_volumes;

        info!(old_count, new_count, "[VOLUME_ROUTER] ✅ Config reloaded");

        Ok(())
    }

    // ============================================
    // Observability
    // ============================================

    /// Get per-volume statistics for the Admin API.
    pub async fn volume_stats(&self) -> Result<Vec<VolumeStats>, VolumeRouterError> {
        let counts: std::collections::HashMap<String, usize> = self
            .system_db
            .count_users_per_volume()
            .await?
            .into_iter()
            .collect();

        let volumes = self.volumes.read().clone();
        let probe = Arc::clone(&self.usage_probe);
        let probe_inputs: Vec<(String, PathBuf)> = volumes
            .iter()
            .map(|volume| (volume.id.clone(), volume.path.clone()))
            .collect();

        // [VOLUME-USAGE-OBSERVABILITY 2026-08-31 by Codex] Directory walking
        // is blocking filesystem I/O. Move the complete scan off Tokio's async
        // workers, with owned paths and no parking_lot guard crossing await.
        let measured = tokio::task::spawn_blocking(move || {
            probe_inputs
                .into_iter()
                .map(|(volume_id, path)| match probe.usage_bytes(&path) {
                    Ok(usage) => Ok((volume_id, usage)),
                    Err(source) => Err(VolumeRouterError::VolumeUsage { volume_id, source }),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|_| VolumeRouterError::VolumeUsageWorkerFailed)??;

        let stats = volumes
            .into_iter()
            .zip(measured)
            .map(|(v, (measured_id, disk_usage_bytes))| {
                debug_assert_eq!(v.id, measured_id);
                VolumeStats {
                    volume_id: v.id.clone(),
                    status: v.status,
                    path: v.path.clone(),
                    user_count: counts.get(&v.id).copied().unwrap_or(0),
                    max_users: v.max_users,
                    max_bytes: v.max_bytes,
                    disk_usage_bytes,
                }
            })
            .collect();

        Ok(stats)
    }

    /// Return the number of configured volumes without running a usage probe.
    ///
    /// Reload responses need only configuration cardinality. Keeping that path
    /// separate avoids an unnecessary filesystem scan and prevents a probe
    /// failure from being misreported as a successful reload with zero volumes.
    pub fn configured_volume_count(&self) -> usize {
        self.volumes.read().len()
    }

    /// Check whether the router has any writable volume available.
    pub fn has_writable_volume(&self) -> bool {
        let vols = self.volumes.read();
        vols.iter().any(|v| v.status == VolumeStatus::ReadWrite)
    }
}

// ============================================
// volumes.toml Loading
// ============================================

/// Load and validate a volumes.toml file.
///
/// Validation rules:
/// 1. File must exist and be valid TOML
/// 2. At least one volume must be defined
/// 3. All volume IDs must be unique
/// 4. All volume paths are created if they do not exist
/// 5. Warns if no ReadWrite volume is configured
fn load_volumes_config(path: &Path) -> Result<Vec<VolumeConfig>, VolumeRouterError> {
    let content = std::fs::read_to_string(path)
        .map_err(|_| VolumeRouterError::ConfigNotFound(path.to_path_buf()))?;

    let file: VolumesFile =
        toml::from_str(&content).map_err(|e| VolumeRouterError::ConfigParse(e.to_string()))?;

    if file.volumes.is_empty() {
        return Err(VolumeRouterError::ConfigParse(
            "volumes.toml must define at least one [[volumes]] entry".into(),
        ));
    }

    // Check for duplicate IDs.
    let mut seen = std::collections::HashSet::new();
    for vol in &file.volumes {
        if !seen.insert(vol.id.as_str()) {
            return Err(VolumeRouterError::DuplicateId(vol.id.clone()));
        }
    }

    // Create directories for volumes that do not exist yet.
    for vol in &file.volumes {
        if !vol.path.exists() {
            std::fs::create_dir_all(&vol.path).map_err(|e| {
                tracing::error!(
                    path = %vol.path.display(),
                    error = %e,
                    "[VOLUME_ROUTER] Failed to create volume directory"
                );
                VolumeRouterError::PathNotFound(vol.path.clone())
            })?;
            info!(
                path = %vol.path.display(),
                "[VOLUME_ROUTER] Created volume directory"
            );
        }
    }

    // Warn if no writable volume configured.
    let has_writable = file
        .volumes
        .iter()
        .any(|v| v.status == VolumeStatus::ReadWrite);
    if !has_writable {
        warn!(
            "[VOLUME_ROUTER] No ReadWrite volume configured — \
             new user assignment will fail until a writable volume is added"
        );
    }

    Ok(file.volumes)
}

// ============================================
// Default volumes.toml Generation
// ============================================

/// If volumes.toml does not exist, generate a default single-volume config.
///
/// Called by Server::new() in SaaS mode before VolumeRouter::new().
/// Creates `{data_root}/volumes/vol-001/` and writes `{data_root}/volumes.toml`.
///
/// # Returns
/// The path to the (possibly newly created) volumes.toml.
pub fn ensure_volumes_config(data_root: &Path) -> Result<PathBuf, VolumeRouterError> {
    let config_path = data_root.join("volumes.toml");
    if config_path.exists() {
        return Ok(config_path);
    }

    // Create the default volume directory.
    let default_vol_path = data_root.join("volumes").join("vol-001");
    std::fs::create_dir_all(&default_vol_path)?;

    // Write the default configuration.
    let content = format!(
        r#"# MemChain Volume Configuration
# Generated automatically on first SaaS mode startup.
#
# To add a disk:  append a new [[volumes]] entry, then reload (SIGHUP or
#                 POST /api/admin/volumes/reload)
# Disk full:      set status = "read-only" — no new users, existing users OK
# Remove a disk:  set status = "draining" (future: triggers user migration)

[[volumes]]
id = "vol-001"
path = "{}"
status = "read-write"
max_users = 10000
max_bytes = 500000000000
"#,
        // Normalize path separators for cross-platform TOML.
        default_vol_path.to_string_lossy().replace('\\', "/")
    );

    std::fs::write(&config_path, content)?;
    info!(
        path = %config_path.display(),
        "[VOLUME_ROUTER] Generated default volumes.toml"
    );

    Ok(config_path)
}

// ============================================
// File Naming Helpers
// ============================================

/// Convert an owner pubkey to a short filename prefix.
///
/// Uses the first 16 hex characters (64 bits) of the pubkey.
///
/// ## Collision probability
/// - 1,000 users/volume: < 0.00001%
/// - 10,000 users/volume: < 0.0003%
///
/// At these scales the probability is negligible. If collisions become
/// a concern, switch to full 64-char hex by changing this function.
fn owner_to_filename(owner: &[u8; 32]) -> String {
    // SAFETY: hex::encode always produces valid UTF-8.
    hex::encode(owner)[..16].to_string()
}

/// Recognize the complete flat-file grammar owned by VolumeRouter and SQLite.
fn is_managed_volume_filename(name: &std::ffi::OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let owner = [".db-wal", ".db-shm", ".db", ".vec"]
        .into_iter()
        .find_map(|suffix| name.strip_suffix(suffix));
    let Some(owner) = owner else {
        return false;
    };
    matches!(owner.len(), 16 | 64) && owner.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_add_usage(
    total: u64,
    file_bytes: u64,
    path: &Path,
) -> Result<u64, VolumeUsageProbeError> {
    total
        .checked_add(file_bytes)
        .ok_or_else(|| VolumeUsageProbeError::Overflow {
            path: path.to_path_buf(),
        })
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::system_db::SystemDb;
    use std::sync::mpsc;
    use tempfile::TempDir;

    // ── Test Helpers ──────────────────────────────────────────────────

    /// Write a volumes.toml with one or two volumes.
    fn write_volumes_toml(dir: &Path, volumes: &[(&str, VolumeStatus)]) -> PathBuf {
        let config_path = dir.join("volumes.toml");
        let mut content = String::new();
        for (id, status) in volumes {
            let vol_dir = dir.join("volumes").join(id);
            std::fs::create_dir_all(&vol_dir).unwrap();
            let status_str = match status {
                VolumeStatus::ReadWrite => "read-write",
                VolumeStatus::ReadOnly => "read-only",
                VolumeStatus::Draining => "draining",
            };
            content.push_str(&format!(
                "[[volumes]]\nid = \"{}\"\npath = \"{}\"\nstatus = \"{}\"\n\n",
                id,
                vol_dir.to_string_lossy().replace('\\', "/"),
                status_str
            ));
        }
        std::fs::write(&config_path, content).unwrap();
        config_path
    }

    async fn make_router(
        dir: &Path,
        volumes: &[(&str, VolumeStatus)],
    ) -> (Arc<SystemDb>, Arc<VolumeRouter>) {
        let db = SystemDb::open(&dir.join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir, volumes);
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        (db, router)
    }

    fn make_owner(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    struct FailingUsageProbe;

    impl VolumeUsageProbe for FailingUsageProbe {
        fn usage_bytes(&self, volume_root: &Path) -> Result<u64, VolumeUsageProbeError> {
            Err(VolumeUsageProbeError::Io {
                operation: "test volume probe",
                path: volume_root.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "injected probe failure",
                ),
            })
        }
    }

    struct ThreadReportingUsageProbe {
        sender: mpsc::Sender<std::thread::ThreadId>,
        usage: u64,
    }

    impl VolumeUsageProbe for ThreadReportingUsageProbe {
        fn usage_bytes(&self, _volume_root: &Path) -> Result<u64, VolumeUsageProbeError> {
            self.sender.send(std::thread::current().id()).unwrap();
            Ok(self.usage)
        }
    }

    // ── Construction ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_new_with_valid_config() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;
        assert!(router.has_writable_volume());
    }

    #[tokio::test]
    async fn test_new_with_missing_config() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let missing = dir.path().join("nonexistent.toml");
        let err = match VolumeRouter::new(&missing, Arc::clone(&db)).await {
            Ok(_) => panic!("missing config should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, VolumeRouterError::ConfigNotFound(_)));
    }

    #[tokio::test]
    async fn test_new_with_duplicate_ids() {
        let dir = TempDir::new().unwrap();
        // Manually write a config with duplicate IDs.
        let vol_dir = dir.path().join("volumes").join("vol-001");
        std::fs::create_dir_all(&vol_dir).unwrap();
        let path = dir.path().join("volumes.toml");
        std::fs::write(
            &path,
            format!(
                "[[volumes]]\nid = \"vol-001\"\npath = \"{}\"\n\n\
                 [[volumes]]\nid = \"vol-001\"\npath = \"{}\"\n",
                vol_dir.to_string_lossy().replace('\\', "/"),
                vol_dir.to_string_lossy().replace('\\', "/"),
            ),
        )
        .unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let err = match VolumeRouter::new(&path, Arc::clone(&db)).await {
            Ok(_) => panic!("duplicate volume ids should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, VolumeRouterError::DuplicateId(_)));
    }

    #[tokio::test]
    async fn test_new_rejects_assignments_for_unconfigured_volume() {
        // [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Startup must not report
        // ready when durable assignments cannot resolve to a configured path.
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        db.assign_volume(&make_owner(0x41), "removed-volume")
            .await
            .unwrap();
        let config_path = write_volumes_toml(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]);

        let error = match VolumeRouter::new(&config_path, db).await {
            Ok(_) => panic!("unconfigured durable assignment should fail startup"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            VolumeRouterError::AssignedVolumeUnavailable {
                volume_id,
                assigned_users: 1
            } if volume_id == "removed-volume"
        ));
    }

    // ── Assignment / Routing ──────────────────────────────────────────

    #[tokio::test]
    async fn test_assign_selects_least_loaded() {
        let dir = TempDir::new().unwrap();
        let (db, router) = make_router(
            dir.path(),
            &[
                ("vol-001", VolumeStatus::ReadWrite),
                ("vol-002", VolumeStatus::ReadWrite),
            ],
        )
        .await;

        // Manually assign 5 users to vol-001 so vol-002 is less loaded.
        for i in 0u8..5 {
            db.assign_volume(&make_owner(i), "vol-001").await.unwrap();
            router.assignments.insert(make_owner(i), "vol-001".into());
        }

        let new_owner = make_owner(0x80);
        let vol = router.assign(&new_owner).await.unwrap();
        assert_eq!(vol, "vol-002", "Should pick least-loaded vol-002");
    }

    #[tokio::test]
    async fn test_assign_skips_readonly_volumes() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(
            dir.path(),
            &[
                ("vol-001", VolumeStatus::ReadOnly),
                ("vol-002", VolumeStatus::ReadWrite),
            ],
        )
        .await;

        let vol = router.assign(&make_owner(0xAA)).await.unwrap();
        assert_eq!(vol, "vol-002");
    }

    #[tokio::test]
    async fn test_assign_no_writable_volume() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadOnly)]).await;

        let err = router.assign(&make_owner(0xAA)).await.unwrap_err();
        assert!(matches!(err, VolumeRouterError::NoWritableVolume));
    }

    #[tokio::test]
    async fn test_route_returns_cached_assignment() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let owner = make_owner(0xAA);
        let vol = router.assign(&owner).await.unwrap();
        assert_eq!(vol, "vol-001");

        // route() should return the same volume from cache.
        assert_eq!(router.route(&owner), Some("vol-001".to_string()));
    }

    #[tokio::test]
    async fn test_route_unknown_owner_returns_none() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;
        assert!(router.route(&make_owner(0xFF)).is_none());
    }

    // ── Path Generation ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_db_path_format() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let owner = make_owner(0xAB);
        let path = router.db_path("vol-001", &owner).unwrap();

        let expected_prefix = hex::encode(owner)[..16].to_string();
        assert!(
            path.to_string_lossy()
                .ends_with(&format!("{}.db", expected_prefix)),
            "Expected path ending in {}.db, got: {}",
            expected_prefix,
            path.display()
        );
    }

    #[tokio::test]
    async fn test_vec_path_format() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let owner = make_owner(0xCD);
        let path = router.vec_path("vol-001", &owner).unwrap();

        let expected_prefix = hex::encode(owner)[..16].to_string();
        assert!(
            path.to_string_lossy()
                .ends_with(&format!("{}.vec", expected_prefix)),
            "Expected path ending in {}.vec, got: {}",
            expected_prefix,
            path.display()
        );
    }

    #[tokio::test]
    async fn test_db_path_unknown_volume_errors() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let err = router
            .db_path("vol-nonexistent", &make_owner(0xAA))
            .unwrap_err();
        assert!(matches!(err, VolumeRouterError::VolumeNotFound(_)));
    }

    // ── Hot Reload ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_reload_config_adds_new_volume() {
        let dir = TempDir::new().unwrap();
        let (db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        // Can only assign to vol-001 initially.
        assert!(router.has_writable_volume());

        // Rewrite config with a second volume.
        let _ = db; // Keep DB alive.
        write_volumes_toml(
            dir.path(),
            &[
                ("vol-001", VolumeStatus::ReadWrite),
                ("vol-002", VolumeStatus::ReadWrite),
            ],
        );

        router.reload_config().await.unwrap();

        // Now vol-002 should be in config.
        let vols = router.volumes.read();
        assert_eq!(vols.len(), 2);
        assert!(vols.iter().any(|v| v.id == "vol-002"));
    }

    #[tokio::test]
    async fn test_reload_config_status_change() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        // Change vol-001 to read-only.
        write_volumes_toml(dir.path(), &[("vol-001", VolumeStatus::ReadOnly)]);
        router.reload_config().await.unwrap();

        assert!(!router.has_writable_volume());
        let err = router.assign(&make_owner(0xAA)).await.unwrap_err();
        assert!(matches!(err, VolumeRouterError::NoWritableVolume));
    }

    #[tokio::test]
    async fn test_reload_config_preserves_canonical_volume_path() {
        // [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] A config typo must not
        // silently redirect existing owner files to another directory.
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;
        let owner = make_owner(0x42);
        let original_path = router.db_path("vol-001", &owner).unwrap();
        let replacement_dir = dir.path().join("replacement");
        std::fs::create_dir_all(&replacement_dir).unwrap();
        std::fs::write(
            dir.path().join("volumes.toml"),
            format!(
                "[[volumes]]\nid = \"vol-001\"\npath = \"{}\"\nstatus = \"read-only\"\n",
                replacement_dir.to_string_lossy().replace('\\', "/")
            ),
        )
        .unwrap();

        router.reload_config().await.unwrap();

        assert_eq!(router.db_path("vol-001", &owner).unwrap(), original_path);
        assert!(!router.has_writable_volume());
    }

    #[tokio::test]
    async fn test_reload_config_rejects_removing_an_assigned_volume() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(
            dir.path(),
            &[
                ("vol-001", VolumeStatus::ReadWrite),
                ("vol-002", VolumeStatus::ReadWrite),
            ],
        )
        .await;
        let owner = make_owner(0x43);
        router.assign(&owner).await.unwrap();
        let assigned_volume = router.route(&owner).unwrap();
        let retained_path = router.db_path(&assigned_volume, &owner).unwrap();
        let retained_id = assigned_volume.clone();
        let replacement_id = if retained_id == "vol-001" {
            "vol-002"
        } else {
            "vol-001"
        };
        write_volumes_toml(dir.path(), &[(replacement_id, VolumeStatus::ReadWrite)]);

        let error = router.reload_config().await.unwrap_err();

        assert!(matches!(
            error,
            VolumeRouterError::AssignedVolumeUnavailable {
                volume_id,
                assigned_users: 1
            } if volume_id == retained_id
        ));
        assert_eq!(
            router.db_path(&assigned_volume, &owner).unwrap(),
            retained_path
        );
    }

    #[tokio::test]
    async fn test_volume_lock_recovers_after_writer_panic() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _volumes = router.volumes.write();
            panic!("sensitive volume writer failure");
        }));

        assert!(panic_result.is_err());
        assert!(router.has_writable_volume());
        assert!(router
            .db_path("vol-001", &make_owner(0x44))
            .unwrap()
            .ends_with("4444444444444444.db"));
    }

    // ── ensure_volumes_config ─────────────────────────────────────────

    #[tokio::test]
    async fn test_ensure_volumes_config_creates_default() {
        let dir = TempDir::new().unwrap();
        let config_path = ensure_volumes_config(dir.path()).unwrap();

        assert!(config_path.exists());

        // The generated file should be parseable.
        let content = std::fs::read_to_string(&config_path).unwrap();
        let file: VolumesFile = toml::from_str(&content).unwrap();
        assert_eq!(file.volumes.len(), 1);
        assert_eq!(file.volumes[0].id, "vol-001");
        assert_eq!(file.volumes[0].status, VolumeStatus::ReadWrite);
    }

    #[tokio::test]
    async fn test_ensure_volumes_config_does_not_overwrite_existing() {
        let dir = TempDir::new().unwrap();
        let custom_config = dir.path().join("volumes.toml");

        let vol_dir = dir.path().join("volumes").join("custom-vol");
        std::fs::create_dir_all(&vol_dir).unwrap();
        let custom_content = format!(
            "[[volumes]]\nid = \"custom-vol\"\npath = \"{}\"\nstatus = \"read-write\"\n",
            vol_dir.to_string_lossy().replace('\\', "/")
        );
        std::fs::write(&custom_config, &custom_content).unwrap();

        let result = ensure_volumes_config(dir.path()).unwrap();
        assert_eq!(result, custom_config);

        // File should be unchanged.
        let after = std::fs::read_to_string(&custom_config).unwrap();
        assert_eq!(after, custom_content);
    }

    // ── Volume Stats ──────────────────────────────────────────────────

    #[test]
    fn filesystem_usage_probe_counts_only_managed_regular_files() {
        // [VOLUME-USAGE-OBSERVABILITY 2026-08-31 by Codex] The allowlist is
        // intentionally narrower than "all bytes under the volume" so operator
        // files cannot distort MemChain capacity decisions.
        let dir = TempDir::new().unwrap();
        let volume = dir.path().join("volume");
        std::fs::create_dir(&volume).unwrap();
        let owner = "0123456789abcdef";
        std::fs::write(volume.join(format!("{owner}.db")), [0u8; 3]).unwrap();
        std::fs::write(volume.join(format!("{owner}.db-wal")), [0u8; 5]).unwrap();
        std::fs::write(volume.join(format!("{owner}.db-shm")), [0u8; 7]).unwrap();
        std::fs::write(volume.join(format!("{owner}.vec")), [0u8; 11]).unwrap();

        std::fs::write(volume.join("operator-notes.txt"), [0u8; 13]).unwrap();
        std::fs::write(volume.join("not-an-owner.db"), [0u8; 17]).unwrap();
        let nested = volume.join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join(format!("{owner}.db")), [0u8; 19]).unwrap();

        let usage = FilesystemVolumeUsageProbe.usage_bytes(&volume).unwrap();
        assert_eq!(usage, 3 + 5 + 7 + 11);
    }

    #[test]
    fn filesystem_usage_probe_accepts_full_owner_filename_and_empty_volume() {
        let dir = TempDir::new().unwrap();
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        assert_eq!(FilesystemVolumeUsageProbe.usage_bytes(&empty).unwrap(), 0);

        let owner = "ab".repeat(32);
        std::fs::write(empty.join(format!("{owner}.db")), [0u8; 23]).unwrap();
        assert_eq!(FilesystemVolumeUsageProbe.usage_bytes(&empty).unwrap(), 23);
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_usage_probe_never_follows_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let volume = dir.path().join("volume");
        std::fs::create_dir(&volume).unwrap();
        let outside = dir.path().join("outside.bin");
        std::fs::write(&outside, [0u8; 31]).unwrap();
        symlink(&outside, volume.join("0123456789abcdef.db")).unwrap();

        assert_eq!(FilesystemVolumeUsageProbe.usage_bytes(&volume).unwrap(), 0);

        let root_link = dir.path().join("volume-link");
        symlink(&volume, &root_link).unwrap();
        assert!(matches!(
            FilesystemVolumeUsageProbe.usage_bytes(&root_link),
            Err(VolumeUsageProbeError::SymlinkRoot(path)) if path == root_link
        ));
    }

    #[test]
    fn filesystem_usage_probe_reports_io_failure_and_overflow() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("missing-volume");
        assert!(matches!(
            FilesystemVolumeUsageProbe.usage_bytes(&missing),
            Err(VolumeUsageProbeError::Io { path, .. }) if path == missing
        ));

        let overflow_path = dir.path().join("0123456789abcdef.db");
        assert!(matches!(
            checked_add_usage(u64::MAX, 1, &overflow_path),
            Err(VolumeUsageProbeError::Overflow { path }) if path == overflow_path
        ));
    }

    #[tokio::test]
    async fn volume_stats_propagates_typed_probe_failure() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]);
        let router =
            VolumeRouter::new_with_usage_probe(&config_path, db, Arc::new(FailingUsageProbe))
                .await
                .unwrap();

        assert!(matches!(
            router.volume_stats().await,
            Err(VolumeRouterError::VolumeUsage { volume_id, source: VolumeUsageProbeError::Io { .. } })
                if volume_id == "vol-001"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn volume_stats_runs_probe_on_blocking_worker() {
        let async_thread = std::thread::current().id();
        let (sender, receiver) = mpsc::channel();
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]);
        let router = VolumeRouter::new_with_usage_probe(
            &config_path,
            db,
            Arc::new(ThreadReportingUsageProbe { sender, usage: 41 }),
        )
        .await
        .unwrap();

        let stats = router.volume_stats().await.unwrap();
        assert_eq!(stats[0].disk_usage_bytes, 41);
        assert_ne!(receiver.recv().unwrap(), async_thread);
    }

    #[tokio::test]
    async fn test_volume_stats() {
        let dir = TempDir::new().unwrap();
        let (db, router) = make_router(
            dir.path(),
            &[
                ("vol-001", VolumeStatus::ReadWrite),
                ("vol-002", VolumeStatus::ReadOnly),
            ],
        )
        .await;

        // Assign 3 users to vol-001.
        for i in 0u8..3 {
            db.assign_volume(&make_owner(i), "vol-001").await.unwrap();
        }
        let owner = "0123456789abcdef";
        std::fs::write(
            dir.path()
                .join("volumes/vol-001")
                .join(format!("{owner}.db")),
            [0u8; 29],
        )
        .unwrap();

        let stats = router.volume_stats().await.unwrap();
        assert_eq!(stats.len(), 2);

        let v1 = stats.iter().find(|s| s.volume_id == "vol-001").unwrap();
        assert_eq!(v1.user_count, 3);
        assert_eq!(v1.status, VolumeStatus::ReadWrite);
        assert_eq!(v1.disk_usage_bytes, 29);

        let v2 = stats.iter().find(|s| s.volume_id == "vol-002").unwrap();
        assert_eq!(v2.user_count, 0);
        assert_eq!(v2.status, VolumeStatus::ReadOnly);
        assert_eq!(v2.disk_usage_bytes, 0);
        assert_eq!(router.configured_volume_count(), 2);
    }
}
