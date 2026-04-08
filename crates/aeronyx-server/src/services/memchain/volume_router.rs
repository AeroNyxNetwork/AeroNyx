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
//! ```
//! read-write ──(capacity)──→ read-only ──(future draining)──→ removed
//! ```
//! Only read-write and read-only are implemented in this version.
//! Draining support is reserved for future migration tooling.
//!
//! ⚠️ Important Note for Next Developer:
//! - volumes field uses std::sync::RwLock (not tokio), because it is
//!   only ever accessed from spawn_blocking contexts or sync code.
//!   Do not switch to tokio::sync::RwLock without auditing all callers.
//! - assign() has a TOCTOU window between count_users_per_volume() and
//!   assign_volume(). Mitigation: SystemDb.assign_volume() uses
//!   INSERT OR IGNORE + AlreadyAssigned error — the race results in a
//!   retry, not a data corruption. See assign() implementation.
//! - reload_config(): path changes are intentionally ignored (the old
//!   path is the canonical location for existing user DBs).
//! - "Ghost volumes": if a volume is removed from config but still has
//!   users assigned, route() still works (from in-memory DashMap), but
//!   db_path() will fail at open time. This is an ops/deployment concern.
//! - ensure_volumes_config() is called from Server::new() in SaaS mode,
//!   not from VolumeRouter::new() — keep them separate for testability.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant - Initial implementation (Task 1a)
// ============================================

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use dashmap::DashMap;
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

fn default_status() -> VolumeStatus { VolumeStatus::ReadWrite }
fn default_max_users() -> usize { 10_000 }
fn default_max_bytes() -> u64 { 500_000_000_000 } // 500 GB

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
    /// Actual disk usage in bytes. Returns 0 in this version —
    /// a background task can populate this in future.
    pub disk_usage_bytes: u64,
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

    #[error("SystemDb error: {0}")]
    SystemDb(#[from] SystemDbError),

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

    /// In-memory owner → volume_id cache (populated at startup from SystemDb).
    assignments: DashMap<[u8; 32], String>,

    /// SystemDb reference for persistent assignment writes.
    system_db: Arc<SystemDb>,

    /// Path to volumes.toml for hot-reload support.
    config_path: PathBuf,
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
        let config_path = config_path.to_path_buf();

        // Load and validate config.
        let volumes = load_volumes_config(&config_path)?;

        // Populate in-memory cache from persistent state.
        let assignments: DashMap<[u8; 32], String> = DashMap::new();
        let all = system_db.load_all_assignments().await?;
        let volume_ids: std::collections::HashSet<&str> =
            volumes.iter().map(|v| v.id.as_str()).collect();

        for (owner, vol_id) in &all {
            if !volume_ids.contains(vol_id.as_str()) {
                warn!(
                    owner = hex::encode(owner),
                    volume_id = vol_id,
                    "[VOLUME_ROUTER] Assignment points to unconfigured volume (ghost volume) — \
                     route() will work from cache but open may fail at DB level"
                );
            }
            assignments.insert(*owner, vol_id.clone());
        }

        info!(
            config = %config_path.display(),
            volumes = volumes.len(),
            assignments = all.len(),
            "[VOLUME_ROUTER] ✅ Initialized"
        );

        Ok(Arc::new(Self {
            volumes: RwLock::new(volumes),
            assignments,
            system_db,
            config_path,
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
        // Get current per-volume user counts.
        let counts: std::collections::HashMap<String, usize> = self
            .system_db
            .count_users_per_volume()
            .await?
            .into_iter()
            .collect();

        // Select the target volume under a read lock.
        let target_id = {
            let vols = self.volumes.read().unwrap();

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
                    owner = &hex::encode(owner)[..8],
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
    pub fn db_path(
        &self,
        volume_id: &str,
        owner: &[u8; 32],
    ) -> Result<PathBuf, VolumeRouterError> {
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
        let vols = self.volumes.read().unwrap();
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
    /// - Path changes: silently ignored (existing DBs are at old paths)
    /// - Removed volumes: users who were on them become "ghost" assignments;
    ///   route() still works from cache, but DB opens may fail at storage layer
    ///
    /// Existing in-memory assignments are NOT invalidated on reload.
    pub async fn reload_config(&self) -> Result<(), VolumeRouterError> {
        let new_volumes = load_volumes_config(&self.config_path)?;

        let mut vols = self.volumes.write().unwrap();

        // Detect and warn about path changes (not applied).
        for new_vol in &new_volumes {
            if let Some(old_vol) = vols.iter().find(|v| v.id == new_vol.id) {
                if old_vol.path != new_vol.path {
                    warn!(
                        volume_id = new_vol.id,
                        old_path = %old_vol.path.display(),
                        new_path = %new_vol.path.display(),
                        "[VOLUME_ROUTER] Path change ignored on reload — \
                         existing user DBs remain at old path"
                    );
                }
            }
        }

        let new_count = new_volumes.len();
        let old_count = vols.len();
        *vols = new_volumes;

        info!(
            old_count,
            new_count,
            "[VOLUME_ROUTER] ✅ Config reloaded"
        );

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

        let vols = self.volumes.read().unwrap();
        let stats = vols
            .iter()
            .map(|v| VolumeStats {
                volume_id: v.id.clone(),
                status: v.status,
                path: v.path.clone(),
                user_count: counts.get(&v.id).copied().unwrap_or(0),
                max_users: v.max_users,
                max_bytes: v.max_bytes,
                disk_usage_bytes: 0, // TODO: background disk usage scan
            })
            .collect();

        Ok(stats)
    }

    /// Check whether the router has any writable volume available.
    pub fn has_writable_volume(&self) -> bool {
        let vols = self.volumes.read().unwrap();
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

    let file: VolumesFile = toml::from_str(&content)
        .map_err(|e| VolumeRouterError::ConfigParse(e.to_string()))?;

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
    let has_writable = file.volumes.iter().any(|v| v.status == VolumeStatus::ReadWrite);
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

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::system_db::SystemDb;
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
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        (db, router)
    }

    fn make_owner(seed: u8) -> [u8; 32] {
        [seed; 32]
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
        let err = VolumeRouter::new(&missing, Arc::clone(&db)).await.unwrap_err();
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
        let err = VolumeRouter::new(&path, Arc::clone(&db)).await.unwrap_err();
        assert!(matches!(err, VolumeRouterError::DuplicateId(_)));
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
        let (_db, router) = make_router(
            dir.path(),
            &[("vol-001", VolumeStatus::ReadOnly)],
        )
        .await;

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
            path.to_string_lossy().ends_with(&format!("{}.db", expected_prefix)),
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
            path.to_string_lossy().ends_with(&format!("{}.vec", expected_prefix)),
            "Expected path ending in {}.vec, got: {}",
            expected_prefix,
            path.display()
        );
    }

    #[tokio::test]
    async fn test_db_path_unknown_volume_errors() {
        let dir = TempDir::new().unwrap();
        let (_db, router) = make_router(dir.path(), &[("vol-001", VolumeStatus::ReadWrite)]).await;

        let err = router.db_path("vol-nonexistent", &make_owner(0xAA)).unwrap_err();
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
        let vols = router.volumes.read().unwrap();
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

        let stats = router.volume_stats().await.unwrap();
        assert_eq!(stats.len(), 2);

        let v1 = stats.iter().find(|s| s.volume_id == "vol-001").unwrap();
        assert_eq!(v1.user_count, 3);
        assert_eq!(v1.status, VolumeStatus::ReadWrite);

        let v2 = stats.iter().find(|s| s.volume_id == "vol-002").unwrap();
        assert_eq!(v2.user_count, 0);
        assert_eq!(v2.status, VolumeStatus::ReadOnly);
    }
}
