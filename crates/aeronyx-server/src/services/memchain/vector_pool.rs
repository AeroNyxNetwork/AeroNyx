// ============================================
// File: crates/aeronyx-server/src/services/memchain/vector_pool.rs
// ============================================
//! # VectorIndexPool — Per-User VectorIndex Pool
//!
//! ## Creation Reason
//! Part of the MemChain Multi-Tenant Architecture (v1.0).
//! Manages per-user VectorIndex instances for SaaS mode, parallel to
//! StoragePool. Each user gets their own isolated in-memory vector index.
//!
//! ## Main Functionality
//! - Lazily creates per-user VectorIndex instances on first request
//! - Caches instances in a DashMap for fast subsequent access
//! - Evicts idle instances to control memory usage
//! - Supports scalar quantization and early termination config from MpiState
//!
//! ## VectorIndex Lifecycle
//! VectorIndex is a PURE IN-MEMORY structure — there is no .vec file.
//! On eviction: the Arc<VectorIndex> is dropped from the pool. If the
//!   middleware has already injected it into a request extension, that
//!   request's Arc keeps the index alive until the handler completes.
//! On next get_or_create after eviction: a NEW empty VectorIndex is created.
//!   The Miner's rebuild step (get_records_with_embedding → index.upsert)
//!   will re-populate it on the next Miner tick for that user.
//! This means eviction causes a temporary degradation in vector recall
//! for that user until the next Miner tick. This is acceptable for idle users.
//!
//! ## Dependencies
//! - VolumeRouter (Task 1a): only used to check if owner has been assigned
//!   a volume (if not, StoragePool::get_or_create must be called first)
//! - Used by unified_auth_middleware (Task 2) after StoragePool::get_or_create
//! - Used by MinerScheduler (Task 4) alongside StoragePool
//!
//! ## Calling Contract
//! Always call StoragePool::get_or_create BEFORE VectorIndexPool::get_or_create
//! for the same owner. StoragePool handles volume assignment; VectorIndexPool
//! assumes the owner already has a volume. The auth middleware enforces this order.
//!
//! ⚠️ Important Note for Next Developer:
//! - VectorIndex has NO file persistence — it is rebuilt from DB on each
//!   cold start or after eviction. The Miner's Step 0 (rebuild from SQLite)
//!   handles this automatically.
//! - vec_path() on VolumeRouter is currently unused but reserved for future
//!   persistence support. The path is returned by VolumeRouter::vec_path().
//! - get_or_create() is synchronous (no spawn_blocking needed) because
//!   VectorIndex::new() only allocates an empty HashMap — no I/O.
//! - quantization_enabled and saturation_threshold come from MpiState config.
//!   Pass them during VectorIndexPool::new() construction.
//! - active_count() reflects the in-memory cache size. Users whose index
//!   was evicted are not counted here but remain in SystemDb.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant - Initial implementation (Task 1c)
// ============================================

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::{info, warn};

use super::vector::VectorIndex;
use super::volume_router::{VolumeRouter, VolumeRouterError};

// ============================================
// Public Types
// ============================================

/// Errors from VectorIndexPool operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorPoolError {
    /// Owner has not been assigned to a volume yet.
    /// Callers must invoke StoragePool::get_or_create first.
    #[error(
        "Owner not assigned to any volume — call StoragePool::get_or_create first \
         to trigger volume assignment"
    )]
    NotAssigned,

    /// Volume router returned an unexpected error.
    #[error("Volume router error: {0}")]
    Router(#[from] VolumeRouterError),
}

// ============================================
// Internal Types
// ============================================

struct PoolEntry {
    index: Arc<VectorIndex>,
    last_accessed: Instant,
}

// ============================================
// VectorIndexPool
// ============================================

/// Pool of per-user VectorIndex instances for SaaS multi-tenant mode.
///
/// Each user gets an isolated VectorIndex partitioned by
/// `(owner_hex, embedding_model)`. The pool is backed by a DashMap and
/// supports idle eviction to bound memory usage.
///
/// ## Thread Safety
/// DashMap provides concurrent-safe access. The `entry().or_insert_with()`
/// pattern ensures at most one VectorIndex is created per owner even under
/// concurrent requests.
pub struct VectorIndexPool {
    /// Active index cache: owner pubkey → PoolEntry.
    indexes: DashMap<[u8; 32], PoolEntry>,

    /// Used to verify that an owner has a volume assignment before
    /// creating their index (routing sanity check).
    volume_router: Arc<VolumeRouter>,

    /// How long an index can be idle before eviction.
    idle_timeout: Duration,

    /// Passed to VectorIndex::with_config when creating new instances.
    quantization_enabled: bool,

    /// Early termination saturation threshold for vector search.
    saturation_threshold: f32,
}

impl VectorIndexPool {
    // ============================================
    // Construction
    // ============================================

    /// Create a new VectorIndexPool.
    ///
    /// No VectorIndex instances are created at construction time — they
    /// are lazily initialized on the first `get_or_create` call per user.
    ///
    /// ## Arguments
    /// - `volume_router`: used to verify owner volume assignment
    /// - `idle_timeout`: evict indexes idle longer than this duration
    /// - `quantization_enabled`: passed to VectorIndex::with_config
    /// - `saturation_threshold`: early termination threshold (0.0 = use default)
    pub fn new(
        volume_router: Arc<VolumeRouter>,
        idle_timeout: Duration,
        quantization_enabled: bool,
        saturation_threshold: f32,
    ) -> Arc<Self> {
        Arc::new(Self {
            indexes: DashMap::new(),
            volume_router,
            idle_timeout,
            quantization_enabled,
            saturation_threshold,
        })
    }

    // ============================================
    // Core: Get or Create
    // ============================================

    /// Get the VectorIndex for an owner, creating it if not cached.
    ///
    /// ## Fast path (cache hit)
    /// O(1) DashMap lookup, updates `last_accessed`, returns the Arc.
    ///
    /// ## Slow path (cache miss)
    /// 1. Verify the owner has a volume assignment (route() != None).
    ///    If not assigned, returns `VectorPoolError::NotAssigned`.
    ///    Callers must ensure StoragePool::get_or_create was called first.
    /// 2. Create a new empty VectorIndex with the configured settings.
    /// 3. Insert via `entry().or_insert_with()` to handle concurrent
    ///    creation races — the first writer wins.
    ///
    /// NOTE: The returned index starts empty. Vector data is populated
    /// by the Miner's rebuild step (get_records_with_embedding → upsert).
    ///
    /// # SECURITY
    /// Assumes `owner` has already been authenticated. Never call with
    /// an unverified owner pubkey.
    pub fn get_or_create(
        &self,
        owner: &[u8; 32],
    ) -> Result<Arc<VectorIndex>, VectorPoolError> {
        // ── Fast path ──────────────────────────────────────────────────
        if let Some(mut entry) = self.indexes.get_mut(owner) {
            entry.last_accessed = Instant::now();
            return Ok(Arc::clone(&entry.index));
        }

        // ── Slow path ──────────────────────────────────────────────────

        // Verify the owner has been assigned a volume. If route() returns
        // None, StoragePool::get_or_create was not called first — that is a
        // middleware bug and should be surfaced as an error.
        if self.volume_router.route(owner).is_none() {
            return Err(VectorPoolError::NotAssigned);
        }

        // Create a new empty VectorIndex with the configured settings.
        let index = if self.quantization_enabled || self.saturation_threshold > 0.0 {
            VectorIndex::with_config(self.quantization_enabled, self.saturation_threshold)
        } else {
            VectorIndex::new()
        };

        let index = Arc::new(index);

        // Insert into cache. Use entry().or_insert_with() so that if two
        // concurrent requests race, only one VectorIndex is retained and
        // both callers get the same Arc.
        let final_index = self
            .indexes
            .entry(*owner)
            .or_insert_with(|| PoolEntry {
                index: Arc::clone(&index),
                last_accessed: Instant::now(),
            })
            .index
            .clone();

        Ok(final_index)
    }

    // ============================================
    // Eviction
    // ============================================

    /// Evict idle VectorIndex instances from the pool.
    ///
    /// Called by the Server's eviction timer every 5 minutes, alongside
    /// StoragePool::evict_idle().
    ///
    /// ## Eviction Safety
    /// VectorIndex is pure in-memory with no persistence. Eviction simply
    /// removes the Arc from the pool. If a request currently holds the index
    /// via an injected extension, that Arc stays alive until the handler
    /// returns. The index is re-created empty on the next request and
    /// re-populated by the Miner on its next tick.
    ///
    /// Returns the number of evicted instances (for logging).
    pub fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let mut evicted = 0usize;

        self.indexes.retain(|_owner, entry| {
            if now.duration_since(entry.last_accessed) > self.idle_timeout {
                evicted += 1;
                false
            } else {
                true
            }
        });

        if evicted > 0 {
            info!(
                evicted,
                active = self.indexes.len(),
                "[VECTOR_POOL] Evicted idle vector indexes"
            );
        }

        evicted
    }

    // ============================================
    // Observability
    // ============================================

    /// Number of currently cached (in-memory) VectorIndex instances.
    pub fn active_count(&self) -> usize {
        self.indexes.len()
    }

    /// Owner pubkeys of all currently cached VectorIndex instances.
    ///
    /// NOTE: Reflects the in-memory cache only. Users whose index was
    /// evicted are not listed here but remain registered in SystemDb.
    pub fn cached_owners(&self) -> Vec<[u8; 32]> {
        self.indexes.iter().map(|e| *e.key()).collect()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::system_db::SystemDb;
    use crate::services::memchain::volume_router::VolumeRouter;
    use tempfile::TempDir;

    // ── Test Helpers ──────────────────────────────────────────────────

    fn make_owner(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn write_volumes_toml(dir: &std::path::Path) -> std::path::PathBuf {
        let vol_dir = dir.join("volumes").join("vol-001");
        std::fs::create_dir_all(&vol_dir).unwrap();
        let config_path = dir.join("volumes.toml");
        std::fs::write(
            &config_path,
            format!(
                "[[volumes]]\nid = \"vol-001\"\npath = \"{}\"\nstatus = \"read-write\"\n",
                vol_dir.to_string_lossy().replace('\\', "/")
            ),
        )
        .unwrap();
        config_path
    }

    /// Create a VectorIndexPool backed by a real VolumeRouter.
    async fn setup(dir: &std::path::Path) -> (Arc<SystemDb>, Arc<VolumeRouter>, Arc<VectorIndexPool>) {
        let db = SystemDb::open(&dir.join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir);
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        let pool = VectorIndexPool::new(
            Arc::clone(&router),
            Duration::from_secs(3600),
            false,
            0.0,
        );
        (db, router, pool)
    }

    // ── get_or_create ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_or_create_after_volume_assignment() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;

        let owner = make_owner(0xAA);

        // Assign to a volume first (simulating what StoragePool does).
        db.assign_volume(&owner, "vol-001").await.unwrap();
        router.assignments.insert(owner, "vol-001".into());

        let index = pool.get_or_create(&owner).unwrap();
        assert_eq!(pool.active_count(), 1);
        // New index is empty.
        assert_eq!(index.total_vectors(), 0);
    }

    #[tokio::test]
    async fn test_get_or_create_without_assignment_fails() {
        let dir = TempDir::new().unwrap();
        let (_db, _router, pool) = setup(dir.path()).await;

        let unassigned = make_owner(0xFF);
        let err = pool.get_or_create(&unassigned).unwrap_err();
        assert!(
            matches!(err, VectorPoolError::NotAssigned),
            "Expected NotAssigned, got: {:?}",
            err
        );
    }

    #[tokio::test]
    async fn test_same_owner_returns_same_arc() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;

        let owner = make_owner(0xAA);
        db.assign_volume(&owner, "vol-001").await.unwrap();
        router.assignments.insert(owner, "vol-001".into());

        let i1 = pool.get_or_create(&owner).unwrap();
        let i2 = pool.get_or_create(&owner).unwrap();

        assert!(Arc::ptr_eq(&i1, &i2), "Same owner must return same Arc");
        assert_eq!(pool.active_count(), 1);
    }

    #[tokio::test]
    async fn test_different_owners_different_instances() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;

        for seed in [0xAAu8, 0xBB] {
            let owner = make_owner(seed);
            db.assign_volume(&owner, "vol-001").await.unwrap();
            router.assignments.insert(owner, "vol-001".into());
        }

        let i_a = pool.get_or_create(&make_owner(0xAA)).unwrap();
        let i_b = pool.get_or_create(&make_owner(0xBB)).unwrap();

        assert!(!Arc::ptr_eq(&i_a, &i_b));
        assert_eq!(pool.active_count(), 2);
    }

    // ── Index is functional ───────────────────────────────────────────

    #[tokio::test]
    async fn test_returned_index_is_usable() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;

        let owner = make_owner(0xAA);
        db.assign_volume(&owner, "vol-001").await.unwrap();
        router.assignments.insert(owner, "vol-001".into());

        let index = pool.get_or_create(&owner).unwrap();

        // Upsert a vector and verify search works.
        let embedding = vec![1.0f32, 0.0, 0.0];
        index.upsert(
            [1u8; 32],
            embedding.clone(),
            aeronyx_core::ledger::MemoryLayer::Episode,
            1_000_000,
            &owner,
            "minilm-l6-v2",
        );

        assert_eq!(index.total_vectors(), 1);

        let results = index.search(&embedding, &owner, "minilm-l6-v2", 5, 0.0);
        assert_eq!(results.len(), 1);
    }

    // ── Eviction ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_evict_idle() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        // Very short idle timeout for testing.
        let pool = VectorIndexPool::new(
            Arc::clone(&router),
            Duration::from_millis(50),
            false,
            0.0,
        );

        for seed in [0xAAu8, 0xBB, 0xCC] {
            let owner = make_owner(seed);
            db.assign_volume(&owner, "vol-001").await.unwrap();
            router.assignments.insert(owner, "vol-001".into());
            pool.get_or_create(&owner).unwrap();
        }

        assert_eq!(pool.active_count(), 3);

        // Wait for timeout.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let evicted = pool.evict_idle();
        assert_eq!(evicted, 3);
        assert_eq!(pool.active_count(), 0);
    }

    #[tokio::test]
    async fn test_evict_does_not_remove_recently_accessed() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        // 200ms timeout.
        let pool = VectorIndexPool::new(
            Arc::clone(&router),
            Duration::from_millis(200),
            false,
            0.0,
        );

        let owner_old = make_owner(0xAA);
        let owner_new = make_owner(0xBB);

        db.assign_volume(&owner_old, "vol-001").await.unwrap();
        db.assign_volume(&owner_new, "vol-001").await.unwrap();
        router.assignments.insert(owner_old, "vol-001".into());
        router.assignments.insert(owner_new, "vol-001".into());

        pool.get_or_create(&owner_old).unwrap();

        // Wait so owner_old becomes idle.
        tokio::time::sleep(Duration::from_millis(150)).await;

        // owner_new is accessed recently (within timeout).
        pool.get_or_create(&owner_new).unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Only owner_old should be evicted.
        let evicted = pool.evict_idle();
        assert_eq!(evicted, 1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.cached_owners().contains(&owner_new));
        assert!(!pool.cached_owners().contains(&owner_old));
    }

    // ── Quantization config propagation ──────────────────────────────

    #[tokio::test]
    async fn test_quantization_config_propagated_to_index() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        let pool = VectorIndexPool::new(
            Arc::clone(&router),
            Duration::from_secs(3600),
            true,  // quantization enabled
            0.002, // custom saturation threshold
        );

        let owner = make_owner(0xAA);
        db.assign_volume(&owner, "vol-001").await.unwrap();
        router.assignments.insert(owner, "vol-001".into());

        let index = pool.get_or_create(&owner).unwrap();
        // The index should have quantization enabled.
        assert!(index.quantization_enabled);
    }

    // ── active_count and cached_owners ───────────────────────────────

    #[tokio::test]
    async fn test_active_count() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;

        assert_eq!(pool.active_count(), 0);

        for seed in 0u8..5 {
            let owner = make_owner(seed);
            db.assign_volume(&owner, "vol-001").await.unwrap();
            router.assignments.insert(owner, "vol-001".into());
            pool.get_or_create(&owner).unwrap();
        }

        assert_eq!(pool.active_count(), 5);
    }

    // ── Concurrent creation ───────────────────────────────────────────

    #[tokio::test]
    async fn test_concurrent_get_or_create_same_owner() {
        let dir = TempDir::new().unwrap();
        let (db, router, pool) = setup(dir.path()).await;
        let pool = Arc::clone(&pool);

        let owner = make_owner(0xAA);
        db.assign_volume(&owner, "vol-001").await.unwrap();
        router.assignments.insert(owner, "vol-001".into());

        // 8 concurrent tasks for the same owner.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                tokio::spawn(async move { pool.get_or_create(&owner).unwrap() })
            })
            .collect();

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All should point to the same instance.
        let first = &results[0];
        for r in &results[1..] {
            assert!(Arc::ptr_eq(first, r));
        }
        assert_eq!(pool.active_count(), 1);
    }
}
