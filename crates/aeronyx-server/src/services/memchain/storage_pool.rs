// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_pool.rs
// ============================================
//! # StoragePool — Per-User MemoryStorage Connection Pool
//!
//! ## Creation Reason
//! Part of the MemChain Multi-Tenant Architecture (v1.0).
//! Manages a pool of per-user MemoryStorage instances for SaaS mode,
//! where each user has their own isolated SQLite database file.
//!
//! ## Main Functionality
//! - Lazily opens per-user MemoryStorage instances on first request
//! - Caches open connections in a DashMap for fast subsequent access
//! - Evicts idle connections on a configurable timeout
//! - Enforces a max_connections cap via LRU-style eviction
//! - Derives a per-user encryption key from their Ed25519 public key
//!
//! ## Key Derivation (TENANT ISOLATION)
//! ⚠️ SECURITY: derive_record_key_from_pubkey() uses the PUBLIC key.
//! This is TENANT ISOLATION, NOT end-to-end encryption.
//! The server derives the key and can therefore decrypt the data.
//! This design prevents cross-user data access but does NOT protect
//! against a malicious server operator.
//! See ARCHITECTURE-VALIDATOR-v1.0.md §1.1 for the full threat model.
//!
//! ## Dependencies
//! - VolumeRouter (Task 1a): resolves owner → DB file path
//! - SystemDb (Task 1a): used for volume assignment if owner is new
//! - MemoryStorage (storage.rs): the per-user SQLite storage instance
//! - Used by unified_auth_middleware (Task 2) on every SaaS request
//! - Used by MinerScheduler (Task 4) during cognitive step execution
//!
//! ## Thread Safety
//! DashMap provides concurrent safe access. The `entry().or_try_insert_with()`
//! pattern ensures that for any given owner, at most one MemoryStorage
//! instance is created even under concurrent requests.
//!
//! ⚠️ Important Note for Next Developer:
//! - MemoryStorage::open() is SYNCHRONOUS — always call via spawn_blocking.
//! - derive_record_key_from_pubkey() uses SHA-256 with a domain prefix.
//!   This is DIFFERENT from derive_record_key() in storage_crypto.rs which
//!   uses the PRIVATE key via HKDF. Do NOT mix up these two functions.
//! - The evict_idle() timer is started by Server::new() in SaaS mode.
//!   Do not start it inside StoragePool::new() — that would make testing hard.
//! - cached_owners() returns only in-memory cached owners, NOT all users.
//!   For Miner scheduling, use system_db.get_active_owners() instead.
//! - When max_connections is reached during get_or_create(), the oldest
//!   cached entry is evicted BEFORE opening the new connection, to avoid
//!   briefly exceeding the cap.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant - Initial implementation (Task 1b)
// ============================================

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use sha2::{Digest, Sha256};
use tokio::task;
use tracing::{info, warn};

use super::storage::MemoryStorage;
use super::volume_router::{VolumeRouter, VolumeRouterError};
use super::system_db::SystemDb;

// ============================================
// Public Types
// ============================================

/// Errors that can arise from StoragePool operations.
#[derive(Debug, thiserror::Error)]
pub enum StoragePoolError {
    #[error("No writable volume available for new user: {0}")]
    NoVolume(#[from] VolumeRouterError),

    #[error("Failed to open database at {path}: {reason}")]
    DbOpen { path: String, reason: String },

    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

// ============================================
// Internal Types
// ============================================

struct PoolEntry {
    storage: Arc<MemoryStorage>,
    last_accessed: Instant,
}

// ============================================
// StoragePool
// ============================================

/// Connection pool for per-user MemoryStorage instances.
///
/// Each user (identified by their 32-byte Ed25519 public key) gets their
/// own SQLite database file. This pool manages open connections, evicting
/// idle ones to control memory and file-handle usage.
pub struct StoragePool {
    /// Active connection cache: owner pubkey → PoolEntry.
    stores: DashMap<[u8; 32], PoolEntry>,

    /// Routes owner pubkeys to their volume and DB file path.
    volume_router: Arc<VolumeRouter>,

    /// Global metadata DB — used for volume assignment of new users.
    system_db: Arc<SystemDb>,

    /// Maximum number of simultaneously cached connections.
    max_connections: usize,

    /// How long a connection can be idle before being evicted.
    idle_timeout: Duration,
}

impl StoragePool {
    // ============================================
    // Construction
    // ============================================

    /// Create a new StoragePool.
    ///
    /// No connections are opened at construction time — they are created
    /// lazily on the first `get_or_create` call per user.
    ///
    /// The eviction timer must be started externally (by Server::new) via
    /// a periodic call to `evict_idle()`.
    pub fn new(
        volume_router: Arc<VolumeRouter>,
        system_db: Arc<SystemDb>,
        max_connections: usize,
        idle_timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            stores: DashMap::new(),
            volume_router,
            system_db,
            max_connections,
            idle_timeout,
        })
    }

    // ============================================
    // Core: Get or Create
    // ============================================

    /// Get the MemoryStorage instance for an owner, creating it if needed.
    ///
    /// ## Fast path (cache hit)
    /// Looks up the DashMap. If found, updates `last_accessed` and returns
    /// the existing Arc<MemoryStorage>. This path holds a write lock on the
    /// DashMap entry for only the duration of the `last_accessed` update.
    ///
    /// ## Slow path (cache miss)
    /// 1. If cache is at max capacity, evict the oldest entry first.
    /// 2. Resolve or assign the owner's volume via VolumeRouter.
    /// 3. Derive the user's encryption key from their public key.
    /// 4. Open MemoryStorage via spawn_blocking (SQLite is synchronous).
    /// 5. Insert into DashMap using `entry().or_insert()` to handle the
    ///    race where two concurrent requests create the same user's storage.
    ///    The first writer wins; the second discards its newly-opened instance.
    ///
    /// # SECURITY
    /// Assumes `owner` has already been authenticated by the auth middleware.
    /// Never call with an unverified owner pubkey.
    pub async fn get_or_create(
        &self,
        owner: &[u8; 32],
    ) -> Result<Arc<MemoryStorage>, StoragePoolError> {
        // ── Fast path ──────────────────────────────────────────────────
        if let Some(mut entry) = self.stores.get_mut(owner) {
            entry.last_accessed = Instant::now();
            return Ok(Arc::clone(&entry.storage));
        }

        // ── Slow path ──────────────────────────────────────────────────

        // If at capacity, evict the oldest entry to make room.
        if self.stores.len() >= self.max_connections {
            self.evict_oldest_one();
        }

        // Resolve or assign the owner's volume.
        let volume_id = match self.volume_router.route(owner) {
            Some(vid) => vid,
            None => {
                // New user: assign to the least-loaded writable volume.
                self.volume_router.assign(owner).await?
            }
        };

        // Derive the per-user DB file path.
        let db_path = self.volume_router.db_path(&volume_id, owner)?;

        // Derive the per-user encryption key from their public key.
        // ⚠️ SECURITY: This is TENANT ISOLATION, not E2E encryption.
        // The server holds this key and can decrypt the data.
        let record_key = derive_record_key_from_pubkey(owner);

        // Open MemoryStorage via spawn_blocking (synchronous SQLite call).
        let db_path_clone = db_path.clone();
        let storage = task::spawn_blocking(move || {
            MemoryStorage::open(&db_path_clone, Some(record_key))
        })
        .await?
        .map_err(|reason| StoragePoolError::DbOpen {
            path: db_path.to_string_lossy().into_owned(),
            reason,
        })?;

        let storage = Arc::new(storage);

        // Insert into cache. Use entry() to handle the concurrent-creation race:
        // if another task raced us and already inserted, we use its instance
        // and discard our newly-opened one (it will be dropped here).
        let final_storage = self
            .stores
            .entry(*owner)
            .or_insert_with(|| PoolEntry {
                storage: Arc::clone(&storage),
                last_accessed: Instant::now(),
            })
            .storage
            .clone();

        Ok(final_storage)
    }

    // ============================================
    // Eviction
    // ============================================

    /// Evict idle and excess connections from the pool.
    ///
    /// Called by the Server's eviction timer every 5 minutes.
    ///
    /// ## Phase 1
    /// Remove any entry whose `last_accessed` time exceeds `idle_timeout`.
    ///
    /// ## Phase 2
    /// If the pool is still over `max_connections` after phase 1, evict
    /// the oldest remaining entries (by `last_accessed`) until under the cap.
    ///
    /// Returns the number of evicted connections (for logging).
    pub async fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let mut evicted = 0usize;

        // Phase 1: remove timed-out entries.
        self.stores.retain(|_owner, entry| {
            if now.duration_since(entry.last_accessed) > self.idle_timeout {
                evicted += 1;
                false
            } else {
                true
            }
        });

        // Phase 2: if still over cap, evict oldest entries.
        while self.stores.len() > self.max_connections {
            self.evict_oldest_one();
            evicted += 1;
        }

        if evicted > 0 {
            info!(
                evicted,
                active = self.stores.len(),
                "[STORAGE_POOL] Evicted idle connections"
            );
        }

        evicted
    }

    /// Evict the single entry with the oldest `last_accessed` time.
    ///
    /// Used both by `evict_idle` (phase 2) and by `get_or_create` when
    /// the pool is at capacity before opening a new connection.
    fn evict_oldest_one(&self) {
        // Find the key of the entry with the minimum last_accessed.
        let oldest_key = self
            .stores
            .iter()
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| *entry.key());

        if let Some(key) = oldest_key {
            self.stores.remove(&key);
            warn!(
                owner = &hex::encode(key)[..8],
                active = self.stores.len(),
                "[STORAGE_POOL] Evicted oldest connection to make room"
            );
        }
    }

    // ============================================
    // Observability
    // ============================================

    /// Number of currently cached (open) connections.
    pub fn active_count(&self) -> usize {
        self.stores.len()
    }

    /// Owner pubkeys of all currently cached connections.
    ///
    /// NOTE: This reflects the in-memory cache only, not all registered users.
    /// For Miner scheduling, use `system_db.get_active_owners()` instead.
    pub fn cached_owners(&self) -> Vec<[u8; 32]> {
        self.stores.iter().map(|entry| *entry.key()).collect()
    }
}

// ============================================
// Key Derivation
// ============================================

/// Derive a per-user data encryption key from their Ed25519 public key.
///
/// Algorithm: SHA-256("memchain-record-key-v1" || pubkey)
/// Output: 32-byte symmetric key for AES-256-GCM / ChaCha20-Poly1305.
///
/// ## ⚠️ SECURITY: TENANT ISOLATION, NOT E2E ENCRYPTION
/// This function is executed server-side, so the server knows the key.
/// The purpose is to prevent cross-user data access (user A cannot read
/// user B's data), not to prevent a server operator from reading data.
///
/// ## Why SHA-256 instead of HKDF?
/// The existing derive_record_key() in storage_crypto.rs uses HKDF with
/// the PRIVATE key. This function deliberately uses a different algorithm
/// and input (PUBLIC key) to make the two modes cryptographically independent.
/// Local mode keys and SaaS mode keys are intentionally incompatible.
///
/// ## Compatibility
/// The domain prefix "memchain-record-key-v1" is a version tag.
/// Changing it would invalidate all existing SaaS user databases.
pub fn derive_record_key_from_pubkey(pubkey: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"memchain-record-key-v1");
    hasher.update(pubkey);
    let result = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&result);
    key
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::system_db::SystemDb;
    use crate::services::memchain::volume_router::{VolumeRouter, VolumeStatus};
    use tempfile::TempDir;

    // ── Test Helpers ──────────────────────────────────────────────────

    fn make_owner(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    /// Write a minimal volumes.toml with a single read-write volume.
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

    /// Create a fully initialized test environment.
    async fn setup() -> (TempDir, Arc<StoragePool>) {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            /* max_connections */ 10,
            /* idle_timeout */ Duration::from_secs(3600),
        );

        (dir, pool)
    }

    // ── get_or_create ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_or_create_new_user() {
        let (_dir, pool) = setup().await;
        let owner = make_owner(0xAA);

        let storage = pool.get_or_create(&owner).await.unwrap();
        // Should have created a MemoryStorage instance.
        assert_eq!(pool.active_count(), 1);
        // The storage should be functional (count returns 0 for empty DB).
        assert_eq!(storage.count().await, 0);
    }

    #[tokio::test]
    async fn test_get_or_create_existing_user_returns_same_arc() {
        let (_dir, pool) = setup().await;
        let owner = make_owner(0xBB);

        let s1 = pool.get_or_create(&owner).await.unwrap();
        let s2 = pool.get_or_create(&owner).await.unwrap();

        // Both calls should return the same underlying Arc.
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(pool.active_count(), 1);
    }

    #[tokio::test]
    async fn test_get_or_create_different_users_different_instances() {
        let (_dir, pool) = setup().await;

        let s_a = pool.get_or_create(&make_owner(0xAA)).await.unwrap();
        let s_b = pool.get_or_create(&make_owner(0xBB)).await.unwrap();

        assert!(!Arc::ptr_eq(&s_a, &s_b));
        assert_eq!(pool.active_count(), 2);
    }

    // ── Eviction ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_evict_idle_removes_timed_out_connections() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        // Use a very short idle timeout so we can test eviction quickly.
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_millis(50),
        );

        pool.get_or_create(&make_owner(0xAA)).await.unwrap();
        pool.get_or_create(&make_owner(0xBB)).await.unwrap();
        assert_eq!(pool.active_count(), 2);

        // Wait for idle timeout to expire.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let evicted = pool.evict_idle().await;
        assert_eq!(evicted, 2);
        assert_eq!(pool.active_count(), 0);
    }

    #[tokio::test]
    async fn test_evict_when_full_removes_oldest() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();

        // Pool capped at 2 connections.
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            2,
            Duration::from_secs(3600),
        );

        pool.get_or_create(&make_owner(0xAA)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        pool.get_or_create(&make_owner(0xBB)).await.unwrap();
        assert_eq!(pool.active_count(), 2);

        // Adding a third owner should evict the oldest (0xAA).
        pool.get_or_create(&make_owner(0xCC)).await.unwrap();
        assert_eq!(pool.active_count(), 2);

        // 0xAA should have been evicted; 0xBB and 0xCC remain.
        let cached = pool.cached_owners();
        assert!(!cached.contains(&make_owner(0xAA)));
        assert!(cached.contains(&make_owner(0xBB)) || cached.contains(&make_owner(0xCC)));
    }

    // ── cached_owners ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cached_owners() {
        let (_dir, pool) = setup().await;

        for i in 0u8..3 {
            pool.get_or_create(&make_owner(i)).await.unwrap();
        }

        let owners = pool.cached_owners();
        assert_eq!(owners.len(), 3);
    }

    // ── Key derivation ────────────────────────────────────────────────

    #[test]
    fn test_derive_record_key_deterministic() {
        let pubkey = [0x42u8; 32];
        let k1 = derive_record_key_from_pubkey(&pubkey);
        let k2 = derive_record_key_from_pubkey(&pubkey);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_derive_record_key_different_owners_different_keys() {
        let k1 = derive_record_key_from_pubkey(&[0xAAu8; 32]);
        let k2 = derive_record_key_from_pubkey(&[0xBBu8; 32]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_derive_record_key_different_from_private_key_derivation() {
        use crate::services::memchain::storage_crypto::derive_record_key;
        // The same 32-byte seed used as public key vs private key must
        // produce different output — the two modes are cryptographically
        // independent to prevent accidental key reuse.
        let seed = [0x42u8; 32];
        let saas_key = derive_record_key_from_pubkey(&seed);
        let local_key = derive_record_key(&seed);
        assert_ne!(
            saas_key, local_key,
            "SaaS and local mode keys must be independent"
        );
    }

    // ── Concurrent access ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_concurrent_get_or_create_same_owner() {
        let (_dir, pool) = setup().await;
        let pool = Arc::clone(&pool);
        let owner = make_owner(0xAA);

        // Fire 8 concurrent tasks for the same owner.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                tokio::spawn(async move { pool.get_or_create(&owner).await.unwrap() })
            })
            .collect();

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All results should be the same Arc instance.
        let first = &results[0];
        for r in &results[1..] {
            assert!(Arc::ptr_eq(first, r), "All concurrent calls must return the same Arc");
        }

        // Only one connection should be in the pool.
        assert_eq!(pool.active_count(), 1);
    }
}
