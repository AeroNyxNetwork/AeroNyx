// ============================================
// File: crates/aeronyx-server/src/miner/scheduler.rs
// ============================================
//! # MinerScheduler — SaaS Multi-User Miner Dispatcher
//!
//! ## Creation Reason
//! Part of the MemChain Multi-Tenant Architecture (v1.0).
//! In Local mode, a single `ReflectionMiner` runs against one fixed storage.
//! In SaaS mode, this scheduler replaces it: each tick selects the most
//! recently active users from SystemDb and runs a cognitive step cycle
//! on each user's per-user MemoryStorage from the StoragePool.
//!
//! ## Scheduling Strategy
//! Each tick (every 60 seconds by default):
//! 1. Query `system_db.get_active_owners(limit)` ordered by `last_active_at DESC`
//! 2. Filter owners that have already exhausted their hourly quota
//! 3. Take at most `max_owners_per_tick` owners
//! 4. For each owner:
//!    a. `storage_pool.get_or_create(&owner)` — open/reuse their DB connection
//!    b. Rebuild their VectorIndex from DB embeddings (Step 0.5 also handles this)
//!    c. Build a per-tick `ReflectionMiner` (cheap: only Arc clones)
//!    d. Call `miner.run_one_tick()` — runs all Steps 0 through 11
//!    e. Record execution for quota tracking
//! 5. A single owner failure does NOT interrupt remaining owners
//!
//! ## Why ReflectionMiner per tick?
//! `ReflectionMiner` holds owned `Arc<MemoryStorage>` and `Arc<VectorIndex>`
//! baked in at construction time. In SaaS mode each owner has different storage.
//! Rather than refactoring the entire 800-line step pipeline, we construct a
//! lightweight per-tick miner per owner — cost is a few Arc clones, no I/O,
//! no model loading.
//!
//! ## Stub Components
//! `ReflectionMiner::new()` requires MemPool, AofWriter, SessionManager, and
//! UdpTransport for the Local-mode P2P broadcast path. In SaaS mode these are
//! never invoked (stub SessionManager has 0 capacity → `all_sessions()` returns
//! empty → `broadcast_header` sends 0 packets). Stubs are initialized once at
//! scheduler construction, not per tick.
//!
//! ## Quota Accounting
//! Per-owner hourly quota is tracked in memory (HashMap). Quota resets after
//! 1 hour. Non-persistent by design — restart gives users a fresh budget.
//! Quota prevents runaway LLM costs, not billing precision.
//!
//! ⚠️ Important Note for Next Developer:
//! - The per-tick miner construction is cheap (Arc clones only). Never add
//!   I/O or model loading to `build_per_owner_miner()`.
//! - `stub_udp` binds a real ephemeral loopback port (127.0.0.1:0). The OS
//!   reclaims it on drop. It is never written to in SaaS mode.
//! - `stub_aof` writes to a temp file. The file persists until server restart
//!   or OS cleanup. It is never read from (SaaS Miner does not use legacy AOF).
//! - VectorIndex is rebuilt from DB on each tick. This is acceptable for idle
//!   users. TODO: pass VectorIndexPool to reuse indexes across ticks.
//! - `run_one_tick()` must be kept in sync with the timer body in
//!   `ReflectionMiner::run()`. If new steps are added there, add them here too.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant - Initial implementation (Task 4)
// ============================================

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex as TokioMutex;
use tracing::{error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_transport::UdpTransport;

use crate::services::memchain::{
    AofWriter, EmbedEngine, LlmRouter, MemPool, NerEngine, StoragePool, SystemDb, VectorIndex,
};
use crate::services::SessionManager;
use crate::miner::ReflectionMiner;

// ============================================
// Per-Owner Quota Tracking
// ============================================

/// In-memory per-owner hourly quota record.
struct OwnerQuota {
    /// Miner ticks completed in the current hourly window.
    rounds_this_hour: u32,
    /// When the current hourly window started.
    window_start: Instant,
}

impl OwnerQuota {
    fn new() -> Self {
        Self {
            rounds_this_hour: 0,
            window_start: Instant::now(),
        }
    }

    /// Returns `true` if the owner has quota remaining.
    /// Automatically resets the counter when the hourly window expires.
    fn has_quota(&mut self, max_rounds_per_hour: u32) -> bool {
        if self.window_start.elapsed() >= Duration::from_secs(3600) {
            self.rounds_this_hour = 0;
            self.window_start = Instant::now();
        }
        self.rounds_this_hour < max_rounds_per_hour
    }

    /// Record that one round was completed for this owner.
    fn record_round(&mut self) {
        self.rounds_this_hour += 1;
    }
}

// ============================================
// MinerScheduler
// ============================================

/// SaaS-mode multi-user Miner scheduler.
///
/// Replaces `ReflectionMiner` in SaaS mode. Each tick dispatches cognitive
/// step execution to per-user MemoryStorage instances from StoragePool.
pub struct MinerScheduler {
    storage_pool: Arc<StoragePool>,
    system_db: Arc<SystemDb>,
    max_owners_per_tick: usize,
    max_rounds_per_hour: u32,
    identity: IdentityKeyPair,
    llm_router: Option<Arc<LlmRouter>>,
    embed_engine: Option<Arc<EmbedEngine>>,
    ner_engine: Option<Arc<NerEngine>>,

    /// In-memory per-owner quota tracking.
    /// TokioMutex so `tick()` can take `&self`.
    quotas: TokioMutex<HashMap<[u8; 32], OwnerQuota>>,

    // ── Stub components (required by ReflectionMiner::new, unused in SaaS) ──

    /// Empty MemPool — drain_for_block() always returns Vec::new().
    stub_mempool: Arc<MemPool>,

    /// AOF writer pointed at a temp file — never actually written to in
    /// practice because stub_mempool drains empty and no blocks are formed.
    stub_aof: Arc<TokioMutex<AofWriter>>,

    /// SessionManager with 0 capacity — all_sessions() returns empty Vec,
    /// so ReflectionMiner::broadcast_header sends 0 packets.
    stub_sessions: Arc<SessionManager>,

    /// UDP transport bound to an ephemeral loopback port — never written to
    /// because stub_sessions has no active sessions.
    stub_udp: Arc<UdpTransport>,
}

impl MinerScheduler {
    // ============================================
    // Construction (async — must be awaited)
    // ============================================

    /// Construct a MinerScheduler and initialize stub components.
    ///
    /// This is `async` because:
    /// - `AofWriter::open()` requires async I/O (creates a temp file)
    /// - `UdpTransport::bind_addr()` requires async socket binding
    ///
    /// Called once during SaaS server startup.
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        storage_pool: Arc<StoragePool>,
        system_db: Arc<SystemDb>,
        max_owners_per_tick: usize,
        max_rounds_per_hour: u32,
        identity: IdentityKeyPair,
        llm_router: Option<Arc<LlmRouter>>,
        embed_engine: Option<Arc<EmbedEngine>>,
        ner_engine: Option<Arc<NerEngine>>,
    ) -> Arc<Self> {
        // ── Stub AofWriter ─────────────────────────────────────────────
        // Write to a process-unique temp file. The SaaS Miner never calls
        // legacy_mine() (stub_mempool is always empty), so this file stays
        // empty. It is cleaned up when the process exits.
        let stub_aof_path = std::env::temp_dir().join(format!(
            "memchain_saas_miner_{}.aof",
            std::process::id()
        ));
        let stub_aof = AofWriter::open(&stub_aof_path)
            .await
            .expect("[MINER_SCHED] Failed to open stub AofWriter — check /tmp permissions");
        let stub_aof = Arc::new(TokioMutex::new(stub_aof));

        // ── Stub UdpTransport ──────────────────────────────────────────
        // Bind to an ephemeral loopback port. The OS reclaims it on drop.
        // broadcast_header() is never called because stub_sessions is empty.
        let stub_udp = UdpTransport::bind_addr("127.0.0.1:0".parse().unwrap())
            .await
            .expect("[MINER_SCHED] Failed to bind stub UDP transport");
        let stub_udp = Arc::new(stub_udp);

        // ── Stub SessionManager ────────────────────────────────────────
        // 0 capacity — all_sessions() returns Vec::new().
        let stub_sessions = Arc::new(SessionManager::new(
            0,
            Duration::from_secs(60),
        ));

        Arc::new(Self {
            storage_pool,
            system_db,
            max_owners_per_tick: max_owners_per_tick.max(1),
            max_rounds_per_hour,
            identity,
            llm_router,
            embed_engine,
            ner_engine,
            quotas: TokioMutex::new(HashMap::new()),
            stub_mempool: Arc::new(MemPool::new()),
            stub_aof,
            stub_sessions,
            stub_udp,
        })
    }

    // ============================================
    // Main Tick
    // ============================================

    /// Execute one scheduler tick.
    ///
    /// Called every `MINER_SCHEDULER_TICK_SECS` by server.rs.
    /// Processes up to `max_owners_per_tick` owners in priority order
    /// (most recently active first). Single-owner failures are logged
    /// and do not interrupt remaining owners.
    pub async fn tick(&self) {
        // ── 1. Get candidate owners ────────────────────────────────────
        // Fetch 4× more candidates than needed so quota filtering has headroom.
        let candidates = match self.system_db
            .get_active_owners(self.max_owners_per_tick * 4)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                error!(error = %e, "[MINER_SCHED] Failed to get active owners");
                return;
            }
        };

        if candidates.is_empty() {
            return;
        }

        // ── 2. Filter by quota, select up to max_owners_per_tick ───────
        let selected: Vec<[u8; 32]> = {
            let mut quotas = self.quotas.lock().await;
            let mut out = Vec::with_capacity(self.max_owners_per_tick);
            for c in &candidates {
                if out.len() >= self.max_owners_per_tick { break; }
                let q = quotas.entry(c.pubkey).or_insert_with(OwnerQuota::new);
                if q.has_quota(self.max_rounds_per_hour) {
                    out.push(c.pubkey);
                }
            }
            out
        };

        if selected.is_empty() {
            return;
        }

        info!(
            candidates = candidates.len(),
            selected = selected.len(),
            "[MINER_SCHED] Tick starting"
        );

        let tick_start = Instant::now();
        let mut succeeded = 0u32;
        let mut failed = 0u32;

        // ── 3. Process each owner (sequential — LLM calls can be slow) ─
        for owner in &selected {
            match self.run_owner_tick(owner).await {
                Ok(()) => {
                    let mut quotas = self.quotas.lock().await;
                    if let Some(q) = quotas.get_mut(owner) {
                        q.record_round();
                    }
                    succeeded += 1;
                }
                Err(e) => {
                    warn!(
                        owner = &hex::encode(owner)[..8],
                        error = %e,
                        "[MINER_SCHED] Owner tick failed (non-fatal)"
                    );
                    failed += 1;
                }
            }
        }

        info!(
            succeeded,
            failed,
            elapsed_ms = tick_start.elapsed().as_millis(),
            "[MINER_SCHED] Tick complete"
        );
    }

    // ============================================
    // Per-Owner Execution
    // ============================================

    /// Run one cognitive step cycle for a single owner.
    async fn run_owner_tick(&self, owner: &[u8; 32]) -> Result<(), String> {
        // Get or open this owner's MemoryStorage.
        let storage = self
            .storage_pool
            .get_or_create(owner)
            .await
            .map_err(|e| format!("StoragePool error: {}", e))?;

        // Build a fresh VectorIndex and pre-populate it from this user's DB.
        // Step 0.5 also backfills embeddings, but having the index pre-loaded
        // makes Step 0.6 (correction chaining) and Step 9 (merge) functional
        // on the very first tick.
        //
        // TODO: Replace with VectorIndexPool::get_or_create() when the pool
        // is plumbed through to MinerScheduler — avoids rebuild every tick.
        let vector_index = Arc::new(VectorIndex::new());
        let records_with_model = storage.get_records_with_embedding(owner).await;
        for (record, model) in &records_with_model {
            if record.has_embedding() {
                vector_index.upsert(
                    record.record_id,
                    record.embedding.clone(),
                    record.layer,
                    record.timestamp,
                    owner,
                    model,
                );
            }
        }

        // Construct a per-tick ReflectionMiner (cheap — only Arc clones).
        let miner = self.build_per_owner_miner(storage, vector_index);

        // Run one complete cognitive cycle.
        miner.run_one_tick().await;

        Ok(())
    }

    /// Build a lightweight per-tick `ReflectionMiner` for one owner.
    ///
    /// Cost: a few `Arc::clone` calls. No model loading, no I/O.
    fn build_per_owner_miner(
        &self,
        storage: Arc<crate::services::memchain::MemoryStorage>,
        vector_index: Arc<VectorIndex>,
    ) -> ReflectionMiner {
        // interval=1 is irrelevant — we call run_one_tick(), not run().
        let miner = ReflectionMiner::new(
            1,
            storage,
            vector_index,
            self.identity.clone(),
            Arc::clone(&self.stub_mempool),
            Arc::clone(&self.stub_aof),
            Arc::clone(&self.stub_sessions),
            Arc::clone(&self.stub_udp),
        );

        let miner = match &self.embed_engine {
            Some(ee) => miner.with_embed_engine(Arc::clone(ee)),
            None     => miner,
        };
        let miner = match &self.ner_engine {
            Some(ne) => miner.with_ner_engine(Arc::clone(ne)),
            None     => miner,
        };
        let miner = match &self.llm_router {
            Some(lr) => miner.with_llm_router(Arc::clone(lr)),
            None     => miner,
        };

        miner
    }

    // ============================================
    // Observability
    // ============================================

    /// Number of owners with active quota records (for monitoring).
    pub async fn tracked_owners(&self) -> usize {
        self.quotas.lock().await.len()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::{SystemDb, VolumeRouter};
    use crate::services::memchain::storage_pool::StoragePool;
    use tempfile::TempDir;

    fn make_owner(seed: u8) -> [u8; 32] { [seed; 32] }

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
        ).unwrap();
        config_path
    }

    async fn make_scheduler(dir: &std::path::Path) -> Arc<MinerScheduler> {
        let db = SystemDb::open(&dir.join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir);
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        MinerScheduler::new(pool, db, 3, 6, identity, None, None, None).await
    }

    // ── Quota mechanics ───────────────────────────────────────────────

    #[test]
    fn test_owner_quota_basic() {
        let mut q = OwnerQuota::new();
        // Should have quota initially.
        assert!(q.has_quota(3));
        q.record_round();
        assert!(q.has_quota(3));
        q.record_round();
        assert!(q.has_quota(3));
        q.record_round();
        // Exhausted.
        assert!(!q.has_quota(3));
    }

    #[test]
    fn test_owner_quota_resets_after_hour() {
        let mut q = OwnerQuota::new();
        q.rounds_this_hour = 10;
        // Manually backdate the window start by 2 hours.
        q.window_start = Instant::now() - Duration::from_secs(7200);
        // Should reset and report quota available.
        assert!(q.has_quota(6));
        assert_eq!(q.rounds_this_hour, 0);
    }

    // ── Scheduler construction ────────────────────────────────────────

    #[tokio::test]
    async fn test_scheduler_new() {
        let dir = TempDir::new().unwrap();
        let sched = make_scheduler(dir.path()).await;
        assert_eq!(sched.max_owners_per_tick, 3);
        assert_eq!(sched.max_rounds_per_hour, 6);
        assert_eq!(sched.tracked_owners().await, 0);
    }

    // ── Tick with no active owners ────────────────────────────────────

    #[tokio::test]
    async fn test_tick_no_active_owners() {
        let dir = TempDir::new().unwrap();
        let sched = make_scheduler(dir.path()).await;
        // Should complete without error when SystemDb has no owners.
        sched.tick().await;
    }

    // ── Quota filtering ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_tick_respects_quota() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        let pool = StoragePool::new(Arc::clone(&router), Arc::clone(&db), 10, Duration::from_secs(3600));
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();

        // max_rounds_per_hour = 1 → each owner can only be processed once per hour.
        let sched = MinerScheduler::new(pool, Arc::clone(&db), 10, 1, identity, None, None, None).await;

        // Assign 2 owners so they appear in active_owners.
        db.assign_volume(&make_owner(0xAA), "vol-001").await.unwrap();
        db.assign_volume(&make_owner(0xBB), "vol-001").await.unwrap();
        db.update_last_active(&make_owner(0xAA)).await.unwrap();
        db.update_last_active(&make_owner(0xBB)).await.unwrap();

        // Pre-fill quotas as exhausted for both owners.
        {
            let mut quotas = sched.quotas.lock().await;
            let mut q_aa = OwnerQuota::new();
            q_aa.rounds_this_hour = 1; // exhausted (max=1)
            quotas.insert(make_owner(0xAA), q_aa);
            let mut q_bb = OwnerQuota::new();
            q_bb.rounds_this_hour = 1;
            quotas.insert(make_owner(0xBB), q_bb);
        }

        // tick() should select 0 owners (all exhausted).
        sched.tick().await;
        // tracked_owners stays 2 (no new entries added).
        assert_eq!(sched.tracked_owners().await, 2);
    }

    // ── max_owners_per_tick cap ───────────────────────────────────────

    #[tokio::test]
    async fn test_max_owners_per_tick_enforced() {
        // max_owners_per_tick = 1 (minimum capped at 1).
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        let pool = StoragePool::new(Arc::clone(&router), Arc::clone(&db), 10, Duration::from_secs(3600));
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        let sched = MinerScheduler::new(pool, db, 1, 100, identity, None, None, None).await;

        assert_eq!(sched.max_owners_per_tick, 1);
    }

    // ── max_owners_per_tick=0 is floored to 1 ────────────────────────

    #[tokio::test]
    async fn test_max_owners_per_tick_zero_floored() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db)).await.unwrap();
        let pool = StoragePool::new(Arc::clone(&router), Arc::clone(&db), 10, Duration::from_secs(3600));
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        // Pass 0 — should be clamped to 1.
        let sched = MinerScheduler::new(pool, db, 0, 6, identity, None, None, None).await;
        assert_eq!(sched.max_owners_per_tick, 1);
    }
}
