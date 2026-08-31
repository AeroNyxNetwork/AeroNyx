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
//! - [MINER-STARTUP-ERROR 2026-08-12 by Codex] Stub resource creation is a
//!   fallible startup boundary. Propagate its typed error to `Server::run`;
//!   never restore `expect` or allow a partial scheduler to start.
//! - VectorIndex is rebuilt from DB on each tick. This is acceptable for idle
//!   users. TODO: pass VectorIndexPool to reuse indexes across ticks.
//! - `run_one_tick()` must be kept in sync with the timer body in
//!   `ReflectionMiner::run()`. If new steps are added there, add them here too.
//! - Managed-volume reads and vector rebuild happen before byte-growth
//!   admission. The permit then covers the complete mutation-bearing miner
//!   tick; a failed admission does not consume the owner's hourly quota.
//! - [MINER-OWNER-CONTEXT 2026-08-31 by Codex] `ActiveOwner.pubkey` is the
//!   authoritative tenant storage owner. It never replaces the node identity
//!   used for decryption, commitment authority, or signatures.
//!
//! ## Last Modified
//! v1.0.3-MinerOwnerContext - Inject authoritative tenant ownership and account
//!                            retryable signed-compaction skips explicitly.
//! v1.0.2-ManagedVolumeGrowth - Fail closed before per-owner miner mutations.
//! v1.0.1-MinerStartupError - Made stub AOF/socket initialization fallible and
//!                            covered inaccessible state paths.
//! v1.0.0-MultiTenant - Initial implementation (Task 4)
// ============================================

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex as TokioMutex;
use tracing::{error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_transport::UdpTransport;

use crate::error::{Result as ServerResult, ServerError};
use crate::miner::reflection::{MinerOwnerContext, MinerTickOutcome};
use crate::miner::ReflectionMiner;
use crate::services::memchain::{
    AofWriter, EmbedEngine, LlmRouter, MemPool, NerEngine, StoragePool, SystemDb, VectorIndex,
};
use crate::services::SessionManager;

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
    ) -> ServerResult<Arc<Self>> {
        // ── Stub AofWriter ─────────────────────────────────────────────
        // Write to a process-unique temp file. The SaaS Miner never calls
        // legacy_mine() (stub_mempool is always empty), so this file stays
        // empty. It is cleaned up when the process exits.
        let stub_aof_path =
            std::env::temp_dir().join(format!("memchain_saas_miner_{}.aof", std::process::id()));
        let (stub_aof, stub_udp) = Self::initialize_stub_runtime(&stub_aof_path).await?;

        // ── Stub SessionManager ────────────────────────────────────────
        // 0 capacity — all_sessions() returns Vec::new().
        let stub_sessions = Arc::new(SessionManager::new(0, Duration::from_secs(60)));

        Ok(Arc::new(Self {
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
        }))
    }

    async fn initialize_stub_runtime(
        stub_aof_path: &Path,
    ) -> ServerResult<(Arc<TokioMutex<AofWriter>>, Arc<UdpTransport>)> {
        // [MINER-STARTUP-ERROR 2026-08-12 by Codex] These resources are
        // process infrastructure even though they carry no user data. Return
        // a contextual startup error so the top-level runtime can unwind all
        // previously acquired resources and report a truthful failed start.
        let stub_aof = AofWriter::open(stub_aof_path).await.map_err(|error| {
            ServerError::startup_failed(format!(
                "SaaS miner stub AOF initialization failed: {error}"
            ))
        })?;
        let stub_udp_address = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
        let stub_udp = UdpTransport::bind_addr(stub_udp_address)
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "SaaS miner loopback transport initialization failed: {error}"
                ))
            })?;

        Ok((Arc::new(TokioMutex::new(stub_aof)), Arc::new(stub_udp)))
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
        let candidates = match self
            .system_db
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
                if out.len() >= self.max_owners_per_tick {
                    break;
                }
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
        let mut retryable_skipped = 0u32;
        let mut failed = 0u32;

        // ── 3. Process each owner (sequential — LLM calls can be slow) ─
        for owner in &selected {
            match self.run_owner_tick(owner).await {
                Ok(MinerTickOutcome::Completed) => {
                    let mut quotas = self.quotas.lock().await;
                    if let Some(q) = quotas.get_mut(owner) {
                        q.record_round();
                    }
                    succeeded += 1;
                }
                Ok(MinerTickOutcome::RetryableSkip(reason)) => {
                    // [MINER-OWNER-CONTEXT 2026-08-31 by Codex] The work is
                    // intentionally retryable and must not consume quota or be
                    // reported as successful distillation.
                    warn!(?reason, "[MINER_SCHED] Owner tick deferred (retryable)");
                    retryable_skipped += 1;
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "[MINER_SCHED] Owner tick failed (non-fatal)"
                    );
                    failed += 1;
                }
            }
        }

        info!(
            succeeded,
            retryable_skipped,
            failed,
            elapsed_ms = tick_start.elapsed().as_millis(),
            "[MINER_SCHED] Tick complete"
        );
    }

    // ============================================
    // Per-Owner Execution
    // ============================================

    /// Run one cognitive step cycle for a single owner.
    async fn run_owner_tick(&self, owner: &[u8; 32]) -> Result<MinerTickOutcome, String> {
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

        // [VOLUME-GROWTH-ADMISSION 2026-08-31 by Codex] Reads and recovery
        // above remain available at capacity. Hold the same-volume permit only
        // across the mutation-bearing tick; failure is retryable by the next
        // scheduler round and returns before quota accounting.
        let _growth_permit = storage
            .acquire_growth_permit(1)
            .await
            .map_err(|error| format!("Storage growth admission failed: {error}"))?;

        // Construct a per-tick ReflectionMiner (cheap — only Arc clones).
        let miner = self.build_per_owner_miner(storage, vector_index, *owner);

        // Run one complete cognitive cycle.
        Ok(miner.run_one_tick_with_outcome().await)
    }

    /// Build a lightweight per-tick `ReflectionMiner` for one owner.
    ///
    /// Cost: a few `Arc::clone` calls. No model loading, no I/O.
    fn build_per_owner_miner(
        &self,
        storage: Arc<crate::services::memchain::MemoryStorage>,
        vector_index: Arc<VectorIndex>,
        storage_owner: [u8; 32],
    ) -> ReflectionMiner {
        // interval=1 is irrelevant — we call run_one_tick(), not run().
        // [MINER-OWNER-CONTEXT 2026-08-31 by Codex] The selected ActiveOwner
        // owns every tenant-scoped DB row and vector partition in this tick.
        let miner = ReflectionMiner::new(
            1,
            storage,
            vector_index,
            self.identity.clone(),
            Arc::clone(&self.stub_mempool),
            Arc::clone(&self.stub_aof),
            Arc::clone(&self.stub_sessions),
            Arc::clone(&self.stub_udp),
        )
        .with_owner_context(MinerOwnerContext::for_storage_owner(storage_owner));

        let miner = match &self.embed_engine {
            Some(ee) => miner.with_embed_engine(Arc::clone(ee)),
            None => miner,
        };
        let miner = match &self.ner_engine {
            Some(ne) => miner.with_ner_engine(Arc::clone(ne)),
            None => miner,
        };
        let miner = match &self.llm_router {
            Some(lr) => miner.with_llm_router(Arc::clone(lr)),
            None => miner,
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
    use crate::services::memchain::storage_pool::StoragePool;
    use crate::services::memchain::volume_router::{VolumeUsageProbe, VolumeUsageProbeError};
    use crate::services::memchain::{SystemDb, VolumeRouter};
    use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};
    use tempfile::TempDir;

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

    fn write_volumes_toml_with_max_bytes(
        dir: &std::path::Path,
        max_bytes: u64,
    ) -> std::path::PathBuf {
        let vol_dir = dir.join("volumes").join("vol-001");
        std::fs::create_dir_all(&vol_dir).unwrap();
        let config_path = dir.join("volumes.toml");
        std::fs::write(
            &config_path,
            format!(
                "[[volumes]]\nid = \"vol-001\"\npath = \"{}\"\nstatus = \"read-write\"\nmax_bytes = {}\n",
                vol_dir.to_string_lossy().replace('\\', "/"),
                max_bytes
            ),
        )
        .unwrap();
        config_path
    }

    struct FixedUsageProbe(u64);

    impl VolumeUsageProbe for FixedUsageProbe {
        fn usage_bytes(
            &self,
            _volume_root: &std::path::Path,
        ) -> Result<u64, VolumeUsageProbeError> {
            Ok(self.0)
        }
    }

    async fn make_scheduler(dir: &std::path::Path) -> Arc<MinerScheduler> {
        let db = SystemDb::open(&dir.join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir);
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        MinerScheduler::new(pool, db, 3, 6, identity, None, None, None)
            .await
            .unwrap()
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

    #[tokio::test]
    async fn capacity_failure_writes_nothing_and_does_not_consume_quota_round() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let owner = make_owner(0xC1);
        db.assign_volume(&owner, "vol-001").await.unwrap();
        db.update_last_active(&owner).await.unwrap();
        let config_path = write_volumes_toml_with_max_bytes(dir.path(), 100);
        let router = VolumeRouter::new_with_usage_probe(
            &config_path,
            Arc::clone(&db),
            Arc::new(FixedUsageProbe(100)),
        )
        .await
        .unwrap();
        let pool = StoragePool::new(router, Arc::clone(&db), 10, Duration::from_secs(3600));
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        let scheduler = MinerScheduler::new(pool, db, 1, 6, identity, None, None, None)
            .await
            .unwrap();

        scheduler.tick().await;

        let quotas = scheduler.quotas.lock().await;
        assert_eq!(quotas.get(&owner).unwrap().rounds_this_hour, 0);
    }

    #[tokio::test]
    async fn saas_tick_binds_feedback_and_vector_queries_to_scheduled_owner() {
        // [MINER-OWNER-CONTEXT 2026-08-31 by Codex] Reproduces the former
        // cross-owner write: the scheduler node and authenticated tenant are
        // deliberately different identities.
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let node_identity = IdentityKeyPair::generate();
        let node_owner = node_identity.public_key_bytes();
        let tenant_identity = IdentityKeyPair::generate();
        let tenant_owner = tenant_identity.public_key_bytes();

        db.assign_volume(&tenant_owner, "vol-001").await.unwrap();
        db.update_last_active(&tenant_owner).await.unwrap();
        let storage = pool.get_or_create(&tenant_owner).await.unwrap();

        let mut recalled = MemoryRecord::new(
            tenant_owner,
            1_700_000_000,
            MemoryLayer::Episode,
            vec!["reference".into()],
            "test".into(),
            b"recalled tenant record".to_vec(),
            vec![1.0, 0.0],
        );
        recalled.signature = tenant_identity.sign(&recalled.record_id);
        assert!(storage.insert(&recalled, "minilm-l6-v2").await);

        let mut correction = MemoryRecord::new(
            tenant_owner,
            1_700_000_001,
            MemoryLayer::Episode,
            vec!["_correction".into()],
            "test".into(),
            b"corrected tenant record".to_vec(),
            vec![1.0, 0.0],
        );
        correction.signature = tenant_identity.sign(&correction.record_id);
        assert!(storage.insert(&correction, "minilm-l6-v2").await);

        let recall_context = serde_json::json!([{
            "id": hex::encode(recalled.record_id),
            "score": 0.99
        }])
        .to_string();
        storage
            .insert_raw_log(
                "tenant-session",
                1,
                "user",
                "this recalled answer was genuinely useful",
                "test",
                Some(&recall_context),
                1,
                None,
                None,
            )
            .await
            .unwrap();

        let scheduler =
            MinerScheduler::new(pool, Arc::clone(&db), 1, 6, node_identity, None, None, None)
                .await
                .unwrap();
        scheduler.tick().await;

        let (feedback_owner, node_feedback_count, recalled_status) = {
            let conn = storage.conn_lock().await;
            let feedback_owner: Vec<u8> = conn
                .query_row("SELECT owner FROM memory_feedback LIMIT 1", [], |row| {
                    row.get(0)
                })
                .unwrap();
            let node_feedback_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memory_feedback WHERE owner = ?1",
                    rusqlite::params![node_owner.as_slice()],
                    |row| row.get(0),
                )
                .unwrap();
            let recalled_status: i64 = conn
                .query_row(
                    "SELECT status FROM records WHERE record_id = ?1",
                    rusqlite::params![recalled.record_id.as_slice()],
                    |row| row.get(0),
                )
                .unwrap();
            (feedback_owner, node_feedback_count, recalled_status)
        };

        assert_eq!(feedback_owner, tenant_owner);
        assert_eq!(node_feedback_count, 0);
        // Superseding proves Step 0.6 searched the tenant vector partition.
        assert_eq!(recalled_status, 1);
        assert_eq!(
            scheduler
                .quotas
                .lock()
                .await
                .get(&tenant_owner)
                .unwrap()
                .rounds_this_hour,
            1
        );
    }

    // ── Quota filtering ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_tick_respects_quota() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();

        // max_rounds_per_hour = 1 → each owner can only be processed once per hour.
        let sched = MinerScheduler::new(pool, Arc::clone(&db), 10, 1, identity, None, None, None)
            .await
            .unwrap();

        // Assign 2 owners so they appear in active_owners.
        db.assign_volume(&make_owner(0xAA), "vol-001")
            .await
            .unwrap();
        db.assign_volume(&make_owner(0xBB), "vol-001")
            .await
            .unwrap();
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
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        let sched = MinerScheduler::new(pool, db, 1, 100, identity, None, None, None)
            .await
            .unwrap();

        assert_eq!(sched.max_owners_per_tick, 1);
    }

    // ── max_owners_per_tick=0 is floored to 1 ────────────────────────

    #[tokio::test]
    async fn test_max_owners_per_tick_zero_floored() {
        let dir = TempDir::new().unwrap();
        let db = SystemDb::open(&dir.path().join("system.db")).await.unwrap();
        let config_path = write_volumes_toml(dir.path());
        let router = VolumeRouter::new(&config_path, Arc::clone(&db))
            .await
            .unwrap();
        let pool = StoragePool::new(
            Arc::clone(&router),
            Arc::clone(&db),
            10,
            Duration::from_secs(3600),
        );
        let identity = aeronyx_core::crypto::IdentityKeyPair::generate();
        // Pass 0 — should be clamped to 1.
        let sched = MinerScheduler::new(pool, db, 0, 6, identity, None, None, None)
            .await
            .unwrap();
        assert_eq!(sched.max_owners_per_tick, 1);
    }

    #[tokio::test]
    async fn scheduler_stub_aof_failure_returns_typed_startup_error() {
        // [MINER-STARTUP-ERROR 2026-08-12 by Codex] A non-directory parent
        // deterministically reproduces an unavailable state path without
        // mutating process-global TMPDIR or requiring privileged filesystem
        // permissions. The constructor boundary must return, never panic.
        let dir = TempDir::new().unwrap();
        let non_directory = dir.path().join("not-a-directory");
        std::fs::write(&non_directory, b"occupied").unwrap();
        let invalid_path = non_directory.join("stub.aof");

        let error = match MinerScheduler::initialize_stub_runtime(&invalid_path).await {
            Ok(_) => panic!("invalid stub AOF parent unexpectedly initialized"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("SaaS miner stub AOF initialization failed"));
    }
}
