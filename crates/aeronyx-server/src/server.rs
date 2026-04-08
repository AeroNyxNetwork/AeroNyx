// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
//! # Server Orchestrator
//!
//! ## Modification History
//! v0.1.0 - Initial server implementation
//! v0.1.1 - Keepalive packet handling
//! v0.2.0 - CMS management integration
//! v0.3.0 - MemChain integration (MemPool, AofWriter, 1st-byte dispatch)
//! v0.3.1 - Fixed Option<Arc<…>> type mismatch
//! v0.4.0 - Ed25519 verify, trust whitelist, SyncReq/SyncRes
//! v1.3.0 - Command Pipeline: CommandHandler + channel wiring
//! v2.1.0 - Dual-engine init (SQLite+Vector + legacy MemPool+AOF),
//!           MPI router merged, BroadcastRecord/SyncRecordRequest handling,
//!           vector index rebuild on startup, new Miner 8-arg signature.
//! v2.1.1 - Fixed duplicate LinuxTun import, removed unused `trace` import
//! v2.1.0+MVF+Encryption - MemoryStorage.open() now receives record_key
//!   derived from Ed25519 private key for transparent record content encryption.
//!   MpiState now receives api_secret from config for Bearer token auth.
//! v2.3.0+RemoteStorage - MpiState now receives allow_remote_storage and
//!   max_remote_owners from config for Phase 1 remote MPI Gateway support.
//! v2.4.0-GraphCognition - NerEngine initialization + MpiState extension.
//! v2.4.0-GraphCognition Phase B - Scalar quantization integration.
//! v2.4.0+Reranker - RerankerEngine initialization.
//! v2.4.0+Conversation - derive_rawlog_key import and rawlog_key field in MpiState.
//! v2.5.0+SuperNode - LlmRouter initialization + MpiState.llm_router field.
//!   - init_llm_router() reads supernode config, constructs providers + router
//!   - reset_stale_processing_tasks() called at startup before TaskWorker spawn
//!   - TaskWorker spawned as background task when supernode.enabled=true
//!   - ReflectionMiner receives .with_llm_router() for Steps 8/9/10 enqueue
//!   - All SuperNode paths gated on is_supernode_enabled() — safe to disable
//! v2.5.2+SecAudit - BroadcastRecord/SyncRecordResponse: signature now
//!   verified against record content hash (record.content_hash_bytes()) rather
//!   than record_id alone, preventing spoofed-content replay attacks.
//!   public_ip resolution: metadata URLs restricted to HTTPS only + IP validation
//!   tightened to reject private/loopback addresses as fallback detection.
//! v2.5.3+Security - Server::new() gains config_path: Option<PathBuf> (3rd arg).
//!   First SaaS startup auto-generates api_secret and jwt_secret and writes
//!   them back to the config file via ensure_api_secret() + ensure_jwt_secret().
//! v1.0.0-MultiTenant - SaaS mode startup branch in run():
//!   init_saas_mode() initializes SystemDb, VolumeRouter, StoragePool,
//!   VectorIndexPool, and jwt_secret. MpiState constructed via MpiState::local()
//!   in Local mode (backward compatible). Pool eviction timer + MinerScheduler
//!   spawned in SaaS mode. build_mpi_router() handles both modes transparently.
//!
//! ⚠️ Important Notes for Next Developer:
//! - record_key is derived from identity.to_bytes() (Ed25519 PRIVATE key)
//! - rawlog_key uses derive_rawlog_key() — a DIFFERENT KDF from derive_record_key()
//! - SuperNode init order: init_llm_router() → reset_stale_processing_tasks() → TaskWorker
//! - NerEngine MUST be loaded AFTER EmbedEngine (shared ORT runtime via Once)
//! - RerankerEngine MUST be loaded AFTER EmbedEngine and NerEngine (shared ORT Once)
//! - SaaS mode: MpiState.storage = None, MpiState.vector_index = None.
//!   Handlers access per-user storage via Extension<Arc<MemoryStorage>> injected
//!   by unified_auth_middleware from StoragePool. Never access state.storage in SaaS.
//! - config_path must be passed to Server::new() so api_secret and jwt_secret
//!   can be auto-generated and persisted on first startup.
//! - Pool eviction timer fires every 5 minutes (300 seconds) in SaaS mode.
//!   It evicts both StoragePool and VectorIndexPool idle connections.
//! - MinerScheduler ticks every 60 seconds in SaaS mode (per-user Miner).
//!   In Local mode, the original ReflectionMiner::run() is used unchanged.
//!
//! ## Last Modified
//! v2.5.2+SecAudit     - BroadcastRecord sig verification hardened.
//!   resolve_public_ip private-IP guard added.
//! v2.5.3+Security     - Server::new() gains config_path; auto-generates secrets.
//! v1.0.0-MultiTenant  - SaaS startup branch; MpiState::local() constructor.

use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::{
    decode_client_hello, encode_server_hello, encode_data_packet, ProtocolCodec,
};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::{DataPacket, MessageType};
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::UdpTransport;

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use rusqlite::OptionalExtension;

use crate::api::mpi::{build_mpi_router, MpiState, BaselineSnapshot, Mode};
use crate::api::auth::{ensure_jwt_secret, generate_secret};
use crate::config::{MemChainConfig, MemChainMode, ServerConfig, VectorQuantizationMode};
use crate::error::{Result, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    CommandHandler, HeartbeatReporter, ManagementClient, SessionReporter,
    reporter::SessionEventSender,
};
use crate::miner::ReflectionMiner;
#[allow(deprecated)]
use crate::services::memchain::{AofWriter, MemPool, MemoryStorage, VectorIndex};
use crate::services::memchain::derive_record_key;
use crate::services::memchain::derive_rawlog_key;
use crate::services::memchain::EmbedEngine;
use crate::services::memchain::NerEngine;
use crate::services::memchain::RerankerEngine;
use crate::services::memchain::{LlmRouter, TaskWorker};
// v1.0.0-MultiTenant: SaaS mode infrastructure
use crate::services::memchain::{
    SystemDb, VolumeRouter, StoragePool, VectorIndexPool,
    ensure_volumes_config,
};
use crate::services::{HandshakeService, IpPoolService, RoutingService, SessionManager};

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE: usize = 17;

#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

const COMMAND_CHANNEL_BUFFER: usize = 100;
const QUANTIZER_CAL_KEY_PREFIX: &str = "quantizer_cal";

/// Pool eviction timer interval (SaaS mode).
const POOL_EVICTION_INTERVAL_SECS: u64 = 300;

/// MinerScheduler tick interval (SaaS mode).
const MINER_SCHEDULER_TICK_SECS: u64 = 60;

// ============================================
// Server
// ============================================

pub struct Server {
    config: ServerConfig,
    identity: IdentityKeyPair,
    /// Path to the config file on disk.
    /// Used to persist auto-generated secrets (api_secret, jwt_secret) on first startup.
    config_path: Option<PathBuf>,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    /// Create a new Server instance.
    ///
    /// # Arguments
    /// - `config`: loaded server configuration
    /// - `identity`: Ed25519 keypair for this node
    /// - `config_path`: path to the config file on disk (for secret auto-generation).
    ///   Pass `None` only in tests; production callers always pass the real path.
    pub fn new(
        config: ServerConfig,
        identity: IdentityKeyPair,
        config_path: Option<PathBuf>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            config,
            identity,
            config_path,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting AeroNyx server v{}", env!("CARGO_PKG_VERSION"));

        // ── Ensure api_secret exists (v2.5.3+Security) ──────────────────
        // Auto-generate and persist if missing. Only for memchain-enabled modes.
        if self.config.memchain.is_enabled() {
            self.ensure_api_secret_on_disk().await;
        }

        let (ip_pool, sessions, routing) = self.init_services()?;
        let session_event_sender = self.init_management_reporter(&sessions).await;

        let handshake_service = Arc::new(HandshakeService::new(
            self.identity.clone(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
        ));

        let packet_handler = Arc::new(PacketHandler::new(
            Arc::clone(&sessions),
            Arc::clone(&routing),
        ));

        // ── MemChain initialization — branch on mode ─────────────────────
        let (storage, vector_index, mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (st, vi, mp, aw) = self.init_memchain().await?;
            (Some(st), Some(vi), Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] Disabled (mode=off)");
            (None, None, None, None)
        };

        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind: {}", e)))?,
        );
        info!("UDP transport listening on {}", self.config.listen_addr());

        #[cfg(target_os = "linux")]
        let tun = self.init_tun().await?;

        let server_pubkey_hex = hex::encode(self.identity.public_key_bytes());
        let mut tasks: Vec<(&str, JoinHandle<()>)> = Vec::new();

        let udp_task = self.spawn_udp_task(
            Arc::clone(&udp),
            #[cfg(target_os = "linux")]
            Arc::clone(&tun),
            Arc::clone(&handshake_service),
            Arc::clone(&packet_handler),
            Arc::clone(&sessions),
            session_event_sender.clone(),
            mempool.clone(),
            aof_writer.clone(),
            storage.clone(),
            vector_index.clone(),
            self.config.memchain.clone(),
            server_pubkey_hex.clone(),
        );
        tasks.push(("udp", udp_task));

        #[cfg(target_os = "linux")]
        {
            let tun_task = self.spawn_tun_task(
                Arc::clone(&tun),
                Arc::clone(&udp),
                Arc::clone(&packet_handler),
            );
            tasks.push(("tun", tun_task));
        }

        let cleanup_task = self.spawn_cleanup_task(
            Arc::clone(&sessions),
            Arc::clone(&ip_pool),
            Arc::clone(&routing),
            session_event_sender.clone(),
        );
        tasks.push(("cleanup", cleanup_task));

        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            // ── Determine operating mode ──────────────────────────────────
            let is_saas = self.config.memchain.mode == MemChainMode::Saas;

            let user_weights = Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new(),
            ));

            // Load MVF user weights from SQLite (Local mode only — in SaaS mode
            // each user has their own DB; weights are loaded per-user by the Miner).
            if !is_saas {
                let owner = self.identity.public_key_bytes();
                if let Some(blob) = st.load_user_weights(&owner).await {
                    if let Some(w) = crate::services::memchain::mvf::WeightVector::from_bytes(&blob)
                    {
                        let mut map = user_weights.write();
                        map.insert(hex::encode(owner), w);
                        info!("[MEMCHAIN] Loaded MVF user weights from SQLite");
                    }
                }
            }

            let mvf_baseline: Option<BaselineSnapshot> = if !is_saas {
                let conn = st.conn_lock().await;
                let raw: Option<Vec<u8>> = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'mvf_baseline'",
                        [],
                        |row: &rusqlite::Row<'_>| row.get::<_, Vec<u8>>(0),
                    )
                    .optional()
                    .unwrap_or(None);
                drop(conn);
                raw.and_then(|bytes| {
                    serde_json::from_str::<BaselineSnapshot>(
                        &String::from_utf8_lossy(&bytes),
                    )
                    .ok()
                })
            } else {
                None
            };

            let owner_key = self.identity.public_key_bytes();
            let api_secret = self.config.memchain.effective_api_secret().map(|s| s.to_string());

            // ── Shared engine initialization (both modes) ─────────────────
            let embed_engine = self.init_embed_engine();
            let ner_engine = self.init_ner_engine();
            let reranker_engine = self.init_reranker_engine();

            // ── v2.5.0+SuperNode: LlmRouter ──────────────────────────────
            // ⚠️ Init order is MANDATORY:
            //   1. init_llm_router()
            //   2. reset_stale_processing_tasks()
            //   3. TaskWorker::spawn()
            let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router().await;

            if llm_router.is_some() {
                let timeout_secs =
                    self.config.memchain.supernode.worker.task_timeout_secs as i64;
                let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
                if recovered > 0 {
                    info!(
                        recovered,
                        timeout_secs,
                        "[SUPERNODE] Recovered stale tasks from previous run"
                    );
                }
            }

            // ── Build MpiState (mode-aware) ───────────────────────────────
            let mpi_state = if is_saas {
                self.init_saas_mpi_state(
                    st,
                    owner_key,
                    api_secret,
                    Arc::clone(&user_weights),
                    embed_engine.clone(),
                    ner_engine.clone(),
                    reranker_engine,
                    llm_router.clone(),
                )
                .await?
            } else {
                // Local mode: use MpiState::local() constructor.
                let mpi = MpiState::local(
                    Arc::clone(st),
                    Arc::clone(vi),
                    self.identity.clone(),
                    parking_lot::RwLock::new(std::collections::HashMap::new()),
                    std::sync::atomic::AtomicBool::new(false),
                    Arc::clone(&user_weights),
                    self.config.memchain.mvf_alpha,
                    self.config.memchain.mvf_enabled,
                    parking_lot::RwLock::new(std::collections::HashMap::new()),
                    parking_lot::RwLock::new(mvf_baseline),
                    owner_key,
                    api_secret,
                    embed_engine.clone(),
                    self.config.memchain.allow_remote_storage,
                    self.config.memchain.max_remote_owners,
                    ner_engine.clone(),
                    self.config.memchain.graph_enabled,
                    self.config.memchain.entropy_filter_enabled,
                    reranker_engine,
                    // ⚠️ Uses derive_rawlog_key(), NOT derive_record_key()
                    Some(derive_rawlog_key(&self.identity.to_bytes())),
                    llm_router.clone(),
                );
                Arc::new(mpi)
            };

            // ── Local mode: pre-populate identity cache ───────────────────
            if !is_saas {
                let owner_hex = hex::encode(owner_key);
                let identity_records = st
                    .get_active_records(
                        &owner_key,
                        Some(aeronyx_core::ledger::MemoryLayer::Identity),
                        100,
                    )
                    .await;
                if !identity_records.is_empty() {
                    let mut cache = mpi_state.identity_cache.write();
                    cache.insert(owner_hex, identity_records);
                }

                mpi_state
                    .index_ready
                    .store(true, std::sync::atomic::Ordering::Relaxed);

                // Freeze MVF baseline if needed (Local mode only).
                if self.config.memchain.mvf_enabled
                    && mpi_state.mvf_baseline.read().is_none()
                {
                    let feedback = st.get_recent_feedback(200).await;
                    if !feedback.is_empty() {
                        let positive = feedback.iter().filter(|(s, _)| *s == 1).count();
                        let rate = positive as f32 / feedback.len() as f32;
                        let now_ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        let baseline = BaselineSnapshot {
                            positive_rate: rate,
                            sample_size: feedback.len(),
                            frozen_at: now_ts,
                        };

                        if let Ok(json) = serde_json::to_string(&baseline) {
                            let conn = st.conn_lock().await;
                            let _ = conn.execute(
                                "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('mvf_baseline', ?1)",
                                rusqlite::params![json.as_bytes()],
                            );
                        }

                        *mpi_state.mvf_baseline.write() = Some(baseline);
                        info!(rate, samples = feedback.len(), "[MVF] Baseline frozen");
                    }
                }
            }

            let api_task = self.start_combined_api(
                self.config.memchain.api_listen_addr,
                Arc::clone(&mpi_state),
                Arc::clone(mp),
                Arc::clone(aw),
                Arc::clone(&sessions),
                Arc::clone(&udp),
            );
            tasks.push(("memchain-api", api_task));

            // ── v2.5.0+SuperNode: TaskWorker spawn ────────────────────────
            // Spawned AFTER reset_stale_processing_tasks().
            if let Some(ref router) = llm_router {
                let worker = TaskWorker::new(
                    Arc::clone(st),
                    Arc::clone(router),
                    self.config.memchain.supernode.worker.clone(),
                );
                let worker_shutdown = self.shutdown_tx.subscribe();
                tasks.push(("supernode-worker", tokio::spawn(async move {
                    worker.run(worker_shutdown).await;
                })));
                info!(
                    poll_interval = self.config.memchain.supernode.worker.poll_interval_secs,
                    max_concurrent = self.config.memchain.supernode.worker.max_concurrent,
                    "[SUPERNODE] TaskWorker spawned"
                );
            }

            // ── SaaS mode: pool eviction timer ────────────────────────────
            if is_saas {
                if let (Some(ref sp), Some(ref vp)) =
                    (&mpi_state.storage_pool, &mpi_state.vector_pool)
                {
                    let sp_clone = Arc::clone(sp);
                    let vp_clone = Arc::clone(vp);
                    let mut evict_rx = self.shutdown_tx.subscribe();
                    tasks.push(("pool-eviction", tokio::spawn(async move {
                        let mut interval = tokio::time::interval(
                            Duration::from_secs(POOL_EVICTION_INTERVAL_SECS)
                        );
                        loop {
                            tokio::select! {
                                _ = evict_rx.recv() => break,
                                _ = interval.tick() => {
                                    let evicted_s = sp_clone.evict_idle().await;
                                    let evicted_v = vp_clone.evict_idle();
                                    if evicted_s + evicted_v > 0 {
                                        info!(
                                            storage = evicted_s,
                                            vector = evicted_v,
                                            "[POOL] Evicted idle connections"
                                        );
                                    }
                                }
                            }
                        }
                    })));
                    info!(
                        interval_secs = POOL_EVICTION_INTERVAL_SECS,
                        "[POOL] Eviction timer started"
                    );
                }
            }

            // ── Miner: Local = ReflectionMiner, SaaS = MinerScheduler ────
            if self.config.memchain.miner_interval_secs > 0 {
                if is_saas {
                    // SaaS mode: MinerScheduler polls all active users.
                    if let (Some(ref sp), Some(ref sys_db)) =
                        (&mpi_state.storage_pool, &mpi_state.system_db)
                    {
                        let scheduler = crate::miner::MinerScheduler::new(
                            Arc::clone(sp),
                            Arc::clone(sys_db),
                            self.config.memchain.saas.as_ref()
                                .map(|s| s.miner_max_owners_per_tick)
                                .unwrap_or(10),
                            self.config.memchain.saas.as_ref()
                                .map(|s| s.miner_max_rounds_per_hour)
                                .unwrap_or(6) as u32,
                            self.identity.clone(),
                            llm_router.clone(),
                            embed_engine.clone(),
                            ner_engine.clone(),
                        ).await;
                        let mut sched_rx = self.shutdown_tx.subscribe();
                        tasks.push(("miner-scheduler", tokio::spawn(async move {
                            let mut interval = tokio::time::interval(
                                Duration::from_secs(MINER_SCHEDULER_TICK_SECS)
                            );
                            loop {
                                tokio::select! {
                                    _ = sched_rx.recv() => break,
                                    _ = interval.tick() => {
                                        scheduler.tick().await;
                                    }
                                }
                            }
                        })));
                        info!(
                            tick_secs = MINER_SCHEDULER_TICK_SECS,
                            "[MINER] SaaS MinerScheduler started"
                        );
                    }
                } else {
                    // Local mode: original ReflectionMiner (unchanged).
                    let miner = ReflectionMiner::new(
                        self.config.memchain.miner_interval_secs,
                        Arc::clone(st),
                        Arc::clone(vi),
                        self.identity.clone(),
                        Arc::clone(mp),
                        Arc::clone(aw),
                        Arc::clone(&sessions),
                        Arc::clone(&udp),
                    )
                    .with_compaction_threshold(self.config.memchain.compaction_threshold)
                    .with_mvf(self.config.memchain.mvf_enabled, Arc::clone(&user_weights));

                    let miner = if let Some(ref ee) = embed_engine {
                        miner.with_embed_engine(Arc::clone(ee))
                    } else {
                        miner
                    };
                    let miner = if let Some(ref ne) = ner_engine {
                        miner.with_ner_engine(Arc::clone(ne))
                    } else {
                        miner
                    };
                    let miner = if let Some(ref lr) = llm_router {
                        miner.with_llm_router(Arc::clone(lr))
                    } else {
                        miner
                    };

                    let miner_shutdown = self.shutdown_tx.subscribe();
                    tasks.push(("miner", tokio::spawn(async move {
                        miner.run(miner_shutdown).await;
                    })));
                }
            } else {
                info!("[MINER] Disabled (interval=0)");
            }
        }

        info!("Server started successfully");
        self.wait_for_shutdown().await;
        info!("Shutting down server...");

        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());

        for (name, task) in tasks {
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => debug!("Task '{}' completed", name),
                Ok(Err(e)) => warn!("Task '{}' failed: {}", name, e),
                Err(_) => warn!("Task '{}' timed out", name),
            }
        }

        if let Err(e) = udp.shutdown().await {
            warn!("UDP shutdown error: {}", e);
        }

        if let (Some(ref st), Some(ref mp), Some(ref aw)) =
            (&storage, &mempool, &aof_writer)
        {
            info!(
                sqlite = st.count().await,
                mempool = mp.count(),
                aof = aw.lock().await.write_count(),
                "Shutdown complete (MemChain stats)"
            );
        } else {
            info!("Shutdown complete");
        }

        Ok(())
    }

    // ============================================
    // v2.5.3+Security: Auto-generate api_secret
    // ============================================

    /// Ensure api_secret exists, auto-generating if missing.
    ///
    /// Only runs when memchain is enabled. The generated secret is written
    /// back to the config file via a regex-safe line replacement.
    /// Non-fatal: if write fails, logs a warning and continues.
    async fn ensure_api_secret_on_disk(&self) {
        // Already configured — nothing to do.
        if self.config.memchain.effective_api_secret().is_some() {
            return;
        }

        let Some(ref path) = self.config_path else { return };

        let secret = generate_secret();
        match crate::api::auth::write_secret_to_config_pub(path, "api_secret", &secret) {
            Ok(()) => {
                info!(
                    path = %path.display(),
                    "[SECURITY] Auto-generated api_secret and written to config"
                );
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "[SECURITY] Failed to persist api_secret — server is unprotected"
                );
            }
        }
    }

    // ============================================
    // v1.0.0-MultiTenant: SaaS MpiState init
    // ============================================

    /// Initialize the SaaS-mode MpiState.
    ///
    /// Creates SystemDb, VolumeRouter, StoragePool, VectorIndexPool, and
    /// ensures the JWT secret is generated/persisted.
    ///
    /// The single-user `storage` and `vector_index` are not used in SaaS mode —
    /// they remain the server-level DB used only for p2p sync paths.
    #[allow(clippy::too_many_arguments)]
    async fn init_saas_mpi_state(
        &self,
        _server_storage: &Arc<MemoryStorage>,
        owner_key: [u8; 32],
        api_secret: Option<String>,
        user_weights: Arc<parking_lot::RwLock<std::collections::HashMap<String, crate::services::memchain::mvf::WeightVector>>>,
        embed_engine: Option<Arc<EmbedEngine>>,
        ner_engine: Option<Arc<NerEngine>>,
        reranker_engine: Option<Arc<RerankerEngine>>,
        llm_router: Option<Arc<LlmRouter>>,
    ) -> Result<Arc<MpiState>> {
        let saas_cfg = self.config.memchain.saas.as_ref()
            .ok_or_else(|| ServerError::startup_failed(
                "mode=saas requires [memchain.saas] config section"
            ))?;

        let data_root = &saas_cfg.data_root;

        // Ensure data_root exists.
        tokio::fs::create_dir_all(data_root).await.map_err(|e| {
            ServerError::startup_failed(format!("SaaS data_root '{}': {}", data_root.display(), e))
        })?;

        // ── 1. SystemDb ───────────────────────────────────────────────
        let system_db = SystemDb::open(&data_root.join("system.db"))
            .await
            .map_err(|e| ServerError::startup_failed(format!("SystemDb: {}", e)))?;

        // ── 2. volumes.toml (auto-generate if missing) ────────────────
        let volumes_config_path = ensure_volumes_config(data_root)
            .map_err(|e| ServerError::startup_failed(format!("volumes.toml: {}", e)))?;

        // ── 3. VolumeRouter ───────────────────────────────────────────
        let volume_router = VolumeRouter::new(&volumes_config_path, Arc::clone(&system_db))
            .await
            .map_err(|e| ServerError::startup_failed(format!("VolumeRouter: {}", e)))?;

        // ── 4. StoragePool ────────────────────────────────────────────
        let storage_pool = StoragePool::new(
            Arc::clone(&volume_router),
            Arc::clone(&system_db),
            saas_cfg.pool_max_connections,
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
        );

        // ── 5. VectorIndexPool ────────────────────────────────────────
        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let saturation_threshold = if self.config.memchain.vector_early_termination {
            0.001_f32
        } else {
            0.0_f32
        };

        let vector_pool = VectorIndexPool::new(
            Arc::clone(&volume_router),
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
            quantization_enabled,
            saturation_threshold,
        );

        // ── 6. JWT secret ─────────────────────────────────────────────
        let jwt_secret = ensure_jwt_secret(
            self.config.memchain.jwt_secret.as_deref(),
            self.config_path.as_deref(),
        )
        .map_err(|e| ServerError::startup_failed(format!("jwt_secret: {}", e)))?;

        info!(
            data_root = %data_root.display(),
            pool_max = saas_cfg.pool_max_connections,
            idle_timeout_secs = saas_cfg.pool_idle_timeout_secs,
            "[SAAS] Infrastructure initialized"
        );

        // ── 7. Build SaaS MpiState ────────────────────────────────────
        let mpi_state = Arc::new(MpiState {
            mode: Mode::Saas,
            // Single-user storage fields: None in SaaS mode.
            // Per-user storage is accessed via storage_pool in middleware.
            storage: None,
            vector_index: None,
            identity: self.identity.clone(),
            identity_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            index_ready: std::sync::atomic::AtomicBool::new(true),
            user_weights,
            mvf_alpha: self.config.memchain.mvf_alpha,
            mvf_enabled: self.config.memchain.mvf_enabled,
            session_embeddings: parking_lot::RwLock::new(std::collections::HashMap::new()),
            mvf_baseline: parking_lot::RwLock::new(None),
            // SaaS: owner_key is placeholder — never used in request handling.
            owner_key,
            api_secret,
            embed_engine,
            allow_remote_storage: false, // SaaS mode doesn't use Ed25519 remote auth.
            max_remote_owners: 0,
            ner_engine,
            graph_enabled: self.config.memchain.graph_enabled,
            entropy_filter_enabled: self.config.memchain.entropy_filter_enabled,
            reranker_engine,
            rawlog_key: Some(derive_rawlog_key(&self.identity.to_bytes())),
            llm_router,
            // SaaS-specific fields.
            storage_pool: Some(storage_pool),
            vector_pool: Some(vector_pool),
            volume_router: Some(volume_router),
            system_db: Some(system_db),
            jwt_secret: Some(jwt_secret),
            token_ttl_secs: self.config.memchain.token_ttl_secs,
        });

        Ok(mpi_state)
    }

    // ============================================
    // Shared Engine Initialization
    // ============================================

    fn init_embed_engine(&self) -> Option<Arc<EmbedEngine>> {
        let model_path = &self.config.memchain.embed_model_path;
        match EmbedEngine::load(
            model_path,
            self.config.memchain.embed_max_tokens,
            self.config.memchain.embed_output_dim,
        ) {
            Ok(engine) => {
                info!(
                    model = %model_path,
                    model_type = %engine.model_type(),
                    dim = engine.dim(),
                    "[EMBED] ✅ Local embedding engine loaded"
                );
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[EMBED] ⚠️ Unavailable");
                None
            }
        }
    }

    fn init_ner_engine(&self) -> Option<Arc<NerEngine>> {
        if !self.config.memchain.ner_enabled {
            debug!("[NER] Disabled (ner_enabled=false)");
            return None;
        }
        let model_path = &self.config.memchain.ner_model_path;
        let threshold = self.config.memchain.ner_confidence_threshold;
        match NerEngine::load(model_path, threshold, 0) {
            Ok(engine) => {
                info!(
                    model = %model_path,
                    threshold,
                    "[NER] ✅ Local NER engine loaded (GLiNER)"
                );
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[NER] ⚠️ Unavailable");
                None
            }
        }
    }

    fn init_reranker_engine(&self) -> Option<Arc<RerankerEngine>> {
        if !self.config.memchain.reranker_enabled {
            debug!("[RERANKER] Disabled (reranker_enabled=false)");
            return None;
        }
        let model_path = &self.config.memchain.reranker_model_path;
        let max_seq = self.config.memchain.reranker_max_seq_length;
        match RerankerEngine::load(model_path, max_seq) {
            Ok(engine) => {
                info!(
                    model = %model_path,
                    blend_weight = %RerankerEngine::blend_weight(),
                    "[RERANKER] ✅ Cross-encoder loaded"
                );
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[RERANKER] ⚠️ Unavailable");
                None
            }
        }
    }

    // ============================================
    // v2.5.0+SuperNode: LlmRouter Initialization
    // ============================================

    async fn init_llm_router(&self) -> Option<Arc<LlmRouter>> {
        use crate::config_supernode::ProviderType;
        use crate::services::memchain::{LlmProvider, OpenAiCompatProvider, AnthropicProvider};

        if !self.config.memchain.is_supernode_enabled() {
            debug!("[SUPERNODE] Disabled (supernode.enabled=false)");
            return None;
        }

        let supernode = &self.config.memchain.supernode;
        if supernode.providers.is_empty() {
            warn!("[SUPERNODE] enabled=true but no providers configured — SuperNode disabled");
            return None;
        }

        let mut providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = Vec::new();

        for provider_cfg in &supernode.providers {
            let api_key: Option<String> = provider_cfg.api_key.as_ref().map(|k| {
                let k: &String = k;
                if k.starts_with('$') {
                    std::env::var(&k[1..]).unwrap_or_else(|_| {
                        warn!(
                            key = %k, provider = %provider_cfg.name,
                            "[SUPERNODE] ENV var not set for api_key"
                        );
                        String::new()
                    })
                } else {
                    k.clone()
                }
            });

            let api_base = if provider_cfg.api_base.is_empty()
                && provider_cfg.provider_type == ProviderType::Anthropic
            {
                "https://api.anthropic.com".to_string()
            } else {
                provider_cfg.api_base.clone()
            };

            let provider: Arc<dyn LlmProvider> = match provider_cfg.provider_type {
                ProviderType::OpenaiCompatible => {
                    match OpenAiCompatProvider::new(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        api_key.unwrap_or_default(),
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ) {
                        Ok(p) => Arc::new(p),
                        Err(e) => {
                            warn!(
                                provider = %provider_cfg.name, error = %e,
                                "[SUPERNODE] OpenAiCompatProvider failed — skipped"
                            );
                            continue;
                        }
                    }
                }
                ProviderType::Anthropic => {
                    let key = match api_key {
                        Some(k) if !k.is_empty() => k,
                        _ => {
                            warn!(
                                provider = %provider_cfg.name,
                                "[SUPERNODE] Anthropic provider requires api_key — skipped"
                            );
                            continue;
                        }
                    };
                    match AnthropicProvider::new(
                        provider_cfg.name.clone(),
                        key,
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ) {
                        Ok(p) => Arc::new(p),
                        Err(e) => {
                            warn!(
                                provider = %provider_cfg.name, error = %e,
                                "[SUPERNODE] AnthropicProvider failed — skipped"
                            );
                            continue;
                        }
                    }
                }
            };

            info!(
                name = %provider_cfg.name,
                type_ = ?provider_cfg.provider_type,
                model = %provider_cfg.model,
                api_base = %api_base,
                "[SUPERNODE] Provider registered"
            );

            providers.push((
                provider_cfg.name.clone(),
                api_base,
                provider_cfg.model.clone(),
                provider,
            ));
        }

        if providers.is_empty() {
            warn!("[SUPERNODE] All providers failed — SuperNode disabled");
            return None;
        }

        let router = LlmRouter::new(providers, supernode.routing.clone());
        info!(
            providers = supernode.providers.len(),
            fallback = ?supernode.routing.fallback,
            "[SUPERNODE] ✅ LlmRouter initialized"
        );
        Some(Arc::new(router))
    }

    // ============================================
    // MemChain Initialization (dual-engine, Local mode)
    // ============================================

    async fn init_memchain(&self) -> Result<(
        Arc<MemoryStorage>,
        Arc<VectorIndex>,
        Arc<MemPool>,
        Arc<TokioMutex<AofWriter>>,
    )> {
        let db_path = &self.config.memchain.db_path;
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("DB dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        // ⚠️ SECURITY: identity.to_bytes() = PRIVATE key. Do NOT use public_key_bytes().
        let record_key = derive_record_key(&self.identity.to_bytes());
        info!("[MEMCHAIN] Record content encryption enabled");

        let storage = Arc::new(
            MemoryStorage::open(db_path, Some(record_key))
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let vector_index = Arc::new(if quantization_enabled {
            let sat = if self.config.memchain.vector_early_termination {
                0.001_f32
            } else {
                0.0_f32
            };
            info!(quantization = "scalar_uint8", "[MEMCHAIN] VectorIndex with scalar quantization");
            VectorIndex::with_config(true, sat)
        } else {
            VectorIndex::new()
        });

        let owner = self.identity.public_key_bytes();
        let records_with_model = storage.get_records_with_embedding(&owner).await;
        let rebuild_count = records_with_model.len();
        for (r, model) in records_with_model {
            if r.has_embedding() {
                vector_index.upsert(
                    r.record_id, r.embedding.clone(), r.layer,
                    r.timestamp, &r.owner, &model,
                );
            }
        }

        info!(
            db = %db_path,
            records = storage.count().await,
            vectors = rebuild_count,
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );

        if quantization_enabled && rebuild_count > 0 {
            let owner_hex = hex::encode(owner);
            let model_name = std::path::Path::new(&self.config.memchain.embed_model_path)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("minilm-l6-v2");

            let cal_key = format!(
                "{}:{}:{}", QUANTIZER_CAL_KEY_PREFIX, owner_hex, model_name
            );
            let restored = {
                let conn = storage.conn_lock().await;
                let cal_data: Option<Vec<u8>> = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = ?1",
                        rusqlite::params![cal_key],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .optional()
                    .unwrap_or(None);
                drop(conn);

                if let Some(data) = cal_data {
                    let ok = vector_index.restore_quantizer(&owner, model_name, &data);
                    if ok {
                        info!(model = model_name, "[VECTOR] ✅ Quantizer restored");
                    }
                    ok
                } else {
                    false
                }
            };

            if !restored {
                vector_index.calibrate_partition(&owner, model_name);
                if let Some(cal_bytes) = vector_index.get_quantizer_bytes(&owner, model_name) {
                    let conn = storage.conn_lock().await;
                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES (?1, ?2)",
                        rusqlite::params![cal_key, cal_bytes.as_slice()],
                    );
                    drop(conn);
                    info!(
                        model = model_name,
                        "[VECTOR] ✅ Quantizer calibrated and persisted"
                    );
                }
            }
        }

        let aof_path = &self.config.memchain.aof_path;
        if let Some(parent) = std::path::Path::new(aof_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("AOF dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let (existing_facts, last_block) = AofWriter::replay(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF replay: {}", e)))?;

        let mempool = Arc::new(MemPool::new());
        let mut loaded = 0u64;
        for fact in existing_facts {
            if mempool.add_fact(fact) { loaded += 1; }
        }

        let aof_writer = AofWriter::open(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF open: {}", e)))?;
        aof_writer.set_chain_state(last_block.as_ref());
        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(facts = loaded, "[MEMCHAIN] Legacy MemPool + AOF initialized");

        Ok((storage, vector_index, mempool, aof_writer))
    }

    // ============================================
    // Combined API Server
    // ============================================

    fn start_combined_api(
        &self,
        listen_addr: std::net::SocketAddr,
        mpi_state: Arc<MpiState>,
        _mempool: Arc<MemPool>,
        _aof_writer: Arc<TokioMutex<AofWriter>>,
        _sessions: Arc<SessionManager>,
        _udp: Arc<UdpTransport>,
    ) -> JoinHandle<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let app = build_mpi_router(mpi_state);

            let listener = match tokio::net::TcpListener::bind(listen_addr).await {
                Ok(l) => {
                    info!("[API] MemChain API on http://{}", listen_addr);
                    l
                }
                Err(e) => {
                    error!("[API] Bind failed {}: {}", listen_addr, e);
                    return;
                }
            };

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
                info!("[API] Shutdown signal received");
            });

            if let Err(e) = server.await {
                error!("[API] Server error: {}", e);
            }
            info!("[API] Stopped");
        })
    }

    // ============================================
    // Management Reporter
    // ============================================

    async fn init_management_reporter(
        &self,
        sessions: &Arc<SessionManager>,
    ) -> SessionEventSender {
        info!("Initializing management reporting...");

        let mgmt_client = Arc::new(ManagementClient::new(
            self.config.management.clone(),
            self.identity.clone(),
        ));
        info!("Node ID: {}", mgmt_client.node_id());

        let public_ip = self.resolve_public_ip().await;

        let agent_manager = Arc::new(
            crate::services::AgentManager::new(Arc::clone(&mgmt_client))
        );
        agent_manager.detect_existing().await;

        {
            let am = Arc::clone(&agent_manager);
            let mut rx = self.shutdown_tx.subscribe();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                loop {
                    tokio::select! {
                        _ = rx.recv() => break,
                        _ = interval.tick() => am.periodic_health_check().await,
                    }
                }
            });
        }

        let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);
        let cmd_handler = CommandHandler::new(
            cmd_rx,
            Arc::clone(&mgmt_client),
            Arc::clone(&agent_manager),
        );
        let cmd_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { cmd_handler.run(cmd_shutdown).await; });

        let memchain_status_fn: Option<crate::management::reporter::MemChainStatusFn> =
            if self.config.memchain.is_enabled() {
                let allow_remote = self.config.memchain.allow_remote_storage;
                let max_owners = self.config.memchain.max_remote_owners;
                Some(Box::new(move || {
                    Some(crate::management::client::MemChainHeartbeatStatus {
                        enabled: true,
                        allow_remote_storage: allow_remote,
                        max_remote_owners: max_owners,
                        current_remote_owners: 0,
                    })
                }))
            } else {
                None
            };

        let mut heartbeat =
            HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
                .with_command_sender(cmd_tx)
                .with_agent_manager(Arc::clone(&agent_manager));

        if let Some(f) = memchain_status_fn {
            heartbeat = heartbeat.with_memchain_status(f);
        }

        let sess = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            heartbeat
                .run(move || sess.count() as u32, hb_shutdown)
                .await;
        });

        let (session_reporter, event_tx) =
            SessionReporter::new(Arc::clone(&mgmt_client));
        let sr_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { session_reporter.run(sr_shutdown).await; });

        let node_info_path = &self.config.management.node_info_path;
        if let Ok(node_info) = crate::management::models::StoredNodeInfo::load(node_info_path) {
            let ws = crate::management::WsTunnel::new(
                self.identity.clone(),
                node_info.node_id.clone(),
                Arc::clone(&agent_manager),
            )
            .with_mpi_api_secret(
                self.config
                    .memchain
                    .effective_api_secret()
                    .map(|s| s.to_string()),
            );
            let ws_shutdown = self.shutdown_tx.subscribe();
            tokio::spawn(async move { ws.run(ws_shutdown).await; });
            info!(node_id = %node_info.node_id, "[WS_TUNNEL] Spawned");
        } else {
            warn!("[WS_TUNNEL] Node not registered — disabled");
        }

        info!("[MANAGEMENT] Reporting started");
        SessionEventSender::new(event_tx)
    }

    // ============================================
    // Public IP Resolution
    // ============================================

    async fn resolve_public_ip(&self) -> String {
        if let Some(ip) = self.config.network.public_ip() {
            return ip.to_string();
        }

        // v2.5.2+SecAudit: these URLs are hardcoded — do NOT extend with
        // user-controlled input (SSRF vector).
        let services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
        ];

        if let Ok(client) = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            for url in &services {
                let mut req = client.get(*url);
                if url.contains("metadata.google.internal") {
                    req = req.header("Metadata-Flavor", "Google");
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        if let Ok(body) = resp.text().await {
                            let ip_str = body.trim();
                            if ip_str.len() <= 45 {
                                if let Ok(addr) = ip_str.parse::<std::net::IpAddr>() {
                                    let is_private = match addr {
                                        std::net::IpAddr::V4(v4) => {
                                            v4.is_loopback()
                                                || v4.is_private()
                                                || v4.is_unspecified()
                                        }
                                        std::net::IpAddr::V6(v6) => {
                                            v6.is_loopback() || v6.is_unspecified()
                                        }
                                    };
                                    if !is_private {
                                        info!(ip = %addr, source = %url, "[NET] Public IP detected");
                                        return addr.to_string();
                                    }
                                    warn!(
                                        ip = %addr, source = %url,
                                        "[NET] Ignoring private/loopback IP from metadata"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        let fallback = self.config.listen_addr().ip().to_string();
        warn!(ip = %fallback, "[NET] Fallback to listen address");
        fallback
    }

    // ============================================
    // Core Services
    // ============================================

    fn init_services(&self) -> Result<(
        Arc<IpPoolService>,
        Arc<SessionManager>,
        Arc<RoutingService>,
    )> {
        let (network, prefix) = self.config.parse_ip_range()?;
        let ip_pool =
            Arc::new(IpPoolService::new(network, prefix, self.config.gateway_ip())?);
        let sessions = Arc::new(SessionManager::new(
            self.config.max_sessions(),
            Duration::from_secs(self.config.session_timeout_secs()),
        ));
        let routing = Arc::new(RoutingService::new());
        info!(
            capacity = ip_pool.capacity(),
            max_sessions = self.config.max_sessions(),
            "Services initialized"
        );
        Ok((ip_pool, sessions, routing))
    }

    #[cfg(target_os = "linux")]
    async fn init_tun(&self) -> Result<Arc<LinuxTun>> {
        let cfg = TunConfig::new(self.config.device_name())
            .with_address(self.config.gateway_ip())
            .with_netmask(Ipv4Addr::new(255, 255, 255, 0))
            .with_mtu(self.config.mtu());
        let tun = LinuxTun::create(cfg)
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN: {}", e)))?;
        tun.up()
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN up: {}", e)))?;
        info!(
            "TUN '{}' initialized @ {}",
            tun.name(),
            self.config.gateway_ip()
        );
        Ok(Arc::new(tun))
    }

    // ============================================
    // UDP Task
    // ============================================

    #[allow(clippy::too_many_arguments)]
    fn spawn_udp_task(
        &self,
        udp: Arc<UdpTransport>,
        #[cfg(target_os = "linux")] tun: Arc<LinuxTun>,
        handshake: Arc<HandshakeService>,
        packet_handler: Arc<PacketHandler>,
        sessions: Arc<SessionManager>,
        session_events: SessionEventSender,
        mempool: Option<Arc<MemPool>>,
        aof_writer: Option<Arc<TokioMutex<AofWriter>>>,
        storage: Option<Arc<MemoryStorage>>,
        vector_index: Option<Arc<VectorIndex>>,
        memchain_config: MemChainConfig,
        server_pubkey_hex: String,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let udp_reply = Arc::clone(&udp);

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            let crypto = DefaultTransportCrypto::new();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = udp.recv(&mut buf) => {
                        match result {
                            Ok((len, source)) => {
                                if shutdown.load(Ordering::SeqCst) { break; }
                                let data = &buf[..len];

                                match ProtocolCodec::peek_message_type(data) {
                                    Ok(MessageType::ClientHello) => {
                                        if let Ok(hello) = decode_client_hello(data) {
                                            match handshake.process(&hello, source.addr) {
                                                Ok(result) => {
                                                    let sid = BASE64.encode(
                                                        &result.response.session_id
                                                    );
                                                    session_events.session_created(&sid, None);
                                                    let resp = encode_server_hello(&result.response);
                                                    let _ = udp.send(&resp, &source.addr).await;
                                                }
                                                Err(e) => {
                                                    warn!("[HANDSHAKE] Failed {}: {}", source.addr, e)
                                                }
                                            }
                                        }
                                    }

                                    Ok(MessageType::Keepalive) => {
                                        if len >= KEEPALIVE_PACKET_SIZE {
                                            let mut sid = [0u8; 16];
                                            sid.copy_from_slice(&data[1..17]);
                                            if let Some(id) = SessionId::from_bytes(&sid) {
                                                if let Some(s) = sessions.get(&id) {
                                                    s.touch();
                                                }
                                            }
                                        }
                                    }

                                    Ok(MessageType::Data) | Err(_) => {
                                        match packet_handler.handle_udp_packet(data) {
                                            Ok((_sess, DecryptedPayload::Vpn(pkt))) => {
                                                #[cfg(target_os = "linux")]
                                                { let _ = tun.write(&pkt).await; }
                                            }
                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                if let (Some(ref mp), Some(ref aw)) =
                                                    (&mempool, &aof_writer)
                                                {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw,
                                                        &storage, &vector_index,
                                                        &memchain_config,
                                                        &server_pubkey_hex,
                                                        &session, &udp_reply, &crypto,
                                                    )
                                                    .await;
                                                }
                                            }
                                            Err(_) => {}
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) {
                                    error!("UDP recv error: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // MemChain Message Handler (unchanged)
    // ============================================

    #[allow(clippy::too_many_arguments)]
    async fn handle_memchain_message(
        msg: MemChainMessage,
        mempool: &Arc<MemPool>,
        aof_writer: &Arc<TokioMutex<AofWriter>>,
        storage: &Option<Arc<MemoryStorage>>,
        vector_index: &Option<Arc<VectorIndex>>,
        config: &MemChainConfig,
        server_pubkey_hex: &str,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        match msg {
            MemChainMessage::BroadcastFact(fact) => {
                let origin_hex = hex::encode(fact.origin);
                let sig_ok = match IdentityPublicKey::from_bytes(&fact.origin) {
                    Ok(pk) => pk.verify(&fact.fact_id, &fact.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!("[MEMCHAIN] BroadcastFact sig failed");
                    return;
                }
                if !config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                    warn!("[MEMCHAIN] BroadcastFact untrusted origin");
                    return;
                }
                if mempool.add_fact(fact.clone()) {
                    let mut w = aof_writer.lock().await;
                    let _ = w.append_fact(&fact).await;
                }
            }

            MemChainMessage::BroadcastRecord(record) => {
                let owner_hex = record.owner_hex();
                // v2.5.2+SecAudit: verify against record_id (content hash).
                let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                    Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!(
                        owner = %owner_hex,
                        "[MEMCHAIN] BroadcastRecord signature verification failed"
                    );
                    return;
                }
                if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                    warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord untrusted owner");
                    return;
                }
                if !record.verify_id() {
                    warn!(
                        owner = %owner_hex,
                        id = hex::encode(record.record_id),
                        "[MEMCHAIN] BroadcastRecord record_id hash mismatch — possible tampering"
                    );
                    return;
                }
                if let Some(ref st) = storage {
                    if st.insert(&record, "p2p-remote").await {
                        info!(
                            id = hex::encode(record.record_id),
                            "[MEMCHAIN] BroadcastRecord stored"
                        );
                        if record.has_embedding() {
                            if let Some(ref vi) = vector_index {
                                vi.upsert(
                                    record.record_id,
                                    record.embedding.clone(),
                                    record.layer,
                                    record.timestamp,
                                    &record.owner,
                                    "p2p-remote",
                                );
                            }
                        }
                    }
                }
            }

            MemChainMessage::SyncRequest { last_known_hash } => {
                let facts = mempool.get_facts_after(last_known_hash);
                let resp = MemChainMessage::SyncResponse { facts };
                Self::send_to_session(&resp, session, udp, crypto).await;
            }

            MemChainMessage::SyncResponse { facts } => {
                for fact in facts {
                    let origin_hex = hex::encode(fact.origin);
                    let sig_ok = match IdentityPublicKey::from_bytes(&fact.origin) {
                        Ok(pk) => pk.verify(&fact.fact_id, &fact.signature).is_ok(),
                        Err(_) => false,
                    };
                    if !sig_ok
                        || !config.is_origin_trusted(&origin_hex, server_pubkey_hex)
                    {
                        continue;
                    }
                    if mempool.add_fact(fact.clone()) {
                        let mut w = aof_writer.lock().await;
                        let _ = w.append_fact(&fact).await;
                    }
                }
            }

            MemChainMessage::SyncRecordRequest { owner, after_timestamp } => {
                if let Some(ref st) = storage {
                    let records = st.query_by_owner_after(&owner, after_timestamp).await;
                    let resp = MemChainMessage::SyncRecordResponse { records };
                    Self::send_to_session(&resp, session, udp, crypto).await;
                }
            }

            MemChainMessage::SyncRecordResponse { records } => {
                if let Some(ref st) = storage {
                    for record in records {
                        let owner_hex = record.owner_hex();
                        let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                            Ok(pk) => {
                                pk.verify(&record.record_id, &record.signature).is_ok()
                            }
                            Err(_) => false,
                        };
                        if !sig_ok {
                            warn!(
                                owner = %owner_hex,
                                "[MEMCHAIN] SyncRecordResponse sig failed — skipping"
                            );
                            continue;
                        }
                        if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                            continue;
                        }
                        if !record.verify_id() {
                            warn!(
                                owner = %owner_hex,
                                id = hex::encode(record.record_id),
                                "[MEMCHAIN] SyncRecordResponse record_id hash mismatch — skipping"
                            );
                            continue;
                        }
                        let _ = st.insert(&record, "p2p-sync").await;
                    }
                }
            }

            MemChainMessage::BlockAnnounce(header) => {
                info!(
                    height = header.height,
                    hash = hex::encode(header.hash()),
                    "[MEMCHAIN] BlockAnnounce received"
                );
            }

            _ => {
                debug!("[MEMCHAIN] Unhandled message variant");
            }
        }
    }

    async fn send_to_session(
        msg: &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        let plaintext = match encode_memchain(msg) {
            Ok(p) => p,
            Err(e) => {
                error!("[MEMCHAIN_TX] Encode: {}", e);
                return;
            }
        };

        let counter = session.next_tx_counter();
        let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];

        let len = match crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            &plaintext,
            &mut encrypted,
        ) {
            Ok(l) => l,
            Err(e) => {
                error!("[MEMCHAIN_TX] Encrypt: {}", e);
                return;
            }
        };
        encrypted.truncate(len);

        let pkt = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let bytes = encode_data_packet(&pkt).to_vec();
        let _ = udp.send(&bytes, &session.client_endpoint).await;
    }

    // ============================================
    // TUN Task
    // ============================================

    #[cfg(target_os = "linux")]
    fn spawn_tun_task(
        &self,
        tun: Arc<LinuxTun>,
        udp: Arc<UdpTransport>,
        handler: Arc<PacketHandler>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    result = tun.read(&mut buf) => {
                        match result {
                            Ok(len) => {
                                if shutdown.load(Ordering::SeqCst) { break; }
                                if let Ok((enc, ep)) = handler.handle_tun_packet(&buf[..len]) {
                                    let _ = udp.send(&enc, &ep).await;
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) {
                                    error!("TUN: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // Cleanup Task
    // ============================================

    fn spawn_cleanup_task(
        &self,
        sessions: Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        routing: Arc<RoutingService>,
        events: SessionEventSender,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        for (sid, vip) in sessions.cleanup_expired() {
                            routing.remove_route(vip);
                            ip_pool.release(vip);
                            events.session_ended(&sid.to_string(), None, 0, 0);
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // Shutdown
    // ============================================

    async fn wait_for_shutdown(&self) {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl+C listener failed");
        info!("Shutdown signal received");
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());
    }
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("listen", &self.config.listen_addr())
            .field("tun", &self.config.device_name())
            .field("mode", &self.config.memchain.mode)
            .finish()
    }
}
