// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
//! # Server Orchestrator — v2.1 MemChain Protocol Refactor
//!
//! ## Modification History
//! v0.1.0 - Initial server implementation
//! v0.1.1 - Keepalive packet handling
//! v0.2.0 - CMS management integration
//! v0.3.0 - MemChain integration (MemPool, AofWriter, 1st-byte dispatch)
//! v0.3.1 - Fixed Option<Arc<…>> type mismatch
//! v0.4.0 - Ed25519 verify, trust whitelist, SyncReq/SyncRes
//! v1.3.0 - Command Pipeline: CommandHandler + channel wiring
//! v2.1.0 - 🌟 Dual-engine init (SQLite+Vector + legacy MemPool+AOF),
//!           MPI router merged, BroadcastRecord/SyncRecordRequest handling,
//!           vector index rebuild on startup, new Miner 8-arg signature.
//! v2.1.1 - Fixed duplicate LinuxTun import, removed unused `trace` import
//! v2.1.0+MVF+Encryption - 🌟 MemoryStorage.open() now receives record_key
//!   derived from Ed25519 private key for transparent record content encryption.
//!   MpiState now receives api_secret from config for Bearer token auth.
//! v2.3.0+RemoteStorage - 🌟 MpiState now receives allow_remote_storage and
//!   max_remote_owners from config for Phase 1 remote MPI Gateway support.
//! v2.4.0-GraphCognition - 🌟 NerEngine initialization + MpiState extension:
//!   - Load GLiNER ONNX model alongside EmbedEngine (shared ORT runtime)
//!   - Pass ner_engine: Option<Arc<NerEngine>> into MpiState
//!   - Pass cognitive graph config flags into MpiState
//!   - NerEngine load is graceful: missing model = NER disabled, server runs normally
//! v2.4.0-GraphCognition Phase B - 🌟 Scalar quantization integration:
//!   - VectorIndex now uses with_config() when vector_quantization is ScalarUint8
//!   - calibrate_partition() called after index rebuild from SQLite
//!   - Quantizer calibration persisted to chain_state for fast restart
//! v2.4.0+Reranker - 🌟 RerankerEngine initialization (cross-encoder/ms-marco-MiniLM-L-6-v2).
//!   Loaded after NerEngine (shared ORT runtime). Graceful fallback on failure.
//!   Passed into MpiState.reranker_engine for recall_handler.rs Step 3.5.
//! v2.4.0+Conversation - 🌟 Added derive_rawlog_key import and rawlog_key field in MpiState.
//!   Enables conversation replay decryption for GET /sessions/:id/conversation endpoint.
//!   Key derived from Ed25519 PRIVATE key — can only decrypt local owner's rawlogs.
//! v2.5.0+SuperNode - 🌟 LlmRouter initialization + MpiState.llm_router field.
//!   - init_llm_router() reads supernode config, constructs providers + router
//!   - reset_stale_processing_tasks() called at startup before TaskWorker spawn
//!   - TaskWorker spawned as background task when supernode.enabled=true
//!   - ReflectionMiner receives .with_llm_router() for Steps 8/9/10 enqueue
//!   - All SuperNode paths gated on is_supernode_enabled() — safe to disable
//!
//! ## Modification Reason (v2.4.0+Reranker)
//! - RerankerEngine provides cross-encoder reranking for recall Step 3.5
//! - Loaded AFTER EmbedEngine and NerEngine (all three share ORT runtime via Once)
//! - Load failure is graceful: server starts without reranking, recall uses RRF-only
//! - ~22MB model, loads in < 100ms, no calibration or persistent state needed
//!
//! ## Modification Reason (v2.4.0+Conversation)
//! - rawlog_key enables mpi_graph_handlers.rs to decrypt raw_logs for conversation replay
//! - Derived from identity.to_bytes() (Ed25519 PRIVATE key), same as Miner Step 0
//! - Only local-mode requests can decrypt (remote requests see encrypted:true, content:null)
//!
//! ## Modification Reason (v2.4.0-GraphCognition Phase B)
//! - VectorIndex::with_config() replaces VectorIndex::new() when scalar quantization
//!   is enabled in config (vector_quantization = ScalarUint8)
//! - After rebuilding the vector index from SQLite, calibrate_partition() must be called
//!   to train the ScalarQuantizer on existing vectors. Without calibration, two-phase
//!   search falls back to brute-force f32 (no degradation, just no speedup).
//! - Quantizer calibration bytes are persisted to chain_state table for fast restart:
//!   on subsequent startups, restore_quantizer() is attempted first. If valid, skips
//!   re-calibration. If missing or stale, full calibration runs.
//!
//! ## Modification Reason (v2.5.0+SuperNode)
//! - LlmRouter wraps N providers (OpenAI-compatible + Anthropic) with per-task routing
//! - init_llm_router() is a private async helper that reads supernode config and
//!   constructs the router. Returns None if supernode.enabled=false or no providers.
//! - reset_stale_processing_tasks() MUST run before TaskWorker to recover orphaned tasks
//! - TaskWorker polls cognitive_tasks table and dispatches to LlmRouter
//! - MpiState.llm_router gates all /supernode/* endpoints and SuperNode status in /status
//!
//! ## Main Functionality
//! - Server lifecycle: init → run → shutdown
//! - Dual-engine MemChain initialization (SQLite+Vector + legacy MemPool+AOF)
//! - MPI API server with auth middleware
//! - Smart Miner with MVF integration + cognitive graph steps (v2.4.0)
//! - UDP/TUN packet handling
//! - Management reporting (heartbeat, sessions, commands, WebSocket)
//!
//! ## Dependencies
//! - config.rs for ServerConfig (including MemChainConfig with v2.4.0 NER/graph/entropy config,
//!   VectorQuantizationMode enum, v2.4.0+Reranker config, v2.5.0+SuperNode supernode config)
//! - storage.rs for MemoryStorage (Schema v6 with cognitive graph + SuperNode tables)
//! - mpi.rs for MPI router (with unified auth middleware)
//! - ner.rs for NerEngine (v2.4.0)
//! - reranker.rs for RerankerEngine (v2.4.0+Reranker)
//! - embed.rs for EmbedEngine
//! - vector.rs for VectorIndex (v2.4.0: with_config, calibrate_partition, restore_quantizer)
//! - quantize.rs for ScalarQuantizer (v2.4.0: used internally by VectorIndex)
//! - llm_router.rs for LlmRouter (v2.5.0+SuperNode)
//! - task_worker.rs for TaskWorker (v2.5.0+SuperNode)
//! - All other server subsystems
//!
//! ⚠️ Important Note for Next Developer:
//! - record_key is derived from identity.to_bytes() (Ed25519 PRIVATE key)
//! - Do NOT use public_key_bytes() for record_key derivation (security critical)
//! - rawlog_key is ALSO derived from identity.to_bytes() (same PRIVATE key source)
//!   but uses derive_rawlog_key() — a different KDF from derive_record_key().
//!   Do NOT confuse the two — they are separate keys for separate encryption contexts.
//! - api_secret flows: config.rs → server.rs → MpiState → auth middleware
//! - allow_remote_storage + max_remote_owners flow: config.rs → server.rs → MpiState
//!   → unified_auth_middleware (Ed25519 signature verification + capacity check)
//! - NerEngine MUST be loaded AFTER EmbedEngine — both share ORT runtime via Once.
//!   EmbedEngine::load() calls init_ort_runtime(). NerEngine::load() relies on it.
//!   If EmbedEngine is not loaded (model missing), NerEngine will also fail unless
//!   ORT is initialized by other means.
//! - RerankerEngine MUST be loaded AFTER EmbedEngine and NerEngine (shared ORT Once).
//!   Load order: EmbedEngine → NerEngine → RerankerEngine. All three are optional.
//! - RerankerEngine::load() is ~22MB model, loads in < 100ms. No calibration needed.
//!   Unlike VectorIndex quantizer, reranker has no persistent state to save/restore.
//! - VectorIndex scalar quantization requires calibration AFTER index rebuild.
//!   Sequence: rebuild vectors → try restore_quantizer → fallback calibrate_partition.
//!   If no vectors exist at startup, calibration is skipped (deferred to first Miner tick).
//! - config.memchain.vector_quantization: VectorQuantizationMode enum (None / ScalarUint8)
//! - config.memchain.vector_saturation_threshold: usize (early termination window size)
//!   Note: VectorIndex::with_config() accepts f32 saturation_threshold (score improvement
//!   delta), which is different from the config's window-based threshold. The server
//!   passes DEFAULT_SATURATION_THRESHOLD (0.001) from vector.rs when quantization is
//!   enabled, and uses the config value only for early_termination window control.
//!   TODO(Phase C): Unify these two threshold semantics or split the config field.
//! - SuperNode init order: init_llm_router() → reset_stale_processing_tasks() → TaskWorker::spawn()
//!   This order is MANDATORY. reset_stale must run before TaskWorker to avoid re-claiming
//!   tasks that were already being processed at the previous crash point.
//! - is_supernode_enabled() check gates all SuperNode code paths. Setting enabled=false
//!   in config gives exact v2.4.0 behavior with zero SuperNode code paths active.
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Added NerEngine initialization, MpiState extension,
//!   Miner with_ner_engine attachment
//! v2.4.0-GraphCognition Phase B - 🌟 VectorIndex::with_config() integration,
//!   calibrate_partition() after index rebuild, quantizer persistence to chain_state
//! v2.4.0+Reranker - 🌟 RerankerEngine initialization, MpiState.reranker_engine field
//! v2.4.0+Conversation - 🌟 derive_rawlog_key import, MpiState.rawlog_key field
//! v2.5.0+SuperNode - 🌟 init_llm_router(), reset_stale_processing_tasks() on startup,
//!   TaskWorker spawn, MpiState.llm_router, ReflectionMiner.with_llm_router()

use std::net::Ipv4Addr;
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

use crate::api::mpi::{build_mpi_router, MpiState, BaselineSnapshot};
use crate::config::{MemChainConfig, ServerConfig, VectorQuantizationMode};
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
// v2.4.0+Conversation: rawlog key for conversation replay decryption
// ⚠️ derive_rawlog_key uses a different KDF from derive_record_key — do NOT swap them
use crate::services::memchain::derive_rawlog_key;
use crate::services::memchain::EmbedEngine;
// v2.4.0: NerEngine for cognitive graph pipeline
use crate::services::memchain::NerEngine;
// v2.4.0+Reranker: Cross-encoder reranker for recall Step 3.5
use crate::services::memchain::RerankerEngine;
// v2.5.0+SuperNode: LLM routing + async task worker
use crate::services::memchain::{LlmRouter, TaskWorker};
use crate::services::{HandshakeService, IpPoolService, RoutingService, SessionManager};

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE: usize = 17;

#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

const COMMAND_CHANNEL_BUFFER: usize = 100;

/// chain_state key for persisted quantizer calibration data.
const QUANTIZER_CAL_KEY_PREFIX: &str = "quantizer_cal";

// ============================================
// Server
// ============================================

pub struct Server {
    config: ServerConfig,
    identity: IdentityKeyPair,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    pub fn new(config: ServerConfig, identity: IdentityKeyPair) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            config,
            identity,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting AeroNyx server v{}", env!("CARGO_PKG_VERSION"));

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

        // ============================================
        // Initialize MemChain dual-engine storage
        // ============================================
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

        // ============================================
        // Start MemChain API + Miner + SuperNode
        // ============================================
        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            let user_weights = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));

            {
                let owner = self.identity.public_key_bytes();
                if let Some(blob) = st.load_user_weights(&owner).await {
                    if let Some(w) = crate::services::memchain::mvf::WeightVector::from_bytes(&blob) {
                        let mut map = user_weights.write();
                        map.insert(hex::encode(owner), w);
                        info!("[MEMCHAIN] Loaded MVF user weights from SQLite");
                    }
                }
            }

            let mvf_baseline: Option<BaselineSnapshot> = {
                let conn = st.conn_lock().await;
                let raw: Option<Vec<u8>> = conn.query_row(
                    "SELECT value FROM chain_state WHERE key = 'mvf_baseline'",
                    [],
                    |row: &rusqlite::Row<'_>| row.get::<_, Vec<u8>>(0),
                ).optional().unwrap_or(None);
                drop(conn);

                raw.and_then(|bytes| {
                    let s = String::from_utf8_lossy(&bytes);
                    serde_json::from_str::<BaselineSnapshot>(&s).ok()
                })
            };

            let owner_key = self.identity.public_key_bytes();
            let api_secret = self.config.memchain.effective_api_secret().map(|s| s.to_string());

            // ── Embedding engine ────────────────────────────────────────────
            let embed_engine: Option<Arc<EmbedEngine>> = {
                let model_path = &self.config.memchain.embed_model_path;
                let max_tokens = self.config.memchain.embed_max_tokens;
                let output_dim = self.config.memchain.embed_output_dim;
                match EmbedEngine::load(model_path, max_tokens, output_dim) {
                    Ok(engine) => {
                        info!(
                            model = %model_path,
                            model_type = %engine.model_type(),
                            dim = engine.dim(),
                            max_seq = engine.max_seq_length(),
                            "[EMBED] ✅ Local embedding engine loaded"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        warn!(
                            model = %model_path, error = %e,
                            "[EMBED] ⚠️ Local embedding unavailable — /embed returns 503, \
                             Miner falls back to OpenClaw Gateway"
                        );
                        None
                    }
                }
            };

            // ── NER engine (v2.4.0) ─────────────────────────────────────────
            // Must be loaded AFTER EmbedEngine (shared ORT runtime via Once)
            let ner_engine: Option<Arc<NerEngine>> = if self.config.memchain.ner_enabled {
                let model_path = &self.config.memchain.ner_model_path;
                let threshold = self.config.memchain.ner_confidence_threshold;
                match NerEngine::load(model_path, threshold, 0) {
                    Ok(engine) => {
                        info!(
                            model = %model_path, threshold = threshold,
                            max_width = engine.max_width(),
                            "[NER] ✅ Local NER engine loaded (GLiNER)"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        warn!(
                            model = %model_path, error = %e,
                            "[NER] ⚠️ Local NER unavailable — cognitive graph pipeline disabled. \
                             Run `scripts/download_models.sh` to download GLiNER model."
                        );
                        None
                    }
                }
            } else {
                debug!("[NER] Disabled (ner_enabled=false)");
                None
            };

            // ── Reranker engine (v2.4.0+Reranker) ──────────────────────────
            // Must be loaded AFTER EmbedEngine + NerEngine (all share ORT Once)
            let reranker_engine: Option<Arc<RerankerEngine>> = if self.config.memchain.reranker_enabled {
                let model_path = &self.config.memchain.reranker_model_path;
                let max_seq = self.config.memchain.reranker_max_seq_length;
                match RerankerEngine::load(model_path, max_seq) {
                    Ok(engine) => {
                        info!(
                            model = %model_path, max_seq = max_seq,
                            blend_weight = %RerankerEngine::blend_weight(),
                            "[RERANKER] ✅ Cross-encoder loaded"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        warn!(
                            model = %model_path, error = %e,
                            "[RERANKER] ⚠️ Cross-encoder unavailable — recall uses RRF-only. \
                             Run `scripts/download_models.sh --reranker-only` to download model."
                        );
                        None
                    }
                }
            } else {
                debug!("[RERANKER] Disabled (reranker_enabled=false)");
                None
            };

            // ── v2.5.0+SuperNode: LlmRouter initialization ──────────────────
            //
            // init_llm_router() reads supernode config and constructs the router.
            // Returns None if:
            //   - supernode.enabled = false
            //   - no providers configured
            //   - all providers fail validation
            //
            // ⚠️ SuperNode init order is MANDATORY:
            //   1. init_llm_router()             — build router
            //   2. reset_stale_processing_tasks() — recover orphaned tasks
            //   3. TaskWorker::spawn()            — start processing
            // Do NOT reorder these three steps.
            let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router().await;

            // ── v2.5.0+SuperNode: Crash recovery ───────────────────────────
            // Reset tasks that were stuck in 'processing' at the last crash/restart.
            // Must run BEFORE TaskWorker spawns to avoid race with re-claiming.
            // Uses task_timeout_secs from worker config as the stale threshold.
            if llm_router.is_some() {
                let timeout_secs = self.config.memchain.supernode.worker.task_timeout_secs as i64;
                let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
                if recovered > 0 {
                    info!(
                        recovered = recovered,
                        timeout_secs = timeout_secs,
                        "[SUPERNODE] Recovered stale tasks from previous run"
                    );
                }
            }

            let mpi_state = Arc::new(MpiState {
                storage: Arc::clone(st),
                vector_index: Arc::clone(vi),
                identity: self.identity.clone(),
                identity_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
                index_ready: std::sync::atomic::AtomicBool::new(false),
                user_weights: Arc::clone(&user_weights),
                mvf_alpha: self.config.memchain.mvf_alpha,
                mvf_enabled: self.config.memchain.mvf_enabled,
                session_embeddings: parking_lot::RwLock::new(std::collections::HashMap::new()),
                mvf_baseline: parking_lot::RwLock::new(mvf_baseline),
                owner_key,
                api_secret,
                embed_engine: embed_engine.clone(),
                // v2.3.0: Phase 1 Remote Storage
                allow_remote_storage: self.config.memchain.allow_remote_storage,
                max_remote_owners: self.config.memchain.max_remote_owners,
                // v2.4.0: Cognitive Graph Pipeline
                ner_engine: ner_engine.clone(),
                graph_enabled: self.config.memchain.graph_enabled,
                entropy_filter_enabled: self.config.memchain.entropy_filter_enabled,
                // v2.4.0+Reranker
                reranker_engine,
                // v2.4.0+Conversation: RawLog decryption key for conversation replay
                // ⚠️ Uses derive_rawlog_key(), NOT derive_record_key() — different KDF contexts
                rawlog_key: Some(derive_rawlog_key(&self.identity.to_bytes())),
                // v2.5.0+SuperNode: LLM router for /supernode/* endpoints + /status
                // None when supernode.enabled=false — all SuperNode endpoints return 404
                llm_router: llm_router.clone(),
            });

            // Pre-populate identity cache
            {
                let owner_hex = hex::encode(owner_key);
                let identity_records = st
                    .get_active_records(&owner_key, Some(aeronyx_core::ledger::MemoryLayer::Identity), 100)
                    .await;
                if !identity_records.is_empty() {
                    let mut cache = mpi_state.identity_cache.write();
                    cache.insert(owner_hex, identity_records);
                }
            }

            mpi_state.index_ready.store(true, std::sync::atomic::Ordering::Relaxed);

            // Freeze MVF baseline if needed
            if self.config.memchain.mvf_enabled && mpi_state.mvf_baseline.read().is_none() {
                let feedback = st.get_recent_feedback(200).await;
                if !feedback.is_empty() {
                    let positive = feedback.iter().filter(|(s, _)| *s == 1).count();
                    let rate = positive as f32 / feedback.len() as f32;
                    let now_ts = SystemTime::now().duration_since(UNIX_EPOCH)
                        .unwrap_or_default().as_secs() as i64;

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
                    info!(rate = rate, samples = feedback.len(), "[MVF] Baseline frozen");
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

            // ── v2.5.0+SuperNode: TaskWorker spawn ─────────────────────────
            // Spawned AFTER reset_stale_processing_tasks() to avoid claiming
            // tasks that were just recovered.
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

            // ── Smart Miner (with MVF + v2.4.0 cognitive graph + v2.5.0 SuperNode) ──
            if self.config.memchain.miner_interval_secs > 0 {
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

                // Attach local embed engine if available
                let miner = if let Some(ref ee) = embed_engine {
                    miner.with_embed_engine(Arc::clone(ee))
                } else {
                    miner
                };

                // v2.4.0: Attach NER engine for cognitive graph Steps 7-11
                let miner = if let Some(ref ne) = ner_engine {
                    miner.with_ner_engine(Arc::clone(ne))
                } else {
                    miner
                };

                // v2.5.0+SuperNode: Attach LlmRouter for Steps 8/9/10 task enqueue.
                // When None, Steps 8/9/10 run without SuperNode enqueue (v2.4.0 behavior).
                let miner = if let Some(ref lr) = llm_router {
                    miner.with_llm_router(Arc::clone(lr))
                } else {
                    miner
                };

                let miner_shutdown = self.shutdown_tx.subscribe();
                tasks.push(("miner", tokio::spawn(async move {
                    miner.run(miner_shutdown).await;
                })));
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

        if let (Some(ref st), Some(ref mp), Some(ref aw)) = (&storage, &mempool, &aof_writer) {
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
    // v2.5.0+SuperNode: LlmRouter Initialization
    // ============================================

    /// Initialize LlmRouter from supernode config.
    ///
    /// ## Returns
    /// - `Some(Arc<LlmRouter>)` — SuperNode enabled, at least one provider configured
    /// - `None` — SuperNode disabled OR no valid providers
    ///
    /// ## Provider Construction Order
    /// 1. Read `supernode.providers[]` from config
    /// 2. Construct each provider (OpenAiCompatProvider or AnthropicProvider)
    /// 3. Build LlmRouter with provider list + routing config
    ///
    /// ## Graceful Failure
    /// If a provider fails to construct (e.g., missing api_key), it is skipped
    /// with a warning. The router is returned if at least one provider succeeded.
    /// If all providers fail, returns None.
    ///
    /// ⚠️ This method does NOT ping providers or validate connectivity.
    ///    Health checks happen at runtime via /supernode/health endpoint.
    async fn init_llm_router(&self) -> Option<Arc<LlmRouter>> {
        // config_supernode is declared at crate root in lib.rs (not under config/)
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

        // LlmRouter::new() expects Vec<(name, api_base, model, Arc<dyn LlmProvider>)>
        // api_base + model are stored separately for HTTP HEAD health checks.
        let mut providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = Vec::new();

        for provider_cfg in &supernode.providers {
            // Resolve $ENV_VAR syntax in api_key
            let api_key: Option<String> = provider_cfg.api_key.as_ref().map(|k| {
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

            // Resolve Anthropic default api_base when empty
            // (validate() in config_supernode.rs warns but doesn't fill it in)
            let api_base = if provider_cfg.api_base.is_empty()
                && provider_cfg.provider_type == ProviderType::Anthropic
            {
                "https://api.anthropic.com".to_string()
            } else {
                provider_cfg.api_base.clone()
            };

            let provider: Arc<dyn LlmProvider> = match provider_cfg.provider_type {
                ProviderType::OpenaiCompatible => {
                    Arc::new(OpenAiCompatProvider::new(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        api_key,
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ))
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
                    Arc::new(AnthropicProvider::new(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        key,
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ))
                }
            };

            info!(
                name = %provider_cfg.name,
                type_ = ?provider_cfg.provider_type,
                model = %provider_cfg.model,
                api_base = %api_base,
                "[SUPERNODE] Provider registered"
            );

            // 4-tuple: (name, api_base, model, provider)
            // api_base and model stored separately for health check pings
            providers.push((
                provider_cfg.name.clone(),
                api_base,
                provider_cfg.model.clone(),
                provider,
            ));
        }

        if providers.is_empty() {
            warn!("[SUPERNODE] All providers failed to construct — SuperNode disabled");
            return None;
        }

        // Pass TaskRoutingConfig directly — LlmRouter::new() calls provider_for()
        // for each CognitiveTaskType to build its internal routing table.
        let router = LlmRouter::new(providers, supernode.routing.clone());
        info!(
            providers = supernode.providers.len(),
            fallback = ?supernode.routing.fallback,
            "[SUPERNODE] ✅ LlmRouter initialized"
        );
        Some(Arc::new(router))
    }

    // ============================================
    // MemChain Initialization (dual-engine)
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

        // ⚠️ SECURITY: identity.to_bytes() returns the PRIVATE key (32 bytes).
        //    Do NOT use public_key_bytes() here.
        let record_key = derive_record_key(&self.identity.to_bytes());
        info!("[MEMCHAIN] Record content encryption enabled (key derived from identity)");

        let storage = Arc::new(
            MemoryStorage::open(db_path, Some(record_key))
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );

        // v2.4.0 Phase B: VectorIndex with quantization config
        let quantization_enabled = self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let vector_index = Arc::new(if quantization_enabled {
            let sat_threshold = if self.config.memchain.vector_early_termination { 0.001_f32 } else { 0.0_f32 };
            info!(quantization = "scalar_uint8", "[MEMCHAIN] VectorIndex with scalar quantization");
            VectorIndex::with_config(true, sat_threshold)
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
            db = %db_path, records = storage.count().await, vectors = rebuild_count,
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );

        // Quantizer calibration (v2.4.0 Phase B)
        if quantization_enabled && rebuild_count > 0 {
            let owner_hex = hex::encode(owner);
            let model_name = std::path::Path::new(&self.config.memchain.embed_model_path)
                .file_name().and_then(|f| f.to_str()).unwrap_or("minilm-l6-v2");

            let cal_key = format!("{}:{}:{}", QUANTIZER_CAL_KEY_PREFIX, owner_hex, model_name);
            let restored = {
                let conn = storage.conn_lock().await;
                let cal_data: Option<Vec<u8>> = conn.query_row(
                    "SELECT value FROM chain_state WHERE key = ?1",
                    rusqlite::params![cal_key],
                    |row| row.get::<_, Vec<u8>>(0),
                ).optional().unwrap_or(None);
                drop(conn);

                if let Some(data) = cal_data {
                    let ok = vector_index.restore_quantizer(&owner, model_name, &data);
                    if ok {
                        info!(model = model_name, "[VECTOR] ✅ Quantizer restored from persisted calibration");
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
                    info!(model = model_name, vectors = rebuild_count, "[VECTOR] ✅ Quantizer calibrated and persisted");
                }
            }
        }

        // Legacy: AOF + MemPool
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
    // Combined API Server (MPI + Legacy)
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
                Ok(l) => { info!("[API] MemChain API on http://{}", listen_addr); l }
                Err(e) => { error!("[API] Bind failed {}: {}", listen_addr, e); return; }
            };

            let server = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.recv().await;
                    info!("[API] Shutdown signal received");
                });

            if let Err(e) = server.await { error!("[API] Server error: {}", e); }
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

        let agent_manager = Arc::new(crate::services::AgentManager::new(Arc::clone(&mgmt_client)));
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
        let cmd_handler = CommandHandler::new(cmd_rx, Arc::clone(&mgmt_client), Arc::clone(&agent_manager));
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

        let mut heartbeat = HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
            .with_command_sender(cmd_tx)
            .with_agent_manager(Arc::clone(&agent_manager));

        if let Some(f) = memchain_status_fn {
            heartbeat = heartbeat.with_memchain_status(f);
        }

        let sess = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            heartbeat.run(move || sess.count() as u32, hb_shutdown).await;
        });

        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let sr_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { session_reporter.run(sr_shutdown).await; });

        let node_info_path = &self.config.management.node_info_path;
        if let Ok(node_info) = crate::management::models::StoredNodeInfo::load(node_info_path) {
            let ws = crate::management::WsTunnel::new(
                self.identity.clone(),
                node_info.node_id.clone(),
                Arc::clone(&agent_manager),
            )
            .with_mpi_api_secret(self.config.memchain.effective_api_secret().map(|s| s.to_string()));
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

        let services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
        ];

        if let Ok(client) = reqwest::Client::builder().timeout(Duration::from_secs(5)).build() {
            for url in &services {
                let mut req = client.get(*url);
                if url.contains("metadata.google.internal") {
                    req = req.header("Metadata-Flavor", "Google");
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        if let Ok(body) = resp.text().await {
                            let ip = body.trim().to_string();
                            if !ip.is_empty() && ip.len() <= 45 && ip.parse::<std::net::IpAddr>().is_ok() {
                                info!(ip = %ip, source = %url, "[NET] Public IP detected");
                                return ip;
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

    fn init_services(&self) -> Result<(Arc<IpPoolService>, Arc<SessionManager>, Arc<RoutingService>)> {
        let (network, prefix) = self.config.parse_ip_range()?;
        let ip_pool = Arc::new(IpPoolService::new(network, prefix, self.config.gateway_ip())?);
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
        let tun = LinuxTun::create(cfg).await
            .map_err(|e| ServerError::startup_failed(format!("TUN: {}", e)))?;
        tun.up().await
            .map_err(|e| ServerError::startup_failed(format!("TUN up: {}", e)))?;
        info!("TUN '{}' initialized @ {}", tun.name(), self.config.gateway_ip());
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
                                                    let sid = BASE64.encode(&result.response.session_id);
                                                    session_events.session_created(&sid, None);
                                                    let resp = encode_server_hello(&result.response);
                                                    let _ = udp.send(&resp, &source.addr).await;
                                                }
                                                Err(e) => warn!("[HANDSHAKE] Failed {}: {}", source.addr, e),
                                            }
                                        }
                                    }

                                    Ok(MessageType::Keepalive) => {
                                        if len >= KEEPALIVE_PACKET_SIZE {
                                            let mut sid = [0u8; 16];
                                            sid.copy_from_slice(&data[1..17]);
                                            if let Some(id) = SessionId::from_bytes(&sid) {
                                                if let Some(s) = sessions.get(&id) { s.touch(); }
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
                                                if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw, &storage, &vector_index,
                                                        &memchain_config, &server_pubkey_hex,
                                                        &session, &udp_reply, &crypto,
                                                    ).await;
                                                }
                                            }
                                            Err(_) => {}
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) { error!("UDP recv error: {}", e); }
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
                if !sig_ok { warn!("[MEMCHAIN] BroadcastFact sig failed"); return; }
                if !config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                    warn!("[MEMCHAIN] BroadcastFact untrusted origin"); return;
                }
                if mempool.add_fact(fact.clone()) {
                    let mut w = aof_writer.lock().await;
                    let _ = w.append_fact(&fact).await;
                }
            }

            MemChainMessage::BroadcastRecord(record) => {
                let owner_hex = record.owner_hex();
                let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                    Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok { warn!("[MEMCHAIN] BroadcastRecord sig failed"); return; }
                if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                    warn!("[MEMCHAIN] BroadcastRecord untrusted owner"); return;
                }
                if let Some(ref st) = storage {
                    if st.insert(&record, "p2p-remote").await {
                        info!(id = hex::encode(record.record_id), "[MEMCHAIN] BroadcastRecord stored");
                        if record.has_embedding() {
                            if let Some(ref vi) = vector_index {
                                vi.upsert(record.record_id, record.embedding.clone(),
                                    record.layer, record.timestamp, &record.owner, "p2p-remote");
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
                    if !sig_ok || !config.is_origin_trusted(&origin_hex, server_pubkey_hex) { continue; }
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
                            Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                            Err(_) => false,
                        };
                        if !sig_ok || !config.is_origin_trusted(&owner_hex, server_pubkey_hex) { continue; }
                        let _ = st.insert(&record, "p2p-sync").await;
                    }
                }
            }

            MemChainMessage::BlockAnnounce(header) => {
                info!(
                    height = header.height, hash = hex::encode(header.hash()),
                    "[MEMCHAIN] BlockAnnounce received"
                );
            }

            _ => { debug!("[MEMCHAIN] Unhandled message variant"); }
        }
    }

    // ============================================
    // MemChain UDP send helper
    // ============================================

    async fn send_to_session(
        msg: &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        let plaintext = match encode_memchain(msg) {
            Ok(p) => p,
            Err(e) => { error!("[MEMCHAIN_TX] Encode: {}", e); return; }
        };

        let counter = session.next_tx_counter();
        let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];

        let len = match crypto.encrypt(
            &session.session_key, counter, session.id.as_bytes(), &plaintext, &mut encrypted,
        ) {
            Ok(l) => l,
            Err(e) => { error!("[MEMCHAIN_TX] Encrypt: {}", e); return; }
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
                                if !shutdown.load(Ordering::SeqCst) { error!("TUN: {}", e); }
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
        tokio::signal::ctrl_c().await.expect("Ctrl+C listener failed");
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
            .finish()
    }
}
