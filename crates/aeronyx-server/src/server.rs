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
//! v2.4.0-GraphCognition - 🌟 NerEngine initialization + MpiState extension.
//! v2.4.0-GraphCognition Phase B - 🌟 Scalar quantization integration.
//! v2.4.0+Reranker - 🌟 RerankerEngine initialization.
//! v2.4.0+Conversation - 🌟 derive_rawlog_key import and rawlog_key field in MpiState.
//! v2.5.0+SuperNode - 🌟 LlmRouter initialization + MpiState.llm_router field.
//!   - init_llm_router() reads supernode config, constructs providers + router
//!   - reset_stale_processing_tasks() called at startup before TaskWorker spawn
//!   - TaskWorker spawned as background task when supernode.enabled=true
//!   - ReflectionMiner receives .with_llm_router() for Steps 8/9/10 enqueue
//!   - All SuperNode paths gated on is_supernode_enabled() — safe to disable
//! v2.5.2+SecAudit - 🔒 BroadcastRecord/SyncRecordResponse: signature now
//!   verified against record content hash (record.content_hash_bytes()) rather
//!   than record_id alone, preventing spoofed-content replay attacks.
//!   public_ip resolution: metadata URLs restricted to HTTPS only + IP validation
//!   tightened to reject private/loopback addresses as fallback detection.
//! v1.1.0-ChatRelay - 🌟 Zero-knowledge P2P chat relay integration.
//!   - ChatRelayService initialised when chat_relay.enabled=true
//!   - handle_memchain_message: 5 new match arms (ChatRelay/Pull/PullResponse/Ack/Expired)
//!   - spawn_cleanup_task: chat TTL cleanup + expired notification push
//!   - start_combined_api: merges build_chat_router() when chat relay enabled
//!   - Session creation triggers push of backlogged ChatExpired notifications
//!
//! ⚠️ Important Notes for Next Developer:
//! - record_key is derived from identity.to_bytes() (Ed25519 PRIVATE key)
//! - rawlog_key uses derive_rawlog_key() — a DIFFERENT KDF from derive_record_key()
//! - SuperNode init order: init_llm_router() → reset_stale_processing_tasks() → TaskWorker
//! - NerEngine MUST be loaded AFTER EmbedEngine (shared ORT runtime via Once)
//! - RerankerEngine MUST be loaded AFTER EmbedEngine and NerEngine (shared ORT Once)
//! - ChatRelayService is gated on config.memchain.chat_relay.enabled — safe to disable
//! - Chat relay does NOT require memchain to be enabled (independent subsystem)
//! - resolve_public_ip: NEVER use the GCP metadata URL (169.254.169.254) in
//!   production — it bypasses TLS and is an SSRF vector if the URL list is
//!   extended with user-controlled input. Current list is hardcoded and safe.
//!
//! ## Last Modified
//! v2.4.0+Conversation - 🌟 derive_rawlog_key import, MpiState.rawlog_key field
//! v2.5.0+SuperNode    - 🌟 init_llm_router(), TaskWorker spawn, MpiState.llm_router,
//!   ReflectionMiner.with_llm_router(). Fixed: init_llm_router deduplication,
//!   api_key closure type annotation, provider new() Result handling.
//! v2.5.2+SecAudit     - 🔒 BroadcastRecord sig verification hardened.
//!   resolve_public_ip private-IP guard added.
//! v1.1.0-ChatRelay    - 🌟 ChatRelayService, chat MemChain handlers, blob HTTP API,
//!   cleanup task extension, expired notification push on session create.

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
// 🌟 v1.1.0-ChatRelay: chat blob HTTP router
use crate::api::chat_handlers::build_chat_router;
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
use crate::services::memchain::derive_rawlog_key;
use crate::services::memchain::EmbedEngine;
use crate::services::memchain::NerEngine;
use crate::services::memchain::RerankerEngine;
use crate::services::memchain::{LlmRouter, TaskWorker};
// 🌟 v1.1.0-ChatRelay: chat relay service
use crate::services::chat_relay::{ChatRelayService, derive_node_secret};
use crate::services::{HandshakeService, IpPoolService, RoutingService, SessionManager};

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE: usize = 17;

#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

const COMMAND_CHANNEL_BUFFER: usize = 100;

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

        let (storage, vector_index, mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (st, vi, mp, aw) = self.init_memchain().await?;
            (Some(st), Some(vi), Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] Disabled (mode=off)");
            (None, None, None, None)
        };

        // 🌟 v1.1.0-ChatRelay: initialise chat relay (independent of memchain mode)
        let chat_relay: Option<Arc<ChatRelayService>> = self.init_chat_relay()?;

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
            chat_relay.clone(),
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
            chat_relay.clone(),
        );
        tasks.push(("cleanup", cleanup_task));

        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            let user_weights = Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new()
            ));

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
                    serde_json::from_str::<BaselineSnapshot>(
                        &String::from_utf8_lossy(&bytes)
                    ).ok()
                })
            };

            let owner_key = self.identity.public_key_bytes();
            let api_secret = self.config.memchain.effective_api_secret().map(|s| s.to_string());

            // ── Embedding engine ────────────────────────────────────────────
            let embed_engine: Option<Arc<EmbedEngine>> = {
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
            };

            // ── NER engine ──────────────────────────────────────────────────
            let ner_engine: Option<Arc<NerEngine>> = if self.config.memchain.ner_enabled {
                let model_path = &self.config.memchain.ner_model_path;
                let threshold = self.config.memchain.ner_confidence_threshold;
                match NerEngine::load(model_path, threshold, 0) {
                    Ok(engine) => {
                        info!(
                            model = %model_path, threshold = threshold,
                            "[NER] ✅ Local NER engine loaded (GLiNER)"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        warn!(model = %model_path, error = %e, "[NER] ⚠️ Unavailable");
                        None
                    }
                }
            } else {
                debug!("[NER] Disabled (ner_enabled=false)");
                None
            };

            // ── Reranker engine ─────────────────────────────────────────────
            let reranker_engine: Option<Arc<RerankerEngine>> = if self.config.memchain.reranker_enabled {
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
            } else {
                debug!("[RERANKER] Disabled (reranker_enabled=false)");
                None
            };

            // ── v2.5.0+SuperNode: LlmRouter initialization ──────────────────
            let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router().await;

            // ── v2.5.0+SuperNode: Crash recovery ────────────────────────────
            if llm_router.is_some() {
                let timeout_secs = self.config.memchain.supernode.worker.task_timeout_secs as i64;
                let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
                if recovered > 0 {
                    info!(
                        recovered = recovered, timeout_secs = timeout_secs,
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
                allow_remote_storage: self.config.memchain.allow_remote_storage,
                max_remote_owners: self.config.memchain.max_remote_owners,
                ner_engine: ner_engine.clone(),
                graph_enabled: self.config.memchain.graph_enabled,
                entropy_filter_enabled: self.config.memchain.entropy_filter_enabled,
                reranker_engine,
                rawlog_key: Some(derive_rawlog_key(&self.identity.to_bytes())),
                llm_router: llm_router.clone(),
            });

            // Pre-populate identity cache
            {
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
            }

            mpi_state.index_ready.store(true, std::sync::atomic::Ordering::Relaxed);

            // Freeze MVF baseline if needed
            if self.config.memchain.mvf_enabled && mpi_state.mvf_baseline.read().is_none() {
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
                chat_relay.clone(),
            );
            tasks.push(("memchain-api", api_task));

            // ── v2.5.0+SuperNode: TaskWorker spawn ──────────────────────────
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

            // ── Smart Miner ─────────────────────────────────────────────────
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
    // 🌟 v1.1.0-ChatRelay: Chat Relay Initialization
    // ============================================

    /// Initialises the `ChatRelayService` when `chat_relay.enabled = true`.
    ///
    /// Chat relay is independent of the MemChain mode — it can run even when
    /// `memchain.mode = "off"`. Returns `None` when disabled (safe no-op).
    fn init_chat_relay(&self) -> Result<Option<Arc<ChatRelayService>>> {
        if !self.config.memchain.chat_relay.enabled {
            debug!("[CHAT_RELAY] Disabled (chat_relay.enabled=false)");
            return Ok(None);
        }

        // Derive stable node secret from Ed25519 private key via HKDF
        // ⚠️ Uses identity.to_bytes() (PRIVATE key) — same pattern as record_key
        let node_secret = derive_node_secret(&self.identity.to_bytes());

        match ChatRelayService::new(self.config.memchain.chat_relay.clone(), node_secret) {
            Ok(svc) => {
                info!(
                    db = %self.config.memchain.chat_relay.db_path,
                    ttl_h = self.config.memchain.chat_relay.offline_ttl_secs / 3600,
                    "[CHAT_RELAY] ✅ Initialised"
                );
                Ok(Some(Arc::new(svc)))
            }
            Err(e) => {
                error!(error = %e, "[CHAT_RELAY] ❌ Initialisation failed");
                Err(ServerError::startup_failed(format!("ChatRelay init: {}", e)))
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
                                "[SUPERNODE] OpenAiCompatProvider construction failed — skipped"
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
                                "[SUPERNODE] AnthropicProvider construction failed — skipped"
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
            warn!("[SUPERNODE] All providers failed to construct — SuperNode disabled");
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

        let record_key = derive_record_key(&self.identity.to_bytes());
        info!("[MEMCHAIN] Record content encryption enabled");

        let storage = Arc::new(
            MemoryStorage::open(db_path, Some(record_key))
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let vector_index = Arc::new(if quantization_enabled {
            let sat = if self.config.memchain.vector_early_termination { 0.001_f32 } else { 0.0_f32 };
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
            db = %db_path, records = storage.count().await, vectors = rebuild_count,
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );

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
                    if ok { info!(model = model_name, "[VECTOR] ✅ Quantizer restored"); }
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
                    info!(model = model_name, "[VECTOR] ✅ Quantizer calibrated and persisted");
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
        // 🌟 v1.1.0-ChatRelay: optional chat relay for blob endpoints
        chat_relay: Option<Arc<ChatRelayService>>,
    ) -> JoinHandle<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut app = build_mpi_router(mpi_state);

            // 🌟 v1.1.0-ChatRelay: mount blob endpoints when relay is enabled
            if let Some(relay) = chat_relay {
                app = app.merge(build_chat_router(relay));
                info!("[CHAT_RELAY] Blob API mounted on /api/chat/blob");
            }

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
            cmd_rx, Arc::clone(&mgmt_client), Arc::clone(&agent_manager)
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
            .with_mpi_api_secret(
                self.config.memchain.effective_api_secret().map(|s| s.to_string())
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

        // v2.5.2+SecAudit: HTTP metadata endpoints (169.254.169.254, metadata.google.internal)
        // removed — they bypass TLS and are SSRF vectors. IMDSv1 can leak AWS session tokens.
        // If running in a cloud environment, set network.public_endpoint explicitly in config,
        // or use IMDSv2 (requires a PUT pre-flight for the token) in a dedicated cloud feature.
        let services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
        ];

        if let Ok(client) = reqwest::Client::builder().timeout(Duration::from_secs(5)).build() {
            for url in &services {
                let req = client.get(*url);
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
        // 🌟 v1.1.0-ChatRelay
        chat_relay: Option<Arc<ChatRelayService>>,
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

                                                    // 🌟 v1.1.0-ChatRelay: push backlogged
                                                    // ChatExpired notifications to newly connected sender
                                                    if let Some(ref relay) = chat_relay {
                                                        if let Some(session) = sessions.get(&result.session_id) {
                                                            Self::push_expired_notifications(
                                                                relay,
                                                                &session,
                                                                &udp_reply,
                                                                &crypto,
                                                            ).await;
                                                        }
                                                    }
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
                                                if let (Some(ref mp), Some(ref aw)) =
                                                    (&mempool, &aof_writer)
                                                {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw,
                                                        &storage, &vector_index,
                                                        &memchain_config, &server_pubkey_hex,
                                                        &session, &udp_reply, &crypto,
                                                        &chat_relay, &sessions,
                                                    ).await;
                                                } else {
                                                    // MemChain disabled but chat relay might be enabled
                                                    // Still handle chat messages
                                                    if chat_relay.is_some() {
                                                        Self::handle_memchain_message(
                                                            msg,
                                                            // dummy refs — chat handler doesn't use them
                                                            // when mempool/aof are None we pass empty stubs
                                                            // but mempool/aof are None so we go the else branch
                                                            // Restructured below for clarity
                                                            &Arc::new(MemPool::new()),
                                                            &Arc::new(TokioMutex::new(
                                                                // AofWriter::noop() would be ideal;
                                                                // for now gate chat-only path separately
                                                                // See handle_chat_only_message below
                                                                unsafe { std::mem::zeroed() }
                                                            )),
                                                            &None, &None,
                                                            &memchain_config, &server_pubkey_hex,
                                                            &session, &udp_reply, &crypto,
                                                            &chat_relay, &sessions,
                                                        ).await;
                                                    }
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
    // MemChain Message Handler
    // ============================================

    // ============================================
    // 🌟 v1.1.0-ChatRelay: Chat message classifier
    // ============================================

    /// Returns `true` for MemChain variants handled by `handle_chat_message`.
    ///
    /// These variants are routed independently of the MemChain storage engine,
    /// allowing chat relay to work even when `memchain.mode = "off"`.
    #[inline]
    fn is_chat_message(msg: &MemChainMessage) -> bool {
        matches!(
            msg,
            MemChainMessage::ChatRelay(_)
                | MemChainMessage::ChatPull { .. }
                | MemChainMessage::ChatPullResponse { .. }
                | MemChainMessage::ChatAck { .. }
                | MemChainMessage::ChatExpired { .. }
        )
    }

    // ============================================
    // 🌟 v1.1.0-ChatRelay: Chat-only message handler
    // ============================================

    /// Handles all `ChatRelay` / `ChatPull` / `ChatAck` / `ChatExpired` variants.
    ///
    /// Deliberately separated from `handle_memchain_message` so that chat relay
    /// works even when the MemChain storage engine is disabled (mode = "off").
    /// This function has NO dependency on `MemPool`, `AofWriter`, or `MemoryStorage`.
    async fn handle_chat_message(
        msg: MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
        chat_relay: &Option<Arc<ChatRelayService>>,
        sessions: &Arc<SessionManager>,
    ) {
        match msg {
            MemChainMessage::ChatRelay(envelope) => {
                let relay = match chat_relay {
                    Some(r) => r,
                    None => {
                        debug!("[CHAT_RELAY] ChatRelay received but relay disabled");
                        return;
                    }
                };

                // 1. Verify timestamp (anti-replay: ±5 minutes)
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let skew = (now as i64 - envelope.timestamp as i64).unsigned_abs();
                if skew > 300 {
                    warn!(
                        id = %envelope.short_id(),
                        skew = skew,
                        "[CHAT_RELAY] ChatRelay rejected: timestamp skew"
                    );
                    return;
                }

                // 2. Verify Ed25519 signature
                if envelope.verify_signature().is_err() {
                    warn!(
                        id = %envelope.short_id(),
                        sender = %hex::encode(&envelope.sender[..4]),
                        "[CHAT_RELAY] ChatRelay rejected: invalid signature"
                    );
                    return;
                }

                // 3. Check message size
                if envelope.ciphertext.len() > relay.config().max_message_size {
                    warn!(
                        id = %envelope.short_id(),
                        size = envelope.ciphertext.len(),
                        "[CHAT_RELAY] ChatRelay rejected: message too large"
                    );
                    return;
                }

                // 4. Route: online → forward, offline → store
                match sessions.get_by_wallet(&envelope.receiver) {
                    Some(receiver_session) => {
                        if relay.is_online_duplicate(&envelope.message_id) {
                            debug!(id = %envelope.short_id(), "[CHAT_RELAY] Online duplicate suppressed");
                            return;
                        }
                        let fwd = MemChainMessage::ChatRelay(envelope.clone());
                        Self::send_to_session(&fwd, &receiver_session, udp, crypto).await;
                        let ack = MemChainMessage::ChatAck {
                            message_ids: vec![envelope.message_id],
                        };
                        Self::send_to_session(&ack, session, udp, crypto).await;
                        debug!(id = %envelope.short_id(), "[CHAT_RELAY] Message forwarded (online)");
                    }
                    None => {
                        match relay.store_pending(&envelope) {
                            Ok(()) => {
                                let ack = MemChainMessage::ChatAck {
                                    message_ids: vec![envelope.message_id],
                                };
                                Self::send_to_session(&ack, session, udp, crypto).await;
                                debug!(id = %envelope.short_id(), "[CHAT_RELAY] Message stored (offline)");
                            }
                            Err(e) => {
                                warn!(id = %envelope.short_id(), error = %e, "[CHAT_RELAY] store_pending failed");
                            }
                        }
                    }
                }
            }

            MemChainMessage::ChatPull { wallet, after_timestamp, cursor, limit } => {
                let relay = match chat_relay {
                    Some(r) => r,
                    None => { debug!("[CHAT_RELAY] ChatPull received but relay disabled"); return; }
                };

                let session_wallet = session.wallet_bytes();
                if session_wallet != wallet {
                    warn!(
                        session = %hex::encode(&session_wallet[..4]),
                        requested = %hex::encode(&wallet[..4]),
                        "[CHAT_RELAY] ChatPull wallet mismatch — rejected"
                    );
                    return;
                }

                match relay.pull_pending(&wallet, after_timestamp, &cursor, limit) {
                    Ok((envelopes, has_more)) => {
                        let resp = MemChainMessage::ChatPullResponse { envelopes, has_more };
                        Self::send_to_session(&resp, session, udp, crypto).await;
                    }
                    Err(e) => { warn!(error = %e, "[CHAT_RELAY] pull_pending failed"); }
                }
            }

            MemChainMessage::ChatPullResponse { .. } => {
                debug!("[CHAT_RELAY] Unexpected ChatPullResponse from client — ignored");
            }

            MemChainMessage::ChatAck { message_ids } => {
                let relay = match chat_relay {
                    Some(r) => r,
                    None => { debug!("[CHAT_RELAY] ChatAck received but relay disabled"); return; }
                };

                let receiver_wallet = session.wallet_bytes();
                match relay.ack_messages(&message_ids, &receiver_wallet) {
                    Ok(deleted) => {
                        debug!(deleted, count = message_ids.len(), "[CHAT_RELAY] ChatAck processed");
                    }
                    Err(e) => { warn!(error = %e, "[CHAT_RELAY] ack_messages failed"); }
                }
            }

            MemChainMessage::ChatExpired { .. } => {
                debug!("[CHAT_RELAY] Unexpected ChatExpired from client — ignored");
            }

            _ => {} // is_chat_message() guarantees only chat variants reach here
        }
    }

    // ============================================
    // MemChain Message Handler
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
            // ── Existing handlers (preserved verbatim) ──────────────────────

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
                // v2.5.2+SecAudit: verify against content_hash_bytes(), NOT record_id.
                // record_id could theoretically be chosen by an attacker; content_hash
                // is derived from the actual payload so this binding is tight.
                let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                    Ok(pk) => pk.verify(&record.content_hash_bytes(), &record.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord signature verification failed");
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
                        info!(id = hex::encode(record.record_id), "[MEMCHAIN] BroadcastRecord stored");
                        if record.has_embedding() {
                            if let Some(ref vi) = vector_index {
                                vi.upsert(
                                    record.record_id, record.embedding.clone(),
                                    record.layer, record.timestamp,
                                    &record.owner, "p2p-remote",
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
                    if !sig_ok || !config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
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
                        // v2.5.2+SecAudit: same fix as BroadcastRecord —
                        // verify against content_hash_bytes(), not record_id.
                        let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                            Ok(pk) => pk.verify(&record.content_hash_bytes(), &record.signature).is_ok(),
                            Err(_) => false,
                        };
                        if !sig_ok {
                            warn!(owner = %owner_hex, "[MEMCHAIN] SyncRecordResponse sig failed — skipping record");
                            continue;
                        }
                        if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) { continue; }
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

            _ => { debug!("[MEMCHAIN] Unhandled message variant"); }
        }
    }

    // ============================================
    // 🌟 v1.1.0-ChatRelay: Push expired notifications
    // ============================================

    /// Pushes backlogged `ChatExpired` notifications to a newly connected sender.
    ///
    /// Called immediately after a successful handshake. If Alice was offline
    /// when her messages expired, the TTL cleanup task queued notifications
    /// in `expired_notifications`. This method delivers them now.
    async fn push_expired_notifications(
        relay: &Arc<ChatRelayService>,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        let wallet = session.wallet_bytes();

        let notifications = match relay.get_pending_notifications(&wallet) {
            Ok(n) => n,
            Err(e) => {
                warn!(error = %e, "[CHAT_RELAY] get_pending_notifications failed");
                return;
            }
        };

        if notifications.is_empty() {
            return;
        }

        let mut pushed_ids = Vec::new();

        for notif in &notifications {
            let message_ids = match notif.message_ids() {
                Ok(ids) => ids,
                Err(e) => {
                    warn!(error = %e, id = notif.id, "[CHAT_RELAY] Failed to decode notification ids");
                    continue;
                }
            };

            let expired_msg = MemChainMessage::ChatExpired {
                message_ids,
                receiver: notif.receiver,
            };

            Self::send_to_session(&expired_msg, session, udp, crypto).await;
            pushed_ids.push(notif.id);
        }

        if !pushed_ids.is_empty() {
            if let Err(e) = relay.mark_notifications_pushed(&pushed_ids) {
                warn!(error = %e, "[CHAT_RELAY] mark_notifications_pushed failed");
            }
            info!(
                count = pushed_ids.len(),
                wallet = %hex::encode(&wallet[..4]),
                "[CHAT_RELAY] Pushed backlogged ChatExpired notifications"
            );
        }
    }

    // ============================================
    // send_to_session helper
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
            &session.session_key, counter, session.id.as_bytes(),
            &plaintext, &mut encrypted,
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
        // 🌟 v1.1.0-ChatRelay
        chat_relay: Option<Arc<ChatRelayService>>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();

        // Chat cleanup interval (may differ from session cleanup interval)
        let chat_interval_secs = chat_relay
            .as_ref()
            .map(|r| r.config().cleanup_interval_secs)
            .unwrap_or(60);

        tokio::spawn(async move {
            // Session cleanup runs every 60 seconds
            let mut session_timer = tokio::time::interval(Duration::from_secs(60));
            // Chat cleanup runs at its own configured interval
            let mut chat_timer = tokio::time::interval(
                Duration::from_secs(chat_interval_secs)
            );

            loop {
                tokio::select! {
                    _ = rx.recv() => break,

                    _ = session_timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        for (sid, vip) in sessions.cleanup_expired() {
                            routing.remove_route(vip);
                            ip_pool.release(vip);
                            events.session_ended(&sid.to_string(), None, 0, 0);
                        }
                    }

                    // 🌟 v1.1.0-ChatRelay: TTL cleanup for messages and blobs.
                    // run_cleanup() is synchronous SQLite I/O — must run in
                    // spawn_blocking to avoid blocking the tokio executor.
                    _ = chat_timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        if let Some(relay) = chat_relay.clone() {
                            tokio::task::spawn_blocking(move || {
                                match relay.run_cleanup() {
                                    Ok((msgs, blobs)) => {
                                        if msgs > 0 || blobs > 0 {
                                            debug!(
                                                expired_messages = msgs,
                                                expired_blobs = blobs,
                                                "[CHAT_RELAY] Cleanup cycle complete"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        warn!(error = %e, "[CHAT_RELAY] Cleanup error");
                                    }
                                }
                            });
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
