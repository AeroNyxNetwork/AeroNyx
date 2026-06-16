// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Wired TrafficTracker into PacketHandler and HeartbeatReporter.
//   spawn_cleanup_task() now accepts + calls traffic_tracker.remove_wallet().
//   HeartbeatReporter receives sessions, traffic, udp via builder methods
//   so it can collect connected_wallets, drain deltas, and enforce
//   membership rules on heartbeat responses.
//   Adds a shared aggregate encrypted VPN message counter for public stats.
//
// What changed vs previous version:
//   1. Added import: use crate::services::traffic_tracker::TrafficTracker;
//   2. In run(): Arc::new(TrafficTracker::new()) created before PacketHandler
//   3. PacketHandler::new() receives Arc::clone(&traffic_tracker)
//   4. HeartbeatReporter gets .with_sessions/.with_traffic_tracker/.with_udp
//   5. spawn_cleanup_task() signature + call site: added traffic_tracker param
//   6. cleanup loop: calls traffic_tracker.remove_wallet(&wallet)
//   7. encrypted_message_counter shared by PacketHandler and VPN health
//   8. Starts a privacy-safe VPN DNS proxy on gateway_ip:53 so commercial
//      clients can resolve domains through the tunnel.
//
// ⚠️ Important Notes for Next Developer:
//   - traffic_tracker is Arc-shared between packet_handler (writes) and
//     heartbeat reporter (drains). Same instance, different usage patterns.
//   - heartbeat is declared `mut` in run() so builder calls can be chained
//     after init_management_reporter() returns.
//   - remove_wallet() is called AFTER sessions.remove() (inside
//     cleanup_expired). Order matters: session must be gone first.
//   - All other logic (VPN, MemChain, ChatRelay, Voice, SuperNode,
//     SaaS pool, Miner) is unchanged from the previous version.
//   - encrypted_message_counter is aggregate only and never stores payload,
//     destination, DNS, URL, voucher, wallet, or client public IP details.
//   - dns_proxy forwards opaque DNS UDP payloads only; it does not parse,
//     log, store, or report queried domains.
//
// Last Modified:
//   v1.0.2-DNSProxy - VPN gateway DNS proxy wiring
//   v2.5.3+Security    - Server::new() gains config_path
//   v1.0.0-MultiTenant - SaaS startup branch
//   v1.2.0-MultiDevice - ChatRelayService init
//   v1.0.0-Voice+SessionFix - Voice API, session fixes
//   v1.0.0-Membership  - TrafficTracker wiring
//   v1.0.1-VpnMessageStats - encrypted message counter wiring
// ============================================

use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn, trace};

use aeronyx_core::protocol::auth::{
    verify_signed_message,
    DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL, DOMAIN_DEVICE_REGISTER, DOMAIN_WALLET_PRESENCE,
};
use sha2::{Digest, Sha256};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::{
    decode_client_hello, encode_server_hello, encode_data_packet, ProtocolCodec,
};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::{DataPacket, MessageType};
use aeronyx_core::protocol::messages::CLIENT_HELLO_SIZE;
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::UdpTransport;

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use rusqlite::OptionalExtension;

use crate::api::mpi::{build_mpi_router, MpiState, BaselineSnapshot, Mode};
use crate::api::auth::{ensure_jwt_secret, generate_secret};
use crate::api::voice::build_voice_router;
use crate::api::vpn_health::{
    build_vpn_health_router, collect_node_operator_status_value, collect_vpn_health_value,
};
use crate::config::{MemChainConfig, MemChainMode, ServerConfig, VectorQuantizationMode};
use crate::error::{Result, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    CommandHandler, HeartbeatReporter, ManagementClient, SessionReporter,
    reporter::{SessionEventSender, SessionQuality},
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
use crate::services::memchain::{
    SystemDb, VolumeRouter, StoragePool, VectorIndexPool,
    ensure_volumes_config,
};
use crate::services::chat_relay::{ChatRelayService, derive_node_secret};
use crate::services::{
    spawn_dns_proxy, HandshakeService, IpPoolService, NodePolicyRuntime, RoutingService,
    SessionManager,
};
// v1.0.0-Membership
use crate::services::traffic_tracker::TrafficTracker;
use crate::services::session::StatsSnapshot;
use crate::services::deny_list::DenyList;
use crate::voucher_verifier::VoucherVerifier;

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE:       usize  = 17;
#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE:  usize  = 18;
const COMMAND_CHANNEL_BUFFER:      usize  = 100;
const QUANTIZER_CAL_KEY_PREFIX:    &str   = "quantizer_cal";
const POOL_EVICTION_INTERVAL_SECS: u64    = 300;
const MINER_SCHEDULER_TICK_SECS:   u64    = 60;
const KEEPALIVE_PROBE_INTERVAL_SECS: u64    = 60;
const KEEPALIVE_ACK_TIMEOUT_SECS:    u64    = 90;

// ============================================
// Server
// ============================================

fn quality_from_stats(snap: StatsSnapshot) -> SessionQuality {
    let rejects = snap.replays_rejected + snap.too_old_rejected;
    let accepted = snap.packets_rx + snap.packets_tx;
    let packet_loss = if accepted + rejects > 0 {
        Some((rejects as f64 / (accepted + rejects) as f64) * 100.0)
    } else {
        None
    };

    SessionQuality {
        last_rx_at: (snap.last_rx_at > 0).then_some(snap.last_rx_at),
        last_tx_at: (snap.last_tx_at > 0).then_some(snap.last_tx_at),
        rtt_ms: (snap.rtt_us > 0).then_some(snap.rtt_us as f64 / 1000.0),
        packet_loss,
        replay_rejections: Some(snap.replays_rejected),
        too_old_rejections: Some(snap.too_old_rejected),
        packets_rx: Some(snap.packets_rx),
        packets_tx: Some(snap.packets_tx),
        keepalive_probes_sent: Some(snap.keepalive_probes_sent),
        keepalive_acks: Some(snap.keepalive_acks),
        keepalive_missed: Some(snap.keepalive_missed),
        keepalive_pending: Some(snap.keepalive_pending),
    }
}

pub struct Server {
    config:      ServerConfig,
    identity:    IdentityKeyPair,
    config_path: Option<PathBuf>,
    shutdown:    Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    pub fn new(
        config:      ServerConfig,
        identity:    IdentityKeyPair,
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

        if self.config.memchain.is_enabled() {
            self.ensure_api_secret_on_disk().await;
        }

        let (ip_pool, sessions, routing) = self.init_services()?;

        let (storage, vector_index, mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (st, vi, mp, aw) = self.init_memchain().await?;
            (Some(st), Some(vi), Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] Disabled (mode=off)");
            (None, None, None, None)
        };

        let chat_relay: Option<Arc<ChatRelayService>> =
            if self.config.memchain.is_enabled() {
                let node_secret = derive_node_secret(&self.identity.to_bytes());
                match ChatRelayService::new(
                    self.config.memchain.chat_relay.clone(),
                    node_secret,
                ) {
                    Ok(svc) => {
                        info!("[CHAT_RELAY] Service initialized");
                        Some(Arc::new(svc))
                    }
                    Err(e) => {
                        warn!(error = %e, "[CHAT_RELAY] Init failed — chat relay disabled");
                        None
                    }
                }
            } else {
                None
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

        // Commercial VPN readiness requires DNS to be available at the tunnel
        // gateway. This proxy forwards opaque UDP DNS bytes only and never
        // records queried domains, DNS contents, destinations, or client IPs.
        let dns_task = spawn_dns_proxy(self.config.gateway_ip(), self.shutdown_tx.subscribe());
        tasks.push(("dns-proxy", dns_task));

        // v1.0.0-Membership: TrafficTracker must be created before
        // PacketHandler AND before init_management_reporter so both
        // can receive the same Arc.
        let traffic_tracker = Arc::new(TrafficTracker::new());
        let encrypted_message_counter = Arc::new(AtomicU64::new(0));
        // v1.0.0-Membership: DenyList shared between HandshakeService
        // (read: reject denied wallets) and HeartbeatReporter (write: add/remove entries).
        let deny_list = Arc::new(DenyList::new());
        let node_policy = Arc::new(NodePolicyRuntime::default());

        let packet_handler = Arc::new(PacketHandler::new(
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::clone(&traffic_tracker),
            Arc::clone(&encrypted_message_counter),
            Arc::clone(&node_policy),
        ));

        let handshake_service = Arc::new(HandshakeService::new(
            self.identity.clone(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::clone(&deny_list),
            Arc::clone(&node_policy),
        ));

        // [VOUCHER-P1] Observe-only verifier. It never rejects handshakes in
        // this phase; it records valid/invalid/missing voucher rates first.
        let voucher_verifier = Arc::new(VoucherVerifier::new());

        // init_management_reporter needs udp + traffic_tracker,
        // so it is called here after both are available.
        let session_event_sender = self.init_management_reporter(
            &sessions,
            Arc::clone(&ip_pool),
            Arc::clone(&udp),
            Arc::clone(&traffic_tracker),
            Arc::clone(&deny_list),
            Arc::clone(&node_policy),
            Arc::clone(&voucher_verifier),
            Arc::clone(&encrypted_message_counter),
        ).await;

        let udp_task = self.spawn_udp_task(
            Arc::clone(&udp),
            #[cfg(target_os = "linux")]
            Arc::clone(&tun),
            Arc::clone(&handshake_service),
            Arc::clone(&packet_handler),
            Arc::clone(&voucher_verifier),
            Arc::clone(&sessions),
            session_event_sender.clone(),
            mempool.clone(),
            aof_writer.clone(),
            storage.clone(),
            vector_index.clone(),
            self.config.memchain.clone(),
            server_pubkey_hex.clone(),
            chat_relay.clone(),
            Arc::clone(&routing),
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
            Arc::clone(&traffic_tracker),
            Arc::clone(&deny_list),
        );
        tasks.push(("cleanup", cleanup_task));

        let snapshot_task = self.spawn_traffic_snapshot_task(
            Arc::clone(&sessions),
            session_event_sender.clone(),
            self.config.management.session_report_interval_secs,
        );
        tasks.push(("traffic-snapshot", snapshot_task));

        let keepalive_task = self.spawn_keepalive_probe_task(
            Arc::clone(&sessions),
            Arc::clone(&udp),
            Arc::clone(&packet_handler),
            self.config.gateway_ip(),
        );
        tasks.push(("vpn-keepalive", keepalive_task));

        if let Some(ref relay) = chat_relay {
            let routes     = Arc::clone(&relay.wallet_routes);
            let mut rx     = self.shutdown_tx.subscribe();
            tasks.push(("wallet-routes-cleanup", tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(60));
                loop {
                    tokio::select! {
                        _ = rx.recv() => break,
                        _ = interval.tick() => {
                            let evicted = routes.cleanup_stale(Duration::from_secs(300));
                            if evicted > 0 {
                                debug!(evicted, "[CHAT_RELAY] Stale wallet routes evicted");
                            }
                        }
                    }
                }
            })));
            info!("[CHAT_RELAY] Wallet route cleanup task started (ttl=300s, interval=60s)");
        }

        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            let is_saas = self.config.memchain.mode == MemChainMode::Saas;

            let user_weights = Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new(),
            ));

            if !is_saas {
                let owner = self.identity.public_key_bytes();
                if let Some(blob) = st.load_user_weights(&owner).await {
                    if let Some(w) = crate::services::memchain::mvf::WeightVector::from_bytes(&blob) {
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
                    ).ok()
                })
            } else {
                None
            };

            let owner_key  = self.identity.public_key_bytes();
            let api_secret = self.config.memchain.effective_api_secret().map(|s| s.to_string());

            let embed_engine    = self.init_embed_engine();
            let ner_engine      = self.init_ner_engine();
            let reranker_engine = self.init_reranker_engine();

            let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router().await;

            if llm_router.is_some() {
                let timeout_secs =
                    self.config.memchain.supernode.worker.task_timeout_secs as i64;
                let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
                if recovered > 0 {
                    info!(recovered, timeout_secs, "[SUPERNODE] Recovered stale tasks");
                }
            }

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
                ).await?
            } else {
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
                    Some(derive_rawlog_key(&self.identity.to_bytes())),
                    llm_router.clone(),
                );
                Arc::new(mpi)
            };

            if !is_saas {
                let owner_hex      = hex::encode(owner_key);
                let identity_records = st
                    .get_active_records(
                        &owner_key,
                        Some(aeronyx_core::ledger::MemoryLayer::Identity),
                        100,
                    ).await;
                if !identity_records.is_empty() {
                    let mut cache = mpi_state.identity_cache.write();
                    cache.insert(owner_hex, identity_records);
                }
                mpi_state.index_ready.store(true, std::sync::atomic::Ordering::Relaxed);

                if self.config.memchain.mvf_enabled
                    && mpi_state.mvf_baseline.read().is_none()
                {
                    let feedback = st.get_recent_feedback(200).await;
                    if !feedback.is_empty() {
                        let positive = feedback.iter().filter(|(s, _)| *s == 1).count();
                        let rate     = positive as f32 / feedback.len() as f32;
                        let now_ts   = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        let baseline = BaselineSnapshot {
                            positive_rate: rate,
                            sample_size:   feedback.len(),
                            frozen_at:     now_ts,
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
                Arc::clone(&ip_pool),
                Arc::clone(&sessions),
                Arc::clone(&udp),
                Arc::clone(&node_policy),
                Arc::clone(&voucher_verifier),
                Arc::clone(&encrypted_message_counter),
            );
            tasks.push(("memchain-api", api_task));

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
                    poll_interval  = self.config.memchain.supernode.worker.poll_interval_secs,
                    max_concurrent = self.config.memchain.supernode.worker.max_concurrent,
                    "[SUPERNODE] TaskWorker spawned"
                );
            }

            if is_saas {
                if let (Some(ref sp), Some(ref vp)) =
                    (&mpi_state.storage_pool, &mpi_state.vector_pool)
                {
                    let sp_clone   = Arc::clone(sp);
                    let vp_clone   = Arc::clone(vp);
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
                                            vector  = evicted_v,
                                            "[POOL] Evicted idle connections"
                                        );
                                    }
                                }
                            }
                        }
                    })));
                    info!(interval_secs = POOL_EVICTION_INTERVAL_SECS, "[POOL] Eviction timer started");
                }
            }

            if self.config.memchain.miner_interval_secs > 0 {
                if is_saas {
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
                                    _ = interval.tick() => { scheduler.tick().await; }
                                }
                            }
                        })));
                        info!(tick_secs = MINER_SCHEDULER_TICK_SECS, "[MINER] SaaS MinerScheduler started");
                    }
                } else {
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

                    let miner = if let Some(ref ee) = embed_engine { miner.with_embed_engine(Arc::clone(ee)) } else { miner };
                    let miner = if let Some(ref ne) = ner_engine   { miner.with_ner_engine(Arc::clone(ne))   } else { miner };
                    let miner = if let Some(ref lr) = llm_router   { miner.with_llm_router(Arc::clone(lr))   } else { miner };

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
                Err(_)     => warn!("Task '{}' timed out", name),
            }
        }

        if let Err(e) = udp.shutdown().await {
            warn!("UDP shutdown error: {}", e);
        }

        if let (Some(ref st), Some(ref mp), Some(ref aw)) =
            (&storage, &mempool, &aof_writer)
        {
            info!(
                sqlite  = st.count().await,
                mempool = mp.count(),
                aof     = aw.lock().await.write_count(),
                "Shutdown complete (MemChain stats)"
            );
        } else {
            info!("Shutdown complete");
        }

        Ok(())
    }

    // ============================================
    // Auto-generate api_secret
    // ============================================

    async fn ensure_api_secret_on_disk(&self) {
        if self.config.memchain.effective_api_secret().is_some() { return; }
        let Some(ref path) = self.config_path else { return };
        let secret = generate_secret();
        match crate::api::auth::write_secret_to_config_pub(path, "api_secret", &secret) {
            Ok(()) => info!(path = %path.display(), "[SECURITY] Auto-generated api_secret written to config"),
            Err(e) => warn!(error = %e, "[SECURITY] Failed to persist api_secret"),
        }
    }

    // ============================================
    // SaaS MpiState init
    // ============================================

    #[allow(clippy::too_many_arguments)]
    async fn init_saas_mpi_state(
        &self,
        _server_storage:  &Arc<MemoryStorage>,
        owner_key:        [u8; 32],
        api_secret:       Option<String>,
        user_weights:     Arc<parking_lot::RwLock<std::collections::HashMap<String, crate::services::memchain::mvf::WeightVector>>>,
        embed_engine:     Option<Arc<EmbedEngine>>,
        ner_engine:       Option<Arc<NerEngine>>,
        reranker_engine:  Option<Arc<RerankerEngine>>,
        llm_router:       Option<Arc<LlmRouter>>,
    ) -> Result<Arc<MpiState>> {
        let saas_cfg = self.config.memchain.saas.as_ref()
            .ok_or_else(|| ServerError::startup_failed(
                "mode=saas requires [memchain.saas] config section"
            ))?;

        let data_root = &saas_cfg.data_root;

        tokio::fs::create_dir_all(data_root).await.map_err(|e| {
            ServerError::startup_failed(format!("SaaS data_root '{}': {}", data_root.display(), e))
        })?;

        let system_db = SystemDb::open(&data_root.join("system.db"))
            .await
            .map_err(|e| ServerError::startup_failed(format!("SystemDb: {}", e)))?;

        let volumes_config_path = ensure_volumes_config(data_root)
            .map_err(|e| ServerError::startup_failed(format!("volumes.toml: {}", e)))?;

        let volume_router = VolumeRouter::new(&volumes_config_path, Arc::clone(&system_db))
            .await
            .map_err(|e| ServerError::startup_failed(format!("VolumeRouter: {}", e)))?;

        let storage_pool = StoragePool::new(
            Arc::clone(&volume_router),
            Arc::clone(&system_db),
            saas_cfg.pool_max_connections,
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let saturation_threshold = if self.config.memchain.vector_early_termination { 0.001_f32 } else { 0.0_f32 };

        let vector_pool = VectorIndexPool::new(
            Arc::clone(&volume_router),
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
            quantization_enabled,
            saturation_threshold,
        );

        let jwt_secret = ensure_jwt_secret(
            self.config.memchain.jwt_secret.as_deref(),
            self.config_path.as_deref(),
        ).map_err(|e| ServerError::startup_failed(format!("jwt_secret: {}", e)))?;

        info!(
            data_root     = %data_root.display(),
            pool_max      = saas_cfg.pool_max_connections,
            idle_timeout  = saas_cfg.pool_idle_timeout_secs,
            "[SAAS] Infrastructure initialized"
        );

        let mpi_state = Arc::new(MpiState {
            mode: Mode::Saas,
            storage:      None,
            vector_index: None,
            identity:     self.identity.clone(),
            identity_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            index_ready:  std::sync::atomic::AtomicBool::new(true),
            user_weights,
            mvf_alpha:    self.config.memchain.mvf_alpha,
            mvf_enabled:  self.config.memchain.mvf_enabled,
            session_embeddings: parking_lot::RwLock::new(std::collections::HashMap::new()),
            mvf_baseline: parking_lot::RwLock::new(None),
            owner_key,
            api_secret,
            embed_engine,
            allow_remote_storage: false,
            max_remote_owners:    0,
            ner_engine,
            graph_enabled:           self.config.memchain.graph_enabled,
            entropy_filter_enabled:  self.config.memchain.entropy_filter_enabled,
            reranker_engine,
            rawlog_key:  Some(derive_rawlog_key(&self.identity.to_bytes())),
            llm_router,
            storage_pool:    Some(storage_pool),
            vector_pool:     Some(vector_pool),
            volume_router:   Some(volume_router),
            system_db:       Some(system_db),
            jwt_secret:      Some(jwt_secret),
            token_ttl_secs:          self.config.memchain.token_ttl_secs,
            pool_max_connections:    saas_cfg.pool_max_connections,
            pool_idle_timeout_secs:  saas_cfg.pool_idle_timeout_secs,
        });

        Ok(mpi_state)
    }

    // ============================================
    // Engine initialization
    // ============================================

    fn init_embed_engine(&self) -> Option<Arc<EmbedEngine>> {
        let model_path = &self.config.memchain.embed_model_path;
        match EmbedEngine::load(model_path, self.config.memchain.embed_max_tokens, self.config.memchain.embed_output_dim) {
            Ok(engine) => {
                info!(model = %model_path, model_type = %engine.model_type(), dim = engine.dim(), "[EMBED] Local embedding engine loaded");
                Some(Arc::new(engine))
            }
            Err(e) => { warn!(model = %model_path, error = %e, "[EMBED] Unavailable"); None }
        }
    }

    fn init_ner_engine(&self) -> Option<Arc<NerEngine>> {
        if !self.config.memchain.ner_enabled { debug!("[NER] Disabled"); return None; }
        let model_path = &self.config.memchain.ner_model_path;
        let threshold  = self.config.memchain.ner_confidence_threshold;
        match NerEngine::load(model_path, threshold, 0) {
            Ok(engine) => { info!(model = %model_path, threshold, "[NER] Local NER engine loaded"); Some(Arc::new(engine)) }
            Err(e)     => { warn!(model = %model_path, error = %e, "[NER] Unavailable"); None }
        }
    }

    fn init_reranker_engine(&self) -> Option<Arc<RerankerEngine>> {
        if !self.config.memchain.reranker_enabled { debug!("[RERANKER] Disabled"); return None; }
        let model_path = &self.config.memchain.reranker_model_path;
        let max_seq    = self.config.memchain.reranker_max_seq_length;
        match RerankerEngine::load(model_path, max_seq) {
            Ok(engine) => { info!(model = %model_path, blend_weight = %RerankerEngine::blend_weight(), "[RERANKER] Cross-encoder loaded"); Some(Arc::new(engine)) }
            Err(e)     => { warn!(model = %model_path, error = %e, "[RERANKER] Unavailable"); None }
        }
    }

    // ============================================
    // LlmRouter initialization
    // ============================================

    async fn init_llm_router(&self) -> Option<Arc<LlmRouter>> {
        use crate::config_supernode::ProviderType;
        use crate::services::memchain::{LlmProvider, OpenAiCompatProvider, AnthropicProvider};

        if !self.config.memchain.is_supernode_enabled() {
            debug!("[SUPERNODE] Disabled");
            return None;
        }

        let supernode = &self.config.memchain.supernode;
        if supernode.providers.is_empty() {
            warn!("[SUPERNODE] enabled=true but no providers — disabled");
            return None;
        }

        let mut providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = Vec::new();

        for provider_cfg in &supernode.providers {
            let api_key: Option<String> = provider_cfg.api_key.as_ref().map(|k| {
                if k.starts_with('$') {
                    std::env::var(&k[1..]).unwrap_or_else(|_| {
                        warn!(key = %k, provider = %provider_cfg.name, "[SUPERNODE] ENV var not set for api_key");
                        String::new()
                    })
                } else { k.clone() }
            });

            let api_base = if provider_cfg.api_base.is_empty() && provider_cfg.provider_type == ProviderType::Anthropic {
                "https://api.anthropic.com".to_string()
            } else {
                provider_cfg.api_base.clone()
            };

            let provider: Arc<dyn LlmProvider> = match provider_cfg.provider_type {
                ProviderType::OpenaiCompatible => {
                    match OpenAiCompatProvider::new(
                        provider_cfg.name.clone(), api_base.clone(),
                        api_key.unwrap_or_default(), provider_cfg.model.clone(),
                        provider_cfg.max_tokens, provider_cfg.temperature,
                    ) {
                        Ok(p)  => Arc::new(p),
                        Err(e) => { warn!(provider = %provider_cfg.name, error = %e, "[SUPERNODE] OpenAiCompatProvider failed"); continue; }
                    }
                }
                ProviderType::Anthropic => {
                    let key = match api_key {
                        Some(k) if !k.is_empty() => k,
                        _ => { warn!(provider = %provider_cfg.name, "[SUPERNODE] Anthropic requires api_key"); continue; }
                    };
                    match AnthropicProvider::new(
                        provider_cfg.name.clone(), key, provider_cfg.model.clone(),
                        provider_cfg.max_tokens, provider_cfg.temperature,
                    ) {
                        Ok(p)  => Arc::new(p),
                        Err(e) => { warn!(provider = %provider_cfg.name, error = %e, "[SUPERNODE] AnthropicProvider failed"); continue; }
                    }
                }
            };

            info!(name = %provider_cfg.name, type_ = ?provider_cfg.provider_type, model = %provider_cfg.model, api_base = %api_base, "[SUPERNODE] Provider registered");
            providers.push((provider_cfg.name.clone(), api_base, provider_cfg.model.clone(), provider));
        }

        if providers.is_empty() {
            warn!("[SUPERNODE] All providers failed — disabled");
            return None;
        }

        let router = LlmRouter::new(providers, supernode.routing.clone());
        info!(providers = supernode.providers.len(), fallback = ?supernode.routing.fallback, "[SUPERNODE] LlmRouter initialized");
        Some(Arc::new(router))
    }

    // ============================================
    // MemChain initialization
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
        let rebuild_count      = records_with_model.len();
        for (r, model) in records_with_model {
            if r.has_embedding() {
                vector_index.upsert(r.record_id, r.embedding.clone(), r.layer, r.timestamp, &r.owner, &model);
            }
        }

        info!(db = %db_path, records = storage.count().await, vectors = rebuild_count, "[MEMCHAIN] SQLite + VectorIndex initialized");

        if quantization_enabled && rebuild_count > 0 {
            let owner_hex  = hex::encode(owner);
            let model_name = std::path::Path::new(&self.config.memchain.embed_model_path)
                .file_name().and_then(|f| f.to_str()).unwrap_or("minilm-l6-v2");
            let cal_key    = format!("{}:{}:{}", QUANTIZER_CAL_KEY_PREFIX, owner_hex, model_name);

            let restored = {
                let conn = storage.conn_lock().await;
                let cal_data: Option<Vec<u8>> = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = ?1",
                        rusqlite::params![cal_key],
                        |row| row.get::<_, Vec<u8>>(0),
                    ).optional().unwrap_or(None);
                drop(conn);
                if let Some(data) = cal_data {
                    let ok = vector_index.restore_quantizer(&owner, model_name, &data);
                    if ok { info!(model = model_name, "[VECTOR] Quantizer restored"); }
                    ok
                } else { false }
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
                    info!(model = model_name, "[VECTOR] Quantizer calibrated and persisted");
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

        let (existing_facts, last_block) = AofWriter::replay(aof_path).await
            .map_err(|e| ServerError::startup_failed(format!("AOF replay: {}", e)))?;

        let mempool = Arc::new(MemPool::new());
        let mut loaded = 0u64;
        for fact in existing_facts {
            if mempool.add_fact(fact) { loaded += 1; }
        }

        let aof_writer = AofWriter::open(aof_path).await
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
        mpi_state:   Arc<MpiState>,
        _mempool:    Arc<MemPool>,
        _aof_writer: Arc<TokioMutex<AofWriter>>,
        ip_pool:     Arc<IpPoolService>,
        sessions:    Arc<SessionManager>,
        _udp:        Arc<UdpTransport>,
        node_policy: Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
    ) -> JoinHandle<()> {
        let mut shutdown_rx     = self.shutdown_tx.subscribe();
        let mut shutdown_rx_vpn = self.shutdown_tx.subscribe();
        let vpn_listen_addr: std::net::SocketAddr = format!(
            "100.64.0.1:{}", listen_addr.port()
        ).parse().unwrap_or_else(|_| "100.64.0.1:8421".parse().unwrap());
        let vpn_health_config = self.config.clone();

        tokio::spawn(async move {
            let app = build_mpi_router(mpi_state)
                .merge(build_voice_router(Arc::clone(&sessions)))
                .merge(build_vpn_health_router(
                    vpn_health_config,
                    Arc::clone(&ip_pool),
                    sessions,
                    node_policy,
                    voucher_verifier,
                    encrypted_message_counter,
                ));

            let listener = match tokio::net::TcpListener::bind(listen_addr).await {
                Ok(l)  => { info!("[API] MemChain API on http://{}", listen_addr); l }
                Err(e) => { error!("[API] Bind failed {}: {}", listen_addr, e); return; }
            };

            match tokio::net::TcpListener::bind(vpn_listen_addr).await {
                Ok(vpn_listener) => {
                    info!("[API] Voice API also available on http://{} (VPN clients only)", vpn_listen_addr);
                    let app_clone = app.clone();
                    tokio::spawn(async move {
                        let server = axum::serve(vpn_listener, app_clone)
                            .with_graceful_shutdown(async move { let _ = shutdown_rx_vpn.recv().await; });
                        if let Err(e) = server.await { error!("[API] VPN listener error: {}", e); }
                        info!("[API] VPN listener stopped");
                    });
                }
                Err(e) => {
                    warn!("[API] VPN listener on {} not ready yet ({}), will retry every 10s", vpn_listen_addr, e);
                    let app_clone = app.clone();
                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
                        interval.tick().await;
                        loop {
                            tokio::select! {
                                _ = shutdown_rx_vpn.recv() => { debug!("[API] VPN listener retry task shutting down"); break; }
                                _ = interval.tick() => {
                                    match tokio::net::TcpListener::bind(vpn_listen_addr).await {
                                        Ok(vpn_listener) => {
                                            info!("[API] VPN listener bound on {} (TUN is now up)", vpn_listen_addr);
                                            let server = axum::serve(vpn_listener, app_clone)
                                                .with_graceful_shutdown(async { std::future::pending::<()>().await });
                                            if let Err(e) = server.await { error!("[API] VPN listener error: {}", e); }
                                            break;
                                        }
                                        Err(e) => { debug!("[API] VPN listener retry failed ({}): {}", vpn_listen_addr, e); }
                                    }
                                }
                            }
                        }
                    });
                }
            }

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
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
        sessions:        &Arc<SessionManager>,
        ip_pool:         Arc<IpPoolService>,
        udp:             Arc<UdpTransport>,
        traffic_tracker: Arc<TrafficTracker>,
        deny_list:       Arc<DenyList>,
        node_policy:     Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
    ) -> SessionEventSender {
        info!("Initializing management reporting...");

        let mgmt_client = Arc::new(ManagementClient::new(
            self.config.management.clone(),
            self.identity.clone(),
        ));
        info!("Node ID: {}", mgmt_client.node_id());

        let public_ip = self.resolve_public_ip().await;

        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let session_event_sender = SessionEventSender::new(event_tx);

        let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);
        let cmd_handler      = CommandHandler::new(cmd_rx, Arc::clone(&mgmt_client))
            .with_session_control(Arc::clone(sessions), session_event_sender.clone())
            .with_deny_list(Arc::clone(&deny_list))
            .with_node_policy(Arc::clone(&node_policy));
        let cmd_shutdown     = self.shutdown_tx.subscribe();
        tokio::spawn(async move { cmd_handler.run(cmd_shutdown).await; });

        let memchain_status_fn: Option<crate::management::reporter::MemChainStatusFn> =
            if self.config.memchain.is_enabled() {
                let allow_remote = self.config.memchain.allow_remote_storage;
                let max_owners   = self.config.memchain.max_remote_owners;
                Some(Box::new(move || {
                    Some(crate::management::client::MemChainHeartbeatStatus {
                        enabled:               true,
                        allow_remote_storage:  allow_remote,
                        max_remote_owners:     max_owners,
                        current_remote_owners: 0,
                    })
                }))
            } else {
                None
            };

        // Note: .with_sessions / .with_traffic_tracker / .with_udp are
        // injected here — all three are available at this call site.
        let mut heartbeat =
            HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
                .with_command_sender(cmd_tx)
                .with_sessions(Arc::clone(sessions))
                .with_traffic_tracker(Arc::clone(&traffic_tracker))
                .with_udp(Arc::clone(&udp))
                .with_deny_list(Arc::clone(&deny_list))
                .with_node_policy(Arc::clone(&node_policy));

        if let Some(f) = memchain_status_fn {
            heartbeat = heartbeat.with_memchain_status(f);
        }

        let vpn_health_config = self.config.clone();
        let vpn_health_ip_pool = Arc::clone(&ip_pool);
        let vpn_health_sessions = Arc::clone(sessions);
        let vpn_health_policy = Arc::clone(&node_policy);
        let vpn_health_verifier = Arc::clone(&voucher_verifier);
        let vpn_health_message_counter = Arc::clone(&encrypted_message_counter);
        heartbeat = heartbeat.with_vpn_health_status(Box::new(move || {
            let config = vpn_health_config.clone();
            let ip_pool = Arc::clone(&vpn_health_ip_pool);
            let sessions = Arc::clone(&vpn_health_sessions);
            let node_policy = Arc::clone(&vpn_health_policy);
            let verifier = Arc::clone(&vpn_health_verifier);
            let message_counter = Arc::clone(&vpn_health_message_counter);
            Box::pin(async move {
                Some(collect_vpn_health_value(
                    config,
                    ip_pool,
                    sessions,
                    node_policy,
                    verifier,
                    message_counter,
                ).await)
            })
        }));

        let operator_status_config = self.config.clone();
        let operator_status_ip_pool = Arc::clone(&ip_pool);
        let operator_status_sessions = Arc::clone(sessions);
        let operator_status_policy = Arc::clone(&node_policy);
        let operator_status_verifier = Arc::clone(&voucher_verifier);
        let operator_status_message_counter = Arc::clone(&encrypted_message_counter);
        heartbeat = heartbeat.with_operator_status(Box::new(move || {
            let config = operator_status_config.clone();
            let ip_pool = Arc::clone(&operator_status_ip_pool);
            let sessions = Arc::clone(&operator_status_sessions);
            let node_policy = Arc::clone(&operator_status_policy);
            let verifier = Arc::clone(&operator_status_verifier);
            let message_counter = Arc::clone(&operator_status_message_counter);
            Box::pin(async move {
                Some(collect_node_operator_status_value(
                    config,
                    ip_pool,
                    sessions,
                    node_policy,
                    verifier,
                    message_counter,
                ).await)
            })
        }));

        let sess        = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            heartbeat
                .run(move || sess.count() as u32, hb_shutdown)
                .await;
        });

        let sr_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { session_reporter.run(sr_shutdown).await; });

        info!("[MANAGEMENT] Reporting started");
        session_event_sender
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
                            let ip_str = body.trim();
                            if ip_str.len() <= 45 {
                                if let Ok(addr) = ip_str.parse::<std::net::IpAddr>() {
                                    let is_private = match addr {
                                        std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_unspecified(),
                                        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
                                    };
                                    if !is_private {
                                        info!(ip = %addr, source = %url, "[NET] Public IP detected");
                                        return addr.to_string();
                                    }
                                    warn!(ip = %addr, source = %url, "[NET] Ignoring private/loopback IP");
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

    fn init_services(&self) -> Result<(Arc<IpPoolService>, Arc<SessionManager>, Arc<RoutingService>)> {
        let (network, prefix) = self.config.parse_ip_range()?;
        let ip_pool  = Arc::new(IpPoolService::new(network, prefix, self.config.gateway_ip())?);
        let sessions = Arc::new(SessionManager::new(
            self.config.max_sessions(),
            Duration::from_secs(self.config.session_timeout_secs()),
        ));
        let routing  = Arc::new(RoutingService::new());
        info!(capacity = ip_pool.capacity(), max_sessions = self.config.max_sessions(), "Services initialized");
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
        udp:              Arc<UdpTransport>,
        #[cfg(target_os = "linux")] tun: Arc<LinuxTun>,
        handshake:        Arc<HandshakeService>,
        packet_handler:   Arc<PacketHandler>,
        voucher_verifier: Arc<VoucherVerifier>,
        sessions:         Arc<SessionManager>,
        session_events:   SessionEventSender,
        mempool:          Option<Arc<MemPool>>,
        aof_writer:       Option<Arc<TokioMutex<AofWriter>>>,
        storage:          Option<Arc<MemoryStorage>>,
        vector_index:     Option<Arc<VectorIndex>>,
        memchain_config:  MemChainConfig,
        server_pubkey_hex: String,
        chat_relay:       Option<Arc<ChatRelayService>>,
        routing:          Arc<RoutingService>,
    ) -> JoinHandle<()> {
        let shutdown    = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let udp_reply   = Arc::clone(&udp);

        tokio::spawn(async move {
            let mut buf    = vec![0u8; 65535];
            let crypto     = DefaultTransportCrypto::new();

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
                                        let extension = if data.len() > CLIENT_HELLO_SIZE {
                                            data[CLIENT_HELLO_SIZE..].to_vec()
                                        } else {
                                            Vec::new()
                                        };
                                        let verifier = Arc::clone(&voucher_verifier);
                                        if !verifier.accept_client_hello_extension(extension).await {
                                            warn!(
                                                client = %source.addr,
                                                "[VOUCHER] rejected ClientHello with invalid voucher"
                                            );
                                            continue;
                                        }

                                        if let Ok(hello) = decode_client_hello(data) {
                                            match handshake.process(&hello, source.addr) {
                                                Ok(result) => {
                                                    let sid        = BASE64.encode(&result.response.session_id);
                                                    let wallet_hex = hex::encode(result.session.client_public_key.to_bytes());
                                                    session_events.session_created(
                                                        &sid,
                                                        Some(wallet_hex),
                                                        Some(result.session.virtual_ip.to_string()),
                                                    );
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
                                            Ok((session, DecryptedPayload::KeepaliveAck { rtt_ms })) => {
                                                trace!(
                                                    session_id = %session.id,
                                                    rtt_ms,
                                                    "[KEEPALIVE] ACK consumed"
                                                );
                                            }
                                            Err(ref e) if e.is_session_not_found() => {
                                                let reset = [0xFFu8];
                                                let _ = udp_reply.send(&reset, &source.addr).await;
                                                debug!(src = %source.addr, "[SESSION] Sent RESET to stale client");
                                            }
                                            Err(_) => {}
                                            Ok((session, DecryptedPayload::VoiceSignal { discriminant, target_wallet, payload })) => {
                                                let signal_name = match discriminant { 31 => "Offer", 32 => "Answer", 33 => "End", _ => "Unknown" };
                                                match target_wallet {
                                                    None => { warn!(src = %session.virtual_ip, disc = discriminant, "[VOICE_SIGNAL] {} — missing 'to' field, dropped", signal_name); }
                                                    Some(ref wallet_hex) => {
                                                        let target_bytes = hex::decode(wallet_hex).ok().and_then(|b| {
                                                            if b.len() == 32 { let mut arr = [0u8; 32]; arr.copy_from_slice(&b); Some(arr) } else { None }
                                                        });
                                                        match target_bytes {
                                                            None => { warn!(wallet = %wallet_hex, "[VOICE_SIGNAL] {} — invalid pubkey hex, dropped", signal_name); }
                                                            Some(pk) => {
                                                                let target = sessions.get_by_wallet(&pk).or_else(|| {
                                                                    sessions.all_sessions().into_iter().find(|s| s.client_public_key.to_bytes() == pk)
                                                                });
                                                                match target {
                                                                    None => { debug!(wallet = %&wallet_hex[..8], "[VOICE_SIGNAL] {} — target offline, dropped", signal_name); }
                                                                    Some(target_session) => {
                                                                        let counter = target_session.next_tx_counter();
                                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                                        match crypto.encrypt(&target_session.session_key, counter, target_session.id.as_bytes(), &payload, &mut encrypted) {
                                                                            Ok(len) => {
                                                                                encrypted.truncate(len);
                                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target_session.id.as_bytes(), counter, encrypted);
                                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                                let _ = udp_reply.send(&bytes, &target_session.client_endpoint).await;
                                                                                debug!(src = %session.virtual_ip, dst = %target_session.virtual_ip, signal = signal_name, "[VOICE_SIGNAL] Forwarded");
                                                                            }
                                                                            Err(e) => { warn!(error = %e, "[VOICE_SIGNAL] Re-encrypt failed"); }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Ok((session, DecryptedPayload::Voice { dst_ip, payload })) => {
                                                if let Some(target_sid) = routing.lookup(dst_ip) {
                                                    if let Some(target) = sessions.get(&target_sid) {
                                                        let counter = target.next_tx_counter();
                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                        match crypto.encrypt(&target.session_key, counter, target.id.as_bytes(), &payload, &mut encrypted) {
                                                            Ok(len) => {
                                                                encrypted.truncate(len);
                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target.id.as_bytes(), counter, encrypted);
                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                let _ = udp_reply.send(&bytes, &target.client_endpoint).await;
                                                                trace!(src = %session.virtual_ip, dst = %dst_ip, "[VOICE] Relayed voice packet");
                                                            }
                                                            Err(e) => { warn!(dst_ip = %dst_ip, "[VOICE] Re-encrypt failed: {}", e); }
                                                        }
                                                    } else {
                                                        debug!(dst_ip = %dst_ip, "[VOICE] Target session not found (disconnected?)");
                                                    }
                                                } else {
                                                    debug!(dst_ip = %dst_ip, "[VOICE] No route to dst_ip (peer offline)");
                                                }
                                            }
                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw, &storage, &vector_index,
                                                        &memchain_config, &server_pubkey_hex,
                                                        &session, &udp_reply, &crypto,
                                                        &sessions, &chat_relay,
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
        msg:              MemChainMessage,
        mempool:          &Arc<MemPool>,
        aof_writer:       &Arc<TokioMutex<AofWriter>>,
        storage:          &Option<Arc<MemoryStorage>>,
        vector_index:     &Option<Arc<VectorIndex>>,
        config:           &MemChainConfig,
        server_pubkey_hex: &str,
        session:          &Arc<crate::services::Session>,
        udp:              &Arc<UdpTransport>,
        crypto:           &DefaultTransportCrypto,
        sessions:         &Arc<SessionManager>,
        chat_relay:       &Option<Arc<ChatRelayService>>,
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
                if !sig_ok { warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord sig failed"); return; }
                if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                    warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord untrusted"); return;
                }
                if !record.verify_id() {
                    warn!(owner = %owner_hex, id = hex::encode(record.record_id), "[MEMCHAIN] record_id hash mismatch"); return;
                }
                if let Some(ref st) = storage {
                    if st.insert(&record, "p2p-remote").await {
                        info!(id = hex::encode(record.record_id), "[MEMCHAIN] BroadcastRecord stored");
                        if record.has_embedding() {
                            if let Some(ref vi) = vector_index {
                                vi.upsert(record.record_id, record.embedding.clone(), record.layer, record.timestamp, &record.owner, "p2p-remote");
                            }
                        }
                    }
                }
            }
            MemChainMessage::SyncRequest { last_known_hash } => {
                let facts = mempool.get_facts_after(last_known_hash);
                let resp  = MemChainMessage::SyncResponse { facts };
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
                    let resp    = MemChainMessage::SyncRecordResponse { records };
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
                        if !sig_ok { warn!(owner = %owner_hex, "[MEMCHAIN] SyncRecordResponse sig failed"); continue; }
                        if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) { continue; }
                        if !record.verify_id() {
                            warn!(owner = %owner_hex, id = hex::encode(record.record_id), "[MEMCHAIN] SyncRecordResponse hash mismatch"); continue;
                        }
                        let _ = st.insert(&record, "p2p-sync").await;
                    }
                }
            }
            MemChainMessage::BlockAnnounce(header) => {
                info!(height = header.height, hash = hex::encode(header.hash()), "[MEMCHAIN] BlockAnnounce received");
            }
            MemChainMessage::ChatRelay(envelope) => {
                if envelope.verify_signature().is_err() {
                    warn!(receiver = %hex::encode(&envelope.receiver[..4]), "[CHAT_RELAY] Envelope sig failed — dropped"); return;
                }
                let Some(ref relay) = chat_relay else {
                    warn!(receiver = %hex::encode(&envelope.receiver[..4]), "[CHAT_RELAY] Relay unavailable — dropped"); return;
                };
                if relay.is_online_duplicate(&envelope.message_id) {
                    debug!(id = %hex::encode(envelope.message_id), "[CHAT_RELAY] Online duplicate — dropped"); return;
                }
                relay.wallet_routes.announce(&envelope.sender, session.id.clone(), session.client_endpoint);
                let receiver      = envelope.receiver;
                let target_routes = relay.wallet_routes.lookup(&receiver);

                if !target_routes.is_empty() {
                    let mut all_failed  = true;
                    let device_count    = target_routes.len();
                    for (target_sid, _endpoint) in &target_routes {
                        if let Some(target_session) = sessions.get(target_sid) {
                            Self::send_to_session(&MemChainMessage::ChatRelay(envelope.clone()), &target_session, udp, crypto).await;
                            all_failed = false;
                        } else {
                            relay.wallet_routes.remove_session(target_sid);
                            debug!(session = %target_sid, "[CHAT_RELAY] Pruned stale route during delivery");
                        }
                    }
                    if all_failed {
                        if let Err(e) = relay.store_pending(&envelope) {
                            warn!(error = %e, receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] Fallback store_pending failed");
                        } else {
                            debug!(receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] All routes stale — stored for offline delivery");
                        }
                    } else {
                        debug!(receiver = %hex::encode(&receiver[..4]), devices = device_count, "[CHAT_RELAY] Online delivery to {} device(s)", device_count);
                    }
                } else {
                    if let Err(e) = relay.store_pending(&envelope) {
                        warn!(error = %e, receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] store_pending failed");
                    } else {
                        debug!(receiver = %hex::encode(&receiver[..4]), id = %hex::encode(envelope.message_id), "[CHAT_RELAY] Stored for offline delivery");
                    }
                }
            }
            MemChainMessage::ChatPull { wallet, after_timestamp, cursor, limit, request_timestamp, signature } => {
                let Some(ref relay) = chat_relay else { return; };
                let at_bytes    = after_timestamp.to_le_bytes();
                let limit_bytes = limit.to_le_bytes();
                let rts_bytes   = request_timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_CHAT_PULL,
                    &[wallet.as_ref(), at_bytes.as_ref(), cursor.as_ref(), limit_bytes.as_ref(), rts_bytes.as_ref()],
                    &wallet, &signature, request_timestamp,
                );
                if verify_result.is_err() { return; }
                relay.wallet_routes.announce(&wallet, session.id.clone(), session.client_endpoint);
                match relay.pull_pending(&wallet, after_timestamp, &cursor, limit) {
                    Ok((messages, has_more)) => {
                        let envelopes: Vec<_> = messages.into_iter().map(|m| m.envelope).collect();
                        let resp = MemChainMessage::ChatPullResponse { envelopes, has_more };
                        Self::send_to_session(&resp, session, udp, crypto).await;
                    }
                    Err(e) => { warn!(error = %e, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] pull_pending failed"); }
                }
            }
            MemChainMessage::ChatAck { message_ids, wallet, ack_timestamp, signature } => {
                let Some(ref relay) = chat_relay else { return; };
                if message_ids.is_empty() { return; }
                let mut id_hasher = Sha256::new();
                for mid in &message_ids { id_hasher.update(mid.as_ref()); }
                let ids_hash: [u8; 32] = id_hasher.finalize().into();
                let ack_ts_bytes = ack_timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_CHAT_ACK,
                    &[wallet.as_ref(), ack_ts_bytes.as_ref(), ids_hash.as_ref()],
                    &wallet, &signature, ack_timestamp,
                );
                if let Err(e) = verify_result {
                    warn!(wallet = %hex::encode(&wallet[..4]), error = %e, "[CHAT_RELAY] ChatAck sig failed"); return;
                }
                match relay.ack_messages(&message_ids, &wallet) {
                    Ok(deleted) => { debug!(deleted, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] ChatAck processed"); }
                    Err(e)      => { warn!(error = %e, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] ack_messages failed"); }
                }
            }
            MemChainMessage::DeviceRegister { device_id, device_name, wallet_pubkey, timestamp, signature } => {
                let ts_bytes = timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_DEVICE_REGISTER,
                    &[session.id.as_bytes().as_ref(), device_id.as_ref(), wallet_pubkey.as_ref(), ts_bytes.as_ref()],
                    &wallet_pubkey, &signature, timestamp,
                );
                if let Err(e) = verify_result {
                    warn!(session = %session.id, wallet = %hex::encode(&wallet_pubkey[..4]), error = %e, "[CHAT_RELAY] DeviceRegister sig failed"); return;
                }
                let name_display = if device_name.len() > 64 {
                    let mut end = 64;
                    while !device_name.is_char_boundary(end) { end -= 1; }
                    &device_name[..end]
                } else { &device_name };
                let Some(ref relay) = chat_relay else { return; };
                relay.wallet_routes.announce(&wallet_pubkey, session.id.clone(), session.client_endpoint);
                sessions.register_device(&wallet_pubkey, device_id, session.id.clone());
                info!(session_id = %session.id, wallet = %hex::encode(&wallet_pubkey[..4]), device_id = %hex::encode(device_id), device_name = %name_display, "[CHAT_RELAY] Device registered");
                match relay.pull_pending(&wallet_pubkey, 0, &[0u8; 16], 100) {
                    Ok((messages, _has_more)) if !messages.is_empty() => {
                        let count = messages.len();
                        for pm in messages {
                            Self::send_to_session(&MemChainMessage::ChatRelay(pm.envelope), session, udp, crypto).await;
                        }
                        info!(count, wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] Delivered pending messages on register");
                    }
                    Ok(_)  => {}
                    Err(e) => { warn!(error = %e, wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] pull_pending on register failed"); }
                }
            }
            MemChainMessage::WalletPresence { wallet_pubkey, timestamp, signature } => {
                let Some(ref relay) = chat_relay else { return; };
                let ts_bytes = timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_WALLET_PRESENCE,
                    &[session.id.as_bytes().as_ref(), wallet_pubkey.as_ref(), ts_bytes.as_ref()],
                    &wallet_pubkey, &signature, timestamp,
                );
                if let Err(e) = verify_result {
                    debug!(wallet = %hex::encode(&wallet_pubkey[..4]), error = %e, "[CHAT_RELAY] WalletPresence sig failed"); return;
                }
                relay.wallet_routes.announce(&wallet_pubkey, session.id.clone(), session.client_endpoint);
                debug!(wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] WalletPresence: route refreshed");
            }
            _ => { debug!("[MEMCHAIN] Unhandled message variant"); }
        }
    }

    async fn send_to_session(
        msg:     &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp:     &Arc<UdpTransport>,
        crypto:  &DefaultTransportCrypto,
    ) {
        let plaintext = match encode_memchain(msg) {
            Ok(p)  => p,
            Err(e) => { error!("[MEMCHAIN_TX] Encode: {}", e); return; }
        };
        let counter       = session.next_tx_counter();
        let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
        let len = match crypto.encrypt(&session.session_key, counter, session.id.as_bytes(), &plaintext, &mut encrypted) {
            Ok(l)  => l,
            Err(e) => { error!("[MEMCHAIN_TX] Encrypt: {}", e); return; }
        };
        encrypted.truncate(len);
        let pkt   = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let bytes = encode_data_packet(&pkt).to_vec();
        let _ = udp.send(&bytes, &session.client_endpoint).await;
    }

    // ============================================
    // TUN Task
    // ============================================

    #[cfg(target_os = "linux")]
    fn spawn_tun_task(
        &self,
        tun:     Arc<LinuxTun>,
        udp:     Arc<UdpTransport>,
        handler: Arc<PacketHandler>,
    ) -> JoinHandle<()> {
        let shutdown    = Arc::clone(&self.shutdown);
        let mut rx      = self.shutdown_tx.subscribe();
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
                            Err(e) => { if !shutdown.load(Ordering::SeqCst) { error!("TUN: {}", e); } }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // Traffic Snapshot Task
    // ============================================

    fn spawn_traffic_snapshot_task(
        &self,
        sessions: Arc<SessionManager>,
        events:   SessionEventSender,
        interval_secs: u64,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx   = self.shutdown_tx.subscribe();
        let interval_secs = interval_secs.clamp(10, 300);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        let all      = sessions.all_sessions();
                        let mut reported = 0usize;
                        for session in all {
                            if !session.is_established() { continue; }
                            let snap = session.stats_snapshot();
                            let wallet_hex = session.wallet_hex.clone();
                            let sid        = BASE64.encode(session.id.as_bytes());
                            events.session_traffic_snapshot(
                                &sid,
                                Some(wallet_hex),
                                Some(session.virtual_ip.to_string()),
                                snap.bytes_rx,
                                snap.bytes_tx,
                                quality_from_stats(snap),
                            );
                            reported += 1;
                        }
                        if reported > 0 {
                            debug!(
                                sessions = reported,
                                interval_secs,
                                "[TRAFFIC_SNAPSHOT] Sent quality snapshots for {} active session(s)",
                                reported
                            );
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // VPN Keepalive / RTT Probe Task
    // ============================================

    fn spawn_keepalive_probe_task(
        &self,
        sessions:       Arc<SessionManager>,
        udp:            Arc<UdpTransport>,
        packet_handler: Arc<PacketHandler>,
        gateway_ip:     Ipv4Addr,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx   = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let mut timer = tokio::time::interval(Duration::from_secs(KEEPALIVE_PROBE_INTERVAL_SECS));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        let mut sent = 0usize;
                        for session in sessions.all_sessions() {
                            if !session.is_established() { continue; }
                            match packet_handler.build_keepalive_probe(
                                &session,
                                gateway_ip,
                                Duration::from_secs(KEEPALIVE_ACK_TIMEOUT_SECS),
                            ) {
                                Ok((bytes, endpoint)) => {
                                    if udp.send(&bytes, &endpoint).await.is_ok() {
                                        sent += 1;
                                    }
                                }
                                Err(e) => {
                                    debug!(
                                        session_id = %session.id,
                                        error = %e,
                                        "[KEEPALIVE] Probe build failed"
                                    );
                                }
                            }
                        }
                        if sent > 0 {
                            trace!(sessions = sent, "[KEEPALIVE] ICMP probes sent");
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
        sessions:         Arc<SessionManager>,
        ip_pool:          Arc<IpPoolService>,
        routing:          Arc<RoutingService>,
        events:           SessionEventSender,
        chat_relay:       Option<Arc<ChatRelayService>>,
        traffic_tracker:  Arc<TrafficTracker>,
        deny_list:        Arc<DenyList>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx   = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }

                        for (sid, vip, wallet, bytes_rx, bytes_tx, snap) in sessions.cleanup_expired() {
                            routing.remove_route(vip);
                            events.session_ended(
                                &sid.to_string(),
                                Some(wallet.clone()),
                                Some(vip.to_string()),
                                bytes_rx,
                                bytes_tx,
                                quality_from_stats(snap),
                            );
                            if let Some(ref relay) = chat_relay {
                                relay.wallet_routes.remove_session(&sid);
                            }
                            traffic_tracker.remove_wallet(&wallet);
                        }

                        for ip in sessions.drain_cooldown_pool() {
                            ip_pool.release(ip);
                        }

                        // Evict expired deny list entries (QuotaExceeded whose
                        // month has rolled over). NoPremiumAccess entries are
                        // permanent and are only removed by handle_membership_response.
                        deny_list.cleanup();
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
            .field("tun",    &self.config.device_name())
            .field("mode",   &self.config.memchain.mode)
            .finish()
    }
}
