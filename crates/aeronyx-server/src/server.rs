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

// Single cfg-gated import — removed the duplicate that was at line 50
#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use rusqlite::OptionalExtension;

use crate::api::mpi::{build_mpi_router, MpiState, BaselineSnapshot};
use crate::config::{MemChainConfig, ServerConfig};
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
use crate::services::{HandshakeService, IpPoolService, RoutingService, SessionManager};

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE: usize = 17;

#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

const COMMAND_CHANNEL_BUFFER: usize = 100;

// ============================================
// Server
// ============================================

/// Main AeroNyx server.
pub struct Server {
    config: ServerConfig,
    identity: IdentityKeyPair,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    /// Creates a new server instance.
    pub fn new(config: ServerConfig, identity: IdentityKeyPair) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            config,
            identity,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
        }
    }

    /// Runs the server until shutdown.
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

        // Initialize transport
        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind: {}", e)))?,
        );
        info!("UDP transport listening on {}", self.config.listen_addr());

        #[cfg(target_os = "linux")]
        let tun = self.init_tun().await?;

        let server_pubkey_hex = hex::encode(self.identity.public_key_bytes());

        // ============================================
        // Spawn worker tasks
        // ============================================
        let mut tasks: Vec<(&str, JoinHandle<()>)> = Vec::new();

        // UDP receive task
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

        // TUN receive task
        #[cfg(target_os = "linux")]
        {
            let tun_task = self.spawn_tun_task(
                Arc::clone(&tun),
                Arc::clone(&udp),
                Arc::clone(&packet_handler),
            );
            tasks.push(("tun", tun_task));
        }

        // Cleanup task
        let cleanup_task = self.spawn_cleanup_task(
            Arc::clone(&sessions),
            Arc::clone(&ip_pool),
            Arc::clone(&routing),
            session_event_sender.clone(),
        );
        tasks.push(("cleanup", cleanup_task));

        // ============================================
        // Start MemChain API + Miner
        // ============================================
        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            // Combined API server with MVF state
            let user_weights = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));

            // Load existing user weights from SQLite
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

            // Load or create MVF baseline from chain_state
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
            });

            // Pre-populate Identity cache
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

            // Mark index ready
            mpi_state.index_ready.store(true, std::sync::atomic::Ordering::Relaxed);

            // Freeze MVF baseline if mvf_enabled just turned on and no baseline exists
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
                mpi_state,
                Arc::clone(mp),
                Arc::clone(aw),
                Arc::clone(&sessions),
                Arc::clone(&udp),
            );
            tasks.push(("memchain-api", api_task));

            // Smart Miner (with MVF integration)
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

                let miner_shutdown = self.shutdown_tx.subscribe();
                tasks.push(("miner", tokio::spawn(async move {
                    miner.run(miner_shutdown).await;
                })));
            } else {
                info!("[MINER] Disabled (interval=0)");
            }
        }

        info!("Server started successfully");

        // Wait for shutdown
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

        // Shutdown stats
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
    // MemChain Initialization (dual-engine)
    // ============================================

    /// Initializes both the new MRS-1 engine (SQLite + VectorIndex)
    /// and the legacy engine (MemPool + AofWriter) for P2P compat.
    ///
    /// Also rebuilds the vector index from SQLite on startup.
    async fn init_memchain(&self) -> Result<(
        Arc<MemoryStorage>,
        Arc<VectorIndex>,
        Arc<MemPool>,
        Arc<TokioMutex<AofWriter>>,
    )> {
        // --- New: SQLite ---
        let db_path = &self.config.memchain.db_path;
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("DB dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let storage = Arc::new(
            MemoryStorage::open(db_path)
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );

        let vector_index = Arc::new(VectorIndex::new());

        // Rebuild vector index from SQLite on startup.
        // Each record is inserted into the correct (owner, model) partition
        // based on the embedding_model stored in SQLite.
        let owner = self.identity.public_key_bytes();
        let records_with_model = storage.get_records_with_embedding(&owner).await;
        let rebuild_count = records_with_model.len();
        for (r, model) in records_with_model {
            if r.has_embedding() {
                vector_index.upsert(
                    r.record_id,
                    r.embedding.clone(),
                    r.layer,
                    r.timestamp,
                    &r.owner,
                    &model,
                );
            }
        }

        info!(
            db = %db_path,
            records = storage.count().await,
            vectors = rebuild_count,
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );

        // --- Legacy: AOF + MemPool ---
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
            if mempool.add_fact(fact) {
                loaded += 1;
            }
        }

        let aof_writer = AofWriter::open(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF open: {}", e)))?;
        aof_writer.set_chain_state(last_block.as_ref());
        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(
            facts = loaded,
            "[MEMCHAIN] Legacy MemPool + AOF initialized"
        );

        Ok((storage, vector_index, mempool, aof_writer))
    }

    // ============================================
    // Combined API Server (MPI + Legacy)
    // ============================================

    /// Starts a single Axum server hosting MPI routes.
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

            let server = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
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
    // Management Reporter (unchanged from v1.3.0)
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

        // Agent manager
        let agent_manager = Arc::new(crate::services::AgentManager::new(
            Arc::clone(&mgmt_client),
        ));
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

        // Command channel + handler
        let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);
        let cmd_handler = CommandHandler::new(
            cmd_rx,
            Arc::clone(&mgmt_client),
            Arc::clone(&agent_manager),
        );
        let cmd_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { cmd_handler.run(cmd_shutdown).await; });

        // Heartbeat reporter
        let heartbeat = HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
            .with_command_sender(cmd_tx)
            .with_agent_manager(Arc::clone(&agent_manager));

        let sess = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            heartbeat.run(move || sess.count() as u32, hb_shutdown).await;
        });

        // Session reporter
        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let sr_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move { session_reporter.run(sr_shutdown).await; });

        // WebSocket tunnel
        let node_info_path = &self.config.management.node_info_path;
        if let Ok(node_info) = crate::management::models::StoredNodeInfo::load(node_info_path) {
            let ws = crate::management::WsTunnel::new(
                self.identity.clone(),
                node_info.node_id.clone(),
                Arc::clone(&agent_manager),
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
    // Public IP Resolution (unchanged)
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
    // Core Services (unchanged)
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
                                    // --- ClientHello ---
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

                                    // --- Keepalive ---
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

                                    // --- Data (VPN + MemChain) ---
                                    Ok(MessageType::Data) | Err(_) => {
                                        match packet_handler.handle_udp_packet(data) {
                                            Ok((_sess, DecryptedPayload::Vpn(pkt))) => {
                                                #[cfg(target_os = "linux")]
                                                { let _ = tun.write(&pkt).await; }
                                            }

                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw,
                                                        &storage, &vector_index,
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
            // --- Legacy: BroadcastFact ---
            MemChainMessage::BroadcastFact(fact) => {
                let origin_hex = hex::encode(fact.origin);

                // Signature verification
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

            // --- New: BroadcastRecord ---
            MemChainMessage::BroadcastRecord(record) => {
                let owner_hex = record.owner_hex();

                let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                    Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!("[MEMCHAIN] BroadcastRecord sig failed");
                    return;
                }
                if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                    warn!("[MEMCHAIN] BroadcastRecord untrusted owner");
                    return;
                }

                if let Some(ref st) = storage {
                    if st.insert(&record, "p2p-remote").await {
                        info!(id = hex::encode(record.record_id), "[MEMCHAIN] BroadcastRecord stored");

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

            // --- Legacy: SyncRequest ---
            MemChainMessage::SyncRequest { last_known_hash } => {
                let facts = mempool.get_facts_after(last_known_hash);
                let resp = MemChainMessage::SyncResponse { facts };
                Self::send_to_session(&resp, session, udp, crypto).await;
            }

            // --- Legacy: SyncResponse ---
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

            // --- New: SyncRecordRequest ---
            MemChainMessage::SyncRecordRequest { owner, after_timestamp } => {
                if let Some(ref st) = storage {
                    let records = st.query_by_owner_after(&owner, after_timestamp).await;
                    let resp = MemChainMessage::SyncRecordResponse { records };
                    Self::send_to_session(&resp, session, udp, crypto).await;
                }
            }

            // --- New: SyncRecordResponse ---
            MemChainMessage::SyncRecordResponse { records } => {
                if let Some(ref st) = storage {
                    for record in records {
                        let owner_hex = record.owner_hex();
                        let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                            Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                            Err(_) => false,
                        };
                        if !sig_ok || !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                            continue;
                        }
                        let _ = st.insert(&record, "p2p-sync").await;
                    }
                }
            }

            // --- BlockAnnounce ---
            MemChainMessage::BlockAnnounce(header) => {
                info!(
                    height = header.height,
                    hash = hex::encode(header.hash()),
                    "[MEMCHAIN] BlockAnnounce received"
                );
            }

            // --- Everything else ---
            _ => {
                debug!("[MEMCHAIN] Unhandled message variant");
            }
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
    // TUN Task (unchanged)
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
    // Cleanup Task (unchanged)
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

    /// Programmatic shutdown trigger.
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
