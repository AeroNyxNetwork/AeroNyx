// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
//! # Server Orchestrator
//!
//! ## Creation Reason
//! Main server implementation that coordinates all components and
//! manages the server lifecycle.
//!
//! ## Modification Reason
//! - Added Keepalive packet handling to prevent session timeout.
//! - Added CMS management integration (heartbeat + session reporting).
//! - 🌟 Added MemChain integration: MemPool + AofWriter initialisation,
//!   1st-byte multiplexing dispatch in UDP task, AOF replay on startup.
//!
//! ## Main Functionality
//! - `Server`: Main server struct and lifecycle management
//! - Component initialization and wiring
//! - Async task management
//! - Graceful shutdown handling
//! - Keepalive packet handling
//! - CMS management reporting
//! - 🌟 MemChain MemPool + AofWriter lifecycle
//!
//! ## Packet Types Handled
//! - 0x01 ClientHello: Handshake initiation
//! - 0x02 ServerHello: Handshake response (sent by server)
//! - 0x03 Data: Encrypted VPN traffic (no prefix, identified by structure)
//!   - 🌟 After decrypt: plaintext[0] == 0x45/0x60 → VPN → TUN
//!   - 🌟 After decrypt: plaintext[0] == 0xAE → MemChain → MemPool
//! - 0x04 Keepalive: Session heartbeat
//!
//! ## ⚠️ Important Note for Next Developer
//! - Server requires root for TUN operations
//! - Graceful shutdown waits for tasks to complete
//! - All services are Arc-wrapped for sharing
//! - Use tokio::select! for concurrent operations
//! - Keepalive packets MUST update session.last_activity
//! - Node must be registered before starting (checked in main.rs)
//! - 🌟 MemPool is Arc<MemPool> (DashMap is internally sync)
//! - 🌟 AofWriter is Arc<TokioMutex<AofWriter>> (file needs exclusive access)
//! - 🌟 AOF replay happens before UDP task starts
//!
//! ## Last Modified
//! v0.1.0 - Initial server implementation
//! v0.1.1 - Added Keepalive packet handling
//! v0.2.0 - Added CMS management integration
//! v0.3.0 - 🌟 Added MemChain integration (MemPool, AofWriter, 1st-byte dispatch)

use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::{broadcast, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::codec::{decode_client_hello, encode_server_hello, ProtocolCodec};
use aeronyx_core::protocol::MessageType;
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::UdpTransport;

use crate::api::start_api_server;
use crate::config::MemChainMode;

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    HeartbeatReporter, ManagementClient, SessionReporter,
    reporter::SessionEventSender,
};
use crate::services::{
    HandshakeService, IpPoolService, RoutingService, SessionManager,
};
use crate::services::memchain::{AofWriter, MemPool};

// ============================================
// Constants
// ============================================

/// Keepalive packet size: 1 byte type + 16 bytes session ID
const KEEPALIVE_PACKET_SIZE: usize = 17;

/// Disconnect packet minimum size: 1 byte type + 16 bytes session ID + 1 byte reason
#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

// ============================================
// Server
// ============================================

/// Main AeroNyx server.
///
/// # Lifecycle
/// 1. Create with `Server::new(config, identity)`
/// 2. Start with `server.run().await`
/// 3. Shutdown via shutdown signal or Ctrl+C
pub struct Server {
    /// Server configuration.
    config: ServerConfig,
    /// Server identity key pair.
    identity: IdentityKeyPair,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Shutdown signal sender.
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

        // Initialize services
        let (ip_pool, sessions, routing) = self.init_services()?;

        // Initialize session event sender (for CMS reporting)
        let session_event_sender = self.init_management_reporter(&sessions).await;

        // Initialize handshake service
        let handshake_service = Arc::new(HandshakeService::new(
            self.identity.clone(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
        ));

        // Initialize packet handler
        let packet_handler = Arc::new(PacketHandler::new(
            Arc::clone(&sessions),
            Arc::clone(&routing),
        ));

        // ====================================================
        // 🌟 Initialize MemChain storage engine (if enabled)
        // ====================================================
        let (mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (mp, aw) = self.init_memchain().await?;
            (Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] MemChain is disabled (mode=off)");
            (None, None)
        };

        // Initialize transport
        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind failed: {}", e)))?,
        );

        info!("UDP transport listening on {}", self.config.listen_addr());

        // Initialize TUN device (Linux only for now)
        #[cfg(target_os = "linux")]
        let tun = self.init_tun().await?;

        // Spawn worker tasks
        let mut tasks = Vec::new();

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

        // ====================================================
        // 🌟 Start MemChain Agent API (if enabled)
        // ====================================================
        if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
            let api_task = start_api_server(
                self.config.memchain.api_listen_addr,
                Arc::clone(mp),
                Arc::clone(aw),
                self.identity.clone(),
                self.shutdown_tx.subscribe(),
            );
            tasks.push(("memchain-api", api_task));
        }

        info!("Server started successfully");

        // Wait for shutdown signal
        self.wait_for_shutdown().await;

        // Shutdown
        info!("Shutting down server...");
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());

        // Wait for tasks to complete
        for (name, task) in tasks {
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => debug!("Task '{}' completed", name),
                Ok(Err(e)) => warn!("Task '{}' failed: {}", name, e),
                Err(_) => warn!("Task '{}' timed out during shutdown", name),
            }
        }

        // Cleanup transport
        if let Err(e) = udp.shutdown().await {
            warn!("UDP shutdown error: {}", e);
        }

        if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
            info!(
                mempool_facts = mp.count(),
                aof_writes = aw.lock().await.write_count(),
                "Server shutdown complete (MemChain stats)"
            );
        } else {
            info!("Server shutdown complete");
        }
        Ok(())
    }

    // ============================================
    // 🌟 MemChain Initialization
    // ============================================

    /// Initializes the MemChain storage engine.
    ///
    /// 1. Opens (or creates) the `.memchain` AOF file.
    /// 2. Replays existing facts from disk into MemPool.
    async fn init_memchain(&self) -> Result<(Arc<MemPool>, Arc<TokioMutex<AofWriter>>)> {
        let aof_path = &self.config.memchain.aof_path;

        // Replay existing facts from disk
        let existing_facts = AofWriter::replay(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF replay failed: {}", e)))?;

        let mempool = Arc::new(MemPool::new());

        // Rehydrate MemPool from disk
        let mut loaded = 0u64;
        for fact in existing_facts {
            if mempool.add_fact(fact) {
                loaded += 1;
            }
        }

        // Open AOF for append
        let aof_writer = AofWriter::open(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF open failed: {}", e)))?;

        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(
            facts_loaded = loaded,
            mempool_size = mempool.count(),
            "[MEMCHAIN] ✅ Storage engine initialized"
        );

        Ok((mempool, aof_writer))
    }

    // ============================================
    // Management Reporter
    // ============================================

    /// Initializes management reporters.
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

        let public_ip = self.config.network.public_ip()
            .map(|ip| ip.to_string())
            .unwrap_or_else(|| {
                self.config.listen_addr().ip().to_string()
            });

        // Start heartbeat reporter
        let heartbeat_reporter = HeartbeatReporter::new(
            Arc::clone(&mgmt_client),
            public_ip,
        );

        let sessions_for_heartbeat = Arc::clone(sessions);
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            heartbeat_reporter.run(
                move || sessions_for_heartbeat.count() as u32,
                shutdown_rx,
            ).await;
        });

        // Start session reporter
        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            session_reporter.run(shutdown_rx).await;
        });

        info!(
            "Management reporting started (heartbeat interval: {}s)",
            self.config.management.heartbeat_interval_secs
        );

        SessionEventSender::new(event_tx)
    }

    // ============================================
    // Core Services
    // ============================================

    /// Initializes core services.
    fn init_services(&self) -> Result<(
        Arc<IpPoolService>,
        Arc<SessionManager>,
        Arc<RoutingService>,
    )> {
        let (network, prefix) = self.config.parse_ip_range()?;

        let ip_pool = Arc::new(IpPoolService::new(
            network,
            prefix,
            self.config.gateway_ip(),
        )?);

        let sessions = Arc::new(SessionManager::new(
            self.config.max_sessions(),
            Duration::from_secs(self.config.session_timeout_secs()),
        ));

        let routing = Arc::new(RoutingService::new());

        info!(
            "Services initialized: IP pool capacity={}, max sessions={}",
            ip_pool.capacity(),
            self.config.max_sessions()
        );

        Ok((ip_pool, sessions, routing))
    }

    /// Initializes the TUN device.
    #[cfg(target_os = "linux")]
    async fn init_tun(&self) -> Result<Arc<LinuxTun>> {
        let tun_config = TunConfig::new(self.config.device_name())
            .with_address(self.config.gateway_ip())
            .with_netmask(Ipv4Addr::new(255, 255, 255, 0))
            .with_mtu(self.config.mtu());

        let tun = LinuxTun::create(tun_config)
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN creation failed: {}", e)))?;

        tun.up()
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN activation failed: {}", e)))?;

        info!(
            "TUN device '{}' initialized with IP {}",
            tun.name(),
            self.config.gateway_ip()
        );

        Ok(Arc::new(tun))
    }

    // ============================================
    // UDP Task
    // ============================================

    /// Spawns the UDP receive task.
    ///
    /// 🌟 Now accepts `mempool` and `aof_writer` to route MemChain
    /// messages to the storage engine.
    #[allow(clippy::too_many_arguments)]
    fn spawn_udp_task(
        &self,
        udp: Arc<UdpTransport>,
        #[cfg(target_os = "linux")]
        tun: Arc<LinuxTun>,
        handshake: Arc<HandshakeService>,
        packet_handler: Arc<PacketHandler>,
        sessions: Arc<SessionManager>,
        session_events: SessionEventSender,
        mempool: Option<Arc<MemPool>>,
        aof_writer: Option<Arc<TokioMutex<AofWriter>>>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        debug!("UDP task received shutdown signal");
                        break;
                    }
                    result = udp.recv(&mut buf) => {
                        match result {
                            Ok((len, source)) => {
                                if shutdown.load(Ordering::SeqCst) {
                                    break;
                                }

                                let data = &buf[..len];

                                debug!(
                                    "[UDP_RX] Received {} bytes from {}, first_byte=0x{:02X}",
                                    len,
                                    source.addr,
                                    data.get(0).copied().unwrap_or(0)
                                );

                                // ========== Dispatch by outer message type ==========
                                match ProtocolCodec::peek_message_type(data) {

                                    // ---------- ClientHello ----------
                                    Ok(MessageType::ClientHello) => {
                                        info!(
                                            "[HANDSHAKE] ClientHello received from {}",
                                            source.addr
                                        );

                                        match decode_client_hello(data) {
                                            Ok(client_hello) => {
                                                match handshake.process(&client_hello, source.addr) {
                                                    Ok(result) => {
                                                        let session_id_b64 = BASE64.encode(&result.response.session_id);
                                                        info!(
                                                            "[HANDSHAKE] ✅ Session created for {}, SessionID={}",
                                                            source.addr,
                                                            session_id_b64
                                                        );

                                                        session_events.session_created(
                                                            &session_id_b64,
                                                            None,
                                                        );

                                                        let response = encode_server_hello(&result.response);
                                                        if let Err(e) = udp.send(&response, &source.addr).await {
                                                            warn!("Failed to send ServerHello: {}", e);
                                                        } else {
                                                            debug!(
                                                                "[HANDSHAKE] ServerHello sent to {}, response_len={}",
                                                                source.addr,
                                                                response.len()
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        warn!(
                                                            "[HANDSHAKE] ❌ Failed from {}: {}",
                                                            source.addr, e
                                                        );
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                debug!("Invalid ClientHello from {}: {}", source.addr, e);
                                            }
                                        }
                                    }

                                    // ---------- Keepalive ----------
                                    Ok(MessageType::Keepalive) => {
                                        if len >= KEEPALIVE_PACKET_SIZE {
                                            let mut session_id_bytes = [0u8; 16];
                                            session_id_bytes.copy_from_slice(&data[1..17]);

                                            if let Some(session_id) = SessionId::from_bytes(&session_id_bytes) {
                                                if let Some(session) = sessions.get(&session_id) {
                                                    session.touch();
                                                    trace!(
                                                        "[KEEPALIVE] ✅ Session {} touched from {}",
                                                        session_id,
                                                        source.addr
                                                    );
                                                } else {
                                                    debug!(
                                                        "[KEEPALIVE] Session not found: {} from {}",
                                                        BASE64.encode(&session_id_bytes),
                                                        source.addr
                                                    );
                                                }
                                            } else {
                                                debug!(
                                                    "[KEEPALIVE] Invalid session ID format from {}",
                                                    source.addr
                                                );
                                            }
                                        } else {
                                            debug!(
                                                "[KEEPALIVE] Packet too short from {}: {} bytes",
                                                source.addr, len
                                            );
                                        }
                                    }

                                    // ========================================
                                    // Data packets (VPN + 🌟 MemChain)
                                    // ========================================
                                    Ok(MessageType::Data) | Err(_) => {
                                        if data.len() >= 16 {
                                            debug!(
                                                "[DATA_RX] Data packet from {}, SessionID={}, len={}",
                                                source.addr,
                                                BASE64.encode(&data[0..16]),
                                                len
                                            );
                                        } else {
                                            warn!(
                                                "[DATA_RX] Packet too short from {}, len={}",
                                                source.addr, len
                                            );
                                        }

                                        // Decrypt and route based on payload type
                                        match packet_handler.handle_udp_packet(data) {

                                            // ---- VPN IP packet → write to TUN ----
                                            Ok((_session, DecryptedPayload::Vpn(ip_packet))) => {
                                                debug!(
                                                    "[DATA_RX] ✅ VPN packet decrypted, len={}",
                                                    ip_packet.len()
                                                );

                                                #[cfg(target_os = "linux")]
                                                if let Err(e) = tun.write(&ip_packet).await {
                                                    debug!("TUN write error: {}", e);
                                                }
                                            }

                                            // ---- 🌟 MemChain → route to MemPool ----
                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                debug!(
                                                    "[MEMCHAIN_RX] ✅ MemChain msg from session {}, type={:?}",
                                                    session.id,
                                                    std::mem::discriminant(&msg)
                                                );

                                                Self::handle_memchain_message(
                                                    msg,
                                                    &mempool,
                                                    &aof_writer,
                                                ).await;
                                            }

                                            // ---- Error ----
                                            Err(e) => {
                                                if data.len() >= 16 {
                                                    warn!(
                                                        "[DATA_RX] ❌ FAILED from {}, SessionID={}, error: {}",
                                                        source.addr,
                                                        BASE64.encode(&data[0..16]),
                                                        e
                                                    );
                                                } else {
                                                    debug!(
                                                        "Packet handling error from {}: {}",
                                                        source.addr, e
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    // ---------- Unknown outer types ----------
                                    _ => {
                                        debug!(
                                            "[UDP_RX] Unknown message type 0x{:02X} from {}",
                                            data.get(0).copied().unwrap_or(0),
                                            source.addr
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) {
                                    error!("UDP receive error: {}", e);
                                }
                            }
                        }
                    }
                }
            }

            debug!("UDP task exiting");
        })
    }

    // ============================================
    // 🌟 MemChain Message Handler
    // ============================================

    /// Handles a decoded MemChain message from the encrypted tunnel.
    ///
    /// Currently supports:
    /// - `BroadcastFact`: Validate, store in MemPool, persist to AOF.
    /// - Other variants: Logged as unimplemented stubs.
    async fn handle_memchain_message(
        msg: aeronyx_core::protocol::MemChainMessage,
        mempool: &Arc<MemPool>,
        aof_writer: &Arc<TokioMutex<AofWriter>>,
    ) {
        use aeronyx_core::protocol::MemChainMessage;

        match msg {
            MemChainMessage::BroadcastFact(fact) => {
                let fact_id_hex = fact.id_hex();
                info!(
                    fact_id = %fact_id_hex,
                    subject = %fact.subject,
                    predicate = %fact.predicate,
                    object = %fact.object,
                    "[MEMCHAIN] 📥 Received BroadcastFact"
                );

                // TODO: Verify Ed25519 signature using fact.origin public key.
                // For MVP, we trust facts from authenticated sessions
                // (the session itself was authenticated via Ed25519 handshake).

                // Add to MemPool (validates hash integrity + dedup)
                if mempool.add_fact(fact.clone()) {
                    // Persist to AOF
                    let mut writer = aof_writer.lock().await;
                    if let Err(e) = writer.append_fact(&fact).await {
                        error!(
                            fact_id = %fact_id_hex,
                            error = %e,
                            "[MEMCHAIN] ❌ AOF write failed"
                        );
                    } else {
                        info!(
                            fact_id = %fact_id_hex,
                            pool_size = mempool.count(),
                            "[MEMCHAIN] ✅ Fact stored and persisted"
                        );
                    }
                }
            }

            MemChainMessage::SyncRequest { last_known_hash } => {
                debug!(
                    last_known = hex::encode(last_known_hash),
                    "[MEMCHAIN] SyncRequest received (stub — not yet implemented)"
                );
                // TODO: Respond with SyncResponse containing missing facts
            }

            MemChainMessage::QueryRequest { fact_id } => {
                debug!(
                    fact_id = hex::encode(fact_id),
                    "[MEMCHAIN] QueryRequest received (stub — not yet implemented)"
                );
                // TODO: Look up in MemPool and respond with QueryResponse
            }

            MemChainMessage::Ping { nonce } => {
                debug!(
                    nonce = nonce,
                    "[MEMCHAIN] Ping received (stub — not yet implemented)"
                );
                // TODO: Send back Pong with same nonce via encrypted tunnel
            }

            other => {
                debug!(
                    msg_type = ?std::mem::discriminant(&other),
                    "[MEMCHAIN] Unhandled message variant"
                );
            }
        }
    }

    // ============================================
    // TUN Task
    // ============================================

    /// Spawns the TUN receive task.
    #[cfg(target_os = "linux")]
    fn spawn_tun_task(
        &self,
        tun: Arc<LinuxTun>,
        udp: Arc<UdpTransport>,
        packet_handler: Arc<PacketHandler>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        debug!("TUN task received shutdown signal");
                        break;
                    }
                    result = tun.read(&mut buf) => {
                        match result {
                            Ok(len) => {
                                if shutdown.load(Ordering::SeqCst) {
                                    break;
                                }

                                let ip_packet = &buf[..len];

                                match packet_handler.handle_tun_packet(ip_packet) {
                                    Ok((encrypted, endpoint)) => {
                                        if let Err(e) = udp.send(&encrypted, &endpoint).await {
                                            debug!("UDP send error: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        debug!("TUN packet handling error: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) {
                                    error!("TUN read error: {}", e);
                                }
                            }
                        }
                    }
                }
            }

            debug!("TUN task exiting");
        })
    }

    // ============================================
    // Cleanup Task
    // ============================================

    /// Spawns the session cleanup task.
    fn spawn_cleanup_task(
        &self,
        sessions: Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        routing: Arc<RoutingService>,
        session_events: SessionEventSender,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let interval = Duration::from_secs(60);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        debug!("Cleanup task received shutdown signal");
                        break;
                    }
                    _ = interval_timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) {
                            break;
                        }

                        let expired = sessions.cleanup_expired();

                        for (session_id, virtual_ip) in expired {
                            routing.remove_route(virtual_ip);
                            ip_pool.release(virtual_ip);

                            session_events.session_ended(
                                &session_id.to_string(),
                                None,
                                0,
                                0,
                            );

                            debug!(
                                session_id = %session_id,
                                virtual_ip = %virtual_ip,
                                "Released resources for expired session"
                            );
                        }

                        debug!(
                            sessions = sessions.count(),
                            ips_allocated = ip_pool.allocated_count(),
                            routes = routing.count(),
                            "Cleanup cycle complete"
                        );
                    }
                }
            }

            debug!("Cleanup task exiting");
        })
    }

    // ============================================
    // Shutdown
    // ============================================

    /// Waits for shutdown signal (Ctrl+C or programmatic).
    async fn wait_for_shutdown(&self) {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Received shutdown signal");
    }

    /// Triggers server shutdown programmatically.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());
    }
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("listen_addr", &self.config.listen_addr())
            .field("tun_device", &self.config.device_name())
            .finish()
    }
}
