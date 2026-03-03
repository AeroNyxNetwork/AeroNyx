//! ============================================
//! File: crates/aeronyx-server/src/server.rs
//! ============================================
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
//! - 🐛 v0.3.1: Fixed E0308 Option<Arc<…>> type mismatch in handle_memchain_message.
//! - 🌟 v0.4.0: Phase 3 — Ed25519 signature verification on received Facts,
//!   trusted_agents whitelist enforcement, SyncRequest/SyncResponse handling,
//!   start_api_server extended with sessions + udp for P2P broadcast.
//! - 🌟 v1.3.0: Phase 1 Command Pipeline — `init_management_reporter` now
//!   creates a command `mpsc` channel, wires `HeartbeatReporter` to forward
//!   CMS commands, and spawns `CommandHandler` as a supervised task.
//!   `ManagementClient` is promoted to `Arc` and shared across reporter
//!   and command handler for signed CMS communication.
//!
//! ## Main Functionality
//! - `Server`: Main server struct and lifecycle management
//! - Component initialization and wiring
//! - Async task management
//! - Graceful shutdown handling
//! - Keepalive packet handling
//! - CMS management reporting
//! - 🌟 MemChain MemPool + AofWriter lifecycle
//! - 🌟 Ed25519 signature verification + trust whitelist
//! - 🌟 SyncRequest → SyncResponse reply via encrypted UDP
//! - 🌟 CMS command dispatch pipeline (v1.3.0)
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
//! - 🌟 mempool / aof_writer are Option — always guard with if-let before use
//! - 🌟 handle_memchain_message now requires config + session + udp + crypto
//!   for signature verification and SyncResponse reply
//! - 🌟 v1.3.0: `mgmt_client` is shared between HeartbeatReporter,
//!   SessionReporter, and CommandHandler — do NOT create separate instances
//!
//! ## Last Modified
//! v0.1.0 - Initial server implementation
//! v0.1.1 - Added Keepalive packet handling
//! v0.2.0 - Added CMS management integration
//! v0.3.0 - 🌟 Added MemChain integration (MemPool, AofWriter, 1st-byte dispatch)
//! v0.3.1 - 🐛 Fixed Option<Arc<…>> type mismatch in handle_memchain_message
//! v0.4.0 - 🌟 Phase 3: Ed25519 verify, trust whitelist, SyncReq/SyncRes
//! v1.3.0 - 🌟 Phase 1 Command Pipeline: CommandHandler + channel wiring

use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::{decode_client_hello, encode_server_hello, encode_data_packet, ProtocolCodec};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::{DataPacket, MessageType};
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::UdpTransport;

use crate::api::start_api_server;
use crate::config::MemChainConfig;

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    CommandHandler, HeartbeatReporter, ManagementClient, SessionReporter,
    reporter::SessionEventSender,
};
use crate::services::{
    HandshakeService, IpPoolService, RoutingService, SessionManager,
};
use crate::miner::ReflectionMiner;
use crate::services::memchain::{AofWriter, MemPool};

// ============================================
// Constants
// ============================================

/// Keepalive packet size: 1 byte type + 16 bytes session ID
const KEEPALIVE_PACKET_SIZE: usize = 17;

/// Disconnect packet minimum size: 1 byte type + 16 bytes session ID + 1 byte reason
#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;

/// 🌟 v1.3.0: Command channel buffer size.
/// Should be large enough to handle burst of commands from CMS
/// without blocking the heartbeat loop.
const COMMAND_CHANNEL_BUFFER: usize = 100;

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
        // 🌟 v1.3.0: Also spawns CommandHandler task
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

        // 🌟 Pre-compute server pubkey hex for trust checks
        let server_pubkey_hex = hex::encode(self.identity.public_key_bytes());

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

        // ====================================================
        // 🌟 Start MemChain Agent API (if enabled)
        // ====================================================
        if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
            let api_task = start_api_server(
                self.config.memchain.api_listen_addr,
                Arc::clone(mp),
                Arc::clone(aw),
                self.identity.clone(),
                Arc::clone(&sessions),
                Arc::clone(&udp),
                self.config.memchain.clone(),
                self.shutdown_tx.subscribe(),
            );
            tasks.push(("memchain-api", api_task));

            // 🌟 Start ReflectionMiner (if interval > 0)
            if self.config.memchain.miner_interval_secs > 0 {
                let miner = ReflectionMiner::new(
                    self.config.memchain.miner_interval_secs,
                    Arc::clone(mp),
                    Arc::clone(aw),
                    Arc::clone(&sessions),
                    Arc::clone(&udp),
                );
                let miner_shutdown_rx = self.shutdown_tx.subscribe();
                let miner_task = tokio::spawn(async move {
                    miner.run(miner_shutdown_rx).await;
                });
                tasks.push(("miner", miner_task));
            } else {
                info!("[MINER] Mining disabled (miner_interval_secs=0)");
            }
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
    /// 1. Ensures the AOF parent directory exists (auto-create).
    /// 2. Opens (or creates) the `.memchain` AOF file.
    /// 3. Replays existing facts and blocks from disk into MemPool.
    /// 4. Restores chain state (last block hash/height) for the Miner.
    async fn init_memchain(&self) -> Result<(Arc<MemPool>, Arc<TokioMutex<AofWriter>>)> {
        let aof_path = &self.config.memchain.aof_path;

        // 🐛 v1.3.1: Auto-create parent directory for AOF file.
        // Previously the server would crash with "No such file or directory"
        // if the parent directory didn't exist (e.g., /var/lib/aeronyx/).
        if let Some(parent) = std::path::Path::new(aof_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ServerError::startup_failed(
                        format!("Failed to create AOF directory '{}': {}", parent.display(), e)
                    ))?;
                info!(
                    path = %parent.display(),
                    "[MEMCHAIN] Created AOF directory"
                );
            }
        }

        // Replay existing records from disk (facts + blocks)
        let (existing_facts, last_block) = AofWriter::replay(aof_path)
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

        // 🌟 Restore chain state from replay
        aof_writer.set_chain_state(last_block.as_ref());

        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(
            facts_loaded = loaded,
            mempool_size = mempool.count(),
            last_block_height = last_block.as_ref().map_or(0, |b| b.height),
            "[MEMCHAIN] ✅ Storage engine initialized"
        );

        Ok((mempool, aof_writer))
    }

    // ============================================
    // Management Reporter
    // ============================================

    /// Initializes management reporters, command handler, and agent manager.
    ///
    /// 🌟 v1.3.0 Phase 2: Now also creates the `AgentManager`, detects
    /// existing OpenClaw installations, and wires it into both the
    /// `CommandHandler` (for mutations) and the heartbeat loop (for
    /// status reporting). A periodic health check task is also spawned.
    ///
    /// ## Architecture
    /// ```text
    /// ManagementClient (Arc)
    ///       │
    ///       ├── HeartbeatReporter ──(commands)──→ mpsc ──→ CommandHandler
    ///       │                                                    │
    ///       ├── SessionReporter                          AgentManager (Arc)
    ///       │                                              │
    ///       └── AgentManager ◄─────────────────────────────┘
    ///             │
    ///             └── Health Check Task (periodic)
    /// ```
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

        // ====================================================
        // 🌟 v1.3.0 Phase 2: Initialize AgentManager
        // ====================================================
        let agent_manager = Arc::new(crate::services::AgentManager::new(
            Arc::clone(&mgmt_client),
        ));

        // Detect existing OpenClaw installation on startup
        agent_manager.detect_existing().await;

        // Spawn periodic health check for the agent (every 30s)
        {
            let am = Arc::clone(&agent_manager);
            let mut health_shutdown_rx = self.shutdown_tx.subscribe();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_secs(30),
                );
                loop {
                    tokio::select! {
                        _ = health_shutdown_rx.recv() => {
                            debug!("[AGENT_HEALTH] Stopping health check");
                            break;
                        }
                        _ = interval.tick() => {
                            am.periodic_health_check().await;
                        }
                    }
                }
            });
        }

        info!("[AGENT] Agent manager initialized with health monitoring");

        // ====================================================
        // 🌟 v1.3.0: Create command channel and handler
        // ====================================================
        let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);

        // Spawn CommandHandler task (with AgentManager)
        let cmd_handler = CommandHandler::new(
            command_rx,
            Arc::clone(&mgmt_client),
            Arc::clone(&agent_manager),
        );
        let cmd_shutdown_rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            cmd_handler.run(cmd_shutdown_rx).await;
        });

        info!("[CMD_HANDLER] Command handler spawned");

        // ====================================================
        // Start heartbeat reporter (with command forwarding)
        // ====================================================
        let heartbeat_reporter = HeartbeatReporter::new(
            Arc::clone(&mgmt_client),
            public_ip,
        ).with_command_sender(command_tx);

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

        // ====================================================
        // 🌟 v1.3.0 Phase 3: Start WebSocket Tunnel to CMS
        // ====================================================
        // Only start if the node is registered (we need the CMS node UUID)
        let node_info_path = &self.config.management.node_info_path;
        if let Ok(node_info) = crate::management::models::StoredNodeInfo::load(node_info_path) {
            let ws_tunnel = crate::management::WsTunnel::new(
                self.identity.clone(),
                node_info.node_id.clone(),
                Arc::clone(&agent_manager),
            );
            let ws_shutdown_rx = self.shutdown_tx.subscribe();
            tokio::spawn(async move {
                ws_tunnel.run(ws_shutdown_rx).await;
            });

            info!(
                cms_node_id = %node_info.node_id,
                "[WS_TUNNEL] WebSocket tunnel task spawned"
            );
        } else {
            warn!(
                "[WS_TUNNEL] Node not registered — WebSocket tunnel disabled. \
                 Register the node first with: aeronyx-server register --code <CODE>"
            );
        }

        info!(
            heartbeat_interval = self.config.management.heartbeat_interval_secs,
            command_channel_buffer = COMMAND_CHANNEL_BUFFER,
            "[MANAGEMENT] ✅ Reporting started (heartbeat + session + command + agent + ws_tunnel)"
        );

        SessionEventSender::new(event_tx)
    }

    // ============================================
    // Public IP Resolution
    // ============================================

    /// Resolves the node's public IP address.
    ///
    /// Priority order:
    /// 1. `network.public_endpoint` from config (explicit override)
    /// 2. Auto-detect via external HTTP services (for cloud VMs behind NAT)
    /// 3. Fallback to listen address (last resort)
    ///
    /// 🐛 v1.3.1: Cloud VMs (GCP, AWS, Azure) sit behind NAT.
    /// `listen_addr` is typically `0.0.0.0` which is useless for CMS.
    /// We now query external services to discover the real public IP.
    async fn resolve_public_ip(&self) -> String {
        // Priority 1: Explicit config
        if let Some(ip) = self.config.network.public_ip() {
            info!(ip = %ip, "[NET] Using public IP from config");
            return ip.to_string();
        }

        // Priority 2: Auto-detect from external services
        info!("[NET] No public_endpoint configured, auto-detecting public IP...");

        let detect_services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
            "http://169.254.169.254/latest/meta-data/public-ipv4", // AWS
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip", // GCP
        ];

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .ok();

        if let Some(client) = client {
            for service_url in &detect_services {
                let mut request = client.get(*service_url);

                // GCP metadata requires a special header
                if service_url.contains("metadata.google.internal") {
                    request = request.header("Metadata-Flavor", "Google");
                }

                match request.send().await {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(body) = resp.text().await {
                            let ip = body.trim().to_string();
                            // Basic validation: should look like an IP
                            if !ip.is_empty()
                                && !ip.starts_with('<')
                                && ip.len() <= 45
                                && ip.parse::<std::net::IpAddr>().is_ok()
                            {
                                info!(
                                    ip = %ip,
                                    source = %service_url,
                                    "[NET] ✅ Public IP detected"
                                );
                                return ip;
                            }
                        }
                    }
                    _ => {
                        debug!(
                            service = %service_url,
                            "[NET] IP detection service unavailable"
                        );
                    }
                }
            }
        }

        // Priority 3: Fallback to listen address
        let fallback = self.config.listen_addr().ip().to_string();
        warn!(
            fallback_ip = %fallback,
            "[NET] ⚠️ Could not detect public IP, using listen address. \
             Set network.public_endpoint in server.toml to fix this."
        );
        fallback
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
    /// 🌟 v0.4.0: Now accepts `memchain_config` and `server_pubkey_hex`
    /// for trust verification and SyncResponse reply.
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
        memchain_config: MemChainConfig,
        server_pubkey_hex: String,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let udp_for_reply = Arc::clone(&udp);

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            let crypto = DefaultTransportCrypto::new();

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
                                    data.first().copied().unwrap_or(0)
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

                                                if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
                                                    Self::handle_memchain_message(
                                                        msg,
                                                        mp,
                                                        aw,
                                                        &memchain_config,
                                                        &server_pubkey_hex,
                                                        &session,
                                                        &udp_for_reply,
                                                        &crypto,
                                                    ).await;
                                                } else {
                                                    warn!(
                                                        "[MEMCHAIN_RX] ⚠️ MemChain message received \
                                                         from session {} but MemChain engine is \
                                                         disabled — dropping message",
                                                        session.id
                                                    );
                                                }
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
                                            data.first().copied().unwrap_or(0),
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
    /// ## v0.4.0 Changes
    /// - **BroadcastFact**: Now verifies Ed25519 signature and checks
    ///   origin against trusted_agents whitelist before storing.
    /// - **SyncRequest**: Queries MemPool for missing facts and sends
    ///   back a SyncResponse via encrypted UDP unicast.
    /// - **SyncResponse**: Validates, verifies, and stores each fact.
    #[allow(clippy::too_many_arguments)]
    async fn handle_memchain_message(
        msg: aeronyx_core::protocol::MemChainMessage,
        mempool: &Arc<MemPool>,
        aof_writer: &Arc<TokioMutex<AofWriter>>,
        memchain_config: &MemChainConfig,
        server_pubkey_hex: &str,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        use aeronyx_core::protocol::MemChainMessage;

        match msg {
            MemChainMessage::BroadcastFact(fact) => {
                let fact_id_hex = fact.id_hex();
                let origin_hex = hex::encode(fact.origin);

                info!(
                    fact_id = %fact_id_hex,
                    origin = %origin_hex,
                    subject = %fact.subject,
                    predicate = %fact.predicate,
                    object = %fact.object,
                    "[MEMCHAIN] 📥 Received BroadcastFact"
                );

                // ===== 🌟 Step 1: Ed25519 signature verification =====
                match IdentityPublicKey::from_bytes(&fact.origin) {
                    Ok(pubkey) => {
                        if pubkey.verify(&fact.fact_id, &fact.signature).is_err() {
                            warn!(
                                fact_id = %fact_id_hex,
                                origin = %origin_hex,
                                "[MEMCHAIN] ❌ Signature verification FAILED — dropping fact"
                            );
                            return;
                        }
                        debug!(
                            fact_id = %fact_id_hex,
                            "[MEMCHAIN] ✅ Signature verified"
                        );
                    }
                    Err(_) => {
                        warn!(
                            fact_id = %fact_id_hex,
                            origin = %origin_hex,
                            "[MEMCHAIN] ❌ Invalid origin public key — dropping fact"
                        );
                        return;
                    }
                }

                // ===== 🌟 Step 2: Trust whitelist check =====
                if !memchain_config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                    warn!(
                        fact_id = %fact_id_hex,
                        origin = %origin_hex,
                        "[MEMCHAIN] 🚫 Origin NOT in trusted_agents whitelist — dropping fact"
                    );
                    return;
                }

                // ===== Step 3: Store in MemPool + persist to AOF =====
                if mempool.add_fact(fact.clone()) {
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

            // ===== 🌟 SyncRequest: peer wants to catch up =====
            MemChainMessage::SyncRequest { last_known_hash } => {
                info!(
                    last_known = hex::encode(last_known_hash),
                    session_id = %session.id,
                    "[MEMCHAIN] 📥 SyncRequest received"
                );

                // Query MemPool for facts after the given hash
                let missing_facts = mempool.get_facts_after(last_known_hash);

                info!(
                    facts_to_send = missing_facts.len(),
                    session_id = %session.id,
                    "[MEMCHAIN] 📤 Preparing SyncResponse"
                );

                // Build SyncResponse
                let response_msg = MemChainMessage::SyncResponse {
                    facts: missing_facts,
                };

                // Encode, encrypt, send
                Self::send_memchain_to_session(
                    &response_msg,
                    session,
                    udp,
                    crypto,
                ).await;
            }

            // ===== 🌟 SyncResponse: peer sent us missing facts =====
            MemChainMessage::SyncResponse { facts } => {
                info!(
                    facts_received = facts.len(),
                    session_id = %session.id,
                    "[MEMCHAIN] 📥 SyncResponse received"
                );

                let mut accepted = 0u64;
                let mut rejected = 0u64;

                for fact in facts {
                    let fact_id_hex = fact.id_hex();
                    let origin_hex = hex::encode(fact.origin);

                    // Verify signature
                    let sig_ok = match IdentityPublicKey::from_bytes(&fact.origin) {
                        Ok(pubkey) => pubkey.verify(&fact.fact_id, &fact.signature).is_ok(),
                        Err(_) => false,
                    };

                    if !sig_ok {
                        warn!(
                            fact_id = %fact_id_hex,
                            "[MEMCHAIN_SYNC] ❌ Signature invalid — skipping"
                        );
                        rejected += 1;
                        continue;
                    }

                    // Trust check
                    if !memchain_config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                        warn!(
                            fact_id = %fact_id_hex,
                            origin = %origin_hex,
                            "[MEMCHAIN_SYNC] 🚫 Untrusted origin — skipping"
                        );
                        rejected += 1;
                        continue;
                    }

                    // Store
                    if mempool.add_fact(fact.clone()) {
                        let mut writer = aof_writer.lock().await;
                        if let Err(e) = writer.append_fact(&fact).await {
                            error!(
                                fact_id = %fact_id_hex,
                                error = %e,
                                "[MEMCHAIN_SYNC] ❌ AOF write failed"
                            );
                        }
                        accepted += 1;
                    }
                    // Duplicate (already had it) — not counted as rejected
                }

                info!(
                    accepted = accepted,
                    rejected = rejected,
                    pool_size = mempool.count(),
                    "[MEMCHAIN_SYNC] ✅ SyncResponse processed"
                );
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

            // ===== 🌟 BlockAnnounce: peer mined a new block =====
            MemChainMessage::BlockAnnounce(header) => {
                let block_hash = hex::encode(header.hash());
                info!(
                    height = header.height,
                    block_hash = %block_hash,
                    prev_hash = hex::encode(header.prev_block_hash),
                    merkle_root = hex::encode(header.merkle_root),
                    block_type = header.block_type,
                    session_id = %session.id,
                    "[MEMCHAIN] 📦 BlockAnnounce received"
                );

                // Validate chain continuity: check prev_block_hash links
                // to our known chain. For now, just log — full validation
                // requires tracking local chain state in the UDP task context.
                // The AOF writer tracks this, but it's behind a Mutex.
                // Phase 6+ can add strict validation here.

                info!(
                    height = header.height,
                    block_hash = %block_hash,
                    "[MEMCHAIN] ✅ Block header recorded (light node mode)"
                );
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
    // 🌟 MemChain UDP Reply Helper
    // ============================================

    /// Encodes a MemChainMessage, encrypts it with the session's key,
    /// and sends it as a DataPacket via UDP to the session's endpoint.
    ///
    /// Used by `handle_memchain_message` to send SyncResponse back to
    /// the requesting peer.
    async fn send_memchain_to_session(
        msg: &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        // Encode with 0xAE prefix
        let plaintext = match encode_memchain(msg) {
            Ok(p) => p,
            Err(e) => {
                error!(
                    session_id = %session.id,
                    error = %e,
                    "[MEMCHAIN_TX] ❌ Failed to encode MemChain message"
                );
                return;
            }
        };

        // Encrypt
        let counter = session.next_tx_counter();
        let encrypted_len = plaintext.len() + ENCRYPTION_OVERHEAD;
        let mut encrypted = vec![0u8; encrypted_len];

        let actual_len = match crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            &plaintext,
            &mut encrypted,
        ) {
            Ok(len) => len,
            Err(e) => {
                error!(
                    session_id = %session.id,
                    error = %e,
                    "[MEMCHAIN_TX] ❌ Encryption failed"
                );
                return;
            }
        };
        encrypted.truncate(actual_len);

        // Build DataPacket
        let data_packet = DataPacket::new(
            *session.id.as_bytes(),
            counter,
            encrypted,
        );

        let packet_bytes = encode_data_packet(&data_packet).to_vec();

        // Send
        if let Err(e) = udp.send(&packet_bytes, &session.client_endpoint).await {
            warn!(
                session_id = %session.id,
                endpoint = %session.client_endpoint,
                error = %e,
                "[MEMCHAIN_TX] ⚠️ UDP send failed"
            );
        } else {
            debug!(
                session_id = %session.id,
                packet_len = packet_bytes.len(),
                "[MEMCHAIN_TX] ✅ MemChain message sent"
            );
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
