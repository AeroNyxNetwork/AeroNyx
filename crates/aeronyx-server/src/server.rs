// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
//! # Server Orchestrator
//!
//! ## Creation Reason
//! Main server implementation that coordinates all components and
//! manages the server lifecycle.
//!
//! ## Main Functionality
//! - `Server`: Main server struct and lifecycle management
//! - Component initialization and wiring
//! - Async task management
//! - Graceful shutdown handling
//!
//! ## Server Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         Server                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │                   Main Loop                          │   │
//! │  │                                                      │   │
//! │  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │   │
//! │  │  │ UDP Task   │  │ TUN Task   │  │ Cleanup Task │   │   │
//! │  │  │            │  │            │  │              │   │   │
//! │  │  │ Receive    │  │ Read       │  │ Expire       │   │   │
//! │  │  │ packets    │  │ packets    │  │ sessions     │   │   │
//! │  │  │            │  │            │  │              │   │   │
//! │  │  └─────┬──────┘  └─────┬──────┘  └──────────────┘   │   │
//! │  │        │               │                            │   │
//! │  │        ▼               ▼                            │   │
//! │  │  ┌─────────────────────────────────────────────┐   │   │
//! │  │  │            Packet Handler                    │   │   │
//! │  │  └─────────────────────────────────────────────┘   │   │
//! │  │                                                      │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │                    Services                          │   │
//! │  │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │   │
//! │  │  │ Session  │ │ Routing  │ │IP Pool │ │Handshake │ │   │
//! │  │  │ Manager  │ │ Service  │ │Service │ │ Service  │ │   │
//! │  │  └──────────┘ └──────────┘ └────────┘ └──────────┘ │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Server requires root for TUN operations
//! - Graceful shutdown waits for tasks to complete
//! - All services are Arc-wrapped for sharing
//! - Use tokio::select! for concurrent operations
//!
//! ## Last Modified
//! v0.1.0 - Initial server implementation

use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::codec::{decode_client_hello, encode_server_hello, ProtocolCodec};
use aeronyx_core::protocol::{MessageType, CURRENT_PROTOCOL_VERSION};
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::{UdpTransport};

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::handlers::PacketHandler;
use crate::services::{
    HandshakeService, IpPoolService, RoutingService, SessionManager,
};

// ============================================
// Server
// ============================================

/// Main AeroNyx server.
///
/// # Lifecycle
/// 1. Create with `Server::new(config)`
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
    ///
    /// # Arguments
    /// * `config` - Server configuration
    /// * `identity` - Server's Ed25519 identity key pair
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
    ///
    /// # Errors
    /// Returns error if server fails to start or encounters a fatal error.
    pub async fn run(&self) -> Result<()> {
        info!("Starting AeroNyx server v{}", env!("CARGO_PKG_VERSION"));

        // Initialize services
        let (ip_pool, sessions, routing) = self.init_services()?;

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

        // Initialize transport
        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.network.listen_addr)
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind failed: {}", e)))?,
        );

        info!("UDP transport listening on {}", self.config.network.listen_addr);

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
        );
        tasks.push(("cleanup", cleanup_task));

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

        info!("Server shutdown complete");
        Ok(())
    }

    /// Initializes core services.
    fn init_services(&self) -> Result<(
        Arc<IpPoolService>,
        Arc<SessionManager>,
        Arc<RoutingService>,
    )> {
        // Parse IP range
        let (network, prefix) = self.config.tunnel.parse_ip_range()?;

        // Create IP pool
        let ip_pool = Arc::new(IpPoolService::new(
            network,
            prefix,
            self.config.tunnel.gateway_ip,
        )?);

        // Create session manager
        let sessions = Arc::new(SessionManager::new(
            self.config.limits.max_sessions,
            Duration::from_secs(self.config.limits.session_timeout_secs),
        ));

        // Create routing service
        let routing = Arc::new(RoutingService::new());

        info!(
            "Services initialized: IP pool capacity={}, max sessions={}",
            ip_pool.capacity(),
            self.config.limits.max_sessions
        );

        Ok((ip_pool, sessions, routing))
    }

    /// Initializes the TUN device.
    #[cfg(target_os = "linux")]
    async fn init_tun(&self) -> Result<Arc<LinuxTun>> {
        let tun_config = TunConfig::new(&self.config.tunnel.device_name)
            .with_address(self.config.tunnel.gateway_ip)
            .with_netmask(Ipv4Addr::new(255, 255, 255, 0))
            .with_mtu(self.config.tunnel.mtu);

        let tun = LinuxTun::create(tun_config)
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN creation failed: {}", e)))?;

        tun.up()
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN activation failed: {}", e)))?;

        info!(
            "TUN device '{}' initialized with IP {}",
            tun.name(),
            self.config.tunnel.gateway_ip
        );

        Ok(Arc::new(tun))
    }

    /// Spawns the UDP receive task.
    fn spawn_udp_task(
        &self,
        udp: Arc<UdpTransport>,
        #[cfg(target_os = "linux")]
        tun: Arc<LinuxTun>,
        handshake: Arc<HandshakeService>,
        packet_handler: Arc<PacketHandler>,
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
                                
                                // Check message type
                                match ProtocolCodec::peek_message_type(data) {
                                    Ok(MessageType::ClientHello) => {
                                        // Handle handshake
                                        match decode_client_hello(data) {
                                            Ok(client_hello) => {
                                                match handshake.process(&client_hello, source.addr) {
                                                    Ok(result) => {
                                                        let response = encode_server_hello(&result.response);
                                                        if let Err(e) = udp.send(&response, &source.addr).await {
                                                            warn!("Failed to send ServerHello: {}", e);
                                                        }
                                                    }
                                                    Err(e) => {
                                                        debug!("Handshake failed from {}: {}", source.addr, e);
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                debug!("Invalid ClientHello from {}: {}", source.addr, e);
                                            }
                                        }
                                    }
                                    Ok(MessageType::Data) | Err(_) => {
                                        // Try to handle as data packet
                                        match packet_handler.handle_udp_packet(data) {
                                            Ok((_session, ip_packet)) => {
                                                #[cfg(target_os = "linux")]
                                                if let Err(e) = tun.write(&ip_packet).await {
                                                    debug!("TUN write error: {}", e);
                                                }
                                            }
                                            Err(e) => {
                                                debug!("Packet handling error from {}: {}", source.addr, e);
                                            }
                                        }
                                    }
                                    _ => {
                                        debug!("Ignoring message type from {}", source.addr);
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
                                        // No route is common, don't log at warn level
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

    /// Spawns the session cleanup task.
    fn spawn_cleanup_task(
        &self,
        sessions: Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        routing: Arc<RoutingService>,
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

                        // Cleanup expired sessions
                        let expired = sessions.cleanup_expired();
                        
                        // Release resources for expired sessions
                        for (session_id, virtual_ip) in expired {
                            routing.remove_route(virtual_ip);
                            ip_pool.release(virtual_ip);
                            debug!(
                                session_id = %session_id,
                                virtual_ip = %virtual_ip,
                                "Released resources for expired session"
                            );
                        }

                        // Log stats
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
            .field("listen_addr", &self.config.network.listen_addr)
            .field("tun_device", &self.config.tunnel.device_name)
            .finish()
    }
}
