// [SERVER-SESSION-RUNTIME-SPLIT 2026-09-25 by Codex] Keep VPN service
// initialization and transport shutdown together without changing handshake,
// session cleanup, or public server configuration behavior.
use super::*;

impl Server {
    pub(super) fn init_services(
        &self,
    ) -> Result<(Arc<IpPoolService>, Arc<SessionManager>, Arc<RoutingService>)> {
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
            capacity = ip_pool.capacity(),
            max_sessions = self.config.max_sessions(),
            "Services initialized"
        );
        Ok((ip_pool, sessions, routing))
    }

    #[cfg(target_os = "linux")]
    pub(super) async fn init_tun(&self) -> Result<Arc<LinuxTun>> {
        let (_network, prefix_len) = self.config.parse_ip_range()?;
        let cfg = TunConfig::new(self.config.device_name())
            .with_address(self.config.gateway_ip())
            .with_netmask(prefix_to_netmask(prefix_len))
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

    pub(super) async fn shutdown_udp_transport(udp: &UdpTransport) {
        if let Err(e) = udp.shutdown().await {
            warn!("UDP shutdown error: {}", e);
        }
    }

    // [NODE-ROLES 2026-10-09 by Claude] Moved verbatim out of `Server::run`:
    // UDP, TUN, gateway DNS and the packet/handshake services. Returns the
    // data plane handle every later role reads.
    pub(super) async fn bind_data_plane(
        &self,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        routing: Arc<RoutingService>,
        tasks: &mut RuntimeTaskRegistry,
        critical_failure_tx: &mpsc::Sender<CriticalRuntimeFailure>,
    ) -> Result<DataPlane> {
        // [VPN-BEFORE-DIRECTORY 2026-10-09 by Claude] Bring the VPN data
        // plane (UDP, TUN, gateway DNS, sessions, management, keepalive) up
        // before the Directory Chain/replica startup audits. Nothing below
        // depends on the directory stores; on JP1 those audits held VPN
        // users offline for ~98 s per restart and grew with chain length.
        // Directory features still start only after their audits pass.
        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind: {}", e)))?,
        );
        info!("UDP transport listening on {}", self.config.listen_addr());

        // [VPN-OPTIONAL-ROLE 2026-10-09 by Claude] The relay/mailbox role
        // (`vpn.enabled = false`) never creates a TUN device, so it runs in a
        // container without NET_ADMIN or /dev/net/tun.
        #[cfg(target_os = "linux")]
        let tun = if self.config.vpn_enabled() {
            Some(self.init_tun().await?)
        } else {
            info!("[VPN] Data plane disabled by vpn.enabled=false; no TUN device");
            None
        };

        // AeroNyx client readiness requires DNS to be available at the tunnel
        // gateway. When enabled, this proxy forwards opaque UDP DNS bytes only
        // and never records queried domains, DNS contents, destinations, or
        // client IPs. Operators may disable it when systemd-resolved or another
        // hardened host resolver intentionally owns gateway_ip:53.
        if self.config.dns_proxy_enabled() {
            // [DNS-STARTUP-READINESS 2026-07-30 by Codex] Bind before
            // systemd READY so the node cannot advertise a usable privacy
            // data plane while its configured DNS listener is unavailable.
            let dns_task = start_dns_proxy(self.config.gateway_ip(), self.shutdown_tx.subscribe())
                .await
                .map_err(|error| {
                    ServerError::startup_failed(format!(
                        "required VPN DNS listener failed to bind: {error}"
                    ))
                })?;
            tasks.push((
                "dns-proxy",
                Self::supervise_required_runtime_task(
                    "dns-proxy",
                    dns_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        } else {
            info!(
                gateway_ip = %self.config.gateway_ip(),
                "[DNS] Built-in VPN DNS proxy disabled by vpn.dns_proxy_enabled=false; expecting external gateway DNS listener"
            );
        }

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

        Ok(DataPlane {
            udp,
            #[cfg(target_os = "linux")]
            tun,
            ip_pool,
            sessions,
            routing,
            traffic_tracker,
            encrypted_message_counter,
            deny_list,
            node_policy,
            packet_handler,
            handshake_service,
            voucher_verifier,
        })
    }
}
