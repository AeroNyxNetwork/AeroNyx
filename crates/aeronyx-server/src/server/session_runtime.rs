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
}
