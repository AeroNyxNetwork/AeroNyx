// ============================================
// File: crates/aeronyx-server/src/api/vpn_health.rs
// ============================================
//! VPN node health endpoint.
//!
//! This endpoint is intentionally read-only. It verifies the Linux node pieces
//! that commonly make a tunnel appear "connected but offline": UDP listener,
//! TUN device and MTU, IPv4 forwarding, NAT masquerade, VPN DNS stub, DNS
//! resolution, basic Internet egress, and aggregate encrypted VPN message
//! forwarding counters. The same router also exposes a node-operator status
//! snapshot for nodeboard so operators can see which AeroNyx services this
//! Rust node is currently ready to provide. The capacity block includes
//! structured placement risks so nodeboard, CLI healthchecks, and backend
//! automation can share the same commercial readiness decisions.
//!
//! DNS ownership telemetry is configuration metadata only. It reports whether
//! Rust or an external gateway resolver owns `gateway_ip:53`; it never includes
//! DNS query names, resolver payloads, destinations, client public IPs,
//! domains, URLs, browsing history, voucher secrets, or wallet-level traffic.
//! Transport capability telemetry is also metadata only. Phase 1 reports that
//! UDP is the only active data-plane carrier while TCP/TLS and WebSocket HTTPS
//! remain planned fallback carriers until their runtime listeners are added.
//! Recent error telemetry is sourced from local service logs, sanitized, and
//! capped so nodeboard can triage node operations without collecting client
//! public IPs, destinations, DNS contents, packet payloads, domains, URLs,
//! voucher secrets, chat plaintext, or wallet-level traffic.
//! Journal severity is mapped to `info`, `warning`, or `critical` before
//! heartbeat reporting so nodeboard can prioritize operator action without
//! shipping raw service logs.
//! Upgrade workflow telemetry is read from `/var/lib/aeronyx/upgrade-status.json`
//! and allow-listed before heartbeat reporting. It reports only install/upgrade
//! workflow state so nodeboard can show the current operator action without
//! exposing registration codes, private keys, user identifiers, destinations,
//! DNS contents, payloads, chat plaintext, or wallet-level traffic.
//! Disk capacity telemetry reports only aggregate filesystem usage for `/` and
//! `/var/lib/aeronyx`; it never lists files, paths below those operational
//! roots, message contents, MemChain records, user identifiers, destinations,
//! DNS contents, payloads, chat plaintext, or wallet-level traffic.
//! Operator action telemetry is derived from the same local checks, capacity
//! risks, service manager state, and upgrade workflow metadata. It is a compact
//! nodeboard/AI runbook summary and does not collect additional user traffic
//! data.
//! Privacy protocol health telemetry is an additive nodeboard contract that
//! summarizes protocol status, failed checks, active aggregate sessions, service
//! state, transport model, and AeroNyx protocol runtime usage. It avoids requiring
//! backend or UI code to infer AeroNyx privacy protocol readiness from several
//! lower-level fields, and it remains aggregate operations metadata only.
//! Packet runtime telemetry reports process-local aggregate packet counters so
//! operators can diagnose restart recovery pressure without reading raw journals.
//! Discovery status telemetry reports the local verified PeerStore summary so
//! healthchecks can prove that nodes know about other signed AeroNyx peers
//! before future multi-hop routing work starts.
//! Startup self-check telemetry aggregates validated config, runtime listeners,
//! peer-cache recovery, seed gossip, and capacity prerequisites into one
//! nodeboard-ready readiness contract. It is operational metadata only and
//! never includes user traffic, chat payloads, DNS names, destinations, wallet
//! traffic, voucher secrets, or private keys.
//! Systemd health and journal reads resolve the current service unit from a
//! validated operator override or the process cgroup, then fall back to the
//! historical `aeronyx-server` unit name. This keeps single-node deployments
//! backward compatible while avoiding false failures on isolated regional or
//! multi-instance units such as `aeronyx-server-jp1.service`.

use std::collections::HashMap;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{extract::State, response::IntoResponse, routing::get, Json, Router};
use serde::Serialize;
use serde_json::Value;
use tokio::net::{TcpStream, UdpSocket};
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;

use crate::config::ServerConfig;
use crate::handlers::packet::{PacketHandler, PacketRuntimeStatus};
use crate::management::integrity;
use crate::services::session::CLIENT_LIVENESS_TIMEOUT_SECS;
use crate::services::{
    IpPoolService, NodePolicyEnforcementSnapshot, NodePolicyPlacementSnapshot, NodePolicyRuntime,
    NodePolicySnapshot, PeerStore, SessionManager,
};
use crate::voucher_verifier::{VoucherMetricsSnapshot, VoucherVerifier};

const CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const DNS_QUERY_NAME: &str = "api.aeronyx.network";
const EGRESS_CHECK_ADDR: &str = "1.1.1.1:443";
const VPN_SERVICE_NAME: &str = "aeronyx-server";
const VPN_SERVICE_NAME_ENV: &str = "AERONYX_SYSTEMD_SERVICE";
const PROC_SELF_CGROUP_PATH: &str = "/proc/self/cgroup";
const UPGRADE_STATUS_FILE: &str = "/var/lib/aeronyx/upgrade-status.json";
const AERONYX_STATE_DIR: &str = "/var/lib/aeronyx";
static RUNTIME_STARTED_AT: OnceLock<u64> = OnceLock::new();

#[derive(Clone)]
pub struct VpnHealthState {
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
}

#[derive(Debug, Clone, Serialize)]
struct HealthCheck {
    name: &'static str,
    ok: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct ServiceManagerStatus {
    manager: &'static str,
    service_name: String,
    load_state: String,
    active_state: String,
    unit_file_state: String,
    restart_supported: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct EncryptedMessageForwardingStatus {
    count: u64,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct SessionCleanupStatus {
    client_liveness_timeout_seconds: u64,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct InterfaceCounterSnapshot {
    timestamp: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    rx_packets: u64,
    tx_packets: u64,
}

#[derive(Debug, Clone, Serialize)]
struct InterfaceCapacityStatus {
    interface: String,
    rx_bytes: Option<u64>,
    tx_bytes: Option<u64>,
    rx_packets: Option<u64>,
    tx_packets: Option<u64>,
    rx_dropped: Option<u64>,
    tx_dropped: Option<u64>,
    packet_drops: Option<u64>,
    rx_pps: Option<f64>,
    tx_pps: Option<f64>,
    total_pps: Option<f64>,
    rx_bps: Option<f64>,
    tx_bps: Option<f64>,
    total_bps: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ConntrackCapacityStatus {
    used: Option<u64>,
    max: Option<u64>,
    used_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct FileDescriptorCapacityStatus {
    used: Option<u64>,
    soft_limit: Option<u64>,
    hard_limit: Option<u64>,
    used_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DiskPathCapacityStatus {
    reported: bool,
    path: &'static str,
    total_bytes: Option<u64>,
    used_bytes: Option<u64>,
    available_bytes: Option<u64>,
    used_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DiskCapacityStatus {
    root: DiskPathCapacityStatus,
    state: DiskPathCapacityStatus,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CapacityRiskStatus {
    severity: &'static str,
    code: &'static str,
    message: String,
    remediation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    recommended_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recommended_command: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct VpnCapacityStatus {
    virtual_ip_range: String,
    ip_pool_capacity: usize,
    ip_pool_used: usize,
    ip_pool_free: usize,
    max_connections: usize,
    policy_max_sessions: u32,
    active_sessions: usize,
    session_capacity_remaining: Option<u32>,
    bandwidth_limit_mbps: u32,
    bandwidth_limit_bytes_per_second: u64,
    bandwidth_window_bytes: u64,
    bandwidth_window_used_percent: Option<f64>,
    traffic_capacity_status: String,
    conntrack: ConntrackCapacityStatus,
    file_descriptors: FileDescriptorCapacityStatus,
    disk: DiskCapacityStatus,
    interface: InterfaceCapacityStatus,
    packet_drops_total: Option<u64>,
    risks: Vec<CapacityRiskStatus>,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct RecentErrorEvent {
    timestamp: Option<String>,
    severity: &'static str,
    source: &'static str,
    message: String,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct NodeUpgradeStatus {
    reported: bool,
    status: Option<String>,
    step: Option<String>,
    message: Option<String>,
    repo_dir: Option<String>,
    branch: Option<String>,
    service: Option<String>,
    config: Option<String>,
    no_restart: Option<bool>,
    force: Option<bool>,
    updated_at: Option<String>,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct TransportCarrierStatus {
    key: &'static str,
    enabled: bool,
    implemented: bool,
    active: bool,
    endpoint: Option<String>,
    status: &'static str,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct VpnTransportHealthStatus {
    supported_transports: Vec<&'static str>,
    configured_transports: Vec<&'static str>,
    preferred_transport: String,
    effective_transport: &'static str,
    fallback_available: bool,
    udp: TransportCarrierStatus,
    tcp_tls: TransportCarrierStatus,
    websocket_https: TransportCarrierStatus,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct PrivacyProtocolRuntimeStatus {
    active: bool,
    status: &'static str,
    detail: &'static str,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct PrivacyProtocolHealthStatus {
    protocol: &'static str,
    label: &'static str,
    status: &'static str,
    checked_at: u64,
    failed_checks: usize,
    active_sessions: usize,
    active_wallet_devices: usize,
    data_plane: &'static str,
    preferred_transport: String,
    effective_transport: &'static str,
    service_active_state: String,
    protocol_runtime: PrivacyProtocolRuntimeStatus,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct OperatorActionSummary {
    status: &'static str,
    priority: &'static str,
    title: String,
    detail: String,
    next_step: String,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct StartupSelfCheckItem {
    name: &'static str,
    ok: bool,
    severity: &'static str,
    detail: String,
    next_step: String,
}

#[derive(Debug, Clone, Serialize)]
struct StartupSelfCheckStatus {
    status: &'static str,
    ready: bool,
    checked_at: u64,
    failed_checks: usize,
    warning_checks: usize,
    blocking_checks: Vec<&'static str>,
    recommended_action: String,
    checks: Vec<StartupSelfCheckItem>,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct VpnHealthResponse {
    status: &'static str,
    checked_at: u64,
    listen_addr: String,
    gateway_ip: String,
    dns_proxy_enabled: bool,
    dns_owner: &'static str,
    supported_transports: Vec<&'static str>,
    preferred_transport: String,
    transport_health: VpnTransportHealthStatus,
    privacy_protocol_health: PrivacyProtocolHealthStatus,
    startup_self_check: StartupSelfCheckStatus,
    virtual_ip_range: String,
    tun_device: String,
    configured_mtu: u16,
    running_mtu: Option<u16>,
    active_sessions: usize,
    active_wallet_devices: usize,
    service_manager: ServiceManagerStatus,
    node_policy: NodePolicySnapshot,
    policy_enforcement: NodePolicyEnforcementSnapshot,
    placement_readiness: NodePolicyPlacementSnapshot,
    capacity: VpnCapacityStatus,
    packet_runtime: PacketRuntimeStatus,
    discovery_status: Value,
    recent_errors: Vec<RecentErrorEvent>,
    upgrade_status: NodeUpgradeStatus,
    operator_action: OperatorActionSummary,
    voucher_metrics: VoucherMetricsSnapshot,
    encrypted_message_forwarding: EncryptedMessageForwardingStatus,
    session_cleanup: SessionCleanupStatus,
    runtime: RuntimeVersionStatus,
    checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize)]
struct OperatorServiceStatus {
    key: &'static str,
    label: &'static str,
    enabled: bool,
    status: &'static str,
    summary: String,
    metrics: Value,
}

#[derive(Debug, Serialize)]
struct OperatorRisk {
    severity: &'static str,
    code: &'static str,
    message: String,
    remediation: String,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeRolloutStatus {
    executable_path: Option<String>,
    executable_replaced: bool,
    restart_required: bool,
    detail: String,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeVersionStatus {
    version: &'static str,
    git_commit: &'static str,
    build_profile: &'static str,
    build_target: String,
    process_id: u32,
    started_at: u64,
    uptime_seconds: u64,
    rollout: RuntimeRolloutStatus,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct NodeOperatorStatusResponse {
    status: &'static str,
    generated_at: u64,
    runtime_rollout: RuntimeRolloutStatus,
    services: Vec<OperatorServiceStatus>,
    risks: Vec<OperatorRisk>,
    privacy_boundary: &'static str,
}

pub fn build_vpn_health_router(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
) -> Router {
    Router::new()
        .route("/api/vpn/health", get(vpn_health_handler))
        .route(
            "/api/node/operator/status",
            get(node_operator_status_handler),
        )
        .route("/api/operator/status", get(node_operator_status_handler))
        .with_state(VpnHealthState {
            config,
            ip_pool,
            sessions,
            node_policy,
            voucher_verifier,
            encrypted_message_counter,
            packet_handler,
            peer_store,
        })
}

async fn vpn_health_handler(State(state): State<VpnHealthState>) -> impl IntoResponse {
    Json(collect_vpn_health_response(state).await)
}

async fn node_operator_status_handler(State(state): State<VpnHealthState>) -> impl IntoResponse {
    Json(collect_node_operator_status_response(state).await)
}

/// Collect privacy-safe VPN node health as JSON for the CMS heartbeat.
///
/// Source path:
///   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
///
/// The payload contains only local node diagnostics such as UDP listener, TUN,
/// MTU, NAT, DNS stub/query, egress reachability, and aggregate counters. It
/// never includes user destinations, DNS query contents, packet payloads, or
/// browsing history.
pub async fn collect_vpn_health_value(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
) -> Value {
    let state = VpnHealthState {
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
    };
    serde_json::to_value(collect_vpn_health_response(state).await).unwrap_or_else(|e| {
        serde_json::json!({
            "status": "failed",
            "checked_at": unix_now_secs(),
            "checks": [{
                "name": "vpn_health_serialization",
                "ok": false,
                "detail": format!("serialization failed: {}", e),
            }],
        })
    })
}

/// Collect the nodeboard-facing operator service snapshot as privacy-safe JSON.
///
/// Source path:
///   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
///
/// This is reported in heartbeat `system_stats.operator_status` and exposed as
/// `/api/node/operator/status`. It contains aggregate service/config state only.
pub async fn collect_node_operator_status_value(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
) -> Value {
    let state = VpnHealthState {
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
    };
    serde_json::to_value(collect_node_operator_status_response(state).await).unwrap_or_else(|e| {
        serde_json::json!({
            "status": "failed",
            "generated_at": unix_now_secs(),
            "risks": [{
                "severity": "critical",
                "code": "operator_status_serialization",
                "message": format!("operator status serialization failed: {}", e),
                "remediation": "Check Rust node operator status response serialization",
            }],
        })
    })
}

async fn collect_vpn_health_response(state: VpnHealthState) -> VpnHealthResponse {
    let config = state.config;
    let gateway_ip = config.gateway_ip();
    let dns_proxy_enabled = config.dns_proxy_enabled();
    let dns_owner = if dns_proxy_enabled {
        "rust_dns_proxy"
    } else {
        "external_gateway_dns"
    };
    let listen_addr = config.listen_addr();
    let tun_device = config.device_name().to_string();
    let configured_mtu = config.mtu();
    let running_mtu = read_tun_mtu(&tun_device).await.ok();
    let ip_range = config.ip_range().to_string();
    let service_name = resolve_vpn_service_name();
    let service_manager = collect_service_manager_status(&service_name).await;

    let mut checks = Vec::new();
    let udp_listener_check = check_udp_listener(listen_addr).await;
    let transport_health = collect_transport_health(&config, listen_addr, udp_listener_check.ok);
    checks.push(udp_listener_check);
    checks.push(check_tun_device(&tun_device).await);
    checks.push(check_mtu_config(&tun_device, configured_mtu, running_mtu).await);
    checks.push(check_ip_forwarding().await);
    checks.push(check_nat_masquerade(&ip_range).await);
    checks.push(check_dns_socket(gateway_ip).await);
    checks.push(check_dns_query(gateway_ip).await);
    checks.push(check_internet_egress().await);

    let failed = checks.iter().filter(|c| !c.ok).count();
    let status = if failed == 0 {
        "ok"
    } else if failed <= 2 {
        "degraded"
    } else {
        "failed"
    };

    let active_sessions = state.sessions.count();
    let active_wallet_devices = state.sessions.wallet_index_count();
    let node_policy = state.node_policy.snapshot();
    let policy_enforcement = state.node_policy.enforcement_snapshot();
    let placement_readiness = state.node_policy.placement_snapshot(active_sessions);
    let capacity = collect_capacity_status(
        &config,
        &state.ip_pool,
        &node_policy,
        &policy_enforcement,
        &placement_readiness,
        active_sessions,
    )
    .await;
    let packet_runtime = state.packet_handler.runtime_status();
    let discovery_status = collect_discovery_status_value(&state.peer_store);
    let runtime = collect_runtime_version_status().await;
    let recent_errors = collect_recent_error_events(&service_name).await;
    let upgrade_status = collect_upgrade_status().await;
    let startup_self_check = collect_startup_self_check(
        &config,
        &checks,
        &service_manager,
        &transport_health,
        &capacity,
        &discovery_status,
    );
    let operator_action = collect_operator_action_summary(
        status,
        &checks,
        &capacity,
        &upgrade_status,
        &service_manager,
    );
    let checked_at = unix_now_secs();
    let privacy_protocol_health = collect_privacy_protocol_health(
        status,
        checked_at,
        failed,
        active_sessions,
        active_wallet_devices,
        &transport_health,
        &service_manager,
    );

    VpnHealthResponse {
        status,
        checked_at,
        listen_addr: listen_addr.to_string(),
        gateway_ip: gateway_ip.to_string(),
        dns_proxy_enabled,
        dns_owner,
        supported_transports: transport_health.supported_transports.clone(),
        preferred_transport: transport_health.preferred_transport.clone(),
        transport_health,
        privacy_protocol_health,
        startup_self_check,
        virtual_ip_range: ip_range,
        tun_device,
        configured_mtu,
        running_mtu,
        active_sessions,
        active_wallet_devices,
        service_manager,
        node_policy,
        policy_enforcement,
        placement_readiness,
        capacity,
        packet_runtime,
        discovery_status,
        recent_errors,
        upgrade_status,
        operator_action,
        voucher_metrics: state.voucher_verifier.metrics_snapshot(),
        encrypted_message_forwarding: EncryptedMessageForwardingStatus {
            count: state.encrypted_message_counter.load(Ordering::Relaxed),
            source: "packet_handler_successful_vpn_data_packets",
            privacy_boundary: concat!(
                "aggregate count only; no destinations, DNS contents, packet ",
                "payloads, domains, URLs, browsing history, voucher secrets, ",
                "client public IPs, or wallet-level traffic"
            ),
        },
        session_cleanup: SessionCleanupStatus {
            client_liveness_timeout_seconds: CLIENT_LIVENESS_TIMEOUT_SECS,
            source: "session_client_activity_timeout",
            privacy_boundary: concat!(
                "local monotonic timeout metadata only; no destinations, DNS ",
                "contents, packet payloads, domains, URLs, browsing history, ",
                "voucher secrets, client public IPs, or wallet-level traffic"
            ),
        },
        runtime,
        checks,
    }
}

fn collect_discovery_status_value(peer_store: &PeerStore) -> Value {
    let now = unix_now_secs();
    let status = peer_store.status(now);
    serde_json::json!({
        "generated_at": now,
        "peer_store": status,
        "source": "rust_peer_store",
        "privacy_boundary": concat!(
            "aggregate node discovery counters only; no client IPs, ",
            "destinations, DNS contents, packet payloads, chat plaintext, ",
            "voucher secrets, private keys, peer private keys, or wallet-level traffic"
        )
    })
}

fn collect_startup_self_check(
    config: &ServerConfig,
    health_checks: &[HealthCheck],
    service_manager: &ServiceManagerStatus,
    transport_health: &VpnTransportHealthStatus,
    capacity: &VpnCapacityStatus,
    discovery_status: &Value,
) -> StartupSelfCheckStatus {
    let mut checks = Vec::new();

    match config.validate() {
        Ok(()) => checks.push(startup_item(
            "config_schema",
            true,
            "critical",
            "ServerConfig validation passed at runtime".to_string(),
            "No action required.".to_string(),
        )),
        Err(error) => checks.push(startup_item(
            "config_schema",
            false,
            "critical",
            format!("ServerConfig validation failed: {error}"),
            "Fix /etc/aeronyx/server.toml, then run deploy/node/aeronyx-node.sh health --json before restart.".to_string(),
        )),
    }

    checks.push(startup_item(
        "service_manager_active",
        service_manager.active_state == "active",
        "critical",
        service_manager.detail.clone(),
        "Run deploy/node/aeronyx-node.sh status and logs --lines 200; restart only after confirming active sessions and maintenance mode.".to_string(),
    ));

    push_runtime_health_check(
        &mut checks,
        health_checks,
        "udp_listener",
        "critical",
        "Ensure UDP listen_addr is reachable and no other process owns the configured port, then restart the AeroNyx Rust node.".to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "tun_device",
        "critical",
        "Run deploy/node/aeronyx-node.sh network to recreate the TUN device and routing rules."
            .to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "mtu_config",
        "warning",
        "Align tun.mtu with the running interface MTU during a maintenance window.".to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "ip_forward",
        "critical",
        "Enable net.ipv4.ip_forward=1, then rerun deploy/node/aeronyx-node.sh network.".to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "nat_masquerade",
        "critical",
        "Restore the VPN MASQUERADE rule with deploy/node/aeronyx-node.sh network.".to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "dns_stub",
        "critical",
        "Start the built-in DNS proxy or provide an external listener on gateway_ip:53."
            .to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "dns_query",
        "warning",
        "Check upstream DNS reachability from the node and gateway DNS listener health."
            .to_string(),
    );
    push_runtime_health_check(
        &mut checks,
        health_checks,
        "internet_egress",
        "critical",
        "Verify host firewall, cloud security group, and default route before accepting traffic."
            .to_string(),
    );

    checks.push(startup_item(
        "effective_transport",
        transport_health.udp.active && transport_health.effective_transport == "udp",
        "critical",
        format!(
            "effective_transport={} udp_active={} fallback_available={}",
            transport_health.effective_transport,
            transport_health.udp.active,
            transport_health.fallback_available
        ),
        "Keep UDP active until TCP/TLS or WebSocket HTTPS fallback listeners are implemented."
            .to_string(),
    ));

    checks.push(startup_item(
        "ip_pool_capacity",
        capacity.ip_pool_free > 0 && capacity.max_connections <= capacity.ip_pool_capacity,
        "warning",
        format!(
            "ip_pool_capacity={} used={} free={} max_connections={}",
            capacity.ip_pool_capacity,
            capacity.ip_pool_used,
            capacity.ip_pool_free,
            capacity.max_connections
        ),
        "Expand vpn.virtual_ip_range or lower limits.max_connections before commercial traffic exceeds the IP pool.".to_string(),
    ));

    let discovery_enabled = config.discovery.enabled;
    let peer_cache_configured = discovery_bool(
        discovery_status,
        &["peer_store", "bootstrap", "peer_cache_configured"],
    )
    .unwrap_or(config.discovery.peer_cache_path.is_some());
    let gossip_enabled = discovery_bool(
        discovery_status,
        &["peer_store", "bootstrap", "gossip_enabled"],
    )
    .unwrap_or(config.discovery.gossip_enabled);
    let seed_count = discovery_u64(
        discovery_status,
        &["peer_store", "bootstrap", "seed_endpoints_configured"],
    )
    .unwrap_or(config.discovery.seed_endpoints.len() as u64);
    let stability_health = discovery_str(discovery_status, &["peer_store", "stability", "health"])
        .unwrap_or("unknown");
    let restart_recovery_configured = discovery_bool(
        discovery_status,
        &["peer_store", "stability", "restart_recovery_configured"],
    )
    .unwrap_or(false);

    checks.push(startup_item(
        "peer_store_restart_recovery",
        !discovery_enabled || (peer_cache_configured && restart_recovery_configured),
        if discovery_enabled { "critical" } else { "warning" },
        format!(
            "discovery_enabled={} peer_cache_configured={} restart_recovery_configured={}",
            discovery_enabled, peer_cache_configured, restart_recovery_configured
        ),
        "Set discovery.peer_cache_path so verified peers survive restart without depending on the center service.".to_string(),
    ));

    checks.push(startup_item(
        "discovery_seed_recovery",
        !gossip_enabled || seed_count > 0,
        if gossip_enabled { "warning" } else { "info" },
        format!(
            "gossip_enabled={} seed_endpoints_configured={}",
            gossip_enabled, seed_count
        ),
        "Configure at least one discovery.seed_endpoints entry for live peer recovery.".to_string(),
    ));

    checks.push(startup_item(
        "public_discovery_api",
        !discovery_enabled || config.discovery.public_api_listen_addr.is_some(),
        if discovery_enabled { "warning" } else { "info" },
        format!(
            "discovery_enabled={} public_api_listen_addr_configured={}",
            discovery_enabled,
            config.discovery.public_api_listen_addr.is_some()
        ),
        "Set discovery.public_api_listen_addr when this node should be discoverable by other AeroNyx nodes.".to_string(),
    ));

    checks.push(startup_item(
        "peer_store_stability",
        !matches!(stability_health, "failed" | "stale"),
        "warning",
        format!("peer_store_stability={stability_health}"),
        "Wait for live gossip recovery or refresh the peer cache/bootstrap snapshot before enabling multi-hop routing.".to_string(),
    ));

    let failed_checks = checks
        .iter()
        .filter(|check| !check.ok && check.severity == "critical")
        .count();
    let warning_checks = checks
        .iter()
        .filter(|check| !check.ok && check.severity != "critical")
        .count();
    let blocking_checks = checks
        .iter()
        .filter(|check| !check.ok && check.severity == "critical")
        .map(|check| check.name)
        .collect::<Vec<_>>();
    let status = if failed_checks > 0 {
        "failed"
    } else if warning_checks > 0 {
        "degraded"
    } else {
        "ready"
    };
    let recommended_action = checks
        .iter()
        .find(|check| !check.ok)
        .map(|check| check.next_step.clone())
        .unwrap_or_else(|| "No action required; startup self-check is ready.".to_string());

    StartupSelfCheckStatus {
        status,
        ready: status == "ready",
        checked_at: unix_now_secs(),
        failed_checks,
        warning_checks,
        blocking_checks,
        recommended_action,
        checks,
        source: "rust_startup_self_check",
        privacy_boundary: concat!(
            "aggregate startup/config readiness only; no client public IPs, ",
            "destinations, DNS query names, packet payloads, chat plaintext, ",
            "ciphertext, voucher secrets, private keys, or wallet-level traffic"
        ),
    }
}

fn startup_item(
    name: &'static str,
    ok: bool,
    severity: &'static str,
    detail: String,
    next_step: String,
) -> StartupSelfCheckItem {
    StartupSelfCheckItem {
        name,
        ok,
        severity,
        detail,
        next_step,
    }
}

fn push_runtime_health_check(
    checks: &mut Vec<StartupSelfCheckItem>,
    health_checks: &[HealthCheck],
    name: &'static str,
    severity: &'static str,
    next_step: String,
) {
    match health_checks.iter().find(|check| check.name == name) {
        Some(check) => checks.push(startup_item(
            name,
            check.ok,
            severity,
            check.detail.clone(),
            next_step,
        )),
        None => checks.push(startup_item(
            name,
            false,
            severity,
            "runtime health check did not run".to_string(),
            next_step,
        )),
    }
}

fn discovery_value<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn discovery_bool(value: &Value, path: &[&str]) -> Option<bool> {
    discovery_value(value, path).and_then(Value::as_bool)
}

fn discovery_u64(value: &Value, path: &[&str]) -> Option<u64> {
    discovery_value(value, path).and_then(Value::as_u64)
}

fn discovery_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    discovery_value(value, path).and_then(Value::as_str)
}

async fn collect_node_operator_status_response(
    state: VpnHealthState,
) -> NodeOperatorStatusResponse {
    let generated_at = unix_now_secs();
    let vpn_health = collect_vpn_health_response(state.clone()).await;
    let config = state.config.clone();
    let memchain_enabled = config.memchain.is_enabled();
    let remote_storage_enabled = config.memchain.is_remote_storage_enabled();
    let chat_relay_enabled = config.memchain.is_chat_relay_enabled();
    let supernode_enabled = config.memchain.is_supernode_enabled();
    let api_secret_configured = config.memchain.effective_api_secret().is_some();
    let encrypted_messages = state.encrypted_message_counter.load(Ordering::Relaxed);
    let runtime_rollout = collect_runtime_rollout_status().await;
    let runtime_version = collect_runtime_version_status_with_rollout(runtime_rollout.clone());
    let mut services = Vec::new();
    let mut risks = Vec::new();

    services.push(OperatorServiceStatus {
        key: "privacy_protocol",
        label: "AeroNyx Privacy Protocol",
        enabled: true,
        status: vpn_health.status,
        summary: format!(
            "{} active sessions, {} wallet devices, {} encrypted packets forwarded",
            vpn_health.active_sessions, vpn_health.active_wallet_devices, encrypted_messages
        ),
        metrics: serde_json::json!({
            "listen_addr": vpn_health.listen_addr,
            "gateway_ip": vpn_health.gateway_ip,
            "dns_proxy_enabled": vpn_health.dns_proxy_enabled,
            "dns_owner": vpn_health.dns_owner,
            "supported_transports": vpn_health.supported_transports.clone(),
            "preferred_transport": vpn_health.preferred_transport.clone(),
            "transport_health": vpn_health.transport_health.clone(),
            "privacy_protocol_health": vpn_health.privacy_protocol_health.clone(),
            "startup_self_check": vpn_health.startup_self_check.clone(),
            "virtual_ip_range": vpn_health.virtual_ip_range,
            "tun_device": vpn_health.tun_device,
            "configured_mtu": vpn_health.configured_mtu,
            "running_mtu": vpn_health.running_mtu,
            "active_sessions": vpn_health.active_sessions,
            "active_wallet_devices": vpn_health.active_wallet_devices,
            "service_manager": vpn_health.service_manager,
            "encrypted_message_forwarding": vpn_health.encrypted_message_forwarding,
            "session_cleanup": vpn_health.session_cleanup,
            "runtime": runtime_version.clone(),
            "placement_readiness": vpn_health.placement_readiness,
            "capacity": vpn_health.capacity,
            "packet_runtime": vpn_health.packet_runtime,
            "discovery_status": vpn_health.discovery_status,
            "recent_errors": vpn_health.recent_errors,
            "upgrade_status": vpn_health.upgrade_status.clone(),
            "operator_action": vpn_health.operator_action.clone(),
            "failed_checks": vpn_health.checks.iter().filter(|check| !check.ok).count(),
            "runtime_rollout": runtime_rollout.clone(),
        }),
    });

    if vpn_health.startup_self_check.status != "ready" {
        risks.push(OperatorRisk {
            severity: if vpn_health.startup_self_check.status == "failed" {
                "critical"
            } else {
                "warning"
            },
            code: "startup_self_check",
            message: format!(
                "Startup self-check is {} with {} blocking checks",
                vpn_health.startup_self_check.status,
                vpn_health.startup_self_check.blocking_checks.len()
            ),
            remediation: vpn_health.startup_self_check.recommended_action.clone(),
        });
    }

    services.push(OperatorServiceStatus {
        key: "memchain",
        label: "MemChain / MPI",
        enabled: memchain_enabled,
        status: if memchain_enabled { "ok" } else { "disabled" },
        summary: if memchain_enabled {
            format!(
                "mode={:?}, API bound at {}",
                config.memchain.mode, config.memchain.api_listen_addr
            )
        } else {
            "MemChain is disabled in this node config".to_string()
        },
        metrics: serde_json::json!({
            "mode": format!("{:?}", config.memchain.mode),
            "api_listen_addr": config.memchain.api_listen_addr.to_string(),
            "db_path": config.memchain.db_path,
            "aof_path": config.memchain.aof_path,
            "api_secret_configured": api_secret_configured,
            "ner_enabled": config.memchain.ner_enabled,
            "graph_enabled": config.memchain.graph_enabled,
            "reranker_enabled": config.memchain.reranker_enabled,
            "remote_storage_enabled": remote_storage_enabled,
            "max_remote_owners": config.memchain.max_remote_owners,
        }),
    });

    services.push(OperatorServiceStatus {
        key: "chat_relay",
        label: "Zero-Knowledge Chat Relay",
        enabled: chat_relay_enabled,
        status: if chat_relay_enabled { "ok" } else { "disabled" },
        summary: if chat_relay_enabled {
            format!(
                "offline TTL {}s, {} pending messages per wallet, max blob {} bytes",
                config.memchain.chat_relay.offline_ttl_secs,
                config.memchain.chat_relay.max_pending_per_wallet,
                config.memchain.chat_relay.max_blob_size
            )
        } else {
            "Chat relay is disabled; encrypted messages are not stored for offline delivery"
                .to_string()
        },
        metrics: serde_json::json!({
            "offline_ttl_secs": config.memchain.chat_relay.offline_ttl_secs,
            "max_pending_per_wallet": config.memchain.chat_relay.max_pending_per_wallet,
            "db_path": config.memchain.chat_relay.db_path,
            "max_message_size": config.memchain.chat_relay.max_message_size,
            "max_blob_size": config.memchain.chat_relay.max_blob_size,
            "max_blobs_per_receiver": config.memchain.chat_relay.max_blobs_per_receiver,
            "cleanup_interval_secs": config.memchain.chat_relay.cleanup_interval_secs,
        }),
    });

    services.push(OperatorServiceStatus {
        key: "sovereign_data_layer",
        label: "Sovereign Data Layer",
        enabled: remote_storage_enabled,
        status: if remote_storage_enabled {
            "ready"
        } else {
            "planned"
        },
        summary: if remote_storage_enabled {
            format!(
                "remote encrypted owner storage enabled for up to {} owners",
                config.memchain.max_remote_owners
            )
        } else {
            "Encrypted user-owned record RPC is not enabled on this node yet".to_string()
        },
        metrics: serde_json::json!({
            "remote_storage_enabled": remote_storage_enabled,
            "max_remote_owners": config.memchain.max_remote_owners,
            "current_protocol_basis": [
                "MemoryRecord.owner",
                "MemoryRecord.encrypted_content",
                "MemoryRecord.signature",
                "MemChainMessage::SyncRecordRequest",
                "MemChainMessage::SyncRecordResponse"
            ],
            "settlement_layer": "ethereum",
            "private_data_on_ethereum": false,
        }),
    });

    services.push(OperatorServiceStatus {
        key: "supernode",
        label: "SuperNode Cognitive Worker",
        enabled: supernode_enabled,
        status: if supernode_enabled {
            "ready"
        } else {
            "disabled"
        },
        summary: if supernode_enabled {
            format!(
                "{} configured provider(s)",
                config.memchain.supernode.providers.len()
            )
        } else {
            "SuperNode LLM worker is disabled".to_string()
        },
        metrics: serde_json::json!({
            "providers": config.memchain.supernode.providers.len(),
            "worker_poll_interval_secs": config.memchain.supernode.worker.poll_interval_secs,
            "worker_max_concurrent": config.memchain.supernode.worker.max_concurrent,
        }),
    });

    if vpn_health.status != "ok" {
        risks.push(OperatorRisk {
            severity: if vpn_health.status == "failed" { "critical" } else { "warning" },
            code: "privacy_protocol_health",
            message: format!("AeroNyx privacy protocol health is {}", vpn_health.status),
            remediation: "Open /api/vpn/health and resolve failed checks before advertising this node as healthy".to_string(),
        });
    }

    if runtime_rollout.restart_required {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "runtime_restart_required",
            message: "Rust process is running an executable that has been replaced on disk".to_string(),
            remediation: "Drain active sessions, enter maintenance mode, then restart the AeroNyx Rust node so the staged binary takes effect".to_string(),
        });
    }

    if vpn_health.upgrade_status.status.as_deref() == Some("failed") {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "upgrade_workflow_failed",
            message: vpn_health
                .upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "Last Rust upgrade workflow failed".to_string()),
            remediation: "Open node detail, inspect upgrade status, run healthcheck, then rerun deploy/node/aeronyx-node.sh upgrade --no-restart after resolving the failed step".to_string(),
        });
    }

    if !api_secret_configured {
        risks.push(OperatorRisk {
            severity: if remote_storage_enabled {
                "critical"
            } else {
                "warning"
            },
            code: "mpi_api_secret_missing",
            message: "MemChain API secret is not configured".to_string(),
            remediation:
                "Set memchain.api_secret before enabling remote RPC or remote encrypted storage"
                    .to_string(),
        });
    }

    if !chat_relay_enabled {
        risks.push(OperatorRisk {
            severity: "info",
            code: "chat_relay_disabled",
            message: "Zero-knowledge chat relay is disabled".to_string(),
            remediation: "Enable [memchain.chat_relay] when this node should store encrypted offline messages and blobs".to_string(),
        });
    }

    if !remote_storage_enabled {
        risks.push(OperatorRisk {
            severity: "info",
            code: "sovereign_data_layer_not_enabled",
            message: "Sovereign Data Layer remote storage is not enabled".to_string(),
            remediation: "Enable memchain.allow_remote_storage after encrypted record limits and API authentication are ready".to_string(),
        });
    }

    if remote_storage_enabled && config.memchain.max_remote_owners == 0 {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "remote_owner_capacity_unbounded",
            message: "Remote encrypted storage owner capacity is unlimited".to_string(),
            remediation: "Set memchain.max_remote_owners for commercial node capacity planning"
                .to_string(),
        });
    }

    let has_critical = risks.iter().any(|risk| risk.severity == "critical");
    let has_warning = risks.iter().any(|risk| risk.severity == "warning");
    let status = if has_critical {
        "critical"
    } else if vpn_health.status == "failed" {
        "failed"
    } else if has_warning || vpn_health.status == "degraded" {
        "attention"
    } else {
        "ok"
    };

    NodeOperatorStatusResponse {
        status,
        generated_at,
        runtime_rollout,
        services,
        risks,
        privacy_boundary: concat!(
            "operator status contains aggregate service health and configuration ",
            "only; no user plaintext, social graph plaintext, destinations, DNS ",
            "contents, packet payloads, domains, URLs, browsing history, voucher ",
            "secrets, or wallet-level traffic"
        ),
    }
}

fn collect_operator_action_summary(
    status: &'static str,
    checks: &[HealthCheck],
    capacity: &VpnCapacityStatus,
    upgrade_status: &NodeUpgradeStatus,
    service_manager: &ServiceManagerStatus,
) -> OperatorActionSummary {
    let privacy_boundary = concat!(
        "operator action is derived from aggregate node operations metadata ",
        "only; no client public IPs, destinations, DNS contents, packet ",
        "payloads, domains, URLs, browsing history, voucher secrets, chat ",
        "plaintext, private keys, or wallet-level traffic"
    );

    if let Some(check) = checks.iter().find(|check| !check.ok) {
        return OperatorActionSummary {
            status: if status == "failed" { "critical" } else { "warning" },
            priority: "fix_failed_health_check",
            title: "AeroNyx privacy protocol check failed".to_string(),
            detail: format!("{}: {}", check.name, check.detail),
            next_step: "Open node detail health checks, fix the failed check, then rerun deploy/node/aeronyx-node.sh health --json.".to_string(),
            source: "rust_vpn_health.checks",
            privacy_boundary,
        };
    }

    if upgrade_status.status.as_deref() == Some("failed") {
        return OperatorActionSummary {
            status: "critical",
            priority: "fix_failed_upgrade",
            title: "Rust upgrade workflow failed".to_string(),
            detail: upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "Last upgrade workflow reported failure.".to_string()),
            next_step: "Inspect upgrade_status, resolve the failed step, then rerun deploy/node/aeronyx-node.sh upgrade --no-restart before a controlled restart.".to_string(),
            source: "rust_vpn_health.upgrade_status",
            privacy_boundary,
        };
    }

    if service_manager.active_state != "active" {
        return OperatorActionSummary {
            status: "warning",
            priority: "service_not_active",
            title: "Rust service is not active".to_string(),
            detail: service_manager.detail.clone(),
            next_step: "Run deploy/node/aeronyx-node.sh status and logs --lines 200 before restart or upgrade actions.".to_string(),
            source: "rust_vpn_health.service_manager",
            privacy_boundary,
        };
    }

    if let Some(risk) = capacity
        .risks
        .iter()
        .find(|risk| risk.severity == "critical")
    {
        return OperatorActionSummary {
            status: "critical",
            priority: risk.code,
            title: "Capacity blocks commercial placement".to_string(),
            detail: risk.message.clone(),
            next_step: risk.remediation.clone(),
            source: "rust_vpn_health.capacity.risks",
            privacy_boundary,
        };
    }

    if let Some(risk) = capacity
        .risks
        .iter()
        .find(|risk| risk.severity == "warning")
    {
        return OperatorActionSummary {
            status: "warning",
            priority: risk.code,
            title: "Capacity needs operator review".to_string(),
            detail: risk.message.clone(),
            next_step: risk.remediation.clone(),
            source: "rust_vpn_health.capacity.risks",
            privacy_boundary,
        };
    }

    if status == "degraded" {
        return OperatorActionSummary {
            status: "warning",
            priority: "privacy_protocol_degraded",
            title: "AeroNyx privacy protocol is degraded".to_string(),
            detail: "Local health checks passed enough to stay online, but the node is not fully clean.".to_string(),
            next_step: "Review node detail health checks, capacity, recent events, and network rules before accepting more traffic.".to_string(),
            source: "rust_vpn_health.status",
            privacy_boundary,
        };
    }

    if upgrade_status.status.as_deref() == Some("staged") {
        return OperatorActionSummary {
            status: "info",
            priority: "upgrade_staged",
            title: "Rust upgrade is staged".to_string(),
            detail: upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "A new Rust build is staged without restart.".to_string()),
            next_step: "Use maintenance mode, drain active sessions, then perform a controlled restart when ready.".to_string(),
            source: "rust_vpn_health.upgrade_status",
            privacy_boundary,
        };
    }

    OperatorActionSummary {
        status: "ok",
        priority: "monitor",
        title: "Node is ready for monitoring".to_string(),
        detail: "No failed health checks or capacity blockers are reported.".to_string(),
        next_step: "Keep monitoring heartbeat freshness, capacity, traffic, and recent events in nodeboard.".to_string(),
        source: "rust_vpn_health",
        privacy_boundary,
    }
}

fn collect_transport_health(
    config: &ServerConfig,
    udp_listen_addr: SocketAddr,
    udp_listener_ok: bool,
) -> VpnTransportHealthStatus {
    let transports = config.vpn_transports();
    let mut configured_transports = Vec::new();
    if transports.udp_enabled {
        configured_transports.push("udp");
    }
    if transports.tcp_tls_enabled {
        configured_transports.push("tcp_tls");
    }
    if transports.websocket_enabled {
        configured_transports.push("websocket_https");
    }

    // Phase 1 reports actual production support, not desired future config.
    // This keeps nodeboard and public stats from advertising a fallback carrier
    // before its server listener and client implementation exist.
    let supported_transports = if transports.udp_enabled {
        vec!["udp"]
    } else {
        Vec::new()
    };

    let udp = TransportCarrierStatus {
        key: "udp",
        enabled: transports.udp_enabled,
        implemented: true,
        active: transports.udp_enabled && udp_listener_ok,
        endpoint: Some(udp_listen_addr.to_string()),
        status: if transports.udp_enabled && udp_listener_ok {
            "active"
        } else {
            "degraded"
        },
        detail: if transports.udp_enabled && udp_listener_ok {
            format!("UDP data-plane listener is active at {}", udp_listen_addr)
        } else {
            format!(
                "UDP is configured but listener check failed at {}",
                udp_listen_addr
            )
        },
    };

    let tcp_tls = TransportCarrierStatus {
        key: "tcp_tls",
        enabled: transports.tcp_tls_enabled,
        implemented: false,
        active: false,
        endpoint: transports.tcp_tls_public_endpoint.clone(),
        status: if transports.tcp_tls_enabled {
            "configured_not_active"
        } else {
            "planned"
        },
        detail: if transports.tcp_tls_enabled {
            "TCP/TLS fallback is configured in metadata but its Rust data-plane listener is not implemented yet".to_string()
        } else {
            "TCP/TLS fallback is planned but disabled on this node".to_string()
        },
    };

    let websocket_https = TransportCarrierStatus {
        key: "websocket_https",
        enabled: transports.websocket_enabled,
        implemented: false,
        active: false,
        endpoint: transports.websocket_public_url.clone(),
        status: if transports.websocket_enabled {
            "configured_not_active"
        } else {
            "planned"
        },
        detail: if transports.websocket_enabled {
            "WebSocket HTTPS fallback is configured in metadata but its Rust data-plane listener is not implemented yet".to_string()
        } else {
            "WebSocket HTTPS fallback is planned but disabled on this node".to_string()
        },
    };

    VpnTransportHealthStatus {
        supported_transports,
        configured_transports,
        preferred_transport: transports.preferred_transport.clone(),
        effective_transport: "udp",
        fallback_available: false,
        udp,
        tcp_tls,
        websocket_https,
        source: "rust_vpn_transport_capability_metadata",
        privacy_boundary: concat!(
            "transport capability metadata only; no packet payloads, DNS ",
            "contents, destinations, domains, URLs, browsing history, voucher ",
            "secrets, client public IPs, or wallet-level traffic"
        ),
    }
}

fn collect_privacy_protocol_health(
    status: &'static str,
    checked_at: u64,
    failed_checks: usize,
    active_sessions: usize,
    active_wallet_devices: usize,
    transport_health: &VpnTransportHealthStatus,
    service_manager: &ServiceManagerStatus,
) -> PrivacyProtocolHealthStatus {
    PrivacyProtocolHealthStatus {
        protocol: "aeronyx_privacy_protocol",
        label: "AeroNyx Privacy Protocol",
        status,
        checked_at,
        failed_checks,
        active_sessions,
        active_wallet_devices,
        data_plane: "aeronyx_privacy_protocol",
        preferred_transport: transport_health.preferred_transport.clone(),
        effective_transport: transport_health.effective_transport,
        service_active_state: service_manager.active_state.clone(),
        protocol_runtime: PrivacyProtocolRuntimeStatus {
            active: status != "failed",
            status,
            detail: "This Rust node reports AeroNyx privacy protocol runtime health from aggregate service, transport, and routing checks.",
            source: "rust_vpn_health.protocol_model",
            privacy_boundary: concat!(
                "protocol model metadata only; no client public IPs, ",
                "destinations, DNS contents, packet payloads, domains, URLs, ",
                "browsing history, voucher secrets, chat plaintext, or ",
                "wallet-level traffic"
            ),
        },
        source: "rust_vpn_health.privacy_protocol_health",
        privacy_boundary: concat!(
            "aggregate privacy protocol operations metadata only; no client ",
            "public IPs, destinations, DNS contents, packet payloads, domains, ",
            "URLs, browsing history, voucher secrets, chat plaintext, private ",
            "keys, or wallet-level traffic"
        ),
    }
}

async fn collect_runtime_rollout_status() -> RuntimeRolloutStatus {
    // Return a privacy-safe rollout signal for nodeboard.
    //
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // Linux keeps a running process alive even after its executable file is
    // replaced on disk. `/proc/self/exe` then points to a path suffixed with
    // `(deleted)`. That is a strong operator signal that a new binary may be
    // staged but the node still needs a controlled maintenance drain/restart.
    //
    // This reports process metadata only: executable path state and whether a
    // restart is required. It never includes user destinations, DNS contents,
    // packet payloads, browsing history, voucher secrets, or wallet traffic.
    let proc_exe = tokio::fs::read_link("/proc/self/exe").await.ok();
    let fallback_exe = if proc_exe.is_none() {
        std::env::current_exe().ok()
    } else {
        None
    };
    let executable_path = proc_exe
        .or(fallback_exe)
        .map(|path| path.to_string_lossy().chars().take(512).collect::<String>());
    let executable_replaced = executable_path
        .as_deref()
        .map(|path| path.contains(" (deleted)") || path.ends_with("(deleted)"))
        .unwrap_or(false);

    RuntimeRolloutStatus {
        executable_path,
        executable_replaced,
        restart_required: executable_replaced,
        detail: if executable_replaced {
            "Running process executable has been replaced on disk; restart after draining active sessions".to_string()
        } else {
            "Running process executable is active; no rollout restart signal detected".to_string()
        },
        source: "/proc/self/exe",
        privacy_boundary: concat!(
            "runtime process executable metadata only; no destinations, DNS ",
            "contents, packet payloads, domains, URLs, browsing history, ",
            "voucher secrets, client public IPs, or wallet-level traffic"
        ),
    }
}

async fn collect_runtime_version_status() -> RuntimeVersionStatus {
    let rollout = collect_runtime_rollout_status().await;
    collect_runtime_version_status_with_rollout(rollout)
}

fn collect_runtime_version_status_with_rollout(
    rollout: RuntimeRolloutStatus,
) -> RuntimeVersionStatus {
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // This is node runtime metadata for commercial operations. It helps
    // nodeboard distinguish "online but old binary" from an actually upgraded
    // node. It intentionally excludes client identifiers, destinations, DNS
    // contents, packet payloads, domains, URLs, browsing history, voucher
    // secrets, and wallet-level traffic.
    let started_at = *RUNTIME_STARTED_AT.get_or_init(unix_now_secs);
    let now = unix_now_secs();
    RuntimeVersionStatus {
        version: integrity::get_version(),
        git_commit: build_git_commit(),
        build_profile: build_profile(),
        build_target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        process_id: std::process::id(),
        started_at,
        uptime_seconds: now.saturating_sub(started_at),
        rollout,
        source: "rust_process_runtime_metadata",
        privacy_boundary: concat!(
            "runtime build/process metadata only; no client public IPs, ",
            "destinations, DNS contents, packet payloads, domains, URLs, ",
            "browsing history, voucher secrets, or wallet-level traffic"
        ),
    }
}

async fn collect_upgrade_status() -> NodeUpgradeStatus {
    // File creation/modification notes:
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    // Local producer:
    //   /root/open/AeroNyx/deploy/node/upgrade.sh
    // Backend consumer:
    //   /root/aeronyx/privacy_network/api/vpn_observability.py
    // Nodeboard consumer:
    //   /root/open/nodeboard/app/dashboard/nodes/[id]/page.tsx
    //
    // Main logical flow:
    // 1. Read `/var/lib/aeronyx/upgrade-status.json` when present.
    // 2. Allow-list only operator workflow fields used by nodeboard.
    // 3. Return `reported=false` for missing or invalid files so old nodes
    //    remain backward compatible.
    //
    // Important note for next developer:
    // - Do not forward arbitrary JSON from the local status file. Keep this
    //   allow-list tight. The heartbeat is signed node telemetry and must
    //   never contain registration codes, private keys, client public IPs,
    //   destinations, DNS contents, packet payloads, chat plaintext, voucher
    //   secrets, or wallet-level traffic.
    let privacy_boundary = concat!(
        "upgrade workflow metadata only; no registration codes, private keys, ",
        "client public IPs, destinations, DNS contents, packet payloads, chat ",
        "plaintext, voucher secrets, or wallet-level traffic"
    );

    let Ok(raw) = tokio::fs::read_to_string(UPGRADE_STATUS_FILE).await else {
        return NodeUpgradeStatus {
            reported: false,
            status: None,
            step: None,
            message: None,
            repo_dir: None,
            branch: None,
            service: None,
            config: None,
            no_restart: None,
            force: None,
            updated_at: None,
            source: UPGRADE_STATUS_FILE,
            privacy_boundary,
        };
    };

    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return NodeUpgradeStatus {
            reported: false,
            status: Some("unreadable".to_string()),
            step: None,
            message: Some("Local upgrade status file is not valid JSON".to_string()),
            repo_dir: None,
            branch: None,
            service: None,
            config: None,
            no_restart: None,
            force: None,
            updated_at: None,
            source: UPGRADE_STATUS_FILE,
            privacy_boundary,
        };
    };

    let string_field = |key: &str, max_len: usize| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(|text| sanitize_status_text(text, max_len))
            .filter(|text| !text.is_empty())
    };

    NodeUpgradeStatus {
        reported: true,
        status: string_field("status", 32),
        step: string_field("step", 64),
        message: string_field("message", 240),
        repo_dir: string_field("repo_dir", 256),
        branch: string_field("branch", 80),
        service: string_field("service", 80),
        config: string_field("config", 256),
        no_restart: value.get("no_restart").and_then(Value::as_bool),
        force: value.get("force").and_then(Value::as_bool),
        updated_at: string_field("updated_at", 80),
        source: UPGRADE_STATUS_FILE,
        privacy_boundary,
    }
}

fn sanitize_status_text(value: &str, max_len: usize) -> String {
    value
        .replace('\0', "")
        .chars()
        .take(max_len)
        .collect::<String>()
        .trim()
        .to_string()
}

fn build_git_commit() -> &'static str {
    option_env!("AERONYX_GIT_COMMIT")
        .or(option_env!("GIT_COMMIT"))
        .or(option_env!("VERGEN_GIT_SHA"))
        .unwrap_or("unknown")
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn resolve_vpn_service_name() -> String {
    let operator_override = std::env::var(VPN_SERVICE_NAME_ENV).ok();
    let process_cgroup = std::fs::read_to_string(PROC_SELF_CGROUP_PATH).ok();
    resolve_vpn_service_name_from(operator_override.as_deref(), process_cgroup.as_deref())
}

fn resolve_vpn_service_name_from(
    operator_override: Option<&str>,
    process_cgroup: Option<&str>,
) -> String {
    operator_override
        .and_then(validated_systemd_service_name)
        .or_else(|| process_cgroup.and_then(systemd_service_name_from_cgroup))
        .unwrap_or_else(|| VPN_SERVICE_NAME.to_string())
}

fn systemd_service_name_from_cgroup(cgroup: &str) -> Option<String> {
    cgroup
        .lines()
        .flat_map(|line| line.split('/').rev())
        .find_map(|segment| {
            if segment.ends_with(".service") {
                validated_systemd_service_name(segment)
            } else {
                None
            }
        })
}

fn validated_systemd_service_name(value: &str) -> Option<String> {
    let value = value.trim();
    let valid_length = !value.is_empty() && value.len() <= 255;
    let valid_prefix = value
        .bytes()
        .next()
        .map(|byte| byte.is_ascii_alphanumeric())
        .unwrap_or(false);
    let valid_suffix = value.ends_with(".service") || !value.contains('.');
    let valid_chars = value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'@' | b':')
    });
    (valid_length && valid_prefix && valid_suffix && valid_chars).then(|| value.to_string())
}

async fn collect_service_manager_status(service_name: &str) -> ServiceManagerStatus {
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // Nodeboard/backend consumers:
    //   /root/aeronyx/privacy_network/api/vpn_observability.py
    //   /root/open/nodeboard/app/dashboard/services/page.tsx
    //
    // This command returns local process manager metadata only. It does not
    // inspect destinations, DNS contents, packet payloads, domains, URLs,
    // browsing history, voucher secrets, client public IPs, or wallet traffic.
    let result = timeout(
        CHECK_TIMEOUT,
        TokioCommand::new("systemctl")
            .args([
                "show",
                service_name,
                "--property=LoadState,ActiveState,UnitFileState",
                "--value",
            ])
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines().map(str::trim);
            let load_state = lines.next().unwrap_or("").to_string();
            let active_state = lines.next().unwrap_or("").to_string();
            let unit_file_state = lines.next().unwrap_or("").to_string();
            let load_state = if load_state.is_empty() {
                "unknown".to_string()
            } else {
                load_state
            };
            let active_state = if active_state.is_empty() {
                "unknown".to_string()
            } else {
                active_state
            };
            let unit_file_state = if unit_file_state.is_empty() {
                "unknown".to_string()
            } else {
                unit_file_state
            };
            let restart_supported = load_state == "loaded";
            ServiceManagerStatus {
                manager: "systemd",
                service_name: service_name.to_string(),
                load_state: load_state.clone(),
                active_state: active_state.clone(),
                unit_file_state: unit_file_state.clone(),
                restart_supported,
                detail: if restart_supported {
                    format!(
                        "{} systemd service is loaded (ActiveState={}, UnitFileState={})",
                        service_name, active_state, unit_file_state
                    )
                } else {
                    format!(
                        "{} systemd service is not restartable from nodeboard (LoadState={}, ActiveState={}, UnitFileState={})",
                        service_name, load_state, active_state, unit_file_state
                    )
                },
            }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim().chars().take(240).collect::<String>();
            ServiceManagerStatus {
                manager: "systemd",
                service_name: service_name.to_string(),
                load_state: "unknown".to_string(),
                active_state: "unknown".to_string(),
                unit_file_state: "unknown".to_string(),
                restart_supported: false,
                detail: format!("systemctl show failed: {}", detail),
            }
        }
        Ok(Err(error)) => ServiceManagerStatus {
            manager: "systemd",
            service_name: service_name.to_string(),
            load_state: "unavailable".to_string(),
            active_state: "unavailable".to_string(),
            unit_file_state: "unavailable".to_string(),
            restart_supported: false,
            detail: format!("systemctl unavailable: {}", error),
        },
        Err(_) => ServiceManagerStatus {
            manager: "systemd",
            service_name: service_name.to_string(),
            load_state: "timeout".to_string(),
            active_state: "timeout".to_string(),
            unit_file_state: "timeout".to_string(),
            restart_supported: false,
            detail: "systemctl show timed out".to_string(),
        },
    }
}

async fn check_udp_listener(listen_addr: SocketAddr) -> HealthCheck {
    let port = listen_addr.port().to_string();
    match run_command("ss", &["-lun"], CHECK_TIMEOUT).await {
        Ok(out) => {
            let needle = format!(":{}", port);
            let ok = out.lines().any(|line| line.contains(&needle));
            HealthCheck {
                name: "udp_listener",
                ok,
                detail: if ok {
                    format!("UDP listener found on port {}", port)
                } else {
                    format!("No UDP listener found on port {}", port)
                },
            }
        }
        Err(e) => HealthCheck {
            name: "udp_listener",
            ok: false,
            detail: format!("ss failed: {}", e),
        },
    }
}

async fn check_tun_device(device: &str) -> HealthCheck {
    match run_command("ip", &["addr", "show", "dev", device], CHECK_TIMEOUT).await {
        Ok(out) => {
            let ok = out.contains("state UP") || out.contains("state UNKNOWN");
            HealthCheck {
                name: "tun_device",
                ok,
                detail: if ok {
                    format!("{} exists and is usable", device)
                } else {
                    format!("{} exists but is not up", device)
                },
            }
        }
        Err(e) => HealthCheck {
            name: "tun_device",
            ok: false,
            detail: format!("{} not found or ip command failed: {}", device, e),
        },
    }
}

async fn read_tun_mtu(device: &str) -> Result<u16, String> {
    let path = format!("/sys/class/net/{}/mtu", device);
    let value = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("read {} failed: {}", path, e))?;
    value
        .trim()
        .parse::<u16>()
        .map_err(|e| format!("parse {} failed: {}", path, e))
}

async fn check_mtu_config(
    device: &str,
    configured_mtu: u16,
    running_mtu: Option<u16>,
) -> HealthCheck {
    match running_mtu {
        Some(actual) => {
            let range_ok = (1280..=1500).contains(&actual);
            let matches_config = actual == configured_mtu;
            HealthCheck {
                name: "mtu_config",
                ok: range_ok && matches_config,
                detail: if range_ok && matches_config {
                    format!("{} MTU {} matches config", device, actual)
                } else if !matches_config {
                    format!(
                        "{} MTU {} does not match configured MTU {}",
                        device, actual, configured_mtu
                    )
                } else {
                    format!(
                        "{} MTU {} is outside the recommended Internet VPN range 1280-1500",
                        device, actual
                    )
                },
            }
        }
        None => HealthCheck {
            name: "mtu_config",
            ok: false,
            detail: format!(
                "{} MTU is unavailable from /sys/class/net/{}/mtu",
                device, device
            ),
        },
    }
}

async fn check_ip_forwarding() -> HealthCheck {
    match tokio::fs::read_to_string("/proc/sys/net/ipv4/ip_forward").await {
        Ok(value) => {
            let trimmed = value.trim();
            HealthCheck {
                name: "ip_forward",
                ok: trimmed == "1",
                detail: format!("net.ipv4.ip_forward={}", trimmed),
            }
        }
        Err(e) => HealthCheck {
            name: "ip_forward",
            ok: false,
            detail: format!("read failed: {}", e),
        },
    }
}

async fn check_nat_masquerade(ip_range: &str) -> HealthCheck {
    match run_command(
        "iptables",
        &["-t", "nat", "-S", "POSTROUTING"],
        CHECK_TIMEOUT,
    )
    .await
    {
        Ok(out) => {
            let ok = out.lines().any(|line| {
                line.contains(ip_range)
                    && line.contains("MASQUERADE")
                    && (line.contains("-s") || line.contains("--source"))
            });
            HealthCheck {
                name: "nat_masquerade",
                ok,
                detail: if ok {
                    format!("MASQUERADE rule found for {}", ip_range)
                } else {
                    format!("No MASQUERADE rule found for {}", ip_range)
                },
            }
        }
        Err(e) => HealthCheck {
            name: "nat_masquerade",
            ok: false,
            detail: format!("iptables failed: {}", e),
        },
    }
}

async fn check_dns_socket(gateway_ip: Ipv4Addr) -> HealthCheck {
    match run_command("ss", &["-lun"], CHECK_TIMEOUT).await {
        Ok(out) => {
            let gateway = gateway_ip.to_string();
            let ok = out.lines().any(|line| {
                line.contains(":53") && (line.contains(&gateway) || line.contains("0.0.0.0:53"))
            });
            HealthCheck {
                name: "dns_stub",
                ok,
                detail: if ok {
                    format!("DNS listener found for {}:53", gateway)
                } else {
                    format!("No DNS listener found for {}:53", gateway)
                },
            }
        }
        Err(e) => HealthCheck {
            name: "dns_stub",
            ok: false,
            detail: format!("ss failed: {}", e),
        },
    }
}

async fn check_dns_query(gateway_ip: Ipv4Addr) -> HealthCheck {
    match dns_query_a(gateway_ip, DNS_QUERY_NAME).await {
        Ok(answers) => HealthCheck {
            name: "dns_query",
            ok: answers > 0,
            detail: format!(
                "{} A answers via {}:53 = {}",
                DNS_QUERY_NAME, gateway_ip, answers
            ),
        },
        Err(e) => HealthCheck {
            name: "dns_query",
            ok: false,
            detail: format!("{} via {}:53 failed: {}", DNS_QUERY_NAME, gateway_ip, e),
        },
    }
}

async fn check_internet_egress() -> HealthCheck {
    match timeout(CHECK_TIMEOUT, TcpStream::connect(EGRESS_CHECK_ADDR)).await {
        Ok(Ok(_stream)) => HealthCheck {
            name: "internet_egress",
            ok: true,
            detail: format!("TCP connect to {} succeeded", EGRESS_CHECK_ADDR),
        },
        Ok(Err(e)) => HealthCheck {
            name: "internet_egress",
            ok: false,
            detail: format!("TCP connect to {} failed: {}", EGRESS_CHECK_ADDR, e),
        },
        Err(_) => HealthCheck {
            name: "internet_egress",
            ok: false,
            detail: format!("TCP connect to {} timed out", EGRESS_CHECK_ADDR),
        },
    }
}

async fn collect_capacity_status(
    config: &ServerConfig,
    ip_pool: &IpPoolService,
    node_policy: &NodePolicySnapshot,
    policy_enforcement: &NodePolicyEnforcementSnapshot,
    placement_readiness: &NodePolicyPlacementSnapshot,
    active_sessions: usize,
) -> VpnCapacityStatus {
    let interface = collect_interface_capacity(config.device_name()).await;
    let interface_drops = interface.packet_drops;
    let policy_drops = policy_enforcement.bandwidth_drops;
    let packet_drops_total = match interface_drops {
        Some(drops) => Some(drops.saturating_add(policy_drops)),
        None if policy_drops > 0 => Some(policy_drops),
        None => None,
    };
    let virtual_ip_range = config.ip_range().to_string();
    let ip_pool_capacity = ip_pool.capacity();
    let ip_pool_used = ip_pool.allocated_count();
    let ip_pool_free = ip_pool.available_count();
    let max_connections = config.max_sessions();
    let policy_max_sessions = node_policy.max_sessions;
    let conntrack = collect_conntrack_capacity();
    let file_descriptors = collect_fd_capacity();
    let disk = collect_disk_capacity().await;
    let risks = collect_capacity_risks(
        &virtual_ip_range,
        ip_pool_capacity,
        ip_pool_free,
        max_connections,
        policy_max_sessions,
        &conntrack,
        &file_descriptors,
        &disk,
        placement_readiness.bandwidth_limit_mbps,
        placement_readiness.bandwidth_limit_bytes_per_second,
        placement_readiness.bandwidth_window_bytes,
        placement_readiness.bandwidth_window_used_percent,
        &placement_readiness.traffic_capacity_status,
        packet_drops_total,
    );

    VpnCapacityStatus {
        virtual_ip_range,
        ip_pool_capacity,
        ip_pool_used,
        ip_pool_free,
        max_connections,
        policy_max_sessions,
        active_sessions,
        session_capacity_remaining: placement_readiness.session_capacity_remaining,
        bandwidth_limit_mbps: placement_readiness.bandwidth_limit_mbps,
        bandwidth_limit_bytes_per_second: placement_readiness.bandwidth_limit_bytes_per_second,
        bandwidth_window_bytes: placement_readiness.bandwidth_window_bytes,
        bandwidth_window_used_percent: placement_readiness.bandwidth_window_used_percent,
        traffic_capacity_status: placement_readiness.traffic_capacity_status.clone(),
        conntrack,
        file_descriptors,
        disk,
        interface,
        packet_drops_total,
        risks,
        source: "rust_vpn_health_capacity_snapshot",
        privacy_boundary: concat!(
            "aggregate node capacity only; no client public IPs, destinations, ",
            "DNS contents, packet payloads, domains, URLs, browsing history, ",
            "voucher secrets, or wallet-level traffic"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_capacity_risks(
    virtual_ip_range: &str,
    ip_pool_capacity: usize,
    ip_pool_free: usize,
    max_connections: usize,
    policy_max_sessions: u32,
    conntrack: &ConntrackCapacityStatus,
    file_descriptors: &FileDescriptorCapacityStatus,
    disk: &DiskCapacityStatus,
    bandwidth_limit_mbps: u32,
    bandwidth_limit_bytes_per_second: u64,
    bandwidth_window_bytes: u64,
    bandwidth_window_used_percent: Option<f64>,
    traffic_capacity_status: &str,
    packet_drops_total: Option<u64>,
) -> Vec<CapacityRiskStatus> {
    let mut risks = Vec::new();

    if max_connections > ip_pool_capacity {
        let required = max_connections;
        let recommended = recommended_ipv4_cidr(virtual_ip_range, required)
            .unwrap_or_else(|| "a larger vpn.virtual_ip_range".to_string());
        risks.push(CapacityRiskStatus {
            severity: "warning",
            code: "vpn_ip_pool_below_max_connections",
            message: format!(
                "Configured max_connections {} exceeds usable VPN IP pool {}.",
                max_connections, ip_pool_capacity
            ),
            remediation: format!(
                "During a maintenance window, expand vpn.virtual_ip_range to at least {} or lower limits.max_connections to {} or below, then run deploy/node/aeronyx-node.sh network.",
                recommended, ip_pool_capacity
            ),
            recommended_value: Some(format!(
                "vpn.virtual_ip_range >= {} or limits.max_connections <= {}",
                recommended, ip_pool_capacity
            )),
            recommended_command: Some(format!(
                "sudo ./deploy/node/aeronyx-node.sh network --set-vpn-cidr {}",
                recommended
            )),
        });
    }

    if policy_max_sessions > 0 && policy_max_sessions as usize > ip_pool_capacity {
        let required = policy_max_sessions as usize;
        let recommended = recommended_ipv4_cidr(virtual_ip_range, required)
            .unwrap_or_else(|| "a larger vpn.virtual_ip_range".to_string());
        risks.push(CapacityRiskStatus {
            severity: "warning",
            code: "vpn_ip_pool_below_policy_max_sessions",
            message: format!(
                "Nodeboard policy max_sessions {} exceeds usable VPN IP pool {}.",
                policy_max_sessions, ip_pool_capacity
            ),
            remediation: format!(
                "Expand vpn.virtual_ip_range to at least {} or lower the nodeboard policy max_sessions before commercial placement.",
                recommended
            ),
            recommended_value: Some(format!(
                "nodeboard max_sessions <= {} or vpn.virtual_ip_range >= {}",
                ip_pool_capacity, recommended
            )),
            recommended_command: Some(format!(
                "sudo ./deploy/node/aeronyx-node.sh network --set-vpn-cidr {}",
                recommended
            )),
        });
    }

    if ip_pool_free == 0 {
        let recommended = recommended_ipv4_cidr(
            virtual_ip_range,
            ip_pool_capacity
                .saturating_add(256)
                .max(ip_pool_capacity.saturating_add(1)),
        )
        .unwrap_or_else(|| "a larger vpn.virtual_ip_range".to_string());
        risks.push(CapacityRiskStatus {
            severity: "critical",
            code: "vpn_ip_pool_exhausted",
            message: "No free VPN virtual IP addresses remain for new sessions.".to_string(),
            remediation:
                "Drain traffic or expand vpn.virtual_ip_range before admitting additional clients."
                    .to_string(),
            recommended_value: Some(format!("vpn.virtual_ip_range >= {}", recommended)),
            recommended_command: Some(format!(
                "sudo ./deploy/node/aeronyx-node.sh network --set-vpn-cidr {}",
                recommended
            )),
        });
    }

    if let Some(percent) = conntrack.used_percent {
        if percent >= 80.0 {
            let recommended = recommended_conntrack_max(conntrack.used, conntrack.max);
            risks.push(CapacityRiskStatus {
                severity: if percent >= 90.0 { "critical" } else { "warning" },
                code: "conntrack_pressure",
                message: format!("Linux conntrack usage is {:.2}%.", percent),
                remediation: "Raise nf_conntrack_max and keep conntrack headroom above 20% before scaling traffic.".to_string(),
                recommended_value: Some(format!("net.netfilter.nf_conntrack_max >= {}", recommended)),
                recommended_command: Some(format!(
                    "sudo sysctl -w net.netfilter.nf_conntrack_max={}",
                    recommended
                )),
            });
        }
    }

    if let Some(percent) = file_descriptors.used_percent {
        if percent >= 80.0 {
            let recommended =
                recommended_fd_soft_limit(file_descriptors.used, file_descriptors.soft_limit);
            risks.push(CapacityRiskStatus {
                severity: if percent >= 90.0 { "critical" } else { "warning" },
                code: "file_descriptor_pressure",
                message: format!("Process file descriptor usage is {:.2}%.", percent),
                remediation: "Raise the systemd LimitNOFILE value or reduce active load before adding clients.".to_string(),
                recommended_value: Some(format!("systemd LimitNOFILE >= {}", recommended)),
                recommended_command: Some(format!(
                    "sudo systemctl edit aeronyx-server # set [Service] LimitNOFILE={}",
                    recommended
                )),
            });
        }
    }

    for disk_path in [&disk.root, &disk.state] {
        if let Some(percent) = disk_path.used_percent {
            if percent >= 85.0 {
                risks.push(CapacityRiskStatus {
                    severity: if percent >= 95.0 {
                        "critical"
                    } else {
                        "warning"
                    },
                    code: "disk_pressure",
                    message: format!("Filesystem {} usage is {:.2}%.", disk_path.path, percent),
                    remediation: concat!(
                        "Free disk space or expand the volume before build logs, ",
                        "upgrade artifacts, MemChain data, or encrypted storage growth ",
                        "interrupt node operations."
                    )
                    .to_string(),
                    recommended_value: Some(format!("Keep {} below 85% used", disk_path.path)),
                    recommended_command: None,
                });
            }
        }
    }

    if bandwidth_limit_bytes_per_second > 0 {
        let bandwidth_percent = bandwidth_window_used_percent.unwrap_or_else(|| {
            round_two(
                (bandwidth_window_bytes as f64 / bandwidth_limit_bytes_per_second as f64) * 100.0,
            )
        });
        if traffic_capacity_status == "saturated" || bandwidth_percent >= 100.0 {
            risks.push(CapacityRiskStatus {
                severity: "critical",
                code: "bandwidth_limit_pressure",
                message: format!(
                    "Bandwidth limiter is saturated at {:.2}% of the {} Mbps cap.",
                    bandwidth_percent, bandwidth_limit_mbps
                ),
                remediation: "Raise bandwidth_limit_mbps, reduce placement weight, or move traffic to another region before admitting more paid sessions.".to_string(),
                recommended_value: Some(format!(
                    "bandwidth_limit_mbps > {} or lower placement weight",
                    bandwidth_limit_mbps
                )),
                recommended_command: None,
            });
        } else if traffic_capacity_status == "near_limit" || bandwidth_percent >= 80.0 {
            risks.push(CapacityRiskStatus {
                severity: "warning",
                code: "bandwidth_limit_pressure",
                message: format!(
                    "Bandwidth limiter is near capacity at {:.2}% of the {} Mbps cap.",
                    bandwidth_percent, bandwidth_limit_mbps
                ),
                remediation: "Review bandwidth_limit_mbps and regional placement before increasing commercial traffic.".to_string(),
                recommended_value: Some(format!(
                    "Keep bandwidth window below 80% of {} Mbps cap",
                    bandwidth_limit_mbps
                )),
                recommended_command: None,
            });
        }
    }

    if let Some(drops) = packet_drops_total {
        if drops > 0 {
            risks.push(CapacityRiskStatus {
                severity: "warning",
                code: "packet_drops_detected",
                message: format!("{} packet drops were reported by the VPN interface or policy layer.", drops),
                remediation: "Inspect host NIC/TUN queues, CPU pressure, and policy bandwidth drops before increasing placement weight.".to_string(),
                recommended_value: Some("packet drops should return to 0 before increasing placement".to_string()),
                recommended_command: Some("./deploy/node/aeronyx-node.sh health --json".to_string()),
            });
        }
    }

    risks
}

fn recommended_ipv4_cidr(current_cidr: &str, required_usable_ips: usize) -> Option<String> {
    let (raw_ip, _) = current_cidr.split_once('/')?;
    let ip: Ipv4Addr = raw_ip.parse().ok()?;
    let ip_u32 = u32::from(ip);

    for prefix in (0..=30).rev() {
        if usable_ipv4_clients_for_prefix(prefix) < required_usable_ips {
            continue;
        }
        let mask = if prefix == 0 {
            0
        } else {
            u32::MAX << (32 - prefix)
        };
        let network = Ipv4Addr::from(ip_u32 & mask);
        return Some(format!("{}/{}", network, prefix));
    }

    None
}

fn recommended_conntrack_max(used: Option<u64>, current_max: Option<u64>) -> u64 {
    let used_target = used
        .map(|value| value.saturating_mul(10).saturating_add(6) / 7)
        .unwrap_or(0);
    let doubled_current = current_max.unwrap_or(0).saturating_mul(2);
    used_target.max(doubled_current).max(262_144)
}

fn recommended_fd_soft_limit(used: Option<u64>, current_soft_limit: Option<u64>) -> u64 {
    let used_target = used.unwrap_or(0).saturating_mul(2);
    let doubled_current = current_soft_limit.unwrap_or(0).saturating_mul(2);
    used_target.max(doubled_current).max(65_535)
}

fn usable_ipv4_clients_for_prefix(prefix: u32) -> usize {
    if prefix >= 31 {
        return 0;
    }
    let total = 1usize << (32 - prefix);
    total.saturating_sub(3)
}

async fn collect_interface_capacity(name: &str) -> InterfaceCapacityStatus {
    let rx_bytes = read_sysfs_counter(name, "rx_bytes");
    let tx_bytes = read_sysfs_counter(name, "tx_bytes");
    let rx_packets = read_sysfs_counter(name, "rx_packets");
    let tx_packets = read_sysfs_counter(name, "tx_packets");
    let rx_dropped = read_sysfs_counter(name, "rx_dropped");
    let tx_dropped = read_sysfs_counter(name, "tx_dropped");
    let packet_drops = match (rx_dropped, tx_dropped) {
        (Some(rx), Some(tx)) => Some(rx.saturating_add(tx)),
        (Some(rx), None) => Some(rx),
        (None, Some(tx)) => Some(tx),
        (None, None) => None,
    };

    let rates = match (rx_bytes, tx_bytes, rx_packets, tx_packets) {
        (Some(rx_b), Some(tx_b), Some(rx_p), Some(tx_p)) => interface_rates(
            name,
            InterfaceCounterSnapshot {
                timestamp: unix_now_secs(),
                rx_bytes: rx_b,
                tx_bytes: tx_b,
                rx_packets: rx_p,
                tx_packets: tx_p,
            },
        ),
        _ => None,
    };

    let (rx_pps, tx_pps, total_pps, rx_bps, tx_bps, total_bps) = rates
        .map(|r| (r.0, r.1, r.2, r.3, r.4, r.5))
        .unwrap_or((None, None, None, None, None, None));

    InterfaceCapacityStatus {
        interface: name.to_string(),
        rx_bytes,
        tx_bytes,
        rx_packets,
        tx_packets,
        rx_dropped,
        tx_dropped,
        packet_drops,
        rx_pps,
        tx_pps,
        total_pps,
        rx_bps,
        tx_bps,
        total_bps,
    }
}

fn read_sysfs_counter(interface: &str, name: &str) -> Option<u64> {
    let path = format!("/sys/class/net/{}/statistics/{}", interface, name);
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn interface_rates(
    interface: &str,
    current: InterfaceCounterSnapshot,
) -> Option<(
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
)> {
    static PREVIOUS: OnceLock<Mutex<HashMap<String, InterfaceCounterSnapshot>>> = OnceLock::new();
    let samples = PREVIOUS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = samples.lock().ok()?;
    let previous = guard.insert(interface.to_string(), current)?;
    let elapsed = current.timestamp.saturating_sub(previous.timestamp);
    if elapsed == 0 {
        return None;
    }

    let seconds = elapsed as f64;
    let rx_packets = current.rx_packets.saturating_sub(previous.rx_packets) as f64 / seconds;
    let tx_packets = current.tx_packets.saturating_sub(previous.tx_packets) as f64 / seconds;
    let rx_bps = current.rx_bytes.saturating_sub(previous.rx_bytes) as f64 * 8.0 / seconds;
    let tx_bps = current.tx_bytes.saturating_sub(previous.tx_bytes) as f64 * 8.0 / seconds;
    Some((
        Some(round_two(rx_packets)),
        Some(round_two(tx_packets)),
        Some(round_two(rx_packets + tx_packets)),
        Some(round_two(rx_bps)),
        Some(round_two(tx_bps)),
        Some(round_two(rx_bps + tx_bps)),
    ))
}

fn collect_conntrack_capacity() -> ConntrackCapacityStatus {
    let used = read_u64_file("/proc/sys/net/netfilter/nf_conntrack_count");
    let max = read_u64_file("/proc/sys/net/netfilter/nf_conntrack_max");
    ConntrackCapacityStatus {
        used,
        max,
        used_percent: percent(used, max),
    }
}

fn collect_fd_capacity() -> FileDescriptorCapacityStatus {
    let used = std::fs::read_dir("/proc/self/fd")
        .ok()
        .map(|entries| entries.filter_map(std::result::Result::ok).count() as u64);
    let (soft_limit, hard_limit) = read_open_file_limits();
    FileDescriptorCapacityStatus {
        used,
        soft_limit,
        hard_limit,
        used_percent: percent(used, soft_limit),
    }
}

async fn collect_disk_capacity() -> DiskCapacityStatus {
    DiskCapacityStatus {
        root: collect_disk_path_capacity("/").await,
        state: collect_disk_path_capacity(AERONYX_STATE_DIR).await,
        source: "df_posix_block_usage",
        privacy_boundary: concat!(
            "aggregate filesystem usage for AeroNyx operations only; no file ",
            "lists, MemChain records, encrypted storage contents, client public ",
            "IPs, destinations, DNS contents, packet payloads, domains, URLs, ",
            "browsing history, voucher secrets, chat plaintext, or wallet-level traffic"
        ),
    }
}

async fn collect_disk_path_capacity(path: &'static str) -> DiskPathCapacityStatus {
    let mut status = DiskPathCapacityStatus {
        reported: false,
        path,
        total_bytes: None,
        used_bytes: None,
        available_bytes: None,
        used_percent: None,
    };

    let Ok(command_result) = timeout(
        CHECK_TIMEOUT,
        TokioCommand::new("df")
            .arg("-P")
            .arg("-B1")
            .arg(path)
            .output(),
    )
    .await
    else {
        return status;
    };

    let Ok(output) = command_result else {
        return status;
    };

    if !output.status.success() {
        return status;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(line) = stdout.lines().nth(1) else {
        return status;
    };
    let columns: Vec<&str> = line.split_whitespace().collect();
    if columns.len() < 6 {
        return status;
    }

    status.total_bytes = columns.get(1).and_then(|value| value.parse::<u64>().ok());
    status.used_bytes = columns.get(2).and_then(|value| value.parse::<u64>().ok());
    status.available_bytes = columns.get(3).and_then(|value| value.parse::<u64>().ok());
    status.used_percent = columns
        .get(4)
        .and_then(|value| value.trim_end_matches('%').parse::<f64>().ok());
    status.reported = status.total_bytes.is_some()
        || status.used_bytes.is_some()
        || status.available_bytes.is_some();
    status
}

fn read_u64_file(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_open_file_limits() -> (Option<u64>, Option<u64>) {
    let content = match std::fs::read_to_string("/proc/self/limits") {
        Ok(content) => content,
        Err(_) => return (None, None),
    };
    for line in content.lines() {
        if !line.starts_with("Max open files") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            return (None, None);
        }
        return (parse_limit(parts[3]), parse_limit(parts[4]));
    }
    (None, None)
}

fn parse_limit(value: &str) -> Option<u64> {
    if value.eq_ignore_ascii_case("unlimited") {
        None
    } else {
        value.parse().ok()
    }
}

fn percent(used: Option<u64>, max: Option<u64>) -> Option<f64> {
    let used = used?;
    let max = max?;
    if max == 0 {
        None
    } else {
        Some(round_two((used as f64 / max as f64) * 100.0))
    }
}

fn round_two(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

async fn collect_recent_error_events(service_name: &str) -> Vec<RecentErrorEvent> {
    let output = match run_command(
        "journalctl",
        &[
            "-u",
            service_name,
            "-p",
            "warning..alert",
            "--since",
            "-24 hours",
            "--no-pager",
            "--output=short-iso",
            "--lines=20",
        ],
        Duration::from_secs(3),
    )
    .await
    {
        Ok(output) => output,
        Err(_) => return Vec::new(),
    };

    output
        .lines()
        .rev()
        .filter_map(parse_recent_error_line)
        .take(5)
        .collect()
}

fn parse_recent_error_line(line: &str) -> Option<RecentErrorEvent> {
    let line = line.trim();
    if line.is_empty() || line == "-- No entries --" {
        return None;
    }

    let mut parts = line.split_whitespace();
    let timestamp = parts.next().map(|value| value.to_string());
    let _host = parts.next();
    let _unit = parts.next();
    let raw_message = parts.collect::<Vec<_>>().join(" ");
    let severity = classify_recent_error_severity(&raw_message);
    let message = raw_message;
    let message = sanitize_operational_log_message(&message);
    if message.is_empty() {
        return None;
    }

    Some(RecentErrorEvent {
        timestamp,
        severity,
        source: "systemd_journal_aeronyx_server_warning_alert",
        message,
        privacy_boundary: concat!(
            "sanitized node service log summary only; client public IPs, URLs, ",
            "long key-like values, DNS contents, destinations, packet payloads, ",
            "voucher secrets, chat plaintext, and wallet-level traffic are redacted"
        ),
    })
}

fn classify_recent_error_severity(message: &str) -> &'static str {
    let normalized = message.to_ascii_lowercase();
    if normalized.contains("emerg")
        || normalized.contains("emergency")
        || normalized.contains("alert")
        || normalized.contains("critical")
        || normalized.contains("crit")
        || normalized.contains("panic")
        || normalized.contains("fatal")
    {
        return "critical";
    }

    if normalized.contains("error")
        || normalized.contains("failed")
        || normalized.contains("failure")
        || normalized.contains("timed out")
        || normalized.contains("timeout")
    {
        return "critical";
    }

    if normalized.contains("notice") || normalized.contains("info") {
        return "info";
    }

    "warning"
}

fn sanitize_operational_log_message(input: &str) -> String {
    let mut out = Vec::new();
    for token in input.split_whitespace() {
        let trimmed = token
            .trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '[' | ']' | '"' | '\''));
        let (prefix, value) = trimmed
            .split_once('=')
            .map(|(key, value)| (Some(key), value))
            .unwrap_or((None, trimmed));
        let lower = value.to_ascii_lowercase();
        let replacement = if lower.starts_with("http://") || lower.starts_with("https://") {
            Some("[url]")
        } else if looks_like_network_destination(value) {
            Some("[ip]")
        } else if looks_like_key_material(value) {
            Some("[redacted]")
        } else {
            None
        };

        if let Some(replacement) = replacement {
            if let Some(prefix) = prefix {
                out.push(format!("{}={}", prefix, replacement));
            } else {
                out.push(replacement.to_string());
            }
        } else {
            out.push(token.to_string());
        }
    }

    let mut message = out.join(" ").replace('\0', "");
    const MAX_LEN: usize = 240;
    if message.len() > MAX_LEN {
        message.truncate(MAX_LEN);
        message.push_str("...");
    }
    message
}

fn looks_like_network_destination(value: &str) -> bool {
    let value = value
        .trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '[' | ']' | '"' | '\'' | '.'));

    if value.parse::<Ipv4Addr>().is_ok()
        || value.parse::<Ipv6Addr>().is_ok()
        || value.parse::<SocketAddr>().is_ok()
    {
        return true;
    }

    if let Some((host, port)) = value.rsplit_once(':') {
        if !host.contains(':')
            && port.parse::<u16>().is_ok()
            && (host.parse::<Ipv4Addr>().is_ok() || looks_like_domain_name(host))
        {
            return true;
        }
    }

    looks_like_domain_name(value)
}

fn looks_like_domain_name(value: &str) -> bool {
    let value = value.trim_end_matches('.');
    if value.len() > 253 || !value.contains('.') {
        return false;
    }

    let mut labels = value.split('.').peekable();
    let mut label_count = 0usize;
    while let Some(label) = labels.next() {
        label_count += 1;
        if label.is_empty()
            || label.len() > 63
            || label.starts_with('-')
            || label.ends_with('-')
            || !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return false;
        }

        if labels.peek().is_none()
            && (label.len() < 2 || !label.chars().all(|c| c.is_ascii_alphabetic()))
        {
            return false;
        }
    }

    label_count >= 2
}

fn looks_like_key_material(value: &str) -> bool {
    if value.len() >= 32 && value.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    if value.len() >= 24
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '=' | '/' | '+'))
    {
        return true;
    }
    false
}

async fn dns_query_a(server_ip: Ipv4Addr, name: &str) -> std::result::Result<u16, String> {
    let server = SocketAddr::from((server_ip, 53));
    let socket = UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|e| e.to_string())?;
    let query = build_dns_query(name)?;
    timeout(CHECK_TIMEOUT, socket.send_to(&query, server))
        .await
        .map_err(|_| "send timeout".to_string())?
        .map_err(|e| e.to_string())?;

    let mut buf = [0u8; 512];
    let (len, _) = timeout(CHECK_TIMEOUT, socket.recv_from(&mut buf))
        .await
        .map_err(|_| "receive timeout".to_string())?
        .map_err(|e| e.to_string())?;
    if len < 12 {
        return Err("short DNS response".to_string());
    }
    let rcode = buf[3] & 0x0f;
    if rcode != 0 {
        return Err(format!("DNS rcode={}", rcode));
    }
    Ok(u16::from_be_bytes([buf[6], buf[7]]))
}

fn build_dns_query(name: &str) -> std::result::Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&0xAE90u16.to_be_bytes()); // transaction id
    out.extend_from_slice(&0x0100u16.to_be_bytes()); // recursion desired
    out.extend_from_slice(&1u16.to_be_bytes()); // qdcount
    out.extend_from_slice(&0u16.to_be_bytes()); // ancount
    out.extend_from_slice(&0u16.to_be_bytes()); // nscount
    out.extend_from_slice(&0u16.to_be_bytes()); // arcount
    for label in name.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err(format!("invalid DNS label in {}", name));
        }
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
    out.extend_from_slice(&1u16.to_be_bytes()); // A
    out.extend_from_slice(&1u16.to_be_bytes()); // IN
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_manager_name_prefers_valid_operator_override() {
        let resolved = resolve_vpn_service_name_from(
            Some("aeronyx-server-jp1.service"),
            Some("0::/system.slice/aeronyx-server-us1.service\n"),
        );

        assert_eq!(resolved, "aeronyx-server-jp1.service");
    }

    #[test]
    fn service_manager_name_detects_current_systemd_cgroup() {
        let resolved = resolve_vpn_service_name_from(
            None,
            Some("0::/system.slice/aeronyx-server-jp1.service\n"),
        );

        assert_eq!(resolved, "aeronyx-server-jp1.service");
    }

    #[test]
    fn service_manager_name_rejects_unsafe_override_and_falls_back() {
        let resolved = resolve_vpn_service_name_from(
            Some("../../another.service"),
            Some("0::/system.slice/aeronyx-server-jp1.service\n"),
        );

        assert_eq!(resolved, "aeronyx-server-jp1.service");
        assert_eq!(
            resolve_vpn_service_name_from(Some("bad unit"), None),
            VPN_SERVICE_NAME
        );
        assert_eq!(
            resolve_vpn_service_name_from(Some("--system"), None),
            VPN_SERVICE_NAME
        );
    }

    #[test]
    fn recommended_ipv4_cidr_matches_thousand_session_profile() {
        assert_eq!(
            recommended_ipv4_cidr("100.64.0.0/24", 1_000),
            Some("100.64.0.0/22".to_string())
        );
    }

    #[test]
    fn capacity_risks_include_actionable_ip_pool_remediation() {
        let disk_path = DiskPathCapacityStatus {
            reported: false,
            path: "/",
            total_bytes: None,
            used_bytes: None,
            available_bytes: None,
            used_percent: None,
        };
        let disk = DiskCapacityStatus {
            root: disk_path.clone(),
            state: disk_path,
            source: "test",
            privacy_boundary: "aggregate disk capacity only",
        };
        let conntrack = ConntrackCapacityStatus {
            used: Some(10),
            max: Some(1_000),
            used_percent: Some(1.0),
        };
        let file_descriptors = FileDescriptorCapacityStatus {
            used: Some(10),
            soft_limit: Some(1_024),
            hard_limit: Some(4_096),
            used_percent: Some(1.0),
        };
        let risks = collect_capacity_risks(
            "100.64.0.0/24",
            253,
            253,
            1_000,
            0,
            &conntrack,
            &file_descriptors,
            &disk,
            0,
            0,
            0,
            None,
            "within_limit",
            Some(0),
        );

        assert_eq!(risks.len(), 1);
        assert_eq!(risks[0].code, "vpn_ip_pool_below_max_connections");
        assert!(risks[0].remediation.contains("100.64.0.0/22"));
        assert!(risks[0]
            .recommended_value
            .as_deref()
            .unwrap_or("")
            .contains("100.64.0.0/22"));
        assert!(risks[0]
            .recommended_command
            .as_deref()
            .unwrap_or("")
            .contains("aeronyx-node.sh network"));
    }

    #[test]
    fn startup_self_check_ready_when_runtime_and_recovery_are_ready() {
        let config = ServerConfig::default();
        let checks = healthy_runtime_checks();
        let service_manager = healthy_service_manager();
        let transport = healthy_transport();
        let capacity = healthy_capacity();
        let discovery_status = serde_json::json!({
            "peer_store": {
                "bootstrap": {
                    "peer_cache_configured": false,
                    "gossip_enabled": false,
                    "seed_endpoints_configured": 0
                },
                "stability": {
                    "health": "healthy",
                    "restart_recovery_configured": false
                }
            }
        });

        let status = collect_startup_self_check(
            &config,
            &checks,
            &service_manager,
            &transport,
            &capacity,
            &discovery_status,
        );

        assert_eq!(status.status, "ready");
        assert!(status.ready);
        assert_eq!(status.failed_checks, 0);
        assert_eq!(status.warning_checks, 0);
        assert!(status.blocking_checks.is_empty());
    }

    #[test]
    fn startup_self_check_blocks_discovery_without_peer_cache_recovery() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.discovery.seed_endpoints = vec!["http://127.0.0.1:8422".to_string()];

        let checks = healthy_runtime_checks();
        let service_manager = healthy_service_manager();
        let transport = healthy_transport();
        let capacity = healthy_capacity();
        let discovery_status = serde_json::json!({
            "peer_store": {
                "bootstrap": {
                    "peer_cache_configured": false,
                    "gossip_enabled": true,
                    "seed_endpoints_configured": 1
                },
                "stability": {
                    "health": "healthy",
                    "restart_recovery_configured": false
                }
            }
        });

        let status = collect_startup_self_check(
            &config,
            &checks,
            &service_manager,
            &transport,
            &capacity,
            &discovery_status,
        );

        assert_eq!(status.status, "failed");
        assert!(!status.ready);
        assert!(status
            .blocking_checks
            .contains(&"peer_store_restart_recovery"));
        assert!(status
            .recommended_action
            .contains("discovery.peer_cache_path"));
    }

    #[test]
    fn recent_error_sanitizer_redacts_sensitive_tokens() {
        let sanitized = sanitize_operational_log_message(
            "failed peer 203.0.113.10 endpoint=198.51.100.8:443 ipv6=2001:db8::1 host=api.example.com url=https://example.com/path key=0123456789abcdef0123456789abcdef",
        );

        assert!(sanitized.contains("[ip]"));
        assert!(sanitized.contains("endpoint=[ip]"));
        assert!(sanitized.contains("ipv6=[ip]"));
        assert!(sanitized.contains("host=[ip]"));
        assert!(sanitized.contains("url=[url]") || sanitized.contains("[url]"));
        assert!(sanitized.contains("key=[redacted]") || sanitized.contains("[redacted]"));
        assert!(!sanitized.contains("203.0.113.10"));
        assert!(!sanitized.contains("198.51.100.8"));
        assert!(!sanitized.contains("2001:db8::1"));
        assert!(!sanitized.contains("api.example.com"));
        assert!(!sanitized.contains("0123456789abcdef0123456789abcdef"));
    }

    fn healthy_runtime_checks() -> Vec<HealthCheck> {
        [
            "udp_listener",
            "tun_device",
            "mtu_config",
            "ip_forward",
            "nat_masquerade",
            "dns_stub",
            "dns_query",
            "internet_egress",
        ]
        .into_iter()
        .map(|name| HealthCheck {
            name,
            ok: true,
            detail: "ok".to_string(),
        })
        .collect()
    }

    fn healthy_service_manager() -> ServiceManagerStatus {
        ServiceManagerStatus {
            manager: "systemd",
            service_name: VPN_SERVICE_NAME.to_string(),
            load_state: "loaded".to_string(),
            active_state: "active".to_string(),
            unit_file_state: "enabled".to_string(),
            restart_supported: true,
            detail: "service is active".to_string(),
        }
    }

    fn healthy_transport() -> VpnTransportHealthStatus {
        let udp = TransportCarrierStatus {
            key: "udp",
            enabled: true,
            implemented: true,
            active: true,
            endpoint: Some("0.0.0.0:51820".to_string()),
            status: "active",
            detail: "UDP listener active".to_string(),
        };
        let tcp_tls = TransportCarrierStatus {
            key: "tcp_tls",
            enabled: false,
            implemented: false,
            active: false,
            endpoint: None,
            status: "planned",
            detail: "planned".to_string(),
        };
        let websocket_https = TransportCarrierStatus {
            key: "websocket_https",
            enabled: false,
            implemented: false,
            active: false,
            endpoint: None,
            status: "planned",
            detail: "planned".to_string(),
        };
        VpnTransportHealthStatus {
            supported_transports: vec!["udp"],
            configured_transports: vec!["udp"],
            preferred_transport: "udp".to_string(),
            effective_transport: "udp",
            fallback_available: false,
            udp,
            tcp_tls,
            websocket_https,
            source: "test",
            privacy_boundary: "aggregate transport state only",
        }
    }

    fn healthy_capacity() -> VpnCapacityStatus {
        let disk_path = DiskPathCapacityStatus {
            reported: true,
            path: "/",
            total_bytes: Some(100),
            used_bytes: Some(10),
            available_bytes: Some(90),
            used_percent: Some(10.0),
        };
        VpnCapacityStatus {
            virtual_ip_range: "100.64.0.0/22".to_string(),
            ip_pool_capacity: 1_021,
            ip_pool_used: 1,
            ip_pool_free: 1_020,
            max_connections: 1_000,
            policy_max_sessions: 0,
            active_sessions: 0,
            session_capacity_remaining: Some(1_000),
            bandwidth_limit_mbps: 0,
            bandwidth_limit_bytes_per_second: 0,
            bandwidth_window_bytes: 0,
            bandwidth_window_used_percent: None,
            traffic_capacity_status: "unlimited".to_string(),
            conntrack: ConntrackCapacityStatus {
                used: Some(10),
                max: Some(10_000),
                used_percent: Some(0.1),
            },
            file_descriptors: FileDescriptorCapacityStatus {
                used: Some(10),
                soft_limit: Some(10_000),
                hard_limit: Some(10_000),
                used_percent: Some(0.1),
            },
            disk: DiskCapacityStatus {
                root: disk_path.clone(),
                state: disk_path,
                source: "test",
                privacy_boundary: "aggregate disk capacity only",
            },
            interface: InterfaceCapacityStatus {
                interface: "aeronyx0".to_string(),
                rx_bytes: Some(0),
                tx_bytes: Some(0),
                rx_packets: Some(0),
                tx_packets: Some(0),
                rx_dropped: Some(0),
                tx_dropped: Some(0),
                packet_drops: Some(0),
                rx_pps: Some(0.0),
                tx_pps: Some(0.0),
                total_pps: Some(0.0),
                rx_bps: Some(0.0),
                tx_bps: Some(0.0),
                total_bps: Some(0.0),
            },
            packet_drops_total: Some(0),
            risks: Vec::new(),
            source: "test",
            privacy_boundary: "aggregate node capacity only",
        }
    }

    #[test]
    fn recent_error_parser_classifies_severity_without_leaking_sensitive_tokens() {
        let critical = parse_recent_error_line(
            "2026-06-18T12:00:00Z host aeronyx-server[42]: ERROR failed endpoint=203.0.113.10:443 key=0123456789abcdef0123456789abcdef",
        )
        .expect("critical event");
        assert_eq!(critical.severity, "critical");
        assert!(critical.message.contains("endpoint=[ip]"));
        assert!(critical.message.contains("key=[redacted]"));
        assert!(!critical.message.contains("203.0.113.10"));
        assert!(!critical
            .message
            .contains("0123456789abcdef0123456789abcdef"));

        let warning = parse_recent_error_line(
            "2026-06-18T12:00:01Z host aeronyx-server[42]: warning capacity threshold near limit",
        )
        .expect("warning event");
        assert_eq!(warning.severity, "warning");

        let info = parse_recent_error_line(
            "2026-06-18T12:00:02Z host aeronyx-server[42]: notice service recovered after restart",
        )
        .expect("info event");
        assert_eq!(info.severity, "info");
    }
}

async fn run_command(
    program: &str,
    args: &[&str],
    limit: Duration,
) -> std::result::Result<String, String> {
    let fut = TokioCommand::new(program).args(args).output();
    let output = timeout(limit, fut)
        .await
        .map_err(|_| format!("{} timed out", program))?
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("{} exited with {}", program, output.status)
        } else {
            stderr
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
