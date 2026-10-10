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
//! [SYSTEMD-CHILD-ISOLATION 2026-08-11 by Codex] Local health commands use the
//! crate-owned isolated process factory so they cannot inherit readiness,
//! watchdog, or socket-activation authority from the Rust service process.
//! [HEALTH-SNAPSHOT-LATENCY 2026-08-12 by Codex] Independent host checks run
//! concurrently, while bounded journal telemetry refreshes in the background.
//! Slow local journals therefore cannot stall VPN health or CMS heartbeats.
//! [OPERATOR-PATH-PRIVACY 2026-08-14 by Codex] Operator and heartbeat
//! telemetry keeps storage/executable readiness observable without exporting
//! host filesystem paths. Sanitized journal summaries redact path tokens too.
//! [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] Host-local health now embeds
//! the existing typed ChatRelay peer status. It exposes aggregate counters and
//! stable reason buckets only, allowing relay smoke failures to be diagnosed
//! without identifiers, endpoints, ciphertext, or payload-derived metadata.
//! [CHAT-RELAY-DURABILITY-STATUS 2026-08-16 by Codex] The same typed snapshot
//! carries verified aggregate custody durability. A missing relay runtime stays
//! `unknown`; health must never infer FULL durability from configuration alone.
//! [RECOVERY-ANCHOR-LOCAL-HEALTH 2026-08-21 by Codex] Local health and startup
//! self-check now consume the shared recovery-anchor projection. A deployment
//! that explicitly requires an exact-generation external witness cannot report
//! startup readiness while that witness is absent or generation-mismatched.
//! [ANONYMOUS-MAILBOX-READINESS-PROJECTION 2026-09-14 by Codex] Optional
//! mailbox readiness is copied only from a composition-root projection. Legacy
//! constructors retain an empty projection and therefore omit unverified data.
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.
//! Every public and crate-visible item is re-exported from this module root,
//! so all existing `api::vpn_health::*` paths are unchanged. The shared
//! report types, constants, and command/clock helpers stay here so every
//! child can build and read them without widening any field visibility.
//! - `vpn_health/router.rs`: HTTP router, handlers, and JSON value entry points
//! - `vpn_health/health_snapshot.rs`: `/api/vpn/health` response assembly
//! - `vpn_health/startup_self_check.rs`: startup readiness self-check
//! - `vpn_health/operator_status.rs`: operator service snapshot and action summary
//! - `vpn_health/transport.rs`: transport, handshake capability, privacy-protocol health
//! - `vpn_health/runtime.rs`: runtime version, rollout, and upgrade status
//! - `vpn_health/service_manager.rs`: systemd unit resolution and status
//! - `vpn_health/host_checks.rs`: UDP/TUN/MTU/forwarding/NAT/DNS/egress probes
//! - `vpn_health/capacity.rs`: capacity snapshot and placement risks
//! - `vpn_health/recent_errors.rs`: sanitized journal error telemetry

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::time::timeout;

use crate::config::ServerConfig;
use crate::handlers::packet::{PacketHandler, PacketRuntimeStatus};
use crate::isolated_child_command;
use crate::services::chat_relay::ChatRelayPeerStatus;
use crate::services::{
    ChatRelayService, IpPoolService, NodePolicyEnforcementSnapshot, NodePolicyPlacementSnapshot,
    NodePolicyRuntime, NodePolicySnapshot, PeerStore, SessionManager,
};
use crate::voucher_verifier::{VoucherMetricsSnapshot, VoucherVerifier};

mod capacity;
mod health_snapshot;
mod host_checks;
mod operator_status;
mod recent_errors;
mod router;
mod runtime;
mod service_manager;
mod startup_self_check;
mod transport;

pub use router::{
    build_vpn_health_router, collect_node_operator_status_value, collect_vpn_health_value,
};
pub(crate) use router::{
    build_vpn_health_router_with_anonymous_mailbox_readiness,
    collect_node_operator_status_value_with_anonymous_mailbox_readiness,
    collect_vpn_health_value_with_anonymous_mailbox_readiness,
};

const CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const DNS_QUERY_NAME: &str = "api.aeronyx.network";
const EGRESS_CHECK_ADDR: &str = "1.1.1.1:443";
const VPN_SERVICE_NAME: &str = "aeronyx-server";
const VPN_SERVICE_UNIT_NAME: &str = "aeronyx-server.service";
const VPN_SERVICE_NAME_ENV: &str = "AERONYX_SYSTEMD_SERVICE";
const PROC_SELF_CGROUP_PATH: &str = "/proc/self/cgroup";
const UPGRADE_STATUS_FILE: &str = "/var/lib/aeronyx/upgrade-status.json";
const AERONYX_STATE_DIR: &str = "/var/lib/aeronyx";
const RECENT_ERROR_CACHE_TTL_SECS: u64 = 60;
const RECENT_ERROR_CACHE_MAX_STALE_SECS: u64 = 600;
const RECENT_ERROR_COMMAND_TIMEOUT: Duration = Duration::from_secs(6);
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
    chat_relay: Option<Arc<ChatRelayService>>,
    anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
}

/// Shared, aggregate-only anonymous mailbox readiness projection.
///
/// The empty state is deliberately serialized as no field. Only the server
/// composition root may publish an observed local-runtime snapshot.
#[derive(Clone, Default)]
pub(crate) struct AnonymousMailboxReadinessProjection {
    snapshot: Arc<parking_lot::RwLock<Option<ChatRelayPeerStatus>>>,
}

impl AnonymousMailboxReadinessProjection {
    /// Publishes one snapshot after actual terminal and dispatcher composition.
    pub(crate) fn publish_local_composition(
        &self,
        configured: bool,
        custody_store_opened: bool,
        ticket_terminal_wired: bool,
        source_coordinator_enabled: bool,
        dispatcher_admitted: bool,
        cleanup_runtime_supervised: bool,
    ) {
        let snapshot = ChatRelayPeerStatus::new(false)
            .with_anonymous_mailbox_readiness_from_local_composition(
                configured,
                custody_store_opened,
                ticket_terminal_wired,
                source_coordinator_enabled,
                dispatcher_admitted,
                cleanup_runtime_supervised,
            );
        *self.snapshot.write() = Some(snapshot);
    }

    pub(crate) fn apply_to(&self, target: &mut ChatRelayPeerStatus) {
        if let Some(source) = self.snapshot.read().as_ref() {
            target.apply_anonymous_mailbox_readiness_from(source);
        }
    }
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

#[derive(Debug, Default)]
struct RecentErrorCache {
    entries: RwLock<Vec<RecentErrorEvent>>,
    last_attempt_at: AtomicU64,
    last_success_at: AtomicU64,
    refresh_in_flight: AtomicBool,
}

static RECENT_ERROR_CACHE: OnceLock<RecentErrorCache> = OnceLock::new();

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

/// Static VPN handshake support projected from the canonical core policy.
///
/// [VPN-HANDSHAKE-CAPABILITY-HEALTH 2026-09-23 by Codex] This is compile-time
/// capability metadata, not a live handshake probe or a client-readiness claim.
/// It contains no endpoint, session, identity, route, or traffic information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct VpnHandshakeCapabilityStatus {
    version: u8,
    v1_supported: bool,
    v2_supported: bool,
    default_version: u8,
    mode: VpnHandshakeCapabilityMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VpnHandshakeCapabilityMode {
    LegacyOnly,
    DualStack,
    V2Only,
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

/// Aggregate-only encrypted relay runtime status exposed by local health.
#[derive(Debug, Clone, Serialize)]
struct ChatRelayHealthStatus {
    configured_enabled: bool,
    runtime_ready: bool,
    peer_relay: ChatRelayPeerStatus,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    vpn_handshake_capability: Option<VpnHandshakeCapabilityStatus>,
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
    chat_relay_status: ChatRelayHealthStatus,
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

async fn run_command(
    program: &str,
    args: &[&str],
    limit: Duration,
) -> std::result::Result<String, String> {
    let fut = isolated_child_command(program).args(args).output();
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
