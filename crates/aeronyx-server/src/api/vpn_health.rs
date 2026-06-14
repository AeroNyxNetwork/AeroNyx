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
//! Rust node is currently ready to provide.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{extract::State, response::IntoResponse, routing::get, Json, Router};
use serde::Serialize;
use serde_json::Value;
use tokio::net::{TcpStream, UdpSocket};
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;

use crate::config::ServerConfig;
use crate::services::{
    NodePolicyEnforcementSnapshot, NodePolicyRuntime, NodePolicySnapshot, SessionManager,
};
use crate::voucher_verifier::{VoucherMetricsSnapshot, VoucherVerifier};

const CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const DNS_QUERY_NAME: &str = "api.aeronyx.network";
const EGRESS_CHECK_ADDR: &str = "1.1.1.1:443";
const VPN_SERVICE_NAME: &str = "aeronyx-server";

#[derive(Clone)]
pub struct VpnHealthState {
    config: ServerConfig,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
}

#[derive(Debug, Serialize)]
struct HealthCheck {
    name: &'static str,
    ok: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct ServiceManagerStatus {
    manager: &'static str,
    service_name: &'static str,
    load_state: String,
    restart_supported: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct EncryptedMessageForwardingStatus {
    count: u64,
    source: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct VpnHealthResponse {
    status: &'static str,
    checked_at: u64,
    listen_addr: String,
    gateway_ip: String,
    virtual_ip_range: String,
    tun_device: String,
    configured_mtu: u16,
    running_mtu: Option<u16>,
    active_sessions: usize,
    active_wallet_devices: usize,
    service_manager: ServiceManagerStatus,
    node_policy: NodePolicySnapshot,
    policy_enforcement: NodePolicyEnforcementSnapshot,
    voucher_metrics: VoucherMetricsSnapshot,
    encrypted_message_forwarding: EncryptedMessageForwardingStatus,
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

#[derive(Debug, Serialize)]
struct NodeOperatorStatusResponse {
    status: &'static str,
    generated_at: u64,
    services: Vec<OperatorServiceStatus>,
    risks: Vec<OperatorRisk>,
    privacy_boundary: &'static str,
}

pub fn build_vpn_health_router(
    config: ServerConfig,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
) -> Router {
    Router::new()
        .route("/api/vpn/health", get(vpn_health_handler))
        .route("/api/node/operator/status", get(node_operator_status_handler))
        .route("/api/operator/status", get(node_operator_status_handler))
        .with_state(VpnHealthState {
            config,
            sessions,
            node_policy,
            voucher_verifier,
            encrypted_message_counter,
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
///   /root/a/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
///
/// The payload contains only local node diagnostics such as UDP listener, TUN,
/// MTU, NAT, DNS stub/query, egress reachability, and aggregate counters. It
/// never includes user destinations, DNS query contents, packet payloads, or
/// browsing history.
pub async fn collect_vpn_health_value(
    config: ServerConfig,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
) -> Value {
    let state = VpnHealthState {
        config,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
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
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
) -> Value {
    let state = VpnHealthState {
        config,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
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
    let listen_addr = config.listen_addr();
    let tun_device = config.device_name().to_string();
    let configured_mtu = config.mtu();
    let running_mtu = read_tun_mtu(&tun_device).await.ok();
    let ip_range = config.ip_range().to_string();
    let service_manager = collect_service_manager_status(VPN_SERVICE_NAME).await;

    let mut checks = Vec::new();
    checks.push(check_udp_listener(listen_addr).await);
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

    VpnHealthResponse {
        status,
        checked_at: unix_now_secs(),
        listen_addr: listen_addr.to_string(),
        gateway_ip: gateway_ip.to_string(),
        virtual_ip_range: ip_range,
        tun_device,
        configured_mtu,
        running_mtu,
        active_sessions: state.sessions.count(),
        active_wallet_devices: state.sessions.wallet_index_count(),
        service_manager,
        node_policy: state.node_policy.snapshot(),
        policy_enforcement: state.node_policy.enforcement_snapshot(),
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
        checks,
    }
}

async fn collect_node_operator_status_response(state: VpnHealthState) -> NodeOperatorStatusResponse {
    let generated_at = unix_now_secs();
    let vpn_health = collect_vpn_health_response(state.clone()).await;
    let config = state.config.clone();
    let memchain_enabled = config.memchain.is_enabled();
    let remote_storage_enabled = config.memchain.is_remote_storage_enabled();
    let chat_relay_enabled = config.memchain.is_chat_relay_enabled();
    let supernode_enabled = config.memchain.is_supernode_enabled();
    let api_secret_configured = config.memchain.effective_api_secret().is_some();
    let encrypted_messages = state.encrypted_message_counter.load(Ordering::Relaxed);
    let mut services = Vec::new();
    let mut risks = Vec::new();

    services.push(OperatorServiceStatus {
        key: "privacy_protocol",
        label: "AeroNyx Privacy Protocol",
        enabled: true,
        status: vpn_health.status,
        summary: format!(
            "{} active sessions, {} wallet devices, {} encrypted packets forwarded",
            vpn_health.active_sessions,
            vpn_health.active_wallet_devices,
            encrypted_messages
        ),
        metrics: serde_json::json!({
            "listen_addr": vpn_health.listen_addr,
            "gateway_ip": vpn_health.gateway_ip,
            "virtual_ip_range": vpn_health.virtual_ip_range,
            "tun_device": vpn_health.tun_device,
            "configured_mtu": vpn_health.configured_mtu,
            "running_mtu": vpn_health.running_mtu,
            "active_sessions": vpn_health.active_sessions,
            "active_wallet_devices": vpn_health.active_wallet_devices,
            "service_manager": vpn_health.service_manager,
            "encrypted_message_forwarding": vpn_health.encrypted_message_forwarding,
            "failed_checks": vpn_health.checks.iter().filter(|check| !check.ok).count(),
        }),
    });

    services.push(OperatorServiceStatus {
        key: "memchain",
        label: "MemChain / MPI",
        enabled: memchain_enabled,
        status: if memchain_enabled { "ok" } else { "disabled" },
        summary: if memchain_enabled {
            format!("mode={:?}, API bound at {}", config.memchain.mode, config.memchain.api_listen_addr)
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
            "Chat relay is disabled; encrypted messages are not stored for offline delivery".to_string()
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
        status: if remote_storage_enabled { "ready" } else { "planned" },
        summary: if remote_storage_enabled {
            format!("remote encrypted owner storage enabled for up to {} owners", config.memchain.max_remote_owners)
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
        status: if supernode_enabled { "ready" } else { "disabled" },
        summary: if supernode_enabled {
            format!("{} configured provider(s)", config.memchain.supernode.providers.len())
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

    if !api_secret_configured {
        risks.push(OperatorRisk {
            severity: if remote_storage_enabled { "critical" } else { "warning" },
            code: "mpi_api_secret_missing",
            message: "MemChain API secret is not configured".to_string(),
            remediation: "Set memchain.api_secret before enabling remote RPC or remote encrypted storage".to_string(),
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
            remediation: "Set memchain.max_remote_owners for commercial node capacity planning".to_string(),
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

async fn collect_service_manager_status(service_name: &'static str) -> ServiceManagerStatus {
    let result = timeout(
        CHECK_TIMEOUT,
        TokioCommand::new("systemctl")
            .args(["show", service_name, "--property=LoadState", "--value"])
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let load_state = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let load_state = if load_state.is_empty() {
                "unknown".to_string()
            } else {
                load_state
            };
            let restart_supported = load_state == "loaded";
            ServiceManagerStatus {
                manager: "systemd",
                service_name,
                load_state: load_state.clone(),
                restart_supported,
                detail: if restart_supported {
                    format!("{} systemd service is loaded", service_name)
                } else {
                    format!(
                        "{} systemd service is not restartable from nodeboard (LoadState={})",
                        service_name, load_state
                    )
                },
            }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim().chars().take(240).collect::<String>();
            ServiceManagerStatus {
                manager: "systemd",
                service_name,
                load_state: "unknown".to_string(),
                restart_supported: false,
                detail: format!("systemctl show failed: {}", detail),
            }
        }
        Ok(Err(error)) => ServiceManagerStatus {
            manager: "systemd",
            service_name,
            load_state: "unavailable".to_string(),
            restart_supported: false,
            detail: format!("systemctl unavailable: {}", error),
        },
        Err(_) => ServiceManagerStatus {
            manager: "systemd",
            service_name,
            load_state: "timeout".to_string(),
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
