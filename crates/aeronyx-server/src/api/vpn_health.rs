// ============================================
// File: crates/aeronyx-server/src/api/vpn_health.rs
// ============================================
//! VPN node health endpoint.
//!
//! This endpoint is intentionally read-only. It verifies the Linux node pieces
//! that commonly make a tunnel appear "connected but offline": UDP listener,
//! TUN device and MTU, IPv4 forwarding, NAT masquerade, VPN DNS stub, DNS
//! resolution, and basic Internet egress.

use std::net::{Ipv4Addr, SocketAddr};
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

#[derive(Clone)]
pub struct VpnHealthState {
    config: ServerConfig,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
}

#[derive(Debug, Serialize)]
struct HealthCheck {
    name: &'static str,
    ok: bool,
    detail: String,
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
    node_policy: NodePolicySnapshot,
    policy_enforcement: NodePolicyEnforcementSnapshot,
    voucher_metrics: VoucherMetricsSnapshot,
    checks: Vec<HealthCheck>,
}

pub fn build_vpn_health_router(
    config: ServerConfig,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
) -> Router {
    Router::new()
        .route("/api/vpn/health", get(vpn_health_handler))
        .with_state(VpnHealthState {
            config,
            sessions,
            node_policy,
            voucher_verifier,
        })
}

async fn vpn_health_handler(State(state): State<VpnHealthState>) -> impl IntoResponse {
    Json(collect_vpn_health_response(state).await)
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
) -> Value {
    let state = VpnHealthState {
        config,
        sessions,
        node_policy,
        voucher_verifier,
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

async fn collect_vpn_health_response(state: VpnHealthState) -> VpnHealthResponse {
    let config = state.config;
    let gateway_ip = config.gateway_ip();
    let listen_addr = config.listen_addr();
    let tun_device = config.device_name().to_string();
    let configured_mtu = config.mtu();
    let running_mtu = read_tun_mtu(&tun_device).await.ok();
    let ip_range = config.ip_range().to_string();

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
        node_policy: state.node_policy.snapshot(),
        policy_enforcement: state.node_policy.enforcement_snapshot(),
        voucher_metrics: state.voucher_verifier.metrics_snapshot(),
        checks,
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
