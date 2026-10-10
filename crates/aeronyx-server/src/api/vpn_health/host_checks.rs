// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/host_checks.rs
// ============================================
//! # Host network health checks
//!
//! Owns the read-only Linux probes behind `/api/vpn/health`: UDP listener,
//! TUN device and MTU, IPv4 forwarding, NAT masquerade, VPN DNS stub and
//! query, and Internet egress, plus the minimal DNS A query they use.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::net::{Ipv4Addr, SocketAddr};

use tokio::net::{TcpStream, UdpSocket};
use tokio::time::timeout;

use super::{run_command, HealthCheck, CHECK_TIMEOUT, DNS_QUERY_NAME, EGRESS_CHECK_ADDR};

pub(super) async fn check_udp_listener(listen_addr: SocketAddr) -> HealthCheck {
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

pub(super) async fn check_tun_device(device: &str) -> HealthCheck {
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

pub(super) async fn read_tun_mtu(device: &str) -> Result<u16, String> {
    let path = format!("/sys/class/net/{}/mtu", device);
    let value = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("read {} failed: {}", path, e))?;
    value
        .trim()
        .parse::<u16>()
        .map_err(|e| format!("parse {} failed: {}", path, e))
}

pub(super) async fn check_mtu_config(
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

pub(super) async fn check_ip_forwarding() -> HealthCheck {
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

pub(super) async fn check_nat_masquerade(ip_range: &str) -> HealthCheck {
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

pub(super) async fn check_dns_socket(gateway_ip: Ipv4Addr) -> HealthCheck {
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

pub(super) async fn check_dns_query(gateway_ip: Ipv4Addr) -> HealthCheck {
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

pub(super) async fn check_internet_egress() -> HealthCheck {
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
