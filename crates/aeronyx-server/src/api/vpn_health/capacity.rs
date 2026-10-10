// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/capacity.rs
// ============================================
//! # Capacity status and placement risks
//!
//! Owns the aggregate capacity snapshot (IP pool, conntrack, file
//! descriptors, disk, interface counters/rates) and the structured
//! capacity placement risks with their recommended values.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::{Mutex, OnceLock};

use tokio::time::timeout;

use crate::config::ServerConfig;
use crate::isolated_child_command;
use crate::services::{
    IpPoolService, NodePolicyEnforcementSnapshot, NodePolicyPlacementSnapshot, NodePolicySnapshot,
};

use super::{
    unix_now_secs, CapacityRiskStatus, ConntrackCapacityStatus, DiskCapacityStatus,
    DiskPathCapacityStatus, FileDescriptorCapacityStatus, InterfaceCapacityStatus,
    InterfaceCounterSnapshot, VpnCapacityStatus, AERONYX_STATE_DIR, CHECK_TIMEOUT,
};

pub(super) async fn collect_capacity_status(
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
    let (root, state) = tokio::join!(
        collect_disk_path_capacity("/"),
        collect_disk_path_capacity(AERONYX_STATE_DIR),
    );
    DiskCapacityStatus {
        root,
        state,
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
        isolated_child_command("df")
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

#[cfg(test)]
mod tests;
