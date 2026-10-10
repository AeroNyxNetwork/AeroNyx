// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/startup_self_check.rs
// ============================================
//! # Startup self-check
//!
//! Owns `collect_startup_self_check`, which folds validated config, runtime
//! health checks, transport, capacity, and discovery/recovery-anchor state
//! into one nodeboard-ready readiness contract, and its JSON path helpers.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use serde_json::Value;

use crate::config::ServerConfig;

use super::{
    unix_now_secs, HealthCheck, ServiceManagerStatus, StartupSelfCheckItem, StartupSelfCheckStatus,
    VpnCapacityStatus, VpnTransportHealthStatus,
};

pub(super) fn collect_startup_self_check(
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
    let recovery_anchor_status =
        discovery_str(discovery_status, &["recovery_anchor", "status"]).unwrap_or("not_reported");
    let recovery_anchor_ready =
        discovery_bool(discovery_status, &["recovery_anchor", "ready_for_restore"])
            .unwrap_or(false);
    let recovery_witness_required = config
        .discovery
        .verified_delivery_witness_required_for_restore
        || discovery_bool(
            discovery_status,
            &["recovery_anchor", "external_witness", "required"],
        )
        .unwrap_or(false);
    let recovery_witness_generation_aligned = discovery_bool(
        discovery_status,
        &["recovery_anchor", "external_witness", "generation_aligned"],
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

    // [RECOVERY-ANCHOR-LOCAL-HEALTH 2026-08-21 by Codex] Preserve optional
    // witness deployments while refusing to call a strict deployment ready
    // until the external witness covers the exact active cache generation.
    // Optional deployments still surface adverse/incomplete anchor state as a
    // warning instead of silently treating it as healthy.
    let recovery_anchor_ok = !discovery_enabled
        || if recovery_witness_required {
            recovery_anchor_ready
        } else {
            recovery_anchor_status == "ready"
        };
    checks.push(startup_item(
        "peer_store_recovery_anchor",
        recovery_anchor_ok,
        if discovery_enabled && recovery_witness_required {
            "critical"
        } else {
            "warning"
        },
        format!(
            "recovery_anchor_status={} ready_for_restore={} witness_required={} witness_generation_aligned={}",
            recovery_anchor_status,
            recovery_anchor_ready,
            recovery_witness_required,
            recovery_witness_generation_aligned
        ),
        if recovery_witness_required {
            "Obtain the configured external witness quorum for the current peer-cache generation before relying on restored routing state.".to_string()
        } else {
            "Persist and verify the local signed peer-cache recovery anchor before relying on restored routing state.".to_string()
        },
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

#[cfg(test)]
mod tests;
