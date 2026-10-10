// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/startup_self_check/tests.rs
// ============================================
//! # Tests: the startup self-check
//!
//! Unit tests for the startup self-check, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

use crate::api::vpn_health::{
    ConntrackCapacityStatus, DiskCapacityStatus, DiskPathCapacityStatus,
    FileDescriptorCapacityStatus, InterfaceCapacityStatus, TransportCarrierStatus,
    VPN_SERVICE_NAME,
};

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
fn startup_self_check_blocks_required_mismatched_recovery_witness() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.peer_cache_path = Some("/var/lib/aeronyx/peers.json".to_string());
    config
        .discovery
        .verified_delivery_witness_required_for_restore = true;

    let discovery_status = serde_json::json!({
        "peer_store": {
            "bootstrap": {
                "peer_cache_configured": true,
                "gossip_enabled": false,
                "seed_endpoints_configured": 0
            },
            "stability": {
                "health": "healthy",
                "restart_recovery_configured": true
            }
        },
        "recovery_anchor": {
            "status": "blocked",
            "ready_for_restore": false,
            "external_witness": {
                "required": true,
                "generation_aligned": false
            }
        }
    });

    let status = collect_startup_self_check(
        &config,
        &healthy_runtime_checks(),
        &healthy_service_manager(),
        &healthy_transport(),
        &healthy_capacity(),
        &discovery_status,
    );

    assert_eq!(status.status, "failed");
    assert!(status
        .blocking_checks
        .contains(&"peer_store_recovery_anchor"));
    let recovery_check = status
        .checks
        .iter()
        .find(|check| check.name == "peer_store_recovery_anchor")
        .expect("recovery anchor self-check");
    assert!(!recovery_check.ok);
    assert_eq!(recovery_check.severity, "critical");
    assert!(recovery_check.detail.contains("generation_aligned=false"));
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
