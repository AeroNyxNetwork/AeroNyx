// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/capacity/tests.rs
// ============================================
//! # Tests: capacity risks
//!
//! Unit tests for capacity risks, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

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
