// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/service_manager/tests.rs
// ============================================
//! # Tests: service manager name resolution
//!
//! Unit tests for service manager name resolution, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

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
    let resolved =
        resolve_vpn_service_name_from(None, Some("0::/system.slice/aeronyx-server-jp1.service\n"));

    assert_eq!(resolved, "aeronyx-server-jp1.service");
    assert_eq!(
        resolve_vpn_service_name_from(None, Some("0::/system.slice/aeronyx-server.service\n"),),
        VPN_SERVICE_NAME
    );
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
