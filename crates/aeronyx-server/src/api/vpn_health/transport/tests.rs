// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/transport/tests.rs
// ============================================
//! # Tests: transport and handshake capability health
//!
//! Unit tests for transport and handshake capability health, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

#[test]
fn vpn_handshake_capability_reports_current_dual_stack_v2_default() {
    let capability =
        collect_vpn_handshake_capability().expect("current policy supports VPN handshakes");

    assert_eq!(capability.version, 1);
    assert!(capability.v1_supported);
    assert!(capability.v2_supported);
    assert_eq!(capability.default_version, PROTOCOL_VERSION_V2);
    assert_eq!(capability.mode, VpnHandshakeCapabilityMode::DualStack);

    let encoded = serde_json::to_value(capability).expect("serialize handshake capability");
    assert_eq!(encoded["version"], 1);
    assert_eq!(encoded["v1_supported"], true);
    assert_eq!(encoded["v2_supported"], true);
    assert_eq!(encoded["default_version"], 2);
    assert_eq!(encoded["mode"], "dual_stack");
    assert_eq!(encoded.as_object().map(serde_json::Map::len), Some(5));
}

#[test]
fn vpn_handshake_capability_uses_canonical_version_policy() {
    let capability =
        collect_vpn_handshake_capability().expect("current policy supports VPN handshakes");

    assert_eq!(
        capability.v1_supported,
        classify_supported_protocol_version(PROTOCOL_VERSION_V1).is_some()
    );
    assert_eq!(
        capability.v2_supported,
        classify_supported_protocol_version(PROTOCOL_VERSION_V2).is_some()
    );
    assert_eq!(capability.default_version, CURRENT_PROTOCOL_VERSION);
    assert_eq!(
        classify_supported_protocol_version(capability.default_version),
        Some(aeronyx_core::protocol::version::SupportedProtocolVersion::V2)
    );
}

#[test]
fn vpn_handshake_capability_modes_are_closed() {
    assert_eq!(
        classify_vpn_handshake_capability_mode(true, false),
        Some(VpnHandshakeCapabilityMode::LegacyOnly)
    );
    assert_eq!(
        classify_vpn_handshake_capability_mode(true, true),
        Some(VpnHandshakeCapabilityMode::DualStack)
    );
    assert_eq!(
        classify_vpn_handshake_capability_mode(false, true),
        Some(VpnHandshakeCapabilityMode::V2Only)
    );
    assert_eq!(classify_vpn_handshake_capability_mode(false, false), None);
}
