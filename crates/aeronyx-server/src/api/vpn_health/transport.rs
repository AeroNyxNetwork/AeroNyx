// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/transport.rs
// ============================================
//! # Transport and privacy-protocol health
//!
//! Owns the VPN transport carrier projection, the static VPN handshake
//! capability derived from the core version policy, and the aggregate
//! privacy-protocol health summary.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::net::SocketAddr;

use aeronyx_core::protocol::version::{
    classify_supported_protocol_version, CURRENT_PROTOCOL_VERSION, PROTOCOL_VERSION_V1,
    PROTOCOL_VERSION_V2,
};

use crate::config::ServerConfig;

use super::{
    PrivacyProtocolHealthStatus, PrivacyProtocolRuntimeStatus, ServiceManagerStatus,
    TransportCarrierStatus, VpnHandshakeCapabilityMode, VpnHandshakeCapabilityStatus,
    VpnTransportHealthStatus,
};

pub(super) fn collect_transport_health(
    config: &ServerConfig,
    udp_listen_addr: SocketAddr,
    udp_listener_ok: bool,
) -> VpnTransportHealthStatus {
    let transports = config.vpn_transports();
    let mut configured_transports = Vec::new();
    if transports.udp_enabled {
        configured_transports.push("udp");
    }
    if transports.tcp_tls_enabled {
        configured_transports.push("tcp_tls");
    }
    if transports.websocket_enabled {
        configured_transports.push("websocket_https");
    }

    // Phase 1 reports actual production support, not desired future config.
    // This keeps nodeboard and public stats from advertising a fallback carrier
    // before its server listener and client implementation exist.
    let supported_transports = if transports.udp_enabled {
        vec!["udp"]
    } else {
        Vec::new()
    };

    let udp = TransportCarrierStatus {
        key: "udp",
        enabled: transports.udp_enabled,
        implemented: true,
        active: transports.udp_enabled && udp_listener_ok,
        endpoint: Some(udp_listen_addr.to_string()),
        status: if transports.udp_enabled && udp_listener_ok {
            "active"
        } else {
            "degraded"
        },
        detail: if transports.udp_enabled && udp_listener_ok {
            format!("UDP data-plane listener is active at {}", udp_listen_addr)
        } else {
            format!(
                "UDP is configured but listener check failed at {}",
                udp_listen_addr
            )
        },
    };

    let tcp_tls = TransportCarrierStatus {
        key: "tcp_tls",
        enabled: transports.tcp_tls_enabled,
        implemented: false,
        active: false,
        endpoint: transports.tcp_tls_public_endpoint.clone(),
        status: if transports.tcp_tls_enabled {
            "configured_not_active"
        } else {
            "planned"
        },
        detail: if transports.tcp_tls_enabled {
            "TCP/TLS fallback is configured in metadata but its Rust data-plane listener is not implemented yet".to_string()
        } else {
            "TCP/TLS fallback is planned but disabled on this node".to_string()
        },
    };

    let websocket_https = TransportCarrierStatus {
        key: "websocket_https",
        enabled: transports.websocket_enabled,
        implemented: false,
        active: false,
        endpoint: transports.websocket_public_url.clone(),
        status: if transports.websocket_enabled {
            "configured_not_active"
        } else {
            "planned"
        },
        detail: if transports.websocket_enabled {
            "WebSocket HTTPS fallback is configured in metadata but its Rust data-plane listener is not implemented yet".to_string()
        } else {
            "WebSocket HTTPS fallback is planned but disabled on this node".to_string()
        },
    };

    VpnTransportHealthStatus {
        supported_transports,
        configured_transports,
        preferred_transport: transports.preferred_transport.clone(),
        effective_transport: "udp",
        fallback_available: false,
        udp,
        tcp_tls,
        websocket_https,
        source: "rust_vpn_transport_capability_metadata",
        privacy_boundary: concat!(
            "transport capability metadata only; no packet payloads, DNS ",
            "contents, destinations, domains, URLs, browsing history, voucher ",
            "secrets, client public IPs, or wallet-level traffic"
        ),
    }
}

pub(super) fn collect_vpn_handshake_capability() -> Option<VpnHandshakeCapabilityStatus> {
    let v1_supported = classify_supported_protocol_version(PROTOCOL_VERSION_V1).is_some();
    let v2_supported = classify_supported_protocol_version(PROTOCOL_VERSION_V2).is_some();
    let mode = classify_vpn_handshake_capability_mode(v1_supported, v2_supported)?;

    Some(VpnHandshakeCapabilityStatus {
        version: 1,
        v1_supported,
        v2_supported,
        default_version: CURRENT_PROTOCOL_VERSION,
        mode,
    })
}

fn classify_vpn_handshake_capability_mode(
    v1_supported: bool,
    v2_supported: bool,
) -> Option<VpnHandshakeCapabilityMode> {
    Some(match (v1_supported, v2_supported) {
        (true, false) => VpnHandshakeCapabilityMode::LegacyOnly,
        (true, true) => VpnHandshakeCapabilityMode::DualStack,
        (false, true) => VpnHandshakeCapabilityMode::V2Only,
        (false, false) => return None,
    })
}

pub(super) fn collect_privacy_protocol_health(
    status: &'static str,
    checked_at: u64,
    failed_checks: usize,
    active_sessions: usize,
    active_wallet_devices: usize,
    transport_health: &VpnTransportHealthStatus,
    service_manager: &ServiceManagerStatus,
) -> PrivacyProtocolHealthStatus {
    PrivacyProtocolHealthStatus {
        protocol: "aeronyx_privacy_protocol",
        label: "AeroNyx Privacy Protocol",
        status,
        checked_at,
        failed_checks,
        active_sessions,
        active_wallet_devices,
        data_plane: "aeronyx_privacy_protocol",
        preferred_transport: transport_health.preferred_transport.clone(),
        effective_transport: transport_health.effective_transport,
        service_active_state: service_manager.active_state.clone(),
        protocol_runtime: PrivacyProtocolRuntimeStatus {
            active: status != "failed",
            status,
            detail: "This Rust node reports AeroNyx privacy protocol runtime health from aggregate service, transport, and routing checks.",
            source: "rust_vpn_health.protocol_model",
            privacy_boundary: concat!(
                "protocol model metadata only; no client public IPs, ",
                "destinations, DNS contents, packet payloads, domains, URLs, ",
                "browsing history, voucher secrets, chat plaintext, or ",
                "wallet-level traffic"
            ),
        },
        source: "rust_vpn_health.privacy_protocol_health",
        privacy_boundary: concat!(
            "aggregate privacy protocol operations metadata only; no client ",
            "public IPs, destinations, DNS contents, packet payloads, domains, ",
            "URLs, browsing history, voucher secrets, chat plaintext, private ",
            "keys, or wallet-level traffic"
        ),
    }
}

#[cfg(test)]
mod tests;
