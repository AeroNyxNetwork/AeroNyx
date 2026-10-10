// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/health_snapshot.rs
// ============================================
//! # VPN health snapshot assembly
//!
//! Owns `collect_vpn_health_response`, which runs the host probes and
//! assembles the `/api/vpn/health` response, plus the discovery status and
//! chat relay health projections embedded in it.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::sync::atomic::Ordering;

use serde_json::Value;

use crate::api::discovery::recovery_anchor_status_value;
use crate::services::chat_relay::ChatRelayPeerStatus;
use crate::services::session::CLIENT_LIVENESS_TIMEOUT_SECS;
use crate::services::{ChatRelayService, PeerStore};

use super::capacity::collect_capacity_status;
use super::host_checks::{
    check_dns_query, check_dns_socket, check_internet_egress, check_ip_forwarding,
    check_mtu_config, check_nat_masquerade, check_tun_device, check_udp_listener, read_tun_mtu,
};
use super::operator_status::collect_operator_action_summary;
use super::recent_errors::collect_recent_error_events;
use super::runtime::{collect_runtime_version_status, collect_upgrade_status};
use super::service_manager::{collect_service_manager_status, resolve_vpn_service_name};
use super::startup_self_check::collect_startup_self_check;
use super::transport::{
    collect_privacy_protocol_health, collect_transport_health, collect_vpn_handshake_capability,
};
use super::{
    unix_now_secs, ChatRelayHealthStatus, EncryptedMessageForwardingStatus, SessionCleanupStatus,
    VpnHealthResponse, VpnHealthState,
};

pub(super) async fn collect_vpn_health_response(state: VpnHealthState) -> VpnHealthResponse {
    let config = state.config;
    let gateway_ip = config.gateway_ip();
    let dns_proxy_enabled = config.dns_proxy_enabled();
    let dns_owner = if dns_proxy_enabled {
        "rust_dns_proxy"
    } else {
        "external_gateway_dns"
    };
    let listen_addr = config.listen_addr();
    let tun_device = config.device_name().to_string();
    let configured_mtu = config.mtu();
    let ip_range = config.ip_range().to_string();
    let service_name = resolve_vpn_service_name();
    // [HEALTH-SNAPSHOT-LATENCY 2026-08-12 by Codex] These probes do not
    // depend on one another. Running them serially made one slow local command
    // multiply the latency of every health request and management heartbeat.
    let (
        running_mtu_result,
        service_manager,
        udp_listener_check,
        tun_device_check,
        ip_forward_check,
        nat_masquerade_check,
        dns_socket_check,
        dns_query_check,
        internet_egress_check,
    ) = tokio::join!(
        read_tun_mtu(&tun_device),
        collect_service_manager_status(&service_name),
        check_udp_listener(listen_addr),
        check_tun_device(&tun_device),
        check_ip_forwarding(),
        check_nat_masquerade(&ip_range),
        check_dns_socket(gateway_ip),
        check_dns_query(gateway_ip),
        check_internet_egress(),
    );
    let running_mtu = running_mtu_result.ok();
    let transport_health = collect_transport_health(&config, listen_addr, udp_listener_check.ok);
    let checks = vec![
        udp_listener_check,
        tun_device_check,
        check_mtu_config(&tun_device, configured_mtu, running_mtu).await,
        ip_forward_check,
        nat_masquerade_check,
        dns_socket_check,
        dns_query_check,
        internet_egress_check,
    ];

    let failed = checks.iter().filter(|c| !c.ok).count();
    let status = if failed == 0 {
        "ok"
    } else if failed <= 2 {
        "degraded"
    } else {
        "failed"
    };

    let active_sessions = state.sessions.count();
    let active_wallet_devices = state.sessions.wallet_index_count();
    let node_policy = state.node_policy.snapshot();
    let policy_enforcement = state.node_policy.enforcement_snapshot();
    let placement_readiness = state.node_policy.placement_snapshot(active_sessions);
    let (capacity, runtime, recent_errors, upgrade_status) = tokio::join!(
        collect_capacity_status(
            &config,
            &state.ip_pool,
            &node_policy,
            &policy_enforcement,
            &placement_readiness,
            active_sessions,
        ),
        collect_runtime_version_status(),
        collect_recent_error_events(&service_name),
        collect_upgrade_status(),
    );
    let packet_runtime = state.packet_handler.runtime_status();
    let discovery_status = collect_discovery_status_value(&state.peer_store);
    let mut chat_relay_status = collect_chat_relay_health_status(
        config.memchain.is_chat_relay_enabled(),
        state
            .chat_relay
            .as_deref()
            .map(ChatRelayService::peer_status),
    );
    state
        .anonymous_mailbox_readiness
        .apply_to(&mut chat_relay_status.peer_relay);
    let startup_self_check = collect_startup_self_check(
        &config,
        &checks,
        &service_manager,
        &transport_health,
        &capacity,
        &discovery_status,
    );
    let operator_action = collect_operator_action_summary(
        status,
        &checks,
        &capacity,
        &upgrade_status,
        &service_manager,
    );
    let checked_at = unix_now_secs();
    let privacy_protocol_health = collect_privacy_protocol_health(
        status,
        checked_at,
        failed,
        active_sessions,
        active_wallet_devices,
        &transport_health,
        &service_manager,
    );

    VpnHealthResponse {
        status,
        checked_at,
        listen_addr: listen_addr.to_string(),
        gateway_ip: gateway_ip.to_string(),
        dns_proxy_enabled,
        dns_owner,
        supported_transports: transport_health.supported_transports.clone(),
        preferred_transport: transport_health.preferred_transport.clone(),
        transport_health,
        vpn_handshake_capability: collect_vpn_handshake_capability(),
        privacy_protocol_health,
        startup_self_check,
        virtual_ip_range: ip_range,
        tun_device,
        configured_mtu,
        running_mtu,
        active_sessions,
        active_wallet_devices,
        service_manager,
        node_policy,
        policy_enforcement,
        placement_readiness,
        capacity,
        packet_runtime,
        discovery_status,
        chat_relay_status,
        recent_errors,
        upgrade_status,
        operator_action,
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
        session_cleanup: SessionCleanupStatus {
            client_liveness_timeout_seconds: CLIENT_LIVENESS_TIMEOUT_SECS,
            source: "session_client_activity_timeout",
            privacy_boundary: concat!(
                "local monotonic timeout metadata only; no destinations, DNS ",
                "contents, packet payloads, domains, URLs, browsing history, ",
                "voucher secrets, client public IPs, or wallet-level traffic"
            ),
        },
        runtime,
        checks,
    }
}

fn collect_discovery_status_value(peer_store: &PeerStore) -> Value {
    let now = unix_now_secs();
    let status = peer_store.status(now);
    let recovery_anchor = recovery_anchor_status_value(&status);
    serde_json::json!({
        "generated_at": now,
        "peer_store": status,
        "recovery_anchor": recovery_anchor,
        "source": "rust_peer_store",
        "privacy_boundary": concat!(
            "aggregate node discovery counters only; no client IPs, ",
            "destinations, DNS contents, packet payloads, chat plaintext, ",
            "voucher secrets, private keys, peer private keys, or wallet-level traffic"
        )
    })
}

fn collect_chat_relay_health_status(
    configured_enabled: bool,
    runtime_status: Option<ChatRelayPeerStatus>,
) -> ChatRelayHealthStatus {
    // [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] Reuse the service-owned
    // status snapshot. The fallback is explicit and typed, so an enabled but
    // missing runtime cannot be mistaken for a healthy idle relay.
    let (runtime_ready, mut peer_relay, source) = match runtime_status {
        Some(status) => (true, status, "rust_chat_relay_service"),
        None => (
            false,
            ChatRelayPeerStatus::new(configured_enabled),
            if configured_enabled {
                "rust_chat_relay_runtime_unavailable"
            } else {
                "rust_chat_relay_disabled_config"
            },
        ),
    };
    if configured_enabled && !runtime_ready {
        peer_relay.last_outbound_status = Some("failed".to_string());
        peer_relay.last_outbound_failure_reason =
            Some("chat_relay_runtime_unavailable".to_string());
    }
    ChatRelayHealthStatus {
        configured_enabled,
        runtime_ready,
        peer_relay,
        source,
        privacy_boundary: concat!(
            "aggregate encrypted relay counters and stable reason buckets only; ",
            "no message ids, wallet ids, sender or receiver keys, blob ids, ",
            "session ids, peer endpoints, client IPs, destinations, DNS contents, ",
            "packet payloads, plaintext, ciphertext, private keys, voucher secrets, ",
            "or per-user traffic"
        ),
    }
}

#[cfg(test)]
mod tests;
