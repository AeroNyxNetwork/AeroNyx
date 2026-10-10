// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/router.rs
// ============================================
//! # HTTP router and JSON entry points
//!
//! Owns the axum router builders for `/api/vpn/health` and the operator
//! status routes, their handlers, and the `collect_*_value` JSON entry
//! points used by the CMS heartbeat.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, routing::get, Json, Router};
use serde_json::Value;

use crate::config::ServerConfig;
use crate::handlers::packet::PacketHandler;
use crate::services::{
    ChatRelayService, IpPoolService, NodePolicyRuntime, PeerStore, SessionManager,
};
use crate::voucher_verifier::VoucherVerifier;

use super::health_snapshot::collect_vpn_health_response;
use super::operator_status::collect_node_operator_status_response;
use super::{unix_now_secs, AnonymousMailboxReadinessProjection, VpnHealthState};

pub fn build_vpn_health_router(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
) -> Router {
    build_vpn_health_router_with_anonymous_mailbox_readiness(
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
        chat_relay,
        AnonymousMailboxReadinessProjection::default(),
    )
}

/// Builds health routes with an observed composition-root mailbox projection.
pub fn build_vpn_health_router_with_anonymous_mailbox_readiness(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
    anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
) -> Router {
    Router::new()
        .route("/api/vpn/health", get(vpn_health_handler))
        .route(
            "/api/node/operator/status",
            get(node_operator_status_handler),
        )
        .route("/api/operator/status", get(node_operator_status_handler))
        .with_state(VpnHealthState {
            config,
            ip_pool,
            sessions,
            node_policy,
            voucher_verifier,
            encrypted_message_counter,
            packet_handler,
            peer_store,
            chat_relay,
            anonymous_mailbox_readiness,
        })
}

async fn vpn_health_handler(State(state): State<VpnHealthState>) -> impl IntoResponse {
    Json(collect_vpn_health_response(state).await)
}

async fn node_operator_status_handler(State(state): State<VpnHealthState>) -> impl IntoResponse {
    Json(collect_node_operator_status_response(state).await)
}

/// Collect privacy-safe VPN node health as JSON for the CMS heartbeat.
///
/// Source path:
///   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
///
/// The payload contains only local node diagnostics such as UDP listener, TUN,
/// MTU, NAT, DNS stub/query, egress reachability, and aggregate counters. It
/// never includes user destinations, DNS query contents, packet payloads, or
/// browsing history.
pub async fn collect_vpn_health_value(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
) -> Value {
    collect_vpn_health_value_with_anonymous_mailbox_readiness(
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
        chat_relay,
        AnonymousMailboxReadinessProjection::default(),
    )
    .await
}

/// Collects health with an observed composition-root mailbox projection.
pub async fn collect_vpn_health_value_with_anonymous_mailbox_readiness(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
    anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
) -> Value {
    let state = VpnHealthState {
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
        chat_relay,
        anonymous_mailbox_readiness,
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

/// Collect the nodeboard-facing operator service snapshot as privacy-safe JSON.
///
/// Source path:
///   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
///
/// This is reported in heartbeat `system_stats.operator_status` and exposed as
/// `/api/node/operator/status`. It contains aggregate service/config state only.
pub async fn collect_node_operator_status_value(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
) -> Value {
    collect_node_operator_status_value_with_anonymous_mailbox_readiness(
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
        chat_relay,
        AnonymousMailboxReadinessProjection::default(),
    )
    .await
}

/// Collects operator status with an observed local mailbox projection.
pub async fn collect_node_operator_status_value_with_anonymous_mailbox_readiness(
    config: ServerConfig,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    node_policy: Arc<NodePolicyRuntime>,
    voucher_verifier: Arc<VoucherVerifier>,
    encrypted_message_counter: Arc<AtomicU64>,
    packet_handler: Arc<PacketHandler>,
    peer_store: Arc<PeerStore>,
    chat_relay: Option<Arc<ChatRelayService>>,
    anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
) -> Value {
    let state = VpnHealthState {
        config,
        ip_pool,
        sessions,
        node_policy,
        voucher_verifier,
        encrypted_message_counter,
        packet_handler,
        peer_store,
        chat_relay,
        anonymous_mailbox_readiness,
    };
    serde_json::to_value(collect_node_operator_status_response(state).await).unwrap_or_else(|e| {
        serde_json::json!({
            "status": "failed",
            "generated_at": unix_now_secs(),
            "risks": [{
                "severity": "critical",
                "code": "operator_status_serialization",
                "message": format!("operator status serialization failed: {}", e),
                "remediation": "Check Rust node operator status response serialization",
            }],
        })
    })
}
