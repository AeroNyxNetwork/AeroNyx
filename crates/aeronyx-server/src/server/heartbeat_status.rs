// ============================================
// File: crates/aeronyx-server/src/server/heartbeat_status.rs
// ============================================
//! # Management heartbeat discovery projection
//!
//! Owns the bounded `PeerStore` and discovery objects carried by the signed
//! management heartbeat.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use crate::api::discovery::{
    blind_relay_runtime_status_value, discovery_readiness_status_value,
    recovery_anchor_status_value, DiscoveryLocalCapabilityStatus,
};
use crate::services::peer_store::{PeerStoreSignedPeerRecordsStatus, PeerStoreStatus};

/// Build the bounded PeerStore projection sent in the management heartbeat.
///
/// [BOUNDED-DISCOVERY-HEARTBEAT 2026-08-02 by Codex] Full route candidates,
/// per-peer health rows, and audit rings remain available from the local node
/// API. Repeating them in every heartbeat caused the backend's 64 KiB safety
/// gate to reject the entire discovery snapshot. This projection keeps the
/// aggregate contracts consumed by nodeboard while excluding those heavy
/// local diagnostics. Never add endpoints, full node ids, route ids, payloads,
/// receiver identities, client IPs, or user traffic here.
pub(super) fn peer_store_heartbeat_status_value(status: &PeerStoreStatus) -> serde_json::Value {
    serde_json::json!({
        "snapshot": &status.snapshot,
        "runtime": &status.runtime,
        "blind_relay_quality": &status.blind_relay_quality,
        "two_hop_path_proof_history": &status.two_hop_path_proof_history,
        "three_hop_path_proof_history": &status.three_hop_path_proof_history,
        "max_peers": status.max_peers,
        "recent_peer_events": &status.recent_peer_events,
        "bootstrap": &status.bootstrap,
        "stability": &status.stability,
        "route_governance": &status.route_governance,
        "peer_quorum": &status.peer_quorum,
        "network_story": &status.network_story,
    })
}

/// Builds the bounded discovery object carried by the signed management heartbeat.
///
/// [RECOVERY-ANCHOR-HEARTBEAT 2026-08-21 by Codex] Keep all derived readiness
/// in shared API helpers so the local status endpoint and backend heartbeat
/// cannot disagree about external-witness generation alignment. The signed
/// peer-record batch is intentionally passed in after its own bounded export;
/// this function must not add local audit rows, route candidates, witness
/// identities/endpoints, anchor material, or user traffic.
pub(super) fn discovery_heartbeat_status_value(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
    signed_peer_records: PeerStoreSignedPeerRecordsStatus,
) -> serde_json::Value {
    serde_json::json!({
        "generated_at": generated_at,
        "peer_store": peer_store_heartbeat_status_value(status),
        "route_governance": &status.route_governance,
        "blind_relay_runtime": blind_relay_runtime_status_value(
            generated_at,
            status,
            local_capabilities,
        ),
        "recovery_anchor": recovery_anchor_status_value(status),
        "signed_peer_records": signed_peer_records,
        "local_capabilities": local_capabilities,
        "discovery_readiness": discovery_readiness_status_value(status, local_capabilities),
        "source": "rust_peer_store",
        "privacy_boundary": "aggregate node discovery, blind relay, and recovery-anchor state plus bounded signed node-level discovery descriptors for central verification; no client IPs, destinations, DNS contents, packet payloads, chat plaintext, voucher secrets, private keys, or wallet-level traffic"
    })
}
