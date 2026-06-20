// ============================================================================
// File: crates/aeronyx-server/src/api/discovery.rs
// ============================================================================
//! # Discovery API
//!
//! ## Creation Reason
//! Exposes a minimal HTTP entry point for decentralized AeroNyx node discovery
//! so nodes can exchange signed descriptors without relying on the centralized
//! management backend.
//!
//! ## Main Functionality
//! - `GET /api/discovery/snapshot`: returns a JSON bootstrap snapshot of
//!   verified descriptors from the local `PeerStore`
//! - `POST /api/discovery/gossip`: accepts a JSON `NodeDiscoveryMessage`,
//!   applies descriptor/snapshot updates, and returns a snapshot response for
//!   request messages
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/discovery.rs: message and snapshot types
//! - aeronyx-server/src/services/peer_store.rs: verification, anti-rollback,
//!   and snapshot export logic
//! - axum: router and JSON extraction/response
//!
//! ## Main Logical Flow
//! 1. Snapshot requests read valid descriptors from `PeerStore`
//! 2. Gossip messages are applied through `PeerStore::apply_discovery_message`
//! 3. Incoming data never bypasses descriptor signature verification
//! 4. Response reports import counts and optionally includes a snapshot response
//!
//! ## Important Note for Next Developer
//! - Do not add client public IPs, packet payloads, destinations, DNS contents,
//!   domains, URLs, browsing history, voucher secrets, private keys, or
//!   wallet-level traffic to these endpoints.
//! - This API exchanges only signed node descriptors and aggregate import
//!   counts. It is not an encrypted message relay endpoint.
//! - Public exit remains disabled by default at descriptor policy level.
//! - Security decisions are recorded as privacy-safe aggregate audit events in
//!   `PeerStoreStatus.recent_audit_events`.
//! - `DiscoveryLocalCapabilityStatus` reports only local configuration and
//!   endpoint readiness; it must not include node ids, route ids, client data,
//!   peer endpoints, payloads, or wallet-level information.
//!
//! ## Last Modified
//! v0.5.0-LocalCapabilityStatus - Report ChatRelay/blind relay readiness self-check
//! v0.4.0-DiscoveryAuditLog - Added audit events for rate-limit/policy decisions
//! v0.3.0-DiscoveryPhase10-11 - Added status endpoint and inbound safety policy
//! v0.2.0-DiscoveryPhase6 - Public gossip response type for outbound sync
//! v0.1.0-DiscoveryPhase5 - Initial discovery snapshot/gossip HTTP API
// ============================================================================

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::{NodeBootstrapSnapshot, NodeDiscoveryMessage};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::config::DiscoveryConfig;
use crate::services::{PeerStore, PeerStoreImportReport, PeerStoreStatus};

// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct DiscoveryApiState {
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    rate_limit: Arc<Mutex<RateLimitState>>,
}

/// API-facing discovery safety policy.
#[derive(Debug, Clone)]
pub struct DiscoveryApiPolicy {
    max_snapshot_limit: usize,
    gossip_rate_limit_per_minute: u32,
    allowed_peer_ids: HashSet<String>,
    denied_peer_ids: HashSet<String>,
}

impl DiscoveryApiPolicy {
    /// Builds policy from server discovery config.
    #[must_use]
    pub fn from_config(config: &DiscoveryConfig) -> Self {
        Self {
            max_snapshot_limit: config.max_snapshot_limit,
            gossip_rate_limit_per_minute: config.gossip_rate_limit_per_minute,
            allowed_peer_ids: normalize_peer_ids(&config.allowed_peer_ids),
            denied_peer_ids: normalize_peer_ids(&config.denied_peer_ids),
        }
    }

    fn snapshot_limit(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.max_snapshot_limit)
            .min(self.max_snapshot_limit)
    }

    fn message_allowed(&self, message: &NodeDiscoveryMessage) -> bool {
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => true,
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => {
                self.node_allowed(&descriptor.node_id())
            }
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => snapshot
                .peers
                .iter()
                .all(|descriptor| self.node_allowed(&descriptor.node_id())),
        }
    }

    fn node_allowed(&self, node_id: &[u8; 32]) -> bool {
        let node_id = hex::encode(node_id);
        if self.denied_peer_ids.contains(&node_id) {
            return false;
        }
        self.allowed_peer_ids.is_empty() || self.allowed_peer_ids.contains(&node_id)
    }
}

impl Default for DiscoveryApiPolicy {
    fn default() -> Self {
        Self {
            max_snapshot_limit: DiscoveryConfig::default_max_snapshot_limit(),
            gossip_rate_limit_per_minute: DiscoveryConfig::default_gossip_rate_limit_per_minute(),
            allowed_peer_ids: HashSet::new(),
            denied_peer_ids: HashSet::new(),
        }
    }
}

fn normalize_peer_ids(peer_ids: &[String]) -> HashSet<String> {
    peer_ids
        .iter()
        .map(|peer_id| peer_id.trim().to_ascii_lowercase())
        .collect()
}

#[derive(Debug)]
struct RateLimitState {
    window_minute: u64,
    used: u32,
}

impl RateLimitState {
    fn new() -> Self {
        Self {
            window_minute: 0,
            used: 0,
        }
    }

    fn allow(&mut self, now: u64, limit: u32) -> bool {
        let window_minute = now / 60;
        if self.window_minute != window_minute {
            self.window_minute = window_minute;
            self.used = 0;
        }
        if self.used >= limit {
            return false;
        }
        self.used += 1;
        true
    }
}

#[derive(Debug, Deserialize)]
struct SnapshotQuery {
    limit: Option<usize>,
    public_only: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GossipResponse {
    pub applied: PeerStoreImportReport,
    pub response: Option<NodeDiscoveryMessage>,
}

#[derive(Debug, Serialize)]
pub struct DiscoveryStatusResponse {
    generated_at: u64,
    peer_store: PeerStoreStatus,
    policy: DiscoveryPolicyStatus,
    local_capabilities: DiscoveryLocalCapabilityStatus,
}

#[derive(Debug, Serialize)]
struct DiscoveryPolicyStatus {
    max_snapshot_limit: usize,
    gossip_rate_limit_per_minute: u32,
    allow_list_enabled: bool,
    allowed_peer_count: usize,
    denied_peer_count: usize,
    snapshot_default_public_only: bool,
    private_descriptors_hidden_by_default: bool,
}

/// Privacy-safe local protocol capability readiness.
///
/// This object is intentionally small and aggregate-only. It tells operators
/// whether the node configuration, public peer API endpoint, and advertised
/// descriptor capabilities agree with each other, without exposing route ids,
/// peer endpoints, client addresses, payloads, or user identifiers.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryLocalCapabilityStatus {
    /// Whether `[memchain.chat_relay].enabled` is true.
    pub chat_relay_configured: bool,
    /// Whether this process has the public discovery/peer API listener and a
    /// public endpoint configured, which is required by peer relay routes.
    pub blind_relay_endpoint_ready: bool,
    /// Whether the self descriptor advertises `NodeCapability::ChatRelay`.
    pub advertised_chat_relay_capability: bool,
    /// Whether config, endpoint readiness, and advertised capability agree.
    pub capability_config_consistent: bool,
    /// Stable operator-facing status: `ready`, `disabled`, or `misconfigured`.
    pub status: &'static str,
    /// Short remediation-oriented detail safe for public discovery status.
    pub detail: &'static str,
}

impl DiscoveryLocalCapabilityStatus {
    /// Builds a privacy-safe readiness summary for local discovery status.
    #[must_use]
    pub fn new(
        chat_relay_configured: bool,
        blind_relay_endpoint_ready: bool,
        advertised_chat_relay_capability: bool,
    ) -> Self {
        let expected_advertisement = chat_relay_configured && blind_relay_endpoint_ready;
        let capability_config_consistent =
            advertised_chat_relay_capability == expected_advertisement;
        let (status, detail) = if !capability_config_consistent {
            (
                "misconfigured",
                "chat relay capability advertisement does not match config and endpoint readiness",
            )
        } else if advertised_chat_relay_capability {
            (
                "ready",
                "chat relay and blind relay peer endpoint are configured and advertised",
            )
        } else if chat_relay_configured {
            (
                "misconfigured",
                "chat relay is enabled but public peer API endpoint is not ready",
            )
        } else {
            (
                "disabled",
                "chat relay is disabled; blind relay endpoint remains available for discovery API plumbing",
            )
        };

        Self {
            chat_relay_configured,
            blind_relay_endpoint_ready,
            advertised_chat_relay_capability,
            capability_config_consistent,
            status,
            detail,
        }
    }
}

impl Default for DiscoveryLocalCapabilityStatus {
    fn default() -> Self {
        Self::new(false, false, false)
    }
}

// ============================================
// Router
// ============================================

/// Builds the discovery API router.
pub fn build_discovery_router(peer_store: Arc<PeerStore>, policy: DiscoveryApiPolicy) -> Router {
    build_discovery_router_with_local_status(
        peer_store,
        policy,
        DiscoveryLocalCapabilityStatus::default(),
    )
}

/// Builds the discovery API router with local capability readiness status.
pub fn build_discovery_router_with_local_status(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
) -> Router {
    let state = DiscoveryApiState {
        peer_store,
        policy,
        local_capabilities,
        rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
    };
    Router::new()
        .route("/api/discovery/snapshot", get(snapshot_handler))
        .route("/api/discovery/gossip", post(gossip_handler))
        .route("/api/discovery/status", get(status_handler))
        .with_state(state)
}

// ============================================
// Handlers
// ============================================

async fn snapshot_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<SnapshotQuery>,
) -> Json<NodeBootstrapSnapshot> {
    let now = now_secs();
    let limit = state.policy.snapshot_limit(query.limit);
    Json(state.peer_store.export_bootstrap_snapshot(
        now,
        now,
        query.public_only.unwrap_or(true),
        Some(limit),
    ))
}

async fn gossip_handler(
    State(state): State<DiscoveryApiState>,
    Json(message): Json<NodeDiscoveryMessage>,
) -> impl IntoResponse {
    let now = now_secs();
    if !state
        .rate_limit
        .lock()
        .expect("discovery rate limiter poisoned")
        .allow(now, state.policy.gossip_rate_limit_per_minute)
    {
        state.peer_store.record_rate_limited(
            now,
            format!(
                "global_limit_per_minute={}",
                state.policy.gossip_rate_limit_per_minute
            ),
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    if !state.policy.message_allowed(&message) {
        state.peer_store.record_policy_rejected(
            now,
            format!(
                "allow_list_enabled={} allowed_peer_count={} denied_peer_count={}",
                !state.policy.allowed_peer_ids.is_empty(),
                state.policy.allowed_peer_ids.len(),
                state.policy.denied_peer_ids.len()
            ),
        );
        return (
            StatusCode::FORBIDDEN,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    let applied = state.peer_store.apply_discovery_message(&message, now);
    state.peer_store.mark_gossip_at(now);
    let response = match message {
        NodeDiscoveryMessage::SnapshotRequest { limit, .. } => {
            Some(state.peer_store.build_snapshot_response(
                now,
                now,
                true,
                Some(state.policy.snapshot_limit(limit.map(usize::from))),
            ))
        }
        NodeDiscoveryMessage::SnapshotResponse { .. }
        | NodeDiscoveryMessage::DescriptorAnnounce { .. } => None,
    };

    (StatusCode::OK, Json(GossipResponse { applied, response })).into_response()
}

async fn status_handler(State(state): State<DiscoveryApiState>) -> Json<DiscoveryStatusResponse> {
    let now = now_secs();
    Json(DiscoveryStatusResponse {
        generated_at: now,
        peer_store: state.peer_store.status(now),
        policy: DiscoveryPolicyStatus {
            max_snapshot_limit: state.policy.max_snapshot_limit,
            gossip_rate_limit_per_minute: state.policy.gossip_rate_limit_per_minute,
            allow_list_enabled: !state.policy.allowed_peer_ids.is_empty(),
            allowed_peer_count: state.policy.allowed_peer_ids.len(),
            denied_peer_count: state.policy.denied_peer_ids.len(),
            snapshot_default_public_only: true,
            private_descriptors_hidden_by_default: true,
        },
        local_capabilities: state.local_capabilities,
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::{NodeCapability, NodeCapacity, NodeDescriptor};
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

    fn signed_descriptor() -> aeronyx_core::protocol::SignedNodeDescriptor {
        let kp = IdentityKeyPair::generate();
        let now = now_secs();
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 64,
            max_bps: None,
            max_pps: None,
        };
        aeronyx_core::protocol::SignedNodeDescriptor::sign(descriptor, &kp).unwrap()
    }

    #[tokio::test]
    async fn test_snapshot_endpoint_returns_snapshot() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/snapshot?limit=10")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_gossip_snapshot_request_returns_response() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now_secs(),
            limit: Some(1),
        })
        .unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_gossip_descriptor_announce_imports_peer() {
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
        let body = serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed_descriptor(),
        })
        .unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn test_status_endpoint_returns_peer_store_status() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_endpoint_returns_local_capability_status() {
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["local_capabilities"]["status"].as_str(),
            Some("ready")
        );
        assert_eq!(
            parsed["local_capabilities"]["chat_relay_configured"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["blind_relay_endpoint_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["advertised_chat_relay_capability"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["capability_config_consistent"].as_bool(),
            Some(true)
        );
    }

    #[tokio::test]
    async fn test_snapshot_endpoint_caps_requested_limit() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let mut policy = DiscoveryApiPolicy::default();
        policy.max_snapshot_limit = 1;
        let app = build_discovery_router(store, policy);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/snapshot?limit=50")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let snapshot: NodeBootstrapSnapshot = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.peers.len(), 1);
    }

    #[tokio::test]
    async fn test_gossip_denies_blocked_descriptor() {
        let store = Arc::new(PeerStore::new());
        let descriptor = signed_descriptor();
        let mut policy = DiscoveryApiPolicy::default();
        policy
            .denied_peer_ids
            .insert(hex::encode(descriptor.node_id()));
        let app = build_discovery_router(Arc::clone(&store), policy);
        let body =
            serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce { descriptor }).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(store.len(), 0);
        let status = store.status(now_secs());
        assert_eq!(status.runtime.policy_rejected, 1);
        assert_eq!(
            status
                .recent_audit_events
                .last()
                .map(|event| event.action.as_str()),
            Some("gossip_policy_rejected")
        );
    }

    #[tokio::test]
    async fn test_gossip_rate_limit_rejects_excess_requests() {
        let store = Arc::new(PeerStore::new());
        let mut policy = DiscoveryApiPolicy::default();
        policy.gossip_rate_limit_per_minute = 1;
        let app = build_discovery_router(Arc::clone(&store), policy);

        for expected_status in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
            let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
                requested_at: now_secs(),
                limit: Some(1),
            })
            .unwrap();
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/api/discovery/gossip")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
        }

        let status = store.status(now_secs());
        assert_eq!(status.runtime.rate_limited, 1);
        assert_eq!(
            status
                .recent_audit_events
                .last()
                .map(|event| event.action.as_str()),
            Some("gossip_rate_limited")
        );
    }
}
