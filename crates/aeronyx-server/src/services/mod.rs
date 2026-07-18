// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Registered `deny_list` submodule and re-exported DenyList + DenyReason.
//   Registered `dns_proxy` so AeroNyx clients can resolve DNS through the gateway.
//   Registered `peer_store` for Phase 1 decentralized node discovery.
//   Re-exported `PeerStoreImportReport` for Phase 2 bootstrap snapshot loading.
//   Re-exported PeerStoreStatus for nodeboard discovery status.
//
// Last Modified:
//   v0.7.0-DirectoryReplicaStatus - Re-exported replica status/runtime types
//   v0.6.0-DirectoryReplicaStore - Registered producer-isolated remote replicas
//   v0.5.0-DirectorySyncReads - Re-exported audit-gated Directory Chain pages
//   v0.4.0-DirectoryChainStore - Registered transactional local directory ledger
//   v0.3.0-DiscoveryStatus - Re-exported PeerStoreStatus
//   v0.2.0-DiscoveryPhase2 - Re-exported PeerStoreImportReport
//   v0.1.0-DiscoveryPhase1 - Added peer_store submodule
//   v1.2.0-DNSProxy - Added VPN gateway DNS proxy
//   v1.1.0-ChatRelay - Added chat_relay submodule
//   v1.0.0-Membership - Added deny_list submodule + traffic_tracker

pub mod chat_relay;
pub mod deny_list;
pub mod directory_chain;
pub mod directory_replica;
pub mod dns_proxy;
pub mod handshake;
pub mod ip_pool;
pub mod memchain;
pub mod node_policy;
pub mod onion_keys;
pub mod peer_store;
pub mod routing;
pub mod session;
pub mod traffic_tracker;
pub mod wallet_routes;

// Re-export primary types
pub use chat_relay::{derive_node_secret, ChatRelayService};
pub use deny_list::{DenyList, DenyReason};
pub use directory_chain::{
    DirectoryChainAppendReport, DirectoryChainAudit, DirectoryChainPage, DirectoryChainStore,
    DirectoryChainStoreError,
};
pub use directory_replica::{
    DirectoryReplicaAudit, DirectoryReplicaImportReport, DirectoryReplicaProducerSnapshot,
    DirectoryReplicaStore, DirectoryReplicaStoreError, DirectoryReplicaStoreSnapshot,
    DirectoryReplicaSyncObservation, DirectoryReplicaSyncRuntime, DirectoryReplicaTip,
};
pub use dns_proxy::spawn_dns_proxy;
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use node_policy::{
    NodePolicyEnforcementSnapshot, NodePolicyPlacementSnapshot, NodePolicyRuntime,
    NodePolicySnapshot,
};
pub use peer_store::{
    PeerStore, PeerStoreError, PeerStoreImportReport, PeerStoreSnapshot, PeerStoreStatus,
};
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
pub use wallet_routes::WalletRouteCache;
