// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Registered `deny_list` submodule and re-exported DenyList + DenyReason.
//   Registered `dns_proxy` so VPN clients can resolve DNS through the gateway.
//
// Last Modified:
//   v1.2.0-DNSProxy - Added VPN gateway DNS proxy
//   v1.1.0-ChatRelay - Added chat_relay submodule
//   v1.0.0-Membership - Added deny_list submodule + traffic_tracker

pub mod chat_relay;
pub mod deny_list;
pub mod dns_proxy;
pub mod handshake;
pub mod ip_pool;
pub mod memchain;
pub mod node_policy;
pub mod routing;
pub mod session;
pub mod traffic_tracker;
pub mod wallet_routes;

// Re-export primary types
pub use chat_relay::{ChatRelayService, derive_node_secret};
pub use deny_list::{DenyList, DenyReason};
pub use dns_proxy::spawn_dns_proxy;
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use node_policy::{NodePolicyEnforcementSnapshot, NodePolicyRuntime, NodePolicySnapshot};
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
pub use wallet_routes::WalletRouteCache;
