// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Registered `deny_list` submodule and re-exported DenyList + DenyReason.
//   All other content unchanged.
//
// Last Modified:
//   v1.1.0-ChatRelay - Added chat_relay submodule
//   v1.0.0-Membership - Added deny_list submodule + traffic_tracker

pub mod chat_relay;
pub mod deny_list;
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
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use node_policy::{NodePolicyRuntime, NodePolicySnapshot};
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
pub use wallet_routes::WalletRouteCache;
