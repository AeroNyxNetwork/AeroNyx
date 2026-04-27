//! ============================================
//! File: crates/aeronyx-server/src/services/mod.rs
//! ============================================
//! # Server Services
//!
//! ## Creation Reason
//! Provides business logic services for the AeroNyx server,
//! separated from transport and protocol concerns.
//!
//! ## Modification Reason
//! - Added `memchain` submodule containing the MemPool (in-memory Fact
//!   buffer) and AofWriter (append-only disk persistence) for the
//!   MemChain distributed AI memory ledger.
//! - 🌟 v1.3.0: Added `agent_manager` submodule for OpenClaw AI agent
//!   lifecycle management (install, start, stop, update, uninstall).
//! - 🌟 v1.1.0-ChatRelay: Added `chat_relay` submodule for zero-knowledge
//!   P2P chat relay. `ChatRelayService` manages offline message storage,
//!   encrypted blob cache, TTL cleanup, and expired notification backlog.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`session`]: Session lifecycle management
//! - [`routing`]: Packet routing by virtual IP
//! - [`ip_pool`]: Virtual IP address allocation
//! - [`handshake`]: Handshake processing service
//! - [`memchain`]: 🌟 MemChain storage engine
//! - [`agent_manager`]: 🌟 OpenClaw agent lifecycle (v1.3.0)
//! - [`chat_relay`]: 🌟 Zero-knowledge P2P chat relay (v1.1.0-ChatRelay)
//!
//! ## Service Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Service Layer                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  ┌─────────────────┐   ┌─────────────────────────────────┐ │
//! │  │  HandshakeService│   │      SessionManager            │ │
//! │  │                  │──►│  - Create/destroy sessions     │ │
//! │  │  - Verify sigs   │   │  - Track session state         │ │
//! │  │  - Key exchange  │   │  - Timeout management          │ │
//! │  └─────────────────┘   │  - 🌟 wallet_index lookup      │ │
//! │                         └──────────────┬──────────────────┘ │
//! │                                        │                    │
//! │  ┌─────────────────┐   ┌──────────────▼──────────────────┐ │
//! │  │   IpPoolService │◄──│      RoutingService             │ │
//! │  │                  │   │  - VIP → Session lookup        │ │
//! │  │  - Allocate IPs  │   │  - Session → Endpoint lookup   │ │
//! │  │  - Release IPs   │   │                                │ │
//! │  └─────────────────┘   └─────────────────────────────────┘ │
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │  🌟 MemChain Storage Engine                             ││
//! │  │  ┌─────────────┐   ┌─────────────────────────────────┐ ││
//! │  │  │   MemPool   │──►│      AofWriter                  │ ││
//! │  │  │  (DashMap)  │   │  (.memchain append-only file)   │ ││
//! │  │  └─────────────┘   └─────────────────────────────────┘ ││
//! │  └─────────────────────────────────────────────────────────┘│
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │  🌟 Agent Manager (v1.3.0)                              ││
//! │  │  ┌─────────────────────────────────────────────────────┐││
//! │  │  │  AgentManager — OpenClaw lifecycle management       │││
//! │  │  │  install → start → health check → stop → uninstall │││
//! │  │  └─────────────────────────────────────────────────────┘││
//! │  └─────────────────────────────────────────────────────────┘│
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │  🌟 Chat Relay (v1.1.0-ChatRelay)                       ││
//! │  │  ┌─────────────────────────────────────────────────────┐││
//! │  │  │  ChatRelayService — zero-knowledge P2P messaging    │││
//! │  │  │  store → pull → ack → expire → notify              │││
//! │  │  │  blob upload → download → TTL delete               │││
//! │  │  └─────────────────────────────────────────────────────┘││
//! │  └─────────────────────────────────────────────────────────┘│
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Service Interactions
//! 1. Handshake creates session via SessionManager
//! 2. SessionManager allocates IP via IpPoolService
//! 3. SessionManager registers route via RoutingService
//! 4. Packet handling uses RoutingService for lookups
//! 5. 🌟 Packet handling routes MemChain messages to MemPool
//! 6. 🌟 MemPool persists facts via AofWriter
//! 7. 🌟 CommandHandler delegates agent ops to AgentManager
//! 8. 🌟 HeartbeatReporter reads AgentManager::status() for CMS
//! 9. 🌟 ChatRelay handles ChatRelay/Pull/Ack/Expired MemChain variants
//! 10. 🌟 SessionManager.get_by_wallet() enables O(1) receiver lookup
//!
//! ## ⚠️ Important Note for Next Developer
//! - Services are designed to be testable in isolation
//! - All services use trait abstractions
//! - Thread-safe by design (Send + Sync)
//! - Session cleanup must release IP and routes
//! - MemPool uses DashMap (lock-free); AofWriter needs Mutex wrapper
//! - 🌟 AgentManager uses RwLock for concurrent read (status) / write (install)
//! - 🌟 ChatRelayService uses parking_lot::Mutex<Connection> for SQLite access
//!   Do NOT call SQLite methods while holding another lock (deadlock risk)
//! - 🌟 chat_relay is independent of memchain mode — can run with mode=off
//!
//! ## Last Modified
//! v0.1.0 - Initial services structure
//! v0.2.0 - Added memchain submodule (MemPool + AofWriter)
//! v1.3.0 - 🌟 Added agent_manager submodule
//! v1.1.0-ChatRelay - 🌟 Added chat_relay submodule (ChatRelayService)

pub mod agent_manager;
pub mod chat_relay;
pub mod handshake;
pub mod ip_pool;
pub mod memchain;
pub mod routing;
pub mod session;
pub mod wallet_routes;

// Re-export primary types
pub use agent_manager::AgentManager;
pub use chat_relay::{ChatRelayService, derive_node_secret};
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
pub use wallet_routes::WalletRouteCache;
