// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
//! # Server Services
//!
//! ## Creation Reason
//! Provides business logic services for the AeroNyx server,
//! separated from transport and protocol concerns.
//!
//! ## Modification Reason
//! Added `memchain` submodule containing the MemPool (in-memory Fact
//! buffer) and AofWriter (append-only disk persistence) for the
//! MemChain distributed AI memory ledger.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`session`]: Session lifecycle management
//! - [`routing`]: Packet routing by virtual IP
//! - [`ip_pool`]: Virtual IP address allocation
//! - [`handshake`]: Handshake processing service
//! - [`memchain`]: 🌟 MemChain storage engine (NEW)
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
//! │  └─────────────────┘   └──────────────┬──────────────────┘ │
//! │                                        │                    │
//! │  ┌─────────────────┐   ┌──────────────▼──────────────────┐ │
//! │  │   IpPoolService │◄──│      RoutingService             │ │
//! │  │                  │   │  - VIP → Session lookup        │ │
//! │  │  - Allocate IPs  │   │  - Session → Endpoint lookup   │ │
//! │  │  - Release IPs   │   │                                │ │
//! │  └─────────────────┘   └─────────────────────────────────┘ │
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │  🌟 MemChain Storage Engine (NEW)                       ││
//! │  │  ┌─────────────┐   ┌─────────────────────────────────┐ ││
//! │  │  │   MemPool   │──►│      AofWriter                  │ ││
//! │  │  │  (DashMap)  │   │  (.memchain append-only file)   │ ││
//! │  │  └─────────────┘   └─────────────────────────────────┘ ││
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
//!
//! ## ⚠️ Important Note for Next Developer
//! - Services are designed to be testable in isolation
//! - All services use trait abstractions
//! - Thread-safe by design (Send + Sync)
//! - Session cleanup must release IP and routes
//! - MemPool uses DashMap (lock-free); AofWriter needs Mutex wrapper
//!
//! ## Last Modified
//! v0.1.0 - Initial services structure
//! v0.2.0 - Added memchain submodule (MemPool + AofWriter)

pub mod handshake;
pub mod ip_pool;
pub mod memchain;
pub mod routing;
pub mod session;

// Re-export primary types
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
