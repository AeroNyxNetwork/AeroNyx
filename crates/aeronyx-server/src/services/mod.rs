// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
//! # Server Services
//!
//! ## Creation Reason
//! Provides business logic services for the AeroNyx server,
//! separated from transport and protocol concerns.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`session`]: Session lifecycle management
//! - [`routing`]: Packet routing by virtual IP
//! - [`ip_pool`]: Virtual IP address allocation
//! - [`handshake`]: Handshake processing service
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
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Service Interactions
//! 1. Handshake creates session via SessionManager
//! 2. SessionManager allocates IP via IpPoolService
//! 3. SessionManager registers route via RoutingService
//! 4. Packet handling uses RoutingService for lookups
//!
//! ## ⚠️ Important Note for Next Developer
//! - Services are designed to be testable in isolation
//! - All services use trait abstractions
//! - Thread-safe by design (Send + Sync)
//! - Session cleanup must release IP and routes
//!
//! ## Last Modified
//! v0.1.0 - Initial services structure

pub mod handshake;
pub mod ip_pool;
pub mod routing;
pub mod session;

// Re-export primary types
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use routing::RoutingService;
pub use session::{Session, SessionManager, SessionState};
