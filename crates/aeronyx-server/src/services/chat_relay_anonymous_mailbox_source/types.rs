// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/types.rs
// ============================================
//! # Source coordinator vocabulary
//!
//! Owns the coarse source error, the durable request phase, the receiver-shared
//! exact target pin and its narrow resolver boundary, and the opaque prepared,
//! result and outbound values exposed to a transport composition root.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};

use crate::api::chat_peer::PeerBlindRelayRequest;
use crate::services::peer_store::PeerStore;

/// Coarse source coordinator failure. No variant carries an opaque request,
/// descriptor, identity, path, or capability value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AnonymousMailboxSourceError {
    #[error("anonymous mailbox source disabled")]
    Disabled,
    #[error("anonymous mailbox source rejected")]
    Rejected,
    #[error("anonymous mailbox source conflict")]
    Conflict,
    #[error("anonymous mailbox source journal unavailable")]
    Unavailable,
    #[error("anonymous mailbox source journal corrupt")]
    Corrupt,
    #[error("anonymous mailbox source result ambiguous")]
    Ambiguous,
}

/// Durable request lifecycle. Armed and ambiguous records may only retry their
/// byte-identical prepared carrier; callers never synthesize a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymousMailboxSourcePhase {
    Prepared,
    Armed,
    Completed,
    Ambiguous,
    Rejected,
}

impl AnonymousMailboxSourcePhase {
    pub(super) fn code(self) -> i64 {
        match self {
            Self::Prepared => 1,
            Self::Armed => 2,
            Self::Completed => 3,
            Self::Ambiguous => 4,
            Self::Rejected => 5,
        }
    }

    pub(super) fn decode(code: i64) -> Result<Self, AnonymousMailboxSourceError> {
        match code {
            1 => Ok(Self::Prepared),
            2 => Ok(Self::Armed),
            3 => Ok(Self::Completed),
            4 => Ok(Self::Ambiguous),
            5 => Ok(Self::Rejected),
            _ => Err(AnonymousMailboxSourceError::Corrupt),
        }
    }
}

/// Receiver-shared exact custody pin. The full descriptor commitment prevents
/// a source from silently selecting a fresher or otherwise different peer.
///
/// [ANONYMOUS-MAILBOX-SOURCE 2026-09-03 by Codex] This type intentionally has
/// no Debug implementation: callers must not accidentally log target metadata.
#[derive(Clone, PartialEq, Eq)]
pub struct ExactAnonymousMailboxTargetPin {
    pub(super) target_node_id: [u8; 32],
    pub(super) descriptor_commitment: DirectoryDescriptorCommitmentV1,
}

impl ExactAnonymousMailboxTargetPin {
    #[must_use]
    pub(crate) const fn new(
        target_node_id: [u8; 32],
        descriptor_commitment: DirectoryDescriptorCommitmentV1,
    ) -> Self {
        Self {
            target_node_id,
            descriptor_commitment,
        }
    }
}

/// Narrow exact-node lookup boundary. The production adapter calls only
/// `PeerStore::get_valid`; no iterator or fallback candidate API is exposed.
pub trait ExactAnonymousMailboxTargetResolver: Send + Sync {
    fn get_valid_exact(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor>;
}

impl ExactAnonymousMailboxTargetResolver for PeerStore {
    fn get_valid_exact(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor> {
        self.get_valid(node_id, now)
    }
}

/// Opaque prepared result exposed to a future transport composition root.
/// It intentionally has no Debug implementation because it retains ciphertext.
pub struct AnonymousMailboxSourcePrepared {
    pub(super) route_id: [u8; 16],
    pub(super) body: Vec<u8>,
    pub(super) phase: AnonymousMailboxSourcePhase,
}

impl AnonymousMailboxSourcePrepared {
    #[must_use]
    pub(crate) const fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    #[must_use]
    pub(crate) fn body(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub(crate) const fn phase(&self) -> AnonymousMailboxSourcePhase {
        self.phase
    }
}

/// Durable terminal result. The bytes are canonical source-sealed terminal
/// response bytes, never a peer's outer HTTP response or receipt.
pub enum AnonymousMailboxSourceResult {
    Prepared,
    Armed,
    Completed(Vec<u8>),
    Ambiguous,
    Rejected,
}

/// Exact outbound work released only after the durable record has been
/// revalidated against the current descriptor. This type intentionally omits
/// Debug so a target endpoint or opaque carrier cannot reach diagnostics.
pub struct AnonymousMailboxSourceOutbound {
    pub(super) route_id: [u8; 16],
    pub(super) target_node_id: [u8; 32],
    pub(super) url: reqwest::Url,
    pub(super) body: Vec<u8>,
    pub(super) request: PeerBlindRelayRequest,
}

impl AnonymousMailboxSourceOutbound {
    #[must_use]
    pub(crate) const fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    #[must_use]
    pub(crate) fn target_node_id(&self) -> &[u8; 32] {
        &self.target_node_id
    }

    #[must_use]
    pub(crate) fn url(&self) -> &reqwest::Url {
        &self.url
    }

    #[must_use]
    pub(crate) fn body(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub(crate) fn request(&self) -> &PeerBlindRelayRequest {
        &self.request
    }
}
