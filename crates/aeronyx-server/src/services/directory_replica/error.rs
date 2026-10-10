// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/error.rs
// ============================================
//! # Directory replica store error
//!
//! Owns `DirectoryReplicaStoreError`, the typed failure surface shared by every replica store operation.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::DirectoryCommitmentValidationError;

/// Failures returned by the producer-isolated replica store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryReplicaStoreError {
    /// Filesystem setup failed.
    #[error("directory replica filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// `SQLite` rejected a schema, query, or transaction operation.
    #[error("directory replica sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A protocol object could not be encoded or decoded safely.
    #[error("directory replica codec error: {0}")]
    Codec(String),
    /// A descriptor object did not reproduce its signed commitment.
    #[error("directory replica descriptor error: {0}")]
    Descriptor(String),
    /// A block failed the canonical Directory Chain V1 contract.
    #[error("directory replica block validation error: {0}")]
    Block(#[from] DirectoryCommitmentValidationError),
    /// Durable metadata, chain, index, or evidence is inconsistent.
    #[error("directory replica integrity error: {0}")]
    Integrity(String),
    /// A bounded import request violates the V1 transport contract.
    #[error("directory replica request error: {0}")]
    Request(String),
    /// The producer is durably isolated pending operator review.
    #[error("directory producer is quarantined: {0}")]
    Quarantined(String),
    /// The configured durable permissionless mirror namespace ceiling is full.
    #[error("directory full-node mirror capacity reached")]
    MirrorCapacity,
    /// A public recovery read requested a namespace not retained as a mirror.
    #[error("directory full-node mirror namespace is not retained")]
    MirrorNotRetained,
    /// [MIRROR-CARRIER 2026-07-24 by Codex] The requested producer range is
    /// valid, but this carrier has not retained that height yet. Keep this
    /// distinct from malformed requests so callers may try another carrier
    /// without making protocol-contract failures retryable.
    #[error(
        "directory replica range from height {from_height} is beyond retained tip {tip_height}"
    )]
    RangeNotRetained {
        /// First block height requested by the authenticated peer.
        from_height: u64,
        /// Highest producer block fully audited on this carrier.
        tip_height: u64,
    },
}
