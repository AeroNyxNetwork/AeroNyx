// ============================================
// File: crates/aeronyx-mailbox-wire/src/codec.rs
// ============================================
//! Bounded fixed-integer bincode, identical to `aeronyx-core::protocol::codec`.
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] Every mailbox frame uses these exact
//! options; drifting to bincode defaults would silently change the wire.

use bincode::Options;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Whether a decoder accepts bytes after one complete value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrailingBytesPolicy {
    /// Ignore a trailing suffix (legacy shapes only).
    #[allow(dead_code)]
    Allow,
    /// One canonical value must consume the whole input.
    Reject,
}

pub(crate) fn encode_bincode_bounded<T: Serialize>(
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, bincode::Error> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(limit)
        .serialize(value)
}

pub(crate) fn decode_bincode_bounded<T: DeserializeOwned>(
    bytes: &[u8],
    limit: u64,
    trailing_bytes: TrailingBytesPolicy,
) -> Result<T, bincode::Error> {
    let input_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if input_len > limit {
        return Err(Box::new(bincode::ErrorKind::SizeLimit));
    }
    let options = bincode::options().with_fixint_encoding().with_limit(limit);
    match trailing_bytes {
        TrailingBytesPolicy::Allow => options.allow_trailing_bytes().deserialize(bytes),
        TrailingBytesPolicy::Reject => options.reject_trailing_bytes().deserialize(bytes),
    }
}
