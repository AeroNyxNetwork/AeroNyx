// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/codec.rs
// ============================================
//! # Bounded wire codec
//!
//! Owns `encode_memchain` / `decode_memchain`: the `0xAE`-prefixed, 2 MiB
//! bounded bincode framing, the legacy trailing-byte policy, and the canonical
//! outer-byte gate for the anonymous mailbox route variants.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

use super::message::MemChainMessage;
use super::{MAX_MEMCHAIN_PAYLOAD_BYTES, MEMCHAIN_MAGIC};

// ============================================
// Encode / Decode helpers
// ============================================

/// Encodes a `MemChainMessage` into a bounded byte vector with the `0xAE` prefix.
///
/// # Errors
/// Returns `bincode::Error` if serialization fails or the payload exceeds the
/// 2 MiB MemChain protocol ceiling.
pub fn encode_memchain(msg: &MemChainMessage) -> std::result::Result<Vec<u8>, bincode::Error> {
    let payload = encode_bincode_bounded(msg, MAX_MEMCHAIN_PAYLOAD_BYTES)?;
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(MEMCHAIN_MAGIC);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decodes a `MemChainMessage` from a plaintext slice whose first byte
/// (`MEMCHAIN_MAGIC`) has **already been verified and stripped** by the caller.
///
/// Rejects payloads that would require allocating more than
/// `MAX_MEMCHAIN_PAYLOAD_BYTES` (2 MB).
///
/// # Errors
/// Returns `bincode::Error` for malformed or oversized payloads.
pub fn decode_memchain(payload: &[u8]) -> std::result::Result<MemChainMessage, bincode::Error> {
    let message = decode_bincode_bounded(
        payload,
        MAX_MEMCHAIN_PAYLOAD_BYTES,
        TrailingBytesPolicy::Allow,
    )?;

    // [ANONYMOUS-MAILBOX-CANONICAL-OUTER 2026-09-02 by Codex] Legacy
    // MemChain variants retain their deployed trailing-byte compatibility,
    // but the new mailbox carriers have no legacy non-canonical form. Requiring
    // the exact canonical bytes prevents an authenticated request from being
    // padded into a different outer wire value with the same retry commitment.
    if matches!(
        &message,
        MemChainMessage::AnonymousMailboxRouteV1(_)
            | MemChainMessage::AnonymousMailboxRouteResponseV1(_)
    ) {
        let canonical = encode_bincode_bounded(&message, MAX_MEMCHAIN_PAYLOAD_BYTES)?;
        if canonical.as_slice() != payload {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "anonymous mailbox outer frame is non-canonical".to_string(),
            )));
        }
    }

    Ok(message)
}

#[cfg(test)]
mod tests;
