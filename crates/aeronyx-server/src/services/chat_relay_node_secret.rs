// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_node_secret.rs
// ============================================
// Version: 1.0.0-NodeSecretDerivation
//
// Creation Reason:
//   [CHAT-NODE-SECRET-DOMAIN 2026-08-28 by Codex] Extract domain-separated
//   relay secret derivation from the relay orchestration service.
//
// Main Functionality:
//   - Derives one stable 32-byte relay secret from an Ed25519 private seed.
//   - Applies the established AeroNyx chat-relay HKDF salt unchanged.
//   - Preserves the existing public helper path through service re-export.
//
// Dependencies:
//   - `hkdf` implements RFC 5869 key derivation.
//   - `sha2` supplies SHA-256 for the established derivation contract.
//   - Relay replay, cursor, backup, and audit domains consume the output only.
//
// Main Logical Flow:
//   1. Treat the Ed25519 private seed as input keying material.
//   2. Extract with the versioned AeroNyx chat-relay salt.
//   3. Expand exactly 32 bytes for private relay-domain capabilities.
//
// Important Note for Next Developer:
//   - The salt is a compatibility and cryptographic domain-separation value.
//   - Changing it makes existing sealed replay and audit state unrecoverable.
//   - Never log, serialize, report, or persist the derived secret directly.
//   - Introduce a versioned migration before changing input or output semantics.
//
// Last Modified:
//   v1.0.0-NodeSecretDerivation - Initial derivation extraction
// ============================================

use hkdf::Hkdf;
use sha2::Sha256;

const CHAT_RELAY_NODE_SECRET_SALT: &[u8] = b"aeronyx-chat-relay-v1";

/// Derives a stable 32-byte node secret from the node's Ed25519 private key.
#[must_use]
pub fn derive_node_secret(ed25519_sk_bytes: &[u8; 32]) -> [u8; 32] {
    let hkdf = Hkdf::<Sha256>::new(Some(CHAT_RELAY_NODE_SECRET_SALT), ed25519_sk_bytes);
    let mut secret = [0_u8; 32];
    hkdf.expand(b"", &mut secret)
        .expect("HKDF expand with 32-byte output always succeeds");
    secret
}
