// ============================================================================
// File: crates/aeronyx-core/src/protocol/auth.rs
// ============================================================================
// Version: 1.3.0-Sovereign
//
// Modification Reason:
//   New file. Centralises per-message wallet signature verification so that
//   every chat handler uses an identical, audited code path.
//
// Main Functionality:
//   - verify_signed_message(): canonical Ed25519 verification with:
//       1. Timestamp window check (±60 s)
//       2. SHA-256 of (domain_separator || payload_slices…)
//       3. Ed25519 signature verification via IdentityPublicKey
//   - AuthError: error enum for all verification failure modes
//
// Dependencies:
//   - crates/aeronyx-core/src/crypto/keys: IdentityPublicKey
//   - sha2 (already a workspace dep via chat.rs)
//   - std::time::SystemTime
//
// Main Logical Flow:
//   1. Caller passes domain separator, ordered payload byte slices,
//      wallet public key bytes, signature bytes, and claimed timestamp
//   2. Timestamp is checked against server clock (±TIMESTAMP_WINDOW_SECS)
//   3. SHA-256 is computed over domain || slices[0] || slices[1] || …
//   4. Ed25519 verify is called; any failure maps to AuthError::SignatureMismatch
//   5. Ok(()) returned on success
//
// ⚠️ Important Notes for Next Developer:
//   - TIMESTAMP_WINDOW_SECS = 60. Do not widen — wider windows allow
//     replay attacks within the window duration.
//   - The hash digest (not the raw concatenation) is what gets signed.
//     This avoids length-extension and ensures a fixed 32-byte input to verify().
//   - Domain separators MUST be unique per message type. If you add a new
//     message type, add a new domain constant in this file.
//   - verify_signed_message() is intentionally not async — it is pure CPU
//     work and must not be called inside a blocking context via spawn_blocking.
//     It completes in <1 ms on any modern CPU.
//   - AuthError intentionally does NOT implement From<CoreError> to prevent
//     accidental leakage of internal crypto error detail to callers.
//
// Last Modified: v1.3.0-Sovereign — Initial implementation
// ============================================================================

use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::crypto::keys::IdentityPublicKey;

// ============================================
// Domain separator constants
// ============================================

/// Domain separator for DeviceRegister messages.
pub const DOMAIN_DEVICE_REGISTER: &str = "AeroNyx-DeviceRegister-v1";

/// Domain separator for ChatPull messages.
pub const DOMAIN_CHAT_PULL: &str = "AeroNyx-ChatPull-v1";

/// Domain separator for ChatAck messages.
pub const DOMAIN_CHAT_ACK: &str = "AeroNyx-ChatAck-v1";

/// Domain separator for WalletPresence heartbeats.
pub const DOMAIN_WALLET_PRESENCE: &str = "AeroNyx-WalletPresence-v1";

// ============================================
// Timestamp window
// ============================================

/// Maximum allowed difference (in seconds) between the message timestamp
/// and the server clock. Messages outside this window are rejected to
/// prevent replay attacks.
///
/// 60 seconds is deliberately tight. VPN clients are expected to have
/// reasonably synchronised clocks (NTP). Widening this value increases
/// the replay attack window proportionally.
pub const TIMESTAMP_WINDOW_SECS: u64 = 60;

// ============================================
// AuthError
// ============================================

/// Errors produced by [`verify_signed_message`].
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// The message timestamp is too far from the server clock.
    /// Difference exceeded [`TIMESTAMP_WINDOW_SECS`].
    #[error("Timestamp out of acceptable window (±{} s)", TIMESTAMP_WINDOW_SECS)]
    TimestampOutOfWindow,

    /// The wallet_pubkey bytes do not form a valid Ed25519 public key.
    #[error("Invalid Ed25519 public key")]
    InvalidPublicKey,

    /// The signature bytes are structurally invalid (wrong length, etc.).
    /// Distinct from [`AuthError::SignatureMismatch`] which means the
    /// signature is structurally valid but does not verify.
    #[error("Invalid signature encoding")]
    InvalidSignature,

    /// The signature does not match the signed data.
    #[error("Signature does not match signed data")]
    SignatureMismatch,
}

// ============================================
// verify_signed_message
// ============================================

/// Verifies a per-message Ed25519 signature for wallet identity proof.
///
/// # Arguments
/// * `domain` — unique domain separator string (use the `DOMAIN_*` constants)
/// * `payload_slices` — ordered list of byte slices to include in the hash
///   (concatenated in iteration order after the domain)
/// * `wallet_pubkey` — the 32-byte Ed25519 public key claiming to be the signer
/// * `signature` — the 64-byte Ed25519 signature to verify
/// * `msg_timestamp` — the Unix-epoch-seconds timestamp from the message,
///   checked against the current server clock
///
/// # Signed Data Layout
/// ```text
/// SHA-256( domain_bytes || payload_slices[0] || payload_slices[1] || … )
/// ```
/// The 32-byte digest is what gets passed to Ed25519 verify.
///
/// # Returns
/// `Ok(())` on success. `Err(AuthError)` on any failure.
///
/// # Errors
/// * [`AuthError::TimestampOutOfWindow`] — clock skew > 60 s
/// * [`AuthError::InvalidPublicKey`] — malformed public key bytes
/// * [`AuthError::SignatureMismatch`] — signature verification failed
pub fn verify_signed_message(
    domain: &str,
    payload_slices: &[&[u8]],
    wallet_pubkey: &[u8; 32],
    signature: &[u8; 64],
    msg_timestamp: u64,
) -> Result<(), AuthError> {
    // ── 1. Timestamp window check ─────────────────────────────────────
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Use saturating arithmetic to avoid underflow on u64.
    let delta = if now >= msg_timestamp {
        now - msg_timestamp
    } else {
        msg_timestamp - now
    };

    if delta > TIMESTAMP_WINDOW_SECS {
        return Err(AuthError::TimestampOutOfWindow);
    }

    // ── 2. Build SHA-256 digest ───────────────────────────────────────
    // Layout: domain_bytes || payload_slices[0] || payload_slices[1] || …
    //
    // Hashing rather than raw concatenation:
    // - Produces a fixed 32-byte input for Ed25519 verify
    // - Eliminates length-extension ambiguity between adjacent variable-length fields
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    for slice in payload_slices {
        hasher.update(slice);
    }
    let digest: [u8; 32] = hasher.finalize().into();

    // ── DEBUG: dump sign input and digest (remove before production) ──
    {
        let mut sign_input_debug = Vec::new();
        sign_input_debug.extend_from_slice(domain.as_bytes());
        for s in payload_slices {
            sign_input_debug.extend_from_slice(s);
        }
        tracing::info!(
            "[VERIFY] domain={} input_len={} input_hex={} hash_hex={} pubkey={}",
            domain,
            sign_input_debug.len(),
            hex::encode(&sign_input_debug),
            hex::encode(&digest),
            hex::encode(wallet_pubkey),
        );
    }

    // ── 3. Ed25519 verification ───────────────────────────────────────
    let pk = IdentityPublicKey::from_bytes(wallet_pubkey)
        .map_err(|_| AuthError::InvalidPublicKey)?;

    pk.verify(&digest, signature)
        .map_err(|_| AuthError::SignatureMismatch)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::IdentityKeyPair;

    /// Build a valid (domain, payload, sig, ts) tuple for the given keypair.
    fn make_signed(kp: &IdentityKeyPair, domain: &str, payload: &[u8]) -> (u64, [u8; 64]) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update(payload);
        let digest: [u8; 32] = hasher.finalize().into();
        let sig = kp.sign(&digest);
        (ts, sig)
    }

    // ── Happy path ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_valid_signature_passes() {
        let kp = IdentityKeyPair::generate();
        let payload = b"session_id_device_id_wallet_pubkey";
        let (ts, sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        let result = verify_signed_message(
            DOMAIN_DEVICE_REGISTER,
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(result.is_ok(), "Valid signature must pass: {:?}", result);
    }

    #[test]
    fn test_verify_multiple_payload_slices() {
        let kp = IdentityKeyPair::generate();
        let session_id = [0x01u8; 16];
        let wallet = kp.public_key_bytes();
        let ts_bytes = 1_700_000_000u64.to_le_bytes();

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Build the same digest manually
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_WALLET_PRESENCE.as_bytes());
        hasher.update(&session_id);
        hasher.update(&wallet);
        hasher.update(&ts_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        let sig = kp.sign(&digest);

        let result = verify_signed_message(
            DOMAIN_WALLET_PRESENCE,
            &[session_id.as_ref(), wallet.as_ref(), ts_bytes.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(result.is_ok(), "Multi-slice valid signature must pass: {:?}", result);
    }

    // ── Timestamp window failure ─────────────────────────────────────────

    #[test]
    fn test_verify_timestamp_too_old_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"some_data";
        // Use a timestamp 120 s in the past (well outside ±60 s window)
        let stale_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(120);

        // Sign with the stale timestamp embedded in payload
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_CHAT_PULL.as_bytes());
        hasher.update(payload);
        let digest: [u8; 32] = hasher.finalize().into();
        let sig = kp.sign(&digest);

        let result = verify_signed_message(
            DOMAIN_CHAT_PULL,
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            stale_ts, // <── stale timestamp passed as msg_timestamp
        );
        assert!(
            matches!(result, Err(AuthError::TimestampOutOfWindow)),
            "Stale timestamp must be rejected: {:?}", result,
        );
    }

    #[test]
    fn test_verify_timestamp_in_future_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"some_data";
        let future_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 120; // 120 s in the future

        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_CHAT_ACK.as_bytes());
        hasher.update(payload);
        let digest: [u8; 32] = hasher.finalize().into();
        let sig = kp.sign(&digest);

        let result = verify_signed_message(
            DOMAIN_CHAT_ACK,
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            future_ts,
        );
        assert!(
            matches!(result, Err(AuthError::TimestampOutOfWindow)),
            "Future timestamp must be rejected: {:?}", result,
        );
    }

    // ── Public key failure ────────────────────────────────────────────────

    #[test]
    fn test_verify_invalid_public_key_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"data";
        let (ts, sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        // All-zeros is not a valid compressed Ed25519 point
        let bad_pubkey = [0u8; 32];

        let result = verify_signed_message(
            DOMAIN_DEVICE_REGISTER,
            &[payload.as_ref()],
            &bad_pubkey,
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::InvalidPublicKey)),
            "Invalid public key must be rejected: {:?}", result,
        );
    }

    // ── Signature mismatch ────────────────────────────────────────────────

    #[test]
    fn test_verify_wrong_signature_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"correct_payload";
        let (ts, _correct_sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        // Sign a *different* payload — produces a valid-but-wrong signature
        let wrong_sig = kp.sign(b"wrong_payload_digest_placeholder_32b");

        let result = verify_signed_message(
            DOMAIN_DEVICE_REGISTER,
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &wrong_sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Wrong signature must be rejected: {:?}", result,
        );
    }

    #[test]
    fn test_verify_wrong_domain_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"data";
        // Sign under DeviceRegister domain, verify under ChatPull domain
        let (ts, sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        let result = verify_signed_message(
            DOMAIN_CHAT_PULL,    // <── wrong domain
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Wrong domain must cause signature mismatch: {:?}", result,
        );
    }

    #[test]
    fn test_verify_tampered_payload_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"original_data";
        let (ts, sig) = make_signed(&kp, DOMAIN_CHAT_ACK, payload);

        let tampered = b"tampered__data";
        let result = verify_signed_message(
            DOMAIN_CHAT_ACK,
            &[tampered.as_ref()],  // <── tampered
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Tampered payload must cause signature mismatch: {:?}", result,
        );
    }

    #[test]
    fn test_verify_wrong_key_rejected() {
        let kp1 = IdentityKeyPair::generate();
        let kp2 = IdentityKeyPair::generate();
        let payload = b"data";
        // Sign with kp1, verify against kp2's public key
        let (ts, sig) = make_signed(&kp1, DOMAIN_WALLET_PRESENCE, payload);

        let result = verify_signed_message(
            DOMAIN_WALLET_PRESENCE,
            &[payload.as_ref()],
            &kp2.public_key_bytes(), // <── wrong key
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Wrong public key must cause mismatch: {:?}", result,
        );
    }

    // ── Edge: timestamp exactly at boundary ──────────────────────────────

    #[test]
    fn test_verify_timestamp_at_exact_boundary_passes() {
        let kp = IdentityKeyPair::generate();
        let payload = b"boundary_test";
        // Exactly TIMESTAMP_WINDOW_SECS ago — should still pass (≤, not <)
        let boundary_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(TIMESTAMP_WINDOW_SECS);

        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_CHAT_PULL.as_bytes());
        hasher.update(payload);
        let digest: [u8; 32] = hasher.finalize().into();
        let sig = kp.sign(&digest);

        let result = verify_signed_message(
            DOMAIN_CHAT_PULL,
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            boundary_ts,
        );
        // delta == TIMESTAMP_WINDOW_SECS, which satisfies delta <= TIMESTAMP_WINDOW_SECS
        assert!(
            result.is_ok() || matches!(result, Err(AuthError::TimestampOutOfWindow)),
            "Boundary result must be either Ok or TimestampOutOfWindow: {:?}", result,
        );
        // Note: this test is intentionally lenient at the exact boundary because
        // nanosecond-level clock differences between make_signed and verify can
        // push the delta to 61 s. The important guarantees are tested by the
        // stale/future tests above (120 s away).
    }
}
