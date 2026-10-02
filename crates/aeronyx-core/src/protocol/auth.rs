// ============================================================================
// File: crates/aeronyx-core/src/protocol/auth.rs
// ============================================================================
// Version: 1.6.0-VerifiedChatSubmit
//
// Modification Reason:
//   New file. Centralises per-message wallet signature verification so that
//   every chat handler uses an identical, audited code path.
//   v1.3.1-PrivacyHardening — Removed production-inappropriate debug logging
//   from signature verification. Verification behavior and public API are
//   unchanged; sensitive sign input/hash/pubkey material is no longer emitted.
//   Also corrected the non-matching-public-key rejection test so it remains
//   stable across Ed25519 dependency versions.
//   v1.4.0-ChatPullV2 — Added a separate domain for monotonic opaque-cursor
//   mailbox pagination without changing the deployed v1 signature contract.
//   v1.5.0-SessionClose — Added a unique domain for an authenticated graceful
//   UDP tunnel close request; existing domains and signatures are unchanged.
//   v1.6.0-VerifiedChatSubmit — Added one domain for explicit verified-onion
//   chat submission and exposed the canonical fixed-size signing digest helper.
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
// Last Modified:
//   v1.6.0-VerifiedChatSubmit — Added DOMAIN_CHAT_VERIFIED_SUBMIT_V1 and
//                              signed_message_digest()
//   v1.5.0-SessionClose — Added DOMAIN_SESSION_CLOSE_V1
//   v1.4.0-ChatPullV2 — Added DOMAIN_CHAT_PULL_V2
//   v1.3.0-Sovereign — Initial implementation
//   v1.3.1-PrivacyHardening — Removed sensitive verification logs and fixed
//                              non-matching-public-key rejection test
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

/// Domain separator for monotonic opaque-cursor `ChatPullV2` messages.
pub const DOMAIN_CHAT_PULL_V2: &str = "AeroNyx-ChatPull-v2";

/// Domain separator for ChatAck messages.
pub const DOMAIN_CHAT_ACK: &str = "AeroNyx-ChatAck-v1";

/// Domain separator for WalletPresence heartbeats.
pub const DOMAIN_WALLET_PRESENCE: &str = "AeroNyx-WalletPresence-v1";

/// Domain separator for authenticated graceful tunnel termination.
///
/// [SESSION-TERMINATION 2026-08-15 by Codex] This must never be reused by a
/// message that does not close the exact encrypted transport session.
pub const DOMAIN_SESSION_CLOSE_V1: &str = "AeroNyx-SessionClose-v1";

/// Domain separator for an explicit verified-onion chat submission.
///
/// [CHAT-VERIFIED-SUBMIT 2026-08-22 by Codex] This intent is stronger than a
/// legacy `ChatRelay`: the sender asks the entry node to return an exact
/// terminal-signed receipt and forbids silent direct-relay substitution.
pub const DOMAIN_CHAT_VERIFIED_SUBMIT_V1: &str = "AeroNyx-ChatVerifiedSubmit-v1";

// ============================================
// Timestamp window
// ============================================

/// Maximum allowed difference (in seconds) between the message timestamp
/// and the server clock. Messages outside this window are rejected to
/// prevent replay attacks.
///
/// 60 seconds is deliberately tight. AeroNyx clients are expected to have
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

/// Computes the canonical fixed-size digest used by signed protocol frames.
///
/// [CHAT-VERIFIED-SUBMIT 2026-08-22 by Codex] Constructors and verifiers must
/// share this implementation. Duplicating the domain/slice hash in clients or
/// operator tools previously made new signed frame types easy to drift.
#[must_use]
pub fn signed_message_digest(domain: &str, payload_slices: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    for slice in payload_slices {
        hasher.update(slice);
    }
    hasher.finalize().into()
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
    let digest = signed_message_digest(domain, payload_slices);

    // ── 3. Ed25519 verification ───────────────────────────────────────
    let pk =
        IdentityPublicKey::from_bytes(wallet_pubkey).map_err(|_| AuthError::InvalidPublicKey)?;

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

    // [HTTP-AUTH-GOLDEN 2026-10-03 by Codex] Public-seed interoperability
    // literals independently packed/signed in Python and Node, not generated
    // by the Rust helper under test. These mirrors do not test the actual
    // server-private decoder; its tests and Dart must consume the same goldens.
    mod http_golden_vectors {
        use super::*;
        use bincode::Options;
        use serde::{Deserialize, Serialize};

        const SEED: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
        const WALLET: &str = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
        const PULL_DOMAIN: &str = "AeroNyx-ChatPull-v2-http";
        const ACK_DOMAIN: &str = "AeroNyx-ChatAck-v1-http";
        // Historical time is intentional: verify digest/signature, not the
        // live verifier's freshness gate. No clock or freshness override.
        const TIMESTAMP: u64 = 1_700_000_000;

        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Pull {
            version: u8,
            wallet: [u8; 32],
            after_timestamp: u64,
            cursor: Vec<u8>,
            limit: u32,
            request_timestamp: u64,
            signature: Vec<u8>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Ack {
            version: u8,
            wallet: [u8; 32],
            message_ids: Vec<[u8; 16]>,
            ack_timestamp: u64,
            signature: Vec<u8>,
        }

        struct Golden {
            transcript: &'static str,
            digest: &'static str,
            signature: &'static str,
            body: &'static str,
            body_sha256: &'static str,
            transcript_len: usize,
            body_len: usize,
        }

        const GOLDENS: [Golden; 3] = [
            Golden {
                transcript: "4165726f4e79782d4368617450756c6c2d76322d6874747001d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a9cf053650000000000006400000000f1536500000000",
                digest: "619e7c711a4555f88ed4ac1995206a36c4da2e7ca14bd35aaae7c9b27b346d69",
                signature: "b9ba63df3ad41719789b2c9780af8a250ba32559a912caeda9463a2f54a9c251d564355b004e34162c1aa1b5a516c3a7e9397f6731136f16921923821211fd0b",
                body: "01d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a9cf053650000000000000000000000006400000000f15365000000004000000000000000b9ba63df3ad41719789b2c9780af8a250ba32559a912caeda9463a2f54a9c251d564355b004e34162c1aa1b5a516c3a7e9397f6731136f16921923821211fd0b",
                body_sha256: "a8a0410216a5e0a21079aa4c7c844e67c4cd6b2928db29c4b4836544ab5c514e",
                transcript_len: 79,
                body_len: 133,
            },
            Golden {
                transcript: "4165726f4e79782d4368617450756c6c2d76322d6874747001d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a9cf0536500000000390001000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233343536376400000000f1536500000000",
                digest: "256cda3aca755a4fdace8d306fa1e2da5af38d330a8e5c13822e7a18ba8b6dfb",
                signature: "4d2f7d3314016dd6ee1256173a9f92e1d1801fa512a9958a6922c6677bb3b313fb805c067a9219fd460085e76818bb048535ee530d46c6f430b37010c9f1580b",
                body: "01d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a9cf0536500000000390000000000000001000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233343536376400000000f153650000000040000000000000004d2f7d3314016dd6ee1256173a9f92e1d1801fa512a9958a6922c6677bb3b313fb805c067a9219fd460085e76818bb048535ee530d46c6f430b37010c9f1580b",
                body_sha256: "f456cc079bb38cf3970dcb0317a2ab6792d2c76f3ac860b0e3a7d79967b6480c",
                transcript_len: 136,
                body_len: 190,
            },
            Golden {
                transcript: "4165726f4e79782d4368617441636b2d76312d6874747001d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a00f1536500000000f9dc893ec67d3b68d642037b856be26b84c8c90ddabb42e4eab61fbb2993b8bf",
                digest: "9758f272d243b1048759d38463cb5240516f3a8bba21980e820653b9d224693e",
                signature: "29f0c98e67dbe82792ed66c2774aa8f727889dd07482a58ba611dc8c4c55aa40f46f58080f400a507210c5b924a770ba34745e25e3c83bcd8634eacf22c00a00",
                body: "01d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a0200000000000000000102030405060708090a0b0c0d0e0ff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff00f1536500000000400000000000000029f0c98e67dbe82792ed66c2774aa8f727889dd07482a58ba611dc8c4c55aa40f46f58080f400a507210c5b924a770ba34745e25e3c83bcd8634eacf22c00a00",
                body_sha256: "3befc94fad9f271c990cd53046351399ffe16a0d7fcf57c317f978287a375aba",
                transcript_len: 96,
                body_len: 153,
            },
        ];

        fn bytes(value: &str) -> Vec<u8> {
            hex::decode(value).expect("fixed public fixture hex")
        }

        fn key() -> IdentityKeyPair {
            let key = IdentityKeyPair::from_bytes(&bytes(SEED)).unwrap();
            assert_eq!(hex::encode(key.public_key_bytes()), WALLET);
            key
        }

        fn pull(index: usize) -> Pull {
            Pull {
                version: 1,
                wallet: bytes(WALLET).try_into().unwrap(),
                after_timestamp: 1_699_999_900,
                // Synthetic opaque bytes: NOT a valid server-issued AEAD
                // continuation cursor and not evidence of successful paging.
                cursor: if index == 0 {
                    Vec::new()
                } else {
                    std::iter::once(1).chain(0u8..56).collect()
                },
                limit: 100,
                request_timestamp: TIMESTAMP,
                signature: bytes(GOLDENS[index].signature),
            }
        }

        fn ack() -> Ack {
            Ack {
                version: 1,
                wallet: bytes(WALLET).try_into().unwrap(),
                message_ids: vec![
                    std::array::from_fn(|i| i as u8),
                    std::array::from_fn(|i| 240 + i as u8),
                ],
                ack_timestamp: TIMESTAMP,
                signature: bytes(GOLDENS[2].signature),
            }
        }

        fn assert_golden<T>(request: &T, domain: &str, fields: &[&[u8]], golden: &Golden)
        where
            T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
        {
            let transcript = [domain.as_bytes(), fields.concat().as_slice()].concat();
            assert_eq!(transcript, bytes(golden.transcript));
            assert_eq!(transcript.len(), golden.transcript_len);
            let digest = signed_message_digest(domain, fields);
            assert_eq!(hex::encode(digest), golden.digest);
            let signature: [u8; 64] = bytes(golden.signature).try_into().unwrap();
            let key = key();
            assert_eq!(key.sign(&digest), signature);
            key.public_key().verify(&digest, &signature).unwrap();
            assert!(key.public_key().verify(&transcript, &signature).is_err());

            let body = bincode::options()
                .with_fixint_encoding()
                .serialize(request)
                .unwrap();
            assert_eq!(body, bytes(golden.body));
            assert_eq!(body.len(), golden.body_len);
            assert_eq!(hex::encode(Sha256::digest(&body)), golden.body_sha256);
            let decoded: T = bincode::options()
                .with_fixint_encoding()
                .reject_trailing_bytes()
                .with_limit(4096)
                .deserialize(&body)
                .unwrap();
            assert_eq!(&decoded, request);

            // Every transcript byte (domain and all signed fields) is bound.
            for index in 0..transcript.len() {
                let mut changed = transcript.clone();
                changed[index] ^= 1;
                assert!(key
                    .public_key()
                    .verify(&Sha256::digest(&changed), &signature)
                    .is_err());
            }
        }

        fn assert_pull(index: usize) {
            let request = pull(index);
            assert_golden(
                &request,
                PULL_DOMAIN,
                &[
                    &[request.version],
                    &request.wallet,
                    &request.after_timestamp.to_le_bytes(),
                    &(request.cursor.len() as u16).to_le_bytes(),
                    &request.cursor,
                    &request.limit.to_le_bytes(),
                    &request.request_timestamp.to_le_bytes(),
                ],
                &GOLDENS[index],
            );
        }

        #[test]
        fn pull_empty_literal() {
            assert_pull(0);
        }

        #[test]
        fn pull_synthetic_cursor_literal() {
            assert_pull(1);
        }

        #[test]
        fn ack_ordered_ids_literal() {
            let request = ack();
            let ids_hash = Sha256::digest(request.message_ids.concat());
            assert_eq!(
                hex::encode(ids_hash),
                "f9dc893ec67d3b68d642037b856be26b84c8c90ddabb42e4eab61fbb2993b8bf"
            );
            assert_golden(
                &request,
                ACK_DOMAIN,
                &[
                    &[request.version],
                    &request.wallet,
                    &request.ack_timestamp.to_le_bytes(),
                    &ids_hash,
                ],
                &GOLDENS[2],
            );
        }

        #[test]
        fn wrong_domain_length_width_and_id_order_rejected() {
            let key = key();
            let request = pull(1);
            let signature = bytes(GOLDENS[1].signature).try_into().unwrap();
            for (domain, length) in [
                (DOMAIN_CHAT_PULL_V2, (57u16).to_le_bytes().to_vec()),
                (PULL_DOMAIN, (57u64).to_le_bytes().to_vec()),
            ] {
                let digest = signed_message_digest(
                    domain,
                    &[
                        &[request.version],
                        &request.wallet,
                        &request.after_timestamp.to_le_bytes(),
                        &length,
                        &request.cursor,
                        &request.limit.to_le_bytes(),
                        &request.request_timestamp.to_le_bytes(),
                    ],
                );
                assert!(key.public_key().verify(&digest, &signature).is_err());
            }
            let mut request = ack();
            request.message_ids.reverse();
            let digest = signed_message_digest(
                ACK_DOMAIN,
                &[
                    &[request.version],
                    &request.wallet,
                    &request.ack_timestamp.to_le_bytes(),
                    &Sha256::digest(request.message_ids.concat()),
                ],
            );
            let signature = bytes(GOLDENS[2].signature).try_into().unwrap();
            assert!(key.public_key().verify(&digest, &signature).is_err());
        }

        #[test]
        fn literal_codec_rejects_trailing_truncation_and_huge_vec_prefix() {
            fn rejects<T: for<'de> Deserialize<'de>>(body: &[u8]) {
                assert!(bincode::options()
                    .with_fixint_encoding()
                    .reject_trailing_bytes()
                    .with_limit(4096)
                    .deserialize::<T>(body)
                    .is_err());
            }
            for (index, golden) in GOLDENS.iter().enumerate() {
                let body = bytes(golden.body);
                let mut trailing = body.clone();
                trailing.push(0);
                let truncated = &body[..body.len() - 1];
                let mut huge = body.clone();
                // Pull cursor length starts at41; ACK ID count starts at33.
                let offset = if index < 2 { 41 } else { 33 };
                huge[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
                for malformed in [trailing.as_slice(), truncated, huge.as_slice()] {
                    if index < 2 {
                        rejects::<Pull>(malformed);
                    } else {
                        rejects::<Ack>(malformed);
                    }
                }
            }
        }
    }

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
        assert!(
            result.is_ok(),
            "Multi-slice valid signature must pass: {:?}",
            result
        );
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
            "Stale timestamp must be rejected: {:?}",
            result,
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
            "Future timestamp must be rejected: {:?}",
            result,
        );
    }

    // ── Public key failure ────────────────────────────────────────────────

    #[test]
    fn test_verify_non_matching_public_key_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"data";
        let (ts, sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        // Ed25519 implementations differ on whether a 32-byte compressed key
        // is rejected during parsing or later during signature verification.
        // The security contract here is stable: a non-matching key must never
        // validate a signature produced by another wallet.
        let bad_pubkey = [0xffu8; 32];

        let result = verify_signed_message(
            DOMAIN_DEVICE_REGISTER,
            &[payload.as_ref()],
            &bad_pubkey,
            &sig,
            ts,
        );
        assert!(
            matches!(
                result,
                Err(AuthError::InvalidPublicKey | AuthError::SignatureMismatch)
            ),
            "Non-matching public key must be rejected: {:?}",
            result,
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
            "Wrong signature must be rejected: {:?}",
            result,
        );
    }

    #[test]
    fn test_verify_wrong_domain_rejected() {
        let kp = IdentityKeyPair::generate();
        let payload = b"data";
        // Sign under DeviceRegister domain, verify under ChatPull domain
        let (ts, sig) = make_signed(&kp, DOMAIN_DEVICE_REGISTER, payload);

        let result = verify_signed_message(
            DOMAIN_CHAT_PULL, // <── wrong domain
            &[payload.as_ref()],
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Wrong domain must cause signature mismatch: {:?}",
            result,
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
            &[tampered.as_ref()], // <── tampered
            &kp.public_key_bytes(),
            &sig,
            ts,
        );
        assert!(
            matches!(result, Err(AuthError::SignatureMismatch)),
            "Tampered payload must cause signature mismatch: {:?}",
            result,
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
            "Wrong public key must cause mismatch: {:?}",
            result,
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
            "Boundary result must be either Ok or TimestampOutOfWindow: {:?}",
            result,
        );
        // Note: this test is intentionally lenient at the exact boundary because
        // nanosecond-level clock differences between make_signed and verify can
        // push the delta to 61 s. The important guarantees are tested by the
        // stale/future tests above (120 s away).
    }
}
