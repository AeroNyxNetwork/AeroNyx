// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pull_cursor.rs
// ============================================
// Version: 1.1.0-PendingDeliveryComposition
//
// Creation Reason:
//   [CHAT-PULL-CURSOR-DOMAIN 2026-08-25 by Codex] Extract the authenticated
//   ChatPullV2 cursor model and protection mechanism from the oversized relay
//   service without changing its wire bytes, binding, or error contract.
//
// Modification Reason:
//   [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Updated ownership after
//   cursor protection became part of the composed pending-delivery use case.
//
// Main Functionality:
//   - Models stable snapshot progress as a bounded domain struct.
//   - Defines a replaceable cursor-protection trait.
//   - Implements the production HKDF + XChaCha20-Poly1305 protector.
//   - Preserves the existing receiver/filter-bound 57-byte cursor format.
//
// Dependencies:
//   - `chat_relay_pending_delivery.rs` composes snapshot reads with this codec.
//   - `chat_relay_error.rs` owns the stable service-facing error boundary.
//
// Main Logical Flow:
//   1. Validate the monotonic snapshot position and ceiling.
//   2. Encode both counters in their existing little-endian wire layout.
//   3. Seal them with AAD bound to receiver identity and pull filter.
//   4. Authenticate, decode, and validate before returning cursor state.
//
// Important Note for Next Developer:
//   - The encoded version, length, byte order, HKDF domains, and AAD are wire
//     compatibility contracts; changing any of them invalidates live cursors.
//   - Never log receiver keys, cursor bytes, counters, nonces, or AEAD errors.
//   - SQLite snapshot capture and paging belong to the delivery coordinator.
//   - Replacement protectors must preserve binding and fail closed.
//
// Last Modified:
//   v1.1.0-PendingDeliveryComposition - Documented coordinator ownership
//   v1.0.0-PullCursorDomain - Initial trait/composition extraction
// ============================================

use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;

// [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Cursor protection depends
// directly on the typed failure contract instead of relay orchestration.
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

const CURSOR_VERSION: u8 = 1;
const NONCE_BYTES: usize = 24;
const PAYLOAD_BYTES: usize = 16;
const TAG_BYTES: usize = 16;
pub(crate) const ENCODED_CURSOR_BYTES: usize = 1 + NONCE_BYTES + PAYLOAD_BYTES + TAG_BYTES;
const AAD_DOMAIN: &[u8] = b"AeroNyx-ChatPullCursor-v2";
const HKDF_SALT: &[u8] = b"AeroNyx-ChatPullCursor-v2-key";
const HKDF_INFO: &[u8] = b"XChaCha20-Poly1305";

/// Stable progress within one receiver-specific queue snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PullCursorV2 {
    pub(crate) position: u64,
    pub(crate) ceiling: u64,
}

impl PullCursorV2 {
    fn is_valid_for_encoding(self) -> bool {
        self.position <= self.ceiling && self.ceiling <= i64::MAX as u64
    }
}

/// Replaceable protection capability for opaque ChatPullV2 cursors.
///
/// Implementations must authenticate the receiver and filter binding, expose
/// no key material, and map malformed or unauthenticated input to the stable
/// invalid-cursor boundary.
pub(crate) trait PullCursorProtector: Send + Sync {
    fn protect(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: PullCursorV2,
    ) -> ChatRelayResult<Vec<u8>>;

    fn recover(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2>;
}

/// Production XChaCha20-Poly1305 cursor protector.
pub(crate) struct XChaChaPullCursorProtector {
    key: [u8; 32],
}

impl XChaChaPullCursorProtector {
    fn new(node_secret: &[u8; 32]) -> ChatRelayResult<Self> {
        let hkdf = hkdf::Hkdf::<Sha256>::new(Some(HKDF_SALT), node_secret);
        let mut key = [0_u8; 32];
        hkdf.expand(HKDF_INFO, &mut key)
            .map_err(|_| ChatRelayError::PullCursorEncryptionFailed)?;
        Ok(Self { key })
    }

    fn aad(receiver: &[u8; 32], after_timestamp: u64) -> Vec<u8> {
        let mut aad =
            Vec::with_capacity(AAD_DOMAIN.len() + receiver.len() + std::mem::size_of::<u64>());
        aad.extend_from_slice(AAD_DOMAIN);
        aad.extend_from_slice(receiver);
        aad.extend_from_slice(&after_timestamp.to_le_bytes());
        aad
    }
}

impl PullCursorProtector for XChaChaPullCursorProtector {
    fn protect(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: PullCursorV2,
    ) -> ChatRelayResult<Vec<u8>> {
        if !cursor.is_valid_for_encoding() {
            return Err(ChatRelayError::PullCursorEncryptionFailed);
        }

        let mut plaintext = [0_u8; PAYLOAD_BYTES];
        plaintext[..8].copy_from_slice(&cursor.position.to_le_bytes());
        plaintext[8..].copy_from_slice(&cursor.ceiling.to_le_bytes());

        let mut nonce = [0_u8; NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(receiver, after_timestamp);
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::PullCursorEncryptionFailed)?;
        if ciphertext.len() != PAYLOAD_BYTES + TAG_BYTES {
            return Err(ChatRelayError::PullCursorEncryptionFailed);
        }

        let mut encoded = Vec::with_capacity(ENCODED_CURSOR_BYTES);
        encoded.push(CURSOR_VERSION);
        encoded.extend_from_slice(&nonce);
        encoded.extend_from_slice(&ciphertext);
        Ok(encoded)
    }

    fn recover(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        if encoded.len() != ENCODED_CURSOR_BYTES || encoded.first().copied() != Some(CURSOR_VERSION)
        {
            return Err(ChatRelayError::InvalidPullCursor);
        }

        let nonce_start = 1;
        let ciphertext_start = nonce_start + NONCE_BYTES;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(receiver, after_timestamp);
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(&encoded[nonce_start..ciphertext_start]),
                Payload {
                    msg: &encoded[ciphertext_start..],
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::InvalidPullCursor)?;
        if plaintext.len() != PAYLOAD_BYTES {
            return Err(ChatRelayError::InvalidPullCursor);
        }

        let mut position = [0_u8; 8];
        position.copy_from_slice(&plaintext[..8]);
        let mut ceiling = [0_u8; 8];
        ceiling.copy_from_slice(&plaintext[8..]);
        let cursor = PullCursorV2 {
            position: u64::from_le_bytes(position),
            ceiling: u64::from_le_bytes(ceiling),
        };
        if !cursor.is_valid_for_encoding() {
            return Err(ChatRelayError::InvalidPullCursor);
        }
        Ok(cursor)
    }
}

/// Composed ChatPullV2 cursor domain used by the relay service.
pub(crate) struct ChatPullCursorCodec<P = XChaChaPullCursorProtector> {
    protector: P,
}

impl ChatPullCursorCodec<XChaChaPullCursorProtector> {
    pub(crate) fn new(node_secret: &[u8; 32]) -> ChatRelayResult<Self> {
        Ok(Self::with_protector(XChaChaPullCursorProtector::new(
            node_secret,
        )?))
    }
}

impl<P: PullCursorProtector> ChatPullCursorCodec<P> {
    fn with_protector(protector: P) -> Self {
        Self { protector }
    }

    pub(crate) fn encode(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: PullCursorV2,
    ) -> ChatRelayResult<Vec<u8>> {
        self.protector.protect(receiver, after_timestamp, cursor)
    }

    pub(crate) fn decode(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        self.protector.recover(receiver, after_timestamp, encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_round_trip_preserves_wire_shape_and_binding() {
        let codec = ChatPullCursorCodec::new(&[0x31; 32]).expect("construct cursor codec");
        let receiver = [0x32; 32];
        let cursor = PullCursorV2 {
            position: 17,
            ceiling: 29,
        };

        let encoded = codec.encode(&receiver, 41, cursor).expect("protect cursor");
        assert_eq!(encoded.len(), ENCODED_CURSOR_BYTES);
        assert_eq!(encoded.first().copied(), Some(CURSOR_VERSION));
        assert_eq!(
            codec
                .decode(&receiver, 41, &encoded)
                .expect("recover cursor"),
            cursor
        );
        assert!(matches!(
            codec.decode(&[0x33; 32], 41, &encoded),
            Err(ChatRelayError::InvalidPullCursor)
        ));
        assert!(matches!(
            codec.decode(&receiver, 42, &encoded),
            Err(ChatRelayError::InvalidPullCursor)
        ));
    }

    #[test]
    fn cursor_tampering_and_invalid_bounds_fail_closed() {
        let codec = ChatPullCursorCodec::new(&[0x34; 32]).expect("construct cursor codec");
        let receiver = [0x35; 32];
        let mut encoded = codec
            .encode(
                &receiver,
                0,
                PullCursorV2 {
                    position: 1,
                    ceiling: 2,
                },
            )
            .expect("protect cursor");
        *encoded.last_mut().expect("cursor tag") ^= 0x01;
        assert!(matches!(
            codec.decode(&receiver, 0, &encoded),
            Err(ChatRelayError::InvalidPullCursor)
        ));
        assert!(matches!(
            codec.encode(
                &receiver,
                0,
                PullCursorV2 {
                    position: 2,
                    ceiling: 1,
                }
            ),
            Err(ChatRelayError::PullCursorEncryptionFailed)
        ));
        assert!(matches!(
            codec.encode(
                &receiver,
                0,
                PullCursorV2 {
                    position: 0,
                    ceiling: i64::MAX as u64 + 1,
                }
            ),
            Err(ChatRelayError::PullCursorEncryptionFailed)
        ));
    }
}
