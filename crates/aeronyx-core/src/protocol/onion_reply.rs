// ============================================
// File: crates/aeronyx-core/src/protocol/onion_reply.rs
// ============================================
//! # Onion Reply Protocol v1
//!
//! ## Creation Reason
//! AeroNyx onion routes already provide blind, multi-hop delivery and durable
//! terminal receipts, but recovery workloads also need a bounded response path
//! that does not expose the response or terminal identity to middle relays.
//!
//! ## Main Functionality
//! - Wraps one terminal request with an ephemeral client reply key.
//! - Owns one single-use source session so the reply private key cannot be
//!   accidentally reused across routes.
//! - Seals one fixed-size response with X25519, HKDF-SHA256, and
//!   XChaCha20-Poly1305.
//! - Keeps the terminal identity and terminal signature inside ciphertext.
//! - Binds every response to the exact route, request commitment, reply key,
//!   terminal ephemeral key, response size class, and payload commitment.
//! - Provides explicit, language-neutral request and response wire codecs.
//!
//! ## Dependencies
//! - `crypto::keys`: audited X25519, Ed25519, and XChaCha20-Poly1305 wrappers.
//! - `protocol::onion`: callers carry the encoded request as the final onion
//!   payload and propagate the encoded response without parsing it.
//! - `protocol::onion_reply/session.rs`: single-use source session ownership.
//!
//! ## Main Logical Flow
//! 1. The source creates an ephemeral X25519 key and encodes a bounded request.
//! 2. Only the terminal peels the final onion layer and reads the reply key.
//! 3. The terminal executes the opaque workload and signs its exact response.
//! 4. The terminal pads, encrypts, and returns the sealed response unchanged.
//! 5. Middle hops forward bytes only; the source decrypts and verifies the
//!   expected terminal identity before accepting the payload.
//!
//! ## Important Note For The Next Developer
//! - Do not move terminal identity, response purpose, payload length, or
//!   signature outside the encrypted plaintext.
//! - Do not add variable response sizes; only the published size classes are
//!   valid, and their exact length is checked during open.
//! - Do not reuse one reply key across logical requests or replica nodes.
//! - Keep this carrier workload-neutral. Blind Vault, mailbox, and Agent RPC
//!   parsing belongs at the terminal dispatch boundary.
//!
//! Last Modified: v1.4.0-SessionRestartState - Kept the private restart codec
//! reachable only to identity-sealed workflow composition inside this crate.
//! v1.3.0-SessionModule - Split single-use source session and
//! recoverable key ownership from the stable request/response wire codec.
//! v1.2.0-RequestBoundReply - Bound key derivation, encrypted
//! response metadata, and terminal signatures to the exact request payload.
//! v1.1.0-OnionReplySession - Added single-use source request
//! preparation and response verification.
//! v1.0.0-OnionReply - Added the bounded encrypted return path.
//! ============================================

use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroize;

use crate::crypto::keys::{E2eSession, EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey};

mod session;

pub use session::OnionReplySession;
pub(crate) use session::{OnionReplySessionRestartError, OnionReplySessionRestartState};

const REQUEST_MAGIC: [u8; 4] = *b"ANRQ";
const RESPONSE_MAGIC: [u8; 4] = *b"ANRS";
const PLAINTEXT_MAGIC: [u8; 4] = *b"ANRP";
const RESPONSE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-Onion-Reply-Response-v1";
const RESPONSE_KEY_SALT: &[u8] = b"AeroNyx-Onion-Reply-Key-v1";
const REQUEST_CONTEXT_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-Onion-Reply-RequestContext-v2";
const LEGACY_REQUEST_VERSION: u8 = 1;
const SOURCE_SEALED_REQUEST_VERSION: u8 = 2;
const RESPONSE_VERSION: u8 = 1;
const AEAD_NONCE_BYTES: usize = 24;
const AEAD_TAG_BYTES: usize = 16;
const SIGNATURE_BYTES: usize = 64;
const LEGACY_REQUEST_HEADER_BYTES: usize = 4 + 1 + 32 + 4 + 4;
const SOURCE_SEALED_REQUEST_HEADER_BYTES: usize = LEGACY_REQUEST_HEADER_BYTES + 1;
const RESPONSE_HEADER_BYTES: usize = 4 + 1 + 32 + AEAD_NONCE_BYTES + 4;
const PLAINTEXT_HEADER_BYTES: usize = 4 + 1 + 4 + 16 + 32 + 32 + 4;

/// Maximum workload request carried inside one reply-capable terminal frame.
pub const MAX_ONION_REPLY_REQUEST_PAYLOAD_BYTES: usize = 16 * 1024;

/// Exact padded plaintext sizes accepted for onion responses.
///
/// The largest class carries one maximum-size Blind Vault ciphertext object
/// plus its signed recovery metadata. The encoded response belongs on the
/// separately bounded reply transport and must not be inserted into a new
/// forward onion layer.
pub const ONION_REPLY_RESPONSE_SIZE_CLASSES: [usize; 4] =
    [8 * 1024, 32 * 1024, 96 * 1024, 272 * 1024];

/// Maximum encoded response bytes accepted by the explicit wire decoder.
pub const MAX_ONION_SEALED_RESPONSE_BYTES: usize = RESPONSE_HEADER_BYTES
    + ONION_REPLY_RESPONSE_SIZE_CLASSES[ONION_REPLY_RESPONSE_SIZE_CLASSES.len() - 1]
    + AEAD_TAG_BYTES;

/// Relay-visible terminal-proof behavior requested by the source.
///
/// [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Version 1 preserves the
/// historical clear terminal receipt for rolling compatibility. Version 2
/// asks every upgraded hop to rely on its immediate-hop success receipt while
/// the final identity and result remain inside the encrypted reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OnionReplyProofMode {
    /// Preserve the historical relay-visible terminal delivery receipt.
    RelayVisibleTerminalReceipt = 0,
    /// Return terminal proof only inside the source-sealed response.
    SourceSealedTerminalProof = 1,
}

impl TryFrom<u8> for OnionReplyProofMode {
    type Error = OnionReplyError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::RelayVisibleTerminalReceipt),
            1 => Ok(Self::SourceSealedTerminalProof),
            _ => Err(OnionReplyError::InvalidProofMode(value)),
        }
    }
}

/// Reply-capable terminal request carried as the final onion payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionReplyRequest {
    /// Independent reply-carrier protocol version.
    pub version: u8,
    /// Requested terminal-proof visibility. V1 is always relay-visible.
    proof_mode: OnionReplyProofMode,
    /// Source-generated, single-use X25519 public key.
    pub reply_public_key: [u8; 32],
    /// Exact padded plaintext size requested for the sealed response.
    pub response_size_class: u32,
    /// Bounded workload frame interpreted only by the terminal.
    pub payload: Vec<u8>,
}

impl OnionReplyRequest {
    /// Creates and validates a reply-capable terminal request.
    pub fn new(
        reply_public_key: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
    ) -> Result<Self, OnionReplyError> {
        Self::new_with_proof_mode(
            LEGACY_REQUEST_VERSION,
            OnionReplyProofMode::RelayVisibleTerminalReceipt,
            reply_public_key,
            response_size_class,
            payload,
        )
    }

    /// Creates a v2 request whose terminal proof remains source-sealed.
    pub fn new_source_sealed(
        reply_public_key: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
    ) -> Result<Self, OnionReplyError> {
        Self::new_with_proof_mode(
            SOURCE_SEALED_REQUEST_VERSION,
            OnionReplyProofMode::SourceSealedTerminalProof,
            reply_public_key,
            response_size_class,
            payload,
        )
    }

    fn new_with_proof_mode(
        version: u8,
        proof_mode: OnionReplyProofMode,
        reply_public_key: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
    ) -> Result<Self, OnionReplyError> {
        let response_size_class =
            u32::try_from(response_size_class).map_err(|_| OnionReplyError::InvalidSizeClass)?;
        let request = Self {
            version,
            proof_mode,
            reply_public_key,
            response_size_class,
            payload,
        };
        request.validate()?;
        Ok(request)
    }

    /// Validates version, key, response class, and request allocation bounds.
    pub fn validate(&self) -> Result<(), OnionReplyError> {
        match (self.version, self.proof_mode) {
            (LEGACY_REQUEST_VERSION, OnionReplyProofMode::RelayVisibleTerminalReceipt)
            | (SOURCE_SEALED_REQUEST_VERSION, OnionReplyProofMode::SourceSealedTerminalProof) => {}
            (version, _)
                if matches!(
                    version,
                    LEGACY_REQUEST_VERSION | SOURCE_SEALED_REQUEST_VERSION
                ) =>
            {
                return Err(OnionReplyError::InvalidProofMode(self.proof_mode as u8))
            }
            _ => return Err(OnionReplyError::UnsupportedVersion(self.version)),
        }
        if self.reply_public_key.iter().all(|byte| *byte == 0) {
            return Err(OnionReplyError::InvalidReplyKey);
        }
        response_size_class(self.response_size_class)?;
        if self.payload.is_empty() || self.payload.len() > MAX_ONION_REPLY_REQUEST_PAYLOAD_BYTES {
            return Err(OnionReplyError::InvalidRequestPayloadSize {
                actual: self.payload.len(),
            });
        }
        Ok(())
    }

    /// Returns the validated terminal-proof visibility requested by the source.
    #[must_use]
    pub const fn proof_mode(&self) -> OnionReplyProofMode {
        self.proof_mode
    }
}

/// Opaque fixed-size response propagated unchanged through middle relays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionSealedResponse {
    /// Independent reply-carrier protocol version.
    pub version: u8,
    /// Terminal-generated, single-use X25519 public key.
    pub ephemeral_public_key: [u8; 32],
    /// Random XChaCha20-Poly1305 nonce.
    pub nonce: [u8; AEAD_NONCE_BYTES],
    /// Fixed-class ciphertext including the Poly1305 tag.
    pub ciphertext: Vec<u8>,
}

impl OnionSealedResponse {
    /// Validates the public response envelope without decrypting it.
    pub fn validate(&self) -> Result<usize, OnionReplyError> {
        if self.version != RESPONSE_VERSION {
            return Err(OnionReplyError::UnsupportedVersion(self.version));
        }
        if self.ephemeral_public_key.iter().all(|byte| *byte == 0) {
            return Err(OnionReplyError::InvalidReplyKey);
        }
        let plaintext_len = self.ciphertext.len().checked_sub(AEAD_TAG_BYTES).ok_or(
            OnionReplyError::InvalidCiphertextSize {
                actual: self.ciphertext.len(),
            },
        )?;
        if !ONION_REPLY_RESPONSE_SIZE_CLASSES.contains(&plaintext_len) {
            return Err(OnionReplyError::InvalidCiphertextSize {
                actual: self.ciphertext.len(),
            });
        }
        Ok(plaintext_len)
    }
}

/// Authenticated plaintext returned after opening one sealed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionReplyPayload {
    /// Terminal descriptor identity that signed the exact response.
    pub terminal_node_id: [u8; 32],
    /// Workload response bytes, such as one encoded Blind Vault pull page.
    pub payload: Vec<u8>,
}

/// Encodes one request with an explicit stable byte layout.
pub fn encode_onion_reply_request(request: &OnionReplyRequest) -> Result<Vec<u8>, OnionReplyError> {
    request.validate()?;
    let payload_len =
        u32::try_from(request.payload.len()).map_err(|_| OnionReplyError::FrameTooLarge)?;
    let header_bytes = match request.version {
        LEGACY_REQUEST_VERSION => LEGACY_REQUEST_HEADER_BYTES,
        SOURCE_SEALED_REQUEST_VERSION => SOURCE_SEALED_REQUEST_HEADER_BYTES,
        _ => return Err(OnionReplyError::UnsupportedVersion(request.version)),
    };
    let mut encoded = Vec::with_capacity(header_bytes + request.payload.len());
    encoded.extend_from_slice(&REQUEST_MAGIC);
    encoded.push(request.version);
    if request.version == SOURCE_SEALED_REQUEST_VERSION {
        encoded.push(request.proof_mode as u8);
    }
    encoded.extend_from_slice(&request.reply_public_key);
    encoded.extend_from_slice(&request.response_size_class.to_be_bytes());
    encoded.extend_from_slice(&payload_len.to_be_bytes());
    encoded.extend_from_slice(&request.payload);
    Ok(encoded)
}

/// Decodes one bounded reply-capable request and rejects trailing bytes.
pub fn decode_onion_reply_request(encoded: &[u8]) -> Result<OnionReplyRequest, OnionReplyError> {
    if encoded.len() > SOURCE_SEALED_REQUEST_HEADER_BYTES + MAX_ONION_REPLY_REQUEST_PAYLOAD_BYTES {
        return Err(OnionReplyError::FrameTooLarge);
    }
    let mut cursor = WireCursor::new(encoded);
    if cursor.take_array::<4>()? != REQUEST_MAGIC {
        return Err(OnionReplyError::InvalidMagic);
    }
    let version = cursor.take_u8()?;
    let proof_mode = match version {
        LEGACY_REQUEST_VERSION => OnionReplyProofMode::RelayVisibleTerminalReceipt,
        SOURCE_SEALED_REQUEST_VERSION => OnionReplyProofMode::try_from(cursor.take_u8()?)?,
        _ => return Err(OnionReplyError::UnsupportedVersion(version)),
    };
    let reply_public_key = cursor.take_array::<32>()?;
    let response_size_class = cursor.take_u32()?;
    let payload_len = cursor.take_u32()? as usize;
    let payload = cursor.take_vec(payload_len)?;
    cursor.finish()?;

    let request = OnionReplyRequest {
        version,
        proof_mode,
        reply_public_key,
        response_size_class,
        payload,
    };
    request.validate()?;
    Ok(request)
}

/// Returns whether bytes declare the reply-capable terminal request format.
#[must_use]
pub fn is_onion_reply_request(encoded: &[u8]) -> bool {
    encoded.starts_with(&REQUEST_MAGIC)
}

/// Seals one terminal response for the request's single-use reply key.
pub fn seal_onion_reply(
    route_id: [u8; 16],
    request: &OnionReplyRequest,
    payload: &[u8],
    terminal_identity: &IdentityKeyPair,
) -> Result<OnionSealedResponse, OnionReplyError> {
    request.validate()?;
    let response_size_class = response_size_class(request.response_size_class)?;
    let minimum_plaintext = PLAINTEXT_HEADER_BYTES
        .checked_add(payload.len())
        .and_then(|size| size.checked_add(SIGNATURE_BYTES))
        .ok_or(OnionReplyError::ResponsePayloadTooLarge)?;
    if minimum_plaintext > response_size_class {
        return Err(OnionReplyError::ResponsePayloadTooLarge);
    }

    let ephemeral = EphemeralKeyPair::generate();
    let ephemeral_public_key = ephemeral.public_key_bytes();
    let request_context_commitment = request_context_commitment(request);
    let mut shared_secret = ephemeral.exchange(&request.reply_public_key);
    if let Err(error) = reject_low_order_shared_secret(&shared_secret) {
        shared_secret.zeroize();
        return Err(error);
    }
    let session_key_result = derive_response_key(
        &shared_secret,
        &route_id,
        &request.reply_public_key,
        &ephemeral_public_key,
        &request_context_commitment,
    );
    shared_secret.zeroize();
    let mut session_key = session_key_result?;

    let terminal_node_id = terminal_identity.public_key_bytes();
    let signature = terminal_identity.sign(&response_signing_bytes(
        route_id,
        request.reply_public_key,
        ephemeral_public_key,
        response_size_class,
        request_context_commitment,
        terminal_node_id,
        payload,
    ));
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| OnionReplyError::ResponsePayloadTooLarge)?;
    let mut plaintext = vec![0u8; response_size_class];
    let mut offset = 0usize;
    write_part(&mut plaintext, &mut offset, &PLAINTEXT_MAGIC)?;
    write_part(&mut plaintext, &mut offset, &[RESPONSE_VERSION])?;
    write_part(
        &mut plaintext,
        &mut offset,
        &request.response_size_class.to_be_bytes(),
    )?;
    write_part(&mut plaintext, &mut offset, &route_id)?;
    write_part(&mut plaintext, &mut offset, &request_context_commitment)?;
    write_part(&mut plaintext, &mut offset, &terminal_node_id)?;
    write_part(&mut plaintext, &mut offset, &payload_len.to_be_bytes())?;
    write_part(&mut plaintext, &mut offset, payload)?;
    write_part(&mut plaintext, &mut offset, &signature)?;
    OsRng.fill_bytes(&mut plaintext[offset..]);

    let mut nonce = [0u8; AEAD_NONCE_BYTES];
    OsRng.fill_bytes(&mut nonce);
    let session = E2eSession::new(session_key, request.reply_public_key);
    session_key.zeroize();
    let ciphertext_result = session.encrypt_raw(&plaintext, &nonce);
    plaintext.zeroize();
    let ciphertext = ciphertext_result.map_err(|_| OnionReplyError::EncryptionFailed)?;

    let response = OnionSealedResponse {
        version: RESPONSE_VERSION,
        ephemeral_public_key,
        nonce,
        ciphertext,
    };
    response.validate()?;
    Ok(response)
}

/// Opens and verifies one response against the exact route and terminal.
pub fn open_onion_reply(
    route_id: [u8; 16],
    response: &OnionSealedResponse,
    reply_key: EphemeralKeyPair,
    request: &OnionReplyRequest,
    expected_terminal_node_id: [u8; 32],
) -> Result<OnionReplyPayload, OnionReplyError> {
    request.validate()?;
    let reply_public_key = reply_key.public_key_bytes();
    if request.reply_public_key != reply_public_key {
        return Err(OnionReplyError::InvalidReplyKey);
    }
    open_onion_reply_with_context(
        route_id,
        response,
        reply_key,
        expected_terminal_node_id,
        response_size_class(request.response_size_class)?,
        request_context_commitment(request),
    )
}

trait ReplyKeyAgreement {
    fn public_key_bytes(&self) -> [u8; 32];

    fn exchange(self, peer_public: &[u8; 32]) -> Result<[u8; 32], OnionReplyError>;
}

impl ReplyKeyAgreement for EphemeralKeyPair {
    fn public_key_bytes(&self) -> [u8; 32] {
        EphemeralKeyPair::public_key_bytes(self)
    }

    fn exchange(self, peer_public: &[u8; 32]) -> Result<[u8; 32], OnionReplyError> {
        if self.is_consumed() {
            return Err(OnionReplyError::InvalidReplyKey);
        }
        Ok(EphemeralKeyPair::exchange(self, peer_public))
    }
}

fn open_onion_reply_with_context<K: ReplyKeyAgreement>(
    route_id: [u8; 16],
    response: &OnionSealedResponse,
    reply_key: K,
    expected_terminal_node_id: [u8; 32],
    expected_size_class: usize,
    expected_request_context_commitment: [u8; 32],
) -> Result<OnionReplyPayload, OnionReplyError> {
    if response.validate()? != expected_size_class {
        return Err(OnionReplyError::InvalidSizeClass);
    }
    let reply_public_key = reply_key.public_key_bytes();
    let mut shared_secret = reply_key.exchange(&response.ephemeral_public_key)?;
    if let Err(error) = reject_low_order_shared_secret(&shared_secret) {
        shared_secret.zeroize();
        return Err(error);
    }
    let session_key_result = derive_response_key(
        &shared_secret,
        &route_id,
        &reply_public_key,
        &response.ephemeral_public_key,
        &expected_request_context_commitment,
    );
    shared_secret.zeroize();
    let mut session_key = session_key_result?;

    let session = E2eSession::new(session_key, response.ephemeral_public_key);
    session_key.zeroize();
    let mut plaintext = session
        .decrypt_raw(&response.ciphertext, &response.nonce)
        .map_err(|_| OnionReplyError::DecryptionFailed)?;
    let result = decode_reply_plaintext(
        &plaintext,
        route_id,
        reply_public_key,
        response.ephemeral_public_key,
        expected_size_class,
        expected_request_context_commitment,
        expected_terminal_node_id,
    );
    plaintext.zeroize();
    result
}

/// Encodes one opaque sealed response for binary transport.
pub fn encode_onion_sealed_response(
    response: &OnionSealedResponse,
) -> Result<Vec<u8>, OnionReplyError> {
    response.validate()?;
    let ciphertext_len =
        u32::try_from(response.ciphertext.len()).map_err(|_| OnionReplyError::FrameTooLarge)?;
    let mut encoded = Vec::with_capacity(RESPONSE_HEADER_BYTES + response.ciphertext.len());
    encoded.extend_from_slice(&RESPONSE_MAGIC);
    encoded.push(response.version);
    encoded.extend_from_slice(&response.ephemeral_public_key);
    encoded.extend_from_slice(&response.nonce);
    encoded.extend_from_slice(&ciphertext_len.to_be_bytes());
    encoded.extend_from_slice(&response.ciphertext);
    Ok(encoded)
}

/// Decodes one opaque sealed response and rejects excess allocation or bytes.
pub fn decode_onion_sealed_response(
    encoded: &[u8],
) -> Result<OnionSealedResponse, OnionReplyError> {
    if encoded.len() > MAX_ONION_SEALED_RESPONSE_BYTES {
        return Err(OnionReplyError::FrameTooLarge);
    }
    let mut cursor = WireCursor::new(encoded);
    if cursor.take_array::<4>()? != RESPONSE_MAGIC {
        return Err(OnionReplyError::InvalidMagic);
    }
    let version = cursor.take_u8()?;
    let ephemeral_public_key = cursor.take_array::<32>()?;
    let nonce = cursor.take_array::<AEAD_NONCE_BYTES>()?;
    let ciphertext_len = cursor.take_u32()? as usize;
    let ciphertext = cursor.take_vec(ciphertext_len)?;
    cursor.finish()?;

    let response = OnionSealedResponse {
        version,
        ephemeral_public_key,
        nonce,
        ciphertext,
    };
    response.validate()?;
    Ok(response)
}

fn decode_reply_plaintext(
    plaintext: &[u8],
    expected_route_id: [u8; 16],
    reply_public_key: [u8; 32],
    ephemeral_public_key: [u8; 32],
    expected_size_class: usize,
    expected_request_context_commitment: [u8; 32],
    expected_terminal_node_id: [u8; 32],
) -> Result<OnionReplyPayload, OnionReplyError> {
    if plaintext.len() != expected_size_class {
        return Err(OnionReplyError::InvalidPlaintextSize);
    }
    let mut cursor = WireCursor::new(plaintext);
    if cursor.take_array::<4>()? != PLAINTEXT_MAGIC {
        return Err(OnionReplyError::InvalidPlaintext);
    }
    let version = cursor.take_u8()?;
    if version != RESPONSE_VERSION {
        return Err(OnionReplyError::UnsupportedVersion(version));
    }
    let declared_size_class = cursor.take_u32()? as usize;
    if declared_size_class != expected_size_class
        || !ONION_REPLY_RESPONSE_SIZE_CLASSES.contains(&declared_size_class)
    {
        return Err(OnionReplyError::InvalidSizeClass);
    }
    let route_id = cursor.take_array::<16>()?;
    if route_id != expected_route_id {
        return Err(OnionReplyError::RouteMismatch);
    }
    let request_context_commitment = cursor.take_array::<32>()?;
    if request_context_commitment != expected_request_context_commitment {
        return Err(OnionReplyError::RequestMismatch);
    }
    let terminal_node_id = cursor.take_array::<32>()?;
    if terminal_node_id != expected_terminal_node_id {
        return Err(OnionReplyError::TerminalMismatch);
    }
    let payload_len = cursor.take_u32()? as usize;
    let maximum_payload = expected_size_class
        .checked_sub(PLAINTEXT_HEADER_BYTES + SIGNATURE_BYTES)
        .ok_or(OnionReplyError::InvalidPlaintextSize)?;
    if payload_len > maximum_payload {
        return Err(OnionReplyError::InvalidPlaintextSize);
    }
    let payload = cursor.take_vec(payload_len)?;
    let signature = cursor.take_array::<SIGNATURE_BYTES>()?;

    let terminal = IdentityPublicKey::from_bytes(&terminal_node_id)
        .map_err(|_| OnionReplyError::InvalidTerminalIdentity)?;
    terminal
        .verify(
            &response_signing_bytes(
                route_id,
                reply_public_key,
                ephemeral_public_key,
                expected_size_class,
                request_context_commitment,
                terminal_node_id,
                &payload,
            ),
            &signature,
        )
        .map_err(|_| OnionReplyError::InvalidSignature)?;

    Ok(OnionReplyPayload {
        terminal_node_id,
        payload,
    })
}

fn response_signing_bytes(
    route_id: [u8; 16],
    reply_public_key: [u8; 32],
    ephemeral_public_key: [u8; 32],
    response_size_class: usize,
    request_payload_commitment: [u8; 32],
    terminal_node_id: [u8; 32],
    payload: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RESPONSE_SIGNING_DOMAIN.len() + 189);
    bytes.extend_from_slice(RESPONSE_SIGNING_DOMAIN);
    bytes.push(RESPONSE_VERSION);
    bytes.extend_from_slice(&route_id);
    bytes.extend_from_slice(&reply_public_key);
    bytes.extend_from_slice(&ephemeral_public_key);
    bytes.extend_from_slice(
        &u32::try_from(response_size_class)
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    bytes.extend_from_slice(&request_payload_commitment);
    bytes.extend_from_slice(&terminal_node_id);
    bytes.extend_from_slice(
        &u32::try_from(payload.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    bytes.extend_from_slice(&Sha256::digest(payload));
    bytes
}

fn derive_response_key(
    shared_secret: &[u8; 32],
    route_id: &[u8; 16],
    reply_public_key: &[u8; 32],
    ephemeral_public_key: &[u8; 32],
    request_payload_commitment: &[u8; 32],
) -> Result<[u8; 32], OnionReplyError> {
    let hkdf = Hkdf::<Sha256>::new(Some(RESPONSE_KEY_SALT), shared_secret);
    let mut info = Vec::with_capacity(16 + 32 + 32 + 32);
    info.extend_from_slice(route_id);
    info.extend_from_slice(reply_public_key);
    info.extend_from_slice(ephemeral_public_key);
    info.extend_from_slice(request_payload_commitment);
    let mut key = [0u8; 32];
    hkdf.expand(&info, &mut key)
        .map_err(|_| OnionReplyError::KeyDerivationFailed)?;
    Ok(key)
}

fn request_context_commitment(request: &OnionReplyRequest) -> [u8; 32] {
    // Preserve the deployed v1 response cryptographic transcript byte for
    // byte. V2 additionally authenticates proof visibility and frame shape so
    // a source-sealed request cannot be reinterpreted under the legacy mode.
    if request.version == LEGACY_REQUEST_VERSION {
        return payload_commitment(&request.payload);
    }
    let mut hasher = Sha256::new();
    hasher.update(REQUEST_CONTEXT_COMMITMENT_DOMAIN);
    hasher.update([request.version, request.proof_mode as u8]);
    hasher.update(request.response_size_class.to_be_bytes());
    hasher.update(
        u64::try_from(request.payload.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    hasher.update(payload_commitment(&request.payload));
    hasher.finalize().into()
}

fn payload_commitment(payload: &[u8]) -> [u8; 32] {
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&Sha256::digest(payload));
    commitment
}

fn reject_low_order_shared_secret(shared_secret: &[u8; 32]) -> Result<(), OnionReplyError> {
    // [ONION-REPLY-LOW-ORDER 2026-08-28 by Codex] X25519 low-order public
    // inputs yield an all-zero shared secret. Accepting one would replace
    // possession of the ephemeral private key with a public constant.
    if shared_secret.iter().all(|byte| *byte == 0) {
        Err(OnionReplyError::InvalidReplyKey)
    } else {
        Ok(())
    }
}

fn response_size_class(encoded: u32) -> Result<usize, OnionReplyError> {
    let value = usize::try_from(encoded).map_err(|_| OnionReplyError::InvalidSizeClass)?;
    ONION_REPLY_RESPONSE_SIZE_CLASSES
        .contains(&value)
        .then_some(value)
        .ok_or(OnionReplyError::InvalidSizeClass)
}

fn write_part(
    destination: &mut [u8],
    offset: &mut usize,
    bytes: &[u8],
) -> Result<(), OnionReplyError> {
    let end = offset
        .checked_add(bytes.len())
        .ok_or(OnionReplyError::ResponsePayloadTooLarge)?;
    let slot = destination
        .get_mut(*offset..end)
        .ok_or(OnionReplyError::ResponsePayloadTooLarge)?;
    slot.copy_from_slice(bytes);
    *offset = end;
    Ok(())
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WireCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take_u8(&mut self) -> Result<u8, OnionReplyError> {
        Ok(self.take_array::<1>()?[0])
    }

    fn take_u32(&mut self) -> Result<u32, OnionReplyError> {
        Ok(u32::from_be_bytes(self.take_array::<4>()?))
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], OnionReplyError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(OnionReplyError::TruncatedFrame)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(OnionReplyError::TruncatedFrame)?;
        let mut output = [0u8; N];
        output.copy_from_slice(bytes);
        self.offset = end;
        Ok(output)
    }

    fn take_vec(&mut self, len: usize) -> Result<Vec<u8>, OnionReplyError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(OnionReplyError::TruncatedFrame)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(OnionReplyError::TruncatedFrame)?;
        self.offset = end;
        Ok(bytes.to_vec())
    }

    fn finish(self) -> Result<(), OnionReplyError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(OnionReplyError::TrailingBytes)
        }
    }
}

/// Fail-closed onion reply protocol errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OnionReplyError {
    /// The frame declares a version this implementation cannot interpret.
    #[error("unsupported onion reply version {0}")]
    UnsupportedVersion(u8),
    /// Request version and terminal-proof visibility were not a valid pair.
    #[error("invalid onion reply proof mode {0}")]
    InvalidProofMode(u8),
    /// The fixed request or response magic did not match.
    #[error("invalid onion reply magic")]
    InvalidMagic,
    /// The frame ended before all declared fields were present.
    #[error("truncated onion reply frame")]
    TruncatedFrame,
    /// Extra bytes followed the exact declared frame.
    #[error("trailing onion reply bytes")]
    TrailingBytes,
    /// The encoded frame exceeded its public allocation ceiling.
    #[error("onion reply frame too large")]
    FrameTooLarge,
    /// The single-use X25519 key was invalid or low order.
    #[error("invalid onion reply key")]
    InvalidReplyKey,
    /// The requested padded response class is not supported.
    #[error("invalid onion reply response size class")]
    InvalidSizeClass,
    /// The bounded terminal request payload was empty or oversized.
    #[error("invalid onion reply request payload size {actual}")]
    InvalidRequestPayloadSize { actual: usize },
    /// The terminal response cannot fit the requested fixed class.
    #[error("onion reply response payload too large")]
    ResponsePayloadTooLarge,
    /// The outer ciphertext length did not match a fixed class plus AEAD tag.
    #[error("invalid onion reply ciphertext size {actual}")]
    InvalidCiphertextSize { actual: usize },
    /// HKDF could not derive the bounded response key.
    #[error("onion reply key derivation failed")]
    KeyDerivationFailed,
    /// Terminal response encryption failed.
    #[error("onion reply encryption failed")]
    EncryptionFailed,
    /// Source response decryption or AEAD authentication failed.
    #[error("onion reply decryption failed")]
    DecryptionFailed,
    /// Decrypted bytes did not match the canonical plaintext envelope.
    #[error("invalid onion reply plaintext")]
    InvalidPlaintext,
    /// Decrypted fixed-size bytes were internally inconsistent.
    #[error("invalid onion reply plaintext size")]
    InvalidPlaintextSize,
    /// The decrypted response was bound to another route.
    #[error("onion reply route mismatch")]
    RouteMismatch,
    /// The response was not bound to the exact source request payload.
    #[error("onion reply request commitment mismatch")]
    RequestMismatch,
    /// The response signer was not the source-selected terminal.
    #[error("onion reply terminal mismatch")]
    TerminalMismatch,
    /// The terminal identity bytes were not a valid Ed25519 public key.
    #[error("invalid onion reply terminal identity")]
    InvalidTerminalIdentity,
    /// The terminal signature did not authenticate the exact response.
    #[error("invalid onion reply signature")]
    InvalidSignature,
}
