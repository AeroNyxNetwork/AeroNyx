// ============================================
// File: crates/aeronyx-core/src/protocol/onion_reply/session.rs
// ============================================
//! Source-owned single-use session for one encrypted onion terminal reply.
//!
//! ## Creation Reason
//! Reply-session ownership and recoverable private key handling are separate
//! from the public request/response wire codec. Keeping them in the monolithic
//! codec file made restart persistence likely to weaken the general ephemeral
//! key API or duplicate response verification logic.
//!
//! ## Main Functionality
//! - Prepares one reply-capable terminal request.
//! - Retains a module-private recoverable X25519 key.
//! - Consumes the key exactly once while opening the bound response.
//! - Keeps recoverability scoped to this session domain, not general crypto.
//!
//! ## Dependencies
//! - `protocol::onion_reply`: request construction and response verification.
//! - `x25519-dalek`: zeroizing `StaticSecret` used with consuming ownership.
//!
//! ## Main Logical Flow
//! 1. Generate a random session key and derive its public reply key.
//! 2. Bind request payload and mode into a request-context commitment.
//! 3. Retain the private key only inside `OnionReplySession`.
//! 4. Consume the complete session to decrypt and authenticate one response.
//!
//! ## Important Note For The Next Developer
//! - Never expose the private key through the public API.
//! - Persistence must be encrypted before leaving process memory.
//! - Do not implement `Clone` for session or key ownership types.
//! - Keep response opening delegated to the single parent verifier.
//!
//! Last Modified: v1.3.0-EncodedRequestBinding - Added a private-key-safe
//! preflight that rejects session/request cursor mismatch before network I/O.
//! v1.2.0-ExactRequestRebuild - Persisted proof mode and added
//! commitment-checked request reconstruction after restart.
//! v1.1.0-RestartState - Added a fixed, crate-private session
//! encoding that is valid only inside an encrypted attempt journal.
//! v1.0.0-RecoverableSessionKey - Initial focused module.
//! ============================================

use rand::rngs::OsRng;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use super::{
    open_onion_reply_with_context, request_context_commitment, OnionReplyError, OnionReplyPayload,
    OnionReplyProofMode, OnionReplyRequest, ReplyKeyAgreement,
};

const RESTART_STATE_MAGIC: [u8; 4] = *b"AXOR";
const RESTART_STATE_VERSION_V1: u16 = 1;
const RESTART_STATE_BYTES: usize = 4 + 2 + 16 + 32 + 4 + 1 + 32 + 32;

/// Zeroizing plaintext representation accepted only by the sealed journal.
pub(crate) struct OnionReplySessionRestartState {
    bytes: Vec<u8>,
}

impl OnionReplySessionRestartState {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for OnionReplySessionRestartState {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnionReplySessionRestartError {
    Malformed,
    UnsupportedVersion,
    InvalidSession,
}

/// Module-private serializable-key foundation with consuming DH semantics.
///
/// [ONION-REPLY-RECOVERABLE-KEY 2026-08-29 by Codex] `StaticSecret` is used
/// only because restart persistence eventually needs a byte representation.
/// The wrapper retains the stronger one-use ownership contract by taking
/// `self` for key agreement and never implementing `Clone`.
struct RecoverableOnionReplyKey {
    secret: Option<StaticSecret>,
    public: X25519PublicKey,
}

impl RecoverableOnionReplyKey {
    fn generate() -> Self {
        let secret = StaticSecret::new(OsRng);
        let public = X25519PublicKey::from(&secret);
        Self {
            secret: Some(secret),
            public,
        }
    }

    fn persistence_secret(&self) -> Result<[u8; 32], OnionReplySessionRestartError> {
        self.secret
            .as_ref()
            .map(StaticSecret::to_bytes)
            .ok_or(OnionReplySessionRestartError::InvalidSession)
    }

    fn from_persistence_secret(
        mut secret_bytes: [u8; 32],
    ) -> Result<Self, OnionReplySessionRestartError> {
        if secret_bytes == [0; 32] {
            secret_bytes.zeroize();
            return Err(OnionReplySessionRestartError::InvalidSession);
        }
        let secret = StaticSecret::from(secret_bytes);
        secret_bytes.zeroize();
        let public = X25519PublicKey::from(&secret);
        Ok(Self {
            secret: Some(secret),
            public,
        })
    }
}

impl ReplyKeyAgreement for RecoverableOnionReplyKey {
    fn public_key_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    fn exchange(mut self, peer_public: &[u8; 32]) -> Result<[u8; 32], OnionReplyError> {
        let secret = self.secret.take().ok_or(OnionReplyError::InvalidReplyKey)?;
        let peer = X25519PublicKey::from(*peer_public);
        let shared = secret.diffie_hellman(&peer);
        Ok(*shared.as_bytes())
    }
}

/// Source-owned single-use state for one encrypted terminal response.
///
/// The private reply key never enters the request or a node API. Consuming
/// `self` on open makes key reuse and accepting two responses for one logical
/// route impossible by type.
pub struct OnionReplySession {
    route_id: [u8; 16],
    expected_terminal_node_id: [u8; 32],
    response_size_class: usize,
    proof_mode: OnionReplyProofMode,
    request_context_commitment: [u8; 32],
    reply_key: RecoverableOnionReplyKey,
}

impl OnionReplySession {
    /// Creates one request and retains the corresponding private reply state.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
    ) -> Result<(OnionReplyRequest, Self), OnionReplyError> {
        Self::prepare_with_mode(
            route_id,
            expected_terminal_node_id,
            response_size_class,
            payload,
            OnionReplyProofMode::RelayVisibleTerminalReceipt,
        )
    }

    /// Creates one v2 request that keeps final proof private to the source.
    pub fn prepare_source_sealed(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
    ) -> Result<(OnionReplyRequest, Self), OnionReplyError> {
        Self::prepare_with_mode(
            route_id,
            expected_terminal_node_id,
            response_size_class,
            payload,
            OnionReplyProofMode::SourceSealedTerminalProof,
        )
    }

    fn prepare_with_mode(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        response_size_class: usize,
        payload: Vec<u8>,
        proof_mode: OnionReplyProofMode,
    ) -> Result<(OnionReplyRequest, Self), OnionReplyError> {
        super::IdentityPublicKey::from_bytes(&expected_terminal_node_id)
            .map_err(|_| OnionReplyError::InvalidTerminalIdentity)?;
        let reply_key = RecoverableOnionReplyKey::generate();
        let request = match proof_mode {
            OnionReplyProofMode::RelayVisibleTerminalReceipt => {
                OnionReplyRequest::new(reply_key.public_key_bytes(), response_size_class, payload)?
            }
            OnionReplyProofMode::SourceSealedTerminalProof => OnionReplyRequest::new_source_sealed(
                reply_key.public_key_bytes(),
                response_size_class,
                payload,
            )?,
        };
        let request_context_commitment = request_context_commitment(&request);
        Ok((
            request,
            Self {
                route_id,
                expected_terminal_node_id,
                response_size_class,
                proof_mode,
                request_context_commitment,
                reply_key,
            },
        ))
    }

    /// Decodes, decrypts, and verifies the one response bound to this session.
    pub fn open(self, encoded_response: &[u8]) -> Result<OnionReplyPayload, OnionReplyError> {
        let response = super::decode_onion_sealed_response(encoded_response)?;
        open_onion_reply_with_context(
            self.route_id,
            &response,
            self.reply_key,
            self.expected_terminal_node_id,
            self.response_size_class,
            self.request_context_commitment,
        )
    }

    /// Rebuilds only the exact request originally bound to this session.
    ///
    /// The caller supplies payload bytes from its private manifest or durable
    /// request store. A different payload, mode, reply key, or size class
    /// produces a different commitment and fails closed before network I/O.
    pub fn rebuild_request(&self, payload: Vec<u8>) -> Result<OnionReplyRequest, OnionReplyError> {
        let request = match self.proof_mode {
            OnionReplyProofMode::RelayVisibleTerminalReceipt => OnionReplyRequest::new(
                self.reply_key.public_key_bytes(),
                self.response_size_class,
                payload,
            )?,
            OnionReplyProofMode::SourceSealedTerminalProof => OnionReplyRequest::new_source_sealed(
                self.reply_key.public_key_bytes(),
                self.response_size_class,
                payload,
            )?,
        };
        if request_context_commitment(&request) != self.request_context_commitment {
            return Err(OnionReplyError::RequestMismatch);
        }
        Ok(request)
    }

    /// Checks whether encoded terminal bytes belong to this exact session.
    ///
    /// [ONION-REPLY-ENCODED-REQUEST-BINDING 2026-08-29 by Codex] This reads
    /// only the public request and retained commitment. It neither performs DH
    /// nor consumes or exports the private one-time reply key.
    pub(crate) fn matches_encoded_request(&self, encoded_request: &[u8]) -> bool {
        super::decode_onion_reply_request(encoded_request).is_ok_and(|request| {
            request.reply_public_key == self.reply_key.public_key_bytes()
                && request_context_commitment(&request) == self.request_context_commitment
        })
    }

    /// Encodes plaintext state solely for immediate identity-sealed journaling.
    ///
    /// [ONION-REPLY-SESSION-RESTART-STATE 2026-08-29 by Codex] This remains
    /// crate-private so no App, FFI, node API, or protocol caller can export
    /// the reply secret. The zeroizing wrapper must not be persisted directly.
    pub(crate) fn encode_restart_state(
        &self,
    ) -> Result<OnionReplySessionRestartState, OnionReplySessionRestartError> {
        let encoded_size_class = u32::try_from(self.response_size_class)
            .map_err(|_| OnionReplySessionRestartError::InvalidSession)?;
        super::response_size_class(encoded_size_class)
            .map_err(|_| OnionReplySessionRestartError::InvalidSession)?;
        let mut secret_bytes = self.reply_key.persistence_secret()?;
        let mut bytes = Vec::with_capacity(RESTART_STATE_BYTES);
        bytes.extend_from_slice(&RESTART_STATE_MAGIC);
        bytes.extend_from_slice(&RESTART_STATE_VERSION_V1.to_be_bytes());
        bytes.extend_from_slice(&self.route_id);
        bytes.extend_from_slice(&self.expected_terminal_node_id);
        bytes.extend_from_slice(&encoded_size_class.to_be_bytes());
        bytes.push(self.proof_mode as u8);
        bytes.extend_from_slice(&self.request_context_commitment);
        bytes.extend_from_slice(&secret_bytes);
        secret_bytes.zeroize();
        Ok(OnionReplySessionRestartState { bytes })
    }

    /// Restores one unconsumed session from authenticated journal plaintext.
    pub(crate) fn decode_restart_state(
        bytes: &[u8],
    ) -> Result<Self, OnionReplySessionRestartError> {
        if bytes.len() != RESTART_STATE_BYTES || bytes[..4] != RESTART_STATE_MAGIC {
            return Err(OnionReplySessionRestartError::Malformed);
        }
        if u16::from_be_bytes([bytes[4], bytes[5]]) != RESTART_STATE_VERSION_V1 {
            return Err(OnionReplySessionRestartError::UnsupportedVersion);
        }

        let mut route_id = [0u8; 16];
        route_id.copy_from_slice(&bytes[6..22]);
        let mut expected_terminal_node_id = [0u8; 32];
        expected_terminal_node_id.copy_from_slice(&bytes[22..54]);
        super::IdentityPublicKey::from_bytes(&expected_terminal_node_id)
            .map_err(|_| OnionReplySessionRestartError::InvalidSession)?;
        let encoded_size_class = u32::from_be_bytes(
            bytes[54..58]
                .try_into()
                .map_err(|_| OnionReplySessionRestartError::Malformed)?,
        );
        let response_size_class = super::response_size_class(encoded_size_class)
            .map_err(|_| OnionReplySessionRestartError::InvalidSession)?;
        let proof_mode = OnionReplyProofMode::try_from(bytes[58])
            .map_err(|_| OnionReplySessionRestartError::InvalidSession)?;
        let mut request_context_commitment = [0u8; 32];
        request_context_commitment.copy_from_slice(&bytes[59..91]);
        let mut secret_bytes = [0u8; 32];
        secret_bytes.copy_from_slice(&bytes[91..123]);
        let reply_key = RecoverableOnionReplyKey::from_persistence_secret(secret_bytes);
        secret_bytes.zeroize();
        let reply_key = reply_key?;

        Ok(Self {
            route_id,
            expected_terminal_node_id,
            response_size_class,
            proof_mode,
            request_context_commitment,
            reply_key,
        })
    }
}
