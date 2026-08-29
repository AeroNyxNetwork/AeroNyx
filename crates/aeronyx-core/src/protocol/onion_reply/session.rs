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
//! Last Modified: v1.0.0-RecoverableSessionKey - Initial focused module.
//! ============================================

use rand::rngs::OsRng;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};

use super::{
    open_onion_reply_with_context, request_context_commitment, OnionReplyError, OnionReplyPayload,
    OnionReplyProofMode, OnionReplyRequest, ReplyKeyAgreement,
};

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
}
