// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/verified_submit.rs
// ============================================
//! # Verified chat submit
//!
//! Owns the opt-in verified client onion delivery contract: the fixed
//! `CHAT_VERIFIED_SUBMIT_*_V1` result vocabulary and its labels, the evidence
//! to result-code mapping, retry-stable route id derivation, and the signed
//! `ChatRelayVerifiedSubmitRequestV1` / `ChatRelayVerifiedSubmitResponseV1`
//! frames carried by message variants 38-39.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::protocol::auth::{
    signed_message_digest, verify_signed_message, AuthError, DOMAIN_CHAT_VERIFIED_SUBMIT_V1,
};
use crate::protocol::chat::{encode_envelope, BlindRelayDeliveryReceipt, ChatEnvelope};
use crate::protocol::onion::OnionRoutePurpose;

use super::serde_bytes64;

/// Verified onion delivery and durable entry-node custody both succeeded.
pub const CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1: u8 = 0;
/// A terminal signed durable custody, but entry-node persistence failed.
pub const CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1: u8 = 1;
/// Entry-node custody succeeded while no verified onion route completed.
pub const CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1: u8 = 2;
/// Neither verified terminal delivery nor entry-node custody succeeded.
pub const CHAT_VERIFIED_SUBMIT_REJECTED_V1: u8 = 3;

/// Privacy-safe label for the fixed verified-submit result vocabulary.
///
/// [CHAT-VERIFIED-SUBMIT-RESULT-LABELS 2026-08-23 by Codex] This helper is the
/// canonical mapping used by health telemetry, SDKs, and dashboards. It maps
/// only closed result codes and never accepts or returns route, receipt,
/// message, endpoint, wallet, or payload metadata.
#[must_use]
pub fn chat_verified_submit_result_label(result: u8) -> Option<&'static str> {
    match result {
        CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1 => Some("onion_and_entry"),
        CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1 => Some("onion_only"),
        CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1 => Some("entry_retry"),
        CHAT_VERIFIED_SUBMIT_REJECTED_V1 => Some("rejected"),
        _ => None,
    }
}

/// Maps independent terminal proof and entry custody evidence into one result code.
///
/// [CHAT-VERIFIED-SUBMIT-OUTCOME-MAPPING 2026-08-23 by Codex] Keep this table
/// in the protocol crate so server implementations, SDKs, and future
/// compatibility shims cannot drift into different client-visible semantics.
/// The two booleans are already aggregate evidence; no route, endpoint,
/// receipt, wallet, payload, or message metadata enters this mapping.
#[must_use]
pub fn chat_verified_submit_result_for_outcomes(verified_onion: bool, entry_custody: bool) -> u8 {
    match (verified_onion, entry_custody) {
        (true, true) => CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
        (true, false) => CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1,
        (false, true) => CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
        (false, false) => CHAT_VERIFIED_SUBMIT_REJECTED_V1,
    }
}

/// Derives one retry-stable route id for an explicit verified submit path.
///
/// [CHAT-VERIFIED-SUBMIT-ROUTE-ID 2026-08-23 by Codex] This derivation is
/// domain-separated from probes and random legacy relay route ids. The selected
/// source, middle, and terminal identities are included so the same client
/// request id cannot replay across independent path surfaces, while only the
/// resulting 16-byte route id needs to cross the relay boundary.
#[must_use]
pub fn chat_verified_submit_route_id(
    request_id: &[u8; 16],
    source_node_id: &[u8; 32],
    middle_node_id: &[u8; 32],
    terminal_node_id: &[u8; 32],
) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-ChatVerifiedSubmit-Route-v1");
    hasher.update(request_id);
    hasher.update(source_node_id);
    hasher.update(middle_node_id);
    hasher.update(terminal_node_id);
    let digest = hasher.finalize();
    let mut route_id = [0u8; 16];
    route_id.copy_from_slice(&digest[..16]);
    route_id
}

/// Opt-in client request for a terminal-verifiable onion chat delivery.
///
/// [CHAT-VERIFIED-SUBMIT 2026-08-22 by Codex] The request is authenticated
/// independently from the encrypted VPN session. Its signature binds the
/// random request id, exact signed envelope commitment, and freshness window.
/// It carries no route, endpoint, terminal, or plaintext metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRelayVerifiedSubmitRequestV1 {
    /// Random client-generated id used for response binding and retry stability.
    pub request_id: [u8; 16],
    /// Exact sender-signed E2E envelope to deliver and durably retain.
    pub envelope: ChatEnvelope,
    /// Unix epoch seconds covered by the request signature.
    pub request_timestamp: u64,
    /// Sender signature over request id, envelope commitment, and timestamp.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl ChatRelayVerifiedSubmitRequestV1 {
    /// Constructs a sender-signed request over one already-signed envelope.
    pub fn signed(
        request_id: [u8; 16],
        envelope: ChatEnvelope,
        request_timestamp: u64,
        sender: &IdentityKeyPair,
    ) -> Result<Self, CoreError> {
        if !envelope.sender_matches_authenticated_identity(&sender.public_key_bytes()) {
            return Err(CoreError::malformed(
                "verified chat submit: envelope sender mismatch",
            ));
        }
        envelope.verify_signature()?;
        let mut request = Self {
            request_id,
            envelope,
            request_timestamp,
            signature: [0u8; 64],
        };
        request.signature = sender.sign(&request.signing_digest());
        Ok(request)
    }

    /// Fixed-size commitment to every signed envelope byte.
    #[must_use]
    pub fn envelope_commitment(&self) -> [u8; 32] {
        let sign_data = self.envelope.sign_data();
        signed_message_digest(
            "AeroNyx-ChatEnvelopeCommitment-v1",
            &[sign_data.as_slice(), self.envelope.signature.as_ref()],
        )
    }

    /// Canonical request digest signed by the authenticated sender.
    #[must_use]
    pub fn signing_digest(&self) -> [u8; 32] {
        let envelope_commitment = self.envelope_commitment();
        let request_timestamp = self.request_timestamp.to_le_bytes();
        signed_message_digest(
            DOMAIN_CHAT_VERIFIED_SUBMIT_V1,
            &[
                self.request_id.as_ref(),
                envelope_commitment.as_ref(),
                request_timestamp.as_ref(),
            ],
        )
    }

    /// Verifies both signed layers without admitting a new delivery effect.
    ///
    /// [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] This deliberately
    /// does not enforce timestamp freshness. Only an authenticated session may
    /// use it to read an unexpired matching durable completion; it is not
    /// permission to reserve, recover pending custody, or relay.
    pub fn verify_signatures_for_replay(&self) -> Result<(), AuthError> {
        self.envelope
            .verify_signature()
            .map_err(|_| AuthError::SignatureMismatch)?;
        let sender = IdentityPublicKey::from_bytes(&self.envelope.sender)
            .map_err(|_| AuthError::InvalidPublicKey)?;
        sender
            .verify(&self.signing_digest(), &self.signature)
            .map_err(|_| AuthError::SignatureMismatch)
    }

    /// Verifies request freshness, sender signature, and envelope signature.
    pub fn verify_authentication(&self) -> Result<(), AuthError> {
        self.envelope
            .verify_signature()
            .map_err(|_| AuthError::SignatureMismatch)?;
        let envelope_commitment = self.envelope_commitment();
        let request_timestamp = self.request_timestamp.to_le_bytes();
        verify_signed_message(
            DOMAIN_CHAT_VERIFIED_SUBMIT_V1,
            &[
                self.request_id.as_ref(),
                envelope_commitment.as_ref(),
                request_timestamp.as_ref(),
            ],
            &self.envelope.sender,
            &self.signature,
            self.request_timestamp,
        )
    }
}

/// Client-visible result for one explicit verified-onion submission.
///
/// The encrypted session authenticates the entry-node response. When present,
/// `terminal_receipt` independently proves that a terminal node durably
/// accepted the exact opaque `ChatEnvelope` bytes for message relay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayVerifiedSubmitResponseV1 {
    /// Exact client request id copied from the authenticated request.
    pub request_id: [u8; 16],
    /// Message id copied from the exact submitted envelope.
    pub message_id: [u8; 16],
    /// Fixed result code from the `CHAT_VERIFIED_SUBMIT_*_V1` constants.
    pub result: u8,
    /// Terminal-signed exact payload receipt for verified result codes only.
    pub terminal_receipt: Option<BlindRelayDeliveryReceipt>,
}

impl ChatRelayVerifiedSubmitResponseV1 {
    /// Builds the canonical fail-closed response for rejected submissions.
    ///
    /// [CHAT-VERIFIED-SUBMIT-REJECTED-RESPONSE 2026-08-23 by Codex] Keep this
    /// shape in the protocol crate so entry relays, SDK simulations, and future
    /// decentralized submit surfaces never drift on the rejected result code or
    /// accidentally attach terminal receipt bytes to a failed request.
    #[must_use]
    pub fn rejected(request_id: [u8; 16], message_id: [u8; 16]) -> Self {
        Self {
            request_id,
            message_id,
            result: CHAT_VERIFIED_SUBMIT_REJECTED_V1,
            terminal_receipt: None,
        }
    }

    /// Builds a response from independently observed delivery evidence.
    ///
    /// [CHAT-VERIFIED-SUBMIT-RESPONSE-EVIDENCE 2026-08-23 by Codex] A terminal
    /// receipt is retained only when the caller simultaneously observed a
    /// verified onion delivery and provided receipt bytes. Any mismatch fails
    /// closed to the non-onion result for the entry-custody state.
    #[must_use]
    pub fn from_evidence(
        request_id: [u8; 16],
        message_id: [u8; 16],
        verified_onion: bool,
        entry_custody: bool,
        terminal_receipt: Option<BlindRelayDeliveryReceipt>,
    ) -> Self {
        let terminal_receipt = if verified_onion {
            terminal_receipt
        } else {
            None
        };
        let verified_onion = terminal_receipt.is_some();
        Self {
            request_id,
            message_id,
            result: chat_verified_submit_result_for_outcomes(verified_onion, entry_custody),
            terminal_receipt,
        }
    }

    /// Validates the closed result vocabulary and receipt-presence invariant.
    pub fn validate_shape(&self) -> Result<(), CoreError> {
        if self.result > CHAT_VERIFIED_SUBMIT_REJECTED_V1 {
            return Err(CoreError::malformed(
                "verified chat submit: unknown result code",
            ));
        }
        if self.verified_onion_delivery() != self.terminal_receipt.is_some() {
            return Err(CoreError::malformed(
                "verified chat submit: receipt/result mismatch",
            ));
        }
        Ok(())
    }

    /// Returns whether the entry node durably retained the envelope.
    #[must_use]
    pub fn entry_custody_accepted(&self) -> bool {
        matches!(
            self.result,
            CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1 | CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1
        )
    }

    /// Returns whether the result claims terminal-verifiable onion delivery.
    #[must_use]
    pub fn verified_onion_delivery(&self) -> bool {
        matches!(
            self.result,
            CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1 | CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1
        )
    }

    /// Validates response shape and correlation with one exact client request.
    ///
    /// [CHAT-VERIFIED-SUBMIT-RESPONSE-CORRELATION 2026-08-23 by Codex] Every
    /// result, including entry custody, retry, and rejection, must bind both
    /// request id and message id before it can update client or agent state.
    /// This keeps concurrent retries from consuming one another's response even
    /// when no terminal receipt exists to provide an additional payload proof.
    pub fn validate_for_request(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> Result<(), CoreError> {
        self.validate_shape()?;
        if self.request_id != request.request_id {
            return Err(CoreError::malformed(
                "verified chat submit: response request mismatch",
            ));
        }
        if self.message_id != request.envelope.message_id {
            return Err(CoreError::malformed(
                "verified chat submit: response message mismatch",
            ));
        }
        Ok(())
    }

    /// Verifies response shape, expected terminal identity, and exact payload.
    ///
    /// `expected_terminal_node_id` must come from the caller's independently
    /// verified signed node directory. Never trust the identity embedded in
    /// the receipt as its own trust root.
    pub fn verify_terminal_receipt(
        &self,
        envelope: &ChatEnvelope,
        expected_terminal_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        self.validate_shape()?;
        self.verify_terminal_receipt_payload(envelope, expected_terminal_node_id)
    }

    fn verify_terminal_receipt_payload(
        &self,
        envelope: &ChatEnvelope,
        expected_terminal_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        if !self.verified_onion_delivery() {
            return Err(CoreError::malformed(
                "verified chat submit: response has no onion delivery",
            ));
        }
        if self.message_id != envelope.message_id {
            return Err(CoreError::malformed(
                "verified chat submit: response message mismatch",
            ));
        }
        let receipt = self.terminal_receipt.as_ref().ok_or_else(|| {
            CoreError::malformed("verified chat submit: terminal receipt missing")
        })?;
        let payload = encode_envelope(envelope)
            .map_err(|_| CoreError::malformed("verified chat submit: envelope encoding failed"))?;
        receipt.verify_expected_for_purpose(
            &receipt.route_id,
            &payload,
            OnionRoutePurpose::MessageRelay,
            expected_terminal_node_id,
        )
    }

    /// Verifies terminal delivery against the exact client submission request.
    ///
    /// [CHAT-VERIFIED-SUBMIT-REQUEST-BINDING 2026-08-23 by Codex] Concurrent
    /// retries may carry the same signed envelope under distinct random request
    /// ids. SDKs and agents should use this method so an authenticated but stale
    /// response cannot satisfy another in-flight request merely because both
    /// refer to the same message. The existing envelope-only verifier remains
    /// available for backward compatibility and offline receipt inspection.
    pub fn verify_terminal_receipt_for_request(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
        expected_terminal_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        self.validate_for_request(request)?;
        self.verify_terminal_receipt_payload(&request.envelope, expected_terminal_node_id)
    }
}

#[cfg(test)]
mod tests;
