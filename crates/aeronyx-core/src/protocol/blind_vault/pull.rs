// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/pull.rs
// ============================================
//! # Encrypted-object recovery
//!
//! Owns the capability-authenticated pull request, recovered objects, the
//! node-signed recovery page, and the single-use onion pull session.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::onion_reply::{
    encode_onion_reply_request, OnionReplyError, OnionReplySession,
    ONION_REPLY_RESPONSE_SIZE_CLASSES,
};

use super::error::BlindVaultError;
use super::frame::{decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame};
use super::terminal_failure::{BlindVaultTerminalFailureCode, BlindVaultTerminalOperation};
use super::{
    require_non_zero, require_version, serde_bytes64, sha256, BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES,
    BLIND_VAULT_PROTOCOL_VERSION, PULL_RESPONSE_SIGNING_DOMAIN,
};

/// Maximum objects returned by one signed recovery page.
pub const MAX_BLIND_VAULT_PULL_OBJECTS: usize = 16;

/// Maximum opaque server cursor carried by a recovery frame.
pub const MAX_BLIND_VAULT_PULL_CURSOR_BYTES: usize = 128;

/// Fixed response class used by anonymous single-object recovery.
///
/// [BLIND-VAULT-ONION-PULL-SIZE 2026-08-30 by Codex] A lease may contain any
/// valid ciphertext class, including objects written through the bounded
/// direct API. Always reserving the largest response class keeps the object's
/// size hidden from relay hops and guarantees that one valid object plus its
/// signed metadata can be sealed. Route construction must require the matching
/// path-wide feature before using this class.
pub const BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS: usize =
    ONION_REPLY_RESPONSE_SIZE_CLASSES[ONION_REPLY_RESPONSE_SIZE_CLASSES.len() - 1];

/// Capability-authenticated request for one stable recovery snapshot page.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultPullRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Random replica-local lease identifier.
    pub lease_id: [u8; 32],
    /// Random bearer read capability whose hash was committed at lease setup.
    pub read_capability: [u8; 32],
    /// Empty for the first page; otherwise a node-encrypted snapshot cursor.
    pub continuation_cursor: Vec<u8>,
    /// Requested page size, bounded again by node policy.
    pub limit: u16,
}

impl std::fmt::Debug for BlindVaultPullRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultPullRequest")
            .field("version", &self.version)
            .field("limit", &self.limit)
            .field("has_continuation", &!self.continuation_cursor.is_empty())
            .field("capability", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl BlindVaultPullRequest {
    /// Validates fixed identifiers and protocol-wide recovery bounds.
    pub fn validate(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("read_capability", &self.read_capability)?;
        if self.limit == 0 || usize::from(self.limit) > MAX_BLIND_VAULT_PULL_OBJECTS {
            return Err(BlindVaultError::InvalidPullLimit);
        }
        if self.continuation_cursor.len() > MAX_BLIND_VAULT_PULL_CURSOR_BYTES {
            return Err(BlindVaultError::InvalidPullCursorLength);
        }
        Ok(())
    }
}

/// One immutable ciphertext object returned by a Blind Vault recovery page.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultRecoveredObject {
    /// Replica-local immutable object identifier.
    pub object_id: [u8; 32],
    /// Exact client ciphertext including its AEAD nonce, tag, and padding.
    pub ciphertext: Vec<u8>,
    /// SHA-256 commitment to the exact ciphertext bytes.
    pub ciphertext_commitment: [u8; 32],
    /// Object retention deadline in Unix milliseconds.
    pub expires_at_ms: u64,
}

impl std::fmt::Debug for BlindVaultRecoveredObject {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultRecoveredObject")
            .field("ciphertext_bytes", &self.ciphertext.len())
            .field("object", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl BlindVaultRecoveredObject {
    fn validate_at(&self, generated_at_ms: u64) -> Result<(), BlindVaultError> {
        require_non_zero("object_id", &self.object_id)?;
        if !BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES.contains(&self.ciphertext.len()) {
            return Err(BlindVaultError::InvalidCiphertextSize {
                actual: self.ciphertext.len(),
            });
        }
        if sha256(&self.ciphertext) != self.ciphertext_commitment {
            return Err(BlindVaultError::CommitmentMismatch);
        }
        if self.expires_at_ms <= generated_at_ms {
            return Err(BlindVaultError::Expired);
        }
        Ok(())
    }
}

/// Node-signed bounded recovery page.
///
/// The signature authenticates ciphertext commitments rather than hashing the
/// multi-megabyte ciphertext a second time. Validation recomputes each stored
/// commitment first, so the signature remains bound to the exact bytes.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultPullResponse {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease answered by this page.
    pub lease_id: [u8; 32],
    /// Bounded immutable ciphertext objects in snapshot order.
    pub objects: Vec<BlindVaultRecoveredObject>,
    /// Empty when this snapshot is complete; otherwise node-encrypted cursor.
    pub continuation_cursor: Vec<u8>,
    /// Node generation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Descriptor identity of the responding storage node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over page commitments and metadata.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl std::fmt::Debug for BlindVaultPullResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultPullResponse")
            .field("version", &self.version)
            .field("object_count", &self.objects.len())
            .field("has_continuation", &!self.continuation_cursor.is_empty())
            .field("page", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl BlindVaultPullResponse {
    /// Builds an unsigned recovery page.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        objects: Vec<BlindVaultRecoveredObject>,
        continuation_cursor: Vec<u8>,
        generated_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            objects,
            continuation_cursor,
            generated_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical signing input containing each already-validated ciphertext
    /// commitment, object metadata, and opaque continuation cursor.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let object_bytes = self.objects.len().saturating_mul(76);
        let mut bytes = Vec::with_capacity(
            PULL_RESPONSE_SIGNING_DOMAIN.len() + 76 + object_bytes + self.continuation_cursor.len(),
        );
        bytes.extend_from_slice(PULL_RESPONSE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&(self.objects.len() as u16).to_be_bytes());
        for object in &self.objects {
            bytes.extend_from_slice(&object.object_id);
            bytes.extend_from_slice(&(object.ciphertext.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&object.ciphertext_commitment);
            bytes.extend_from_slice(&object.expires_at_ms.to_be_bytes());
        }
        bytes.extend_from_slice(&(self.continuation_cursor.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&self.continuation_cursor);
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }

    /// Validates page bounds and signs with the responding node identity.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates all ciphertext commitments, node identity, and page signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        if self.objects.len() > MAX_BLIND_VAULT_PULL_OBJECTS {
            return Err(BlindVaultError::InvalidPullLimit);
        }
        if self.continuation_cursor.len() > MAX_BLIND_VAULT_PULL_CURSOR_BYTES {
            return Err(BlindVaultError::InvalidPullCursorLength);
        }
        for object in &self.objects {
            object.validate_at(self.generated_at_ms)?;
        }
        Ok(())
    }
}

/// Source-owned state for one anonymous inline Blind Vault recovery request.
///
/// [BLIND-VAULT-ONION-PULL-SESSION 2026-08-28 by Codex] Capability bytes are
/// retained only inside the encoded final onion payload; the session keeps a
/// request commitment and single-use reply key rather than a plaintext copy.
///
/// The encoded request returned by [`Self::prepare`] is the final payload for
/// `build_onion_envelope`; it must never be sent directly to a storage node.
/// The retained state contains the single-use reply private key and is consumed
/// when opening one response.
pub struct BlindVaultOnionPullSession {
    lease_id: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionPullSession {
    /// Encodes one capability-bearing pull for the final onion layer.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultPullRequest,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionPullError> {
        request.validate()?;
        if request.limit != 1 {
            return Err(BlindVaultOnionPullError::UnsupportedInlinePullLimit);
        }
        let lease_id = request.lease_id;
        let encoded_pull = encode_blind_vault_frame(&BlindVaultFrame::PullRequest(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS,
            encoded_pull,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                lease_id,
                reply_session,
            },
        ))
    }

    /// Opens the terminal response and verifies the inner signed recovery page.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultPullResponse, BlindVaultOnionPullError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::Pull {
                return Err(BlindVaultOnionPullError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionPullError::TerminalFailure(failure.code()));
        }
        let BlindVaultFrame::PullResponse(response) = frame else {
            return Err(BlindVaultOnionPullError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionPullError::InvalidTerminalIdentity)?;
        response.validate_and_verify(&terminal_key)?;
        if response.lease_id != self.lease_id {
            return Err(BlindVaultOnionPullError::LeaseMismatch);
        }
        Ok(response)
    }
}

/// Fail-closed source errors for anonymous inline Blind Vault recovery.
#[derive(Debug, Error)]
pub enum BlindVaultOnionPullError {
    /// The Blind Vault request or signed response violated its wire contract.
    #[error("blind vault onion pull frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, identity, size, or signature checks.
    #[error("blind vault onion reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion pull terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// Inline v1 deliberately returns at most one ciphertext object.
    #[error("blind vault onion pull requires limit 1")]
    UnsupportedInlinePullLimit,
    /// The decrypted workload response was not a Blind Vault pull page.
    #[error("unexpected blind vault onion response frame")]
    UnexpectedResponseFrame,
    /// The verified page belongs to another replica-local lease.
    #[error("blind vault onion pull lease mismatch")]
    LeaseMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion terminal identity")]
    InvalidTerminalIdentity,
}

#[cfg(test)]
mod tests;
