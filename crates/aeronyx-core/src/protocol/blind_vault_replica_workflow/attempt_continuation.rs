// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/attempt_continuation.rs
// ============================================
//! Typed private continuation state for restart-safe replica dispatch.
//!
//! ## Creation Reason
//! A raw opaque journal can protect adapter bytes but cannot safely export an
//! `OnionReplySession` private key. Replacement and provisioning need those
//! exact single-use sessions after restart to open terminal responses without
//! generating a different request identity or repeating blind side effects.
//!
//! ## Main Functionality
//! - Owns bounded adapter state and single-use onion reply sessions.
//! - Encodes session state only into the identity-sealed attempt journal.
//! - Restores typed sessions after workflow and journal authentication.
//! - Zeroizes adapter plaintext and all temporary continuation encodings.
//! - Preserves ownership by avoiding `Clone` and redacting Debug output.
//!
//! ## Dependencies
//! - `attempt_journal.rs`: action/attempt/deadline-bound encrypted container.
//! - `protocol::onion_reply/session.rs`: crate-private session restart codec.
//!
//! ## Main Logical Flow
//! 1. Prepare terminal requests and retain their source reply sessions here.
//! 2. Prepare and persist a typed journal before committing dispatch.
//! 3. Persist the dispatched workflow snapshot before sending requests.
//! 4. On restart, authenticate both records and restore this continuation.
//! 5. Consume each restored session while opening its exact terminal reply.
//!
//! ## Important Note For The Next Developer
//! - Never add request payloads that are not required for exact continuation.
//! - Never expose the encoded continuation or session restart bytes publicly.
//! - Session order is adapter-owned and must match the prepared request order.
//! - Use `BlindVaultReplicaBoundAttemptContinuation` when payload identity and
//!   send order must be verified after restart; this low-level type only owns
//!   adapter state and reply sessions.
//! - A continuation without a reply session must use the opaque journal path.
//!
//! Last Modified: v1.1.0-BoundAttemptComposition - Clarified this low-level
//! session owner and exposed sibling-only restart codec composition.
//! v1.0.0-TypedAttemptContinuation - Initial typed recovery.
//! ============================================

use std::{fmt, mem};

use zeroize::Zeroize;

use super::{
    BlindVaultReplicaAttemptJournal, BlindVaultReplicaAttemptJournalError,
    BlindVaultReplicaExecution, BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaWorkId,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES,
};
use crate::crypto::keys::IdentityKeyPair;
use crate::protocol::onion_reply::{
    OnionReplySession, OnionReplySessionRestartError, OnionReplySessionRestartState,
};

const CONTINUATION_MAGIC: [u8; 4] = *b"AXAC";
const CONTINUATION_VERSION_V1: u16 = 1;
const CONTINUATION_HEADER_BYTES: usize = 4 + 2 + 4 + 2;

/// Maximum sessions retained for one bounded replacement/provisioning attempt.
pub const MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS: usize = 64;

/// Maximum adapter-owned bytes inside one typed attempt continuation.
pub const MAX_BLIND_VAULT_REPLICA_ATTEMPT_ADAPTER_STATE_BYTES: usize = 16 * 1024;

/// Private continuation used before dispatch and reconstructed after restart.
///
/// [BLIND-VAULT-TYPED-ATTEMPT-CONTINUATION 2026-08-29 by Codex] Session
/// restart bytes never cross this composition boundary without first entering
/// the action-bound, identity-sealed attempt journal.
pub struct BlindVaultReplicaAttemptContinuation {
    adapter_state: Vec<u8>,
    reply_sessions: Vec<OnionReplySession>,
}

struct ContinuationPlaintext(Vec<u8>);

impl ContinuationPlaintext {
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    fn into_bytes(mut self) -> Vec<u8> {
        mem::take(&mut self.0)
    }
}

impl Drop for ContinuationPlaintext {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl BlindVaultReplicaAttemptContinuation {
    /// Creates bounded adapter state; add at least one reply session before seal.
    pub fn new(mut adapter_state: Vec<u8>) -> Result<Self, BlindVaultReplicaAttemptJournalError> {
        if adapter_state.len() > MAX_BLIND_VAULT_REPLICA_ATTEMPT_ADAPTER_STATE_BYTES {
            adapter_state.zeroize();
            return Err(BlindVaultReplicaAttemptJournalError::TooLarge);
        }
        Ok(Self {
            adapter_state,
            reply_sessions: Vec::new(),
        })
    }

    /// Adds one independently generated, single-use terminal reply session.
    pub fn push_reply_session(
        &mut self,
        session: OnionReplySession,
    ) -> Result<(), BlindVaultReplicaAttemptJournalError> {
        if self.reply_sessions.len() >= MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS {
            return Err(BlindVaultReplicaAttemptJournalError::TooManyReplySessions);
        }
        self.reply_sessions.push(session);
        Ok(())
    }

    /// Borrows adapter-owned state without copying it.
    #[must_use]
    pub fn adapter_state(&self) -> &[u8] {
        &self.adapter_state
    }

    /// Number of exact single-use sessions retained by this continuation.
    #[must_use]
    pub fn reply_session_count(&self) -> usize {
        self.reply_sessions.len()
    }

    /// Transfers private adapter state and session ownership to the runtime.
    #[must_use]
    pub fn into_parts(mut self) -> (Vec<u8>, Vec<OnionReplySession>) {
        (
            mem::take(&mut self.adapter_state),
            mem::take(&mut self.reply_sessions),
        )
    }

    pub(super) fn encode_restart_state(
        &self,
    ) -> Result<Vec<u8>, BlindVaultReplicaAttemptJournalError> {
        if self.reply_sessions.is_empty() {
            return Err(BlindVaultReplicaAttemptJournalError::ReplySessionsRequired);
        }
        if self.reply_sessions.len() > MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS {
            return Err(BlindVaultReplicaAttemptJournalError::TooManyReplySessions);
        }
        let adapter_len = u32::try_from(self.adapter_state.len())
            .map_err(|_| BlindVaultReplicaAttemptJournalError::TooLarge)?;
        let session_count = u16::try_from(self.reply_sessions.len())
            .map_err(|_| BlindVaultReplicaAttemptJournalError::TooManyReplySessions)?;
        let mut encoded = ContinuationPlaintext::with_capacity(
            CONTINUATION_HEADER_BYTES
                .saturating_add(self.adapter_state.len())
                .saturating_add(self.reply_sessions.len().saturating_mul(128)),
        );
        encoded.0.extend_from_slice(&CONTINUATION_MAGIC);
        encoded
            .0
            .extend_from_slice(&CONTINUATION_VERSION_V1.to_be_bytes());
        encoded.0.extend_from_slice(&adapter_len.to_be_bytes());
        encoded.0.extend_from_slice(&session_count.to_be_bytes());
        encoded.0.extend_from_slice(&self.adapter_state);
        for session in &self.reply_sessions {
            let restart_state: OnionReplySessionRestartState = session
                .encode_restart_state()
                .map_err(map_session_restart_error)?;
            let state_len = u16::try_from(restart_state.as_bytes().len())
                .map_err(|_| BlindVaultReplicaAttemptJournalError::TooLarge)?;
            encoded.0.extend_from_slice(&state_len.to_be_bytes());
            encoded.0.extend_from_slice(restart_state.as_bytes());
        }
        if encoded.0.len() > MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES {
            return Err(BlindVaultReplicaAttemptJournalError::TooLarge);
        }
        Ok(encoded.into_bytes())
    }

    pub(super) fn decode_restart_state(
        encoded: &[u8],
    ) -> Result<Self, BlindVaultReplicaAttemptJournalError> {
        if encoded.len() < CONTINUATION_HEADER_BYTES || encoded[..4] != CONTINUATION_MAGIC {
            return Err(BlindVaultReplicaAttemptJournalError::ContinuationMalformed);
        }
        if u16::from_be_bytes([encoded[4], encoded[5]]) != CONTINUATION_VERSION_V1 {
            return Err(BlindVaultReplicaAttemptJournalError::ContinuationVersionUnsupported);
        }
        let adapter_len = u32::from_be_bytes([encoded[6], encoded[7], encoded[8], encoded[9]]);
        let adapter_len = usize::try_from(adapter_len)
            .map_err(|_| BlindVaultReplicaAttemptJournalError::ContinuationMalformed)?;
        let session_count = usize::from(u16::from_be_bytes([encoded[10], encoded[11]]));
        if adapter_len > MAX_BLIND_VAULT_REPLICA_ATTEMPT_ADAPTER_STATE_BYTES {
            return Err(BlindVaultReplicaAttemptJournalError::TooLarge);
        }
        if session_count == 0 {
            return Err(BlindVaultReplicaAttemptJournalError::ReplySessionsRequired);
        }
        if session_count > MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS {
            return Err(BlindVaultReplicaAttemptJournalError::TooManyReplySessions);
        }

        let mut offset = CONTINUATION_HEADER_BYTES;
        let adapter_bytes = take_part(encoded, &mut offset, adapter_len)?;
        let mut continuation = Self {
            adapter_state: adapter_bytes.to_vec(),
            reply_sessions: Vec::with_capacity(session_count),
        };
        for _ in 0..session_count {
            let length_bytes = take_part(encoded, &mut offset, 2)?;
            let state_len = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
            if state_len == 0 {
                return Err(BlindVaultReplicaAttemptJournalError::ContinuationMalformed);
            }
            let state_bytes = take_part(encoded, &mut offset, state_len)?;
            let session = OnionReplySession::decode_restart_state(state_bytes)
                .map_err(map_session_restart_error)?;
            continuation.reply_sessions.push(session);
        }
        if offset != encoded.len() {
            return Err(BlindVaultReplicaAttemptJournalError::ContinuationMalformed);
        }
        Ok(continuation)
    }
}

impl Drop for BlindVaultReplicaAttemptContinuation {
    fn drop(&mut self) {
        self.adapter_state.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaAttemptContinuation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaAttemptContinuation")
            .field("adapter_state", &"<redacted>")
            .field("reply_session_count", &self.reply_sessions.len())
            .finish_non_exhaustive()
    }
}

impl BlindVaultReplicaExecution {
    /// Prepares a journal containing exact recoverable onion reply sessions.
    pub fn prepare_attempt_continuation_for_dispatch(
        &self,
        identity: &IdentityKeyPair,
        work_id: BlindVaultReplicaWorkId,
        planned_dispatch_at_ms: u64,
        journal_sequence: u64,
        retain_until_ms: u64,
        continuation: &BlindVaultReplicaAttemptContinuation,
    ) -> Result<BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaAttemptJournalError> {
        let mut encoded = continuation.encode_restart_state()?;
        let prepared = self.prepare_attempt_journal_for_dispatch(
            identity,
            work_id,
            planned_dispatch_at_ms,
            journal_sequence,
            retain_until_ms,
            &encoded,
        );
        encoded.zeroize();
        prepared
    }
}

impl BlindVaultReplicaAttemptJournal {
    /// Restores typed adapter state and single-use reply sessions after restart.
    pub fn into_attempt_continuation(
        self,
    ) -> Result<BlindVaultReplicaAttemptContinuation, BlindVaultReplicaAttemptJournalError> {
        let mut encoded = self.into_private_state();
        let continuation = BlindVaultReplicaAttemptContinuation::decode_restart_state(&encoded);
        encoded.zeroize();
        continuation
    }
}

fn take_part<'a>(
    encoded: &'a [u8],
    offset: &mut usize,
    length: usize,
) -> Result<&'a [u8], BlindVaultReplicaAttemptJournalError> {
    let end = offset
        .checked_add(length)
        .ok_or(BlindVaultReplicaAttemptJournalError::ContinuationMalformed)?;
    let part = encoded
        .get(*offset..end)
        .ok_or(BlindVaultReplicaAttemptJournalError::ContinuationMalformed)?;
    *offset = end;
    Ok(part)
}

fn map_session_restart_error(
    error: OnionReplySessionRestartError,
) -> BlindVaultReplicaAttemptJournalError {
    match error {
        OnionReplySessionRestartError::UnsupportedVersion => {
            BlindVaultReplicaAttemptJournalError::ContinuationVersionUnsupported
        }
        OnionReplySessionRestartError::Malformed
        | OnionReplySessionRestartError::InvalidSession => {
            BlindVaultReplicaAttemptJournalError::ContinuationMalformed
        }
    }
}
