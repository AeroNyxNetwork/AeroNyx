// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/attempt_runtime.rs
// ============================================
//! Fail-closed runtime for ordered terminal effects and one-time replies.
//!
//! ## Creation Reason
//! The durable send capability and restart-safe reply sessions previously had
//! independent cursors. An integration could send effect N and accidentally
//! consume session N+1, or continue a compound action after an invalid reply.
//!
//! ## Main Functionality
//! - Owns one reply session for every remaining ordered terminal effect.
//! - Retains private adapter verification state without logging or cloning it.
//! - Preserves a reply session when transport fails before a response exists.
//! - Opens and semantically verifies each response before allowing the next.
//! - Poisons the complete runtime after any authenticated-reply failure.
//!
//! ## Dependencies
//! - `send_sequence.rs`: payload-bound ordered network authority.
//! - `attempt_continuation.rs`: adapter state and one-time reply ownership.
//! - `protocol::onion_reply`: source-only response authentication.
//!
//! ## Main Logical Flow
//! 1. Consume one send sequence and its exact private continuation.
//! 2. Verify that remaining effects and sessions have equal cardinality.
//! 3. Match the encoded request against the front session before network I/O.
//! 4. Send the current effect without removing its reply session first.
//! 5. On transport success, consume only the matching front session.
//! 6. Open the sealed response and require workload-specific verification.
//! 7. Advance to the next effect only while every invariant remains valid.
//!
//! ## Important Note For The Next Developer
//! - A successful I/O call is ambiguous even when reply verification fails.
//! - Never retry a poisoned runtime in memory; recover from its durable journal.
//! - Verifiers must validate workload frame, request identity, and signed result.
//! - Do not expose adapter state, reply keys, payloads, or work ids in telemetry.
//!
//! Last Modified: v1.0.0-TerminalAttemptRuntime - Initial ordered send,
//! one-time reply, and semantic verification composition.
//! ============================================

use std::{collections::VecDeque, error::Error as StdError, fmt};

use thiserror::Error;
use zeroize::Zeroize;

use super::send_sequence::{
    BlindVaultReplicaTerminalEffectTransport, BlindVaultReplicaTerminalSendContext,
    BlindVaultReplicaTerminalSendError, BlindVaultReplicaTerminalSendSequence,
};
use crate::protocol::blind_vault_replica_workflow::BlindVaultReplicaAttemptContinuation;
use crate::protocol::onion::OnionRoutePurpose;
use crate::protocol::onion_reply::{OnionReplyError, OnionReplyPayload, OnionReplySession};

/// Observable local state of one effect/session runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaTerminalAttemptState {
    /// The next committed effect may be sent.
    Ready,
    /// Every effect returned one authenticated and semantically valid reply.
    Complete,
    /// An ambiguous or invalid response permanently closed this runtime.
    Poisoned,
}

/// Workload-specific terminal reply verification boundary.
///
/// The onion layer authenticates route, request context, and terminal identity.
/// This trait must additionally decode the inner Blind Vault frame and verify
/// its exact request/result semantics before a compound attempt may continue.
pub trait BlindVaultReplicaTerminalReplyVerifier {
    type Output;
    type Error;

    /// Verifies one already opened source-only terminal reply.
    fn verify_terminal_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        purpose: OnionRoutePurpose,
        adapter_state: &[u8],
        reply: OnionReplyPayload,
    ) -> Result<Self::Output, Self::Error>;
}

/// Runtime owning aligned effect, reply-session, and adapter-state cursors.
pub struct BlindVaultReplicaTerminalAttemptRuntime<'effects> {
    send_sequence: BlindVaultReplicaTerminalSendSequence<'effects>,
    adapter_state: Vec<u8>,
    reply_sessions: VecDeque<OnionReplySession>,
    state: BlindVaultReplicaTerminalAttemptState,
}

impl<'effects> BlindVaultReplicaTerminalAttemptRuntime<'effects> {
    /// Composes already-durable send authority with its private continuation.
    ///
    /// [BLIND-VAULT-TERMINAL-ATTEMPT-RUNTIME 2026-08-29 by Codex] The
    /// constructor consumes both owners so no second component can advance an
    /// independent cursor after this runtime has been created.
    pub fn new(
        send_sequence: BlindVaultReplicaTerminalSendSequence<'effects>,
        continuation: BlindVaultReplicaAttemptContinuation,
    ) -> Result<Self, BlindVaultReplicaTerminalAttemptRuntimeBuildError> {
        let (mut adapter_state, reply_sessions) = continuation.into_parts();
        let remaining_effects = send_sequence.remaining_effects();
        if remaining_effects == 0 || remaining_effects != reply_sessions.len() {
            adapter_state.zeroize();
            return Err(
                BlindVaultReplicaTerminalAttemptRuntimeBuildError::SessionEffectCountMismatch {
                    effects: remaining_effects,
                    sessions: reply_sessions.len(),
                },
            );
        }
        Ok(Self {
            send_sequence,
            adapter_state,
            reply_sessions: reply_sessions.into(),
            state: BlindVaultReplicaTerminalAttemptState::Ready,
        })
    }

    /// Sends, opens, and semantically verifies exactly the next effect.
    ///
    /// Transport failure leaves the session and cursor retryable. Once the
    /// transport returns bytes, any cryptographic or semantic failure poisons
    /// the runtime because the remote side effect may already have occurred.
    pub fn send_next<Transport, Verifier>(
        &mut self,
        transport: &mut Transport,
        verifier: &mut Verifier,
        purpose: OnionRoutePurpose,
        payload: &[u8],
    ) -> Result<
        Verifier::Output,
        BlindVaultReplicaTerminalAttemptError<Transport::Error, Verifier::Error>,
    >
    where
        Transport: BlindVaultReplicaTerminalEffectTransport,
        Transport::Response: AsRef<[u8]>,
        Verifier: BlindVaultReplicaTerminalReplyVerifier,
    {
        if self.state != BlindVaultReplicaTerminalAttemptState::Ready {
            return Err(match self.state {
                BlindVaultReplicaTerminalAttemptState::Complete => {
                    BlindVaultReplicaTerminalAttemptError::SequenceComplete
                }
                BlindVaultReplicaTerminalAttemptState::Poisoned => {
                    BlindVaultReplicaTerminalAttemptError::RuntimePoisoned
                }
                BlindVaultReplicaTerminalAttemptState::Ready => unreachable!(),
            });
        }
        let Some(context) = self.send_sequence.next_context() else {
            self.poison();
            return Err(BlindVaultReplicaTerminalAttemptError::StateMismatch);
        };
        if !self.send_sequence.matches_next_payload(purpose, payload) {
            return Err(BlindVaultReplicaTerminalAttemptError::PayloadMismatch);
        }
        let Some(session) = self.reply_sessions.front() else {
            self.poison();
            return Err(BlindVaultReplicaTerminalAttemptError::StateMismatch);
        };
        // [BLIND-VAULT-SESSION-REQUEST-PREFLIGHT 2026-08-29 by Codex] A
        // cardinality-correct but reordered continuation must be rejected
        // before its terminal operation can create an ambiguous side effect.
        if !session.matches_encoded_request(payload) {
            self.poison();
            return Err(BlindVaultReplicaTerminalAttemptError::SessionRequestMismatch);
        }
        let response = match self.send_sequence.send_next(transport, purpose, payload) {
            Ok(response) => response,
            Err(BlindVaultReplicaTerminalSendError::Transport(error)) => {
                return Err(BlindVaultReplicaTerminalAttemptError::Transport(error));
            }
            Err(BlindVaultReplicaTerminalSendError::PayloadMismatch) => {
                return Err(BlindVaultReplicaTerminalAttemptError::PayloadMismatch);
            }
            Err(
                BlindVaultReplicaTerminalSendError::BindingInvalid
                | BlindVaultReplicaTerminalSendError::SequenceComplete,
            ) => {
                self.poison();
                return Err(BlindVaultReplicaTerminalAttemptError::StateMismatch);
            }
        };
        let Some(session) = self.reply_sessions.pop_front() else {
            self.poison();
            return Err(BlindVaultReplicaTerminalAttemptError::StateMismatch);
        };
        let reply = match session.open(response.as_ref()) {
            Ok(reply) => reply,
            Err(error) => {
                self.poison();
                return Err(BlindVaultReplicaTerminalAttemptError::Reply(error));
            }
        };
        let output =
            match verifier.verify_terminal_reply(context, purpose, &self.adapter_state, reply) {
                Ok(output) => output,
                Err(error) => {
                    self.poison();
                    return Err(BlindVaultReplicaTerminalAttemptError::Verification(error));
                }
            };
        let send_complete = self.send_sequence.is_complete();
        let sessions_complete = self.reply_sessions.is_empty();
        if send_complete != sessions_complete {
            self.poison();
            return Err(BlindVaultReplicaTerminalAttemptError::StateMismatch);
        }
        if send_complete {
            self.state = BlindVaultReplicaTerminalAttemptState::Complete;
        }
        Ok(output)
    }

    /// Current fail-closed runtime state.
    #[must_use]
    pub const fn state(&self) -> BlindVaultReplicaTerminalAttemptState {
        self.state
    }

    /// Count of effects still requiring a valid terminal reply.
    #[must_use]
    pub fn remaining_effects(&self) -> usize {
        self.send_sequence.remaining_effects()
    }

    fn poison(&mut self) {
        self.reply_sessions.clear();
        self.adapter_state.zeroize();
        self.state = BlindVaultReplicaTerminalAttemptState::Poisoned;
    }
}

impl Drop for BlindVaultReplicaTerminalAttemptRuntime<'_> {
    fn drop(&mut self) {
        self.adapter_state.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaTerminalAttemptRuntime<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaTerminalAttemptRuntime")
            .field("state", &self.state)
            .field("remaining_effects", &self.send_sequence.remaining_effects())
            .field("reply_sessions", &self.reply_sessions.len())
            .field("adapter_state", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

/// Construction failure before any runtime I/O is allowed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaTerminalAttemptRuntimeBuildError {
    /// Remaining committed effects and private one-time sessions differed.
    #[error("blind vault terminal runtime effect/session counts differ")]
    SessionEffectCountMismatch { effects: usize, sessions: usize },
}

/// Ordered send, authenticated reply, or workload verification failure.
#[derive(Debug)]
pub enum BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError> {
    /// Runtime had already authenticated every expected terminal response.
    SequenceComplete,
    /// Runtime was closed after an ambiguous or invalid terminal response.
    RuntimePoisoned,
    /// Caller bytes did not match the next durable prepared effect.
    PayloadMismatch,
    /// Internal cursor/session state no longer matched its durable binding.
    StateMismatch,
    /// The next reply session was not created for the committed request bytes.
    SessionRequestMismatch,
    /// Transport failed before returning a response; the same effect may retry.
    Transport(TransportError),
    /// The source-only onion response failed cryptographic authentication.
    Reply(OnionReplyError),
    /// The opened workload response failed exact semantic verification.
    Verification(VerificationError),
}

impl<TransportError: fmt::Display, VerificationError: fmt::Display> fmt::Display
    for BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceComplete => {
                formatter.write_str("blind vault terminal attempt is complete")
            }
            Self::RuntimePoisoned => {
                formatter.write_str("blind vault terminal attempt is poisoned")
            }
            Self::PayloadMismatch => {
                formatter.write_str("blind vault terminal attempt payload mismatched")
            }
            Self::StateMismatch => {
                formatter.write_str("blind vault terminal attempt state mismatched")
            }
            Self::SessionRequestMismatch => {
                formatter.write_str("blind vault terminal reply session mismatched its request")
            }
            Self::Transport(error) => {
                write!(formatter, "blind vault terminal transport failed: {error}")
            }
            Self::Reply(error) => fmt::Display::fmt(error, formatter),
            Self::Verification(error) => {
                write!(
                    formatter,
                    "blind vault terminal reply verification failed: {error}"
                )
            }
        }
    }
}

impl<TransportError, VerificationError> StdError
    for BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError>
where
    TransportError: StdError + 'static,
    VerificationError: StdError + 'static,
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Transport(error) => Some(error),
            Self::Reply(error) => Some(error),
            Self::Verification(error) => Some(error),
            Self::SequenceComplete
            | Self::RuntimePoisoned
            | Self::PayloadMismatch
            | Self::StateMismatch
            | Self::SessionRequestMismatch => None,
        }
    }
}
