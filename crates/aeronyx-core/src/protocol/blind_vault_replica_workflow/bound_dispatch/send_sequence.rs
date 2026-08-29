// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/send_sequence.rs
// ============================================
//! Ordered transport capability for one durable prepared effect set.
//!
//! ## Creation Reason
//! A final durability marker must verify the same payload at the trusted
//! transport boundary and prevent skipping or reordering compound effects.
//!
//! ## Main Functionality
//! - Defines a replaceable terminal transport trait.
//! - Verifies next purpose and payload immediately before transport call.
//! - Requires workflow-issued authorization before retiring an old lease.
//! - Advances only after transport reports success.
//! - Exposes bounded local context without route or private payload metadata.
//!
//! ## Dependencies
//! - Parent bound durable dispatch marker.
//! - `prepared_effect.rs`: exact indexed payload matching.
//! - `BlindVaultReplacementRetirementPermit`: old-lease safety gate.
//! - `OnionRoutePurpose`: canonical terminal workload purpose.
//!
//! ## Main Logical Flow
//! 1. A durable marker creates one sequence at index zero.
//! 2. `send_next` matches purpose and bytes against the current effect.
//! 3. Retirement additionally binds its permit to work, attempt, lease, node.
//! 4. The same borrowed payload is passed to the trusted transport trait.
//! 5. Success advances; failure leaves the same effect retryable.
//!
//! ## Important Note For The Next Developer
//! - Transport implementations are a trusted source adapter boundary.
//! - A remote timeout is ambiguous; idempotency still relies on request IDs.
//! - Never advance on transport error or allow caller-selected effect indexes.
//! - A transport must honor `authorized_terminal_node_id` when it is present.
//! - Route replacement belongs inside the transport adapter, not this binding.
//!
//! Last Modified: v1.3.0-RetirementPermitGate - Required an exact workflow
//! permit and terminal binding before old-lease retirement transport.
//! v1.2.0-OwnedEffectSource - Added an internal owned binding
//! path so complete runtimes do not require self-referential integration.
//! v1.1.0-TerminalAttemptRuntime - Shared canonical next-effect
//! context with the aligned reply-session runtime.
//! v1.0.0-OrderedTerminalSend - Initial verified transport boundary and
//! monotonic effect cursor.
//! ============================================

use std::{error::Error, fmt};

use super::super::{
    BlindVaultReplacementRetirementPermit, BlindVaultReplicaPreparedEffectSet,
    BlindVaultReplicaWorkId,
};
use crate::protocol::blind_vault::{decode_blind_vault_frame, BlindVaultFrame};
use crate::protocol::onion::OnionRoutePurpose;
use crate::protocol::onion_reply::decode_onion_reply_request;

/// Coarse source-local context supplied to a trusted transport adapter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaTerminalSendContext {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    effect_index: u16,
    effect_count: u16,
    authorized_terminal_node_id: Option<[u8; 32]>,
}

impl fmt::Debug for BlindVaultReplicaTerminalSendContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaTerminalSendContext")
            .field("attempt", &self.attempt)
            .field("effect_index", &self.effect_index)
            .field("effect_count", &self.effect_count)
            .field(
                "authorized_terminal",
                &self.authorized_terminal_node_id.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl BlindVaultReplicaTerminalSendContext {
    /// Source-local work identity for correlation outside telemetry.
    #[must_use]
    pub const fn work_id(self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Exact bounded workflow attempt.
    #[must_use]
    pub const fn attempt(self) -> u8 {
        self.attempt
    }

    /// Zero-based effect index selected by the sequence, never by the caller.
    #[must_use]
    pub const fn effect_index(self) -> u16 {
        self.effect_index
    }

    /// Total ordered effects committed for the attempt.
    #[must_use]
    pub const fn effect_count(self) -> u16 {
        self.effect_count
    }

    /// Exact terminal authorized by a workflow lifecycle permit, when any.
    ///
    /// Transport adapters must fail closed rather than selecting another
    /// terminal when this value is present. It must never enter telemetry.
    #[must_use]
    pub const fn authorized_terminal_node_id(self) -> Option<[u8; 32]> {
        self.authorized_terminal_node_id
    }

    fn with_authorized_terminal_node_id(mut self, node_id: Option<[u8; 32]>) -> Self {
        self.authorized_terminal_node_id = node_id;
        self
    }

    pub(super) fn without_terminal_authorization(mut self) -> Self {
        self.authorized_terminal_node_id = None;
        self
    }
}

/// Replaceable trusted boundary that performs one exact terminal send.
pub trait BlindVaultReplicaTerminalEffectTransport {
    type Response;
    type Error;

    /// Sends the same payload bytes verified by the ordered capability.
    ///
    /// Implementations must bind route selection to
    /// `context.authorized_terminal_node_id()` when it is present.
    fn send_terminal_effect(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        purpose: OnionRoutePurpose,
        payload: &[u8],
    ) -> Result<Self::Response, Self::Error>;
}

/// Single-use-in-order capability created only by a durable bound marker.
pub struct BlindVaultReplicaTerminalSendSequence<'effects> {
    effect_set: BlindVaultReplicaTerminalEffectSetSource<'effects>,
    next_index: usize,
    snapshot_sequence: u64,
    journal_sequence: u64,
}

enum BlindVaultReplicaTerminalEffectSetSource<'effects> {
    Borrowed(&'effects BlindVaultReplicaPreparedEffectSet),
    Owned(Box<BlindVaultReplicaPreparedEffectSet>),
}

impl BlindVaultReplicaTerminalEffectSetSource<'_> {
    fn as_effect_set(&self) -> &BlindVaultReplicaPreparedEffectSet {
        match self {
            Self::Borrowed(effect_set) => effect_set,
            Self::Owned(effect_set) => effect_set,
        }
    }
}

impl<'effects> BlindVaultReplicaTerminalSendSequence<'effects> {
    pub(in crate::protocol::blind_vault_replica_workflow) fn from_durable_parts(
        effect_set: &'effects BlindVaultReplicaPreparedEffectSet,
        snapshot_sequence: u64,
        journal_sequence: u64,
    ) -> Self {
        Self {
            effect_set: BlindVaultReplicaTerminalEffectSetSource::Borrowed(effect_set),
            next_index: 0,
            snapshot_sequence,
            journal_sequence,
        }
    }

    pub(super) fn from_owned_durable_parts(
        effect_set: BlindVaultReplicaPreparedEffectSet,
        snapshot_sequence: u64,
        journal_sequence: u64,
    ) -> BlindVaultReplicaTerminalSendSequence<'static> {
        BlindVaultReplicaTerminalSendSequence {
            effect_set: BlindVaultReplicaTerminalEffectSetSource::Owned(Box::new(effect_set)),
            next_index: 0,
            snapshot_sequence,
            journal_sequence,
        }
    }

    fn effect_set(&self) -> &BlindVaultReplicaPreparedEffectSet {
        self.effect_set.as_effect_set()
    }

    pub(super) fn next_context(&self) -> Option<BlindVaultReplicaTerminalSendContext> {
        let effect_index = u16::try_from(self.next_index).ok()?;
        let effect_count = u16::try_from(self.effect_set().effects().len()).ok()?;
        (self.next_index < self.effect_set().effects().len()).then_some(
            BlindVaultReplicaTerminalSendContext {
                work_id: self.effect_set().work_id(),
                attempt: self.effect_set().attempt(),
                effect_index,
                effect_count,
                authorized_terminal_node_id: None,
            },
        )
    }

    pub(super) fn matches_next_payload(&self, purpose: OnionRoutePurpose, payload: &[u8]) -> bool {
        self.effect_set()
            .matches_payload(self.next_index, purpose, payload)
    }

    /// Sends next exact effect and advances only after transport success.
    ///
    /// [BLIND-VAULT-ORDERED-TERMINAL-SEND 2026-08-29 by Codex] Validation and
    /// invocation share one borrowed payload, closing the gap between an
    /// earlier commitment check and bytes presented to the adapter.
    pub fn send_next<Transport>(
        &mut self,
        transport: &mut Transport,
        purpose: OnionRoutePurpose,
        payload: &[u8],
    ) -> Result<Transport::Response, BlindVaultReplicaTerminalSendError<Transport::Error>>
    where
        Transport: BlindVaultReplicaTerminalEffectTransport,
    {
        self.send_next_authorized_with_context(transport, purpose, payload, None)
            .map(|(_, response)| response)
    }

    /// Sends the next retirement effect under an exact workflow-issued gate.
    ///
    /// Calling this method for a non-retirement purpose fails closed. The
    /// permit is checked before transport against the canonical sequence
    /// context and the inner signed retirement request.
    ///
    /// [BLIND-VAULT-SEQUENCE-RETIREMENT-PERMIT 2026-08-29 by Codex]
    pub fn send_next_with_retirement_permit<Transport>(
        &mut self,
        transport: &mut Transport,
        purpose: OnionRoutePurpose,
        payload: &[u8],
        permit: &BlindVaultReplacementRetirementPermit,
    ) -> Result<Transport::Response, BlindVaultReplicaTerminalSendError<Transport::Error>>
    where
        Transport: BlindVaultReplicaTerminalEffectTransport,
    {
        self.send_next_authorized_with_context(transport, purpose, payload, Some(permit))
            .map(|(_, response)| response)
    }

    pub(super) fn send_next_authorized_with_context<Transport>(
        &mut self,
        transport: &mut Transport,
        purpose: OnionRoutePurpose,
        payload: &[u8],
        retirement_permit: Option<&BlindVaultReplacementRetirementPermit>,
    ) -> Result<
        (BlindVaultReplicaTerminalSendContext, Transport::Response),
        BlindVaultReplicaTerminalSendError<Transport::Error>,
    >
    where
        Transport: BlindVaultReplicaTerminalEffectTransport,
    {
        if self.next_index >= self.effect_set().effects().len() {
            return Err(BlindVaultReplicaTerminalSendError::SequenceComplete);
        }
        if !self.matches_next_payload(purpose, payload) {
            return Err(BlindVaultReplicaTerminalSendError::PayloadMismatch);
        }
        // [BLIND-VAULT-CANONICAL-SEND-CONTEXT 2026-08-29 by Codex] Runtime
        // reply verification and transport receive the same sequence-owned
        // index; neither adapter can select or advance it independently.
        let context = self
            .next_context()
            .ok_or(BlindVaultReplicaTerminalSendError::BindingInvalid)?;
        let authorized_terminal_node_id =
            retirement_authorized_terminal(context, purpose, payload, retirement_permit)?;
        let context = context.with_authorized_terminal_node_id(authorized_terminal_node_id);
        let response = transport
            .send_terminal_effect(context, purpose, payload)
            .map_err(BlindVaultReplicaTerminalSendError::Transport)?;
        self.next_index += 1;
        Ok((context, response))
    }

    /// Whether every effect has returned transport success.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.next_index == self.effect_set().effects().len()
    }

    /// Count of effects still awaiting transport success.
    #[must_use]
    pub fn remaining_effects(&self) -> usize {
        self.effect_set()
            .effects()
            .len()
            .saturating_sub(self.next_index)
    }

    /// Durable workflow snapshot sequence that authorized network send.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// Durable private-journal sequence that authorized network send.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }
}

impl fmt::Debug for BlindVaultReplicaTerminalSendSequence<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaTerminalSendSequence")
            .field("attempt", &self.effect_set().attempt())
            .field("next_index", &self.next_index)
            .field("effect_count", &self.effect_set().effects().len())
            .field("snapshot_sequence", &self.snapshot_sequence)
            .field("journal_sequence", &self.journal_sequence)
            .finish_non_exhaustive()
    }
}

/// Payload-binding or transport failure for one ordered send.
#[derive(Debug, PartialEq, Eq)]
pub enum BlindVaultReplicaTerminalSendError<TransportError> {
    /// Durable effect count or cursor could not fit the bounded context.
    BindingInvalid,
    /// Caller attempted another send after every committed effect completed.
    SequenceComplete,
    /// Purpose, length, or bytes did not match the next committed effect.
    PayloadMismatch,
    /// Old-lease retirement was attempted without a workflow-issued permit.
    RetirementPermitRequired,
    /// Permit, request, work, attempt, lease, or purpose did not match exactly.
    RetirementPermitMismatch,
    /// Committed retirement bytes were not one valid reply-capable request.
    RetirementRequestInvalid,
    /// Transport failed; the sequence remains at the same effect.
    Transport(TransportError),
}

impl<TransportError: fmt::Display> fmt::Display
    for BlindVaultReplicaTerminalSendError<TransportError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BindingInvalid => {
                formatter.write_str("blind vault terminal send binding is invalid")
            }
            Self::SequenceComplete => {
                formatter.write_str("blind vault terminal send sequence is complete")
            }
            Self::PayloadMismatch => {
                formatter.write_str("blind vault terminal send payload mismatched")
            }
            Self::RetirementPermitRequired => {
                formatter.write_str("blind vault replacement retirement permit is required")
            }
            Self::RetirementPermitMismatch => {
                formatter.write_str("blind vault replacement retirement permit mismatched")
            }
            Self::RetirementRequestInvalid => {
                formatter.write_str("blind vault replacement retirement request is invalid")
            }
            Self::Transport(error) => {
                write!(formatter, "blind vault terminal transport failed: {error}")
            }
        }
    }
}

impl<TransportError> Error for BlindVaultReplicaTerminalSendError<TransportError>
where
    TransportError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Transport(error) => Some(error),
            Self::BindingInvalid
            | Self::SequenceComplete
            | Self::PayloadMismatch
            | Self::RetirementPermitRequired
            | Self::RetirementPermitMismatch
            | Self::RetirementRequestInvalid => None,
        }
    }
}

fn retirement_authorized_terminal<TransportError>(
    context: BlindVaultReplicaTerminalSendContext,
    purpose: OnionRoutePurpose,
    encoded_request: &[u8],
    permit: Option<&BlindVaultReplacementRetirementPermit>,
) -> Result<Option<[u8; 32]>, BlindVaultReplicaTerminalSendError<TransportError>> {
    if purpose != OnionRoutePurpose::BlindVaultLeaseRetire {
        return if permit.is_some() {
            Err(BlindVaultReplicaTerminalSendError::RetirementPermitMismatch)
        } else {
            Ok(None)
        };
    }
    let permit = permit.ok_or(BlindVaultReplicaTerminalSendError::RetirementPermitRequired)?;
    if permit.work_id() != context.work_id() || permit.attempt() != context.attempt() {
        return Err(BlindVaultReplicaTerminalSendError::RetirementPermitMismatch);
    }
    // [BLIND-VAULT-RETIREMENT-PERMIT-PREFLIGHT 2026-08-29 by Codex] Decode
    // only at the source boundary and never retain or expose request bytes.
    let onion_request = decode_onion_reply_request(encoded_request)
        .map_err(|_| BlindVaultReplicaTerminalSendError::RetirementRequestInvalid)?;
    let frame = decode_blind_vault_frame(&onion_request.payload)
        .map_err(|_| BlindVaultReplicaTerminalSendError::RetirementRequestInvalid)?;
    let BlindVaultFrame::LeaseRetire(request) = frame else {
        return Err(BlindVaultReplicaTerminalSendError::RetirementRequestInvalid);
    };
    if request.lease_id != permit.replaced_lease_id() {
        return Err(BlindVaultReplicaTerminalSendError::RetirementPermitMismatch);
    }
    Ok(Some(permit.replaced_node_id()))
}
