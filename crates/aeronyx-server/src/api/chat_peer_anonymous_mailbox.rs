// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_anonymous_mailbox.rs
// ============================================
//! Anonymous mailbox terminal dispatch for the authenticated blind-relay path.
//!
//! The node sees only signed opaque capabilities and sealed bytes.  This module
//! deliberately does not expose participant identity, contact, route, wallet,
//! content-key, or plaintext fields.
//!
//! ## Last Modified
//! v1.0.0-AnonymousMailboxTerminal — M13C bounded terminal wiring.

use std::sync::Arc;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxOperationV1, AnonymousMailboxOutcomeV1, AnonymousMailboxPullResultV1,
    AnonymousMailboxRouteRequestV1, AnonymousMailboxSourceSealedResponseV1,
    AnonymousMailboxSourceTerminalCarrierV1, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTerminalResponseV1,
};
use aeronyx_core::protocol::memchain::{decode_memchain, MemChainMessage, MEMCHAIN_MAGIC};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

use crate::services::chat_relay_mailbox::{
    AnonymousMailboxAckOutcome, AnonymousMailboxCreateOutcome, AnonymousMailboxCustodyRepository,
    AnonymousMailboxPullOutcome, AnonymousMailboxPutOutcome, AnonymousMailboxStoreError,
};

/// [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-02 by Codex] A prepared request owns
/// the decoded source carrier and exact route request.  It has no Debug output
/// because it contains opaque capability and sealed-byte material.
pub(super) struct PreparedAnonymousMailboxTerminal {
    route_request: AnonymousMailboxRouteRequestV1,
    carrier: AnonymousMailboxSourceTerminalCarrierV1,
    request: AnonymousMailboxTerminalFrameV1,
}

/// Coarse terminal execution failure; it intentionally has no source detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AnonymousMailboxTerminalFailure {
    Rejected,
    Unavailable,
}

impl PreparedAnonymousMailboxTerminal {
    /// Decodes the sole canonical mailbox terminal carrier after onion peel.
    /// No route signature is verified against the previous hop: each inner
    /// ticket/deposit/read signature remains the mutation authority.
    pub(super) fn decode(
        payload: &[u8],
        actual_route_id: [u8; 16],
        local_target_node_id: [u8; 32],
    ) -> Result<Self, AnonymousMailboxTerminalFailure> {
        if payload.first().copied() != Some(MEMCHAIN_MAGIC) {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        let MemChainMessage::AnonymousMailboxRouteV1(route_request) =
            decode_memchain(&payload[1..])
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?
        else {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        };
        if route_request.request_id != actual_route_id
            || route_request.target_node_id != local_target_node_id
            || route_request.signing_bytes().is_err()
        {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        let carrier = AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
            &route_request.sealed_terminal_frame,
            actual_route_id,
            local_target_node_id,
        )
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        let request = decode_anonymous_mailbox_terminal_frame(carrier.terminal_frame())
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        if !matches!(
            request,
            AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
                | AnonymousMailboxTerminalFrameV1::Put(_)
                | AnonymousMailboxTerminalFrameV1::PullOne(_)
                | AnonymousMailboxTerminalFrameV1::Ack(_)
        ) {
            return Err(AnonymousMailboxTerminalFailure::Rejected);
        }
        Ok(Self {
            route_request,
            carrier,
            request,
        })
    }

    pub(super) fn execute(
        self,
        repository: Arc<dyn AnonymousMailboxCustodyRepository>,
        terminal_identity: Arc<IdentityKeyPair>,
        now: u64,
    ) -> Result<String, AnonymousMailboxTerminalFailure> {
        let (operation, request_id, request_commitment, outcome, payload) = match &self.request {
            AnonymousMailboxTerminalFrameV1::LeaseCreate(request) => {
                request
                    .verify_for_target(&terminal_identity.public_key_bytes(), now)
                    .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
                let outcome = map_create(repository.create(request, now))?;
                (
                    AnonymousMailboxOperationV1::LeaseCreate,
                    request.admission.ticket_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            AnonymousMailboxTerminalFrameV1::Put(request) => {
                let outcome = map_put(repository.put(request, now))?;
                (
                    AnonymousMailboxOperationV1::Put,
                    request.item_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            AnonymousMailboxTerminalFrameV1::PullOne(request) => {
                let (outcome, payload) = map_pull(repository.pull_one(request, now))?;
                (
                    AnonymousMailboxOperationV1::PullOne,
                    request.request_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    payload,
                )
            }
            AnonymousMailboxTerminalFrameV1::Ack(request) => {
                let outcome = map_ack(repository.ack(request, now))?;
                (
                    AnonymousMailboxOperationV1::Ack,
                    request.request_id,
                    request
                        .request_commitment()
                        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?,
                    outcome,
                    Vec::new(),
                )
            }
            _ => return Err(AnonymousMailboxTerminalFailure::Rejected),
        };
        let response = AnonymousMailboxTerminalResponseV1::signed(
            operation,
            request_id,
            request_commitment,
            outcome,
            payload,
            now,
            &terminal_identity,
        )
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        let response_frame = match operation {
            AnonymousMailboxOperationV1::LeaseCreate => {
                AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response)
            }
            AnonymousMailboxOperationV1::Put => {
                AnonymousMailboxTerminalFrameV1::PutResponse(response)
            }
            AnonymousMailboxOperationV1::PullOne => {
                AnonymousMailboxTerminalFrameV1::PullOneResponse(response)
            }
            AnonymousMailboxOperationV1::Ack => {
                AnonymousMailboxTerminalFrameV1::AckResponse(response)
            }
        };
        let response_frame = encode_anonymous_mailbox_terminal_frame(&response_frame)
            .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            self.route_request.request_id,
            self.carrier.context_commitment(),
            terminal_identity.public_key_bytes(),
            self.carrier.reply_public_key(),
            &response_frame,
            &terminal_identity,
        )
        .and_then(|value| value.encode())
        .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
        // [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-03 by Codex] The onion ACK
        // carries only the compact source-sealed response. The signed route
        // response remains a compatible core type, but wrapping AMSR in it
        // would force middle hops to classify a terminal workload.
        Ok(BASE64.encode(sealed))
    }
}

fn map_store_error(error: AnonymousMailboxStoreError) -> AnonymousMailboxTerminalFailure {
    match error {
        AnonymousMailboxStoreError::Rejected => AnonymousMailboxTerminalFailure::Rejected,
        AnonymousMailboxStoreError::Disabled
        | AnonymousMailboxStoreError::Busy
        | AnonymousMailboxStoreError::UnsupportedSchema
        | AnonymousMailboxStoreError::Corrupt
        | AnonymousMailboxStoreError::Unavailable => AnonymousMailboxTerminalFailure::Unavailable,
    }
}

fn map_create(
    value: Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxCreateOutcome::Created(_) | AnonymousMailboxCreateOutcome::Existing(_) => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxCreateOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxCreateOutcome::AtCapacity => Ok(AnonymousMailboxOutcomeV1::AtCapacity),
    }
}

fn map_put(
    value: Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxPutOutcome::Stored(_) | AnonymousMailboxPutOutcome::Existing(_) => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxPutOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxPutOutcome::AtCapacity => Ok(AnonymousMailboxOutcomeV1::AtCapacity),
        AnonymousMailboxPutOutcome::LeaseExpired => Ok(AnonymousMailboxOutcomeV1::Expired),
        AnonymousMailboxPutOutcome::LeaseNotFound => Ok(AnonymousMailboxOutcomeV1::Rejected),
    }
}

fn map_pull(
    value: Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError>,
) -> Result<(AnonymousMailboxOutcomeV1, Vec<u8>), AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxPullOutcome::Item(item) => {
            let sealed_len = usize::try_from(item.sealed_length)
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            let sealed_item = item
                .padded_sealed_envelope
                .get(..sealed_len)
                .ok_or(AnonymousMailboxTerminalFailure::Rejected)?
                .to_vec();
            let result = AnonymousMailboxPullResultV1::new(item.item_id, item.cursor, sealed_item)
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            if result.sealed_commitment != item.sealed_commitment {
                return Err(AnonymousMailboxTerminalFailure::Rejected);
            }
            let result = result
                .encode()
                .map_err(|_| AnonymousMailboxTerminalFailure::Rejected)?;
            Ok((AnonymousMailboxOutcomeV1::Accepted, result))
        }
        AnonymousMailboxPullOutcome::Empty => Ok((AnonymousMailboxOutcomeV1::Accepted, Vec::new())),
        AnonymousMailboxPullOutcome::LeaseExpired => {
            Ok((AnonymousMailboxOutcomeV1::Expired, Vec::new()))
        }
        AnonymousMailboxPullOutcome::LeaseNotFound => {
            Ok((AnonymousMailboxOutcomeV1::Rejected, Vec::new()))
        }
    }
}

fn map_ack(
    value: Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError>,
) -> Result<AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFailure> {
    match value.map_err(map_store_error)? {
        AnonymousMailboxAckOutcome::Acknowledged
        | AnonymousMailboxAckOutcome::AlreadyAcknowledged => {
            Ok(AnonymousMailboxOutcomeV1::Accepted)
        }
        AnonymousMailboxAckOutcome::Conflict => Ok(AnonymousMailboxOutcomeV1::Conflict),
        AnonymousMailboxAckOutcome::NotFound => Ok(AnonymousMailboxOutcomeV1::Rejected),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxPullOneV1,
    };
    use aeronyx_core::protocol::memchain::encode_memchain;

    fn routed_pull(route_id: [u8; 16], target: &IdentityKeyPair) -> Vec<u8> {
        let source = IdentityKeyPair::from_bytes(&[0x61; 32]).expect("source");
        let reader = IdentityKeyPair::from_bytes(&[0x62; 32]).expect("reader");
        let pull = AnonymousMailboxPullOneV1::new(
            [0x63; 32],
            [0x64; 16],
            Vec::new(),
            1_800_000_000,
            &reader,
        )
        .expect("pull");
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(pull),
        )
        .expect("terminal frame");
        let (carrier, _session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("carrier");
        let route = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            target.public_key_bytes(),
            carrier.encode().expect("carrier bytes"),
            1_800_000_000,
            &source,
        )
        .expect("route");
        encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(route)).expect("outer")
    }

    #[test]
    fn canonical_carrier_rejects_route_or_target_substitution_before_repository_access() {
        let target = IdentityKeyPair::from_bytes(&[0x65; 32]).expect("target");
        let route_id = [0x66; 16];
        let encoded = routed_pull(route_id, &target);
        assert!(PreparedAnonymousMailboxTerminal::decode(
            &encoded,
            route_id,
            target.public_key_bytes(),
        )
        .is_ok());
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(
                &encoded,
                [0x67; 16],
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
        assert!(matches!(
            PreparedAnonymousMailboxTerminal::decode(&encoded, route_id, [0x68; 32]),
            Err(AnonymousMailboxTerminalFailure::Rejected)
        ));
    }
}
