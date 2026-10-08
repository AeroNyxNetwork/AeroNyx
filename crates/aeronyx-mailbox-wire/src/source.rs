// ============================================
// File: crates/aeronyx-mailbox-wire/src/source.rs
// ============================================
//! A client acting as its own onion source for one mailbox exchange.
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] The same steps as the node's reference
//! client (`aeronyx-server mailbox-probe`, `Probe::exchange`):
//! 1. source carrier with a one-shot reply key over the terminal frame;
//! 2. route request signed by a fresh per-exchange source key (never an
//!    identity key);
//! 3. MemChain framing: `0xAE ‖ u32 LE variant 40 ‖ bincode(route request)`;
//! 4. one- or two-hop onion envelope signed by the same source key;
//! 5. the JSON body for `POST {entry}/api/chat/peer/blind-relay`.
//!
//! The returned [`SourceExchange`] owns the reply session; dropping it
//! discards the reply key. Opening consumes it, so a reply opens once.

use serde::Deserialize;

use crate::chat::PeerBlindRelayRequest;
use crate::codec::encode_bincode_bounded;
use crate::crypto::IdentityKeyPair;
use crate::mailbox::{
    decode_anonymous_mailbox_terminal_frame, AnonymousMailboxRouteRequestV1,
    AnonymousMailboxSourceSealSessionV1, AnonymousMailboxSourceTerminalCarrierV1,
    AnonymousMailboxTerminalFrameV1,
};
use crate::onion::{build_source_envelope, OnionHop};

/// MemChain frame magic.
pub const MEMCHAIN_MAGIC: u8 = 0xAE;
/// Frozen `MemChainMessage::AnonymousMailboxRouteV1` variant index.
pub const MEMCHAIN_ANONYMOUS_MAILBOX_ROUTE_V1: u32 = 40;
/// Path of the entry node's blind relay route.
pub const BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";
const MAX_MEMCHAIN_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024;
const MAX_RESPONSE_JSON_BYTES: usize = 512 * 1024;

/// Source-side failures. No keys, ids or payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SourceError {
    /// Path must be 1..=3 hops ending at the target.
    #[error("invalid source path")]
    Path,
    /// Frame or carrier construction failed.
    #[error("source request construction failed")]
    Build,
    /// The relay rejected or returned no sealed reply.
    #[error("relay did not return a sealed terminal reply")]
    NoReply,
    /// The reply could not be opened or decoded for this exchange.
    #[error("sealed terminal reply rejected")]
    Reply,
}

/// One prepared exchange: the HTTP body plus the one-shot reply session.
pub struct SourceExchange {
    route_id: [u8; 16],
    body_json: Vec<u8>,
    session: AnonymousMailboxSourceSealSessionV1,
}

impl SourceExchange {
    /// Builds the request for `terminal_frame` (already-encoded terminal frame
    /// bytes) to the last hop of `path`.
    ///
    /// # Errors
    /// Fails for a bad path or frame.
    pub fn prepare(path: &[OnionHop], terminal_frame: Vec<u8>, now: u64) -> Result<Self, SourceError> {
        let target = path.last().ok_or(SourceError::Path)?.node_id;
        let mut route_id = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut route_id);
        let source = IdentityKeyPair::generate();
        let (carrier, session) =
            AnonymousMailboxSourceTerminalCarrierV1::prepare(route_id, target, terminal_frame)
                .map_err(|_| SourceError::Build)?;
        let route = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            target,
            carrier.encode().map_err(|_| SourceError::Build)?,
            now,
            &source,
        )
        .map_err(|_| SourceError::Build)?;
        let payload = encode_route_payload(&route)?;
        let envelope = build_source_envelope(path, &payload, route_id, now, &source)
            .map_err(|_| SourceError::Path)?;
        let body_json = serde_json_body(&PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
        })?;
        Ok(Self {
            route_id,
            body_json,
            session,
        })
    }

    /// The exact JSON body to POST to the entry node.
    #[must_use]
    pub fn body_json(&self) -> &[u8] {
        &self.body_json
    }

    /// Route id (diagnostics only; never log with identities).
    #[must_use]
    pub fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    /// Opens the relay's JSON response and returns the decoded terminal
    /// frame. Consumes the exchange: a reply opens at most once.
    ///
    /// The caller must still verify the frame against its request
    /// (`verify_for_request`) before trusting the outcome.
    ///
    /// # Errors
    /// Fails if the relay rejected the request or the reply is not the one
    /// sealed to this exchange.
    pub fn open(mut self, response_json: &[u8]) -> Result<AnonymousMailboxTerminalFrameV1, SourceError> {
        if response_json.len() > MAX_RESPONSE_JSON_BYTES {
            return Err(SourceError::Reply);
        }
        let response: RelayResponse =
            serde_json::from_slice(response_json).map_err(|_| SourceError::NoReply)?;
        if !response.accepted {
            return Err(SourceError::NoReply);
        }
        let sealed_b64 = response.opaque_terminal_response_b64.ok_or(SourceError::NoReply)?;
        let sealed = base64_decode(&sealed_b64).ok_or(SourceError::Reply)?;
        let opened = self.session.open(&sealed).map_err(|_| SourceError::Reply)?;
        decode_anonymous_mailbox_terminal_frame(&opened).map_err(|_| SourceError::Reply)
    }
}

impl std::fmt::Debug for SourceExchange {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourceExchange")
            .field("reply_session", &"<redacted>")
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct RelayResponse {
    accepted: bool,
    #[serde(default)]
    opaque_terminal_response_b64: Option<String>,
}

/// `0xAE ‖ bincode(MemChainMessage::AnonymousMailboxRouteV1(route))`, written
/// without the node's whole MemChain enum: a fixint enum is a u32 LE variant
/// index followed by the variant's fields.
///
/// # Errors
/// Fails if the encoding exceeds the MemChain ceiling.
pub fn encode_route_payload(route: &AnonymousMailboxRouteRequestV1) -> Result<Vec<u8>, SourceError> {
    let body = encode_bincode_bounded(route, MAX_MEMCHAIN_PAYLOAD_BYTES).map_err(|_| SourceError::Build)?;
    let mut out = Vec::with_capacity(1 + 4 + body.len());
    out.push(MEMCHAIN_MAGIC);
    out.extend_from_slice(&MEMCHAIN_ANONYMOUS_MAILBOX_ROUTE_V1.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

fn serde_json_body(request: &PeerBlindRelayRequest) -> Result<Vec<u8>, SourceError> {
    serde_json::to_vec(request).map_err(|_| SourceError::Build)
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(text).ok()
}
