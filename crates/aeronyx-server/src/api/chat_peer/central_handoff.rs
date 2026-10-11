// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer/central_handoff.rs
// ============================================================================
//! # Onion terminal hand-off to the receiver's relay inbox
//!
//! [ONION-CENTRAL-HANDOFF 2026-10-11 by Claude] A `MessageRelay` onion
//! terminal used to keep the delivered envelope in its own pending store and
//! push it only to the receiver's sessions on this node. Terminals are picked
//! at random, so a receiver connected anywhere else - in practice the central
//! relay every App keeps open - never received the message, while the sender
//! held a valid receipt (iOS simulator test, 2026-10-11).
//!
//! With a hand-off URL configured, the terminal instead gives the exact signed
//! envelope to the central relay, which verifies the sender's own signature
//! and queues it for the receiver like any relay send. Only an accepted
//! hand-off lets the terminal sign its receipt, so a receipt now means "in the
//! receiver's inbox", not "held by a node the receiver may never reach".
//!
//! What each party sees is unchanged: this terminal already held the envelope
//! (sender, receiver, ciphertext); the central relay sees what a normal send
//! shows it, except the sender's network address, which only the entry hop
//! ever saw.
//!
//! The node authenticates with the same `X-Node-*` signature as its heartbeat.
//! That only identifies the forwarding node; the envelope's Ed25519 signature
//! is what authorises the message, so a node cannot forge or alter one.
// ============================================================================

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::{encode_envelope, ChatEnvelope};
use base64::Engine;
use tokio::sync::Semaphore;

use crate::config_chat_relay::validate_central_handoff_url;
use crate::management::client::sign_node_request;

/// One hand-off, connection setup included. Shorter than the 10 s client
/// deadline on the onion entry POST, so a slow relay surfaces as a failure
/// the sender can act on instead of a client timeout.
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(7);
const HANDOFF_CONNECT_TIMEOUT: Duration = Duration::from_secs(4);

/// Concurrent hand-offs per node. Beyond this the terminal reports
/// unavailable rather than queueing outbound requests without bound.
const MAX_IN_FLIGHT: usize = 32;

/// Why a hand-off did not end in acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CentralHandoffError {
    /// The relay refused this envelope (malformed, bad signature, expired,
    /// blocked by the receiver). Sending it again will not help.
    Rejected,
    /// The relay could not be reached or could not take it now.
    Unavailable,
}

/// Hands accepted terminal envelopes to the central relay inbox.
pub(super) struct CentralHandoff {
    url: reqwest::Url,
    client: reqwest::Client,
    identity: Arc<IdentityKeyPair>,
    node_id_hex: String,
    admission: Arc<Semaphore>,
}

impl CentralHandoff {
    /// A hand-off client for `url` (see [`validate_central_handoff_url`]):
    /// `https` verified against public CAs, plain `http` only to loopback.
    ///
    /// # Errors
    /// Returns a description when the URL is unusable or the HTTP client
    /// cannot be built.
    pub(super) fn new(url: &str, identity: Arc<IdentityKeyPair>) -> Result<Self, String> {
        validate_central_handoff_url(url)?;
        let url = reqwest::Url::parse(url).map_err(|error| format!("invalid url: {error}"))?;
        let client = reqwest::Client::builder()
            .timeout(HANDOFF_TIMEOUT)
            .connect_timeout(HANDOFF_CONNECT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .user_agent(concat!("aeronyx-node/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| format!("http client: {error}"))?;
        let node_id_hex = hex::encode(identity.public_key_bytes());
        Ok(Self {
            url,
            client,
            identity,
            node_id_hex,
            admission: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        })
    }

    /// The exact request body: the canonical envelope bytes, base64.
    fn request_body(envelope: &ChatEnvelope) -> Result<String, CentralHandoffError> {
        let bytes = encode_envelope(envelope).map_err(|_| CentralHandoffError::Rejected)?;
        let envelope_b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        serde_json::to_string(&serde_json::json!({ "envelope_b64": envelope_b64 }))
            .map_err(|_| CentralHandoffError::Rejected)
    }

    /// Gives one envelope to the relay; `Ok` only once it has been accepted
    /// into the receiver's inbox (or was already accepted earlier).
    ///
    /// # Errors
    /// [`CentralHandoffError::Rejected`] for a refusal that a retry cannot
    /// change, [`CentralHandoffError::Unavailable`] otherwise.
    pub(super) async fn deliver(&self, envelope: &ChatEnvelope) -> Result<(), CentralHandoffError> {
        let _permit = self
            .admission
            .try_acquire()
            .map_err(|_| CentralHandoffError::Unavailable)?;
        let body = Self::request_body(envelope)?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CentralHandoffError::Unavailable)?
            .as_secs();
        let signature = sign_node_request(&self.identity, &self.node_id_hex, timestamp, &body);
        let response = self
            .client
            .post(self.url.clone())
            .header("Content-Type", "application/json")
            .header("X-Node-ID", &self.node_id_hex)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", signature)
            .body(body)
            .send()
            .await
            .map_err(|_| CentralHandoffError::Unavailable)?;
        classify_response(
            response.status().as_u16(),
            &response.bytes().await.unwrap_or_default(),
        )
    }
}

/// The relay's verdict. Acceptance must be explicit: a 2xx without
/// `"accepted": true` is not a delivery, whatever proxy produced it.
fn classify_response(status: u16, body: &[u8]) -> Result<(), CentralHandoffError> {
    match status {
        200 | 202 => {
            let accepted = serde_json::from_slice::<serde_json::Value>(body)
                .ok()
                .and_then(|value| value.get("accepted").and_then(serde_json::Value::as_bool))
                == Some(true);
            if accepted {
                Ok(())
            } else {
                Err(CentralHandoffError::Unavailable)
            }
        }
        400 | 403 | 413 | 422 => Err(CentralHandoffError::Rejected),
        _ => Err(CentralHandoffError::Unavailable),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_explicit_acceptance_is_a_delivery() {
        assert_eq!(classify_response(202, br#"{"accepted":true}"#), Ok(()));
        assert_eq!(
            classify_response(200, br#"{"accepted":true,"idempotent_replay":true}"#),
            Ok(())
        );
        for (status, body) in [
            (202, &br#"{"accepted":false}"#[..]),
            (202, &b"<html>cdn</html>"[..]),
            (204, &b""[..]),
            (301, &b""[..]),
            (401, &br#"{"error":"Invalid signature"}"#[..]),
            (409, &br#"{"error":"request_in_progress"}"#[..]),
            (429, &b""[..]),
            (503, &b""[..]),
        ] {
            assert_eq!(
                classify_response(status, body),
                Err(CentralHandoffError::Unavailable),
                "{status}"
            );
        }
        for status in [400, 403, 413, 422] {
            assert_eq!(
                classify_response(status, br#"{"error":"invalid_signature"}"#),
                Err(CentralHandoffError::Rejected),
                "{status}"
            );
        }
    }

    #[test]
    fn a_network_url_must_be_https_without_credentials() {
        let identity = Arc::new(IdentityKeyPair::generate());
        assert!(CentralHandoff::new(
            crate::config_chat_relay::DEFAULT_CENTRAL_HANDOFF_URL,
            Arc::clone(&identity)
        )
        .is_ok());
        assert!(CentralHandoff::new("http://127.0.0.1:9/x", Arc::clone(&identity)).is_ok());
        for refused in [
            "http://api.aeronyx.network/api/relay/node/v1/onion-handoff",
            "ftp://api.aeronyx.network/x",
            "https://user:pw@api.aeronyx.network/x",
            "not a url",
        ] {
            assert!(
                CentralHandoff::new(refused, Arc::clone(&identity)).is_err(),
                "{refused}"
            );
        }
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use aeronyx_core::protocol::chat::ChatContentType;

    /// [ONION-CENTRAL-HANDOFF 2026-10-11 by Claude] The exact request body
    /// for a fixed envelope. The relay's Python parser test
    /// (relay/test_onion_handoff.py) decodes this same string, so the two
    /// encoders cannot drift apart unnoticed.
    pub(super) const GOLDEN_BODY: &str = "{\"envelope_b64\":\"EREREREREREREREREREREepKbGPinFIKvvVQexMuxfmVR3auvr57kkIe6mkURtIsIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIA0klrAAAAAAYAAAAAAAAAb3BhcXVlMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzAAAAAOJCtyF8OtAsOqQJyigxvcRPtAxSTO5XCBiojWx+sLbgxRADOweCRa8pfakXYGF1k06axnZ2VckTqXMQgLMDxwU=\"}";

    pub(super) fn golden_envelope() -> ChatEnvelope {
        let sender = IdentityKeyPair::from_bytes(&[7u8; 32]).unwrap();
        let mut envelope = ChatEnvelope {
            message_id: [0x11; 16],
            sender: sender.public_key_bytes(),
            receiver: [0x22; 32],
            timestamp: 1_800_000_000,
            ciphertext: b"opaque".to_vec(),
            nonce: [0x33; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        envelope
    }

    #[test]
    fn the_request_body_matches_the_relay_golden_vector() {
        let body = CentralHandoff::request_body(&golden_envelope()).unwrap();
        assert_eq!(body, GOLDEN_BODY);
    }
}
