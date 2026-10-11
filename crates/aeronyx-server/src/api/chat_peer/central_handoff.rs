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
//! and queues it for the receiver like any relay send. The terminal signs its
//! receipt only after the relay accepted it.
//!
//! [ONION-HANDOFF-REVIEW 2026-10-11 by Claude] (Codex review) What the receipt
//! proves is unchanged: "accepted into relay custody". A node with the
//! hand-off disabled still signs the same `message_relay` receipt for its own
//! node-local custody, and nothing in the receipt tells the two apart, so a
//! client must not read it as "in the receiver's inbox". Telling them apart
//! needs a verifiable contract (a distinct purpose or proof mode), not a
//! deployment setting.
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

/// [ONION-HANDOFF-REVIEW 2026-10-11 by Claude] The relay's verdict is a few
/// dozen bytes. Anything longer is read only up to this bound and then
/// treated as no verdict, so each in-flight hand-off holds at most this much.
const MAX_RESPONSE_BYTES: usize = 4096;

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

    /// Gives one envelope to the relay; `Ok` only once the relay accepted it
    /// (or had already accepted it).
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
        let status = response.status().as_u16();
        let body = read_bounded(response, MAX_RESPONSE_BYTES).await?;
        classify_response(status, &body)
    }
}

/// The response body, read chunk by chunk and abandoned as soon as it grows
/// past `limit`; an oversized answer is no verdict.
async fn read_bounded(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, CentralHandoffError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(CentralHandoffError::Unavailable);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| CentralHandoffError::Unavailable)?
    {
        if body.len() + chunk.len() > limit {
            return Err(CentralHandoffError::Unavailable);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
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

    /// [ONION-HANDOFF-REVIEW 2026-10-11 by Claude] An answer larger than the
    /// bound is no verdict, whether it announces its length or streams.
    #[tokio::test]
    async fn an_oversized_answer_is_no_verdict_and_is_not_read_whole() {
        use axum::{body::Body, http::StatusCode, response::Response, routing::post, Router};

        let announced = Router::new().route(
            "/announced",
            // A genuine acceptance, padded: without the bound it would pass.
            post(|| async {
                (
                    StatusCode::ACCEPTED,
                    format!(r#"{{"accepted":true,"pad":"{}"}}"#, "x".repeat(1 << 20)),
                )
            }),
        );
        let streamed = Router::new().route(
            "/streamed",
            post(|| async {
                // The same padded acceptance as 64 KiB chunks, no
                // Content-Length; far beyond the bound.
                let mut parts = vec![br#"{"accepted":true,"pad":""#.to_vec()];
                parts.extend((0..64).map(|_| vec![b'x'; 64 * 1024]));
                parts.push(br#""}"#.to_vec());
                let chunks = futures::stream::iter(parts.into_iter().map(Ok::<_, std::io::Error>));
                Response::builder()
                    .status(StatusCode::ACCEPTED)
                    .body(Body::from_stream(chunks))
                    .unwrap()
            }),
        );
        let small = Router::new().route(
            "/small",
            post(|| async { (StatusCode::ACCEPTED, r#"{"accepted":true}"#) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, announced.merge(streamed).merge(small))
                .await
                .unwrap();
        });
        let identity = Arc::new(IdentityKeyPair::generate());
        let envelope = golden::golden_envelope();
        for (path, expected) in [
            ("announced", Err(CentralHandoffError::Unavailable)),
            ("streamed", Err(CentralHandoffError::Unavailable)),
            ("small", Ok(())),
        ] {
            let handoff =
                CentralHandoff::new(&format!("http://{address}/{path}"), Arc::clone(&identity))
                    .unwrap();
            assert_eq!(handoff.deliver(&envelope).await, expected, "{path}");
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
