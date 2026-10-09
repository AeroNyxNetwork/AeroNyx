// ============================================
// File: crates/aeronyx-server/src/tee.rs
// ============================================
//! # dstack TEE integration (Phala Cloud / self-hosted dstack)
//!
//! ## Creation Reason
//! [TEE-DSTACK 2026-10-09 by Claude] A node in a TDX confidential VM is only
//! worth more than a VPS if its identity is bound to the measured app and a
//! peer can check that. This module talks to the dstack guest agent for two
//! things: deriving the node identity from the app-bound KMS key, and
//! producing a TDX quote whose report data commits to that identity.
//!
//! ## Main Functionality
//! - `DstackClient::derive_identity_seed`: `POST /GetKey` → 32-byte seed. The
//!   KMS derives it from the app id and a fixed path, so the same app gets the
//!   same node identity across restarts and redeploys, and nothing outside the
//!   measured app can obtain it. The seed is never written to disk.
//! - `DstackClient::quote_for_identity`: `POST /GetQuote` with
//!   `report_data = SHA-512(domain ‖ node public key ‖ nonce)`.
//!
//! ## Dependencies
//! - dstack guest agent HTTP API on a Unix socket (default
//!   `/var/run/dstack.sock`, mounted into the container by dstack).
//!
//! ## Important Notes for Next Developer
//! - Changing `IDENTITY_KEY_PATH` changes every TEE node's identity. Treat it
//!   as a protocol constant.
//! - `ATTESTATION_DOMAIN` is part of the verifier contract: clients recompute
//!   the same SHA-512 to check a quote.
//!
//! Last Modified: v1.0.0 - Identity derivation and identity-bound quotes.
// ============================================

use std::path::{Path, PathBuf};
use std::time::Duration;

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::Request;
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};
use tokio::net::UnixStream;
use tokio::sync::Semaphore;

/// Default dstack guest agent socket inside a dstack CVM container.
pub const DEFAULT_DSTACK_SOCKET: &str = "/var/run/dstack.sock";
/// KMS key path for the node identity. Protocol constant; see module notes.
pub const IDENTITY_KEY_PATH: &str = "aeronyx/node-identity/v1";
/// Domain separator for identity-bound quote report data.
pub const ATTESTATION_DOMAIN: &[u8] = b"aeronyx-node-attestation-v1";
/// Caller nonce size for identity-bound quotes.
pub const ATTESTATION_NONCE_BYTES: usize = 32;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RESPONSE_BYTES: usize = 256 * 1024;

/// Errors from the guest agent. Messages never include key material.
#[derive(Debug, thiserror::Error)]
pub enum TeeError {
    /// The socket could not be reached or the exchange failed.
    #[error("dstack guest agent unavailable: {0}")]
    Unavailable(String),
    /// The agent answered with something this client does not accept.
    #[error("dstack guest agent returned an invalid response: {0}")]
    InvalidResponse(&'static str),
}

/// Thin client for the dstack guest agent.
#[derive(Debug, Clone)]
pub struct DstackClient {
    socket: PathBuf,
}

#[derive(Serialize)]
struct GetKeyRequest<'a> {
    path: &'a str,
    purpose: &'a str,
}

#[derive(Deserialize)]
struct GetKeyResponse {
    key: String,
}

#[derive(Serialize)]
struct GetQuoteRequest {
    report_data: String,
}

/// TDX quote plus the event log needed to replay RTMR3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DstackQuote {
    /// Hex-encoded TDX quote.
    pub quote: String,
    /// JSON event log as returned by the agent.
    #[serde(default)]
    pub event_log: String,
}

impl DstackClient {
    /// Creates a client for the given socket path.
    #[must_use]
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    /// Socket path in use.
    #[must_use]
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Derives the 32-byte node identity seed from the app-bound KMS key.
    ///
    /// # Errors
    /// Fails closed if the agent is unreachable or the key is shorter than
    /// 32 bytes; the caller must not fall back to a generated key.
    pub async fn derive_identity_seed(&self) -> Result<[u8; 32], TeeError> {
        let body = serde_json::to_vec(&GetKeyRequest {
            path: IDENTITY_KEY_PATH,
            purpose: "ed25519-node-identity",
        })
        .map_err(|_| TeeError::InvalidResponse("encode GetKey"))?;
        let response: GetKeyResponse = self.post_json("/GetKey", body).await?;
        let key = hex::decode(response.key.trim_start_matches("0x"))
            .map_err(|_| TeeError::InvalidResponse("GetKey key is not hex"))?;
        if key.len() < 32 {
            return Err(TeeError::InvalidResponse(
                "GetKey key shorter than 32 bytes",
            ));
        }
        // Hash rather than truncate so the Ed25519 seed is uniformly derived
        // from the whole KMS output whatever its curve or length.
        let mut hasher = Sha512::new();
        hasher.update(IDENTITY_KEY_PATH.as_bytes());
        hasher.update(&key);
        let digest = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&digest[..32]);
        Ok(seed)
    }

    /// Returns a TDX quote whose report data commits to `node_id` and `nonce`.
    ///
    /// # Errors
    /// Fails if the agent cannot produce a quote.
    pub async fn quote_for_identity(
        &self,
        node_id: &[u8; 32],
        nonce: &[u8; ATTESTATION_NONCE_BYTES],
    ) -> Result<DstackQuote, TeeError> {
        let report_data = identity_report_data(node_id, nonce);
        let body = serde_json::to_vec(&GetQuoteRequest {
            report_data: hex::encode(report_data),
        })
        .map_err(|_| TeeError::InvalidResponse("encode GetQuote"))?;
        let quote: DstackQuote = self.post_json("/GetQuote", body).await?;
        if quote.quote.is_empty() || hex::decode(&quote.quote).is_err() {
            return Err(TeeError::InvalidResponse("GetQuote quote is not hex"));
        }
        Ok(quote)
    }

    async fn post_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &'static str,
        body: Vec<u8>,
    ) -> Result<T, TeeError> {
        let exchange = async {
            let stream = UnixStream::connect(&self.socket)
                .await
                .map_err(|error| TeeError::Unavailable(error.kind().to_string()))?;
            let (mut sender, connection) =
                hyper::client::conn::http1::handshake(TokioIo::new(stream))
                    .await
                    .map_err(|_| TeeError::Unavailable("handshake".into()))?;
            tokio::spawn(connection);
            let request = Request::post(path)
                .header(hyper::header::HOST, "localhost")
                .header(hyper::header::CONTENT_TYPE, "application/json")
                .body(Full::new(Bytes::from(body)))
                .map_err(|_| TeeError::InvalidResponse("build request"))?;
            let response = sender
                .send_request(request)
                .await
                .map_err(|_| TeeError::Unavailable("request".into()))?;
            if !response.status().is_success() {
                return Err(TeeError::InvalidResponse("non-success status"));
            }
            let bytes = Limited::new(response.into_body(), MAX_RESPONSE_BYTES)
                .collect()
                .await
                .map_err(|_| TeeError::InvalidResponse("body too large or truncated"))?
                .to_bytes();
            serde_json::from_slice(&bytes).map_err(|_| TeeError::InvalidResponse("JSON"))
        };
        tokio::time::timeout(REQUEST_TIMEOUT, exchange)
            .await
            .map_err(|_| TeeError::Unavailable("timeout".into()))?
    }
}

/// Public path of the identity-bound quote endpoint.
pub const ATTESTATION_QUOTE_PATH: &str = "/api/attestation/quote";
/// Quote generation is comparatively expensive; bound concurrent requests.
const ATTESTATION_MAX_IN_FLIGHT: usize = 2;

#[derive(Clone)]
struct AttestationState {
    client: DstackClient,
    node_id: [u8; 32],
    admission: Arc<Semaphore>,
}

#[derive(Deserialize)]
struct QuoteQuery {
    nonce: String,
}

#[derive(Serialize)]
struct QuoteResponse {
    node_id: String,
    nonce: String,
    report_data: String,
    report_data_scheme: &'static str,
    quote: String,
    event_log: String,
}

/// `GET /api/attestation/quote?nonce=<64 hex>`: a fresh TDX quote whose report
/// data commits to this node's identity and the caller's nonce. Mounted only
/// when the identity itself comes from dstack, so the quote and the signed
/// descriptor speak for the same key.
pub fn build_attestation_router(client: DstackClient, node_id: [u8; 32]) -> Router {
    Router::new()
        .route(ATTESTATION_QUOTE_PATH, get(handle_quote))
        .with_state(AttestationState {
            client,
            node_id,
            admission: Arc::new(Semaphore::new(ATTESTATION_MAX_IN_FLIGHT)),
        })
}

async fn handle_quote(
    State(state): State<AttestationState>,
    Query(query): Query<QuoteQuery>,
) -> Response {
    let Ok(nonce) = hex::decode(&query.nonce) else {
        return (StatusCode::BAD_REQUEST, "nonce must be hex").into_response();
    };
    let Ok(nonce) = <[u8; ATTESTATION_NONCE_BYTES]>::try_from(nonce.as_slice()) else {
        return (StatusCode::BAD_REQUEST, "nonce must be 32 bytes").into_response();
    };
    let Ok(_permit) = state.admission.try_acquire() else {
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    };
    match state
        .client
        .quote_for_identity(&state.node_id, &nonce)
        .await
    {
        Ok(quote) => Json(QuoteResponse {
            node_id: hex::encode(state.node_id),
            nonce: hex::encode(nonce),
            report_data: hex::encode(identity_report_data(&state.node_id, &nonce)),
            report_data_scheme: "sha512(\"aeronyx-node-attestation-v1\" || node_id || nonce)",
            quote: quote.quote,
            event_log: quote.event_log,
        })
        .into_response(),
        Err(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

/// `SHA-512(ATTESTATION_DOMAIN ‖ node_id ‖ nonce)`, the 64-byte TDX report data.
#[must_use]
pub fn identity_report_data(node_id: &[u8; 32], nonce: &[u8; ATTESTATION_NONCE_BYTES]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(ATTESTATION_DOMAIN);
    hasher.update(node_id);
    hasher.update(nonce);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    /// Minimal fake guest agent: records the request and answers one fixed body.
    async fn fake_agent(answer: &'static str) -> (tempfile::TempDir, Arc<Mutex<String>>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = dir.path().join("dstack.sock");
        let listener = UnixListener::bind(&socket).expect("bind");
        let seen = Arc::new(Mutex::new(String::new()));
        let record = Arc::clone(&seen);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut buffer = vec![0u8; 8192];
            let read = stream.read(&mut buffer).await.expect("read");
            *record.lock().expect("lock") = String::from_utf8_lossy(&buffer[..read]).into_owned();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                answer.len(),
                answer
            );
            stream.write_all(response.as_bytes()).await.expect("write");
        });
        (dir, seen)
    }

    #[tokio::test]
    async fn identity_seed_is_derived_from_the_fixed_kms_path() {
        let key = "11".repeat(32);
        let answer: &'static str =
            Box::leak(format!(r#"{{"key":"{key}","signature_chain":[]}}"#).into_boxed_str());
        let (dir, seen) = fake_agent(answer).await;
        let client = DstackClient::new(dir.path().join("dstack.sock"));
        let seed = client.derive_identity_seed().await.expect("seed");
        let request = seen.lock().expect("lock").clone();
        assert!(request.starts_with("POST /GetKey HTTP/1.1"));
        assert!(request.contains(IDENTITY_KEY_PATH));
        let mut hasher = Sha512::new();
        hasher.update(IDENTITY_KEY_PATH.as_bytes());
        hasher.update([0x11u8; 32]);
        assert_eq!(seed[..], hasher.finalize()[..32]);
    }

    #[tokio::test]
    async fn short_kms_key_fails_closed() {
        let (dir, _) = fake_agent(r#"{"key":"1111"}"#).await;
        let client = DstackClient::new(dir.path().join("dstack.sock"));
        assert!(matches!(
            client.derive_identity_seed().await,
            Err(TeeError::InvalidResponse(_))
        ));
    }

    #[tokio::test]
    async fn missing_agent_is_unavailable_not_a_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let client = DstackClient::new(dir.path().join("absent.sock"));
        assert!(matches!(
            client.derive_identity_seed().await,
            Err(TeeError::Unavailable(_))
        ));
    }

    #[tokio::test]
    async fn quote_endpoint_rejects_bad_nonce_and_serves_bound_quote() {
        use tower::ServiceExt;
        let (dir, _) = fake_agent(r#"{"quote":"abcd","event_log":"[]"}"#).await;
        let app =
            build_attestation_router(DstackClient::new(dir.path().join("dstack.sock")), [7u8; 32]);
        let bad = app
            .clone()
            .oneshot(
                axum::http::Request::get(format!("{ATTESTATION_QUOTE_PATH}?nonce=00"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let nonce = "09".repeat(32);
        let ok = app
            .oneshot(
                axum::http::Request::get(format!("{ATTESTATION_QUOTE_PATH}?nonce={nonce}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ok.status(), StatusCode::OK);
        let body = ok.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["quote"], "abcd");
        assert_eq!(
            value["report_data"],
            hex::encode(identity_report_data(&[7u8; 32], &[9u8; 32]))
        );
    }

    #[tokio::test]
    async fn quote_request_carries_identity_bound_report_data() {
        let (dir, seen) = fake_agent(r#"{"quote":"abcd","event_log":"[]"}"#).await;
        let client = DstackClient::new(dir.path().join("dstack.sock"));
        let (node_id, nonce) = ([7u8; 32], [9u8; 32]);
        let quote = client
            .quote_for_identity(&node_id, &nonce)
            .await
            .expect("quote");
        assert_eq!(quote.quote, "abcd");
        let request = seen.lock().expect("lock").clone();
        assert!(request.starts_with("POST /GetQuote HTTP/1.1"));
        assert!(request.contains(&hex::encode(identity_report_data(&node_id, &nonce))));
    }
}
