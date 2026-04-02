// ============================================
// File: crates/aeronyx-server/src/api/chat_handlers.rs
// ============================================
//! # Chat Relay HTTP API — Blob Endpoints
//!
//! ## Creation Reason
//! Provides the HTTP API for encrypted media blob upload/download/delete,
//! complementing the UDP-based `ChatRelay` message path. Large files
//! (images, documents) travel via HTTP while the small `ChatEnvelope`
//! (containing an encrypted `MediaPointer`) travels via UDP MemChain.
//!
//! ## Main Functionality
//! - `POST /api/chat/blob` — Upload an encrypted blob (Alice → Node)
//! - `GET  /api/chat/blob/{blob_id}` — Download an encrypted blob (Node → Bob)
//! - `DELETE /api/chat/blob/{blob_id}` — Retract a blob (Alice → Node)
//! - `build_chat_router()` — Assembles the Axum sub-router for mounting
//!
//! ## Request Authentication
//! All mutating endpoints (POST, DELETE) require Ed25519 signature auth:
//!
//! ```text
//! POST /api/chat/blob
//!   Header: X-Sender-Wallet: <64-hex Ed25519 public key>
//!   Header: X-Receiver-Wallet: <64-hex Ed25519 public key>
//!   Header: X-Signature: <128-hex Ed25519 signature over SHA-256(body)>
//!   Header: X-File-Hash: <64-hex SHA-256 of the encrypted file bytes>
//!   Body: raw encrypted bytes (application/octet-stream)
//!   Response 200: { "blob_id": "...", "expires_in_secs": 259200 }
//!   Response 400: { "error": "..." }
//!   Response 413: { "error": "blob too large: ..." }
//!
//! GET /api/chat/blob/{blob_id}
//!   Header: X-Wallet: <64-hex> (optional, for logging only)
//!   Response 200: raw bytes (application/octet-stream)
//!   Response 404: { "error": "blob not found" }
//!
//! DELETE /api/chat/blob/{blob_id}
//!   Header: X-Sender-Wallet: <64-hex>
//!   Header: X-Signature: <128-hex Ed25519 signature over SHA-256(blob_id bytes)>
//!   Response 200: { "status": "deleted" }
//!   Response 401: { "error": "unauthorized" }
//!   Response 404: { "error": "blob not found" }
//! ```
//!
//! ## Signature Verification
//! - POST: `sig = Ed25519.sign(sk, SHA256(body_bytes))`
//! - DELETE: `sig = Ed25519.sign(sk, SHA256(blob_id.as_bytes()))`
//!
//! Both use the same `IdentityPublicKey::verify()` from `aeronyx-core`.
//! The node verifies with the public key from the header — this prevents
//! an attacker from uploading blobs on behalf of another wallet.
//!
//! ## Why No Download ACL
//! Blobs are encrypted with a `file_key` that only the intended receiver
//! knows (embedded inside the E2E-encrypted `MediaPointer`). Downloading
//! an encrypted blob without `file_key` is useless. The `blob_id` itself
//! is HMAC-derived and unguessable, providing implicit access control.
//!
//! ## Dependencies
//! - `aeronyx-server/src/services/chat_relay.rs`: `ChatRelayService`
//! - `aeronyx-core/src/crypto/keys.rs`: `IdentityPublicKey::verify()`
//! - `axum`: routing, extractors, response types
//! - Called from `server.rs::start_combined_api()` via `build_chat_router()`
//!
//! ## ⚠️ Important Notes for Next Developer
//! - `Arc<ChatRelayService>` is passed via Axum `State` — do NOT store
//!   a `Mutex` around the entire service in State (the service already
//!   has internal locking).
//! - Body size is limited by `axum::extract::DefaultBodyLimit` set in
//!   `build_chat_router()`. This is a hard limit enforced BEFORE the
//!   handler runs, so `max_blob_size` in `ChatRelayConfig` is a second
//!   defence-in-depth check inside `put_blob()`.
//! - Hex-decoding headers is intentionally strict: any malformed header
//!   returns 400 immediately without touching the database.
//! - `X-File-Hash` must be SHA-256 of the ENCRYPTED bytes (not plaintext).
//!   This is used for blob_id derivation — the node never sees plaintext.
//!
//! ## Last Modified
//! v1.1.0-ChatRelay — Initial implementation

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use aeronyx_core::crypto::keys::IdentityPublicKey;

use crate::services::chat_relay::{ChatRelayError, ChatRelayService};

// ============================================
// Shared state
// ============================================

/// Axum state for the chat blob API.
///
/// Wraps `ChatRelayService` in an `Arc` for cheap cloning across handler calls.
/// The service itself uses internal `Mutex<Connection>` for SQLite access.
#[derive(Clone)]
pub struct ChatBlobState {
    pub relay: Arc<ChatRelayService>,
}

// ============================================
// Router builder
// ============================================

/// Builds the Axum sub-router for chat blob endpoints.
///
/// Mount this under the existing MPI router in `server.rs`:
/// ```rust,ignore
/// let app = build_mpi_router(mpi_state)
///     .merge(build_chat_router(Arc::clone(&chat_relay)));
/// ```
///
/// Body size limit is set to `max_blob_size + 4096` (extra headroom for
/// multipart boundaries; the service enforces the exact limit internally).
pub fn build_chat_router(relay: Arc<ChatRelayService>) -> Router {
    let max_body = relay.config().max_blob_size + 4096;
    let state = ChatBlobState { relay };

    Router::new()
        .route("/api/chat/blob", post(handle_blob_upload))
        .route("/api/chat/blob/:blob_id", get(handle_blob_download))
        .route("/api/chat/blob/:blob_id", delete(handle_blob_delete))
        .layer(axum::extract::DefaultBodyLimit::max(max_body))
        .with_state(state)
}

// ============================================
// POST /api/chat/blob
// ============================================

/// Upload an encrypted blob.
///
/// Required headers:
/// - `X-Sender-Wallet`: 64-hex Ed25519 public key of the uploader
/// - `X-Receiver-Wallet`: 64-hex Ed25519 public key of the intended recipient
/// - `X-Signature`: 128-hex Ed25519 signature over `SHA256(body)`
/// - `X-File-Hash`: 64-hex SHA-256 of the encrypted file bytes (= body bytes)
///
/// Returns `{ "blob_id": "...", "expires_in_secs": <u64> }` on success.
async fn handle_blob_upload(
    State(state): State<ChatBlobState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // ── Parse headers ──
    let sender = match parse_wallet_header(&headers, "x-sender-wallet") {
        Ok(w) => w,
        Err(msg) => return bad_request(msg),
    };
    let receiver = match parse_wallet_header(&headers, "x-receiver-wallet") {
        Ok(w) => w,
        Err(msg) => return bad_request(msg),
    };
    let signature = match parse_signature_header(&headers, "x-signature") {
        Ok(s) => s,
        Err(msg) => return bad_request(msg),
    };
    let file_hash = match parse_hash_header(&headers, "x-file-hash") {
        Ok(h) => h,
        Err(msg) => return bad_request(msg),
    };

    // ── Verify signature: sig covers SHA256(body) ──
    let body_hash: [u8; 32] = Sha256::digest(&body).into();
    if let Err(e) = verify_sig(&sender, &body_hash, &signature) {
        warn!(
            sender = %hex::encode(&sender[..4]),
            error = %e,
            "[CHAT_BLOB] Upload signature verification failed"
        );
        return error_response(StatusCode::UNAUTHORIZED, "signature verification failed");
    }

    // ── Verify X-File-Hash matches actual body hash ──
    // This prevents the client from supplying a mismatched hash that would
    // produce a blob_id not reachable by the receiver.
    if body_hash != file_hash {
        return bad_request("X-File-Hash does not match SHA-256 of body");
    }

    // ── Store blob ──
    match state.relay.put_blob(&sender, &receiver, &body, &file_hash) {
        Ok(blob_id) => {
            let expires_in = state.relay.config().offline_ttl_secs;
            debug!(
                blob_id = %blob_id,
                size = body.len(),
                "[CHAT_BLOB] Upload accepted"
            );
            (
                StatusCode::OK,
                Json(json!({
                    "blob_id": blob_id,
                    "expires_in_secs": expires_in,
                })),
            )
                .into_response()
        }
        Err(ChatRelayError::BlobTooLarge { size, limit }) => {
            warn!(size, limit, "[CHAT_BLOB] Upload rejected: too large");
            error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("blob too large: {} bytes (limit {})", size, limit),
            )
        }
        Err(ChatRelayError::BlobQuotaExceeded { current, limit }) => {
            warn!(current, limit, "[CHAT_BLOB] Upload rejected: quota exceeded");
            error_response(
                StatusCode::TOO_MANY_REQUESTS,
                format!("receiver blob quota exceeded: {}/{}", current, limit),
            )
        }
        Err(e) => {
            warn!(error = %e, "[CHAT_BLOB] Upload failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
        }
    }
}

// ============================================
// GET /api/chat/blob/:blob_id
// ============================================

/// Download an encrypted blob.
///
/// No authentication required — the blob is encrypted and the `blob_id`
/// is HMAC-derived (unguessable without the node secret).
///
/// Optional header:
/// - `X-Wallet`: requester wallet, used for logging only (not enforced)
///
/// Returns raw bytes (`application/octet-stream`) on success.
async fn handle_blob_download(
    State(state): State<ChatBlobState>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    // Log requester wallet if provided (best-effort, not enforced)
    if let Some(wallet_hex) = headers.get("x-wallet").and_then(|v| v.to_str().ok()) {
        debug!(
            blob_id = %blob_id,
            wallet = %wallet_hex.get(..8).unwrap_or(wallet_hex),
            "[CHAT_BLOB] Download request"
        );
    }

    match state.relay.get_blob(&blob_id) {
        Ok(data) => {
            use axum::http::header;
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/octet-stream")],
                data,
            )
                .into_response()
        }
        Err(ChatRelayError::BlobNotFound { .. }) => {
            error_response(StatusCode::NOT_FOUND, "blob not found or expired")
        }
        Err(e) => {
            warn!(error = %e, blob_id = %blob_id, "[CHAT_BLOB] Download error");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
        }
    }
}

// ============================================
// DELETE /api/chat/blob/:blob_id
// ============================================

/// Delete a blob (sender retraction).
///
/// Required headers:
/// - `X-Sender-Wallet`: 64-hex Ed25519 public key (must match original uploader)
/// - `X-Signature`: 128-hex Ed25519 signature over `SHA256(blob_id.as_bytes())`
///
/// Returns `{ "status": "deleted" }` on success.
async fn handle_blob_delete(
    State(state): State<ChatBlobState>,
    Path(blob_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let sender = match parse_wallet_header(&headers, "x-sender-wallet") {
        Ok(w) => w,
        Err(msg) => return bad_request(msg),
    };
    let signature = match parse_signature_header(&headers, "x-signature") {
        Ok(s) => s,
        Err(msg) => return bad_request(msg),
    };

    // Verify signature: sig covers SHA256(blob_id bytes)
    let id_hash: [u8; 32] = Sha256::digest(blob_id.as_bytes()).into();
    if let Err(e) = verify_sig(&sender, &id_hash, &signature) {
        warn!(
            sender = %hex::encode(&sender[..4]),
            error = %e,
            "[CHAT_BLOB] Delete signature verification failed"
        );
        return error_response(StatusCode::UNAUTHORIZED, "signature verification failed");
    }

    match state.relay.delete_blob(&blob_id, &sender) {
        Ok(()) => {
            debug!(blob_id = %blob_id, "[CHAT_BLOB] Deleted by sender");
            (StatusCode::OK, Json(json!({ "status": "deleted" }))).into_response()
        }
        Err(ChatRelayError::BlobNotFound { .. }) => {
            error_response(StatusCode::NOT_FOUND, "blob not found")
        }
        Err(ChatRelayError::Unauthorized) => {
            error_response(StatusCode::UNAUTHORIZED, "unauthorized")
        }
        Err(e) => {
            warn!(error = %e, "[CHAT_BLOB] Delete error");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
        }
    }
}

// ============================================
// Header parsing helpers
// ============================================

/// Parse a 64-hex wallet address from a request header.
fn parse_wallet_header(headers: &HeaderMap, name: &str) -> Result<[u8; 32], String> {
    let val = headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| format!("missing header: {}", name))?;

    let bytes = hex::decode(val)
        .map_err(|_| format!("invalid hex in header {}", name))?;

    if bytes.len() != 32 {
        return Err(format!("header {} must be 64 hex chars (32 bytes)", name));
    }

    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Parse a 128-hex Ed25519 signature from a request header.
fn parse_signature_header(headers: &HeaderMap, name: &str) -> Result<[u8; 64], String> {
    let val = headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| format!("missing header: {}", name))?;

    let bytes = hex::decode(val)
        .map_err(|_| format!("invalid hex in header {}", name))?;

    if bytes.len() != 64 {
        return Err(format!("header {} must be 128 hex chars (64 bytes)", name));
    }

    let mut arr = [0u8; 64];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Parse a 64-hex SHA-256 hash from a request header.
fn parse_hash_header(headers: &HeaderMap, name: &str) -> Result<[u8; 32], String> {
    // Same layout as wallet — reuse the wallet parser
    parse_wallet_header(headers, name)
}

// ============================================
// Signature verification
// ============================================

/// Verify an Ed25519 signature using the provided public key bytes.
fn verify_sig(
    pubkey_bytes: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), aeronyx_core::error::CoreError> {
    let pk = IdentityPublicKey::from_bytes(pubkey_bytes)?;
    pk.verify(message, signature)
}

// ============================================
// Response helpers
// ============================================

fn bad_request(msg: impl Into<String>) -> Response {
    error_response(StatusCode::BAD_REQUEST, msg)
}

fn error_response(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use axum::{
        body::Body,
        http::{Method, Request},
    };
    use tower::ServiceExt; // for `oneshot`

    use crate::config::ChatRelayConfig;
    use crate::services::chat_relay::derive_node_secret;

    fn make_relay() -> Arc<ChatRelayService> {
        let config = ChatRelayConfig {
            enabled: true,
            db_path: ":memory:".to_string(),
            offline_ttl_secs: 259_200,
            max_pending_per_wallet: 500,
            max_message_size: 65_536,
            max_blob_size: 1_024,
            max_blobs_per_receiver: 50,
            cleanup_interval_secs: 60,
            dedup_lru_capacity: 10_000,
            expired_notification_ttl_secs: 604_800,
        };
        let secret = derive_node_secret(&[0x42u8; 32]);
        Arc::new(ChatRelayService::new(config, secret).expect("init"))
    }

    fn sign_bytes(kp: &IdentityKeyPair, data: &[u8]) -> String {
        let hash: [u8; 32] = Sha256::digest(data).into();
        hex::encode(kp.sign(&hash))
    }

    // ── Upload ──

    #[tokio::test]
    async fn test_upload_success() {
        let relay = make_relay();
        let app = build_chat_router(Arc::clone(&relay));

        let kp = IdentityKeyPair::generate();
        let sender_hex = hex::encode(kp.public_key_bytes());
        let receiver_hex = hex::encode([0xBBu8; 32]);
        let body_bytes = b"encrypted_image_data";
        let file_hash: [u8; 32] = Sha256::digest(body_bytes).into();
        let file_hash_hex = hex::encode(file_hash);
        let sig_hex = sign_bytes(&kp, body_bytes);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/chat/blob")
            .header("x-sender-wallet", &sender_hex)
            .header("x-receiver-wallet", &receiver_hex)
            .header("x-signature", &sig_hex)
            .header("x-file-hash", &file_hash_hex)
            .header("content-type", "application/octet-stream")
            .body(Body::from(body_bytes.as_slice()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["blob_id"].is_string());
        assert_eq!(json["blob_id"].as_str().unwrap().len(), 32);
        assert!(json["expires_in_secs"].is_number());
    }

    #[tokio::test]
    async fn test_upload_missing_header_rejected() {
        let relay = make_relay();
        let app = build_chat_router(Arc::clone(&relay));

        // Missing x-receiver-wallet
        let kp = IdentityKeyPair::generate();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/chat/blob")
            .header("x-sender-wallet", hex::encode(kp.public_key_bytes()))
            .header("x-signature", "00".repeat(64))
            .header("x-file-hash", "00".repeat(32))
            .body(Body::from("data"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_upload_wrong_signature_rejected() {
        let relay = make_relay();
        let app = build_chat_router(Arc::clone(&relay));

        let kp = IdentityKeyPair::generate();
        let body_bytes = b"data";
        let file_hash: [u8; 32] = Sha256::digest(body_bytes).into();

        // Wrong signature (signed wrong message)
        let bad_sig = hex::encode(kp.sign(b"wrong_message"));

        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/chat/blob")
            .header("x-sender-wallet", hex::encode(kp.public_key_bytes()))
            .header("x-receiver-wallet", hex::encode([0xBBu8; 32]))
            .header("x-signature", bad_sig)
            .header("x-file-hash", hex::encode(file_hash))
            .body(Body::from(body_bytes.as_slice()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_upload_too_large_rejected() {
        let relay = make_relay(); // max_blob_size = 1024
        let app = build_chat_router(Arc::clone(&relay));

        let kp = IdentityKeyPair::generate();
        let body_bytes = vec![0u8; 2048];
        let file_hash: [u8; 32] = Sha256::digest(&body_bytes).into();
        let sig_hex = sign_bytes(&kp, &body_bytes);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/chat/blob")
            .header("x-sender-wallet", hex::encode(kp.public_key_bytes()))
            .header("x-receiver-wallet", hex::encode([0xBBu8; 32]))
            .header("x-signature", sig_hex)
            .header("x-file-hash", hex::encode(file_hash))
            .body(Body::from(body_bytes))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    // ── Download ──

    #[tokio::test]
    async fn test_download_success() {
        let relay = make_relay();

        // Pre-insert blob
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"blob_content";
        let file_hash: [u8; 32] = Sha256::digest(data).into();
        let blob_id = relay.put_blob(&sender, &receiver, data, &file_hash).unwrap();

        let app = build_chat_router(Arc::clone(&relay));
        let req = Request::builder()
            .method(Method::GET)
            .uri(format!("/api/chat/blob/{}", blob_id))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), data);
    }

    #[tokio::test]
    async fn test_download_not_found() {
        let relay = make_relay();
        let app = build_chat_router(relay);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/chat/blob/nonexistentblobid00000000000000")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── Delete ──

    #[tokio::test]
    async fn test_delete_success() {
        let relay = make_relay();
        let kp = IdentityKeyPair::generate();
        let sender = kp.public_key_bytes();
        let receiver = [0xBBu8; 32];
        let data = b"to_delete";
        let file_hash: [u8; 32] = Sha256::digest(data).into();
        let blob_id = relay.put_blob(&sender, &receiver, data, &file_hash).unwrap();

        let app = build_chat_router(Arc::clone(&relay));

        // Sign SHA256(blob_id bytes)
        let id_hash: [u8; 32] = Sha256::digest(blob_id.as_bytes()).into();
        let sig = hex::encode(kp.sign(&id_hash));

        let req = Request::builder()
            .method(Method::DELETE)
            .uri(format!("/api/chat/blob/{}", blob_id))
            .header("x-sender-wallet", hex::encode(sender))
            .header("x-signature", sig)
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify blob is gone
        assert!(matches!(
            relay.get_blob(&blob_id),
            Err(ChatRelayError::BlobNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_delete_wrong_sender_rejected() {
        let relay = make_relay();
        let kp_owner = IdentityKeyPair::generate();
        let kp_attacker = IdentityKeyPair::generate();

        let sender = kp_owner.public_key_bytes();
        let receiver = [0xBBu8; 32];
        let data = b"protected";
        let file_hash: [u8; 32] = Sha256::digest(data).into();
        let blob_id = relay.put_blob(&sender, &receiver, data, &file_hash).unwrap();

        let app = build_chat_router(Arc::clone(&relay));

        // Attacker signs with their own key
        let id_hash: [u8; 32] = Sha256::digest(blob_id.as_bytes()).into();
        let sig = hex::encode(kp_attacker.sign(&id_hash));

        let req = Request::builder()
            .method(Method::DELETE)
            .uri(format!("/api/chat/blob/{}", blob_id))
            .header("x-sender-wallet", hex::encode(kp_attacker.public_key_bytes()))
            .header("x-signature", sig)
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Header parsing ──

    #[test]
    fn test_parse_wallet_header_valid() {
        let mut headers = HeaderMap::new();
        headers.insert("x-test", hex::encode([0xAAu8; 32]).parse().unwrap());
        let result = parse_wallet_header(&headers, "x-test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), [0xAAu8; 32]);
    }

    #[test]
    fn test_parse_wallet_header_missing() {
        let headers = HeaderMap::new();
        assert!(parse_wallet_header(&headers, "x-missing").is_err());
    }

    #[test]
    fn test_parse_wallet_header_wrong_length() {
        let mut headers = HeaderMap::new();
        headers.insert("x-test", "deadbeef".parse().unwrap()); // 4 bytes, not 32
        assert!(parse_wallet_header(&headers, "x-test").is_err());
    }
}
