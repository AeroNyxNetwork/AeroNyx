// ============================================
// File: crates/aeronyx-server/src/management/ws_client.rs
// ============================================
//! # WebSocket Tunnel — CMS ↔ Local Services
//!
//! ## Modification Reason (v2.3.0+RemoteStorage)
//! MPI Proxy now supports `auth_headers` passthrough for Phase 1 remote storage.
//! When CMS forwards a remote user's MPI request through the WebSocket tunnel,
//! the `X-MemChain-*` Ed25519 signature headers are extracted from the WS message
//! and injected into the HTTP request to the local MPI API.
//!
//! Also fixed:
//! - Reuse `self.http_client` in MPI proxy (was creating new client per request)
//! - Inject Bearer token when `api_secret` is available (MPI proxy + store_chat)
//!
//! ## Previous Modifications
//! v1.3.0 - Initial creation (Phase 3: WebSocket Tunnel)
//! v1.3.2 - Replaced OpenClaw WS RPC with HTTP API
//! v1.4.0 - Fixed SSE [DONE] signal lost in residual buffer
//! v1.4.1 - Fixed done=true silently lost (missing flush after send)
//! v2.2.0 - 🌟 Added MPI proxy for MemExplorer frontend
//! v2.3.0 - 🌟 MPI proxy auth_headers passthrough (Ed25519 signature from CMS)
//!   🐛 Fixed: MPI proxy now reuses self.http_client instead of creating new client
//!   🐛 Fixed: MPI proxy + store_chat inject Bearer token when api_secret configured
//!
//! ## MPI Proxy Auth Flow (v2.3.0)
//! ```text
//! CMS WebSocket message:
//!   {
//!     "action": "mpi_remember",
//!     "payload": { "content": "...", ... },
//!     "auth_headers": {                          ← NEW (v2.3.0)
//!       "X-MemChain-PublicKey": "abcd1234...",
//!       "X-MemChain-Timestamp": "1741651200",
//!       "X-MemChain-Signature": "deadbeef..."
//!     }
//!   }
//!
//! Rust MPI Proxy forwards to 127.0.0.1:8421:
//!   POST /api/mpi/remember
//!   X-MemChain-PublicKey: abcd1234...      ← Passthrough
//!   X-MemChain-Timestamp: 1741651200       ← Passthrough
//!   X-MemChain-Signature: deadbeef...      ← Passthrough
//!   Content-Type: application/json
//!   { "content": "...", ... }
//!
//! MPI unified_auth_middleware:
//!   Detects X-MemChain-Signature → Ed25519 verification
//!   owner = signer's public key (remote user)
//! ```
//!
//! ## When auth_headers is absent (backward compatible):
//! - If api_secret is configured → inject Bearer token (local owner)
//! - If api_secret is not configured → no auth headers (open access)
//! This preserves existing MemExplorer frontend behavior (no auth_headers).
//!
//! ⚠️ Important Note for Next Developer:
//! - auth_headers in WS message takes priority over Bearer token
//! - If auth_headers is present, Bearer token is NOT injected (they are mutually exclusive)
//! - The 3 header names must match exactly what MPI middleware expects:
//!   X-MemChain-PublicKey, X-MemChain-Timestamp, X-MemChain-Signature
//! - api_secret is loaded once at construction time from the config file
//!
//! ## Last Modified
//! v2.3.0+RemoteStorage - 🌟 auth_headers passthrough + Bearer token injection + client reuse

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{connect_async, MaybeTlsStream};
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::E2eSession;
use crate::services::AgentManager;

// ============================================
// Constants
// ============================================

const CMS_WS_BASE_URL: &str = "wss://api.aeronyx.network/ws/node/tunnel";
const OPENCLAW_HTTP_BASE: &str = "http://127.0.0.1:18789";
const OPENCLAW_CHAT_PATH: &str = "/v1/chat/completions";
const MAX_RECONNECT_BACKOFF_SECS: u64 = 60;
const INITIAL_RECONNECT_DELAY_SECS: u64 = 1;
const PING_INTERVAL_SECS: u64 = 30;
const AUTH_TIMEOUT_SECS: u64 = 25;
const OPENCLAW_REQUEST_TIMEOUT_SECS: u64 = 180;
const OPENCLAW_AGENT_ID: &str = "main";

/// MPI API base URL (local Axum server).
const MPI_BASE_URL: &str = "http://127.0.0.1:8421";

/// Timeout for MPI proxy requests (embed is slowest at ~300ms).
const MPI_PROXY_TIMEOUT_SECS: u64 = 10;

/// Allowed MPI actions that can be proxied from CMS WebSocket.
/// Actions not in this list are rejected with an error response.
///
/// ## Security
/// - `mpi_log` is deliberately excluded: /log is called by the plugin
///   after each session, not by the frontend. Also local-only in v2.3.0.
/// - Only read-oriented actions + remember/forget (user-initiated) are allowed.
const MPI_ALLOWED_ACTIONS: &[&str] = &[
    "mpi_status",
    "mpi_recall",
    "mpi_remember",
    "mpi_forget",
    "mpi_embed",
    "mpi_record",
    "mpi_overview",
];

/// v2.3.0: Header names for Ed25519 remote auth passthrough.
/// These must match exactly what MPI unified_auth_middleware expects.
const HEADER_MEMCHAIN_PUBKEY: &str = "X-MemChain-PublicKey";
const HEADER_MEMCHAIN_TIMESTAMP: &str = "X-MemChain-Timestamp";
const HEADER_MEMCHAIN_SIGNATURE: &str = "X-MemChain-Signature";

// ============================================
// Type aliases
// ============================================

type WsStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<WsStream, WsMessage>;
type WsSource = SplitStream<WsStream>;

fn ws_text(value: &serde_json::Value) -> WsMessage {
    WsMessage::Text(value.to_string().into())
}

/// Helper: send a WebSocket text frame AND flush to TCP.
async fn ws_send_flush(sink: &mut WsSink, msg: WsMessage) -> Result<(), String> {
    sink.send(msg).await.map_err(|e| format!("WebSocket send failed: {}", e))?;
    sink.flush().await.map_err(|e| format!("WebSocket flush failed: {}", e))?;
    Ok(())
}

// ============================================
// WsTunnel
// ============================================

pub struct WsTunnel {
    identity: IdentityKeyPair,
    cms_node_id: String,
    agent_manager: Arc<AgentManager>,
    http_client: reqwest::Client,
    /// E2E encryption session, established via e2e_init handshake.
    /// None until a frontend sends e2e_init with its ephemeral public key.
    /// Reset to None on disconnect (ephemeral — one per WS connection).
    e2e_session: std::sync::Mutex<Option<E2eSession>>,
    /// v2.3.0: MPI Bearer token secret for local auth.
    /// Loaded from config at construction time. When Some, injected into
    /// MPI proxy requests that don't have auth_headers (local/MemExplorer requests).
    /// When None, no Bearer token is injected (backward compatible).
    mpi_api_secret: Option<String>,
}

impl WsTunnel {
    pub fn new(
        identity: IdentityKeyPair,
        cms_node_id: String,
        agent_manager: Arc<AgentManager>,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .pool_idle_timeout(Duration::from_secs(60))
            .timeout(Duration::from_secs(MPI_PROXY_TIMEOUT_SECS))
            .build()
            .expect("Failed to build reqwest::Client");

        Self {
            identity, cms_node_id, agent_manager, http_client,
            e2e_session: std::sync::Mutex::new(None),
            mpi_api_secret: None,
        }
    }

    /// Set the MPI API secret for Bearer token injection.
    ///
    /// Called from server.rs after reading config.memchain.effective_api_secret().
    /// When set, MPI proxy requests without auth_headers will include
    /// `Authorization: Bearer <secret>` for local MPI auth.
    #[must_use]
    pub fn with_mpi_api_secret(mut self, secret: Option<String>) -> Self {
        self.mpi_api_secret = secret;
        self
    }

    pub async fn run(self, mut shutdown: broadcast::Receiver<()>) {
        info!(cms_node_id = %self.cms_node_id, "[WS_TUNNEL] Starting WebSocket tunnel");
        let mut backoff_secs = INITIAL_RECONNECT_DELAY_SECS;

        loop {
            if shutdown.try_recv().is_ok() {
                info!("[WS_TUNNEL] Shutdown signal received before connect");
                break;
            }

            let url = format!("{}/{}/", CMS_WS_BASE_URL, self.cms_node_id);
            info!(url = %url, "[WS_TUNNEL] Connecting to CMS...");

            match self.connect_and_run(&url, &mut shutdown).await {
                Ok(ShutdownReason::GracefulShutdown) => {
                    info!("[WS_TUNNEL] Graceful shutdown");
                    break;
                }
                Ok(ShutdownReason::AuthFailed(reason)) => {
                    error!(reason = %reason, "[WS_TUNNEL] ❌ Auth failed — will retry");
                    backoff_secs = (backoff_secs * 2).min(MAX_RECONNECT_BACKOFF_SECS);
                }
                Ok(ShutdownReason::Disconnected(reason)) => {
                    warn!(reason = %reason, backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ⚠️ Disconnected — reconnecting in {}s", backoff_secs);
                }
                Err(e) => {
                    warn!(error = %e, backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ❌ Connection error — reconnecting in {}s", backoff_secs);
                }
            }

            tokio::select! {
                _ = shutdown.recv() => { info!("[WS_TUNNEL] Shutdown during backoff"); break; }
                _ = tokio::time::sleep(Duration::from_secs(backoff_secs)) => {}
            }
            backoff_secs = (backoff_secs * 2).min(MAX_RECONNECT_BACKOFF_SECS);
        }
        info!("[WS_TUNNEL] WebSocket tunnel stopped");
    }

    async fn connect_and_run(
        &self, url: &str, shutdown: &mut broadcast::Receiver<()>,
    ) -> Result<ShutdownReason, String> {
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| format!("WebSocket connect failed: {}", e))?;
        info!("[WS_TUNNEL] ✅ Connected to CMS");

        let (mut sink, mut source) = ws_stream.split();
        if let Err(reason) = self.authenticate(&mut sink, &mut source).await {
            let _ = sink.close().await;
            return Ok(ShutdownReason::AuthFailed(reason));
        }
        info!("[WS_TUNNEL] ✅ Authenticated with CMS");
        self.message_loop(&mut sink, &mut source, shutdown).await
    }

    // ============================================
    // Authentication
    // ============================================

    async fn authenticate(&self, sink: &mut WsSink, source: &mut WsSource) -> Result<(), String> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let public_key_hex = hex::encode(self.identity.public_key_bytes());
        let message = format!("{}:{}", public_key_hex, timestamp);
        let signature_hex = hex::encode(self.identity.sign(message.as_bytes()));

        let auth_msg = serde_json::json!({
            "type": "auth", "node_id": public_key_hex,
            "timestamp": timestamp, "signature": signature_hex,
        });
        ws_send_flush(sink, ws_text(&auth_msg)).await
            .map_err(|e| format!("Failed to send auth: {}", e))?;

        let auth_result = tokio::time::timeout(
            Duration::from_secs(AUTH_TIMEOUT_SECS),
            Self::wait_for_auth_ok(source),
        ).await;

        match auth_result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("Auth timeout — no auth_ok received".to_string()),
        }
    }

    async fn wait_for_auth_ok(source: &mut WsSource) -> Result<(), String> {
        while let Some(msg_result) = source.next().await {
            let msg = msg_result.map_err(|e| format!("WS read error during auth: {}", e))?;
            match msg {
                WsMessage::Text(text) => {
                    let parsed: serde_json::Value = serde_json::from_str(&text)
                        .map_err(|e| format!("Invalid JSON from CMS: {}", e))?;
                    match parsed.get("type").and_then(|t| t.as_str()) {
                        Some("auth_ok") => {
                            let name = parsed.get("node_name").and_then(|n| n.as_str()).unwrap_or("unknown");
                            info!(node_name = %name, "[WS_TUNNEL] ✅ Auth OK");
                            return Ok(());
                        }
                        Some("error") => {
                            let msg = parsed.get("message").and_then(|m| m.as_str()).unwrap_or("unknown error");
                            return Err(format!("Auth rejected: {}", msg));
                        }
                        other => debug!(msg_type = ?other, "[WS_TUNNEL] Unexpected message during auth"),
                    }
                }
                WsMessage::Close(frame) => {
                    let reason = frame.map(|f| format!("code={}, reason={}", f.code, f.reason))
                        .unwrap_or_else(|| "no frame".to_string());
                    return Err(format!("Connection closed during auth: {}", reason));
                }
                _ => {}
            }
        }
        Err("Connection closed before auth_ok".to_string())
    }

    // ============================================
    // Message Loop
    // ============================================

    async fn message_loop(
        &self, sink: &mut WsSink, source: &mut WsSource,
        shutdown: &mut broadcast::Receiver<()>,
    ) -> Result<ShutdownReason, String> {
        let mut ping_interval = tokio::time::interval(Duration::from_secs(PING_INTERVAL_SECS));
        ping_interval.tick().await;

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    let _ = sink.close().await;
                    return Ok(ShutdownReason::GracefulShutdown);
                }
                _ = ping_interval.tick() => {
                    let ping_msg = serde_json::json!({"type": "ping"});
                    if let Err(e) = ws_send_flush(sink, ws_text(&ping_msg)).await {
                        return Ok(ShutdownReason::Disconnected(format!("Ping failed: {}", e)));
                    }
                }
                msg = source.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            if let Err(e) = self.handle_cms_message(&text, sink).await {
                                warn!(error = %e, "[WS_TUNNEL] ⚠️ Error handling CMS message");
                            }
                        }
                        Some(Ok(WsMessage::Ping(data))) => { let _ = sink.send(WsMessage::Pong(data)).await; }
                        Some(Ok(WsMessage::Close(frame))) => {
                            let reason = frame.map(|f| format!("code={}, reason={}", f.code, f.reason))
                                .unwrap_or_else(|| "no frame".to_string());
                            return Ok(ShutdownReason::Disconnected(reason));
                        }
                        Some(Err(e)) => return Ok(ShutdownReason::Disconnected(format!("WS read error: {}", e))),
                        None => return Ok(ShutdownReason::Disconnected("Stream ended".to_string())),
                        _ => {}
                    }
                }
            }
        }
    }

    async fn handle_cms_message(&self, text: &str, sink: &mut WsSink) -> Result<(), String> {
        let parsed: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| format!("Invalid JSON: {}", e))?;

        let msg_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match msg_type {
            "agent_request" => self.handle_agent_request(&parsed, sink).await?,

            // ── E2E handshake (v2.2.0) ─────────────────────────────
            "e2e_init" => self.handle_e2e_init(&parsed, sink).await?,

            // ── E2E encrypted message (v2.2.0) ─────────────────────
            "e2e_message" => self.handle_e2e_message(&parsed, sink).await?,

            "pong" => debug!("[WS_TUNNEL] Pong received"),
            "ping" => {
                let pong = serde_json::json!({"type": "pong"});
                ws_send_flush(sink, ws_text(&pong)).await
                    .map_err(|e| format!("Pong send failed: {}", e))?;
            }
            other => debug!(msg_type = %other, "[WS_TUNNEL] Unhandled message type"),
        }
        Ok(())
    }

    // ============================================
    // Agent Request Handling
    // ============================================

    async fn handle_agent_request(
        &self, request: &serde_json::Value, cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let request_id = request.get("request_id").and_then(|r| r.as_str()).unwrap_or("unknown");
        let action = request.get("action").and_then(|a| a.as_str()).unwrap_or("unknown");
        let payload = request.get("payload").cloned().unwrap_or(serde_json::Value::Null);

        info!(request_id = %request_id, action = %action, "[WS_TUNNEL] 📥 Agent request");

        // ── v2.2.0: MPI proxy ──────────────────────────────────────
        // v2.3.0: Extract auth_headers from the WS message for remote user passthrough
        if action.starts_with("mpi_") {
            let auth_headers = request.get("auth_headers");
            let response = self.handle_mpi_proxy(action, request_id, &payload, auth_headers).await;
            ws_send_flush(cms_sink, ws_text(&response)).await
                .map_err(|e| format!("MPI proxy response send failed: {}", e))?;
            return Ok(());
        }

        // ── Original: OpenClaw actions ─────────────────────────────
        let gateway_token = match self.agent_manager.gateway_token().await {
            Some(token) => token,
            None => {
                self.send_done_and_error(request_id,
                    "OpenClaw gateway token not available. Is OpenClaw installed?", cms_sink).await;
                return Ok(());
            }
        };

        match action {
            "chat" => {
                if let Err(e) = self.handle_chat_request(request_id, &payload, &gateway_token, cms_sink).await {
                    warn!(request_id = %request_id, error = %e, "[WS_TUNNEL] ❌ Chat request failed");
                    self.send_done_and_error(request_id,
                        &format!("Failed to communicate with OpenClaw: {}", e), cms_sink).await;
                }
            }
            "status" => {
                let status = self.agent_manager.status().await;
                let response = serde_json::json!({
                    "type": "agent_response", "request_id": request_id,
                    "status": "success",
                    "payload": serde_json::to_value(&status).unwrap_or(serde_json::Value::Null)
                });
                ws_send_flush(cms_sink, ws_text(&response)).await
                    .map_err(|e| format!("Status response send failed: {}", e))?;
            }
            other => {
                let response = serde_json::json!({
                    "type": "agent_response", "request_id": request_id,
                    "status": "error",
                    "payload": {"error": format!("Unknown action: {}", other)}
                });
                ws_send_flush(cms_sink, ws_text(&response)).await
                    .map_err(|e| format!("Error response send failed: {}", e))?;
            }
        }
        Ok(())
    }

    // ============================================
    // E2E Handshake + Encrypted Chat (v2.2.0)
    // ============================================

    /// Handle `e2e_init` — frontend sends its ephemeral X25519 public key.
    ///
    /// Flow:
    /// 1. Parse frontend's ephemeral_pk (32 bytes hex)
    /// 2. Convert node's Ed25519 identity → X25519 key pair
    /// 3. Compute shared_secret = X25519(node_sk, frontend_pk)
    /// 4. Store E2eSession for this WebSocket connection
    /// 5. Send `e2e_ready` with node's X25519 public key
    ///
    /// Frontend then computes the same shared_secret on its side.
    async fn handle_e2e_init(
        &self, msg: &serde_json::Value, sink: &mut WsSink,
    ) -> Result<(), String> {
        let ephemeral_pk_hex = msg.get("ephemeral_pk")
            .and_then(|v| v.as_str())
            .ok_or("e2e_init: missing ephemeral_pk")?;

        let pk_bytes = hex::decode(ephemeral_pk_hex)
            .map_err(|e| format!("e2e_init: invalid ephemeral_pk hex: {}", e))?;

        if pk_bytes.len() != 32 {
            return Err(format!("e2e_init: ephemeral_pk must be 32 bytes, got {}", pk_bytes.len()));
        }

        let mut frontend_pk_arr = [0u8; 32];
        frontend_pk_arr.copy_from_slice(&pk_bytes);

        // Single call does: Ed25519→X25519 conversion + DH + E2eSession creation
        let (session, node_x25519_pk_bytes) = self.identity.e2e_handshake(&frontend_pk_arr);

        // Store E2E session
        {
            let mut guard = self.e2e_session.lock()
                .map_err(|e| format!("E2E session lock: {}", e))?;
            *guard = Some(session);
        }

        // Send e2e_ready with node's X25519 public key
        let ready_msg = serde_json::json!({
            "type": "e2e_ready",
            "x25519_pk": hex::encode(node_x25519_pk_bytes),
        });

        ws_send_flush(sink, ws_text(&ready_msg)).await
            .map_err(|e| format!("e2e_ready send failed: {}", e))?;

        info!(
            frontend_pk = %&ephemeral_pk_hex[..ephemeral_pk_hex.len().min(8)],
            "[WS_E2E] ✅ E2E session established"
        );
        Ok(())
    }

    /// Handle `e2e_message` — decrypt, process, encrypt response.
    ///
    /// Flow:
    /// 1. Decrypt the incoming message using E2eSession
    /// 2. Parse the plaintext JSON to get action + payload
    /// 3. Dispatch to the appropriate handler (chat, MPI, etc.)
    /// 4. Encrypt the response and send as `e2e_response` or `e2e_stream`
    async fn handle_e2e_message(
        &self, msg: &serde_json::Value, sink: &mut WsSink,
    ) -> Result<(), String> {
        let request_id = msg.get("request_id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let nonce = msg.get("nonce").and_then(|v| v.as_str())
            .ok_or("e2e_message: missing nonce")?;
        let ciphertext = msg.get("ciphertext").and_then(|v| v.as_str())
            .ok_or("e2e_message: missing ciphertext")?;
        let action = msg.get("action").and_then(|v| v.as_str()).unwrap_or("chat");

        // Decrypt
        let plaintext_bytes = {
            let guard = self.e2e_session.lock()
                .map_err(|e| format!("E2E session lock: {}", e))?;
            let session = guard.as_ref()
                .ok_or("e2e_message received but no E2E session (send e2e_init first)")?;
            session.decrypt(nonce, ciphertext)
                .map_err(|e| format!("E2E decryption failed: {}", e))?
        };

        let plaintext = String::from_utf8(plaintext_bytes)
            .map_err(|e| format!("E2E decrypted content is not valid UTF-8: {}", e))?;

        info!(
            request_id = %request_id,
            action = %action,
            len = plaintext.len(),
            "[WS_E2E] 📥 Decrypted message"
        );

        // Parse the decrypted plaintext as JSON payload
        let payload: serde_json::Value = serde_json::from_str(&plaintext)
            .unwrap_or_else(|_| {
                // If not JSON, treat as a chat prompt string
                serde_json::json!({"prompt": plaintext})
            });

        // Dispatch based on action
        if action.starts_with("mpi_") {
            // MPI proxy — same logic as non-E2E, but encrypt the response
            // E2E MPI requests don't have auth_headers (they go through E2E encryption)
            let mpi_response = self.handle_mpi_proxy(action, request_id, &payload, None).await;
            let response_str = serde_json::to_string(&mpi_response.get("payload")
                .unwrap_or(&serde_json::Value::Null))
                .unwrap_or_default();
            self.send_e2e_response(request_id, &response_str, sink).await?;
        } else if action == "chat" {
            // Chat — decrypt prompt, send to OpenClaw, encrypt streaming response
            self.handle_e2e_chat(request_id, &payload, sink).await?;
        } else {
            let error_msg = format!("Unknown E2E action: {}", action);
            self.send_e2e_response(request_id, &serde_json::json!({"error": error_msg}).to_string(), sink).await?;
        }

        Ok(())
    }

    /// Handle E2E chat: decrypt prompt → OpenClaw → encrypt streaming response.
    async fn handle_e2e_chat(
        &self, request_id: &str, payload: &serde_json::Value, sink: &mut WsSink,
    ) -> Result<(), String> {
        let prompt = payload.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
        if prompt.is_empty() {
            return self.send_e2e_response(
                request_id,
                &serde_json::json!({"error": "empty prompt"}).to_string(),
                sink,
            ).await;
        }

        let gateway_token = match self.agent_manager.gateway_token().await {
            Some(t) => t,
            None => {
                return self.send_e2e_response(
                    request_id,
                    &serde_json::json!({"error": "OpenClaw gateway not available"}).to_string(),
                    sink,
                ).await;
            }
        };

        let session_user = if let Some(uid) = payload.get("user_id").and_then(|u| u.as_str()) {
            format!("aeronyx:user:{}", uid)
        } else {
            format!("aeronyx:req:{}", request_id)
        };

        let request_body = serde_json::json!({
            "model": "openclaw", "stream": true, "user": session_user,
            "messages": [{"role": "user", "content": prompt}]
        });

        let url = format!("{}{}", OPENCLAW_HTTP_BASE, OPENCLAW_CHAT_PATH);

        let response = tokio::time::timeout(
            Duration::from_secs(OPENCLAW_REQUEST_TIMEOUT_SECS),
            self.http_client.post(&url)
                .header("Authorization", format!("Bearer {}", gateway_token))
                .header("Content-Type", "application/json")
                .header("x-openclaw-agent-id", OPENCLAW_AGENT_ID)
                .json(&request_body).send()
        ).await
            .map_err(|_| format!("OpenClaw timed out ({}s)", OPENCLAW_REQUEST_TIMEOUT_SECS))?
            .map_err(|e| format!("OpenClaw HTTP failed: {}", e))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            let error = format!("OpenClaw HTTP {}: {}", status_code, &body[..body.len().min(500)]);
            return self.send_e2e_response(request_id, &serde_json::json!({"error": error}).to_string(), sink).await;
        }

        // Stream SSE response with E2E encryption per chunk
        self.stream_e2e_sse_response(request_id, response, sink).await
    }

    /// Stream SSE from OpenClaw, encrypting each chunk for E2E delivery.
    async fn stream_e2e_sse_response(
        &self, request_id: &str, mut response: reqwest::Response, sink: &mut WsSink,
    ) -> Result<(), String> {
        let mut full_response = String::new();
        let mut line_buffer = String::new();
        let idle_timeout = Duration::from_secs(10);
        let tool_timeout = Duration::from_secs(45);
        let mut current_timeout = idle_timeout;

        loop {
            let chunk_result = tokio::time::timeout(current_timeout, response.chunk()).await;
            match chunk_result {
                Ok(Ok(Some(bytes))) => {
                    current_timeout = idle_timeout;
                    line_buffer.push_str(&String::from_utf8_lossy(&bytes));

                    // Process SSE lines, encrypt each content chunk
                    let got_done = self.process_e2e_sse_lines(
                        request_id, &mut line_buffer, &mut full_response, sink
                    ).await?;

                    if got_done { current_timeout = tool_timeout; }
                }
                Ok(Ok(None)) => break,
                Ok(Err(e)) => { warn!(error = %e, "[WS_E2E] SSE read error"); break; }
                Err(_) => break,
            }
        }

        if !line_buffer.is_empty() {
            if !line_buffer.ends_with('\n') { line_buffer.push('\n'); }
            let _ = self.process_e2e_sse_lines(
                request_id, &mut line_buffer, &mut full_response, sink
            ).await?;
        }

        // Send encrypted done + full response
        self.send_e2e_stream_done(request_id, sink).await?;

        if !full_response.is_empty() {
            // Store chat to MPI /log (plaintext, locally)
            let _ = self.store_chat_to_mpi(request_id, &full_response).await;
        }

        self.send_e2e_response(
            request_id,
            &serde_json::json!({"response": full_response}).to_string(),
            sink,
        ).await
    }

    /// Process SSE lines and send each chunk encrypted via E2E.
    async fn process_e2e_sse_lines(
        &self, request_id: &str, line_buffer: &mut String,
        full_response: &mut String, sink: &mut WsSink,
    ) -> Result<bool, String> {
        let mut got_done = false;

        while let Some(pos) = line_buffer.find('\n') {
            let line = line_buffer[..pos].trim_end_matches('\r').to_string();
            *line_buffer = line_buffer[pos + 1..].to_string();
            if line.is_empty() { continue; }

            let data = if let Some(s) = line.strip_prefix("data: ") { s.trim() }
                else if let Some(s) = line.strip_prefix("data:") { s.trim() }
                else { continue };

            if data == "[DONE]" { got_done = true; continue; }

            let parsed: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v, Err(_) => continue,
            };

            let content = parsed.get("choices")
                .and_then(|c| c.as_array()).and_then(|a| a.first())
                .and_then(|ch| ch.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str()).unwrap_or("");

            if !content.is_empty() {
                full_response.push_str(content);

                // Encrypt the chunk
                let (nonce_hex, ct_hex) = {
                    let guard = self.e2e_session.lock()
                        .map_err(|e| format!("E2E lock: {}", e))?;
                    let session = guard.as_ref()
                        .ok_or("E2E session lost during streaming")?;
                    session.encrypt(content.as_bytes())
                        .map_err(|e| format!("E2E encrypt chunk: {}", e))?
                };

                let stream_msg = serde_json::json!({
                    "type": "e2e_stream",
                    "request_id": request_id,
                    "nonce": nonce_hex,
                    "ciphertext": ct_hex,
                    "done": false,
                });
                ws_send_flush(sink, ws_text(&stream_msg)).await
                    .map_err(|e| format!("E2E stream send: {}", e))?;
            }
        }
        Ok(got_done)
    }

    /// Send encrypted stream-done signal.
    async fn send_e2e_stream_done(
        &self, request_id: &str, sink: &mut WsSink,
    ) -> Result<(), String> {
        let (nonce_hex, ct_hex) = {
            let guard = self.e2e_session.lock()
                .map_err(|e| format!("E2E lock: {}", e))?;
            let session = guard.as_ref().ok_or("E2E session lost")?;
            session.encrypt(b"").map_err(|e| format!("E2E encrypt done: {}", e))?
        };
        let msg = serde_json::json!({
            "type": "e2e_stream",
            "request_id": request_id,
            "nonce": nonce_hex,
            "ciphertext": ct_hex,
            "done": true,
        });
        ws_send_flush(sink, ws_text(&msg)).await
    }

    /// Send an encrypted response (non-streaming).
    async fn send_e2e_response(
        &self, request_id: &str, plaintext: &str, sink: &mut WsSink,
    ) -> Result<(), String> {
        let (nonce_hex, ct_hex) = {
            let guard = self.e2e_session.lock()
                .map_err(|e| format!("E2E lock: {}", e))?;
            let session = guard.as_ref()
                .ok_or("E2E session not established")?;
            session.encrypt(plaintext.as_bytes())
                .map_err(|e| format!("E2E encrypt response: {}", e))?
        };
        let msg = serde_json::json!({
            "type": "e2e_response",
            "request_id": request_id,
            "status": "success",
            "nonce": nonce_hex,
            "ciphertext": ct_hex,
        });
        ws_send_flush(sink, ws_text(&msg)).await
    }

    /// Store chat turn to local MPI /log (plaintext, for memory extraction).
    ///
    /// v2.3.0: Now injects Bearer token when api_secret is configured.
    /// This is a local-only call (store_chat always uses the node's own identity).
    async fn store_chat_to_mpi(&self, request_id: &str, response: &str) -> Result<(), String> {
        let url = format!("{}/api/mpi/log", MPI_BASE_URL);
        let body = serde_json::json!({
            "session_id": format!("e2e-{}", request_id),
            "turns": [
                {"role": "assistant", "content": response}
            ],
            "source_ai": "openclaw-e2e"
        });

        let mut req = self.http_client.post(&url)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(5))
            .json(&body);

        // v2.3.0: Inject Bearer token for local MPI auth
        if let Some(ref secret) = self.mpi_api_secret {
            req = req.header("Authorization", format!("Bearer {}", secret));
        }

        let _ = req.send().await;
        Ok(())
    }

    // ============================================
    // MPI Proxy (v2.2.0 + v2.3.0 auth_headers)
    // ============================================

    /// Handle MPI proxy requests from CMS WebSocket.
    ///
    /// Maps `action: "mpi_*"` to local HTTP requests to the MPI API,
    /// then wraps the response in `agent_response` format for the frontend.
    ///
    /// ## v2.3.0: auth_headers passthrough
    /// When the WS message contains `auth_headers` (a JSON object with
    /// `X-MemChain-PublicKey`, `X-MemChain-Timestamp`, `X-MemChain-Signature`),
    /// these are injected into the HTTP request to MPI. This enables remote
    /// users (authenticated via CMS) to have their Ed25519 signatures verified
    /// by the MPI unified_auth_middleware.
    ///
    /// When `auth_headers` is absent (e.g., MemExplorer frontend requests),
    /// the Bearer token is injected if `mpi_api_secret` is configured.
    ///
    /// ## Security
    /// - Only actions in MPI_ALLOWED_ACTIONS are forwarded
    /// - `mpi_log` is deliberately excluded (local-only in v2.3.0)
    /// - Requests go to localhost only (no external network access)
    /// - 10s timeout prevents MPI hangs from blocking the WebSocket
    /// - auth_headers are opaque to the proxy — validation happens in MPI middleware
    async fn handle_mpi_proxy(
        &self,
        action: &str,
        request_id: &str,
        payload: &serde_json::Value,
        auth_headers: Option<&serde_json::Value>,
    ) -> serde_json::Value {
        // 1. Whitelist check
        if !MPI_ALLOWED_ACTIONS.contains(&action) {
            warn!(action = %action, "[WS_MPI] ⛔ Action not in whitelist");
            return serde_json::json!({
                "type": "agent_response",
                "request_id": request_id,
                "status": "error",
                "payload": {"error": format!("action '{}' not allowed", action)}
            });
        }

        // 2. Map action → HTTP method + path
        let (method, path) = match action {
            "mpi_status"   => ("GET",  "/api/mpi/status".to_string()),
            "mpi_recall"   => ("POST", "/api/mpi/recall".to_string()),
            "mpi_remember" => ("POST", "/api/mpi/remember".to_string()),
            "mpi_forget"   => ("POST", "/api/mpi/forget".to_string()),
            "mpi_embed"    => ("POST", "/api/mpi/embed".to_string()),
            "mpi_overview" => ("GET",  "/api/mpi/records/overview".to_string()),
            "mpi_record"   => {
                let record_id = payload.get("record_id").and_then(|v| v.as_str()).unwrap_or("");
                if record_id.is_empty() {
                    return serde_json::json!({
                        "type": "agent_response",
                        "request_id": request_id,
                        "status": "error",
                        "payload": {"error": "missing record_id in payload"}
                    });
                }
                ("GET", format!("/api/mpi/record/{}", record_id))
            }
            _ => {
                return serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "payload": {"error": format!("unknown mpi action: {}", action)}
                });
            }
        };

        let url = format!("{}{}", MPI_BASE_URL, path);

        // 3. Determine auth mode: remote (auth_headers) or local (Bearer token)
        let has_remote_auth = auth_headers
            .and_then(|h| h.as_object())
            .map_or(false, |h| h.contains_key(HEADER_MEMCHAIN_SIGNATURE));

        if has_remote_auth {
            debug!(
                action = %action, url = %url, method = %method,
                "[WS_MPI] Proxying with remote auth (Ed25519 passthrough)"
            );
        } else {
            debug!(action = %action, url = %url, method = %method, "[WS_MPI] Proxying with local auth");
        }

        // 4. Build HTTP request
        // v2.3.0: Reuse self.http_client (was creating new client per request — bug fix)
        let mut req_builder = match method {
            "GET" => self.http_client.get(&url),
            "POST" => self.http_client.post(&url)
                .header("Content-Type", "application/json")
                .json(payload),
            _ => unreachable!(),
        };

        // 5. Inject auth headers
        if has_remote_auth {
            // v2.3.0: Passthrough Ed25519 signature headers from CMS
            if let Some(headers_obj) = auth_headers.and_then(|h| h.as_object()) {
                if let Some(pk) = headers_obj.get(HEADER_MEMCHAIN_PUBKEY).and_then(|v| v.as_str()) {
                    req_builder = req_builder.header(HEADER_MEMCHAIN_PUBKEY, pk);
                }
                if let Some(ts) = headers_obj.get(HEADER_MEMCHAIN_TIMESTAMP).and_then(|v| v.as_str()) {
                    req_builder = req_builder.header(HEADER_MEMCHAIN_TIMESTAMP, ts);
                }
                if let Some(sig) = headers_obj.get(HEADER_MEMCHAIN_SIGNATURE).and_then(|v| v.as_str()) {
                    req_builder = req_builder.header(HEADER_MEMCHAIN_SIGNATURE, sig);
                }
            }
        } else if let Some(ref secret) = self.mpi_api_secret {
            // v2.3.0: Inject Bearer token for local auth (MemExplorer, E2E, etc.)
            req_builder = req_builder.header("Authorization", format!("Bearer {}", secret));
        }

        // 6. Send and wrap response
        let result = req_builder.send().await;

        match result {
            Ok(resp) => {
                let status_code = resp.status().as_u16();
                let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);

                debug!(
                    action = %action,
                    status_code = status_code,
                    "[WS_MPI] MPI response received"
                );

                serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": if status_code < 400 { "success" } else { "error" },
                    "status_code": status_code,
                    "payload": body
                })
            }
            Err(e) => {
                let is_connect_error = e.is_connect();
                warn!(action = %action, error = %e, "[WS_MPI] ❌ MPI request failed");

                serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "status_code": if is_connect_error { 503 } else { 500 },
                    "payload": {
                        "error": if is_connect_error {
                            "MemChain MPI is not available on this node".to_string()
                        } else {
                            format!("MPI request failed: {}", e)
                        }
                    }
                })
            }
        }
    }

    // ============================================
    // Done + Error/Response Helpers
    // ============================================

    async fn send_done_and_error(&self, request_id: &str, error_message: &str, cms_sink: &mut WsSink) {
        let done_msg = serde_json::json!({
            "type": "agent_stream", "request_id": request_id, "chunk": "", "done": true
        });
        let _ = ws_send_flush(cms_sink, ws_text(&done_msg)).await;

        let error_response = serde_json::json!({
            "type": "agent_response", "request_id": request_id,
            "status": "error", "payload": {"error": error_message}
        });
        let _ = ws_send_flush(cms_sink, ws_text(&error_response)).await;
    }

    async fn send_done_and_response(
        &self, request_id: &str, full_response: &str, cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let done_msg = serde_json::json!({
            "type": "agent_stream", "request_id": request_id, "chunk": "", "done": true
        });
        info!(request_id = %request_id, "[WS_TUNNEL] Sending done=true...");
        ws_send_flush(cms_sink, ws_text(&done_msg)).await
            .map_err(|e| { error!(request_id = %request_id, error = %e, "[WS_TUNNEL] ❌ Failed done=true"); e })?;
        info!(request_id = %request_id, "[WS_TUNNEL] ✅ done=true SENT+FLUSHED");

        if !full_response.is_empty() {
            let final_msg = serde_json::json!({
                "type": "agent_response", "request_id": request_id,
                "status": "success", "payload": {"response": full_response}
            });
            info!(request_id = %request_id, response_len = full_response.len(), "[WS_TUNNEL] Sending agent_response...");
            ws_send_flush(cms_sink, ws_text(&final_msg)).await
                .map_err(|e| { error!(request_id = %request_id, error = %e, "[WS_TUNNEL] ❌ Failed agent_response"); e })?;
            info!(request_id = %request_id, "[WS_TUNNEL] ✅ agent_response SENT+FLUSHED");
        }
        Ok(())
    }

    // ============================================
    // Chat Request: HTTP SSE
    // ============================================

    async fn handle_chat_request(
        &self, request_id: &str, payload: &serde_json::Value,
        gateway_token: &str, cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let prompt = payload.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
        if prompt.is_empty() {
            return Err("Empty prompt in chat request".to_string());
        }

        let session_user = if let Some(uid) = payload.get("user_id").and_then(|u| u.as_str()) {
            format!("aeronyx:user:{}", uid)
        } else {
            format!("aeronyx:req:{}", request_id)
        };

        let request_body = serde_json::json!({
            "model": "openclaw", "stream": true, "user": session_user,
            "messages": [{"role": "user", "content": prompt}]
        });

        let url = format!("{}{}", OPENCLAW_HTTP_BASE, OPENCLAW_CHAT_PATH);
        info!(request_id = %request_id, url = %url, session = %session_user, "[WS_TUNNEL] 📤 Chat → OpenClaw");

        let response = tokio::time::timeout(
            Duration::from_secs(OPENCLAW_REQUEST_TIMEOUT_SECS),
            self.http_client.post(&url)
                .header("Authorization", format!("Bearer {}", gateway_token))
                .header("Content-Type", "application/json")
                .header("x-openclaw-agent-id", OPENCLAW_AGENT_ID)
                .json(&request_body).send()
        ).await
            .map_err(|_| format!("OpenClaw timed out ({}s)", OPENCLAW_REQUEST_TIMEOUT_SECS))?
            .map_err(|e| format!("OpenClaw HTTP failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("OpenClaw HTTP {}: {}", status.as_u16(), body.chars().take(500).collect::<String>()));
        }

        info!(request_id = %request_id, http_status = %status.as_u16(), "[WS_TUNNEL] ✅ OpenClaw responded, starting SSE");
        self.stream_sse_response(request_id, response, cms_sink).await
    }

    // ============================================
    // SSE Stream Processing
    // ============================================

    async fn stream_sse_response(
        &self, request_id: &str, mut response: reqwest::Response, cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let mut full_response = String::new();
        let mut line_buffer = String::new();

        let idle_timeout = Duration::from_secs(10);
        let tool_timeout = Duration::from_secs(45);
        let mut current_timeout = idle_timeout;

        loop {
            let chunk_result = tokio::time::timeout(current_timeout, response.chunk()).await;

            match chunk_result {
                Ok(Ok(Some(bytes))) => {
                    current_timeout = idle_timeout;
                    line_buffer.push_str(&String::from_utf8_lossy(&bytes));
                    let got_done = self.process_sse_lines(
                        request_id, &mut line_buffer, &mut full_response, cms_sink
                    ).await?;
                    if got_done {
                        current_timeout = tool_timeout;
                        info!(request_id = %request_id, "[WS_TUNNEL] Extended timeout for tool execution");
                    }
                }
                Ok(Ok(None)) => {
                    info!(request_id = %request_id, residual_len = line_buffer.len(), "[WS_TUNNEL] SSE body exhausted");
                    break;
                }
                Ok(Err(e)) => { warn!(request_id = %request_id, error = %e, "[WS_TUNNEL] SSE read error"); break; }
                Err(_) => {
                    info!(request_id = %request_id, timeout_secs = current_timeout.as_secs(),
                        response_len = full_response.len(), "[WS_TUNNEL] SSE idle timeout");
                    break;
                }
            }
        }

        if !line_buffer.is_empty() {
            if !line_buffer.ends_with('\n') { line_buffer.push('\n'); }
            let _ = self.process_sse_lines(request_id, &mut line_buffer, &mut full_response, cms_sink).await?;
        }

        info!(request_id = %request_id, response_len = full_response.len(), "[WS_TUNNEL] ✅ Stream ended");
        self.send_done_and_response(request_id, &full_response, cms_sink).await
    }

    async fn process_sse_lines(
        &self, request_id: &str, line_buffer: &mut String,
        full_response: &mut String, cms_sink: &mut WsSink,
    ) -> Result<bool, String> {
        let mut got_done_marker = false;

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            *line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.is_empty() { continue; }

            let data = if let Some(s) = line.strip_prefix("data: ") { s.trim() }
                else if let Some(s) = line.strip_prefix("data:") { s.trim() }
                else { continue };

            if data == "[DONE]" {
                info!(request_id = %request_id, "[WS_TUNNEL] SSE [DONE]");
                got_done_marker = true;
                continue;
            }

            let parsed: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let content = parsed.get("choices")
                .and_then(|c| c.as_array()).and_then(|arr| arr.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(|c| c.as_str()).unwrap_or("");

            if !content.is_empty() {
                full_response.push_str(content);
                let stream_msg = serde_json::json!({
                    "type": "agent_stream", "request_id": request_id,
                    "chunk": content, "done": false
                });
                ws_send_flush(cms_sink, ws_text(&stream_msg)).await
                    .map_err(|e| format!("Stream send failed: {}", e))?;
            }
        }
        Ok(got_done_marker)
    }
}

// ============================================
// Shutdown Reason
// ============================================

enum ShutdownReason {
    GracefulShutdown,
    AuthFailed(String),
    Disconnected(String),
}
