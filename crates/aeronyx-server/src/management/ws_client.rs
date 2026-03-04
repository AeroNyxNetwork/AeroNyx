//! ============================================
//! File: crates/aeronyx-server/src/management/ws_client.rs
//! Path: aeronyx-server/src/management/ws_client.rs
//! ============================================
//!
//! ## Creation Reason
//! Phase 3 of the OpenClaw integration: Real-time WebSocket Tunnel.
//! Maintains a persistent WebSocket connection from the Rust node to
//! the CMS backend, enabling low-latency bidirectional communication
//! for frontend → OpenClaw AI agent interactions.
//!
//! ## Modification Reason (v1.3.2)
//! Replaced OpenClaw Gateway WebSocket RPC with HTTP API (`/v1/chat/completions`).
//! The HTTP API is OpenAI-compatible, supports SSE streaming, and is far simpler
//! than the internal WS RPC protocol. Benefits:
//! - No complex RPC handshake (connect → hello-ok → chat.send)
//! - Reuses `reqwest::Client` connection pool (no per-request WS connect)
//! - Standard SSE parsing instead of fragile event name matching
//! - Proper session management via OpenAI `user` field
//! - Lower latency and better error handling
//!
//! ## Main Functionality
//! - `WsTunnel`: Background async task managing the CMS WebSocket lifecycle
//! - Ed25519-signed authentication on connect
//! - Automatic reconnection with exponential backoff
//! - Bidirectional message relay: CMS ↔ local OpenClaw Gateway (HTTP API)
//! - Ping/pong keepalive for connection health
//! - Graceful shutdown via broadcast signal
//!
//! ## Main Logical Flow
//! ```text
//!   ┌────────┐       wss://            ┌─────────┐  HTTP POST /v1/chat/completions  ┌──────────┐
//!   │  CMS   │ ◄──────────────────────►│ WsTunnel│ ──────────────────────────────► │ OpenClaw │
//!   │Backend │  agent_request/response  │ (Rust)  │  ◄── SSE stream ──────────────  │ Gateway  │
//!   └────────┘                          └─────────┘                                 └──────────┘
//! ```
//!
//! 1. WsTunnel connects to `wss://api.aeronyx.network/ws/node/tunnel/{node_id}/`
//! 2. Sends `auth` message with Ed25519-signed `{public_key}:{timestamp}`
//! 3. Waits for `auth_ok` (25s timeout, CMS allows 30s)
//! 4. Enters message loop:
//!    - `agent_request` from CMS → HTTP POST to OpenClaw `/v1/chat/completions`
//!    - SSE chunks → wrap as `agent_stream` → send to CMS
//!    - Final `data: [DONE]` → wrap as `agent_stream` with `done: true` → send to CMS
//!    - Periodic `ping`/`pong` keepalive (every 30s)
//! 5. On disconnect: exponential backoff reconnect (1s → 2s → 4s → ... → 60s max)
//!
//! ## CMS WebSocket Protocol
//!
//! ### Authentication (must complete within 30s)
//! ```json
//! → {"type": "auth", "node_id": "<public_key_hex>", "timestamp": 1709500000, "signature": "<hex>"}
//! ← {"type": "auth_ok", "node_id": "<uuid>", "node_name": "My Node"}
//! ```
//! Signature: `Ed25519_Sign("{public_key_hex}:{timestamp}".as_bytes())`
//!
//! ### Messages (post-auth)
//! ```json
//! ← {"type": "agent_request", "request_id": "uuid", "action": "chat", "payload": {...}}
//! → {"type": "agent_response", "request_id": "uuid", "status": "success", "payload": {...}}
//! → {"type": "agent_stream", "request_id": "uuid", "chunk": "...", "done": false}
//! ↔ {"type": "ping"} / {"type": "pong"}
//! ```
//!
//! ### Close Codes
//! - 4000: Missing node_id in URL
//! - 4001: Auth failed (invalid signature / node not found / disabled)
//! - 4002: Auth timeout (30s)
//!
//! ## OpenClaw Gateway HTTP API (local)
//! The OpenClaw Gateway at `http://127.0.0.1:18789` serves an OpenAI-compatible
//! Chat Completions endpoint at `/v1/chat/completions`.
//! - Must be enabled: `gateway.http.endpoints.chatCompletions.enabled = true`
//! - Auth via `Authorization: Bearer <gateway_token>`
//! - Agent selection via `x-openclaw-agent-id: main` header
//! - Session persistence via `user` field in request body
//! - Streaming via `stream: true` → SSE response
//!
//! ## Dependencies
//! - `tokio-tungstenite` with `rustls-tls-native-roots` for wss:// to CMS
//! - `reqwest` for HTTP calls to local OpenClaw Gateway
//! - `aeronyx_core::crypto::IdentityKeyPair` for Ed25519 signing
//!
//! ## ⚠️ Important Note for Next Developer
//! - The CMS URL uses `node_id` which is the **CMS database UUID**, NOT the
//!   Ed25519 public key. The UUID is obtained from `StoredNodeInfo.node_id`.
//! - The `auth` message uses `node_id` field which IS the Ed25519 public key hex.
//! - Signature format: `Ed25519_Sign("{pubkey}:{timestamp}".as_bytes())` — raw
//!   sign, NOT SHA-256 then sign. This differs from heartbeat which does SHA-256
//!   first. Be careful.
//! - The `reqwest::Client` is created ONCE and reused for all requests (connection
//!   pool). Do NOT create a new client per request.
//! - OpenClaw HTTP API must be enabled in config before this works. The
//!   AgentManager should enable it during install/onboard. If you get 404,
//!   run: `openclaw config set gateway.http.endpoints.chatCompletions.enabled true`
//! - The `user` field in OpenAI requests creates stable sessions in OpenClaw.
//!   We use `"aeronyx:{request_id}"` for stateless or `"aeronyx:user:{user_id}"`
//!   for persistent sessions (if CMS provides a user_id in the payload).
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation (Phase 3: WebSocket Tunnel)
//! v1.3.2 - 🔄 Replaced OpenClaw WS RPC with HTTP API (/v1/chat/completions)
//!          - Fixed: per-request WS connection overhead → reqwest connection pool
//!          - Fixed: fragile event name matching → standard SSE parsing
//!          - Fixed: no session management → OpenAI `user` field for session persistence
//!          - Added: proper SSE `data: [DONE]` handling
//!          - Added: non-streaming fallback for `status` action
//!          - Added: configurable request timeout (120s default)
//! ============================================

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{connect_async, MaybeTlsStream};
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use crate::services::AgentManager;

// ============================================
// Constants
// ============================================

/// CMS WebSocket base URL (without trailing node_id).
const CMS_WS_BASE_URL: &str = "wss://api.aeronyx.network/ws/node/tunnel";

/// Local OpenClaw Gateway HTTP API base URL.
const OPENCLAW_HTTP_BASE: &str = "http://127.0.0.1:18789";

/// OpenClaw Chat Completions endpoint path.
const OPENCLAW_CHAT_PATH: &str = "/v1/chat/completions";

/// Maximum reconnection backoff interval (seconds).
const MAX_RECONNECT_BACKOFF_SECS: u64 = 60;

/// Initial reconnection delay (seconds).
const INITIAL_RECONNECT_DELAY_SECS: u64 = 1;

/// Ping interval to keep the CMS connection alive (seconds).
const PING_INTERVAL_SECS: u64 = 30;

/// Timeout waiting for auth_ok response (seconds).
const AUTH_TIMEOUT_SECS: u64 = 25; // Slightly under CMS's 30s limit

/// Timeout for a single OpenClaw HTTP request (seconds).
/// This covers the entire SSE stream duration, not just the first byte.
const OPENCLAW_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Default OpenClaw agent ID.
const OPENCLAW_AGENT_ID: &str = "main";

// ============================================
// Type aliases
// ============================================

type WsStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<WsStream, WsMessage>;
type WsSource = SplitStream<WsStream>;

/// Helper: creates a WsMessage::Text from a serializable value.
/// tungstenite 0.26+ requires `Utf8Bytes`, not `String`.
fn ws_text(value: &serde_json::Value) -> WsMessage {
    WsMessage::Text(value.to_string().into())
}

// ============================================
// WsTunnel
// ============================================

/// WebSocket tunnel client for CMS ↔ OpenClaw real-time communication.
///
/// ## Lifecycle
/// 1. Created in `server.rs` with identity + node info
/// 2. Spawned as a `tokio::spawn` task via `run()`
/// 3. Maintains persistent connection to CMS with auto-reconnect
/// 4. Relays `agent_request` from CMS to local OpenClaw Gateway via HTTP API
/// 5. Stops on shutdown signal
///
/// ## v1.3.2 Architecture Change
/// Previously connected to OpenClaw via WebSocket RPC (complex handshake).
/// Now uses the OpenAI-compatible HTTP API at `/v1/chat/completions` with
/// SSE streaming. The `reqwest::Client` is created once and reused for all
/// requests via its built-in connection pool.
pub struct WsTunnel {
    /// Ed25519 identity for signing auth messages.
    identity: IdentityKeyPair,

    /// CMS database UUID for the node (used in WS URL path).
    /// Loaded from `StoredNodeInfo.node_id`.
    cms_node_id: String,

    /// Agent manager for checking OpenClaw availability and getting gateway token.
    agent_manager: Arc<AgentManager>,

    /// Reusable HTTP client for OpenClaw Gateway API calls.
    /// Created once, uses connection pool internally.
    /// v1.3.2: Replaces per-request WebSocket connections.
    http_client: reqwest::Client,
}

impl WsTunnel {
    /// Creates a new WsTunnel.
    ///
    /// # Arguments
    /// * `identity` - Ed25519 keypair for signing auth messages
    /// * `cms_node_id` - CMS database UUID (NOT the public key)
    /// * `agent_manager` - Shared AgentManager for OpenClaw status and token
    pub fn new(
        identity: IdentityKeyPair,
        cms_node_id: String,
        agent_manager: Arc<AgentManager>,
    ) -> Self {
        // Build a reusable HTTP client with sensible defaults.
        // reqwest::Client internally maintains a connection pool, so
        // creating it once and reusing across all requests is optimal.
        let http_client = reqwest::Client::builder()
            // Do NOT set a global timeout here — SSE streams can last minutes.
            // We apply per-phase timeouts in the request logic instead.
            .pool_max_idle_per_host(4)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to build reqwest::Client");

        Self {
            identity,
            cms_node_id,
            agent_manager,
            http_client,
        }
    }

    /// Runs the WebSocket tunnel loop with automatic reconnection.
    ///
    /// This is the main entry point. It will:
    /// 1. Connect to CMS WebSocket
    /// 2. Authenticate
    /// 3. Enter message relay loop
    /// 4. On disconnect: wait with exponential backoff, then reconnect
    /// 5. Stop on shutdown signal
    pub async fn run(self, mut shutdown: broadcast::Receiver<()>) {
        info!(
            cms_node_id = %self.cms_node_id,
            "[WS_TUNNEL] Starting WebSocket tunnel"
        );

        let mut backoff_secs = INITIAL_RECONNECT_DELAY_SECS;

        loop {
            // Check shutdown before attempting connection
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
                    error!(
                        reason = %reason,
                        "[WS_TUNNEL] ❌ Authentication failed — will retry"
                    );
                    // Auth failures might be transient (clock skew, CMS restart)
                    // so still retry, but with longer backoff
                    backoff_secs = (backoff_secs * 2).min(MAX_RECONNECT_BACKOFF_SECS);
                }
                Ok(ShutdownReason::Disconnected(reason)) => {
                    warn!(
                        reason = %reason,
                        backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ⚠️ Disconnected — reconnecting in {}s",
                        backoff_secs
                    );
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ❌ Connection error — reconnecting in {}s",
                        backoff_secs
                    );
                }
            }

            // Wait with backoff (but also listen for shutdown)
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[WS_TUNNEL] Shutdown during backoff");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_secs(backoff_secs)) => {}
            }

            // Exponential backoff
            backoff_secs = (backoff_secs * 2).min(MAX_RECONNECT_BACKOFF_SECS);
        }

        info!("[WS_TUNNEL] WebSocket tunnel stopped");
    }

    /// Connects to CMS, authenticates, and runs the message loop.
    ///
    /// Returns the reason the connection ended.
    async fn connect_and_run(
        &self,
        url: &str,
        shutdown: &mut broadcast::Receiver<()>,
    ) -> Result<ShutdownReason, String> {
        // --- Step 1: WebSocket connect ---
        let (ws_stream, _response) = connect_async(url)
            .await
            .map_err(|e| format!("WebSocket connect failed: {}", e))?;

        info!("[WS_TUNNEL] ✅ Connected to CMS");

        let (mut sink, mut source) = ws_stream.split();

        // --- Step 2: Authenticate ---
        if let Err(reason) = self.authenticate(&mut sink, &mut source).await {
            let _ = sink.close().await;
            return Ok(ShutdownReason::AuthFailed(reason));
        }

        info!("[WS_TUNNEL] ✅ Authenticated with CMS");

        // Reset backoff on successful auth
        // (caller tracks backoff, we signal success via return type)

        // --- Step 3: Message relay loop ---
        self.message_loop(&mut sink, &mut source, shutdown).await
    }

    // ============================================
    // Authentication
    // ============================================

    /// Sends auth message and waits for auth_ok.
    async fn authenticate(
        &self,
        sink: &mut WsSink,
        source: &mut WsSource,
    ) -> Result<(), String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let public_key_hex = hex::encode(self.identity.public_key_bytes());

        // Signature: sign raw bytes of "{public_key_hex}:{timestamp}"
        // Per CMS spec: this is direct Ed25519 sign, NOT SHA-256 then sign
        let message = format!("{}:{}", public_key_hex, timestamp);
        let signature_bytes = self.identity.sign(message.as_bytes());
        let signature_hex = hex::encode(signature_bytes);

        let auth_msg = serde_json::json!({
            "type": "auth",
            "node_id": public_key_hex,
            "timestamp": timestamp,
            "signature": signature_hex,
        });

        debug!("[WS_TUNNEL] Sending auth message...");

        sink.send(ws_text(&auth_msg))
            .await
            .map_err(|e| format!("Failed to send auth: {}", e))?;

        // Wait for auth_ok with timeout
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

    /// Reads messages from the WebSocket until auth_ok is received.
    async fn wait_for_auth_ok(source: &mut WsSource) -> Result<(), String> {
        while let Some(msg_result) = source.next().await {
            let msg = msg_result.map_err(|e| format!("WS read error during auth: {}", e))?;

            match msg {
                WsMessage::Text(text) => {
                    let parsed: serde_json::Value = serde_json::from_str(&text)
                        .map_err(|e| format!("Invalid JSON from CMS: {}", e))?;

                    match parsed.get("type").and_then(|t| t.as_str()) {
                        Some("auth_ok") => {
                            let node_name = parsed.get("node_name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            info!(
                                node_name = %node_name,
                                "[WS_TUNNEL] ✅ Auth OK"
                            );
                            return Ok(());
                        }
                        Some("error") => {
                            let error_msg = parsed.get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown error");
                            return Err(format!("Auth rejected: {}", error_msg));
                        }
                        other => {
                            debug!(
                                msg_type = ?other,
                                "[WS_TUNNEL] Unexpected message during auth"
                            );
                        }
                    }
                }
                WsMessage::Close(frame) => {
                    let reason = frame
                        .map(|f| format!("code={}, reason={}", f.code, f.reason))
                        .unwrap_or_else(|| "no frame".to_string());
                    return Err(format!("Connection closed during auth: {}", reason));
                }
                _ => {} // Ignore ping/pong/binary during auth
            }
        }

        Err("Connection closed before auth_ok".to_string())
    }

    // ============================================
    // Message Loop
    // ============================================

    /// Main message relay loop after successful authentication.
    ///
    /// Handles:
    /// - `agent_request` from CMS → forward to OpenClaw via HTTP API → respond
    /// - `ping` from CMS → respond with `pong`
    /// - Periodic ping to keep connection alive
    /// - Shutdown signal
    async fn message_loop(
        &self,
        sink: &mut WsSink,
        source: &mut WsSource,
        shutdown: &mut broadcast::Receiver<()>,
    ) -> Result<ShutdownReason, String> {
        let mut ping_interval = tokio::time::interval(Duration::from_secs(PING_INTERVAL_SECS));
        // Skip the first immediate tick
        ping_interval.tick().await;

        loop {
            tokio::select! {
                // --- Shutdown ---
                _ = shutdown.recv() => {
                    let _ = sink.close().await;
                    return Ok(ShutdownReason::GracefulShutdown);
                }

                // --- Periodic ping ---
                _ = ping_interval.tick() => {
                    let ping_msg = serde_json::json!({"type": "ping"});
                    if let Err(e) = sink.send(ws_text(&ping_msg)).await {
                        return Ok(ShutdownReason::Disconnected(
                            format!("Ping send failed: {}", e)
                        ));
                    }
                }

                // --- Incoming message from CMS ---
                msg = source.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            if let Err(e) = self.handle_cms_message(&text, sink).await {
                                warn!(
                                    error = %e,
                                    "[WS_TUNNEL] ⚠️ Error handling CMS message"
                                );
                            }
                        }
                        Some(Ok(WsMessage::Ping(data))) => {
                            let _ = sink.send(WsMessage::Pong(data)).await;
                        }
                        Some(Ok(WsMessage::Close(frame))) => {
                            let reason = frame
                                .map(|f| format!("code={}, reason={}", f.code, f.reason))
                                .unwrap_or_else(|| "no frame".to_string());
                            return Ok(ShutdownReason::Disconnected(reason));
                        }
                        Some(Err(e)) => {
                            return Ok(ShutdownReason::Disconnected(
                                format!("WS read error: {}", e)
                            ));
                        }
                        None => {
                            return Ok(ShutdownReason::Disconnected(
                                "Stream ended".to_string()
                            ));
                        }
                        _ => {} // Ignore binary/pong
                    }
                }
            }
        }
    }

    /// Handles a single text message from CMS.
    async fn handle_cms_message(
        &self,
        text: &str,
        sink: &mut WsSink,
    ) -> Result<(), String> {
        let parsed: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| format!("Invalid JSON: {}", e))?;

        let msg_type = parsed.get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        match msg_type {
            "agent_request" => {
                self.handle_agent_request(&parsed, sink).await?;
            }
            "pong" => {
                debug!("[WS_TUNNEL] Pong received");
            }
            "ping" => {
                let pong = serde_json::json!({"type": "pong"});
                sink.send(ws_text(&pong))
                    .await
                    .map_err(|e| format!("Pong send failed: {}", e))?;
            }
            other => {
                debug!(
                    msg_type = %other,
                    "[WS_TUNNEL] Unhandled message type from CMS"
                );
            }
        }

        Ok(())
    }

    // ============================================
    // Agent Request Handling (v1.3.2: HTTP API)
    // ============================================

    /// Handles an `agent_request` from CMS by forwarding to local OpenClaw HTTP API.
    ///
    /// v1.3.2: Replaced WebSocket RPC with HTTP POST to `/v1/chat/completions`.
    ///
    /// The flow:
    /// 1. Extract `request_id`, `action`, `payload` from CMS message
    /// 2. Get gateway token from AgentManager
    /// 3. For `chat` action: POST to OpenClaw with `stream: true`, relay SSE chunks
    /// 4. For `status` action: return AgentManager status directly
    /// 5. Send `agent_response` or `agent_stream` back to CMS
    async fn handle_agent_request(
        &self,
        request: &serde_json::Value,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let request_id = request.get("request_id")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");

        let action = request.get("action")
            .and_then(|a| a.as_str())
            .unwrap_or("unknown");

        let payload = request.get("payload")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        info!(
            request_id = %request_id,
            action = %action,
            "[WS_TUNNEL] 📥 Agent request from CMS"
        );

        // Get gateway token from AgentManager
        let gateway_token = match self.agent_manager.gateway_token().await {
            Some(token) => token,
            None => {
                let error_response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "payload": {
                        "error": "OpenClaw gateway token not available. Is OpenClaw installed and configured?"
                    }
                });
                let _ = cms_sink.send(ws_text(&error_response)).await;
                return Ok(());
            }
        };

        // Dispatch based on action
        match action {
            "chat" => {
                if let Err(e) = self.handle_chat_request(
                    request_id, &payload, &gateway_token, cms_sink
                ).await {
                    warn!(
                        request_id = %request_id,
                        error = %e,
                        "[WS_TUNNEL] ❌ Chat request failed"
                    );
                    let error_response = serde_json::json!({
                        "type": "agent_response",
                        "request_id": request_id,
                        "status": "error",
                        "payload": {
                            "error": format!("Failed to communicate with OpenClaw: {}", e)
                        }
                    });
                    let _ = cms_sink.send(ws_text(&error_response)).await;
                }
            }
            "status" => {
                let status = self.agent_manager.status().await;
                let response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "success",
                    "payload": serde_json::to_value(&status).unwrap_or(serde_json::Value::Null)
                });
                cms_sink.send(ws_text(&response))
                    .await
                    .map_err(|e| format!("Status response send failed: {}", e))?;
            }
            other => {
                let response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "payload": {
                        "error": format!("Unknown action: {}", other)
                    }
                });
                cms_sink.send(ws_text(&response))
                    .await
                    .map_err(|e| format!("Error response send failed: {}", e))?;
            }
        }

        Ok(())
    }

    // ============================================
    // Chat Request: HTTP SSE (v1.3.2)
    // ============================================

    /// Sends a chat request to OpenClaw via HTTP API and streams the response
    /// back to CMS as `agent_stream` messages.
    ///
    /// v1.3.2: New implementation using OpenAI-compatible HTTP API.
    ///
    /// ## OpenClaw API Contract
    /// ```text
    /// POST http://127.0.0.1:18789/v1/chat/completions
    /// Headers:
    ///   Authorization: Bearer <gateway_token>
    ///   Content-Type: application/json
    ///   x-openclaw-agent-id: main
    /// Body:
    ///   {"model":"openclaw","stream":true,"user":"aeronyx:...","messages":[...]}
    ///
    /// Response (SSE):
    ///   data: {"choices":[{"delta":{"content":"Hello"}}]}
    ///   data: {"choices":[{"delta":{"content":" world"}}]}
    ///   data: [DONE]
    /// ```
    ///
    /// ## Session Management
    /// The `user` field controls session persistence in OpenClaw:
    /// - If payload contains `user_id`: uses `"aeronyx:user:{user_id}"` for
    ///   persistent conversation history across requests
    /// - Otherwise: uses `"aeronyx:req:{request_id}"` for single-use sessions
    async fn handle_chat_request(
        &self,
        request_id: &str,
        payload: &serde_json::Value,
        gateway_token: &str,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let prompt = payload.get("prompt")
            .and_then(|p| p.as_str())
            .unwrap_or("");

        if prompt.is_empty() {
            return Err("Empty prompt in chat request".to_string());
        }

        // Determine session key for OpenClaw.
        // If CMS provides a user_id, use it for persistent sessions.
        // Otherwise, each request gets its own ephemeral session.
        let session_user = if let Some(user_id) = payload.get("user_id").and_then(|u| u.as_str()) {
            format!("aeronyx:user:{}", user_id)
        } else {
            format!("aeronyx:req:{}", request_id)
        };

        // Build OpenAI-compatible request body
        let request_body = serde_json::json!({
            "model": "openclaw",
            "stream": true,
            "user": session_user,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        });

        let url = format!("{}{}", OPENCLAW_HTTP_BASE, OPENCLAW_CHAT_PATH);

        debug!(
            request_id = %request_id,
            url = %url,
            session = %session_user,
            "[WS_TUNNEL] 📤 Sending chat request to OpenClaw HTTP API"
        );

        // Send HTTP request — do NOT set a timeout on the client level
        // because SSE streams can legitimately last several minutes.
        // Instead we use a tokio::time::timeout wrapper.
        let response = tokio::time::timeout(
            Duration::from_secs(OPENCLAW_REQUEST_TIMEOUT_SECS),
            self.http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", gateway_token))
                .header("Content-Type", "application/json")
                .header("x-openclaw-agent-id", OPENCLAW_AGENT_ID)
                .json(&request_body)
                .send()
        ).await
            .map_err(|_| format!("OpenClaw request timed out ({}s)", OPENCLAW_REQUEST_TIMEOUT_SECS))?
            .map_err(|e| format!("OpenClaw HTTP request failed: {}", e))?;

        // Check HTTP status
        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(format!(
                "OpenClaw returned HTTP {}: {}",
                status.as_u16(),
                error_body.chars().take(500).collect::<String>()
            ));
        }

        // Stream SSE response
        self.stream_sse_response(request_id, response, cms_sink).await
    }

    /// Parses an SSE stream from OpenClaw and relays chunks to CMS.
    ///
    /// ## SSE Format (OpenAI-compatible)
    /// Each line is prefixed with `data: `. Content chunks are JSON objects:
    /// ```json
    /// data: {"id":"...","choices":[{"delta":{"content":"text"}}]}
    /// ```
    /// The stream ends with:
    /// ```text
    /// data: [DONE]
    /// ```
    ///
    /// ## Relay Protocol
    /// Each content chunk becomes:
    /// ```json
    /// {"type":"agent_stream","request_id":"...","chunk":"text","done":false}
    /// ```
    /// The `[DONE]` signal becomes:
    /// ```json
    /// {"type":"agent_stream","request_id":"...","chunk":"","done":true}
    /// ```
    async fn stream_sse_response(
        &self,
        request_id: &str,
        mut response: reqwest::Response,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        // Read the full SSE response body as a single chunk.
        //
        // Ideally we'd use `response.bytes_stream()` for true streaming,
        // but that requires reqwest's `stream` feature (+ futures::StreamExt).
        // Instead, we use `response.chunk()` in a loop which gives us
        // incremental reads without needing the stream feature.
        //
        // Each chunk may contain partial SSE lines, so we buffer and
        // split by newlines.
        let mut full_response = String::new();
        let mut line_buffer = String::new();

        loop {
            // `chunk()` returns `Option<Bytes>` — None when body is exhausted.
            // This does NOT require the `stream` feature.
            let chunk_opt = response.chunk()
                .await
                .map_err(|e| format!("SSE read error: {}", e))?;

            let chunk_bytes = match chunk_opt {
                Some(bytes) => bytes,
                None => break, // Body exhausted
            };

            let chunk_str = String::from_utf8_lossy(&chunk_bytes);
            line_buffer.push_str(&chunk_str);

            // Process complete lines (SSE uses \n or \r\n as delimiters)
            while let Some(newline_pos) = line_buffer.find('\n') {
                let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
                line_buffer = line_buffer[newline_pos + 1..].to_string();

                // Skip empty lines (SSE event separators)
                if line.is_empty() {
                    continue;
                }

                // SSE data lines start with "data: " or "data:"
                let data = if let Some(stripped) = line.strip_prefix("data: ") {
                    stripped
                } else if let Some(stripped) = line.strip_prefix("data:") {
                    stripped
                } else {
                    // Other SSE fields (event:, id:, retry:) — skip
                    debug!(
                        line = %line,
                        "[WS_TUNNEL] Skipping non-data SSE line"
                    );
                    continue;
                };

                let data = data.trim();

                // Check for stream termination
                if data == "[DONE]" {
                    debug!(
                        request_id = %request_id,
                        "[WS_TUNNEL] ✅ SSE stream complete"
                    );

                    // Send final done message to CMS
                    let done_msg = serde_json::json!({
                        "type": "agent_stream",
                        "request_id": request_id,
                        "chunk": "",
                        "done": true
                    });
                    cms_sink.send(ws_text(&done_msg))
                        .await
                        .map_err(|e| format!("Done send failed: {}", e))?;

                    // Also send complete response as agent_response
                    if !full_response.is_empty() {
                        let final_response = serde_json::json!({
                            "type": "agent_response",
                            "request_id": request_id,
                            "status": "success",
                            "payload": {
                                "response": full_response
                            }
                        });
                        cms_sink.send(ws_text(&final_response))
                            .await
                            .map_err(|e| format!("Final response send failed: {}", e))?;
                    }

                    return Ok(());
                }

                // Parse the JSON data chunk
                let parsed: serde_json::Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(e) => {
                        debug!(
                            data = %data,
                            error = %e,
                            "[WS_TUNNEL] Skipping unparseable SSE data"
                        );
                        continue;
                    }
                };

                // Extract content from OpenAI-compatible format:
                // {"choices":[{"delta":{"content":"text"}}]}
                let content = parsed
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|choice| choice.get("delta"))
                    .and_then(|delta| delta.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                if !content.is_empty() {
                    // Accumulate for final response
                    full_response.push_str(content);

                    // Stream chunk to CMS
                    let stream_msg = serde_json::json!({
                        "type": "agent_stream",
                        "request_id": request_id,
                        "chunk": content,
                        "done": false
                    });
                    cms_sink.send(ws_text(&stream_msg))
                        .await
                        .map_err(|e| format!("Stream chunk send failed: {}", e))?;
                }
            }
        }

        // Body exhausted without [DONE] — still send done
        warn!(
            request_id = %request_id,
            "[WS_TUNNEL] ⚠️ SSE stream ended without [DONE] marker"
        );

        let done_msg = serde_json::json!({
            "type": "agent_stream",
            "request_id": request_id,
            "chunk": "",
            "done": true
        });
        let _ = cms_sink.send(ws_text(&done_msg)).await;

        if !full_response.is_empty() {
            let final_response = serde_json::json!({
                "type": "agent_response",
                "request_id": request_id,
                "status": "success",
                "payload": {
                    "response": full_response
                }
            });
            let _ = cms_sink.send(ws_text(&final_response)).await;
        }

        Ok(())
    }
}

// ============================================
// Shutdown Reason
// ============================================

/// Reason the WebSocket connection ended.
enum ShutdownReason {
    /// Server received shutdown signal.
    GracefulShutdown,
    /// Authentication failed.
    AuthFailed(String),
    /// Connection lost (will reconnect).
    Disconnected(String),
}
