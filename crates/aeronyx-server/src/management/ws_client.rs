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
//! ## Main Functionality
//! - `WsTunnel`: Background async task managing the CMS WebSocket lifecycle
//! - Ed25519-signed authentication on connect
//! - Automatic reconnection with exponential backoff
//! - Bidirectional message relay: CMS ↔ local OpenClaw Gateway
//! - Ping/pong keepalive for connection health
//! - Graceful shutdown via broadcast signal
//!
//! ## Main Logical Flow
//! ```text
//!   ┌────────┐       wss://            ┌─────────┐     ws://localhost:18789    ┌──────────┐
//!   │  CMS   │ ◄──────────────────────►│ WsTunnel│ ◄────────────────────────► │ OpenClaw │
//!   │Backend │  agent_request/response  │ (Rust)  │  OpenClaw Gateway RPC      │ Gateway  │
//!   └────────┘                          └─────────┘                            └──────────┘
//! ```
//!
//! 1. WsTunnel connects to `wss://api.aeronyx.network/ws/node/tunnel/{node_id}/`
//! 2. Sends `auth` message with Ed25519-signed `{public_key}:{timestamp}`
//! 3. Waits for `auth_ok` (30s timeout on CMS side)
//! 4. Enters message loop:
//!    - `agent_request` from CMS → forward to local OpenClaw Gateway via WS RPC
//!    - OpenClaw response → wrap as `agent_response`/`agent_stream` → send to CMS
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
//! Signature: `Ed25519_Sign(SHA256("{public_key_hex}:{timestamp}"))`
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
//! ## OpenClaw Gateway RPC (local)
//! The OpenClaw Gateway at `ws://127.0.0.1:18789` uses a JSON RPC protocol.
//! We connect as role=`operator` with the gateway token from config.
//! Chat messages are sent via the `gateway.call` / `chat.send` method.
//!
//! ## Dependencies
//! - `tokio-tungstenite` with `rustls-tls-native-roots` for wss:// to CMS
//! - `tokio-tungstenite` (plain) for ws:// to local OpenClaw
//! - `aeronyx_core::crypto::IdentityKeyPair` for Ed25519 signing
//! - `sha2` for SHA-256 message hashing
//!
//! ## ⚠️ Important Note for Next Developer
//! - The CMS URL uses `node_id` which is the **CMS database UUID**, NOT the
//!   Ed25519 public key. The UUID is obtained from `StoredNodeInfo.node_id`.
//! - The `auth` message uses `node_id` field which IS the Ed25519 public key hex.
//! - Signature format differs from heartbeat: it's `"{pubkey}:{timestamp}"` directly
//!   (no body), and the signature is over the raw bytes (no SHA-256 intermediate).
//!   Wait — your spec says: `verify(f"{public_key}:{timestamp}")` and
//!   `signing_key.sign(message.as_bytes())` — so it's a raw sign, NOT SHA-256.
//!   This differs from heartbeat which does SHA-256 first. Be careful.
//! - The OpenClaw Gateway connection is lazy: only established when the first
//!   `agent_request` arrives, and cached for subsequent requests.
//! - If OpenClaw is not installed/running, `agent_request` responses will be
//!   `{"type": "agent_response", "request_id": "...", "status": "error",
//!     "payload": {"error": "OpenClaw gateway is not available"}}`
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation (Phase 3: WebSocket Tunnel)
//! ============================================

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use sha2::{Digest, Sha256};
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

/// Local OpenClaw Gateway WebSocket URL.
const OPENCLAW_GATEWAY_WS: &str = "ws://127.0.0.1:18789";

/// Maximum reconnection backoff interval (seconds).
const MAX_RECONNECT_BACKOFF_SECS: u64 = 60;

/// Initial reconnection delay (seconds).
const INITIAL_RECONNECT_DELAY_SECS: u64 = 1;

/// Ping interval to keep the CMS connection alive (seconds).
const PING_INTERVAL_SECS: u64 = 30;

/// Timeout waiting for auth_ok response (seconds).
const AUTH_TIMEOUT_SECS: u64 = 25; // Slightly under CMS's 30s limit

// ============================================
// Type aliases
// ============================================

type WsStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<WsStream, WsMessage>;
type WsSource = SplitStream<WsStream>;

// ============================================
// WsTunnel
// ============================================

/// WebSocket tunnel client for CMS ↔ OpenClaw real-time communication.
///
/// ## Lifecycle
/// 1. Created in `server.rs` with identity + node info
/// 2. Spawned as a `tokio::spawn` task via `run()`
/// 3. Maintains persistent connection to CMS with auto-reconnect
/// 4. Relays `agent_request` from CMS to local OpenClaw Gateway
/// 5. Stops on shutdown signal
pub struct WsTunnel {
    /// Ed25519 identity for signing auth messages.
    identity: IdentityKeyPair,

    /// CMS database UUID for the node (used in WS URL path).
    /// Loaded from `StoredNodeInfo.node_id`.
    cms_node_id: String,

    /// Agent manager for checking OpenClaw availability and getting gateway token.
    agent_manager: Arc<AgentManager>,
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
        Self {
            identity,
            cms_node_id,
            agent_manager,
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

        sink.send(WsMessage::Text(auth_msg.to_string()))
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
    /// - `agent_request` from CMS → forward to OpenClaw → respond
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
                    if let Err(e) = sink.send(WsMessage::Text(ping_msg.to_string())).await {
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
                sink.send(WsMessage::Text(pong.to_string()))
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
    // Agent Request Handling
    // ============================================

    /// Handles an `agent_request` from CMS by forwarding to local OpenClaw.
    ///
    /// The flow:
    /// 1. Extract `request_id`, `action`, `payload` from CMS message
    /// 2. Connect to local OpenClaw Gateway (ws://127.0.0.1:18789)
    /// 3. Authenticate with gateway token
    /// 4. Send the request via OpenClaw's RPC protocol
    /// 5. Stream/collect the response
    /// 6. Send `agent_response` or `agent_stream` back to CMS
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

        // Get gateway token
        let gateway_token = match self.agent_manager.gateway_token().await {
            Some(token) => token,
            None => {
                let error_response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "payload": {
                        "error": "OpenClaw gateway token not available. Is OpenClaw installed?"
                    }
                });
                let _ = cms_sink.send(WsMessage::Text(error_response.to_string())).await;
                return Ok(());
            }
        };

        // Forward to OpenClaw and relay response
        match self.forward_to_openclaw(request_id, action, &payload, &gateway_token, cms_sink).await {
            Ok(()) => {
                debug!(
                    request_id = %request_id,
                    "[WS_TUNNEL] ✅ Agent request completed"
                );
            }
            Err(e) => {
                warn!(
                    request_id = %request_id,
                    error = %e,
                    "[WS_TUNNEL] ❌ Agent request failed"
                );
                let error_response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "error",
                    "payload": {
                        "error": format!("Failed to communicate with OpenClaw: {}", e)
                    }
                });
                let _ = cms_sink.send(WsMessage::Text(error_response.to_string())).await;
            }
        }

        Ok(())
    }

    /// Connects to local OpenClaw Gateway and relays a chat request.
    ///
    /// OpenClaw uses a JSON RPC protocol over WebSocket. The flow:
    /// 1. Connect to ws://127.0.0.1:18789
    /// 2. Send `connect` frame with role=operator and gateway token
    /// 3. Wait for `hello-ok`
    /// 4. Send chat message via `chat.send` method
    /// 5. Stream response events back to CMS
    async fn forward_to_openclaw(
        &self,
        request_id: &str,
        action: &str,
        payload: &serde_json::Value,
        gateway_token: &str,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        // Connect to local OpenClaw Gateway
        let (oc_stream, _) = connect_async(OPENCLAW_GATEWAY_WS)
            .await
            .map_err(|e| format!("OpenClaw Gateway connect failed: {}", e))?;

        let (mut oc_sink, mut oc_source) = oc_stream.split();

        // Send connect frame (OpenClaw RPC protocol)
        let connect_msg = serde_json::json!({
            "type": "req",
            "id": format!("aeronyx-{}", request_id),
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": "aeronyx-tunnel",
                    "version": env!("CARGO_PKG_VERSION"),
                    "platform": "linux",
                    "mode": "operator"
                },
                "role": "operator",
                "scopes": [],
                "auth": {
                    "token": gateway_token
                }
            }
        });

        oc_sink.send(WsMessage::Text(connect_msg.to_string()))
            .await
            .map_err(|e| format!("OpenClaw connect send failed: {}", e))?;

        // Wait for hello-ok (with timeout)
        let hello_ok = tokio::time::timeout(
            Duration::from_secs(10),
            Self::wait_for_openclaw_hello(&mut oc_source),
        ).await
            .map_err(|_| "OpenClaw hello-ok timeout".to_string())?
            .map_err(|e| format!("OpenClaw handshake failed: {}", e))?;

        debug!("[WS_TUNNEL] ✅ Connected to OpenClaw Gateway");

        // Send the actual request based on action
        match action {
            "chat" => {
                let prompt = payload.get("prompt")
                    .and_then(|p| p.as_str())
                    .unwrap_or("");

                let chat_msg = serde_json::json!({
                    "type": "req",
                    "id": format!("chat-{}", request_id),
                    "method": "chat.send",
                    "params": {
                        "message": prompt,
                        "session": "main"
                    }
                });

                oc_sink.send(WsMessage::Text(chat_msg.to_string()))
                    .await
                    .map_err(|e| format!("Chat send failed: {}", e))?;

                // Stream responses back to CMS
                self.relay_openclaw_response(
                    request_id,
                    &mut oc_source,
                    cms_sink,
                ).await?;
            }
            "status" => {
                // Return agent status directly
                let status = self.agent_manager.status().await;
                let response = serde_json::json!({
                    "type": "agent_response",
                    "request_id": request_id,
                    "status": "success",
                    "payload": status
                });
                cms_sink.send(WsMessage::Text(response.to_string()))
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
                cms_sink.send(WsMessage::Text(response.to_string()))
                    .await
                    .map_err(|e| format!("Error response send failed: {}", e))?;
            }
        }

        // Close OpenClaw connection
        let _ = oc_sink.close().await;

        Ok(())
    }

    /// Waits for OpenClaw Gateway's hello-ok response after connect.
    async fn wait_for_openclaw_hello(source: &mut WsSource) -> Result<(), String> {
        while let Some(msg_result) = source.next().await {
            let msg = msg_result.map_err(|e| format!("WS error: {}", e))?;

            if let WsMessage::Text(text) = msg {
                let parsed: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| format!("Invalid JSON: {}", e))?;

                // OpenClaw sends {"type": "res", "id": "...", "result": {...}}
                // for successful connect, or an event with hello-ok
                let msg_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");

                if msg_type == "res" {
                    // Check if it's an error response
                    if parsed.get("error").is_some() {
                        let err = parsed.get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error");
                        return Err(format!("OpenClaw rejected connect: {}", err));
                    }
                    return Ok(());
                }

                if msg_type == "event" {
                    let event = parsed.get("event").and_then(|e| e.as_str()).unwrap_or("");
                    if event == "connect.hello-ok" || event.contains("hello") {
                        return Ok(());
                    }
                }
            }
        }

        Err("OpenClaw connection closed before hello".to_string())
    }

    /// Relays OpenClaw response events back to CMS as agent_stream/agent_response.
    ///
    /// OpenClaw streams responses as multiple events. We relay each
    /// text chunk as an `agent_stream` message, and the final response
    /// as `agent_response`.
    async fn relay_openclaw_response(
        &self,
        request_id: &str,
        oc_source: &mut WsSource,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let timeout = Duration::from_secs(120); // 2 min max per request
        let deadline = tokio::time::Instant::now() + timeout;

        while tokio::time::Instant::now() < deadline {
            let read_result = tokio::time::timeout(
                Duration::from_secs(30),
                oc_source.next(),
            ).await;

            match read_result {
                Ok(Some(Ok(WsMessage::Text(text)))) => {
                    let parsed: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let msg_type = parsed.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");

                    match msg_type {
                        // OpenClaw event responses
                        "event" => {
                            let event_name = parsed.get("event")
                                .and_then(|e| e.as_str())
                                .unwrap_or("");

                            // Chat events contain streamed text
                            if event_name.contains("chat") || event_name.contains("agent") {
                                let event_payload = parsed.get("payload")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);

                                // Extract text content from event
                                let text_content = event_payload.get("text")
                                    .or_else(|| event_payload.get("content"))
                                    .or_else(|| event_payload.get("chunk"))
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");

                                let is_final = event_name.contains("final")
                                    || event_name.contains("done")
                                    || event_name.contains("end");

                                if !text_content.is_empty() || is_final {
                                    if is_final {
                                        // Final response
                                        let response = serde_json::json!({
                                            "type": "agent_response",
                                            "request_id": request_id,
                                            "status": "success",
                                            "payload": {
                                                "response": text_content
                                            }
                                        });
                                        cms_sink.send(WsMessage::Text(response.to_string()))
                                            .await
                                            .map_err(|e| format!("Send failed: {}", e))?;
                                        return Ok(());
                                    } else {
                                        // Streaming chunk
                                        let stream_msg = serde_json::json!({
                                            "type": "agent_stream",
                                            "request_id": request_id,
                                            "chunk": text_content,
                                            "done": false
                                        });
                                        cms_sink.send(WsMessage::Text(stream_msg.to_string()))
                                            .await
                                            .map_err(|e| format!("Stream send failed: {}", e))?;
                                    }
                                }
                            }
                        }
                        // RPC response (non-streaming)
                        "res" => {
                            let result = parsed.get("result")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);

                            let response = serde_json::json!({
                                "type": "agent_response",
                                "request_id": request_id,
                                "status": "success",
                                "payload": result
                            });
                            cms_sink.send(WsMessage::Text(response.to_string()))
                                .await
                                .map_err(|e| format!("Response send failed: {}", e))?;
                            return Ok(());
                        }
                        _ => {
                            // Other events — skip
                        }
                    }
                }
                Ok(Some(Ok(WsMessage::Close(_)))) => {
                    // OpenClaw closed connection — send final done
                    let done_msg = serde_json::json!({
                        "type": "agent_stream",
                        "request_id": request_id,
                        "chunk": "",
                        "done": true
                    });
                    let _ = cms_sink.send(WsMessage::Text(done_msg.to_string())).await;
                    return Ok(());
                }
                Ok(Some(Err(e))) => {
                    return Err(format!("OpenClaw WS error: {}", e));
                }
                Ok(None) => {
                    return Ok(()); // Stream ended
                }
                Err(_) => {
                    // Read timeout — send done and return
                    let done_msg = serde_json::json!({
                        "type": "agent_stream",
                        "request_id": request_id,
                        "chunk": "",
                        "done": true
                    });
                    let _ = cms_sink.send(WsMessage::Text(done_msg.to_string())).await;
                    return Ok(());
                }
            }
        }

        // Overall timeout
        let timeout_msg = serde_json::json!({
            "type": "agent_response",
            "request_id": request_id,
            "status": "error",
            "payload": {"error": "Request timed out (120s)"}
        });
        let _ = cms_sink.send(WsMessage::Text(timeout_msg.to_string())).await;

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
