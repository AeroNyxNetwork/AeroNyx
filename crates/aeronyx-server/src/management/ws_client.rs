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
//! The HTTP API is OpenAI-compatible, supports SSE streaming, and is simpler
//! than the internal WS RPC protocol.
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
//! ## CMS WebSocket Protocol
//!
//! ### Authentication (must complete within 30s)
//! ```json
//! → {"type": "auth", "node_id": "<public_key_hex>", "timestamp": 1709500000, "signature": "<hex>"}
//! ← {"type": "auth_ok", "node_id": "<uuid>", "node_name": "My Node"}
//! ```
//!
//! ### Messages (post-auth)
//! ```json
//! ← {"type": "agent_request", "request_id": "uuid", "action": "chat", "payload": {...}}
//! → {"type": "agent_response", "request_id": "uuid", "status": "success", "payload": {...}}
//! → {"type": "agent_stream", "request_id": "uuid", "chunk": "...", "done": false}
//! ↔ {"type": "ping"} / {"type": "pong"}
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - The CMS URL uses `node_id` which is the **CMS database UUID**, NOT the
//!   Ed25519 public key.
//! - The `auth` message `node_id` field IS the Ed25519 public key hex.
//! - Signature: `Ed25519_Sign("{pubkey}:{timestamp}".as_bytes())` — raw sign,
//!   NOT SHA-256 then sign. Differs from heartbeat.
//! - The `reqwest::Client` is created ONCE and reused (connection pool).
//! - OpenClaw HTTP API must be enabled in config for this to work.
//! - The `user` field in requests creates stable sessions in OpenClaw.
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation (Phase 3: WebSocket Tunnel)
//! v1.3.2 - 🔄 Replaced OpenClaw WS RPC with HTTP API
//! v1.4.0 - 🐛 Fixed SSE [DONE] signal lost in residual buffer
//!          - 🔄 Refactored stream_sse_response into 3 focused methods
//!          - 🔄 Error path now sends done: true before error response
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

const CMS_WS_BASE_URL: &str = "wss://api.aeronyx.network/ws/node/tunnel";
const OPENCLAW_HTTP_BASE: &str = "http://127.0.0.1:18789";
const OPENCLAW_CHAT_PATH: &str = "/v1/chat/completions";
const MAX_RECONNECT_BACKOFF_SECS: u64 = 60;
const INITIAL_RECONNECT_DELAY_SECS: u64 = 1;
const PING_INTERVAL_SECS: u64 = 30;
const AUTH_TIMEOUT_SECS: u64 = 25;
const OPENCLAW_REQUEST_TIMEOUT_SECS: u64 = 180;
const OPENCLAW_AGENT_ID: &str = "main";

// ============================================
// Type aliases
// ============================================

type WsStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<WsStream, WsMessage>;
type WsSource = SplitStream<WsStream>;

fn ws_text(value: &serde_json::Value) -> WsMessage {
    WsMessage::Text(value.to_string().into())
}

// ============================================
// WsTunnel
// ============================================

pub struct WsTunnel {
    identity: IdentityKeyPair,
    cms_node_id: String,
    agent_manager: Arc<AgentManager>,
    http_client: reqwest::Client,
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
            .build()
            .expect("Failed to build reqwest::Client");

        Self {
            identity,
            cms_node_id,
            agent_manager,
            http_client,
        }
    }

    pub async fn run(self, mut shutdown: broadcast::Receiver<()>) {
        info!(
            cms_node_id = %self.cms_node_id,
            "[WS_TUNNEL] Starting WebSocket tunnel"
        );

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
                    warn!(
                        reason = %reason,
                        backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ⚠️ Disconnected — reconnecting in {}s", backoff_secs
                    );
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        backoff_secs = backoff_secs,
                        "[WS_TUNNEL] ❌ Connection error — reconnecting in {}s", backoff_secs
                    );
                }
            }

            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[WS_TUNNEL] Shutdown during backoff");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_secs(backoff_secs)) => {}
            }

            backoff_secs = (backoff_secs * 2).min(MAX_RECONNECT_BACKOFF_SECS);
        }

        info!("[WS_TUNNEL] WebSocket tunnel stopped");
    }

    async fn connect_and_run(
        &self,
        url: &str,
        shutdown: &mut broadcast::Receiver<()>,
    ) -> Result<ShutdownReason, String> {
        let (ws_stream, _response) = connect_async(url)
            .await
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
                            let node_name = parsed.get("node_name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            info!(node_name = %node_name, "[WS_TUNNEL] ✅ Auth OK");
                            return Ok(());
                        }
                        Some("error") => {
                            let error_msg = parsed.get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown error");
                            return Err(format!("Auth rejected: {}", error_msg));
                        }
                        other => {
                            debug!(msg_type = ?other, "[WS_TUNNEL] Unexpected message during auth");
                        }
                    }
                }
                WsMessage::Close(frame) => {
                    let reason = frame
                        .map(|f| format!("code={}, reason={}", f.code, f.reason))
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
        &self,
        sink: &mut WsSink,
        source: &mut WsSource,
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
                    if let Err(e) = sink.send(ws_text(&ping_msg)).await {
                        return Ok(ShutdownReason::Disconnected(
                            format!("Ping send failed: {}", e)
                        ));
                    }
                }

                msg = source.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            if let Err(e) = self.handle_cms_message(&text, sink).await {
                                warn!(error = %e, "[WS_TUNNEL] ⚠️ Error handling CMS message");
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
                            return Ok(ShutdownReason::Disconnected(format!("WS read error: {}", e)));
                        }
                        None => {
                            return Ok(ShutdownReason::Disconnected("Stream ended".to_string()));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

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
                debug!(msg_type = %other, "[WS_TUNNEL] Unhandled message type from CMS");
            }
        }

        Ok(())
    }

    // ============================================
    // Agent Request Handling
    // ============================================

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

        let gateway_token = match self.agent_manager.gateway_token().await {
            Some(token) => token,
            None => {
                self.send_done_and_error(
                    request_id,
                    "OpenClaw gateway token not available. Is OpenClaw installed and configured?",
                    cms_sink,
                ).await;
                return Ok(());
            }
        };

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
                    self.send_done_and_error(
                        request_id,
                        &format!("Failed to communicate with OpenClaw: {}", e),
                        cms_sink,
                    ).await;
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
                    "payload": { "error": format!("Unknown action: {}", other) }
                });
                cms_sink.send(ws_text(&response))
                    .await
                    .map_err(|e| format!("Error response send failed: {}", e))?;
            }
        }

        Ok(())
    }

    /// Sends `agent_stream(done=true)` + `agent_response(error)` to CMS.
    /// Ensures the frontend always receives a termination signal,
    /// even on errors or timeouts.
    async fn send_done_and_error(
        &self,
        request_id: &str,
        error_message: &str,
        cms_sink: &mut WsSink,
    ) {
        let done_msg = serde_json::json!({
            "type": "agent_stream",
            "request_id": request_id,
            "chunk": "",
            "done": true
        });
        let _ = cms_sink.send(ws_text(&done_msg)).await;

        let error_response = serde_json::json!({
            "type": "agent_response",
            "request_id": request_id,
            "status": "error",
            "payload": { "error": error_message }
        });
        let _ = cms_sink.send(ws_text(&error_response)).await;
    }

    // ============================================
    // Chat Request: HTTP SSE
    // ============================================

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

        let session_user = if let Some(user_id) = payload.get("user_id").and_then(|u| u.as_str()) {
            format!("aeronyx:user:{}", user_id)
        } else {
            format!("aeronyx:req:{}", request_id)
        };

        let request_body = serde_json::json!({
            "model": "openclaw",
            "stream": true,
            "user": session_user,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        });

        let url = format!("{}{}", OPENCLAW_HTTP_BASE, OPENCLAW_CHAT_PATH);

        info!(
            request_id = %request_id,
            url = %url,
            session = %session_user,
            "[WS_TUNNEL] 📤 Sending chat to OpenClaw"
        );

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
            .map_err(|_| format!(
                "OpenClaw request timed out ({}s). The AI may be stuck in an interactive loop.",
                OPENCLAW_REQUEST_TIMEOUT_SECS
            ))?
            .map_err(|e| format!("OpenClaw HTTP request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(format!(
                "OpenClaw returned HTTP {}: {}",
                status.as_u16(),
                error_body.chars().take(500).collect::<String>()
            ));
        }

        info!(
            request_id = %request_id,
            http_status = %status.as_u16(),
            "[WS_TUNNEL] ✅ OpenClaw responded, starting SSE stream"
        );

        self.stream_sse_response(request_id, response, cms_sink).await
    }

    // ============================================
    // SSE Stream Processing (3 methods)
    // ============================================

    /// Main SSE processing entry point.
    ///
    /// Phase 1: Read body chunks via `response.chunk()` and process lines.
    /// Phase 2: Process residual data left in the buffer after body exhaustion.
    /// Phase 3: Fallback — send done signal if [DONE] was never received.
    async fn stream_sse_response(
        &self,
        request_id: &str,
        mut response: reqwest::Response,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let mut full_response = String::new();
        let mut line_buffer = String::new();

        // Phase 1: Read body incrementally.
        // Each chunk() call has a 10-second timeout — if no data arrives
        // in 10 seconds, we consider this turn's output complete.
        //
        // Why 10 seconds instead of longer:
        // OpenClaw keeps the SSE connection open for multi-turn interaction
        // (e.g., "choose 1/2/3" then waits for user reply in the same stream).
        // But our architecture uses one HTTP request per user message, so
        // when the AI stops outputting text, this turn is done.
        // Normal streaming has sub-second gaps between chunks, so 10s of
        // silence reliably indicates the AI finished its current response.
        // Tool execution (browser, code) may cause pauses, but OpenClaw
        // sends status events during tool execution, keeping the stream alive.
        let chunk_timeout = Duration::from_secs(10);
        loop {
            let chunk_result = tokio::time::timeout(
                chunk_timeout,
                response.chunk(),
            ).await;

            match chunk_result {
                Ok(Ok(Some(bytes))) => {
                    line_buffer.push_str(&String::from_utf8_lossy(&bytes));

                    if self.process_sse_lines(
                        request_id, &mut line_buffer, &mut full_response, cms_sink
                    ).await? {
                        return Ok(()); // [DONE] found and handled
                    }
                }
                Ok(Ok(None)) => {
                    info!(
                        request_id = %request_id,
                        residual_len = line_buffer.len(),
                        "[WS_TUNNEL] SSE body exhausted"
                    );
                    break;
                }
                Ok(Err(e)) => {
                    warn!(
                        request_id = %request_id,
                        error = %e,
                        "[WS_TUNNEL] SSE read error"
                    );
                    break;
                }
                Err(_) => {
                    info!(
                        request_id = %request_id,
                        response_len = full_response.len(),
                        "[WS_TUNNEL] SSE idle timeout (10s no data) — ending stream"
                    );
                    break;
                }
            }
        }

        // Phase 2: Process residual buffer
        if !line_buffer.is_empty() {
            if !line_buffer.ends_with('\n') {
                line_buffer.push('\n');
            }
            if self.process_sse_lines(
                request_id, &mut line_buffer, &mut full_response, cms_sink
            ).await? {
                return Ok(());
            }
        }

        // Phase 3: Fallback — [DONE] was never received
        warn!(
            request_id = %request_id,
            response_len = full_response.len(),
            "[WS_TUNNEL] ⚠️ SSE ended without [DONE] — sending fallback done"
        );
        self.send_done_and_response(request_id, &full_response, cms_sink).await
    }

    /// Extracts complete SSE lines from the buffer and processes them.
    ///
    /// Returns `Ok(true)` if `[DONE]` was found and final messages were sent.
    /// Returns `Ok(false)` if more data is needed.
    async fn process_sse_lines(
        &self,
        request_id: &str,
        line_buffer: &mut String,
        full_response: &mut String,
        cms_sink: &mut WsSink,
    ) -> Result<bool, String> {
        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            *line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            // Extract "data: ..." payload
            let data = if let Some(s) = line.strip_prefix("data: ") {
                s.trim()
            } else if let Some(s) = line.strip_prefix("data:") {
                s.trim()
            } else {
                continue;
            };

            // Stream termination
            if data == "[DONE]" {
                info!(request_id = %request_id, "[WS_TUNNEL] ✅ SSE [DONE] received");
                self.send_done_and_response(request_id, full_response, cms_sink).await?;
                return Ok(true);
            }

            // Parse OpenAI-compatible JSON chunk
            let parsed: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Extract delta content: choices[0].delta.content
            let content = parsed
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");

            if !content.is_empty() {
                full_response.push_str(content);

                let stream_msg = serde_json::json!({
                    "type": "agent_stream",
                    "request_id": request_id,
                    "chunk": content,
                    "done": false
                });
                cms_sink.send(ws_text(&stream_msg))
                    .await
                    .map_err(|e| format!("Stream send failed: {}", e))?;
            }
        }

        Ok(false)
    }

    /// Sends `agent_stream(done=true)` + `agent_response(success)` to CMS.
    async fn send_done_and_response(
        &self,
        request_id: &str,
        full_response: &str,
        cms_sink: &mut WsSink,
    ) -> Result<(), String> {
        let done_msg = serde_json::json!({
            "type": "agent_stream",
            "request_id": request_id,
            "chunk": "",
            "done": true
        });
        let _ = cms_sink.send(ws_text(&done_msg)).await;

        if !full_response.is_empty() {
            let final_msg = serde_json::json!({
                "type": "agent_response",
                "request_id": request_id,
                "status": "success",
                "payload": {
                    "response": full_response
                }
            });
            let _ = cms_sink.send(ws_text(&final_msg)).await;
        }

        Ok(())
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
