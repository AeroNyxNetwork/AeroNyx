// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_anthropic.rs
// ============================================
//! # Anthropic Provider (Messages API)
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Implements `LlmProvider` for the Anthropic Messages API.
//! Anthropic's API differs from OpenAI in several ways:
//!   - System prompt is a top-level field, not a message role
//!   - Authentication uses `x-api-key` + `anthropic-version` headers
//!   - Response shape uses `content[].text` not `choices[].message.content`
//!   - Token usage field names: `input_tokens`, `output_tokens`, `cache_read_input_tokens`
//!
//! ## Configuration
//! ```toml
//! [[memchain.supernode.providers]]
//! name = "anthropic"
//! type = "anthropic"
//! api_key = "$ANTHROPIC_API_KEY"
//! model = "claude-haiku-4-5-20251001"
//! max_tokens = 1000
//! temperature = 0.3
//! ```
//!
//! ## Request Format
//! POST https://api.anthropic.com/v1/messages
//! Headers: x-api-key, anthropic-version: 2023-06-01, content-type: application/json
//! ```json
//! {
//!   "model": "claude-haiku-4-5-20251001",
//!   "system": "...",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "max_tokens": 1000
//! }
//! ```
//!
//! ⚠️ Important Note for Next Developer:
//! - `max_tokens` is REQUIRED by Anthropic API (no default). We use 1000 if not set.
//! - `system` messages MUST be extracted from the messages array and sent as a
//!   top-level field. This is done in `chat()` before building the request.
//! - Anthropic uses `cache_read_input_tokens` (not `cached_tokens`) for prompt cache.
//! - `anthropic-version` header is fixed at "2023-06-01" (stable API version).
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, warn};

use super::llm_provider::{ChatRequest, ChatResponse, LlmError, LlmProvider, TokenUsage};

// ============================================
// Anthropic API Version
// ============================================

const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const DEFAULT_MAX_TOKENS: u32 = 1000; // Required by Anthropic — no server default

// ============================================
// Wire types
// ============================================

#[derive(serde::Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [String]>,
}

#[derive(serde::Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(serde::Deserialize, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    /// Prompt cache read tokens (subset of input_tokens)
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(serde::Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicErrorBody,
}

#[derive(serde::Deserialize)]
struct AnthropicErrorBody {
    message: String,
}

// ============================================
// AnthropicProvider
// ============================================

/// LLM provider for the Anthropic Messages API.
/// Handles system prompt extraction, x-api-key auth, and cache token tracking.
pub struct AnthropicProvider {
    /// Provider name for logging and writeback.
    name: String,
    /// Resolved API key (after $ENV_VAR expansion).
    api_key: String,
    /// Default model identifier (e.g. "claude-haiku-4-5-20251001").
    model: String,
    /// Optional max_tokens override. Defaults to DEFAULT_MAX_TOKENS if not set.
    max_tokens: Option<u32>,
    /// Optional temperature override.
    temperature: Option<f32>,
    /// Shared HTTP client.
    client: reqwest::Client,
    /// Health flag.
    healthy: Arc<AtomicBool>,
}

impl AnthropicProvider {
    /// Construct a new Anthropic provider.
    ///
    /// `api_key` supports `$ENV_VAR` syntax — resolved at construction time.
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self, String> {
        let api_key_raw = api_key.into();
        let api_key = if api_key_raw.starts_with('$') {
            let var_name = &api_key_raw[1..];
            std::env::var(var_name).unwrap_or_else(|_| {
                warn!("[LLM_ANTHROPIC] Env var '{}' not set", var_name);
                String::new()
            })
        } else {
            api_key_raw
        };

        if api_key.is_empty() {
            warn!("[LLM_ANTHROPIC] api_key is empty — calls will fail with 401");
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Build HTTP client: {}", e))?;

        Ok(Self {
            name: name.into(),
            api_key,
            model: model.into(),
            max_tokens,
            temperature,
            client,
            healthy: Arc::new(AtomicBool::new(true)),
        })
    }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let start = Instant::now();

        let model = req.model_override.as_deref().unwrap_or(&self.model);
        let max_tokens = req.max_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = req.temperature.or(self.temperature);

        // Extract system prompt from messages array (Anthropic uses top-level field)
        let mut system_prompt: Option<&str> = None;
        let user_messages: Vec<AnthropicMessage> = req.messages.iter()
            .filter_map(|m| {
                if m.role == "system" {
                    system_prompt = Some(&m.content);
                    None // Don't include system in messages array
                } else {
                    Some(AnthropicMessage { role: &m.role, content: &m.content })
                }
            })
            .collect();

        if user_messages.is_empty() {
            return Err(LlmError::EmptyResponse);
        }

        let body = AnthropicRequest {
            model,
            system: system_prompt,
            messages: user_messages,
            max_tokens,
            temperature,
            stop_sequences: req.stop.as_deref(),
        };

        let url = format!("{}/v1/messages", ANTHROPIC_API_BASE);

        let resp = self.client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Transport(e.to_string()))?;

        let status = resp.status().as_u16();
        let latency_ms = start.elapsed().as_millis() as u64;

        if status == 429 {
            let retry_after = resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            self.healthy.store(false, Ordering::Relaxed);
            warn!(provider = %self.name, "[LLM_ANTHROPIC] Rate limited");
            return Err(LlmError::RateLimit { retry_after_secs: retry_after });
        }

        if status == 400 {
            let body_text = resp.text().await.unwrap_or_default();
            // Check for context length error
            if body_text.contains("prompt is too long") || body_text.contains("context_length_exceeded") {
                return Err(LlmError::ContextTooLong);
            }
            return Err(LlmError::ApiError { status, body: body_text });
        }

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<AnthropicErrorResponse>(&body_text)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| body_text);
            return Err(LlmError::ApiError { status, body: msg });
        }

        let resp_json: AnthropicResponse = resp.json().await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        // Extract text from first text block
        let content = resp_json.content
            .into_iter()
            .find(|b| b.block_type == "text")
            .map(|b| b.text.trim().to_string())
            .unwrap_or_default();

        if content.is_empty() {
            return Err(LlmError::EmptyResponse);
        }

        let usage = resp_json.usage.unwrap_or_default();
        let model_used = resp_json.model.unwrap_or_else(|| model.to_string());

        debug!(
            provider = %self.name,
            model = %model_used,
            input = usage.input_tokens,
            output = usage.output_tokens,
            cached = usage.cache_read_input_tokens,
            latency_ms = latency_ms,
            "[LLM_ANTHROPIC] Call complete"
        );

        self.healthy.store(true, Ordering::Relaxed);

        Ok(ChatResponse {
            content,
            usage: TokenUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cached_tokens: usage.cache_read_input_tokens,
            },
            model_used,
            provider_name: self.name.clone(),
            latency_ms,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("name", &self.name)
            .field("model", &self.model)
            .finish()
    }
}
