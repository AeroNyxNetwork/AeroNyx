// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_openai.rs
// ============================================
//! # OpenAI-Compatible Provider
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Implements `LlmProvider` for the OpenAI Chat Completions API format.
//! Covers OpenAI, DeepSeek, Groq, Ollama, and any other provider that
//! implements `/v1/chat/completions` with the OpenAI request/response shape.
//!
//! ## Configuration
//! ```toml
//! [[memchain.supernode.providers]]
//! name = "deepseek"
//! type = "openai_compatible"
//! api_base = "https://api.deepseek.com"
//! api_key = "$DEEPSEEK_API_KEY"          # $ENV_VAR syntax supported
//! model = "deepseek-chat"
//! max_tokens = 1000
//! temperature = 0.3
//!
//! [[memchain.supernode.providers]]
//! name = "ollama"
//! type = "openai_compatible"
//! api_base = "http://localhost:11434/v1"
//! api_key = ""                            # Ollama: no key needed
//! model = "llama3.2"
//! ```
//!
//! ## Request Format
//! POST {api_base}/v1/chat/completions
//! ```json
//! {
//!   "model": "deepseek-chat",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "max_tokens": 1000,
//!   "temperature": 0.3
//! }
//! ```
//!
//! ## Response Parsing
//! Extracts `choices[0].message.content` and `usage.{prompt,completion}_tokens`.
//! Anthropic-style `cached_tokens` is extracted from `usage.prompt_tokens_details.cached_tokens`
//! if present (OpenAI o-series models).
//!
//! ⚠️ Important Note for Next Developer:
//! - $ENV_VAR api_key resolution happens at construction time (in `new()`).
//!   If the env var is missing, api_key is set to empty string (no panic).
//! - Ollama does not require Authorization header — empty api_key → no header sent.
//! - Timeout is fixed at 60s for now. TODO: make configurable per provider.
//! - The provider does NOT retry on failure — LlmRouter handles retry policy.
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, warn};

use super::llm_provider::{ChatRequest, ChatResponse, LlmError, LlmProvider, TokenUsage};

// ============================================
// Request / Response wire types
// ============================================

#[derive(serde::Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
}

#[derive(serde::Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(serde::Deserialize)]
struct OpenAiChoiceMessage {
    content: String,
}

#[derive(serde::Deserialize, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    /// OpenAI o-series / cached prompt tokens
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiTokenDetails>,
}

#[derive(serde::Deserialize)]
struct OpenAiTokenDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(serde::Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorBody,
}

#[derive(serde::Deserialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type", default)]
    error_type: String,
}

// ============================================
// OpenAiCompatProvider
// ============================================

/// LLM provider for any OpenAI Chat Completions-compatible endpoint.
/// Covers: OpenAI, DeepSeek, Groq, Ollama, Together AI, Fireworks, etc.
pub struct OpenAiCompatProvider {
    /// Provider name (e.g. "deepseek", "ollama") — for logging and writeback.
    name: String,
    /// Full API base URL (e.g. "https://api.deepseek.com").
    api_base: String,
    /// API key (empty string = no Authorization header, e.g. for Ollama).
    api_key: String,
    /// Default model identifier (e.g. "deepseek-chat").
    model: String,
    /// Optional max_tokens override for all requests.
    max_tokens: Option<u32>,
    /// Optional temperature override for all requests.
    temperature: Option<f32>,
    /// Shared HTTP client (keep-alive connection pool).
    client: reqwest::Client,
    /// Health flag — set to false on rate limit, reset after backoff.
    healthy: Arc<AtomicBool>,
}

impl OpenAiCompatProvider {
    /// Construct a new provider.
    ///
    /// `api_key` supports `$ENV_VAR` syntax — resolved at construction time.
    /// Empty string = no Authorization header (for Ollama and local endpoints).
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self, String> {
        let api_key_raw = api_key.into();
        // Resolve $ENV_VAR syntax
        let api_key = if api_key_raw.starts_with('$') {
            let var_name = &api_key_raw[1..];
            std::env::var(var_name).unwrap_or_else(|_| {
                warn!("[LLM_OPENAI] Env var '{}' not set, using empty api_key", var_name);
                String::new()
            })
        } else {
            api_key_raw
        };

        let mut api_base = api_base.into();
        // Normalize: strip trailing slash
        while api_base.ends_with('/') {
            api_base.pop();
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Build HTTP client: {}", e))?;

        Ok(Self {
            name: name.into(),
            api_base,
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
impl LlmProvider for OpenAiCompatProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let start = Instant::now();

        let model = req.model_override.as_deref().unwrap_or(&self.model);
        let max_tokens = req.max_tokens.or(self.max_tokens);
        let temperature = req.temperature.or(self.temperature);

        let messages: Vec<OpenAiMessage> = req.messages.iter()
            .map(|m| OpenAiMessage { role: &m.role, content: &m.content })
            .collect();

        let body = OpenAiRequest {
            model,
            messages,
            max_tokens,
            temperature,
            stop: req.stop.as_deref(),
        };

        let url = format!("{}/v1/chat/completions", self.api_base);

        let mut request_builder = self.client
            .post(&url)
            .header("Content-Type", "application/json");

        // Only add Authorization if api_key is non-empty
        if !self.api_key.is_empty() {
            request_builder = request_builder
                .header("Authorization", format!("Bearer {}", self.api_key));
        }

        let resp = request_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Transport(e.to_string()))?;

        let status = resp.status().as_u16();
        let latency_ms = start.elapsed().as_millis() as u64;

        if status == 429 {
            // Rate limit — extract Retry-After header if present
            let retry_after = resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            self.healthy.store(false, Ordering::Relaxed);
            warn!(provider = %self.name, retry_after = ?retry_after, "[LLM_OPENAI] Rate limited");
            return Err(LlmError::RateLimit { retry_after_secs: retry_after });
        }

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            // Try to parse structured error
            let msg = serde_json::from_str::<OpenAiErrorResponse>(&body_text)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| body_text);
            return Err(LlmError::ApiError { status, body: msg });
        }

        let resp_json: OpenAiResponse = resp.json().await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = resp_json.choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .unwrap_or_default();

        if content.is_empty() {
            return Err(LlmError::EmptyResponse);
        }

        let usage = resp_json.usage.unwrap_or_default();
        let cached = usage.prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0);

        let model_used = resp_json.model.unwrap_or_else(|| model.to_string());

        debug!(
            provider = %self.name,
            model = %model_used,
            input = usage.prompt_tokens,
            output = usage.completion_tokens,
            latency_ms = latency_ms,
            "[LLM_OPENAI] Call complete"
        );

        // Reset health on successful call
        self.healthy.store(true, Ordering::Relaxed);

        Ok(ChatResponse {
            content,
            usage: TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cached_tokens: cached,
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

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("name", &self.name)
            .field("api_base", &self.api_base)
            .field("model", &self.model)
            .finish()
    }
}
