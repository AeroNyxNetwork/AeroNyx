// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_provider.rs
// ============================================
//! # LLM Provider — Trait + Shared Types
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Defines the `LlmProvider` async trait and all shared request/response types
//! used by the two provider implementations (OpenAI-compatible, Anthropic) and
//! the router (`LlmRouter`).
//!
//! ## Main Types
//! - `LlmProvider` trait — single method: `chat(req) → Result<ChatResponse>`
//! - `ChatRequest` — messages + parameters sent to any provider
//! - `ChatResponse` — model output + token usage
//! - `ChatMessage` — role + content pair
//! - `TokenUsage` — input/output/cached token counts
//! - `LlmError` — structured error type for provider failures
//!
//! ## CognitiveTaskType — RE-EXPORTED from config_supernode.rs
//! ⚠️ CognitiveTaskType is defined in config_supernode.rs (the single source of truth)
//! and re-exported here for backward compatibility with code that imports from
//! `llm_provider::CognitiveTaskType`. Do NOT define CognitiveTaskType in this file.
//!
//! ## Design Decisions
//! - `LlmProvider` is object-safe (`async_trait` macro expands to boxed futures)
//! - `TokenUsage` intentionally omits `cost_usd` — fee rates change; compute at query time
//! - All types derive `serde::Serialize/Deserialize` for JSON storage in DB
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding a new task type, add the variant to config_supernode::CognitiveTaskType,
//!   NOT here. This file only re-exports it.
//! - `ChatRequest::system` is optional. For task types that don't need a system
//!   prompt (simple completion), leave it None.
//! - `LlmProvider::chat()` must be cancel-safe — the caller may drop the future
//!   if the task is cancelled.
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.
//! v2.5.0+Unify     - 🔧 [BUG FIX] Removed duplicate CognitiveTaskType definition.
//!   CognitiveTaskType is now defined ONLY in config_supernode.rs and re-exported
//!   here. The old definition had different variant names (CommunitySummary vs
//!   CommunityNarrative, NaturalSummary vs RecallSynthesis, CustomPrompt vs
//!   ConflictResolution/CodeAnalysis) which caused compilation errors across
//!   task_worker.rs, llm_router.rs, and mod.rs re-exports.

use std::fmt;

// ============================================
// Re-export CognitiveTaskType from canonical location
// ============================================

/// Re-exported from config_supernode.rs — the SINGLE SOURCE OF TRUTH.
/// All code that previously imported `llm_provider::CognitiveTaskType`
/// will continue to work without changes.
pub use crate::config_supernode::CognitiveTaskType;

// ============================================
// Error Type
// ============================================

/// Structured error returned by LLM provider calls.
#[derive(Debug, Clone)]
pub enum LlmError {
    /// HTTP transport error (connection refused, timeout, etc.)
    Transport(String),
    /// Provider returned a non-2xx HTTP status
    ApiError { status: u16, body: String },
    /// Response body could not be parsed
    ParseError(String),
    /// Model returned an empty or unusable response
    EmptyResponse,
    /// Rate limit hit (HTTP 429)
    RateLimit { retry_after_secs: Option<u64> },
    /// Context too long for this model
    ContextTooLong,
    /// Provider is not configured
    NotConfigured(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "transport error: {}", e),
            Self::ApiError { status, body } => write!(f, "API error {}: {}", status, &body[..body.len().min(200)]),
            Self::ParseError(e) => write!(f, "parse error: {}", e),
            Self::EmptyResponse => write!(f, "empty response from model"),
            Self::RateLimit { retry_after_secs } => write!(f, "rate limit hit (retry after: {:?}s)", retry_after_secs),
            Self::ContextTooLong => write!(f, "context too long for this model"),
            Self::NotConfigured(name) => write!(f, "provider '{}' not configured", name),
        }
    }
}

impl std::error::Error for LlmError {}

// ============================================
// Chat Types
// ============================================

/// A single message in a conversation (role + content).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// Request sent to an LLM provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    /// Conversation messages (system + user + optional assistant)
    pub messages: Vec<ChatMessage>,
    /// Optional override for model (if None, provider uses its configured default)
    pub model_override: Option<String>,
    /// Maximum tokens to generate (None = provider default)
    pub max_tokens: Option<u32>,
    /// Temperature 0.0-2.0 (None = provider default)
    pub temperature: Option<f32>,
    /// Stop sequences (None = no stop sequences)
    pub stop: Option<Vec<String>>,
}

impl ChatRequest {
    /// Convenience constructor: single user message, no system prompt.
    pub fn simple(user_content: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::user(user_content)],
            model_override: None,
            max_tokens: None,
            temperature: None,
            stop: None,
        }
    }

    /// Convenience constructor: system + user message.
    pub fn with_system(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
            model_override: None,
            max_tokens: None,
            temperature: None,
            stop: None,
        }
    }
}

/// Token usage for a single LLM call.
///
/// ## Note on cost_usd
/// Cost is intentionally NOT stored here. Fee rates change frequently and vary
/// by context (cached vs. uncached, batch vs. real-time). Compute at query time
/// using rate tables in `LlmRouter::estimate_cost()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens served from prompt cache (subset of input_tokens)
    pub cached_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    pub fn billable_input(&self) -> u32 {
        self.input_tokens.saturating_sub(self.cached_tokens)
    }
}

/// Response from an LLM provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatResponse {
    /// The model's text output (first choice, trimmed)
    pub content: String,
    /// Token usage for this call
    pub usage: TokenUsage,
    /// The model identifier actually used (may differ from request if overridden)
    pub model_used: String,
    /// Provider name (for logging and writeback)
    pub provider_name: String,
    /// Wall-clock latency in milliseconds
    pub latency_ms: u64,
}

// ============================================
// LlmProvider Trait
// ============================================

/// Async trait for a single LLM provider backend.
///
/// Implementations: `OpenAiCompatProvider` (covers OpenAI/DeepSeek/Groq/Ollama),
/// `AnthropicProvider` (Anthropic Messages API).
///
/// Each implementation handles its own:
/// - HTTP transport (reqwest)
/// - Authentication header format
/// - Request/response JSON shape
/// - Rate limit detection
/// - Timeout (from its config)
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a chat completion request and return the response.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Provider name for logging and writeback (e.g. "deepseek", "anthropic").
    fn name(&self) -> &str;

    /// Default model identifier for this provider (e.g. "deepseek-chat").
    fn default_model(&self) -> &str;

    /// Whether this provider is currently healthy (not rate-limited, not in backoff).
    /// Default: always healthy. Providers can override to implement circuit breaking.
    fn is_healthy(&self) -> bool {
        true
    }
}
