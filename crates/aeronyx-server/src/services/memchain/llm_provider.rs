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
//! - `CognitiveTaskType` — enum of all task types the SuperNode can execute
//! - `LlmError` — structured error type for provider failures
//!
//! ## Design Decisions
//! - `LlmProvider` is object-safe (`async_trait` macro expands to boxed futures)
//! - `TokenUsage` intentionally omits `cost_usd` — fee rates change; compute at query time
//! - `CognitiveTaskType` maps to `cognitive_tasks.task_type` TEXT column
//! - All types derive `serde::Serialize/Deserialize` for JSON storage in DB
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding a new task type, add a variant to `CognitiveTaskType` AND
//!   update `task_type_str()` AND add a handler branch in `TaskWorker`.
//! - `ChatRequest::system` is optional. For task types that don't need a system
//!   prompt (simple completion), leave it None.
//! - `LlmProvider::chat()` must be cancel-safe — the caller may drop the future
//!   if the task is cancelled.
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.

use std::fmt;

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

// ============================================
// CognitiveTaskType
// ============================================

/// All task types the SuperNode LLM worker can execute.
///
/// Maps to `cognitive_tasks.task_type` TEXT column. Use `task_type_str()` to
/// get the canonical string representation for DB storage.
///
/// ## Privacy Levels per Task Type
/// - `SessionTitle`: structured (only session_id + top entity names)
/// - `CommunitySummary`: structured (only entity names + types)
/// - `EntityDescription`: structured (only entity name + type + relation names)
/// - `NaturalSummary`: summary (anonymized summary text)
/// - `CustomPrompt`: full (caller-provided prompt, may contain raw content)
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveTaskType {
    /// Generate a human-readable session title from entity names.
    /// Target: sessions.title
    SessionTitle,
    /// Generate a natural language community summary from member entity names.
    /// Target: communities.summary
    CommunitySummary,
    /// Generate a concise entity description from name, type, and relations.
    /// Target: entities.description
    EntityDescription,
    /// Generate a natural language session summary (replaces entity-list summary).
    /// Target: sessions.summary
    NaturalSummary,
    /// Custom prompt for ad-hoc enrichment (e.g. key decision extraction).
    /// Target: specified in task payload
    CustomPrompt,
}

impl CognitiveTaskType {
    /// Canonical string representation for DB storage in `task_type` column.
    pub fn task_type_str(&self) -> &'static str {
        match self {
            Self::SessionTitle => "session_title",
            Self::CommunitySummary => "community_summary",
            Self::EntityDescription => "entity_description",
            Self::NaturalSummary => "natural_summary",
            Self::CustomPrompt => "custom_prompt",
        }
    }

    /// Default privacy level for this task type.
    pub fn default_privacy_level(&self) -> &'static str {
        match self {
            Self::SessionTitle => "structured",
            Self::CommunitySummary => "structured",
            Self::EntityDescription => "structured",
            Self::NaturalSummary => "summary",
            Self::CustomPrompt => "full",
        }
    }

    /// Default task priority (1-10, higher = processed sooner).
    pub fn default_priority(&self) -> i64 {
        match self {
            Self::SessionTitle => 7,       // Users see this in search results — high priority
            Self::CommunitySummary => 5,   // Background enrichment — medium
            Self::EntityDescription => 4,  // Graph quality — medium-low
            Self::NaturalSummary => 6,     // Session recall quality — medium-high
            Self::CustomPrompt => 5,       // Caller specifies
        }
    }

    /// Parse from the `task_type` TEXT stored in cognitive_tasks.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "session_title" => Some(Self::SessionTitle),
            "community_summary" => Some(Self::CommunitySummary),
            "entity_description" => Some(Self::EntityDescription),
            "natural_summary" => Some(Self::NaturalSummary),
            "custom_prompt" => Some(Self::CustomPrompt),
            _ => None,
        }
    }
}

impl fmt::Display for CognitiveTaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.task_type_str())
    }
}
