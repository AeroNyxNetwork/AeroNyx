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
//! - Provider response bodies must be consumed through
//!   [`read_bounded_llm_response`]. Never call unbounded `Response::text/json`.
//! - Provider construction must use [`normalize_llm_api_base`] and
//!   [`resolve_llm_api_key`], then [`build_llm_http_client`], so explicitly
//!   configured runtimes fail closed without copying endpoints,
//!   environment-variable names, or secrets into process-health diagnostics.
//!
//! ## Last Modified
//! v2.5.4-StartupIntegrity - [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex]
//!   Added typed, privacy-safe provider initialization errors plus shared API
//!   endpoint and environment-backed key validation.
//! v2.5.3-ResponseBoundary - [LLM-RESPONSE-BOUNDARY 2026-07-30 by Codex]
//!   Added a shared bounded response reader, UTF-8-safe error formatting, and
//!   a monotonic provider cooldown that recovers without a background timer.
//! v2.5.0+SuperNode - 🌟 Created.
//! v2.5.0+Unify     - 🔧 [BUG FIX] Removed duplicate CognitiveTaskType definition.
//!   CognitiveTaskType is now defined ONLY in config_supernode.rs and re-exported
//!   here. The old definition had different variant names (CommunitySummary vs
//!   CommunityNarrative, NaturalSummary vs RecallSynthesis, CustomPrompt vs
//!   ConflictResolution/CodeAnalysis) which caused compilation errors across
//!   task_worker.rs, llm_router.rs, and mod.rs re-exports.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Maximum successful LLM response body retained before JSON parsing.
///
/// This is intentionally much larger than normal cognitive-task output while
/// still preventing a custom or compromised provider from exhausting memory.
pub(super) const MAX_LLM_SUCCESS_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Maximum non-success response body retained for diagnostics.
pub(super) const MAX_LLM_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Maximum API-error bytes rendered through `Display`.
const MAX_LLM_ERROR_DISPLAY_BYTES: usize = 200;

/// Default cooldown after transport, parsing, or provider API failure.
const DEFAULT_LLM_PROVIDER_COOLDOWN: Duration = Duration::from_secs(30);

/// Maximum provider-requested cooldown accepted from `Retry-After`.
const MAX_LLM_PROVIDER_COOLDOWN_SECS: u64 = 5 * 60;

/// Fixed provider request deadline until per-provider limits are introduced.
const LLM_PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

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

/// Privacy-safe reason for rejecting one configured LLM provider at startup.
///
/// [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] These variants are the
/// complete diagnostics boundary used by process health. They intentionally
/// retain no endpoint, provider name, environment-variable name, or secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProviderInitError {
    InvalidApiBase,
    ProviderSecretUnavailable,
    ProviderSecretRequired,
    HttpClientInitializationFailed,
}

impl LlmProviderInitError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::InvalidApiBase => "invalid_api_base",
            Self::ProviderSecretUnavailable => "provider_secret_unavailable",
            Self::ProviderSecretRequired => "provider_secret_required",
            Self::HttpClientInitializationFailed => "http_client_initialization_failed",
        }
    }
}

impl fmt::Display for LlmProviderInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LLM provider initialization failed ({})",
            self.reason_code()
        )
    }
}

impl std::error::Error for LlmProviderInitError {}

/// Canonicalizes and validates a configured provider endpoint.
///
/// Only HTTP(S) hierarchical URLs without embedded credentials, query, or
/// fragment state are accepted. Provider adapters append their fixed API path
/// after this boundary, so accepting those components would make routing
/// ambiguous and could leak credentials through ordinary URL diagnostics.
pub(super) fn normalize_llm_api_base(raw_api_base: &str) -> Result<String, LlmProviderInitError> {
    let normalized = raw_api_base.trim().trim_end_matches('/');
    if normalized.is_empty() {
        return Err(LlmProviderInitError::InvalidApiBase);
    }

    let parsed =
        reqwest::Url::parse(normalized).map_err(|_| LlmProviderInitError::InvalidApiBase)?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.cannot_be_a_base()
    {
        return Err(LlmProviderInitError::InvalidApiBase);
    }

    Ok(normalized.to_owned())
}

/// Resolves an optional `$ENV_VAR` provider key at the startup boundary.
///
/// A missing explicitly referenced environment variable is configuration
/// drift and therefore an error even for keyless OpenAI-compatible endpoints.
/// A literal empty key remains valid only when `required` is false.
pub(super) fn resolve_llm_api_key(
    raw_api_key: &str,
    required: bool,
) -> Result<String, LlmProviderInitError> {
    let api_key = if let Some(var_name) = raw_api_key.strip_prefix('$') {
        if var_name.is_empty() {
            return Err(LlmProviderInitError::ProviderSecretUnavailable);
        }
        std::env::var(var_name).map_err(|_| LlmProviderInitError::ProviderSecretUnavailable)?
    } else {
        raw_api_key.to_owned()
    };

    if required && api_key.trim().is_empty() {
        return Err(LlmProviderInitError::ProviderSecretRequired);
    }

    Ok(api_key)
}

/// Builds the shared provider HTTP transport without implicit host proxies.
///
/// [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Provider routing is an
/// explicit node configuration decision. Consulting OS or environment proxy
/// state could redirect private cognitive traffic to an undeclared endpoint;
/// on some macOS hosts the system proxy adapter can also panic during client
/// construction. Explicit future proxy support must be validated configuration.
pub(super) fn build_llm_http_client() -> Result<reqwest::Client, LlmProviderInitError> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(LLM_PROVIDER_REQUEST_TIMEOUT)
        .build()
        .map_err(|_| LlmProviderInitError::HttpClientInitializationFailed)
}

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
            Self::ApiError { status, body } => {
                write!(
                    f,
                    "API error {}: {}",
                    status,
                    utf8_prefix(body, MAX_LLM_ERROR_DISPLAY_BYTES)
                )
            }
            Self::ParseError(e) => write!(f, "parse error: {}", e),
            Self::EmptyResponse => write!(f, "empty response from model"),
            Self::RateLimit { retry_after_secs } => {
                write!(f, "rate limit hit (retry after: {:?}s)", retry_after_secs)
            }
            Self::ContextTooLong => write!(f, "context too long for this model"),
            Self::NotConfigured(name) => write!(f, "provider '{}' not configured", name),
        }
    }
}

impl std::error::Error for LlmError {}

/// Monotonic provider availability state shared by all HTTP implementations.
///
/// [LLM-PROVIDER-COOLDOWN 2026-07-30 by Codex] A permanent boolean creates a
/// one-way failure latch: once the router skips a provider, no future success
/// can make it healthy. A monotonic deadline permits bounded automatic retry
/// without wall-clock jumps or a background timer.
#[derive(Debug, Default)]
pub(super) struct ProviderHealth {
    unhealthy_until_ms: AtomicU64,
}

impl ProviderHealth {
    pub(super) fn mark_unhealthy(&self) {
        self.mark_unhealthy_for(DEFAULT_LLM_PROVIDER_COOLDOWN);
    }

    pub(super) fn mark_rate_limited(&self, retry_after_secs: Option<u64>) {
        self.mark_unhealthy_for(rate_limit_cooldown(retry_after_secs));
    }

    pub(super) fn mark_healthy(&self) {
        self.mark_healthy_at(monotonic_millis());
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.is_healthy_at(monotonic_millis())
    }

    fn mark_unhealthy_for(&self, duration: Duration) {
        self.mark_unhealthy_at(monotonic_millis(), duration_to_millis(duration));
    }

    fn mark_unhealthy_at(&self, now_ms: u64, duration_ms: u64) {
        let deadline = now_ms.saturating_add(duration_ms.max(1));
        self.unhealthy_until_ms
            .fetch_max(deadline, Ordering::Relaxed);
    }

    fn mark_healthy_at(&self, now_ms: u64) {
        // [LLM-PROVIDER-COOLDOWN 2026-07-30 by Codex] A successful request may
        // have started before a concurrent request recorded a newer failure.
        // Only clear an already-expired deadline so that stale success cannot
        // erase a live cooldown.
        let _ = self.unhealthy_until_ms.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |deadline| (deadline <= now_ms).then_some(0),
        );
    }

    fn is_healthy_at(&self, now_ms: u64) -> bool {
        now_ms >= self.unhealthy_until_ms.load(Ordering::Relaxed)
    }
}

fn monotonic_millis() -> u64 {
    static PROCESS_EPOCH: OnceLock<Instant> = OnceLock::new();
    let elapsed = PROCESS_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis();
    u64::try_from(elapsed).unwrap_or(u64::MAX)
}

fn duration_to_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn rate_limit_cooldown(retry_after_secs: Option<u64>) -> Duration {
    Duration::from_secs(
        retry_after_secs
            .unwrap_or(DEFAULT_LLM_PROVIDER_COOLDOWN.as_secs())
            .clamp(1, MAX_LLM_PROVIDER_COOLDOWN_SECS),
    )
}

/// Size-checked accumulator shared by fixed-length and chunked responses.
struct BoundedLlmBody {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl BoundedLlmBody {
    fn new(max_bytes: usize, content_length: Option<u64>) -> Result<Self, LlmError> {
        if content_length
            .is_some_and(|length| length > u64::try_from(max_bytes).unwrap_or(u64::MAX))
        {
            return Err(response_body_too_large(max_bytes));
        }
        let initial_capacity = content_length
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0)
            .min(max_bytes);

        Ok(Self {
            bytes: Vec::with_capacity(initial_capacity),
            max_bytes,
        })
    }

    fn push(&mut self, chunk: &[u8]) -> Result<(), LlmError> {
        let next_length = self
            .bytes
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| response_body_too_large(self.max_bytes))?;
        if next_length > self.max_bytes {
            return Err(response_body_too_large(self.max_bytes));
        }
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Consume one HTTP response under an explicit byte ceiling.
///
/// [LLM-RESPONSE-BOUNDARY 2026-07-30 by Codex] `Content-Length` is only an
/// early rejection hint because chunked responses can omit or falsify it.
/// Every received chunk is checked again before extending the accumulator.
pub(super) async fn read_bounded_llm_response(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, LlmError> {
    let mut body = BoundedLlmBody::new(max_bytes, response.content_length())?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| LlmError::Transport(error.to_string()))?
    {
        body.push(&chunk)?;
    }

    Ok(body.into_bytes())
}

fn response_body_too_large(max_bytes: usize) -> LlmError {
    LlmError::ParseError(format!("LLM response body exceeds {max_bytes} byte limit"))
}

fn utf8_prefix(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

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
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_display_truncates_on_utf8_boundary() {
        let error = LlmError::ApiError {
            status: 500,
            body: "界".repeat(100),
        };
        let rendered = error.to_string();

        assert!(rendered.starts_with("API error 500: "));
        assert!(rendered.is_char_boundary(rendered.len()));
        assert!(!rendered.contains('\u{fffd}'));
        assert!(rendered.len() <= "API error 500: ".len() + MAX_LLM_ERROR_DISPLAY_BYTES);
    }

    #[test]
    fn utf8_prefix_handles_ascii_unicode_and_zero_budget() {
        assert_eq!(utf8_prefix("abcdef", 3), "abc");
        assert_eq!(utf8_prefix("认证模块", 7), "认证");
        assert_eq!(utf8_prefix("认证模块", 0), "");
        assert_eq!(utf8_prefix("short", usize::MAX), "short");
    }

    #[test]
    fn oversized_response_error_is_bounded_and_stable() {
        let error = response_body_too_large(MAX_LLM_SUCCESS_BODY_BYTES);
        assert!(matches!(error, LlmError::ParseError(_)));
        assert!(error.to_string().contains("exceeds"));
        assert!(error.to_string().contains("8388608"));
    }

    #[test]
    fn bounded_body_accepts_exact_limit_and_rejects_all_overflow_paths() {
        let mut body = BoundedLlmBody::new(5, Some(5)).unwrap();
        body.push(b"ab").unwrap();
        body.push(b"cde").unwrap();
        assert_eq!(body.into_bytes(), b"abcde");

        assert!(BoundedLlmBody::new(5, Some(6)).is_err());

        let mut chunked = BoundedLlmBody::new(5, None).unwrap();
        chunked.push(b"abc").unwrap();
        assert!(chunked.push(b"def").is_err());
        assert_eq!(chunked.into_bytes(), b"abc");
    }

    #[test]
    fn provider_health_recovers_after_monotonic_cooldown() {
        let health = ProviderHealth::default();
        assert!(health.is_healthy_at(100));

        health.mark_unhealthy_at(100, 30_000);
        assert!(!health.is_healthy_at(100));
        assert!(!health.is_healthy_at(30_099));
        assert!(health.is_healthy_at(30_100));

        health.mark_unhealthy_at(200, 1_000);
        assert!(!health.is_healthy_at(30_099));
        health.mark_healthy_at(200);
        assert!(!health.is_healthy_at(200));
        health.mark_healthy_at(30_100);
        assert!(health.is_healthy_at(200));
    }

    #[test]
    fn rate_limit_cooldown_is_clamped_to_operational_bounds() {
        assert_eq!(rate_limit_cooldown(Some(0)), Duration::from_secs(1));
        assert_eq!(rate_limit_cooldown(None), DEFAULT_LLM_PROVIDER_COOLDOWN);
        assert_eq!(
            rate_limit_cooldown(Some(MAX_LLM_PROVIDER_COOLDOWN_SECS + 1)),
            Duration::from_secs(MAX_LLM_PROVIDER_COOLDOWN_SECS)
        );
    }

    #[test]
    fn provider_api_base_rejects_ambiguous_or_secret_bearing_urls() {
        assert_eq!(
            normalize_llm_api_base(" https://api.example.com/v1/// ").unwrap(),
            "https://api.example.com/v1"
        );
        for invalid in [
            "",
            "api.example.com",
            "ftp://api.example.com",
            "https://user:secret@api.example.com",
            "https://api.example.com/v1?token=secret",
            "https://api.example.com/v1#fragment",
        ] {
            assert_eq!(
                normalize_llm_api_base(invalid),
                Err(LlmProviderInitError::InvalidApiBase)
            );
        }
    }

    #[test]
    fn provider_key_resolution_distinguishes_keyless_and_required_modes() {
        assert_eq!(resolve_llm_api_key("", false).unwrap(), "");
        assert_eq!(
            resolve_llm_api_key("", true),
            Err(LlmProviderInitError::ProviderSecretRequired)
        );
        assert_eq!(
            resolve_llm_api_key(
                "$AERONYX_TEST_SUPERNODE_SECRET_MUST_NOT_EXIST_20260814",
                false
            ),
            Err(LlmProviderInitError::ProviderSecretUnavailable)
        );
    }
}
