// ============================================
// File: crates/aeronyx-server/src/config_supernode.rs
// ============================================
//! # SuperNode Configuration — LLM Cognitive Enhancement Layer
//!
//! ## Creation Reason
//! v2.5.0-SuperNode — Extracted from config.rs to keep MemChainConfig manageable.
//! SuperNode configuration has nested structures (providers array, routing map,
//! privacy settings, worker params) that would add 50+ lines and 15+ fields
//! to the already-large MemChainConfig struct.
//!
//! Splitting follows the same principle as storage.rs → storage_ops.rs / storage_graph.rs:
//! keep files focused on a single responsibility domain.
//!
//! ## Main Functionality
//! - SuperNodeConfig: top-level supernode config (enabled flag + sub-configs)
//! - ProviderConfig: individual LLM provider definition (API endpoint, model, auth)
//! - ProviderType: enum for provider protocol (OpenAI-compatible vs Anthropic)
//! - CognitiveTaskType: enum for the 5 cognitive task types
//! - TaskRoutingConfig: maps each task type to a named provider
//! - PrivacyConfig: controls what data is sent to external LLM APIs
//! - WorkerConfig: async task worker parameters (polling, concurrency, retries)
//! - Validation for all config sections
//!
//! ## Dependencies
//! - Used by config.rs — MemChainConfig embeds SuperNodeConfig as a field
//! - Used by server.rs — initializes LlmRouter + TaskWorker from this config
//! - Used by llm_router.rs — reads provider configs and routing rules
//! - Used by task_worker.rs — reads worker params (poll interval, concurrency)
//! - Used by reflection.rs — reads privacy config when submitting cognitive tasks
//!
//! ## TOML Configuration Example
//! ```toml
//! [memchain.supernode]
//! enabled = true
//!
//! [[memchain.supernode.providers]]
//! name = "deepseek"
//! type = "openai_compatible"
//! api_base = "https://api.deepseek.com/v1"
//! api_key = "$DEEPSEEK_API_KEY"
//! model = "deepseek-reasoner"
//! max_tokens = 2000
//! temperature = 0.6
//!
//! [[memchain.supernode.providers]]
//! name = "local_ollama"
//! type = "openai_compatible"
//! api_base = "http://localhost:11434/v1"
//! model = "deepseek-r1:32b"
//!
//! [[memchain.supernode.providers]]
//! name = "claude"
//! type = "anthropic"
//! api_key = "$ANTHROPIC_API_KEY"
//! model = "claude-sonnet-4-20250514"
//!
//! [memchain.supernode.routing]
//! session_title = "deepseek"
//! community_narrative = "deepseek"
//! conflict_resolution = "deepseek"
//! recall_synthesis = "local_ollama"
//! code_analysis = "claude"
//! fallback = "deepseek"
//!
//! [memchain.supernode.privacy]
//! default_level = "structured"
//! allow_full_for = ["session_title", "code_analysis"]
//!
//! [memchain.supernode.worker]
//! poll_interval_secs = 5
//! max_concurrent = 3
//! max_retries = 3
//! task_timeout_secs = 120
//! ```
//!
//! ## Main Logical Flow
//! 1. TOML `[memchain.supernode]` section deserializes into SuperNodeConfig
//! 2. When SuperNodeConfig.enabled = false (default), entire section is inert —
//!    system behavior is identical to v2.4.0
//! 3. When enabled = true, validate() checks:
//!    a. At least one provider is configured
//!    b. Provider names are unique and non-empty
//!    c. api_base is non-empty for each provider
//!    d. model is non-empty for each provider
//!    e. temperature in [0.0, 2.0] when set
//!    f. All routing references point to existing provider names
//!    g. Worker params are within valid ranges
//! 4. server.rs reads validated config → initializes LlmRouter + TaskWorker
//!
//! ⚠️ Important Note for Next Developer:
//! - SuperNodeConfig::default() returns enabled=false — existing nodes upgrading
//!   to v2.5.0 see ZERO behavior change until explicitly enabled in config.
//! - api_key supports "$ENV_VAR" syntax — resolved at runtime by the provider
//!   implementation (llm_openai.rs / llm_anthropic.rs), NOT during config loading.
//!   The config stores the raw string including the "$" prefix.
//! - ProviderType::OpenaiCompatible covers DeepSeek, OpenAI, Groq, Together,
//!   Ollama, vLLM, and any other OpenAI Chat Completion API compatible endpoint.
//!   Ollama does NOT need a separate provider type — it's OpenAI-compatible.
//! - TaskRoutingConfig fields are all Option<String>. When None, the fallback
//!   provider is used. When fallback is also None, the first configured provider
//!   is used as implicit fallback.
//! - PrivacyConfig.allow_full_for accepts task type names as strings (lowercase).
//!   Invalid task type names are warned during validation but NOT rejected,
//!   to allow forward compatibility with future task types.
//! - CognitiveTaskType is used both in config (routing) and at runtime (task queue).
//!   It derives Serialize/Deserialize for storage in cognitive_tasks.task_type column.
//! - max_concurrent = 0 means "unlimited" (not recommended for external APIs).
//! - task_timeout_secs = 0 means "no timeout" (not recommended for external APIs).
//!
//! ## Last Modified
//! v2.5.0-SuperNode - 🌟 Created. Full SuperNode configuration with providers,
//!   routing, privacy, and worker settings.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{Result, ServerError};

// ============================================
// CognitiveTaskType
// ============================================

/// The 5 cognitive task types that can be dispatched to LLM providers.
///
/// Each task type can be routed to a different provider via TaskRoutingConfig.
/// Stored as lowercase strings in the cognitive_tasks table (task_type column).
///
/// ## Task Descriptions
/// - SessionTitle: Generate a natural-language session title from entities + context
/// - CommunityNarrative: Generate a rich community description from member entities
/// - ConflictResolution: Resolve ambiguous temporal conflicts between knowledge edges
/// - RecallSynthesis: Synthesize a coherent answer from multiple recall candidates
/// - CodeAnalysis: Analyze code artifacts for documentation, patterns, or issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveTaskType {
    SessionTitle,
    CommunityNarrative,
    ConflictResolution,
    RecallSynthesis,
    CodeAnalysis,
}

impl CognitiveTaskType {
    /// All known task types. Used for validation and iteration.
    pub const ALL: &'static [CognitiveTaskType] = &[
        Self::SessionTitle,
        Self::CommunityNarrative,
        Self::ConflictResolution,
        Self::RecallSynthesis,
        Self::CodeAnalysis,
    ];

    /// Convert to the string representation used in TOML routing config
    /// and cognitive_tasks.task_type column.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SessionTitle => "session_title",
            Self::CommunityNarrative => "community_narrative",
            Self::ConflictResolution => "conflict_resolution",
            Self::RecallSynthesis => "recall_synthesis",
            Self::CodeAnalysis => "code_analysis",
        }
    }

    /// Parse from string. Returns None for unknown task types.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "session_title" => Some(Self::SessionTitle),
            "community_narrative" => Some(Self::CommunityNarrative),
            "conflict_resolution" => Some(Self::ConflictResolution),
            "recall_synthesis" => Some(Self::RecallSynthesis),
            "code_analysis" => Some(Self::CodeAnalysis),
            _ => None,
        }
    }
}

impl std::fmt::Display for CognitiveTaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================
// ProviderType
// ============================================

/// LLM provider protocol type.
///
/// - OpenaiCompatible: Covers any endpoint implementing the OpenAI Chat Completion API.
///   This includes DeepSeek, OpenAI, Groq, Together, Ollama, vLLM, and many others.
///   Ollama is NOT a separate type — it speaks OpenAI-compatible protocol.
///
/// - Anthropic: The Anthropic Messages API (different message format from OpenAI).
///   Required for Claude models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    /// OpenAI Chat Completion API compatible endpoint.
    /// Covers: DeepSeek, OpenAI, Groq, Together, Ollama, vLLM, etc.
    OpenaiCompatible,
    /// Anthropic Messages API (Claude models).
    Anthropic,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenaiCompatible => write!(f, "openai_compatible"),
            Self::Anthropic => write!(f, "anthropic"),
        }
    }
}

// ============================================
// ProviderConfig
// ============================================

/// Configuration for a single LLM provider.
///
/// ## API Key Resolution
/// `api_key` supports two formats:
/// - Direct string: `api_key = "sk-abc123..."` — used as-is
/// - Environment variable: `api_key = "$DEEPSEEK_API_KEY"` — resolved at runtime
///   by the provider implementation. The "$" prefix signals env var lookup.
///
/// ## Ollama / Local Models
/// For local Ollama deployments, set:
/// ```toml
/// type = "openai_compatible"
/// api_base = "http://localhost:11434/v1"
/// model = "deepseek-r1:32b"
/// ```
/// No api_key needed — leave it as None or empty string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Unique name for this provider. Referenced by TaskRoutingConfig.
    /// Must be non-empty and unique across all configured providers.
    pub name: String,

    /// Provider protocol type.
    #[serde(rename = "type")]
    pub provider_type: ProviderType,

    /// Base URL for the API endpoint.
    /// - OpenAI-compatible: e.g., "https://api.deepseek.com/v1" or "http://localhost:11434/v1"
    /// - Anthropic: e.g., "https://api.anthropic.com" (default, can be omitted)
    #[serde(default)]
    pub api_base: String,

    /// API authentication key. Optional for local deployments (Ollama).
    /// Supports "$ENV_VAR" syntax for environment variable resolution at runtime.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model identifier to use for completions.
    /// Examples: "deepseek-reasoner", "gpt-4o", "claude-sonnet-4-20250514", "deepseek-r1:32b"
    pub model: String,

    /// Maximum tokens for completion responses.
    /// When None, uses provider default (typically 2000).
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Sampling temperature. Range [0.0, 2.0].
    /// When None, uses provider default (typically 0.6 for cognitive tasks).
    #[serde(default)]
    pub temperature: Option<f32>,
}

// ============================================
// TaskRoutingConfig
// ============================================

/// Maps each cognitive task type to a named provider.
///
/// All fields are optional. When a field is None, the `fallback` provider
/// is used. When `fallback` is also None, the first configured provider
/// in the providers array is used as the implicit fallback.
///
/// ## Example
/// ```toml
/// [memchain.supernode.routing]
/// session_title = "deepseek"          # cheap tasks use DeepSeek
/// community_narrative = "deepseek"
/// conflict_resolution = "deepseek"
/// recall_synthesis = "local_ollama"   # latency-sensitive uses local
/// code_analysis = "claude"            # complex tasks use Claude
/// fallback = "deepseek"              # default for any unrouted task
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRoutingConfig {
    #[serde(default)]
    pub session_title: Option<String>,
    #[serde(default)]
    pub community_narrative: Option<String>,
    #[serde(default)]
    pub conflict_resolution: Option<String>,
    #[serde(default)]
    pub recall_synthesis: Option<String>,
    #[serde(default)]
    pub code_analysis: Option<String>,

    /// Default provider when a task type has no explicit routing.
    /// When None, the first provider in the providers array is used.
    #[serde(default)]
    pub fallback: Option<String>,
}

impl TaskRoutingConfig {
    /// Get the provider name for a given task type.
    /// Returns the explicitly configured provider, or fallback, or None.
    #[must_use]
    pub fn provider_for(&self, task_type: CognitiveTaskType) -> Option<&str> {
        let explicit = match task_type {
            CognitiveTaskType::SessionTitle => self.session_title.as_deref(),
            CognitiveTaskType::CommunityNarrative => self.community_narrative.as_deref(),
            CognitiveTaskType::ConflictResolution => self.conflict_resolution.as_deref(),
            CognitiveTaskType::RecallSynthesis => self.recall_synthesis.as_deref(),
            CognitiveTaskType::CodeAnalysis => self.code_analysis.as_deref(),
        };
        explicit.or(self.fallback.as_deref())
    }

    /// Collect all provider names referenced in routing (including fallback).
    /// Used during validation to ensure all references point to existing providers.
    fn all_referenced_providers(&self) -> Vec<&str> {
        let fields = [
            self.session_title.as_deref(),
            self.community_narrative.as_deref(),
            self.conflict_resolution.as_deref(),
            self.recall_synthesis.as_deref(),
            self.code_analysis.as_deref(),
            self.fallback.as_deref(),
        ];
        fields.iter().filter_map(|f| *f).collect()
    }
}

impl Default for TaskRoutingConfig {
    fn default() -> Self {
        Self {
            session_title: None,
            community_narrative: None,
            conflict_resolution: None,
            recall_synthesis: None,
            code_analysis: None,
            fallback: None,
        }
    }
}

// ============================================
// PrivacyLevel
// ============================================

/// Privacy level for data sent to external LLM APIs.
///
/// - Structured: Only structured/anonymized data (entity names, relation types,
///   basic summaries). No raw conversation content. Safe for external APIs.
/// - Full: Includes decrypted conversation content. Use only for trusted/local
///   providers, or when the user explicitly consents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyLevel {
    /// Only structured data (entity names, types, summaries). No raw conversations.
    Structured,
    /// Full conversation content included. Use for trusted/local providers only.
    Full,
}

impl Default for PrivacyLevel {
    fn default() -> Self { Self::Structured }
}

impl std::fmt::Display for PrivacyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structured => write!(f, "structured"),
            Self::Full => write!(f, "full"),
        }
    }
}

// ============================================
// PrivacyConfig
// ============================================

/// Privacy settings for the SuperNode cognitive enhancement layer.
///
/// Controls what data is sent to external LLM APIs. Default is "structured"
/// (safe — no raw conversation content leaves the node).
///
/// ## Example
/// ```toml
/// [memchain.supernode.privacy]
/// default_level = "structured"
/// allow_full_for = ["session_title", "code_analysis"]
/// ```
///
/// This means: most tasks send only structured data, but session_title and
/// code_analysis tasks are allowed to include raw conversation content
/// (for better quality).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Default privacy level for all task types.
    #[serde(default)]
    pub default_level: PrivacyLevel,

    /// Task types that are allowed to use "full" privacy level.
    /// Values are task type names in lowercase (e.g., "session_title", "code_analysis").
    /// Only effective when default_level is "structured" — if default is "full",
    /// this list is irrelevant (everything is already full).
    #[serde(default)]
    pub allow_full_for: Vec<String>,
}

impl PrivacyConfig {
    /// Get the effective privacy level for a given task type.
    #[must_use]
    pub fn level_for(&self, task_type: CognitiveTaskType) -> &PrivacyLevel {
        if self.default_level == PrivacyLevel::Full {
            return &self.default_level;
        }
        // Default is structured — check if this task type is in the allow list
        if self.allow_full_for.iter().any(|t| t == task_type.as_str()) {
            // Caller should use PrivacyLevel::Full for this task type
            // We return a static reference for the Full variant
            // (safe because PrivacyLevel is a simple enum)
            return &PrivacyLevel::Full;
        }
        &self.default_level
    }

    /// Check if a specific task type is allowed to send full conversation content.
    #[must_use]
    pub fn is_full_allowed(&self, task_type: CognitiveTaskType) -> bool {
        self.default_level == PrivacyLevel::Full
            || self.allow_full_for.iter().any(|t| t == task_type.as_str())
    }
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            default_level: PrivacyLevel::Structured,
            allow_full_for: Vec::new(),
        }
    }
}

// ============================================
// WorkerConfig
// ============================================

/// Async task worker configuration.
///
/// The TaskWorker polls the cognitive_tasks table for pending tasks,
/// dispatches them to LLM providers, and writes back results.
///
/// ## Defaults
/// - poll_interval_secs: 5 (check for new tasks every 5 seconds)
/// - max_concurrent: 3 (up to 3 simultaneous LLM API calls)
/// - max_retries: 3 (retry failed tasks up to 3 times)
/// - task_timeout_secs: 120 (abort if a single task takes > 2 minutes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Seconds between polling the task queue. Must be >= 1.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,

    /// Maximum simultaneous LLM API calls. 0 = unlimited (not recommended).
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Maximum retry attempts for failed tasks. 0 = no retries.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Timeout per individual task in seconds. 0 = no timeout (not recommended).
    #[serde(default = "default_task_timeout")]
    pub task_timeout_secs: u64,
}

fn default_poll_interval() -> u64 { 5 }
fn default_max_concurrent() -> usize { 3 }
fn default_max_retries() -> u32 { 3 }
fn default_task_timeout() -> u64 { 120 }

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: default_poll_interval(),
            max_concurrent: default_max_concurrent(),
            max_retries: default_max_retries(),
            task_timeout_secs: default_task_timeout(),
        }
    }
}

// ============================================
// SuperNodeConfig
// ============================================

/// Top-level SuperNode configuration.
///
/// Embeds into MemChainConfig as:
/// ```rust
/// pub struct MemChainConfig {
///     // ...existing fields...
///     #[serde(default)]
///     pub supernode: SuperNodeConfig,
/// }
/// ```
///
/// ## Backward Compatibility
/// SuperNodeConfig::default() has `enabled = false`. Old config files without
/// a `[memchain.supernode]` section will deserialize to the default (disabled).
/// System behavior is identical to v2.4.0 in this case.
///
/// ## Validation
/// When enabled = false, no validation is performed on sub-configs.
/// When enabled = true, all sub-configs are validated:
/// - At least one provider must be configured
/// - Provider names must be unique and non-empty
/// - Routing references must point to existing providers
/// - Worker params must be in valid ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperNodeConfig {
    /// Master switch. When false, the entire SuperNode system is disabled
    /// and all cognitive tasks fall back to local-only processing (v2.4.0 behavior).
    #[serde(default)]
    pub enabled: bool,

    /// LLM provider definitions. At least one required when enabled = true.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,

    /// Task type → provider routing rules.
    #[serde(default)]
    pub routing: TaskRoutingConfig,

    /// Privacy controls for data sent to external LLM APIs.
    #[serde(default)]
    pub privacy: PrivacyConfig,

    /// Async task worker parameters.
    #[serde(default)]
    pub worker: WorkerConfig,
}

impl SuperNodeConfig {
    /// Validate the SuperNode configuration.
    ///
    /// When enabled = false, all validation is skipped (same pattern as
    /// MemChainConfig's NER/graph/entropy validation blocks).
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // ── Providers validation ──

        if self.providers.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.supernode.providers",
                "at least one provider must be configured when supernode is enabled",
            ));
        }

        let mut seen_names: HashSet<String> = HashSet::new();
        for (i, provider) in self.providers.iter().enumerate() {
            let prefix = format!("memchain.supernode.providers[{}]", i);

            // Name must be non-empty and unique
            if provider.name.is_empty() {
                return Err(ServerError::config_invalid(
                    &format!("{}.name", prefix),
                    "provider name cannot be empty",
                ));
            }
            if !seen_names.insert(provider.name.clone()) {
                return Err(ServerError::config_invalid(
                    &format!("{}.name", prefix),
                    format!("duplicate provider name '{}'", provider.name),
                ));
            }

            // api_base: required for openai_compatible, optional for anthropic (has default)
            if provider.api_base.is_empty() && provider.provider_type == ProviderType::OpenaiCompatible {
                return Err(ServerError::config_invalid(
                    &format!("{}.api_base", prefix),
                    "api_base is required for openai_compatible providers",
                ));
            }

            // Model must be non-empty
            if provider.model.is_empty() {
                return Err(ServerError::config_invalid(
                    &format!("{}.model", prefix),
                    "model cannot be empty",
                ));
            }

            // Temperature range: [0.0, 2.0] when set
            if let Some(temp) = provider.temperature {
                if temp < 0.0 || temp > 2.0 {
                    return Err(ServerError::config_invalid(
                        &format!("{}.temperature", prefix),
                        format!("must be in [0.0, 2.0], got {}", temp),
                    ));
                }
            }

            // max_tokens sanity check: if set, must be > 0
            if let Some(max_t) = provider.max_tokens {
                if max_t == 0 {
                    return Err(ServerError::config_invalid(
                        &format!("{}.max_tokens", prefix),
                        "must be > 0 when set",
                    ));
                }
            }
        }

        // ── Routing validation ──

        let provider_names: HashSet<&str> = self.providers.iter()
            .map(|p| p.name.as_str())
            .collect();

        for referenced in self.routing.all_referenced_providers() {
            if !provider_names.contains(referenced) {
                return Err(ServerError::config_invalid(
                    "memchain.supernode.routing",
                    format!(
                        "references unknown provider '{}'. Available: {:?}",
                        referenced,
                        provider_names.iter().collect::<Vec<_>>()
                    ),
                ));
            }
        }

        // ── Privacy validation ──

        // Warn (but don't reject) unknown task type names in allow_full_for.
        // This preserves forward compatibility with future task types.
        for task_name in &self.privacy.allow_full_for {
            if CognitiveTaskType::from_str(task_name).is_none() {
                warn!(
                    task_type = %task_name,
                    "[SUPERNODE] Unknown task type in privacy.allow_full_for — \
                     will be ignored. Valid types: session_title, community_narrative, \
                     conflict_resolution, recall_synthesis, code_analysis"
                );
            }
        }

        // ── Worker validation ──

        if self.worker.poll_interval_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.supernode.worker.poll_interval_secs",
                "must be >= 1 (seconds between task queue polls)",
            ));
        }

        // Log the effective configuration
        info!(
            providers = self.providers.len(),
            fallback = ?self.routing.fallback,
            privacy = %self.privacy.default_level,
            poll_interval = self.worker.poll_interval_secs,
            max_concurrent = self.worker.max_concurrent,
            "[SUPERNODE] Configuration validated"
        );

        Ok(())
    }

    /// Check whether the SuperNode system is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the effective fallback provider name.
    /// Returns the explicit fallback, or the first provider's name if no fallback is set.
    #[must_use]
    pub fn effective_fallback(&self) -> Option<&str> {
        self.routing.fallback.as_deref()
            .or_else(|| self.providers.first().map(|p| p.name.as_str()))
    }

    /// Find a provider config by name.
    #[must_use]
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Get the provider config for a given task type (follows routing → fallback chain).
    #[must_use]
    pub fn provider_for_task(&self, task_type: CognitiveTaskType) -> Option<&ProviderConfig> {
        let provider_name = self.routing.provider_for(task_type)
            .or_else(|| self.effective_fallback())?;
        self.get_provider(provider_name)
    }
}

impl Default for SuperNodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            providers: Vec::new(),
            routing: TaskRoutingConfig::default(),
            privacy: PrivacyConfig::default(),
            worker: WorkerConfig::default(),
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // Default / Disabled State Tests
    // ========================================

    #[test]
    fn test_default_is_disabled() {
        let cfg = SuperNodeConfig::default();
        assert!(!cfg.enabled);
        assert!(!cfg.is_enabled());
        assert!(cfg.providers.is_empty());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_disabled_skips_all_validation() {
        // Even with invalid sub-configs, disabled = no validation
        let cfg = SuperNodeConfig {
            enabled: false,
            providers: Vec::new(), // would fail if enabled
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ========================================
    // Provider Validation Tests
    // ========================================

    #[test]
    fn test_enabled_requires_providers() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: Vec::new(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_provider_empty_name_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: String::new(),
                provider_type: ProviderType::OpenaiCompatible,
                api_base: "http://localhost:11434/v1".into(),
                api_key: None,
                model: "test".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_provider_duplicate_name_rejected() {
        let provider = ProviderConfig {
            name: "deepseek".into(),
            provider_type: ProviderType::OpenaiCompatible,
            api_base: "http://api.deepseek.com/v1".into(),
            api_key: None,
            model: "deepseek-reasoner".into(),
            max_tokens: None,
            temperature: None,
        };
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![provider.clone(), provider],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_openai_compat_requires_api_base() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::OpenaiCompatible,
                api_base: String::new(), // empty
                api_key: None,
                model: "test-model".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_anthropic_allows_empty_api_base() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "claude".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(), // OK for Anthropic (has default)
                api_key: Some("$ANTHROPIC_API_KEY".into()),
                model: "claude-sonnet-4-20250514".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_provider_empty_model_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: String::new(), // empty
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_temperature_out_of_range() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: "test".into(),
                max_tokens: None,
                temperature: Some(2.5), // > 2.0
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_temperature_negative_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: "test".into(),
                max_tokens: None,
                temperature: Some(-0.1),
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_temperature_boundary_values() {
        for temp in [0.0f32, 1.0, 2.0] {
            let cfg = SuperNodeConfig {
                enabled: true,
                providers: vec![ProviderConfig {
                    name: "test".into(),
                    provider_type: ProviderType::Anthropic,
                    api_base: String::new(),
                    api_key: None,
                    model: "test".into(),
                    max_tokens: None,
                    temperature: Some(temp),
                }],
                ..Default::default()
            };
            assert!(cfg.validate().is_ok(), "temperature {} should be valid", temp);
        }
    }

    #[test]
    fn test_max_tokens_zero_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: "test".into(),
                max_tokens: Some(0),
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ========================================
    // Routing Validation Tests
    // ========================================

    #[test]
    fn test_routing_unknown_provider_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "deepseek".into(),
                provider_type: ProviderType::OpenaiCompatible,
                api_base: "http://api.deepseek.com/v1".into(),
                api_key: None,
                model: "deepseek-reasoner".into(),
                max_tokens: None,
                temperature: None,
            }],
            routing: TaskRoutingConfig {
                session_title: Some("nonexistent_provider".into()), // unknown
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_routing_fallback_unknown_provider_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "deepseek".into(),
                provider_type: ProviderType::OpenaiCompatible,
                api_base: "http://api.deepseek.com/v1".into(),
                api_key: None,
                model: "deepseek-reasoner".into(),
                max_tokens: None,
                temperature: None,
            }],
            routing: TaskRoutingConfig {
                fallback: Some("ghost_provider".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_routing_valid_references() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![
                ProviderConfig {
                    name: "deepseek".into(),
                    provider_type: ProviderType::OpenaiCompatible,
                    api_base: "http://api.deepseek.com/v1".into(),
                    api_key: None,
                    model: "deepseek-reasoner".into(),
                    max_tokens: None,
                    temperature: None,
                },
                ProviderConfig {
                    name: "claude".into(),
                    provider_type: ProviderType::Anthropic,
                    api_base: String::new(),
                    api_key: Some("$ANTHROPIC_API_KEY".into()),
                    model: "claude-sonnet-4-20250514".into(),
                    max_tokens: None,
                    temperature: None,
                },
            ],
            routing: TaskRoutingConfig {
                session_title: Some("deepseek".into()),
                code_analysis: Some("claude".into()),
                fallback: Some("deepseek".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ========================================
    // Worker Validation Tests
    // ========================================

    #[test]
    fn test_worker_poll_interval_zero_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: "test".into(),
                max_tokens: None,
                temperature: None,
            }],
            worker: WorkerConfig {
                poll_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_worker_defaults_valid() {
        let worker = WorkerConfig::default();
        assert_eq!(worker.poll_interval_secs, 5);
        assert_eq!(worker.max_concurrent, 3);
        assert_eq!(worker.max_retries, 3);
        assert_eq!(worker.task_timeout_secs, 120);
    }

    // ========================================
    // Privacy Tests
    // ========================================

    #[test]
    fn test_privacy_default_structured() {
        let privacy = PrivacyConfig::default();
        assert_eq!(privacy.default_level, PrivacyLevel::Structured);
        assert!(privacy.allow_full_for.is_empty());
    }

    #[test]
    fn test_privacy_level_for_task() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Structured,
            allow_full_for: vec!["session_title".into(), "code_analysis".into()],
        };
        assert!(privacy.is_full_allowed(CognitiveTaskType::SessionTitle));
        assert!(privacy.is_full_allowed(CognitiveTaskType::CodeAnalysis));
        assert!(!privacy.is_full_allowed(CognitiveTaskType::CommunityNarrative));
        assert!(!privacy.is_full_allowed(CognitiveTaskType::RecallSynthesis));
    }

    #[test]
    fn test_privacy_full_default_overrides_all() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Full,
            allow_full_for: Vec::new(), // irrelevant when default is full
        };
        assert!(privacy.is_full_allowed(CognitiveTaskType::SessionTitle));
        assert!(privacy.is_full_allowed(CognitiveTaskType::CommunityNarrative));
        assert!(privacy.is_full_allowed(CognitiveTaskType::RecallSynthesis));
    }

    // ========================================
    // CognitiveTaskType Tests
    // ========================================

    #[test]
    fn test_task_type_roundtrip() {
        for task in CognitiveTaskType::ALL {
            let s = task.as_str();
            let parsed = CognitiveTaskType::from_str(s);
            assert_eq!(parsed, Some(*task), "Roundtrip failed for {:?}", task);
        }
    }

    #[test]
    fn test_task_type_unknown_returns_none() {
        assert!(CognitiveTaskType::from_str("unknown_task").is_none());
        assert!(CognitiveTaskType::from_str("").is_none());
    }

    #[test]
    fn test_task_type_display() {
        assert_eq!(CognitiveTaskType::SessionTitle.to_string(), "session_title");
        assert_eq!(CognitiveTaskType::RecallSynthesis.to_string(), "recall_synthesis");
    }

    // ========================================
    // Convenience Method Tests
    // ========================================

    #[test]
    fn test_effective_fallback() {
        // Explicit fallback
        let cfg = SuperNodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "a".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None,
                },
                ProviderConfig {
                    name: "b".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None,
                },
            ],
            routing: TaskRoutingConfig {
                fallback: Some("b".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.effective_fallback(), Some("b"));

        // No explicit fallback → first provider
        let cfg2 = SuperNodeConfig {
            providers: vec![ProviderConfig {
                name: "first".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: None, model: "m".into(),
                max_tokens: None, temperature: None,
            }],
            ..Default::default()
        };
        assert_eq!(cfg2.effective_fallback(), Some("first"));

        // No providers → None
        let cfg3 = SuperNodeConfig::default();
        assert_eq!(cfg3.effective_fallback(), None);
    }

    #[test]
    fn test_provider_for_task_routing() {
        let cfg = SuperNodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "cheap".into(), provider_type: ProviderType::OpenaiCompatible,
                    api_base: "http://test".into(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None,
                },
                ProviderConfig {
                    name: "smart".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None,
                },
            ],
            routing: TaskRoutingConfig {
                code_analysis: Some("smart".into()),
                fallback: Some("cheap".into()),
                ..Default::default()
            },
            ..Default::default()
        };

        // Explicitly routed
        let ca = cfg.provider_for_task(CognitiveTaskType::CodeAnalysis).unwrap();
        assert_eq!(ca.name, "smart");

        // Falls back
        let st = cfg.provider_for_task(CognitiveTaskType::SessionTitle).unwrap();
        assert_eq!(st.name, "cheap");
    }

    #[test]
    fn test_get_provider_by_name() {
        let cfg = SuperNodeConfig {
            providers: vec![ProviderConfig {
                name: "deepseek".into(), provider_type: ProviderType::OpenaiCompatible,
                api_base: "http://test".into(), api_key: None, model: "m".into(),
                max_tokens: None, temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.get_provider("deepseek").is_some());
        assert!(cfg.get_provider("nonexistent").is_none());
    }

    // ========================================
    // TOML Parsing Tests
    // ========================================

    #[test]
    fn test_toml_full_config() {
        let toml_str = r#"
enabled = true

[[providers]]
name = "deepseek"
type = "openai_compatible"
api_base = "https://api.deepseek.com/v1"
api_key = "$DEEPSEEK_API_KEY"
model = "deepseek-reasoner"
max_tokens = 2000
temperature = 0.6

[[providers]]
name = "local_ollama"
type = "openai_compatible"
api_base = "http://localhost:11434/v1"
model = "deepseek-r1:32b"

[[providers]]
name = "claude"
type = "anthropic"
api_key = "$ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"

[routing]
session_title = "deepseek"
community_narrative = "deepseek"
conflict_resolution = "deepseek"
recall_synthesis = "local_ollama"
code_analysis = "claude"
fallback = "deepseek"

[privacy]
default_level = "structured"
allow_full_for = ["session_title", "code_analysis"]

[worker]
poll_interval_secs = 10
max_concurrent = 5
max_retries = 2
task_timeout_secs = 180
"#;
        let cfg: SuperNodeConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.providers.len(), 3);

        // Provider details
        assert_eq!(cfg.providers[0].name, "deepseek");
        assert_eq!(cfg.providers[0].provider_type, ProviderType::OpenaiCompatible);
        assert_eq!(cfg.providers[0].api_base, "https://api.deepseek.com/v1");
        assert_eq!(cfg.providers[0].api_key, Some("$DEEPSEEK_API_KEY".into()));
        assert_eq!(cfg.providers[0].model, "deepseek-reasoner");
        assert_eq!(cfg.providers[0].max_tokens, Some(2000));
        assert!((cfg.providers[0].temperature.unwrap() - 0.6).abs() < f32::EPSILON);

        assert_eq!(cfg.providers[1].name, "local_ollama");
        assert!(cfg.providers[1].api_key.is_none());

        assert_eq!(cfg.providers[2].name, "claude");
        assert_eq!(cfg.providers[2].provider_type, ProviderType::Anthropic);

        // Routing
        assert_eq!(cfg.routing.session_title, Some("deepseek".into()));
        assert_eq!(cfg.routing.code_analysis, Some("claude".into()));
        assert_eq!(cfg.routing.recall_synthesis, Some("local_ollama".into()));
        assert_eq!(cfg.routing.fallback, Some("deepseek".into()));

        // Privacy
        assert_eq!(cfg.privacy.default_level, PrivacyLevel::Structured);
        assert_eq!(cfg.privacy.allow_full_for, vec!["session_title", "code_analysis"]);

        // Worker
        assert_eq!(cfg.worker.poll_interval_secs, 10);
        assert_eq!(cfg.worker.max_concurrent, 5);
        assert_eq!(cfg.worker.max_retries, 2);
        assert_eq!(cfg.worker.task_timeout_secs, 180);

        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_minimal_config() {
        let toml_str = r#"
enabled = true

[[providers]]
name = "ollama"
type = "openai_compatible"
api_base = "http://localhost:11434/v1"
model = "llama3"
"#;
        let cfg: SuperNodeConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.providers.len(), 1);
        // Routing defaults to empty → fallback → first provider
        assert_eq!(cfg.effective_fallback(), Some("ollama"));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_backward_compat_empty() {
        // Empty TOML (old config without supernode section)
        let toml_str = "";
        let cfg: SuperNodeConfig = toml::from_str(toml_str).unwrap();
        assert!(!cfg.enabled);
        assert!(cfg.providers.is_empty());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_disabled_with_partial_config() {
        // User started configuring but left enabled = false
        let toml_str = r#"
enabled = false

[[providers]]
name = "deepseek"
type = "openai_compatible"
api_base = "https://api.deepseek.com/v1"
model = "deepseek-reasoner"
"#;
        let cfg: SuperNodeConfig = toml::from_str(toml_str).unwrap();
        assert!(!cfg.enabled);
        assert_eq!(cfg.providers.len(), 1);
        // Validation passes because enabled = false → skip all checks
        assert!(cfg.validate().is_ok());
    }
}
