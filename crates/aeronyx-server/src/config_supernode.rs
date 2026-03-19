// ============================================
// File: crates/aeronyx-server/src/config_supernode.rs
// ============================================
//! # SuperNode Configuration — LLM Cognitive Enhancement Layer
//!
//! ## Creation Reason
//! v2.5.0-SuperNode — Extracted from config.rs to keep MemChainConfig manageable.
//!
//! ## Main Functionality
//! - SuperNodeConfig: top-level supernode config (enabled flag + sub-configs)
//! - ProviderConfig: individual LLM provider definition (API endpoint, model, auth)
//! - ProviderType: enum for provider protocol (OpenAI-compatible vs Anthropic)
//! - CognitiveTaskType: **CANONICAL** enum for the 6 cognitive task types
//!   (SessionTitle, CommunityNarrative, ConflictResolution, RecallSynthesis,
//!    CodeAnalysis, EntityDescription). This is the SINGLE SOURCE OF TRUTH —
//!   llm_provider.rs re-exports this type, NOT the other way around.
//! - TaskRoutingConfig: maps each task type to a named provider
//! - PrivacyLevel: Structured / Summary / Full — controls what data is sent to LLM
//! - PrivacyConfig: controls what data is sent to external LLM APIs
//! - WorkerConfig: async task worker parameters (polling, concurrency, retries)
//! - Validation for all config sections
//!
//! ## Dependencies
//! - Used by config.rs — MemChainConfig embeds SuperNodeConfig as a field
//! - Used by server.rs — initializes LlmRouter + TaskWorker from this config
//! - Used by llm_router.rs — reads provider configs and routing rules
//! - Used by llm_provider.rs — re-exports CognitiveTaskType from here
//! - Used by task_worker.rs — reads worker params (poll interval, concurrency)
//! - Used by reflection.rs — reads privacy config when submitting cognitive tasks
//! - Used by prompts.rs — re-exports PrivacyLevel from here
//!
//! ⚠️ Important Note for Next Developer:
//! - CognitiveTaskType is defined HERE and ONLY here. llm_provider.rs re-exports it.
//!   Do NOT create a second CognitiveTaskType anywhere else.
//! - PrivacyLevel has 3 variants: Structured, Summary, Full.
//!   Summary is treated as Structured by most prompt builders (future enhancement).
//! - SuperNodeConfig::default() returns enabled=false — existing nodes upgrading
//!   to v2.5.0 see ZERO behavior change until explicitly enabled in config.
//! - api_key supports "$ENV_VAR" syntax — resolved at runtime by the provider
//!   implementation (llm_openai.rs / llm_anthropic.rs), NOT during config loading.
//! - ProviderType::OpenaiCompatible covers DeepSeek, OpenAI, Groq, Together,
//!   Ollama, vLLM, and any other OpenAI Chat Completion API compatible endpoint.
//! - TaskRoutingConfig fields are all Option<String>. When None, the fallback
//!   provider is used.
//! - PrivacyConfig.level_for() returns PrivacyLevel (owned, cloned).
//!
//! ## Last Modified
//! v2.5.0-SuperNode - 🌟 Created. Full SuperNode configuration with providers,
//!   routing, privacy, and worker settings.
//! v2.5.0+Audit Fix 9  - 🔧 CognitiveTaskType::from_str renamed to parse() to avoid
//!   shadowing std::str::FromStr trait signature. All callers updated.
//! v2.5.0+Audit Fix 10 - 🔧 validate() now fills in the default Anthropic api_base
//!   ("https://api.anthropic.com") when empty, rather than silently accepting it.
//! v2.5.0+Fix       - 🔧 [BUG FIX] PrivacyConfig::level_for() changed return type
//!   from &PrivacyLevel to PrivacyLevel (owned) to avoid temporary-value lifetime error.
//! v2.5.0+Unify     - 🔧 [BUG FIX] CognitiveTaskType is now the SINGLE canonical
//!   definition. Added task_type_str(), default_privacy_level(), default_priority()
//!   methods (merged from llm_provider.rs duplicate). Removed duplicate in llm_provider.rs.
//!   PrivacyLevel gains Summary variant + from_str()/as_str() methods for DB round-trip.
//!   validate() fixed: from_str → parse. Tests fixed: from_str → parse.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{Result, ServerError};

// ============================================
// CognitiveTaskType — SINGLE SOURCE OF TRUTH
// ============================================

/// The cognitive task types that can be dispatched to LLM providers.
///
/// ⚠️ This is the CANONICAL definition. llm_provider.rs re-exports this type.
/// Do NOT define CognitiveTaskType anywhere else in the codebase.
///
/// Each task type can be routed to a different provider via TaskRoutingConfig.
/// Stored as lowercase strings in the cognitive_tasks table (task_type column).
///
/// ## DB String Mapping
/// - SessionTitle → "session_title"
/// - CommunityNarrative → "community_narrative"
/// - ConflictResolution → "conflict_resolution"
/// - RecallSynthesis → "recall_synthesis"
/// - CodeAnalysis → "code_analysis"
/// - EntityDescription → "entity_description"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveTaskType {
    SessionTitle,
    CommunityNarrative,
    ConflictResolution,
    RecallSynthesis,
    CodeAnalysis,
    /// Entity description enrichment (v2.5.0+SuperNode Phase B, enqueued by Step 9)
    EntityDescription,
}

impl CognitiveTaskType {
    pub const ALL: &'static [CognitiveTaskType] = &[
        Self::SessionTitle,
        Self::CommunityNarrative,
        Self::ConflictResolution,
        Self::RecallSynthesis,
        Self::CodeAnalysis,
        Self::EntityDescription,
    ];

    /// Canonical string for DB storage in `task_type` column.
    /// Also used as routing key in LlmRouter.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SessionTitle => "session_title",
            Self::CommunityNarrative => "community_narrative",
            Self::ConflictResolution => "conflict_resolution",
            Self::RecallSynthesis => "recall_synthesis",
            Self::CodeAnalysis => "code_analysis",
            Self::EntityDescription => "entity_description",
        }
    }

    /// Alias for as_str() — kept for compatibility with code that was written
    /// against the old llm_provider.rs CognitiveTaskType which had task_type_str().
    #[must_use]
    pub fn task_type_str(&self) -> &'static str {
        self.as_str()
    }

    /// Default privacy level string for this task type.
    /// Used when inserting cognitive tasks without explicit privacy override.
    #[must_use]
    pub fn default_privacy_level(&self) -> &'static str {
        match self {
            Self::SessionTitle => "structured",
            Self::CommunityNarrative => "structured",
            Self::EntityDescription => "structured",
            Self::ConflictResolution => "structured",
            Self::RecallSynthesis => "structured",
            Self::CodeAnalysis => "structured",
        }
    }

    /// Default task priority (1-10, higher = processed sooner).
    #[must_use]
    pub fn default_priority(&self) -> i64 {
        match self {
            Self::SessionTitle => 7,
            Self::CommunityNarrative => 5,
            Self::EntityDescription => 4,
            Self::RecallSynthesis => 6,
            Self::ConflictResolution => 5,
            Self::CodeAnalysis => 5,
        }
    }

    /// Parse a task type from its string representation.
    ///
    /// ## v2.5.0+Audit Fix 9
    /// Renamed from `from_str` to `parse` to avoid shadowing the `std::str::FromStr`
    /// trait method, which has a different return type (`Result`, not `Option`).
    ///
    /// All callers use `CognitiveTaskType::parse()`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "session_title" => Some(Self::SessionTitle),
            "community_narrative" => Some(Self::CommunityNarrative),
            "conflict_resolution" => Some(Self::ConflictResolution),
            "recall_synthesis" => Some(Self::RecallSynthesis),
            "code_analysis" => Some(Self::CodeAnalysis),
            "entity_description" => Some(Self::EntityDescription),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    /// OpenAI Chat Completion API compatible endpoint.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    #[serde(default)]
    pub api_base: String,
    #[serde(default)]
    pub api_key: Option<String>,
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

// ============================================
// TaskRoutingConfig
// ============================================

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
    #[serde(default)]
    pub entity_description: Option<String>,
    #[serde(default)]
    pub fallback: Option<String>,
}

impl TaskRoutingConfig {
    #[must_use]
    pub fn provider_for(&self, task_type: CognitiveTaskType) -> Option<&str> {
        let explicit = match task_type {
            CognitiveTaskType::SessionTitle => self.session_title.as_deref(),
            CognitiveTaskType::CommunityNarrative => self.community_narrative.as_deref(),
            CognitiveTaskType::ConflictResolution => self.conflict_resolution.as_deref(),
            CognitiveTaskType::RecallSynthesis => self.recall_synthesis.as_deref(),
            CognitiveTaskType::CodeAnalysis => self.code_analysis.as_deref(),
            CognitiveTaskType::EntityDescription => self.entity_description.as_deref(),
        };
        explicit.or(self.fallback.as_deref())
    }

    fn all_referenced_providers(&self) -> Vec<&str> {
        let fields = [
            self.session_title.as_deref(),
            self.community_narrative.as_deref(),
            self.conflict_resolution.as_deref(),
            self.recall_synthesis.as_deref(),
            self.code_analysis.as_deref(),
            self.entity_description.as_deref(),
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
            entity_description: None,
            fallback: None,
        }
    }
}

// ============================================
// PrivacyLevel
// ============================================

/// Privacy level controlling what data is sent to LLM providers.
///
/// ## Variants
/// - `Structured`: Only metadata (entity names, relation types, IDs).
///   Safe to send to external providers.
/// - `Summary`: Anonymized summary text. Treated as Structured by most prompt
///   builders currently — future enhancement will differentiate.
/// - `Full`: Includes decrypted conversation content. Should only be used
///   with local providers unless user has explicitly consented.
///
/// ## v2.5.0+Unify
/// Summary variant added to match task_worker.rs usage. Previously Summary
/// was only in prompts.rs local definition but not in config_supernode.rs.
/// Now unified: this is the single PrivacyLevel definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyLevel {
    Structured,
    /// Anonymized summary — treated as Structured by most builders.
    /// Future enhancement: pass summary text but no raw conversation.
    Summary,
    Full,
}

impl PrivacyLevel {
    /// Parse from DB string. Unknown values default to Structured.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s {
            "full" => Self::Full,
            "summary" => Self::Summary,
            "structured" => Self::Structured,
            _ => Self::Structured,
        }
    }

    /// Canonical string for DB storage.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Structured => "structured",
            Self::Summary => "summary",
            Self::Full => "full",
        }
    }
}

impl Default for PrivacyLevel {
    fn default() -> Self { Self::Structured }
}

impl std::fmt::Display for PrivacyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================
// PrivacyConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    #[serde(default)]
    pub default_level: PrivacyLevel,
    #[serde(default)]
    pub allow_full_for: Vec<String>,
}

impl PrivacyConfig {
    /// Get the effective privacy level for a given task type.
    ///
    /// Returns owned `PrivacyLevel` to avoid lifetime issues.
    #[must_use]
    pub fn level_for(&self, task_type: CognitiveTaskType) -> PrivacyLevel {
        if self.default_level == PrivacyLevel::Full {
            return PrivacyLevel::Full;
        }
        if self.allow_full_for.iter().any(|t| t == task_type.as_str()) {
            return PrivacyLevel::Full;
        }
        // Return the default level (Structured or Summary)
        self.default_level.clone()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperNodeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub routing: TaskRoutingConfig,
    #[serde(default)]
    pub privacy: PrivacyConfig,
    #[serde(default)]
    pub worker: WorkerConfig,
}

impl SuperNodeConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if self.providers.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.supernode.providers",
                "at least one provider must be configured when supernode is enabled",
            ));
        }

        let mut seen_names: HashSet<String> = HashSet::new();
        for (i, provider) in self.providers.iter().enumerate() {
            let prefix = format!("memchain.supernode.providers[{}]", i);

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

            if provider.api_base.is_empty() {
                match provider.provider_type {
                    ProviderType::OpenaiCompatible => {
                        return Err(ServerError::config_invalid(
                            &format!("{}.api_base", prefix),
                            "api_base is required for openai_compatible providers",
                        ));
                    }
                    ProviderType::Anthropic => {
                        warn!(
                            provider = %provider.name,
                            "[SUPERNODE] Anthropic provider has empty api_base — \
                             will use default https://api.anthropic.com at runtime"
                        );
                    }
                }
            }

            if provider.model.is_empty() {
                return Err(ServerError::config_invalid(
                    &format!("{}.model", prefix),
                    "model cannot be empty",
                ));
            }

            if let Some(temp) = provider.temperature {
                if temp < 0.0 || temp > 2.0 {
                    return Err(ServerError::config_invalid(
                        &format!("{}.temperature", prefix),
                        format!("must be in [0.0, 2.0], got {}", temp),
                    ));
                }
            }

            if let Some(max_t) = provider.max_tokens {
                if max_t == 0 {
                    return Err(ServerError::config_invalid(
                        &format!("{}.max_tokens", prefix),
                        "must be > 0 when set",
                    ));
                }
            }
        }

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

        // v2.5.0+Unify: Fixed from_str → parse (from_str was renamed in Audit Fix 9)
        for task_name in &self.privacy.allow_full_for {
            if CognitiveTaskType::parse(task_name).is_none() {
                warn!(
                    task_type = %task_name,
                    "[SUPERNODE] Unknown task type in privacy.allow_full_for — ignored. \
                     Valid types: session_title, community_narrative, conflict_resolution, \
                     recall_synthesis, code_analysis, entity_description"
                );
            }
        }

        if self.worker.poll_interval_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.supernode.worker.poll_interval_secs",
                "must be >= 1",
            ));
        }

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

    #[must_use]
    pub fn is_enabled(&self) -> bool { self.enabled }

    #[must_use]
    pub fn effective_fallback(&self) -> Option<&str> {
        self.routing.fallback.as_deref()
            .or_else(|| self.providers.first().map(|p| p.name.as_str()))
    }

    #[must_use]
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

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
        let cfg = SuperNodeConfig { enabled: false, providers: Vec::new(), ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_enabled_requires_providers() {
        let cfg = SuperNodeConfig { enabled: true, providers: Vec::new(), ..Default::default() };
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
                api_key: None, model: "test".into(), max_tokens: None, temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_provider_duplicate_name_rejected() {
        let provider = ProviderConfig {
            name: "deepseek".into(), provider_type: ProviderType::OpenaiCompatible,
            api_base: "http://api.deepseek.com/v1".into(), api_key: None,
            model: "deepseek-reasoner".into(), max_tokens: None, temperature: None,
        };
        let cfg = SuperNodeConfig {
            enabled: true, providers: vec![provider.clone(), provider], ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_openai_compat_requires_api_base() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(), provider_type: ProviderType::OpenaiCompatible,
                api_base: String::new(), api_key: None, model: "test-model".into(),
                max_tokens: None, temperature: None,
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
                name: "claude".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: Some("$ANTHROPIC_API_KEY".into()),
                model: "claude-sonnet-4-20250514".into(), max_tokens: None, temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_temperature_out_of_range() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: None, model: "test".into(),
                max_tokens: None, temperature: Some(2.5),
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
                    name: "test".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "test".into(),
                    max_tokens: None, temperature: Some(temp),
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
                name: "test".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: None, model: "test".into(),
                max_tokens: Some(0), temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_routing_unknown_provider_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "deepseek".into(), provider_type: ProviderType::OpenaiCompatible,
                api_base: "http://api.deepseek.com/v1".into(), api_key: None,
                model: "deepseek-reasoner".into(), max_tokens: None, temperature: None,
            }],
            routing: TaskRoutingConfig {
                session_title: Some("nonexistent_provider".into()), ..Default::default()
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
                    name: "deepseek".into(), provider_type: ProviderType::OpenaiCompatible,
                    api_base: "http://api.deepseek.com/v1".into(), api_key: None,
                    model: "deepseek-reasoner".into(), max_tokens: None, temperature: None,
                },
                ProviderConfig {
                    name: "claude".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: Some("$ANTHROPIC_API_KEY".into()),
                    model: "claude-sonnet-4-20250514".into(), max_tokens: None, temperature: None,
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

    #[test]
    fn test_worker_poll_interval_zero_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: None, model: "test".into(),
                max_tokens: None, temperature: None,
            }],
            worker: WorkerConfig { poll_interval_secs: 0, ..Default::default() },
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

    #[test]
    fn test_privacy_level_for_task_returns_owned() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Structured,
            allow_full_for: vec!["session_title".into(), "code_analysis".into()],
        };
        assert_eq!(privacy.level_for(CognitiveTaskType::SessionTitle), PrivacyLevel::Full);
        assert_eq!(privacy.level_for(CognitiveTaskType::CodeAnalysis), PrivacyLevel::Full);
        assert_eq!(privacy.level_for(CognitiveTaskType::CommunityNarrative), PrivacyLevel::Structured);
    }

    #[test]
    fn test_privacy_is_full_allowed() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Structured,
            allow_full_for: vec!["session_title".into()],
        };
        assert!(privacy.is_full_allowed(CognitiveTaskType::SessionTitle));
        assert!(!privacy.is_full_allowed(CognitiveTaskType::CommunityNarrative));
    }

    #[test]
    fn test_privacy_full_default_overrides_all() {
        let privacy = PrivacyConfig { default_level: PrivacyLevel::Full, allow_full_for: Vec::new() };
        for task in CognitiveTaskType::ALL {
            assert!(privacy.is_full_allowed(*task));
        }
    }

    #[test]
    fn test_task_type_parse() {
        // v2.5.0+Unify: all tests use parse(), not from_str
        assert_eq!(CognitiveTaskType::parse("session_title"), Some(CognitiveTaskType::SessionTitle));
        assert_eq!(CognitiveTaskType::parse("entity_description"), Some(CognitiveTaskType::EntityDescription));
        assert_eq!(CognitiveTaskType::parse("community_narrative"), Some(CognitiveTaskType::CommunityNarrative));
        assert_eq!(CognitiveTaskType::parse("unknown"), None);
        assert_eq!(CognitiveTaskType::parse(""), None);
    }

    #[test]
    fn test_task_type_roundtrip() {
        for task in CognitiveTaskType::ALL {
            let s = task.as_str();
            assert_eq!(CognitiveTaskType::parse(s), Some(*task));
            // task_type_str() is an alias for as_str()
            assert_eq!(task.task_type_str(), s);
        }
    }

    #[test]
    fn test_task_type_entity_description() {
        assert_eq!(CognitiveTaskType::parse("entity_description"), Some(CognitiveTaskType::EntityDescription));
        assert_eq!(CognitiveTaskType::EntityDescription.as_str(), "entity_description");
    }

    #[test]
    fn test_task_type_unknown_returns_none() {
        assert!(CognitiveTaskType::parse("unknown_task").is_none());
        assert!(CognitiveTaskType::parse("").is_none());
    }

    #[test]
    fn test_privacy_level_from_str_roundtrip() {
        // v2.5.0+Unify: PrivacyLevel now has from_str/as_str methods
        assert_eq!(PrivacyLevel::from_str("full"), PrivacyLevel::Full);
        assert_eq!(PrivacyLevel::from_str("summary"), PrivacyLevel::Summary);
        assert_eq!(PrivacyLevel::from_str("structured"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::from_str("unknown"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::Full.as_str(), "full");
        assert_eq!(PrivacyLevel::Summary.as_str(), "summary");
        assert_eq!(PrivacyLevel::Structured.as_str(), "structured");
    }

    #[test]
    fn test_effective_fallback() {
        let cfg = SuperNodeConfig {
            providers: vec![
                ProviderConfig { name: "a".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(), max_tokens: None, temperature: None },
                ProviderConfig { name: "b".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(), max_tokens: None, temperature: None },
            ],
            routing: TaskRoutingConfig { fallback: Some("b".into()), ..Default::default() },
            ..Default::default()
        };
        assert_eq!(cfg.effective_fallback(), Some("b"));

        let cfg2 = SuperNodeConfig {
            providers: vec![ProviderConfig { name: "first".into(), provider_type: ProviderType::Anthropic,
                api_base: String::new(), api_key: None, model: "m".into(), max_tokens: None, temperature: None }],
            ..Default::default()
        };
        assert_eq!(cfg2.effective_fallback(), Some("first"));

        assert_eq!(SuperNodeConfig::default().effective_fallback(), None);
    }

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
name = "claude"
type = "anthropic"
api_key = "$ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"

[routing]
session_title = "deepseek"
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
        assert_eq!(cfg.providers.len(), 2);
        assert_eq!(cfg.routing.code_analysis, Some("claude".into()));
        assert_eq!(cfg.privacy.default_level, PrivacyLevel::Structured);
        assert_eq!(cfg.worker.poll_interval_secs, 10);
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
        assert_eq!(cfg.effective_fallback(), Some("ollama"));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_backward_compat_empty() {
        let cfg: SuperNodeConfig = toml::from_str("").unwrap();
        assert!(!cfg.enabled);
        assert!(cfg.validate().is_ok());
    }
}
