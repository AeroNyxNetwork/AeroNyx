// ============================================
// File: crates/aeronyx-server/src/config_supernode.rs
// ============================================
//! # SuperNode Configuration — LLM Cognitive Enhancement Layer
//!
//! ## Creation Reason
//! v2.5.0-SuperNode — Extracted from config.rs to keep MemChainConfig manageable.
//!
//! ## CognitiveTaskType alignment (CRITICAL)
//! config_supernode::CognitiveTaskType MUST use the same string keys as
//! llm_provider::CognitiveTaskType.task_type_str(). The canonical enum for
//! runtime dispatch lives in llm_provider.rs; this file drives TOML routing config.
//!
//! Variant → DB string mapping:
//!   SessionTitle      → "session_title"
//!   CommunitySummary  → "community_summary"
//!   EntityDescription → "entity_description"
//!   NaturalSummary    → "natural_summary"
//!   CustomPrompt      → "custom_prompt"
//!
//! ⚠️ Important Note for Next Developer:
//! - SuperNodeConfig::default() returns enabled=false — zero behavior change on upgrade.
//! - CognitiveTaskType variants here MUST stay in sync with llm_provider::CognitiveTaskType.
//! - PrivacyLevel has three variants: Structured / Summary / Full.
//!   Summary = anonymized text, no raw conversation. Treated as Structured in prompts.rs.
//! - CognitiveTaskType::from_str() is renamed to parse() (Audit Fix 9).
//!
//! ## Last Modified
//! v2.5.0-SuperNode    - 🌟 Created.
//! v2.5.0+Fix          - 🔧 PrivacyConfig::level_for() returns owned PrivacyLevel.
//! v2.5.0+Audit Fix 9  - 🔧 from_str renamed to parse().
//! v2.5.0+Audit Fix 10 - 🔧 validate() warns on empty Anthropic api_base.
//! v2.5.0+Align        - 🔧 CognitiveTaskType variants aligned to llm_provider.rs:
//!   CommunitySummary/NaturalSummary/CustomPrompt replace old narrative/synthesis names.
//!   PrivacyLevel::Summary variant added.
//!   All from_str() references in tests updated to parse().

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{Result, ServerError};

// ============================================
// CognitiveTaskType
// ============================================

/// Cognitive task types for TOML routing config.
///
/// Maps to llm_provider::CognitiveTaskType variants via as_str() / parse().
/// The canonical runtime enum is in llm_provider.rs — keep these in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveTaskType {
    /// Generate a human-readable session title. → "session_title"
    SessionTitle,
    /// Generate community summary from member entities. → "community_summary"
    CommunitySummary,
    /// Generate entity description. → "entity_description"
    EntityDescription,
    /// Generate natural session summary. → "natural_summary"
    NaturalSummary,
    /// Custom/ad-hoc prompt (conflict resolution, code analysis, etc.). → "custom_prompt"
    CustomPrompt,
}

impl CognitiveTaskType {
    pub const ALL: &'static [CognitiveTaskType] = &[
        Self::SessionTitle,
        Self::CommunitySummary,
        Self::EntityDescription,
        Self::NaturalSummary,
        Self::CustomPrompt,
    ];

    /// String representation stored in cognitive_tasks.task_type column.
    /// Must match llm_provider::CognitiveTaskType::task_type_str().
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SessionTitle => "session_title",
            Self::CommunitySummary => "community_summary",
            Self::EntityDescription => "entity_description",
            Self::NaturalSummary => "natural_summary",
            Self::CustomPrompt => "custom_prompt",
        }
    }

    /// Parse from the string stored in cognitive_tasks.task_type.
    /// Renamed from from_str (Audit Fix 9) to avoid shadowing std::str::FromStr.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
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
    OpenaiCompatible,
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
    pub community_summary: Option<String>,
    #[serde(default)]
    pub entity_description: Option<String>,
    #[serde(default)]
    pub natural_summary: Option<String>,
    #[serde(default)]
    pub custom_prompt: Option<String>,
    #[serde(default)]
    pub fallback: Option<String>,
}

impl TaskRoutingConfig {
    #[must_use]
    pub fn provider_for(&self, task_type: CognitiveTaskType) -> Option<&str> {
        let explicit = match task_type {
            CognitiveTaskType::SessionTitle => self.session_title.as_deref(),
            CognitiveTaskType::CommunitySummary => self.community_summary.as_deref(),
            CognitiveTaskType::EntityDescription => self.entity_description.as_deref(),
            CognitiveTaskType::NaturalSummary => self.natural_summary.as_deref(),
            CognitiveTaskType::CustomPrompt => self.custom_prompt.as_deref(),
        };
        explicit.or(self.fallback.as_deref())
    }

    fn all_referenced_providers(&self) -> Vec<&str> {
        [
            self.session_title.as_deref(),
            self.community_summary.as_deref(),
            self.entity_description.as_deref(),
            self.natural_summary.as_deref(),
            self.custom_prompt.as_deref(),
            self.fallback.as_deref(),
        ]
        .iter()
        .filter_map(|f| *f)
        .collect()
    }
}

impl Default for TaskRoutingConfig {
    fn default() -> Self {
        Self {
            session_title: None,
            community_summary: None,
            entity_description: None,
            natural_summary: None,
            custom_prompt: None,
            fallback: None,
        }
    }
}

// ============================================
// PrivacyLevel
// ============================================

/// Privacy level controlling what data is sent to external LLM APIs.
///
/// - Structured: only metadata (entity names, relation types). Safe for external APIs.
/// - Summary: anonymized summary text, no raw conversation. Cloud-provider safe.
/// - Full: full decrypted conversation. Use only with local/trusted providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyLevel {
    Structured,
    Summary,
    Full,
}

impl Default for PrivacyLevel {
    fn default() -> Self { Self::Structured }
}

impl std::fmt::Display for PrivacyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structured => write!(f, "structured"),
            Self::Summary => write!(f, "summary"),
            Self::Full => write!(f, "full"),
        }
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
    /// Get effective privacy level for a task type (returns owned value).
    #[must_use]
    pub fn level_for(&self, task_type: CognitiveTaskType) -> PrivacyLevel {
        if self.default_level == PrivacyLevel::Full {
            return PrivacyLevel::Full;
        }
        if self.allow_full_for.iter().any(|t| t == task_type.as_str()) {
            return PrivacyLevel::Full;
        }
        self.default_level.clone()
    }

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
                             will use https://api.anthropic.com at runtime"
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

        let provider_names: HashSet<&str> =
            self.providers.iter().map(|p| p.name.as_str()).collect();

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

        for task_name in &self.privacy.allow_full_for {
            if CognitiveTaskType::parse(task_name).is_none() {
                warn!(
                    task_type = %task_name,
                    "[SUPERNODE] Unknown task type in privacy.allow_full_for — ignored. \
                     Valid types: session_title, community_summary, entity_description, \
                     natural_summary, custom_prompt"
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
    fn test_openai_compat_requires_api_base() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(), provider_type: ProviderType::OpenaiCompatible,
                api_base: String::new(), api_key: None, model: "test".into(),
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
                custom_prompt: Some("claude".into()),
                fallback: Some("deepseek".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
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
                session_title: Some("nonexistent".into()), ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
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
        let w = WorkerConfig::default();
        assert_eq!(w.poll_interval_secs, 5);
        assert_eq!(w.max_concurrent, 3);
        assert_eq!(w.max_retries, 3);
        assert_eq!(w.task_timeout_secs, 120);
    }

    #[test]
    fn test_privacy_level_has_summary_variant() {
        // Summary variant must exist — prompts.rs uses it
        let l = PrivacyLevel::Summary;
        assert_eq!(l.to_string(), "summary");
        assert_ne!(l, PrivacyLevel::Structured);
        assert_ne!(l, PrivacyLevel::Full);
    }

    #[test]
    fn test_privacy_level_for_task() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Structured,
            allow_full_for: vec!["session_title".into()],
        };
        assert_eq!(privacy.level_for(CognitiveTaskType::SessionTitle), PrivacyLevel::Full);
        assert_eq!(privacy.level_for(CognitiveTaskType::CommunitySummary), PrivacyLevel::Structured);
    }

    #[test]
    fn test_privacy_full_default_overrides_all() {
        let privacy = PrivacyConfig { default_level: PrivacyLevel::Full, allow_full_for: Vec::new() };
        for task in CognitiveTaskType::ALL {
            assert!(privacy.is_full_allowed(*task));
        }
    }

    #[test]
    fn test_task_type_parse_roundtrip() {
        for task in CognitiveTaskType::ALL {
            let s = task.as_str();
            assert_eq!(CognitiveTaskType::parse(s), Some(*task), "roundtrip failed for {:?}", task);
        }
    }

    #[test]
    fn test_task_type_parse_unknown_returns_none() {
        assert!(CognitiveTaskType::parse("unknown_task").is_none());
        assert!(CognitiveTaskType::parse("community_narrative").is_none()); // old name, must fail
        assert!(CognitiveTaskType::parse("conflict_resolution").is_none()); // old name, must fail
        assert!(CognitiveTaskType::parse("").is_none());
    }

    #[test]
    fn test_effective_fallback() {
        let cfg = SuperNodeConfig {
            providers: vec![
                ProviderConfig { name: "a".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None },
                ProviderConfig { name: "b".into(), provider_type: ProviderType::Anthropic,
                    api_base: String::new(), api_key: None, model: "m".into(),
                    max_tokens: None, temperature: None },
            ],
            routing: TaskRoutingConfig { fallback: Some("b".into()), ..Default::default() },
            ..Default::default()
        };
        assert_eq!(cfg.effective_fallback(), Some("b"));

        let cfg2 = SuperNodeConfig {
            providers: vec![ProviderConfig { name: "first".into(),
                provider_type: ProviderType::Anthropic, api_base: String::new(),
                api_key: None, model: "m".into(), max_tokens: None, temperature: None }],
            ..Default::default()
        };
        assert_eq!(cfg2.effective_fallback(), Some("first"));
        assert_eq!(SuperNodeConfig::default().effective_fallback(), None);
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
