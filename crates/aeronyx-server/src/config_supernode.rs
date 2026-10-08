// ============================================
// File: crates/aeronyx-server/src/config_supernode.rs
// ============================================
//! # SuperNode Configuration — LLM Cognitive Enhancement Layer
//!
//! ## Creation Reason
//! v2.5.0-SuperNode — Extracted from config.rs to keep MemChainConfig manageable.
//!
//! ## Main Functionality
//! - SuperNodeConfig: top-level inference config, disabled by default
//! - ProviderConfig: provider endpoint, model, and authentication metadata
//! - ProviderType: legacy variants remain deserializable, but enabled MemChain
//!   inference accepts only the Phala ACI provider.
//! - Server-side SuperNode is restricted to SaaS mode; local/P2P node profiles
//!   must not accept plaintext cognition requests and do not proxy client ACI calls.
//! - CognitiveTaskType: **CANONICAL** enum for the 6 cognitive task types
//!   (SessionTitle, CommunityNarrative, ConflictResolution, RecallSynthesis,
//!    CodeAnalysis, EntityDescription). This is the SINGLE SOURCE OF TRUTH —
//!   llm_provider.rs re-exports this type, NOT the other way around.
//! - TaskRoutingConfig: maps each task type to a named provider
//! - PrivacyLevel: Structured / Summary / Full — controls the task payload sent
//!   to the Phala ACI gateway; it does not imply a blind gateway proxy.
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
//!   implementation. Never print the value or treat ACI attestation as proof
//!   that plaintext is hidden from every gateway/backend component.
//! - [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex] Server-side inference is SaaS-only
//!   and fail-closed: only Phala ACI, canonical Phala HTTPS endpoint, explicit
//!   reviewed compose measurements, and accepted KMS roots are valid.
//! - [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Response proof
//!   follows the Cargo.lock-pinned ACI v1 headers, receipt and session formats;
//!   do not assume fields from a newer, unpinned draft.
//! - [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Enabled SuperNode inference
//!   accepts only Phala ACI providers; legacy adapter variants are rejected.
//! - TaskRoutingConfig fields are all Option<String>. When None, the fallback
//!   provider is used.
//! - PrivacyConfig.level_for() returns PrivacyLevel (owned, cloned).
//!
//! ## Last Modified
//! v2.5.8-CrashLeaseRecovery - [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex]
//!   Defines one overflow-safe stale-claim deadline shared by startup recovery
//!   and the live task worker.
//! v2.5.7-TaskOwnership - [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex]
//!   Rejects task timeout values that cannot share one safe meaning between
//!   Tokio execution deadlines and signed SQLite startup recovery timestamps.
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
/// - EntityExtraction → "entity_extraction"
/// - EntityExtraction → "entity_extraction"
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
    /// Phala-only extraction from user-approved full session content.
    // [MEMCHAIN-PHALA-EXTRACTION 2026-10-05 by Codex]
    EntityExtraction,
}

impl CognitiveTaskType {
    pub const ALL: &'static [CognitiveTaskType] = &[
        Self::SessionTitle,
        Self::CommunityNarrative,
        Self::ConflictResolution,
        Self::RecallSynthesis,
        Self::CodeAnalysis,
        Self::EntityDescription,
        Self::EntityExtraction,
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
            Self::EntityExtraction => "entity_extraction",
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
            Self::EntityExtraction => "full",
            Self::ConflictResolution => "structured",
            // [MEMCHAIN-PHALA-SUMMARY-PRIVACY 2026-10-06 by Codex] Summary
            // synthesis receives only a bounded prior summary and topic labels.
            Self::RecallSynthesis => "summary",
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
            Self::EntityExtraction => 6,
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
            "entity_extraction" => Some(Self::EntityExtraction),
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
    /// Legacy OpenAI Chat Completion adapter; rejected for enabled MemChain inference.
    OpenaiCompatible,
    /// Legacy Anthropic adapter; rejected for enabled MemChain inference.
    Anthropic,
    /// Phala Confidential AI API with ACI-verified upstream routing.
    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    PhalaAci,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenaiCompatible => write!(f, "openai_compatible"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::PhalaAci => write!(f, "phala_aci"),
        }
    }
}

// ============================================
// ProviderConfig
// ============================================

#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    /// For `phala_aci`, the ACI/1 gateway origin or `/v1` base. Responses
    /// must include ACI/1 version, keyset digest, and receipt-id headers;
    /// runtime verification binds the digest to the fresh report and verifies
    /// the signed receipt/session under the Cargo.lock-pinned contract.
    // [MEMCHAIN-PHALA-IDENTITY-BINDING 2026-10-06 by Codex]
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

// [PRIVACY-SAFE-DEBUG 2026-09-23 by Codex] Provider configuration may include
// credentials and private service endpoints, so formatting is constant.
impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProviderConfig(<redacted>)")
    }
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
    /// Phala provider route for consented full-content entity extraction.
    // [MEMCHAIN-PHALA-EXTRACTION 2026-10-05 by Codex]
    #[serde(default)]
    pub entity_extraction: Option<String>,
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
            CognitiveTaskType::EntityExtraction => self.entity_extraction.as_deref(),
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
            self.entity_extraction.as_deref(),
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
            entity_extraction: None,
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
    fn default() -> Self {
        Self::Structured
    }
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

impl WorkerConfig {
    /// Deadline after which a processing row cannot still have a live owner.
    ///
    /// [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex] Recovery waits one complete
    /// poll interval plus one second after the Tokio execution timeout. This
    /// gives the owning worker time to persist its timeout transition before a
    /// different process reclaims the row. Saturation keeps config parsing
    /// backward compatible for extreme legacy values while preserving SQLite's
    /// signed timestamp boundary.
    pub(crate) fn stale_claim_recovery_secs(&self) -> i64 {
        let seconds = self
            .task_timeout_secs
            .saturating_add(self.poll_interval_secs)
            .saturating_add(1);
        i64::try_from(seconds).unwrap_or(i64::MAX)
    }
}

fn default_poll_interval() -> u64 {
    5
}
fn default_max_concurrent() -> usize {
    3
}
fn default_max_retries() -> u32 {
    3
}
fn default_task_timeout() -> u64 {
    120
}

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
    /// Exact ACI-measured Phala compose hashes accepted for confidential inference.
    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    #[serde(default)]
    pub accepted_compose_hashes: Vec<String>,
    /// Operator-reviewed link from each measured compose to its public source or image.
    // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
    #[serde(default)]
    pub accepted_source_provenance: Vec<AcceptedAciSourceProvenance>,
    /// Dstack KMS roots trusted to have custody of the attested ACI receipt key.
    // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex]
    #[serde(default)]
    pub accepted_kms_root_public_keys: Vec<String>,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Explicit Phala ACI provider used for semantic embeddings. Both fields
    /// must be configured together; absent configuration disables embeddings.
    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub routing: TaskRoutingConfig,
    #[serde(default)]
    pub privacy: PrivacyConfig,
    #[serde(default)]
    pub worker: WorkerConfig,
}

/// An operator-reviewed provenance mapping tied to one quote-measured compose.
/// The mapping is relying-party policy, not a cryptographic proof by itself.
// [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedAciSourceProvenance {
    pub compose_hash: String,
    #[serde(default)]
    pub repo_url: Option<String>,
    #[serde(default)]
    pub repo_commit: Option<String>,
    #[serde(default)]
    pub image_digest: Option<String>,
}

impl SuperNodeConfig {
    // [MEMCHAIN-PHALA-NODE-BOUNDARY 2026-10-05 by Codex]
    /// Validate the queue configuration against the process trust boundary.
    /// P2P/local nodes cannot accept plaintext cognition payloads; their clients
    /// call the attested Phala endpoint directly instead.
    pub fn validate_for_mode(&self, is_saas: bool) -> Result<()> {
        if self.enabled && !is_saas {
            return Err(ServerError::config_invalid(
                "memchain.supernode.enabled",
                "server-side cognition tasks are restricted to SaaS mode; ordinary nodes must not handle plaintext prompts",
            ));
        }
        self.validate()
    }

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

        // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
        match (&self.embedding_provider, &self.embedding_model) {
            (None, None) => {}
            (Some(provider_name), Some(model)) => {
                let provider = self.providers.iter().find(|provider| provider.name == *provider_name);
                if provider.is_none_or(|provider| provider.provider_type != ProviderType::PhalaAci) {
                    return Err(ServerError::config_invalid(
                        "memchain.supernode.embedding_provider",
                        "semantic embeddings require an explicitly named phala_aci provider",
                    ));
                }
                if model.trim().is_empty() || model.len() > 256 {
                    return Err(ServerError::config_invalid(
                        "memchain.supernode.embedding_model",
                        "embedding model must contain 1..=256 bytes",
                    ));
                }
            }
            _ => {
                return Err(ServerError::config_invalid(
                    "memchain.supernode.embedding_model",
                    "embedding_provider and embedding_model must be configured together",
                ));
            }
        }

        // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex] ACI defines
        // artifact bindings, not which workload the relying party trusts.
        // Never interpret an empty allowlist as accepting any measured code.
        if self.accepted_compose_hashes.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.supernode.accepted_compose_hashes",
                "at least one reviewed sha256 compose measurement is required",
            ));
        }
        let mut accepted_compose_hashes = HashSet::new();
        for (index, digest) in self.accepted_compose_hashes.iter().enumerate() {
            let Some(hex) = digest.strip_prefix("sha256:") else {
                return Err(ServerError::config_invalid(
                    &format!("memchain.supernode.accepted_compose_hashes[{index}]"),
                    "expected canonical sha256:<64 lowercase hex> measurement",
                ));
            };
            if hex.len() != 64
                || !hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ServerError::config_invalid(
                    &format!("memchain.supernode.accepted_compose_hashes[{index}]"),
                    "expected canonical sha256:<64 lowercase hex> measurement",
                ));
            }
            if !accepted_compose_hashes.insert(digest) {
                return Err(ServerError::config_invalid(
                    "memchain.supernode.accepted_compose_hashes",
                    "duplicate compose measurements are not allowed",
                ));
            }
        }

        // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex] ACI source
        // fields are outside the quote; require the operator's explicit,
        // reviewed mapping from every measured workload to its provenance.
        if self.accepted_source_provenance.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.supernode.accepted_source_provenance",
                "each accepted compose measurement requires reviewed source provenance",
            ));
        }
        let accepted_compose_set: HashSet<&str> =
            self.accepted_compose_hashes.iter().map(String::as_str).collect();
        let mut mapped_compose_hashes = HashSet::new();
        let mut source_mappings = HashSet::new();
        for (index, mapping) in self.accepted_source_provenance.iter().enumerate() {
            let prefix = format!("memchain.supernode.accepted_source_provenance[{index}]");
            let Some(compose_hex) = mapping.compose_hash.strip_prefix("sha256:") else {
                return Err(ServerError::config_invalid(
                    &format!("{prefix}.compose_hash"),
                    "expected canonical sha256:<64 lowercase hex> measurement",
                ));
            };
            if compose_hex.len() != 64
                || !compose_hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                || !accepted_compose_set.contains(mapping.compose_hash.as_str())
            {
                return Err(ServerError::config_invalid(
                    &format!("{prefix}.compose_hash"),
                    "mapping must reference an accepted canonical compose measurement",
                ));
            }
            let repo_mapping = match (
                mapping.repo_url.as_deref(),
                mapping.repo_commit.as_deref(),
            ) {
                (Some(url), Some(revision)) => {
                    let valid_url = reqwest::Url::parse(url).is_ok_and(|parsed| {
                        parsed.scheme() == "https"
                            && parsed.host_str().is_some()
                            && parsed.username().is_empty()
                            && parsed.password().is_none()
                            && parsed.query().is_none()
                            && parsed.fragment().is_none()
                    });
                    valid_url
                        && matches!(revision.len(), 40 | 64)
                        && revision.bytes().all(|byte| {
                            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                        })
                }
                (None, None) => false,
                _ => false,
            };
            let image_mapping = match mapping.image_digest.as_deref() {
                Some(digest) => digest.strip_prefix("sha256:").is_some_and(|hex| {
                    hex.len() == 64
                        && hex
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                }),
                None => false,
            };
            let repo_fields_well_formed = match (
                mapping.repo_url.as_deref(),
                mapping.repo_commit.as_deref(),
            ) {
                (None, None) => true,
                (Some(_), Some(_)) => repo_mapping,
                _ => false,
            };
            let image_field_well_formed = mapping.image_digest.is_none() || image_mapping;
            if !repo_fields_well_formed
                || !image_field_well_formed
                || (!repo_mapping && !image_mapping)
            {
                return Err(ServerError::config_invalid(
                    &prefix,
                    "mapping requires a complete HTTPS repository revision or sha256 image digest",
                ));
            }
            if !source_mappings.insert(mapping) {
                return Err(ServerError::config_invalid(
                    "memchain.supernode.accepted_source_provenance",
                    "duplicate source-provenance mappings are not allowed",
                ));
            }
            mapped_compose_hashes.insert(mapping.compose_hash.as_str());
        }
        if mapped_compose_hashes != accepted_compose_set {
            return Err(ServerError::config_invalid(
                "memchain.supernode.accepted_source_provenance",
                "every accepted compose measurement must have a source-provenance mapping",
            ));
        }

        // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex] A valid TDX quote
        // binds workload keys, but this deployment policy must separately
        // decide which Dstack KMS roots may attest the receipt signer.
        if self.accepted_kms_root_public_keys.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.supernode.accepted_kms_root_public_keys",
                "at least one reviewed compressed secp256k1 KMS root is required",
            ));
        }
        let mut accepted_kms_roots = HashSet::new();
        for (index, root) in self.accepted_kms_root_public_keys.iter().enumerate() {
            let Some(hex) = root.strip_prefix("0x") else {
                return Err(ServerError::config_invalid(
                    &format!("memchain.supernode.accepted_kms_root_public_keys[{index}]"),
                    "expected canonical 0x-prefixed compressed secp256k1 public key",
                ));
            };
            if hex.len() != 66
                || !(hex.starts_with("02") || hex.starts_with("03"))
                || !hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ServerError::config_invalid(
                    &format!("memchain.supernode.accepted_kms_root_public_keys[{index}]"),
                    "expected canonical 0x-prefixed compressed secp256k1 public key",
                ));
            }
            if !accepted_kms_roots.insert(root) {
                return Err(ServerError::config_invalid(
                    "memchain.supernode.accepted_kms_root_public_keys",
                    "duplicate KMS roots are not allowed",
                ));
            }
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
                    ProviderType::PhalaAci => {}
                }
            }

            // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] A configured
            // ordinary provider must never become a MemChain inference path.
            if provider.provider_type != ProviderType::PhalaAci {
                return Err(ServerError::config_invalid(
                    &format!("{}.type", prefix),
                    "MemChain inference requires Phala ACI providers",
                ));
            }

            // [MEMCHAIN-PHALA-ENDPOINT-CONTRACT 2026-10-05 by Codex]
            // Reject endpoint drift during config validation, not only while
            // constructing the runtime provider.
            let phala_api_base: &str = if provider.api_base.is_empty() {
                crate::services::memchain::llm_provider::PHALA_ACI_API_BASE_DEFAULT
            } else {
                &provider.api_base
            };
            if crate::services::memchain::llm_provider::validate_phala_aci_api_base(
                phala_api_base,
            )
            .is_err()
            {
                return Err(ServerError::config_invalid(
                    &format!("{}.api_base", prefix),
                    "must use https://inference.phala.com with an optional /v1 path",
                ));
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

        // v2.5.0+Unify: Fixed from_str → parse (from_str was renamed in Audit Fix 9)
        for task_name in &self.privacy.allow_full_for {
            if CognitiveTaskType::parse(task_name).is_none() {
                warn!(
                    task_type = %task_name,
                    "[SUPERNODE] Unknown task type in privacy.allow_full_for — ignored. \
                     Valid types: session_title, community_narrative, conflict_resolution, \
                     recall_synthesis, code_analysis, entity_description, entity_extraction"
                );
            }
        }

        if self.worker.poll_interval_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.supernode.worker.poll_interval_secs",
                "must be >= 1",
            ));
        }
        // [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex] The same timeout is
        // used by Tokio for live task execution and converted to i64 seconds by
        // SQLite startup recovery. Reject values whose meanings would diverge
        // across those two ownership boundaries.
        if self.worker.task_timeout_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.supernode.worker.task_timeout_secs",
                "must be >= 1",
            ));
        }
        if self.worker.task_timeout_secs > i64::MAX as u64 {
            return Err(ServerError::config_invalid(
                "memchain.supernode.worker.task_timeout_secs",
                format!("must be <= {}", i64::MAX),
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
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[must_use]
    pub fn effective_fallback(&self) -> Option<&str> {
        self.routing
            .fallback
            .as_deref()
            .or_else(|| self.providers.first().map(|p| p.name.as_str()))
    }

    #[must_use]
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    #[must_use]
    pub fn provider_for_task(&self, task_type: CognitiveTaskType) -> Option<&ProviderConfig> {
        let provider_name = self
            .routing
            .provider_for(task_type)
            .or_else(|| self.effective_fallback())?;
        self.get_provider(provider_name)
    }
}

impl Default for SuperNodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            accepted_compose_hashes: Vec::new(),
            accepted_source_provenance: Vec::new(),
            accepted_kms_root_public_keys: Vec::new(),
            providers: Vec::new(),
            embedding_provider: None,
            embedding_model: None,
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

    // [MEMCHAIN-PHALA-SUMMARY-PRIVACY 2026-10-06 by Codex]
    #[test]
    fn recall_synthesis_defaults_to_summary_privacy() {
        assert_eq!(
            CognitiveTaskType::RecallSynthesis.default_privacy_level(),
            "summary"
        );
    }

    #[test]
    fn provider_debug_does_not_expose_credentials_or_endpoint_metadata() {
        let provider = ProviderConfig {
            name: "provider-name-marker-8b36".into(),
            provider_type: ProviderType::OpenaiCompatible,
            api_base: "https://private-endpoint-marker.invalid/v1".into(),
            api_key: Some("api-key-marker-764f".into()),
            model: "model-marker-a291".into(),
            max_tokens: Some(91_337),
            temperature: Some(1.75),
        };

        let debug = format!("{provider:?}");
        assert_eq!(debug, "ProviderConfig(<redacted>)");
        for marker in [
            provider.name.as_str(),
            provider.api_base.as_str(),
            provider.api_key.as_deref().unwrap(),
            provider.model.as_str(),
            "91337",
            "1.75",
        ] {
            assert!(!debug.contains(marker), "leaked marker: {marker}");
        }
    }

    #[test]
    fn test_default_is_disabled() {
        let cfg = SuperNodeConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.accepted_compose_hashes.is_empty());
        assert!(!cfg.is_enabled());
        assert!(cfg.providers.is_empty());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_disabled_skips_all_validation() {
        let cfg = SuperNodeConfig {
            enabled: false,
            providers: Vec::new(),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_enabled_requires_providers() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: Vec::new(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_provider_empty_name_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: String::new(),
                provider_type: ProviderType::PhalaAci,
                api_base: String::new(),
                api_key: Some("$PHALA_API_KEY".into()),
                model: "test".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_provider_duplicate_name_rejected() {
        let provider = ProviderConfig {
            name: "phala".into(),
            provider_type: ProviderType::PhalaAci,
            api_base: String::new(),
            api_key: Some("$PHALA_API_KEY".into()),
            model: "confidential-model".into(),
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
                api_base: String::new(),
                api_key: None,
                model: "test-model".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_ordinary_provider_rejected_even_with_valid_endpoint() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "ordinary".into(),
                provider_type: ProviderType::OpenaiCompatible,
                api_base: "https://api.example.invalid/v1".into(),
                api_key: Some("$ORDINARY_API_KEY".into()),
                model: "ordinary-model".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    fn test_phala_aci_allows_default_api_base() {
        let cfg = SuperNodeConfig {
            enabled: true,
            accepted_compose_hashes: vec![format!("sha256:{}", "a".repeat(64))],
            accepted_source_provenance: vec![AcceptedAciSourceProvenance {
                compose_hash: format!("sha256:{}", "a".repeat(64)),
                repo_url: Some("https://example.invalid/phala-gateway".into()),
                repo_commit: Some("0123456789abcdef0123456789abcdef01234567".into()),
                image_digest: None,
            }],
            accepted_kms_root_public_keys: vec![format!("0x02{}", "a".repeat(64))],
            providers: vec![ProviderConfig {
                name: "phala".into(),
                provider_type: ProviderType::PhalaAci,
                api_base: String::new(),
                api_key: Some("$PHALA_API_KEY".into()),
                model: "confidential-model".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
        // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
        let mut missing_provenance_mapping = cfg.clone();
        missing_provenance_mapping.accepted_source_provenance.clear();
        assert!(missing_provenance_mapping.validate().is_err());
        let mut mismatched_compose_mapping = cfg.clone();
        mismatched_compose_mapping.accepted_source_provenance[0].compose_hash =
            format!("sha256:{}", "b".repeat(64));
        assert!(mismatched_compose_mapping.validate().is_err());
        let mut malformed_source_mapping = cfg.clone();
        malformed_source_mapping.accepted_source_provenance[0].repo_commit =
            Some("not-a-revision".into());
        assert!(malformed_source_mapping.validate().is_err());
    }

    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    #[test]
    fn test_phala_measurement_policy_is_explicit_and_canonical() {
        let valid_provider = || ProviderConfig {
            name: "phala".into(),
            provider_type: ProviderType::PhalaAci,
            api_base: String::new(),
            api_key: Some("$PHALA_API_KEY".into()),
            model: "confidential-model".into(),
            max_tokens: None,
            temperature: None,
        };
        for accepted_compose_hashes in [
            Vec::new(),
            vec!["sha256:ABC".to_string()],
            vec![format!("sha256:{}", "A".repeat(64))],
            vec![
                format!("sha256:{}", "b".repeat(64)),
                format!("sha256:{}", "b".repeat(64)),
            ],
        ] {
            let cfg = SuperNodeConfig {
                enabled: true,
                accepted_compose_hashes,
                accepted_kms_root_public_keys: vec![format!("0x02{}", "a".repeat(64))],
                providers: vec![valid_provider()],
                ..Default::default()
            };
            assert!(cfg.validate().is_err());
        }
    }

    // [MEMCHAIN-PHALA-ENDPOINT-CONTRACT 2026-10-05 by Codex]
    #[test]
    fn test_phala_aci_rejects_noncanonical_endpoint_during_config_validation() {
        for api_base in [
            "http://inference.phala.com/v1",
            "https://example.invalid/v1",
            "https://inference.phala.com/v1?target=elsewhere",
            "https://user@inference.phala.com/v1",
            "https://inference.phala.com/custom",
        ] {
            let cfg = SuperNodeConfig {
                enabled: true,
                providers: vec![ProviderConfig {
                    name: "phala".into(),
                    provider_type: ProviderType::PhalaAci,
                    api_base: api_base.into(),
                    api_key: Some("$PHALA_API_KEY".into()),
                    model: "confidential-model".into(),
                    max_tokens: None,
                    temperature: None,
                }],
                ..Default::default()
            };
            assert!(cfg.validate().is_err(), "accepted endpoint: {api_base}");
        }
    }

    #[test]
    fn test_temperature_out_of_range() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::PhalaAci,
                api_base: String::new(),
                api_key: None,
                model: "test".into(),
                max_tokens: None,
                temperature: Some(2.5),
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
                    provider_type: ProviderType::PhalaAci,
                    api_base: String::new(),
                    api_key: None,
                    model: "test".into(),
                    max_tokens: None,
                    temperature: Some(temp),
                }],
                ..Default::default()
            };
            assert!(
                cfg.validate().is_ok(),
                "temperature {} should be valid",
                temp
            );
        }
    }

    #[test]
    fn test_max_tokens_zero_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "test".into(),
                provider_type: ProviderType::PhalaAci,
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

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_routing_unknown_provider_rejected() {
        let cfg = SuperNodeConfig {
            enabled: true,
            providers: vec![ProviderConfig {
                name: "deepseek".into(),
                provider_type: ProviderType::PhalaAci,
                api_base: String::new(),
                api_key: Some("$PHALA_API_KEY".into()),
                model: "confidential-model".into(),
                max_tokens: None,
                temperature: None,
            }],
            routing: TaskRoutingConfig {
                session_title: Some("nonexistent_provider".into()),
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
                    name: "phala-primary".into(),
                    provider_type: ProviderType::PhalaAci,
                    api_base: String::new(),
                    api_key: Some("$PHALA_API_KEY".into()),
                    model: "confidential-model-a".into(),
                    max_tokens: None,
                    temperature: None,
                },
                ProviderConfig {
                    name: "phala-fallback".into(),
                    provider_type: ProviderType::PhalaAci,
                    api_base: String::new(),
                    api_key: Some("$PHALA_API_KEY".into()),
                    model: "confidential-model-b".into(),
                    max_tokens: None,
                    temperature: None,
                },
            ],
            routing: TaskRoutingConfig {
                session_title: Some("phala-primary".into()),
                code_analysis: Some("phala-fallback".into()),
                fallback: Some("phala-primary".into()),
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
                name: "test".into(),
                provider_type: ProviderType::PhalaAci,
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
    fn test_worker_task_timeout_out_of_range_rejected() {
        for task_timeout_secs in [0, i64::MAX as u64 + 1] {
            let cfg = SuperNodeConfig {
                enabled: true,
                providers: vec![ProviderConfig {
                    name: "test".into(),
                    provider_type: ProviderType::PhalaAci,
                    api_base: String::new(),
                    api_key: None,
                    model: "test".into(),
                    max_tokens: None,
                    temperature: None,
                }],
                worker: WorkerConfig {
                    task_timeout_secs,
                    ..Default::default()
                },
                ..Default::default()
            };
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn test_worker_defaults_valid() {
        let worker = WorkerConfig::default();
        assert_eq!(worker.poll_interval_secs, 5);
        assert_eq!(worker.max_concurrent, 3);
        assert_eq!(worker.max_retries, 3);
        assert_eq!(worker.task_timeout_secs, 120);
        assert_eq!(worker.stale_claim_recovery_secs(), 126);
    }

    #[test]
    fn stale_claim_recovery_deadline_saturates_to_sqlite_timestamp_range() {
        // [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex] Legacy/extreme values
        // must not wrap when timeout, poll interval, and settlement grace add.
        let worker = WorkerConfig {
            poll_interval_secs: u64::MAX,
            task_timeout_secs: u64::MAX,
            ..WorkerConfig::default()
        };
        assert_eq!(worker.stale_claim_recovery_secs(), i64::MAX);
    }

    #[test]
    fn test_privacy_level_for_task_returns_owned() {
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Structured,
            allow_full_for: vec!["session_title".into(), "code_analysis".into()],
        };
        assert_eq!(
            privacy.level_for(CognitiveTaskType::SessionTitle),
            PrivacyLevel::Full
        );
        assert_eq!(
            privacy.level_for(CognitiveTaskType::CodeAnalysis),
            PrivacyLevel::Full
        );
        assert_eq!(
            privacy.level_for(CognitiveTaskType::CommunityNarrative),
            PrivacyLevel::Structured
        );
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
        let privacy = PrivacyConfig {
            default_level: PrivacyLevel::Full,
            allow_full_for: Vec::new(),
        };
        for task in CognitiveTaskType::ALL {
            assert!(privacy.is_full_allowed(*task));
        }
    }

    #[test]
    fn test_task_type_parse() {
        // v2.5.0+Unify: all tests use parse(), not from_str
        assert_eq!(
            CognitiveTaskType::parse("session_title"),
            Some(CognitiveTaskType::SessionTitle)
        );
        assert_eq!(
            CognitiveTaskType::parse("entity_description"),
            Some(CognitiveTaskType::EntityDescription)
        );
        assert_eq!(
            CognitiveTaskType::parse("community_narrative"),
            Some(CognitiveTaskType::CommunityNarrative)
        );
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
        assert_eq!(
            CognitiveTaskType::parse("entity_description"),
            Some(CognitiveTaskType::EntityDescription)
        );
        assert_eq!(
            CognitiveTaskType::EntityDescription.as_str(),
            "entity_description"
        );
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
        assert_eq!(
            PrivacyLevel::from_str("structured"),
            PrivacyLevel::Structured
        );
        assert_eq!(PrivacyLevel::from_str("unknown"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::Full.as_str(), "full");
        assert_eq!(PrivacyLevel::Summary.as_str(), "summary");
        assert_eq!(PrivacyLevel::Structured.as_str(), "structured");
    }

    #[test]
    fn test_effective_fallback() {
        let cfg = SuperNodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "a".into(),
                    provider_type: ProviderType::Anthropic,
                    api_base: String::new(),
                    api_key: None,
                    model: "m".into(),
                    max_tokens: None,
                    temperature: None,
                },
                ProviderConfig {
                    name: "b".into(),
                    provider_type: ProviderType::Anthropic,
                    api_base: String::new(),
                    api_key: None,
                    model: "m".into(),
                    max_tokens: None,
                    temperature: None,
                },
            ],
            routing: TaskRoutingConfig {
                fallback: Some("b".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.effective_fallback(), Some("b"));

        let cfg2 = SuperNodeConfig {
            providers: vec![ProviderConfig {
                name: "first".into(),
                provider_type: ProviderType::Anthropic,
                api_base: String::new(),
                api_key: None,
                model: "m".into(),
                max_tokens: None,
                temperature: None,
            }],
            ..Default::default()
        };
        assert_eq!(cfg2.effective_fallback(), Some("first"));

        assert_eq!(SuperNodeConfig::default().effective_fallback(), None);
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_toml_full_config() {
        let toml_str = r#"
enabled = true
accepted_compose_hashes = ["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
# [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
accepted_source_provenance = [{ compose_hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", repo_url = "https://example.invalid/phala-gateway", repo_commit = "0123456789abcdef0123456789abcdef01234567" }]
accepted_kms_root_public_keys = ["0x02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]

[[providers]]
name = "phala-primary"
type = "phala_aci"
api_key = "$PHALA_API_KEY"
model = "confidential-model-a"
max_tokens = 2000
temperature = 0.6

[[providers]]
name = "phala-fallback"
type = "phala_aci"
api_key = "$PHALA_API_KEY"
model = "confidential-model-b"

[routing]
session_title = "phala-primary"
code_analysis = "phala-fallback"
fallback = "phala-primary"

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
        assert_eq!(cfg.routing.code_analysis, Some("phala-fallback".into()));
        assert_eq!(cfg.privacy.default_level, PrivacyLevel::Structured);
        assert_eq!(cfg.worker.poll_interval_secs, 10);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_minimal_config() {
        let toml_str = r#"
enabled = true
accepted_compose_hashes = ["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
# [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
accepted_source_provenance = [{ compose_hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", repo_url = "https://example.invalid/phala-gateway", repo_commit = "0123456789abcdef0123456789abcdef01234567" }]
accepted_kms_root_public_keys = ["0x02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]

[[providers]]
name = "phala"
type = "phala_aci"
api_key = "$PHALA_API_KEY"
model = "confidential-model"
"#;
        let cfg: SuperNodeConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.effective_fallback(), Some("phala"));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_toml_backward_compat_empty() {
        let cfg: SuperNodeConfig = toml::from_str("").unwrap();
        assert!(!cfg.enabled);
        assert!(cfg.validate().is_ok());
    }
}
