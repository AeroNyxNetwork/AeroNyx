// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration
//!
//! ## Creation Reason
//! Central configuration for all AeroNyx server subsystems, loaded from TOML.
//!
//! ## Modification Reason
//! v2.1+MVF+Auth — Added `api_secret` field to MemChainConfig for MPI Bearer token auth.
//! v2.3.0+RemoteStorage — 🌟 Added `allow_remote_storage` and `max_remote_owners` fields
//!   to MemChainConfig for Phase 1 remote MPI Gateway support.
//!   Also added validation for `mvf_alpha`, `miner_interval_secs`, `embed_dim`,
//!   and `embed_max_tokens` (bug fixes from code review).
//! v2.4.0-GraphCognition — 🌟 Added cognitive graph configuration:
//!   - NER engine: ner_enabled, ner_model_path, ner_tokenizer_path, ner_confidence_threshold
//!   - Knowledge graph: graph_enabled, graph_max_depth, graph_max_nodes_per_hop, graph_min_edge_weight
//!   - Entropy filter: entropy_filter_enabled, entropy_filter_threshold, entropy_window_size, entropy_window_overlap
//!   - Miner cognitive steps: miner_entity_extraction, miner_community_detection,
//!     miner_session_summary, miner_artifact_extraction, miner_llm_endpoint
//!   - Vector optimization: vector_quantization, vector_early_termination, vector_saturation_threshold
//! v2.5.0-SuperNode — 🌟 Added SuperNode configuration:
//!   - Extracted SuperNode config to config_supernode.rs (nested structures too complex for flat fields)
//!   - MemChainConfig.supernode: SuperNodeConfig (LLM providers, routing, privacy, worker)
//!   - validate() delegates to SuperNodeConfig.validate()
//!   - Backward compatible: missing [memchain.supernode] section → disabled (v2.4.0 behavior)
//!
//! ## Main Functionality
//! - ServerConfig: top-level config with network, vpn, tun, limits, logging, management, memchain
//! - MemChainConfig: memory system config with MVF parameters, DB paths, API auth, remote storage,
//!   NER engine, cognitive graph, entropy filter, miner steps, vector optimization, and SuperNode
//! - Validation for all config sections
//!
//! ## Dependencies
//! - Used by server.rs to initialize all subsystems
//! - MemChainConfig consumed by MPI router (api/mpi.rs) for auth middleware + remote storage checks
//! - MemChainConfig consumed by storage.rs for DB path
//! - MemChainConfig consumed by ner.rs for NER engine configuration (v2.4.0)
//! - MemChainConfig consumed by log_handler.rs for entropy filter configuration (v2.4.0)
//! - MemChainConfig consumed by miner/reflection.rs for cognitive steps configuration (v2.4.0)
//! - MemChainConfig.supernode consumed by server.rs for LlmRouter + TaskWorker initialization (v2.5.0)
//! - config_supernode.rs — SuperNodeConfig, ProviderConfig, CognitiveTaskType etc. (v2.5.0)
//!
//! ## Main Logical Flow
//! 1. Load TOML file → deserialize into ServerConfig
//! 2. Validate all sections (network, vpn, tun, limits, management, memchain)
//!    2a. MemChainConfig.validate() includes self.supernode.validate() (v2.5.0)
//! 3. Return validated config for server initialization
//!
//! ⚠️ Important Note for Next Developer:
//! - api_secret validation: must be >= 16 chars when set (prevents weak secrets)
//! - Empty/None api_secret = open access (backward compatible)
//! - All MemChain config fields have serde defaults for backward compatibility
//! - allow_remote_storage defaults to false — existing nodes are NOT affected
//! - When allow_remote_storage is true, Ed25519 signature auth is used for remote requests
//!   (parallel to Bearer token auth for local requests)
//! - max_remote_owners caps how many distinct remote users this node will serve
//! - mvf_alpha must be in [0.0, 1.0] — validated since v2.3.0
//! - miner_interval_secs must be > 0 — validated since v2.3.0
//! - embed_dim and embed_max_tokens must be > 0 when memchain is enabled — validated since v2.3.0
//! - v2.4.0 NER/graph/entropy configs all default to disabled or safe values —
//!   existing nodes upgrading to v2.4.0 see zero behavior change until explicitly enabled
//! - ner_confidence_threshold must be in (0.0, 1.0) when NER is enabled
//! - graph_max_depth max is 3 (to prevent runaway BFS)
//! - entropy_filter_threshold must be in [0.0, 1.0]
//! - vector_quantization only supports "none" or "scalar_uint8"
//! - v2.5.0 SuperNode config is in config_supernode.rs — MemChainConfig.supernode field.
//!   SuperNodeConfig::default() has enabled=false, so upgrading nodes see no behavior change.
//!   SuperNode validation is only performed when enabled=true.
//!   See config_supernode.rs for detailed documentation.
//!
//! ## Last Modified
//! v2.1.0 - Added db_path, compaction_threshold, mvf_alpha, mvf_enabled,
//!           cold_start_threshold, cold_start_until, rawlog_batch_threshold
//! v2.1.0+MVF - MVF fields added
//! v2.1.0+MVF+Auth - 🌟 Added api_secret for MPI Bearer token authentication
//! v2.3.0+RemoteStorage - 🌟 Added allow_remote_storage, max_remote_owners for Phase 1
//!   🐛 Added validation for mvf_alpha range, miner_interval_secs > 0,
//!      embed_dim > 0, embed_max_tokens > 0
//! v2.4.0-GraphCognition - 🌟 Added ner_*, graph_*, entropy_*, miner cognitive,
//!   vector optimization config fields
//! v2.5.0-SuperNode - 🌟 Added SuperNode configuration (config_supernode.rs),
//!   MemChainConfig.supernode field, validate() delegation, is_supernode_enabled()

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

// v2.5.0-SuperNode: SuperNode configuration (providers, routing, privacy, worker)
pub use crate::config_supernode::SuperNodeConfig;

// ============================================
// ServerConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub vpn: VpnConfig,
    #[serde(default)]
    pub tun: TunConfig,
    #[serde(default)]
    pub server_key: ServerKeyConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub management: ManagementConfig,
    #[serde(default)]
    pub memchain: MemChainConfig,
}

impl ServerConfig {
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        config.validate()?;
        info!("Configuration loaded successfully");
        Ok(config)
    }

    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;
        self.management.validate().map_err(|e| ServerError::config_invalid("management", e))?;
        self.memchain.validate()?;
        Ok(())
    }

    #[must_use]
    pub fn to_toml(&self) -> String { toml::to_string_pretty(self).unwrap_or_default() }
    pub fn listen_addr(&self) -> SocketAddr { self.network.listen_addr }
    pub fn device_name(&self) -> &str { &self.tun.device_name }
    pub fn ip_range(&self) -> &str { &self.vpn.virtual_ip_range }
    pub fn gateway_ip(&self) -> Ipv4Addr { self.vpn.gateway_ip }
    pub fn mtu(&self) -> u16 { self.tun.mtu }
    pub fn max_sessions(&self) -> usize { self.limits.max_connections }
    pub fn session_timeout_secs(&self) -> u64 { self.limits.session_timeout }
    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> { self.vpn.parse_ip_range() }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            vpn: VpnConfig::default(),
            tun: TunConfig::default(),
            server_key: ServerKeyConfig::default(),
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
            management: ManagementConfig::default(),
            memchain: MemChainConfig::default(),
        }
    }
}

// ============================================
// MemChainMode
// ============================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemChainMode {
    Off,
    Local,
    P2p,
}

impl Default for MemChainMode {
    fn default() -> Self { Self::Local }
}

// ============================================
// VectorQuantizationMode (v2.4.0)
// ============================================

/// Vector quantization strategy for the HNSW index.
///
/// - `None`: Full f32 vectors (1536 bytes/vector for 384-dim). Maximum precision.
/// - `ScalarUint8`: float32 → uint8 quantization (384 bytes/vector). ~75% memory savings.
///   Uses coarse quantized search for candidate retrieval, then f32 re-ranking for final results.
///   Expected precision loss < 3%.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorQuantizationMode {
    /// No quantization — full f32 precision (default)
    None,
    /// Scalar uint8 quantization — 75% memory reduction, < 3% precision loss
    ScalarUint8,
}

impl Default for VectorQuantizationMode {
    fn default() -> Self { Self::None }
}

// ============================================
// MemChainConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemChainConfig {
    #[serde(default)]
    pub mode: MemChainMode,
    #[serde(default = "default_memchain_api_addr")]
    pub api_listen_addr: SocketAddr,
    #[serde(default = "default_memchain_db_path")]
    pub db_path: String,
    #[serde(default = "default_memchain_aof_path")]
    pub aof_path: String,
    #[serde(default)]
    pub trusted_agents: Vec<String>,
    #[serde(default = "default_miner_interval")]
    pub miner_interval_secs: u64,
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: u64,
    #[serde(default = "default_mvf_alpha")]
    pub mvf_alpha: f32,
    #[serde(default)]
    pub mvf_enabled: bool,
    #[serde(default = "default_cold_start_threshold")]
    pub cold_start_threshold: usize,
    #[serde(default = "default_cold_start_until")]
    pub cold_start_until: usize,
    #[serde(default = "default_rawlog_batch_threshold")]
    pub rawlog_batch_threshold: usize,

    // ── Embedding Engine (v2.1.0) ──

    /// Path to the local embedding model directory.
    /// Must contain `model.onnx` and `tokenizer.json` for MiniLM-L6-v2.
    #[serde(default = "default_embed_model_path")]
    pub embed_model_path: String,

    /// Expected embedding dimension from the local model (384 for MiniLM-L6-v2).
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Maximum token sequence length for embedding inference.
    #[serde(default = "default_embed_max_tokens")]
    pub embed_max_tokens: usize,

    /// Output embedding dimension after Matryoshka truncation (v2.5.0).
    ///
    /// For MiniLM-L6-v2: native dimension is 384, this value must be ≤ 384.
    /// For EmbeddingGemma-300M: native dimension is 768, truncated to this value
    ///   via Matryoshka Representation Learning. Supported: 768, 512, 384, 256, 128.
    ///
    /// Default: 384 (compatible with both models, no downstream changes needed).
    /// Pass 0 to EmbedEngine::load() to use this default.
    ///
    /// ⚠️ Changing this value requires rebuilding ALL existing embeddings.
    ///    Miner Step 0.5 handles this automatically on next startup.
    #[serde(default = "default_embed_output_dim")]
    pub embed_output_dim: usize,

    // ── MPI Auth (v2.1.0+MVF+Auth) ──

    /// MPI Bearer token secret for API authentication.
    /// When set (non-empty, >= 16 chars), all MPI endpoints require Bearer auth.
    #[serde(default)]
    pub api_secret: Option<String>,

    // ── Remote Storage (v2.3.0) ──

    /// When true, accepts MPI requests from remote users via Ed25519 signature auth.
    #[serde(default)]
    pub allow_remote_storage: bool,

    /// Maximum number of distinct remote owners this node will serve.
    #[serde(default = "default_max_remote_owners")]
    pub max_remote_owners: usize,

    // ── NER Engine (v2.4.0-GraphCognition) ──

    /// Enable the local GLiNER NER engine for entity extraction.
    ///
    /// When false (default), the entire cognitive graph pipeline is disabled:
    /// - Stage 1 entropy filter uses only semantic divergence (no entity novelty)
    /// - Stage 2 skips entity/relation extraction
    /// - Query analyzer uses regex-only entity detection
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain]
    /// ner_enabled = true
    /// ner_model_path = "models/gliner"
    /// ner_confidence_threshold = 0.5
    /// ```
    #[serde(default)]
    pub ner_enabled: bool,

    /// Path to the GLiNER ONNX model directory.
    /// Must contain `model.onnx` and `tokenizer.json`.
    /// Downloaded via `scripts/download_models.sh`.
    #[serde(default = "default_ner_model_path")]
    pub ner_model_path: String,

    /// Path to the GLiNER tokenizer file.
    /// Typically `tokenizer.json` inside the model directory.
    /// If empty, defaults to `{ner_model_path}/tokenizer.json`.
    #[serde(default)]
    pub ner_tokenizer_path: String,

    /// Minimum confidence score (sigmoid output) to accept an entity detection.
    /// Must be in (0.0, 1.0). Lower values → more entities (higher recall, lower precision).
    /// Recommended: 0.4-0.6 for MemChain use case.
    #[serde(default = "default_ner_confidence_threshold")]
    pub ner_confidence_threshold: f32,

    // ── Knowledge Graph (v2.4.0-GraphCognition) ──

    /// Enable knowledge graph traversal in recall queries.
    ///
    /// When false, recall uses pure vector search (v2.3.0 behavior).
    /// When true, matched entities trigger BFS graph traversal for richer context.
    ///
    /// Requires ner_enabled = true to populate the graph.
    /// If the graph is empty (no entities extracted yet), automatically falls back
    /// to pure vector search.
    #[serde(default)]
    pub graph_enabled: bool,

    /// Maximum BFS traversal depth from matched entities.
    /// 1 = direct relationships only, 2 = relationships of relationships.
    /// Max allowed: 3 (validated). Higher values risk exponential node expansion.
    #[serde(default = "default_graph_max_depth")]
    pub graph_max_depth: usize,

    /// Maximum nodes to expand per BFS hop.
    /// Nodes are sorted by weight × confidence; only top-k are expanded.
    /// Prevents runaway traversal on densely connected entities.
    #[serde(default = "default_graph_max_nodes_per_hop")]
    pub graph_max_nodes_per_hop: usize,

    /// Minimum edge weight to traverse during BFS.
    /// Edges with weight below this threshold are skipped.
    /// Range [0.0, 1.0]. Lower = more edges traversed.
    #[serde(default = "default_graph_min_edge_weight")]
    pub graph_min_edge_weight: f32,

    // ── Entropy Filter (v2.4.0-GraphCognition) ──

    /// Enable entropy-aware filtering on /log ingestion (Stage 1).
    ///
    /// When enabled, conversation windows with low information scores
    /// (repetitive, confirmatory, greeting-only) are discarded before
    /// entering the Rule Engine pipeline.
    ///
    /// Requires ner_enabled = true for entity novelty scoring.
    /// Without NER, only semantic divergence is used (less effective).
    #[serde(default)]
    pub entropy_filter_enabled: bool,

    /// Information score threshold for entropy filtering.
    /// Windows scoring below this are discarded.
    /// Range [0.0, 1.0]. Default 0.35.
    /// - 0.0 = keep everything (filter disabled effectively)
    /// - 1.0 = discard everything (too aggressive)
    /// - 0.35 = recommended for typical dev conversations
    #[serde(default = "default_entropy_filter_threshold")]
    pub entropy_filter_threshold: f32,

    /// Number of messages per sliding window for entropy calculation.
    #[serde(default = "default_entropy_window_size")]
    pub entropy_window_size: usize,

    /// Number of overlapping messages between consecutive windows.
    #[serde(default = "default_entropy_window_overlap")]
    pub entropy_window_overlap: usize,

    // ── Miner Cognitive Steps (v2.4.0-GraphCognition) ──

    /// Enable Miner Step 7: Entity/relation extraction from conversations.
    /// Writes to entities + knowledge_edges tables.
    /// Requires ner_enabled = true.
    #[serde(default)]
    pub miner_entity_extraction: bool,

    /// Enable Miner Step 8: Community detection via label propagation.
    /// Writes to communities + projects tables.
    /// Requires miner_entity_extraction = true (needs entities to cluster).
    #[serde(default)]
    pub miner_community_detection: bool,

    /// Enable Miner Step 10: Session summary generation.
    /// Updates sessions.summary and sessions.key_decisions.
    #[serde(default)]
    pub miner_session_summary: bool,

    /// Enable Miner Step 10: Code artifact extraction from conversations.
    /// Detects code blocks and stores them in the artifacts table.
    #[serde(default)]
    pub miner_artifact_extraction: bool,

    /// Optional LLM endpoint for enhanced cognitive processing.
    ///
    /// When empty (default), all processing is done with local models only
    /// (MiniLM + GLiNER). This is the recommended zero-LLM configuration.
    ///
    /// When set to a valid URL (e.g., "http://localhost:11434/v1/chat/completions"),
    /// the Miner uses LLM for:
    /// - Better session summaries (Step 10)
    /// - Ambiguous conflict resolution (Step 9)
    /// - Richer community descriptions (Step 8)
    ///
    /// Supports any OpenAI-compatible API endpoint (Ollama, vLLM, etc.).
    ///
    /// ⚠️ v2.5.0 Note: This field is superseded by [memchain.supernode] configuration.
    /// When supernode.enabled = true, this field is ignored. Kept for backward
    /// compatibility with v2.4.0 configs that used this single-endpoint approach.
    #[serde(default)]
    pub miner_llm_endpoint: String,

    // ── Vector Optimization (v2.4.0-GraphCognition) ──

    /// Vector quantization strategy.
    /// - "none": Full f32 precision (default, backward compatible)
    /// - "scalar_uint8": 75% memory reduction, < 3% precision loss
    #[serde(default)]
    pub vector_quantization: VectorQuantizationMode,

    /// Enable HNSW early termination via saturation detection.
    /// When true, search stops after `vector_saturation_threshold` consecutive
    /// steps with no improvement. Saves 40-60% unnecessary node visits.
    #[serde(default)]
    pub vector_early_termination: bool,

    /// Number of consecutive non-improving search steps before early termination.
    /// Only effective when vector_early_termination = true.
    /// Lower = faster but potentially less accurate. Default 5.
    #[serde(default = "default_vector_saturation_threshold")]
    pub vector_saturation_threshold: usize,

    // ── Reranker (v2.4.0+Reranker) ──

    /// Enable cross-encoder reranking for recall Step 3.5.
    #[serde(default)]
    pub reranker_enabled: bool,

    /// Path to the cross-encoder model directory.
    #[serde(default = "default_reranker_model_path")]
    pub reranker_model_path: String,

    /// Maximum sequence length for reranker input (query + document).
    #[serde(default = "default_reranker_max_seq_length")]
    pub reranker_max_seq_length: usize,

    // ── SuperNode (v2.5.0) ──

    /// LLM cognitive enhancement layer configuration.
    ///
    /// When supernode.enabled = false (default), the system behaves identically
    /// to v2.4.0 — all processing uses local models only.
    ///
    /// When supernode.enabled = true, cognitive tasks (session title generation,
    /// community narratives, conflict resolution, recall synthesis, code analysis)
    /// are dispatched to configurable LLM providers via an async task queue.
    ///
    /// See config_supernode.rs for full documentation.
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain.supernode]
    /// enabled = true
    ///
    /// [[memchain.supernode.providers]]
    /// name = "deepseek"
    /// type = "openai_compatible"
    /// api_base = "https://api.deepseek.com/v1"
    /// api_key = "$DEEPSEEK_API_KEY"
    /// model = "deepseek-reasoner"
    ///
    /// [memchain.supernode.routing]
    /// fallback = "deepseek"
    /// ```
    #[serde(default)]
    pub supernode: SuperNodeConfig,
}

// ── Default functions ──

fn default_memchain_api_addr() -> SocketAddr { "127.0.0.1:8421".parse().unwrap() }
fn default_memchain_db_path() -> String { "memchain.db".into() }
fn default_memchain_aof_path() -> String { ".memchain".into() }
fn default_miner_interval() -> u64 { 3600 }
fn default_compaction_threshold() -> u64 { 500 }
fn default_mvf_alpha() -> f32 { 0.5 }
fn default_cold_start_threshold() -> usize { 10 }
fn default_cold_start_until() -> usize { 200 }
fn default_rawlog_batch_threshold() -> usize { 100 }
fn default_embed_model_path() -> String { "models/minilm-l6-v2".into() }
fn default_embed_dim() -> usize { 384 }
fn default_embed_max_tokens() -> usize { 128 }
fn default_embed_output_dim() -> usize { 384 }
fn default_max_remote_owners() -> usize { 100 }

// v2.4.0 defaults
fn default_ner_model_path() -> String { "models/gliner".into() }
fn default_ner_confidence_threshold() -> f32 { 0.5 }
fn default_graph_max_depth() -> usize { 2 }
fn default_graph_max_nodes_per_hop() -> usize { 20 }
fn default_graph_min_edge_weight() -> f32 { 0.3 }
fn default_entropy_filter_threshold() -> f32 { 0.35 }
fn default_entropy_window_size() -> usize { 10 }
fn default_entropy_window_overlap() -> usize { 2 }
fn default_vector_saturation_threshold() -> usize { 5 }

// v2.4.0+Reranker defaults
fn default_reranker_model_path() -> String { "models/reranker".into() }
fn default_reranker_max_seq_length() -> usize { 512 }

impl MemChainConfig {
    pub fn validate(&self) -> Result<()> {
        if self.mode != MemChainMode::Off {
            if self.api_listen_addr.port() == 0 {
                return Err(ServerError::config_invalid("memchain.api_listen_addr", "port cannot be 0"));
            }
            if self.db_path.is_empty() {
                return Err(ServerError::config_invalid("memchain.db_path", "cannot be empty"));
            }
            if self.aof_path.is_empty() {
                return Err(ServerError::config_invalid("memchain.aof_path", "cannot be empty"));
            }
            if !self.api_listen_addr.ip().is_loopback() {
                tracing::warn!("[MEMCHAIN] API binding to non-loopback {}", self.api_listen_addr);
            }
            for (i, hex_key) in self.trusted_agents.iter().enumerate() {
                if hex_key.len() != 64 {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!("expected 64 hex chars, got {}", hex_key.len()),
                    ));
                }
                if hex::decode(hex_key).is_err() {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!("invalid hex: '{}'", hex_key),
                    ));
                }
            }

            // Validate api_secret: if set, must be >= 16 characters
            if let Some(ref secret) = self.api_secret {
                if !secret.is_empty() && secret.len() < 16 {
                    return Err(ServerError::config_invalid(
                        "memchain.api_secret",
                        format!("must be at least 16 characters, got {}", secret.len()),
                    ));
                }
            }

            // 🐛 v2.3.0: Validate mvf_alpha range [0.0, 1.0]
            if self.mvf_alpha < 0.0 || self.mvf_alpha > 1.0 {
                return Err(ServerError::config_invalid(
                    "memchain.mvf_alpha",
                    format!("must be in [0.0, 1.0], got {}", self.mvf_alpha),
                ));
            }

            // 🐛 v2.3.0: Validate miner_interval_secs > 0
            if self.miner_interval_secs == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.miner_interval_secs",
                    "must be > 0 (seconds between miner runs)",
                ));
            }

            // 🐛 v2.3.0: Validate embed_dim > 0
            if self.embed_dim == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.embed_dim",
                    "must be > 0",
                ));
            }

            // 🐛 v2.3.0: Validate embed_max_tokens > 0
            if self.embed_max_tokens == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.embed_max_tokens",
                    "must be > 0",
                ));
            }

            // v2.5.0: Validate embed_output_dim > 0
            if self.embed_output_dim == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.embed_output_dim",
                    "must be > 0",
                ));
            }

            // v2.3.0: Validate remote storage configuration
            if self.allow_remote_storage {
                info!(
                    "[MEMCHAIN] Remote storage enabled (max_remote_owners: {})",
                    if self.max_remote_owners == 0 { "unlimited".to_string() }
                    else { self.max_remote_owners.to_string() }
                );

                if self.effective_api_secret().is_none() {
                    tracing::warn!(
                        "[MEMCHAIN] allow_remote_storage=true but api_secret is not set. \
                         Local MPI endpoints are unprotected. Consider setting api_secret \
                         to prevent unauthorized local access."
                    );
                }
            }

            // ── v2.4.0: NER Engine validation ──

            if self.ner_enabled {
                // ner_confidence_threshold must be in (0.0, 1.0)
                if self.ner_confidence_threshold <= 0.0 || self.ner_confidence_threshold >= 1.0 {
                    return Err(ServerError::config_invalid(
                        "memchain.ner_confidence_threshold",
                        format!("must be in (0.0, 1.0), got {}", self.ner_confidence_threshold),
                    ));
                }

                if self.ner_model_path.is_empty() {
                    return Err(ServerError::config_invalid(
                        "memchain.ner_model_path",
                        "cannot be empty when ner_enabled = true",
                    ));
                }

                info!(
                    "[MEMCHAIN] NER engine enabled (model: {}, threshold: {})",
                    self.ner_model_path, self.ner_confidence_threshold
                );
            }

            // ── v2.4.0: Knowledge Graph validation ──

            if self.graph_enabled {
                if self.graph_max_depth == 0 || self.graph_max_depth > 3 {
                    return Err(ServerError::config_invalid(
                        "memchain.graph_max_depth",
                        format!("must be in [1, 3], got {}", self.graph_max_depth),
                    ));
                }

                if self.graph_max_nodes_per_hop == 0 {
                    return Err(ServerError::config_invalid(
                        "memchain.graph_max_nodes_per_hop",
                        "must be > 0",
                    ));
                }

                if self.graph_min_edge_weight < 0.0 || self.graph_min_edge_weight > 1.0 {
                    return Err(ServerError::config_invalid(
                        "memchain.graph_min_edge_weight",
                        format!("must be in [0.0, 1.0], got {}", self.graph_min_edge_weight),
                    ));
                }

                if !self.ner_enabled {
                    tracing::warn!(
                        "[MEMCHAIN] graph_enabled=true but ner_enabled=false. \
                         Graph traversal will have no data until NER is enabled."
                    );
                }
            }

            // ── v2.4.0: Entropy Filter validation ──

            if self.entropy_filter_enabled {
                if self.entropy_filter_threshold < 0.0 || self.entropy_filter_threshold > 1.0 {
                    return Err(ServerError::config_invalid(
                        "memchain.entropy_filter_threshold",
                        format!("must be in [0.0, 1.0], got {}", self.entropy_filter_threshold),
                    ));
                }

                if self.entropy_window_size < 2 {
                    return Err(ServerError::config_invalid(
                        "memchain.entropy_window_size",
                        format!("must be >= 2, got {}", self.entropy_window_size),
                    ));
                }

                if self.entropy_window_overlap >= self.entropy_window_size {
                    return Err(ServerError::config_invalid(
                        "memchain.entropy_window_overlap",
                        format!(
                            "must be < entropy_window_size ({}), got {}",
                            self.entropy_window_size, self.entropy_window_overlap
                        ),
                    ));
                }
            }

            // ── v2.4.0: Miner cognitive steps validation ──

            if self.miner_entity_extraction && !self.ner_enabled {
                tracing::warn!(
                    "[MEMCHAIN] miner_entity_extraction=true but ner_enabled=false. \
                     Entity extraction will be skipped until NER is enabled."
                );
            }

            if self.miner_community_detection && !self.miner_entity_extraction {
                tracing::warn!(
                    "[MEMCHAIN] miner_community_detection=true but miner_entity_extraction=false. \
                     Community detection requires entities to cluster."
                );
            }

            // ── v2.4.0: Vector optimization validation ──

            if self.vector_early_termination && self.vector_saturation_threshold == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.vector_saturation_threshold",
                    "must be > 0 when vector_early_termination is enabled",
                ));
            }

            // ── v2.5.0: SuperNode validation ──
            // Delegates to SuperNodeConfig.validate() which performs its own
            // enabled-gated checks. When supernode.enabled = false, this is a no-op.
            self.supernode.validate()?;
        }
        Ok(())
    }

    #[must_use] pub fn is_enabled(&self) -> bool { self.mode != MemChainMode::Off }
    #[must_use] pub fn is_p2p(&self) -> bool { self.mode == MemChainMode::P2p }

    /// Returns the effective API secret, or None if auth is disabled.
    #[must_use]
    pub fn effective_api_secret(&self) -> Option<&str> {
        self.api_secret.as_deref().filter(|s| !s.is_empty())
    }

    /// Check whether remote storage is enabled.
    #[must_use]
    pub fn is_remote_storage_enabled(&self) -> bool {
        self.allow_remote_storage
    }

    /// v2.4.0: Check whether the cognitive graph pipeline is fully enabled.
    ///
    /// Requires both NER and graph to be enabled. If NER is disabled,
    /// the graph has no data to traverse.
    #[must_use]
    pub fn is_cognitive_graph_enabled(&self) -> bool {
        self.ner_enabled && self.graph_enabled
    }

    /// v2.4.0: Check whether any Miner cognitive steps are enabled.
    #[must_use]
    pub fn has_cognitive_miner_steps(&self) -> bool {
        self.miner_entity_extraction
            || self.miner_community_detection
            || self.miner_session_summary
            || self.miner_artifact_extraction
    }

    /// v2.5.0: Check whether the SuperNode cognitive enhancement layer is enabled.
    ///
    /// When true, the server should initialize LlmRouter + TaskWorker
    /// from self.supernode configuration.
    #[must_use]
    pub fn is_supernode_enabled(&self) -> bool {
        self.supernode.enabled
    }

    /// v2.4.0: Get the effective NER tokenizer path.
    ///
    /// If ner_tokenizer_path is empty, defaults to `{ner_model_path}/tokenizer.json`.
    #[must_use]
    pub fn effective_ner_tokenizer_path(&self) -> String {
        if self.ner_tokenizer_path.is_empty() {
            format!("{}/tokenizer.json", self.ner_model_path)
        } else {
            self.ner_tokenizer_path.clone()
        }
    }

    #[must_use]
    pub fn is_origin_trusted(&self, origin_hex: &str, server_pubkey_hex: &str) -> bool {
        if origin_hex == server_pubkey_hex { return true; }
        if self.trusted_agents.is_empty() { return true; }
        self.trusted_agents.iter().any(|t| t == origin_hex)
    }
}

impl Default for MemChainConfig {
    fn default() -> Self {
        Self {
            mode: MemChainMode::default(),
            api_listen_addr: default_memchain_api_addr(),
            db_path: default_memchain_db_path(),
            aof_path: default_memchain_aof_path(),
            trusted_agents: Vec::new(),
            miner_interval_secs: default_miner_interval(),
            compaction_threshold: default_compaction_threshold(),
            mvf_alpha: default_mvf_alpha(),
            mvf_enabled: false,
            cold_start_threshold: default_cold_start_threshold(),
            cold_start_until: default_cold_start_until(),
            rawlog_batch_threshold: default_rawlog_batch_threshold(),
            embed_model_path: default_embed_model_path(),
            embed_dim: default_embed_dim(),
            embed_max_tokens: default_embed_max_tokens(),
            embed_output_dim: default_embed_output_dim(),
            api_secret: None,
            allow_remote_storage: false,
            max_remote_owners: default_max_remote_owners(),
            // v2.4.0: NER Engine — disabled by default
            ner_enabled: false,
            ner_model_path: default_ner_model_path(),
            ner_tokenizer_path: String::new(),
            ner_confidence_threshold: default_ner_confidence_threshold(),
            // v2.4.0: Knowledge Graph — disabled by default
            graph_enabled: false,
            graph_max_depth: default_graph_max_depth(),
            graph_max_nodes_per_hop: default_graph_max_nodes_per_hop(),
            graph_min_edge_weight: default_graph_min_edge_weight(),
            // v2.4.0: Entropy Filter — disabled by default
            entropy_filter_enabled: false,
            entropy_filter_threshold: default_entropy_filter_threshold(),
            entropy_window_size: default_entropy_window_size(),
            entropy_window_overlap: default_entropy_window_overlap(),
            // v2.4.0: Miner Cognitive Steps — disabled by default
            miner_entity_extraction: false,
            miner_community_detection: false,
            miner_session_summary: false,
            miner_artifact_extraction: false,
            miner_llm_endpoint: String::new(),
            // v2.4.0: Vector Optimization — disabled by default
            vector_quantization: VectorQuantizationMode::default(),
            vector_early_termination: false,
            vector_saturation_threshold: default_vector_saturation_threshold(),
            // v2.4.0+Reranker — disabled by default
            reranker_enabled: false,
            reranker_model_path: default_reranker_model_path(),
            reranker_max_seq_length: default_reranker_max_seq_length(),
            // v2.5.0: SuperNode — disabled by default
            supernode: SuperNodeConfig::default(),
        }
    }
}

// ============================================
// NetworkConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: SocketAddr,
    #[serde(default)]
    pub public_endpoint: Option<String>,
}

fn default_listen_addr() -> SocketAddr { "0.0.0.0:51820".parse().unwrap() }

impl NetworkConfig {
    fn validate(&self) -> Result<()> {
        if self.listen_addr.port() == 0 {
            return Err(ServerError::config_invalid("network.listen_addr", "port cannot be 0"));
        }
        Ok(())
    }

    pub fn public_ip(&self) -> Option<Ipv4Addr> {
        self.public_endpoint.as_ref().and_then(|ep| {
            ep.split(':').next().and_then(|ip| ip.parse().ok())
        })
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self { listen_addr: default_listen_addr(), public_endpoint: None }
    }
}

// ============================================
// VpnConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpnConfig {
    #[serde(default = "default_ip_range")]
    pub virtual_ip_range: String,
    #[serde(default = "default_gateway_ip")]
    pub gateway_ip: Ipv4Addr,
}

fn default_ip_range() -> String { "100.64.0.0/24".into() }
fn default_gateway_ip() -> Ipv4Addr { Ipv4Addr::new(100, 64, 0, 1) }

impl VpnConfig {
    fn validate(&self) -> Result<()> {
        if !self.virtual_ip_range.contains('/') {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "must be CIDR"));
        }
        Ok(())
    }

    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        let parts: Vec<&str> = self.virtual_ip_range.split('/').collect();
        if parts.len() != 2 {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "invalid CIDR"));
        }
        let network: Ipv4Addr = parts[0].parse()
            .map_err(|_| ServerError::config_invalid("vpn.virtual_ip_range", "invalid address"))?;
        let prefix: u8 = parts[1].parse()
            .map_err(|_| ServerError::config_invalid("vpn.virtual_ip_range", "invalid prefix"))?;
        if prefix > 32 {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "prefix > 32"));
        }
        Ok((network, prefix))
    }
}

impl Default for VpnConfig {
    fn default() -> Self {
        Self { virtual_ip_range: default_ip_range(), gateway_ip: default_gateway_ip() }
    }
}

// ============================================
// TunConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunConfig {
    #[serde(default = "default_device_name")]
    pub device_name: String,
    #[serde(default = "default_mtu")]
    pub mtu: u16,
}

fn default_device_name() -> String { "aeronyx0".into() }
fn default_mtu() -> u16 { 1420 }

impl TunConfig {
    fn validate(&self) -> Result<()> {
        if self.device_name.is_empty() {
            return Err(ServerError::config_invalid("tun.device_name", "empty"));
        }
        if self.device_name.len() > 15 {
            return Err(ServerError::config_invalid("tun.device_name", "> 15 chars"));
        }
        if self.mtu < 576 {
            return Err(ServerError::config_invalid("tun.mtu", "< 576"));
        }
        if self.mtu > 9000 {
            return Err(ServerError::config_invalid("tun.mtu", "> 9000"));
        }
        Ok(())
    }
}

impl Default for TunConfig {
    fn default() -> Self { Self { device_name: default_device_name(), mtu: default_mtu() } }
}

// ============================================
// ServerKeyConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerKeyConfig {
    #[serde(default = "default_key_file")]
    pub key_file: String,
}

fn default_key_file() -> String { "/etc/aeronyx/server_key.json".into() }

impl Default for ServerKeyConfig {
    fn default() -> Self { Self { key_file: default_key_file() } }
}

// ============================================
// LimitsConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_session_timeout")]
    pub session_timeout: u64,
}

fn default_max_connections() -> usize { 1000 }
fn default_session_timeout() -> u64 { 300 }

impl LimitsConfig {
    fn validate(&self) -> Result<()> {
        if self.max_connections == 0 {
            return Err(ServerError::config_invalid("limits.max_connections", "= 0"));
        }
        if self.session_timeout == 0 {
            return Err(ServerError::config_invalid("limits.session_timeout", "= 0"));
        }
        Ok(())
    }
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self { max_connections: default_max_connections(), session_timeout: default_session_timeout() }
    }
}

// ============================================
// LoggingConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_log_level() -> String { "info".into() }

impl Default for LoggingConfig {
    fn default() -> Self { Self { level: default_log_level() } }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // Existing tests (preserved from v2.3.0)
    // ========================================

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.memchain.mvf_alpha, 0.5);
        assert!(!config.memchain.mvf_enabled);
        assert_eq!(config.memchain.cold_start_threshold, 10);
        assert!(config.memchain.api_secret.is_none());
        // v2.3.0: remote storage defaults
        assert!(!config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 100);
        // v2.4.0: cognitive graph defaults
        assert!(!config.memchain.ner_enabled);
        assert!(!config.memchain.graph_enabled);
        assert!(!config.memchain.entropy_filter_enabled);
        assert!(!config.memchain.miner_entity_extraction);
        assert!(!config.memchain.miner_community_detection);
        assert!(!config.memchain.vector_early_termination);
        assert_eq!(config.memchain.vector_quantization, VectorQuantizationMode::None);
        // v2.5.0: supernode defaults
        assert!(!config.memchain.supernode.enabled);
        assert!(!config.memchain.is_supernode_enabled());
        assert!(config.memchain.supernode.providers.is_empty());
    }

    #[test]
    fn test_embed_output_dim_zero_rejected() {
        let mc = MemChainConfig { embed_output_dim: 0, ..Default::default() };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_output_dim_custom_valid() {
        let mc = MemChainConfig { embed_output_dim: 768, ..Default::default() };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_v250_toml_embed_gemma_config() {
        let toml_str = r#"
[memchain]
mode = "local"
embed_model_path = "models/embeddinggemma"
embed_max_tokens = 256
embed_output_dim = 384
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert_eq!(mc.embed_model_path, "models/embeddinggemma");
        assert_eq!(mc.embed_max_tokens, 256);
        assert_eq!(mc.embed_output_dim, 384);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memchain_defaults() {
        let mc = MemChainConfig::default();
        assert_eq!(mc.db_path, "memchain.db");
        assert_eq!(mc.compaction_threshold, 500);
        assert_eq!(mc.mvf_alpha, 0.5);
        assert!(!mc.mvf_enabled);
        assert_eq!(mc.cold_start_threshold, 10);
        assert_eq!(mc.cold_start_until, 200);
        assert_eq!(mc.rawlog_batch_threshold, 100);
        assert_eq!(mc.embed_model_path, "models/minilm-l6-v2");
        assert_eq!(mc.embed_dim, 384);
        assert_eq!(mc.embed_max_tokens, 128);
        assert_eq!(mc.embed_output_dim, 384);
        assert!(mc.api_secret.is_none());
        assert!(mc.effective_api_secret().is_none());
        // v2.3.0
        assert!(!mc.allow_remote_storage);
        assert!(!mc.is_remote_storage_enabled());
        assert_eq!(mc.max_remote_owners, 100);
        // v2.4.0
        assert!(!mc.ner_enabled);
        assert_eq!(mc.ner_model_path, "models/gliner");
        assert!(mc.ner_tokenizer_path.is_empty());
        assert!((mc.ner_confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert!(!mc.graph_enabled);
        assert_eq!(mc.graph_max_depth, 2);
        assert_eq!(mc.graph_max_nodes_per_hop, 20);
        assert!((mc.graph_min_edge_weight - 0.3).abs() < f32::EPSILON);
        assert!(!mc.entropy_filter_enabled);
        assert!((mc.entropy_filter_threshold - 0.35).abs() < f32::EPSILON);
        assert_eq!(mc.entropy_window_size, 10);
        assert_eq!(mc.entropy_window_overlap, 2);
        assert!(!mc.miner_entity_extraction);
        assert!(!mc.miner_community_detection);
        assert!(!mc.miner_session_summary);
        assert!(!mc.miner_artifact_extraction);
        assert!(mc.miner_llm_endpoint.is_empty());
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::None);
        assert!(!mc.vector_early_termination);
        assert_eq!(mc.vector_saturation_threshold, 5);
        // v2.5.0
        assert!(!mc.supernode.enabled);
        assert!(mc.supernode.providers.is_empty());
        // Convenience methods
        assert!(!mc.is_cognitive_graph_enabled());
        assert!(!mc.has_cognitive_miner_steps());
        assert!(!mc.is_supernode_enabled());
    }

    #[test]
    fn test_api_secret_validation_too_short() {
        let mc = MemChainConfig {
            api_secret: Some("short".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_api_secret_validation_valid() {
        let mc = MemChainConfig {
            api_secret: Some("this-is-a-long-secret-key-1234".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_api_secret_empty_is_none() {
        let mc = MemChainConfig {
            api_secret: Some(String::new()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.effective_api_secret().is_none());
    }

    #[test]
    fn test_api_secret_none_is_open() {
        let mc = MemChainConfig {
            api_secret: None,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.effective_api_secret().is_none());
    }

    #[test]
    fn test_effective_api_secret() {
        let mc = MemChainConfig {
            api_secret: Some("my-secure-secret-token-here".into()),
            ..Default::default()
        };
        assert_eq!(mc.effective_api_secret(), Some("my-secure-secret-token-here"));
    }

    #[test]
    fn test_is_origin_trusted() {
        let config = MemChainConfig {
            trusted_agents: vec![
                "aaaa0000000000000000000000000000000000000000000000000000000000aa".into(),
            ],
            ..Default::default()
        };
        let server = "bbbb0000000000000000000000000000000000000000000000000000000000bb";
        assert!(config.is_origin_trusted(server, server));
        assert!(config.is_origin_trusted(
            "aaaa0000000000000000000000000000000000000000000000000000000000aa", server));
        assert!(!config.is_origin_trusted(
            "cccc0000000000000000000000000000000000000000000000000000000000cc", server));
    }

    // ========================================
    // v2.3.0: Remote Storage Tests (preserved)
    // ========================================

    #[test]
    fn test_remote_storage_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.allow_remote_storage);
        assert!(!mc.is_remote_storage_enabled());
    }

    #[test]
    fn test_remote_storage_enabled() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 50,
            ..Default::default()
        };
        assert!(mc.is_remote_storage_enabled());
        assert_eq!(mc.max_remote_owners, 50);
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_with_api_secret() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 200,
            api_secret: Some("my-secure-remote-secret".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.is_remote_storage_enabled());
    }

    #[test]
    fn test_remote_storage_unlimited_owners() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_toml_parsing() {
        let toml_str = r#"
[memchain]
mode = "local"
allow_remote_storage = true
max_remote_owners = 75
api_secret = "a-very-secure-secret-key"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 75);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_toml_backward_compat() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 100);
        assert!(config.validate().is_ok());
    }

    // ========================================
    // v2.3.0: Bug Fix Validation Tests (preserved)
    // ========================================

    #[test]
    fn test_mvf_alpha_out_of_range_rejected() {
        let mc = MemChainConfig { mvf_alpha: 1.5, ..Default::default() };
        assert!(mc.validate().is_err());

        let mc2 = MemChainConfig { mvf_alpha: -0.1, ..Default::default() };
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_mvf_alpha_boundary_values() {
        let mc_zero = MemChainConfig { mvf_alpha: 0.0, ..Default::default() };
        assert!(mc_zero.validate().is_ok());

        let mc_one = MemChainConfig { mvf_alpha: 1.0, ..Default::default() };
        assert!(mc_one.validate().is_ok());
    }

    #[test]
    fn test_miner_interval_zero_rejected() {
        let mc = MemChainConfig { miner_interval_secs: 0, ..Default::default() };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_dim_zero_rejected() {
        let mc = MemChainConfig { embed_dim: 0, ..Default::default() };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_max_tokens_zero_rejected() {
        let mc = MemChainConfig { embed_max_tokens: 0, ..Default::default() };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_memchain_off_skips_all_validation() {
        let mc = MemChainConfig {
            mode: MemChainMode::Off,
            mvf_alpha: 999.0,
            miner_interval_secs: 0,
            embed_dim: 0,
            embed_max_tokens: 0,
            db_path: String::new(),
            // v2.4.0: invalid values should also pass when mode=off
            ner_enabled: true,
            ner_confidence_threshold: 2.0,
            graph_max_depth: 99,
            entropy_filter_threshold: -1.0,
            // v2.5.0: invalid supernode config should also pass when mode=off
            supernode: SuperNodeConfig { enabled: true, providers: Vec::new(), ..Default::default() },
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ========================================
    // v2.4.0: NER Engine Tests
    // ========================================

    #[test]
    fn test_ner_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.ner_enabled);
        assert_eq!(mc.ner_model_path, "models/gliner");
        assert!((mc.ner_confidence_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ner_enabled_valid() {
        let mc = MemChainConfig {
            ner_enabled: true,
            ner_model_path: "models/gliner".into(),
            ner_confidence_threshold: 0.4,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_ner_confidence_out_of_range() {
        // threshold = 0.0 (must be > 0.0)
        let mc = MemChainConfig {
            ner_enabled: true,
            ner_confidence_threshold: 0.0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());

        // threshold = 1.0 (must be < 1.0)
        let mc2 = MemChainConfig {
            ner_enabled: true,
            ner_confidence_threshold: 1.0,
            ..Default::default()
        };
        assert!(mc2.validate().is_err());

        // threshold = -0.1 (must be > 0.0)
        let mc3 = MemChainConfig {
            ner_enabled: true,
            ner_confidence_threshold: -0.1,
            ..Default::default()
        };
        assert!(mc3.validate().is_err());
    }

    #[test]
    fn test_ner_empty_model_path_rejected() {
        let mc = MemChainConfig {
            ner_enabled: true,
            ner_model_path: String::new(),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_ner_disabled_skips_validation() {
        // Invalid threshold but NER disabled → should pass
        let mc = MemChainConfig {
            ner_enabled: false,
            ner_confidence_threshold: 2.0,
            ner_model_path: String::new(),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ========================================
    // v2.4.0: Knowledge Graph Tests
    // ========================================

    #[test]
    fn test_graph_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.graph_enabled);
        assert!(!mc.is_cognitive_graph_enabled());
    }

    #[test]
    fn test_graph_enabled_valid() {
        let mc = MemChainConfig {
            ner_enabled: true,
            graph_enabled: true,
            graph_max_depth: 2,
            graph_max_nodes_per_hop: 20,
            graph_min_edge_weight: 0.3,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.is_cognitive_graph_enabled());
    }

    #[test]
    fn test_graph_max_depth_zero_rejected() {
        let mc = MemChainConfig {
            graph_enabled: true,
            graph_max_depth: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_graph_max_depth_too_large_rejected() {
        let mc = MemChainConfig {
            graph_enabled: true,
            graph_max_depth: 4,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_graph_max_depth_boundary() {
        // depth = 1 (minimum valid)
        let mc1 = MemChainConfig {
            graph_enabled: true, graph_max_depth: 1, ..Default::default()
        };
        assert!(mc1.validate().is_ok());

        // depth = 3 (maximum valid)
        let mc3 = MemChainConfig {
            graph_enabled: true, graph_max_depth: 3, ..Default::default()
        };
        assert!(mc3.validate().is_ok());
    }

    #[test]
    fn test_graph_nodes_per_hop_zero_rejected() {
        let mc = MemChainConfig {
            graph_enabled: true,
            graph_max_nodes_per_hop: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_graph_edge_weight_out_of_range() {
        let mc = MemChainConfig {
            graph_enabled: true,
            graph_min_edge_weight: -0.1,
            ..Default::default()
        };
        assert!(mc.validate().is_err());

        let mc2 = MemChainConfig {
            graph_enabled: true,
            graph_min_edge_weight: 1.1,
            ..Default::default()
        };
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_graph_disabled_skips_validation() {
        let mc = MemChainConfig {
            graph_enabled: false,
            graph_max_depth: 99,
            graph_max_nodes_per_hop: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ========================================
    // v2.4.0: Entropy Filter Tests
    // ========================================

    #[test]
    fn test_entropy_filter_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.entropy_filter_enabled);
    }

    #[test]
    fn test_entropy_filter_enabled_valid() {
        let mc = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_filter_threshold: 0.35,
            entropy_window_size: 10,
            entropy_window_overlap: 2,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_entropy_threshold_out_of_range() {
        let mc = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_filter_threshold: -0.1,
            ..Default::default()
        };
        assert!(mc.validate().is_err());

        let mc2 = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_filter_threshold: 1.1,
            ..Default::default()
        };
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_entropy_window_too_small() {
        let mc = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_window_size: 1,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_entropy_overlap_exceeds_window() {
        let mc = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_window_size: 5,
            entropy_window_overlap: 5,
            ..Default::default()
        };
        assert!(mc.validate().is_err());

        let mc2 = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_window_size: 5,
            entropy_window_overlap: 6,
            ..Default::default()
        };
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_entropy_disabled_skips_validation() {
        let mc = MemChainConfig {
            entropy_filter_enabled: false,
            entropy_filter_threshold: -99.0,
            entropy_window_size: 0,
            entropy_window_overlap: 999,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ========================================
    // v2.4.0: Miner Cognitive Steps Tests
    // ========================================

    #[test]
    fn test_miner_cognitive_steps_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.has_cognitive_miner_steps());
    }

    #[test]
    fn test_miner_cognitive_steps_enabled() {
        let mc = MemChainConfig {
            ner_enabled: true,
            miner_entity_extraction: true,
            miner_community_detection: true,
            miner_session_summary: true,
            miner_artifact_extraction: true,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.has_cognitive_miner_steps());
    }

    #[test]
    fn test_miner_llm_endpoint_empty_is_local_only() {
        let mc = MemChainConfig::default();
        assert!(mc.miner_llm_endpoint.is_empty());
    }

    // ========================================
    // v2.4.0: Vector Optimization Tests
    // ========================================

    #[test]
    fn test_vector_quantization_default_none() {
        let mc = MemChainConfig::default();
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::None);
    }

    #[test]
    fn test_vector_early_termination_valid() {
        let mc = MemChainConfig {
            vector_early_termination: true,
            vector_saturation_threshold: 5,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_vector_early_termination_zero_threshold_rejected() {
        let mc = MemChainConfig {
            vector_early_termination: true,
            vector_saturation_threshold: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_vector_disabled_early_term_skips_threshold_validation() {
        let mc = MemChainConfig {
            vector_early_termination: false,
            vector_saturation_threshold: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ========================================
    // v2.4.0: TOML Parsing Tests
    // ========================================

    #[test]
    fn test_v240_toml_full_config() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true
ner_model_path = "models/gliner"
ner_confidence_threshold = 0.45
graph_enabled = true
graph_max_depth = 2
graph_max_nodes_per_hop = 30
graph_min_edge_weight = 0.25
entropy_filter_enabled = true
entropy_filter_threshold = 0.4
entropy_window_size = 8
entropy_window_overlap = 1
miner_entity_extraction = true
miner_community_detection = true
miner_session_summary = true
miner_artifact_extraction = true
miner_llm_endpoint = ""
vector_quantization = "scalar_uint8"
vector_early_termination = true
vector_saturation_threshold = 3
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;

        assert!(mc.ner_enabled);
        assert_eq!(mc.ner_model_path, "models/gliner");
        assert!((mc.ner_confidence_threshold - 0.45).abs() < f32::EPSILON);
        assert!(mc.graph_enabled);
        assert_eq!(mc.graph_max_depth, 2);
        assert_eq!(mc.graph_max_nodes_per_hop, 30);
        assert!((mc.graph_min_edge_weight - 0.25).abs() < f32::EPSILON);
        assert!(mc.entropy_filter_enabled);
        assert!((mc.entropy_filter_threshold - 0.4).abs() < f32::EPSILON);
        assert_eq!(mc.entropy_window_size, 8);
        assert_eq!(mc.entropy_window_overlap, 1);
        assert!(mc.miner_entity_extraction);
        assert!(mc.miner_community_detection);
        assert!(mc.miner_session_summary);
        assert!(mc.miner_artifact_extraction);
        assert!(mc.miner_llm_endpoint.is_empty());
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::ScalarUint8);
        assert!(mc.vector_early_termination);
        assert_eq!(mc.vector_saturation_threshold, 3);

        assert!(config.validate().is_ok());
        assert!(mc.is_cognitive_graph_enabled());
        assert!(mc.has_cognitive_miner_steps());
        // Supernode should be disabled (not in TOML)
        assert!(!mc.is_supernode_enabled());
    }

    #[test]
    fn test_v240_toml_backward_compat() {
        // Old TOML without any v2.4.0 fields → all default to disabled
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
mvf_alpha = 0.5
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;

        assert!(!mc.ner_enabled);
        assert!(!mc.graph_enabled);
        assert!(!mc.entropy_filter_enabled);
        assert!(!mc.miner_entity_extraction);
        assert!(!mc.vector_early_termination);
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::None);
        // v2.5.0: supernode defaults when section missing
        assert!(!mc.is_supernode_enabled());

        assert!(config.validate().is_ok());
    }

    // ========================================
    // v2.4.0: Convenience Method Tests
    // ========================================

    #[test]
    fn test_is_cognitive_graph_enabled() {
        // Both NER and graph must be enabled
        let mc1 = MemChainConfig {
            ner_enabled: true, graph_enabled: true, ..Default::default()
        };
        assert!(mc1.is_cognitive_graph_enabled());

        let mc2 = MemChainConfig {
            ner_enabled: true, graph_enabled: false, ..Default::default()
        };
        assert!(!mc2.is_cognitive_graph_enabled());

        let mc3 = MemChainConfig {
            ner_enabled: false, graph_enabled: true, ..Default::default()
        };
        assert!(!mc3.is_cognitive_graph_enabled());
    }

    #[test]
    fn test_effective_ner_tokenizer_path() {
        let mc1 = MemChainConfig::default();
        assert_eq!(mc1.effective_ner_tokenizer_path(), "models/gliner/tokenizer.json");

        let mc2 = MemChainConfig {
            ner_tokenizer_path: "/custom/tokenizer.json".into(),
            ..Default::default()
        };
        assert_eq!(mc2.effective_ner_tokenizer_path(), "/custom/tokenizer.json");
    }

    // ========================================
    // v2.5.0: SuperNode Integration Tests
    // ========================================

    #[test]
    fn test_supernode_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.is_supernode_enabled());
        assert!(!mc.supernode.enabled);
        assert!(mc.supernode.providers.is_empty());
    }

    #[test]
    fn test_supernode_validate_delegation() {
        // When supernode is enabled with no providers, MemChainConfig.validate() should fail
        let mc = MemChainConfig {
            supernode: SuperNodeConfig {
                enabled: true,
                providers: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_supernode_validate_passes_with_valid_provider() {
        let mc = MemChainConfig {
            supernode: SuperNodeConfig {
                enabled: true,
                providers: vec![config_supernode::ProviderConfig {
                    name: "ollama".into(),
                    provider_type: config_supernode::ProviderType::OpenaiCompatible,
                    api_base: "http://localhost:11434/v1".into(),
                    api_key: None,
                    model: "llama3".into(),
                    max_tokens: None,
                    temperature: None,
                }],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.is_supernode_enabled());
    }

    #[test]
    fn test_supernode_toml_full_integration() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true
ner_model_path = "models/gliner"
graph_enabled = true
miner_entity_extraction = true

[memchain.supernode]
enabled = true

[[memchain.supernode.providers]]
name = "deepseek"
type = "openai_compatible"
api_base = "https://api.deepseek.com/v1"
api_key = "$DEEPSEEK_API_KEY"
model = "deepseek-reasoner"
max_tokens = 2000
temperature = 0.6

[[memchain.supernode.providers]]
name = "claude"
type = "anthropic"
api_key = "$ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"

[memchain.supernode.routing]
session_title = "deepseek"
code_analysis = "claude"
fallback = "deepseek"

[memchain.supernode.privacy]
default_level = "structured"
allow_full_for = ["code_analysis"]

[memchain.supernode.worker]
poll_interval_secs = 10
max_concurrent = 5
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;

        // v2.4.0 fields work alongside v2.5.0 supernode
        assert!(mc.ner_enabled);
        assert!(mc.graph_enabled);
        assert!(mc.miner_entity_extraction);

        // v2.5.0 supernode fields
        assert!(mc.is_supernode_enabled());
        assert_eq!(mc.supernode.providers.len(), 2);
        assert_eq!(mc.supernode.providers[0].name, "deepseek");
        assert_eq!(mc.supernode.providers[1].name, "claude");
        assert_eq!(mc.supernode.routing.session_title, Some("deepseek".into()));
        assert_eq!(mc.supernode.routing.code_analysis, Some("claude".into()));
        assert_eq!(mc.supernode.routing.fallback, Some("deepseek".into()));
        assert_eq!(mc.supernode.privacy.default_level, config_supernode::PrivacyLevel::Structured);
        assert_eq!(mc.supernode.worker.poll_interval_secs, 10);
        assert_eq!(mc.supernode.worker.max_concurrent, 5);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_supernode_toml_backward_compat_no_section() {
        // TOML from v2.4.0 without any supernode section → disabled
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true
graph_enabled = true
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.is_supernode_enabled());
        assert!(config.memchain.supernode.providers.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_supernode_disabled_with_partial_config_passes() {
        // User started configuring supernode but left enabled = false
        let toml_str = r#"
[memchain]
mode = "local"

[memchain.supernode]
enabled = false

[[memchain.supernode.providers]]
name = "deepseek"
type = "openai_compatible"
api_base = "https://api.deepseek.com/v1"
model = "deepseek-reasoner"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.is_supernode_enabled());
        assert_eq!(config.memchain.supernode.providers.len(), 1);
        // Should pass because enabled=false skips validation
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_supernode_routing_bad_reference_rejected() {
        let toml_str = r#"
[memchain]
mode = "local"

[memchain.supernode]
enabled = true

[[memchain.supernode.providers]]
name = "deepseek"
type = "openai_compatible"
api_base = "https://api.deepseek.com/v1"
model = "deepseek-reasoner"

[memchain.supernode.routing]
session_title = "nonexistent_provider"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_is_supernode_enabled_method() {
        let mc1 = MemChainConfig::default();
        assert!(!mc1.is_supernode_enabled());

        let mc2 = MemChainConfig {
            supernode: SuperNodeConfig { enabled: true, ..Default::default() },
            ..Default::default()
        };
        assert!(mc2.is_supernode_enabled());
    }
}
