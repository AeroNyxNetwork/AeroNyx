// ============================================
// File: crates/aeronyx-server/src/config_memchain.rs
// ============================================
//! # MemChain Configuration
//!
//! ## Creation Reason
//! Extracted from config.rs as part of the v1.1.0-ChatRelay refactor split.
//! `MemChainConfig` + `MemChainMode` were the largest section of config.rs
//! (~600 lines) and have no coupling to pure infrastructure configs.
//!
//! ## Modification Reason
//! v2.1.0            — Added db_path, compaction_threshold, mvf_alpha, mvf_enabled,
//!                     cold_start_threshold, cold_start_until, rawlog_batch_threshold
//! v2.1.0+MVF        — MVF fields added
//! v2.1.0+MVF+Auth   — 🌟 Added api_secret for MPI Bearer token authentication
//! v2.3.0+RemoteStorage — 🌟 Added allow_remote_storage, max_remote_owners for Phase 1
//!                     🐛 Added validation for mvf_alpha range, miner_interval_secs > 0,
//!                        embed_dim > 0, embed_max_tokens > 0
//! v2.4.0-GraphCognition — 🌟 Added ner_*, graph_*, entropy_*, miner cognitive,
//!                     vector optimization config fields
//! v2.5.0-SuperNode  — 🌟 Added SuperNode configuration (config_supernode.rs),
//!                     MemChainConfig.supernode field, validate() delegation,
//!                     is_supernode_enabled()
//! v1.0.0-MultiTenant — 🌟 Added MemChainMode::Saas, SaasConfig, jwt_secret,
//!                     token_ttl_secs, SaaS-gated validate() checks
//! v1.1.0-ChatRelay  — 🌟 Added ChatRelayConfig (chat_relay field),
//!                     is_chat_relay_enabled(), extracted to config_memchain.rs
//! v2.7.0-BlockSync  — Added an explicit, default-off commitment coordinator
//!                     role so independent nodes cannot accidentally fork.
//! v2.8.1-BlockSync  — Made the follower's bounded pages-per-round budget
//!                     configurable for controlled recovery and low-I/O nodes.
//!
//! ## Main Functionality
//! - `MemChainMode`   — Off / Local / P2p / Saas
//! - `VectorQuantizationMode` — None / ScalarUint8
//! - `MemChainConfig` — all MemChain subsystem knobs, full validate()
//! - Convenience methods: is_enabled, is_p2p, is_saas, is_chat_relay_enabled,
//!   is_cognitive_graph_enabled, has_cognitive_miner_steps,
//!   is_supernode_enabled, effective_api_secret, effective_saas_config,
//!   effective_ner_tokenizer_path, is_remote_storage_enabled,
//!   is_origin_trusted
//!
//! ## Dependencies
//! - `config_saas.rs`       — SaasConfig
//! - `config_chat_relay.rs` — ChatRelayConfig
//! - `config_supernode.rs`  — SuperNodeConfig
//! - `error.rs`             — Result, ServerError
//! - Consumed by `config.rs` (ServerConfig.memchain)
//! - Consumed by `server.rs`, `api/mpi.rs`, `storage.rs`,
//!   `ner.rs`, `log_handler.rs`, `miner/reflection.rs`
//!
//! ## Main Logical Flow
//! 1. TOML `[memchain]` → deserialize into `MemChainConfig`
//! 2. `validate()` — skips everything when mode = Off
//! 3. Validates fields in dependency order:
//!    addr/path → api_secret → mvf/embed → remote storage →
//!    NER → graph → entropy → miner steps → vector →
//!    supernode (delegates) → saas (gated) → chat_relay (delegates)
//! 4. All feature flags default to false — upgrading nodes see no change
//!
//! ⚠️ Important Note for Next Developer:
//! - api_secret: must be >= 16 chars when set (prevents weak secrets).
//!   Empty/None = open access (backward compatible).
//! - mvf_alpha must be in [0.0, 1.0] — validated since v2.3.0.
//! - graph_max_depth max is 3 (to prevent runaway BFS).
//! - vector_quantization only supports "none" or "scalar_uint8".
//! - SaaS validate is gated on mode == Saas; other modes skip all SaaS checks.
//! - jwt_secret auto-generation is handled at server startup, not here —
//!   validate() only rejects manually-set secrets that are too short.
//! - effective_saas_config() returns Cow; cache result in hot paths.
//! - chat_relay.db_path and saas.data_root share "data/" prefix by
//!   convention but are NOT linked — must be updated independently.
//! - commitment_coordinator_enabled must remain default-off. Until a
//!   consensus protocol exists, exactly one configured coordinator may append
//!   the canonical commitment chain; all other nodes are verifier/followers.
//! - commitment_sync_enabled must remain default-off. A follower accepts
//!   blocks only from the explicitly pinned coordinator node identity and
//!   fails closed on rollback, fork, signature, or continuity errors.
//! - commitment_tip_anchor_path is a coordinator-local rollback guard outside
//!   SQLite. An empty value derives a sibling file beside db_path. Never point
//!   it at the SQLite database itself.
//! - commitment_witness_node_ids are explicit operator trust pins. Automatic
//!   discovery may supply endpoints, but an unpinned permissionless peer must
//!   never gain authority to block coordinator startup.
//! - commitment_witness_min_verified defaults to one for compatibility. It is
//!   an operator startup threshold over distinct pinned identities, not network
//!   consensus, quorum, finality, leader election, or fork choice.
//!
//! ## Last Modified
//! v2.8.1-BlockSync - Added a validated follower pages-per-round budget.
//! v2.7.16-WitnessThreshold - Added a configurable pinned-witness startup threshold.
//! v2.7.15-ExternalWitnessGuard - Added pinned startup checkpoint witnesses.
//! v2.7.14-CommitmentTipAnchor - Added the coordinator-local signed tip anchor.
//! v2.7.1-BlockFollower — Added default-off pinned coordinator follower settings.
//! v2.7.0-BlockSync — Added default-off commitment coordinator role.

use std::borrow::Cow;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::config_chat_relay::ChatRelayConfig;
use crate::config_saas::SaasConfig;
use crate::config_supernode::SuperNodeConfig;
use crate::error::{Result, ServerError};

// ============================================
// MemChainMode
// ============================================

/// Operating mode for the MemChain memory subsystem.
///
/// ## Modification Reason
/// v1.0.0-MultiTenant: Added `Saas` variant for multi-tenant SaaS deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemChainMode {
    Off,
    Local,
    P2p,
    /// v1.0.0-MultiTenant: SaaS multi-tenant mode.
    ///
    /// Each user gets their own isolated SQLite DB, JWT auth, and StoragePool entry.
    /// Requires [memchain.saas] section (or defaults are used with a warning).
    Saas,
}

impl Default for MemChainMode {
    fn default() -> Self {
        Self::Local
    }
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
    fn default() -> Self {
        Self::None
    }
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

    // ── Embedding Engine (v2.1.0) ──────────────────────────────────────
    /// Enable local embedding inference for MemChain semantic recall.
    ///
    /// Protocol-only nodes may set this to `false` to keep encrypted storage,
    /// public discovery, and ChatRelay available without requiring a local ONNX
    /// model bundle. The default remains `true` for backward compatibility.
    #[serde(default = "default_embed_enabled")]
    pub embed_enabled: bool,

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
    /// For EmbeddingGemma-300M: native dimension is 768, truncated via
    ///   Matryoshka Representation Learning. Supported: 768, 512, 384, 256, 128.
    ///
    /// Default: 384 (compatible with both models, no downstream changes needed).
    /// Pass 0 to EmbedEngine::load() to use this default.
    ///
    /// ⚠️ Changing this value requires rebuilding ALL existing embeddings.
    ///    Miner Step 0.5 handles this automatically on next startup.
    #[serde(default = "default_embed_output_dim")]
    pub embed_output_dim: usize,

    // ── MPI Auth (v2.1.0+MVF+Auth) ────────────────────────────────────
    /// MPI Bearer token secret for API authentication.
    /// When set (non-empty, >= 16 chars), all MPI endpoints require Bearer auth.
    #[serde(default)]
    pub api_secret: Option<String>,

    // ── Remote Storage (v2.3.0) ───────────────────────────────────────
    /// When true, accepts MPI requests from remote users via Ed25519 signature auth.
    #[serde(default)]
    pub allow_remote_storage: bool,

    /// When true, accepts node-blind sealed records (POST /api/mpi/remember_sealed)
    /// even if `allow_remote_storage` is off. Blind storage is a strictly safer
    /// subset — the node cannot read the content — so it can be enabled on its own.
    #[serde(default)]
    pub blind_storage_enabled: bool,

    /// Allows this node to append canonical record commitment blocks.
    ///
    /// This is deliberately `false` by default. Block Sync v1 has signed
    /// ordering but no distributed fork-choice/consensus protocol, so enabling
    /// more than one independent writer would create divergent height chains.
    /// Follower nodes still verify and serve already-synchronised blocks.
    #[serde(default)]
    pub commitment_coordinator_enabled: bool,

    /// Pull and verify canonical commitment blocks from one pinned coordinator.
    ///
    /// This is deliberately `false` by default. The pinned Ed25519 node id is
    /// the trust root for Block Sync v1; discovery supplies only its current
    /// signed endpoint. This is authenticated replication, not consensus.
    #[serde(default)]
    pub commitment_sync_enabled: bool,

    /// Hex-encoded Ed25519 identity of the only accepted block coordinator.
    ///
    /// Required when `commitment_sync_enabled=true`. An empty value is inert
    /// while sync is disabled so existing configurations remain compatible.
    #[serde(default)]
    pub commitment_coordinator_node_id: String,

    /// Base interval between successful follower catch-up rounds.
    ///
    /// Failures use bounded exponential backoff independently of this value.
    #[serde(default = "default_commitment_sync_interval_secs")]
    pub commitment_sync_interval_secs: u64,

    /// Maximum verified commitment pages fetched in one follower round.
    ///
    /// Each page is transactionally verified before append. Bounding a round
    /// keeps shutdown responsive and limits burst I/O; a remaining backlog is
    /// resumed after a short delay. The default of eight preserves historical
    /// behavior, while one is useful for constrained nodes and recovery drills.
    #[serde(default = "default_commitment_sync_max_pages_per_round")]
    pub commitment_sync_max_pages_per_round: usize,

    /// Optional path for the coordinator's signed commitment-tip high-water mark.
    ///
    /// Empty (the backward-compatible default) derives
    /// `record-commitment-tip-v1.json` beside `db_path`. The sidecar detects a
    /// valid but older SQLite file after restart while host state remains. It
    /// is not a whole-machine snapshot rollback or distributed-consensus proof.
    #[serde(default)]
    pub commitment_tip_anchor_path: String,

    /// Ed25519 node identities trusted as external startup witnesses.
    ///
    /// Discovery resolves each pinned identity's current signed endpoint. A
    /// valid `remote_ahead` or `diverged` checkpoint from a pin blocks the
    /// coordinator before its transport/API listeners start. Empty preserves
    /// the previous permissionless evidence-only behavior. Pin only audited
    /// followers of this coordinator; an unrelated chain is not a witness.
    #[serde(default)]
    pub commitment_witness_node_ids: Vec<String>,

    /// Require at least one pinned witness to return valid signed evidence.
    ///
    /// This is default-off for availability and backward compatibility. When
    /// false, positive rollback/divergence evidence still blocks startup, but
    /// a temporary absence of every pin is reported as degraded and allowed.
    #[serde(default)]
    pub commitment_witness_startup_required: bool,

    /// Minimum distinct pinned witnesses required by strict startup.
    ///
    /// The default of one preserves existing strict deployments. Operators
    /// can stage a higher value while strict mode is disabled, verify repeated
    /// signed responses, and then enable strict startup. This is an explicit
    /// trust policy over pinned identities, not a consensus quorum.
    #[serde(default = "default_commitment_witness_min_verified")]
    pub commitment_witness_min_verified: usize,

    /// Maximum number of distinct remote owners this node will serve.
    #[serde(default = "default_max_remote_owners")]
    pub max_remote_owners: usize,

    // ── NER Engine (v2.4.0-GraphCognition) ───────────────────────────
    /// Enable the local GLiNER NER engine for entity extraction.
    ///
    /// When false (default), the entire cognitive graph pipeline is disabled.
    #[serde(default)]
    pub ner_enabled: bool,

    /// Path to the GLiNER ONNX model directory.
    #[serde(default = "default_ner_model_path")]
    pub ner_model_path: String,

    /// Path to the GLiNER tokenizer file.
    /// If empty, defaults to `{ner_model_path}/tokenizer.json`.
    #[serde(default)]
    pub ner_tokenizer_path: String,

    /// Minimum confidence score (sigmoid output) to accept an entity detection.
    /// Must be in (0.0, 1.0). Recommended: 0.4–0.6.
    #[serde(default = "default_ner_confidence_threshold")]
    pub ner_confidence_threshold: f32,

    // ── Knowledge Graph (v2.4.0-GraphCognition) ───────────────────────
    /// Enable knowledge graph traversal in recall queries.
    /// Requires ner_enabled = true to populate the graph.
    #[serde(default)]
    pub graph_enabled: bool,

    /// Maximum BFS traversal depth from matched entities. Max allowed: 3.
    #[serde(default = "default_graph_max_depth")]
    pub graph_max_depth: usize,

    /// Maximum nodes to expand per BFS hop.
    #[serde(default = "default_graph_max_nodes_per_hop")]
    pub graph_max_nodes_per_hop: usize,

    /// Minimum edge weight to traverse during BFS. Range [0.0, 1.0].
    #[serde(default = "default_graph_min_edge_weight")]
    pub graph_min_edge_weight: f32,

    // ── Entropy Filter (v2.4.0-GraphCognition) ────────────────────────
    /// Enable entropy-aware filtering on /log ingestion (Stage 1).
    #[serde(default)]
    pub entropy_filter_enabled: bool,

    /// Information score threshold for entropy filtering. Range [0.0, 1.0].
    #[serde(default = "default_entropy_filter_threshold")]
    pub entropy_filter_threshold: f32,

    /// Number of messages per sliding window for entropy calculation.
    #[serde(default = "default_entropy_window_size")]
    pub entropy_window_size: usize,

    /// Number of overlapping messages between consecutive windows.
    #[serde(default = "default_entropy_window_overlap")]
    pub entropy_window_overlap: usize,

    // ── Miner Cognitive Steps (v2.4.0-GraphCognition) ─────────────────
    /// Enable Miner Step 7: Entity/relation extraction. Requires ner_enabled.
    #[serde(default)]
    pub miner_entity_extraction: bool,

    /// Enable Miner Step 8: Community detection. Requires miner_entity_extraction.
    #[serde(default)]
    pub miner_community_detection: bool,

    /// Enable Miner Step 10: Session summary generation.
    #[serde(default)]
    pub miner_session_summary: bool,

    /// Enable Miner Step 10: Code artifact extraction.
    #[serde(default)]
    pub miner_artifact_extraction: bool,

    /// Optional LLM endpoint for enhanced cognitive processing.
    ///
    /// ⚠️ v2.5.0 Note: Superseded by [memchain.supernode] when
    /// supernode.enabled = true. Kept for backward compatibility.
    #[serde(default)]
    pub miner_llm_endpoint: String,

    // ── Vector Optimization (v2.4.0-GraphCognition) ───────────────────
    /// Vector quantization strategy ("none" | "scalar_uint8").
    #[serde(default)]
    pub vector_quantization: VectorQuantizationMode,

    /// Enable HNSW early termination via saturation detection.
    #[serde(default)]
    pub vector_early_termination: bool,

    /// Consecutive non-improving steps before early termination.
    #[serde(default = "default_vector_saturation_threshold")]
    pub vector_saturation_threshold: usize,

    // ── Reranker (v2.4.0+Reranker) ────────────────────────────────────
    /// Enable cross-encoder reranking for recall Step 3.5.
    #[serde(default)]
    pub reranker_enabled: bool,

    /// Path to the cross-encoder model directory.
    #[serde(default = "default_reranker_model_path")]
    pub reranker_model_path: String,

    /// Maximum sequence length for reranker input (query + document).
    #[serde(default = "default_reranker_max_seq_length")]
    pub reranker_max_seq_length: usize,

    // ── SuperNode (v2.5.0) ────────────────────────────────────────────
    /// LLM cognitive enhancement layer configuration.
    ///
    /// When supernode.enabled = false (default), system behaves identically
    /// to v2.4.0. See config_supernode.rs for full documentation.
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
    /// ```
    #[serde(default)]
    pub supernode: SuperNodeConfig,

    // ── Multi-Tenant SaaS (v1.0.0-MultiTenant) ───────────────────────
    /// SaaS mode infrastructure configuration.
    /// Only used when `mode = "saas"`. See config_saas.rs.
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain.saas]
    /// data_root = "data"
    /// pool_max_connections = 100
    /// ```
    #[serde(default)]
    pub saas: Option<SaasConfig>,

    /// JWT signing secret for SaaS mode token issuance.
    /// Empty/None → auto-generated on first startup. Min 32 chars when set.
    #[serde(default)]
    pub jwt_secret: Option<String>,

    /// JWT token validity in seconds. Default 86400 (24 hours).
    #[serde(default = "default_jwt_ttl")]
    pub token_ttl_secs: u64,

    // ── Chat Relay (v1.1.0-ChatRelay) ────────────────────────────────
    /// Zero-knowledge P2P chat relay configuration.
    ///
    /// When `chat_relay.enabled = false` (default), all ChatRelay/ChatPull/
    /// ChatAck/ChatExpired MemChain messages are silently ignored.
    /// Existing nodes upgrading to v1.1.0-ChatRelay see zero behavior change
    /// until explicitly enabled. See config_chat_relay.rs.
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain.chat_relay]
    /// enabled = true
    /// offline_ttl_secs = 259200
    /// max_pending_per_wallet = 500
    /// ```
    #[serde(default)]
    pub chat_relay: ChatRelayConfig,
}

// ── Default functions ──────────────────────────────────────────────────────

fn default_memchain_api_addr() -> SocketAddr {
    "127.0.0.1:8421".parse().unwrap()
}
fn default_memchain_db_path() -> String {
    "memchain.db".into()
}
fn default_memchain_aof_path() -> String {
    ".memchain".into()
}
fn default_miner_interval() -> u64 {
    3600
}
fn default_compaction_threshold() -> u64 {
    500
}
fn default_mvf_alpha() -> f32 {
    0.5
}
fn default_cold_start_threshold() -> usize {
    10
}
fn default_cold_start_until() -> usize {
    200
}
fn default_rawlog_batch_threshold() -> usize {
    100
}
fn default_embed_enabled() -> bool {
    true
}
fn default_embed_model_path() -> String {
    "models/minilm-l6-v2".into()
}
fn default_embed_dim() -> usize {
    384
}
fn default_embed_max_tokens() -> usize {
    128
}
fn default_embed_output_dim() -> usize {
    384
}
fn default_max_remote_owners() -> usize {
    100
}
fn default_commitment_sync_interval_secs() -> u64 {
    30
}
fn default_commitment_sync_max_pages_per_round() -> usize {
    8
}
fn default_commitment_witness_min_verified() -> usize {
    1
}

const MAX_COMMITMENT_WITNESS_NODE_IDS: usize = 3;
fn default_ner_model_path() -> String {
    "models/gliner".into()
}
fn default_ner_confidence_threshold() -> f32 {
    0.5
}
fn default_graph_max_depth() -> usize {
    2
}
fn default_graph_max_nodes_per_hop() -> usize {
    20
}
fn default_graph_min_edge_weight() -> f32 {
    0.3
}
fn default_entropy_filter_threshold() -> f32 {
    0.35
}
fn default_entropy_window_size() -> usize {
    10
}
fn default_entropy_window_overlap() -> usize {
    2
}
fn default_vector_saturation_threshold() -> usize {
    5
}
fn default_reranker_model_path() -> String {
    "models/reranker".into()
}
fn default_reranker_max_seq_length() -> usize {
    512
}
fn default_jwt_ttl() -> u64 {
    86400
}

// ── validate() ────────────────────────────────────────────────────────────

impl MemChainConfig {
    pub fn validate(&self) -> Result<()> {
        if self.mode == MemChainMode::Off {
            return Ok(());
        }

        // ── Basic address / path checks ──────────────────────────────────
        if self.api_listen_addr.port() == 0 {
            return Err(ServerError::config_invalid(
                "memchain.api_listen_addr",
                "port cannot be 0",
            ));
        }
        if self.db_path.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.db_path",
                "cannot be empty",
            ));
        }
        if self.aof_path.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.aof_path",
                "cannot be empty",
            ));
        }
        if !self.api_listen_addr.ip().is_loopback() {
            tracing::warn!(
                "[MEMCHAIN] API binding to non-loopback {}",
                self.api_listen_addr
            );
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

        // ── api_secret: must be >= 16 chars when set ──────────────────
        if let Some(ref secret) = self.api_secret {
            if !secret.is_empty() && secret.len() < 16 {
                return Err(ServerError::config_invalid(
                    "memchain.api_secret",
                    format!("must be at least 16 characters, got {}", secret.len()),
                ));
            }
        }

        // ── 🐛 v2.3.0: mvf_alpha [0.0, 1.0] ──────────────────────────
        if self.mvf_alpha < 0.0 || self.mvf_alpha > 1.0 {
            return Err(ServerError::config_invalid(
                "memchain.mvf_alpha",
                format!("must be in [0.0, 1.0], got {}", self.mvf_alpha),
            ));
        }

        // ── 🐛 v2.3.0: miner_interval_secs > 0 ───────────────────────
        if self.miner_interval_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.miner_interval_secs",
                "must be > 0 (seconds between miner runs)",
            ));
        }

        // ── 🐛 v2.3.0: embed dims > 0 ────────────────────────────────
        if self.embed_dim == 0 {
            return Err(ServerError::config_invalid(
                "memchain.embed_dim",
                "must be > 0",
            ));
        }
        if self.embed_max_tokens == 0 {
            return Err(ServerError::config_invalid(
                "memchain.embed_max_tokens",
                "must be > 0",
            ));
        }

        // ── v2.5.0: embed_output_dim > 0 ─────────────────────────────
        if self.embed_output_dim == 0 {
            return Err(ServerError::config_invalid(
                "memchain.embed_output_dim",
                "must be > 0",
            ));
        }

        // ── v2.3.0: Remote storage ────────────────────────────────────
        if self.allow_remote_storage {
            info!(
                "[MEMCHAIN] Remote storage enabled (max_remote_owners: {})",
                if self.max_remote_owners == 0 {
                    "unlimited".to_string()
                } else {
                    self.max_remote_owners.to_string()
                }
            );
            if self.effective_api_secret().is_none() {
                tracing::warn!(
                    "[MEMCHAIN] allow_remote_storage=true but api_secret is not set. \
                     Local MPI endpoints are unprotected. Consider setting api_secret \
                     to prevent unauthorized local access."
                );
            }
        }

        // Block Sync v1 deliberately uses a single blind coordinator. This is
        // a safety boundary, not a claim of distributed consensus.
        if self.commitment_coordinator_enabled {
            if self.mode != MemChainMode::Local {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_coordinator_enabled",
                    "requires memchain.mode = 'local' until fork choice is implemented",
                ));
            }
            if !self.blind_storage_enabled {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_coordinator_enabled",
                    "requires memchain.blind_storage_enabled = true",
                ));
            }
            if self.effective_commitment_tip_anchor_path() == Path::new(&self.db_path) {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_tip_anchor_path",
                    "must not resolve to memchain.db_path",
                ));
            }
            debug!("[MEMCHAIN_BLOCK] Canonical commitment coordinator enabled");
        }

        if self.commitment_witness_min_verified == 0 {
            return Err(ServerError::config_invalid(
                "memchain.commitment_witness_min_verified",
                "must be at least one",
            ));
        }

        if !self.commitment_witness_node_ids.is_empty()
            || self.commitment_witness_startup_required
            || self.commitment_witness_min_verified != default_commitment_witness_min_verified()
        {
            if !self.commitment_coordinator_enabled {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_witness_node_ids",
                    "requires memchain.commitment_coordinator_enabled = true",
                ));
            }
            if self.commitment_witness_node_ids.is_empty() {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_witness_startup_required",
                    "requires at least one memchain.commitment_witness_node_ids entry",
                ));
            }
            if self.commitment_witness_node_ids.len() > MAX_COMMITMENT_WITNESS_NODE_IDS {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_witness_node_ids",
                    format!("supports at most {MAX_COMMITMENT_WITNESS_NODE_IDS} pinned witnesses"),
                ));
            }

            let mut validated =
                Vec::<[u8; 32]>::with_capacity(self.commitment_witness_node_ids.len());
            for configured in &self.commitment_witness_node_ids {
                let value = configured.trim();
                let decoded = hex::decode(value).map_err(|_| {
                    ServerError::config_invalid(
                        "memchain.commitment_witness_node_ids",
                        "each entry must be a 64-character Ed25519 public key in hexadecimal",
                    )
                })?;
                let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
                    ServerError::config_invalid(
                        "memchain.commitment_witness_node_ids",
                        "each entry must decode to exactly 32 bytes",
                    )
                })?;
                if value.len() != 64 || node_id.iter().all(|byte| *byte == 0) {
                    return Err(ServerError::config_invalid(
                        "memchain.commitment_witness_node_ids",
                        "each entry must be a non-zero 64-character Ed25519 public key",
                    ));
                }
                if validated.contains(&node_id) {
                    return Err(ServerError::config_invalid(
                        "memchain.commitment_witness_node_ids",
                        "duplicate witness identities are not allowed",
                    ));
                }
                validated.push(node_id);
            }
            if self.commitment_witness_min_verified > validated.len() {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_witness_min_verified",
                    "cannot exceed the number of configured pinned witnesses",
                ));
            }
        }

        // A Block Sync v1 follower has exactly one trust root. It never falls
        // back to an arbitrary discovered peer and never produces blocks.
        if self.commitment_sync_enabled {
            if self.mode != MemChainMode::Local {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_enabled",
                    "requires memchain.mode = 'local'",
                ));
            }
            if !self.blind_storage_enabled {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_enabled",
                    "requires memchain.blind_storage_enabled = true",
                ));
            }
            if self.commitment_coordinator_enabled {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_enabled",
                    "cannot be enabled on the commitment coordinator",
                ));
            }
            let coordinator = self.commitment_coordinator_node_id.trim();
            let decoded = hex::decode(coordinator).map_err(|_| {
                ServerError::config_invalid(
                    "memchain.commitment_coordinator_node_id",
                    "must be a 64-character Ed25519 public key in hexadecimal",
                )
            })?;
            if coordinator.len() != 64
                || decoded.len() != 32
                || decoded.iter().all(|byte| *byte == 0)
            {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_coordinator_node_id",
                    "must be a non-zero 64-character Ed25519 public key in hexadecimal",
                ));
            }
            if !(5..=3_600).contains(&self.commitment_sync_interval_secs) {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_interval_secs",
                    "must be between 5 and 3600 seconds",
                ));
            }
            if !(1..=64).contains(&self.commitment_sync_max_pages_per_round) {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_max_pages_per_round",
                    "must be between 1 and 64 pages",
                ));
            }
            debug!("[MEMCHAIN_BLOCK] Pinned coordinator follower sync enabled");
        }

        // ── v2.4.0: NER Engine ────────────────────────────────────────
        if self.ner_enabled {
            if self.ner_confidence_threshold <= 0.0 || self.ner_confidence_threshold >= 1.0 {
                return Err(ServerError::config_invalid(
                    "memchain.ner_confidence_threshold",
                    format!(
                        "must be in (0.0, 1.0), got {}",
                        self.ner_confidence_threshold
                    ),
                ));
            }
            if self.ner_model_path.is_empty() {
                return Err(ServerError::config_invalid(
                    "memchain.ner_model_path",
                    "cannot be empty when ner_enabled = true",
                ));
            }
            // [M1-STABILITY-LOGGING 2026-06-30] `validate()` can run on
            // runtime policy / heartbeat paths, not only at process startup.
            // Keep the success case at debug so an enabled NER config does not
            // look like repeated model initialization in production journals.
            debug!(
                "[MEMCHAIN] NER engine enabled (model: {}, threshold: {})",
                self.ner_model_path, self.ner_confidence_threshold
            );
        }

        // ── v2.4.0: Knowledge Graph ───────────────────────────────────
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

        // ── v2.4.0: Entropy Filter ────────────────────────────────────
        if self.entropy_filter_enabled {
            if self.entropy_filter_threshold < 0.0 || self.entropy_filter_threshold > 1.0 {
                return Err(ServerError::config_invalid(
                    "memchain.entropy_filter_threshold",
                    format!(
                        "must be in [0.0, 1.0], got {}",
                        self.entropy_filter_threshold
                    ),
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

        // ── v2.4.0: Miner cognitive step warnings ────────────────────
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

        // ── v2.4.0: Vector optimization ───────────────────────────────
        if self.vector_early_termination && self.vector_saturation_threshold == 0 {
            return Err(ServerError::config_invalid(
                "memchain.vector_saturation_threshold",
                "must be > 0 when vector_early_termination is enabled",
            ));
        }

        // ── v2.5.0: SuperNode — delegates to SuperNodeConfig::validate()
        self.supernode.validate()?;

        // ── v1.0.0-MultiTenant: SaaS ──────────────────────────────────
        if self.mode == MemChainMode::Saas {
            if let Some(ref secret) = self.jwt_secret {
                if !secret.is_empty() && secret.len() < 32 {
                    return Err(ServerError::config_invalid(
                        "memchain.jwt_secret",
                        format!(
                            "must be at least 32 characters when set manually, got {}",
                            secret.len()
                        ),
                    ));
                }
            }
            if self.token_ttl_secs == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.token_ttl_secs",
                    "must be > 0",
                ));
            }
            if let Some(ref saas) = self.saas {
                saas.validate()?;
            } else {
                tracing::warn!(
                    "[MEMCHAIN] mode=saas but [memchain.saas] section is missing — \
                     server will use default SaaS configuration"
                );
            }
        }

        // ── v1.1.0-ChatRelay: delegates to ChatRelayConfig::validate()
        self.chat_relay.validate()?;

        Ok(())
    }
}

// ── Convenience methods ────────────────────────────────────────────────────

impl MemChainConfig {
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.mode != MemChainMode::Off
    }
    #[must_use]
    pub fn is_p2p(&self) -> bool {
        self.mode == MemChainMode::P2p
    }

    /// v1.0.0-MultiTenant: Returns true when running in SaaS multi-tenant mode.
    #[must_use]
    pub fn is_saas(&self) -> bool {
        self.mode == MemChainMode::Saas
    }

    /// v1.1.0-ChatRelay: Returns true when the chat relay subsystem is enabled.
    #[must_use]
    pub fn is_chat_relay_enabled(&self) -> bool {
        self.chat_relay.enabled
    }

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

    /// Returns the validated Block Sync v1 coordinator identity.
    ///
    /// `None` means follower sync is disabled or the caller bypassed config
    /// validation. Runtime code must fail closed when sync is enabled and this
    /// method returns `None`.
    #[must_use]
    pub fn commitment_sync_coordinator_node_id(&self) -> Option<[u8; 32]> {
        if !self.commitment_sync_enabled {
            return None;
        }
        let bytes = hex::decode(self.commitment_coordinator_node_id.trim()).ok()?;
        bytes.try_into().ok()
    }

    /// Returns validated pinned witness identities in operator order.
    ///
    /// Config validation rejects malformed values. `filter_map` keeps this
    /// helper panic-free for tests or callers that deliberately bypass it.
    #[must_use]
    pub fn commitment_witness_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.commitment_witness_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Resolves the coordinator-local signed tip anchor without changing old
    /// configurations. Relative explicit paths remain relative to the process
    /// working directory, matching the existing database path behavior.
    #[must_use]
    pub fn effective_commitment_tip_anchor_path(&self) -> PathBuf {
        let configured = self.commitment_tip_anchor_path.trim();
        if !configured.is_empty() {
            return PathBuf::from(configured);
        }

        let db_path = Path::new(&self.db_path);
        match db_path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                parent.join("record-commitment-tip-v1.json")
            }
            _ => PathBuf::from("record-commitment-tip-v1.json"),
        }
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
    #[must_use]
    pub fn is_supernode_enabled(&self) -> bool {
        self.supernode.enabled
    }

    /// v2.4.0: Get the effective NER tokenizer path.
    ///
    /// If `ner_tokenizer_path` is empty, defaults to `{ner_model_path}/tokenizer.json`.
    #[must_use]
    pub fn effective_ner_tokenizer_path(&self) -> String {
        if self.ner_tokenizer_path.is_empty() {
            format!("{}/tokenizer.json", self.ner_model_path)
        } else {
            self.ner_tokenizer_path.clone()
        }
    }

    /// v1.0.0-MultiTenant: Returns the effective SaasConfig.
    ///
    /// Returns a reference to the configured SaasConfig when present,
    /// or a lazily constructed default. Callers in SaaS mode should use
    /// this rather than accessing `self.saas` directly.
    ///
    /// # Performance
    /// Returns `Cow` — borrows when `saas` is `Some`, allocates when `None`.
    /// Cache the result if calling in a hot path.
    #[must_use]
    pub fn effective_saas_config(&self) -> Cow<'_, SaasConfig> {
        match &self.saas {
            Some(c) => Cow::Borrowed(c),
            None => Cow::Owned(SaasConfig::default()),
        }
    }

    #[must_use]
    pub fn is_origin_trusted(&self, origin_hex: &str, server_pubkey_hex: &str) -> bool {
        if origin_hex == server_pubkey_hex {
            return true;
        }
        if self.trusted_agents.is_empty() {
            return true;
        }
        self.trusted_agents.iter().any(|t| t == origin_hex)
    }
}

// ── Default impl ───────────────────────────────────────────────────────────

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
            embed_enabled: default_embed_enabled(),
            embed_model_path: default_embed_model_path(),
            embed_dim: default_embed_dim(),
            embed_max_tokens: default_embed_max_tokens(),
            embed_output_dim: default_embed_output_dim(),
            api_secret: None,
            allow_remote_storage: false,
            blind_storage_enabled: false,
            commitment_coordinator_enabled: false,
            commitment_sync_enabled: false,
            commitment_coordinator_node_id: String::new(),
            commitment_sync_interval_secs: default_commitment_sync_interval_secs(),
            commitment_sync_max_pages_per_round: default_commitment_sync_max_pages_per_round(),
            commitment_tip_anchor_path: String::new(),
            commitment_witness_node_ids: Vec::new(),
            commitment_witness_startup_required: false,
            commitment_witness_min_verified: default_commitment_witness_min_verified(),
            max_remote_owners: default_max_remote_owners(),
            ner_enabled: false,
            ner_model_path: default_ner_model_path(),
            ner_tokenizer_path: String::new(),
            ner_confidence_threshold: default_ner_confidence_threshold(),
            graph_enabled: false,
            graph_max_depth: default_graph_max_depth(),
            graph_max_nodes_per_hop: default_graph_max_nodes_per_hop(),
            graph_min_edge_weight: default_graph_min_edge_weight(),
            entropy_filter_enabled: false,
            entropy_filter_threshold: default_entropy_filter_threshold(),
            entropy_window_size: default_entropy_window_size(),
            entropy_window_overlap: default_entropy_window_overlap(),
            miner_entity_extraction: false,
            miner_community_detection: false,
            miner_session_summary: false,
            miner_artifact_extraction: false,
            miner_llm_endpoint: String::new(),
            vector_quantization: VectorQuantizationMode::default(),
            vector_early_termination: false,
            vector_saturation_threshold: default_vector_saturation_threshold(),
            reranker_enabled: false,
            reranker_model_path: default_reranker_model_path(),
            reranker_max_seq_length: default_reranker_max_seq_length(),
            supernode: SuperNodeConfig::default(),
            saas: None,
            jwt_secret: None,
            token_ttl_secs: default_jwt_ttl(),
            // v1.1.0-ChatRelay
            chat_relay: ChatRelayConfig::default(),
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_supernode;

    // ── Defaults ──────────────────────────────────────────────────────────

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
        assert!(mc.embed_enabled);
        assert_eq!(mc.embed_model_path, "models/minilm-l6-v2");
        assert_eq!(mc.embed_dim, 384);
        assert_eq!(mc.embed_max_tokens, 128);
        assert_eq!(mc.embed_output_dim, 384);
        assert!(mc.api_secret.is_none());
        assert!(mc.effective_api_secret().is_none());
        assert!(!mc.allow_remote_storage);
        assert!(!mc.is_remote_storage_enabled());
        assert!(!mc.commitment_coordinator_enabled);
        assert!(!mc.commitment_sync_enabled);
        assert!(mc.commitment_coordinator_node_id.is_empty());
        assert_eq!(mc.commitment_sync_interval_secs, 30);
        assert_eq!(mc.commitment_sync_max_pages_per_round, 8);
        assert!(mc.commitment_tip_anchor_path.is_empty());
        assert!(mc.commitment_witness_node_ids.is_empty());
        assert!(!mc.commitment_witness_startup_required);
        assert_eq!(mc.commitment_witness_min_verified, 1);
        assert_eq!(
            mc.effective_commitment_tip_anchor_path(),
            PathBuf::from("record-commitment-tip-v1.json")
        );
        assert_eq!(mc.max_remote_owners, 100);
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
        assert!(!mc.supernode.enabled);
        assert!(mc.supernode.providers.is_empty());
        assert!(mc.saas.is_none());
        assert!(mc.jwt_secret.is_none());
        assert_eq!(mc.token_ttl_secs, 86400);
        assert!(!mc.is_saas());
        assert!(!mc.chat_relay.enabled);
        assert!(!mc.is_chat_relay_enabled());
        assert!(!mc.is_cognitive_graph_enabled());
        assert!(!mc.has_cognitive_miner_steps());
        assert!(!mc.is_supernode_enabled());
    }

    // ── mode=off skips all validation ─────────────────────────────────────

    #[test]
    fn test_memchain_off_skips_all_validation() {
        let mc = MemChainConfig {
            mode: MemChainMode::Off,
            mvf_alpha: 999.0,
            miner_interval_secs: 0,
            embed_dim: 0,
            embed_max_tokens: 0,
            db_path: String::new(),
            ner_enabled: true,
            ner_confidence_threshold: 2.0,
            graph_max_depth: 99,
            entropy_filter_threshold: -1.0,
            supernode: SuperNodeConfig {
                enabled: true,
                providers: Vec::new(),
                ..Default::default()
            },
            jwt_secret: Some("short".into()),
            token_ttl_secs: 0,
            saas: Some(SaasConfig {
                pool_max_connections: 0,
                ..Default::default()
            }),
            chat_relay: ChatRelayConfig {
                enabled: true,
                offline_ttl_secs: 0,
                db_path: String::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    // ── api_secret ────────────────────────────────────────────────────────

    #[test]
    fn test_api_secret_too_short_rejected() {
        let mc = MemChainConfig {
            api_secret: Some("short".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_api_secret_valid() {
        let mc = MemChainConfig {
            api_secret: Some("this-is-a-long-secret-key-1234".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_api_secret_empty_is_open() {
        let mc = MemChainConfig {
            api_secret: Some(String::new()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.effective_api_secret().is_none());
    }

    // ── embed ─────────────────────────────────────────────────────────────

    #[test]
    fn test_embed_output_dim_zero_rejected() {
        let mc = MemChainConfig {
            embed_output_dim: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_dim_zero_rejected() {
        let mc = MemChainConfig {
            embed_dim: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_max_tokens_zero_rejected() {
        let mc = MemChainConfig {
            embed_max_tokens: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    // ── mvf / miner ───────────────────────────────────────────────────────

    #[test]
    fn test_mvf_alpha_out_of_range() {
        assert!(MemChainConfig {
            mvf_alpha: 1.5,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(MemChainConfig {
            mvf_alpha: -0.1,
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_mvf_alpha_boundary() {
        assert!(MemChainConfig {
            mvf_alpha: 0.0,
            ..Default::default()
        }
        .validate()
        .is_ok());
        assert!(MemChainConfig {
            mvf_alpha: 1.0,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn test_miner_interval_zero_rejected() {
        let mc = MemChainConfig {
            miner_interval_secs: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    // ── remote storage ────────────────────────────────────────────────────

    #[test]
    fn test_remote_storage_enabled() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 50,
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
    fn test_commitment_coordinator_requires_local_blind_storage() {
        assert!(MemChainConfig {
            commitment_coordinator_enabled: true,
            ..Default::default()
        }
        .validate()
        .is_err());

        assert!(MemChainConfig {
            mode: MemChainMode::P2p,
            blind_storage_enabled: true,
            commitment_coordinator_enabled: true,
            ..Default::default()
        }
        .validate()
        .is_err());

        assert!(MemChainConfig {
            blind_storage_enabled: true,
            commitment_coordinator_enabled: true,
            ..Default::default()
        }
        .validate()
        .is_ok());

        let explicit = MemChainConfig {
            db_path: "/var/lib/aeronyx/memchain.db".into(),
            commitment_tip_anchor_path: "/var/lib/aeronyx/tip.json".into(),
            ..Default::default()
        };
        assert_eq!(
            explicit.effective_commitment_tip_anchor_path(),
            PathBuf::from("/var/lib/aeronyx/tip.json")
        );

        assert!(MemChainConfig {
            blind_storage_enabled: true,
            commitment_coordinator_enabled: true,
            db_path: "/var/lib/aeronyx/memchain.db".into(),
            commitment_tip_anchor_path: "/var/lib/aeronyx/memchain.db".into(),
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_commitment_witness_pins_require_valid_coordinator_configuration() {
        let first = "11".repeat(32);
        let second = "22".repeat(32);
        let valid = MemChainConfig {
            blind_storage_enabled: true,
            commitment_coordinator_enabled: true,
            commitment_witness_node_ids: vec![first.clone(), second.clone()],
            commitment_witness_startup_required: true,
            commitment_witness_min_verified: 2,
            ..Default::default()
        };
        assert!(valid.validate().is_ok());
        assert_eq!(
            valid.commitment_witness_node_id_bytes(),
            vec![[0x11; 32], [0x22; 32]]
        );

        for invalid in [
            MemChainConfig {
                commitment_witness_node_ids: vec![first.clone()],
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_startup_required: true,
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_node_ids: vec![first.clone()],
                commitment_witness_min_verified: 0,
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_node_ids: vec![first.clone(), second.clone()],
                commitment_witness_min_verified: 3,
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_node_ids: vec!["00".repeat(32)],
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_node_ids: vec![first.clone(), first.clone()],
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_witness_node_ids: vec![
                    first.clone(),
                    second,
                    "33".repeat(32),
                    "44".repeat(32),
                ],
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn test_commitment_follower_requires_pinned_coordinator_and_safe_role() {
        let pinned = "11".repeat(32);
        let valid = MemChainConfig {
            blind_storage_enabled: true,
            commitment_sync_enabled: true,
            commitment_coordinator_node_id: pinned.clone(),
            ..Default::default()
        };
        assert!(valid.validate().is_ok());
        assert_eq!(
            valid.commitment_sync_coordinator_node_id(),
            Some([0x11; 32])
        );

        for invalid in [
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_sync_enabled: true,
                ..Default::default()
            },
            MemChainConfig {
                mode: MemChainMode::P2p,
                blind_storage_enabled: true,
                commitment_sync_enabled: true,
                commitment_coordinator_node_id: pinned.clone(),
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_coordinator_enabled: true,
                commitment_sync_enabled: true,
                commitment_coordinator_node_id: pinned.clone(),
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_sync_enabled: true,
                commitment_coordinator_node_id: "00".repeat(32),
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_sync_enabled: true,
                commitment_coordinator_node_id: pinned,
                commitment_sync_interval_secs: 4,
                ..Default::default()
            },
            MemChainConfig {
                blind_storage_enabled: true,
                commitment_sync_enabled: true,
                commitment_coordinator_node_id: "11".repeat(32),
                commitment_sync_max_pages_per_round: 65,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    // ── NER ───────────────────────────────────────────────────────────────

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
    fn test_ner_confidence_boundary() {
        for bad in [0.0f32, 1.0, -0.1, 1.1] {
            let mc = MemChainConfig {
                ner_enabled: true,
                ner_confidence_threshold: bad,
                ..Default::default()
            };
            assert!(mc.validate().is_err(), "expected err for threshold={bad}");
        }
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

    // ── Graph ─────────────────────────────────────────────────────────────

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
    fn test_graph_max_depth_rejected() {
        assert!(MemChainConfig {
            graph_enabled: true,
            graph_max_depth: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(MemChainConfig {
            graph_enabled: true,
            graph_max_depth: 4,
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_graph_max_depth_boundary() {
        for d in [1usize, 3] {
            assert!(MemChainConfig {
                graph_enabled: true,
                graph_max_depth: d,
                ..Default::default()
            }
            .validate()
            .is_ok());
        }
    }

    // ── Entropy ───────────────────────────────────────────────────────────

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
    fn test_entropy_overlap_exceeds_window_rejected() {
        let mc = MemChainConfig {
            entropy_filter_enabled: true,
            entropy_window_size: 5,
            entropy_window_overlap: 5,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    // ── Vector ────────────────────────────────────────────────────────────

    #[test]
    fn test_vector_early_term_zero_threshold_rejected() {
        let mc = MemChainConfig {
            vector_early_termination: true,
            vector_saturation_threshold: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    // ── SuperNode ─────────────────────────────────────────────────────────

    #[test]
    fn test_supernode_validate_delegation_no_providers() {
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
    fn test_supernode_valid_provider() {
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
    }

    // ── SaaS ──────────────────────────────────────────────────────────────

    #[test]
    fn test_saas_jwt_too_short_rejected() {
        let mc = MemChainConfig {
            mode: MemChainMode::Saas,
            jwt_secret: Some("tooshort".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_saas_token_ttl_zero_rejected() {
        let mc = MemChainConfig {
            mode: MemChainMode::Saas,
            token_ttl_secs: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_saas_config_pool_zero_rejected() {
        let mc = MemChainConfig {
            mode: MemChainMode::Saas,
            saas: Some(SaasConfig {
                pool_max_connections: 0,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_local_mode_skips_saas_checks() {
        let mc = MemChainConfig {
            mode: MemChainMode::Local,
            saas: Some(SaasConfig {
                pool_max_connections: 0,
                ..Default::default()
            }),
            jwt_secret: Some("x".into()),
            token_ttl_secs: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_effective_saas_config_fallback() {
        let mc = MemChainConfig {
            mode: MemChainMode::Saas,
            saas: None,
            ..Default::default()
        };
        let esc = mc.effective_saas_config();
        assert_eq!(esc.pool_max_connections, 100);
    }

    #[test]
    fn test_effective_saas_config_custom() {
        let mc = MemChainConfig {
            mode: MemChainMode::Saas,
            saas: Some(SaasConfig {
                pool_max_connections: 50,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(mc.effective_saas_config().pool_max_connections, 50);
    }

    // ── ChatRelay ─────────────────────────────────────────────────────────

    #[test]
    fn test_chat_relay_disabled_by_default_in_memchain() {
        let mc = MemChainConfig::default();
        assert!(!mc.chat_relay.enabled);
        assert!(!mc.is_chat_relay_enabled());
    }

    #[test]
    fn test_chat_relay_enabled_propagates() {
        let mc = MemChainConfig {
            chat_relay: ChatRelayConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(mc.is_chat_relay_enabled());
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_invalid_ttl_rejected_via_memchain() {
        let mc = MemChainConfig {
            chat_relay: ChatRelayConfig {
                enabled: true,
                offline_ttl_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    // ── Convenience methods ───────────────────────────────────────────────

    #[test]
    fn test_is_cognitive_graph_enabled() {
        assert!(MemChainConfig {
            ner_enabled: true,
            graph_enabled: true,
            ..Default::default()
        }
        .is_cognitive_graph_enabled());
        assert!(!MemChainConfig {
            ner_enabled: true,
            graph_enabled: false,
            ..Default::default()
        }
        .is_cognitive_graph_enabled());
        assert!(!MemChainConfig {
            ner_enabled: false,
            graph_enabled: true,
            ..Default::default()
        }
        .is_cognitive_graph_enabled());
    }

    #[test]
    fn test_effective_ner_tokenizer_path() {
        let mc = MemChainConfig::default();
        assert_eq!(
            mc.effective_ner_tokenizer_path(),
            "models/gliner/tokenizer.json"
        );
        let mc2 = MemChainConfig {
            ner_tokenizer_path: "/custom/tokenizer.json".into(),
            ..Default::default()
        };
        assert_eq!(mc2.effective_ner_tokenizer_path(), "/custom/tokenizer.json");
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
            "aaaa0000000000000000000000000000000000000000000000000000000000aa",
            server
        ));
        assert!(!config.is_origin_trusted(
            "cccc0000000000000000000000000000000000000000000000000000000000cc",
            server
        ));
    }
}
