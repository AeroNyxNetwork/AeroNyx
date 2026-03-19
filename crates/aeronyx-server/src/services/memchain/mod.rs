// ============================================
// File: crates/aeronyx-server/src/services/memchain/mod.rs
// ============================================
//! # MemChain Storage Engine
//!
//! ## Submodules
//! - [`storage`] + [`storage_crypto`] + [`storage_ops`]: SQLite core (split v2.2.0)
//! - [`storage_graph`]: Cognitive graph CRUD (Episodes/Entities/Edges/etc.) (v2.4.0+Search split)
//! - [`storage_miner`]: Miner step support + EntityTimeline (v2.4.0+Search split)
//! - [`storage_supernode`]: cognitive_tasks + llm_usage_log CRUD (v2.5.0+SuperNode)
//! - [`storage_fts`]: FTS5 BM25 full-text search + snippet highlight (v2.4.0+BM25)
//! - [`vector`]: Partitioned vector index for cosine similarity search
//! - [`mvf`]: Multi-Variate Feature scoring (10-dim, v2.4.0)
//! - [`graph`]: Co-occurrence graph + BFS + community detection (v2.4.0)
//! - [`embed`]: Local embedding inference (ort + tokenizers)
//! - [`ner`]: Local GLiNER NER inference (v2.4.0)
//! - [`query_analyzer`]: Query analysis — entity detection + classification (v2.4.0)
//! - [`quantize`]: Scalar Quantization float32→uint8 (v2.4.0)
//! - [`reranker`]: Cross-encoder reranking ms-marco-MiniLM-L-6-v2 (v2.4.0+Reranker)
//! - [`llm_provider`]: LlmProvider trait + shared types (v2.5.0+SuperNode)
//! - [`llm_openai`]: OpenAI-compatible provider (v2.5.0+SuperNode)
//! - [`llm_anthropic`]: Anthropic Messages API provider (v2.5.0+SuperNode)
//! - [`llm_router`]: Task routing + fallback + cost estimation (v2.5.0+SuperNode)
//! - [`task_worker`]: Async cognitive task queue worker (v2.5.0+SuperNode)
//! - [`prompts`]: Prompt template engine for 6 cognitive task types (v2.5.0+SuperNode Phase B)
//! - [`mempool`]: In-memory Fact buffer (legacy/deprecated)
//! - [`aof`]: Append-Only File persistence (legacy/deprecated)
//!
//! ## Storage Split History
//! v2.2.0: storage.rs → storage.rs + storage_crypto.rs + storage_ops.rs
//! v2.4.0+Search: storage_ops.rs → storage_ops.rs + storage_graph.rs + storage_miner.rs
//! v2.5.0+SuperNode: added storage_supernode.rs
//!
//! ⚠️ Important Note for Next Developer:
//! - All storage_*.rs files use `impl MemoryStorage` extension pattern.
//!   Rust allows multiple impl blocks across files within the same crate.
//! - When adding a new storage submodule: declare `pub mod` here AND add re-exports.
//! - Re-export order: storage types → engines → SuperNode → legacy.
//! - CognitiveTaskType is defined in config_supernode.rs and re-exported through
//!   llm_provider.rs. Do NOT define it here or in storage_supernode.rs.
//! - PrivacyLevel is defined in config_supernode.rs and re-exported through prompts.rs.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine
//! v2.1.0 - Added storage, vector, mvf, graph modules
//! v2.1.0+Embed - Added embed module
//! v2.2.0 - 🌟 Split storage into 3 files; storage_ops added
//! v2.4.0-GraphCognition - 🌟 Added ner, query_analyzer, quantize + re-exports
//! v2.4.0+Search - 🌟 Split storage_ops → storage_graph + storage_miner
//! v2.4.0+Reranker - 🌟 Added reranker module + RerankerEngine re-export
//! v2.5.0+SuperNode Phase A - 🌟 Added storage_supernode, llm_provider, llm_openai,
//!   llm_anthropic, llm_router, task_worker modules + re-exports
//! v2.5.0+SuperNode Phase B - 🌟 Added prompts module + PrivacyLevel re-export
//! v2.5.0+Unify - 🔧 [BUG FIX] Fixed re-exports to match actual types defined in
//!   storage_supernode.rs. CognitiveTaskType now re-exported from llm_provider
//!   (which itself re-exports from config_supernode). Removed non-existent type
//!   re-exports (LlmUsageStats, ProviderUsage, TaskTypeUsage) that were never
//!   defined in storage_supernode.rs.

// ── Storage engine ──
pub mod storage;
pub mod storage_crypto;
pub mod storage_ops;
// v2.4.0+Search: Cognitive graph CRUD (split from storage_ops.rs)
pub mod storage_graph;
// v2.4.0+Search: Miner step support + entity timeline (split from storage_ops.rs)
pub mod storage_miner;
// v2.5.0+SuperNode: Task queue + usage log CRUD
pub mod storage_supernode;
// v2.4.0+BM25: FTS5 full-text search
pub mod storage_fts;

// ── Cognitive engine ──
pub mod vector;
pub mod mvf;
pub mod graph;

// ── Local inference engines ──
pub mod embed;
// v2.4.0: GLiNER NER engine
pub mod ner;
// v2.4.0: Query analyzer
pub mod query_analyzer;
// v2.4.0: Scalar quantization
pub mod quantize;
// v2.4.0+Reranker: Cross-encoder reranker
pub mod reranker;

// ── v2.5.0+SuperNode: LLM provider infrastructure ──
pub mod llm_provider;
pub mod llm_openai;
pub mod llm_anthropic;
pub mod llm_router;
pub mod task_worker;
// v2.5.0+SuperNode Phase B: Prompt template engine
pub mod prompts;

// ── Legacy engine (deprecated) ──
pub mod aof;
pub mod mempool;

// ============================================
// Re-exports
// ============================================

// ── Storage core ──
pub use storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow};
pub use storage_crypto::{derive_record_key, derive_rawlog_key, decrypt_rawlog_content_pub};
pub use storage_ops::{OverviewRecord, OverviewData};

// ── Storage graph types (v2.4.0+Search) ──
pub use storage_graph::{
    EntityRow,
    KnowledgeEdgeRow,
    SessionRow,
    CommunityRow,
    ProjectRow,
    ArtifactRow,
    GraphStats,
};

// ── Storage miner types (v2.4.0+Search) ──
pub use storage_miner::EntityTimelineEntry;

// ── Storage SuperNode types (v2.5.0+SuperNode) ──
// v2.5.0+Unify: Only re-export types that actually exist in storage_supernode.rs.
// CognitiveTaskRow is the task queue row struct.
// LlmUsageStats, ProviderUsage, TaskTypeUsage were listed in the original mod.rs
// but never defined in storage_supernode.rs — they need to be added there or
// the re-exports removed. For now, only re-export what exists.
pub use storage_supernode::CognitiveTaskRow;
// TODO: Add these re-exports once the types are defined in storage_supernode.rs:
// pub use storage_supernode::{LlmUsageStats, ProviderUsage, TaskTypeUsage};

// ── Vector ──
pub use vector::{
    VectorIndex, SearchResult, DedupResult,
    cosine_similarity, compute_recall_score,
    dedup_threshold_for_layer, EPISODE_DEDUP_WINDOW_SECS,
};

// ── Embedding engine ──
pub use embed::EmbedEngine;

// ── NER engine (v2.4.0) ──
pub use ner::{NerEngine, DetectedEntity};

// ── Query analyzer (v2.4.0) ──
pub use query_analyzer::{analyze_query, QueryAnalysis, QueryType, MatchedEntity};

// ── Scalar quantization (v2.4.0) ──
pub use quantize::ScalarQuantizer;

// ── Cross-encoder reranker (v2.4.0+Reranker) ──
pub use reranker::RerankerEngine;

// ── LLM provider infrastructure (v2.5.0+SuperNode) ──
// v2.5.0+Unify: CognitiveTaskType is re-exported from llm_provider, which itself
// re-exports from config_supernode.rs (the single source of truth).
pub use llm_provider::{
    LlmProvider, ChatRequest, ChatResponse, ChatMessage,
    TokenUsage, CognitiveTaskType, LlmError,
};
pub use llm_openai::OpenAiCompatProvider;
pub use llm_anthropic::AnthropicProvider;
pub use llm_router::LlmRouter;
pub use task_worker::TaskWorker;

// ── Prompt template engine (v2.5.0+SuperNode Phase B) ──
// v2.5.0+Unify: PrivacyLevel is re-exported from prompts, which itself
// re-exports from config_supernode.rs (the single source of truth).
pub use prompts::{PrivacyLevel, EntityDescriptionInput};

// ── Legacy ──
pub use aof::AofWriter;
pub use mempool::MemPool;
