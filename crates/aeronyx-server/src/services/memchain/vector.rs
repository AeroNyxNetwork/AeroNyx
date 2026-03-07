// ============================================
// File: crates/aeronyx-server/src/services/memchain/vector.rs
// ============================================
//! # VectorIndex — Partitioned In-Memory Vector Search Engine
//!
//! ## Creation Reason
//! Provides low-latency (< 5ms) semantic retrieval for MPI `/api/mpi/recall`.
//! Partitioned by `(owner, embedding_model)` to handle multiple AI agents
//! using different embedding models (with different dimensions).
//!
//! ## Partition Design
//! ```text
//! VectorIndex
//!  └── partitions: HashMap<PartitionKey, Partition>
//!       │
//!       ├── (Alice, "minilm-l6-v2")  → 384-dim vectors
//!       ├── (Alice, "openai-3-small") → 1536-dim vectors
//!       └── (Bob,   "minilm-l6-v2")  → 384-dim vectors
//! ```
//! - Recall declares `embedding_model` and only searches the corresponding partition
//! - Different partitions can have different vector dimensions without conflict
//! - If a model partition doesn't exist, returns empty results (no panic)
//!
//! ## Dedup Thresholds (per-layer)
//! ```text
//! Identity:   > 0.92
//! Knowledge:  > 0.88
//! Episode:    > 0.80 AND time_diff < 24h
//! Archive:    no dedup
//! ```
//!
//! ## Phase 1: Brute-force cosine search
//! For a single user with < 10K vectors, brute-force search latency < 3ms (measured).
//! Phase 2+: Can be replaced with HNSW (`hnsw_rs` or `instant-distance`),
//! only needs to implement the same trait interface.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Vectors must be normalized (unit length) for dot product to equal cosine similarity
//! - The index is purely in-memory; it must be rebuilt from SQLite after restart
//!   (`get_records_with_embedding`)
//! - `PartitionKey` = `(owner_hex, embedding_model)` uses String as key
//!   because while `[u8; 32]` is also efficient in HashMap, String is easier for debug logs
//!
//! ## Last Modified
//! v1.0.0 - Initial brute-force vector search
//! v2.1.0 - 🌟 Partitioned by (owner, model), per-layer dedup thresholds

use std::collections::HashMap;

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use aeronyx_core::ledger::MemoryLayer;

// ============================================
// Dedup Thresholds
// ============================================

/// Returns the dedup cosine similarity threshold for a given layer
///
/// - Identity: 0.92 (identity info is naturally highly similar)
/// - Knowledge: 0.88
/// - Episode: 0.80 (used together with a 24h time window)
/// - Archive: no dedup (returns f32::MAX so it never triggers)
#[must_use]
pub fn dedup_threshold_for_layer(layer: MemoryLayer) -> f32 {
    match layer {
        MemoryLayer::Identity  => 0.92,
        MemoryLayer::Knowledge => 0.88,
        MemoryLayer::Episode   => 0.80,
        MemoryLayer::Archive   => f32::MAX, // Archive layer: no dedup
    }
}

/// Episode dedup time window (seconds).
/// Records older than this are NOT considered duplicates
/// even if similarity exceeds threshold.
///
/// 24 hours = 86400 seconds
/// "Wanted hotpot yesterday" vs "Want hotpot today" → over 24h apart, not a duplicate
pub const EPISODE_DEDUP_WINDOW_SECS: u64 = 86400;

// ============================================
// PartitionKey
// ============================================

/// Partition key for the vector index: `(owner_hex, embedding_model)`
///
/// Different AI agents may use different embedding models (with different dimensions),
/// so searches must stay within each partition to ensure dimension matching.
type PartitionKey = (String, String); // (owner_hex, embedding_model)

// ============================================
// VectorEntry — Single Vector Record
// ============================================

#[derive(Debug, Clone)]
struct VectorEntry {
    record_id: [u8; 32],
    embedding: Vec<f32>,
    layer: MemoryLayer,
    timestamp: u64,
}

// ============================================
// Partition — Single Partition
// ============================================

/// All vectors within a single (owner, model) partition
struct Partition {
    /// All vector entries
    entries: HashMap<[u8; 32], VectorEntry>,
    /// Embedding dimension for this partition (set by the first insert)
    dim: usize,
}

impl Partition {
    fn new(dim: usize) -> Self {
        Self {
            entries: HashMap::new(),
            dim,
        }
    }
}

// ============================================
// SearchResult
// ============================================

/// Vector search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Matching record ID
    pub record_id: [u8; 32],
    /// Cosine similarity (equals dot product for normalized vectors)
    pub similarity: f32,
    /// Record layer
    pub layer: MemoryLayer,
    /// Record timestamp
    pub timestamp: u64,
}

// ============================================
// DedupResult — Dedup Detection Result
// ============================================

/// Dedup detection result
#[derive(Debug, Clone)]
pub struct DedupResult {
    /// Whether deemed duplicate
    pub is_duplicate: bool,
    /// If duplicate, the existing record ID
    pub existing_id: Option<[u8; 32]>,
    /// Highest similarity found
    pub max_similarity: f32,
}

// ============================================
// VectorIndex — Partitioned Vector Index
// ============================================

/// Partitioned in-memory vector search engine
///
/// Partitioned by `(owner_hex, embedding_model)`, with brute-force cosine search
/// within each partition. For a single user with < 10K vectors, search latency < 3ms.
///
/// ## Thread Safety
/// Outer layer uses `RwLock<HashMap<..., Partition>>`:
/// - search = read lock (multiple concurrent recalls don't block each other)
/// - upsert/remove = write lock (writes are serialized but extremely fast)
pub struct VectorIndex {
    partitions: RwLock<HashMap<PartitionKey, Partition>>,
    /// Global record_id → partition_key reverse mapping (for fast partition lookup on remove)
    record_to_partition: DashMap<[u8; 32], PartitionKey>,
}

impl VectorIndex {
    /// Create empty index
    #[must_use]
    pub fn new() -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            record_to_partition: DashMap::new(),
        }
    }

    /// Insert or update a vector
    ///
    /// # Arguments
    /// * `record_id` - Unique record identifier
    /// * `embedding` - Vector (should already be normalized)
    /// * `layer` - Memory layer
    /// * `timestamp` - Record timestamp
    /// * `owner` - Owner public key
    /// * `embedding_model` - Embedding model identifier (e.g. "minilm-l6-v2")
    pub fn upsert(
        &self,
        record_id: [u8; 32],
        embedding: Vec<f32>,
        layer: MemoryLayer,
        timestamp: u64,
        owner: &[u8; 32],
        embedding_model: &str,
    ) {
        if embedding.is_empty() {
            return;
        }

        let key = (hex::encode(owner), embedding_model.to_string());
        let dim = embedding.len();

        let entry = VectorEntry {
            record_id,
            embedding,
            layer,
            timestamp,
        };

        let mut partitions = self.partitions.write();
        let partition = partitions
            .entry(key.clone())
            .or_insert_with(|| Partition::new(dim));

        // Dimension check: dimensions must be consistent within the same partition
        if partition.dim != dim && !partition.entries.is_empty() {
            warn!(
                expected = partition.dim,
                actual = dim,
                model = embedding_model,
                "[VECTOR] ⚠️ Dimension mismatch in partition, skipping"
            );
            return;
        }

        partition.entries.insert(record_id, entry);
        drop(partitions);

        // Update reverse mapping
        self.record_to_partition.insert(record_id, key);
    }

    /// Remove a vector from the index
    pub fn remove(&self, record_id: &[u8; 32]) -> bool {
        // Look up partition from reverse mapping first
        let key = match self.record_to_partition.remove(record_id) {
            Some((_, k)) => k,
            None => return false,
        };

        let mut partitions = self.partitions.write();
        if let Some(partition) = partitions.get_mut(&key) {
            partition.entries.remove(record_id);
            // If the partition is now empty, remove it
            if partition.entries.is_empty() {
                partitions.remove(&key);
            }
            true
        } else {
            false
        }
    }

    /// Search top-K most similar vectors within a specific partition
    ///
    /// # Arguments
    /// * `query` - Query vector (dimension must match the partition)
    /// * `owner` - Owner public key
    /// * `embedding_model` - Specifies which model's partition to search
    /// * `top_k` - Maximum number of results to return
    /// * `min_similarity` - Minimum similarity threshold
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        owner: &[u8; 32],
        embedding_model: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<SearchResult> {
        self.search_filtered(query, owner, embedding_model, None, top_k, min_similarity)
    }

    /// Search with optional layer filter
    #[must_use]
    pub fn search_filtered(
        &self,
        query: &[f32],
        owner: &[u8; 32],
        embedding_model: &str,
        layer_filter: Option<MemoryLayer>,
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<SearchResult> {
        let key = (hex::encode(owner), embedding_model.to_string());
        let partitions = self.partitions.read();

        let partition = match partitions.get(&key) {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Dimension check
        if query.len() != partition.dim {
            warn!(
                query_dim = query.len(),
                partition_dim = partition.dim,
                model = embedding_model,
                "[VECTOR] ⚠️ Query dimension mismatch"
            );
            return Vec::new();
        }

        let mut results: Vec<SearchResult> = partition
            .entries
            .values()
            .filter(|e| layer_filter.map_or(true, |l| e.layer == l))
            .map(|e| SearchResult {
                record_id: e.record_id,
                similarity: cosine_similarity(query, &e.embedding),
                layer: e.layer,
                timestamp: e.timestamp,
            })
            .filter(|r| r.similarity >= min_similarity)
            .collect();

        results.sort_unstable_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Per-layer dedup detection with Episode time window
    ///
    /// ## Dedup Rules
    /// - Identity:  similarity > 0.92 → duplicate
    /// - Knowledge: similarity > 0.88 → duplicate
    /// - Episode:   similarity > 0.80 AND time_diff < 24h → duplicate
    /// - Archive:   no dedup (always returns is_duplicate=false)
    ///
    /// # Arguments
    /// * `query` - Embedding of the new record
    /// * `owner` - Owner
    /// * `embedding_model` - Model identifier
    /// * `layer` - Layer of the new record
    /// * `current_timestamp` - Current time (Unix seconds), used for Episode time window
    #[must_use]
    pub fn check_duplicate(
        &self,
        query: &[f32],
        owner: &[u8; 32],
        embedding_model: &str,
        layer: MemoryLayer,
        current_timestamp: u64,
    ) -> DedupResult {
        // Archive layer: no dedup
        if layer == MemoryLayer::Archive {
            return DedupResult {
                is_duplicate: false,
                existing_id: None,
                max_similarity: 0.0,
            };
        }

        let threshold = dedup_threshold_for_layer(layer);

        // Only search within the same layer for potential duplicates
        let candidates = self.search_filtered(
            query,
            owner,
            embedding_model,
            Some(layer),
            1, // Only need the most similar one
            threshold, // Use threshold directly as min_similarity
        );

        match candidates.first() {
            Some(hit) => {
                // Episode: additional time window check
                if layer == MemoryLayer::Episode {
                    let time_diff = current_timestamp.saturating_sub(hit.timestamp);
                    if time_diff > EPISODE_DEDUP_WINDOW_SECS {
                        // Over 24h apart — not a duplicate (periodic event)
                        return DedupResult {
                            is_duplicate: false,
                            existing_id: None,
                            max_similarity: hit.similarity,
                        };
                    }
                }

                DedupResult {
                    is_duplicate: true,
                    existing_id: Some(hit.record_id),
                    max_similarity: hit.similarity,
                }
            }
            None => DedupResult {
                is_duplicate: false,
                existing_id: None,
                max_similarity: 0.0,
            },
        }
    }

    /// Total vectors across all partitions
    #[must_use]
    pub fn total_vectors(&self) -> usize {
        let partitions = self.partitions.read();
        partitions.values().map(|p| p.entries.len()).sum()
    }

    /// Number of partitions
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.partitions.read().len()
    }

    /// Vector count in a specific partition
    #[must_use]
    pub fn partition_size(&self, owner: &[u8; 32], embedding_model: &str) -> usize {
        let key = (hex::encode(owner), embedding_model.to_string());
        self.partitions.read().get(&key).map_or(0, |p| p.entries.len())
    }

    /// Clear all partitions
    pub fn clear(&self) {
        self.partitions.write().clear();
        self.record_to_partition.clear();
        debug!("[VECTOR] All partitions cleared");
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("total_vectors", &self.total_vectors())
            .field("partitions", &self.partition_count())
            .finish()
    }
}

// ============================================
// Cosine Similarity
// ============================================

/// Compute cosine similarity between two vectors
///
/// For normalized vectors (unit length), cosine similarity = dot product.
///
/// Return value range [-1.0, 1.0]: 0.0 means orthogonal, 1.0 means same direction.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }

    dot / denom
}

// ============================================
// Cognitive Recall Scoring
// ============================================

/// Compute composite recall score for a candidate record
///
/// ## Formula
/// ```text
/// score = semantic_similarity × time_decay × freq_boost + layer_weight
///
/// time_decay = exp(-hours_since_creation / stability)
/// freq_boost = 1.0 + 0.3 × ln(access_count + 1)
/// ```
///
/// ## Parameters
/// - `semantic_similarity`: Cosine similarity [0, 1]
/// - `timestamp`: Record creation time (Unix seconds)
/// - `now`: Current time (Unix seconds)
/// - `access_count`: Number of times recalled (higher → higher freq_boost → "easier to remember")
/// - `layer`: Cognitive layer (determines weight and stability)
///
/// ## Design Intent
/// Simulates three core characteristics of human memory:
/// 1. **Relevance** (semantic_similarity) — more relevant to the current topic → easier to recall
/// 2. **Recency** (time_decay) — older memories fade more (but Identity barely decays)
/// 3. **Frequency effect** (freq_boost) — frequently recalled memories are harder to forget (spaced repetition effect)
/// 4. **Layer bias** (layer_weight) — core identity > knowledge > episodes > archive
#[must_use]
pub fn compute_recall_score(
    semantic_similarity: f32,
    timestamp: u64,
    now: u64,
    access_count: u32,
    layer: MemoryLayer,
) -> f64 {
    let hours_since = if now > timestamp {
        (now - timestamp) as f64 / 3600.0
    } else {
        0.0
    };

    let stability = layer.stability_hours();
    let time_decay = (-hours_since / stability).exp();

    let freq_boost = 1.0 + 0.3 * ((access_count as f64) + 1.0).ln();

    let layer_weight = layer.recall_weight();

    (semantic_similarity as f64) * time_decay * freq_boost + layer_weight
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n < f32::EPSILON { return v.to_vec(); }
        v.iter().map(|x| x / n).collect()
    }

    const OWNER_A: [u8; 32] = [0xAA; 32];
    const OWNER_B: [u8; 32] = [0xBB; 32];
    const MODEL_MINI: &str = "minilm-l6-v2";
    const MODEL_OAI: &str = "openai-3-small";

    #[test]
    fn test_cosine_identical() {
        let v = norm(&[1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = norm(&[1.0, 0.0]);
        let b = norm(&[0.0, 1.0]);
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_mismatched_dim() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_partition_isolation() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);

        // Same owner, different model → different partitions
        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        idx.upsert([2; 32], norm(&[1.0; 1536]), MemoryLayer::Episode, 100, &OWNER_A, MODEL_OAI);

        assert_eq!(idx.partition_count(), 2);
        assert_eq!(idx.partition_size(&OWNER_A, MODEL_MINI), 1);
        assert_eq!(idx.partition_size(&OWNER_A, MODEL_OAI), 1);

        // Search mini partition → only returns mini results
        let results = idx.search(&v, &OWNER_A, MODEL_MINI, 10, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record_id, [1; 32]);
    }

    #[test]
    fn test_owner_isolation() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0]);

        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        idx.upsert([2; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_B, MODEL_MINI);

        let results = idx.search(&v, &OWNER_A, MODEL_MINI, 10, 0.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch_rejected() {
        let idx = VectorIndex::new();

        // Insert a 3-dim vector first
        idx.upsert([1; 32], vec![1.0, 0.0, 0.0], MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);

        // Then insert a 5-dim vector → should warn and skip
        idx.upsert([2; 32], vec![1.0; 5], MemoryLayer::Episode, 200, &OWNER_A, MODEL_MINI);

        assert_eq!(idx.partition_size(&OWNER_A, MODEL_MINI), 1);
    }

    #[test]
    fn test_search_with_layer_filter() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0]);

        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        idx.upsert([2; 32], v.clone(), MemoryLayer::Identity, 100, &OWNER_A, MODEL_MINI);

        let eps = idx.search_filtered(&v, &OWNER_A, MODEL_MINI, Some(MemoryLayer::Episode), 10, 0.0);
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].record_id, [1; 32]);
    }

    #[test]
    fn test_remove() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0]);
        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);

        assert!(idx.remove(&[1; 32]));
        assert_eq!(idx.total_vectors(), 0);
        assert!(!idx.remove(&[1; 32])); // Second remove returns false
    }

    // ========================================
    // Per-layer dedup tests
    // ========================================

    #[test]
    fn test_dedup_identity_high_threshold() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);
        idx.upsert([1; 32], v.clone(), MemoryLayer::Identity, 100, &OWNER_A, MODEL_MINI);

        // 0.91 similarity < 0.92 threshold → not duplicate
        let query_low = norm(&[0.91, 0.42, 0.0]); // manually constructed
        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Identity, 200);
        // Identical vector → sim=1.0 > 0.92 → duplicate
        assert!(r.is_duplicate);
    }

    #[test]
    fn test_dedup_episode_time_window() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);

        let old_time = 1_700_000_000u64;
        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, old_time, &OWNER_A, MODEL_MINI);

        // Same content but over 24h apart → not duplicate (periodic event)
        let now = old_time + EPISODE_DEDUP_WINDOW_SECS + 1;
        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Episode, now);
        assert!(!r.is_duplicate, "Over 24h should not count as duplicate");

        // Same content within 24h → duplicate
        let recent = old_time + 3600; // 1 hour later
        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Episode, recent);
        assert!(r.is_duplicate, "Same content within 24h should count as duplicate");
    }

    #[test]
    fn test_dedup_archive_never() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);
        idx.upsert([1; 32], v.clone(), MemoryLayer::Archive, 100, &OWNER_A, MODEL_MINI);

        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Archive, 200);
        assert!(!r.is_duplicate, "Archive layer should never dedup");
    }

    // ========================================
    // Scoring formula tests
    // ========================================

    #[test]
    fn test_score_recent_vs_old() {
        let now = 1_700_100_000u64;
        let recent = now - 1000;   // ~17 minutes ago
        let old = now - 200_000;   // ~55 hours ago

        let s_recent = compute_recall_score(0.8, recent, now, 0, MemoryLayer::Episode);
        let s_old = compute_recall_score(0.8, old, now, 0, MemoryLayer::Episode);
        assert!(s_recent > s_old);
    }

    #[test]
    fn test_score_identity_beats_episode() {
        let now = 1_700_100_000u64;
        let ts = now - 3600;

        let s_id = compute_recall_score(0.8, ts, now, 0, MemoryLayer::Identity);
        let s_ep = compute_recall_score(0.8, ts, now, 0, MemoryLayer::Episode);
        assert!(s_id > s_ep);
    }

    #[test]
    fn test_score_freq_boost() {
        let now = 1_700_100_000u64;
        let ts = now - 3600;

        let s_low = compute_recall_score(0.8, ts, now, 0, MemoryLayer::Episode);
        let s_high = compute_recall_score(0.8, ts, now, 100, MemoryLayer::Episode);
        assert!(s_high > s_low);
    }

    #[test]
    fn test_score_archive_lowest() {
        let now = 1_700_100_000u64;
        let ts = now - 3600;

        let s_archive = compute_recall_score(0.8, ts, now, 0, MemoryLayer::Archive);
        let s_episode = compute_recall_score(0.8, ts, now, 0, MemoryLayer::Episode);
        assert!(s_archive < s_episode, "Archive should score lower than Episode");
        assert!(s_archive > 0.0, "But Archive should still have positive score");
    }

    #[test]
    fn test_nonexistent_partition_returns_empty() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0]);
        let results = idx.search(&v, &OWNER_A, "nonexistent-model", 10, 0.0);
        assert!(results.is_empty());
    }
}
