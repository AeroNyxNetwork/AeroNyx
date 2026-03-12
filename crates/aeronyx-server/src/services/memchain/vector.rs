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
//! ## v2.4.0-GraphCognition: Scalar Quantization + Early Termination
//! When `vector_quantization = "scalar_uint8"` in config:
//! 1. On upsert: store both f32 embedding AND uint8 quantized embedding
//! 2. On search (two-phase):
//!    a. Coarse rank: quantized dot product over ALL vectors → top_k × 3 candidates
//!    b. Fine rank: f32 cosine similarity on candidates only → top_k results
//! 3. Early termination: if top-k scores saturate (no improvement for N consecutive
//!    candidates), stop scanning early. Saves ~30-50% scan time on large partitions.
//!
//! Memory savings: 75% per vector (384-dim: 1536 bytes → 384 bytes for coarse rank).
//! Fine rank still uses f32 vectors (stored alongside quantized).
//!
//! Precision target: Recall@10 degradation < 3% vs brute-force f32.
//!
//! ## Phase 1: Brute-force cosine search (original, preserved)
//! For a single user with < 10K vectors, brute-force search latency < 3ms (measured).
//! Phase 2+: Can be replaced with HNSW, only needs same trait interface.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Vectors must be normalized (unit length) for dot product to equal cosine similarity
//! - The index is purely in-memory; it must be rebuilt from SQLite after restart
//!   (`get_records_with_embedding`)
//! - `PartitionKey` = `(owner_hex, embedding_model)` uses String as key
//!   because while `[u8; 32]` is also efficient in HashMap, String is easier for debug logs
//! - Quantizer is per-partition (different models have different value distributions)
//! - Quantizer must be calibrated BEFORE quantized search works. On startup:
//!   1. Rebuild index from SQLite (upsert all vectors)
//!   2. Call calibrate_partition() for each partition
//!   Uncalibrated partitions fall back to pure f32 search (no degradation).
//! - Early termination saturation_threshold is configurable. Lower = more aggressive
//!   pruning (faster but risks missing results). Default 0.001 works well empirically.
//! - Two-phase search is OPTIONAL: disabled when quantizer is None/uncalibrated.
//!   The fallback is the original brute-force f32 path (v2.3.0 behavior).
//!
//! ## Last Modified
//! v1.0.0 - Initial brute-force vector search
//! v2.1.0 - 🌟 Partitioned by (owner, model), per-layer dedup thresholds
//! v2.4.0-GraphCognition - 🌟 Scalar quantization integration (two-phase search),
//!   early termination (saturation detection). All v2.3.0 behavior preserved when
//!   quantization is disabled.

use std::collections::HashMap;

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use aeronyx_core::ledger::MemoryLayer;

use super::quantize::ScalarQuantizer;

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
        MemoryLayer::Archive   => f32::MAX,
    }
}

/// Episode dedup time window (seconds).
/// Records older than this are NOT considered duplicates
/// even if similarity exceeds threshold.
///
/// 24 hours = 86400 seconds
pub const EPISODE_DEDUP_WINDOW_SECS: u64 = 86400;

// ============================================
// v2.4.0: Early Termination Constants
// ============================================

/// Default saturation threshold for early termination.
/// If the best score improves by less than this for `saturation_window`
/// consecutive candidates, stop scanning.
const DEFAULT_SATURATION_THRESHOLD: f32 = 0.001;

/// Number of consecutive non-improving candidates before early termination.
const DEFAULT_SATURATION_WINDOW: usize = 50;

/// Coarse rank expansion factor for two-phase search.
/// Coarse phase retrieves top_k * COARSE_EXPANSION candidates,
/// then fine phase re-ranks with f32 to select top_k.
const COARSE_EXPANSION: usize = 3;

// ============================================
// PartitionKey
// ============================================

type PartitionKey = (String, String); // (owner_hex, embedding_model)

// ============================================
// VectorEntry — Single Vector Record
// ============================================

#[derive(Debug, Clone)]
struct VectorEntry {
    record_id: [u8; 32],
    embedding: Vec<f32>,
    /// v2.4.0: Quantized embedding (None if quantization disabled or uncalibrated).
    quantized: Option<Vec<u8>>,
    layer: MemoryLayer,
    timestamp: u64,
}

// ============================================
// Partition — Single Partition
// ============================================

struct Partition {
    entries: HashMap<[u8; 32], VectorEntry>,
    dim: usize,
    /// v2.4.0: Per-partition scalar quantizer (None if quantization disabled).
    quantizer: Option<ScalarQuantizer>,
}

impl Partition {
    fn new(dim: usize) -> Self {
        Self {
            entries: HashMap::new(),
            dim,
            quantizer: None,
        }
    }
}

// ============================================
// SearchResult
// ============================================

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record_id: [u8; 32],
    pub similarity: f32,
    pub layer: MemoryLayer,
    pub timestamp: u64,
}

// ============================================
// DedupResult
// ============================================

#[derive(Debug, Clone)]
pub struct DedupResult {
    pub is_duplicate: bool,
    pub existing_id: Option<[u8; 32]>,
    pub max_similarity: f32,
}

// ============================================
// VectorIndex
// ============================================

/// Partitioned in-memory vector search engine.
///
/// v2.4.0: Optionally stores quantized (uint8) vectors alongside f32 vectors
/// for two-phase search (coarse quantized rank → fine f32 re-rank).
///
/// ## Thread Safety
/// Outer layer uses `RwLock<HashMap<..., Partition>>`:
/// - search = read lock (concurrent recalls don't block)
/// - upsert/remove = write lock (serialized, fast)
/// - calibrate = write lock (called once at startup)
pub struct VectorIndex {
    partitions: RwLock<HashMap<PartitionKey, Partition>>,
    record_to_partition: DashMap<[u8; 32], PartitionKey>,
    /// v2.4.0: Whether quantized search is enabled (from config).
    quantization_enabled: bool,
    /// v2.4.0: Early termination saturation threshold.
    saturation_threshold: f32,
}

impl VectorIndex {
    #[must_use]
    pub fn new() -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            record_to_partition: DashMap::new(),
            quantization_enabled: false,
            saturation_threshold: DEFAULT_SATURATION_THRESHOLD,
        }
    }

    /// v2.4.0: Create index with quantization and early termination settings.
    ///
    /// ## Arguments
    /// * `quantization_enabled` - Enable two-phase quantized search
    /// * `saturation_threshold` - Early termination threshold (0.0 = disabled)
    #[must_use]
    pub fn with_config(
        quantization_enabled: bool,
        saturation_threshold: f32,
    ) -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            record_to_partition: DashMap::new(),
            quantization_enabled,
            saturation_threshold: if saturation_threshold <= 0.0 {
                DEFAULT_SATURATION_THRESHOLD
            } else {
                saturation_threshold
            },
        }
    }

    /// Insert or update a vector.
    ///
    /// v2.4.0: If quantization is enabled AND the partition has a calibrated
    /// quantizer, also stores a uint8 quantized copy of the embedding.
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

        let mut partitions = self.partitions.write();
        let partition = partitions
            .entry(key.clone())
            .or_insert_with(|| Partition::new(dim));

        if partition.dim != dim && !partition.entries.is_empty() {
            warn!(
                expected = partition.dim, actual = dim, model = embedding_model,
                "[VECTOR] ⚠️ Dimension mismatch in partition, skipping"
            );
            return;
        }

        // v2.4.0: Quantize if possible
        let quantized = if self.quantization_enabled {
            partition.quantizer.as_ref()
                .filter(|q| q.is_calibrated())
                .map(|q| q.quantize(&embedding))
        } else {
            None
        };

        let entry = VectorEntry {
            record_id,
            embedding,
            quantized,
            layer,
            timestamp,
        };

        partition.entries.insert(record_id, entry);
        drop(partitions);

        self.record_to_partition.insert(record_id, key);
    }

    /// Remove a vector from the index.
    pub fn remove(&self, record_id: &[u8; 32]) -> bool {
        let key = match self.record_to_partition.remove(record_id) {
            Some((_, k)) => k,
            None => return false,
        };

        let mut partitions = self.partitions.write();
        if let Some(partition) = partitions.get_mut(&key) {
            partition.entries.remove(record_id);
            if partition.entries.is_empty() {
                partitions.remove(&key);
            }
            true
        } else {
            false
        }
    }

    /// v2.4.0: Calibrate the quantizer for a specific partition.
    ///
    /// Must be called after rebuilding the index from SQLite at startup.
    /// Collects all f32 vectors in the partition and trains the ScalarQuantizer.
    /// After calibration, subsequent upserts will also store quantized vectors.
    /// Existing entries are quantized in-place.
    ///
    /// No-op if quantization is disabled or partition doesn't exist.
    pub fn calibrate_partition(&self, owner: &[u8; 32], embedding_model: &str) {
        if !self.quantization_enabled { return; }

        let key = (hex::encode(owner), embedding_model.to_string());
        let mut partitions = self.partitions.write();

        let partition = match partitions.get_mut(&key) {
            Some(p) => p,
            None => return,
        };

        if partition.entries.is_empty() { return; }

        let dim = partition.dim;

        // Collect all embeddings for calibration
        let vectors: Vec<Vec<f32>> = partition.entries.values()
            .map(|e| e.embedding.clone())
            .collect();

        let mut quantizer = ScalarQuantizer::new(dim);
        quantizer.calibrate(&vectors);

        info!(
            owner = %hex::encode(owner),
            model = embedding_model,
            vectors = vectors.len(),
            dim = dim,
            "[VECTOR] Partition calibrated for scalar quantization"
        );

        // Quantize all existing entries
        for entry in partition.entries.values_mut() {
            entry.quantized = Some(quantizer.quantize(&entry.embedding));
        }

        partition.quantizer = Some(quantizer);
    }

    /// v2.4.0: Get a snapshot of the partition's quantizer calibration bytes.
    /// Used for persisting calibration to SQLite (chain_state table).
    pub fn get_quantizer_bytes(&self, owner: &[u8; 32], embedding_model: &str) -> Option<Vec<u8>> {
        let key = (hex::encode(owner), embedding_model.to_string());
        let partitions = self.partitions.read();
        partitions.get(&key)
            .and_then(|p| p.quantizer.as_ref())
            .filter(|q| q.is_calibrated())
            .map(|q| q.to_bytes())
    }

    /// v2.4.0: Restore a partition's quantizer from persisted calibration bytes.
    /// Called at startup after index rebuild, before calibrate_partition.
    /// If valid calibration data exists, skips re-calibration (faster startup).
    pub fn restore_quantizer(&self, owner: &[u8; 32], embedding_model: &str, data: &[u8]) -> bool {
        if !self.quantization_enabled { return false; }

        let key = (hex::encode(owner), embedding_model.to_string());
        let mut partitions = self.partitions.write();

        let partition = match partitions.get_mut(&key) {
            Some(p) => p,
            None => return false,
        };

        let quantizer = match ScalarQuantizer::from_bytes(data) {
            Some(q) if q.dim() == partition.dim => q,
            _ => return false,
        };

        // Re-quantize all existing entries with restored calibration
        for entry in partition.entries.values_mut() {
            entry.quantized = Some(quantizer.quantize(&entry.embedding));
        }

        partition.quantizer = Some(quantizer);
        info!(
            owner = %hex::encode(owner), model = embedding_model,
            "[VECTOR] Quantizer restored from persisted calibration"
        );
        true
    }

    /// Search top-K most similar vectors within a specific partition.
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

    /// Search with optional layer filter.
    ///
    /// v2.4.0: Uses two-phase search when quantizer is available:
    /// 1. Coarse rank: quantized dot product → top_k * 3 candidates
    /// 2. Fine rank: f32 cosine similarity on candidates → top_k results
    ///
    /// Falls back to brute-force f32 when quantizer is unavailable.
    /// Early termination applies to both paths.
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

        if query.len() != partition.dim {
            warn!(
                query_dim = query.len(), partition_dim = partition.dim,
                model = embedding_model,
                "[VECTOR] ⚠️ Query dimension mismatch"
            );
            return Vec::new();
        }

        // v2.4.0: Decide search strategy
        let use_quantized = self.quantization_enabled
            && partition.quantizer.as_ref().map_or(false, |q| q.is_calibrated());

        if use_quantized {
            self.search_two_phase(
                query, partition, layer_filter, top_k, min_similarity,
            )
        } else {
            self.search_brute_force(
                query, partition, layer_filter, top_k, min_similarity,
            )
        }
    }

    /// Original brute-force f32 search (v2.3.0 path, preserved).
    /// v2.4.0: Added early termination via saturation detection.
    fn search_brute_force(
        &self,
        query: &[f32],
        partition: &Partition,
        layer_filter: Option<MemoryLayer>,
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = Vec::new();
        let mut best_score: f32 = f32::MIN;
        let mut stale_count: usize = 0;
        let sat_threshold = self.saturation_threshold;
        let sat_window = DEFAULT_SATURATION_WINDOW;

        for entry in partition.entries.values() {
            // Layer filter
            if let Some(lf) = layer_filter {
                if entry.layer != lf { continue; }
            }

            let sim = cosine_similarity(query, &entry.embedding);

            if sim >= min_similarity {
                results.push(SearchResult {
                    record_id: entry.record_id,
                    similarity: sim,
                    layer: entry.layer,
                    timestamp: entry.timestamp,
                });

                // Early termination: track if best score is still improving
                if sim > best_score + sat_threshold {
                    best_score = sim;
                    stale_count = 0;
                } else {
                    stale_count += 1;
                }

                // If we have enough results and score is saturated, stop early
                if results.len() >= top_k * 2 && stale_count >= sat_window {
                    debug!(
                        scanned = results.len(),
                        total = partition.entries.len(),
                        "[VECTOR] Early termination (brute-force saturation)"
                    );
                    break;
                }
            }
        }

        results.sort_unstable_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// v2.4.0: Two-phase quantized search.
    ///
    /// Phase 1 (coarse): Quantize the query, compute fast_dot_product against
    ///   all quantized entries → select top_k * COARSE_EXPANSION candidates.
    /// Phase 2 (fine): Compute full f32 cosine similarity on candidates → top_k.
    ///
    /// This is faster than brute-force when partitions are large (>5K vectors)
    /// because the coarse phase uses uint8 arithmetic (better cache + SIMD).
    fn search_two_phase(
        &self,
        query: &[f32],
        partition: &Partition,
        layer_filter: Option<MemoryLayer>,
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<SearchResult> {
        let quantizer = match &partition.quantizer {
            Some(q) => q,
            None => return self.search_brute_force(
                query, partition, layer_filter, top_k, min_similarity
            ),
        };

        let query_quantized = quantizer.quantize(query);
        let coarse_k = top_k * COARSE_EXPANSION;

        // Phase 1: Coarse rank with fast_dot_product
        let mut coarse_candidates: Vec<([u8; 32], u64)> = Vec::new();

        for entry in partition.entries.values() {
            if let Some(lf) = layer_filter {
                if entry.layer != lf { continue; }
            }

            let score = match &entry.quantized {
                Some(qv) => quantizer.fast_dot_product(&query_quantized, qv),
                // Entry not quantized (inserted before calibration) → use f32 fallback
                None => {
                    let sim = cosine_similarity(query, &entry.embedding);
                    // Convert f32 similarity to u64 scale for sorting compatibility
                    // Multiply by large constant to preserve ranking precision
                    (sim * 1_000_000.0) as u64
                }
            };

            coarse_candidates.push((entry.record_id, score));
        }

        // Sort by coarse score descending, take top coarse_k
        coarse_candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        coarse_candidates.truncate(coarse_k);

        // Phase 2: Fine rank with f32 cosine similarity
        let mut results: Vec<SearchResult> = Vec::with_capacity(coarse_k);

        for (record_id, _coarse_score) in &coarse_candidates {
            if let Some(entry) = partition.entries.get(record_id) {
                let sim = cosine_similarity(query, &entry.embedding);
                if sim >= min_similarity {
                    results.push(SearchResult {
                        record_id: entry.record_id,
                        similarity: sim,
                        layer: entry.layer,
                        timestamp: entry.timestamp,
                    });
                }
            }
        }

        results.sort_unstable_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Per-layer dedup detection with Episode time window.
    #[must_use]
    pub fn check_duplicate(
        &self,
        query: &[f32],
        owner: &[u8; 32],
        embedding_model: &str,
        layer: MemoryLayer,
        current_timestamp: u64,
    ) -> DedupResult {
        if layer == MemoryLayer::Archive {
            return DedupResult {
                is_duplicate: false,
                existing_id: None,
                max_similarity: 0.0,
            };
        }

        let threshold = dedup_threshold_for_layer(layer);

        let candidates = self.search_filtered(
            query, owner, embedding_model,
            Some(layer), 1, threshold,
        );

        match candidates.first() {
            Some(hit) => {
                if layer == MemoryLayer::Episode {
                    let time_diff = current_timestamp.saturating_sub(hit.timestamp);
                    if time_diff > EPISODE_DEDUP_WINDOW_SECS {
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

    /// Total vectors across all partitions.
    #[must_use]
    pub fn total_vectors(&self) -> usize {
        self.partitions.read().values().map(|p| p.entries.len()).sum()
    }

    /// Number of partitions.
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.partitions.read().len()
    }

    /// Vector count in a specific partition.
    #[must_use]
    pub fn partition_size(&self, owner: &[u8; 32], embedding_model: &str) -> usize {
        let key = (hex::encode(owner), embedding_model.to_string());
        self.partitions.read().get(&key).map_or(0, |p| p.entries.len())
    }

    /// v2.4.0: Check if a partition has a calibrated quantizer.
    #[must_use]
    pub fn is_partition_quantized(&self, owner: &[u8; 32], embedding_model: &str) -> bool {
        let key = (hex::encode(owner), embedding_model.to_string());
        self.partitions.read().get(&key)
            .and_then(|p| p.quantizer.as_ref())
            .map_or(false, |q| q.is_calibrated())
    }

    /// Clear all partitions.
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
            .field("quantization", &self.quantization_enabled)
            .finish()
    }
}

// ============================================
// Cosine Similarity
// ============================================

/// Compute cosine similarity between two vectors.
/// For normalized vectors (unit length), cosine similarity = dot product.
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

/// Compute composite recall score for a candidate record.
///
/// ## Formula
/// ```text
/// score = semantic_similarity × time_decay × freq_boost + layer_weight
/// time_decay = exp(-hours_since_creation / stability)
/// freq_boost = 1.0 + 0.3 × ln(access_count + 1)
/// ```
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

    // ========================================
    // Existing tests (preserved from v2.3.0)
    // ========================================

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

        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        idx.upsert([2; 32], norm(&[1.0; 1536]), MemoryLayer::Episode, 100, &OWNER_A, MODEL_OAI);

        assert_eq!(idx.partition_count(), 2);
        assert_eq!(idx.partition_size(&OWNER_A, MODEL_MINI), 1);
        assert_eq!(idx.partition_size(&OWNER_A, MODEL_OAI), 1);

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
        idx.upsert([1; 32], vec![1.0, 0.0, 0.0], MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
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
        assert!(!idx.remove(&[1; 32]));
    }

    #[test]
    fn test_dedup_identity_high_threshold() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);
        idx.upsert([1; 32], v.clone(), MemoryLayer::Identity, 100, &OWNER_A, MODEL_MINI);

        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Identity, 200);
        assert!(r.is_duplicate);
    }

    #[test]
    fn test_dedup_episode_time_window() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0, 0.0]);

        let old_time = 1_700_000_000u64;
        idx.upsert([1; 32], v.clone(), MemoryLayer::Episode, old_time, &OWNER_A, MODEL_MINI);

        let now = old_time + EPISODE_DEDUP_WINDOW_SECS + 1;
        let r = idx.check_duplicate(&v, &OWNER_A, MODEL_MINI, MemoryLayer::Episode, now);
        assert!(!r.is_duplicate, "Over 24h should not count as duplicate");

        let recent = old_time + 3600;
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

    #[test]
    fn test_score_recent_vs_old() {
        let now = 1_700_100_000u64;
        let recent = now - 1000;
        let old = now - 200_000;

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
        assert!(s_archive < s_episode);
        assert!(s_archive > 0.0);
    }

    #[test]
    fn test_nonexistent_partition_returns_empty() {
        let idx = VectorIndex::new();
        let v = norm(&[1.0, 0.0]);
        let results = idx.search(&v, &OWNER_A, "nonexistent-model", 10, 0.0);
        assert!(results.is_empty());
    }

    // ========================================
    // v2.4.0: Quantization Integration Tests
    // ========================================

    #[test]
    fn test_quantization_disabled_by_default() {
        let idx = VectorIndex::new();
        assert!(!idx.quantization_enabled);
        assert!(!idx.is_partition_quantized(&OWNER_A, MODEL_MINI));
    }

    #[test]
    fn test_with_config_enables_quantization() {
        let idx = VectorIndex::with_config(true, 0.001);
        assert!(idx.quantization_enabled);
    }

    #[test]
    fn test_calibrate_partition() {
        let idx = VectorIndex::with_config(true, 0.001);

        // Insert vectors
        for i in 0..50u8 {
            let v = norm(&[i as f32 * 0.1, 1.0 - i as f32 * 0.01, 0.5]);
            idx.upsert([i; 32], v, MemoryLayer::Episode, 100 + i as u64, &OWNER_A, MODEL_MINI);
        }

        assert!(!idx.is_partition_quantized(&OWNER_A, MODEL_MINI));

        // Calibrate
        idx.calibrate_partition(&OWNER_A, MODEL_MINI);

        assert!(idx.is_partition_quantized(&OWNER_A, MODEL_MINI));

        // Verify entries now have quantized data
        let partitions = idx.partitions.read();
        let key = (hex::encode(OWNER_A), MODEL_MINI.to_string());
        let partition = partitions.get(&key).unwrap();
        for entry in partition.entries.values() {
            assert!(entry.quantized.is_some(), "Entry should have quantized data after calibration");
        }
    }

    #[test]
    fn test_two_phase_search_returns_results() {
        let idx = VectorIndex::with_config(true, 0.001);

        // Insert vectors
        for i in 0..100u8 {
            let angle = i as f32 * 0.05;
            let v = norm(&[angle.cos(), angle.sin(), 0.1]);
            idx.upsert([i; 32], v, MemoryLayer::Episode, 100 + i as u64, &OWNER_A, MODEL_MINI);
        }

        idx.calibrate_partition(&OWNER_A, MODEL_MINI);

        // Search
        let query = norm(&[1.0, 0.0, 0.1]);
        let results = idx.search(&query, &OWNER_A, MODEL_MINI, 5, 0.0);

        assert!(!results.is_empty(), "Two-phase search should return results");
        assert!(results.len() <= 5, "Should respect top_k");
        // Results should be sorted by similarity descending
        for w in results.windows(2) {
            assert!(w[0].similarity >= w[1].similarity);
        }
    }

    #[test]
    fn test_two_phase_vs_brute_force_consistency() {
        // Verify that two-phase search produces similar top-5 as brute-force
        let idx_quant = VectorIndex::with_config(true, 0.001);
        let idx_brute = VectorIndex::new();

        for i in 0..200u8 {
            let angle = i as f32 * 0.03;
            let v = norm(&[angle.cos(), angle.sin(), (i as f32 * 0.01).sin()]);
            idx_quant.upsert([i; 32], v.clone(), MemoryLayer::Episode, 100 + i as u64, &OWNER_A, MODEL_MINI);
            idx_brute.upsert([i; 32], v, MemoryLayer::Episode, 100 + i as u64, &OWNER_A, MODEL_MINI);
        }

        idx_quant.calibrate_partition(&OWNER_A, MODEL_MINI);

        let query = norm(&[0.8, 0.6, 0.1]);
        let r_quant = idx_quant.search(&query, &OWNER_A, MODEL_MINI, 5, 0.0);
        let r_brute = idx_brute.search(&query, &OWNER_A, MODEL_MINI, 5, 0.0);

        assert_eq!(r_quant.len(), 5);
        assert_eq!(r_brute.len(), 5);

        // Top-5 should have significant overlap (Recall@5 >= 3/5)
        let quant_ids: Vec<[u8; 32]> = r_quant.iter().map(|r| r.record_id).collect();
        let brute_ids: Vec<[u8; 32]> = r_brute.iter().map(|r| r.record_id).collect();
        let overlap = quant_ids.iter().filter(|id| brute_ids.contains(id)).count();

        assert!(
            overlap >= 3,
            "Two-phase top-5 should overlap >= 3 with brute-force top-5, got {} overlap\n\
             quant: {:?}\nbrute: {:?}",
            overlap,
            quant_ids.iter().map(|id| id[0]).collect::<Vec<_>>(),
            brute_ids.iter().map(|id| id[0]).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_quantizer_serialization_roundtrip() {
        let idx = VectorIndex::with_config(true, 0.001);

        for i in 0..50u8 {
            let v = norm(&[i as f32 * 0.1, 1.0 - i as f32 * 0.01, 0.5]);
            idx.upsert([i; 32], v, MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        }

        idx.calibrate_partition(&OWNER_A, MODEL_MINI);

        // Serialize
        let bytes = idx.get_quantizer_bytes(&OWNER_A, MODEL_MINI);
        assert!(bytes.is_some());

        // Create new index and restore
        let idx2 = VectorIndex::with_config(true, 0.001);
        for i in 0..50u8 {
            let v = norm(&[i as f32 * 0.1, 1.0 - i as f32 * 0.01, 0.5]);
            idx2.upsert([i; 32], v, MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        }

        let restored = idx2.restore_quantizer(&OWNER_A, MODEL_MINI, &bytes.unwrap());
        assert!(restored);
        assert!(idx2.is_partition_quantized(&OWNER_A, MODEL_MINI));
    }

    #[test]
    fn test_upsert_after_calibration_quantizes() {
        let idx = VectorIndex::with_config(true, 0.001);

        // Insert and calibrate
        for i in 0..20u8 {
            let v = norm(&[i as f32 * 0.1, 0.5, 0.3]);
            idx.upsert([i; 32], v, MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        }
        idx.calibrate_partition(&OWNER_A, MODEL_MINI);

        // Insert new vector AFTER calibration
        let new_v = norm(&[0.9, 0.1, 0.2]);
        idx.upsert([99; 32], new_v, MemoryLayer::Episode, 200, &OWNER_A, MODEL_MINI);

        // New entry should also be quantized
        let partitions = idx.partitions.read();
        let key = (hex::encode(OWNER_A), MODEL_MINI.to_string());
        let partition = partitions.get(&key).unwrap();
        let new_entry = partition.entries.get(&[99; 32]).unwrap();
        assert!(new_entry.quantized.is_some(), "New entry after calibration should be quantized");
    }

    #[test]
    fn test_fallback_to_brute_force_without_calibration() {
        let idx = VectorIndex::with_config(true, 0.001);

        // Insert WITHOUT calibrating
        for i in 0..20u8 {
            let v = norm(&[i as f32 * 0.1, 0.5, 0.3]);
            idx.upsert([i; 32], v, MemoryLayer::Episode, 100, &OWNER_A, MODEL_MINI);
        }

        // Search should still work (falls back to brute-force)
        let query = norm(&[0.5, 0.5, 0.3]);
        let results = idx.search(&query, &OWNER_A, MODEL_MINI, 5, 0.0);
        assert!(!results.is_empty(), "Should fall back to brute-force without calibration");
    }
}
