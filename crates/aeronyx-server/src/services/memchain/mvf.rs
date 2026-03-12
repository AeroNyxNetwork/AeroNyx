// ============================================
// File: crates/aeronyx-server/src/services/memchain/mvf.rs
// ============================================
//! # Memory Value Function (MVF) — D9
//!
//! A learnable linear model that predicts the probability a memory will
//! be "adopted" (positively received) by the user in a given context.
//!
//! ## Core Formula
//! ```text
//! V(m, q, ctx) = σ(w · φ̂(m, q, ctx))
//!
//! σ(x) = 1 / (1 + e⁻ˣ)         sigmoid → (0, 1)
//! φ ∈ ℝ¹⁰                        10-dim feature vector (v2.4.0: was 9)
//! φ̂ = normalize(φ)               per-dimension running norm
//! w ∈ ℝ¹⁰                        per-user learned weights
//! ```
//!
//! ## 10-Dimensional Feature Vector φ (v2.4.0: +φ₉)
//! ```text
//! φ₀ = cos(embed(m), embed(q))                  semantic similarity [0,1]
//! φ₁ = exp(-Δt / S(layer))                      time decay (0,1]
//! φ₂ = min(1 + 0.3 × ln(access+1), 3.5)        freq boost [1.0,3.5]
//! φ₃ = L(layer)                                  layer score [0.17,1.0]
//! φ₄ = (n_pos - n_neg) / (n_pos + n_neg + 1)   feedback score (-1,1)
//! φ₅ = degree(m) / max_degree                   graph centrality [0,1]
//! φ₆ = 𝟙(timestamp ∈ time_hint)                 time match {0,1}
//! φ₇ = cos(embed(m), centroid_session)           topic coherence [-1,1]
//! φ₈ = -𝟙(has_conflict)                         conflict penalty {-1,0}
//! φ₉ = graph_traverse_weight                    knowledge graph distance [0,1]  ← v2.4.0 NEW
//! ```
//!
//! ## φ₉ graph_traverse_weight (v2.4.0-GraphCognition)
//! When recall uses hybrid retrieval (graph BFS + vector search), records
//! discovered via graph traversal receive a φ₉ score based on traversal distance:
//! - Direct entity match (hop 0): φ₉ = 1.0
//! - 1 hop away: φ₉ = 0.7 (default decay factor per hop)
//! - 2 hops away: φ₉ = 0.49 (0.7²)
//! - Not discovered via graph: φ₉ = 0.0
//!
//! For SEMANTIC query type (no entity match), φ₉ = 0.0 and the weight is
//! dynamically set to 0.0, making MVF fully backward compatible with v2.3.0.
//!
//! ## SGD Online Learning
//! ```text
//! Loss: Binary Cross-Entropy
//! Update: w ← w - 0.01 × (ŷ - y) × φ̂
//! Regularization: w ← w × (1 - 0.0001)
//! Hard constraints: w₀ ∈ [0.05, 1.0], w₈ ∈ [-1.0, 0.0], w₉ ∈ [0.0, 1.0], rest ∈ [-1.0, 1.0]
//! Collapse guard: all Identity V < 0.01 → reset to defaults
//! ```
//!
//! ## Backward Compatibility (v2.4.0)
//! - WeightVector serialization: 120 bytes (was 108). from_bytes() accepts both
//!   108 bytes (v2.3.0 format, φ₉ defaults added) and 120 bytes (v2.4.0 format).
//! - compute_features() now takes graph_traverse_weight parameter. Callers that
//!   don't use graph traversal pass 0.0.
//! - default_weights() includes φ₉ with weight 0.0 (no effect until graph is populated).
//!
//! ## Last Modified
//! v2.1.0 - New file: MVF 9-dim features + SGD + normalization + collapse guard
//! v2.4.0-GraphCognition - 🌟 Extended to 10 dimensions (+φ₉ graph_traverse_weight).
//!   Backward-compatible serialization. Hard constraint w₉ ∈ [0.0, 1.0].

// ============================================
// Constants
// ============================================

/// Feature vector dimension.
/// v2.4.0: Extended from 9 to 10 (added φ₉ graph_traverse_weight).
pub const MVF_DIM: usize = 10;

/// v2.3.0 feature vector dimension (for backward-compatible deserialization).
const MVF_DIM_V23: usize = 9;

/// Learning rate for SGD.
const LEARNING_RATE: f32 = 0.01;

/// L2 regularization strength.
const WEIGHT_DECAY: f32 = 0.0001;

/// Gradient clipping range for (ŷ - y).
const GRAD_CLIP: f32 = 0.99;

/// EMA smoothing factor for running mean/std.
const EMA_BETA: f32 = 0.99;

/// Layer score mapping: Identity=1.0, Knowledge=0.67, Episode=0.33, Archive=0.17
const LAYER_SCORES: [f32; 4] = [1.0, 0.67, 0.33, 0.17];

/// Stability hours per layer (same as MemoryLayer::stability_hours but as f32).
const STABILITY_HOURS: [f32; 4] = [8760.0, 2160.0, 168.0, 720.0];

/// Serialized size for v2.3.0 weights (9 dims × 3 arrays × 4 bytes).
const WEIGHT_BYTES_V23: usize = MVF_DIM_V23 * 3 * 4; // 108

/// Serialized size for v2.4.0 weights (10 dims × 3 arrays × 4 bytes).
const WEIGHT_BYTES_V24: usize = MVF_DIM * 3 * 4; // 120

// ============================================
// FeatureVector
// ============================================

/// 10-dimensional feature vector φ (v2.4.0: was 9).
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector(pub [f32; MVF_DIM]);

impl FeatureVector {
    /// Create a zero feature vector.
    pub fn zeros() -> Self {
        Self([0.0; MVF_DIM])
    }
}

// ============================================
// WeightVector
// ============================================

/// Per-user learned weight vector with running normalization stats.
///
/// v2.4.0: 10 dims × 3 arrays × 4 bytes = 120 bytes per user (was 108).
#[derive(Debug, Clone)]
pub struct WeightVector {
    /// Learned weights w ∈ ℝ¹⁰.
    pub weights: [f32; MVF_DIM],
    /// Exponential moving average of feature means μ.
    pub running_mean: [f32; MVF_DIM],
    /// Exponential moving average of feature stds σ.
    pub running_std: [f32; MVF_DIM],
    /// SGD update count (for persistence versioning).
    pub version: u64,
}

impl WeightVector {
    /// Serialize to 120 bytes: w(40) + μ(40) + σ(40).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(WEIGHT_BYTES_V24);
        for &v in &self.weights { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.running_mean { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.running_std { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    /// Deserialize from bytes.
    ///
    /// ## Backward Compatibility
    /// - 120 bytes: v2.4.0 format (10 dims) — direct deserialize
    /// - 108 bytes: v2.3.0 format (9 dims) — deserialize 9 dims, append φ₉ defaults
    /// - Other sizes: return None
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() == WEIGHT_BYTES_V24 {
            // v2.4.0 format: 10 dimensions
            return Self::from_bytes_ndim(data, MVF_DIM);
        }

        if data.len() == WEIGHT_BYTES_V23 || data.len() >= WEIGHT_BYTES_V23 {
            // v2.3.0 format: 9 dimensions — read 9, pad φ₉ with defaults
            let mut weights = [0.0f32; MVF_DIM];
            let mut running_mean = [0.0f32; MVF_DIM];
            let mut running_std = [0.0f32; MVF_DIM];

            let read_dim = MVF_DIM_V23;
            for i in 0..read_dim {
                let off = i * 4;
                weights[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
            }
            for i in 0..read_dim {
                let off = read_dim * 4 + i * 4;
                running_mean[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
            }
            for i in 0..read_dim {
                let off = read_dim * 2 * 4 + i * 4;
                running_std[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
            }

            // φ₉ defaults (graph_traverse_weight)
            weights[9] = 0.0;        // no graph influence initially
            running_mean[9] = 0.0;
            running_std[9] = 0.3;

            return Some(Self { weights, running_mean, running_std, version: 0 });
        }

        None
    }

    /// Internal: deserialize exactly `ndim` dimensions from data.
    fn from_bytes_ndim(data: &[u8], ndim: usize) -> Option<Self> {
        let expected = ndim * 3 * 4;
        if data.len() < expected { return None; }

        let mut weights = [0.0f32; MVF_DIM];
        let mut running_mean = [0.0f32; MVF_DIM];
        let mut running_std = [0.0f32; MVF_DIM];

        for i in 0..ndim {
            let off = i * 4;
            weights[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }
        for i in 0..ndim {
            let off = ndim * 4 + i * 4;
            running_mean[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }
        for i in 0..ndim {
            let off = ndim * 2 * 4 + i * 4;
            running_std[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }

        Some(Self { weights, running_mean, running_std, version: 0 })
    }
}

/// Create the default (initial) weight vector.
///
/// v2.4.0: Added φ₉ = 0.0 (graph_traverse_weight, no influence until graph is populated).
pub fn default_weights() -> WeightVector {
    WeightVector {
        //                φ₀    φ₁    φ₂    φ₃    φ₄    φ₅    φ₆    φ₇    φ₈    φ₉
        weights:      [0.35, 0.15, 0.10, 0.20, 0.10, 0.02, 0.15, 0.05, -0.20, 0.0],
        running_mean: [0.5,  0.5,  1.5,  0.5,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0],
        running_std:  [0.3,  0.3,  0.5,  0.3,  0.3,  0.3,  0.5,  0.3,   0.5,  0.3],
        version: 0,
    }
}

// ============================================
// Feature Computation
// ============================================

/// Compute the 10-dimensional feature vector for a memory-query pair.
///
/// ## v2.4.0 Changes
/// - Added `graph_traverse_weight` parameter for φ₉
/// - Returns 10-dim vector (was 9)
///
/// ## Arguments (new in v2.4.0)
/// * `graph_traverse_weight` - Graph traversal distance score [0,1].
///   0.0 = not discovered via graph (SEMANTIC query type),
///   1.0 = direct entity match (hop 0),
///   0.7^n for n hops away.
///   Callers not using graph traversal should pass 0.0.
pub fn compute_features(
    similarity: f32,
    layer: u8,
    timestamp: u64,
    now: u64,
    access_count: u32,
    positive_feedback: u32,
    negative_feedback: u32,
    has_conflict: bool,
    time_hint: Option<(i64, i64)>,
    session_centroid_sim: f32,
    graph_degree: u32,
    max_graph_degree: u32,
    graph_traverse_weight: f32,
) -> FeatureVector {
    let layer_idx = (layer as usize).min(3);

    // φ₀: semantic similarity [0, 1]
    let phi_0 = similarity.clamp(0.0, 1.0);

    // φ₁: time decay = exp(-Δt_hours / stability)
    let hours = if now > timestamp { (now - timestamp) as f32 / 3600.0 } else { 0.0 };
    let stability = STABILITY_HOURS[layer_idx];
    let phi_1 = (-hours / stability).exp();

    // φ₂: freq boost = min(1 + 0.3 × ln(access+1), 3.5)
    let phi_2 = (1.0 + 0.3 * ((access_count as f32) + 1.0).ln()).min(3.5);

    // φ₃: layer score
    let phi_3 = LAYER_SCORES[layer_idx];

    // φ₄: feedback score = (pos - neg) / (pos + neg + 1)
    let n_pos = positive_feedback as f32;
    let n_neg = negative_feedback as f32;
    let phi_4 = (n_pos - n_neg) / (n_pos + n_neg + 1.0);

    // φ₅: graph centrality [0, 1]
    let phi_5 = if max_graph_degree > 0 {
        (graph_degree as f32) / (max_graph_degree as f32)
    } else { 0.0 };

    // φ₆: time match {0, 1}
    let phi_6 = match time_hint {
        Some((start, end)) => {
            let ts = timestamp as i64;
            if ts >= start && ts <= end { 1.0 } else { 0.0 }
        }
        None => 0.0,
    };

    // φ₇: session topic coherence [-1, 1]
    let phi_7 = session_centroid_sim.clamp(-1.0, 1.0);

    // φ₈: conflict penalty {-1, 0}
    let phi_8 = if has_conflict { -1.0 } else { 0.0 };

    // φ₉: graph traverse weight [0, 1] (v2.4.0)
    let phi_9 = graph_traverse_weight.clamp(0.0, 1.0);

    FeatureVector([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9])
}

// ============================================
// Normalization
// ============================================

/// Normalize features using per-dimension running mean/std.
/// Also updates the running statistics (EMA with β=0.99).
pub fn normalize(phi: &FeatureVector, w: &mut WeightVector) -> FeatureVector {
    let mut normalized = [0.0f32; MVF_DIM];

    for i in 0..MVF_DIM {
        w.running_mean[i] = EMA_BETA * w.running_mean[i] + (1.0 - EMA_BETA) * phi.0[i];
        w.running_std[i] = EMA_BETA * w.running_std[i]
            + (1.0 - EMA_BETA) * (phi.0[i] - w.running_mean[i]).abs();
        normalized[i] = (phi.0[i] - w.running_mean[i]) / (w.running_std[i] + 1e-6);
    }

    FeatureVector(normalized)
}

// ============================================
// Value Computation
// ============================================

/// Compute V(m, q, ctx) = σ(w · φ̂).
pub fn compute_value(w: &WeightVector, phi_norm: &FeatureVector) -> f32 {
    let dot: f32 = w.weights.iter()
        .zip(phi_norm.0.iter())
        .map(|(wi, pi)| wi * pi)
        .sum();
    sigmoid(dot)
}

/// Sigmoid function: σ(x) = 1 / (1 + e⁻ˣ)
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================
// SGD Online Learning
// ============================================

/// Perform one SGD update step.
///
/// ## v2.4.0 Hard Constraints
/// - w₀ (similarity) ∈ [0.05, 1.0] (must be positive)
/// - w₈ (conflict) ∈ [-1.0, 0.0] (must be non-positive)
/// - w₉ (graph_traverse) ∈ [0.0, 1.0] (must be non-negative)
/// - all others ∈ [-1.0, 1.0]
pub fn sgd_update(w: &mut WeightVector, phi_norm: &FeatureVector, y: f32) {
    let y_hat = compute_value(w, phi_norm);
    let error = (y_hat - y).clamp(-GRAD_CLIP, GRAD_CLIP);

    for i in 0..MVF_DIM {
        w.weights[i] -= LEARNING_RATE * error * phi_norm.0[i];
        w.weights[i] *= 1.0 - WEIGHT_DECAY;
    }

    // Hard constraints per dimension
    w.weights[0] = w.weights[0].clamp(0.05, 1.0);  // φ₀ similarity: positive
    w.weights[8] = w.weights[8].clamp(-1.0, 0.0);   // φ₈ conflict: non-positive
    w.weights[9] = w.weights[9].clamp(0.0, 1.0);    // φ₉ graph_traverse: non-negative
    for i in 1..8 {
        w.weights[i] = w.weights[i].clamp(-1.0, 1.0);
    }

    w.version += 1;
}

// ============================================
// Collapse Detection
// ============================================

/// Check if MVF has collapsed: all Identity memories scoring < 0.01.
pub fn check_collapse(identity_values: &[f32]) -> bool {
    if identity_values.is_empty() { return false; }
    identity_values.iter().all(|&v| v < 0.01)
}

// ============================================
// Fusion
// ============================================

/// Compute the fused score: α × V_mvf + (1 - α) × V_old.
pub fn fuse_scores(v_mvf: f32, v_old: f64, alpha: f32) -> f64 {
    let alpha = alpha.clamp(0.0, 0.7) as f64;
    alpha * (v_mvf as f64) + (1.0 - alpha) * v_old
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let w = default_weights();
        assert_eq!(w.weights.len(), 10); // v2.4.0: 10 dims
        assert!((w.weights[0] - 0.35).abs() < 1e-6);
        assert!((w.weights[8] - (-0.20)).abs() < 1e-6);
        assert!((w.weights[9] - 0.0).abs() < 1e-6); // φ₉ default
    }

    #[test]
    fn test_weight_serialization_roundtrip_v24() {
        let w = default_weights();
        let bytes = w.to_bytes();
        assert_eq!(bytes.len(), 120); // v2.4.0: 10 × 3 × 4 = 120
        let restored = WeightVector::from_bytes(&bytes).unwrap();
        for i in 0..MVF_DIM {
            assert!((w.weights[i] - restored.weights[i]).abs() < 1e-6);
            assert!((w.running_mean[i] - restored.running_mean[i]).abs() < 1e-6);
            assert!((w.running_std[i] - restored.running_std[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_deserialization_backward_compat_v23() {
        // Simulate v2.3.0 format: 9 dims × 3 arrays × 4 bytes = 108 bytes
        let old_w = WeightVector {
            weights:      [0.35, 0.15, 0.10, 0.20, 0.10, 0.02, 0.15, 0.05, -0.20, 0.0],
            running_mean: [0.5,  0.5,  1.5,  0.5,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0],
            running_std:  [0.3,  0.3,  0.5,  0.3,  0.3,  0.3,  0.5,  0.3,   0.5,  0.3],
            version: 0,
        };

        // Serialize only first 9 dims (simulate v2.3.0)
        let mut v23_bytes = Vec::with_capacity(108);
        for i in 0..9 { v23_bytes.extend_from_slice(&old_w.weights[i].to_le_bytes()); }
        for i in 0..9 { v23_bytes.extend_from_slice(&old_w.running_mean[i].to_le_bytes()); }
        for i in 0..9 { v23_bytes.extend_from_slice(&old_w.running_std[i].to_le_bytes()); }
        assert_eq!(v23_bytes.len(), 108);

        // Deserialize with v2.4.0 from_bytes — should pad φ₉
        let restored = WeightVector::from_bytes(&v23_bytes).unwrap();
        assert_eq!(restored.weights.len(), 10);

        // First 9 dims should match
        for i in 0..9 {
            assert!((old_w.weights[i] - restored.weights[i]).abs() < 1e-6,
                "weights[{}] mismatch", i);
        }

        // φ₉ should have defaults
        assert!((restored.weights[9] - 0.0).abs() < 1e-6);
        assert!((restored.running_mean[9] - 0.0).abs() < 1e-6);
        assert!((restored.running_std[9] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_compute_features_basic() {
        let phi = compute_features(
            0.8, 2, 1_700_000_000, 1_700_010_000,
            5, 3, 1, false, None, 0.5, 4, 10, 0.0,
        );

        assert!((phi.0[0] - 0.8).abs() < 1e-6);      // similarity
        assert!(phi.0[1] > 0.0 && phi.0[1] < 1.0);    // time decay
        assert!(phi.0[2] > 1.0);                        // freq boost
        assert!((phi.0[3] - 0.33).abs() < 1e-2);       // Episode layer score
        assert!((phi.0[4] - 0.4).abs() < 1e-2);        // (3-1)/(3+1+1) = 0.4
        assert!((phi.0[5] - 0.4).abs() < 1e-2);        // 4/10
        assert!((phi.0[6] - 0.0).abs() < 1e-6);        // no time_hint
        assert!((phi.0[7] - 0.5).abs() < 1e-6);        // session centroid
        assert!((phi.0[8] - 0.0).abs() < 1e-6);        // no conflict
        assert!((phi.0[9] - 0.0).abs() < 1e-6);        // no graph traverse
    }

    #[test]
    fn test_compute_features_with_graph_traverse() {
        let phi = compute_features(
            0.8, 2, 1_700_000_000, 1_700_010_000,
            5, 3, 1, false, None, 0.5, 4, 10,
            0.7, // 1 hop away
        );
        assert!((phi.0[9] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_penalty() {
        let phi = compute_features(0.8, 0, 0, 0, 0, 0, 0, true, None, 0.0, 0, 0, 0.0);
        assert!((phi.0[8] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_time_hint_match() {
        let phi = compute_features(0.5, 2, 1000, 2000, 0, 0, 0, false, Some((500, 1500)), 0.0, 0, 0, 0.0);
        assert!((phi.0[6] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_hint_no_match() {
        let phi = compute_features(0.5, 2, 2000, 3000, 0, 0, 0, false, Some((500, 1500)), 0.0, 0, 0, 0.0);
        assert!((phi.0[6] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_compute_value() {
        let w = default_weights();
        let phi = FeatureVector([0.5; MVF_DIM]);
        let v = compute_value(&w, &phi);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_sgd_negative_feedback_decreases_value() {
        let mut w = default_weights();
        let phi_norm = FeatureVector([0.5; MVF_DIM]);
        let v_before = compute_value(&w, &phi_norm);
        sgd_update(&mut w, &phi_norm, 0.0);
        let v_after = compute_value(&w, &phi_norm);
        assert!(v_after < v_before, "Negative feedback should decrease V");
    }

    #[test]
    fn test_sgd_positive_feedback_increases_value() {
        let mut w = default_weights();
        let phi_norm = FeatureVector([0.5; MVF_DIM]);
        let v_before = compute_value(&w, &phi_norm);
        sgd_update(&mut w, &phi_norm, 1.0);
        let v_after = compute_value(&w, &phi_norm);
        assert!(v_after > v_before, "Positive feedback should increase V");
    }

    #[test]
    fn test_hard_constraints_v24() {
        let mut w = default_weights();
        w.weights[0] = -5.0;
        w.weights[8] = 5.0;
        w.weights[9] = -5.0; // Try to make graph_traverse negative

        let phi = FeatureVector([1.0; MVF_DIM]);
        sgd_update(&mut w, &phi, 0.5);

        assert!(w.weights[0] >= 0.05, "w₀ must stay >= 0.05");
        assert!(w.weights[8] <= 0.0, "w₈ must stay <= 0.0");
        assert!(w.weights[9] >= 0.0, "w₉ must stay >= 0.0"); // v2.4.0
    }

    #[test]
    fn test_collapse_detection() {
        assert!(check_collapse(&[0.001, 0.005, 0.009]));
        assert!(!check_collapse(&[0.001, 0.5, 0.009]));
        assert!(!check_collapse(&[]));
    }

    #[test]
    fn test_fuse_scores() {
        let v_mvf = 0.8;
        let v_old = 1.2;
        let fused = fuse_scores(v_mvf, v_old, 0.5);
        assert!((fused - 1.0).abs() < 1e-6);

        let fused = fuse_scores(v_mvf, v_old, 0.0);
        assert!((fused - 1.2).abs() < 1e-6);

        let fused = fuse_scores(v_mvf, v_old, 1.0);
        let expected = 0.7 * 0.8 + 0.3 * 1.2;
        assert!((fused - expected).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_updates_stats() {
        let mut w = default_weights();
        let phi = FeatureVector([1.0; MVF_DIM]);
        let old_mean = w.running_mean[0];
        let _ = normalize(&phi, &mut w);
        assert!(w.running_mean[0] > old_mean);
    }

    #[test]
    fn test_version_increments() {
        let mut w = default_weights();
        assert_eq!(w.version, 0);
        sgd_update(&mut w, &FeatureVector([0.5; MVF_DIM]), 1.0);
        assert_eq!(w.version, 1);
        sgd_update(&mut w, &FeatureVector([0.5; MVF_DIM]), 0.0);
        assert_eq!(w.version, 2);
    }

    #[test]
    fn test_graph_traverse_weight_clamping() {
        // Values > 1.0 should be clamped
        let phi = compute_features(0.5, 0, 0, 0, 0, 0, 0, false, None, 0.0, 0, 0, 1.5);
        assert!((phi.0[9] - 1.0).abs() < 1e-6);

        // Values < 0.0 should be clamped
        let phi = compute_features(0.5, 0, 0, 0, 0, 0, 0, false, None, 0.0, 0, 0, -0.5);
        assert!((phi.0[9] - 0.0).abs() < 1e-6);
    }
}
