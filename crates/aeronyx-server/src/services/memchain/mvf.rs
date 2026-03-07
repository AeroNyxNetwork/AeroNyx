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
//! φ ∈ ℝ⁹                         9-dim feature vector
//! φ̂ = normalize(φ)               per-dimension running norm
//! w ∈ ℝ⁹                         per-user learned weights
//! ```
//!
//! ## 9-Dimensional Feature Vector φ
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
//! ```
//!
//! ## SGD Online Learning
//! ```text
//! Loss: Binary Cross-Entropy
//! Update: w ← w - 0.01 × (ŷ - y) × φ̂
//! Regularization: w ← w × (1 - 0.0001)
//! Hard constraints: w₀ ∈ [0.05, 1.0], w₈ ∈ [-1.0, 0.0], rest ∈ [-1.0, 1.0]
//! Collapse guard: all Identity V < 0.01 → reset to defaults
//! ```
//!
//! ## Last Modified
//! v2.1.0 - New file: MVF 9-dim features + SGD + normalization + collapse guard

// ============================================
// Constants
// ============================================

/// Feature vector dimension.
pub const MVF_DIM: usize = 9;

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

// ============================================
// FeatureVector
// ============================================

/// 9-dimensional feature vector φ.
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
/// Total storage: 9 × 3 × 4 bytes = 108 bytes per user.
#[derive(Debug, Clone)]
pub struct WeightVector {
    /// Learned weights w ∈ ℝ⁹.
    pub weights: [f32; MVF_DIM],
    /// Exponential moving average of feature means μ.
    pub running_mean: [f32; MVF_DIM],
    /// Exponential moving average of feature stds σ.
    pub running_std: [f32; MVF_DIM],
    /// SGD update count (for persistence versioning).
    pub version: u64,
}

impl WeightVector {
    /// Serialize to 108 bytes: w(36) + μ(36) + σ(36).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(108);
        for &v in &self.weights { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.running_mean { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.running_std { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    /// Deserialize from 108 bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 108 { return None; }

        let mut weights = [0.0f32; MVF_DIM];
        let mut running_mean = [0.0f32; MVF_DIM];
        let mut running_std = [0.0f32; MVF_DIM];

        for i in 0..MVF_DIM {
            let off = i * 4;
            weights[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }
        for i in 0..MVF_DIM {
            let off = 36 + i * 4;
            running_mean[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }
        for i in 0..MVF_DIM {
            let off = 72 + i * 4;
            running_std[i] = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        }

        Some(Self { weights, running_mean, running_std, version: 0 })
    }
}

/// Create the default (initial) weight vector per §1.5 and §1.6.
pub fn default_weights() -> WeightVector {
    WeightVector {
        //                φ₀    φ₁    φ₂    φ₃    φ₄    φ₅    φ₆    φ₇    φ₈
        weights:      [0.35, 0.15, 0.10, 0.20, 0.10, 0.02, 0.15, 0.05, -0.20],
        running_mean: [0.5,  0.5,  1.5,  0.5,  0.0,  0.0,  0.0,  0.0,   0.0],
        running_std:  [0.3,  0.3,  0.5,  0.3,  0.3,  0.3,  0.5,  0.3,   0.5],
        version: 0,
    }
}

// ============================================
// Feature Computation
// ============================================

/// Compute the 9-dimensional feature vector for a memory-query pair.
///
/// # Arguments
/// * `similarity` - Pre-computed cosine similarity φ₀ (from vector search)
/// * `layer` - Memory layer (0=Identity, 1=Knowledge, 2=Episode, 3=Archive)
/// * `timestamp` - Memory creation timestamp (Unix seconds)
/// * `now` - Current timestamp (Unix seconds)
/// * `access_count` - Times this memory has been recalled
/// * `positive_feedback` - Positive feedback count
/// * `negative_feedback` - Negative feedback count
/// * `has_conflict` - Whether this memory conflicts with another
/// * `time_hint` - Optional (start, end) time range from query
/// * `session_centroid_sim` - Cosine similarity to session centroid φ₇
/// * `graph_degree` - Number of co-occurrence edges for this memory
/// * `max_graph_degree` - Max degree across all memories for this owner
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
) -> FeatureVector {
    let layer_idx = (layer as usize).min(3);

    // φ₀: semantic similarity [0, 1]
    let phi_0 = similarity.clamp(0.0, 1.0);

    // φ₁: time decay = exp(-Δt_hours / stability)
    let hours = if now > timestamp {
        (now - timestamp) as f32 / 3600.0
    } else {
        0.0
    };
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
    } else {
        0.0
    };

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

    FeatureVector([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8])
}

// ============================================
// Normalization
// ============================================

/// Normalize features using per-dimension running mean/std.
///
/// Also updates the running statistics (EMA with β=0.99).
///
/// ```text
/// μᵢ ← β × μᵢ + (1-β) × φᵢ
/// σᵢ ← β × σᵢ + (1-β) × |φᵢ - μᵢ|
/// φ̂ᵢ = (φᵢ - μᵢ) / (σᵢ + 1e-6)
/// ```
pub fn normalize(phi: &FeatureVector, w: &mut WeightVector) -> FeatureVector {
    let mut normalized = [0.0f32; MVF_DIM];

    for i in 0..MVF_DIM {
        // Update running stats
        w.running_mean[i] = EMA_BETA * w.running_mean[i] + (1.0 - EMA_BETA) * phi.0[i];
        w.running_std[i] = EMA_BETA * w.running_std[i]
            + (1.0 - EMA_BETA) * (phi.0[i] - w.running_mean[i]).abs();

        // Normalize
        normalized[i] = (phi.0[i] - w.running_mean[i]) / (w.running_std[i] + 1e-6);
    }

    FeatureVector(normalized)
}

// ============================================
// Value Computation
// ============================================

/// Compute V(m, q, ctx) = σ(w · φ̂).
///
/// Returns a value in (0, 1) representing the predicted probability
/// that this memory will be adopted by the user.
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
/// ```text
/// ŷ = σ(w · φ̂)
/// gradient = (ŷ - y) × φ̂    (clipped to [-0.99, 0.99])
/// w ← w - lr × gradient
/// w ← w × (1 - weight_decay)  (L2 regularization)
/// Apply hard constraints on each dimension
/// ```
///
/// # Arguments
/// * `w` - Mutable weight vector (updated in-place)
/// * `phi_norm` - Normalized feature vector
/// * `y` - Ground truth label (1.0 = positive, 0.0 = negative)
pub fn sgd_update(w: &mut WeightVector, phi_norm: &FeatureVector, y: f32) {
    let y_hat = compute_value(w, phi_norm);
    let error = (y_hat - y).clamp(-GRAD_CLIP, GRAD_CLIP);

    for i in 0..MVF_DIM {
        // Gradient descent
        w.weights[i] -= LEARNING_RATE * error * phi_norm.0[i];

        // L2 regularization (weight decay)
        w.weights[i] *= 1.0 - WEIGHT_DECAY;
    }

    // Hard constraints per dimension
    //   w₀ (similarity) must be positive: [0.05, 1.0]
    //   w₈ (conflict) must be negative: [-1.0, 0.0]
    //   all others: [-1.0, 1.0]
    w.weights[0] = w.weights[0].clamp(0.05, 1.0);
    w.weights[8] = w.weights[8].clamp(-1.0, 0.0);
    for i in 1..8 {
        w.weights[i] = w.weights[i].clamp(-1.0, 1.0);
    }

    w.version += 1;
}

// ============================================
// Collapse Detection
// ============================================

/// Check if MVF has collapsed: all Identity memories scoring < 0.01.
///
/// This indicates the weights have drifted to a pathological state
/// where core identity is suppressed. Trigger: reset to defaults.
pub fn check_collapse(identity_values: &[f32]) -> bool {
    if identity_values.is_empty() {
        return false; // No Identity memories = not a collapse
    }
    identity_values.iter().all(|&v| v < 0.01)
}

// ============================================
// Fusion
// ============================================

/// Compute the fused score: α × V_mvf + (1 - α) × V_old.
///
/// α schedule (documented in architecture, controlled by config):
///   Week 1-2: α = 0.0  (pure old formula, collecting baseline)
///   Week 3+:  α = 0.5  (MVF half-weight)
///   Week 5+:  α = 0.7  (MVF dominant)
///   Always:   α ≤ 0.7  (30% safety net)
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
        assert_eq!(w.weights.len(), 9);
        assert!((w.weights[0] - 0.35).abs() < 1e-6);
        assert!((w.weights[8] - (-0.20)).abs() < 1e-6);
    }

    #[test]
    fn test_weight_serialization_roundtrip() {
        let w = default_weights();
        let bytes = w.to_bytes();
        assert_eq!(bytes.len(), 108);
        let restored = WeightVector::from_bytes(&bytes).unwrap();
        for i in 0..MVF_DIM {
            assert!((w.weights[i] - restored.weights[i]).abs() < 1e-6);
            assert!((w.running_mean[i] - restored.running_mean[i]).abs() < 1e-6);
            assert!((w.running_std[i] - restored.running_std[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_features_basic() {
        let phi = compute_features(
            0.8,  // similarity
            2,    // Episode
            1_700_000_000, // timestamp
            1_700_010_000, // now (~2.8h later)
            5,    // access_count
            3,    // positive
            1,    // negative
            false,
            None,
            0.5,  // session_centroid_sim
            4,    // graph_degree
            10,   // max_degree
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
    }

    #[test]
    fn test_conflict_penalty() {
        let phi = compute_features(0.8, 0, 0, 0, 0, 0, 0, true, None, 0.0, 0, 0);
        assert!((phi.0[8] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_time_hint_match() {
        let phi = compute_features(
            0.5, 2, 1000, 2000, 0, 0, 0, false,
            Some((500, 1500)), // timestamp 1000 is within range
            0.0, 0, 0,
        );
        assert!((phi.0[6] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_hint_no_match() {
        let phi = compute_features(
            0.5, 2, 2000, 3000, 0, 0, 0, false,
            Some((500, 1500)), // timestamp 2000 is outside range
            0.0, 0, 0,
        );
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
        sgd_update(&mut w, &phi_norm, 0.0); // negative feedback
        let v_after = compute_value(&w, &phi_norm);

        assert!(v_after < v_before, "Negative feedback should decrease V");
    }

    #[test]
    fn test_sgd_positive_feedback_increases_value() {
        let mut w = default_weights();
        let phi_norm = FeatureVector([0.5; MVF_DIM]);

        let v_before = compute_value(&w, &phi_norm);
        sgd_update(&mut w, &phi_norm, 1.0); // positive feedback
        let v_after = compute_value(&w, &phi_norm);

        assert!(v_after > v_before, "Positive feedback should increase V");
    }

    #[test]
    fn test_hard_constraints() {
        let mut w = default_weights();
        w.weights[0] = -5.0; // Try to make similarity weight negative
        w.weights[8] = 5.0;  // Try to make conflict weight positive

        let phi = FeatureVector([1.0; MVF_DIM]);
        sgd_update(&mut w, &phi, 0.5);

        assert!(w.weights[0] >= 0.05, "w₀ must stay >= 0.05");
        assert!(w.weights[8] <= 0.0, "w₈ must stay <= 0.0");
    }

    #[test]
    fn test_collapse_detection() {
        assert!(check_collapse(&[0.001, 0.005, 0.009]));
        assert!(!check_collapse(&[0.001, 0.5, 0.009]));
        assert!(!check_collapse(&[])); // No identities = not a collapse
    }

    #[test]
    fn test_fuse_scores() {
        let v_mvf = 0.8;
        let v_old = 1.2;

        // α = 0.5: equal blend
        let fused = fuse_scores(v_mvf, v_old, 0.5);
        assert!((fused - 1.0).abs() < 1e-6); // 0.5*0.8 + 0.5*1.2 = 1.0

        // α = 0.0: pure old
        let fused = fuse_scores(v_mvf, v_old, 0.0);
        assert!((fused - 1.2).abs() < 1e-6);

        // α clamped to 0.7 max
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

        // Mean should have moved toward 1.0
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
}
