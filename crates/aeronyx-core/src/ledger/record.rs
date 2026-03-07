// ============================================
// File: crates/aeronyx-core/src/ledger/record.rs
// ============================================
//! # MemoryRecord — MRS-1 Standard AI Memory Atomic Unit
//!
//! ## Creation Reason
//! Core data structure for the 4-layer cognitive memory model. Every memory
//! in the system — whether from explicit /remember, rule engine extraction,
//! or Miner compaction — is represented as a MemoryRecord.
//!
//! ## Main Functionality
//! - `MemoryRecord`: Atomic memory unit with content-addressed ID (SHA-256)
//! - `MemoryLayer`: 4-layer cognitive architecture (Identity/Knowledge/Episode/Archive)
//! - `RecordStatus`: Lifecycle states (Active/Superseded/Revoked)
//! - Deterministic `record_id` computation for dedup and integrity verification
//!
//! ## Dependencies
//! - Used by: storage.rs (persistence), vector.rs (search), mvf.rs (scoring),
//!   mpi.rs (API), log_handler.rs (extraction), reflection.rs (compaction)
//! - This is the most widely depended-upon struct in the entire codebase.
//!
//! ## Main Logical Flow
//! 1. `new()` computes content-addressed `record_id` via SHA-256
//! 2. `verify_id()` validates integrity (detects tampering)
//! 3. Runtime metadata (embedding, feedback, conflict) is NOT in the hash
//!
//! ## ⚠️ Important Note for Next Developer
//! - `compute_record_id()` is an IMMUTABLE CONTRACT. Changing it breaks all existing records.
//! - Fields NOT in hash: embedding, positive_feedback, negative_feedback, conflict_with,
//!   access_count, status, supersedes, signature — these are mutable runtime metadata.
//! - `conflict_with` stores the record_id of a conflicting memory (for MVF φ₈ feature).
//!
//! ## Version History
//! v1.0.0 - Initial MemoryRecord with 4-layer cognitive model
//! v2.1.0 - Plaintext embedding Vec<f32>, u32 access_count
//! v2.1.0+MVF (Schema v4) - 🌟 Added positive_feedback, negative_feedback, conflict_with
//!   for MVF φ₄ (feedback score) and φ₈ (conflict penalty) features.
//!   These fields enable the Memory Value Function to learn from user corrections
//!   and down-rank conflicting or negatively-received memories via SGD online learning.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

// ============================================
// Serde helpers for [u8; 64]
// ============================================

mod serde_bytes_64 {
    use super::*;

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        serde_bytes::serialize(bytes.as_slice(), s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let slice: Vec<u8> = serde_bytes::deserialize(d)?;
        slice.try_into().map_err(|v: Vec<u8>| {
            serde::de::Error::invalid_length(v.len(), &"64 bytes")
        })
    }
}

// ============================================
// MemoryLayer — 4-Layer Cognitive Architecture
// ============================================

/// 4-layer cognitive memory architecture modeled after human memory systems.
///
/// ```text
/// ┌─────────────┬────────┬──────────────┬────────────────────────────┐
/// │ Layer       │ Weight │ Stability    │ Description                │
/// ├─────────────┼────────┼──────────────┼────────────────────────────┤
/// │ Identity    │ 0.30   │ 8760h (1yr)  │ Near-permanent, always top │
/// │ Knowledge   │ 0.20   │ 2160h (90d)  │ Distilled knowledge        │
/// │ Episode     │ 0.10   │ 168h  (7d)   │ Fast decay, compactable    │
/// │ Archive     │ 0.05   │ 720h  (30d)  │ Very low, "subconscious"   │
/// └─────────────┴────────┴──────────────┴────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MemoryLayer {
    Identity = 0,
    Knowledge = 1,
    Episode = 2,
    Archive = 3,
}

impl MemoryLayer {
    /// Recall scoring layer weight bias.
    /// Higher = more likely to appear in recall results.
    #[must_use]
    pub fn recall_weight(self) -> f64 {
        match self {
            Self::Identity  => 0.30,
            Self::Knowledge => 0.20,
            Self::Episode   => 0.10,
            Self::Archive   => 0.05,
        }
    }

    /// Time decay stability in hours.
    /// Higher = slower decay (Identity barely decays over a year).
    #[must_use]
    pub fn stability_hours(self) -> f64 {
        match self {
            Self::Identity  => 8760.0,
            Self::Knowledge => 2160.0,
            Self::Episode   => 168.0,
            Self::Archive   => 720.0,
        }
    }

    /// Convert from raw u8 value (for SQLite INTEGER column).
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Identity),
            1 => Some(Self::Knowledge),
            2 => Some(Self::Episode),
            3 => Some(Self::Archive),
            _ => None,
        }
    }
}

impl std::fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Identity  => write!(f, "identity"),
            Self::Knowledge => write!(f, "knowledge"),
            Self::Episode   => write!(f, "episode"),
            Self::Archive   => write!(f, "archive"),
        }
    }
}

// ============================================
// RecordStatus
// ============================================

/// Memory record lifecycle status.
///
/// - `Active`: Normal state, participates in recall
/// - `Superseded`: Replaced by a newer version (correction chaining)
/// - `Revoked`: User-requested deletion, content permanently erased
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RecordStatus {
    Active = 0,
    Superseded = 1,
    Revoked = 2,
}

impl RecordStatus {
    #[must_use]
    pub fn is_active(self) -> bool { self == Self::Active }

    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Active),
            1 => Some(Self::Superseded),
            2 => Some(Self::Revoked),
            _ => None,
        }
    }
}

impl std::fmt::Display for RecordStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active     => write!(f, "active"),
            Self::Superseded => write!(f, "superseded"),
            Self::Revoked    => write!(f, "revoked"),
        }
    }
}

// ============================================
// MemoryRecord
// ============================================

/// MRS-1 Standard AI Memory Atomic Unit.
///
/// Content-addressed by SHA-256 hash of (owner, timestamp, layer, topic_tags,
/// source_ai, encrypted_content). Mutable runtime metadata (embedding, feedback,
/// conflict, access_count, status, signature) is NOT included in the hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Content-addressed unique identifier (SHA-256 hash).
    /// IMMUTABLE CONTRACT: hash algorithm and input fields must never change.
    pub record_id: [u8; 32],

    /// Owner's Ed25519 public key (memory wallet address).
    pub owner: [u8; 32],

    /// Creation timestamp (Unix epoch seconds, set by the AI agent).
    pub timestamp: u64,

    /// Cognitive layer determining recall weight and decay rate.
    pub layer: MemoryLayer,

    /// Semantic tags for categorization and rule engine markers.
    /// Special tags: "_correction" (P3), "identity", "allergy", "preference", etc.
    pub topic_tags: Vec<String>,

    /// Identifier of the AI agent that created this memory.
    pub source_ai: String,

    /// Encrypted memory content (ChaCha20-Poly1305 in production,
    /// plaintext during development for debugging).
    pub encrypted_content: Vec<u8>,

    /// Plaintext f32 embedding for vector search. Not included in record_id hash.
    /// Empty vec means no embedding (rule engine extractions before Miner backfill).
    #[serde(default)]
    pub embedding: Vec<f32>,

    /// Lifecycle status (Active/Superseded/Revoked).
    pub status: RecordStatus,

    /// If this record supersedes another, the old record's ID.
    pub supersedes: Option<[u8; 32]>,

    /// Ed25519 signature over record_id, proving owner authenticity.
    #[serde(with = "serde_bytes_64")]
    pub signature: [u8; 64],

    /// Number of times this memory has been recalled (frequency boost in scoring).
    #[serde(default)]
    pub access_count: u32,

    // ========================================
    // v2.1.0+MVF (Schema v4) — Feedback & Conflict Fields
    // ========================================

    /// Positive feedback count from users (MVF φ₄ numerator component).
    ///
    /// Incremented by Miner Step 0 when detecting positive signals
    /// (e.g., user continues conversation in a way that validates the memory).
    /// Used in MVF feature: φ₄ = (pos - neg) / (pos + neg + 1)
    #[serde(default)]
    pub positive_feedback: u32,

    /// Negative feedback count from users (MVF φ₄ numerator component).
    ///
    /// Incremented by /log negative feedback detection when user says
    /// "wrong", "not correct", "搞错了" etc. and recall_context references
    /// this memory. Used in MVF feature: φ₄ = (pos - neg) / (pos + neg + 1)
    #[serde(default)]
    pub negative_feedback: u32,

    /// Record ID of a conflicting memory, if any (MVF φ₈ feature).
    ///
    /// Set by Miner Step 0.6 correction chaining when a _correction record
    /// supersedes this one. The conflict relationship is bidirectional:
    /// both the old and new record reference each other.
    /// Used in MVF feature: φ₈ = -𝟙(has_conflict) → penalty of -1.0
    #[serde(default)]
    pub conflict_with: Option<[u8; 32]>,
}

impl MemoryRecord {
    /// Canonical SHA-256 hash for content-addressed record_id.
    ///
    /// ## IMMUTABLE CONTRACT
    /// The following fields are included in the hash (in this exact order):
    /// owner, timestamp, layer, topic_tags (count + length-prefixed), source_ai, encrypted_content
    ///
    /// The following fields are EXCLUDED (mutable runtime metadata):
    /// embedding, status, supersedes, signature, access_count,
    /// positive_feedback, negative_feedback, conflict_with
    #[must_use]
    pub fn compute_record_id(
        owner: &[u8; 32],
        timestamp: u64,
        layer: MemoryLayer,
        topic_tags: &[String],
        source_ai: &str,
        encrypted_content: &[u8],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(owner);
        hasher.update(timestamp.to_le_bytes());
        hasher.update([layer as u8]);

        let tag_count = topic_tags.len() as u32;
        hasher.update(tag_count.to_le_bytes());
        for tag in topic_tags {
            let b = tag.as_bytes();
            hasher.update((b.len() as u32).to_le_bytes());
            hasher.update(b);
        }

        let sb = source_ai.as_bytes();
        hasher.update((sb.len() as u32).to_le_bytes());
        hasher.update(sb);

        let cl = encrypted_content.len() as u32;
        hasher.update(cl.to_le_bytes());
        hasher.update(encrypted_content);

        let r = hasher.finalize();
        let mut h = [0u8; 32];
        h.copy_from_slice(&r);
        h
    }

    /// Verify that `record_id` matches the canonical hash of the record's content fields.
    /// Returns false if the record has been tampered with.
    #[must_use]
    pub fn verify_id(&self) -> bool {
        let expected = Self::compute_record_id(
            &self.owner, self.timestamp, self.layer,
            &self.topic_tags, &self.source_ai, &self.encrypted_content,
        );
        self.record_id == expected
    }

    /// Create a new MemoryRecord with computed record_id.
    ///
    /// All mutable metadata fields (feedback, conflict) are initialized to defaults.
    /// The caller must set `signature` after creation using the owner's Ed25519 key.
    #[must_use]
    pub fn new(
        owner: [u8; 32],
        timestamp: u64,
        layer: MemoryLayer,
        topic_tags: Vec<String>,
        source_ai: String,
        encrypted_content: Vec<u8>,
        embedding: Vec<f32>,
    ) -> Self {
        let record_id = Self::compute_record_id(
            &owner, timestamp, layer, &topic_tags, &source_ai, &encrypted_content,
        );
        Self {
            record_id, owner, timestamp, layer, topic_tags, source_ai,
            encrypted_content, embedding,
            status: RecordStatus::Active,
            supersedes: None,
            signature: [0u8; 64],
            access_count: 0,
            positive_feedback: 0,
            negative_feedback: 0,
            conflict_with: None,
        }
    }

    // ========================================
    // Convenience accessors
    // ========================================

    #[must_use] pub fn id_hex(&self) -> String { hex::encode(self.record_id) }
    #[must_use] pub fn owner_hex(&self) -> String { hex::encode(self.owner) }
    #[must_use] pub fn is_active(&self) -> bool { self.status.is_active() }
    #[must_use] pub fn content_size(&self) -> usize { self.encrypted_content.len() }
    #[must_use] pub fn embedding_dim(&self) -> usize { self.embedding.len() }
    #[must_use] pub fn has_embedding(&self) -> bool { !self.embedding.is_empty() }

    /// Whether this memory has a conflict marker (for MVF φ₈ feature).
    #[must_use]
    pub fn has_conflict(&self) -> bool { self.conflict_with.is_some() }

    /// Compute MVF φ₄ feedback score: (pos - neg) / (pos + neg + 1).
    /// Range: (-1.0, 1.0). Neutral (no feedback) = 0.0.
    #[must_use]
    pub fn feedback_score(&self) -> f32 {
        let pos = self.positive_feedback as f32;
        let neg = self.negative_feedback as f32;
        (pos - neg) / (pos + neg + 1.0)
    }
}

impl PartialEq for MemoryRecord {
    fn eq(&self, other: &Self) -> bool { self.record_id == other.record_id }
}
impl Eq for MemoryRecord {}

impl std::hash::Hash for MemoryRecord {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.record_id.hash(state); }
}

impl std::fmt::Display for MemoryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryRecord({}: layer={}, status={}, emb={}, fb={}/{})",
            &self.id_hex()[..8], self.layer, self.status,
            self.embedding_dim(), self.positive_feedback, self.negative_feedback)
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make(ts: u64, layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new([0xAA; 32], ts, layer, vec!["t".into()], "ai".into(),
            b"content".to_vec(), vec![0.1, 0.2, 0.3])
    }

    #[test]
    fn test_hash_deterministic() {
        assert_eq!(make(100, MemoryLayer::Episode).record_id, make(100, MemoryLayer::Episode).record_id);
    }

    #[test]
    fn test_hash_differs_layer() {
        assert_ne!(make(100, MemoryLayer::Episode).record_id, make(100, MemoryLayer::Archive).record_id);
    }

    #[test]
    fn test_verify_id() {
        assert!(make(100, MemoryLayer::Identity).verify_id());
    }

    #[test]
    fn test_tamper() {
        let mut r = make(100, MemoryLayer::Episode);
        r.encrypted_content = b"tampered".to_vec();
        assert!(!r.verify_id());
    }

    #[test]
    fn test_embedding_not_in_hash() {
        let mut r = make(100, MemoryLayer::Episode);
        let r2 = make(100, MemoryLayer::Episode);
        r.embedding = vec![9.9];
        assert_eq!(r.record_id, r2.record_id);
    }

    #[test]
    fn test_feedback_not_in_hash() {
        let mut r = make(100, MemoryLayer::Episode);
        let r2 = make(100, MemoryLayer::Episode);
        r.positive_feedback = 42;
        r.negative_feedback = 7;
        r.conflict_with = Some([0xBB; 32]);
        // Feedback and conflict fields are NOT in the hash
        assert_eq!(r.record_id, r2.record_id);
    }

    #[test]
    fn test_serde() {
        let r = make(100, MemoryLayer::Knowledge);
        let b = bincode::serialize(&r).unwrap();
        let r2: MemoryRecord = bincode::deserialize(&b).unwrap();
        assert_eq!(r, r2);
        assert_eq!(r2.embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(r2.positive_feedback, 0);
        assert_eq!(r2.negative_feedback, 0);
        assert_eq!(r2.conflict_with, None);
    }

    #[test]
    fn test_serde_with_feedback() {
        let mut r = make(100, MemoryLayer::Episode);
        r.positive_feedback = 10;
        r.negative_feedback = 3;
        r.conflict_with = Some([0xCC; 32]);
        let b = bincode::serialize(&r).unwrap();
        let r2: MemoryRecord = bincode::deserialize(&b).unwrap();
        assert_eq!(r2.positive_feedback, 10);
        assert_eq!(r2.negative_feedback, 3);
        assert_eq!(r2.conflict_with, Some([0xCC; 32]));
    }

    #[test]
    fn test_layer_from_u8() {
        assert_eq!(MemoryLayer::from_u8(3), Some(MemoryLayer::Archive));
        assert_eq!(MemoryLayer::from_u8(4), None);
    }

    #[test]
    fn test_has_embedding() {
        assert!(make(0, MemoryLayer::Episode).has_embedding());
        let r = MemoryRecord::new([0;32],0,MemoryLayer::Episode,vec![],"".into(),vec![],vec![]);
        assert!(!r.has_embedding());
    }

    #[test]
    fn test_has_conflict() {
        let mut r = make(100, MemoryLayer::Episode);
        assert!(!r.has_conflict());
        r.conflict_with = Some([0xBB; 32]);
        assert!(r.has_conflict());
    }

    #[test]
    fn test_feedback_score() {
        let mut r = make(100, MemoryLayer::Episode);
        // No feedback → 0.0
        assert!((r.feedback_score() - 0.0).abs() < 1e-6);

        // 3 pos, 1 neg → (3-1)/(3+1+1) = 0.4
        r.positive_feedback = 3;
        r.negative_feedback = 1;
        assert!((r.feedback_score() - 0.4).abs() < 1e-6);

        // 0 pos, 5 neg → (0-5)/(0+5+1) = -0.833...
        r.positive_feedback = 0;
        r.negative_feedback = 5;
        assert!((r.feedback_score() - (-5.0 / 6.0)).abs() < 1e-4);
    }

    #[test]
    fn test_default_fields() {
        let r = make(100, MemoryLayer::Identity);
        assert_eq!(r.positive_feedback, 0);
        assert_eq!(r.negative_feedback, 0);
        assert_eq!(r.conflict_with, None);
        assert_eq!(r.access_count, 0);
        assert_eq!(r.status, RecordStatus::Active);
    }

    #[test]
    fn test_display_includes_feedback() {
        let mut r = make(100, MemoryLayer::Episode);
        r.positive_feedback = 5;
        r.negative_feedback = 2;
        let s = format!("{}", r);
        assert!(s.contains("fb=5/2"), "Display should show feedback: {}", s);
    }
}
