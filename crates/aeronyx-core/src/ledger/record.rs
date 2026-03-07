// ============================================
// File: crates/aeronyx-core/src/ledger/record.rs
// ============================================
//! # MemoryRecord — MRS-1 Standard AI Memory Unit
//!
//! ## Creation Reason
//! Replaces the simple `Fact` (subject-predicate-object triple) with a
//! richer, layered, encrypted memory record that supports the MemChain
//! Protocol's vision: **"DNS gave websites an address; MemChain gives
//! human AI memory a wallet."**
//!
//! ## Modification Reason (v2.1.0)
//! - Added `MemoryLayer::Archive` variant (value=3) to support the 4-layer
//!   cognitive model required by the smart Miner compaction pipeline.
//!   Archive represents compacted episodes — the "subconscious" layer with
//!   very low recall weight (0.05) but never deleted.
//! - Added `stability_hours()` and `recall_weight()` for Archive layer.
//! - Fixed `MemoryLayer::from_u8(3)` to return `Some(Archive)` instead of `None`.
//! - Added `MemoryLayer::as_str()` for JSON API serialization.
//!
//! ## Main Functionality
//! - `MemoryRecord` struct — the atomic unit of AI memory (MRS-1 standard)
//! - `MemoryLayer` enum — Identity / Knowledge / Episode / Archive classification
//! - `RecordStatus` enum — Active / Superseded / Revoked / Archived lifecycle
//! - `MemoryRecord::compute_record_id()` — deterministic SHA-256 content hash
//! - `MemoryRecord::verify_id()` — integrity check
//! - `MemoryRecord::verify_signature()` — Ed25519 signature verification
//!
//! ## Wire / Storage Format (MRS-1)
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ record_id          [u8; 32]   SHA-256 of identity fields    │
//! │ owner              [u8; 32]   Ed25519 wallet public key     │
//! │ timestamp          u64        Unix epoch seconds (UTC)      │
//! │ layer              u8         0=Identity, 1=Knowledge,      │
//! │                               2=Episode, 3=Archive          │
//! │ topic_tags         Vec<Str>   Categorisation tags            │
//! │ source_ai          String     e.g. "openclaw-v1"            │
//! │ status             u8         0=Active, 1=Superseded,       │
//! │                               2=Revoked, 3=Archived         │
//! │ supersedes         Option<[u8;32]>  Previous record ID      │
//! │ encrypted_content  Vec<u8>    AES/ChaCha encrypted payload  │
//! │ encrypted_embedding Vec<u8>   Encrypted vector embedding    │
//! │ signature          [u8; 64]   Ed25519 sig over record_id    │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Hashing Canonical Form (record_id)
//! To guarantee deterministic hashing across platforms:
//! ```text
//! SHA-256(
//!     owner (32 bytes)
//!     || timestamp (8 bytes LE)
//!     || layer (1 byte)
//!     || topic_tags_count (4 bytes LE)
//!     || for each tag: tag_len (4 bytes LE) || tag_utf8
//!     || source_ai_len (4 bytes LE) || source_ai_utf8
//!     || encrypted_content_len (4 bytes LE) || encrypted_content
//! )
//! ```
//!
//! ## Relationship to Fact
//! `Fact` is preserved for backward compatibility with existing P2P
//! messages (`BroadcastFact`, `SyncResponse`). New code should use
//! `MemoryRecord` exclusively. The two types are NOT interchangeable.
//!
//! ## Dependencies
//! - `sha2::Sha256` (workspace)
//! - `serde` (workspace)
//!
//! ## ⚠️ Important Note for Next Developer
//! - The canonical hash format MUST NOT change — it would invalidate
//!   every existing MemoryRecord's `record_id` and break signatures.
//! - `signature` covers `record_id` (the hash), not the raw fields.
//! - `owner` is the Ed25519 **public** key of the memory wallet owner.
//! - `encrypted_content` and `encrypted_embedding` are opaque blobs;
//!   only the owner's private key can decrypt them.
//! - `status` and `supersedes` enable record lifecycle management
//!   without deleting data from the append-only ledger.
//! - `MemoryLayer` discriminant values (0-3) are stored in SQLite `layer`
//!   column and in bincode P2P messages — do NOT reorder or renumber.
//! - `RecordStatus` discriminant values (0-3) are stored in SQLite `status`
//!   column — do NOT reorder or renumber.
//! - Keep `MemoryRecord` clonable and serialisable for P2P broadcast.
//!
//! ## Last Modified
//! v1.0.0 - Initial MemoryRecord definition (MRS-1 standard)
//! v2.1.0 - 🌟 Added Archive layer, stability_hours(), recall_weight(), as_str()

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

// ============================================
// Serde helpers for [u8; 64]
// ============================================
// serde only implements Serialize/Deserialize for arrays up to [T; 32].
// We provide a custom module for the 64-byte signature field.

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
// MemoryLayer — 4-Layer Cognitive Model
// ============================================

/// Classification layer for a memory record.
///
/// The 4-layer cognitive model, inspired by human memory architecture:
///
/// | Variant   | Value | Weight | Stability  | Description                    |
/// |-----------|-------|--------|------------|--------------------------------|
/// | Identity  | 0     | 0.30   | 8760h (1y) | Core user facts, near-permanent|
/// | Knowledge | 1     | 0.20   | 2160h (90d)| Distilled compacted knowledge  |
/// | Episode   | 2     | 0.10   | 168h  (7d) | Conversational events          |
/// | Archive   | 3     | 0.05   | 720h  (30d)| Compacted episodes, subconscious|
///
/// ## Discriminant Values (Stable Contract)
/// These integer values are stored in SQLite and serialized over P2P.
/// Do NOT reorder or renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MemoryLayer {
    /// Core identity attributes — highest priority, never auto-archived.
    Identity = 0,
    /// Accumulated knowledge — medium priority, created by compaction.
    Knowledge = 1,
    /// Individual episodes — subject to compaction by the Miner.
    Episode = 2,
    /// Compacted episodes — the "subconscious". Very low recall weight
    /// but never deleted. Created when Miner compacts Episodes.
    ///
    /// ## Design Note
    /// Archive records participate in recall with weight 0.05 (vs Episode 0.10).
    /// They "sink into the subconscious" but don't die — the user can still
    /// recall them if the semantic match is strong enough.
    Archive = 3,
}

impl MemoryLayer {
    /// Returns the recall weight (bias) added to the cognitive recall score.
    ///
    /// Used in the scoring formula:
    /// `score = semantic_similarity * time_decay * freq_boost + layer_weight`
    ///
    /// Identity has the highest weight to ensure it always surfaces in recall.
    #[must_use]
    pub fn recall_weight(self) -> f64 {
        match self {
            Self::Identity  => 0.30,
            Self::Knowledge => 0.20,
            Self::Episode   => 0.10,
            Self::Archive   => 0.05,
        }
    }

    /// Returns the time decay stability factor (in hours).
    ///
    /// Higher = slower decay. Used in:
    /// `time_decay = exp(-hours_since_creation / stability)`
    #[must_use]
    pub fn stability_hours(self) -> f64 {
        match self {
            Self::Identity  => 8760.0,  // 1 year — near-permanent
            Self::Knowledge => 2160.0,  // 90 days
            Self::Episode   => 168.0,   // 7 days
            Self::Archive   => 720.0,   // 30 days
        }
    }

    /// Converts a u8 value to a MemoryLayer.
    ///
    /// Returns `None` for unrecognised values (forward-compatibility).
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

    /// Convert to integer for SQLite storage.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns a human-readable name (for JSON API responses and logging).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Identity  => "identity",
            Self::Knowledge => "knowledge",
            Self::Episode   => "episode",
            Self::Archive   => "archive",
        }
    }

    /// Parse from a string (case-insensitive). Used by MPI JSON deserialization.
    ///
    /// Returns `None` for unrecognized strings.
    #[must_use]
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "identity"  => Some(Self::Identity),
            "knowledge" => Some(Self::Knowledge),
            "episode"   => Some(Self::Episode),
            "archive"   => Some(Self::Archive),
            _ => None,
        }
    }
}

impl std::fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================
// RecordStatus
// ============================================

/// Lifecycle status of a memory record.
///
/// Records progress through states but never truly disappear from
/// the append-only ledger. "Deletion" is achieved by marking a
/// record as `Revoked` and clearing its encrypted content locally.
///
/// ## Discriminant Values (Stable Contract)
/// These integer values are stored in SQLite `status` column.
/// Do NOT reorder or renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RecordStatus {
    /// Active and valid — included in recall results.
    Active = 0,
    /// Replaced by a newer record (pointed to by `supersedes` in the
    /// new record). Excluded from recall but retained for audit.
    Superseded = 1,
    /// Owner explicitly revoked this memory. Encrypted content is
    /// erased locally; only the tombstone metadata remains on-chain.
    Revoked = 2,
    /// Compacted by the Miner into a Knowledge record. Original
    /// Episode data may be pruned after archival.
    Archived = 3,
}

impl RecordStatus {
    /// Returns `true` if this record should be included in recall queries.
    #[must_use]
    pub fn is_active(self) -> bool {
        self == Self::Active
    }

    /// Converts a u8 value to a RecordStatus.
    ///
    /// Returns `None` for unrecognised values.
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Active),
            1 => Some(Self::Superseded),
            2 => Some(Self::Revoked),
            3 => Some(Self::Archived),
            _ => None,
        }
    }

    /// Returns a human-readable name for JSON/logging.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active     => "active",
            Self::Superseded => "superseded",
            Self::Revoked    => "revoked",
            Self::Archived   => "archived",
        }
    }
}

impl std::fmt::Display for RecordStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================
// MemoryRecord
// ============================================

/// A single, immutable AI memory record conforming to the MRS-1 standard.
///
/// # Integrity
/// `record_id` = SHA-256 of canonical identity + content fields.
/// `signature` = Ed25519(owner_private_key, record_id).
///
/// # Privacy
/// `encrypted_content` and `encrypted_embedding` are encrypted with
/// the owner's public key. Only the owner can decrypt and read them.
/// Metadata fields (`layer`, `topic_tags`, `status`, `timestamp`) are
/// in cleartext to enable indexing and lifecycle management.
///
/// # Thread Safety
/// `MemoryRecord` is `Send + Sync` (all fields are owned, no interior mutability).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    // ---- Identity Fields (public, participate in hash) ----

    /// Content-addressable identifier: SHA-256 of canonical fields.
    pub record_id: [u8; 32],

    /// Ed25519 public key of the memory wallet owner.
    pub owner: [u8; 32],

    /// Creation timestamp — Unix epoch seconds (UTC).
    pub timestamp: u64,

    /// Memory classification layer.
    pub layer: MemoryLayer,

    /// Categorisation tags for topic-based filtering.
    pub topic_tags: Vec<String>,

    /// Source AI system that generated this memory.
    /// Examples: "openclaw-v1", "claude-3.5", "gpt-4o"
    pub source_ai: String,

    // ---- Lifecycle Fields (public) ----

    /// Current lifecycle status.
    pub status: RecordStatus,

    /// If this record supersedes a previous one, its `record_id`.
    /// Used for update chains: new record points back to old record.
    pub supersedes: Option<[u8; 32]>,

    // ---- Content Fields (encrypted with owner's public key) ----

    /// Encrypted memory content (natural language text).
    /// Only decryptable by the owner's private key.
    pub encrypted_content: Vec<u8>,

    /// Encrypted vector embedding for semantic search.
    /// Only decryptable by the owner's private key.
    /// Empty if no embedding was generated.
    pub encrypted_embedding: Vec<u8>,

    // ---- Signature ----

    /// Ed25519 signature over `record_id`, produced by the owner.
    #[serde(with = "serde_bytes_64")]
    pub signature: [u8; 64],

    // ---- Local-only fields (not hashed, not broadcast) ----

    /// Access count — incremented on each recall hit.
    /// Used for frequency boost in scoring. Not part of the hash.
    #[serde(default)]
    pub access_count: u64,
}

impl MemoryRecord {
    /// Computes the canonical SHA-256 hash for a MemoryRecord.
    ///
    /// The canonical byte sequence is:
    /// ```text
    /// owner (32) || timestamp (8 LE) || layer (1)
    /// || topic_tags_count (4 LE) || for each tag: tag_len (4 LE) || tag_utf8
    /// || source_ai_len (4 LE) || source_ai_utf8
    /// || encrypted_content_len (4 LE) || encrypted_content
    /// ```
    ///
    /// Note: `encrypted_embedding` is NOT included in the hash to allow
    /// embedding regeneration without invalidating the record ID.
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

        // Owner
        hasher.update(owner);

        // Timestamp
        hasher.update(timestamp.to_le_bytes());

        // Layer
        hasher.update([layer as u8]);

        // Topic tags (length-prefixed for unambiguous parsing)
        let tag_count = topic_tags.len() as u32;
        hasher.update(tag_count.to_le_bytes());
        for tag in topic_tags {
            let tag_bytes = tag.as_bytes();
            let tag_len = tag_bytes.len() as u32;
            hasher.update(tag_len.to_le_bytes());
            hasher.update(tag_bytes);
        }

        // Source AI
        let source_bytes = source_ai.as_bytes();
        let source_len = source_bytes.len() as u32;
        hasher.update(source_len.to_le_bytes());
        hasher.update(source_bytes);

        // Encrypted content
        let content_len = encrypted_content.len() as u32;
        hasher.update(content_len.to_le_bytes());
        hasher.update(encrypted_content);

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verifies that `self.record_id` matches the canonical hash of the
    /// identity + content fields.
    ///
    /// # Returns
    /// `true` if the stored `record_id` is consistent with the content.
    #[must_use]
    pub fn verify_id(&self) -> bool {
        let expected = Self::compute_record_id(
            &self.owner,
            self.timestamp,
            self.layer,
            &self.topic_tags,
            &self.source_ai,
            &self.encrypted_content,
        );
        self.record_id == expected
    }

    /// Creates a new `MemoryRecord` with `record_id` automatically computed.
    ///
    /// `signature` is left zeroed — the caller is responsible for signing
    /// after construction.
    #[must_use]
    pub fn new(
        owner: [u8; 32],
        timestamp: u64,
        layer: MemoryLayer,
        topic_tags: Vec<String>,
        source_ai: String,
        encrypted_content: Vec<u8>,
        encrypted_embedding: Vec<u8>,
    ) -> Self {
        let record_id = Self::compute_record_id(
            &owner,
            timestamp,
            layer,
            &topic_tags,
            &source_ai,
            &encrypted_content,
        );
        Self {
            record_id,
            owner,
            timestamp,
            layer,
            topic_tags,
            source_ai,
            status: RecordStatus::Active,
            supersedes: None,
            encrypted_content,
            encrypted_embedding,
            signature: [0u8; 64],
            access_count: 0,
        }
    }

    /// Returns the `record_id` as a hex string (useful for logging / indexing).
    #[must_use]
    pub fn id_hex(&self) -> String {
        hex::encode(self.record_id)
    }

    /// Returns the `owner` as a hex string.
    #[must_use]
    pub fn owner_hex(&self) -> String {
        hex::encode(self.owner)
    }

    /// Returns `true` if this record is active and eligible for recall.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    /// Returns the size in bytes of the encrypted content.
    #[must_use]
    pub fn content_size(&self) -> usize {
        self.encrypted_content.len()
    }

    /// Returns the size in bytes of the encrypted embedding.
    #[must_use]
    pub fn embedding_size(&self) -> usize {
        self.encrypted_embedding.len()
    }
}

impl PartialEq for MemoryRecord {
    fn eq(&self, other: &Self) -> bool {
        // Two records are equal if their content-addressed IDs match.
        // This deliberately excludes `access_count` (local-only metadata).
        self.record_id == other.record_id
    }
}

impl Eq for MemoryRecord {}

impl std::hash::Hash for MemoryRecord {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.record_id.hash(state);
    }
}

impl std::fmt::Display for MemoryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryRecord({}: layer={}, status={}, tags={:?}, source={})",
            &self.id_hex()[..8],
            self.layer,
            self.status,
            self.topic_tags,
            self.source_ai,
        )
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(ts: u64, layer: MemoryLayer, source: &str) -> MemoryRecord {
        MemoryRecord::new(
            [0xAA; 32],
            ts,
            layer,
            vec!["test".to_string()],
            source.to_string(),
            b"encrypted_content_here".to_vec(),
            b"encrypted_embedding_here".to_vec(),
        )
    }

    #[test]
    fn test_compute_record_id_deterministic() {
        let r1 = make_record(100, MemoryLayer::Episode, "test-ai");
        let r2 = make_record(100, MemoryLayer::Episode, "test-ai");
        assert_eq!(r1.record_id, r2.record_id, "Same inputs must produce the same hash");
    }

    #[test]
    fn test_compute_record_id_differs_on_content() {
        let r1 = make_record(100, MemoryLayer::Episode, "test-ai");
        let r2 = make_record(100, MemoryLayer::Knowledge, "test-ai");
        assert_ne!(r1.record_id, r2.record_id);
    }

    #[test]
    fn test_compute_record_id_differs_on_timestamp() {
        let r1 = make_record(100, MemoryLayer::Episode, "test-ai");
        let r2 = make_record(200, MemoryLayer::Episode, "test-ai");
        assert_ne!(r1.record_id, r2.record_id);
    }

    #[test]
    fn test_compute_record_id_differs_on_source() {
        let r1 = make_record(100, MemoryLayer::Episode, "ai-a");
        let r2 = make_record(100, MemoryLayer::Episode, "ai-b");
        assert_ne!(r1.record_id, r2.record_id);
    }

    #[test]
    fn test_new_sets_record_id() {
        let record = make_record(1_700_000_000, MemoryLayer::Identity, "openclaw-v1");
        assert!(record.verify_id(), "record_id must match content hash after new()");
    }

    #[test]
    fn test_verify_id_detects_tamper() {
        let mut record = make_record(1_700_000_000, MemoryLayer::Episode, "test");
        record.encrypted_content = b"tampered".to_vec();
        assert!(!record.verify_id(), "Tampered record must fail verify_id");
    }

    #[test]
    fn test_new_defaults() {
        let record = make_record(100, MemoryLayer::Episode, "test");
        assert_eq!(record.status, RecordStatus::Active);
        assert!(record.supersedes.is_none());
        assert_eq!(record.signature, [0u8; 64]);
        assert_eq!(record.access_count, 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let record = make_record(1_700_000_000, MemoryLayer::Knowledge, "claude-3.5");
        let bytes = bincode::serialize(&record).expect("serialize");
        let restored: MemoryRecord = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(record, restored);
        assert_eq!(restored.layer, MemoryLayer::Knowledge);
        assert_eq!(restored.source_ai, "claude-3.5");
    }

    #[test]
    fn test_display() {
        let record = make_record(0, MemoryLayer::Identity, "test-ai");
        let s = format!("{}", record);
        assert!(s.starts_with("MemoryRecord("));
        assert!(s.contains("identity"));
        assert!(s.contains("test-ai"));
    }

    #[test]
    fn test_memory_layer_recall_weight() {
        assert!(MemoryLayer::Identity.recall_weight() > MemoryLayer::Knowledge.recall_weight());
        assert!(MemoryLayer::Knowledge.recall_weight() > MemoryLayer::Episode.recall_weight());
        assert!(MemoryLayer::Episode.recall_weight() > MemoryLayer::Archive.recall_weight());
        assert!(MemoryLayer::Archive.recall_weight() > 0.0);
    }

    #[test]
    fn test_memory_layer_stability() {
        assert!(MemoryLayer::Identity.stability_hours() > MemoryLayer::Knowledge.stability_hours());
        assert!(MemoryLayer::Knowledge.stability_hours() > MemoryLayer::Episode.stability_hours());
        // Archive (720h) > Episode (168h) because archived memories decay slower
        assert!(MemoryLayer::Archive.stability_hours() > MemoryLayer::Episode.stability_hours());
    }

    #[test]
    fn test_memory_layer_from_u8() {
        assert_eq!(MemoryLayer::from_u8(0), Some(MemoryLayer::Identity));
        assert_eq!(MemoryLayer::from_u8(1), Some(MemoryLayer::Knowledge));
        assert_eq!(MemoryLayer::from_u8(2), Some(MemoryLayer::Episode));
        assert_eq!(MemoryLayer::from_u8(3), Some(MemoryLayer::Archive));
        assert_eq!(MemoryLayer::from_u8(4), None);
    }

    #[test]
    fn test_memory_layer_as_str() {
        assert_eq!(MemoryLayer::Identity.as_str(), "identity");
        assert_eq!(MemoryLayer::Knowledge.as_str(), "knowledge");
        assert_eq!(MemoryLayer::Episode.as_str(), "episode");
        assert_eq!(MemoryLayer::Archive.as_str(), "archive");
    }

    #[test]
    fn test_memory_layer_from_str_loose() {
        assert_eq!(MemoryLayer::from_str_loose("Identity"), Some(MemoryLayer::Identity));
        assert_eq!(MemoryLayer::from_str_loose("ARCHIVE"), Some(MemoryLayer::Archive));
        assert_eq!(MemoryLayer::from_str_loose("episode"), Some(MemoryLayer::Episode));
        assert_eq!(MemoryLayer::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_memory_layer_roundtrip_u8() {
        for v in 0..=3u8 {
            let layer = MemoryLayer::from_u8(v).unwrap();
            assert_eq!(layer.as_u8(), v);
        }
    }

    #[test]
    fn test_record_status_from_u8() {
        assert_eq!(RecordStatus::from_u8(0), Some(RecordStatus::Active));
        assert_eq!(RecordStatus::from_u8(1), Some(RecordStatus::Superseded));
        assert_eq!(RecordStatus::from_u8(2), Some(RecordStatus::Revoked));
        assert_eq!(RecordStatus::from_u8(3), Some(RecordStatus::Archived));
        assert_eq!(RecordStatus::from_u8(4), None);
    }

    #[test]
    fn test_is_active() {
        let mut record = make_record(100, MemoryLayer::Episode, "test");
        assert!(record.is_active());

        record.status = RecordStatus::Revoked;
        assert!(!record.is_active());

        record.status = RecordStatus::Archived;
        assert!(!record.is_active());

        record.status = RecordStatus::Superseded;
        assert!(!record.is_active());
    }

    #[test]
    fn test_equality_ignores_access_count() {
        let mut r1 = make_record(100, MemoryLayer::Episode, "test");
        let mut r2 = make_record(100, MemoryLayer::Episode, "test");
        r1.access_count = 10;
        r2.access_count = 999;
        assert_eq!(r1, r2, "Equality should be based on record_id only");
    }

    #[test]
    fn test_tags_order_matters_for_hash() {
        let r1 = MemoryRecord::new(
            [0xAA; 32], 100, MemoryLayer::Episode,
            vec!["a".into(), "b".into()],
            "test".into(), b"content".to_vec(), vec![],
        );
        let r2 = MemoryRecord::new(
            [0xAA; 32], 100, MemoryLayer::Episode,
            vec!["b".into(), "a".into()],
            "test".into(), b"content".to_vec(), vec![],
        );
        assert_ne!(r1.record_id, r2.record_id, "Tag order must affect hash");
    }

    #[test]
    fn test_archive_layer_record() {
        let record = make_record(100, MemoryLayer::Archive, "miner-v1");
        assert!(record.verify_id());
        assert_eq!(record.layer, MemoryLayer::Archive);
        assert_eq!(record.layer.as_u8(), 3);
        let s = format!("{}", record);
        assert!(s.contains("archive"));
    }
}
