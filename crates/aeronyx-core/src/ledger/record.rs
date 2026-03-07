// ============================================
// File: crates/aeronyx-core/src/ledger/record.rs
// ============================================
//! # MemoryRecord — MRS-1 Standard AI Memory Atomic Unit
//!
//! v2.1.0 — 4-layer cognitive model, plaintext embedding, u32 access_count.
//!
//! ## Key differences from v1.0:
//! - `MemoryLayer` has 4 variants (Identity/Knowledge/Episode/Archive)
//! - `embedding: Vec<f32>` replaces `encrypted_embedding: Vec<u8>`
//! - `access_count: u32` replaces `u64`
//! - `RecordStatus` has 3 variants (Active/Superseded/Revoked) — no Archived

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MemoryLayer {
    Identity = 0,
    Knowledge = 1,
    Episode = 2,
    Archive = 3,
}

impl MemoryLayer {
    #[must_use]
    pub fn recall_weight(self) -> f64 {
        match self {
            Self::Identity  => 0.30,
            Self::Knowledge => 0.20,
            Self::Episode   => 0.10,
            Self::Archive   => 0.05,
        }
    }

    #[must_use]
    pub fn stability_hours(self) -> f64 {
        match self {
            Self::Identity  => 8760.0,
            Self::Knowledge => 2160.0,
            Self::Episode   => 168.0,
            Self::Archive   => 720.0,
        }
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub record_id: [u8; 32],
    pub owner: [u8; 32],
    pub timestamp: u64,
    pub layer: MemoryLayer,
    pub topic_tags: Vec<String>,
    pub source_ai: String,
    pub encrypted_content: Vec<u8>,

    /// Plaintext f32 embedding for vector search. Not included in record_id hash.
    #[serde(default)]
    pub embedding: Vec<f32>,

    pub status: RecordStatus,
    pub supersedes: Option<[u8; 32]>,

    #[serde(with = "serde_bytes_64")]
    pub signature: [u8; 64],

    #[serde(default)]
    pub access_count: u32,
}

impl MemoryRecord {
    /// Canonical SHA-256 hash. embedding is NOT included.
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

    #[must_use]
    pub fn verify_id(&self) -> bool {
        let expected = Self::compute_record_id(
            &self.owner, self.timestamp, self.layer,
            &self.topic_tags, &self.source_ai, &self.encrypted_content,
        );
        self.record_id == expected
    }

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
        }
    }

    #[must_use] pub fn id_hex(&self) -> String { hex::encode(self.record_id) }
    #[must_use] pub fn owner_hex(&self) -> String { hex::encode(self.owner) }
    #[must_use] pub fn is_active(&self) -> bool { self.status.is_active() }
    #[must_use] pub fn content_size(&self) -> usize { self.encrypted_content.len() }
    #[must_use] pub fn embedding_dim(&self) -> usize { self.embedding.len() }
    #[must_use] pub fn has_embedding(&self) -> bool { !self.embedding.is_empty() }
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
        write!(f, "MemoryRecord({}: layer={}, status={}, emb={})",
            &self.id_hex()[..8], self.layer, self.status, self.embedding_dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(ts: u64, layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new([0xAA; 32], ts, layer, vec!["t".into()], "ai".into(),
            b"content".to_vec(), vec![0.1, 0.2, 0.3])
    }

    #[test] fn test_hash_deterministic() {
        assert_eq!(make(100, MemoryLayer::Episode).record_id, make(100, MemoryLayer::Episode).record_id);
    }
    #[test] fn test_hash_differs_layer() {
        assert_ne!(make(100, MemoryLayer::Episode).record_id, make(100, MemoryLayer::Archive).record_id);
    }
    #[test] fn test_verify_id() { assert!(make(100, MemoryLayer::Identity).verify_id()); }
    #[test] fn test_tamper() {
        let mut r = make(100, MemoryLayer::Episode);
        r.encrypted_content = b"tampered".to_vec();
        assert!(!r.verify_id());
    }
    #[test] fn test_embedding_not_in_hash() {
        let mut r = make(100, MemoryLayer::Episode);
        let r2 = make(100, MemoryLayer::Episode);
        r.embedding = vec![9.9];
        assert_eq!(r.record_id, r2.record_id);
    }
    #[test] fn test_serde() {
        let r = make(100, MemoryLayer::Knowledge);
        let b = bincode::serialize(&r).unwrap();
        let r2: MemoryRecord = bincode::deserialize(&b).unwrap();
        assert_eq!(r, r2);
        assert_eq!(r2.embedding, vec![0.1, 0.2, 0.3]);
    }
    #[test] fn test_layer_from_u8() {
        assert_eq!(MemoryLayer::from_u8(3), Some(MemoryLayer::Archive));
        assert_eq!(MemoryLayer::from_u8(4), None);
    }
    #[test] fn test_has_embedding() {
        assert!(make(0, MemoryLayer::Episode).has_embedding());
        let r = MemoryRecord::new([0;32],0,MemoryLayer::Episode,vec![],"".into(),vec![],vec![]);
        assert!(!r.has_embedding());
    }
}
