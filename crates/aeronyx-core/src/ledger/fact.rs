// ============================================
// File: crates/aeronyx-core/src/ledger/fact.rs
// ============================================
//! # Fact — Atomic Unit of AI Memory
//!
//! ## Creation Reason
//! Represents a single, immutable piece of AI knowledge stored in the
//! MemChain ledger. Every Fact is a subject-predicate-object triple
//! (like an RDF triple) that is content-hashed and Ed25519-signed by
//! the originating node.
//!
//! ## Main Functionality
//! - `Fact` struct with serde Serialize/Deserialize
//! - `Fact::compute_hash()` — deterministic SHA-256 digest of canonical content
//! - `Fact::verify_id()` — checks that `fact_id` matches the content hash
//!
//! ## Wire / Storage Format
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ fact_id      [u8; 32]   SHA-256 of canonical content │
//! │ timestamp    u64        Unix epoch seconds (UTC)     │
//! │ subject      String     e.g. "user.preference"       │
//! │ predicate    String     e.g. "favorite_language"      │
//! │ object       String     e.g. "Rust"                   │
//! │ origin       [u8; 32]   Ed25519 public key of author │
//! │ signature    [u8; 64]   Ed25519 sig over hash        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Hashing Canonical Form
//! To guarantee deterministic hashing across platforms:
//! ```text
//! SHA-256( timestamp_le_bytes || subject_utf8 || 0x00
//!          || predicate_utf8 || 0x00 || object_utf8 )
//! ```
//! The null byte `0x00` acts as an unambiguous field separator.
//!
//! ## Dependencies
//! - `sha2::Sha256` (workspace)
//! - `serde` (workspace)
//!
//! ## ⚠️ Important Note for Next Developer
//! - The canonical hash format MUST NOT change — it would invalidate
//!   every existing Fact's `fact_id` and break signature verification.
//! - `signature` covers `fact_id` (the hash), not the raw fields.
//! - `origin` is the Ed25519 **public** key of the signing node so
//!   that any peer can verify the signature without a lookup.
//! - Keep `Fact` clonable and cheaply serialisable for P2P broadcast.
//!
//! ## Last Modified
//! v0.2.0 - Initial Fact definition for MemChain integration

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ============================================
// Fact
// ============================================

/// A single, immutable AI memory record.
///
/// # Integrity
/// `fact_id` = SHA-256 of the canonical content bytes.
/// `signature` = Ed25519(origin_private_key, fact_id).
///
/// # Thread Safety
/// `Fact` is `Send + Sync` (all fields are owned, no interior mutability).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    /// Content-addressable identifier: SHA-256 of canonical fields.
    pub fact_id: [u8; 32],

    /// Creation timestamp — Unix epoch seconds (UTC).
    pub timestamp: u64,

    /// Subject of the triple (e.g. `"user.preference"`).
    pub subject: String,

    /// Predicate / relation (e.g. `"favorite_language"`).
    pub predicate: String,

    /// Object / value (e.g. `"Rust"`).
    pub object: String,

    /// Ed25519 public key of the node that created this Fact.
    pub origin: [u8; 32],

    /// Ed25519 signature over `fact_id`, produced by the origin node.
    pub signature: [u8; 64],
}

impl Fact {
    /// Computes the canonical SHA-256 hash for the given content fields.
    ///
    /// The canonical byte sequence is:
    /// ```text
    /// timestamp (8 bytes LE) || subject (UTF-8) || 0x00
    ///                        || predicate (UTF-8) || 0x00
    ///                        || object (UTF-8)
    /// ```
    ///
    /// # Arguments
    /// * `timestamp` - Unix epoch seconds
    /// * `subject`   - Subject string
    /// * `predicate` - Predicate string
    /// * `object`    - Object string
    #[must_use]
    pub fn compute_hash(
        timestamp: u64,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(timestamp.to_le_bytes());
        hasher.update(subject.as_bytes());
        hasher.update(b"\x00");
        hasher.update(predicate.as_bytes());
        hasher.update(b"\x00");
        hasher.update(object.as_bytes());

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verifies that `self.fact_id` matches the canonical hash of the
    /// content fields.
    ///
    /// # Returns
    /// `true` if the stored `fact_id` is consistent with the content.
    #[must_use]
    pub fn verify_id(&self) -> bool {
        let expected = Self::compute_hash(
            self.timestamp,
            &self.subject,
            &self.predicate,
            &self.object,
        );
        self.fact_id == expected
    }

    /// Creates a new `Fact` with the `fact_id` automatically computed.
    ///
    /// `origin` and `signature` are left zeroed — the caller is
    /// responsible for signing after construction (see `MemPool::sign_and_add`).
    #[must_use]
    pub fn new(
        timestamp: u64,
        subject: String,
        predicate: String,
        object: String,
    ) -> Self {
        let fact_id = Self::compute_hash(timestamp, &subject, &predicate, &object);
        Self {
            fact_id,
            timestamp,
            subject,
            predicate,
            object,
            origin: [0u8; 32],
            signature: [0u8; 64],
        }
    }

    /// Returns the `fact_id` as a hex string (useful for logging / indexing).
    #[must_use]
    pub fn id_hex(&self) -> String {
        hex::encode(self.fact_id)
    }
}

// ============================================
// Display
// ============================================

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fact({}: {} {} {})",
            &self.id_hex()[..8],
            self.subject,
            self.predicate,
            self.object,
        )
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash_deterministic() {
        let h1 = Fact::compute_hash(100, "user", "likes", "Rust");
        let h2 = Fact::compute_hash(100, "user", "likes", "Rust");
        assert_eq!(h1, h2, "Same inputs must produce the same hash");
    }

    #[test]
    fn test_compute_hash_differs_on_content_change() {
        let h1 = Fact::compute_hash(100, "user", "likes", "Rust");
        let h2 = Fact::compute_hash(100, "user", "likes", "Go");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_new_sets_fact_id() {
        let fact = Fact::new(1_700_000_000, "a".into(), "b".into(), "c".into());
        assert!(fact.verify_id(), "fact_id must match content hash after new()");
    }

    #[test]
    fn test_verify_id_detects_tamper() {
        let mut fact = Fact::new(1_700_000_000, "a".into(), "b".into(), "c".into());
        fact.object = "tampered".into();
        assert!(!fact.verify_id(), "Tampered fact must fail verify_id");
    }

    #[test]
    fn test_serde_roundtrip() {
        let fact = Fact::new(1_700_000_000, "subj".into(), "pred".into(), "obj".into());
        let bytes = bincode::serialize(&fact).expect("serialize");
        let restored: Fact = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(fact, restored);
    }

    #[test]
    fn test_display() {
        let fact = Fact::new(0, "user".into(), "likes".into(), "Rust".into());
        let s = format!("{}", fact);
        assert!(s.starts_with("Fact("));
        assert!(s.contains("user"));
    }
}
