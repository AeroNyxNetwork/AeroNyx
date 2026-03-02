// ============================================
// File: crates/aeronyx-core/src/ledger/merkle.rs
// ============================================
//! # Merkle Tree — SHA-256 Hash Tree for Fact Integrity
//!
//! ## Creation Reason
//! Provides a minimal SHA-256 Merkle Tree implementation for computing
//! a single root hash from a list of Fact IDs. This root is embedded
//! in each `BlockHeader` to cryptographically bind all Facts in a Block.
//!
//! ## Main Functionality
//! - `merkle_root(leaves)` — compute the Merkle root of N leaf hashes
//!
//! ## Algorithm
//! ```text
//!          root = H(H01 || H23)
//!           /                \
//!     H01 = H(L0||L1)    H23 = H(L2||L3)
//!       /      \            /       \
//!      L0      L1         L2        L3
//! ```
//!
//! - If the number of leaves is odd, the last leaf is duplicated.
//! - Single leaf: root = leaf (no hashing).
//! - Empty input: root = `[0u8; 32]` (null hash).
//!
//! ## Dependencies
//! - `sha2::Sha256`
//!
//! ## ⚠️ Important Note for Next Developer
//! - This is a **stable contract**: the Merkle root computation must
//!   not change, or all existing Block headers become invalid.
//! - The duplication-of-odd-leaf approach matches Bitcoin's design.
//!
//! ## Last Modified
//! v0.5.0 - Initial Merkle Tree for Block integrity

use sha2::{Digest, Sha256};

/// Computes the SHA-256 Merkle root of a list of 32-byte leaf hashes.
///
/// # Arguments
/// * `leaves` - Slice of `[u8; 32]` leaf hashes (typically `fact_id` values).
///
/// # Returns
/// - `[0u8; 32]` if `leaves` is empty.
/// - The single leaf if `leaves.len() == 1`.
/// - The Merkle root hash otherwise.
///
/// # Algorithm
/// Bottom-up construction: pairs of adjacent hashes are concatenated
/// and hashed (`SHA-256(left || right)`). If a level has an odd number
/// of nodes, the last node is duplicated before pairing.
#[must_use]
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level: Vec<[u8; 32]> = leaves.to_vec();

    while current_level.len() > 1 {
        // If odd, duplicate the last element
        if current_level.len() % 2 != 0 {
            let last = *current_level.last().unwrap();
            current_level.push(last);
        }

        let mut next_level = Vec::with_capacity(current_level.len() / 2);

        for pair in current_level.chunks_exact(2) {
            let mut hasher = Sha256::new();
            hasher.update(pair[0]);
            hasher.update(pair[1]);
            let result = hasher.finalize();
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&result);
            next_level.push(hash);
        }

        current_level = next_level;
    }

    current_level[0]
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_leaves() {
        assert_eq!(merkle_root(&[]), [0u8; 32]);
    }

    #[test]
    fn test_single_leaf() {
        let leaf = [0xAB; 32];
        assert_eq!(merkle_root(&[leaf]), leaf);
    }

    #[test]
    fn test_two_leaves() {
        let a = [0x01; 32];
        let b = [0x02; 32];
        let root = merkle_root(&[a, b]);

        // Manual: SHA-256(a || b)
        let mut hasher = Sha256::new();
        hasher.update(a);
        hasher.update(b);
        let expected: [u8; 32] = hasher.finalize().into();

        assert_eq!(root, expected);
    }

    #[test]
    fn test_three_leaves_odd_duplication() {
        let a = [0x01; 32];
        let b = [0x02; 32];
        let c = [0x03; 32];
        let root = merkle_root(&[a, b, c]);

        // Level 1: H(a||b), H(c||c)
        let mut h = Sha256::new();
        h.update(a);
        h.update(b);
        let h_ab: [u8; 32] = h.finalize().into();

        let mut h = Sha256::new();
        h.update(c);
        h.update(c);
        let h_cc: [u8; 32] = h.finalize().into();

        // Root: H(h_ab || h_cc)
        let mut h = Sha256::new();
        h.update(h_ab);
        h.update(h_cc);
        let expected: [u8; 32] = h.finalize().into();

        assert_eq!(root, expected);
    }

    #[test]
    fn test_deterministic() {
        let leaves: Vec<[u8; 32]> = (0..10u8).map(|i| [i; 32]).collect();
        let r1 = merkle_root(&leaves);
        let r2 = merkle_root(&leaves);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_order_matters() {
        let a = [0x01; 32];
        let b = [0x02; 32];
        assert_ne!(merkle_root(&[a, b]), merkle_root(&[b, a]));
    }
}
