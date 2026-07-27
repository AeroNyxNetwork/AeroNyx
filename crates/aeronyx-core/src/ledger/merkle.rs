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
//! - `build_merkle_inclusion_proof(leaves, index)` — build the canonical
//!   sibling path for exactly one leaf
//! - `verify_merkle_inclusion_proof(...)` — verify that path against an
//!   independently trusted root without reconstructing the complete tree
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
//! - [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] Proof verification
//!   checks the declared leaf count and requires the duplicated odd sibling to
//!   equal the current node. Accepting an arbitrary virtual sibling would
//!   verify a different tree than `merkle_root`.
//!
//! ## Last Modified
//! v0.5.0 - Initial Merkle Tree for Block integrity
//! v0.6.0 - [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] Added canonical,
//! count-bound inclusion proof construction and verification.

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

        next_level.extend(
            current_level
                .chunks_exact(2)
                .map(|pair| hash_pair(&pair[0], &pair[1])),
        );

        current_level = next_level;
    }

    current_level[0]
}

/// Builds the canonical sibling path for one Merkle leaf.
///
/// The returned vector is ordered from the leaf level toward the root. For an
/// odd final node at any level, the current node itself is included as the
/// sibling because [`merkle_root`] duplicates that node.
///
/// # Returns
/// `None` when `leaves` is empty or `leaf_index` is out of range. A one-leaf
/// tree returns an empty, valid proof.
#[must_use]
pub fn build_merkle_inclusion_proof(
    leaves: &[[u8; 32]],
    leaf_index: usize,
) -> Option<Vec<[u8; 32]>> {
    if leaves.is_empty() || leaf_index >= leaves.len() {
        return None;
    }

    let mut current_level = leaves.to_vec();
    let mut current_index = leaf_index;
    let mut siblings = Vec::with_capacity(merkle_tree_depth(leaves.len()));

    while current_level.len() > 1 {
        if current_level.len() % 2 != 0 {
            let last = *current_level
                .last()
                .expect("non-empty Merkle level has a final node");
            current_level.push(last);
        }

        let sibling_index = if current_index % 2 == 0 {
            current_index + 1
        } else {
            current_index - 1
        };
        siblings.push(current_level[sibling_index]);

        current_level = current_level
            .chunks_exact(2)
            .map(|pair| hash_pair(&pair[0], &pair[1]))
            .collect();
        current_index /= 2;
    }

    Some(siblings)
}

/// Verifies one canonical Merkle inclusion path against `expected_root`.
///
/// `leaf_count` is part of the proof context and fixes the expected path depth
/// and odd-tail positions. Protocols using duplicate-odd trees must still bind
/// that count outside the root when exact cardinality matters; AeroNyx signs it
/// in the Directory block header and separately rejects duplicate commitments.
#[must_use]
pub fn verify_merkle_inclusion_proof(
    expected_root: &[u8; 32],
    leaf: &[u8; 32],
    leaf_index: usize,
    leaf_count: usize,
    siblings: &[[u8; 32]],
) -> bool {
    if leaf_count == 0
        || leaf_index >= leaf_count
        || siblings.len() != merkle_tree_depth(leaf_count)
    {
        return false;
    }

    let mut current = *leaf;
    let mut current_index = leaf_index;
    let mut level_nodes = leaf_count;

    for sibling in siblings {
        let duplicated_odd_tail =
            current_index % 2 == 0 && current_index.saturating_add(1) >= level_nodes;
        if duplicated_odd_tail && sibling != &current {
            return false;
        }

        current = if current_index % 2 == 0 {
            hash_pair(&current, sibling)
        } else {
            hash_pair(sibling, &current)
        };
        current_index /= 2;
        level_nodes = level_nodes.saturating_add(1) / 2;
    }

    &current == expected_root
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn merkle_tree_depth(mut leaf_count: usize) -> usize {
    let mut depth = 0usize;
    while leaf_count > 1 {
        leaf_count = leaf_count.saturating_add(1) / 2;
        depth = depth.saturating_add(1);
    }
    depth
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

    #[test]
    fn inclusion_proofs_roundtrip_every_leaf_and_tree_shape() {
        for leaf_count in 1usize..=33 {
            let leaves = (0..leaf_count)
                .map(|index| {
                    let mut hasher = Sha256::new();
                    hasher.update(b"AeroNyx-Merkle-Proof-Test");
                    hasher.update(index.to_le_bytes());
                    hasher.finalize().into()
                })
                .collect::<Vec<[u8; 32]>>();
            let root = merkle_root(&leaves);

            for (leaf_index, leaf) in leaves.iter().enumerate() {
                let proof = build_merkle_inclusion_proof(&leaves, leaf_index)
                    .expect("in-range leaf must produce a proof");
                assert!(verify_merkle_inclusion_proof(
                    &root, leaf, leaf_index, leaf_count, &proof,
                ));
            }
        }
    }

    #[test]
    fn inclusion_proof_rejects_shape_position_and_hash_tampering() {
        let leaves = [[0x01; 32], [0x02; 32], [0x03; 32]];
        let root = merkle_root(&leaves);
        let proof =
            build_merkle_inclusion_proof(&leaves, 2).expect("third leaf must produce a proof");

        assert!(!verify_merkle_inclusion_proof(
            &root, &leaves[2], 3, 3, &proof,
        ));
        assert!(!verify_merkle_inclusion_proof(
            &root, &leaves[2], 2, 5, &proof,
        ));

        let mut wrong_sibling = proof.clone();
        wrong_sibling[0][0] ^= 0x01;
        assert!(!verify_merkle_inclusion_proof(
            &root,
            &leaves[2],
            2,
            leaves.len(),
            &wrong_sibling,
        ));

        let mut wrong_root = root;
        wrong_root[0] ^= 0x01;
        assert!(!verify_merkle_inclusion_proof(
            &wrong_root,
            &leaves[2],
            2,
            leaves.len(),
            &proof,
        ));
    }

    #[test]
    fn inclusion_proof_handles_single_leaf_and_rejects_invalid_requests() {
        let leaf = [0xAB; 32];
        let proof = build_merkle_inclusion_proof(&[leaf], 0)
            .expect("single leaf must produce an empty proof");
        assert!(proof.is_empty());
        assert!(verify_merkle_inclusion_proof(&leaf, &leaf, 0, 1, &proof));

        assert!(build_merkle_inclusion_proof(&[], 0).is_none());
        assert!(build_merkle_inclusion_proof(&[leaf], 1).is_none());
        assert!(!verify_merkle_inclusion_proof(
            &[0u8; 32], &leaf, 0, 0, &proof,
        ));
    }
}
