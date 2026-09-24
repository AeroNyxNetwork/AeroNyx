// =============================================================================
// File: crates/aeronyx-server/src/services/discovery_peer_sampling.rs
// =============================================================================
//! Private peer sampling for permissionless discovery and exact chat selection.
//!
//! [PERMISSIONLESS-DISCOVERY-SAMPLING 2026-09-24 by Codex] This module only
//! projects current, signed PeerStore evidence. It does not import peers,
//! grant routeability, send traffic, or publish peer-level diagnostics.
//! Callers must supply fresh per-round entropy and the existing operation-
//! specific endpoint safety/canonicalization policy. Runtime gossip and chat
//! dispatch integration are deliberately outside this file's ownership.

use std::collections::HashMap;

use aeronyx_core::protocol::discovery::{
    NodeCapability, NodeProtocolFeature, SignedNodeDescriptor,
};
use sha2::{Digest, Sha256};

use super::peer_store::PeerStore;

const GOSSIP_RANK_DOMAIN: &[u8] = b"aeronyx:gossip-peer-sample:v1";
const CHAT_RANK_DOMAIN: &[u8] = b"aeronyx:receipt-proven-chat-terminal:v1";

/// A signed descriptor together with the caller-approved canonical transport.
/// Neither field is suitable for public health, logs, or aggregate telemetry.
pub(crate) struct SampledGossipPeer {
    pub(crate) descriptor: SignedNodeDescriptor,
    pub(crate) canonical_endpoint: String,
}

fn rank(domain: &[u8], round_nonce: &[u8; 32], node_id: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(round_nonce);
    hasher.update(node_id);
    hasher.finalize().into()
}

/// Samples at most `limit` fresh public peers from the complete verified view.
///
/// The endpoint adapter must reject unsafe or non-gossip transports and return
/// their canonical URL. A URL claimed by multiple identities is excluded in
/// full, including when one claim would otherwise fall outside the round
/// budget. Network anti-affinity is deliberately coarse: IP /24 or /48 and
/// DNS hostname, not an operator/AS proof. A short sample stays short instead
/// of filling it with collocated peers. The nonce must be fresh and private per
/// round; no identity, endpoint, or rank is emitted to diagnostics.
pub(crate) fn sample_public_gossip_peers<F>(
    peer_store: &PeerStore,
    now: u64,
    round_nonce: [u8; 32],
    limit: usize,
    excluded_node_ids: &[[u8; 32]],
    canonical_safe_endpoint: F,
) -> Vec<SampledGossipPeer>
where
    F: Fn(&str) -> Option<String>,
{
    if limit == 0 {
        return Vec::new();
    }

    // Read the complete bounded store before ranking. Truncating the sorted
    // bootstrap snapshot first would let low node IDs monopolize every round.
    let mut candidates = Vec::new();
    let mut canonical_claims = HashMap::<String, usize>::new();
    let public_identities = peer_store.valid_public_endpoint_identities(now);
    for (node_id, endpoint) in &public_identities {
        let Some(canonical_endpoint) = canonical_safe_endpoint(&endpoint) else {
            continue;
        };
        if canonical_endpoint.is_empty() {
            continue;
        }
        let Some(descriptor) = peer_store.get_valid(node_id, now) else {
            continue;
        };
        // A descriptor may rotate between the two reads. Never borrow an old
        // endpoint's identity for a newer signed descriptor.
        if !descriptor.descriptor.policy.public_discovery
            || descriptor.descriptor.public_endpoint.as_deref() != Some(endpoint.as_str())
        {
            continue;
        }
        *canonical_claims
            .entry(canonical_endpoint.clone())
            .or_default() += 1;
        if !excluded_node_ids.contains(node_id) {
            candidates.push(SampledGossipPeer {
                descriptor,
                canonical_endpoint,
            });
        }
    }

    // Do not form a sample from mixed generations of a concurrently rotated
    // store. A subsequent round can retry with its own fresh nonce.
    if peer_store.valid_public_endpoint_identities(now) != public_identities {
        return Vec::new();
    }

    candidates.retain(|candidate| canonical_claims[&candidate.canonical_endpoint] == 1);
    candidates.sort_by_cached_key(|candidate| {
        let node_id = candidate.descriptor.node_id();
        (rank(GOSSIP_RANK_DOMAIN, &round_nonce, &node_id), node_id)
    });

    let mut selected: Vec<SampledGossipPeer> = Vec::with_capacity(limit.min(candidates.len()));
    for candidate in candidates {
        if selected.iter().all(|prior| {
            PeerStore::route_endpoints_are_network_diverse(&candidate.descriptor, &prior.descriptor)
        }) {
            selected.push(candidate);
            if selected.len() == limit {
                break;
            }
        }
    }
    selected
}

/// Coarse, ID-free reason for refusing a chat terminal selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatTerminalSelectionFailure {
    NoReceiptProvenChatTerminal,
    PinnedTerminalUnavailable,
}

/// Selects exactly one current, receipt-proven chat terminal, never a fanout.
///
/// PeerStore first enforces routeability, non-quarantine, current purpose-bound
/// receipt evidence, and multi-hop policy. This boundary then requires the
/// exact signed chat capability and v2 purpose feature. A missing or stale pin
/// fails closed; it can never degrade into a random recipient. This selector
/// does not prove that the remaining middle hop/path is ready, and must not be
/// used as a delivery-success signal.
pub(crate) fn select_receipt_proven_chat_terminal(
    peer_store: &PeerStore,
    now: u64,
    round_nonce: [u8; 32],
    pinned_node_id: Option<[u8; 32]>,
    excluded_node_ids: &[[u8; 32]],
) -> Result<SignedNodeDescriptor, ChatTerminalSelectionFailure> {
    let candidates = peer_store
        .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            usize::MAX,
            excluded_node_ids,
        );
    select_chat_terminal_from_route_proven_candidates(candidates, now, round_nonce, pinned_node_id)
}

// Private pure seam: production input comes only from PeerStore's current
// receipt-proof selector above. Tests exercise pin and feature semantics here.
fn select_chat_terminal_from_route_proven_candidates(
    candidates: Vec<SignedNodeDescriptor>,
    now: u64,
    round_nonce: [u8; 32],
    pinned_node_id: Option<[u8; 32]>,
) -> Result<SignedNodeDescriptor, ChatTerminalSelectionFailure> {
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| {
            candidate.verify_at(now).is_ok()
                && candidate
                    .descriptor
                    .capabilities
                    .contains(&NodeCapability::ChatRelay)
                && candidate
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::PurposeBoundDeliveryReceiptV2)
        })
        .collect::<Vec<_>>();

    if let Some(pinned_node_id) = pinned_node_id {
        return candidates
            .into_iter()
            .find(|candidate| candidate.node_id() == pinned_node_id)
            .ok_or(ChatTerminalSelectionFailure::PinnedTerminalUnavailable);
    }
    if candidates.is_empty() {
        return Err(ChatTerminalSelectionFailure::NoReceiptProvenChatTerminal);
    }
    candidates.sort_by_cached_key(|candidate| {
        let node_id = candidate.node_id();
        (rank(CHAT_RANK_DOMAIN, &round_nonce, &node_id), node_id)
    });
    Ok(candidates.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::NodeDescriptor;

    const NOW: u64 = 1_780_000_000;

    fn descriptor(seed: u8, endpoint: &str, public: bool, expires_at: u64) -> SignedNodeDescriptor {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut body = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            NOW - 10,
            expires_at,
            "1.0.0",
        );
        body.public_endpoint = Some(endpoint.to_owned());
        body.policy.public_discovery = public;
        body.capabilities = vec![NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(body, &identity).unwrap()
    }

    fn store_with(descriptors: impl IntoIterator<Item = SignedNodeDescriptor>) -> PeerStore {
        let store = PeerStore::new();
        for descriptor in descriptors {
            store.upsert_verified(descriptor, NOW).unwrap();
        }
        store
    }

    fn safe_url(endpoint: &str) -> Option<String> {
        endpoint
            .starts_with("https://")
            .then(|| endpoint.to_ascii_lowercase())
    }

    fn ids(peers: &[SampledGossipPeer]) -> Vec<[u8; 32]> {
        peers.iter().map(|peer| peer.descriptor.node_id()).collect()
    }

    #[test]
    fn sample_converges_across_insertion_order_and_rejects_collocated_peers() {
        let peers = vec![
            descriptor(1, "https://198.51.100.1", true, NOW + 100),
            descriptor(2, "https://198.51.100.2", true, NOW + 100),
            descriptor(3, "https://203.0.113.1", true, NOW + 100),
            descriptor(4, "https://[2001:db8:1::1]", true, NOW + 100),
        ];
        let first = store_with(peers.clone());
        let second = store_with(peers.into_iter().rev());
        let left = sample_public_gossip_peers(&first, NOW, [7; 32], 4, &[], safe_url);
        let right = sample_public_gossip_peers(&second, NOW, [7; 32], 4, &[], safe_url);
        assert_eq!(ids(&left), ids(&right));
        assert_eq!(left.len(), 3);
        for (index, peer) in left.iter().enumerate() {
            assert!(peer.descriptor.verify_at(NOW).is_ok());
            assert!(peer.canonical_endpoint.starts_with("https://"));
            for previous in &left[..index] {
                assert!(PeerStore::route_endpoints_are_network_diverse(
                    &peer.descriptor,
                    &previous.descriptor,
                ));
            }
        }
    }

    #[test]
    fn sample_rejects_private_expired_unsafe_and_excluded_peers() {
        let excluded = descriptor(1, "https://198.51.100.1", true, NOW + 100);
        let store = store_with([
            excluded.clone(),
            descriptor(2, "https://203.0.113.1", false, NOW + 100),
            descriptor(3, "http://192.0.2.1", true, NOW + 100),
            descriptor(4, "https://203.0.113.2", true, NOW + 1),
        ]);
        assert!(sample_public_gossip_peers(
            &store,
            NOW + 2,
            [8; 32],
            4,
            &[excluded.node_id()],
            safe_url,
        )
        .is_empty());
        assert!(sample_public_gossip_peers(&store, NOW, [8; 32], 0, &[], safe_url).is_empty());
    }

    #[test]
    fn canonical_endpoint_collision_excludes_all_claimants_before_limit() {
        let clean = descriptor(3, "https://203.0.113.7", true, NOW + 100);
        let store = store_with([
            descriptor(1, "https://198.51.100.1", true, NOW + 100),
            descriptor(2, "https://198.51.100.1", true, NOW + 100),
            clean.clone(),
        ]);
        let selected = sample_public_gossip_peers(&store, NOW, [9; 32], 1, &[], safe_url);
        assert_eq!(ids(&selected), vec![clean.node_id()]);
    }

    #[test]
    fn excluded_claimant_does_not_launder_a_canonical_collision() {
        let excluded = descriptor(1, "https://198.51.100.1", true, NOW + 100);
        let store = store_with([
            excluded.clone(),
            descriptor(2, "https://198.51.100.1", true, NOW + 100),
        ]);
        assert!(sample_public_gossip_peers(
            &store,
            NOW,
            [9; 32],
            1,
            &[excluded.node_id()],
            safe_url,
        )
        .is_empty());
    }

    #[test]
    fn round_entropy_rotates_bounded_choice_without_changing_peer_set() {
        let store = store_with([
            descriptor(1, "https://198.51.100.1", true, NOW + 100),
            descriptor(2, "https://203.0.113.1", true, NOW + 100),
        ]);
        let first = sample_public_gossip_peers(&store, NOW, [0; 32], 1, &[], safe_url);
        assert_eq!(
            ids(&first),
            ids(&sample_public_gossip_peers(
                &store,
                NOW,
                [0; 32],
                1,
                &[],
                safe_url
            ))
        );
        assert!((1..=255).any(|nonce| {
            ids(&sample_public_gossip_peers(
                &store,
                NOW,
                [nonce; 32],
                1,
                &[],
                safe_url,
            )) != ids(&first)
        }));
    }

    #[test]
    fn chat_selection_fails_closed_without_receipt_proof_even_when_signed() {
        let peer = descriptor(1, "https://198.51.100.1", true, NOW + 100);
        let node_id = peer.node_id();
        let store = store_with([peer]);
        assert!(matches!(
            select_receipt_proven_chat_terminal(&store, NOW, [1; 32], Some(node_id), &[]),
            Err(ChatTerminalSelectionFailure::PinnedTerminalUnavailable)
        ));
        assert!(matches!(
            select_receipt_proven_chat_terminal(&store, NOW, [1; 32], None, &[]),
            Err(ChatTerminalSelectionFailure::NoReceiptProvenChatTerminal)
        ));
    }

    #[test]
    fn chat_pin_is_exact_and_missing_purpose_never_falls_back() {
        let eligible = descriptor(1, "https://198.51.100.1", true, NOW + 100);
        let incompatible = descriptor(2, "https://203.0.113.1", true, NOW + 100);
        let eligible_identity = IdentityKeyPair::from_bytes(&[1; 32]).unwrap();
        let eligible_body = eligible
            .descriptor
            .clone()
            .with_protocol_features([NodeProtocolFeature::PurposeBoundDeliveryReceiptV2]);
        let eligible = SignedNodeDescriptor::sign(eligible_body, &eligible_identity).unwrap();
        let candidates = vec![incompatible.clone(), eligible.clone()];
        let selected = select_chat_terminal_from_route_proven_candidates(
            candidates.clone(),
            NOW,
            [2; 32],
            Some(eligible.node_id()),
        )
        .unwrap();
        assert_eq!(selected.node_id(), eligible.node_id());
        assert!(matches!(
            select_chat_terminal_from_route_proven_candidates(
                candidates.clone(),
                NOW,
                [2; 32],
                Some(incompatible.node_id()),
            ),
            Err(ChatTerminalSelectionFailure::PinnedTerminalUnavailable)
        ));
        assert_eq!(
            select_chat_terminal_from_route_proven_candidates(candidates, NOW, [2; 32], None)
                .unwrap()
                .node_id(),
            eligible.node_id()
        );
    }
}
