// [ARCH-SPLIT 2026-10-02]
// Publish the local descriptor to pinned commitment witnesses.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

// [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] A follower may use an
// authentic expired cache descriptor only for its explicitly pinned
// coordinator's signed checkpoint/lease control traffic. The request timestamp
// and payload signature are still verified by each handler. This breaks the
// reverse half of the cold-start deadlock without admitting stale
// permissionless peers to block sync, routing, gossip, or public membership.
pub(super) fn coordinator_control_requester_is_admitted(
    state: &MemChainPeerState,
    requester: &[u8; 32],
    now: u64,
) -> bool {
    state.peer_store.get_valid(requester, now).is_some()
        || (state.lease_authorized_coordinator == Some(*requester)
            && state
                .peer_store
                .get_signature_verified_cached(requester)
                .is_some())
}

pub(super) fn commitment_peer_descriptor(
    peer_store: &PeerStore,
    node_id: &[u8; 32],
    now: u64,
    policy: CommitmentPeerDescriptorPolicy,
) -> Option<SignedNodeDescriptor> {
    peer_store.get_valid(node_id, now).or_else(|| {
        (policy == CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness)
            .then(|| peer_store.get_signature_verified_cached(node_id))
            .flatten()
    })
}

/// Republishes this coordinator's current signed descriptor to pinned witnesses.
///
/// This bounded compatibility preflight runs before strict startup checkpoint
/// and coordinator-lease gates. It may use an authentic expired descriptor as
/// a transport hint for an exact operator-pinned witness, but it sends only the
/// coordinator's public signed discovery descriptor. Witness acceptance grants
/// no authority and cannot bypass the subsequent signed protocol gates.
pub async fn publish_current_descriptor_to_commitment_witnesses(
    peer_store: &PeerStore,
    self_descriptor: &SignedNodeDescriptor,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
) -> CommitmentWitnessDescriptorPublishRound {
    publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
        peer_store,
        self_descriptor,
        client,
        witness_node_ids,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    self_descriptor: &SignedNodeDescriptor,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    endpoint_allowed: &F,
) -> CommitmentWitnessDescriptorPublishRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    // [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] Preserve first-seen
    // operator order while deduplicating and enforcing the existing protocol
    // fan-out cap. A malformed configuration cannot amplify startup traffic.
    let mut distinct_witnesses = Vec::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    let mut seen = HashSet::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    let self_node_id = self_descriptor.node_id();
    for witness in witness_node_ids {
        if *witness == self_node_id || !seen.insert(*witness) {
            continue;
        }
        distinct_witnesses.push(*witness);
        if distinct_witnesses.len() == MAX_PINNED_WITNESSES_PER_ROUND {
            break;
        }
    }

    let now = now_secs();
    let configured = distinct_witnesses.len();
    let mut urls = Vec::with_capacity(configured);
    for witness in distinct_witnesses {
        let Some(descriptor) = commitment_peer_descriptor(
            peer_store,
            &witness,
            now,
            CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        ) else {
            continue;
        };
        let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
            continue;
        };
        if !endpoint_allowed(endpoint) {
            continue;
        }
        let Ok(url) = canonical_peer_http_url(endpoint, "/api/discovery/gossip") else {
            continue;
        };
        urls.push(url);
    }

    let attempted = urls.len();
    let accepted = futures::stream::iter(urls)
        .map(|url| {
            let message = NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: self_descriptor.clone(),
            };
            async move {
                let Ok(response) = client.post(url).json(&message).send().await else {
                    return false;
                };
                if !response.status().is_success() {
                    return false;
                }
                let Ok(body) =
                    read_bounded_http_response(response, MAX_DESCRIPTOR_PREFLIGHT_RESPONSE_BYTES)
                        .await
                else {
                    return false;
                };
                let Ok(receipt) = serde_json::from_slice::<GossipResponse>(&body) else {
                    return false;
                };
                receipt.applied.total == 1
                    && receipt.applied.stale == 0
                    && receipt.applied.rejected == 0
                    && receipt
                        .applied
                        .inserted
                        .saturating_add(receipt.applied.unchanged)
                        == 1
            }
        })
        .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
        .filter(|accepted| std::future::ready(*accepted))
        .count()
        .await;

    CommitmentWitnessDescriptorPublishRound {
        configured,
        attempted,
        accepted,
        failed: configured.saturating_sub(accepted),
    }
}
