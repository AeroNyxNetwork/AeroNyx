// [ARCH-SPLIT 2026-10-02]
// Verified-delivery anchor witness round.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Sends one signed aggregate-only cache anchor to operator-pinned witnesses.
///
/// Witnesses are discovery-admitted and endpoint-pinned on every request. The
/// returned counters are operational evidence only; they are not consensus,
/// finality, voting weight, or a source of user-visible delivery statistics.
///
/// # Errors
///
/// Returns an error only when the local anchor input is structurally invalid.
/// Individual witness failures are counted in the bounded round result.
pub async fn witness_verified_delivery_anchor(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    generation: u64,
    anchor_digest: &[u8; 32],
) -> Result<VerifiedDeliveryAnchorWitnessRound, String> {
    witness_verified_delivery_anchor_with_endpoint_policy(
        peer_store,
        identity,
        client,
        witness_node_ids,
        generation,
        anchor_digest,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn witness_verified_delivery_anchor_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    generation: u64,
    anchor_digest: &[u8; 32],
    endpoint_allowed: &F,
) -> Result<VerifiedDeliveryAnchorWitnessRound, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if generation == 0 || generation > i64::MAX as u64 {
        return Err("verified_delivery_witness_generation_invalid".to_string());
    }
    if anchor_digest == &[0u8; 32] {
        return Err("verified_delivery_witness_digest_invalid".to_string());
    }

    let self_node_id = identity.public_key_bytes();
    let mut distinct = HashSet::new();
    let witnesses = witness_node_ids
        .iter()
        .copied()
        .filter(|node_id| *node_id != self_node_id && distinct.insert(*node_id))
        .take(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>();
    let mut round = VerifiedDeliveryAnchorWitnessRound {
        configured: witnesses.len(),
        ..VerifiedDeliveryAnchorWitnessRound::default()
    };

    for witness_node_id in witnesses {
        let request_timestamp = now_secs();
        let Some(peer) = peer_store.get_valid(&witness_node_id, request_timestamp) else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };
        let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };
        if !endpoint_allowed(endpoint) {
            round.failed = round.failed.saturating_add(1);
            continue;
        }
        let Ok(url) = verified_delivery_anchor_witness_url(endpoint) else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };

        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
            &self_node_id,
            generation,
            anchor_digest,
            &request_id,
            request_timestamp,
        );
        let request = MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
            requester: self_node_id,
            generation,
            anchor_digest: *anchor_digest,
            request_id,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = match encode_memchain(&request) {
            Ok(frame) => frame,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        round.attempted = round.attempted.saturating_add(1);
        let response = match client
            .post(url)
            .header("content-type", "application/octet-stream")
            .body(frame)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => response,
            Ok(_) | Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        let body = match read_bounded_response(response).await {
            Ok(body) => body,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        let outcome = match control_plane::verify_delivery_anchor_witness_response(
            &body,
            &request_id,
            &self_node_id,
            generation,
            anchor_digest,
            &witness_node_id,
            now_secs(),
        ) {
            Ok(outcome) => outcome,
            Err(_) => {
                round.failed = round.failed.saturating_add(1);
                continue;
            }
        };
        round.verified = round.verified.saturating_add(1);
        match outcome {
            VERIFIED_DELIVERY_WITNESS_ADVANCED_V1 => {
                round.advanced = round.advanced.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1 => {
                round.idempotent = round.idempotent.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_STALE_V1 => round.stale = round.stale.saturating_add(1),
            VERIFIED_DELIVERY_WITNESS_CONFLICT_V1 => {
                round.conflicts = round.conflicts.saturating_add(1)
            }
            VERIFIED_DELIVERY_WITNESS_GAP_V1 => round.gaps = round.gaps.saturating_add(1),
            _ => unreachable!("verified response outcome was validated"),
        }
    }
    Ok(round)
}
