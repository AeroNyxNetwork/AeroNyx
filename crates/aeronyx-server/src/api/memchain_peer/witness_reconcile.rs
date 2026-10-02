// [ARCH-SPLIT 2026-10-02]
// Pinned witness reconciliation.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Collects a bounded set of signed checkpoint observations from discovered
/// encrypted-storage peers.
///
/// This is evidence collection for the current single-writer Block Sync v1
/// architecture, not distributed consensus. The function never adopts a
/// remote chain, changes the coordinator, selects a longest chain, or derives
/// truth from peer count. Every accepted response is independently verified
/// and durably stored by `pull_record_commitment_checkpoint`.
pub async fn reconcile_record_commitment_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        max_witnesses,
        commitment_peer_endpoint_is_public,
    )
    .await
}

/// Collects checkpoints only from explicit operator-pinned identities.
///
/// This is the trust boundary used by the coordinator startup guard. Signed
/// discovery still resolves endpoint rotation, but no unpinned permissionless
/// peer can become startup authority merely by advertising a capability.
pub async fn reconcile_record_commitment_pinned_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        2,
    )
    .await
}

/// Collects pinned witness proofs and attempts one immutable certificate using
/// the operator's configured minimum. Values below two preserve legacy
/// one-witness startup behavior but cannot be represented as a multi-witness
/// certificate.
pub async fn reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        minimum_certificate_signers,
        commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn reconcile_record_commitment_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidates: Vec<_> = peer_store
        .peers_with_capability(NodeCapability::EncryptedStorage, now)
        .into_iter()
        .filter(|candidate| candidate.descriptor.node_id != self_node_id)
        .filter(|candidate| {
            candidate
                .descriptor
                .public_endpoint
                .as_deref()
                .is_some_and(&endpoint_allowed)
        })
        .collect();
    candidates.sort_by_key(|candidate| candidate.descriptor.node_id);
    if !candidates.is_empty() {
        // Rotate the deterministic signed-descriptor order so a larger network
        // does not permanently starve peers beyond the per-round fan-out cap.
        // The selector is local scheduling state only and is never reported.
        let node_selector = u64::from_be_bytes(
            self_node_id[..8]
                .try_into()
                .expect("fixed identity prefix length"),
        );
        let offset =
            usize::try_from((node_selector ^ (now / 300)) % candidates.len() as u64).unwrap_or(0);
        candidates.rotate_left(offset);
    }

    let candidate_ids = candidates
        .into_iter()
        .map(|candidate| candidate.descriptor.node_id)
        .collect();
    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        max_witnesses,
        false,
        None,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &endpoint_allowed,
    )
    .await
}

pub(crate) async fn reconcile_record_commitment_pinned_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidate_ids = Vec::with_capacity(witness_node_ids.len());
    for node_id in witness_node_ids {
        if *node_id == self_node_id || candidate_ids.contains(node_id) {
            continue;
        }
        let Some(peer) = commitment_peer_descriptor(
            peer_store,
            node_id,
            now,
            CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        ) else {
            continue;
        };
        if peer
            .descriptor
            .public_endpoint
            .as_deref()
            .is_some_and(&endpoint_allowed)
        {
            candidate_ids.push(*node_id);
        }
    }

    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        witness_node_ids.len().min(MAX_PINNED_WITNESSES_PER_ROUND),
        true,
        Some(minimum_certificate_signers),
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
        &endpoint_allowed,
    )
    .await
}

pub(super) async fn reconcile_record_commitment_candidate_ids<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    mut candidate_ids: Vec<[u8; 32]>,
    max_witnesses: usize,
    track_trusted_witness_incidents: bool,
    minimum_certificate_signers: Option<usize>,
    descriptor_policy: CommitmentPeerDescriptorPolicy,
    endpoint_allowed: &F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    // Preserve operator order while ensuring one identity can contribute at
    // most one request, one result, and one certificate member. This remains
    // defense in depth even when config parsing already rejects duplicate IDs.
    let mut distinct_candidates = HashSet::with_capacity(candidate_ids.len());
    candidate_ids.retain(|candidate| distinct_candidates.insert(*candidate));
    let eligible_witnesses = candidate_ids.len();
    candidate_ids.truncate(max_witnesses);
    let attempted = candidate_ids.len();
    let mut outcome = CommitmentReconciliationOutcome {
        eligible_witnesses,
        attempted,
        ..CommitmentReconciliationOutcome::default()
    };
    let certificate_witnesses = candidate_ids.clone();
    let mut verified = Vec::with_capacity(attempted);
    let mut certificate_evidence = Vec::with_capacity(attempted);

    for candidate_node_id in candidate_ids {
        match pull_record_commitment_checkpoint_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &candidate_node_id,
            client,
            track_trusted_witness_incidents,
            descriptor_policy,
            endpoint_allowed,
        )
        .await
        {
            Ok(proof) => {
                outcome.verified = outcome.verified.saturating_add(1);
                match proof.relation {
                    CommitmentCheckpointRelation::Converged => {
                        outcome.converged = outcome.converged.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteAhead => {
                        outcome.remote_ahead = outcome.remote_ahead.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteBehind => {
                        outcome.remote_behind = outcome.remote_behind.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::Diverged => {
                        outcome.diverged = outcome.diverged.saturating_add(1);
                    }
                }
                if matches!(
                    proof.relation,
                    CommitmentCheckpointRelation::Converged
                        | CommitmentCheckpointRelation::RemoteAhead
                ) && proof.checkpoint_height == proof.local_tip_height
                {
                    certificate_evidence.push(proof.evidence_digest);
                }
                verified.push(proof);
            }
            Err(_) => {
                outcome.failed = outcome.failed.saturating_add(1);
            }
        }
    }

    let completed_at = now_secs();
    if let Some(configured_threshold) = minimum_certificate_signers {
        let threshold = configured_threshold
            .max(2)
            .min(MAX_PINNED_WITNESSES_PER_ROUND);
        outcome.certificate_signers = certificate_evidence.len();
        outcome.certificate_required_signers = threshold;
        if certificate_evidence.len() >= threshold {
            match storage
                .persist_record_commitment_checkpoint_certificate(
                    completed_at,
                    threshold,
                    &certificate_witnesses,
                    &certificate_evidence,
                )
                .await
            {
                Ok(persisted) => outcome.certificate_persisted = persisted,
                Err(_) => outcome.certificate_persistence_failed = true,
            }
        }
    }
    for _ in 0..outcome.failed {
        storage.record_commitment_checkpoint_failure(completed_at);
    }
    // Record valid proofs after failures and from least to most severe. A
    // partial transport failure must not hide valid evidence, while a signed
    // divergence must remain the final operator-visible state for the round.
    verified.sort_by_key(|proof| checkpoint_relation_priority(proof.relation));
    for proof in verified {
        storage.record_commitment_checkpoint_verified(
            completed_at,
            proof.relation.as_str(),
            proof.local_tip_height,
            proof.remote_tip_height,
        );
    }
    storage.record_commitment_checkpoint_witness_round(
        completed_at,
        outcome.eligible_witnesses,
        outcome.attempted,
        outcome.verified,
        outcome.failed,
        outcome.converged,
        outcome.remote_ahead,
        outcome.remote_behind,
        outcome.diverged,
    );

    outcome
}

pub(super) const fn checkpoint_relation_priority(relation: CommitmentCheckpointRelation) -> u8 {
    match relation {
        CommitmentCheckpointRelation::RemoteBehind => 0,
        CommitmentCheckpointRelation::Converged => 1,
        CommitmentCheckpointRelation::RemoteAhead => 2,
        CommitmentCheckpointRelation::Diverged => 3,
    }
}
