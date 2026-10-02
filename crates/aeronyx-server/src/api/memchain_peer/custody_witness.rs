// [ARCH-SPLIT 2026-10-02]
// Custody-audit witness planning and round collection.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Builds a bounded local eligibility plan for custody-audit witnesses.
///
/// This function performs no network I/O and does not construct an anchor.
/// It evaluates only operator pins against the current authenticated PeerStore
/// view and the production public-endpoint policy.
///
/// # Errors
///
/// Returns a stable error when the local threshold is zero, exceeds the hard
/// witness bound, or the caller bypasses configuration validation with too
/// many pins.
pub fn plan_custody_audit_witnesses(
    peer_store: &PeerStore,
    producer_node_id: &[u8; 32],
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    now: u64,
) -> Result<CustodyAuditWitnessPlan, String> {
    plan_custody_audit_witnesses_with_endpoint_policy(
        peer_store,
        producer_node_id,
        witness_node_ids,
        minimum_verified,
        now,
        &commitment_peer_endpoint_is_public,
    )
}

pub(super) fn plan_custody_audit_witnesses_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    producer_node_id: &[u8; 32],
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    now: u64,
    endpoint_allowed: &F,
) -> Result<CustodyAuditWitnessPlan, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if minimum_verified == 0 || minimum_verified > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_minimum_invalid".to_string());
    }
    if witness_node_ids.len() > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_pin_limit_exceeded".to_string());
    }

    let mut plan = CustodyAuditWitnessPlan {
        minimum_verified,
        ..CustodyAuditWitnessPlan::default()
    };
    let mut distinct = HashSet::with_capacity(witness_node_ids.len());
    for witness_node_id in witness_node_ids {
        if witness_node_id == producer_node_id {
            plan.self_excluded = plan.self_excluded.saturating_add(1);
            continue;
        }
        if !distinct.insert(*witness_node_id) {
            plan.duplicates_ignored = plan.duplicates_ignored.saturating_add(1);
            continue;
        }
        plan.configured = plan.configured.saturating_add(1);

        let eligible = peer_store
            .get_valid(witness_node_id, now)
            .is_some_and(|peer| {
                peer.descriptor
                    .capabilities
                    .contains(&NodeCapability::EncryptedStorage)
                    && peer
                        .descriptor
                        .public_endpoint
                        .as_deref()
                        .is_some_and(|endpoint| {
                            endpoint_allowed(endpoint)
                                && custody_audit_anchor_witness_url(endpoint).is_ok()
                        })
            });
        if eligible {
            plan.eligible = plan.eligible.saturating_add(1);
        } else {
            plan.unavailable = plan.unavailable.saturating_add(1);
        }
    }
    plan.quorum_ready = plan.configured >= minimum_verified && plan.eligible >= minimum_verified;
    Ok(plan)
}

/// Sends one producer-signed custody anchor to one exact caller-pinned witness.
///
/// This explicit primitive is not called by node startup or a background
/// scheduler. The configured witness list therefore remains non-transmitting
/// until a later rollout deliberately invokes this function. It never selects
/// an arbitrary discovery peer or falls back to a different witness identity.
///
/// # Errors
///
/// Returns a stable privacy-safe error when the anchor is invalid, the exact
/// witness is unavailable or ineligible, transport fails, or either response
/// signature does not bind the expected request, producer, witness, and anchor.
pub async fn witness_custody_audit_anchor(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_id: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
) -> Result<CustodyAuditWitnessReceiptV1, String> {
    witness_custody_audit_anchor_with_endpoint_policy(
        peer_store,
        identity,
        client,
        witness_node_id,
        anchor,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn witness_custody_audit_anchor_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_id: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
    endpoint_allowed: &F,
) -> Result<CustodyAuditWitnessReceiptV1, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let producer_node_id = identity.public_key_bytes();
    if witness_node_id == &producer_node_id {
        return Err("custody_witness_independence_required".to_string());
    }
    let anchor_frame_sha256 = validate_custody_anchor_for_producer(anchor, &producer_node_id)?;
    let request_timestamp = now_secs();
    let peer = peer_store
        .get_valid(witness_node_id, request_timestamp)
        .ok_or_else(|| "custody_witness_unavailable".to_string())?;
    if !peer
        .descriptor
        .capabilities
        .contains(&NodeCapability::EncryptedStorage)
    {
        return Err("custody_witness_capability_missing".to_string());
    }
    let endpoint = peer
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "custody_witness_endpoint_missing".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("custody_witness_endpoint_unsafe".to_string());
    }
    let url = custody_audit_anchor_witness_url(endpoint)?;

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let signing_bytes = custody_audit_anchor_witness_request_signing_bytes(
        &request_id,
        &producer_node_id,
        request_timestamp,
        &anchor_frame_sha256,
    );
    let request = MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id,
        requester: producer_node_id,
        request_timestamp,
        anchor: anchor.clone(),
        signature: identity.sign(&signing_bytes),
    };
    let frame =
        encode_memchain(&request).map_err(|_| "custody_witness_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("custody_witness", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "custody_witness_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_http_response(response, MAX_CUSTODY_WITNESS_RESPONSE_BYTES)
        .await
        .map_err(|_| "custody_witness_response_body_invalid".to_string())?;
    control_plane::verify_custody_audit_anchor_witness_response(
        &body,
        &request_id,
        &producer_node_id,
        witness_node_id,
        anchor,
        &anchor_frame_sha256,
        now_secs(),
    )
}

/// Sends one custody anchor to a hard-bounded set of exact witness pins.
///
/// [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] The return value is
/// aggregate-only diagnostic evidence. This non-persisting variant must never
/// back a startup/runtime safety gate; gates must use
/// [`witness_custody_audit_anchor_round_durable`]. No scheduler invokes either
/// primitive in this release.
///
/// # Errors
///
/// Returns an error only when the local anchor or threshold policy is invalid.
/// Individual witness failures remain bounded aggregate counters.
pub async fn witness_custody_audit_anchor_round(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    anchor: &CustodyAuditAnchorV1,
) -> Result<CustodyAuditWitnessRound, String> {
    witness_custody_audit_anchor_round_with_endpoint_policy(
        peer_store,
        identity,
        client,
        witness_node_ids,
        minimum_verified,
        anchor,
        None,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Sends one custody anchor and durably retains every verified signed receipt.
///
/// [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] This is the only round
/// suitable for a future startup/runtime safety gate: a receipt contributes to
/// the aggregate result only after atomic producer-side persistence and full
/// vault re-audit. No scheduler invokes this primitive in this release.
///
/// # Errors
///
/// Returns an error when local policy/anchor validation fails or a verified
/// receipt cannot be durably retained. Individual transport failures remain
/// bounded aggregate counters.
pub async fn witness_custody_audit_anchor_round_durable(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    anchor: &CustodyAuditAnchorV1,
) -> Result<CustodyAuditWitnessRound, String> {
    witness_custody_audit_anchor_round_with_endpoint_policy(
        peer_store,
        identity,
        client,
        witness_node_ids,
        minimum_verified,
        anchor,
        Some(storage),
        &commitment_peer_endpoint_is_public,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn witness_custody_audit_anchor_round_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    anchor: &CustodyAuditAnchorV1,
    receipt_storage: Option<&MemoryStorage>,
    endpoint_allowed: &F,
) -> Result<CustodyAuditWitnessRound, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if minimum_verified == 0 || minimum_verified > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_minimum_invalid".to_string());
    }
    if witness_node_ids.len() > MAX_PINNED_WITNESSES_PER_ROUND {
        return Err("custody_witness_pin_limit_exceeded".to_string());
    }
    let producer_node_id = identity.public_key_bytes();
    let anchor_frame_sha256 = validate_custody_anchor_for_producer(anchor, &producer_node_id)?;

    let mut round = CustodyAuditWitnessRound {
        minimum_verified,
        ..CustodyAuditWitnessRound::default()
    };
    let mut distinct = HashSet::with_capacity(witness_node_ids.len());
    let mut witnesses = Vec::with_capacity(witness_node_ids.len());
    for witness_node_id in witness_node_ids {
        if witness_node_id == &producer_node_id {
            round.self_excluded = round.self_excluded.saturating_add(1);
            continue;
        }
        if !distinct.insert(*witness_node_id) {
            round.duplicates_ignored = round.duplicates_ignored.saturating_add(1);
            continue;
        }
        round.configured = round.configured.saturating_add(1);
        witnesses.push(*witness_node_id);
    }

    // [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] Every future is
    // tied to one already de-duplicated operator pin, and the stream cannot
    // exceed the protocol's fixed witness fan-out. Persistence remains inside
    // the future: a receipt is never returned to the aggregate counter until
    // its durable write succeeds.
    let deliveries = futures::stream::iter(witnesses)
        .map(|witness_node_id| async move {
            let receipt = match witness_custody_audit_anchor_with_endpoint_policy(
                peer_store,
                identity,
                client,
                &witness_node_id,
                anchor,
                endpoint_allowed,
            )
            .await
            {
                Ok(receipt) => receipt,
                Err(_) => {
                    return Ok::<Option<CustodyAuditWitnessReceiptV1>, String>(None);
                }
            };
            if let Some(storage) = receipt_storage {
                storage
                    .persist_custody_audit_witness_receipt(
                        &receipt,
                        &producer_node_id,
                        anchor.checkpoint_generation,
                        &anchor_frame_sha256,
                        now_secs(),
                    )
                    .await
                    .map_err(|_| "custody_witness_receipt_persist_failed".to_string())?;
            }
            Ok(Some(receipt))
        })
        .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>()
        .await;

    for delivery in deliveries {
        let Some(receipt) = delivery? else {
            round.failed = round.failed.saturating_add(1);
            continue;
        };
        round.verified = round.verified.saturating_add(1);
        match receipt.outcome {
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1 => {
                round.advanced = round.advanced.saturating_add(1);
                round.accepted = round.accepted.saturating_add(1);
            }
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1 => {
                round.idempotent = round.idempotent.saturating_add(1);
                round.accepted = round.accepted.saturating_add(1);
            }
            CUSTODY_AUDIT_WITNESS_STALE_V1 => {
                round.stale = round.stale.saturating_add(1);
            }
            CUSTODY_AUDIT_WITNESS_CONFLICT_V1 => {
                round.conflicts = round.conflicts.saturating_add(1);
            }
            CUSTODY_AUDIT_WITNESS_GAP_V1 => {
                round.gaps = round.gaps.saturating_add(1);
            }
            _ => unreachable!("verified custody receipt outcome was validated"),
        }
    }
    round.adverse_evidence = round.stale > 0 || round.conflicts > 0 || round.gaps > 0;
    round.quorum_satisfied = round.accepted >= minimum_verified && !round.adverse_evidence;
    Ok(round)
}

pub(super) fn validate_custody_anchor_for_producer(
    anchor: &CustodyAuditAnchorV1,
    producer_node_id: &[u8; 32],
) -> Result<[u8; 32], String> {
    // [CUSTODY-WITNESS-TRANSPORT 2026-08-16 by Codex] Validate the nested
    // producer signature before endpoint selection or network I/O. A caller
    // cannot turn the node into an oracle for an unrelated producer anchor.
    if anchor.checkpoint_generation > i64::MAX as u64 {
        return Err("custody_witness_anchor_invalid".to_string());
    }
    anchor
        .verify_expected(producer_node_id, anchor.checkpoint_generation)
        .map_err(|_| "custody_witness_anchor_invalid".to_string())?;
    match custody_audit_anchor_frame_sha256(anchor) {
        Ok(digest) if digest != [0u8; 32] => Ok(digest),
        Ok(_) | Err(_) => Err("custody_witness_anchor_invalid".to_string()),
    }
}
