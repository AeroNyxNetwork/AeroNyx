// [ARCH-SPLIT 2026-10-02]
// Mirror carrier selection, cold-bootstrap smoke, and recovery disposition.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn directory_carrier_recovery_disposition(
    reason: &str,
) -> DirectoryCarrierRecoveryDisposition {
    if directory_mirror_failure_allows_recovery(reason) {
        DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
    } else {
        DirectoryCarrierRecoveryDisposition::StopClosed
    }
}

/// Verifies that at least one explicit signed mirror carrier can return one
/// locally retained producer anchor and its exact descriptor objects.
///
/// No direct producer request is attempted. The returned evidence is audited
/// against the local retained mirror and then discarded without import.
pub(crate) async fn run_directory_mirror_carrier_smoke(
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: Option<&reqwest::Client>,
) -> DirectoryMirrorCarrierSmokeReport {
    let Some(replica_store) = replica_store else {
        return DirectoryMirrorCarrierSmokeReport::unavailable("replica_store_disabled");
    };
    let Some(client) = client else {
        return DirectoryMirrorCarrierSmokeReport::unavailable(
            "smoke_http_client_initialization_failed",
        );
    };
    let Ok(producers) = replica_store.mirror_producer_ids() else {
        return DirectoryMirrorCarrierSmokeReport::unavailable(
            "retained_mirror_registry_audit_failed",
        );
    };
    let mut report = DirectoryMirrorCarrierSmokeReport::pending();
    report.retained_producers = u64::try_from(producers.len()).unwrap_or(u64::MAX);
    if producers.is_empty() {
        report.failure_reason = Some("no_retained_mirror_producers");
        return report;
    }

    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let requester = identity.public_key_bytes();
    let mut last_failure = "no_retained_mirror_evidence";
    for producer in producers
        .into_iter()
        .take(DIRECTORY_MIRROR_CARRIER_SMOKE_MAX_PRODUCERS)
    {
        let tip = match replica_store.producer_tip(&producer) {
            Ok(tip) if tip.tip_height > 0 && !tip.quarantined => tip,
            Ok(_) => continue,
            Err(_) => {
                last_failure = "retained_mirror_tip_audit_failed";
                continue;
            }
        };
        report.eligible_retained_producers = report.eligible_retained_producers.saturating_add(1);
        let selection = directory_mirror_recovery_carriers_with_requirement(
            peer_store,
            &capability_cache,
            &producer,
            &requester,
            unix_now_secs(),
            true,
        );
        report.explicit_carrier_candidates = report
            .explicit_carrier_candidates
            .max(selection.explicitly_advertised_candidate_count);
        report.selected_routeable_carriers = report
            .selected_routeable_carriers
            .max(selection.selected_routeable_count);
        if selection.carriers.is_empty() {
            last_failure = "no_explicit_carrier_candidates";
            continue;
        }

        for carrier in selection.carriers {
            report.attempted_carriers = report.attempted_carriers.saturating_add(1);
            let context = DirectoryMirrorCarrierSmokeAttemptContext {
                replica_store: Arc::clone(&replica_store),
                peer_store,
                identity,
                client,
                producer,
                retained_tip_height: tip.tip_height,
                requester,
            };
            match verify_directory_mirror_carrier_smoke_candidate(&context, carrier).await {
                Ok((verified_blocks, verified_descriptor_objects)) => {
                    report.success = true;
                    report.status = "verified";
                    report.verified_blocks = verified_blocks;
                    report.verified_descriptor_objects = verified_descriptor_objects;
                    report.carrier_signature_verified = true;
                    report.producer_evidence_verified = true;
                    report.local_anchor_verified = true;
                    report.failure_reason = None;
                    return report;
                }
                Err((reason, carrier_signature_verified)) => {
                    report.carrier_signature_verified |= carrier_signature_verified;
                    last_failure = reason;
                }
            }
        }
    }
    report.failure_reason = Some(last_failure);
    report
}

/// Cold-bootstraps one operator-pinned producer through an explicit carrier.
///
/// The target producer is never contacted. Every producer attempt gets one new
/// SQLite `:memory:` store, starts at height one, imports up to three bounded
/// pages, rotates the first carrier between pages, and runs the complete
/// replica audit before the store is dropped. Only transport/availability
/// failures may try another carrier; evidence or import failures stop closed.
pub(crate) async fn run_directory_carrier_cold_bootstrap_smoke(
    configured_producers: &[[u8; 32]],
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: Option<&reqwest::Client>,
) -> DirectoryCarrierColdBootstrapSmokeReport {
    let configured_count = configured_producers.len();
    let Some(client) = client else {
        return DirectoryCarrierColdBootstrapSmokeReport::unavailable(
            configured_count,
            "smoke_http_client_initialization_failed",
        );
    };
    let requester = identity.public_key_bytes();
    let mut producers = configured_producers
        .iter()
        .copied()
        .filter(|producer| *producer != [0u8; 32] && *producer != requester)
        .collect::<Vec<_>>();
    producers.sort_unstable();
    producers.dedup();
    let mut report = DirectoryCarrierColdBootstrapSmokeReport::pending(configured_count);
    if producers.is_empty() {
        report.failure_reason = Some("no_configured_producers");
        return report;
    }

    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let mut last_failure = "no_cold_bootstrap_evidence";
    for producer in producers
        .into_iter()
        .take(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PRODUCERS)
    {
        report.eligible_producers = report.eligible_producers.saturating_add(1);
        let selection = directory_mirror_recovery_carriers_with_requirement(
            peer_store,
            &capability_cache,
            &producer,
            &requester,
            unix_now_secs(),
            true,
        );
        report.explicit_carrier_candidates = report
            .explicit_carrier_candidates
            .max(selection.explicitly_advertised_candidate_count);
        report.selected_routeable_carriers = report
            .selected_routeable_carriers
            .max(selection.selected_routeable_count);
        if selection.carriers.is_empty() {
            last_failure = "no_explicit_carrier_candidates";
            continue;
        }

        let local_node_id = requester;
        let isolated_store = match tokio::task::spawn_blocking(move || {
            DirectoryReplicaStore::open(":memory:", local_node_id, unix_now_secs())
                .map(|(store, _)| Arc::new(store))
        })
        .await
        {
            Ok(Ok(store)) => store,
            _ => {
                last_failure = "isolated_store_initialization_failed";
                continue;
            }
        };
        let carriers = selection.carriers;
        let mut pages_imported = 0u32;
        let mut requests_used = 0u32;
        let mut imported_blocks = 0u64;
        let mut imported_commitments = 0u64;
        let mut successful_carriers = HashSet::new();
        let mut last_outcome = None;
        let mut producer_attempt_failed = false;

        while pages_imported < DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES {
            let first_carrier = usize::try_from(pages_imported)
                .unwrap_or(0)
                .checked_rem(carriers.len())
                .unwrap_or(0);
            let mut page_outcome = None;

            for offset in 0..carriers.len() {
                if requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
                    > DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET
                {
                    last_failure = "smoke_request_budget_exhausted";
                    break;
                }
                let carrier = carriers[(first_carrier + offset) % carriers.len()];
                report.attempted_carriers = report.attempted_carriers.saturating_add(1);
                match pull_directory_chain_pinned_page_via_discovered_carrier(
                    Arc::clone(&isolated_store),
                    peer_store,
                    identity,
                    &producer,
                    carrier,
                    client,
                )
                .await
                {
                    Ok(outcome) => {
                        requests_used = requests_used.saturating_add(outcome.requests_made);
                        successful_carriers.insert(carrier.node_id);
                        page_outcome = Some(outcome);
                        break;
                    }
                    Err(failure)
                        if directory_carrier_recovery_disposition(&failure.reason)
                            == DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure =>
                    {
                        // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex]
                        // The tracked pull reports every range/object request
                        // consumed before failure. A complete page is still
                        // reserved before the next carrier is attempted.
                        requests_used = requests_used.saturating_add(failure.requests_made);
                        report.availability_failovers =
                            report.availability_failovers.saturating_add(1);
                        last_failure =
                            directory_mirror_carrier_smoke_failure_bucket(&failure.reason);
                    }
                    Err(failure) => {
                        requests_used = requests_used.saturating_add(failure.requests_made);
                        report.requests_used = report.requests_used.max(u64::from(requests_used));
                        report.failure_reason = Some(
                            directory_mirror_carrier_smoke_failure_bucket(&failure.reason),
                        );
                        return report;
                    }
                }
            }

            let Some(outcome) = page_outcome else {
                // [CARRIER-PARTIAL-PREFIX 2026-07-26 by Codex] A carrier may
                // retain only a bounded prefix or become unavailable after
                // serving earlier pages. Two or more imported pages already
                // prove multi-page third-party recovery once the isolated
                // store passes its full chain audit below. Availability may
                // stop extension, but must not erase verified evidence.
                producer_attempt_failed =
                    !directory_carrier_cold_bootstrap_prefix_ready(pages_imported);
                break;
            };
            pages_imported = pages_imported.saturating_add(1);
            imported_blocks = imported_blocks.saturating_add(outcome.import.blocks_inserted);
            imported_commitments =
                imported_commitments.saturating_add(outcome.import.commitments_inserted);
            report.carrier_signature_verified = true;
            last_outcome = Some(outcome);

            if !should_continue_directory_carrier_cold_bootstrap(
                pages_imported,
                requests_used,
                outcome.has_more,
            ) {
                break;
            }
        }

        report.pages_imported = report.pages_imported.max(u64::from(pages_imported));
        report.requests_used = report.requests_used.max(u64::from(requests_used));
        if producer_attempt_failed {
            continue;
        }
        let Some(last_outcome) = last_outcome else {
            last_failure = "no_cold_bootstrap_evidence";
            continue;
        };
        if !directory_carrier_cold_bootstrap_prefix_ready(pages_imported) {
            last_failure = "insufficient_multi_page_evidence";
            continue;
        }

        let verification_store = Arc::clone(&isolated_store);
        let verification = tokio::task::spawn_blocking(move || {
            let tip = verification_store.producer_tip(&producer)?;
            let audit = verification_store.audit(unix_now_secs())?;
            let mirrors = verification_store.mirror_producer_ids()?;
            if tip.tip_height == 0
                || tip.quarantined
                || audit.producers != 1
                || audit.mirror_producers != 0
                || audit.quarantined_producers != 0
                || audit.blocks != tip.tip_height
                || !mirrors.is_empty()
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "isolated cold-bootstrap audit contract mismatch".to_string(),
                ));
            }
            Ok((tip.tip_height, audit.blocks, audit.commitments))
        })
        .await;
        let Ok(Ok((tip_height, audited_blocks, audited_commitments))) = verification else {
            last_failure = "isolated_store_audit_failed";
            continue;
        };
        if imported_blocks == 0
            || last_outcome.import.tip_height != tip_height
            || audited_blocks != imported_blocks
            || audited_commitments != imported_commitments
        {
            last_failure = "isolated_store_import_contract_mismatch";
            continue;
        }

        report.success = true;
        report.status = "verified";
        report.distinct_successful_carriers =
            u64::try_from(successful_carriers.len()).unwrap_or(u64::MAX);
        report.imported_blocks = imported_blocks;
        report.imported_commitments = imported_commitments;
        report.bootstrapped_tip_height = tip_height;
        report.multi_page_prefix_verified = true;
        report.reached_observed_remote_tip = directory_sync_outcome_is_checkpoint_complete(
            &last_outcome,
            DirectoryMirrorPullSource::PublicCarrier,
        );
        report.producer_chain_verified = true;
        report.genesis_anchor_verified = true;
        report.isolated_store_audit_verified = true;
        report.failure_reason = None;
        return report;
    }
    report.failure_reason = Some(last_failure);
    report
}

pub(super) async fn verify_directory_mirror_carrier_smoke_candidate(
    context: &DirectoryMirrorCarrierSmokeAttemptContext<'_>,
    carrier: DirectoryMirrorRecoveryCarrier,
) -> Result<(u64, u64), (&'static str, bool)> {
    let request_timestamp = unix_now_secs();
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        context.peer_store,
        &carrier.node_id,
        carrier.descriptor_sequence,
        request_timestamp,
    )
    .map_err(|reason| {
        (
            directory_mirror_carrier_smoke_failure_bucket(&reason),
            false,
        )
    })?;
    let page = request_directory_replica_block_page(
        context.identity,
        &context.producer,
        &carrier.node_id,
        context.client,
        range_url,
        context.retained_tip_height,
        request_timestamp,
    )
    .await
    .map_err(|reason| {
        (
            directory_mirror_carrier_smoke_failure_bucket(&reason),
            false,
        )
    })?;
    let (objects, _) = hydrate_directory_replica_descriptor_objects(
        context.identity,
        &context.producer,
        &carrier.node_id,
        context.client,
        object_url,
        &context.requester,
        &page.blocks,
    )
    .await
    .map_err(|reason| (directory_mirror_carrier_smoke_failure_bucket(&reason), true))?;
    let DirectoryRangePage {
        blocks,
        has_more: _,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let store = Arc::clone(&context.replica_store);
    let producer = context.producer;
    let retained_tip_height = context.retained_tip_height;
    tokio::task::spawn_blocking(move || {
        store.verify_retained_carrier_page(
            producer,
            retained_tip_height,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| ("carrier_verification_task_failed", true))?
    .map_err(|_| ("carrier_evidence_rejected", true))
}

pub(super) fn directory_mirror_carrier_smoke_failure_bucket(reason: &str) -> &'static str {
    if reason.contains("_transport_failed")
        || reason.contains("_http_status_")
        || reason.contains("_peer_replica_")
        || reason.contains("_peer_mirror_")
        || reason.contains("_carrier_unavailable")
        || reason.contains("_carrier_descriptor_changed")
        || reason.contains("_carrier_not_public")
        || reason.contains("_carrier_missing_endpoint")
        || reason.contains("_carrier_unsafe_endpoint")
        || reason.contains("_carrier_invalid_endpoint")
    {
        "carrier_unavailable"
    } else if reason.contains("_response_")
        || reason.contains("_invalid_")
        || reason.contains("_hash_mismatch")
        || reason.contains("_noncanonical")
        || reason.contains("_contract_mismatch")
    {
        "carrier_evidence_rejected"
    } else {
        "carrier_request_failed"
    }
}

pub(super) fn directory_sync_failure_allows_carrier_fallback(reason: &str) -> bool {
    if matches!(
        reason,
        "pinned_directory_peer_unavailable"
            | "pinned_directory_peer_missing_endpoint"
            | "pinned_directory_peer_unsafe_endpoint"
            | "pinned_directory_peer_invalid_endpoint"
    ) || reason == "directory_range_transport_failed"
        || reason == "directory_replica_range_transport_failed"
        || reason == "directory_replica_objects_transport_failed"
        || reason == "directory_replica_range_peer_replica_not_found"
        || reason == "directory_replica_range_peer_replica_range_not_retained"
        || reason == "directory_replica_range_peer_mirror_replica_not_retained"
        || reason == "directory_replica_objects_peer_replica_object_not_found"
        || reason == "directory_replica_objects_peer_mirror_replica_not_retained"
    {
        return true;
    }
    for prefix in [
        "directory_range_http_status_",
        "directory_replica_range_http_status_",
        "directory_replica_objects_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 403 | 404 | 408 | 429) || status >= 500;
    }
    false
}

pub(super) fn directory_mirror_failure_allows_recovery(reason: &str) -> bool {
    if matches!(
        reason,
        "directory_mirror_peer_unavailable"
            | "directory_range_transport_failed"
            | "directory_objects_transport_failed"
            | "directory_replica_range_transport_failed"
            | "directory_replica_objects_transport_failed"
            | "directory_replica_range_peer_replica_not_found"
            | "directory_replica_range_peer_replica_range_not_retained"
            | "directory_replica_range_peer_mirror_replica_not_retained"
            | "directory_replica_objects_peer_replica_object_not_found"
            | "directory_replica_objects_peer_mirror_replica_not_retained"
            | "directory_mirror_recovery_carrier_unavailable"
            | "directory_mirror_recovery_carrier_descriptor_changed"
            | "directory_mirror_recovery_carrier_not_public"
            | "directory_mirror_recovery_carrier_missing_endpoint"
            | "directory_mirror_recovery_carrier_unsafe_endpoint"
            | "directory_mirror_recovery_carrier_invalid_endpoint"
    ) {
        return true;
    }
    for prefix in [
        "directory_range_http_status_",
        "directory_objects_http_status_",
        "directory_replica_range_http_status_",
        "directory_replica_objects_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 403 | 404 | 405 | 408 | 429) || status >= 500;
    }
    false
}

pub(super) fn directory_mirror_carrier_capability_unavailable(reason: &str) -> bool {
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Cache only explicit absence of
    // optional replica-carrier endpoints. A direct producer endpoint, generic
    // transport error, overload response, or invalid signed frame must not
    // suppress a future carrier attempt.
    for prefix in [
        "directory_replica_range_http_status_",
        "directory_replica_objects_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 404 | 405 | 501);
    }
    false
}

pub(super) fn directory_mirror_recovery_carriers(
    peer_store: &PeerStore,
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
) -> DirectoryMirrorRecoveryCarrierSelection {
    directory_mirror_recovery_carriers_with_requirement(
        peer_store,
        capability_cache,
        producer,
        requester,
        now,
        false,
    )
}

pub(super) fn directory_mirror_recovery_carriers_with_requirement(
    peer_store: &PeerStore,
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
    require_explicitly_advertised: bool,
) -> DirectoryMirrorRecoveryCarrierSelection {
    directory_mirror_recovery_carriers_with_policy(
        peer_store,
        capability_cache,
        producer,
        requester,
        now,
        require_explicitly_advertised,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn directory_mirror_recovery_carriers_with_policy(
    peer_store: &PeerStore,
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
    require_explicitly_advertised: bool,
    eligible_carriers: Option<&[[u8; 32]]>,
) -> DirectoryMirrorRecoveryCarrierSelection {
    let mut descriptors = peer_store
        .valid_public_descriptors(now, usize::MAX)
        .into_iter()
        .filter(|descriptor| {
            let node_id = descriptor.node_id();
            node_id != *producer
                && node_id != *requester
                && eligible_carriers
                    .is_none_or(|eligible| eligible.iter().any(|candidate| *candidate == node_id))
                && !peer_store.is_route_quarantined_now(&node_id, now)
                && descriptor.descriptor.policy.public_discovery
                && descriptor
                    .descriptor
                    .public_endpoint
                    .as_deref()
                    .is_some_and(commitment_peer_endpoint_is_public)
                && (!require_explicitly_advertised
                    || descriptor
                        .descriptor
                        .capabilities
                        .contains(&NodeCapability::DirectoryMirrorCarrier))
        })
        .collect::<Vec<_>>();
    descriptors.sort_by_key(SignedNodeDescriptor::node_id);
    descriptors.dedup_by_key(|descriptor| descriptor.node_id());
    if descriptors.is_empty() {
        return DirectoryMirrorRecoveryCarrierSelection::default();
    }

    let producer_seed = u64::from_be_bytes(producer[..8].try_into().unwrap_or([0u8; 8]));
    let requester_seed = u64::from_be_bytes(requester[..8].try_into().unwrap_or([0u8; 8]));
    let epoch_seed = now / DIRECTORY_MIRROR_RECOVERY_ROTATION_SECS;
    let cursor = usize::try_from(producer_seed ^ requester_seed ^ epoch_seed).unwrap_or(0)
        % descriptors.len();
    descriptors.rotate_left(cursor);

    // [MIRROR-DIVERSITY 2026-07-24 by Codex] Routeability is local observed
    // evidence and therefore outranks self-reported descriptor metadata.
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Within that local-evidence tier,
    // signed carrier capability outranks separately measured unadvertised
    // compatibility fallback.
    // Freshness is bucketed to avoid permanent affinity to tiny timestamp
    // differences; deterministic rotation remains the tie-breaker.
    let candidate_count = u64::try_from(descriptors.len()).unwrap_or(u64::MAX);
    let routeable_candidate_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| peer_store.is_routeable_now(&descriptor.node_id(), now))
            .count(),
    )
    .unwrap_or(u64::MAX);
    let explicitly_advertised_candidate_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| {
                descriptor
                    .descriptor
                    .capabilities
                    .contains(&NodeCapability::DirectoryMirrorCarrier)
            })
            .count(),
    )
    .unwrap_or(u64::MAX);
    let unadvertised_compatibility_candidate_count =
        candidate_count.saturating_sub(explicitly_advertised_candidate_count);
    let capability_cached_unavailable_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| {
                !capability_cache.should_attempt(&descriptor.node_id(), descriptor.sequence())
            })
            .count(),
    )
    .unwrap_or(u64::MAX);

    let mut candidates = descriptors
        .into_iter()
        .filter(|descriptor| {
            capability_cache.should_attempt(&descriptor.node_id(), descriptor.sequence())
        })
        .enumerate()
        .map(|(rotation_rank, descriptor)| {
            let issued_at = descriptor.descriptor.issued_at;
            let age = now.checked_sub(issued_at);
            let freshness_rank = match age {
                Some(age) if age <= DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS => 0,
                Some(age) if age <= DIRECTORY_MIRROR_RECOVERY_AGING_DESCRIPTOR_SECS => 1,
                Some(_) => 2,
                None => 3,
            };
            let signed_region_hint = descriptor
                .descriptor
                .policy
                .region
                .as_deref()
                .map(str::trim)
                .filter(|region| !region.is_empty())
                .map(str::to_ascii_lowercase);
            let explicitly_advertised = descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::DirectoryMirrorCarrier);
            let node_id = descriptor.node_id();
            DirectoryMirrorRecoveryCarrierCandidate {
                node_id,
                descriptor_sequence: descriptor.sequence(),
                explicitly_advertised,
                routeable: peer_store.is_routeable_now(&node_id, now),
                freshness_rank,
                rotation_rank,
                signed_region_hint,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        let (routeability_rank, capability_rank, freshness_rank) = candidate.availability_tier();
        (
            routeability_rank,
            capability_rank,
            freshness_rank,
            candidate.rotation_rank,
        )
    });

    let mut selected = Vec::with_capacity(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE);
    let mut selected_regions = HashSet::new();
    while !candidates.is_empty() && selected.len() < DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE
    {
        let best_tier = candidates[0].availability_tier();
        let position = candidates
            .iter()
            .position(|candidate| {
                candidate.availability_tier() == best_tier
                    && candidate
                        .signed_region_hint
                        .as_ref()
                        .is_some_and(|region| !selected_regions.contains(region))
            })
            .or_else(|| {
                candidates
                    .iter()
                    .position(|candidate| candidate.availability_tier() == best_tier)
            })
            .unwrap_or(0);
        let candidate = candidates.remove(position);
        if let Some(region) = candidate.signed_region_hint.as_ref() {
            selected_regions.insert(region.clone());
        }
        selected.push(candidate);
    }

    DirectoryMirrorRecoveryCarrierSelection {
        carriers: selected
            .iter()
            .map(|candidate| DirectoryMirrorRecoveryCarrier {
                node_id: candidate.node_id,
                descriptor_sequence: candidate.descriptor_sequence,
            })
            .collect(),
        candidate_count,
        routeable_candidate_count,
        explicitly_advertised_candidate_count,
        unadvertised_compatibility_candidate_count,
        capability_cached_unavailable_count,
        selected_routeable_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.routeable)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_explicitly_advertised_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.explicitly_advertised)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_unadvertised_compatibility_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| !candidate.explicitly_advertised)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_region_hint_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.signed_region_hint.is_some())
                .count(),
        )
        .unwrap_or(u64::MAX),
        distinct_selected_region_hint_count: u64::try_from(selected_regions.len())
            .unwrap_or(u64::MAX),
    }
}
