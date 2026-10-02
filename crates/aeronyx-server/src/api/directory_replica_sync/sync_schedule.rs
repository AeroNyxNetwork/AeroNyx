// [ARCH-SPLIT 2026-10-02]
// Catch-up predicates, round delay, and failure backoff used by the coordinator.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn directory_sync_outcome_is_checkpoint_complete(
    outcome: &DirectorySyncPullOutcome,
    source: DirectoryMirrorPullSource,
) -> bool {
    // [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] A carrier can prove
    // producer-signed blocks in a verified prefix, but its reported terminal
    // tip cannot establish that the producer has no later block.
    source.authenticates_producer_tip()
        && !outcome.has_more
        && outcome.import.tip_height == outcome.remote_tip_height
        && outcome.import.tip_hash == outcome.remote_tip_hash
}

pub(super) fn directory_full_node_mirror_candidates(
    retained: &[DirectoryRetainedMirrorCursor],
    mut live_candidates: Vec<([u8; 32], u64)>,
    max_producers: usize,
) -> Vec<([u8; 32], u64)> {
    // [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] Retained registry
    // cursors are eligible availability work even when the producer's public
    // descriptor has expired. They do not bypass discovery for endpoints: a
    // direct attempt still requires a current exact descriptor, and carrier
    // recovery still selects only current admitted carriers.
    let retained_set = retained
        .iter()
        .map(|cursor| cursor.producer)
        .collect::<HashSet<_>>();
    let mut candidates = retained
        .iter()
        .map(|cursor| (cursor.producer, cursor.descriptor_sequence))
        .collect::<HashMap<_, _>>();
    live_candidates.sort_unstable_by_key(|(producer, _)| *producer);
    for (producer, descriptor_sequence) in live_candidates {
        if let Some(retained_sequence) = candidates.get_mut(&producer) {
            *retained_sequence = (*retained_sequence).max(descriptor_sequence);
        } else if candidates.len() < max_producers {
            candidates.insert(producer, descriptor_sequence);
        }
    }
    let mut candidates = candidates.into_iter().collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(producer, _)| (!retained_set.contains(producer), *producer));
    candidates
}

/// Whether another page can be requested without violating the conservative
/// worst-case request budget.
#[must_use]
pub(crate) const fn should_continue_directory_replica_catch_up(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_SYNC_MAX_PAGES_PER_ROUND
        && requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
}

/// Whether the isolated carrier cold-bootstrap smoke may request another page.
///
/// [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Reserve a complete
/// worst-case page before continuing. The attempt loop applies the same check
/// before every carrier, so an availability failure can never overrun the
/// operator smoke budget.
pub(super) const fn should_continue_directory_carrier_cold_bootstrap(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES
        && requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET
}

/// Whether the isolated store contains enough pages to prove multi-page
/// third-party recovery after a later availability-only failure.
pub(super) const fn directory_carrier_cold_bootstrap_prefix_ready(pages_completed: u32) -> bool {
    pages_completed >= 2
}

pub(super) const fn directory_sync_next_round_delay(
    configured_interval: Duration,
    all_producers_synchronized: bool,
) -> Duration {
    if all_producers_synchronized {
        configured_interval
    } else {
        let catch_up_interval = Duration::from_secs(DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS);
        if configured_interval.as_secs() < catch_up_interval.as_secs() {
            configured_interval
        } else {
            catch_up_interval
        }
    }
}

pub(super) fn directory_sync_request_count_for_objects(object_count: usize) -> u32 {
    let object_requests = object_count.div_ceil(MAX_DIRECTORY_SYNC_OBJECTS_V1);
    1u32.saturating_add(u32::try_from(object_requests).unwrap_or(u32::MAX))
}

/// Returns the retry delay after a consecutive producer failure.
///
/// The first failure is retried on the next ordinary tick. Later failures skip
/// 1, 3, 7, then at most 15 nominal intervals before the hard delay cap.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(super) fn directory_sync_failure_backoff_delay_secs(
    interval_secs: u64,
    consecutive_failures: u64,
) -> u64 {
    if consecutive_failures <= 1 {
        return 0;
    }
    let exponent = consecutive_failures.saturating_sub(1).min(4) as u32;
    let multiplier = (1u64 << exponent).saturating_sub(1);
    interval_secs
        .saturating_mul(multiplier)
        .min(DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS)
}

pub(super) fn should_continue_directory_mirror_catch_up(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND
        && requests_used.saturating_add(DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
}

#[must_use]
pub(super) const fn directory_sync_startup_delay_secs(local_node_id: &[u8; 32]) -> u64 {
    DIRECTORY_SYNC_STARTUP_DELAY_MIN_SECS
        + (local_node_id[0] as u64 % DIRECTORY_SYNC_STARTUP_DELAY_SPAN_SECS)
}
