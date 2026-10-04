// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_message_dedup.rs
// ============================================
// Version: 1.1.0-LegacyCustodyRetryAdmission
//
// Creation Reason:
//   [CHAT-ONLINE-DEDUP-DOMAIN 2026-08-28 by Codex] Extract concurrent
//   online-path message deduplication from the relay orchestration service.
//
// Main Functionality:
//   - Defines the online message deduplication capability boundary.
//   - Atomically admits exactly one first observer for a message identifier.
//   - Retains only a bounded, process-local approximation of recent IDs.
//   - Reserves capacity before insertion; never evicts an active legacy lease.
//   - Failed/cancelled legacy attempts permit only exact-envelope custody retry.
//
// Dependencies:
//   - `parking_lot` supplies one short-held lock for both admission APIs.
//   - An owned Arc lease supplies cancellation-safe, generation-checked release.
//   - `chat_relay.rs` composes this capability without owning its container.
//
// Main Logical Flow:
//   1. Inspect ID, commitment and phase under the shared admission lock.
//   2. Reserve a non-wrapping generation and bounded slot before admission.
//   3. Release the lock before delivery; the lease owns no lock guard.
//   4. Complete explicitly, or Drop into local-custody-only retry eligibility.
//
// Important Note for Next Developer:
//   - This is an online-path optimization, not durable idempotency evidence.
//   - Keep admission atomic; a contains-then-insert sequence is race-prone.
//   - Do not store senders, receivers, payloads, routes, or timestamps here.
//   - Boolean true suppresses duplicates and fail-closed capacity exhaustion.
//   - Durable verified-submit and blind-route replay remain separate domains.
//   - Inactive eviction and process restart forget this non-durable barrier.
//
// Last Modified:
//   [LEGACY-CUSTODY-RETRY 2026-10-04 by Codex] Shared bounded lease admission.
//   v1.0.0-BoundedOnlineDedupDomain - Initial capability extraction
// ============================================

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

/// Capability for process-local online message duplicate detection.
pub(crate) trait OnlineMessageDeduplication {
    /// Returns `true` for retained IDs or fail-closed admission exhaustion.
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool;
}

/// Fixed-capacity concurrent online message deduplicator.
pub(crate) struct BoundedOnlineMessageDedup {
    state: Arc<Mutex<DedupState>>,
    capacity: usize,
}

// [LEGACY-CUSTODY-RETRY 2026-10-04 by Codex] Both APIs use this one domain;
// the boolean API cannot evict or overwrite a live legacy attempt.
struct DedupState {
    observations: HashMap<[u8; 16], Observation>,
    generation: u64,
}

struct Observation {
    generation: u64,
    commitment: Option<[u8; 32]>,
    phase: LegacyPhase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegacyPhase {
    InFlight,
    Completed,
    EntryRetryOnly,
}

#[derive(Clone, Copy)]
enum LegacyAttemptMode {
    InitialDelivery,
    EntryRetryOnly,
}

/// Coarse admission diagnostics; no envelope or identifiers are retained here.
pub(crate) enum LegacyAdmissionError {
    InvalidEnvelope,
    Duplicate,
    InFlight,
    Conflict,
    Capacity,
    GenerationExhausted,
}

impl LegacyAdmissionError {
    pub(crate) fn reason_bucket(&self) -> &'static str {
        match self {
            Self::InvalidEnvelope => "invalid_envelope",
            Self::Duplicate => "duplicate",
            Self::InFlight => "in_flight",
            Self::Conflict => "envelope_conflict",
            Self::Capacity => "dedup_capacity",
            Self::GenerationExhausted => "dedup_generation_exhausted",
        }
    }
}

/// Owned cancellation barrier, intentionally not Clone or Debug. It holds no
/// lock across await and retains no ciphertext, route, or private session.
#[must_use]
pub(crate) struct LegacyDeliveryLease {
    state: Arc<Mutex<DedupState>>,
    message_id: [u8; 16],
    generation: u64,
    mode: LegacyAttemptMode,
}

impl LegacyDeliveryLease {
    pub(crate) fn custody_retry_only(&self) -> bool {
        matches!(self.mode, LegacyAttemptMode::EntryRetryOnly)
    }

    /// Preserves legacy socket-write success semantics, NOT durable delivery.
    pub(crate) fn complete_online(self) {
        self.finish();
    }

    /// Call only after store_pending returns success, not after a send attempt.
    pub(crate) fn complete_custody(self) {
        self.finish();
    }

    fn finish(&self) {
        let mut state = self.state.lock();
        if let Some(entry) = state.observations.get_mut(&self.message_id) {
            if entry.generation == self.generation && entry.phase == LegacyPhase::InFlight {
                entry.phase = LegacyPhase::Completed;
            }
        }
    }
}

impl Drop for LegacyDeliveryLease {
    fn drop(&mut self) {
        // [LEGACY-CUSTODY-RETRY 2026-10-04 by Codex] No I/O or async cleanup:
        // cancellation after an ambiguous effect must not grant another route.
        let mut state = self.state.lock();
        if let Some(entry) = state.observations.get_mut(&self.message_id) {
            if entry.generation == self.generation && entry.phase == LegacyPhase::InFlight {
                entry.phase = LegacyPhase::EntryRetryOnly;
            }
        }
    }
}

impl DedupState {
    fn next_generation(&mut self) -> Result<u64, LegacyAdmissionError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(LegacyAdmissionError::GenerationExhausted)?;
        Ok(self.generation)
    }

    fn reserve_slot(&mut self, capacity: usize) -> Result<(), LegacyAdmissionError> {
        if capacity == 0 {
            return Err(LegacyAdmissionError::Capacity);
        }
        if self.observations.len() < capacity {
            return Ok(());
        }
        let oldest = self
            .observations
            .iter()
            .filter(|(_, entry)| entry.phase != LegacyPhase::InFlight)
            .min_by_key(|(_, entry)| entry.generation)
            .map(|(message_id, _)| *message_id)
            .ok_or(LegacyAdmissionError::Capacity)?;
        self.observations.remove(&oldest);
        Ok(())
    }
}

impl BoundedOnlineMessageDedup {
    /// Creates an empty process-local deduplicator with the supplied capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(DedupState {
                observations: HashMap::new(),
                generation: 0,
            })),
            capacity,
        }
    }

    pub(crate) fn begin_legacy(
        &self,
        message_id: &[u8; 16],
        commitment: [u8; 32],
    ) -> Result<LegacyDeliveryLease, LegacyAdmissionError> {
        let mut state = self.state.lock();
        let mode = match state.observations.get(message_id) {
            Some(entry) => {
                if let Some(retained) = entry.commitment {
                    if retained != commitment {
                        return Err(LegacyAdmissionError::Conflict);
                    }
                }
                match entry.phase {
                    LegacyPhase::Completed => return Err(LegacyAdmissionError::Duplicate),
                    LegacyPhase::InFlight => return Err(LegacyAdmissionError::InFlight),
                    LegacyPhase::EntryRetryOnly => LegacyAttemptMode::EntryRetryOnly,
                }
            }
            None => LegacyAttemptMode::InitialDelivery,
        };
        let generation = state.next_generation()?;
        if matches!(mode, LegacyAttemptMode::InitialDelivery) {
            state.reserve_slot(self.capacity)?;
        }
        state.observations.insert(
            *message_id,
            Observation {
                generation,
                commitment: Some(commitment),
                phase: LegacyPhase::InFlight,
            },
        );
        Ok(LegacyDeliveryLease {
            state: Arc::clone(&self.state),
            message_id: *message_id,
            generation,
            mode,
        })
    }
}

impl OnlineMessageDeduplication for BoundedOnlineMessageDedup {
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool {
        let mut state = self.state.lock();
        if state.observations.contains_key(message_id) {
            return true;
        }
        let Ok(generation) = state.next_generation() else {
            return true;
        };
        if state.reserve_slot(self.capacity).is_err() {
            return true;
        }
        state.observations.insert(
            *message_id,
            Observation {
                generation,
                commitment: None,
                phase: LegacyPhase::Completed,
            },
        );
        false
    }
}
