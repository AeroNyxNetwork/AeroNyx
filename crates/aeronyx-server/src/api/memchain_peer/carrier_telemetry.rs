// [ARCH-SPLIT 2026-10-02]
// Circuit-breaker counters for commitment carriers.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

pub(super) fn record_commitment_block_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentBlockCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Observe the
    // monotonic circuit at one instant, then discard every per-slot detail.
    // Storage receives only bounded aggregate counts and cannot reconstruct a
    // source identity or endpoint from this call.
    storage.record_commitment_block_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
}

pub(super) fn record_commitment_authority_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentAuthorityCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Authority transport
    // has a distinct typed circuit. Collapse it to source-blind aggregates at
    // the storage boundary so telemetry cannot reconstruct a carrier or route.
    storage.record_commitment_authority_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
}

pub(super) fn record_commitment_certificate_carrier_circuit_telemetry(
    storage: &MemoryStorage,
    circuit_breaker: &CommitmentCertificateCarrierCircuitBreaker,
    cooldown_skips: usize,
    half_open_attempts: usize,
) {
    // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] Preserve the same
    // source-blind aggregate contract as block-page recovery while keeping an
    // independent typed circuit and independent counters.
    storage.record_commitment_certificate_carrier_circuit_observation(
        circuit_breaker.cooling_slots(Instant::now()),
        cooldown_skips,
        half_open_attempts,
    );
}
