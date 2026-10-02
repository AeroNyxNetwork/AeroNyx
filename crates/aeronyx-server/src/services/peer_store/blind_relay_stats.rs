// [ARCH-SPLIT 2026-10-02]
// Blind relay probe, retry, and quarantine counters.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Records a blind relay request that terminates at this node.
    ///
    /// Only aggregate routing facts are recorded. The route id, previous-hop
    /// id, full next-hop id, endpoint URL, exact encrypted blob bytes, and any
    /// payload-derived metadata remain outside PeerStore status.
    pub fn record_blind_relay_terminal(&self, now: u64, ttl_remaining: u8, blob_bytes: usize) {
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_terminal
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_terminal",
            "accepted",
            format!(
                "ttl_remaining={ttl_remaining} encrypted_blob_size_bucket={}",
                Self::blind_relay_blob_size_bucket(blob_bytes)
            ),
        );
    }

    pub(super) fn blind_relay_blob_size_bucket(blob_bytes: usize) -> &'static str {
        match blob_bytes {
            0..=4_096 => "lte_4kb",
            4_097..=65_536 => "lte_64kb",
            65_537..=262_144 => "lte_256kb",
            262_145..=1_048_576 => "lte_1mb",
            _ => "gt_1mb",
        }
    }

    /// Records a blind relay request forwarded to the next verified node.
    pub fn record_blind_relay_forwarded(&self, now: u64, ttl_remaining: u8) {
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_forwarded
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_forward",
            "accepted",
            format!("ttl_remaining={ttl_remaining} encrypted_blob_size_bucket=opaque"),
        );
    }

    /// Records a scheduled retry after a transient blind relay forward failure.
    ///
    /// This is intentionally aggregate-only. The reason bucket may be `http_503`,
    /// `blind_relay_request_timeout`, or similar transport state, but callers
    /// must not pass route ids, full peer ids, endpoint URLs, encrypted blobs,
    /// receiver identities, client IPs, DNS contents, voucher secrets, or
    /// payload-derived metadata.
    pub fn record_blind_relay_retry_attempt(&self, now: u64, reason: impl AsRef<str>) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_retry_attempted
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "scheduled",
            format!("reason_bucket={reason}"),
        );
    }

    /// Records that a blind relay forward succeeded after retrying.
    pub fn record_blind_relay_retry_succeeded(&self, now: u64, attempts: usize) {
        self.counters
            .blind_relay_retry_succeeded
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "accepted",
            format!("attempts={attempts}"),
        );
    }

    /// Records that retry attempts were exhausted before the final forward failed.
    pub fn record_blind_relay_retry_exhausted(
        &self,
        now: u64,
        attempts: usize,
        reason: impl AsRef<str>,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_retry_exhausted
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "rejected",
            format!("attempts={attempts} reason_bucket={reason}"),
        );
    }

    /// Records the result of a low-frequency synthetic blind relay route probe.
    ///
    /// Probe traffic is operator readiness evidence, not user traffic. It must
    /// never be added to encrypted message totals, packet totals, payload byte
    /// totals, billing, rewards, or user-facing usage claims. The detail is a
    /// stable reason bucket only; callers must not pass endpoint URLs, full
    /// node ids, route ids, encrypted blobs, receiver identities, client IPs,
    /// DNS contents, destinations, voucher secrets, wallet-level metadata, or
    /// plaintext.
    pub fn record_blind_relay_probe_result(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_probe_attempted
            .fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.counters
                .blind_relay_probe_succeeded
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters
                .blind_relay_probe_failed
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_probe_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_probe",
            if accepted { "accepted" } else { "rejected" },
            format!("reason_bucket={reason}"),
        );
    }

    /// Records a low-frequency synthetic two-hop blind relay path proof.
    ///
    /// This is protocol-health evidence only, not user traffic. Callers must
    /// pass stable reason buckets only. Never include path members, endpoint
    /// URLs, route ids, node ids, encrypted blobs, receiver identities, client
    /// IPs, DNS contents, destinations, Memory Chain plaintext, voucher
    /// secrets, private keys, wallet-level traffic, or social graph metadata.
    pub fn record_blind_relay_two_hop_probe_result(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
    ) {
        self.record_blind_relay_two_hop_probe_result_with_context(
            now, accepted, reason, 0, 0, 2, 1,
        );
    }

    /// Records a two-hop synthetic proof with privacy-safe route quality context.
    ///
    /// The context is intentionally bucketed before it enters history or audit
    /// output. It must never include node ids, endpoints, route ids, encrypted
    /// blobs, receiver identities, client IPs, DNS contents, domains, URLs,
    /// Memory Chain plaintext, voucher secrets, wallet-level traffic, or
    /// social graph metadata.
    pub fn record_blind_relay_two_hop_probe_result_with_context(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_two_hop_probe_attempted
            .fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.counters
                .blind_relay_two_hop_probe_succeeded
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters
                .blind_relay_two_hop_probe_failed
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_two_hop_probe_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_two_hop_probe",
            if accepted { "accepted" } else { "rejected" },
            format!(
                "reason_bucket={}; middle_candidates={}; terminal_candidates={}; ttl_shape={}",
                Self::two_hop_path_proof_reason_bucket(reason),
                Self::two_hop_candidate_count_bucket(middle_candidate_count),
                Self::two_hop_candidate_count_bucket(terminal_candidate_count),
                Self::two_hop_ttl_shape(entry_ttl, onward_ttl),
            ),
        );
        self.record_path_proof_event(
            now,
            accepted,
            reason,
            middle_candidate_count,
            terminal_candidate_count,
            entry_ttl,
            onward_ttl,
            2,
            "entry_middle_terminal",
        );
    }

    /// Records an entry -> middle -> middle -> terminal runtime proof.
    ///
    /// [THREE-HOP-SIGNED-RECOVERY 2026-08-02 by Codex] Three-hop evidence is
    /// isolated from established two-hop admission and relay counters. Its
    /// bounded aggregate history can be independently signed for warm restart,
    /// but is accepted only after the current three-hop route pool is rebuilt.
    /// Inputs are coarse
    /// counts and stable reason buckets only; never pass route members, node
    /// ids, endpoints, route ids, ciphertext, receivers, or client metadata.
    pub fn record_blind_relay_three_hop_probe_result_with_context(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
    ) {
        let reason = reason.as_ref();
        self.record_audit_event(
            now,
            "blind_relay_three_hop_probe",
            if accepted { "accepted" } else { "rejected" },
            format!(
                "reason_bucket={}; middle_candidates={}; terminal_candidates={}; ttl_shape={}",
                Self::two_hop_path_proof_reason_bucket(reason),
                Self::two_hop_candidate_count_bucket(middle_candidate_count),
                Self::two_hop_candidate_count_bucket(terminal_candidate_count),
                Self::two_hop_ttl_shape(entry_ttl, onward_ttl),
            ),
        );
        self.record_path_proof_event(
            now,
            accepted,
            reason,
            middle_candidate_count,
            terminal_candidate_count,
            entry_ttl,
            onward_ttl,
            3,
            "entry_middle_middle_terminal",
        );
    }

    /// Records a blind relay rejection with a stable privacy-safe reason.
    pub fn record_blind_relay_rejected(&self, now: u64, reason: impl AsRef<str>) {
        let reason = PrivacySafePeerHealthReason::blind_relay_rejection(reason.as_ref());
        let reason = reason.as_str();
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);

        match reason {
            "backpressure" => {
                self.counters
                    .blind_relay_backpressure_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
            "invalid_previous_hop" | "invalid_signature" => {
                self.counters
                    .blind_relay_invalid_signature
                    .fetch_add(1, Ordering::Relaxed);
            }
            "envelope_too_large" => {
                self.counters
                    .blind_relay_envelope_too_large
                    .fetch_add(1, Ordering::Relaxed);
            }
            "ttl_exhausted" => {
                self.counters
                    .blind_relay_ttl_exhausted
                    .fetch_add(1, Ordering::Relaxed);
            }
            "no_route" => {
                self.counters
                    .blind_relay_no_route
                    .fetch_add(1, Ordering::Relaxed);
            }
            "missing_endpoint" | "invalid_endpoint" => {
                self.counters
                    .blind_relay_invalid_endpoint
                    .fetch_add(1, Ordering::Relaxed);
            }
            "request_failed"
            | "forward_failed"
            | "delivery_receipt_invalid"
            | "failure_receipt_invalid" => {
                self.counters
                    .blind_relay_forward_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            "self_loop" | "route_loop" => {
                self.counters
                    .blind_relay_loop_detected
                    .fetch_add(1, Ordering::Relaxed);
            }
            "duplicate_route" | "replay_conflict" | "replay_response_expired" => {
                self.counters
                    .blind_relay_replay_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
            "timestamp_expired" | "timestamp_in_future" => {
                self.counters
                    .blind_relay_timestamp_rejected
                    .fetch_add(1, Ordering::Relaxed);
            }
            "rate_limited" => {
                self.counters
                    .blind_relay_rate_limited
                    .fetch_add(1, Ordering::Relaxed);
            }
            "quarantined" => {
                self.counters
                    .blind_relay_quarantined
                    .fetch_add(1, Ordering::Relaxed);
            }
            reason if reason.starts_with("http_") || reason.starts_with("blind_relay_request_") => {
                self.counters
                    .blind_relay_forward_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        self.record_audit_event(now, "blind_relay_forward", "rejected", reason.to_string());
    }

    /// Records that the local blind relay abuse guard started a short
    /// previous-hop quarantine.
    ///
    /// The detail is a stable bucket such as `rate_limit` or
    /// `failure_threshold`; callers must not pass node ids, route ids, endpoint
    /// values, encrypted blobs, wallet ids, or payload-derived details.
    pub fn record_blind_relay_quarantine_started(&self, now: u64, detail: impl Into<String>) {
        let detail = detail.into();
        let detail = PrivacySafePeerHealthReason::quarantine(&detail).into_inner();
        self.counters
            .blind_relay_quarantine_started
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(now, "blind_relay_quarantine", "limited", detail);
    }
}
