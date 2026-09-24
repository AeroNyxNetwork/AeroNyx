// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/cache_persistence.rs
// ============================================================================
//! Peer-cache persistence and fail-closed restart recovery projections.
//!
//! [PEER-CACHE-PERSISTENCE-SPLIT 2026-09-25 by Codex] These inherent methods
//! keep their existing signatures and behavior while separating local cache
//! lifecycle from live discovery, descriptor admission, and route selection.

use super::*;

impl PeerStore {
    /// Records a bootstrap or peer-cache load source result.
    pub fn record_bootstrap_source(
        &self,
        now: u64,
        source_kind: impl Into<String>,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_kind = source_kind.into();
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.last_source_kind = Some(source_kind.clone());
            status.last_source_status = Some(source_status.clone());
            status.last_source_detail = Some(detail.clone());
            status.last_source_at = Some(now);
            status.recovery_status = Some(source_status.clone());
            status.recovery_detail = Some(format!("source_kind={source_kind} {detail}"));
            status.recovery_at = Some(now);
            if matches!(source_kind.as_str(), "cache" | "cache_backup") {
                status.last_cache_load_source = Some(source_kind.clone());
                status.last_cache_load_status = Some(source_status.clone());
                status.last_cache_load_detail = Some(detail.clone());
                status.last_cache_load_at = Some(now);
            }
        }
        self.record_audit_event(
            now,
            "bootstrap_source",
            source_status,
            format!("kind={source_kind} {detail}"),
        );
    }

    /// Records peer-cache startup load status without exposing local paths.
    ///
    /// This method exists for callers that need to update cache recovery
    /// evidence without changing the generic bootstrap source state. Most
    /// bootstrap imports should continue using `record_bootstrap_source()`,
    /// which already mirrors cache/cache_backup events into these fields.
    pub fn record_peer_cache_load_status(
        &self,
        now: u64,
        source_kind: impl Into<String>,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_kind = source_kind.into();
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.last_cache_load_source = Some(source_kind.clone());
            status.last_cache_load_status = Some(source_status.clone());
            status.last_cache_load_detail = Some(detail.clone());
            status.last_cache_load_at = Some(now);
        }
        self.record_audit_event(
            now,
            "peer_cache_load",
            source_status,
            format!("kind={source_kind} {detail}"),
        );
    }

    /// Records peer-cache save status.
    pub fn record_cache_save_status(
        &self,
        now: u64,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.last_cache_save_status = Some(source_status.clone());
            status.last_cache_save_detail = Some(detail.clone());
            status.last_cache_save_at = Some(now);
        }
        self.record_audit_event(now, "peer_cache_save", source_status, detail);
    }

    /// Records that the latest cache write durably included a separately
    /// signed two-hop proof section.
    ///
    /// This count is aggregate-only and process-local. Admission uses it to
    /// distinguish merely configured restart recovery from a proof stability
    /// window that has actually reached durable storage.
    pub fn record_two_hop_proof_cache_persisted(
        &self,
        now: u64,
        persisted: usize,
        stability_ready: bool,
    ) {
        self.record_path_proof_cache_persisted(
            PathProofCacheKind::TwoHop,
            now,
            persisted,
            stability_ready,
        );
    }

    /// Records durable persistence of the independently signed three-hop
    /// aggregate proof section.
    pub fn record_three_hop_proof_cache_persisted(
        &self,
        now: u64,
        persisted: usize,
        stability_ready: bool,
    ) {
        self.record_path_proof_cache_persisted(
            PathProofCacheKind::ThreeHop,
            now,
            persisted,
            stability_ready,
        );
    }

    fn record_path_proof_cache_persisted(
        &self,
        kind: PathProofCacheKind,
        now: u64,
        persisted: usize,
        stability_ready: bool,
    ) {
        {
            let mut status = self.bootstrap_status.write();
            match kind {
                PathProofCacheKind::TwoHop => {
                    status.last_two_hop_proof_cache_persisted = persisted as u64;
                    status.last_two_hop_proof_cache_persisted_stability_ready = stability_ready;
                    status.last_two_hop_proof_cache_persisted_at = Some(now);
                    status.last_two_hop_proof_cache_rollback_protection =
                        Some("anchored".to_string());
                }
                PathProofCacheKind::ThreeHop => {
                    status.last_three_hop_proof_cache_persisted = persisted as u64;
                    status.last_three_hop_proof_cache_persisted_stability_ready = stability_ready;
                    status.last_three_hop_proof_cache_persisted_at = Some(now);
                    status.last_three_hop_proof_cache_rollback_protection =
                        Some("anchored".to_string());
                }
            }
        }
        self.record_audit_event(
            now,
            format!("{}_persist", kind.audit_prefix()),
            "success",
            format!(
                "schema_version={} persisted={persisted} stability_ready={stability_ready}",
                kind.schema_version()
            ),
        );
    }

    /// Records the local recovery-anchor decision for a signed two-hop cache.
    pub fn record_two_hop_proof_cache_rollback_protection(
        &self,
        now: u64,
        generation: u64,
        protection: &str,
    ) {
        self.record_path_proof_cache_rollback_protection(
            PathProofCacheKind::TwoHop,
            now,
            generation,
            protection,
        );
    }

    /// Records the local recovery-anchor decision for a signed three-hop cache.
    pub fn record_three_hop_proof_cache_rollback_protection(
        &self,
        now: u64,
        generation: u64,
        protection: &str,
    ) {
        self.record_path_proof_cache_rollback_protection(
            PathProofCacheKind::ThreeHop,
            now,
            generation,
            protection,
        );
    }

    /// Records the local recovery-anchor decision for the signed route-state
    /// section without retaining or exporting its digest.
    ///
    /// [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] This additive status
    /// mirrors existing proof/delivery protection buckets for node operators.
    pub fn record_routeability_cache_rollback_protection(
        &self,
        now: u64,
        generation: u64,
        protection: &str,
    ) {
        let protection = Self::recovery_anchor_protection_bucket(protection);
        self.bootstrap_status
            .write()
            .last_routeability_cache_rollback_protection = Some(protection.to_string());
        let outcome = match protection {
            "anchored" => "accepted",
            "cache_ahead" | "legacy_unanchored" | "not_checked" => "warning",
            _ => "rejected",
        };
        self.record_audit_event(
            now,
            "routeability_cache_rollback_protection",
            outcome,
            format!("generation={generation} protection={protection}"),
        );
    }

    fn recovery_anchor_protection_bucket(protection: &str) -> &str {
        match protection {
            "anchored" | "cache_ahead" | "legacy_unanchored" | "anchor_missing"
            | "anchor_invalid" | "anchor_conflict" | "rollback_detected" | "not_checked" => {
                protection
            }
            _ => "unknown",
        }
    }

    fn record_path_proof_cache_rollback_protection(
        &self,
        kind: PathProofCacheKind,
        now: u64,
        generation: u64,
        protection: &str,
    ) {
        let protection = Self::recovery_anchor_protection_bucket(protection);
        {
            let mut status = self.bootstrap_status.write();
            match kind {
                PathProofCacheKind::TwoHop => {
                    status.last_two_hop_proof_cache_rollback_protection =
                        Some(protection.to_string());
                }
                PathProofCacheKind::ThreeHop => {
                    status.last_three_hop_proof_cache_rollback_protection =
                        Some(protection.to_string());
                }
            }
        }
        let outcome = match protection {
            "anchored" => "accepted",
            "cache_ahead" | "legacy_unanchored" | "not_checked" => "warning",
            _ => "rejected",
        };
        self.record_audit_event(
            now,
            format!("{}_rollback_protection", kind.audit_prefix()),
            outcome,
            format!("generation={generation} protection={protection}"),
        );
    }

    /// Records that the latest atomic peer-cache write and its independent
    /// signed rollback anchor durably included aggregate delivery evidence.
    pub fn record_client_delivery_cache_persisted(
        &self,
        now: u64,
        persisted: u64,
        generation: u64,
    ) {
        {
            let mut status = self.bootstrap_status.write();
            status.last_client_delivery_cache_persisted = persisted;
            status.last_client_delivery_cache_persisted_at = Some(now);
            status.last_client_delivery_cache_generation = generation;
            status.last_client_delivery_cache_rollback_protection = Some("anchored".to_string());
        }
        self.record_audit_event(
            now,
            "client_delivery_cache_persist",
            "success",
            format!(
                "schema_version={} generation={generation} persisted={persisted} rollback_protection=anchored",
                VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
            ),
        );
    }

    /// Records the allowlisted local rollback-protection result without
    /// retaining anchor bytes, signatures, paths, peer ids, or route data.
    pub fn record_client_delivery_cache_rollback_protection(
        &self,
        now: u64,
        generation: u64,
        protection: &str,
    ) {
        let protection = Self::recovery_anchor_protection_bucket(protection);
        {
            let mut status = self.bootstrap_status.write();
            status.last_client_delivery_cache_generation = generation;
            status.last_client_delivery_cache_rollback_protection = Some(protection.to_string());
        }
        let outcome = match protection {
            "anchored" => "accepted",
            "cache_ahead" | "legacy_unanchored" | "not_checked" => "warning",
            _ => "rejected",
        };
        self.record_audit_event(
            now,
            "client_delivery_cache_rollback_protection",
            outcome,
            format!("generation={generation} protection={protection}"),
        );
    }

    /// Returns the aggregate generation represented by current recovered state.
    ///
    /// [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] The value is
    /// deliberately detached from cache paths, signatures, digests, peers,
    /// routes, messages, and clients. Startup orchestration uses it only to
    /// prove that the local anchor sent to external witnesses protects the
    /// exact cache generation whose readiness evidence is currently loaded.
    #[must_use]
    pub fn peer_cache_recovery_generation(&self) -> u64 {
        self.bootstrap_status
            .read()
            .last_client_delivery_cache_generation
    }

    /// Clears only restored aggregate client-delivery readiness evidence.
    ///
    /// This startup-only fail-closed operation must run before public listeners
    /// can accept new client receipts. It does not touch descriptors,
    /// routeability evidence, proof history, or ordinary relay counters.
    pub fn clear_restored_verified_client_delivery_evidence(&self, now: u64, reason: &str) {
        let reason = Self::external_witness_gate_reason(reason);
        self.counters
            .verified_client_onion_deliveries
            .store(0, Ordering::Release);
        self.counters
            .last_verified_client_onion_delivery_at
            .store(0, Ordering::Release);
        {
            let mut status = self.bootstrap_status.write();
            status.last_client_delivery_cache_status = Some("rejected".to_string());
            status.last_client_delivery_cache_restored = 0;
            status.last_client_delivery_cache_at = Some(now);
        }
        self.mark_peer_cache_dirty();
        self.record_audit_event(
            now,
            "client_delivery_cache_external_witness_gate",
            "rejected",
            format!("restored_deliveries=0 reason={reason}"),
        );
    }

    /// Clears every restart-readiness section covered by the recovery anchor.
    ///
    /// [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] This operation is
    /// startup-only and must run before public listeners or route probes. An
    /// external witness rollback/conflict means the node cannot distinguish a
    /// whole-host replay from current state, so route health, quarantine,
    /// two-hop/three-hop proof history, and aggregate client delivery all fail
    /// closed together. Independently verified descriptors remain available
    /// so bounded fresh probes can rebuild readiness without a discovery
    /// outage. Ordinary process relay counters are intentionally untouched.
    pub fn clear_restored_peer_cache_readiness_evidence(&self, now: u64, reason: &str) {
        let reason = Self::external_witness_gate_reason(reason);
        let route_records_cleared = {
            let mut route_health = self.route_health.write();
            let count = route_health.len();
            route_health.clear();
            count
        };
        let two_hop_events_cleared = {
            let mut events = self.two_hop_path_proof_events.write();
            let count = events.len();
            events.clear();
            count
        };
        let three_hop_events_cleared = {
            let mut events = self.three_hop_path_proof_events.write();
            let count = events.len();
            events.clear();
            count
        };
        self.counters
            .verified_client_onion_deliveries
            .store(0, Ordering::Release);
        self.counters
            .last_verified_client_onion_delivery_at
            .store(0, Ordering::Release);

        {
            let mut status = self.bootstrap_status.write();
            status.last_routeability_cache_rejected = status
                .last_routeability_cache_rejected
                .saturating_add(status.last_routeability_cache_restored);
            status.last_routeability_cache_status = Some("rejected".to_string());
            status.last_routeability_cache_restored = 0;
            status.last_routeability_cache_at = Some(now);

            status.last_two_hop_proof_cache_rejected = status
                .last_two_hop_proof_cache_rejected
                .saturating_add(status.last_two_hop_proof_cache_restored);
            status.last_two_hop_proof_cache_status = Some("rejected".to_string());
            status.last_two_hop_proof_cache_restored = 0;
            status.last_two_hop_proof_cache_restored_stability_ready = false;
            status.last_two_hop_proof_cache_at = Some(now);

            status.last_three_hop_proof_cache_rejected = status
                .last_three_hop_proof_cache_rejected
                .saturating_add(status.last_three_hop_proof_cache_restored);
            status.last_three_hop_proof_cache_status = Some("rejected".to_string());
            status.last_three_hop_proof_cache_restored = 0;
            status.last_three_hop_proof_cache_restored_stability_ready = false;
            status.last_three_hop_proof_cache_at = Some(now);

            status.last_client_delivery_cache_status = Some("rejected".to_string());
            status.last_client_delivery_cache_restored = 0;
            status.last_client_delivery_cache_at = Some(now);
        }
        self.mark_peer_cache_dirty();
        self.record_audit_event(
            now,
            "peer_cache_external_witness_gate",
            "rejected",
            format!(
                "route_records_cleared={route_records_cleared} two_hop_events_cleared={two_hop_events_cleared} three_hop_events_cleared={three_hop_events_cleared} restored_deliveries=0 reason={reason}"
            ),
        );
    }

    fn external_witness_gate_reason(reason: &str) -> &'static str {
        match reason {
            "external_witness_unavailable" => "external_witness_unavailable",
            "external_witness_rollback" => "external_witness_rollback",
            "external_witness_conflict" => "external_witness_conflict",
            "external_witness_gap" => "external_witness_gap",
            _ => "external_witness_invalid",
        }
    }

    /// Records the independently evaluated proof-cache authentication bucket.
    ///
    /// The parser owns signature verification; PeerStore stores only this
    /// allowlisted result so API admission never depends on free-form log text.
    pub fn record_two_hop_proof_cache_authentication(&self, now: u64, authentication: &str) {
        self.record_path_proof_cache_authentication(
            PathProofCacheKind::TwoHop,
            now,
            authentication,
        );
    }

    /// Records independent authentication of the signed three-hop proof
    /// section without retaining signature bytes or route metadata.
    pub fn record_three_hop_proof_cache_authentication(&self, now: u64, authentication: &str) {
        self.record_path_proof_cache_authentication(
            PathProofCacheKind::ThreeHop,
            now,
            authentication,
        );
    }

    fn record_path_proof_cache_authentication(
        &self,
        kind: PathProofCacheKind,
        now: u64,
        authentication: &str,
    ) {
        let authentication = match authentication {
            "verified"
            | "legacy_descriptor_only"
            | "signature_invalid"
            | "identity_unavailable" => authentication,
            _ => "unknown",
        };
        let mut status = self.bootstrap_status.write();
        match kind {
            PathProofCacheKind::TwoHop => {
                status.last_two_hop_proof_cache_authentication = Some(authentication.to_string());
            }
            PathProofCacheKind::ThreeHop => {
                status.last_three_hop_proof_cache_authentication = Some(authentication.to_string());
            }
        }
        drop(status);
        self.record_audit_event(
            now,
            format!("{}_authentication", kind.audit_prefix()),
            if authentication == "verified" {
                "accepted"
            } else if authentication == "legacy_descriptor_only" {
                "warning"
            } else {
                "rejected"
            },
            format!("authentication={authentication}"),
        );
    }

    /// Records the independently evaluated aggregate client-delivery cache
    /// authentication bucket without retaining any proof inputs.
    pub fn record_client_delivery_cache_authentication(&self, now: u64, authentication: &str) {
        let authentication = match authentication {
            "verified"
            | "legacy_descriptor_only"
            | "signature_invalid"
            | "identity_unavailable" => authentication,
            _ => "unknown",
        };
        self.bootstrap_status
            .write()
            .last_client_delivery_cache_authentication = Some(authentication.to_string());
        self.record_audit_event(
            now,
            "client_delivery_cache_authentication",
            if authentication == "verified" {
                "accepted"
            } else if authentication == "legacy_descriptor_only" {
                "warning"
            } else {
                "rejected"
            },
            format!("authentication={authentication}"),
        );
    }

    /// Imports descriptors from a local peer-cache snapshot.
    ///
    /// Unlike gossip/bootstrap imports, peer-cache recovery may retain
    /// expired-but-authentic signed descriptors so operators do not lose local
    /// peer history after restart. Retained expired records are never counted
    /// as valid, exported to public gossip, or selected as routeable peers
    /// until a fresh signed descriptor is received and `verify_at(now)` passes.
    pub fn load_peer_cache_snapshot_from_source(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
        source: impl Into<String>,
    ) -> PeerStoreImportReport {
        let source = source.into();
        let mut report = PeerStoreImportReport {
            total: snapshot.peers.len(),
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };

        for descriptor in &snapshot.peers {
            if descriptor.verify_at(now).is_ok() {
                match self.upsert_verified_from_source(descriptor.clone(), now, source.clone()) {
                    Ok(true) => report.inserted += 1,
                    Ok(false) => report.unchanged += 1,
                    Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                    Err(_) => report.rejected += 1,
                }
                continue;
            }

            match self.retain_signed_expired_from_cache(descriptor.clone(), now, source.clone()) {
                Ok(true) => report.inserted += 1,
                Ok(false) => report.unchanged += 1,
                Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                Err(_) => report.rejected += 1,
            }
        }

        self.record_import_report(&report, now);
        report
    }

    fn retain_signed_expired_from_cache(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: String,
    ) -> Result<bool, PeerStoreError> {
        if descriptor.verify_signature().is_err() || descriptor.descriptor.is_valid_at(now) {
            return Err(PeerStoreError::VerificationFailed);
        }

        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let mut peers = self.peers.write();
        let existing = peers.get(&node_id);
        let existing_sequence = existing.map(SignedNodeDescriptor::sequence);

        if let Some(current) = existing_sequence {
            if incoming_sequence < current {
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
            // [DISCOVERY-CACHE-SEQUENCE-FENCING 2026-09-13 by Codex] Expiry
            // changes routeability, not descriptor identity. Mirror the live
            // upsert fence so restart recovery cannot silently treat two
            // authentic but different descriptors at one sequence as an exact
            // retry. The retained original remains non-routeable and unchanged.
            if incoming_sequence == current && existing != Some(&descriptor) {
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("sequence_conflict"),
                );
                return Err(PeerStoreError::VerificationFailed);
            }
        } else if let Some(max_peers) = self.max_peers() {
            if peers.len() >= max_peers {
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        let changed = existing_sequence != Some(incoming_sequence);
        if changed {
            peers.insert(node_id, descriptor.clone());
        }
        drop(peers);

        {
            let mut metadata = self.peer_runtime.write();
            let entry = metadata
                .entry(node_id)
                .or_insert_with(|| PeerRuntimeMetadata {
                    source: source.clone(),
                    first_seen_at: now,
                    last_seen_at: now,
                    last_sequence: incoming_sequence,
                    imported_count: 0,
                    expired_degraded_at: Some(now),
                });
            entry.source = source.clone();
            entry.last_seen_at = now;
            entry.last_sequence = incoming_sequence;
            entry.expired_degraded_at = Some(now);
            if changed {
                entry.imported_count = entry.imported_count.saturating_add(1);
            }
        }

        self.record_peer_event(
            now,
            "peer_expired",
            "retained",
            source,
            &node_id,
            Some(incoming_sequence),
            Some("signature_valid_descriptor_expired"),
        );
        Ok(changed)
    }

    /// Waits until security-relevant peer evidence needs a durable refresh.
    /// Tokio `Notify` coalesces route and delivery bursts into bounded wakeups.
    pub async fn wait_for_peer_cache_dirty(&self) {
        self.peer_cache_notify.notified().await;
    }

    /// Atomically claims the pending peer-cache evidence update.
    ///
    /// The persistence task calls this immediately before exporting a cache
    /// snapshot. A new evidence transition during the write sets the flag
    /// again and schedules a follow-up flush, so abrupt cancellation cannot
    /// silently clear evidence that was not part of the snapshot.
    pub fn take_peer_cache_dirty(&self) -> bool {
        self.peer_cache_dirty.swap(false, Ordering::AcqRel)
    }

    /// Re-arms peer-cache persistence after a deferred write.
    ///
    /// This preserves coalescing semantics: repeated calls set one dirty bit
    /// and one notification permit, without allocating an unbounded queue.
    pub fn mark_peer_cache_dirty(&self) {
        self.peer_cache_dirty.store(true, Ordering::Release);
        self.peer_cache_notify.notify_one();
    }

    /// Backward-compatible aggregate-delivery cache waiter.
    pub async fn wait_for_client_delivery_cache_dirty(&self) {
        self.wait_for_peer_cache_dirty().await;
    }

    /// Backward-compatible aggregate-delivery dirty-bit claim.
    pub fn take_client_delivery_cache_dirty(&self) -> bool {
        self.take_peer_cache_dirty()
    }

    /// Backward-compatible aggregate-delivery dirty marker.
    pub fn mark_client_delivery_cache_dirty(&self) {
        self.mark_peer_cache_dirty();
    }

    /// Exports a local peer-cache snapshot, including expired signed peers.
    ///
    /// This snapshot is for local restart recovery only. Public gossip and
    /// bootstrap responses must continue to use `export_bootstrap_snapshot()`,
    /// which filters to descriptors that are valid at `now`. Cache consumers
    /// must call `load_peer_cache_snapshot_from_source()` so expired records
    /// are retained only after signature verification and remain non-routeable.
    #[must_use]
    pub fn export_peer_cache_snapshot(&self, generated_at: u64) -> NodeBootstrapSnapshot {
        self.counters
            .last_snapshot_at
            .store(generated_at, Ordering::Relaxed);
        let mut valid = 0usize;
        let mut expired = 0usize;
        let mut descriptors: Vec<SignedNodeDescriptor> = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_signature().is_ok())
            // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] The
            // descriptor-only legacy cache cannot persist promotion authority.
            // On restart the node must rejoin Stage-A and revalidate probation.
            .filter(|descriptor| {
                !self
                    .permissionless_promotions
                    .read()
                    .contains_key(&descriptor.node_id())
            })
            .inspect(|descriptor| {
                if descriptor.descriptor.is_valid_at(generated_at) {
                    valid += 1;
                } else {
                    expired += 1;
                }
            })
            .cloned()
            .collect();

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        self.record_audit_event(
            generated_at,
            "peer_cache_export",
            "accepted",
            format!(
                "retained={} valid={} expired={}",
                descriptors.len(),
                valid,
                expired
            ),
        );

        NodeBootstrapSnapshot::new(generated_at, descriptors)
    }

    /// Exports a bounded, freshness-checked subset of synthetic two-hop proofs.
    ///
    /// The returned events contain only allowlisted aggregate fields and are
    /// intended for a separately signed local cache section. They are not a
    /// relay receipt, user-message history, route log, or consensus record.
    #[must_use]
    pub fn export_two_hop_path_proof_cache_events(
        &self,
        generated_at: u64,
    ) -> Vec<PeerStoreTwoHopPathProofEvent> {
        self.export_path_proof_cache_events(PathProofCacheKind::TwoHop, generated_at)
    }

    /// Exports a bounded, freshness-checked three-hop aggregate proof window.
    #[must_use]
    pub fn export_three_hop_path_proof_cache_events(
        &self,
        generated_at: u64,
    ) -> Vec<PeerStoreTwoHopPathProofEvent> {
        self.export_path_proof_cache_events(PathProofCacheKind::ThreeHop, generated_at)
    }

    fn export_path_proof_cache_events(
        &self,
        kind: PathProofCacheKind,
        generated_at: u64,
    ) -> Vec<PeerStoreTwoHopPathProofEvent> {
        let fresh_events = self
            .path_proof_events(kind)
            .read()
            .iter()
            .filter(|event| Self::path_proof_cache_event_valid(event, generated_at, kind))
            .cloned()
            .collect::<Vec<_>>();
        let message_delivery_count = fresh_events
            .iter()
            .filter(|event| event.proof_scope == "message_delivery")
            .count();
        let events = if message_delivery_count >= TWO_HOP_PATH_PROOF_STABILITY_MIN_ATTEMPTS as usize
        {
            // Preserve the same evidence used by runtime stability while also
            // retaining up to the three trailing failures that control the
            // circuit breaker. Fill remaining slots from newest aggregate
            // events, then restore chronological ordering for signature and
            // import determinism.
            let mut selected = Vec::with_capacity(TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES);
            for (index, _) in fresh_events
                .iter()
                .enumerate()
                .rev()
                .take_while(|(_, event)| event.outcome == "rejected")
                .take(TWO_HOP_PATH_PROOF_FAILURE_CIRCUIT_BREAKER_THRESHOLD as usize)
            {
                selected.push(index);
            }
            for (index, event) in fresh_events.iter().enumerate().rev() {
                if selected.len() >= TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES {
                    break;
                }
                if event.proof_scope == "message_delivery" && !selected.contains(&index) {
                    selected.push(index);
                }
            }
            for index in (0..fresh_events.len()).rev() {
                if selected.len() >= TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES {
                    break;
                }
                if !selected.contains(&index) {
                    selected.push(index);
                }
            }
            selected.sort_unstable();
            selected
                .into_iter()
                .map(|index| fresh_events[index].clone())
                .collect::<Vec<_>>()
        } else {
            let mut latest = fresh_events
                .into_iter()
                .rev()
                .take(TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES)
                .collect::<Vec<_>>();
            latest.reverse();
            latest
        };
        self.record_audit_event(
            generated_at,
            format!("{}_export", kind.audit_prefix()),
            "accepted",
            format!(
                "schema_version={} exported={} max_entries={} stale_after_seconds={}",
                kind.schema_version(),
                events.len(),
                TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES,
                PEER_ROUTEABILITY_STALE_AFTER_SECS
            ),
        );
        events
    }

    fn path_proof_route_pool_ready(&self, kind: PathProofCacheKind, now: u64) -> bool {
        match kind {
            PathProofCacheKind::TwoHop => {
                self.route_path_status(now)
                    .chat_two_hop_onion_ready
                    .complete
            }
            PathProofCacheKind::ThreeHop => self
                .scored_route_path_with_capabilities_excluding(
                    &[
                        NodeCapability::OnionMiddle,
                        NodeCapability::OnionMiddle,
                        NodeCapability::ChatRelay,
                    ],
                    now,
                    &[],
                )
                .is_some(),
        }
    }

    /// Restores a signed, bounded two-hop synthetic proof window.
    ///
    /// Recovery is permitted only after descriptor-bound routeability evidence
    /// has rebuilt a complete, distinct middle/terminal path. This prevents an
    /// old proof cache from claiming readiness for a changed or unavailable
    /// route pool. Only synthetic probe counters are reconstructed; real relay
    /// received, terminal, forwarded, and payload-volume counters remain zero.
    pub fn restore_two_hop_path_proof_cache_events(
        &self,
        records: &[PeerStoreTwoHopPathProofEvent],
        now: u64,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        self.restore_path_proof_cache_events(PathProofCacheKind::TwoHop, records, now)
    }

    /// Restores an independently signed, bounded three-hop aggregate proof
    /// window after rebuilding a complete network-diverse route pool.
    pub fn restore_three_hop_path_proof_cache_events(
        &self,
        records: &[PeerStoreTwoHopPathProofEvent],
        now: u64,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        self.restore_path_proof_cache_events(PathProofCacheKind::ThreeHop, records, now)
    }

    fn restore_path_proof_cache_events(
        &self,
        kind: PathProofCacheKind,
        records: &[PeerStoreTwoHopPathProofEvent],
        now: u64,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        if records.is_empty() {
            return self.finish_path_proof_cache_restore(kind, 0, 0, 0, now, None);
        }
        if !self.path_proof_route_pool_ready(kind, now) {
            return self.finish_path_proof_cache_restore(
                kind,
                records.len(),
                0,
                records.len(),
                now,
                Some("route_pool_unready"),
            );
        }

        let bounded_start = records
            .len()
            .saturating_sub(TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES);
        let mut rejected = bounded_start;
        let mut restored_events = Vec::new();
        let mut previous_at = None;
        {
            let mut history = self.path_proof_events(kind).write();
            for event in records.iter().skip(bounded_start) {
                let timestamp_order_valid = previous_at.is_none_or(|at| event.at >= at);
                previous_at = Some(event.at);
                if !timestamp_order_valid
                    || !Self::path_proof_cache_event_valid(event, now, kind)
                    || history.iter().any(|existing| existing == event)
                {
                    rejected = rejected.saturating_add(1);
                    continue;
                }
                if history.len() >= MAX_TWO_HOP_PATH_PROOF_EVENTS {
                    history.pop_front();
                }
                history.push_back(event.clone());
                restored_events.push(event.clone());
            }
        }

        let restored = restored_events.len();
        if restored > 0 && kind == PathProofCacheKind::TwoHop {
            let succeeded = restored_events
                .iter()
                .filter(|event| event.outcome == "accepted")
                .count();
            let failed = restored.saturating_sub(succeeded);
            let latest_at = restored_events
                .iter()
                .map(|event| event.at)
                .max()
                .unwrap_or(0);
            self.counters
                .blind_relay_two_hop_probe_attempted
                .fetch_add(restored as u64, Ordering::Relaxed);
            self.counters
                .blind_relay_two_hop_probe_succeeded
                .fetch_add(succeeded as u64, Ordering::Relaxed);
            self.counters
                .blind_relay_two_hop_probe_failed
                .fetch_add(failed as u64, Ordering::Relaxed);
            self.counters
                .last_blind_relay_two_hop_probe_at
                .fetch_max(latest_at, Ordering::Relaxed);
            self.counters
                .last_blind_relay_at
                .fetch_max(latest_at, Ordering::Relaxed);
        }

        self.finish_path_proof_cache_restore(kind, records.len(), restored, rejected, now, None)
    }

    /// Records cache authentication failure without discarding independently
    /// verified descriptors or routeability evidence.
    pub fn reject_two_hop_path_proof_cache_events(
        &self,
        total: usize,
        now: u64,
        reason: &str,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        self.reject_path_proof_cache_events(PathProofCacheKind::TwoHop, total, now, reason)
    }

    /// Records a three-hop cache authentication failure without discarding
    /// independently verified descriptors, routeability, or two-hop history.
    pub fn reject_three_hop_path_proof_cache_events(
        &self,
        total: usize,
        now: u64,
        reason: &str,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        self.reject_path_proof_cache_events(PathProofCacheKind::ThreeHop, total, now, reason)
    }

    fn reject_path_proof_cache_events(
        &self,
        kind: PathProofCacheKind,
        total: usize,
        now: u64,
        reason: &str,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        let reason_bucket = match reason {
            "identity_unavailable" => "identity_unavailable",
            "legacy_unanchored" => "legacy_unanchored",
            "anchor_missing" => "anchor_missing",
            "anchor_invalid" => "anchor_invalid",
            "anchor_conflict" => "anchor_conflict",
            "rollback_detected" => "rollback_detected",
            _ => "signature_invalid",
        };
        {
            let mut status = self.bootstrap_status.write();
            match kind {
                PathProofCacheKind::TwoHop => {
                    status.last_two_hop_proof_cache_status = Some("rejected".to_string());
                    status.last_two_hop_proof_cache_restored = 0;
                    status.last_two_hop_proof_cache_restored_stability_ready = false;
                    status.last_two_hop_proof_cache_rejected = total as u64;
                    status.last_two_hop_proof_cache_at = Some(now);
                }
                PathProofCacheKind::ThreeHop => {
                    status.last_three_hop_proof_cache_status = Some("rejected".to_string());
                    status.last_three_hop_proof_cache_restored = 0;
                    status.last_three_hop_proof_cache_restored_stability_ready = false;
                    status.last_three_hop_proof_cache_rejected = total as u64;
                    status.last_three_hop_proof_cache_at = Some(now);
                }
            }
        }
        self.record_audit_event(
            now,
            format!("{}_restore", kind.audit_prefix()),
            "rejected",
            format!(
                "schema_version={} total={} restored=0 rejected={} status=rejected reason={reason_bucket}",
                kind.schema_version(),
                total,
                total
            ),
        );
        PeerStoreTwoHopProofCacheRestoreReport {
            total,
            restored: 0,
            rejected: total,
        }
    }

    fn finish_path_proof_cache_restore(
        &self,
        kind: PathProofCacheKind,
        total: usize,
        restored: usize,
        rejected: usize,
        now: u64,
        rejection_reason: Option<&str>,
    ) -> PeerStoreTwoHopProofCacheRestoreReport {
        let status_bucket = if total == 0 {
            "empty"
        } else if restored == total {
            "restored"
        } else if restored > 0 {
            "partial"
        } else {
            "rejected"
        };
        let restored_stability_ready = restored > 0
            && match kind {
                PathProofCacheKind::TwoHop => self.two_hop_path_proof_history(now).stability_ready,
                PathProofCacheKind::ThreeHop => {
                    self.three_hop_path_proof_history(now).stability_ready
                }
            };
        {
            let mut status = self.bootstrap_status.write();
            match kind {
                PathProofCacheKind::TwoHop => {
                    status.last_two_hop_proof_cache_status = Some(status_bucket.to_string());
                    status.last_two_hop_proof_cache_restored = restored as u64;
                    status.last_two_hop_proof_cache_restored_stability_ready =
                        restored_stability_ready;
                    status.last_two_hop_proof_cache_rejected = rejected as u64;
                    status.last_two_hop_proof_cache_at = Some(now);
                }
                PathProofCacheKind::ThreeHop => {
                    status.last_three_hop_proof_cache_status = Some(status_bucket.to_string());
                    status.last_three_hop_proof_cache_restored = restored as u64;
                    status.last_three_hop_proof_cache_restored_stability_ready =
                        restored_stability_ready;
                    status.last_three_hop_proof_cache_rejected = rejected as u64;
                    status.last_three_hop_proof_cache_at = Some(now);
                }
            }
        }
        let reason_detail = rejection_reason
            .map(|reason| format!(" reason={reason}"))
            .unwrap_or_default();
        self.record_audit_event(
            now,
            format!("{}_restore", kind.audit_prefix()),
            if restored > 0 || total == 0 {
                "accepted"
            } else {
                "rejected"
            },
            format!(
                "schema_version={} total={} restored={} rejected={} status={} stability_ready={}{}",
                kind.schema_version(),
                total,
                restored,
                rejected,
                status_bucket,
                restored_stability_ready,
                reason_detail
            ),
        );
        PeerStoreTwoHopProofCacheRestoreReport {
            total,
            restored,
            rejected,
        }
    }

    fn path_proof_cache_event_valid(
        event: &PeerStoreTwoHopPathProofEvent,
        now: u64,
        kind: PathProofCacheKind,
    ) -> bool {
        if event.at == 0
            || event.at > now
            || now.saturating_sub(event.at) > PEER_ROUTEABILITY_STALE_AFTER_SECS
            || event.path_shape != kind.path_shape()
            || event.hop_count != kind.hop_count()
            || event.path_policy != TWO_HOP_PATH_POLICY_NETWORK_DIVERSE
            || event.ttl_shape != kind.ttl_shape()
            || !matches!(
                event.middle_candidate_bucket.as_str(),
                "none" | "one" | "few" | "healthy" | "deep"
            )
            || !matches!(
                event.terminal_candidate_bucket.as_str(),
                "none" | "one" | "few" | "healthy" | "deep"
            )
        {
            return false;
        }

        match event.reason_bucket.as_str() {
            "onion_terminal_delivered" => {
                event.outcome == "accepted"
                    && event.evidence_mode == "synthetic_onion_message_delivery_probe"
                    && event.proof_scope == "message_delivery"
            }
            "accepted" => {
                event.outcome == "accepted"
                    && event.evidence_mode == kind.control_evidence_mode()
                    && event.proof_scope == "control_plane"
            }
            "onion_ack_rejected"
            | "onion_ack_decode"
            | "onion_kem_unavailable"
            | "ack_rejected"
            | "ack_decode"
            | "no_distinct_path"
            | "no_network_diverse_path"
            | "middle_missing_endpoint"
            | "middle_invalid_endpoint"
            | "onion_http_error"
            | "http_error"
            | "onion_request_error"
            | "request_error" => {
                event.outcome == "rejected"
                    && event.evidence_mode == kind.control_evidence_mode()
                    && event.proof_scope == "control_plane"
            }
            _ => false,
        }
    }

    /// Exports fresh aggregate terminal-receipt evidence for the signed local
    /// peer cache.
    ///
    /// Receipt bytes and all correlatable proof inputs have already been
    /// discarded. Returning `None` for empty or stale evidence prevents an old
    /// cache from reviving current relay readiness indefinitely.
    #[must_use]
    pub fn export_verified_client_delivery_cache_evidence(
        &self,
        generated_at: u64,
    ) -> Option<PeerStoreVerifiedClientDeliveryCacheEvidence> {
        let stats = self.counters.snapshot();
        let last_verified_at = stats.blind_relay.last_verified_client_onion_delivery_at?;
        if stats.blind_relay.verified_client_onion_deliveries == 0
            || last_verified_at == 0
            || last_verified_at > generated_at
            || generated_at.saturating_sub(last_verified_at) > PEER_ROUTEABILITY_STALE_AFTER_SECS
        {
            return None;
        }
        Some(PeerStoreVerifiedClientDeliveryCacheEvidence {
            verified_deliveries: stats.blind_relay.verified_client_onion_deliveries,
            last_verified_at,
        })
    }

    /// Restores signed aggregate terminal-receipt evidence after descriptor and
    /// routeability recovery.
    ///
    /// The historical count may be restored before receipt-capable peers are
    /// re-proven, but `real_relay_ready` remains false until at least two fresh
    /// receipt-capable peers exist in the current process.
    pub fn restore_verified_client_delivery_cache_evidence(
        &self,
        evidence: Option<&PeerStoreVerifiedClientDeliveryCacheEvidence>,
        now: u64,
    ) -> PeerStoreVerifiedClientDeliveryCacheRestoreReport {
        let Some(evidence) = evidence else {
            return self.finish_client_delivery_cache_restore(false, false, 0, now, None);
        };
        let valid = evidence.verified_deliveries > 0
            && evidence.last_verified_at > 0
            && evidence.last_verified_at <= now
            && now.saturating_sub(evidence.last_verified_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
            && self
                .route_path_status(now)
                .chat_two_hop_onion_ready
                .complete;
        if !valid {
            return self.finish_client_delivery_cache_restore(
                true,
                false,
                0,
                now,
                Some("evidence_or_route_pool_invalid"),
            );
        }

        self.counters
            .verified_client_onion_deliveries
            .fetch_max(evidence.verified_deliveries, Ordering::Relaxed);
        self.counters
            .last_verified_client_onion_delivery_at
            .fetch_max(evidence.last_verified_at, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .fetch_max(evidence.last_verified_at, Ordering::Relaxed);
        self.finish_client_delivery_cache_restore(
            true,
            true,
            evidence.verified_deliveries,
            now,
            None,
        )
    }

    /// Rejects an unauthenticated aggregate delivery section without
    /// discarding independently verified descriptors, routeability, or proof
    /// history.
    pub fn reject_verified_client_delivery_cache_evidence(
        &self,
        present: bool,
        now: u64,
        reason: &str,
    ) -> PeerStoreVerifiedClientDeliveryCacheRestoreReport {
        let reason = match reason {
            "identity_unavailable" => "identity_unavailable",
            "anchor_missing" => "anchor_missing",
            "anchor_invalid" => "anchor_invalid",
            "anchor_conflict" => "anchor_conflict",
            "rollback_detected" => "rollback_detected",
            _ => "signature_invalid",
        };
        self.finish_client_delivery_cache_restore(present, false, 0, now, Some(reason))
    }

    fn finish_client_delivery_cache_restore(
        &self,
        present: bool,
        restored: bool,
        restored_deliveries: u64,
        now: u64,
        rejection_reason: Option<&str>,
    ) -> PeerStoreVerifiedClientDeliveryCacheRestoreReport {
        let status_bucket = if restored {
            "restored"
        } else if present {
            "rejected"
        } else {
            "empty"
        };
        {
            let mut status = self.bootstrap_status.write();
            status.last_client_delivery_cache_status = Some(status_bucket.to_string());
            status.last_client_delivery_cache_restored = restored_deliveries;
            status.last_client_delivery_cache_at = Some(now);
        }
        let reason = rejection_reason
            .map(|value| format!(" reason={value}"))
            .unwrap_or_default();
        self.record_audit_event(
            now,
            "client_delivery_cache_restore",
            if restored || !present {
                "accepted"
            } else {
                "rejected"
            },
            format!(
                "schema_version={} present={} restored={} restored_deliveries={} status={}{}",
                VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
                present,
                restored,
                restored_deliveries,
                status_bucket,
                reason
            ),
        );
        PeerStoreVerifiedClientDeliveryCacheRestoreReport {
            present,
            restored,
            restored_deliveries,
        }
    }

    /// Exports fresh successful routeability evidence for local restart recovery.
    ///
    /// Every record is bound to the signed route surface proven by the latest
    /// direct opaque success. Sequence/TTL-only descriptor refreshes can retain
    /// that evidence; endpoint, capability, discovery-policy, or KEM changes
    /// cannot. Failure counters, quarantine state, proof history, endpoint
    /// values, route ids, and payload metadata remain excluded.
    #[must_use]
    pub fn export_routeability_cache_evidence(
        &self,
        generated_at: u64,
    ) -> Vec<PeerStoreRouteabilityCacheEvidence> {
        let peers = self.peers.read();
        let mut evidence = Vec::new();

        for (node_id, descriptor) in peers.iter() {
            if evidence.len() >= PEER_ROUTEABILITY_CACHE_MAX_ENTRIES {
                break;
            }
            if descriptor.verify_at(generated_at).is_err()
                || !self.permissionless_gate_allows(descriptor, generated_at, true)
                || descriptor
                    .descriptor
                    .public_endpoint
                    .as_deref()
                    .map(str::trim)
                    .is_none_or(str::is_empty)
            {
                continue;
            }
            let route_health = self.route_health.read();
            let Some(health) = route_health.get(node_id) else {
                continue;
            };
            let Some(last_success_at) = health.last_success_at else {
                continue;
            };
            if last_success_at > generated_at
                || generated_at.saturating_sub(last_success_at) > PEER_ROUTEABILITY_STALE_AFTER_SECS
                || health
                    .last_failure_at
                    .is_some_and(|failure_at| failure_at >= last_success_at)
                || Self::route_quarantine_remaining_seconds(health, generated_at).is_some()
            {
                continue;
            }
            let Some(descriptor_fingerprint_sha256) =
                Self::descriptor_routeability_surface_fingerprint(descriptor)
            else {
                continue;
            };
            if health.last_success_route_fingerprint_sha256.as_deref()
                != Some(descriptor_fingerprint_sha256.as_str())
            {
                continue;
            }
            evidence.push(PeerStoreRouteabilityCacheEvidence {
                node_id_hex: hex::encode(node_id),
                descriptor_sequence: descriptor.sequence(),
                descriptor_fingerprint_sha256,
                last_success_at,
                evidence_kind: ROUTEABILITY_EVIDENCE_KIND_ROUTE_SURFACE.to_string(),
            });
        }
        drop(peers);

        evidence.sort_by(|left, right| left.node_id_hex.cmp(&right.node_id_hex));
        self.record_audit_event(
            generated_at,
            "routeability_cache_export",
            "accepted",
            format!(
                "schema_version={} exported={} stale_after_seconds={}",
                ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION,
                evidence.len(),
                PEER_ROUTEABILITY_STALE_AFTER_SECS
            ),
        );
        evidence
    }

    /// Exports active, route-surface-bound quarantine windows for restart.
    ///
    /// [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Only an unexpired
    /// quarantine created by the fixed local route policy is eligible. Failure
    /// counts and reasons are deliberately omitted; the enclosing cache v2
    /// signature prevents this section from being removed while retaining old
    /// success evidence.
    #[must_use]
    pub fn export_route_quarantine_cache_evidence(
        &self,
        generated_at: u64,
    ) -> Vec<PeerStoreRouteQuarantineCacheEvidence> {
        let peers = self.peers.read();
        let route_health = self.route_health.read();
        let mut evidence = Vec::new();

        for (node_id, descriptor) in peers.iter() {
            if evidence.len() >= PEER_ROUTEABILITY_CACHE_MAX_ENTRIES {
                break;
            }
            if descriptor.verify_at(generated_at).is_err() {
                continue;
            }
            let Some(health) = route_health.get(node_id) else {
                continue;
            };
            let (Some(quarantined_at), Some(quarantine_until)) =
                (health.last_quarantine_at, health.quarantine_until)
            else {
                continue;
            };
            if quarantined_at > generated_at
                || quarantine_until <= generated_at
                || quarantine_until <= quarantined_at
                || quarantine_until.saturating_sub(quarantined_at)
                    > PEER_ROUTE_FAILURE_QUARANTINE_SECS
            {
                continue;
            }
            let Some(route_surface_fingerprint_sha256) =
                Self::descriptor_routeability_surface_fingerprint(descriptor)
            else {
                continue;
            };
            evidence.push(PeerStoreRouteQuarantineCacheEvidence {
                node_id_hex: hex::encode(node_id),
                descriptor_sequence: descriptor.sequence(),
                route_surface_fingerprint_sha256,
                quarantined_at,
                quarantine_until,
            });
        }
        drop(route_health);
        drop(peers);

        evidence.sort_by(|left, right| left.node_id_hex.cmp(&right.node_id_hex));
        self.record_audit_event(
            generated_at,
            "route_quarantine_cache_export",
            "accepted",
            format!(
                "schema_version={} exported={} max_duration_seconds={}",
                ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION,
                evidence.len(),
                PEER_ROUTE_FAILURE_QUARANTINE_SECS
            ),
        );
        evidence
    }

    /// Restores fresh active route quarantine after descriptors are verified.
    ///
    /// Every record is rebound to the current signed route surface. A newer
    /// sequence is accepted only when endpoint, capabilities, discovery policy,
    /// and KEM remain unchanged. Expired, future-dated, oversized, malformed,
    /// or rotated records fail closed per item.
    pub fn restore_route_quarantine_cache_evidence(
        &self,
        records: &[PeerStoreRouteQuarantineCacheEvidence],
        now: u64,
    ) -> PeerStoreRouteQuarantineCacheRestoreReport {
        let mut restored = 0usize;
        let mut rejected = records
            .len()
            .saturating_sub(PEER_ROUTEABILITY_CACHE_MAX_ENTRIES);

        for record in records.iter().take(PEER_ROUTEABILITY_CACHE_MAX_ENTRIES) {
            let mut node_id = [0u8; 32];
            if hex::decode_to_slice(&record.node_id_hex, &mut node_id).is_err()
                || record.quarantined_at > now
                || record.quarantine_until <= now
                || record.quarantine_until <= record.quarantined_at
                || record
                    .quarantine_until
                    .saturating_sub(record.quarantined_at)
                    > PEER_ROUTE_FAILURE_QUARANTINE_SECS
            {
                rejected = rejected.saturating_add(1);
                continue;
            }

            let descriptor = self.peers.read().get(&node_id).cloned();
            let Some(descriptor) = descriptor else {
                rejected = rejected.saturating_add(1);
                continue;
            };
            let route_surface_fingerprint =
                Self::descriptor_routeability_surface_fingerprint(&descriptor);
            if descriptor.verify_at(now).is_err()
                || descriptor.sequence() < record.descriptor_sequence
                || route_surface_fingerprint.as_deref()
                    != Some(record.route_surface_fingerprint_sha256.as_str())
            {
                rejected = rejected.saturating_add(1);
                continue;
            }

            let mut route_health = self.route_health.write();
            let health = route_health.entry(node_id).or_default();
            health.failure_count = health
                .failure_count
                .max(PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD);
            health.consecutive_failures = health
                .consecutive_failures
                .max(PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD);
            health.last_failure_at = Some(
                health
                    .last_failure_at
                    .unwrap_or_default()
                    .max(record.quarantined_at),
            );
            health.last_failure_reason = Some("restart_restored_quarantine".to_string());
            health.quarantine_count = health.quarantine_count.max(1);
            health.quarantine_until = Some(
                health
                    .quarantine_until
                    .unwrap_or_default()
                    .max(record.quarantine_until),
            );
            health.last_quarantine_at = Some(
                health
                    .last_quarantine_at
                    .unwrap_or_default()
                    .max(record.quarantined_at),
            );
            health.last_quarantine_reason = Some("consecutive_route_failures".to_string());
            restored = restored.saturating_add(1);
        }

        let report = PeerStoreRouteQuarantineCacheRestoreReport {
            total: records.len(),
            restored,
            rejected,
        };
        self.record_audit_event(
            now,
            "route_quarantine_cache_restore",
            if restored > 0 || records.is_empty() {
                "accepted"
            } else {
                "rejected"
            },
            format!(
                "schema_version={} total={} restored={} rejected={}",
                ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION,
                records.len(),
                restored,
                rejected
            ),
        );
        report
    }

    /// Rejects an unauthenticated route-quarantine section without restoring it.
    pub fn reject_route_quarantine_cache_evidence(
        &self,
        total: usize,
        now: u64,
        reason: &str,
    ) -> PeerStoreRouteQuarantineCacheRestoreReport {
        let reason_bucket = Self::route_cache_rejection_reason_bucket(reason);
        self.record_audit_event(
            now,
            "route_quarantine_cache_restore",
            "rejected",
            format!(
                "schema_version={} total={} restored=0 rejected={} reason={reason_bucket}",
                ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION, total, total
            ),
        );
        PeerStoreRouteQuarantineCacheRestoreReport {
            total,
            restored: 0,
            rejected: total,
        }
    }

    /// Restores fresh descriptor-bound routeability evidence from local cache.
    ///
    /// This method fails closed per record. It restores only a successful
    /// timestamp, never failures, scores, quarantine state, route paths, proof
    /// history, or traffic counters. Route-surface fingerprint matching
    /// prevents evidence from surviving endpoint, capability, KEM key, or
    /// routing-policy changes while allowing a newer signed sequence/TTL.
    /// Legacy exact-descriptor cache records remain accepted. Startup direct
    /// probes must still run.
    pub fn restore_routeability_cache_evidence(
        &self,
        records: &[PeerStoreRouteabilityCacheEvidence],
        now: u64,
    ) -> PeerStoreRouteabilityCacheRestoreReport {
        let mut restored = 0usize;
        let mut rejected = records
            .len()
            .saturating_sub(PEER_ROUTEABILITY_CACHE_MAX_ENTRIES);

        for record in records.iter().take(PEER_ROUTEABILITY_CACHE_MAX_ENTRIES) {
            let mut node_id = [0u8; 32];
            let supported_evidence_kind = matches!(
                record.evidence_kind.as_str(),
                ROUTEABILITY_EVIDENCE_KIND_EXACT_DESCRIPTOR
                    | ROUTEABILITY_EVIDENCE_KIND_ROUTE_SURFACE
            );
            if !supported_evidence_kind
                || hex::decode_to_slice(&record.node_id_hex, &mut node_id).is_err()
                || record.last_success_at > now
                || now.saturating_sub(record.last_success_at) > PEER_ROUTEABILITY_STALE_AFTER_SECS
            {
                rejected = rejected.saturating_add(1);
                continue;
            }

            // [ROUTEABILITY-RESTORE-RACE 2026-09-25 by Codex] Keep the current
            // surface pinned until its evidence is written.
            let peers = self.peers.read();
            let Some(descriptor) = peers.get(&node_id) else {
                rejected = rejected.saturating_add(1);
                continue;
            };
            let route_surface_fingerprint =
                Self::descriptor_routeability_surface_fingerprint(&descriptor);
            let binding_matches = match record.evidence_kind.as_str() {
                ROUTEABILITY_EVIDENCE_KIND_EXACT_DESCRIPTOR => {
                    descriptor.sequence() == record.descriptor_sequence
                        && descriptor.descriptor.issued_at <= record.last_success_at
                        && Self::descriptor_routeability_fingerprint(&descriptor).is_some_and(
                            |fingerprint| fingerprint == record.descriptor_fingerprint_sha256,
                        )
                }
                ROUTEABILITY_EVIDENCE_KIND_ROUTE_SURFACE => {
                    descriptor.sequence() >= record.descriptor_sequence
                        && route_surface_fingerprint
                            .as_ref()
                            .is_some_and(|fingerprint| {
                                fingerprint == &record.descriptor_fingerprint_sha256
                            })
                }
                _ => false,
            };
            let descriptor_matches = descriptor.verify_at(now).is_ok()
                && route_surface_fingerprint.is_some()
                && binding_matches;
            if !descriptor_matches {
                rejected = rejected.saturating_add(1);
                continue;
            }

            let mut route_health = self.route_health.write();
            let health = route_health.entry(node_id).or_default();
            let conflicts_with_newer_runtime_state = health
                .last_failure_at
                .is_some_and(|failure_at| failure_at >= record.last_success_at)
                || Self::route_quarantine_remaining_seconds(health, now).is_some();
            if conflicts_with_newer_runtime_state {
                rejected = rejected.saturating_add(1);
                continue;
            }
            if health
                .last_success_at
                .is_none_or(|last_success_at| record.last_success_at > last_success_at)
            {
                health.last_success_at = Some(record.last_success_at);
            }
            health.last_success_route_fingerprint_sha256 = route_surface_fingerprint;
            health.success_count = health.success_count.max(1);
            restored = restored.saturating_add(1);
        }

        let report = PeerStoreRouteabilityCacheRestoreReport {
            total: records.len(),
            restored,
            rejected,
        };
        let status_bucket = if records.is_empty() {
            "empty"
        } else if restored == records.len() {
            "restored"
        } else if restored > 0 {
            "partial"
        } else {
            "rejected"
        };
        {
            let mut status = self.bootstrap_status.write();
            status.last_routeability_cache_status = Some(status_bucket.to_string());
            status.last_routeability_cache_restored = restored as u64;
            status.last_routeability_cache_rejected = rejected as u64;
            status.last_routeability_cache_at = Some(now);
        }
        self.record_audit_event(
            now,
            "routeability_cache_restore",
            if restored > 0 || records.is_empty() {
                "accepted"
            } else {
                "rejected"
            },
            format!(
                "schema_version={} total={} restored={} rejected={} status={status_bucket}",
                ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION,
                records.len(),
                restored,
                rejected
            ),
        );
        report
    }

    /// Records a cache-level routeability authentication rejection while
    /// allowing independently signed peer descriptors to remain recoverable.
    ///
    /// The reason is normalized to a fixed bucket so parser or key details do
    /// not enter operator telemetry. No routeability state is restored.
    pub fn reject_routeability_cache_evidence(
        &self,
        total: usize,
        now: u64,
        reason: &str,
    ) -> PeerStoreRouteabilityCacheRestoreReport {
        let reason_bucket = Self::route_cache_rejection_reason_bucket(reason);
        {
            let mut status = self.bootstrap_status.write();
            status.last_routeability_cache_status = Some("rejected".to_string());
            status.last_routeability_cache_restored = 0;
            status.last_routeability_cache_rejected = total as u64;
            status.last_routeability_cache_at = Some(now);
        }
        self.record_audit_event(
            now,
            "routeability_cache_restore",
            "rejected",
            format!(
                "schema_version={} total={} restored=0 rejected={} status=rejected reason={reason_bucket}",
                ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION,
                total,
                total
            ),
        );
        PeerStoreRouteabilityCacheRestoreReport {
            total,
            restored: 0,
            rejected: total,
        }
    }

    /// Admits only fixed cache authentication/rollback reasons into local
    /// audit telemetry.
    ///
    /// [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Anchor decisions are
    /// coarse policy buckets. Never attach cache paths, digests, signatures,
    /// peer identities, routes, endpoints, or parser errors to these values.
    fn route_cache_rejection_reason_bucket(reason: &str) -> &str {
        match reason {
            "identity_unavailable"
            | "signature_invalid"
            | "legacy_unanchored"
            | "anchor_missing"
            | "anchor_invalid"
            | "anchor_conflict"
            | "rollback_detected" => reason,
            _ => "unknown",
        }
    }
}
