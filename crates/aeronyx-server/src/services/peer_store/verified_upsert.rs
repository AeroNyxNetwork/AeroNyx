// [ARCH-SPLIT 2026-10-02]
// Verified descriptor import and discovery-message application.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Verifies and inserts or refreshes a descriptor.
    ///
    /// Returns `Ok(true)` when the store changed, `Ok(false)` when the same
    /// sequence was already present, and an error when the descriptor is
    /// invalid or would roll the node backward to an older sequence.
    pub fn upsert_verified(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        self.upsert_verified_from_source(descriptor, now, "unknown")
    }

    /// Verifies and inserts or refreshes a descriptor with a source bucket.
    ///
    /// Source buckets are intentionally coarse (`cache`, `gossip_snapshot`,
    /// `self`, etc.). They let nodeboard distinguish restart recovery from
    /// live discovery without exposing peer URLs or transport metadata.
    pub fn upsert_verified_from_source(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: impl Into<String>,
    ) -> Result<bool, PeerStoreError> {
        let _tls_directory = IdentityTlsDirectoryRefresh(self);
        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let incoming_route_fingerprint =
            Self::descriptor_routeability_surface_fingerprint(&descriptor);
        let source = source.into();
        if descriptor.verify_at(now).is_err() {
            self.record_peer_event(
                now,
                "peer_rejected",
                "rejected",
                source,
                &node_id,
                Some(incoming_sequence),
                Some("verification_failed"),
            );
            return Err(PeerStoreError::VerificationFailed);
        }

        let mut peers = self.peers.write();
        let mut route_surface_changed = false;

        let is_existing_peer = if let Some(existing) = peers.get(&node_id) {
            let current = existing.sequence();
            if incoming_sequence < current {
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("stale_sequence"),
                );
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
            if incoming_sequence == current {
                if existing != &descriptor {
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
                drop(peers);
                self.record_peer_runtime(&descriptor, now, source.clone(), false);
                self.record_peer_event(
                    now,
                    "peer_refreshed",
                    "ignored",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("same_sequence"),
                );
                return Ok(false);
            }
            route_surface_changed = Self::descriptor_routeability_surface_fingerprint(existing)
                != incoming_route_fingerprint;
            true
        } else {
            false
        };

        if let Some(max_peers) = *self.max_peers.read() {
            if !is_existing_peer && peers.len() >= max_peers {
                self.counters
                    .capacity_rejected
                    .fetch_add(1, Ordering::Relaxed);
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("capacity_exceeded"),
                );
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        peers.insert(node_id, descriptor.clone());
        drop(peers);
        if route_surface_changed {
            self.invalidate_route_success_for_surface_change(&node_id, now);
        }
        self.record_peer_runtime(&descriptor, now, source.clone(), true);
        self.record_peer_event(
            now,
            if is_existing_peer {
                "peer_upgraded"
            } else {
                "peer_inserted"
            },
            "accepted",
            source,
            &node_id,
            Some(incoming_sequence),
            None,
        );
        Ok(true)
    }

    /// Invalidates positive route and receipt evidence after a signed
    /// route-surface change. Failure and quarantine history is retained so
    /// descriptor rotation cannot evade local abuse or failure isolation.
    ///
    /// [RECEIPT-EVIDENCE-LIFECYCLE 2026-08-10 by Codex] Purpose-bound receipt
    /// evidence proves one concrete endpoint/KEM/capability surface. Carrying
    /// it across a signed surface rotation would let a merely reachable new
    /// endpoint inherit cryptographic authority established by the old one.
    pub(super) fn invalidate_route_success_for_surface_change(&self, node_id: &[u8; 32], now: u64) {
        let routeability_invalidated = {
            let mut route_health = self.route_health.write();
            if let Some(health) = route_health.get_mut(node_id) {
                let invalidated = health.last_success_at.take().is_some()
                    || health
                        .last_success_route_fingerprint_sha256
                        .take()
                        .is_some();
                if invalidated {
                    health.success_count = 0;
                }
                invalidated
            } else {
                false
            }
        };
        let receipt_authority_invalidated = self
            .purpose_bound_delivery_receipt_capability
            .write()
            .remove(node_id)
            .is_some();

        if routeability_invalidated || receipt_authority_invalidated {
            self.record_audit_event(
                now,
                "routeability_surface_changed",
                "invalidated",
                format!(
                    "node_prefix={} result=reprobe_required routeability_invalidated={} receipt_authority_invalidated={}",
                    hex::encode(&node_id[..4]),
                    routeability_invalidated,
                    receipt_authority_invalidated,
                ),
            );
        }
    }

    pub(super) fn record_peer_runtime(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        source: String,
        inserted_or_upgraded: bool,
    ) {
        let node_id = descriptor.node_id();
        let mut metadata = self.peer_runtime.write();
        let entry = metadata
            .entry(node_id)
            .or_insert_with(|| PeerRuntimeMetadata {
                source: source.clone(),
                first_seen_at: now,
                last_seen_at: now,
                last_sequence: descriptor.sequence(),
                imported_count: 0,
                expired_degraded_at: None,
            });
        entry.source = source;
        entry.last_seen_at = now;
        entry.last_sequence = descriptor.sequence();
        entry.expired_degraded_at = None;
        if inserted_or_upgraded {
            entry.imported_count = entry.imported_count.saturating_add(1);
        }
    }

    /// Imports all descriptors from a validated bootstrap snapshot.
    ///
    /// Invalid descriptors are counted and skipped. This lets a node keep using
    /// the healthy part of a bootstrap snapshot while surfacing corruption or
    /// expiry through the returned report.
    pub fn load_bootstrap_snapshot(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
    ) -> PeerStoreImportReport {
        self.load_bootstrap_snapshot_from_source(snapshot, now, "unknown")
    }

    /// Imports descriptors from a snapshot and tags runtime metadata source.
    pub fn load_bootstrap_snapshot_from_source(
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
            match self.upsert_verified_from_source(descriptor.clone(), now, source.clone()) {
                Ok(true) => report.inserted += 1,
                Ok(false) => report.unchanged += 1,
                Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                Err(_) => report.rejected += 1,
            }
        }

        self.record_import_report(&report, now);
        report
    }

    /// Applies a discovery gossip message to this store.
    ///
    /// Snapshot requests are read-only and return an empty report; callers can
    /// use `build_snapshot_response()` to construct the actual response.
    pub fn apply_discovery_message(
        &self,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        if self
            .untrusted_discovery_candidate_mode
            .load(Ordering::Acquire)
        {
            // [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] An
            // anonymous announce may refresh only an identity already held in
            // the receiver's locally established peer cache. This preserves
            // the pinned-witness reboot preflight without admitting a new
            // self-signed identity to live routing. The same receiver-local
            // lifetime limits apply before the normal verified upsert.
            if let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = message {
                if Self::untrusted_candidate_is_within_limits(descriptor, now)
                    && self.has_locally_established_identity(&descriptor.node_id())
                {
                    return self.apply_verified_descriptor_from_source(
                        descriptor.clone(),
                        now,
                        "local_identity_refresh",
                    );
                }
            }
            return self.apply_untrusted_discovery_message(message, now);
        }
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => PeerStoreImportReport::empty(),
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => {
                self.load_bootstrap_snapshot_from_source(snapshot, now, "gossip_snapshot")
            }
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => self
                .apply_verified_descriptor_from_source(descriptor.clone(), now, "gossip_announce"),
            // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] PeerStore has
            // no Directory replica trust anchor. Direct callers therefore fail
            // closed; the discovery API may use the dedicated locally anchored
            // admission function before calling the shared descriptor path.
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                self.record_rejected_directory_proof_import(now)
            }
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                Self::rejected_endpoint_attestation_report()
            }
        }
    }

    /// Admits legacy gossip as a bounded, non-routeable candidate only.
    ///
    /// [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] A valid
    /// self-signature proves control of one key, not endpoint possession,
    /// independent operation, or routing authority. This boundary must remain
    /// separate from `upsert_verified_from_source`, which is reserved for
    /// self/local cache or independently anchored imports.
    pub(crate) fn apply_untrusted_discovery_message(
        &self,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        let descriptors: Vec<SignedNodeDescriptor> = match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => return PeerStoreImportReport::empty(),
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => vec![descriptor.clone()],
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => snapshot.peers.clone(),
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                return self.record_rejected_directory_proof_import(now);
            }
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                return Self::rejected_endpoint_attestation_report();
            }
        };

        let admitted_limit = descriptors
            .len()
            .min(UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE);
        let mut report = PeerStoreImportReport {
            total: descriptors.len(),
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: descriptors.len().saturating_sub(admitted_limit),
        };
        for descriptor in descriptors.into_iter().take(admitted_limit) {
            match self.admit_untrusted_candidate(descriptor, now) {
                CandidateAdmissionOutcome::Candidate => report.candidates += 1,
                CandidateAdmissionOutcome::Unchanged => report.unchanged += 1,
                CandidateAdmissionOutcome::Stale => report.stale += 1,
                CandidateAdmissionOutcome::Conflict
                | CandidateAdmissionOutcome::Saturated
                | CandidateAdmissionOutcome::Rejected => report.rejected += 1,
            }
        }
        self.record_import_report(&report, now);
        report
    }

    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] PeerStore is not
    // an attestation verifier or promotion authority. Direct callers receive
    // a coarse rejection report without mutating peer, audit, or counters.
    pub(super) fn rejected_endpoint_attestation_report() -> PeerStoreImportReport {
        PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 1,
        }
    }

    /// Applies one descriptor through the normal verification, capacity, and
    /// anti-rollback path while producing the same aggregate import contract as
    /// snapshot and legacy gossip ingestion.
    ///
    /// This is crate-visible for the Directory-authenticated admission boundary.
    /// Callers must complete their own trust-anchor checks before invoking it.
    pub(crate) fn apply_verified_descriptor_from_source(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: &'static str,
    ) -> PeerStoreImportReport {
        let mut report = PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };
        match self.upsert_verified_from_source(descriptor, now, source) {
            Ok(true) => report.inserted = 1,
            Ok(false) => report.unchanged = 1,
            Err(PeerStoreError::StaleSequence { .. }) => report.stale = 1,
            Err(_) => report.rejected = 1,
        }
        self.record_import_report(&report, now);
        report
    }

    /// Records one fail-closed proof-gossip rejection without retaining any
    /// producer, descriptor, block, endpoint, route, or sender identity.
    pub(crate) fn record_rejected_directory_proof_import(&self, now: u64) -> PeerStoreImportReport {
        let report = PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 1,
        };
        self.record_import_report(&report, now);
        report
    }

    pub(super) fn record_import_report(&self, report: &PeerStoreImportReport, now: u64) {
        if report.total == 0 {
            return;
        }

        self.counters
            .total_imported
            .fetch_add(report.total as u64, Ordering::Relaxed);
        self.counters
            .inserted
            .fetch_add(report.inserted as u64, Ordering::Relaxed);
        self.counters
            .candidate_admitted
            .fetch_add(report.candidates as u64, Ordering::Relaxed);
        self.counters
            .unchanged
            .fetch_add(report.unchanged as u64, Ordering::Relaxed);
        self.counters
            .stale
            .fetch_add(report.stale as u64, Ordering::Relaxed);
        self.counters
            .rejected
            .fetch_add(report.rejected as u64, Ordering::Relaxed);
        self.counters.last_import_at.store(now, Ordering::Relaxed);

        let outcome = if report.rejected > 0 || report.stale > 0 {
            "warning"
        } else if report.inserted > 0 || report.candidates > 0 || report.unchanged > 0 {
            "accepted"
        } else {
            "ignored"
        };
        self.record_audit_event(
            now,
            "descriptor_import",
            outcome,
            format!(
                "total={} inserted={} candidates={} unchanged={} stale={} rejected={}",
                report.total,
                report.inserted,
                report.candidates,
                report.unchanged,
                report.stale,
                report.rejected
            ),
        );
    }
}
