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
        self.upsert_verified_with_origin(descriptor, now, source, PeerOrigin::Other, None)
    }

    /// W1, the write boundary of every verified insert or refresh.
    ///
    /// [PINNED-SEEDS 2026-10-11 by Claude] `origin` and `proof` take part in
    /// the pin rules, which run inside the `peers` write lock before any other
    /// decision, so "already established?" and the insert are one critical
    /// section. `source` stays a diagnostic label only.
    pub(crate) fn upsert_verified_with_origin(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: impl Into<String>,
        origin: PeerOrigin,
        proof: Option<&PinnedTlsProof>,
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
        #[cfg(test)]
        pinned_seeds::test_pause_point(&node_id);
        // [HELD-IDENTITY-REFRESH 2026-10-15 by Claude] "Is it held?" is
        // decided here, in the same critical section as the write: an
        // identity removed after a caller classified it is not re-created.
        if origin == PeerOrigin::HeldRefresh && !peers.contains_key(&node_id) {
            drop(peers);
            self.record_peer_event(
                now,
                "peer_rejected",
                "rejected",
                source,
                &node_id,
                Some(incoming_sequence),
                Some("held_refresh_not_held"),
            );
            return Err(PeerStoreError::VerificationFailed);
        }
        if let Err(rejection) = self.pin_admission(&peers, &descriptor, origin, proof) {
            drop(peers);
            self.record_peer_event(
                now,
                "peer_rejected",
                "rejected",
                source,
                &node_id,
                Some(incoming_sequence),
                Some(rejection.reason()),
            );
            return Err(PeerStoreError::VerificationFailed);
        }
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
                    return self.apply_held_refresh(descriptor.clone(), now);
                }
            }
            if let NodeDiscoveryMessage::SnapshotResponse { snapshot } = message {
                return self.apply_untrusted_snapshot_with_held_refresh(snapshot, now);
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

        let report = self.untrusted_candidates_report(descriptors, now);
        self.record_import_report(&report, now);
        report
    }

    /// The candidate admission of `apply_untrusted_discovery_message`
    /// without recording it, so a caller that merges several parts of one
    /// message records the whole result exactly once.
    pub(super) fn untrusted_candidates_report(
        &self,
        descriptors: Vec<SignedNodeDescriptor>,
        now: u64,
    ) -> PeerStoreImportReport {
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
        report
    }

    /// [HELD-IDENTITY-REFRESH 2026-10-15 by Claude] One refresh of an
    /// identity this store holds, through W1 with `PeerOrigin::HeldRefresh`
    /// (refused inside the write lock if the identity is no longer held),
    /// recorded as one import.
    pub(crate) fn apply_held_refresh(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> PeerStoreImportReport {
        let report = self.held_refresh_unrecorded(descriptor, now);
        self.record_import_report(&report, now);
        report
    }

    fn held_refresh_unrecorded(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> PeerStoreImportReport {
        #[cfg(test)]
        test_pause_before_held_refresh(&descriptor.node_id());
        let mut report = PeerStoreImportReport {
            total: 1,
            ..PeerStoreImportReport::empty()
        };
        match self.upsert_verified_with_origin(
            descriptor,
            now,
            "local_identity_refresh",
            PeerOrigin::HeldRefresh,
            None,
        ) {
            Ok(true) => report.inserted = 1,
            Ok(false) => report.unchanged = 1,
            Err(PeerStoreError::StaleSequence { .. }) => report.stale = 1,
            Err(_) => report.rejected = 1,
        }
        report
    }

    /// [HELD-IDENTITY-REFRESH 2026-10-15 by Claude] Untrusted-mode snapshot
    /// ingestion that can also refresh identities this store already holds.
    ///
    /// Scope of the claim: for an identity that is established here and still
    /// held when W1 takes the write lock, a fully verified newer copy of its
    /// own signed descriptor is as authentic relayed as announced - the
    /// announce path is itself anonymous. It does NOT make snapshots trusted:
    /// a descriptor of any identity not held here stays a bounded,
    /// non-routeable candidate, exactly as before.
    ///
    /// Work per message: at most `UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE`
    /// descriptors are looked at; the rest are counted rejected without being
    /// verified, as before. A held identity's descriptor is verified once
    /// while classifying (`verify_at` inside the receiver limits) and its
    /// single winner once more inside W1; candidates are admitted as before.
    /// Nothing is copied beyond those descriptors.
    ///
    /// Rules among the descriptors that are looked at (the first
    /// `UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE`), each independent of
    /// their order. Reordering a longer message can change WHICH
    /// descriptors are looked at, and so the outcome:
    /// - a held descriptor is verified BEFORE it can compete, so an invalid
    ///   high sequence cannot hide a valid lower one;
    /// - exact copies collapse into one;
    /// - two different valid descriptors at the highest sequence: none is
    ///   applied for that identity in this message (the stored one stays);
    /// - the winner goes through W1 as `HeldRefresh`.
    ///
    /// The whole message is recorded exactly once, and
    /// inserted + candidates + unchanged + stale + rejected == total.
    pub(crate) fn apply_untrusted_snapshot_with_held_refresh(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
    ) -> PeerStoreImportReport {
        let considered = snapshot
            .peers
            .len()
            .min(UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE);
        let mut report = PeerStoreImportReport {
            total: snapshot.peers.len(),
            rejected: snapshot.peers.len() - considered,
            ..PeerStoreImportReport::empty()
        };
        let mut held: std::collections::BTreeMap<[u8; 32], Vec<SignedNodeDescriptor>> =
            std::collections::BTreeMap::new();
        let mut rest = Vec::new();
        for descriptor in snapshot.peers.iter().take(considered) {
            let node_id = descriptor.node_id();
            if !self.has_locally_established_identity(&node_id) {
                rest.push(descriptor.clone());
            } else if Self::untrusted_candidate_is_within_limits(descriptor, now) {
                held.entry(node_id).or_default().push(descriptor.clone());
            } else {
                report.rejected += 1;
            }
        }
        for (node_id, copies) in held {
            let top = copies
                .iter()
                .map(SignedNodeDescriptor::sequence)
                .max()
                .unwrap_or(0);
            let mut winners: Vec<&SignedNodeDescriptor> = Vec::new();
            for copy in copies.iter().filter(|copy| copy.sequence() == top) {
                if !winners.contains(&copy) {
                    winners.push(copy);
                }
            }
            if winners.len() != 1 {
                report.rejected += copies.len();
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    "local_identity_refresh".to_string(),
                    &node_id,
                    Some(top),
                    Some("held_refresh_conflict"),
                );
                continue;
            }
            let one = self.held_refresh_unrecorded(winners[0].clone(), now);
            // Lower or duplicate copies are superseded, never applied.
            report.inserted += one.inserted;
            report.stale += one.stale;
            report.rejected += one.rejected;
            report.unchanged += one.unchanged + (copies.len() - 1);
        }
        let others = self.untrusted_candidates_report(rest, now);
        report.inserted += others.inserted;
        report.candidates += others.candidates;
        report.unchanged += others.unchanged;
        report.stale += others.stale;
        report.rejected += others.rejected;
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

#[cfg(test)]
type HeldRefreshPause = (std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>);

// Keyed by identity, so tests that run in parallel never take or overwrite
// each other's pause.
#[cfg(test)]
static PAUSE_BEFORE_HELD_REFRESH: std::sync::LazyLock<
    parking_lot::Mutex<std::collections::HashMap<[u8; 32], HeldRefreshPause>>,
> = std::sync::LazyLock::new(|| parking_lot::Mutex::new(std::collections::HashMap::new()));

/// How long a paused writer waits to be released. This bounds the pause
/// only, not the writer's later lock acquisition or the whole thread.
#[cfg(test)]
pub(crate) const HELD_REFRESH_PAUSE_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);

/// [HELD-IDENTITY-REFRESH 2026-10-15 by Claude] Arms a one-shot pause after a
/// caller classified `node_id` as held and before W1 takes the write lock.
/// Returns `(entered, release)`: the writer signals `entered` when paused and
/// waits on `release` at most `HELD_REFRESH_PAUSE_LIMIT`; dropping `release`
/// frees it at once.
#[cfg(test)]
pub(crate) fn arm_pause_before_held_refresh(
    node_id: [u8; 32],
) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    PAUSE_BEFORE_HELD_REFRESH
        .lock()
        .insert(node_id, (entered_tx, release_rx));
    (entered_rx, release_tx)
}

/// Removes an armed pause. Its `entered` sender is dropped, so a waiting test
/// sees a disconnect at once - the same thing it sees when a writer took the
/// pause and died before signalling.
#[cfg(test)]
pub(crate) fn disarm_pause_before_held_refresh(node_id: &[u8; 32]) {
    PAUSE_BEFORE_HELD_REFRESH.lock().remove(node_id);
}

#[cfg(test)]
fn test_pause_before_held_refresh(node_id: &[u8; 32]) {
    let armed = PAUSE_BEFORE_HELD_REFRESH.lock().remove(node_id);
    if let Some((entered, release)) = armed {
        let _ = entered.send(());
        let _ = release.recv_timeout(HELD_REFRESH_PAUSE_LIMIT);
    }
}
