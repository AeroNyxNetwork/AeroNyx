// [ARCH-SPLIT 2026-10-02]
// Route success, failure, and verified delivery evidence.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Runs one evidence write only while the caller's exact signed route
    /// surface remains current in this store.
    ///
    /// [ROUTE-OBSERVATION-SURFACE-BINDING 2026-08-10 by Codex] The peer read
    /// lock is intentionally held through `observe`. A concurrent descriptor
    /// upgrade must therefore complete before this comparison (and reject an
    /// obsolete observation) or afterwards (and invalidate the observation).
    pub(super) fn with_current_verified_route_surface<R>(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        observe: impl FnOnce([u8; 32], String) -> R,
    ) -> Result<R, &'static str> {
        let node_id = descriptor.node_id();
        let expected_fingerprint = descriptor
            .verify_at(now)
            .ok()
            .and_then(|_| Self::descriptor_routeability_surface_fingerprint(descriptor))
            .ok_or("invalid_expected_route_surface")?;

        let peers = self.peers.read();
        let current_fingerprint = peers
            .get(&node_id)
            .filter(|current| current.verify_at(now).is_ok())
            .and_then(Self::descriptor_routeability_surface_fingerprint);
        if current_fingerprint.as_deref() != Some(expected_fingerprint.as_str()) {
            return Err("route_surface_mismatch");
        }

        Ok(observe(node_id, expected_fingerprint))
    }

    /// Records that one exact signed peer route surface participated in a
    /// route carrying a valid purpose-bound version-2 terminal receipt.
    ///
    /// The node id is process-private selection state. Public status exposes
    /// only a fresh aggregate count; no route id, endpoint, receipt, payload
    /// commitment, sender, receiver, message id, or social-graph edge is kept.
    ///
    /// [RECEIPT-EVIDENCE-SURFACE-BINDING 2026-08-10 by Codex] The caller must
    /// pass the descriptor actually used by the verified route. The write is
    /// accepted only while that route surface is still current in this store.
    #[must_use]
    pub fn record_purpose_bound_delivery_receipt_capability_for_descriptor(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        let result = self.with_current_verified_route_surface(
            descriptor,
            now,
            |node_id, route_surface_fingerprint_sha256| {
                self.purpose_bound_delivery_receipt_capability
                    .write()
                    .insert(
                        node_id,
                        PurposeBoundDeliveryReceiptEvidence {
                            observed_at: now,
                            route_surface_fingerprint_sha256,
                        },
                    );
            },
        );
        if let Err(reason) = result {
            self.record_audit_event(
                now,
                "blind_relay_purpose_bound_receipt_capability",
                "rejected",
                reason.to_string(),
            );
            return false;
        }
        self.record_audit_event(
            now,
            "blind_relay_purpose_bound_receipt_capability",
            "accepted",
            "fresh_purpose_bound_delivery_receipt_v2".to_string(),
        );
        true
    }

    /// Backward-compatible node-id recorder. New route code must call
    /// [`Self::record_purpose_bound_delivery_receipt_capability_for_descriptor`]
    /// with the exact descriptor that carried the verified receipt.
    pub fn record_purpose_bound_delivery_receipt_capability(&self, node_id: &[u8; 32], now: u64) {
        let Some(descriptor) = self.get_valid(node_id, now) else {
            return;
        };
        let _ =
            self.record_purpose_bound_delivery_receipt_capability_for_descriptor(&descriptor, now);
    }

    /// Backward-compatible alias for callers compiled against the v1 method
    /// name. Callers must pass only evidence that has already satisfied the v2
    /// purpose-bound verifier; this method does not inspect receipt bytes.
    pub fn record_delivery_receipt_capability(&self, node_id: &[u8; 32], now: u64) {
        self.record_purpose_bound_delivery_receipt_capability(node_id, now);
    }

    pub(super) fn purpose_bound_delivery_receipt_evidence_matches_descriptor(
        evidence: &PurposeBoundDeliveryReceiptEvidence,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        Self::purpose_bound_delivery_receipt_evidence_is_fresh(evidence.observed_at, now)
            && Self::descriptor_routeability_surface_fingerprint(descriptor)
                .is_some_and(|fingerprint| fingerprint == evidence.route_surface_fingerprint_sha256)
    }

    /// [PURPOSE-BOUND-RECEIPT-EVIDENCE 2026-08-10 by Codex] Returns whether
    /// this process has fresh cryptographic v2 evidence for a currently valid
    /// peer. This is stronger than an unsigned discovery hint and is
    /// intentionally cleared by process restart.
    #[must_use]
    pub fn has_fresh_purpose_bound_delivery_receipt_capability(
        &self,
        node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        let peers = self.peers.read();
        let Some(descriptor) = peers
            .get(node_id)
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
        else {
            return false;
        };
        self.purpose_bound_delivery_receipt_capability
            .read()
            .get(node_id)
            .is_some_and(|evidence| {
                Self::purpose_bound_delivery_receipt_evidence_matches_descriptor(
                    evidence, descriptor, now,
                )
            })
    }

    /// Counts only fresh v2 evidence attached to descriptors that are valid at
    /// the same snapshot time. Peer IDs stay process-local and are discarded
    /// before this aggregate reaches any API or heartbeat contract.
    pub(super) fn fresh_purpose_bound_delivery_receipt_peer_count(&self, now: u64) -> usize {
        // [RECEIPT-EVIDENCE-LIFECYCLE 2026-08-10 by Codex] Lock peers before
        // evidence, matching write-side lifecycle order. Iterate only the
        // usually smaller evidence map and avoid allocating identity copies.
        let peers = self.peers.read();
        let capability_evidence = self.purpose_bound_delivery_receipt_capability.read();
        capability_evidence
            .iter()
            .filter(|(node_id, evidence)| {
                peers.get(*node_id).is_some_and(|descriptor| {
                    descriptor.verify_at(now).is_ok()
                        && Self::purpose_bound_delivery_receipt_evidence_matches_descriptor(
                            evidence, descriptor, now,
                        )
                })
            })
            .count()
    }

    /// Records one authenticated App/client-originated onion delivery whose
    /// terminal receipt was signature-, route-, payload-, and freshness-checked.
    ///
    /// This deliberately records only an aggregate count and timestamp. It
    /// never stores receipt bytes, route ids, peer ids, payload commitments,
    /// sender/receiver keys, message ids, endpoints, or ciphertext.
    pub(super) fn record_verified_client_onion_delivery_aggregate(&self, now: u64) {
        self.counters
            .verified_client_onion_deliveries
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_verified_client_onion_delivery_at
            .store(now, Ordering::Relaxed);
        self.mark_peer_cache_dirty();
    }

    /// Runs one evidence transition while a complete signed route is current.
    ///
    /// [ATOMIC-MULTIHOP-PROOF-EVIDENCE 2026-08-11 by Codex] Every hop is
    /// role-checked and fingerprinted before the peer snapshot is locked. The
    /// snapshot remains locked through `commit`, so descriptor rotation can
    /// occur either before the comparison (and reject stale evidence) or after
    /// the entire transition. Callers preserve lock order by taking only
    /// route-health, receipt-capability, and aggregate locks inside `commit`.
    pub(super) fn with_current_verified_route_path<R>(
        &self,
        route: &[(&SignedNodeDescriptor, NodeCapability)],
        now: u64,
        commit: impl FnOnce(&[([u8; 32], String)]) -> R,
    ) -> Result<(Vec<[u8; 32]>, R), &'static str> {
        if route.len() < 2 {
            return Err("insufficient_route_hops");
        }

        let mut seen_node_ids = HashSet::with_capacity(route.len());
        let mut expected_surfaces = Vec::with_capacity(route.len());
        for (descriptor, required_capability) in route {
            let node_id = descriptor.node_id();
            if !seen_node_ids.insert(node_id) {
                return Err("duplicate_route_hop");
            }
            if !descriptor
                .descriptor
                .capabilities
                .contains(required_capability)
            {
                return Err("route_role_mismatch");
            }
            let fingerprint = descriptor
                .verify_at(now)
                .ok()
                .and_then(|_| Self::descriptor_routeability_surface_fingerprint(descriptor))
                .ok_or("invalid_expected_route_surface")?;
            expected_surfaces.push((node_id, fingerprint));
        }

        let committed = {
            let peers = self.peers.read();
            let surfaces_are_current =
                expected_surfaces
                    .iter()
                    .all(|(node_id, expected_fingerprint)| {
                        peers
                            .get(node_id)
                            .filter(|descriptor| descriptor.verify_at(now).is_ok())
                            .and_then(Self::descriptor_routeability_surface_fingerprint)
                            .is_some_and(|current| current == *expected_fingerprint)
                    });
            if !surfaces_are_current {
                return Err("route_surface_mismatch");
            }

            commit(&expected_surfaces)
        };

        let node_ids = expected_surfaces
            .into_iter()
            .map(|(node_id, _)| node_id)
            .collect();
        Ok((node_ids, committed))
    }

    /// Commits route-health and receipt-capability evidence for one complete
    /// signed path, then publishes the caller's aggregate success marker.
    ///
    /// The peer snapshot is held through all positive writes. Lock order is
    /// peers -> route health -> receipt capability -> aggregate.
    pub(super) fn commit_verified_route_delivery_evidence<R>(
        &self,
        route: &[(&SignedNodeDescriptor, NodeCapability)],
        now: u64,
        publish: impl FnOnce() -> R,
    ) -> Result<(Vec<[u8; 32]>, R), &'static str> {
        let result = self.with_current_verified_route_path(route, now, |expected_surfaces| {
            let mut route_health = self.route_health.write();
            for (node_id, fingerprint) in expected_surfaces {
                let health = route_health.entry(*node_id).or_default();
                health.success_count = health.success_count.saturating_add(1);
                health.consecutive_failures = 0;
                health.last_success_at = Some(now);
                health.last_success_route_fingerprint_sha256 = Some(fingerprint.clone());
                health.quarantine_until = None;
            }

            let mut receipt_capability = self.purpose_bound_delivery_receipt_capability.write();
            for (node_id, fingerprint) in expected_surfaces {
                receipt_capability.insert(
                    *node_id,
                    PurposeBoundDeliveryReceiptEvidence {
                        observed_at: now,
                        route_surface_fingerprint_sha256: fingerprint.clone(),
                    },
                );
            }

            // Publish last: readers may observe an older conservative state,
            // never a success marker without its supporting route evidence.
            publish()
        });
        if result.is_ok() {
            // [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] A verified path
            // both creates positive routeability evidence and can clear a
            // restored quarantine. Persist that transition promptly instead
            // of waiting for the periodic cache interval.
            self.mark_peer_cache_dirty();
        }
        result
    }

    /// Records a legacy two-hop control-plane ACK without upgrading it to
    /// terminal-delivery or receipt-capability evidence.
    ///
    /// [LEGACY-CONTROL-PROOF-SURFACE-BINDING 2026-08-11 by Codex] The exact
    /// middle and terminal descriptors carried by the request must still be
    /// current together. Only the directly observed middle receives route
    /// health credit; the unsigned terminal claim never receives delivery
    /// receipt capability and cannot unlock authenticated App routing.
    #[must_use]
    pub fn record_verified_two_hop_control_probe(
        &self,
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
    ) -> bool {
        let middle_node_id = middle.node_id();
        let result = self.with_current_verified_route_path(
            &[
                (middle, NodeCapability::OnionMiddle),
                (terminal, NodeCapability::ChatRelay),
            ],
            now,
            |expected_surfaces| {
                let (middle_node_id, middle_fingerprint) = &expected_surfaces[0];
                let mut route_health = self.route_health.write();
                let health = route_health.entry(*middle_node_id).or_default();
                health.success_count = health.success_count.saturating_add(1);
                health.consecutive_failures = 0;
                health.last_success_at = Some(now);
                health.last_success_route_fingerprint_sha256 = Some(middle_fingerprint.clone());
                health.quarantine_until = None;

                self.record_blind_relay_two_hop_probe_result_with_context(
                    now,
                    true,
                    "legacy_control_forwarded",
                    middle_candidate_count,
                    terminal_candidate_count,
                    2,
                    1,
                );
            },
        );

        let (_, ()) = match result {
            Ok(committed) => committed,
            Err(reason) => {
                self.record_audit_event(
                    now,
                    "blind_relay_control_path_proof_evidence",
                    "rejected",
                    format!("hop_count=2 reason={reason}"),
                );
                return false;
            }
        };
        // [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] The compatibility
        // control probe can clear only the directly observed middle. Its
        // route-health transition still needs the generic peer-cache flush.
        self.mark_peer_cache_dirty();
        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "accepted",
            format!(
                "node_prefix={} result=success",
                hex::encode(&middle_node_id[..4])
            ),
        );
        true
    }

    /// Records a terminal-signed synthetic two-hop delivery proof only while
    /// the complete route remains current at the response observation time.
    #[must_use]
    pub fn record_verified_two_hop_probe_delivery(
        &self,
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
    ) -> bool {
        self.record_verified_synthetic_probe_delivery(
            &[
                (middle, NodeCapability::OnionMiddle),
                (terminal, NodeCapability::ChatRelay),
            ],
            now,
            middle_candidate_count,
            terminal_candidate_count,
            2,
            1,
        )
    }

    /// Records a terminal-signed synthetic three-hop delivery proof only while
    /// all two middle surfaces and the terminal surface remain current.
    #[must_use]
    pub fn record_verified_three_hop_probe_delivery(
        &self,
        first_middle: &SignedNodeDescriptor,
        second_middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
    ) -> bool {
        self.record_verified_synthetic_probe_delivery(
            &[
                (first_middle, NodeCapability::OnionMiddle),
                (second_middle, NodeCapability::OnionMiddle),
                (terminal, NodeCapability::ChatRelay),
            ],
            now,
            middle_candidate_count,
            terminal_candidate_count,
            3,
            2,
        )
    }

    pub(super) fn record_verified_synthetic_probe_delivery(
        &self,
        route: &[(&SignedNodeDescriptor, NodeCapability)],
        now: u64,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
    ) -> bool {
        let hop_count = route.len() as u8;
        let result = self.commit_verified_route_delivery_evidence(route, now, || {
            if hop_count == 3 {
                self.record_blind_relay_three_hop_probe_result_with_context(
                    now,
                    true,
                    "onion_terminal_delivered",
                    middle_candidate_count,
                    terminal_candidate_count,
                    entry_ttl,
                    onward_ttl,
                );
            } else {
                self.record_blind_relay_two_hop_probe_result_with_context(
                    now,
                    true,
                    "onion_terminal_delivered",
                    middle_candidate_count,
                    terminal_candidate_count,
                    entry_ttl,
                    onward_ttl,
                );
            }
        });

        let (node_ids, ()) = match result {
            Ok(committed) => committed,
            Err(reason) => {
                self.record_audit_event(
                    now,
                    "blind_relay_path_proof_evidence",
                    "rejected",
                    format!("hop_count={hop_count} reason={reason}"),
                );
                return false;
            }
        };

        for node_id in node_ids {
            self.record_audit_event(
                now,
                "blind_relay_route_health",
                "accepted",
                format!("node_prefix={} result=success", hex::encode(&node_id[..4])),
            );
            self.record_audit_event(
                now,
                "blind_relay_purpose_bound_receipt_capability",
                "accepted",
                "fresh_purpose_bound_delivery_receipt_v2".to_string(),
            );
        }
        true
    }

    /// Commits one verified two-hop client delivery against a single current
    /// signed route snapshot.
    ///
    /// [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex] A terminal
    /// receipt is useful only if the exact middle and terminal route surfaces
    /// that carried it are still current together. Holding the peer read lock
    /// through every positive evidence write prevents descriptor rotation from
    /// leaving a partial capability pair or an aggregate delivery unsupported
    /// by one coherent signed path. Lock order remains peers -> route health ->
    /// receipt capability, matching surface invalidation and avoiding inversion.
    ///
    /// The receipt bytes, route id, payload commitment, endpoints, sender,
    /// receiver, and message id remain outside this store.
    #[must_use]
    pub fn record_verified_client_onion_route_delivery(
        &self,
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        let result = self.commit_verified_route_delivery_evidence(
            &[
                (middle, NodeCapability::OnionMiddle),
                (terminal, NodeCapability::ChatRelay),
            ],
            now,
            || self.record_verified_client_onion_delivery_aggregate(now),
        );
        let (node_ids, ()) = match result {
            Ok(committed) => committed,
            Err(reason) => {
                self.record_audit_event(
                    now,
                    "blind_relay_client_delivery_receipt",
                    "rejected",
                    reason.to_string(),
                );
                return false;
            }
        };

        for node_id in node_ids {
            self.record_audit_event(
                now,
                "blind_relay_route_health",
                "accepted",
                format!("node_prefix={} result=success", hex::encode(&node_id[..4])),
            );
            self.record_audit_event(
                now,
                "blind_relay_purpose_bound_receipt_capability",
                "accepted",
                "fresh_purpose_bound_delivery_receipt_v2".to_string(),
            );
        }
        self.record_audit_event(
            now,
            "blind_relay_client_delivery_receipt",
            "accepted",
            "terminal_signature_verified".to_string(),
        );
        true
    }

    /// Backward-compatible aggregate recorder for cache and migration tests.
    /// Production relay paths must use
    /// [`Self::record_verified_client_onion_route_delivery`] so delivery,
    /// capability, and route-health evidence share one signed path snapshot.
    pub fn record_verified_client_onion_delivery(&self, now: u64) {
        self.record_verified_client_onion_delivery_aggregate(now);
        self.record_audit_event(
            now,
            "blind_relay_client_delivery_receipt",
            "accepted",
            "terminal_signature_verified".to_string(),
        );
    }

    /// Records successful opaque node-to-node forwarding on one exact signed
    /// route surface.
    ///
    /// [ROUTE-SUCCESS-SURFACE-BINDING 2026-08-10 by Codex] Callers must pass
    /// the descriptor whose endpoint/KEM/capabilities were used by the actual
    /// request. A concurrent route-surface rotation rejects this observation;
    /// sequence/TTL-only descriptor refreshes remain compatible.
    #[must_use]
    pub fn record_route_forward_success_for_descriptor(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        self.record_route_forward_success_with_quarantine_policy(descriptor, now, true)
    }

    pub(super) fn record_route_forward_success_with_quarantine_policy(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        allow_active_quarantine: bool,
    ) -> bool {
        let node_id = descriptor.node_id();
        let result = self.with_current_verified_route_surface(
            descriptor,
            now,
            |observed_node_id, route_fingerprint| {
                let mut route_health = self.route_health.write();
                let health = route_health.entry(observed_node_id).or_default();
                if !allow_active_quarantine
                    && Self::route_quarantine_remaining_seconds(health, now).is_some()
                {
                    return None;
                }
                let cleared_quarantine = health.quarantine_until.take().is_some();
                health.success_count = health.success_count.saturating_add(1);
                health.consecutive_failures = 0;
                health.last_success_at = Some(now);
                health.last_success_route_fingerprint_sha256 = Some(route_fingerprint);
                Some(cleared_quarantine)
            },
        );
        let cleared_quarantine = match result {
            Ok(Some(cleared_quarantine)) => cleared_quarantine,
            Ok(None) => return false,
            Err(reason) => {
                self.record_audit_event(
                    now,
                    "blind_relay_route_health",
                    "rejected",
                    format!(
                        "node_prefix={} result=ignored reason={reason}",
                        hex::encode(&node_id[..4])
                    ),
                );
                return false;
            }
        };
        if cleared_quarantine {
            self.mark_peer_cache_dirty();
        }

        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "accepted",
            format!("node_prefix={} result=success", hex::encode(&node_id[..4])),
        );
        true
    }

    /// Backward-compatible node-id recorder. New route code must call
    /// [`Self::record_route_forward_success_for_descriptor`] with the exact
    /// descriptor whose route surface carried the successful request.
    ///
    /// The key is retained only inside this process for route health scoring.
    /// Public status exposes only a short prefix and aggregate counters. Never
    /// pass route ids, encrypted blobs, client identifiers, endpoint URLs, or
    /// payload-derived details into this method.
    pub fn record_route_forward_success(&self, node_id: &[u8; 32], now: u64) {
        let Some(descriptor) = self.get_valid(node_id, now) else {
            self.record_audit_event(
                now,
                "blind_relay_route_health",
                "rejected",
                format!(
                    "node_prefix={} result=ignored reason=missing_verified_route_surface",
                    hex::encode(&node_id[..4])
                ),
            );
            return;
        };
        let _ = self.record_route_forward_success_for_descriptor(&descriptor, now);
    }

    /// Records failed opaque node-to-node forwarding on one exact signed route
    /// surface.
    ///
    /// [ROUTE-FAILURE-SURFACE-BINDING 2026-08-11 by Codex] Callers must pass the
    /// descriptor whose endpoint/KEM/capabilities were used by the request. If
    /// that route surface rotated while the request was in flight, the stale
    /// failure is rejected instead of penalizing the replacement endpoint.
    /// Existing identity-level failure and quarantine history still survives a
    /// later descriptor rotation, preventing sequence churn from evading local
    /// abuse isolation.
    ///
    /// The reason must be a stable coarse bucket such as `request_failed` or
    /// `http_502`; no endpoint URL, route id, ciphertext, receiver, or client
    /// traffic metadata may be recorded here.
    #[must_use]
    pub fn record_route_forward_failure_for_descriptor(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        reason: impl Into<String>,
    ) -> bool {
        let reason = reason.into();
        let reason = PrivacySafePeerHealthReason::route_failure(&reason);
        let node_id = descriptor.node_id();
        let result = self.with_current_verified_route_surface(
            descriptor,
            now,
            |observed_node_id, _route_fingerprint| {
                self.record_route_forward_failure_for_node(&observed_node_id, now, &reason);
            },
        );
        if let Err(rejection_reason) = result {
            self.record_audit_event(
                now,
                "blind_relay_route_health",
                "rejected",
                format!(
                    "node_prefix={} result=ignored reason={rejection_reason}",
                    hex::encode(&node_id[..4])
                ),
            );
            return false;
        }
        true
    }

    /// Backward-compatible node-id recorder. New route code must call
    /// [`Self::record_route_forward_failure_for_descriptor`] with the exact
    /// descriptor whose route surface carried the failed request.
    pub fn record_route_forward_failure(
        &self,
        node_id: &[u8; 32],
        now: u64,
        reason: impl Into<String>,
    ) {
        let Some(descriptor) = self.get_valid(node_id, now) else {
            self.record_audit_event(
                now,
                "blind_relay_route_health",
                "rejected",
                format!(
                    "node_prefix={} result=ignored reason=missing_verified_route_surface",
                    hex::encode(&node_id[..4])
                ),
            );
            return;
        };
        let _ = self.record_route_forward_failure_for_descriptor(&descriptor, now, reason);
    }

    pub(super) fn record_route_forward_failure_for_node(
        &self,
        node_id: &[u8; 32],
        now: u64,
        reason: &PrivacySafePeerHealthReason,
    ) {
        let reason = reason.as_str();
        let mut route_health = self.route_health.write();
        let health = route_health.entry(*node_id).or_default();
        health.failure_count = health.failure_count.saturating_add(1);
        health.consecutive_failures = health.consecutive_failures.saturating_add(1);
        health.last_failure_at = Some(now);
        health.last_failure_reason = Some(reason.to_string());
        let starts_new_quarantine = health.consecutive_failures
            >= PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD
            && health
                .quarantine_until
                .map(|quarantine_until| now >= quarantine_until)
                .unwrap_or(true);
        if starts_new_quarantine {
            health.quarantine_count = health.quarantine_count.saturating_add(1);
            health.quarantine_until = Some(now.saturating_add(PEER_ROUTE_FAILURE_QUARANTINE_SECS));
            health.last_quarantine_at = Some(now);
            health.last_quarantine_reason = Some("consecutive_route_failures".to_string());
        }
        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "rejected",
            format!(
                "node_prefix={} result=failure reason={reason} consecutive_failures={}",
                hex::encode(&node_id[..4]),
                health.consecutive_failures
            ),
        );
        if starts_new_quarantine {
            self.record_audit_event(
                now,
                "blind_relay_route_quarantine",
                "limited",
                format!(
                    "node_prefix={} reason_bucket=consecutive_route_failures duration_seconds={}",
                    hex::encode(&node_id[..4]),
                    PEER_ROUTE_FAILURE_QUARANTINE_SECS
                ),
            );
            self.mark_peer_cache_dirty();
        }
    }

    /// Records a relay-protection rejection for a previous-hop peer.
    ///
    /// This is peer-level control-plane health only. Status exposes a short
    /// node prefix, counters, and coarse reason buckets; it never includes full
    /// node ids, route ids, endpoint URLs, encrypted blobs, receiver identities,
    /// client IPs, DNS contents, voucher secrets, private keys, wallet-level
    /// traffic, or payload-derived details.
    pub fn record_peer_relay_rejection(
        &self,
        node_id: &[u8; 32],
        now: u64,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let reason = PrivacySafePeerHealthReason::peer_relay_rejection(&reason).into_inner();
        let mut relay_health = self.relay_protection_health.write();
        let health = relay_health.entry(*node_id).or_default();
        health.rejection_count = health.rejection_count.saturating_add(1);
        health.last_rejection_at = Some(now);
        health.last_rejection_reason = Some(reason.clone());
        self.record_audit_event(
            now,
            "blind_relay_peer_protection",
            "rejected",
            format!(
                "node_prefix={} reason_bucket={reason}",
                hex::encode(&node_id[..4])
            ),
        );
    }

    /// Records that relay protection started a short quarantine for a peer.
    ///
    /// `quarantine_until` is a local control-plane timestamp. It is exposed as
    /// remaining seconds only; callers must not pass route ids, endpoint URLs,
    /// encrypted blobs, receivers, client traffic metadata, wallet ids, or
    /// plaintext-derived values into this method.
    pub fn record_peer_relay_quarantine_started(
        &self,
        node_id: &[u8; 32],
        now: u64,
        quarantine_until: u64,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let reason = PrivacySafePeerHealthReason::quarantine(&reason).into_inner();
        let mut relay_health = self.relay_protection_health.write();
        let health = relay_health.entry(*node_id).or_default();
        health.quarantine_count = health.quarantine_count.saturating_add(1);
        health.quarantine_until = Some(quarantine_until);
        health.last_quarantine_at = Some(now);
        health.last_quarantine_reason = Some(reason.clone());
        self.record_audit_event(
            now,
            "blind_relay_peer_quarantine",
            "limited",
            format!(
                "node_prefix={} reason_bucket={reason}",
                hex::encode(&node_id[..4])
            ),
        );
    }

    /// Fingerprints only signed fields that can change route behavior.
    ///
    /// Sequence, validity timestamps, software version, capacity, and region
    /// are deliberately excluded: refreshing those fields must not claim a new
    /// transport surface. Endpoint, capabilities, discovery/exit policy, and
    /// onion KEM material are included so a behavior change requires a probe.
    pub(super) fn descriptor_routeability_surface_fingerprint(
        descriptor: &SignedNodeDescriptor,
    ) -> Option<String> {
        let endpoint = descriptor
            .descriptor
            .public_endpoint
            .as_deref()?
            .trim()
            .trim_end_matches('/');
        if endpoint.is_empty() {
            return None;
        }

        let mut capabilities = descriptor
            .descriptor
            .capabilities
            .iter()
            .map(|capability| match capability {
                NodeCapability::PrivacyRelay => 1u8,
                NodeCapability::ChatRelay => 2,
                NodeCapability::EncryptedStorage => 3,
                NodeCapability::AgentRelay => 4,
                NodeCapability::OnionMiddle => 5,
                // [MIRROR-CAPABILITY 2026-07-24 by Codex] Keep the appended
                // role distinct so enabling it invalidates stale routeability
                // evidence bound to an older signed transport surface.
                NodeCapability::DirectoryMirrorCarrier => 6,
                // [BLIND-VAULT-REPLICA-CAPABILITY 2026-08-10 by Codex] Keep
                // this appended role distinct in routeability evidence.
                NodeCapability::BlindVaultReplica => 7,
            })
            .collect::<Vec<_>>();
        capabilities.sort_unstable();
        capabilities.dedup();

        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx-routeability-surface-v1");
        hasher.update(descriptor.descriptor.node_id);
        hasher.update(u64::try_from(endpoint.len()).ok()?.to_be_bytes());
        hasher.update(endpoint.as_bytes());
        hasher.update(u64::try_from(capabilities.len()).ok()?.to_be_bytes());
        hasher.update(&capabilities);
        hasher.update([u8::from(descriptor.descriptor.policy.public_discovery)]);
        hasher.update([u8::from(descriptor.descriptor.policy.allows_public_exit)]);
        hasher.update([descriptor.descriptor.kem_alg]);
        hasher.update(descriptor.descriptor.kem_public);
        // [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] Append only present,
        // recognized feature tokens. Legacy descriptors therefore retain their
        // historical v1 fingerprint, while enabling a response contract forces
        // fresh route evidence and cannot inherit success from the old surface.
        for feature in NodeProtocolFeature::ALL {
            if descriptor.descriptor.advertises_protocol_feature(feature) {
                let token = feature.semver_build_token();
                hasher.update(b"\0aeronyx-protocol-feature:");
                hasher.update(u64::try_from(token.len()).ok()?.to_be_bytes());
                hasher.update(token.as_bytes());
            }
        }
        Some(hex::encode(hasher.finalize()))
    }

    /// Legacy exact-descriptor fingerprint retained for reading v0.52 caches.
    pub(super) fn descriptor_routeability_fingerprint(
        descriptor: &SignedNodeDescriptor,
    ) -> Option<String> {
        let signing_bytes = descriptor.descriptor.signing_bytes().ok()?;
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx-routeability-cache-evidence-v1");
        hasher.update(signing_bytes);
        hasher.update(descriptor.signature);
        Some(hex::encode(hasher.finalize()))
    }
}
