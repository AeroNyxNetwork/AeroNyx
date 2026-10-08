// [ARCH-SPLIT 2026-10-02]
// Untrusted discovery admission and witness-requester allow lists.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Enables the Stage-A boundary for legacy unauthenticated discovery.
    ///
    /// Production discovery routers call this during construction. Keeping the
    /// switch explicit preserves existing local/test-only direct imports until
    /// their callers are migrated to a transport-authenticated admission path.
    pub(crate) fn enable_untrusted_discovery_candidate_mode(&self) {
        self.untrusted_discovery_candidate_mode
            .store(true, Ordering::Release);
    }

    pub(super) fn untrusted_candidate_commitment(
        descriptor: &SignedNodeDescriptor,
    ) -> Option<[u8; 32]> {
        let signing_bytes = descriptor.descriptor.signing_bytes().ok()?;
        let mut digest = Sha256::new();
        digest.update(signing_bytes);
        digest.update(descriptor.signature);
        Some(digest.finalize().into())
    }

    pub(super) fn untrusted_candidate_is_within_limits(
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        let issued_at = descriptor.descriptor.issued_at;
        let expires_at = descriptor.descriptor.expires_at;
        let Some(lifetime) = expires_at.checked_sub(issued_at) else {
            return false;
        };
        descriptor.verify_at(now).is_ok()
            && lifetime <= UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS
            && issued_at <= now.saturating_add(UNTRUSTED_DISCOVERY_MAX_FUTURE_SKEW_SECS)
            && expires_at
                <= now
                    .saturating_add(UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS)
                    .saturating_add(UNTRUSTED_DISCOVERY_MAX_FUTURE_SKEW_SECS)
    }

    pub(super) fn permissionless_descriptor_shape_is_valid(
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        if !Self::untrusted_candidate_is_within_limits(descriptor, now)
            || descriptor.sequence() == 0
            || !descriptor.descriptor.policy.public_discovery
            || descriptor.descriptor.software_version.is_empty()
            || descriptor.descriptor.software_version.len() > 256
            || descriptor
                .descriptor
                .software_version
                .bytes()
                .any(|byte| byte.is_ascii_control())
        {
            return false;
        }

        let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
            return false;
        };
        if endpoint.trim() != endpoint || !crate::api::peer_endpoint_is_permitted(endpoint) {
            return false;
        }

        let mut capabilities = HashSet::new();
        if !descriptor
            .descriptor
            .capabilities
            .iter()
            .all(|capability| capabilities.insert(*capability))
        {
            return false;
        }

        match (
            descriptor.descriptor.schema_version,
            descriptor.descriptor.kem_alg,
            descriptor.descriptor.kem_public,
        ) {
            (1 | 2, 0, public_key) if public_key == [0; 32] => {}
            (2, 1, public_key) if public_key != [0; 32] => {}
            _ => return false,
        }

        let Ok(canonical) = descriptor.encode_canonical() else {
            return false;
        };
        SignedNodeDescriptor::decode_canonical(&canonical)
            .is_ok_and(|decoded| decoded == *descriptor)
    }

    pub(super) fn has_locally_established_identity(&self, node_id: &[u8; 32]) -> bool {
        self.peers.read().contains_key(node_id)
    }

    pub(super) fn prune_untrusted_candidate_state(
        state: &mut UntrustedDiscoveryCandidateState,
        now: u64,
    ) {
        let expired = state
            .candidates
            .iter()
            .filter_map(|(node_id, candidate)| {
                (candidate.descriptor.descriptor.expires_at <= now)
                    .then_some((*node_id, candidate.clone()))
            })
            .collect::<Vec<_>>();
        for (node_id, candidate) in expired {
            state.candidates.remove(&node_id);
            state.tombstones.insert(
                node_id,
                UntrustedDiscoveryTombstone {
                    sequence: candidate.descriptor.sequence(),
                    commitment: candidate.commitment,
                    expires_at: now.saturating_add(UNTRUSTED_DISCOVERY_TOMBSTONE_TTL_SECS),
                },
            );
        }
        state
            .tombstones
            .retain(|_, tombstone| tombstone.expires_at > now);

        while state.tombstones.len() > UNTRUSTED_DISCOVERY_TOMBSTONE_CAPACITY {
            let Some(node_id) = state
                .tombstones
                .iter()
                .min_by_key(|(node_id, tombstone)| (tombstone.expires_at, *node_id))
                .map(|(node_id, _)| *node_id)
            else {
                break;
            };
            state.tombstones.remove(&node_id);
        }
    }

    pub(super) fn admit_untrusted_candidate(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> CandidateAdmissionOutcome {
        if !Self::untrusted_candidate_is_within_limits(&descriptor, now) {
            return CandidateAdmissionOutcome::Rejected;
        }
        let Some(commitment) = Self::untrusted_candidate_commitment(&descriptor) else {
            return CandidateAdmissionOutcome::Rejected;
        };
        let node_id = descriptor.node_id();
        let sequence = descriptor.sequence();

        // [OPEN-NODE-ADMISSION 2026-09-24 by Codex] Candidate admission must
        // never forget a stronger sequence already held in the live store.
        // A higher sequence may wait as a candidate, but a rollback or
        // same-sequence content conflict fails before candidate mutation.
        // Keep the live-store read guard until the candidate mutation is
        // complete. Otherwise a concurrent live upsert could advance the
        // sequence between this comparison and candidate insertion.
        let peers = self.peers.read();
        if let Some(existing) = peers.get(&node_id) {
            if sequence < existing.sequence() {
                return CandidateAdmissionOutcome::Stale;
            }
            if sequence == existing.sequence() {
                return if &descriptor == existing {
                    CandidateAdmissionOutcome::Unchanged
                } else {
                    CandidateAdmissionOutcome::Conflict
                };
            }
        }

        let mut state = self.untrusted_discovery_candidates.write();
        Self::prune_untrusted_candidate_state(&mut state, now);

        if let Some(existing) = state.candidates.get(&node_id) {
            if sequence < existing.descriptor.sequence() {
                return CandidateAdmissionOutcome::Stale;
            }
            if sequence == existing.descriptor.sequence() {
                return if commitment == existing.commitment {
                    CandidateAdmissionOutcome::Unchanged
                } else {
                    CandidateAdmissionOutcome::Conflict
                };
            }
        } else if let Some(tombstone) = state.tombstones.get(&node_id) {
            if sequence < tombstone.sequence {
                return CandidateAdmissionOutcome::Stale;
            }
            if sequence == tombstone.sequence {
                return if commitment == tombstone.commitment {
                    CandidateAdmissionOutcome::Unchanged
                } else {
                    CandidateAdmissionOutcome::Conflict
                };
            }
        }

        if !state.candidates.contains_key(&node_id)
            && state.candidates.len() >= UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY
        {
            return CandidateAdmissionOutcome::Saturated;
        }

        state.tombstones.remove(&node_id);
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A newer
        // candidate invalidates any old promotion before it can be selected.
        if let Some(gate) = self.permissionless_promotions.write().get_mut(&node_id) {
            gate.active = false;
        }
        state.candidates.insert(
            node_id,
            UntrustedDiscoveryCandidate {
                descriptor,
                commitment,
            },
        );
        drop(state);
        drop(peers);
        CandidateAdmissionOutcome::Candidate
    }

    /// Admits one canonical self-signed descriptor into the bounded Stage-A
    /// candidate lane without consulting a central allowlist.
    ///
    /// [OPEN-NODE-ADMISSION 2026-09-24 by Codex] This boundary authenticates
    /// the node key and exact descriptor fields, constrains lifetime and SSRF
    /// surface, and fences replay/rollback. It intentionally grants no
    /// endpoint-possession, routeability, ranking, advertisement, stake, or
    /// consensus authority; future ETH-derived economics must enter through a
    /// separate candidate-bound projection rather than new descriptor fields.
    pub(crate) fn admit_permissionless_descriptor(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> PermissionlessNodeAdmissionOutcome {
        let outcome = if Self::permissionless_descriptor_shape_is_valid(&descriptor, now) {
            match self.admit_untrusted_candidate(descriptor, now) {
                CandidateAdmissionOutcome::Candidate => {
                    PermissionlessNodeAdmissionOutcome::Admitted
                }
                CandidateAdmissionOutcome::Unchanged => {
                    PermissionlessNodeAdmissionOutcome::ExactReplay
                }
                CandidateAdmissionOutcome::Stale => PermissionlessNodeAdmissionOutcome::Stale,
                CandidateAdmissionOutcome::Conflict => PermissionlessNodeAdmissionOutcome::Conflict,
                CandidateAdmissionOutcome::Saturated => {
                    PermissionlessNodeAdmissionOutcome::Saturated
                }
                CandidateAdmissionOutcome::Rejected => PermissionlessNodeAdmissionOutcome::Rejected,
            }
        } else {
            PermissionlessNodeAdmissionOutcome::Rejected
        };

        let mut report = PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };
        match outcome {
            PermissionlessNodeAdmissionOutcome::Admitted => report.candidates = 1,
            PermissionlessNodeAdmissionOutcome::ExactReplay => report.unchanged = 1,
            PermissionlessNodeAdmissionOutcome::Stale => report.stale = 1,
            PermissionlessNodeAdmissionOutcome::Conflict
            | PermissionlessNodeAdmissionOutcome::Saturated
            | PermissionlessNodeAdmissionOutcome::Rejected => report.rejected = 1,
        }
        self.record_import_report(&report, now);
        outcome
    }

    /// Replaces the exact identities allowed to store a delivery-cache anchor.
    ///
    /// This bilateral pin is deliberately separate from permissionless peer
    /// discovery. An empty set disables witness writes without preventing the
    /// node from discovering or relaying through other protocol peers.
    pub fn configure_verified_delivery_witness_requesters(&self, requesters: &[[u8; 32]]) {
        *self.verified_delivery_witness_requesters.write() = requesters.iter().copied().collect();
    }

    /// Returns whether this requester is explicitly pinned for witness writes.
    #[must_use]
    pub fn verified_delivery_witness_requester_allowed(&self, requester: &[u8; 32]) -> bool {
        self.verified_delivery_witness_requesters
            .read()
            .contains(requester)
    }

    /// Replaces identities allowed to store a custody-audit anchor decision.
    ///
    /// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] This admission set is
    /// intentionally independent from delivery witnesses and permissionless
    /// discovery. Reusing either would silently broaden state-write authority.
    pub fn configure_custody_audit_witness_requesters(&self, requesters: &[[u8; 32]]) {
        *self.custody_audit_witness_requesters.write() = requesters.iter().copied().collect();
    }

    /// Returns whether this producer is explicitly pinned for custody witness writes.
    #[must_use]
    pub fn custody_audit_witness_requester_allowed(&self, requester: &[u8; 32]) -> bool {
        self.custody_audit_witness_requesters
            .read()
            .contains(requester)
    }
}
