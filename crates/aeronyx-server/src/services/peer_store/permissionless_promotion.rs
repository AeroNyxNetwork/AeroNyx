// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/permissionless_promotion.rs
// ============================================================================
//! Permissionless candidate promotion and its bounded route gate.
//!
//! [PERMISSIONLESS-PROMOTION-SPLIT 2026-09-25 by Codex] Keeps the Stage-A
//! candidate capacity, exact promotion transition, and verified control-probe
//! gate together while preserving the parent store's lock ordering and fields.

use std::sync::atomic::Ordering;

use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};

use super::{
    PeerStore, PeerStoreError, UntrustedDiscoveryCandidateState, VerifiedPromotionMaterial,
    UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY,
};

// [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A promoted Stage-A
// descriptor stays isolated until its exact durable proof binding and a fresh
// route probe are both current. Pending/invalidated entries remain deny gates.
#[derive(Clone, Copy)]
pub(super) struct PermissionlessPromotionGate {
    pub(super) descriptor_hash: [u8; 32],
    pub(super) valid_until: u64,
    pub(super) active: bool,
    pub(super) verified_control_probe: bool,
    pub(super) generation: u64,
}

// [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Historic gates
// must not grow with an unbounded sequence of public Stage-A identities.
// Saturation fails closed; a later lifecycle pass may reclaim a gate only
// together with its matching live descriptor, never by deleting the deny bit.
pub(super) const MAX_PERMISSIONLESS_PROMOTION_GATES: usize = 4_096;

impl PeerStore {
    /// Returns a small, exact, still-untrusted batch for bounded verification.
    /// The result never enters the live peer view or public API by itself.
    pub(crate) fn permissionless_candidate_batch(
        &self,
        now: u64,
        limit: usize,
    ) -> Vec<SignedNodeDescriptor> {
        let state = self.untrusted_discovery_candidates.read();
        let mut candidates = state
            .candidates
            .values()
            .filter(|candidate| {
                Self::untrusted_candidate_is_within_limits(&candidate.descriptor, now)
            })
            .map(|candidate| candidate.descriptor.clone())
            .collect::<Vec<_>>();
        candidates.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        if !candidates.is_empty() {
            let round = self
                .permissionless_candidate_round
                .fetch_add(1, Ordering::Relaxed);
            let start = (round as usize) % candidates.len();
            candidates.rotate_left(start);
        }
        candidates.truncate(limit.min(UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY));
        candidates
    }

    pub(crate) fn prepare_permissionless_promotion_capacity_for(
        &self,
        node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        if now == 0 {
            return false;
        }
        if self.permissionless_promotions.read().len() >= MAX_PERMISSIONLESS_PROMOTION_GATES {
            self.prune_expired_permissionless_promotions(now);
        }
        let gates = self.permissionless_promotions.read();
        gates.contains_key(node_id) || gates.len() < MAX_PERMISSIONLESS_PROMOTION_GATES
    }

    // [PHALA-PROMOTION-NETWORK-ADMISSION 2026-10-07 by Codex] Preflight
    // exact Stage-A work before spending transport/QVL capacity. This is not a
    // reservation or route authority; final promotion still rechecks its gates.
    pub(crate) fn permissionless_candidate_probe_is_admitted(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        if now == 0 || !Self::permissionless_descriptor_shape_is_valid(descriptor, now) {
            return false;
        }
        let Some(commitment) = Self::untrusted_candidate_commitment(descriptor) else {
            return false;
        };
        let node_id = descriptor.node_id();
        if !self.prepare_permissionless_promotion_capacity_for(&node_id, now) {
            return false;
        }
        // Match Stage-A insertion's peers -> candidates -> gates order. A
        // stronger live import may have arrived without removing old Stage-A.
        let peers = self.peers.read();
        if peers.get(&node_id).is_some_and(|current| {
            current.sequence() > descriptor.sequence()
                || (current.sequence() == descriptor.sequence() && current != descriptor)
        }) {
            return false;
        }
        let candidates = self.untrusted_discovery_candidates.read();
        let gates = self.permissionless_promotions.read();
        candidates.candidates.get(&node_id).is_some_and(|candidate| {
            candidate.descriptor == *descriptor && candidate.commitment == commitment
        }) && (gates.contains_key(&node_id) || gates.len() < MAX_PERMISSIONLESS_PROMOTION_GATES)
    }

    pub(super) fn prune_expired_permissionless_promotions(&self, now: u64) -> usize {
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Never drop
        // a deny gate while its exact live descriptor remains in the peer
        // map: that would turn expiry into route authority on clock rollback.
        // Lock peers before gates, matching read-side lock order.
        // [REVERSE-ONION-AUTHORITY-FENCE 2026-10-05 by Codex] Expiry can
        // remove a live descriptor; order it against in-flight Lease issue.
        let _authority_update = self.private_onion_authority_gate.write();
        let mut peers = self.peers.write();
        let mut gates = self.permissionless_promotions.write();
        let expired = gates
            .iter()
            .filter(|(_, gate)| gate.valid_until < now)
            .map(|(node_id, gate)| (*node_id, gate.descriptor_hash))
            .collect::<Vec<_>>();
        let mut removed_live = false;
        for (node_id, descriptor_hash) in &expired {
            let matching_peer = peers.get(node_id).is_some_and(|descriptor| {
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                    .is_ok_and(|pin| pin.descriptor_hash == *descriptor_hash)
            });
            if matching_peer {
                peers.remove(node_id);
                removed_live = true;
            }
            gates.remove(node_id);
        }
        drop(gates);
        drop(peers);
        if removed_live {
            self.mark_peer_cache_dirty();
        }
        expired.len()
    }

    /// Accepts only exact resolver-minted material for a still-current Stage-A
    /// candidate. The deny gate is installed before the live descriptor write;
    /// a concurrent candidate rotation leaves it closed.
    pub(crate) fn promote_permissionless_candidate(
        &self,
        material: &VerifiedPromotionMaterial,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        self.promote_permissionless_candidate_if(material, now, || true)
    }

    // [PHALA-PROMOTION-CANCEL-OWNERSHIP 2026-10-08 by Codex] Owner
    // admission can only veto verified material. A cancelled cell may finish
    // an admitted descriptor write, but cannot activate its closed deny gate.
    pub(crate) fn promote_permissionless_candidate_if(
        &self,
        material: &VerifiedPromotionMaterial,
        now: u64,
        mut owner_running: impl FnMut() -> bool,
    ) -> Result<bool, PeerStoreError> {
        if !owner_running() { return Err(PeerStoreError::VerificationFailed); }
        let descriptor = material.descriptor();
        let node_id = descriptor.node_id();
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|_| PeerStoreError::VerificationFailed)?;
        if now == 0 || material.valid_until() < now || pin != material.descriptor_commitment() {
            return Err(PeerStoreError::VerificationFailed);
        }
        if !self.prepare_permissionless_promotion_capacity_for(&node_id, now) {
            return Err(PeerStoreError::CapacityExceeded {
                max_peers: MAX_PERMISSIONLESS_PROMOTION_GATES,
            });
        }
        let candidate_is_exact = |state: &UntrustedDiscoveryCandidateState| {
            state.candidates.get(&node_id).is_some_and(|candidate| {
                candidate.descriptor == *descriptor
                    && Self::untrusted_candidate_commitment(descriptor)
                        == Some(candidate.commitment)
            })
        };
        if !candidate_is_exact(&self.untrusted_discovery_candidates.read()) {
            return Err(PeerStoreError::VerificationFailed);
        }
        let mut gates = self.permissionless_promotions.write();
        if !owner_running() { return Err(PeerStoreError::VerificationFailed); }
        if !gates.contains_key(&node_id) && gates.len() >= MAX_PERMISSIONLESS_PROMOTION_GATES {
            return Err(PeerStoreError::CapacityExceeded {
                max_peers: MAX_PERMISSIONLESS_PROMOTION_GATES,
            });
        }
        let generation = self
            .permissionless_promotion_generation
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        let previous = gates.insert(
            node_id,
            PermissionlessPromotionGate {
                descriptor_hash: pin.descriptor_hash,
                valid_until: material.valid_until(),
                active: false,
                verified_control_probe: false,
                generation,
            },
        );
        drop(gates);
        if !owner_running() { return Err(PeerStoreError::VerificationFailed); }
        let changed = match self.upsert_verified_from_source(
            descriptor.clone(),
            now,
            "permissionless_promotion",
        ) {
            Ok(changed) => changed,
            Err(error) => {
                let mut gates = self.permissionless_promotions.write();
                if gates
                    .get(&node_id)
                    .is_some_and(|gate| gate.generation == generation)
                {
                    if let Some(previous) = previous {
                        gates.insert(node_id, previous);
                    } else {
                        gates.remove(&node_id);
                    }
                }
                return Err(error);
            }
        };
        let candidates = self.untrusted_discovery_candidates.read();
        if !candidate_is_exact(&candidates) {
            return Err(PeerStoreError::VerificationFailed);
        }
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Generic
        // route success is not promotion authority. Keep the candidate read
        // lock through activation so rotation cannot reopen the old gate.
        // [PHALA-PROMOTION-CANCEL-OWNERSHIP 2026-10-08 by Codex] Sample
        // after the final gate lock wait. A concurrent same-descriptor round
        // cannot lend this round its generation or bypass its owner's veto.
        let mut gates = self.permissionless_promotions.write();
        if !owner_running() { return Err(PeerStoreError::VerificationFailed); }
        let Some(gate) = gates.get_mut(&node_id) else { return Err(PeerStoreError::VerificationFailed); };
        if gate.descriptor_hash != pin.descriptor_hash || gate.generation != generation {
            return Err(PeerStoreError::VerificationFailed);
        }
        gate.active = true;
        drop(gates);
        drop(candidates);
        Ok(changed)
    }

    /// Records the server's exact target-signed terminal control probe after
    /// its separate receipt verifier has accepted the immutable request and
    /// response. Ordinary route observations cannot call this transition.
    pub(crate) fn record_permissionless_promotion_probe_verified(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] DNS
        // probes must retain the same descriptor/appraisal epoch through the
        // route-gate write. A valid receipt alone cannot revive expired quotes.
        let dns_probe = descriptor.descriptor.public_endpoint.as_deref().is_some_and(|endpoint| {
            aeronyx_core::protocol::discovery_endpoint_proof::
                canonical_public_https_dns_endpoint_commitment_v1(endpoint).is_ok()
        });
        let _authority_snapshot = if dns_probe {
            let Some(guard) = self.private_onion_authority_gate.try_read() else { return false; };
            Some(guard)
        } else { None };
        if dns_probe && !self.phala_promotion_control_probe_under_authority_guard(descriptor, now) {
            return false;
        }
        let node_id = descriptor.node_id();
        let Ok(pin) = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor) else {
            return false;
        };
        let current = self.permissionless_promotions.read().get(&node_id).copied();
        if current.is_none_or(|gate| {
            !gate.active || gate.valid_until < now || gate.descriptor_hash != pin.descriptor_hash
        }) {
            return false;
        }
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A new
        // signed descriptor and successful control ACK cannot shorten this
        // identity's existing route quarantine. The quarantine check and
        // success mutation share one route-health write lock.
        if !self.record_route_forward_success_with_quarantine_policy(descriptor, now, false) {
            return false;
        }
        let mut gates = self.permissionless_promotions.write();
        let Some(gate) = gates.get_mut(&node_id) else {
            return false;
        };
        if !gate.active || gate.valid_until < now || gate.descriptor_hash != pin.descriptor_hash {
            return false;
        }
        if dns_probe && !self.phala_promotion_control_appraisal_is_fresh(descriptor, now) {
            return false;
        }
        gate.verified_control_probe = true;
        true
    }

    // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] This
    // narrow pre-route capability breaks the readiness bootstrap cycle: the
    // exact promoted descriptor may be probed before verified_control_probe,
    // but Stage-A input, an ordinary DNS peer or a pin alone is never enough.
    pub(crate) fn phala_promotion_control_probe_is_admitted(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        let Some(_authority_snapshot) = self.private_onion_authority_gate.try_read() else {
            return false;
        };
        self.phala_promotion_control_probe_under_authority_guard(descriptor, now)
    }

    fn phala_promotion_control_probe_under_authority_guard(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        if now == 0 || !self.phala_attested_peer_routes_required()
            || !Self::permissionless_descriptor_shape_is_valid(descriptor, now)
            || !descriptor.descriptor.capabilities.contains(&aeronyx_core::protocol::discovery::NodeCapability::ChatRelay)
            || !descriptor.descriptor.public_endpoint.as_deref().is_some_and(|endpoint| {
                aeronyx_core::protocol::discovery_endpoint_proof::
                    canonical_public_https_dns_endpoint_commitment_v1(endpoint).is_ok()
            })
        {
            return false;
        }
        let Ok(pin) = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor) else {
            return false;
        };
        let Some(peers) = self.peers.try_read() else { return false; };
        if peers.get(&descriptor.node_id()) != Some(descriptor) { return false; }
        let Some(gates) = self.permissionless_promotions.try_read() else { return false; };
        gates.get(&descriptor.node_id()).is_some_and(|gate| {
            gate.active && gate.valid_until >= now && gate.descriptor_hash == pin.descriptor_hash
        }) && self.phala_promotion_control_appraisal_is_fresh(descriptor, now)
    }

    fn phala_promotion_control_appraisal_is_fresh(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> bool {
        use aeronyx_core::protocol::{discovery::signed_descriptor_commitment_hash, NodeProtocolFeature};
        let Ok(commitment) = signed_descriptor_commitment_hash(descriptor) else { return false; };
        let Some(cache) = self.phala_peer_attestations.try_read() else { return false; };
        descriptor.descriptor.advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1)
            && cache.get(&descriptor.node_id()).is_some_and(|entry| {
                entry.commitment == commitment
                    && entry.is_fresh_at(descriptor, now, self.phala_peer_attestation_max_age_secs())
            })
    }

    pub(super) fn permissionless_gate_allows(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        require_route_probe: bool,
    ) -> bool {
        let node_id = descriptor.node_id();
        let gate = self.permissionless_promotions.read().get(&node_id).copied();
        let Some(gate) = gate else { return true };
        if !gate.active
            || gate.valid_until < now
            || DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                .map_or(true, |pin| pin.descriptor_hash != gate.descriptor_hash)
        {
            return false;
        }
        !require_route_probe || {
            let health = self.route_health.read();
            gate.verified_control_probe
                && Self::routeability_state_and_ready(health.get(&node_id), now).1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        PermissionlessNodeAdmissionOutcome, PEER_ROUTE_FAILURE_QUARANTINE_SECS,
        PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD, PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS,
    };
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{NodeCapability, NodeDescriptor, SignedNodeDescriptor};

    fn permissionless_descriptor_for(
        kp: &IdentityKeyPair,
        sequence: u64,
        now: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            now.saturating_sub(1),
            now + 600,
            "1.0.0+anpf1-brsr1",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, kp).unwrap()
    }

    // [PHALA-PROMOTION-CANCEL-OWNERSHIP 2026-10-08 by Codex] Authored
    // only: both gate admissions run under the actual write lock. Cancellation
    // after a descriptor write leaves its deny gate closed, not a ready route.
    #[test]
    fn promotion_owner_veto_fences_gate_install_activation_and_generation() {
        let now = 1_780_000_000;
        let key = IdentityKeyPair::from_bytes(&[0x56; 32]).unwrap();
        let descriptor = permissionless_descriptor_for(&key, 7, now, "https://8.8.8.8:8422");
        let material = VerifiedPromotionMaterial::test_only_from_descriptor(descriptor.clone(), now, now + 90).unwrap();
        let store = PeerStore::new();
        assert_eq!(store.admit_permissionless_descriptor(descriptor.clone(), now), PermissionlessNodeAdmissionOutcome::Admitted);
        assert!(store.promote_permissionless_candidate_if(&material, now, || false).is_err());
        assert!(store.permissionless_promotions.read().is_empty());
        assert!(store.peers.read().is_empty());
        let mut samples = 0;
        assert!(store.promote_permissionless_candidate_if(&material, now, || {
            samples += 1;
            if samples == 2 || samples == 4 {
                assert!(store.permissionless_promotions.try_write().is_none(), "sample after gate lock acquisition");
            }
            samples < 4
        }).is_err());
        assert_eq!(samples, 4);
        assert_eq!(store.peers.read().get(&descriptor.node_id()), Some(&descriptor));
        assert!(!store.permissionless_promotions.read().get(&descriptor.node_id()).unwrap().active);
        assert!(store.get_valid(&descriptor.node_id(), now).is_none());
        let mut samples = 0;
        assert!(store.promote_permissionless_candidate_if(&material, now, || {
            samples += 1;
            if samples == 3 {
                store.permissionless_promotions.write().get_mut(&descriptor.node_id()).unwrap().generation += 1;
            }
            true
        }).is_err(), "an otherwise live owner cannot activate another generation");
        assert!(!store.permissionless_promotions.read().get(&descriptor.node_id()).unwrap().active);
        assert!(store.promote_permissionless_candidate_if(&material, now, || true).is_ok(), "positive admission calibration");
        assert!(store.permissionless_promotions.read().get(&descriptor.node_id()).unwrap().active);
        assert!(!store.permissionless_promotions.read().get(&descriptor.node_id()).unwrap().verified_control_probe);
        assert!(store.get_valid(&descriptor.node_id(), now).is_none());
    }

    // [PHALA-PROMOTION-CANCEL-OWNERSHIP 2026-10-08 by Codex] Authored
    // only: first owner sampling precedes a known held gate; stop before unlock
    // must be observed by the post-lock admission, with no descriptor insertion.
    #[test]
    fn promotion_stopped_while_waiting_for_gate_cannot_publish() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let now = 1_780_000_000;
        let key = IdentityKeyPair::from_bytes(&[0x57; 32]).unwrap();
        let descriptor = permissionless_descriptor_for(&key, 7, now, "https://8.8.8.8:8422");
        let material = VerifiedPromotionMaterial::test_only_from_descriptor(descriptor.clone(), now, now + 90).unwrap();
        let store = Arc::new(PeerStore::new());
        assert_eq!(store.admit_permissionless_descriptor(descriptor, now), PermissionlessNodeAdmissionOutcome::Admitted);
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_store = Arc::clone(&store);
        let worker_stopped = Arc::clone(&stopped);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let held = store.permissionless_promotions.write();
        let worker = std::thread::spawn(move || {
            let mut ready = Some(ready_tx);
            worker_store.promote_permissionless_candidate_if(&material, now, || {
                let running = !worker_stopped.load(Ordering::SeqCst);
                if let Some(ready) = ready.take() { ready.send(()).unwrap(); }
                running
            })
        });
        ready_rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        stopped.store(true, Ordering::SeqCst);
        drop(held);
        assert!(worker.join().unwrap().is_err());
        assert!(store.permissionless_promotions.read().is_empty());
        assert!(store.peers.read().is_empty());
    }

    // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] Authored
    // only: this pre-route capability requires both independently established
    // promotion and exact fresh quote evidence, including on receipt commit.
    #[test]
    fn phala_dns_control_probe_requires_exact_promotion_and_fresh_appraisal() {
        use aeronyx_core::protocol::NodeProtocolFeature;
        let now = 1_780_000_000;
        let identity = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
        // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] Use a
        // public-shape synthetic origin; this test performs no DNS or sockets.
        let descriptor = permissionless_descriptor_for(&identity, 7, now, "https://relay.aeronyx.network");
        let descriptor = SignedNodeDescriptor::sign(descriptor.descriptor.with_protocol_features(
            [NodeProtocolFeature::PhalaNodeAttestationV1],
        ), &identity).unwrap();
        let store = PeerStore::new();
        store.configure_phala_attested_peer_routes(true, 60);
        assert_eq!(store.admit_permissionless_descriptor(descriptor.clone(), now),
            PermissionlessNodeAdmissionOutcome::Admitted);
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        let material = VerifiedPromotionMaterial::test_only_from_descriptor(descriptor.clone(), now, now + 300).unwrap();
        store.promote_permissionless_candidate(&material, now).unwrap();
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        assert!(!store.record_permissionless_promotion_probe_verified(&descriptor, now));
        assert!(!store.permissionless_promotions.read().get(&descriptor.node_id()).unwrap().verified_control_probe);
        assert!(store.record_phala_peer_attestation(&descriptor, now));
        assert!(store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        assert!(store.get_valid(&descriptor.node_id(), now).is_none(),
            "the capability must exist before the control readiness bit opens");
        {
            let _writer = store.private_onion_authority_gate.write();
            assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
            assert!(!store.record_permissionless_promotion_probe_verified(&descriptor, now));
        }
        {
            let _writer = store.phala_peer_attestations.write();
            assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        }
        {
            let mut gates = store.permissionless_promotions.write();
            gates.get_mut(&descriptor.node_id()).unwrap().active = false;
        }
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        store.permissionless_promotions.write().get_mut(&descriptor.node_id()).unwrap().active = true;
        store.permissionless_promotions.write().get_mut(&descriptor.node_id()).unwrap().valid_until = now - 1;
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now));
        store.permissionless_promotions.write().get_mut(&descriptor.node_id()).unwrap().valid_until = now + 300;
        let restarted = PeerStore::new();
        restarted.configure_phala_attested_peer_routes(true, 60);
        restarted.upsert_verified(descriptor.clone(), now).unwrap();
        assert!(restarted.record_phala_peer_attestation(&descriptor, now));
        assert!(!restarted.phala_promotion_control_probe_is_admitted(&descriptor, now),
            "descriptor and quote alone do not restore the promotion gate");
        assert!(store.record_permissionless_promotion_probe_verified(&descriptor, now + 1));
        assert!(store.get_valid(&descriptor.node_id(), now + 1).is_some());
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now + 61));
        assert!(!store.record_permissionless_promotion_probe_verified(&descriptor, now + 61),
            "late valid receipt cannot revive expired appraisal");
        // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] A signed
        // descriptor remains bootstrap evidence, not fresh route authority.
        assert!(store.get_valid(&descriptor.node_id(), now + 61).is_some());
        assert!(!store.phala_peer_route_is_eligible(&descriptor, now + 61));
        let mut rotated = descriptor.descriptor.clone();
        rotated.sequence += 1;
        rotated.public_endpoint = Some("https://rotated.aeronyx.network".into());
        let rotated = SignedNodeDescriptor::sign(rotated, &identity).unwrap();
        store.upsert_verified(rotated.clone(), now + 61).unwrap();
        assert!(store.record_phala_peer_attestation(&rotated, now + 61));
        assert!(!store.phala_promotion_control_probe_is_admitted(&descriptor, now + 61));
        assert!(!store.phala_promotion_control_probe_is_admitted(&rotated, now + 61),
            "new quote cannot reuse the old promotion commitment");
        store.configure_phala_attested_peer_routes(false, 60);
        assert!(!store.phala_promotion_control_probe_is_admitted(&rotated, now + 61));
    }

    #[test]
    fn permissionless_gate_requires_verified_probe_before_export() {
        let now = 1_780_000_000;
        let identity = IdentityKeyPair::generate();
        let descriptor = permissionless_descriptor_for(&identity, 7, now, "https://8.8.8.8:8422");
        let store = PeerStore::new();
        assert!(store.upsert_verified(descriptor.clone(), now).unwrap());
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        store.permissionless_promotions.write().insert(
            descriptor.node_id(),
            PermissionlessPromotionGate {
                descriptor_hash: pin.descriptor_hash,
                valid_until: now + 90,
                active: true,
                verified_control_probe: false,
                generation: 0,
            },
        );
        assert!(store.get_valid(&descriptor.node_id(), now + 2).is_none());
        assert!(store.export_peer_cache_snapshot(now + 2).peers.is_empty());
        assert!(store.record_permissionless_promotion_probe_verified(&descriptor, now + 4));
        assert_eq!(
            store.get_valid(&descriptor.node_id(), now + 4),
            Some(descriptor)
        );
    }

    #[test]
    fn permissionless_gate_capacity_reclaims_only_expired_entries() {
        let now = 1_780_200_000;
        let identity = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
        let descriptor = permissionless_descriptor_for(&identity, 7, now, "https://8.8.8.8:8422");
        let store = PeerStore::new();
        {
            let mut gates = store.permissionless_promotions.write();
            for index in 0..MAX_PERMISSIONLESS_PROMOTION_GATES {
                let mut node_id = [0u8; 32];
                node_id[..8].copy_from_slice(&(index as u64).to_le_bytes());
                gates.insert(
                    node_id,
                    PermissionlessPromotionGate {
                        descriptor_hash: [4; 32],
                        valid_until: now + 90,
                        active: false,
                        verified_control_probe: false,
                        generation: 0,
                    },
                );
            }
        }
        assert!(!store.prepare_permissionless_promotion_capacity_for(&descriptor.node_id(), now));
        assert_eq!(
            store.permissionless_promotions.read().len(),
            MAX_PERMISSIONLESS_PROMOTION_GATES
        );
        assert_eq!(
            store.cleanup_expired(now + 91),
            MAX_PERMISSIONLESS_PROMOTION_GATES
        );
        assert!(
            store.prepare_permissionless_promotion_capacity_for(&descriptor.node_id(), now + 91)
        );
    }

    #[test]
    fn expired_promotion_prunes_exact_peer_and_gate() {
        let now = 1_780_300_000;
        let identity = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
        let descriptor = permissionless_descriptor_for(&identity, 7, now, "https://8.8.8.8:8422");
        let material =
            VerifiedPromotionMaterial::test_only_from_descriptor(descriptor.clone(), now, now + 90)
                .unwrap();
        let store = PeerStore::new();
        store.enable_untrusted_discovery_candidate_mode();
        assert_eq!(
            store.admit_permissionless_descriptor(descriptor.clone(), now),
            PermissionlessNodeAdmissionOutcome::Admitted
        );
        assert!(store
            .promote_permissionless_candidate(&material, now)
            .is_ok());
        assert!(store.get_valid(&descriptor.node_id(), now).is_none());
        assert_eq!(store.cleanup_expired(now + 91), 1);
        assert!(store.permissionless_promotions.read().is_empty());
        assert!(!store.peers.read().contains_key(&descriptor.node_id()));
    }

    #[test]
    fn permissionless_probe_cannot_clear_quarantine_until_recovery() {
        let now = 1_780_100_000;
        let identity = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
        let descriptor = permissionless_descriptor_for(&identity, 8, now, "https://8.8.8.8:8422");
        let store = PeerStore::new();
        store.upsert_verified(descriptor.clone(), now).unwrap();
        for offset in 1..=PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD {
            assert!(store.record_route_forward_failure_for_descriptor(
                &descriptor,
                now + u64::from(offset),
                "request_failed",
            ));
        }
        let quarantined_at = now + u64::from(PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD);
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        store.permissionless_promotions.write().insert(
            descriptor.node_id(),
            PermissionlessPromotionGate {
                descriptor_hash: pin.descriptor_hash,
                valid_until: quarantined_at + PEER_ROUTE_FAILURE_QUARANTINE_SECS + 90,
                active: true,
                verified_control_probe: false,
                generation: 0,
            },
        );
        assert!(!store.record_permissionless_promotion_probe_verified(
            &descriptor,
            quarantined_at + PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS,
        ));
        assert!(store.record_permissionless_promotion_probe_verified(
            &descriptor,
            quarantined_at + PEER_ROUTE_FAILURE_QUARANTINE_SECS + 1,
        ));
    }
}
