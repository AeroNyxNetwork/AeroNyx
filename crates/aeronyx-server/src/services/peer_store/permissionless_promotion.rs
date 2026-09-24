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

    pub(super) fn prune_expired_permissionless_promotions(&self, now: u64) -> usize {
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Never drop
        // a deny gate while its exact live descriptor remains in the peer
        // map: that would turn expiry into route authority on clock rollback.
        // Lock peers before gates, matching read-side lock order.
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
        if let Some(gate) = self.permissionless_promotions.write().get_mut(&node_id) {
            if gate.descriptor_hash == pin.descriptor_hash {
                gate.active = true;
            }
        }
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
        gate.verified_control_probe = true;
        true
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
