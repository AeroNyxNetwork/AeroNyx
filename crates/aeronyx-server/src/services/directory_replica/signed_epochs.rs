// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/signed_epochs.rs
// ============================================
//! # Signed local policy epochs
//!
//! Owns the node-signed, hash-linked witness, route-domain, and route-domain
//! attestor policy epochs, their reconcile reports, and the opaque policy anchor
//! head and decision values.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    validate_observation_witness_policy_members, validate_route_domain_attestor_policy_members,
    validate_route_domain_policy_assignments, Digest, DirectoryReplicaStoreError, IdentityKeyPair,
    PinnedRouteDomainAssignment, Sha256, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1, DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
    DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1, DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
};

/// One node-identity-signed, hash-linked local witness admission policy.
///
/// This is operator configuration history, not a network vote, validator set,
/// fork-choice rule, consensus object, or finality certificate. Full member
/// identities remain in the host-local database and are never returned by the
/// public aggregate status endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DirectoryObservationWitnessPolicyEpoch {
    pub(super) epoch: u64,
    pub(super) previous_policy_digest: [u8; 32],
    pub(super) activated_at: u64,
    pub(super) witness_node_ids: Vec<[u8; 32]>,
    pub(super) minimum_witnesses: usize,
    pub(super) signer_node_id: [u8; 32],
    pub(super) signature: [u8; 64],
}

impl DirectoryObservationWitnessPolicyEpoch {
    pub(super) fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        witness_node_ids: Vec<[u8; 32]>,
        minimum_witnesses: usize,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            witness_node_ids,
            minimum_witnesses,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    pub(super) fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_observation_witness_policy_members(&self.witness_node_ids, self.minimum_witnesses)
    }

    pub(super) fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.witness_node_ids.len() * 32);
        bytes.extend_from_slice(b"AeroNyx-DirectoryObservationWitnessPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.extend_from_slice(&(self.minimum_witnesses as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.witness_node_ids.len() as u64).to_le_bytes());
        for witness_node_id in &self.witness_node_ids {
            bytes.extend_from_slice(witness_node_id);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling validated runtime pins into the signed local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationWitnessPolicyReconcileReport {
    /// True only when pins or threshold created a new durable epoch.
    pub(crate) appended: bool,
    /// Current local policy epoch after reconciliation.
    pub(crate) epoch: u64,
    /// Content digest of the current signed policy.
    pub(crate) policy_digest: [u8; 32],
    /// Timestamp bound into the current policy.
    pub(crate) activated_at: u64,
    /// Number of canonical current witness pins.
    pub(crate) witness_members: u64,
    /// Required distinct external receipts.
    pub(crate) minimum_witnesses: u64,
}

/// One node-identity-signed, hash-linked local route-domain policy epoch.
///
/// [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] Assignments are opaque
/// operator-reviewed failure-domain groups. The signature proves what this
/// node configured and when; it does not prove legal ownership, ASN identity,
/// physical independence, honest operation, consensus, or Sybil resistance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DirectoryRouteDomainPolicyEpoch {
    pub(super) epoch: u64,
    pub(super) previous_policy_digest: [u8; 32],
    pub(super) activated_at: u64,
    pub(super) strict_required: bool,
    pub(super) assignments: Vec<PinnedRouteDomainAssignment>,
    pub(super) signer_node_id: [u8; 32],
    pub(super) signature: [u8; 64],
}

impl DirectoryRouteDomainPolicyEpoch {
    pub(super) fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        strict_required: bool,
        assignments: Vec<PinnedRouteDomainAssignment>,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            strict_required,
            assignments,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    pub(super) fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_route_domain_policy_assignments(&self.assignments, self.strict_required)
    }

    pub(super) fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.assignments.len() * 48);
        bytes.extend_from_slice(b"AeroNyx-DirectoryRouteDomainPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.push(u8::from(self.strict_required));
        bytes.extend_from_slice(&(self.assignments.len() as u64).to_le_bytes());
        for assignment in &self.assignments {
            bytes.extend_from_slice(&assignment.node_id);
            bytes.extend_from_slice(&assignment.route_domain);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling runtime route-domain pins into signed local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryRouteDomainPolicyReconcileReport {
    pub(crate) appended: bool,
    pub(crate) epoch: u64,
    pub(crate) policy_digest: [u8; 32],
    pub(crate) activated_at: u64,
    pub(crate) assignments: u64,
    pub(crate) strict_required: bool,
}

/// One node-identity-signed, hash-linked local route-domain attestor policy.
///
/// [ROUTE-DOMAIN-ATTESTOR-HISTORY 2026-08-03 by Codex] These identities are
/// host-local trust roots for validating portable route-domain attestations.
/// A valid signature proves only authorship under a locally pinned key; this
/// policy does not prove operator independence, geography, ASN ownership,
/// consensus, honest behavior, or Sybil resistance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DirectoryRouteDomainAttestorPolicyEpoch {
    pub(super) epoch: u64,
    pub(super) previous_policy_digest: [u8; 32],
    pub(super) activated_at: u64,
    pub(super) strict_required: bool,
    pub(super) attestor_node_ids: Vec<[u8; 32]>,
    pub(super) minimum_attestors: usize,
    pub(super) signer_node_id: [u8; 32],
    pub(super) signature: [u8; 64],
}

impl DirectoryRouteDomainAttestorPolicyEpoch {
    pub(super) fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        strict_required: bool,
        attestor_node_ids: Vec<[u8; 32]>,
        minimum_attestors: usize,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            strict_required,
            attestor_node_ids,
            minimum_attestors,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    pub(super) fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_route_domain_attestor_policy_members(
            &self.attestor_node_ids,
            self.minimum_attestors,
            self.strict_required,
        )
    }

    pub(super) fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(200 + self.attestor_node_ids.len() * 32);
        bytes.extend_from_slice(b"AeroNyx-DirectoryRouteDomainAttestorPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.push(u8::from(self.strict_required));
        bytes.extend_from_slice(&(self.minimum_attestors as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.attestor_node_ids.len() as u64).to_le_bytes());
        for attestor_node_id in &self.attestor_node_ids {
            bytes.extend_from_slice(attestor_node_id);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling runtime route-domain attestor pins into local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryRouteDomainAttestorPolicyReconcileReport {
    pub(crate) appended: bool,
    pub(crate) epoch: u64,
    pub(crate) policy_digest: [u8; 32],
    pub(crate) activated_at: u64,
    pub(crate) attestors: u64,
    pub(crate) minimum_attestors: u64,
    pub(crate) strict_required: bool,
}

/// Privacy-bounded current local policy head exported to pinned witnesses.
///
/// The digest commits to the complete node-signed local policy, while member
/// identities remain host-local and never enter the anchor protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationWitnessPolicyAnchor {
    pub(crate) epoch: u64,
    pub(crate) previous_policy_digest: [u8; 32],
    pub(crate) policy_digest: [u8; 32],
}

/// Result of evaluating one authenticated foreign policy-head anchor request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessPolicyAnchorDecision {
    /// Exact head was already retained or appended durably.
    Accepted,
    /// Request regressed below the latest retained observer epoch.
    Rollback,
    /// The same epoch was previously retained with another digest.
    Conflict,
    /// A forward request did not link to the immediately retained head.
    HistoryGap,
}

impl DirectoryObservationWitnessPolicyAnchorDecision {
    #[must_use]
    pub(crate) const fn outcome(self) -> u8 {
        match self {
            Self::Accepted => DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            Self::Rollback => DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
            Self::Conflict => DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            Self::HistoryGap => DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
        }
    }
}
