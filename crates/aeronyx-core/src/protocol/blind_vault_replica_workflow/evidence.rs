// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/evidence.rs
// ============================================
//! Cryptographic and manifest evidence for source-owned replica work.
//!
//! [BLIND-VAULT-REPLICA-EVIDENCE 2026-08-28 by Codex] Constructors in this
//! module are the only path to action evidence. They verify terminal identity,
//! exact request binding, live lease state, and source-owned manifests without
//! retaining credentials, ciphertext, object identifiers, or social metadata.
//!
//! Last Modified: v1.1.0-DistilledAdmissionEvidence - Split admission and
//! inventory verification so one-time credentials can be discarded early.

use std::collections::BTreeSet;

use crate::crypto::keys::IdentityPublicKey;

use super::{
    require_timestamp, BlindVaultReplacementRetirementPermit, BlindVaultReplicaActionEvidence,
    BlindVaultReplicaActionEvidenceKind, BlindVaultReplicaWorkId, BlindVaultReplicaWorkflowError,
    BlindVaultVerifiedProvisionedReplica, BlindVaultVerifiedReplicaAdmission,
    BlindVaultVerifiedRetiredReplica,
};
use crate::protocol::blind_vault::{
    BlindVaultBlindLeaseAcceptedReceipt, BlindVaultBlindLeaseAdmissionRequest,
    BlindVaultBlindLeaseRenewalRequest, BlindVaultBlindLeaseRenewedReceipt,
    BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt, BlindVaultReplicaAction,
    BlindVaultVerifiedReplicaInventory, MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS,
};

impl BlindVaultVerifiedReplicaAdmission {
    /// Verifies and distills one exact anonymous admission request and receipt.
    ///
    /// The returned value deliberately excludes the admission token, token
    /// signature, lease keys, read capability, and request identifier.
    pub fn verify(
        request: &BlindVaultBlindLeaseAdmissionRequest,
        receipt: &BlindVaultBlindLeaseAcceptedReceipt,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        request.admission.validate_shape()?;
        request
            .lease
            .validate_and_verify(now_ms, maximum_lease_ttl_ms)?;
        let terminal_key = IdentityPublicKey::from_bytes(&receipt.node_id)
            .map_err(|_| BlindVaultReplicaWorkflowError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if !receipt.matches_admission(request) {
            return Err(BlindVaultReplicaWorkflowError::EvidenceRequestMismatch);
        }
        require_not_too_far_in_future(
            receipt.accepted_at_ms,
            now_ms,
            maximum_future_clock_skew_ms,
        )?;
        if receipt.lease_expires_at_ms <= now_ms {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        Ok(Self {
            node_id: receipt.node_id,
            lease_id: receipt.lease_id,
            lease_expires_at_ms: receipt.lease_expires_at_ms,
            accepted_at_ms: receipt.accepted_at_ms,
        })
    }
}

impl BlindVaultVerifiedProvisionedReplica {
    /// Verifies the exact admission receipt and requires a matching live
    /// manifest observation from the same terminal and lease.
    ///
    /// This compatibility entry point now delegates through the distilled
    /// admission model so sequential adapters can use the same invariants.
    pub fn verify(
        request: &BlindVaultBlindLeaseAdmissionRequest,
        receipt: &BlindVaultBlindLeaseAcceptedReceipt,
        inventory: &BlindVaultVerifiedReplicaInventory,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        let admission = BlindVaultVerifiedReplicaAdmission::verify(
            request,
            receipt,
            now_ms,
            maximum_lease_ttl_ms,
            maximum_future_clock_skew_ms,
        )?;
        Self::verify_admitted_inventory(&admission, inventory, now_ms)
    }

    /// Completes provisioning from credential-free admission and inventory.
    ///
    /// [BLIND-VAULT-DISTILLED-ADMISSION 2026-08-29 by Codex] Callers may drop
    /// the complete blind credential immediately after admission verification;
    /// only this bounded lifecycle summary crosses the inventory wait.
    pub fn verify_admitted_inventory(
        admission: &BlindVaultVerifiedReplicaAdmission,
        inventory: &BlindVaultVerifiedReplicaInventory,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        if inventory.node_id() != admission.node_id
            || inventory.lease_id() != admission.lease_id
            || inventory.expires_at_ms() != admission.lease_expires_at_ms
            || inventory.observed_at_ms() < admission.accepted_at_ms
            || inventory.expires_at_ms() <= now_ms
            || !inventory.matches_expected_manifest()
        {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        Ok(Self {
            node_id: admission.node_id,
            lease_id: admission.lease_id,
            accepted_at_ms: admission.accepted_at_ms,
            observed_at_ms: inventory.observed_at_ms(),
        })
    }
}

impl BlindVaultVerifiedRetiredReplica {
    /// Verifies one exact old-lease retirement against the intended terminal.
    ///
    /// [BLIND-VAULT-REPLACEMENT-RETIREMENT 2026-08-28 by Codex] A replacement
    /// is not complete merely because a new replica exists. The old terminal
    /// must sign an exact-request receipt proving its complete lease was
    /// durably retired before the workflow accepts replacement evidence.
    pub fn verify(
        expected_node_id: [u8; 32],
        request: &BlindVaultLeaseRetireRequest,
        receipt: &BlindVaultLeaseRetiredReceipt,
        now_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        let terminal_key = IdentityPublicKey::from_bytes(&expected_node_id)
            .map_err(|_| BlindVaultReplicaWorkflowError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.node_id != expected_node_id
            || receipt.lease_id != request.lease_id
            || !receipt.matches_retire(request)
            || receipt
                .retired_at_ms
                .saturating_add(maximum_future_clock_skew_ms)
                < request.requested_at_ms
        {
            return Err(BlindVaultReplicaWorkflowError::EvidenceRequestMismatch);
        }
        require_not_too_far_in_future(receipt.retired_at_ms, now_ms, maximum_future_clock_skew_ms)?;
        Ok(Self {
            node_id: receipt.node_id,
            lease_id: receipt.lease_id,
            retired_at_ms: receipt.retired_at_ms,
        })
    }
}

impl BlindVaultReplicaActionEvidence {
    /// Verifies a terminal-signed renewal receipt against the exact request.
    pub fn verify_renewal(
        request: &BlindVaultBlindLeaseRenewalRequest,
        receipt: &BlindVaultBlindLeaseRenewedReceipt,
        now_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        request.admission.validate_shape()?;
        let terminal_key = IdentityPublicKey::from_bytes(&receipt.node_id)
            .map_err(|_| BlindVaultReplicaWorkflowError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if !receipt.matches_renewal(request) {
            return Err(BlindVaultReplicaWorkflowError::EvidenceRequestMismatch);
        }
        require_not_too_far_in_future(receipt.renewed_at_ms, now_ms, maximum_future_clock_skew_ms)?;
        if receipt.renewed_expires_at_ms <= now_ms {
            return Err(BlindVaultReplicaWorkflowError::EvidenceExpired);
        }
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::LeaseRenewed {
                node_id: receipt.node_id,
                lease_id: receipt.lease_id,
                previous_expires_at_ms: receipt.previous_expires_at_ms,
            },
            verified_at_ms: now_ms,
        })
    }

    /// Builds reconciliation evidence from a live, matching, already verified
    /// terminal inventory.
    pub fn inventory_reconciled(
        inventory: BlindVaultVerifiedReplicaInventory,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        validate_live_inventory(&inventory, now_ms, true)?;
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::InventoryReconciled {
                node_id: inventory.node_id(),
                lease_id: inventory.lease_id(),
                expected_object_count: inventory.expected_object_count(),
                expected_ciphertext_bytes: inventory.expected_ciphertext_bytes(),
                expected_inventory_commitment: inventory.expected_inventory_commitment(),
            },
            verified_at_ms: now_ms,
        })
    }

    /// Builds recovered-observation evidence. Divergence remains evidence and
    /// becomes reconciliation work on the mandatory fresh plan.
    pub fn observation_recovered(
        inventory: BlindVaultVerifiedReplicaInventory,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        validate_live_inventory(&inventory, now_ms, false)?;
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::ObservationRecovered {
                node_id: inventory.node_id(),
                lease_id: inventory.lease_id(),
            },
            verified_at_ms: now_ms,
        })
    }

    /// Legacy source-compatible replacement entry point.
    ///
    /// [BLIND-VAULT-REPLACEMENT-RETIREMENT 2026-08-28 by Codex] The previous
    /// contract could mark replacement complete after provisioning the new
    /// replica while leaving the old lease live. Existing callers continue to
    /// compile, but this unsafe half-complete transition now fails closed. Use
    /// [`Self::replica_replaced_with_retirement`] with verified terminal proof.
    pub fn replica_replaced(
        replaced_node_id: [u8; 32],
        replaced_lease_id: [u8; 32],
        replacement: BlindVaultVerifiedProvisionedReplica,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        let _ = (replaced_node_id, replaced_lease_id, replacement);
        Err(BlindVaultReplicaWorkflowError::RetirementEvidenceRequired)
    }

    /// Confirms replacement only after both sides of the lifecycle transition.
    ///
    /// The new independently wrapped replica must have verified admission and
    /// a matching live manifest, while the old terminal must have signed an
    /// exact-request complete lease-retirement receipt.
    pub fn replica_replaced_with_retirement(
        replacement: BlindVaultVerifiedProvisionedReplica,
        retirement: BlindVaultVerifiedRetiredReplica,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        if replacement.node_id == retirement.node_id
            || replacement.lease_id == retirement.lease_id
            || replacement.observed_at_ms > now_ms
            // [BLIND-VAULT-REPLACEMENT-RETIREMENT-PERMIT 2026-08-29 by Codex]
            // Preserve the source-compatible constructor but fail closed when
            // the old lease was retired before the replacement became live.
            || replacement.observed_at_ms > retirement.retired_at_ms
            || retirement.retired_at_ms > now_ms
        {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::ReplicaReplaced {
                replaced_node_id: retirement.node_id,
                replaced_lease_id: retirement.lease_id,
            },
            verified_at_ms: now_ms,
        })
    }

    /// Confirms replacement through an active evidence-backed retirement gate.
    ///
    /// [BLIND-VAULT-REPLACEMENT-RETIREMENT-PERMIT 2026-08-29 by Codex] The
    /// terminal retirement must occur after the source issued the permit and
    /// must match its exact old node/lease target. The permit already proves a
    /// distinct replacement had a fresh matching inventory in this attempt.
    pub fn replica_replaced_with_retirement_permit(
        permit: &BlindVaultReplacementRetirementPermit,
        retirement: BlindVaultVerifiedRetiredReplica,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        if retirement.node_id != permit.replaced_node_id
            || retirement.lease_id != permit.replaced_lease_id
            || retirement.retired_at_ms < permit.authorized_at_ms
            || retirement.retired_at_ms > now_ms
        {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::ReplicaReplacedWithPermit {
                work_id: permit.work_id,
                attempt: permit.attempt,
                replaced_node_id: retirement.node_id,
                replaced_lease_id: retirement.lease_id,
            },
            verified_at_ms: now_ms,
        })
    }

    /// Confirms aggregate provisioning from distinct, independently verified
    /// new replicas. No cross-replica commitments are retained.
    pub fn replicas_provisioned(
        replicas: &[BlindVaultVerifiedProvisionedReplica],
        expected_count: u8,
        now_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        if expected_count == 0
            || usize::from(expected_count) > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS
            || replicas.len() != usize::from(expected_count)
        {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        let mut node_ids = BTreeSet::new();
        let mut lease_ids = BTreeSet::new();
        for replica in replicas {
            if replica.observed_at_ms > now_ms
                || !node_ids.insert(replica.node_id)
                || !lease_ids.insert(replica.lease_id)
            {
                return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
            }
        }
        Ok(Self {
            kind: BlindVaultReplicaActionEvidenceKind::ReplicasProvisioned {
                count: expected_count,
            },
            verified_at_ms: now_ms,
        })
    }

    /// Source verification time attached to the accepted evidence.
    #[must_use]
    pub const fn verified_at_ms(&self) -> u64 {
        self.verified_at_ms
    }

    pub(super) fn matches(&self, action: &BlindVaultReplicaAction) -> bool {
        match (&self.kind, action) {
            (
                BlindVaultReplicaActionEvidenceKind::LeaseRenewed {
                    node_id,
                    lease_id,
                    previous_expires_at_ms,
                },
                BlindVaultReplicaAction::RenewLease {
                    node_id: expected_node_id,
                    lease_id: expected_lease_id,
                    expected_expires_at_ms,
                },
            ) => {
                node_id == expected_node_id
                    && lease_id == expected_lease_id
                    && previous_expires_at_ms == expected_expires_at_ms
            }
            (
                BlindVaultReplicaActionEvidenceKind::InventoryReconciled {
                    node_id,
                    lease_id,
                    expected_object_count,
                    expected_ciphertext_bytes,
                    expected_inventory_commitment,
                },
                BlindVaultReplicaAction::ReconcileInventory {
                    node_id: expected_node_id,
                    lease_id: expected_lease_id,
                    expected_object_count: action_object_count,
                    expected_ciphertext_bytes: action_ciphertext_bytes,
                    expected_inventory_commitment: action_inventory_commitment,
                    ..
                },
            ) => {
                node_id == expected_node_id
                    && lease_id == expected_lease_id
                    && expected_object_count == action_object_count
                    && expected_ciphertext_bytes == action_ciphertext_bytes
                    && expected_inventory_commitment == action_inventory_commitment
            }
            (
                BlindVaultReplicaActionEvidenceKind::ObservationRecovered { node_id, lease_id },
                BlindVaultReplicaAction::RetryObservation {
                    node_id: expected_node_id,
                    lease_id: expected_lease_id,
                },
            ) => node_id == expected_node_id && lease_id == expected_lease_id,
            (
                BlindVaultReplicaActionEvidenceKind::ReplicaReplaced {
                    replaced_node_id,
                    replaced_lease_id,
                }
                | BlindVaultReplicaActionEvidenceKind::ReplicaReplacedWithPermit {
                    replaced_node_id,
                    replaced_lease_id,
                    ..
                },
                BlindVaultReplicaAction::ReplaceReplica { node_id, lease_id },
            ) => replaced_node_id == node_id && replaced_lease_id == lease_id,
            (
                BlindVaultReplicaActionEvidenceKind::ReplicasProvisioned { count },
                BlindVaultReplicaAction::ProvisionReplicas {
                    count: expected_count,
                },
            ) => count == expected_count,
            _ => false,
        }
    }

    /// Binds permit-gated evidence to its exact workflow generation/attempt.
    pub(super) fn matches_attempt(&self, id: BlindVaultReplicaWorkId, attempt: u8) -> bool {
        match &self.kind {
            BlindVaultReplicaActionEvidenceKind::ReplicaReplacedWithPermit {
                work_id,
                attempt: permitted_attempt,
                ..
            } => *work_id == id && *permitted_attempt == attempt,
            _ => true,
        }
    }
}

fn require_not_too_far_in_future(
    observed_at_ms: u64,
    now_ms: u64,
    maximum_future_clock_skew_ms: u64,
) -> Result<(), BlindVaultReplicaWorkflowError> {
    if observed_at_ms > now_ms && observed_at_ms - now_ms > maximum_future_clock_skew_ms {
        return Err(BlindVaultReplicaWorkflowError::EvidenceFromFuture);
    }
    Ok(())
}

fn validate_live_inventory(
    inventory: &BlindVaultVerifiedReplicaInventory,
    now_ms: u64,
    require_matching_manifest: bool,
) -> Result<(), BlindVaultReplicaWorkflowError> {
    require_timestamp(now_ms)?;
    if inventory.expires_at_ms() <= now_ms
        || (require_matching_manifest && !inventory.matches_expected_manifest())
    {
        return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
    }
    Ok(())
}
