// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/replica_plan.rs
// ============================================
//! # Replica evidence and lifecycle planning
//!
//! Owns source-owned per-replica manifest expectations, verified inventory
//! evidence, replica policy/health/actions/targets/plans, the manifest
//! replica planner, and the evidence and plan error types.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::crypto::keys::IdentityPublicKey;

use super::error::BlindVaultError;
use super::lease_inventory::{BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest};
use super::require_non_zero;

/// Maximum replica members accepted by one local lifecycle plan.
pub const MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS: usize = 16;

/// Maximum deterministic actions emitted for one bounded replica plan.
pub const MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS: usize =
    (MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS * 2) + 1;

/// Source-owned expected state for one independently wrapped replica.
///
/// This type is deliberately not serializable. Replica object identifiers and
/// commitments are local repair material and must never become public-chain,
/// discovery, or cross-replica correlation metadata.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaManifestExpectation {
    node_id: [u8; 32],
    lease_id: [u8; 32],
    object_count: u64,
    ciphertext_bytes: u64,
    inventory_commitment: [u8; 32],
}

impl BlindVaultReplicaManifestExpectation {
    /// Creates one validated source-side replica expectation.
    pub fn new(
        node_id: [u8; 32],
        lease_id: [u8; 32],
        object_count: u64,
        ciphertext_bytes: u64,
        inventory_commitment: [u8; 32],
    ) -> Result<Self, BlindVaultReplicaEvidenceError> {
        require_non_zero("node_id", &node_id)?;
        require_non_zero("lease_id", &lease_id)?;
        require_non_zero("inventory_commitment", &inventory_commitment)?;
        if (object_count == 0) != (ciphertext_bytes == 0) {
            return Err(BlindVaultReplicaEvidenceError::InvalidExpectation);
        }
        Ok(Self {
            node_id,
            lease_id,
            object_count,
            ciphertext_bytes,
            inventory_commitment,
        })
    }

    /// Descriptor identity expected to sign this replica's observation.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease identifier.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Expected number of still-live encrypted objects.
    #[must_use]
    pub const fn object_count(&self) -> u64 {
        self.object_count
    }

    /// Expected total padded ciphertext bytes.
    #[must_use]
    pub const fn ciphertext_bytes(&self) -> u64 {
        self.ciphertext_bytes
    }

    /// Expected domain-separated root for this replica's own manifest.
    #[must_use]
    pub const fn inventory_commitment(&self) -> [u8; 32] {
        self.inventory_commitment
    }
}

/// Verified, freshness-bounded inventory evidence for one replica.
///
/// Construction requires the exact request, expected terminal identity, and
/// source-owned manifest. A valid but divergent receipt remains evidence: the
/// planner classifies it for reconciliation instead of confusing divergence
/// with an invalid signature.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedReplicaInventory {
    node_id: [u8; 32],
    lease_id: [u8; 32],
    expires_at_ms: u64,
    observed_at_ms: u64,
    expected_object_count: u64,
    observed_object_count: u64,
    expected_ciphertext_bytes: u64,
    observed_ciphertext_bytes: u64,
    expected_inventory_commitment: [u8; 32],
    observed_inventory_commitment: [u8; 32],
    matches_expected_manifest: bool,
}

impl BlindVaultVerifiedReplicaInventory {
    /// Verifies one terminal receipt against its exact request and local
    /// per-replica expectation.
    ///
    /// `maximum_receipt_age_ms` must be non-zero. Future observations are
    /// accepted only within `maximum_future_clock_skew_ms`. Lease expiry is not
    /// rejected here because the planner must convert signed expired evidence
    /// into an explicit replacement action.
    pub fn verify(
        receipt: &BlindVaultLeaseInventoryReceipt,
        request: &BlindVaultLeaseInventoryRequest,
        expectation: &BlindVaultReplicaManifestExpectation,
        now_ms: u64,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaEvidenceError> {
        if now_ms == 0 || maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaEvidenceError::InvalidFreshnessPolicy);
        }
        if request.lease_id != expectation.lease_id {
            return Err(BlindVaultReplicaEvidenceError::RequestMismatch);
        }
        if receipt.node_id != expectation.node_id {
            return Err(BlindVaultReplicaEvidenceError::TerminalIdentityMismatch);
        }
        let terminal_key = IdentityPublicKey::from_bytes(&expectation.node_id)
            .map_err(|_| BlindVaultReplicaEvidenceError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if !receipt.matches_inventory(request) {
            return Err(BlindVaultReplicaEvidenceError::RequestMismatch);
        }

        if receipt.observed_at_ms > now_ms {
            if receipt.observed_at_ms - now_ms > maximum_future_clock_skew_ms {
                return Err(BlindVaultReplicaEvidenceError::ReceiptFromFuture);
            }
        } else if now_ms - receipt.observed_at_ms > maximum_receipt_age_ms {
            return Err(BlindVaultReplicaEvidenceError::ReceiptStale);
        }

        let matches_expected_manifest = receipt.live_object_count == expectation.object_count
            && receipt.live_ciphertext_bytes == expectation.ciphertext_bytes
            && receipt.inventory_commitment == expectation.inventory_commitment;
        Ok(Self {
            node_id: expectation.node_id,
            lease_id: expectation.lease_id,
            expires_at_ms: receipt.expires_at_ms,
            observed_at_ms: receipt.observed_at_ms,
            expected_object_count: expectation.object_count,
            observed_object_count: receipt.live_object_count,
            expected_ciphertext_bytes: expectation.ciphertext_bytes,
            observed_ciphertext_bytes: receipt.live_ciphertext_bytes,
            expected_inventory_commitment: expectation.inventory_commitment,
            observed_inventory_commitment: receipt.inventory_commitment,
            matches_expected_manifest,
        })
    }

    /// Descriptor identity that signed the evidence.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease identifier.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Signed lease expiry observed at the terminal.
    #[must_use]
    pub const fn expires_at_ms(&self) -> u64 {
        self.expires_at_ms
    }

    /// Signed terminal observation time.
    #[must_use]
    pub const fn observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }

    /// Whether all signed aggregate fields match this replica's expectation.
    #[must_use]
    pub const fn matches_expected_manifest(&self) -> bool {
        self.matches_expected_manifest
    }

    /// Source-owned expected live object count used to bind repair evidence to
    /// the exact planner generation.
    #[must_use]
    pub const fn expected_object_count(&self) -> u64 {
        self.expected_object_count
    }

    /// Source-owned expected padded ciphertext bytes for this replica only.
    #[must_use]
    pub const fn expected_ciphertext_bytes(&self) -> u64 {
        self.expected_ciphertext_bytes
    }

    /// Source-owned per-replica manifest root. This remains local-only and is
    /// never serialized into discovery or public ledger state.
    // [BLIND-VAULT-REPLICA-WORKFLOW 2026-08-28 by Codex] Reconciliation must
    // bind to the planner's exact expectation, not merely the node and lease.
    #[must_use]
    pub const fn expected_inventory_commitment(&self) -> [u8; 32] {
        self.expected_inventory_commitment
    }
}

/// Most recent source-owned evidence for one intended replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaEvidence {
    /// A terminal-signed, request-bound, freshness-checked inventory.
    Observed(BlindVaultVerifiedReplicaInventory),
    /// No valid inventory could be obtained for this intended member.
    Unavailable {
        /// Expected descriptor identity.
        node_id: [u8; 32],
        /// Replica-local lease identifier.
        lease_id: [u8; 32],
        /// Consecutive completed observation attempts that failed.
        consecutive_failures: u16,
    },
}

impl BlindVaultReplicaEvidence {
    /// Expected descriptor identity for deterministic planning and deduplication.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        match self {
            Self::Observed(inventory) => inventory.node_id,
            Self::Unavailable { node_id, .. } => *node_id,
        }
    }

    /// Replica-local lease identifier for duplicate detection.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        match self {
            Self::Observed(inventory) => inventory.lease_id,
            Self::Unavailable { lease_id, .. } => *lease_id,
        }
    }
}

/// Source policy for maintaining independently wrapped replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaPolicy {
    /// Desired replica floor. Transitional extra replicas are permitted.
    pub target_replicas: u8,
    /// Minimum live replicas whose signed inventory matches local expectations.
    pub minimum_healthy_replicas: u8,
    /// Lead time before expiry at which a live lease should be renewed.
    pub renewal_lead_ms: u64,
    /// Consecutive failed observations before replacing a replica.
    pub replace_after_consecutive_failures: u16,
}

impl BlindVaultReplicaPolicy {
    fn validate(self) -> Result<(), BlindVaultReplicaPlanError> {
        let target_replicas = usize::from(self.target_replicas);
        if target_replicas == 0
            || target_replicas > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS
            || self.minimum_healthy_replicas == 0
            || self.minimum_healthy_replicas > self.target_replicas
            || self.renewal_lead_ms == 0
            || self.replace_after_consecutive_failures == 0
        {
            return Err(BlindVaultReplicaPlanError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Aggregate safety state of a source's current replica set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaPlanHealth {
    /// Matching live evidence meets policy and no action is due.
    Healthy,
    /// Matching live evidence meets policy and only lease renewal is due.
    MaintenanceDue,
    /// Matching live evidence meets policy but repair or membership work is due.
    Degraded,
    /// Too few live, matching replicas exist to satisfy the configured quorum.
    QuorumUnavailable,
}

/// One declarative source-side lifecycle action.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaAction {
    /// Renew one still-live lease before its current generation expires.
    RenewLease {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_expires_at_ms: u64,
    },
    /// Rebuild one divergent replica from this source's private manifest.
    ReconcileInventory {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_object_count: u64,
        observed_object_count: u64,
        expected_ciphertext_bytes: u64,
        observed_ciphertext_bytes: u64,
        expected_inventory_commitment: [u8; 32],
        observed_inventory_commitment: [u8; 32],
    },
    /// Retry a temporarily unavailable member without changing membership.
    RetryObservation {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    /// Replace an expired or repeatedly unreachable member.
    ReplaceReplica {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    /// Provision anonymous replicas until the configured floor is restored.
    ProvisionReplicas { count: u8 },
}

impl std::fmt::Debug for BlindVaultReplicaAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // [BLIND-VAULT-PRIVACY-SAFE-DEBUG 2026-08-30 by Codex] Lifecycle
        // diagnostics expose only the operation class. Topology identifiers,
        // inventory commitments, and replica measurements remain source-local.
        formatter.write_str(match self {
            Self::RenewLease { .. } => "RenewLease",
            Self::ReconcileInventory { .. } => "ReconcileInventory",
            Self::RetryObservation { .. } => "RetryObservation",
            Self::ReplaceReplica { .. } => "ReplaceReplica",
            Self::ProvisionReplicas { .. } => "ProvisionReplicas",
        })
    }
}

/// Replica-local terminal target shared by lifecycle actions.
///
/// [BLIND-VAULT-REPLICA-TARGET 2026-08-29 by Codex] Keeping this identity in
/// the domain model lets workflow schedulers enforce per-lease single-flight
/// without duplicating action matching or exposing any user-level identity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlindVaultReplicaTarget {
    node_id: [u8; 32],
    lease_id: [u8; 32],
}

impl std::fmt::Debug for BlindVaultReplicaTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaTarget")
            .field("node_id", &"[REDACTED]")
            .field("lease_id", &"[REDACTED]")
            .finish()
    }
}

impl BlindVaultReplicaTarget {
    /// Terminal descriptor identity.
    #[must_use]
    pub const fn node_id(self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease identity.
    #[must_use]
    pub const fn lease_id(self) -> [u8; 32] {
        self.lease_id
    }
}

impl BlindVaultReplicaAction {
    /// Returns the exact terminal/lease target, if the action has one.
    ///
    /// Aggregate anonymous provisioning has no existing target and returns
    /// `None`; the resulting admitted replicas are verified separately.
    #[must_use]
    pub const fn target(self) -> Option<BlindVaultReplicaTarget> {
        let (node_id, lease_id) = match self {
            Self::RenewLease {
                node_id, lease_id, ..
            }
            | Self::ReconcileInventory {
                node_id, lease_id, ..
            }
            | Self::RetryObservation { node_id, lease_id }
            | Self::ReplaceReplica { node_id, lease_id } => (node_id, lease_id),
            Self::ProvisionReplicas { .. } => return None,
        };
        Some(BlindVaultReplicaTarget { node_id, lease_id })
    }
}

/// Deterministic, declarative result of one replica lifecycle evaluation.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaPlan {
    /// Safety state after evaluating the supplied evidence.
    pub health: BlindVaultReplicaPlanHealth,
    /// Number of intended members represented by the supplied evidence.
    pub configured_replicas: u8,
    /// Number of verified observations for leases that have not expired.
    pub live_verified_replicas: u8,
    /// Number of live observations matching their own expected manifests.
    pub live_matching_replicas: u8,
    /// Deterministic local actions; no repair source is selected here.
    pub actions: Vec<BlindVaultReplicaAction>,
}

impl std::fmt::Debug for BlindVaultReplicaPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPlan")
            .field("health", &self.health)
            .field("configured_replicas", &self.configured_replicas)
            .field("live_verified_replicas", &self.live_verified_replicas)
            .field("live_matching_replicas", &self.live_matching_replicas)
            .field("action_count", &self.actions.len())
            .field("actions", &"[REDACTED]")
            .finish()
    }
}

impl BlindVaultReplicaPlan {
    /// Validates the internally consistent shape of one declarative plan.
    ///
    /// [BLIND-VAULT-PLAN-SHAPE 2026-08-28 by Codex] Plan fields remain public
    /// for language-neutral adapters, so every execution boundary must reject
    /// impossible count relationships, contradictory health/action states,
    /// oversized action sets, and unusable action targets before authorizing
    /// any network work.
    pub fn validate_shape(&self) -> Result<(), BlindVaultReplicaPlanError> {
        if usize::from(self.configured_replicas) > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS {
            return Err(BlindVaultReplicaPlanError::TooManyReplicas {
                actual: usize::from(self.configured_replicas),
            });
        }
        if self.actions.len() > MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS {
            return Err(BlindVaultReplicaPlanError::TooManyActions {
                actual: self.actions.len(),
            });
        }
        if self.live_verified_replicas > self.configured_replicas
            || self.live_matching_replicas > self.live_verified_replicas
        {
            return Err(BlindVaultReplicaPlanError::InconsistentPlan);
        }

        let has_actions = !self.actions.is_empty();
        let only_renewals = has_actions
            && self
                .actions
                .iter()
                .all(|action| matches!(action, BlindVaultReplicaAction::RenewLease { .. }));
        let health_matches_actions = match self.health {
            BlindVaultReplicaPlanHealth::Healthy => !has_actions,
            BlindVaultReplicaPlanHealth::MaintenanceDue => only_renewals,
            BlindVaultReplicaPlanHealth::Degraded
            | BlindVaultReplicaPlanHealth::QuorumUnavailable => has_actions && !only_renewals,
        };
        if !health_matches_actions
            || self.actions.iter().any(replica_action_has_invalid_target)
            || !replica_actions_are_consistent(&self.actions)
        {
            return Err(BlindVaultReplicaPlanError::InconsistentPlan);
        }
        Ok(())
    }
}

fn replica_action_has_invalid_target(action: &BlindVaultReplicaAction) -> bool {
    match action {
        BlindVaultReplicaAction::RenewLease {
            node_id,
            lease_id,
            expected_expires_at_ms,
        } => *node_id == [0; 32] || *lease_id == [0; 32] || *expected_expires_at_ms == 0,
        BlindVaultReplicaAction::ReconcileInventory {
            node_id, lease_id, ..
        }
        | BlindVaultReplicaAction::RetryObservation { node_id, lease_id }
        | BlindVaultReplicaAction::ReplaceReplica { node_id, lease_id } => {
            *node_id == [0; 32] || *lease_id == [0; 32]
        }
        BlindVaultReplicaAction::ProvisionReplicas { count } => {
            *count == 0 || usize::from(*count) > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS
        }
    }
}

fn replica_actions_are_consistent(actions: &[BlindVaultReplicaAction]) -> bool {
    // [BLIND-VAULT-PLAN-ACTION-CONSISTENCY 2026-08-28 by Codex] One member
    // may reconcile and renew in the same generation. Retry and replacement
    // are exclusive, and provisioning is one aggregate planner instruction.
    const RENEW: u8 = 1 << 0;
    const RECONCILE: u8 = 1 << 1;
    const RETRY: u8 = 1 << 2;
    const REPLACE: u8 = 1 << 3;
    const EXCLUSIVE: u8 = RETRY | REPLACE;

    let mut targets = BTreeMap::<[u8; 32], ([u8; 32], u8)>::new();
    let mut provision_seen = false;
    for action in actions {
        let (node_id, lease_id, action_flag) = match action {
            BlindVaultReplicaAction::RenewLease {
                node_id, lease_id, ..
            } => (*node_id, *lease_id, RENEW),
            BlindVaultReplicaAction::ReconcileInventory {
                node_id, lease_id, ..
            } => (*node_id, *lease_id, RECONCILE),
            BlindVaultReplicaAction::RetryObservation { node_id, lease_id } => {
                (*node_id, *lease_id, RETRY)
            }
            BlindVaultReplicaAction::ReplaceReplica { node_id, lease_id } => {
                (*node_id, *lease_id, REPLACE)
            }
            BlindVaultReplicaAction::ProvisionReplicas { .. } => {
                if provision_seen {
                    return false;
                }
                provision_seen = true;
                continue;
            }
        };

        let entry = targets.entry(node_id).or_insert((lease_id, 0));
        if entry.0 != lease_id
            || entry.1 & action_flag != 0
            || entry.1 & EXCLUSIVE != 0
            || (action_flag & EXCLUSIVE != 0 && entry.1 != 0)
            || (action_flag == RENEW && entry.1 & RECONCILE != 0)
        {
            return false;
        }
        entry.1 |= action_flag;
    }
    true
}

/// Replaceable capability for source-owned replica lifecycle planning.
pub trait BlindVaultReplicaPlanner {
    /// Produces a deterministic plan from already verified or unavailable
    /// per-replica evidence.
    fn plan(
        &self,
        now_ms: u64,
        evidence: &[BlindVaultReplicaEvidence],
    ) -> Result<BlindVaultReplicaPlan, BlindVaultReplicaPlanError>;
}

/// Manifest-bound planner that never infers truth from a replica majority.
///
/// [BLIND-VAULT-REPLICA-PLANNER 2026-08-28 by Codex] Each node is checked only
/// against its own independently randomized expected manifest. The planner
/// intentionally does not select a repair source: if matching live evidence
/// falls below policy, callers receive `QuorumUnavailable` and must not treat
/// another replica's object identifiers as authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultManifestReplicaPlanner {
    policy: BlindVaultReplicaPolicy,
}

impl BlindVaultManifestReplicaPlanner {
    /// Creates a planner after validating all policy invariants.
    pub fn new(policy: BlindVaultReplicaPolicy) -> Result<Self, BlindVaultReplicaPlanError> {
        policy.validate()?;
        Ok(Self { policy })
    }

    /// Returns the immutable policy used for planning.
    #[must_use]
    pub const fn policy(&self) -> BlindVaultReplicaPolicy {
        self.policy
    }
}

impl BlindVaultReplicaPlanner for BlindVaultManifestReplicaPlanner {
    fn plan(
        &self,
        now_ms: u64,
        evidence: &[BlindVaultReplicaEvidence],
    ) -> Result<BlindVaultReplicaPlan, BlindVaultReplicaPlanError> {
        self.policy.validate()?;
        if now_ms == 0 {
            return Err(BlindVaultReplicaPlanError::TimestampOutOfRange);
        }
        if evidence.len() > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS {
            return Err(BlindVaultReplicaPlanError::TooManyReplicas {
                actual: evidence.len(),
            });
        }
        let renewal_deadline = now_ms
            .checked_add(self.policy.renewal_lead_ms)
            .ok_or(BlindVaultReplicaPlanError::TimestampOutOfRange)?;

        let mut node_ids = BTreeSet::new();
        let mut lease_ids = BTreeSet::new();
        for member in evidence {
            let node_id = member.node_id();
            let lease_id = member.lease_id();
            if node_id == [0; 32] || lease_id == [0; 32] {
                return Err(BlindVaultReplicaPlanError::InvalidEvidenceIdentity);
            }
            if !node_ids.insert(node_id) {
                return Err(BlindVaultReplicaPlanError::DuplicateNode);
            }
            if !lease_ids.insert(lease_id) {
                return Err(BlindVaultReplicaPlanError::DuplicateLease);
            }
            if matches!(
                member,
                BlindVaultReplicaEvidence::Unavailable {
                    consecutive_failures: 0,
                    ..
                }
            ) {
                return Err(BlindVaultReplicaPlanError::InvalidUnavailableEvidence);
            }
        }

        let mut ordered = evidence.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|member| member.node_id());
        let mut live_verified_replicas = 0_u8;
        let mut live_matching_replicas = 0_u8;
        let mut actions = Vec::new();

        for member in ordered {
            match member {
                BlindVaultReplicaEvidence::Observed(inventory) => {
                    if inventory.expires_at_ms <= now_ms {
                        actions.push(BlindVaultReplicaAction::ReplaceReplica {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                        });
                        continue;
                    }
                    live_verified_replicas += 1;
                    // [BLIND-VAULT-RENEW-BEFORE-REPAIR 2026-08-29 by Codex]
                    // A near-expiry lease is extended before source-owned
                    // reconciliation starts, preventing a valid repair from
                    // crossing the old lease deadline midway through work.
                    if inventory.expires_at_ms <= renewal_deadline {
                        actions.push(BlindVaultReplicaAction::RenewLease {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                            expected_expires_at_ms: inventory.expires_at_ms,
                        });
                    }
                    if inventory.matches_expected_manifest {
                        live_matching_replicas += 1;
                    } else {
                        actions.push(BlindVaultReplicaAction::ReconcileInventory {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                            expected_object_count: inventory.expected_object_count,
                            observed_object_count: inventory.observed_object_count,
                            expected_ciphertext_bytes: inventory.expected_ciphertext_bytes,
                            observed_ciphertext_bytes: inventory.observed_ciphertext_bytes,
                            expected_inventory_commitment: inventory.expected_inventory_commitment,
                            observed_inventory_commitment: inventory.observed_inventory_commitment,
                        });
                    }
                }
                BlindVaultReplicaEvidence::Unavailable {
                    node_id,
                    lease_id,
                    consecutive_failures,
                } => {
                    if *consecutive_failures >= self.policy.replace_after_consecutive_failures {
                        actions.push(BlindVaultReplicaAction::ReplaceReplica {
                            node_id: *node_id,
                            lease_id: *lease_id,
                        });
                    } else {
                        actions.push(BlindVaultReplicaAction::RetryObservation {
                            node_id: *node_id,
                            lease_id: *lease_id,
                        });
                    }
                }
            }
        }

        if evidence.len() < usize::from(self.policy.target_replicas) {
            let missing = usize::from(self.policy.target_replicas) - evidence.len();
            actions.push(BlindVaultReplicaAction::ProvisionReplicas {
                count: u8::try_from(missing)
                    .map_err(|_| BlindVaultReplicaPlanError::TooManyReplicas { actual: missing })?,
            });
        }

        let health = if live_matching_replicas < self.policy.minimum_healthy_replicas {
            BlindVaultReplicaPlanHealth::QuorumUnavailable
        } else if actions.is_empty() {
            BlindVaultReplicaPlanHealth::Healthy
        } else if actions
            .iter()
            .all(|action| matches!(action, BlindVaultReplicaAction::RenewLease { .. }))
        {
            BlindVaultReplicaPlanHealth::MaintenanceDue
        } else {
            BlindVaultReplicaPlanHealth::Degraded
        };

        let plan = BlindVaultReplicaPlan {
            health,
            configured_replicas: u8::try_from(evidence.len()).map_err(|_| {
                BlindVaultReplicaPlanError::TooManyReplicas {
                    actual: evidence.len(),
                }
            })?,
            live_verified_replicas,
            live_matching_replicas,
            actions,
        };
        plan.validate_shape()?;
        Ok(plan)
    }
}

/// Fail-closed verification errors for private replica evidence.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaEvidenceError {
    /// The underlying inventory contract was malformed or unauthenticated.
    #[error("blind vault replica evidence violated the inventory protocol")]
    BlindVault(#[from] BlindVaultError),
    /// Expected object and byte aggregates were internally inconsistent.
    #[error("blind vault replica manifest expectation is invalid")]
    InvalidExpectation,
    /// Receipt terminal identity did not match the intended replica member.
    #[error("blind vault replica terminal identity mismatch")]
    TerminalIdentityMismatch,
    /// The expected terminal identity could not be reconstructed as Ed25519.
    #[error("invalid blind vault replica terminal identity")]
    InvalidTerminalIdentity,
    /// Receipt did not answer the exact request and replica-local lease.
    #[error("blind vault replica inventory request mismatch")]
    RequestMismatch,
    /// Caller supplied an unusable local freshness policy.
    #[error("blind vault replica evidence freshness policy is invalid")]
    InvalidFreshnessPolicy,
    /// Receipt observation exceeded allowed future clock skew.
    #[error("blind vault replica inventory receipt is from the future")]
    ReceiptFromFuture,
    /// Receipt observation was older than the configured evidence lifetime.
    #[error("blind vault replica inventory receipt is stale")]
    ReceiptStale,
}

/// Invalid policy or evidence-set errors for replica lifecycle planning.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaPlanError {
    /// Policy quorum, target, renewal, or replacement values were invalid.
    #[error("blind vault replica policy is invalid")]
    InvalidPolicy,
    /// Evidence exceeded the bounded local planning set.
    #[error("blind vault replica set contains too many members: {actual}")]
    TooManyReplicas { actual: usize },
    /// Declarative actions exceeded the bounded planner output.
    #[error("blind vault replica plan contains too many actions: {actual}")]
    TooManyActions { actual: usize },
    /// Counts, health, actions, or action targets contradicted one another.
    #[error("blind vault replica plan is internally inconsistent")]
    InconsistentPlan,
    /// Two evidence entries declared the same terminal descriptor identity.
    #[error("blind vault replica set contains a duplicate node")]
    DuplicateNode,
    /// Two evidence entries declared the same replica-local lease.
    #[error("blind vault replica set contains a duplicate lease")]
    DuplicateLease,
    /// An evidence member used an all-zero terminal or lease identity.
    #[error("blind vault replica evidence identity is invalid")]
    InvalidEvidenceIdentity,
    /// Unavailable evidence must represent at least one completed failure.
    #[error("blind vault unavailable replica evidence is invalid")]
    InvalidUnavailableEvidence,
    /// Planning time or renewal-window arithmetic was not representable.
    #[error("blind vault replica planning timestamp is out of range")]
    TimestampOutOfRange,
}
