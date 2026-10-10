// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/audit_state.rs
// ============================================
//! # Replica audit intermediate state
//!
//! Owns the private audit accumulators for witnesses, checkpoints, policy
//! history, and the audited resolution index.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    DirectoryObservationWitnessPolicyEpoch, DirectoryObservationWitnessReceiptV1,
    DirectoryReplicaResolutionCommand, DirectoryRouteDomainAttestorPolicyEpoch,
    DirectoryRouteDomainPolicyEpoch, HashMap, HashSet,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifiedObservationWitness {
    pub(super) sequence: u64,
    pub(super) checkpoint_hash: [u8; 32],
    pub(super) observer: [u8; 32],
    pub(super) response_timestamp: u64,
    pub(super) receipt: DirectoryObservationWitnessReceiptV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ObservationCheckpointTip {
    pub(super) sequence: u64,
    pub(super) checkpoint_hash: [u8; 32],
    pub(super) observed_at: u64,
    pub(super) producer_count: u16,
    pub(super) observation_root: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ObservationWitnessAudit {
    pub(super) witnesses: u64,
    pub(super) latest_sequence: u64,
    pub(super) latest_witnesses: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct ObservationWitnessPolicyAudit {
    pub(super) epochs: u64,
    pub(super) current: Option<DirectoryObservationWitnessPolicyEpoch>,
    pub(super) current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct RouteDomainPolicyAudit {
    pub(super) epochs: u64,
    pub(super) current: Option<DirectoryRouteDomainPolicyEpoch>,
    pub(super) current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct RouteDomainAttestorPolicyAudit {
    pub(super) epochs: u64,
    pub(super) current: Option<DirectoryRouteDomainAttestorPolicyEpoch>,
    pub(super) current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct VerifiedObservationWitnessSet {
    pub(super) sequence: u64,
    pub(super) witness_node_ids: Vec<[u8; 32]>,
    pub(super) receipts: Vec<DirectoryObservationWitnessReceiptV1>,
}

#[derive(Debug, Default)]
pub(super) struct AuditedResolutionIndex {
    pub(super) commands: HashMap<[u8; 32], DirectoryReplicaResolutionCommand>,
    pub(super) by_producer: HashMap<[u8; 32], HashSet<[u8; 32]>>,
    pub(super) resolved_incidents: HashMap<[u8; 32], HashSet<[u8; 32]>>,
}
