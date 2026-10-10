// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/evidence_types.rs
// ============================================
//! # Producer, incident, tip and evidence values
//!
//! Owns per-producer snapshots, incident summaries/pages/evidence, accepted tips,
//! exported evidence pages, and outbound gossip announcement values.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    DirectoryCommitmentBlockV1, DirectoryDescriptorInclusionProofV1, SignedNodeDescriptor,
};

/// Persisted aggregate state for one producer namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaProducerSnapshot {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Timestamp signed into the accepted tip block.
    pub tip_timestamp: u64,
    /// Whether imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable authenticated incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Last time this namespace metadata changed locally.
    pub updated_at: u64,
    /// Verified blocks retained for this producer.
    pub blocks: u64,
    /// Verified commitments retained for this producer.
    pub commitments: u64,
    /// Durable incidents attributed to this producer response stream.
    pub incidents: u64,
    /// Signed operator resolutions retained for this producer.
    pub resolutions: u64,
}

/// Bounded metadata for one startup-audited Directory Replica incident.
///
/// The summary intentionally excludes the potentially large signed response
/// frame. Call [`DirectoryReplicaStore::incident_evidence`] for an independent,
/// fail-closed verification immediately before exporting that frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentSummary {
    /// Content-addressed incident identifier used as the pagination cursor.
    pub incident_digest: [u8; 32],
    /// Producer that signed the conflicting Directory Sync response.
    pub producer: [u8; 32],
    /// Identity whose chain or descriptor assertion conflicts.
    pub subject_node_id: [u8; 32],
    /// Stable internal incident classification.
    pub kind: String,
    /// Conflicting block or advertised tip height.
    pub height: u64,
    /// Previously accepted local claim.
    pub local_hash: [u8; 32],
    /// Conflicting producer-signed remote claim.
    pub remote_hash: [u8; 32],
    /// Local Unix timestamp at which the signed evidence was persisted.
    pub observed_at: u64,
    /// Whether this producer remains quarantined at read time.
    pub producer_quarantined: bool,
}

/// Deterministic cursor page of incident metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentPage {
    /// Incident summaries ordered by ascending content digest.
    pub incidents: Vec<DirectoryReplicaIncidentSummary>,
    /// Last returned digest when another page exists.
    pub next_cursor: Option<[u8; 32]>,
}

/// Complete independently verifiable evidence for one durable incident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentEvidence {
    /// Validated incident metadata and current quarantine state.
    pub summary: DirectoryReplicaIncidentSummary,
    /// Exact canonical producer-signed `BlockRangeResponseV1` bytes.
    pub evidence_frame: Vec<u8>,
    /// SHA-256 digest of `evidence_frame` for transport/file verification.
    pub evidence_sha256: [u8; 32],
}

/// Current accepted prefix and isolation state for one producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaTip {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Accepted tip hash, or zero for an empty prefix.
    pub tip_hash: [u8; 32],
    /// Accepted tip timestamp, or zero for an empty prefix.
    pub tip_timestamp: u64,
    /// Whether further imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Exact unresolved incident when quarantined.
    pub active_incident_digest: Option<[u8; 32]>,
    /// Latest signed resolution in this producer's linked audit history.
    pub last_resolution_digest: Option<[u8; 32]>,
}

/// One bounded page exported from a fully audited producer replica.
///
/// The carrier signs transport metadata separately. Every block in this page
/// remains signed by the original producer and is re-verified by the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaEvidencePage {
    /// Contiguous producer-signed blocks in ascending height order.
    pub blocks: Vec<DirectoryCommitmentBlockV1>,
    /// Audited accepted producer tip height at export time.
    pub tip_height: u64,
    /// Audited accepted producer tip hash at export time.
    pub tip_hash: [u8; 32],
}

/// One producer-authenticated descriptor proof ready for outbound gossip.
///
/// This value contains only public node-directory evidence. It deliberately
/// excludes carrier identity, selected routes, endpoints outside the signed
/// descriptor, user data, messages, ciphertext, and traffic observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaGossipAnnouncement {
    /// Original Directory block producer.
    pub(crate) producer: [u8; 32],
    /// Exact producer-signed block selected from the audited local replica.
    pub(crate) block_hash: [u8; 32],
    /// Exact authenticated descriptor object hash.
    pub(crate) descriptor_hash: [u8; 32],
    /// Compact producer-signed inclusion proof and descriptor object.
    pub(crate) proof: DirectoryDescriptorInclusionProofV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DirectoryReplicaGossipCandidate {
    pub(super) producer: [u8; 32],
    pub(super) block_hash: [u8; 32],
    pub(super) descriptor_hash: [u8; 32],
    pub(super) descriptor: SignedNodeDescriptor,
}

impl DirectoryReplicaTip {
    pub(super) const fn empty(producer: [u8; 32]) -> Self {
        Self {
            producer,
            tip_height: 0,
            tip_hash: [0u8; 32],
            tip_timestamp: 0,
            quarantined: false,
            quarantine_kind: None,
            active_incident_digest: None,
            last_resolution_digest: None,
        }
    }
}
