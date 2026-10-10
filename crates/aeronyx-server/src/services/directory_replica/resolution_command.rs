// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/resolution_command.rs
// ============================================
//! # Quarantine resolution command
//!
//! Owns the node-identity-signed `resume_existing_prefix` command and its
//! durable resolution report.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    validate_incident_kind, Digest, DirectoryReplicaStoreError, IdentityKeyPair, Sha256,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, DIRECTORY_REPLICA_RESOLUTION_ACTION,
};

/// Node-identity-signed command that resumes one exact quarantined prefix.
///
/// The command cannot select a fork, delete evidence, or rewind a chain. Its
/// compare-and-swap fields bind one immutable incident to the exact prefix and
/// previous resolution history inspected by the host-local operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionCommand {
    /// Random operator command identifier; unique across this replica store.
    pub command_id: [u8; 16],
    /// Immutable incident explicitly approved by the operator.
    pub incident_digest: [u8; 32],
    /// Producer namespace whose existing prefix may resume synchronization.
    pub producer: [u8; 32],
    /// Accepted prefix height observed before signing.
    pub expected_tip_height: u64,
    /// Accepted prefix hash observed before signing.
    pub expected_tip_hash: [u8; 32],
    /// Quarantine classification observed before signing.
    pub expected_quarantine_kind: String,
    /// Previous linked resolution, or `None` for this producer's first one.
    pub previous_resolution_digest: Option<[u8; 32]>,
    /// Host timestamp at which the operator approved the command.
    pub resolved_at: u64,
    /// Local node identity that must match replica metadata.
    pub resolver_node_id: [u8; 32],
    /// Ed25519 signature over every command field and the fixed action.
    pub signature: [u8; 64],
}

impl DirectoryReplicaResolutionCommand {
    /// Constructs and signs one exact `resume_existing_prefix` command.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when any bounded command field is
    /// invalid. Signing never reads or modifies the replica database.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        identity: &IdentityKeyPair,
        command_id: [u8; 16],
        incident_digest: [u8; 32],
        producer: [u8; 32],
        expected_tip_height: u64,
        expected_tip_hash: [u8; 32],
        expected_quarantine_kind: String,
        previous_resolution_digest: Option<[u8; 32]>,
        resolved_at: u64,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut command = Self {
            command_id,
            incident_digest,
            producer,
            expected_tip_height,
            expected_tip_hash,
            expected_quarantine_kind,
            previous_resolution_digest,
            resolved_at,
            resolver_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        command.validate_unsigned_fields()?;
        command.signature = identity.sign(&command.signing_bytes());
        Ok(command)
    }

    pub(super) fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.command_id == [0u8; 16]
            || self.incident_digest == [0u8; 32]
            || self.producer == [0u8; 32]
            || self.resolver_node_id == [0u8; 32]
            || self.resolved_at == 0
            || (self.expected_tip_height == 0 && self.expected_tip_hash != [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution command contains an invalid sentinel".to_string(),
            ));
        }
        validate_incident_kind(&self.expected_quarantine_kind)
    }

    pub(super) fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(320);
        bytes.extend_from_slice(b"AeroNyx-DirectoryReplicaResolution-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.command_id);
        bytes.extend_from_slice(&self.incident_digest);
        bytes.extend_from_slice(&self.producer);
        bytes.extend_from_slice(&self.expected_tip_height.to_le_bytes());
        bytes.extend_from_slice(&self.expected_tip_hash);
        bytes.extend_from_slice(&(self.expected_quarantine_kind.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.expected_quarantine_kind.as_bytes());
        match self.previous_resolution_digest {
            Some(digest) => {
                bytes.push(1);
                bytes.extend_from_slice(&digest);
            }
            None => bytes.push(0),
        }
        bytes.extend_from_slice(&self.resolved_at.to_le_bytes());
        bytes.extend_from_slice(&self.resolver_node_id);
        bytes.extend_from_slice(&(DIRECTORY_REPLICA_RESOLUTION_ACTION.len() as u64).to_le_bytes());
        bytes.extend_from_slice(DIRECTORY_REPLICA_RESOLUTION_ACTION.as_bytes());
        bytes
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Durable result of one successful compare-and-swap resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionReport {
    /// Content address of the signed resolution audit record.
    pub resolution_digest: [u8; 32],
    /// Unique command identifier supplied by the operator CLI.
    pub command_id: [u8; 16],
    /// Producer namespace that resumed its already accepted prefix.
    pub producer: [u8; 32],
    /// Prefix height retained without rewind or fork selection.
    pub retained_tip_height: u64,
    /// Prefix hash retained without modification.
    pub retained_tip_hash: [u8; 32],
    /// Signed operator approval timestamp.
    pub resolved_at: u64,
}
