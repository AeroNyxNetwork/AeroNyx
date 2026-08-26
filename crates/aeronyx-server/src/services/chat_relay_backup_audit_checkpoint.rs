// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_checkpoint.rs
// ============================================
// Version: 1.0.0-AuthenticatedAuditCheckpoint
//
// Creation Reason:
//   [CHAT-RELAY-AUDIT-CHECKPOINT-DOMAIN 2026-08-26 by Codex] Isolate the
//   immutable maintenance checkpoint contract and authentication policy from
//   segment hashing, filesystem publication, and crash recovery.
//
// Main Functionality:
//   - Models cumulative path-free audit-chain checkpoint state.
//   - Builds the existing v1 checkpoint JSON contract.
//   - Authenticates checkpoint fields and HMAC-SHA256 in constant time.
//   - Preserves the existing canonical bincode signing-field order.
//
// Dependencies:
//   - `chat_relay.rs` owns segment files, hashes, locks, atomic publication,
//     archived-byte accounting, and crash-window recovery.
//   - `hmac`, `sha2`, `bincode`, and `hex` preserve the existing v1 format.
//
// Main Logical Flow:
//   1. Receive a service-verified cumulative state and segment fingerprint.
//   2. Encode the exact legacy v1 signing frame.
//   3. Sign or authenticate the checkpoint with a domain-separated HMAC.
//   4. Reject any field drift, invalid digest, count mismatch, or unknown JSON.
//
// Important Note for Next Developer:
//   - Keep v1 JSON fields and signing order byte-for-byte stable.
//   - Never add paths, artifact identities, message metadata, or ciphertext.
//   - Filesystem publication and sequence continuity remain service-owned.
//   - New checkpoint semantics require a versioned migration.
//
// Last Modified:
//   v1.0.0-AuthenticatedAuditCheckpoint - Initial trait-based extraction
// ============================================

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const BACKUP_AUDIT_CHECKPOINT_VERSION: u8 = 1;
const BACKUP_AUDIT_CHECKPOINT_HMAC_DOMAIN: &[u8] =
    b"AeroNyx-RelayCustodyBackup-MaintenanceAuditCheckpoint-v1";
const SHA256_HEX_BYTES: usize = 64;

/// Cumulative path-free state committed by one immutable audit checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupAuditCheckpointState {
    pub(crate) checkpoint_index: u64,
    pub(crate) segment_first_sequence: u64,
    pub(crate) segment_last_sequence: u64,
    pub(crate) segment_bytes: u64,
    pub(crate) segment_sha256: String,
    pub(crate) cumulative_verified_bytes: u64,
    pub(crate) cumulative_last_recorded_at: Option<u64>,
    pub(crate) cumulative_dry_run_count: u64,
    pub(crate) cumulative_planned_count: u64,
    pub(crate) cumulative_completed_count: u64,
    pub(crate) cumulative_failed_count: u64,
    pub(crate) head_mac: String,
    pub(crate) previous_checkpoint_mac: String,
}

/// Authenticated v1 checkpoint stored beside an immutable audit segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatRelayBackupAuditCheckpoint {
    pub(crate) version: u8,
    pub(crate) checkpoint_index: u64,
    pub(crate) segment_first_sequence: u64,
    pub(crate) segment_last_sequence: u64,
    pub(crate) segment_bytes: u64,
    pub(crate) segment_sha256: String,
    pub(crate) cumulative_verified_bytes: u64,
    pub(crate) cumulative_last_recorded_at: Option<u64>,
    pub(crate) cumulative_dry_run_count: u64,
    pub(crate) cumulative_planned_count: u64,
    pub(crate) cumulative_completed_count: u64,
    pub(crate) cumulative_failed_count: u64,
    pub(crate) head_mac: String,
    pub(crate) previous_checkpoint_mac: String,
    pub(crate) checkpoint_mac: String,
}

/// Closed failure vocabulary mapped to stable service errors by the I/O owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditCheckpointError {
    EncodingFailed,
    AuthenticatorInitFailed,
    InvalidCheckpoint,
}

/// Replaceable authentication boundary for immutable maintenance checkpoints.
pub(crate) trait BackupAuditCheckpointAuthenticator {
    fn build(
        &self,
        node_secret: &[u8; 32],
        state: BackupAuditCheckpointState,
    ) -> Result<ChatRelayBackupAuditCheckpoint, BackupAuditCheckpointError>;

    fn authenticate(
        &self,
        node_secret: &[u8; 32],
        checkpoint: &ChatRelayBackupAuditCheckpoint,
        expected: &BackupAuditCheckpointState,
        expected_record_count: u64,
    ) -> Result<(), BackupAuditCheckpointError>;
}

/// HMAC-SHA256 implementation preserving the checkpoint v1 contract.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HmacBackupAuditCheckpointAuthenticator;

impl HmacBackupAuditCheckpointAuthenticator {
    fn signing_bytes(
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> Result<Vec<u8>, BackupAuditCheckpointError> {
        bincode::serialize(&(
            checkpoint.version,
            checkpoint.checkpoint_index,
            checkpoint.segment_first_sequence,
            checkpoint.segment_last_sequence,
            checkpoint.segment_bytes,
            checkpoint.segment_sha256.as_str(),
            checkpoint.cumulative_verified_bytes,
            checkpoint.cumulative_last_recorded_at,
            checkpoint.cumulative_dry_run_count,
            checkpoint.cumulative_planned_count,
            checkpoint.cumulative_completed_count,
            checkpoint.cumulative_failed_count,
            checkpoint.head_mac.as_str(),
            checkpoint.previous_checkpoint_mac.as_str(),
        ))
        .map_err(|_| BackupAuditCheckpointError::EncodingFailed)
    }

    fn mac_engine(
        node_secret: &[u8; 32],
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> Result<HmacSha256, BackupAuditCheckpointError> {
        let mut mac = HmacSha256::new_from_slice(node_secret)
            .map_err(|_| BackupAuditCheckpointError::AuthenticatorInitFailed)?;
        mac.update(BACKUP_AUDIT_CHECKPOINT_HMAC_DOMAIN);
        mac.update(&Self::signing_bytes(checkpoint)?);
        Ok(mac)
    }

    fn mac(
        node_secret: &[u8; 32],
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> Result<String, BackupAuditCheckpointError> {
        Ok(hex::encode(
            Self::mac_engine(node_secret, checkpoint)?
                .finalize()
                .into_bytes(),
        ))
    }

    fn matches_state(
        checkpoint: &ChatRelayBackupAuditCheckpoint,
        expected: &BackupAuditCheckpointState,
        expected_record_count: u64,
    ) -> bool {
        let phase_total = checkpoint
            .cumulative_dry_run_count
            .checked_add(checkpoint.cumulative_planned_count)
            .and_then(|total| total.checked_add(checkpoint.cumulative_completed_count))
            .and_then(|total| total.checked_add(checkpoint.cumulative_failed_count));
        checkpoint.version == BACKUP_AUDIT_CHECKPOINT_VERSION
            && checkpoint.checkpoint_index == expected.checkpoint_index
            && checkpoint.segment_first_sequence == expected.segment_first_sequence
            && checkpoint.segment_last_sequence == expected.segment_last_sequence
            && checkpoint.segment_bytes == expected.segment_bytes
            && checkpoint.segment_sha256 == expected.segment_sha256
            && is_lower_hex(&checkpoint.segment_sha256, SHA256_HEX_BYTES)
            && checkpoint.cumulative_verified_bytes == expected.cumulative_verified_bytes
            && checkpoint.cumulative_last_recorded_at == expected.cumulative_last_recorded_at
            && checkpoint.cumulative_dry_run_count == expected.cumulative_dry_run_count
            && checkpoint.cumulative_planned_count == expected.cumulative_planned_count
            && checkpoint.cumulative_completed_count == expected.cumulative_completed_count
            && checkpoint.cumulative_failed_count == expected.cumulative_failed_count
            && phase_total == Some(expected_record_count)
            && checkpoint.head_mac == expected.head_mac
            && checkpoint.previous_checkpoint_mac == expected.previous_checkpoint_mac
            && is_lower_hex(&checkpoint.head_mac, SHA256_HEX_BYTES)
            && is_lower_hex(&checkpoint.previous_checkpoint_mac, SHA256_HEX_BYTES)
            && is_lower_hex(&checkpoint.checkpoint_mac, SHA256_HEX_BYTES)
    }
}

impl BackupAuditCheckpointAuthenticator for HmacBackupAuditCheckpointAuthenticator {
    fn build(
        &self,
        node_secret: &[u8; 32],
        state: BackupAuditCheckpointState,
    ) -> Result<ChatRelayBackupAuditCheckpoint, BackupAuditCheckpointError> {
        let mut checkpoint = ChatRelayBackupAuditCheckpoint {
            version: BACKUP_AUDIT_CHECKPOINT_VERSION,
            checkpoint_index: state.checkpoint_index,
            segment_first_sequence: state.segment_first_sequence,
            segment_last_sequence: state.segment_last_sequence,
            segment_bytes: state.segment_bytes,
            segment_sha256: state.segment_sha256,
            cumulative_verified_bytes: state.cumulative_verified_bytes,
            cumulative_last_recorded_at: state.cumulative_last_recorded_at,
            cumulative_dry_run_count: state.cumulative_dry_run_count,
            cumulative_planned_count: state.cumulative_planned_count,
            cumulative_completed_count: state.cumulative_completed_count,
            cumulative_failed_count: state.cumulative_failed_count,
            head_mac: state.head_mac,
            previous_checkpoint_mac: state.previous_checkpoint_mac,
            checkpoint_mac: String::new(),
        };
        checkpoint.checkpoint_mac = Self::mac(node_secret, &checkpoint)?;
        Ok(checkpoint)
    }

    fn authenticate(
        &self,
        node_secret: &[u8; 32],
        checkpoint: &ChatRelayBackupAuditCheckpoint,
        expected: &BackupAuditCheckpointState,
        expected_record_count: u64,
    ) -> Result<(), BackupAuditCheckpointError> {
        if !Self::matches_state(checkpoint, expected, expected_record_count) {
            return Err(BackupAuditCheckpointError::InvalidCheckpoint);
        }
        let decoded_mac = hex::decode(&checkpoint.checkpoint_mac)
            .map_err(|_| BackupAuditCheckpointError::InvalidCheckpoint)?;
        Self::mac_engine(node_secret, checkpoint)?
            .verify_slice(&decoded_mac)
            .map_err(|_| BackupAuditCheckpointError::InvalidCheckpoint)
    }
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> BackupAuditCheckpointState {
        BackupAuditCheckpointState {
            checkpoint_index: 1,
            segment_first_sequence: 1,
            segment_last_sequence: 4,
            segment_bytes: 1_024,
            segment_sha256: "a".repeat(SHA256_HEX_BYTES),
            cumulative_verified_bytes: 1_024,
            cumulative_last_recorded_at: Some(1_000),
            cumulative_dry_run_count: 1,
            cumulative_planned_count: 1,
            cumulative_completed_count: 1,
            cumulative_failed_count: 1,
            head_mac: "b".repeat(SHA256_HEX_BYTES),
            previous_checkpoint_mac: "0".repeat(SHA256_HEX_BYTES),
        }
    }

    #[test]
    fn builds_and_authenticates_checkpoint() {
        let authenticator = HmacBackupAuditCheckpointAuthenticator;
        let state = state();
        let checkpoint = authenticator
            .build(&[0x42; 32], state.clone())
            .expect("build checkpoint");

        assert_eq!(checkpoint.version, BACKUP_AUDIT_CHECKPOINT_VERSION);
        assert_eq!(checkpoint.checkpoint_mac.len(), SHA256_HEX_BYTES);
        authenticator
            .authenticate(&[0x42; 32], &checkpoint, &state, 4)
            .expect("authenticate checkpoint");
    }

    #[test]
    fn rejects_tampering_wrong_secret_and_count_drift() {
        let authenticator = HmacBackupAuditCheckpointAuthenticator;
        let state = state();
        let checkpoint = authenticator
            .build(&[0x42; 32], state.clone())
            .expect("build checkpoint");

        let mut tampered = checkpoint.clone();
        tampered.segment_bytes += 1;
        assert_eq!(
            authenticator.authenticate(&[0x42; 32], &tampered, &state, 4),
            Err(BackupAuditCheckpointError::InvalidCheckpoint)
        );
        assert_eq!(
            authenticator.authenticate(&[0x24; 32], &checkpoint, &state, 4),
            Err(BackupAuditCheckpointError::InvalidCheckpoint)
        );
        assert_eq!(
            authenticator.authenticate(&[0x42; 32], &checkpoint, &state, 5),
            Err(BackupAuditCheckpointError::InvalidCheckpoint)
        );
    }

    #[test]
    fn preserves_legacy_v1_canonical_signing_order() {
        let checkpoint = HmacBackupAuditCheckpointAuthenticator
            .build(&[0x42; 32], state())
            .expect("build checkpoint");
        let actual = HmacBackupAuditCheckpointAuthenticator::signing_bytes(&checkpoint)
            .expect("encode current signing frame");
        let expected = bincode::serialize(&(
            checkpoint.version,
            checkpoint.checkpoint_index,
            checkpoint.segment_first_sequence,
            checkpoint.segment_last_sequence,
            checkpoint.segment_bytes,
            checkpoint.segment_sha256.as_str(),
            checkpoint.cumulative_verified_bytes,
            checkpoint.cumulative_last_recorded_at,
            checkpoint.cumulative_dry_run_count,
            checkpoint.cumulative_planned_count,
            checkpoint.cumulative_completed_count,
            checkpoint.cumulative_failed_count,
            checkpoint.head_mac.as_str(),
            checkpoint.previous_checkpoint_mac.as_str(),
        ))
        .expect("encode legacy v1 signing frame");

        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_unknown_json_field() {
        let checkpoint = HmacBackupAuditCheckpointAuthenticator
            .build(&[0x42; 32], state())
            .expect("build checkpoint");
        let mut json = serde_json::to_value(checkpoint).expect("encode checkpoint");
        json.as_object_mut()
            .expect("checkpoint object")
            .insert("artifact_path".to_string(), serde_json::json!("forbidden"));

        assert!(serde_json::from_value::<ChatRelayBackupAuditCheckpoint>(json).is_err());
    }
}
