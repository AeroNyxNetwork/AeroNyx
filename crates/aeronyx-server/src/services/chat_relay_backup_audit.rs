// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit.rs
// ============================================
// Version: 1.0.0-AuthenticatedAuditRecord
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-AUDIT-DOMAIN 2026-08-26 by Codex] Isolate the
//   append-only maintenance audit record contract and authentication policy
//   from filesystem parsing, checkpoint publication, and segment rotation.
//
// Main Functionality:
//   - Models the closed maintenance phase vocabulary as an enum.
//   - Builds the existing v1 JSON record from bounded aggregate counts.
//   - Authenticates record sequence, previous-MAC linkage, and HMAC-SHA256.
//   - Preserves the existing canonical bincode signing-field order.
//
// Dependencies:
//   - `chat_relay.rs` owns private files, locks, bounded JSONL reads, and
//     immutable checkpoint/segment orchestration.
//   - `hmac`, `sha2`, `bincode`, and `hex` preserve the existing v1 format.
//
// Main Logical Flow:
//   1. Convert service-owned aggregate counters into the v1 wire widths.
//   2. Encode the exact legacy signing frame and append its chain predecessor.
//   3. Sign with a node-local, domain-separated HMAC.
//   4. Parse the closed phase enum and authenticate linkage before acceptance.
//
// Important Note for Next Developer:
//   - Keep v1 JSON field names and signing-field order byte-for-byte stable.
//   - Never add paths, artifact names, identities, routes, or payload metadata.
//   - Filesystem and rotation side effects must remain outside this module.
//   - A new action or phase requires a versioned migration, not a new string.
//
// Last Modified:
//   v1.0.0-AuthenticatedAuditRecord - Initial trait-based extraction
// ============================================

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const BACKUP_AUDIT_RECORD_VERSION: u8 = 1;
const BACKUP_AUDIT_ACTION: &str = "prune";
const BACKUP_AUDIT_HMAC_DOMAIN: &[u8] = b"AeroNyx-RelayCustodyBackup-MaintenanceAudit-v1";
const SHA256_HEX_BYTES: usize = 64;

/// Closed phase vocabulary for one backup-maintenance audit record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditPhase {
    DryRun,
    Planned,
    Completed,
    Failed,
}

impl BackupAuditPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DryRun => "dry_run",
            Self::Planned => "planned",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "dry_run" => Some(Self::DryRun),
            "planned" => Some(Self::Planned),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

/// Aggregate path-free counts recorded around one maintenance command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ChatRelayBackupMaintenanceAuditCounts {
    pub(crate) planned_backup_count: usize,
    pub(crate) planned_backup_bytes: u64,
    pub(crate) planned_partial_count: usize,
    pub(crate) planned_partial_bytes: u64,
    pub(crate) completed_backup_count: usize,
    pub(crate) completed_backup_bytes: u64,
    pub(crate) completed_partial_count: usize,
    pub(crate) completed_partial_bytes: u64,
}

/// Authenticated v1 JSONL record stored in the private maintenance log.
///
/// Unknown fields fail closed because they are not covered by the canonical v1
/// signing frame. This contract contains aggregate counts only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatRelayBackupMaintenanceAuditRecord {
    pub(crate) version: u8,
    pub(crate) sequence: u64,
    pub(crate) timestamp: u64,
    pub(crate) action: String,
    pub(crate) phase: String,
    pub(crate) planned_backup_count: u64,
    pub(crate) planned_backup_bytes: u64,
    pub(crate) planned_partial_count: u64,
    pub(crate) planned_partial_bytes: u64,
    pub(crate) completed_backup_count: u64,
    pub(crate) completed_backup_bytes: u64,
    pub(crate) completed_partial_count: u64,
    pub(crate) completed_partial_bytes: u64,
    pub(crate) previous_mac: String,
    pub(crate) mac: String,
}

/// Closed failure vocabulary mapped to stable service errors by the I/O owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditRecordError {
    CountOutOfRange,
    EncodingFailed,
    AuthenticatorInitFailed,
    InvalidRecord,
}

/// Replaceable authentication boundary for maintenance audit records.
pub(crate) trait BackupAuditRecordAuthenticator {
    fn build(
        &self,
        node_secret: &[u8; 32],
        sequence: u64,
        previous_mac: String,
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> Result<ChatRelayBackupMaintenanceAuditRecord, BackupAuditRecordError>;

    fn authenticate(
        &self,
        node_secret: &[u8; 32],
        record: &ChatRelayBackupMaintenanceAuditRecord,
        expected_sequence: u64,
        expected_previous_mac: &str,
    ) -> Result<BackupAuditPhase, BackupAuditRecordError>;
}

/// HMAC-SHA256 implementation preserving the maintenance audit v1 contract.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HmacBackupAuditRecordAuthenticator;

impl HmacBackupAuditRecordAuthenticator {
    fn signing_bytes(
        record: &ChatRelayBackupMaintenanceAuditRecord,
    ) -> Result<Vec<u8>, BackupAuditRecordError> {
        bincode::serialize(&(
            record.version,
            record.sequence,
            record.timestamp,
            record.action.as_str(),
            record.phase.as_str(),
            record.planned_backup_count,
            record.planned_backup_bytes,
            record.planned_partial_count,
            record.planned_partial_bytes,
            record.completed_backup_count,
            record.completed_backup_bytes,
            record.completed_partial_count,
            record.completed_partial_bytes,
            record.previous_mac.as_str(),
        ))
        .map_err(|_| BackupAuditRecordError::EncodingFailed)
    }

    fn mac_engine(
        node_secret: &[u8; 32],
        record: &ChatRelayBackupMaintenanceAuditRecord,
    ) -> Result<HmacSha256, BackupAuditRecordError> {
        let mut mac = HmacSha256::new_from_slice(node_secret)
            .map_err(|_| BackupAuditRecordError::AuthenticatorInitFailed)?;
        mac.update(BACKUP_AUDIT_HMAC_DOMAIN);
        mac.update(&Self::signing_bytes(record)?);
        Ok(mac)
    }

    fn mac(
        node_secret: &[u8; 32],
        record: &ChatRelayBackupMaintenanceAuditRecord,
    ) -> Result<String, BackupAuditRecordError> {
        Ok(hex::encode(
            Self::mac_engine(node_secret, record)?
                .finalize()
                .into_bytes(),
        ))
    }
}

impl BackupAuditRecordAuthenticator for HmacBackupAuditRecordAuthenticator {
    fn build(
        &self,
        node_secret: &[u8; 32],
        sequence: u64,
        previous_mac: String,
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> Result<ChatRelayBackupMaintenanceAuditRecord, BackupAuditRecordError> {
        let count = |value: usize| {
            u64::try_from(value).map_err(|_| BackupAuditRecordError::CountOutOfRange)
        };
        let mut record = ChatRelayBackupMaintenanceAuditRecord {
            version: BACKUP_AUDIT_RECORD_VERSION,
            sequence,
            timestamp,
            action: BACKUP_AUDIT_ACTION.to_string(),
            phase: phase.as_str().to_string(),
            planned_backup_count: count(counts.planned_backup_count)?,
            planned_backup_bytes: counts.planned_backup_bytes,
            planned_partial_count: count(counts.planned_partial_count)?,
            planned_partial_bytes: counts.planned_partial_bytes,
            completed_backup_count: count(counts.completed_backup_count)?,
            completed_backup_bytes: counts.completed_backup_bytes,
            completed_partial_count: count(counts.completed_partial_count)?,
            completed_partial_bytes: counts.completed_partial_bytes,
            previous_mac,
            mac: String::new(),
        };
        record.mac = Self::mac(node_secret, &record)?;
        Ok(record)
    }

    fn authenticate(
        &self,
        node_secret: &[u8; 32],
        record: &ChatRelayBackupMaintenanceAuditRecord,
        expected_sequence: u64,
        expected_previous_mac: &str,
    ) -> Result<BackupAuditPhase, BackupAuditRecordError> {
        let phase =
            BackupAuditPhase::parse(&record.phase).ok_or(BackupAuditRecordError::InvalidRecord)?;
        if record.version != BACKUP_AUDIT_RECORD_VERSION
            || record.sequence != expected_sequence
            || record.previous_mac != expected_previous_mac
            || !is_lower_hex(&record.previous_mac, SHA256_HEX_BYTES)
            || !is_lower_hex(&record.mac, SHA256_HEX_BYTES)
            || record.action != BACKUP_AUDIT_ACTION
        {
            return Err(BackupAuditRecordError::InvalidRecord);
        }
        let decoded_mac =
            hex::decode(&record.mac).map_err(|_| BackupAuditRecordError::InvalidRecord)?;
        Self::mac_engine(node_secret, record)?
            .verify_slice(&decoded_mac)
            .map_err(|_| BackupAuditRecordError::InvalidRecord)?;
        Ok(phase)
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

    fn counts() -> ChatRelayBackupMaintenanceAuditCounts {
        ChatRelayBackupMaintenanceAuditCounts {
            planned_backup_count: 2,
            planned_backup_bytes: 4_096,
            planned_partial_count: 1,
            planned_partial_bytes: 128,
            completed_backup_count: 1,
            completed_backup_bytes: 2_048,
            completed_partial_count: 1,
            completed_partial_bytes: 128,
        }
    }

    #[test]
    fn builds_and_authenticates_linked_record() {
        let authenticator = HmacBackupAuditRecordAuthenticator;
        let previous_mac = "0".repeat(SHA256_HEX_BYTES);
        let record = authenticator
            .build(
                &[0x42; 32],
                1,
                previous_mac.clone(),
                BackupAuditPhase::Completed,
                1_000,
                counts(),
            )
            .expect("build audit record");

        assert_eq!(record.action, "prune");
        assert_eq!(record.phase, "completed");
        assert_eq!(record.mac.len(), SHA256_HEX_BYTES);
        assert_eq!(
            authenticator
                .authenticate(&[0x42; 32], &record, 1, &previous_mac)
                .expect("authenticate linked record"),
            BackupAuditPhase::Completed
        );
    }

    #[test]
    fn rejects_tampering_wrong_secret_and_broken_link() {
        let authenticator = HmacBackupAuditRecordAuthenticator;
        let previous_mac = "0".repeat(SHA256_HEX_BYTES);
        let record = authenticator
            .build(
                &[0x42; 32],
                1,
                previous_mac.clone(),
                BackupAuditPhase::Planned,
                1_000,
                counts(),
            )
            .expect("build audit record");

        let mut tampered = record.clone();
        tampered.completed_backup_bytes += 1;
        assert_eq!(
            authenticator.authenticate(&[0x42; 32], &tampered, 1, &previous_mac),
            Err(BackupAuditRecordError::InvalidRecord)
        );
        assert_eq!(
            authenticator.authenticate(&[0x24; 32], &record, 1, &previous_mac),
            Err(BackupAuditRecordError::InvalidRecord)
        );
        assert_eq!(
            authenticator.authenticate(&[0x42; 32], &record, 2, &record.mac),
            Err(BackupAuditRecordError::InvalidRecord)
        );
    }

    #[test]
    fn preserves_legacy_v1_canonical_signing_order() {
        let authenticator = HmacBackupAuditRecordAuthenticator;
        let record = authenticator
            .build(
                &[0x42; 32],
                7,
                "0".repeat(SHA256_HEX_BYTES),
                BackupAuditPhase::Failed,
                1_000,
                counts(),
            )
            .expect("build audit record");
        let actual = HmacBackupAuditRecordAuthenticator::signing_bytes(&record)
            .expect("encode current signing frame");
        let expected = bincode::serialize(&(
            record.version,
            record.sequence,
            record.timestamp,
            record.action.as_str(),
            record.phase.as_str(),
            record.planned_backup_count,
            record.planned_backup_bytes,
            record.planned_partial_count,
            record.planned_partial_bytes,
            record.completed_backup_count,
            record.completed_backup_bytes,
            record.completed_partial_count,
            record.completed_partial_bytes,
            record.previous_mac.as_str(),
        ))
        .expect("encode legacy v1 signing frame");

        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_unknown_phase_and_unknown_json_field() {
        let authenticator = HmacBackupAuditRecordAuthenticator;
        let mut record = authenticator
            .build(
                &[0x42; 32],
                1,
                "0".repeat(SHA256_HEX_BYTES),
                BackupAuditPhase::DryRun,
                1_000,
                counts(),
            )
            .expect("build audit record");
        record.phase = "unknown".to_string();
        assert_eq!(
            authenticator.authenticate(&[0x42; 32], &record, 1, &record.previous_mac),
            Err(BackupAuditRecordError::InvalidRecord)
        );

        let mut json = serde_json::to_value(&record).expect("encode audit record");
        json.as_object_mut()
            .expect("audit record object")
            .insert("private_path".to_string(), serde_json::json!("forbidden"));
        assert!(serde_json::from_value::<ChatRelayBackupMaintenanceAuditRecord>(json).is_err());
    }
}
