// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_anchor.rs
// ============================================
// Version: 1.0.0-BackupAuditAnchorDigest
//
// Creation Reason:
//   [CHAT-BACKUP-AUDIT-ANCHOR-DOMAIN 2026-08-28 by Codex] Extract conversion
//   of a private authenticated checkpoint into a path-free public digest from
//   the oversized relay orchestration service.
//
// Main Functionality:
//   - Requires a non-empty immutable checkpoint history.
//   - Decodes the exact private checkpoint authenticator shape.
//   - Derives a domain-separated SHA-256 public anchor digest.
//   - Exposes a closed storage-independent failure vocabulary.
//
// Dependencies:
//   - `chat_relay_backup_audit_verification.rs` owns verified chain state.
//   - `sha2` hashes the aggregate checkpoint frame into the public domain.
//   - The relay service maps closed failures to stable path-free API errors.
//
// Main Logical Flow:
//   1. Read the aggregate receipt from already authenticated chain state.
//   2. Reject state with no immutable checkpoint coverage.
//   3. Decode the fixed 32-byte private checkpoint authenticator.
//   4. Hash domain, aggregate coverage, and authenticator into one digest.
//
// Important Note for Next Developer:
//   - Never return, serialize, or log the private checkpoint authenticator.
//   - Keep the v1 domain and byte order stable for signed anchor compatibility.
//   - This boundary accepts authenticated state only; it does not verify files.
//   - New digest semantics require a new explicit domain/version.
//
// Last Modified:
//   v1.0.0-BackupAuditAnchorDigest - Initial pure domain extraction
// ============================================

use sha2::{Digest, Sha256};

use super::chat_relay_backup_audit_verification::ChatRelayBackupAuditVerificationState;

const BACKUP_AUDIT_ANCHOR_DIGEST_DOMAIN: &[u8] =
    b"AeroNyx-RelayCustodyBackup-MaintenanceAuditAnchorDigest-v1";

/// Closed failures while deriving one public checkpoint anchor digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditAnchorDigestError {
    MissingImmutableCheckpoint,
    InvalidCheckpointAuthenticator,
}

/// Derives a public opaque digest from authenticated immutable checkpoint state.
pub(crate) fn derive_backup_audit_anchor_digest(
    state: &ChatRelayBackupAuditVerificationState,
) -> Result<[u8; 32], BackupAuditAnchorDigestError> {
    let receipt = state.receipt();
    if receipt.checkpoint_count == 0
        || receipt.archived_record_count == 0
        || receipt.archived_bytes == 0
    {
        return Err(BackupAuditAnchorDigestError::MissingImmutableCheckpoint);
    }
    let checkpoint_mac = hex::decode(state.checkpoint_head_mac())
        .map_err(|_| BackupAuditAnchorDigestError::InvalidCheckpointAuthenticator)?;
    let checkpoint_mac: [u8; 32] = checkpoint_mac
        .try_into()
        .map_err(|_| BackupAuditAnchorDigestError::InvalidCheckpointAuthenticator)?;

    // [CHAT-BACKUP-AUDIT-ANCHOR-DOMAIN 2026-08-28 by Codex] A separate
    // one-way domain prevents a public anchor from becoming a reusable private
    // HMAC capability or exposing the private signing frame.
    let mut hasher = Sha256::new();
    hasher.update(BACKUP_AUDIT_ANCHOR_DIGEST_DOMAIN);
    hasher.update(receipt.checkpoint_count.to_le_bytes());
    hasher.update(receipt.archived_record_count.to_le_bytes());
    hasher.update(receipt.archived_bytes.to_le_bytes());
    hasher.update(checkpoint_mac);
    Ok(hasher.finalize().into())
}
