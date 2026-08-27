// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_namespace.rs
// ============================================
// Version: 1.0.0-BackupNamespaceDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-NAMESPACE-DOMAIN 2026-08-27 by Codex] Isolate the
//   private recovery-image naming grammar and operation-key derivation from
//   filesystem orchestration.
//
// Main Functionality:
//   - Creates canonical unique and idempotent recovery-image names.
//   - Derives operation-scoped names without persisting raw operation IDs.
//   - Creates canonical interrupted-publication temporary names.
//   - Classifies only the closed set of managed recovery namespace entries.
//
// Dependencies:
//   - `hmac` and `sha2` derive opaque operation-scoped artifact keys.
//   - `chat_relay.rs` owns paths, permissions, metadata, SQLite, and deletion.
//
// Main Logical Flow:
//   1. Validate a management operation ID before any storage access.
//   2. Derive a domain-separated node-local HMAC and truncate it to 128 bits.
//   3. Render one canonical private artifact name.
//   4. Classify directory entries through the same exact naming grammar.
//
// Important Note for Next Developer:
//   - Never persist or return a raw operation ID.
//   - Preserve the current names for restart and rolling-version compatibility.
//   - Unknown, malformed, uppercase, or partially matching names are unmanaged.
//   - Paths and filesystem side effects must remain outside this pure module.
//
// Last Modified:
//   v1.0.0-BackupNamespaceDomain - Initial trait-based namespace policy
// ============================================

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const UNIQUE_ARTIFACT_PREFIX: &str = "relay-custody-";
const OPERATION_ARTIFACT_PREFIX: &str = "relay-custody-operation-";
const RECOVERY_IMAGE_SUFFIX: &str = ".sqlite";
const TEMPORARY_ARTIFACT_PREFIX: &str = ".relay-custody-";
const TEMPORARY_ARTIFACT_SUFFIX: &str = ".tmp";
const SQLITE_SIDECAR_SUFFIXES: [&str; 3] = ["-journal", "-wal", "-shm"];
const OPERATION_KEY_HEX_BYTES: usize = 16;
const UNIQUE_NONCE_HEX_LEN: usize = 16;
const OPERATION_KEY_HEX_LEN: usize = OPERATION_KEY_HEX_BYTES * 2;
const BACKUP_OPERATION_HMAC_DOMAIN: &[u8] = b"AeroNyx-RelayCustodyBackup-Operation-v1";

/// Valid canonical name produced by the private backup namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupArtifactName(String);

impl BackupArtifactName {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Closed classification for one entry in the private backup directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupArtifactKind {
    RecoveryImage,
    InterruptedTemporary,
    Unmanaged,
}

/// Closed failure vocabulary for canonical name derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupNamespaceError {
    EmptyOperationId,
    OperationIdTooLarge,
    OperationIdLengthOverflow,
    SecretRejected,
}

/// Replaceable pure capability for the private recovery artifact namespace.
pub(crate) trait BackupArtifactNamespace {
    fn unique_recovery_image_name(&self, created_at: u64, nonce: u64) -> BackupArtifactName;

    fn idempotent_recovery_image_name(
        &self,
        node_secret: &[u8; 32],
        operation_id: &str,
    ) -> Result<BackupArtifactName, BackupNamespaceError>;

    fn temporary_recovery_image_name(&self, created_at: u64, nonce: u64) -> BackupArtifactName;

    fn classify(&self, name: &str) -> BackupArtifactKind;
}

/// Production namespace preserving the v1 HMAC and filename contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HmacBackupArtifactNamespace {
    max_operation_id_bytes: usize,
}

impl HmacBackupArtifactNamespace {
    pub(crate) const fn new(max_operation_id_bytes: usize) -> Self {
        Self {
            max_operation_id_bytes,
        }
    }

    fn is_lower_hex(value: &str, expected_len: usize) -> bool {
        value.len() == expected_len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn is_decimal(value: &str) -> bool {
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
    }

    fn is_recovery_image_name(name: &str) -> bool {
        if let Some(operation_key) = name
            .strip_prefix(OPERATION_ARTIFACT_PREFIX)
            .and_then(|value| value.strip_suffix(RECOVERY_IMAGE_SUFFIX))
        {
            return Self::is_lower_hex(operation_key, OPERATION_KEY_HEX_LEN);
        }

        let Some(stem) = name
            .strip_prefix(UNIQUE_ARTIFACT_PREFIX)
            .and_then(|value| value.strip_suffix(RECOVERY_IMAGE_SUFFIX))
        else {
            return false;
        };
        let Some((created_at, nonce)) = stem.rsplit_once('-') else {
            return false;
        };
        Self::is_decimal(created_at) && Self::is_lower_hex(nonce, UNIQUE_NONCE_HEX_LEN)
    }

    fn is_temporary_name(name: &str) -> bool {
        let base = SQLITE_SIDECAR_SUFFIXES
            .into_iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .unwrap_or(name);
        let Some(stem) = base
            .strip_prefix(TEMPORARY_ARTIFACT_PREFIX)
            .and_then(|value| value.strip_suffix(TEMPORARY_ARTIFACT_SUFFIX))
        else {
            return false;
        };
        let Some((created_at, nonce)) = stem.rsplit_once('-') else {
            return false;
        };
        Self::is_decimal(created_at) && Self::is_lower_hex(nonce, UNIQUE_NONCE_HEX_LEN)
    }
}

impl BackupArtifactNamespace for HmacBackupArtifactNamespace {
    fn unique_recovery_image_name(&self, created_at: u64, nonce: u64) -> BackupArtifactName {
        BackupArtifactName(format!(
            "{UNIQUE_ARTIFACT_PREFIX}{created_at}-{nonce:016x}{RECOVERY_IMAGE_SUFFIX}"
        ))
    }

    fn idempotent_recovery_image_name(
        &self,
        node_secret: &[u8; 32],
        operation_id: &str,
    ) -> Result<BackupArtifactName, BackupNamespaceError> {
        if operation_id.is_empty() {
            return Err(BackupNamespaceError::EmptyOperationId);
        }
        if operation_id.len() > self.max_operation_id_bytes {
            return Err(BackupNamespaceError::OperationIdTooLarge);
        }
        let operation_id_len = u64::try_from(operation_id.len())
            .map_err(|_| BackupNamespaceError::OperationIdLengthOverflow)?;
        let mut mac = HmacSha256::new_from_slice(node_secret)
            .map_err(|_| BackupNamespaceError::SecretRejected)?;
        mac.update(BACKUP_OPERATION_HMAC_DOMAIN);
        mac.update(&operation_id_len.to_be_bytes());
        mac.update(operation_id.as_bytes());
        let digest = mac.finalize().into_bytes();
        let operation_key = hex::encode(&digest[..OPERATION_KEY_HEX_BYTES]);
        Ok(BackupArtifactName(format!(
            "{OPERATION_ARTIFACT_PREFIX}{operation_key}{RECOVERY_IMAGE_SUFFIX}"
        )))
    }

    fn temporary_recovery_image_name(&self, created_at: u64, nonce: u64) -> BackupArtifactName {
        BackupArtifactName(format!(
            "{TEMPORARY_ARTIFACT_PREFIX}{created_at}-{nonce:016x}{TEMPORARY_ARTIFACT_SUFFIX}"
        ))
    }

    fn classify(&self, name: &str) -> BackupArtifactKind {
        if Self::is_recovery_image_name(name) {
            BackupArtifactKind::RecoveryImage
        } else if Self::is_temporary_name(name) {
            BackupArtifactKind::InterruptedTemporary
        } else {
            BackupArtifactKind::Unmanaged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace() -> HmacBackupArtifactNamespace {
        HmacBackupArtifactNamespace::new(128)
    }

    #[test]
    fn canonical_names_preserve_existing_namespace_contract() {
        let namespace = namespace();
        let unique = namespace.unique_recovery_image_name(1_800_000_000, 0x0123_4567_89ab_cdef);
        let temporary =
            namespace.temporary_recovery_image_name(1_800_000_000, 0x0123_4567_89ab_cdef);
        let operation = namespace
            .idempotent_recovery_image_name(&[0x11; 32], "backup-op-42")
            .expect("derive operation artifact");

        assert_eq!(
            unique.as_str(),
            "relay-custody-1800000000-0123456789abcdef.sqlite"
        );
        assert_eq!(
            temporary.as_str(),
            ".relay-custody-1800000000-0123456789abcdef.tmp"
        );
        assert_eq!(
            operation.as_str(),
            "relay-custody-operation-9036a4d3fdf33161a1f37da01c75541d.sqlite"
        );
    }

    #[test]
    fn operation_names_are_deterministic_secret_bound_and_opaque() {
        let namespace = namespace();
        let first = namespace
            .idempotent_recovery_image_name(&[0x21; 32], "private-operation-id")
            .expect("derive first name");
        let replay = namespace
            .idempotent_recovery_image_name(&[0x21; 32], "private-operation-id")
            .expect("derive replay name");
        let other_node = namespace
            .idempotent_recovery_image_name(&[0x22; 32], "private-operation-id")
            .expect("derive other-node name");

        assert_eq!(first, replay);
        assert_ne!(first, other_node);
        assert!(!first.as_str().contains("private-operation-id"));
        assert_eq!(
            namespace.classify(first.as_str()),
            BackupArtifactKind::RecoveryImage
        );
    }

    #[test]
    fn invalid_operation_ids_fail_before_name_creation() {
        let namespace = HmacBackupArtifactNamespace::new(4);
        assert_eq!(
            namespace.idempotent_recovery_image_name(&[0x31; 32], ""),
            Err(BackupNamespaceError::EmptyOperationId)
        );
        assert_eq!(
            namespace.idempotent_recovery_image_name(&[0x31; 32], "12345"),
            Err(BackupNamespaceError::OperationIdTooLarge)
        );
    }

    #[test]
    fn classification_is_closed_and_accepts_only_exact_managed_grammar() {
        let namespace = namespace();
        let temporary = ".relay-custody-1800000000-0123456789abcdef.tmp";
        for suffix in ["", "-journal", "-wal", "-shm"] {
            assert_eq!(
                namespace.classify(&format!("{temporary}{suffix}")),
                BackupArtifactKind::InterruptedTemporary
            );
        }
        for unmanaged in [
            "relay-custody-1800000000-0123456789abcdeF.sqlite",
            "relay-custody-1800000000-0123456789abcde.sqlite",
            "relay-custody-operation-0123456789abcdef.sqlite",
            ".relay-custody--0123456789abcdef.tmp",
            ".relay-custody-1800000000-0123456789abcdef.tmp-lock",
            "unrelated.sqlite",
        ] {
            assert_eq!(
                namespace.classify(unmanaged),
                BackupArtifactKind::Unmanaged,
                "unexpected managed name: {unmanaged}"
            );
        }
    }
}
