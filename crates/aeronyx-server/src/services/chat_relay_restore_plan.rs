// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_restore_plan.rs
// ============================================
// Version: 1.1.0-ComposedRestoreCommandIntegration
//
// Creation Reason:
//   [CHAT-RELAY-RESTORE-PLAN-DOMAIN 2026-08-26 by Codex] Isolate the
//   short-lived restore-plan contract and authentication policy from the
//   filesystem-owning relay service.
//
// Modification Reason:
//   [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Document the
//   composed command as the owner of private snapshot orchestration while this
//   module remains a side-effect-free authentication policy.
//
// Main Functionality:
//   - Models the public, path-free restore-plan receipt.
//   - Binds aggregate recovery state to private artifact identities by HMAC.
//   - Enforces the fixed version, lifetime, nonce, and commitment contract.
//   - Detects public or private state drift without performing filesystem I/O.
//
// Dependencies:
//   - `chat_relay_restore_command` supplies verified private boundary snapshots.
//   - `chat_relay.rs` retains public compatibility and maintenance-lock scope.
//   - `hmac`, `sha2`, `bincode`, and `hex` preserve the existing v1 wire and
//     commitment format.
//
// Main Logical Flow:
//   1. Build a path-free public receipt from verified aggregate state.
//   2. Canonically encode public fields plus private artifact identities.
//   3. Sign the canonical frame with a domain-separated node-local HMAC.
//   4. Revalidate lifetime/state and verify the commitment in constant time.
//
// Important Note for Next Developer:
//   - Keep the v1 signing-field order byte-for-byte compatible.
//   - Private boundary fields must never be serialized into the public receipt.
//   - This module must remain side-effect free; command I/O executes only
//     inside the maintenance-lock scope retained by `ChatRelayService`.
//   - New wire fields require a new version and explicit migration handling.
//
// Last Modified:
//   v1.1.0-ComposedRestoreCommandIntegration - Documented command ownership
//   v1.0.0-AuthenticatedRestorePlan - Initial trait-based extraction
// ============================================

use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub(crate) const RESTORE_PLAN_VALIDITY_SECS: u64 = 10 * 60;
pub(crate) const RESTORE_PLAN_VERSION: u8 = 1;
pub(crate) const RESTORE_PLAN_NONCE_BYTES: usize = 16;
const RESTORE_PLAN_HMAC_DOMAIN: &[u8] = b"AeroNyx-RelayCustodyRestorePlan-v1";

/// Short-lived, path-free commitment to one verified recovery plan.
///
/// The HMAC binds public aggregate state to private backup and active-file
/// identities without disclosing those identities. This is a stale-state guard,
/// not restore authorization: execution still requires a stopped node, explicit
/// confirmation, a rollback image, and post-restore verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRelayRestorePlanReceipt {
    /// Restore-plan wire contract version.
    pub version: u8,
    /// Host wall-clock issue time in Unix seconds.
    pub issued_at: u64,
    /// Exclusive expiry time in Unix seconds.
    pub expires_at: u64,
    /// Number of verified recovery images observed when planning.
    pub verified_backup_count: u64,
    /// Size of the selected newest recovery image.
    pub selected_backup_bytes: u64,
    /// Whether the configured active main database existed at issuance.
    pub active_database_present: bool,
    /// Size of the active main database at issuance, or zero when absent.
    pub active_database_bytes: u64,
    /// Per-plan random lowercase hexadecimal nonce.
    pub nonce: String,
    /// Lowercase HMAC-SHA256 commitment over public and private plan state.
    pub commitment: String,
}

/// Aggregate recovery state permitted to cross the service boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestorePlanAggregate {
    pub(crate) verified_backup_count: u64,
    pub(crate) selected_backup_bytes: u64,
    pub(crate) active_database_present: bool,
    pub(crate) active_database_bytes: u64,
}

/// Private artifact identity snapshot committed into, but never exposed by,
/// the public restore plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RestorePlanPrivateBoundary<'a> {
    pub(crate) configured_database_path: &'a str,
    pub(crate) selected_backup_name: &'a str,
    pub(crate) selected_backup_modified_at: SystemTime,
    pub(crate) active_database_modified_at: Option<SystemTime>,
    pub(crate) selected_backup_device_id: u64,
    pub(crate) selected_backup_inode: u64,
    pub(crate) active_database_device_id: u64,
    pub(crate) active_database_inode: u64,
}

/// Closed failure vocabulary mapped to stable service errors by the I/O owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestorePlanError {
    ExpiryOutOfRange,
    FilesystemTimeOutOfRange,
    EncodingFailed,
    AuthenticatorInitFailed,
    InvalidOrStale,
}

/// Replaceable authentication boundary for restore-plan policy.
pub(crate) trait RestorePlanAuthenticator {
    fn validate_public_contract(
        &self,
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> Result<(), RestorePlanError>;

    fn issue(
        &self,
        node_secret: &[u8; 32],
        issued_at: u64,
        aggregate: RestorePlanAggregate,
        boundary: RestorePlanPrivateBoundary<'_>,
        nonce: [u8; RESTORE_PLAN_NONCE_BYTES],
    ) -> Result<ChatRelayRestorePlanReceipt, RestorePlanError>;

    fn verify(
        &self,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
        aggregate: RestorePlanAggregate,
        boundary: RestorePlanPrivateBoundary<'_>,
    ) -> Result<(), RestorePlanError>;
}

/// HMAC-SHA256 implementation preserving the restore-plan v1 contract.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HmacRestorePlanAuthenticator;

#[derive(Serialize)]
struct RestorePlanSigningState<'a> {
    version: u8,
    issued_at: u64,
    expires_at: u64,
    verified_backup_count: u64,
    selected_backup_bytes: u64,
    active_database_present: bool,
    active_database_bytes: u64,
    nonce: &'a str,
    configured_database_path: &'a str,
    selected_backup_name: &'a str,
    selected_backup_modified_secs: u64,
    selected_backup_modified_nanos: u32,
    active_database_modified_secs: u64,
    active_database_modified_nanos: u32,
    selected_backup_device_id: u64,
    selected_backup_inode: u64,
    active_database_device_id: u64,
    active_database_inode: u64,
}

impl HmacRestorePlanAuthenticator {
    fn time_components(time: SystemTime) -> Result<(u64, u32), RestorePlanError> {
        let elapsed = time
            .duration_since(UNIX_EPOCH)
            .map_err(|_| RestorePlanError::FilesystemTimeOutOfRange)?;
        Ok((elapsed.as_secs(), elapsed.subsec_nanos()))
    }

    fn signing_bytes(
        plan: &ChatRelayRestorePlanReceipt,
        boundary: RestorePlanPrivateBoundary<'_>,
    ) -> Result<Vec<u8>, RestorePlanError> {
        let (selected_backup_modified_secs, selected_backup_modified_nanos) =
            Self::time_components(boundary.selected_backup_modified_at)?;
        let (active_database_modified_secs, active_database_modified_nanos) = boundary
            .active_database_modified_at
            .map(Self::time_components)
            .transpose()?
            .unwrap_or_default();
        let signing_state = RestorePlanSigningState {
            version: plan.version,
            issued_at: plan.issued_at,
            expires_at: plan.expires_at,
            verified_backup_count: plan.verified_backup_count,
            selected_backup_bytes: plan.selected_backup_bytes,
            active_database_present: plan.active_database_present,
            active_database_bytes: plan.active_database_bytes,
            nonce: &plan.nonce,
            configured_database_path: boundary.configured_database_path,
            selected_backup_name: boundary.selected_backup_name,
            selected_backup_modified_secs,
            selected_backup_modified_nanos,
            active_database_modified_secs,
            active_database_modified_nanos,
            selected_backup_device_id: boundary.selected_backup_device_id,
            selected_backup_inode: boundary.selected_backup_inode,
            active_database_device_id: boundary.active_database_device_id,
            active_database_inode: boundary.active_database_inode,
        };
        bincode::serialize(&signing_state).map_err(|_| RestorePlanError::EncodingFailed)
    }

    fn mac(
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        boundary: RestorePlanPrivateBoundary<'_>,
    ) -> Result<HmacSha256, RestorePlanError> {
        let mut mac = HmacSha256::new_from_slice(node_secret)
            .map_err(|_| RestorePlanError::AuthenticatorInitFailed)?;
        mac.update(RESTORE_PLAN_HMAC_DOMAIN);
        mac.update(&Self::signing_bytes(plan, boundary)?);
        Ok(mac)
    }

    fn valid_public_contract(plan: &ChatRelayRestorePlanReceipt, now_unix_secs: u64) -> bool {
        plan.version == RESTORE_PLAN_VERSION
            && plan.issued_at.checked_add(RESTORE_PLAN_VALIDITY_SECS) == Some(plan.expires_at)
            && plan.issued_at <= now_unix_secs
            && now_unix_secs < plan.expires_at
            && is_lower_hex(&plan.nonce, RESTORE_PLAN_NONCE_BYTES * 2)
            && is_lower_hex(&plan.commitment, 64)
    }

    fn aggregate_matches(
        plan: &ChatRelayRestorePlanReceipt,
        aggregate: RestorePlanAggregate,
    ) -> bool {
        plan.verified_backup_count == aggregate.verified_backup_count
            && plan.selected_backup_bytes == aggregate.selected_backup_bytes
            && plan.active_database_present == aggregate.active_database_present
            && plan.active_database_bytes == aggregate.active_database_bytes
    }
}

impl RestorePlanAuthenticator for HmacRestorePlanAuthenticator {
    fn validate_public_contract(
        &self,
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> Result<(), RestorePlanError> {
        if Self::valid_public_contract(plan, now_unix_secs) {
            Ok(())
        } else {
            Err(RestorePlanError::InvalidOrStale)
        }
    }

    fn issue(
        &self,
        node_secret: &[u8; 32],
        issued_at: u64,
        aggregate: RestorePlanAggregate,
        boundary: RestorePlanPrivateBoundary<'_>,
        nonce: [u8; RESTORE_PLAN_NONCE_BYTES],
    ) -> Result<ChatRelayRestorePlanReceipt, RestorePlanError> {
        let expires_at = issued_at
            .checked_add(RESTORE_PLAN_VALIDITY_SECS)
            .ok_or(RestorePlanError::ExpiryOutOfRange)?;
        let mut plan = ChatRelayRestorePlanReceipt {
            version: RESTORE_PLAN_VERSION,
            issued_at,
            expires_at,
            verified_backup_count: aggregate.verified_backup_count,
            selected_backup_bytes: aggregate.selected_backup_bytes,
            active_database_present: aggregate.active_database_present,
            active_database_bytes: aggregate.active_database_bytes,
            nonce: hex::encode(nonce),
            commitment: String::new(),
        };
        plan.commitment = hex::encode(
            Self::mac(node_secret, &plan, boundary)?
                .finalize()
                .into_bytes(),
        );
        Ok(plan)
    }

    fn verify(
        &self,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
        aggregate: RestorePlanAggregate,
        boundary: RestorePlanPrivateBoundary<'_>,
    ) -> Result<(), RestorePlanError> {
        self.validate_public_contract(plan, now_unix_secs)?;
        if !Self::aggregate_matches(plan, aggregate) {
            return Err(RestorePlanError::InvalidOrStale);
        }
        let commitment =
            hex::decode(&plan.commitment).map_err(|_| RestorePlanError::InvalidOrStale)?;
        Self::mac(node_secret, plan, boundary)?
            .verify_slice(&commitment)
            .map_err(|_| RestorePlanError::InvalidOrStale)
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
    use std::time::Duration;

    fn aggregate() -> RestorePlanAggregate {
        RestorePlanAggregate {
            verified_backup_count: 3,
            selected_backup_bytes: 4096,
            active_database_present: true,
            active_database_bytes: 8192,
        }
    }

    fn boundary<'a>(path: &'a str, backup_name: &'a str) -> RestorePlanPrivateBoundary<'a> {
        RestorePlanPrivateBoundary {
            configured_database_path: path,
            selected_backup_name: backup_name,
            selected_backup_modified_at: UNIX_EPOCH + Duration::from_secs(100),
            active_database_modified_at: Some(UNIX_EPOCH + Duration::from_secs(200)),
            selected_backup_device_id: 10,
            selected_backup_inode: 11,
            active_database_device_id: 12,
            active_database_inode: 13,
        }
    }

    #[test]
    fn issues_unique_path_free_contract_and_verifies() {
        let authenticator = HmacRestorePlanAuthenticator;
        let plan = authenticator
            .issue(
                &[0x42; 32],
                1_000,
                aggregate(),
                boundary("/private/relay.sqlite", "opaque-backup"),
                [0xA5; RESTORE_PLAN_NONCE_BYTES],
            )
            .expect("issue restore plan");

        assert_eq!(plan.version, RESTORE_PLAN_VERSION);
        assert_eq!(plan.expires_at, 1_000 + RESTORE_PLAN_VALIDITY_SECS);
        assert!(!serde_json::to_string(&plan)
            .expect("encode public plan")
            .contains("private"));
        authenticator
            .verify(
                &[0x42; 32],
                &plan,
                1_000,
                aggregate(),
                boundary("/private/relay.sqlite", "opaque-backup"),
            )
            .expect("verify restore plan");
    }

    #[test]
    fn rejects_public_and_private_state_drift() {
        let authenticator = HmacRestorePlanAuthenticator;
        let plan = authenticator
            .issue(
                &[0x42; 32],
                1_000,
                aggregate(),
                boundary("relay.sqlite", "backup-a"),
                [0x5A; RESTORE_PLAN_NONCE_BYTES],
            )
            .expect("issue restore plan");

        let mut changed = aggregate();
        changed.selected_backup_bytes += 1;
        assert_eq!(
            authenticator.verify(
                &[0x42; 32],
                &plan,
                1_000,
                changed,
                boundary("relay.sqlite", "backup-a"),
            ),
            Err(RestorePlanError::InvalidOrStale)
        );
        assert_eq!(
            authenticator.verify(
                &[0x42; 32],
                &plan,
                1_000,
                aggregate(),
                boundary("relay.sqlite", "backup-b"),
            ),
            Err(RestorePlanError::InvalidOrStale)
        );
    }

    #[test]
    fn rejects_expired_or_wrong_secret_plan() {
        let authenticator = HmacRestorePlanAuthenticator;
        let plan = authenticator
            .issue(
                &[0x42; 32],
                1_000,
                aggregate(),
                boundary("relay.sqlite", "backup-a"),
                [0xC3; RESTORE_PLAN_NONCE_BYTES],
            )
            .expect("issue restore plan");

        assert_eq!(
            authenticator.verify(
                &[0x42; 32],
                &plan,
                plan.expires_at,
                aggregate(),
                boundary("relay.sqlite", "backup-a"),
            ),
            Err(RestorePlanError::InvalidOrStale)
        );
        assert_eq!(
            authenticator.verify(
                &[0x24; 32],
                &plan,
                1_000,
                aggregate(),
                boundary("relay.sqlite", "backup-a"),
            ),
            Err(RestorePlanError::InvalidOrStale)
        );
    }

    #[test]
    fn preserves_legacy_v1_canonical_signing_order() {
        // [CHAT-RELAY-RESTORE-PLAN-DOMAIN 2026-08-26 by Codex] The original
        // service serialized a named struct in this exact field order. Compare
        // against the equivalent tuple so future field movement fails loudly.
        let plan = ChatRelayRestorePlanReceipt {
            version: 1,
            issued_at: 1_000,
            expires_at: 1_600,
            verified_backup_count: 3,
            selected_backup_bytes: 4_096,
            active_database_present: true,
            active_database_bytes: 8_192,
            nonce: "a5".repeat(RESTORE_PLAN_NONCE_BYTES),
            commitment: String::new(),
        };
        let private = boundary("/private/relay.sqlite", "opaque-backup");
        let actual = HmacRestorePlanAuthenticator::signing_bytes(&plan, private)
            .expect("encode canonical signing state");
        #[derive(Serialize)]
        struct LegacyV1SigningState<'a> {
            version: u8,
            issued_at: u64,
            expires_at: u64,
            verified_backup_count: u64,
            selected_backup_bytes: u64,
            active_database_present: bool,
            active_database_bytes: u64,
            nonce: &'a str,
            configured_database_path: &'a str,
            selected_backup_name: &'a str,
            selected_backup_modified_secs: u64,
            selected_backup_modified_nanos: u32,
            active_database_modified_secs: u64,
            active_database_modified_nanos: u32,
            selected_backup_device_id: u64,
            selected_backup_inode: u64,
            active_database_device_id: u64,
            active_database_inode: u64,
        }
        let legacy = LegacyV1SigningState {
            version: plan.version,
            issued_at: plan.issued_at,
            expires_at: plan.expires_at,
            verified_backup_count: plan.verified_backup_count,
            selected_backup_bytes: plan.selected_backup_bytes,
            active_database_present: plan.active_database_present,
            active_database_bytes: plan.active_database_bytes,
            nonce: &plan.nonce,
            configured_database_path: private.configured_database_path,
            selected_backup_name: private.selected_backup_name,
            selected_backup_modified_secs: 100,
            selected_backup_modified_nanos: 0,
            active_database_modified_secs: 200,
            active_database_modified_nanos: 0,
            selected_backup_device_id: private.selected_backup_device_id,
            selected_backup_inode: private.selected_backup_inode,
            active_database_device_id: private.active_database_device_id,
            active_database_inode: private.active_database_inode,
        };
        let expected = bincode::serialize(&legacy).expect("encode legacy v1 field order");

        assert_eq!(actual, expected);
    }
}
