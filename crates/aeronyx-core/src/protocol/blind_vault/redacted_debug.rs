// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/redacted_debug.rs
// ============================================
//! # Privacy-safe Debug formatting
//!
//! Owns the closed redacted `Debug` implementation shared by capability-,
//! topology-, and commitment-bearing protocol values.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::delete::{BlindVaultDeleteRequest, BlindVaultDeletedReceipt};
use super::lease_admission::{
    BlindVaultAdmissionTicket, BlindVaultBlindAdmissionToken, BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultBlindLeaseAdmissionRequest, BlindVaultLeaseAdmissionRequest,
    BlindVaultLeaseCreateRequest,
};
use super::lease_inventory::{
    BlindVaultInventoryCommitmentEntry, BlindVaultInventoryCommitmentSummary,
    BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest,
};
use super::lease_renewal::{
    BlindVaultBlindLeaseRenewalRequest, BlindVaultBlindLeaseRenewedReceipt,
    BlindVaultLeaseRenewRequest,
};
use super::lease_retire::{BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt};
use super::lease_status::{BlindVaultLeaseStatusReceipt, BlindVaultLeaseStatusRequest};
use super::replica_plan::{
    BlindVaultReplicaManifestExpectation, BlindVaultVerifiedReplicaInventory,
};

// [BLIND-VAULT-PRIVACY-SAFE-TYPED-DEBUG 2026-08-30 by Codex] Typed handlers
// frequently format a decoded request directly rather than its outer frame.
// Capability- and topology-bearing protocol values therefore share one closed
// redacted implementation instead of relying on every caller to remember.
macro_rules! impl_blind_vault_redacted_debug {
    ($($value:ty),+ $(,)?) => {
        $(
            impl std::fmt::Debug for $value {
                fn fmt(
                    &self,
                    formatter: &mut std::fmt::Formatter<'_>,
                ) -> std::fmt::Result {
                    formatter
                        .debug_struct(stringify!($value))
                        .field("private_fields", &"[REDACTED]")
                        .finish_non_exhaustive()
                }
            }
        )+
    };
}

impl_blind_vault_redacted_debug!(
    BlindVaultLeaseCreateRequest,
    BlindVaultAdmissionTicket,
    BlindVaultLeaseAdmissionRequest,
    BlindVaultBlindAdmissionToken,
    BlindVaultBlindLeaseAdmissionRequest,
    BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultDeleteRequest,
    BlindVaultLeaseRetireRequest,
    BlindVaultLeaseRetiredReceipt,
    BlindVaultLeaseRenewRequest,
    BlindVaultBlindLeaseRenewalRequest,
    BlindVaultBlindLeaseRenewedReceipt,
    BlindVaultLeaseStatusRequest,
    BlindVaultLeaseStatusReceipt,
    BlindVaultInventoryCommitmentEntry,
    BlindVaultInventoryCommitmentSummary,
    BlindVaultLeaseInventoryRequest,
    BlindVaultLeaseInventoryReceipt,
    BlindVaultReplicaManifestExpectation,
    BlindVaultVerifiedReplicaInventory,
    BlindVaultDeletedReceipt,
);
