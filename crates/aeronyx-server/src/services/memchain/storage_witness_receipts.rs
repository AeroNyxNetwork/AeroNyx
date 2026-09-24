// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_witness_receipts.rs
// ============================================
//! # Bounded Custody Witness Receipt Vault
//!
//! [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Owns signed receipt
//! admission, exact-frame retention, complete bounded restart audit, and
//! exact-anchor readiness policy. The node stores opaque signed evidence;
//! this module never reads ciphertext, content keys, or social graphs.
//!
//! ## Last Modified
//! [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Extracted the entire
//! receipt state machine and its tests from storage_ops.rs without changing
//! public MemoryStorage methods, signed frames, or the v17/v18 SQLite schema.

use std::collections::HashMap;

use aeronyx_core::protocol::chat::{
    custody_audit_witness_receipt_frame_sha256, decode_custody_audit_witness_receipt,
    encode_custody_audit_witness_receipt, CustodyAuditWitnessReceiptV1,
    CUSTODY_AUDIT_WITNESS_ADVANCED_V1, CUSTODY_AUDIT_WITNESS_CONFLICT_V1,
    CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1, CUSTODY_AUDIT_WITNESS_STALE_V1,
    MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
};
use rusqlite::{params, OptionalExtension};

use super::storage::{MemoryStorage, CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY};
use super::storage_ops::ensure_full_sqlite_durability;
#[cfg(test)]
use super::storage_ops::read_sqlite_durability;

/// Result of atomically retaining one producer-side portable witness receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustodyAuditWitnessReceiptPersistOutcome {
    /// A new immutable receipt was inserted without rotating older evidence.
    Inserted,
    /// A new receipt was inserted after the oldest normal receipt was pruned.
    InsertedAfterNormalPrune,
    /// The exact canonical receipt was already durable.
    AlreadyPresent,
}

impl CustodyAuditWitnessReceiptPersistOutcome {
    /// Stable operator-facing disposition without exposing storage internals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inserted => "inserted",
            Self::InsertedAfterNormalPrune => "inserted_after_normal_prune",
            Self::AlreadyPresent => "already_present",
        }
    }
}

/// Aggregate result of a complete producer receipt-vault cryptographic audit.
///
/// [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] This type intentionally
/// excludes producer/witness identities, frame hashes, signatures, endpoints,
/// and custody counters so it is safe for local operational reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustodyAuditWitnessReceiptVaultAudit {
    /// Canonical signature-verified receipt frames retained locally.
    pub records: usize,
    /// Retained `advanced` or `idempotent` decisions.
    pub accepted_records: usize,
    /// Retained `stale`, `conflict`, or `gap` decisions.
    pub adverse_records: usize,
    /// Latest signed witness observation time in the vault.
    pub latest_observed_at: Option<u64>,
}

/// Restart-safe policy evidence for one exact producer custody anchor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustodyAuditWitnessReceiptPolicyEvidence {
    /// Distinct non-self operator pins considered by the evaluation.
    pub configured: usize,
    /// Fresh canonical receipts matching the exact producer anchor.
    pub fresh_verified: usize,
    /// Fresh matching witnesses retaining the requested anchor.
    pub accepted: usize,
    /// Fresh matching adverse or ambiguous witness decisions.
    pub adverse: usize,
    /// Configured witnesses without a fresh matching receipt.
    pub missing: usize,
    /// Duplicate pins ignored defensively.
    pub duplicates_ignored: usize,
    /// Producer identity pins excluded from independent witness policy.
    pub self_excluded: usize,
    /// Accepted independent receipts required by local policy.
    pub minimum_verified: usize,
    /// Whether the threshold is met with no fresh adverse evidence.
    pub quorum_satisfied: bool,
    /// Inclusive expiry horizon of the newest accepted threshold set.
    ///
    /// [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] This aggregate timestamp
    /// identifies no witness and lets local operations renew evidence before
    /// strict runtime readiness fails.
    pub quorum_valid_through: Option<u64>,
}

/// Typed readiness derived from one internally consistent custody policy.
///
/// [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Startup and every
/// operator command consume this single contract instead of independently
/// interpreting aggregate counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustodyAuditWitnessPolicyReadiness {
    /// The configured independent witness threshold is satisfied.
    Ready,
    /// No configured witness has a fresh exact-anchor receipt.
    EvidenceUnavailable,
    /// Some fresh accepted evidence exists, but it is below the threshold.
    ThresholdUnmet,
    /// Fresh adverse or ambiguous evidence exists for the exact anchor.
    AdverseEvidence,
}

/// Privacy-safe failure class for one atomic custody-readiness snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustodyAuditWitnessReadinessError {
    /// The durable receipt vault failed canonical cryptographic audit.
    VaultInvalid,
    /// Effective pins, thresholds, or derived counters are inconsistent.
    PolicyInvalid,
}

impl CustodyAuditWitnessReadinessError {
    /// Stable aggregate reason code containing no private evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VaultInvalid => "receipt_vault_invalid",
            Self::PolicyInvalid => "receipt_policy_invalid",
        }
    }
}

impl std::fmt::Display for CustodyAuditWitnessReadinessError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::error::Error for CustodyAuditWitnessReadinessError {}

impl CustodyAuditWitnessPolicyReadiness {
    /// Stable aggregate status used by existing operator JSON contracts.
    #[must_use]
    pub const fn status_label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::AdverseEvidence => "adverse",
            Self::EvidenceUnavailable | Self::ThresholdUnmet => "collecting",
        }
    }
}

impl CustodyAuditWitnessReceiptPolicyEvidence {
    /// Returns the remaining inclusive quorum validity at one evaluation time.
    #[must_use]
    pub fn quorum_valid_for_secs(&self, evaluated_at: u64) -> Option<u64> {
        self.quorum_valid_through
            .map(|valid_through| valid_through.saturating_sub(evaluated_at))
    }

    /// Validates counter invariants and derives the authoritative readiness.
    ///
    /// # Errors
    ///
    /// Returns a privacy-safe static error when counters cannot originate from
    /// one valid exact-anchor policy evaluation.
    pub fn readiness(&self) -> Result<CustodyAuditWitnessPolicyReadiness, &'static str> {
        let classified = self
            .accepted
            .checked_add(self.adverse)
            .ok_or("custody witness policy counters are inconsistent")?;
        let expected_ready = self.adverse == 0 && self.accepted >= self.minimum_verified;
        if self.configured == 0
            || self.minimum_verified == 0
            || self.minimum_verified > self.configured
            || self.fresh_verified > self.configured
            || classified != self.fresh_verified
            || self.missing != self.configured.saturating_sub(self.fresh_verified)
            || self.quorum_satisfied != expected_ready
            || self.quorum_valid_through.is_some() != (self.accepted >= self.minimum_verified)
        {
            return Err("custody witness policy counters are inconsistent");
        }
        if self.adverse > 0 {
            Ok(CustodyAuditWitnessPolicyReadiness::AdverseEvidence)
        } else if expected_ready {
            Ok(CustodyAuditWitnessPolicyReadiness::Ready)
        } else if self.fresh_verified == 0 {
            Ok(CustodyAuditWitnessPolicyReadiness::EvidenceUnavailable)
        } else {
            Ok(CustodyAuditWitnessPolicyReadiness::ThresholdUnmet)
        }
    }
}

/// One cryptographically audited SQLite snapshot and its typed policy result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustodyAuditWitnessReceiptReadinessSnapshot {
    /// Complete aggregate vault audit from the same SQLite snapshot.
    pub vault: CustodyAuditWitnessReceiptVaultAudit,
    /// Exact-anchor policy evidence reconstructed from that snapshot.
    pub policy: CustodyAuditWitnessReceiptPolicyEvidence,
    /// Authoritative typed readiness derived from validated policy counters.
    pub readiness: CustodyAuditWitnessPolicyReadiness,
}

/// Returns the bounded local warning window before a ready quorum expires.
///
/// [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] One shared calculation keeps
/// runtime logs and operator CLI reports consistent. It performs no storage or
/// network operation and receives only the configured aggregate age window.
#[must_use]
pub const fn custody_witness_renewal_warning_window_secs(max_age_secs: u64) -> u64 {
    const MIN_WARNING_SECS: u64 = 60;
    const MAX_WARNING_SECS: u64 = 900;
    let quarter_window = max_age_secs / 4;
    if quarter_window < MIN_WARNING_SECS {
        MIN_WARNING_SECS
    } else if quarter_window > MAX_WARNING_SECS {
        MAX_WARNING_SECS
    } else {
        quarter_window
    }
}

const LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS: u64 = 60;
const MAX_OPERATOR_CUSTODY_WITNESS_RECEIPT_IMPORT_AGE_SECS: u64 = 7 * 24 * 60 * 60;
const CUSTODY_WITNESS_RECEIPT_MAX_FUTURE_SKEW_SECS: u64 = 60;

const CUSTODY_WITNESS_RECEIPT_ADMISSION_LIVE: i64 = 0;
const CUSTODY_WITNESS_RECEIPT_ADMISSION_OPERATOR_IMPORT: i64 = 1;

/// Typed local admission boundary for producer-side witness evidence.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Keeping this private
/// prevents any network caller from accidentally selecting the wider manual
/// import window. Its stable numeric evidence is persisted and revalidated on
/// every vault audit; it is not part of the portable signed receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CustodyWitnessReceiptAdmission {
    LiveTransport,
    OperatorImport { max_delay_secs: u64 },
}

/// Canonical, range-checked fields ready for one receipt-vault transaction.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Separating pure frame
/// admission from transactional retention keeps the cryptographic boundary
/// independently reviewable and the lock-held database path narrowly scoped.
struct PreparedCustodyAuditWitnessReceipt {
    frame: Vec<u8>,
    receipt_digest: [u8; 32],
    requested_generation: i64,
    retained_generation: i64,
    observed_at: i64,
    persisted_at: i64,
    admission_kind: i64,
    admission_max_delay_secs: i64,
}

/// Applies one-sided freshness while tolerating only bounded positive skew.
///
/// [CUSTODY-WITNESS-TIME-HARDENING 2026-08-18 by Codex] `abs_diff` made an
/// operator import window double as a future-clock allowance. Keep the wider
/// window exclusively for delayed past evidence and cap future observations
/// independently at the live transport skew budget.
const fn custody_witness_receipt_time_is_admissible(
    reference_at: u64,
    observed_at: u64,
    max_past_age_secs: u64,
) -> bool {
    reference_at > 0
        && observed_at > 0
        && observed_at <= reference_at.saturating_add(CUSTODY_WITNESS_RECEIPT_MAX_FUTURE_SKEW_SECS)
        && reference_at.saturating_sub(observed_at) <= max_past_age_secs
}

fn prepare_custody_audit_witness_receipt(
    receipt: &CustodyAuditWitnessReceiptV1,
    expected_producer: &[u8; 32],
    expected_generation: u64,
    expected_frame_sha256: &[u8; 32],
    persisted_at: u64,
    admission: CustodyWitnessReceiptAdmission,
) -> Result<PreparedCustodyAuditWitnessReceipt, String> {
    let (admission_kind, admission_max_delay_secs) = admission.evidence();
    if expected_producer == &[0u8; 32]
        || expected_frame_sha256 == &[0u8; 32]
        || expected_generation == 0
        || persisted_at == 0
        || expected_generation > i64::MAX as u64
        || !custody_witness_receipt_time_is_admissible(
            persisted_at,
            receipt.observed_at,
            admission.max_delay_secs(),
        )
        || receipt.witness_node_id == *expected_producer
    {
        return Err("custody witness receipt persistence policy is invalid".to_string());
    }
    receipt
        .verify_signature()
        .map_err(|_| "custody witness receipt signature is invalid".to_string())?;
    if &receipt.producer_node_id != expected_producer
        || receipt.requested_checkpoint_generation != expected_generation
        || &receipt.requested_frame_sha256 != expected_frame_sha256
    {
        return Err("custody witness receipt exact-anchor binding mismatch".to_string());
    }
    let frame = encode_custody_audit_witness_receipt(receipt)
        .map_err(|_| "custody witness receipt encode failed".to_string())?;
    if frame.is_empty() || frame.len() > MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES {
        return Err("custody witness receipt frame violates bounds".to_string());
    }
    Ok(PreparedCustodyAuditWitnessReceipt {
        receipt_digest: custody_audit_witness_receipt_frame_sha256(receipt)
            .map_err(|_| "custody witness receipt digest failed".to_string())?,
        frame,
        requested_generation: i64::try_from(receipt.requested_checkpoint_generation)
            .map_err(|_| "custody witness requested generation exceeds SQLite range".to_string())?,
        retained_generation: i64::try_from(receipt.retained_checkpoint_generation)
            .map_err(|_| "custody witness retained generation exceeds SQLite range".to_string())?,
        observed_at: i64::try_from(receipt.observed_at)
            .map_err(|_| "custody witness receipt time exceeds SQLite range".to_string())?,
        persisted_at: i64::try_from(persisted_at)
            .map_err(|_| "custody witness persistence time exceeds SQLite range".to_string())?,
        admission_kind,
        admission_max_delay_secs: i64::try_from(admission_max_delay_secs)
            .map_err(|_| "custody witness admission delay exceeds SQLite range".to_string())?,
    })
}

fn prune_oldest_normal_custody_witness_receipt_if_full(
    transaction: &rusqlite::Transaction<'_>,
    current_records: usize,
) -> Result<bool, String> {
    if current_records < CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY {
        return Ok(false);
    }
    // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Capacity pressure
    // may rotate only normal evidence. Adverse witness decisions remain sticky
    // so a later successful receipt cannot erase evidence of prior divergence.
    let removed = transaction
        .execute(
            "DELETE FROM custody_audit_witness_receipt_evidence WHERE receipt_digest=(
                SELECT receipt_digest FROM custody_audit_witness_receipt_evidence
                WHERE outcome IN (?1,?2)
                ORDER BY observed_at ASC,persisted_at ASC,receipt_digest ASC
                LIMIT 1
            )",
            params![
                i64::from(CUSTODY_AUDIT_WITNESS_ADVANCED_V1),
                i64::from(CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1),
            ],
        )
        .map_err(|error| format!("rotate custody witness receipt evidence: {error}"))?;
    if removed != 1 {
        return Err(
            "custody witness receipt evidence capacity is reserved by adverse evidence".to_string(),
        );
    }
    Ok(true)
}

impl CustodyWitnessReceiptAdmission {
    fn operator_import(max_delay_secs: u64) -> Result<Self, String> {
        if !(LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS
            ..=MAX_OPERATOR_CUSTODY_WITNESS_RECEIPT_IMPORT_AGE_SECS)
            .contains(&max_delay_secs)
        {
            return Err("custody witness receipt import age policy is invalid".to_string());
        }
        Ok(Self::OperatorImport { max_delay_secs })
    }

    fn from_evidence(kind: i64, max_delay_secs: i64) -> Result<Self, String> {
        let max_delay_secs = u64::try_from(max_delay_secs)
            .map_err(|_| "custody witness receipt admission delay is invalid".to_string())?;
        match kind {
            CUSTODY_WITNESS_RECEIPT_ADMISSION_LIVE
                if max_delay_secs == LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS =>
            {
                Ok(Self::LiveTransport)
            }
            CUSTODY_WITNESS_RECEIPT_ADMISSION_OPERATOR_IMPORT => {
                Self::operator_import(max_delay_secs)
            }
            _ => Err("custody witness receipt admission evidence is invalid".to_string()),
        }
    }

    const fn evidence(self) -> (i64, u64) {
        match self {
            Self::LiveTransport => (
                CUSTODY_WITNESS_RECEIPT_ADMISSION_LIVE,
                LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS,
            ),
            Self::OperatorImport { max_delay_secs } => (
                CUSTODY_WITNESS_RECEIPT_ADMISSION_OPERATOR_IMPORT,
                max_delay_secs,
            ),
        }
    }

    const fn max_delay_secs(self) -> u64 {
        self.evidence().1
    }
}

struct StoredCustodyAuditWitnessReceiptRow {
    receipt_digest: Vec<u8>,
    producer: Vec<u8>,
    witness: Vec<u8>,
    requested_generation: i64,
    requested_frame_sha256: Vec<u8>,
    retained_generation: i64,
    retained_frame_sha256: Vec<u8>,
    outcome: i64,
    observed_at: i64,
    receipt_frame: Vec<u8>,
    persisted_at: i64,
    admission_kind: i64,
    admission_max_delay_secs: i64,
}

#[derive(Debug, Clone, Copy)]
struct LatestCustodyAuditWitnessDecision {
    observed_at: u64,
    accepted: bool,
    receipt_digest: [u8; 32],
    same_time_ambiguous: bool,
    sticky_adverse: bool,
}

fn exact_32_bytes(bytes: &[u8], context: &str) -> Result<[u8; 32], String> {
    bytes
        .try_into()
        .map_err(|_| format!("{context} has invalid length"))
}

/// Copies one bounded raw receipt snapshot without doing cryptographic work.
fn load_custody_audit_witness_receipt_rows(
    connection: &rusqlite::Connection,
) -> Result<Vec<StoredCustodyAuditWitnessReceiptRow>, String> {
    let count_i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM custody_audit_witness_receipt_evidence",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| format!("count custody witness receipt evidence: {error}"))?;
    let count = usize::try_from(count_i64)
        .map_err(|_| "custody witness receipt evidence count is invalid".to_string())?;
    if count > CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY {
        return Err("custody witness receipt evidence exceeds its configured capacity".to_string());
    }
    // [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] Row count alone
    // is not a memory bound when a locally replaced database can contain an
    // oversized BLOB. Preflight lengths in the same snapshot before Vec reads.
    let (
        max_receipt_digest_len,
        max_producer_len,
        max_witness_len,
        max_requested_digest_len,
        max_retained_digest_len,
        max_receipt_frame_len,
    ): (i64, i64, i64, i64, i64, i64) = connection
        .query_row(
            "SELECT COALESCE(MAX(length(receipt_digest)),0),
                    COALESCE(MAX(length(producer)),0),
                    COALESCE(MAX(length(witness)),0),
                    COALESCE(MAX(length(requested_frame_sha256)),0),
                    COALESCE(MAX(length(retained_frame_sha256)),0),
                    COALESCE(MAX(length(receipt_frame)),0)
             FROM custody_audit_witness_receipt_evidence",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .map_err(|error| format!("bound custody witness receipt evidence: {error}"))?;
    if [
        max_receipt_digest_len,
        max_producer_len,
        max_witness_len,
        max_requested_digest_len,
        max_retained_digest_len,
    ]
    .into_iter()
    .any(|length| length > 32)
    {
        return Err("custody witness receipt indexed blob violates bounds".to_string());
    }
    let max_frame_bytes = i64::try_from(MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES)
        .map_err(|_| "custody witness receipt frame bound is invalid".to_string())?;
    if max_receipt_frame_len > max_frame_bytes {
        return Err("custody witness receipt frame violates bounds".to_string());
    }

    let mut statement = connection
        .prepare(
            "SELECT receipt_digest,producer,witness,requested_generation,
                    requested_frame_sha256,retained_generation,retained_frame_sha256,
                    outcome,observed_at,receipt_frame,persisted_at,
                    admission_kind,admission_max_delay_secs
             FROM custody_audit_witness_receipt_evidence
             ORDER BY observed_at ASC,receipt_digest ASC",
        )
        .map_err(|error| format!("prepare custody witness receipt audit: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok(StoredCustodyAuditWitnessReceiptRow {
                receipt_digest: row.get(0)?,
                producer: row.get(1)?,
                witness: row.get(2)?,
                requested_generation: row.get(3)?,
                requested_frame_sha256: row.get(4)?,
                retained_generation: row.get(5)?,
                retained_frame_sha256: row.get(6)?,
                outcome: row.get(7)?,
                observed_at: row.get(8)?,
                receipt_frame: row.get(9)?,
                persisted_at: row.get(10)?,
                admission_kind: row.get(11)?,
                admission_max_delay_secs: row.get(12)?,
            })
        })
        .map_err(|error| format!("query custody witness receipt audit: {error}"))?;

    let mut snapshot = Vec::with_capacity(count);
    for row in rows {
        snapshot.push(row.map_err(|error| format!("read custody witness receipt row: {error}"))?);
    }
    if snapshot.len() != count {
        return Err("custody witness receipt evidence count changed during audit".to_string());
    }
    Ok(snapshot)
}

/// Revalidates every retained producer-side custody witness receipt.
///
/// [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] The audit treats the
/// encoded signed frame as authoritative and every SQL column as a redundant
/// index that must match it exactly. A replaced or partially edited database
/// therefore cannot manufacture restart readiness from denormalized columns.
fn audit_custody_audit_witness_receipt_rows(
    rows: Vec<StoredCustodyAuditWitnessReceiptRow>,
) -> Result<
    (
        CustodyAuditWitnessReceiptVaultAudit,
        Vec<CustodyAuditWitnessReceiptV1>,
    ),
    String,
> {
    let mut report = CustodyAuditWitnessReceiptVaultAudit::default();
    let mut receipts = Vec::with_capacity(rows.len());
    for row in rows {
        if row.receipt_frame.is_empty()
            || row.receipt_frame.len() > MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES
        {
            return Err("custody witness receipt frame violates bounds".to_string());
        }
        let receipt = decode_custody_audit_witness_receipt(&row.receipt_frame)
            .map_err(|_| "custody witness receipt frame decode failed".to_string())?;
        let canonical = encode_custody_audit_witness_receipt(&receipt)
            .map_err(|_| "custody witness receipt canonical encode failed".to_string())?;
        if canonical != row.receipt_frame {
            return Err("custody witness receipt frame is non-canonical".to_string());
        }
        receipt
            .verify_signature()
            .map_err(|_| "custody witness receipt signature is invalid".to_string())?;
        let digest = custody_audit_witness_receipt_frame_sha256(&receipt)
            .map_err(|_| "custody witness receipt digest failed".to_string())?;
        if exact_32_bytes(&row.receipt_digest, "custody witness receipt digest")? != digest {
            return Err("custody witness receipt digest mismatch".to_string());
        }

        let requested_generation = u64::try_from(row.requested_generation)
            .map_err(|_| "custody witness requested generation is invalid".to_string())?;
        let retained_generation = u64::try_from(row.retained_generation)
            .map_err(|_| "custody witness retained generation is invalid".to_string())?;
        let outcome = u8::try_from(row.outcome)
            .map_err(|_| "custody witness receipt outcome is invalid".to_string())?;
        let observed_at = u64::try_from(row.observed_at)
            .map_err(|_| "custody witness receipt observation time is invalid".to_string())?;
        let persisted_at = u64::try_from(row.persisted_at)
            .map_err(|_| "custody witness receipt persistence time is invalid".to_string())?;
        let admission = CustodyWitnessReceiptAdmission::from_evidence(
            row.admission_kind,
            row.admission_max_delay_secs,
        )?;
        if !custody_witness_receipt_time_is_admissible(
            persisted_at,
            receipt.observed_at,
            admission.max_delay_secs(),
        ) {
            return Err("custody witness receipt persistence time is inconsistent".to_string());
        }
        if exact_32_bytes(&row.producer, "custody witness receipt producer")?
            != receipt.producer_node_id
            || exact_32_bytes(&row.witness, "custody witness receipt witness")?
                != receipt.witness_node_id
            || requested_generation != receipt.requested_checkpoint_generation
            || exact_32_bytes(
                &row.requested_frame_sha256,
                "custody witness requested frame digest",
            )? != receipt.requested_frame_sha256
            || retained_generation != receipt.retained_checkpoint_generation
            || exact_32_bytes(
                &row.retained_frame_sha256,
                "custody witness retained frame digest",
            )? != receipt.retained_frame_sha256
            || outcome != receipt.outcome
            || observed_at != receipt.observed_at
        {
            return Err(
                "custody witness receipt indexed columns mismatch signed frame".to_string(),
            );
        }

        report.records = report.records.saturating_add(1);
        if receipt.accepted() {
            report.accepted_records = report.accepted_records.saturating_add(1);
        } else {
            report.adverse_records = report.adverse_records.saturating_add(1);
        }
        report.latest_observed_at = Some(
            report
                .latest_observed_at
                .map_or(receipt.observed_at, |latest| {
                    latest.max(receipt.observed_at)
                }),
        );
        receipts.push(receipt);
    }
    Ok((report, receipts))
}

/// Performs an in-transaction audit for receipt-vault mutation paths.
fn audit_custody_audit_witness_receipt_snapshot(
    connection: &rusqlite::Connection,
) -> Result<
    (
        CustodyAuditWitnessReceiptVaultAudit,
        Vec<CustodyAuditWitnessReceiptV1>,
    ),
    String,
> {
    audit_custody_audit_witness_receipt_rows(load_custody_audit_witness_receipt_rows(connection)?)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_custody_audit_witness_receipts(
    receipts: Vec<CustodyAuditWitnessReceiptV1>,
    producer: &[u8; 32],
    requested_generation: u64,
    requested_frame_sha256: &[u8; 32],
    witness_node_ids: &[[u8; 32]],
    minimum_verified: usize,
    now: u64,
    max_age_secs: u64,
) -> Result<
    (
        CustodyAuditWitnessReceiptPolicyEvidence,
        CustodyAuditWitnessPolicyReadiness,
    ),
    CustodyAuditWitnessReadinessError,
> {
    const MAX_POLICY_WITNESSES: usize = 3;
    if producer == &[0u8; 32]
        || requested_frame_sha256 == &[0u8; 32]
        || requested_generation == 0
        || requested_generation > i64::MAX as u64
        || witness_node_ids.len() > MAX_POLICY_WITNESSES
        || minimum_verified == 0
        || minimum_verified > MAX_POLICY_WITNESSES
        || now == 0
        || max_age_secs == 0
    {
        return Err(CustodyAuditWitnessReadinessError::PolicyInvalid);
    }

    let mut evidence = CustodyAuditWitnessReceiptPolicyEvidence {
        minimum_verified,
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    let mut configured = Vec::with_capacity(witness_node_ids.len());
    for witness in witness_node_ids {
        if witness == producer {
            evidence.self_excluded = evidence.self_excluded.saturating_add(1);
        } else if configured.contains(witness) {
            evidence.duplicates_ignored = evidence.duplicates_ignored.saturating_add(1);
        } else {
            configured.push(*witness);
        }
    }
    evidence.configured = configured.len();
    // [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Callers outside
    // config loading must not receive a structurally impossible policy after
    // self/duplicate exclusion changes the effective pin set.
    if minimum_verified > evidence.configured {
        return Err(CustodyAuditWitnessReadinessError::PolicyInvalid);
    }

    let mut latest = HashMap::<[u8; 32], LatestCustodyAuditWitnessDecision>::new();
    for receipt in receipts {
        if receipt.producer_node_id != *producer
            || receipt.requested_checkpoint_generation != requested_generation
            || receipt.requested_frame_sha256 != *requested_frame_sha256
            || !configured.contains(&receipt.witness_node_id)
            || !custody_witness_receipt_time_is_admissible(now, receipt.observed_at, max_age_secs)
        {
            continue;
        }
        let digest = custody_audit_witness_receipt_frame_sha256(&receipt)
            .map_err(|_| CustodyAuditWitnessReadinessError::PolicyInvalid)?;
        let sticky_adverse = matches!(
            receipt.outcome,
            CUSTODY_AUDIT_WITNESS_STALE_V1 | CUSTODY_AUDIT_WITNESS_CONFLICT_V1
        );
        if let Some(current) = latest.get_mut(&receipt.witness_node_id) {
            current.sticky_adverse |= sticky_adverse;
            if receipt.observed_at > current.observed_at {
                current.observed_at = receipt.observed_at;
                current.accepted = receipt.accepted();
                current.receipt_digest = digest;
                current.same_time_ambiguous = false;
            } else if receipt.observed_at == current.observed_at && digest != current.receipt_digest
            {
                // An immediate retry can legitimately replace `advanced`
                // with `idempotent` in one wall-clock second. Any accepted /
                // adverse ambiguity remains fail-closed and order-independent.
                if !(current.accepted && receipt.accepted()) {
                    current.same_time_ambiguous = true;
                }
            }
        } else {
            latest.insert(
                receipt.witness_node_id,
                LatestCustodyAuditWitnessDecision {
                    observed_at: receipt.observed_at,
                    accepted: receipt.accepted(),
                    receipt_digest: digest,
                    same_time_ambiguous: false,
                    sticky_adverse,
                },
            );
        }
    }

    evidence.fresh_verified = latest.len();
    let mut accepted_observed_at = Vec::with_capacity(latest.len());
    for decision in latest.values() {
        if decision.accepted && !decision.same_time_ambiguous && !decision.sticky_adverse {
            evidence.accepted = evidence.accepted.saturating_add(1);
            accepted_observed_at.push(decision.observed_at);
        } else {
            evidence.adverse = evidence.adverse.saturating_add(1);
        }
    }
    // [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] The threshold-th newest
    // accepted observation is the exact point at which the currently usable
    // quorum loses one required member. Older surplus evidence must not make
    // the renewal horizon look earlier than it really is.
    accepted_observed_at.sort_unstable_by(|left, right| right.cmp(left));
    evidence.quorum_valid_through = accepted_observed_at
        .get(minimum_verified.saturating_sub(1))
        .map(|observed_at| observed_at.saturating_add(max_age_secs));
    evidence.missing = evidence.configured.saturating_sub(evidence.fresh_verified);
    evidence.quorum_satisfied = evidence.accepted >= minimum_verified && evidence.adverse == 0;
    let readiness = evidence
        .readiness()
        .map_err(|_| CustodyAuditWitnessReadinessError::PolicyInvalid)?;
    Ok((evidence, readiness))
}

// ============================================
// impl MemoryStorage — Custody Witness Receipt Evidence
// ============================================

/// One narrow owner for the receipt vault's SQLite transaction and snapshot
/// boundaries; neither the connection nor its durability atomic is exported.
// [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Admission and policy
// remain pure above, while this component alone arms a write after FULL
// readback and releases a bounded raw snapshot before signature work.
struct CustodyWitnessReceiptVault<'a> {
    storage: &'a MemoryStorage,
}

impl<'a> CustodyWitnessReceiptVault<'a> {
    const fn new(storage: &'a MemoryStorage) -> Self {
        Self { storage }
    }

    fn begin_write<'connection>(
        &self,
        connection: &'connection mut rusqlite::Connection,
    ) -> Result<rusqlite::Transaction<'connection>, String> {
        // SQLite forbids changing synchronous after BEGIN. Keep the readback
        // and transaction start under the caller's same connection lock.
        let (level, _) = ensure_full_sqlite_durability(connection)?;
        self.storage.note_effective_sqlite_durability(level);
        connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin custody witness receipt transaction: {error}"))
    }

    async fn snapshot(&self) -> Result<Vec<StoredCustodyAuditWitnessReceiptRow>, String> {
        let mut connection = self.storage.conn_lock().await;
        let transaction = connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin custody witness receipt audit: {error}"))?;
        let rows = load_custody_audit_witness_receipt_rows(&transaction)?;
        transaction
            .commit()
            .map_err(|error| format!("finish custody witness receipt audit: {error}"))?;
        Ok(rows)
    }
}

impl MemoryStorage {
    /// Atomically retains one exact signed witness receipt and re-audits the
    /// complete bounded producer-side vault before commit.
    ///
    /// Normal accepted evidence may rotate oldest-first at capacity. Adverse
    /// `stale`, `conflict`, or `gap` evidence is never silently pruned; a vault
    /// containing only adverse evidence fails closed and requires review.
    ///
    /// # Errors
    ///
    /// Returns an error when the signed receipt violates live admission or
    /// exact-anchor policy, or when the transactional vault audit cannot finish.
    pub async fn persist_custody_audit_witness_receipt(
        &self,
        receipt: &CustodyAuditWitnessReceiptV1,
        expected_producer: &[u8; 32],
        expected_generation: u64,
        expected_frame_sha256: &[u8; 32],
        persisted_at: u64,
    ) -> Result<CustodyAuditWitnessReceiptPersistOutcome, String> {
        self.persist_custody_audit_witness_receipt_with_admission(
            receipt,
            expected_producer,
            expected_generation,
            expected_frame_sha256,
            persisted_at,
            CustodyWitnessReceiptAdmission::LiveTransport,
        )
        .await
    }

    /// Imports one operator-carried signed receipt into the same bounded vault.
    ///
    /// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Live transport
    /// retains its strict 60-second persistence delay. This separate host-local
    /// path permits an explicitly configured delay of at most seven days so an
    /// air-gapped witness workflow can complete without falsifying
    /// `persisted_at` or weakening automatic network admission.
    ///
    /// # Errors
    ///
    /// Returns an error when the import window, signature, exact-anchor binding,
    /// or transactional receipt-vault invariants cannot be verified.
    pub async fn import_custody_audit_witness_receipt(
        &self,
        receipt: &CustodyAuditWitnessReceiptV1,
        expected_producer: &[u8; 32],
        expected_generation: u64,
        expected_frame_sha256: &[u8; 32],
        imported_at: u64,
        max_receipt_age_secs: u64,
    ) -> Result<CustodyAuditWitnessReceiptPersistOutcome, String> {
        let admission = CustodyWitnessReceiptAdmission::operator_import(max_receipt_age_secs)?;
        self.persist_custody_audit_witness_receipt_with_admission(
            receipt,
            expected_producer,
            expected_generation,
            expected_frame_sha256,
            imported_at,
            admission,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn persist_custody_audit_witness_receipt_with_admission(
        &self,
        receipt: &CustodyAuditWitnessReceiptV1,
        expected_producer: &[u8; 32],
        expected_generation: u64,
        expected_frame_sha256: &[u8; 32],
        persisted_at: u64,
        admission: CustodyWitnessReceiptAdmission,
    ) -> Result<CustodyAuditWitnessReceiptPersistOutcome, String> {
        let PreparedCustodyAuditWitnessReceipt {
            frame,
            receipt_digest,
            requested_generation,
            retained_generation,
            observed_at,
            persisted_at,
            admission_kind,
            admission_max_delay_secs,
        } = prepare_custody_audit_witness_receipt(
            receipt,
            expected_producer,
            expected_generation,
            expected_frame_sha256,
            persisted_at,
            admission,
        )?;

        let vault = CustodyWitnessReceiptVault::new(self);
        let mut conn = self.conn_lock().await;
        let transaction = vault.begin_write(&mut conn)?;
        let (before, _) = audit_custody_audit_witness_receipt_snapshot(&transaction)?;
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT receipt_frame FROM custody_audit_witness_receipt_evidence
                 WHERE receipt_digest=?1",
                params![receipt_digest.as_slice()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| format!("read existing custody witness receipt: {error}"))?;
        if let Some(existing) = existing {
            if existing != frame {
                return Err("custody witness receipt digest collision".to_string());
            }
            transaction
                .commit()
                .map_err(|error| format!("finish custody witness receipt transaction: {error}"))?;
            drop(conn);
            return Ok(CustodyAuditWitnessReceiptPersistOutcome::AlreadyPresent);
        }

        let rotated =
            prune_oldest_normal_custody_witness_receipt_if_full(&transaction, before.records)?;

        transaction
            .execute(
                "INSERT INTO custody_audit_witness_receipt_evidence
                 (receipt_digest,producer,witness,requested_generation,
                  requested_frame_sha256,retained_generation,retained_frame_sha256,
                  outcome,observed_at,receipt_frame,persisted_at,admission_kind,
                  admission_max_delay_secs)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
                params![
                    receipt_digest.as_slice(),
                    receipt.producer_node_id.as_slice(),
                    receipt.witness_node_id.as_slice(),
                    requested_generation,
                    receipt.requested_frame_sha256.as_slice(),
                    retained_generation,
                    receipt.retained_frame_sha256.as_slice(),
                    i64::from(receipt.outcome),
                    observed_at,
                    frame,
                    persisted_at,
                    admission_kind,
                    admission_max_delay_secs,
                ],
            )
            .map_err(|error| format!("insert custody witness receipt evidence: {error}"))?;
        let (after, _) = audit_custody_audit_witness_receipt_snapshot(&transaction)?;
        if after.records > CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY {
            return Err("custody witness receipt evidence capacity invariant failed".to_string());
        }
        transaction
            .commit()
            .map_err(|error| format!("commit custody witness receipt evidence: {error}"))?;
        drop(conn);
        Ok(if rotated {
            CustodyAuditWitnessReceiptPersistOutcome::InsertedAfterNormalPrune
        } else {
            CustodyAuditWitnessReceiptPersistOutcome::Inserted
        })
    }

    /// Revalidates all retained producer-side receipt evidence in one `SQLite`
    /// snapshot and returns aggregate-only operational state.
    ///
    /// # Errors
    ///
    /// Returns an error when the bounded row snapshot cannot be read or any
    /// retained receipt fails canonical, signature, or redundant-index audit.
    pub async fn audit_custody_audit_witness_receipt_evidence(
        &self,
    ) -> Result<CustodyAuditWitnessReceiptVaultAudit, String> {
        // [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] The bounded
        // raw rows are immutable process-owned values after this block. Decode,
        // canonicalization, digesting, and signature checks do not hold SQLite.
        let rows = CustodyWitnessReceiptVault::new(self).snapshot().await?;
        let (report, _) = audit_custody_audit_witness_receipt_rows(rows)?;
        Ok(report)
    }

    /// Reconstructs current local witness policy for one exact producer anchor
    /// exclusively from fresh, canonical, signature-verified durable receipts.
    ///
    /// The newest receipt per distinct non-self operator pin supplies current
    /// state. Any fresh `stale`/`conflict` receipt remains sticky adverse for
    /// that exact anchor, and same-time accepted/adverse ambiguity fails closed.
    ///
    /// # Errors
    ///
    /// Returns an error when the canonical receipt vault fails audit or the
    /// effective witness pin policy is structurally impossible.
    #[allow(clippy::too_many_arguments)]
    pub async fn evaluate_custody_audit_witness_receipt_policy(
        &self,
        producer: &[u8; 32],
        requested_generation: u64,
        requested_frame_sha256: &[u8; 32],
        witness_node_ids: &[[u8; 32]],
        minimum_verified: usize,
        now: u64,
        max_age_secs: u64,
    ) -> Result<CustodyAuditWitnessReceiptPolicyEvidence, String> {
        Ok(self
            .audit_custody_audit_witness_receipt_readiness(
                producer,
                requested_generation,
                requested_frame_sha256,
                witness_node_ids,
                minimum_verified,
                now,
                max_age_secs,
            )
            .await
            .map_err(|error| error.to_string())?
            .policy)
    }

    /// Audits the complete vault and derives exact-anchor readiness from the
    /// same `SQLite` snapshot.
    ///
    /// [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] This is the
    /// authoritative startup/operator boundary. A command can no longer log
    /// one vault snapshot while making its decision from a later snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when the vault is malformed, signatures or redundant
    /// indexes fail verification, or the effective pin policy is impossible.
    #[allow(clippy::too_many_arguments)]
    pub async fn audit_custody_audit_witness_receipt_readiness(
        &self,
        producer: &[u8; 32],
        requested_generation: u64,
        requested_frame_sha256: &[u8; 32],
        witness_node_ids: &[[u8; 32]],
        minimum_verified: usize,
        now: u64,
        max_age_secs: u64,
    ) -> Result<CustodyAuditWitnessReceiptReadinessSnapshot, CustodyAuditWitnessReadinessError>
    {
        let rows = CustodyWitnessReceiptVault::new(self)
            .snapshot()
            .await
            .map_err(|_| CustodyAuditWitnessReadinessError::VaultInvalid)?;
        let (vault, receipts) = audit_custody_audit_witness_receipt_rows(rows)
            .map_err(|_| CustodyAuditWitnessReadinessError::VaultInvalid)?;
        let (policy, readiness) = evaluate_custody_audit_witness_receipts(
            receipts,
            producer,
            requested_generation,
            requested_frame_sha256,
            witness_node_ids,
            minimum_verified,
            now,
            max_age_secs,
        )?;
        Ok(CustodyAuditWitnessReceiptReadinessSnapshot {
            vault,
            policy,
            readiness,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::CUSTODY_AUDIT_WITNESS_GAP_V1;
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;

    #[tokio::test]
    async fn custody_witness_receipt_import_is_bounded_and_preserves_real_import_time() {
        // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Manual transfer
        // may exceed the live 60-second window, but must retain the real import
        // time, reject clock rollback/future evidence, and remain idempotent.
        // [CUSTODY-WITNESS-TIME-HARDENING 2026-08-18 by Codex] The manual
        // import age applies only to past evidence. Positive skew remains
        // bounded to 60 seconds regardless of the configured import window.
        let producer = IdentityKeyPair::from_bytes(&[0x7d; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x7e; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let frame_sha256 = [0x7f; 32];
        let observed_at = 1_700_700_000;
        let receipt = CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            1,
            frame_sha256,
            observed_at,
            1,
            frame_sha256,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .unwrap();
        let storage = MemoryStorage::open(":memory:", None).unwrap();

        assert!(custody_witness_receipt_time_is_admissible(
            observed_at,
            observed_at + CUSTODY_WITNESS_RECEIPT_MAX_FUTURE_SKEW_SECS,
            300,
        ));
        assert!(!custody_witness_receipt_time_is_admissible(
            observed_at,
            observed_at + CUSTODY_WITNESS_RECEIPT_MAX_FUTURE_SKEW_SECS + 1,
            300,
        ));
        assert!(custody_witness_receipt_time_is_admissible(
            observed_at,
            observed_at - 300,
            300,
        ));
        assert!(!custody_witness_receipt_time_is_admissible(
            observed_at,
            observed_at - 301,
            300,
        ));

        assert!(storage
            .persist_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at + 120,
            )
            .await
            .is_err());
        assert_eq!(
            storage
                .import_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at + 120,
                    300,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::Inserted
        );
        assert_eq!(
            storage
                .import_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at + 121,
                    300,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::AlreadyPresent
        );
        assert!(storage
            .import_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at - 301,
                300,
            )
            .await
            .is_err());
        assert!(storage
            .import_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at - CUSTODY_WITNESS_RECEIPT_MAX_FUTURE_SKEW_SECS - 1,
                300,
            )
            .await
            .is_err());
        assert!(storage
            .import_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at + 120,
                7 * 24 * 60 * 60 + 1,
            )
            .await
            .is_err());
        assert!(storage
            .import_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at + 120,
                LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS - 1,
            )
            .await
            .is_err());

        let persisted_policy: (i64, i64, i64) = {
            let conn = storage.conn_lock().await;
            conn.query_row(
                "SELECT persisted_at,admission_kind,admission_max_delay_secs
                 FROM custody_audit_witness_receipt_evidence",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap()
        };
        assert_eq!(
            persisted_policy,
            (
                (observed_at + 120) as i64,
                CUSTODY_WITNESS_RECEIPT_ADMISSION_OPERATOR_IMPORT,
                300,
            )
        );

        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE custody_audit_witness_receipt_evidence
                 SET admission_kind=?1,admission_max_delay_secs=?2",
                params![
                    CUSTODY_WITNESS_RECEIPT_ADMISSION_LIVE,
                    LIVE_CUSTODY_WITNESS_RECEIPT_MAX_DELAY_SECS as i64,
                ],
            )
            .unwrap();
        }
        assert!(storage
            .audit_custody_audit_witness_receipt_evidence()
            .await
            .unwrap_err()
            .contains("persistence time is inconsistent"));
    }

    #[test]
    fn custody_witness_quorum_expiry_tracks_threshold_newest_receipts() {
        // [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] With three accepted
        // witnesses and a threshold of two, the second-newest observation is
        // the real renewal horizon; an older surplus receipt is irrelevant.
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let frame_sha256 = [0x72; 32];
        let witnesses = [
            IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap(),
            IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap(),
            IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap(),
        ];
        let observed_at = [100, 120, 140];
        let receipts = witnesses
            .iter()
            .zip(observed_at)
            .map(|(witness, observed_at)| {
                CustodyAuditWitnessReceiptV1::signed(
                    producer_id,
                    1,
                    frame_sha256,
                    observed_at,
                    1,
                    frame_sha256,
                    CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
                    witness,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let witness_ids = witnesses
            .iter()
            .map(IdentityKeyPair::public_key_bytes)
            .collect::<Vec<_>>();

        let (ready, readiness) = evaluate_custody_audit_witness_receipts(
            receipts.clone(),
            &producer_id,
            1,
            &frame_sha256,
            &witness_ids,
            2,
            150,
            100,
        )
        .unwrap();
        assert_eq!(readiness, CustodyAuditWitnessPolicyReadiness::Ready);
        assert_eq!(ready.quorum_valid_through, Some(220));
        assert_eq!(ready.quorum_valid_for_secs(150), Some(70));

        let (expired, readiness) = evaluate_custody_audit_witness_receipts(
            receipts,
            &producer_id,
            1,
            &frame_sha256,
            &witness_ids,
            2,
            221,
            100,
        )
        .unwrap();
        assert_eq!(
            readiness,
            CustodyAuditWitnessPolicyReadiness::ThresholdUnmet
        );
        assert_eq!(expired.quorum_valid_through, None);
        assert_eq!(expired.quorum_valid_for_secs(221), None);
    }

    #[tokio::test]
    async fn custody_witness_receipt_vault_survives_restart_and_rejects_tampering() {
        // [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] Exercise a real
        // file reopen so restart readiness never depends on process-local round
        // counters, then prove a denormalized-column edit fails full re-audit.
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("custody-witness-receipts.db");
        let producer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let witness_id = witness.public_key_bytes();
        let frame_sha256 = [0x83; 32];
        let observed_at = 1_700_800_001;
        let receipt = CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            1,
            frame_sha256,
            observed_at,
            1,
            frame_sha256,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .unwrap();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        assert_eq!(
            storage
                .persist_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::Inserted
        );
        assert_eq!(
            storage
                .persist_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at.saturating_add(1),
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::AlreadyPresent
        );
        assert_eq!(
            storage
                .audit_custody_audit_witness_receipt_evidence()
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptVaultAudit {
                records: 1,
                accepted_records: 1,
                adverse_records: 0,
                latest_observed_at: Some(observed_at),
            }
        );
        drop(storage);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        let readiness = reopened
            .audit_custody_audit_witness_receipt_readiness(
                &producer_id,
                1,
                &frame_sha256,
                &[witness_id, witness_id, producer_id],
                1,
                observed_at.saturating_add(10),
                60,
            )
            .await
            .unwrap();
        assert_eq!(readiness.vault.records, 1);
        assert_eq!(
            readiness.readiness,
            CustodyAuditWitnessPolicyReadiness::Ready
        );
        let policy = readiness.policy;
        assert_eq!(policy.configured, 1);
        assert_eq!(policy.duplicates_ignored, 1);
        assert_eq!(policy.self_excluded, 1);
        assert_eq!(policy.accepted, 1);
        assert!(policy.quorum_satisfied);
        assert_eq!(policy.quorum_valid_through, Some(observed_at + 60));
        assert_eq!(policy.quorum_valid_for_secs(observed_at + 10), Some(50));
        assert_eq!(
            reopened
                .audit_custody_audit_witness_receipt_readiness(
                    &producer_id,
                    1,
                    &frame_sha256,
                    &[producer_id],
                    1,
                    observed_at.saturating_add(10),
                    60,
                )
                .await,
            Err(CustodyAuditWitnessReadinessError::PolicyInvalid)
        );

        let conflict_receipt = CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            1,
            frame_sha256,
            observed_at.saturating_add(20),
            1,
            [0x87; 32],
            CUSTODY_AUDIT_WITNESS_CONFLICT_V1,
            &witness,
        )
        .unwrap();
        reopened
            .persist_custody_audit_witness_receipt(
                &conflict_receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at.saturating_add(20),
            )
            .await
            .unwrap();
        let impossible_recovery = reopened
            .audit_custody_audit_witness_receipt_readiness(
                &producer_id,
                1,
                &frame_sha256,
                &[witness_id],
                1,
                observed_at.saturating_add(30),
                60,
            )
            .await
            .unwrap();
        assert_eq!(
            impossible_recovery.readiness,
            CustodyAuditWitnessPolicyReadiness::AdverseEvidence
        );
        assert_eq!(impossible_recovery.policy.accepted, 0);
        assert_eq!(impossible_recovery.policy.adverse, 1);
        assert!(!impossible_recovery.policy.quorum_satisfied);
        // [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] Capture the
        // exact raw snapshot, release SQLite, then prove later database edits
        // cannot alter the detached bytes being cryptographically audited.
        let detached_rows = {
            let mut conn = reopened.conn_lock().await;
            let transaction = conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
                .unwrap();
            let rows = load_custody_audit_witness_receipt_rows(&transaction).unwrap();
            transaction.commit().unwrap();
            rows
        };
        {
            let conn = reopened.conn_lock().await;
            conn.execute(
                "UPDATE custody_audit_witness_receipt_evidence SET outcome=?1",
                params![i64::from(CUSTODY_AUDIT_WITNESS_GAP_V1)],
            )
            .unwrap();
        }
        let (detached_audit, detached_receipts) =
            audit_custody_audit_witness_receipt_rows(detached_rows).unwrap();
        assert_eq!(detached_audit.records, 2);
        assert_eq!(detached_receipts.len(), 2);
        assert!(reopened
            .audit_custody_audit_witness_receipt_evidence()
            .await
            .unwrap_err()
            .contains("indexed columns mismatch"));

        let timestamp_tamper = MemoryStorage::open(":memory:", None).unwrap();
        timestamp_tamper
            .persist_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at,
            )
            .await
            .unwrap();
        {
            let conn = timestamp_tamper.conn_lock().await;
            conn.execute(
                "UPDATE custody_audit_witness_receipt_evidence SET persisted_at=1",
                [],
            )
            .unwrap();
        }
        assert!(timestamp_tamper
            .audit_custody_audit_witness_receipt_evidence()
            .await
            .unwrap_err()
            .contains("persistence time is inconsistent"));

        let oversized_blob = MemoryStorage::open(":memory:", None).unwrap();
        oversized_blob
            .persist_custody_audit_witness_receipt(
                &receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at,
            )
            .await
            .unwrap();
        {
            let conn = oversized_blob.conn_lock().await;
            let oversized_frame_bytes =
                i64::try_from(MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES).unwrap() + 1;
            conn.execute_batch("PRAGMA ignore_check_constraints=ON;")
                .unwrap();
            conn.execute(
                "UPDATE custody_audit_witness_receipt_evidence
                 SET receipt_frame=zeroblob(?1)",
                params![oversized_frame_bytes],
            )
            .unwrap();
        }
        assert!(oversized_blob
            .audit_custody_audit_witness_receipt_evidence()
            .await
            .unwrap_err()
            .contains("frame violates bounds"));

        let mut self_receipt = receipt.clone();
        self_receipt.witness_node_id = producer_id;
        assert!(timestamp_tamper
            .persist_custody_audit_witness_receipt(
                &self_receipt,
                &producer_id,
                1,
                &frame_sha256,
                observed_at,
            )
            .await
            .unwrap_err()
            .contains("persistence policy is invalid"));
    }

    #[tokio::test]
    async fn custody_witness_receipt_capacity_rotates_normal_but_preserves_adverse_evidence() {
        // [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] Seed canonical
        // signed rows directly, then exercise the public insertion path. This
        // keeps the test bounded while verifying full-vault re-audit, oldest
        // normal rotation, and fail-closed adverse-evidence reservation.
        async fn seed_receipts(
            storage: &MemoryStorage,
            producer_id: [u8; 32],
            witness: &IdentityKeyPair,
            adverse: bool,
        ) {
            let mut conn = storage.conn_lock().await;
            let transaction = conn.transaction().unwrap();
            for index in 0..CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY {
                let retained_generation = u64::try_from(index).unwrap().saturating_add(1);
                let requested_generation = if adverse {
                    retained_generation.saturating_add(2)
                } else {
                    retained_generation
                };
                let retained_hash = Sha256::digest(retained_generation.to_le_bytes()).into();
                let requested_hash = if adverse {
                    Sha256::digest(requested_generation.to_be_bytes()).into()
                } else {
                    retained_hash
                };
                let outcome = if adverse {
                    CUSTODY_AUDIT_WITNESS_GAP_V1
                } else {
                    CUSTODY_AUDIT_WITNESS_ADVANCED_V1
                };
                let observed_at = 1_700_810_000u64.saturating_add(retained_generation);
                let receipt = CustodyAuditWitnessReceiptV1::signed(
                    producer_id,
                    requested_generation,
                    requested_hash,
                    observed_at,
                    retained_generation,
                    retained_hash,
                    outcome,
                    witness,
                )
                .unwrap();
                let frame = encode_custody_audit_witness_receipt(&receipt).unwrap();
                let digest = custody_audit_witness_receipt_frame_sha256(&receipt).unwrap();
                transaction
                    .execute(
                        "INSERT INTO custody_audit_witness_receipt_evidence
                         (receipt_digest,producer,witness,requested_generation,
                          requested_frame_sha256,retained_generation,retained_frame_sha256,
                          outcome,observed_at,receipt_frame,persisted_at)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?9)",
                        params![
                            digest.as_slice(),
                            receipt.producer_node_id.as_slice(),
                            receipt.witness_node_id.as_slice(),
                            i64::try_from(receipt.requested_checkpoint_generation).unwrap(),
                            receipt.requested_frame_sha256.as_slice(),
                            i64::try_from(receipt.retained_checkpoint_generation).unwrap(),
                            receipt.retained_frame_sha256.as_slice(),
                            i64::from(receipt.outcome),
                            i64::try_from(receipt.observed_at).unwrap(),
                            frame,
                        ],
                    )
                    .unwrap();
            }
            transaction.commit().unwrap();
        }

        let producer = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x85; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let normal = MemoryStorage::open(":memory:", None).unwrap();
        seed_receipts(&normal, producer_id, &witness, false).await;
        let next_generation = CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY as u64 + 1;
        let next_hash = [0x86; 32];
        let next_receipt = CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            next_generation,
            next_hash,
            1_700_820_000,
            next_generation,
            next_hash,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .unwrap();
        assert_eq!(
            normal
                .persist_custody_audit_witness_receipt(
                    &next_receipt,
                    &producer_id,
                    next_generation,
                    &next_hash,
                    1_700_820_001,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::InsertedAfterNormalPrune
        );
        let normal_audit = normal
            .audit_custody_audit_witness_receipt_evidence()
            .await
            .unwrap();
        assert_eq!(
            normal_audit.records,
            CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY
        );
        assert_eq!(normal_audit.adverse_records, 0);

        let adverse = MemoryStorage::open(":memory:", None).unwrap();
        seed_receipts(&adverse, producer_id, &witness, true).await;
        assert_eq!(
            adverse
                .audit_custody_audit_witness_receipt_evidence()
                .await
                .unwrap()
                .adverse_records,
            CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY
        );
        assert!(adverse
            .persist_custody_audit_witness_receipt(
                &next_receipt,
                &producer_id,
                next_generation,
                &next_hash,
                1_700_820_001,
            )
            .await
            .unwrap_err()
            .contains("reserved by adverse evidence"));
    }

    #[tokio::test]
    async fn custody_witness_receipt_ack_requires_full_sqlite_durability() {
        // [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] A file-backed
        // non-coordinator starts in WAL/NORMAL. Its first acknowledged receipt
        // must upgrade the same connection before commit, and a reopened vault
        // must still audit the exact opaque signed evidence.
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("custody-receipt-durability.db");
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        assert_eq!(
            storage
                .configure_record_commitment_durability(false)
                .await
                .unwrap(),
            "normal"
        );
        let producer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let frame_sha256 = [0x93; 32];
        let observed_at = 1_700_830_000;
        let receipt = CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            1,
            frame_sha256,
            observed_at,
            1,
            frame_sha256,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .unwrap();
        assert_eq!(
            storage
                .persist_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::Inserted
        );
        let level: i64 = {
            let conn = storage.conn_lock().await;
            conn.query_row("PRAGMA synchronous", [], |row| row.get(0))
                .unwrap()
        };
        assert!(level >= 2);
        assert_eq!(
            storage
                .record_commitment_chain_integrity_status()
                .durability_mode,
            "full"
        );
        drop(storage);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        assert_eq!(
            reopened
                .audit_custody_audit_witness_receipt_evidence()
                .await
                .unwrap()
                .records,
            1
        );
        assert_eq!(
            reopened
                .persist_custody_audit_witness_receipt(
                    &receipt,
                    &producer_id,
                    1,
                    &frame_sha256,
                    observed_at + 1,
                )
                .await
                .unwrap(),
            CustodyAuditWitnessReceiptPersistOutcome::AlreadyPresent
        );
        let conn = reopened.conn_lock().await;
        assert!(read_sqlite_durability(&conn).unwrap().0 >= 2);
    }

    #[test]
    fn custody_witness_receipt_durability_upgrade_fails_inside_transaction() {
        // [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] This is the
        // deterministic SQLite failure boundary: a synchronous change after
        // BEGIN is forbidden, so the shared helper must never certify NORMAL
        // or allow the caller to report an inserted receipt.
        let connection = rusqlite::Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA synchronous=NORMAL; BEGIN IMMEDIATE;")
            .unwrap();
        assert!(ensure_full_sqlite_durability(&connection).is_err());
        assert_eq!(read_sqlite_durability(&connection).unwrap().0, 1);
        let changes: i64 = connection
            .query_row("SELECT total_changes()", [], |row| row.get(0))
            .unwrap();
        assert_eq!(changes, 0);
        connection.execute_batch("ROLLBACK;").unwrap();
    }
}
