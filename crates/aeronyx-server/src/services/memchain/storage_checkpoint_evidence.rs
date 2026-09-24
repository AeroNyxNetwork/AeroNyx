// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_checkpoint_evidence.rs
// ============================================
//! Checkpoint evidence vault validation and bounded incident retention.
//!
//! [MEMCHAIN-CHECKPOINT-EVIDENCE-SPLIT 2026-09-25 by Codex]
//! Owns canonical frame validation, bounded evidence re-audit, certificate
//! reconstruction, and trusted incident retention without changing the public
//! MemoryStorage API, SQLite schema, wire format, or privacy boundary.

use std::collections::HashMap;

use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};

use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::ledger::{AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH};
use aeronyx_core::protocol::memchain::{
    decode_memchain, encode_memchain, record_chain_checkpoint_response_signing_bytes,
    record_checkpoint_certificate_digest_v1, MemChainMessage, MEMCHAIN_MAGIC,
};

use super::storage::{
    CHECKPOINT_CERTIFICATE_CAPACITY, CHECKPOINT_EQUIVOCATION_CAPACITY,
    CHECKPOINT_EVIDENCE_CAPACITY, CHECKPOINT_TRUSTED_DIVERGENCE_CAPACITY,
    MAX_CHECKPOINT_CERTIFICATE_SIGNERS, MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES,
};

/// Aggregate result of a complete local checkpoint-evidence vault audit.
///
/// Raw frames, peer identities, hashes, signatures, request IDs, and endpoints
/// are intentionally absent so this report is safe for startup logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordCommitmentCheckpointEvidenceAudit {
    /// Total cryptographically valid frames retained in the bounded vault.
    pub evidence_records: u64,
    /// Frames whose recorded local tip is covered by the audited local chain.
    pub applicable_evidence_records: u64,
    /// Historical frames above the audited local tip.
    pub deferred_evidence_records: u64,
    /// Applicable frames proving a shared-prefix mismatch.
    pub divergence_evidence_records: u64,
    /// Durable same-signer/same-height conflicting-hash incidents.
    pub equivocation_incidents: u64,
    /// Durable operator-pinned witness divergent-prefix incidents.
    pub trusted_divergence_incidents: u64,
    /// Immutable threshold certificates retained after complete re-audit.
    pub checkpoint_certificates: u64,
    /// Highest retained certified local checkpoint.
    pub latest_certified_height: Option<u64>,
    /// Distinct signed witness frames in the latest certificate.
    pub latest_certificate_signers: usize,
    /// Threshold recorded in the latest certificate.
    pub latest_certificate_required_signers: usize,
    /// Latest applicable observation; deferred frames cannot advance it.
    pub last_evidence_at: Option<u64>,
}

/// Result of atomically retaining one verified checkpoint frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentCheckpointEvidencePersistOutcome {
    /// No new trusted-witness conflict was established.
    Stored,
    /// A pinned witness supplied the first durable divergent-prefix proof.
    TrustedDivergenceDetected,
    /// The frame completed at least one durable signed equivocation proof.
    EquivocationDetected,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CheckpointEvidenceClaims {
    pub(super) responder: [u8; 32],
    pub(super) local_tip_height: u64,
    pub(super) checkpoint_height: u64,
    pub(super) checkpoint_hash: [u8; 32],
    pub(super) tip_height: u64,
    pub(super) tip_hash: [u8; 32],
    pub(super) applicable: bool,
    pub(super) diverged: bool,
    pub(super) certifiable: bool,
}

pub(super) fn decode_checkpoint_evidence_claims(
    frame: &[u8],
) -> Result<CheckpointEvidenceClaims, String> {
    if frame.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("checkpoint evidence frame violates bounds".to_string());
    }
    let message = decode_memchain(&frame[1..])
        .map_err(|_| "checkpoint evidence frame decode failed".to_string())?;
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        responder,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        ..
    } = message
    else {
        return Err("checkpoint evidence contains an unexpected frame".to_string());
    };
    Ok(CheckpointEvidenceClaims {
        responder,
        local_tip_height: 0,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        applicable: false,
        diverged: false,
        certifiable: false,
    })
}

pub(super) fn audit_checkpoint_evidence_connection(
    conn: &mut rusqlite::Connection,
) -> Result<RecordCommitmentCheckpointEvidenceAudit, String> {
    let transaction = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
        .map_err(|error| format!("begin checkpoint evidence audit snapshot: {error}"))?;
    let report = audit_checkpoint_evidence_snapshot(&transaction)?;
    transaction
        .commit()
        .map_err(|error| format!("finish checkpoint evidence audit snapshot: {error}"))?;
    Ok(report)
}

pub(super) fn audit_checkpoint_evidence_snapshot(
    connection: &rusqlite::Connection,
) -> Result<RecordCommitmentCheckpointEvidenceAudit, String> {
    let current_tip_height_i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(height), 0) FROM record_commitment_blocks",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| format!("read checkpoint evidence local tip: {error}"))?;
    let current_tip_height = u64::try_from(current_tip_height_i64)
        .map_err(|_| "checkpoint evidence local tip is invalid".to_string())?;

    let mut evidence_records = 0u64;
    let mut applicable_evidence_records = 0u64;
    let mut deferred_evidence_records = 0u64;
    let mut divergence_evidence_records = 0u64;
    let mut equivocation_incidents = 0u64;
    let mut trusted_divergence_incidents = 0u64;
    let mut last_evidence_at = None;
    let mut claims_by_digest = HashMap::with_capacity(CHECKPOINT_EVIDENCE_CAPACITY);
    {
        let mut statement = connection
            .prepare(
                "SELECT evidence_digest,observed_at,relation,local_tip_height,
                        remote_tip_height,checkpoint_height,signed_response
                 FROM record_checkpoint_evidence
                 ORDER BY observed_at ASC,evidence_digest ASC",
            )
            .map_err(|error| format!("prepare checkpoint evidence audit: {error}"))?;
        let mut rows = statement
            .query([])
            .map_err(|error| format!("query checkpoint evidence audit: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read checkpoint evidence row: {error}"))?
        {
            let digest: Vec<u8> = row
                .get(0)
                .map_err(|error| format!("read checkpoint evidence digest: {error}"))?;
            let observed_at_i64: i64 = row
                .get(1)
                .map_err(|error| format!("read checkpoint evidence time: {error}"))?;
            let relation: String = row
                .get(2)
                .map_err(|error| format!("read checkpoint evidence relation: {error}"))?;
            let local_tip_i64: i64 = row
                .get(3)
                .map_err(|error| format!("read checkpoint evidence local height: {error}"))?;
            let remote_tip_i64: i64 = row
                .get(4)
                .map_err(|error| format!("read checkpoint evidence remote height: {error}"))?;
            let stored_checkpoint_i64: i64 = row
                .get(5)
                .map_err(|error| format!("read checkpoint evidence height: {error}"))?;
            let frame: Vec<u8> = row
                .get(6)
                .map_err(|error| format!("read checkpoint evidence frame: {error}"))?;

            let observed_at = u64::try_from(observed_at_i64)
                .map_err(|_| "checkpoint evidence time is invalid".to_string())?;
            let local_tip_height = u64::try_from(local_tip_i64)
                .map_err(|_| "checkpoint evidence local height is invalid".to_string())?;
            let stored_remote_tip = u64::try_from(remote_tip_i64)
                .map_err(|_| "checkpoint evidence remote height is invalid".to_string())?;
            let stored_checkpoint_height = u64::try_from(stored_checkpoint_i64)
                .map_err(|_| "checkpoint evidence height is invalid".to_string())?;
            if !matches!(
                relation.as_str(),
                "converged" | "remote_ahead" | "remote_behind" | "diverged"
            ) {
                return Err("checkpoint evidence relation is invalid".to_string());
            }
            let digest_bytes: [u8; 32] = digest
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint evidence digest has invalid length".to_string())?;
            if Sha256::digest(&frame).as_slice() != digest.as_slice() {
                return Err("checkpoint evidence digest mismatch".to_string());
            }
            if frame.is_empty()
                || frame.len() > MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES
                || frame.first().copied() != Some(MEMCHAIN_MAGIC)
            {
                return Err("checkpoint evidence frame violates bounds".to_string());
            }
            let message = decode_memchain(&frame[1..])
                .map_err(|_| "checkpoint evidence frame decode failed".to_string())?;
            let canonical = encode_memchain(&message)
                .map_err(|_| "checkpoint evidence canonical encode failed".to_string())?;
            if canonical != frame {
                return Err("checkpoint evidence frame is non-canonical".to_string());
            }
            let MemChainMessage::RecordChainCheckpointResponseV1 {
                chain_id,
                request_id,
                responder,
                response_timestamp,
                checkpoint_height,
                checkpoint_hash,
                tip_height,
                tip_hash,
                signature,
            } = message
            else {
                return Err("checkpoint evidence contains an unexpected frame".to_string());
            };
            if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
                return Err("checkpoint evidence chain mismatch".to_string());
            }
            if observed_at.abs_diff(response_timestamp) > 60 {
                return Err("checkpoint evidence observation time mismatch".to_string());
            }
            let signing_bytes = record_chain_checkpoint_response_signing_bytes(
                &chain_id,
                &request_id,
                &responder,
                response_timestamp,
                checkpoint_height,
                &checkpoint_hash,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&responder)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| "checkpoint evidence signature is invalid".to_string())?;
            claims_by_digest.insert(
                digest_bytes,
                CheckpointEvidenceClaims {
                    responder,
                    local_tip_height,
                    checkpoint_height,
                    checkpoint_hash,
                    tip_height,
                    tip_hash,
                    applicable: local_tip_height <= current_tip_height,
                    diverged: relation == "diverged",
                    certifiable: matches!(relation.as_str(), "converged" | "remote_ahead")
                        && checkpoint_height == local_tip_height,
                },
            );
            if tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
                return Err("checkpoint evidence genesis tip is invalid".to_string());
            }
            if tip_height != stored_remote_tip
                || checkpoint_height != stored_checkpoint_height
                || checkpoint_height != local_tip_height.min(tip_height)
            {
                return Err("checkpoint evidence height relation mismatch".to_string());
            }
            if checkpoint_height == tip_height && checkpoint_hash != tip_hash {
                return Err("checkpoint evidence tip hash is inconsistent".to_string());
            }
            let relation_matches_heights = match relation.as_str() {
                "converged" => local_tip_height == tip_height,
                "remote_ahead" => local_tip_height < tip_height,
                "remote_behind" => local_tip_height > tip_height,
                // Divergence takes precedence over relative height once the
                // shared checkpoint hash can be compared to the local chain.
                "diverged" => true,
                _ => false,
            };
            if !relation_matches_heights {
                return Err("checkpoint evidence height classification mismatch".to_string());
            }

            evidence_records = evidence_records.saturating_add(1);
            if local_tip_height > current_tip_height {
                // A follower can retain signed observations from a previously
                // audited higher local tip after restoring an older database
                // snapshot. The frame remains cryptographically verifiable,
                // but its relation to the current chain cannot be established
                // until block sync restores that prefix. Keep it immutable and
                // bounded, but never let it refresh current operational state.
                deferred_evidence_records = deferred_evidence_records.saturating_add(1);
                continue;
            }

            let local_checkpoint_hash: Vec<u8> = if checkpoint_height == 0 {
                GENESIS_PREV_HASH.to_vec()
            } else {
                connection
                    .query_row(
                        "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                        params![i64::try_from(checkpoint_height).map_err(|_| {
                            "checkpoint evidence height exceeds SQLite range".to_string()
                        })?],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|error| format!("read checkpoint evidence local block: {error}"))?
                    .ok_or_else(|| "checkpoint evidence local block is unavailable".to_string())?
            };
            let expected_relation = if local_checkpoint_hash.as_slice() != checkpoint_hash {
                "diverged"
            } else if local_tip_height == tip_height {
                "converged"
            } else if local_tip_height < tip_height {
                "remote_ahead"
            } else {
                "remote_behind"
            };
            if relation != expected_relation {
                return Err("checkpoint evidence classification mismatch".to_string());
            }

            applicable_evidence_records = applicable_evidence_records.saturating_add(1);
            if relation == "diverged" {
                divergence_evidence_records = divergence_evidence_records.saturating_add(1);
            }
            last_evidence_at =
                Some(last_evidence_at.map_or(observed_at, |last: u64| last.max(observed_at)));
        }
    }
    if evidence_records > CHECKPOINT_EVIDENCE_CAPACITY as u64 {
        return Err("checkpoint evidence vault exceeds its configured capacity".to_string());
    }
    {
        let mut statement = connection
            .prepare(
                "SELECT responder,conflict_scope,conflict_height,
                        first_evidence_digest,second_evidence_digest,detected_at
                 FROM record_checkpoint_equivocations
                 ORDER BY detected_at ASC,responder ASC,conflict_scope ASC,conflict_height ASC",
            )
            .map_err(|error| format!("prepare checkpoint equivocation audit: {error}"))?;
        let mut rows = statement
            .query([])
            .map_err(|error| format!("query checkpoint equivocation audit: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read checkpoint equivocation row: {error}"))?
        {
            let responder: Vec<u8> = row
                .get(0)
                .map_err(|error| format!("read checkpoint equivocation responder: {error}"))?;
            let conflict_scope: String = row
                .get(1)
                .map_err(|error| format!("read checkpoint equivocation scope: {error}"))?;
            let conflict_height_i64: i64 = row
                .get(2)
                .map_err(|error| format!("read checkpoint equivocation height: {error}"))?;
            let first_digest: Vec<u8> = row
                .get(3)
                .map_err(|error| format!("read first checkpoint equivocation digest: {error}"))?;
            let second_digest: Vec<u8> = row
                .get(4)
                .map_err(|error| format!("read second checkpoint equivocation digest: {error}"))?;
            let detected_at_i64: i64 = row
                .get(5)
                .map_err(|error| format!("read checkpoint equivocation time: {error}"))?;
            let responder: [u8; 32] = responder
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint equivocation responder has invalid length".to_string())?;
            let conflict_height = u64::try_from(conflict_height_i64)
                .map_err(|_| "checkpoint equivocation height is invalid".to_string())?;
            let _detected_at = u64::try_from(detected_at_i64)
                .map_err(|_| "checkpoint equivocation time is invalid".to_string())?;
            let first_digest: [u8; 32] = first_digest.as_slice().try_into().map_err(|_| {
                "first checkpoint equivocation digest has invalid length".to_string()
            })?;
            let second_digest: [u8; 32] = second_digest.as_slice().try_into().map_err(|_| {
                "second checkpoint equivocation digest has invalid length".to_string()
            })?;
            if first_digest == second_digest {
                return Err("checkpoint equivocation references the same frame twice".to_string());
            }
            let first = claims_by_digest
                .get(&first_digest)
                .ok_or_else(|| "checkpoint equivocation first frame is unavailable".to_string())?;
            let second = claims_by_digest
                .get(&second_digest)
                .ok_or_else(|| "checkpoint equivocation second frame is unavailable".to_string())?;
            if first.responder != responder || second.responder != responder {
                return Err("checkpoint equivocation responder mismatch".to_string());
            }
            let valid_conflict = match conflict_scope.as_str() {
                "checkpoint" => {
                    first.checkpoint_height == conflict_height
                        && second.checkpoint_height == conflict_height
                        && first.checkpoint_hash != second.checkpoint_hash
                }
                "tip" => {
                    first.tip_height == conflict_height
                        && second.tip_height == conflict_height
                        && first.tip_hash != second.tip_hash
                }
                _ => false,
            };
            if !valid_conflict {
                return Err("checkpoint equivocation claim is invalid".to_string());
            }
            equivocation_incidents = equivocation_incidents.saturating_add(1);
        }
    }
    if equivocation_incidents > CHECKPOINT_EQUIVOCATION_CAPACITY as u64 {
        return Err("checkpoint equivocation vault exceeds its configured capacity".to_string());
    }
    {
        let mut statement = connection
            .prepare(
                "SELECT responder,evidence_digest,checkpoint_height,detected_at
                 FROM record_checkpoint_trusted_divergences
                 ORDER BY detected_at ASC,responder ASC",
            )
            .map_err(|error| format!("prepare trusted checkpoint divergence audit: {error}"))?;
        let mut rows = statement
            .query([])
            .map_err(|error| format!("query trusted checkpoint divergence audit: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read trusted checkpoint divergence row: {error}"))?
        {
            let responder: Vec<u8> = row
                .get(0)
                .map_err(|error| format!("read trusted divergence responder: {error}"))?;
            let evidence_digest: Vec<u8> = row
                .get(1)
                .map_err(|error| format!("read trusted divergence digest: {error}"))?;
            let checkpoint_height_i64: i64 = row
                .get(2)
                .map_err(|error| format!("read trusted divergence height: {error}"))?;
            let detected_at_i64: i64 = row
                .get(3)
                .map_err(|error| format!("read trusted divergence time: {error}"))?;
            let responder: [u8; 32] = responder
                .as_slice()
                .try_into()
                .map_err(|_| "trusted divergence responder has invalid length".to_string())?;
            let evidence_digest: [u8; 32] = evidence_digest
                .as_slice()
                .try_into()
                .map_err(|_| "trusted divergence digest has invalid length".to_string())?;
            let checkpoint_height = u64::try_from(checkpoint_height_i64)
                .map_err(|_| "trusted divergence height is invalid".to_string())?;
            let _detected_at = u64::try_from(detected_at_i64)
                .map_err(|_| "trusted divergence time is invalid".to_string())?;
            let claim = claims_by_digest
                .get(&evidence_digest)
                .ok_or_else(|| "trusted divergence evidence is unavailable".to_string())?;
            if claim.responder != responder
                || claim.checkpoint_height != checkpoint_height
                || !claim.diverged
                || !claim.applicable
            {
                return Err("trusted checkpoint divergence claim is invalid".to_string());
            }
            trusted_divergence_incidents = trusted_divergence_incidents.saturating_add(1);
        }
    }
    if trusted_divergence_incidents > CHECKPOINT_TRUSTED_DIVERGENCE_CAPACITY as u64 {
        return Err(
            "trusted checkpoint divergence vault exceeds its configured capacity".to_string(),
        );
    }
    let (
        checkpoint_certificates,
        latest_certified_height,
        latest_certificate_signers,
        latest_certificate_required_signers,
    ) = audit_checkpoint_certificates_snapshot(connection, &claims_by_digest)?;
    Ok(RecordCommitmentCheckpointEvidenceAudit {
        evidence_records,
        applicable_evidence_records,
        deferred_evidence_records,
        divergence_evidence_records,
        equivocation_incidents,
        trusted_divergence_incidents,
        checkpoint_certificates,
        latest_certified_height,
        latest_certificate_signers,
        latest_certificate_required_signers,
        last_evidence_at,
    })
}

/// Reconstructs every retained certificate from exact signed member frames.
/// A certificate is valid only when all members are distinct, applicable,
/// certifiable pinned-witness observations over the same local block.
pub(super) fn audit_checkpoint_certificates_snapshot(
    connection: &rusqlite::Connection,
    claims_by_digest: &HashMap<[u8; 32], CheckpointEvidenceClaims>,
) -> Result<(u64, Option<u64>, usize, usize), String> {
    type CertificateRow = (u64, [u8; 32], [u8; 32], usize, usize, [u8; 32]);
    let mut certificates: Vec<CertificateRow> = Vec::new();
    {
        let mut statement = connection
            .prepare(
                "SELECT checkpoint_height,chain_id,checkpoint_hash,required_signers,
                        signer_count,certificate_digest,certified_at
                 FROM record_checkpoint_certificates ORDER BY checkpoint_height ASC",
            )
            .map_err(|error| format!("prepare checkpoint certificate audit: {error}"))?;
        let mut rows = statement
            .query([])
            .map_err(|error| format!("query checkpoint certificate audit: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read checkpoint certificate row: {error}"))?
        {
            let height = u64::try_from(
                row.get::<_, i64>(0)
                    .map_err(|error| format!("read certificate height: {error}"))?,
            )
            .map_err(|_| "checkpoint certificate height is invalid".to_string())?;
            let chain_id: Vec<u8> = row
                .get(1)
                .map_err(|error| format!("read certificate chain id: {error}"))?;
            let checkpoint_hash: Vec<u8> = row
                .get(2)
                .map_err(|error| format!("read certificate checkpoint hash: {error}"))?;
            let required_signers = usize::try_from(
                row.get::<_, i64>(3)
                    .map_err(|error| format!("read certificate threshold: {error}"))?,
            )
            .map_err(|_| "checkpoint certificate threshold is invalid".to_string())?;
            let signer_count = usize::try_from(
                row.get::<_, i64>(4)
                    .map_err(|error| format!("read certificate signer count: {error}"))?,
            )
            .map_err(|_| "checkpoint certificate signer count is invalid".to_string())?;
            let certificate_digest: Vec<u8> = row
                .get(5)
                .map_err(|error| format!("read certificate digest: {error}"))?;
            let _certified_at = u64::try_from(
                row.get::<_, i64>(6)
                    .map_err(|error| format!("read certificate time: {error}"))?,
            )
            .map_err(|_| "checkpoint certificate time is invalid".to_string())?;
            let chain_id: [u8; 32] = chain_id
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate chain id has invalid length".to_string())?;
            let checkpoint_hash: [u8; 32] = checkpoint_hash
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate hash has invalid length".to_string())?;
            let certificate_digest: [u8; 32] = certificate_digest
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate digest has invalid length".to_string())?;
            if height == 0
                || chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
                || !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS).contains(&required_signers)
                || signer_count < required_signers
                || signer_count > MAX_CHECKPOINT_CERTIFICATE_SIGNERS
            {
                return Err("checkpoint certificate metadata is invalid".to_string());
            }
            certificates.push((
                height,
                chain_id,
                checkpoint_hash,
                required_signers,
                signer_count,
                certificate_digest,
            ));
        }
    }
    if certificates.len() > CHECKPOINT_CERTIFICATE_CAPACITY {
        return Err("checkpoint certificate vault exceeds its configured capacity".to_string());
    }

    let mut latest_height = None;
    let mut latest_signers = 0usize;
    let mut latest_required_signers = 0usize;
    for (height, chain_id, checkpoint_hash, required_signers, signer_count, stored_digest) in
        &certificates
    {
        let (local_chain_id, local_hash): (Vec<u8>, Vec<u8>) = connection
            .query_row(
                "SELECT chain_id,block_hash FROM record_commitment_blocks WHERE height=?1",
                params![i64::try_from(*height)
                    .map_err(|_| "checkpoint certificate height exceeds SQLite range")?],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| "checkpoint certificate local block is unavailable".to_string())?;
        if local_chain_id.as_slice() != chain_id || local_hash.as_slice() != checkpoint_hash {
            return Err("checkpoint certificate local block mismatch".to_string());
        }

        let mut members = Vec::with_capacity(*signer_count);
        let mut statement = connection
            .prepare(
                "SELECT responder,evidence_digest
                 FROM record_checkpoint_certificate_members
                 WHERE checkpoint_height=?1 ORDER BY responder ASC",
            )
            .map_err(|error| format!("prepare checkpoint certificate members: {error}"))?;
        let mut rows = statement
            .query(params![i64::try_from(*height).map_err(|_| {
                "checkpoint certificate height exceeds SQLite range"
            })?])
            .map_err(|error| format!("query checkpoint certificate members: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read checkpoint certificate member: {error}"))?
        {
            let responder: Vec<u8> = row
                .get(0)
                .map_err(|error| format!("read certificate responder: {error}"))?;
            let evidence_digest: Vec<u8> = row
                .get(1)
                .map_err(|error| format!("read certificate evidence digest: {error}"))?;
            let responder: [u8; 32] = responder
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate responder has invalid length".to_string())?;
            let evidence_digest: [u8; 32] =
                evidence_digest.as_slice().try_into().map_err(|_| {
                    "checkpoint certificate evidence digest has invalid length".to_string()
                })?;
            let claim = claims_by_digest
                .get(&evidence_digest)
                .ok_or_else(|| "checkpoint certificate evidence is unavailable".to_string())?;
            if claim.responder != responder
                || claim.local_tip_height != *height
                || claim.checkpoint_height != *height
                || claim.checkpoint_hash != *checkpoint_hash
                || !claim.applicable
                || !claim.certifiable
                || claim.diverged
            {
                return Err("checkpoint certificate member claim is invalid".to_string());
            }
            members.push((responder, evidence_digest));
        }
        if members.len() != *signer_count {
            return Err("checkpoint certificate signer count mismatch".to_string());
        }
        let computed = record_checkpoint_certificate_digest_v1(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            *height,
            checkpoint_hash,
            *required_signers,
            &members,
        );
        if computed != *stored_digest {
            return Err("checkpoint certificate digest mismatch".to_string());
        }
        latest_height = Some(*height);
        latest_signers = *signer_count;
        latest_required_signers = *required_signers;
    }

    Ok((
        certificates.len() as u64,
        latest_height,
        latest_signers,
        latest_required_signers,
    ))
}
pub(super) fn insert_checkpoint_equivocation_incident(
    connection: &rusqlite::Connection,
    responder: &[u8; 32],
    conflict_scope: &str,
    conflict_height: u64,
    first_evidence_digest: &[u8; 32],
    second_evidence_digest: &[u8; 32],
    detected_at: u64,
) -> Result<usize, String> {
    let conflict_height = i64::try_from(conflict_height)
        .map_err(|_| "checkpoint equivocation height exceeds SQLite range".to_string())?;
    let detected_at = i64::try_from(detected_at)
        .map_err(|_| "checkpoint equivocation time exceeds SQLite range".to_string())?;
    connection
        .execute(
            "INSERT OR IGNORE INTO record_checkpoint_equivocations
             (responder,conflict_scope,conflict_height,first_evidence_digest,
              second_evidence_digest,detected_at)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![
                responder.as_slice(),
                conflict_scope,
                conflict_height,
                first_evidence_digest.as_slice(),
                second_evidence_digest.as_slice(),
                detected_at,
            ],
        )
        .map_err(|error| format!("insert checkpoint equivocation incident: {error}"))
}

/// Detects only conflicts from a caller-designated trusted witness.
///
/// Permissionless observations remain useful signed evidence but cannot fill
/// the permanent incident ledger or become startup authority. The caller must
/// set this policy only after matching the responder against explicit operator
/// pins. Both conflicting frames are already in the same SQLite transaction.
pub(super) fn detect_trusted_checkpoint_equivocations(
    connection: &rusqlite::Connection,
    new_evidence_digest: &[u8; 32],
    signed_response: &[u8],
    detected_at: u64,
) -> Result<usize, String> {
    let new_claim = decode_checkpoint_evidence_claims(signed_response)?;
    let mut inserted = 0usize;
    let mut statement = connection
        .prepare(
            "SELECT evidence_digest,signed_response
             FROM record_checkpoint_evidence
             WHERE evidence_digest != ?1
             ORDER BY observed_at ASC,evidence_digest ASC",
        )
        .map_err(|error| format!("prepare trusted checkpoint conflict scan: {error}"))?;
    let mut rows = statement
        .query(params![new_evidence_digest.as_slice()])
        .map_err(|error| format!("query trusted checkpoint conflict scan: {error}"))?;
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("read trusted checkpoint conflict row: {error}"))?
    {
        let old_digest: Vec<u8> = row
            .get(0)
            .map_err(|error| format!("read trusted checkpoint conflict digest: {error}"))?;
        let old_frame: Vec<u8> = row
            .get(1)
            .map_err(|error| format!("read trusted checkpoint conflict frame: {error}"))?;
        let old_digest: [u8; 32] = old_digest
            .as_slice()
            .try_into()
            .map_err(|_| "trusted checkpoint conflict digest has invalid length".to_string())?;
        let old_claim = decode_checkpoint_evidence_claims(&old_frame)?;
        if old_claim.responder != new_claim.responder {
            continue;
        }
        let checkpoint_conflict = old_claim.checkpoint_height == new_claim.checkpoint_height
            && old_claim.checkpoint_hash != new_claim.checkpoint_hash;
        if checkpoint_conflict {
            inserted = inserted.saturating_add(insert_checkpoint_equivocation_incident(
                connection,
                &new_claim.responder,
                "checkpoint",
                new_claim.checkpoint_height,
                &old_digest,
                new_evidence_digest,
                detected_at,
            )?);
        }
        let duplicate_tip_claim = checkpoint_conflict
            && old_claim.checkpoint_height == old_claim.tip_height
            && new_claim.checkpoint_height == new_claim.tip_height;
        if !duplicate_tip_claim
            && old_claim.tip_height == new_claim.tip_height
            && old_claim.tip_hash != new_claim.tip_hash
        {
            inserted = inserted.saturating_add(insert_checkpoint_equivocation_incident(
                connection,
                &new_claim.responder,
                "tip",
                new_claim.tip_height,
                &old_digest,
                new_evidence_digest,
                detected_at,
            )?);
        }
    }
    let incident_count = connection
        .query_row(
            "SELECT COUNT(*) FROM record_checkpoint_equivocations",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| format!("count checkpoint equivocation incidents: {error}"))?;
    if incident_count > CHECKPOINT_EQUIVOCATION_CAPACITY as i64 {
        return Err("checkpoint equivocation vault exceeds its configured capacity".to_string());
    }
    Ok(inserted)
}

/// Retains the first verified divergent-prefix proof for one operator-pinned
/// witness. A later converged frame cannot replace or delete this incident.
pub(super) fn insert_trusted_checkpoint_divergence_incident(
    connection: &rusqlite::Connection,
    evidence_digest: &[u8; 32],
    signed_response: &[u8],
    checkpoint_height: u64,
    detected_at: u64,
) -> Result<usize, String> {
    let claim = decode_checkpoint_evidence_claims(signed_response)?;
    if claim.checkpoint_height != checkpoint_height {
        return Err("trusted divergence checkpoint height mismatch".to_string());
    }
    let checkpoint_height = i64::try_from(checkpoint_height)
        .map_err(|_| "trusted divergence height exceeds SQLite range".to_string())?;
    let detected_at = i64::try_from(detected_at)
        .map_err(|_| "trusted divergence time exceeds SQLite range".to_string())?;
    let inserted = connection
        .execute(
            "INSERT OR IGNORE INTO record_checkpoint_trusted_divergences
             (responder,evidence_digest,checkpoint_height,detected_at)
             VALUES (?1,?2,?3,?4)",
            params![
                claim.responder.as_slice(),
                evidence_digest.as_slice(),
                checkpoint_height,
                detected_at,
            ],
        )
        .map_err(|error| format!("insert trusted checkpoint divergence incident: {error}"))?;
    let incident_count = connection
        .query_row(
            "SELECT COUNT(*) FROM record_checkpoint_trusted_divergences",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| format!("count trusted checkpoint divergence incidents: {error}"))?;
    if incident_count > CHECKPOINT_TRUSTED_DIVERGENCE_CAPACITY as i64 {
        return Err(
            "trusted checkpoint divergence vault exceeds its configured capacity".to_string(),
        );
    }
    Ok(inserted)
}
