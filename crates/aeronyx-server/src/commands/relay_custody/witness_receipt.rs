// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/witness_receipt.rs
// ============================================
//! # Relay custody witness receipts
//!
//! Owns the witness side of custody audit anchors: durable independent-node
//! countersigning, exact offline receipt verification, and the receipt report
//! with its decision/outcome labels.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::Path;

use sha2::{Digest, Sha256};

use aeronyx_core::protocol::chat::{
    decode_custody_audit_witness_receipt, encode_custody_audit_witness_receipt,
    CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1, CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
    CUSTODY_AUDIT_WITNESS_CONFLICT_V1, CUSTODY_AUDIT_WITNESS_GAP_V1,
    CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1, CUSTODY_AUDIT_WITNESS_STALE_V1,
    MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES, MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
};
use aeronyx_server::services::memchain::{
    derive_record_key, CustodyAuditAnchorWitnessOutcome, MemoryStorage,
};
use aeronyx_server::ServerConfig;

use super::audit_anchor::verify_relay_custody_anchor_frame;
use super::host_io::{
    load_relay_custody_identity, read_bounded_relay_custody_artifact,
    write_new_relay_custody_artifact,
};
use crate::commands::helpers::{parse_hex32, unix_timestamp_now};

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessReport {
    contract_version: &'static str,
    status: &'static str,
    accepted: bool,
    outcome: &'static str,
    producer_node_id: String,
    witness_node_id: String,
    checkpoint_generation: u64,
    observed_at: u64,
    retained_checkpoint_generation: u64,
    anchor_frame_sha256: String,
    retained_frame_sha256: String,
    receipt_sha256: String,
    receipt_bytes: usize,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn cmd_relay_witness_audit_anchor(
    config_path: &Path,
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_producer_hex: &str,
    minimum_checkpoint_generation: u64,
    output_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let config = ServerConfig::load(config_path).await?;
    anyhow::ensure!(
        config.memchain.is_enabled() && config.memchain.db_path.trim() != ":memory:",
        "custody audit witnessing requires persistent local MemChain storage"
    );
    let witness_identity = load_relay_custody_identity(&config, "audit witness").await?;
    let expected_sha256 = parse_hex32(expected_sha256_hex, "audit anchor SHA-256")?;
    let expected_producer = parse_hex32(expected_producer_hex, "expected producer node identity")?;
    anyhow::ensure!(
        witness_identity.public_key_bytes() != expected_producer,
        "custody audit anchor must be witnessed by an independent node identity"
    );

    let anchor_frame = read_bounded_relay_custody_artifact(
        input_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor = verify_relay_custody_anchor_frame(
        &anchor_frame,
        &expected_sha256,
        &expected_producer,
        minimum_checkpoint_generation,
    )?;
    let observed_at = unix_timestamp_now()?;
    anyhow::ensure!(observed_at > 0, "custody audit witness clock is invalid");

    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Persist the witness
    // high-water decision before signing or publishing its receipt. A failed
    // output write is safely retryable as an idempotent observation.
    let record_key = derive_record_key(&witness_identity.to_bytes());
    let storage = MemoryStorage::open(&config.memchain.db_path, Some(record_key))
        .map_err(|_| anyhow::anyhow!("unable to open custody audit witness storage"))?;
    let decision = storage
        .witness_custody_audit_anchor(
            &expected_producer,
            anchor.checkpoint_generation,
            &expected_sha256,
            observed_at,
        )
        .await
        .map_err(|_| anyhow::anyhow!("unable to persist custody audit witness decision"))?;
    let (outcome, retained_generation, retained_sha256) =
        custody_audit_witness_decision_fields(decision);
    let receipt = CustodyAuditWitnessReceiptV1::signed(
        expected_producer,
        anchor.checkpoint_generation,
        expected_sha256,
        observed_at,
        retained_generation,
        retained_sha256,
        outcome,
        &witness_identity,
    )
    .map_err(|_| anyhow::anyhow!("unable to sign custody audit witness receipt"))?;
    let receipt_frame = encode_custody_audit_witness_receipt(&receipt)
        .map_err(|_| anyhow::anyhow!("unable to encode custody audit witness receipt"))?;
    let receipt_sha256: [u8; 32] = Sha256::digest(&receipt_frame).into();
    write_new_relay_custody_artifact(
        output_path,
        &receipt_frame,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )?;
    print_relay_custody_audit_witness(
        &receipt,
        &receipt_sha256,
        receipt_frame.len(),
        "created",
        json,
    )?;
    anyhow::ensure!(
        receipt.accepted(),
        "custody audit witness rejected the producer anchor; retain the signed negative receipt"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cmd_relay_verify_audit_witness(
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_producer_hex: &str,
    expected_witness_hex: &str,
    minimum_checkpoint_generation: u64,
    json: bool,
) -> anyhow::Result<()> {
    let anchor_sha256 = parse_hex32(anchor_sha256_hex, "audit anchor SHA-256")?;
    let receipt_sha256 = parse_hex32(receipt_sha256_hex, "audit witness receipt SHA-256")?;
    let expected_producer = parse_hex32(expected_producer_hex, "expected producer node identity")?;
    let expected_witness = parse_hex32(expected_witness_hex, "expected witness node identity")?;
    anyhow::ensure!(
        expected_producer != expected_witness,
        "producer and independent witness identities must differ"
    );

    let anchor_frame = read_bounded_relay_custody_artifact(
        anchor_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor = verify_relay_custody_anchor_frame(
        &anchor_frame,
        &anchor_sha256,
        &expected_producer,
        minimum_checkpoint_generation,
    )?;
    let receipt_frame = read_bounded_relay_custody_artifact(
        receipt_path,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )?;
    let receipt = verify_relay_custody_witness_receipt_frame(
        &receipt_frame,
        &receipt_sha256,
        &anchor,
        &anchor_sha256,
        &expected_producer,
        &expected_witness,
        minimum_checkpoint_generation,
    )?;
    anyhow::ensure!(
        receipt.accepted(),
        "custody audit witness receipt does not prove accepted custody"
    );
    print_relay_custody_audit_witness(
        &receipt,
        &receipt_sha256,
        receipt_frame.len(),
        "verified",
        json,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_relay_custody_witness_receipt_frame(
    receipt_frame: &[u8],
    expected_receipt_sha256: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
    anchor_sha256: &[u8; 32],
    expected_producer: &[u8; 32],
    expected_witness: &[u8; 32],
    minimum_checkpoint_generation: u64,
) -> anyhow::Result<CustodyAuditWitnessReceiptV1> {
    anyhow::ensure!(
        !receipt_frame.is_empty()
            && receipt_frame.len() <= MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "custody audit witness receipt violates its complete-frame bound"
    );
    let actual_receipt_sha256: [u8; 32] = Sha256::digest(receipt_frame).into();
    anyhow::ensure!(
        &actual_receipt_sha256 == expected_receipt_sha256,
        "custody audit witness receipt SHA-256 does not match the explicit pin"
    );
    let receipt = decode_custody_audit_witness_receipt(receipt_frame)
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt is malformed"))?;
    let canonical = encode_custody_audit_witness_receipt(&receipt)
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt cannot be canonicalized"))?;
    anyhow::ensure!(
        canonical == receipt_frame,
        "custody audit witness receipt is not canonically encoded"
    );
    receipt
        .verify_for_anchor(
            anchor,
            anchor_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
        )
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt trust policy failed"))?;
    Ok(receipt)
}

const fn custody_audit_witness_decision_fields(
    decision: CustodyAuditAnchorWitnessOutcome,
) -> (u8, u64, [u8; 32]) {
    match decision {
        CustodyAuditAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_ADVANCED_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        } => (
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        CustodyAuditAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_STALE_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_CONFLICT_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_GAP_V1, generation, anchor_digest),
    }
}

pub(super) const fn custody_audit_witness_outcome_label(outcome: u8) -> &'static str {
    match outcome {
        CUSTODY_AUDIT_WITNESS_ADVANCED_V1 => "advanced",
        CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1 => "idempotent",
        CUSTODY_AUDIT_WITNESS_STALE_V1 => "stale",
        CUSTODY_AUDIT_WITNESS_CONFLICT_V1 => "conflict",
        CUSTODY_AUDIT_WITNESS_GAP_V1 => "gap",
        _ => "invalid",
    }
}

fn print_relay_custody_audit_witness(
    receipt: &CustodyAuditWitnessReceiptV1,
    receipt_sha256: &[u8; 32],
    receipt_bytes: usize,
    status: &'static str,
    json: bool,
) -> anyhow::Result<()> {
    let report = RelayCustodyAuditWitnessReport {
        contract_version: "relay_custody_audit_witness.v1",
        status,
        accepted: receipt.accepted(),
        outcome: custody_audit_witness_outcome_label(receipt.outcome),
        producer_node_id: hex::encode(receipt.producer_node_id),
        witness_node_id: hex::encode(receipt.witness_node_id),
        checkpoint_generation: receipt.requested_checkpoint_generation,
        observed_at: receipt.observed_at,
        retained_checkpoint_generation: receipt.retained_checkpoint_generation,
        anchor_frame_sha256: hex::encode(receipt.requested_frame_sha256),
        retained_frame_sha256: hex::encode(receipt.retained_frame_sha256),
        receipt_sha256: hex::encode(receipt_sha256),
        receipt_bytes,
        security_model: "independent node signature over a durable producer-scoped monotonic decision; accepted receipts detect later rollback only while the witness high-water state and verifier pins remain available; not consensus or global finality",
        privacy_boundary: "producer and witness node identities, checkpoint generation, exact opaque frame digests, witness time, and coarse outcome only; no private HMAC, custody path, message, route, endpoint, payload, ciphertext, memory, destination, DNS, or social graph metadata",
    };
    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("Relay custody independent witness receipt");
        println!("════════════════════════════════════════");
        println!("Status:               {}", report.status);
        println!("Accepted:             {}", report.accepted);
        println!("Outcome:              {}", report.outcome);
        println!("Producer node:        {}", report.producer_node_id);
        println!("Witness node:         {}", report.witness_node_id);
        println!("Checkpoint generation: {}", report.checkpoint_generation);
        println!("Witness observed at:  {}", report.observed_at);
        println!(
            "Retained generation:   {}",
            report.retained_checkpoint_generation
        );
        println!("Anchor frame SHA-256: {}", report.anchor_frame_sha256);
        println!("Receipt SHA-256:      {}", report.receipt_sha256);
        println!("Receipt bytes:        {}", report.receipt_bytes);
        println!();
        println!("Security model: {}", report.security_model);
        println!("Privacy: {}", report.privacy_boundary);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
