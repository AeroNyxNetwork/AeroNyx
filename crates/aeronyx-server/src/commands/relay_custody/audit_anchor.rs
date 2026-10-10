// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/audit_anchor.rs
// ============================================
//! # Relay custody audit anchors
//!
//! Owns create-new export and fail-closed offline verification of exact
//! node-signed custody audit anchor frames, and their aggregate report.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::Path;

use sha2::{Digest, Sha256};

use aeronyx_core::protocol::chat::{
    decode_custody_audit_anchor, encode_custody_audit_anchor, CustodyAuditAnchorV1,
    MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
};
use aeronyx_server::services::ChatRelayService;

use super::host_io::{
    load_relay_custody_config, load_relay_custody_identity, read_bounded_relay_custody_anchor,
    write_new_relay_custody_anchor,
};
use crate::commands::helpers::parse_hex32;

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditAnchorReport {
    contract_version: &'static str,
    status: &'static str,
    protocol_version: u8,
    producer_node_id: String,
    checkpoint_generation: u64,
    archived_record_count: u64,
    archived_bytes: u64,
    anchor_digest: String,
    frame_sha256: String,
    frame_bytes: usize,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

// [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Export remains host-local and
// create-new. The exact frame digest is intended for a separately administered
// retainer; the producer must not silently replace evidence already retained.
pub(super) async fn cmd_relay_create_audit_anchor(
    config_path: &Path,
    output_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let identity = load_relay_custody_identity(&server_config, "audit anchor export").await?;
    let anchor = ChatRelayService::create_backup_maintenance_audit_anchor_for_config(
        &server_config.memchain.chat_relay,
        &identity,
    )
    .map_err(|error| anyhow::anyhow!("relay custody audit anchor export failed: {error}"))?;
    let frame = encode_custody_audit_anchor(&anchor)
        .map_err(|_| anyhow::anyhow!("unable to encode relay custody audit anchor"))?;
    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "encoded relay custody audit anchor violates its protocol bound"
    );
    let frame_sha256: [u8; 32] = Sha256::digest(&frame).into();
    write_new_relay_custody_anchor(output_path, &frame)?;
    print_relay_custody_audit_anchor(&anchor, &frame_sha256, frame.len(), "created", json)
}

pub(super) fn cmd_relay_verify_audit_anchor(
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_node_hex: &str,
    minimum_checkpoint_generation: u64,
    json: bool,
) -> anyhow::Result<()> {
    let expected_sha256 = parse_hex32(expected_sha256_hex, "audit anchor SHA-256")?;
    let expected_node = parse_hex32(expected_node_hex, "expected producer node identity")?;
    let frame = read_bounded_relay_custody_anchor(input_path)?;
    let anchor = verify_relay_custody_anchor_frame(
        &frame,
        &expected_sha256,
        &expected_node,
        minimum_checkpoint_generation,
    )?;
    print_relay_custody_audit_anchor(&anchor, &expected_sha256, frame.len(), "verified", json)
}

pub(super) fn verify_relay_custody_anchor_frame(
    frame: &[u8],
    expected_sha256: &[u8; 32],
    expected_node: &[u8; 32],
    minimum_checkpoint_generation: u64,
) -> anyhow::Result<CustodyAuditAnchorV1> {
    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "relay custody audit anchor violates its complete-frame bound"
    );
    let actual_sha256: [u8; 32] = Sha256::digest(frame).into();
    anyhow::ensure!(
        &actual_sha256 == expected_sha256,
        "relay custody audit anchor SHA-256 does not match the explicit pin"
    );
    let anchor = decode_custody_audit_anchor(frame)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor is malformed"))?;
    let canonical = encode_custody_audit_anchor(&anchor)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor cannot be canonicalized"))?;
    anyhow::ensure!(
        canonical == frame,
        "relay custody audit anchor is not canonically encoded"
    );
    anchor
        .verify_expected(expected_node, minimum_checkpoint_generation)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor trust policy failed"))?;
    Ok(anchor)
}

fn print_relay_custody_audit_anchor(
    anchor: &CustodyAuditAnchorV1,
    frame_sha256: &[u8; 32],
    frame_bytes: usize,
    status: &'static str,
    json: bool,
) -> anyhow::Result<()> {
    let report = RelayCustodyAuditAnchorReport {
        contract_version: "relay_custody_audit_anchor.v1",
        status,
        protocol_version: anchor.version,
        producer_node_id: hex::encode(anchor.producer_node_id),
        checkpoint_generation: anchor.checkpoint_generation,
        archived_record_count: anchor.archived_record_count,
        archived_bytes: anchor.archived_bytes,
        anchor_digest: hex::encode(anchor.anchor_digest),
        frame_sha256: hex::encode(frame_sha256),
        frame_bytes,
        security_model: "producer-signed opaque checkpoint commitment with explicit identity, exact-frame digest, and verifier-owned rollback floor; not an independent witness receipt, validator vote, consensus, or global finality",
        privacy_boundary: "checkpoint generation and aggregate archived record/byte counts only; no private HMAC, path, operation id, message id, endpoint, route, identity owner, payload, ciphertext, memory, destination, DNS, or social graph metadata",
    };
    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("Relay custody audit anchor");
        println!("════════════════════════════════════════");
        println!("Status:               {}", report.status);
        println!("Producer node:        {}", report.producer_node_id);
        println!("Checkpoint generation: {}", report.checkpoint_generation);
        println!("Archived records:     {}", report.archived_record_count);
        println!("Archived bytes:       {}", report.archived_bytes);
        println!("Anchor digest:        {}", report.anchor_digest);
        println!("Frame SHA-256:        {}", report.frame_sha256);
        println!("Frame bytes:          {}", report.frame_bytes);
        println!();
        println!("Security model: {}", report.security_model);
        println!("Privacy: {}", report.privacy_boundary);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
