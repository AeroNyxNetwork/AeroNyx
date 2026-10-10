// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/quarantine.rs
// ============================================
//! # Directory Replica quarantine inspection and resolution
//!
//! Owns host-local incident inspection, node-identity-signed compare-and-swap
//! quarantine resolution, and opening/auditing the local replica store.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::PathBuf;

use anyhow::Context;
use rand::RngCore;
use tracing::info;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::services::{
    DirectoryReplicaResolutionCommand, DirectoryReplicaStore, DirectoryReplicaTip,
};
use aeronyx_server::ServerConfig;

use crate::commands::helpers::{load_node_identity, parse_hex32, unix_timestamp};

#[derive(Debug)]
pub(super) struct DirectoryReplicaResolveRequest {
    pub(super) digest: String,
    pub(super) producer: String,
    pub(super) expected_tip_height: u64,
    pub(super) expected_tip_hash: String,
    pub(super) expected_kind: String,
    pub(super) expected_previous_resolution_digest: Option<String>,
    pub(super) confirm_incident: String,
}

pub(super) async fn cmd_directory_replica_inspect(
    config_path: &PathBuf,
    digest_hex: &str,
) -> anyhow::Result<()> {
    let digest = parse_hex32(digest_hex, "incident digest")?;
    let (store, _identity) = open_directory_replica_store(config_path).await?;
    let evidence = store
        .incident_evidence(&digest)?
        .with_context(|| format!("incident {} was not found", hex::encode(digest)))?;
    let tip = store.producer_tip(&evidence.summary.producer)?;
    print_directory_replica_incident(&evidence, &tip);
    Ok(())
}

pub(super) async fn cmd_directory_replica_resolve(
    config_path: &PathBuf,
    request: &DirectoryReplicaResolveRequest,
) -> anyhow::Result<()> {
    let digest = parse_hex32(&request.digest, "incident digest")?;
    let confirmation = parse_hex32(&request.confirm_incident, "confirmed incident digest")?;
    anyhow::ensure!(
        confirmation == digest,
        "--confirm-incident must exactly repeat --digest"
    );
    let producer = parse_hex32(&request.producer, "producer identity")?;
    let expected_tip_hash = parse_hex32(&request.expected_tip_hash, "expected tip hash")?;
    let previous_resolution_digest = request
        .expected_previous_resolution_digest
        .as_deref()
        .map(|value| parse_hex32(value, "previous resolution digest"))
        .transpose()?;
    let (store, identity) = open_directory_replica_store(config_path).await?;
    let evidence = store
        .incident_evidence(&digest)?
        .with_context(|| format!("incident {} was not found", hex::encode(digest)))?;
    let tip = store.producer_tip(&producer)?;
    validate_resolution_request(
        request,
        digest,
        producer,
        expected_tip_hash,
        &evidence,
        &tip,
    )?;
    anyhow::ensure!(
        tip.last_resolution_digest == previous_resolution_digest,
        "previous resolution digest changed; inspect the incident again"
    );

    let mut command_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut command_id);
    let now = unix_timestamp()?;
    let command = DirectoryReplicaResolutionCommand::sign(
        &identity,
        command_id,
        digest,
        producer,
        request.expected_tip_height,
        expected_tip_hash,
        request.expected_kind.clone(),
        previous_resolution_digest,
        now,
    )?;
    let report = store.resolve_quarantine(&command, now)?;
    println!("Directory Replica quarantine resolved");
    println!(
        "  resolution_digest: {}",
        hex::encode(report.resolution_digest)
    );
    println!("  command_id: {}", hex::encode(report.command_id));
    println!("  producer: {}", hex::encode(report.producer));
    println!("  retained_tip_height: {}", report.retained_tip_height);
    println!(
        "  retained_tip_hash: {}",
        hex::encode(report.retained_tip_hash)
    );
    println!("  resolved_at: {}", report.resolved_at);
    println!("  action: resume_existing_prefix");
    Ok(())
}

fn validate_resolution_request(
    request: &DirectoryReplicaResolveRequest,
    digest: [u8; 32],
    producer: [u8; 32],
    expected_tip_hash: [u8; 32],
    evidence: &aeronyx_server::services::DirectoryReplicaIncidentEvidence,
    tip: &DirectoryReplicaTip,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        evidence.summary.incident_digest == digest,
        "incident digest changed"
    );
    anyhow::ensure!(
        evidence.summary.producer == producer,
        "incident producer mismatch"
    );
    anyhow::ensure!(
        evidence.summary.subject_node_id == producer,
        "incident is not a producer quarantine"
    );
    anyhow::ensure!(tip.quarantined, "producer is not quarantined");
    anyhow::ensure!(
        tip.active_incident_digest == Some(digest),
        "incident is not the active quarantine"
    );
    anyhow::ensure!(
        tip.tip_height == request.expected_tip_height,
        "accepted tip height changed"
    );
    anyhow::ensure!(
        tip.tip_hash == expected_tip_hash,
        "accepted tip hash changed"
    );
    anyhow::ensure!(
        tip.quarantine_kind.as_deref() == Some(request.expected_kind.as_str()),
        "quarantine kind changed"
    );
    Ok(())
}

fn print_directory_replica_incident(
    evidence: &aeronyx_server::services::DirectoryReplicaIncidentEvidence,
    tip: &DirectoryReplicaTip,
) {
    println!("Directory Replica incident (verified, read-only)");
    println!(
        "  incident_digest: {}",
        hex::encode(evidence.summary.incident_digest)
    );
    println!("  producer: {}", hex::encode(evidence.summary.producer));
    println!("  kind: {}", evidence.summary.kind);
    println!("  incident_height: {}", evidence.summary.height);
    println!("  local_hash: {}", hex::encode(evidence.summary.local_hash));
    println!(
        "  remote_hash: {}",
        hex::encode(evidence.summary.remote_hash)
    );
    println!(
        "  evidence_sha256: {}",
        hex::encode(evidence.evidence_sha256)
    );
    println!("  observed_at: {}", evidence.summary.observed_at);
    println!("  quarantined: {}", tip.quarantined);
    println!("  accepted_tip_height: {}", tip.tip_height);
    println!("  accepted_tip_hash: {}", hex::encode(tip.tip_hash));
    println!(
        "  previous_resolution_digest: {}",
        tip.last_resolution_digest
            .map_or_else(|| "none".to_string(), hex::encode)
    );
    println!("No block, incident, or evidence was modified.");
    if tip.quarantined
        && tip.active_incident_digest == Some(evidence.summary.incident_digest)
        && evidence.summary.subject_node_id == evidence.summary.producer
    {
        println!("Exact resolution command after independent evidence review:");
        print!(
            "  aeronyx-server directory-replica resolve-quarantine --digest {} \
--producer {} --expected-tip-height {} --expected-tip-hash {} \
--expected-kind {}",
            hex::encode(evidence.summary.incident_digest),
            hex::encode(evidence.summary.producer),
            tip.tip_height,
            hex::encode(tip.tip_hash),
            evidence.summary.kind,
        );
        if let Some(previous) = tip.last_resolution_digest {
            print!(
                " --expected-previous-resolution-digest {}",
                hex::encode(previous)
            );
        }
        println!(
            " --confirm-incident {}",
            hex::encode(evidence.summary.incident_digest)
        );
    } else {
        println!("Resolution command unavailable: this is not the active producer quarantine.");
    }
}

pub(super) async fn open_directory_replica_store(
    config_path: &PathBuf,
) -> anyhow::Result<(DirectoryReplicaStore, IdentityKeyPair)> {
    anyhow::ensure!(
        config_path.exists(),
        "configuration file not found: {}",
        config_path.display()
    );
    let config = ServerConfig::load(config_path).await?;
    let database_path = config
        .discovery
        .directory_chain_path
        .as_deref()
        .context("discovery.directory_chain_path is not configured")?;
    let identity = load_node_identity(&config).await?;
    let now = unix_timestamp()?;
    let (store, audit) =
        DirectoryReplicaStore::open(database_path, identity.public_key_bytes(), now)?;
    info!(
        producers = audit.producers,
        quarantined_producers = audit.quarantined_producers,
        blocks = audit.blocks,
        incidents = audit.incidents,
        resolutions = audit.resolutions,
        "host-local Directory Replica audit passed"
    );
    Ok((store, identity))
}

#[cfg(test)]
mod tests;
