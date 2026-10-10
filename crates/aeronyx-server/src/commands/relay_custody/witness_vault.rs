// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/witness_vault.rs
// ============================================
//! # Relay custody producer witness vault
//!
//! Owns the producer side of witness evidence: bounded signed-receipt import,
//! the local current-anchor vault audit with quorum expiry/renewal reporting,
//! and explicit snapshot-pinned operator collection.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::Path;

use sha2::{Digest, Sha256};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::{
    encode_custody_audit_anchor, CustodyAuditWitnessReceiptV1,
    MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES, MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
};
use aeronyx_core::protocol::discovery::NodeBootstrapSnapshot;
use aeronyx_server::api::memchain_peer::{
    witness_custody_audit_anchor_round_durable, CustodyAuditWitnessRound,
};
use aeronyx_server::services::chat_relay::ChatRelayCustodyAuditAnchorGuard;
use aeronyx_server::services::memchain::{
    custody_witness_renewal_warning_window_secs, derive_record_key,
    CustodyAuditWitnessReceiptPolicyEvidence, MemoryStorage,
};
use aeronyx_server::services::{ChatRelayService, PeerStore};
use aeronyx_server::ServerConfig;

use super::audit_anchor::verify_relay_custody_anchor_frame;
use super::host_io::{
    load_relay_custody_config, load_relay_custody_identity, read_bounded_relay_custody_artifact,
};
use super::witness_receipt::{
    custody_audit_witness_outcome_label, verify_relay_custody_witness_receipt_frame,
};
use crate::commands::helpers::{parse_hex32, unix_timestamp_now};

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessImportReport {
    contract_version: &'static str,
    status: &'static str,
    import_disposition: &'static str,
    receipt_outcome: &'static str,
    checkpoint_generation: u64,
    observed_at: u64,
    max_age_seconds: u64,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    configured_witnesses: usize,
    fresh_verified: usize,
    accepted: usize,
    adverse: usize,
    missing: usize,
    minimum_verified: usize,
    policy_ready: bool,
    quorum_valid_through: Option<u64>,
    quorum_valid_for_seconds: Option<u64>,
    renewal_warning_window_seconds: u64,
    renewal_recommended: bool,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessVaultReport {
    contract_version: &'static str,
    status: &'static str,
    evaluated_at: u64,
    checkpoint_generation: u64,
    max_age_seconds: u64,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    configured_witnesses: usize,
    fresh_verified: usize,
    accepted: usize,
    adverse: usize,
    missing: usize,
    minimum_verified: usize,
    policy_ready: bool,
    required_ready: bool,
    quorum_valid_through: Option<u64>,
    quorum_valid_for_seconds: Option<u64>,
    renewal_warning_window_seconds: u64,
    renewal_recommended: bool,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessCollectionReport {
    contract_version: &'static str,
    status: &'static str,
    evaluated_at: u64,
    checkpoint_generation: u64,
    timeout_seconds: u64,
    max_age_seconds: u64,
    snapshot_records: usize,
    snapshot_pinned_records: usize,
    snapshot_verified_records: usize,
    snapshot_rejected_records: usize,
    round_configured: usize,
    round_verified: usize,
    round_accepted: usize,
    round_advanced: usize,
    round_idempotent: usize,
    round_stale: usize,
    round_conflicts: usize,
    round_gaps: usize,
    round_failed: usize,
    round_adverse: bool,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    configured_witnesses: usize,
    fresh_verified: usize,
    accepted: usize,
    adverse: usize,
    missing: usize,
    minimum_verified: usize,
    policy_ready: bool,
    quorum_valid_through: Option<u64>,
    quorum_valid_for_seconds: Option<u64>,
    renewal_warning_window_seconds: u64,
    renewal_recommended: bool,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

/// Operator-supplied, signature-verified witness descriptor view.
///
/// [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] Only configured
/// witness identities enter this ephemeral store. Unrelated snapshot peers
/// cannot consume capacity, influence endpoint selection, or appear in output.
struct RelayCustodyWitnessSnapshot {
    peer_store: PeerStore,
    records: usize,
    pinned_records: usize,
    verified_records: usize,
    rejected_records: usize,
}

/// Current producer custody context protected from concurrent checkpoint change.
///
/// [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Both import and audit use
/// this single boundary so identity, configured pins, current anchor generation,
/// and canonical frame digest cannot be checked under different policies.
struct CurrentRelayCustodyAuditWitnessContext {
    config: ServerConfig,
    identity: IdentityKeyPair,
    producer: [u8; 32],
    configured_witnesses: Vec<[u8; 32]>,
    anchor_guard: ChatRelayCustodyAuditAnchorGuard,
    anchor_sha256: [u8; 32],
}

/// Fully verified pre-persistence context for one producer receipt import.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] File, identity, pin,
/// signature, and exact-current-anchor checks complete before this value can
/// exist. The value keeps the cross-process maintenance lock alive, while
/// receipt-vault mutation remains a separate phase in the command handler.
struct VerifiedRelayCustodyAuditWitnessImport {
    current: CurrentRelayCustodyAuditWitnessContext,
    receipt: CustodyAuditWitnessReceiptV1,
}

async fn load_current_relay_custody_audit_witness_context(
    config_path: &Path,
    operation: &'static str,
) -> anyhow::Result<CurrentRelayCustodyAuditWitnessContext> {
    let config = load_relay_custody_config(config_path).await?;
    anyhow::ensure!(
        config.memchain.is_enabled() && config.memchain.db_path.trim() != ":memory:",
        "custody witness receipt policy requires persistent local MemChain storage"
    );
    let identity = load_relay_custody_identity(&config, operation).await?;
    let producer = identity.public_key_bytes();
    let configured_witnesses = config.discovery.custody_audit_witness_node_id_bytes();
    anyhow::ensure!(
        !configured_witnesses.is_empty(),
        "custody witness receipt policy has no configured independent witnesses"
    );

    // [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Keep the exact
    // checkpoint immutable through every local policy query. A concurrent
    // backup-maintenance process cannot make the final report stale mid-command.
    let anchor_guard = ChatRelayService::hold_backup_maintenance_audit_anchor_for_config(
        &config.memchain.chat_relay,
        &identity,
    )
    .map_err(|_| anyhow::anyhow!("unable to hold current relay custody anchor"))?;
    let current_anchor_frame = encode_custody_audit_anchor(anchor_guard.anchor())
        .map_err(|_| anyhow::anyhow!("unable to encode current relay custody anchor"))?;
    let anchor_sha256: [u8; 32] = Sha256::digest(&current_anchor_frame).into();
    Ok(CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
async fn verify_relay_custody_audit_witness_import(
    config_path: &Path,
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_witness_hex: &str,
) -> anyhow::Result<VerifiedRelayCustodyAuditWitnessImport> {
    let current =
        load_current_relay_custody_audit_witness_context(config_path, "audit witness import")
            .await?;
    let expected_witness = parse_hex32(expected_witness_hex, "expected witness node identity")?;
    anyhow::ensure!(
        expected_witness != current.producer,
        "producer and independent witness identities must differ"
    );
    anyhow::ensure!(
        current.configured_witnesses.contains(&expected_witness),
        "witness identity is not pinned by discovery.custody_audit_witness_node_ids"
    );

    let anchor_sha256 = parse_hex32(anchor_sha256_hex, "audit anchor SHA-256")?;
    let anchor_frame = read_bounded_relay_custody_artifact(
        anchor_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor =
        verify_relay_custody_anchor_frame(&anchor_frame, &anchor_sha256, &current.producer, 1)?;
    anyhow::ensure!(
        current.anchor_guard.anchor().checkpoint_generation == anchor.checkpoint_generation
            && current.anchor_sha256 == anchor_sha256,
        "custody witness receipt anchor is not the current local checkpoint"
    );

    let receipt_sha256 = parse_hex32(receipt_sha256_hex, "audit witness receipt SHA-256")?;
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
        &current.producer,
        &expected_witness,
        anchor.checkpoint_generation,
    )?;
    Ok(VerifiedRelayCustodyAuditWitnessImport { current, receipt })
}

fn print_relay_custody_audit_witness_import(
    report: &RelayCustodyAuditWitnessImportReport,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
        return Ok(());
    }
    println!("Relay custody witness receipt import");
    println!("════════════════════════════════════════");
    println!("Status:               {}", report.status);
    println!("Import disposition:   {}", report.import_disposition);
    println!("Receipt outcome:      {}", report.receipt_outcome);
    println!("Checkpoint generation: {}", report.checkpoint_generation);
    println!("Freshness window:     {}s", report.max_age_seconds);
    println!(
        "Quorum valid through:  {}",
        optional_u64_label(report.quorum_valid_through, "")
    );
    println!(
        "Quorum valid for:      {}",
        optional_u64_label(report.quorum_valid_for_seconds, "s")
    );
    println!(
        "Renewal window:        {}s",
        report.renewal_warning_window_seconds
    );
    println!("Renewal recommended:  {}", report.renewal_recommended);
    println!("Vault records:        {}", report.vault_records);
    println!("Configured witnesses: {}", report.configured_witnesses);
    println!("Fresh verified:       {}", report.fresh_verified);
    println!(
        "Accepted / adverse:   {} / {}",
        report.accepted, report.adverse
    );
    println!("Missing:              {}", report.missing);
    println!("Minimum verified:     {}", report.minimum_verified);
    println!("Policy ready:         {}", report.policy_ready);
    println!();
    println!("Security model: {}", report.security_model);
    println!("Privacy: {}", report.privacy_boundary);
    Ok(())
}

fn open_relay_custody_witness_storage(
    config: &ServerConfig,
    identity: &IdentityKeyPair,
) -> anyhow::Result<MemoryStorage> {
    let record_key = derive_record_key(&identity.to_bytes());
    MemoryStorage::open(&config.memchain.db_path, Some(record_key))
        .map_err(|_| anyhow::anyhow!("unable to open custody witness receipt vault"))
}

fn custody_audit_witness_policy_status(
    policy: &CustodyAuditWitnessReceiptPolicyEvidence,
) -> &'static str {
    policy
        .readiness()
        .map_or("invalid", |readiness| readiness.status_label())
}

fn optional_u64_label(value: Option<u64>, suffix: &str) -> String {
    // [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Human output must preserve
    // the distinction between a real zero-second boundary and unavailable
    // evidence. JSON already carries that distinction as a nullable number.
    value.map_or_else(
        || "unavailable".to_owned(),
        |number| format!("{number}{suffix}"),
    )
}

fn custody_audit_witness_renewal_fields(
    policy: &CustodyAuditWitnessReceiptPolicyEvidence,
    evaluated_at: u64,
    max_age_seconds: u64,
) -> (Option<u64>, Option<u64>, u64, bool) {
    // [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] These are aggregate local
    // operations fields only. They identify no witness and trigger no network
    // request, evidence mutation, or automatic authority change.
    let valid_for_seconds = policy.quorum_valid_for_secs(evaluated_at);
    let warning_window_seconds = custody_witness_renewal_warning_window_secs(max_age_seconds);
    let renewal_recommended =
        valid_for_seconds.is_some_and(|valid_for| valid_for <= warning_window_seconds);
    (
        policy.quorum_valid_through,
        valid_for_seconds,
        warning_window_seconds,
        renewal_recommended,
    )
}

fn print_relay_custody_audit_witness_vault(
    report: &RelayCustodyAuditWitnessVaultReport,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
        return Ok(());
    }
    println!("Relay custody witness vault audit");
    println!("════════════════════════════════════════");
    println!("Status:               {}", report.status);
    println!("Evaluated at:         {}", report.evaluated_at);
    println!("Checkpoint generation: {}", report.checkpoint_generation);
    println!("Freshness window:     {}s", report.max_age_seconds);
    println!(
        "Quorum valid through:  {}",
        optional_u64_label(report.quorum_valid_through, "")
    );
    println!(
        "Quorum valid for:      {}",
        optional_u64_label(report.quorum_valid_for_seconds, "s")
    );
    println!(
        "Renewal window:        {}s",
        report.renewal_warning_window_seconds
    );
    println!("Renewal recommended:  {}", report.renewal_recommended);
    println!("Vault records:        {}", report.vault_records);
    println!(
        "Vault accepted/adverse: {} / {}",
        report.vault_accepted_records, report.vault_adverse_records
    );
    println!("Configured witnesses: {}", report.configured_witnesses);
    println!("Fresh verified:       {}", report.fresh_verified);
    println!(
        "Accepted / adverse:   {} / {}",
        report.accepted, report.adverse
    );
    println!("Missing:              {}", report.missing);
    println!("Minimum verified:     {}", report.minimum_verified);
    println!("Policy ready:         {}", report.policy_ready);
    println!("Ready required:       {}", report.required_ready);
    println!();
    println!("Security model: {}", report.security_model);
    println!("Privacy: {}", report.privacy_boundary);
    Ok(())
}

fn print_relay_custody_audit_witness_collection(
    report: &RelayCustodyAuditWitnessCollectionReport,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
        return Ok(());
    }
    println!("Relay custody witness collection");
    println!("════════════════════════════════════════");
    println!("Status:               {}", report.status);
    println!("Evaluated at:         {}", report.evaluated_at);
    println!("Checkpoint generation: {}", report.checkpoint_generation);
    println!("Request timeout:      {}s", report.timeout_seconds);
    println!("Freshness window:     {}s", report.max_age_seconds);
    println!(
        "Quorum valid through:  {}",
        optional_u64_label(report.quorum_valid_through, "")
    );
    println!(
        "Quorum valid for:      {}",
        optional_u64_label(report.quorum_valid_for_seconds, "s")
    );
    println!(
        "Renewal window:        {}s",
        report.renewal_warning_window_seconds
    );
    println!("Renewal recommended:  {}", report.renewal_recommended);
    println!(
        "Snapshot total/pinned: {} / {}",
        report.snapshot_records, report.snapshot_pinned_records
    );
    println!(
        "Snapshot verified/rejected: {} / {}",
        report.snapshot_verified_records, report.snapshot_rejected_records
    );
    println!(
        "Round verified/accepted: {} / {}",
        report.round_verified, report.round_accepted
    );
    println!(
        "Round advanced/idempotent: {} / {}",
        report.round_advanced, report.round_idempotent
    );
    println!(
        "Round stale/conflict/gap: {} / {} / {}",
        report.round_stale, report.round_conflicts, report.round_gaps
    );
    println!("Round transport failures: {}", report.round_failed);
    println!("Round adverse:        {}", report.round_adverse);
    println!("Vault records:        {}", report.vault_records);
    println!(
        "Vault accepted/adverse: {} / {}",
        report.vault_accepted_records, report.vault_adverse_records
    );
    println!("Configured witnesses: {}", report.configured_witnesses);
    println!("Fresh verified:       {}", report.fresh_verified);
    println!(
        "Accepted / adverse:   {} / {}",
        report.accepted, report.adverse
    );
    println!("Missing:              {}", report.missing);
    println!("Minimum verified:     {}", report.minimum_verified);
    println!("Policy ready:         {}", report.policy_ready);
    println!();
    println!("Security model: {}", report.security_model);
    println!("Privacy: {}", report.privacy_boundary);
    Ok(())
}

const MAX_CUSTODY_WITNESS_DISCOVERY_SNAPSHOT_BYTES: usize = 512 * 1024;

fn load_relay_custody_witness_snapshot(
    path: &Path,
    configured_witnesses: &[[u8; 32]],
    max_peers: usize,
    now: u64,
) -> anyhow::Result<RelayCustodyWitnessSnapshot> {
    let bytes = read_bounded_relay_custody_artifact(
        path,
        MAX_CUSTODY_WITNESS_DISCOVERY_SNAPSHOT_BYTES,
        "discovery snapshot",
    )?;
    let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes)
        .map_err(|_| anyhow::anyhow!("relay custody discovery snapshot is malformed"))?;
    let records = snapshot.peers.len();
    let pinned_peers = snapshot
        .peers
        .into_iter()
        .filter(|peer| configured_witnesses.contains(&peer.descriptor.node_id))
        .collect::<Vec<_>>();
    let pinned_records = pinned_peers.len();
    let pinned_snapshot = NodeBootstrapSnapshot::new(snapshot.generated_at, pinned_peers);
    let peer_store = PeerStore::with_max_peers(max_peers);
    let imported = peer_store.load_bootstrap_snapshot_from_source(
        &pinned_snapshot,
        now,
        "operator_custody_witness_snapshot",
    );
    let verified_records = imported.inserted.saturating_add(imported.unchanged);
    let rejected_records = imported.rejected.saturating_add(imported.stale);
    Ok(RelayCustodyWitnessSnapshot {
        peer_store,
        records,
        pinned_records,
        verified_records,
        rejected_records,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_relay_custody_audit_witness_collection_report(
    round: CustodyAuditWitnessRound,
    snapshot: &RelayCustodyWitnessSnapshot,
    policy: &CustodyAuditWitnessReceiptPolicyEvidence,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    evaluated_at: u64,
    checkpoint_generation: u64,
    timeout_seconds: u64,
    max_age_seconds: u64,
) -> RelayCustodyAuditWitnessCollectionReport {
    let status = custody_audit_witness_policy_status(policy);
    let (
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
    ) = custody_audit_witness_renewal_fields(policy, evaluated_at, max_age_seconds);
    RelayCustodyAuditWitnessCollectionReport {
        contract_version: "relay_custody_audit_witness_collection.v1",
        status,
        evaluated_at,
        checkpoint_generation,
        timeout_seconds,
        max_age_seconds,
        snapshot_records: snapshot.records,
        snapshot_pinned_records: snapshot.pinned_records,
        snapshot_verified_records: snapshot.verified_records,
        snapshot_rejected_records: snapshot.rejected_records,
        round_configured: round.configured,
        round_verified: round.verified,
        round_accepted: round.accepted,
        round_advanced: round.advanced,
        round_idempotent: round.idempotent,
        round_stale: round.stale,
        round_conflicts: round.conflicts,
        round_gaps: round.gaps,
        round_failed: round.failed,
        round_adverse: round.adverse_evidence,
        vault_records,
        vault_accepted_records,
        vault_adverse_records,
        configured_witnesses: policy.configured,
        fresh_verified: policy.fresh_verified,
        accepted: policy.accepted,
        adverse: policy.adverse,
        missing: policy.missing,
        minimum_verified: policy.minimum_verified,
        policy_ready: status == "ready",
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
        security_model: "explicit one-shot exact-pin witness transport with signed descriptor admission, durable receipt-before-counting, complete vault re-audit, and no background scheduler, voting, fork choice, consensus, or global finality",
        privacy_boundary: "aggregate snapshot, round, vault, and current policy counts only; no node identities, hashes, signatures, paths, endpoints, messages, users, routes, payloads, memory, destinations, DNS, IP addresses, or social graph metadata",
    }
}

pub(super) async fn cmd_relay_collect_audit_witnesses(
    config_path: &Path,
    discovery_snapshot_path: &Path,
    timeout_seconds: u64,
    max_age_seconds: u64,
    json: bool,
) -> anyhow::Result<()> {
    let current =
        load_current_relay_custody_audit_witness_context(config_path, "collect audit witnesses")
            .await?;
    let CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    } = current;
    let started_at = unix_timestamp_now()?;
    let snapshot = load_relay_custody_witness_snapshot(
        discovery_snapshot_path,
        &configured_witnesses,
        config.discovery.max_peers,
        started_at,
    )?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(timeout_seconds))
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|_| anyhow::anyhow!("unable to build custody witness transport"))?;
    let storage = open_relay_custody_witness_storage(&config, &identity)?;

    // [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] The maintenance
    // guard remains held across network contact and durable re-audit. The
    // command therefore cannot report receipts for a checkpoint that changed
    // while witnesses were responding.
    let round = witness_custody_audit_anchor_round_durable(
        &storage,
        &snapshot.peer_store,
        &identity,
        &client,
        &configured_witnesses,
        config.discovery.custody_audit_witness_min_verified,
        anchor_guard.anchor(),
    )
    .await
    .map_err(|reason| anyhow::anyhow!("custody witness collection failed: {reason}"))?;
    let evaluated_at = unix_timestamp_now()?;
    // [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Report and exit
    // status must describe the exact same cryptographically audited snapshot.
    let readiness = storage
        .audit_custody_audit_witness_receipt_readiness(
            &producer,
            anchor_guard.anchor().checkpoint_generation,
            &anchor_sha256,
            &configured_witnesses,
            config.discovery.custody_audit_witness_min_verified,
            evaluated_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness readiness audit failed closed"))?;
    let vault = readiness.vault;
    let policy = readiness.policy;
    let report = build_relay_custody_audit_witness_collection_report(
        round,
        &snapshot,
        &policy,
        vault.records,
        vault.accepted_records,
        vault.adverse_records,
        evaluated_at,
        anchor_guard.anchor().checkpoint_generation,
        timeout_seconds,
        max_age_seconds,
    );
    let policy_ready = report.policy_ready;
    print_relay_custody_audit_witness_collection(&report, json)?;
    anyhow::ensure!(
        policy_ready,
        "current custody witness policy is not ready after collection"
    );
    Ok(())
}

pub(super) async fn cmd_relay_audit_witness_vault(
    config_path: &Path,
    max_age_seconds: u64,
    require_ready: bool,
    json: bool,
) -> anyhow::Result<()> {
    let current =
        load_current_relay_custody_audit_witness_context(config_path, "audit witness vault")
            .await?;
    let CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    } = current;
    let evaluated_at = unix_timestamp_now()?;
    let storage = open_relay_custody_witness_storage(&config, &identity)?;
    let readiness = storage
        .audit_custody_audit_witness_receipt_readiness(
            &producer,
            anchor_guard.anchor().checkpoint_generation,
            &anchor_sha256,
            &configured_witnesses,
            config.discovery.custody_audit_witness_min_verified,
            evaluated_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness readiness audit failed closed"))?;
    let vault = readiness.vault;
    let policy = readiness.policy;
    let status = custody_audit_witness_policy_status(&policy);
    let policy_ready = status == "ready";
    let (
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
    ) = custody_audit_witness_renewal_fields(&policy, evaluated_at, max_age_seconds);
    let report = RelayCustodyAuditWitnessVaultReport {
        contract_version: "relay_custody_audit_witness_vault.v1",
        status,
        evaluated_at,
        checkpoint_generation: anchor_guard.anchor().checkpoint_generation,
        max_age_seconds,
        vault_records: vault.records,
        vault_accepted_records: vault.accepted_records,
        vault_adverse_records: vault.adverse_records,
        configured_witnesses: policy.configured,
        fresh_verified: policy.fresh_verified,
        accepted: policy.accepted,
        adverse: policy.adverse,
        missing: policy.missing,
        minimum_verified: policy.minimum_verified,
        policy_ready,
        required_ready: require_ready,
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
        security_model: "host-local current-checkpoint receipt re-audit under an exclusive maintenance guard; no witness contact, consensus, voting, fork choice, or global finality",
        privacy_boundary: "aggregate vault and current policy counts only; no node identities, hashes, signatures, paths, endpoints, messages, users, routes, payloads, memory, destinations, DNS, IP addresses, or social graph metadata",
    };
    print_relay_custody_audit_witness_vault(&report, json)?;
    anyhow::ensure!(
        !require_ready || policy_ready,
        "current custody witness policy is not ready"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn cmd_relay_import_audit_witness(
    config_path: &Path,
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_witness_hex: &str,
    max_age_seconds: u64,
    json: bool,
) -> anyhow::Result<()> {
    let verified = verify_relay_custody_audit_witness_import(
        config_path,
        anchor_path,
        anchor_sha256_hex,
        receipt_path,
        receipt_sha256_hex,
        expected_witness_hex,
    )
    .await?;
    let VerifiedRelayCustodyAuditWitnessImport { current, receipt } = verified;
    let CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    } = current;
    let anchor = anchor_guard.anchor();
    let imported_at = unix_timestamp_now()?;
    let storage = open_relay_custody_witness_storage(&config, &identity)?;
    let disposition = storage
        .import_custody_audit_witness_receipt(
            &receipt,
            &producer,
            anchor.checkpoint_generation,
            &anchor_sha256,
            imported_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt import failed closed"))?;
    let readiness = storage
        .audit_custody_audit_witness_receipt_readiness(
            &producer,
            anchor.checkpoint_generation,
            &anchor_sha256,
            &configured_witnesses,
            config.discovery.custody_audit_witness_min_verified,
            imported_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness readiness audit failed closed"))?;
    let vault = readiness.vault;
    let policy = readiness.policy;
    let status = custody_audit_witness_policy_status(&policy);
    let (
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
    ) = custody_audit_witness_renewal_fields(&policy, imported_at, max_age_seconds);
    let report = RelayCustodyAuditWitnessImportReport {
        contract_version: "relay_custody_audit_witness_import.v1",
        status,
        import_disposition: disposition.as_str(),
        receipt_outcome: custody_audit_witness_outcome_label(receipt.outcome),
        checkpoint_generation: anchor.checkpoint_generation,
        observed_at: receipt.observed_at,
        max_age_seconds,
        vault_records: vault.records,
        vault_accepted_records: vault.accepted_records,
        vault_adverse_records: vault.adverse_records,
        configured_witnesses: policy.configured,
        fresh_verified: policy.fresh_verified,
        accepted: policy.accepted,
        adverse: policy.adverse,
        missing: policy.missing,
        minimum_verified: policy.minimum_verified,
        policy_ready: status == "ready",
        quorum_valid_through,
        quorum_valid_for_seconds,
        renewal_warning_window_seconds,
        renewal_recommended,
        security_model: "host-local exact-current-anchor import; canonical signed receipts are re-audited before policy evaluation and do not establish consensus or global finality",
        privacy_boundary: "aggregate vault and exact-anchor policy counts only; no message, user, route, endpoint, IP, payload, memory, DNS, destination, or social graph metadata",
    };
    print_relay_custody_audit_witness_import(&report, json)?;
    anyhow::ensure!(
        receipt.accepted(),
        "signed adverse custody witness evidence was retained for operator review"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
