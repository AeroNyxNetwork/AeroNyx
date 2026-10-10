// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody.rs
// ============================================
//! # `relay-custody` commands
//!
//! Owns the host-local `relay-custody` dispatcher (retention audit, prune,
//! restore readiness) and maintenance-audit chain verification. Anchors,
//! witness receipts, the producer witness vault, restore plans and the bounded
//! artifact files live in the child modules below.
//!
//! ## Module Layout
//! - `relay_custody/audit_anchor.rs`: audit anchor export and offline verification
//! - `relay_custody/witness_receipt.rs`: witness countersigning and receipt verification
//! - `relay_custody/witness_vault.rs`: producer receipt import, vault audit, collection
//! - `relay_custody/restore_plan.rs`: restore plans and readiness output
//! - `relay_custody/host_io.rs`: bounded artifact files and config/secret/identity loading
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::Path;

use aeronyx_server::services::{
    ChatRelayBackupPruneRequest, ChatRelayService, CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
};

use crate::RelayCustodyCommands;
use audit_anchor::{cmd_relay_create_audit_anchor, cmd_relay_verify_audit_anchor};
use host_io::{load_relay_custody_config, load_relay_custody_node_secret};
use restore_plan::{
    cmd_relay_restore_plan, cmd_relay_verify_restore_plan, print_relay_restore_readiness,
};
use witness_receipt::{cmd_relay_verify_audit_witness, cmd_relay_witness_audit_anchor};
use witness_vault::{
    cmd_relay_audit_witness_vault, cmd_relay_collect_audit_witnesses,
    cmd_relay_import_audit_witness,
};

mod audit_anchor;
mod host_io;
mod restore_plan;
mod witness_receipt;
mod witness_vault;

/// Runs host-local relay custody maintenance without opening an HTTP surface.
pub async fn cmd_relay_custody(command: RelayCustodyCommands) -> anyhow::Result<()> {
    // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Keep this boundary local:
    // it reads node-owned config/key material and never calls CMS or HTTP.
    // Plans and readiness checks cannot replace or remove custody data.
    match command {
        RelayCustodyCommands::Audit { config, json } => {
            let server_config = load_relay_custody_config(&config).await?;
            let receipt = ChatRelayService::audit_verified_backup_retention_for_config(
                &server_config.memchain.chat_relay,
            )
            .map_err(|error| anyhow::anyhow!("relay custody audit failed: {error}"))?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!("Relay custody retention audit");
                println!("════════════════════════════════════════");
                println!("Retained backups:   {}", receipt.retained_count);
                println!("Retained bytes:     {}", receipt.retained_bytes);
                println!("Excess backups:     {}", receipt.excess_count);
                println!("Excess bytes:       {}", receipt.excess_bytes);
                println!("Interrupted files:  {}", receipt.partial_count);
                println!("Interrupted bytes:  {}", receipt.partial_bytes);
                println!("Budget exceeded:    {}", receipt.budget_exceeded);
            }
        }
        RelayCustodyCommands::VerifyAudit { config, json } => {
            cmd_relay_verify_audit(&config, json).await?;
        }
        RelayCustodyCommands::CreateAuditAnchor {
            config,
            output,
            json,
        } => {
            cmd_relay_create_audit_anchor(&config, &output, json).await?;
        }
        RelayCustodyCommands::VerifyAuditAnchor {
            input,
            expected_sha256,
            expected_node,
            minimum_checkpoint_generation,
            json,
        } => {
            cmd_relay_verify_audit_anchor(
                &input,
                &expected_sha256,
                &expected_node,
                minimum_checkpoint_generation,
                json,
            )?;
        }
        RelayCustodyCommands::WitnessAuditAnchor {
            config,
            input,
            expected_sha256,
            expected_producer,
            minimum_checkpoint_generation,
            output,
            json,
        } => {
            cmd_relay_witness_audit_anchor(
                &config,
                &input,
                &expected_sha256,
                &expected_producer,
                minimum_checkpoint_generation,
                &output,
                json,
            )
            .await?;
        }
        RelayCustodyCommands::VerifyAuditWitness {
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
            json,
        } => {
            cmd_relay_verify_audit_witness(
                &anchor,
                &anchor_sha256,
                &receipt,
                &receipt_sha256,
                &expected_producer,
                &expected_witness,
                minimum_checkpoint_generation,
                json,
            )?;
        }
        RelayCustodyCommands::ImportAuditWitness {
            config,
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_witness,
            max_age_seconds,
            json,
        } => {
            cmd_relay_import_audit_witness(
                &config,
                &anchor,
                &anchor_sha256,
                &receipt,
                &receipt_sha256,
                &expected_witness,
                max_age_seconds,
                json,
            )
            .await?;
        }
        RelayCustodyCommands::AuditWitnessVault {
            config,
            max_age_seconds,
            require_ready,
            json,
        } => {
            cmd_relay_audit_witness_vault(&config, max_age_seconds, require_ready, json).await?;
        }
        RelayCustodyCommands::CollectAuditWitnesses {
            config,
            discovery_snapshot,
            timeout_seconds,
            max_age_seconds,
            json,
        } => {
            cmd_relay_collect_audit_witnesses(
                &config,
                &discovery_snapshot,
                timeout_seconds,
                max_age_seconds,
                json,
            )
            .await?;
        }
        RelayCustodyCommands::RestoreReadiness { config, json } => {
            let server_config = load_relay_custody_config(&config).await?;
            let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(
                &server_config.memchain.chat_relay,
            )
            .map_err(|error| anyhow::anyhow!("relay custody restore preflight failed: {error}"))?;
            print_relay_restore_readiness(&receipt, json)?;
        }
        RelayCustodyCommands::RestorePlan { config, json } => {
            cmd_relay_restore_plan(&config, json).await?;
        }
        RelayCustodyCommands::VerifyRestorePlan {
            config,
            plan_file,
            json,
        } => {
            cmd_relay_verify_restore_plan(&config, &plan_file, json).await?;
        }
        RelayCustodyCommands::Prune {
            config,
            execute,
            confirm_node_stopped,
            confirm_prune,
            json,
        } => {
            let server_config = load_relay_custody_config(&config).await?;
            let node_secret = load_relay_custody_node_secret(&server_config, "prune").await?;
            let request = ChatRelayBackupPruneRequest {
                execute,
                confirmation: confirm_prune,
                node_stopped_confirmed: confirm_node_stopped,
            };
            let receipt = ChatRelayService::prune_verified_backup_retention_for_config(
                &server_config.memchain.chat_relay,
                &node_secret,
                &request,
            )
            .map_err(|error| anyhow::anyhow!("relay custody prune failed: {error}"))?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!(
                    "Relay custody retention {}",
                    if receipt.executed { "prune" } else { "dry-run" }
                );
                println!("════════════════════════════════════════");
                println!("Planned backups:    {}", receipt.planned_backup_count);
                println!("Planned bytes:      {}", receipt.planned_backup_bytes);
                println!("Planned partials:   {}", receipt.planned_partial_count);
                println!("Partial bytes:      {}", receipt.planned_partial_bytes);
                println!("Deleted backups:    {}", receipt.deleted_backup_count);
                println!("Deleted bytes:      {}", receipt.deleted_backup_bytes);
                println!("Deleted partials:   {}", receipt.deleted_partial_count);
                println!("Partial bytes freed: {}", receipt.deleted_partial_bytes);
                println!("Remaining backups:  {}", receipt.remaining.retained_count);
                println!("Remaining excess:   {}", receipt.remaining.excess_count);
                if !receipt.executed {
                    println!();
                    println!("Dry-run only; no recovery artifact was deleted.");
                    println!(
                        "Execution requires --execute --confirm-node-stopped --confirm-prune {CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION}"
                    );
                }
            }
        }
    }
    Ok(())
}

// [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Verification owns no network
// client and emits only fixed aggregate fields. Keep node-secret loading and
// chain authentication outside the general command dispatcher.
async fn cmd_relay_verify_audit(config_path: &Path, json: bool) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret =
        load_relay_custody_node_secret(&server_config, "maintenance audit verification").await?;
    let receipt = ChatRelayService::verify_backup_maintenance_audit_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
    )
    .map_err(|error| anyhow::anyhow!("relay custody audit verification failed: {error}"))?;
    if json {
        println!("{}", serde_json::to_string(&receipt)?);
    } else {
        println!("Relay custody maintenance audit");
        println!("════════════════════════════════════════");
        println!("Verified:            {}", receipt.verified);
        println!("Records:             {}", receipt.record_count);
        println!(
            "Last recorded at:    {}",
            receipt.last_recorded_at.unwrap_or(0)
        );
        println!("Dry runs:            {}", receipt.dry_run_count);
        println!("Planned executions:  {}", receipt.planned_count);
        println!("Completed:           {}", receipt.completed_count);
        println!("Failed:              {}", receipt.failed_count);
        println!("Verified bytes:      {}", receipt.verified_bytes);
        println!("Checkpoints:         {}", receipt.checkpoint_count);
        println!("Archived records:    {}", receipt.archived_record_count);
        println!("Active records:      {}", receipt.active_record_count);
        println!("Archived bytes:      {}", receipt.archived_bytes);
        println!("Rotation pending:    {}", receipt.rotation_pending);
        println!();
        println!("Read-only verification; no audit or custody data was changed.");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
