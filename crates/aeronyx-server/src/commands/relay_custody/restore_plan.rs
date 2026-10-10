// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/restore_plan.rs
// ============================================
//! # Relay custody restore plans
//!
//! Owns short-lived, node-secret-authenticated restore plans, restore
//! readiness output, and the strict private restore-plan file loader.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::io::Read;
use std::path::Path;

use aeronyx_server::services::{
    ChatRelayRestorePlanReceipt, ChatRelayRestoreReadinessReceipt, ChatRelayService,
    CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS,
};

use super::host_io::{load_relay_custody_config, load_relay_custody_node_secret};

// [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Keep credential issuance and
// verification isolated from the command dispatcher and all network surfaces.
pub(super) async fn cmd_relay_restore_plan(config_path: &Path, json: bool) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret = load_relay_custody_node_secret(&server_config, "restore plan").await?;
    let plan = ChatRelayService::create_latest_restore_plan_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
    )
    .map_err(|error| anyhow::anyhow!("relay custody restore planning failed: {error}"))?;
    print_relay_restore_plan(&plan, json)
}

pub(super) async fn cmd_relay_verify_restore_plan(
    config_path: &Path,
    plan_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret =
        load_relay_custody_node_secret(&server_config, "restore-plan verification").await?;
    let plan = load_private_restore_plan(plan_path)?;
    ChatRelayService::verify_latest_restore_plan_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
        &plan,
    )
    .map_err(|error| anyhow::anyhow!("relay custody restore-plan verification failed: {error}"))?;
    if json {
        println!(
            "{}",
            serde_json::json!({"valid": true, "expires_at": plan.expires_at})
        );
    } else {
        println!(
            "Relay custody restore plan is valid until {}.",
            plan.expires_at
        );
        println!("Verification is read-only and does not authorize restoration.");
    }
    Ok(())
}

pub(super) fn print_relay_restore_readiness(
    receipt: &ChatRelayRestoreReadinessReceipt,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(receipt)?);
        return Ok(());
    }

    println!("Relay custody restore readiness");
    println!("════════════════════════════════════════");
    println!("Ready:               {}", receipt.ready);
    println!("Verified backups:    {}", receipt.verified_backup_count);
    println!("Selected bytes:      {}", receipt.selected_backup_bytes);
    println!("Active DB present:   {}", receipt.active_database_present);
    println!("Active DB bytes:     {}", receipt.active_database_bytes);
    println!("Active sidecars:     {}", receipt.active_sidecars_present);
    println!(
        "Blocker:              {}",
        receipt.blocker.unwrap_or("none")
    );
    println!();
    println!("Read-only preflight; no custody data was replaced.");
    Ok(())
}

fn print_relay_restore_plan(plan: &ChatRelayRestorePlanReceipt, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(plan)?);
        return Ok(());
    }

    println!("Relay custody authenticated restore plan");
    println!("════════════════════════════════════════");
    println!("Version:             {}", plan.version);
    println!("Issued at:           {}", plan.issued_at);
    println!("Expires at:          {}", plan.expires_at);
    println!("Validity:            {CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS}s");
    println!("Verified backups:    {}", plan.verified_backup_count);
    println!("Selected bytes:      {}", plan.selected_backup_bytes);
    println!("Active DB present:   {}", plan.active_database_present);
    println!("Active DB bytes:     {}", plan.active_database_bytes);
    println!("Nonce:               {}", plan.nonce);
    println!("Commitment:          {}", plan.commitment);
    println!();
    println!("Preflight credential only; this does not authorize or execute restoration.");
    Ok(())
}

fn load_private_restore_plan(path: &Path) -> anyhow::Result<ChatRelayRestorePlanReceipt> {
    #[cfg(unix)]
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    const MAX_RESTORE_PLAN_BYTES: u64 = 4096;

    // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Treat the plan as a local
    // maintenance credential. Never follow its final symlink or include its
    // path/content in an error returned to an operator surface.
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to open private relay restore plan"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to inspect private relay restore plan"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_RESTORE_PLAN_BYTES {
        anyhow::bail!("private relay restore plan has an invalid file boundary");
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o077 != 0 {
        anyhow::bail!("private relay restore plan must be owner-private");
    }

    let capacity = usize::try_from(metadata.len())
        .map_err(|_| anyhow::anyhow!("private relay restore plan exceeds platform capacity"))?;
    let mut encoded = Vec::with_capacity(capacity);
    file.by_ref()
        .take(MAX_RESTORE_PLAN_BYTES + 1)
        .read_to_end(&mut encoded)
        .map_err(|_| anyhow::anyhow!("unable to read private relay restore plan"))?;
    if encoded.len() as u64 != metadata.len() || encoded.len() as u64 > MAX_RESTORE_PLAN_BYTES {
        anyhow::bail!("private relay restore plan changed during bounded read");
    }
    serde_json::from_slice(&encoded)
        .map_err(|_| anyhow::anyhow!("private relay restore plan is malformed"))
}

#[cfg(test)]
mod tests;
