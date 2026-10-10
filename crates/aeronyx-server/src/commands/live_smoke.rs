// ============================================
// File: crates/aeronyx-server/src/commands/live_smoke.rs
// ============================================
//! # `mailbox-probe`, `relay-smoke` and `v1-compatibility-smoke` commands
//!
//! Owns the explicit, confirmation-gated live proofs: the anonymous mailbox
//! probe, the authenticated live relay smoke, and the frozen-v0x01
//! compatibility smoke. The protocol work itself lives in the binary's
//! `mailbox_probe` and `relay_smoke` modules.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::{Path, PathBuf};

use anyhow::Context;

use aeronyx_server::ServerConfig;

use crate::{mailbox_probe, relay_smoke};

/// Runs the client-sourced anonymous mailbox lifecycle against live nodes.
///
/// [MAILBOX-PROBE 2026-10-09 by Claude] Creates one short-lived mailbox and
/// one random test item on the target, so it requires explicit confirmation.
pub async fn cmd_mailbox_probe(
    options: mailbox_probe::MailboxProbeOptions,
    confirmed: bool,
    emit_json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        confirmed,
        "mailbox probe requires --confirm-live-mailbox-probe"
    );
    let report = mailbox_probe::run(options).await?;
    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "AeroNyx anonymous mailbox probe: {} ({} hop)",
            report.status, report.hops
        );
        for step in &report.steps {
            println!(
                "  {:<13} {:<10} {} ms",
                step.operation, step.outcome, step.elapsed_ms
            );
        }
    }
    Ok(())
}

/// Runs one explicit, aggregate-only proof against the node on this host.
pub async fn cmd_relay_smoke(
    server_addr: std::net::SocketAddr,
    health_url: String,
    config_path: PathBuf,
    timeout_seconds: u64,
    confirmed: bool,
    emit_json: bool,
) -> anyhow::Result<()> {
    // [LIVE-RELAY-SMOKE 2026-08-15 by Codex] This command creates protocol
    // traffic and two ephemeral sessions. Requiring an explicit confirmation
    // prevents an operator from confusing a read-only health check with a live
    // end-to-end proof.
    anyhow::ensure!(confirmed, "relay smoke requires --confirm-live-relay-smoke");
    let config = ServerConfig::load(&config_path)
        .await
        .with_context(|| format!("load node config {}", config_path.display()))?;
    anyhow::ensure!(
        server_addr.port() == config.listen_addr().port(),
        "relay smoke UDP port does not match the configured node listener"
    );
    // [RELAY-SMOKE-HEALTH-AUTHORITY 2026-09-01 by Codex] Bind readiness and
    // active-session evidence to the same configured node before opening any
    // ephemeral protocol session. An arbitrary loopback HTTP server is not a
    // valid substitute for the running node's aggregate health surface.
    let health_authority =
        relay_smoke::RelaySmokeHealthAuthority::new(server_addr, config.memchain.api_listen_addr)?;
    let key_path = PathBuf::from(&config.server_key.key_file);
    let expected_server_key = relay_smoke::load_expected_server_public_key(&key_path).await?;
    let report = relay_smoke::run(relay_smoke::RelaySmokeOptions {
        server_addr,
        health_url,
        health_authority,
        expected_server_key,
        timeout: std::time::Duration::from_secs(timeout_seconds),
    })
    .await?;

    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("AeroNyx authenticated live relay smoke");
        println!("  Status:                         {}", report.status);
        println!("  Transport:                      {}", report.transport);
        println!(
            "  Verified client deliveries:     {} -> {}",
            report.verified_client_deliveries_before, report.verified_client_deliveries_after
        );
        println!(
            "  Terminal receipt observed:      {}",
            report.terminal_receipt_observed
        );
        println!(
            "  Entry mailbox round trip:       {}",
            report.entry_mailbox_round_trip_verified
        );
        println!(
            "  Entry mailbox ACK:              {}",
            report.entry_mailbox_ack_verified
        );
        println!(
            "  E2E ciphertext verified:        {}",
            report.e2e_ciphertext_verified
        );
        println!(
            "  Ephemeral sessions:             {}",
            report.ephemeral_sessions_created
        );
        println!(
            "  Session cleanup:                {}",
            report.session_cleanup
        );
        println!(
            "  Terminal replica cleanup:       {}",
            report.terminal_replica_cleanup
        );
        println!(
            "  Evidence scope:                 {}",
            report.evidence_scope
        );
        println!("  Elapsed:                        {} ms", report.elapsed_ms);
        println!(
            "  Privacy boundary:               {}",
            report.privacy_boundary
        );
    }
    Ok(())
}

/// Runs one explicit frozen-v0x01 compatibility proof against this host.
pub async fn cmd_v1_compatibility_smoke(
    server_addr: std::net::SocketAddr,
    health_url: String,
    config_path: PathBuf,
    timeout_seconds: u64,
    confirmed: bool,
    emit_json: bool,
) -> anyhow::Result<()> {
    // [V1-COMPATIBILITY-SMOKE 2026-09-13 by Codex] This creates one real
    // session and waits for the node's scheduled keepalive, so it is never
    // implicit in status/validate and requires an operator confirmation.
    anyhow::ensure!(
        confirmed,
        "v1 compatibility smoke requires --confirm-v1-compatibility-smoke"
    );
    let config = ServerConfig::load(&config_path)
        .await
        .with_context(|| format!("load node config {}", config_path.display()))?;
    anyhow::ensure!(
        server_addr.port() == config.listen_addr().port(),
        "v1 compatibility smoke UDP port does not match the configured node listener"
    );
    let health_authority =
        relay_smoke::RelaySmokeHealthAuthority::new(server_addr, config.memchain.api_listen_addr)?;
    let expected_server_key =
        relay_smoke::load_expected_server_public_key(Path::new(&config.server_key.key_file))
            .await?;
    let report =
        relay_smoke::run_v1_compatibility_smoke(relay_smoke::V1CompatibilitySmokeOptions {
            server_addr,
            health_url,
            health_authority,
            expected_server_key,
            timeout: std::time::Duration::from_secs(timeout_seconds),
        })
        .await?;

    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("AeroNyx frozen v1 compatibility smoke");
        println!("  Status:                         {}", report.status);
        println!(
            "  Protocol version:               0x{:02x}",
            report.protocol_version
        );
        println!("  Transport:                      {}", report.transport);
        println!(
            "  Server keepalive observed:      {}",
            report.server_keepalive_probe_observed
        );
        println!(
            "  Canonical echo reply sent:      {}",
            report.canonical_echo_reply_sent
        );
        println!(
            "  Session cleanup:                {}",
            report.session_cleanup
        );
        println!(
            "  Evidence scope:                 {}",
            report.evidence_scope
        );
        println!("  Elapsed:                        {} ms", report.elapsed_ms);
        println!(
            "  Privacy boundary:               {}",
            report.privacy_boundary
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
