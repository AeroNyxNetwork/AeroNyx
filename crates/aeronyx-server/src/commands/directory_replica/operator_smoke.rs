// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/operator_smoke.rs
// ============================================
//! # Directory Replica operator smoke calls
//!
//! Owns the bounded, read-only calls to the running node's loopback operator
//! API (carrier smoke and cold-bootstrap smoke).
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::PathBuf;

use anyhow::Context;

use aeronyx_server::ServerConfig;

/// Calls the running node's local-only read-only carrier verification route.
pub(super) async fn cmd_directory_replica_carrier_smoke(
    config_path: &PathBuf,
    emit_json: bool,
) -> anyhow::Result<()> {
    cmd_directory_replica_operator_smoke(
        config_path,
        emit_json,
        "carrier-smoke",
        "Directory Mirror carrier smoke",
        &[
            "retained_producers",
            "eligible_retained_producers",
            "explicit_carrier_candidates",
            "attempted_carriers",
            "verified_blocks",
            "verified_descriptor_objects",
            "storage_effect",
        ],
    )
    .await
}

/// Proves that an empty isolated replica can bootstrap without the producer.
pub(super) async fn cmd_directory_replica_carrier_cold_bootstrap_smoke(
    config_path: &PathBuf,
    emit_json: bool,
) -> anyhow::Result<()> {
    cmd_directory_replica_operator_smoke(
        config_path,
        emit_json,
        "carrier-cold-bootstrap-smoke",
        "Directory carrier cold-bootstrap smoke",
        &[
            "configured_producers",
            "eligible_producers",
            "explicit_carrier_candidates",
            "attempted_carriers",
            "imported_blocks",
            "imported_commitments",
            "bootstrapped_tip_height",
            "live_store_effect",
        ],
    )
    .await
}

/// Calls one bounded local-only Directory Replica smoke endpoint.
async fn cmd_directory_replica_operator_smoke(
    config_path: &PathBuf,
    emit_json: bool,
    operation: &'static str,
    title: &'static str,
    fields: &[&str],
) -> anyhow::Result<()> {
    const MAX_SMOKE_RESPONSE_BYTES: usize = 64 * 1024;

    let config = ServerConfig::load(config_path)
        .await
        .with_context(|| format!("load node config {}", config_path.display()))?;
    let url = directory_replica_operator_smoke_url(&config, operation);
    // [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] Both smoke commands stay
    // on loopback, ignore proxy settings, reject redirects, and stream into a
    // hard response limit. `Content-Length` is never trusted as the bound.
    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(45))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("initialize local Directory Replica smoke HTTP client")?;
    let mut response = client
        .post(url)
        .send()
        .await
        .context("contact the running node Directory Replica smoke endpoint")?;
    let http_status = response.status();
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read bounded Directory Replica smoke response")?
    {
        anyhow::ensure!(
            body.len().saturating_add(chunk.len()) <= MAX_SMOKE_RESPONSE_BYTES,
            "Directory Replica smoke response exceeded its local protocol bound"
        );
        body.extend_from_slice(&chunk);
    }
    let report: serde_json::Value =
        serde_json::from_slice(&body).context("decode Directory Replica smoke response")?;
    if emit_json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("{title}");
        println!(
            "  status: {}",
            report["status"].as_str().unwrap_or("unavailable")
        );
        for field in fields {
            let value = &report[*field];
            if let Some(value) = value.as_str() {
                println!("  {field}: {value}");
            } else if let Some(value) = value.as_u64() {
                println!("  {field}: {value}");
            } else if let Some(value) = value.as_bool() {
                println!("  {field}: {value}");
            }
        }
        if let Some(reason) = report["failure_reason"].as_str() {
            println!("  failure_reason: {reason}");
        }
        println!(
            "  privacy: {}",
            report["privacy_boundary"]
                .as_str()
                .unwrap_or("aggregate verification metadata only")
        );
    }
    anyhow::ensure!(
        http_status.is_success() && report["success"].as_bool() == Some(true),
        "Directory Replica smoke was not verified"
    );
    Ok(())
}

/// Resolve the loopback operator API independently from the UDP tunnel socket.
///
/// [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] `network.listen_addr` is the
/// privacy tunnel's UDP endpoint and cannot accept this HTTP request. Reusing
/// only the configured operator API port preserves custom deployments while
/// ensuring the CLI never follows a non-loopback bind address.
fn directory_replica_operator_smoke_url(config: &ServerConfig, operation: &str) -> String {
    format!(
        "http://127.0.0.1:{}/api/discovery/directory/{operation}",
        config.memchain.api_listen_addr.port(),
    )
}

#[cfg(test)]
mod tests;
