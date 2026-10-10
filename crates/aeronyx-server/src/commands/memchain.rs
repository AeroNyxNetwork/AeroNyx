// ============================================
// File: crates/aeronyx-server/src/commands/memchain.rs
// ============================================
//! # `memchain` commands
//!
//! Owns the read-only `MemChain` operator commands (aggregate-only AOF
//! integrity verification).
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::PathBuf;

use anyhow::Context;

use aeronyx_server::services::AofWriter;
use aeronyx_server::ServerConfig;

use crate::MemchainCommands;

/// Runs read-only `MemChain` operator commands.
pub async fn cmd_memchain(command: MemchainCommands) -> anyhow::Result<()> {
    match command {
        MemchainCommands::VerifyAof { path, config } => {
            cmd_memchain_verify_aof(&config, path).await
        }
    }
}

/// Verifies only aggregate AOF integrity and never prints record contents.
async fn cmd_memchain_verify_aof(
    config_path: &PathBuf,
    path_override: Option<PathBuf>,
) -> anyhow::Result<()> {
    let path = if let Some(path) = path_override {
        path
    } else {
        let config = if config_path.exists() {
            ServerConfig::load(config_path).await?
        } else {
            ServerConfig::default()
        };
        PathBuf::from(config.memchain.aof_path)
    };
    let report = AofWriter::verify(&path)
        .await
        .with_context(|| format!("verify MemChain AOF {}", path.display()))?;

    // [AOF-INTEGRITY-CLI 2026-07-24 by Codex] This output is deliberately
    // aggregate-only. Never add Fact values, identities, hashes, signatures,
    // or record-level offsets to the operator command.
    println!("MemChain AOF integrity");
    println!("  path: {}", path.display());
    println!("  file_bytes: {}", report.file_bytes);
    println!("  valid_bytes: {}", report.valid_bytes);
    println!("  fact_records: {}", report.fact_records);
    println!("  block_records: {}", report.block_records);
    println!("  last_block_height: {}", report.last_block_height);
    println!("  torn_tail_bytes: {}", report.torn_tail_bytes);
    println!(
        "  status: {}",
        if report.is_clean() {
            "verified"
        } else {
            "torn_tail_detected"
        }
    );
    println!("  privacy: aggregate integrity metadata only; no record contents or identities");

    anyhow::ensure!(
        report.is_clean(),
        "AOF has an incomplete physical tail; start the node through the guarded recovery path"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
