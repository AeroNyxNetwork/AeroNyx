// ============================================
// File: crates/aeronyx-server/src/commands/register.rs
// ============================================
//! # `register` command
//!
//! Owns one-time node registration with the CMS: bounded, secret-safe
//! registration-code input (`--code` or `--code-stdin`), operator name/region
//! normalization, and the registration request itself.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::io::BufRead;
use std::path::PathBuf;

use anyhow::Context;
use tracing::info;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::config::ServerKeySource;
use aeronyx_server::management::models::{NodeRegistrationProfile, StoredNodeInfo};
use aeronyx_server::ManagementClient;

use super::helpers::{load_key, load_node_identity, load_or_default_config, save_key};

/// Maximum accepted one-time registration-code length after trimming.
const MAX_REGISTRATION_CODE_BYTES: usize = 128;

/// Validates a registration code without logging or retaining surrounding input.
fn normalize_registration_code(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "registration code cannot be empty");
    anyhow::ensure!(
        value.len() <= MAX_REGISTRATION_CODE_BYTES,
        "registration code cannot exceed {MAX_REGISTRATION_CODE_BYTES} bytes"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "registration code cannot contain control characters"
    );
    Ok(value.to_string())
}

/// Reads at most one bounded registration-code line from an anonymous stream.
fn read_registration_code<R: BufRead>(reader: R) -> anyhow::Result<String> {
    // [REGISTRATION-CODE-STDIN 2026-08-02 by Codex] `take` bounds allocation
    // before UTF-8 validation. A malformed or unbounded pipe must fail closed
    // without copying secret material into logs or process arguments.
    let mut bounded = reader.take((MAX_REGISTRATION_CODE_BYTES + 2) as u64);
    let mut line = String::new();
    bounded
        .read_line(&mut line)
        .context("failed to read registration code from standard input")?;
    normalize_registration_code(&line)
}

/// Resolves the backward-compatible CLI value or the secret-safe stdin mode.
pub fn resolve_registration_code(code: Option<String>, code_stdin: bool) -> anyhow::Result<String> {
    match (code, code_stdin) {
        (Some(code), false) => normalize_registration_code(&code),
        (None, true) => read_registration_code(std::io::stdin().lock()),
        _ => anyhow::bail!("provide exactly one of --code or --code-stdin"),
    }
}

/// Validates an optional node name before it reaches the one-time bind API.
fn normalize_registration_name(value: Option<String>) -> anyhow::Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "node name cannot be empty");
    anyhow::ensure!(
        value.chars().count() <= 100,
        "node name cannot exceed 100 characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "node name cannot contain control characters"
    );
    Ok(Some(value.to_string()))
}

/// Normalizes an optional ISO 3166-1 alpha-2 region code.
fn normalize_registration_region(value: Option<String>) -> anyhow::Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    anyhow::ensure!(
        value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_alphabetic()),
        "region must be a two-letter ISO 3166-1 alpha-2 code"
    );
    Ok(Some(value.to_ascii_uppercase()))
}

/// Registers node with CMS.
pub async fn cmd_register(
    code: String,
    config_path: PathBuf,
    cms_url_override: Option<String>,
    node_name: Option<String>,
    region: Option<String>,
    public_vpn: bool,
) -> anyhow::Result<()> {
    println!("🚀 AeroNyx Node Registration");
    println!("════════════════════════════════════════");
    println!();

    let config = load_or_default_config(&config_path).await;
    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    if std::path::Path::new(node_info_path).exists() {
        if let Ok(info) = StoredNodeInfo::load(node_info_path) {
            println!("⚠️  This node is already registered!");
            println!();
            println!("   Node ID:  {}", info.node_id);
            println!("   Name:     {}", info.name);
            println!("   Owner:    {}", info.owner_wallet);
            println!();
            println!("If you want to re-register, delete the file:");
            println!("   rm {node_info_path}");
            return Ok(());
        }
    }

    let identity = if config.server_key.source == ServerKeySource::Dstack {
        info!("Deriving node key from dstack KMS...");
        load_node_identity(&config).await?
    } else if key_path.exists() {
        info!("Loading existing node key...");
        load_key(&key_path).await?
    } else {
        info!("Generating secure node key...");
        let identity = IdentityKeyPair::generate();
        save_key(&identity, &key_path).await?;
        identity
    };

    let mut mgmt_config = config.management.clone();
    if let Some(url) = cms_url_override {
        mgmt_config.cms_url = url;
    }

    // [MANAGEMENT-CLIENT-STARTUP 2026-08-12 by Codex] Registration is a CLI
    // transaction: connector initialization must return an actionable error
    // before any remote request or local registration record is created.
    let client = ManagementClient::new(mgmt_config.clone(), identity)
        .context("Failed to initialize management HTTP client")?;
    let registration_profile = NodeRegistrationProfile {
        name: normalize_registration_name(node_name)?,
        port: Some(config.listen_addr().port()),
        region_code: normalize_registration_region(region)?,
        visibility: public_vpn.then(|| "public".to_string()),
        is_vpn_node: Some(true),
    };

    println!("📡 Connecting to AeroNyx network...");
    println!();

    match client
        .register_node_with_profile(&code, registration_profile)
        .await
    {
        Ok(node_info) => {
            let stored = StoredNodeInfo {
                node_id: node_info.id.clone(),
                owner_wallet: node_info.owner_wallet.clone(),
                name: node_info.name.clone(),
                registered_at: node_info.created_at.clone(),
            };
            stored.save(&mgmt_config.node_info_path)?;

            println!("✅ Registration successful!");
            println!();
            println!("════════════════════════════════════════");
            println!("   Node ID:  {}", node_info.id);
            println!("   Name:     {}", node_info.name);
            println!("   Owner:    {}", node_info.owner_wallet);
            println!("════════════════════════════════════════");
            println!();
            println!("🎉 Your node is ready! Start it with:");
            println!();
            println!("   aeronyx-server start");
            println!();
        }
        Err(e) => {
            println!("❌ Registration failed: {e}");
            println!();
            println!("Please check:");
            println!("  • Is the registration code correct?");
            println!("  • Has the code expired? (codes expire in 15 minutes)");
            println!("  • Is there network connectivity?");
            println!();
            println!("Get a new code from: https://app.aeronyx.network");
            std::process::exit(1);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
