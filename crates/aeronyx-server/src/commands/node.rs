// ============================================
// File: crates/aeronyx-server/src/commands/node.rs
// ============================================
//! # `start`, `status`, `validate` and `pubkey` commands
//!
//! Owns the node lifecycle commands: starting the server, showing registration
//! and `MemChain` status, validating a configuration file, and the hidden
//! troubleshooting `pubkey` command.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::PathBuf;

use tracing::{error, info};

use aeronyx_server::api::auth::ensure_api_secret;
use aeronyx_server::config::ServerKeySource;
use aeronyx_server::management::models::StoredNodeInfo;
use aeronyx_server::{Server, ServerConfig};

use super::helpers::{init_logging, load_node_identity, load_or_default_config};

/// Starts the server.
///
/// v1.0.0-MultiTenant: passes `Some(config_path.clone())` to Server::new()
/// so auto-generated api_secret and jwt_secret are persisted to the config
/// file on first SaaS startup.
pub async fn cmd_start(config_path: PathBuf) -> anyhow::Result<()> {
    info!("Starting AeroNyx server...");

    let mut config = if config_path.exists() {
        ServerConfig::load(&config_path).await?
    } else {
        info!("Config file not found, using defaults");
        ServerConfig::default()
    };

    init_logging(&config.logging.level);

    // Resolve before constructing Server so first-start admin routes receive
    // the same secret that is persisted to `[memchain]`. The legacy writer ran
    // after config loading and left the current process unauthenticated.
    if config.memchain.is_enabled() {
        let persisted_path = config_path.exists().then_some(config_path.as_path());
        let api_secret = ensure_api_secret(config.memchain.effective_api_secret(), persisted_path)
            .map_err(anyhow::Error::msg)?;
        config.memchain.api_secret = Some(api_secret);
    }

    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    if !std::path::Path::new(node_info_path).exists() {
        println!();
        println!("❌ Node is not registered!");
        println!();
        println!("All nodes must be registered to join the AeroNyx network.");
        println!();
        println!("To register your node:");
        println!("  1. Get a registration code from https://app.aeronyx.network");
        println!("  2. Pipe it privately: printf '%s\\n' '<YOUR_CODE>' | aeronyx-server register --code-stdin");
        println!("     Legacy compatibility: aeronyx-server register --code <YOUR_CODE>");
        println!();
        std::process::exit(1);
    }

    let node_info = match StoredNodeInfo::load(node_info_path) {
        Ok(info) => info,
        Err(e) => {
            error!("Failed to load registration info: {}", e);
            println!();
            println!("❌ Registration data is corrupted.");
            println!();
            println!("Please re-register your node:");
            println!("  rm {node_info_path}");
            println!("  aeronyx-server register --code <YOUR_CODE>");
            std::process::exit(1);
        }
    };

    let identity = if config.server_key.source == ServerKeySource::Dstack || key_path.exists() {
        load_node_identity(&config).await?
    } else {
        println!();
        println!("❌ Server key not found!");
        println!();
        println!("The key file is missing. Please re-register your node:");
        println!("  aeronyx-server register --code <YOUR_CODE>");
        std::process::exit(1);
    };

    info!("════════════════════════════════════════");
    info!("Node ID:    {}", node_info.node_id);
    info!("Node Name:  {}", node_info.name);
    info!("Owner:      {}", node_info.owner_wallet);
    info!("════════════════════════════════════════");

    // v1.0.0-MultiTenant: pass config_path so auto-generated secrets
    // (api_secret, jwt_secret) are written back to disk on first startup.
    let server = Server::new(config, identity, Some(config_path.clone()));
    server.run().await?;

    Ok(())
}

/// Shows node registration status + MemChain status.
pub async fn cmd_status(config_path: PathBuf) -> anyhow::Result<()> {
    let config = load_or_default_config(&config_path).await;
    let node_info_path = &config.management.node_info_path;
    let key_path = PathBuf::from(&config.server_key.key_file);

    println!();
    println!("AeroNyx Node Status");
    println!("════════════════════════════════════════");
    println!();

    // Check registration
    match StoredNodeInfo::load(node_info_path) {
        Ok(info) => {
            println!("Registration:  ✅ Registered");
            println!();
            println!("   Node ID:       {}", info.node_id);
            println!("   Name:          {}", info.name);
            println!("   Owner:         {}", info.owner_wallet);
            println!("   Registered:    {}", info.registered_at);
        }
        Err(_) => {
            println!("Registration:  ❌ Not registered");
            println!();
            println!("Run this command to register:");
            println!("   aeronyx-server register --code <YOUR_CODE>");
            return Ok(());
        }
    }

    println!();

    // Check node identity
    if config.server_key.source == ServerKeySource::Dstack || key_path.exists() {
        match load_node_identity(&config).await {
            Ok(identity) => {
                println!("Server Key:    ✅ Valid");
                println!(
                    "   Public Key:    {}",
                    hex::encode(identity.public_key_bytes())
                );
            }
            Err(_) => {
                println!("Server Key:    ⚠️  Invalid or corrupted");
            }
        }
    } else {
        println!("Server Key:    ❌ Missing");
    }

    println!();

    // MemChain Status
    println!("MemChain:");
    println!("   Mode:          {:?}", config.memchain.mode);

    if config.memchain.is_enabled() {
        println!("   API Address:   {}", config.memchain.api_listen_addr);
        println!("   AOF Path:      {}", config.memchain.aof_path);

        let aof_path = std::path::Path::new(&config.memchain.aof_path);
        if aof_path.exists() {
            match std::fs::metadata(aof_path) {
                Ok(meta) => {
                    let size_kb = meta.len() as f64 / 1024.0;
                    if size_kb < 1024.0 {
                        println!("   AOF Size:      {size_kb:.1} KB");
                    } else {
                        println!("   AOF Size:      {:.2} MB", size_kb / 1024.0);
                    }
                }
                Err(_) => {
                    println!("   AOF Size:      ⚠️  Could not read");
                }
            }
        } else {
            println!("   AOF File:      (not yet created — will be created on first write)");
        }
    } else {
        println!("   Status:        Disabled");
    }

    println!();
    println!("════════════════════════════════════════");
    println!();

    Ok(())
}

/// Validates configuration file + shows MemChain config.
pub async fn cmd_validate(config_path: PathBuf) -> anyhow::Result<()> {
    if !config_path.exists() {
        println!("⚠️  Config file not found: {}", config_path.display());
        println!("   Server will use default values.");
        return Ok(());
    }

    let config = ServerConfig::load(&config_path).await?;

    println!("✅ Configuration is valid");
    println!();
    println!("Network:");
    println!("   Listen:     {}", config.listen_addr());
    if let Some(ep) = &config.network.public_endpoint {
        println!("   Public:     {ep}");
    }
    println!();
    println!("AeroNyx Privacy Protocol:");
    println!("   IP Range:   {}", config.ip_range());
    println!("   Gateway:    {}", config.gateway_ip());
    println!();
    println!("TUN:");
    println!("   Device:     {}", config.device_name());
    println!("   MTU:        {}", config.mtu());
    println!();
    println!("Limits:");
    println!("   Max Connections:  {}", config.max_sessions());
    println!("   Session Timeout:  {}s", config.session_timeout_secs());
    println!();
    println!("MemChain:");
    println!("   Mode:             {:?}", config.memchain.mode);
    if config.memchain.is_enabled() {
        println!("   API Listen:       {}", config.memchain.api_listen_addr);
        println!("   AOF Path:         {}", config.memchain.aof_path);
    }
    println!();
    println!("Relay custody:");
    println!(
        "   Enabled:          {}",
        config.memchain.chat_relay.enabled
    );
    if config.memchain.chat_relay.enabled {
        // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Surface policy during
        // config validation without exposing artifact names or audit contents.
        println!(
            "   Retained backups: {}",
            config
                .memchain
                .chat_relay
                .custody_backup_retention_target_artifacts
        );
        println!(
            "   Retained bytes:   {}",
            config
                .memchain
                .chat_relay
                .custody_backup_retention_target_bytes
        );
        println!(
            "   Partial grace:    {}s",
            config.memchain.chat_relay.custody_backup_partial_grace_secs
        );
    }
    println!();
    println!("Discovery:");
    println!("   Enabled:          {}", config.discovery.enabled);
    if let Some(path) = &config.discovery.bootstrap_snapshot_path {
        println!("   Snapshot Path:    {path}");
    }
    if let Some(url) = &config.discovery.bootstrap_snapshot_url {
        println!("   Snapshot URL:     {url}");
        println!(
            "   Fetch Timeout:    {}s",
            config.discovery.fetch_timeout_secs
        );
    }
    println!();

    Ok(())
}

/// Shows node public key (hidden command for troubleshooting).
pub async fn cmd_pubkey(config_path: PathBuf, format: String) -> anyhow::Result<()> {
    let config = load_or_default_config(&config_path).await;
    let Ok(identity) = load_node_identity(&config).await else {
        println!("❌ Node key not found. Register first:");
        println!("   aeronyx-server register --code <YOUR_CODE>");
        std::process::exit(1);
    };

    match format.as_str() {
        "base64" => println!("{}", identity.public_key()),
        _ => println!("{}", hex::encode(identity.public_key_bytes())),
    }

    Ok(())
}
