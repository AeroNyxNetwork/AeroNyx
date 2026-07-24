// ============================================
// File: crates/aeronyx-server/src/main.rs
// ============================================
//! # AeroNyx Server Entry Point
//!
//! ## Modification Reason
//! - Added MemChain status display in `status` command (AOF file size,
//!   mode, API address).
//! - Added MemChain config display in `validate` command.
//! - v2.5.3+Security / v1.0.0-MultiTenant: Server::new() gains third
//!   argument `config_path: Option<PathBuf>` for auto-generated secret
//!   persistence. cmd_start passes Some(config_path.clone()) so that
//!   api_secret and jwt_secret are written back to disk on first startup.
//! - v1.1.0-SectionSafeAuth: resolve and inject `memchain.api_secret` before
//!   constructing Server, closing the first-start unauthenticated window.
//! - v1.2.0-DirectoryReplicaQuarantineResolution: add host-local incident
//!   inspection and node-identity-signed compare-and-swap resolution commands.
//! - v1.3.0-AofIntegrityCommand: add a read-only, privacy-safe MemChain AOF
//!   verification command for framing, semantic, Merkle, and ancestry checks.
//!
//! ## Last Modified
//! v0.1.0 - Initial CLI implementation
//! v0.2.0 - Added register command, simplified user flow
//! v0.3.0 - Added MemChain status and config display
//! v1.0.0-MultiTenant - Pass config_path to Server::new() (3rd argument)
//! v0.3.0-DiscoveryBootstrap - Show discovery bootstrap config in validate
//! v1.1.0-SectionSafeAuth - Resolve/migrate API secret before server startup
//! v1.2.0-DirectoryReplicaQuarantineResolution - Add audited host-local
//! quarantine inspection and resolution without exposing a mutation API
//! v1.3.0-AofIntegrityCommand - Add aggregate-only `memchain verify-aof`

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::{Parser, Subcommand};
use rand::RngCore;
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::api::auth::ensure_api_secret;
use aeronyx_server::management::models::StoredNodeInfo;
use aeronyx_server::services::{
    AofWriter, DirectoryReplicaResolutionCommand, DirectoryReplicaStore, DirectoryReplicaTip,
};
use aeronyx_server::{ManagementClient, Server, ServerConfig};

// ============================================
// CLI Definition
// ============================================

/// AeroNyx Privacy Network Server
#[derive(Parser, Debug)]
#[command(name = "aeronyx-server")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Register this node with AeroNyx network
    Register {
        /// Registration code from dashboard (e.g., NYX-1234-ABCDE)
        #[arg(short = 'C', long)]
        code: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// CMS API URL (usually not needed, uses default)
        #[arg(long, hide = true)]
        cms_url: Option<String>,
    },

    /// Start the server
    Start {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Check node registration status
    Status {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Validate configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Inspect MemChain persistence without exposing memory contents
    #[command(subcommand)]
    Memchain(MemchainCommands),

    /// Show node public key (for troubleshooting)
    #[command(hide = true)]
    Pubkey {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Output format: base64 (default), hex
        #[arg(long, default_value = "hex")]
        format: String,
    },

    /// Inspect or resolve a quarantined Directory Replica producer locally
    #[command(subcommand)]
    DirectoryReplica(DirectoryReplicaCommands),
}

#[derive(Subcommand, Debug)]
enum MemchainCommands {
    /// Verify AOF framing, content IDs, Merkle roots, and Block ancestry
    VerifyAof {
        /// Optional AOF path override
        #[arg(long)]
        path: Option<PathBuf>,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum DirectoryReplicaCommands {
    /// Verify an incident and print its exact compare-and-swap state
    InspectIncident {
        /// Content-addressed incident digest (64 hexadecimal characters)
        #[arg(long)]
        digest: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Resume one exact accepted prefix after explicit operator review
    ResolveQuarantine {
        /// Content-addressed incident digest
        #[arg(long)]
        digest: String,

        /// Quarantined producer identity
        #[arg(long)]
        producer: String,

        /// Accepted prefix height printed by `inspect-incident`
        #[arg(long)]
        expected_tip_height: u64,

        /// Accepted prefix hash printed by `inspect-incident`
        #[arg(long)]
        expected_tip_hash: String,

        /// Quarantine kind printed by `inspect-incident`
        #[arg(long)]
        expected_kind: String,

        /// Previous linked resolution digest, when one exists
        #[arg(long)]
        expected_previous_resolution_digest: Option<String>,

        /// Must exactly repeat `--digest` to prevent accidental execution
        #[arg(long)]
        confirm_incident: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },
}

// ============================================
// Main
// ============================================

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_logging("info");

    let result = match cli.command {
        Commands::Register {
            code,
            config,
            cms_url,
        } => cmd_register(code, config, cms_url).await,
        Commands::Start { config } => cmd_start(config).await,
        Commands::Status { config } => cmd_status(config).await,
        Commands::Validate { config } => cmd_validate(config).await,
        Commands::Memchain(command) => cmd_memchain(command).await,
        Commands::Pubkey { config, format } => cmd_pubkey(config, format).await,
        Commands::DirectoryReplica(command) => cmd_directory_replica(command).await,
    };

    if let Err(e) = result {
        error!("{}", e);
        std::process::exit(1);
    }
}

// ============================================
// Commands
// ============================================

/// Registers node with CMS.
async fn cmd_register(
    code: String,
    config_path: PathBuf,
    cms_url_override: Option<String>,
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
            println!("   rm {}", node_info_path);
            return Ok(());
        }
    }

    let identity = if key_path.exists() {
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

    let client = ManagementClient::new(mgmt_config.clone(), identity);

    println!("📡 Connecting to AeroNyx network...");
    println!();

    match client.register_node(&code).await {
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
            println!("❌ Registration failed: {}", e);
            println!();
            println!("Please check:");
            println!("  • Is the registration code correct?");
            println!("  • Has the code expired? (codes expire in 15 minutes)");
            println!("  • Is there network connectivity?");
            println!();
            println!("Get a new code from: https://dashboard.aeronyx.network");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Starts the server.
///
/// v1.0.0-MultiTenant: passes `Some(config_path.clone())` to Server::new()
/// so auto-generated api_secret and jwt_secret are persisted to the config
/// file on first SaaS startup.
async fn cmd_start(config_path: PathBuf) -> anyhow::Result<()> {
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
        println!("  1. Get a registration code from https://dashboard.aeronyx.network");
        println!("  2. Run: aeronyx-server register --code <YOUR_CODE>");
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
            println!("  rm {}", node_info_path);
            println!("  aeronyx-server register --code <YOUR_CODE>");
            std::process::exit(1);
        }
    };

    let identity = if key_path.exists() {
        load_key(&key_path).await?
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
async fn cmd_status(config_path: PathBuf) -> anyhow::Result<()> {
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

    // Check key file
    if key_path.exists() {
        match load_key(&key_path).await {
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
                        println!("   AOF Size:      {:.1} KB", size_kb);
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
async fn cmd_validate(config_path: PathBuf) -> anyhow::Result<()> {
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
        println!("   Public:     {}", ep);
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
    println!("Discovery:");
    println!("   Enabled:          {}", config.discovery.enabled);
    if let Some(path) = &config.discovery.bootstrap_snapshot_path {
        println!("   Snapshot Path:    {}", path);
    }
    if let Some(url) = &config.discovery.bootstrap_snapshot_url {
        println!("   Snapshot URL:     {}", url);
        println!(
            "   Fetch Timeout:    {}s",
            config.discovery.fetch_timeout_secs
        );
    }
    println!();

    Ok(())
}

/// Runs read-only `MemChain` operator commands.
async fn cmd_memchain(command: MemchainCommands) -> anyhow::Result<()> {
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

/// Shows node public key (hidden command for troubleshooting).
async fn cmd_pubkey(config_path: PathBuf, format: String) -> anyhow::Result<()> {
    let config = load_or_default_config(&config_path).await;
    let key_path = PathBuf::from(&config.server_key.key_file);

    if !key_path.exists() {
        println!("❌ Node key not found. Register first:");
        println!("   aeronyx-server register --code <YOUR_CODE>");
        std::process::exit(1);
    }

    let identity = load_key(&key_path).await?;

    match format.as_str() {
        "base64" => println!("{}", identity.public_key()),
        _ => println!("{}", hex::encode(identity.public_key_bytes())),
    }

    Ok(())
}

/// Runs privileged Directory Replica operations without a network endpoint.
async fn cmd_directory_replica(command: DirectoryReplicaCommands) -> anyhow::Result<()> {
    match command {
        DirectoryReplicaCommands::InspectIncident { digest, config } => {
            cmd_directory_replica_inspect(&config, &digest).await
        }
        DirectoryReplicaCommands::ResolveQuarantine {
            digest,
            producer,
            expected_tip_height,
            expected_tip_hash,
            expected_kind,
            expected_previous_resolution_digest,
            confirm_incident,
            config,
        } => {
            let request = DirectoryReplicaResolveRequest {
                digest,
                producer,
                expected_tip_height,
                expected_tip_hash,
                expected_kind,
                expected_previous_resolution_digest,
                confirm_incident,
            };
            cmd_directory_replica_resolve(&config, &request).await
        }
    }
}

#[derive(Debug)]
struct DirectoryReplicaResolveRequest {
    digest: String,
    producer: String,
    expected_tip_height: u64,
    expected_tip_hash: String,
    expected_kind: String,
    expected_previous_resolution_digest: Option<String>,
    confirm_incident: String,
}

async fn cmd_directory_replica_inspect(
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

async fn cmd_directory_replica_resolve(
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

async fn open_directory_replica_store(
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
    let key_path = PathBuf::from(&config.server_key.key_file);
    anyhow::ensure!(
        key_path.exists(),
        "node identity key not found: {}",
        key_path.display()
    );
    let identity = load_key(&key_path).await?;
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

// ============================================
// Helper Functions
// ============================================

fn init_logging(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .try_init()
        .ok();
}

fn parse_hex32(value: &str, field: &str) -> anyhow::Result<[u8; 32]> {
    anyhow::ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{field} must contain exactly 64 hexadecimal characters"
    );
    let decoded = hex::decode(value).with_context(|| format!("invalid {field}"))?;
    decoded
        .try_into()
        .map_err(|_| anyhow::anyhow!("{field} must decode to exactly 32 bytes"))
}

fn unix_timestamp() -> anyhow::Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

async fn load_or_default_config(path: &PathBuf) -> ServerConfig {
    if path.exists() {
        ServerConfig::load(path).await.unwrap_or_default()
    } else {
        ServerConfig::default()
    }
}

async fn load_key(path: &PathBuf) -> anyhow::Result<IdentityKeyPair> {
    let content = tokio::fs::read_to_string(path).await?;
    let key_data: KeyFile = serde_json::from_str(&content)?;

    let private_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &key_data.private_key,
    )?;

    let identity = IdentityKeyPair::from_bytes(&private_bytes)?;
    Ok(identity)
}

async fn save_key(identity: &IdentityKeyPair, path: &PathBuf) -> anyhow::Result<()> {
    use base64::Engine;

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let key_data = KeyFile {
        version: "1.0".to_string(),
        key_type: "ed25519".to_string(),
        public_key: base64::engine::general_purpose::STANDARD.encode(identity.public_key_bytes()),
        private_key: base64::engine::general_purpose::STANDARD.encode(identity.to_bytes()),
        created_at: chrono_lite_timestamp(),
    };

    let content = serde_json::to_string_pretty(&key_data)?;
    tokio::fs::write(path, content).await?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(path).await?.permissions();
        perms.set_mode(0o600);
        tokio::fs::set_permissions(path, perms).await?;
    }

    Ok(())
}

fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}Z", duration.as_secs())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyFile {
    version: String,
    key_type: String,
    public_key: String,
    private_key: String,
    created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_replica_resolution_cli_requires_explicit_cas_fields() {
        let digest = "11".repeat(32);
        let producer = "22".repeat(32);
        let tip_hash = "33".repeat(32);
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "resolve-quarantine",
            "--digest",
            &digest,
            "--producer",
            &producer,
            "--expected-tip-height",
            "7",
            "--expected-tip-hash",
            &tip_hash,
            "--expected-kind",
            "signed_tip_fork",
            "--confirm-incident",
            &digest,
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::ResolveQuarantine {
            expected_tip_height,
            expected_previous_resolution_digest,
            ..
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(expected_tip_height, 7);
        assert_eq!(expected_previous_resolution_digest, None);
    }

    #[test]
    fn strict_hex_parser_rejects_ambiguous_or_unbounded_identifiers() {
        assert_eq!(parse_hex32(&"a5".repeat(32), "test").unwrap(), [0xa5; 32]);
        assert!(parse_hex32(&"a5".repeat(31), "test").is_err());
        assert!(parse_hex32(&format!("{}gg", "a5".repeat(31)), "test").is_err());
    }

    #[test]
    fn memchain_verify_aof_cli_accepts_explicit_read_only_path() {
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "memchain",
            "verify-aof",
            "--path",
            "/tmp/test.memchain",
        ])
        .unwrap();
        let Commands::Memchain(MemchainCommands::VerifyAof { path, .. }) = cli.command else {
            panic!("unexpected CLI command")
        };
        assert_eq!(path, Some(PathBuf::from("/tmp/test.memchain")));
    }
}
