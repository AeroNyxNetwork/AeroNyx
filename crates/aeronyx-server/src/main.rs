// ============================================
// File: crates/aeronyx-server/src/main.rs
// ============================================
//! # AeroNyx Server Entry Point
//!
//! ## Creation Reason
//! Main entry point for the AeroNyx privacy network server binary.
//! Handles CLI parsing, logging setup, and server initialization.
//!
//! ## Main Functionality
//! - CLI argument parsing with clap
//! - Logging initialization with tracing
//! - Configuration loading
//! - Node registration with CMS
//! - Server execution
//!
//! ## Usage
//! ```bash
//! # Step 1: Register node (get code from web dashboard)
//! aeronyx-server register --code NYX-1234-ABCDE
//!
//! # Step 2: Start server
//! aeronyx-server start
//!
//! # Other commands
//! aeronyx-server status              # Check registration status
//! aeronyx-server validate            # Validate config file
//! aeronyx-server pubkey              # Show node public key (for troubleshooting)
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Server requires root or CAP_NET_ADMIN for TUN
//! - Node MUST be registered before starting
//! - Key is auto-generated during registration (user doesn't need to know)
//! - Use systemd for production deployments
//!
//! ## Last Modified
//! v0.1.0 - Initial CLI implementation
//! v0.2.0 - Added register command, simplified user flow

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::{Server, ServerConfig, ManagementClient};
use aeronyx_server::management::models::StoredNodeInfo;

// ============================================
// CLI Definition
// ============================================

/// AeroNyx Privacy Network Server
///
/// Quick Start:
///   1. Get registration code from https://dashboard.aeronyx.network
///   2. Run: aeronyx-server register --code <YOUR_CODE>
///   3. Run: aeronyx-server start
#[derive(Parser, Debug)]
#[command(name = "aeronyx-server")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Register this node with AeroNyx network
    ///
    /// Get your registration code from the AeroNyx dashboard.
    /// This command will automatically generate a secure key pair
    /// and bind this node to your account.
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
    ///
    /// Node must be registered first. Run 'register' command if not done yet.
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
}

// ============================================
// Main
// ============================================

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    init_logging("info");

    // Execute command
    let result = match cli.command {
        Commands::Register { code, config, cms_url } => {
            cmd_register(code, config, cms_url).await
        }
        Commands::Start { config } => cmd_start(config).await,
        Commands::Status { config } => cmd_status(config).await,
        Commands::Validate { config } => cmd_validate(config).await,
        Commands::Pubkey { config, format } => cmd_pubkey(config, format).await,
    };

    // Handle errors
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

    // Load config
    let config = load_or_default_config(&config_path).await;
    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    // Check if already registered
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

    // Generate or load key (user doesn't need to know this detail)
    let identity = if key_path.exists() {
        info!("Loading existing node key...");
        load_key(&key_path).await?
    } else {
        info!("Generating secure node key...");
        let identity = IdentityKeyPair::generate();
        save_key(&identity, &key_path).await?;
        identity
    };

    // Build management config
    let mut mgmt_config = config.management.clone();
    if let Some(url) = cms_url_override {
        mgmt_config.cms_url = url;
    }

    // Create management client and register
    let client = ManagementClient::new(mgmt_config.clone(), identity);

    println!("📡 Connecting to AeroNyx network...");
    println!();

    match client.register_node(&code).await {
        Ok(node_info) => {
            // Save node info
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
async fn cmd_start(config_path: PathBuf) -> anyhow::Result<()> {
    info!("Starting AeroNyx server...");

    // Load configuration
    let config = if config_path.exists() {
        ServerConfig::load(&config_path).await?
    } else {
        info!("Config file not found, using defaults");
        ServerConfig::default()
    };

    // Re-initialize logging with config level
    init_logging(&config.logging.level);

    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    // ========================================
    // Check registration (MANDATORY)
    // ========================================
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

    // Load registration info
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

    // Load server key
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

    // Create and run server
    let server = Server::new(config, identity);
    server.run().await?;

    Ok(())
}

/// Shows node registration status.
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
                println!("   Public Key:    {}", hex::encode(identity.public_key_bytes()));
            }
            Err(_) => {
                println!("Server Key:    ⚠️  Invalid or corrupted");
            }
        }
    } else {
        println!("Server Key:    ❌ Missing");
    }

    println!();
    println!("════════════════════════════════════════");
    println!();

    Ok(())
}

/// Validates configuration file.
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
    println!("VPN:");
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
        "hex" | _ => println!("{}", hex::encode(identity.public_key_bytes())),
    }
    
    Ok(())
}

// ============================================
// Helper Functions
// ============================================

/// Initializes the tracing subscriber.
fn init_logging(level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .try_init()
        .ok();
}

/// Loads config or returns default.
async fn load_or_default_config(path: &PathBuf) -> ServerConfig {
    if path.exists() {
        ServerConfig::load(path).await.unwrap_or_default()
    } else {
        ServerConfig::default()
    }
}

/// Loads a server key from a JSON file.
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

/// Saves a server key to a JSON file.
async fn save_key(identity: &IdentityKeyPair, path: &PathBuf) -> anyhow::Result<()> {
    use base64::Engine;

    // Ensure parent directory exists
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

    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(path).await?.permissions();
        perms.set_mode(0o600);
        tokio::fs::set_permissions(path, perms).await?;
    }

    Ok(())
}

/// Returns current timestamp as ISO 8601 string.
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    
    format!("{}Z", duration.as_secs())
}

/// Server key file format (internal, user doesn't need to know).
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyFile {
    version: String,
    key_type: String,
    public_key: String,
    private_key: String,
    created_at: String,
}
