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
//! - Server key management
//! - Server execution
//!
//! ## Usage
//! ```bash
//! # Start with default config
//! aeronyx-server start
//!
//! # Start with custom config
//! aeronyx-server start --config /path/to/config.toml
//!
//! # Generate new server key
//! aeronyx-server keygen --output /etc/aeronyx/server_key.json
//!
//! # Show server public key
//! aeronyx-server pubkey --key /etc/aeronyx/server_key.json
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Server requires root or CAP_NET_ADMIN for TUN
//! - Key files should have restricted permissions (600)
//! - Use systemd for production deployments
//!
//! ## Last Modified
//! v0.1.0 - Initial CLI implementation

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::{Server, ServerConfig};

// ============================================
// CLI Definition
// ============================================

/// AeroNyx Privacy Network Server
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
    /// Start the server
    Start {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Path to server key file
        #[arg(short, long, default_value = "/etc/aeronyx/server_key.json")]
        key: PathBuf,
    },

    /// Generate a new server key pair
    Keygen {
        /// Output path for the key file
        #[arg(short, long, default_value = "/etc/aeronyx/server_key.json")]
        output: PathBuf,

        /// Force overwrite if file exists
        #[arg(short, long)]
        force: bool,
    },

    /// Display the server's public key
    Pubkey {
        /// Path to server key file
        #[arg(short, long, default_value = "/etc/aeronyx/server_key.json")]
        key: PathBuf,
    },

    /// Validate configuration file
    Validate {
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
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging (can be overridden by config in start command)
    init_logging("info");

    // Execute command
    let result = match cli.command {
        Commands::Start { config, key } => cmd_start(config, key).await,
        Commands::Keygen { output, force } => cmd_keygen(output, force).await,
        Commands::Pubkey { key } => cmd_pubkey(key).await,
        Commands::Validate { config } => cmd_validate(config).await,
    };

    // Handle errors
    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

// ============================================
// Commands
// ============================================

/// Starts the server.
async fn cmd_start(config_path: PathBuf, key_path: PathBuf) -> anyhow::Result<()> {
    info!("Loading configuration from: {}", config_path.display());

    // Load configuration
    let config = if config_path.exists() {
        ServerConfig::load(&config_path).await?
    } else {
        info!("Config file not found, using defaults");
        ServerConfig::default()
    };

    // Re-initialize logging with config level
    init_logging(&config.logging.level);

    // Load or generate server key
    let identity = if key_path.exists() {
        info!("Loading server key from: {}", key_path.display());
        load_key(&key_path).await?
    } else if config.server_key.auto_generate {
        info!("Generating new server key");
        let identity = IdentityKeyPair::generate();
        
        // Try to save it (may fail if directory doesn't exist)
        if let Err(e) = save_key(&identity, &key_path).await {
            info!("Could not save generated key: {} (continuing anyway)", e);
        }
        
        identity
    } else {
        anyhow::bail!("Server key file not found: {}", key_path.display());
    };

    info!("Server public key: {}", identity.public_key());

    // Create and run server
    let server = Server::new(config, identity);
    server.run().await?;

    Ok(())
}

/// Generates a new server key.
async fn cmd_keygen(output: PathBuf, force: bool) -> anyhow::Result<()> {
    // Check if file exists
    if output.exists() && !force {
        anyhow::bail!(
            "Key file already exists: {}. Use --force to overwrite.",
            output.display()
        );
    }

    // Generate key
    let identity = IdentityKeyPair::generate();

    // Save key
    save_key(&identity, &output).await?;

    println!("Generated new server key:");
    println!("  File: {}", output.display());
    println!("  Public key: {}", identity.public_key());

    Ok(())
}

/// Displays the server's public key.
async fn cmd_pubkey(key_path: PathBuf) -> anyhow::Result<()> {
    let identity = load_key(&key_path).await?;
    println!("{}", identity.public_key());
    Ok(())
}

/// Validates a configuration file.
async fn cmd_validate(config_path: PathBuf) -> anyhow::Result<()> {
    let config = ServerConfig::load(&config_path).await?;
    
    println!("Configuration is valid:");
    println!("  Listen address: {}", config.network.listen_addr);
    println!("  TUN device: {}", config.tunnel.device_name);
    println!("  IP range: {}", config.tunnel.ip_range);
    println!("  Max sessions: {}", config.limits.max_sessions);

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
        .ok(); // Ignore error if already initialized
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

/// Returns current timestamp as ISO 8601 string (simple implementation).
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    
    let secs = duration.as_secs();
    // Simple UTC timestamp (not fully ISO 8601 compliant but good enough)
    format!("{}Z", secs)
}

/// Server key file format.
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyFile {
    version: String,
    key_type: String,
    public_key: String,
    private_key: String,
    created_at: String,
}
