// ============================================
// File: crates/aeronyx-server/src/commands/helpers.rs
// ============================================
//! # Shared command helpers
//!
//! Owns the helpers shared by several subcommands: logging initialization,
//! strict 32-byte hex parsing, the Unix clock (both error-reporting forms),
//! best-effort config loading, and node-identity loading (configured key file
//! or dstack KMS) plus the JSON key-file format.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_server::config::ServerKeySource;
use aeronyx_server::tee::DstackClient;
use aeronyx_server::ServerConfig;

// ============================================
// Helper Functions
// ============================================

pub fn init_logging(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .try_init()
        .ok();
}

pub(super) fn parse_hex32(value: &str, field: &str) -> anyhow::Result<[u8; 32]> {
    anyhow::ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{field} must contain exactly 64 hexadecimal characters"
    );
    let decoded = hex::decode(value).with_context(|| format!("invalid {field}"))?;
    decoded
        .try_into()
        .map_err(|_| anyhow::anyhow!("{field} must decode to exactly 32 bytes"))
}

pub(super) fn unix_timestamp() -> anyhow::Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

pub(super) fn unix_timestamp_now() -> anyhow::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")
        .map(|duration| duration.as_secs())
}

pub(super) async fn load_or_default_config(path: &PathBuf) -> ServerConfig {
    if path.exists() {
        ServerConfig::load(path).await.unwrap_or_default()
    } else {
        ServerConfig::default()
    }
}

/// Loads the node identity from its configured source.
///
/// [TEE-DSTACK 2026-10-09 by Claude] The single identity entry point for
/// every command, so a dstack node never silently reads or creates a key file.
pub(super) async fn load_node_identity(config: &ServerConfig) -> anyhow::Result<IdentityKeyPair> {
    match config.server_key.source {
        ServerKeySource::File => {
            let key_path = PathBuf::from(&config.server_key.key_file);
            anyhow::ensure!(
                key_path.exists(),
                "node identity key not found: {}",
                key_path.display()
            );
            load_key(&key_path).await
        }
        ServerKeySource::Dstack => {
            let seed = DstackClient::new(&config.server_key.dstack_socket)
                .derive_identity_seed()
                .await
                .context("derive node identity from dstack KMS")?;
            Ok(IdentityKeyPair::from_bytes(&seed)?)
        }
    }
}

pub(super) async fn load_key(path: &PathBuf) -> anyhow::Result<IdentityKeyPair> {
    let content = tokio::fs::read_to_string(path).await?;
    let key_data: KeyFile = serde_json::from_str(&content)?;

    let private_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &key_data.private_key,
    )?;

    let identity = IdentityKeyPair::from_bytes(&private_bytes)?;
    Ok(identity)
}

pub(super) async fn save_key(identity: &IdentityKeyPair, path: &PathBuf) -> anyhow::Result<()> {
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
mod tests;
