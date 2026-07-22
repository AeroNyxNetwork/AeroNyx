// ============================================
// File: crates/aeronyx-blind-issuer/src/main.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] Standalone custody process and explicit,
// non-overwriting key/token provisioning commands.
//! # `AeroNyx` Blind Issuer Binary
//!
//! ## Creation Reason
//! Provide a separately deployable process for RFC 9474 private-key custody.
//!
//! ## Main Functionality
//! - Serves the loopback-only authenticated blind-signing API.
//! - Generates owner-only RSA private-key files and public registration data.
//! - Generates owner-only random backend bearer-token files.
//!
//! ## Calling Relationships
//! `config.rs` validates startup; `signer.rs` holds keys; `api.rs` serves the
//! internal protocol. This binary must not be linked into the storage node.
//!
//! ## Next Developer Guide
//! Keep key generation explicit and `create_new`; never overwrite custody
//! material. Add systemd hardening in deployment packaging, not runtime code.
//!
//! Last Modified: v0.1.1-BlindIssuerBinary - Graceful SIGTERM shutdown.
//! ============================================

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};
use clap::{Parser, Subcommand};
use rand::RngCore;
use sha2::{Digest, Sha256};
use tracing::info;
use zeroize::Zeroizing;

use aeronyx_blind_issuer::{build_router, BlindIssuerConfig, BlindSigner};

#[derive(Debug, Parser)]
#[command(name = "aeronyx-blind-issuer")]
#[command(about = "Isolated RFC 9474 admission signer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the loopback-only signing process.
    Serve {
        /// TOML configuration path.
        #[arg(long, default_value = "/etc/aeronyx/blind-issuer.toml")]
        config: PathBuf,
    },
    /// Generate a new owner-only RSA private key without overwriting files.
    GenerateKey {
        /// Destination PKCS#8 DER path.
        #[arg(long)]
        output: PathBuf,
        /// RSA modulus size: 2048, 3072, or 4096.
        #[arg(long, default_value_t = 3072)]
        bits: usize,
    },
    /// Generate a random owner-only backend bearer token.
    GenerateAuthToken {
        /// Destination token path.
        #[arg(long)]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aeronyx_blind_issuer=info".into()),
        )
        .json()
        .init();

    match Cli::parse().command {
        Command::Serve { config } => serve(&config).await?,
        Command::GenerateKey { output, bits } => generate_key(&output, bits)?,
        Command::GenerateAuthToken { output } => generate_auth_token(&output)?,
    }
    Ok(())
}

async fn serve(path: &Path) -> Result<(), Box<dyn Error>> {
    let config = BlindIssuerConfig::load(path)?;
    let listen_addr = config.listen_addr()?;
    let auth_token = config.load_auth_token()?;
    let signer = Arc::new(BlindSigner::from_config(&config)?);
    let key_count = signer.key_count();
    let router = build_router(
        signer,
        auth_token,
        config.max_requests_per_second,
        config.max_in_flight,
    );
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(%listen_addr, key_count, "blind issuer ready");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    // [BLIND-ISSUER-HARDENING 2026-07-23 by Codex] systemd sends SIGTERM;
    // waiting only for Ctrl-C would skip Axum's graceful-drain path in service.
    let Ok(mut terminate) = signal(SignalKind::terminate()) else {
        wait_for_ctrl_c().await;
        return;
    };
    tokio::select! {
        () = wait_for_ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    wait_for_ctrl_c().await;
}

async fn wait_for_ctrl_c() {
    if tokio::signal::ctrl_c().await.is_err() {
        std::future::pending::<()>().await;
    }
}

fn generate_key(path: &Path, bits: usize) -> Result<(), Box<dyn Error>> {
    if !matches!(bits, 2048 | 3072 | 4096) {
        return Err("RSA bits must be 2048, 3072, or 4096".into());
    }
    let key_pair = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, bits)?;
    let private_der = Zeroizing::new(key_pair.sk.to_der()?);
    write_secure_new(path, &private_der)?;
    let public_der = key_pair.pk.to_der()?;
    let key_id: [u8; 32] = Sha256::digest(&public_der).into();
    let mut output = std::io::stdout().lock();
    writeln!(output, "issuer_key_id={}", hex::encode(key_id))?;
    writeln!(
        output,
        "public_key_der_base64={}",
        BASE64.encode(public_der)
    )?;
    Ok(())
}

fn generate_auth_token(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut random = Zeroizing::new([0u8; 32]);
    let mut rng = rand::rngs::OsRng;
    rng.fill_bytes(random.as_mut());
    let token = Zeroizing::new(BASE64.encode(random.as_ref()).into_bytes());
    write_secure_new(path, &token)?;
    Ok(())
}

#[cfg(unix)]
fn write_secure_new(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(not(unix))]
fn write_secure_new(_path: &Path, _bytes: &[u8]) -> Result<(), std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "secure issuer file creation requires a Unix platform",
    ))
}
