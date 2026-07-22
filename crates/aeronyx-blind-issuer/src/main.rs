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
//! - Applies a bounded caller deadline without cancelling custody operations.
//! - Applies fail-closed circuit-breaker policy for unhealthy custody backends.
//! - Atomically reloads validated rotating key epochs on Unix SIGHUP.
//! - Preflights the complete startup material without opening a listener.
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
//! Last Modified: v0.6.0-BlindIssuerBinary - Added one shared startup-material
//! validation path for serving and offline configuration preflight.
//! ============================================

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};
use clap::{Parser, Subcommand};
use rand::RngCore;
use sha2::{Digest, Sha256};
use tracing::info;
#[cfg(unix)]
use tracing::warn;
use zeroize::Zeroizing;

use aeronyx_blind_issuer::{
    build_router_with_runtime, BlindIssuerApiPolicy, BlindIssuerConfig, BlindIssuerRuntime,
    BlindSigner, ConfigError,
};

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
    /// Validate all startup configuration and custody inputs without serving.
    CheckConfig {
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
        Command::CheckConfig { config } => check_config(&config)?,
        Command::GenerateKey { output, bits } => generate_key(&output, bits)?,
        Command::GenerateAuthToken { output } => generate_auth_token(&output)?,
    }
    Ok(())
}

// [BLIND-ISSUER-PREFLIGHT 2026-07-23 by Codex] Keep `serve` and offline
// validation on one loader so a preflight can never accept weaker inputs than
// the process that later owns the private-key boundary.
struct StartupMaterial {
    config: BlindIssuerConfig,
    listen_addr: SocketAddr,
    auth_token: Zeroizing<Vec<u8>>,
    signer: Arc<BlindSigner>,
}

fn load_startup_material(path: &Path) -> Result<StartupMaterial, ConfigError> {
    let config = BlindIssuerConfig::load(path)?;
    let listen_addr = config.listen_addr()?;
    let auth_token = config.load_auth_token()?;
    let signer = Arc::new(BlindSigner::from_config(&config)?);
    if !signer.has_active_key(now_millis()) {
        return Err(ConfigError::InvalidPolicy(
            "at least one issuer key must be active at startup",
        ));
    }
    Ok(StartupMaterial {
        config,
        listen_addr,
        auth_token,
        signer,
    })
}

fn check_config(path: &Path) -> Result<(), ConfigError> {
    let material = load_startup_material(path)?;
    info!(
        listen_addr = %material.listen_addr,
        key_count = material.signer.key_count(),
        "blind issuer configuration valid"
    );
    Ok(())
}

async fn serve(path: &Path) -> Result<(), Box<dyn Error>> {
    let StartupMaterial {
        config,
        listen_addr,
        auth_token,
        signer,
    } = load_startup_material(path)?;
    let key_count = signer.key_count();
    let runtime = BlindIssuerRuntime::new(signer);
    let policy = BlindIssuerApiPolicy::new(config.max_requests_per_second, config.max_in_flight)
        .with_signing_timeout(config.signing_timeout())
        .with_circuit_breaker(config.circuit_failure_threshold, config.circuit_cooldown());
    let router = build_router_with_runtime(runtime.clone(), auth_token, policy);
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(
        %listen_addr,
        key_count,
        signing_timeout_ms = config.signing_timeout_ms,
        circuit_failure_threshold = config.circuit_failure_threshold,
        circuit_cooldown_ms = config.circuit_cooldown_ms,
        "blind issuer ready"
    );
    #[cfg(unix)]
    let reload_task = tokio::spawn(reload_on_hangup(path.to_path_buf(), config, runtime));
    let serve_result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    #[cfg(unix)]
    {
        reload_task.abort();
        let _ = reload_task.await;
    }
    serve_result?;
    Ok(())
}

#[cfg(unix)]
async fn reload_on_hangup(
    path: PathBuf,
    startup_config: BlindIssuerConfig,
    runtime: BlindIssuerRuntime,
) {
    use tokio::signal::unix::{signal, SignalKind};

    let Ok(mut hangup) = signal(SignalKind::hangup()) else {
        warn!("blind issuer key reload signal unavailable");
        return;
    };
    while hangup.recv().await.is_some() {
        let reload_path = path.clone();
        let baseline = startup_config.clone();
        let candidate =
            tokio::task::spawn_blocking(move || load_reload_candidate(&reload_path, &baseline))
                .await;
        let Ok(Ok(signer)) = candidate else {
            // [BLIND-ISSUER-RELOAD 2026-07-23 by Codex] Do not log provider,
            // path, key, or parsing details from the custody boundary.
            runtime.record_reload_rejection(now_millis());
            warn!("blind issuer key reload rejected");
            continue;
        };
        let key_count = signer.key_count();
        if let Ok(signer_generation) = runtime.replace_signer(signer, now_millis()) {
            info!(
                signer_generation,
                key_count, "blind issuer key reload applied"
            );
        } else {
            warn!("blind issuer key reload rejected");
        }
    }
}

#[cfg(unix)]
fn load_reload_candidate(
    path: &Path,
    startup_config: &BlindIssuerConfig,
) -> Result<Arc<BlindSigner>, ConfigError> {
    let candidate_config = BlindIssuerConfig::load(path)?;
    if !startup_config.key_reload_compatible_with(&candidate_config) {
        return Err(ConfigError::InvalidPolicy(
            "non-key configuration changes require restart",
        ));
    }
    BlindSigner::from_config(&candidate_config).map(Arc::new)
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

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
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

#[cfg(all(test, unix))]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use std::fs;

    const BACKEND_TOKEN: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFG";

    fn write_test_config(
        path: &Path,
        token_path: &Path,
        private_key_path: &Path,
        not_before_unix_secs: u64,
        expires_at_unix_secs: u64,
    ) {
        let contents = format!(
            r#"listen_addr = "127.0.0.1:19090"
auth_token_file = "{}"
max_requests_per_second = 128
max_in_flight = 8
signing_timeout_ms = 10000
circuit_failure_threshold = 5
circuit_cooldown_ms = 30000

[[keys]]
private_key_der_file = "{}"
not_before_unix_secs = {}
expires_at_unix_secs = {}
max_lease_ttl_secs = 604800
"#,
            token_path.display(),
            private_key_path.display(),
            not_before_unix_secs,
            expires_at_unix_secs,
        );
        fs::write(path, contents).expect("write test config");
    }

    #[test]
    fn startup_preflight_validates_active_custody_material() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let token_path = directory.path().join("backend.token");
        let private_key_path = directory.path().join("issuer.der");
        let config_path = directory.path().join("blind-issuer.toml");
        write_secure_new(&token_path, BACKEND_TOKEN).expect("write auth token");

        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("generate test key");
        let private_der = Zeroizing::new(key_pair.sk.to_der().expect("private DER"));
        write_secure_new(&private_key_path, &private_der).expect("write private key");
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current Unix time")
            .as_secs();

        write_test_config(
            &config_path,
            &token_path,
            &private_key_path,
            now_secs - 60,
            now_secs + 24 * 60 * 60,
        );
        let material = load_startup_material(&config_path).expect("valid startup material");
        assert_eq!(
            material.listen_addr,
            SocketAddr::from(([127, 0, 0, 1], 19090))
        );
        assert_eq!(material.auth_token.as_slice(), BACKEND_TOKEN);
        assert_eq!(material.signer.key_count(), 1);

        write_test_config(
            &config_path,
            &token_path,
            &private_key_path,
            now_secs + 60,
            now_secs + 24 * 60 * 60,
        );
        assert!(matches!(
            load_startup_material(&config_path),
            Err(ConfigError::InvalidPolicy(
                "at least one issuer key must be active at startup"
            ))
        ));
    }
}
