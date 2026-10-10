// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/host_io.rs
// ============================================
//! # Relay custody host-local I/O
//!
//! Owns the create-new / bounded artifact file boundary for custody anchors and
//! witness receipts, and loading of the node config, node secret and node
//! identity for relay custody commands.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES;
use aeronyx_server::services::derive_node_secret;
use aeronyx_server::ServerConfig;

use crate::commands::helpers::load_node_identity;

pub(super) fn write_new_relay_custody_anchor(path: &Path, frame: &[u8]) -> anyhow::Result<()> {
    write_new_relay_custody_artifact(
        path,
        frame,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )
}

pub(super) fn write_new_relay_custody_artifact(
    path: &Path,
    frame: &[u8],
    max_frame_bytes: usize,
    artifact_label: &'static str,
) -> anyhow::Result<()> {
    use std::io::Write as _;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= max_frame_bytes,
        "relay custody {artifact_label} violates its write bound"
    );
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to create new relay custody {artifact_label}"))?;

    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Anchors and witness receipts
    // share one create-new, no-final-symlink, file+directory durability
    // boundary. If the exact frame is not durable, remove only the inode this
    // invocation created; a retry can safely reproduce or re-witness it.
    let write_result = (|| -> anyhow::Result<()> {
        file.write_all(frame)
            .map_err(|_| anyhow::anyhow!("unable to write relay custody {artifact_label}"))?;
        file.sync_all()
            .map_err(|_| anyhow::anyhow!("unable to sync relay custody {artifact_label}"))?;
        let final_len = file
            .metadata()
            .map_err(|_| anyhow::anyhow!("unable to inspect relay custody {artifact_label}"))?
            .len();
        anyhow::ensure!(
            final_len == frame.len() as u64,
            "relay custody {artifact_label} changed during publication"
        );
        Ok(())
    })();
    drop(file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }

    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|candidate| !candidate.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let directory = File::open(parent)
            .map_err(|_| anyhow::anyhow!("unable to open custody artifact parent directory"))?;
        directory
            .sync_all()
            .map_err(|_| anyhow::anyhow!("unable to sync custody artifact parent directory"))?;
    }
    Ok(())
}

pub(super) fn read_bounded_relay_custody_anchor(path: &Path) -> anyhow::Result<Vec<u8>> {
    read_bounded_relay_custody_artifact(path, MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES, "audit anchor")
}

pub(super) fn read_bounded_relay_custody_artifact(
    path: &Path,
    max_frame_bytes: usize,
    artifact_label: &'static str,
) -> anyhow::Result<Vec<u8>> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to open relay custody {artifact_label}"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to inspect relay custody {artifact_label}"))?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= max_frame_bytes as u64,
        "relay custody {artifact_label} has an invalid file boundary"
    );

    let capacity = usize::try_from(metadata.len())
        .map_err(|_| anyhow::anyhow!("relay custody {artifact_label} exceeds platform capacity"))?;
    let mut frame = Vec::with_capacity(capacity);
    file.by_ref()
        .take(max_frame_bytes as u64 + 1)
        .read_to_end(&mut frame)
        .map_err(|_| anyhow::anyhow!("unable to read relay custody {artifact_label}"))?;
    let final_len = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to re-inspect relay custody {artifact_label}"))?
        .len();
    anyhow::ensure!(
        frame.len() as u64 == metadata.len()
            && final_len == metadata.len()
            && frame.len() <= max_frame_bytes,
        "relay custody {artifact_label} changed during bounded read"
    );
    Ok(frame)
}

pub(super) async fn load_relay_custody_config(config_path: &Path) -> anyhow::Result<ServerConfig> {
    let config = ServerConfig::load(config_path).await?;
    if !config.memchain.is_chat_relay_enabled() {
        anyhow::bail!("relay custody maintenance requires chat relay to be enabled");
    }
    if config.memchain.chat_relay.db_path == ":memory:" {
        anyhow::bail!("in-memory relay custody has no recoverable backup boundary");
    }
    Ok(config)
}

pub(super) async fn load_relay_custody_node_secret(
    config: &ServerConfig,
    operation: &str,
) -> anyhow::Result<[u8; 32]> {
    let identity = load_relay_custody_identity(config, operation).await?;
    Ok(derive_node_secret(&identity.to_bytes()))
}

pub(super) async fn load_relay_custody_identity(
    config: &ServerConfig,
    operation: &str,
) -> anyhow::Result<IdentityKeyPair> {
    load_node_identity(config)
        .await
        .map_err(|_| anyhow::anyhow!("relay custody {operation} requires the node identity key"))
}
