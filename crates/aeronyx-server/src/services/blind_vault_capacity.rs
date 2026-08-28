// ============================================
// File: crates/aeronyx-server/src/services/blind_vault_capacity.rs
// ============================================
//! # Blind Vault Filesystem Capacity Probe
//!
//! ## Creation Reason
//! Keeps operating-system filesystem inspection outside Blind Vault domain
//! transactions while providing a replaceable capability boundary.
//!
//! ## Main Functionality
//! - Reports bytes available to the unprivileged node process.
//! - Uses the safe `nix::sys::statvfs` wrapper on Unix hosts.
//! - Fails closed on unsupported platforms or arithmetic overflow.
//! - Supports injected implementations without coupling storage policy to an
//!   operating-system API.
//!
//! ## Dependencies
//! - `nix::sys::statvfs`: safe Unix filesystem statistics.
//! - `services/blind_vault.rs`: applies the operator's reserve policy before
//!   capacity-consuming writes.
//!
//! ## Main Logical Flow
//! 1. The Blind Vault service supplies the database parent directory.
//! 2. The probe reads filesystem statistics without inspecting vault content.
//! 3. Available blocks are converted to bytes with checked arithmetic.
//! 4. The service decides whether the requested write preserves its reserve.
//!
//! ## Important Note For The Next Developer
//! - This module reports physical capacity only; it does not decide policy.
//! - Use blocks available to the running process, not total free blocks.
//! - Never turn a probe failure into permission to write.
//! - Keep alternate probes content-blind and free of lease or object IDs.
//!
//! Last Modified: v1.0.0-BlindVaultDiskReserve - Initial replaceable,
//! fail-closed filesystem capacity probe.
//! ============================================

use std::path::Path;

/// Coarse local failures from a filesystem capacity observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BlindVaultFilesystemCapacityProbeError {
    /// The host platform does not provide the required safe probe.
    #[error("filesystem capacity probing is unsupported on this platform")]
    UnsupportedPlatform,
    /// The operating system rejected the filesystem observation.
    #[error("filesystem capacity probe failed")]
    ProbeFailed,
    /// Filesystem block statistics could not be represented as bytes.
    #[error("filesystem capacity is outside the supported range")]
    CapacityOverflow,
}

/// Replaceable capability for observing storage available to this process.
pub trait BlindVaultFilesystemCapacityProbe: Send + Sync {
    /// Returns bytes available to the process on the filesystem containing
    /// `path`.
    fn available_bytes(&self, path: &Path) -> Result<u64, BlindVaultFilesystemCapacityProbeError>;
}

/// Production probe backed by the host operating system.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemBlindVaultFilesystemCapacityProbe;

impl BlindVaultFilesystemCapacityProbe for SystemBlindVaultFilesystemCapacityProbe {
    fn available_bytes(&self, path: &Path) -> Result<u64, BlindVaultFilesystemCapacityProbeError> {
        system_available_bytes(path)
    }
}

#[cfg(unix)]
fn system_available_bytes(path: &Path) -> Result<u64, BlindVaultFilesystemCapacityProbeError> {
    // [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] `blocks_available`
    // respects filesystem reservations for the node process. `blocks_free`
    // could overstate usable space and defeat the safety watermark.
    let statistics = nix::sys::statvfs::statvfs(path)
        .map_err(|_| BlindVaultFilesystemCapacityProbeError::ProbeFailed)?;
    let available_blocks = u64::try_from(statistics.blocks_available())
        .map_err(|_| BlindVaultFilesystemCapacityProbeError::CapacityOverflow)?;
    let fragment_size = u64::try_from(statistics.fragment_size())
        .map_err(|_| BlindVaultFilesystemCapacityProbeError::CapacityOverflow)?;
    available_blocks
        .checked_mul(fragment_size)
        .ok_or(BlindVaultFilesystemCapacityProbeError::CapacityOverflow)
}

#[cfg(not(unix))]
fn system_available_bytes(_path: &Path) -> Result<u64, BlindVaultFilesystemCapacityProbeError> {
    // [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] An explicitly enabled
    // watermark must never silently degrade into allow-all on an unsupported
    // host. Operators may leave the backward-compatible policy disabled.
    Err(BlindVaultFilesystemCapacityProbeError::UnsupportedPlatform)
}
