// ============================================
// File: crates/aeronyx-server/src/lib.rs
// ============================================
//! # AeroNyx Server Library
//!
//! ## Modification Reason
//! - Added management module for CMS integration.
//! - Added api module for legacy MemChain local HTTP API.
//! - v0.5.0: Added miner module for ReflectionMiner block packing.
//! - v2.5.0: Added config_supernode module for SuperNode LLM config.
//! - v1.0.0-MultiTenant: miner module now exports MinerScheduler in addition
//!   to ReflectionMiner. MinerScheduler is the SaaS-mode Miner dispatcher.
//! - v1.0.0-MultiTenant: Added config sub-modules (config_chat_relay,
//!   config_infra, config_memchain, config_saas) that are referenced by
//!   config.rs via pub use re-exports.
//! - v1.0.0-BlindVaultService: Added an independent, default-off Blind Vault
//!   configuration module for anonymous encrypted durable objects.
//! - [SYSTEMD-CHILD-ISOLATION 2026-08-11 by Codex] Added one process factory
//!   that strips inherited systemd readiness, watchdog, and socket-activation
//!   authority from every operating-system child spawned by the node.
//!
//! ## Last Modified
//! v0.1.0 - Initial server library
//! v0.2.0 - Added management module for CMS integration
//! v0.3.0 - Added api module for legacy MemChain local HTTP API
//! v0.5.0 - Added miner module for ReflectionMiner
//! v2.5.0 - Added config_supernode module
//! v1.0.0-MultiTenant - MinerScheduler added to miner module;
//!                      config sub-modules declared at crate root
//! v1.0.0-BlindVaultService - Added bounded Blind Vault configuration
//! v2.8.42-SystemdChildIsolation - Prevent child processes from inheriting
//!                                  service-manager protocol authority

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod api;
pub mod config;

// v1.0.0-MultiTenant: Config sub-modules referenced by config.rs pub use re-exports.
// These are declared at crate root (not under config/) because config.rs is a single
// file module. config.rs re-exports their types via `pub use crate::config_xxx::...`.
pub mod config_blind_vault;
pub mod config_chat_relay;
pub mod config_infra;
pub mod config_memchain;
pub mod config_saas;

// v2.5.0+SuperNode: SuperNode LLM configuration types.
// Declared at crate root (not under config/) because config.rs is a single file,
// not a directory module. All imports use `crate::config_supernode::...`.
pub mod config_supernode;

pub mod error;
pub mod handlers;
pub mod management;
pub mod miner;
pub mod server;
pub mod services;
pub mod voucher_verifier;

// Re-export primary types
pub use config::ServerConfig;
pub use error::{Result, ServerError};
pub use server::Server;

// Re-export management types
pub use management::{ManagementClient, ManagementConfig};

/// Environment variables that confer systemd service-manager protocol access.
///
/// [SYSTEMD-CHILD-ISOLATION 2026-08-11 by Codex] The Rust node is the only
/// process permitted to report readiness or watchdog state. It also does not
/// delegate socket activation to commands used for health collection or node
/// operations. Keeping this policy at the crate boundary prevents individual
/// command call sites from silently forgetting one of these variables.
const SYSTEMD_CHILD_AUTHORITY_ENV: [&str; 6] = [
    "NOTIFY_SOCKET",
    "WATCHDOG_PID",
    "WATCHDOG_USEC",
    "LISTEN_PID",
    "LISTEN_FDS",
    "LISTEN_FDNAMES",
];

/// Creates a Tokio child command without inherited systemd protocol authority.
///
/// Ordinary configuration environment variables remain available to preserve
/// backward compatibility. All long-lived server subprocesses must be created
/// through this function so only the Rust main process can signal readiness.
pub(crate) fn isolated_child_command(
    program: impl AsRef<std::ffi::OsStr>,
) -> tokio::process::Command {
    let mut command = std::process::Command::new(program);
    strip_systemd_child_authority(&mut command);
    tokio::process::Command::from(command)
}

fn strip_systemd_child_authority(command: &mut std::process::Command) {
    for variable in SYSTEMD_CHILD_AUTHORITY_ENV {
        command.env_remove(variable);
    }
}

#[cfg(test)]
mod process_tests {
    use std::collections::HashMap;

    use super::{strip_systemd_child_authority, SYSTEMD_CHILD_AUTHORITY_ENV};

    #[test]
    fn child_processes_explicitly_remove_systemd_authority() {
        let mut command = std::process::Command::new("test-only-command");
        strip_systemd_child_authority(&mut command);

        let environment: HashMap<_, _> = command
            .get_envs()
            .map(|(name, value)| (name.to_os_string(), value.map(ToOwned::to_owned)))
            .collect();
        for variable in SYSTEMD_CHILD_AUTHORITY_ENV {
            assert_eq!(
                environment.get(std::ffi::OsStr::new(variable)),
                Some(&None),
                "{variable} must be explicitly removed from child environments"
            );
        }
    }
}
