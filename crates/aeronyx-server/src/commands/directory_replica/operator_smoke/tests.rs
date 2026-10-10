// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/operator_smoke/tests.rs
// ============================================
//! # Tests: Directory Replica operator smoke calls
//!
//! Unit tests for the directory Replica operator smoke calls, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use clap::Parser;

use crate::{Cli, Commands, DirectoryReplicaCommands};

#[test]
fn directory_replica_carrier_smoke_cli_is_read_only_and_json_capable() {
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "directory-replica",
        "carrier-smoke",
        "--json",
    ])
    .unwrap();
    let Commands::DirectoryReplica(DirectoryReplicaCommands::CarrierSmoke { json, config }) =
        cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert!(json);
    assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
}

#[test]
fn directory_replica_carrier_smoke_targets_operator_api_not_udp_tunnel() {
    let mut config = ServerConfig::default();
    config.memchain.api_listen_addr = "0.0.0.0:19421".parse().unwrap();

    assert_eq!(
        directory_replica_operator_smoke_url(&config, "carrier-smoke"),
        "http://127.0.0.1:19421/api/discovery/directory/carrier-smoke"
    );
}

#[test]
fn directory_replica_cold_bootstrap_smoke_cli_is_read_only_and_json_capable() {
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "directory-replica",
        "carrier-cold-bootstrap-smoke",
        "--json",
    ])
    .unwrap();
    let Commands::DirectoryReplica(DirectoryReplicaCommands::CarrierColdBootstrapSmoke {
        json,
        config,
    }) = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert!(json);
    assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
}
