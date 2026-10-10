// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/quarantine/tests.rs
// ============================================
//! # Tests: Directory Replica quarantine inspection and resolution
//!
//! Unit tests for the directory Replica quarantine inspection and resolution, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use clap::Parser;

use crate::{Cli, Commands, DirectoryReplicaCommands};

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
