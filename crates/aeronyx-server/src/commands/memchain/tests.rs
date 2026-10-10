// ============================================
// File: crates/aeronyx-server/src/commands/memchain/tests.rs
// ============================================
//! # Tests: `memchain` commands
//!
//! Unit tests for `memchain` commands, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use clap::Parser;

use crate::{Cli, Commands};

#[test]
fn memchain_verify_aof_cli_accepts_explicit_read_only_path() {
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "memchain",
        "verify-aof",
        "--path",
        "/tmp/test.memchain",
    ])
    .unwrap();
    let Commands::Memchain(MemchainCommands::VerifyAof { path, .. }) = cli.command else {
        panic!("unexpected CLI command")
    };
    assert_eq!(path, Some(PathBuf::from("/tmp/test.memchain")));
}
