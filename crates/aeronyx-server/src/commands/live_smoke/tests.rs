// ============================================
// File: crates/aeronyx-server/src/commands/live_smoke/tests.rs
// ============================================
//! # Tests: `mailbox-probe`, `relay-smoke` and `v1-compatibility-smoke` commands
//!
//! Unit tests for `mailbox-probe`, `relay-smoke` and `v1-compatibility-smoke` commands, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use clap::Parser;

use crate::{Cli, Commands};

#[test]
fn relay_smoke_cli_is_local_bounded_and_explicit() {
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "relay-smoke",
        "--confirm-live-relay-smoke",
        "--json",
    ])
    .unwrap();
    let Commands::RelaySmoke {
        server,
        health_url,
        config,
        timeout_seconds,
        confirm_live_relay_smoke,
        json,
    } = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(server, "127.0.0.1:51820".parse().unwrap());
    assert_eq!(health_url, "http://127.0.0.1:8421/api/vpn/health");
    assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
    assert_eq!(timeout_seconds, 30);
    assert!(confirm_live_relay_smoke);
    assert!(json);

    assert!(
        Cli::try_parse_from(["aeronyx-server", "relay-smoke", "--timeout-seconds", "121",])
            .is_err()
    );
}

#[test]
fn v1_compatibility_smoke_cli_is_additive_bounded_and_explicit() {
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "v1-compatibility-smoke",
        "--confirm-v1-compatibility-smoke",
        "--json",
    ])
    .unwrap();
    let Commands::V1CompatibilitySmoke {
        server,
        health_url,
        config,
        timeout_seconds,
        confirm_v1_compatibility_smoke,
        json,
    } = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(server, "127.0.0.1:51820".parse().unwrap());
    assert_eq!(health_url, "http://127.0.0.1:8421/api/vpn/health");
    assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
    assert_eq!(timeout_seconds, 90);
    assert!(confirm_v1_compatibility_smoke);
    assert!(json);

    for rejected in [64, 181] {
        let rejected = rejected.to_string();
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "v1-compatibility-smoke",
            "--timeout-seconds",
            rejected.as_str(),
        ])
        .is_err());
    }
}
