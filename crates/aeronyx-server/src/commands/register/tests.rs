// ============================================
// File: crates/aeronyx-server/src/commands/register/tests.rs
// ============================================
//! # Tests: `register` command
//!
//! Unit tests for `register` command, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use clap::Parser;

use crate::{Cli, Commands};

#[test]
fn registration_cli_keeps_legacy_code_and_accepts_stdin_mode() {
    let legacy =
        Cli::try_parse_from(["aeronyx-server", "register", "--code", "NYX-LEGACY-123"]).unwrap();
    let Commands::Register {
        code, code_stdin, ..
    } = legacy.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(code.as_deref(), Some("NYX-LEGACY-123"));
    assert!(!code_stdin);

    let stdin = Cli::try_parse_from(["aeronyx-server", "register", "--code-stdin"]).unwrap();
    let Commands::Register {
        code, code_stdin, ..
    } = stdin.command
    else {
        panic!("unexpected CLI command")
    };
    assert!(code.is_none());
    assert!(code_stdin);

    assert!(Cli::try_parse_from(["aeronyx-server", "register"]).is_err());
    assert!(Cli::try_parse_from([
        "aeronyx-server",
        "register",
        "--code",
        "NYX-123",
        "--code-stdin",
    ])
    .is_err());
}

#[test]
fn registration_code_stdin_is_trimmed_bounded_and_private_by_contract() {
    let code = read_registration_code(std::io::Cursor::new("  NYX-STDIN-123  \n")).unwrap();
    assert_eq!(code, "NYX-STDIN-123");

    let oversized = format!("{}\n", "A".repeat(MAX_REGISTRATION_CODE_BYTES + 1));
    assert!(read_registration_code(std::io::Cursor::new(oversized)).is_err());
    assert!(normalize_registration_code("NYX-123\0hidden").is_err());
    assert!(normalize_registration_code("   ").is_err());
}

#[test]
fn registration_profile_normalizes_operator_metadata() {
    assert_eq!(
        normalize_registration_name(Some("  TW1  ".to_string())).unwrap(),
        Some("TW1".to_string())
    );
    assert_eq!(
        normalize_registration_region(Some("tw".to_string())).unwrap(),
        Some("TW".to_string())
    );
}

#[test]
fn registration_profile_rejects_unsafe_metadata() {
    assert!(normalize_registration_name(Some("TW1\nadmin".to_string())).is_err());
    assert!(normalize_registration_name(Some(" ".to_string())).is_err());
    assert!(normalize_registration_region(Some("taiwan".to_string())).is_err());
    assert!(normalize_registration_region(Some("T1".to_string())).is_err());
}
