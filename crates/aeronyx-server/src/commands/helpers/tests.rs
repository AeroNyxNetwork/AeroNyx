// ============================================
// File: crates/aeronyx-server/src/commands/helpers/tests.rs
// ============================================
//! # Tests: Shared command helpers
//!
//! Unit tests for the shared command helpers, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

#[test]
fn strict_hex_parser_rejects_ambiguous_or_unbounded_identifiers() {
    assert_eq!(parse_hex32(&"a5".repeat(32), "test").unwrap(), [0xa5; 32]);
    assert!(parse_hex32(&"a5".repeat(31), "test").is_err());
    assert!(parse_hex32(&format!("{}gg", "a5".repeat(31)), "test").is_err());
}
