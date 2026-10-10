// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/operator_status/tests.rs
// ============================================
//! # Tests: operator telemetry
//!
//! Unit tests for operator telemetry, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

use crate::api::vpn_health::runtime::runtime_rollout_status_from_executable_path;

#[test]
fn operator_telemetry_excludes_storage_and_executable_paths() {
    // [OPERATOR-PATH-PRIVACY 2026-08-14 by Codex] Canary values prove
    // configured and runtime paths remain local while operational state
    // stays machine-readable and backward-compatible.
    let mut config = ServerConfig::default();
    config.memchain.db_path = "/home/private-node/memchain.sqlite".to_string();
    config.memchain.aof_path = "/home/private-node/memchain.aof".to_string();
    config.memchain.chat_relay.db_path = "/home/private-node/chat-relay.sqlite".to_string();

    let memchain_metrics = memchain_operator_metrics(&config);
    let relay_metrics = chat_relay_operator_metrics(&config);
    let rollout = runtime_rollout_status_from_executable_path(Some(std::path::PathBuf::from(
        "/home/private-node/aeronyx-server (deleted)",
    )));
    let rendered = serde_json::to_string(&(&memchain_metrics, &relay_metrics, &rollout)).unwrap();

    assert!(rollout.executable_replaced);
    assert!(rollout.restart_required);
    assert!(rollout.executable_path.is_none());
    assert!(memchain_metrics["db_path"].is_null());
    assert!(memchain_metrics["aof_path"].is_null());
    assert!(relay_metrics["db_path"].is_null());
    assert_eq!(memchain_metrics["storage_paths_exposed"], false);
    assert_eq!(relay_metrics["storage_paths_exposed"], false);
    assert!(!rendered.contains("private-node"));
    assert!(!rendered.contains("memchain.sqlite"));
    assert!(!rendered.contains("chat-relay.sqlite"));
    assert!(!rendered.contains("aeronyx-server (deleted)"));
}
