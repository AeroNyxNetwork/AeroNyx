// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/runtime.rs
// ============================================
//! # Runtime version, rollout, and upgrade status
//!
//! Owns the process runtime/build metadata, the executable-replacement
//! rollout signal, and the allow-listed local upgrade workflow status.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use serde_json::Value;

use crate::management::integrity;

use super::{
    unix_now_secs, NodeUpgradeStatus, RuntimeRolloutStatus, RuntimeVersionStatus,
    RUNTIME_STARTED_AT, UPGRADE_STATUS_FILE,
};

pub(super) async fn collect_runtime_rollout_status() -> RuntimeRolloutStatus {
    // Return a privacy-safe rollout signal for nodeboard.
    //
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // Linux keeps a running process alive even after its executable file is
    // replaced on disk. `/proc/self/exe` then points to a path suffixed with
    // `(deleted)`. That is a strong operator signal that a new binary may be
    // staged but the node still needs a controlled maintenance drain/restart.
    //
    // This reports process metadata only: executable replacement state and
    // whether a restart is required. The exact path remains process-local.
    let proc_exe = tokio::fs::read_link("/proc/self/exe").await.ok();
    let fallback_exe = if proc_exe.is_none() {
        std::env::current_exe().ok()
    } else {
        None
    };
    runtime_rollout_status_from_executable_path(proc_exe.or(fallback_exe))
}

/// Converts a local executable observation into aggregate rollout telemetry.
///
/// [OPERATOR-PATH-PRIVACY 2026-08-14 by Codex] Detection still uses the exact
/// path because Linux appends ` (deleted)` after replacement. Serialization
/// deliberately retains the legacy `executable_path` key as `null`, preserving
/// the response shape without exporting host filesystem identity.
pub(super) fn runtime_rollout_status_from_executable_path(
    executable_path: Option<std::path::PathBuf>,
) -> RuntimeRolloutStatus {
    let executable_replaced = executable_path
        .as_deref()
        .map(|path| {
            let path = path.to_string_lossy();
            path.contains(" (deleted)") || path.ends_with("(deleted)")
        })
        .unwrap_or(false);

    RuntimeRolloutStatus {
        executable_path: None,
        executable_replaced,
        restart_required: executable_replaced,
        detail: if executable_replaced {
            "Running process executable has been replaced on disk; restart after draining active sessions".to_string()
        } else {
            "Running process executable is active; no rollout restart signal detected".to_string()
        },
        source: "rust_process_executable_state",
        privacy_boundary: concat!(
            "aggregate runtime replacement state only; exact executable path, ",
            "destinations, DNS contents, packet payloads, domains, URLs, browsing ",
            "history, voucher secrets, client public IPs, and wallet-level traffic ",
            "are excluded"
        ),
    }
}

pub(super) async fn collect_runtime_version_status() -> RuntimeVersionStatus {
    let rollout = collect_runtime_rollout_status().await;
    collect_runtime_version_status_with_rollout(rollout)
}

pub(super) fn collect_runtime_version_status_with_rollout(
    rollout: RuntimeRolloutStatus,
) -> RuntimeVersionStatus {
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // This is node runtime metadata for commercial operations. It helps
    // nodeboard distinguish "online but old binary" from an actually upgraded
    // node. It intentionally excludes client identifiers, destinations, DNS
    // contents, packet payloads, domains, URLs, browsing history, voucher
    // secrets, and wallet-level traffic.
    let started_at = *RUNTIME_STARTED_AT.get_or_init(unix_now_secs);
    let now = unix_now_secs();
    RuntimeVersionStatus {
        version: integrity::get_version(),
        git_commit: build_git_commit(),
        build_profile: build_profile(),
        build_target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        process_id: std::process::id(),
        started_at,
        uptime_seconds: now.saturating_sub(started_at),
        rollout,
        source: "rust_process_runtime_metadata",
        privacy_boundary: concat!(
            "runtime build/process metadata only; no client public IPs, ",
            "destinations, DNS contents, packet payloads, domains, URLs, ",
            "browsing history, voucher secrets, or wallet-level traffic"
        ),
    }
}

pub(super) async fn collect_upgrade_status() -> NodeUpgradeStatus {
    // File creation/modification notes:
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    // Local producer:
    //   /root/open/AeroNyx/deploy/node/upgrade.sh
    // Backend consumer:
    //   /root/aeronyx/privacy_network/api/vpn_observability.py
    // Nodeboard consumer:
    //   /root/open/nodeboard/app/dashboard/nodes/[id]/page.tsx
    //
    // Main logical flow:
    // 1. Read `/var/lib/aeronyx/upgrade-status.json` when present.
    // 2. Allow-list only operator workflow fields used by nodeboard.
    // 3. Return `reported=false` for missing or invalid files so old nodes
    //    remain backward compatible.
    //
    // Important note for next developer:
    // - Do not forward arbitrary JSON from the local status file. Keep this
    //   allow-list tight. The heartbeat is signed node telemetry and must
    //   never contain registration codes, private keys, client public IPs,
    //   destinations, DNS contents, packet payloads, chat plaintext, voucher
    //   secrets, or wallet-level traffic.
    let privacy_boundary = concat!(
        "upgrade workflow metadata only; no registration codes, private keys, ",
        "client public IPs, destinations, DNS contents, packet payloads, chat ",
        "plaintext, voucher secrets, or wallet-level traffic"
    );

    let Ok(raw) = tokio::fs::read_to_string(UPGRADE_STATUS_FILE).await else {
        return NodeUpgradeStatus {
            reported: false,
            status: None,
            step: None,
            message: None,
            repo_dir: None,
            branch: None,
            service: None,
            config: None,
            no_restart: None,
            force: None,
            updated_at: None,
            source: UPGRADE_STATUS_FILE,
            privacy_boundary,
        };
    };

    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return NodeUpgradeStatus {
            reported: false,
            status: Some("unreadable".to_string()),
            step: None,
            message: Some("Local upgrade status file is not valid JSON".to_string()),
            repo_dir: None,
            branch: None,
            service: None,
            config: None,
            no_restart: None,
            force: None,
            updated_at: None,
            source: UPGRADE_STATUS_FILE,
            privacy_boundary,
        };
    };

    let string_field = |key: &str, max_len: usize| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(|text| sanitize_status_text(text, max_len))
            .filter(|text| !text.is_empty())
    };

    NodeUpgradeStatus {
        reported: true,
        status: string_field("status", 32),
        step: string_field("step", 64),
        message: string_field("message", 240),
        repo_dir: string_field("repo_dir", 256),
        branch: string_field("branch", 80),
        service: string_field("service", 80),
        config: string_field("config", 256),
        no_restart: value.get("no_restart").and_then(Value::as_bool),
        force: value.get("force").and_then(Value::as_bool),
        updated_at: string_field("updated_at", 80),
        source: UPGRADE_STATUS_FILE,
        privacy_boundary,
    }
}

fn sanitize_status_text(value: &str, max_len: usize) -> String {
    value
        .replace('\0', "")
        .chars()
        .take(max_len)
        .collect::<String>()
        .trim()
        .to_string()
}

fn build_git_commit() -> &'static str {
    option_env!("AERONYX_GIT_COMMIT")
        .or(option_env!("GIT_COMMIT"))
        .or(option_env!("VERGEN_GIT_SHA"))
        .unwrap_or("unknown")
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}
