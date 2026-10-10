// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/service_manager.rs
// ============================================
//! # Service manager status
//!
//! Owns systemd service-unit resolution (operator override, process cgroup,
//! historical fallback) and the `systemctl show` service manager status.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use tokio::time::timeout;

use crate::isolated_child_command;

use super::{
    ServiceManagerStatus, CHECK_TIMEOUT, PROC_SELF_CGROUP_PATH, VPN_SERVICE_NAME,
    VPN_SERVICE_NAME_ENV, VPN_SERVICE_UNIT_NAME,
};

pub(super) fn resolve_vpn_service_name() -> String {
    let operator_override = std::env::var(VPN_SERVICE_NAME_ENV).ok();
    let process_cgroup = std::fs::read_to_string(PROC_SELF_CGROUP_PATH).ok();
    resolve_vpn_service_name_from(operator_override.as_deref(), process_cgroup.as_deref())
}

fn resolve_vpn_service_name_from(
    operator_override: Option<&str>,
    process_cgroup: Option<&str>,
) -> String {
    let resolved = operator_override
        .and_then(validated_systemd_service_name)
        .or_else(|| process_cgroup.and_then(systemd_service_name_from_cgroup))
        .unwrap_or_else(|| VPN_SERVICE_NAME.to_string());
    if resolved == VPN_SERVICE_UNIT_NAME {
        VPN_SERVICE_NAME.to_string()
    } else {
        resolved
    }
}

fn systemd_service_name_from_cgroup(cgroup: &str) -> Option<String> {
    cgroup
        .lines()
        .flat_map(|line| line.split('/').rev())
        .find_map(|segment| {
            if segment.ends_with(".service") {
                validated_systemd_service_name(segment)
            } else {
                None
            }
        })
}

fn validated_systemd_service_name(value: &str) -> Option<String> {
    let value = value.trim();
    let valid_length = !value.is_empty() && value.len() <= 255;
    let valid_prefix = value
        .bytes()
        .next()
        .map(|byte| byte.is_ascii_alphanumeric())
        .unwrap_or(false);
    let valid_suffix = value.ends_with(".service") || !value.contains('.');
    let valid_chars = value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'@' | b':')
    });
    (valid_length && valid_prefix && valid_suffix && valid_chars).then(|| value.to_string())
}

pub(super) async fn collect_service_manager_status(service_name: &str) -> ServiceManagerStatus {
    // Source path:
    //   /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
    //
    // Nodeboard/backend consumers:
    //   /root/aeronyx/privacy_network/api/vpn_observability.py
    //   /root/open/nodeboard/app/dashboard/services/page.tsx
    //
    // This command returns local process manager metadata only. It does not
    // inspect destinations, DNS contents, packet payloads, domains, URLs,
    // browsing history, voucher secrets, client public IPs, or wallet traffic.
    let result = timeout(
        CHECK_TIMEOUT,
        isolated_child_command("systemctl")
            .args([
                "show",
                service_name,
                "--property=LoadState,ActiveState,UnitFileState",
                "--value",
            ])
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines().map(str::trim);
            let load_state = lines.next().unwrap_or("").to_string();
            let active_state = lines.next().unwrap_or("").to_string();
            let unit_file_state = lines.next().unwrap_or("").to_string();
            let load_state = if load_state.is_empty() {
                "unknown".to_string()
            } else {
                load_state
            };
            let active_state = if active_state.is_empty() {
                "unknown".to_string()
            } else {
                active_state
            };
            let unit_file_state = if unit_file_state.is_empty() {
                "unknown".to_string()
            } else {
                unit_file_state
            };
            let restart_supported = load_state == "loaded";
            ServiceManagerStatus {
                manager: "systemd",
                service_name: service_name.to_string(),
                load_state: load_state.clone(),
                active_state: active_state.clone(),
                unit_file_state: unit_file_state.clone(),
                restart_supported,
                detail: if restart_supported {
                    format!(
                        "{} systemd service is loaded (ActiveState={}, UnitFileState={})",
                        service_name, active_state, unit_file_state
                    )
                } else {
                    format!(
                        "{} systemd service is not restartable from nodeboard (LoadState={}, ActiveState={}, UnitFileState={})",
                        service_name, load_state, active_state, unit_file_state
                    )
                },
            }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim().chars().take(240).collect::<String>();
            ServiceManagerStatus {
                manager: "systemd",
                service_name: service_name.to_string(),
                load_state: "unknown".to_string(),
                active_state: "unknown".to_string(),
                unit_file_state: "unknown".to_string(),
                restart_supported: false,
                detail: format!("systemctl show failed: {}", detail),
            }
        }
        Ok(Err(error)) => ServiceManagerStatus {
            manager: "systemd",
            service_name: service_name.to_string(),
            load_state: "unavailable".to_string(),
            active_state: "unavailable".to_string(),
            unit_file_state: "unavailable".to_string(),
            restart_supported: false,
            detail: format!("systemctl unavailable: {}", error),
        },
        Err(_) => ServiceManagerStatus {
            manager: "systemd",
            service_name: service_name.to_string(),
            load_state: "timeout".to_string(),
            active_state: "timeout".to_string(),
            unit_file_state: "timeout".to_string(),
            restart_supported: false,
            detail: "systemctl show timed out".to_string(),
        },
    }
}

#[cfg(test)]
mod tests;
