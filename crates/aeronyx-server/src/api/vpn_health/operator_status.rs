// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/operator_status.rs
// ============================================
//! # Node operator status
//!
//! Owns the nodeboard-facing operator service snapshot
//! (`collect_node_operator_status_response`), its `MemChain` / chat relay
//! metric contracts, and the compact operator action summary.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::sync::atomic::Ordering;

use serde_json::Value;

use crate::config::ServerConfig;

use super::health_snapshot::collect_vpn_health_response;
use super::runtime::{collect_runtime_rollout_status, collect_runtime_version_status_with_rollout};
use super::{
    unix_now_secs, HealthCheck, NodeOperatorStatusResponse, NodeUpgradeStatus,
    OperatorActionSummary, OperatorRisk, OperatorServiceStatus, ServiceManagerStatus,
    VpnCapacityStatus, VpnHealthState,
};

/// Builds the aggregate MemChain operator contract without reading or
/// serializing operator-local storage paths.
fn memchain_operator_metrics(config: &ServerConfig) -> Value {
    let enabled = config.memchain.is_enabled();
    serde_json::json!({
        "mode": format!("{:?}", config.memchain.mode),
        "api_listen_addr": config.memchain.api_listen_addr.to_string(),
        "db_path": Value::Null,
        "aof_path": Value::Null,
        "storage_backend": if enabled { "sqlite_aof" } else { "disabled" },
        "storage_paths_exposed": false,
        "api_secret_configured": config.memchain.effective_api_secret().is_some(),
        "ner_enabled": config.memchain.ner_enabled,
        "graph_enabled": config.memchain.graph_enabled,
        "reranker_enabled": config.memchain.reranker_enabled,
        "remote_storage_enabled": config.memchain.is_remote_storage_enabled(),
        "max_remote_owners": config.memchain.max_remote_owners,
    })
}

/// Builds the aggregate Chat Relay operator contract without exporting the
/// durable queue location.
fn chat_relay_operator_metrics(config: &ServerConfig) -> Value {
    let enabled = config.memchain.is_chat_relay_enabled();
    serde_json::json!({
        "offline_ttl_secs": config.memchain.chat_relay.offline_ttl_secs,
        "max_pending_per_wallet": config.memchain.chat_relay.max_pending_per_wallet,
        "db_path": Value::Null,
        "storage_backend": if enabled { "sqlite" } else { "disabled" },
        "storage_paths_exposed": false,
        "max_message_size": config.memchain.chat_relay.max_message_size,
        "max_blob_size": config.memchain.chat_relay.max_blob_size,
        "max_blobs_per_receiver": config.memchain.chat_relay.max_blobs_per_receiver,
        "cleanup_interval_secs": config.memchain.chat_relay.cleanup_interval_secs,
    })
}

pub(super) async fn collect_node_operator_status_response(
    state: VpnHealthState,
) -> NodeOperatorStatusResponse {
    let generated_at = unix_now_secs();
    let vpn_health = collect_vpn_health_response(state.clone()).await;
    let config = state.config.clone();
    let memchain_enabled = config.memchain.is_enabled();
    let remote_storage_enabled = config.memchain.is_remote_storage_enabled();
    let chat_relay_enabled = config.memchain.is_chat_relay_enabled();
    let supernode_enabled = config.memchain.is_supernode_enabled();
    let api_secret_configured = config.memchain.effective_api_secret().is_some();
    let encrypted_messages = state.encrypted_message_counter.load(Ordering::Relaxed);
    let runtime_rollout = collect_runtime_rollout_status().await;
    let runtime_version = collect_runtime_version_status_with_rollout(runtime_rollout.clone());
    let mut services = Vec::new();
    let mut risks = Vec::new();

    services.push(OperatorServiceStatus {
        key: "privacy_protocol",
        label: "AeroNyx Privacy Protocol",
        enabled: true,
        status: vpn_health.status,
        summary: format!(
            "{} active sessions, {} wallet devices, {} encrypted packets forwarded",
            vpn_health.active_sessions, vpn_health.active_wallet_devices, encrypted_messages
        ),
        metrics: serde_json::json!({
            "listen_addr": vpn_health.listen_addr,
            "gateway_ip": vpn_health.gateway_ip,
            "dns_proxy_enabled": vpn_health.dns_proxy_enabled,
            "dns_owner": vpn_health.dns_owner,
            "supported_transports": vpn_health.supported_transports.clone(),
            "preferred_transport": vpn_health.preferred_transport.clone(),
            "transport_health": vpn_health.transport_health.clone(),
            "privacy_protocol_health": vpn_health.privacy_protocol_health.clone(),
            "startup_self_check": vpn_health.startup_self_check.clone(),
            "virtual_ip_range": vpn_health.virtual_ip_range,
            "tun_device": vpn_health.tun_device,
            "configured_mtu": vpn_health.configured_mtu,
            "running_mtu": vpn_health.running_mtu,
            "active_sessions": vpn_health.active_sessions,
            "active_wallet_devices": vpn_health.active_wallet_devices,
            "service_manager": vpn_health.service_manager,
            "encrypted_message_forwarding": vpn_health.encrypted_message_forwarding,
            "session_cleanup": vpn_health.session_cleanup,
            "runtime": runtime_version.clone(),
            "placement_readiness": vpn_health.placement_readiness,
            "capacity": vpn_health.capacity,
            "packet_runtime": vpn_health.packet_runtime,
            "discovery_status": vpn_health.discovery_status,
            "chat_relay_status": vpn_health.chat_relay_status,
            "recent_errors": vpn_health.recent_errors,
            "upgrade_status": vpn_health.upgrade_status.clone(),
            "operator_action": vpn_health.operator_action.clone(),
            "failed_checks": vpn_health.checks.iter().filter(|check| !check.ok).count(),
            "runtime_rollout": runtime_rollout.clone(),
        }),
    });

    if vpn_health.startup_self_check.status != "ready" {
        risks.push(OperatorRisk {
            severity: if vpn_health.startup_self_check.status == "failed" {
                "critical"
            } else {
                "warning"
            },
            code: "startup_self_check",
            message: format!(
                "Startup self-check is {} with {} blocking checks",
                vpn_health.startup_self_check.status,
                vpn_health.startup_self_check.blocking_checks.len()
            ),
            remediation: vpn_health.startup_self_check.recommended_action.clone(),
        });
    }

    services.push(OperatorServiceStatus {
        key: "memchain",
        label: "MemChain / MPI",
        enabled: memchain_enabled,
        status: if memchain_enabled { "ok" } else { "disabled" },
        summary: if memchain_enabled {
            format!(
                "mode={:?}, API bound at {}",
                config.memchain.mode, config.memchain.api_listen_addr
            )
        } else {
            "MemChain is disabled in this node config".to_string()
        },
        // [OPERATOR-PATH-PRIVACY 2026-08-14 by Codex] The backend and
        // nodeboard need storage readiness, not the operator's username,
        // home directory, mount layout, or deployment convention.
        metrics: memchain_operator_metrics(&config),
    });

    services.push(OperatorServiceStatus {
        key: "chat_relay",
        label: "Zero-Knowledge Chat Relay",
        enabled: chat_relay_enabled,
        status: if chat_relay_enabled { "ok" } else { "disabled" },
        summary: if chat_relay_enabled {
            format!(
                "offline TTL {}s, {} pending messages per wallet, max blob {} bytes",
                config.memchain.chat_relay.offline_ttl_secs,
                config.memchain.chat_relay.max_pending_per_wallet,
                config.memchain.chat_relay.max_blob_size
            )
        } else {
            "Chat relay is disabled; encrypted messages are not stored for offline delivery"
                .to_string()
        },
        metrics: chat_relay_operator_metrics(&config),
    });

    services.push(OperatorServiceStatus {
        key: "sovereign_data_layer",
        label: "Sovereign Data Layer",
        enabled: remote_storage_enabled,
        status: if remote_storage_enabled {
            "ready"
        } else {
            "planned"
        },
        summary: if remote_storage_enabled {
            format!(
                "remote encrypted owner storage enabled for up to {} owners",
                config.memchain.max_remote_owners
            )
        } else {
            "Encrypted user-owned record RPC is not enabled on this node yet".to_string()
        },
        metrics: serde_json::json!({
            "remote_storage_enabled": remote_storage_enabled,
            "max_remote_owners": config.memchain.max_remote_owners,
            "current_protocol_basis": [
                "MemoryRecord.owner",
                "MemoryRecord.encrypted_content",
                "MemoryRecord.signature",
                "MemChainMessage::SyncRecordRequest",
                "MemChainMessage::SyncRecordResponse"
            ],
            "settlement_layer": "ethereum",
            "private_data_on_ethereum": false,
        }),
    });

    services.push(OperatorServiceStatus {
        key: "supernode",
        label: "SuperNode Cognitive Worker",
        enabled: supernode_enabled,
        status: if supernode_enabled {
            "ready"
        } else {
            "disabled"
        },
        summary: if supernode_enabled {
            format!(
                "{} configured provider(s)",
                config.memchain.supernode.providers.len()
            )
        } else {
            "SuperNode LLM worker is disabled".to_string()
        },
        metrics: serde_json::json!({
            "providers": config.memchain.supernode.providers.len(),
            "worker_poll_interval_secs": config.memchain.supernode.worker.poll_interval_secs,
            "worker_max_concurrent": config.memchain.supernode.worker.max_concurrent,
        }),
    });

    if vpn_health.status != "ok" {
        risks.push(OperatorRisk {
            severity: if vpn_health.status == "failed" { "critical" } else { "warning" },
            code: "privacy_protocol_health",
            message: format!("AeroNyx privacy protocol health is {}", vpn_health.status),
            remediation: "Open /api/vpn/health and resolve failed checks before advertising this node as healthy".to_string(),
        });
    }

    if runtime_rollout.restart_required {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "runtime_restart_required",
            message: "Rust process is running an executable that has been replaced on disk".to_string(),
            remediation: "Drain active sessions, enter maintenance mode, then restart the AeroNyx Rust node so the staged binary takes effect".to_string(),
        });
    }

    if vpn_health.upgrade_status.status.as_deref() == Some("failed") {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "upgrade_workflow_failed",
            message: vpn_health
                .upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "Last Rust upgrade workflow failed".to_string()),
            remediation: "Open node detail, inspect upgrade status, run healthcheck, then rerun deploy/node/aeronyx-node.sh upgrade --no-restart after resolving the failed step".to_string(),
        });
    }

    if !api_secret_configured {
        risks.push(OperatorRisk {
            severity: if remote_storage_enabled {
                "critical"
            } else {
                "warning"
            },
            code: "mpi_api_secret_missing",
            message: "MemChain API secret is not configured".to_string(),
            remediation:
                "Set memchain.api_secret before enabling remote RPC or remote encrypted storage"
                    .to_string(),
        });
    }

    if !chat_relay_enabled {
        risks.push(OperatorRisk {
            severity: "info",
            code: "chat_relay_disabled",
            message: "Zero-knowledge chat relay is disabled".to_string(),
            remediation: "Enable [memchain.chat_relay] when this node should store encrypted offline messages and blobs".to_string(),
        });
    }

    if !remote_storage_enabled {
        risks.push(OperatorRisk {
            severity: "info",
            code: "sovereign_data_layer_not_enabled",
            message: "Sovereign Data Layer remote storage is not enabled".to_string(),
            remediation: "Enable memchain.allow_remote_storage after encrypted record limits and API authentication are ready".to_string(),
        });
    }

    if remote_storage_enabled && config.memchain.max_remote_owners == 0 {
        risks.push(OperatorRisk {
            severity: "warning",
            code: "remote_owner_capacity_unbounded",
            message: "Remote encrypted storage owner capacity is unlimited".to_string(),
            remediation: "Set memchain.max_remote_owners for commercial node capacity planning"
                .to_string(),
        });
    }

    let has_critical = risks.iter().any(|risk| risk.severity == "critical");
    let has_warning = risks.iter().any(|risk| risk.severity == "warning");
    let status = if has_critical {
        "critical"
    } else if vpn_health.status == "failed" {
        "failed"
    } else if has_warning || vpn_health.status == "degraded" {
        "attention"
    } else {
        "ok"
    };

    NodeOperatorStatusResponse {
        status,
        generated_at,
        runtime_rollout,
        services,
        risks,
        privacy_boundary: concat!(
            "operator status contains aggregate service health and configuration ",
            "only; host filesystem paths, user plaintext, social graph plaintext, ",
            "destinations, DNS contents, packet payloads, domains, URLs, browsing ",
            "history, voucher secrets, and wallet-level traffic are excluded"
        ),
    }
}

pub(super) fn collect_operator_action_summary(
    status: &'static str,
    checks: &[HealthCheck],
    capacity: &VpnCapacityStatus,
    upgrade_status: &NodeUpgradeStatus,
    service_manager: &ServiceManagerStatus,
) -> OperatorActionSummary {
    let privacy_boundary = concat!(
        "operator action is derived from aggregate node operations metadata ",
        "only; no client public IPs, destinations, DNS contents, packet ",
        "payloads, domains, URLs, browsing history, voucher secrets, chat ",
        "plaintext, private keys, or wallet-level traffic"
    );

    if let Some(check) = checks.iter().find(|check| !check.ok) {
        return OperatorActionSummary {
            status: if status == "failed" { "critical" } else { "warning" },
            priority: "fix_failed_health_check",
            title: "AeroNyx privacy protocol check failed".to_string(),
            detail: format!("{}: {}", check.name, check.detail),
            next_step: "Open node detail health checks, fix the failed check, then rerun deploy/node/aeronyx-node.sh health --json.".to_string(),
            source: "rust_vpn_health.checks",
            privacy_boundary,
        };
    }

    if upgrade_status.status.as_deref() == Some("failed") {
        return OperatorActionSummary {
            status: "critical",
            priority: "fix_failed_upgrade",
            title: "Rust upgrade workflow failed".to_string(),
            detail: upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "Last upgrade workflow reported failure.".to_string()),
            next_step: "Inspect upgrade_status, resolve the failed step, then rerun deploy/node/aeronyx-node.sh upgrade --no-restart before a controlled restart.".to_string(),
            source: "rust_vpn_health.upgrade_status",
            privacy_boundary,
        };
    }

    if service_manager.active_state != "active" {
        return OperatorActionSummary {
            status: "warning",
            priority: "service_not_active",
            title: "Rust service is not active".to_string(),
            detail: service_manager.detail.clone(),
            next_step: "Run deploy/node/aeronyx-node.sh status and logs --lines 200 before restart or upgrade actions.".to_string(),
            source: "rust_vpn_health.service_manager",
            privacy_boundary,
        };
    }

    if let Some(risk) = capacity
        .risks
        .iter()
        .find(|risk| risk.severity == "critical")
    {
        return OperatorActionSummary {
            status: "critical",
            priority: risk.code,
            title: "Capacity blocks commercial placement".to_string(),
            detail: risk.message.clone(),
            next_step: risk.remediation.clone(),
            source: "rust_vpn_health.capacity.risks",
            privacy_boundary,
        };
    }

    if let Some(risk) = capacity
        .risks
        .iter()
        .find(|risk| risk.severity == "warning")
    {
        return OperatorActionSummary {
            status: "warning",
            priority: risk.code,
            title: "Capacity needs operator review".to_string(),
            detail: risk.message.clone(),
            next_step: risk.remediation.clone(),
            source: "rust_vpn_health.capacity.risks",
            privacy_boundary,
        };
    }

    if status == "degraded" {
        return OperatorActionSummary {
            status: "warning",
            priority: "privacy_protocol_degraded",
            title: "AeroNyx privacy protocol is degraded".to_string(),
            detail: "Local health checks passed enough to stay online, but the node is not fully clean.".to_string(),
            next_step: "Review node detail health checks, capacity, recent events, and network rules before accepting more traffic.".to_string(),
            source: "rust_vpn_health.status",
            privacy_boundary,
        };
    }

    if upgrade_status.status.as_deref() == Some("staged") {
        return OperatorActionSummary {
            status: "info",
            priority: "upgrade_staged",
            title: "Rust upgrade is staged".to_string(),
            detail: upgrade_status
                .message
                .clone()
                .unwrap_or_else(|| "A new Rust build is staged without restart.".to_string()),
            next_step: "Use maintenance mode, drain active sessions, then perform a controlled restart when ready.".to_string(),
            source: "rust_vpn_health.upgrade_status",
            privacy_boundary,
        };
    }

    OperatorActionSummary {
        status: "ok",
        priority: "monitor",
        title: "Node is ready for monitoring".to_string(),
        detail: "No failed health checks or capacity blockers are reported.".to_string(),
        next_step: "Keep monitoring heartbeat freshness, capacity, traffic, and recent events in nodeboard.".to_string(),
        source: "rust_vpn_health",
        privacy_boundary,
    }
}

#[cfg(test)]
mod tests;
