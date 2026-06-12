# VPN Command Control Plane

## Source Paths

- `crates/aeronyx-server/src/management/command_handler.rs`
  - Receives CMS commands from the heartbeat reporter.
  - Dispatches only VPN operations commands:
    - `system_info`
    - `collect_logs`
    - `refresh_config`
    - `kick_session`
    - `ban_wallet`
    - `unban_wallet`
    - `restart_service`
  - Legacy non-VPN lifecycle actions are not dispatched. Unknown actions are
    reported back to the CMS as failed command results.

- `crates/aeronyx-server/src/management/models.rs`
  - Documents the VPN command action set.
  - Defaults command status reports to `agent_type = "vpn"` so backend command
    auditing stays attached to the VPN operations console.

## Operator Behavior

The nodeboard backend only queues bounded VPN operations commands. The Rust node
executes those commands without exposing arbitrary shell, file path, or process
control. Command results are reported to the CMS through the signed
`/node/vpn/status/` endpoint with `agent_type = "vpn"`.

## Policy Enforcement Telemetry

The Rust node enforces nodeboard policy locally in the VPN hot paths:
`maintenance_mode` and `max_sessions` are checked before new session allocation,
and `bandwidth_limit_mbps` is checked before VPN packet counters are recorded.

`system_stats.vpn_health.policy_enforcement` reports aggregate counters for
maintenance rejections, max-session rejections, bandwidth drops, and the last
rejection reason/time. The backend turns these counters into
`node_policy_enforced` events for nodeboard Alerts / Events.

## Privacy Boundary

Command results are operational diagnostics only. They must not include traffic
destinations, DNS query contents, packet payloads, browsing history, or full
client-identifying data. Policy enforcement telemetry is aggregate-only and
does not include destinations, DNS contents, packet payloads, or browsing
history. `collect_logs` returns a bounded and redacted service tail intended for
VPN stability diagnosis.
