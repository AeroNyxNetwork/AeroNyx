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

## Session Quality Snapshot Cadence

Source paths:

- `crates/aeronyx-server/src/server.rs`
- `crates/aeronyx-server/src/management/config.rs`

The Rust node drives active VPN session quality snapshots from
`management.session_report_interval_secs` instead of a hard-coded five-minute
timer. The default is 30 seconds, and runtime scheduling clamps the interval to
10-300 seconds.

Snapshots are emitted for every established session, even when the current
traffic counters are still zero. This keeps low-traffic commercial VPN tunnels
visible in nodeboard with fresh `last_tx_at`, keepalive RTT, replay-window
counters, and packet-loss metadata.

## Keepalive ACK Counters

Source paths:

- `crates/aeronyx-server/src/services/session.rs`
- `crates/aeronyx-server/src/handlers/packet.rs`
- `crates/aeronyx-server/src/server.rs`
- `crates/aeronyx-server/src/management/reporter.rs`
- `crates/aeronyx-server/src/management/models.rs`

The Rust node now records in-tunnel keepalive ACK health per session:

- `keepalive_probes_sent`: probes queued for the assigned virtual IP.
- `keepalive_acks`: probes that received a matching ACK.
- `keepalive_missed`: probes that expired without an ACK.
- `keepalive_pending`: probes still waiting for an ACK.

The keepalive task sends probes every 60 seconds and treats an ACK as missed
after 90 seconds. These counters make "connected but unstable" tunnels visible
to the CMS and nodeboard without requiring SSH.

## Node Policy Sync Snapshot

Source paths:

- `crates/aeronyx-server/src/api/vpn_health.rs`
- `crates/aeronyx-server/src/services/node_policy.rs`

Rust includes the current runtime `node_policy` snapshot in
`system_stats.vpn_health` on heartbeat. The CMS can compare this snapshot with
nodeboard Settings to show whether tier, maintenance mode, max sessions,
bandwidth cap, and heartbeat interval have reached the node.

This is operational control-plane metadata only. It does not include user
destinations, DNS contents, packet payloads, domains, URLs, or browsing history.

## Apply Policy Command

Source paths:

- `crates/aeronyx-server/src/management/command_handler.rs`
- `crates/aeronyx-server/src/server.rs`

`apply_policy` is a safe acknowledgement command for nodeboard Settings. The
CMS still sends authoritative policy values in the heartbeat response; Rust
updates `NodePolicyRuntime` before dispatching heartbeat commands. When
`apply_policy` runs, it reports the current runtime policy snapshot back to the
CMS command history so operators can confirm policy delivery without SSH.

The command does not accept arbitrary policy values, file paths, shell
arguments, or traffic inspection parameters.

## Legacy Non-VPN Cleanup

Source paths:

- `privacy_network/consumers.py` on the CMS backend
- `crates/aeronyx-server/Cargo.toml`
- `crates/aeronyx-server/src/lib.rs`
- `crates/aeronyx-core/src/ledger/block.rs`
- `crates/aeronyx-core/src/protocol/memchain.rs`
- `scripts/download_models.sh`

Nodeboard no longer exposes legacy non-VPN pages, navigation, or API calls.
The CMS WebSocket keeps VPN/E2E node operations available, but plaintext legacy
frontend request/stream/response traffic is no longer forwarded between the
browser and Rust node. Internal MPI compatibility traffic remains bounded to the
CMS-to-node bridge and is not exposed as a nodeboard product surface.

Rust comments and test fixtures use neutral AeroNyx memory/runtime names, keeping
the commercial VPN operator experience focused on VPN health, policy, sessions,
billing, commands, and events.

## Privacy Boundary

Command results are operational diagnostics only. They must not include traffic
destinations, DNS query contents, packet payloads, browsing history, or full
client-identifying data. Policy enforcement telemetry is aggregate-only and
does not include destinations, DNS contents, packet payloads, or browsing
history. Session quality snapshots contain only operational tunnel metadata and
never include destination IP addresses, DNS contents, packet payloads, domains,
URLs, or browsing history. `collect_logs` returns a bounded and redacted service
tail intended for VPN stability diagnosis.
