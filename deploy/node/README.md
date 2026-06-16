# AeroNyx Production Node Deployment

<!--
============================================
File Creation/Modification Notes
============================================
Creation Reason:
- Provide operator-facing documentation for the production Rust privacy node
  deployment scripts.

Modification Reason:
- Document production upgrade unit-template synchronization, rollback behavior,
  shared node-local deployment locking, and install-time systemd unit
  verification, purge path safety, service-name validation, and release-backup
  retention/diagnostics, plus network restore command-path portability and unit
  verification/synchronization, low-risk maintenance, and tracked dirty
  worktree protection, config-driven VPN network rules, network-only
  maintenance, install-time commercial capacity plan checks, and healthcheck
  capacity-risk JSON export.

Main Functionality:
- Explains first install, registration, upgrade, healthcheck, configuration
  ownership, compatibility scope, and next-developer guidance.

Dependencies:
- deploy/node/install.sh
- deploy/node/upgrade.sh
- deploy/node/healthcheck.sh
- deploy/node/server.example.toml
- deploy/node/aeronyx-server.service
- crates/aeronyx-server/src/main.rs

Main Logical Flow:
1. Operator installs the node with install.sh.
2. Operator registers with a nodeboard registration code.
3. systemd runs aeronyx-server and healthcheck.sh verifies runtime status.

Important Note for Next Developer:
- Do not document workflows that require exposing private keys or user traffic.
- Keep the commands compatible with Linux/systemd production nodes.
- macOS, iOS, Android, and Windows are client/development platforms for this
  deployment package, not production node targets.

Last Modified:
v1.23.0-node-deploy - Documented healthcheck capacity telemetry warnings and
                     JSON export for nodeboard automation.
v1.22.0-node-deploy - Documented installer capacity plan preflight for IP pool,
                     max connections, fd limit, and conntrack headroom.
v1.21.0-node-deploy - Documented --network-only maintenance for config-driven
                     NAT/FORWARD refresh.
v1.20.0-node-deploy - Documented config-driven VPN subnet/TUN network rules
                     and health diagnostics.
v1.19.0-node-deploy - Documented tracked dirty-worktree protection for install
                     and upgrade.
v1.18.0-node-deploy - Documented live systemd unit binding diagnostics.
v1.17.0-node-deploy - Documented mutually exclusive maintenance flags.
v1.16.0-node-deploy - Documented --service-unit-only maintenance mode.
v1.15.0-node-deploy - Documented systemd restart-policy diagnostics.
v1.14.0-node-deploy - Documented network restore backup count diagnostics.
v1.13.0-node-deploy - Documented --network-restore-only maintenance mode.
v1.12.0-node-deploy - Documented upgrade-time network restore synchronization.
v1.11.0-node-deploy - Documented network restore unit verification.
v1.10.0-node-deploy - Documented structured network restore command diagnostics.
v1.9.0-node-deploy - Documented portable network restore command paths.
v1.8.0-node-deploy - Documented healthcheck release-backup diagnostics.
v1.7.0-node-deploy - Documented upgrade release-backup retention.
v1.6.0-node-deploy - Documented --service name validation.
v1.5.0-node-deploy - Documented uninstall purge path allow-list protection.
v1.4.0-node-deploy - Documented install-time systemd unit verification.
v1.3.0-node-deploy - Documented shared install/upgrade deployment locking.
v1.2.0-node-deploy - Documented node-local upgrade locking.
v1.1.0-node-deploy - Documented upgrade-time systemd unit synchronization and
                     rollback behavior.
v1.0.0-node-deploy - Added production deployment documentation.
============================================
-->

## File Purpose

This directory is the production deployment package for AeroNyx Rust privacy
nodes. It gives node operators a predictable path for first install, upgrade,
healthcheck, and systemd service management.

## Files

- `install.sh`: one-command production installer.
- `upgrade.sh`: safe source update, release build, config validation, and
  restart workflow.
- `healthcheck.sh`: read-only node diagnostics and capacity telemetry summary.
- `uninstall.sh`: safe service removal while preserving node identity by default.
- `server.example.toml`: public, safe default config template.
- `aeronyx-server.service`: systemd unit template rendered by `install.sh`.

## First Install

```bash
sudo ./deploy/node/install.sh --registration-code <NODEBOARD_CODE> --start
```

For an existing checkout in a custom path:

```bash
sudo ./deploy/node/install.sh --repo-dir /root/open/AeroNyx --no-build --no-network
```

The installer never overwrites these files when they already exist:

- `/etc/aeronyx/server.toml`
- `/etc/aeronyx/server_key.json`
- `/etc/aeronyx/node_info.json`
- `/etc/aeronyx/aeronyx.env`

Installation and upgrade share a node-local deployment lock, so an operator or
automation system cannot run a second install/upgrade process while one is
already replacing the repository, service unit, binary, or network rules.

When using an existing repository checkout, `install.sh` refuses to pull if
tracked Git files have local staged or unstaged changes. Untracked runtime and
build paths, such as `target/`, `data/`, and local model files, do not block the
check. For emergency maintenance only, an operator can pass `--allow-dirty`.

Before installation, `install.sh` performs non-blocking production preflight
checks for:

- `/dev/net/tun`
- default route interface
- memory
- disk space
- common AeroNyx ports `51820` and `8421`
- commercial capacity plan:
  - configured VPN pool and estimated usable client IPs
  - configured `limits.max_connections`
  - systemd `LimitNOFILE` plus shell file-descriptor soft/hard limit
  - current and maximum Linux conntrack entries

Capacity-plan warnings are non-blocking, but they should be resolved before a
node is placed into paid commercial routing. In particular,
`limits.max_connections` should not exceed usable client IPs in
`vpn.virtual_ip_range`, and the host should keep enough file-descriptor and
conntrack headroom for the configured session target.

The file-descriptor check prefers the installed or template systemd
`LimitNOFILE` value because that is the limit used by the production
`aeronyx-server` service. The shell `ulimit` is still printed as context for
manual debugging.

When network setup is enabled, `install.sh` persists forwarding/NAT with:

- `/etc/sysctl.d/99-aeronyx.conf`
- `/etc/iptables/rules.v4`
- `aeronyx-network-restore.service`

The VPN source subnet and TUN interface are read from the installed
`server.toml` values:

- `vpn.virtual_ip_range`
- `tun.device_name`

This keeps NAT and forwarding rules aligned when operators expand the IP pool
or customize the TUN device for higher-capacity nodes.

Refresh only host forwarding/NAT and reboot recovery after changing
`vpn.virtual_ip_range` or `tun.device_name`:

```bash
sudo ./deploy/node/install.sh --network-only
```

This mode does not pull source, build the Rust binary, register the node,
install the main systemd unit, or restart `aeronyx-server`.

The generated network restore service uses detected absolute paths for
`sysctl` and `iptables-restore` so reboot recovery works across Linux
distributions that place these commands under `/usr/sbin` instead of `/sbin`.

Before installing the main service or generated network restore service,
`install.sh` renders the systemd unit to `/tmp` and verifies it with
`systemd-analyze verify`. A malformed service unit fails before it can replace
the installed unit.

Run preflight only:

```bash
sudo ./deploy/node/install.sh --repo-dir /opt/aeronyx/AeroNyx --preflight-only
```

## Upgrade

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx
```

`upgrade.sh` checks active VPN sessions before restart. If users are connected,
the script stops unless the operator explicitly passes `--force`.

Only one install or upgrade can run on the same node at a time. The script takes
the shared node-local deployment lock before pulling, building, replacing the
systemd unit, or restarting the service.

Before a source upgrade, `upgrade.sh` verifies that tracked Git files are clean.
This prevents a production node from mixing local edits with a pulled release.
Untracked runtime/build data is ignored. For emergency maintenance only, pass
`--allow-dirty`.

During upgrades, the script also renders `deploy/node/aeronyx-server.service`
into the installed systemd unit and verifies it with `systemd-analyze verify`
before restarting. When persisted iptables rules exist, it also regenerates and
verifies `aeronyx-network-restore.service` so existing nodes receive reboot
recovery improvements without a full reinstall.

Build and validate without restarting:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --no-restart
```

Keep the currently installed systemd unit while upgrading the binary:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --skip-unit-update
```

Repair only the main systemd unit without pulling, building, or restarting the
Rust node service:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --service-unit-only
```

The unit-only maintenance modes are intentionally mutually exclusive and cannot
be combined with their matching `--skip-*-update` flags.

Keep the currently installed network restore unit:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --skip-network-restore-update
```

Repair only the reboot network restore unit without pulling, building, or
restarting the Rust node service:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --network-restore-only
```

Post-restart health is polled automatically. If restart or health verification
fails, `upgrade.sh` restores both the previous systemd unit and previous release
binary from `/var/lib/aeronyx/releases`, then restarts the service again.

After a successful upgrade, old backups in `/var/lib/aeronyx/releases` are
pruned per backup type. The default keeps the latest 10 binary backups, latest
10 main systemd unit backups, and latest 10 network restore unit backups:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --keep-releases 20
```

## Healthcheck

```bash
./deploy/node/healthcheck.sh --repo-dir /opt/aeronyx/AeroNyx
```

Machine-readable output for nodeboard or automation:

```bash
./deploy/node/healthcheck.sh --repo-dir /opt/aeronyx/AeroNyx --json-only
```

The healthcheck prints:

- system commands and OS support
- host capacity: TUN, default route, memory, disk, and ports
- runtime metadata: git commit, branch, binary/config timestamps, service state
- live systemd unit binding: WorkingDirectory, ExecStart binary, config path
- config-driven VPN subnet/TUN diagnostics for NAT and FORWARD rules
- tracked worktree and current-start journal warning checks
- release-backup counts for binary, main unit, and network restore unit
- release binary presence
- config validation result
- node registration files
- systemd status
- systemd restart policy: restart mode, restart delay, start limits, timeouts
- systemd hardening status
- IPv4 forwarding, NAT, and reboot persistence hints
- network restore command path checks
- structured JSON runtime fields for release backups and network restore commands
- local VPN health endpoint status
- capacity telemetry: IP pool, conntrack, file descriptors, drops, pps, bps
- capacity risk checks: `max_connections` / policy `max_sessions` versus
  usable VPN IP pool, IP-pool exhaustion, fd usage, conntrack usage, and packet
  drops

It does not print private keys, user traffic destinations, DNS contents,
payloads, wallet-level traffic, or client public IPs.

`--json-only` includes a top-level `capacity` object copied from the local Rust
VPN health endpoint and a `local_vpn_health` summary for nodeboard automation.
These fields remain aggregate-only and preserve the same privacy boundary as
the Rust `/api/vpn/health` response.

## Safe Uninstall

```bash
sudo ./deploy/node/uninstall.sh
```

Default uninstall behavior stops/disables the main service, removes the main
systemd unit, and also stops/disables/removes `aeronyx-network-restore.service`.
It preserves:

- `/etc/aeronyx/server.toml`
- `/etc/aeronyx/server_key.json`
- `/etc/aeronyx/node_info.json`
- `/var/lib/aeronyx`
- `/var/log/aeronyx`
- `/etc/sysctl.d/99-aeronyx.conf`
- `/etc/iptables/rules.v4`

Full purge requires explicit confirmation:

```bash
sudo ./deploy/node/uninstall.sh --purge
```

Even with `--purge --yes`, `uninstall.sh` only deletes paths on the AeroNyx
purge allow-list:

- `/etc/aeronyx`
- `/var/lib/aeronyx`
- `/var/log/aeronyx`
- `/etc/sysctl.d/99-aeronyx.conf`
- `/etc/iptables/rules.v4`

## Important Configuration Items

`server.example.toml` defaults to a commercial VPN node profile:

- VPN listen address: `0.0.0.0:51820`
- virtual IP pool: `100.64.0.0/24`
- TUN device: `aeronyx0`
- max connections: `1000`
- management API: `https://api.aeronyx.network/api/privacy_network`
- MemChain: `off`

`vpn.virtual_ip_range` and `tun.device_name` are operational inputs, not only
application settings. `install.sh` uses them when writing host NAT/FORWARD
rules, and `healthcheck.sh` verifies runtime and persisted rules against the
same values.

`limits.max_connections` is the node-local session ceiling used during install
capacity planning and by the Rust runtime as the default maximum session limit.
Remote nodeboard policy may apply a stricter commercial `max_sessions` value at
runtime; capacity planning should use the lower of the local limit, the remote
policy limit, and available client IPs.

The systemd template applies production-safe hardening:

- `NoNewPrivileges=true`
- `ProtectSystem=full`
- restricted `CapabilityBoundingSet`
- explicit `ReadWritePaths` for `/etc/aeronyx`, `/var/lib/aeronyx`, and
  `/var/log/aeronyx`
- explicit restart limits: `Restart=on-failure`, `RestartSec=5`,
  `StartLimitIntervalSec=300`, `StartLimitBurst=10`

It intentionally does not enable `PrivateDevices` or `ProtectHome` because VPN
nodes need `/dev/net/tun`, and existing deployments may keep the repository
under `/root`.

MemChain and local AI model setup remain available through the existing
`scripts/init.sh` and `scripts/download_models.sh` workflows. They are not part
of the minimal commercial VPN node install path.

## Compatibility

Production node host:

- Linux with systemd
- Ubuntu/Debian preferred
- Fedora/RHEL/CentOS supported on a best-effort package-install basis

Client/development platforms:

- macOS, iOS, Android, and Windows are not production node targets for these
  scripts.
- These scripts do not change mobile or desktop client APIs.
- Scripts that accept `--service` reject names containing `/`, names beginning
  with `-`, and names outside `[A-Za-z0-9_.@-]`.
- Install and upgrade dirty-worktree protection only checks tracked Git files,
  preserving compatibility with untracked runtime/build directories on Linux
  production nodes.

## Next Developer Guide

- Keep install and upgrade idempotent.
- Preserve existing CLI compatibility:
  - `aeronyx-server register`
  - `aeronyx-server start`
  - `aeronyx-server validate`
  - `aeronyx-server status`
- Keep uninstall safe by default. Node identity must not be deleted unless the
  operator explicitly asks for purge.
- Never overwrite private node state unless a future migration explicitly asks
  the operator for confirmation.
- Keep nodeboard compatibility by preserving systemd service name
  `aeronyx-server` unless backend and nodeboard are updated together.
