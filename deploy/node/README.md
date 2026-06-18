# AeroNyx Production Node Deployment

<!--
============================================
File Creation/Modification Notes
============================================
Creation Reason:
- Provide operator-facing documentation for the production Rust privacy node
  deployment scripts.

Modification Reason:
- Document that nodeboard-generated preview commands include `--quick` so the
  read-only plan matches the exact first-install path that the operator will
  run after approval.
- Clarify where the unified `deploy/node/aeronyx-node.sh` entrypoint comes
  from, so human operators and AI assistants know to clone/update the AeroNyx
  Rust repository before running repository-local commands.
- Document VPN DNS ownership so production operators can choose the default
  built-in Rust DNS proxy or an external systemd-resolved listener without
  confusing port-bind warnings.
- Document --set-vpn-cidr so operators can update vpn.virtual_ip_range and
  refresh NAT/restore rules in one network-only maintenance command before a
  controlled service restart.
- Document stale AeroNyx NAT cleanup during VPN pool migrations so operators
  know --network-only removes old overlapping 100.64.0.0/* MASQUERADE rules.
- Document read-only --print-plan for verifying generated one-command install
  commands without requiring root access or mutating the host.
- Document environment-variable defaults and --quick first-install mode for
  one-command commercial node setup.
- Document production upgrade unit-template synchronization, rollback behavior,
  shared node-local deployment locking, and install-time systemd unit
  verification, purge path safety, service-name validation, and release-backup
  retention/diagnostics, plus network restore command-path portability and unit
  verification/synchronization, low-risk maintenance, and tracked dirty
  worktree protection, config-driven VPN network rules, network-only
  maintenance, install-time commercial capacity plan checks, and healthcheck
  capacity-risk JSON export. Document the /22 default VPN pool that matches the
  commercial 1000-session profile, and document healthcheck repo path
  auto-detection for non-standard node checkouts.

Main Functionality:
- Explains first install, registration, upgrade, healthcheck, configuration
  ownership, compatibility scope, and next-developer guidance.

Dependencies:
- deploy/node/install.sh
- deploy/node/upgrade.sh
- deploy/node/healthcheck.sh
- deploy/node/aeronyx-node.sh
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
v1.32.0-node-deploy - Documented quick install preview alignment.
v1.31.0-node-deploy - Documented aeronyx-node.sh GitHub origin and
                     repository-local execution path.
v1.30.0-node-deploy - Documented VPN DNS ownership modes.
v1.29.0-node-deploy - Documented --set-vpn-cidr network-only VPN pool updates.
v1.28.0-node-deploy - Documented stale NAT cleanup for VPN pool migrations.
v1.27.0-node-deploy - Documented --print-plan for safe install command checks.
v1.26.0-node-deploy - Documented --quick and AERONYX_* install defaults.
v1.25.0-node-deploy - Documented healthcheck systemd repo path auto-detection.
v1.24.0-node-deploy - Documented /22 default VPN pool for 1000-session
                     commercial capacity.
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

## Where `aeronyx-node.sh` Comes From

`./deploy/node/aeronyx-node.sh` is not a Linux system command and is not
installed globally by default. It is part of the open-source AeroNyx Rust
repository:

```bash
https://github.com/AeroNyxNetwork/AeroNyx
```

After cloning or updating the repository, the script path is:

```bash
AeroNyx/deploy/node/aeronyx-node.sh
```

Every command that starts with `./deploy/node/aeronyx-node.sh` expects the
current shell to already be inside the `AeroNyx` repository. From a fresh
server, start with:

```bash
mkdir -p /root/open
cd /root/open
git clone https://github.com/AeroNyxNetwork/AeroNyx.git AeroNyx
cd AeroNyx
./deploy/node/aeronyx-node.sh plan --repo-dir "$PWD" --branch main
```

If the repository already exists, update it first:

```bash
cd /root/open/AeroNyx
git fetch origin main
git checkout main
git pull --ff-only origin main
./deploy/node/aeronyx-node.sh plan --repo-dir "$PWD" --branch main
```

## Files

- `install.sh`: one-command production installer.
- `upgrade.sh`: safe source update, release build, config validation, and
  restart workflow.
- `aeronyx-node.sh`: unified operator entrypoint that delegates to install,
  upgrade, healthcheck, status, logs, and network maintenance commands.
- `healthcheck.sh`: read-only node diagnostics and capacity telemetry summary.
- `uninstall.sh`: safe service removal while preserving node identity by default.
- `server.example.toml`: public, safe default config template.
- `aeronyx-server.service`: systemd unit template rendered by `install.sh`.

## First Install

```bash
sudo ./deploy/node/install.sh --registration-code <NODEBOARD_CODE> --start
```

For the lowest-friction first install, pass the nodeboard registration code as
an environment variable and let `--quick` keep the normal commercial defaults:

```bash
sudo AERONYX_REGISTRATION_CODE=<NODEBOARD_CODE> ./deploy/node/install.sh --quick
```

`--quick` is intentionally a thin wrapper. It still runs preflight checks,
capacity-plan warnings, package/Rust setup, repository update, config
installation, network setup, release build, systemd verification, node
registration, and service start. It fails when no registration code is
provided, so an operator does not mistake an unregistered node for a live
commercial node.

The installer also accepts these environment defaults for automation systems
that generate one-line setup commands:

- `AERONYX_REPO_URL`
- `AERONYX_BRANCH`
- `AERONYX_REPO_DIR`
- `AERONYX_REGISTRATION_CODE`
- `AERONYX_START=1`

Verify a generated command without root access, package installation, network
changes, registration, or service start:

```bash
AERONYX_REGISTRATION_CODE=<NODEBOARD_CODE> ./deploy/node/aeronyx-node.sh plan --repo-dir "$PWD" --branch main --quick
```

`aeronyx-node.sh plan --quick` delegates to the same read-only `install.sh
--quick --print-plan` path used by the lower-level installer. It hides the
registration code value and prints only whether a code is present. This makes
the preview safe to paste into support tickets and nodeboard diagnostic logs,
while matching the actual quick install command nodeboard displays after
operator approval.

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

## VPN DNS Ownership

Commercial VPN clients need DNS on the tunnel gateway, normally
`100.64.0.1:53`. AeroNyx supports two ownership modes:

- Built-in Rust proxy: keep `vpn.dns_proxy_enabled = true`. The Rust node binds
  UDP `gateway_ip:53` and forwards opaque DNS datagrams to upstream resolvers.
- External host resolver: set `vpn.dns_proxy_enabled = false` and configure a
  host resolver, for example systemd-resolved, to listen on `gateway_ip:53`.

The default remains `true` for backward compatibility. Use the external mode
only when the host resolver is intentionally managed by operations automation.
The health endpoint still checks for a DNS listener and performs a DNS query
through `gateway_ip:53`; it does not expose user DNS contents or destinations.

For the common commercial pool expansion from `/24` to `/22`, update the
persisted config and refresh host networking in one idempotent command:

```bash
sudo ./deploy/node/install.sh --network-only --set-vpn-cidr 100.64.0.0/22
```

`--set-vpn-cidr` is intentionally restricted to `--network-only`. It creates a
timestamped backup such as:

```text
/etc/aeronyx/server.toml.bak.20260617T045733Z.vpn_cidr
```

Then it updates only `[vpn].virtual_ip_range` in `/etc/aeronyx/server.toml`,
prints the refreshed capacity plan, applies the matching MASQUERADE rule, and
persists reboot recovery. It does **not** restart `aeronyx-server`; the running
Rust process and TUN prefix change only after a controlled restart.

Recommended safe maintenance sequence for a live commercial node:

1. Set the node to maintenance mode from nodeboard or backend policy.
2. Wait until active sessions drain to zero.
3. Run `sudo ./deploy/node/install.sh --network-only --set-vpn-cidr 100.64.0.0/22`.
4. Restart `aeronyx-server` during the maintenance window.
5. Verify `ip addr show aeronyx0`, `ip route`, nodeboard capacity, and backend
   `data.nodes[].system.capacity`.
6. End maintenance mode after the backend heartbeat reports the new capacity.

When the VPN pool changes, for example from `100.64.0.0/24` to
`100.64.0.0/22`, `--network-only` removes stale AeroNyx
`100.64.0.0/*` MASQUERADE rules on the detected egress interface before
persisting `/etc/iptables/rules.v4`. The cleanup is scoped to the AeroNyx
CGNAT pool so unrelated host NAT rules are left alone.

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

`upgrade.sh` writes a local structured progress snapshot to:

```text
/var/lib/aeronyx/upgrade-status.json
```

The file contains only operator workflow metadata: status, step, message,
repo path, branch, service name, config path, `--no-restart`, `--force`, and
`updated_at`. It intentionally excludes registration codes, private keys,
client public IPs, DNS contents, destinations, packet payloads, chat plaintext,
voucher secrets, and wallet-level traffic. `aeronyx-node.sh status` displays a
short summary of this file, and `healthcheck.sh --json-only` exposes it as
top-level `upgrade_status` for nodeboard or AI maintenance automation.

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

When `--repo-dir` is omitted, `healthcheck.sh` reads the live systemd
`WorkingDirectory` first and then the `ExecStart` binary path before falling
back to `/opt/aeronyx/AeroNyx`. Pass `--repo-dir` explicitly when auditing a
different checkout than the currently running service.

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
- upgrade workflow status from `/var/lib/aeronyx/upgrade-status.json`
- capacity telemetry: IP pool, conntrack, file descriptors, drops, pps, bps
- capacity risk checks: `max_connections` / policy `max_sessions` versus
  usable VPN IP pool, IP-pool exhaustion, fd usage, conntrack usage, and packet
  drops

It does not print private keys, user traffic destinations, DNS contents,
payloads, wallet-level traffic, or client public IPs.

`--json-only` includes top-level `capacity` and `upgrade_status` objects plus a
`local_vpn_health` summary for nodeboard automation.
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
- virtual IP pool: `100.64.0.0/22`
- TUN device: `aeronyx0`
- max connections: `1000`
- management API: `https://api.aeronyx.network/api/privacy_network`
- MemChain: `off`

`vpn.virtual_ip_range` and `tun.device_name` are operational inputs, not only
application settings. `install.sh` uses them when writing host NAT/FORWARD
rules, and `healthcheck.sh` verifies runtime and persisted rules against the
same values.

The default `100.64.0.0/22` pool gives roughly 1021 usable client addresses
after the gateway reservation, which matches the default `max_connections =
1000` commercial profile. Existing nodes are not rewritten automatically:
expand a live pool only during an operator-approved maintenance window, then
run `install.sh --network-only` to refresh NAT/FORWARD rules and restart the
Rust service only after active sessions are safely drained.

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
