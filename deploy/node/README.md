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
  retention/diagnostics, plus network restore command-path portability.

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

Before installation, `install.sh` performs non-blocking production preflight
checks for:

- `/dev/net/tun`
- default route interface
- memory
- disk space
- common AeroNyx ports `51820` and `8421`

When network setup is enabled, `install.sh` persists forwarding/NAT with:

- `/etc/sysctl.d/99-aeronyx.conf`
- `/etc/iptables/rules.v4`
- `aeronyx-network-restore.service`

The generated network restore service uses detected absolute paths for
`sysctl` and `iptables-restore` so reboot recovery works across Linux
distributions that place these commands under `/usr/sbin` instead of `/sbin`.

Before installing the main service, `install.sh` renders the systemd template
and verifies it with `systemd-analyze verify`. A malformed service unit fails
before it can replace the installed unit.

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

During restart upgrades, the script also renders
`deploy/node/aeronyx-server.service` into the installed systemd unit and
verifies it with `systemd-analyze verify` before restarting. This keeps live
nodes aligned with the production hardening profile shipped in Git.

Build and validate without restarting:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --no-restart
```

Keep the currently installed systemd unit while upgrading the binary:

```bash
sudo ./deploy/node/upgrade.sh --repo-dir /opt/aeronyx/AeroNyx --skip-unit-update
```

Post-restart health is polled automatically. If restart or health verification
fails, `upgrade.sh` restores both the previous systemd unit and previous release
binary from `/var/lib/aeronyx/releases`, then restarts the service again.

After a successful upgrade, old backups in `/var/lib/aeronyx/releases` are
pruned per backup type. The default keeps the latest 10 binary backups and the
latest 10 systemd unit backups:

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
- tracked worktree and current-start journal warning checks
- release-backup counts for upgrade retention observability
- release binary presence
- config validation result
- node registration files
- systemd status
- systemd hardening status
- IPv4 forwarding, NAT, and reboot persistence hints
- network restore command path checks
- local VPN health endpoint status
- capacity telemetry: IP pool, conntrack, file descriptors, drops, pps, bps

It does not print private keys, user traffic destinations, DNS contents,
payloads, wallet-level traffic, or client public IPs.

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

The systemd template applies production-safe hardening:

- `NoNewPrivileges=true`
- `ProtectSystem=full`
- restricted `CapabilityBoundingSet`
- explicit `ReadWritePaths` for `/etc/aeronyx`, `/var/lib/aeronyx`, and
  `/var/log/aeronyx`

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
