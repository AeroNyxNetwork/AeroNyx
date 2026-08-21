# AeroNyx Production Node Deployment

<!--
============================================
File Creation/Modification Notes
============================================
Creation Reason:
- Provide operator-facing documentation for the production Rust privacy node
  deployment scripts.

Modification Reason:
- [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Document the explicit
  opt-in, exact-pin pre-expiry renewal path inside the supervised runtime gate.
- [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] Document the
  hard-bounded concurrent witness round and its durable-before-counting rule.
- [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] Document edge-triggered
  renewal warnings, duplicate suppression, and explicit recovery events.
- [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Document exact aggregate
  threshold lifetime and local-only pre-expiry renewal warnings.
- [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Document the opt-in,
  network-silent runtime re-audit and supervised fail-closed recovery policy.
- [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] Document the bounded
  row-copy and lock-free cryptographic verification boundary for read audits.
- [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Document that startup
  and operator status derive from one typed, cryptographically audited snapshot.
- [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Document the opt-in,
  network-silent fail-closed startup gate and one-sided receipt freshness.
- [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] Document explicit,
  signed-snapshot-pinned network collection with durable receipt re-audit and
  fail-closed command status, without enabling a background scheduler.
- [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Document restart-safe,
  current-checkpoint local receipt-vault auditing and optional readiness exit.
- [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Document bounded
  producer-side import of an operator-carried signed witness receipt, current
  checkpoint binding, durable vault re-audit, and adverse-evidence retention.
- [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Document independent-node
  countersigning, durable producer-scoped high-water state, signed negative
  decisions, and exact offline producer/witness verification.
- [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Document exact create-new export,
  offline verification, rollback-floor retention, and the boundary between a
  producer-signed anchor and future independent witness evidence.
- [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] Document automatic,
  crash-safe maintenance audit segmentation and authenticated checkpoints.
- [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Document bounded verification
  of the private HMAC-chained custody maintenance history.
- [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Document short-lived,
  state-bound restore plans and their non-authorization security boundary.
- [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Document the
  non-destructive latest-image recovery preflight and stable blocker codes.
- [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Document host-local custody
  backup audit, default dry-run, mandatory stop/confirmation gates, and the
  private HMAC-chained aggregate maintenance log.
- [NODE-ADMISSION-GATE 2026-08-02 by Codex] Document the bounded acceptance
  gate that prevents systemd-only starts from being reported as successful
  network admission.
- [REGISTRATION-CODE-STDIN 2026-08-02 by Codex] Document hidden interactive
  onboarding and bounded stdin automation so one-time codes do not appear in
  child process command lines.
- [NODE-REGISTRATION-PROFILE 2026-08-02 by Codex] Document explicit public
  VPN registration metadata and signed discovery restart defaults so a fresh
  node does not require manual backend or server.toml repair.
- [SESSION-GATED-PROMOTION 2026-07-31 by Codex] Document post-build session
  gates that prevent binary/unit promotion from leaving a half-deployed node
  when traffic appears during a long release build.
- [LIVE-BUILD-RESOURCE-GUARD 2026-07-31 by Codex] Document bounded Cargo
  parallelism and reduced CPU/I/O scheduling priority for upgrades that build
  on a host still serving privacy-network traffic.
- [COMMIT-PINNED-SOURCE 2026-07-29 by Codex] Document exact commit-pinned
  isolated upgrades for production nodes whose runtime checkout is dirty or
  intentionally diverged from the reviewed GitHub release.
- [BUILD-CACHE-MAINTENANCE 2026-07-26 by Codex] Document read-only Cargo cache
  inventory and explicitly confirmed pruning that preserves the production
  binary, current pinned build cache, release backups, and all node state.
- [PINNED-RUST-BUILD 2026-07-26 by Codex] Document exact Rust toolchain
  pinning, isolated build targets, staged validation, and atomic binary
  promotion for reproducible node releases.
- Document the `aeronyx-node.sh refresh-bootstrap` and `fleet-drift-check`
  commands so production nodes can refresh signed discovery bootstrap
  snapshots and audit seed/binary/config drift without exposing user data.
- Document the gated `aeronyx-node.sh relay-probe --two-hop` operator command,
  which attempts an outer+onward live middle-hop proof only when three distinct
  routeable nodes exist.
- Document the Rust BlindRelay `onward_envelope` handler support for controlled
  no-exit middle-hop experiments, including the production requirement for a
  third non-returning node before a full live two-hop probe can be claimed.
- Document the `aeronyx-node.sh relay-probe` evidence boundary: it proves
  single-hop BlindRelay transport with a synthetic opaque blob, while reporting
  two-hop OnionMiddle readiness separately until the protocol adds a path-aware
  encrypted route envelope.
- Document the guarded `aeronyx-node.sh chat-relay` helper for enabling or
  disabling blind ChatRelay with config backup, validation, active-session
  warning, and optional restart.
- Document ChatRelay capability readiness so operators understand how
  `[memchain.chat_relay]`, the public peer API, descriptor advertisement, and
  peer quorum route readiness relate without exposing relay payloads or user
  metadata.
- Document that `aeronyx-node.sh status` includes the healthcheck
  operator_action recommendation so ordinary operators can see the next step
  without parsing JSON or service logs.
- Document recent operational event severity mapping so nodeboard can
  prioritize critical service failures without exposing raw logs or user data.
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
3. The installer verifies bounded network admission before reporting success.
4. Upgrades compile with a live-safe resource policy before atomic promotion.
5. systemd runs aeronyx-server and healthcheck.sh verifies runtime status.

Important Note for Next Developer:
- Do not document workflows that require exposing private keys or user traffic.
- Keep the commands compatible with Linux/systemd production nodes.
- macOS, iOS, Android, and Windows are client/development platforms for this
  deployment package, not production node targets.

Last Modified:
v1.61.0-node-deploy - Documented bounded concurrent custody witness collection.
v1.60.0-node-deploy - Documented custody renewal warning lifecycle.
v1.59.0-node-deploy - Documented custody quorum expiry preflight telemetry.
v1.58.0-node-deploy - Documented strict runtime custody witness re-auditing.
v1.57.0-node-deploy - Documented one-shot durable witness collection.
v1.56.0-node-deploy - Documented current-checkpoint witness vault re-audit.
v1.55.0-node-deploy - Documented host-local custody witness receipt import.
v1.54.0-node-deploy - Documented segmented custody audit checkpoints.
v1.53.0-node-deploy - Documented custody maintenance audit verification.
v1.52.0-node-deploy - Documented authenticated relay restore planning.
v1.51.0-node-deploy - Documented read-only relay restore readiness.
v1.50.0-node-deploy - Documented confirmation-gated relay custody pruning.
v1.49.0-node-deploy - Documented post-start network admission acceptance.
v1.48.0-node-deploy - Documented secret-safe registration-code input for the
                     unified quickstart and lower-level installer.
v1.47.0-node-deploy - Documented policy-safe public VPN onboarding and signed
                     discovery bootstrap defaults.
v1.46.0-node-deploy - Documented transactional post-build session gates.
v1.45.0-node-deploy - Documented live-safe same-host release builds.
v1.44.0-node-deploy - Documented exact commit-pinned isolated source upgrades.
v1.43.0-node-deploy - Documented guarded Cargo build-cache maintenance.
v1.42.0-node-deploy - Documented pinned Rust builds and atomic promotion from
                     isolated toolchain/service-scoped Cargo targets.
v1.41.0-node-deploy - Documented bootstrap refresh and fleet drift check commands.
v1.40.0-node-deploy - Documented gated relay-probe --two-hop live proof mode.
v1.39.0-node-deploy - Documented optional onward envelope support for controlled
                     two-hop middle-hop forwarding.
v1.38.0-node-deploy - Documented relay-probe single-hop evidence and two-hop
                     readiness boundary.
v1.37.0-node-deploy - Documented guarded OnionMiddle config helper for no-exit
                     two-hop encrypted relay readiness.
v1.36.0-node-deploy - Documented guarded ChatRelay config helper.
v1.35.0-node-deploy - Documented ChatRelay capability readiness and peer quorum
                     route-ready checks.
v1.34.0-node-deploy - Documented status operator recommendation.
v1.33.0-node-deploy - Documented recent error severity mapping.
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

## Reproducible Rust Builds

Production node builds are controlled by two repository files:

- `rust-toolchain.toml` pins one exact Rust compiler release. A moving
  `stable`, `beta`, or `nightly` channel is not accepted for release builds.
- `Cargo.lock` pins the complete dependency graph and is always consumed with
  `cargo build --locked`.

`install.sh` and `upgrade.sh` resolve the exact toolchain, install it through
rustup when permitted, and compile into:

```text
/var/lib/aeronyx/build-targets/rust-<version>/<service-name>
```

This path is intentionally separate from the stable systemd binary path:

```text
<repo>/target/release/aeronyx-server
```

The scripts first build and validate the isolated candidate. Only then do they
copy it to a same-filesystem staging path and atomically rename it over the
stable binary. The currently running process is never used as Cargo output.
Upgrade rollback continues to use timestamped binaries under:

```text
/var/lib/aeronyx/releases
```

Hosts that run a non-default isolated node may override the build root without
changing source:

```bash
sudo AERONYX_BUILD_TARGET_ROOT=/var/lib/aeronyx-jp1/build-targets \
  ./deploy/node/upgrade.sh --service aeronyx-server-jp1
```

An exact toolchain bump is a release change. It requires the full Rust test
suite, Clippy, a locked release build, and controlled node rollout evidence.

### Live-safe upgrade builds

`upgrade.sh` assumes the current node may continue serving VPN, discovery,
blind relay, and Directory Chain APIs while the next release is compiling.
Its default `live` policy therefore:

- uses approximately half of the online CPUs for Cargo, with at least one job;
- runs the compiler with CPU nice level 10;
- uses the idle I/O scheduling class when `ionice` is available;
- records the selected priority, job count, nice level, and I/O class in the
  privacy-safe upgrade status snapshot.

The policy affects compilation only. It does not change the running systemd
service, protocol threads, active-session limits, node identity, or stored
state:

```bash
sudo ./deploy/node/aeronyx-node.sh upgrade \
  --repo-dir /root/open/AeroNyx \
  --build-priority live \
  --build-jobs auto
```

On a two-CPU node, `live` plus `auto` resolves to one Cargo job. An operator may
choose an explicit positive job count no larger than the online CPU count:

```bash
sudo ./deploy/node/aeronyx-node.sh upgrade \
  --repo-dir /root/open/AeroNyx \
  --build-jobs 1
```

`normal` uses all online CPUs by default and does not lower CPU or I/O
priority. Use it only in an approved maintenance window after traffic has
drained:

```bash
sudo ./deploy/node/aeronyx-node.sh upgrade \
  --repo-dir /root/open/AeroNyx \
  --build-priority normal
```

Automation may set `AERONYX_BUILD_PRIORITY` and `AERONYX_BUILD_JOBS`; explicit
command-line options override those defaults. Invalid modes, non-positive job
counts, and job counts above the online CPU count fail during preflight before
source, binary, service, or protocol state is changed.

## Cargo Build-Cache Maintenance

Production compilation can consume significant disk space because Cargo keeps
dependency objects, incremental data, and old toolchain targets. Inspect the
node without changing the host:

```bash
./deploy/node/aeronyx-node.sh build-cache \
  --repo-dir /root/open/AeroNyx
```

The inventory reports the legacy repository `target/`, the isolated build
root, the exact pinned-toolchain target, the protected binary SHA-256, and
filesystem capacity. Preview every removable entry before deletion:

```bash
sudo ./deploy/node/aeronyx-node.sh prune-build-cache \
  --repo-dir /root/open/AeroNyx \
  --dry-run
```

Run the controlled prune only after reviewing that output:

```bash
sudo ./deploy/node/aeronyx-node.sh prune-build-cache \
  --repo-dir /root/open/AeroNyx \
  --yes
```

The prune command takes the same deployment lock used by install and upgrade.
It verifies that a running systemd service maps to the protected stable binary,
records that binary's SHA-256 before cleanup, and verifies the hash again
afterward. It removes only regenerable Cargo entries:

- Legacy repository debug/cross-target directories and known Cargo-generated
  release caches such as `deps`, `build`, `.fingerprint`, `incremental`,
  `.rlib`, and `.d`.
- Older pinned-toolchain targets for the same systemd service.

It deliberately preserves:

- The current pinned-toolchain/service build target.
- The stable production binary.
- Other release executables, staged binaries, and historical rollback artifacts
  found under `<repo>/target/release`.
- `/var/lib/aeronyx/releases` rollback binaries.
- `/var/lib/aeronyx` protocol state, ledgers, peer stores, and node data.
- `/etc/aeronyx` configuration, identity, and key material.
- Build targets belonging to other AeroNyx services on the same host.

Cache pruning does not restart the service and does not require an active
session drain. The running binary and all runtime state remain in place.

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

The recommended human workflow uses the unified entrypoint. It prompts for the
nodeboard registration code with terminal echo disabled, previews the resolved
plan, asks for confirmation, and then installs, registers, starts, and verifies
the node:

```bash
sudo ./deploy/node/aeronyx-node.sh quickstart
```

Register a named public VPN node without a follow-up database edit:

```bash
sudo ./deploy/node/aeronyx-node.sh quickstart \
  --node-name TW1 \
  --region TW \
  --public-vpn
```

The legacy `--registration-code <NODEBOARD_CODE>` option remains supported,
but it can be visible to same-host process inspection while the installer is
running. Prefer the hidden prompt above. For non-interactive automation, pass
the credential over one bounded stdin line and use `--yes` only after the
generated plan has been approved:

```bash
read -r -s -p 'Nodeboard registration code: ' AERONYX_NODE_CODE; echo
printf '%s\n' "${AERONYX_NODE_CODE}" | sudo ./deploy/node/aeronyx-node.sh \
  quickstart --registration-code-stdin --node-name TW1 --region TW --public-vpn --yes
unset AERONYX_NODE_CODE
```

The wrapper forwards the credential between shell/Rust processes through
anonymous pipes, and curl reads the install-progress JSON from stdin. No child
command line contains the code. Plan output contains only
`registration_code_present=yes`; it never includes the code value.

`--public-vpn` is an explicit operator choice. Without it, the Rust runtime is
still registered as VPN-capable with its configured listener port, but remains
private in nodeboard and is not returned by the public VPN pool endpoint.

The environment variable remains available for backward-compatible automation,
but stdin is preferred because inherited environments can be inspected by
same-privilege processes:

```bash
sudo AERONYX_REGISTRATION_CODE=<NODEBOARD_CODE> ./deploy/node/install.sh --quick
```

`--quick` is intentionally a thin wrapper. It still runs preflight checks,
capacity-plan warnings, package/Rust setup, repository update, config
installation, network setup, release build, systemd verification, node
registration, service start, and bounded network admission verification. It
fails when no registration code is provided, so an operator does not mistake
an unregistered node for a live commercial node.

### Post-start admission gate

An active systemd unit is necessary but does not prove that a new node joined
the AeroNyx privacy network. Before the installer reports `completed`, the
default admission gate waits up to 120 seconds for all applicable evidence:

- `/api/vpn/health` reports `status=ok`, proving the Rust listener, TUN,
  forwarding, NAT, DNS, and egress checks are usable.
- A registered node has a fresh backend policy timestamp, proving a completed
  signed management heartbeat round trip rather than only a local process
  start.
- `/api/discovery/status` reports a consistent local relay capability, at
  least one validated peer, and a completed gossip round.
- `/api/discovery/snapshot` exposes at least one validated signed descriptor.
- A node installed with `--public-vpn` appears under its exact backend UUID in
  the public privacy-network pool with `visibility=public`, VPN capability,
  and `status=online`.

Private nodes are deliberately not required to appear in the public pool.
They still must pass local health, backend heartbeat (when registered), and
signed discovery checks. The gate reads only aggregate runtime and routing
metadata; it never reads encrypted payloads, destinations, DNS contents,
client addresses, private keys, wallet traffic, or social graph data.

Slow first boots can extend the bounded window without weakening the checks:

```bash
sudo ./deploy/node/aeronyx-node.sh quickstart --admission-timeout 240
```

`--skip-admission-check` is retained only for isolated development and
operator recovery. It is explicit, emits a warning, and should not be used by
normal nodeboard-generated production onboarding commands.

The installer also accepts these environment defaults for automation systems
that generate one-line setup commands:

- `AERONYX_REPO_URL`
- `AERONYX_BRANCH`
- `AERONYX_REPO_DIR`
- `AERONYX_REGISTRATION_CODE`
- `AERONYX_NODE_NAME`
- `AERONYX_NODE_REGION`
- `AERONYX_PUBLIC_VPN=1`
- `AERONYX_ADMISSION_TIMEOUT=120`
- `AERONYX_ADMISSION_CHECK=0` (isolated development/recovery only)
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

## ChatRelay Capability Readiness

ChatRelay is the blind relay layer for E2E-encrypted chat and encrypted media
envelopes. It is separate from the VPN data plane and separate from local
Memory Chain mining. Relay nodes must stay blind: they store and forward
ciphertext plus delivery metadata only, and must not inspect chat plaintext,
message content, client public IPs, DNS contents, destinations, browsing
history, wallet-level traffic, voucher secrets, or private keys.

The default commercial node template keeps ChatRelay disabled:

```toml
[memchain]
mode = "off"

[memchain.chat_relay]
enabled = false
db_path = "/var/lib/aeronyx/chat_pending.db"
```

To advertise this node as a routeable ChatRelay peer, enable it only after the
public peer API is reachable:

```toml
[discovery]
public_endpoint = "https://node.example.com"
public_api_listen_addr = "0.0.0.0:8422"

[memchain.chat_relay]
enabled = true
db_path = "/var/lib/aeronyx/chat_pending.db"
peer_relay_requests_per_minute = 1200
peer_relay_authenticated_requests_per_minute = 240
custody_backup_retention_target_artifacts = 8
custody_backup_retention_target_bytes = 8589934592
custody_backup_partial_grace_secs = 86400
```

<!-- [PEER-RELAY-ADMISSION 2026-08-15 by Codex] -->
`peer_relay_requests_per_minute` bounds both direct relay versions before JSON
parsing. It is intentionally node-global because v1 cannot authenticate a
previous hop and permissionless identities can be rotated. AeroNyx does not
create rate-limit buckets from user keys, receiver keys, source IPs, endpoints,
or ciphertext metadata.

<!-- [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] -->
`peer_relay_authenticated_requests_per_minute` adds a bounded fairness ceiling
for one direct-relay v2 node identity only after its Ed25519 signature verifies.
It cannot replace the global ceiling and does not imply Sybil resistance,
operator trust, or permissioned membership.

The recommended path is the guarded node entrypoint helper:

```bash
./deploy/node/aeronyx-node.sh chat-relay --enable-chat-relay --dry-run
sudo ./deploy/node/aeronyx-node.sh chat-relay --enable-chat-relay --restart
```

The helper creates a timestamped `/etc/aeronyx/server.toml` backup, updates only
`[memchain.chat_relay].enabled`, validates the config, restores the backup if
validation fails, and refuses a restart while active sessions are present unless
the operator explicitly passes `--yes` during a maintenance window.

To disable the blind relay capability:

```bash
sudo ./deploy/node/aeronyx-node.sh chat-relay --disable-chat-relay --restart
```

### Relay custody backup maintenance

<!-- [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] -->
Relay custody maintenance is host-local. It does not add a CMS, HTTP, or
Nodeboard deletion endpoint. The retention settings are planning targets, not
background timers, and the newest fully verified recovery image is always
preserved even when it alone exceeds the configured byte target.

Audit the private backup boundary without deleting or writing an audit record:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody audit -c /etc/aeronyx/server.toml --json
```

Authenticate the complete private maintenance history independently of the
current backup inventory:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody verify-audit -c /etc/aeronyx/server.toml --json
```

<!-- [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] -->
`verify-audit` loads the node identity key locally and replays the audit from
genesis under the same cross-process maintenance lock used by backup and prune.
It rejects truncation, malformed or unknown records, sequence/hash-chain
discontinuity, an invalid HMAC or wrong node key, permission drift, oversized
records/files, and a file whose length changes during verification. An absent
audit is a valid verified history with zero records; the command does not create
or repair the audit file. Output is limited to aggregate record/phase/byte
counts and the last timestamp. It never emits paths, filenames, MACs, operation
IDs, identities, routes, ciphertext, or custody contents.

<!-- [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] -->
When the active audit reaches 64 MiB or 65,536 records, the next append rotates
it automatically. The node preserves the global v1 sequence and record-MAC
chain, hashes the immutable segment with SHA-256, and publishes a separate
node-secret HMAC checkpoint containing only cumulative aggregates. Checkpoint
publication, immutable hard-link publication, active-name retirement, and
parent-directory fsync are ordered so a power loss leaves one of two detectable
recovery states. `verify-audit` reports either state through `rotation_pending`;
the next locked maintenance append finishes the publication before writing a
new record and removes only strictly named, owner-private checkpoint
temporaries abandoned by an interrupted publication. Verification remains
bounded to 16 immutable segments and 1 GiB of authenticated audit bytes.

These are host-local integrity checkpoints, not public consensus or an external
timestamp witness. They detect modification, gaps, and partial publication in
the retained local history, but a root operator who deletes or rolls back every
audit/checkpoint artifact cannot be detected without a separately anchored
witness. Do not describe this mechanism as a blockchain or third-party proof.

Export the latest complete checkpoint as a portable producer-signed anchor:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody create-audit-anchor \
  -c /etc/aeronyx/server.toml \
  --output /root/relay-custody-anchor.bin \
  --json
```

<!-- [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] -->
The command first verifies the entire private audit under the cross-process
maintenance lock, then signs a fixed-size canonical frame with the node Ed25519
identity. It refuses an absent checkpoint, a wrong identity key, an incomplete
rotation, malformed private state, and an existing output path. On Unix the new
binary file is owner-private, opened without following the final symlink, synced
before success, and its parent directory is synced. The output report contains
the exact frame SHA-256, producer node identity, checkpoint generation,
aggregate archived record/byte counts, and opaque anchor digest. It never emits
the private checkpoint HMAC, operation IDs, paths, messages, routes, endpoints,
ciphertext, memory contents, destinations, DNS, or social-graph metadata.

Copy both the exact binary frame and its JSON report to a separately
administered evidence retainer. That retainer must preserve the highest accepted
`checkpoint_generation` and the corresponding frame SHA-256 for each pinned
producer. Verify without the producer config or private key:

```bash
/opt/aeronyx/aeronyx-server relay-custody verify-audit-anchor \
  --input ./relay-custody-anchor.bin \
  --expected-sha256 <64-hex-frame-sha256> \
  --expected-node <64-hex-producer-node-id> \
  --minimum-checkpoint-generation <last-trusted-generation> \
  --json
```

Verification rejects a non-regular or symlinked input, empty/oversized/changed
file, wrong exact-frame hash, padded or non-canonical encoding, invalid
signature, unexpected producer, and generation below the verifier-owned floor.
Active audit-tail records are intentionally not covered until their segment is
checkpointed. The anchor has no producer-controlled timestamp: repeated exports
of the same checkpoint are byte-for-byte identical and keep the same frame
SHA-256. A future independent witness must add and sign its own observation
time rather than treating the producer clock as trusted time.

This portable anchor makes complete local rollback detectable only when an
independent retainer compares it with previously retained evidence. It is still
not a witness receipt, validator vote, consensus checkpoint, transaction proof,
or global finality. The optional independent witness workflow below countersigns
the exact frame without gaining access to the private audit or user data.

### Independent custody checkpoint witness

Copy the exact anchor frame to a separately administered AeroNyx node whose
Ed25519 identity differs from the producer. On that witness node, pin the
producer identity and exact anchor SHA-256 shown by `create-audit-anchor`:

```bash
sudo /opt/aeronyx/aeronyx-server relay-custody witness-audit-anchor \
  -c /etc/aeronyx/server.toml \
  --input ./relay-custody-anchor.bin \
  --expected-sha256 <64-hex-anchor-frame-sha256> \
  --expected-producer <64-hex-producer-node-id> \
  --minimum-checkpoint-generation <operator-trusted-first-generation> \
  --output ./relay-custody-witness.bin \
  --json
```

<!-- [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] -->
The command verifies the producer signature, exact canonical anchor bytes,
producer pin, and operator-owned bootstrap floor before touching witness state.
It then atomically persists one high-water generation and exact anchor-frame
SHA-256 for that producer in the witness node's local MemChain SQLite database.
The custody witness table is physically separate from delivery-cache witness
state because those generation counters advance independently.

After the first accepted observation, only the exact next generation advances.
Repeating the same frame is idempotent. Older, same-generation-different-frame,
and skipped-generation requests produce signed `stale`, `conflict`, or `gap`
receipts without replacing the retained high-water row. The CLI writes those
negative receipts before returning failure so an operator can preserve the
evidence. A failed receipt-file write is safely retryable: durable state has
already advanced and the retry becomes an authenticated idempotent decision.

Copy the exact receipt frame and its JSON report back to the evidence retainer.
Verify the complete producer-to-witness binding offline:

```bash
/opt/aeronyx/aeronyx-server relay-custody verify-audit-witness \
  --anchor ./relay-custody-anchor.bin \
  --anchor-sha256 <64-hex-anchor-frame-sha256> \
  --receipt ./relay-custody-witness.bin \
  --receipt-sha256 <64-hex-receipt-frame-sha256> \
  --expected-producer <64-hex-producer-node-id> \
  --expected-witness <64-hex-independent-witness-node-id> \
  --minimum-checkpoint-generation <last-trusted-generation> \
  --json
```

Successful verification requires an `advanced` or `idempotent` receipt and
checks both Ed25519 signatures, both exact frame hashes, canonical encoding,
independent producer/witness identities, the producer generation floor, and
the receipt-to-anchor binding. The witness supplies its own signed observation
time; the producer still supplies no trusted timestamp.

Return the exact anchor, receipt, and both operator-recorded SHA-256 pins to the
producer. Pin that independent witness in
`discovery.custody_audit_witness_node_ids`, then import the receipt into the
producer's durable evidence vault:

```bash
sudo /opt/aeronyx/aeronyx-server relay-custody import-audit-witness \
  -c /etc/aeronyx/server.toml \
  --anchor ./relay-custody-anchor.bin \
  --anchor-sha256 <64-hex-anchor-frame-sha256> \
  --receipt ./relay-custody-witness.bin \
  --receipt-sha256 <64-hex-receipt-frame-sha256> \
  --expected-witness <64-hex-independent-witness-node-id> \
  --max-age-seconds 7200 \
  --json
```

<!-- [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] -->
Import is host-local and performs no HTTP request. It loads the producer
identity from the local config, requires persistent MemChain storage, verifies
both exact frame hashes and both signatures, requires the witness to be in the
producer's current pin set, and regenerates the producer checkpoint before
accepting the receipt. A correctly signed receipt for an older checkpoint is
therefore rejected even when it was once valid.

`--max-age-seconds` is explicit and bounded from 60 seconds to seven days. The
selected value and `operator_import` admission type are persisted with that
receipt and revalidated after restart. Automatic network receipt persistence
remains a separate typed path fixed at 60 seconds. Existing schema-v17 rows
migrate conservatively to that strict live policy.

The command re-audits every canonical receipt before commit and evaluates the
configured exact-anchor threshold afterward. An accepted receipt may report
`ready` or `collecting`; a signed `stale`, `conflict`, or `gap` receipt is still
retained for operator review and then returns a non-zero command result. Do not
delete adverse evidence merely to make policy appear healthy.

The witness stores only producer identity, checkpoint generation, exact opaque
frame SHA-256, and observation time. It never receives the private audit HMAC,
custody paths, messages, routes, endpoints, payloads, ciphertext, memory,
destinations, DNS, or social graph. One receipt proves one independent node's
durable observation. Multiple receipts improve administrative independence but
are not consensus, fork choice, validator voting, or global finality.

For an explicit online round, obtain a fresh signed discovery snapshot from a
trusted AeroNyx discovery source and keep it as a regular local file. The
command parses at most 512 KiB, verifies descriptors again at execution time,
and discards every descriptor whose identity is not currently pinned in
`discovery.custody_audit_witness_node_ids`:

```bash
curl --fail --proto '=https' --tlsv1.2 --max-time 20 \
  https://<trusted-discovery-node>/api/discovery/snapshot \
  --output ./aeronyx-witness-snapshot.json

sudo /opt/aeronyx/aeronyx-server relay-custody collect-audit-witnesses \
  -c /etc/aeronyx/server.toml \
  --discovery-snapshot ./aeronyx-witness-snapshot.json \
  --timeout-seconds 15 \
  --max-age-seconds 7200 \
  --json
```

<!-- [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] -->
This is a deliberate one-shot network operation. It does not read environment
proxy settings, follow redirects, gossip the snapshot, contact unconfigured
peers, or install a retry/background scheduler. Each selected descriptor must
have a valid Ed25519 signature, be fresh, advertise `EncryptedStorage`, and
carry a public-safe witness endpoint. The request contains only producer
identity, current checkpoint generation, coarse archived record/byte totals,
the opaque anchor digest, a random request id, timestamp, and signatures. It
never contains messages, archive contents, paths, users, routes, payloads,
memory, DNS, destinations, client IPs, or social-graph data.

Every valid response is request-bound and witness-signed. A receipt contributes
to the round only after durable producer-side insertion. The command then
re-audits the complete vault under the still-held checkpoint maintenance lock.
It exits zero only when the configured current-checkpoint threshold is ready;
transport shortfall and all authentic `stale`, `conflict`, or `gap` evidence
produce aggregate output followed by non-zero exit. Valid adverse receipts stay
stored for investigation and cannot be outvoted by accepted receipts.

<!-- [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] -->
The command contacts each distinct configured witness concurrently, with the
existing protocol limit of 16 pins as the absolute concurrency ceiling. One
unavailable witness therefore consumes at most one configured timeout window
instead of multiplying that window by the number of pins. Duplicate pins and
the producer's own identity are removed before any request starts. Concurrent
completion does not relax durability: each verified receipt must be persisted
before it can increase the aggregate verified or accepted counts, and any
storage failure still fails the whole command closed.

The JSON contract reports only aggregate snapshot, round, vault, and policy
counts. It does not expose witness identities or endpoints. This operator
command removes manual receipt shuttling when reviewed nodes are online, but it
does not establish consensus, finality, validator voting, fork choice, or
automatic startup transmission.

After a reviewed deployment has collected enough current-anchor receipts, an
operator may make that evidence mandatory for later starts:

```toml
[discovery]
custody_audit_witness_node_ids = [
  "<reviewed-independent-witness-ed25519-node-id-hex>",
]
custody_audit_witness_min_verified = 1
custody_audit_witness_startup_required = true
custody_audit_witness_max_age_secs = 7200
```

<!-- [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] -->
The default remains `false`. Strict startup regenerates the exact current
ChatRelay custody anchor while holding the cross-process maintenance lock,
cryptographically audits every durable receipt, and evaluates only distinct
configured witnesses whose signed receipt matches that exact anchor. It runs
before PeerStore bootstrap, listeners, self-advertisement, gossip, or runtime
tasks and performs no network request. Permissionless peers therefore cannot
become startup authority.

Startup fails closed when the current anchor cannot be produced, the vault is
malformed, no fresh receipt exists, the configured threshold is unmet, or an
authentic `stale`, `conflict`, or `gap` decision exists for the current anchor.
Rolling the host back to an older custody checkpoint also changes the exact
anchor policy and leaves newer witness receipts unable to authorize it.

Receipt age is one-sided. The configured window accepts delayed past evidence;
it does not permit a future timestamp to extend readiness. At most 60 seconds
of positive clock skew is tolerated for both live and operator-imported
receipts. Keep node clocks synchronized and collect a new receipt after the
current immutable custody checkpoint advances. Always validate this rollout
with `audit-witness-vault --require-ready` before changing the flag to `true`.

<!-- [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] -->
The audit command, import result, collection result, and strict startup gate now
share one typed readiness contract. Vault totals and policy readiness are read
from the same SQLite snapshot; malformed counters, an impossible effective pin
set, or a mismatch between the aggregate `quorum_satisfied` flag and its signed
evidence fail closed. Existing JSON field names and `ready` / `collecting` /
`adverse` labels remain compatible. An internal inconsistency is reported as
`invalid` and can never satisfy `--require-ready` or process startup.

<!-- [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] -->
Read-only startup and operator audits copy at most the configured receipt-vault
capacity from one SQLite transaction. Fixed-size index BLOBs and signed-frame
lengths are preflighted before any row copy, so a replaced database cannot turn
the detached snapshot into an unbounded allocation. The transaction then
commits, releases the connection mutex, and only afterward decodes,
canonicalizes, hashes, and verifies every signed frame. Receipt insertion still
performs its before/after vault audits inside the `Immediate` write transaction;
that stronger lock is required so a malformed pre-state or post-state can never
be committed.

After the startup-only policy has been validated in production, operators may
also require the same exact-anchor evidence throughout the process lifetime:

```toml
[discovery]
custody_audit_witness_startup_required = true
custody_audit_witness_runtime_required = true
# Keep false during the initial rollout. Enable only after every exact pin is
# independently operated, reachable, and the explicit collection drill passes.
custody_audit_witness_auto_renewal_enabled = false
```

<!-- [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] -->
The runtime flag is independently default-off and is invalid unless the strict
startup gate is also enabled. The node reuses the same atomic local-vault audit
and typed readiness decision every 30 to 300 seconds; the cadence is one quarter
of `custody_audit_witness_max_age_secs`, clamped to those bounds. Missed timer
ticks are skipped rather than replayed in a burst.

By default the runtime guard never discovers authority, contacts a witness,
exports an anchor, or automatically collects evidence. If the immutable custody
checkpoint advances, signed evidence expires, the configured threshold is no
longer met, adverse evidence applies, or vault/policy integrity fails, the guard
sends one privacy-safe reason bucket to the existing required-task supervisor.
The main runtime then performs its normal bounded graceful shutdown and exits
non-zero so the service manager can recover only after current evidence exists.

Enable this flag only when the witness collection/import workflow is part of the
node's maintenance procedure. Before a planned checkpoint rotation or restart,
collect and durably import fresh exact-anchor receipts, run
`audit-witness-vault --require-ready`, and then start the service. Repeated
service-manager restarts cannot manufacture readiness and must not replace that
operator workflow.

After the explicit workflow has been exercised against every independent pin,
the node may renew expiring evidence without an operator timer:

```toml
[discovery]
custody_audit_witness_startup_required = true
custody_audit_witness_runtime_required = true
custody_audit_witness_auto_renewal_enabled = true
```

<!-- [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] -->
Automatic renewal is invalid unless both strict local gates are enabled. It
runs only after authenticated PeerStore bootstrap and only when the existing
threshold enters the bounded renewal window. One attempt contacts at most the
three exact configured pins concurrently, with the process-lifetime no-proxy,
no-redirect, bounded control HTTP client. Permissionless peers cannot become
witnesses and a healthy quorum produces no witness traffic.

The cross-process maintenance guard remains held from current-anchor creation
through transport, durable receipt persistence, and final atomic vault audit.
Receipts count only after storage succeeds. A temporary transport shortfall is
reported in aggregate and retried on the next skipped-tick cadence while the
old quorum remains valid; it never extends validity. Any authentic stale,
conflict, or generation-gap receipt is persisted and immediately follows the
same supervised fail-closed shutdown path as the local runtime audit.

Runtime logs contain checkpoint generation and aggregate round/policy counters
only. They never include witness identities, endpoint strings, signatures,
anchor hashes, messages, users, routes, payloads, memory, destinations, DNS,
IP addresses, or social-graph metadata. This remains independent evidence for
an opaque custody checkpoint, not voting, consensus, fork choice, or finality.

<!-- [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] -->
Every successful atomic audit also derives `quorum_valid_through` from the
threshold-th newest accepted receipt, rather than from the newest vault row or
the oldest surplus receipt. Operator JSON reports expose that inclusive Unix
timestamp, `quorum_valid_for_seconds`, a bounded 60-to-900-second renewal
window, and `renewal_recommended`. The runtime emits the fixed local warning
reason `receipt_renewal_required` inside that window. With automatic renewal
disabled this is advance notice only. Enabling renewal may contact exact pins,
but never changes trust policy or postpones the fail-closed boundary.

<!-- [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] -->
The runtime emits `receipt_renewal_required` once per aggregate quorum expiry
horizon. Later timer checks for that same horizon are debug-only, preventing a
long warning window from flooding the journal. After an operator explicitly
imports, explicitly collects, or automatically renews fresher signed receipts
and the quorum leaves the warning window, the runtime emits
`receipt_renewal_recovered` once. A refreshed quorum
that is still near expiry opens one new warning for its new horizon. None of
these log-state transitions changes policy or delays shutdown when the strict
audit actually fails.

Re-audit the current checkpoint after restart or before a maintenance window:

```bash
sudo /opt/aeronyx/aeronyx-server relay-custody audit-witness-vault \
  -c /etc/aeronyx/server.toml \
  --max-age-seconds 7200 \
  --json
```

<!-- [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] -->
The command holds the cross-process custody maintenance lock, regenerates the
current immutable checkpoint, verifies every retained canonical receipt, and
reconstructs the configured threshold from distinct current witness pins. It
does not contact any witness, transmit an anchor, start a scheduler, or modify
receipt rows. Opening an older local MemChain database may still perform its
normal backward-compatible schema migration before the audit.

The stable states are `ready`, `collecting`, and `adverse`. By default the
command reports policy state and exits successfully when storage is intact.
Add `--require-ready` for a systemd `ExecStartPre`, deployment health gate, or
operator script that must return non-zero unless the current checkpoint has
enough fresh accepted receipts and no adverse evidence:

```bash
sudo /opt/aeronyx/aeronyx-server relay-custody audit-witness-vault \
  -c /etc/aeronyx/server.toml \
  --max-age-seconds 7200 \
  --require-ready \
  --json
```

Output is aggregate-only: current generation, freshness window, vault totals,
accepted/adverse/missing counts, threshold, and readiness. It excludes node
identities, hashes, signatures, paths, endpoints, messages, users, routes,
payloads, memory, destinations, DNS, IP addresses, and social-graph metadata.
This command is an operator health primitive, not an enabled-by-default startup
gate and not evidence of consensus or global finality.

Verify whether the newest recovery image is usable before planning a restore:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody restore-readiness -c /etc/aeronyx/server.toml --json
```

This preflight fully verifies every managed recovery image and reports only
aggregate counts, bytes, active-main-file presence, and whether SQLite
`-journal`/`-wal`/`-shm` sidecars are present. `ready=true` means a verified
latest image exists and no active sidecar blocks a future stopped-node restore.
Stable blockers are `no_verified_backup` and
`active_sqlite_sidecars_present`. The command never opens or replaces active
custody, never deletes an artifact, and does not claim that restoration ran.
An execution-capable restore remains intentionally unavailable until its
rollback and explicit operator-approval contract are separately reviewed.

After readiness succeeds, create a short-lived state-bound plan:

```bash
umask 077
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody restore-plan -c /etc/aeronyx/server.toml --json \
  > /root/relay-restore-plan.json

sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody verify-restore-plan -c /etc/aeronyx/server.toml \
  --plan-file /root/relay-restore-plan.json --json
```

The command loads the node identity key locally and emits a ten-minute HMAC
commitment. It binds the selected verified image, configured database boundary,
active-file identity, aggregate sizes/counts, issue/expiry times, and a random
nonce. Paths, filenames, message identifiers, wallet identities, ciphertext,
and routing metadata are never emitted. Any backup rotation, database/config
change, tampering, wrong node key, or expiry invalidates the plan.
Verification accepts only a bounded regular JSON file; on Unix it must be
owner-private and the final path component must not be a symlink. Unknown JSON
fields are rejected so a credential cannot smuggle uncommitted state.

Treat the JSON as a host-local maintenance credential and do not send it to the
CMS, nodeboard, logs, or public APIs. A valid plan is only stale-state evidence:
it does not prove the process is stopped and does not authorize or execute a
restore. Future restoration must still require an explicit stopped-node gate,
an exact confirmation phrase, a rollback image of the current boundary, atomic
replacement, and post-start custody/health verification. CLI verification
releases the shared maintenance lock before returning; an execution path must
therefore re-verify the plan and replace storage inside one uninterrupted lock
scope to prevent a time-of-check/time-of-use race.

Preview the exact policy candidates. This is the default and deletes nothing:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody prune -c /etc/aeronyx/server.toml
```

Execution requires a maintenance window and all three explicit gates. Stop the
node first so an older binary that predates the cross-process lock cannot be
publishing a backup concurrently:

```bash
sudo systemctl stop aeronyx-server
sudo /root/open/AeroNyx/target/release/aeronyx-server \
  relay-custody prune -c /etc/aeronyx/server.toml \
  --execute \
  --confirm-node-stopped \
  --confirm-prune PRUNE-VERIFIED-RELAY-BACKUPS
sudo systemctl start aeronyx-server
```

The command deletes only fully re-verified policy-excess recovery images and
interrupted private SQLite files older than
`custody_backup_partial_grace_secs` (minimum 86,400 seconds). It rechecks file
identity immediately before deletion, syncs the private directory afterward,
and records only aggregate counts/bytes in a node-secret HMAC-chained local
audit. Paths, filenames, operation IDs, identities, routes, and encrypted
payload data are never written to that audit.

### No-exit OnionMiddle readiness

`OnionMiddle` is the no-exit middle-hop capability used for future two-hop
encrypted relay paths. It is intentionally opt-in. Enabling it does not make the
node a public exit and does not grant the node access to message plaintext,
payloads, DNS contents, destinations, wallet-level traffic, voucher secrets, or
private keys.

Use the guarded entrypoint helper rather than editing the TOML by hand:

```bash
./deploy/node/aeronyx-node.sh onion-middle --enable-onion-middle --dry-run
sudo ./deploy/node/aeronyx-node.sh onion-middle --enable-onion-middle --restart
```

The helper creates a timestamped `/etc/aeronyx/server.toml` backup, updates only
`[discovery].advertise_onion_middle`, validates the config, restores the backup
if validation fails, and refuses a restart while active sessions are present
unless the operator explicitly passes `--yes` during a maintenance window.

To remove the no-exit middle-hop advertisement:

```bash
sudo ./deploy/node/aeronyx-node.sh onion-middle --disable-onion-middle --restart
```

Manual changes are still possible, but they should follow the same validation
and maintenance-window pattern:

```bash
sudo /root/open/AeroNyx/target/release/aeronyx-server validate -c /etc/aeronyx/server.toml
sudo systemctl restart aeronyx-server
./deploy/node/aeronyx-node.sh status
```

`aeronyx-node.sh status` prints a privacy-safe discovery readiness summary:

- `chat_relay_capability_status`: whether local config, runtime service, public
  peer API, and descriptor advertisement agree.
- `chat_relay_blockers`: stable reason buckets such as
  `chat_relay_disabled`, `public_peer_api_not_ready`, or
  `chat_relay_runtime_not_ready`.
- `peer_quorum_status`: whether the node has enough fresh peer view state for
  the next relay/multi-hop protocol layer.
- `peer_quorum_next_action`: the next safe operator action, for example
  enabling at least one verified peer that advertises public ChatRelay.

Peer quorum is local peer-view readiness, not public-chain consensus. A node can
be healthy and still report `peer_view_ready` instead of `route_ready` when no
verified peer advertises ChatRelay yet. This is expected and safer than
pretending a relay path exists.

### Relay probe evidence boundary

`aeronyx-node.sh relay-probe` is a privacy-safe live transport check. It sends
one synthetic opaque BlindRelay envelope from the local node to a discovered
ChatRelay peer and verifies aggregate counter deltas:

```bash
./deploy/node/aeronyx-node.sh relay-probe --json
```

The command proves single-hop BlindRelay transport only:

- local node receives and forwards one synthetic opaque blob;
- remote ChatRelay peer receives it as terminal relay work;
- output contains no user message, receiver identity, DNS content,
  destination, packet payload, wallet-level traffic, private key, or full node
  identifier.

The command also reports `two_hop_readiness`, including protocol foundation
stage, routeable `OnionMiddle` count, routeable `ChatRelay` count, and planned
two-hop prefixes. That readiness means the peer store can plan a two-hop
privacy path. It is not yet a full two-hop transport proof because the current
`BlindRelayEnvelope` carries one visible `next_hop` per hop.

The Rust peer handler now accepts an optional `onward_envelope` for controlled
no-exit middle-hop experiments. When a node receives an outer frame addressed
to itself, it may forward the already-opaque onward frame to the next verified
ChatRelay peer. The middle hop still must not parse encrypted blobs and still
learns only node-level routing metadata: previous node, next node, TTL, route
bucket, and aggregate counters.

Operators can preflight the live two-hop path with:

```bash
./deploy/node/aeronyx-node.sh relay-probe --two-hop --json
```

This command is gated. It attempts a live outer+onward proof only when it can
select three distinct routeable nodes: the local entry node, one `OnionMiddle`,
and one different terminal `ChatRelay`. If the fleet has only two routeable
nodes, it returns `status=blocked` with `reason=needs_three_distinct_routeable_nodes`
instead of pretending that a return path is a valid two-hop proof.

Production operators should not claim a live full two-hop proof until there
are at least three distinct routeable nodes. With only two nodes, a synthetic
path would need to return to the previous hop (`A -> B -> A`), and the loop
guard correctly rejects that shape. A complete production probe should use a
path-aware encrypted route envelope where each hop learns only the next routing
step, never plaintext or the user social graph.

## Discovery Bootstrap And Drift Control

Discovery bootstrap snapshots contain signed node descriptors. Operators must
not hand-edit descriptor endpoints inside `bootstrap-peers.json`; changing the
JSON by hand invalidates the signature relationship that Rust verifies. Use the
repository-local entrypoint to fetch a fresh signed snapshot from a live
discovery node:

```bash
sudo ./deploy/node/aeronyx-node.sh refresh-bootstrap \
  --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422
```

Preview without writing the target file:

```bash
./deploy/node/aeronyx-node.sh refresh-bootstrap \
  --dry-run \
  --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422 \
  --json
```

The command reads `[discovery].bootstrap_snapshot_path` from `server.toml`
unless `--bootstrap-path` is provided. It backs up the existing snapshot before
writing the replacement. The output contains only signed discovery endpoints,
snapshot hash, peer count, backup path, and status. It must not include
registration codes, API secrets, private keys, user messages, DNS contents,
destinations, packet payloads, client public IPs, wallet-level traffic, or
social graph metadata.

Use `fleet-drift-check` as the read-only preflight before upgrades, restarts,
or new region rollout:

```bash
./deploy/node/aeronyx-node.sh fleet-drift-check \
  --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422 \
  --json
```

For exact release audits, add the currently expected binary hash and bootstrap
snapshot hash:

```bash
./deploy/node/aeronyx-node.sh fleet-drift-check \
  --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422 \
  --expected-binary-sha256 6d4c382907011d8da0adb7038fdb62d2bc5af859aff2ddd6d43d785462af6184 \
  --json
```

Bootstrap snapshot hashes can legitimately change as descriptors rotate, so
`--expected-bootstrap-sha256` is best for a just-distributed maintenance window,
while endpoint-set checks are better for normal daily drift monitoring.

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

The active-session decision is made after release compilation, not only before
it. A build can take several minutes and traffic may arrive while Cargo is
running. The workflow therefore checks again before installing systemd units,
again before binary promotion, and immediately before restart. When a session
appears before promotion, prepared units are restored and the candidate binary
is not installed. If a session appears in the final promotion-to-restart
window, the previous binary and units are restored atomically without stopping
the process that is still serving traffic. This prevents a rejected upgrade
from leaving a mixed old-process/new-disk state.

When the health endpoint cannot provide an active-session count for a running
service, the gate fails closed unless the operator explicitly selected
`--force`. An unavailable counter is not treated as proof that zero users are
connected.

### Commit-pinned isolated upgrade

Production nodes with local diagnostics, unfinished development, or other
tracked changes should not reset or clean that runtime checkout merely to
deploy a reviewed release. Pin the complete Git commit instead:

```bash
sudo ./deploy/node/aeronyx-node.sh upgrade \
  --repo-dir /root/open/AeroNyx \
  --branch main \
  --commit c400afec6cd6337da3f62ef56f28f55f723f07ac
```

The commit must be the full 40-hex object ID and must be reachable from the
selected `origin/main`. The workflow:

1. Leaves the runtime repository, its index, staged files, untracked files, and
   current branch untouched.
2. Clones `origin/main` into a process-scoped checkout under
   `/var/lib/aeronyx/source-checkouts`.
3. Verifies ancestry and checks out the exact commit in detached mode.
4. Reads `rust-toolchain.toml`, `Cargo.lock`, the systemd template, and the
   healthcheck from that isolated source.
5. Builds with the exact Rust toolchain and `cargo build --locked` into the
   service-scoped Cargo target.
6. Records the embedded Git commit and candidate SHA-256, validates config,
   backs up the running image, and uses the existing atomic promotion,
   active-session gate, health polling, and rollback flow.
7. Removes only the process-scoped source checkout when the command exits.

Preview the complete operation without creating a checkout or changing the
host:

```bash
sudo ./deploy/node/aeronyx-node.sh upgrade \
  --repo-dir /root/open/AeroNyx \
  --branch main \
  --commit c400afec6cd6337da3f62ef56f28f55f723f07ac \
  --dry-run
```

`--commit` cannot be mixed with `--skip-pull`, `--allow-dirty`, or unit-only
maintenance modes. Those options describe worktree-based upgrades, while
commit-pinned mode deliberately makes the runtime worktree irrelevant.

The rollback backup is taken from the executable currently mapped by the
configured systemd service whenever that process exists. This remains true
when the selected repository is a clean worktree with no local
`target/release/aeronyx-server` yet. If neither a running process nor an
existing repository binary is present, the backup step is treated as a
first-install no-op and the candidate build continues.

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
repo path, branch, source mode, requested/build commit, candidate binary
SHA-256, build resource policy, service name, config path, `--no-restart`,
`--force`, and `updated_at`.
It intentionally excludes registration codes, private keys, client public IPs,
DNS contents, destinations, packet payloads, chat plaintext, voucher secrets,
and wallet-level traffic. `aeronyx-node.sh status` displays a short summary of
this file, and `healthcheck.sh --json-only` exposes it as top-level
`upgrade_status` for nodeboard or AI maintenance automation. Healthcheck also
reports `runtime.binary_git_commit` from the running process separately from
the runtime repository HEAD, so a deliberately dirty checkout cannot make
binary provenance ambiguous. Nodes built before embedded provenance support may
report `runtime.binary_git_commit` as `unknown`; commit-pinned status fields may
be `null` until the first upgrade using this workflow. These compatibility
values are not health failures.

`aeronyx-node.sh status` also runs the read-only healthcheck JSON path and
prints the privacy-safe `operator_action` summary:

```text
operator_status=warning priority=review_warnings source=deploy/node/healthcheck.sh checks
operator_title=Healthcheck has warnings
operator_detail=...
operator_next_step=Review warning checks and capacity risks before accepting more commercial traffic.
```

This is the recommended first command for human operators and AI maintenance
assistants because it combines service state, local endpoints, upgrade state,
and the next action in one place without exposing client public IPs,
destinations, DNS contents, packet payloads, chat plaintext, registration
codes, private keys, voucher secrets, or wallet-level traffic.

Build, validate, and atomically stage the next binary without restarting the
current process:

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

Rust `/api/vpn/health` also reports capped recent operational events from local
systemd service warnings. These events are sanitized and classified as `info`,
`warning`, or `critical` before nodeboard sees them. The classifier is for
operator prioritization only: fatal/error/failed/timeout/alert-style messages
become `critical`, notice/info-style messages become `info`, and remaining
warning-level service summaries stay `warning`. The payload must remain an
operations summary and must not include client public IPs, destinations, DNS
contents, packet payloads, domains, URLs, browsing history, voucher secrets,
chat plaintext, private keys, registration secrets, or wallet-level traffic.

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
- signed bootstrap snapshot: `/etc/aeronyx/bootstrap-peers.json`
- discovery recovery: three independent public seed endpoints
- MemChain: `off`
- ChatRelay: disabled by default; explicit opt-in through
  `[memchain.chat_relay].enabled = true`
- OnionMiddle: disabled by default; explicit no-exit opt-in through
  `[discovery].advertise_onion_middle = true`

The bootstrap snapshot contains signed public node descriptors only. The
runtime verifies each descriptor before use; operators must refresh it with
`aeronyx-node.sh refresh-bootstrap` instead of hand-editing endpoints inside
the JSON. Live `seed_endpoints` provide recovery when the local cache or
snapshot is absent, while the signed peer store remains the source of routing
identity and capability truth.

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
