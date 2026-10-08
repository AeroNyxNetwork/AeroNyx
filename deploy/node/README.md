# AeroNyx Production Node Deployment

<!-- [PHALA-ISOLATED-DEPLOYMENT 2026-10-06 by Codex]
The local Phala trial profile is deliberately isolated from production CMS and
public listeners. It is a container entrypoint, not proof of TEE application
identity, VPN reachability, or reverse-onion end-to-end delivery.
-->

<!--
============================================
File Creation/Modification Notes
============================================
Creation Reason:
- Provide operator-facing documentation for the production Rust privacy node
  deployment scripts.

Modification Reason:
- [PERMISSIONLESS-NODE-JOIN 2026-09-24 by Codex] Document one-command
  operator-selected seed install and fail-closed aggregate join evidence,
  separate from central registration and relay route authority.
- [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Document bounded,
  CPU-aware previous-hop signature verification and unsigned pre-auth failures.
- [BLIND-RELAY-MONOTONIC-ABUSE-CLOCK 2026-08-21 by Codex] Document
  wall-clock-independent previous-hop rate, decay, quarantine, and LRU policy.
- [RECOVERY-ANCHOR-LOCAL-HEALTH 2026-08-21 by Codex] Document the local
  startup/operator admission rule and deployment healthcheck projection.
- [RECOVERY-ANCHOR-HEARTBEAT 2026-08-21 by Codex] Document the signed
  management-heartbeat projection for exact-generation recovery readiness.
- [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Document the additive public
  aggregate and exact-generation runtime admission gate.
- [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Document exact
  cache/anchor generation binding during startup external witness checks.
- [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] Document startup-wide
  readiness revocation when pinned witnesses reject recovery-anchor v3.
- [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Document recovery-anchor
  v3 rollback protection for signed routeability and active quarantine state.
- [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Document signed peer-cache
  v2 recovery for active route quarantine and v1 rolling-upgrade compatibility.
- [PEER-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Document closed reason
  admission for route reputation, relay rejection, and quarantine diagnostics.
- [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Document the typed,
  compatibility-preserving allowlist for heartbeat-visible relay failures.
- [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] Document the aggregate,
  process-lifetime Chat Relay heartbeat contract for custody renewal health.
- [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex] Document bounded,
  identity-jittered retry scheduling that never delays strict local audits.
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

<!-- [PHALA-CHAT-RELAY-INDEPENDENT-CONFIG 2026-10-06 by Codex]
The Phala private-recipient profile keeps MemChain model inference off while
opting into its separate durable ChatRelay and Blind Vault roles. Enabled
ChatRelay configuration is validated even when `memchain.mode = "off"`.
-->

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
v1.75.0-node-deploy - Documented bounded blind-relay signature verification.
v1.74.0-node-deploy - Documented monotonic previous-hop abuse enforcement.
v1.73.0-node-deploy - Documented fair fixed-memory relay abuse buckets.
v1.72.0-node-deploy - Documented identity-independent blind-relay admission.
v1.71.0-node-deploy - Documented fail-closed adverse optional-witness evidence.
v1.70.0-node-deploy - Documented signed recovery-anchor heartbeat telemetry.
v1.69.0-node-deploy - Documented recovery-anchor status and runtime generation gate.
v1.68.0-node-deploy - Documented exact-generation external witness binding.
v1.67.0-node-deploy - Documented whole-host rollback fail-closed route gating.
v1.66.0-node-deploy - Documented route-state rollback protection in recovery-anchor v3.
v1.65.0-node-deploy - Documented restart-safe active route quarantine.
v1.64.0-node-deploy - Documented PeerStore reputation reason admission.
v1.63.0-node-deploy - Documented typed relay health reason privacy boundary.
v1.62.0-node-deploy - Documented custody renewal runtime heartbeat telemetry.
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

### Permissionless discovery join (no nodeboard registration)

`join` is a separate one-command path for a new external discovery node. Supply
one to eight **operator-selected, public IP** seed APIs and this node's public
discovery API base URL; do not rely on the historical example seed list in the
template. The example below contains placeholders, not live node addresses:
DNS names are rejected for this admission path to avoid rebinding.

```bash
sudo ./deploy/node/aeronyx-node.sh join \
  --branch main --commit "FULL_40_HEX_RELEASE_COMMIT" \
  --seed "https://SEED_PUBLIC_IP:8422" \
  --public-endpoint "https://YOUR_PUBLIC_IP:8422" \
  --join-timeout 180 --json
```

Repeat `--seed` for independent seeds; the first must run the Stage-A join
API. The command requires Linux/systemd, curl, and Python 3.11+ (or Python
with `tomli`). It installs through the
existing installer **without a registration code**, atomically updates only
six exact `[discovery]` values in the private config, validates that config,
and starts an inactive service. It waits for the local service's current
signed self descriptor, encodes the canonical binary wire form, then sends
one `POST /api/discovery/join` to the first selected seed. It never restarts
an active service; without `--commit`, an identical rerun while it remains
active skips installation, and the seed can return idempotent `exact_replay`
while it retains the candidate. `--check-only` validates local readiness without a
POST. Before an attempted config replacement, the command creates a private
backup beside `server.toml` for operator review.

[PERMISSIONLESS-JOIN-COMMIT-PIN 2026-09-24 by Codex] For a release-controlled
join, supply the full 40-hex `--commit` and the trusted `--branch`. The
installer fetches that origin branch once (or clones it on first install),
requires the commit to be reachable from the fetched branch, checks out the
exact commit detached, and embeds the full SHA in the release binary. The
join gate checks source HEAD and that binary marker before service start or
POST. [JOIN-RELEASE-ACCEPTANCE 2026-09-24 by Codex] The pre-start gate also
requires the v2 purpose-bound receipt marker in the binary and rechecks the
pinned source tree for tracked or untracked drift. A signed local descriptor
that omits the v2 marker cannot be submitted, even if the binary passed its
pre-start consistency check. A moving branch tip cannot silently replace the
requested commit; a changed checkout, missing marker, untrusted existing
origin, or already active service fails closed without restarting it.
`join --commit --check-only`
is rejected because it does not build. Omitting `--commit` while leaving
`AERONYX_COMMIT` unset retains the existing branch-following behavior and
must not be described as pinned. The embedded-SHA marker scan is a local
consistency check, not cryptographic binary attestation; a concurrent
same-root actor can still mutate source or binaries between checks.

The repository-local entrypoint sources `lib/operator_join.sh` from its own
directory; copy or deploy that sibling module with `aeronyx-node.sh`. Sourcing
the module defines functions only and does not install or contact a seed.

Success requires the local Rust service to publish its current public-key-
matched, unexpired, signed descriptor and the chosen seed API to return HTTPS
200 over a system-validated public-IP certificate with `accepted=true`,
`route_authority=false`, and status `candidate_admitted` or `exact_replay`.
Output is aggregate JSON only:
Stage-A acceptance, readiness hints, seed counts, and a fixed reason code.
It never prints node identities, seed URLs, routes, messages, payloads, or
keys. Signature/TTL, TLS trust failure, HTTP rejection, and malformed
responses fail closed. [PERMISSIONLESS-JOIN-HTTP-HONESTY 2026-09-24 by Codex]
A plain HTTP seed may report the same JSON without storing the candidate:
the CLI therefore returns nonzero with `reason=submission_unconfirmed` and
`signed_descriptor_accepted=false`, even for HTTP 200. Do not treat that
report as Stage-A confirmation or automatically resend it.
Transport timeout is ambiguous after bytes may have been sent, so the command
does not automatically retry or fall through to another seed.

This API grants only a **bounded, non-routeable Stage-A candidate**. It is not
proof of endpoint possession, routeability, terminal delivery, economic
admission, or any central registration requirement. The HTTP response is not
a seed-signed receipt; HTTPS authenticates the responding seed transport, but
does not cryptographically prove durable storage. Production-grade confirmation
across arbitrary seeds needs a future seed-signed receipt bound to the exact
descriptor commitment, seed identity, and time. Commercial VPN registration
remains the separate `quickstart` workflow above.

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

<!-- [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] -->
The same node-global ceiling now protects `/api/chat/peer/blind-relay` before
JSON parsing. A valid Ed25519 node identity is still required by the handler,
and verified previous hops retain their separate fairness/quarantine guard,
but rotating permissionless keys cannot multiply parser capacity. The blind
relay counter remains aggregate-only and does not identify callers or inspect
the opaque encrypted relay body.

<!-- [BLIND-RELAY-BUCKET-FAIRNESS 2026-08-21 by Codex] -->
Verified previous-hop fairness remains fixed at 4096 in-memory buckets. Expired
buckets are removed regardless of insertion order; under capacity pressure the
least-recently-used non-quarantined bucket is replaced. Active quarantine is
never erased to admit a newly generated permissionless identity. If every slot
is still quarantined, the new request receives the existing aggregate
`rate_limited` response without changing peer reputation.

<!-- [BLIND-RELAY-MONOTONIC-ABUSE-CLOCK 2026-08-21 by Codex] -->
Previous-hop request windows, failure-score decay, active quarantine, idle
retention, and LRU order now use process-local monotonic time. NTP corrections,
manual clock changes, and host wall-clock rollback therefore cannot reset a
peer's quota or keep process-local quarantine alive beyond its configured
duration. The existing Unix `quarantine_until` value remains a rounded-up
projection for API and Nodeboard compatibility; it is not derived from payload,
route, user, receiver, endpoint, or source-address data.

<!-- [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] -->
Previous-hop Ed25519 verification and exact failure-receipt commitment hashing
now run outside Tokio's asynchronous workers behind a process-wide semaphore.
The active capacity is approximately half the host's reported hardware threads,
with a minimum of one and a hard maximum of eight. Admission uses
`try_acquire_owned`:
when every verifier is busy, the request receives the existing retryable
`429 backpressure` result instead of entering a blocking-task queue.

The permit is owned by the blocking worker until verification returns, even if
the HTTP request is cancelled. Invalid signatures, invalid previous-hop keys,
and pre-verification backpressure update aggregate health only; they cannot
mutate a claimed peer's state and receive no node-signed failure receipt. Once
authentication succeeds, the established signed failure-receipt contract stays
unchanged. This adds no configuration key, API route, JSON field, node identity
bucket, payload inspection, or deployment migration.

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
reported in aggregate and moves network collection into bounded exponential
backoff. The retry is aligned to the existing 30-to-300-second audit cadence,
spread by a locally derived node-identity tick, and capped at the last timer
tick before the old quorum expires. Strict local audits continue on every tick;
backoff never extends validity or delays fail-closed shutdown. If no retry tick
exists before expiry, telemetry says so and the next failed audit stops the
node. Any authentic stale, conflict, or generation-gap receipt is persisted
and immediately follows the same supervised fail-closed shutdown path.

The post-round durable audit remains authoritative even when collection
reports a partial transport or persistence failure. If completed peers already
refreshed the configured quorum, the runtime adopts that newer state instead
of retaining a stale warning. If the quorum remains inside the warning window,
the aggregate `receipt_renewal_collection_failed` or
`receipt_renewal_quorum_not_refreshed` event includes only retry delay, failure
streak, and whether another pre-expiry attempt is possible.

Runtime logs contain checkpoint generation and aggregate round/policy counters
only. They never include witness identities, endpoint strings, signatures,
anchor hashes, messages, users, routes, payloads, memory, destinations, DNS,
IP addresses, or social-graph metadata. This remains independent evidence for
an opaque custody checkpoint, not voting, consensus, fork choice, or finality.

<!-- [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] -->
Relay failure fields in the signed management heartbeat use a closed,
privacy-safe vocabulary. Recognized direct, authenticated-onion, admission,
transport, HTTP, ACK, receipt, and durable-store buckets retain their existing
JSON strings so current nodeboard and backend consumers do not need a contract
migration. Valid HTTP buckets accept only a three-digit 100-599 status.

Any unregistered value is reported as `unknown`. A raw transport error, peer
response, URL, endpoint, request/message identifier, wallet, payload, or
ciphertext can therefore never be copied into `last_outbound_failure_reason`
or `last_inbound_failure_reason`, including through the legacy compatibility
recording methods. Operators should treat `unknown` as a code/version mismatch
or a newly introduced bucket that must be reviewed and explicitly registered,
not as permission to expose the original error text.

<!-- [PEER-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] -->
The same privacy boundary now applies before PeerStore mutates route reputation,
blind-relay rejection counters, previous-hop protection state, or quarantine
diagnostics. Existing recorder method signatures remain compatible, and known
reason strings keep their current API representation. An unknown value,
malformed HTTP status, uppercase variant, or reason with appended endpoint,
route, request, receiver, or payload detail is reduced to `unknown`.

This sanitization does not forgive a failed route. Failure counters and the
existing consecutive-failure quarantine policy still advance. It prevents only
unreviewed text from entering peer health, nodeboard status, and the bounded
process audit trail. Operators who observe `unknown` should align node versions
or register a newly reviewed coarse bucket; they must never copy the raw error
into telemetry as a workaround.

<!-- [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] -->
Active consecutive-failure quarantine is part of the signed local peer cache
starting with routeability schema v2. Restarting or upgrading a node during the
fixed quarantine window no longer revives a peer merely because an older route
success was still fresh. The startup importer verifies the cache signature and
rebinds each quarantine item to the current signed descriptor route surface
before route admission.

The persisted section contains no failure reason, endpoint, route, request or
message id, payload, ciphertext, user, wallet, destination, DNS, IP address, or
social-graph information. It contains only the peer identity, descriptor
sequence, route-surface fingerprint, and bounded quarantine timestamps. Cache
v1 remains accepted during rolling upgrades but has no quarantine section and
must never be treated as proof that a peer is currently quarantined.

Quarantine entry and verified recovery trigger the same debounced atomic cache
flush used by other security-relevant peer evidence. Operators do not need a
new command or configuration flag; the periodic write and graceful-shutdown
flush remain fallback durability paths.

<!-- [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] -->
Recovery-anchor v3 also commits to an opaque digest of the exact signed
routeability/quarantine section. During startup, a valid cache signature alone
cannot authorize route state: its generation and digest must agree with the
monotonic anchor. Older, conflicting, missing, invalid, or v1/v2-unanchored
route state is rejected, while independently verified peer descriptors remain
available and bounded startup probes rebuild readiness.

No operator migration command is required. The next successful cache write
creates v3. Existing v1/v2 anchors remain readable for their historical
delivery/proof contracts, but route state from those generations is deliberately
re-probed. `last_routeability_cache_rollback_protection` reports only a fixed
aggregate bucket such as `anchored`, `cache_ahead`, `legacy_unanchored`,
`anchor_missing`, `anchor_invalid`, `anchor_conflict`, or `rollback_detected`.

The local anchor detects replacement of the cache file alone. Detecting a
whole-host snapshot rollback that replaces both files requires the existing
operator-pinned external delivery-anchor witnesses. Their opaque witness digest
now covers v3 route state without exposing routes, endpoints, failure reasons,
payloads, messages, users, wallets, IP addresses, or social relationships.

<!-- [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] -->
The external witness startup gate now applies to the complete recovery-anchor
v3 readiness bundle. A signed rollback, conflict, or generation gap clears
restored route health/quarantine, two-hop and three-hop proof windows, and
aggregate delivery readiness before public listeners start. An invalid local
anchor does the same; required witness coverage that is unavailable also fails
closed. Verified peer descriptors remain present, so live bounded probes can
rebuild current route readiness without waiting for full discovery recovery.

The deployed configuration keys and peer wire frames intentionally retain the
`verified_delivery_witness_*` prefix for rolling compatibility. Operators must
interpret those settings as recovery-anchor witness policy on v3 nodes, not as
protection for one delivery counter. No new migration command is required.

<!-- [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] -->
Startup witness approval is valid only when the local signed cache and recovery
anchor carry the same generation. If a crash lands after a newer cache rename
but before its matching anchor rename, the node reports the witness state as
`unavailable` without sending the older anchor. With
`verified_delivery_witness_required_for_restore = true`, all restored route and
proof readiness is discarded before listeners start, while verified peer
descriptors remain available for fresh probes. This behavior is automatic and
does not require an operator repair command or configuration migration.

<!-- [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] -->
`/api/discovery/status` and `/api/discovery/summary` now include an additive
`recovery_anchor` object. It reports the local cache generation, fixed
protection buckets for routeability, two-hop proof, three-hop proof, and
aggregate delivery, plus bounded external-witness counts and a
`generation_aligned` decision. It never reports anchor digests, signatures,
file paths, witness identities/endpoints, selected routes, peer identities,
messages, clients, or payload metadata.

When witnesses are required, a previously `verified` result stops authorizing
proof continuity as soon as a newer local cache generation is persisted. The
node remains fail closed until the post-write witness round verifies that exact
generation. This closes the small runtime interval between atomic local
persistence and external witness completion without changing configuration or
peer wire frames.

<!-- [RECOVERY-ANCHOR-HEARTBEAT 2026-08-21 by Codex] -->
The signed management heartbeat now carries the same `recovery_anchor.v1`
object under its existing discovery status payload. Backend and Nodeboard can
therefore distinguish `ready`, `attention`, `blocked`, and `idle` recovery
state without reconstructing policy from raw PeerStore fields. The projection
is built by the same Rust helper as the local discovery endpoints, preventing
generation-alignment rules from drifting between operator and central views.

This is an additive heartbeat field. Existing consumers may ignore it, and no
registration, configuration, database, or wire migration is required. The
heartbeat remains bounded and includes no anchor digest/signature, witness
identity/endpoint, selected route, message, client, payload, secret, or
wallet-level traffic.

<!-- [RECOVERY-ANCHOR-LOCAL-HEALTH 2026-08-21 by Codex] -->
The same `recovery_anchor.v1` object is now present under `discovery_status` in
`/api/vpn/health` and consequently in `/api/node/operator/status`. The startup
self-check includes `peer_store_recovery_anchor`, so operators and Nodeboard
receive the same admission result as the discovery API and signed heartbeat.

When `verified_delivery_witness_required_for_restore = true`, a missing,
rejected, or older-generation witness is a critical startup health failure.
Optional witness deployments remain operational, but incomplete or adverse
local anchor state produces a warning until signed recovery state is ready.

<!-- [EXTERNAL-WITNESS-ADVERSE-GATE 2026-08-21 by Codex] -->
Optional witness **availability** is advisory; authenticated witness evidence
is not. A signed `rollback_detected`, `conflict`, or `gap` result always marks
`recovery_anchor.v1` blocked and removes the proof window from restart
continuity and multi-hop admission, even when
`verified_delivery_witness_required_for_restore = false`. An unavailable or
partial optional witness round remains non-blocking. Operators and monitoring
should use the additive `external_witness.adverse_evidence` boolean to
distinguish these cases instead of inferring policy from transport failures.

`deploy/node/healthcheck.sh --json` copies only the bounded status, generation,
fixed protection buckets, aggregate witness counts, and next action into
`discovery_readiness.recovery_anchor`. Its terminal mode applies the same strict
failure rule. It does not expose anchor digests/signatures, cache paths, witness
identities/endpoints, peer ids, routes, messages, clients, or payload data.

<!-- [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] -->
The existing signed management heartbeat now includes the additive object
`system_stats.chat_relay_status.custody_witness`. Its `status` is one of
`disabled`, `monitoring`, `healthy`, `renewal_due`, `backing_off`, `exhausted`,
or `failed_closed`. The remaining fields are node-wide process counters, Unix
timestamps, audit/freshness intervals, aggregate quorum lifetime, checkpoint
generation, retry delay, consecutive failures, and fixed reason buckets.
Strict runtime mode seeds the snapshot from its successful startup audit before
listeners start, so a freshly admitted node does not wait for the first timer
tick before reporting the already-verified quorum state.

These counters reset on process restart and are operational evidence only. The
durable signed receipt vault and a fresh atomic audit remain authoritative.
Telemetry is updated from the post-round durable audit before a renewal is
reported as successful, including the case where collection returned a partial
error after enough peer futures had already persisted valid receipts.

The heartbeat never publishes witness identities or membership, endpoints,
signatures, anchors or hashes, receipts, request IDs, messages, routes, users,
wallets, payloads, ciphertext, IP addresses, destinations, DNS, or social-graph
metadata. Reading or losing telemetry cannot influence trust, routing,
readiness, renewal, fail-closed shutdown, consensus, or finality.

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

### Optional Phala Node Attestation

<!-- [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] -->
The existing two process-wide quote slots now cover generation, JSON
serialization and delivery of the resulting large response buffer. Successful
responses retain their slot through EOF, body drop or a fixed 30-second
delivery deadline; there is no new quote endpoint or trust decision. The body
copies at most 16 KiB per chunk, so a socket-held chunk cannot retain the full
quote allocation. HTTP's own write/connection buffers are not bounded by this
application-level chunk limit. Both local and public policy clones share one
delivery owner and the existing process-wide limiter.

When the quote API is enabled, a required server task sweeps expired/unpolled
buffers once per second. Normal shutdown, task cancellation or unwind closes
that policy owner and releases its remaining buffers, without closing other
server owners or releasing their shared permits. Late generation cannot hand
off a new response after that owner stops. New admission and body polling also
perform expiry cleanup. Embedded routers without the server task still retain
bounded capacity; idle expiry is reclaimed on the next admission/poll rather
than by an independent timer. Expired/stopped bodies report a stream error,
not successful truncated JSON. This does not forcibly cancel work already
accepted by the guest agent. Wire fields, no-store headers, recipient binding
and quote appraisal remain unchanged. Regression source is authored only;
no build, test execution, live slow-reader or shutdown acceptance has run.

<!-- [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] -->

An operator may mount the Phala guest-agent Unix socket and set
`discovery.phala_attestation_socket_path` to its absolute in-container path.
Only the explicitly rendered `--public-peer` Compose mode mounts
`/var/run/dstack.sock` at `/var/run/aeronyx/phala-agent.sock`; the default
template and private-recipient service do not mount it. dstack v0 and v1 guest
APIs share that socket. `/var/run/tappd.sock` is a separate Tappd `/prpc`
service and is not compatible with the HTTP guest API used here; this profile
does not support Tappd transport.
Configuration also requires `discovery.enabled`, `advertise_self`,
`public_discovery`, and `public_api_listen_addr`. The dstack socket may be
configured only when a public HTTPS endpoint is available; Rust rejects the
socket otherwise. During Phala's first boot, leave the socket disabled until
the app origin is assigned, then rerender with the endpoint and socket together.
The signed descriptor and feature gate use `[discovery].public_endpoint`
first, then `[network].public_endpoint`, so both resolve to the same advertised
origin.
The node then advertises the signed `anpf1-pdna1` protocol feature and exposes
`GET /api/discovery/phala-attestation?nonce=<64 hex characters>`. The node
derives dstack `report_data` from a domain separator, its 32-byte node identity,
and the caller's 32-byte nonce, then returns the guest-agent's opaque
attestation with that expected report data. Decoded evidence is capped at 256
KiB; because guest API v1 hex-encodes it, the bounded guest HTTP frame allows
up to 528 KiB including protocol framing. Legacy v0 JSON evidence remains
capped at 256 KiB. The endpoint is opt-in and bounded to 12 requests per
minute and an 8-second agent timeout.
Every response carries `Cache-Control: no-store, private` and `Pragma:
no-cache`; the quote is bound to the request nonce and must be freshly fetched.

<!-- [PHALA-PEER-ATTESTED-ROUTING 2026-10-06 by Codex] -->
An operator may separately opt this node into requiring attestation for
PeerStore-selected outbound peers and direct blind-relay/source-pull/recipient-
poll paths. Peers must be publicly discoverable and advertise the signed
`PhalaNodeAttestationV1` endpoint before they can be appraised:
`phala_attested_peers_required = true` requires both
`phala_trusted_app_ids` and `phala_trusted_compose_hashes` to be explicit local
allowlists. App IDs use `0x` followed by lowercase hex of the raw bytes from
the verified RTMR3 `app-id` event; this matches ACI's dstack policy subject
representation, not a display name. The route verifier challenges the signed HTTPS endpoint with a
fresh nonce, bounds the response, validates node/nonce/report-data binding,
appraises only dstack v1 TDX evidence through DCAP QVL, and requires QVL status
`UpToDate` plus exact app-id and measured-compose matches. A route appraisal is
kept only in process memory, bound to the exact signed descriptor, and expires
within `phala_peer_attestation_max_age_secs`; restart, descriptor change, quote
failure or expiry removes route eligibility. This does not imply inference
proxy blindness or that an ordinary backend does not see plaintext.

<!-- [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] -->
The Phala peer and endpoint-free private-recipient Compose roles carry the same
optional protected outbound-route policy. Set
`AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED=true` together with both
`AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS` and
`AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES`, each a JSON array of 1..=64
distinct canonical pins, capped at 8192 UTF-8 bytes. App pins are `0x` plus
lowercase hex for 1..=64 measured bytes; compose pins are `sha256:` plus 64
lowercase hex digits. Optional
`AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS` accepts canonical decimal
1..=86400. Unset/empty values preserve the mounted TOML policy, including strict
mode; they never default it to false. A partial bundle, malformed pin or invalid
age fails rendering/configuration without partially replacing policy. An
explicit `false` must have no other policy inputs and is the only environment
override that disables strict mode. To manage policy exclusively in mounted
TOML, leave all four environment values empty.

Renderer guards check each service's complete policy environment independently,
before and after role substitution. Key counts are scoped to indented YAML
environment keys, not matching variable names inside interpolation values;
this also corrects the existing public-seed guard's false duplicate rejection.
These public measurement pins do not mount
the guest socket, add private ingress, enable a reverse role, or authorize model
inference. Config startup still installs the policy into PeerStore before cache
import, and only fresh verifier-local appraisal grants route eligibility. Do
not copy pins from unverified discovery/quote responses or infer proxy blindness
from this setting. Tests for atomic overrides, inherited strict policy and
rendered role coverage are authored only; no build, test, render execution or
deployment has been performed for this batch.

<!-- [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] -->
<!-- [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex] -->
The in-memory route appraisal also retains its post-lock publication time and
the highest later route observation (including monotonic expiry projections)
for that exact descriptor. Earlier request
samples fail eligibility rather than rolling the cache back; an older appraisal
completion cannot replace newer evidence or erase an observed expiry. These
are local observation times, not timestamps trusted from remote JSON. Both
publication and selection apply the original challenge's monotonic age to the
signed descriptor validity window as well as the configured appraisal lifetime.
An observation behind that monotonic lower bound is denied even while the raw
wall-clock value alone would still pass the descriptor's validity window.
The source's final fresh-dispatch POST and historical-evidence GET gate share
this cache check; rejecting a stale sample does not re-arm execution or delete
durable ciphertext. Fresh current appraisal may renew eligibility. Restart
starts with an empty appraisal cache, never disk-restored route trust.

These checks are descriptor-local, second-resolution boundary checks, not a
process-wide or hardware clock attestation. A delayed request sample is denied
without interpreting it as proof that the whole host clock failed. They do not
establish recipient-key TEE residency or enable plaintext inference. Synthetic
time/expiry/publication and final-source-gate tests are authored but unexecuted;
no build, test execution, live QVL request or deployment was performed for this
batch.

Legacy fixed-clock cache fixtures freeze elapsed time only under `cfg(test)`;
real verifier-result publication and all non-test route checks retain the
original `Instant`. Explicit monotonic-expiry fixtures control their own elapsed
sample rather than depending on sleeps or machine speed.

<!-- [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] -->
Gossip and DNS endpoint promotion share a 15-second appraisal deadline and two
process-wide permits. Blocking QVL/ACI verification retains its permit even
after its async caller is cancelled; cancellation does not create extra crypto
capacity. Each collateral response is streamed with a 1 MiB body limit and
32 KiB aggregate header limit. Quote framing is checked before QVL's declared
byte-vector allocations; official QVL still performs parsing and verification.
Collateral is downloaded only from the fixed HTTPS Phala PCCS origin using
public-address DNS pinning, hostname TLS, no redirects and no ambient proxy.
Certificate-selected CRL URLs are never followed. If the PCCS root-CRL resource
is unavailable, appraisal fails closed rather than falling back to another URL.
These limits may reject unusually large collateral; they do not permit a
non-TEE inference fallback.

<!-- [PHALA-PROMOTION-NETWORK-ADMISSION 2026-10-07 by Codex] -->
Permissionless DNS endpoint promotion checks historic gate capacity and the
exact current Stage-A descriptor before fetching Phala evidence, then repeats
that check after appraisal and before each endpoint probe or attestation gossip
send. Time is sampled inside the blocking admission job, after any pool wait.
A rotated, expired, missing or inconsistent candidate cannot continue using
the selected descriptor at a later preflight. A newer live descriptor or a
same-sequence live conflict also vetoes old Stage-A work; a saturated unexpired gate table
rejects a new identity before consuming the quote/QVL transport budget.
This is a bounded-work preflight, not a capacity reservation, atomic HTTP send
fence or promotion authority. A concurrent change after admission may race an
already-admitted request; final promotion, exact evidence and fresh route-probe
gates remain mandatory. Existing-identity retries retain their historic slot.
Regression source covers capacity/reclamation, exact commitment, rotation,
stronger live imports, expiry and
restart denial. It has not been compiled or executed, and does not establish
network, cancellation or hardware-attestation acceptance.

<!-- [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] -->
The final signed blind-relay control probe has a dedicated HTTPS DNS lane for
an exact promoted Phala descriptor with a current local appraisal. It does not
require the still-closed control-probe readiness bit, and therefore does not
deadlock its own bootstrap. Stage-A input, a known identity pin, a DNS URL or a
cached appraisal alone cannot enter this lane. Ordinary warmup/probes retain
their public-IP-only URL policy.
This lane uses the existing bounded public-answer DNS pinning and hostname TLS
client, with no proxy inheritance, redirects or unpinned-client fallback. After
DNS/request preparation it rechecks the exact descriptor, active promotion gate
and appraisal before POST. Receipt processing uses the actual observation time
and rechecks the DNS appraisal under the authority epoch before opening the
control-probe bit. Lock contention rejects admission. An already-admitted send
may race a later change; this is not an atomic network revocation guarantee.
This control proof is still only route readiness, not task execution, source
reply verification, proxy blindness or client-to-TEE E2EE. Regression source is
authored only; no build, tests, live DNS/TLS or Phala deployment were performed.

<!-- [PHALA-APPRAISAL-TASK-OWNERSHIP 2026-10-07 by Codex] -->
The gossip future owns its single appraisal child through an abort-on-drop
owner. Parent cancellation/startup unwind closes a shared publication stop bit
and requests child cancellation; normal gossip shutdown also joins that async
child. Finished children permit the next bounded appraisal, but a stopped
owner never reopens. Publication rechecks both owner stop and process shutdown
inside the authority/cache locks, after any lock wait, without renewing the
challenge age or bypassing descriptor/QVL checks. This veto leaves previously
published evidence unchanged. A publication admitted before the stop check may
finish; stopping does not revoke evidence retroactively.
Running blocking QVL is not preempted or falsely reported as drained: it retains
the existing shared permit until completion. An already-started blocking cache
publisher also cannot be forcibly cancelled, but must pass the post-lock stop
check before being admitted to mutate the cache. Regression source covers
parent-registry cancellation, child-slot reuse, sticky stop, normal async join
and post-lock publication veto. These cases are authored only, not executed;
no build, runtime cancellation or hardware-attestation acceptance is claimed.

<!-- [PHALA-PROMOTION-CANCEL-OWNERSHIP 2026-10-08 by Codex] -->
The permissionless-promotion supervisor now observes shutdown throughout the
round, including appraisal, observation delays and the final control probe.
Shutdown wins over simultaneously ready work. Each coordinator future owns a
sticky per-round stop state; dropping it fences retained blocking work, also
checked against the process shutdown flag. Network preflight samples that state
before and after potentially blocking capacity work. Final promotion rechecks
owner admission after acquiring the gate lock and requires this round's exact
generation; Phala cache publication uses the existing post-lock owner veto.
An already-admitted synchronous evidence/descriptor write may finish, leaving
an inactive deny gate if activation is vetoed. This is not database rollback,
forced interruption of QVL or proof of draining all blocking work. Evidence
published before the stop check is not revoked retroactively. Existing QVL
permits remain retained until their blocking verification completes. Cancellation
and post-lock gate regressions are authored only, not compiled or executed.

<!-- [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] -->
TCB Info and QE Identity use a local restrictive signed-JSON profile, not a new
Intel protocol or general ACI JSON rule. The locked `dcap-qvl 0.3.12` offline
verifier (`src/verify.rs`, `verify_tcb_info_signature` and
`verify_qe_identity_signature`) verifies the supplied strings directly. Its
downloader instead reconstructs generic JSON values; we do not use that
downloader. Intel C++ QVL at `d12717e3f1f2ab81313f88001fd902b5ef5f9c8c`
uses a RapidJSON Writer step. [Intel PCS](https://api.portal.trustedservices.intel.com/content/documentation.html)
describes signatures over the enclosed body without whitespace. These are not
interchangeable for all valid JSON spellings.

Within either signed subtree, including nested names/values, reject Unicode
escapes and escaped slash; allow raw valid UTF-8 and standard short escapes.
A literal backslash followed by `u` is not a Unicode escape. Numbers must be
canonical nonnegative integer tokens no greater than 9007199254740991, checked
from original digits, then subject to QVL's narrower field schemas. Negative,
fractional/exponent and larger numbers are unsupported. Valid but unsupported
forms return `peer_collateral_unsupported_form`; malformed/duplicate-key JSON
returns `peer_collateral_malformed`. Both stop appraisal, with no alternative
serialization, signature attempt or inference fallback. Bounds remain enforced.

Accepted payloads retain field order and all string lexemes, removing only the
four JSON whitespace bytes outside strings. The same profile and 64-byte r||s
framing guard apply to downloaded and injected/offline collateral. QVL still
checks signatures, P-256 scalar/key validity, certificate chain, revocation,
dates and quote claims; passing this lexical guard proves none of those. ACI
reports, keysets and receipts retain their existing JSON/JCS rules. This shared
path serves peer appraisal and the retained ACI verifier, not node inference.
Regression source is authored only. No build, official-vector execution or live
TDX attestation acceptance has been established.

<!-- [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] -->
Bounded Phala trust-evidence JSON inputs reject duplicate decoded member names at every
depth, including escaped aliases and members unknown to the current schema.
This guard runs before ACI report/keyset, receipt and session interpretation,
and before node evidence and PCCS signed-object parsing. It uses serde's JSON
parser and default recursion limit; it neither rewrites authenticated bytes nor
replaces QVL or ACI cryptographic verification. Distinct objects may reuse names.
ACI JCS input must have unique members under
[RFC 8785 section 3.1](https://www.rfc-editor.org/rfc/rfc8785.html#section-3.1).
Ambiguous JSON is rejected, not repaired with a first/last-member preference.
Regression source is authored only; this is uncompiled and untested code, not
proof of a working hardware-appraisal or source-E2EE closed loop.

Cache publication requires the verifier-owned result and the still-current
signed descriptor, rechecked under the route-authority lock. Age starts before
the challenge request, not at verification or quarantine completion. Both wall
time and process-local monotonic time must remain within the configured age;
an older concurrent appraisal cannot overwrite a newer fresh one. Neither the
appraisal nor its monotonic clock survives a restart. Regression source covers
malformed framing, destination restrictions, cancellation and publication age;
it has not been executed. This development batch is uncompiled and undeployed,
and establishes no live TEE, end-to-end delivery or proxy-blindness result.

<!-- [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] -->
MemChain's ACI provider uses the same bounded collateral transport and shares
the two appraisal permits with node discovery and ACI receipt-signature crypto.
Identity appraisal includes report fetch, collateral fetch and QVL/ACI work in
one 15-second budget; bounded blocking jobs keep their permits after caller
cancellation. ACI receipt signing input is limited to 1 MiB before crypto.
The offline collateral verifier also checks quote allocation framing.
Each keyset role is limited to 64 entries before the profile's cross-role
comparisons. Omitted optional TLS entries remain compatible;
oversized or malformed roles fail closed rather than consuming unbounded CPU.

Each chat/embedding attempt owns a fresh nonce and an in-memory context. Its
wall-clock floor begins before the challenge and never moves backwards. A
five-minute monotonic lifetime ceiling additionally bounds that local context;
it does not extend any keyset or signed session validity. The original
nonce/report binding and exclusive keyset expiry are checked again
before sending inference, around receipt verification, after session lookup
and before returning the result. Verification uses current time after
collateral download, not a timestamp captured before that download.
Reports, contexts and permits are not persisted or reused after restart.
No wire contract, source E2EE gate or upstream-assertion scope is relaxed:
server model calls remain disabled without client-to-TEE field encryption.
Clock-floor and bounded-signature regression source is authored but not run;
this batch has not been compiled, tested or deployed.

<!-- [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] -->
ACI chat and embedding payloads are serialized through a one-MiB bounded
writer before acquiring attestation capacity or starting provider IO. JSON
escaping and wire metadata count toward this limit; bounded raw text alone
does not establish a bounded wire body. Chat preflight also limits requests
to 128 messages, 64 bytes per role, 16 stop sequences of at most 1,024 bytes
each, and one MiB of aggregate message fields. Model names are at most 256
bytes; zero output-token limits and nonfinite/out-of-range temperatures are
rejected. The provider rechecks its effective default options, so a missing
request override does not bypass validation. Invalid operator defaults return
`llm_aci_verifier_unavailable` rather than permanently rejecting a valid task.
Invalid request input returns only the
stable `llm_aci_request_rejected` category, stops router fallback and is
rejected without inference retry by the cognitive queue. Existing generic
provider compatibility and the disabled server E2EE gate are unchanged.
These source changes and their regression source are uncompiled and untested.

The same gate is checked immediately before private source-pull dispatch and
when reading signed recovery evidence from its pinned relay. A fresh route
appraisal is therefore required for those direct paths too; the fixed relay
identity and HTTPS-origin pins remain independently required.
The private recipient's signed Pull authority snapshot also requires the
appraisal, and is reloaded after DNS pinning before a Claim or Result request.

<!-- [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] -->
The retained ACI chat and embedding request builders explicitly set
`X-Upstream-Verification: required`, replacing any previous values. This is
the ACI/1 section 6.1 policy header, not an attestation result. The existing
`provider.aci_verified` body field is retained for product routing compatibility;
it is not an ACI-defined body field or independent evidence of verification.
The pinned reference gateway passes `provider` to its control-plane consult;
the public source does not establish that control plane's `aci_verified` policy.
Ordinary-node plaintext inference remains disabled before appraisal or send,
including chat and embeddings. This request-contract correction does not
connect source-owned E2EE or enable inference, and is uncompiled and untested.

For client integration, the reference commit
`133171efb115bb0437f69a4679b5522c5e36139e` serializes the decrypted request with
`serde_json::to_vec(Value)` and `preserve_order`, not JCS. The locked serializer
is `serde_json 1.0.149` with `zmij 1.0.21`. The bytes are hashed before subsequent
tool-call normalization or provider rewriting. Whole-content decryption also
parses any valid JSON array plaintext as an array, even when the original
source value was a string such as `[1,2]`. Exact received-body reconstruction
must account for that conversion, key order, numeric rendering and escaping;
hashing the original plaintext request or trusting a server-supplied hash is
not an equivalent check. The spec's response `wire_hash` plus AAD alternative
must not be extended into a request-hash bypass. These are pinned source
contracts, not observations of the deployed Phala service. Reference locations:
[`e2ee.rs`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/src/aggregator/service/e2ee.rs#L307),
[`e2ee_crypto.rs`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/src/aggregator/service/e2ee_crypto.rs#L394),
[`handlers.rs`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/src/http/app/handlers.rs#L544),
[`completion.rs`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/src/middleware/completion.rs#L77).

<!-- [PHALA-133171-PROFILE 2026-10-08 by Codex] -->
The retained public MemChain ACI appraisal/receipt path now explicitly targets
that same exact 133171 commit's wire profile. `aci/1` alone is insufficient to
select a draft: the pinned a991 helper crates describe different report and
keyset shapes. They remain dependencies for JCS and hardened quote/event-log
mechanisms, not wire authority. The 133171 report is parsed separately: its
stable `workload_id` hashes the identity public-key object, while the keyset
digest hashes the exact served closed-schema keyset, including `keyset_epoch`.
The nonce-bound statement includes both digests. Required identity endorsement
verification signs the keyset-digest payload under that identity; its secp256k1
signature is 64-byte r||s, distinct from the receipt's 65-byte r||s||v.
`X-ACI-Identity`, the report and the signed receipt must name the same identity,
independently of their common operational-keyset digest. No alias, profile
autodetection, downgrade or historical-tag backfill is provided.

The selected dstack custody subset accepts secp256k1 identity keys and fixes
the KMS purpose to `aci.identity.v1`, as declared in that commit's provider
implementation. Exactly one identity chain must recover an explicitly accepted
root using the semantically measured app-id; root recovery alone is not trust.
Ed25519 endorsements can be checked, but do not satisfy this dstack identity
custody subset. Receipt/E2EE/TLS private-key custody still relies on the accepted
measured-code/provenance contract and identity endorsement. It is NOT an
independent per-operational-key KMS proof. Existing semantic RTMR3 replay,
compose/provenance policy, UpToDate TCB, bounded collateral/crypto and additive
TLS hostname/chain/SPKI checks are retained. Compatibility of the stricter
event-log format with an actual deployed report remains unverified.

Persisted evidence now records the exact internal profile and this limited
custody scope. Historical records lacking those tags can still deserialize,
but cannot qualify as newly verified evidence. Raw report/receipt evidence is
not rewritten or deleted. Binding, signature-encoding, KMS-purpose, identity
rotation, header and persisted-shape regression source is authored only.
No tests, dependency resolution, compilation, deployment or native-library
rebuild were run for this batch. The ordinary-node plaintext transport remains
hard-disabled before appraisal/send; source-owned E2EE, client authority,
revocation/rollback state, approved deployment trust material and full reverse
delivery verification are separate unfinished requirements. This profile
correction establishes neither a deployed Phala wire contract nor proxy
blindness nor inference activation.

<!-- [PHALA-ACI-SERDE-BYTE-VECTORS 2026-10-08 by Codex] -->
Request reconstruction must reproduce parsing as well as printing. In
`serde_json 1.0.149`'s ordinary numeric representation, an integer token stays
`u64`/`i64` while a decimal or exponent token becomes `f64`; `-0` also becomes
a negative-zero float. Consequently `1`, `1.0` and `1e0` do not all retain
the same serialized bytes. Integers outside the integer range take a float
path. The `float_roundtrip` feature selects a different decimal parser, and
`arbitrary_precision` changes numeric representation altogether. A Cargo.lock
version pin alone does not establish the effective feature closure, target,
or the deployed measured binary. Do not replace this pipeline with Dart
`jsonEncode`, JCS or a formatter-only port and call the request binding verified.

The ordinary serializer emits compact UTF-8, retains object insertion order
under `preserve_order`, and uses `zmij` for finite floats. It escapes quotes,
backslashes and ASCII controls, but does not escape solidus, HTML punctuation,
U+2028/U+2029 or other valid Unicode merely for being non-ASCII. Decoded
surrogate pairs become their UTF-8 scalar. Strict duplicate-name and malformed
UTF-8/Unicode rejection must precede client reconstruction; last-member-wins
parsing is not an intent binding. Preserve numeric token/type information until
the matching parser has consumed the frozen request; a generic decoded map can
already have lost it.

Source-only regression cases in `llm_provider.rs` compare the existing bounded
writer to literal expected bytes, not another call to the same serializer.
They cover nested key order, integer/float/exponent tokens, signed zero, integer
limits, escapes/Unicode and text-part string versus array type. These cases
neither decrypt a request nor establish full floating-point compatibility.
The existing per-part text decryption keeps a UTF-8 string such as `[1,2]` a
string; selecting that existing wire form must be explicit in the frozen client
intent, and must not silently change an unsupported model/role request.

Before source integration, bind the accepted gateway source/build and numeric
feature profile to the appraised deployment, reconstruct the post-decryption
Value locally with that exact profile, and compare its independently computed
hash to the signed `request.received` event. Keep intent type/value checking
separate from this byte comparison. Unknown profiles fail closed without
silently rounding, rewriting user intent or falling back to plaintext. These
requirements are not implemented client wiring or evidence about the hosted
service; all new regression source remains uncompiled and unexecuted.
Reference parser/serializer locations:
[`de.rs`](https://github.com/serde-rs/json/blob/v1.0.149/src/de.rs#L462),
[`ser.rs`](https://github.com/serde-rs/json/blob/v1.0.149/src/ser.rs#L1716),
[`map.rs`](https://github.com/serde-rs/json/blob/v1.0.149/src/map.rs),
[`zmij`](https://github.com/dtolnay/zmij/blob/1.0.21/src/lib.rs#L939).

<!-- [PHALA-ACI-NUMERIC-PROFILE 2026-10-08 by Codex] -->
The pinned reference's **default source-build candidate**, not the hosted
service's measured profile, can now be narrowed further. Its root manifest
requests `preserve_order`; root Axum's JSON feature requests `raw_value`, and
`dstack-sdk-types` requests `alloc`. Serde JSON's default enables `std`, and
`preserve_order` enables `indexmap` and `std`. The resulting candidate flags are
`alloc`, `default`, `indexmap`, `preserve_order`, `raw_value` and `std`, with
`float_roundtrip`, `arbitrary_precision` and `unbounded_depth` absent. This is a
static manifest inference for the pinned default build, not Cargo execution or
attestation of an effective target/build environment.

The lockfile's 18 registry packages that directly depend on `serde_json` were
inspected using their published Cargo manifests, after each crate archive's
SHA-256 matched its lockfile checksum. None declares `float_roundtrip`.
`rust_decimal 1.42.1` exposes the only `arbitrary_precision` activation found:
`serde-arbitrary-precision` -> `serde-with-arbitrary-precision` ->
`serde_json/arbitrary_precision`. Its defaults are only `serde` and `std`.
The root gateway is its only locked parent and requests no extra features,
so that optional path is not selected by the candidate default build. The
reference smoke Dockerfile builds the locked release binary without extra
feature arguments; its image tag and source-label arguments alone still do
not prove the binary running behind the hosted endpoint.

An additional source-only regression distinguishes the ordinary parser using
the upstream roundtrip counterexample `51.248178375505404`: the default parser
produces the float printed as `51.24817837550541`. It checks numeric bits as
well as literal output bytes, and rejects an exponent outside finite f64.
These sentinels must detect feature drift rather than silently adopting new
numeric behavior. They do not authorize rewriting source intent or skipping
independent request-hash comparison. The tests remain uncompiled/unexecuted;
future verification must confirm the actual unified features and target,
execute cross-language numeric/escape/type vectors, and bind the accepted
profile to appraised deployment provenance before client use.
Reference evidence:
[`gateway manifest`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/Cargo.toml),
[`gateway lockfile`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/Cargo.lock),
[`smoke Dockerfile`](https://github.com/Phala-Network/private-ai-gateway-with-vllm-router-as-middleware/blob/133171efb115bb0437f69a4679b5522c5e36139e/Dockerfile.smoke),
[`rust_decimal crate archive`](https://static.crates.io/crates/rust_decimal/rust_decimal-1.42.1.crate),
[`serde_json features`](https://github.com/serde-rs/json/blob/v1.0.149/Cargo.toml#L46),
[`upstream roundtrip counterexample`](https://github.com/serde-rs/json/blob/v1.0.149/tests/test.rs#L907).

<!-- [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] -->
Source and recipient roles install their configured R/P identity pins before
cache or gossip import, including recovery-only startup. The bounded gossip
appraisal scheduler gives stale, known, locally pinned R descriptors a separate
round-robin lane, alternating with ordinary peers when both need appraisal and
the local role permits those outbound targets.
Neither an unavailable fixed relay nor a large public set starves the other
lane; the shared one-at-a-time bound is unchanged. Such an R may disable public discovery;
the private recipient P still has no endpoint and is never an appraisal target.
A private-field PeerStore target carries this narrow verifier scope: general
peer/promotion verification still requires public discovery. Pins alone neither
create descriptors nor bypass signed descriptor, HTTPS/public-address, app/compose
allowlist, nonce/report, QVL, exact cache-publication or freshness checks. The
same one-at-a-time scheduler and shared appraisal permits/budget remain in use.
No P grant is required just to refresh R's appraisal for historical recovery;
fresh dispatch and Claim gates still independently require their signed authority
and reject recovery-only mode. Unknown/Stage-A identities, source-only pins,
expired descriptors are excluded. Existing self-descriptor behavior is retained.
This does not prove signing-key TEE
residency or gateway blindness. Regression source is authored, not executed;
this connected startup/selection/verifier change is uncompiled and undeployed.

<!-- [PHALA-APPRAISAL-EGRESS-PIN 2026-10-07 by Codex] -->
Appraisal does not widen a private recipient's outbound discovery scope. Its
scheduler can select only its configured R identity at the configured canonical
HTTPS origin, in both live and recovery-only mode. A fresh, missing, expired or
rotated R never triggers public-peer fallback. A source retains ordinary public
peer appraisal, but its fixed R is subject to the same origin pin. Other local
roles retain the existing bounded public/pinned-lane selection. Invalid enabled
role pins disable appraisal rather than selecting an unrestricted default.
Selection checks the configured origin, and the immutable PeerStore target
retains it for another check at the verifier entry before DNS/evidence I/O.
Equivalent default HTTPS ports are accepted; a new host, port or scheme needs
operator configuration, not just a valid descriptor signature. Current signed
descriptor, public-address DNS pinning, hostname TLS, challenge/QVL and exact
cache-publication checks remain independent and mandatory. This constraint
adds no listener, route authority, stored secret, or model-inference path.
Regression source covers role/recovery selection, fresh-R no-fallback, signed
endpoint rotation, same-origin renewal and restart. It has not been compiled or
executed; this batch is not runtime or Phala end-to-end acceptance.

<!-- [PHALA-ATTESTED-DNS-ENDPOINT 2026-10-06 by Codex] -->
Permissionless discovery accepts an HTTPS DNS endpoint only when its signed
descriptor advertises `PhalaNodeAttestationV1`. It remains a bounded Stage-A
candidate until permissionless endpoint promotion, strict Phala peer mode, and
both local allowlists are enabled;
promotion then requires an appraised dstack quote and the existing repeated
endpoint-possession quorum. Observers on older releases remain fail-closed and
cannot contribute DNS endpoint observations. DNS answers are validated as a complete set of
public-unicast addresses and pinned per request while TLS verifies the signed
hostname. The hostname commitment is domain-separated from the unchanged IP
socket commitment. This binds the endpoint to the node signing key and checks
a separately appraised TEE quote; it does not prove that the signing key is
inside the TEE or that a gateway is not proxying the quote.

This is evidence transport, not local quote verification: callers must verify
the attestation chain, report-data binding, app identity/compose, and TCB using
independently trusted policy. The endpoint uses dstack guest API v1 by default.
A dstack 0.5.x agent that lacks the `/v1` method can be enabled for legacy
compatibility by explicitly setting
`discovery.phala_attestation_allow_legacy_v0 = true`; fallback occurs only
after a non-JSON missing-mount 404, not after timeouts or request errors. The
Phala peer example keeps this option false and fails closed if v1 is
unavailable. Select the matching host socket when rendering Compose; the
in-container API path remains stable.
The response identifies `dstack_guest_v1_msgpack` versus
`dstack_guest_v0_get_quote_json`; legacy evidence is the original GetQuote JSON
and remains caller-verified. A configured socket or signed feature token alone
does not prove the node is running in an accepted Phala CVM. The node container
should receive only the selected guest-agent sockets and persistent node state;
do not mount host secrets or enable privileged/host networking to expose the
attestation endpoint.

<!-- [PHALA-ATTESTATION-RESPONSE-CONTRACT 2026-10-06 by Codex] -->
The stable JSON transport type is `PhalaNodeAttestationResponseV1` in
`aeronyx_core::protocol`. Node-only responses use
`contract_version = "phala_node_attestation.v1"`; recipient-bound responses
use `"phala_private_recipient_attestation.v1"` and add
`recipient_node_id` plus `authorization_sha256`. Both include `node_id`, the
request `nonce`, 64-byte hex `expected_report_data`, `attestation_format`, hex
`attestation`, and the fixed `verification` note. Callers can use
`validate_for(node_id, nonce, None)` for node-only binding or
`validate_for(node_id, nonce, Some((recipient_id, grant_sha256)))` for the
recipient form. The validator checks request fields, contract variant,
recipient/grant digest, recomputed padded report-data, known evidence-format
label, evidence encoding/size, and the fixed note. It does **not** parse or
appraise the quote, verify DCAP collateral/signatures, prove compose/app
identity, enforce TCB policy, or show that recipient key material is inside
the TEE. Treat successful binding validation as a transport consistency check
only; full attestation verification remains a separate relying-party duty.

<!-- [PHALA-DSTACK-V1-ACI-EVIDENCE 2026-10-06 by Codex] -->
The guest endpoint preserves the original v1 named-MessagePack bytes in its
hex transport field. Both serving-side contract validation and peer appraisal
decode the full envelope, reject trailing bytes, require the `tdx` platform,
and check the 64-byte stack report data. The peer verifier then explicitly
maps the quote bytes, TDX event log and stack config into the JSON evidence
shape expected by `aci-verify`; raw MessagePack is never passed as a JSON value.
GCP-TDX is rejected until its accompanying TPM quote is independently
validated. The peer verifier still separately appraises DCAP collateral, TCB,
RTMR3 event replay, measured compose hash, and app ID against local pins. This
does not attest the node's own identity to a remote client or imply inference
proxy blindness.

<!-- [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] -->
When the relay queue has exactly one configured recipient, the endpoint also
accepts `recipient_node_id=<64-hex>` with the nonce and returns
`phala_private_recipient_attestation.v1`. Its report data binds the public
relay ID, recipient ID, SHA-256 of the grant's canonical bytes, and nonce
under a separate domain; the response returns that grant digest for an
independent comparison with authenticated discovery data. The relay emits
this quote only while PeerStore has the current signed R/P descriptors and
P-signed Pull grant, and rechecks the exact grant digest after quote generation.
The signed discovery descriptor advertises
`PhalaPrivateRecipientAttestationV1` only with a dstack socket and one fixed
queue recipient while fresh queue admission is enabled; recovery-only mode
does not advertise this recipient-bound feature. In recovery-only mode the
node-level quote endpoint remains available, but a request naming the pinned
recipient receives the generic `attestation_unavailable` response. A node-level
quote is not evidence of current Pull authority. This remains opaque evidence:
callers must appraise the quote and approved compose measurement independently.

The peer-attestation profile signs its public HTTPS endpoint into the node
descriptor. Do not enable `reverse_onion.recipient` on that same identity:
private Pull route authorization rejects recipient descriptors with a public
endpoint, so startup now fails configuration validation instead of leaving an
apparently enabled worker unable to poll. The Phala peer identity can serve as
the public peer/attestation endpoint, or a private recipient can keep its
descriptor endpoint-free; using both roles on one identity requires an explicit
privacy/protocol decision. No automatic identity provisioning or authority
transfer is implemented.

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

<!-- [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex] -->
MemChain inference is Phala ACI-only; ordinary Rust nodes do not download or
execute embedding, extraction, or reranking models. Encrypted storage and
deterministic indexes remain local. The experimental `scripts/init.sh` can
configure local MemChain storage only; it deliberately does not enable the
server-side SuperNode because that runtime is restricted to SaaS mode. Clients
must use their direct, consented Phala ACI route rather than sending prompts
through an ordinary node. `scripts/download_models.sh` remains only as a
fail-closed compatibility entry point and does not download models. ACI
attestation does not by itself establish that every gateway/backend proxy is
blind to request plaintext; keep payload scope explicit and fail closed when
the trusted route is unavailable.

<!-- [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex] -->
The Rust SuperNode worker is also fail-closed for now: its current ACI client
verifies workload and receipt evidence but does not implement client-to-TEE
field encryption or source-sealed responses. Enabling that server worker would
expose prompt text and returned semantic vectors to the node, so an enabled
SuperNode configuration is rejected at startup until that complete transport
exists. Pending cognitive tasks are retained rather than claimed and retried.

<!-- [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] -->
Direct calls to the retained OpenAI/Phala and Anthropic provider adapters also
fail closed before appraisal or inference IO, even without the confidential
request flag. Phala's ACI attestation and verified-provider constraint cannot
authorize node-side plaintext inference. Bounded ACI request validation still
reports malformed or oversized inputs; valid requests report
`llm_client_to_tee_e2ee_unavailable` without changing provider health. Completing
client E2EE must not enable these legacy plaintext APIs. This is source-only,
uncompiled and untested, not an implemented client inference route.

The setup wizard writes credentials to `/etc/aeronyx/server.toml` with owner-
only permissions and never prints them. It does not configure or probe Phala;
client route configuration and reviewed measurements belong to the client-side
Phala integration, not this local-mode node profile.

## Phala Integration Acceptance Boundary

<!-- [PHALA-IDENTITY-PUBLICATION 2026-10-08 by Codex] -->
Unmanaged `start` and `pubkey` now share cancellation-owned, no-clobber identity
publication. A bounded unique owner-only staging file is completely written
and synced before an atomic same-filesystem hard link publishes the final
name. Concurrent initializers load the winner rather than replacing it or
reading a partially written final file. On Unix, newly created parent entries
are synced within the state filesystem, and every successful reader syncs the
identity inode and final parent before startup/public-ID output. This stops at
the volume boundary rather than syncing the unrelated container root. Mounted
volume attachment and storage-device durability remain operator dependencies.
Existing corrupt, oversized, non-regular or final-symlink inputs fail without
replacement; existing CMS-managed key loading and JSON encoding stay unchanged.
Failure after publication retains the final path. A crash before cleanup can
leave a private `.aeronyx-identity-*.pending` file; it is never automatically
promoted or used as a recovery identity. Do not delete an established identity
to repair a startup error, because sealed journals depend on that same key.
Non-Unix no-clobber publication does not claim Unix directory-fsync guarantees.
Regression source covers concurrent initialization, no replacement, failed
publication cleanup, links, oversized inputs and orphan staging files. It is
authored but uncompiled/unexecuted; power-loss and filesystem acceptance still
require later authorized verification on the exact Phala volume/artifact.

<!-- [PHALA-SOURCE-RETENTION-ADMISSION 2026-10-08 by Codex] -->
Fresh source admission now reclaims at most 64 authenticated expired journal
rows in the same transaction that inserts its Prepared row. The production
prepare/arm path already owns the bounded permit, journal lane, checked clock,
stop gate and immutable authority epoch; no background owner or new endpoint
is needed. Previously the standalone cleanup operation had no production
caller, so expired reservations could leave the source permanently full.
Only the sealed row's existing retention horizon authorizes removal, not its
execution deadline or a forged SQL expiry. Existing-route retries return
before reclamation and preserve their exact phases. A failed insertion or
post-mutation quota audit rolls back reclamation as well; reduced quotas do
not erase custody to fit. This does not promise permanent nonce/route exclusion
after the approved retention horizon, or physical SQLite file compaction.
Regression source covers horizon boundaries, restart, bounded batches, forged
expiry, insertion rollback and the actual prepare/arm producer's stop gate.
It is authored but uncompiled and unexecuted; it is not runtime acceptance.

<!-- [PHALA-SOURCE-MPI-COMPOSITION 2026-10-08 by Codex] -->
The source Pull role requires the existing VPN/MPI listener and MemChain
`local` or `p2p` storage composition, in live and recovery-only modes. `off`
does not construct MPI, and SaaS JWT owners are not private source identities;
these combinations now fail the parent configuration gate before source
journal open or migration. Startup also checks the observed complete storage
tuple, and API assembly checks the actual Local MPI mode plus VPN ownership
before evidence-store or listener effects. This does not enable local model
inference: deterministic encrypted storage/indexes remain distinct from models.
The endpoint-free private recipient still uses MemChain `off` and VPN disabled;
no source-only restriction is applied to it or to public relay-only nodes.
Regression source covers the real early Server::run rejection and the runtime
composition matrix. It is authored only, not compiled or executed.

<!-- [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex] -->
The private recipient journal reserves one complete Claim plus the maximum
Lease and Result carrier frames and route/origin identifiers for each job,
before returning fresh Claim bytes to the outbound worker. It charges at least
the actual stored bytes even for malformed oversized rows. A full byte budget
returns ordinary fresh-poll backpressure; it must not allow terminal execution
to consume another job's future Result space. Exact recovery does not charge
another slot. Result and ambiguous evidence retain their reservation until
authenticated NoWork retirement or expiry cleanup removes the row.

Reopen and legacy migration audit the same reserved budget transactionally.
A smaller budget is rejected without deleting custody or committing logical
schema migration; explicitly restore an adequate permitted budget to recover.
Current production recipient limits remain 64 MiB and at most 64 jobs; no new
configuration, wire format, endpoint or ephemeral-key persistence is added.
Authored regression source covers fresh-producer backpressure, exact recovery,
maximum Result growth, reservation release and migration rejection. It has not
been compiled or executed, and is not Phala runtime acceptance evidence.

<!-- [PHALA-QUEUE-CAPACITY-OPEN 2026-10-08 by Codex] -->
Reverse-queue `max_bytes` charges every non-tombstone item's envelope plus the
maximum Claim, Lease and Result frame budgets, not just bytes already written.
The reservation stays charged through Result recovery, so later frame growth
cannot consume space admitted for another item. Tombstones retain their item
slot but release frame reservation after payload removal. Durable NoWork polls
share both the global and per-recipient item caps and their replay byte charge;
exact existing replay does not acquire another slot.

Timed queue open now audits combined item/recipient/reserved-byte limits,
rejects a clock below the durable floor, extends eligible recovery horizons,
and cleans expired rows inside the same schema transaction. A pre-commit
quota/clock/audit failure rolls back logical migrations, clock and retention
changes; it never deletes custody to make a reduced quota fit. Existing
databases admitted by older accounting may
require an explicitly larger permitted budget before reopening. Both live and
recovery startup recheck the blocking opener's own timestamp before exposing
the queue API. A post-commit clock failure prevents API publication but does
not undo the committed database transaction. This is implemented source only:
regression source covers quota boundaries, exact replay, frame growth, restart
and migration rollback, but no
node compilation, tests, deployment or Phala runtime acceptance ran for it.

<!-- [PHALA-LEGACY-MEMORY-INGRESS 2026-10-08 by Codex] -->
The legacy MPI `/log` ingestion and `/search` plaintext FTS paths return a
fixed private/no-store HTTP 410 before unified authentication reads a body or
allocates a SaaS owner pool. Local/Bearer and SaaS/JWT modes are not evidence
of TEE execution. Artifact `/artifacts/search` rejects every nonempty `q`,
including encoded/duplicate query keys; empty/omitted `q` retains existing
owner-scoped enumeration and session/language pagination. Direct handler
mounts also reject before storage or rule-engine work. Search terms are not
included in the retained legacy telemetry. Status advertises
`plaintext_log_enabled=false` and `plaintext_search_enabled=false`.
No historical rows are deleted, migrated, decrypted or re-extracted by this
gate. Sealed recall, sealed writes and owner-authorized deletion keep their
existing authentication/wire contracts. URI/header bytes can still reach the
HTTP stack; rejection does not claim to prevent a caller from transmitting
plaintext or control external access logs. These changes are source-only:
regression tests are authored, not executed; no build or deployment was run.

<!-- [PHALA-DISABLED-BODY-BOUNDARY 2026-10-08 by Codex] -->
The same pre-authentication rejection covers the already-disabled `/remember`,
`/embed` and `PATCH /record/:record_id` routes. Their bodies must not be polled,
buffered or hashed by remote authentication before the 410 response. Existing
mutation error strings are retained, and direct mounts use the same no-store
response. This is method-specific for record mutation: authenticated sealed
record GET, provenance GET, sealed writes, sealed recall and deletion remain
on their existing paths. It does not make the HTTP stack or external access
logs unable to receive caller-supplied bytes. This increment is source-only,
uncompiled and untested.

<!-- [PHALA-FULL-PATH-ACCEPTANCE 2026-10-08 by Codex] -->
Keep the node delivery path and the source-to-ACI inference path separate.
An opaque private `BlindVaultPull` result proves neither that an inference
request used source-owned field encryption nor that a deployed gateway matches
the serializer, key custody and application identity expected by that source.
The retained node plaintext provider API must stay disabled even after a
separate client E2EE path is connected. Candidate serializers, successful quote
generation and source-only test assertions do not satisfy these requirements.

| Boundary | Required acceptance evidence | Current evidence limit |
| --- | --- | --- |
| Source-to-ACI caller | Actual caller sends the one frozen encrypted wire body, verifies appraised key/provenance and signed request/response binding, then opens only that same response snapshot | The node evidence bundle contains no accepted source-owned caller/transport result; request/response primitives and pinned-source byte contracts are not that wiring |
| Hosted gateway identity | Accepted measured application/build, effective numeric feature profile and response-key custody match the source's pins | Formal deployment trust and positive real quote/collateral evidence remain absent; a reference source commit, image label or peer quote is insufficient |
| Reverse-onion private delivery | Authenticated source admission, exact ciphertext custody, recipient-selected Lease, source-verified opaque result, duplicates/ambiguity, renewal and restart recovery | The connected in-process test now passes 18 scenario combinations, and production HTTP carrier fault tests pass 14 combinations; neither proves independent-host TLS or KEM custody |
| Deployment artifact | Exact reviewed source, locked dependencies, Linux amd64 executable/image/Compose hashes and unwind policy, least-privilege roles and persistent state agree | Linux amd64 compilation and executable hashes are recorded; the image has not been built or the current executable run on Linux/Phala. The later Docker context repair still requires context refresh before image construction |
| Lifecycle and limits | Actual accepted work remains bounded and owned through stop, cancellation, durable completion and response drain; forced termination recovers without re-execution | Native loopback startup, SIGTERM and identity-preserving restart pass with VPN/discovery/MemChain disabled; unit fault/drain tests are not independent-host forced-termination or Linux TUN acceptance |

These rows are conjunctive, not interchangeable assurance levels. Preserve
fail-closed activation while developing dependency-ready callers; missing
hosted evidence is not permission to substitute a candidate pin or plaintext
fallback. Durable inference recovery, charging and retention also require their
own approved contracts; the vault queue's custody protocol does not approve
those policies by analogy.

<!-- [PHALA-COMPLETION-EVIDENCE-AUDIT 2026-10-08 by Codex] -->
The later user-authorized Rust verification phase supersedes the earlier
source-only status notes, without retroactively making them execution evidence.
The local evidence bundle now records 1,209 distinct related passing Rust
tests, not a whole-repository test result. Fourteen of those specifically cover
the plaintext ingress boundary: actual MPI middleware rejects before body
polling in Local and SaaS modes; direct handlers reject plaintext writes and
search, preserve historical rows, reject unsealed semantic vectors and retain
authenticated sealed/metadata paths. The raw-log storage helper's historical
plaintext compatibility is not enabled by these rejected ingestion routes.
<!-- [PHALA-SEALED-MPI-REOPEN 2026-10-08 by Codex] -->
Eight additional signed MPI tests cover positive writes/retries, owner quotas,
storage errors and pagination. One new regression closes every router/database
holder and reopens disk storage twice, retaining exact opaque envelope/source
signature bytes, rejecting other owners and inner-record tampering, and keeping
revocation effective against retry. Its AMV2 fixtures are synthetic opaque
envelopes, not a source AES decryption test. Same-process database reopening
does not prove cross-host replication, process-kill or power-loss durability.
Separately, 40 renderer-to-native-configuration subprocess cases match their
expected results (21 accepted, 19 rejected). They are not additional Rust unit
tests and use restricted documented interpolation, not the Docker Compose
engine. Synthetic trust pins validate configuration shape only, never trust.

The evidence files `verification-evidence.json`, `memory-boundary-evidence.json`
and `rendered-config-evidence.json` retain exact artifact/source/log hashes and
scope limits outside the repository. The exact original trial's latest update
was refused by the platform with HTTP 400 and a terms-of-use message; the API
did not disclose the actual trigger. Its last confirmed state is stopped. Do not retry or use
another application/host to bypass that refusal. Platform resolution and later
authorized Linux/Phala execution must establish the outstanding rows against
the exact artifacts; the full goal is not complete.

<!-- [REPLICA-ADMISSION-CLOCK 2026-10-08 by Codex] -->
Explicit replica-job admission samples time inside its blocking lane, rejects
a sample older than ingress, and validates the complete target bundle there.
The coordinator also revalidates immutable target effects against their
original configured bounds when a typed submission is admitted later. This
does not extend signed inventory freshness or refresh a source authorization.
The durable replica-job store currently stages one exact job only: no production
caller loads that job into an outbound replica workflow, and the exact-replay
transport has no production I/O implementation. HTTP 202 therefore means
admission, not replication completion. A connected source-owned worker with
durable exact dispatch/reply recovery, fresh authority, cancellation/drain and
terminal verification remains required; do not infer it from core workflow
types or the separate reverse-onion Pull runtime. These production-source
changes also make the earlier native/Linux executable hashes stale for this
admission path until the corresponding artifacts are rebuilt and verified.
The updated native test executable compiled and passed 17 exact inspected
replica admission/storage regressions, including the two new clock cases.
The cumulative count includes earlier executions on their recorded artifacts;
it does not mean all earlier tests were rerun against this production change.

<!-- [REPLICA-VOLUME-DURABILITY 2026-10-08 by Codex] -->
The shared replica job/recovery file opener retains pinned ancestry descriptors
until the final storage device is known, then syncs parents deepest-first
within that filesystem only. It stops at the nearest device boundary instead
of requiring synchronization of an unrelated read-only container root. A failed
sync inside the final filesystem still prevents lock/state activation, and a
retry rechecks existing entries rather than skipping ambiguous mkdir results.
No-follow resolution, owner checks, private modes, single-link files, exclusive
process fencing and exact publication/cleanup rules remain enforced. A path
temporarily retains one descriptor per component; descriptor exhaustion fails
closed. Device IDs do not independently identify same-filesystem bind mounts.
The native test executable compiled and passed 22 exact inspected I/O and
job/recovery-store cases, including two new volume-boundary/walk regressions.
The boundary test simulates device IDs; it is not a real Linux read-only-root
mount test. Store reopening uses synthetic opaque state in the same process,
not process-kill/power-loss acceptance. Actual Phala volume guarantees and
updated native/Linux production artifacts remain unverified. This shared I/O
repair does not implement the missing outbound replica worker.

<!-- [VAULT-CANCELLATION-BUDGET 2026-10-08 by Codex] -->
Blind Vault request admission now shares its pre-body permit with the queued
or running blocking operation. Cancelling an HTTP future cannot release a
slot while that operation still owns work. Lease admission, writes, deletion,
pulls, issuer-directory reads and replica-job staging all use this ownership
path; local/public mounts retain their existing shared mutation/pull budgets.
Errors, malformed/oversized extraction and unwinding release the final owner.
The updated native test executable compiled and passed all 14 exact inspected
Blind Vault API tests, including three new cancellation/release regressions.
The cancellation case uses a real in-memory router and a gated synthetic
admission callback, not an actual slow-disk or network replication workload.
Already started storage work is not cancelled or rolled back merely because
the client disconnects. This change adds neither an outbound replica worker
nor a shutdown/drain guarantee, and production executables require rebuilding.

<!-- [PHALA-CURRENT-NODE-PROVENANCE 2026-10-09 by Codex] -->
The subsequent artifact refresh incorporates the admission-clock, final-volume
durability and cancellation-budget repairs into both production executables.
One read-only 450-file public-source snapshot, with locked dependencies and
all current Rust sources in the five public crates, produced the native macOS
executable and the Linux amd64 `phala` executable. Exact source, build-log and
immutable artifact hashes are recorded in `current-artifact-evidence.json`
outside the repository. The Linux ELF retains unwind data, requires GLIBC at
most 2.34, and has no ONNX/Torch/CUDA dynamic-library dependency. These are
compilation and static linkage checks, not Linux execution or TEE appraisal.
The refreshed native executable passed two real startup/health/SIGTERM rounds
with one loopback listener and the same persistent public identity and 0600
identity inode; identity contents were never read or hashed. VPN, DNS,
MemChain and discovery were disabled in these process rounds. Historical
passing tests were not all rerun against these refreshed executables. Earlier
artifact-staleness notes above describe the pre-refresh state; the current
source archive is refreshed, but the image remains unbuilt. No Linux/Phala
execution, outbound replica worker, cross-host recovery or full-goal completion
is claimed.

<!-- [PHALA-CURRENT-SOURCE-PACKAGE 2026-10-09 by Codex] -->
The refreshed local public-source archive contains exactly the snapshot's 450
build inputs and its manifest. Every archived file's actual bytes match the
same compiled snapshot and current authoritative source; missing, duplicate,
extra, private-client, identity and operator-config names are rejected by the
calibrated packaging gate. Extended attributes, ACLs and file flags are omitted.
The literal Docker COPY/deny-all exceptions were rechecked, including both
non-secret role templates; this is not execution of Docker's pattern engine.
`current-source-package-evidence.json` records the immutable archive hash and
checks outside the repository. No source package, artifact or image was
uploaded, and the platform refusal has not been retried.

## Isolated Phala Container Profile

`Dockerfile`, `compose.phala.yaml`, and `server.phala.example.toml` provide a
local-build container profile for an isolated Phala trial. It deliberately has
no published ports or host networking, drops all capabilities except
`NET_ADMIN`, and passes only `/dev/net/tun`. The sample binds the VPN UDP
listener to loopback, disables discovery and MemChain, and sets
`management.enabled = false`; the disabled management runtime does not create
the CMS client, perform public-IP discovery, or start heartbeat/session workers.
This prevents an isolated trial from accidentally contacting the production
CMS. Existing production configurations remain management-enabled by default.

<!-- [PHALA-ISOLATED-VPN-IMAGE 2026-10-08 by Codex] -->
The isolated Compose build explicitly selects Dockerfile stage `isolated-vpn`,
which adds only `iproute2` for the Linux TUN lifecycle and bakes the isolated
sample with `vpn.enabled = true`. The default final stage remains `peer`,
without network tools and with VPN disabled. The isolated build uses its own
`aeronyx-node:isolated-vpn-local` tag, leaving the default peer tag untouched.
Do not substitute the peer image
for the isolated VPN stage: Linux TUN activation requires the `ip` executable.
Disabling VPN still starts the required node HTTP API; it does not exercise
the Linux TUN lifecycle. Neither stage expands container capabilities or publishes
ports; those remain explicit Compose/operator decisions. This stage-selection
repair is statically inspected only; no container image or Linux TUN runtime
has yet been verified for it.

Use a separately controlled config file and persistent volume when launching:

```sh
AERONYX_SERVER_CONFIG=/absolute/path/server.phala.toml \
  docker compose -f deploy/node/compose.phala.yaml up --build -d
```

Start from `server.phala.example.toml` and keep node identity in the named
`aeronyx-state` volume. Do not place registration codes, private keys, or live
production node configuration in the image or repository. The sample is a
loopback-only harness, not a public VPN endpoint: without an operator-provided
network profile and reachable listener it cannot accept public client traffic.
The container/Compose profile alone does not attest application identity or
prove TEE status, VPN reachability, or reverse-onion delivery. Phala CLI
deployment, image provenance, and the restricted-CVM runtime still require
separate verification before any trial deployment.

### Phala Peer Attestation API Image

<!-- [PHALA-PEER-ATTESTATION-PROFILE 2026-10-06 by Codex] -->

`Dockerfile` bakes `server.phala.peer.example.toml` as the default config for a
Cloud image. The Dockerfile and Phala Compose services target `linux/amd64`
for the Intel TDX CVM profile; an ARM-hosted builder must build this target,
not emit a native `linux/arm64` image. Build and publish the image through the
separately authorized release process. The default rendered manifest has no
externally published port. Only after ingress is authorized, render the public
peer mapping with its immutable image digest and discovery seeds. The public
HTTPS endpoint may be omitted on the first render because Phala may assign it
only after creating the app.

<!-- [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] -->
The Dockerfile selects `cargo build --profile phala --locked -p aeronyx-server`
and copies `target/phala/aeronyx-server`. This profile inherits release
optimization but uses `panic = "unwind"` so reverse-owner fault/drain boundaries
can run. Ordinary `--release` retains its existing `panic = "abort"` policy and
cannot enable the reverse queue, recipient or source, including recovery-only
roles. Configuration and the runtime catch boundary reject that incompatible
binary before reverse work begins; there is no environment bypass. A profile
name, image tag or digest alone is not proof of the compiled panic policy:
authorized verification must establish the exact executable and exercise its
failure/drain behavior. Existing ordinary nodes with all reverse roles disabled
retain their prior build/config behavior. These source changes are unbuilt and
untested, and do not change the currently deployed trial executable. Regression
source covers all role/recovery combinations against both panic policies and
the workspace-profile/Dockerfile build-and-copy contract; it has not run.

<!-- [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex] -->
The server now registers SIGTERM and SIGINT before yielding into startup and
carries the first signal observation through shutdown. A signal buffered during startup
is checked before READY; the recipient readiness wait may also yield to that
signal without dropping its worker. Known runtime failures retain precedence,
and reverse intake closes before ordered owner/journal drain and generic task
shutdown. Signal setup failure rejects startup before any runtime work is
accepted. This does not cancel arbitrary storage initialization mid-operation
or guarantee prompt progress through a stalled disk operation. On non-Unix
platforms the retained CTRL_C future is polled once before startup.

Both Phala Compose templates explicitly request SIGTERM with a two-minute
stop grace; the peer renderer rejects missing, duplicated or changed policy
keys independently for the public and private service. The isolated trial
template retains its existing permissions and has no new ingress. The grace
is an operator allowance, not a proven upper bound on drain: external forced
termination, runtime teardown and a stalled filesystem still require normal
authenticated restart recovery. The process does not force exit, abandon an
accepted journal write or renew an execution lease to meet that allowance.
Regression source covers buffered signals, cancelled waiters, failure
precedence, rendered role combinations and deliberately broken templates.
It has not been executed; no container stop/restart or signal probe was run.

For the separately authorized public-peer manifest:

```sh
AERONYX_NODE_IMAGE='registry.example/aeronyx-node@sha256:<64-lowercase-hex-digest>' \
AERONYX_DISCOVERY_SEED_ENDPOINTS='["https://seed.example.net"]' \
  ./deploy/node/prepare-phala-compose.sh --public-peer \
  > /Volumes/disk/compose.phala.peer.locked.yaml
```

This first-stage manifest intentionally has no descriptor endpoint or quote
socket. After Phala assigns the public origin, rerender with
`AERONYX_DISCOVERY_PUBLIC_ENDPOINT='https://peer.example.net'` to enable them.
<!-- [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] -->
An explicitly empty endpoint environment value removes both the mounted
`discovery.public_endpoint` and its `network.public_endpoint` fallback. An
absent variable preserves TOML configuration. Nonempty endpoints are validated
verbatim, so leading/trailing whitespace is rejected rather than trimmed.
The Rust loader applies the API listener override before checking the endpoint,
then validates the final configuration; it does not silently enable discovery
or a quote socket to make an incompatible profile pass. Isolated-process
loader and signed-descriptor regression tests cover these transitions in source
but have not been executed.

Without `--public-peer`, the peer process has no Phala app ingress, API
listener, or quote socket. Its public endpoint is forcibly empty and public
discovery is disabled, so it cannot gossip an unreachable host address. The
explicit public mode requires 1-64 public HTTPS discovery seeds and enables
the `8422` API listener and matching port mapping. If the endpoint is already
assigned, it also enables the dstack socket and nonce-bound attestation API.
Otherwise the first-stage render leaves the endpoint and attestation socket
empty; Rust rejects quote transport without a public descriptor origin. After
Phala assigns the endpoint, render again with
`AERONYX_DISCOVERY_PUBLIC_ENDPOINT='https://peer.example.net'` and deploy that
exact manifest to enable the quote API and signed feature advertisement.
To render an additional, endpoint-free private-recipient process with its own
persistent identity volume and no published port, render with:

```sh
AERONYX_NODE_IMAGE='registry.example/aeronyx-node@sha256:<64-lowercase-hex-digest>' \
AERONYX_REVERSE_ONION_RELAY_NODE_ID='<64-hex-character relay identity>' \
AERONYX_REVERSE_ONION_RELAY_ENDPOINT='https://relay.example.net' \
  ./deploy/node/prepare-phala-compose.sh --private-recipient \
  > /Volumes/disk/compose.phala.private-recipient.locked.yaml
```

The renderer fails before producing output unless the relay ID is a nonzero
64-hex node identity and the relay endpoint is a credential-free public HTTPS
origin. It also renders both values as Compose required-value expressions, so
the same protected environment must be present when the locked manifest is
deployed. Rust independently validates the endpoint and signed relay identity
at startup. The renderer removes the Compose profile marker itself, so
activation does not depend on Phala CLI handling of Docker Compose profiles.
Keep and review the exact rendered manifest: enabling the private service
changes the app composition measured for attestation.

<!-- [PHALA-RENDER-REQUIRED-PINS 2026-10-08 by Codex] -->
An enabled public reverse queue requires an explicit one-element recipient-ID
JSON array, including in recovery-only mode. Live mode also requires a
nonempty source-ID array; only explicit recovery-only mode omits those sources.
The renderer rejects unset/empty required arrays before emitting a manifest,
and bounds identity inputs to 8 KiB of UTF-8 bytes, matching Rust startup.
For an explicit private-recipient render, its recovery-mode override must be
empty, `true` or `false`; empty preserves the mounted TOML policy. These are
pre-render checks, not replacements for Rust's identity/signature validation,
existing-history checks, admission snapshots or drain ownership. The manifest
still contains protected environment expressions, not frozen environment
values: keep the same reviewed inputs at deployment. Regression source for
missing live/recovery pins is authored only; no renderer, tests, build or
deployment has been executed for this correction.

<!-- [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] -->
DNS origins, including Phala gateway hostnames, are checked syntactically
without DNS lookups during rendering. Use ASCII DNS labels (or their explicit
punycode form); empty, oversized, underscore, and edge-hyphen labels are
rejected. IP literals must pass the same explicit public-unicast exclusions
as Rust, including IPv6 documentation and tunnel ranges. Recipient identity
and endpoint values are validated exactly as supplied, without trimming.
Rendering does not establish reachability, signed authority, or TEE trust;
runtime resolution still checks and pins the complete public address set.

<!-- [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] -->
The public peer needs operator-selected discovery bootstrap peers. The
`--public-peer` renderer requires `AERONYX_DISCOVERY_SEED_ENDPOINTS`; it parses
this value and fails closed unless it is a JSON array of 1-64 public HTTPS
origins. Supply the array in the protected app environment, for example
`["https://seed-a.example.net","https://seed-b.example.net"]`. The Rust loader bounds
the encoded value to 8 KiB and rejects non-HTTPS, credentialed, or private
targets. The private recipient uses its identity-pinned relay as its only peer
destination. Rust rejects general seed endpoints, remote bootstrap snapshot
URLs, Directory Replica Sync peers/mirroring, and MemChain commitment sync.
<!-- [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] -->
Each service owns one discovery-seed environment entry. The private service
always has a literal empty value, even in combined public/private renders;
only the public role interpolates operator-selected seeds. The Rust loader
treats an explicitly empty seed value as clearing the mounted TOML seed list;
an absent environment variable preserves that list. Nonempty values still
require 1-64 valid origins, so whitespace and `[]` are not clearing aliases.
The startup self-check does not report a private role as pinned-relay-only
while general seeds remain. Role-matrix rendering, clearing, and startup
regression tests are authored but have not been executed.
At runtime, HTTPS DNS seed requests resolve a bounded complete address set;
every answer must be public-unicast, and the client pins those addresses while
TLS continues to authenticate the configured hostname. Mixed public/private
answers fail closed. Public-IP seed behavior is unchanged.
<!-- [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] -->
The gossip runtime suppresses cached-peer fanout and generic relay probes for
this role. Local peer-cache and snapshot-file imports remain local and do not
authorize outbound connections. Its startup self-check reports
`pinned_relay_only` for the endpoint-free recipient profile.

To offer the public Phala peer as a no-exit onion middle hop, render with
`--public-peer`, set the assigned `AERONYX_DISCOVERY_PUBLIC_ENDPOINT`, and set
`AERONYX_PHALA_ONION_RELAY_ENABLED=true` in the protected app environment.
The renderer rejects relay opt-in without both the public ingress profile and
the assigned HTTPS origin; first-stage endpoint discovery must keep the relay
disabled and re-render after Phala assigns the origin.
This explicitly enables the bounded ChatRelay ciphertext store and the
`OnionMiddle` descriptor capability; advertisement still requires the assigned
HTTPS endpoint, enabled public discovery, and a ready peer API. It does not
publish a VPN UDP port, enable Blind Vault's public API, or turn on MemChain
inference. The default is `false`. Keep this setting false for the separate
endpoint-free private-recipient identity; Compose pins that role to false and
the Rust config loader rejects enabling both roles in one process. Existing
ChatRelay quotas remain the configured resource ceiling, so review them before
accepting public relay traffic.
<!-- [PHALA-PRIVATE-RELAY-DURABILITY 2026-10-06 by Codex] -->
An explicit `false` always removes the signed `OnionMiddle` advertisement. On a
public-peer process it also disables the local ChatRelay service, overriding a
mounted TOML value. On the endpoint-free private-recipient process, Rust keeps
ChatRelay's local durable store enabled because recipient polling depends on
it; the process remains unadvertised and has no ingress listener or published
port. Unset preserves the mounted TOML value.

<!-- [PHALA-ONION-DNS-PIN 2026-10-06 by Codex] -->
Signed HTTPS DNS origins, including the Phala-assigned app hostname, are
supported for explicitly pinned onion-middle routes at both the source and
forwarding relay. Before constructing/sending the onion POST, the source
resolves the selected signed hostname once; before a relay arms its forwarding
effect, it resolves the next-hop hostname once. Each rejects the entire answer
set if any address is non-public, pins the validated addresses for that
request, disables proxy inheritance and redirects, and retains the original
hostname for TLS certificate verification. DNS names remain rejected for
ordinary direct peer transports, generic probes, and permissionless seed
admission; HTTP DNS endpoints are never accepted. A successful TLS connection
or HTTP receipt is still not execution completion: the existing signed opaque
response must be verified at the source.

<!-- [PHALA-REVERSE-QUEUE-DEPLOYMENT 2026-10-06 by Codex] -->
The reverse-onion task queue is a separate public-relay opt-in. It remains
disabled unless `AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED=true` is supplied
with exactly one recipient ID and 1-64 source IDs. It also requires
`AERONYX_PHALA_ONION_RELAY_ENABLED=true`; the renderer rejects an enabled queue
without this public OnionMiddle/ChatRelay role before producing a manifest.
The renderer checks ID array size/shape, duplicates, role overlap, recovery
mode, and all-or-none authority seed fields before output. These checks are
configuration preflight only; Rust still verifies Ed25519 identities and the
canonical signed descriptors/grant at startup and per request. Recovery-only
mode may omit source IDs and authority seeds but still requires exactly one
recipient ID and the existing durable queue database.
Enabling the generic onion relay does not enable the queue. The queue uses its
persistent database path and bounded limits from the TOML profile. Signed
relay/recipient descriptors and authorization may be supplied as historical
cache seeds, but are not required: fresh signed descriptors and the
recipient-signed grant must arrive through authenticated discovery before the
request-time authority check permits any enqueue or new lease. Until then the
queue may start and recover durable rows, while fresh work fails closed.

The queue also requires the assigned public HTTPS origin. On Phala's first
`--public-peer` render that origin may not exist yet; keep queue enablement
false, obtain the assigned origin, then render again with
`AERONYX_DISCOVERY_PUBLIC_ENDPOINT` set before enabling the queue. The renderer
and Rust startup both reject an enabled queue without that routable origin.

Provision each role's identity in its own persistent volume before enabling
the queue. Run `aeronyx-server pubkey --config /etc/aeronyx/server.toml` with
that volume mounted; in an unmanaged profile it creates the key with
create-new semantics and prints only the public node ID. Never copy the key
between the public peer and private recipient volumes. Configure the private
recipient with the public relay ID and the public queue with the recipient ID
plus the explicit source-ID allowlist. Then start both roles and allow gossip
to exchange current descriptors and P's signed grant. This bootstrapping is
identity-pinned and signed, not anonymous registration; wrong identities,
stale descriptors, missing grants, or an absent source allowlist leave new
work rejected.

<!-- [PHALA-QUEUE-RECOVERY-ENV 2026-10-06 by Codex] -->
To recover an existing relay queue, set
`AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED=true`,
`AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY=true`, and the one pinned
recipient ID in the protected public-peer environment. Recovery mode may omit
the source allowlist and signed authority seeds: it opens only the existing
nonempty queue database, replays exact durable Claim/Lease state, and rejects
fresh Claims and all new enqueue attempts. Keep the database volume attached.
To return to live admission, review durable rows first, restore the explicit
source allowlist, set recovery mode to `false`, and rely on current authenticated
discovery authority. Unset preserves the TOML setting. Invalid values, a
missing recipient pin, or applying this override to the private-recipient
service fail closed. This switch does not bypass signature/identity checks or
prove TEE key custody.

<!-- [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] -->
The private recipient rechecks a restored Lease inside its Armed transaction.
If a recovery page ages past execution expiry before that transaction runs,
the worker skips dispatch without treating normal expiry as process failure.
The exact row remains until its original replay-evidence cleanup bound; an
Armed or ambiguous job never becomes executable again through this handling.
Claim and Result retries are also checked after DNS preflight and their
network waits are capped by their immutable retry bounds. Claim recovery may
outlive its short admission freshness, but Result retry never extends its
signed result grace. Signature failure, corruption, and clock rollback remain
fail-closed rather than being classified as ordinary expiry.

<!-- [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] -->
SourceQuery now samples trusted time after relay operation/connection locks
and SQLite reads, rather than reusing a pre-wait timestamp. The read-only
snapshot carries its original state-specific availability bound internally.
Relay signing and publication recheck that bound and the signed query's
freshness; source chain verification and journal-lane entry also recheck the
source's immutable recovery bound. Waiting cannot renew an expired query or
extend result retention. No wire format, stored deadline, or SQL read-clock
durability changes. Clock regression/failure remains fail-closed. This is
implemented source only, uncompiled and untested, not deployment acceptance.

<!-- [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] -->
Both recipient and source Result persistence now sample their runtime clock
inside the audited SQLite transaction, after lock acquisition and integrity
work. The terminal's owned late-completion task and the source evidence owner
call these refreshed entry points directly. The source rechecks its immutable
recovery bound there; the recipient verifies the exact Result against the
stored Claim/Lease at that sampled time. Exact duplicate acknowledgement never
authorizes another execution, and rollback, quota, poison and post-commit fences
remain in place. Test source covers post-lock expiry/clock failure and unchanged
durable state, but has not been executed. This is not closed-loop acceptance.

<!-- [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] -->
The private terminal carries the committed Armed timestamp through queued
preparation, the final router-entry sample, response completion and the result
transaction. Each phase must be at or after its predecessor, not merely after
the original preparation time. Local clock failure or rollback closes intake
and publishes the existing recipient-owner fault even if the HTTP waiter has
already timed out or disconnected. It never authorizes restoring the signed
lease; Armed remains non-reexecutable and restart audit retains ambiguity.
Forward execution expiry remains a zero-entry deferral, and result grace is
not extended. The bounded test-only clock samples cover a post-entry rollback
that the former preparation-only comparison accepted. Test source is authored,
not executed; no build, runtime clock experiment or deployment was performed.

<!-- [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] -->
Source result opening now takes a live local clock sample inside the audited
SQL transaction before restoring its sealed reply session and again after
local crypto, before publishing Verified data. Cached Verified reads use the
same post-lock live fence. Raw samples must not regress below the trusted
entry or previous observation; the independent monotonic retention bound is
applied only after that check, never as a fallback hiding clock failure.
OS-clock failure or rollback poisons the existing source owner, preserves the
last durable ResultReady/Opening/Verified phase and cannot authorize another
task POST. The source lane and permit still outlive a cancelled waiter; failure
remains visible during shutdown. An authenticated reopen may repeat only the
local Opening operation or read the retained Verified cache. Forward retention
expiry is not an owner fault and never extends retention. Private HTTP mapping
returns unavailable without a completed page or typed Pending for local faults.
Authored regression source covers these paths, including cancellation/restart;
it has not been compiled or executed. No runtime clock probe or deployment ran.

<!-- [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] -->
Recipient Claim persistence, Lease acquisition, Armed admission, NoWork poll
retirement, recovery scans, cleanup and zero-dispatch restoration now use a
checked time after SQLite lock/audit waits. The worker's Claim, Lease, Armed
and NoWork mutations sample the live clock inside the transaction, including
time spent signing or decoding before the call. Scalar journal interfaces
advance one per-call monotonic anchor across the wait. Fresh Claim
admission still differs from exact historical Claim replay. An authenticated
NoWork receipt or Lease that expires during the wait leaves the poll unresolved;
clock rollback, altered bindings and storage faults remain fail-closed. An
expired Lease cannot be restored even after proven zero dispatch. The adapter
also checks its execution deadline immediately before local router entry,
after request construction and queued preparation gates. Wire bytes, immutable
deadlines, recovery-only mode and custody-versus-execution semantics are unchanged.
Regression source is authored only; no build, test or deployment was run.

<!-- [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] -->
The source checks fresh admission separately inside the Prepared and Armed
transactions, after sealing, SQLite waits and integrity audit. Its live owner
also checks the shared stop flag and clock floor at each transaction boundary
while retaining the same signed-authority read guard. Expiry or shutdown before
Armed leaves an existing Prepared row zero-send; it cannot authorize a POST or
claim relay custody. Clock failure/regression remains a sticky owner failure,
not a normal shutdown. Historical Armed/DispatchAmbiguous recovery, exact bytes,
immutable deadlines, wire formats and existing private API statuses are unchanged.
Regression source covers rejected prepare/arm state, restart, stop/fault
supervision and non-Pending API errors. It has not been compiled or executed.

<!-- [PHALA-CONNECTED-REVERSE-LOOP 2026-10-07 by Codex] -->
Connected regression source now composes real relay HTTP middleware, signed
queue APIs, recipient journal/Armed admission, the private Blind Vault router,
terminal Result persistence and the source's signed evidence verification and
reply opening. Source HTTP transport uses in-process router calls; the fixture
drives recipient Claim/Result exchanges through the real APIs and the worker's
Lease response verifier. Actual recovery-only worker startup and scheduling
perform Lease arming and terminal dispatch, then drain before journal reopen.
The synthetic ciphertext is provisioned through signed admission and stored in
a real vault with its public API disabled. Cases include exact duplicate
ingress/Claim/Result, lost custody ACK, altered signed evidence, source retry
without another POST/execution, and both journals reopened after drain.
The fixture uses one process's ephemeral KEM manager with separate signing
identities, so it cannot establish independent-host key custody, complete
server startup, live polling transport, DNS/TLS, deployed Phala provenance or ACI attestation.
This is authored test source only,
not an executed test result or completed deployment acceptance.

<!-- [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] -->
Six additional connected cases use fresh recipient mode and the actual owned
worker to create/persist its Claim, accept its Lease, arm the journal, execute
the private vault request, persist the Result and send it back to the relay.
Only the network carrier is replaced through a `cfg(test)` constructor;
production startup still constructs the fixed pinned HTTP carrier. The test
carrier calls real relay router APIs and deliberately loses the first accepted
Claim and Result responses. It requires byte-identical retries, the identical
relay Lease, no replacement Claim, no Result before terminal persistence, one
terminal execution, and the exact durable Result after worker drain/reopen.
Each fresh case then starts a new recovery-only worker with a new terminal
adapter/stop gate and requires one further identical Result send, without a
new Claim or terminal re-execution. No live adapter is reopened after drain.
The existing recovery-only stop-during-persistence cases are retained. All six
source combinations (direct/authenticated HTTP, success/lost custody ACK/altered
signed evidence) now have both recipient modes. The injected carrier's HTTPS
acknowledgement flag is synthetic; these cases do not test certificates,
DNS/SSRF enforcement, independent-host KEM ownership, complete server startup,
TEE inference or deployed ACI/gateway E2EE. They remain authored and unexecuted.

<!-- [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex] -->
Six further cases extend that same connected loop rather than substituting a
separate success oracle. After the real relay has durably armed its Lease,
the test carrier imports newer signed R/P descriptors with unchanged identity,
origin and KEM, withholds their replacement grant, and loses the accepted Claim
reply. The actual recipient worker must retry the identical Claim/Lease and
persist/retry the identical Result while fresh Pull authority is absent.
Only after completion is the matching new grant signed at a later live second
and imported as a positive control (the existing same-second grant conflict
check is preserved, with a bounded wait rather than a forged future timestamp);
a new recovery-only worker must still resend Result without a new Claim or
terminal re-execution. Direct and authenticated source HTTP cases retain the
success, lost custody ACK and altered signed evidence variants. Authenticated
source journal metadata must retain the original descriptor commitments, not
the renewed pair; one source task POST and one terminal entry remain required.
These are uncompiled, unexecuted test assertions, not independent-host,
DNS/TLS, KEM-rotation or deployed Phala/ACI acceptance evidence. No production
authority gate, wire format, transport, persistent secret or inference path is
changed by this test-source extension.

<!-- [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] -->
The connected fixture also starts at the mounted source HTTP router using the
core wallet-signed request constructor. Its API cases use the production
identity-pinned runtime constructor with current authenticated PeerStore
authority, not the direct lifecycle fixture's static policy. They cover wrong
owner and altered signature rejection before journal/transport effects, stale
signed input refusing an unknown route, lost custody ACK and altered evidence
recovering the same durable route, source-verified signed ciphertext pages,
cached HTTP retries without another POST/execution, and audited journal reopen
after drain. <!-- [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] -->
The API cases now compose the same dedicated MPI/source constructor as the VPN
server and supply actual Local Bearer or Remote Ed25519 authentication, not a
trusted-owner extension. The remote digest covers the exact source JSON bytes.
Separate MPI cases cover valid SaaS JWT rejection, pre-auth body capacity,
caps/timeouts/cancellation/drain, and generic chat route compatibility.
<!-- [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] -->
The actual connected source API cases also discard a completed HTTP response
before consuming its body, then recover the identical Verified page without
another POST, evidence query or execution. Separate resource fixtures cover
unpolled/partial response expiry, cancellation-safe drain, exact bytes on normal
completion, detached bounded chunks and oversized-stream rejection.
These are authored test cases, not executed results. They do
not prove complete server startup, independent-host key custody, HTTPS carrier
behavior, deployed Phala provenance, attestation, or client-to-TEE E2EE.

<!-- [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] -->
The connected fixture starts the production recipient worker from a durable
LeaseReady row with recovery-only enabled and an empty discovery store. That
store prevents carrier network entry; the fixture, not the HTTPS carrier,
submits the persisted Result to the real relay router. A bounded test-only
gate holds the owned terminal result before persistence: worker stop/drain
must stay pending until the gate is released, then both journals reopen via
their existing-only audited openers. Terminal router entries are counted at
the actual middleware, not by manually invoking dispatch. Separate actual
worker-start cases reject missing and unowned recovery journals without READY
or empty-state bootstrap. These cases are authored but not executed and do
not prove live polling, certificate validation or deployed TEE attestation.

<!-- [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] -->
The recipient worker rechecks its shared worker/terminal stop flag inside the
audited journal transaction immediately before allocating a fresh Claim or
changing a validated Lease to Armed. The gate follows stored-clock and frame
validation, so concurrent stop does not hide rollback, corruption or storage
failure. Explicit local intake closure rolls back the transaction and leaves
the previous exact LeaseReady state restartable; a later stop alone cannot
reopen an Armed row. The existing separately proven zero-dispatch restoration
contract remains unchanged. A LeaseReady projection skipped at the final gate
still blocks fresh polling for that recovery pass; the next scan must classify
its durable state rather than treating it as resolved custody.
Existing exact polls and accepted Lease/Result completion
remain recovery operations, not fresh intake, and keep their normal durability
and expiry checks. In particular, closing intake does not cancel a terminal
Result writer already waiting on the journal lane. Regression source covers
post-lock stop, unchanged rows/clock, restart, failure precedence and late
Result persistence. These cases have not been compiled or executed.

<!-- [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] -->
Recipient recovery scans, fresh Claim persistence, and late terminal-result
commits share one per-journal blocking-operation lane. Its callers are the
single worker and at most four tracked terminal operations, not public request
waiters. A terminal operation retains its execution permit through the DB wait
and durability fence, including after its original caller times out. Shutdown
closes new execution admission but does not close this completion lane; drain
waits for accepted operations and their result commits. This prevents ordinary
in-process scan/write contention from discarding a completed opaque result.
Storage failure, corruption, and result-grace expiry still fail closed and
leave the durable Armed barrier non-reexecutable.
Fresh Claim preparation and Lease arming recheck the worker stop signal after
waiting for the lane. Accepted result commits remain drainable after stop;
shutdown does not cancel an execution already admitted to the local router.

<!-- [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] -->
Worker shutdown and terminal preparation now share the same process-lifetime
stop gate. Router entry checks that gate after queued crypto preparation and
before authorizing local execution. A stop observed there proves zero dispatch
and permits only restoration of the exact signed Lease, never a replacement
payload. Once entry is authorized, a later stop drains the operation and its
result persistence instead of cancelling effects. A stopped adapter is never
reopened; restart recovery still treats an un-restored Armed row as ambiguous.

<!-- [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] -->
The source journal also serializes its runtime DB operations through one
shared lane. Source admission stays bounded while waiting; cancellation after
blocking work starts cannot release either permit ahead of its durability
fence. Fresh preparation/arming rechecks stop after the wait, while accepted
ambiguity/result completion writes remain drainable. Database time is sampled
inside the serialized operation rather than before waiting behind other jobs.
If recording an observed uncertain POST outcome fails, the live source runtime
stops further work and returns the failure instead of silently leaving Armed
eligible for live replay. Corrupt or failed storage is not repaired or bypassed
by this handling; restart still uses the existing authenticated journal audits,
exact-request recovery and source-owned reply verification contracts.

<!-- [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] -->
Every observed source POST outcome is durably recorded before ACK verification
or evidence polling, including a valid relay custody ACK. The existing
`DispatchAmbiguous` phase means execution remains unresolved; it is evidence-only
and does not imply that custody failed. A later `pending` response or restart
cannot resend that POST. Exact same-relay replay is reserved for the unobserved
`Armed` crash window and still requires the original authority and deadline.
Direct internal dispatch shares typed submit/resume's per-route exclusion.
An unwind in lifecycle-owned source work closes the source runtime and API
admission gates, preserves the durable barrier, and makes drain report failure.
It does not claim zero dispatch, restore Prepared, or silently continue sending.
Neither a custody ACK nor this dispatch-observation marker is execution proof;
only authenticated evidence and source-owned reply opening establish completion.

<!-- [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] -->
Source POST-observation storage/clock failures and lifecycle-owned unwinds now
publish a sticky, source-blind runtime fault. Source installation binds its
exact stop flag to the API's existing pre-body gate, so intake closes even
before the server supervisor is scheduled. Installation cannot replace or
reopen a stopped owner. The existing server readiness gate rejects an already
failed source, and its post-READY shutdown select observes subsequent faults
alongside the recipient and other required tasks. It then uses the existing
stop-all-reverse-intake and ordered owner drain path; it does not abort accepted
source journal work or manufacture a successful result. Cancelling an observer
cannot erase the fault. Normal operator shutdown is not itself a source fault,
and a disabled source role does not produce a missing-owner failure.
These changes and their regression test source are uncompiled and untested;
they do not establish a deployed Phala channel or attestation result.

<!-- [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] -->
Lifecycle-owned source operations also publish sticky local faults when they
return Unavailable normally, not only when they unwind or fail inside a journal
transaction. The owner closes intake before delivering that error, even if
the HTTP waiter has disconnected. Evidence-verification clock failures retain
this classification instead of becoming remote-response ambiguity. Durable
phases are not rewritten by this fault publication; ordered completion drain
and authenticated existing-journal recovery remain required. A normally
returned local fault is not itself a child join failure, so a successful drain
does not clear its separately retained supervisor failure.
Journal capacity maps to existing Busy/HTTP 429 with Retry-After, while protocol
rejection, expiry, conflict, ambiguity and Pending remain nonfatal to the owner.
Full quota does not block exact accepted-route continuation. Regression source
covers signed API quota rejection without transport entry, full-slot custody
continuation/restart, and normal local faults with live or cancelled waiters.
These cases are authored and statically inspected, not executed; no OS clock,
independent-host TLS, deployed channel or end-to-end attestation claim follows.

<!-- [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex] -->
After the post-READY shutdown select, source-only and combined-role servers
recheck sticky reverse-owner faults even when normal shutdown won the select.
They recheck again after the existing recipient, source and queue drain order,
so a late accepted completion fault cannot be reported as a healthy exit merely
because its lifetime drain succeeded. An already selected required-task fault
retains priority; disabled roles and healthy stopped owners remain inert.
The outer startup owner also closes both reverse intake gates and attempts
both owner drains before returning an early startup error, retaining that error
rather than masking it with a later drain error. A selected critical runtime
fault still reaches generic task shutdown after reverse drains report failure.
This is returned-error reconciliation; the retained unwind owner below covers
the separate dependency-lifetime requirement.
Regression source covers actual journal fence failure after stop/caller
cancellation, retained recipient worker faults and healthy stop controls.
These changes are implemented but uncompiled and untested; no trial restart,
deployment, independent-host TLS or end-to-end Phala attestation was performed.

<!-- [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] -->
When any reverse role is enabled, the server owns its generic task registry and
relay queue outside the fallible startup/runtime future. A Rust unwind becomes
a fixed local runtime failure without forwarding its payload. Returned errors
and caught unwinds both close all reverse intake gates, attempt recipient and
source drains, and await the queue's shared accepted-work permits before global
shutdown and bounded generic task joins. Accepted operations retain their
service/router Arcs throughout; the registry is not aborted by the inner unwind.
The original server failure remains the result even if later drains fail.
Normal shutdown keeps its existing order; disabled reverse roles keep the
previous startup abort policy. No failed owner is restarted or reused.
Regression source exercises retained generic-task lifetime across a real source
journal completion, actual recipient worker drain, and relay queue permit drain,
including cancellation/retry of a drain waiter. This is not protection against
panic=abort, process kill, runtime teardown or malicious same-process memory
corruption. The panic hook may still report its configured local diagnostic;
only the returned process-health error is sanitized here.
These changes and regression source are uncompiled and untested. No builds,
runtime probes, deployment, restarts or new endpoints were performed.

<!-- [PHALA-READY-PUBLICATION 2026-10-07 by Codex] -->
The final process READY decision rechecks required-task supervision, source
ownership/intake, recipient ownership/worker completion and global shutdown
after asynchronous readiness waits, with no await before sending READY.
A normally stopped source or cancelled recipient cannot publish READY and
still follows the existing ordered drain path. Sticky source faults, missing
required owners and unexpected recipient exits remain failures; disabled roles
are not missing required owners. A previous successful recipient wait is not
reused as a worker-health permit. This is a local observation immediately
before publication, not an atomic guarantee against later faults. Post-READY
supervision remains required. Startup does not require a current discovery
grant, and READY is not proof of task acceptance, execution, VPN reachability
or Phala attestation. This gate and its regression source are implemented but
uncompiled and untested; no trial image was rebuilt or deployed.

<!-- [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] -->
Source task POSTs, historical exact POST retries and signed evidence queries
carry the owner's actual stop flag and immutable time bounds into the HTTP
transport. After constructing headers/body, transport rechecks those bounds
immediately before HTTP entry. The recipient carrier shares the exact worker/
terminal stop flag and rechecks after DNS and at the same final send boundary.
Stop or expiry before entry defers the recipient's exact persisted Claim/Result;
it never deletes the journal row or creates a replacement Claim. Already-entered
HTTP work is not cancelled by this gate and retains its bounded response wait.
Recipient pre-entry clock/frame faults propagate to the existing worker
supervision/drain path instead of becoming routine transport retries. Its
response wait is also clipped to the final immutable-frame time sample.

Only a locally produced pre-entry no-send proof may restore a newly Armed
source task to Prepared. A clock fault instead stops the source owner and
retains its durable barrier. A no-send proof for a historical retry cannot
prove what happened before the original crash, so it never restores Prepared;
normal stop preserves Armed for authenticated restart recovery. A query that
never entered HTTP cannot change custody or authorize another task POST.
After HTTP entry, timeouts, lost responses and unverifiable acceptance remain
ambiguous and durably disable new task POST attempts as before. HTTP status,
custody ACKs and remote error text cannot manufacture a no-send proof.
These gates and regression sources are implemented but uncompiled and untested;
they do not prove deployed delivery, DNS/TLS behavior or Phala attestation.

<!-- [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] -->
The final reverse-onion HTTP entry gate now revalidates the originally selected
signed authority as well as stop/time bounds. Source task POSTs require their
exact R/P/grant; evidence queries require the selected current descriptor for
the same historically authorized relay/origin and do not require a new P grant.
Recipient Claim/Result attempts retain their pre-DNS snapshot and defer rather
than silently adopting a changed descriptor or grant. Appraisal expiry or a
stricter local appraisal policy also rejects entry even when the descriptor is
still time-valid. Appraisal publication and policy/cache changes share the
signed-authority epoch. The final clock is sampled after acquiring that epoch;
entry uses a nonblocking read so a busy writer defers recipient work or returns
source Busy without starting HTTP. A historical Armed retry remains Armed in
that case. No read guard is held through HTTP or DNS awaits. This is a final admission
point, not a guarantee against revocation after the gate has returned. Existing
first-attempt versus historical no-send recovery rules remain unchanged.
Regression source covers policy tightening, appraisal renewal, signed rotation
and historical evidence after grant expiry. This batch is uncompiled/untested;
it does not establish deployed application attestation or proxy blindness.

<!-- [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex] -->
<!-- [PHALA-AUTHORITY-DESCRIPTOR-EPOCH 2026-10-08 by Codex] -->
Live reverse-onion roles retain an unchanged, exact signed self-descriptor
within one running gossip owner's memory until half of its original lifetime.
This prevents a timestamp/sequence-only heartbeat from invalidating a P grant
before it can be forwarded or used. Every round still constructs the current
readiness/KEM candidate first: any changed signed field, configured lifetime,
or invalid/expired previous descriptor forces replacement immediately. The
original signature and expiry are retained verbatim, never extended. New
owners/processes start without a retained generation; this does not restore
onion secrets or trust from disk. Ordinary discovery and recovery-only roles
keep their existing descriptor renewal behavior.

Actual descriptor renewal still invalidates the old exact P authorization;
fresh grants must be issued and delivered through the existing authenticated
gossip path. This is not a grace-period bypass or uninterrupted-availability
guarantee: missing authority, network outages and insufficient forwarding
cadence keep fresh work paused while historical custody/evidence recovery
remains separate. Selector and PeerStore regression source was authored but
not compiled or executed; no deployment was performed.

<!-- [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] -->
P now retransmits an ACK-confirmed, still-valid exact grant instead of signing
a replacement on every unchanged heartbeat. Selection uses one nonblocking
authority epoch and a new post-lock clock sample; descriptor rotation, grant
expiry or missing confirmed authority still requires a new signature. This
selection timestamp also becomes the final send gate's clock floor, even when
the reused grant's original issue time is older; a later rollback still defers
the attempt. This
does not cache a successful delivery before ACK, extend any validity interval,
or authorize public-recipient/grant-purpose fallback. Lost ACKs may still cause
a later fresh issuance; gossip acceptance is not execution completion.

R rotates across up to `min(discovery.gossip_concurrency_limit, 8)` configured
source pins each round, rather than only one. All source pins must parse under
the same bounded, duplicate-rejecting configuration parser; malformed input
never becomes a partially accepted allowlist. Failed/offline targets do not
hold the cursor. Each selected source still needs its current signed endpoint,
feature support, successful legacy exchange and the final exact bundle gate.
Private and ordinary requests share the existing bounded gossip executor; this
adds neither an unbounded fanout nor another concurrency pool. A 64-source
list with the default concurrency visits each pin in eight scheduled rounds,
not 64; timeouts/backoff and grant/descriptor expiry can still pause fresh work.
Recovery-only queues select no fresh-authority targets. Parser, scheduling and
grant-coalescing regression source is authored but not executed; builds,
runtime verification and deployment remain deferred.

<!-- [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex] -->
Private-role startup and gossip sign current runtime descriptors above the
retained authenticated local sequence when the bytes changed, including a
same-second restart or capability withdrawal. The cache supplies only a counter:
it cannot restore an old ephemeral KEM, readiness feature, grant or running
owner epoch. Unchanged heartbeats may retain exact bytes only from the current
owner while the cache agrees. A conflicting newer cache entry forces current
runtime bytes to be rebuilt rather than reviving the owner's older surface.
Clock rollback below an authenticated local issue time, malformed candidates
and sequence exhaustion reject publication without wrapping or future-dating.
The normal PeerStore sequence/conflict checks remain intact; a concurrent newer
import can still reject publication and the next round must retry. This is a
floor over retained cache/owner evidence, not a permanent counter guarantee when
all such evidence is absent. Regression source is authored but unexecuted;
no tests, builds, network probes or deployment were performed.

<!-- [PHALA-SELF-CACHE-RESTART 2026-10-08 by Codex] -->
Additional authored regression source round-trips the local snapshot JSON into
fresh PeerStores after expiry cleanup. It checks that expired signed self bytes
remain local counter evidence, never valid public discovery, and that rebuilding
uses a new counter, current KEM and current readiness. Corrupt imports and older
cache replay must not replace that state. This is an in-memory serialization and
store-recovery case, not disk crash durability, cache-file atomicity, or an
executed process restart. It has not been compiled or run.

<!-- [PHALA-POLICY-HEARTBEAT-EPOCH 2026-10-08 by Codex] -->
An Anonymous Mailbox work-policy token also signs descriptor sequence and TTL.
The private running-owner epoch comparison now authenticates both nested policies
and compares their target and work bits, substituting only the exact canonical
token in an internal comparison. Otherwise unchanged heartbeats may retain the
original signed descriptor, so the same P grant still names the exact R epoch.
The comparison copy is never signed, exported or installed in PeerStore.
All other metadata, KEM, capabilities and policy fields still require equality;
changed work bits, missing/invalid/duplicate policies, runtime withdrawal, half-TTL
renewal and cache disagreement cannot restore the older epoch. A changed valid
descriptor still invalidates its old grant through the existing commitment check.
Regression source covers retained grants and published mailbox policy pins; it
is authored but not executed. No compilation, tests or deployment were performed.

<!-- [PHALA-ROTATED-CUSTODY-RECOVERY 2026-10-08 by Codex] -->
Cross-renewal regression source covers actual signed R/P replacements with
unchanged identities and relay origin. Existing Claim replay and Result
completion must keep the exact durable frames while fresh polling stays closed
until the replacement grant arrives. A newly valid grant cannot re-lease an
Armed row. Recipient Result retry selects the current same-origin R descriptor
without requiring a fresh Pull grant; a descriptor selected before renewal
still fails its final send gate.

Source journal reopen must recover the authenticated historical R/P/grant
bytes, not replace them with the current bundle. Both restarted Armed and
DispatchAmbiguous paths are exercised through the identity-pinned runtime:
replacement authority may admit new work but only evidence queries, not another
task POST, are allowed for the old route. These are authored assertions, not
execution evidence. No tests, builds, network verification or deployment were
run for this batch; hosted E2EE and full closed-loop acceptance remain unproven.

Private authority renewal now samples a new clock after legacy synchronization.
Each authority POST builds its request before a nonblocking final check of the
original signed target, exact current R/P pair, grant validity and purpose, and
canonical gossip origin. P may send a new locally signed grant only to its
configured R; R may forward only the current cached grant to a configured source
identity. A changed target, descriptor pair or newer cached grant defers the
attempt rather than disclosing the old private bundle to a replacement route.
An ACK-time cache write also uses a nonblocking authority epoch and samples time
inside it; busy, expired, rolled-back or rotated authority is not remembered.
These optional authority exchanges do not replace or suppress the normal
legacy/bootstrap synchronization, and they do not require an already-existing
route appraisal to bootstrap signed descriptors. No new wire version, API
endpoint or persistent secret is introduced. Tests for these gates are authored,
not executed; actual renewal scheduling and deployment remain unverified.

<!-- [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] -->
Source and recipient HTTP attempts now retain an independent local clock floor
across final authority admission, response waiting and timeout cancellation.
These volatile observations neither renew authority nor change persisted or
wire timestamps. A backward/unavailable local sample is sticky for that attempt,
even if the carrier returns an ambiguous outcome; normal forward expiry and
remote transport ambiguity remain separate and cannot authorize a new task.
The source rechecks this floor before no-send restoration or POST-observation
mutation. The recipient rechecks after dropping a timed-out future and carries
HTTP completion's sample through poll-response DB admission and Result echo
verification, which also retains the immutable Result retry deadline.
Regression source covers retained samples, sticky failure after timeout, source
API intake closure without a new Prepared attempt, and recipient DB rejection
without replacing exact poll custody. Tests are authored, not executed; no
independent-host DNS/TLS, deployed reverse delivery, proxy blindness or complete
Phala application attestation is established by these source changes.

<!-- [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] -->
Source local clock observations now enforce their trusted prior sample rather
than ignoring it or substituting a fallback. First dispatch carries the actual
Armed SQL observation into post-commit validation and HTTP admission. Historical
Armed replay refreshes under authority/SQL ownership and carries that checked
read time through DNS, including failed resolution. A local clock fault retains
the Armed barrier and closes its owner; it is not ordinary forward expiry or
permission to restore a new Prepared attempt.
Duplicate submission, authenticated metadata/authority recovery and durable POST
observation completion also carry their latest local sample into the next step.
Authority validation samples after acquiring its epoch, not before lock wait.
Known zero-send rejection and delayed custody ACK verification preserve that
floor too; custody acknowledgement is still not execution completion.
Evidence collection advances one floor through signing, DNS, HTTP response,
each Claim/Lease/Result verification (including Pending/malformed evidence),
combined chain verification, journal-lane/SQL admission and local reply opening.
The floor comes only from local process/DB observations, not a remote request
timestamp. No on-disk field, wire format, lease or evidence deadline changes.
Regression source checks monotonic floors versus local failures, SQL sample
propagation, rejection/restart with exact Armed bytes, actual lifecycle failure
after caller cancellation, required-server readiness priority and private API
pre-body closure without transport entry. Fresh, Armed-duplicate and Verified
admission paths also assert their returned time floor without HTTP entry.
These tests are authored but not run;
they do not emulate a live OS clock change, independent-host DNS/TLS, deployed
Phala delivery, proxy blindness or end-to-end TEE attestation.

<!-- [PHALA-BUILD-CONTEXT-ALLOWLIST 2026-10-08 by Codex] -->
The workspace-root Docker context is now default-deny. Only the root Cargo
manifest/lock/toolchain, each declared public crate's manifest and `src` tree,
and the one baked Phala peer example configuration are admitted. Unknown root
directories, operator records, private client trees and local worktrees are not
implicit build inputs. Existing identity/database/cache exclusions still apply
after the source exceptions, including interrupted identity staging files.
Dockerfile and `.dockerignore` remain builder control inputs. This is not a
secret scanner: the admitted public source trees and example configuration
still require provenance review before an authorized build. A new workspace
crate, build script or production embedded asset requires explicit context
review rather than silently widening the exception set. The source-contract
regression checks exact exceptions, exclusion order, workspace coverage and
Dockerfile COPY boundaries; neither it nor a real Docker context/build has been
executed in this development phase.

<!-- [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] -->
Reverse-enabled server shutdown requests and cancellation of the public
`Server::run` waiter now use a separate sticky stop carrier. The retained
supervisor observes it during recipient readiness and normal operation, closes
reverse intake, and drains accepted source/recipient/queue work before publishing
the global dependency stop flag and broadcast. A cancelled observer cannot lose
the request; an already known required failure still outranks normal stop.
Legacy servers without reverse roles retain immediate programmatic shutdown.
This protects cooperative cancellation and Rust unwinding, not process kill,
panic-abort or loss of the Tokio runtime. Regression source exercises the real
waiter guard and queue-permit drain, pre-start stop, required-failure precedence,
and all three individual reverse roles. It has not been compiled or executed.

<!-- [PHALA-SOURCE-JOURNAL-FAULT 2026-10-07 by Codex] -->
The source journal now owns the shared live intake-stop flag and sticky failure
signal adopted by both seeded and identity-only runtimes. A corrupt/unavailable
transaction, clock rollback or poisoned mutex closes intake and publishes the
fault directly at the DB boundary, including evidence reads and local reply
opening, even if an async caller has already disconnected. API pre-body gating
and required-source supervision observe those exact signals without an extra
watcher task. Capacity, busy, expiry and protocol rejection do not poison the
owner. Normal stop does not publish failure or close the DB completion lane.
A stopped journal cannot be repackaged into a fresh runtime; restart must reopen
and audit the encrypted journal. Existing Armed/Opening/Verified recovery rules,
descriptor commitments and wire/schema versions are unchanged. This wiring and
its regression source are implemented but uncompiled and untested.

<!-- [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] -->
<!-- [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] -->
Recipient carrier preflight keeps local clock failure/observed rollback
separate from missing signed authority, expired frames, DNS failure and route
rotation. The latter remain zero-send deferral; clock faults propagate through
the real Claim/Result worker calls as local Unavailable. Time is checked before
route lookup, immediately before DNS and after it (including failed resolution).
The last checked sample travels with that exact route to final HTTP entry or
fresh Claim admission after the shared journal-lane wait. A later sample cannot
roll back below a newer DNS sample merely because it remains above the first
sample. No failed clock sample becomes route absence or a newly persisted Claim.
The required worker now publishes sticky, source-blind failure before awaiting
accepted terminal completion drain. Server readiness and post-READY supervision
observe that signal independently from task exit, close other reverse intake,
and retain the same ordered drain path without aborting journal/terminal work.
Known faults outrank a racing normal cancellation; normal operator stop alone
does not publish failure. Dropping or replacing an observer cannot clear it.
Regression source covers checked clock floors versus missing routes, failure
visibility with an owned task still blocked, observer replacement, cancellation
priority, actual existing-only worker startup rejection and normal recovery
startup/stop, and fresh Claim clock-floor rejection through the real journal lane
and existing-only restart audit with a healthy persistence control. These are
unexecuted development tests, not live DNS/TLS, OS-clock,
full process-startup, deployed message delivery or TEE-attestation acceptance.

Recipient terminal work likewise publishes a sticky local-owner fault if an
owned operation unwinds, its post-router crypto task fails to join, or exact
Result persistence reports a corrupt, unavailable or uncertain DB owner. This
closes the same intake flag used by the worker
and terminal entry gate, including after the original caller has timed out or
disconnected. Idle polling wakes on the fault; active DB/router/network effects
are still awaited rather than cancelled. The required recipient worker then
fails through its existing supervision and ordered drain path. Repeated adapter
drain reports the fault, while normal stop and ordinary unverified-response
ambiguity do not manufacture an owner fault or a zero-dispatch proof. Per-job
expiry, rejection and capacity/backpressure also remain non-reexecutable
uncertainty rather than becoming a recipient-process shutdown trigger.
A failed post-commit fence can leave exact Result bytes committed but unpublished.
The live journal stays poisoned; only the normal authenticated restart audit may
recover those exact bytes. Missing Result evidence remains non-reexecutable
ambiguity. No replacement task, new route, or persisted ephemeral key is created.
The implementation and regression test source remain uncompiled and untested.

<!-- [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] -->
The existing reverse-onion Claim, Result and SourceQuery routes acquire the
same bounded queue permit as ciphertext ingress before buffering a POST body.
The private source Pull route likewise acquires its source admission permit
before buffering. Both readers enforce their existing route-specific byte caps
and a ten-second body-read deadline: oversized bodies remain HTTP `413`, and
an incomplete read returns HTTP `408` without starting a journal/queue effect.
Unavailable admission rejects before polling the body. Unsupported methods
retain the existing `405` response without body buffering. Source responses,
including these early errors, retain the private no-store policy.
This deadline applies only to pre-effect input collection, not execution or
durability completion. Accepted blocking queue work retains its permit after
HTTP cancellation; stopping intake drains that work and waits at most the
body-read window for a still-incomplete admitted body. A source stop observed
after buffering rejects before JSON/authentication work or new dispatch.

<!-- [PHALA-QUEUE-EXPLICIT-OFF 2026-10-06 by Codex] -->
An explicit `AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED=false` disables the
queue even when the mounted TOML has `recovery_only = true`. Supplying
`AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY=true` at the same time is a
configuration conflict and is rejected; remove that override when disabling
the queue.

The renderer refuses tag-only image references. It mounts the dstack guest
socket only in `--public-peer` mode, where the quote API contract is enabled;
it does not accept a Tappd socket override. The current Phala Cloud CLI accepts
Compose and protected env files with `-c` and `-e`; `--compose`,
`--no-public-logs`, and
`--no-public-sysinfo` are not the documented deployment interface. Keep the
exact rendered Compose and env-file versions used for attestation review.
Supply an operator-owned env file with the values referenced by the Compose
file, including the pinned image digest and role-specific settings.

<!-- [PHALA-CLI-DEPLOY-SYNTAX 2026-10-06 by Codex] -->

```sh
phala deploy -n aeronyx-peer \
  -c /Volumes/disk/compose.phala.peer.locked.yaml \
  -e /Volumes/disk/aeronyx-phala.env --wait
```

Set `public_logs = false` and `public_sysinfo = false` in the project
`phala.toml` before deployment; do not rely on undocumented CLI flags for
privacy settings. Keep credentials out of the Compose file, `phala.toml`, and
checked-in environment examples. The CLI encrypts values supplied through
`-e` for the CVM deployment.

The assigned HTTPS service endpoint is not known until the CVM is created. On
the first `--public-peer` deployment, the mapped public API starts without a
dstack socket or quote route, and the descriptor does not advertise
`PhalaNodeAttestationV1`. Without `--public-peer`, neither app ingress nor the
dstack quote socket is present. After Phala shows the actual `8422` endpoint,
set `AERONYX_DISCOVERY_PUBLIC_ENDPOINT` in the protected env file, render again,
review, and deploy the updated locked Compose file. Only this second render
mounts the guest socket; Rust then validates and signs the assigned HTTPS
origin into the descriptor. Empty values remain unconfigured, never a guessed
placeholder URL.

The container mounts only the selected guest socket for fresh quote retrieval.
The API uses guest-agent v1 only in this profile; a missing v1 route is an
attestation failure, not a reason to downgrade. The quote remains opaque
evidence: callers must independently verify report-data binding, app
identity/compose, and TCB. CLI deployment or a healthy CVM does not by itself
establish application-level or end-to-end verification.

<!-- [PHALA-PEER-INGRESS-OPT-IN 2026-10-06 by Codex] -->

The peer HTTP API listens inside the container on `8422`, but the default
Compose template contains no published port. Phala generates externally
reachable app endpoints from Compose port mappings, so a peer endpoint is
created only when the renderer is explicitly invoked with `--public-peer`;
`--private-recipient` separately materializes its endpoint-free worker. In a
private-recipient-only render the public-peer service remains profile-gated
and does not start; combine both flags to start both identities in one app.
With neither flag, the existing non-ingress peer default is preserved. The
locked Compose passed to Phala must be the exact renderer output. Do not use
`--public-peer` for a trial without explicit authorization and a reviewed
ingress policy. The private-recipient service never publishes a host port.
The VPN UDP listener is loopback-only and Compose publishes no VPN UDP port,
so this is **not** a client-reachable VPN node. Gossip is configured, but
without ordinary seed peers or an operator-provided seed it has no configured
bootstrap target.

### Phala And Private-Recipient Role Boundary

<!-- [PHALA-REVERSE-ONION-RECIPIENT 2026-10-06 by Codex] -->

The public attestation identity and private onion-recipient identity cannot
share one signed descriptor: the recipient descriptor must have no public
endpoint. The Compose file offers an opt-in `private-recipient` service with a
separate persistent volume/key. A private-recipient-only render starts only
that service; combine `--public-peer` and `--private-recipient` when both
identities should run in one app.
<!-- [PHALA-PRIVATE-RECIPIENT-NO-VPN 2026-10-06 by Codex] -->
The private process is an outbound task recipient only. Rust rejects
`vpn.enabled=true` for this role, so configuration drift cannot turn it into a
VPN/TUN data-plane node.

<!-- [PHALA-PRIVATE-RECIPIENT-SERVICE-ISOLATION 2026-10-06 by Codex] -->
Keep `memchain.mode = "off"` for this identity. Rust rejects Local, P2P, and
SaaS MemChain runtimes here; the separately enabled ChatRelay and Blind Vault
stores provide the required durable ciphertext custody without opening the
MemChain API or its unrelated workers.

<!-- [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] -->
Keep `management.enabled = false` for this identity: its CMS client and
public-IP discovery probes use destinations outside the pinned relay. Rust
rejects management-enabled recipient configurations and the public-IP resolver
has a second no-network guard for this role. The Compose bridge separates
container ingress but is not a destination allowlist or host firewall.

For first deployment, initialize the public-peer and private-recipient keys
separately in their respective persistent volumes using the unmanaged-profile
`pubkey` command described above, then set the two public-ID pins before
starting the services. The command never prints private key material. On
subsequent restarts it returns the same public identity; replacing either
volume creates a different identity and requires new pins and fresh gossip
authority.

When `management.enabled = false`, startup creates a stable local server
identity in the configured persistent key path if it is absent. This is for
the unmanaged Phala profiles only; CMS-managed nodes still require their
registration record and registered key. Keep the public peer and private
recipient state volumes separate and persistent across restarts.

```text
AERONYX_REVERSE_ONION_RELAY_NODE_ID=<64-hex-character adjacent relay identity>
AERONYX_REVERSE_ONION_RELAY_ENDPOINT=https://<adjacent-relay-origin>
```

<!-- [PHALA-RECIPIENT-RECOVERY-OVERRIDE 2026-10-06 by Codex] -->
For incident recovery, set `AERONYX_REVERSE_ONION_RECOVERY_ONLY=true` in the
protected Phala app environment and redeploy the same pinned image. The worker
then requires its existing nonempty recipient journal, resends only exact
durable frames, and refuses fresh Claims. Leave the variable unset to preserve
the image's TOML setting. Set it to `false` only after reviewing recovery
state and explicitly authorizing new work; invalid values or use without the
private-recipient role fail startup. This switch does not recover a missing or
corrupt journal and does not establish that the recipient key is TEE-bound.

<!-- [PHALA-EXISTING-CUSTODY-OPEN 2026-10-07 by Codex] -->
Recovery-only queue, source and recipient startup opens the existing nonempty
primary descriptor without create flags, directory creation or permission
repair. That descriptor is retained for the normal lock and durability audits;
missing custody cannot fall through to live empty-database bootstrap. Existing
schema migration and exact-frame recovery rules remain unchanged. This is not
a defense against a hostile process with the same OS user rewriting storage.

<!-- [PHALA-OWNED-RECOVERY-SCHEMA 2026-10-07 by Codex] -->
A nonempty SQLite file is not sufficient recovery evidence: the queue requires
its existing ownership metadata and never creates a new custody schema in
recovery mode. Owned legacy queue migrations remain enabled. Recipient/source
startup also checks the locked primary's role/version header before writable
SQLite recovery; recipient v1 and v2 remain supported. Legacy queues use their
SQL ownership tag, not these role header markers, so their schema audit still
follows SQLite rollback recovery. None of these checks authenticates disk
freshness against a hostile same-user rollback.

The peer process retains `AERONYX_DISCOVERY_PUBLIC_ENDPOINT`; Compose explicitly
clears it for the private service. It also clears
`AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH`, which disables the quote API
for that process and removes its access to the dstack socket. The recipient
service publishes no host port and explicitly clears
`AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR`; the server also rejects a
private-recipient configuration that retains an inbound peer API listener.
Compose places it on a separate bridge with outbound connectivity, not the
public peer's container network. This prevents direct container routing from
the peer while allowing the recipient to resolve and poll its pinned relay
over HTTPS. The worker retains its own state volume across restarts and
bootstraps gossip to the identity-pinned relay. Its descriptor stays
endpoint-free and is signed with `public_discovery = false`; Compose sets
`AERONYX_DISCOVERY_PUBLIC_DISCOVERY=false`, and startup rejects the private
recipient role if that visibility flag remains true. It enables only the local
ChatRelay and Blind Vault stores; it
does not enable the Blind Vault public API, model inference, or a VPN UDP
listener. Relays hold ciphertext only and must not receive plaintext. Do not
copy or merge the two volumes: their separate server keys are the identity
boundary.

<!-- [PHALA-RECIPIENT-ATTESTATION-BOUNDARY 2026-10-06 by Codex] -->
The recipient-bound quote commits the public peer key, recipient key, nonce,
and canonical signed Pull grant digest into report data. It is not a quote issued by
the recipient process and does not prove the recipient private key is held
inside the TEE. Independent appraisal must confirm the accepted Phala app and
compose measurement, and rely on the audited recipient worker implementation.
Without that appraisal, the recipient's task authority comes only from signed
R/P discovery material and the P-signed purpose grant, not an attested
execution claim.

<!-- [REVERSE-ONION-METADATA-BOUNDARY 2026-10-06 by Codex] -->
“Ciphertext only” describes payload access, not anonymity. In this explicitly
supported direct-source topology, the adjacent relay learns the authenticated
source-node ID, random route ID, private recipient ID, request/result timing
and sizes, plus the ciphertext. The source ID is retained with the route so a
recipient that already knows the route ID cannot query the relay's signed
source-evidence API as if it were the source. This does not hide the source
node from its adjacent relay or provide traffic-analysis resistance; deployers
must not describe this profile as anonymous onion transport.

The peer profile caps ChatRelay at 2,048 messages / 64 MiB, 128 blobs / 256
MiB, and 5 MiB per blob. Blind Vault is capped at 128 leases, 64 MiB per
lease, and 2 GiB aggregate ciphertext, with a 1 GiB minimum-free-space
admission reserve. These are application-level admission limits, not
filesystem quotas; SQLite metadata/WAL and other node state also consume the
volume. Both stores remain disabled until recipient opt-in is explicit.

The relay identity and origin are bootstrap constraints, not task authority.
They are validated before startup; outgoing requests use the currently signed
relay descriptor from PeerStore with HTTPS and pinned public DNS resolution.
New Claims are not issued until discovery gossip supplies a current signed
relay/recipient descriptor pair and recipient-signed grant. Missing or expired
authority leaves the worker fail-closed; restart recovery uses only its
dedicated durable journal. Keep the volume attached across restarts. Do not set
`recovery_only` false to bypass stale authority checks, and do not place keys,
registration codes, or message plaintext in Compose variables.

<!-- [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] -->
The source caller is a separate opt-in on the authenticated VPN/MPI node, not
on the public Phala peer. Pin `relay_node_id`, `relay_endpoint`, and
`recipient_node_id` in `[reverse_onion.source]`. The HTTPS origin is a fixed
operator constraint, not a caller-selected destination. Startup may proceed
before discovery has delivered authority, but a new Pull remains rejected
until PeerStore contains the current signed R/P descriptor pair and P-signed
grant. Optional Base64 signed fields are cache seeds only. The source journal
retains the exact historical descriptors and grant for recovery; after restart,
an Armed task can only replay its exact bytes to the same pinned relay origin
under the existing immutable deadline, then verify the opaque signed evidence.
No startup seed or newer grant can rewrite an Armed task. Keep this caller
disabled unless the established VPN listener and authenticated owner checks
are enabled; there is no public/admin task endpoint.

<!-- [REVERSE-ONION-SOURCE-API-CONTRACT 2026-10-06 by Codex] -->
The authenticated caller uses `POST /api/chat/reverse-onion/source/pull` on
that VPN/MPI listener with JSON fields `version`, `wallet_b64`, `nonce_b64`,
`request_timestamp`, `pull`, `authorization_b64`, and `signature_b64`. The
body is capped at 16 KiB; byte fields use canonical standard Base64. `pull`
must be one valid Blind Vault Pull with `limit = 1`. `wallet_b64` must equal
the owner injected by MPI authentication. The Ed25519 signature is over
`SHA-256("AeroNyx-ReverseOnion-SourcePull-v1" || version_u8 || wallet[32] ||
nonce[16] || request_timestamp_be_u64 || canonical_pull_frame ||
canonical_authorization_bytes)`. The Pull frame is the existing canonical
`BlindVaultFrame::PullRequest`; authorization is either empty (resolve the
current signed grant from the authenticated discovery cache) or the exact
canonical P-signed grant. Do not sign JSON serialization or caller-supplied
relay/recipient/endpoints/deadlines; none are accepted by this route.

HTTP `200` with `state = "completed"` contains the canonical Base64 sealed
PullResponse frame after the server's source workflow completed. The caller
must still validate the frame and page signature against its pinned replica
authority before using records. HTTP `202` with `state = "pending"` and an
empty `response_frame_b64` means unresolved custody/execution, not success;
retry the same owner + nonce + Pull identity, refreshing only the signed
request timestamp/signature as needed. HTTP `409` means ambiguous; preserve
the same route and do not create a replacement task. HTTP `429` is bounded
admission pressure and includes `Retry-After`; `400`, `401`, and `503` are
rejection/authentication/unavailability, never delivery evidence. Responses
carry `Cache-Control: no-store`; the API never returns an HTTP-success claim
as a substitute for the signed PullResponse.

<!-- [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] -->
Source admission requires a nonzero Local or Remote owner from MPI before
JSON/body extraction. A missing owner, zero owner, or SaaS JWT owner is rejected
with a coarse private `401` by the inner source gate. This is defense in depth
for VPN-only composition, not a new SaaS capability.
MPI classifies this exact path as privacy-sensitive:
authentication does not provision per-owner storage/vector resources or update
owner activity. Other MPI/chat routes retain their existing behavior. Remote
Ed25519 authentication necessarily reads the signed body first, but its buffer
is capped at the same 16 KiB as source parsing and has the existing 10-second
pre-effect read deadline. A stalled read returns `408`; this deadline never
cancels accepted journal or terminal work. MPI also applies the source's
private/no-store headers to early authentication/body-read failures, before
the inner source router runs. No request identity, destination, or task
payload is included in the new rejection diagnostics.

<!-- [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] -->
Production VPN composition uses `build_mpi_router_with_reverse_onion_source`:
an exact-path HTTP slot is acquired outside Local-mode MPI authentication, so
even a remote request still awaiting body authentication consumes the existing
lifecycle's bounded capacity and is included in its stop/drain. Full/stopped
capacity returns private `429` plus `Retry-After: 1` before reading the body;
this resource rejection can precede owner authentication. The inner source
gate reuses a local, typed permit only when it belongs to the identical
lifecycle admission owner and intake is still open. It never reacquires a
second slot, and a foreign permit grants no capacity or owner authority.
SaaS JWT authentication does not read a body and continues to reject source
owners without allocating a source slot. Other paths bypass this outer gate.
Unsupported methods on the exact source path are also bounded because MPI
can otherwise authenticate their bodies before returning its method fallback.
Dropping a pre-effect HTTP future releases its shared slot. Stop/drain waits
for an already-admitted body read to finish, time out, or be cancelled; cancelling
the drain waiter does not erase that HTTP owner. Accepted journal/terminal
operations retain their separate durable, cancellation-surviving ownership.
Authored regression source covers one-slot reuse, foreign-owner rejection,
pre-auth saturation/cancellation, stopped intake, cancelled drain/retry, and
unaffected chat routes. No test or runtime verification has been run.

<!-- [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] -->
A successful source Pull now retains that same HTTP slot through response-body
delivery, not merely through response construction. Inner source and outer MPI
layers identify the same local response owner and wrap only once; small error
and Pending replies keep their existing immediate-release behavior. The body
enforces the existing protocol response-size bound, with a fixed 30-second
delivery deadline starting at handoff, not a renewed task or lease deadline.
It copies at most 16 KiB into each HTTP data chunk: a socket-held chunk cannot
retain the entire multi-MiB source buffer through a shared `Bytes` slice.
Polling, new source admission and lifecycle drain expire the same buffer owner.
During idle periods an unpolled body may remain allocated until one of those
events, but it retains its bounded slot; repeated requests cannot accumulate
large expired bodies outside that capacity. No background timer task is added.
Drain wakes when a new admitted response is handed off, retains its original
deadline after drain-waiter cancellation, and releases the large buffer even
if HTTP never polls it. Body drop, EOS, error or expiry releases the slot;
expiry reports a body error, not a successful empty response. HTTP can buffer
multiple copied chunks under its own flow/write controls; this change bounds
source-owned large buffers, not listener connection counts or transport write
buffers. No claim of a one-chunk bound inside the HTTP implementation is made.
Delivery failure does not erase a Verified journal row, change evidence/lease
deadlines or authorize re-execution. An authenticated retry recovers the same
durable result. This batch is implemented but unverified: test source was
authored, with no compilation, test execution, deployment or TEE/E2EE acceptance.

<!-- [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] -->
The public adjacent-hop Claim/Result/SourceQuery HTTP adapter now retains its
already-admitted queue slot through successful binary response delivery too.
Previously that slot ended with SQLite work, leaving slow HTTP consumers able
to retain ciphertext response buffers outside the configured queue concurrency
ceiling. Live ingress, the mounted API and shutdown now share both the existing
semaphore/stop bit and one weak response registry. Recovery-only APIs retain
their own default-closed owner; no fresh task or execution authority is added.
The VPN source API uses the same body implementation while retaining its
separate lifecycle and source-permit identity checks.

The original canonical response bytes, content type and HTTP status are
preserved. SourceQuery uses the core evidence response cap; Claim/Result use
the frame cap. A body owns its original slot until drop, EOS, error, or a fixed
30-second delivery deadline. Chunks are detached copies of at most 16 KiB, not
shared slices retaining the full response allocation. New public API requests
and ciphertext ingress reclaim expired unpolled bodies before acquiring
capacity; stop/drain drives the same expiry even when HTTP never polls.
Cancelling a drain waiter neither erases response ownership nor renews its
deadline. Accepted SQL retains its existing cancellation-surviving permit;
an already-admitted reply may be handed off after stop and is still drained.

Body expiry reports an HTTP body error, never a successful empty frame. It
does not delete durable custody, refresh a Claim/lease/grant, authorize another
onion POST, or turn transport delivery into execution completion. An exact
authenticated retry can recover the stored Lease/Result. Empty error replies
retain immediate-release behavior. This bounds application-owned response
buffers, not listener connection counts or HTTP/socket transport buffering.
During a completely idle period an unpolled body may remain allocated until
polling, admission or drain observes expiry, but its bounded slot stays held.
Regression source covers actual Lease HTTP replay, response-held saturation,
chunk equality, drop/expiry, cancelled drain/retry, shared ingress reclamation
and distinct evidence/frame caps; existing VPN source body cases remain.
Implemented but unverified: no compilation, test execution, service startup,
deployment or closed-loop TEE/E2EE acceptance was performed.

<!-- [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] -->
Live queue composition, discovery startup, grant-forwarding selection, and both
internal admission constructors now consume the same decoded identity policy:
one distinct relay/recipient pair and 1..=64 unique, valid Ed25519 source pins,
none equal to either route role. Hex case remains compatible; case aliases are
duplicates, not distinct identities. PeerStore installs the route pair and the
complete source set under one authority write epoch, after both pin tables pass
capacity preflight. Reapplying an existing complete policy is idempotent, even
at capacity; rejecting a new policy leaves both tables unchanged. Embedded
queue startup validates limits and the complete policy before opening SQL.
Signed-seed construction also obeys these identity, permit, private-recipient
policy, and protocol route-lifetime bounds. Pins remain operator bootstrap
constraints only: they create no descriptors, grants, appraisal, or execution
authority. Recovery-only replay and shared stop/permit/response-drain ownership
are unchanged; no new listener or endpoint is added. Source fixtures cover
atomic rejection, idempotence, hex aliases, cold bootstrap and both constructor
paths. These changes and fixtures are uncompiled and unexecuted; future
verification must include full startup/restart and authority-renewal coverage.

<!-- [PHALA-QUEUE-EFFECT-ADMISSION 2026-10-08 by Codex] -->
Queue capacity is now acquired before arming a fresh private-recipient route
effect. The earlier read-only queue lookup does not reserve the later enqueue
slot: another request or slow response can fill capacity between those steps.
Full/stopped capacity therefore rejects with the existing backpressure category
without arming fresh zero-write work; dropping that owner releases only its
exact unarmed replay reservation. An already-Armed/recovered route is never
downgraded or released by capacity failure. Once capacity is acquired, arming
still precedes cancellation-surviving enqueue work, and uncertainty after that
boundary retains its existing conservative custody/recovery semantics.
The same acquired permit is moved into the DB worker; an arming failure drops
it without leaking capacity. No caller is authorized to infer "unexecuted"
from an ambiguous network result or to try another node after an onion POST.
Authored local/durable replay cases distinguish no-write pressure from Armed
work, with successful arming as a positive control. A separate controlled-worker
HTTP case covers accepted late handoff after stop and cancellation before
handoff; it is not a SQLite execution or cryptographic acceptance test.
These additions remain uncompiled and unexecuted.

<!-- [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] -->
An empty HTTP `204` or an HTTPS status alone is never proof that a Claim had no
work. The relay returns HTTP `200` with a fixed 197-byte `AXRA` v1 receipt:
`magic[4] || version[1] || relay[32] || recipient[32] || claim_id[16] ||
claim_commitment[32] || issued_at_be[8] || expires_at_be[8] || signature[64]`.
The Ed25519 signature covers the canonical prefix `AeroNyx-ReverseOnion-NoWork-v1\0`
followed by all unsigned receipt fields. It binds the recipient, Claim ID,
exact Claim commitment, and a maximum 30-second validity window. The recipient
deletes its durable poll only after verifying that signature against the
configured adjacent relay identity and matching the exact journaled Claim.
Legacy empty responses, rejections, malformed receipts, and ambiguous network
outcomes preserve the durable Claim for recovery/retry. This receipt means
only “no queued item for this exact poll,” not message delivery or execution.
The relay's configured no-work marker retention must be at least 930 seconds
from poll admission, covering the 30-second Claim window plus the Claim's
existing envelope-plus-result evidence horizon (900 seconds after Claim
expiry). A lost receipt can then be reissued after restart; this does not
refresh the Claim or grant another lease.

This profile separation does not by itself prove that the private recipient
identity is inside an accepted Phala workload, nor that a task ran or its reply
is source-verified. The private-recipient process does not host a quote API;
the public peer can serve the grant-bound quote described above. That quote
binds the recipient identity and signed Pull grant to the public peer's Phala
measurement, but does not prove that the recipient private key is held inside
the TEE or that a task executed.
Neither the Compose profile nor descriptor proves closed-loop delivery. Those
require static review and focused runtime verification; this code-only
checkpoint has not run that verification.
The existing `compose.phala.yaml` loopback trial remains separate and unchanged;
do not use this peer profile on the restricted isolated trial CVM.

<!-- [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] -->
Onion KEM overlap is bounded to the current and one retired in-memory key.
The retired key's fixed grace deadline starts at actual rotation, including
when discovery runs late, rather than at the key's original creation time.
Discovery descriptor TTL is accepted only in `60..=85799` seconds; the TTL
plus 600 seconds (with a 3600-second floor) stays below the 86400-second
rotation period. Larger lifetimes are rejected, not silently shortened.
Startup rejects failed key initialization/rotation; a rejected gossip epoch
skips replacement descriptor issuance. Peeling rejects observations earlier
than the manager's last successful initialization/rotation observation.
Reads do not rotate keys or extend retirement deadlines. Restart still loses
all old ephemeral secrets: retention does not make old ciphertext decryptable
across process restart and is not durable-delivery or Phala custody evidence.
Delayed-rotation, lifetime-boundary, clock and restart regression source is
authored only; no tests, builds, deployment or independent-host KEM validation
have run for this change.

<!-- [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] -->
Journal schema ownership is not sufficient to admit a running identity.
Source initialization checks its exact owner and durable clock in the same
transaction before committing. Recipient v1-to-v2 migration also checks the
exact relay/recipient pair, canonical metadata types, durable clock, configured
quotas and every bounded signed row before committing. Rejected migration
rolls back the schema upgrade rather than leaving v2 behind after a failed
startup. Valid legacy Poll/Lease rows keep the existing origin-less recovery
semantics; no new origin, key, deadline or execution permission is synthesized.
The recipient's subsequent Armed-to-ambiguous restart barrier and durable
filesystem fence remain required before readiness. Source/recipient startup
regression source is authored only, not compiled or executed. These checks do
not prove rollback-resistant storage against a hostile same-UID operator or
an independently attested Phala application.

<!-- [PHALA-CLAIM-EXECUTION-CAP 2026-10-08 by Codex] -->
### Separate Route And Execution Horizons

The Claim HTTP producer derives a fresh Lease deadline from the selected
durable row and the queue owner's immutable execution cap. The core also
applies the signed envelope's freshness bound. A longer route must not cause
the producer to sign an overlong Lease that its own database then rejects.
Claim freshness remains independent: exact stored Claim-to-Lease replay does
not regenerate or extend the Lease after the Claim's 30-second window.
Result grace and route retention continue to use their existing bounded
contracts. Zero or overflowing clocks and expired route horizons reject.

The native shorter-Lease/exact-replay regression passes. The connected
private-pull transport test also passes its 18 combinations of source API,
recipient recovery, authority renewal, lost custody acknowledgement and
altered evidence. That test uses the real routers, journals and worker with
synthetic in-process HTTPS transport and ephemeral keys, not independent
hosts or application attestation. The corrected combined native regression
run passes all 416 cases, including the private-egress and maximum-page clock
fixtures. The full core protocol/cryptography suite passes 406 cases, including
the 23 reverse-delivery cases, and the executable's
standalone identity suite passes four cases. The frozen Linux source snapshot
also compiles successfully with the unwind-capable Phala profile. ELF
inspection confirms x86_64, a highest glibc symbol requirement of 2.34, and no
ONNX, Torch or CUDA dynamic dependency. The executable's SHA256 is
`21fd583fedecc8e8af4a20f74d7332e9aa942d3dd5d1ed9c641bd43d90071800`.
These are local verification results, not Linux execution or Phala deployment
evidence.

<!-- [PHALA-NATIVE-PROCESS-VERIFICATION 2026-10-08 by Codex] -->
The native macOS executable also builds successfully. An isolated unmanaged
process with VPN, DNS, MemChain and discovery disabled starts its existing
node HTTP API at an explicitly configured loopback address. Two successive
launches return HTTP 200 from `/api/vpn/health`, expose exactly that one
loopback IP listener, and exit successfully on SIGTERM. The second launch
retains the same public identity and existing regular 0600 identity inode;
verification never reads the identity file's private contents. This is actual
local process and restart evidence, not Linux TUN, public VPN, Phala or TEE
verification. Disabling VPN does not disable the required node HTTP API.

<!-- [PHALA-CORE-CONTRACT-REGRESSION 2026-10-08 by Codex] -->
The wider core run initially found three stale private-purpose test contracts.
Private Pull retains ChatRelay and the complete signed operation feature set,
but does not require public BlindVaultReplica admission. Removing that public
role changes the signed descriptor and therefore requires a new exact grant;
the previous descriptor-bound grant must still reject. Private lease admission
similarly uses its explicit terminal feature rather than the public storage
role. Corrected tests retain missing-role, missing-feature, wrong-purpose,
public-recipient and descriptor-substitution rejection. The full rerun passes
406/406 without changing production code or weakening wire validation.

<!-- [PHALA-ISOLATED-TUN-REGRESSION 2026-10-08 by Codex] -->
The Linux transport test module also includes an ignored, explicitly opted-in
TUN lifecycle check. It uses a kernel-allocated interface name, non-persistent
ownership, a synthetic /32 address, typed kernel address/flag observations and
the actual kernel MTU. It checks up/down/up and interface deletion on fd drop;
it opens no network socket and sends no traffic. Execution requires a reviewed
isolated trial network namespace, NET_ADMIN, `/dev/net/tun` and `iproute2`.
Its environment acknowledgement is not evidence of isolation. Keep the test
ignored in ordinary runs; do not run it on a production or host-network
namespace. This check has not yet executed on Linux or Phala.
The updated test module cross-compiles successfully into an x86_64 Linux ELF
with the same unwind-capable Phala profile. Compilation is not a passing
device test; retain the ignored status until the reviewed trial is available.

<!-- [PHALA-CONFIG-EXECUTED-REGRESSION 2026-10-08 by Codex] -->
The additional configuration run passes 122/122 tests, and a separately
selected, network-free authority/descriptor lifecycle run passes 13/13.
The initial configuration run had five fixture failures: private-recipient
fixtures retained legacy management/MemChain defaults, a graph-default
assertion contradicted its explicit switch, and an empty TOML document was
spelled as a JSON object. Only test modules were corrected; the complete
production prefixes still match the compiled Linux snapshot. Private-role
egress, missing authority, direct API and legacy compatibility rejection
checks remain enforced. By test-name deduplication, these runs add 108 cases
to the earlier 826, for 934 distinct passing related tests, not a whole-repo
suite. The authority tests use signed synthetic fixtures and controlled clocks;
they do not execute DNS/TLS, Linux TUN, Phala or hardware appraisal.

<!-- [PHALA-TUN-DEACTIVATION-ERROR 2026-10-08 by Codex] -->
Linux TUN deactivation now returns a failed or timed-out `ip link down`
operation to its caller instead of reporting success and clearing the local
state. The last confirmed state survives failure and retry; successful
deactivation still clears it. The lifecycle mutex, bounded kill-on-drop
command runner and non-persistent fd ownership remain unchanged. A regression
uses a private Unix socket as an async descriptor and an impossible interface
name, without opening `/dev/net/tun` or configuring any real interface.
This new Linux-specific regression is not yet executed. The earlier Linux
node artifact predates this production repair and must be replaced before
deployment. Server shutdown still relies on dropping non-persistent TUN
ownership; this transport API repair does not add a new shutdown caller.
The repaired source now cross-compiles successfully with the same Phala
profile, as does its updated Linux transport test executable. The replacement
node ELF has SHA256
`211911bd819fdfa4abc2869492019efe172398552622dddf46b2fdd0e0f1b62e`;
the earlier `21fd583f...` artifact is historical, not the deployment candidate.
Neither the new Linux regression nor the opt-in TUN test has been executed.

<!-- [PHALA-PEER-STORE-EXECUTED-REGRESSION 2026-10-08 by Codex] -->
The additional native PeerStore run passes 144/144 tests. Two initial fixture
failures were corrected without changing production validation: the unpinned
import rejection now precedes installation of the identity-pair pin, and the
same-sequence signed-content conflict changes capacity rather than corrupting
the exact SemVer protocol feature tokens. All existing rejection assertions
remain. The server test package alone uses `debug=0` to avoid expensive debug
symbol generation; test assertions and production profiles are unchanged.
By exact test-name deduplication this adds 132 cases to the earlier 934, for
1,066 distinct passing related native tests, not a whole-repository suite.
Compiler and linker warnings remain. Neither this run nor Linux compilation
establishes Linux execution, independent-host TLS, Phala deployment, VPN
reachability or hardware attestation. This fixture-only repair does not change
the compiled production Rust source or the existing Linux artifact.

<!-- [PHALA-GOSSIP-EXECUTED-REGRESSION 2026-10-08 by Codex] -->
Additional native runs pass discovery ingress 22/22, discovery gossip 52/52
and onion-candidate API selection 14/14. Exact-name deduplication adds 74
cases to the earlier 1,066, for 1,140 distinct passing related native tests.
The initial gossip run failed its bounded-fanout timing assertion because
the timer included HTTP-client and fixture setup. The timer now starts
immediately before the bounded exchange; its 450ms ceiling, 250ms peer
timeout, timeout classifications and exact call-count assertions remain.
No production code or limits changed. The separate three-hop probe test
that binds a wildcard listener was explicitly excluded from this host run;
it remains unexecuted here. The ingress and candidate API cases exercise
in-process routers, while gossip transport mocks use loopback listeners.
These results do not establish independent-host TLS, Linux execution,
Phala deployment or hardware attestation. The provider rejected the isolated
deployment with HTTP 400 and a terms-of-use message; the specific trigger
remains unconfirmed and no retry or alternate deployment was attempted.

<!-- [PHALA-SOURCE-HTTP-FAULTS-EXECUTED 2026-10-08 by Codex] -->
The native source-runtime suite passes 47/47 tests, including a new real
loopback-socket regression with 14 combinations across task POST and source
evidence transport. Complete and exactly-at-limit bodies remain readable;
declared overflow, chunked overflow, truncated bodies and stalled bodies stay
ambiguous after HTTP entry, never becoming a no-send proof. A 307 response
does not contact its redirect target or forward the authorization header.
The stalled peer retains its incomplete response until the client returns;
an independent harness deadline prevents peer disconnect from masquerading
as the production timeout. The test uses HTTP and clock-only admission only
inside cfg(test); production DNS pinning, TLS, authority gates and ceilings
are unchanged. Exact-name deduplication adds one case, for 1,141 passing
related native tests, not 14 additional tests or a whole-repository suite.
The existing connected private-pull regression also passes its 18 combinations
again on this rebuilt binary (one existing test, no additional distinct count).
Its routers, journals and workers are real; its HTTPS carrier is synthetic
and its ephemeral KEM state remains in one process, not separate hosts.
The production source prefix still matches the frozen Linux artifact context.
This remains local transport evidence, not independent-host TLS, Linux/Phala
execution, private-client E2EE or hardware attestation. Existing compiler and
linker warnings remain, and the provider deployment restriction is unresolved.

<!-- [PHALA-COPY-CONTEXT-CONTRACT 2026-10-08 by Codex] -->
The isolated-vpn Docker stage's literal COPY of
`deploy/node/server.phala.example.toml` now has a matching exact .dockerignore
exception. The previous deny-all context admitted only the peer template,
so the isolated stage could not access its required configuration. No parent
directory exception was added; identity, database, private-client and operator
state exclusions remain intact. Docker removes ignored inputs before sending
the context to its builder; see the official
[build-context documentation](https://docs.docker.com/build/concepts/context/#dockerignore-files).
The external static contract now checks both configuration COPY inputs against
the exact 15-entry allowlist derived from the five workspace crates and three
root build files. It reproduces the old missing-input failure and rejects
missing isolated configuration, broad deployment/crate exceptions and a late
identity-file exception, alongside the existing target/port/role checks.
This is a literal source-contract check, not execution of Docker's pattern
engine or a successful image build. No Rust source or binary changed; the
1,141 related native test count is unchanged. The existing frozen Linux
compilation snapshot and archive retain their original hashes and old context
rules. A later actual container build must refresh and verify its context;
the old archive is not evidence that the repaired Docker image was built.
No Docker engine or Phala deployment was executed for this repair.

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
