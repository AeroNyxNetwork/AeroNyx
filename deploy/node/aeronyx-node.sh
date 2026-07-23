#!/usr/bin/env bash
# ============================================
# File: deploy/node/aeronyx-node.sh
# ============================================
# Creation Reason:
# - Provide one operator-facing command for AeroNyx privacy protocol node
#   install, upgrade, health checks, status, logs, and network maintenance.
# - Keep existing install.sh, upgrade.sh, and healthcheck.sh as stable
#   lower-level building blocks while reducing operator confusion.
#
# Modification Reason:
# - Add a guarded OnionMiddle config helper so operators can explicitly enable
#   or disable no-exit middle-hop advertisement for future two-hop encrypted
#   relay paths with backup, validation, active-session warning, and optional
#   restart.
# - Add a guarded ChatRelay config helper so operators and AI assistants can
#   enable or disable blind relay advertisement with backup, validation,
#   active-session warning, and optional restart.
# - Add a guarded staged-binary promotion helper for dirty/diverged production
#   nodes where Git cannot be pulled safely. This supports drain-aware binary
#   replacement with hash verification, config validation, backup, restart, and
#   post-restart health checks.
# - Harden staged-binary promotion with a bounded cold-start health wait,
#   mapped-running-binary backup, and atomic rollback when restart or health
#   verification fails.
# - Add a privacy-safe `relay-probe` command that sends one synthetic opaque
#   BlindRelay envelope through the local node to a discovered ChatRelay peer
#   and reports only aggregate counter deltas. The command also reports
#   two-hop readiness separately so operators do not mistake a single-hop
#   transport probe for a full multi-hop proof.
# - Add a privacy-safe `fleet-smoke` command for three-or-more public Rust
#   nodes. It verifies discovery status, signed snapshots, onion candidates,
#   terminal BlindRelay acceptance, and one-hop forwarding with synthetic opaque
#   blobs only. It is intended for release gates and post-restart checks.
# - Extend `fleet-smoke` with an optional `--two-hop` proof. The proof submits
#   outer+onward opaque BlindRelay frames through entry -> middle -> terminal
#   nodes and treats success as a privacy-safe multi-hop path proof.
# - Add `refresh-bootstrap` and `fleet-drift-check` so operators and AI
#   assistants can keep discovery seeds, signed bootstrap snapshots, and
#   deployed binary hashes consistent across a commercial node fleet.
# - Add `healthcheck` as a command alias and summarize local health/operator
#   JSON in `status` instead of printing full endpoint payloads. This keeps the
#   one-command operator view readable for humans and AI assistants while
#   preserving the explicit privacy boundary.
# - Add a read-only discovery readiness summary to `status` so operators can
#   see ChatRelay advertisement and peer quorum state without hand-curling the
#   public peer API.
# - Add `quickstart` as the commercial one-command operator workflow:
#   clone/pull is handled by nodeboard's generated shell command, then this
#   repository-local entrypoint runs plan, waits for explicit operator approval,
#   installs/registers/starts the node, prints status, and runs healthcheck.
# - Add a privacy-safe operator recommendation to `status` by reading the
#   existing healthcheck `operator_action` JSON. This gives ordinary node
#   operators and AI assistants a clear next step without scraping logs.
# - Keep preview/plan aligned with the commercial quick install path when a
#   registration code is present, so operators do not approve a different plan
#   from the command that nodeboard actually installs.
# - Show the local structured upgrade status file from `status` so operators and
#   AI assistants can see staged/failed upgrade state without scraping logs.
# - Initial production entrypoint for ordinary node operators. This script
#   delegates to existing deployment scripts instead of duplicating their
#   host-writing logic.
# - Clarify that this script is repository-local and is obtained from the
#   open-source AeroNyx Rust repository, not from the host operating system.
#
# Main Functionality:
# - Offers a guided interactive menu when no command is provided.
# - Provides command aliases for quickstart, plan, install, upgrade, health,
#   status, logs, doctor, relay route probing, binary promotion, and network
#   maintenance.
# - Passes common options such as --repo-dir, --branch, --registration-code,
#   --config, and --service to the appropriate lower-level script.
# - Keeps secret handling safe: registration codes are accepted as arguments or
#   environment variables and are not printed by the plan path.
#
# Dependencies:
# - deploy/node/install.sh
# - deploy/node/upgrade.sh
# - deploy/node/healthcheck.sh
# - systemctl and journalctl for status/log views on Linux/systemd hosts.
# - GitHub source repository:
#   https://github.com/AeroNyxNetwork/AeroNyx
#
# Main Logical Flow:
# 1. Parse the high-level command and common options.
# 2. If no command is provided, show an interactive operator menu.
# 3. Delegate host-changing operations to the existing battle-tested scripts.
# 4. Keep read-only views available without requiring operators to remember
#    multiple script names.
#
# Important Note for Next Developer:
# - Do not remove install.sh, upgrade.sh, or healthcheck.sh. They are the
#   compatibility layer used by docs, automation, and existing operators.
# - Keep this wrapper thin. Business logic belongs in the lower-level scripts
#   where it can be tested independently.
# - Never print registration codes, private keys, API secrets, wallet-level
#   data, DNS contents, destinations, packet payloads, chat plaintext, or
#   client public IPs.
# - This script targets Linux/systemd production nodes. macOS, iOS, Android,
#   and Windows remain client/development platforms, not production node hosts.
#
# Last Modified:
# v1.18.0-node-entrypoint - Made staged-binary promotion model-startup-aware and rollback-safe.
# v1.17.0-node-entrypoint - Added bootstrap snapshot refresh and fleet drift check commands.
# v1.16.0-node-entrypoint - Added fleet-smoke --two-hop multi-hop relay path proof.
# v1.15.0-node-entrypoint - Added fleet-smoke public mesh smoke test command.
# v1.14.0-node-entrypoint - Added gated relay-probe --two-hop live probe mode.
# v1.13.0-node-entrypoint - Updated relay-probe two-hop blocker wording after onward envelope support.
# v1.12.0-node-entrypoint - Clarified relay-probe scope and added two-hop readiness output.
# v1.11.0-node-entrypoint - Added guarded no-exit OnionMiddle enable/disable config helper.
# v1.10.0-node-entrypoint - Added healthcheck alias and compact status endpoint summaries.
# v1.9.0-node-entrypoint - Added synthetic BlindRelay route probe command.
# v1.8.0-node-entrypoint - Added guarded staged-binary promotion command.
# v1.7.0-node-entrypoint - Added guarded ChatRelay enable/disable config helper.
# v1.6.0-node-entrypoint - Show discovery ChatRelay and peer quorum readiness in status.
# v1.5.0-node-entrypoint - Added quickstart one-command install workflow.
# v1.4.0-node-entrypoint - Show healthcheck operator recommendation in status.
# v1.3.0-node-entrypoint - Align quick install preview with nodeboard install.
# v1.2.0-node-entrypoint - Show local upgrade-status.json in status output.
# v1.1.0-node-entrypoint - Documented GitHub origin and repository-local path.
# v1.0.0-node-entrypoint - Added single operator-facing AeroNyx node command.
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT="${SCRIPT_DIR}/install.sh"
UPGRADE_SCRIPT="${SCRIPT_DIR}/upgrade.sh"
HEALTHCHECK_SCRIPT="${SCRIPT_DIR}/healthcheck.sh"

DEFAULT_BRANCH="main"
DEFAULT_REPO_DIR="/opt/aeronyx/AeroNyx"
DEFAULT_CONFIG_FILE="/etc/aeronyx/server.toml"
DEFAULT_SERVICE_NAME="aeronyx-server"
UPGRADE_STATUS_FILE="/var/lib/aeronyx/upgrade-status.json"
PROMOTE_HEALTH_RETRIES="${AERONYX_PROMOTE_HEALTH_RETRIES:-90}"
PROMOTE_HEALTH_DELAY="${AERONYX_PROMOTE_HEALTH_DELAY:-2}"

COMMAND=""
REPO_DIR="${AERONYX_REPO_DIR:-${DEFAULT_REPO_DIR}}"
BRANCH="${AERONYX_BRANCH:-${DEFAULT_BRANCH}}"
CONFIG_FILE="${AERONYX_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
SERVICE_NAME="${AERONYX_SERVICE_NAME:-${DEFAULT_SERVICE_NAME}}"
REGISTRATION_CODE="${AERONYX_REGISTRATION_CODE:-}"
FORCE=0
NO_RESTART=0
DRY_RUN=0
JSON=0
JSON_ONLY=0
FOLLOW_LOGS=0
LINES=160
SET_VPN_CIDR=""
CHAT_RELAY_ENABLE=0
CHAT_RELAY_DISABLE=0
ONION_MIDDLE_ENABLE=0
ONION_MIDDLE_DISABLE=0
RESTART_AFTER_CONFIG=0
PROMOTE_BINARY_PATH=""
EXPECTED_SHA256=""
PEER_CACHE_FILE="${AERONYX_PEER_CACHE_FILE:-/var/lib/aeronyx/peers-cache.json}"
SERVER_KEY_FILE="${AERONYX_SERVER_KEY_FILE:-/etc/aeronyx/server_key.json}"
RELAY_PROBE_PEER_PREFIX=""
RELAY_PROBE_TWO_HOP=0
FLEET_SMOKE_ENDPOINTS="${AERONYX_FLEET_SMOKE_ENDPOINTS:-}"
FLEET_SMOKE_INCLUDE_NEGATIVE=0
FLEET_SMOKE_TWO_HOP=0
BOOTSTRAP_SOURCE_ENDPOINT="${AERONYX_BOOTSTRAP_SOURCE_ENDPOINT:-}"
BOOTSTRAP_SNAPSHOT_PATH="${AERONYX_BOOTSTRAP_SNAPSHOT_PATH:-}"
EXPECTED_DISCOVERY_ENDPOINTS="${AERONYX_EXPECTED_DISCOVERY_ENDPOINTS:-}"
EXPECTED_BOOTSTRAP_SHA256="${AERONYX_EXPECTED_BOOTSTRAP_SHA256:-}"
EXPECTED_BINARY_SHA256="${AERONYX_EXPECTED_BINARY_SHA256:-}"
YES=0
EXTRA_ARGS=()

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage:
  ./deploy/node/aeronyx-node.sh [COMMAND] [OPTIONS]

Source:
  This script is not installed globally by Linux. It is repository-local:
    https://github.com/AeroNyxNetwork/AeroNyx
    AeroNyx/deploy/node/aeronyx-node.sh

  From a fresh server:
    mkdir -p /root/open
    cd /root/open
    git clone https://github.com/AeroNyxNetwork/AeroNyx.git AeroNyx
    cd AeroNyx
    ./deploy/node/aeronyx-node.sh plan --repo-dir "$PWD" --branch main

Commands:
  plan       Print the resolved one-command install plan without host changes.
  quickstart Preview, confirm, install/register/start, status, and healthcheck.
  install    Install/register/start an AeroNyx privacy protocol node.
  upgrade    Pull/build/validate/upgrade the Rust node with safety checks.
  health     Run the node health check. Use --json or --json-only for tooling.
  healthcheck Alias for health.
  doctor     Alias for health.
  status     Show service, local endpoints, discovery readiness, upgrade state, and next action.
  chat-relay Enable or disable blind ChatRelay capability in server.toml.
  onion-middle Enable or disable no-exit OnionMiddle capability in server.toml.
  relay-probe Send one synthetic BlindRelay route probe to a discovered peer.
  fleet-smoke Verify a public multi-node discovery + blind relay mesh.
  refresh-bootstrap Refresh the signed discovery bootstrap snapshot.
  fleet-drift-check Read-only check for seed/snapshot/binary drift.
  promote-binary Promote a staged aeronyx-server binary with drain checks.
  logs       Show recent systemd logs. Use --follow to tail.
  network    Refresh forwarding/NAT or update the privacy protocol IP pool.
  menu       Open the interactive operator menu.
  help       Show this help.

Common options:
  --repo-dir PATH          Repository path. Default: /opt/aeronyx/AeroNyx
  --branch NAME            Git branch/ref. Default: main
  --config PATH            Config path for upgrade/health/status.
  --service NAME           systemd service name. Default: aeronyx-server
  --registration-code CODE Registration code for install.
  --dry-run                Preview actions where the delegated script supports it.

Command-specific options:
  install:
    --quick                First-install shortcut. Requires a registration code.
    --start                Start service after install.
    --no-build             Skip release build.
    --no-network           Skip sysctl and NAT setup.
    --no-enable            Do not enable systemd service.
    --skip-package-install Do not install OS packages automatically.
    --skip-rust-install    Do not install Rust automatically.
    --allow-dirty          Allow tracked Git changes during install update.

  upgrade:
    --force                Restart even when active sessions exist.
    --no-restart           Build and validate only; do not restart.
    --skip-pull            Build currently checked-out source.
    --allow-dirty          Allow tracked Git changes during source upgrade.

  health:
    --json                 Emit JSON summary as final output line.
    --json-only            Emit only JSON.

  logs:
    --follow               Tail logs.
    --lines N              Number of recent log lines. Default: 160.

  network:
    --set-vpn-cidr CIDR    Update vpn.virtual_ip_range and refresh NAT.

  chat-relay:
    --enable-chat-relay     Set [memchain.chat_relay].enabled = true.
    --disable-chat-relay    Set [memchain.chat_relay].enabled = false.
    --restart               Restart service after config validation.
    --yes                   Skip confirmation prompts and allow restart with active sessions.

  onion-middle:
    --enable-onion-middle   Set [discovery].advertise_onion_middle = true.
    --disable-onion-middle  Set [discovery].advertise_onion_middle = false.
    --restart               Restart service after config validation.
    --yes                   Skip confirmation prompts and allow restart with active sessions.

  relay-probe:
    --two-hop               Attempt a live outer+onward middle-hop probe.
    --peer-prefix PREFIX     Optional privacy-safe peer prefix to target.
    --peer-cache PATH        Peer cache path. Default: /var/lib/aeronyx/peers-cache.json.
    --server-key PATH        Local node key path. Default: /etc/aeronyx/server_key.json.
    --json                  Emit JSON result for nodeboard/automation.
    --json-only             Emit only JSON.

  fleet-smoke:
    --endpoints URLS         Comma-separated public discovery base URLs.
    --endpoint URL           Add one public discovery base URL. Repeatable.
    --two-hop                Prove entry -> middle -> terminal BlindRelay path.
    --include-negative       Also run an invalid-signature probe against the first endpoint.
    --json                  Emit JSON result for nodeboard/automation.
    --json-only             Emit only JSON.

  refresh-bootstrap:
    --source-endpoint URL     Discovery base URL to fetch /api/discovery/snapshot from.
                              Default: local 127.0.0.1 discovery API from --config.
    --bootstrap-path PATH     Target bootstrap snapshot path. Default: config value
                              or /etc/aeronyx/bootstrap-peers.json.
    --expected-endpoints URLS Comma-separated endpoints that must be present.
    --dry-run                Validate and preview without writing the snapshot.
    --json                  Emit JSON result for nodeboard/automation.
    --json-only             Emit only JSON.

  fleet-drift-check:
    --expected-endpoints URLS        Comma-separated expected discovery endpoints.
    --expected-bootstrap-sha256 HASH Expected bootstrap snapshot SHA-256.
    --expected-binary-sha256 HASH    Expected aeronyx-server binary SHA-256.
    --bootstrap-path PATH            Bootstrap snapshot path override.
    --json                           Emit JSON result for nodeboard/automation.
    --json-only                      Emit only JSON.

  promote-binary:
    --binary PATH            Staged aeronyx-server binary to promote.
    --expected-sha256 HASH   Optional SHA-256 expected for the staged binary.
    --force                  Allow promotion when active sessions are non-zero or unknown.
    --yes                   Skip confirmation prompt. Required with --force.

Examples:
  ./deploy/node/aeronyx-node.sh plan --quick --registration-code NYX-1234-ABCDE
  ./deploy/node/aeronyx-node.sh quickstart --quick --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh install --quick --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh upgrade --no-restart
  ./deploy/node/aeronyx-node.sh health --json
  ./deploy/node/aeronyx-node.sh status
  ./deploy/node/aeronyx-node.sh relay-probe --json
  ./deploy/node/aeronyx-node.sh fleet-smoke --endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422 --two-hop --json
  sudo ./deploy/node/aeronyx-node.sh refresh-bootstrap --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422
  ./deploy/node/aeronyx-node.sh fleet-drift-check --expected-endpoints http://35.253.79.169:8422,http://8.213.146.244:8422,http://149.33.18.44:8422,http://111.68.15.70:8422 --json
  sudo ./deploy/node/aeronyx-node.sh chat-relay --enable-chat-relay --restart
  sudo ./deploy/node/aeronyx-node.sh onion-middle --enable-onion-middle --restart
  sudo ./deploy/node/aeronyx-node.sh promote-binary --binary ./target/release/aeronyx-server.next --expected-sha256 HASH
USAGE
}

require_script() {
    local script="$1"
    [ -x "${script}" ] || die "Required script is missing or not executable: ${script}"
}

validate_service_name() {
    case "${SERVICE_NAME}" in
        ""|-*|*/*)
            die "Invalid service name: ${SERVICE_NAME}"
            ;;
    esac

    printf '%s' "${SERVICE_NAME}" | grep -Eq '^[A-Za-z0-9_.@-]+$' \
        || die "Invalid service name: ${SERVICE_NAME}"
}

append_registration_code() {
    if [ -n "${REGISTRATION_CODE}" ]; then
        EXTRA_ARGS+=("--registration-code" "${REGISTRATION_CODE}")
    fi
}

array_has_value() {
    local needle="$1"
    shift
    local value
    for value in "$@"; do
        [ "${value}" = "${needle}" ] && return 0
    done
    return 1
}

ensure_extra_arg() {
    local value="$1"
    array_has_value "${value}" "${EXTRA_ARGS[@]}" || EXTRA_ARGS+=("${value}")
}

parse_args() {
    if [ "$#" -eq 0 ]; then
        COMMAND="menu"
        return
    fi

    COMMAND="$1"
    shift

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
            --branch) BRANCH="${2:?missing value}"; shift 2 ;;
            --config) CONFIG_FILE="${2:?missing value}"; shift 2 ;;
            --service) SERVICE_NAME="${2:?missing value}"; shift 2 ;;
            --registration-code) REGISTRATION_CODE="${2:?missing value}"; shift 2 ;;
            --force) FORCE=1; shift ;;
            --no-restart) NO_RESTART=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            --json) JSON=1; shift ;;
            --json-only) JSON_ONLY=1; shift ;;
            --follow) FOLLOW_LOGS=1; shift ;;
            --lines) LINES="${2:?missing value}"; shift 2 ;;
            --set-vpn-cidr) SET_VPN_CIDR="${2:?missing value}"; shift 2 ;;
            --enable-chat-relay) CHAT_RELAY_ENABLE=1; shift ;;
            --disable-chat-relay) CHAT_RELAY_DISABLE=1; shift ;;
            --enable-onion-middle) ONION_MIDDLE_ENABLE=1; shift ;;
            --disable-onion-middle) ONION_MIDDLE_DISABLE=1; shift ;;
            --restart) RESTART_AFTER_CONFIG=1; shift ;;
            --binary|--staged-binary) PROMOTE_BINARY_PATH="${2:?missing value}"; shift 2 ;;
            --expected-sha256) EXPECTED_SHA256="${2:?missing value}"; shift 2 ;;
            --two-hop) RELAY_PROBE_TWO_HOP=1; FLEET_SMOKE_TWO_HOP=1; shift ;;
            --peer-prefix) RELAY_PROBE_PEER_PREFIX="${2:?missing value}"; shift 2 ;;
            --peer-cache) PEER_CACHE_FILE="${2:?missing value}"; shift 2 ;;
            --server-key) SERVER_KEY_FILE="${2:?missing value}"; shift 2 ;;
            --endpoints) FLEET_SMOKE_ENDPOINTS="${2:?missing value}"; shift 2 ;;
            --endpoint)
                if [ -n "${FLEET_SMOKE_ENDPOINTS}" ]; then
                    FLEET_SMOKE_ENDPOINTS="${FLEET_SMOKE_ENDPOINTS},${2:?missing value}"
                else
                    FLEET_SMOKE_ENDPOINTS="${2:?missing value}"
                fi
                shift 2
                ;;
            --include-negative) FLEET_SMOKE_INCLUDE_NEGATIVE=1; shift ;;
            --source-endpoint) BOOTSTRAP_SOURCE_ENDPOINT="${2:?missing value}"; shift 2 ;;
            --bootstrap-path) BOOTSTRAP_SNAPSHOT_PATH="${2:?missing value}"; shift 2 ;;
            --expected-endpoints) EXPECTED_DISCOVERY_ENDPOINTS="${2:?missing value}"; shift 2 ;;
            --expected-bootstrap-sha256) EXPECTED_BOOTSTRAP_SHA256="${2:?missing value}"; shift 2 ;;
            --expected-binary-sha256) EXPECTED_BINARY_SHA256="${2:?missing value}"; shift 2 ;;
            --yes|-y) YES=1; shift ;;
            -h|--help) COMMAND="help"; shift ;;
            --quick|--start|--no-build|--no-network|--no-enable|--skip-package-install|--skip-rust-install|--allow-dirty|--skip-pull)
                EXTRA_ARGS+=("$1")
                shift
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

run_plan() {
    require_script "${INSTALL_SCRIPT}"
    append_registration_code
    "${INSTALL_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --branch "${BRANCH}" \
        --print-plan \
        "${EXTRA_ARGS[@]}"
}

run_install() {
    require_script "${INSTALL_SCRIPT}"
    append_registration_code

    if [ "${DRY_RUN}" -eq 1 ]; then
        EXTRA_ARGS+=("--dry-run")
    fi

    "${INSTALL_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --branch "${BRANCH}" \
        "${EXTRA_ARGS[@]}"
}

run_install_with_privilege() {
    if [ "${EUID:-$(id -u)}" -eq 0 ]; then
        run_install
        return
    fi

    command -v sudo >/dev/null 2>&1 || die "quickstart install requires root. Re-run with sudo or install sudo."

    log "Requesting sudo for install/register/start after approved plan."
    sudo env \
        AERONYX_REGISTRATION_CODE="${REGISTRATION_CODE}" \
        AERONYX_REPO_DIR="${REPO_DIR}" \
        AERONYX_BRANCH="${BRANCH}" \
        AERONYX_CONFIG_FILE="${CONFIG_FILE}" \
        AERONYX_SERVICE_NAME="${SERVICE_NAME}" \
        "${SCRIPT_DIR}/aeronyx-node.sh" install \
        --repo-dir "${REPO_DIR}" \
        --branch "${BRANCH}" \
        "${EXTRA_ARGS[@]}"
}

confirm_quickstart_install() {
    cat <<'CONFIRM'

AeroNyx quickstart has printed the read-only install plan above.
Type INSTALL to continue with host changes, registration, service start,
status, and healthcheck. Press Enter to stop safely.
CONFIRM
    printf 'Confirm: '
    local confirmation
    IFS= read -r confirmation || confirmation=""
    [ "${confirmation}" = "INSTALL" ] || die "Quickstart stopped before install."
}

run_quickstart() {
    [ -n "${REGISTRATION_CODE}" ] || die "quickstart requires --registration-code or AERONYX_REGISTRATION_CODE."

    local original_extra=("${EXTRA_ARGS[@]}")

    log "Step 1/4: preview install plan"
    EXTRA_ARGS=("${original_extra[@]}")
    ensure_extra_arg "--quick"
    run_plan

    confirm_quickstart_install

    log "Step 2/4: install, register, and start AeroNyx privacy protocol node"
    EXTRA_ARGS=("${original_extra[@]}")
    ensure_extra_arg "--quick"
    ensure_extra_arg "--start"
    run_install_with_privilege

    log "Step 3/4: status and operator recommendation"
    EXTRA_ARGS=()
    show_status

    log "Step 4/4: healthcheck JSON summary"
    EXTRA_ARGS=()
    JSON=1
    run_health
}

run_upgrade() {
    require_script "${UPGRADE_SCRIPT}"
    validate_service_name

    if [ "${FORCE}" -eq 1 ]; then
        EXTRA_ARGS+=("--force")
    fi
    if [ "${NO_RESTART}" -eq 1 ]; then
        EXTRA_ARGS+=("--no-restart")
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        EXTRA_ARGS+=("--dry-run")
    fi

    "${UPGRADE_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --branch "${BRANCH}" \
        --config "${CONFIG_FILE}" \
        --service "${SERVICE_NAME}" \
        "${EXTRA_ARGS[@]}"
}

run_health() {
    require_script "${HEALTHCHECK_SCRIPT}"
    validate_service_name

    if [ "${JSON_ONLY}" -eq 1 ]; then
        EXTRA_ARGS+=("--json-only")
    elif [ "${JSON}" -eq 1 ]; then
        EXTRA_ARGS+=("--json")
    fi

    "${HEALTHCHECK_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --config "${CONFIG_FILE}" \
        --service "${SERVICE_NAME}" \
        "${EXTRA_ARGS[@]}"
}

show_operator_recommendation() {
    log "Operator recommendation"
    if [ ! -x "${HEALTHCHECK_SCRIPT}" ]; then
        warn "Healthcheck script not executable or missing: ${HEALTHCHECK_SCRIPT}"
        return
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found; run ./deploy/node/aeronyx-node.sh health --json for the full recommendation"
        return
    fi

    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/aeronyx-health.XXXXXX")" || {
        warn "Could not create temporary file for healthcheck recommendation"
        return
    }

    local healthcheck_exit=0
    "${HEALTHCHECK_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --config "${CONFIG_FILE}" \
        --service "${SERVICE_NAME}" \
        --json-only >"${tmp}" 2>/dev/null || healthcheck_exit=$?

    if [ ! -s "${tmp}" ]; then
        rm -f "${tmp}"
        warn "Healthcheck recommendation unavailable; run ./deploy/node/aeronyx-node.sh health --json for details"
        return
    fi

    python3 - "${tmp}" <<'PY' || true
import json
import sys

def clean(value, limit=520):
    if value is None:
        return "unknown"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] if text else "unknown"

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception as exc:
    print(f"operator_action=unreadable error={clean(exc)}")
    raise SystemExit(0)

action = data.get("operator_action") or {}
if not isinstance(action, dict) or not action:
    print("operator_action=unavailable")
    raise SystemExit(0)

print(
    "operator_status={status} priority={priority} source={source}".format(
        status=clean(action.get("status")),
        priority=clean(action.get("priority")),
        source=clean(action.get("source")),
    )
)
print(f"operator_title={clean(action.get('title'))}")
print(f"operator_detail={clean(action.get('detail'))}")
print(f"operator_next_step={clean(action.get('next_step'))}")
PY
    if [ "${healthcheck_exit}" -ne 0 ]; then
        warn "Healthcheck exited with ${healthcheck_exit}; recommendation was still parsed from JSON"
    fi
    rm -f "${tmp}"
}

discovery_status_url() {
    if command -v python3 >/dev/null 2>&1 && [ -r "${CONFIG_FILE}" ]; then
        python3 - "${CONFIG_FILE}" <<'PY' || {
import sys

config_path = sys.argv[1]
section = None
listen = "127.0.0.1:8422"

try:
    with open(config_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip()
                continue
            if section == "discovery" and line.startswith("public_api_listen_addr"):
                _, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if value:
                    listen = value
except Exception:
    pass

port = "8422"
if ":" in listen:
    port = listen.rsplit(":", 1)[-1].strip()
if not port.isdigit():
    port = "8422"

print(f"http://127.0.0.1:{port}/api/discovery/status")
PY
            printf 'http://127.0.0.1:8422/api/discovery/status\n'
        }
        return
    fi

    printf 'http://127.0.0.1:8422/api/discovery/status\n'
}

show_discovery_readiness() {
    log "Local discovery readiness"

    if ! command -v curl >/dev/null 2>&1; then
        warn "curl not found; discovery readiness unavailable"
        return
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found; discovery readiness summary unavailable"
        return
    fi

    local url
    url="$(discovery_status_url)"
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/aeronyx-discovery.XXXXXX")" || {
        warn "Could not create temporary file for discovery readiness"
        return
    }

    if ! curl -fsS "${url}" >"${tmp}" 2>/dev/null; then
        rm -f "${tmp}"
        warn "Local discovery status is not reachable at ${url}"
        return
    fi

    python3 - "${tmp}" <<'PY' || true
import json
import sys

def clean(value, limit=520):
    if value is None:
        return "unknown"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] if text else "unknown"

def bool_text(value):
    return "true" if bool(value) else "false"

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception as exc:
    print(f"discovery_status=unreadable error={clean(exc)}")
    raise SystemExit(0)

local = data.get("local_capabilities") or {}
peer_store = data.get("peer_store") or {}
quorum = peer_store.get("peer_quorum") or {}
blockers = local.get("advertisement_blockers") or []
if not isinstance(blockers, list):
    blockers = []

print(
    "chat_relay_capability_status={status} configured={configured} runtime_ready={runtime_ready} advertised={advertised} safe_to_advertise={safe}".format(
        status=clean(local.get("status")),
        configured=bool_text(local.get("chat_relay_configured")),
        runtime_ready=bool_text(local.get("chat_relay_runtime_ready")),
        advertised=bool_text(local.get("advertised_chat_relay_capability")),
        safe=bool_text(local.get("safe_to_advertise_chat_relay")),
    )
)
if blockers:
    print("chat_relay_blockers=" + ",".join(clean(item, 80) for item in blockers))
print("chat_relay_detail=" + clean(local.get("detail")))

if quorum:
    print(
        "peer_quorum_status={status} quorum_ready={ready} valid_peers={valid} healthy_peers={healthy} routeable_chat_relays={relays} onion_hops={hops}".format(
            status=clean(quorum.get("status")),
            ready=bool_text(quorum.get("quorum_ready")),
            valid=clean(quorum.get("valid_peers"), 40),
            healthy=clean(quorum.get("healthy_peers"), 40),
            relays=clean(quorum.get("routeable_chat_relays"), 40),
            hops=clean(quorum.get("routeable_onion_middle_hops"), 40),
        )
    )
    print("peer_quorum_next_action=" + clean(quorum.get("next_action")))
else:
    print("peer_quorum_status=unavailable")
PY
    rm -f "${tmp}"
}

show_local_health_summary() {
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl not found; local health summary unavailable"
        return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found; run curl http://127.0.0.1:8421/api/vpn/health for raw health JSON"
        return
    fi

    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/aeronyx-local-health.XXXXXX")" || {
        warn "Could not create temporary file for local health summary"
        return
    }

    if ! curl -fsS http://127.0.0.1:8421/api/vpn/health >"${tmp}" 2>/dev/null; then
        rm -f "${tmp}"
        warn "Local /api/vpn/health is not reachable"
        return
    fi

    log "Local privacy protocol health summary"
    python3 - "${tmp}" <<'PY' || true
import json
import sys

def clean(value, limit=220):
    if value is None:
        return "unknown"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] if text else "unknown"

def number(value):
    try:
        return str(int(value))
    except Exception:
        return "unknown"

def percent(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "unknown"

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception as exc:
    print(f"health_status=unreadable error={clean(exc)}")
    raise SystemExit(0)

protocol = data.get("privacy_protocol_health") or {}
transport = data.get("transport_health") or {}
capacity = data.get("capacity") or {}
interface = capacity.get("interface") or {}
startup = data.get("startup_self_check") or {}
policy = data.get("node_policy") or {}
discovery = data.get("discovery_status") or {}
peer_store = discovery.get("peer_store") or {}
runtime = peer_store.get("runtime") or {}
blind = runtime.get("blind_relay") or {}

print(
    "health_status={status} protocol_status={protocol_status} active_sessions={sessions} active_wallet_devices={devices}".format(
        status=clean(data.get("status")),
        protocol_status=clean(protocol.get("status")),
        sessions=number(data.get("active_sessions")),
        devices=number(data.get("active_wallet_devices")),
    )
)
print(
    "transport=preferred:{preferred} effective:{effective} udp:{udp_status} fallback_available:{fallback}".format(
        preferred=clean(transport.get("preferred_transport")),
        effective=clean(transport.get("effective_transport")),
        udp_status=clean((transport.get("udp") or {}).get("status")),
        fallback=clean(transport.get("fallback_available")),
    )
)
print(
    "startup_self_check=status:{status} failed:{failed} warnings:{warnings}".format(
        status=clean(startup.get("status")),
        failed=number(startup.get("failed_checks")),
        warnings=number(startup.get("warning_checks")),
    )
)
print(
    "capacity=ip_pool:{used}/{total} free:{free} max_connections:{max_conn} policy_max_sessions:{policy_max}".format(
        used=number(capacity.get("ip_pool_used")),
        total=number(capacity.get("ip_pool_capacity")),
        free=number(capacity.get("ip_pool_free")),
        max_conn=number(capacity.get("max_connections")),
        policy_max=number(capacity.get("policy_max_sessions")),
    )
)
print(
    "runtime_limits=conntrack:{conntrack} fd:{fd} packet_drops:{drops} pps:{pps} bps:{bps}".format(
        conntrack=percent((capacity.get("conntrack") or {}).get("used_percent")),
        fd=percent((capacity.get("file_descriptors") or {}).get("used_percent")),
        drops=number(capacity.get("packet_drops_total")),
        pps=clean(interface.get("total_pps"), 40),
        bps=clean(interface.get("total_bps"), 40),
    )
)
print(
    "node_policy=tier:{tier} maintenance:{maintenance} max_sessions:{max_sessions} bandwidth_mbps:{bandwidth}".format(
        tier=clean(policy.get("node_tier"), 80),
        maintenance=clean(policy.get("maintenance_mode"), 20),
        max_sessions=number(policy.get("max_sessions")),
        bandwidth=number(policy.get("bandwidth_limit_mbps")),
    )
)
if blind:
    failures = int(blind.get("rejected") or 0) + int(blind.get("forward_failed") or 0) + int(blind.get("no_route") or 0)
    print(
        "blind_relay=received:{received} forwarded:{forwarded} terminal:{terminal} failures:{failures} last_event_at:{last}".format(
            received=number(blind.get("received")),
            forwarded=number(blind.get("forwarded")),
            terminal=number(blind.get("terminal")),
            failures=failures,
            last=clean(blind.get("last_event_at"), 40),
        )
    )
print("privacy_boundary=aggregate node health only; no client public IPs, destinations, DNS contents, packet payloads, chat plaintext, private keys, voucher secrets, wallet-level traffic, route IDs, or social graph edges")
PY
    rm -f "${tmp}"
}

show_local_operator_status_summary() {
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl not found; local operator status summary unavailable"
        return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found; run curl http://127.0.0.1:8421/api/operator/status for raw operator JSON"
        return
    fi

    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/aeronyx-operator-status.XXXXXX")" || {
        warn "Could not create temporary file for operator status summary"
        return
    }

    if ! curl -fsS http://127.0.0.1:8421/api/operator/status >"${tmp}" 2>/dev/null; then
        rm -f "${tmp}"
        warn "Local /api/operator/status is not reachable"
        return
    fi

    log "Local operator status summary"
    python3 - "${tmp}" <<'PY' || true
import json
import sys

def clean(value, limit=220):
    if value is None:
        return "unknown"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] if text else "unknown"

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception as exc:
    print(f"operator_status=unreadable error={clean(exc)}")
    raise SystemExit(0)

services = data.get("services") or []
risks = data.get("risks") or []
if not isinstance(services, list):
    services = []
if not isinstance(risks, list):
    risks = []

service_summary = []
for service in services[:8]:
    if not isinstance(service, dict):
        continue
    service_summary.append(f"{clean(service.get('key'), 60)}:{clean(service.get('status'), 40)}")

print("operator_services=" + (", ".join(service_summary) if service_summary else "unavailable"))
print(f"operator_risks={len(risks)}")
for risk in risks[:5]:
    if isinstance(risk, dict):
        print(
            "operator_risk={severity}:{code}:{message}".format(
                severity=clean(risk.get("severity"), 40),
                code=clean(risk.get("code"), 80),
                message=clean(risk.get("message"), 180),
            )
        )
PY
    rm -f "${tmp}"
}

show_status() {
    validate_service_name

    log "Service status: ${SERVICE_NAME}"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --no-pager --lines=20 status "${SERVICE_NAME}" || true
    else
        warn "systemctl not found"
    fi

    show_local_health_summary
    show_local_operator_status_summary

    show_discovery_readiness

    log "Local upgrade workflow status"
    if [ -s "${UPGRADE_STATUS_FILE}" ]; then
        if command -v python3 >/dev/null 2>&1; then
            python3 - "${UPGRADE_STATUS_FILE}" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception as exc:
    print(f"upgrade_status=unreadable error={exc}")
    raise SystemExit(0)

print(
    "upgrade_status={status} step={step} updated_at={updated_at} no_restart={no_restart}".format(
        status=data.get("status", "unknown"),
        step=data.get("step", "unknown"),
        updated_at=data.get("updated_at", "unknown"),
        no_restart=data.get("no_restart", "unknown"),
    )
)
message = data.get("message")
if message:
    print(f"upgrade_message={message}")
PY
        else
            cat "${UPGRADE_STATUS_FILE}"
        fi
    else
        warn "No local upgrade status file found: ${UPGRADE_STATUS_FILE}"
    fi

    show_operator_recommendation
}

show_logs() {
    validate_service_name

    if ! command -v journalctl >/dev/null 2>&1; then
        die "journalctl not found"
    fi

    if [ "${FOLLOW_LOGS}" -eq 1 ]; then
        journalctl -u "${SERVICE_NAME}" -n "${LINES}" -f
    else
        journalctl -u "${SERVICE_NAME}" -n "${LINES}" --no-pager
    fi
}

run_network() {
    require_script "${INSTALL_SCRIPT}"
    [ -n "${SET_VPN_CIDR}" ] || die "network requires --set-vpn-cidr CIDR"

    if [ "${DRY_RUN}" -eq 1 ]; then
        EXTRA_ARGS+=("--dry-run")
    fi

    "${INSTALL_SCRIPT}" \
        --repo-dir "${REPO_DIR}" \
        --branch "${BRANCH}" \
        --network-only \
        --set-vpn-cidr "${SET_VPN_CIDR}" \
        "${EXTRA_ARGS[@]}"
}

active_sessions_count() {
    if ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
        printf 'unknown\n'
        return
    fi

    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/aeronyx-health-active.XXXXXX")" || {
        printf 'unknown\n'
        return
    }

    if ! curl -fsS http://127.0.0.1:8421/api/vpn/health >"${tmp}" 2>/dev/null; then
        rm -f "${tmp}"
        printf 'unknown\n'
        return
    fi

    python3 - "${tmp}" <<'PY' || printf 'unknown\n'
import json
import sys

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    print("unknown")
    raise SystemExit(0)

value = data.get("active_sessions")
try:
    print(int(value))
except Exception:
    print("unknown")
PY
    rm -f "${tmp}"
}

run_refresh_bootstrap() {
    command -v python3 >/dev/null 2>&1 || die "refresh-bootstrap requires python3"

    if [ "${JSON_ONLY}" -ne 1 ]; then
        log "Refreshing signed AeroNyx discovery bootstrap snapshot"
        log "Config: ${CONFIG_FILE}"
        if [ -n "${BOOTSTRAP_SOURCE_ENDPOINT}" ]; then
            log "Source endpoint: ${BOOTSTRAP_SOURCE_ENDPOINT}"
        else
            log "Source endpoint: local discovery API derived from config"
        fi
        if [ "${DRY_RUN}" -eq 1 ]; then
            warn "Dry run enabled; no bootstrap snapshot will be written."
        fi
    fi

    python3 - \
        "${CONFIG_FILE}" \
        "${BOOTSTRAP_SOURCE_ENDPOINT}" \
        "${BOOTSTRAP_SNAPSHOT_PATH}" \
        "${EXPECTED_DISCOVERY_ENDPOINTS}" \
        "${DRY_RUN}" \
        "${JSON}" \
        "${JSON_ONLY}" <<'PY'
import ast
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

config_path, source_endpoint, bootstrap_path, expected_endpoints_raw, dry_run_raw, json_raw, json_only_raw = sys.argv[1:8]
dry_run = dry_run_raw == "1"
emit_json = json_raw == "1"
json_only = json_only_raw == "1"


def clean(value, limit=320):
    text = str(value or "").replace("\x00", "").replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip('"').strip("'")


def parse_config(path):
    section = None
    parsed = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line.strip("[]").strip()
                    parsed.setdefault(section, {})
                    continue
                if "=" not in line or not section:
                    continue
                key, value = line.split("=", 1)
                parsed.setdefault(section, {})[key.strip()] = parse_scalar(value)
    except FileNotFoundError:
        pass
    return parsed


def expected_endpoints():
    return [item.strip().rstrip("/") for item in expected_endpoints_raw.split(",") if item.strip()]


def local_discovery_base(config):
    listen = str(config.get("discovery", {}).get("public_api_listen_addr") or "127.0.0.1:8422")
    port = "8422"
    if ":" in listen:
        port = listen.rsplit(":", 1)[-1].strip()
    if not port.isdigit():
        port = "8422"
    return f"http://127.0.0.1:{port}"


def snapshot_url(base):
    base = (base or "").strip().rstrip("/")
    if not base:
        raise ValueError("missing discovery source endpoint")
    if "/api/discovery/snapshot" in base:
        return base
    return f"{base}/api/discovery/snapshot?limit=64"


def fetch_snapshot(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=8) as response:
        raw = response.read(1024 * 1024)
    return json.loads(raw.decode("utf-8"))


def snapshot_endpoints(snapshot):
    endpoints = []
    for peer in snapshot.get("peers") or []:
        if not isinstance(peer, dict):
            continue
        descriptor = peer.get("descriptor")
        if not isinstance(descriptor, dict):
            continue
        endpoint = descriptor.get("public_endpoint")
        if isinstance(endpoint, str) and endpoint.strip():
            endpoints.append(endpoint.strip().rstrip("/"))
    return endpoints


def validate_snapshot(snapshot, expected):
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot is not a JSON object")
    peers = snapshot.get("peers")
    if not isinstance(peers, list) or not peers:
        raise ValueError("snapshot has no peers")

    endpoints = snapshot_endpoints(snapshot)
    if not endpoints:
        raise ValueError("snapshot contains no public endpoints")
    for index, peer in enumerate(peers):
        if not isinstance(peer, dict):
            raise ValueError(f"peer[{index}] is not an object")
        if not isinstance(peer.get("descriptor"), dict):
            raise ValueError(f"peer[{index}] missing descriptor")
        if "signature" not in peer:
            raise ValueError(f"peer[{index}] missing signature")

    missing = sorted(set(expected) - set(endpoints))
    if missing:
        raise ValueError("snapshot missing expected endpoints: " + ",".join(missing))
    return endpoints


def write_snapshot(target_path, snapshot):
    directory = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(directory, exist_ok=True)
    backup_path = None
    if os.path.exists(target_path):
        backup_path = f"{target_path}.backup.refresh-bootstrap-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}"
        shutil.copy2(target_path, backup_path)
    temp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.chmod(temp_path, 0o644)
    os.replace(temp_path, target_path)
    return backup_path


config = parse_config(config_path)
target_path = (
    bootstrap_path
    or str(config.get("discovery", {}).get("bootstrap_snapshot_path") or "")
    or "/etc/aeronyx/bootstrap-peers.json"
)
source_base = source_endpoint or local_discovery_base(config)
url = snapshot_url(source_base)
expected = expected_endpoints()

result = {
    "source": "aeronyx-node.sh refresh-bootstrap",
    "status": "unknown",
    "config_path": config_path,
    "source_url": url,
    "target_path": target_path,
    "dry_run": dry_run,
    "expected_endpoints": expected,
    "privacy_boundary": (
        "signed node discovery descriptors only; no user messages, DNS contents, "
        "destinations, packet payloads, client public IPs, private keys, voucher "
        "secrets, wallet-level traffic, or social graph metadata"
    ),
}

try:
    snapshot = fetch_snapshot(url)
    endpoints = validate_snapshot(snapshot, expected)
    encoded = (json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    sha256 = hashlib.sha256(encoded).hexdigest()
    backup_path = None
    if not dry_run:
        backup_path = write_snapshot(target_path, snapshot)
    result.update({
        "status": "ok",
        "peer_count": len(snapshot.get("peers") or []),
        "endpoints": sorted(endpoints),
        "sha256": sha256,
        "wrote": not dry_run,
        "backup_path": backup_path,
    })
except Exception as exc:
    result.update({
        "status": "error",
        "reason": clean(exc),
    })

if not json_only:
    print(
        "refresh_bootstrap_status={status} peers={peers} wrote={wrote} sha256={sha}".format(
            status=result.get("status"),
            peers=result.get("peer_count", 0),
            wrote=str(result.get("wrote", False)).lower(),
            sha=result.get("sha256", "none"),
        )
    )
    if result.get("backup_path"):
        print("refresh_bootstrap_backup=" + clean(result.get("backup_path")))
    for endpoint in result.get("endpoints") or []:
        print("refresh_bootstrap_endpoint=" + clean(endpoint, 180))
    if result.get("status") != "ok":
        print("refresh_bootstrap_reason=" + clean(result.get("reason")))

if emit_json or json_only:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

if result.get("status") != "ok":
    raise SystemExit(1)
PY
}


run_fleet_drift_check() {
    command -v python3 >/dev/null 2>&1 || die "fleet-drift-check requires python3"

    if [ "${JSON_ONLY}" -ne 1 ]; then
        log "Checking local AeroNyx node fleet drift"
        log "Config: ${CONFIG_FILE}"
    fi

    python3 - \
        "${CONFIG_FILE}" \
        "${BOOTSTRAP_SNAPSHOT_PATH}" \
        "${EXPECTED_DISCOVERY_ENDPOINTS}" \
        "${EXPECTED_BOOTSTRAP_SHA256}" \
        "${EXPECTED_BINARY_SHA256}" \
        "${SERVICE_NAME}" \
        "${REPO_DIR}" \
        "${JSON}" \
        "${JSON_ONLY}" <<'PY'
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

(
    config_path,
    bootstrap_path_override,
    expected_endpoints_raw,
    expected_bootstrap_sha256,
    expected_binary_sha256,
    service_name,
    repo_dir,
    json_raw,
    json_only_raw,
) = sys.argv[1:10]
emit_json = json_raw == "1"
json_only = json_only_raw == "1"


def clean(value, limit=320):
    text = str(value or "").replace("\x00", "").replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip('"').strip("'")


def parse_config(path):
    section = None
    parsed = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line.strip("[]").strip()
                    parsed.setdefault(section, {})
                    continue
                if "=" not in line or not section:
                    continue
                key, value = line.split("=", 1)
                parsed.setdefault(section, {})[key.strip()] = parse_scalar(value)
    except FileNotFoundError:
        pass
    return parsed


def normalize_endpoints(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return sorted({str(item).strip().rstrip("/") for item in values if str(item).strip()})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def service_binary_path(service):
    if not service:
        return None
    unit = service if service.endswith(".service") else f"{service}.service"
    try:
        output = subprocess.check_output(
            ["systemctl", "show", unit, "-p", "ExecStart", "--value"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    path_match = re.search(r"(?:^|\s)path=([^ ;]+)", output)
    if path_match and os.path.exists(path_match.group(1)):
        return path_match.group(1)
    try:
        parts = shlex.split(output)
    except ValueError:
        return None
    for part in parts:
        if part.endswith("aeronyx-server") and os.path.exists(part):
            return part
    return None


def local_discovery_base(discovery):
    listen = str(discovery.get("public_api_listen_addr") or "127.0.0.1:8422")
    port = "8422"
    if ":" in listen:
        port = listen.rsplit(":", 1)[-1].strip()
    if not port.isdigit():
        port = "8422"
    return f"http://127.0.0.1:{port}"


def fetch_public_card(discovery):
    base = local_discovery_base(discovery)
    request = urllib.request.Request(
        f"{base}/api/discovery/public-card",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read(128 * 1024).decode("utf-8"))


def bootstrap_endpoints(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    endpoints = []
    for peer in payload.get("peers") or []:
        if not isinstance(peer, dict):
            continue
        descriptor = peer.get("descriptor")
        if not isinstance(descriptor, dict):
            continue
        endpoint = descriptor.get("public_endpoint")
        if isinstance(endpoint, str) and endpoint:
            endpoints.append(endpoint)
    return normalize_endpoints(endpoints)


def endpoint_comparison(actual, expected):
    actual_set = set(actual)
    expected_set = set(expected)
    return {
        "matches": bool(expected) and actual_set == expected_set,
        "missing": sorted(expected_set - actual_set),
        "extra": sorted(actual_set - expected_set),
    }


config = parse_config(config_path)
discovery = config.get("discovery") or {}
memchain = config.get("memchain") or {}
expected_endpoints = normalize_endpoints(
    [item.strip() for item in expected_endpoints_raw.split(",") if item.strip()]
)
seed_endpoints = normalize_endpoints(discovery.get("seed_endpoints"))
public_endpoint = clean(discovery.get("public_endpoint"), 200)
bootstrap_path = (
    bootstrap_path_override
    or str(discovery.get("bootstrap_snapshot_path") or "")
    or "/etc/aeronyx/bootstrap-peers.json"
)
bootstrap_sha256 = sha256_file(bootstrap_path) if os.path.exists(bootstrap_path) else None
bootstrap_peers = bootstrap_endpoints(bootstrap_path)

binary_path = service_binary_path(service_name)
if not binary_path:
    candidate = os.path.join(repo_dir, "target", "release", "aeronyx-server")
    binary_path = candidate if os.path.exists(candidate) else None
binary_sha256 = sha256_file(binary_path) if binary_path and os.path.exists(binary_path) else None

public_card = {}
public_card_error = None
try:
    public_card = fetch_public_card(discovery)
except Exception as exc:
    public_card_error = clean(exc)

issues = []
seed_cmp = endpoint_comparison(seed_endpoints, expected_endpoints) if expected_endpoints else None
bootstrap_cmp = endpoint_comparison(bootstrap_peers, expected_endpoints) if expected_endpoints else None
if expected_endpoints and not seed_cmp["matches"]:
    issues.append("seed_endpoints_drift")
if expected_endpoints and not bootstrap_cmp["matches"]:
    issues.append("bootstrap_snapshot_drift")
if expected_bootstrap_sha256 and bootstrap_sha256 != expected_bootstrap_sha256:
    issues.append("bootstrap_sha256_mismatch")
if expected_binary_sha256 and binary_sha256 != expected_binary_sha256:
    issues.append("binary_sha256_mismatch")
if not public_card or public_card.get("status") not in ("ready", "live"):
    issues.append("public_card_not_ready")

status = "ok" if not issues else "attention"
result = {
    "source": "aeronyx-node.sh fleet-drift-check",
    "status": status,
    "issues": issues,
    "config_path": config_path,
    "service": service_name,
    "public_endpoint": public_endpoint,
    "seed_endpoints": {
        "actual": seed_endpoints,
        "expected": expected_endpoints,
        "matches_expected": None if seed_cmp is None else seed_cmp["matches"],
        "missing": [] if seed_cmp is None else seed_cmp["missing"],
        "extra": [] if seed_cmp is None else seed_cmp["extra"],
    },
    "bootstrap_snapshot": {
        "path": bootstrap_path,
        "exists": os.path.exists(bootstrap_path),
        "sha256": bootstrap_sha256,
        "expected_sha256": expected_bootstrap_sha256 or None,
        "sha256_matches_expected": (
            None if not expected_bootstrap_sha256 else bootstrap_sha256 == expected_bootstrap_sha256
        ),
        "endpoints": bootstrap_peers,
        "matches_expected_endpoints": None if bootstrap_cmp is None else bootstrap_cmp["matches"],
        "missing": [] if bootstrap_cmp is None else bootstrap_cmp["missing"],
        "extra": [] if bootstrap_cmp is None else bootstrap_cmp["extra"],
    },
    "binary": {
        "path": binary_path,
        "sha256": binary_sha256,
        "expected_sha256": expected_binary_sha256 or None,
        "sha256_matches_expected": (
            None if not expected_binary_sha256 else binary_sha256 == expected_binary_sha256
        ),
    },
    "memchain": {
        "mode": clean(memchain.get("mode"), 80),
        "embed_enabled": memchain.get("embed_enabled") if "embed_enabled" in memchain else None,
    },
    "public_card": {
        "reachable": bool(public_card),
        "error": public_card_error,
        "status": public_card.get("status") if isinstance(public_card, dict) else None,
        "stage": public_card.get("stage") if isinstance(public_card, dict) else None,
        "contract_version": public_card.get("contract_version") if isinstance(public_card, dict) else None,
        "ready_nodes_view": (public_card.get("cards") or {}).get("verified_mesh") if isinstance(public_card, dict) else None,
    },
    "privacy_boundary": (
        "node configuration and aggregate discovery health only; no registration "
        "codes, API secrets, private keys, user messages, DNS contents, destinations, "
        "packet payloads, client public IPs, wallet-level traffic, or social graph metadata"
    ),
}

if not json_only:
    print(f"fleet_drift_status={result['status']} issues={','.join(issues) if issues else 'none'}")
    print("seed_endpoints_match=" + str(result["seed_endpoints"]["matches_expected"]).lower())
    print("bootstrap_endpoints_match=" + str(result["bootstrap_snapshot"]["matches_expected_endpoints"]).lower())
    print("bootstrap_sha256=" + clean(bootstrap_sha256 or "missing", 96))
    if expected_bootstrap_sha256:
        print("bootstrap_sha256_match=" + str(result["bootstrap_snapshot"]["sha256_matches_expected"]).lower())
    print("binary_sha256=" + clean(binary_sha256 or "missing", 96))
    if expected_binary_sha256:
        print("binary_sha256_match=" + str(result["binary"]["sha256_matches_expected"]).lower())
    print(
        "public_card_status={status} stage={stage}".format(
            status=clean(result["public_card"].get("status")),
            stage=clean(result["public_card"].get("stage")),
        )
    )
    if result["memchain"].get("embed_enabled") is None:
        print("memchain_embed_enabled=default")
    else:
        print("memchain_embed_enabled=" + str(result["memchain"].get("embed_enabled")).lower())

if emit_json or json_only:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

if status != "ok":
    raise SystemExit(2)
PY
}


run_fleet_smoke() {
    command -v python3 >/dev/null 2>&1 || die "fleet-smoke requires python3"
    [ -n "${FLEET_SMOKE_ENDPOINTS}" ] || die "fleet-smoke requires --endpoints URL1,URL2,... or repeated --endpoint URL"

    if [ "${JSON_ONLY}" -ne 1 ]; then
        log "Running public AeroNyx fleet smoke test"
        log "Endpoints: ${FLEET_SMOKE_ENDPOINTS}"
        if [ "${FLEET_SMOKE_INCLUDE_NEGATIVE}" -eq 1 ]; then
            warn "Negative invalid-signature probe is enabled; this intentionally increments aggregate rejection counters."
        fi
        if [ "${FLEET_SMOKE_TWO_HOP}" -eq 1 ]; then
            log "Two-hop path proof enabled: entry -> middle -> terminal"
        fi
    fi

    python3 - \
        "${FLEET_SMOKE_ENDPOINTS}" \
        "${FLEET_SMOKE_INCLUDE_NEGATIVE}" \
        "${FLEET_SMOKE_TWO_HOP}" \
        "${JSON}" \
        "${JSON_ONLY}" <<'PY'
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    def new_signing_key():
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return private_key.sign, public_key, "cryptography"
except Exception as crypto_exc:
    try:
        from nacl.signing import SigningKey

        def new_signing_key():
            private_key = SigningKey.generate()
            public_key = bytes(private_key.verify_key)
            return private_key.sign, public_key, "pynacl"
    except Exception as nacl_exc:
        print(json.dumps({
            "status": "error",
            "reason": "missing_ed25519_python_dependency",
            "cryptography_error": str(crypto_exc)[:240],
            "pynacl_error": str(nacl_exc)[:240],
            "next_action": "Install python3-cryptography or python3-nacl on the operator host.",
        }, sort_keys=True))
        raise SystemExit(2)

DOMAIN = b"AeroNyx-BlindRelay-v1"
endpoints_arg, include_negative_arg, two_hop_arg, json_arg, json_only_arg = sys.argv[1:6]
include_negative = include_negative_arg == "1"
two_hop_requested = two_hop_arg == "1"
json_requested = json_arg == "1"
json_only = json_only_arg == "1"
endpoints = []
for item in endpoints_arg.split(","):
    endpoint = item.strip().rstrip("/")
    if endpoint:
        endpoints.append(endpoint)


def fail(reason, detail=None):
    result = {
        "status": "error",
        "reason": reason,
        "detail": detail,
        "source": "aeronyx-node.sh fleet-smoke",
    }
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(2)


if len(endpoints) < 2:
    fail("not_enough_endpoints", "fleet-smoke needs at least two public discovery endpoints")
if two_hop_requested and len(endpoints) < 3:
    fail("not_enough_endpoints_for_two_hop", "fleet-smoke --two-hop needs at least three public discovery endpoints")


def request_json(url, method="GET", body=None, timeout=18):
    started = time.perf_counter()
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return {
                "ok": True,
                "status_code": response.status,
                "body": parsed,
                "latency_ms": int(round((time.perf_counter() - started) * 1000)),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw[:500]}
        return {
            "ok": False,
            "status_code": exc.code,
            "body": parsed,
            "latency_ms": int(round((time.perf_counter() - started) * 1000)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "body": {"error": str(exc)[:500]},
            "latency_ms": int(round((time.perf_counter() - started) * 1000)),
        }


def as_bytes(value):
    if isinstance(value, list):
        return bytes(int(item) & 0xFF for item in value)
    if isinstance(value, str):
        return bytes.fromhex(value)
    raise ValueError("unsupported byte value")


def signature_tuple(signature):
    signature = bytes(signature)
    return [list(signature[:32]), list(signature[32:])]


def sign_data(route_id, next_hop, ttl, timestamp, blob, sign):
    payload = (
        DOMAIN
        + route_id
        + next_hop
        + bytes([ttl])
        + int(timestamp).to_bytes(8, "little")
        + hashlib.sha256(blob).digest()
    )
    signed = sign(payload)
    if hasattr(signed, "signature"):
        return bytes(signed.signature)
    return bytes(signed)


def public_endpoint(peer):
    descriptor = peer.get("descriptor") if isinstance(peer, dict) else None
    if not isinstance(descriptor, dict):
        descriptor = peer if isinstance(peer, dict) else {}
    return str(descriptor.get("public_endpoint") or "").rstrip("/")


def descriptor_node_id(peer):
    descriptor = peer.get("descriptor") if isinstance(peer, dict) else None
    if not isinstance(descriptor, dict):
        descriptor = peer if isinstance(peer, dict) else {}
    return as_bytes(descriptor.get("node_id"))


def find_self_peer(snapshot, endpoint):
    peers = snapshot.get("peers") if isinstance(snapshot, dict) else []
    for peer in peers if isinstance(peers, list) else []:
        if public_endpoint(peer) == endpoint:
            return peer
    return None


def first_candidate_node_id(candidates):
    values = candidates.get("candidates") if isinstance(candidates, dict) else []
    if not isinstance(values, list) or not values:
        return None, None
    candidate = values[0]
    node_id = candidate.get("node_id")
    return as_bytes(node_id), str(node_id)[:8]


def candidate_node_ids(candidates):
    values = candidates.get("candidates") if isinstance(candidates, dict) else []
    out = []
    for candidate in values if isinstance(values, list) else []:
        node_id = candidate.get("node_id") if isinstance(candidate, dict) else None
        if not node_id:
            continue
        try:
            node_id_bytes = as_bytes(node_id)
        except Exception:
            continue
        out.append((node_id_bytes, node_id_bytes.hex()[:8]))
    return out


def build_envelope(next_hop, ttl, label, corrupt=False):
    sign, previous_pub, signer = new_signing_key()
    route_id = os.urandom(16)
    timestamp = int(time.time())
    blob = ("synthetic-fleet-smoke:%s:no-user-payload" % label).encode("utf-8")
    signature = sign_data(route_id, next_hop, ttl, timestamp, blob, sign)
    if corrupt:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    return {
        "body": {
            "envelope": {
                "route_id": list(route_id),
                "next_hop": list(next_hop),
                "ttl": ttl,
                "encrypted_blob": list(blob),
                "timestamp": timestamp,
                "signature": signature_tuple(signature),
            },
            "previous_hop_node_id": list(previous_pub),
        },
        "route_id_prefix": route_id.hex()[:8],
        "signer": signer,
    }


endpoint_results = []
terminal_results = []
forward_results = []
two_hop_results = []
negative_result = None
all_ok = True
node_hex_by_endpoint = {}
endpoint_by_node_hex = {}
candidate_ids_by_endpoint = {}

for endpoint in endpoints:
    status_result = request_json(endpoint + "/api/discovery/status")
    snapshot_result = request_json(endpoint + "/api/discovery/snapshot?limit=8")
    candidate_result = request_json(endpoint + "/api/discovery/onion-candidates")
    snapshot = snapshot_result.get("body") if snapshot_result.get("ok") else {}
    peer = find_self_peer(snapshot, endpoint)
    if peer is not None:
        try:
            self_node_id_for_map = descriptor_node_id(peer)
            node_hex_by_endpoint[endpoint] = self_node_id_for_map.hex()
            endpoint_by_node_hex[self_node_id_for_map.hex()] = endpoint
        except Exception:
            pass
    candidate_body = candidate_result.get("body") if isinstance(candidate_result.get("body"), dict) else {}
    candidate_ids_by_endpoint[endpoint] = candidate_node_ids(candidate_body)
    endpoint_ok = bool(
        status_result.get("ok")
        and snapshot_result.get("ok")
        and candidate_result.get("ok")
        and peer is not None
    )
    status_body = status_result.get("body") if isinstance(status_result.get("body"), dict) else {}
    peer_store = status_body.get("peer_store") if isinstance(status_body, dict) else {}
    snapshot_summary = peer_store.get("snapshot") if isinstance(peer_store, dict) else {}
    readiness = status_body.get("discovery_readiness") if isinstance(status_body, dict) else {}
    foundation = readiness.get("protocol_foundation") if isinstance(readiness, dict) else {}
    endpoint_summary = {
        "endpoint": endpoint,
        "status": "ok" if endpoint_ok else "failed",
        "status_latency_ms": status_result.get("latency_ms"),
        "snapshot_latency_ms": snapshot_result.get("latency_ms"),
        "candidate_latency_ms": candidate_result.get("latency_ms"),
        "valid_peers": snapshot_summary.get("valid_peers"),
        "public_peers": snapshot_summary.get("public_peers"),
        "protocol_stage": foundation.get("stage"),
        "protocol_status": foundation.get("status"),
        "self_descriptor_found": peer is not None,
        "status_code": status_result.get("status_code"),
        "snapshot_status_code": snapshot_result.get("status_code"),
        "candidate_status_code": candidate_result.get("status_code"),
    }
    endpoint_results.append(endpoint_summary)
    all_ok = all_ok and endpoint_ok

    if peer is None:
        continue

    self_node_id = descriptor_node_id(peer)
    terminal_probe = build_envelope(self_node_id, 1, "terminal")
    terminal_http = request_json(
        endpoint + "/api/chat/peer/blind-relay",
        method="POST",
        body=terminal_probe["body"],
    )
    terminal_body = terminal_http.get("body") if isinstance(terminal_http.get("body"), dict) else {}
    terminal_ok = bool(
        terminal_http.get("status_code") == 200
        and terminal_body.get("accepted") is True
        and terminal_body.get("terminal") is True
        and terminal_body.get("forwarded") is False
    )
    terminal_results.append({
        "endpoint": endpoint,
        "status": "ok" if terminal_ok else "failed",
        "response_status": terminal_http.get("status_code"),
        "latency_ms": terminal_http.get("latency_ms"),
        "reason": terminal_body.get("reason"),
        "route_id_prefix": terminal_probe["route_id_prefix"],
    })
    all_ok = all_ok and terminal_ok

    next_hop, next_prefix = first_candidate_node_id(candidate_body)
    if next_hop is None:
        forward_results.append({
            "endpoint": endpoint,
            "status": "failed",
            "reason": "no_onion_candidate",
        })
        all_ok = False
        continue

    forward_probe = build_envelope(next_hop, 1, "forward")
    forward_http = request_json(
        endpoint + "/api/chat/peer/blind-relay",
        method="POST",
        body=forward_probe["body"],
    )
    forward_body = forward_http.get("body") if isinstance(forward_http.get("body"), dict) else {}
    forward_ok = bool(
        forward_http.get("status_code") == 200
        and forward_body.get("accepted") is True
        and forward_body.get("forwarded") is True
        and forward_body.get("terminal") is False
    )
    forward_results.append({
        "endpoint": endpoint,
        "status": "ok" if forward_ok else "failed",
        "response_status": forward_http.get("status_code"),
        "latency_ms": forward_http.get("latency_ms"),
        "next_hop_prefix": next_prefix,
        "reason": forward_body.get("reason"),
        "route_id_prefix": forward_probe["route_id_prefix"],
    })
    all_ok = all_ok and forward_ok

if two_hop_requested:
    for entry_endpoint in endpoints:
        entry_hex = node_hex_by_endpoint.get(entry_endpoint)
        selected = None
        for middle_id, middle_prefix in candidate_ids_by_endpoint.get(entry_endpoint, []):
            middle_hex = middle_id.hex()
            middle_endpoint = endpoint_by_node_hex.get(middle_hex)
            if not middle_endpoint or middle_endpoint == entry_endpoint:
                continue
            for terminal_id, terminal_prefix in candidate_ids_by_endpoint.get(middle_endpoint, []):
                terminal_hex = terminal_id.hex()
                terminal_endpoint = endpoint_by_node_hex.get(terminal_hex)
                if not terminal_endpoint:
                    continue
                if terminal_endpoint in {entry_endpoint, middle_endpoint}:
                    continue
                selected = {
                    "middle_id": middle_id,
                    "middle_prefix": middle_prefix,
                    "middle_endpoint": middle_endpoint,
                    "terminal_id": terminal_id,
                    "terminal_prefix": terminal_prefix,
                    "terminal_endpoint": terminal_endpoint,
                }
                break
            if selected is not None:
                break

        if selected is None:
            two_hop_results.append({
                "entry_endpoint": entry_endpoint,
                "status": "failed",
                "reason": "no_distinct_routeable_two_hop_path",
            })
            all_ok = False
            continue

        outer_probe = build_envelope(selected["middle_id"], 2, "two-hop-outer")
        onward_probe = build_envelope(selected["terminal_id"], 1, "two-hop-onward")
        body = dict(outer_probe["body"])
        body["onward_envelope"] = onward_probe["body"]["envelope"]
        proof_http = request_json(
            entry_endpoint + "/api/chat/peer/blind-relay",
            method="POST",
            body=body,
            timeout=24,
        )
        proof_body = proof_http.get("body") if isinstance(proof_http.get("body"), dict) else {}
        proof_ok = bool(
            proof_http.get("status_code") == 200
            and proof_body.get("accepted") is True
            and proof_body.get("forwarded") is True
            and proof_body.get("terminal") is False
            and proof_body.get("reason") in ("forwarded", "onion_middle_forwarded")
        )
        two_hop_results.append({
            "entry_endpoint": entry_endpoint,
            "middle_endpoint": selected["middle_endpoint"],
            "terminal_endpoint": selected["terminal_endpoint"],
            "middle_prefix": selected["middle_prefix"],
            "terminal_prefix": selected["terminal_prefix"],
            "status": "ok" if proof_ok else "failed",
            "response_status": proof_http.get("status_code"),
            "latency_ms": proof_http.get("latency_ms"),
            "reason": proof_body.get("reason"),
            "outer_route_id_prefix": outer_probe["route_id_prefix"],
            "onward_route_id_prefix": onward_probe["route_id_prefix"],
        })
        all_ok = all_ok and proof_ok

if include_negative and endpoint_results:
    first_endpoint = endpoints[0]
    snapshot_result = request_json(first_endpoint + "/api/discovery/snapshot?limit=8")
    peer = find_self_peer(snapshot_result.get("body") or {}, first_endpoint)
    if peer is not None:
        invalid_probe = build_envelope(descriptor_node_id(peer), 1, "invalid-signature", corrupt=True)
        invalid_http = request_json(
            first_endpoint + "/api/chat/peer/blind-relay",
            method="POST",
            body=invalid_probe["body"],
        )
        invalid_body = invalid_http.get("body") if isinstance(invalid_http.get("body"), dict) else {}
        negative_ok = bool(
            invalid_http.get("status_code") in (400, 401, 403, 422)
            and invalid_body.get("accepted") is not True
        )
        negative_result = {
            "endpoint": first_endpoint,
            "status": "ok" if negative_ok else "failed",
            "response_status": invalid_http.get("status_code"),
            "reason": invalid_body.get("reason"),
            "route_id_prefix": invalid_probe["route_id_prefix"],
        }
        all_ok = all_ok and negative_ok
    else:
        negative_result = {
            "endpoint": first_endpoint,
            "status": "failed",
            "reason": "self_descriptor_not_found",
        }
        all_ok = False

result = {
    "status": "ok" if all_ok else "failed",
    "endpoint_count": len(endpoints),
    "endpoints": endpoint_results,
    "terminal_probes": terminal_results,
    "forward_probes": forward_results,
    "two_hop_requested": two_hop_requested,
    "two_hop_proofs": two_hop_results,
    "negative_probe": negative_result,
    "source": "aeronyx-node.sh fleet-smoke",
    "privacy_boundary": "synthetic opaque blobs only; no user messages, wallet ids, DNS contents, destinations, domains, URLs, packet payloads, client public IPs, or private keys",
}

if not json_only:
    print("fleet_smoke_status={status} endpoints={count}".format(
        status=result["status"],
        count=result["endpoint_count"],
    ))
    for item in endpoint_results:
        print(
            "endpoint_status={status} url={endpoint} valid_peers={valid} stage={stage} latency_ms={latency}".format(
                status=item["status"],
                endpoint=item["endpoint"],
                valid=item.get("valid_peers"),
                stage=item.get("protocol_stage"),
                latency=item.get("status_latency_ms"),
            )
        )
    for item in terminal_results:
        print(
            "terminal_probe={status} url={endpoint} code={code} reason={reason}".format(
                status=item["status"],
                endpoint=item["endpoint"],
                code=item.get("response_status"),
                reason=item.get("reason"),
            )
        )
    for item in forward_results:
        print(
            "forward_probe={status} url={endpoint} next={next_hop} code={code} reason={reason}".format(
                status=item["status"],
                endpoint=item["endpoint"],
                next_hop=item.get("next_hop_prefix"),
                code=item.get("response_status"),
                reason=item.get("reason"),
            )
        )
    for item in two_hop_results:
        print(
            "two_hop_proof={status} entry={entry} middle={middle} terminal={terminal} code={code} reason={reason}".format(
                status=item["status"],
                entry=item["entry_endpoint"],
                middle=item.get("middle_prefix"),
                terminal=item.get("terminal_prefix"),
                code=item.get("response_status"),
                reason=item.get("reason"),
            )
        )
    if negative_result is not None:
        print(
            "negative_probe={status} url={endpoint} code={code} reason={reason}".format(
                status=negative_result["status"],
                endpoint=negative_result["endpoint"],
                code=negative_result.get("response_status"),
                reason=negative_result.get("reason"),
            )
        )

if json_requested or json_only:
    print(json.dumps(result, sort_keys=True))

raise SystemExit(0 if all_ok else 3)
PY
}

run_relay_probe() {
    command -v python3 >/dev/null 2>&1 || die "relay-probe requires python3"
    [ -r "${PEER_CACHE_FILE}" ] || die "Peer cache not readable: ${PEER_CACHE_FILE}"
    [ -r "${SERVER_KEY_FILE}" ] || die "Server key public metadata not readable: ${SERVER_KEY_FILE}"

    local local_status_url local_relay_url
    local_status_url="$(discovery_status_url)"
    local_relay_url="${local_status_url%/api/discovery/status}/api/chat/peer/blind-relay"

    if [ "${JSON_ONLY}" -ne 1 ]; then
        log "Synthetic BlindRelay route probe"
        log "Local relay endpoint: ${local_relay_url}"
        log "Peer cache: ${PEER_CACHE_FILE}"
        log "Privacy boundary: synthetic opaque blob only; no user message, wallet id, DNS, destination, domain, URL, or packet payload"
    fi

    python3 - \
        "${local_status_url}" \
        "${local_relay_url}" \
        "${PEER_CACHE_FILE}" \
        "${SERVER_KEY_FILE}" \
        "${RELAY_PROBE_PEER_PREFIX}" \
        "${RELAY_PROBE_TWO_HOP}" \
        "${JSON}" \
        "${JSON_ONLY}" <<'PY'
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "reason": "missing_python_cryptography",
        "detail": str(exc)[:240],
    }, sort_keys=True))
    raise SystemExit(2)

DOMAIN = b"AeroNyx-BlindRelay-v1"
(
    local_status_url,
    local_relay_url,
    peer_cache_path,
    server_key_path,
    peer_prefix_filter,
    two_hop_requested,
    json_requested,
    json_only,
) = sys.argv[1:9]
two_hop_requested = two_hop_requested == "1"
json_requested = json_requested == "1"
json_only = json_only == "1"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def blind_stats(status):
    return status["peer_store"]["runtime"]["blind_relay"]


def int_or_none(value):
    try:
        return int(value)
    except Exception:
        return None


def route_readiness_summary(status):
    readiness = status.get("discovery_readiness") or {}
    protocol = readiness.get("protocol_foundation") or {}
    story = readiness.get("network_story") or {}
    quorum = readiness.get("peer_quorum") or {}
    route_candidates = (status.get("peer_store") or {}).get("route_candidates") or {}
    planned_paths = route_candidates.get("planned_paths") or {}
    single_hop = planned_paths.get("chat_single_hop") or {}
    two_hop = planned_paths.get("chat_two_hop_onion_ready") or {}
    two_hop_hops = two_hop.get("hops") if isinstance(two_hop.get("hops"), list) else []

    two_hop_prefixes = []
    for hop in two_hop_hops:
        if not isinstance(hop, dict):
            continue
        prefix = hop.get("node_id_prefix")
        capability = hop.get("capability")
        if prefix and capability:
            two_hop_prefixes.append(f"{capability}:{prefix}")

    return {
        "stage": protocol.get("stage") or "unknown",
        "status": protocol.get("status") or "unknown",
        "single_hop_ready": bool(
            single_hop.get("complete")
            or protocol.get("single_hop_relay_ready")
            or story.get("chat_single_hop_ready")
        ),
        "two_hop_ready": bool(
            two_hop.get("complete")
            or protocol.get("two_hop_onion_ready")
            or story.get("chat_two_hop_onion_ready")
        ),
        "routeable_chat_relays": int_or_none(
            story.get("routeable_chat_relays", quorum.get("routeable_chat_relays"))
        ),
        "routeable_onion_middle_hops": int_or_none(
            story.get("routeable_onion_middle_hops", quorum.get("routeable_onion_middle_hops"))
        ),
        "planned_two_hop_count": int_or_none(two_hop.get("hop_count")),
        "planned_two_hop_prefixes": two_hop_prefixes[:3],
        "two_hop_probe_supported": True,
        "two_hop_probe_blocker": None,
    }


def int_delta(after, before, key):
    return int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)


def b64decode_key(value):
    return base64.b64decode(value)


def node_id_prefix(node_id):
    return bytes(node_id).hex()[:8]


def has_capability(capabilities, expected):
    aliases = {
        "chat_relay": {"chatrelay", "chat_relay"},
        "onion_middle": {"onionmiddle", "onion_middle"},
    }.get(expected, {expected})
    return any(str(item).lower() in aliases for item in capabilities or [])


def has_chat_relay(capabilities):
    return has_capability(capabilities, "chat_relay")


def choose_peer(snapshot, self_pub, prefix_filter):
    peers = snapshot.get("peers") if isinstance(snapshot, dict) else []
    if not isinstance(peers, list):
        peers = []
    candidates = []
    for peer in peers:
        if not isinstance(peer, dict):
            continue
        descriptor = peer.get("descriptor")
        if not isinstance(descriptor, dict):
            continue
        node_id = descriptor.get("node_id")
        endpoint = descriptor.get("public_endpoint")
        capabilities = descriptor.get("capabilities")
        if not isinstance(node_id, list) or len(node_id) != 32:
            continue
        try:
            node_id_bytes = bytes(int(item) & 0xFF for item in node_id)
        except Exception:
            continue
        if node_id_bytes == self_pub:
            continue
        if not endpoint or not has_chat_relay(capabilities):
            continue
        prefix = node_id_bytes.hex()[:8]
        if prefix_filter and not prefix.startswith(prefix_filter.lower()):
            continue
        candidates.append((node_id_bytes, str(endpoint).rstrip("/"), prefix, descriptor))
    if not candidates:
        raise RuntimeError("no_routeable_chat_relay_peer")
    candidates.sort(key=lambda item: item[2])
    return candidates[0]


def routeable_peer_candidates(snapshot, self_pub, capability, prefix_filter=""):
    peers = snapshot.get("peers") if isinstance(snapshot, dict) else []
    if not isinstance(peers, list):
        peers = []
    candidates = []
    for peer in peers:
        if not isinstance(peer, dict):
            continue
        descriptor = peer.get("descriptor")
        if not isinstance(descriptor, dict):
            continue
        node_id = descriptor.get("node_id")
        endpoint = descriptor.get("public_endpoint")
        capabilities = descriptor.get("capabilities")
        if not isinstance(node_id, list) or len(node_id) != 32:
            continue
        try:
            node_id_bytes = bytes(int(item) & 0xFF for item in node_id)
        except Exception:
            continue
        if node_id_bytes == self_pub:
            continue
        if not endpoint or not has_capability(capabilities, capability):
            continue
        prefix = node_id_bytes.hex()[:8]
        if prefix_filter and not prefix.startswith(prefix_filter.lower()):
            continue
        candidates.append((node_id_bytes, str(endpoint).rstrip("/"), prefix, descriptor))
    candidates.sort(key=lambda item: item[2])
    return candidates


def choose_two_hop_path(snapshot, self_pub, prefix_filter):
    middles = routeable_peer_candidates(snapshot, self_pub, "onion_middle", prefix_filter)
    terminals = routeable_peer_candidates(snapshot, self_pub, "chat_relay", "")
    for middle in middles:
        for terminal in terminals:
            if terminal[0] != middle[0] and terminal[0] != self_pub:
                return middle, terminal
    raise RuntimeError("needs_three_distinct_routeable_nodes")


def sign_data(route_id, next_hop, ttl, timestamp, blob, private_key):
    payload = (
        DOMAIN
        + route_id
        + next_hop
        + bytes([ttl])
        + int(timestamp).to_bytes(8, "little")
        + hashlib.sha256(blob).digest()
    )
    return private_key.sign(payload)


def signature_tuple(signature):
    return [list(signature[:32]), list(signature[32:])]


def post_json(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=18) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw[:500]}
        return exc.code, parsed


server_key = json.load(open(server_key_path, "r", encoding="utf-8"))
self_pub = b64decode_key(server_key["public_key"])
snapshot = json.load(open(peer_cache_path, "r", encoding="utf-8"))
local_status_before = fetch_json(local_status_url)
two_hop_readiness = route_readiness_summary(local_status_before)
before_local = blind_stats(local_status_before)

previous_private = Ed25519PrivateKey.generate()
previous_pub = previous_private.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)
timestamp = int(time.time())
counter_keys = ["received", "forwarded", "terminal", "rejected", "forward_failed", "no_route"]

if two_hop_requested:
    try:
        middle, terminal = choose_two_hop_path(snapshot, self_pub, peer_prefix_filter.strip())
    except RuntimeError as exc:
        result = {
            "status": "blocked",
            "reason": str(exc),
            "probe_scope": "two_hop_blind_relay_transport",
            "two_hop_readiness": two_hop_readiness,
            "required_distinct_nodes": 3,
            "source": "aeronyx-node.sh relay-probe --two-hop",
            "privacy_boundary": "synthetic opaque blobs only; no user message, wallet id, DNS, destination, domain, URL, or packet payload",
        }
        if not json_only:
            print(
                "relay_probe_status=blocked scope=two_hop_blind_relay_transport reason={reason}".format(
                    **result
                )
            )
            print(
                "relay_probe_two_hop_readiness=stage:{stage} two_hop_ready:{two_hop_ready} onion_middle_hops:{routeable_onion_middle_hops} chat_relays:{routeable_chat_relays} planned_hops:{planned_two_hop_count}".format(
                    **two_hop_readiness
                )
            )
        if json_requested or json_only:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise SystemExit(0)

    middle_node_id, middle_endpoint, middle_prefix, _middle_descriptor = middle
    terminal_node_id, terminal_endpoint, terminal_prefix, _terminal_descriptor = terminal
    middle_status_url = f"{middle_endpoint}/api/discovery/status"
    terminal_status_url = f"{terminal_endpoint}/api/discovery/status"
    before_middle = blind_stats(fetch_json(middle_status_url))
    before_terminal = blind_stats(fetch_json(terminal_status_url))

    outer_route_id = os.urandom(16)
    onward_route_id = os.urandom(16)
    outer_blob = b"synthetic-two-hop-outer:v1:no-user-payload"
    onward_blob = b"synthetic-two-hop-onward:v1:no-user-payload"
    outer_ttl = 2
    onward_ttl = 1
    outer_signature = sign_data(
        outer_route_id,
        middle_node_id,
        outer_ttl,
        timestamp,
        outer_blob,
        previous_private,
    )
    onward_signature = sign_data(
        onward_route_id,
        terminal_node_id,
        onward_ttl,
        timestamp,
        onward_blob,
        previous_private,
    )
    body = {
        "envelope": {
            "route_id": list(outer_route_id),
            "next_hop": list(middle_node_id),
            "ttl": outer_ttl,
            "encrypted_blob": list(outer_blob),
            "timestamp": timestamp,
            "signature": signature_tuple(outer_signature),
        },
        "previous_hop_node_id": list(previous_pub),
        "onward_envelope": {
            "route_id": list(onward_route_id),
            "next_hop": list(terminal_node_id),
            "ttl": onward_ttl,
            "encrypted_blob": list(onward_blob),
            "timestamp": timestamp,
            "signature": signature_tuple(onward_signature),
        },
    }

    response_status, response_body = post_json(local_relay_url, body)
    time.sleep(3)
    after_local = blind_stats(fetch_json(local_status_url))
    after_middle = blind_stats(fetch_json(middle_status_url))
    after_terminal = blind_stats(fetch_json(terminal_status_url))
    local_delta = {key: int_delta(after_local, before_local, key) for key in counter_keys}
    middle_delta = {key: int_delta(after_middle, before_middle, key) for key in counter_keys}
    terminal_delta = {key: int_delta(after_terminal, before_terminal, key) for key in counter_keys}
    ok = (
        response_status == 200
        and response_body.get("accepted") is True
        and response_body.get("forwarded") is True
        and local_delta.get("forwarded") == 1
        and middle_delta.get("forwarded") == 1
        and terminal_delta.get("terminal") == 1
        and local_delta.get("rejected") == 0
        and middle_delta.get("rejected") == 0
        and terminal_delta.get("rejected") == 0
    )
    result = {
        "status": "ok" if ok else "attention",
        "probe_scope": "two_hop_blind_relay_transport",
        "two_hop_readiness": two_hop_readiness,
        "response_status": response_status,
        "response_body": response_body,
        "middle_peer_prefix": middle_prefix,
        "terminal_peer_prefix": terminal_prefix,
        "outer_route_id_prefix": outer_route_id.hex()[:8],
        "onward_route_id_prefix": onward_route_id.hex()[:8],
        "local_delta": local_delta,
        "middle_delta": middle_delta,
        "terminal_delta": terminal_delta,
        "source": "aeronyx-node.sh relay-probe --two-hop",
        "privacy_boundary": "synthetic opaque blobs only; no user message, wallet id, DNS, destination, domain, URL, or packet payload",
    }
else:
    target_node_id, target_endpoint, target_prefix, _descriptor = choose_peer(
        snapshot,
        self_pub,
        peer_prefix_filter.strip(),
    )

    remote_status_url = f"{target_endpoint}/api/discovery/status"
    before_remote = blind_stats(fetch_json(remote_status_url))

    route_id = os.urandom(16)
    ttl = 2
    blob = b"synthetic-blind-relay-probe:v1:no-user-payload"
    signature = sign_data(route_id, target_node_id, ttl, timestamp, blob, previous_private)

    body = {
        "envelope": {
            "route_id": list(route_id),
            "next_hop": list(target_node_id),
            "ttl": ttl,
            "encrypted_blob": list(blob),
            "timestamp": timestamp,
            "signature": signature_tuple(signature),
        },
        "previous_hop_node_id": list(previous_pub),
    }

    response_status, response_body = post_json(local_relay_url, body)
    time.sleep(2)
    after_local = blind_stats(fetch_json(local_status_url))
    after_remote = blind_stats(fetch_json(remote_status_url))

    local_delta = {key: int_delta(after_local, before_local, key) for key in counter_keys}
    remote_delta = {key: int_delta(after_remote, before_remote, key) for key in counter_keys}
    ok = (
        response_status == 200
        and response_body.get("accepted") is True
        and response_body.get("forwarded") is True
        and local_delta.get("forwarded") == 1
        and remote_delta.get("terminal") == 1
        and local_delta.get("rejected") == 0
        and remote_delta.get("rejected") == 0
    )
    result = {
        "status": "ok" if ok else "attention",
        "probe_scope": "single_hop_blind_relay_transport",
        "two_hop_readiness": two_hop_readiness,
        "response_status": response_status,
        "response_body": response_body,
        "target_peer_prefix": target_prefix,
        "route_id_prefix": route_id.hex()[:8],
        "local_delta": local_delta,
        "remote_delta": remote_delta,
        "source": "aeronyx-node.sh relay-probe",
        "privacy_boundary": "synthetic opaque blob only; no user message, wallet id, DNS, destination, domain, URL, or packet payload",
    }

if not json_only:
    print(
        "relay_probe_status={status} target_peer_prefix={peer} response_status={code}".format(
            status=result["status"],
            peer=result.get("target_peer_prefix", result.get("middle_peer_prefix", "none")),
            code=response_status,
        )
    )
    print(
        "relay_probe_scope={scope} two_hop_requested={two_hop}".format(
            scope=result["probe_scope"],
            two_hop=str(two_hop_requested).lower(),
        )
    )
    print(
        "relay_probe_two_hop_readiness=stage:{stage} two_hop_ready:{two_hop_ready} onion_middle_hops:{routeable_onion_middle_hops} chat_relays:{routeable_chat_relays} planned_hops:{planned_two_hop_count}".format(
            **two_hop_readiness
        )
    )
    print(
        "relay_probe_local_delta=received:{received} forwarded:{forwarded} terminal:{terminal} rejected:{rejected} forward_failed:{forward_failed} no_route:{no_route}".format(
            **local_delta
        )
    )
    if two_hop_requested:
        print(
            "relay_probe_middle_delta=received:{received} forwarded:{forwarded} terminal:{terminal} rejected:{rejected} forward_failed:{forward_failed} no_route:{no_route}".format(
                **middle_delta
            )
        )
        print(
            "relay_probe_terminal_delta=received:{received} forwarded:{forwarded} terminal:{terminal} rejected:{rejected} forward_failed:{forward_failed} no_route:{no_route}".format(
                **terminal_delta
            )
        )
    else:
        print(
            "relay_probe_remote_delta=received:{received} forwarded:{forwarded} terminal:{terminal} rejected:{rejected} forward_failed:{forward_failed} no_route:{no_route}".format(
                **remote_delta
            )
        )

if json_requested or json_only:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

raise SystemExit(0 if ok else 3)
PY
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print $1}'
        return
    fi
    die "sha256sum or shasum is required for binary promotion"
}

validate_promote_health_polling() {
    case "${PROMOTE_HEALTH_RETRIES}" in
        ''|*[!0-9]*|0) die "AERONYX_PROMOTE_HEALTH_RETRIES must be a positive integer" ;;
    esac
    case "${PROMOTE_HEALTH_DELAY}" in
        ''|*[!0-9]*|0) die "AERONYX_PROMOTE_HEALTH_DELAY must be a positive integer" ;;
    esac
}

promotion_backup_source() {
    local target_binary="$1"
    local main_pid
    main_pid="$(systemctl show "${SERVICE_NAME}" --property=MainPID --value 2>/dev/null || true)"
    case "${main_pid}" in
        ''|*[!0-9]*|0)
            printf '%s\n' "${target_binary}"
            ;;
        *)
            if [ -r "/proc/${main_pid}/exe" ]; then
                printf '%s\n' "/proc/${main_pid}/exe"
            else
                printf '%s\n' "${target_binary}"
            fi
            ;;
    esac
}

wait_for_promoted_service_health() {
    local attempt
    for ((attempt = 1; attempt <= PROMOTE_HEALTH_RETRIES; attempt++)); do
        if systemctl is-active --quiet "${SERVICE_NAME}" \
            && curl -fsS --max-time 3 http://127.0.0.1:8421/api/vpn/health >/dev/null 2>&1; then
            ok "${SERVICE_NAME} health endpoint ready after ${attempt} attempt(s)"
            return 0
        fi
        if [ "${attempt}" -lt "${PROMOTE_HEALTH_RETRIES}" ]; then
            sleep "${PROMOTE_HEALTH_DELAY}"
        fi
    done
    return 1
}

rollback_promoted_binary() {
    local backup="$1"
    local target_binary="$2"
    local rollback_target="${target_binary}.rollback.$$"

    warn "Promoted binary failed readiness; restoring the previous running release"
    systemctl stop "${SERVICE_NAME}" >/dev/null 2>&1 || true
    install -m 0755 "${backup}" "${rollback_target}"
    mv -f "${rollback_target}" "${target_binary}"
    if ! systemctl start "${SERVICE_NAME}"; then
        return 1
    fi
    wait_for_promoted_service_health
}

confirm_promote_binary() {
    [ "${YES}" -eq 1 ] && return
    [ "${DRY_RUN}" -eq 1 ] && return

    cat <<CONFIRM

This will replace the active aeronyx-server binary, restart ${SERVICE_NAME},
and keep a timestamped backup of the previous binary.
Type PROMOTE BINARY to continue. Press Enter to stop safely.
CONFIRM
    printf 'Confirm: '
    local confirmation
    IFS= read -r confirmation || confirmation=""
    [ "${confirmation}" = "PROMOTE BINARY" ] || die "Binary promotion stopped before modifying the active binary."
}

run_promote_binary() {
    validate_service_name
    validate_promote_health_polling

    [ -n "${PROMOTE_BINARY_PATH}" ] || die "promote-binary requires --binary PATH"
    [ -f "${PROMOTE_BINARY_PATH}" ] || die "Staged binary not found: ${PROMOTE_BINARY_PATH}"
    command -v systemctl >/dev/null 2>&1 || die "promote-binary requires systemctl"
    command -v curl >/dev/null 2>&1 || die "promote-binary requires curl"

    local target_binary
    target_binary="${REPO_DIR}/target/release/aeronyx-server"
    [ -x "${target_binary}" ] || die "Active binary is missing or not executable: ${target_binary}"

    chmod 755 "${PROMOTE_BINARY_PATH}" 2>/dev/null || true
    [ -x "${PROMOTE_BINARY_PATH}" ] || die "Staged binary is not executable: ${PROMOTE_BINARY_PATH}"

    local staged_sha current_sha
    staged_sha="$(sha256_file "${PROMOTE_BINARY_PATH}")"
    current_sha="$(sha256_file "${target_binary}")"

    if [ -n "${EXPECTED_SHA256}" ] && [ "${staged_sha}" != "${EXPECTED_SHA256}" ]; then
        die "Staged binary SHA-256 mismatch. expected=${EXPECTED_SHA256} actual=${staged_sha}"
    fi

    local active_sessions
    active_sessions="$(active_sessions_count)"
    log "Current active_sessions=${active_sessions}"
    log "Current binary sha256=${current_sha}"
    log "Staged binary sha256=${staged_sha}"

    if [ "${FORCE}" -eq 1 ] && [ "${YES}" -ne 1 ]; then
        die "Refusing forced binary promotion without --yes"
    fi

    if [ "${DRY_RUN}" -eq 0 ] && [ "${FORCE}" -ne 1 ]; then
        case "${active_sessions}" in
            ''|*[!0-9]*)
                die "Cannot prove active sessions are drained. Re-run after health is reachable, or use --force --yes during maintenance."
                ;;
            0)
                ;;
            *)
                die "Refusing binary promotion while active_sessions=${active_sessions}. Drain sessions first."
                ;;
        esac
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf 'would_promote_binary=%s\n' "${PROMOTE_BINARY_PATH}"
        printf 'would_replace_binary=%s\n' "${target_binary}"
        printf 'active_sessions=%s\n' "${active_sessions}"
        printf 'current_sha256=%s\n' "${current_sha}"
        printf 'staged_sha256=%s\n' "${staged_sha}"
        printf 'would_restart_service=%s\n' "${SERVICE_NAME}"
        return
    fi

    confirm_promote_binary

    if ! "${PROMOTE_BINARY_PATH}" validate -c "${CONFIG_FILE}"; then
        die "Staged binary cannot validate config: ${CONFIG_FILE}"
    fi

    local timestamp backup backup_source tmp_target
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup="${target_binary}.bak.${timestamp}.promote"
    tmp_target="${target_binary}.tmp.${timestamp}.promote"
    backup_source="$(promotion_backup_source "${target_binary}")"

    # [NODE-BINARY-PROMOTION 2026-07-23 by Codex] Cargo may have already
    # replaced the on-disk path while systemd still runs the previous mapped
    # executable. Back up the actual live release so rollback is truthful.
    cp --dereference -p "${backup_source}" "${backup}"
    ok "Backup created: ${backup}"

    install -m 0755 "${PROMOTE_BINARY_PATH}" "${tmp_target}"
    mv -f "${tmp_target}" "${target_binary}"
    ok "Promoted staged binary to ${target_binary}"

    log "Restarting ${SERVICE_NAME}; waiting up to $((PROMOTE_HEALTH_RETRIES * PROMOTE_HEALTH_DELAY)) seconds for cold-start health"
    if ! systemctl restart "${SERVICE_NAME}" || ! wait_for_promoted_service_health; then
        if rollback_promoted_binary "${backup}" "${target_binary}"; then
            die "Promoted binary failed readiness and the previous release was restored."
        fi
        die "Promoted binary failed readiness and rollback did not recover ${SERVICE_NAME}."
    fi
    ok "${SERVICE_NAME} restarted and passed cold-start health"

    show_discovery_readiness
    run_health
}

confirm_chat_relay_change() {
    local action="$1"
    [ "${YES}" -eq 1 ] && return
    [ "${DRY_RUN}" -eq 1 ] && return

    cat <<CONFIRM

This will update ${CONFIG_FILE} and create a timestamped backup.
It does not expose private keys, payloads, DNS contents, destinations, or user traffic.
Type ${action} to continue. Press Enter to stop safely.
CONFIRM
    printf 'Confirm: '
    local confirmation
    IFS= read -r confirmation || confirmation=""
    [ "${confirmation}" = "${action}" ] || die "ChatRelay config change stopped before modifying ${CONFIG_FILE}."
}

run_chat_relay_config() {
    validate_service_name

    if [ "${CHAT_RELAY_ENABLE}" -eq "${CHAT_RELAY_DISABLE}" ]; then
        die "chat-relay requires exactly one of --enable-chat-relay or --disable-chat-relay"
    fi
    command -v python3 >/dev/null 2>&1 || die "chat-relay requires python3"
    [ -f "${CONFIG_FILE}" ] || die "Config file not found: ${CONFIG_FILE}"

    local target="false"
    local confirm_word="DISABLE CHATRELAY"
    if [ "${CHAT_RELAY_ENABLE}" -eq 1 ]; then
        target="true"
        confirm_word="ENABLE CHATRELAY"
    fi

    local active_sessions
    active_sessions="$(active_sessions_count)"
    log "Current active_sessions=${active_sessions}"
    if [ "${RESTART_AFTER_CONFIG}" -eq 1 ] && [ "${YES}" -ne 1 ]; then
        case "${active_sessions}" in
            ''|*[!0-9]*)
                die "Cannot prove active sessions are drained. Re-run with --yes during a maintenance window if restart is intentional."
                ;;
            0)
                ;;
            *)
                die "Refusing restart while active_sessions=${active_sessions}. Drain sessions or re-run with --yes during maintenance."
                ;;
        esac
    fi

    confirm_chat_relay_change "${confirm_word}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        python3 - "${CONFIG_FILE}" "${target}" "1" <<'PY'
import re
import sys

path, target, dry_run = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
lines = open(path, "r", encoding="utf-8").read().splitlines(True)
section_start = None
section_end = len(lines)
for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "[memchain.chat_relay]":
        section_start = idx
        continue
    if section_start is not None and idx > section_start and re.match(r"^\s*\[[^]]+\]\s*$", line):
        section_end = idx
        break

enabled_found = False
if section_start is not None:
    for line in lines[section_start + 1:section_end]:
        if re.match(r"^\s*enabled\s*=", line):
            enabled_found = True
            break

print(f"would_set_memchain_chat_relay_enabled={target}")
print(f"would_create_memchain_chat_relay_section={str(section_start is None).lower()}")
print(f"would_insert_enabled_key={str(section_start is not None and not enabled_found).lower()}")
PY
        return
    fi

    [ -w "${CONFIG_FILE}" ] || die "Config file is not writable. Re-run with sudo: ${CONFIG_FILE}"

    local timestamp backup
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup="${CONFIG_FILE}.bak.${timestamp}.chat_relay"
    cp -p "${CONFIG_FILE}" "${backup}"
    ok "Backup created: ${backup}"

    python3 - "${CONFIG_FILE}" "${target}" <<'PY'
import os
import re
import sys
import tempfile

path, target = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines(True)

section_start = None
section_end = len(lines)
for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "[memchain.chat_relay]":
        section_start = idx
        section_end = len(lines)
        continue
    if section_start is not None and idx > section_start and re.match(r"^\s*\[[^]]+\]\s*$", line):
        section_end = idx
        break

if section_start is None:
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend([
        "[memchain.chat_relay]\n",
        f"enabled = {target}\n",
        "offline_ttl_secs = 259200\n",
        "max_pending_per_wallet = 500\n",
        'db_path = "/var/lib/aeronyx/chat_pending.db"\n',
        "max_message_size = 65536\n",
        "max_blob_size = 10485760\n",
        "max_blobs_per_receiver = 50\n",
        "cleanup_interval_secs = 60\n",
        "dedup_lru_capacity = 10000\n",
        "expired_notification_ttl_secs = 604800\n",
    ])
else:
    enabled_idx = None
    for idx in range(section_start + 1, section_end):
        if re.match(r"^\s*enabled\s*=", lines[idx]):
            enabled_idx = idx
            break
    if enabled_idx is None:
        lines.insert(section_start + 1, f"enabled = {target}\n")
    else:
        indent = re.match(r"^(\s*)", lines[enabled_idx]).group(1)
        lines[enabled_idx] = f"{indent}enabled = {target}\n"

directory = os.path.dirname(path) or "."
fd, tmp_path = tempfile.mkstemp(prefix=".aeronyx-chat-relay.", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    os.replace(tmp_path, path)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
PY

    local binary
    binary="${REPO_DIR}/target/release/aeronyx-server"
    if [ ! -x "${binary}" ]; then
        binary="aeronyx-server"
    fi
    if command -v "${binary}" >/dev/null 2>&1 || [ -x "${binary}" ]; then
        if ! "${binary}" validate -c "${CONFIG_FILE}"; then
            cp -p "${backup}" "${CONFIG_FILE}"
            die "Config validation failed; restored ${CONFIG_FILE} from ${backup}"
        fi
    else
        warn "aeronyx-server binary not found; config was updated but not validated"
    fi

    ok "ChatRelay enabled=${target} in ${CONFIG_FILE}"
    if [ "${RESTART_AFTER_CONFIG}" -eq 1 ]; then
        log "Restarting ${SERVICE_NAME}"
        systemctl restart "${SERVICE_NAME}"
        systemctl is-active "${SERVICE_NAME}" >/dev/null
        ok "${SERVICE_NAME} restarted"
        show_discovery_readiness
    else
        warn "Service not restarted. Run with --restart during maintenance, or restart ${SERVICE_NAME} manually."
    fi
}

confirm_onion_middle_change() {
    local action="$1"
    [ "${YES}" -eq 1 ] && return
    [ "${DRY_RUN}" -eq 1 ] && return

    cat <<CONFIRM

This will update ${CONFIG_FILE} and create a timestamped backup.
OnionMiddle is a no-exit encrypted relay role. It advertises only aggregate
node capability for future multi-hop path planning and must not expose private
keys, payloads, DNS contents, destinations, wallet-level traffic, or user data.
Type ${action} to continue. Press Enter to stop safely.
CONFIRM
    printf 'Confirm: '
    local confirmation
    IFS= read -r confirmation || confirmation=""
    [ "${confirmation}" = "${action}" ] || die "OnionMiddle config change stopped before modifying ${CONFIG_FILE}."
}

run_onion_middle_config() {
    validate_service_name

    if [ "${ONION_MIDDLE_ENABLE}" -eq "${ONION_MIDDLE_DISABLE}" ]; then
        die "onion-middle requires exactly one of --enable-onion-middle or --disable-onion-middle"
    fi
    command -v python3 >/dev/null 2>&1 || die "onion-middle requires python3"
    [ -f "${CONFIG_FILE}" ] || die "Config file not found: ${CONFIG_FILE}"

    local target="false"
    local confirm_word="DISABLE ONIONMIDDLE"
    if [ "${ONION_MIDDLE_ENABLE}" -eq 1 ]; then
        target="true"
        confirm_word="ENABLE ONIONMIDDLE"
    fi

    local active_sessions
    active_sessions="$(active_sessions_count)"
    log "Current active_sessions=${active_sessions}"
    if [ "${RESTART_AFTER_CONFIG}" -eq 1 ] && [ "${YES}" -ne 1 ]; then
        case "${active_sessions}" in
            ''|*[!0-9]*)
                die "Cannot prove active sessions are drained. Re-run with --yes during a maintenance window if restart is intentional."
                ;;
            0)
                ;;
            *)
                die "Refusing restart while active_sessions=${active_sessions}. Drain sessions or re-run with --yes during maintenance."
                ;;
        esac
    fi

    confirm_onion_middle_change "${confirm_word}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        python3 - "${CONFIG_FILE}" "${target}" <<'PY'
import re
import sys

path, target = sys.argv[1], sys.argv[2]
lines = open(path, "r", encoding="utf-8").read().splitlines(True)
section_start = None
section_end = len(lines)
for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "[discovery]":
        section_start = idx
        section_end = len(lines)
        continue
    if section_start is not None and idx > section_start and re.match(r"^\s*\[[^]]+\]\s*$", line):
        section_end = idx
        break

key_found = False
if section_start is not None:
    for line in lines[section_start + 1:section_end]:
        if re.match(r"^\s*advertise_onion_middle\s*=", line):
            key_found = True
            break

print(f"would_set_discovery_advertise_onion_middle={target}")
print(f"would_create_discovery_section={str(section_start is None).lower()}")
print(f"would_insert_advertise_onion_middle_key={str(section_start is not None and not key_found).lower()}")
PY
        return
    fi

    [ -w "${CONFIG_FILE}" ] || die "Config file is not writable. Re-run with sudo: ${CONFIG_FILE}"

    local timestamp backup
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup="${CONFIG_FILE}.bak.${timestamp}.onion_middle"
    cp -p "${CONFIG_FILE}" "${backup}"
    ok "Backup created: ${backup}"

    python3 - "${CONFIG_FILE}" "${target}" <<'PY'
import os
import re
import sys
import tempfile

path, target = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines(True)

section_start = None
section_end = len(lines)
for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "[discovery]":
        section_start = idx
        section_end = len(lines)
        continue
    if section_start is not None and idx > section_start and re.match(r"^\s*\[[^]]+\]\s*$", line):
        section_end = idx
        break

if section_start is None:
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend([
        "[discovery]\n",
        f"advertise_onion_middle = {target}\n",
    ])
else:
    key_idx = None
    for idx in range(section_start + 1, section_end):
        if re.match(r"^\s*advertise_onion_middle\s*=", lines[idx]):
            key_idx = idx
            break
    if key_idx is None:
        lines.insert(section_end, f"advertise_onion_middle = {target}\n")
    else:
        indent = re.match(r"^(\s*)", lines[key_idx]).group(1)
        lines[key_idx] = f"{indent}advertise_onion_middle = {target}\n"

directory = os.path.dirname(path) or "."
fd, tmp_path = tempfile.mkstemp(prefix=".aeronyx-onion-middle.", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    os.replace(tmp_path, path)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
PY

    local binary
    binary="${REPO_DIR}/target/release/aeronyx-server"
    if [ ! -x "${binary}" ]; then
        binary="aeronyx-server"
    fi
    if command -v "${binary}" >/dev/null 2>&1 || [ -x "${binary}" ]; then
        if ! "${binary}" validate -c "${CONFIG_FILE}"; then
            cp -p "${backup}" "${CONFIG_FILE}"
            die "Config validation failed; restored ${CONFIG_FILE} from ${backup}"
        fi
    else
        warn "aeronyx-server binary not found; config was updated but not validated"
    fi

    ok "OnionMiddle advertise_onion_middle=${target} in ${CONFIG_FILE}"
    if [ "${RESTART_AFTER_CONFIG}" -eq 1 ]; then
        log "Restarting ${SERVICE_NAME}"
        systemctl restart "${SERVICE_NAME}"
        systemctl is-active "${SERVICE_NAME}" >/dev/null
        ok "${SERVICE_NAME} restarted"
        show_discovery_readiness
    else
        warn "Service not restarted. Run with --restart during maintenance, or restart ${SERVICE_NAME} manually."
    fi
}

read_optional_registration_code() {
    if [ -n "${REGISTRATION_CODE}" ]; then
        return
    fi

    printf 'Registration code (leave empty to skip): '
    IFS= read -r REGISTRATION_CODE || REGISTRATION_CODE=""
}

interactive_menu() {
    cat <<'MENU'
AeroNyx Node Operator

1) Preview install plan
2) Quickstart: preview, install/register/start, status, healthcheck
3) Upgrade build without restart
4) Upgrade and restart when safe
5) Health check
6) Service status
7) Recent logs
8) Refresh network/IP pool
9) Exit
MENU
    printf 'Select an action: '
    local choice
    IFS= read -r choice || choice="9"

    case "${choice}" in
        1)
            read_optional_registration_code
            if [ -n "${REGISTRATION_CODE}" ]; then
                EXTRA_ARGS+=("--quick")
            fi
            run_plan
            ;;
        2)
            read_optional_registration_code
            EXTRA_ARGS+=("--quick")
            run_quickstart
            ;;
        3)
            NO_RESTART=1
            run_upgrade
            ;;
        4)
            run_upgrade
            ;;
        5)
            run_health
            ;;
        6)
            show_status
            ;;
        7)
            show_logs
            ;;
        8)
            printf 'New AeroNyx privacy protocol IP pool CIDR: '
            IFS= read -r SET_VPN_CIDR
            run_network
            ;;
        9|"")
            ok "No action selected"
            ;;
        *)
            die "Unknown menu selection: ${choice}"
            ;;
    esac
}

main() {
    parse_args "$@"

    case "${COMMAND}" in
        help|-h|--help)
            usage
            ;;
        menu)
            interactive_menu
            ;;
        plan)
            run_plan
            ;;
        quickstart)
            run_quickstart
            ;;
        install)
            run_install
            ;;
        upgrade)
            run_upgrade
            ;;
        health|healthcheck|doctor)
            run_health
            ;;
        status)
            show_status
            ;;
        chat-relay)
            run_chat_relay_config
            ;;
        onion-middle)
            run_onion_middle_config
            ;;
        relay-probe)
            run_relay_probe
            ;;
        fleet-smoke)
            run_fleet_smoke
            ;;
        refresh-bootstrap)
            run_refresh_bootstrap
            ;;
        fleet-drift-check)
            run_fleet_drift_check
            ;;
        promote-binary)
            run_promote_binary
            ;;
        logs)
            show_logs
            ;;
        network)
            run_network
            ;;
        *)
            usage >&2
            die "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"
