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
# - Add a guarded ChatRelay config helper so operators and AI assistants can
#   enable or disable blind relay advertisement with backup, validation,
#   active-session warning, and optional restart.
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
#   status, logs, doctor, and network maintenance.
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
RESTART_AFTER_CONFIG=0
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
  doctor     Alias for health.
  status     Show service, local endpoints, discovery readiness, upgrade state, and next action.
  chat-relay Enable or disable blind ChatRelay capability in server.toml.
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

Examples:
  ./deploy/node/aeronyx-node.sh plan --quick --registration-code NYX-1234-ABCDE
  ./deploy/node/aeronyx-node.sh quickstart --quick --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh install --quick --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh upgrade --no-restart
  ./deploy/node/aeronyx-node.sh health --json
  ./deploy/node/aeronyx-node.sh status
  sudo ./deploy/node/aeronyx-node.sh chat-relay --enable-chat-relay --restart
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
            --restart) RESTART_AFTER_CONFIG=1; shift ;;
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

show_status() {
    validate_service_name

    log "Service status: ${SERVICE_NAME}"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --no-pager --lines=20 status "${SERVICE_NAME}" || true
    else
        warn "systemctl not found"
    fi

    if command -v curl >/dev/null 2>&1; then
        log "Local privacy protocol health endpoint"
        curl -fsS http://127.0.0.1:8421/api/vpn/health 2>/dev/null || warn "Local /api/vpn/health is not reachable"
        printf '\n'
        log "Local operator status endpoint"
        curl -fsS http://127.0.0.1:8421/api/operator/status 2>/dev/null || warn "Local /api/operator/status is not reachable"
        printf '\n'
    else
        warn "curl not found"
    fi

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
        health|doctor)
            run_health
            ;;
        status)
            show_status
            ;;
        chat-relay)
            run_chat_relay_config
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
