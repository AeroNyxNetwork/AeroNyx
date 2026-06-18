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
- Show the local structured upgrade status file from `status` so operators and
  AI assistants can see staged/failed upgrade state without scraping logs.
# - Initial production entrypoint for ordinary node operators. This script
#   delegates to existing deployment scripts instead of duplicating their
#   host-writing logic.
# - Clarify that this script is repository-local and is obtained from the
#   open-source AeroNyx Rust repository, not from the host operating system.
#
# Main Functionality:
# - Offers a guided interactive menu when no command is provided.
# - Provides command aliases for plan, install, upgrade, health, status, logs,
#   doctor, and network maintenance.
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
  install    Install/register/start an AeroNyx privacy protocol node.
  upgrade    Pull/build/validate/upgrade the Rust node with safety checks.
  health     Run the node health check. Use --json or --json-only for tooling.
  doctor     Alias for health.
  status     Show systemd service status and local runtime endpoints.
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

Examples:
  ./deploy/node/aeronyx-node.sh plan --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh install --quick --registration-code NYX-1234-ABCDE
  sudo ./deploy/node/aeronyx-node.sh upgrade --no-restart
  ./deploy/node/aeronyx-node.sh health --json
  ./deploy/node/aeronyx-node.sh status
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
2) Install or register node
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
            run_plan
            ;;
        2)
            read_optional_registration_code
            EXTRA_ARGS+=("--quick")
            run_install
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
