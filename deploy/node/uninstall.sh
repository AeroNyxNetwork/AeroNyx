#!/usr/bin/env bash
# ============================================
# File: deploy/node/uninstall.sh
# ============================================
# Creation Reason:
# - Provide a safe rollback/removal path for AeroNyx Rust privacy node service
#   management without risking accidental deletion of node identity or operator
#   registration state.
#
# Modification Reason:
# - Initial production node uninstall script for commercial operations.
#
# Main Functionality:
# - Stops and disables the systemd service.
# - Removes the installed systemd unit when requested.
# - Preserves /etc/aeronyx and /var/lib/aeronyx by default.
# - Supports explicit purge with typed confirmation.
#
# Dependencies:
# - systemd service installed by deploy/node/install.sh.
# - /etc/aeronyx/server.toml, server_key.json, node_info.json.
# - /var/lib/aeronyx runtime state.
#
# Main Logical Flow:
# 1. Parse flags and require root.
# 2. Stop/disable the systemd service.
# 3. Optionally remove service unit and, only with confirmation, purge state.
#
# Important Note for Next Developer:
# - Default behavior must preserve node identity and registration data.
# - Do not delete /etc/aeronyx or /var/lib/aeronyx unless --purge is used and
#   the operator types the exact confirmation string.
# - This script is Linux/systemd only; mobile/desktop clients are unaffected.
#
# Last Modified:
# v1.0.0-node-deploy - Added safe production node uninstall script.
# ============================================

set -euo pipefail

SERVICE_NAME="aeronyx-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CONFIG_DIR="/etc/aeronyx"
STATE_DIR="/var/lib/aeronyx"
REMOVE_UNIT=1
PURGE=0
DRY_RUN=0
YES=0

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage:
  sudo ./deploy/node/uninstall.sh [OPTIONS]

Options:
  --service NAME       systemd service name. Default: aeronyx-server
  --keep-unit          Stop/disable service but keep the systemd unit file.
  --purge              Also remove /etc/aeronyx and /var/lib/aeronyx after confirmation.
  --yes                Non-interactive confirmation for --purge.
  --dry-run            Print actions without changing the host.
  -h, --help           Show this help.

Default behavior:
  - stop service
  - disable service
  - remove /etc/systemd/system/aeronyx-server.service
  - preserve /etc/aeronyx and /var/lib/aeronyx
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --service) SERVICE_NAME="${2:?missing value}"; SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"; shift 2 ;;
        --keep-unit) REMOVE_UNIT=0; shift ;;
        --purge) PURGE=1; shift ;;
        --yes) YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

run() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] %s\n' "$*"
    else
        "$@"
    fi
}

require_root() {
    [ "$(id -u)" -eq 0 ] || die "Please run as root, for example: sudo $0"
}

require_linux_systemd() {
    [ "$(uname -s)" = "Linux" ] || die "uninstall.sh supports Linux production nodes only."
    command -v systemctl >/dev/null 2>&1 || die "systemctl is required."
}

stop_service() {
    if systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend 2>/dev/null | grep -q "${SERVICE_NAME}.service"; then
        log "Stopping ${SERVICE_NAME}"
        run systemctl stop "${SERVICE_NAME}" || warn "systemctl stop failed or service was already stopped"
        log "Disabling ${SERVICE_NAME}"
        run systemctl disable "${SERVICE_NAME}" || warn "systemctl disable failed or service was not enabled"
    else
        ok "systemd unit is not installed: ${SERVICE_NAME}"
    fi
}

remove_unit() {
    [ "${REMOVE_UNIT}" -eq 1 ] || { ok "systemd unit preserved"; return; }

    if [ -f "${SERVICE_FILE}" ]; then
        log "Removing systemd unit: ${SERVICE_FILE}"
        run rm -f "${SERVICE_FILE}"
        run systemctl daemon-reload
    else
        ok "systemd unit file already absent: ${SERVICE_FILE}"
    fi
}

confirm_purge() {
    [ "${PURGE}" -eq 1 ] || return 1

    if [ "${YES}" -eq 1 ]; then
        return 0
    fi

    printf '\n'
    warn "Purge will delete node config, private server key, registration data, and runtime state:"
    warn "  ${CONFIG_DIR}"
    warn "  ${STATE_DIR}"
    printf 'Type DELETE-AERONYX-NODE to continue: '
    local answer
    read -r answer
    [ "${answer}" = "DELETE-AERONYX-NODE" ]
}

purge_state() {
    if confirm_purge; then
        log "Purging node state"
        run rm -rf "${CONFIG_DIR}" "${STATE_DIR}"
    elif [ "${PURGE}" -eq 1 ]; then
        die "Purge confirmation failed; node state preserved."
    else
        ok "Node state preserved: ${CONFIG_DIR}, ${STATE_DIR}"
    fi
}

main() {
    require_root
    require_linux_systemd
    stop_service
    remove_unit
    purge_state
    ok "Uninstall flow complete."
}

main "$@"
