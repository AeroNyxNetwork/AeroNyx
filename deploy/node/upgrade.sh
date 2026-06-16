#!/usr/bin/env bash
# ============================================
# File: deploy/node/upgrade.sh
# ============================================
# Creation Reason:
# - Provide a safe, repeatable upgrade path for production AeroNyx Rust privacy
#   nodes without requiring manual git/build/systemd commands.
#
# Modification Reason:
# - Initial production node upgrade script with active-session protection.
#
# Main Functionality:
# - Pulls the configured branch.
# - Builds aeronyx-server release binary.
# - Validates /etc/aeronyx/server.toml.
# - Checks active VPN sessions before restart.
# - Restarts systemd service and verifies post-upgrade health.
#
# Dependencies:
# - deploy/node/healthcheck.sh
# - crates/aeronyx-server/src/main.rs validate/start commands
# - systemd unit installed by deploy/node/install.sh
#
# Main Logical Flow:
# 1. Update repo from Git.
# 2. Build and validate the release binary.
# 3. Restart only when no active sessions are present, unless --force is used.
#
# Important Note for Next Developer:
# - Do not remove active-session protection. Commercial VPN users should not be
#   disconnected by routine upgrades unless the operator explicitly forces it.
# - Do not overwrite /etc/aeronyx/server.toml during upgrades.
# - Keep this script compatible with current and older installed service units.
#
# Last Modified:
# v1.0.0-node-deploy - Added production node upgrade script.
# ============================================

set -euo pipefail

REPO_DIR="/opt/aeronyx/AeroNyx"
BRANCH="main"
CONFIG_FILE="/etc/aeronyx/server.toml"
SERVICE_NAME="aeronyx-server"
FORCE=0
DRY_RUN=0
SKIP_PULL=0

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage:
  sudo ./deploy/node/upgrade.sh [OPTIONS]

Options:
  --repo-dir PATH     Repository path. Default: /opt/aeronyx/AeroNyx
  --branch NAME       Branch/ref to pull. Default: main
  --config PATH       Config path. Default: /etc/aeronyx/server.toml
  --service NAME      systemd service name. Default: aeronyx-server
  --force             Restart even when active VPN sessions exist.
  --skip-pull         Build the currently checked out source.
  --dry-run           Print actions without changing the host.
  -h, --help          Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --branch) BRANCH="${2:?missing value}"; shift 2 ;;
        --config) CONFIG_FILE="${2:?missing value}"; shift 2 ;;
        --service) SERVICE_NAME="${2:?missing value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --skip-pull) SKIP_PULL=1; shift ;;
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

ensure_cargo_path() {
    if command -v cargo >/dev/null 2>&1; then
        return
    fi

    if [ -x "${HOME}/.cargo/bin/cargo" ]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
        return
    fi

    die "cargo not found. Install Rust or run deploy/node/install.sh first."
}

active_sessions() {
    if ! command -v curl >/dev/null 2>&1; then
        printf 'unknown'
        return
    fi

    curl -fsS --max-time 3 http://127.0.0.1:8421/api/vpn/health 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("active_sessions", "unknown"))' 2>/dev/null \
        || printf 'unknown'
}

backup_current_binary() {
    local binary backup_dir stamp
    binary="${REPO_DIR}/target/release/aeronyx-server"
    [ -f "${binary}" ] || return

    backup_dir="/var/lib/aeronyx/releases"
    stamp="$(date -u +%Y%m%d_%H%M%S)"
    run mkdir -p "${backup_dir}"
    run cp "${binary}" "${backup_dir}/aeronyx-server.${stamp}"
    ok "Current binary backed up to ${backup_dir}/aeronyx-server.${stamp}"
}

update_source() {
    [ "${SKIP_PULL}" -eq 0 ] || { ok "Git pull skipped"; return; }

    log "Updating source from origin/${BRANCH}"
    run git -C "${REPO_DIR}" fetch origin "${BRANCH}"
    run git -C "${REPO_DIR}" checkout "${BRANCH}"
    run git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
}

build_release() {
    log "Building release binary"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] cd %s && cargo build -p aeronyx-server --release\n' "${REPO_DIR}"
    else
        (
            cd "${REPO_DIR}"
            cargo build -p aeronyx-server --release
        )
    fi
}

validate_config() {
    local binary
    binary="${REPO_DIR}/target/release/aeronyx-server"
    [ "${DRY_RUN}" -eq 1 ] || [ -x "${binary}" ] || die "Binary not found: ${binary}"

    log "Validating config: ${CONFIG_FILE}"
    run "${binary}" validate -c "${CONFIG_FILE}"
}

restart_service() {
    local sessions
    sessions="$(active_sessions)"

    if [ "${sessions}" != "unknown" ] && [ "${sessions}" -gt 0 ] 2>/dev/null && [ "${FORCE}" -ne 1 ]; then
        die "Active VPN sessions detected (${sessions}). Re-run with --force or drain sessions first."
    fi

    if [ "${sessions}" = "unknown" ]; then
        warn "Could not read active session count; continuing because health endpoint may be unavailable before restart."
    else
        ok "Active VPN sessions before restart: ${sessions}"
    fi

    log "Restarting ${SERVICE_NAME}"
    run systemctl daemon-reload
    run systemctl restart "${SERVICE_NAME}"
    run systemctl is-active "${SERVICE_NAME}"
}

run_healthcheck() {
    local checker="${REPO_DIR}/deploy/node/healthcheck.sh"
    if [ -x "${checker}" ]; then
        run "${checker}" --repo-dir "${REPO_DIR}" --config "${CONFIG_FILE}" --service "${SERVICE_NAME}"
    else
        warn "Healthcheck script not executable or missing: ${checker}"
    fi
}

main() {
    require_root
    ensure_cargo_path
    [ -d "${REPO_DIR}/.git" ] || die "Repository not found: ${REPO_DIR}"
    backup_current_binary
    update_source
    build_release
    validate_config
    restart_service
    run_healthcheck
}

main "$@"
