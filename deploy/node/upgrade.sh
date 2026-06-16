#!/usr/bin/env bash
# ============================================
# File: deploy/node/upgrade.sh
# ============================================
# Creation Reason:
# - Provide a safe, repeatable upgrade path for production AeroNyx Rust privacy
#   nodes without requiring manual git/build/systemd commands.
#
# Modification Reason:
# - Add production systemd unit synchronization, rollback, no-restart, and
#   health polling controls while preserving active-session protection.
#
# Main Functionality:
# - Pulls the configured branch.
# - Prevents concurrent upgrade runs on the same node.
# - Builds aeronyx-server release binary.
# - Validates /etc/aeronyx/server.toml.
# - Syncs the repository systemd unit template before restart.
# - Checks active VPN sessions before restart.
# - Restarts systemd service and verifies post-upgrade health.
# - Restores the previous systemd unit and binary if restart or health
#   verification fails.
#
# Dependencies:
# - deploy/node/healthcheck.sh
# - crates/aeronyx-server/src/main.rs validate/start commands
# - systemd unit installed by deploy/node/install.sh
#
# Main Logical Flow:
# 1. Acquire the node-local upgrade lock.
# 2. Update repo from Git.
# 3. Build and validate the release binary.
# 4. Restart only when no active sessions are present, unless --force is used.
# 5. Sync and verify the systemd unit template.
# 6. Verify local health and roll back the unit/binary if restart health fails.
#
# Important Note for Next Developer:
# - Do not remove active-session protection. Commercial VPN users should not be
#   disconnected by routine upgrades unless the operator explicitly forces it.
# - Do not overwrite /etc/aeronyx/server.toml during upgrades.
# - Keep this script compatible with current and older installed service units.
#
# Last Modified:
# v1.4.0-node-deploy - Uses the shared node deployment lock across install and
#                      upgrade workflows.
# v1.3.0-node-deploy - Added node-local upgrade locking to prevent concurrent
#                      binary/systemd unit replacement.
# v1.2.0-node-deploy - Syncs systemd unit template during restart upgrades and
#                      rolls back the unit together with the binary.
# v1.1.0-node-deploy - Added --no-restart, health polling, and binary rollback
#                      on restart/health failure.
# v1.0.0-node-deploy - Added production node upgrade script.
# ============================================

set -euo pipefail

REPO_DIR="/opt/aeronyx/AeroNyx"
BRANCH="main"
CONFIG_FILE="/etc/aeronyx/server.toml"
SERVICE_NAME="aeronyx-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOCK_FILE="/run/lock/${SERVICE_NAME}.deploy.lock"
LOCK_DIR=""
FORCE=0
DRY_RUN=0
SKIP_PULL=0
SKIP_UNIT_UPDATE=0
NO_RESTART=0
HEALTH_RETRIES=10
HEALTH_DELAY=2
BACKUP_BINARY=""
BACKUP_SERVICE_FILE=""

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
  --no-restart        Build and validate only; do not restart the service.
  --skip-pull         Build the currently checked out source.
  --skip-unit-update  Do not render/install deploy/node/aeronyx-server.service.
  --health-retries N  Health polling attempts after restart. Default: 10
  --health-delay N    Seconds between health polling attempts. Default: 2
  --dry-run           Print actions without changing the host.
  -h, --help          Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --branch) BRANCH="${2:?missing value}"; shift 2 ;;
        --config) CONFIG_FILE="${2:?missing value}"; shift 2 ;;
        --service) SERVICE_NAME="${2:?missing value}"; SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"; LOCK_FILE="/run/lock/${SERVICE_NAME}.deploy.lock"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --no-restart) NO_RESTART=1; shift ;;
        --skip-pull) SKIP_PULL=1; shift ;;
        --skip-unit-update) SKIP_UNIT_UPDATE=1; shift ;;
        --health-retries) HEALTH_RETRIES="${2:?missing value}"; shift 2 ;;
        --health-delay) HEALTH_DELAY="${2:?missing value}"; shift 2 ;;
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

release_lock() {
    if [ -n "${LOCK_DIR}" ] && [ -d "${LOCK_DIR}" ]; then
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}

acquire_lock() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] acquire deployment lock %s\n' "${LOCK_FILE}"
        return
    fi

    if command -v flock >/dev/null 2>&1; then
        mkdir -p "$(dirname "${LOCK_FILE}")"
        exec 9>"${LOCK_FILE}"
        flock -n 9 || die "Another ${SERVICE_NAME} install or upgrade is already running."
        ok "Deployment lock acquired: ${LOCK_FILE}"
        return
    fi

    LOCK_DIR="/tmp/${SERVICE_NAME}.deploy.lock"
    mkdir "${LOCK_DIR}" 2>/dev/null || die "Another ${SERVICE_NAME} install or upgrade appears to be running: ${LOCK_DIR}"
    trap release_lock EXIT
    ok "Deployment lock acquired: ${LOCK_DIR}"
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
    BACKUP_BINARY="${backup_dir}/aeronyx-server.${stamp}"
    run mkdir -p "${backup_dir}"
    run cp "${binary}" "${BACKUP_BINARY}"
    ok "Current binary backed up to ${BACKUP_BINARY}"
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

backup_current_service_unit() {
    local backup_dir stamp
    [ -f "${SERVICE_FILE}" ] || return

    backup_dir="/var/lib/aeronyx/releases"
    stamp="$(date -u +%Y%m%d_%H%M%S)"
    BACKUP_SERVICE_FILE="${backup_dir}/${SERVICE_NAME}.service.${stamp}"
    run mkdir -p "${backup_dir}"
    run cp "${SERVICE_FILE}" "${BACKUP_SERVICE_FILE}"
    ok "Current systemd unit backed up to ${BACKUP_SERVICE_FILE}"
}

render_service_unit() {
    local template rendered
    [ "${SKIP_UNIT_UPDATE}" -eq 0 ] || { ok "Systemd unit update skipped"; return; }
    [ "${NO_RESTART}" -eq 0 ] || { ok "Systemd unit update skipped by --no-restart"; return; }

    template="${REPO_DIR}/deploy/node/aeronyx-server.service"
    rendered="/tmp/${SERVICE_NAME}.upgrade.service"

    if [ ! -f "${template}" ]; then
        warn "Systemd unit template missing; leaving installed unit unchanged: ${template}"
        return
    fi

    log "Rendering systemd unit template to ${SERVICE_FILE}"
    backup_current_service_unit
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] render %s to %s\n' "${template}" "${SERVICE_FILE}"
        printf '[DRY-RUN] systemd-analyze verify %s\n' "${SERVICE_FILE}"
        return
    fi

    sed \
        -e "s|@REPO_DIR@|${REPO_DIR}|g" \
        -e "s|@CONFIG_FILE@|${CONFIG_FILE}|g" \
        "${template}" > "${rendered}"
    systemd-analyze verify "${rendered}"
    cp "${rendered}" "${SERVICE_FILE}"
    chmod 644 "${SERVICE_FILE}"
    systemctl daemon-reload
}

rollback_service_unit() {
    if [ -z "${BACKUP_SERVICE_FILE}" ]; then
        warn "Systemd unit rollback skipped; no backup unit path recorded."
        return 0
    fi
    if [ "${DRY_RUN}" -eq 0 ] && [ ! -f "${BACKUP_SERVICE_FILE}" ]; then
        warn "Systemd unit rollback skipped; backup missing: ${BACKUP_SERVICE_FILE}"
        return 1
    fi

    warn "Rolling back systemd unit: ${BACKUP_SERVICE_FILE}"
    run cp "${BACKUP_SERVICE_FILE}" "${SERVICE_FILE}"
    run systemctl daemon-reload
}

health_endpoint_ok() {
    command -v curl >/dev/null 2>&1 || return 1
    curl -fsS --max-time 5 http://127.0.0.1:8421/api/vpn/health 2>/dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("status") == "ok" else 1)' 2>/dev/null
}

wait_for_health() {
    local attempt

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] poll http://127.0.0.1:8421/api/vpn/health up to %s times\n' "${HEALTH_RETRIES}"
        return 0
    fi

    attempt=1
    while [ "${attempt}" -le "${HEALTH_RETRIES}" ]; do
        if health_endpoint_ok; then
            ok "Post-restart VPN health endpoint is ok"
            return 0
        fi
        warn "Post-restart health not ready (${attempt}/${HEALTH_RETRIES})"
        sleep "${HEALTH_DELAY}"
        attempt=$((attempt + 1))
    done

    return 1
}

rollback_binary() {
    local binary
    binary="${REPO_DIR}/target/release/aeronyx-server"

    if [ -z "${BACKUP_BINARY}" ]; then
        warn "Rollback skipped; no backup binary path recorded."
        return 1
    fi
    if [ "${DRY_RUN}" -eq 0 ] && [ ! -f "${BACKUP_BINARY}" ]; then
        warn "Rollback skipped; backup binary missing: ${BACKUP_BINARY}"
        return 1
    fi

    warn "Rolling back to previous binary: ${BACKUP_BINARY}"
    rollback_service_unit
    run cp "${BACKUP_BINARY}" "${binary}"
    run systemctl daemon-reload
    run systemctl restart "${SERVICE_NAME}"
    run systemctl is-active "${SERVICE_NAME}"
}

restart_service() {
    local sessions
    [ "${NO_RESTART}" -eq 0 ] || { ok "Restart skipped by --no-restart"; return; }

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
    if ! run systemctl restart "${SERVICE_NAME}"; then
        warn "Restart failed; attempting rollback."
        rollback_binary
        die "Upgrade failed during restart and rollback was attempted."
    fi
    if ! run systemctl is-active "${SERVICE_NAME}"; then
        warn "Service is not active after restart; attempting rollback."
        rollback_binary
        die "Upgrade failed because service did not become active."
    fi
    if ! wait_for_health; then
        warn "Health endpoint failed after restart; attempting rollback."
        rollback_binary
        die "Upgrade failed because post-restart health check did not pass."
    fi
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
    acquire_lock
    ensure_cargo_path
    [ -d "${REPO_DIR}/.git" ] || die "Repository not found: ${REPO_DIR}"
    backup_current_binary
    update_source
    build_release
    validate_config
    render_service_unit
    restart_service
    if [ "${NO_RESTART}" -eq 0 ]; then
        run_healthcheck
    else
        ok "Build and validation complete. Service was not restarted."
    fi
}

main "$@"
