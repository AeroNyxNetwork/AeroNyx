#!/usr/bin/env bash
# ============================================
# File: deploy/node/healthcheck.sh
# ============================================
# Creation Reason:
# - Provide a single operator-facing diagnostic command for AeroNyx Rust privacy
#   nodes after install, upgrade, or incident response.
#
# Modification Reason:
# - Add service-name validation while preserving the production healthcheck for
#   node deployment workflows.
# - Add release-backup diagnostics for node upgrade retention observability.
# - Validate generated network restore command paths for reboot reliability.
# - Expose network restore command paths in JSON for nodeboard automation.
# - Include network restore unit backups in retention diagnostics.
#
# Main Functionality:
# - Checks repository, binary, config, registration state, systemd status,
#   host capacity, IP forwarding, NAT runtime/persistence, local VPN health
#   endpoint, release-backup retention, and capacity telemetry.
# - Checks that generated network restore ExecStart command paths exist.
# - Emits machine-readable JSON for nodeboard or support automation.
# - Warns when installed systemd hardening is weaker than the production
#   template.
#
# Dependencies:
# - /etc/aeronyx/server.toml
# - /etc/aeronyx/node_info.json
# - aeronyx-server validate/status
# - http://127.0.0.1:8421/api/vpn/health
#
# Main Logical Flow:
# 1. Perform static checks on files and systemd.
# 2. Validate config with aeronyx-server when binary exists.
# 3. Check release-backup retention and query local health endpoint.
#
# Important Note for Next Developer:
# - Do not print private keys, registration secrets, API secrets, wallet-level
#   traffic, DNS contents, destinations, payloads, or client public IPs.
# - Keep output stable; nodeboard or support tooling may parse these lines later.
# - This script should never modify host state.
# - Reject service names that look like paths or command-line options.
#
# Last Modified:
# v1.10.0-node-deploy - Added network restore unit backup counts to release
#                       retention diagnostics.
# v1.9.0-node-deploy - Added structured network restore command diagnostics to
#                      JSON output.
# v1.8.0-node-deploy - Added network restore ExecStart command path checks.
# v1.7.0-node-deploy - Added release-backup count diagnostics for upgrade
#                      retention observability.
# v1.6.0-node-deploy - Validates --service names before systemd and journal
#                      diagnostics.
# v1.5.0-node-deploy - Added runtime metadata, tracked worktree, and current
#                      service-start journal diagnostics.
# v1.4.0-node-deploy - Added sysctl/iptables persistence diagnostics.
# v1.3.1-node-deploy - Checks systemd-managed AeroNyx directories.
# v1.3.0-node-deploy - Added systemd hardening diagnostics.
# v1.2.0-node-deploy - Added full checks[] JSON output and --json-only mode.
# v1.1.0-node-deploy - Added host capacity, TUN, route, disk, and port checks.
# v1.0.0-node-deploy - Added production node healthcheck.
# ============================================

set -euo pipefail

REPO_DIR="/opt/aeronyx/AeroNyx"
CONFIG_FILE="/etc/aeronyx/server.toml"
SERVICE_NAME="aeronyx-server"
JSON=0
JSON_ONLY=0
CHECK_LOG="$(mktemp)"
SYSCTL_FILE="/etc/sysctl.d/99-aeronyx.conf"
IPTABLES_RULES_FILE="/etc/iptables/rules.v4"
NETWORK_RESTORE_SERVICE="aeronyx-network-restore"
RELEASE_DIR="/var/lib/aeronyx/releases"
RELEASE_BACKUP_KEEP_TARGET=10
trap 'rm -f "${CHECK_LOG}"' EXIT

usage() {
    cat <<'USAGE'
Usage:
  ./deploy/node/healthcheck.sh [OPTIONS]

Options:
  --repo-dir PATH   Repository path. Default: /opt/aeronyx/AeroNyx
  --config PATH     Config path. Default: /etc/aeronyx/server.toml
  --service NAME    systemd service name. Default: aeronyx-server
  --json            Emit JSON summary as the final output line.
  --json-only       Emit only JSON, suitable for nodeboard automation.
  -h, --help        Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --config) CONFIG_FILE="${2:?missing value}"; shift 2 ;;
        --service) SERVICE_NAME="${2:?missing value}"; shift 2 ;;
        --json) JSON=1; shift ;;
        --json-only) JSON=1; JSON_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[ERROR] Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

PASS=0
WARN=0
FAIL=0

validate_service_name() {
    case "${SERVICE_NAME}" in
        ""|-*|*/*)
            printf '[ERROR] Invalid service name: %s\n' "${SERVICE_NAME}" >&2
            exit 1
            ;;
    esac

    printf '%s' "${SERVICE_NAME}" | grep -Eq '^[A-Za-z0-9_.@-]+$' || {
        printf '[ERROR] Invalid service name: %s\n' "${SERVICE_NAME}" >&2
        exit 1
    }
}

record_check() {
    local status="$1"
    shift
    printf '%s\t%s\n' "${status}" "$*" >> "${CHECK_LOG}"
}

pass() {
    PASS=$((PASS + 1))
    record_check "pass" "$*"
    [ "${JSON_ONLY}" -eq 1 ] || printf '[PASS] %s\n' "$*"
}

warn() {
    WARN=$((WARN + 1))
    record_check "warn" "$*"
    [ "${JSON_ONLY}" -eq 1 ] || printf '[WARN] %s\n' "$*" >&2
}

fail() {
    FAIL=$((FAIL + 1))
    record_check "fail" "$*"
    [ "${JSON_ONLY}" -eq 1 ] || printf '[FAIL] %s\n' "$*" >&2
}

check_file() {
    local path="$1" label="$2"
    if [ -e "${path}" ]; then
        pass "${label}: ${path}"
    else
        fail "${label} missing: ${path}"
    fi
}

check_command() {
    local cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        pass "command found: ${cmd}"
    else
        warn "command missing: ${cmd}"
    fi
}

existing_path_for_df() {
    local path="$1"
    while [ ! -e "${path}" ] && [ "${path}" != "/" ]; do
        path="$(dirname "${path}")"
    done
    printf '%s\n' "${path}"
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -lntu 2>/dev/null | awk '{print $5}' | grep -Eq "[:.]${port}$"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -lntu 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
    else
        return 2
    fi
}

check_system() {
    if [ "$(uname -s)" = "Linux" ]; then
        pass "host OS: Linux"
    else
        fail "host OS is not Linux; production node service is Linux/systemd only"
    fi

    check_command systemctl
    check_command curl
    check_command ip
    check_command iptables
    check_command python3
}

check_host_capacity() {
    local default_iface disk_path disk_free_mb mem_mb

    if [ -e /dev/net/tun ]; then
        pass "TUN device available"
    else
        fail "TUN device missing: /dev/net/tun"
    fi

    default_iface="$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')"
    if [ -n "${default_iface}" ]; then
        pass "default route interface: ${default_iface}"
    else
        warn "default route not detected"
    fi

    mem_mb="$(awk '/MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || printf '0')"
    if [ "${mem_mb}" -ge 2048 ]; then
        pass "memory available: ${mem_mb} MB"
    else
        warn "memory low: ${mem_mb} MB; 2GB+ recommended for production VPN nodes"
    fi

    disk_path="$(existing_path_for_df "${REPO_DIR}")"
    disk_free_mb="$(df -Pm "${disk_path}" 2>/dev/null | awk 'NR==2 {print $4}' || printf '0')"
    if [ "${disk_free_mb:-0}" -ge 4096 ]; then
        pass "disk free near ${disk_path}: ${disk_free_mb} MB"
    else
        warn "disk free near ${disk_path}: ${disk_free_mb:-0} MB; 4GB+ recommended"
    fi

    if port_in_use 51820; then
        pass "port 51820 is listening"
    else
        warn "port 51820 is not listening"
    fi

    if port_in_use 8421; then
        pass "port 8421 is listening"
    else
        warn "port 8421 is not listening"
    fi
}

check_repo_and_binary() {
    check_file "${REPO_DIR}/Cargo.toml" "repository Cargo.toml"
    check_file "${REPO_DIR}/target/release/aeronyx-server" "release binary"
    check_file "${CONFIG_FILE}" "config"
    check_file "/etc/aeronyx/node_info.json" "node registration"
    check_file "/etc/aeronyx/server_key.json" "node server key"
    check_file "/var/lib/aeronyx" "state directory"
    check_file "/var/log/aeronyx" "log directory"
}

check_runtime_metadata() {
    local active_since dirty_lines journal_output journal_warnings

    if command -v git >/dev/null 2>&1 && [ -d "${REPO_DIR}/.git" ]; then
        dirty_lines="$(git -C "${REPO_DIR}" status --short --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${dirty_lines:-0}" = "0" ]; then
            pass "git tracked worktree clean"
        else
            warn "git tracked worktree has ${dirty_lines} modified file(s)"
        fi
    else
        warn "git tracked worktree check skipped"
    fi

    if command -v journalctl >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1; then
        active_since="$(systemctl show "${SERVICE_NAME}" -p ActiveEnterTimestamp --value 2>/dev/null || true)"
        if [ -n "${active_since}" ]; then
            journal_output="$(journalctl -u "${SERVICE_NAME}" --since "${active_since}" -p warning --no-pager 2>/dev/null || true)"
            journal_output="$(printf '%s\n' "${journal_output}" | grep -Ev '^-- No entries --$' || true)"
            journal_warnings="$(printf '%s\n' "${journal_output}" | sed '/^$/d' | wc -l | tr -d ' ')"
            if [ "${journal_warnings:-0}" = "0" ]; then
                pass "journal warnings since current service start: 0"
            else
                warn "journal warnings since current service start: ${journal_warnings}"
            fi
        else
            warn "journal warning check skipped; service start timestamp unavailable"
        fi
    else
        warn "journal warning check skipped"
    fi
}

count_release_backups() {
    local pattern="$1"
    find "${RELEASE_DIR}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
}

check_release_backups() {
    local binary_count unit_count network_restore_unit_count

    if [ ! -d "${RELEASE_DIR}" ]; then
        pass "release backup directory absent; no retained upgrade backups"
        return
    fi

    binary_count="$(count_release_backups "aeronyx-server.*")"
    unit_count="$(count_release_backups "${SERVICE_NAME}.service.*")"
    network_restore_unit_count="$(count_release_backups "${NETWORK_RESTORE_SERVICE}.service.*")"

    if [ "${binary_count:-0}" -le "${RELEASE_BACKUP_KEEP_TARGET}" ] \
        && [ "${unit_count:-0}" -le "${RELEASE_BACKUP_KEEP_TARGET}" ] \
        && [ "${network_restore_unit_count:-0}" -le "${RELEASE_BACKUP_KEEP_TARGET}" ]; then
        pass "release backups within retention target: binary=${binary_count:-0} systemd_unit=${unit_count:-0} network_restore_unit=${network_restore_unit_count:-0} keep_target=${RELEASE_BACKUP_KEEP_TARGET}"
    else
        warn "release backups exceed retention target: binary=${binary_count:-0} systemd_unit=${unit_count:-0} network_restore_unit=${network_restore_unit_count:-0} keep_target=${RELEASE_BACKUP_KEEP_TARGET}; run a successful upgrade or prune manually"
    fi
}

check_config_validation() {
    local binary="${REPO_DIR}/target/release/aeronyx-server"
    if [ -x "${binary}" ] && [ -f "${CONFIG_FILE}" ]; then
        if "${binary}" validate -c "${CONFIG_FILE}" >/tmp/aeronyx-validate.out 2>&1; then
            pass "config validation"
        else
            fail "config validation failed"
            sed -n '1,80p' /tmp/aeronyx-validate.out >&2 || true
        fi
    else
        warn "config validation skipped; binary or config missing"
    fi
}

check_systemd() {
    if ! command -v systemctl >/dev/null 2>&1; then
        warn "systemd checks skipped"
        return
    fi

    if systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend 2>/dev/null | grep -q "${SERVICE_NAME}.service"; then
        pass "systemd unit installed: ${SERVICE_NAME}"
    else
        fail "systemd unit not installed: ${SERVICE_NAME}"
    fi

    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        pass "systemd service active: ${SERVICE_NAME}"
    else
        warn "systemd service not active: ${SERVICE_NAME}"
    fi
}

systemd_property() {
    local property="$1"
    systemctl show "${SERVICE_NAME}" -p "${property}" --value 2>/dev/null || true
}

check_systemd_hardening() {
    local capability_bounding ambient_capabilities protect_system read_write_paths

    if ! command -v systemctl >/dev/null 2>&1; then
        warn "systemd hardening checks skipped"
        return
    fi
    if ! systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend 2>/dev/null | grep -q "${SERVICE_NAME}.service"; then
        warn "systemd hardening checks skipped; service unit missing"
        return
    fi

    [ "$(systemd_property NoNewPrivileges)" = "yes" ] \
        && pass "systemd hardening NoNewPrivileges=yes" \
        || warn "systemd hardening NoNewPrivileges is not enabled"

    [ "$(systemd_property PrivateTmp)" = "yes" ] \
        && pass "systemd hardening PrivateTmp=yes" \
        || warn "systemd hardening PrivateTmp is not enabled"

    protect_system="$(systemd_property ProtectSystem)"
    [ "${protect_system}" = "full" ] || [ "${protect_system}" = "strict" ] \
        && pass "systemd hardening ProtectSystem=${protect_system}" \
        || warn "systemd hardening ProtectSystem is not full/strict"

    [ "$(systemd_property ProtectControlGroups)" = "yes" ] \
        && pass "systemd hardening ProtectControlGroups=yes" \
        || warn "systemd hardening ProtectControlGroups is not enabled"

    [ "$(systemd_property ProtectKernelTunables)" = "yes" ] \
        && pass "systemd hardening ProtectKernelTunables=yes" \
        || warn "systemd hardening ProtectKernelTunables is not enabled"

    [ "$(systemd_property ProtectKernelModules)" = "yes" ] \
        && pass "systemd hardening ProtectKernelModules=yes" \
        || warn "systemd hardening ProtectKernelModules is not enabled"

    [ "$(systemd_property RestrictSUIDSGID)" = "yes" ] \
        && pass "systemd hardening RestrictSUIDSGID=yes" \
        || warn "systemd hardening RestrictSUIDSGID is not enabled"

    [ "$(systemd_property LockPersonality)" = "yes" ] \
        && pass "systemd hardening LockPersonality=yes" \
        || warn "systemd hardening LockPersonality is not enabled"

    capability_bounding="$(systemd_property CapabilityBoundingSet)"
    if printf '%s\n' "${capability_bounding}" | grep -q 'cap_sys_admin'; then
        warn "systemd hardening CapabilityBoundingSet still includes cap_sys_admin"
    elif printf '%s\n' "${capability_bounding}" | grep -q 'cap_net_admin' \
        && printf '%s\n' "${capability_bounding}" | grep -q 'cap_net_raw'; then
        pass "systemd hardening CapabilityBoundingSet is restricted for VPN networking"
    else
        warn "systemd hardening CapabilityBoundingSet may not include required VPN capabilities"
    fi

    ambient_capabilities="$(systemd_property AmbientCapabilities)"
    if printf '%s\n' "${ambient_capabilities}" | grep -q 'cap_net_admin' \
        && printf '%s\n' "${ambient_capabilities}" | grep -q 'cap_net_raw'; then
        pass "systemd hardening AmbientCapabilities includes VPN capabilities"
    else
        warn "systemd hardening AmbientCapabilities missing expected VPN capabilities"
    fi

    read_write_paths="$(systemd_property ReadWritePaths)"
    if printf '%s\n' "${read_write_paths}" | grep -q '/etc/aeronyx' \
        && printf '%s\n' "${read_write_paths}" | grep -q '/var/lib/aeronyx'; then
        pass "systemd hardening ReadWritePaths include AeroNyx state directories"
    else
        warn "systemd hardening ReadWritePaths missing AeroNyx state directories"
    fi
}

check_network() {
    local forwarding restore_cmd restore_paths
    forwarding="$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null || printf 'unknown')"
    if [ "${forwarding}" = "1" ]; then
        pass "IPv4 forwarding enabled"
    else
        warn "IPv4 forwarding is not enabled"
    fi

    if command -v iptables >/dev/null 2>&1; then
        if iptables -t nat -S POSTROUTING 2>/dev/null | grep -q '100\.64\.0\.0/24.*MASQUERADE'; then
            pass "iptables NAT rule present for 100.64.0.0/24"
        else
            warn "iptables NAT rule for 100.64.0.0/24 not found"
        fi
    fi

    if [ -f "${SYSCTL_FILE}" ] && grep -q '^net.ipv4.ip_forward=1' "${SYSCTL_FILE}" 2>/dev/null; then
        pass "IPv4 forwarding persisted in ${SYSCTL_FILE}"
    else
        warn "IPv4 forwarding persistence file missing or incomplete: ${SYSCTL_FILE}"
    fi

    if [ -f "${IPTABLES_RULES_FILE}" ] && grep -q '100\.64\.0\.0/24.*MASQUERADE' "${IPTABLES_RULES_FILE}" 2>/dev/null; then
        pass "iptables NAT rules persisted in ${IPTABLES_RULES_FILE}"
    else
        warn "iptables NAT persistence file missing or incomplete: ${IPTABLES_RULES_FILE}"
    fi

    if command -v systemctl >/dev/null 2>&1; then
        if systemctl list-unit-files "${NETWORK_RESTORE_SERVICE}.service" --no-legend 2>/dev/null | grep -q "${NETWORK_RESTORE_SERVICE}.service"; then
            pass "network restore service installed: ${NETWORK_RESTORE_SERVICE}"
        else
            warn "network restore service not installed: ${NETWORK_RESTORE_SERVICE}"
        fi
        if systemctl is-enabled --quiet "${NETWORK_RESTORE_SERVICE}.service" 2>/dev/null; then
            pass "network restore service enabled: ${NETWORK_RESTORE_SERVICE}"
        else
            warn "network restore service not enabled: ${NETWORK_RESTORE_SERVICE}"
        fi
        restore_paths="$(systemctl cat "${NETWORK_RESTORE_SERVICE}.service" 2>/dev/null \
            | awk -F= '/^ExecStart=/ {print $2}' \
            | awk '{print $1}')"
        if [ -n "${restore_paths}" ]; then
            while IFS= read -r restore_cmd; do
                [ -n "${restore_cmd}" ] || continue
                if [ -x "${restore_cmd}" ]; then
                    pass "network restore command path executable: ${restore_cmd}"
                else
                    warn "network restore command path missing or not executable: ${restore_cmd}"
                fi
            done <<EOF
${restore_paths}
EOF
        else
            warn "network restore ExecStart commands not readable"
        fi
    fi
}

check_health_endpoint() {
    if ! command -v curl >/dev/null 2>&1; then
        warn "health endpoint skipped; curl missing"
        return
    fi

    local tmp
    tmp="$(mktemp)"
    if curl -fsS --max-time 5 http://127.0.0.1:8421/api/vpn/health >"${tmp}" 2>/dev/null; then
        pass "local VPN health endpoint"
        if [ "${JSON_ONLY}" -eq 0 ] && command -v python3 >/dev/null 2>&1; then
            python3 - "${tmp}" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
capacity = data.get("capacity") or {}
interface = capacity.get("interface") or {}
print("[INFO] status=%s active_sessions=%s active_wallet_devices=%s" % (
    data.get("status"),
    data.get("active_sessions"),
    data.get("active_wallet_devices"),
))
if capacity:
    print("[INFO] capacity ip_pool=%s used=%s free=%s max_connections=%s policy_max_sessions=%s" % (
        capacity.get("ip_pool_capacity"),
        capacity.get("ip_pool_used"),
        capacity.get("ip_pool_free"),
        capacity.get("max_connections"),
        capacity.get("policy_max_sessions"),
    ))
    print("[INFO] capacity conntrack=%s fd=%s drops=%s pps=%s bps=%s interface=%s" % (
        (capacity.get("conntrack") or {}).get("used"),
        (capacity.get("file_descriptors") or {}).get("used"),
        capacity.get("packet_drops_total"),
        interface.get("total_pps"),
        interface.get("total_bps"),
        interface.get("interface"),
    ))
else:
    print("[WARN] capacity telemetry missing; upgrade Rust node binary if this is unexpected")
PY
        fi
    else
        warn "local VPN health endpoint unavailable"
    fi
    rm -f "${tmp}"
}

emit_json_summary() {
    if [ "${JSON}" -ne 1 ]; then
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 - "${CHECK_LOG}" "${REPO_DIR}" "${CONFIG_FILE}" "${SERVICE_NAME}" "${NETWORK_RESTORE_SERVICE}" "${PASS}" "${WARN}" "${FAIL}" <<'PY'
import json
import glob
import os
import subprocess
import sys
from datetime import datetime, timezone

checks_path, repo_dir, config_path, service_name, network_restore_service = sys.argv[1:6]
pass_count, warn_count, fail_count = [int(value) for value in sys.argv[6:9]]

def run(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

def file_mtime(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat()
    except Exception:
        return None

def network_restore_commands(unit_name):
    try:
        unit_text = subprocess.check_output(["systemctl", "cat", f"{unit_name}.service"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []

    commands = []
    for line in unit_text.splitlines():
        if not line.startswith("ExecStart="):
            continue
        command = line.split("=", 1)[1].strip().split(" ", 1)[0]
        if not command:
            continue
        commands.append({"path": command, "executable": os.access(command, os.X_OK)})
    return commands

checks = []
with open(checks_path, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        status, _, message = line.partition("\t")
        checks.append({"status": status, "message": message})

binary_path = os.path.join(repo_dir, "target/release/aeronyx-server")
release_dir = "/var/lib/aeronyx/releases"
release_backups = {
    "dir": release_dir,
    "binary_count": len(glob.glob(os.path.join(release_dir, "aeronyx-server.*"))),
    "systemd_unit_count": len(glob.glob(os.path.join(release_dir, f"{service_name}.service.*"))),
    "network_restore_unit_count": len(glob.glob(os.path.join(release_dir, f"{network_restore_service}.service.*"))),
    "keep_target": 10,
}
runtime = {
    "git_commit": run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"]),
    "git_branch": run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"]),
    "git_tracked_dirty": bool(run(["git", "-C", repo_dir, "status", "--short", "--untracked-files=no"])),
    "binary_path": binary_path,
    "binary_mtime": file_mtime(binary_path),
    "config_mtime": file_mtime(config_path),
    "service_active": run(["systemctl", "is-active", service_name]),
    "service_enabled": run(["systemctl", "is-enabled", service_name]),
    "network_restore_active": run(["systemctl", "is-active", network_restore_service]),
    "network_restore_enabled": run(["systemctl", "is-enabled", network_restore_service]),
    "network_restore_commands": network_restore_commands(network_restore_service),
    "release_backups": release_backups,
}

print(json.dumps({
    "success": fail_count == 0,
    "summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
    "repo_dir": repo_dir,
    "config": config_path,
    "service": service_name,
    "runtime": runtime,
    "checks": checks,
    "privacy_boundary": "diagnostics only; no private keys, registration secrets, client public IPs, destinations, DNS contents, packet payloads, or wallet-level traffic",
    "generated_at": datetime.now(timezone.utc).isoformat(),
}, separators=(",", ":")))
PY
    fi
}

main() {
    validate_service_name
    check_system
    check_host_capacity
    check_repo_and_binary
    check_runtime_metadata
    check_release_backups
    check_config_validation
    check_systemd
    check_systemd_hardening
    check_network
    check_health_endpoint

    [ "${JSON_ONLY}" -eq 1 ] || printf '[SUMMARY] pass=%s warn=%s fail=%s\n' "${PASS}" "${WARN}" "${FAIL}"
    emit_json_summary
    [ "${FAIL}" -eq 0 ]
}

main "$@"
