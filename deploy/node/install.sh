#!/usr/bin/env bash
# ============================================
# File: deploy/node/install.sh
# ============================================
# Creation Reason:
# - Provide a one-command, production-grade installer for AeroNyx Rust privacy
#   nodes without requiring operators to manually copy config, systemd units,
#   sysctl, iptables, and build commands.
#
# Modification Reason:
# - Add structured, privacy-safe install context to nodeboard progress reports
#   so operators can see command mode, service/config path, host OS/arch, failed
#   phase, and exit code without scraping SSH logs.
# - Preserve the current install phase for ERR traps so nodeboard can show
#   whether a failed install stopped during preflight, dependencies, repository
#   setup, network, build, systemd, registration, or service start.
# - Add --set-vpn-cidr for network-only VPN pool maintenance so operators can
#   safely update /etc/aeronyx/server.toml and refresh NAT/restore rules
#   without rebuilding, registering, or restarting the Rust service.
# - Report code-scoped install progress to nodeboard so operators and AI
#   assistants can see plan/build/register/start failures without SSH log
#   archaeology.
# - Inject the current Git commit into release builds so nodeboard can display
#   exact Rust runtime provenance after install or upgrade.
# - Remove stale AeroNyx 100.64.0.0/* NAT rules during network refresh so
#   commercial pool migrations, such as /24 to /22, do not leave overlapping
#   MASQUERADE rules in the persisted iptables set.
# - Add --print-plan so nodeboard, operators, and support automation can verify
#   environment-variable parsing and one-command install intent without touching
#   host state or printing registration secrets.
# - Add environment-variable defaults and --quick first-install mode so
#   operators can launch a commercial node with one copy-paste command while
#   still using the existing preflight, lock, build, network, systemd, and
#   registration safeguards.
# - Add install-time systemd unit verification while preserving the production
#   deployment entrypoint for commercial VPN node operators.
# - Render the generated network restore unit with detected system command
#   paths instead of assuming /sbin.
# - Verify the generated network restore unit before replacing the installed
#   reboot recovery service.
# - Refuse to pull an existing tracked-dirty repository unless the operator
#   explicitly opts in.
# - Make --no-enable complete cleanly after service rendering instead of
#   inheriting a false shell test status.
# - Make installs without a registration code complete cleanly instead of
#   inheriting a false shell test status.
# - Derive VPN NAT subnet and TUN interface from server.toml instead of using
#   hard-coded deployment defaults.
# - Add a network-only maintenance mode for refreshing forwarding/NAT after
#   VPN subnet or TUN-device changes.
# - Add an install-time commercial capacity plan summary so operators can see
#   IP-pool, max_connections, file-descriptor, and conntrack risk before the
#   first build or service restart.
# - Align default/fallback VPN pool with the commercial 1000-session profile.
#
# Main Functionality:
# - Detects Linux/systemd environment.
# - Runs production preflight checks for TUN, default route, memory, disk, and
#   common AeroNyx ports.
# - Prints a read-only commercial capacity plan from server.toml/defaults and
#   host kernel limits.
# - Prevents concurrent install/upgrade runs on the same node.
# - Installs host dependencies on supported Linux distributions.
# - Clones or uses the AeroNyx repository.
# - Creates /etc/aeronyx and /var/lib/aeronyx state directories.
# - Installs a safe default server.toml without overwriting existing config.
# - Builds aeronyx-server release binary.
# - Verifies, installs, and enables the systemd service.
# - Optionally configures IP forwarding/NAT and registers/starts the node.
# - Persists VPN forwarding/NAT across host reboots.
# - Renders and verifies reboot network restore with distro-specific
#   sysctl/iptables paths.
#
# Dependencies:
# - deploy/node/server.example.toml
# - deploy/node/aeronyx-server.service
# - crates/aeronyx-server/src/main.rs CLI commands:
#   register, start, validate, status
#
# Main Logical Flow:
# 1. Parse flags and run production preflight checks.
# 2. Exit early for --preflight-only without taking the deployment lock.
# 3. Acquire the shared node deployment lock before host writes.
# 4. Prepare repository, directories, config, and network forwarding.
# 5. Build release binary, install systemd unit, optionally register/start.
#
# Important Note for Next Developer:
# - Never overwrite /etc/aeronyx/server.toml, server_key.json, or node_info.json
#   without an explicit future migration flow.
# - Keep all operations idempotent so operators can safely rerun the installer.
# - This script is Linux/systemd only; macOS, iOS, Android, and Windows are
#   development/client platforms, not production node hosts for this script.
# - Keep the dirty-worktree check limited to tracked files so untracked runtime
#   data and build artifacts do not block reinstall flows.
# - Keep network setup aligned with /etc/aeronyx/server.toml when operators
#   expand the VPN IP pool or customize the TUN device.
# - Keep --network-only free of binary build, registration, and service restart
#   side effects.
# - Keep --quick as a thin first-install wrapper; do not bypass preflight,
#   dirty-worktree protection, systemd verification, or registration failure.
# - Keep --print-plan read-only and secret-safe; it must never print the
#   registration code, private keys, API secrets, or wallet-level data.
# - Keep stale NAT cleanup scoped to AeroNyx CGNAT pool rules on the detected
#   egress interface; do not delete unrelated MASQUERADE rules.
# - Keep --set-vpn-cidr restricted to --network-only. Changing the persisted
#   pool is safe without restart, but Rust/TUN capacity only changes after a
#   separate maintenance-window service restart.
#
# Last Modified:
# v1.23.0-node-deploy - Adds structured install report context for nodeboard.
# v1.22.0-node-deploy - Reports exact failed install phase to nodeboard.
# v1.21.0-node-deploy - Reports privacy-safe install progress to nodeboard.
# v1.20.0-node-deploy - Injects AERONYX_GIT_COMMIT into release builds for
#                       nodeboard runtime provenance.
# v1.19.0-node-deploy - Added --set-vpn-cidr for safe network-only VPN pool
#                       config updates.
# v1.18.0-node-deploy - Cleans stale AeroNyx NAT rules during VPN pool
#                       migrations before persisting iptables.
# v1.17.0-node-deploy - Added read-only --print-plan for one-command install
#                       verification and nodeboard automation.
# v1.16.0-node-deploy - Added environment defaults and --quick first-install
#                       mode for one-command commercial node setup.
# v1.15.0-node-deploy - Expanded default VPN pool fallback to /22 for
#                       1000-session commercial capacity.
# v1.14.0-node-deploy - Added capacity plan preflight for IP pool, configured
#                       max connections, fd limit, and conntrack headroom.
# v1.13.0-node-deploy - Added --network-only for config-driven forwarding/NAT
#                       maintenance.
# v1.12.0-node-deploy - Reads VPN subnet and TUN device from server.toml for
#                       forwarding/NAT setup.
# v1.11.0-node-deploy - Made registration-skipped installs return
#                       successfully.
# v1.10.0-node-deploy - Made --no-enable service installation return
#                       successfully after rendering the unit.
# v1.9.0-node-deploy - Added tracked dirty-worktree protection before updating
#                      an existing repository.
# v1.8.0-node-deploy - Verifies the generated network restore systemd unit
#                      before installing it.
# v1.7.0-node-deploy - Uses detected sysctl and iptables-restore paths in the
#                      generated network restore service.
# v1.6.0-node-deploy - Verifies the rendered systemd service unit before
#                      installing it.
# v1.5.0-node-deploy - Added shared deployment locking with upgrade.sh.
# v1.4.0-node-deploy - Persisted sysctl and iptables NAT with a restore unit.
# v1.3.0-node-deploy - Added --preflight-only for install readiness checks.
# v1.2.0-node-deploy - Added production preflight checks for host readiness.
# v1.1.1-node-deploy - Only checks/installs Rust when release build is enabled.
# v1.1.0-node-deploy - Added --skip-package-install and made --config-only avoid
#                      package/Rust installation by default.
# v1.0.0-node-deploy - Added production node installer.
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_URL="https://github.com/AeroNyxNetwork/AeroNyx.git"
DEFAULT_BRANCH="main"
DEFAULT_REPO_DIR="/opt/aeronyx/AeroNyx"
DEFAULT_CMS_URL="https://api.aeronyx.network/api/privacy_network"
CONFIG_DIR="/etc/aeronyx"
CONFIG_FILE="${CONFIG_DIR}/server.toml"
ENV_FILE="${CONFIG_DIR}/aeronyx.env"
STATE_DIR="/var/lib/aeronyx"
SERVICE_NAME="aeronyx-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
NETWORK_RESTORE_SERVICE="aeronyx-network-restore.service"
NETWORK_RESTORE_FILE="/etc/systemd/system/${NETWORK_RESTORE_SERVICE}"
SYSCTL_FILE="/etc/sysctl.d/99-aeronyx.conf"
IPTABLES_RULES_FILE="/etc/iptables/rules.v4"
LOCK_FILE="/run/lock/${SERVICE_NAME}.deploy.lock"
LOCK_DIR=""
SCRIPT_VERSION="v1.23.0-node-deploy"

REPO_URL="${AERONYX_REPO_URL:-${DEFAULT_REPO_URL}}"
BRANCH="${AERONYX_BRANCH:-${DEFAULT_BRANCH}}"
REPO_DIR="${AERONYX_REPO_DIR:-${DEFAULT_REPO_DIR}}"
REGISTRATION_CODE="${AERONYX_REGISTRATION_CODE:-}"
DO_BUILD=1
DO_NETWORK=1
DO_START=0
DO_ENABLE=1
INSTALL_RUST=1
INSTALL_PACKAGES=1
DRY_RUN=0
CONFIG_ONLY=0
PREFLIGHT_ONLY=0
ALLOW_DIRTY=0
NETWORK_ONLY=0
QUICK=0
PRINT_PLAN=0
SET_VPN_CIDR=""
CURRENT_INSTALL_STEP="not_started"
CURRENT_INSTALL_MESSAGE="Install has not started."

case "${AERONYX_START:-}" in
    1|true|TRUE|yes|YES|on|ON) DO_START=1 ;;
esac

if [ -n "${REGISTRATION_CODE}" ]; then
    DO_START=1
fi

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

install_progress_url() {
    local cms_url="${AERONYX_CMS_URL:-}"
    if [ -z "${cms_url}" ] && [ -f "${CONFIG_FILE}" ] && command -v python3 >/dev/null 2>&1; then
        cms_url="$(python3 - "${CONFIG_FILE}" <<'PY' 2>/dev/null || true
import sys
path = sys.argv[1]
current = None
for raw in open(path, "r", encoding="utf-8"):
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("[") and line.endswith("]"):
        current = line[1:-1].strip()
        continue
    if current == "management" and line.startswith("cms_url") and "=" in line:
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        print(value)
        break
PY
)"
    fi
    cms_url="${cms_url:-${DEFAULT_CMS_URL}}"
    cms_url="${cms_url%/}"
    printf '%s/codes/install-progress/\n' "${cms_url}"
}

report_install_progress() {
    local status_name="$1"
    local step="$2"
    local message="$3"
    local exit_code="${4:-}"

    [ -n "${REGISTRATION_CODE}" ] || return 0
    command -v curl >/dev/null 2>&1 || return 0
    command -v python3 >/dev/null 2>&1 || return 0

    local url payload command_mode host_name os_name arch_name
    url="$(install_progress_url)"
    command_mode="$(install_command_mode)"
    host_name="$(hostname 2>/dev/null || printf 'unknown')"
    os_name="$(uname -s 2>/dev/null || printf 'unknown')"
    arch_name="$(uname -m 2>/dev/null || printf 'unknown')"
    payload="$(REGISTRATION_CODE="${REGISTRATION_CODE}" \
        STATUS_NAME="${status_name}" \
        STEP_NAME="${step}" \
        MESSAGE_TEXT="${message}" \
        COMMAND_MODE="${command_mode}" \
        REPO_DIR_VALUE="${REPO_DIR}" \
        BRANCH_VALUE="${BRANCH}" \
        SERVICE_NAME_VALUE="${SERVICE_NAME}" \
        CONFIG_FILE_VALUE="${CONFIG_FILE}" \
        DRY_RUN_VALUE="${DRY_RUN}" \
        QUICK_VALUE="${QUICK}" \
        PRINT_PLAN_VALUE="${PRINT_PLAN}" \
        EXIT_CODE_VALUE="${exit_code}" \
        SCRIPT_VERSION_VALUE="${SCRIPT_VERSION}" \
        HOST_VALUE="${host_name}" \
        OS_VALUE="${os_name}" \
        ARCH_VALUE="${arch_name}" \
        python3 - <<'PY'
import json
import os

def env_bool(name):
    return os.environ.get(name, "0") == "1"

status = os.environ.get("STATUS_NAME", "running")
step = os.environ.get("STEP_NAME", "")
details = {
    "entrypoint": "deploy/node/install.sh",
    "command": os.environ.get("COMMAND_MODE", ""),
    "repo_dir": os.environ.get("REPO_DIR_VALUE", ""),
    "branch": os.environ.get("BRANCH_VALUE", ""),
    "service": os.environ.get("SERVICE_NAME_VALUE", ""),
    "config": os.environ.get("CONFIG_FILE_VALUE", ""),
    "dry_run": env_bool("DRY_RUN_VALUE"),
    "quick": env_bool("QUICK_VALUE"),
    "print_plan": env_bool("PRINT_PLAN_VALUE"),
    "script_version": os.environ.get("SCRIPT_VERSION_VALUE", ""),
    "host": os.environ.get("HOST_VALUE", ""),
    "os": os.environ.get("OS_VALUE", ""),
    "arch": os.environ.get("ARCH_VALUE", ""),
}

if status == "failed":
    details["failed_phase"] = step
    exit_code = os.environ.get("EXIT_CODE_VALUE", "")
    if exit_code:
        details["exit_code"] = exit_code

details = {key: value for key, value in details.items() if value not in ("", None)}

print(json.dumps({
    "code": os.environ["REGISTRATION_CODE"],
    "status": status,
    "step": step,
    "message": os.environ.get("MESSAGE_TEXT", ""),
    "details": details,
}))
PY
)"

    curl -fsS --max-time 4 -X POST "${url}" \
        -H 'Content-Type: application/json' \
        --data "${payload}" >/dev/null 2>&1 \
        || warn "Unable to report install progress to nodeboard (${step})."
}

set_install_step() {
    CURRENT_INSTALL_STEP="$1"
    CURRENT_INSTALL_MESSAGE="$2"
    report_install_progress "running" "${CURRENT_INSTALL_STEP}" "${CURRENT_INSTALL_MESSAGE}"
}

install_command_mode() {
    if [ "${PRINT_PLAN}" -eq 1 ]; then
        printf 'plan\n'
    elif [ "${NETWORK_ONLY}" -eq 1 ]; then
        printf 'network-only\n'
    elif [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
        printf 'preflight-only\n'
    elif [ "${CONFIG_ONLY}" -eq 1 ]; then
        printf 'config-only\n'
    elif [ "${QUICK}" -eq 1 ]; then
        printf 'quick-install\n'
    else
        printf 'install\n'
    fi
}

install_failed_trap() {
    local exit_code="$?"
    report_install_progress "failed" "${CURRENT_INSTALL_STEP}" "${CURRENT_INSTALL_MESSAGE} failed with exit code ${exit_code}." "${exit_code}"
    exit "${exit_code}"
}

usage() {
    cat <<'USAGE'
Usage:
  sudo ./deploy/node/install.sh [OPTIONS]

Options:
  --repo-url URL          Git repository URL. Default: https://github.com/AeroNyxNetwork/AeroNyx.git
  --branch NAME           Git branch or ref. Default: main
  --repo-dir PATH         Install repository path. Default: /opt/aeronyx/AeroNyx
  --registration-code C   Register node after build.
  --quick                 First-install shortcut. Requires --registration-code
                          or AERONYX_REGISTRATION_CODE and starts the service.
  --print-plan            Print resolved install options and exit without
                          requiring root, systemd, package install, build,
                          network changes, registration, or service start.
  --start                 Start service after install. Automatically enabled when --registration-code is used.
  --no-build              Skip cargo release build.
  --no-network            Skip sysctl and NAT setup.
  --no-enable             Do not enable systemd service.
  --skip-package-install  Do not install OS packages automatically.
  --skip-rust-install     Do not install Rust automatically if cargo is missing.
  --config-only           Only create config/env directories and server.toml if missing.
  --preflight-only        Run production readiness checks and exit.
  --network-only          Only refresh sysctl, iptables, and network restore service.
  --set-vpn-cidr CIDR     With --network-only, update vpn.virtual_ip_range in
                          /etc/aeronyx/server.toml before refreshing NAT.
                          Does not restart aeronyx-server.
  --allow-dirty           Allow install to update an existing repo with tracked Git changes.
  --dry-run               Print actions without changing the host.
  -h, --help              Show this help.

Examples:
  sudo ./deploy/node/install.sh --registration-code NYX-1234-ABCDE --start
  sudo AERONYX_REGISTRATION_CODE=NYX-1234-ABCDE ./deploy/node/install.sh --quick
  AERONYX_REGISTRATION_CODE=NYX-1234-ABCDE ./deploy/node/install.sh --quick --print-plan
  sudo ./deploy/node/install.sh --repo-dir /root/open/AeroNyx --no-build --no-network
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-url) REPO_URL="${2:?missing value}"; shift 2 ;;
        --branch) BRANCH="${2:?missing value}"; shift 2 ;;
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --registration-code) REGISTRATION_CODE="${2:?missing value}"; DO_START=1; shift 2 ;;
        --quick) QUICK=1; DO_START=1; shift ;;
        --print-plan) PRINT_PLAN=1; shift ;;
        --start) DO_START=1; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        --no-network) DO_NETWORK=0; shift ;;
        --no-enable) DO_ENABLE=0; shift ;;
        --skip-package-install) INSTALL_PACKAGES=0; shift ;;
        --skip-rust-install) INSTALL_RUST=0; shift ;;
        --config-only) CONFIG_ONLY=1; DO_BUILD=0; DO_NETWORK=0; DO_START=0; DO_ENABLE=0; INSTALL_PACKAGES=0; INSTALL_RUST=0; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; DO_BUILD=0; DO_NETWORK=0; DO_START=0; DO_ENABLE=0; INSTALL_PACKAGES=0; INSTALL_RUST=0; shift ;;
        --network-only) NETWORK_ONLY=1; DO_BUILD=0; DO_NETWORK=1; DO_START=0; DO_ENABLE=0; INSTALL_PACKAGES=0; INSTALL_RUST=0; shift ;;
        --set-vpn-cidr) SET_VPN_CIDR="${2:?missing value}"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
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

run_shell() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] %s\n' "$*"
    else
        bash -c "$*"
    fi
}

require_root() {
    [ "$(id -u)" -eq 0 ] || die "Please run as root, for example: sudo $0"
}

require_linux_systemd() {
    [ "$(uname -s)" = "Linux" ] || die "install.sh supports Linux production nodes only."
    command -v systemctl >/dev/null 2>&1 || die "systemctl is required for production node service management."
}

validate_option_combinations() {
    if [ "${NETWORK_ONLY}" -eq 1 ] && [ "${CONFIG_ONLY}" -eq 1 ]; then
        die "--network-only cannot be combined with --config-only."
    fi
    if [ "${NETWORK_ONLY}" -eq 1 ] && [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
        die "--network-only cannot be combined with --preflight-only."
    fi
    if [ "${NETWORK_ONLY}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ] && [ "${PRINT_PLAN}" -eq 0 ] && [ ! -f "${CONFIG_FILE}" ]; then
        die "--network-only requires existing config: ${CONFIG_FILE}"
    fi
    if [ -n "${SET_VPN_CIDR}" ] && [ "${NETWORK_ONLY}" -ne 1 ]; then
        die "--set-vpn-cidr must be combined with --network-only."
    fi
    if [ -n "${SET_VPN_CIDR}" ] && [ "${CONFIG_ONLY}" -eq 1 ]; then
        die "--set-vpn-cidr cannot be combined with --config-only."
    fi
    if [ -n "${SET_VPN_CIDR}" ] && [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
        die "--set-vpn-cidr cannot be combined with --preflight-only."
    fi
    if [ -n "${SET_VPN_CIDR}" ] && [ "${QUICK}" -eq 1 ]; then
        die "--set-vpn-cidr cannot be combined with --quick."
    fi
    if [ "${QUICK}" -eq 1 ] && [ -z "${REGISTRATION_CODE}" ]; then
        die "--quick requires --registration-code or AERONYX_REGISTRATION_CODE."
    fi
    if [ "${QUICK}" -eq 1 ] && { [ "${CONFIG_ONLY}" -eq 1 ] || [ "${PREFLIGHT_ONLY}" -eq 1 ] || [ "${NETWORK_ONLY}" -eq 1 ]; }; then
        die "--quick cannot be combined with --config-only, --preflight-only, or --network-only."
    fi
}

bool_word() {
    if [ "${1:-0}" -eq 1 ]; then
        printf 'yes\n'
    else
        printf 'no\n'
    fi
}

print_install_plan() {
    cat <<PLAN
AeroNyx node install plan
repo_url=${REPO_URL}
branch=${BRANCH}
repo_dir=${REPO_DIR}
config_file=${CONFIG_FILE}
service_name=${SERVICE_NAME}
quick=$(bool_word "${QUICK}")
config_only=$(bool_word "${CONFIG_ONLY}")
preflight_only=$(bool_word "${PREFLIGHT_ONLY}")
network_only=$(bool_word "${NETWORK_ONLY}")
build=$(bool_word "${DO_BUILD}")
network=$(bool_word "${DO_NETWORK}")
enable_service=$(bool_word "${DO_ENABLE}")
start_service=$(bool_word "${DO_START}")
install_packages=$(bool_word "${INSTALL_PACKAGES}")
install_rust=$(bool_word "${INSTALL_RUST}")
allow_dirty=$(bool_word "${ALLOW_DIRTY}")
dry_run=$(bool_word "${DRY_RUN}")
set_vpn_cidr=$([ -n "${SET_VPN_CIDR}" ] && printf '%s' "${SET_VPN_CIDR}" || printf 'none')
registration_code_present=$([ -n "${REGISTRATION_CODE}" ] && printf 'yes' || printf 'no')
registration_code_value=hidden
PLAN
}

resolve_command_path() {
    local cmd="$1"
    local path
    path="$(command -v "${cmd}" 2>/dev/null || true)"
    [ -n "${path}" ] || die "Required command not found: ${cmd}"
    printf '%s\n' "${path}"
}

release_lock() {
    if [ -n "${LOCK_DIR}" ] && [ -d "${LOCK_DIR}" ]; then
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}

acquire_deploy_lock() {
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

existing_path_for_df() {
    local path="$1"
    while [ ! -e "${path}" ] && [ "${path}" != "/" ]; do
        path="$(dirname "${path}")"
    done
    printf '%s\n' "${path}"
}

config_value() {
    local section="$1" key="$2" default_value="$3"
    if ! command -v python3 >/dev/null 2>&1 || [ ! -f "${CONFIG_FILE}" ]; then
        printf '%s\n' "${default_value}"
        return
    fi

    python3 - "${CONFIG_FILE}" "${section}" "${key}" "${default_value}" <<'PY'
import sys

path, section, key, default = sys.argv[1:5]

def fallback_parse(text):
    current = None
    values = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            continue
        if current == section and "=" in line:
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            values[name] = value
    return values.get(key, default)

try:
    raw = open(path, "rb").read()
    try:
        import tomllib
        data = tomllib.loads(raw.decode("utf-8"))
        value = data.get(section, {}).get(key, default)
    except Exception:
        value = fallback_parse(raw.decode("utf-8", "replace"))
    print(value if value is not None else default)
except Exception:
    print(default)
PY
}

config_cidr() {
    local raw default_value="$1"
    if [ -n "${SET_VPN_CIDR}" ]; then
        normalize_cidr_arg "${SET_VPN_CIDR}"
        return
    fi

    raw="$(config_value vpn virtual_ip_range "${default_value}")"
    if ! command -v python3 >/dev/null 2>&1; then
        if printf '%s' "${raw}" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}/([0-9]|[12][0-9]|3[0-2])$'; then
            printf '%s\n' "${raw}"
        else
            printf '%s\n' "${default_value}"
        fi
        return
    fi

    python3 - "${raw}" "${default_value}" <<'PY'
import ipaddress
import sys

raw, default = sys.argv[1:3]
try:
    print(ipaddress.ip_network(raw, strict=False).with_prefixlen)
except Exception:
    print(default)
PY
}

normalize_cidr_arg() {
    local raw="$1"
    command -v python3 >/dev/null 2>&1 || die "python3 is required to validate --set-vpn-cidr."
    python3 - "${raw}" <<'PY'
import ipaddress
import sys

raw = sys.argv[1]
try:
    network = ipaddress.ip_network(raw, strict=False)
except Exception as exc:
    raise SystemExit(f"invalid CIDR {raw!r}: {exc}")

if network.version != 4:
    raise SystemExit("AeroNyx VPN pools must be IPv4 CIDR ranges.")

print(network.with_prefixlen)
PY
}

update_vpn_cidr_config() {
    local normalized backup_path timestamp
    [ -n "${SET_VPN_CIDR}" ] || return

    normalized="$(normalize_cidr_arg "${SET_VPN_CIDR}")"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup_path="${CONFIG_FILE}.bak.${timestamp}.vpn_cidr"

    log "Updating ${CONFIG_FILE} vpn.virtual_ip_range to ${normalized}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] cp -a %s %s\n' "${CONFIG_FILE}" "${backup_path}"
        printf '[DRY-RUN] set [vpn] virtual_ip_range = \"%s\" in %s\n' "${normalized}" "${CONFIG_FILE}"
        return
    fi

    [ -f "${CONFIG_FILE}" ] || die "Cannot update VPN CIDR; config not found: ${CONFIG_FILE}"
    cp -a "${CONFIG_FILE}" "${backup_path}"
    python3 - "${CONFIG_FILE}" "${normalized}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
cidr = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
out = []
current = None
in_vpn = False
updated = False
inserted = False

for raw in lines:
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if in_vpn and not updated and not inserted:
            out.append(f'virtual_ip_range = "{cidr}"')
            inserted = True
        current = stripped[1:-1].strip()
        in_vpn = current == "vpn"
        out.append(raw)
        continue

    if in_vpn and stripped.startswith("virtual_ip_range") and "=" in stripped:
        prefix = raw[: len(raw) - len(raw.lstrip())]
        comment = ""
        if "#" in raw:
            comment_text = raw.split("#", 1)[1].strip()
            comment = f" # {comment_text}" if comment_text else ""
        out.append(f'{prefix}virtual_ip_range = "{cidr}"{comment}')
        updated = True
        continue

    out.append(raw)

if in_vpn and not updated and not inserted:
    out.append(f'virtual_ip_range = "{cidr}"')
    inserted = True

if not updated and not inserted:
    raise SystemExit("Missing [vpn] section in config; refusing to update.")

path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
    chmod 600 "${CONFIG_FILE}"
    ok "VPN CIDR updated; backup saved: ${backup_path}"
    warn "Rust/TUN capacity changes after a controlled ${SERVICE_NAME} restart. Use maintenance mode and wait for active sessions to drain first."
}

config_tun_device() {
    local raw default_value="$1"
    raw="$(config_value tun device_name "${default_value}")"
    if printf '%s' "${raw}" | grep -Eq '^[A-Za-z0-9_.-]{1,15}$'; then
        printf '%s\n' "${raw}"
    else
        warn "Invalid tun.device_name in ${CONFIG_FILE}: ${raw}; using ${default_value}"
        printf '%s\n' "${default_value}"
    fi
}

config_uint() {
    local section="$1" key="$2" default_value="$3" raw
    raw="$(config_value "${section}" "${key}" "${default_value}")"
    if printf '%s' "${raw}" | grep -Eq '^[0-9]+$'; then
        printf '%s\n' "${raw}"
    else
        warn "Invalid ${section}.${key} in ${CONFIG_FILE}: ${raw}; using ${default_value}"
        printf '%s\n' "${default_value}"
    fi
}

cidr_usable_client_ips() {
    local cidr="$1"
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'unknown\n'
        return
    fi

    python3 - "${cidr}" <<'PY'
import ipaddress
import sys

try:
    network = ipaddress.ip_network(sys.argv[1], strict=False)
    hosts = sum(1 for _ in network.hosts())
    # AeroNyx reserves one usable host address for the gateway inside the pool.
    print(max(hosts - 1, 0))
except Exception:
    print("unknown")
PY
}

read_uint_file() {
    local path="$1" fallback="$2" value
    if [ -r "${path}" ]; then
        value="$(cat "${path}" 2>/dev/null || true)"
        if printf '%s' "${value}" | grep -Eq '^[0-9]+$'; then
            printf '%s\n' "${value}"
            return
        fi
    fi
    printf '%s\n' "${fallback}"
}

capacity_warn() {
    printf '[WARN] %s\n' "$*"
}

service_limit_nofile() {
    local candidate value
    for candidate in \
        "${SERVICE_FILE}" \
        "${SCRIPT_DIR}/aeronyx-server.service" \
        "${REPO_DIR}/deploy/node/aeronyx-server.service"
    do
        if [ -r "${candidate}" ]; then
            value="$(grep -E '^[[:space:]]*LimitNOFILE=' "${candidate}" 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d '[:space:]' || true)"
            if printf '%s' "${value}" | grep -Eq '^[0-9]+$'; then
                printf '%s\n' "${value}"
                return
            fi
        fi
    done
    printf 'unknown\n'
}

capacity_plan_checks() {
    local vpn_subnet tun_device max_connections ip_capacity shell_fd_soft
    local shell_fd_hard service_fd_limit effective_fd_limit conntrack_max
    local conntrack_used recommended_fd recommended_conntrack

    vpn_subnet="$(config_cidr "100.64.0.0/22")"
    tun_device="$(config_tun_device "aeronyx0")"
    max_connections="$(config_uint limits max_connections 1000)"
    ip_capacity="$(cidr_usable_client_ips "${vpn_subnet}")"
    shell_fd_soft="$(ulimit -Sn 2>/dev/null || printf 'unknown')"
    shell_fd_hard="$(ulimit -Hn 2>/dev/null || printf 'unknown')"
    service_fd_limit="$(service_limit_nofile)"
    effective_fd_limit="${service_fd_limit}"
    if ! printf '%s' "${effective_fd_limit}" | grep -Eq '^[0-9]+$'; then
        effective_fd_limit="${shell_fd_soft}"
    fi
    conntrack_max="$(read_uint_file /proc/sys/net/netfilter/nf_conntrack_max unknown)"
    conntrack_used="$(read_uint_file /proc/sys/net/netfilter/nf_conntrack_count 0)"
    recommended_fd=$((max_connections * 4 + 1024))
    recommended_conntrack=$((max_connections * 8))

    log "Commercial capacity plan"
    ok "VPN pool ${vpn_subnet} on ${tun_device}; usable client IPs: ${ip_capacity}; configured max_connections: ${max_connections}"
    ok "Host limits: service LimitNOFILE=${service_fd_limit}; shell fd soft=${shell_fd_soft} hard=${shell_fd_hard}; conntrack used=${conntrack_used} max=${conntrack_max}"

    if [ "${ip_capacity}" != "unknown" ] && [ "${max_connections}" -gt "${ip_capacity}" ]; then
        capacity_warn "max_connections (${max_connections}) exceeds usable client IPs (${ip_capacity}). Expand vpn.virtual_ip_range or lower limits.max_connections before commercial placement."
    fi

    if printf '%s' "${effective_fd_limit}" | grep -Eq '^[0-9]+$' && [ "${effective_fd_limit}" -lt "${recommended_fd}" ]; then
        capacity_warn "Effective file descriptor limit (${effective_fd_limit}) is below recommended headroom (${recommended_fd}) for ${max_connections} configured sessions."
    fi

    if printf '%s' "${conntrack_max}" | grep -Eq '^[0-9]+$' && [ "${conntrack_max}" -lt "${recommended_conntrack}" ]; then
        capacity_warn "nf_conntrack_max (${conntrack_max}) is below recommended headroom (${recommended_conntrack}) for ${max_connections} configured sessions."
    fi
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

preflight_checks() {
    local default_iface disk_path disk_free_mb mem_mb

    log "Running production preflight checks"

    if [ -e /dev/net/tun ]; then
        ok "TUN device available: /dev/net/tun"
    else
        warn "TUN device missing: /dev/net/tun. Enable the tun kernel module before starting the VPN node."
    fi

    default_iface="$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')"
    if [ -n "${default_iface}" ]; then
        ok "Default route interface detected: ${default_iface}"
    else
        warn "No default route detected. VPN client internet forwarding may fail."
    fi

    mem_mb="$(awk '/MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || printf '0')"
    if [ "${mem_mb}" -ge 2048 ]; then
        ok "Memory check: ${mem_mb} MB"
    else
        warn "Memory check: ${mem_mb} MB. 2GB+ is recommended for a production VPN node; MemChain models require more."
    fi

    disk_path="$(existing_path_for_df "${REPO_DIR}")"
    disk_free_mb="$(df -Pm "${disk_path}" 2>/dev/null | awk 'NR==2 {print $4}' || printf '0')"
    if [ "${disk_free_mb:-0}" -ge 4096 ]; then
        ok "Disk check: ${disk_free_mb} MB free near ${disk_path}"
    else
        warn "Disk check: ${disk_free_mb:-0} MB free near ${disk_path}. 4GB+ is recommended for build artifacts and backups."
    fi

    if port_in_use 51820; then
        warn "Port 51820 already appears to be listening. If this is an existing AeroNyx node, this is expected during reinstall."
    else
        ok "Port 51820 appears available"
    fi

    if port_in_use 8421; then
        warn "Port 8421 already appears to be listening. Existing AeroNyx API service may already be running."
    else
        ok "Port 8421 appears available"
    fi

    capacity_plan_checks
}

install_packages() {
    [ "${INSTALL_PACKAGES}" -eq 1 ] || { ok "Package installation skipped"; return; }

    if command -v apt-get >/dev/null 2>&1; then
        log "Installing Debian/Ubuntu host dependencies"
        run apt-get update
        run apt-get install -y ca-certificates curl git build-essential pkg-config libssl-dev iproute2 iptables
    elif command -v dnf >/dev/null 2>&1; then
        log "Installing Fedora/RHEL host dependencies"
        run dnf install -y ca-certificates curl git gcc gcc-c++ make pkg-config openssl-devel iproute iptables
    elif command -v yum >/dev/null 2>&1; then
        log "Installing RHEL/CentOS host dependencies"
        run yum install -y ca-certificates curl git gcc gcc-c++ make pkgconfig openssl-devel iproute iptables
    else
        warn "Unsupported package manager. Please install: curl git gcc make pkg-config openssl-dev iproute2 iptables"
    fi
}

install_rust_if_needed() {
    if command -v cargo >/dev/null 2>&1; then
        ok "Rust cargo found: $(command -v cargo)"
        return
    fi

    if [ -x "${HOME}/.cargo/bin/cargo" ]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
        ok "Rust cargo found after PATH update: ${HOME}/.cargo/bin/cargo"
        return
    fi

    [ "${INSTALL_RUST}" -eq 1 ] || die "cargo not found and --skip-rust-install was set."
    command -v curl >/dev/null 2>&1 || die "curl is required to install rustup."

    log "Installing Rust toolchain with rustup"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] curl https://sh.rustup.rs -sSf | sh -s -- -y\n'
    else
        curl https://sh.rustup.rs -sSf | sh -s -- -y
        # shellcheck disable=SC1091
        . "${HOME}/.cargo/env"
    fi
}

ensure_tracked_worktree_clean() {
    [ "${ALLOW_DIRTY}" -eq 0 ] || { warn "Tracked worktree check skipped by --allow-dirty"; return; }

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] verify tracked Git worktree is clean in %s\n' "${REPO_DIR}"
        return
    fi

    git -C "${REPO_DIR}" diff --quiet --ignore-submodules -- \
        || die "Tracked Git worktree has unstaged changes. Commit/stash them or re-run with --allow-dirty."
    git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules -- \
        || die "Tracked Git worktree has staged changes. Commit/stash them or re-run with --allow-dirty."
    ok "Tracked Git worktree clean"
}

prepare_repo() {
    if [ -d "${REPO_DIR}/.git" ]; then
        log "Using existing repository: ${REPO_DIR}"
        ensure_tracked_worktree_clean
        run git -C "${REPO_DIR}" fetch origin "${BRANCH}"
        run git -C "${REPO_DIR}" checkout "${BRANCH}"
        run git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
    else
        log "Cloning AeroNyx repository into ${REPO_DIR}"
        run mkdir -p "$(dirname "${REPO_DIR}")"
        run git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
    fi

    [ "${DRY_RUN}" -eq 1 ] || [ -f "${REPO_DIR}/Cargo.toml" ] || die "Cargo.toml not found in ${REPO_DIR}"
}

prepare_directories() {
    log "Preparing runtime directories"
    run mkdir -p "${CONFIG_DIR}" "${STATE_DIR}" "/var/log/aeronyx"
    run chmod 700 "${CONFIG_DIR}"
    run chmod 700 "${STATE_DIR}"
}

install_config() {
    local template="${REPO_DIR}/deploy/node/server.example.toml"
    [ "${DRY_RUN}" -eq 1 ] || [ -f "${template}" ] || die "Missing config template: ${template}"

    if [ -f "${CONFIG_FILE}" ]; then
        ok "Existing config preserved: ${CONFIG_FILE}"
    else
        log "Installing default config: ${CONFIG_FILE}"
        run cp "${template}" "${CONFIG_FILE}"
        run chmod 600 "${CONFIG_FILE}"
    fi

    if [ -f "${ENV_FILE}" ]; then
        ok "Existing environment file preserved: ${ENV_FILE}"
    else
        log "Creating optional environment file: ${ENV_FILE}"
        if [ "${DRY_RUN}" -eq 1 ]; then
            printf '[DRY-RUN] create %s\n' "${ENV_FILE}"
        else
            cat >"${ENV_FILE}" <<'ENV'
# AeroNyx optional service environment.
# Keep secrets here instead of committing them to Git.
#
# Current Rust configuration is TOML-driven. These environment values are
# reserved for deployment automation and future runtime overrides.
ENV
            chmod 600 "${ENV_FILE}"
        fi
    fi
}

configure_network() {
    [ "${DO_NETWORK}" -eq 1 ] || { ok "Network setup skipped"; return; }

    local vpn_subnet tun_device default_iface
    vpn_subnet="$(config_cidr "100.64.0.0/22")"
    tun_device="$(config_tun_device "aeronyx0")"
    default_iface="$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')"
    [ -n "${default_iface}" ] || default_iface="eth0"

    log "Enabling IPv4 forwarding"
    run_shell "printf '1\n' > /proc/sys/net/ipv4/ip_forward"
    run_shell "printf '%s\n' '# AeroNyx VPN forwarding' 'net.ipv4.ip_forward=1' > '${SYSCTL_FILE}'"
    run sysctl -w net.ipv4.ip_forward=1

    if command -v iptables >/dev/null 2>&1; then
        log "Applying idempotent iptables NAT rules for ${vpn_subnet} on ${default_iface} via ${tun_device}"
        run_shell "iptables -t nat -C POSTROUTING -s '${vpn_subnet}' -o '${default_iface}' -j MASQUERADE 2>/dev/null || iptables -t nat -A POSTROUTING -s '${vpn_subnet}' -o '${default_iface}' -j MASQUERADE"
        cleanup_stale_aeronyx_nat_rules "${vpn_subnet}" "${default_iface}"
        run_shell "iptables -C FORWARD -i '${tun_device}' -j ACCEPT 2>/dev/null || iptables -A FORWARD -i '${tun_device}' -j ACCEPT"
        run_shell "iptables -C FORWARD -o '${tun_device}' -j ACCEPT 2>/dev/null || iptables -A FORWARD -o '${tun_device}' -j ACCEPT"
        persist_network_rules
    else
        warn "iptables not found; VPN clients may not reach the internet until NAT is configured."
    fi
}

cleanup_stale_aeronyx_nat_rules() {
    local vpn_subnet="$1" default_iface="$2"
    local out_iface rule source delete_rule

    command -v iptables-save >/dev/null 2>&1 || return

    iptables-save -t nat 2>/dev/null \
        | awk '/^-A POSTROUTING / && / -j MASQUERADE/ {print}' \
        | while IFS= read -r rule; do
            source="$(printf '%s\n' "${rule}" | sed -n 's/.* -s \([^ ]*\) .*/\1/p')"
            out_iface="$(printf '%s\n' "${rule}" | sed -n 's/.* -o \([^ ]*\) .*/\1/p')"

            case "${source}" in
                100.64.0.0/*) ;;
                *) continue ;;
            esac

            [ "${source}" != "${vpn_subnet}" ] || continue
            [ "${out_iface}" = "${default_iface}" ] || continue

            delete_rule="$(printf '%s\n' "${rule}" | sed 's/^-A /-D /')"
            log "Removing stale AeroNyx NAT rule for ${source} on ${default_iface}"
            run_shell "iptables -t nat ${delete_rule}"
        done
}

persist_network_rules() {
    if ! command -v iptables-save >/dev/null 2>&1; then
        warn "iptables-save not found; NAT rules are active but not persisted."
        return
    fi

    log "Persisting iptables rules to ${IPTABLES_RULES_FILE}"
    run mkdir -p "$(dirname "${IPTABLES_RULES_FILE}")"
    run_shell "iptables-save > '${IPTABLES_RULES_FILE}'"
    install_network_restore_service
}

render_network_restore_unit() {
    local output_file="$1"
    local sysctl_path="$2"
    local iptables_restore_path="$3"

    cat > "${output_file}" <<SERVICE
# ============================================
# File: /etc/systemd/system/${NETWORK_RESTORE_SERVICE}
# ============================================
# Creation Reason:
# - Restore AeroNyx VPN forwarding/NAT rules on host boot.
#
# Main Functionality:
# - Applies sysctl forwarding from ${SYSCTL_FILE}.
# - Restores iptables rules from ${IPTABLES_RULES_FILE}.
#
# Important Note for Next Developer:
# - This service is generated by deploy/node/install.sh.
# - It does not inspect or log client traffic, DNS contents, destinations, or
#   packet payloads.
# ============================================

[Unit]
Description=AeroNyx VPN network restore
DefaultDependencies=no
After=local-fs.target
Before=network-pre.target
Wants=network-pre.target
ConditionPathExists=${IPTABLES_RULES_FILE}

[Service]
Type=oneshot
ExecStart=${sysctl_path} -w net.ipv4.ip_forward=1
ExecStart=${iptables_restore_path} ${IPTABLES_RULES_FILE}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SERVICE
}

install_network_restore_service() {
    local sysctl_path iptables_restore_path rendered
    sysctl_path="$(resolve_command_path sysctl)"
    iptables_restore_path="$(resolve_command_path iptables-restore)"
    rendered="/tmp/aeronyx-network-restore.install.service"

    log "Installing network restore service: ${NETWORK_RESTORE_FILE}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] create %s\n' "${NETWORK_RESTORE_FILE}"
        printf '[DRY-RUN] network restore commands: sysctl=%s iptables-restore=%s\n' "${sysctl_path}" "${iptables_restore_path}"
        printf '[DRY-RUN] systemd-analyze verify %s\n' "${rendered}"
    else
        render_network_restore_unit "${rendered}" "${sysctl_path}" "${iptables_restore_path}"
        systemd-analyze verify "${rendered}"
        cp "${rendered}" "${NETWORK_RESTORE_FILE}"
        chmod 644 "${NETWORK_RESTORE_FILE}"
        systemctl daemon-reload
    fi
    run systemctl enable "${NETWORK_RESTORE_SERVICE}"
}

resolve_build_git_commit() {
    git -C "${REPO_DIR}" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown'
}

build_binary() {
    local build_git_commit
    [ "${DO_BUILD}" -eq 1 ] || { ok "Build skipped"; return; }

    build_git_commit="$(resolve_build_git_commit)"
    log "Building aeronyx-server release binary"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] cd %s && AERONYX_GIT_COMMIT=%s cargo build -p aeronyx-server --release\n' "${REPO_DIR}" "${build_git_commit}"
    else
        (
            cd "${REPO_DIR}"
            AERONYX_GIT_COMMIT="${build_git_commit}" cargo build -p aeronyx-server --release
        )
    fi
}

install_service() {
    local template="${REPO_DIR}/deploy/node/aeronyx-server.service"
    local rendered="/tmp/${SERVICE_NAME}.install.service"
    [ "${DRY_RUN}" -eq 1 ] || [ -f "${template}" ] || die "Missing service template: ${template}"

    log "Installing systemd service: ${SERVICE_FILE}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] render %s to %s\n' "${template}" "${SERVICE_FILE}"
        printf '[DRY-RUN] systemd-analyze verify %s\n' "${rendered}"
    else
        sed \
            -e "s|@REPO_DIR@|${REPO_DIR}|g" \
            -e "s|@CONFIG_FILE@|${CONFIG_FILE}|g" \
            "${template}" > "${rendered}"
        systemd-analyze verify "${rendered}"
        cp "${rendered}" "${SERVICE_FILE}"
        chmod 644 "${SERVICE_FILE}"
        systemctl daemon-reload
    fi

    if [ "${DO_ENABLE}" -eq 1 ]; then
        run systemctl enable "${SERVICE_NAME}"
    else
        ok "Systemd enable skipped"
    fi
}

register_node() {
    [ -n "${REGISTRATION_CODE}" ] || { ok "Node registration skipped"; return 0; }

    log "Registering node with provided registration code"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] %s/target/release/aeronyx-server register --code *** -c %s\n' "${REPO_DIR}" "${CONFIG_FILE}"
    else
        "${REPO_DIR}/target/release/aeronyx-server" register --code "${REGISTRATION_CODE}" -c "${CONFIG_FILE}"
    fi
}

start_service() {
    [ "${DO_START}" -eq 1 ] || {
        ok "Install complete. Start after registration with: systemctl start ${SERVICE_NAME}"
        return
    }

    log "Starting ${SERVICE_NAME}"
    run systemctl restart "${SERVICE_NAME}"
    run systemctl --no-pager --full status "${SERVICE_NAME}"
}

main() {
    validate_option_combinations
    if [ "${PRINT_PLAN}" -eq 1 ]; then
        report_install_progress "planning" "plan" "Install plan generated; waiting for operator approval."
        print_install_plan
        exit 0
    fi

    trap install_failed_trap ERR
    set_install_step "preflight" "Running host preflight checks."
    require_root
    require_linux_systemd
    preflight_checks
    if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
        ok "Preflight-only checks complete."
        report_install_progress "completed" "preflight" "Preflight-only checks complete."
        exit 0
    fi
    acquire_deploy_lock
    if [ "${NETWORK_ONLY}" -eq 1 ]; then
        set_install_step "network" "Refreshing AeroNyx privacy protocol network rules."
        update_vpn_cidr_config
        if [ -n "${SET_VPN_CIDR}" ]; then
            capacity_plan_checks
        fi
        configure_network
        ok "Network-only maintenance complete."
        report_install_progress "completed" "network" "Network-only maintenance complete."
        if [ -n "${SET_VPN_CIDR}" ]; then
            warn "Persisted VPN CIDR is updated, but running Rust/TUN state changes only after a controlled ${SERVICE_NAME} restart."
        fi
        exit 0
    fi

    set_install_step "dependencies" "Installing host dependencies and checking Rust toolchain."
    install_packages
    if [ "${DO_BUILD}" -eq 1 ]; then
        install_rust_if_needed
    else
        ok "Rust toolchain check skipped"
    fi
    set_install_step "repository" "Preparing AeroNyx repository and configuration."
    prepare_repo
    prepare_directories
    install_config

    if [ "${CONFIG_ONLY}" -eq 1 ]; then
        ok "Config-only install complete."
        report_install_progress "completed" "config" "Config-only install complete."
        exit 0
    fi

    set_install_step "network" "Configuring forwarding and NAT."
    configure_network
    set_install_step "build" "Building AeroNyx Rust release binary."
    build_binary
    set_install_step "systemd" "Installing and verifying systemd service."
    install_service
    set_install_step "register" "Registering node with nodeboard."
    register_node
    set_install_step "start" "Starting or verifying AeroNyx service."
    start_service
    report_install_progress "completed" "completed" "Install workflow completed."
    trap - ERR
}

main "$@"
