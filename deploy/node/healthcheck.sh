#!/usr/bin/env bash
# ============================================
# File: deploy/node/healthcheck.sh
# ============================================
# Creation Reason:
# - Provide a single operator-facing diagnostic command for AeroNyx Rust privacy
#   nodes after install, upgrade, or incident response.
#
# Modification Reason:
# - Initial production healthcheck for node deployment workflows.
#
# Main Functionality:
# - Checks repository, binary, config, registration state, systemd status,
#   host capacity, IP forwarding, NAT hints, local VPN health endpoint, and
#   capacity telemetry.
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
# 3. Query local health endpoint and summarize capacity if available.
#
# Important Note for Next Developer:
# - Do not print private keys, registration secrets, API secrets, wallet-level
#   traffic, DNS contents, destinations, payloads, or client public IPs.
# - Keep output stable; nodeboard or support tooling may parse these lines later.
# - This script should never modify host state.
#
# Last Modified:
# v1.1.0-node-deploy - Added host capacity, TUN, route, disk, and port checks.
# v1.0.0-node-deploy - Added production node healthcheck.
# ============================================

set -euo pipefail

REPO_DIR="/opt/aeronyx/AeroNyx"
CONFIG_FILE="/etc/aeronyx/server.toml"
SERVICE_NAME="aeronyx-server"
JSON=0

usage() {
    cat <<'USAGE'
Usage:
  ./deploy/node/healthcheck.sh [OPTIONS]

Options:
  --repo-dir PATH   Repository path. Default: /opt/aeronyx/AeroNyx
  --config PATH     Config path. Default: /etc/aeronyx/server.toml
  --service NAME    systemd service name. Default: aeronyx-server
  --json            Emit compact JSON summary when python3 is available.
  -h, --help        Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --config) CONFIG_FILE="${2:?missing value}"; shift 2 ;;
        --service) SERVICE_NAME="${2:?missing value}"; shift 2 ;;
        --json) JSON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[ERROR] Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

PASS=0
WARN=0
FAIL=0

pass() { PASS=$((PASS + 1)); printf '[PASS] %s\n' "$*"; }
warn() { WARN=$((WARN + 1)); printf '[WARN] %s\n' "$*" >&2; }
fail() { FAIL=$((FAIL + 1)); printf '[FAIL] %s\n' "$*" >&2; }

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

check_network() {
    local forwarding
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
        if command -v python3 >/dev/null 2>&1; then
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
        python3 - <<PY
import json
print(json.dumps({"pass": ${PASS}, "warn": ${WARN}, "fail": ${FAIL}}, separators=(",", ":")))
PY
    fi
}

main() {
    check_system
    check_host_capacity
    check_repo_and_binary
    check_config_validation
    check_systemd
    check_network
    check_health_endpoint

    printf '[SUMMARY] pass=%s warn=%s fail=%s\n' "${PASS}" "${WARN}" "${FAIL}"
    emit_json_summary
    [ "${FAIL}" -eq 0 ]
}

main "$@"
