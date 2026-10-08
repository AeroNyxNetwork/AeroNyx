#!/usr/bin/env bash
# ============================================
# scripts/init.sh — AeroNyx MemChain
# ============================================
# Interactive setup wizard that:
#   1. Detects system resources (CPU, RAM, disk)
#   2. Explains the Phala-only inference boundary (no local model downloads)
#   3. Asks local storage/API preferences; inference remains client-to-Phala
#   4. Generates server.toml configuration
#   5. Configures IP forwarding + NAT (VPN routing)
#   6. Optionally builds and starts the server
#
# Usage:
#   chmod +x scripts/init.sh
#   ./scripts/init.sh              # Interactive mode
#   ./scripts/init.sh --defaults   # Non-interactive, all defaults
#   ./scripts/init.sh --help       # Show help
#
# Last Modified:
# v2.6.1 - Added AeroNyx VPN DNS stub setup for in-tunnel DNS
# v2.6.0 - Added Step 4.5: IP forwarding + iptables NAT setup

set -euo pipefail
umask 077

# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] This wizard writes credentials;
# keep generated configuration and backups private by default.

# ── Colors ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()      { echo -e "${GREEN}[✅]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }
# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] Keep prompts off stdout so
# command substitutions capture only values, never prompt text.
ask()     { printf '%b ' "${BOLD}$*${NC}" >&2; }

# ── Resolve paths ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="/etc/aeronyx"
CONFIG_FILE="${CONFIG_DIR}/server.toml"
DB_DIR="/var/lib/aeronyx"

# ── Parse arguments ────────────────────────────────────────────
USE_DEFAULTS=false
for arg in "$@"; do
    case "${arg}" in
        --defaults) USE_DEFAULTS=true ;;
        --help|-h)
            echo "Usage: $0 [--defaults|--help]"
            echo ""
            echo "Interactive setup wizard for AeroNyx MemChain."
            echo ""
            echo "  --defaults    Use all default values (non-interactive)"
            echo "  --help        Show this help"
            exit 0
            ;;
    esac
done

# ── Prompt helper ──────────────────────────────────────────────
prompt() {
    local question="$1"
    local default="$2"

    if [ "${USE_DEFAULTS}" = true ]; then
        echo "${default}"
        return
    fi

    local answer
    ask "${question} [${default}]: "
    read -r answer
    echo "${answer:-${default}}"
}

prompt_yn() {
    local question="$1"
    local default="$2"

    if [ "${USE_DEFAULTS}" = true ]; then
        [ "${default}" = "y" ] && return 0 || return 1
    fi

    local hint
    if [ "${default}" = "y" ]; then hint="Y/n"; else hint="y/N"; fi

    ask "${question} [${hint}]: "
    local answer
    read -r answer
    answer="${answer:-${default}}"

    case "${answer}" in
        [yY]*) return 0 ;;
        *)     return 1 ;;
    esac
}

# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] Read credentials without terminal
# echo; never route their prompt text through a captured stdout value.
prompt_secret() {
    local question="$1"
    local answer
    if [ "${USE_DEFAULTS}" = true ]; then
        printf '%s' ""
        return
    fi
    printf '%b' "${BOLD}${question}${NC} " >&2
    IFS= read -r -s answer
    printf '\n' >&2
    printf '%s' "${answer}"
}

# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] Quote operator strings as TOML
# basic strings rather than allowing key/model text to alter generated config.
toml_quote() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\t'/\\t}
    value=${value//$'\r'/\\r}
    value=${value//$'\n'/\\n}
    printf '"%s"' "${value}"
}

# ── System detection ───────────────────────────────────────────
detect_system() {
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    RAM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1048576)}' || echo 4096)
    DISK_FREE_GB=$(df -BG "${PROJECT_ROOT}" 2>/dev/null | tail -1 | awk '{print int($4)}' || echo 10)
    ARCH=$(uname -m)
    OS=$(uname -s)
}

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

clear 2>/dev/null || true
echo ""
echo -e "${BOLD}${CYAN}"
echo "    ╔══════════════════════════════════════════╗"
echo "    ║                                          ║"
echo "    ║     AeroNyx MemChain — Setup Wizard      ║"
echo "    ║     AI Cognitive Engine v2.6.0            ║"
echo "    ║                                          ║"
echo "    ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ════════════════════════════════════════════════════════════════
# Step 1: System Detection
# ════════════════════════════════════════════════════════════════

header "Step 1/5 — System Detection"

detect_system

echo -e "  CPU:      ${BOLD}${CPU_CORES} cores${NC}"
echo -e "  RAM:      ${BOLD}${RAM_MB} MB${NC}"
echo -e "  Disk:     ${BOLD}${DISK_FREE_GB} GB free${NC}"
echo -e "  Arch:     ${BOLD}${ARCH}${NC}"
echo -e "  OS:       ${BOLD}${OS}${NC}"
echo -e "  Project:  ${DIM}${PROJECT_ROOT}${NC}"
echo ""

if [ "${RAM_MB}" -lt 3072 ]; then
    warn "Low RAM (${RAM_MB}MB). Minimum 4GB recommended."
fi
if [ "${DISK_FREE_GB}" -lt 3 ]; then
    warn "Low disk space (${DISK_FREE_GB}GB). Leave room for the database and durable queues."
fi

# ════════════════════════════════════════════════════════════════
# Step 2: Inference Boundary
# ════════════════════════════════════════════════════════════════

header "Step 2/5 — MemChain Inference Boundary"

# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] Embeddings, extraction, and other
# model inference belong to attested Phala ACI. This wizard never downloads
# local models; encrypted storage and deterministic retrieval remain available.
echo "  Local model downloads and inference are disabled on ordinary Rust nodes."
echo "  MemChain can still store encrypted records and run deterministic indexes."
echo "  Clients must use their own fail-closed Phala ACI route for model work."
echo ""

# ════════════════════════════════════════════════════════════════
# Step 3: Configuration
# ════════════════════════════════════════════════════════════════

header "Step 3/5 — Configuration"

# ── Network ────────────────────────────────────────────────────
echo -e "${BOLD}Network Settings${NC}"
echo ""

API_PORT=$(prompt "  API port" "8421")
if ! [[ "${API_PORT}" =~ ^[0-9]{1,5}$ ]] || [ "${API_PORT}" -lt 1 ] || [ "${API_PORT}" -gt 65535 ]; then
    error "API port must be an integer between 1 and 65535."
    exit 1
fi
API_ADDR="127.0.0.1:${API_PORT}"

VPN_ENABLED=false
if prompt_yn "  Enable VPN tunnel?" "n"; then
    VPN_ENABLED=true
fi

# ── Miner ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Miner Settings${NC}"
echo ""
# [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex] Do not imply local extraction runs.
echo "  Background maintenance interval; model-powered miner tasks stay disabled here."
echo "  Interval = how often it runs (seconds)."
echo ""

MINER_INTERVAL=$(prompt "  Miner interval (seconds)" "60")
if ! [[ "${MINER_INTERVAL}" =~ ^[0-9]+$ ]] || [ "${MINER_INTERVAL}" -lt 1 ]; then
    error "Miner interval must be a positive integer."
    exit 1
fi

# ── Security ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Security${NC}"
echo ""

API_SECRET=""
if prompt_yn "  Require API key for access?" "n"; then
    API_SECRET="$(prompt_secret "  API secret (leave blank to generate one):")"
    if [ ${#API_SECRET} -lt 16 ]; then
        [ -z "${API_SECRET}" ] || warn "Secret too short; generating a random one"
        API_SECRET=$(head -c 32 /dev/urandom | xxd -p | head -c 32)
    fi
    ok "API secret configured (value hidden)"
fi

# ── Model inference boundary ──────────────────────────────────
echo ""
# [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex] This wizard creates a local-mode
# MemChain service. Rust rejects server-side SuperNode outside SaaS mode, and
# ordinary nodes must not accept plaintext model prompts. The client owns its
# direct Phala route and its separate consent/attestation policy.
echo -e "${BOLD}Phala-only model inference${NC}"
echo "  This local MemChain profile stores records and deterministic indexes only."
echo "  Model requests must go directly from the client to its approved Phala ACI route."
echo "  This server will not proxy prompts or configure SuperNode in local mode."
echo ""

# ════════════════════════════════════════════════════════════════
# Step 4: Generate Configuration
# ════════════════════════════════════════════════════════════════

header "Step 4/5 — Generate Configuration"

sudo mkdir -p "${CONFIG_DIR}" 2>/dev/null || mkdir -p "${CONFIG_DIR}"
sudo mkdir -p "${DB_DIR}" 2>/dev/null || mkdir -p "${DB_DIR}"

if [ -f "${CONFIG_FILE}" ]; then
    BACKUP="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "${CONFIG_FILE}" "${BACKUP}"
    warn "Existing config backed up to: ${BACKUP}"
fi

cat > "${CONFIG_FILE}" << TOML
# ════════════════════════════════════════════════════════════════
# AeroNyx Server Configuration
# Generated by init.sh on $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# ════════════════════════════════════════════════════════════════

[network]
listen_addr = "0.0.0.0:51820"

[vpn]
virtual_ip_range = "100.64.0.0/24"
gateway_ip = "100.64.0.1"

[tun]
device_name = "aeronyx0"
mtu = 1420

[limits]
max_connections = 1000
session_timeout = 86400

[logging]
level = "info"

# ════════════════════════════════════════════════════════════════
# MemChain — AI Cognitive Engine
# ════════════════════════════════════════════════════════════════

[memchain]
mode = "local"
api_listen_addr = "${API_ADDR}"
db_path = "${DB_DIR}/memchain.db"
aof_path = "${DB_DIR}/.memchain"
miner_interval_secs = ${MINER_INTERVAL}
TOML

if [ -n "${API_SECRET}" ]; then
    cat >> "${CONFIG_FILE}" << TOML
api_secret = $(toml_quote "${API_SECRET}")
TOML
fi

cat >> "${CONFIG_FILE}" << TOML

# [MEMCHAIN-PHALA-SETUP 2026-10-06 by Codex] Legacy model controls are kept
# parse-compatible but explicitly disabled; no model bundle is installed.
embed_enabled = false
ner_enabled = false
reranker_enabled = false

# Deterministic graph/index support does not execute local models.
graph_enabled = false
entropy_filter_enabled = true

# Model-powered tasks require a separately configured explicit consent policy.
miner_entity_extraction = false
miner_community_detection = false
miner_session_summary = false
TOML

ok "Configuration written to: ${CONFIG_FILE}"
chmod 600 "${CONFIG_FILE}"
echo -e "${DIM}Configuration contents (including credentials) were not printed.${NC}"

# ════════════════════════════════════════════════════════════════
# Step 4.5: Network Setup (IP Forwarding + NAT)
# ════════════════════════════════════════════════════════════════

header "Step 4.5/5 — Network Setup (IP Forwarding + NAT)"

# VPN 子网和 TUN 设备（与 server.toml 保持一致）
VPN_SUBNET="100.64.0.0/24"
TUN_DEVICE="aeronyx0"

# 自动检测默认网卡
DEFAULT_IFACE=$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')
if [ -z "${DEFAULT_IFACE}" ]; then
    DEFAULT_IFACE="eth0"
    warn "Could not detect default interface, using fallback: ${DEFAULT_IFACE}"
fi
info "Detected default network interface: ${DEFAULT_IFACE}"

# ── 1. IP Forwarding ──────────────────────────────────────────
info "Enabling IP forwarding..."

echo 1 > /proc/sys/net/ipv4/ip_forward

if grep -q "net.ipv4.ip_forward" /etc/sysctl.conf 2>/dev/null; then
    sed -i 's/^#*net.ipv4.ip_forward.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf
else
    echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
fi
sysctl -p /etc/sysctl.conf >/dev/null 2>&1 || true
ok "IP forwarding enabled"

# ── 2. 安装 iptables（如果没有）─────────────────────────────
if ! command -v iptables &>/dev/null; then
    info "Installing iptables..."
    if command -v apt-get &>/dev/null; then
        apt-get install -y iptables iptables-persistent 2>&1 | tail -3
    elif command -v yum &>/dev/null; then
        yum install -y iptables iptables-services 2>&1 | tail -3
    else
        warn "Cannot install iptables automatically. Please install manually."
    fi
fi

# ── 3. 应用 iptables NAT 规则 ─────────────────────────────────
if command -v iptables &>/dev/null; then
    info "Applying iptables NAT rules (interface: ${DEFAULT_IFACE})..."

    # 避免重复添加（-C 检查已存在则跳过，否则 -A 添加）
    iptables -t nat -C POSTROUTING -s "${VPN_SUBNET}" -o "${DEFAULT_IFACE}" -j MASQUERADE 2>/dev/null \
        || iptables -t nat -A POSTROUTING -s "${VPN_SUBNET}" -o "${DEFAULT_IFACE}" -j MASQUERADE

    iptables -C FORWARD -i "${TUN_DEVICE}" -j ACCEPT 2>/dev/null \
        || iptables -A FORWARD -i "${TUN_DEVICE}" -j ACCEPT

    iptables -C FORWARD -o "${TUN_DEVICE}" -j ACCEPT 2>/dev/null \
        || iptables -A FORWARD -o "${TUN_DEVICE}" -j ACCEPT

    ok "iptables NAT rules applied"

    # ── 4. 持久化 iptables ────────────────────────────────────
    if command -v netfilter-persistent &>/dev/null; then
        netfilter-persistent save >/dev/null 2>&1
        ok "iptables rules persisted (netfilter-persistent)"
    elif command -v iptables-save &>/dev/null; then
        mkdir -p /etc/iptables
        iptables-save > /etc/iptables/rules.v4
        ok "iptables rules saved to /etc/iptables/rules.v4"

        # 开机自动恢复
        RC_LOCAL="/etc/rc.local"
        RESTORE_CMD="iptables-restore < /etc/iptables/rules.v4"
        if [ -f "${RC_LOCAL}" ]; then
            if ! grep -q "iptables-restore" "${RC_LOCAL}"; then
                sed -i "s|^exit 0|${RESTORE_CMD}\nexit 0|" "${RC_LOCAL}"
                ok "Added iptables restore to ${RC_LOCAL}"
            fi
        else
            cat > "${RC_LOCAL}" << EOF
#!/bin/sh
${RESTORE_CMD}
exit 0
EOF
            chmod +x "${RC_LOCAL}"
            ok "Created ${RC_LOCAL} with iptables restore"
        fi
    fi
else
    warn "iptables not available. VPN clients will connect but have no internet."
    warn "Run manually after install:"
    warn "  iptables -t nat -A POSTROUTING -s ${VPN_SUBNET} -o ${DEFAULT_IFACE} -j MASQUERADE"
    warn "  iptables -A FORWARD -i ${TUN_DEVICE} -j ACCEPT"
    warn "  iptables -A FORWARD -o ${TUN_DEVICE} -j ACCEPT"
fi

# ── 5. 开放 UDP 51820（ufw / firewalld）──────────────────────
if command -v ufw &>/dev/null && ufw status 2>/dev/null | grep -q "active"; then
    ufw allow 51820/udp >/dev/null 2>&1
    ok "ufw: opened UDP 51820"
elif command -v firewall-cmd &>/dev/null; then
    firewall-cmd --permanent --add-port=51820/udp >/dev/null 2>&1
    firewall-cmd --reload >/dev/null 2>&1
    ok "firewalld: opened UDP 51820"
fi

echo ""
# ── 6. Configure in-tunnel DNS stub ─────────────────────────────
# iOS/macOS clients use the VPN gateway as DNS server to avoid local-network
# DNS leakage. The node must therefore listen on 100.64.0.1:53.
if [ -x "${SCRIPT_DIR}/setup_vpn_dns.sh" ]; then
    info "Configuring AeroNyx VPN DNS stub..."
    if "${SCRIPT_DIR}/setup_vpn_dns.sh" --gateway "100.64.0.1"; then
        ok "VPN DNS stub configured"
    else
        warn "VPN DNS stub setup failed. Clients may connect but DNS may not resolve."
    fi
else
    warn "Missing ${SCRIPT_DIR}/setup_vpn_dns.sh. Clients may connect but DNS may not resolve."
fi

echo ""
ok "Network setup complete"
info "Note: On GCP/AWS/Azure, also open UDP 51820 in the cloud security group."

# ════════════════════════════════════════════════════════════════
# Step 5: Build & Start
# ════════════════════════════════════════════════════════════════

header "Step 5/5 — Build & Start"

BUILD_NOW=false
if prompt_yn "Build the server now? (takes ~2 minutes)" "y"; then
    BUILD_NOW=true
fi

if [ "${BUILD_NOW}" = true ]; then
    info "Building aeronyx-server (release mode)..."
    echo ""
    cd "${PROJECT_ROOT}"

    if cargo build --release -p aeronyx-server 2>&1 | tail -5; then
        ok "Build successful!"
    else
        error "Build failed. Check errors above."
        exit 1
    fi

    echo ""
    if prompt_yn "Start the server now?" "y"; then
        info "Starting AeroNyx server..."
        echo ""

        BINARY="${PROJECT_ROOT}/target/release/aeronyx-server"
        if [ ! -f "${BINARY}" ]; then
            error "Binary not found: ${BINARY}"
            exit 1
        fi

        "${BINARY}" --config "${CONFIG_FILE}" &
        SERVER_PID=$!
        sleep 5

        if curl -s "http://${API_ADDR}/api/mpi/status" >/dev/null 2>&1; then
            ok "Server is running! (PID: ${SERVER_PID})"
            echo ""

            if [ -x "${SCRIPT_DIR}/setup_vpn_dns.sh" ]; then
                info "Activating AeroNyx VPN DNS stub after server start..."
                if ! "${SCRIPT_DIR}/setup_vpn_dns.sh" --gateway "100.64.0.1"; then
                    warn "VPN DNS stub activation failed; run scripts/setup_vpn_dns.sh after the TUN interface is up."
                fi
                echo ""
            fi

            STATUS=$(curl -s "http://${API_ADDR}/api/mpi/status")
            GRAPH=$(echo "${STATUS}" | jq -r '.graph_enabled // false')

            echo -e "  Server-side model inference: ${DIM}Disabled (client-to-Phala ACI only)${NC}"
            echo -e "  Knowledge Graph:   $([ "${GRAPH}" = "true" ] && echo -e "${GREEN}✅ Enabled${NC}" || echo -e "${YELLOW}⚠️  Disabled${NC}")"
        else
            warn "Server may still be starting. Check logs."
        fi
    fi
fi

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}${GREEN}"
echo "    ╔══════════════════════════════════════════╗"
echo "    ║                                          ║"
echo "    ║     ✅ Setup Complete!                   ║"
echo "    ║                                          ║"
echo "    ╚══════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "  ${BOLD}Config:${NC}  ${CONFIG_FILE}"
echo -e "  ${BOLD}Data:${NC}    ${DB_DIR}/memchain.db"
echo -e "  ${BOLD}API:${NC}     http://${API_ADDR}/api/mpi/"
echo ""

echo -e "  ${BOLD}Quick Test:${NC}"
echo ""
echo "    # Store a conversation"
echo "    curl -X POST http://${API_ADDR}/api/mpi/log \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"session_id\":\"hello\",\"turns\":[{\"role\":\"user\",\"content\":\"Hello world\"}],\"source_ai\":\"test\"}'"
echo ""
echo "    # Search"
echo "    curl http://${API_ADDR}/api/mpi/search?q=hello"
echo ""
echo "    # System status"
echo "    curl http://${API_ADDR}/api/mpi/status | jq ."
echo ""

if [ -n "${API_SECRET}" ]; then
    echo -e "  ${BOLD}${YELLOW}API Key Required:${NC}"
    echo "    Read the key from ${CONFIG_FILE}; it is intentionally not displayed here."
    echo ""
fi

echo -e "  ${BOLD}Manage:${NC}"
echo "    Start:   ${PROJECT_ROOT}/target/release/aeronyx-server --config ${CONFIG_FILE}"
echo "    Stop:    pkill -f aeronyx-server"
echo "    Logs:    (stdout — use systemd or screen for background)"
echo "    Config:  ${CONFIG_FILE}"
echo ""
echo -e "  ${DIM}Documentation: https://github.com/AeroNyx/AeroNyx${NC}"
echo ""
