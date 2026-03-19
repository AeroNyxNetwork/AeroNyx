#!/usr/bin/env bash
# ============================================
# scripts/init.sh — AeroNyx MemChain 
# ============================================
# Interactive setup wizard that:
#   1. Detects system resources (CPU, RAM, disk)
#   2. Downloads AI models (EmbeddingGemma + GLiNER + ORT)
#   3. Asks user preferences (LLM provider, privacy level)
#   4. Generates server.toml configuration
#   5. Optionally builds and starts the server
#
# Usage:
#   chmod +x scripts/init.sh
#   ./scripts/init.sh              # Interactive mode
#   ./scripts/init.sh --defaults   # Non-interactive, all defaults
#   ./scripts/init.sh --help       # Show help
#
# Last Modified:
# v2.5.0 - 🌟 Created. Full interactive setup wizard.

set -euo pipefail

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
ask()     { echo -en "${BOLD}$*${NC} "; }

# ── Resolve paths ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="/etc/aeronyx"
CONFIG_FILE="${CONFIG_DIR}/server.toml"
MODELS_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models"
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
# Usage: result=$(prompt "Question?" "default_value")
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

# Yes/No prompt, returns 0 for yes, 1 for no
prompt_yn() {
    local question="$1"
    local default="$2"  # "y" or "n"

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
echo "    ║     AI Cognitive Engine v2.5.0            ║"
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

# Warn on low resources
if [ "${RAM_MB}" -lt 3072 ]; then
    warn "Low RAM (${RAM_MB}MB). Minimum 4GB recommended."
    warn "EmbeddingGemma may fail. Consider using MiniLM (--embed-minilm)."
fi
if [ "${DISK_FREE_GB}" -lt 3 ]; then
    warn "Low disk space (${DISK_FREE_GB}GB). Models need ~2GB."
fi

# ════════════════════════════════════════════════════════════════
# Step 2: Model Download
# ════════════════════════════════════════════════════════════════

header "Step 2/5 — AI Model Download"

echo "  MemChain uses 3 local AI models (no API needed):"
echo ""
echo -e "  ${GREEN}1. EmbeddingGemma-300M${NC} — Semantic vectors (100+ languages)"
echo -e "     Size: ~1.5GB | Quality: Best | ${DIM}Required${NC}"
echo ""
echo -e "  ${GREEN}2. GLiNER small-v2.1${NC} — Entity recognition (zero-shot NER)"
echo -e "     Size: ~200MB | ${DIM}Required for knowledge graph${NC}"
echo ""
echo -e "  ${GREEN}3. ONNX Runtime${NC} — Inference engine"
echo -e "     Size: ~30MB | ${DIM}Required${NC}"
echo ""

DOWNLOAD_MODELS=true
if [ -f "${MODELS_DIR}/embeddinggemma/model.onnx" ] && [ -f "${MODELS_DIR}/gliner/model.onnx" ]; then
    ok "Models already downloaded!"
    if prompt_yn "Re-download models?" "n"; then
        DOWNLOAD_MODELS=true
    else
        DOWNLOAD_MODELS=false
    fi
fi

if [ "${DOWNLOAD_MODELS}" = true ]; then
    info "Downloading models... (this may take a few minutes)"
    echo ""
    bash "${SCRIPT_DIR}/download_models.sh"
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Step 3: Configuration
# ════════════════════════════════════════════════════════════════

header "Step 3/5 — Configuration"

# ── Network ────────────────────────────────────────────────────
echo -e "${BOLD}Network Settings${NC}"
echo ""

API_PORT=$(prompt "  API port" "8421")
API_ADDR="127.0.0.1:${API_PORT}"

VPN_ENABLED=false
if prompt_yn "  Enable VPN tunnel?" "n"; then
    VPN_ENABLED=true
fi

# ── Miner ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Miner Settings${NC}"
echo ""
echo "  The Miner automatically builds the knowledge graph from conversations."
echo "  Interval = how often it runs (seconds)."
echo ""

MINER_INTERVAL=$(prompt "  Miner interval (seconds)" "60")

# ── Security ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Security${NC}"
echo ""

API_SECRET=""
if prompt_yn "  Require API key for access?" "n"; then
    API_SECRET=$(prompt "  API secret (min 16 chars)" "$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | xxd -p)")
    if [ ${#API_SECRET} -lt 16 ]; then
        warn "Secret too short, generating random one"
        API_SECRET=$(head -c 32 /dev/urandom | xxd -p | head -c 32)
    fi
    ok "API secret set: ${API_SECRET:0:8}..."
fi

# ── SuperNode (LLM Enhancement) ───────────────────────────────
echo ""
echo -e "${BOLD}SuperNode — LLM Enhancement (Optional)${NC}"
echo ""
echo "  MemChain works fully without LLM. SuperNode adds:"
echo "  • Better session titles (LLM-generated)"
echo "  • Community narratives (natural language summaries)"
echo "  • Entity descriptions (enriched from context)"
echo ""

SUPERNODE_ENABLED=false
SUPERNODE_PROVIDER=""
SUPERNODE_API_KEY=""
SUPERNODE_MODEL=""
SUPERNODE_API_BASE=""
SUPERNODE_PROVIDER_NAME=""

if prompt_yn "  Enable SuperNode LLM enhancement?" "n"; then
    SUPERNODE_ENABLED=true
    echo ""
    echo "  Choose a provider:"
    echo ""
    echo -e "  ${BOLD}1${NC}) DeepSeek     — Cheapest (\$0.07/M tokens), good quality"
    echo -e "  ${BOLD}2${NC}) OpenAI       — GPT-4o-mini, reliable"
    echo -e "  ${BOLD}3${NC}) Anthropic    — Claude Sonnet, best reasoning"
    echo -e "  ${BOLD}4${NC}) xAI Grok     — Grok-3-mini"
    echo -e "  ${BOLD}5${NC}) Groq         — Fast inference, Llama models"
    echo -e "  ${BOLD}6${NC}) Ollama       — Local, free, needs 16GB+ RAM"
    echo -e "  ${BOLD}7${NC}) Custom       — Any OpenAI-compatible endpoint"
    echo ""

    PROVIDER_CHOICE=$(prompt "  Provider [1-7]" "1")

    case "${PROVIDER_CHOICE}" in
        1)
            SUPERNODE_PROVIDER_NAME="deepseek"
            SUPERNODE_API_BASE="https://api.deepseek.com"
            SUPERNODE_MODEL="deepseek-chat"
            ask "  DeepSeek API key: "; read -r SUPERNODE_API_KEY
            ;;
        2)
            SUPERNODE_PROVIDER_NAME="openai"
            SUPERNODE_API_BASE="https://api.openai.com"
            SUPERNODE_MODEL="gpt-4o-mini"
            ask "  OpenAI API key: "; read -r SUPERNODE_API_KEY
            ;;
        3)
            SUPERNODE_PROVIDER_NAME="anthropic"
            SUPERNODE_PROVIDER="anthropic"
            SUPERNODE_API_BASE="https://api.anthropic.com"
            SUPERNODE_MODEL="claude-sonnet-4-20250514"
            ask "  Anthropic API key: "; read -r SUPERNODE_API_KEY
            ;;
        4)
            SUPERNODE_PROVIDER_NAME="grok"
            SUPERNODE_API_BASE="https://api.x.ai"
            SUPERNODE_MODEL="grok-3-mini"
            ask "  xAI API key: "; read -r SUPERNODE_API_KEY
            ;;
        5)
            SUPERNODE_PROVIDER_NAME="groq"
            SUPERNODE_API_BASE="https://api.groq.com/openai"
            SUPERNODE_MODEL="llama-3.3-70b-versatile"
            ask "  Groq API key: "; read -r SUPERNODE_API_KEY
            ;;
        6)
            SUPERNODE_PROVIDER_NAME="ollama"
            SUPERNODE_API_BASE="http://localhost:11434"
            SUPERNODE_MODEL=$(prompt "  Ollama model name" "qwen2.5:7b")
            SUPERNODE_API_KEY=""
            if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                warn "Ollama not running on localhost:11434"
                warn "Install: curl -fsSL https://ollama.com/install.sh | sh"
                warn "Then: ollama pull ${SUPERNODE_MODEL}"
            fi
            ;;
        7)
            SUPERNODE_PROVIDER_NAME=$(prompt "  Provider name" "custom")
            SUPERNODE_API_BASE=$(prompt "  API base URL" "http://localhost:8080")
            SUPERNODE_MODEL=$(prompt "  Model name" "default")
            ask "  API key (empty for none): "; read -r SUPERNODE_API_KEY
            ;;
        *)
            warn "Invalid choice, disabling SuperNode"
            SUPERNODE_ENABLED=false
            ;;
    esac

    # Default provider type
    if [ -z "${SUPERNODE_PROVIDER}" ]; then
        SUPERNODE_PROVIDER="openai_compatible"
    fi

    if [ "${SUPERNODE_ENABLED}" = true ] && [ -n "${SUPERNODE_API_KEY}" ]; then
        # Quick connectivity test
        echo ""
        info "Testing API connectivity..."
        TEST_URL="${SUPERNODE_API_BASE}/v1/chat/completions"
        if [ "${SUPERNODE_PROVIDER}" = "anthropic" ]; then
            TEST_URL="${SUPERNODE_API_BASE}/v1/messages"
        fi

        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${SUPERNODE_API_KEY}" \
            -H "Content-Type: application/json" \
            --max-time 10 \
            "${TEST_URL}" 2>/dev/null || echo "000")

        if [ "${HTTP_CODE}" = "000" ]; then
            warn "Cannot reach ${SUPERNODE_API_BASE} — check URL and network"
        elif [ "${HTTP_CODE}" = "401" ] || [ "${HTTP_CODE}" = "403" ]; then
            warn "API returned ${HTTP_CODE} — check your API key"
        else
            ok "API reachable (HTTP ${HTTP_CODE})"
        fi
    fi

    if [ "${SUPERNODE_ENABLED}" = true ]; then
        echo ""
        echo -e "  ${BOLD}Privacy level for LLM:${NC}"
        echo "  1) structured — Only entity names sent to LLM (safest, default)"
        echo "  2) full       — Full conversation content sent (better quality)"
        echo ""
        PRIVACY_CHOICE=$(prompt "  Privacy level [1-2]" "1")
        case "${PRIVACY_CHOICE}" in
            2) PRIVACY_LEVEL="full" ;;
            *) PRIVACY_LEVEL="structured" ;;
        esac
    fi
fi

# ════════════════════════════════════════════════════════════════
# Step 4: Generate Configuration
# ════════════════════════════════════════════════════════════════

header "Step 4/5 — Generate Configuration"

# Create directories
sudo mkdir -p "${CONFIG_DIR}" 2>/dev/null || mkdir -p "${CONFIG_DIR}"
sudo mkdir -p "${DB_DIR}" 2>/dev/null || mkdir -p "${DB_DIR}"

# Backup existing config
if [ -f "${CONFIG_FILE}" ]; then
    BACKUP="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "${CONFIG_FILE}" "${BACKUP}"
    warn "Existing config backed up to: ${BACKUP}"
fi

# Generate config
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
session_timeout = 300

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

# API secret
if [ -n "${API_SECRET}" ]; then
    cat >> "${CONFIG_FILE}" << TOML
api_secret = "${API_SECRET}"
TOML
fi

# Embedding model
cat >> "${CONFIG_FILE}" << TOML

# Local AI Models
embed_model_path = "${MODELS_DIR}/embeddinggemma"
embed_max_tokens = 256
embed_output_dim = 384

# NER + Knowledge Graph
ner_enabled = true
ner_model_path = "${MODELS_DIR}/gliner"
graph_enabled = true
entropy_filter_enabled = true

# Miner Cognitive Steps
miner_entity_extraction = true
miner_community_detection = true
miner_session_summary = true
TOML

# SuperNode config
if [ "${SUPERNODE_ENABLED}" = true ]; then
    cat >> "${CONFIG_FILE}" << TOML

# ════════════════════════════════════════════════════════════════
# SuperNode — LLM Cognitive Enhancement
# ════════════════════════════════════════════════════════════════

[memchain.supernode]
enabled = true

[[memchain.supernode.providers]]
name = "${SUPERNODE_PROVIDER_NAME}"
type = "${SUPERNODE_PROVIDER}"
api_base = "${SUPERNODE_API_BASE}"
TOML

    if [ -n "${SUPERNODE_API_KEY}" ]; then
        cat >> "${CONFIG_FILE}" << TOML
api_key = "${SUPERNODE_API_KEY}"
TOML
    fi

    cat >> "${CONFIG_FILE}" << TOML
model = "${SUPERNODE_MODEL}"
max_tokens = 1500
temperature = 0.3

[memchain.supernode.routing]
fallback = "${SUPERNODE_PROVIDER_NAME}"

[memchain.supernode.privacy]
default_level = "${PRIVACY_LEVEL:-structured}"
allow_full_for = ["session_title", "code_analysis"]

[memchain.supernode.worker]
poll_interval_secs = 5
max_concurrent = 3
max_retries = 3
task_timeout_secs = 120
TOML
fi

ok "Configuration written to: ${CONFIG_FILE}"
echo ""
echo -e "${DIM}─── Generated Config ───${NC}"
cat "${CONFIG_FILE}"
echo -e "${DIM}────────────────────────${NC}"

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

        # Quick health check
        if curl -s "http://${API_ADDR}/api/mpi/status" >/dev/null 2>&1; then
            ok "Server is running! (PID: ${SERVER_PID})"
            echo ""

            STATUS=$(curl -s "http://${API_ADDR}/api/mpi/status")
            EMBED_READY=$(echo "${STATUS}" | jq -r '.embed_ready // false')
            NER_READY=$(echo "${STATUS}" | jq -r '.ner_ready // false')
            GRAPH=$(echo "${STATUS}" | jq -r '.graph_enabled // false')
            SN=$(echo "${STATUS}" | jq -r '.supernode.enabled // false')

            echo -e "  Embedding Engine:  $([ "${EMBED_READY}" = "true" ] && echo -e "${GREEN}✅ Ready${NC}" || echo -e "${RED}❌ Not ready${NC}")"
            echo -e "  NER Engine:        $([ "${NER_READY}" = "true" ] && echo -e "${GREEN}✅ Ready${NC}" || echo -e "${RED}❌ Not ready${NC}")"
            echo -e "  Knowledge Graph:   $([ "${GRAPH}" = "true" ] && echo -e "${GREEN}✅ Enabled${NC}" || echo -e "${YELLOW}⚠️  Disabled${NC}")"
            echo -e "  SuperNode LLM:     $([ "${SN}" = "true" ] && echo -e "${GREEN}✅ Enabled${NC}" || echo -e "${DIM}Disabled${NC}")"
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
echo -e "  ${BOLD}Models:${NC}  ${MODELS_DIR}/"
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
    echo "    Add header: -H 'Authorization: Bearer ${API_SECRET}'"
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
