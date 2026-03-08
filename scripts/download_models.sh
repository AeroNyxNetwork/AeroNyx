#!/usr/bin/env bash
# ============================================
# scripts/download_models.sh
# ============================================
# One-click download of embedding model files for MemChain local inference.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# What it does:
#   Downloads MiniLM-L6-v2 ONNX model (~22MB) and tokenizer (~700KB) from
#   HuggingFace into crates/aeronyx-server/models/minilm-l6-v2/
#
# After running:
#   - `cargo build` will compile normally (no include_bytes dependency)
#   - MemChain server will auto-detect model files at startup
#   - /api/mpi/embed will return 200 with local embeddings
#   - /api/mpi/status will report embed_ready: true
#
# If you skip this script:
#   - Server starts normally (embed is optional)
#   - /api/mpi/embed returns 503
#   - Miner falls back to OpenClaw Gateway for embeddings
#
# Model source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# License: Apache-2.0
# ============================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
REPO="sentence-transformers/all-MiniLM-L6-v2"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

# Resolve project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/minilm-l6-v2"

# Files to download
MODEL_FILE="onnx/model.onnx"
TOKENIZER_FILE="tokenizer.json"

# ── Colors ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Dependency check ───────────────────────────────────────────
download_cmd=""
if command -v curl &>/dev/null; then
    download_cmd="curl"
elif command -v wget &>/dev/null; then
    download_cmd="wget"
else
    error "Neither curl nor wget found. Please install one of them."
    exit 1
fi

info "Using ${download_cmd} for downloads"

# ── Download function ──────────────────────────────────────────
download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "${dest}" ]; then
        local size
        size=$(wc -c < "${dest}" | tr -d ' ')
        if [ "${size}" -gt 1000 ]; then
            ok "${desc} already exists (${size} bytes), skipping"
            return 0
        else
            warn "${desc} exists but looks incomplete (${size} bytes), re-downloading"
        fi
    fi

    info "Downloading ${desc}..."
    info "  URL: ${url}"
    info "  Dest: ${dest}"

    if [ "${download_cmd}" = "curl" ]; then
        curl -L --progress-bar -o "${dest}" "${url}"
    else
        wget --show-progress -O "${dest}" "${url}"
    fi

    if [ -f "${dest}" ]; then
        local size
        size=$(wc -c < "${dest}" | tr -d ' ')
        ok "${desc} downloaded (${size} bytes)"
    else
        error "Failed to download ${desc}"
        return 1
    fi
}

# ── Main ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  AeroNyx MemChain — Embedding Model Downloader"
echo "  Model: all-MiniLM-L6-v2 (384-dim, Apache-2.0)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create model directory
mkdir -p "${MODEL_DIR}"
info "Model directory: ${MODEL_DIR}"
echo ""

# Download model.onnx (~22MB)
download_file \
    "${BASE_URL}/${MODEL_FILE}" \
    "${MODEL_DIR}/model.onnx" \
    "model.onnx (~22MB)"

echo ""

# Download tokenizer.json (~700KB)
download_file \
    "${BASE_URL}/${TOKENIZER_FILE}" \
    "${MODEL_DIR}/tokenizer.json" \
    "tokenizer.json (~700KB)"

# ── Verify ─────────────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────────────────"

errors=0

if [ -f "${MODEL_DIR}/model.onnx" ]; then
    model_size=$(wc -c < "${MODEL_DIR}/model.onnx" | tr -d ' ')
    if [ "${model_size}" -gt 1000000 ]; then
        ok "model.onnx: ${model_size} bytes ✓"
    else
        error "model.onnx looks too small (${model_size} bytes)"
        errors=$((errors + 1))
    fi
else
    error "model.onnx not found"
    errors=$((errors + 1))
fi

if [ -f "${MODEL_DIR}/tokenizer.json" ]; then
    tok_size=$(wc -c < "${MODEL_DIR}/tokenizer.json" | tr -d ' ')
    if [ "${tok_size}" -gt 10000 ]; then
        ok "tokenizer.json: ${tok_size} bytes ✓"
    else
        error "tokenizer.json looks too small (${tok_size} bytes)"
        errors=$((errors + 1))
    fi
else
    error "tokenizer.json not found"
    errors=$((errors + 1))
fi

echo "───────────────────────────────────────────────────────────"
echo ""

if [ "${errors}" -eq 0 ]; then
    ok "All model files ready!"
    echo ""
    info "Next steps:"
    info "  1. cargo build -p aeronyx-server"
    info "  2. Server will auto-detect model at: ${MODEL_DIR}"
    info "  3. POST /api/mpi/embed will return local embeddings (384-dim)"
    echo ""
    info "To verify: curl -X POST http://127.0.0.1:8421/api/mpi/embed \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"texts\":[\"hello world\"]}'"
else
    error "${errors} file(s) failed. Please retry or download manually from:"
    error "  https://huggingface.co/${REPO}"
    exit 1
fi
