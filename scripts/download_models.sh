#!/usr/bin/env bash
# ============================================
# scripts/download_models.sh
# ============================================
# One-click download of embedding model files AND ONNX Runtime library
# for MemChain local inference.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# What it does:
#   1. Downloads MiniLM-L6-v2 ONNX model (~22MB) and tokenizer (~700KB)
#      from HuggingFace
#   2. Downloads ONNX Runtime shared library (~30MB) from Microsoft GitHub
#      releases — compatible with Ubuntu 20.04+ (glibc ≥ 2.28)
#   3. Places everything into crates/aeronyx-server/models/minilm-l6-v2/
#
# After running:
#   - `cargo build` will compile normally (load-dynamic, no static linking)
#   - MemChain server will auto-detect libonnxruntime.so and model files
#   - /api/mpi/embed will return 200 with local embeddings (384-dim)
#   - /api/mpi/status will report embed_ready: true
#
# If you skip this script:
#   - Server starts normally (embed is optional)
#   - /api/mpi/embed returns 503
#   - Miner falls back to OpenClaw Gateway for embeddings
#
# Model source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# ORT source: https://github.com/microsoft/onnxruntime/releases
# License: Apache-2.0 (both)
#
# ⚠️ Important Note for Next Developer:
# - ORT_VERSION must be compatible with ort crate's expected ABI.
#   ort 2.0.0-rc.11 works with ONNX Runtime 1.20 through 1.22.
#   Check https://ort.pyke.io/migrating/version-mapping before upgrading.
# - Microsoft's official .tgz only requires glibc ≥ 2.28 (Ubuntu 20.04+).
#   This is why we use load-dynamic instead of pyke's download-binaries
#   (which requires glibc ≥ 2.38 due to their build environment).
# - The script creates a symlink libonnxruntime.so → libonnxruntime.so.X.Y.Z
#   so that ort's dlopen() can find it by the short name.
#
# Last Modified:
# v2.1.0+Embed - 🌟 Initial: download model.onnx + tokenizer.json
# v2.1.0+Embed-fix2 - 🔧 Added ONNX Runtime .so download for load-dynamic
# ============================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
REPO="sentence-transformers/all-MiniLM-L6-v2"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

# ONNX Runtime version — must be ABI-compatible with ort crate
# ort 2.0.0-rc.11 ↔ ORT 1.20-1.22 (C API is stable across minors)
# ort 2.0.0-rc.11 requires ORT >= 1.23.x
ORT_VERSION="1.23.2"

# Resolve project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/minilm-l6-v2"

# Files to download
MODEL_FILE="onnx/model.onnx"
TOKENIZER_FILE="tokenizer.json"

# ORT platform detection
detect_ort_archive() {
    local arch
    arch="$(uname -m)"
    local os
    os="$(uname -s)"

    case "${os}" in
        Linux)
            case "${arch}" in
                x86_64)  echo "onnxruntime-linux-x64-${ORT_VERSION}.tgz" ;;
                aarch64) echo "onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" ;;
                *)       echo "" ;;
            esac
            ;;
        Darwin)
            case "${arch}" in
                x86_64)  echo "onnxruntime-osx-x86_64-${ORT_VERSION}.tgz" ;;
                arm64)   echo "onnxruntime-osx-arm64-${ORT_VERSION}.tgz" ;;
                *)       echo "" ;;
            esac
            ;;
        *)
            echo ""
            ;;
    esac
}

ORT_ARCHIVE="$(detect_ort_archive)"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"

# Platform-specific library name
case "$(uname -s)" in
    Linux)  ORT_LIB_NAME="libonnxruntime.so" ;;
    Darwin) ORT_LIB_NAME="libonnxruntime.dylib" ;;
    *)      ORT_LIB_NAME="onnxruntime.dll" ;;
esac

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
echo "  AeroNyx MemChain — Embedding Model + Runtime Downloader"
echo "  Model: all-MiniLM-L6-v2 (384-dim, Apache-2.0)"
echo "  ORT:   ONNX Runtime v${ORT_VERSION} (Apache-2.0)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create model directory
mkdir -p "${MODEL_DIR}"
info "Model directory: ${MODEL_DIR}"
echo ""

# ── 1. Download model.onnx (~22MB) ────────────────────────────
download_file \
    "${BASE_URL}/${MODEL_FILE}" \
    "${MODEL_DIR}/model.onnx" \
    "model.onnx (~22MB)"

echo ""

# ── 2. Download tokenizer.json (~700KB) ───────────────────────
download_file \
    "${BASE_URL}/${TOKENIZER_FILE}" \
    "${MODEL_DIR}/tokenizer.json" \
    "tokenizer.json (~700KB)"

echo ""

# ── 3. Download ONNX Runtime shared library (~30MB) ───────────
if [ -z "${ORT_ARCHIVE}" ]; then
    warn "Unsupported platform $(uname -s)/$(uname -m) for ORT auto-download"
    warn "Please manually download from: https://github.com/microsoft/onnxruntime/releases"
    warn "Place ${ORT_LIB_NAME} in ${MODEL_DIR}/"
else
    # Check if already present
    if [ -f "${MODEL_DIR}/${ORT_LIB_NAME}" ]; then
        ort_size=$(wc -c < "${MODEL_DIR}/${ORT_LIB_NAME}" | tr -d ' ')
        if [ "${ort_size}" -gt 1000000 ]; then
            ok "${ORT_LIB_NAME} already exists (${ort_size} bytes), skipping"
        else
            warn "${ORT_LIB_NAME} looks too small, re-downloading"
            rm -f "${MODEL_DIR}/${ORT_LIB_NAME}"
        fi
    fi

    if [ ! -f "${MODEL_DIR}/${ORT_LIB_NAME}" ]; then
        info "Downloading ONNX Runtime v${ORT_VERSION}..."
        TMP_DIR=$(mktemp -d)
        trap "rm -rf ${TMP_DIR}" EXIT

        download_file \
            "${ORT_URL}" \
            "${TMP_DIR}/${ORT_ARCHIVE}" \
            "ONNX Runtime v${ORT_VERSION} (~30MB)"

        info "Extracting ${ORT_LIB_NAME}..."
        tar -xzf "${TMP_DIR}/${ORT_ARCHIVE}" -C "${TMP_DIR}"

        # Find the library in the extracted archive
        ORT_EXTRACTED_DIR="${TMP_DIR}/$(basename "${ORT_ARCHIVE}" .tgz)"
        ORT_LIB_FOUND=""

        # Look for the versioned .so file (e.g. libonnxruntime.so.1.22.0)
        for candidate in \
            "${ORT_EXTRACTED_DIR}/lib/${ORT_LIB_NAME}.${ORT_VERSION}" \
            "${ORT_EXTRACTED_DIR}/lib/${ORT_LIB_NAME}" \
            ; do
            if [ -f "${candidate}" ]; then
                ORT_LIB_FOUND="${candidate}"
                break
            fi
        done

        if [ -z "${ORT_LIB_FOUND}" ]; then
            # Fallback: find any matching library
            ORT_LIB_FOUND="$(find "${TMP_DIR}" -name "${ORT_LIB_NAME}*" -type f | head -1)"
        fi

        if [ -n "${ORT_LIB_FOUND}" ]; then
            cp "${ORT_LIB_FOUND}" "${MODEL_DIR}/${ORT_LIB_NAME}"
            ort_size=$(wc -c < "${MODEL_DIR}/${ORT_LIB_NAME}" | tr -d ' ')
            ok "${ORT_LIB_NAME} installed (${ort_size} bytes)"
        else
            error "Could not find ${ORT_LIB_NAME} in extracted archive"
            error "Contents of ${ORT_EXTRACTED_DIR}/lib/:"
            ls -la "${ORT_EXTRACTED_DIR}/lib/" 2>/dev/null || echo "  (directory not found)"
        fi

        # Cleanup handled by trap
    fi
fi

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

if [ -f "${MODEL_DIR}/${ORT_LIB_NAME}" ]; then
    ort_size=$(wc -c < "${MODEL_DIR}/${ORT_LIB_NAME}" | tr -d ' ')
    if [ "${ort_size}" -gt 1000000 ]; then
        ok "${ORT_LIB_NAME}: ${ort_size} bytes ✓"
    else
        error "${ORT_LIB_NAME} looks too small (${ort_size} bytes)"
        errors=$((errors + 1))
    fi
else
    error "${ORT_LIB_NAME} not found"
    errors=$((errors + 1))
fi

echo "───────────────────────────────────────────────────────────"
echo ""

if [ "${errors}" -eq 0 ]; then
    ok "All files ready!"
    echo ""
    info "Files in ${MODEL_DIR}:"
    ls -lh "${MODEL_DIR}/"
    echo ""
    info "Next steps:"
    info "  1. cargo build -p aeronyx-server"
    info "  2. Server will auto-detect all files at: ${MODEL_DIR}"
    info "  3. POST /api/mpi/embed will return local embeddings (384-dim)"
    echo ""
    info "To verify: curl -X POST http://127.0.0.1:8421/api/mpi/embed \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"texts\":[\"hello world\"]}'"
else
    error "${errors} file(s) failed. Please retry or download manually from:"
    error "  Model: https://huggingface.co/${REPO}"
    error "  ORT:   https://github.com/microsoft/onnxruntime/releases/tag/v${ORT_VERSION}"
    exit 1
fi
