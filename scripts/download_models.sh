#!/usr/bin/env bash
# ============================================
# scripts/download_models.sh
# ============================================
# One-click download of ALL model files AND ONNX Runtime library
# for MemChain local inference.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh              # Download all models
#   ./scripts/download_models.sh --embed-only # Download only embedding model
#   ./scripts/download_models.sh --ner-only   # Download only GLiNER NER model
#
# What it does:
#   1. Downloads MiniLM-L6-v2 ONNX model (~22MB) and tokenizer (~700KB)
#      from HuggingFace → models/minilm-l6-v2/
#   2. Downloads ONNX Runtime shared library (~30MB) from Microsoft GitHub
#      releases → models/minilm-l6-v2/ (shared by both models)
#   3. Downloads GLiNER small-v2.1 ONNX model (~200MB) and tokenizer
#      from HuggingFace → models/gliner/ (v2.4.0)
#
# After running:
#   - MemChain server will auto-detect all model files
#   - /api/mpi/embed returns 200 with local embeddings (384-dim)
#   - /api/mpi/status reports embed_ready: true, ner_ready: true
#   - GLiNER powers entity extraction for cognitive graph pipeline
#
# If you skip this script:
#   - Server starts normally (both engines are optional)
#   - /api/mpi/embed returns 503
#   - NER engine disabled, cognitive graph pipeline inactive
#   - Miner falls back to OpenClaw Gateway for embeddings
#
# Model sources:
#   MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#   GLiNER: https://huggingface.co/onnx-community/gliner_small-v2.1
#   ORT:    https://github.com/microsoft/onnxruntime/releases
# License: Apache-2.0 (all)
#
# ⚠️ Important Note for Next Developer:
# - ORT_VERSION must be compatible with ort crate's expected ABI.
#   ort 2.0.0-rc.11 works with ONNX Runtime 1.20 through 1.22.
#   Check https://ort.pyke.io/migrating/version-mapping before upgrading.
# - Microsoft's official .tgz only requires glibc ≥ 2.28 (Ubuntu 20.04+).
# - The script creates a symlink libonnxruntime.so → libonnxruntime.so.X.Y.Z
#   so that ort's dlopen() can find it by the short name.
# - GLiNER model must have <<ENT>> and <<SEP>> tokens in its tokenizer vocabulary.
#   The onnx-community exports include these. Custom exports may not.
# - GLiNER tokenizer is typically DeBERTa-v3 based (different from MiniLM's WordPiece).
#   Each model has its own tokenizer.json — do NOT share between models.
# - 🐛 v2.4.0: Fixed trap quoting — use single quotes to delay TMP_DIR expansion
#   (prevents empty rm -rf if mktemp fails before trap is evaluated)
#
# Last Modified:
# v2.1.0+Embed - 🌟 Initial: download model.onnx + tokenizer.json
# v2.1.0+Embed-fix2 - 🔧 Added ONNX Runtime .so download for load-dynamic
# v2.4.0-GraphCognition - 🌟 Added GLiNER model download; --embed-only/--ner-only flags;
#   fixed trap quoting bug
# ============================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────

# MiniLM embedding model
MINILM_REPO="sentence-transformers/all-MiniLM-L6-v2"
MINILM_BASE_URL="https://huggingface.co/${MINILM_REPO}/resolve/main"

# GLiNER NER model (v2.4.0)
# Using onnx-community export which includes <<ENT>> and <<SEP>> tokens
# and is pre-converted to ONNX format (no Python conversion needed).
GLINER_REPO="onnx-community/gliner_small-v2.1"
GLINER_BASE_URL="https://huggingface.co/${GLINER_REPO}/resolve/main"

# ONNX Runtime version — must be ABI-compatible with ort crate
# ort 2.0.0-rc.11 requires ORT >= 1.23.x
ORT_VERSION="1.23.2"

# Resolve project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MINILM_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/minilm-l6-v2"
GLINER_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/gliner"

# Parse command-line arguments
DOWNLOAD_EMBED=true
DOWNLOAD_NER=true

for arg in "$@"; do
    case "${arg}" in
        --embed-only) DOWNLOAD_NER=false ;;
        --ner-only)   DOWNLOAD_EMBED=false ;;
        --help|-h)
            echo "Usage: $0 [--embed-only|--ner-only]"
            echo ""
            echo "  --embed-only   Download only the MiniLM embedding model"
            echo "  --ner-only     Download only the GLiNER NER model"
            echo "  (no flags)     Download all models"
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Usage: $0 [--embed-only|--ner-only]"
            exit 1
            ;;
    esac
done

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

# ── Download ORT shared library ────────────────────────────────
# Shared by both MiniLM and GLiNER (placed in MiniLM dir, GLiNER
# also searches there via ORT_DYLIB_PATH or init_ort_runtime fallback)
download_ort_library() {
    local target_dir="$1"

    if [ -z "${ORT_ARCHIVE}" ]; then
        warn "Unsupported platform $(uname -s)/$(uname -m) for ORT auto-download"
        warn "Please manually download from: https://github.com/microsoft/onnxruntime/releases"
        warn "Place ${ORT_LIB_NAME} in ${target_dir}/"
        return 1
    fi

    # Check if already present
    if [ -f "${target_dir}/${ORT_LIB_NAME}" ]; then
        local ort_size
        ort_size=$(wc -c < "${target_dir}/${ORT_LIB_NAME}" | tr -d ' ')
        if [ "${ort_size}" -gt 1000000 ]; then
            ok "${ORT_LIB_NAME} already exists (${ort_size} bytes), skipping"
            return 0
        else
            warn "${ORT_LIB_NAME} looks too small, re-downloading"
            rm -f "${target_dir}/${ORT_LIB_NAME}"
        fi
    fi

    info "Downloading ONNX Runtime v${ORT_VERSION}..."
    local tmp_dir
    tmp_dir=$(mktemp -d)
    # 🐛 v2.4.0: Use single quotes to delay variable expansion in trap.
    # If mktemp fails, $tmp_dir would be empty and `rm -rf ""` is dangerous.
    # With single quotes, the trap evaluates tmp_dir at execution time (after mktemp succeeds).
    trap 'rm -rf "${tmp_dir}"' EXIT

    download_file \
        "${ORT_URL}" \
        "${tmp_dir}/${ORT_ARCHIVE}" \
        "ONNX Runtime v${ORT_VERSION} (~30MB)"

    info "Extracting ${ORT_LIB_NAME}..."
    tar -xzf "${tmp_dir}/${ORT_ARCHIVE}" -C "${tmp_dir}"

    # Find the library in the extracted archive
    local ort_extracted_dir="${tmp_dir}/$(basename "${ORT_ARCHIVE}" .tgz)"
    local ort_lib_found=""

    for candidate in \
        "${ort_extracted_dir}/lib/${ORT_LIB_NAME}.${ORT_VERSION}" \
        "${ort_extracted_dir}/lib/${ORT_LIB_NAME}" \
        ; do
        if [ -f "${candidate}" ]; then
            ort_lib_found="${candidate}"
            break
        fi
    done

    if [ -z "${ort_lib_found}" ]; then
        ort_lib_found="$(find "${tmp_dir}" -name "${ORT_LIB_NAME}*" -type f | head -1)"
    fi

    if [ -n "${ort_lib_found}" ]; then
        cp "${ort_lib_found}" "${target_dir}/${ORT_LIB_NAME}"
        local final_size
        final_size=$(wc -c < "${target_dir}/${ORT_LIB_NAME}" | tr -d ' ')
        ok "${ORT_LIB_NAME} installed (${final_size} bytes)"
        return 0
    else
        error "Could not find ${ORT_LIB_NAME} in extracted archive"
        error "Contents of ${ort_extracted_dir}/lib/:"
        ls -la "${ort_extracted_dir}/lib/" 2>/dev/null || echo "  (directory not found)"
        return 1
    fi
}

# ── Main ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  AeroNyx MemChain — Model + Runtime Downloader"
echo "  Embedding: all-MiniLM-L6-v2 (384-dim, Apache-2.0)"
echo "  NER:       GLiNER small-v2.1 (zero-shot NER, Apache-2.0)"
echo "  ORT:       ONNX Runtime v${ORT_VERSION} (Apache-2.0)"
echo "═══════════════════════════════════════════════════════════"
echo ""

errors=0

# ════════════════════════════════════════════════════════════════
# Section 1: MiniLM Embedding Model + ORT Runtime
# ════════════════════════════════════════════════════════════════

if [ "${DOWNLOAD_EMBED}" = true ]; then
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Section 1: MiniLM-L6-v2 Embedding Model + ORT Runtime │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    mkdir -p "${MINILM_DIR}"
    info "Model directory: ${MINILM_DIR}"
    echo ""

    # 1a. Download model.onnx (~22MB)
    download_file \
        "${MINILM_BASE_URL}/onnx/model.onnx" \
        "${MINILM_DIR}/model.onnx" \
        "MiniLM model.onnx (~22MB)"
    echo ""

    # 1b. Download tokenizer.json (~700KB)
    download_file \
        "${MINILM_BASE_URL}/tokenizer.json" \
        "${MINILM_DIR}/tokenizer.json" \
        "MiniLM tokenizer.json (~700KB)"
    echo ""

    # 1c. Download ONNX Runtime shared library (~30MB)
    download_ort_library "${MINILM_DIR}" || true
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Section 2: GLiNER NER Model (v2.4.0)
# ════════════════════════════════════════════════════════════════

if [ "${DOWNLOAD_NER}" = true ]; then
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Section 2: GLiNER small-v2.1 NER Model (v2.4.0)      │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    mkdir -p "${GLINER_DIR}"
    info "Model directory: ${GLINER_DIR}"
    echo ""

    # 2a. Download GLiNER ONNX model
    # onnx-community exports provide model.onnx directly
    download_file \
        "${GLINER_BASE_URL}/onnx/model.onnx" \
        "${GLINER_DIR}/model.onnx" \
        "GLiNER model.onnx (~200MB)"
    echo ""

    # 2b. Download GLiNER tokenizer
    # GLiNER uses DeBERTa-v3 tokenizer (different from MiniLM's WordPiece)
    download_file \
        "${GLINER_BASE_URL}/tokenizer.json" \
        "${GLINER_DIR}/tokenizer.json" \
        "GLiNER tokenizer.json"
    echo ""

    # 2c. Download GLiNER config (optional, used for max_width and other settings)
    download_file \
        "${GLINER_BASE_URL}/gliner_config.json" \
        "${GLINER_DIR}/gliner_config.json" \
        "GLiNER config (optional)" || true
    echo ""

    # 2d. Ensure ORT library is also accessible from GLiNER dir
    # GLiNER uses the same ORT runtime. If the ORT lib is in minilm dir,
    # create a symlink so ner.rs can find it via its model_dir search path.
    if [ "${DOWNLOAD_EMBED}" = true ] && [ -f "${MINILM_DIR}/${ORT_LIB_NAME}" ]; then
        if [ ! -f "${GLINER_DIR}/${ORT_LIB_NAME}" ]; then
            info "Symlinking ${ORT_LIB_NAME} from MiniLM dir to GLiNER dir"
            ln -sf "${MINILM_DIR}/${ORT_LIB_NAME}" "${GLINER_DIR}/${ORT_LIB_NAME}"
            ok "Symlink created"
        fi
    elif [ ! -f "${GLINER_DIR}/${ORT_LIB_NAME}" ]; then
        # NER-only mode: download ORT directly to GLiNER dir
        download_ort_library "${GLINER_DIR}" || true
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Verification
# ════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Verification"
echo "═══════════════════════════════════════════════════════════"
echo ""

verify_file() {
    local path="$1"
    local desc="$2"
    local min_size="$3"

    if [ -f "${path}" ]; then
        local size
        size=$(wc -c < "${path}" | tr -d ' ')
        if [ "${size}" -gt "${min_size}" ]; then
            ok "${desc}: ${size} bytes ✓"
            return 0
        else
            error "${desc} looks too small (${size} bytes, expected > ${min_size})"
            return 1
        fi
    else
        error "${desc} not found: ${path}"
        return 1
    fi
}

if [ "${DOWNLOAD_EMBED}" = true ]; then
    echo "── MiniLM Embedding Model ──"
    verify_file "${MINILM_DIR}/model.onnx"      "MiniLM model.onnx"      1000000 || errors=$((errors + 1))
    verify_file "${MINILM_DIR}/tokenizer.json"   "MiniLM tokenizer.json"  10000   || errors=$((errors + 1))
    verify_file "${MINILM_DIR}/${ORT_LIB_NAME}"  "ORT ${ORT_LIB_NAME}"   1000000 || errors=$((errors + 1))
    echo ""
fi

if [ "${DOWNLOAD_NER}" = true ]; then
    echo "── GLiNER NER Model (v2.4.0) ──"
    verify_file "${GLINER_DIR}/model.onnx"       "GLiNER model.onnx"      1000000 || errors=$((errors + 1))
    verify_file "${GLINER_DIR}/tokenizer.json"   "GLiNER tokenizer.json"  10000   || errors=$((errors + 1))
    echo ""
fi

echo "───────────────────────────────────────────────────────────"
echo ""

if [ "${errors}" -eq 0 ]; then
    ok "All files ready!"
    echo ""
    if [ "${DOWNLOAD_EMBED}" = true ]; then
        info "Embedding model: ${MINILM_DIR}/"
        ls -lh "${MINILM_DIR}/" 2>/dev/null
        echo ""
    fi
    if [ "${DOWNLOAD_NER}" = true ]; then
        info "NER model: ${GLINER_DIR}/"
        ls -lh "${GLINER_DIR}/" 2>/dev/null
        echo ""
    fi
    info "Next steps:"
    info "  1. cargo build -p aeronyx-server"
    info "  2. Server auto-detects all model files at startup"
    info "  3. Enable in config.toml:"
    info "     [memchain]"
    info "     ner_enabled = true          # Enable GLiNER NER"
    info "     graph_enabled = true        # Enable knowledge graph"
    info "     entropy_filter_enabled = true"
    info "     miner_entity_extraction = true"
    echo ""
    info "Verify embedding: curl -X POST http://127.0.0.1:8421/api/mpi/embed \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"texts\":[\"hello world\"]}'"
else
    error "${errors} file(s) failed. Please retry or download manually from:"
    if [ "${DOWNLOAD_EMBED}" = true ]; then
        error "  MiniLM: https://huggingface.co/${MINILM_REPO}"
    fi
    if [ "${DOWNLOAD_NER}" = true ]; then
        error "  GLiNER: https://huggingface.co/${GLINER_REPO}"
    fi
    error "  ORT:    https://github.com/microsoft/onnxruntime/releases/tag/v${ORT_VERSION}"
    exit 1
fi
