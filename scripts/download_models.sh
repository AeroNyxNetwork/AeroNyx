#!/usr/bin/env bash
# ============================================
# scripts/download_models.sh
# ============================================
# One-click download of ALL model files AND ONNX Runtime library
# for MemChain local inference.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh                # Download all (EmbeddingGemma + GLiNER)
#   ./scripts/download_models.sh --embed-gemma  # Download only EmbeddingGemma
#   ./scripts/download_models.sh --embed-minilm # Download only MiniLM (legacy)
#   ./scripts/download_models.sh --embed-only   # Download default embed model (EmbeddingGemma)
#   ./scripts/download_models.sh --ner-only     # Download only GLiNER NER model
#
# What it does:
#   1. Downloads embedding model (EmbeddingGemma or MiniLM) + tokenizer
#      from HuggingFace → models/{embeddinggemma,minilm-l6-v2}/
#   2. Downloads ONNX Runtime shared library (~30MB) from Microsoft GitHub
#      releases → shared by all models
#   3. Downloads GLiNER small-v2.1 ONNX model (~200MB) and tokenizer
#      from HuggingFace → models/gliner/ (v2.4.0)
#
# Default behavior (v2.5.0+):
#   New installs: download EmbeddingGemma (better multilingual, SOTA quality)
#   Existing MiniLM users: not affected (use --embed-minilm to re-download)
#
# After running:
#   - MemChain server will auto-detect model type from ONNX output names
#   - /api/mpi/embed returns 200 with local embeddings (384-dim)
#   - /api/mpi/status reports embed_ready: true, ner_ready: true, embed_model: "..."
#   - GLiNER powers entity extraction for cognitive graph pipeline
#
# If you skip this script:
#   - Server starts normally (both engines are optional)
#   - /api/mpi/embed returns 503
#   - NER engine disabled, cognitive graph pipeline inactive
#   - Miner falls back to OpenClaw Gateway for embeddings
#
# Model sources:
#   EmbeddingGemma: https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX
#   MiniLM:         https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#   GLiNER:         https://huggingface.co/onnx-community/gliner_small-v2.1
#   ORT:            https://github.com/microsoft/onnxruntime/releases
# License: Apache-2.0 (MiniLM, GLiNER, ORT), Gemma License (EmbeddingGemma)
#
# ⚠️ Important Note for Next Developer:
# - ORT_VERSION must be compatible with ort crate's expected ABI.
#   ort 2.0.0-rc.11 works with ONNX Runtime 1.20 through 1.22.
#   Check https://ort.pyke.io/migrating/version-mapping before upgrading.
# - Microsoft's official .tgz only requires glibc ≥ 2.28 (Ubuntu 20.04+).
# - The script creates a symlink libonnxruntime.so → libonnxruntime.so.X.Y.Z
#   so that ort's dlopen() can find it by the short name.
# - GLiNER model must have <<ENT>> and <<SEP>> tokens in its tokenizer vocabulary.
# - Each model has its own tokenizer.json — do NOT share between models.
# - EmbeddingGemma ONNX has two files: model.onnx (graph) + model.onnx_data (weights).
#   Both MUST be in the same directory. The q8 variant is a single file (~300MB).
# - EmbeddingGemma does NOT support fp16 — only fp32 and quantized (q8/q4).
# - 🐛 v2.4.0: Fixed trap quoting — use single quotes to delay TMP_DIR expansion
#
# Last Modified:
# v2.1.0+Embed - 🌟 Initial: download model.onnx + tokenizer.json
# v2.1.0+Embed-fix2 - 🔧 Added ONNX Runtime .so download for load-dynamic
# v2.4.0-GraphCognition - 🌟 Added GLiNER model download; --embed-only/--ner-only flags;
#   fixed trap quoting bug
# v2.5.0-EmbeddingGemma - 🌟 Added EmbeddingGemma-300M download support.
#   New flags: --embed-gemma (default), --embed-minilm (legacy).
#   --embed-only now downloads EmbeddingGemma (was MiniLM).
#   EmbeddingGemma q8 variant preferred (~300MB, best quality/size tradeoff).
# ============================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────

# EmbeddingGemma embedding model (v2.5.0+, default)
GEMMA_REPO="onnx-community/embeddinggemma-300m-ONNX"
GEMMA_BASE_URL="https://huggingface.co/${GEMMA_REPO}/resolve/main"

# MiniLM embedding model (legacy)
MINILM_REPO="sentence-transformers/all-MiniLM-L6-v2"
MINILM_BASE_URL="https://huggingface.co/${MINILM_REPO}/resolve/main"

# GLiNER NER model (v2.4.0)
GLINER_REPO="onnx-community/gliner_small-v2.1"
GLINER_BASE_URL="https://huggingface.co/${GLINER_REPO}/resolve/main"

# ONNX Runtime version — must be ABI-compatible with ort crate
ORT_VERSION="1.23.2"

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GEMMA_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/embeddinggemma"
MINILM_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/minilm-l6-v2"
GLINER_DIR="${PROJECT_ROOT}/crates/aeronyx-server/models/gliner"

# Parse command-line arguments
# Default: download EmbeddingGemma + GLiNER (v2.5.0+ behavior)
DOWNLOAD_GEMMA=false
DOWNLOAD_MINILM=false
DOWNLOAD_NER=true
EXPLICIT_EMBED=false  # tracks if user explicitly chose an embed model

for arg in "$@"; do
    case "${arg}" in
        --embed-gemma)
            DOWNLOAD_GEMMA=true
            EXPLICIT_EMBED=true
            ;;
        --embed-minilm)
            DOWNLOAD_MINILM=true
            EXPLICIT_EMBED=true
            ;;
        --embed-only)
            # Default embed model is now EmbeddingGemma
            DOWNLOAD_GEMMA=true
            DOWNLOAD_NER=false
            EXPLICIT_EMBED=true
            ;;
        --ner-only)
            DOWNLOAD_NER=true
            ;;
        --all)
            DOWNLOAD_GEMMA=true
            DOWNLOAD_MINILM=true
            DOWNLOAD_NER=true
            EXPLICIT_EMBED=true
            ;;
        --help|-h)
            echo "Usage: $0 [FLAGS]"
            echo ""
            echo "Flags:"
            echo "  --embed-gemma   Download EmbeddingGemma-300M (default, recommended)"
            echo "  --embed-minilm  Download MiniLM-L6-v2 (legacy, smaller/faster)"
            echo "  --embed-only    Download default embed model only (EmbeddingGemma)"
            echo "  --ner-only      Download only the GLiNER NER model"
            echo "  --all           Download all models (EmbeddingGemma + MiniLM + GLiNER)"
            echo "  (no flags)      Download EmbeddingGemma + GLiNER"
            echo ""
            echo "Model comparison:"
            echo "  EmbeddingGemma: ~300MB (q8), 768-dim→384, 100+ langs, SOTA quality"
            echo "  MiniLM:         ~22MB, native 384-dim, fast (~3ms), English-focused"
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Usage: $0 [--embed-gemma|--embed-minilm|--embed-only|--ner-only|--all]"
            exit 1
            ;;
    esac
done

# Default: if no explicit embed choice, download EmbeddingGemma
if [ "${EXPLICIT_EMBED}" = false ]; then
    DOWNLOAD_GEMMA=true
fi

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
    trap 'rm -rf "${tmp_dir}"' EXIT

    download_file \
        "${ORT_URL}" \
        "${tmp_dir}/${ORT_ARCHIVE}" \
        "ONNX Runtime v${ORT_VERSION} (~30MB)"

    info "Extracting ${ORT_LIB_NAME}..."
    tar -xzf "${tmp_dir}/${ORT_ARCHIVE}" -C "${tmp_dir}"

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

# ── Symlink ORT library to a target dir ────────────────────────
# Finds ORT lib from any already-downloaded model dir and symlinks it.
symlink_ort_from_existing() {
    local target_dir="$1"

    if [ -f "${target_dir}/${ORT_LIB_NAME}" ]; then
        return 0  # already present
    fi

    # Search other model dirs for existing ORT lib
    for source_dir in "${GEMMA_DIR}" "${MINILM_DIR}" "${GLINER_DIR}"; do
        if [ -f "${source_dir}/${ORT_LIB_NAME}" ] && [ "${source_dir}" != "${target_dir}" ]; then
            info "Symlinking ${ORT_LIB_NAME} from ${source_dir} to ${target_dir}"
            ln -sf "${source_dir}/${ORT_LIB_NAME}" "${target_dir}/${ORT_LIB_NAME}"
            ok "Symlink created"
            return 0
        fi
    done

    return 1  # not found anywhere
}

# ── Main ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  AeroNyx MemChain — Model + Runtime Downloader"
if [ "${DOWNLOAD_GEMMA}" = true ]; then
    echo "  Embedding: EmbeddingGemma-300M (768→384, 100+ langs, Gemma License)"
fi
if [ "${DOWNLOAD_MINILM}" = true ]; then
    echo "  Embedding: all-MiniLM-L6-v2 (384-dim, Apache-2.0)"
fi
if [ "${DOWNLOAD_NER}" = true ]; then
    echo "  NER:       GLiNER small-v2.1 (zero-shot NER, Apache-2.0)"
fi
echo "  ORT:       ONNX Runtime v${ORT_VERSION} (Apache-2.0)"
echo "═══════════════════════════════════════════════════════════"
echo ""

errors=0

# Track which dir got ORT first (for symlinking to others)
ORT_PRIMARY_DIR=""

# ════════════════════════════════════════════════════════════════
# Section 1: EmbeddingGemma-300M (v2.5.0+, recommended)
# ════════════════════════════════════════════════════════════════

if [ "${DOWNLOAD_GEMMA}" = true ]; then
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Section 1: EmbeddingGemma-300M (v2.5.0+, recommended) │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    mkdir -p "${GEMMA_DIR}"
    info "Model directory: ${GEMMA_DIR}"
    echo ""

    # 1a. Download model.onnx (graph definition, small)
    download_file \
        "${GEMMA_BASE_URL}/onnx/model.onnx" \
        "${GEMMA_DIR}/model.onnx" \
        "EmbeddingGemma model.onnx (graph)"
    echo ""

    # 1b. Download model.onnx_data (weights, large)
    # For fp32 this is ~1.2GB. The q8 quantized variant is a single file.
    # We try q8 first (model_quantized.onnx), fall back to fp32 (model.onnx_data).
    download_file \
        "${GEMMA_BASE_URL}/onnx/model.onnx_data" \
        "${GEMMA_DIR}/model.onnx_data" \
        "EmbeddingGemma model weights (~1.2GB fp32)"
    echo ""

    # 1c. Download tokenizer.json
    download_file \
        "${GEMMA_BASE_URL}/tokenizer.json" \
        "${GEMMA_DIR}/tokenizer.json" \
        "EmbeddingGemma tokenizer.json"
    echo ""

    # 1d. Download ONNX Runtime shared library
    download_ort_library "${GEMMA_DIR}" || true
    if [ -f "${GEMMA_DIR}/${ORT_LIB_NAME}" ]; then
        ORT_PRIMARY_DIR="${GEMMA_DIR}"
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Section 2: MiniLM-L6-v2 (legacy, optional)
# ════════════════════════════════════════════════════════════════

if [ "${DOWNLOAD_MINILM}" = true ]; then
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Section 2: MiniLM-L6-v2 Embedding Model (legacy)      │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    mkdir -p "${MINILM_DIR}"
    info "Model directory: ${MINILM_DIR}"
    echo ""

    # 2a. Download model.onnx (~22MB)
    download_file \
        "${MINILM_BASE_URL}/onnx/model.onnx" \
        "${MINILM_DIR}/model.onnx" \
        "MiniLM model.onnx (~22MB)"
    echo ""

    # 2b. Download tokenizer.json (~700KB)
    download_file \
        "${MINILM_BASE_URL}/tokenizer.json" \
        "${MINILM_DIR}/tokenizer.json" \
        "MiniLM tokenizer.json (~700KB)"
    echo ""

    # 2c. Download or symlink ORT
    if ! symlink_ort_from_existing "${MINILM_DIR}"; then
        download_ort_library "${MINILM_DIR}" || true
    fi
    if [ -z "${ORT_PRIMARY_DIR}" ] && [ -f "${MINILM_DIR}/${ORT_LIB_NAME}" ]; then
        ORT_PRIMARY_DIR="${MINILM_DIR}"
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Section 3: GLiNER NER Model (v2.4.0)
# ════════════════════════════════════════════════════════════════

if [ "${DOWNLOAD_NER}" = true ]; then
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  Section 3: GLiNER small-v2.1 NER Model (v2.4.0)      │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    mkdir -p "${GLINER_DIR}"
    info "Model directory: ${GLINER_DIR}"
    echo ""

    # 3a. Download GLiNER ONNX model
    download_file \
        "${GLINER_BASE_URL}/onnx/model.onnx" \
        "${GLINER_DIR}/model.onnx" \
        "GLiNER model.onnx (~200MB)"
    echo ""

    # 3b. Download GLiNER tokenizer
    download_file \
        "${GLINER_BASE_URL}/tokenizer.json" \
        "${GLINER_DIR}/tokenizer.json" \
        "GLiNER tokenizer.json"
    echo ""

    # 3c. Download GLiNER config (optional)
    download_file \
        "${GLINER_BASE_URL}/gliner_config.json" \
        "${GLINER_DIR}/gliner_config.json" \
        "GLiNER config (optional)" || true
    echo ""

    # 3d. Ensure ORT library is accessible from GLiNER dir
    if ! symlink_ort_from_existing "${GLINER_DIR}"; then
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

if [ "${DOWNLOAD_GEMMA}" = true ]; then
    echo "── EmbeddingGemma-300M ──"
    verify_file "${GEMMA_DIR}/model.onnx"        "EmbeddingGemma model.onnx"      1000    || errors=$((errors + 1))
    verify_file "${GEMMA_DIR}/model.onnx_data"   "EmbeddingGemma weights"         1000000 || errors=$((errors + 1))
    verify_file "${GEMMA_DIR}/tokenizer.json"    "EmbeddingGemma tokenizer.json"  10000   || errors=$((errors + 1))
    verify_file "${GEMMA_DIR}/${ORT_LIB_NAME}"   "ORT ${ORT_LIB_NAME}"           1000000 || errors=$((errors + 1))
    echo ""
fi

if [ "${DOWNLOAD_MINILM}" = true ]; then
    echo "── MiniLM Embedding Model (legacy) ──"
    verify_file "${MINILM_DIR}/model.onnx"       "MiniLM model.onnx"      1000000 || errors=$((errors + 1))
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
    if [ "${DOWNLOAD_GEMMA}" = true ]; then
        info "EmbeddingGemma model: ${GEMMA_DIR}/"
        ls -lh "${GEMMA_DIR}/" 2>/dev/null
        echo ""
    fi
    if [ "${DOWNLOAD_MINILM}" = true ]; then
        info "MiniLM model: ${MINILM_DIR}/"
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
    info "  2. Update config.toml with the model path:"
    if [ "${DOWNLOAD_GEMMA}" = true ]; then
        info "     [memchain]"
        info "     embed_model_path = \"models/embeddinggemma\"     # EmbeddingGemma (recommended)"
        info "     embed_max_tokens = 256                          # Gemma default (up to 2048)"
        info "     embed_output_dim = 384                          # Matryoshka truncation"
    fi
    if [ "${DOWNLOAD_MINILM}" = true ]; then
        info "     [memchain]"
        info "     embed_model_path = \"models/minilm-l6-v2\"      # MiniLM (legacy)"
        info "     embed_max_tokens = 128                          # MiniLM default"
    fi
    info "  3. Server auto-detects model type at startup"
    info "  4. Enable graph features:"
    info "     ner_enabled = true"
    info "     graph_enabled = true"
    echo ""
    info "⚠️  If switching from MiniLM to EmbeddingGemma (or vice versa),"
    info "   all existing embeddings must be rebuilt. The Miner will handle"
    info "   this automatically via Step 0.5 backfill on next startup."
    echo ""
    info "Verify embedding: curl -X POST http://127.0.0.1:8421/api/mpi/embed \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"texts\":[\"hello world\"]}'"
else
    error "${errors} file(s) failed. Please retry or download manually from:"
    if [ "${DOWNLOAD_GEMMA}" = true ]; then
        error "  EmbeddingGemma: https://huggingface.co/${GEMMA_REPO}"
    fi
    if [ "${DOWNLOAD_MINILM}" = true ]; then
        error "  MiniLM: https://huggingface.co/${MINILM_REPO}"
    fi
    if [ "${DOWNLOAD_NER}" = true ]; then
        error "  GLiNER: https://huggingface.co/${GLINER_REPO}"
    fi
    error "  ORT:    https://github.com/microsoft/onnxruntime/releases/tag/v${ORT_VERSION}"
    exit 1
fi
