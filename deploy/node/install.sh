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
# - Initial production deployment entrypoint for commercial VPN node operators.
#
# Main Functionality:
# - Detects Linux/systemd environment.
# - Installs host dependencies on supported Linux distributions.
# - Clones or uses the AeroNyx repository.
# - Creates /etc/aeronyx and /var/lib/aeronyx state directories.
# - Installs a safe default server.toml without overwriting existing config.
# - Builds aeronyx-server release binary.
# - Installs and enables systemd service.
# - Optionally configures IP forwarding/NAT and registers/starts the node.
#
# Dependencies:
# - deploy/node/server.example.toml
# - deploy/node/aeronyx-server.service
# - crates/aeronyx-server/src/main.rs CLI commands:
#   register, start, validate, status
#
# Main Logical Flow:
# 1. Parse flags and validate host assumptions.
# 2. Prepare repository, directories, config, and network forwarding.
# 3. Build release binary, install systemd unit, optionally register/start.
#
# Important Note for Next Developer:
# - Never overwrite /etc/aeronyx/server.toml, server_key.json, or node_info.json
#   without an explicit future migration flow.
# - Keep all operations idempotent so operators can safely rerun the installer.
# - This script is Linux/systemd only; macOS, iOS, Android, and Windows are
#   development/client platforms, not production node hosts for this script.
#
# Last Modified:
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
CONFIG_DIR="/etc/aeronyx"
CONFIG_FILE="${CONFIG_DIR}/server.toml"
ENV_FILE="${CONFIG_DIR}/aeronyx.env"
STATE_DIR="/var/lib/aeronyx"
SERVICE_NAME="aeronyx-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

REPO_URL="${DEFAULT_REPO_URL}"
BRANCH="${DEFAULT_BRANCH}"
REPO_DIR="${DEFAULT_REPO_DIR}"
REGISTRATION_CODE=""
DO_BUILD=1
DO_NETWORK=1
DO_START=0
DO_ENABLE=1
INSTALL_RUST=1
INSTALL_PACKAGES=1
DRY_RUN=0
CONFIG_ONLY=0

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage:
  sudo ./deploy/node/install.sh [OPTIONS]

Options:
  --repo-url URL          Git repository URL. Default: https://github.com/AeroNyxNetwork/AeroNyx.git
  --branch NAME           Git branch or ref. Default: main
  --repo-dir PATH         Install repository path. Default: /opt/aeronyx/AeroNyx
  --registration-code C   Register node after build.
  --start                 Start service after install. Automatically enabled when --registration-code is used.
  --no-build              Skip cargo release build.
  --no-network            Skip sysctl and NAT setup.
  --no-enable             Do not enable systemd service.
  --skip-package-install  Do not install OS packages automatically.
  --skip-rust-install     Do not install Rust automatically if cargo is missing.
  --config-only           Only create config/env directories and server.toml if missing.
  --dry-run               Print actions without changing the host.
  -h, --help              Show this help.

Examples:
  sudo ./deploy/node/install.sh --registration-code NYX-1234-ABCDE --start
  sudo ./deploy/node/install.sh --repo-dir /root/open/AeroNyx --no-build --no-network
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-url) REPO_URL="${2:?missing value}"; shift 2 ;;
        --branch) BRANCH="${2:?missing value}"; shift 2 ;;
        --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
        --registration-code) REGISTRATION_CODE="${2:?missing value}"; DO_START=1; shift 2 ;;
        --start) DO_START=1; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        --no-network) DO_NETWORK=0; shift ;;
        --no-enable) DO_ENABLE=0; shift ;;
        --skip-package-install) INSTALL_PACKAGES=0; shift ;;
        --skip-rust-install) INSTALL_RUST=0; shift ;;
        --config-only) CONFIG_ONLY=1; DO_BUILD=0; DO_NETWORK=0; DO_START=0; DO_ENABLE=0; INSTALL_PACKAGES=0; INSTALL_RUST=0; shift ;;
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

prepare_repo() {
    if [ -d "${REPO_DIR}/.git" ]; then
        log "Using existing repository: ${REPO_DIR}"
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
    vpn_subnet="100.64.0.0/24"
    tun_device="aeronyx0"
    default_iface="$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')"
    [ -n "${default_iface}" ] || default_iface="eth0"

    log "Enabling IPv4 forwarding"
    run_shell "printf '1\n' > /proc/sys/net/ipv4/ip_forward"
    if grep -q '^#\?net.ipv4.ip_forward' /etc/sysctl.conf 2>/dev/null; then
        run sed -i 's/^#\?net.ipv4.ip_forward.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf
    else
        run_shell "printf '\nnet.ipv4.ip_forward=1\n' >> /etc/sysctl.conf"
    fi

    if command -v iptables >/dev/null 2>&1; then
        log "Applying idempotent iptables NAT rules on ${default_iface}"
        run_shell "iptables -t nat -C POSTROUTING -s '${vpn_subnet}' -o '${default_iface}' -j MASQUERADE 2>/dev/null || iptables -t nat -A POSTROUTING -s '${vpn_subnet}' -o '${default_iface}' -j MASQUERADE"
        run_shell "iptables -C FORWARD -i '${tun_device}' -j ACCEPT 2>/dev/null || iptables -A FORWARD -i '${tun_device}' -j ACCEPT"
        run_shell "iptables -C FORWARD -o '${tun_device}' -j ACCEPT 2>/dev/null || iptables -A FORWARD -o '${tun_device}' -j ACCEPT"
    else
        warn "iptables not found; VPN clients may not reach the internet until NAT is configured."
    fi
}

build_binary() {
    [ "${DO_BUILD}" -eq 1 ] || { ok "Build skipped"; return; }

    log "Building aeronyx-server release binary"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] cd %s && cargo build -p aeronyx-server --release\n' "${REPO_DIR}"
    else
        (
            cd "${REPO_DIR}"
            cargo build -p aeronyx-server --release
        )
    fi
}

install_service() {
    local template="${REPO_DIR}/deploy/node/aeronyx-server.service"
    [ "${DRY_RUN}" -eq 1 ] || [ -f "${template}" ] || die "Missing service template: ${template}"

    log "Installing systemd service: ${SERVICE_FILE}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[DRY-RUN] render %s to %s\n' "${template}" "${SERVICE_FILE}"
    else
        sed \
            -e "s|@REPO_DIR@|${REPO_DIR}|g" \
            -e "s|@CONFIG_FILE@|${CONFIG_FILE}|g" \
            "${template}" > "${SERVICE_FILE}"
        chmod 644 "${SERVICE_FILE}"
        systemctl daemon-reload
    fi

    if [ "${DO_ENABLE}" -eq 1 ]; then
        run systemctl enable "${SERVICE_NAME}"
    fi
}

register_node() {
    [ -n "${REGISTRATION_CODE}" ] || return

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
    require_root
    require_linux_systemd
    install_packages
    if [ "${DO_BUILD}" -eq 1 ]; then
        install_rust_if_needed
    else
        ok "Rust toolchain check skipped"
    fi
    prepare_repo
    prepare_directories
    install_config

    if [ "${CONFIG_ONLY}" -eq 1 ]; then
        ok "Config-only install complete."
        exit 0
    fi

    configure_network
    build_binary
    install_service
    register_node
    start_service
}

main "$@"
