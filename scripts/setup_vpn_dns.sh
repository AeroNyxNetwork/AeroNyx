#!/usr/bin/env bash
# ============================================
# scripts/setup_vpn_dns.sh — AeroNyx VPN DNS stub setup
# ============================================
# Configures systemd-resolved to listen on the AeroNyx VPN gateway
# address so iOS/macOS clients can use in-tunnel DNS without leaking
# queries to the local network.
#
# This script is idempotent. It can be run during node setup and again
# after aeronyx-server has created the TUN interface.
#
# Usage:
#   sudo scripts/setup_vpn_dns.sh
#   sudo scripts/setup_vpn_dns.sh --gateway 100.64.0.1
#
# The default client DNS server is 100.64.0.1, matching server.toml:
#   [vpn]
#   gateway_ip = "100.64.0.1"
# ============================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

VPN_GATEWAY="100.64.0.1"
UPSTREAM_DNS="1.1.1.1 9.9.9.9"
FALLBACK_DNS="1.0.0.1 149.112.112.112"
DROPIN_DIR="/etc/systemd/resolved.conf.d"
DROPIN_FILE="${DROPIN_DIR}/aeronyx-vpn.conf"
TEST_DOMAIN="api.aeronyx.network"

while [ $# -gt 0 ]; do
    case "$1" in
        --gateway)
            VPN_GATEWAY="${2:-}"
            shift 2
            ;;
        --upstream)
            UPSTREAM_DNS="${2:-}"
            shift 2
            ;;
        --fallback)
            FALLBACK_DNS="${2:-}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--gateway 100.64.0.1] [--upstream '1.1.1.1 9.9.9.9'] [--fallback '1.0.0.1 149.112.112.112']"
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            exit 2
            ;;
    esac
done

if [ "${EUID}" -ne 0 ]; then
    error "This script must run as root because it writes systemd-resolved config."
    exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
    error "systemctl is required. This setup currently supports systemd-based Linux nodes."
    exit 1
fi

if ! systemctl list-unit-files systemd-resolved.service >/dev/null 2>&1; then
    error "systemd-resolved.service is not available on this node."
    exit 1
fi

mkdir -p "${DROPIN_DIR}"
cat > "${DROPIN_FILE}" <<EOF
[Resolve]
DNS=${UPSTREAM_DNS}
FallbackDNS=${FALLBACK_DNS}
DNSStubListener=yes
DNSStubListenerExtra=${VPN_GATEWAY}
EOF

ok "Wrote ${DROPIN_FILE}"

if ! ip addr show | grep -q "inet ${VPN_GATEWAY}/"; then
    warn "${VPN_GATEWAY} is not assigned yet. DNS config is persisted but listener activation is deferred."
    warn "Run this script again after aeronyx-server has created the TUN interface."
    exit 0
fi

info "Restarting systemd-resolved..."
systemctl restart systemd-resolved
sleep 1

if ss -lunpt | grep -Eq "[[:space:]]${VPN_GATEWAY}:53[[:space:]]"; then
    ok "DNS stub is listening on ${VPN_GATEWAY}:53"
else
    error "DNS stub is not listening on ${VPN_GATEWAY}:53"
    ss -lunpt | grep -E ':53\b' || true
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    info "Testing DNS query through ${VPN_GATEWAY}:53 (${TEST_DOMAIN})..."
    VPN_GATEWAY="${VPN_GATEWAY}" TEST_DOMAIN="${TEST_DOMAIN}" python3 - <<'INNERPY'
import os
import random
import socket
import struct
import sys

server = os.environ['VPN_GATEWAY']
name = os.environ['TEST_DOMAIN']
qid = random.randrange(65536)
packet = struct.pack('!HHHHHH', qid, 0x0100, 1, 0, 0, 0)
for part in name.split('.'):
    packet += bytes([len(part)]) + part.encode('ascii')
packet += b'\x00' + struct.pack('!HH', 1, 1)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(4)
sock.sendto(packet, (server, 53))
data, _ = sock.recvfrom(1500)
rqid, flags, _qd, an, _ns, _ar = struct.unpack('!HHHHHH', data[:12])
rcode = flags & 0xF
if rqid != qid or rcode != 0 or an < 1:
    print({'qid_ok': rqid == qid, 'rcode': rcode, 'answers': an}, file=sys.stderr)
    sys.exit(1)
print({'qid_ok': True, 'rcode': rcode, 'answers': an, 'bytes': len(data)})
INNERPY
    ok "DNS query test passed"
else
    warn "python3 not found; skipped DNS query test"
fi
