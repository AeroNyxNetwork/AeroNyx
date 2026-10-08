#!/bin/sh
# ============================================
# File: deploy/node/tee-entrypoint.sh
# ============================================
# Creation Reason:
# - [TEE-NODE-IMAGE 2026-10-09 by Claude] Container entrypoint for a dstack
#   CVM node. Renders the two deployment-specific values into the config,
#   registers on first start, then execs the node.
#
# Main Logical Flow:
# 1. AERONYX_PUBLIC_ENDPOINT (required): the HTTPS origin the dstack gateway
#    assigns to port 8422. It is part of the compose file, so it is measured.
# 2. AERONYX_REGION (optional, default "tee").
# 3. First start only: AERONYX_REGISTRATION_CODE (encrypted secret) is piped
#    to `register --code-stdin`. The node identity comes from dstack KMS, so
#    registration binds the app-derived key, never a generated file.
#
# Important Notes for Next Developer:
# - Never echo the registration code or the rendered config.
# ============================================
set -eu

STATE=/var/lib/aeronyx
CONFIG="$STATE/server.toml"
TEMPLATE=/etc/aeronyx/server.tee.toml

: "${AERONYX_PUBLIC_ENDPOINT:?set AERONYX_PUBLIC_ENDPOINT to the HTTPS origin of port 8422}"
case "$AERONYX_PUBLIC_ENDPOINT" in
  https://*) ;;
  *) echo "AERONYX_PUBLIC_ENDPOINT must be an https:// origin" >&2; exit 64 ;;
esac
REGION="${AERONYX_REGION:-tee}"

sed -e "s#@PUBLIC_ENDPOINT@#${AERONYX_PUBLIC_ENDPOINT%/}#" \
    -e "s#@REGION@#${REGION}#" \
    "$TEMPLATE" > "$CONFIG.tmp"
mv "$CONFIG.tmp" "$CONFIG"

if [ ! -f "$STATE/node_info.json" ]; then
  : "${AERONYX_REGISTRATION_CODE:?first start needs AERONYX_REGISTRATION_CODE as an encrypted secret}"
  printf '%s\n' "$AERONYX_REGISTRATION_CODE" \
    | /usr/local/bin/aeronyx-server register --code-stdin --config "$CONFIG"
fi

exec /usr/local/bin/aeronyx-server start --config "$CONFIG"
