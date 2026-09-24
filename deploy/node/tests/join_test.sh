#!/usr/bin/env bash
# ============================================
# File: deploy/node/tests/join_test.sh
# ============================================
# Creation Reason:
# - [PERMISSIONLESS-NODE-JOIN 2026-09-24 by Codex] Exercise the operator join
#   command against deterministic in-memory/mock HTTP and temp-file fixtures.
#   No socket, systemd service, registration API, or live node is contacted.
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="${SCRIPT_DIR}/../aeronyx-node.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/aeronyx-join-test.XXXXXX")"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

# [PERMISSIONLESS-NODE-JOIN-MODULE 2026-09-24 by Codex] Sourcing the wrapper
# and sibling module must be inert, even before HTTP/systemd fixtures exist.
curl() { printf 'FAIL: source attempted HTTP\n' >&2; exit 1; }
systemctl() { printf 'FAIL: source attempted systemd\n' >&2; exit 1; }
source "${NODE_SCRIPT}"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf 'PASS: %s\n' "$*"; }
assert_fails() {
    local label="$1"
    shift
    if ( "$@" ) >/dev/null 2>&1; then
        fail "${label} unexpectedly succeeded"
    fi
    pass "${label}"
}

JOIN_PUBLIC_ENDPOINT="https://9.9.9.9:8422"
JOIN_SEEDS=("http://8.8.8.8:8422")
JOIN_TIMEOUT=1
JSON=1
CONFIG_FILE="${TEST_ROOT}/server.toml"
select_join_python

write_config() {
    printf '[server_key]\nkey_file = "%s/key.json"\n\n[discovery]\nenabled = true\nadvertise_self = true\ngossip_enabled = true\npublic_discovery = false\npublic_api_listen_addr = "0.0.0.0:18422"\nseed_endpoints = [\n  "http://9.9.9.9:8422",\n]\nunknown_setting = "preserve-me"\n\n[other]\nunknown = 42\n' \
        "${TEST_ROOT}" >"${CONFIG_FILE}"
    chmod 600 "${CONFIG_FILE}"
}

write_fixtures() {
    "${JOIN_PYTHON}" - "${TEST_ROOT}" <<'PY'
import base64
import json
import os
import sys
import time

root = sys.argv[1]
now = int(time.time())

def peer(byte, endpoint):
    return {
        "descriptor": {
            "schema_version": 2,
            "node_id": [byte] * 32,
            "sequence": 7,
            "issued_at": now - 10,
            "expires_at": now + 300,
            "public_endpoint": endpoint,
            "policy": {
                "allows_public_exit": False,
                "public_discovery": True,
                "region": None,
            },
            "capabilities": ["PrivacyRelay"],
            "capacity": {
                "max_sessions": 64, "max_bps": None, "max_pps": None,
            },
            "software_version": "0.1.0+anpf1-pbdr2",
            "kem_alg": 0,
            "kem_public": [0] * 32,
        },
        "signature": [[byte] * 32, [byte + 1] * 32],
    }

own = peer(1, "https://9.9.9.9:8422")
other = peer(3, "https://1.1.1.1:8422")
snapshot = {"schema_version": 1, "generated_at": now, "peers": [own, other]}
status = {
    "peer_store": {
        "snapshot": {"valid_peers": 2},
        "runtime": {"last_gossip_at": now},
    },
    "local_capabilities": {
        "status": "ready", "capability_config_consistent": True,
    },
}

def write(name, body):
    with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
        json.dump(body, handle)

write("key.json", {"public_key": base64.b64encode(bytes([1] * 32)).decode()})
write("status.json", status)
write("local.json", snapshot)
write("response_accept.json", {
    "accepted": True, "status": "candidate_admitted",
    "route_authority": False,
    "economic_admission": "reserved_future_eth_projection_not_enforced",
})
write("response_replay.json", {
    "accepted": True, "status": "exact_replay",
    "route_authority": False,
    "economic_admission": "reserved_future_eth_projection_not_enforced",
})
write("response_reject.json", {
    "accepted": False, "status": "rejected",
    "route_authority": False,
    "economic_admission": "reserved_future_eth_projection_not_enforced",
})
write("response_authority.json", {
    "accepted": True, "status": "candidate_admitted",
    "route_authority": True,
    "economic_admission": "reserved_future_eth_projection_not_enforced",
})
unsigned = json.loads(json.dumps(snapshot))
unsigned["peers"][0].pop("signature")
write("unsigned.json", unsigned)
expired = json.loads(json.dumps(snapshot))
expired["peers"][0]["descriptor"]["expires_at"] = now - 1
write("expired.json", expired)
PY
}

# The function shadows curl only in this test shell. Its final argument is the
# URL, but neither that URL nor any response body reaches test stdout.
curl() {
    local requested="${@: -1}"
    case "${requested}" in
        */api/discovery/status) command cat "${TEST_ROOT}/status.json" ;;
        *127.0.0.1*/api/discovery/snapshot*)
            command cat "${TEST_ROOT}/${LOCAL_MODE:-local}.json"
            ;;
        */api/discovery/join)
            local output="" previous="" argument
            for argument in "$@"; do
                if [ "${previous}" = "--output" ]; then
                    output="${argument}"
                fi
                previous="${argument}"
            done
            [ -n "${output}" ] || return 22
            printf 'post\n' >>"${TEST_ROOT}/post_calls"
            if [ "${SEED_MODE:-accept}" = "timeout" ]; then
                return 28
            fi
            local source_mode="${SEED_MODE:-accept}"
            case "${source_mode}" in
                conflict|malformed|false_ok) source_mode="reject" ;;
            esac
            command cp "${TEST_ROOT}/response_${source_mode}.json" "${output}"
            case "${SEED_MODE:-accept}" in
                conflict) printf '409|application/json' ;;
                reject) printf '400|application/json' ;;
                malformed) printf '200|text/plain' ;;
                *) printf '200|application/json' ;;
            esac
            ;;
        *) return 22 ;;
    esac
}

write_config
write_fixtures

assert_fails "initial config differs from requested join" join_config check
join_config write || fail "config edit failed"
join_config check || fail "updated config does not match"
cp "${CONFIG_FILE}" "${TEST_ROOT}/first.toml"
join_config write || fail "idempotent config write failed"
cmp -s "${CONFIG_FILE}" "${TEST_ROOT}/first.toml" || fail "idempotent rerun changed config"
"${JOIN_PYTHON}" - "${CONFIG_FILE}" <<'PY' || fail "unknown TOML changed"
import sys
import tomllib
with open(sys.argv[1], "rb") as handle:
    config = tomllib.load(handle)
assert config["discovery"]["unknown_setting"] == "preserve-me"
assert config["other"]["unknown"] == 42
assert config["discovery"]["seed_endpoints"] == ["http://8.8.8.8:8422"]
assert config["discovery"]["public_discovery"] is True
PY
pass "config edit preserves unknown TOML and is idempotent"

request_path="${TEST_ROOT}/request.bin"
result="$(join_prepare_once "${request_path}")" || fail "valid local descriptor was not prepared"
[ -s "${request_path}" ] || fail "canonical request is empty"
"${JOIN_PYTHON}" - "${result}" "${request_path}" <<'PY' || fail "local-ready contract invalid"
import json, sys
value = json.loads(sys.argv[1])
request = open(sys.argv[2], "rb").read()
assert value["status"] == "ready_to_submit"
assert value["signed_descriptor_accepted"] is False
assert value["route_ready"] is False
assert value["peer_converged"] is True
assert value["privacy_relay_advertised"] is True
assert value["purpose_receipt_v2_advertised"] is True
assert value["nodeboard_registration_required"] is False
assert request[:2] == b"\x02\x00"
assert request[2:34] == bytes([1] * 32)
assert len(request) < 16 * 1024
assert all(term not in sys.argv[1] for term in (
    "8.8.8.8", "9.9.9.9", "node_id", "signature", "public_key", "payload"
))
PY
pass "fresh local descriptor yields canonical binary and aggregate-only readiness"

SEED_MODE="accept"
result="$(join_submit_once "${request_path}")" || fail "candidate admission rejected valid response"
[[ "${result}" == *'"admission_stage":"stage_a_candidate"'* ]] \
    && [[ "${result}" == *'"route_ready":false'* ]] \
    || fail "Stage-A result implied route authority"
[[ "${result}" != *'8.8.8.8'* && "${result}" != *'9.9.9.9'* \
    && "${result}" != *'node_id'* && "${result}" != *'signature'* \
    && "${result}" != *'payload'* ]] || fail "admission output leaked descriptor data"
pass "request-bound Stage-A admission is not route-ready"

SEED_MODE="replay"
result="$(join_submit_once "${request_path}")" || fail "exact replay was not idempotent"
[[ "${result}" == *'"status":"accepted"'* ]] || fail "exact replay was not accepted"
pass "exact replay remains idempotent"

SEED_MODE="timeout"
result="$(join_submit_once "${request_path}")" && fail "ambiguous timeout was accepted"
[[ "${result}" == *'"reason":"transport_ambiguous"'* ]] || fail "timeout reason missing"
pass "transport timeout is ambiguous and fails closed"

SEED_MODE="reject"
result="$(join_submit_once "${request_path}")" && fail "HTTP 400 was accepted"
[[ "${result}" == *'"reason":"descriptor_rejected"'* ]] || fail "HTTP rejection reason missing"
pass "HTTP descriptor rejection fails closed"

SEED_MODE="conflict"
result="$(join_submit_once "${request_path}")" && fail "HTTP 409 was accepted"
[[ "${result}" == *'"reason":"descriptor_conflict_or_stale"'* ]] \
    || fail "conflict reason missing"
pass "stale or conflicting descriptor fails closed"

SEED_MODE="malformed"
result="$(join_submit_once "${request_path}")" && fail "malformed response was accepted"
[[ "${result}" == *'"reason":"malformed_response"'* ]] || fail "malformed reason missing"
pass "malformed response fails closed"

SEED_MODE="false_ok"
result="$(join_submit_once "${request_path}")" && fail "HTTP 200 with accepted=false was accepted"
[[ "${result}" == *'"reason":"admission_rejected"'* ]] || fail "false acceptance reason missing"
pass "HTTP 200 rejected body fails closed"

SEED_MODE="authority"
result="$(join_submit_once "${request_path}")" && fail "route-authority response was accepted"
[[ "${result}" == *'"reason":"malformed_response"'* ]] \
    || fail "route-authority rejection reason missing"
pass "Stage-A response cannot grant route authority"

LOCAL_MODE="unsigned"
assert_fails "unsigned local descriptor is rejected" join_prepare_once "${TEST_ROOT}/unsigned.bin"
LOCAL_MODE="expired"
assert_fails "expired local descriptor is rejected" join_prepare_once "${TEST_ROOT}/expired.bin"
unset LOCAL_MODE
unset SEED_MODE

write_config
"${JOIN_PYTHON}" - "${CONFIG_FILE}" <<'PY'
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    content = handle.read()
content = content.replace("\n[other]\n", '\nseed_endpoints = ["http://8.8.8.8:8422"]\n\n[other]\n')
with open(path, "w", encoding="utf-8") as handle:
    handle.write(content)
PY
cp "${CONFIG_FILE}" "${TEST_ROOT}/malformed.toml"
assert_fails "duplicate TOML key is rejected" join_config write
cmp -s "${CONFIG_FILE}" "${TEST_ROOT}/malformed.toml" || fail "malformed config was modified"
pass "malformed config remains untouched"

write_config
JOIN_SEEDS=("http://127.0.0.1:8422")
assert_fails "loopback seed rejected" bash -c 'source "$1"; JOIN_PUBLIC_ENDPOINT=https://9.9.9.9:8422; JOIN_SEEDS=(http://127.0.0.1:8422); validate_join_options' bash "${NODE_SCRIPT}"
assert_fails "DNS self endpoint rejected by Stage-A policy" bash -c 'source "$1"; JOIN_PUBLIC_ENDPOINT=https://node.example; JOIN_SEEDS=(http://8.8.8.8:8422); validate_join_options' bash "${NODE_SCRIPT}"
assert_fails "seed credentials rejected" bash -c 'source "$1"; JOIN_PUBLIC_ENDPOINT=https://9.9.9.9:8422; JOIN_SEEDS=(http://user:pass@8.8.8.8:8422); validate_join_options' bash "${NODE_SCRIPT}"
assert_fails "seed query rejected" bash -c 'source "$1"; JOIN_PUBLIC_ENDPOINT=https://9.9.9.9:8422; JOIN_SEEDS=(http://8.8.8.8:8422?x=1); validate_join_options' bash "${NODE_SCRIPT}"
JOIN_SEEDS=("http://8.8.8.8:8422")
pass "seed policy rejects local/private targets"

# Exercise the full orchestration without root, systemd, install, or sockets.
# Explicit function fakes record only aggregate call shape.
DEFAULT_CONFIG_FILE="${CONFIG_FILE}"
REPO_DIR="${TEST_ROOT}/repo"
uname() { printf 'Linux\n'; }
id() { printf '0\n'; }
systemctl() {
    if [ "$1" = "is-active" ]; then
        [ "${SERVICE_ACTIVE:-0}" -eq 0 ] || return 0
        if [ "${ACTIVATE_DURING_INSTALL:-0}" -eq 1 ]; then
            local seen=0
            [ ! -f "${TEST_ROOT}/active_checks" ] \
                || seen="$(command cat "${TEST_ROOT}/active_checks")"
            seen=$((seen + 1))
            printf '%s\n' "${seen}" >"${TEST_ROOT}/active_checks"
            [ "${seen}" -le 1 ] && return 1
            return 0
        fi
        return 1
    fi
    printf '%s\n' "$1" >>"${TEST_ROOT}/systemctl_calls"
}
require_script() { :; }
run_installer() {
    [ "${AERONYX_START:-}" = "0" ] || return 1
    printf '%s\n' "$@" >"${TEST_ROOT}/installer_args"
}
validate_join_binary() { :; }
result="$(run_join)" || fail "one-command join orchestration failed"
[[ "${result}" != *'8.8.8.8'* && "${result}" != *'9.9.9.9'* ]] \
    || fail "join output exposed endpoint"
[[ "${result}" == *'"status":"accepted"'* ]] || fail "join did not report acceptance"
[[ "${result}" == *'"route_ready":false'* ]] || fail "join claimed route readiness"
if command grep -i 'registration\|register' "${TEST_ROOT}/installer_args" >/dev/null; then
    fail "installer received central registration option"
fi
[[ "$(command cat "${TEST_ROOT}/systemctl_calls")" == $'enable\nstart' ]] \
    || fail "join restarted or skipped enable/start"
command grep -q '^--no-enable$' "${TEST_ROOT}/installer_args" \
    || fail "installer could enable the service before config validation"
pass "one-command install sends signed Stage-A request without registration or restart"

SERVICE_ACTIVE=1
JOIN_CHECK_ONLY=1
post_count_before="$(wc -l <"${TEST_ROOT}/post_calls")"
result="$(run_join)" || fail "read-only check failed"
[[ "${result}" == *'"status":"ready_to_submit"'* ]] \
    || fail "read-only check claimed remote acceptance"
[[ "$(wc -l <"${TEST_ROOT}/post_calls")" -eq "${post_count_before}" ]] \
    || fail "read-only check sent a POST"
JOIN_CHECK_ONLY=0
pass "check-only proves local readiness without network admission"

SEED_MODE="timeout"
result="$(run_join)" && fail "ambiguous run_join timeout was accepted"
[[ "${result}" == *'"reason":"transport_ambiguous"'* ]] \
    || fail "run_join timeout reason missing"
[[ "$(wc -l <"${TEST_ROOT}/post_calls")" -eq "$((post_count_before + 1))" ]] \
    || fail "ambiguous timeout triggered an automatic retry"
unset SEED_MODE SERVICE_ACTIVE
pass "ambiguous join timeout emits one POST and does not retry"

write_config
cp "${CONFIG_FILE}" "${TEST_ROOT}/before-active.toml"
post_count_before="$(wc -l <"${TEST_ROOT}/post_calls")"
ACTIVATE_DURING_INSTALL=1
assert_fails "service activation during install blocks config edit" run_join
cmp -s "${CONFIG_FILE}" "${TEST_ROOT}/before-active.toml" \
    || fail "config changed after service became active"
[[ "$(wc -l <"${TEST_ROOT}/post_calls")" -eq "${post_count_before}" ]] \
    || fail "service race sent a join request"
pass "active-service race leaves config untouched"

unset ACTIVATE_DURING_INSTALL
join_config write || fail "preparing same-config race failed"
rm -f -- "${TEST_ROOT}/active_checks"
post_count_before="$(wc -l <"${TEST_ROOT}/post_calls")"
ACTIVATE_DURING_INSTALL=1
assert_fails "service activation during identical install blocks POST" run_join
[[ "$(wc -l <"${TEST_ROOT}/post_calls")" -eq "${post_count_before}" ]] \
    || fail "same-config service race sent a join request"
pass "same-config active-service race remains fail-closed"

printf 'PASS: all join fixtures\n'
