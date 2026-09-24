#!/usr/bin/env bash
# ============================================================================
# File: deploy/node/lib/operator_join.sh
# ============================================================================
# Creation Reason:
# - [PERMISSIONLESS-NODE-JOIN-MODULE 2026-09-24 by Codex] Keep the
#   permissionless operator join state machine separate from the unified
#   node command wrapper. This module is sourced after wrapper defaults,
#   parser helpers, and installer functions are defined.
#
# Important Note:
# - Source only from deploy/node/aeronyx-node.sh. The module intentionally
#   inherits strict shell options and emits no network traffic when sourced.
# - Stage-A admission is non-routeable and a plain HTTP response is not a
#   cryptographic seed receipt.
# ============================================================================
# [PERMISSIONLESS-NODE-JOIN 2026-09-24 by Codex] Validate only operator-
# selected public IP base URLs, matching the seed admission endpoint policy.
# The join path never prints them, follows redirects, or calls registration.
validate_join_options() {
    [ -z "${REGISTRATION_CODE}" ] && [ "${REGISTRATION_CODE_STDIN}" -eq 0 ] \
        || die "join does not accept a nodeboard registration credential"
    [ "${#EXTRA_ARGS[@]}" -eq 0 ] || die "Unsupported join option"
    [ -n "${JOIN_PUBLIC_ENDPOINT}" ] && [ "${#JOIN_SEEDS[@]}" -ge 1 ] \
        && [ "${#JOIN_SEEDS[@]}" -le 8 ] \
        || die "join requires one public endpoint and 1-8 independent seeds"
    [[ "${JOIN_TIMEOUT}" =~ ^[0-9]+$ ]] \
        && [ "${JOIN_TIMEOUT}" -ge 1 ] && [ "${JOIN_TIMEOUT}" -le 600 ] \
        || die "join timeout must be 1-600 seconds"
    select_join_python
    command -v curl >/dev/null 2>&1 || die "join requires curl"
    JOIN_PUBLIC_ENDPOINT="${JOIN_PUBLIC_ENDPOINT%/}"
    local seed normalized prior found i
    local -a unique=()
    for seed in "${JOIN_SEEDS[@]}"; do
        normalized="${seed%/}"
        found=0
        for ((i = 0; i < ${#unique[@]}; i++)); do
            prior="${unique[$i]}"
            if [ "${prior}" = "${normalized}" ]; then
                found=1
                break
            fi
        done
        [ "${found}" -eq 1 ] || unique+=("${normalized}")
    done
    JOIN_SEEDS=("${unique[@]}")
    "${JOIN_PYTHON}" - "${JOIN_PUBLIC_ENDPOINT}" "${JOIN_SEEDS[@]}" <<'PY' \
        || die "join endpoint policy rejected a URL"
import ipaddress
import sys
from urllib.parse import urlsplit

def safe_base(raw, seed):
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname
        port = parsed.port
        if (parsed.scheme not in ("http", "https") or not host or
                parsed.username is not None or parsed.password is not None or
                parsed.path or parsed.query or parsed.fragment or
                (port is not None and not 1 <= port <= 65535)):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        else:
            if not address.is_global:
                return False
        return not seed or raw != sys.argv[1]
    except ValueError:
        return False

if not safe_base(sys.argv[1], False) or any(
    not safe_base(seed, True) for seed in sys.argv[2:]
):
    raise SystemExit(1)
PY
}

select_join_python() {
    local candidate
    for candidate in python3 python3.12 python3.11; do
        if command -v "${candidate}" >/dev/null 2>&1 \
            && "${candidate}" -c 'import tomllib' >/dev/null 2>&1; then
            JOIN_PYTHON="${candidate}"
            return 0
        fi
        if command -v "${candidate}" >/dev/null 2>&1 \
            && "${candidate}" -c 'import tomli' >/dev/null 2>&1; then
            JOIN_PYTHON="${candidate}"
            return 0
        fi
    done
    die "join requires Python 3.11+ or python3 with tomli"
}

# Preserve every unknown TOML key and comment. Parse the whole document first,
# reject duplicate/malformed TOML and ambiguous target spans, then replace only
# six exact [discovery] values through a same-directory atomic file replace.
join_config() {
    local mode="$1"
    "${JOIN_PYTHON}" - "${mode}" "${CONFIG_FILE}" "${JOIN_PUBLIC_ENDPOINT}" "${JOIN_SEEDS[@]}" <<'PY'
import json
import os
import re
import stat
import sys
import tempfile
import time

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise SystemExit(1)

mode, path, endpoint, *seeds = sys.argv[1:]
if mode not in ("check", "write"):
    raise SystemExit(1)
if not os.path.exists(path):
    raise SystemExit(2 if mode == "check" else 1)
try:
    before = os.lstat(path)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            before.st_uid != os.geteuid() or before.st_size > 1024 * 1024 or
            before.st_mode & 0o077):
        raise ValueError("unsafe_config")
    file_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(file_fd, "rb") as handle:
        pinned = os.fstat(handle.fileno())
        if (pinned.st_dev, pinned.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("config_changed_before_read")
        raw = handle.read(1024 * 1024 + 1)
    text = raw.decode("utf-8")
    document = tomllib.loads(text)
    discovery = document.get("discovery")
    if not isinstance(discovery, dict):
        raise ValueError("missing_discovery")
    expected = {
        "enabled": True,
        "advertise_self": True,
        "public_discovery": True,
        "gossip_enabled": True,
        "public_endpoint": endpoint,
        "seed_endpoints": seeds,
    }
    if all(discovery.get(key) == value for key, value in expected.items()):
        raise SystemExit(0)
    if mode == "check":
        raise SystemExit(2)

    lines = text.splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines)
               if re.fullmatch(r"\s*\[discovery\]\s*(?:#.*)?\n?", line)]
    if len(headers) != 1:
        raise ValueError("ambiguous_section")
    start = headers[0] + 1
    end = next((i for i in range(start, len(lines))
                if re.match(r"\s*\[", lines[i])), len(lines))
    targets = {}
    for i in range(start, end):
        match = re.match(r"\s*([A-Za-z0-9_-]+)\s*=", lines[i])
        if match and match.group(1) in expected:
            key = match.group(1)
            if key in targets:
                raise ValueError("duplicate_target")
            targets[key] = i
    if any(key in discovery and key not in targets for key in expected):
        raise ValueError("ambiguous_dotted_key")

    def array_end(first):
        depth = 0
        quote = None
        escaped = False
        for row in range(first, end):
            content = lines[row].split("=", 1)[1] if row == first else lines[row]
            for char in content:
                if quote:
                    if quote == '"' and escaped:
                        escaped = False
                    elif quote == '"' and char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                elif char in ("'", '"'):
                    quote = char
                elif char == "#":
                    break
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        return row
        raise ValueError("ambiguous_seed_array")

    replacements = {
        "enabled": "true", "advertise_self": "true",
        "public_discovery": "true", "gossip_enabled": "true",
        "public_endpoint": json.dumps(endpoint),
        "seed_endpoints": json.dumps(seeds),
    }
    edits = []
    for key, row in targets.items():
        if key == "seed_endpoints":
            if not lines[row].split("=", 1)[1].lstrip().startswith("["):
                raise ValueError("ambiguous_seed_array")
            last = array_end(row)
        else:
            if lines[row].split("=", 1)[1].lstrip().startswith(('"""', "'''")):
                raise ValueError("multiline_target")
            last = row
        edits.append((row, last + 1, f"{key} = {replacements[key]}\n"))
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first:last] = [replacement]
    missing = [key for key in expected if key not in targets]
    lines[start:start] = [f"{key} = {replacements[key]}\n" for key in missing]
    updated = "".join(lines)
    after_doc = tomllib.loads(updated)
    if any(after_doc["discovery"].get(key) != value
           for key, value in expected.items()):
        raise ValueError("edit_mismatch")
    old_other = dict(document)
    new_other = dict(after_doc)
    for item in (old_other, new_other):
        item["discovery"] = {key: value for key, value in item["discovery"].items()
                             if key not in expected}
    if old_other != new_other:
        raise ValueError("unrelated_config_changed")

    parent = os.path.dirname(path) or "."
    backup = f"{path}.join-backup.{int(time.time())}.{os.getpid()}"
    backup_fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(backup_fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        fd, temporary = tempfile.mkstemp(prefix=".aeronyx-join-", dir=parent)
        try:
            os.fchmod(fd, stat.S_IMODE(before.st_mode))
            os.fchown(fd, before.st_uid, before.st_gid)
            with os.fdopen(fd, "wb") as handle:
                handle.write(updated.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            current = os.lstat(path)
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise ValueError("config_changed_during_edit")
            os.replace(temporary, path)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except BaseException:
        raise
except SystemExit:
    raise
except (OSError, ValueError, UnicodeError, TypeError, KeyError):
    raise SystemExit(1)
PY
}

join_local_base_url() {
    "${JOIN_PYTHON}" - "${CONFIG_FILE}" <<'PY'
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
try:
    with open(sys.argv[1], "rb") as handle:
        config = tomllib.load(handle)
    listen = config["discovery"]["public_api_listen_addr"]
    if not isinstance(listen, str):
        raise ValueError()
    host, port = listen.rsplit(":", 1)
    if host not in ("0.0.0.0", "127.0.0.1", "[::]", "[::1]"):
        raise ValueError()
    port_number = int(port)
    if not 1 <= port_number <= 65535:
        raise ValueError()
    family = "[::1]" if host.startswith("[") else "127.0.0.1"
    print(f"http://{family}:{port_number}")
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
PY
}

# One read-only local preparation attempt. The output bytes are canonical
# bincode for exactly the locally published self-signed descriptor; the seed
# verifies signature, lifetime, and admission policy at its own boundary.
join_prepare_once() {
    local request_path="$1"
    local base temporary result rc
    base="$(join_local_base_url)" || {
        printf '%s\n' '{"contract_version":"node_join.v1","status":"pending","reason":"local_authority_invalid"}'
        return 1
    }
    temporary="$(mktemp -d "${TMPDIR:-/tmp}/aeronyx-join.XXXXXX")" || return 1
    chmod 700 "${temporary}"
    : >"${temporary}/status.json"
    : >"${temporary}/local.json"
    curl -q -fsS --noproxy '*' --connect-timeout 3 --max-time 5 --max-filesize 524288 \
        --proto '=http,https' "${base}/api/discovery/status" \
        >"${temporary}/status.json" 2>/dev/null || : >"${temporary}/status.json"
    curl -q -fsS --noproxy '*' --connect-timeout 3 --max-time 5 --max-filesize 524288 \
        --proto '=http,https' "${base}/api/discovery/snapshot?limit=64" \
        >"${temporary}/local.json" 2>/dev/null || : >"${temporary}/local.json"
    if result="$("${JOIN_PYTHON}" - "${temporary}" "${CONFIG_FILE}" "${JOIN_PUBLIC_ENDPOINT}" "${#JOIN_SEEDS[@]}" "${request_path}" <<'PY'
import base64
import binascii
import json
import os
import struct
import sys
import time

root, config_path, endpoint, seed_count_raw, request_path = sys.argv[1:]
seed_count = int(seed_count_raw)
now = int(time.time())
try:
    import tomllib
except ImportError:
    import tomli as tomllib
result = {
    "contract_version": "node_join.v1",
    "status": "pending",
    "reason": "local_status_unavailable",
    "signed_descriptor_accepted": False,
    "peer_converged": False,
    "privacy_relay_advertised": False,
    "chat_relay_advertised": False,
    "purpose_receipt_v2_advertised": False,
    "nodeboard_registration_required": False,
    "configured_seeds": seed_count,
    "admission_stage": "none",
    "route_ready": False,
}

def finish(reason, ready=False):
    result["reason"] = reason
    result["status"] = "ready_to_submit" if ready else "pending"
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if ready else 1)

def load(name):
    path = os.path.join(root, name)
    if os.path.getsize(path) > 524288:
        raise ValueError()
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def valid_snapshot(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return False
    generated = snapshot.get("generated_at")
    return (type(generated) is int and abs(now - generated) <= 600 and
            isinstance(snapshot.get("peers"), list) and len(snapshot["peers"]) <= 64)

def signed_fresh(peer):
    if not isinstance(peer, dict):
        return False
    body, signature = peer.get("descriptor"), peer.get("signature")
    if not isinstance(body, dict) or not isinstance(signature, list) or len(signature) != 2:
        return False
    if any(not isinstance(part, list) or len(part) != 32 or
           any(type(byte) is not int or not 0 <= byte <= 255 for byte in part)
           for part in signature):
        return False
    if not any(byte for part in signature for byte in part):
        return False
    node_id = body.get("node_id")
    issued, expires = body.get("issued_at"), body.get("expires_at")
    sequence = body.get("sequence")
    return (isinstance(node_id, list) and len(node_id) == 32 and
            all(type(byte) is int and 0 <= byte <= 255 for byte in node_id) and
            type(sequence) is int and sequence > 0 and
            type(issued) is int and type(expires) is int and
            issued <= now and now + 30 < expires and
            0 < expires - issued <= 7200)

def uint(value, bits):
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError("invalid_unsigned_integer")
    return struct.pack({8: "<B", 16: "<H", 32: "<I", 64: "<Q"}[bits], value)

def fixed(value, length):
    if (not isinstance(value, list) or len(value) != length or
            any(type(item) is not int or not 0 <= item <= 255 for item in value)):
        raise ValueError("invalid_fixed_bytes")
    return bytes(value)

def string(value):
    if not isinstance(value, str):
        raise ValueError("invalid_string")
    encoded = value.encode("utf-8")
    return uint(len(encoded), 64) + encoded

def optional_string(value):
    return b"\x00" if value is None else b"\x01" + string(value)

def optional_u64(value):
    return b"\x00" if value is None else b"\x01" + uint(value, 64)

def boolean(value):
    if type(value) is not bool:
        raise ValueError("invalid_boolean")
    return b"\x01" if value else b"\x00"

def canonical_descriptor(peer):
    body = peer["descriptor"]
    capacity, policy = body["capacity"], body["policy"]
    if not isinstance(capacity, dict) or not isinstance(policy, dict):
        raise ValueError("invalid_descriptor")
    variants = {
        "PrivacyRelay": 0, "ChatRelay": 1, "EncryptedStorage": 2,
        "AgentRelay": 3, "OnionMiddle": 4,
        "DirectoryMirrorCarrier": 5, "BlindVaultReplica": 6,
    }
    capabilities = body["capabilities"]
    if not isinstance(capabilities, list) or len(capabilities) > len(variants):
        raise ValueError("invalid_capabilities")
    encoded_caps = [uint(variants[capability], 32) for capability in capabilities]
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("duplicate_capability")
    encoded = b"".join((
        uint(body["schema_version"], 16), fixed(body["node_id"], 32),
        uint(body["sequence"], 64), uint(body["issued_at"], 64),
        uint(body["expires_at"], 64),
        optional_string(body["public_endpoint"]), string(body["software_version"]),
        uint(len(capabilities), 64), b"".join(encoded_caps),
        uint(capacity["max_sessions"], 32), optional_u64(capacity["max_bps"]),
        optional_u64(capacity["max_pps"]),
        boolean(policy["allows_public_exit"]), boolean(policy["public_discovery"]),
        optional_string(policy["region"]),
        uint(body.get("kem_alg", 0), 8), fixed(body.get("kem_public", [0] * 32), 32),
        fixed(peer["signature"][0], 32), fixed(peer["signature"][1], 32),
    ))
    if len(encoded) > 16 * 1024:
        raise ValueError("descriptor_too_large")
    return encoded

try:
    status = load("status.json")
    if not isinstance(status, dict):
        finish("local_status_unavailable")
    peer_store = status.get("peer_store") or {}
    snapshot_status = peer_store.get("snapshot") or {}
    runtime = peer_store.get("runtime") or {}
    capabilities = status.get("local_capabilities") or {}
    if (capabilities.get("status") == "misconfigured" or
            capabilities.get("capability_config_consistent") is False):
        finish("capability_misconfigured")
    local_snapshot = load("local.json")
    if not valid_snapshot(local_snapshot):
        finish("local_snapshot_invalid")
    with open(config_path, "rb") as handle:
        key_path = tomllib.load(handle)["server_key"]["key_file"]
    if not isinstance(key_path, str) or not key_path.startswith("/"):
        finish("local_identity_invalid")
    with open(key_path, "rb") as handle:
        raw_key = handle.read(65537)
    if len(raw_key) > 65536:
        finish("local_identity_invalid")
    public_key = base64.b64decode(json.loads(raw_key)["public_key"], validate=True)
    if len(public_key) != 32:
        finish("local_identity_invalid")
    matches = [peer for peer in local_snapshot["peers"]
               if signed_fresh(peer) and peer["descriptor"].get("node_id") == list(public_key)]
    if len(matches) != 1:
        finish("own_signed_descriptor_missing")
    own = matches[0]
    body = own["descriptor"]
    if (body.get("public_endpoint") != endpoint or
            (body.get("policy") or {}).get("public_discovery") is not True):
        finish("own_advertisement_mismatch")
    advertised = body.get("capabilities") or []
    features = str(body.get("software_version") or "").partition("+")[2].split(".")
    result["privacy_relay_advertised"] = "PrivacyRelay" in advertised
    result["chat_relay_advertised"] = "ChatRelay" in advertised
    result["purpose_receipt_v2_advertised"] = "anpf1-pbdr2" in features
    if not result["privacy_relay_advertised"]:
        finish("capability_missing")
    other_peers = [peer for peer in local_snapshot["peers"]
                   if signed_fresh(peer) and peer["descriptor"].get("node_id") != list(public_key)]
    valid_count = snapshot_status.get("valid_peers")
    gossip_at = runtime.get("last_gossip_at")
    result["peer_converged"] = (type(valid_count) is int and valid_count >= 2 and
                                 bool(other_peers) and type(gossip_at) is int and
                                 0 <= now - gossip_at <= 600)
    canonical = canonical_descriptor(own)
    request_fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(request_fd, "wb") as handle:
        handle.write(canonical)
        handle.flush()
        os.fsync(handle.fileno())
    finish("local_descriptor_ready", True)
except SystemExit:
    raise
except (OSError, ValueError, TypeError, KeyError, AttributeError,
        UnicodeError, binascii.Error, json.JSONDecodeError):
    finish("local_evidence_invalid")
PY
)"; then
        rc=0
    else
        rc=1
    fi
    rm -f -- "${temporary}"/*.json
    rmdir -- "${temporary}"
    printf '%s\n' "${result}"
    return "${rc}"
}
# Submit exactly once to the first operator-selected seed. A transport timeout
# is ambiguous after bytes may have been sent, so this command never retries
# or silently falls through to another seed.
join_submit_once() {
    local request_path="$1"
    local response_file meta="" transport_rc=0 result rc=0
    response_file="$(mktemp "${TMPDIR:-/tmp}/aeronyx-join-response.XXXXXX")" || return 1
    meta="$(curl -q -sS --noproxy '*' --connect-timeout 3 --max-time 10 \
        --max-filesize 4096 --proto '=http,https' --max-redirs 0 \
        --request POST --header 'Accept: application/json' \
        --header 'Content-Type: application/octet-stream' \
        --data-binary "@${request_path}" --output "${response_file}" \
        --write-out '%{http_code}|%{content_type}' \
        "${JOIN_SEEDS[0]}/api/discovery/join" 2>/dev/null)" || transport_rc=$?
    result="$("${JOIN_PYTHON}" - "${response_file}" "${meta}" "${transport_rc}" "${#JOIN_SEEDS[@]}" <<'PY'
import json
import re
import sys

path, meta, transport_raw, seed_count_raw = sys.argv[1:]
result = {
    "contract_version": "node_join.v1",
    "status": "pending",
    "reason": "transport_ambiguous",
    "signed_descriptor_accepted": False,
    "admission_stage": "none",
    "route_authority": False,
    "route_ready": False,
    "nodeboard_registration_required": False,
    "configured_seeds": int(seed_count_raw),
    "attempted_seeds": 1,
}

def finish(reason, accepted=False):
    result["reason"] = reason
    if accepted:
        result["status"] = "accepted"
        result["signed_descriptor_accepted"] = True
        result["admission_stage"] = "stage_a_candidate"
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if accepted else 1)

if transport_raw != "0":
    finish("transport_ambiguous")
status, separator, content_type = meta.partition("|")
if not separator or not re.fullmatch(r"[0-9]{3}", status):
    finish("transport_ambiguous")
if status != "200":
    finish({
        "400": "descriptor_rejected",
        "403": "descriptor_rejected",
        "409": "descriptor_conflict_or_stale",
        "429": "rate_limited",
        "503": "seed_capacity_reached",
    }.get(status, "http_rejected"))
if content_type.split(";", 1)[0].strip().lower() != "application/json":
    finish("malformed_response")
try:
    with open(path, "rb") as handle:
        raw = handle.read(4097)
    if len(raw) > 4096:
        finish("malformed_response")
    body = json.loads(raw)
except (OSError, ValueError, UnicodeError):
    finish("malformed_response")
if not isinstance(body, dict) or body.get("route_authority") is not False:
    finish("malformed_response")
if body.get("economic_admission") != "reserved_future_eth_projection_not_enforced":
    finish("malformed_response")
if body.get("accepted") is True and body.get("status") in (
    "candidate_admitted", "exact_replay"
):
    finish("stage_a_candidate_admitted", True)
finish("admission_rejected")
PY
)" || rc=1
    rm -f -- "${response_file}"
    printf '%s\n' "${result}"
    return "${rc}"
}

validate_join_binary() {
    local binary="${REPO_DIR}/target/release/aeronyx-server"
    [ -x "${binary}" ] || return 1
    "${binary}" validate -c "${CONFIG_FILE}" >/dev/null 2>&1
}

run_join() {
    validate_join_options
    validate_service_name
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '%s\n' '{"contract_version":"node_join.v1","status":"planned","nodeboard_registration_required":false}'
        return 0
    fi
    if [ "${JOIN_CHECK_ONLY}" -eq 0 ]; then
        [ "${CONFIG_FILE}" = "${DEFAULT_CONFIG_FILE}" ] \
            || die "join install requires the standard node config path"
        [ "${SERVICE_NAME}" = "${DEFAULT_SERVICE_NAME}" ] \
            || die "join install requires the standard systemd service"
        [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" -eq 0 ] \
            || die "join install requires root on Linux/systemd"
        command -v systemctl >/dev/null 2>&1 || die "join install requires systemd"
    fi

    local config_result=0
    join_config check || config_result=$?
    if [ "${config_result}" -eq 1 ]; then
        die "join config is unsafe or malformed"
    fi
    if [ "${JOIN_CHECK_ONLY}" -eq 1 ]; then
        [ "${config_result}" -eq 0 ] || die "join config does not match requested discovery settings"
    elif [ "${config_result}" -eq 2 ]; then
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            die "join settings differ while service is active; no restart or config edit attempted"
        fi
        require_script "${INSTALL_SCRIPT}"
        # Build and install without starting, then make the discovery-only
        # config change. The installer receives no registration code.
        AERONYX_START=0 run_installer --repo-dir "${REPO_DIR}" --branch "${BRANCH}" \
            --no-enable \
            >/dev/null 2>&1 || die "join install failed before service start"
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            die "service became active during install; no config edit or restart attempted"
        fi
        join_config write || die "join config edit failed closed"
        validate_join_binary || die "join config validation failed; service not started"
        systemctl enable "${SERVICE_NAME}" >/dev/null 2>&1 \
            || die "join service enable failed"
        systemctl start "${SERVICE_NAME}" >/dev/null 2>&1 \
            || die "join service start failed"
    elif ! systemctl is-active --quiet "${SERVICE_NAME}"; then
        require_script "${INSTALL_SCRIPT}"
        AERONYX_START=0 run_installer --repo-dir "${REPO_DIR}" --branch "${BRANCH}" \
            --no-enable \
            >/dev/null 2>&1 || die "join install failed before service start"
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            die "service became active during install; no enable, start, or join POST attempted"
        fi
        validate_join_binary || die "join config validation failed; service not started"
        systemctl enable "${SERVICE_NAME}" >/dev/null 2>&1 \
            || die "join service enable failed"
        systemctl start "${SERVICE_NAME}" >/dev/null 2>&1 \
            || die "join service start failed"
    fi

    local deadline=$((SECONDS + JOIN_TIMEOUT)) result="" request_dir request_path rc=0
    request_dir="$(mktemp -d "${TMPDIR:-/tmp}/aeronyx-join-request.XXXXXX")" \
        || die "join could not allocate a private request directory"
    chmod 700 "${request_dir}"
    request_path="${request_dir}/descriptor.bin"
    while true; do
        if result="$(join_prepare_once "${request_path}")"; then
            break
        fi
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            rmdir -- "${request_dir}"
            if [ "${JSON}" -eq 1 ] || [ "${JSON_ONLY}" -eq 1 ]; then
                printf '%s\n' "${result}"
            else
                warn "Join pending: local signed descriptor was not ready before timeout"
            fi
            return 1
        fi
        sleep 3
    done
    if [ "${JOIN_CHECK_ONLY}" -eq 1 ]; then
        rm -f -- "${request_path}"
        rmdir -- "${request_dir}"
        if [ "${JSON}" -eq 1 ] || [ "${JSON_ONLY}" -eq 1 ]; then
            printf '%s\n' "${result}"
        else
            ok "Local signed descriptor ready; no join POST was sent"
        fi
        return 0
    fi
    result="$(join_submit_once "${request_path}")" || rc=1
    rm -f -- "${request_path}"
    rmdir -- "${request_dir}"
    if [ "${JSON}" -eq 1 ] || [ "${JSON_ONLY}" -eq 1 ]; then
        printf '%s\n' "${result}"
    elif [ "${rc}" -eq 0 ]; then
        ok "Stage-A candidate admitted; routeability and economics remain unproven"
    else
        warn "Join not confirmed; no automatic resubmission was attempted"
    fi
    return "${rc}"
}
