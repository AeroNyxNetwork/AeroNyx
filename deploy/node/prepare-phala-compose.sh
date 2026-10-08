#!/usr/bin/env bash
# [PHALA-IMAGE-DIGEST-PIN 2026-10-06 by Codex]
# Render the Phala app's Compose input with an immutable image and the dstack
# guest-agent socket required by this profile's HTTP API. Public ingress and
# the private-recipient service are separate explicit renderer options.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($# > 2)); then
  printf '%s\n' 'usage: prepare-phala-compose.sh [--public-peer] [--private-recipient]' >&2
  exit 2
fi
python3 - "${AERONYX_NODE_IMAGE-}" "$script_dir/compose.phala.peer.yaml" "$@" <<'PY'
import re
import sys
import os
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

image = sys.argv[1]
template_path = Path(sys.argv[2])
requested_modes = sys.argv[3:]
allowed_modes = {"--public-peer", "--private-recipient"}
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}", image):
    raise SystemExit("AERONYX_NODE_IMAGE must be a repository reference pinned by sha256 digest")
if len(set(requested_modes)) != len(requested_modes) or not set(requested_modes) <= allowed_modes:
    raise SystemExit("usage: prepare-phala-compose.sh [--public-peer] [--private-recipient]")

# [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Validate the protected
# outgoing-route policy before output. These pins are operator supplied, not
# learned from a deployment response. Empty values preserve mounted TOML.
phala_policy_names = (
    "AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED",
    "AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS",
    "AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES",
    "AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS",
)
policy_required, policy_apps, policy_compose, policy_age = (
    os.environ.get(name, "") for name in phala_policy_names
)
def phala_policy_pins(raw, pattern):
    if not raw or len(raw.encode("utf-8")) > 8192:
        raise SystemExit("Phala route pins require bounded nonempty JSON arrays")
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise SystemExit("Phala route pins must be JSON arrays")
    if not isinstance(values, list) or not 1 <= len(values) <= 64 or any(
        not isinstance(value, str) or not re.fullmatch(pattern, value) for value in values
    ) or len(set(values)) != len(values):
        raise SystemExit("Phala route pins must contain 1..=64 distinct canonical measured identities")

if policy_required == "true":
    phala_policy_pins(policy_apps, r"0x(?:[0-9a-f]{2}){1,64}")
    phala_policy_pins(policy_compose, r"sha256:[0-9a-f]{64}")
    if policy_age and (not re.fullmatch(r"[1-9][0-9]{0,4}", policy_age)
                       or int(policy_age) > 86400):
        raise SystemExit("Phala route appraisal age must be canonical decimal 1..=86400")
elif policy_required in ("", "false"):
    if policy_apps or policy_compose or policy_age:
        raise SystemExit("Phala route pin inputs require explicit strict mode")
else:
    raise SystemExit("Phala route strict mode must be empty, true, or false")

# [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Keep literal-address
# exclusions explicit and aligned with Rust's peer transport gate rather than
# relying on Python-version-dependent is_global classifications.
ipv4_nonpublic = tuple(ipaddress.ip_network(prefix) for prefix in (
    "0.0.0.0/8", "10.0.0.0/8", "127.0.0.0/8", "100.64.0.0/10",
    "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24",
    "203.0.113.0/24", "224.0.0.0/3",
))
ipv6_global_unicast = ipaddress.ip_network("2000::/3")
ipv6_nonpublic = tuple(ipaddress.ip_network(prefix) for prefix in (
    "2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20",
))

def is_public_unicast_address(address):
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is None and int(address) >> 32 == 0:
            mapped = ipaddress.IPv4Address(int(address))
        if mapped is not None:
            return is_public_unicast_address(mapped)
        return address in ipv6_global_unicast and not any(
            address in network for network in ipv6_nonpublic
        )
    return not any(address in network for network in ipv4_nonpublic)

def is_public_https_origin(value):
    if not value or len(value) > 2048 or any(character.isspace() for character in value):
        return False
    try:
        url = urlsplit(value)
        host = url.hostname
        port = url.port
    except ValueError:
        return False
    if not host:
        return False
    # DNS names are not failed IP literals. Parse the URL/port independently
    # before classifying the host, then leave answer-set pinning to Rust.
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    reserved_dns_suffixes = (
        ".localhost", ".local", ".internal", ".test", ".invalid", ".example", ".onion"
    )
    return not (
        url.scheme != "https"
        or not host
        or url.username is not None
        or url.password is not None
        or url.path not in ("", "/")
        or "?" in value
        or "#" in value
        or port == 0
        or (address is not None and not is_public_unicast_address(address))
        or (address is None and (
            "." not in host
            or host.startswith(".")
            or host.endswith(".")
            or len(host) > 253
            or any(host.lower().endswith(suffix) for suffix in reserved_dns_suffixes)
            or host.rsplit(".", 1)[-1].isdigit()
            or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                   for label in host.split("."))
        ))
    )

# [PHALA-RECIPIENT-BOOTSTRAP-RENDER 2026-10-06 by Codex] Validate identity
# and public transport pins before rendering an enabled endpoint-free worker.
if "--private-recipient" in requested_modes:
    # [PHALA-RENDER-REQUIRED-PINS 2026-10-08 by Codex] Reject an invalid
    # safety-mode override before emitting an otherwise usable manifest.
    recipient_recovery_only = os.environ.get("AERONYX_REVERSE_ONION_RECOVERY_ONLY", "")
    if recipient_recovery_only not in ("", "true", "false"):
        raise SystemExit("AERONYX_REVERSE_ONION_RECOVERY_ONLY must be empty, true, or false")
    # [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Validate the exact
    # protected values that Compose will supply, not silently trimmed copies.
    relay_id = os.environ.get("AERONYX_REVERSE_ONION_RELAY_NODE_ID", "")
    relay_endpoint = os.environ.get("AERONYX_REVERSE_ONION_RELAY_ENDPOINT", "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", relay_id) or relay_id == "0" * 64:
        raise SystemExit(
            "--private-recipient requires AERONYX_REVERSE_ONION_RELAY_NODE_ID as a nonzero 64-hex node ID"
        )
    if not is_public_https_origin(relay_endpoint):
        raise SystemExit(
            "--private-recipient requires AERONYX_REVERSE_ONION_RELAY_ENDPOINT to be a public HTTPS origin"
        )

queue_enabled = os.environ.get("AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED", "false")
if queue_enabled not in ("true", "false"):
    raise SystemExit("AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED must be exactly true or false")
# [PHALA-QUEUE-ROLE-RENDER-GATE 2026-10-06 by Codex] Match Rust's public
# relay-role dependency before emitting a manifest that would fail at startup.
onion_relay_enabled = os.environ.get("AERONYX_PHALA_ONION_RELAY_ENABLED", "false")
if onion_relay_enabled not in ("true", "false"):
    raise SystemExit("AERONYX_PHALA_ONION_RELAY_ENABLED must be exactly true or false")
if onion_relay_enabled == "true" and "--public-peer" not in requested_modes:
    # [PHALA-RELAY-INGRESS-RENDER-GATE 2026-10-06 by Codex] Do not silently
    # render an enabled local store that cannot be reached or advertised.
    raise SystemExit(
        "AERONYX_PHALA_ONION_RELAY_ENABLED=true requires the explicit --public-peer profile"
    )
if queue_enabled == "true" and onion_relay_enabled != "true":
    raise SystemExit(
        "reverse-onion queue requires AERONYX_PHALA_ONION_RELAY_ENABLED=true"
    )
queue_recovery_only = os.environ.get("AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY", "")
if queue_recovery_only not in ("", "true", "false"):
    raise SystemExit(
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY must be empty, true, or false"
    )

def queue_node_ids(environment_name, minimum, maximum):
    raw = os.environ.get(environment_name, "")
    # [PHALA-RENDER-REQUIRED-PINS 2026-10-08 by Codex] The Rust startup
    # gate bounds bytes and requires these pins for both live and recovery
    # queues. Missing input must not bypass the minimum cardinality below.
    if len(raw.encode("utf-8")) > 8 * 1024:
        raise SystemExit(f"{environment_name} exceeds 8192 bytes")
    if not raw:
        raise SystemExit(f"{environment_name} requires a JSON array of node IDs")
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise SystemExit(f"{environment_name} must be a JSON array of node IDs")
    if not isinstance(values, list) or not minimum <= len(values) <= maximum or any(
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
        or value == "0" * 64
        for value in values
    ):
        raise SystemExit(f"{environment_name} must contain {minimum}..={maximum} nonzero 64-hex node IDs")
    normalized = [value.lower() for value in values]
    if len(set(normalized)) != len(normalized):
        raise SystemExit(f"{environment_name} contains duplicate node IDs")
    return normalized

if queue_enabled == "true":
    recipient_ids = queue_node_ids(
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS", 1, 1
    )
    source_value = os.environ.get("AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS", "")
    if queue_recovery_only == "true":
        if source_value not in ("", "[]"):
            raise SystemExit("recovery-only queue must not receive source IDs")
        source_ids = []
    else:
        source_ids = queue_node_ids(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS", 1, 64
        )
    if set(recipient_ids) & set(source_ids):
        raise SystemExit("queue source IDs must not include the recipient ID")

    authority_names = (
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64",
    )
    authority_present = [bool(os.environ.get(name, "")) for name in authority_names]
    if any(authority_present) and not all(authority_present):
        raise SystemExit("queue signed authority inputs must be supplied together")
    if queue_recovery_only == "true" and any(authority_present):
        raise SystemExit("recovery-only queue must not receive live authority inputs")
elif queue_recovery_only == "true":
    raise SystemExit("queue recovery mode requires AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED=true")
else:
    disabled_queue_inputs = (
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64",
        "AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64",
    )
    if any(os.environ.get(name, "") for name in disabled_queue_inputs):
        raise SystemExit("disabled queue must not receive identity or authority inputs")

public_endpoint = ""
if "--public-peer" in requested_modes:
    # [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] The required-value
    # expression receives the original input, so never validate a trimmed copy.
    public_endpoint = os.environ.get("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", "")
    seeds_encoded = os.environ.get("AERONYX_DISCOVERY_SEED_ENDPOINTS", "")
    if public_endpoint and not is_public_https_origin(public_endpoint):
        raise SystemExit(
            "--public-peer AERONYX_DISCOVERY_PUBLIC_ENDPOINT must be a public HTTPS origin when set"
        )
    if queue_enabled == "true" and not public_endpoint:
        raise SystemExit(
            "reverse-onion queue requires an assigned public HTTPS endpoint; leave it disabled on first render"
        )
    if onion_relay_enabled == "true" and not public_endpoint:
        # [PHALA-RELAY-ENDPOINT-RENDER-GATE 2026-10-06 by Codex] OnionMiddle
        # is useful only after the assigned HTTPS origin can be signed.
        raise SystemExit(
            "AERONYX_PHALA_ONION_RELAY_ENABLED=true requires an assigned public HTTPS endpoint"
        )
    if len(seeds_encoded) > 8 * 1024:
        raise SystemExit("--public-peer discovery seeds exceed 8192 bytes")
    try:
        seeds = json.loads(seeds_encoded)
    except (TypeError, json.JSONDecodeError):
        raise SystemExit("--public-peer requires AERONYX_DISCOVERY_SEED_ENDPOINTS as a JSON array")
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 64 or any(
        not isinstance(seed, str) or not is_public_https_origin(seed) for seed in seeds
    ):
        raise SystemExit(
            "--public-peer requires 1..=64 public HTTPS origins in AERONYX_DISCOVERY_SEED_ENDPOINTS"
        )
elif queue_enabled == "true":
    raise SystemExit("reverse-onion queue requires the explicit --public-peer profile")

image_marker = "${AERONYX_NODE_IMAGE:?Set AERONYX_NODE_IMAGE to a built image}"
profile_marker = '    profiles: ["private-recipient"]\n'
public_peer_profile_marker = "    # PHALA_PUBLIC_PEER_PROFILE\n"
public_peer_profile = '    profiles: ["public-peer"]\n'
private_network_binding = "    networks:\n      - private-recipient-egress\n"
private_network_definition = (
    "networks:\n  private-recipient-egress:\n    driver: bridge\n"
)
# [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] Verify the renderer cannot
# silently drop or inherit the public/private relay role boundary.
onion_relay_public_env = '      AERONYX_PHALA_ONION_RELAY_ENABLED: "${AERONYX_PHALA_ONION_RELAY_ENABLED:-false}"\n'
onion_relay_private_env = '      AERONYX_PHALA_ONION_RELAY_ENABLED: "false"\n'
# [PHALA-REVERSE-QUEUE-CONFIG 2026-10-06 by Codex] A public middle-hop
# advertisement is separate from the durable private-recipient task queue.
public_reverse_queue_env = (
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED:-false}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY:-}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS:-}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS:-}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64:-}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64:-}"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64: "${AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64:-}"\n'
)
private_reverse_queue_env = (
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED: "false"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY: "false"\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS: ""\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS: ""\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64: ""\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64: ""\n'
    '      AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64: ""\n'
)
public_reverse_queue_marker = "      # PHALA_PUBLIC_REVERSE_QUEUE_ENV\n"
# [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
private_visibility_env = '      AERONYX_DISCOVERY_PUBLIC_DISCOVERY: "false"\n'
# [PHALA-PRIVATE-RECIPIENT-RENDER-GUARD 2026-10-06 by Codex] Keep the
# endpoint-free recipient from inheriting the public listener or quote socket.
private_endpoint_env = '      AERONYX_DISCOVERY_PUBLIC_ENDPOINT: ""\n'
private_api_env = '      AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR: ""\n'
private_attestation_env = '      AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH: ""\n'
private_seeds_env = '      AERONYX_DISCOVERY_SEED_ENDPOINTS: ""\n'
recipient_relay_id_optional = '      AERONYX_REVERSE_ONION_RELAY_NODE_ID: "${AERONYX_REVERSE_ONION_RELAY_NODE_ID:-}"\n'
recipient_relay_endpoint_optional = '      AERONYX_REVERSE_ONION_RELAY_ENDPOINT: "${AERONYX_REVERSE_ONION_RELAY_ENDPOINT:-}"\n'
recipient_recovery_mode_env = '      AERONYX_REVERSE_ONION_RECOVERY_ONLY: "${AERONYX_REVERSE_ONION_RECOVERY_ONLY:-}"\n'
recipient_relay_id_required = '      AERONYX_REVERSE_ONION_RELAY_NODE_ID: "${AERONYX_REVERSE_ONION_RELAY_NODE_ID:?Set AERONYX_REVERSE_ONION_RELAY_NODE_ID to the pinned relay node ID}"\n'
recipient_relay_endpoint_required = '      AERONYX_REVERSE_ONION_RELAY_ENDPOINT: "${AERONYX_REVERSE_ONION_RELAY_ENDPOINT:?Set AERONYX_REVERSE_ONION_RELAY_ENDPOINT to the pinned relay HTTPS origin}"\n'
public_ingress_marker = "    # PHALA_PUBLIC_PEER_INGRESS\n"
public_endpoint_marker = "      # PHALA_PUBLIC_PEER_ENDPOINT_ENV\n"
public_discovery_marker = "      # PHALA_PUBLIC_PEER_DISCOVERY_ENV\n"
public_api_marker = "      # PHALA_PUBLIC_PEER_API_LISTENER_ENV\n"
public_attestation_marker = "      # PHALA_PUBLIC_PEER_ATTESTATION_SOCKET_ENV\n"
public_seeds_marker = "      # PHALA_PUBLIC_PEER_SEEDS_ENV\n"
public_dstack_marker = "      # PHALA_PUBLIC_PEER_DSTACK_SOCKET\n"
# [PHALA-PEER-INGRESS-OPT-IN 2026-10-06 by Codex] Phala treats this mapping as
# externally reachable; keep it absent unless the caller explicitly opts in.
public_ingress_mapping = '    ports:\n      - "8422:8422"\n'
# [PHALA-PEER-ENDPOINT-ADVERTISEMENT 2026-10-06 by Codex] Only the explicit
# public-ingress render may receive and advertise an externally assigned host.
public_peer_endpoint_env = '      AERONYX_DISCOVERY_PUBLIC_ENDPOINT: "${AERONYX_DISCOVERY_PUBLIC_ENDPOINT:?Set AERONYX_DISCOVERY_PUBLIC_ENDPOINT to a public HTTPS origin}"\n'
public_peer_endpoint_unassigned_env = '      AERONYX_DISCOVERY_PUBLIC_ENDPOINT: ""\n'
public_peer_visibility_env = '      AERONYX_DISCOVERY_PUBLIC_DISCOVERY: "true"\n'
public_peer_api_env = '      AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR: "0.0.0.0:8422"\n'
# [PHALA-PEER-QUOTE-GATE 2026-10-06 by Codex] Keep the guest socket and
# nonce-bound quote API coupled to the same explicit public-peer profile.
public_peer_attestation_env = '      AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH: "/var/run/aeronyx/phala-agent.sock"\n'
public_peer_seeds_env = '      AERONYX_DISCOVERY_SEED_ENDPOINTS: "${AERONYX_DISCOVERY_SEED_ENDPOINTS:?Set AERONYX_DISCOVERY_SEED_ENDPOINTS to a JSON array of public HTTPS seed origins}"\n'
public_dstack_mount = (
    "      - type: bind\n"
    "        source: /var/run/dstack.sock\n"
    "        target: /var/run/aeronyx/phala-agent.sock\n"
    "        bind:\n"
    "          create_host_path: false\n"
)
template = template_path.read_text(encoding="utf-8")
# [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Counts across the whole
# template cannot prove role ownership: a duplicated public marker formerly
# hid the missing private override. Validate both sections before substitution.
public_service, private_separator, private_service = template.partition("  aeronyx-private-recipient:\n")
private_service = private_service.partition("\nvolumes:\n")[0]
# [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex] Validate each role
# independently; a duplicated public key cannot hide a missing private key.
def shutdown_policy_matches(service):
    return (
        service.count("stop_signal:") == 1
        and service.count("    stop_signal: SIGTERM\n") == 1
        and service.count("stop_grace_period:") == 1
        and service.count("    stop_grace_period: 2m\n") == 1
    )

if not shutdown_policy_matches(public_service) or not shutdown_policy_matches(private_service):
    raise SystemExit("each Phala role requires exactly SIGTERM and a two-minute stop grace")

# [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Check each process, not
# aggregate marker counts: public duplication cannot hide a private omission.
def phala_policy_environment_matches(service):
    return all(
        service.count("      " + name + ":") == 1
        and service.count(f'      {name}: "${{{name}:-}}"\n') == 1
        for name in phala_policy_names
    )
if not all(phala_policy_environment_matches(service) for service in (public_service, private_service)):
    raise SystemExit("each Phala role requires the complete optional outbound trust-policy environment")
if (
    template.count(image_marker) != 2
    or template.count(profile_marker) != 1
    or template.count(public_peer_profile_marker) != 1
    or template.count(private_network_binding) != 1
    or template.count(private_network_definition) != 1
    or template.count(onion_relay_public_env) != 1
    or template.count(public_reverse_queue_marker) != 1
    or template.count(onion_relay_private_env) != 1
    or template.count(private_visibility_env) != 1
    or template.count(private_endpoint_env) != 1
    or template.count(private_api_env) != 1
    or template.count(private_attestation_env) != 1
    or template.count(recipient_relay_id_optional) != 1
    or template.count(recipient_relay_endpoint_optional) != 1
    or template.count(recipient_recovery_mode_env) != 1
    or template.count(public_ingress_marker) != 1
    or template.count(public_endpoint_marker) != 1
    or template.count(public_discovery_marker) != 1
    or template.count(public_api_marker) != 1
    or template.count(public_attestation_marker) != 1
    or template.count(public_seeds_marker) != 1
    or not private_separator
    or public_service.count(public_seeds_marker) != 1
    or "AERONYX_DISCOVERY_SEED_ENDPOINTS:" in public_service
    or private_service.count("      AERONYX_DISCOVERY_SEED_ENDPOINTS:") != 1
    or private_service.count(private_seeds_env) != 1
    or template.count(public_dstack_marker) != 1
):
    raise SystemExit("Phala compose role, network, private API, attestation, or onion relay markers do not match the expected profile")
if "--public-peer" in requested_modes or "--private-recipient" not in requested_modes:
    template = template.replace(public_peer_profile_marker, "")
else:
    # [PHALA-PRIVATE-ONLY-COMPOSE 2026-10-06 by Codex] Do not start an
    # unadvertised public identity in a private-recipient-only deployment.
    template = template.replace(public_peer_profile_marker, public_peer_profile)
if "--public-peer" in requested_modes:
    template = template.replace(public_ingress_marker, public_ingress_mapping)
    if public_endpoint:
        template = template.replace(public_endpoint_marker, public_peer_endpoint_env)
    else:
        # [PHALA-ENDPOINT-TWO-PHASE-BOOTSTRAP 2026-10-06 by Codex]
        # First deployment obtains the Phala origin; quote transport is enabled
        # only on a later render where the signed endpoint is known.
        template = template.replace(public_endpoint_marker, public_peer_endpoint_unassigned_env)
    template = template.replace(public_discovery_marker, public_peer_visibility_env)
    template = template.replace(public_api_marker, public_peer_api_env)
    template = template.replace(
        public_attestation_marker,
        public_peer_attestation_env if public_endpoint else private_attestation_env,
    )
    template = template.replace(public_seeds_marker, public_peer_seeds_env, 1)
    template = template.replace(public_dstack_marker, public_dstack_mount if public_endpoint else "")
    template = template.replace(public_reverse_queue_marker, public_reverse_queue_env)
else:
    template = template.replace(public_ingress_marker, "")
    template = template.replace(public_endpoint_marker, private_endpoint_env)
    template = template.replace(public_discovery_marker, private_visibility_env)
    template = template.replace(public_api_marker, private_api_env)
    template = template.replace(public_attestation_marker, private_attestation_env)
    template = template.replace(public_seeds_marker, private_seeds_env, 1)
    template = template.replace(public_dstack_marker, "")
    template = template.replace(public_reverse_queue_marker, private_reverse_queue_env)

if "--private-recipient" in requested_modes:
    template = template.replace(profile_marker, "")
    template = template.replace(recipient_relay_id_optional, recipient_relay_id_required)
    template = template.replace(recipient_relay_endpoint_optional, recipient_relay_endpoint_required)

public_service, _, private_service = template.partition("  aeronyx-private-recipient:\n")
private_service = private_service.partition("\nvolumes:\n")[0]
if not all(phala_policy_environment_matches(service) for service in (public_service, private_service)):
    raise SystemExit("rendered Phala route trust policy lost its process boundary")
# [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Every rendered role owns
# exactly one seed entry; public interpolation must never reach the recipient.
# [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Count YAML keys, not the
# same name inside ${NAME:?} or ${NAME:-} interpolation expressions.
expected_public_seeds = public_peer_seeds_env if "--public-peer" in requested_modes else private_seeds_env
if (
    public_service.count("      AERONYX_DISCOVERY_SEED_ENDPOINTS:") != 1
    or public_service.count(expected_public_seeds) != 1
    or public_seeds_marker in template
):
    raise SystemExit("public peer discovery seeds do not match the selected role")
# [PHALA-PRIVATE-SEED-RENDER-GUARD 2026-10-06 by Codex] The explicit empty
# override is safe and prevents inherited seed configuration; reject only a
# missing, duplicated, or non-empty seed entry in the rendered private role.
if (
    "    ports:\n" in private_service
    or "source: /var/run/dstack.sock" in private_service
    or private_service.count("      AERONYX_DISCOVERY_SEED_ENDPOINTS:") != 1
    or private_service.count(private_seeds_env) != 1
    or recipient_recovery_mode_env not in private_service
    or "${AERONYX_PHALA_REVERSE_ONION_QUEUE_" in private_service
    or 'AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED: "false"' not in private_service
    or 'AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY: "false"' not in private_service
):
    raise SystemExit("private recipient must not publish ingress, mount dstack, or inherit public seeds")

sys.stdout.write(template.replace(image_marker, image))
PY
