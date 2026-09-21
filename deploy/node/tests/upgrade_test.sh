#!/usr/bin/env bash
# ============================================
# File: deploy/node/tests/upgrade_test.sh
# ============================================
# Creation Reason:
# - Deterministically exercise the release-manifest, health-authority, and
#   rollback-backup gates without opening sockets or touching systemd.
#
# [DEPLOY-MANIFEST-AUTHORITY 2026-09-21 by Codex] The fixture uses only a
# private temporary directory and shell-function fakes. It never invokes a
# live service, release build, deployment, or health endpoint.
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPGRADE_SCRIPT="${REPO_DIR}/deploy/node/upgrade.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/aeronyx-upgrade-test.XXXXXX")"

trap 'rm -rf -- "${TEST_ROOT}"' EXIT

PASS_COUNT=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$*"
}

assert_eq() {
    local actual="$1"
    local expected="$2"
    local label="$3"
    [ "${actual}" = "${expected}" ] \
        || fail "${label}: expected ${expected}, got ${actual}"
    pass "${label}"
}

assert_file_exists() {
    [ -f "$1" ] || fail "$2"
    pass "$2"
}

assert_fails() {
    local label="$1"
    shift
    if "$@"; then
        fail "${label}: unexpectedly succeeded"
    fi
    pass "${label}"
}

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

COMMIT="$(printf 'a%.0s' {1..40})"
TREE="$(printf 'b%.0s' {1..40})"
BINARY_SHA="$(printf 'c%.0s' {1..64})"
MANIFEST="${TEST_ROOT}/release.manifest"

write_manifest() {
    cat >"${MANIFEST}" <<EOF
aeronyx-release-manifest-v1
commit=${COMMIT}
tree=${TREE}
binary_sha256=${BINARY_SHA}
EOF
}

source "${UPGRADE_SCRIPT}"

test_manifest_acceptance_and_full_commit() {
    write_manifest
    SOURCE_COMMIT="${COMMIT}"
    RELEASE_MANIFEST="${MANIFEST}"
    RELEASE_MANIFEST_SHA256="$(sha256 "${MANIFEST}")"
    EXPECTED_MANIFEST_TREE=""
    EXPECTED_BINARY_SHA256=""
    validate_release_manifest_options
    load_release_manifest
    assert_eq "${EXPECTED_MANIFEST_TREE}" "${TREE}" "manifest records exact tree"
    assert_eq "${EXPECTED_BINARY_SHA256}" "${BINARY_SHA}" "manifest records exact binary SHA"
    assert_eq "$(resolve_build_git_commit)" "${COMMIT}" "build embeds full commit"
}

test_manifest_rejections() {
    write_manifest
    local good_sha
    good_sha="$(sha256 "${MANIFEST}")"

    assert_fails "manifest digest mismatch fails closed" bash -c '
        upgrade="$1"; commit="$2"; manifest="$3"; set --; source "${upgrade}"
        die() { return 1; }
        SOURCE_COMMIT="${commit}"; RELEASE_MANIFEST="${manifest}"; RELEASE_MANIFEST_SHA256="$(printf "0%.0s" {1..64})"
        load_release_manifest
    ' bash "${UPGRADE_SCRIPT}" "${COMMIT}" "${MANIFEST}"

    cat >"${MANIFEST}" <<EOF
aeronyx-release-manifest-v1
commit=${COMMIT}
tree=${TREE}
binary_sha256=${BINARY_SHA}
unknown=value
EOF
    assert_fails "manifest unknown field fails closed" bash -c '
        upgrade="$1"; commit="$2"; manifest="$3"; digest="$4"; set --; source "${upgrade}"
        die() { return 1; }
        SOURCE_COMMIT="${commit}"; RELEASE_MANIFEST="${manifest}"; RELEASE_MANIFEST_SHA256="${digest}"
        load_release_manifest
    ' bash "${UPGRADE_SCRIPT}" "${COMMIT}" "${MANIFEST}" "$(sha256 "${MANIFEST}")"

    write_manifest
    assert_fails "manifest commit mismatch fails closed" bash -c '
        upgrade="$1"; manifest="$2"; digest="$3"; set --; source "${upgrade}"
        die() { return 1; }
        SOURCE_COMMIT="$(printf "d%.0s" {1..40})"; RELEASE_MANIFEST="${manifest}"; RELEASE_MANIFEST_SHA256="${digest}"
        load_release_manifest
    ' bash "${UPGRADE_SCRIPT}" "${MANIFEST}" "$(sha256 "${MANIFEST}")"

    RELEASE_MANIFEST=""
    RELEASE_MANIFEST_SHA256=""
    SOURCE_COMMIT=""
    validate_release_manifest_options
    pass "legacy invocation remains valid without strict manifest flags"
    : "${good_sha}"
}

test_manifest_tree_binding() {
    SOURCE_MODE="commit_pinned"
    SOURCE_DIR="${TEST_ROOT}/isolated-source"
    SOURCE_COMMIT="${COMMIT}"
    RELEASE_MANIFEST="${MANIFEST}"
    EXPECTED_MANIFEST_TREE="${TREE}"
    git() { printf '%s\n' "${TREE}"; }
    verify_isolated_source_manifest_tree
    unset -f git
    pass "isolated source tree matches manifest"

    assert_fails "isolated source tree mismatch fails closed" bash -c '
        upgrade="$1"; commit="$2"; expected_tree="$3"; set --; source "${upgrade}"
        die() { return 1; }
        SOURCE_MODE="commit_pinned"; SOURCE_DIR="/unused"; SOURCE_COMMIT="${commit}"
        RELEASE_MANIFEST="/strict"; EXPECTED_MANIFEST_TREE="${expected_tree}"
        git() { printf "%s\\n" "$(printf "f%.0s" {1..40})"; }
        verify_isolated_source_manifest_tree
    ' bash "${UPGRADE_SCRIPT}" "${COMMIT}" "${TREE}"
}

write_config() {
    local name="$1"
    local address="$2"
    local config="${TEST_ROOT}/${name}.toml"
    printf '[memchain]\napi_listen_addr = "%s"\n' "${address}" >"${config}"
    printf '%s\n' "${config}"
}

test_health_authority() {
    local config
    config="$(write_config v4 127.0.0.1:19421)"
    CONFIG_FILE="${config}"
    derive_health_authority
    assert_eq "${HEALTH_URL}" "http://127.0.0.1:19421/api/vpn/health" "IPv4 loopback authority"

    config="$(write_config v6 '[::1]:19422')"
    CONFIG_FILE="${config}"
    derive_health_authority
    assert_eq "${HEALTH_URL}" "http://[::1]:19422/api/vpn/health" "IPv6 loopback authority"

    config="$(write_config wildcard-v4 0.0.0.0:19423)"
    CONFIG_FILE="${config}"
    derive_health_authority
    assert_eq "${HEALTH_URL}" "http://127.0.0.1:19423/api/vpn/health" "IPv4 wildcard maps to loopback"

    config="$(write_config wildcard-v6 '[::]:19424')"
    CONFIG_FILE="${config}"
    derive_health_authority
    assert_eq "${HEALTH_URL}" "http://[::1]:19424/api/vpn/health" "IPv6 wildcard maps to loopback"

    config="${TEST_ROOT}/single-quote.toml"
    printf "[memchain]\\napi_listen_addr = '127.0.0.1:19426'\\n" >"${config}"
    CONFIG_FILE="${config}"
    derive_health_authority
    assert_eq "${HEALTH_URL}" "http://127.0.0.1:19426/api/vpn/health" "single-quoted legacy authority"

    config="$(write_config remote 192.0.2.10:19425)"
    assert_fails "nonloopback health authority fails closed" bash -c '
        upgrade="$1"; config="$2"; set --; source "${upgrade}"
        die() { return 1; }
        CONFIG_FILE="${config}"
        derive_health_authority
    ' bash "${UPGRADE_SCRIPT}" "${config}"

    config="$(write_config port-zero 127.0.0.1:0)"
    assert_fails "zero-port health authority fails closed" bash -c '
        upgrade="$1"; config="$2"; set --; source "${upgrade}"
        die() { return 1; }
        CONFIG_FILE="${config}"
        derive_health_authority
    ' bash "${UPGRADE_SCRIPT}" "${config}"

    config="$(write_config uri 'http://127.0.0.1:19427/?query=1')"
    assert_fails "URI-shaped health authority fails closed" bash -c '
        upgrade="$1"; config="$2"; set --; source "${upgrade}"
        die() { return 1; }
        CONFIG_FILE="${config}"
        derive_health_authority
    ' bash "${UPGRADE_SCRIPT}" "${config}"
}

test_health_payload_and_session_force() {
    printf '{"status":"ok","runtime":{"git_commit":"%s"}}\n' "${COMMIT}" \
        | health_payload_is_acceptable "${COMMIT}"
    pass "health payload binds exact full runtime commit"
    if printf '{"status":"ok","runtime":{"git_commit":"wrong"}}\n' \
        | health_payload_is_acceptable "${COMMIT}"; then
        fail "foreign status=ok response bypassed runtime commit proof"
    fi
    pass "foreign status=ok response is rejected"

    active_sessions() { printf '2'; }
    NO_RESTART=0
    FORCE=0
    if restart_session_gate_passes; then
        fail "active sessions bypassed default restart gate"
    fi
    pass "default restart gate blocks active sessions"
    FORCE=1
    restart_session_gate_passes
    pass "explicit force preserves approved active-session bypass"
    FORCE=0
}

test_git_ssh_authority() {
    local authority captured_authority captured_args visible
    authority="ssh -i /operator/deploy-key -o IdentitiesOnly=yes -o BatchMode=yes"
    GIT_SSH_AUTHORITY="${authority}"
    GIT_SSH_COMMAND=""
    captured_authority=""
    captured_args=""
    git() {
        captured_authority="${GIT_SSH_COMMAND:-}"
        captured_args="$*"
    }
    run_git_for_origin "git@source.example:organization/project.git" clone --no-checkout source destination
    [ "${captured_authority}" = "${authority}" ] \
        || fail "configured SSH authority was not propagated to the isolated clone"
    [ "${captured_args}" = "clone --no-checkout source destination" ] \
        || fail "isolated clone arguments changed while adding SSH authority"
    pass "configured SSH authority is propagated to SSH origin"
    unset -f git

    git() { :; }
    visible="$(run_git_for_origin "git@source.example:organization/project.git" fetch origin main 2>&1)"
    [ -z "${visible}" ] || fail "Git SSH authority leaked through deploy output"
    unset -f git
    pass "Git SSH authority is not logged"

    GIT_SSH_AUTHORITY=""
    GIT_SSH_COMMAND=""
    captured_authority="unset"
    git() { captured_authority="${GIT_SSH_COMMAND-unset}"; }
    run_git_for_origin "https://source.example/organization/project.git" fetch origin main
    [ "${captured_authority}" = "" ] \
        || fail "HTTPS origin unexpectedly received an SSH authority"
    unset -f git
    pass "legacy HTTPS origin retains keyless transport"

    assert_fails "SSH origin without authority fails closed" bash -c '
        upgrade="$1"; set --; source "${upgrade}"
        die() { return 1; }
        GIT_SSH_AUTHORITY=""
        run_git_for_origin "git@source.example:organization/project.git" fetch origin main
    ' bash "${UPGRADE_SCRIPT}"

    assert_fails "non-SSH authority command fails closed" bash -c '
        upgrade="$1"; set --; source "${upgrade}"
        die() { return 1; }
        GIT_SSH_AUTHORITY="custom-wrapper --key /operator/deploy-key"
        run_git_for_origin "git@source.example:organization/project.git" fetch origin main
    ' bash "${UPGRADE_SCRIPT}"
}

test_candidate_and_backup_protection() {
    local candidate="${TEST_ROOT}/candidate"
    local candidate_sha first_backup second_backup protected
    printf 'candidate-bytes' >"${candidate}"
    candidate_sha="$(sha256 "${candidate}")"
    RELEASE_MANIFEST="${MANIFEST}"
    EXPECTED_BINARY_SHA256="${candidate_sha}"
    BUILD_BINARY_SHA256="${candidate_sha}"
    verify_manifest_candidate_binary
    pass "candidate SHA matches manifest before promotion"

    if ( BUILD_BINARY_SHA256="$(printf 'e%.0s' {1..64})"; verify_manifest_candidate_binary ); then
        fail "candidate SHA mismatch bypassed manifest gate"
    fi
    pass "candidate SHA mismatch fails closed"

    REPO_DIR="${TEST_ROOT}/repo"
    RELEASE_DIR="${TEST_ROOT}/releases"
    mkdir -p "${REPO_DIR}/target/release"
    cp "${candidate}" "${REPO_DIR}/target/release/aeronyx-server"
    DRY_RUN=0
    systemctl() { return 1; }
    BACKUP_BINARY=""
    backup_current_binary
    first_backup="${BACKUP_BINARY}"
    backup_current_binary
    second_backup="${BACKUP_BINARY}"
    [ "${first_backup}" != "${second_backup}" ] \
        || fail "same-second backup allocation reused a path"
    assert_file_exists "${first_backup}" "first no-clobber rollback backup retained"
    assert_file_exists "${second_backup}" "second no-clobber rollback backup retained"

    RELEASE_MANIFEST="${MANIFEST}"
    EXPECTED_BINARY_SHA256="${candidate_sha}"
    systemctl() { printf '7\n'; }
    mapped_executable_path() { printf '%s\n' "${candidate}"; }
    verify_promoted_runtime
    pass "stable and mapped binary SHA proof matches manifest"
    mapped_executable_path() { printf '%s\n' "${TEST_ROOT}/missing-mapped-binary"; }
    if verify_promoted_runtime; then
        fail "mapped binary SHA mismatch bypassed manifest proof"
    fi
    pass "mapped binary SHA mismatch fails closed"

    protected="${RELEASE_DIR}/aeronyx-server.00000000000000.protected"
    printf protected >"${protected}"
    printf old-a >"${RELEASE_DIR}/aeronyx-server.00000000000000.old-a"
    printf old-b >"${RELEASE_DIR}/aeronyx-server.00000000000000.old-b"
    touch -t 202401010000 "${protected}" "${RELEASE_DIR}/aeronyx-server.00000000000000.old-a" "${RELEASE_DIR}/aeronyx-server.00000000000000.old-b"
    KEEP_RELEASES=1
    BACKUP_BINARY="${protected}"
    BACKUP_SERVICE_FILE=""
    BACKUP_NETWORK_RESTORE_FILE=""
    # GNU find's -printf is used on deployment hosts. Feed tied timestamps
    # through the real sort/awk/prune path here so this portable fixture proves
    # that the protected current backup is skipped if selected for deletion.
    find() {
        local pattern=""
        local previous=""
        local argument
        for argument in "$@"; do
            if [ "${previous}" = "-name" ]; then
                pattern="${argument}"
                break
            fi
            previous="${argument}"
        done
        [ "${pattern}" = "aeronyx-server.[0-9]*" ] || return 0
        printf '1.0 %s\n' "${protected}"
        printf '1.0 %s\n' "${RELEASE_DIR}/aeronyx-server.00000000000000.old-a"
        printf '1.0 %s\n' "${RELEASE_DIR}/aeronyx-server.00000000000000.old-b"
    }
    prune_release_backups
    unset -f find
    assert_file_exists "${protected}" "current rollback backup survives tied retention"
    if [ -f "${RELEASE_DIR}/aeronyx-server.00000000000000.old-a" ] \
        && [ -f "${RELEASE_DIR}/aeronyx-server.00000000000000.old-b" ]; then
        fail "tied retention fixture did not exercise old-backup pruning"
    fi
    pass "tied retention prunes only non-protected backups"

    NO_RESTART=1
    restart_service
    pass "no-restart remains staged-only without a service action"
}

test_manifest_acceptance_and_full_commit
test_manifest_rejections
test_manifest_tree_binding
test_health_authority
test_health_payload_and_session_force
test_git_ssh_authority
test_candidate_and_backup_protection

printf 'PASS: %s deployment manifest authority checks\n' "${PASS_COUNT}"
