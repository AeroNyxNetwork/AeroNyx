// ============================================
// File: crates/aeronyx-server/src/config/discovery/validation/tests.rs
// ============================================
//! # Tests: discovery configuration validation
//!
//! Fail-closed discovery validation tests (witness, Directory Chain, seed,
//! gossip, safety, endpoint-proof rollout, route-domain policies),
//! moved from the former `config::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `config.rs`; bodies unchanged.

use super::*;

use crate::config::ServerConfig;

#[test]
fn test_verified_delivery_witness_policy_rejects_unsafe_configuration() {
    let duplicate = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
peer_cache_path = "/tmp/peers.json"
verified_delivery_witness_node_ids = [
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
"#;
    assert!(ServerConfig::from_str(duplicate).is_err());

    let disabled_storage = r#"
[memchain]
mode = "off"

[discovery]
enabled = true
peer_cache_path = "/tmp/peers.json"
verified_delivery_witness_node_ids = [
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
"#;
    assert!(ServerConfig::from_str(disabled_storage).is_err());

    // MemChain historically defaults to local mode when the section is
    // omitted. Preserve that backward-compatible deployment behavior.
    let default_local_storage = r#"
[discovery]
enabled = true
peer_cache_path = "/tmp/peers.json"
verified_delivery_witness_node_ids = [
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
"#;
    assert!(ServerConfig::from_str(default_local_storage).is_ok());

    let duplicate_requesters = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
verified_delivery_witness_requester_node_ids = [
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
]
"#;
    assert!(ServerConfig::from_str(duplicate_requesters).is_err());

    let disabled_witness_service = r#"
[memchain]
mode = "off"

[discovery]
enabled = true
verified_delivery_witness_requester_node_ids = [
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
]
"#;
    assert!(ServerConfig::from_str(disabled_witness_service).is_err());

    let duplicate_custody_requesters = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
custody_audit_witness_requester_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
"#;
    assert!(ServerConfig::from_str(duplicate_custody_requesters).is_err());

    let disabled_custody_witness_service = r#"
[memchain]
mode = "off"

[discovery]
enabled = true
custody_audit_witness_requester_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
"#;
    assert!(ServerConfig::from_str(disabled_custody_witness_service).is_err());

    let duplicate_custody_witnesses = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
"#;
    assert!(ServerConfig::from_str(duplicate_custody_witnesses).is_err());

    let impossible_custody_quorum = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
custody_audit_witness_min_verified = 2
"#;
    assert!(ServerConfig::from_str(impossible_custody_quorum).is_err());

    // [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Strict startup
    // cannot be enabled without pins, and freshness is bounded to the
    // same seven-day operational ceiling as explicit receipt import.
    let strict_custody_without_pins = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_startup_required = true
"#;
    assert!(ServerConfig::from_str(strict_custody_without_pins).is_err());

    // [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Runtime strict
    // mode cannot create a post-startup policy stronger than startup.
    let runtime_custody_without_startup = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
custody_audit_witness_runtime_required = true
"#;
    assert!(ServerConfig::from_str(runtime_custody_without_startup).is_err());

    // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Automatic
    // network transmission cannot be enabled behind a weaker local-only
    // runtime policy, even when a valid witness pin exists.
    let renewal_without_runtime_guard = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
custody_audit_witness_startup_required = true
custody_audit_witness_auto_renewal_enabled = true
"#;
    assert!(ServerConfig::from_str(renewal_without_runtime_guard).is_err());

    for invalid_age in [59, MAX_CUSTODY_AUDIT_WITNESS_AGE_SECS + 1] {
        let invalid_freshness = format!(
            r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
custody_audit_witness_max_age_secs = {invalid_age}
"#
        );
        assert!(ServerConfig::from_str(&invalid_freshness).is_err());
    }

    let custody_without_relay = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
]
"#;
    assert!(ServerConfig::from_str(custody_without_relay).is_err());

    let missing_pins = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
peer_cache_path = "/tmp/peers.json"
verified_delivery_witness_required_for_restore = true
"#;
    assert!(ServerConfig::from_str(missing_pins).is_err());

    let impossible_threshold = r#"
[memchain]
mode = "local"

[discovery]
enabled = true
peer_cache_path = "/tmp/peers.json"
verified_delivery_witness_node_ids = [
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
verified_delivery_witness_min_verified = 2
"#;
    assert!(ServerConfig::from_str(impossible_threshold).is_err());
}

#[test]
fn test_discovery_rejects_invalid_url_scheme() {
    let toml_str = r#"
[discovery]
enabled = true
bootstrap_snapshot_url = "file:///tmp/bootstrap.json"
"#;
    assert!(ServerConfig::from_str(toml_str).is_err());
}

#[test]
fn test_discovery_rejects_invalid_directory_chain_path() {
    let disabled = r#"
[discovery]
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
"#;
    assert!(ServerConfig::from_str(disabled).is_err());

    let collides_with_cache = r#"
[discovery]
enabled = true
peer_cache_path = "/var/lib/aeronyx/discovery-state"
directory_chain_path = "/var/lib/aeronyx/discovery-state"
"#;
    assert!(ServerConfig::from_str(collides_with_cache).is_err());

    let collides_with_memchain = r#"
[memchain]
mode = "local"
db_path = "/var/lib/aeronyx/shared.db"

[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/shared.db"
"#;
    assert!(ServerConfig::from_str(collides_with_memchain).is_err());

    let pins_without_store = r#"
[discovery]
enabled = true
directory_chain_sync_peer_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
"#;
    assert!(ServerConfig::from_str(pins_without_store).is_err());

    let duplicate_pins = r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_chain_sync_peer_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
"#;
    assert!(ServerConfig::from_str(duplicate_pins).is_err());

    let zero_witness_threshold = r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_chain_sync_peer_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
directory_observation_witness_min_verified = 0
"#;
    assert!(ServerConfig::from_str(zero_witness_threshold).is_err());

    let witness_threshold_exceeds_pins = r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_chain_sync_peer_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
directory_observation_witness_min_verified = 2
"#;
    assert!(ServerConfig::from_str(witness_threshold_exceeds_pins).is_err());

    let witness_threshold_without_pins = r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_observation_witness_min_verified = 2
"#;
    assert!(ServerConfig::from_str(witness_threshold_without_pins).is_err());

    let mirror_without_store = r#"
[discovery]
enabled = true
directory_full_node_mirror_enabled = true
"#;
    assert!(ServerConfig::from_str(mirror_without_store).is_err());

    let valid_carrier_advertisement = r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_full_node_mirror_enabled = true
advertise_directory_mirror_carrier = true
public_discovery = true
public_endpoint = "https://node.example.com"
public_api_listen_addr = "0.0.0.0:8422"
"#;
    let config = ServerConfig::from_str(valid_carrier_advertisement).unwrap();
    assert!(config.discovery.advertise_directory_mirror_carrier);

    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Never publish a signed
    // capability for a disabled/private/unreachable carrier role.
    for invalid_carrier_advertisement in [
        r#"
[discovery]
enabled = true
advertise_directory_mirror_carrier = true
"#,
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_full_node_mirror_enabled = true
advertise_directory_mirror_carrier = true
public_discovery = false
public_endpoint = "https://node.example.com"
public_api_listen_addr = "0.0.0.0:8422"
"#,
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_full_node_mirror_enabled = true
advertise_directory_mirror_carrier = true
public_discovery = true
"#,
    ] {
        assert!(ServerConfig::from_str(invalid_carrier_advertisement).is_err());
    }

    for invalid_capacity in [0, 65] {
        let mirror_capacity = format!(
            r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_full_node_mirror_enabled = true
directory_full_node_mirror_max_producers = {invalid_capacity}
"#
        );
        assert!(ServerConfig::from_str(&mirror_capacity).is_err());
    }
}

#[test]
fn test_discovery_rejects_invalid_seed_endpoint() {
    let empty_seed = r#"
[discovery]
enabled = true
seed_endpoints = [""]
"#;
    assert!(ServerConfig::from_str(empty_seed).is_err());

    let missing_scheme_or_port = r#"
[discovery]
enabled = true
seed_endpoints = ["node.example.com"]
"#;
    assert!(ServerConfig::from_str(missing_scheme_or_port).is_err());
}

#[test]
fn test_discovery_rejects_zero_timeout() {
    let toml_str = r#"
[discovery]
enabled = true
fetch_timeout_secs = 0
"#;
    assert!(ServerConfig::from_str(toml_str).is_err());
}

#[test]
fn test_discovery_rejects_short_peer_cache_interval() {
    let toml_str = r#"
[discovery]
enabled = true
peer_cache_path = "/var/lib/aeronyx/peers-cache.json"
peer_cache_write_interval_secs = 5
"#;
    assert!(ServerConfig::from_str(toml_str).is_err());
}

#[test]
fn test_discovery_rejects_zero_public_api_port() {
    let toml_str = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:0"
"#;
    assert!(ServerConfig::from_str(toml_str).is_err());
}

#[test]
fn test_permissionless_endpoint_proof_rollout_gate_and_bounds() {
    let legacy = r#"
[discovery]
enabled = true
"#;
    let legacy_config = ServerConfig::from_str(legacy).expect("legacy discovery config");
    assert!(
        !legacy_config
            .discovery
            .permissionless_endpoint_proof_enabled
    );

    let valid = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_proof_enabled = true
permissionless_endpoint_proof_max_entries = 64
permissionless_endpoint_proof_ttl_secs = 30
"#;
    let valid_config = ServerConfig::from_str(valid).expect("valid endpoint proof config");
    assert!(valid_config.discovery.permissionless_endpoint_proof_enabled);

    for invalid in [
        r#"
[discovery]
enabled = false
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_proof_enabled = true
"#,
        r#"
[discovery]
enabled = true
permissionless_endpoint_proof_enabled = true
"#,
        r#"
[discovery]
permissionless_endpoint_proof_max_entries = 0
"#,
        r#"
[discovery]
permissionless_endpoint_proof_max_entries = 65537
"#,
        r#"
[discovery]
permissionless_endpoint_proof_ttl_secs = 0
"#,
        r#"
[discovery]
permissionless_endpoint_proof_ttl_secs = 301
"#,
    ] {
        assert!(ServerConfig::from_str(invalid).is_err());
    }
}

#[test]
fn test_permissionless_endpoint_evidence_is_default_off_and_dependency_bound() {
    let legacy =
        ServerConfig::from_str("[discovery]\nenabled = true\n").expect("legacy discovery config");
    assert!(!legacy.discovery.permissionless_endpoint_evidence_enabled);
    assert!(legacy
        .discovery
        .permissionless_endpoint_evidence_db_path
        .is_empty());

    let valid = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_proof_enabled = true
permissionless_endpoint_evidence_enabled = true
permissionless_endpoint_evidence_db_path = "/var/lib/aeronyx/endpoint-evidence.sqlite3"
permissionless_endpoint_evidence_max_entries = 64
permissionless_endpoint_evidence_ttl_secs = 3600
permissionless_endpoint_evidence_cleanup_batch = 16
"#;
    assert!(ServerConfig::from_str(valid).is_ok());

    for invalid in [
        r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_evidence_enabled = true
permissionless_endpoint_evidence_db_path = "/var/lib/aeronyx/evidence.sqlite3"
"#,
        r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_proof_enabled = true
permissionless_endpoint_evidence_enabled = true
"#,
        "[discovery]\npermissionless_endpoint_evidence_max_entries = 0\n",
        "[discovery]\npermissionless_endpoint_evidence_ttl_secs = 604801\n",
        "[discovery]\npermissionless_endpoint_evidence_cleanup_batch = 4097\n",
    ] {
        assert!(ServerConfig::from_str(invalid).is_err());
    }
}

#[test]
fn test_endpoint_attestation_inbox_is_default_off_and_bounded() {
    let legacy =
        ServerConfig::from_str("[discovery]\nenabled = true\n").expect("legacy discovery config");
    assert!(
        !legacy
            .discovery
            .permissionless_endpoint_attestation_inbox_enabled
    );
    assert!(legacy
        .discovery
        .permissionless_endpoint_attestation_inbox_db_path
        .is_empty());

    let valid = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_attestation_inbox_enabled = true
permissionless_endpoint_attestation_inbox_db_path = "/var/lib/aeronyx/endpoint-attestations.sqlite3"
permissionless_endpoint_attestation_inbox_max_entries = 64
permissionless_endpoint_attestation_inbox_max_bytes = 65536
permissionless_endpoint_attestation_inbox_ttl_secs = 3600
permissionless_endpoint_attestation_inbox_cleanup_batch = 16
"#;
    assert!(ServerConfig::from_str(valid).is_ok());

    for invalid in [
        "[discovery]\nenabled = true\npermissionless_endpoint_attestation_inbox_enabled = true\npermissionless_endpoint_attestation_inbox_db_path = \"/tmp/inbox.sqlite3\"\n",
        "[discovery]\nenabled = true\npublic_api_listen_addr = \"0.0.0.0:8422\"\npermissionless_endpoint_attestation_inbox_enabled = true\n",
        "[discovery]\npermissionless_endpoint_attestation_inbox_max_entries = 0\n",
        "[discovery]\npermissionless_endpoint_attestation_inbox_max_bytes = 288\n",
        "[discovery]\npermissionless_endpoint_attestation_inbox_max_bytes = 67108865\n",
        "[discovery]\npermissionless_endpoint_attestation_inbox_ttl_secs = 604801\n",
        "[discovery]\npermissionless_endpoint_attestation_inbox_cleanup_batch = 4097\n",
    ] {
        assert!(ServerConfig::from_str(invalid).is_err());
    }
}

#[test]
fn test_permissionless_promotion_is_additive_default_off_and_requires_private_gates() {
    let legacy =
        ServerConfig::from_str("[discovery]\nenabled = true\n").expect("legacy discovery config");
    assert!(!legacy.discovery.permissionless_endpoint_promotion_enabled);
    assert!(legacy
        .discovery
        .permissionless_endpoint_promotion_db_prefix
        .is_empty());
    let enabled = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:8422"
permissionless_endpoint_proof_enabled = true
permissionless_endpoint_evidence_enabled = true
permissionless_endpoint_evidence_db_path = "/var/lib/aeronyx/evidence.sqlite3"
permissionless_endpoint_attestation_inbox_enabled = true
permissionless_endpoint_attestation_inbox_db_path = "/var/lib/aeronyx/inbox.sqlite3"
permissionless_endpoint_promotion_enabled = true
permissionless_endpoint_promotion_db_prefix = "/var/lib/aeronyx/promotion"
"#;
    assert!(ServerConfig::from_str(enabled).is_ok());
    let undersized =
        format!("{enabled}\npermissionless_endpoint_attestation_inbox_max_entries = 1\n");
    for invalid in [
        "[discovery]\npermissionless_endpoint_promotion_enabled = true\n",
        "[discovery]\nenabled = true\npublic_api_listen_addr = \"0.0.0.0:8422\"\npermissionless_endpoint_promotion_enabled = true\n",
        undersized.as_str(),
    ] {
        assert!(ServerConfig::from_str(invalid).is_err());
    }
}

#[test]
fn test_discovery_rejects_invalid_gossip_policy() {
    let short_interval = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_interval_secs = 5
"#;
    assert!(ServerConfig::from_str(short_interval).is_err());

    let zero_limit = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_peer_limit = 0
"#;
    assert!(ServerConfig::from_str(zero_limit).is_err());

    let zero_concurrency = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_concurrency_limit = 0
"#;
    assert!(ServerConfig::from_str(zero_concurrency).is_err());

    let excessive_concurrency = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_concurrency_limit = 65
"#;
    assert!(ServerConfig::from_str(excessive_concurrency).is_err());

    let immature_directory_proof = r"
[discovery]
enabled = true
directory_chain_sync_interval_secs = 120
directory_gossip_proof_min_age_secs = 239
";
    assert!(ServerConfig::from_str(immature_directory_proof).is_err());

    let excessive_jitter = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_jitter_percent = 51
"#;
    assert!(ServerConfig::from_str(excessive_jitter).is_err());

    let zero_backpressure_threshold = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_backpressure_failure_threshold = 0
"#;
    assert!(ServerConfig::from_str(zero_backpressure_threshold).is_err());

    let short_backoff_max = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_interval_secs = 60
gossip_failure_backoff_max_secs = 30
"#;
    assert!(ServerConfig::from_str(short_backoff_max).is_err());
}

#[test]
fn test_discovery_rejects_invalid_safety_policy() {
    let zero_max_peers = r#"
[discovery]
enabled = true
max_peers = 0
"#;
    assert!(ServerConfig::from_str(zero_max_peers).is_err());

    let zero_snapshot_limit = r#"
[discovery]
enabled = true
max_snapshot_limit = 0
"#;
    assert!(ServerConfig::from_str(zero_snapshot_limit).is_err());

    let zero_rate_limit = r#"
[discovery]
enabled = true
gossip_rate_limit_per_minute = 0
"#;
    assert!(ServerConfig::from_str(zero_rate_limit).is_err());

    let bad_peer_id = r#"
[discovery]
enabled = true
allowed_peer_ids = ["not-a-node-id"]
"#;
    assert!(ServerConfig::from_str(bad_peer_id).is_err());
}

#[test]
fn test_discovery_rejects_unsafe_pinned_route_domain_policy() {
    let missing_assignments = r#"
[discovery]
enabled = true
require_pinned_route_domains_for_multi_hop = true
"#;
    assert!(ServerConfig::from_str(missing_assignments).is_err());

    let node_id = "11".repeat(32);
    let named_domain = format!(
        r#"
[discovery]
enabled = true
pinned_route_domains = {{ "{node_id}" = "cloud-provider-a" }}
"#
    );
    assert!(ServerConfig::from_str(&named_domain).is_err());

    let zero_domain = format!(
        r#"
[discovery]
enabled = true
pinned_route_domains = {{ "{node_id}" = "00000000000000000000000000000000" }}
"#
    );
    assert!(ServerConfig::from_str(&zero_domain).is_err());

    let disabled_strict_policy = format!(
        r#"
[discovery]
require_pinned_route_domains_for_multi_hop = true
pinned_route_domains = {{ "{node_id}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }}
"#
    );
    assert!(ServerConfig::from_str(&disabled_strict_policy).is_err());

    let missing_policy_history = format!(
        r#"
[discovery]
enabled = true
require_pinned_route_domains_for_multi_hop = true
pinned_route_domains = {{ "{node_id}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }}
"#
    );
    assert!(ServerConfig::from_str(&missing_policy_history).is_err());

    let uppercase_node_id = node_id.to_ascii_uppercase();
    let duplicate_identity = format!(
        r#"
[discovery]
enabled = true
pinned_route_domains = {{ "{node_id}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "{uppercase_node_id}" = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }}
"#
    );
    assert!(ServerConfig::from_str(&duplicate_identity).is_err());
}

#[test]
fn test_discovery_rejects_unsafe_route_domain_attestor_policy() {
    let subject = "11".repeat(32);
    let attestor = "22".repeat(32);
    let missing_strict_gate = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
route_domain_attestor_node_ids = ["{attestor}"]
require_route_domain_attestations_for_multi_hop = true
"#
    );
    assert!(ServerConfig::from_str(&missing_strict_gate).is_err());

    let threshold_exceeds_pins = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
route_domain_attestor_node_ids = ["{attestor}"]
route_domain_attestation_min_verified = 2
"#
    );
    assert!(ServerConfig::from_str(&threshold_exceeds_pins).is_err());

    let uppercase_attestor = attestor.to_ascii_uppercase();
    let duplicate_attestor = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
route_domain_attestor_node_ids = ["{attestor}", "{uppercase_attestor}"]
"#
    );
    assert!(ServerConfig::from_str(&duplicate_attestor).is_err());

    let missing_history = format!(
        r#"
[discovery]
enabled = true
route_domain_attestor_node_ids = ["{attestor}"]
"#
    );
    assert!(ServerConfig::from_str(&missing_history).is_err());

    let overlapping_subject = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
pinned_route_domains = {{ "{subject}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }}
route_domain_attestor_node_ids = ["{subject}"]
"#
    );
    assert!(ServerConfig::from_str(&overlapping_subject).is_err());

    let missing_attestors = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
require_pinned_route_domains_for_multi_hop = true
pinned_route_domains = {{ "{subject}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }}
require_route_domain_attestations_for_multi_hop = true
"#
    );
    assert!(ServerConfig::from_str(&missing_attestors).is_err());
}

#[test]
fn test_discovery_rejects_short_descriptor_ttl() {
    let toml_str = r#"
[discovery]
enabled = true
descriptor_ttl_secs = 10
"#;
    assert!(ServerConfig::from_str(toml_str).is_err());
}
