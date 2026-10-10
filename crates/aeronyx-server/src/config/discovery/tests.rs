// ============================================
// File: crates/aeronyx-server/src/config/discovery/tests.rs
// ============================================
//! # Tests: discovery configuration
//!
//! Discovery TOML parsing, default, and decoded-pin accessor tests,
//! moved from the former `config::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `config.rs`; bodies unchanged.

use super::*;

use crate::config::ServerConfig;

#[test]
fn test_discovery_backward_compat_default_disabled() {
    let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
    let config = ServerConfig::from_str(toml_str).unwrap();
    assert!(!config.discovery.enabled);
    assert!(config.discovery.bootstrap_snapshot_path.is_none());
    assert!(config.discovery.bootstrap_snapshot_url.is_none());
    assert!(config.discovery.directory_chain_path.is_none());
    assert!(config
        .discovery
        .directory_chain_sync_peer_node_ids
        .is_empty());
    assert_eq!(
        config.discovery.directory_chain_sync_interval_secs,
        DiscoveryConfig::default_directory_chain_sync_interval_secs()
    );
    assert!(config.validate().is_ok());
}

#[test]
fn test_discovery_bootstrap_toml_parse() {
    let toml_str = r#"
[discovery]
enabled = true
bootstrap_snapshot_path = "/etc/aeronyx/bootstrap-peers.json"
bootstrap_snapshot_url = "https://nodes.aeronyx.network/bootstrap.json"
seed_endpoints = ["http://34.136.167.59:8422", "8.213.146.244:8422"]
fetch_timeout_secs = 15
peer_cache_path = "/var/lib/aeronyx/peers-cache.json"
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
directory_chain_sync_peer_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
directory_chain_sync_interval_secs = 180
directory_gossip_proof_min_age_secs = 360
directory_full_node_mirror_enabled = true
advertise_directory_mirror_carrier = false
directory_full_node_mirror_max_producers = 24
directory_observation_witness_min_verified = 1
peer_cache_write_interval_secs = 120
gossip_enabled = true
gossip_interval_secs = 45
gossip_peer_limit = 8
gossip_jitter_percent = 15
gossip_backpressure_failure_threshold = 4
gossip_failure_backoff_max_secs = 180
max_peers = 512
max_snapshot_limit = 64
gossip_rate_limit_per_minute = 30
allowed_peer_ids = ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
denied_peer_ids = ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
public_endpoint = "node.example.com:443"
public_api_listen_addr = "0.0.0.0:8422"
region = "us-central"
descriptor_ttl_secs = 7200
public_discovery = false
advertise_onion_middle = true
"#;
    let config = ServerConfig::from_str(toml_str).unwrap();
    assert!(config.discovery.enabled);
    assert!(config.discovery.advertise_self);
    assert_eq!(
        config.discovery.bootstrap_snapshot_path.as_deref(),
        Some("/etc/aeronyx/bootstrap-peers.json")
    );
    assert_eq!(
        config.discovery.bootstrap_snapshot_url.as_deref(),
        Some("https://nodes.aeronyx.network/bootstrap.json")
    );
    assert_eq!(
        config.discovery.seed_endpoints,
        vec![
            "http://34.136.167.59:8422".to_string(),
            "8.213.146.244:8422".to_string()
        ]
    );
    assert_eq!(config.discovery.fetch_timeout_secs, 15);
    assert_eq!(
        config.discovery.peer_cache_path.as_deref(),
        Some("/var/lib/aeronyx/peers-cache.json")
    );
    assert_eq!(
        config.discovery.directory_chain_path.as_deref(),
        Some("/var/lib/aeronyx/directory-chain.db")
    );
    assert_eq!(
        config.discovery.directory_chain_sync_peer_node_id_bytes(),
        vec![[0xcc; 32]]
    );
    assert_eq!(config.discovery.directory_chain_sync_interval_secs, 180);
    assert_eq!(
        config.discovery.directory_gossip_proof_min_age_secs,
        Some(360)
    );
    assert_eq!(
        config
            .discovery
            .effective_directory_gossip_proof_min_age_secs(),
        360
    );
    assert!(config.discovery.directory_full_node_mirror_enabled);
    assert!(!config.discovery.advertise_directory_mirror_carrier);
    assert_eq!(
        config.discovery.directory_full_node_mirror_max_producers,
        24
    );
    assert_eq!(
        config.discovery.directory_observation_witness_min_verified,
        1
    );
    assert_eq!(config.discovery.peer_cache_write_interval_secs, 120);
    assert!(config.discovery.gossip_enabled);
    assert_eq!(config.discovery.gossip_interval_secs, 45);
    assert_eq!(config.discovery.gossip_peer_limit, 8);
    assert_eq!(config.discovery.gossip_jitter_percent, 15);
    assert_eq!(config.discovery.gossip_backpressure_failure_threshold, 4);
    assert_eq!(config.discovery.gossip_failure_backoff_max_secs, 180);
    assert_eq!(config.discovery.max_peers, 512);
    assert_eq!(config.discovery.max_snapshot_limit, 64);
    assert_eq!(config.discovery.gossip_rate_limit_per_minute, 30);
    assert_eq!(
        config.discovery.allowed_peer_ids,
        vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
    );
    assert_eq!(
        config.discovery.denied_peer_ids,
        vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
    );
    assert_eq!(
        config.discovery.public_endpoint.as_deref(),
        Some("node.example.com:443")
    );
    assert_eq!(
        config.discovery.public_api_listen_addr,
        Some("0.0.0.0:8422".parse().unwrap())
    );
    assert_eq!(config.discovery.region.as_deref(), Some("us-central"));
    assert_eq!(config.discovery.descriptor_ttl_secs, 7200);
    assert!(!config.discovery.public_discovery);
    assert!(config.discovery.advertise_onion_middle);
}

#[test]
fn test_verified_delivery_witness_policy_parses_and_decodes_pins() {
    let toml_str = r#"
[memchain]
mode = "local"
db_path = "/var/lib/aeronyx/memchain.db"

[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
peer_cache_path = "/var/lib/aeronyx/peers-cache.json"
verified_delivery_witness_node_ids = [
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
]
verified_delivery_witness_requester_node_ids = [
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
]
custody_audit_witness_requester_node_ids = [
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
]
custody_audit_witness_node_ids = [
  "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
]
custody_audit_witness_min_verified = 2
custody_audit_witness_startup_required = true
custody_audit_witness_runtime_required = true
custody_audit_witness_auto_renewal_enabled = true
custody_audit_witness_max_age_secs = 3600
verified_delivery_witness_min_verified = 2
verified_delivery_witness_required_for_restore = true
"#;
    let config = ServerConfig::from_str(toml_str).unwrap();
    assert_eq!(
        config.discovery.verified_delivery_witness_node_id_bytes(),
        vec![[0xAA; 32], [0xBB; 32]]
    );
    assert_eq!(
        config
            .discovery
            .verified_delivery_witness_requester_node_id_bytes(),
        vec![[0xCC; 32]]
    );
    assert_eq!(
        config
            .discovery
            .custody_audit_witness_requester_node_id_bytes(),
        vec![[0xDD; 32]]
    );
    assert_eq!(
        config.discovery.custody_audit_witness_node_id_bytes(),
        vec![[0xEE; 32], [0xFF; 32]]
    );
    assert_eq!(config.discovery.custody_audit_witness_min_verified, 2);
    assert!(config.discovery.custody_audit_witness_startup_required);
    assert!(config.discovery.custody_audit_witness_runtime_required);
    assert!(config.discovery.custody_audit_witness_auto_renewal_enabled);
    assert_eq!(config.discovery.custody_audit_witness_max_age_secs, 3600);
    assert_eq!(config.discovery.verified_delivery_witness_min_verified, 2);
    assert!(
        config
            .discovery
            .verified_delivery_witness_required_for_restore
    );
    assert!(config
        .discovery
        .validate_runtime_identity(&[0xAB; 32])
        .is_ok());
    assert!(config
        .discovery
        .validate_runtime_identity(&[0xEE; 32])
        .is_err());
}

#[test]
fn test_discovery_accepts_opaque_pinned_route_domains() {
    let first_node = "11".repeat(32);
    let second_node = "22".repeat(32);
    let toml_str = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
require_pinned_route_domains_for_multi_hop = true
pinned_route_domains = {{ "{first_node}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "{second_node}" = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }}
"#
    );

    let config = ServerConfig::from_str(&toml_str).unwrap();
    assert!(config.discovery.require_pinned_route_domains_for_multi_hop);
    assert_eq!(config.discovery.pinned_route_domains.len(), 2);
    let assignments = config.discovery.pinned_route_domain_assignments();
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].node_id, [0x11; 32]);
    assert_eq!(assignments[0].route_domain, [0xaa; 16]);
    assert_eq!(assignments[1].node_id, [0x22; 32]);
    assert_eq!(assignments[1].route_domain, [0xbb; 16]);
}

#[test]
fn test_discovery_accepts_route_domain_attestor_quorum() {
    let subject = "11".repeat(32);
    let attestor_a = "22".repeat(32);
    let attestor_b = "33".repeat(32);
    let toml_str = format!(
        r#"
[discovery]
enabled = true
directory_chain_path = "/var/lib/aeronyx/directory-chain.db"
require_pinned_route_domains_for_multi_hop = true
pinned_route_domains = {{ "{subject}" = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }}
route_domain_attestor_node_ids = ["{attestor_a}", "{attestor_b}"]
route_domain_attestation_min_verified = 2
require_route_domain_attestations_for_multi_hop = true
"#
    );

    let config = ServerConfig::from_str(&toml_str).unwrap();
    assert_eq!(
        config.discovery.route_domain_attestor_node_id_bytes(),
        vec![[0x22; 32], [0x33; 32]]
    );
    assert_eq!(config.discovery.route_domain_attestation_min_verified, 2);
    assert!(
        config
            .discovery
            .require_route_domain_attestations_for_multi_hop
    );
}
