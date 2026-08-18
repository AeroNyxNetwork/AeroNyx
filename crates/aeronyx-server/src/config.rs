// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration — Entry Layer
//!
//! ## Creation Reason
//! Central configuration entry point for all AeroNyx server subsystems.
//! Loaded from a TOML file at startup.
//!
//! ## Modification Reason
//! v1.1.0-ChatRelay — 🌟 Refactored into multi-file layout:
//!   - config_infra.rs    — NetworkConfig, VpnConfig, TunConfig,
//!                          ServerKeyConfig, LimitsConfig, LoggingConfig
//!   - config_saas.rs     — SaasConfig
//!   - config_chat_relay.rs — ChatRelayConfig
//!   - config_memchain.rs — MemChainConfig, MemChainMode, VectorQuantizationMode
//!   - config_supernode.rs — SuperNodeConfig (pre-existing)
//!   This file is now a thin composition + re-export layer only.
//!   All logic lives in the sub-modules above.
//!
//! ## Main Functionality
//! - `ServerConfig` — top-level struct; owns all sub-configs
//! - `ServerConfig::load(path)` — async TOML load + validate
//! - `ServerConfig::from_str(s)` — sync TOML parse + validate (tests)
//! - `ServerConfig::validate()` — delegates to every sub-config
//! - Re-exports all public types from sub-modules for downstream crates
//!   that `use crate::config::*`
//!
//! ## Dependencies
//! - config_infra.rs    — infrastructure configs (no MemChain awareness)
//! - config_memchain.rs — MemChain + all nested subsystem configs
//! - config_supernode.rs — SuperNodeConfig (re-exported via config_memchain)
//! - config_saas.rs     — SaasConfig (re-exported via config_memchain)
//! - config_chat_relay.rs — ChatRelayConfig (re-exported via config_memchain)
//! - config_blind_vault.rs — BlindVaultConfig (independent node-blind store)
//! - management.rs      — ManagementConfig (owned by that subsystem)
//! - server.rs          — consumes ServerConfig for full initialization
//!
//! ## Main Logical Flow
//! 1. `load(path)` reads TOML file → `toml::from_str` → `ServerConfig`
//! 2. `validate()` calls each sub-config's validate in order:
//!    network → vpn → tun → limits → management → memchain
//! 3. Validated config returned to server.rs for subsystem initialization
//!
//! ⚠️ Important Note for Next Developer:
//! - Do NOT add business logic to this file. It is an orchestration layer only.
//! - All serde defaults on sub-structs ensure any missing TOML section is
//!   backward-compatible (defaults to disabled / safe values).
//! - Adding a new subsystem config: create config_<name>.rs, add a field
//!   to MemChainConfig (or ServerConfig if infra-level), add validate()
//!   delegation, and add a `pub use` here.
//! - Integration tests that span multiple sub-configs belong in this file's
//!   #[cfg(test)] block; unit tests belong in each sub-module's own tests.
//!
//! ## Last Modified
//! v0.25.0-CustodyWitnessRuntimeGuard - Added a default-off, local-only
//! runtime re-audit gate that requires the strict startup gate.
//! v0.24.0-CustodyWitnessStartupGate - Added a default-off, local-only
//! current-anchor receipt threshold and bounded freshness policy for startup
//! v0.23.0-CustodyWitnessReceiptVault - Clarified that explicit durable rounds
//! use producer pins while configuration alone never schedules transmission
//! v0.22.0-CustodyWitnessPlanner - Added independent producer witness pins and
//! a validated local quorum target without enabling outbound transmission
//! v0.21.0-CustodyWitnessAdmission - Added an independent fail-closed producer
//! pin set for custody-audit witness writes
//! v0.20.0-RouteDomainAttestors - Added opt-in pinned attestor quorum and
//! fail-closed certificate requirement for multi-hop route-domain assignments
//! v0.19.0-PinnedRouteDomains - Added optional fail-closed, operator-audited
//! opaque route-domain assignments for multi-hop anti-affinity
//! v0.18.0-DirectoryProofMaturity - Added an automatically safe, operator-
//! configurable minimum age for outbound Directory gossip proofs
//! v0.17.0-DiscoveryGossipIsolation - Added bounded outbound gossip concurrency
//! with a fail-closed operator limit
//! v0.16.0-DirectoryMirrorCarrierCapability - Added a fail-closed staged
//! advertisement gate for signed Directory Mirror carrier capability
//! v0.15.0-FullNodeMirror - Added opt-in bounded non-authoritative Directory mirrors
//! v0.14.0-DirectoryWitnessThreshold — Added a bounded independent checkpoint corroboration threshold
//! v0.13.0-DirectorySyncPins — Added fail-closed Directory Sync peer admission pins
//! v0.12.0-DirectoryChainStore — Added optional fail-closed local directory ledger path
//! v0.11.0-VerifiedDeliveryWitnessAdmission — Added explicit witness requester pins
//! v0.10.0-VerifiedDeliveryWitness — Added optional pinned cache-anchor witnesses
//! v1.4.0-DiscoveryRelayCapabilities — Added explicit onion-middle advertisement gate
//! v1.3.0-TransportCapability — Added VPN transport capability accessors
//! v2.1.0            — Added MemChain config fields
//! v1.2.0-DNSOwnership — Added DNS proxy ownership accessor for server startup
//! v2.1.0+MVF+Auth   — Added api_secret
//! v2.3.0+RemoteStorage — Added allow_remote_storage, max_remote_owners
//! v2.4.0-GraphCognition — Added NER/graph/entropy/miner/vector fields
//! v2.5.0-SuperNode  — Added SuperNode config
//! v1.0.0-MultiTenant — Added SaaS mode
//! v1.1.0-ChatRelay  — 🌟 Split into multi-file layout; this file now thin
//! v0.7.0-DiscoverySafetyStatus — Added nodeboard status and safety policy config
//! v0.6.0-DiscoveryOutboundGossip — Added optional outbound peer gossip config
//! v0.5.0-DiscoveryPeerCache — Added optional local PeerStore cache config
//! v0.4.0-DiscoverySelfDescriptor — Added signed self descriptor config
//! v0.3.0-DiscoveryBootstrap — Added optional discovery bootstrap config
//! v0.9.0-DiscoveryGossipBackpressure — Added outbound gossip jitter/backpressure controls
//! v0.8.0-DiscoveryPublicApi — Added optional public-only discovery listener

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

const MAX_VERIFIED_DELIVERY_WITNESS_NODE_IDS: usize = 3;
const MAX_VERIFIED_DELIVERY_WITNESS_REQUESTER_NODE_IDS: usize = 64;
const MAX_CUSTODY_AUDIT_WITNESS_NODE_IDS: usize = 3;
const MAX_CUSTODY_AUDIT_WITNESS_REQUESTER_NODE_IDS: usize = 64;
const MAX_CUSTODY_AUDIT_WITNESS_AGE_SECS: u64 = 7 * 24 * 60 * 60;
const MAX_DIRECTORY_CHAIN_SYNC_PEER_NODE_IDS: usize = 16;
const MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS: usize = 64;
const MAX_DIRECTORY_GOSSIP_PROOF_MIN_AGE_SECS: u64 = 48 * 60 * 60;
const MAX_DISCOVERY_GOSSIP_CONCURRENCY: u16 = 64;
const MAX_PINNED_ROUTE_DOMAINS: usize = 256;
const MAX_ROUTE_DOMAIN_ATTESTOR_NODE_IDS: usize = 16;

/// Validates one bounded fail-closed node identity pin set.
///
/// [WITNESS-ADMISSION-PINS 2026-08-16 by Codex] Delivery and custody witnesses
/// remain separate policy domains, but must share identical parsing and
/// duplicate rejection so one endpoint cannot accidentally become weaker.
fn validate_node_id_pin_set(
    field: &'static str,
    configured: &[String],
    max_entries: usize,
) -> Result<Vec<[u8; 32]>> {
    if configured.len() > max_entries {
        return Err(ServerError::config_invalid(
            field,
            format!("supports at most {max_entries} pinned identities"),
        ));
    }
    let mut validated = Vec::<[u8; 32]>::with_capacity(configured.len());
    for configured_id in configured {
        let value = configured_id.trim();
        let decoded = hex::decode(value).map_err(|_| {
            ServerError::config_invalid(
                field,
                "each entry must be a 64-character Ed25519 public key in hexadecimal",
            )
        })?;
        let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
            ServerError::config_invalid(field, "each entry must decode to exactly 32 bytes")
        })?;
        if value.len() != 64 || node_id.iter().all(|byte| *byte == 0) {
            return Err(ServerError::config_invalid(
                field,
                "each entry must be a non-zero 64-character Ed25519 public key",
            ));
        }
        if validated.contains(&node_id) {
            return Err(ServerError::config_invalid(
                field,
                "duplicate identities are not allowed",
            ));
        }
        validated.push(node_id);
    }
    Ok(validated)
}

/// One validated, canonical local route-domain assignment.
///
/// [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] The opaque 128-bit
/// token groups nodes that the operator has independently reviewed as sharing
/// one routing failure domain. It is not an AS proof, operator identity,
/// network vote, or Sybil-resistance credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PinnedRouteDomainAssignment {
    pub(crate) node_id: [u8; 32],
    pub(crate) route_domain: [u8; 16],
}

// ── Sub-module re-exports (keep callers' use-paths stable) ────────────────
pub use crate::config_blind_vault::BlindVaultConfig;
pub use crate::config_chat_relay::ChatRelayConfig;
pub use crate::config_infra::{
    LimitsConfig, LoggingConfig, NetworkConfig, ServerKeyConfig, TunConfig, VpnConfig,
    VpnTransportConfig,
};
pub use crate::config_memchain::{MemChainConfig, MemChainMode, VectorQuantizationMode};
pub use crate::config_saas::SaasConfig;
pub use crate::config_supernode::SuperNodeConfig;

// ============================================
// DiscoveryConfig
// ============================================

/// Configuration for decentralized node discovery bootstrap and self advertisement.
///
/// The bootstrap layer is disabled by default for backward compatibility.
/// When enabled, the node can hydrate its verified in-memory peer store from
/// a local JSON snapshot and/or an HTTPS JSON snapshot URL, then optionally
/// sign and publish its own descriptor into the local peer store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enables bootstrap snapshot loading at server startup.
    #[serde(default)]
    pub enabled: bool,
    /// Enables generating this node's signed descriptor at startup.
    #[serde(default = "DiscoveryConfig::default_advertise_self")]
    pub advertise_self: bool,
    /// Optional local JSON bootstrap snapshot path.
    #[serde(default)]
    pub bootstrap_snapshot_path: Option<String>,
    /// Optional HTTP(S) JSON bootstrap snapshot URL.
    #[serde(default)]
    pub bootstrap_snapshot_url: Option<String>,
    /// Optional public discovery endpoints contacted on every gossip round.
    ///
    /// These seed endpoints are not trusted authorities; they only provide
    /// signed discovery gossip/snapshot transport so nodes can recover when a
    /// cached peer descriptor has an outdated public endpoint.
    #[serde(default)]
    pub seed_endpoints: Vec<String>,
    /// Timeout in seconds for fetching a remote bootstrap snapshot.
    #[serde(default = "DiscoveryConfig::default_fetch_timeout_secs")]
    pub fetch_timeout_secs: u64,
    /// Optional local verified peer cache path.
    ///
    /// The cache uses the same JSON schema as bootstrap snapshots and is
    /// re-verified on every load, so stale or tampered descriptors are skipped.
    #[serde(default)]
    pub peer_cache_path: Option<String>,
    /// Optional SQLite path for the signed local Directory Chain journal.
    ///
    /// Configuring this path opts the node into fail-closed startup auditing.
    /// It must never reuse a bootstrap snapshot or mutable peer-cache path.
    #[serde(default)]
    pub directory_chain_path: Option<String>,
    /// Operator-pinned node identities allowed to exchange Directory Sync V1.
    ///
    /// This is deliberately independent of permissionless discovery allow/deny
    /// policy. An empty list keeps tip/block/object peer routes fail-closed.
    #[serde(default)]
    pub directory_chain_sync_peer_node_ids: Vec<String>,
    /// Low-frequency interval for one bounded replica page per pinned peer.
    ///
    /// Empty peer pins disable all outbound Directory Sync regardless of this
    /// value. One page contains one block and bounded object requests so a
    /// normal round remains below the peer API's per-minute request budget.
    #[serde(default = "DiscoveryConfig::default_directory_chain_sync_interval_secs")]
    pub directory_chain_sync_interval_secs: u64,
    /// Optional minimum age of a Directory block before its proof is gossiped.
    ///
    /// [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] `None` derives a safe
    /// value of two replica-sync intervals. An explicit value must be at least
    /// that derived floor so proof publication cannot outrun exact-anchor
    /// convergence on healthy peers. Legacy descriptor gossip is unaffected.
    #[serde(default)]
    pub directory_gossip_proof_min_age_secs: Option<u64>,
    /// Enables bounded, non-authoritative mirroring from verified public peers.
    ///
    /// Mirror producers are selected from permissionless signed discovery, but
    /// their replicas never participate in configured observation checkpoints,
    /// witness thresholds, policy anchors, fork choice, consensus, or finality.
    /// This remains disabled by default for backward compatibility.
    #[serde(default)]
    pub directory_full_node_mirror_enabled: bool,
    /// Publishes the signed `DirectoryMirrorCarrier` descriptor capability.
    ///
    /// [MIRROR-CAPABILITY 2026-07-24 by Codex] This staged rollout gate stays
    /// disabled by default because older binaries cannot decode a newly
    /// appended capability enum variant. Enable it only after the peer fleet
    /// has upgraded, and only on a public, routeable Full-node Mirror.
    #[serde(default)]
    pub advertise_directory_mirror_carrier: bool,
    /// Maximum distinct permissionless producer namespaces retained as mirrors.
    ///
    /// The durable admission registry enforces this ceiling before importing a
    /// first page, preventing descriptor churn from creating unbounded replica
    /// namespaces. Operator-pinned producers do not consume mirror capacity.
    #[serde(default = "DiscoveryConfig::default_directory_full_node_mirror_max_producers")]
    pub directory_full_node_mirror_max_producers: usize,
    /// Minimum independent accepted receipts required for a local observation
    /// checkpoint to satisfy the external corroboration target.
    ///
    /// This is an evidence threshold only. It does not assign voting weight,
    /// select forks, establish consensus, or grant finality. The default of one
    /// preserves the original witness behavior for existing configurations.
    #[serde(default = "DiscoveryConfig::default_directory_observation_witness_min_verified")]
    pub directory_observation_witness_min_verified: usize,
    /// Operator-pinned nodes that witness the signed delivery-cache generation.
    ///
    /// Witnesses receive only this node's identity, a monotonic generation,
    /// and an opaque digest. Delivery counts, timestamps, routes, message ids,
    /// payloads, endpoints, and client metadata are never sent.
    #[serde(default)]
    pub verified_delivery_witness_node_ids: Vec<String>,
    /// Requester identities this node explicitly agrees to witness.
    ///
    /// This is independent of the permissionless discovery allow/deny policy.
    /// An empty list keeps the witness endpoint fail-closed while preserving
    /// ordinary descriptor discovery and encrypted relay participation.
    #[serde(default)]
    pub verified_delivery_witness_requester_node_ids: Vec<String>,
    /// Operator-pinned independent nodes considered for custody witnessing.
    ///
    /// [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] This producer-side
    /// list feeds local eligibility planning and the explicit durable witness
    /// transport primitive. Merely configuring candidates still never starts
    /// a scheduler or transmits an anchor; callers must invoke a bounded round.
    #[serde(default)]
    pub custody_audit_witness_node_ids: Vec<String>,
    /// Minimum independently eligible custody witnesses required by policy.
    #[serde(default = "DiscoveryConfig::default_custody_audit_witness_min_verified")]
    pub custody_audit_witness_min_verified: usize,
    /// Requires fresh durable receipts for the current custody anchor at startup.
    ///
    /// [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] This gate is
    /// deliberately local-only: it re-audits receipts already stored in the
    /// node's `MemChain` database and never contacts a witness during startup.
    #[serde(default)]
    pub custody_audit_witness_startup_required: bool,
    /// Keeps re-auditing current-anchor receipt readiness while running.
    ///
    /// [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] This is separately
    /// default-off for backward compatibility, never contacts witnesses, and
    /// may be enabled only together with the strict startup gate.
    #[serde(default)]
    pub custody_audit_witness_runtime_required: bool,
    /// Maximum age of a signed receipt accepted by strict local policy.
    #[serde(default = "DiscoveryConfig::default_custody_audit_witness_max_age_secs")]
    pub custody_audit_witness_max_age_secs: u64,
    /// Producer identities this node explicitly agrees to witness for custody.
    ///
    /// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] These pins are separate
    /// from delivery witnesses and permissionless discovery. An empty list
    /// keeps custody witness writes fail-closed without affecting relay or
    /// descriptor participation.
    #[serde(default)]
    pub custody_audit_witness_requester_node_ids: Vec<String>,
    /// Minimum valid signed witness responses required to protect a generation.
    #[serde(default = "DiscoveryConfig::default_verified_delivery_witness_min_verified")]
    pub verified_delivery_witness_min_verified: usize,
    /// Clears restored delivery evidence when the external threshold is absent.
    ///
    /// This remains default-off for backward compatibility. Signed stale,
    /// conflict, or generation-gap evidence always fails closed regardless of
    /// this availability policy.
    #[serde(default)]
    pub verified_delivery_witness_required_for_restore: bool,
    /// Periodic cache write interval in seconds.
    #[serde(default = "DiscoveryConfig::default_peer_cache_write_interval_secs")]
    pub peer_cache_write_interval_secs: u64,
    /// Enables periodic outbound discovery gossip to known public peers.
    ///
    /// Kept disabled by default so simply enabling bootstrap does not create
    /// unexpected outbound network traffic.
    #[serde(default)]
    pub gossip_enabled: bool,
    /// Periodic outbound gossip interval in seconds.
    #[serde(default = "DiscoveryConfig::default_gossip_interval_secs")]
    pub gossip_interval_secs: u64,
    /// Maximum number of public peers contacted per gossip round.
    #[serde(default = "DiscoveryConfig::default_gossip_peer_limit")]
    pub gossip_peer_limit: u16,
    /// Maximum number of peer gossip exchanges polled concurrently.
    ///
    /// [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] This bounds outbound
    /// sockets and memory while preventing one slow peer from serially blocking
    /// every later peer in the round. Runtime fan-out is additionally capped by
    /// `gossip_peer_limit` and the number of selected peers.
    #[serde(default = "DiscoveryConfig::default_gossip_concurrency_limit")]
    pub gossip_concurrency_limit: u16,
    /// Percent of `gossip_interval_secs` used as per-node scheduling jitter.
    ///
    /// This keeps a fleet of nodes from contacting seeds at the exact same
    /// second after restart or after a temporary network incident.
    #[serde(default = "DiscoveryConfig::default_gossip_jitter_percent")]
    pub gossip_jitter_percent: u8,
    /// Consecutive failed outbound gossip rounds before seed-only backpressure.
    ///
    /// Backpressure does not disable discovery. It temporarily reduces fanout to
    /// configured seed endpoints so a failing node does not amplify errors by
    /// retrying every stale peer endpoint.
    #[serde(default = "DiscoveryConfig::default_gossip_backpressure_failure_threshold")]
    pub gossip_backpressure_failure_threshold: u64,
    /// Maximum delay between outbound gossip attempts while backpressure is active.
    #[serde(default = "DiscoveryConfig::default_gossip_failure_backoff_max_secs")]
    pub gossip_failure_backoff_max_secs: u64,
    /// Maximum descriptors retained in the local verified peer store.
    #[serde(default = "DiscoveryConfig::default_max_peers")]
    pub max_peers: usize,
    /// Maximum descriptors returned by a single snapshot response.
    #[serde(default = "DiscoveryConfig::default_max_snapshot_limit")]
    pub max_snapshot_limit: usize,
    /// Global inbound gossip request budget per minute.
    #[serde(default = "DiscoveryConfig::default_gossip_rate_limit_per_minute")]
    pub gossip_rate_limit_per_minute: u32,
    /// Optional allow-list of peer node ids as lowercase/uppercase hex.
    #[serde(default)]
    pub allowed_peer_ids: Vec<String>,
    /// Optional deny-list of peer node ids as lowercase/uppercase hex.
    #[serde(default)]
    pub denied_peer_ids: Vec<String>,
    /// Operator-audited opaque route-domain pins keyed by node id.
    ///
    /// [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Each key is one 32-byte
    /// Ed25519 node id in hexadecimal. Each value is a random 128-bit domain
    /// token encoded as 32 hexadecimal characters. Nodes assigned the same
    /// token are treated as one administrative/routing failure domain during
    /// multi-hop admission. Tokens stay local and must not contain operator,
    /// provider, geography, or ownership names.
    ///
    /// These pins are reviewed local policy, not peer self-attestation,
    /// permissionless consensus, autonomous-system proof, or Sybil resistance.
    #[serde(default)]
    pub pinned_route_domains: BTreeMap<String, String>,
    /// Requires complete pinned route-domain coverage for every multi-hop path.
    ///
    /// Default `false` preserves mixed-version behavior. When enabled, the
    /// local entry and every remote hop must have a pin, and no two hops may
    /// share a token. Missing coverage fails requested multi-hop readiness
    /// closed while leaving single-hop encrypted relay behavior available.
    #[serde(default)]
    pub require_pinned_route_domains_for_multi_hop: bool,
    /// Operator-pinned identities allowed to attest opaque route domains.
    ///
    /// [ROUTE-DOMAIN-ATTESTOR-POLICY 2026-08-03 by Codex] These identities
    /// are local trust anchors, independent of permissionless discovery and
    /// checkpoint witnesses. A signature proves only one opaque assignment;
    /// it does not prove ASN, ownership, geography, honest operation, or Sybil
    /// resistance. Keep this set small, independently reviewed, and private.
    #[serde(default)]
    pub route_domain_attestor_node_ids: Vec<String>,
    /// Minimum currently valid pinned signatures required per assignment.
    #[serde(default = "DiscoveryConfig::default_route_domain_attestation_min_verified")]
    pub route_domain_attestation_min_verified: usize,
    /// Requires quorum-valid route-domain certificates for multi-hop paths.
    ///
    /// Default `false` preserves the existing local-pin behavior. Enabling
    /// this gate also requires strict pinned-domain coverage and a durable
    /// Directory Chain store; an unavailable or expired certificate fails
    /// multi-hop selection closed while single-hop relay remains available.
    #[serde(default)]
    pub require_route_domain_attestations_for_multi_hop: bool,
    /// Optional discovery control-plane endpoint advertised to other nodes.
    ///
    /// When absent, `network.public_endpoint` is reused. If both are absent,
    /// the node still signs a descriptor but leaves endpoint discovery empty.
    #[serde(default)]
    pub public_endpoint: Option<String>,
    /// Optional public-only API listener for discovery and peer chat relay.
    ///
    /// This listener is separate from `memchain.api_listen_addr` and exposes
    /// only `/api/discovery/*` plus `/api/chat/peer/relay`. It stays disabled
    /// by default so existing deployments never expose the full local API.
    #[serde(default)]
    pub public_api_listen_addr: Option<SocketAddr>,
    /// Optional region label for nodeboard and future peer selection.
    #[serde(default)]
    pub region: Option<String>,
    /// Descriptor validity window in seconds.
    #[serde(default = "DiscoveryConfig::default_descriptor_ttl_secs")]
    pub descriptor_ttl_secs: u64,
    /// Whether this node may appear in public bootstrap snapshots.
    #[serde(default = "DiscoveryConfig::default_public_discovery")]
    pub public_discovery: bool,
    /// Whether this node explicitly advertises future no-exit onion middle-hop relay.
    ///
    /// This stays disabled by default because it changes a node's public
    /// routing role. Enabling it only announces node-level relay capability;
    /// payloads remain opaque and are never parsed by the node.
    #[serde(default)]
    pub advertise_onion_middle: bool,
}

impl DiscoveryConfig {
    /// Default self advertisement behavior when discovery is enabled.
    #[must_use]
    pub const fn default_advertise_self() -> bool {
        true
    }

    /// Default remote bootstrap fetch timeout.
    #[must_use]
    pub const fn default_fetch_timeout_secs() -> u64 {
        10
    }

    /// Default local peer cache write interval.
    #[must_use]
    pub const fn default_peer_cache_write_interval_secs() -> u64 {
        300
    }

    /// Default low-frequency Directory Chain replica pull interval.
    #[must_use]
    pub const fn default_directory_chain_sync_interval_secs() -> u64 {
        120
    }

    /// Effective minimum age for outbound Directory gossip proofs.
    #[must_use]
    pub fn effective_directory_gossip_proof_min_age_secs(&self) -> u64 {
        self.directory_gossip_proof_min_age_secs
            .unwrap_or_else(|| self.directory_chain_sync_interval_secs.saturating_mul(2))
    }

    /// Default durable capacity for permissionless non-authoritative mirrors.
    #[must_use]
    pub const fn default_directory_full_node_mirror_max_producers() -> usize {
        32
    }

    /// Default independent Directory observation witness threshold.
    #[must_use]
    pub const fn default_directory_observation_witness_min_verified() -> usize {
        1
    }

    /// Default independent route-domain attestor threshold.
    #[must_use]
    pub const fn default_route_domain_attestation_min_verified() -> usize {
        1
    }

    /// Default external cache-anchor witness threshold.
    #[must_use]
    pub const fn default_verified_delivery_witness_min_verified() -> usize {
        1
    }

    /// Default independent custody witness eligibility threshold.
    #[must_use]
    pub const fn default_custody_audit_witness_min_verified() -> usize {
        1
    }

    /// Default freshness window for producer-side custody witness receipts.
    #[must_use]
    pub const fn default_custody_audit_witness_max_age_secs() -> u64 {
        2 * 60 * 60
    }

    /// Default outbound gossip interval.
    #[must_use]
    pub const fn default_gossip_interval_secs() -> u64 {
        60
    }

    /// Default outbound gossip peer limit per round.
    #[must_use]
    pub const fn default_gossip_peer_limit() -> u16 {
        32
    }

    /// Default bounded outbound gossip concurrency.
    #[must_use]
    pub const fn default_gossip_concurrency_limit() -> u16 {
        8
    }

    /// Default outbound gossip scheduling jitter as a percent of base interval.
    #[must_use]
    pub const fn default_gossip_jitter_percent() -> u8 {
        20
    }

    /// Default consecutive failure threshold before seed-only backpressure.
    #[must_use]
    pub const fn default_gossip_backpressure_failure_threshold() -> u64 {
        3
    }

    /// Default maximum outbound gossip delay while backpressure is active.
    #[must_use]
    pub const fn default_gossip_failure_backoff_max_secs() -> u64 {
        300
    }

    /// Default maximum verified peers retained locally.
    #[must_use]
    pub const fn default_max_peers() -> usize {
        2048
    }

    /// Default maximum descriptors in one snapshot response.
    #[must_use]
    pub const fn default_max_snapshot_limit() -> usize {
        256
    }

    /// Default global inbound gossip request budget per minute.
    #[must_use]
    pub const fn default_gossip_rate_limit_per_minute() -> u32 {
        120
    }

    /// Default signed descriptor time-to-live.
    #[must_use]
    pub const fn default_descriptor_ttl_secs() -> u64 {
        3600
    }

    /// Default public discovery visibility.
    #[must_use]
    pub const fn default_public_discovery() -> bool {
        true
    }

    /// Validates discovery bootstrap configuration.
    pub fn validate(&self) -> Result<()> {
        if self.fetch_timeout_secs == 0 {
            return Err(ServerError::config_invalid(
                "discovery.fetch_timeout_secs",
                "must be greater than zero",
            ));
        }

        if let Some(path) = &self.bootstrap_snapshot_path {
            if path.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_path",
                    "must not be empty when provided",
                ));
            }
        }

        if let Some(url) = &self.bootstrap_snapshot_url {
            let trimmed = url.trim();
            if trimmed.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_url",
                    "must not be empty when provided",
                ));
            }
            if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_url",
                    "must start with http:// or https://",
                ));
            }
        }

        for endpoint in &self.seed_endpoints {
            Self::validate_seed_endpoint(endpoint)?;
        }

        if let Some(path) = &self.peer_cache_path {
            if path.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.peer_cache_path",
                    "must not be empty when provided",
                ));
            }
        }

        if let Some(path) = &self.directory_chain_path {
            let path = path.trim();
            if path.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_path",
                    "must not be empty when provided",
                ));
            }
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_path",
                    "requires discovery.enabled = true",
                ));
            }
            let conflicts_with = [
                self.peer_cache_path.as_deref(),
                self.bootstrap_snapshot_path.as_deref(),
            ]
            .into_iter()
            .flatten()
            .any(|other| other.trim() == path);
            if conflicts_with {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_path",
                    "must not reuse peer_cache_path or bootstrap_snapshot_path",
                ));
            }
        }

        if self.directory_observation_witness_min_verified == 0
            || self.directory_observation_witness_min_verified
                > MAX_DIRECTORY_CHAIN_SYNC_PEER_NODE_IDS
        {
            return Err(ServerError::config_invalid(
                "discovery.directory_observation_witness_min_verified",
                format!("must be between 1 and {MAX_DIRECTORY_CHAIN_SYNC_PEER_NODE_IDS}"),
            ));
        }

        if !self.directory_chain_sync_peer_node_ids.is_empty() {
            if self.directory_chain_path.is_none() {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_sync_peer_node_ids",
                    "requires discovery.directory_chain_path",
                ));
            }
            if self.directory_chain_sync_peer_node_ids.len()
                > MAX_DIRECTORY_CHAIN_SYNC_PEER_NODE_IDS
            {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_sync_peer_node_ids",
                    format!(
                        "supports at most {MAX_DIRECTORY_CHAIN_SYNC_PEER_NODE_IDS} pinned peers"
                    ),
                ));
            }
            let mut validated =
                Vec::<[u8; 32]>::with_capacity(self.directory_chain_sync_peer_node_ids.len());
            for configured in &self.directory_chain_sync_peer_node_ids {
                let value = configured.trim();
                let decoded = hex::decode(value).map_err(|_| {
                    ServerError::config_invalid(
                        "discovery.directory_chain_sync_peer_node_ids",
                        "each entry must be a 64-character Ed25519 public key in hexadecimal",
                    )
                })?;
                let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
                    ServerError::config_invalid(
                        "discovery.directory_chain_sync_peer_node_ids",
                        "each entry must decode to exactly 32 bytes",
                    )
                })?;
                if value.len() != 64 || node_id.iter().all(|byte| *byte == 0) {
                    return Err(ServerError::config_invalid(
                        "discovery.directory_chain_sync_peer_node_ids",
                        "each entry must be a non-zero 64-character Ed25519 public key",
                    ));
                }
                if validated.contains(&node_id) {
                    return Err(ServerError::config_invalid(
                        "discovery.directory_chain_sync_peer_node_ids",
                        "duplicate peer identities are not allowed",
                    ));
                }
                validated.push(node_id);
            }
            if self.directory_observation_witness_min_verified > validated.len() {
                return Err(ServerError::config_invalid(
                    "discovery.directory_observation_witness_min_verified",
                    "must not exceed the number of pinned Directory Sync peers",
                ));
            }
        } else if self.directory_observation_witness_min_verified
            != Self::default_directory_observation_witness_min_verified()
        {
            return Err(ServerError::config_invalid(
                "discovery.directory_observation_witness_min_verified",
                "requires at least one discovery.directory_chain_sync_peer_node_ids entry",
            ));
        }

        if !(60..=86_400).contains(&self.directory_chain_sync_interval_secs) {
            return Err(ServerError::config_invalid(
                "discovery.directory_chain_sync_interval_secs",
                "must be between 60 seconds and 24 hours",
            ));
        }

        if let Some(min_age_secs) = self.directory_gossip_proof_min_age_secs {
            // [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] An operator may
            // lengthen convergence time but cannot publish ahead of the safe
            // two-round floor derived from the configured replica cadence.
            let convergence_floor = self.directory_chain_sync_interval_secs.saturating_mul(2);
            if !(convergence_floor..=MAX_DIRECTORY_GOSSIP_PROOF_MIN_AGE_SECS)
                .contains(&min_age_secs)
            {
                return Err(ServerError::config_invalid(
                    "discovery.directory_gossip_proof_min_age_secs",
                    format!(
                        "must be between {convergence_floor} seconds and \
                         {MAX_DIRECTORY_GOSSIP_PROOF_MIN_AGE_SECS} seconds"
                    ),
                ));
            }
        }

        if !(1..=MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS)
            .contains(&self.directory_full_node_mirror_max_producers)
        {
            return Err(ServerError::config_invalid(
                "discovery.directory_full_node_mirror_max_producers",
                format!("must be between 1 and {MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS}"),
            ));
        }
        if self.directory_full_node_mirror_enabled && self.directory_chain_path.is_none() {
            return Err(ServerError::config_invalid(
                "discovery.directory_full_node_mirror_enabled",
                "requires discovery.directory_chain_path",
            ));
        }
        if self.advertise_directory_mirror_carrier {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.advertise_directory_mirror_carrier",
                    "requires discovery.enabled = true",
                ));
            }
            if !self.directory_full_node_mirror_enabled {
                return Err(ServerError::config_invalid(
                    "discovery.advertise_directory_mirror_carrier",
                    "requires discovery.directory_full_node_mirror_enabled = true",
                ));
            }
            if !self.public_discovery {
                return Err(ServerError::config_invalid(
                    "discovery.advertise_directory_mirror_carrier",
                    "requires discovery.public_discovery = true",
                ));
            }
            if self.public_api_listen_addr.is_none()
                || self
                    .public_endpoint
                    .as_deref()
                    .map(str::trim)
                    .map(str::is_empty)
                    .unwrap_or(true)
            {
                return Err(ServerError::config_invalid(
                    "discovery.advertise_directory_mirror_carrier",
                    "requires discovery.public_api_listen_addr and discovery.public_endpoint",
                ));
            }
        }

        if !self.verified_delivery_witness_node_ids.is_empty()
            || self.verified_delivery_witness_required_for_restore
            || self.verified_delivery_witness_min_verified
                != Self::default_verified_delivery_witness_min_verified()
        {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.verified_delivery_witness_node_ids",
                    "requires discovery.enabled = true",
                ));
            }
            if self.peer_cache_path.is_none() {
                return Err(ServerError::config_invalid(
                    "discovery.peer_cache_path",
                    "is required when verified-delivery witnesses are configured",
                ));
            }
            if self.verified_delivery_witness_node_ids.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.verified_delivery_witness_node_ids",
                    "requires at least one pinned witness identity",
                ));
            }
            if self.verified_delivery_witness_node_ids.len()
                > MAX_VERIFIED_DELIVERY_WITNESS_NODE_IDS
            {
                return Err(ServerError::config_invalid(
                    "discovery.verified_delivery_witness_node_ids",
                    format!(
                        "supports at most {MAX_VERIFIED_DELIVERY_WITNESS_NODE_IDS} pinned witnesses"
                    ),
                ));
            }
            let mut validated =
                Vec::<[u8; 32]>::with_capacity(self.verified_delivery_witness_node_ids.len());
            for configured in &self.verified_delivery_witness_node_ids {
                let value = configured.trim();
                let decoded = hex::decode(value).map_err(|_| {
                    ServerError::config_invalid(
                        "discovery.verified_delivery_witness_node_ids",
                        "each entry must be a 64-character Ed25519 public key in hexadecimal",
                    )
                })?;
                let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
                    ServerError::config_invalid(
                        "discovery.verified_delivery_witness_node_ids",
                        "each entry must decode to exactly 32 bytes",
                    )
                })?;
                if value.len() != 64 || node_id.iter().all(|byte| *byte == 0) {
                    return Err(ServerError::config_invalid(
                        "discovery.verified_delivery_witness_node_ids",
                        "each entry must be a non-zero 64-character Ed25519 public key",
                    ));
                }
                if validated.contains(&node_id) {
                    return Err(ServerError::config_invalid(
                        "discovery.verified_delivery_witness_node_ids",
                        "duplicate witness identities are not allowed",
                    ));
                }
                validated.push(node_id);
            }
            if self.verified_delivery_witness_min_verified == 0
                || self.verified_delivery_witness_min_verified > validated.len()
            {
                return Err(ServerError::config_invalid(
                    "discovery.verified_delivery_witness_min_verified",
                    "must be between one and the number of configured witnesses",
                ));
            }
        }

        if !self.verified_delivery_witness_requester_node_ids.is_empty() {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.verified_delivery_witness_requester_node_ids",
                    "requires discovery.enabled = true",
                ));
            }
            validate_node_id_pin_set(
                "discovery.verified_delivery_witness_requester_node_ids",
                &self.verified_delivery_witness_requester_node_ids,
                MAX_VERIFIED_DELIVERY_WITNESS_REQUESTER_NODE_IDS,
            )?;
        }

        if !self.custody_audit_witness_node_ids.is_empty()
            || self.custody_audit_witness_min_verified
                != Self::default_custody_audit_witness_min_verified()
            || self.custody_audit_witness_startup_required
            || self.custody_audit_witness_runtime_required
            || self.custody_audit_witness_max_age_secs
                != Self::default_custody_audit_witness_max_age_secs()
        {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_node_ids",
                    "requires discovery.enabled = true",
                ));
            }
            if self.custody_audit_witness_node_ids.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_node_ids",
                    "requires at least one pinned independent witness identity",
                ));
            }
            let validated = validate_node_id_pin_set(
                "discovery.custody_audit_witness_node_ids",
                &self.custody_audit_witness_node_ids,
                MAX_CUSTODY_AUDIT_WITNESS_NODE_IDS,
            )?;
            if self.custody_audit_witness_min_verified == 0
                || self.custody_audit_witness_min_verified > validated.len()
            {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_min_verified",
                    "must be between one and the number of configured custody witnesses",
                ));
            }
            if !(60..=MAX_CUSTODY_AUDIT_WITNESS_AGE_SECS)
                .contains(&self.custody_audit_witness_max_age_secs)
            {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_max_age_secs",
                    "must be between 60 and 604800 seconds",
                ));
            }
            if self.custody_audit_witness_runtime_required
                && !self.custody_audit_witness_startup_required
            {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_runtime_required",
                    "requires custody_audit_witness_startup_required = true",
                ));
            }
        }

        if !self.custody_audit_witness_requester_node_ids.is_empty() {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_requester_node_ids",
                    "requires discovery.enabled = true",
                ));
            }
            validate_node_id_pin_set(
                "discovery.custody_audit_witness_requester_node_ids",
                &self.custody_audit_witness_requester_node_ids,
                MAX_CUSTODY_AUDIT_WITNESS_REQUESTER_NODE_IDS,
            )?;
        }

        if self.peer_cache_write_interval_secs < 30 {
            return Err(ServerError::config_invalid(
                "discovery.peer_cache_write_interval_secs",
                "must be at least 30 seconds",
            ));
        }

        if self.gossip_interval_secs < 30 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_interval_secs",
                "must be at least 30 seconds",
            ));
        }

        if self.gossip_peer_limit == 0 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_peer_limit",
                "must be greater than zero",
            ));
        }

        if self.gossip_concurrency_limit == 0
            || self.gossip_concurrency_limit > MAX_DISCOVERY_GOSSIP_CONCURRENCY
        {
            return Err(ServerError::config_invalid(
                "discovery.gossip_concurrency_limit",
                "must be between 1 and 64",
            ));
        }

        if self.gossip_jitter_percent > 50 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_jitter_percent",
                "must be 50 or less",
            ));
        }

        if self.gossip_backpressure_failure_threshold == 0 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_backpressure_failure_threshold",
                "must be greater than zero",
            ));
        }

        if self.gossip_failure_backoff_max_secs < self.gossip_interval_secs {
            return Err(ServerError::config_invalid(
                "discovery.gossip_failure_backoff_max_secs",
                "must be at least discovery.gossip_interval_secs",
            ));
        }

        if self.max_peers == 0 {
            return Err(ServerError::config_invalid(
                "discovery.max_peers",
                "must be greater than zero",
            ));
        }

        if self.max_snapshot_limit == 0 {
            return Err(ServerError::config_invalid(
                "discovery.max_snapshot_limit",
                "must be greater than zero",
            ));
        }

        if self.gossip_rate_limit_per_minute == 0 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_rate_limit_per_minute",
                "must be greater than zero",
            ));
        }

        for peer_id in self
            .allowed_peer_ids
            .iter()
            .chain(self.denied_peer_ids.iter())
        {
            let trimmed = peer_id.trim();
            if trimmed.len() != 64 || !trimmed.chars().all(|ch| ch.is_ascii_hexdigit()) {
                return Err(ServerError::config_invalid(
                    "discovery.allowed_peer_ids/denied_peer_ids",
                    "peer ids must be 32-byte hex strings",
                ));
            }
        }

        // [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Route-domain pins are
        // operator-reviewed local trust input. Strict identity/token syntax
        // keeps normalization deterministic and prevents labels from leaking
        // operator or infrastructure names into logs and future API surfaces.
        if self.pinned_route_domains.len() > MAX_PINNED_ROUTE_DOMAINS {
            return Err(ServerError::config_invalid(
                "discovery.pinned_route_domains",
                format!("supports at most {MAX_PINNED_ROUTE_DOMAINS} node assignments"),
            ));
        }
        let mut normalized_node_ids =
            Vec::<[u8; 32]>::with_capacity(self.pinned_route_domains.len());
        for (configured_node_id, configured_domain) in &self.pinned_route_domains {
            let node_id_text = configured_node_id.trim();
            let decoded = hex::decode(node_id_text).map_err(|_| {
                ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "keys must be 64-character Ed25519 public keys in hexadecimal",
                )
            })?;
            let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
                ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "keys must decode to exactly 32 bytes",
                )
            })?;
            if node_id_text.len() != 64 || node_id.iter().all(|byte| *byte == 0) {
                return Err(ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "keys must be non-zero 64-character Ed25519 public keys",
                ));
            }
            if normalized_node_ids.contains(&node_id) {
                return Err(ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "duplicate node identities after hexadecimal normalization are not allowed",
                ));
            }
            normalized_node_ids.push(node_id);

            let domain = configured_domain.trim();
            if domain.len() != 32 || !domain.chars().all(|ch| ch.is_ascii_hexdigit()) {
                return Err(ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "values must be opaque 128-bit route-domain tokens encoded as 32 hexadecimal characters",
                ));
            }
            let domain_bytes = hex::decode(domain).map_err(|_| {
                ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "values must be valid hexadecimal route-domain tokens",
                )
            })?;
            if domain_bytes.iter().all(|byte| *byte == 0) {
                return Err(ServerError::config_invalid(
                    "discovery.pinned_route_domains",
                    "route-domain tokens must not be all zero",
                ));
            }
        }
        if self.require_pinned_route_domains_for_multi_hop && self.pinned_route_domains.is_empty() {
            return Err(ServerError::config_invalid(
                "discovery.require_pinned_route_domains_for_multi_hop",
                "requires at least one discovery.pinned_route_domains assignment",
            ));
        }
        if self.require_pinned_route_domains_for_multi_hop && !self.enabled {
            return Err(ServerError::config_invalid(
                "discovery.require_pinned_route_domains_for_multi_hop",
                "requires discovery.enabled = true",
            ));
        }
        if self.require_pinned_route_domains_for_multi_hop && self.directory_chain_path.is_none() {
            return Err(ServerError::config_invalid(
                "discovery.require_pinned_route_domains_for_multi_hop",
                "requires discovery.directory_chain_path so the active policy has a signed, restart-audited history",
            ));
        }

        // [ROUTE-DOMAIN-ATTESTOR-POLICY 2026-08-03 by Codex] Attestor pins
        // are verifier-local trust roots. Strict syntax, a bounded set, and a
        // local threshold prevent malformed or duplicate identities from
        // weakening certificate admission. Configuring the policy requires a
        // Directory store so later imports and pin changes remain auditable.
        let attestor_policy_configured = !self.route_domain_attestor_node_ids.is_empty()
            || self.require_route_domain_attestations_for_multi_hop
            || self.route_domain_attestation_min_verified
                != Self::default_route_domain_attestation_min_verified();
        if self.route_domain_attestor_node_ids.len() > MAX_ROUTE_DOMAIN_ATTESTOR_NODE_IDS {
            return Err(ServerError::config_invalid(
                "discovery.route_domain_attestor_node_ids",
                format!("supports at most {MAX_ROUTE_DOMAIN_ATTESTOR_NODE_IDS} pinned attestors"),
            ));
        }
        let mut normalized_attestors =
            Vec::<[u8; 32]>::with_capacity(self.route_domain_attestor_node_ids.len());
        for configured_attestor in &self.route_domain_attestor_node_ids {
            let value = configured_attestor.trim();
            let decoded = hex::decode(value).map_err(|_| {
                ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "entries must be 64-character Ed25519 public keys in hexadecimal",
                )
            })?;
            let node_id: [u8; 32] = decoded.try_into().map_err(|_| {
                ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "entries must decode to exactly 32 bytes",
                )
            })?;
            if value.len() != 64 || node_id == [0u8; 32] {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "entries must be non-zero 64-character Ed25519 public keys",
                ));
            }
            if normalized_attestors.contains(&node_id) {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "duplicate attestor identities after hexadecimal normalization are not allowed",
                ));
            }
            normalized_attestors.push(node_id);
        }
        if normalized_attestors
            .iter()
            .any(|attestor| normalized_node_ids.contains(attestor))
        {
            return Err(ServerError::config_invalid(
                "discovery.route_domain_attestor_node_ids",
                "attestors must not overlap route-domain subjects because self-attestation is invalid",
            ));
        }
        if attestor_policy_configured {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "requires discovery.enabled = true",
                ));
            }
            if self.directory_chain_path.is_none() {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "requires discovery.directory_chain_path for signed policy and certificate history",
                ));
            }
            if normalized_attestors.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestor_node_ids",
                    "requires at least one pinned attestor",
                ));
            }
            if self.route_domain_attestation_min_verified == 0
                || self.route_domain_attestation_min_verified > normalized_attestors.len()
            {
                return Err(ServerError::config_invalid(
                    "discovery.route_domain_attestation_min_verified",
                    "must be between one and the number of configured attestors",
                ));
            }
        }
        if self.require_route_domain_attestations_for_multi_hop
            && !self.require_pinned_route_domains_for_multi_hop
        {
            return Err(ServerError::config_invalid(
                "discovery.require_route_domain_attestations_for_multi_hop",
                "requires discovery.require_pinned_route_domains_for_multi_hop = true",
            ));
        }

        if let Some(endpoint) = &self.public_endpoint {
            if endpoint.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.public_endpoint",
                    "must not be empty when provided",
                ));
            }
        }

        if let Some(addr) = self.public_api_listen_addr {
            if addr.port() == 0 {
                return Err(ServerError::config_invalid(
                    "discovery.public_api_listen_addr",
                    "port must be greater than zero",
                ));
            }
        }

        if let Some(region) = &self.region {
            if region.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.region",
                    "must not be empty when provided",
                ));
            }
        }

        if self.descriptor_ttl_secs < 60 {
            return Err(ServerError::config_invalid(
                "discovery.descriptor_ttl_secs",
                "must be at least 60 seconds",
            ));
        }

        Ok(())
    }

    /// Returns validated external cache-anchor witness identities in pin order.
    ///
    /// Validation rejects malformed values. `filter_map` keeps this accessor
    /// panic-free for tests and internal callers that bypass validation.
    #[must_use]
    pub fn verified_delivery_witness_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.verified_delivery_witness_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Returns validated Directory Sync peer identities in operator pin order.
    #[must_use]
    pub fn directory_chain_sync_peer_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.directory_chain_sync_peer_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Returns validated route-domain assignments in canonical node-id order.
    ///
    /// Configuration validation rejects malformed inputs. The defensive
    /// `filter_map` keeps this internal accessor panic-free for embedders that
    /// construct an unchecked `DiscoveryConfig` directly.
    #[must_use]
    pub(crate) fn pinned_route_domain_assignments(&self) -> Vec<PinnedRouteDomainAssignment> {
        let mut assignments = self
            .pinned_route_domains
            .iter()
            .filter_map(|(node_id, route_domain)| {
                let node_id: [u8; 32] = hex::decode(node_id.trim()).ok()?.try_into().ok()?;
                let route_domain: [u8; 16] =
                    hex::decode(route_domain.trim()).ok()?.try_into().ok()?;
                Some(PinnedRouteDomainAssignment {
                    node_id,
                    route_domain,
                })
            })
            .collect::<Vec<_>>();
        assignments.sort_unstable();
        assignments
    }

    /// Returns validated route-domain attestor identities in configured order.
    ///
    /// Configuration validation rejects malformed or duplicate values. The
    /// defensive `filter_map` keeps internal callers panic-free when tests or
    /// embedders construct an unchecked `DiscoveryConfig` directly.
    #[must_use]
    pub(crate) fn route_domain_attestor_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.route_domain_attestor_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Returns validated identities allowed to use this node as a witness.
    #[must_use]
    pub fn verified_delivery_witness_requester_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.verified_delivery_witness_requester_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Returns validated producer identities allowed to request custody proof.
    #[must_use]
    pub fn custody_audit_witness_requester_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.custody_audit_witness_requester_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Returns producer-side custody witness candidates in operator pin order.
    #[must_use]
    pub fn custody_audit_witness_node_id_bytes(&self) -> Vec<[u8; 32]> {
        self.custody_audit_witness_node_ids
            .iter()
            .filter_map(|value| {
                let decoded = hex::decode(value.trim()).ok()?;
                decoded.try_into().ok()
            })
            .collect()
    }

    /// Rejects a producer policy that counts this node as its own witness.
    ///
    /// [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Static TOML
    /// validation cannot compare pins with the identity derived from the node
    /// key. Startup calls this before opening protocol transports or storage.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the local node identity is present
    /// in the producer's independent custody-witness pin set.
    pub fn validate_runtime_identity(&self, local_node_id: &[u8; 32]) -> Result<()> {
        if self
            .custody_audit_witness_node_id_bytes()
            .contains(local_node_id)
        {
            return Err(ServerError::config_invalid(
                "discovery.custody_audit_witness_node_ids",
                "must not contain this node's own identity",
            ));
        }
        Ok(())
    }

    fn validate_seed_endpoint(endpoint: &str) -> Result<()> {
        let trimmed = endpoint.trim();
        if trimmed.is_empty() {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must not be empty",
            ));
        }
        if trimmed.contains(char::is_whitespace) {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must not contain whitespace",
            ));
        }
        if !(trimmed.starts_with("http://")
            || trimmed.starts_with("https://")
            || trimmed.contains(':'))
        {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must be http(s) URLs or host:port endpoints",
            ));
        }
        Ok(())
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            advertise_self: Self::default_advertise_self(),
            bootstrap_snapshot_path: None,
            bootstrap_snapshot_url: None,
            seed_endpoints: Vec::new(),
            fetch_timeout_secs: Self::default_fetch_timeout_secs(),
            peer_cache_path: None,
            directory_chain_path: None,
            directory_chain_sync_peer_node_ids: Vec::new(),
            directory_chain_sync_interval_secs: Self::default_directory_chain_sync_interval_secs(),
            directory_gossip_proof_min_age_secs: None,
            directory_full_node_mirror_enabled: false,
            advertise_directory_mirror_carrier: false,
            directory_full_node_mirror_max_producers:
                Self::default_directory_full_node_mirror_max_producers(),
            directory_observation_witness_min_verified:
                Self::default_directory_observation_witness_min_verified(),
            verified_delivery_witness_node_ids: Vec::new(),
            verified_delivery_witness_requester_node_ids: Vec::new(),
            custody_audit_witness_node_ids: Vec::new(),
            custody_audit_witness_min_verified: Self::default_custody_audit_witness_min_verified(),
            custody_audit_witness_startup_required: false,
            custody_audit_witness_runtime_required: false,
            custody_audit_witness_max_age_secs: Self::default_custody_audit_witness_max_age_secs(),
            custody_audit_witness_requester_node_ids: Vec::new(),
            verified_delivery_witness_min_verified:
                Self::default_verified_delivery_witness_min_verified(),
            verified_delivery_witness_required_for_restore: false,
            peer_cache_write_interval_secs: Self::default_peer_cache_write_interval_secs(),
            gossip_enabled: false,
            gossip_interval_secs: Self::default_gossip_interval_secs(),
            gossip_peer_limit: Self::default_gossip_peer_limit(),
            gossip_concurrency_limit: Self::default_gossip_concurrency_limit(),
            gossip_jitter_percent: Self::default_gossip_jitter_percent(),
            gossip_backpressure_failure_threshold:
                Self::default_gossip_backpressure_failure_threshold(),
            gossip_failure_backoff_max_secs: Self::default_gossip_failure_backoff_max_secs(),
            max_peers: Self::default_max_peers(),
            max_snapshot_limit: Self::default_max_snapshot_limit(),
            gossip_rate_limit_per_minute: Self::default_gossip_rate_limit_per_minute(),
            allowed_peer_ids: Vec::new(),
            denied_peer_ids: Vec::new(),
            pinned_route_domains: BTreeMap::new(),
            require_pinned_route_domains_for_multi_hop: false,
            route_domain_attestor_node_ids: Vec::new(),
            route_domain_attestation_min_verified:
                Self::default_route_domain_attestation_min_verified(),
            require_route_domain_attestations_for_multi_hop: false,
            public_endpoint: None,
            public_api_listen_addr: None,
            region: None,
            descriptor_ttl_secs: Self::default_descriptor_ttl_secs(),
            public_discovery: Self::default_public_discovery(),
            advertise_onion_middle: false,
        }
    }
}

// ============================================
// ServerConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub vpn: VpnConfig,
    #[serde(default)]
    pub tun: TunConfig,
    #[serde(default)]
    pub server_key: ServerKeyConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub management: ManagementConfig,
    #[serde(default)]
    pub memchain: MemChainConfig,
    #[serde(default)]
    pub discovery: DiscoveryConfig,
    /// [BLIND-VAULT-SERVICE 2026-07-23 by Codex] Independent anonymous
    /// encrypted-object storage; disabled unless explicitly configured.
    #[serde(default)]
    pub blind_vault: BlindVaultConfig,
}

impl ServerConfig {
    /// Load and validate configuration from a TOML file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        config.validate()?;
        info!("Configuration loaded successfully");
        Ok(config)
    }

    /// Parse and validate configuration from a TOML string (used in tests).
    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate all sub-configs in dependency order.
    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;
        self.management
            .validate()
            .map_err(|e| ServerError::config_invalid("management", e))?;
        self.memchain.validate()?;
        self.discovery.validate()?;
        self.blind_vault.validate()?;
        if let Some(directory_path) = self.discovery.directory_chain_path.as_deref() {
            let directory_path = directory_path.trim();
            if [
                self.memchain.db_path.as_str(),
                self.memchain.chat_relay.db_path.as_str(),
            ]
            .into_iter()
            .any(|other| other.trim() == directory_path)
            {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_path",
                    "must not reuse a MemChain or ChatRelay database path",
                ));
            }
        }
        if self.blind_vault.enabled {
            let blind_vault_path = self.blind_vault.db_path.trim();
            if [
                self.memchain.db_path.as_str(),
                self.memchain.chat_relay.db_path.as_str(),
            ]
            .into_iter()
            .any(|other| other.trim() == blind_vault_path)
                || self
                    .discovery
                    .directory_chain_path
                    .as_deref()
                    .is_some_and(|path| path.trim() == blind_vault_path)
            {
                return Err(ServerError::config_invalid(
                    "blind_vault.db_path",
                    "must not reuse a MemChain, ChatRelay, or Directory Chain database path",
                ));
            }
        }
        // [WITNESS-CONFIG-DIAGNOSTICS 2026-08-16 by Codex] Keep cross-module
        // errors bound to real TOML fields so installers can report and repair
        // the exact unsafe policy instead of receiving a wildcard pseudo-key.
        if !self.memchain.is_enabled()
            && !self.discovery.verified_delivery_witness_node_ids.is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.verified_delivery_witness_node_ids",
                "authenticated witness transport requires a local MemChain storage mode",
            ));
        }
        if !self.memchain.is_enabled()
            && !self
                .discovery
                .verified_delivery_witness_requester_node_ids
                .is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.verified_delivery_witness_requester_node_ids",
                "the authenticated delivery witness endpoint requires a local MemChain storage mode",
            ));
        }
        if !self.memchain.is_enabled()
            && !self
                .discovery
                .custody_audit_witness_requester_node_ids
                .is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.custody_audit_witness_requester_node_ids",
                "the authenticated custody witness endpoint requires a local MemChain storage mode",
            ));
        }
        if !self.discovery.custody_audit_witness_node_ids.is_empty()
            && !self.memchain.is_chat_relay_enabled()
        {
            return Err(ServerError::config_invalid(
                "discovery.custody_audit_witness_node_ids",
                "requires memchain.chat_relay.enabled = true to produce custody anchors",
            ));
        }
        Ok(())
    }

    // ── Convenience accessors ──────────────────────────────────────────

    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    #[must_use]
    pub fn listen_addr(&self) -> SocketAddr {
        self.network.listen_addr
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.tun.device_name
    }

    #[must_use]
    pub fn ip_range(&self) -> &str {
        &self.vpn.virtual_ip_range
    }

    #[must_use]
    pub fn gateway_ip(&self) -> Ipv4Addr {
        self.vpn.gateway_ip
    }

    #[must_use]
    pub fn dns_proxy_enabled(&self) -> bool {
        self.vpn.dns_proxy_enabled
    }

    #[must_use]
    pub fn vpn_transports(&self) -> &VpnTransportConfig {
        &self.vpn.transports
    }

    #[must_use]
    pub fn mtu(&self) -> u16 {
        self.tun.mtu
    }

    #[must_use]
    pub fn max_sessions(&self) -> usize {
        self.limits.max_connections
    }

    #[must_use]
    pub fn session_timeout_secs(&self) -> u64 {
        self.limits.session_timeout
    }

    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        self.vpn.parse_ip_range()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            vpn: VpnConfig::default(),
            tun: TunConfig::default(),
            server_key: ServerKeyConfig::default(),
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
            management: ManagementConfig::default(),
            memchain: MemChainConfig::default(),
            discovery: DiscoveryConfig::default(),
            blind_vault: BlindVaultConfig::default(),
        }
    }
}

// ============================================
// Integration Tests
// (unit tests live in each sub-module's own #[cfg(test)])
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Full-stack default validation ─────────────────────────────────────

    #[test]
    fn test_default_config_valid() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        // Spot-check a few fields to ensure sub-modules wired correctly
        assert_eq!(config.memchain.mvf_alpha, 0.5);
        assert!(!config.memchain.mvf_enabled);
        assert!(config.memchain.api_secret.is_none());
        assert!(!config.memchain.allow_remote_storage);
        assert!(!config.memchain.ner_enabled);
        assert!(!config.memchain.graph_enabled);
        assert!(!config.memchain.supernode.enabled);
        assert!(!config.memchain.is_saas());
        assert!(!config.memchain.is_chat_relay_enabled());
        assert!(config.dns_proxy_enabled());
        assert!(!config.discovery.enabled);
        assert!(config.discovery.advertise_self);
        assert_eq!(
            config.discovery.fetch_timeout_secs,
            DiscoveryConfig::default_fetch_timeout_secs()
        );
        assert!(config.discovery.peer_cache_path.is_none());
        assert!(config
            .discovery
            .verified_delivery_witness_node_ids
            .is_empty());
        assert!(config
            .discovery
            .verified_delivery_witness_requester_node_ids
            .is_empty());
        assert!(config
            .discovery
            .custody_audit_witness_requester_node_ids
            .is_empty());
        assert!(config.discovery.custody_audit_witness_node_ids.is_empty());
        assert_eq!(
            config.discovery.custody_audit_witness_min_verified,
            DiscoveryConfig::default_custody_audit_witness_min_verified()
        );
        assert!(!config.discovery.custody_audit_witness_startup_required);
        assert!(!config.discovery.custody_audit_witness_runtime_required);
        assert_eq!(
            config.discovery.custody_audit_witness_max_age_secs,
            DiscoveryConfig::default_custody_audit_witness_max_age_secs()
        );
        assert_eq!(
            config.discovery.verified_delivery_witness_min_verified,
            DiscoveryConfig::default_verified_delivery_witness_min_verified()
        );
        assert!(
            !config
                .discovery
                .verified_delivery_witness_required_for_restore
        );
        assert_eq!(
            config.discovery.peer_cache_write_interval_secs,
            DiscoveryConfig::default_peer_cache_write_interval_secs()
        );
        assert_eq!(
            config.discovery.directory_chain_sync_interval_secs,
            DiscoveryConfig::default_directory_chain_sync_interval_secs()
        );
        assert!(config
            .discovery
            .directory_gossip_proof_min_age_secs
            .is_none());
        assert_eq!(
            config
                .discovery
                .effective_directory_gossip_proof_min_age_secs(),
            DiscoveryConfig::default_directory_chain_sync_interval_secs() * 2
        );
        assert!(!config.discovery.directory_full_node_mirror_enabled);
        assert!(!config.discovery.advertise_directory_mirror_carrier);
        assert_eq!(
            config.discovery.directory_full_node_mirror_max_producers,
            DiscoveryConfig::default_directory_full_node_mirror_max_producers()
        );
        assert_eq!(
            config.discovery.directory_observation_witness_min_verified,
            DiscoveryConfig::default_directory_observation_witness_min_verified()
        );
        assert!(!config.discovery.gossip_enabled);
        assert_eq!(
            config.discovery.gossip_interval_secs,
            DiscoveryConfig::default_gossip_interval_secs()
        );
        assert_eq!(
            config.discovery.gossip_peer_limit,
            DiscoveryConfig::default_gossip_peer_limit()
        );
        assert_eq!(
            config.discovery.gossip_concurrency_limit,
            DiscoveryConfig::default_gossip_concurrency_limit()
        );
        assert_eq!(
            config.discovery.gossip_jitter_percent,
            DiscoveryConfig::default_gossip_jitter_percent()
        );
        assert_eq!(
            config.discovery.gossip_backpressure_failure_threshold,
            DiscoveryConfig::default_gossip_backpressure_failure_threshold()
        );
        assert_eq!(
            config.discovery.gossip_failure_backoff_max_secs,
            DiscoveryConfig::default_gossip_failure_backoff_max_secs()
        );
        assert_eq!(
            config.discovery.max_peers,
            DiscoveryConfig::default_max_peers()
        );
        assert_eq!(
            config.discovery.max_snapshot_limit,
            DiscoveryConfig::default_max_snapshot_limit()
        );
        assert_eq!(
            config.discovery.gossip_rate_limit_per_minute,
            DiscoveryConfig::default_gossip_rate_limit_per_minute()
        );
        assert!(config.discovery.allowed_peer_ids.is_empty());
        assert!(config.discovery.denied_peer_ids.is_empty());
        assert!(config.discovery.pinned_route_domains.is_empty());
        assert!(!config.discovery.require_pinned_route_domains_for_multi_hop);
        assert!(config.discovery.route_domain_attestor_node_ids.is_empty());
        assert_eq!(config.discovery.route_domain_attestation_min_verified, 1);
        assert!(
            !config
                .discovery
                .require_route_domain_attestations_for_multi_hop
        );
        assert_eq!(
            config.discovery.descriptor_ttl_secs,
            DiscoveryConfig::default_descriptor_ttl_secs()
        );
        assert!(config.discovery.public_discovery);
    }

    #[test]
    fn test_dns_proxy_enabled_backward_compat_default() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(config.dns_proxy_enabled());
    }

    #[test]
    fn test_vpn_transports_backward_compat_default_udp_only() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(config.vpn_transports().udp_enabled);
        assert!(!config.vpn_transports().tcp_tls_enabled);
        assert!(!config.vpn_transports().websocket_enabled);
        assert_eq!(config.vpn_transports().preferred_transport, "udp");
    }

    #[test]
    fn test_vpn_transports_toml_parse_future_fallback_metadata() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"

[vpn.transports]
udp_enabled = true
tcp_tls_enabled = true
tcp_tls_public_endpoint = "vpn.example.com:443"
websocket_enabled = true
websocket_public_url = "wss://vpn.example.com/aeronyx/vpn"
preferred_transport = "udp"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        let transports = config.vpn_transports();
        assert!(transports.udp_enabled);
        assert!(transports.tcp_tls_enabled);
        assert!(transports.websocket_enabled);
        assert_eq!(
            transports.tcp_tls_public_endpoint.as_deref(),
            Some("vpn.example.com:443")
        );
        assert_eq!(
            transports.websocket_public_url.as_deref(),
            Some("wss://vpn.example.com/aeronyx/vpn")
        );
    }

    #[test]
    fn test_dns_proxy_can_be_disabled_for_external_gateway_dns() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
dns_proxy_enabled = false
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(!config.dns_proxy_enabled());
    }

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

    // ── v1.1.0-ChatRelay: full TOML integration ───────────────────────────

    #[test]
    fn test_chat_relay_full_toml_integration() {
        let toml_str = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true
offline_ttl_secs = 86400
max_pending_per_wallet = 200
db_path = "data/chat_test.db"
max_message_size = 32768
max_blob_size = 5242880
max_blobs_per_receiver = 20
cleanup_interval_secs = 30
dedup_lru_capacity = 5000
expired_notification_ttl_secs = 172800
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let cr = &config.memchain.chat_relay;
        assert!(cr.enabled);
        assert_eq!(cr.offline_ttl_secs, 86_400);
        assert_eq!(cr.max_pending_per_wallet, 200);
        assert_eq!(cr.db_path, "data/chat_test.db");
        assert_eq!(cr.max_message_size, 32_768);
        assert_eq!(cr.max_blob_size, 5_242_880);
        assert_eq!(cr.max_blobs_per_receiver, 20);
        assert_eq!(cr.cleanup_interval_secs, 30);
        assert_eq!(cr.dedup_lru_capacity, 5_000);
        assert_eq!(cr.expired_notification_ttl_secs, 172_800);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_backward_compat_no_section() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.chat_relay.enabled);
        assert!(config.validate().is_ok());
    }

    // ── v2.5.0: SuperNode + v1.1.0 ChatRelay combined ────────────────────

    #[test]
    fn test_supernode_and_chat_relay_combined() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true

[memchain.supernode]
enabled = true

[[memchain.supernode.providers]]
name = "ollama"
type = "openai_compatible"
api_base = "http://localhost:11434/v1"
model = "llama3"

[memchain.chat_relay]
enabled = true
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.is_supernode_enabled());
        assert!(config.memchain.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v1.0.0-MT + v1.1.0-ChatRelay combined ────────────────────────────

    #[test]
    fn test_saas_and_chat_relay_combined() {
        let toml_str = r#"
[memchain]
mode = "saas"
jwt_secret = "a-very-long-secret-key-that-is-at-least-32-chars"

[memchain.saas]
data_root = "/var/memchain"
pool_max_connections = 50

[memchain.chat_relay]
enabled = true
db_path = "/var/memchain/chat_pending.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.is_saas());
        assert!(config.memchain.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v2.4.0: Full cognitive graph TOML ────────────────────────────────

    #[test]
    fn test_v240_toml_full_config() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true
ner_model_path = "models/gliner"
ner_confidence_threshold = 0.45
graph_enabled = true
graph_max_depth = 2
graph_max_nodes_per_hop = 30
graph_min_edge_weight = 0.25
entropy_filter_enabled = true
entropy_filter_threshold = 0.4
entropy_window_size = 8
entropy_window_overlap = 1
miner_entity_extraction = true
miner_community_detection = true
miner_session_summary = true
miner_artifact_extraction = true
vector_quantization = "scalar_uint8"
vector_early_termination = true
vector_saturation_threshold = 3
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(mc.ner_enabled);
        assert!(mc.graph_enabled);
        assert!(mc.entropy_filter_enabled);
        assert!(mc.miner_entity_extraction);
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::ScalarUint8);
        assert!(config.validate().is_ok());
        assert!(mc.is_cognitive_graph_enabled());
        assert!(mc.has_cognitive_miner_steps());
        assert!(!mc.is_supernode_enabled());
        assert!(!mc.is_saas());
        assert!(!mc.is_chat_relay_enabled());
    }

    // ── Backward compatibility: old TOML with none of the new sections ────

    #[test]
    fn test_full_backward_compat() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
mvf_alpha = 0.5
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(!mc.ner_enabled);
        assert!(!mc.graph_enabled);
        assert!(!mc.entropy_filter_enabled);
        assert!(!mc.supernode.enabled);
        assert!(!mc.is_saas());
        assert!(!mc.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v2.5.0: EmbeddingGemma config ────────────────────────────────────

    #[test]
    fn test_embed_gemma_config() {
        let toml_str = r#"
[memchain]
mode = "local"
embed_model_path = "models/embeddinggemma"
embed_max_tokens = 256
embed_output_dim = 384
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert_eq!(mc.embed_model_path, "models/embeddinggemma");
        assert_eq!(mc.embed_output_dim, 384);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_embed_engine_can_be_disabled_for_protocol_nodes() {
        let toml_str = r#"
[memchain]
mode = "local"
embed_enabled = false
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(!mc.embed_enabled);
        assert!(config.validate().is_ok());
    }
}
