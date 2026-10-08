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
//! v0.26.0-CustodyWitnessAutoRenewal - Added an explicit default-off runtime
//! renewal policy that requires both strict startup and runtime custody gates.
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
const MAX_PERMISSIONLESS_ENDPOINT_PROOF_ENTRIES: usize = 65_536;
const MAX_PERMISSIONLESS_ENDPOINT_PROOF_TTL_SECS: u64 = 300;

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
    /// [PHALA-DNS-SEED-PIN 2026-10-07 by Codex] HTTPS DNS origins are resolved
    /// to a bounded, all-public address set and pinned for the gossip request;
    /// TLS still authenticates the configured hostname.
    #[serde(default)]
    pub seed_endpoints: Vec<String>,
    /// Optional local dstack v1 guest-agent socket used for nonce-bound node
    /// attestations. Setting this enables the discovery attestation endpoint.
    /// The socket is never exposed or proxied; `/v1/Attest` is preferred.
    #[serde(default)]
    pub phala_attestation_socket_path: Option<String>,
    /// [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex] Allows the frozen dstack
    /// v0 `/GetQuote` API only when the v1 method
    /// returns the documented missing-mount 404. It is TDX-only and returns
    /// distinct raw-JSON evidence; no trust decision is made by this node.
    #[serde(default)]
    pub phala_attestation_allow_legacy_v0: bool,
    /// [PHALA-PRIVATE-SOURCE-ROUTE-GATE 2026-10-06 by Codex] Require
    /// PeerStore-selected peers plus direct blind-relay, source-pull and
    /// recipient-poll routes to pass fresh dstack v1 TDX appraisal. The peer
    /// must advertise its signed attestation endpoint. This verifier currently
    /// rejects GCP-TDX because it does not validate the accompanying TPM quote.
    /// Inbound admission is unchanged.
    #[serde(default)]
    pub phala_attested_peers_required: bool,
    /// Locally trusted dstack app IDs as `0x` plus lowercase hex of the raw
    /// measured app-id bytes. Never learned from discovery.
    #[serde(default)]
    pub phala_trusted_app_ids: Vec<String>,
    /// Locally trusted measured compose hashes in `sha256:<64 lowercase hex>` form.
    #[serde(default)]
    pub phala_trusted_compose_hashes: Vec<String>,
    /// Maximum age of a process-local peer attestation decision, in seconds.
    #[serde(default = "DiscoveryConfig::default_phala_peer_attestation_max_age_secs")]
    pub phala_peer_attestation_max_age_secs: u64,
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
    /// Renews current-anchor receipts before the strict runtime gate expires.
    ///
    /// [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] This remains
    /// independently default-off because enabling it transmits an aggregate,
    /// producer-signed custody anchor to the exact configured witness pins.
    /// It is valid only when both strict local gates are enabled.
    #[serde(default)]
    pub custody_audit_witness_auto_renewal_enabled: bool,
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
    // [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-24 by Codex] Roll out public
    // candidate verification independently from ordinary discovery.
    /// Enables descriptor-authenticated endpoint proof routes on the public listener.
    ///
    /// This is independently default-off. It never mounts on local, node, or
    /// VPN listeners and does not grant discovery promotion authority.
    #[serde(default)]
    pub permissionless_endpoint_proof_enabled: bool,
    /// Maximum retained endpoint-proof challenge and replay records.
    #[serde(default = "DiscoveryConfig::default_permissionless_endpoint_proof_max_entries")]
    pub permissionless_endpoint_proof_max_entries: usize,
    /// Lifetime of one issued endpoint challenge, in seconds.
    #[serde(default = "DiscoveryConfig::default_permissionless_endpoint_proof_ttl_secs")]
    pub permissionless_endpoint_proof_ttl_secs: u64,
    /// Enables durable retention of already-verified endpoint evidence.
    #[serde(default)]
    pub permissionless_endpoint_evidence_enabled: bool,
    /// Dedicated private SQLite path for endpoint evidence.
    #[serde(default)]
    pub permissionless_endpoint_evidence_db_path: String,
    /// Maximum retained endpoint evidence rows.
    #[serde(default = "DiscoveryConfig::default_permissionless_endpoint_evidence_max_entries")]
    pub permissionless_endpoint_evidence_max_entries: usize,
    /// Evidence retention after local observation, in seconds.
    #[serde(default = "DiscoveryConfig::default_permissionless_endpoint_evidence_ttl_secs")]
    pub permissionless_endpoint_evidence_ttl_secs: u64,
    /// Maximum expired rows removed during one evidence admission.
    #[serde(default = "DiscoveryConfig::default_permissionless_endpoint_evidence_cleanup_batch")]
    pub permissionless_endpoint_evidence_cleanup_batch: usize,
    // [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX-COMPOSITION 2026-09-24 by Codex]
    // Persist third-party ADAT observations only under an independent opt-in.
    /// Enables the durable endpoint-attestation quarantine inbox.
    #[serde(default)]
    pub permissionless_endpoint_attestation_inbox_enabled: bool,
    /// Dedicated private `SQLite` path for endpoint attestations.
    #[serde(default)]
    pub permissionless_endpoint_attestation_inbox_db_path: String,
    /// Maximum retained endpoint-attestation rows.
    #[serde(default = "DiscoveryConfig::default_endpoint_attestation_inbox_max_entries")]
    pub permissionless_endpoint_attestation_inbox_max_entries: usize,
    /// Maximum logical canonical-frame bytes.
    #[serde(default = "DiscoveryConfig::default_endpoint_attestation_inbox_max_bytes")]
    pub permissionless_endpoint_attestation_inbox_max_bytes: u64,
    /// Maximum local retention after admission, in seconds.
    #[serde(default = "DiscoveryConfig::default_endpoint_attestation_inbox_ttl_secs")]
    pub permissionless_endpoint_attestation_inbox_ttl_secs: u64,
    /// Maximum expired rows removed by one admission.
    #[serde(default = "DiscoveryConfig::default_endpoint_attestation_inbox_cleanup_batch")]
    pub permissionless_endpoint_attestation_inbox_cleanup_batch: usize,
    // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] This separate
    // opt-in never upgrades a self-signed Stage-A descriptor by itself.
    /// Enables bounded, evidence-backed candidate promotion on the public node listener.
    #[serde(default)]
    pub permissionless_endpoint_promotion_enabled: bool,
    /// Private SQLite path prefix for quarantine, observation, and revocation.
    #[serde(default)]
    pub permissionless_endpoint_promotion_db_prefix: String,
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
    /// Maximum accepted age for remote Phala quote appraisal by default.
    #[must_use]
    pub const fn default_phala_peer_attestation_max_age_secs() -> u64 {
        15 * 60
    }

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

    /// Default retained endpoint-proof challenge capacity.
    #[must_use]
    pub const fn default_permissionless_endpoint_proof_max_entries() -> usize {
        1024
    }

    /// Default endpoint-proof challenge lifetime.
    #[must_use]
    pub const fn default_permissionless_endpoint_proof_ttl_secs() -> u64 {
        120
    }

    /// Default maximum number of retained endpoint evidence rows.
    #[must_use]
    pub const fn default_permissionless_endpoint_evidence_max_entries() -> usize {
        16_384
    }

    /// Default endpoint evidence retention interval, in seconds.
    #[must_use]
    pub const fn default_permissionless_endpoint_evidence_ttl_secs() -> u64 {
        24 * 60 * 60
    }

    /// Default maximum expired rows removed during one admission.
    #[must_use]
    pub const fn default_permissionless_endpoint_evidence_cleanup_batch() -> usize {
        256
    }

    /// Default maximum retained endpoint-attestation rows.
    #[must_use]
    pub const fn default_endpoint_attestation_inbox_max_entries() -> usize {
        16_384
    }

    /// Default maximum logical endpoint-attestation bytes.
    #[must_use]
    pub const fn default_endpoint_attestation_inbox_max_bytes() -> u64 {
        16 * 1024 * 1024
    }

    /// Default endpoint-attestation retention interval.
    #[must_use]
    pub const fn default_endpoint_attestation_inbox_ttl_secs() -> u64 {
        24 * 60 * 60
    }

    /// Default bounded cleanup batch for endpoint attestations.
    #[must_use]
    pub const fn default_endpoint_attestation_inbox_cleanup_batch() -> usize {
        256
    }

    /// Default public discovery visibility.
    #[must_use]
    pub const fn default_public_discovery() -> bool {
        true
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Validate the outbound
    // trust policy independently of listener/socket overrides still in progress.
    fn validate_phala_peer_policy(&self) -> Result<()> {
        if self.phala_attested_peers_required {
            if !self.enabled || !self.gossip_enabled {
                return Err(ServerError::config_invalid(
                    "discovery.phala_attested_peers_required",
                    "requires discovery and gossip to be enabled",
                ));
            }
            if self.phala_trusted_app_ids.is_empty()
                || self.phala_trusted_compose_hashes.is_empty()
            {
                return Err(ServerError::config_invalid(
                    "discovery.phala_trusted_app_ids",
                    "required Phala peer routing needs explicit app-id and compose-hash allowlists",
                ));
            }
        }
        if self.phala_peer_attestation_max_age_secs == 0
            || self.phala_peer_attestation_max_age_secs > 24 * 60 * 60
        {
            return Err(ServerError::config_invalid(
                "discovery.phala_peer_attestation_max_age_secs",
                "must be between 1 and 86400 seconds",
            ));
        }
        for app_id in &self.phala_trusted_app_ids {
            let Some(hex_app_id) = app_id.strip_prefix("0x") else {
                return Err(ServerError::config_invalid(
                    "discovery.phala_trusted_app_ids",
                    "entries must use 0x<lowercase hex of measured app-id bytes>",
                ));
            };
            if app_id.trim() != app_id
                || hex_app_id.is_empty()
                || hex_app_id.len() > 128
                || hex_app_id.len() % 2 != 0
                || !hex_app_id.bytes().all(|byte| {
                    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                })
            {
                return Err(ServerError::config_invalid(
                    "discovery.phala_trusted_app_ids",
                    "entries must use 0x<lowercase hex of 1 to 64 measured app-id bytes>",
                ));
            }
        }
        for digest in &self.phala_trusted_compose_hashes {
            let Some(hex_digest) = digest.strip_prefix("sha256:") else {
                return Err(ServerError::config_invalid(
                    "discovery.phala_trusted_compose_hashes",
                    "entries must use sha256:<64 lowercase hex> format",
                ));
            };
            if hex_digest.len() != 64
                || !hex_digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ServerError::config_invalid(
                    "discovery.phala_trusted_compose_hashes",
                    "entries must use sha256:<64 lowercase hex> format",
                ));
            }
        }
        Ok(())
    }

    /// Validates discovery bootstrap configuration.
    pub fn validate(&self) -> Result<()> {
        self.validate_phala_peer_policy()?;
        if self.phala_attestation_allow_legacy_v0 && self.phala_attestation_socket_path.is_none() {
            return Err(ServerError::config_invalid(
                "discovery.phala_attestation_allow_legacy_v0",
                "requires discovery.phala_attestation_socket_path",
            ));
        }
        if let Some(path) = &self.phala_attestation_socket_path {
            let raw_path = path.as_str();
            let path = path.trim();
            if raw_path != path
                || path.is_empty()
                || path.len() > 4096
                || path.contains('\0')
                || !Path::new(path).is_absolute()
            {
                return Err(ServerError::config_invalid(
                    "discovery.phala_attestation_socket_path",
                    "must be a non-empty absolute Unix socket path of at most 4096 bytes",
                ));
            }
            if !self.enabled
                || !self.advertise_self
                || !self.public_discovery
                || self.public_api_listen_addr.is_none()
            {
                return Err(ServerError::config_invalid(
                    "discovery.phala_attestation_socket_path",
                    "requires enabled self-advertisement, public discovery, and public API listener",
                ));
            }
        }
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
            || self.custody_audit_witness_auto_renewal_enabled
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
            if self.custody_audit_witness_auto_renewal_enabled
                && !self.custody_audit_witness_runtime_required
            {
                return Err(ServerError::config_invalid(
                    "discovery.custody_audit_witness_auto_renewal_enabled",
                    "requires custody_audit_witness_runtime_required = true",
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

        // [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-24 by Codex] Bound all
        // public verifier state and reject partial listener activation.
        if self.permissionless_endpoint_proof_max_entries == 0
            || self.permissionless_endpoint_proof_max_entries
                > MAX_PERMISSIONLESS_ENDPOINT_PROOF_ENTRIES
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_proof_max_entries",
                "must be between 1 and 65536",
            ));
        }
        if self.permissionless_endpoint_proof_ttl_secs == 0
            || self.permissionless_endpoint_proof_ttl_secs
                > MAX_PERMISSIONLESS_ENDPOINT_PROOF_TTL_SECS
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_proof_ttl_secs",
                "must be between 1 and 300 seconds",
            ));
        }
        if self.permissionless_endpoint_proof_enabled {
            if !self.enabled {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_proof_enabled",
                    "requires discovery.enabled = true",
                ));
            }
            if self.public_api_listen_addr.is_none() {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_proof_enabled",
                    "requires discovery.public_api_listen_addr",
                ));
            }
        }
        // [PERMISSIONLESS-ENDPOINT-EVIDENCE 2026-09-24 by Codex] Durable
        // evidence is an independent opt-in, but it may never outlive the
        // authenticated V2 proof surface that supplies its authority.
        if self.permissionless_endpoint_evidence_max_entries == 0
            || self.permissionless_endpoint_evidence_max_entries > 65_536
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_evidence_max_entries",
                "must be between 1 and 65536",
            ));
        }
        if self.permissionless_endpoint_evidence_ttl_secs == 0
            || self.permissionless_endpoint_evidence_ttl_secs > 7 * 24 * 60 * 60
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_evidence_ttl_secs",
                "must be between 1 and 604800 seconds",
            ));
        }
        if self.permissionless_endpoint_evidence_cleanup_batch == 0
            || self.permissionless_endpoint_evidence_cleanup_batch > 4_096
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_evidence_cleanup_batch",
                "must be between 1 and 4096",
            ));
        }
        if self.permissionless_endpoint_evidence_enabled {
            if !self.permissionless_endpoint_proof_enabled {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_evidence_enabled",
                    "requires discovery.permissionless_endpoint_proof_enabled = true",
                ));
            }
            if self
                .permissionless_endpoint_evidence_db_path
                .trim()
                .is_empty()
            {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_evidence_db_path",
                    "must be configured when endpoint evidence is enabled",
                ));
            }
        }
        if self.permissionless_endpoint_attestation_inbox_max_entries == 0
            || self.permissionless_endpoint_attestation_inbox_max_entries > 65_536
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_attestation_inbox_max_entries",
                "must be between 1 and 65536",
            ));
        }
        if self.permissionless_endpoint_attestation_inbox_max_bytes < 289
            || self.permissionless_endpoint_attestation_inbox_max_bytes > 64 * 1024 * 1024
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_attestation_inbox_max_bytes",
                "must be between 289 and 67108864 bytes",
            ));
        }
        if self.permissionless_endpoint_attestation_inbox_ttl_secs == 0
            || self.permissionless_endpoint_attestation_inbox_ttl_secs > 7 * 24 * 60 * 60
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_attestation_inbox_ttl_secs",
                "must be between 1 and 604800 seconds",
            ));
        }
        if self.permissionless_endpoint_attestation_inbox_cleanup_batch == 0
            || self.permissionless_endpoint_attestation_inbox_cleanup_batch > 4_096
        {
            return Err(ServerError::config_invalid(
                "discovery.permissionless_endpoint_attestation_inbox_cleanup_batch",
                "must be between 1 and 4096",
            ));
        }
        if self.permissionless_endpoint_attestation_inbox_enabled {
            if !self.enabled || self.public_api_listen_addr.is_none() {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_attestation_inbox_enabled",
                    "requires discovery.enabled and discovery.public_api_listen_addr",
                ));
            }
            if self
                .permissionless_endpoint_attestation_inbox_db_path
                .trim()
                .is_empty()
            {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_attestation_inbox_db_path",
                    "must be configured when endpoint attestation inbox is enabled",
                ));
            }
        }
        if self.permissionless_endpoint_promotion_enabled {
            if !self.enabled
                || self.public_api_listen_addr.is_none()
                || !self.permissionless_endpoint_proof_enabled
                || !self.permissionless_endpoint_evidence_enabled
                || !self.permissionless_endpoint_attestation_inbox_enabled
            {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_promotion_enabled",
                    "requires public discovery, endpoint proof, evidence and attestation inbox",
                ));
            }
            if self
                .permissionless_endpoint_promotion_db_prefix
                .trim()
                .is_empty()
            {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_promotion_db_prefix",
                    "must be configured when endpoint promotion is enabled",
                ));
            }
            // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A
            // syntactically valid one-row inbox cannot satisfy the mandatory
            // two-observer gate; reject that permanently-unready deployment.
            if self.permissionless_endpoint_attestation_inbox_max_entries < 2
                || self.permissionless_endpoint_attestation_inbox_max_bytes < 2 * 289
                || self.permissionless_endpoint_attestation_inbox_ttl_secs < 120
            {
                return Err(ServerError::config_invalid(
                    "discovery.permissionless_endpoint_promotion_enabled",
                    "requires capacity and retention for two current attestations",
                ));
            }
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

        // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] The two-epoch
        // in-memory KEM store must cover every signed descriptor lifetime.
        if !crate::services::onion_keys::descriptor_ttl_is_supported(self.descriptor_ttl_secs) {
            return Err(ServerError::config_invalid(
                "discovery.descriptor_ttl_secs",
                "must be between 60 and 85799 seconds for bounded onion key overlap",
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
            phala_attestation_socket_path: None,
            phala_attestation_allow_legacy_v0: false,
            phala_attested_peers_required: false,
            phala_trusted_app_ids: Vec::new(),
            phala_trusted_compose_hashes: Vec::new(),
            phala_peer_attestation_max_age_secs: Self::default_phala_peer_attestation_max_age_secs(),
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
            custody_audit_witness_auto_renewal_enabled: false,
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
            permissionless_endpoint_proof_enabled: false,
            permissionless_endpoint_proof_max_entries:
                Self::default_permissionless_endpoint_proof_max_entries(),
            permissionless_endpoint_proof_ttl_secs:
                Self::default_permissionless_endpoint_proof_ttl_secs(),
            permissionless_endpoint_evidence_enabled: false,
            permissionless_endpoint_evidence_db_path: String::new(),
            permissionless_endpoint_evidence_max_entries:
                Self::default_permissionless_endpoint_evidence_max_entries(),
            permissionless_endpoint_evidence_ttl_secs:
                Self::default_permissionless_endpoint_evidence_ttl_secs(),
            permissionless_endpoint_evidence_cleanup_batch:
                Self::default_permissionless_endpoint_evidence_cleanup_batch(),
            permissionless_endpoint_attestation_inbox_enabled: false,
            permissionless_endpoint_attestation_inbox_db_path: String::new(),
            permissionless_endpoint_attestation_inbox_max_entries:
                Self::default_endpoint_attestation_inbox_max_entries(),
            permissionless_endpoint_attestation_inbox_max_bytes:
                Self::default_endpoint_attestation_inbox_max_bytes(),
            permissionless_endpoint_attestation_inbox_ttl_secs:
                Self::default_endpoint_attestation_inbox_ttl_secs(),
            permissionless_endpoint_attestation_inbox_cleanup_batch:
                Self::default_endpoint_attestation_inbox_cleanup_batch(),
            permissionless_endpoint_promotion_enabled: false,
            permissionless_endpoint_promotion_db_prefix: String::new(),
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
    /// [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Intermediate,
    /// recipient-only composition. Queue admission is not enabled by this field.
    #[serde(default)]
    pub reverse_onion: crate::config_reverse_onion::ReverseOnionConfig,
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
    // [PHALA-EFFECTIVE-DISCOVERY-ENDPOINT 2026-10-06 by Codex] Keep the
    // validated and descriptor-advertised public origin on one precedence rule.
    pub(crate) fn effective_public_endpoint(&self) -> Option<&str> {
        self.discovery
            .public_endpoint
            .as_deref()
            .or(self.network.public_endpoint.as_deref())
    }

    /// Load and validate configuration from a TOML file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        let mut config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] The
        // endpoint-free recipient must not inherit public visibility from
        // the peer image's shared configuration template.
        if let Some(value) = optional_config_env("AERONYX_DISCOVERY_PUBLIC_DISCOVERY")? {
            apply_discovery_public_visibility_override(&mut config, &value)?;
        }
        // [PHALA-PRIVATE-API-ISOLATION 2026-10-06 by Codex] The endpoint-free
        // recipient must not inherit the public peer's internal 0.0.0.0
        // listener; an empty value explicitly removes that inbound surface.
        match std::env::var("AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR") {
            Ok(addr) => apply_discovery_api_listener_override(&mut config, &addr)?,
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ServerError::config_load(
                    "AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR",
                    "must be valid UTF-8",
                ));
            }
        }
        // [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Resolve the
        // listener before checking the origin it serves. Empty removes both
        // descriptor endpoint sources; missing preserves mounted TOML. These
        // values configure transport only, never establish attestation trust.
        match std::env::var("AERONYX_DISCOVERY_PUBLIC_ENDPOINT") {
            Ok(endpoint) => {
                apply_discovery_public_endpoint_if_configured(&mut config, &endpoint)?
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ServerError::config_load(
                    "AERONYX_DISCOVERY_PUBLIC_ENDPOINT",
                    "must be valid UTF-8",
                ));
            }
        }
        // [PHALA-DISCOVERY-SEEDS 2026-10-06 by Codex] Public peer containers
        // need operator-selected HTTPS seeds to join outbound gossip; private
        // recipient containers use their separately pinned relay bootstrap.
        // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Missing preserves
        // mounted TOML; explicitly empty clears inherited general peers.
        match std::env::var("AERONYX_DISCOVERY_SEED_ENDPOINTS") {
            Ok(seeds) => apply_discovery_seed_endpoints(&mut config, &seeds)?,
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ServerError::config_load(
                    "AERONYX_DISCOVERY_SEED_ENDPOINTS",
                    "must be valid UTF-8",
                ));
            }
        }
        // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] An explicit,
        // complete protected bundle changes outbound trust, never quote serving
        // or E2EE availability. Blank Compose values preserve mounted TOML.
        let phala_routes_required = optional_config_env("AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED")?;
        let phala_route_apps = optional_config_env("AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS")?;
        let phala_route_compose = optional_config_env("AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES")?;
        let phala_route_age = optional_config_env("AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS")?;
        apply_phala_peer_policy_env(&mut config, phala_routes_required.as_deref(),
            phala_route_apps.as_deref(), phala_route_compose.as_deref(), phala_route_age.as_deref())?;
        // [PHALA-PRIVATE-ATTESTATION-ISOLATION 2026-10-06 by Codex] The
        // endpoint-free recipient process must not inherit quote authority
        // merely because the public peer shares its baked config template.
        match std::env::var("AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH") {
            Ok(path) => apply_phala_attestation_socket_override(&mut config, &path)?,
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ServerError::config_load(
                    "AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH",
                    "must be valid UTF-8",
                ));
            }
        }
        // [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
        // Empty Compose defaults preserve TOML policy; only explicit strings
        // override the default-off recipient role.
        let recipient_enabled = optional_config_env("AERONYX_REVERSE_ONION_RECIPIENT_ENABLED")?;
        let recipient_relay_id = optional_config_env("AERONYX_REVERSE_ONION_RELAY_NODE_ID")?;
        let recipient_relay_endpoint =
            optional_config_env("AERONYX_REVERSE_ONION_RELAY_ENDPOINT")?;
        apply_reverse_onion_recipient_env(
            &mut config,
            recipient_enabled.as_deref(),
            recipient_relay_id.as_deref(),
            recipient_relay_endpoint.as_deref(),
        )?;
        // [PHALA-RECIPIENT-RECOVERY-OVERRIDE 2026-10-06 by Codex] An unset
        // value preserves the baked TOML safety mode; only an explicit value
        // changes whether this recipient may request fresh Claims.
        let recipient_recovery_only = optional_config_env(
            "AERONYX_REVERSE_ONION_RECOVERY_ONLY",
        )?;
        apply_reverse_onion_recovery_only_env(
            &mut config,
            recipient_recovery_only.as_deref(),
        )?;
        // [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] The public Phala
        // identity may offer bounded ciphertext relay, but never infer a VPN
        // role or enable it merely because the image is running in Phala.
        if let Some(enabled) = optional_config_env("AERONYX_PHALA_ONION_RELAY_ENABLED")? {
            apply_phala_onion_relay_override(&mut config, &enabled)?;
        }
        // [PHALA-REVERSE-QUEUE-CONFIG 2026-10-06 by Codex] ChatRelay
        // capability advertisement does not itself mount a reverse task
        // queue. Require a separate opt-in and complete signed admission
        // material for the public relay process.
        let phala_queue_enabled = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED",
        )?;
        // [PHALA-QUEUE-RECOVERY-ENV 2026-10-06 by Codex] Unset preserves the
        // baked recovery policy; the public role can opt in explicitly.
        let phala_queue_recovery_only = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY",
        )?;
        let phala_queue_recipient_ids = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_NODE_IDS",
        )?;
        let phala_queue_source_ids = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_SOURCE_NODE_IDS",
        )?;
        let phala_queue_relay_descriptor = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_RELAY_DESCRIPTOR_B64",
        )?;
        let phala_queue_recipient_descriptor = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_RECIPIENT_DESCRIPTOR_B64",
        )?;
        let phala_queue_authorization = optional_config_env(
            "AERONYX_PHALA_REVERSE_ONION_QUEUE_AUTHORIZATION_B64",
        )?;
        config.reverse_onion.queue.apply_phala_environment(
            phala_queue_enabled.as_deref(),
            phala_queue_recovery_only.as_deref(),
            phala_queue_recipient_ids.as_deref(),
            phala_queue_source_ids.as_deref(),
            phala_queue_relay_descriptor.as_deref(),
            phala_queue_recipient_descriptor.as_deref(),
            phala_queue_authorization.as_deref(),
            config.discovery.advertise_onion_middle
                && config.memchain.chat_relay.enabled,
            config.reverse_onion.recipient.enabled,
        )?;
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
        // [REVERSE-AUTHORITY-RENEWAL-GATE 2026-10-05 by Codex] Live source
        // submissions carry a recipient-signed current authority snapshot;
        // relay admission verifies it against current PeerStore descriptors
        // immediately before durable enqueue. Recovery-only remains explicit.
        // [PHALA-REVERSE-ONION-LISTENER-GATES 2026-10-06 by Codex]
        // Source pulls are composed only on the authenticated VPN/MPI listener.
        if self.reverse_onion.source.enabled && !self.vpn.enabled {
            return Err(ServerError::config_invalid(
                "reverse_onion.source",
                "source pulls require the authenticated VPN/MPI listener",
            ));
        }
        // [PHALA-SOURCE-MPI-COMPOSITION 2026-10-08 by Codex] Off never
        // constructs MPI, and SaaS JWT owners cannot authorize this private
        // source route. Reject before opening/migrating its durable journal.
        if self.reverse_onion.source.enabled
            && !matches!(&self.memchain.mode, MemChainMode::Local | MemChainMode::P2p)
        {
            return Err(ServerError::config_invalid(
                "reverse_onion.source",
                "source pulls require local or p2p MemChain MPI runtime",
            ));
        }
        self.reverse_onion.validate()?;
        // [PHALA-REVERSE-ONION-LISTENER-GATES 2026-10-06 by Codex]
        // Queue frames are a peer protocol on the existing public peer API;
        // source pulls remain client/MPI-only on the VPN listener.
        if self.reverse_onion.queue.enabled
            && self.discovery.public_api_listen_addr.is_none()
        {
            return Err(ServerError::config_invalid(
                "reverse_onion.queue",
                "enabled queue requires discovery.public_api_listen_addr",
            ));
        }
        // [PHALA-REVERSE-QUEUE-ORIGIN-GATE 2026-10-06 by Codex] A bound
        // listener without the signed HTTPS origin is not a routable relay.
        // Phala's first render may omit that origin, so queue activation waits
        // until the operator re-renders with the assigned endpoint.
        if self.reverse_onion.queue.enabled
            && !self.effective_public_endpoint().is_some_and(|endpoint| {
                crate::api::reverse_onion_endpoint_supported(endpoint.trim())
            })
        {
            return Err(ServerError::config_invalid(
                "reverse_onion.queue",
                "enabled queue requires a public HTTPS discovery endpoint",
            ));
        }
        // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex] Fresh grants
        // travel only over the existing authenticated discovery-gossip
        // exchange. Reject configurations that enable new relay, source, or
        // recipient work without that renewal channel; recovery-only needs no
        // renewal.
        let discovery_renewal_ready = self.discovery.enabled && self.discovery.gossip_enabled;
        if self.reverse_onion.requires_live_authority_gossip() && !discovery_renewal_ready {
            return Err(ServerError::config_invalid(
                "reverse_onion",
                "live private admission requires discovery gossip",
            ));
        }
        for (enabled, path) in [
            (self.reverse_onion.queue.enabled, self.reverse_onion.queue.db_path.as_str()),
            (self.reverse_onion.source.enabled, self.reverse_onion.source.state_db_path.as_str()),
        ] {
            if enabled && ([self.memchain.db_path.as_str(), self.memchain.chat_relay.db_path.as_str(),
                self.blind_vault.db_path.as_str(), self.reverse_onion.recipient.state_db_path.as_str()]
                .into_iter().any(|other| !other.is_empty() && other.trim() == path)
                || self.discovery.directory_chain_path.as_deref().is_some_and(|other| other.trim() == path))
            {
                return Err(ServerError::config_invalid("reverse_onion", "reverse role requires a dedicated database"));
            }
        }
        if self.reverse_onion.recipient.enabled {
            // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] The recipient
            // serves source-sealed Pulls through the existing authenticated
            // peer route; it does not require the separate direct HTTP API.
            // [PHALA-PRIVATE-RECIPIENT-DESCRIPTOR 2026-10-06 by Codex] The
            // signed recipient descriptor is intentionally non-public. Reject
            // endpoint publication here instead of starting a worker whose
            // authority can never satisfy the route's private-recipient gate.
            // [PHALA-PRIVATE-RECIPIENT-NO-VPN 2026-10-06 by Codex] This role is
            // an outbound task worker, not a VPN/TUN data-plane node.
            if self.vpn.enabled {
                return Err(ServerError::config_invalid(
                    "vpn.enabled",
                    "private recipient cannot enable the VPN/TUN data plane",
                ));
            }
            // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] The
            // management client and its public-IP discovery probes create
            // independent outbound destinations, outside the pinned relay.
            if self.management.enabled {
                return Err(ServerError::config_invalid(
                    "management.enabled",
                    "private recipient management must remain disabled outside its pinned-relay egress profile",
                ));
            }
            // [PHALA-PRIVATE-RECIPIENT-SERVICE-ISOLATION 2026-10-06 by Codex]
            // ChatRelay and Blind Vault are independent local stores; the
            // MemChain runtime can expose APIs or start unrelated peer workers.
            if self.memchain.mode != crate::config_memchain::MemChainMode::Off {
                return Err(ServerError::config_invalid(
                    "memchain.mode",
                    "private recipient requires MemChain mode off; ChatRelay remains independently enabled",
                ));
            }
            let advertises_public_endpoint = self
                .discovery
                .public_endpoint
                .as_deref()
                .is_some_and(|endpoint| !endpoint.trim().is_empty())
                || self
                    .network
                    .public_endpoint
                    .as_deref()
                    .is_some_and(|endpoint| !endpoint.trim().is_empty());
            if advertises_public_endpoint {
                return Err(ServerError::config_invalid(
                    "reverse_onion.recipient",
                    "private recipient identity cannot advertise a public endpoint",
                ));
            }
            if self.discovery.public_discovery {
                return Err(ServerError::config_invalid(
                    "discovery.public_discovery",
                    "private recipient identity cannot be publicly discoverable",
                ));
            }
            if self.discovery.public_api_listen_addr.is_some() {
                return Err(ServerError::config_invalid(
                    "reverse_onion.recipient",
                    "private recipient identity cannot listen for inbound peer API traffic",
                ));
            }
            // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] The
            // recipient's only peer destination is its identity-pinned relay.
            // Reject independent bootstrap and replication destinations even
            // though the gossip runtime also applies a relay-only target policy.
            if self.discovery.bootstrap_snapshot_url.is_some() {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_url",
                    "private recipient may bootstrap only through its pinned relay",
                ));
            }
            if !self.discovery.seed_endpoints.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.seed_endpoints",
                    "private recipient may gossip only with its pinned relay",
                ));
            }
            if !self.discovery.directory_chain_sync_peer_node_ids.is_empty()
                || self.discovery.directory_full_node_mirror_enabled
            {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_sync_peer_node_ids",
                    "private recipient cannot run outbound Directory Replica Sync",
                ));
            }
            if self.memchain.commitment_sync_enabled || self.memchain.commitment_coordinator_enabled
            {
                return Err(ServerError::config_invalid(
                    "memchain.commitment_sync_enabled",
                    "private recipient cannot run outbound MemChain commitment synchronization",
                ));
            }
            // [PHALA-PRIVATE-RECIPIENT-API-GATE 2026-10-06 by Codex]
            // TOML loading must enforce the same private-only boundary as the
            // environment bootstrap helper; callers can construct ServerConfig
            // directly and must not turn this terminal into a general API.
            if self.blind_vault.public_api_enabled {
                return Err(ServerError::config_invalid(
                    "blind_vault.public_api_enabled",
                    "private recipient requires the public Blind Vault API to remain disabled",
                ));
            }
            if !self.blind_vault.enabled
                || !self.memchain.is_chat_relay_enabled()
            {
                return Err(ServerError::config_invalid(
                    "reverse_onion",
                    "recipient requires Blind Vault storage and local ChatRelay durability",
                ));
            }
            let path = self.reverse_onion.recipient.state_db_path.trim();
            if [self.memchain.db_path.as_str(), self.memchain.chat_relay.db_path.as_str(),
                self.blind_vault.db_path.as_str(), self.reverse_onion.queue.db_path.as_str()]
                .into_iter().any(|other| !other.is_empty() && other.trim() == path)
                || self.discovery.directory_chain_path.as_deref().is_some_and(|other| other.trim() == path)
            {
                return Err(ServerError::config_invalid("reverse_onion", "recipient requires a dedicated private database"));
            }
        }
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;
        self.management
            .validate()
            .map_err(|e| ServerError::config_invalid("management", e))?;
        self.memchain.validate()?;
        self.discovery.validate()?;
        // [PHALA-ATTESTATION-ENDPOINT-GATE 2026-10-06 by Codex] An attested
        // public peer must have the HTTPS origin clients will actually dial;
        // otherwise it starts an unreachable quote API and cannot advertise
        // the signed Phala feature. Honor the descriptor's existing fallback.
        if self.discovery.phala_attestation_socket_path.is_some() {
            let endpoint = self.effective_public_endpoint();
            if endpoint.is_none_or(|endpoint| {
                !crate::api::reverse_onion_endpoint_supported(endpoint)
            }) {
                return Err(ServerError::config_invalid(
                    "discovery.phala_attestation_socket_path",
                    "requires a public HTTPS discovery endpoint",
                ));
            }
        }
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

    // [PHALA-NO-VPN-PROFILE 2026-10-06 by Codex] Relay profiles preserve the
    // legacy default while exposing the explicit no-tunnel runtime gate.
    /// Returns whether the UDP/TUN VPN data plane is enabled on this node.
    #[must_use]
    pub fn vpn_enabled(&self) -> bool {
        self.vpn.enabled
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

// [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] An explicit empty
// Compose origin must suppress both mounted endpoints, including the legacy
// network fallback. Missing environment still preserves both at the caller.
fn apply_discovery_public_endpoint_if_configured(
    config: &mut ServerConfig,
    endpoint: &str,
) -> Result<()> {
    if endpoint.is_empty() {
        config.discovery.public_endpoint = None;
        config.network.public_endpoint = None;
        return Ok(());
    }
    apply_discovery_public_endpoint_override(config, endpoint)
}

// [PHALA-PUBLIC-ENDPOINT-OVERRIDE 2026-10-06 by Codex]
fn apply_discovery_public_endpoint_override(
    config: &mut ServerConfig,
    endpoint: &str,
) -> Result<()> {
    // [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Compose retains
    // the protected value verbatim; validate that same value without trimming.
    let parsed = reqwest::Url::parse(endpoint).ok();
    if endpoint.is_empty()
        || endpoint.len() > 2048
        || endpoint.contains(char::is_whitespace)
        || !config.discovery.enabled
        || !config.discovery.advertise_self
        || config.discovery.public_api_listen_addr.is_none()
        || parsed.as_ref().is_none_or(|url| {
            url.scheme() != "https"
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || !matches!(url.path(), "" | "/")
                || url.query().is_some()
                || url.fragment().is_some()
        })
        || !crate::api::reverse_onion_endpoint_supported(endpoint)
    {
        return Err(ServerError::config_invalid(
            "AERONYX_DISCOVERY_PUBLIC_ENDPOINT",
            "requires enabled self-advertisement, a peer API listener, and a public HTTPS origin",
        ));
    }
    config.discovery.public_endpoint = Some(endpoint.to_string());
    Ok(())
}

// [PHALA-PRIVATE-API-ISOLATION 2026-10-06 by Codex]
fn apply_discovery_api_listener_override(config: &mut ServerConfig, addr: &str) -> Result<()> {
    if addr.is_empty() {
        config.discovery.public_api_listen_addr = None;
        return Ok(());
    }
    let parsed = addr.parse::<SocketAddr>().map_err(|_| {
        ServerError::config_invalid(
            "AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR",
            "must be an IP socket address or empty to disable the listener",
        )
    })?;
    if parsed.port() == 0 {
        return Err(ServerError::config_invalid(
            "AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR",
            "port must be greater than zero",
        ));
    }
    config.discovery.public_api_listen_addr = Some(parsed);
    Ok(())
}

// [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
fn apply_discovery_public_visibility_override(
    config: &mut ServerConfig,
    value: &str,
) -> Result<()> {
    config.discovery.public_discovery = match value {
        "true" => true,
        "false" => false,
        _ => {
            return Err(ServerError::config_invalid(
                "AERONYX_DISCOVERY_PUBLIC_DISCOVERY",
                "must be exactly 'true' or 'false'",
            ));
        }
    };
    Ok(())
}

// [PHALA-DISCOVERY-SEEDS 2026-10-06 by Codex] JSON avoids delimiter parsing
// ambiguity and keeps the deployment value bounded before allocation/validation.
fn apply_discovery_seed_endpoints(config: &mut ServerConfig, encoded: &str) -> Result<()> {
    // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Clearing outbound
    // seeds requires no enabled gossip transport, including recovery startup.
    if encoded.is_empty() {
        config.discovery.seed_endpoints.clear();
        return Ok(());
    }
    if encoded.len() > 8 * 1024 || !config.discovery.enabled || !config.discovery.gossip_enabled {
        return Err(ServerError::config_invalid(
            "AERONYX_DISCOVERY_SEED_ENDPOINTS",
            "requires enabled discovery gossip and at most 8192 bytes",
        ));
    }
    let endpoints: Vec<String> = serde_json::from_str(encoded).map_err(|_| {
        ServerError::config_invalid(
            "AERONYX_DISCOVERY_SEED_ENDPOINTS",
            "must be a JSON array of public HTTPS peer origins",
        )
    })?;
    if endpoints.is_empty() || endpoints.len() > 64 {
        return Err(ServerError::config_invalid(
            "AERONYX_DISCOVERY_SEED_ENDPOINTS",
            "requires 1..=64 peer origins",
        ));
    }
    for endpoint in &endpoints {
        let parsed = reqwest::Url::parse(endpoint).ok();
        if endpoint.len() > 2048
            || endpoint.trim() != endpoint
            || parsed.as_ref().is_none_or(|url| {
                url.scheme() != "https"
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || !matches!(url.path(), "" | "/")
                    || url.query().is_some()
                    || url.fragment().is_some()
            })
            || !crate::api::reverse_onion_endpoint_supported(endpoint)
        {
            return Err(ServerError::config_invalid(
                "AERONYX_DISCOVERY_SEED_ENDPOINTS",
                "entries must be bounded public HTTPS peer origins without credentials",
            ));
        }
        DiscoveryConfig::validate_seed_endpoint(endpoint)?;
    }
    config.discovery.seed_endpoints = endpoints;
    Ok(())
}

// [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] These are local operator
// pins, never values copied from quote/discovery responses. Validate a complete
// candidate before changing any field; invalid input must not clear strict mode.
fn apply_phala_peer_policy_env(config: &mut ServerConfig, required: Option<&str>,
    apps: Option<&str>, compose: Option<&str>, age: Option<&str>) -> Result<()> {
    let invalid = || ServerError::config_invalid("discovery.phala_peer_policy",
        "protected policy needs explicit true with both bounded JSON pin lists, or false without policy inputs");
    let Some(required) = required else {
        if apps.is_some() || compose.is_some() || age.is_some() { return Err(invalid()); }
        return Ok(());
    };
    let mut candidate = config.discovery.clone();
    match required {
        "false" if apps.is_none() && compose.is_none() && age.is_none() => {
            candidate.phala_attested_peers_required = false;
        }
        "true" => {
            let pins = |raw: Option<&str>| -> Result<Vec<String>> {
                let raw = raw.ok_or_else(&invalid)?;
                if raw.len() > 8192 { return Err(invalid()); }
                let values: Vec<String> = serde_json::from_str(raw).map_err(|_| invalid())?;
                if values.is_empty() || values.len() > 64
                    || values.iter().collect::<std::collections::HashSet<_>>().len() != values.len()
                { return Err(invalid()); }
                Ok(values)
            };
            candidate.phala_trusted_app_ids = pins(apps)?;
            candidate.phala_trusted_compose_hashes = pins(compose)?;
            if let Some(age) = age {
                if age.is_empty() || age.len() > 5 || age.starts_with('0')
                    || !age.bytes().all(|byte| byte.is_ascii_digit())
                { return Err(invalid()); }
                candidate.phala_peer_attestation_max_age_secs = age.parse().map_err(|_| invalid())?;
            }
            candidate.phala_attested_peers_required = true;
        }
        _ => return Err(invalid()),
    }
    candidate.validate_phala_peer_policy()?;
    config.discovery = candidate;
    Ok(())
}

// [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
fn optional_config_env(name: &'static str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(ServerError::config_load(
            name,
            "must be valid UTF-8",
        )),
    }
}

// [PHALA-PRIVATE-ATTESTATION-ISOLATION 2026-10-06 by Codex]
fn apply_phala_attestation_socket_override(config: &mut ServerConfig, path: &str) -> Result<()> {
    if path.is_empty() {
        config.discovery.phala_attestation_socket_path = None;
        config.discovery.phala_attestation_allow_legacy_v0 = false;
    } else {
        config.discovery.phala_attestation_socket_path = Some(path.to_owned());
    }
    config.discovery.validate()
}

// [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
fn apply_reverse_onion_recipient_env(
    config: &mut ServerConfig,
    enabled: Option<&str>,
    relay_node_id: Option<&str>,
    relay_endpoint: Option<&str>,
) -> Result<()> {
    let Some(enabled) = enabled else {
        if relay_node_id.is_some() || relay_endpoint.is_some() {
            return Err(ServerError::config_invalid(
                "reverse_onion.recipient",
                "relay identity and endpoint require an explicit enable setting",
            ));
        }
        return Ok(());
    };
    let enabled = match enabled {
        "true" => true,
        "false" => false,
        _ => {
            return Err(ServerError::config_invalid(
                "AERONYX_REVERSE_ONION_RECIPIENT_ENABLED",
                "must be exactly 'true' or 'false'",
            ));
        }
    };
    if !enabled {
        if relay_node_id.is_some() || relay_endpoint.is_some() {
            return Err(ServerError::config_invalid(
                "reverse_onion.recipient",
                "relay identity and endpoint are not accepted while disabled",
            ));
        }
        config.reverse_onion.recipient.enabled = false;
        return Ok(());
    }

    let relay_node_id = relay_node_id.filter(|value| !value.is_empty()).ok_or_else(|| {
        ServerError::config_invalid(
            "AERONYX_REVERSE_ONION_RELAY_NODE_ID",
            "is required when the private recipient role is enabled",
        )
    })?;
    let relay_endpoint = relay_endpoint.filter(|value| !value.is_empty()).ok_or_else(|| {
        ServerError::config_invalid(
            "AERONYX_REVERSE_ONION_RELAY_ENDPOINT",
            "is required when the private recipient role is enabled",
        )
    })?;
    if config.blind_vault.public_api_enabled {
        return Err(ServerError::config_invalid(
            "blind_vault.public_api_enabled",
            "Phala private recipient bootstrap does not enable public Blind Vault routes",
        ));
    }

    config.reverse_onion.recipient.enabled = true;
    config.reverse_onion.recipient.relay_node_id = relay_node_id.to_owned();
    config.reverse_onion.recipient.relay_endpoint = relay_endpoint.to_owned();
    config.blind_vault.enabled = true;
    config.memchain.chat_relay.enabled = true;
    config.memchain.chat_relay.validate()?;
    Ok(())
}

// [PHALA-RECIPIENT-RECOVERY-OVERRIDE 2026-10-06 by Codex]
fn apply_reverse_onion_recovery_only_env(
    config: &mut ServerConfig,
    value: Option<&str>,
) -> Result<()> {
    let Some(value) = value else { return Ok(()); };
    if !config.reverse_onion.recipient.enabled {
        return Err(ServerError::config_invalid(
            "AERONYX_REVERSE_ONION_RECOVERY_ONLY",
            "requires the private recipient role to be enabled",
        ));
    }
    config.reverse_onion.recipient.recovery_only = match value {
        "true" => true,
        "false" => false,
        _ => return Err(ServerError::config_invalid(
            "AERONYX_REVERSE_ONION_RECOVERY_ONLY",
            "must be exactly 'true' or 'false'",
        )),
    };
    Ok(())
}

// [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex]
fn apply_phala_onion_relay_override(config: &mut ServerConfig, value: &str) -> Result<()> {
    match value {
        "true" => {
            if config.reverse_onion.recipient.enabled {
                return Err(ServerError::config_invalid(
                    "AERONYX_PHALA_ONION_RELAY_ENABLED",
                    "public onion relay and endpoint-free private recipient are separate roles",
                ));
            }
            config.memchain.chat_relay.enabled = true;
            config.memchain.chat_relay.validate()?;
            config.discovery.advertise_onion_middle = true;
        }
        "false" => {
            // [PHALA-PRIVATE-RELAY-DURABILITY 2026-10-06 by Codex] Private
            // recipient polling uses ChatRelay's local durable store, but
            // must never advertise this endpoint-free identity as a relay.
            // For public peers, explicit false closes both the service and
            // its signed capability even when mounted TOML enables it.
            if !config.reverse_onion.recipient.enabled {
                config.memchain.chat_relay.enabled = false;
            }
            config.discovery.advertise_onion_middle = false;
        }
        _ => {
            return Err(ServerError::config_invalid(
                "AERONYX_PHALA_ONION_RELAY_ENABLED",
                "must be exactly 'true' or 'false'",
            ));
        }
    }
    Ok(())
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            reverse_onion: crate::config_reverse_onion::ReverseOnionConfig::default(),
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

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Authored only: an
    // endpoint-free role can enforce outbound trust without quote authority.
    fn phala_policy_fixture() -> ServerConfig {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.phala_attested_peers_required = true;
        config.discovery.phala_trusted_app_ids = vec!["0xab".into()];
        config.discovery.phala_trusted_compose_hashes = vec![format!("sha256:{}", "a".repeat(64))];
        config.discovery.phala_peer_attestation_max_age_secs = 42;
        config
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Empty protected
    // environment is normalized to None by optional_config_env, never false.
    #[test]
    fn phala_policy_env_preserves_toml_and_changes_only_explicit_policy() {
        let mut config = phala_policy_fixture();
        let before = serde_json::to_value(&config).unwrap();
        apply_phala_peer_policy_env(&mut config, None, None, None, None).unwrap();
        assert_eq!(serde_json::to_value(&config).unwrap(), before);
        let apps = r#"["0xcd"]"#;
        let compose = format!(r#"["sha256:{}"]"#, "b".repeat(64));
        let mut expected = config.clone();
        expected.discovery.phala_trusted_app_ids = vec!["0xcd".into()];
        expected.discovery.phala_trusted_compose_hashes = vec![format!("sha256:{}", "b".repeat(64))];
        apply_phala_peer_policy_env(&mut config, Some("true"), Some(apps), Some(&compose), None).unwrap();
        assert_eq!(serde_json::to_value(&config).unwrap(), serde_json::to_value(&expected).unwrap());
        for age in ["1", "86400"] {
            apply_phala_peer_policy_env(&mut config, Some("true"), Some(apps), Some(&compose), Some(age)).unwrap();
            expected.discovery.phala_peer_attestation_max_age_secs = age.parse().unwrap();
            assert_eq!(serde_json::to_value(&config).unwrap(), serde_json::to_value(&expected).unwrap());
        }
        apply_phala_peer_policy_env(&mut config, Some("false"), None, None, None).unwrap();
        expected.discovery.phala_attested_peers_required = false;
        assert_eq!(serde_json::to_value(&config).unwrap(), serde_json::to_value(&expected).unwrap());
        assert!(config.discovery.phala_attestation_socket_path.is_none());
        assert!(!config.discovery.phala_attestation_allow_legacy_v0);
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] A rejected bundle
    // must leave every config field intact, including an inherited strict gate.
    #[test]
    fn phala_policy_env_rejects_partial_noncanonical_and_unbounded_inputs_atomically() {
        let apps = r#"["0xcd"]"#;
        let compose = format!(r#"["sha256:{}"]"#, "b".repeat(64));
        let mut config = phala_policy_fixture();
        let before = serde_json::to_value(&config).unwrap();
        let mut reject = |required, app_pins, compose_pins, age| {
            assert!(apply_phala_peer_policy_env(&mut config, required, app_pins, compose_pins, age).is_err());
            assert_eq!(serde_json::to_value(&config).unwrap(), before);
        };
        for flag in ["", "TRUE", "1", " true", "false "] {
            reject(Some(flag), Some(apps), Some(&compose), None);
        }
        reject(None, Some(apps), None, None);
        reject(None, None, Some(&compose), None);
        reject(None, None, None, Some("42"));
        reject(Some("true"), None, Some(&compose), None);
        reject(Some("true"), Some(apps), None, None);
        reject(Some("false"), Some(apps), None, None);
        reject(Some("false"), None, Some(&compose), None);
        reject(Some("false"), None, None, Some("42"));
        let oversized = format!("{}{}", " ".repeat(8192), apps);
        let too_many_apps = serde_json::to_string(&(0..65).map(|n| format!("0x{n:02x}")).collect::<Vec<_>>()).unwrap();
        for bad in ["", "[]", "{}", "null", "[1]", r#"["0xcd",null]"#,
            r#"["0xcd","\u0030xcd"]"#, r#"["0xCD"]"#, r#"["0xabc"]"#,
            r#"["0xgg"]"#, r#"["0x"]"#, r#"[" 0xcd"]"#, r#"["0xcd"] trailing"#,
            oversized.as_str(), too_many_apps.as_str()] {
            reject(Some("true"), Some(bad), Some(&compose), None);
        }
        let duplicate_compose = format!(r#"["sha256:{0}","\u0073ha256:{0}"]"#, "b".repeat(64));
        let too_many_compose = serde_json::to_string(&(0..65).map(|n| format!("sha256:{n:064x}")).collect::<Vec<_>>()).unwrap();
        let uppercase_compose = format!(r#"["sha256:{}"]"#, "B".repeat(64));
        let oversized_compose = format!("{}{}", " ".repeat(8192), compose);
        for bad in ["[]", "[true]", r#"["sha256:ab"]"#, duplicate_compose.as_str(),
            too_many_compose.as_str(), uppercase_compose.as_str(), oversized_compose.as_str()] {
            reject(Some("true"), Some(apps), Some(bad), None);
        }
        for bad in ["", "0", "01", "+1", " 1", "1 ", "86401", "999999", "1.0"] {
            reject(Some("true"), Some(apps), Some(&compose), Some(bad));
        }
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Validate discovery
    // prerequisites now, but leave socket/listener validation to role overrides.
    #[test]
    fn phala_policy_env_requires_discovery_but_does_not_grant_listener_authority() {
        let apps = r#"["0xcd"]"#;
        let compose = format!(r#"["sha256:{}"]"#, "b".repeat(64));
        for gossip_disabled in [false, true] {
            let mut config = phala_policy_fixture();
            if gossip_disabled { config.discovery.gossip_enabled = false; }
            else { config.discovery.enabled = false; }
            let before = serde_json::to_value(&config).unwrap();
            assert!(apply_phala_peer_policy_env(&mut config, Some("true"), Some(apps), Some(&compose), None).is_err());
            assert_eq!(serde_json::to_value(&config).unwrap(), before);
        }
        let mut config: ServerConfig = toml::from_str(include_str!(
            "../../../deploy/node/server.phala.peer.example.toml")).unwrap();
        // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] The
        // endpoint fallback is owned by ServerConfig, not DiscoveryConfig.
        assert!(config.validate().is_err());
        apply_phala_peer_policy_env(&mut config, Some("true"), Some(apps), Some(&compose), None).unwrap();
        assert!(config.validate().is_err());
        apply_phala_attestation_socket_override(&mut config, "").unwrap();
        apply_discovery_api_listener_override(&mut config, "").unwrap();
        assert!(config.discovery.validate().is_ok());
        assert!(config.discovery.phala_attested_peers_required);
        assert!(config.discovery.public_endpoint.is_none());
        assert!(config.discovery.public_api_listen_addr.is_none());
        assert!(config.discovery.phala_attestation_socket_path.is_none());
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Authored only: use
    // the actual renderer without launching Compose, containers or services.
    #[cfg(unix)]
    fn render_phala_route_policy(modes: &[&str], required: &str, apps: &str,
        compose: &str, age: &str) -> std::process::Output {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/node/prepare-phala-compose.sh");
        std::process::Command::new("/bin/bash").arg(script).args(modes)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
            .env("AERONYX_NODE_IMAGE", format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)))
            .env("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", "https://peer.aeronyx.network")
            .env("AERONYX_DISCOVERY_SEED_ENDPOINTS", r#"["https://seed.aeronyx.network"]"#)
            .env("AERONYX_REVERSE_ONION_RELAY_NODE_ID", "11".repeat(32))
            .env("AERONYX_REVERSE_ONION_RELAY_ENDPOINT", "https://relay.aeronyx.network")
            .env("AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED", required)
            .env("AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS", apps)
            .env("AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES", compose)
            .env("AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS", age)
            .output().expect("Phala renderer should launch")
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Independently check
    // both process environments for every supported role selection.
    #[cfg(unix)]
    #[test]
    fn phala_route_policy_renderer_preserves_optional_policy_in_each_role() {
        let names = ["AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED",
            "AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS",
            "AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES",
            "AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS"];
        let compose = format!(r#"["sha256:{}"]"#, "b".repeat(64));
        let selections: &[&[&str]] = &[&[], &["--public-peer"], &["--private-recipient"],
            &["--public-peer", "--private-recipient"], &["--private-recipient", "--public-peer"]];
        for modes in selections {
            for (required, apps, hashes, age) in [("", "", "", ""), ("false", "", "", ""),
                ("true", r#"["0xcd"]"#, compose.as_str(), "86400")] {
                let output = render_phala_route_policy(modes, required, apps, hashes, age);
                assert!(output.status.success(), "{modes:?}: {}", String::from_utf8_lossy(&output.stderr));
                let rendered = String::from_utf8(output.stdout).unwrap();
                let (public, remaining) = rendered.split_once("  aeronyx-private-recipient:\n").unwrap();
                let private = remaining.split_once("\nvolumes:\n").unwrap().0;
                for service in [public, private] {
                    for name in names {
                        let key = format!("      {name}:");
                        assert_eq!(service.matches(key.as_str()).count(), 1);
                        assert!(service.contains(&format!("      {name}: \"${{{name}:-}}\"\n")));
                    }
                }
                assert!(!private.contains("    ports:\n"));
                assert!(!private.contains("source: /var/run/dstack.sock"));
                assert!(private.contains("AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR: \"\""));
            }
        }
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Invalid protected
    // inputs reject before any partial manifest is emitted. Not executed.
    #[cfg(unix)]
    #[test]
    fn phala_route_policy_renderer_rejects_incomplete_and_invalid_policy() {
        let apps = r#"["0xcd"]"#;
        let compose = format!(r#"["sha256:{}"]"#, "b".repeat(64));
        let oversized = format!("{}{}", " ".repeat(8192), apps);
        let too_many = serde_json::to_string(&(0..65).map(|n| format!("0x{n:02x}")).collect::<Vec<_>>()).unwrap();
        for (required, pins, hashes, age) in [("", apps, "", ""), ("false", "", "", "1"),
            ("TRUE", apps, compose.as_str(), ""), ("true", "", compose.as_str(), ""),
            ("true", apps, "", ""), ("true", "[1]", compose.as_str(), ""),
            ("true", r#"["0xcd","\u0030xcd"]"#, compose.as_str(), ""),
            ("true", r#"["0xCD"]"#, compose.as_str(), ""),
            ("true", oversized.as_str(), compose.as_str(), ""),
            ("true", too_many.as_str(), compose.as_str(), ""),
            ("true", apps, "[]", ""), ("true", apps, compose.as_str(), "01"),
            ("true", apps, compose.as_str(), "86401")] {
            let output = render_phala_route_policy(&[], required, pins, hashes, age);
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
        }
    }

    // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Authored only: a
    // role-local omission/duplicate/default must reject even when the other
    // service keeps the template-wide count apparently correct.
    #[cfg(unix)]
    #[test]
    fn phala_route_policy_renderer_rejects_weakened_role_environment() {
        let script = include_str!("../../../deploy/node/prepare-phala-compose.sh");
        let template = include_str!("../../../deploy/node/compose.phala.peer.yaml");
        let (public, private) = template.split_once("  aeronyx-private-recipient:\n").unwrap();
        let directory = tempfile::Builder::new().prefix("phala-route-policy-")
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let script_path = directory.path().join("prepare-phala-compose.sh");
        let template_path = directory.path().join("compose.phala.peer.yaml");
        std::fs::write(&script_path, script).unwrap();
        let render = |contents: &str| {
            std::fs::write(&template_path, contents).unwrap();
            std::process::Command::new("/bin/bash").arg(&script_path)
                .env_clear()
                .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
                .env("AERONYX_NODE_IMAGE", format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)))
                .output().expect("Phala renderer should launch")
        };
        let healthy = render(template);
        assert!(healthy.status.success(), "{}", String::from_utf8_lossy(&healthy.stderr));
        for name in ["AERONYX_DISCOVERY_PHALA_ATTESTED_PEERS_REQUIRED",
            "AERONYX_DISCOVERY_PHALA_TRUSTED_APP_IDS",
            "AERONYX_DISCOVERY_PHALA_TRUSTED_COMPOSE_HASHES",
            "AERONYX_DISCOVERY_PHALA_PEER_ATTESTATION_MAX_AGE_SECS"] {
            let original = format!("      {name}: \"${{{name}:-}}\"\n");
            for replacement in [String::new(), original.repeat(2),
                format!("      {name}: \"${{{name}:-false}}\"\n")] {
                for private_role in [false, true] {
                    let mutated = if private_role {
                        format!("{public}  aeronyx-private-recipient:\n{}", private.replacen(original.as_str(), &replacement, 1))
                    } else {
                        format!("{}  aeronyx-private-recipient:\n{private}", public.replacen(original.as_str(), &replacement, 1))
                    };
                    let rejected = render(&mutated);
                    assert!(!rejected.status.success(), "{name} private={private_role}");
                    assert!(rejected.stdout.is_empty());
                }
            }
            let moved = format!("{}  aeronyx-private-recipient:\n{}",
                public.replacen(original.as_str(), &original.repeat(2), 1), private.replacen(original.as_str(), "", 1));
            let rejected = render(&moved);
            assert!(!rejected.status.success());
            assert!(rejected.stdout.is_empty());
        }
    }

    // [PHALA-NODE-ATTESTATION-CONFIG 2026-10-06 by Codex] Opt-in socket
    // configuration must remain default-off and require a public signed
    // discovery identity before it can advertise the endpoint.
    #[test]
    fn phala_node_attestation_socket_is_default_off_and_publicly_gated() {
        let mut config = DiscoveryConfig::default();
        assert!(config.phala_attestation_socket_path.is_none());
        assert!(!config.phala_attestation_allow_legacy_v0);
        config.phala_attestation_allow_legacy_v0 = true;
        assert!(config.validate().is_err());
        config.phala_attestation_allow_legacy_v0 = false;

        config.phala_attestation_socket_path = Some("/var/run/dstack.sock".into());
        assert!(config.validate().is_err());
        config.enabled = true;
        config.advertise_self = true;
        config.public_discovery = true;
        config.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.public_endpoint = Some("https://node.example.com".into());
        assert!(config.validate().is_ok());

        config.phala_attestation_socket_path = Some("relative/dstack.sock".into());
        assert!(config.validate().is_err());
        config.phala_attestation_socket_path = Some(" /var/run/dstack.sock ".into());
        assert!(config.validate().is_err());
    }

    // [PHALA-PEER-ATTESTATION-POLICY 2026-10-06 by Codex] Route trust must
    // fail closed when enabled without locally supplied measured identity.
    #[test]
    fn phala_peer_route_gate_requires_local_app_and_compose_pins() {
        let mut config = DiscoveryConfig::default();
        assert!(!config.phala_attested_peers_required);
        config.phala_attested_peers_required = true;
        config.enabled = true;
        config.gossip_enabled = true;
        assert!(config.validate().is_err());

        config.phala_trusted_app_ids = vec![format!("0x{}", "ab".repeat(20))];
        config.phala_trusted_compose_hashes = vec![format!("sha256:{}", "ab".repeat(32))];
        assert!(config.validate().is_ok());

        // [PHALA-APP-ID-PIN-FORMAT 2026-10-06 by Codex] Pins are canonical
        // lower-case hex encodings of the measured event payload bytes.
        config.phala_trusted_app_ids = vec!["0xAB".into()];
        assert!(config.validate().is_err());
        config.phala_trusted_app_ids = vec!["0xabc".into()];
        assert!(config.validate().is_err());
        config.phala_trusted_app_ids = vec!["0xgg".into()];
        assert!(config.validate().is_err());
        config.phala_trusted_app_ids = vec![format!("0x{}", "ab".repeat(20))];

        config.phala_trusted_compose_hashes = vec!["sha256:AB".into()];
        assert!(config.validate().is_err());
        config.phala_trusted_compose_hashes = vec![format!("sha256:{}", "ab".repeat(32))];
        config.phala_peer_attestation_max_age_secs = 0;
        assert!(config.validate().is_err());
    }

    // [PHALA-PRIVATE-ATTESTATION-ISOLATION 2026-10-06 by Codex]
    #[test]
    fn empty_attestation_socket_override_disables_quote_and_legacy_fallback() {
        let mut config = ServerConfig::default();
        config.discovery.phala_attestation_socket_path = Some("/var/run/dstack.sock".into());
        config.discovery.phala_attestation_allow_legacy_v0 = true;

        apply_phala_attestation_socket_override(&mut config, "").unwrap();

        assert!(config.discovery.phala_attestation_socket_path.is_none());
        assert!(!config.discovery.phala_attestation_allow_legacy_v0);
    }

    // [PHALA-PRIVATE-API-ISOLATION 2026-10-06 by Codex]
    #[test]
    fn empty_peer_api_listener_override_disables_inbound_listener() {
        let mut config = ServerConfig::default();
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());

        apply_discovery_api_listener_override(&mut config, "").unwrap();

        assert!(config.discovery.public_api_listen_addr.is_none());
        assert!(apply_discovery_api_listener_override(&mut config, "0.0.0.0:0").is_err());
        assert!(apply_discovery_api_listener_override(&mut config, "localhost:8422").is_err());
    }

    // [PHALA-PRIVATE-SEED-RENDER-REGRESSION 2026-10-06 by Codex] Exercise
    // the actual profile renderer with an inherited seed value: private role
    // output must succeed while replacing that value with an explicit blank.
    #[cfg(unix)]
    #[test]
    fn phala_private_recipient_renderer_clears_inherited_discovery_seeds() {
        use std::process::Command;

        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/node/prepare-phala-compose.sh");
        let path = std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into());
        let output = Command::new("/bin/bash")
            .arg(script)
            .arg("--private-recipient")
            .env_clear()
            .env("PATH", path)
            .env(
                "AERONYX_NODE_IMAGE",
                format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)),
            )
            .env("AERONYX_REVERSE_ONION_RELAY_NODE_ID", "11".repeat(32))
            .env(
                "AERONYX_REVERSE_ONION_RELAY_ENDPOINT",
                "https://relay.aeronyx.network",
            )
            .env(
                "AERONYX_DISCOVERY_SEED_ENDPOINTS",
                r#"["https://seed.attacker.net"]"#,
            )
            .output()
            .expect("Phala Compose renderer should launch");
        assert!(
            output.status.success(),
            "renderer rejected private role: {}",
            String::from_utf8_lossy(&output.stderr),
        );
        let rendered = String::from_utf8(output.stdout).expect("renderer output is UTF-8");
        let private_service = rendered
            .split("  aeronyx-private-recipient:\n")
            .nth(1)
            .expect("private recipient service is rendered")
            .split("\nvolumes:\n")
            .next()
            .expect("private service ends before top-level volumes");
        assert!(private_service.contains("AERONYX_DISCOVERY_SEED_ENDPOINTS: \"\""));
        assert!(!private_service.contains("seed.attacker.net"));
        assert!(!private_service.contains("    ports:\n"));
        assert!(!private_service.contains("source: /var/run/dstack.sock"));
        assert!(!private_service.contains("profiles: [\"private-recipient\"]"));
    }

    // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Authored, unexecuted:
    // inspect actual output for every role selection, not marker presence alone.
    #[cfg(unix)]
    #[test]
    fn phala_renderer_scopes_seeds_for_all_role_combinations() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/node/prepare-phala-compose.sh");
        let selections: &[&[&str]] = &[
            &[],
            &["--public-peer"],
            &["--private-recipient"],
            &["--public-peer", "--private-recipient"],
            &["--private-recipient", "--public-peer"],
        ];
        for modes in selections {
            let output = std::process::Command::new("/bin/bash")
                .arg(&script)
                .args(*modes)
                .env_clear()
                .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
                .env("AERONYX_NODE_IMAGE", format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)))
                .env("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", "https://peer.aeronyx.network")
                .env("AERONYX_DISCOVERY_SEED_ENDPOINTS", r#"["https://seed.aeronyx.network"]"#)
                .env("AERONYX_REVERSE_ONION_RELAY_NODE_ID", "11".repeat(32))
                .env("AERONYX_REVERSE_ONION_RELAY_ENDPOINT", "https://relay.aeronyx.network")
                .output()
                .expect("Phala Compose renderer should launch");
            assert!(output.status.success(), "{modes:?}: {}", String::from_utf8_lossy(&output.stderr));
            let rendered = String::from_utf8(output.stdout).unwrap();
            let (public, remaining) = rendered.split_once("  aeronyx-private-recipient:\n").unwrap();
            let private = remaining.split_once("\nvolumes:\n").unwrap().0;
            for service in [public, private] {
                // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] A key and
                // its ${NAME:?} value are one environment entry, not two.
                assert_eq!(service.matches("      AERONYX_DISCOVERY_SEED_ENDPOINTS:").count(), 1, "{modes:?}");
                assert!(!service.contains("PHALA_PUBLIC_PEER_SEEDS_ENV"));
                // [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex]
                // Authored only: inspect each real rendered role separately.
                assert_eq!(service.matches("stop_signal:").count(), 1, "{modes:?}");
                assert!(service.contains("    stop_signal: SIGTERM\n"));
                assert_eq!(service.matches("stop_grace_period:").count(), 1, "{modes:?}");
                assert!(service.contains("    stop_grace_period: 2m\n"));
            }
            let public_selected = modes.contains(&"--public-peer");
            let private_selected = modes.contains(&"--private-recipient");
            assert_eq!(public.contains("${AERONYX_DISCOVERY_SEED_ENDPOINTS:?"), public_selected);
            assert_eq!(public.contains("AERONYX_DISCOVERY_SEED_ENDPOINTS: \"\""), !public_selected);
            assert_eq!(public.contains("    ports:\n"), public_selected);
            assert_eq!(public.contains("source: /var/run/dstack.sock"), public_selected);
            assert_eq!(public.contains("profiles: [\"public-peer\"]"), private_selected && !public_selected);
            assert!(private.contains("AERONYX_DISCOVERY_SEED_ENDPOINTS: \"\""));
            assert!(!private.contains("${AERONYX_DISCOVERY_SEED_ENDPOINTS"));
            assert!(!private.contains("    ports:\n"));
            assert!(!private.contains("source: /var/run/dstack.sock"));
            assert_eq!(private.contains("profiles: [\"private-recipient\"]"), !private_selected);
            assert_eq!(private.contains("${AERONYX_REVERSE_ONION_RELAY_NODE_ID:?"), private_selected);
            assert_eq!(private.contains("${AERONYX_REVERSE_ONION_RELAY_ENDPOINT:?"), private_selected);
        }
    }

    // [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex] Authored, not
    // run: exercise the actual renderer's role-scoped guard with deliberately
    // broken templates. No Compose process or service is started by this test.
    #[cfg(unix)]
    #[test]
    fn phala_renderer_rejects_missing_duplicate_or_weakened_stop_policy() {
        let script = include_str!("../../../deploy/node/prepare-phala-compose.sh");
        let template = include_str!("../../../deploy/node/compose.phala.peer.yaml");
        let (public, private) = template.split_once("  aeronyx-private-recipient:\n").unwrap();
        let directory = tempfile::Builder::new().prefix("phala-stop-policy-")
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let script_path = directory.path().join("prepare-phala-compose.sh");
        let template_path = directory.path().join("compose.phala.peer.yaml");
        std::fs::write(&script_path, script).unwrap();
        let render = |contents: &str| {
            std::fs::write(&template_path, contents).unwrap();
            std::process::Command::new("/bin/bash").arg(&script_path)
                .env_clear()
                .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
                .env("AERONYX_NODE_IMAGE", format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)))
                .output().expect("Phala renderer should launch")
        };
        let healthy = render(template);
        assert!(healthy.status.success(), "{}", String::from_utf8_lossy(&healthy.stderr));
        for private_role in [false, true] {
            for (original, replacement) in [
                ("    stop_signal: SIGTERM\n", ""),
                ("    stop_signal: SIGTERM\n", "    stop_signal: SIGKILL\n"),
                ("    stop_signal: SIGTERM\n", "    stop_signal: SIGTERM\n    stop_signal: SIGTERM\n"),
                ("    stop_grace_period: 2m\n", ""),
                ("    stop_grace_period: 2m\n", "    stop_grace_period: 10s\n"),
                ("    stop_grace_period: 2m\n", "    stop_grace_period: 2m\n    stop_grace_period: 2m\n"),
            ] {
                let mutated = if private_role {
                    format!("{public}  aeronyx-private-recipient:\n{}", private.replacen(original, replacement, 1))
                } else {
                    format!("{}  aeronyx-private-recipient:\n{private}", public.replacen(original, replacement, 1))
                };
                let rejected = render(&mutated);
                assert!(!rejected.status.success(), "accepted broken policy in private={private_role}");
                assert!(rejected.stdout.is_empty());
                assert!(String::from_utf8_lossy(&rejected.stderr)
                    .contains("each Phala role requires exactly SIGTERM and a two-minute stop grace"));
            }
        }
        // Whole-template counts still match when both keys move into one role.
        let moved = format!("{}  aeronyx-private-recipient:\n{}",
            public.replace("    stop_signal: SIGTERM\n", "    stop_signal: SIGTERM\n    stop_signal: SIGTERM\n"),
            private.replace("    stop_signal: SIGTERM\n", ""));
        assert!(!render(&moved).status.success());
        let isolated = include_str!("../../../deploy/node/compose.phala.yaml");
        assert_eq!(isolated.matches("stop_signal:").count(), 1);
        assert!(isolated.contains("    stop_signal: SIGTERM\n"));
        assert_eq!(isolated.matches("stop_grace_period:").count(), 1);
        assert!(isolated.contains("    stop_grace_period: 2m\n"));
    }

    // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Authored only: exercise
    // each real renderer input independently, without starting any services.
    #[cfg(unix)]
    fn render_phala_origin(role: &str, endpoint: &str, relay_id: &str) -> std::process::Output {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/node/prepare-phala-compose.sh");
        let mut command = std::process::Command::new("/bin/bash");
        command.arg(script).env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
            .env("AERONYX_NODE_IMAGE", format!("ghcr.io/aeronyx/node@sha256:{}", "a".repeat(64)));
        match role {
            "recipient" => {
                command.arg("--private-recipient")
                    .env("AERONYX_REVERSE_ONION_RELAY_NODE_ID", relay_id)
                    .env("AERONYX_REVERSE_ONION_RELAY_ENDPOINT", endpoint);
            }
            "seed" => {
                command.arg("--public-peer")
                    .env("AERONYX_DISCOVERY_SEED_ENDPOINTS", serde_json::to_string(&[endpoint]).unwrap());
            }
            "public" => {
                command.arg("--public-peer")
                    .env("AERONYX_DISCOVERY_SEED_ENDPOINTS", r#"["https://8.8.8.8"]"#)
                    .env("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", endpoint);
            }
            _ => panic!("unknown renderer test role"),
        }
        command.output().expect("Phala Compose renderer should launch")
    }

    // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Positive DNS cases
    // catch the former IP-parse error before the DNS branch was reachable.
    #[cfg(unix)]
    #[test]
    fn phala_renderer_and_startup_accept_public_dns_and_literal_origins() {
        let relay_id = "11".repeat(32);
        for endpoint in [
            "https://relay.example.net:443",
            "https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network",
            "https://8.8.8.8",
            "https://[2606:4700:4700::1111]",
            "https://[::ffff:8.8.8.8]",
        ] {
            let mut config: ServerConfig = toml::from_str(include_str!(
                "../../../deploy/node/server.phala.peer.example.toml",
            )).unwrap();
            apply_discovery_seed_endpoints(&mut config, &serde_json::to_string(&[endpoint]).unwrap()).unwrap();
            apply_discovery_public_endpoint_override(&mut config, endpoint).unwrap();
            for role in ["recipient", "seed", "public"] {
                let output = render_phala_origin(role, endpoint, &relay_id);
                assert!(output.status.success(), "{role} {endpoint}: {}",
                    String::from_utf8_lossy(&output.stderr));
                assert!(!output.stdout.is_empty());
            }
        }
    }

    // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Reject before writing
    // any manifest, including when only the public endpoint is invalid.
    #[cfg(unix)]
    #[test]
    fn phala_renderer_and_startup_reject_nonpublic_and_malformed_origins() {
        let oversized_label = format!("https://{}.example.net", "a".repeat(64));
        let relay_id = "11".repeat(32);
        for endpoint in [
            "http://relay.example.net", "https://127.0.0.1", "https://10.0.0.1",
            "https://100.64.0.1", "https://203.0.113.1", "https://224.0.0.1",
            "https://[::ffff:127.0.0.1]", "https://[2001:db8::1]",
            "https://[2002:808:808::1]", "https://[3fff::1]", "https://[ff0e::1]",
            "https://user@relay.example.net", "https://relay.example.net/api",
            "https://relay.example.net/?token=x", "https://relay.example.net/#fragment",
            "https://relay.example.net:0", "https://relay.internal",
            "https://relay..example.net", "https://-relay.example.net",
            "https://relay-.example.net", "https://relay_name.example.net",
            oversized_label.as_str(),
        ] {
            let mut config: ServerConfig = toml::from_str(include_str!(
                "../../../deploy/node/server.phala.peer.example.toml",
            )).unwrap();
            assert!(apply_discovery_seed_endpoints(&mut config,
                &serde_json::to_string(&[endpoint]).unwrap()).is_err(), "seed {endpoint}");
            assert!(apply_discovery_public_endpoint_override(&mut config, endpoint).is_err(),
                "public {endpoint}");
            for role in ["recipient", "seed", "public"] {
                let output = render_phala_origin(role, endpoint, &relay_id);
                assert!(!output.status.success(), "{role} {endpoint}");
                assert!(output.stdout.is_empty(), "partial manifest for {role} {endpoint}");
            }
        }
    }

    // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Required-value
    // placeholders must not carry a value different from the validated input.
    #[cfg(unix)]
    #[test]
    fn phala_recipient_renderer_rejects_whitespace_in_protected_values() {
        let relay_id = "11".repeat(32);
        // [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] A public
        // origin must be the same exact protected value validated by Rust.
        for role in ["recipient", "public"] {
            for endpoint in [" ", " https://relay.example.net", "https://relay.example.net "] {
                let output = render_phala_origin(role, endpoint, &relay_id);
                assert!(!output.status.success());
                assert!(output.stdout.is_empty());
            }
        }
        for invalid_id in [format!(" {relay_id}"), format!("{relay_id} ")] {
            let output = render_phala_origin("recipient", "https://relay.example.net", &invalid_id);
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
        }
    }

    // [PHALA-CLOUD-PROFILE 2026-10-06 by Codex] The unrendered template is
    // invalid until its role-specific renderer disables attestation or
    // supplies the public HTTPS endpoint bound into the signed descriptor.
    #[test]
    fn phala_peer_profile_requires_endpoint_when_attestation_is_enabled() {
        let source = include_str!("../../../deploy/node/server.phala.peer.example.toml");
        let mut config: ServerConfig = toml::from_str(source).unwrap();
        assert!(config.validate().is_err());
        apply_phala_attestation_socket_override(&mut config, "").unwrap();
        assert!(config.validate().is_ok());
        apply_phala_attestation_socket_override(
            &mut config,
            "/var/run/aeronyx/phala-agent.sock",
        )
        .unwrap();
        assert!(config.validate().is_err());
        assert!(!config.memchain.chat_relay.enabled);
        assert!(!config.discovery.advertise_onion_middle);
        assert_eq!(config.memchain.chat_relay.max_pending_per_wallet, 128);
        assert_eq!(
            config.memchain.chat_relay.max_pending_messages_total,
            2_048
        );
        assert_eq!(
            config.memchain.chat_relay.max_pending_message_bytes_total,
            64 * 1024 * 1024
        );
        assert_eq!(config.memchain.chat_relay.max_message_size, 64 * 1024);
        assert_eq!(config.memchain.chat_relay.max_blob_size, 5 * 1024 * 1024);
        assert_eq!(config.memchain.chat_relay.max_blobs_per_receiver, 8);
        assert_eq!(config.memchain.chat_relay.max_pending_blobs_total, 128);
        assert_eq!(
            config.memchain.chat_relay.max_pending_blob_bytes_total,
            256 * 1024 * 1024
        );
        assert!(!config.blind_vault.enabled);
        assert!(!config.blind_vault.public_api_enabled);
        assert_eq!(config.blind_vault.max_lease_ttl_secs, 7 * 24 * 60 * 60);
        assert_eq!(config.blind_vault.max_object_ttl_secs, 7 * 24 * 60 * 60);
        assert_eq!(config.blind_vault.max_objects_per_lease, 1_024);
        assert_eq!(config.blind_vault.max_bytes_per_lease, 64 * 1024 * 1024);
        assert_eq!(config.blind_vault.max_live_leases, 128);
        assert_eq!(
            config.blind_vault.max_total_ciphertext_bytes,
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(config.blind_vault.min_free_disk_bytes, 1024 * 1024 * 1024);
        assert!(config.discovery.public_endpoint.is_none());
        assert!(apply_discovery_public_endpoint_if_configured(&mut config, " ").is_err());
        apply_discovery_public_endpoint_if_configured(&mut config, "").unwrap();
        assert!(config.validate().is_err());
        assert!(apply_discovery_public_endpoint_override(&mut config, " ").is_err());
        assert!(apply_discovery_public_endpoint_override(
            &mut config,
            "http://node.example.com"
        )
        .is_err());
        for invalid in [
            "https://localhost",
            "https://127.0.0.1:8422",
            "https://node.example.com/path",
            "https://node.example.com/?token=secret",
        ] {
            assert!(apply_discovery_public_endpoint_override(&mut config, invalid).is_err());
        }
        apply_discovery_public_endpoint_override(
            &mut config,
            "https://node.phala.network",
        )
        .unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(
            config.discovery.public_endpoint.as_deref(),
            Some("https://node.phala.network")
        );
        assert_eq!(config.network.listen_addr, "127.0.0.1:51820".parse().unwrap());
        assert!(!config.management.enabled);
        assert_eq!(
            config.discovery.phala_attestation_socket_path.as_deref(),
            Some("/var/run/aeronyx/phala-agent.sock")
        );
        // [PHALA-PEER-INGRESS-OPT-IN 2026-10-06 by Codex] The source template
        // publishes no port; renderer opt-in is the sole peer ingress path.
        let compose = include_str!("../../../deploy/node/compose.phala.peer.yaml");
        assert!(compose.contains("    platform: linux/amd64\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_INGRESS\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_ENDPOINT_ENV\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_DISCOVERY_ENV\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_API_LISTENER_ENV\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_ATTESTATION_SOCKET_ENV\n"));
        assert!(compose.contains("# PHALA_PUBLIC_PEER_SEEDS_ENV\n"));
        // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] The public
        // substitution marker and private literal belong to distinct services.
        assert_eq!(compose.matches("# PHALA_PUBLIC_PEER_SEEDS_ENV\n").count(), 1);
        assert!(compose.contains("# PHALA_PUBLIC_PEER_DSTACK_SOCKET\n"));
        let renderer = include_str!("../../../deploy/node/prepare-phala-compose.sh");
        assert!(renderer.contains("AERONYX_DISCOVERY_PUBLIC_ENDPOINT:?Set"));
        // [PHALA-RELAY-INGRESS-RENDER-GATE 2026-10-06 by Codex] An enabled
        // public relay must render with both its ingress profile and assigned
        // origin, rather than starting as an unreachable local store.
        assert!(renderer.contains(
            "AERONYX_PHALA_ONION_RELAY_ENABLED=true requires the explicit --public-peer profile"
        ));
        assert!(renderer.contains(
            "AERONYX_PHALA_ONION_RELAY_ENABLED=true requires an assigned public HTTPS endpoint"
        ));
        // [PHALA-QUEUE-ROLE-RENDER-GATE 2026-10-06 by Codex] Rendering must
        // reject the queue unless its required public OnionMiddle role is on.
        assert!(renderer.contains("queue requires AERONYX_PHALA_ONION_RELAY_ENABLED=true"));
        assert!(renderer.contains("queue_node_ids("));
        assert!(renderer.contains("queue signed authority inputs must be supplied together"));
        assert!(!compose.contains("    ports:\n"));
        assert!(!compose.contains("source: /var/run/dstack.sock"));
        let renderer = include_str!("../../../deploy/node/prepare-phala-compose.sh");
        let dockerfile = include_str!("../../../deploy/node/Dockerfile");
        let isolated_compose = include_str!("../../../deploy/node/compose.phala.yaml");
        assert!(dockerfile.contains("FROM --platform=linux/amd64 rust:1.97.1-bookworm AS builder"));
        assert!(dockerfile.contains("FROM --platform=linux/amd64 debian:bookworm-slim"));
        assert!(isolated_compose.contains("    platform: linux/amd64\n"));
        assert!(renderer.contains("if \"--public-peer\" in requested_modes:"));
        assert!(renderer.contains("public_ingress_mapping = '    ports:\\n      - \"8422:8422\"\\n'"));
        assert!(renderer.contains(
            "template.replace(public_endpoint_marker, public_peer_endpoint_env)"
        ));
        assert!(renderer.contains(
            "template.replace(public_discovery_marker, public_peer_visibility_env)"
        ));
        assert!(renderer.contains("template.replace(public_api_marker, public_peer_api_env)"));
        assert!(renderer.contains("public_peer_seeds_env ="));
        assert!(renderer.contains(
            "${AERONYX_DISCOVERY_SEED_ENDPOINTS:?Set AERONYX_DISCOVERY_SEED_ENDPOINTS"
        ));
        // [PHALA-ATTESTATION-TWO-PHASE-CONFIG 2026-10-06 by Codex]
        // The renderer must validate public seeds and keep quote transport off
        // until the Phala-assigned origin is available.
        assert!(renderer.contains("seeds = json.loads(seeds_encoded)"));
        assert!(renderer.contains("not 1 <= len(seeds) <= 64"));
        assert!(renderer.contains("public_peer_endpoint_unassigned_env ="));
        assert!(renderer.contains("public_peer_attestation_env if public_endpoint else private_attestation_env"));
        assert!(renderer.contains(
            "template.replace(public_seeds_marker, public_peer_seeds_env, 1)"
        ));
        assert!(renderer.contains(
            "template.replace(public_seeds_marker, private_seeds_env, 1)"
        ));
        assert!(renderer.contains("public_dstack_mount if public_endpoint else \"\""));
        assert!(renderer.contains("template.replace(public_dstack_marker, \"\")"));
        assert!(renderer.contains("template.replace(public_attestation_marker, private_attestation_env)"));
        assert!(renderer.contains(
            "template.replace(public_endpoint_marker, private_endpoint_env)"
        ));
        assert!(renderer.contains(
            "template.replace(public_discovery_marker, private_visibility_env)"
        ));
        assert!(renderer.contains("template.replace(public_api_marker, private_api_env)"));
        assert!(renderer.contains(
            "template.replace(public_attestation_marker, private_attestation_env)"
        ));
        // [PHALA-PRIVATE-SEED-RENDER-GUARD 2026-10-06 by Codex] The private
        // renderer allows exactly its empty seed override, never a public seed.
        assert!(renderer.contains(
            "private_service.count(\"      AERONYX_DISCOVERY_SEED_ENDPOINTS:\") != 1"
        ));
        assert!(renderer.contains("private_service.count(private_seeds_env) != 1"));
        assert!(!renderer.contains(
            "\"AERONYX_DISCOVERY_SEED_ENDPOINTS\" in private_service"
        ));
        let recipient_compose = compose
            .split("  aeronyx-private-recipient:\n")
            .nth(1)
            .unwrap()
            .split("\nvolumes:\n")
            .next()
            .unwrap();
        assert!(!recipient_compose.contains("    ports:\n"));
        // [PHALA-ROUTE-POLICY-ENV 2026-10-07 by Codex] Count the YAML key.
        assert_eq!(recipient_compose.matches("      AERONYX_DISCOVERY_SEED_ENDPOINTS:").count(), 1);
        assert!(recipient_compose.contains("AERONYX_DISCOVERY_SEED_ENDPOINTS: \"\""));
        // [PHALA-ATTESTATION-FAIL-CLOSED 2026-10-06 by Codex] The deployed
        // Phala peer profile must not silently accept a legacy quote contract.
        assert!(!config.discovery.phala_attestation_allow_legacy_v0);
        apply_phala_onion_relay_override(&mut config, "true").unwrap();
        assert!(config.memchain.chat_relay.enabled);
        assert!(config.discovery.advertise_onion_middle);
        assert!(config.validate().is_ok());
        apply_phala_onion_relay_override(&mut config, "false").unwrap();
        assert!(!config.memchain.chat_relay.enabled);
        assert!(!config.discovery.advertise_onion_middle);
        let relay_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[46; 32])
                .unwrap()
                .public_key_bytes(),
        );
        apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay_id),
            // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex]
            // Public-shape origin only; this configuration test performs no IO.
            Some("https://relay.aeronyx.network"),
        )
        .unwrap();
        // [PHALA-PRIVATE-RELAY-DURABILITY 2026-10-06 by Codex] Match the
        // actual loader order: private role first, public relay override last.
        config.discovery.advertise_onion_middle = true;
        apply_phala_onion_relay_override(&mut config, "false").unwrap();
        assert!(config.reverse_onion.recipient.enabled);
        assert!(config.blind_vault.enabled);
        assert!(config.memchain.chat_relay.enabled);
        assert!(!config.discovery.advertise_onion_middle);
        // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] Switching
        // public relay to private recipient requires the loader's full egress
        // isolation, not merely its onion-middle advertisement override.
        apply_discovery_public_endpoint_if_configured(&mut config, "").unwrap();
        apply_discovery_public_visibility_override(&mut config, "false").unwrap();
        apply_discovery_api_listener_override(&mut config, "").unwrap();
        apply_phala_attestation_socket_override(&mut config, "").unwrap();
        apply_discovery_seed_endpoints(&mut config, "").unwrap();
        config.reverse_onion.recipient.state_db_path =
            "/var/lib/aeronyx/reverse-onion-recipient.sqlite".into();
        assert!(config.validate().is_ok(), "{:?}", config.validate());
        assert!(!config.blind_vault.public_api_enabled);
        assert_eq!(config.memchain.chat_relay.max_pending_messages_total, 2_048);
        assert_eq!(
            config.memchain.chat_relay.max_pending_blob_bytes_total,
            256 * 1024 * 1024
        );
        assert_eq!(config.blind_vault.max_live_leases, 128);
        assert_eq!(
            config.blind_vault.max_total_ciphertext_bytes,
            2 * 1024 * 1024 * 1024
        );
    }

    // [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Clearing the
    // discovery origin alone would resurrect the legacy fallback in signed
    // descriptors. An explicit empty value must remove both, not other policy.
    #[test]
    fn phala_empty_origin_override_clears_both_descriptor_sources_only() {
        let mut config = ServerConfig::default();
        config.discovery.public_endpoint = Some("https://old.aeronyx.network".into());
        config.network.public_endpoint = Some("https://fallback.aeronyx.network".into());
        config.discovery.seed_endpoints = vec!["https://seed.aeronyx.network".into()];
        apply_discovery_public_endpoint_if_configured(&mut config, "").unwrap();
        assert!(config.discovery.public_endpoint.is_none());
        assert!(config.network.public_endpoint.is_none());
        assert!(config.effective_public_endpoint().is_none());
        assert_eq!(config.discovery.seed_endpoints, vec!["https://seed.aeronyx.network"]);
        assert!(!config.discovery.enabled);
        assert!(config.discovery.public_api_listen_addr.is_none());
    }

    // [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] Authored, not run.
    #[test]
    fn phala_onion_relay_override_is_exact_and_separates_private_role() {
        let mut config = ServerConfig::default();
        assert!(apply_phala_onion_relay_override(&mut config, "false").is_ok());
        assert!(!config.discovery.advertise_onion_middle);
        assert!(!config.memchain.chat_relay.enabled);

        config.memchain.chat_relay.enabled = true;
        config.discovery.advertise_onion_middle = true;
        assert!(apply_phala_onion_relay_override(&mut config, "false").is_ok());
        assert!(!config.memchain.chat_relay.enabled);
        assert!(!config.discovery.advertise_onion_middle);

        assert!(apply_phala_onion_relay_override(&mut config, "yes").is_err());

        config.reverse_onion.recipient.enabled = true;
        config.memchain.chat_relay.enabled = true;
        config.discovery.advertise_onion_middle = true;
        assert!(apply_phala_onion_relay_override(&mut config, "false").is_ok());
        assert!(config.memchain.chat_relay.enabled);
        assert!(!config.discovery.advertise_onion_middle);
        assert!(apply_phala_onion_relay_override(&mut config, "true").is_err());
        assert!(config.memchain.chat_relay.enabled);
    }

    // [REVERSE-RECOVERY-BOOT 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn reverse_recovery_passes_parent_config_without_enabling_new_admission() {
        let mut config = ServerConfig::default();
        config.reverse_onion.queue.enabled = true;
        config.reverse_onion.queue.recovery_only = true;
        config.reverse_onion.queue.db_path = "/Volumes/disk/reverse-onion-test/queue.sqlite".into();
        config.reverse_onion.queue.recipient_node_ids.push(hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        ));
        // [PHALA-REVERSE-ONION-LISTENER-GATES 2026-10-06 by Codex]
        assert!(config.validate().is_err());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        // [PHALA-REVERSE-QUEUE-ORIGIN-GATE 2026-10-06 by Codex] A listener
        // alone cannot advertise or route durable recovery claims.
        assert!(config.validate().is_err());
        config.discovery.public_endpoint = Some("https://relay.example.net".into());
        assert!(config.validate().is_ok());
        let mut collision = config.clone();
        collision.blind_vault.db_path = collision.reverse_onion.queue.db_path.clone();
        assert!(collision.validate().is_err());
        config.reverse_onion.queue.recovery_only = false;
        assert!(config.validate().is_err());
    }

    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] The root server
    // validation gate must reject unbounded source evidence polling too.
    #[test]
    fn source_result_polling_limits_reach_parent_server_gate() {
        let mut config = ServerConfig::default();
        config.reverse_onion.source.enabled = true;
        config.reverse_onion.source.state_db_path =
            "/Volumes/disk/reverse-onion-test/source.sqlite".into();
        config.reverse_onion.source.result_wait_secs = 31;
        assert!(config.validate().is_err());
    }

    // [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Authored, not
    // executed: parent config admits an inert live source with operator pins
    // and authenticated gossip, without requiring a startup grant bundle.
    #[test]
    fn source_bootstrap_uses_identity_origin_pins_and_discovery_gate() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.reverse_onion.source.enabled = true;
        config.reverse_onion.source.state_db_path =
            "/Volumes/disk/reverse-onion-test/source-bootstrap.sqlite".into();
        config.reverse_onion.source.relay_node_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[41; 32])
                .unwrap().public_key_bytes(),
        );
        config.reverse_onion.source.recipient_node_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[42; 32])
                .unwrap().public_key_bytes(),
        );
        config.reverse_onion.source.relay_endpoint = "https://relay.example.net".into();
        assert!(config.validate().is_ok());
        // [PHALA-SOURCE-MPI-COMPOSITION 2026-10-08 by Codex] Recovery
        // also returns through MPI; it cannot relax the composition contract.
        for recovery in [false, true] {
            for mode in [MemChainMode::Off, MemChainMode::Saas, MemChainMode::Local, MemChainMode::P2p] {
                let mut candidate = config.clone();
                candidate.reverse_onion.source.recovery_only = recovery;
                candidate.memchain.mode = mode.clone();
                if matches!(mode, MemChainMode::Off | MemChainMode::Saas) {
                    assert!(candidate.validate().unwrap_err().to_string()
                        .contains("source pulls require local or p2p MemChain MPI runtime"));
                } else {
                    assert!(candidate.validate().is_ok());
                }
            }
        }
        config.discovery.gossip_enabled = false;
        assert!(config.validate().is_err());
    }

    // [PHALA-REVERSE-ONION-LISTENER-GATES 2026-10-06 by Codex]
    #[test]
    fn no_vpn_profile_rejects_source_pull_role() {
        let mut config = ServerConfig::default();
        config.vpn.enabled = false;
        config.reverse_onion.source.enabled = true;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source pulls require the authenticated VPN/MPI listener"));
    }

    // [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn reverse_recipient_omitted_config_stays_disabled_and_queue_cannot_opt_in() {
        let config: ServerConfig = toml::from_str("").unwrap();
        assert!(!config.reverse_onion.recipient.enabled);
        assert!(!config.reverse_onion.queue.enabled);
        let mut queue = ServerConfig::default();
        queue.reverse_onion.queue.enabled = true;
        assert!(queue.validate().is_err());
        let mut recipient = ServerConfig::default();
        recipient.reverse_onion.recipient.enabled = true;
        assert!(recipient.validate().is_err());
    }

    // [PHALA-REVERSE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Queue startup
    // must accept identity pins without a pre-issued descriptor/grant bundle,
    // while retaining every parent listener, role, and gossip gate.
    #[test]
    fn live_phala_queue_can_boot_before_discovery_authority_arrives() {
        let mut config = ServerConfig::default();
        config.memchain.mode = crate::config_memchain::MemChainMode::Off;
        config.memchain.chat_relay.enabled = true;
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.discovery.public_endpoint = Some("https://relay.example.net".into());
        config.discovery.advertise_onion_middle = true;
        config.reverse_onion.queue.enabled = true;
        config.reverse_onion.queue.db_path =
            "/Volumes/disk/reverse-onion-test/phala-queue.sqlite".into();
        config.reverse_onion.queue.recipient_node_ids = vec![hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[51; 32])
                .unwrap().public_key_bytes(),
        )];
        config.reverse_onion.queue.source_node_ids = vec![hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[52; 32])
                .unwrap().public_key_bytes(),
        )];
        assert!(config.validate().is_ok());

        config.discovery.gossip_enabled = false;
        assert!(config.validate().is_err());
        config.discovery.gossip_enabled = true;
        config.discovery.public_api_listen_addr = None;
        assert!(config.validate().is_err());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.discovery.public_endpoint = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn reverse_recipient_missing_terminal_and_db_collision_fail_closed() {
        let mut config = ServerConfig::default();
        config.reverse_onion.recipient.enabled = true;
        config.reverse_onion.recipient.relay_node_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes());
        config.reverse_onion.recipient.relay_endpoint = "https://8.8.8.8".into();
        // [REVERSE-ONION-TEST-PATH 2026-10-05 by Codex] Keep fixture paths on
        // the designated data volume; validation never opens this database.
        config.reverse_onion.recipient.state_db_path =
            "/Volumes/disk/aeronyx-reverse-onion-tests/recipient-test.sqlite".into();
        assert!(config.validate().is_err());
        config.blind_vault.enabled = true;
        config.blind_vault.db_path = config.reverse_onion.recipient.state_db_path.clone();
        assert!(config.validate().is_err());
    }

    // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn reverse_recipient_does_not_require_direct_blind_vault_api() {
        let mut config = ServerConfig::default();
        config.vpn.enabled = false;
        // [PHALA-CONFIG-EXECUTED-REGRESSION 2026-10-08 by Codex] A private
        // recipient must satisfy management isolation before API role checks.
        config.management.enabled = false;
        // [PHALA-CHAT-RELAY-INDEPENDENT-CONFIG 2026-10-06 by Codex]
        // The recipient stores ciphertext locally without enabling MemChain
        // inference or the general-purpose memory runtime.
        config.memchain.mode = crate::config_memchain::MemChainMode::Off;
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.blind_vault.enabled = true;
        config.blind_vault.public_api_enabled = false;
        config.memchain.chat_relay.enabled = true;
        config.reverse_onion.recipient.enabled = true;
        config.reverse_onion.recipient.relay_node_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        );
        config.reverse_onion.recipient.relay_endpoint = "https://8.8.8.8".into();
        config.reverse_onion.recipient.state_db_path = "/Volumes/disk/reverse-onion-test/recipient.sqlite".into();
        assert!(config.validate().is_err());
        config.discovery.public_discovery = false;
        config.blind_vault.public_api_enabled = true;
        assert!(config.validate().is_err());
        config.blind_vault.public_api_enabled = false;
        assert!(config.validate().is_ok());
        assert!(!config.blind_vault.public_api_enabled);
    }

    // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn reverse_recipient_recovery_only_does_not_require_authority_gossip() {
        let mut config = ServerConfig::default();
        config.vpn.enabled = false;
        // [PHALA-CONFIG-EXECUTED-REGRESSION 2026-10-08 by Codex] Recovery
        // retains the same independent egress/runtime isolation as live mode.
        config.management.enabled = false;
        config.memchain.mode = crate::config_memchain::MemChainMode::Off;
        config.blind_vault.enabled = true;
        config.memchain.chat_relay.enabled = true;
        config.reverse_onion.recipient.enabled = true;
        config.reverse_onion.recipient.recovery_only = true;
        config.reverse_onion.recipient.relay_node_id = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[42; 32]).unwrap().public_key_bytes(),
        );
        config.reverse_onion.recipient.relay_endpoint = "https://8.8.8.8".into();
        config.reverse_onion.recipient.state_db_path = "/Volumes/disk/reverse-onion-test/recovery.sqlite".into();
        config.discovery.public_discovery = false;
        assert!(config.validate().is_ok());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        assert!(config.validate().is_err());
    }

    // [PHALA-PRIVATE-RECIPIENT-DESCRIPTOR 2026-10-06 by Codex] Authored,
    // unexecuted. The discovery endpoint may be injected by Compose at load.
    #[test]
    fn private_reverse_recipient_rejects_public_descriptor_endpoint() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.blind_vault.enabled = true;
        config.memchain.chat_relay.enabled = true;
        let relay = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[44; 32])
                .unwrap()
                .public_key_bytes(),
        );
        config.reverse_onion.recipient.state_db_path =
            "/Volumes/disk/aeronyx-reverse-onion-tests/private-recipient.sqlite".into();

        apply_discovery_public_endpoint_override(
            &mut config,
            "https://peer.phala.network",
        )
        .unwrap();
        apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay),
            Some("https://relay.example.net"),
        )
        .unwrap();
        assert!(config.validate().is_err());
        config.discovery.public_endpoint = None;
        config.network.public_endpoint = Some("https://peer.phala.network".into());
        assert!(config.validate().is_err());
    }

    // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] Authored, unexecuted.
    #[test]
    fn private_recipient_rejects_non_relay_peer_destinations() {
        let mut base = ServerConfig::default();
        base.vpn.enabled = false;
        // [PHALA-CONFIG-EXECUTED-REGRESSION 2026-10-08 by Codex] Establish
        // a valid isolated base before testing each forbidden destination.
        base.management.enabled = false;
        base.memchain.mode = crate::config_memchain::MemChainMode::Off;
        base.blind_vault.enabled = true;
        base.memchain.chat_relay.enabled = true;
        base.reverse_onion.recipient.enabled = true;
        base.reverse_onion.recipient.recovery_only = true;
        base.reverse_onion.recipient.relay_node_id = "11".repeat(32);
        base.reverse_onion.recipient.relay_endpoint = "https://relay.example.net".into();
        base.reverse_onion.recipient.state_db_path =
            "/Volumes/disk/reverse-onion-tests/private-recipient-egress.sqlite".into();
        base.discovery.public_discovery = false;
        assert!(base.validate().is_ok());

        let mut with_management = base.clone();
        with_management.management.enabled = true;
        assert!(with_management.validate().is_err());

        for mode in [
            crate::config_memchain::MemChainMode::Local,
            crate::config_memchain::MemChainMode::P2p,
            crate::config_memchain::MemChainMode::Saas,
        ] {
            let mut with_memchain_runtime = base.clone();
            with_memchain_runtime.memchain.mode = mode;
            assert!(with_memchain_runtime.validate().is_err());
        }

        let mut with_vpn = base.clone();
        with_vpn.vpn.enabled = true;
        assert!(with_vpn.validate().is_err());

        let mut with_bootstrap_url = base.clone();
        with_bootstrap_url.discovery.bootstrap_snapshot_url =
            Some("https://bootstrap.example.net/peers.json".into());
        assert!(with_bootstrap_url.validate().is_err());

        let mut with_seed = base.clone();
        with_seed.discovery.seed_endpoints = vec!["https://seed.example.net".into()];
        assert!(with_seed.validate().is_err());

        let mut with_directory_peer = base.clone();
        with_directory_peer.discovery.directory_chain_sync_peer_node_ids =
            vec!["22".repeat(32)];
        assert!(with_directory_peer.validate().is_err());

        let mut with_mirror = base.clone();
        with_mirror.discovery.directory_full_node_mirror_enabled = true;
        assert!(with_mirror.validate().is_err());

        let mut with_memchain_sync = base.clone();
        with_memchain_sync.memchain.commitment_sync_enabled = true;
        assert!(with_memchain_sync.validate().is_err());

        let mut with_memchain_coordinator = base;
        with_memchain_coordinator.memchain.commitment_coordinator_enabled = true;
        assert!(with_memchain_coordinator.validate().is_err());
    }

    // [PHALA-DISCOVERY-SEEDS 2026-10-06 by Codex] Authored, unexecuted.
    #[test]
    fn phala_seed_env_is_bounded_tls_only_and_replaces_seed_list() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.seed_endpoints = vec!["https://old.example".into()];
        apply_discovery_seed_endpoints(
            &mut config,
            r#"["https://seed-a.example.net","https://8.8.8.8:8422"]"#,
        )
        .unwrap();
        assert_eq!(
            config.discovery.seed_endpoints,
            vec![
                "https://seed-a.example.net".to_string(),
                "https://8.8.8.8:8422".to_string()
            ]
        );
        for invalid in [
            r#"["http://seed.example.net"]"#,
            r#"["https://localhost"]"#,
            r#"["https://user:password@seed.example.net"]"#,
            r#"["https://seed.example.net/path"]"#,
            r#"[]"#,
        ] {
            assert!(apply_discovery_seed_endpoints(&mut config, invalid).is_err());
        }
        let too_many = serde_json::to_string(&vec!["https://seed.example.net"; 65]).unwrap();
        assert!(apply_discovery_seed_endpoints(&mut config, &too_many).is_err());
        assert!(apply_discovery_seed_endpoints(&mut config, &"x".repeat(8193)).is_err());
    }

    // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Explicit clearing is
    // distinct from a missing override and must not enable disabled discovery.
    #[test]
    fn phala_empty_seed_override_clears_without_enabling_gossip() {
        for enabled in [false, true] {
            let mut config = ServerConfig::default();
            config.discovery.enabled = enabled;
            config.discovery.gossip_enabled = enabled;
            config.discovery.seed_endpoints = vec!["https://seed.aeronyx.network".into()];
            apply_discovery_seed_endpoints(&mut config, "").unwrap();
            assert!(config.discovery.seed_endpoints.is_empty());
            assert_eq!(config.discovery.enabled, enabled);
            assert_eq!(config.discovery.gossip_enabled, enabled);
            for invalid in [" ", "\n", "[]"] {
                config.discovery.seed_endpoints = vec!["https://kept.aeronyx.network".into()];
                assert!(apply_discovery_seed_endpoints(&mut config, invalid).is_err());
                assert_eq!(config.discovery.seed_endpoints, vec!["https://kept.aeronyx.network"]);
            }
        }
    }

    // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] The same mounted
    // peer template must pass the real parent gate after private role overrides,
    // retaining the pinned relay and the independent recovery policy.
    #[test]
    fn phala_private_seed_clear_preserves_pinned_bootstrap_and_recovery_mode() {
        let template = include_str!("../../../deploy/node/server.phala.peer.example.toml");
        for recovery_only in [false, true] {
            let mut config: ServerConfig = toml::from_str(template).unwrap();
            config.discovery.seed_endpoints = vec!["https://seed.aeronyx.network".into()];
            let relay_id = hex::encode(
                aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[47; 32])
                    .unwrap().public_key_bytes(),
            );
            apply_discovery_public_visibility_override(&mut config, "false").unwrap();
            apply_discovery_api_listener_override(&mut config, "").unwrap();
            apply_phala_attestation_socket_override(&mut config, "").unwrap();
            apply_reverse_onion_recipient_env(
                &mut config, Some("true"), Some(&relay_id), Some("https://relay.aeronyx.network"),
            ).unwrap();
            apply_phala_onion_relay_override(&mut config, "false").unwrap();
            config.reverse_onion.recipient.recovery_only = recovery_only;
            assert!(matches!(
                config.validate(),
                Err(ServerError::ConfigInvalid { field, .. }) if field == "discovery.seed_endpoints"
            ));
            apply_discovery_seed_endpoints(&mut config, "").unwrap();
            assert!(config.validate().is_ok());
            assert_eq!(config.reverse_onion.recipient.relay_node_id, relay_id);
            assert_eq!(config.reverse_onion.recipient.relay_endpoint, "https://relay.aeronyx.network");
            assert_eq!(config.reverse_onion.recipient.recovery_only, recovery_only);
            assert!(config.memchain.chat_relay.enabled);
            assert!(config.blind_vault.enabled);
            assert!(!config.discovery.public_discovery);
            assert!(config.discovery.public_api_listen_addr.is_none());
        }
    }

    // [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
    #[test]
    fn phala_recipient_env_is_explicit_and_keeps_public_vault_closed() {
        let mut config = ServerConfig::default();
        config.vpn.enabled = false;
        // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] Private
        // execution has no independent management or MemChain egress.
        config.management.enabled = false;
        config.memchain.mode = crate::config_memchain::MemChainMode::Off;
        let relay = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[43; 32])
                .unwrap()
                .public_key_bytes(),
        );
        assert!(apply_reverse_onion_recipient_env(&mut config, None, None, None).is_ok());
        assert!(!config.reverse_onion.recipient.enabled);
        assert!(!config.blind_vault.enabled);
        assert!(!config.memchain.chat_relay.enabled);

        assert!(apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay),
            Some("https://relay.aeronyx.network"),
        )
        .is_ok());
        assert!(apply_discovery_public_visibility_override(&mut config, "false").is_ok());
        assert!(config.reverse_onion.recipient.enabled);
        assert!(!config.discovery.public_discovery);
        assert_eq!(config.reverse_onion.recipient.relay_node_id, relay);
        assert_eq!(config.reverse_onion.recipient.relay_endpoint, "https://relay.aeronyx.network");
        assert!(config.blind_vault.enabled);
        assert!(config.memchain.chat_relay.enabled);
        assert!(!config.blind_vault.public_api_enabled);
        assert!(apply_discovery_public_visibility_override(&mut config, "yes").is_err());
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.reverse_onion.recipient.state_db_path =
            "/var/lib/aeronyx/reverse-onion-recipient.sqlite".into();
        config.blind_vault.db_path = "/var/lib/aeronyx/private-terminal-vault.sqlite".into();
        config.memchain.chat_relay.db_path = "/var/lib/aeronyx/chat-pending.sqlite".into();
        assert!(config.validate().is_ok(), "{:?}", config.validate());
    }

    // [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
    #[test]
    fn phala_recipient_env_rejects_partial_or_unsafe_opt_in() {
        let mut config = ServerConfig::default();
        let relay = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[44; 32])
                .unwrap()
                .public_key_bytes(),
        );
        let orphan_pin = "04".repeat(32);
        assert!(apply_reverse_onion_recipient_env(&mut config, Some("yes"), None, None).is_err());
        assert!(apply_reverse_onion_recipient_env(&mut config, Some("true"), None, None).is_err());
        assert!(apply_reverse_onion_recipient_env(
            &mut config,
            None,
            Some(&orphan_pin),
            None,
        )
        .is_err());
        config.blind_vault.public_api_enabled = true;
        assert!(apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay),
            Some("https://relay.example"),
        )
        .is_err());
    }

    // [PHALA-REVERSE-ONION-RECIPIENT-CONFIG 2026-10-06 by Codex]
    #[test]
    fn phala_recipient_env_cannot_release_a_recovery_hold() {
        let mut config = ServerConfig::default();
        config.reverse_onion.recipient.recovery_only = true;
        let relay = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[45; 32])
                .unwrap()
                .public_key_bytes(),
        );
        assert!(apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay),
            Some("https://relay.example"),
        )
        .is_ok());
        assert!(config.reverse_onion.recipient.enabled);
        assert!(config.reverse_onion.recipient.recovery_only);
        assert!(!config.reverse_onion.recipient.permits_new_claims());
    }

    // [PHALA-RECIPIENT-RECOVERY-OVERRIDE 2026-10-06 by Codex] Authored only;
    // verify opt-in recovery, explicit release, and fail-closed invalid input.
    #[test]
    fn phala_recipient_recovery_override_is_explicit_and_role_scoped() {
        let mut config = ServerConfig::default();
        assert!(apply_reverse_onion_recovery_only_env(&mut config, None).is_ok());
        assert!(!config.reverse_onion.recipient.recovery_only);
        assert!(apply_reverse_onion_recovery_only_env(&mut config, Some("true")).is_err());

        let relay = hex::encode(
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[47; 32])
                .unwrap()
                .public_key_bytes(),
        );
        apply_reverse_onion_recipient_env(
            &mut config,
            Some("true"),
            Some(&relay),
            Some("https://relay.example"),
        )
        .unwrap();
        assert!(apply_reverse_onion_recovery_only_env(&mut config, Some("yes")).is_err());
        assert!(apply_reverse_onion_recovery_only_env(&mut config, Some("true")).is_ok());
        assert!(config.reverse_onion.recipient.recovery_only);
        assert!(!config.reverse_onion.recipient.permits_new_claims());
        assert!(apply_reverse_onion_recovery_only_env(&mut config, Some("false")).is_ok());
        assert!(!config.reverse_onion.recipient.recovery_only);
        assert!(config.reverse_onion.recipient.permits_new_claims());
    }

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
        assert!(!config.discovery.custody_audit_witness_auto_renewal_enabled);
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
        assert!(!config.discovery.permissionless_endpoint_proof_enabled);
        assert_eq!(
            config.discovery.permissionless_endpoint_proof_max_entries,
            DiscoveryConfig::default_permissionless_endpoint_proof_max_entries()
        );
        assert_eq!(
            config.discovery.permissionless_endpoint_proof_ttl_secs,
            DiscoveryConfig::default_permissionless_endpoint_proof_ttl_secs()
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
        let legacy = ServerConfig::from_str("[discovery]\nenabled = true\n")
            .expect("legacy discovery config");
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
        let legacy = ServerConfig::from_str("[discovery]\nenabled = true\n")
            .expect("legacy discovery config");
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
        let legacy = ServerConfig::from_str("[discovery]\nenabled = true\n")
            .expect("legacy discovery config");
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

    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Authored only:
    // exercise the parent parser/validation gate, not a duplicate predicate.
    #[test]
    fn discovery_descriptor_lifetime_matches_bounded_kem_overlap() {
        let max = crate::services::onion_keys::MAX_ONION_DESCRIPTOR_TTL_SECS;
        for ttl in [60, DiscoveryConfig::default_descriptor_ttl_secs(), 7200, max] {
            let encoded = format!("[discovery]\nenabled = true\ndescriptor_ttl_secs = {ttl}\n");
            assert!(ServerConfig::from_str(&encoded).is_ok(), "supported TTL {ttl}");
        }
        for ttl in [0, 59, max + 1, i64::MAX as u64] {
            let encoded = format!("[discovery]\nenabled = true\ndescriptor_ttl_secs = {ttl}\n");
            assert!(ServerConfig::from_str(&encoded).is_err(), "unsupported TTL {ttl}");
        }
        let mut discovery = DiscoveryConfig::default();
        discovery.descriptor_ttl_secs = u64::MAX;
        assert!(discovery.validate().is_err());
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

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[test]
    fn test_supernode_and_chat_relay_combined() {
        let toml_str = r#"
[memchain]
mode = "saas"
# [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex] Legacy local-model switches stay
# off in the Phala ACI integration fixture.
ner_enabled = false
jwt_secret = "a-very-long-secret-key-for-this-test-fixture"

[memchain.saas]
data_root = "/var/lib/aeronyx/memchain-saas-test"

[memchain.supernode]
enabled = true
accepted_compose_hashes = ["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
# [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
accepted_source_provenance = [{ compose_hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", repo_url = "https://example.invalid/phala-gateway", repo_commit = "0123456789abcdef0123456789abcdef01234567" }]
accepted_kms_root_public_keys = ["0x02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]

[[memchain.supernode.providers]]
name = "phala"
type = "phala_aci"
api_key = "$PHALA_API_KEY"
model = "confidential-model"

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

    // ── v2.4.0: Legacy cognitive TOML compatibility ──────────────────────

    #[test]
    fn test_v240_legacy_model_toml_parses_but_fails_validation() {
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
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("memchain.ner_enabled"));
        // [MEMCHAIN-PHALA-CONFIG-GATE 2026-10-06 by Codex] Old task flags
        // remain parseable but cannot silently activate an unwired inference path.
        assert!(error.contains("legacy inference switch is unsupported"));
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
