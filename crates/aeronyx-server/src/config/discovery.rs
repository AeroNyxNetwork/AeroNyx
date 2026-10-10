// ============================================
// File: crates/aeronyx-server/src/config/discovery.rs
// ============================================
//! # Discovery configuration
//!
//! Owns the `[discovery]` TOML section, `DiscoveryConfig`: bootstrap
//! snapshots and seeds, the verified peer cache, the Directory Chain store,
//! sync pins and mirrors, delivery and custody witness pins, outbound gossip,
//! peer safety limits, pinned route domains and attestors, the public API
//! listener and identity TLS, permissionless endpoint proof / evidence /
//! attestation inbox / promotion, and self-descriptor advertisement. Also
//! owns the serde default constructors, `Default`, and the decoded pin
//! accessors. Validation lives in `discovery/validation.rs`.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `config.rs`; bodies unchanged.

use std::collections::BTreeMap;
use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use super::PinnedRouteDomainAssignment;

#[cfg(test)]
mod tests;
mod validation;

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
    /// Serves identity-bound TLS on the public API listener's own port.
    ///
    /// [NODE-TLS-BINDING 2026-10-10 by Claude] The listener tells TLS from
    /// plain HTTP by the first byte, so one port serves both and operators
    /// open nothing new. The certificate is self-signed and endorsed by this
    /// node's identity (`aeronyx-node-tls`), which lets clients and peers that
    /// know the node id reach it over HTTPS with neither a domain nor a CA.
    /// Plain HTTP on the same port is unchanged.
    #[serde(default = "DiscoveryConfig::default_public_api_identity_tls")]
    pub public_api_identity_tls: bool,
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
    /// Identity-bound TLS on the public listener is on unless disabled.
    #[must_use]
    pub const fn default_public_api_identity_tls() -> bool {
        true
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
            public_api_identity_tls: Self::default_public_api_identity_tls(),
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
