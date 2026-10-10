// ============================================
// File: crates/aeronyx-server/src/config/discovery/validation.rs
// ============================================
//! # Discovery configuration validation
//!
//! Owns the fail-closed bounds for the `[discovery]` section,
//! `DiscoveryConfig::validate`, the node-identity pin-set validator shared by
//! the delivery and custody witness policies, the startup runtime-identity
//! check, and seed endpoint syntax checks.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `config.rs`; bodies unchanged.

use crate::error::{Result, ServerError};

use super::DiscoveryConfig;

#[cfg(test)]
mod tests;

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

impl DiscoveryConfig {
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

        if self.descriptor_ttl_secs < 60 {
            return Err(ServerError::config_invalid(
                "discovery.descriptor_ttl_secs",
                "must be at least 60 seconds",
            ));
        }

        Ok(())
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
