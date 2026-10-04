// ============================================
// File: crates/aeronyx-server/src/config_reverse_onion.rs
// ============================================
//! Opt-in configuration for private-recipient reverse onion delivery.
//!
//! [REVERSE-ONION-CONFIG 2026-10-04 by Codex] This module validates operator
//! policy only. It neither advertises a capability nor starts a listener.
//! The composition layer must refuse enabled roles until their durable store,
//! authenticated handlers, and recovery worker have initialized successfully.
//! Existing permissionless endpoint and source-sealed reply rules still apply.
//! No model inference, TEE attestation, or public exposure is enabled here.

use std::collections::HashSet;
use std::path::{Component, Path};

use aeronyx_core::crypto::keys::IdentityPublicKey;
use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

/// Independently opt-in queue and outbound recipient roles.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReverseOnionConfig {
    /// Public relay custody role. Does not itself open a listener.
    pub queue: ReverseOnionQueueConfig,
    /// Private recipient polling role. Does not require an inbound listener.
    pub recipient: ReverseOnionRecipientConfig,
}

/// Finite SQLite custody policy; all retained rows count toward quotas.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReverseOnionQueueConfig {
    /// Explicit runtime opt-in.
    pub enabled: bool,
    /// Dedicated durable SQLite file, never an in-memory database.
    pub db_path: String,
    /// Immediate recipients permitted to claim work in the isolated rollout.
    pub recipient_node_ids: Vec<String>,
    /// Maximum number of pending and retained records combined.
    pub max_items: u32,
    /// Per-recipient pending and retained record ceiling.
    pub max_items_per_recipient: u32,
    /// Aggregate stored frame bytes, including leases and results.
    pub max_bytes: u64,
    /// Maximum execution lease duration; retries cannot extend it.
    pub lease_max_secs: u64,
    /// Retention after execution expiry; retention never permits execution.
    pub recovery_retention_secs: u64,
}

impl Default for ReverseOnionQueueConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: String::new(),
            recipient_node_ids: Vec::new(),
            max_items: 256,
            max_items_per_recipient: 16,
            max_bytes: 64 * 1024 * 1024,
            lease_max_secs: 120,
            recovery_retention_secs: 3600,
        }
    }
}

/// One pinned outbound queue; no automatic fallback or endpoint discovery.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReverseOnionRecipientConfig {
    /// Explicit runtime opt-in.
    pub enabled: bool,
    /// Credential-free HTTP(S) public IP literal, with no path or query.
    pub relay_endpoint: String,
    /// Expected adjacent relay Ed25519 identity, exactly 64 hex digits.
    pub relay_node_id: String,
    /// Dedicated durable recipient recovery database.
    pub state_db_path: String,
    /// Minimum delay between polling attempts, including failed polls.
    pub poll_interval_ms: u64,
    /// Bounded request timeout; a timeout does not authorize a new dispatch.
    pub request_timeout_secs: u64,
    /// Finite local pending/recovery record ceiling.
    pub max_pending_items: u32,
}

impl Default for ReverseOnionRecipientConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            relay_endpoint: String::new(),
            relay_node_id: String::new(),
            state_db_path: String::new(),
            poll_interval_ms: 1000,
            request_timeout_secs: 15,
            max_pending_items: 16,
        }
    }
}

impl std::fmt::Debug for ReverseOnionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReverseOnionConfig")
            .field("queue_enabled", &self.queue.enabled)
            .field("recipient_enabled", &self.recipient.enabled)
            .finish_non_exhaustive()
    }
}

impl ReverseOnionConfig {
    /// Validate configuration without resolving DNS or contacting any peer.
    pub fn validate(&self) -> Result<()> {
        let q = &self.queue;
        if q.enabled {
            durable_path(&q.db_path)?;
            if q.recipient_node_ids.is_empty() || q.recipient_node_ids.len() > 64 {
                return Err(invalid("queue requires 1..=64 recipient identity pins"));
            }
            let mut pins = HashSet::new();
            for pin in &q.recipient_node_ids {
                if !pins.insert(node_id(pin)?) {
                    return Err(invalid("duplicate queue recipient identity"));
                }
            }
            if !(1..=4096).contains(&q.max_items)
                || q.max_items_per_recipient == 0
                || q.max_items_per_recipient > q.max_items
                || !(1024 * 1024..=1024 * 1024 * 1024).contains(&q.max_bytes)
                || !(1..=600).contains(&q.lease_max_secs)
                || !(600..=86400).contains(&q.recovery_retention_secs)
            {
                return Err(invalid("queue limits outside bounded rollout policy"));
            }
        }
        let p = &self.recipient;
        if p.enabled {
            durable_path(&p.state_db_path)?;
            node_id(&p.relay_node_id)?;
            // Do not silently discard an operator's query, credentials, or
            // path through the normal peer URL canonicalizer.
            let url = reqwest::Url::parse(&p.relay_endpoint)
                .map_err(|_| invalid("invalid recipient relay endpoint"))?;
            if p.relay_endpoint.trim() != p.relay_endpoint
                || !matches!(url.scheme(), "http" | "https")
                || !url.username().is_empty()
                || url.password().is_some()
                || !matches!(url.path(), "" | "/")
                || url.query().is_some()
                || url.fragment().is_some()
                || url.port() == Some(0)
                || !crate::api::peer_endpoint_is_public_ip(&p.relay_endpoint)
            {
                return Err(invalid("recipient relay requires a plain public IP HTTP(S) origin"));
            }
            if !(250..=60000).contains(&p.poll_interval_ms)
                || !(1..=30).contains(&p.request_timeout_secs)
                || !(1..=64).contains(&p.max_pending_items)
            {
                return Err(invalid("recipient limits outside bounded rollout policy"));
            }
        }
        if q.enabled && p.enabled && q.db_path == p.state_db_path {
            return Err(invalid("queue and recipient databases must be distinct"));
        }
        Ok(())
    }
}

fn invalid(reason: &'static str) -> ServerError {
    ServerError::config_invalid("reverse_onion", reason)
}

fn node_id(value: &str) -> Result<[u8; 32]> {
    let mut bytes = [0u8; 32];
    if value.len() != 64
        || hex::decode_to_slice(value, &mut bytes).is_err()
        || bytes == [0; 32]
        || IdentityPublicKey::from_bytes(&bytes).is_err()
    {
        return Err(invalid("invalid Ed25519 node identity pin"));
    }
    Ok(bytes)
}

fn durable_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.trim() != value
        || !path.is_absolute()
        || value.contains('\0')
        || path.file_name().is_none()
        || value.split('/').any(|part| matches!(part, "." | ".."))
        || path.components().any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(invalid("recovery database requires an absolute normalized file path"));
    }
    // Runtime opening must additionally reject symlinks and aliases to any
    // existing subsystem database; lexical validation cannot prove inode isolation.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::keys::IdentityKeyPair;

    fn recipient_config() -> ReverseOnionConfig {
        let mut config = ReverseOnionConfig::default();
        config.recipient.enabled = true;
        config.recipient.relay_endpoint = "https://8.8.8.8".to_owned();
        config.recipient.relay_node_id =
            hex::encode(IdentityKeyPair::generate().public_key().to_bytes());
        config.recipient.state_db_path = "/Volumes/disk/reverse-onion-test/state.db".to_owned();
        config
    }

    #[test]
    fn omitted_configuration_has_no_active_role() {
        let config: ReverseOnionConfig = toml::from_str("").unwrap();
        assert!(!config.queue.enabled);
        assert!(!config.recipient.enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn recipient_does_not_relax_permissionless_endpoint_policy() {
        let mut config = recipient_config();
        assert!(config.validate().is_ok());
        for endpoint in [
            "https://example.com", "http://127.0.0.1", "http://10.0.0.1",
            "https://user@8.8.8.8", "https://8.8.8.8/api", "https://8.8.8.8/?token=x",
            "https://8.8.8.8/#fragment",
        ] {
            config.recipient.relay_endpoint = endpoint.to_owned();
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn durable_state_and_finite_polling_are_required() {
        let mut config = recipient_config();
        for path in ["", ":memory:", "relative.db", "/Volumes/disk/../state.db"] {
            config.recipient.state_db_path = path.to_owned();
            assert!(config.validate().is_err());
        }
        let mut config = recipient_config();
        config.recipient.poll_interval_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn debug_omits_topology_and_storage_paths() {
        let config = recipient_config();
        let text = format!("{config:?}");
        assert!(!text.contains(&config.recipient.relay_endpoint));
        assert!(!text.contains(&config.recipient.relay_node_id));
        assert!(!text.contains(&config.recipient.state_db_path));
    }
}
