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
//! Last Modified: v0.2.0-SignedPrivateRecipientConfig — canonical authority
//! inputs and bounded source admission fields remain default-off.

// [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Signed route bundles
// are optional cache seeds. Operator-pinned identities and relay origin are
// stable bootstrap constraints; PeerStore supplies live dispatch authority.

use std::collections::HashSet;
use std::path::{Component, Path};

use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::protocol::discovery::{
    SignedNodeDescriptor, SignedPrivateOnionRecipientAuthorizationV1,
    MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES, MAX_SIGNED_NODE_DESCRIPTOR_BYTES,
};
use aeronyx_core::protocol::discovery::MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_LIFETIME_SECS_V1;
use aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_NO_WORK_MARKER_MIN_RETENTION_SECS;
use aeronyx_core::protocol::onion::reverse_delivery::MAX_REVERSE_ONION_RECOVERY_RETENTION_SECS;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
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
    /// Internal source role; never creates a public dispatch endpoint.
    pub source: ReverseOnionSourceConfig,
}

// [REVERSE-ONION-SOURCE-COMPOSITION 2026-10-05 by Codex] Independent
// source storage and authority prevent accidental reuse of a custody database.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReverseOnionSourceConfig {
    pub enabled: bool,
    /// [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Evidence recovery only;
    /// dispatch cannot create a new session or arm a new journal row.
    pub recovery_only: bool,
    pub state_db_path: String,
    /// Stable relay identity and HTTPS origin used only as the configured
    /// route pin/bootstrap target. Current signed descriptors still come from
    /// authenticated discovery before any new dispatch.
    pub relay_node_id: String,
    pub relay_endpoint: String,
    /// Stable private-recipient identity. The signed descriptor/grant are
    /// refreshed through authenticated discovery and never caller supplied.
    pub recipient_node_id: String,
    /// Optional historical signed seed bundle for offline cache bootstrap.
    pub relay_descriptor_b64: String,
    pub recipient_descriptor_b64: String,
    pub recipient_authorization_b64: String,
    pub max_pending_items: usize,
    pub max_bytes: u64,
    pub max_in_flight: usize,
    pub request_timeout_secs: u64,
    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex]
    /// Bounded evidence polling after custody while the terminal produces a result.
    pub result_wait_secs: u64,
    pub result_poll_interval_ms: u64,
    /// Maximum age of a newly admitted route, further clipped by signed authority and envelope freshness.
    // [REVERSE-SOURCE-LEASE-CAP 2026-10-05 by Codex] Never caller supplied.
    pub lease_max_secs: u64,
}

impl Default for ReverseOnionSourceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            recovery_only: false,
            state_db_path: String::new(),
            relay_node_id: String::new(),
            relay_endpoint: String::new(),
            recipient_node_id: String::new(),
            relay_descriptor_b64: String::new(),
            recipient_descriptor_b64: String::new(),
            recipient_authorization_b64: String::new(),
            max_pending_items: 16,
            max_bytes: 64 * 1024 * 1024,
            max_in_flight: 4,
            request_timeout_secs: 15,
            result_wait_secs: 5,
            result_poll_interval_ms: 250,
            lease_max_secs: 120,
        }
    }
}

impl ReverseOnionSourceConfig {
    pub(crate) fn identity_pins(&self) -> Result<([u8; 32], [u8; 32], String)> {
        let relay_id = if self.relay_node_id.is_empty() {
            decode_descriptor_blob(&self.relay_descriptor_b64)?.node_id()
        } else { node_id(&self.relay_node_id)? };
        let recipient_id = if self.recipient_node_id.is_empty() {
            decode_descriptor_blob(&self.recipient_descriptor_b64)?.node_id()
        } else { node_id(&self.recipient_node_id)? };
        let endpoint = if self.relay_endpoint.is_empty() {
            decode_descriptor_blob(&self.relay_descriptor_b64)?
                .descriptor.public_endpoint.ok_or_else(|| invalid("source relay endpoint pin is required"))?
        } else { self.relay_endpoint.clone() };
        validate_source_endpoint(&endpoint)?;
        if relay_id == recipient_id { return Err(invalid("source relay and recipient pins collide")); }
        Ok((relay_id, recipient_id, endpoint))
    }

    pub(crate) fn optional_authority_material(&self) -> Result<Option<(
        SignedNodeDescriptor, SignedNodeDescriptor, SignedPrivateOnionRecipientAuthorizationV1,
    )>> {
        let fields = [&self.relay_descriptor_b64, &self.recipient_descriptor_b64,
            &self.recipient_authorization_b64];
        if fields.iter().all(|value| value.is_empty()) { return Ok(None); }
        if fields.iter().any(|value| value.is_empty()) {
            return Err(invalid("source historical authority seed must be complete"));
        }
        Ok(Some((decode_descriptor_blob(&self.relay_descriptor_b64)?,
            decode_descriptor_blob(&self.recipient_descriptor_b64)?,
            decode_authorization_blob(&self.recipient_authorization_b64)?)))
    }
}

fn validate_source_endpoint(endpoint: &str) -> Result<()> {
    // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Reuse reqwest's
    // URL type instead of adding a redundant direct dependency.
    let url = reqwest::Url::parse(endpoint).map_err(|_| invalid("source relay endpoint pin is invalid"))?;
    if endpoint.trim() != endpoint || url.scheme() != "https"
        || !url.username().is_empty() || url.password().is_some()
        || !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some()
        || url.port() == Some(0)
        || !crate::api::reverse_onion_endpoint_supported(endpoint)
    {
        return Err(invalid("source relay endpoint pin must be a public HTTPS origin"));
    }
    Ok(())
}

fn same_https_origin(left: &str, right: &str) -> bool {
    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Historical seeds
    // must pass the same public HTTPS origin gate used by live transport.
    crate::api::reverse_onion_same_origin(left, right)
}

/// Finite SQLite custody policy; all retained rows count toward quotas.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReverseOnionQueueConfig {
    /// Explicit runtime opt-in.
    pub enabled: bool,
    /// [REVERSE-RECOVERY-BOOT 2026-10-05 by Codex] Recover existing signed
    /// custody only. No new enqueue or lease authority is constructed.
    pub recovery_only: bool,
    /// Dedicated durable SQLite file, never an in-memory database.
    pub db_path: String,
    /// Exactly one private-recipient identity pin for live queue operation.
    pub recipient_node_ids: Vec<String>,
    /// Canonical signed R descriptor, encoded with standard Base64. This
    /// seeds the stable relay identity; live work always checks PeerStore.
    /// Expiry does not prevent durable queue recovery or gossip startup.
    #[serde(default)]
    pub relay_descriptor_b64: String,
    /// Canonical signed P descriptor, encoded with standard Base64. This is
    /// historical identity/KEM seed material; it may expire between restarts.
    /// It never authorizes a new lease; current R/P descriptors and P's
    /// current grant are required at the request boundary.
    #[serde(default)]
    pub recipient_descriptor_b64: String,
    /// Canonical P-signed private-recipient authorization, standard Base64.
    #[serde(default)]
    pub recipient_authorization_b64: String,
    /// Explicit source identities permitted to enqueue this experiment.
    #[serde(default)]
    pub source_node_ids: Vec<String>,
    /// Shared bounded HTTP blocking admission for the queue API.
    #[serde(default = "default_reverse_onion_max_in_flight")]
    pub max_in_flight: usize,
    /// Maximum number of pending and retained records combined.
    pub max_items: u32,
    /// Per-recipient pending and retained record ceiling.
    pub max_items_per_recipient: u32,
    /// Aggregate stored frame bytes, including leases and results.
    pub max_bytes: u64,
    /// Maximum execution lease duration; retries cannot extend it.
    pub lease_max_secs: u64,
    /// Maximum accepted signed-route validity window, independent of execution.
    #[serde(default = "default_reverse_onion_route_max_secs")]
    pub route_max_secs: u64,
    /// Retention after execution expiry; retention never permits execution.
    pub recovery_retention_secs: u64,
}

fn default_reverse_onion_route_max_secs() -> u64 {
    600
}

impl Default for ReverseOnionQueueConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: String::new(),
            recovery_only: false,
            recipient_node_ids: Vec::new(),
            relay_descriptor_b64: String::new(),
            recipient_descriptor_b64: String::new(),
            recipient_authorization_b64: String::new(),
            source_node_ids: Vec::new(),
            max_in_flight: default_reverse_onion_max_in_flight(),
            max_items: 256,
            max_items_per_recipient: 16,
            max_bytes: 64 * 1024 * 1024,
            lease_max_secs: 120,
            route_max_secs: default_reverse_onion_route_max_secs(),
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
    /// [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Resume the existing
    /// journal without allocating new Claim identifiers on idle passes.
    pub recovery_only: bool,
    /// Credential-free HTTPS public IP literal or DNS hostname, with no path or query.
    /// [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Immutable origin
    /// pin for bootstrap gossip and task/recovery transport. Signed descriptor
    /// renewal may update keys but cannot change this configured origin.
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
            recovery_only: false,
            relay_endpoint: String::new(),
            relay_node_id: String::new(),
            state_db_path: String::new(),
            poll_interval_ms: 1000,
            request_timeout_secs: 15,
            max_pending_items: 16,
        }
    }
}

impl ReverseOnionRecipientConfig {
    // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
    /// Fresh Claim creation is separate from recovery of exact journal bytes.
    /// A discovery renewal must never lift an operator recovery hold.
    pub(crate) const fn permits_new_claims(&self) -> bool {
        self.enabled && !self.recovery_only
    }
}

impl std::fmt::Debug for ReverseOnionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReverseOnionConfig")
            .field("queue_enabled", &self.queue.enabled)
            .field("recipient_enabled", &self.recipient.enabled)
            .field("source_enabled", &self.source.enabled)
            .finish_non_exhaustive()
    }
}

impl ReverseOnionConfig {
    // [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] The real caller
    // supplies rustc's compile-time panic policy, never an environment toggle.
    // A private policy helper lets unexecuted regression source cover abort too.
    fn validate_build_strategy(&self, supports_unwind: bool) -> Result<()> {
        validate_reverse_unwind_support(
            self.queue.enabled || self.recipient.enabled || self.source.enabled,
            supports_unwind,
        )
    }

    // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Recovery-only roles must
    // remain usable without a live authority renewal channel. They may read
    // and forward already durable evidence, but cannot create new work.
    pub(crate) fn requires_live_authority_gossip(&self) -> bool {
        (self.queue.enabled && !self.queue.recovery_only)
            || self.recipient.permits_new_claims()
            || (self.source.enabled && !self.source.recovery_only)
    }

    /// Validate configuration without resolving DNS or contacting any peer.
    pub fn validate(&self) -> Result<()> {
        // [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] Recovery
        // roles also own accepted blocking work; abort cannot honor its drain.
        self.validate_build_strategy(cfg!(panic = "unwind"))?;
        let q = &self.queue;
        if q.enabled {
            durable_path(&q.db_path)?;
            // [PHALA-REVERSE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex]
            // One process owns one recipient-specific queue/API view.
            if q.recipient_node_ids.len() != 1 {
                return Err(invalid("queue requires exactly one explicit recipient pin"));
            }
            if !q.recovery_only && q.source_node_ids.is_empty() {
                return Err(invalid("live queue requires an explicit source identity allowlist"));
            }
            if q.recipient_node_ids.is_empty() || q.recipient_node_ids.len() > 64 {
                return Err(invalid("queue requires 1..=64 recipient identity pins"));
            }
            let mut pins = HashSet::new();
            for pin in &q.recipient_node_ids {
                if !pins.insert(node_id(pin)?) {
                    return Err(invalid("duplicate queue recipient identity"));
                }
            }
            if !(1..=64).contains(&q.max_in_flight) {
                return Err(invalid("queue max_in_flight outside bounded rollout policy"));
            }
            let authority_fields = [
                !q.relay_descriptor_b64.is_empty(),
                !q.recipient_descriptor_b64.is_empty(),
                !q.recipient_authorization_b64.is_empty(),
            ];
            if authority_fields.iter().any(|present| *present) {
                if authority_fields.iter().any(|present| !*present)
                    || q.recipient_node_ids.len() != 1
                {
                    return Err(invalid("queue signed recipient authority is incomplete"));
                }
                if q.source_node_ids.is_empty() || q.source_node_ids.len() > 64 {
                    return Err(invalid("queue requires 1..=64 source identity pins"));
                }
                // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                // Runtime batching must parse the exact same bounded pins.
                let sources = q.source_identity_pins()?;
                let relay_descriptor = decode_descriptor_blob(&q.relay_descriptor_b64)?;
                let recipient_descriptor = decode_descriptor_blob(&q.recipient_descriptor_b64)?;
                let authorization = decode_authorization_blob(&q.recipient_authorization_b64)?;
                let relay_node_id = relay_descriptor.node_id();
                let recipient_node_id = recipient_descriptor.node_id();
                if relay_node_id == [0; 32]
                    || recipient_node_id == [0; 32]
                    || relay_node_id == recipient_node_id
                    || recipient_node_id != node_id(&q.recipient_node_ids[0])?
                    || authorization.relay_node_id() != relay_node_id
                    || authorization.recipient_node_id() != recipient_node_id
                {
                    return Err(invalid("queue signed recipient authority bindings are invalid"));
                }
                if sources
                    .iter()
                    .any(|source| *source == relay_node_id || *source == recipient_node_id)
                {
                    return Err(invalid("queue source identity overlaps route authority"));
                }
            } else if !q.source_node_ids.is_empty() {
                if q.source_node_ids.len() > 64 {
                    return Err(invalid("queue source identity pins exceed rollout bound"));
                }
                // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                // Identity-only bootstrap uses the same complete pin parser.
                let sources = q.source_identity_pins()?;
                if sources.iter().any(|source| pins.contains(source)) {
                    return Err(invalid("duplicate queue source identity"));
                }
            }
            if !(1..=4096).contains(&q.max_items)
                || q.max_items_per_recipient == 0
                || q.max_items_per_recipient > q.max_items
                || !(1024 * 1024..=1024 * 1024 * 1024).contains(&q.max_bytes)
                || !(1..=600).contains(&q.lease_max_secs)
                || !(1..=MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_LIFETIME_SECS_V1)
                    .contains(&q.route_max_secs)
                // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] The
                // no-work marker must outlive the recipient's full poll evidence
                // horizon so a lost signed receipt can be reissued after restart.
                || !(REVERSE_ONION_NO_WORK_MARKER_MIN_RETENTION_SECS
                    ..=MAX_REVERSE_ONION_RECOVERY_RETENTION_SECS)
                    .contains(&q.recovery_retention_secs)
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
            // [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] DNS is accepted
            // only for this identity-pinned private route; transport resolves
            // and pins all-public answers before any request is sent.
            // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Recipient
            // journals advance only on authenticated relay responses.
            if p.relay_endpoint.trim() != p.relay_endpoint
                || url.scheme() != "https"
                || !url.username().is_empty()
                || url.password().is_some()
                || !matches!(url.path(), "" | "/")
                || url.query().is_some()
                || url.fragment().is_some()
                || url.port() == Some(0)
                || !crate::api::reverse_onion_endpoint_supported(&p.relay_endpoint)
            {
                return Err(invalid("recipient relay requires an HTTPS public IP or hostname origin"));
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
        let s = &self.source;
        if s.enabled {
            durable_path(&s.state_db_path)?;
            if !(1..=64).contains(&s.max_pending_items)
                || !(1024 * 1024..=512 * 1024 * 1024).contains(&s.max_bytes)
                || !(1..=64).contains(&s.max_in_flight)
                || !(1..=60).contains(&s.request_timeout_secs)
                || s.result_wait_secs > 30
                || !(100..=5000).contains(&s.result_poll_interval_ms)
                || !(1..=600).contains(&s.lease_max_secs)
            {
                return Err(invalid("source limits outside bounded rollout policy"));
            }
            let (relay_id, recipient_id, endpoint) = s.identity_pins()?;
            if let Some((relay, recipient, authorization)) = s.optional_authority_material()? {
                if relay.node_id() != relay_id || recipient.node_id() != recipient_id
                    || authorization.relay_node_id() != relay_id
                    || authorization.recipient_node_id() != recipient_id
                    || relay.descriptor.public_endpoint.as_deref().is_none_or(|seed|
                        !same_https_origin(seed, &endpoint))
                {
                    return Err(invalid("source historical authority seed does not match identity/origin pins"));
                }
            }
            if (q.enabled && s.state_db_path == q.db_path)
                || (p.enabled && s.state_db_path == p.state_db_path)
            {
                return Err(invalid("source database must be distinct"));
            }
        }
        Ok(())
    }
}

// [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] The runtime catch
// boundary reuses this gate, even for internal callers without parsed config.
// There is no production argument that can assert another build strategy.
pub(crate) fn require_reverse_onion_unwind_support() -> Result<()> {
    validate_reverse_unwind_support(true, cfg!(panic = "unwind"))
}

fn validate_reverse_unwind_support(enabled: bool, supports_unwind: bool) -> Result<()> {
    if enabled && !supports_unwind {
        return Err(invalid("reverse roles require a panic=unwind binary; use the phala build profile"));
    }
    Ok(())
}

fn invalid(reason: &'static str) -> ServerError {
    ServerError::config_invalid("reverse_onion", reason)
}

fn default_reverse_onion_max_in_flight() -> usize {
    4
}

// [PHALA-QUEUE-ENV-BOUNDS 2026-10-06 by Codex] Bound JSON before parsing IDs.
const MAX_PHALA_QUEUE_ID_JSON_BYTES: usize = 8 * 1024;

impl ReverseOnionQueueConfig {
    // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Local identity is
    // known only at composition time. Validate the entire decoded policy
    // before any PeerStore pin or durable-store side effect.
    pub(crate) fn live_identity_pins(
        &self, local_relay: [u8; 32],
    ) -> Result<crate::services::peer_store::PrivateOnionQueueIdentityPins> {
        if !self.enabled || self.recovery_only || self.recipient_node_ids.len() != 1 {
            return Err(invalid("live queue identity policy is unavailable"));
        }
        crate::services::peer_store::PrivateOnionQueueIdentityPins::new(
            local_relay, node_id(&self.recipient_node_ids[0])?, self.source_identity_pins()?,
        ).map_err(|_| invalid("live queue identities are invalid, duplicated, or overlap"))
    }

    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Parse the
    // whole operator allowlist or reject it. A malformed/duplicate pin must
    // never silently become a smaller, partially accepted gossip policy.
    pub(crate) fn source_identity_pins(&self) -> Result<Vec<[u8; 32]>> {
        if self.source_node_ids.len() > 64 {
            return Err(invalid("queue source identity pins exceed rollout bound"));
        }
        let mut seen = HashSet::new();
        self.source_node_ids.iter().map(|value| {
            let pin = node_id(value)?;
            if !seen.insert(pin) {
                return Err(invalid("duplicate queue source identity"));
            }
            Ok(pin)
        }).collect()
    }

    // [PHALA-QUEUE-RECOVERY-GATE 2026-10-06 by Codex] Recipient-bound
    // attestation is only advertised when this queue accepts fresh work.
    pub(crate) const fn permits_new_claims(&self) -> bool {
        self.enabled && !self.recovery_only
    }

    // [PHALA-REVERSE-QUEUE-CONFIG 2026-10-06 by Codex] Phala's public peer
    // image is immutable, so its protected environment must preserve the
    // distinction between current live admission and durable recovery.
    pub(crate) fn apply_phala_environment(
        &mut self,
        enabled: Option<&str>,
        recovery_only: Option<&str>,
        recipient_ids_json: Option<&str>,
        source_ids_json: Option<&str>,
        relay_descriptor_b64: Option<&str>,
        recipient_descriptor_b64: Option<&str>,
        authorization_b64: Option<&str>,
        relay_role_enabled: bool,
        private_recipient_enabled: bool,
    ) -> Result<()> {
        let requested_recovery_only = match recovery_only {
            None => self.recovery_only,
            Some("true") => true,
            Some("false") => false,
            Some(_) => return Err(invalid("Phala queue recovery mode must be exactly 'true' or 'false'")),
        };
        let identity_and_source_inputs = [recipient_ids_json, source_ids_json];
        let authority_inputs = [
            relay_descriptor_b64,
            recipient_descriptor_b64,
            authorization_b64,
        ];
        let Some(enabled) = enabled else {
            if identity_and_source_inputs.iter().chain(authority_inputs.iter())
                .any(|value| value.is_some()) || recovery_only.is_some()
            {
                return Err(invalid("Phala queue inputs require an explicit enable setting"));
            }
            return Ok(());
        };
        match enabled {
            "false" => {
                if identity_and_source_inputs.iter().chain(authority_inputs.iter())
                    .any(|value| value.is_some_and(|item| !item.is_empty()))
                    || recovery_only == Some("true")
                {
                    return Err(invalid("disabled Phala queue must not receive authority inputs"));
                }
                self.enabled = false;
                // [PHALA-QUEUE-EXPLICIT-OFF 2026-10-06 by Codex] An explicit
                // disable wins over an inherited TOML recovery mode; only an
                // explicitly contradictory recovery=true remains invalid.
                self.recovery_only = false;
                return Ok(());
            }
            "true" => {}
            _ => return Err(invalid("Phala queue enable must be exactly 'true' or 'false'")),
        }
        if recipient_ids_json.is_some_and(|value| value.len() > MAX_PHALA_QUEUE_ID_JSON_BYTES)
            || source_ids_json.is_some_and(|value| value.len() > MAX_PHALA_QUEUE_ID_JSON_BYTES)
        {
            return Err(invalid("Phala queue identity input exceeds 8192 bytes"));
        }
        if !relay_role_enabled || private_recipient_enabled {
            return Err(invalid("Phala queue requires the public relay role only"));
        }
        if recipient_ids_json.is_none_or(str::is_empty) {
            return Err(invalid("Phala queue requires a recipient identity pin"));
        }
        if requested_recovery_only {
            if source_ids_json.is_some_and(|value| !value.is_empty() && value != "[]")
                || authority_inputs.iter().any(|value| value.is_some_and(|item| !item.is_empty()))
            {
                return Err(invalid("recovery-only Phala queue must not receive live source or authority inputs"));
            }
        } else if source_ids_json.is_none_or(str::is_empty) {
            return Err(invalid("live Phala queue requires source identity pins"));
        }
        let authority_count = authority_inputs.iter().filter(|value| {
            value.is_some_and(|item| !item.is_empty())
        }).count();
        if authority_count != 0 && authority_count != authority_inputs.len() {
            return Err(invalid("Phala queue signed authority seed must be complete"));
        }
        let recipient_ids: Vec<String> = serde_json::from_str(recipient_ids_json.unwrap())
            .map_err(|_| invalid("Phala queue recipients must be a JSON identity array"))?;
        let source_ids: Vec<String> = match source_ids_json.filter(|value| !value.is_empty()) {
            Some(value) => serde_json::from_str(value)
                .map_err(|_| invalid("Phala queue sources must be a JSON identity array"))?,
            None => Vec::new(),
        };
        if recipient_ids.len() != 1 || source_ids.len() > 64
            || (!requested_recovery_only && source_ids.is_empty())
        {
            return Err(invalid("Phala queue requires one recipient and a valid source policy"));
        }
        self.enabled = true;
        self.recovery_only = requested_recovery_only;
        self.recipient_node_ids = recipient_ids;
        self.source_node_ids = source_ids;
        self.relay_descriptor_b64 = relay_descriptor_b64.unwrap_or_default().to_owned();
        self.recipient_descriptor_b64 = recipient_descriptor_b64.unwrap_or_default().to_owned();
        self.recipient_authorization_b64 = authorization_b64.unwrap_or_default().to_owned();
        if self.db_path.is_empty() {
            self.db_path = "/var/lib/aeronyx/reverse-onion-queue.sqlite".to_owned();
        }
        Ok(())
    }

    /// Returns whether a complete optional historical signed seed was supplied.
    /// Actual live authority always comes from current authenticated gossip.
    pub(crate) fn signed_private_admission_configured(&self) -> bool {
        !self.relay_descriptor_b64.is_empty()
            || !self.recipient_descriptor_b64.is_empty()
            || !self.recipient_authorization_b64.is_empty()
    }

    /// Decode the complete signed R/P authority bundle for startup composition.
    /// Shape validation remains in [`ReverseOnionConfig::validate`]. Runtime
    /// callers verify this signed historical identity bundle at its issuance
    /// time; current TTL, route roles, and grant validity gate each new effect.
    pub(crate) fn signed_private_authority_material(
        &self,
    ) -> Result<(
        SignedNodeDescriptor,
        SignedNodeDescriptor,
        SignedPrivateOnionRecipientAuthorizationV1,
    )> {
        if !self.signed_private_admission_configured()
            || self.relay_descriptor_b64.is_empty()
            || self.recipient_descriptor_b64.is_empty()
            || self.recipient_authorization_b64.is_empty()
        {
            return Err(invalid("queue signed recipient authority is required"));
        }
        Ok((
            decode_descriptor_blob(&self.relay_descriptor_b64)?,
            decode_descriptor_blob(&self.recipient_descriptor_b64)?,
            decode_authorization_blob(&self.recipient_authorization_b64)?,
        ))
    }
}

fn decode_descriptor_blob(value: &str) -> Result<SignedNodeDescriptor> {
    let bytes = decode_canonical_blob(value, MAX_SIGNED_NODE_DESCRIPTOR_BYTES)?;
    SignedNodeDescriptor::decode_canonical(&bytes)
        .map_err(|_| invalid("invalid canonical reverse-onion descriptor"))
}

fn decode_authorization_blob(
    value: &str,
) -> Result<SignedPrivateOnionRecipientAuthorizationV1> {
    let bytes = decode_canonical_blob(
        value,
        MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES,
    )?;
    SignedPrivateOnionRecipientAuthorizationV1::decode_canonical(&bytes)
        .map_err(|_| invalid("invalid canonical reverse-onion authorization"))
}

fn decode_canonical_blob(value: &str, max_bytes: usize) -> Result<Vec<u8>> {
    if value.is_empty() {
        return Err(invalid("reverse-onion signed authority is required"));
    }
    let max_encoded = max_bytes
        .checked_add(2)
        .and_then(|value| value.checked_div(3))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("reverse-onion signed authority bound is invalid"))?;
    if value.len() > max_encoded {
        return Err(invalid("reverse-onion signed authority exceeds bound"));
    }
    let bytes = BASE64
        .decode(value)
        .map_err(|_| invalid("reverse-onion signed authority encoding is invalid"))?;
    if bytes.is_empty() || bytes.len() > max_bytes || BASE64.encode(&bytes) != value {
        return Err(invalid("reverse-onion signed authority encoding is non-canonical"));
    }
    Ok(bytes)
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

    // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Authored, not run:
    // composition validates decoded roles without tightening legacy hex case.
    #[test]
    fn live_queue_pins_are_role_distinct_and_never_available_in_recovery() {
        let id = |seed| IdentityKeyPair::from_bytes(&[seed; 32]).unwrap().public_key_bytes();
        let (relay, recipient, source) = (id(91), id(92), id(93));
        let mut queue = ReverseOnionQueueConfig::default();
        queue.enabled = true;
        queue.recipient_node_ids = vec![hex::encode(recipient).to_uppercase()];
        queue.source_node_ids = vec![hex::encode(source).to_uppercase()];
        let pins = queue.live_identity_pins(relay).unwrap();
        assert_eq!(pins.relay(), relay);
        assert_eq!(pins.recipient(), recipient);
        assert_eq!(pins.sources(), [source].as_slice());
        for sources in [Vec::new(), vec![hex::encode(source), hex::encode(source).to_uppercase()],
            vec![hex::encode(relay)], vec![hex::encode(recipient)]]
        {
            queue.source_node_ids = sources;
            assert!(queue.live_identity_pins(relay).is_err());
        }
        queue.source_node_ids = vec![hex::encode(source)];
        assert!(queue.live_identity_pins(recipient).is_err());
        queue.recovery_only = true;
        assert!(queue.live_identity_pins(relay).is_err());
        queue.recovery_only = false;
        queue.enabled = false;
        assert!(queue.live_identity_pins(relay).is_err());
    }

    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Authored,
    // not run: source parsing is identical in configuration and scheduling.
    #[test]
    fn queue_source_identity_pins_are_ordered_bounded_and_all_or_nothing() {
        let first = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap().public_key_bytes();
        let second = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap().public_key_bytes();
        let mut queue = ReverseOnionQueueConfig::default();
        assert!(queue.source_identity_pins().unwrap().is_empty());
        queue.source_node_ids = vec![hex::encode(first), hex::encode(second)];
        assert_eq!(queue.source_identity_pins().unwrap(), vec![first, second]);
        // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Hex case has
        // always been accepted; uniqueness is over decoded identities.
        queue.source_node_ids = vec![hex::encode(first).to_uppercase(), hex::encode(second)];
        assert_eq!(queue.source_identity_pins().unwrap(), vec![first, second]);
        queue.source_node_ids = vec![hex::encode(first), hex::encode(first).to_uppercase()];
        assert!(queue.source_identity_pins().is_err());
        for invalid in [String::new(), "00".repeat(32), "not-an-identity".into()] {
            queue.source_node_ids = vec![hex::encode(first), invalid];
            assert!(queue.source_identity_pins().is_err());
        }
        queue.source_node_ids = (1..=64).map(|seed| {
            hex::encode(IdentityKeyPair::from_bytes(&[seed; 32]).unwrap().public_key_bytes())
        }).collect();
        assert_eq!(queue.source_identity_pins().unwrap().len(), 64);
        queue.source_node_ids.push(hex::encode(first));
        assert!(queue.source_identity_pins().is_err());
    }

    // [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] Authored, not
    // run: every role combination and recovery mode rejects abort, while a
    // fully disabled config keeps the existing release build compatible.
    #[test]
    fn reverse_roles_require_unwind_for_live_and_recovery_work() {
        for roles in 0..8 {
            for recovery_only in [false, true] {
                let mut config = ReverseOnionConfig::default();
                config.queue.enabled = roles & 1 != 0;
                config.recipient.enabled = roles & 2 != 0;
                config.source.enabled = roles & 4 != 0;
                config.queue.recovery_only = recovery_only;
                config.recipient.recovery_only = recovery_only;
                config.source.recovery_only = recovery_only;
                assert!(config.validate_build_strategy(true).is_ok());
                let abort = config.validate_build_strategy(false);
                if roles == 0 {
                    assert!(abort.is_ok());
                    assert!(config.validate().is_ok());
                } else {
                    assert!(matches!(abort, Err(ServerError::ConfigInvalid { field, reason })
                        if field == "reverse_onion" && reason ==
                            "reverse roles require a panic=unwind binary; use the phala build profile"));
                }
            }
        }
        assert_eq!(require_reverse_onion_unwind_support().is_ok(), cfg!(panic = "unwind"));
    }

    // [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] Authored, not
    // run: keep build selection and copied artifact tied to the dedicated
    // profile, without changing ordinary release or claiming binary evidence.
    #[test]
    fn phala_image_selects_the_unwind_profile_and_its_exact_artifact() {
        let manifest: toml::Value = toml::from_str(include_str!("../../../Cargo.toml")).unwrap();
        assert_eq!(manifest["profile"]["release"]["panic"].as_str(), Some("abort"));
        assert_eq!(manifest["profile"]["phala"]["inherits"].as_str(), Some("release"));
        assert_eq!(manifest["profile"]["phala"]["panic"].as_str(), Some("unwind"));
        let dockerfile = include_str!("../../../deploy/node/Dockerfile");
        let lines: Vec<_> = dockerfile.lines().map(str::trim).collect();
        assert_eq!(lines.iter().filter(|line| line.starts_with("RUN cargo build ")).count(), 1);
        assert!(lines.contains(&"RUN cargo build --profile phala --locked -p aeronyx-server"));
        assert!(lines.contains(&"COPY --from=builder /workspace/target/phala/aeronyx-server /usr/local/bin/aeronyx-server"));
        assert!(!lines.iter().any(|line| line.starts_with("COPY --from=builder /workspace/target/release/")));
    }

    // [PHALA-BUILD-CONTEXT-ALLOWLIST 2026-10-08 by Codex] Authored, not
    // run: this is a source-contract regression, not a replacement Docker
    // matcher or proof of the bytes sent by an actual builder.
    #[test]
    fn phala_build_context_admits_only_declared_workspace_inputs() {
        let rules: Vec<_> = include_str!("../../../.dockerignore").lines()
            .map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#')).collect();
        assert_eq!(rules.first().copied(), Some("**"));
        let admitted: Vec<_> = rules.iter().copied().filter(|line| line.starts_with('!')).collect();
        let expected = [
            "!Cargo.toml", "!Cargo.lock", "!rust-toolchain.toml",
            "!crates/aeronyx-blind-issuer/Cargo.toml", "!crates/aeronyx-blind-issuer/src/**",
            "!crates/aeronyx-core/Cargo.toml", "!crates/aeronyx-core/src/**",
            "!crates/aeronyx-transport/Cargo.toml", "!crates/aeronyx-transport/src/**",
            "!crates/aeronyx-server/Cargo.toml", "!crates/aeronyx-server/src/**",
            "!crates/aeronyx-common/Cargo.toml", "!crates/aeronyx-common/src/**",
            "!deploy/node/server.phala.peer.example.toml",
        ];
        assert_eq!(admitted, expected);
        let last_admission = rules.iter().rposition(|line| line.starts_with('!')).unwrap();
        for excluded in ["**/.git", "**/target", "**/.codex-tmp", "**/.codex-cache",
            "**/.codex-target", "**/.env", "**/.env.*", "**/server_key.json",
            "**/node_info.json", "**/.aeronyx-identity-*.pending", "**/*.db", "**/*.db-*",
            "**/*.sqlite*", "**/secrets"]
        {
            assert!(rules.iter().position(|line| *line == excluded).unwrap() > last_admission);
        }
        let manifest: toml::Value = toml::from_str(include_str!("../../../Cargo.toml")).unwrap();
        for member in manifest["workspace"]["members"].as_array().unwrap() {
            let member = member.as_str().unwrap();
            for input in [format!("!{member}/Cargo.toml"), format!("!{member}/src/**")] {
                assert!(admitted.contains(&input.as_str()), "unreviewed image workspace input: {member}");
            }
        }
        let dockerfile = include_str!("../../../deploy/node/Dockerfile");
        let copies: Vec<_> = dockerfile.lines().map(str::trim)
            .filter(|line| line.starts_with("COPY ")).collect();
        assert_eq!(copies, [
            "COPY Cargo.toml Cargo.lock rust-toolchain.toml ./",
            "COPY crates ./crates",
            "COPY --from=builder /workspace/target/phala/aeronyx-server /usr/local/bin/aeronyx-server",
            "COPY deploy/node/server.phala.peer.example.toml /etc/aeronyx/server.toml",
        ]);
        assert!(!dockerfile.lines().any(|line| line.trim_start().starts_with("ADD ")));
    }

    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Authored, unexecuted;
    // config seeds use the same normalization and invalid-origin rejection as IO.
    #[test]
    fn private_authority_seed_origin_uses_the_transport_gate() {
        assert!(same_https_origin("https://relay.example.net", "https://RELAY.example.net:443/"));
        assert!(!same_https_origin("https://relay.example.net", "https://relay.example.net:8443"));
        for endpoint in ["http://relay.example.net", "https://relay.internal",
            "https://127.0.0.1", "https://user@relay.example.net"] {
            assert!(!same_https_origin(endpoint, endpoint));
        }
    }

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
        assert!(!config.source.enabled);
        assert!(!config.source.recovery_only);
        assert!(!config.recipient.recovery_only);
        assert!(!config.queue.recovery_only);
        assert!(config.validate().is_ok());
    }

    // [PHALA-REVERSE-QUEUE-CONFIG 2026-10-06 by Codex]
    #[test]
    fn phala_queue_environment_is_explicit_complete_and_role_bound() {
        let mut queue = ReverseOnionQueueConfig::default();
        assert!(queue.apply_phala_environment(
            None, None, None, None, None, None, None, false, false,
        ).is_ok());
        assert!(queue.apply_phala_environment(
            Some("yes"), None, None, None, None, None, None, true, false,
        ).is_err());
        assert!(queue.apply_phala_environment(
            Some("true"), None, Some("[]"), Some("[]"), Some("r"), Some("p"), Some("a"), true, false,
        ).is_err());
        assert!(queue.apply_phala_environment(
            Some("true"), None, Some("[]"), Some("[]"), Some("r"), Some("p"), Some("a"), false, false,
        ).is_err());
        let recipient = hex::encode(IdentityKeyPair::from_bytes(&[31; 32]).unwrap().public_key_bytes());
        let source = hex::encode(IdentityKeyPair::from_bytes(&[32; 32]).unwrap().public_key_bytes());
        let recipient_json = format!("[\"{recipient}\"]");
        let source_json = format!("[\"{source}\"]");
        assert!(queue.apply_phala_environment(
            Some("true"), None, Some(&recipient_json), Some(&source_json),
            None, None, None, true, false,
        ).is_ok());
        assert!(queue.enabled);
        assert_eq!(queue.recipient_node_ids, vec![recipient]);
        assert!(!queue.signed_private_admission_configured());
    }

    // [PHALA-RENDER-REQUIRED-PINS 2026-10-08 by Codex] Authored only:
    // missing/empty protected inputs must fail before changing queue mode.
    // The Compose renderer rejects the same absent required pins up front.
    #[test]
    fn phala_queue_required_identity_inputs_cannot_be_omitted() {
        let recipient = hex::encode(IdentityKeyPair::from_bytes(&[61; 32]).unwrap().public_key_bytes());
        let recipient_json = format!("[\"{recipient}\"]");
        let source = hex::encode(IdentityKeyPair::from_bytes(&[62; 32]).unwrap().public_key_bytes());
        let source_json = format!("[\"{source}\"]");
        for recovery in ["false", "true"] {
            for missing_recipient in [None, Some(""), Some("[]")] {
                let mut queue = ReverseOnionQueueConfig::default();
                let sources = if recovery == "true" { None } else { Some(source_json.as_str()) };
                assert!(queue.apply_phala_environment(
                    Some("true"), Some(recovery), missing_recipient, sources,
                    None, None, None, true, false,
                ).is_err());
                assert!(!queue.enabled);
                assert!(!queue.recovery_only);
                assert!(queue.recipient_node_ids.is_empty());
                assert!(queue.source_node_ids.is_empty());
            }
        }
        for missing_source in [None, Some(""), Some("[]")] {
            let mut queue = ReverseOnionQueueConfig::default();
            assert!(queue.apply_phala_environment(
                Some("true"), Some("false"), Some(&recipient_json), missing_source,
                None, None, None, true, false,
            ).is_err());
            assert!(!queue.enabled);
            assert!(queue.recipient_node_ids.is_empty());
        }
        let mut recovery = ReverseOnionQueueConfig::default();
        assert!(recovery.apply_phala_environment(
            Some("true"), Some("true"), Some(&recipient_json), None,
            None, None, None, true, false,
        ).is_ok());
        assert!(recovery.recovery_only);
        assert!(!recovery.permits_new_claims());
        assert!(recovery.source_node_ids.is_empty());
    }

    // [PHALA-QUEUE-RECOVERY-ENV 2026-10-06 by Codex]
    #[test]
    fn phala_queue_recovery_override_preserves_live_admission_boundary() {
        let recipient = hex::encode(IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes());
        let recipient_json = format!("[\"{recipient}\"]");
        let source = hex::encode(IdentityKeyPair::from_bytes(&[42; 32]).unwrap().public_key_bytes());
        let source_json = format!("[\"{source}\"]");
        let mut queue = ReverseOnionQueueConfig::default();
        queue.recovery_only = true;

        assert!(queue.apply_phala_environment(
            Some("true"), None, Some(&recipient_json), None,
            None, None, None, true, false,
        ).is_ok());
        assert!(queue.recovery_only);
        assert!(!queue.permits_new_claims());
        assert!(queue.source_node_ids.is_empty());

        assert!(queue.apply_phala_environment(
            Some("true"), Some("false"), Some(&recipient_json), Some(&source_json),
            None, None, None, true, false,
        ).is_ok());
        assert!(!queue.recovery_only);
        assert!(queue.permits_new_claims());
        assert!(queue.apply_phala_environment(
            Some("true"), Some("true"), Some(&recipient_json), Some(&source_json),
            None, None, None, true, false,
        ).is_err());
        assert!(queue.apply_phala_environment(
            Some("true"), Some("yes"), Some(&recipient_json), None,
            None, None, None, true, false,
        ).is_err());
        assert!(queue.apply_phala_environment(
            Some("true"), Some("true"), Some(&recipient_json), None,
            None, Some("relay"), Some("authorization"), true, false,
        ).is_err());
        assert!(queue.apply_phala_environment(
            Some("false"), Some("true"), None, None,
            None, None, None, true, false,
        ).is_err());

        queue.recovery_only = true;
        assert!(queue.apply_phala_environment(
            Some("false"), None, None, None,
            None, None, None, true, false,
        ).is_ok());
        assert!(!queue.enabled);
        assert!(!queue.recovery_only);
    }

    // [PHALA-QUEUE-ENV-BOUNDS 2026-10-06 by Codex] Bound protected identity
    // JSON before deserializing environment-controlled arrays.
    #[test]
    fn phala_queue_identity_environment_inputs_are_bounded_before_parse() {
        let recipient = hex::encode(IdentityKeyPair::from_bytes(&[51; 32]).unwrap().public_key_bytes());
        let recipient_json = format!("[\"{recipient}\"]");
        let source_json = format!("\"{}\"", "a".repeat(MAX_PHALA_QUEUE_ID_JSON_BYTES));
        let mut queue = ReverseOnionQueueConfig::default();
        assert!(queue.apply_phala_environment(
            Some("true"), Some("true"), Some(&recipient_json), None,
            None, None, None, true, false,
        ).is_ok());

        let mut queue = ReverseOnionQueueConfig::default();
        assert!(queue.apply_phala_environment(
            Some("true"), Some("true"), Some(&recipient_json), Some(&source_json),
            None, None, None, true, false,
        ).is_err());
    }

    #[test]
    fn recipient_accepts_only_canonical_public_https_hostname_endpoints() {
        let mut config = recipient_config();
        assert!(config.validate().is_ok());
        // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] The configured
        // recipient pin accepts Phala DNS origins but not malformed labels.
        for endpoint in [
            "https://relay.example.net:443",
            "https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network",
            "https://[2606:4700:4700::1111]",
        ] {
            config.recipient.relay_endpoint = endpoint.to_owned();
            assert!(config.validate().is_ok(), "{endpoint}");
        }
        for endpoint in [
            "http://relay.example.net:8422", "https://localhost", "https://relay.internal",
            "http://8.8.8.8:8422", "http://127.0.0.1", "http://10.0.0.1",
            "https://user@8.8.8.8", "https://8.8.8.8/api", "https://8.8.8.8/?token=x",
            "https://8.8.8.8/#fragment",
            "https://relay..example.net", "https://-relay.example.net",
            "https://relay-.example.net", "https://relay_name.example.net",
            "https://[2001:db8::1]", "https://[ff0e::1]",
            " https://relay.example.net", "https://relay.example.net ",
        ] {
            config.recipient.relay_endpoint = endpoint.to_owned();
            assert!(config.validate().is_err(), "{endpoint}");
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

    // [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Live source
    // startup needs operator-pinned identities/origin, not a startup grant.
    #[test]
    fn source_cannot_enable_with_implicit_authority_or_storage() {
        let mut config = ReverseOnionConfig::default();
        config.source.enabled = true;
        assert!(config.validate().is_err());
        config.source.state_db_path = "/Volumes/disk/reverse-onion-test/source.db".to_owned();
        assert!(config.validate().is_err());
        config.source.relay_descriptor_b64 = "not canonical base64".to_owned();
        assert!(config.source.optional_authority_material().is_err());
    }

    // [PHALA-SOURCE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Authored, not
    // executed: source role can start inert with fixed topology pins while
    // current signed route authority is still arriving through discovery.
    #[test]
    fn live_source_accepts_identity_and_origin_pins_without_startup_grant() {
        let mut config = ReverseOnionConfig::default();
        config.source.enabled = true;
        config.source.state_db_path = "/Volumes/disk/reverse-onion-test/source.db".to_owned();
        config.source.relay_node_id = hex::encode(
            IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        );
        config.source.recipient_node_id = hex::encode(
            IdentityKeyPair::from_bytes(&[42; 32]).unwrap().public_key_bytes(),
        );
        config.source.relay_endpoint = "https://relay.example.net".to_owned();
        assert!(config.validate().is_ok());
        assert!(config.source.optional_authority_material().unwrap().is_none());
    }

    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Evidence polling
    // cannot be configured into an unbounded request or a busy loop.
    #[test]
    fn source_result_poll_window_is_bounded() {
        let mut config = ReverseOnionConfig::default();
        assert_eq!(config.source.result_wait_secs, 5);
        assert_eq!(config.source.result_poll_interval_ms, 250);
        config.source.enabled = true;
        config.source.state_db_path = "/Volumes/disk/reverse-onion-test/source.db".to_owned();
        config.source.result_wait_secs = 31;
        assert!(config.validate().is_err());
        config.source.result_wait_secs = 5;
        config.source.result_poll_interval_ms = 99;
        assert!(config.validate().is_err());
    }

    // [REVERSE-RECOVERY-BOOT 2026-10-05 by Codex] No new route authority
    // is inferred from the identity-only historical recovery configuration.
    #[test]
    fn recovery_queue_is_explicit_and_single_recipient() {
        let mut config = ReverseOnionConfig::default();
        config.queue.enabled = true;
        config.queue.recovery_only = true;
        config.queue.db_path = "/Volumes/disk/reverse-onion-test/queue.db".to_owned();
        assert!(config.validate().is_err());
        config.queue.recipient_node_ids.push(hex::encode(
            IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        ));
        assert!(config.validate().is_ok());
        assert!(!config.queue.signed_private_admission_configured());
    }

    // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] Authored, not run.
    #[test]
    fn relay_marker_retention_covers_the_full_poll_evidence_horizon() {
        let mut config = ReverseOnionConfig::default();
        config.queue.enabled = true;
        config.queue.recovery_only = true;
        config.queue.db_path = "/Volumes/disk/reverse-onion-test/queue.db".to_owned();
        config.queue.recipient_node_ids.push(hex::encode(
            IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        ));
        config.queue.recovery_retention_secs =
            REVERSE_ONION_NO_WORK_MARKER_MIN_RETENTION_SECS - 1;
        assert!(config.validate().is_err());
        config.queue.recovery_retention_secs = REVERSE_ONION_NO_WORK_MARKER_MIN_RETENTION_SECS;
        assert!(config.validate().is_ok());
    }

    // [REVERSE-ONION-ROUTE-CAP 2026-10-06 by Codex] Route freshness can
    // outlive one execution lease, but remains explicitly bounded.
    #[test]
    fn queue_route_window_is_independent_and_bounded() {
        let mut config = ReverseOnionConfig::default();
        assert_eq!(config.queue.route_max_secs, 600);
        config.queue.route_max_secs =
            MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_LIFETIME_SECS_V1 + 1;
        config.queue.enabled = true;
        config.queue.recovery_only = true;
        config.queue.db_path = "/Volumes/disk/reverse-onion-test/queue.db".to_owned();
        config.queue.recipient_node_ids.push(hex::encode(
            IdentityKeyPair::from_bytes(&[41; 32]).unwrap().public_key_bytes(),
        ));
        assert!(config.validate().is_err());
    }

    // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn only_live_reverse_roles_require_authority_gossip() {
        let mut config = ReverseOnionConfig::default();
        assert!(!config.requires_live_authority_gossip());
        assert!(!config.recipient.permits_new_claims());

        config.queue.enabled = true;
        assert!(config.requires_live_authority_gossip());
        config.queue.recovery_only = true;
        assert!(!config.requires_live_authority_gossip());
        config.queue.enabled = false;

        config.recipient.enabled = true;
        assert!(config.recipient.permits_new_claims());
        assert!(config.requires_live_authority_gossip());
        config.recipient.recovery_only = true;
        assert!(!config.recipient.permits_new_claims());
        assert!(!config.requires_live_authority_gossip());
        config.recipient.enabled = false;

        config.source.enabled = true;
        assert!(config.requires_live_authority_gossip());
        config.source.recovery_only = true;
        assert!(!config.requires_live_authority_gossip());
    }
}
