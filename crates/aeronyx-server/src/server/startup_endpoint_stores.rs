// ============================================
// File: crates/aeronyx-server/src/server/startup_endpoint_stores.rs
// ============================================
//! # Endpoint evidence and attestation stores
//!
//! Owns the default-off, blocking private-database admission of the
//! permissionless endpoint evidence store and attestation inbox.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use std::sync::Arc;

use aeronyx_core::crypto::IdentityKeyPair;

use crate::api::public_node_router::public_endpoint_flow_context;
use crate::config::DiscoveryConfig;
use crate::error::{Result, ServerError};
use crate::services::{
    DiscoveryEndpointAttestationInboxConfig, DiscoveryEndpointEvidenceStoreConfig,
    SqliteDiscoveryEndpointAttestationInbox, SqliteDiscoveryEndpointEvidenceStore,
};

// [PERMISSIONLESS-ENDPOINT-EVIDENCE 2026-09-24 by Codex] Keep blocking
// private-database admission out of the listener composition and make the
// default-off zero-side-effect boundary directly testable.
pub(super) async fn open_endpoint_evidence_store(
    discovery: &DiscoveryConfig,
    identity: &IdentityKeyPair,
) -> Result<Option<Arc<SqliteDiscoveryEndpointEvidenceStore>>> {
    if !discovery.permissionless_endpoint_evidence_enabled {
        return Ok(None);
    }
    let config = DiscoveryEndpointEvidenceStoreConfig {
        db_path: discovery
            .permissionless_endpoint_evidence_db_path
            .clone()
            .into(),
        max_entries: discovery.permissionless_endpoint_evidence_max_entries,
        retention_ttl_secs: discovery.permissionless_endpoint_evidence_ttl_secs,
        cleanup_batch_size: discovery.permissionless_endpoint_evidence_cleanup_batch,
    };
    let verifier_node_id = identity.public_key_bytes();
    let expected_context = public_endpoint_flow_context(verifier_node_id);
    let store = tokio::task::spawn_blocking(move || {
        SqliteDiscoveryEndpointEvidenceStore::open(config, verifier_node_id, expected_context)
    })
    .await
    .map_err(|_| ServerError::startup_failed("Endpoint evidence startup task failed"))?
    .map_err(|_| ServerError::startup_failed("Endpoint evidence store unavailable"))?;
    Ok(Some(Arc::new(store)))
}

// [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX-COMPOSITION 2026-09-24 by Codex]
// Open and audit the independent quarantine before any listener is bound.
// Disabled configuration returns before deriving a path or touching disk.
pub(super) async fn open_endpoint_attestation_inbox(
    discovery: &DiscoveryConfig,
) -> Result<Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>> {
    if !discovery.permissionless_endpoint_attestation_inbox_enabled {
        return Ok(None);
    }
    let config = DiscoveryEndpointAttestationInboxConfig {
        db_path: discovery
            .permissionless_endpoint_attestation_inbox_db_path
            .clone()
            .into(),
        max_entries: discovery.permissionless_endpoint_attestation_inbox_max_entries,
        max_logical_bytes: discovery.permissionless_endpoint_attestation_inbox_max_bytes,
        retention_ttl_secs: discovery.permissionless_endpoint_attestation_inbox_ttl_secs,
        cleanup_batch_size: discovery.permissionless_endpoint_attestation_inbox_cleanup_batch,
    };
    let inbox =
        tokio::task::spawn_blocking(move || SqliteDiscoveryEndpointAttestationInbox::open(config))
            .await
            .map_err(|_| ServerError::startup_failed("Endpoint attestation startup task failed"))?
            .map_err(|_| ServerError::startup_failed("Endpoint attestation inbox unavailable"))?;
    Ok(Some(Arc::new(inbox)))
}
