// ============================================
// File: crates/aeronyx-server/src/server/peer_cache_recovery.rs
// ============================================
// [SERVER-DECOMPOSITION-PHASE12A 2026-09-21 by Codex] Owns only the pure,
// backward-compatible peer-cache persistence outcomes and signed recovery-anchor
// domain. Filesystem, network, import mutation, and lifecycle order remain in
// the server composition root.

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use sha2::{Digest, Sha256};

use crate::error::{Result, ServerError};
use crate::services::peer_store::PeerStoreVerifiedClientDeliveryCacheEvidence;

// Published main still owns the backward-compatible cache document in the
// composition root; Phase12A extracts only recovery outcomes and anchors.
use super::{PeerStoreCacheDocument, VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION};

/// Signed rollback anchors are intentionally tiny aggregate-only documents.
pub(super) const VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES: usize = 4 * 1024;
pub(super) const VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT: &str =
    "peer_store_verified_client_delivery_anchor.v1";
pub(super) const VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT: &str =
    "peer_store_recovery_anchor.v2";
pub(super) const VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT: &str = "peer_store_recovery_anchor.v3";

/// Aggregate result of one durable peer-cache write.
///
/// [THREE-HOP-SIGNED-RECOVERY 2026-08-02 by Codex] A named report replaces the
/// positional tuple so adding independently signed recovery sections cannot
/// silently swap counters at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PeerStoreCachePersistReport {
    pub(super) two_hop_events: usize,
    pub(super) two_hop_stability_ready: bool,
    pub(super) three_hop_events: usize,
    pub(super) three_hop_stability_ready: bool,
    pub(super) route_domain_certificates: usize,
    pub(super) client_deliveries: u64,
    pub(super) client_delivery_generation: u64,
}

/// Durable state reached by one peer-cache persistence attempt.
///
/// [PEER-CACHE-RETRY-STATE 2026-08-12 by Codex] `Deferred` is not an error: it
/// means the generation intentionally remained pending because its configured
/// external delivery witness was not yet protected. Callers must not log it as
/// persisted or clear retry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PeerStoreCachePersistOutcome {
    Persisted,
    Deferred,
}

/// Independent signed local high-water mark for aggregate delivery evidence.
///
/// The anchor intentionally repeats only the cache generation, cache time,
/// aggregate count, and latest verification time. It contains no route,
/// endpoint, peer pair, sender, receiver, message id, payload commitment, or
/// ciphertext. Because it is host-local, it detects single-file rollback but
/// cannot detect a whole-host snapshot rollback that also replaces the anchor.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct PeerStoreVerifiedClientDeliveryAnchor {
    pub(super) contract_version: String,
    pub(super) cache_generation: u64,
    pub(super) cache_generated_at: u64,
    pub(super) evidence: Option<PeerStoreVerifiedClientDeliveryCacheEvidence>,
    /// Opaque digest of signed routeability plus active quarantine state.
    #[serde(default)]
    pub(super) route_state_digest: Option<String>,
    /// Opaque digest of the independently signed two-hop proof section.
    #[serde(default)]
    pub(super) two_hop_path_proof_digest: Option<String>,
    /// Opaque digest of the independently signed three-hop proof section.
    #[serde(default)]
    pub(super) three_hop_path_proof_digest: Option<String>,
    pub(super) signer_node_id: String,
    pub(super) signature_ed25519: String,
}

impl PeerStoreVerifiedClientDeliveryAnchor {
    pub(super) fn new(
        document: &PeerStoreCacheDocument,
        identity: &IdentityKeyPair,
    ) -> Result<Self> {
        let mut anchor = Self {
            contract_version: VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT.to_string(),
            cache_generation: document.verified_client_delivery_generation,
            cache_generated_at: document.descriptor_snapshot.generated_at,
            evidence: document.verified_client_delivery_evidence,
            route_state_digest: Some(
                document
                    .route_state_digest()
                    .map_err(ServerError::internal)?,
            ),
            two_hop_path_proof_digest: Some(
                document
                    .two_hop_path_proof_digest()
                    .map_err(ServerError::internal)?,
            ),
            three_hop_path_proof_digest: Some(
                document
                    .three_hop_path_proof_digest()
                    .map_err(ServerError::internal)?,
            ),
            signer_node_id: hex::encode(identity.public_key_bytes()),
            signature_ed25519: String::new(),
        };
        if anchor.cache_generation == 0 {
            return Err(ServerError::internal(
                "verified client delivery anchor generation must be positive",
            ));
        }
        anchor.signature_ed25519 =
            hex::encode(identity.sign(&anchor.signing_bytes().map_err(ServerError::internal)?));
        Ok(anchor)
    }

    pub(super) fn from_json_bytes(bytes: &[u8]) -> std::result::Result<Self, String> {
        if bytes.len() > VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES {
            return Err(format!(
                "verified client delivery anchor exceeds {} bytes",
                VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES
            ));
        }
        let anchor: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("verified client delivery anchor json: {error}"))?;
        if !matches!(
            anchor.contract_version.as_str(),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT
                | VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT
                | VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT
        ) {
            return Err("verified client delivery anchor contract unsupported".to_string());
        }
        if anchor.cache_generation == 0 {
            return Err("verified client delivery anchor generation invalid".to_string());
        }
        if matches!(
            anchor.contract_version.as_str(),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT
                | VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT
        ) {
            for digest in [
                anchor.two_hop_path_proof_digest.as_deref(),
                anchor.three_hop_path_proof_digest.as_deref(),
            ] {
                let digest =
                    digest.ok_or_else(|| "recovery anchor proof digest missing".to_string())?;
                let mut decoded = [0u8; 32];
                hex::decode_to_slice(digest, &mut decoded)
                    .map_err(|_| "recovery anchor proof digest encoding invalid".to_string())?;
            }
        }
        match anchor.contract_version.as_str() {
            VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT => {
                if anchor.route_state_digest.is_some()
                    || anchor.two_hop_path_proof_digest.is_some()
                    || anchor.three_hop_path_proof_digest.is_some()
                {
                    return Err("legacy recovery anchor contains unsupported digest".to_string());
                }
            }
            VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT => {
                if anchor.route_state_digest.is_some() {
                    return Err("v2 recovery anchor contains route-state digest".to_string());
                }
            }
            VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT => {
                let digest = anchor
                    .route_state_digest
                    .as_deref()
                    .ok_or_else(|| "recovery anchor route-state digest missing".to_string())?;
                let mut decoded = [0u8; 32];
                hex::decode_to_slice(digest, &mut decoded).map_err(|_| {
                    "recovery anchor route-state digest encoding invalid".to_string()
                })?;
            }
            _ => unreachable!("anchor contract checked above"),
        }
        Ok(anchor)
    }

    pub(super) fn signing_bytes(&self) -> std::result::Result<Vec<u8>, String> {
        match self.contract_version.as_str() {
            VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT => bincode::serialize(&(
                "aeronyx-peer-cache-verified-client-delivery-anchor-v1",
                self.contract_version.as_str(),
                self.cache_generation,
                self.cache_generated_at,
                self.evidence,
            )),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT => bincode::serialize(&(
                "aeronyx-peer-cache-recovery-anchor-v2",
                self.contract_version.as_str(),
                self.cache_generation,
                self.cache_generated_at,
                self.evidence,
                self.two_hop_path_proof_digest.as_deref(),
                self.three_hop_path_proof_digest.as_deref(),
            )),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT => bincode::serialize(&(
                "aeronyx-peer-cache-recovery-anchor-v3",
                self.contract_version.as_str(),
                self.cache_generation,
                self.cache_generated_at,
                self.evidence,
                self.route_state_digest.as_deref(),
                self.two_hop_path_proof_digest.as_deref(),
                self.three_hop_path_proof_digest.as_deref(),
            )),
            _ => return Err("recovery anchor contract unsupported".to_string()),
        }
        .map_err(|error| format!("verified client delivery anchor signing bytes: {error}"))
    }

    pub(super) fn verify(&self, identity: &IdentityKeyPair) -> std::result::Result<(), String> {
        let expected_signer = hex::encode(identity.public_key_bytes());
        if self.signer_node_id != expected_signer {
            return Err("verified client delivery anchor signer mismatch".to_string());
        }
        let mut signature = [0u8; 64];
        hex::decode_to_slice(&self.signature_ed25519, &mut signature).map_err(|_| {
            "verified client delivery anchor signature encoding invalid".to_string()
        })?;
        let public_key = IdentityPublicKey::from_bytes(&identity.public_key_bytes())
            .map_err(|_| "verified client delivery anchor signer key invalid".to_string())?;
        public_key
            .verify(&self.signing_bytes()?, &signature)
            .map_err(|_| "verified client delivery anchor signature invalid".to_string())
    }

    pub(super) fn matches_document(&self, document: &PeerStoreCacheDocument) -> bool {
        self.cache_generation == document.verified_client_delivery_generation
            && self.cache_generated_at == document.descriptor_snapshot.generated_at
            && self.evidence == document.verified_client_delivery_evidence
    }

    pub(super) fn matches_two_hop_path_proof_section(
        &self,
        document: &PeerStoreCacheDocument,
    ) -> bool {
        matches!(
            self.contract_version.as_str(),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT
                | VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT
        ) && self.cache_generation == document.verified_client_delivery_generation
            && self.cache_generated_at == document.descriptor_snapshot.generated_at
            && document.two_hop_path_proof_digest().is_ok_and(|digest| {
                self.two_hop_path_proof_digest.as_deref() == Some(digest.as_str())
            })
    }

    pub(super) fn matches_three_hop_path_proof_section(
        &self,
        document: &PeerStoreCacheDocument,
    ) -> bool {
        matches!(
            self.contract_version.as_str(),
            VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT
                | VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT
        ) && self.cache_generation == document.verified_client_delivery_generation
            && self.cache_generated_at == document.descriptor_snapshot.generated_at
            && document.three_hop_path_proof_digest().is_ok_and(|digest| {
                self.three_hop_path_proof_digest.as_deref() == Some(digest.as_str())
            })
    }

    pub(super) fn matches_route_state_section(&self, document: &PeerStoreCacheDocument) -> bool {
        self.contract_version == VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT
            && self.cache_generation == document.verified_client_delivery_generation
            && self.cache_generated_at == document.descriptor_snapshot.generated_at
            && document
                .route_state_digest()
                .is_ok_and(|digest| self.route_state_digest.as_deref() == Some(digest.as_str()))
    }

    /// Returns a domain-separated opaque digest of the exact signed anchor.
    ///
    /// External witnesses receive this digest, never the embedded aggregate
    /// delivery count or verification time. Binding the signer and signature
    /// prevents two differently signed anchor documents from sharing witness
    /// state even if their canonical fields were otherwise equal.
    pub(super) fn witness_digest(&self) -> std::result::Result<[u8; 32], String> {
        let mut signer = [0u8; 32];
        hex::decode_to_slice(&self.signer_node_id, &mut signer)
            .map_err(|_| "verified client delivery anchor signer encoding invalid".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(&self.signature_ed25519, &mut signature).map_err(|_| {
            "verified client delivery anchor signature encoding invalid".to_string()
        })?;
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-VerifiedDeliveryAnchorWitnessDigest-v1");
        hasher.update(self.signing_bytes()?);
        hasher.update(signer);
        hasher.update(signature);
        Ok(hasher.finalize().into())
    }

    pub(super) fn to_json_pretty(&self) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| {
            ServerError::internal(format!("verified client delivery anchor json: {error}"))
        })?;
        if bytes.len() > VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES {
            return Err(ServerError::internal(format!(
                "verified client delivery anchor exceeds {} bytes",
                VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES
            )));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone)]
pub(super) enum PeerStoreVerifiedClientDeliveryAnchorState {
    /// Compatibility path for direct parser callers and non-cache sources.
    NotChecked,
    Missing,
    Invalid,
    Verified(PeerStoreVerifiedClientDeliveryAnchor),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PeerStoreVerifiedClientDeliveryExternalWitnessDecision {
    /// No external witness policy is configured; local anchor behavior remains unchanged.
    Disabled,
    /// No local anchor exists yet. Only a genuinely fresh cache may bootstrap.
    Missing,
    /// The exact local anchor reached the configured accepted-response threshold.
    Protected,
    /// The local anchor was unavailable, adverse, or below threshold.
    Unprotected(&'static str),
}

impl PeerStoreVerifiedClientDeliveryAnchorState {
    pub(super) fn protection_for(&self, document: &PeerStoreCacheDocument) -> &'static str {
        if document.verified_client_delivery_schema_version
            == VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION
        {
            return match self {
                Self::NotChecked | Self::Missing => "legacy_unanchored",
                Self::Invalid => "anchor_invalid",
                Self::Verified(_) => "rollback_detected",
            };
        }
        if document.verified_client_delivery_schema_version == 0 {
            return "not_checked";
        }
        match self {
            Self::NotChecked => "not_checked",
            Self::Missing => "anchor_missing",
            Self::Invalid => "anchor_invalid",
            Self::Verified(anchor) => {
                if document.verified_client_delivery_generation < anchor.cache_generation {
                    "rollback_detected"
                } else if document.verified_client_delivery_generation > anchor.cache_generation {
                    "cache_ahead"
                } else if anchor.matches_document(document) {
                    "anchored"
                } else {
                    "anchor_conflict"
                }
            }
        }
    }

    pub(super) fn two_hop_path_proof_protection_for(
        &self,
        document: &PeerStoreCacheDocument,
    ) -> &'static str {
        self.path_proof_protection_for(document, |anchor, document| {
            anchor.matches_two_hop_path_proof_section(document)
        })
    }

    pub(super) fn three_hop_path_proof_protection_for(
        &self,
        document: &PeerStoreCacheDocument,
    ) -> &'static str {
        self.path_proof_protection_for(document, |anchor, document| {
            anchor.matches_three_hop_path_proof_section(document)
        })
    }

    /// Evaluates the signed routeability/quarantine snapshot against the
    /// monotonic local generation without exposing its digest.
    pub(super) fn route_state_protection_for(
        &self,
        document: &PeerStoreCacheDocument,
    ) -> &'static str {
        match self {
            Self::NotChecked => "not_checked",
            Self::Missing => "anchor_missing",
            Self::Invalid => "anchor_invalid",
            Self::Verified(anchor) => {
                if document.verified_client_delivery_generation < anchor.cache_generation {
                    "rollback_detected"
                } else if document.verified_client_delivery_generation > anchor.cache_generation {
                    "cache_ahead"
                } else if anchor.contract_version != VERIFIED_CLIENT_DELIVERY_ANCHOR_CONTRACT {
                    "legacy_unanchored"
                } else if anchor.matches_route_state_section(document) {
                    "anchored"
                } else {
                    "anchor_conflict"
                }
            }
        }
    }

    /// Evaluates rollback protection independently from aggregate delivery
    /// evidence so one proof-anchor mismatch cannot discard the other proof
    /// section or valid descriptors.
    pub(super) fn path_proof_protection_for(
        &self,
        document: &PeerStoreCacheDocument,
        matches_section: impl FnOnce(
            &PeerStoreVerifiedClientDeliveryAnchor,
            &PeerStoreCacheDocument,
        ) -> bool,
    ) -> &'static str {
        match self {
            Self::NotChecked => "not_checked",
            Self::Missing => "anchor_missing",
            Self::Invalid => "anchor_invalid",
            Self::Verified(anchor) => {
                if document.verified_client_delivery_generation < anchor.cache_generation {
                    "rollback_detected"
                } else if document.verified_client_delivery_generation > anchor.cache_generation {
                    "cache_ahead"
                } else if anchor.contract_version == VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT
                {
                    "legacy_unanchored"
                } else if matches_section(anchor, document) {
                    "anchored"
                } else {
                    "anchor_conflict"
                }
            }
        }
    }
}
