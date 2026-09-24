// ============================================
// File: crates/aeronyx-server/src/server/peer_cache_runtime.rs
// ============================================
// [PEER-CACHE-RUNTIME 2026-09-25 by Codex] Keep the signed local cache
// document, bounded restore/backup, external-witness fencing, and scheduled
// durable publication together. Parent Server retains startup order and the
// generic bounded reader; wire and persisted JSON remain unchanged.
use super::*;

/// Coalesce verified-delivery bursts before atomically refreshing peer cache.
const CLIENT_DELIVERY_CACHE_FLUSH_DEBOUNCE_MILLIS: u64 = 250;
/// First retry delay after a failed or witness-deferred peer-cache write.
const PEER_CACHE_PERSIST_RETRY_BASE_MILLIS: u64 = 1_000;
/// Prevent persistent disk or witness failure from creating a background loop.
const PEER_CACHE_PERSIST_RETRY_MAX_MILLIS: u64 = 60_000;
/// Backward-compatible local peer-cache document.
///
/// `descriptor_snapshot` is flattened so legacy readers still see the exact
/// `NodeBootstrapSnapshot` top-level shape and ignore the additive routeability
/// and path-proof fields. Each recovery section has an independent signature
/// domain, allowing one invalid section to fail closed without discarding the
/// separately signed descriptors or other recovery evidence. This document is
/// local-only and must never be served by gossip.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(super) struct PeerStoreCacheDocument {
    #[serde(flatten)]
    pub(super) descriptor_snapshot: NodeBootstrapSnapshot,
    #[serde(default)]
    pub(super) routeability_evidence_schema_version: u16,
    #[serde(default)]
    pub(super) routeability_evidence: Vec<PeerStoreRouteabilityCacheEvidence>,
    #[serde(default)]
    pub(super) route_quarantine_schema_version: u16,
    #[serde(default)]
    pub(super) route_quarantine_evidence: Vec<PeerStoreRouteQuarantineCacheEvidence>,
    #[serde(default)]
    pub(super) routeability_evidence_signer_node_id: Option<String>,
    #[serde(default)]
    pub(super) routeability_evidence_signature_ed25519: Option<String>,
    #[serde(default)]
    pub(super) two_hop_path_proof_schema_version: u16,
    #[serde(default)]
    pub(super) two_hop_path_proof_events: Vec<PeerStoreTwoHopPathProofEvent>,
    #[serde(default)]
    pub(super) two_hop_path_proof_signer_node_id: Option<String>,
    #[serde(default)]
    pub(super) two_hop_path_proof_signature_ed25519: Option<String>,
    #[serde(default)]
    pub(super) three_hop_path_proof_schema_version: u16,
    #[serde(default)]
    pub(super) three_hop_path_proof_events: Vec<PeerStoreTwoHopPathProofEvent>,
    #[serde(default)]
    pub(super) three_hop_path_proof_signer_node_id: Option<String>,
    #[serde(default)]
    pub(super) three_hop_path_proof_signature_ed25519: Option<String>,
    #[serde(default)]
    pub(super) verified_client_delivery_schema_version: u16,
    #[serde(default)]
    pub(super) verified_client_delivery_generation: u64,
    #[serde(default)]
    pub(super) verified_client_delivery_evidence:
        Option<PeerStoreVerifiedClientDeliveryCacheEvidence>,
    #[serde(default)]
    pub(super) verified_client_delivery_signer_node_id: Option<String>,
    #[serde(default)]
    pub(super) verified_client_delivery_signature_ed25519: Option<String>,
    /// Portable certificates remain self-authenticating through their pinned
    /// attestor signatures; this local-only section adds no host authority.
    #[serde(default)]
    pub(super) route_domain_certificate_schema_version: u16,
    #[serde(default)]
    pub(super) route_domain_certificates: Vec<RouteDomainAttestationCertificateV1>,
}

impl PeerStoreCacheDocument {
    pub(super) fn new(
        descriptor_snapshot: NodeBootstrapSnapshot,
        routeability_evidence: Vec<PeerStoreRouteabilityCacheEvidence>,
        route_quarantine_evidence: Vec<PeerStoreRouteQuarantineCacheEvidence>,
        two_hop_path_proof_events: Vec<PeerStoreTwoHopPathProofEvent>,
        three_hop_path_proof_events: Vec<PeerStoreTwoHopPathProofEvent>,
        route_domain_certificates: Vec<RouteDomainAttestationCertificateV1>,
        verified_client_delivery_generation: u64,
        verified_client_delivery_evidence: Option<PeerStoreVerifiedClientDeliveryCacheEvidence>,
        identity: &IdentityKeyPair,
    ) -> Result<Self> {
        let mut document = Self {
            descriptor_snapshot,
            routeability_evidence_schema_version: ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION,
            routeability_evidence,
            route_quarantine_schema_version: ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION,
            route_quarantine_evidence,
            routeability_evidence_signer_node_id: None,
            routeability_evidence_signature_ed25519: None,
            two_hop_path_proof_schema_version: TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
            two_hop_path_proof_events,
            two_hop_path_proof_signer_node_id: None,
            two_hop_path_proof_signature_ed25519: None,
            three_hop_path_proof_schema_version: THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
            three_hop_path_proof_events,
            three_hop_path_proof_signer_node_id: None,
            three_hop_path_proof_signature_ed25519: None,
            verified_client_delivery_schema_version: VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
            verified_client_delivery_generation,
            verified_client_delivery_evidence,
            verified_client_delivery_signer_node_id: None,
            verified_client_delivery_signature_ed25519: None,
            route_domain_certificate_schema_version: ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION,
            route_domain_certificates,
        };
        let signing_bytes = document
            .routeability_evidence_signing_bytes()
            .map_err(ServerError::internal)?;
        document.routeability_evidence_signer_node_id =
            Some(hex::encode(identity.public_key_bytes()));
        document.routeability_evidence_signature_ed25519 =
            Some(hex::encode(identity.sign(&signing_bytes)));
        let proof_signing_bytes = document
            .two_hop_path_proof_signing_bytes()
            .map_err(ServerError::internal)?;
        document.two_hop_path_proof_signer_node_id = Some(hex::encode(identity.public_key_bytes()));
        document.two_hop_path_proof_signature_ed25519 =
            Some(hex::encode(identity.sign(&proof_signing_bytes)));
        // [THREE-HOP-SIGNED-RECOVERY 2026-08-02 by Codex] Three-hop aggregate
        // evidence uses an independent signing domain so tampering cannot
        // invalidate descriptors, routeability, or the two-hop admission cache.
        let three_hop_signing_bytes = document
            .three_hop_path_proof_signing_bytes()
            .map_err(ServerError::internal)?;
        document.three_hop_path_proof_signer_node_id =
            Some(hex::encode(identity.public_key_bytes()));
        document.three_hop_path_proof_signature_ed25519 =
            Some(hex::encode(identity.sign(&three_hop_signing_bytes)));
        let client_delivery_signing_bytes = document
            .verified_client_delivery_signing_bytes()
            .map_err(ServerError::internal)?;
        document.verified_client_delivery_signer_node_id =
            Some(hex::encode(identity.public_key_bytes()));
        document.verified_client_delivery_signature_ed25519 =
            Some(hex::encode(identity.sign(&client_delivery_signing_bytes)));
        Ok(document)
    }

    pub(super) fn from_json_bytes(bytes: &[u8]) -> std::result::Result<Self, String> {
        if bytes.len() > DISCOVERY_SNAPSHOT_MAX_BYTES {
            return Err(format!(
                "peer cache exceeds {} bytes",
                DISCOVERY_SNAPSHOT_MAX_BYTES
            ));
        }
        let document: Self =
            serde_json::from_slice(bytes).map_err(|error| format!("peer cache json: {error}"))?;
        document
            .descriptor_snapshot
            .validate_schema()
            .map_err(|error| format!("peer cache descriptor snapshot: {error}"))?;
        let evidence_version_valid = (document.routeability_evidence.is_empty()
            && document.routeability_evidence_schema_version == 0)
            || matches!(
                document.routeability_evidence_schema_version,
                ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION
                    | ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION
            );
        if !evidence_version_valid {
            return Err(format!(
                "unsupported routeability evidence schema version: {}",
                document.routeability_evidence_schema_version
            ));
        }
        let quarantine_version_valid = match document.routeability_evidence_schema_version {
            ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION => {
                document.route_quarantine_schema_version == ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION
            }
            _ => {
                document.route_quarantine_schema_version == 0
                    && document.route_quarantine_evidence.is_empty()
            }
        };
        if !quarantine_version_valid {
            return Err(format!(
                "unsupported route quarantine cache schema version: {}",
                document.route_quarantine_schema_version
            ));
        }
        // [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] Keep the
        // additive section backward compatible while rejecting count abuse
        // before any cryptographic work or PeerStore allocation is attempted.
        let route_domain_certificate_version_valid =
            (document.route_domain_certificates.is_empty()
                && document.route_domain_certificate_schema_version == 0)
                || document.route_domain_certificate_schema_version
                    == ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION;
        if !route_domain_certificate_version_valid {
            return Err(format!(
                "unsupported route-domain certificate schema version: {}",
                document.route_domain_certificate_schema_version
            ));
        }
        if document.route_domain_certificates.len() > ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES {
            return Err(format!(
                "route-domain certificate cache exceeds {} entries",
                ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES
            ));
        }
        let proof_version_valid = (document.two_hop_path_proof_events.is_empty()
            && document.two_hop_path_proof_schema_version == 0)
            || document.two_hop_path_proof_schema_version
                == TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION;
        if !proof_version_valid {
            return Err(format!(
                "unsupported two-hop proof schema version: {}",
                document.two_hop_path_proof_schema_version
            ));
        }
        let three_hop_proof_version_valid = (document.three_hop_path_proof_events.is_empty()
            && document.three_hop_path_proof_schema_version == 0)
            || document.three_hop_path_proof_schema_version
                == THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION;
        if !three_hop_proof_version_valid {
            return Err(format!(
                "unsupported three-hop proof schema version: {}",
                document.three_hop_path_proof_schema_version
            ));
        }
        let client_delivery_version_valid = match document.verified_client_delivery_schema_version {
            0 => {
                document.verified_client_delivery_evidence.is_none()
                    && document.verified_client_delivery_generation == 0
            }
            VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION => {
                document.verified_client_delivery_generation == 0
            }
            VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION => {
                document.verified_client_delivery_generation > 0
            }
            _ => false,
        };
        if !client_delivery_version_valid {
            return Err(format!(
                "unsupported verified client delivery schema version: {}",
                document.verified_client_delivery_schema_version
            ));
        }
        Ok(document)
    }

    pub(super) fn verify_routeability_evidence_signature(
        &self,
        identity: &IdentityKeyPair,
    ) -> std::result::Result<(), String> {
        if self.routeability_evidence.is_empty() && self.routeability_evidence_schema_version == 0 {
            return Ok(());
        }
        let expected_signer = hex::encode(identity.public_key_bytes());
        if self.routeability_evidence_signer_node_id.as_deref() != Some(&expected_signer) {
            return Err("routeability evidence signer mismatch".to_string());
        }
        let signature_hex = self
            .routeability_evidence_signature_ed25519
            .as_deref()
            .ok_or_else(|| "routeability evidence signature missing".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(signature_hex, &mut signature)
            .map_err(|_| "routeability evidence signature encoding invalid".to_string())?;
        let public_key = IdentityPublicKey::from_bytes(&identity.public_key_bytes())
            .map_err(|_| "routeability evidence signer key invalid".to_string())?;
        let signing_bytes = self.routeability_evidence_signing_bytes()?;
        public_key
            .verify(&signing_bytes, &signature)
            .map_err(|_| "routeability evidence signature invalid".to_string())
    }

    pub(super) fn routeability_evidence_signing_bytes(
        &self,
    ) -> std::result::Result<Vec<u8>, String> {
        let encoded = match self.routeability_evidence_schema_version {
            ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION => bincode::serialize(&(
                "aeronyx-peer-cache-routeability-v1",
                self.descriptor_snapshot.generated_at,
                self.routeability_evidence_schema_version,
                &self.routeability_evidence,
            )),
            ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION => bincode::serialize(&(
                "aeronyx-peer-cache-routeability-v2",
                self.descriptor_snapshot.generated_at,
                self.routeability_evidence_schema_version,
                &self.routeability_evidence,
                self.route_quarantine_schema_version,
                &self.route_quarantine_evidence,
            )),
            version => {
                return Err(format!(
                    "unsupported routeability evidence schema: {version}"
                ))
            }
        };
        encoded.map_err(|error| format!("routeability evidence signing bytes: {error}"))
    }

    /// Returns an opaque digest of the exact independently signed route-state
    /// section, including active quarantine in cache schema v2.
    ///
    /// [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] The local monotonic
    /// anchor commits to this digest, never to decoded endpoints or failure
    /// details. Binding signer and signature prevents an unsigned replacement
    /// from sharing the same anchor state even if its payload bytes match.
    pub(super) fn route_state_digest(&self) -> std::result::Result<String, String> {
        let signer_hex = self
            .routeability_evidence_signer_node_id
            .as_deref()
            .ok_or_else(|| "route state signer missing".to_string())?;
        let signature_hex = self
            .routeability_evidence_signature_ed25519
            .as_deref()
            .ok_or_else(|| "route state signature missing".to_string())?;
        let mut signer = [0u8; 32];
        hex::decode_to_slice(signer_hex, &mut signer)
            .map_err(|_| "route state signer encoding invalid".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(signature_hex, &mut signature)
            .map_err(|_| "route state signature encoding invalid".to_string())?;
        let signing_bytes = self.routeability_evidence_signing_bytes()?;

        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-PeerCache-RouteStateDigest-v1");
        let signing_bytes_len = u64::try_from(signing_bytes.len())
            .map_err(|_| "route state signing bytes length invalid".to_string())?;
        hasher.update(signing_bytes_len.to_be_bytes());
        hasher.update(signing_bytes);
        hasher.update(signer);
        hasher.update(signature);
        Ok(hex::encode(hasher.finalize()))
    }

    pub(super) fn verify_two_hop_path_proof_signature(
        &self,
        identity: &IdentityKeyPair,
    ) -> std::result::Result<(), String> {
        if self.two_hop_path_proof_events.is_empty() && self.two_hop_path_proof_schema_version == 0
        {
            return Ok(());
        }
        let expected_signer = hex::encode(identity.public_key_bytes());
        if self.two_hop_path_proof_signer_node_id.as_deref() != Some(&expected_signer) {
            return Err("two-hop proof signer mismatch".to_string());
        }
        let signature_hex = self
            .two_hop_path_proof_signature_ed25519
            .as_deref()
            .ok_or_else(|| "two-hop proof signature missing".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(signature_hex, &mut signature)
            .map_err(|_| "two-hop proof signature encoding invalid".to_string())?;
        let public_key = IdentityPublicKey::from_bytes(&identity.public_key_bytes())
            .map_err(|_| "two-hop proof signer key invalid".to_string())?;
        let signing_bytes = self.two_hop_path_proof_signing_bytes()?;
        public_key
            .verify(&signing_bytes, &signature)
            .map_err(|_| "two-hop proof signature invalid".to_string())
    }

    pub(super) fn two_hop_path_proof_signing_bytes(&self) -> std::result::Result<Vec<u8>, String> {
        bincode::serialize(&(
            "aeronyx-peer-cache-two-hop-proof-v1",
            self.descriptor_snapshot.generated_at,
            self.two_hop_path_proof_schema_version,
            &self.two_hop_path_proof_events,
        ))
        .map_err(|error| format!("two-hop proof signing bytes: {error}"))
    }

    pub(super) fn two_hop_path_proof_digest(&self) -> std::result::Result<String, String> {
        Ok(hex::encode(Sha256::digest(
            self.two_hop_path_proof_signing_bytes()?,
        )))
    }

    pub(super) fn verify_three_hop_path_proof_signature(
        &self,
        identity: &IdentityKeyPair,
    ) -> std::result::Result<(), String> {
        if self.three_hop_path_proof_events.is_empty()
            && self.three_hop_path_proof_schema_version == 0
        {
            return Ok(());
        }
        let expected_signer = hex::encode(identity.public_key_bytes());
        if self.three_hop_path_proof_signer_node_id.as_deref() != Some(&expected_signer) {
            return Err("three-hop proof signer mismatch".to_string());
        }
        let signature_hex = self
            .three_hop_path_proof_signature_ed25519
            .as_deref()
            .ok_or_else(|| "three-hop proof signature missing".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(signature_hex, &mut signature)
            .map_err(|_| "three-hop proof signature encoding invalid".to_string())?;
        let public_key = IdentityPublicKey::from_bytes(&identity.public_key_bytes())
            .map_err(|_| "three-hop proof signer key invalid".to_string())?;
        let signing_bytes = self.three_hop_path_proof_signing_bytes()?;
        public_key
            .verify(&signing_bytes, &signature)
            .map_err(|_| "three-hop proof signature invalid".to_string())
    }

    pub(super) fn three_hop_path_proof_signing_bytes(
        &self,
    ) -> std::result::Result<Vec<u8>, String> {
        bincode::serialize(&(
            "aeronyx-peer-cache-three-hop-proof-v1",
            self.descriptor_snapshot.generated_at,
            self.three_hop_path_proof_schema_version,
            &self.three_hop_path_proof_events,
        ))
        .map_err(|error| format!("three-hop proof signing bytes: {error}"))
    }

    pub(super) fn three_hop_path_proof_digest(&self) -> std::result::Result<String, String> {
        Ok(hex::encode(Sha256::digest(
            self.three_hop_path_proof_signing_bytes()?,
        )))
    }

    pub(super) fn verify_verified_client_delivery_signature(
        &self,
        identity: &IdentityKeyPair,
    ) -> std::result::Result<(), String> {
        if self.verified_client_delivery_evidence.is_none()
            && self.verified_client_delivery_schema_version == 0
        {
            return Ok(());
        }
        let expected_signer = hex::encode(identity.public_key_bytes());
        if self.verified_client_delivery_signer_node_id.as_deref() != Some(&expected_signer) {
            return Err("verified client delivery signer mismatch".to_string());
        }
        let signature_hex = self
            .verified_client_delivery_signature_ed25519
            .as_deref()
            .ok_or_else(|| "verified client delivery signature missing".to_string())?;
        let mut signature = [0u8; 64];
        hex::decode_to_slice(signature_hex, &mut signature)
            .map_err(|_| "verified client delivery signature encoding invalid".to_string())?;
        let public_key = IdentityPublicKey::from_bytes(&identity.public_key_bytes())
            .map_err(|_| "verified client delivery signer key invalid".to_string())?;
        let signing_bytes = self.verified_client_delivery_signing_bytes()?;
        public_key
            .verify(&signing_bytes, &signature)
            .map_err(|_| "verified client delivery signature invalid".to_string())
    }

    pub(super) fn verified_client_delivery_signing_bytes(
        &self,
    ) -> std::result::Result<Vec<u8>, String> {
        match self.verified_client_delivery_schema_version {
            VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION => bincode::serialize(&(
                "aeronyx-peer-cache-verified-client-delivery-v1",
                self.descriptor_snapshot.generated_at,
                self.verified_client_delivery_schema_version,
                self.verified_client_delivery_evidence,
            )),
            VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION => bincode::serialize(&(
                "aeronyx-peer-cache-verified-client-delivery-v2",
                self.descriptor_snapshot.generated_at,
                self.verified_client_delivery_schema_version,
                self.verified_client_delivery_generation,
                self.verified_client_delivery_evidence,
            )),
            version => {
                return Err(format!(
                    "verified client delivery signing schema unsupported: {version}"
                ));
            }
        }
        .map_err(|error| format!("verified client delivery signing bytes: {error}"))
    }

    pub(super) fn to_json_pretty(&self) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| ServerError::internal(format!("peer cache json: {error}")))?;
        if bytes.len() > DISCOVERY_SNAPSHOT_MAX_BYTES {
            return Err(ServerError::internal(format!(
                "peer cache exceeds {} bytes",
                DISCOVERY_SNAPSHOT_MAX_BYTES
            )));
        }
        Ok(bytes)
    }
}

impl Server {
    pub(super) async fn read_peer_cache_client_delivery_anchor(
        path: &str,
        identity: &IdentityKeyPair,
    ) -> PeerStoreVerifiedClientDeliveryAnchorState {
        let anchor_path = Self::peer_cache_client_delivery_anchor_path(path);
        match Self::read_bounded_file(&anchor_path, VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES).await
        {
            Ok(bytes) => match PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(&bytes) {
                Ok(anchor) if anchor.verify(identity).is_ok() => {
                    PeerStoreVerifiedClientDeliveryAnchorState::Verified(anchor)
                }
                Ok(_) | Err(_) => PeerStoreVerifiedClientDeliveryAnchorState::Invalid,
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                PeerStoreVerifiedClientDeliveryAnchorState::Missing
            }
            Err(_) => PeerStoreVerifiedClientDeliveryAnchorState::Invalid,
        }
    }

    pub(super) async fn load_peer_cache(&self, peer_store: &PeerStore, path: &str, now: u64) {
        let client_delivery_anchor =
            Self::read_peer_cache_client_delivery_anchor(path, &self.identity).await;
        match Self::read_bounded_file(Path::new(path), DISCOVERY_SNAPSHOT_MAX_BYTES).await {
            Ok(bytes) => {
                if !Self::import_bootstrap_snapshot_bytes_with_anchor(
                    peer_store,
                    "cache",
                    path,
                    &bytes,
                    now,
                    Some(&self.identity),
                    &client_delivery_anchor,
                ) {
                    self.load_peer_cache_backup(peer_store, path, now, &client_delivery_anchor)
                        .await;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                peer_store.record_bootstrap_source(now, "cache", "missing", "file_not_found");
                info!(
                    source = %path,
                    "[DISCOVERY] Peer cache not found; starting with bootstrap only"
                );
                self.load_peer_cache_backup(peer_store, path, now, &client_delivery_anchor)
                    .await;
            }
            Err(e) => {
                peer_store.record_bootstrap_source(
                    now,
                    "cache",
                    "failed",
                    Self::bounded_file_error_reason(&e),
                );
                warn!(
                    source = %path,
                    error = %e,
                    "[DISCOVERY] Failed to read peer cache"
                );
                self.load_peer_cache_backup(peer_store, path, now, &client_delivery_anchor)
                    .await;
            }
        }
    }

    pub(super) async fn load_peer_cache_backup(
        &self,
        peer_store: &PeerStore,
        path: &str,
        now: u64,
        client_delivery_anchor: &PeerStoreVerifiedClientDeliveryAnchorState,
    ) {
        let backup_path = Self::peer_cache_backup_path(path);
        let backup_source = backup_path.to_string_lossy().to_string();
        match Self::read_bounded_file(&backup_path, DISCOVERY_SNAPSHOT_MAX_BYTES).await {
            Ok(bytes) => {
                if Self::import_bootstrap_snapshot_bytes_with_anchor(
                    peer_store,
                    "cache_backup",
                    &backup_source,
                    &bytes,
                    now,
                    Some(&self.identity),
                    client_delivery_anchor,
                ) {
                    info!(
                        source = %backup_source,
                        "[DISCOVERY] Peer cache backup restored"
                    );
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                peer_store.record_bootstrap_source(
                    now,
                    "cache_backup",
                    "missing",
                    "file_not_found",
                );
            }
            Err(e) => {
                peer_store.record_bootstrap_source(
                    now,
                    "cache_backup",
                    "failed",
                    Self::bounded_file_error_reason(&e),
                );
                warn!(
                    source = %backup_source,
                    error = %e,
                    "[DISCOVERY] Failed to read peer cache backup"
                );
            }
        }
    }

    pub(super) fn peer_store_delivery_witness_round(
        round: VerifiedDeliveryAnchorWitnessRound,
    ) -> PeerStoreVerifiedDeliveryWitnessRound {
        PeerStoreVerifiedDeliveryWitnessRound {
            configured: round.configured as u64,
            attempted: round.attempted as u64,
            verified: round.verified as u64,
            advanced: round.advanced as u64,
            idempotent: round.idempotent as u64,
            stale: round.stale as u64,
            conflicts: round.conflicts as u64,
            gaps: round.gaps as u64,
            failed: round.failed as u64,
        }
    }

    /// Reconciles the exact local signed recovery anchor with pinned peers.
    ///
    /// This is intentionally separate from Memory Chain checkpoint witnesses.
    /// [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] The legacy wire name
    /// remains stable, while anchor v3 commits to route state and both proof
    /// sections in addition to aggregate delivery. Witness traffic still
    /// carries only generation plus opaque digest, never delivery counts,
    /// timestamps, routes, message ids, payloads, endpoints, or client data.
    pub(super) async fn reconcile_peer_cache_delivery_witnesses(
        identity: &IdentityKeyPair,
        peer_store: &PeerStore,
        discovery: &DiscoveryConfig,
        client: &reqwest::Client,
        path: &str,
        startup_gate: bool,
    ) -> PeerStoreVerifiedClientDeliveryExternalWitnessDecision {
        let witness_node_ids = discovery.verified_delivery_witness_node_id_bytes();
        let anchor_state = Self::read_peer_cache_client_delivery_anchor(path, identity).await;
        let generation = match &anchor_state {
            PeerStoreVerifiedClientDeliveryAnchorState::Verified(anchor) => anchor.cache_generation,
            _ => 0,
        };
        if witness_node_ids.is_empty() {
            peer_store.record_client_delivery_witness_round(
                unix_now_secs(),
                generation,
                false,
                discovery.verified_delivery_witness_min_verified,
                PeerStoreVerifiedDeliveryWitnessRound::default(),
            );
            return PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Disabled;
        }

        let evaluated_at = unix_now_secs();
        let anchor = match anchor_state {
            PeerStoreVerifiedClientDeliveryAnchorState::Verified(anchor) => anchor,
            PeerStoreVerifiedClientDeliveryAnchorState::Missing => {
                let round = PeerStoreVerifiedDeliveryWitnessRound {
                    configured: witness_node_ids.len() as u64,
                    ..PeerStoreVerifiedDeliveryWitnessRound::default()
                };
                peer_store.record_client_delivery_witness_round(
                    evaluated_at,
                    0,
                    discovery.verified_delivery_witness_required_for_restore,
                    discovery.verified_delivery_witness_min_verified,
                    round,
                );
                if startup_gate && discovery.verified_delivery_witness_required_for_restore {
                    peer_store.clear_restored_peer_cache_readiness_evidence(
                        evaluated_at,
                        "external_witness_unavailable",
                    );
                }
                return PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Missing;
            }
            PeerStoreVerifiedClientDeliveryAnchorState::Invalid
            | PeerStoreVerifiedClientDeliveryAnchorState::NotChecked => {
                let round = PeerStoreVerifiedDeliveryWitnessRound {
                    configured: witness_node_ids.len() as u64,
                    failed: witness_node_ids.len() as u64,
                    ..PeerStoreVerifiedDeliveryWitnessRound::default()
                };
                peer_store.record_client_delivery_witness_round(
                    evaluated_at,
                    0,
                    discovery.verified_delivery_witness_required_for_restore,
                    discovery.verified_delivery_witness_min_verified,
                    round,
                );
                if startup_gate {
                    peer_store.clear_restored_peer_cache_readiness_evidence(
                        evaluated_at,
                        "external_witness_invalid",
                    );
                }
                return PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected(
                    "unavailable",
                );
            }
        };

        // [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Cache and
        // anchor are independently replaced durable files. A crash can land
        // after the new signed cache rename but before its matching anchor
        // rename. Never ask witnesses to approve the older anchor and then
        // apply that decision to readiness restored from the newer cache.
        let recovered_generation = peer_store.peer_cache_recovery_generation();
        if startup_gate
            && recovered_generation != 0
            && recovered_generation != anchor.cache_generation
        {
            let round = PeerStoreVerifiedDeliveryWitnessRound {
                configured: witness_node_ids.len() as u64,
                failed: witness_node_ids.len() as u64,
                ..PeerStoreVerifiedDeliveryWitnessRound::default()
            };
            peer_store.record_client_delivery_witness_round(
                evaluated_at,
                recovered_generation,
                discovery.verified_delivery_witness_required_for_restore,
                discovery.verified_delivery_witness_min_verified,
                round,
            );
            if discovery.verified_delivery_witness_required_for_restore {
                peer_store.clear_restored_peer_cache_readiness_evidence(
                    evaluated_at,
                    "external_witness_unavailable",
                );
            }
            return PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected(
                "unavailable",
            );
        }

        let digest = match anchor.witness_digest() {
            Ok(digest) => digest,
            Err(_) => {
                let round = PeerStoreVerifiedDeliveryWitnessRound {
                    configured: witness_node_ids.len() as u64,
                    failed: witness_node_ids.len() as u64,
                    ..PeerStoreVerifiedDeliveryWitnessRound::default()
                };
                peer_store.record_client_delivery_witness_round(
                    evaluated_at,
                    anchor.cache_generation,
                    discovery.verified_delivery_witness_required_for_restore,
                    discovery.verified_delivery_witness_min_verified,
                    round,
                );
                if startup_gate {
                    peer_store.clear_restored_peer_cache_readiness_evidence(
                        evaluated_at,
                        "external_witness_invalid",
                    );
                }
                return PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected(
                    "unavailable",
                );
            }
        };
        let round = match witness_verified_delivery_anchor(
            peer_store,
            identity,
            client,
            &witness_node_ids,
            anchor.cache_generation,
            &digest,
        )
        .await
        {
            Ok(round) => Self::peer_store_delivery_witness_round(round),
            Err(_) => PeerStoreVerifiedDeliveryWitnessRound {
                configured: witness_node_ids.len() as u64,
                failed: witness_node_ids.len() as u64,
                ..PeerStoreVerifiedDeliveryWitnessRound::default()
            },
        };
        let status = peer_store.record_client_delivery_witness_round(
            unix_now_secs(),
            anchor.cache_generation,
            discovery.verified_delivery_witness_required_for_restore,
            discovery.verified_delivery_witness_min_verified,
            round,
        );
        let rejection_reason = match status {
            "rollback_detected" => Some("external_witness_rollback"),
            "conflict" => Some("external_witness_conflict"),
            "gap" => Some("external_witness_gap"),
            "partial" | "unavailable"
                if discovery.verified_delivery_witness_required_for_restore =>
            {
                Some("external_witness_unavailable")
            }
            _ => None,
        };
        if startup_gate {
            if let Some(reason) = rejection_reason {
                peer_store.clear_restored_peer_cache_readiness_evidence(unix_now_secs(), reason);
            }
        }
        if status == "verified" {
            PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Protected
        } else {
            PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected(status)
        }
    }

    /// Persists at most one new local generation after the prior generation is
    /// externally protected, then witnesses the exact newly written anchor.
    pub(super) async fn persist_peer_store_cache_with_delivery_witnesses(
        identity: &IdentityKeyPair,
        peer_store: &PeerStore,
        discovery: &DiscoveryConfig,
        control_http_client: &reqwest::Client,
        path: &str,
        now: u64,
        startup_gate: bool,
    ) -> Result<PeerStoreCachePersistOutcome> {
        let witnesses_configured = !discovery.verified_delivery_witness_node_ids.is_empty();
        if witnesses_configured {
            let primary_exists = tokio::fs::metadata(path).await.is_ok();
            let backup_exists = tokio::fs::metadata(Self::peer_cache_backup_path(path))
                .await
                .is_ok();
            let decision = Self::reconcile_peer_cache_delivery_witnesses(
                identity,
                peer_store,
                discovery,
                control_http_client,
                path,
                startup_gate,
            )
            .await;
            let fresh_bootstrap = decision
                == PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Missing
                && !primary_exists
                && !backup_exists;
            if decision != PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Protected
                && !fresh_bootstrap
            {
                let status = match decision {
                    PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected(status) => {
                        status
                    }
                    PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Missing => {
                        "anchor_missing"
                    }
                    PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Disabled => "disabled",
                    PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Protected => "verified",
                };
                peer_store.record_cache_save_status(
                    now,
                    "skipped",
                    format!("external_delivery_witness={status}"),
                );
                peer_store.mark_peer_cache_dirty();
                return Ok(PeerStoreCachePersistOutcome::Deferred);
            }
        } else {
            peer_store.record_client_delivery_witness_round(
                now,
                0,
                false,
                discovery.verified_delivery_witness_min_verified,
                PeerStoreVerifiedDeliveryWitnessRound::default(),
            );
        }

        if let Err(error) =
            Self::persist_peer_store_cache_once(identity, peer_store, path, now).await
        {
            // [PEER-CACHE-DIRTY-RECOVERY 2026-08-12 by Codex] The persistence
            // task claims the dirty bit immediately before exporting. A failed
            // atomic write did not acknowledge that evidence, so retain it for
            // a later bounded retry instead of waiting for another receipt.
            peer_store.mark_peer_cache_dirty();
            return Err(error);
        }
        if witnesses_configured {
            let decision = Self::reconcile_peer_cache_delivery_witnesses(
                identity,
                peer_store,
                discovery,
                control_http_client,
                path,
                startup_gate,
            )
            .await;
            if decision != PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Protected {
                peer_store.mark_peer_cache_dirty();
                return Ok(PeerStoreCachePersistOutcome::Deferred);
            }
        }
        Ok(PeerStoreCachePersistOutcome::Persisted)
    }

    pub(super) fn spawn_peer_store_persistence_task(
        &self,
        peer_store: Arc<PeerStore>,
        control_http_client: Arc<reqwest::Client>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.discovery.enabled {
            return None;
        }
        let Some(path) = self.config.discovery.peer_cache_path.clone() else {
            return None;
        };

        let interval_secs = self.config.discovery.peer_cache_write_interval_secs;
        let shutdown = Arc::clone(&self.shutdown);
        let identity = Arc::new(self.identity.clone());
        let discovery = Arc::new(self.config.discovery.clone());
        let mut rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            let mut consecutive_retry_rounds = 0u32;
            let mut retry_not_before = None;
            let persist_snapshot = |identity: Arc<IdentityKeyPair>,
                                    peer_store: Arc<PeerStore>,
                                    discovery: Arc<DiscoveryConfig>,
                                    control_http_client: Arc<reqwest::Client>,
                                    path: String,
                                    reason: &'static str| async move {
                let now = unix_now_secs();
                match Self::persist_peer_store_cache_with_delivery_witnesses(
                    identity.as_ref(),
                    &peer_store,
                    discovery.as_ref(),
                    control_http_client.as_ref(),
                    &path,
                    now,
                    false,
                )
                .await
                {
                    Ok(PeerStoreCachePersistOutcome::Persisted) => {
                        debug!(
                            source = %path,
                            reason = reason,
                            "[DISCOVERY] Peer cache snapshot persisted"
                        );
                        true
                    }
                    Ok(PeerStoreCachePersistOutcome::Deferred) => {
                        debug!(
                            source = %path,
                            reason = reason,
                            "[DISCOVERY] Peer cache snapshot retained for a bounded retry"
                        );
                        false
                    }
                    Err(e) => {
                        warn!(
                            source = %path,
                            reason = reason,
                            error = %e,
                            "[DISCOVERY] Failed to persist peer cache snapshot"
                        );
                        false
                    }
                }
            };

            let update_retry =
                |stable: bool,
                 consecutive_retry_rounds: &mut u32,
                 retry_not_before: &mut Option<tokio::time::Instant>| {
                    if stable {
                        *consecutive_retry_rounds = 0;
                        *retry_not_before = None;
                        return;
                    }
                    *consecutive_retry_rounds = consecutive_retry_rounds.saturating_add(1);
                    *retry_not_before = Some(
                        tokio::time::Instant::now()
                            + Self::peer_cache_persist_retry_delay(*consecutive_retry_rounds),
                    );
                };

            loop {
                tokio::select! {
                    // [PEER-CACHE-RETRY-STATE 2026-08-12 by Codex] Shutdown
                    // outranks a simultaneously-ready retry or periodic tick.
                    biased;
                    _ = rx.recv() => {
                        peer_store.take_peer_cache_dirty();
                        let _ = persist_snapshot(
                            Arc::clone(&identity),
                            Arc::clone(&peer_store),
                            Arc::clone(&discovery),
                            Arc::clone(&control_http_client),
                            path.clone(),
                            "shutdown",
                        )
                        .await;
                        break;
                    }
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) {
                            peer_store.take_peer_cache_dirty();
                            persist_snapshot(
                                Arc::clone(&identity),
                                Arc::clone(&peer_store),
                                Arc::clone(&discovery),
                                Arc::clone(&control_http_client),
                                path.clone(),
                                "shutdown_flag",
                            )
                            .await;
                            break;
                        }
                        if retry_not_before
                            .is_some_and(|deadline| tokio::time::Instant::now() < deadline)
                        {
                            continue;
                        }
                        peer_store.take_peer_cache_dirty();
                        let stable = persist_snapshot(
                            Arc::clone(&identity),
                            Arc::clone(&peer_store),
                            Arc::clone(&discovery),
                            Arc::clone(&control_http_client),
                            path.clone(),
                            "interval",
                        )
                        .await;
                        update_retry(
                            stable,
                            &mut consecutive_retry_rounds,
                            &mut retry_not_before,
                        );
                    }
                    _ = peer_store.wait_for_peer_cache_dirty(), if retry_not_before.is_none() => {
                        tokio::time::sleep(Duration::from_millis(
                            CLIENT_DELIVERY_CACHE_FLUSH_DEBOUNCE_MILLIS,
                        )).await;
                        if peer_store.take_peer_cache_dirty() {
                            let stable = persist_snapshot(
                                Arc::clone(&identity),
                                Arc::clone(&peer_store),
                                Arc::clone(&discovery),
                                Arc::clone(&control_http_client),
                                path.clone(),
                                "peer_evidence",
                            ).await;
                            update_retry(
                                stable,
                                &mut consecutive_retry_rounds,
                                &mut retry_not_before,
                            );
                        }
                    }
                    _ = tokio::time::sleep_until(
                        retry_not_before.unwrap_or_else(tokio::time::Instant::now)
                    ), if retry_not_before.is_some() => {
                        retry_not_before = None;
                        if peer_store.take_peer_cache_dirty() {
                            let stable = persist_snapshot(
                                Arc::clone(&identity),
                                Arc::clone(&peer_store),
                                Arc::clone(&discovery),
                                Arc::clone(&control_http_client),
                                path.clone(),
                                "bounded_retry",
                            ).await;
                            update_retry(
                                stable,
                                &mut consecutive_retry_rounds,
                                &mut retry_not_before,
                            );
                        } else {
                            consecutive_retry_rounds = 0;
                        }
                    }
                }
            }
        }))
    }

    pub(super) fn peer_cache_persist_retry_delay(consecutive_retry_rounds: u32) -> Duration {
        // [PEER-CACHE-RETRY-STATE 2026-08-12 by Codex] Keep retry state
        // process-local. It limits I/O and witness pressure only; it never
        // enters the signed cache document or changes route selection.
        let shift = consecutive_retry_rounds.saturating_sub(1).min(6);
        let multiplier = 1u64 << shift;
        Duration::from_millis(
            PEER_CACHE_PERSIST_RETRY_BASE_MILLIS
                .saturating_mul(multiplier)
                .min(PEER_CACHE_PERSIST_RETRY_MAX_MILLIS),
        )
    }

    pub(super) async fn save_peer_store_cache_snapshot(
        identity: &IdentityKeyPair,
        peer_store: &PeerStore,
        path: &str,
        now: u64,
    ) -> Result<PeerStoreCachePersistReport> {
        let path = PathBuf::from(path);
        let descriptor_snapshot = peer_store.export_peer_cache_snapshot(now);
        let routeability_evidence = peer_store.export_routeability_cache_evidence(now);
        let route_quarantine_evidence = peer_store.export_route_quarantine_cache_evidence(now);
        let two_hop_path_proof_events = peer_store.export_two_hop_path_proof_cache_events(now);
        let two_hop_path_proof_event_count = two_hop_path_proof_events.len();
        let two_hop_path_proof_stability_ready = peer_store
            .status(now)
            .two_hop_path_proof_history
            .stability_ready;
        let three_hop_path_proof_events = peer_store.export_three_hop_path_proof_cache_events(now);
        let three_hop_path_proof_event_count = three_hop_path_proof_events.len();
        let three_hop_path_proof_stability_ready = peer_store
            .status(now)
            .three_hop_path_proof_history
            .stability_ready;
        let route_domain_certificates =
            peer_store.export_route_domain_attestation_certificates(now);
        let route_domain_certificate_count = route_domain_certificates.len();
        let verified_client_delivery_evidence =
            peer_store.export_verified_client_delivery_cache_evidence(now);
        let verified_client_delivery_count = verified_client_delivery_evidence
            .map(|evidence| evidence.verified_deliveries)
            .unwrap_or(0);
        let verified_client_delivery_generation = Self::next_peer_cache_client_delivery_generation(
            path.to_string_lossy().as_ref(),
            identity,
        )
        .await?;
        let document = PeerStoreCacheDocument::new(
            descriptor_snapshot,
            routeability_evidence,
            route_quarantine_evidence,
            two_hop_path_proof_events,
            three_hop_path_proof_events,
            route_domain_certificates,
            verified_client_delivery_generation,
            verified_client_delivery_evidence,
            identity,
        )?;
        let client_delivery_anchor =
            PeerStoreVerifiedClientDeliveryAnchor::new(&document, identity)?;
        let bytes = document.to_json_pretty()?;
        // [PEER-CACHE-BACKUP-DURABILITY 2026-09-21 by Codex] A single
        // spawn_blocking transaction now syncs the new temp, preserves the
        // pinned old primary as an atomically replaced durable backup, and
        // only then publishes the new primary. Any typed failure prevents the
        // dependent recovery anchor from advancing.
        publish_peer_cache_snapshot(path.clone(), bytes)
            .await
            .map_err(|error| ServerError::internal(error.to_string()))?;
        Self::write_peer_cache_client_delivery_anchor(
            path.to_string_lossy().as_ref(),
            &client_delivery_anchor,
        )
        .await?;
        Ok(PeerStoreCachePersistReport {
            two_hop_events: two_hop_path_proof_event_count,
            two_hop_stability_ready: two_hop_path_proof_stability_ready,
            three_hop_events: three_hop_path_proof_event_count,
            three_hop_stability_ready: three_hop_path_proof_stability_ready,
            route_domain_certificates: route_domain_certificate_count,
            client_deliveries: verified_client_delivery_count,
            client_delivery_generation: verified_client_delivery_generation,
        })
    }

    pub(super) async fn next_peer_cache_client_delivery_generation(
        path: &str,
        identity: &IdentityKeyPair,
    ) -> Result<u64> {
        let mut high_water_mark = match Self::read_peer_cache_client_delivery_anchor(path, identity)
            .await
        {
            PeerStoreVerifiedClientDeliveryAnchorState::Verified(anchor) => anchor.cache_generation,
            _ => 0,
        };

        for candidate in [PathBuf::from(path), Self::peer_cache_backup_path(path)] {
            let Ok(bytes) = Self::read_bounded_file(&candidate, DISCOVERY_SNAPSHOT_MAX_BYTES).await
            else {
                continue;
            };
            let Ok(document) = PeerStoreCacheDocument::from_json_bytes(&bytes) else {
                continue;
            };
            if document.verified_client_delivery_schema_version
                == VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION
                && document
                    .verify_verified_client_delivery_signature(identity)
                    .is_ok()
            {
                high_water_mark = high_water_mark.max(document.verified_client_delivery_generation);
            }
        }

        high_water_mark.checked_add(1).ok_or_else(|| {
            ServerError::internal("verified client delivery cache generation exhausted")
        })
    }

    pub(super) async fn write_peer_cache_client_delivery_anchor(
        cache_path: &str,
        anchor: &PeerStoreVerifiedClientDeliveryAnchor,
    ) -> Result<()> {
        let anchor_path = Self::peer_cache_client_delivery_anchor_path(cache_path);
        let tmp_path = PathBuf::from(format!("{}.tmp", anchor_path.display()));
        let bytes = anchor.to_json_pretty()?;
        let mut tmp_file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .await?;
        tmp_file.write_all(&bytes).await?;
        tmp_file.flush().await?;
        tmp_file.sync_all().await?;
        drop(tmp_file);
        tokio::fs::rename(&tmp_path, &anchor_path).await?;
        Self::sync_parent_dir_for_durability(&anchor_path).await
    }

    pub(super) async fn sync_parent_dir_for_durability(path: &PathBuf) -> Result<()> {
        let Some(parent) = path.parent() else {
            return Ok(());
        };
        if parent.as_os_str().is_empty() {
            return Ok(());
        }

        match tokio::fs::File::open(parent).await {
            Ok(dir) => {
                dir.sync_all().await?;
                Ok(())
            }
            Err(e) if cfg!(target_os = "windows") => {
                debug!(
                    source = %path.display(),
                    error = %e,
                    "[DISCOVERY] Parent directory fsync unavailable on this platform"
                );
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub(super) fn peer_cache_backup_path(path: &str) -> PathBuf {
        PathBuf::from(format!("{path}.bak"))
    }

    pub(super) fn peer_cache_client_delivery_anchor_path(path: &str) -> PathBuf {
        PathBuf::from(format!("{path}.verified-client-delivery-anchor"))
    }

    pub(super) async fn persist_peer_store_cache_once(
        identity: &IdentityKeyPair,
        peer_store: &PeerStore,
        path: &str,
        now: u64,
    ) -> Result<()> {
        let pending_snapshot = peer_store.export_bootstrap_snapshot(now, now, false, None);
        if pending_snapshot.verified_count_at(now) == 0
            && Self::peer_cache_has_usable_recovery_snapshot(path, now).await
        {
            peer_store.record_cache_save_status(now, "skipped", "preserved_existing_snapshot");
            return Ok(());
        }

        match Self::save_peer_store_cache_snapshot(identity, peer_store, path, now).await {
            Ok(report) => {
                peer_store.record_cache_save_status(now, "success", "snapshot_persisted");
                peer_store.record_audit_event(
                    now,
                    "route_domain_certificate_cache_persist",
                    "success",
                    format!("persisted={}", report.route_domain_certificates),
                );
                peer_store.record_routeability_cache_rollback_protection(
                    now,
                    report.client_delivery_generation,
                    "anchored",
                );
                peer_store.record_two_hop_proof_cache_persisted(
                    now,
                    report.two_hop_events,
                    report.two_hop_stability_ready,
                );
                peer_store.record_three_hop_proof_cache_persisted(
                    now,
                    report.three_hop_events,
                    report.three_hop_stability_ready,
                );
                peer_store.record_client_delivery_cache_persisted(
                    now,
                    report.client_deliveries,
                    report.client_delivery_generation,
                );
                Ok(())
            }
            Err(e) => {
                peer_store.record_cache_save_status(now, "failed", "write_failed");
                Err(e)
            }
        }
    }

    pub(super) async fn peer_cache_has_usable_recovery_snapshot(path: &str, now: u64) -> bool {
        let primary = PathBuf::from(path);
        if Self::cache_snapshot_file_has_usable_peers(&primary, now).await {
            return true;
        }

        let backup = Self::peer_cache_backup_path(path);
        Self::cache_snapshot_file_has_usable_peers(&backup, now).await
    }

    pub(super) async fn cache_snapshot_file_has_usable_peers(path: &PathBuf, now: u64) -> bool {
        let Ok(bytes) = Self::read_bounded_file(path, DISCOVERY_SNAPSHOT_MAX_BYTES).await else {
            return false;
        };
        let Ok(document) = PeerStoreCacheDocument::from_json_bytes(&bytes) else {
            return false;
        };

        document.descriptor_snapshot.verified_count_at(now) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_store_cache_retry_backoff_is_bounded() {
        assert_eq!(
            Server::peer_cache_persist_retry_delay(1),
            Duration::from_secs(1)
        );
        assert_eq!(
            Server::peer_cache_persist_retry_delay(2),
            Duration::from_secs(2)
        );
        assert_eq!(
            Server::peer_cache_persist_retry_delay(3),
            Duration::from_secs(4)
        );
        assert_eq!(
            Server::peer_cache_persist_retry_delay(7),
            Duration::from_secs(60)
        );
        assert_eq!(
            Server::peer_cache_persist_retry_delay(u32::MAX),
            Duration::from_secs(60)
        );
    }
}
