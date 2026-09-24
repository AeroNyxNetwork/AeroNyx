// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/route_domain_certificates.rs
// ============================================================================
//! Host-local route-domain attestation policy and certificate cache.
//!
//! [ROUTE-DOMAIN-CERTIFICATE-SPLIT 2026-09-25 by Codex] Keeps strict
//! multi-hop trust admission, bounded certificate recovery, and policy
//! rotation together without changing PeerStore's public API or wire surface.

use std::collections::HashMap;
#[cfg(test)]
use std::sync::{Arc, Barrier};

use aeronyx_core::protocol::discovery::{
    RouteDomainAttestationCertificateV1, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
};
use serde::{Deserialize, Serialize};

use super::{PeerStore, ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES};

const MAX_ROUTE_DOMAIN_CERTIFICATES: usize = ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES;

/// Host-local policy for admitting portable route-domain certificates.
///
/// [ROUTE-DOMAIN-ATTESTED-SELECTION 2026-08-03 by Codex] This policy is never
/// serialized into public status. Attestor identities and opaque assignments
/// are trust input, not discovery metadata, votes, consensus, or Sybil proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PeerStoreRouteDomainAttestorPolicy {
    pub(super) pinned_route_domains: HashMap<[u8; 32], [u8; 16]>,
    pub(super) allowed_attestors: Vec<[u8; 32]>,
    pub(super) minimum_attestors: usize,
    pub(super) strict_multi_hop: bool,
}

impl Default for PeerStoreRouteDomainAttestorPolicy {
    fn default() -> Self {
        Self {
            pinned_route_domains: HashMap::new(),
            allowed_attestors: Vec::new(),
            minimum_attestors: 1,
            strict_multi_hop: false,
        }
    }
}

/// Errors returned while installing host-local route-domain trust policy.
///
/// [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] This focused error
/// type prevents certificate lifecycle concerns from leaking into unrelated
/// signed-descriptor admission call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RouteDomainAttestorPolicyError {
    /// Local route-domain pins, attestors, threshold, or strict mode are invalid.
    #[error("route-domain attestor policy is invalid")]
    InvalidPolicy,
}

/// Errors returned while importing one portable route-domain certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RouteDomainCertificateImportError {
    /// A portable route-domain certificate failed local policy verification.
    #[error("route-domain attestation certificate was rejected")]
    Rejected,
    /// A valid but older/weaker certificate cannot replace fresher evidence.
    #[error("route-domain attestation certificate is stale")]
    Stale,
    /// The bounded process-local certificate cache is full.
    #[error("route-domain certificate capacity exceeded")]
    CapacityExceeded,
}

/// Aggregate result of restoring independently signed route-domain certificates.
///
/// [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] This report is
/// intentionally identity-blind. Operators can diagnose cache health without
/// learning subjects, route-domain tokens, attestors, or certificate hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteDomainCertificateCacheReport {
    /// Number of certificate records presented by the bounded local cache.
    pub total: usize,
    /// Number of current certificates inserted or upgraded.
    pub restored: usize,
    /// Number of certificates already present with the same signed content.
    pub unchanged: usize,
    /// Number rejected by bounds, local pins, quorum, signature, or freshness.
    pub rejected: usize,
}

impl PeerStoreRouteDomainCertificateCacheReport {
    /// Empty recovery report.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total: 0,
            restored: 0,
            unchanged: 0,
            rejected: 0,
        }
    }
}

impl PeerStore {
    /// Replaces the exact local route-domain and attestor trust policy.
    ///
    /// Policy rotation invalidates all process-local certificates so every
    /// subject must be re-verified under the new pins. Identical configuration
    /// is idempotent and preserves valid evidence. This method never publishes
    /// attestor identities or opaque route-domain tokens.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestorPolicyError::InvalidPolicy`] for zero or
    /// duplicate identities/tokens, invalid threshold, unsafe strict mode, or
    /// attestor/subject overlap that would make strict coverage impossible.
    pub fn configure_route_domain_attestor_policy(
        &self,
        pinned_route_domains: &[([u8; 32], [u8; 16])],
        allowed_attestors: &[[u8; 32]],
        minimum_attestors: usize,
        strict_multi_hop: bool,
    ) -> Result<(), RouteDomainAttestorPolicyError> {
        let mut canonical_domains = pinned_route_domains.to_vec();
        canonical_domains.sort_unstable_by_key(|(node_id, _)| *node_id);
        let domains_invalid = canonical_domains.len() > MAX_ROUTE_DOMAIN_CERTIFICATES
            || canonical_domains
                .iter()
                .any(|(node_id, domain)| *node_id == [0u8; 32] || *domain == [0u8; 16])
            || canonical_domains
                .windows(2)
                .any(|entries| entries[0].0 == entries[1].0);

        let mut canonical_attestors = allowed_attestors.to_vec();
        canonical_attestors.sort_unstable();
        let attestors_invalid = canonical_attestors.len() > 16
            || canonical_attestors
                .iter()
                .any(|node_id| *node_id == [0u8; 32])
            || canonical_attestors
                .windows(2)
                .any(|entries| entries[0] == entries[1])
            || canonical_attestors.iter().any(|attestor| {
                canonical_domains
                    .binary_search_by_key(attestor, |(subject, _)| *subject)
                    .is_ok()
            });
        let empty_policy_invalid =
            canonical_attestors.is_empty() && (strict_multi_hop || minimum_attestors != 1);
        let populated_policy_invalid = !canonical_attestors.is_empty()
            && (minimum_attestors == 0 || minimum_attestors > canonical_attestors.len());
        if domains_invalid
            || attestors_invalid
            || empty_policy_invalid
            || populated_policy_invalid
            || (strict_multi_hop && canonical_domains.is_empty())
        {
            return Err(RouteDomainAttestorPolicyError::InvalidPolicy);
        }

        let next = PeerStoreRouteDomainAttestorPolicy {
            pinned_route_domains: canonical_domains.into_iter().collect(),
            allowed_attestors: canonical_attestors,
            minimum_attestors,
            strict_multi_hop,
        };
        let mut current = self.route_domain_attestor_policy.write();
        if *current != next {
            self.route_domain_certificates.write().clear();
            *current = next;
        }
        Ok(())
    }

    /// Verifies and retains one portable route-domain certificate.
    ///
    /// The certificate must bind an exact locally pinned subject/domain pair
    /// and satisfy the verifier's current attestor quorum. At most one freshest
    /// certificate is retained per subject; stale evidence cannot overwrite a
    /// longer-lived valid certificate. Evidence remains process-local unless a
    /// caller persists the bounded export and reimports it under the same
    /// current local policy after restart.
    ///
    /// # Errors
    /// Returns a bounded [`RouteDomainCertificateImportError`] when policy,
    /// certificate, capacity, subject/domain binding, signature, expiry, or
    /// replacement order fails.
    pub fn import_route_domain_attestation_certificate(
        &self,
        certificate: RouteDomainAttestationCertificateV1,
        now: u64,
    ) -> Result<bool, RouteDomainCertificateImportError> {
        // [ROUTE-DOMAIN-POLICY-ROTATION 2026-09-25 by Codex] Hold the policy
        // read lock through certificate replacement. Rotation acquires these
        // locks in the same policy -> certificate order, so a certificate
        // verified under an old policy cannot reappear after rotation clears
        // the cache and installs the new policy.
        let policy = self.route_domain_attestor_policy.read();
        #[cfg(test)]
        if let Some(gate) = self.route_domain_import_test_gate.as_ref() {
            gate.policy_read.wait();
            gate.continue_after_policy.wait();
        }
        let expected_domain = policy
            .pinned_route_domains
            .get(&certificate.subject_node_id)
            .copied()
            .ok_or(RouteDomainCertificateImportError::Rejected)?;
        if expected_domain != certificate.route_domain {
            return Err(RouteDomainCertificateImportError::Rejected);
        }
        certificate
            .verify_with_policy_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &policy.allowed_attestors,
                policy.minimum_attestors,
                now,
            )
            .map_err(|_| RouteDomainCertificateImportError::Rejected)?;
        let new_expiry = Self::route_domain_certificate_effective_expiry(
            &certificate,
            &policy.allowed_attestors,
            policy.minimum_attestors,
        )
        .ok_or(RouteDomainCertificateImportError::Rejected)?;

        let subject_node_id = certificate.subject_node_id;
        let mut certificates = self.route_domain_certificates.write();
        if let Some(current) = certificates.get(&subject_node_id) {
            if current.hash() == certificate.hash() {
                return Ok(false);
            }
            let current_valid = current
                .verify_with_policy_at(
                    &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                    &policy.allowed_attestors,
                    policy.minimum_attestors,
                    now,
                )
                .is_ok();
            let current_expiry = Self::route_domain_certificate_effective_expiry(
                current,
                &policy.allowed_attestors,
                policy.minimum_attestors,
            )
            .unwrap_or(0);
            if current_valid && new_expiry <= current_expiry {
                return Err(RouteDomainCertificateImportError::Stale);
            }
        } else if certificates.len() >= MAX_ROUTE_DOMAIN_CERTIFICATES {
            return Err(RouteDomainCertificateImportError::CapacityExceeded);
        }
        certificates.insert(subject_node_id, certificate);
        Ok(true)
    }

    /// Exports only certificates that still satisfy the current local policy.
    ///
    /// The result is deterministic by subject identity and is intended only
    /// for the host-local peer cache. Public discovery and monitoring surfaces
    /// must never publish this trust metadata.
    #[must_use]
    pub fn export_route_domain_attestation_certificates(
        &self,
        now: u64,
    ) -> Vec<RouteDomainAttestationCertificateV1> {
        // [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] Revalidate
        // at export time so expired evidence never gains a fresh cache lease.
        let policy = self.route_domain_attestor_policy.read();
        let certificates = self.route_domain_certificates.read();
        let mut exported = certificates
            .values()
            .filter(|certificate| {
                policy
                    .pinned_route_domains
                    .get(&certificate.subject_node_id)
                    .is_some_and(|expected_domain| *expected_domain == certificate.route_domain)
                    && certificate
                        .verify_with_policy_at(
                            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                            &policy.allowed_attestors,
                            policy.minimum_attestors,
                            now,
                        )
                        .is_ok()
            })
            .cloned()
            .collect::<Vec<_>>();
        exported.sort_unstable_by_key(|certificate| certificate.subject_node_id);
        exported.truncate(ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES);
        exported
    }

    /// Restores a bounded certificate section under the current local policy.
    ///
    /// Every certificate is reverified independently. Invalid entries are
    /// isolated and counted rather than preventing descriptor or proof-cache
    /// recovery from the same local document.
    #[must_use]
    pub fn restore_route_domain_attestation_certificates(
        &self,
        certificates: &[RouteDomainAttestationCertificateV1],
        now: u64,
    ) -> PeerStoreRouteDomainCertificateCacheReport {
        let mut report = PeerStoreRouteDomainCertificateCacheReport::empty();
        report.total = certificates.len();
        for certificate in certificates
            .iter()
            .take(ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES)
            .cloned()
        {
            match self.import_route_domain_attestation_certificate(certificate, now) {
                Ok(true) => report.restored += 1,
                Ok(false) => report.unchanged += 1,
                Err(_) => report.rejected += 1,
            }
        }
        report.rejected = report.rejected.saturating_add(
            certificates
                .len()
                .saturating_sub(ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES),
        );
        self.record_audit_event(
            now,
            "route_domain_certificate_cache_restore",
            if report.rejected == 0 {
                "accepted"
            } else {
                "warning"
            },
            format!(
                "total={} restored={} unchanged={} rejected={}",
                report.total, report.restored, report.unchanged, report.rejected
            ),
        );
        report
    }

    fn route_domain_certificate_effective_expiry(
        certificate: &RouteDomainAttestationCertificateV1,
        allowed_attestors: &[[u8; 32]],
        minimum_attestors: usize,
    ) -> Option<u64> {
        let mut expiries = certificate
            .attestations
            .iter()
            .filter(|attestation| allowed_attestors.contains(&attestation.attestor_node_id))
            .map(|attestation| attestation.expires_at)
            .collect::<Vec<_>>();
        expiries.sort_unstable_by(|left, right| right.cmp(left));
        expiries.get(minimum_attestors.checked_sub(1)?).copied()
    }

    pub(super) fn route_domain_certificate_allows_multi_hop(
        &self,
        subject_node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        let policy = self.route_domain_attestor_policy.read();
        if !policy.strict_multi_hop {
            return true;
        }
        let Some(expected_domain) = policy.pinned_route_domains.get(subject_node_id) else {
            return false;
        };
        self.route_domain_certificates
            .read()
            .get(subject_node_id)
            .is_some_and(|certificate| {
                certificate.route_domain == *expected_domain
                    && certificate
                        .verify_with_policy_at(
                            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                            &policy.allowed_attestors,
                            policy.minimum_attestors,
                            now,
                        )
                        .is_ok()
            })
    }
}

#[cfg(test)]
pub(super) struct RouteDomainImportTestGate {
    pub(super) policy_read: Arc<Barrier>,
    pub(super) continue_after_policy: Arc<Barrier>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy,
        RouteDomainAttestationCertificateV1, RouteDomainAttestationV1, SignedNodeDescriptor,
    };

    fn signed_descriptor_for(
        kp: &IdentityKeyPair,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            1_700_000_000,
            expires_at,
            "test",
        );
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 128,
            max_bps: Some(500_000_000),
            max_pps: None,
        };
        descriptor.policy = NodePolicy::default();
        SignedNodeDescriptor::sign(descriptor, kp).unwrap()
    }

    fn route_domain_certificate_for(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        attestors: &[&IdentityKeyPair],
    ) -> RouteDomainAttestationCertificateV1 {
        let statements = attestors
            .iter()
            .enumerate()
            .map(|(index, attestor)| {
                RouteDomainAttestationV1::new_signed(
                    subject_node_id,
                    route_domain,
                    issued_at + u64::try_from(index).unwrap(),
                    expires_at,
                    attestor,
                )
                .unwrap()
            })
            .collect();
        RouteDomainAttestationCertificateV1::new_verified(
            subject_node_id,
            route_domain,
            statements,
            issued_at + u64::try_from(attestors.len()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_strict_route_domain_attestations_gate_only_multi_hop_selection() {
        // [ROUTE-DOMAIN-ATTESTED-SELECTION 2026-08-03 by Codex] Strict mode
        // fails closed for multi-hop routing without changing the established
        // direct single-hop compatibility path.
        let store = PeerStore::new();
        let now = 1_700_010_000;
        let middle = IdentityKeyPair::generate();
        let terminal = IdentityKeyPair::generate();
        let attestor_a = IdentityKeyPair::generate();
        let attestor_b = IdentityKeyPair::generate();
        let middle_domain = [0x31; 16];
        let terminal_domain = [0x32; 16];

        store
            .configure_route_domain_attestor_policy(
                &[
                    (middle.public_key_bytes(), middle_domain),
                    (terminal.public_key_bytes(), terminal_domain),
                ],
                &[attestor_a.public_key_bytes(), attestor_b.public_key_bytes()],
                2,
                true,
            )
            .unwrap();

        let mut middle_descriptor = signed_descriptor_for(&middle, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("http://198.51.100.20:8422".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle).unwrap();
        let mut terminal_descriptor = signed_descriptor_for(&terminal, 1, now + 2_000);
        terminal_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        terminal_descriptor.descriptor.public_endpoint =
            Some("http://203.0.113.20:8422".to_string());
        terminal_descriptor =
            SignedNodeDescriptor::sign(terminal_descriptor.descriptor, &terminal).unwrap();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(terminal_descriptor, now, "gossip_announce")
            .unwrap();
        store.record_route_forward_success(&middle.public_key_bytes(), now);
        store.record_route_forward_success(&terminal.public_key_bytes(), now);

        assert!(store
            .route_path_with_capabilities_excluding(&[NodeCapability::ChatRelay], now, &[])
            .is_some());
        assert!(store
            .route_path_with_capabilities_excluding(
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now,
                &[],
            )
            .is_none());
        assert_eq!(
            store
                .route_probe_candidates_with_capability_excluding(
                    NodeCapability::OnionMiddle,
                    now,
                    8,
                    &[],
                )
                .len(),
            1
        );
        assert!(store
            .multi_hop_route_probe_candidates_with_capability_excluding(
                NodeCapability::OnionMiddle,
                now,
                8,
                &[],
            )
            .is_empty());

        for (subject, route_domain) in [
            (middle.public_key_bytes(), middle_domain),
            (terminal.public_key_bytes(), terminal_domain),
        ] {
            let certificate = route_domain_certificate_for(
                subject,
                route_domain,
                now - 2,
                now + 600,
                &[&attestor_a, &attestor_b],
            );
            assert!(store
                .import_route_domain_attestation_certificate(certificate, now)
                .unwrap());
        }

        let path = store
            .route_path_with_capabilities_excluding(
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now,
                &[],
            )
            .expect("quorum-valid certificates should unlock the complete route");
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].node_id(), middle.public_key_bytes());
        assert_eq!(path[1].node_id(), terminal.public_key_bytes());
        assert_eq!(
            store
                .multi_hop_route_probe_candidates_with_capability_excluding(
                    NodeCapability::OnionMiddle,
                    now,
                    8,
                    &[],
                )
                .len(),
            1
        );

        assert!(store
            .route_path_with_capabilities_excluding(
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now + 600,
                &[],
            )
            .is_none());
        assert!(store
            .route_path_with_capabilities_excluding(&[NodeCapability::ChatRelay], now + 600, &[],)
            .is_some());
    }

    #[test]
    fn test_route_domain_certificate_import_rejects_untrusted_and_stale_evidence() {
        // [ROUTE-DOMAIN-ATTESTED-SELECTION 2026-08-03 by Codex] Certificate
        // replacement is monotonic under one policy epoch, while any policy
        // rotation clears process evidence and requires fresh verification.
        let store = PeerStore::new();
        let now = 1_700_020_000;
        let subject = IdentityKeyPair::generate();
        let attestor_a = IdentityKeyPair::generate();
        let attestor_b = IdentityKeyPair::generate();
        let untrusted = IdentityKeyPair::generate();
        let replacement_attestor = IdentityKeyPair::generate();
        let route_domain = [0x41; 16];
        let allowed = [attestor_a.public_key_bytes(), attestor_b.public_key_bytes()];

        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &allowed,
                2,
                true,
            )
            .unwrap();
        let untrusted_certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now - 2,
            now + 900,
            &[&attestor_a, &untrusted],
        );
        assert!(matches!(
            store.import_route_domain_attestation_certificate(untrusted_certificate, now),
            Err(RouteDomainCertificateImportError::Rejected)
        ));

        let current = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now - 2,
            now + 800,
            &[&attestor_a, &attestor_b],
        );
        assert!(store
            .import_route_domain_attestation_certificate(current.clone(), now)
            .unwrap());
        assert!(!store
            .import_route_domain_attestation_certificate(current, now)
            .unwrap());

        let stale = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now - 1,
            now + 700,
            &[&attestor_a, &attestor_b],
        );
        assert!(matches!(
            store.import_route_domain_attestation_certificate(stale, now),
            Err(RouteDomainCertificateImportError::Stale)
        ));
        assert!(store.route_domain_certificate_allows_multi_hop(&subject.public_key_bytes(), now));

        let rotated_allowed = [
            attestor_a.public_key_bytes(),
            attestor_b.public_key_bytes(),
            replacement_attestor.public_key_bytes(),
        ];
        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &rotated_allowed,
                2,
                true,
            )
            .unwrap();
        assert!(!store.route_domain_certificate_allows_multi_hop(&subject.public_key_bytes(), now));
    }

    #[test]
    fn test_route_domain_certificate_cache_revalidates_across_restart() {
        // [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] A cache is
        // only transport. A fresh PeerStore must independently install policy
        // and verify quorum/freshness before multi-hop becomes eligible.
        let now = 1_700_025_000;
        let subject = IdentityKeyPair::generate();
        let attestor_a = IdentityKeyPair::generate();
        let attestor_b = IdentityKeyPair::generate();
        let untrusted_a = IdentityKeyPair::generate();
        let untrusted_b = IdentityKeyPair::generate();
        let route_domain = [0x51; 16];
        let allowed = [attestor_a.public_key_bytes(), attestor_b.public_key_bytes()];
        let certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now - 2,
            now + 600,
            &[&attestor_a, &attestor_b],
        );

        let source = PeerStore::new();
        source
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &allowed,
                2,
                true,
            )
            .unwrap();
        assert!(source
            .import_route_domain_attestation_certificate(certificate, now)
            .unwrap());
        let exported = source.export_route_domain_attestation_certificates(now);
        assert_eq!(exported.len(), 1);

        let restored = PeerStore::new();
        restored
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &allowed,
                2,
                true,
            )
            .unwrap();
        let report = restored.restore_route_domain_attestation_certificates(&exported, now);
        assert_eq!(report.total, 1);
        assert_eq!(report.restored, 1);
        assert_eq!(report.unchanged, 0);
        assert_eq!(report.rejected, 0);
        assert!(
            restored.route_domain_certificate_allows_multi_hop(&subject.public_key_bytes(), now)
        );

        let duplicate = restored.restore_route_domain_attestation_certificates(&exported, now);
        assert_eq!(duplicate.restored, 0);
        assert_eq!(duplicate.unchanged, 1);
        assert_eq!(duplicate.rejected, 0);

        let wrong_policy = PeerStore::new();
        wrong_policy
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &[
                    untrusted_a.public_key_bytes(),
                    untrusted_b.public_key_bytes(),
                ],
                2,
                true,
            )
            .unwrap();
        let rejected = wrong_policy.restore_route_domain_attestation_certificates(&exported, now);
        assert_eq!(rejected.restored, 0);
        assert_eq!(rejected.rejected, 1);
        assert!(!wrong_policy
            .route_domain_certificate_allows_multi_hop(&subject.public_key_bytes(), now));

        assert!(source
            .export_route_domain_attestation_certificates(now + 600)
            .is_empty());
    }

    #[test]
    fn test_policy_rotation_cannot_restore_certificate_verified_under_old_policy() {
        // [ROUTE-DOMAIN-POLICY-ROTATION 2026-09-25 by Codex] The barriers
        // force import to hold the old policy read lock while rotation waits;
        // the final cache must reflect the rotated policy, never stale
        // evidence inserted after its clear.
        let now = 1_700_030_000;
        let subject = IdentityKeyPair::generate();
        let old_attestor_a = IdentityKeyPair::generate();
        let old_attestor_b = IdentityKeyPair::generate();
        let new_attestor_a = IdentityKeyPair::generate();
        let new_attestor_b = IdentityKeyPair::generate();
        let route_domain = [0x61; 16];
        let old_allowed = [
            old_attestor_a.public_key_bytes(),
            old_attestor_b.public_key_bytes(),
        ];
        let new_allowed = [
            new_attestor_a.public_key_bytes(),
            new_attestor_b.public_key_bytes(),
        ];
        let old_certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now - 2,
            now + 900,
            &[&old_attestor_a, &old_attestor_b],
        );
        let mut store = PeerStore::new();
        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &old_allowed,
                2,
                true,
            )
            .unwrap();
        assert!(store
            .import_route_domain_attestation_certificate(old_certificate.clone(), now)
            .unwrap());

        let gate = Arc::new(RouteDomainImportTestGate {
            policy_read: Arc::new(Barrier::new(2)),
            continue_after_policy: Arc::new(Barrier::new(2)),
        });
        store.route_domain_import_test_gate = Some(Arc::clone(&gate));
        let store = Arc::new(store);
        let import_store = Arc::clone(&store);
        let import_thread = std::thread::spawn(move || {
            import_store.import_route_domain_attestation_certificate(old_certificate, now)
        });

        gate.policy_read.wait();
        let rotate_store = Arc::clone(&store);
        let rotation_thread = std::thread::spawn(move || {
            rotate_store.configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &new_allowed,
                2,
                true,
            )
        });
        gate.continue_after_policy.wait();

        assert!(import_thread.join().unwrap().is_ok());
        rotation_thread.join().unwrap().unwrap();
        assert!(!store.route_domain_certificate_allows_multi_hop(&subject.public_key_bytes(), now,));
        assert!(store
            .export_route_domain_attestation_certificates(now)
            .is_empty());
    }
}
