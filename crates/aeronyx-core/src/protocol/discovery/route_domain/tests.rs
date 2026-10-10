// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/route_domain/tests.rs
// ============================================
//! # Tests: portable route-domain attestations
//!
//! Unit tests for portable route-domain attestations, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

#[test]
fn route_domain_certificate_is_canonical_bounded_and_policy_verified() {
    // [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] The fixture proves
    // exact pinned signatures over an opaque token. It intentionally makes
    // no ASN, operator-independence, consensus, or Sybil-resistance claim.
    let subject = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
    let attestor_a = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
    let attestor_b = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
    let route_domain = [0xa4; 16];
    let issued_at = 1_700_001_000;
    let expires_at = issued_at + 3_600;
    let statement_a = RouteDomainAttestationV1::new_signed(
        subject.public_key_bytes(),
        route_domain,
        issued_at,
        expires_at,
        &attestor_a,
    )
    .unwrap();
    let statement_b = RouteDomainAttestationV1::new_signed(
        subject.public_key_bytes(),
        route_domain,
        issued_at + 1,
        expires_at,
        &attestor_b,
    )
    .unwrap();
    let certificate = RouteDomainAttestationCertificateV1::new_verified(
        subject.public_key_bytes(),
        route_domain,
        vec![statement_b, statement_a],
        issued_at + 2,
    )
    .unwrap();

    assert!(
        certificate.attestations[0].attestor_node_id < certificate.attestations[1].attestor_node_id
    );
    assert_eq!(
        certificate
            .verify_with_policy_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &[attestor_b.public_key_bytes(), attestor_a.public_key_bytes(),],
                2,
                issued_at + 2,
            )
            .unwrap(),
        2
    );
    let encoded = encode_route_domain_attestation_certificate(&certificate).unwrap();
    assert_eq!(
        encoded.first().copied(),
        Some(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC)
    );
    assert!(encoded.len() <= MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES);
    let decoded = decode_route_domain_attestation_certificate(&encoded).unwrap();
    assert_eq!(decoded, certificate);
    assert_eq!(decoded.hash(), certificate.hash());

    let mut trailing = encoded;
    trailing.push(0);
    assert!(decode_route_domain_attestation_certificate(&trailing).is_err());
    let mut oversized = vec![0u8; MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES + 1];
    oversized[0] = ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC;
    assert!(decode_route_domain_attestation_certificate(&oversized).is_err());
}

#[test]
fn route_domain_certificate_rejects_expiry_tamper_duplicates_and_untrusted_quorum() {
    let subject = IdentityKeyPair::from_bytes(&[0xb1; 32]).unwrap();
    let attestor_a = IdentityKeyPair::from_bytes(&[0xb2; 32]).unwrap();
    let attestor_b = IdentityKeyPair::from_bytes(&[0xb3; 32]).unwrap();
    let untrusted = IdentityKeyPair::from_bytes(&[0xb4; 32]).unwrap();
    let route_domain = [0xb5; 16];
    let issued_at = 1_700_002_000;
    let expires_at = issued_at + 600;
    let statement_a = RouteDomainAttestationV1::new_signed(
        subject.public_key_bytes(),
        route_domain,
        issued_at,
        expires_at,
        &attestor_a,
    )
    .unwrap();
    let statement_b = RouteDomainAttestationV1::new_signed(
        subject.public_key_bytes(),
        route_domain,
        issued_at,
        expires_at,
        &attestor_b,
    )
    .unwrap();
    let certificate = RouteDomainAttestationCertificateV1::new_verified(
        subject.public_key_bytes(),
        route_domain,
        vec![statement_a, statement_b],
        issued_at + 1,
    )
    .unwrap();

    assert_eq!(
        statement_a.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, expires_at),
        Err(RouteDomainAttestationValidationError::Expired)
    );
    assert_eq!(
        RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at,
            issued_at + MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1 + 1,
            &attestor_a,
        ),
        Err(RouteDomainAttestationValidationError::InvalidTimestamp)
    );
    assert_eq!(
        RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at,
            expires_at,
            &subject,
        ),
        Err(RouteDomainAttestationValidationError::InvalidAttestor)
    );

    let duplicate = RouteDomainAttestationCertificateV1 {
        protocol_version: ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1,
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        subject_node_id: subject.public_key_bytes(),
        route_domain,
        attestations: vec![statement_a, statement_a],
    };
    assert_eq!(
        duplicate.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
        Err(RouteDomainAttestationCertificateValidationError::DuplicateAttestor)
    );

    let mut tampered = certificate.clone();
    tampered.attestations[0].signature[0] ^= 1;
    assert_eq!(
        tampered.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
        Err(RouteDomainAttestationCertificateValidationError::InvalidAttestation)
    );
    let mut rebound = certificate.clone();
    rebound.route_domain = [0xb6; 16];
    assert_eq!(
        rebound.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
        Err(RouteDomainAttestationCertificateValidationError::InvalidAttestationContract)
    );

    assert_eq!(
        certificate.verify_with_policy_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &[attestor_a.public_key_bytes(), untrusted.public_key_bytes(),],
            2,
            issued_at + 1,
        ),
        Err(RouteDomainAttestationCertificateValidationError::InsufficientTrustedAttestations)
    );
    assert_eq!(
        certificate.verify_with_policy_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &[attestor_a.public_key_bytes(), attestor_a.public_key_bytes(),],
            1,
            issued_at + 1,
        ),
        Err(RouteDomainAttestationCertificateValidationError::InvalidPolicy)
    );
}
