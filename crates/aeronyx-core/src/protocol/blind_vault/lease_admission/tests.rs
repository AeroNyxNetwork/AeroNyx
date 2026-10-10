// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_admission/tests.rs
// ============================================
//! # Tests: anonymous lease creation and admission
//!
//! Unit tests for anonymous lease creation and admission, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::frame::FRAME_KIND_BLIND_LEASE_ADMISSION;
use crate::protocol::blind_vault::test_support::{
    admin_key, admission_issuer_key, lease_key, MAX_TTL_MS, NOW_MS,
};

const MAX_TICKET_TTL_MS: u64 = 24 * 60 * 60 * 1_000;

fn signed_lease() -> BlindVaultLeaseCreateRequest {
    let mut lease = BlindVaultLeaseCreateRequest::new(
        [1; 32],
        [8; 16],
        lease_key().public_key_bytes(),
        admin_key().public_key_bytes(),
        sha256(&[13; 32]),
        NOW_MS + 7 * 24 * 60 * 60 * 1_000,
    );
    lease.sign(&admin_key()).expect("matching admin key");
    lease
}

fn signed_admission() -> BlindVaultAdmissionTicket {
    let issuer = admission_issuer_key();
    let mut ticket = BlindVaultAdmissionTicket::new(
        [17; 32],
        issuer.public_key_bytes(),
        NOW_MS - 1_000,
        NOW_MS + 60 * 60 * 1_000,
        14 * 24 * 60 * 60 * 1_000,
    );
    ticket.sign(&issuer).expect("matching admission issuer");
    ticket
}

#[test]
fn anonymous_lease_is_self_authenticating_and_round_trips() {
    let lease = signed_lease();
    lease
        .validate_and_verify(NOW_MS, MAX_TTL_MS)
        .expect("valid anonymous lease");

    let encoded = encode_blind_vault_frame(&BlindVaultFrame::LeaseCreate(lease.clone()))
        .expect("encode lease");
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode lease"),
        BlindVaultFrame::LeaseCreate(lease)
    );
}

#[test]
fn bearer_admission_validates_and_round_trips_without_identity_metadata() {
    let request = BlindVaultLeaseAdmissionRequest {
        admission: signed_admission(),
        lease: signed_lease(),
    };
    request
        .validate_and_verify(
            NOW_MS,
            MAX_TTL_MS,
            MAX_TICKET_TTL_MS,
            &admission_issuer_key().public_key(),
        )
        .expect("valid admission and lease");

    let encoded = encode_blind_vault_frame(&BlindVaultFrame::LeaseAdmission(request.clone()))
        .expect("encode admission");
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode admission"),
        BlindVaultFrame::LeaseAdmission(request)
    );
}

// [BLIND-VAULT-BLIND-ADMISSION 2026-07-23 by Codex] Core validates only
// bounded wire shape; the server crate performs RFC 9474 verification with
// an operator-pinned RSA epoch key.
#[test]
fn blind_admission_shape_is_domain_separated_and_round_trips_additively() {
    let admission = BlindVaultBlindAdmissionToken::new(
        [41; 32],
        [42; 32],
        [43; 32],
        vec![44; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
    );
    admission.validate_shape().expect("bounded blind token");
    assert!(admission
        .message_bytes()
        .starts_with(BLIND_ADMISSION_MESSAGE_DOMAIN));
    assert_ne!(admission.spend_id(), admission.token_id);

    let request = BlindVaultBlindLeaseAdmissionRequest {
        admission,
        lease: signed_lease(),
    };
    let encoded = encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAdmission(request.clone()))
        .expect("encode blind admission");
    assert_eq!(encoded[6], FRAME_KIND_BLIND_LEASE_ADMISSION);
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode blind admission"),
        BlindVaultFrame::BlindLeaseAdmission(request)
    );
}

#[test]
fn blind_admission_rejects_zero_randomizer_and_unbounded_signature() {
    let zero_randomizer = BlindVaultBlindAdmissionToken::new(
        [45; 32],
        [46; 32],
        [0; 32],
        vec![47; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
    );
    assert!(matches!(
        zero_randomizer.validate_shape(),
        Err(BlindVaultError::ZeroIdentifier(
            "blind_admission_message_randomizer"
        ))
    ));

    let oversized = BlindVaultBlindAdmissionToken::new(
        [45; 32],
        [46; 32],
        [48; 32],
        vec![49; MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES + 1],
    );
    assert_eq!(
        oversized.validate_shape(),
        Err(BlindVaultError::InvalidBlindAdmissionSignatureLength {
            actual: MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES + 1,
        })
    );
}

#[test]
fn admission_rejects_future_window_and_overlong_lease() {
    let issuer = admission_issuer_key();
    let mut future = signed_admission();
    future.not_before_ms = NOW_MS + 1;
    future.sign(&issuer).expect("sign future ticket");
    assert_eq!(
        future.validate_and_verify(NOW_MS, MAX_TICKET_TTL_MS, &issuer.public_key()),
        Err(BlindVaultError::AdmissionNotYetValid)
    );

    let mut narrow = signed_admission();
    narrow.maximum_lease_ttl_ms = 60 * 60 * 1_000;
    narrow.sign(&issuer).expect("sign narrow ticket");
    let request = BlindVaultLeaseAdmissionRequest {
        admission: narrow,
        lease: signed_lease(),
    };
    assert_eq!(
        request.validate_and_verify(NOW_MS, MAX_TTL_MS, MAX_TICKET_TTL_MS, &issuer.public_key(),),
        Err(BlindVaultError::LifetimeTooLong)
    );
}

#[test]
fn lease_rejects_write_and_admin_key_reuse() {
    let key = admin_key();
    let mut lease = BlindVaultLeaseCreateRequest::new(
        [1; 32],
        [8; 16],
        key.public_key_bytes(),
        key.public_key_bytes(),
        sha256(&[13; 32]),
        NOW_MS + 7 * 24 * 60 * 60 * 1_000,
    );
    lease.sign(&key).expect("matching admin key");
    assert_eq!(
        lease.validate_and_verify(NOW_MS, MAX_TTL_MS),
        Err(BlindVaultError::LeaseKeyReuse)
    );
}
