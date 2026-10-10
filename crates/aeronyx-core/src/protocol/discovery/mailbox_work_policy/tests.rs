// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/mailbox_work_policy/tests.rs
// ============================================
//! # Tests: the Anonymous Mailbox admission-work policy
//!
//! Unit tests for the Anonymous Mailbox admission-work policy, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use sha2::{Digest, Sha256};

use crate::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};

fn work_policy_descriptor(identity: &IdentityKeyPair) -> NodeDescriptor {
    NodeDescriptor::new(
        identity.public_key_bytes(),
        42,
        1_800_000_000,
        1_800_000_300,
        "1.2.3",
    )
    .with_protocol_features([NodeProtocolFeature::AnonymousMailboxV1])
}

fn replace_work_policy_token(mut descriptor: NodeDescriptor, replacement: &str) -> NodeDescriptor {
    let (release, metadata) = descriptor
        .software_version
        .split_once('+')
        .expect("policy fixture metadata");
    let retained = metadata
        .split('.')
        .filter(|identifier| {
            !identifier
                .get(..ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY.len())
                .is_some_and(|prefix| {
                    prefix.eq_ignore_ascii_case(ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY)
                })
        })
        .collect::<Vec<_>>();
    descriptor.software_version = format!("{release}+{}.{}", retained.join("."), replacement);
    descriptor
}

#[test]
fn anonymous_mailbox_work_policy_fixed_codec_and_descriptor_pin_are_frozen() {
    // [ANONYMOUS-MAILBOX-WORK-POLICY 2026-09-07 by Codex] A deterministic
    // identity freezes both nested policy bytes and pre-policy descriptor
    // signing bytes.
    let identity = IdentityKeyPair::from_bytes(&[0x31; 32]).expect("identity");
    let baseline = work_policy_descriptor(&identity);
    let baseline_signing_hash: [u8; 32] = Sha256::digest(baseline.signing_bytes().unwrap()).into();
    assert_eq!(
        hex::encode(baseline_signing_hash),
        "121526fba3dd4c32b747a0ea27a80f2d6b52127c635d0974e910adcc5967e222"
    );

    let descriptor = baseline
        .with_anonymous_mailbox_work_policy(12, &identity)
        .expect("policy");
    let signed = SignedNodeDescriptor::sign(descriptor, &identity).expect("outer signature");
    let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed).expect("pin");
    let policy = signed
        .anonymous_mailbox_work_policy_for_pin_at(&pin, 1_800_000_001)
        .expect("nested policy");
    assert_eq!(policy.target_node_id(), identity.public_key_bytes());
    assert_eq!(policy.descriptor_sequence(), 42);
    assert_eq!(policy.issued_at(), 1_800_000_000);
    assert_eq!(policy.expires_at(), 1_800_000_300);
    assert_eq!(policy.work_bits(), 12);
    assert_eq!(
        policy.encode_fixed().len(),
        ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1
    );
    assert_eq!(policy.semver_build_token().len(), 260);
    let encoded_hash: [u8; 32] = Sha256::digest(policy.encode_fixed()).into();
    assert_eq!(
        hex::encode(encoded_hash),
        "b142722deeb0b85b57631426b291e1bf415ac84e937521f555363aafc4895111"
    );

    let mut wrong_pin = pin;
    wrong_pin.descriptor_hash[0] ^= 1;
    assert_eq!(
        signed.anonymous_mailbox_work_policy_for_pin_at(&wrong_pin, 1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::ClaimsConflict)
    );
}

#[test]
fn anonymous_mailbox_work_policy_parser_and_signatures_fail_closed() {
    let identity = IdentityKeyPair::from_bytes(&[0x32; 32]).expect("identity");
    let base = work_policy_descriptor(&identity);
    let with_policy = base
        .clone()
        .with_anonymous_mailbox_work_policy(12, &identity)
        .expect("policy");
    let signed = SignedNodeDescriptor::sign(with_policy.clone(), &identity).unwrap();

    assert_eq!(
        SignedNodeDescriptor::sign(base, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::MissingPolicy)
    );

    let token = anonymous_mailbox_work_policy_tokens(&with_policy.software_version)
        .next()
        .unwrap()
        .to_string();
    let duplicate = append_semver_build_identifier(&with_policy.software_version, &token);
    let mut duplicate_descriptor = with_policy.clone();
    duplicate_descriptor.software_version = duplicate;
    assert_eq!(
        SignedNodeDescriptor::sign(duplicate_descriptor, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::ClaimsConflict)
    );

    let uppercase = replace_work_policy_token(with_policy.clone(), &token.to_uppercase());
    assert_eq!(
        SignedNodeDescriptor::sign(uppercase, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::MalformedPolicy)
    );

    let unknown =
        replace_work_policy_token(with_policy.clone(), &token.replacen("anmp1-", "anmp2-", 1));
    assert_eq!(
        SignedNodeDescriptor::sign(unknown, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::UnsupportedVersion)
    );

    for malformed in [token[..259].to_string(), format!("{token}0")] {
        let malformed = replace_work_policy_token(with_policy.clone(), &malformed);
        assert_eq!(
            SignedNodeDescriptor::sign(malformed, &identity)
                .unwrap()
                .anonymous_mailbox_work_policy_at(1_800_000_001),
            Err(AnonymousMailboxWorkPolicyError::MalformedPolicy)
        );
    }

    let mut mismatched_version =
        hex::decode(&token[ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1.len()..])
            .expect("policy hex");
    mismatched_version[4..6].copy_from_slice(&2u16.to_le_bytes());
    let mismatched_version = format!(
        "{ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1}{}",
        hex::encode(mismatched_version)
    );
    let mismatched_version = replace_work_policy_token(with_policy.clone(), &mismatched_version);
    assert_eq!(
        SignedNodeDescriptor::sign(mismatched_version, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::UnsupportedVersion)
    );

    let mut nested_tamper = token.clone();
    nested_tamper.replace_range(
        259..260,
        if nested_tamper.ends_with('0') {
            "1"
        } else {
            "0"
        },
    );
    let nested_tamper = replace_work_policy_token(with_policy, &nested_tamper);
    assert_eq!(
        SignedNodeDescriptor::sign(nested_tamper, &identity)
            .unwrap()
            .anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::SignatureRejected)
    );

    let mut outer_tamper = signed;
    outer_tamper.signature[0] ^= 1;
    assert_eq!(
        outer_tamper.anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::SignatureRejected)
    );
}

#[test]
fn anonymous_mailbox_work_policy_bounds_expiry_and_rotation_are_enforced() {
    let identity = IdentityKeyPair::from_bytes(&[0x33; 32]).expect("identity");
    for invalid in [0, 25] {
        assert_eq!(
            work_policy_descriptor(&identity)
                .with_anonymous_mailbox_work_policy(invalid, &identity),
            Err(AnonymousMailboxWorkPolicyError::MalformedPolicy)
        );
    }
    for valid in [1, 24] {
        let descriptor = work_policy_descriptor(&identity)
            .with_anonymous_mailbox_work_policy(valid, &identity)
            .expect("bounded policy");
        let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
        assert_eq!(
            signed
                .anonymous_mailbox_work_policy_at(1_800_000_001)
                .unwrap()
                .work_bits(),
            valid
        );
        assert_eq!(
            signed.anonymous_mailbox_work_policy_at(1_800_000_300),
            Err(AnonymousMailboxWorkPolicyError::NotCurrentlyValid)
        );
    }

    let first = work_policy_descriptor(&identity)
        .with_anonymous_mailbox_work_policy(12, &identity)
        .unwrap();
    let mut rotated = work_policy_descriptor(&identity);
    rotated.sequence += 1;
    rotated.issued_at += 1;
    rotated.expires_at += 1;
    let rotated = rotated
        .with_anonymous_mailbox_work_policy(13, &identity)
        .unwrap();
    let first = SignedNodeDescriptor::sign(first, &identity).unwrap();
    let rotated = SignedNodeDescriptor::sign(rotated, &identity).unwrap();
    assert_ne!(
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first).unwrap(),
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&rotated).unwrap()
    );
    assert_ne!(
        first.descriptor.software_version,
        rotated.descriptor.software_version
    );
}
