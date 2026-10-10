// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/observation_certificate/tests.rs
// ============================================
//! # Tests: Directory observation certificates
//!
//! Unit tests for the directory observation certificates, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use clap::Parser;
use sha2::{Digest, Sha256};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::discovery::{
    directory_observation_witness_response_signing_bytes, encode_directory_observation_certificate,
    DirectoryObservationCertificateV1, DirectoryObservationCheckpointV1, DirectoryObservationTipV1,
    DirectoryObservationWitnessReceiptV1, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
};

use crate::{Cli, Commands, DirectoryReplicaCommands};

fn portable_observation_certificate_fixture() -> (Vec<u8>, [u8; 32], [u8; 32], Vec<[u8; 32]>) {
    let observer = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0x35; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        7,
        1_700_000_700,
        [0x36; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 41,
                tip_hash: [0x37; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 42,
                tip_hash: [0x38; 32],
            },
        ],
        [0x39; 32],
        &observer,
    )
    .unwrap();

    let receipt = |witness: &IdentityKeyPair, request_id: [u8; 16], response_timestamp: u64| {
        let checkpoint_hash = checkpoint.hash();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &checkpoint.chain_id,
            &request_id,
            &checkpoint.observer,
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            response_timestamp,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        DirectoryObservationWitnessReceiptV1 {
            chain_id: checkpoint.chain_id,
            request_id,
            observer: checkpoint.observer,
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        }
    };
    let receipt_b = receipt(&witness_b, [0x3a; 16], 1_700_000_702);
    let receipt_a = receipt(&witness_a, [0x3b; 16], 1_700_000_701);
    let certificate = DirectoryObservationCertificateV1::new_verified(
        checkpoint,
        2,
        vec![receipt_b, receipt_a],
        1_700_000_702,
    )
    .unwrap();
    let frame = encode_directory_observation_certificate(&certificate).unwrap();
    let frame_sha256 = Sha256::digest(&frame).into();
    (
        frame,
        frame_sha256,
        observer.public_key_bytes(),
        vec![witness_a.public_key_bytes(), witness_b.public_key_bytes()],
    )
}

#[test]
fn directory_observation_certificate_verifier_cli_requires_explicit_trust_policy() {
    let expected_sha256 = "a5".repeat(32);
    let expected_observer = "b6".repeat(32);
    let witness_a = "c7".repeat(32);
    let witness_b = "d8".repeat(32);
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "directory-replica",
        "verify-observation-certificate",
        "--input",
        "/tmp/observation.certificate",
        "--expected-sha256",
        &expected_sha256,
        "--expected-observer",
        &expected_observer,
        "--allowed-witness",
        &witness_a,
        "--allowed-witness",
        &witness_b,
        "--minimum-witnesses",
        "2",
        "--json",
    ])
    .unwrap();
    let Commands::DirectoryReplica(DirectoryReplicaCommands::VerifyObservationCertificate {
        input,
        expected_sha256: parsed_sha256,
        expected_observer: parsed_observer,
        allowed_witnesses,
        minimum_witnesses,
        json,
    }) = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(input, PathBuf::from("/tmp/observation.certificate"));
    assert_eq!(parsed_sha256, expected_sha256);
    assert_eq!(parsed_observer, expected_observer);
    assert_eq!(allowed_witnesses, vec![witness_a, witness_b]);
    assert_eq!(minimum_witnesses, 2);
    assert!(json);
}

#[test]
fn directory_observation_certificate_import_cli_requires_local_store_and_pins() {
    let expected_sha256 = "a6".repeat(32);
    let expected_observer = "b7".repeat(32);
    let witness = "c8".repeat(32);
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "directory-replica",
        "import-observation-certificate",
        "--input",
        "/tmp/third-party.certificate",
        "--expected-sha256",
        &expected_sha256,
        "--expected-observer",
        &expected_observer,
        "--allowed-witness",
        &witness,
        "--minimum-witnesses",
        "1",
        "--config",
        "/tmp/server.toml",
        "--json",
    ])
    .unwrap();
    let Commands::DirectoryReplica(DirectoryReplicaCommands::ImportObservationCertificate {
        input,
        expected_sha256: parsed_sha256,
        expected_observer: parsed_observer,
        allowed_witnesses,
        minimum_witnesses,
        config,
        json,
    }) = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(input, PathBuf::from("/tmp/third-party.certificate"));
    assert_eq!(parsed_sha256, expected_sha256);
    assert_eq!(parsed_observer, expected_observer);
    assert_eq!(allowed_witnesses, vec![witness]);
    assert_eq!(minimum_witnesses, 1);
    assert_eq!(config, PathBuf::from("/tmp/server.toml"));
    assert!(json);
}

#[test]
fn directory_observation_certificate_pull_cli_requires_source_pins_and_age() {
    let expected_observer = "b8".repeat(32);
    let witness = "c9".repeat(32);
    let cli = Cli::try_parse_from([
        "aeronyx-server",
        "directory-replica",
        "pull-observation-certificate",
        "--source-endpoint",
        "https://203.0.113.9:8422",
        "--expected-observer",
        &expected_observer,
        "--allowed-witness",
        &witness,
        "--minimum-witnesses",
        "1",
        "--max-age-seconds",
        "600",
        "--config",
        "/tmp/server.toml",
        "--json",
    ])
    .unwrap();
    let Commands::DirectoryReplica(DirectoryReplicaCommands::PullObservationCertificate {
        source_endpoint,
        expected_observer: parsed_observer,
        allowed_witnesses,
        minimum_witnesses,
        max_age_seconds,
        config,
        json,
    }) = cli.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(source_endpoint, "https://203.0.113.9:8422");
    assert_eq!(parsed_observer, expected_observer);
    assert_eq!(allowed_witnesses, vec![witness]);
    assert_eq!(minimum_witnesses, 1);
    assert_eq!(max_age_seconds, 600);
    assert_eq!(config, PathBuf::from("/tmp/server.toml"));
    assert!(json);
}

#[test]
fn directory_observation_certificate_verifier_is_bounded_and_fail_closed() {
    // [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] The server-side
    // adapter must preserve the core verifier's canonical and signature
    // checks rather than treating a matching transport digest as trust.
    let (frame, frame_sha256, observer, witnesses) = portable_observation_certificate_fixture();
    let allowed_witness_hex = witnesses.iter().map(hex::encode).collect::<Vec<_>>();
    let trust_policy =
        parse_observation_certificate_trust_policy(&hex::encode(observer), &allowed_witness_hex, 2)
            .unwrap();
    let report = verify_directory_observation_certificate_frame(
        &frame,
        &frame_sha256,
        &trust_policy,
        1_700_000_702,
    )
    .unwrap();
    assert_eq!(report.status, "verified");
    assert_eq!(report.checkpoint_sequence, 7);
    assert_eq!(report.trust_policy_status, "matched");
    assert_eq!(report.policy_minimum_witnesses, 2);
    assert_eq!(report.policy_allowed_witnesses, 2);
    assert_eq!(report.certificate_minimum_witnesses, 2);
    assert_eq!(report.witness_receipts, 2);
    assert_eq!(report.frame_bytes, frame.len());
    assert_eq!(report.observer_fingerprint.len(), 12);

    let mut wrong_sha256 = frame_sha256;
    wrong_sha256[0] ^= 0x80;
    assert!(verify_directory_observation_certificate_frame(
        &frame,
        &wrong_sha256,
        &trust_policy,
        1_700_000_702,
    )
    .is_err());

    let mut tampered = frame.clone();
    let final_byte = tampered.last_mut().unwrap();
    *final_byte ^= 0x01;
    let tampered_sha256 = Sha256::digest(&tampered).into();
    assert!(verify_directory_observation_certificate_frame(
        &tampered,
        &tampered_sha256,
        &trust_policy,
        1_700_000_702,
    )
    .is_err());

    let oversized = vec![0u8; MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES.saturating_add(1)];
    let oversized_sha256 = Sha256::digest(&oversized).into();
    assert!(verify_directory_observation_certificate_frame(
        &oversized,
        &oversized_sha256,
        &trust_policy,
        1_700_000_702,
    )
    .is_err());

    let wrong_observer_policy =
        parse_observation_certificate_trust_policy(&"e9".repeat(32), &allowed_witness_hex, 2)
            .unwrap();
    assert!(verify_directory_observation_certificate_frame(
        &frame,
        &frame_sha256,
        &wrong_observer_policy,
        1_700_000_702,
    )
    .is_err());

    let untrusted_witness_hex = vec![hex::encode(witnesses[0]), "ea".repeat(32)];
    let untrusted_witness_policy = parse_observation_certificate_trust_policy(
        &hex::encode(observer),
        &untrusted_witness_hex,
        1,
    )
    .unwrap();
    assert!(verify_directory_observation_certificate_frame(
        &frame,
        &frame_sha256,
        &untrusted_witness_policy,
        1_700_000_702,
    )
    .is_err());
    assert!(parse_observation_certificate_trust_policy(
        &hex::encode(observer),
        &[hex::encode(witnesses[0]), hex::encode(witnesses[0])],
        1,
    )
    .is_err());
}
