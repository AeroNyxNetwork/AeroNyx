// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn lease_provisioning_is_idempotent_but_not_mutable() {
    let fixture = Fixture::new(10, 1024 * 1024);
    fixture.provision();
    assert_eq!(
        fixture
            .service
            .provision_lease_with_admission(
                &fixture.admission_request(fixture.lease.clone()),
                NOW_MS,
            )
            .expect("idempotent lease"),
        BlindVaultLeaseProvisionOutcome::Existing
    );

    let mut conflict = fixture.lease.clone();
    conflict.expires_at_ms += 1;
    conflict.sign(&fixture.admin_key).expect("sign conflict");
    assert!(matches!(
        fixture
            .service
            .provision_lease_with_admission(&fixture.admission_request(conflict), NOW_MS,),
        Err(BlindVaultServiceError::LeaseConflict)
    ));
}

#[test]
fn admission_ticket_is_spent_atomically_and_retry_is_idempotent() {
    let directory = tempfile::tempdir().expect("temp directory");
    let issuer = IdentityKeyPair::from_bytes(&[21; 32]).expect("issuer key");
    let config = BlindVaultConfig {
        enabled: true,
        public_api_enabled: true,
        admission_issuer_public_keys: vec![hex::encode(issuer.public_key_bytes())],
        db_path: directory.path().join("vault.db").display().to_string(),
        ..BlindVaultConfig::default()
    };
    let node_key = IdentityKeyPair::from_bytes(&[22; 32]).expect("node key");
    let service = BlindVaultService::new(config, node_key).expect("service");

    let write_key = IdentityKeyPair::from_bytes(&[23; 32]).expect("write key");
    let admin_key = IdentityKeyPair::from_bytes(&[24; 32]).expect("admin key");
    let mut lease = BlindVaultLeaseCreateRequest::new(
        [25; 32],
        [26; 16],
        write_key.public_key_bytes(),
        admin_key.public_key_bytes(),
        Sha256::digest([27; 32]).into(),
        NOW_MS + 7 * 24 * 60 * 60 * 1_000,
    );
    lease.sign(&admin_key).expect("sign lease");
    let mut admission = BlindVaultAdmissionTicket::new(
        [28; 32],
        issuer.public_key_bytes(),
        NOW_MS - 1_000,
        NOW_MS + 60 * 60 * 1_000,
        14 * 24 * 60 * 60 * 1_000,
    );
    admission.sign(&issuer).expect("sign admission");
    let request = BlindVaultLeaseAdmissionRequest { admission, lease };

    assert_eq!(
        service
            .provision_lease_with_admission(&request, NOW_MS)
            .expect("first admission"),
        BlindVaultLeaseProvisionOutcome::Created
    );
    assert_eq!(
        service
            .provision_lease_with_admission(&request, NOW_MS + 1)
            .expect("idempotent admission retry"),
        BlindVaultLeaseProvisionOutcome::Existing
    );

    let mut second_lease = request.lease.clone();
    second_lease.lease_id = [29; 32];
    second_lease.request_id = [30; 16];
    second_lease.sign(&admin_key).expect("sign second lease");
    let replay = BlindVaultLeaseAdmissionRequest {
        admission: request.admission.clone(),
        lease: second_lease,
    };
    assert!(matches!(
        service.provision_lease_with_admission(&replay, NOW_MS + 2),
        Err(BlindVaultServiceError::AdmissionSpent)
    ));
    assert_eq!(
        service
            .status(NOW_MS + 3)
            .expect("status")
            .retained_admission_spends,
        1
    );

    let cleanup = service
        .run_cleanup(NOW_MS + 60 * 60 * 1_000 + 1)
        .expect("cleanup spent ticket");
    assert_eq!(cleanup.admission_spends_removed, 1);
}

// [BLIND-VAULT-BLIND-REDEMPTION 2026-07-23 by Codex] Simulate the full
// RFC 9474 lifecycle. The storage service receives only the finalized
// token and cannot recover or correlate the issuer's blind transcript.
#[test]
fn blind_admission_is_unlinkable_one_time_and_policy_bounded() {
    let key_pair =
        KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
    let public_der = key_pair.pk.to_der().expect("public key DER");
    let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
    let directory = tempfile::tempdir().expect("temp directory");
    let config = BlindVaultConfig {
        enabled: true,
        public_api_enabled: true,
        blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
            public_key_der_base64: BASE64.encode(&public_der),
            not_before_unix_secs: NOW_MS / 1_000 - 60,
            expires_at_unix_secs: NOW_MS / 1_000 + 24 * 60 * 60,
            max_lease_ttl_secs: 14 * 24 * 60 * 60,
        }],
        db_path: directory.path().join("vault.db").display().to_string(),
        ..BlindVaultConfig::default()
    };
    let node_key = IdentityKeyPair::from_bytes(&[51; 32]).expect("node key");
    let service = BlindVaultService::new(config, node_key).expect("service");
    let advertised_epochs = service
        .blind_admission_issuer_epochs(NOW_MS)
        .expect("issuer epochs");
    assert_eq!(advertised_epochs.len(), 1);
    assert_eq!(advertised_epochs[0].issuer_key_id, issuer_key_id);
    assert_eq!(advertised_epochs[0].public_key_der, public_der);
    assert!(service
        .blind_admission_issuer_epochs(NOW_MS + 24 * 60 * 60 * 1_000)
        .expect("expired issuer filter")
        .is_empty());

    let write_key = IdentityKeyPair::from_bytes(&[52; 32]).expect("write key");
    let admin_key = IdentityKeyPair::from_bytes(&[53; 32]).expect("admin key");
    let mut lease = BlindVaultLeaseCreateRequest::new(
        [54; 32],
        [55; 16],
        write_key.public_key_bytes(),
        admin_key.public_key_bytes(),
        Sha256::digest([56; 32]).into(),
        NOW_MS + 7 * 24 * 60 * 60 * 1_000,
    );
    lease.sign(&admin_key).expect("sign lease");

    let token_id = [57; 32];
    let unsigned =
        BlindVaultBlindAdmissionToken::new(issuer_key_id, token_id, [1; 32], vec![0; 256]);
    let message = unsigned.message_bytes();
    let blinding = key_pair
        .pk
        .blind(&mut DefaultRng, &message)
        .expect("blind admission message");
    let blind_signature = key_pair
        .sk
        .blind_sign(&blinding.blind_message)
        .expect("blind sign admission");
    let signature = key_pair
        .pk
        .finalize(&blind_signature, &blinding, &message)
        .expect("finalize admission");
    let randomizer = blinding.msg_randomizer.expect("randomized-message mode").0;
    let admission =
        BlindVaultBlindAdmissionToken::new(issuer_key_id, token_id, randomizer, signature.0);
    let request = BlindVaultBlindLeaseAdmissionRequest { admission, lease };

    assert_eq!(
        service
            .provision_lease_with_blind_admission(&request, NOW_MS)
            .expect("first blind redemption"),
        BlindVaultLeaseProvisionOutcome::Created
    );
    assert_eq!(
        service
            .provision_lease_with_blind_admission(&request, NOW_MS + 1)
            .expect("idempotent blind retry"),
        BlindVaultLeaseProvisionOutcome::Existing
    );

    let mut second_lease = request.lease.clone();
    second_lease.lease_id = [58; 32];
    second_lease.request_id = [59; 16];
    second_lease.sign(&admin_key).expect("sign second lease");
    let replay = BlindVaultBlindLeaseAdmissionRequest {
        admission: request.admission.clone(),
        lease: second_lease,
    };
    assert!(matches!(
        service.provision_lease_with_blind_admission(&replay, NOW_MS + 2),
        Err(BlindVaultServiceError::AdmissionSpent)
    ));

    let mut forged = request.clone();
    forged.admission.token_id[0] ^= 1;
    assert!(matches!(
        service.provision_lease_with_blind_admission(&forged, NOW_MS + 3),
        Err(BlindVaultServiceError::AdmissionProofRejected)
    ));
}

// [BLIND-VAULT-ISSUER-RUNTIME 2026-07-23 by Codex] Exercise rotation as a
// durable state machine: exact retries are idempotent, generation reuse and
// rollback fail closed, and still-valid verifier keys cannot disappear.
#[test]
fn blind_issuer_rotation_is_monotonic_continuous_and_restart_safe() {
    let old_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("old RSA key");
    let new_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("new RSA key");
    let future_key =
        KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("future RSA key");
    let max_lease_ttl_ms = 14 * 24 * 60 * 60 * 1_000;
    let old_epoch = blind_issuer_epoch(
        &old_key,
        NOW_MS - 60_000,
        NOW_MS + 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let new_epoch = blind_issuer_epoch(
        &new_key,
        NOW_MS - 60_000,
        NOW_MS + 2 * 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let future_epoch = blind_issuer_epoch(
        &future_key,
        NOW_MS + 60_000,
        NOW_MS + 3 * 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let directory = tempfile::tempdir().expect("temp directory");
    let config = BlindVaultConfig {
        enabled: true,
        public_api_enabled: true,
        blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
            public_key_der_base64: BASE64.encode(&old_epoch.public_key_der),
            not_before_unix_secs: old_epoch.not_before_ms / 1_000,
            expires_at_unix_secs: old_epoch.expires_at_ms / 1_000,
            max_lease_ttl_secs: old_epoch.max_lease_ttl_ms / 1_000,
        }],
        db_path: directory.path().join("vault.db").display().to_string(),
        ..BlindVaultConfig::default()
    };
    let node_seed = [61; 32];
    let service = BlindVaultService::new(
        config.clone(),
        IdentityKeyPair::from_bytes(&node_seed).expect("node key"),
    )
    .expect("service");
    assert_eq!(
        service.blind_admission_issuer_runtime_status(NOW_MS),
        BlindVaultIssuerRuntimeStatus {
            generation: 0,
            updated_at_ms: 0,
            epoch_count: 1,
            active_epoch_count: 1,
        }
    );

    let rotated = vec![old_epoch.clone(), new_epoch.clone()];
    assert_eq!(
        service
            .install_blind_admission_issuer_epochs(1, rotated.clone(), NOW_MS)
            .expect("install generation one"),
        BlindVaultIssuerInstallOutcome::Installed { generation: 1 }
    );
    assert_eq!(
        service
            .install_blind_admission_issuer_epochs(1, rotated.clone(), NOW_MS)
            .expect("retry generation one"),
        BlindVaultIssuerInstallOutcome::Unchanged { generation: 1 }
    );
    assert!(matches!(
        service.install_blind_admission_issuer_epochs(0, rotated.clone(), NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryRollback)
    ));
    assert!(matches!(
        service.install_blind_admission_issuer_epochs(2, rotated.clone(), NOW_MS - 1),
        Err(BlindVaultServiceError::IssuerDirectoryRollback)
    ));

    let mut conflicting = rotated.clone();
    conflicting[1].max_lease_ttl_ms -= 1_000;
    assert!(matches!(
        service.install_blind_admission_issuer_epochs(1, conflicting, NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryGenerationConflict)
    ));
    assert!(matches!(
        service.install_blind_admission_issuer_epochs(2, vec![new_epoch.clone()], NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryContinuity)
    ));
    assert!(matches!(
        service.install_blind_admission_issuer_epochs(2, vec![future_epoch], NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryNoActiveEpoch)
    ));
    drop(service);

    let restarted = BlindVaultService::new(
        config,
        IdentityKeyPair::from_bytes(&node_seed).expect("restart node key"),
    )
    .expect("restart service");
    assert_eq!(
        restarted.blind_admission_issuer_runtime_status(NOW_MS),
        BlindVaultIssuerRuntimeStatus {
            generation: 1,
            updated_at_ms: NOW_MS,
            epoch_count: 2,
            active_epoch_count: 2,
        }
    );
    assert_eq!(
        restarted
            .blind_admission_issuer_epochs(NOW_MS)
            .expect("restarted epochs"),
        {
            let mut expected = rotated;
            expected.sort_by_key(|epoch| epoch.issuer_key_id);
            expected
        }
    );

    let write_key = IdentityKeyPair::from_bytes(&[62; 32]).expect("write key");
    let admin_key = IdentityKeyPair::from_bytes(&[63; 32]).expect("admin key");
    let mut lease = BlindVaultLeaseCreateRequest::new(
        [64; 32],
        [65; 16],
        write_key.public_key_bytes(),
        admin_key.public_key_bytes(),
        Sha256::digest([66; 32]).into(),
        NOW_MS + 7 * 24 * 60 * 60 * 1_000,
    );
    lease.sign(&admin_key).expect("sign lease");
    let unsigned = BlindVaultBlindAdmissionToken::new(
        new_epoch.issuer_key_id,
        [67; 32],
        [1; 32],
        vec![0; 256],
    );
    let message = unsigned.message_bytes();
    let blinding = new_key
        .pk
        .blind(&mut DefaultRng, &message)
        .expect("blind rotated admission");
    let blind_signature = new_key
        .sk
        .blind_sign(&blinding.blind_message)
        .expect("sign rotated admission");
    let signature = new_key
        .pk
        .finalize(&blind_signature, &blinding, &message)
        .expect("finalize rotated admission");
    let admission = BlindVaultBlindAdmissionToken::new(
        new_epoch.issuer_key_id,
        [67; 32],
        blinding.msg_randomizer.expect("message randomizer").0,
        signature.0,
    );
    assert_eq!(
        restarted
            .provision_lease_with_blind_admission(
                &BlindVaultBlindLeaseAdmissionRequest { admission, lease },
                NOW_MS + 1,
            )
            .expect("redeem after restart"),
        BlindVaultLeaseProvisionOutcome::Created
    );
}

// [BLIND-VAULT-ISSUER-AUTHORITY 2026-07-23 by Codex] The public update
// boundary verifies a separately pinned Ed25519 authority before the
// existing atomic generation state machine can observe candidate epochs.
#[test]
fn signed_issuer_update_rejects_forgery_staleness_and_unknown_authority() {
    let authority = IdentityKeyPair::from_bytes(&[69; 32]).expect("authority key");
    let unpinned_authority =
        IdentityKeyPair::from_bytes(&[70; 32]).expect("unpinned authority key");
    let old_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("old RSA key");
    let new_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("new RSA key");
    let max_lease_ttl_ms = 14 * 24 * 60 * 60 * 1_000;
    let old_epoch = blind_issuer_epoch(
        &old_key,
        NOW_MS - 60_000,
        NOW_MS + 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let new_epoch = blind_issuer_epoch(
        &new_key,
        NOW_MS - 60_000,
        NOW_MS + 2 * 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let directory = tempfile::tempdir().expect("temp directory");
    let service = BlindVaultService::new(
        BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            blind_issuer_update_authority_public_keys: vec![hex::encode(
                authority.public_key_bytes(),
            )],
            blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
                public_key_der_base64: BASE64.encode(&old_epoch.public_key_der),
                not_before_unix_secs: old_epoch.not_before_ms / 1_000,
                expires_at_unix_secs: old_epoch.expires_at_ms / 1_000,
                max_lease_ttl_secs: old_epoch.max_lease_ttl_ms / 1_000,
            }],
            db_path: directory.path().join("vault.db").display().to_string(),
            ..BlindVaultConfig::default()
        },
        IdentityKeyPair::from_bytes(&[71; 32]).expect("node key"),
    )
    .expect("service");

    let mut epochs = vec![old_epoch, new_epoch];
    epochs.sort_by_key(|epoch| epoch.issuer_key_id);
    let mut update =
        BlindVaultBlindIssuerUpdate::new(1, NOW_MS, authority.public_key_bytes(), epochs.clone());
    update.sign(&authority).expect("sign update");
    assert_eq!(
        service
            .install_signed_blind_admission_issuer_update(&update, NOW_MS)
            .expect("install signed update"),
        BlindVaultIssuerInstallOutcome::Installed { generation: 1 }
    );
    assert_eq!(
        service
            .install_signed_blind_admission_issuer_update(&update, NOW_MS)
            .expect("idempotent signed replay"),
        BlindVaultIssuerInstallOutcome::Unchanged { generation: 1 }
    );

    let mut forged = update.clone();
    forged.generation = 2;
    assert!(matches!(
        service.install_signed_blind_admission_issuer_update(&forged, NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryUpdateRejected)
    ));
    assert!(matches!(
        service.install_signed_blind_admission_issuer_update(&update, NOW_MS + 5 * 60 * 1_000 + 1,),
        Err(BlindVaultServiceError::IssuerDirectoryUpdateRejected)
    ));

    let mut unpinned =
        BlindVaultBlindIssuerUpdate::new(2, NOW_MS, unpinned_authority.public_key_bytes(), epochs);
    unpinned
        .sign(&unpinned_authority)
        .expect("sign unpinned update");
    assert!(matches!(
        service.install_signed_blind_admission_issuer_update(&unpinned, NOW_MS),
        Err(BlindVaultServiceError::IssuerDirectoryAuthorityRejected)
    ));
    assert_eq!(
        service
            .blind_admission_issuer_runtime_status(NOW_MS)
            .generation,
        1
    );
}

#[test]
fn blind_issuer_readers_never_observe_a_partial_generation() {
    let old_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("old RSA key");
    let new_key = KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("new RSA key");
    let max_lease_ttl_ms = 14 * 24 * 60 * 60 * 1_000;
    let old_epoch = blind_issuer_epoch(
        &old_key,
        NOW_MS - 60_000,
        NOW_MS + 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let new_epoch = blind_issuer_epoch(
        &new_key,
        NOW_MS - 60_000,
        NOW_MS + 2 * 24 * 60 * 60 * 1_000,
        max_lease_ttl_ms,
    );
    let directory = tempfile::tempdir().expect("temp directory");
    let service = Arc::new(
        BlindVaultService::new(
            BlindVaultConfig {
                enabled: true,
                public_api_enabled: true,
                blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
                    public_key_der_base64: BASE64.encode(&old_epoch.public_key_der),
                    not_before_unix_secs: old_epoch.not_before_ms / 1_000,
                    expires_at_unix_secs: old_epoch.expires_at_ms / 1_000,
                    max_lease_ttl_secs: old_epoch.max_lease_ttl_ms / 1_000,
                }],
                db_path: directory.path().join("vault.db").display().to_string(),
                ..BlindVaultConfig::default()
            },
            IdentityKeyPair::from_bytes(&[68; 32]).expect("node key"),
        )
        .expect("service"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(5));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let service = Arc::clone(&service);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..256 {
                    let epochs = service
                        .blind_admission_issuer_epochs(NOW_MS)
                        .expect("read issuer generation");
                    assert!(epochs.len() == 1 || epochs.len() == 2);
                    assert!(epochs
                        .windows(2)
                        .all(|pair| pair[0].issuer_key_id < pair[1].issuer_key_id));
                    std::thread::yield_now();
                }
            });
        }
        barrier.wait();
        assert_eq!(
            service
                .install_blind_admission_issuer_epochs(1, vec![old_epoch, new_epoch], NOW_MS,)
                .expect("install generation"),
            BlindVaultIssuerInstallOutcome::Installed { generation: 1 }
        );
    });
    assert_eq!(
        service
            .blind_admission_issuer_runtime_status(NOW_MS)
            .generation,
        1
    );
}
