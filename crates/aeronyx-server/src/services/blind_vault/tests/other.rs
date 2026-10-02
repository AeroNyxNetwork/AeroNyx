// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn replica_authorization_canonical_bytes_are_frozen() {
    let authorization = BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
        REPLICA_JOB_AUTHORIZATION_VERSION_V1,
        [40; 16],
        [1; 32],
        [42; 32],
        [43; 32],
        [5; 32],
        [44; 32],
        0x0102_0304_0506_0708,
        0x1112_1314_1516_1718,
        [45; 64],
    )
    .expect("V1 golden authorization");
    // [BLIND-VAULT-REPLICA-CLAIMS 2026-09-01 by Codex] This complete,
    // independent byte vector freezes the M12A signing domain, version,
    // field order, widths, and integer endianness. Future optional claims
    // require a new version; they must not silently alter this transcript.
    let expected = hex::decode(concat!(
        "4165726f4e79782d426c696e645661756c742d5265706c6963614a6f62417574686f72697a6174696f6e2d7631",
        "0001",
        "28282828282828282828282828282828",
        "0101010101010101010101010101010101010101010101010101010101010101",
        "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a",
        "2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b",
        "0505050505050505050505050505050505050505050505050505050505050505",
        "2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c",
        "0102030405060708",
        "1112131415161718",
    ))
    .expect("literal V1 golden bytes");

    let canonical = authorization.signing_bytes();
    assert_eq!(canonical.len(), 239);
    assert_eq!(canonical, expected, "V1 signing bytes must remain stable");
    assert_eq!(authorization.claims().job_id(), [40; 16]);
    assert_eq!(authorization.claims().target_node_id(), [5; 32]);
    assert_eq!(authorization.claims().target_bundle_commitment(), [44; 32]);
    assert_ne!(authorization.claims().commitment(), [0; 32]);
}

#[test]
fn replica_authorization_wire_boundary_rejects_unknown_versions() {
    let decode = |version| {
        BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
            version,
            [40; 16],
            [1; 32],
            [42; 32],
            [43; 32],
            [5; 32],
            [44; 32],
            NOW_MS,
            NOW_MS + 1,
            [45; 64],
        )
    };

    assert!(decode(REPLICA_JOB_AUTHORIZATION_VERSION_V1).is_ok());
    for version in [0, 2, u16::MAX] {
        assert!(matches!(
            decode(version),
            Err(BlindVaultReplicaJobAuthorizationError::Rejected)
        ));
    }
}

#[test]
fn replica_authorization_private_store_round_trip_is_exact() {
    let authorization = BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
        REPLICA_JOB_AUTHORIZATION_VERSION_V1,
        [46; 16],
        [47; 32],
        [48; 32],
        [49; 32],
        [5; 32],
        [50; 32],
        NOW_MS,
        NOW_MS + 1,
        [51; 64],
    )
    .expect("authorization");
    let canonical = authorization.canonical_authorization_bytes();
    let restored =
        BlindVaultReplicaJobAuthorizationV1::from_canonical_authorization_bytes(&canonical)
            .expect("restore exact canonical proof");
    assert_eq!(restored.canonical_authorization_bytes(), canonical);

    let mut wrong_version = canonical.clone();
    let version_offset = REPLICA_JOB_AUTHORIZATION_DOMAIN.len();
    wrong_version[version_offset..version_offset + 2].copy_from_slice(&2_u16.to_be_bytes());
    let mut wrong_domain = canonical.clone();
    wrong_domain[0] ^= 1;
    let mut trailing = canonical.clone();
    trailing.push(0);
    for invalid in [
        &canonical[..canonical.len() - 1],
        wrong_version.as_slice(),
        wrong_domain.as_slice(),
        trailing.as_slice(),
    ] {
        assert!(matches!(
            BlindVaultReplicaJobAuthorizationV1::from_canonical_authorization_bytes(invalid),
            Err(BlindVaultReplicaJobAuthorizationError::Rejected)
        ));
    }
}

#[test]
fn replica_authorization_rejects_node_identity_admin_collision_without_mutation() {
    let fixture = Fixture::new_with_admin_and_node_seeds(10, 1024 * 1024, [50; 32], [50; 32]);
    fixture.provision();
    let put = fixture.store_object(51, 52);
    let authorization = fixture.replica_authorization(&put);
    assert_eq!(
        fixture.admin_key.public_key_bytes(),
        fixture.service.node_identity.public_key_bytes()
    );
    IdentityPublicKey::from_bytes(&fixture.service.node_identity.public_key_bytes())
        .expect("node verifier")
        .verify(&authorization.signing_bytes(), &authorization.signature)
        .expect("proof is valid under colliding node identity");
    let before =
        replica_authorization_mutation_snapshot(&fixture.service, &put.lease_id, &put.object_id);

    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&authorization, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let after =
        replica_authorization_mutation_snapshot(&fixture.service, &put.lease_id, &put.object_id);
    assert!(before == after, "collision rejection must be read-only");
}

#[test]
fn replica_authorization_binds_source_target_and_target_bundle() {
    let (fixture, put) = provisioned_object_fixture();

    let mut wrong_source = fixture.replica_authorization(&put);
    let authorized_claims_commitment = wrong_source.claims().commitment();
    wrong_source.claims.source_ciphertext_commitment = [45; 32];
    resign_replica_authorization(&mut wrong_source, &fixture.admin_key);
    assert_ne!(
        authorized_claims_commitment,
        wrong_source.claims().commitment()
    );
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&wrong_source, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let alternate_target =
        IdentityKeyPair::from_bytes(&[46; 32]).expect("alternate target node key");
    let mut wrong_target = fixture.replica_authorization(&put);
    wrong_target.claims.target_node_id = alternate_target.public_key_bytes();
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&wrong_target, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut wrong_bundle = fixture.replica_authorization(&put);
    wrong_bundle.claims.target_bundle_commitment = [47; 32];
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&wrong_bundle, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
}

#[test]
fn replica_authorization_rejects_wrong_authority_domain_and_version() {
    let (fixture, put) = provisioned_object_fixture();

    let mut node_authorized = fixture.replica_authorization(&put);
    node_authorized.signature = fixture
        .service
        .node_identity
        .sign(&node_authorized.signing_bytes());
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&node_authorized, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut wrong_domain = fixture.replica_authorization(&put);
    let canonical = wrong_domain.signing_bytes();
    let mut wrong_domain_bytes = b"AeroNyx-BlindVault-ReplicaJobAuthorization-v0".to_vec();
    wrong_domain_bytes.extend_from_slice(&canonical[REPLICA_JOB_AUTHORIZATION_DOMAIN.len()..]);
    wrong_domain.signature = fixture.admin_key.sign(&wrong_domain_bytes);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&wrong_domain, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut wrong_version = fixture.replica_authorization(&put);
    wrong_version.claims.version = REPLICA_JOB_AUTHORIZATION_VERSION_V1 + 1;
    resign_replica_authorization(&mut wrong_version, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&wrong_version, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
}

#[test]
fn replica_authorization_rejects_invalid_time_bounds_and_shape() {
    let (fixture, put) = provisioned_object_fixture();

    let mut expired = fixture.replica_authorization(&put);
    expired.claims.authorized_at_ms = NOW_MS - 2;
    expired.claims.expires_at_ms = NOW_MS - 1;
    resign_replica_authorization(&mut expired, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&expired, NOW_MS),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut future = fixture.replica_authorization(&put);
    future.claims.authorized_at_ms = NOW_MS + 3;
    future.claims.expires_at_ms = NOW_MS + 4;
    resign_replica_authorization(&mut future, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&future, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut overlong = fixture.replica_authorization(&put);
    overlong.claims.expires_at_ms =
        overlong.claims.authorized_at_ms + MAX_REPLICA_JOB_AUTHORIZATION_TTL_MS + 1;
    resign_replica_authorization(&mut overlong, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&overlong, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut zero_identifier = fixture.replica_authorization(&put);
    zero_identifier.claims.job_id = [0; 16];
    resign_replica_authorization(&mut zero_identifier, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&zero_identifier, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
}

#[test]
fn replica_authorization_rejects_missing_or_expired_source_rows() {
    let (fixture, put) = provisioned_object_fixture();

    let mut missing_lease = fixture.replica_authorization(&put);
    missing_lease.claims.source_lease_id = [48; 32];
    resign_replica_authorization(&mut missing_lease, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&missing_lease, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let mut missing_object = fixture.replica_authorization(&put);
    missing_object.claims.source_object_id = [49; 32];
    resign_replica_authorization(&mut missing_object, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&missing_object, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    {
        let connection = fixture.service.connection.lock();
        connection
            .execute(
                "UPDATE blind_vault_objects SET expires_at_ms = ?1
                     WHERE lease_id = ?2 AND object_id = ?3",
                params![
                    sqlite_i64(NOW_MS + 2).expect("time"),
                    &put.lease_id[..],
                    &put.object_id[..]
                ],
            )
            .expect("expire object");
    }
    let expired_object = fixture.replica_authorization(&put);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&expired_object, NOW_MS + 3),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let (expired_lease_fixture, expired_lease_put) = provisioned_object_fixture();
    {
        let connection = expired_lease_fixture.service.connection.lock();
        connection
            .execute(
                "UPDATE blind_vault_objects SET expires_at_ms = ?1
                     WHERE lease_id = ?2 AND object_id = ?3",
                params![
                    sqlite_i64(NOW_MS + 2).expect("time"),
                    &expired_lease_put.lease_id[..],
                    &expired_lease_put.object_id[..]
                ],
            )
            .expect("bound object retention");
        connection
            .execute(
                "UPDATE blind_vault_leases SET expires_at_ms = ?1 WHERE lease_id = ?2",
                params![
                    sqlite_i64(NOW_MS + 3).expect("time"),
                    &expired_lease_put.lease_id[..]
                ],
            )
            .expect("expire lease");
    }
    let expired_lease = expired_lease_fixture.replica_authorization(&expired_lease_put);
    assert!(matches!(
        expired_lease_fixture
            .service
            .verify_replica_job_authorization(&expired_lease, NOW_MS + 4),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
}

#[test]
fn replica_authorization_must_fit_source_retention() {
    let (fixture, put) = provisioned_object_fixture();
    {
        let connection = fixture.service.connection.lock();
        connection
            .execute(
                "UPDATE blind_vault_objects SET expires_at_ms = ?1
                     WHERE lease_id = ?2 AND object_id = ?3",
                params![
                    sqlite_i64(NOW_MS + 60_000).expect("time"),
                    &put.lease_id[..],
                    &put.object_id[..]
                ],
            )
            .expect("shorten object retention");
    }
    let authorization = fixture.replica_authorization(&put);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&authorization, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));

    let (lease_fixture, lease_put) = provisioned_object_fixture();
    {
        let connection = lease_fixture.service.connection.lock();
        for table in ["blind_vault_objects", "blind_vault_leases"] {
            connection
                .execute(
                    &format!("UPDATE {table} SET expires_at_ms = ?1 WHERE lease_id = ?2"),
                    params![
                        sqlite_i64(NOW_MS + 2 * 60_000).expect("time"),
                        &lease_put.lease_id[..]
                    ],
                )
                .expect("shorten source retention");
        }
    }
    let authorization = lease_fixture.replica_authorization(&lease_put);
    assert!(matches!(
        lease_fixture
            .service
            .verify_replica_job_authorization(&authorization, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
}

#[test]
fn replica_authorization_maps_corrupt_or_unavailable_storage_coarsely() {
    let (fixture, put) = provisioned_object_fixture();
    let malformed_admin = [0xff_u8; 31];
    {
        let connection = fixture.service.connection.lock();
        connection
            .pragma_update(None, "ignore_check_constraints", true)
            .expect("inject corrupt durable row");
        connection
            .execute(
                "UPDATE blind_vault_leases SET admin_verifying_key = ?1
                     WHERE lease_id = ?2",
                params![&malformed_admin[..], &put.lease_id[..]],
            )
            .expect("store malformed admin verifier fixture");
    }
    let authorization = fixture.replica_authorization(&put);
    let error = fixture
        .service
        .verify_replica_job_authorization(&authorization, NOW_MS + 2)
        .expect_err("malformed durable verifier must fail closed");
    assert!(matches!(
        error,
        BlindVaultReplicaJobAuthorizationError::Unavailable
    ));
    assert_eq!(
        format!("{error}"),
        "blind vault replica job authorization unavailable"
    );
    assert_eq!(
        format!("{error:?}"),
        "blind vault replica job authorization unavailable"
    );

    let (unavailable_fixture, unavailable_put) = provisioned_object_fixture();
    let authorization = unavailable_fixture.replica_authorization(&unavailable_put);
    unavailable_fixture
        .service
        .connection
        .lock()
        .execute("DROP TABLE blind_vault_objects", [])
        .expect("make authority store unavailable");
    assert!(matches!(
        unavailable_fixture
            .service
            .verify_replica_job_authorization(&authorization, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Unavailable)
    ));
}

#[test]
fn invalid_mutation_signatures_do_not_compete_for_the_sqlite_writer() {
    let fixture = Fixture::new(10, 1024 * 1024);
    fixture.provision();
    let stored = fixture.put(3, 4);
    fixture
        .service
        .put(&stored, NOW_MS + 10)
        .expect("store object before writer contention");

    let blocker =
        Connection::open(&fixture.service.config.db_path).expect("open competing connection");
    blocker
        .execute_batch("BEGIN IMMEDIATE")
        .expect("hold competing write transaction");

    let wrong_key = IdentityKeyPair::from_bytes(&[11; 32]).expect("wrong authority key");
    let mut invalid_put = fixture.put(5, 6);
    invalid_put.sign(&wrong_key);
    assert!(matches!(
        fixture.service.put(&invalid_put, NOW_MS + 20),
        Err(BlindVaultServiceError::Protocol(
            BlindVaultError::InvalidSignature
        ))
    ));

    let mut invalid_delete = BlindVaultDeleteRequest::new([1; 32], [3; 32], [7; 16], NOW_MS + 20);
    invalid_delete.sign(&wrong_key);
    assert!(matches!(
        fixture.service.delete(&invalid_delete, NOW_MS + 20),
        Err(BlindVaultServiceError::Protocol(
            BlindVaultError::InvalidSignature
        ))
    ));

    blocker
        .execute_batch("ROLLBACK")
        .expect("release competing write transaction");
}
