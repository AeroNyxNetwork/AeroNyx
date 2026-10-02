// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn replica_authorization_accepts_valid_per_lease_admin_proof_without_mutation() {
    let (fixture, put) = provisioned_object_fixture();
    let authorization = fixture.replica_authorization(&put);
    let before =
        replica_authorization_mutation_snapshot(&fixture.service, &put.lease_id, &put.object_id);

    fixture
        .service
        .verify_replica_job_authorization(&authorization, NOW_MS + 2)
        .expect("valid per-lease admin authorization");

    let mut rejected = fixture.replica_authorization(&put);
    rejected.claims.source_ciphertext_commitment = [44; 32];
    resign_replica_authorization(&mut rejected, &fixture.admin_key);
    assert!(matches!(
        fixture
            .service
            .verify_replica_job_authorization(&rejected, NOW_MS + 2),
        Err(BlindVaultReplicaJobAuthorizationError::Rejected)
    ));
    let after =
        replica_authorization_mutation_snapshot(&fixture.service, &put.lease_id, &put.object_id);
    assert!(before == after, "authorization must be read-only");
}

#[test]
fn per_lease_quota_is_transactional() {
    let fixture = Fixture::new(1, 4 * 1024);
    fixture.provision();
    fixture
        .service
        .put(&fixture.put(3, 4), NOW_MS + 10)
        .expect("first object");
    assert!(matches!(
        fixture.service.put(&fixture.put(5, 6), NOW_MS + 20),
        Err(BlindVaultServiceError::QuotaExceeded)
    ));
    let status = fixture.service.status(NOW_MS + 30).expect("status");
    assert_eq!(status.live_objects, 1);
    assert_eq!(status.live_ciphertext_bytes, 4 * 1024);
}
