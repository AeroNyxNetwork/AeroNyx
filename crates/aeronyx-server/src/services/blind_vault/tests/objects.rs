// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn immutable_put_retry_and_capability_pull_preserve_ciphertext() {
    let fixture = Fixture::new(10, 1024 * 1024);
    fixture.provision();
    let put = fixture.put(3, 4);
    let first = fixture.service.put(&put, NOW_MS + 10).expect("put");
    let retry = fixture.service.put(&put, NOW_MS + 20).expect("retry");
    assert_eq!(first.accepted_at_ms, retry.accepted_at_ms);
    assert!(first.matches_put(&put));

    let page = fixture
        .service
        .pull_page(&[1; 32], &fixture.read_capability, None, 10, NOW_MS + 30)
        .expect("authorised pull");
    assert_eq!(page.objects.len(), 1);
    assert_eq!(page.objects[0].ciphertext, put.ciphertext);
    assert_eq!(
        page.objects[0].ciphertext_commitment,
        put.ciphertext_commitment
    );
    assert!(page.continuation_cursor.is_none());

    assert!(matches!(
        fixture
            .service
            .pull_page(&[1; 32], &[0; 32], None, 10, NOW_MS + 30),
        Err(BlindVaultServiceError::ReadUnauthorized)
    ));
}

#[test]
fn admin_delete_is_idempotent_and_prevents_object_id_reuse() {
    let fixture = Fixture::new(10, 1024 * 1024);
    fixture.provision();
    let put = fixture.put(3, 4);
    fixture.service.put(&put, NOW_MS + 10).expect("put");

    let mut delete = BlindVaultDeleteRequest::new([1; 32], [3; 32], [5; 16], NOW_MS + 20);
    delete.sign(&fixture.admin_key);
    let first = fixture
        .service
        .delete(&delete, NOW_MS + 20)
        .expect("delete");
    let retry = fixture
        .service
        .delete(&delete, NOW_MS + 30)
        .expect("delete retry");
    assert_eq!(
        first.previous_ciphertext_commitment,
        put.ciphertext_commitment
    );
    assert_eq!(first.deleted_at_ms, retry.deleted_at_ms);
    assert!(first.matches_delete(&delete));
    assert!(matches!(
        fixture.service.put(&put, NOW_MS + 40),
        Err(BlindVaultServiceError::ObjectDeleted)
    ));
}

#[test]
fn pull_pages_are_bounded_and_cleanup_is_batched() {
    let fixture = Fixture::new(10, 1024 * 1024);
    fixture.provision();
    let first = fixture.put(3, 4);
    let second = fixture.put(5, 6);
    fixture.service.put(&first, NOW_MS + 10).expect("first");
    fixture.service.put(&second, NOW_MS + 20).expect("second");

    let page = fixture
        .service
        .pull_page(&[1; 32], &fixture.read_capability, None, 1, NOW_MS + 30)
        .expect("first page");
    assert_eq!(page.objects.len(), 1);
    let cursor = page.continuation_cursor.expect("continuation");
    assert_eq!(cursor.len(), PULL_CURSOR_BYTES);
    let mut tampered_cursor = cursor.clone();
    tampered_cursor[10] ^= 1;
    assert!(matches!(
        fixture
            .service
            .decode_pull_cursor(&[1; 32], &tampered_cursor),
        Err(BlindVaultServiceError::InvalidPullCursor)
    ));
    assert!(matches!(
        fixture.service.decode_pull_cursor(&[2; 32], &cursor),
        Err(BlindVaultServiceError::InvalidPullCursor)
    ));

    // Objects appended after the first page belong to the next snapshot,
    // never to the in-progress recovery page sequence.
    let third = fixture.put(7, 8);
    fixture.service.put(&third, NOW_MS + 25).expect("third");
    let page = fixture
        .service
        .pull_page(
            &[1; 32],
            &fixture.read_capability,
            Some(&cursor),
            1,
            NOW_MS + 30,
        )
        .expect("second page");
    assert_eq!(page.objects.len(), 1);
    assert_eq!(page.objects[0].object_id, second.object_id);
    assert!(page.continuation_cursor.is_none());

    let cleanup = fixture
        .service
        .run_cleanup(NOW_MS + 2 * 24 * 60 * 60 * 1_000)
        .expect("cleanup");
    assert_eq!(cleanup.objects_removed, 3);
    assert_eq!(
        fixture
            .service
            .status(NOW_MS + 3)
            .expect("status")
            .live_objects,
        0
    );
}

#[test]
fn put_failure_classes_are_coarse_and_retry_safe() {
    assert_eq!(
        BlindVaultServiceError::LeaseNotFound.put_failure_class(),
        BlindVaultPutFailureClass::Rejected
    );
    assert_eq!(
        BlindVaultServiceError::QuotaExceeded.put_failure_class(),
        BlindVaultPutFailureClass::Capacity
    );
    assert_eq!(
        BlindVaultServiceError::Disabled.put_failure_class(),
        BlindVaultPutFailureClass::Unavailable
    );
}
