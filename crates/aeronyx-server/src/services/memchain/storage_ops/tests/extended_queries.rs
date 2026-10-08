// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn test_stats() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m")
        .await;
    let stats = s.stats().await;
    assert_eq!(stats.total_records, 2);
    assert_eq!(stats.active_records, 2);
    assert_eq!(stats.by_layer.episode, 1);
    assert_eq!(stats.by_layer.identity, 1);
}

#[tokio::test]
async fn test_get_embedding_model() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let r = make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode);
    let rid = r.record_id;
    s.insert(&r, "minilm-l6-v2").await;
    assert_eq!(
        s.get_embedding_model(&rid).await,
        Some("minilm-l6-v2".into())
    );
}

// [MEMCHAIN-PHALA-EMBEDDING-OWNER-SCOPE 2026-10-06 by Codex]
#[tokio::test]
async fn test_embedding_backfill_query_is_owner_scoped() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let owner_a = [0xA1; 32];
    let owner_b = [0xB2; 32];
    let mut record_a = make_rec_owner(100, owner_a, MemoryLayer::Episode);
    let mut record_b = make_rec_owner(200, owner_b, MemoryLayer::Episode);
    let oversized = MemoryRecord::new(
        owner_a,
        50,
        MemoryLayer::Episode,
        vec!["test".into()],
        "ai".into(),
        vec![b'x'; 16 * 1024 + 1],
        Vec::new(),
    );
    record_a.embedding.clear();
    record_b.embedding.clear();
    let id_a = record_a.record_id;
    let id_b = record_b.record_id;
    assert!(storage.insert(&record_a, "old-model").await);
    assert!(storage.insert(&record_b, "old-model").await);
    assert!(storage.insert(&oversized, "old-model").await);

    let pending = storage
        .get_records_needing_embedding(&owner_a, "new-model", 32)
        .await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].record_id, id_a);
    assert_ne!(pending[0].record_id, id_b);
}

#[tokio::test]
async fn test_get_overview() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m")
        .await;
    s.insert(&make_rec_owner(300, owner, MemoryLayer::Knowledge), "m")
        .await;
    let ov = s.get_overview(&owner, 20).await;
    assert_eq!(*ov.by_layer.get("episode").unwrap(), 1);
    assert_eq!(*ov.by_layer.get("identity").unwrap(), 1);
    assert_eq!(*ov.by_layer.get("knowledge").unwrap(), 1);
    assert_eq!(ov.last_memory_at, 300);
}

#[tokio::test]
async fn test_count_distinct_owners_empty() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    assert_eq!(s.count_distinct_owners().await, 0);
}

#[tokio::test]
async fn test_owner_exists_after_insert() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
        .await;
    assert!(s.owner_exists(&owner).await);
    assert!(!s.owner_exists(&[0xBB; 32]).await);
}

// ── v2.5.3+Isolation: get_active_records_by_context tests ──

#[tokio::test]
async fn test_get_active_records_by_context_via_session() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Register a session with project_id = "work"
    s.upsert_session("sess_work", &owner, Some("work"), "chat", now, 2)
        .await
        .unwrap();

    // Insert two records linked to the work session
    let mut r1 = make_rec_owner(100, owner, MemoryLayer::Episode);
    // Manually set session_id on the record row after insert
    s.insert(&r1, "m").await;
    {
        let conn = s.conn_lock().await;
        let _ = conn.execute(
            "UPDATE records SET session_id = 'sess_work' WHERE record_id = ?1",
            params![r1.record_id.as_slice()],
        );
    }

    // Insert a record with no session (personal — untagged)
    let r2 = make_rec_owner(200, owner, MemoryLayer::Episode);
    s.insert(&r2, "m").await;

    // Query context = "work" should return only r1
    let results = s
        .get_active_records_by_context(&owner, "work", None, 100)
        .await;
    assert_eq!(
        results.len(),
        1,
        "Only the work-tagged record should be returned"
    );
    assert_eq!(results[0].record_id, r1.record_id);

    // Query context = "personal" should return nothing (no records tagged personal)
    let personal = s
        .get_active_records_by_context(&owner, "personal", None, 100)
        .await;
    assert!(personal.is_empty(), "No personal records exist");
}

#[tokio::test]
async fn test_get_active_records_by_context_no_cross_owner() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner_a = [0xAA; 32];
    let owner_b = [0xBB; 32];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    s.upsert_session("sess_a", &owner_a, Some("work"), "chat", now, 1)
        .await
        .unwrap();

    let r = make_rec_owner(100, owner_a, MemoryLayer::Episode);
    s.insert(&r, "m").await;
    {
        let conn = s.conn_lock().await;
        let _ = conn.execute(
            "UPDATE records SET session_id = 'sess_a' WHERE record_id = ?1",
            params![r.record_id.as_slice()],
        );
    }

    // owner_b querying "work" must see nothing (different owner)
    let results = s
        .get_active_records_by_context(&owner_b, "work", None, 100)
        .await;
    assert!(results.is_empty(), "Cross-owner isolation must be enforced");
}
