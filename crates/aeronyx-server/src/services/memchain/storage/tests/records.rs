// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

// ========================================
// v2.6.0+NodeBlind (Brick 1) — node-blind storage round-trip
// ========================================

#[tokio::test]
async fn test_blind_record_stored_and_read_verbatim() {
    // Encrypted store: the node holds a record_key. A blind record must
    // still be stored/returned verbatim — never encrypted or decrypted by
    // the node — while a normal record is encrypted at rest as usual.
    let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();

    // Node-blind record: content is the CLIENT's own ciphertext (opaque to
    // the node); blind = true.
    let sealed = b"client-sealed-ciphertext-opaque-to-node".to_vec();
    let mut blind_rec = MemoryRecord::new(
        [0xBB; 32],
        200,
        MemoryLayer::Knowledge,
        vec!["sealed".into()],
        "client".into(),
        sealed.clone(),
        vec![0.9, 0.8],
    );
    blind_rec.blind = true;
    let bid = blind_rec.record_id;
    assert!(s.insert(&blind_rec, "client-embed").await);

    // Force a real DB read (bypass the write-through cache) to exercise
    // row_to_record's blind branch.
    s.cache.write().invalidate(&bid);
    let got = s.get(&bid).await.unwrap();
    assert!(got.blind, "blind flag must survive the DB round-trip");
    assert_eq!(
        got.encrypted_content, sealed,
        "blind content must be stored and returned byte-for-byte (no node crypto)"
    );

    // Sanity: a normal (sighted) record in the SAME encrypted store is
    // encrypted at rest and decrypted back to plaintext on read.
    let plain = b"node-visible-plaintext".to_vec();
    let sighted = MemoryRecord::new(
        [0xBB; 32],
        201,
        MemoryLayer::Knowledge,
        vec!["plain".into()],
        "node".into(),
        plain.clone(),
        vec![0.1],
    );
    let sid = sighted.record_id;
    assert!(s.insert(&sighted, "minilm").await);
    s.cache.write().invalidate(&sid);
    let got2 = s.get(&sid).await.unwrap();
    assert!(!got2.blind);
    assert_eq!(
        got2.encrypted_content, plain,
        "sighted content round-trips to plaintext"
    );
}

#[tokio::test]
async fn test_blind_record_project_grouping() {
    let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
    let owner = [0xBB; 32];
    let mut rec = MemoryRecord::new(
        owner,
        500,
        MemoryLayer::Knowledge,
        vec![],
        "client".into(),
        b"opaque".to_vec(),
        vec![],
    );
    rec.blind = true;
    assert!(s.insert(&rec, "client").await);

    // Archive it under a project (a label or an opaque hash), then recall by it.
    s.set_record_project_id(&rec.record_id, &owner, "proj_alpha")
        .await;
    let hits = s
        .get_active_records_by_context(&owner, "proj_alpha", None, 10)
        .await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record_id, rec.record_id);
    assert!(hits[0].blind, "blind flag survives the project query");

    // A different project returns nothing.
    assert!(s
        .get_active_records_by_context(&owner, "proj_other", None, 10)
        .await
        .is_empty());
}

#[tokio::test]
async fn test_insert_blind_replica_stored_verbatim() {
    // This node has its OWN record_key (encrypted store).
    let s = MemoryStorage::open(":memory:", Some([0x22u8; 32])).unwrap();

    // A replica from ANOTHER owner: content is that owner's opaque ciphertext.
    // blind is false on the incoming record (it is #[serde(skip)] over the wire).
    let foreign_ct = b"another-owners-ciphertext-opaque".to_vec();
    let replica = MemoryRecord::new(
        [0xEE; 32],
        700,
        MemoryLayer::Knowledge,
        vec![],
        "peer".into(),
        foreign_ct.clone(),
        vec![],
    );
    let rid = replica.record_id;
    assert!(!replica.blind);
    assert!(s.insert_blind_replica(&replica, "peer-model").await);

    // Read back: stored byte-for-byte (NOT re-encrypted with this node's key),
    // blind marker set, foreign owner preserved.
    s.cache.write().invalidate(&rid);
    let got = s.get(&rid).await.unwrap();
    assert!(got.blind);
    assert_eq!(
        got.encrypted_content, foreign_ct,
        "replica must be stored verbatim, not re-encrypted"
    );
    assert_eq!(got.owner, [0xEE; 32]);

    let mut vector_bearing_replica = replica.clone();
    vector_bearing_replica.embedding = vec![0.3, 0.4];
    assert!(!s
        .insert_blind_replica(&vector_bearing_replica, "peer-model")
        .await);

    // A tampered record (content no longer matches record_id) is rejected.
    let mut bad = replica.clone();
    bad.encrypted_content = b"tampered".to_vec();
    assert!(!s.insert_blind_replica(&bad, "peer-model").await);
}


#[tokio::test]
async fn test_insert_and_get() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let r = make_rec(100, MemoryLayer::Episode, "test");
    let id = r.record_id;
    assert!(s.insert(&r, "minilm").await);
    assert!(!s.insert(&r, "minilm").await);
    let got = s.get(&id).await.unwrap();
    assert_eq!(got.source_ai, "test");
    assert_eq!(got.layer, MemoryLayer::Episode);
    assert_eq!(got.embedding, vec![0.1, 0.2, 0.3]);
}

#[tokio::test]
async fn test_get_active_records() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let o = [0xAA; 32];
    s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(200, o, MemoryLayer::Knowledge), "m")
        .await;
    s.insert(&make_rec_owner(300, o, MemoryLayer::Archive), "m")
        .await;
    assert_eq!(s.get_active_records(&o, None, 100).await.len(), 3);
    assert_eq!(
        s.get_active_records(&o, Some(MemoryLayer::Episode), 100)
            .await
            .len(),
        1
    );
}

#[tokio::test]
async fn test_all_owner_embedding_rebuild_preserves_owner_and_status_scope() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner_a = [0xA1; 32];
    let owner_b = [0xB2; 32];
    let active_a = make_rec_owner(100, owner_a, MemoryLayer::Episode);
    let active_b = make_rec_owner(200, owner_b, MemoryLayer::Knowledge);
    let revoked_b = make_rec_owner(300, owner_b, MemoryLayer::Archive);

    assert!(s.insert(&active_a, "model-a").await);
    assert!(s.insert(&active_b, "model-b").await);
    assert!(s.insert(&revoked_b, "model-b").await);
    assert!(s.revoke(&revoked_b.record_id).await);

    let local_only = s.get_records_with_embedding(&owner_a).await;
    assert_eq!(
        local_only.len(),
        1,
        "legacy owner-scoped rebuild is unchanged"
    );
    assert_eq!(local_only[0].0.owner, owner_a);

    let all = s.get_all_records_with_embedding().await;
    assert_eq!(all.len(), 2, "revoked records must not re-enter the index");
    assert!(all.iter().any(|(record, model)| {
        record.record_id == active_a.record_id && record.owner == owner_a && model == "model-a"
    }));
    assert!(all.iter().any(|(record, model)| {
        record.record_id == active_b.record_id && record.owner == owner_b && model == "model-b"
    }));
    assert!(!all
        .iter()
        .any(|(record, _)| record.record_id == revoked_b.record_id));
}

#[tokio::test]
async fn test_revoke() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let r = make_rec(100, MemoryLayer::Episode, "t");
    s.insert(&r, "m").await;
    assert!(s.revoke(&r.record_id).await);
    let got = s.get(&r.record_id).await.unwrap();
    assert_eq!(got.status, RecordStatus::Revoked);
    assert!(got.encrypted_content.is_empty());
}

#[tokio::test]
async fn test_encrypted_insert_and_get() {
    use super::super::super::storage_crypto::derive_record_key;
    let key = derive_record_key(&[0x42; 32]);
    let s = MemoryStorage::open(":memory:", Some(key)).unwrap();
    let r = make_rec(100, MemoryLayer::Episode, "test");
    s.insert(&r, "m").await;
    s.cache.write().clear();
    let got = s.get(&r.record_id).await.unwrap();
    assert_eq!(got.encrypted_content, b"encrypted_data");
}

// ========================================
// v2.4.0: Existing schema tests (preserved)
// ========================================

#[tokio::test]
async fn test_v5_records_has_new_columns() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;
    let result = conn.prepare("SELECT project_id, session_id, episode_id FROM records LIMIT 0");
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_v5_backward_compat_insert_without_new_cols() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let r = make_rec(100, MemoryLayer::Episode, "test");
    assert!(s.insert(&r, "minilm").await);

    let conn = s.conn.lock().await;
    let (pid, sid, eid): (Option<String>, Option<String>, Option<String>) = conn
        .query_row(
            "SELECT project_id, session_id, episode_id FROM records WHERE record_id = ?1",
            params![r.record_id.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert!(pid.is_none());
    assert!(sid.is_none());
    assert!(eid.is_none());
}

/// Verify update_record_content clears embedding on content change.
#[tokio::test]
async fn test_update_record_content_clears_embedding() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    let r = make_rec_owner(100, owner, MemoryLayer::Knowledge);
    let id = r.record_id;
    s.insert(&r, "minilm").await;

    // Add an embedding manually
    {
        let conn = s.conn.lock().await;
        conn.execute(
            "UPDATE records SET embedding = x'01020304', embedding_model = 'minilm', embedding_dim = 1 WHERE record_id = ?1",
            params![id.as_slice()],
        ).unwrap();
    }
    s.cache.write().clear();

    // Patch content
    let patched = s
        .update_record_content(&id, &owner, Some("new content"), None, None, None)
        .await
        .unwrap();
    assert!(patched);

    // Verify embedding was cleared
    let conn = s.conn.lock().await;
    let (emb, model, dim): (Option<Vec<u8>>, String, i64) = conn
        .query_row(
            "SELECT embedding, embedding_model, embedding_dim FROM records WHERE record_id = ?1",
            params![id.as_slice()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert!(emb.is_none(), "Embedding must be NULL after content change");
    assert_eq!(model, "");
    assert_eq!(dim, 0);
}

/// Verify update_record_content rejects wrong owner.
#[tokio::test]
async fn test_update_record_content_wrong_owner_rejected() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    let other = [0xBB; 32];
    let r = make_rec_owner(100, owner, MemoryLayer::Knowledge);
    let id = r.record_id;
    s.insert(&r, "minilm").await;

    let result = s
        .update_record_content(&id, &other, Some("hacked"), None, None, None)
        .await
        .unwrap();
    assert!(!result, "Wrong owner should be rejected");
}

/// Verify get_records_for_session returns correct records.
#[tokio::test]
async fn test_get_records_for_session() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    let r = make_rec_owner(100, owner, MemoryLayer::Episode);
    let id = r.record_id;
    s.insert(&r, "m").await;
    s.set_record_session_id(&id, &owner, "sess_xyz").await;

    let records = s.get_records_for_session("sess_xyz", &owner).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].record_id, id);

    let empty = s.get_records_for_session("sess_other", &owner).await;
    assert!(empty.is_empty());
}
