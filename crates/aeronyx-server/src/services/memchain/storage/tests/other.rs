// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn unmanaged_local_storage_keeps_backward_compatible_growth_behavior() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let permit = storage.acquire_growth_permit(u64::MAX).await.unwrap();
    drop(permit);
    assert_eq!(storage.count().await, 0);
}

#[tokio::test]
async fn test_blind_fts_index_and_search() {
    let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
    let owner = [0xBB; 32];
    let mut rec = MemoryRecord::new(
        owner,
        300,
        MemoryLayer::Knowledge,
        vec![],
        "client".into(),
        b"opaque-ciphertext".to_vec(),
        vec![],
    );
    rec.blind = true;
    assert!(s.insert(&rec, "client").await);

    // Client-supplied keyed token-hashes (hex). The node never sees plaintext.
    let terms = vec!["a3f29c4d".to_string(), "9c4d2eb1".to_string()];
    s.fts_index_blind_terms(&rec.record_id, &owner, &terms)
        .await;

    // Query by an indexed term-hash → finds the record.
    let hits = s
        .bm25_search_blind(&["a3f29c4d".to_string()], &owner, 10)
        .await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, hex::encode(rec.record_id));

    // Query by a non-indexed hash → no hits.
    let none = s
        .bm25_search_blind(&["deadbeef".to_string()], &owner, 10)
        .await;
    assert!(none.is_empty());

    // Wrong owner → no hits (access scoping via owner_hex).
    let other = s
        .bm25_search_blind(&["a3f29c4d".to_string()], &[0xCC; 32], 10)
        .await;
    assert!(other.is_empty());

    // Non-hex query terms are rejected (no FTS syntax injection).
    let bad = s
        .bm25_search_blind(&["a3f2 OR x".to_string()], &owner, 10)
        .await;
    assert!(bad.is_empty());
}

#[tokio::test]
async fn test_blind_provenance_roundtrip() {
    let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
    let owner = [0xBB; 32];
    let derived = MemoryRecord::new(
        owner,
        400,
        MemoryLayer::Knowledge,
        vec![],
        "summary".into(),
        b"summary-ciphertext".to_vec(),
        vec![],
    );
    let did = derived.record_id;
    let src_a = hex::encode([0x01u8; 32]);
    let src_b = hex::encode([0x02u8; 32]);

    // Self-links and empties are ignored; sources are recorded.
    s.insert_blind_provenance(
        &did,
        &owner,
        &[
            src_a.clone(),
            src_b.clone(),
            hex::encode(did),
            String::new(),
        ],
    )
    .await;

    let mut got = s.get_blind_provenance(&did).await;
    got.sort();
    let mut want = vec![src_a, src_b];
    want.sort();
    assert_eq!(got, want);

    // A record with no provenance returns empty.
    assert!(s.get_blind_provenance(&[0x09u8; 32]).await.is_empty());
}

#[tokio::test]
async fn test_reject_invalid_hash() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let mut r = make_rec(100, MemoryLayer::Episode, "t");
    r.record_id = [0xFF; 32];
    assert!(!s.insert(&r, "m").await);
    assert_eq!(s.total_rejected(), 1);
}

#[tokio::test]
async fn test_v5_tables_exist() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let expected_tables = [
        "episodes",
        "entities",
        "knowledge_edges",
        "episode_edges",
        "communities",
        "projects",
        "sessions",
        "artifacts",
    ];

    for table in &expected_tables {
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                params![table],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
            > 0;
        assert!(exists, "Table '{}' should exist in schema v5", table);
    }
}

#[tokio::test]
async fn test_v6_tables_exist() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let expected_tables = ["cognitive_tasks", "llm_usage_log"];
    for table in &expected_tables {
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                params![table],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
            > 0;
        assert!(exists, "Table '{}' should exist in schema v6", table);
    }

    let title_ok = conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok();
    assert!(title_ok, "sessions.title column should exist in schema v6");
}

#[tokio::test]
async fn test_v6_sessions_title_column() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO sessions (session_id, owner, session_type, started_at, turn_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        params!["sess_001", [0xAAu8; 32].as_slice(), "chat", now, 5i64],
    )
    .unwrap();

    let title: Option<String> = conn
        .query_row(
            "SELECT title FROM sessions WHERE session_id = 'sess_001'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(title.is_none());

    conn.execute(
        "UPDATE sessions SET title = ?1 WHERE session_id = ?2",
        params!["JWT Auth Implementation Discussion", "sess_001"],
    )
    .unwrap();

    let title: Option<String> = conn
        .query_row(
            "SELECT title FROM sessions WHERE session_id = 'sess_001'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        title,
        Some("JWT Auth Implementation Discussion".to_string())
    );
}

#[tokio::test]
async fn test_v5_knowledge_edges_temporal() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?10)",
        params![
            [0xAAu8; 32].as_slice(),
            "ent_auth",
            "ent_jwt",
            "USES",
            "auth module uses JWT",
            1.0f64,
            0.95f64,
            now,
            now,
            now,
        ],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            [0xAAu8; 32].as_slice(),
            "ent_auth",
            "ent_basic",
            "USES",
            "auth module uses Basic Auth",
            1.0f64,
            0.9f64,
            now - 86400,
            now,
            now - 86400,
            now,
        ],
    )
    .unwrap();

    let valid_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM knowledge_edges WHERE valid_until IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(valid_count, 1);

    let total_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM knowledge_edges", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total_count, 2);
}

#[tokio::test]
async fn test_v5_episode_edges_bidirectional() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO episode_edges (owner, episode_id, entity_id, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            [0xAAu8; 32].as_slice(),
            "ep_001",
            "ent_jwt",
            "mentioned",
            now
        ],
    )
    .unwrap();

    let entity_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM episode_edges WHERE episode_id = 'ep_001'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(entity_count, 1);

    let episode_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM episode_edges WHERE entity_id = 'ent_jwt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(episode_count, 1);
}

#[tokio::test]
async fn test_v5_sessions_and_projects() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO communities (community_id, owner, name, summary, entity_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params!["comm_1", [0xAAu8; 32].as_slice(), "Project B", "Auth system", 5i64, now, now],
    ).unwrap();

    conn.execute(
        "INSERT INTO projects (project_id, owner, name, status, community_id, summary,
                created_at, updated_at, last_active_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            "comm_1",
            [0xAAu8; 32].as_slice(),
            "Project B",
            "active",
            "comm_1",
            "Auth system project",
            now,
            now,
            now
        ],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO sessions (session_id, owner, project_id, session_type,
                started_at, turn_count, summary)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            "sess_001",
            [0xAAu8; 32].as_slice(),
            "comm_1",
            "code",
            now,
            15i64,
            "Implemented JWT auth"
        ],
    )
    .unwrap();

    let session_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE project_id = 'comm_1'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(session_count, 1);

    let project_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE owner = ?1 AND status = 'active'",
            params![[0xAAu8; 32].as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(project_count, 1);
}

#[tokio::test]
async fn test_v5_artifacts_version_chain() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10)",
        params![
            "art_v1",
            [0xAAu8; 32].as_slice(),
            "sess_001",
            "code",
            "auth.rs",
            "rust",
            1i64,
            b"fn auth() {}".as_slice(),
            "hash_v1",
            now,
        ],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            "art_v2",
            [0xAAu8; 32].as_slice(),
            "sess_002",
            "code",
            "auth.rs",
            "rust",
            2i64,
            "art_v1",
            b"fn auth() { jwt() }".as_slice(),
            "hash_v2",
            now + 100,
        ],
    )
    .unwrap();

    let latest_version: i64 = conn
        .query_row(
            "SELECT MAX(version) FROM artifacts WHERE owner = ?1 AND filename = 'auth.rs'",
            params![[0xAAu8; 32].as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(latest_version, 2);

    let parent: Option<String> = conn
        .query_row(
            "SELECT parent_id FROM artifacts WHERE artifact_id = 'art_v2'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(parent, Some("art_v1".to_string()));
}

#[tokio::test]
async fn test_migration_v5_to_v6() {
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
             CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (5);
             CREATE TABLE sessions (
                 session_id TEXT PRIMARY KEY,
                 owner BLOB NOT NULL,
                 started_at INTEGER NOT NULL
             );
             CREATE TABLE chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);",
    )
    .unwrap();

    MemoryStorage::maybe_migrate(&conn).unwrap();

    let ct_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cognitive_tasks'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert!(ct_exists);

    let ul_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='llm_usage_log'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert!(ul_exists);

    let title_ok = conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok();
    assert!(title_ok);

    let v: u32 = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    // maybe_migrate() runs through the latest schema: v6 adds SuperNode,
    // v7 adds the blind marker, v8 creates commitment tables, v9 adds the
    // bounded proof vault, v10 adds durable equivocation incidents, and
    // v11 adds sticky trusted-divergence incidents, v12 adds immutable
    // checkpoint certificates, v13 adds durable coordinator leases, and
    // v14 adds aggregate-only delivery-anchor witness high-water state,
    // v15 adds replayable coordinator authority history, v16 adds an
    // independent custody-audit witness high-water namespace, and v17
    // adds bounded producer-side signed witness receipt evidence
    // even when create_schema() was not called.
    // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] This legacy
    // fixture must follow the authoritative latest additive migration
    // instead of freezing a prior schema version.
    assert_eq!(v, SCHEMA_VERSION);
    for table in [
        "record_commitment_blocks",
        "record_block_commitments",
        "record_checkpoint_evidence",
        "record_checkpoint_equivocations",
        "record_checkpoint_trusted_divergences",
        "record_checkpoint_certificates",
        "record_checkpoint_certificate_members",
        "record_coordinator_leases",
        "verified_delivery_anchor_witnesses",
        "record_coordinator_handovers",
        "custody_audit_anchor_witnesses",
        "custody_audit_witness_receipt_evidence",
    ] {
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                params![table],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
            > 0;
        assert!(exists, "missing migrated table: {table}");
    }
}

#[test]
fn test_v15_to_current_migration_preserves_existing_witness_state() {
    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] The additive migration
    // must not rewrite or conflate the established delivery-cache witness
    // namespace while creating the independent custody table.
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (15);
             CREATE TABLE verified_delivery_anchor_witnesses (
                 requester BLOB PRIMARY KEY,
                 generation INTEGER NOT NULL,
                 anchor_digest BLOB NOT NULL,
                 observed_at INTEGER NOT NULL
             );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO verified_delivery_anchor_witnesses
             (requester, generation, anchor_digest, observed_at)
             VALUES (?1, 41, ?2, 1000)",
        params![[0x91u8; 32].as_slice(), [0x92u8; 32].as_slice()],
    )
    .unwrap();

    MemoryStorage::maybe_migrate(&conn).unwrap();

    let version: u32 = conn
        .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
        .unwrap();
    let delivery_generation: i64 = conn
        .query_row(
            "SELECT generation FROM verified_delivery_anchor_witnesses
                 WHERE requester=?1",
            params![[0x91u8; 32].as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    let custody_table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='custody_audit_anchor_witnesses'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    let receipt_table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='custody_audit_witness_receipt_evidence'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(delivery_generation, 41);
    assert!(custody_table_exists);
    assert!(receipt_table_exists);
}

#[test]
fn test_v16_to_v17_migration_preserves_custody_witness_high_water() {
    // [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] Producer-side
    // receipt evidence is additive and must not rewrite the independent
    // witness-side monotonic high-water namespace during upgrade.
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (16);
             CREATE TABLE custody_audit_anchor_witnesses (
                 producer BLOB PRIMARY KEY,
                 generation INTEGER NOT NULL,
                 frame_sha256 BLOB NOT NULL,
                 observed_at INTEGER NOT NULL
             );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO custody_audit_anchor_witnesses
             (producer,generation,frame_sha256,observed_at)
             VALUES (?1,7,?2,1001)",
        params![[0x93u8; 32].as_slice(), [0x94u8; 32].as_slice()],
    )
    .unwrap();

    MemoryStorage::maybe_migrate(&conn).unwrap();

    let version: u32 = conn
        .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
        .unwrap();
    let retained: (i64, Vec<u8>, i64) = conn
        .query_row(
            "SELECT generation,frame_sha256,observed_at
                 FROM custody_audit_anchor_witnesses WHERE producer=?1",
            params![[0x93u8; 32].as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    let receipt_table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='custody_audit_witness_receipt_evidence'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(retained, (7, vec![0x94; 32], 1001));
    assert!(receipt_table_exists);
}

#[test]
fn test_v17_to_v18_migration_defaults_existing_receipts_to_live_policy() {
    // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Legacy rows
    // were admitted only by the strict network path. Migration must retain
    // them verbatim and must not grant the wider operator-import window.
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (17);
             CREATE TABLE custody_audit_witness_receipt_evidence (
                 receipt_digest BLOB PRIMARY KEY,
                 producer BLOB NOT NULL,
                 witness BLOB NOT NULL,
                 requested_generation INTEGER NOT NULL,
                 requested_frame_sha256 BLOB NOT NULL,
                 retained_generation INTEGER NOT NULL,
                 retained_frame_sha256 BLOB NOT NULL,
                 outcome INTEGER NOT NULL,
                 observed_at INTEGER NOT NULL,
                 receipt_frame BLOB NOT NULL,
                 persisted_at INTEGER NOT NULL
             );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO custody_audit_witness_receipt_evidence
             VALUES (?1,?2,?3,1,?4,1,?5,0,1000,?6,1001)",
        params![
            [0x95u8; 32].as_slice(),
            [0x96u8; 32].as_slice(),
            [0x97u8; 32].as_slice(),
            [0x98u8; 32].as_slice(),
            [0x99u8; 32].as_slice(),
            [0x9au8; 8].as_slice(),
        ],
    )
    .unwrap();

    MemoryStorage::maybe_migrate(&conn).unwrap();

    let version: u32 = conn
        .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
        .unwrap();
    let admission: (i64, i64, i64) = conn
        .query_row(
            "SELECT admission_kind,admission_max_delay_secs,persisted_at
                 FROM custody_audit_witness_receipt_evidence",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(admission, (0, 60, 1001));
}

// ========================================
// v2.5.2+Provenance: Bug fix tests
// ========================================

/// Verify that the memory_edges migration uses record.owner, not source_id as owner.
#[tokio::test]
async fn test_memory_edges_migration_uses_correct_owner() {
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
             CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (4);
             CREATE TABLE chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);
             -- Simulate v4 records table with owner column
             CREATE TABLE records (
                 record_id BLOB PRIMARY KEY,
                 owner BLOB NOT NULL,
                 timestamp INTEGER,
                 layer INTEGER,
                 topic_tags TEXT DEFAULT '[]',
                 source_ai TEXT DEFAULT '',
                 status INTEGER DEFAULT 0,
                 supersedes BLOB,
                 encrypted_content BLOB DEFAULT x'',
                 embedding BLOB,
                 embedding_model TEXT DEFAULT '',
                 embedding_dim INTEGER DEFAULT 0,
                 signature BLOB NOT NULL DEFAULT x'',
                 access_count INTEGER DEFAULT 0,
                 created_at INTEGER,
                 positive_feedback INTEGER DEFAULT 0,
                 negative_feedback INTEGER DEFAULT 0,
                 conflict_with BLOB
             );
             -- Simulate v4 memory_edges
             CREATE TABLE memory_edges (
                 source_id BLOB NOT NULL,
                 target_id BLOB NOT NULL,
                 edge_type TEXT DEFAULT 'co_occurred',
                 weight REAL DEFAULT 1.0,
                 created_at INTEGER NOT NULL,
                 PRIMARY KEY (source_id, target_id)
             );",
    )
    .unwrap();

    let owner_bytes = [0xBBu8; 32];
    let source_id = [0x01u8; 32];
    let target_id = [0x02u8; 32];
    let now = 1_700_000_000i64;

    // Insert a record so the JOIN can resolve owner
    conn.execute(
        "INSERT INTO records (record_id, owner, timestamp, layer, signature, created_at)
             VALUES (?1, ?2, ?3, 1, x'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', ?4)",
        params![source_id.as_slice(), owner_bytes.as_slice(), now, now],
    ).unwrap();

    conn.execute(
        "INSERT INTO memory_edges (source_id, target_id, weight, created_at)
             VALUES (?1, ?2, 1.0, ?3)",
        params![source_id.as_slice(), target_id.as_slice(), now],
    )
    .unwrap();

    // Match MemoryStorage::open() upgrade order: create any missing modern
    // tables first, then run incremental migrations against the old data.
    MemoryStorage::create_schema(&conn).unwrap();
    MemoryStorage::maybe_migrate(&conn).unwrap();

    // Verify the migrated edge has the correct owner (owner_bytes), not source_id
    let migrated_owner: Vec<u8> = conn
        .query_row("SELECT owner FROM knowledge_edges LIMIT 1", [], |r| {
            r.get(0)
        })
        .unwrap();

    assert_eq!(migrated_owner.len(), 32);
    assert_eq!(
        migrated_owner.as_slice(),
        owner_bytes.as_slice(),
        "Migrated edge owner should be record.owner ([0xBB;32]), not source_id ([0x01;32])"
    );
}
