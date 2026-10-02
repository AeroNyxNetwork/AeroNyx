// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

// ========================================
// Existing tests (preserved)
// ========================================

#[tokio::test]
async fn test_open_in_memory() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    assert_eq!(s.count().await, 0);
}

// ========================================
// v2.4.0: Schema v5 Tests
// ========================================

#[tokio::test]
async fn test_schema_version_is_current() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;
    let v: u32 = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        v, SCHEMA_VERSION,
        "Schema version should be {}",
        SCHEMA_VERSION
    );
}

#[tokio::test]
async fn test_v6_cognitive_tasks_schema() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let result = conn.execute(
        "INSERT INTO cognitive_tasks
                (task_type, priority, status, payload, target_table, target_id,
                 privacy_level, created_at, max_retries)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            "session_title",
            5i64,
            "pending",
            r#"{"session_id":"sess_001","summary":"JWT auth discussion"}"#,
            "sessions",
            "sess_001",
            "structured",
            now,
            3i64,
        ],
    );
    assert!(
        result.is_ok(),
        "cognitive_tasks insert should succeed: {:?}",
        result.err()
    );

    let claimed = conn
        .execute(
            "UPDATE cognitive_tasks SET status='processing', started_at=?1
             WHERE id = (
                 SELECT id FROM cognitive_tasks
                 WHERE status='pending'
                 ORDER BY priority DESC, created_at ASC
                 LIMIT 1
             )",
            params![now],
        )
        .unwrap();
    assert_eq!(claimed, 1);

    let status: String = conn
        .query_row(
            "SELECT status FROM cognitive_tasks WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(status, "processing");
}

#[tokio::test]
async fn test_v6_llm_usage_log_schema() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO llm_usage_log
                (task_id, provider, model, input_tokens, output_tokens,
                 cached_tokens, latency_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            1i64,
            "deepseek",
            "deepseek-reasoner",
            512i64,
            128i64,
            64i64,
            1200i64,
            now
        ],
    )
    .unwrap();

    let (input_sum, output_sum): (i64, i64) = conn.query_row(
        "SELECT SUM(input_tokens), SUM(output_tokens) FROM llm_usage_log WHERE provider = 'deepseek'",
        [], |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(input_sum, 512);
    assert_eq!(output_sum, 128);
}

#[tokio::test]
async fn test_v5_episodes_table_schema() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let result = conn.execute(
        "INSERT INTO episodes (episode_id, owner, episode_type, source,
                session_id, encrypted_content, content_hash, token_count,
                created_at, ingested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            "ep_001",
            [0xAAu8; 32].as_slice(),
            "conversation",
            "test",
            "session_001",
            b"encrypted".as_slice(),
            "hash123",
            100i64,
            now,
            now,
        ],
    );
    assert!(
        result.is_ok(),
        "Episode insert should succeed: {:?}",
        result.err()
    );

    let ep_type: String = conn
        .query_row(
            "SELECT episode_type FROM episodes WHERE episode_id = 'ep_001'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(ep_type, "conversation");
}

#[tokio::test]
async fn test_v5_entities_table_schema() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let conn = s.conn.lock().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let result = conn.execute(
        "INSERT INTO entities (entity_id, owner, name, name_normalized,
                entity_type, description, community_id, created_at, updated_at, mention_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            "ent_jwt",
            [0xAAu8; 32].as_slice(),
            "JWT",
            "jwt",
            "technology",
            "JSON Web Token",
            Option::<String>::None,
            now,
            now,
            3i64,
        ],
    );
    assert!(
        result.is_ok(),
        "Entity insert should succeed: {:?}",
        result.err()
    );

    let mention_count: i64 = conn
        .query_row(
            "SELECT mention_count FROM entities WHERE entity_id = 'ent_jwt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(mention_count, 3);
}
