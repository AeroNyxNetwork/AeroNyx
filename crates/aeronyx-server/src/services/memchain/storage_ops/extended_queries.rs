// [ARCH-SPLIT 2026-10-02]
// Stats, miner support, weights, overview, owner checks, and context recall.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl MemoryStorage {
    pub async fn stats(&self) -> StorageStats {
        let conn = self.conn.lock().await;
        let q = |sql: &str, p: &[&dyn rusqlite::ToSql]| -> u64 {
            conn.query_row(sql, p, |r| r.get::<_, i64>(0)).unwrap_or(0) as u64
        };
        let total = q("SELECT COUNT(*) FROM records", &[]);
        let active = q("SELECT COUNT(*) FROM records WHERE status=0", &[]);
        StorageStats {
            total_records: total,
            active_records: active,
            by_layer: LayerCounts {
                identity: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=0",
                    &[],
                ),
                knowledge: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=1",
                    &[],
                ),
                episode: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=2",
                    &[],
                ),
                archive: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=3",
                    &[],
                ),
            },
            content_bytes: q(
                "SELECT COALESCE(SUM(LENGTH(encrypted_content)),0) FROM records WHERE status=0",
                &[],
            ),
            records_with_embedding: q(
                "SELECT COUNT(*) FROM records WHERE status=0 AND embedding IS NOT NULL",
                &[],
            ),
            session_inserts: self.total_inserted(),
            session_rejects: self.total_rejected(),
        }
    }

    pub async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    pub async fn count_by_layer(&self, layer: MemoryLayer) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM records WHERE status=0 AND layer=?1",
            params![layer as u8 as i64],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as u64
    }

    pub async fn compact_episodes_to_archive(
        &self,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        let records = self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE owner=?1 AND status=0 AND layer=?2 ORDER BY timestamp ASC LIMIT ?3",
            params![
                owner.as_slice(),
                MemoryLayer::Episode as u8 as i64,
                limit as i64
            ],
        );
        if records.is_empty() {
            return records;
        }
        if conn.execute_batch("BEGIN TRANSACTION").is_err() {
            return Vec::new();
        }
        for r in &records {
            let now_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            if let Err(e) = conn.execute(
                "UPDATE records SET layer=?1, archived_at=?2 WHERE record_id=?3",
                params![
                    MemoryLayer::Archive as u8 as i64,
                    now_ts,
                    r.record_id.as_slice()
                ],
            ) {
                error!(error=%e, "[STORAGE] ❌ Compact update failed, rolling back");
                let _ = conn.execute_batch("ROLLBACK");
                return Vec::new();
            }
        }
        if conn.execute_batch("COMMIT").is_err() {
            return Vec::new();
        }
        info!(
            count = records.len(),
            "[STORAGE] ⛏️ Episodes compacted to Archive layer"
        );
        records
    }

    pub async fn get_records_needing_embedding(&self, limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE embedding IS NULL AND status = 0 LIMIT ?1",
            params![limit as i64],
        )
    }

    pub async fn get_correction_records(&self) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE topic_tags LIKE '%_correction%' AND status = 0",
            [],
        )
    }

    pub async fn update_topic_tags(&self, record_id: &[u8; 32], tags: &[String]) {
        let json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2",
            params![json, record_id.as_slice()],
        );
    }

    pub async fn supersede_record(&self, old_id: &[u8; 32], new_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        let r1 = conn.execute(
            "UPDATE records SET status = 1 WHERE record_id = ?1",
            params![old_id.as_slice()],
        );
        let r2 = conn.execute(
            "UPDATE records SET supersedes = ?1 WHERE record_id = ?2",
            params![old_id.as_slice(), new_id.as_slice()],
        );
        r1.is_ok() && r2.is_ok()
    }

    pub async fn load_user_weights(&self, owner: &[u8; 32]) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT weights FROM user_weights WHERE owner = ?1",
            params![owner.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .ok()?
    }

    pub async fn save_user_weights(&self, owner: &[u8; 32], weights_blob: &[u8], version: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO user_weights (owner, weights, version, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(owner) DO UPDATE SET weights=?2, version=?3, updated_at=?4",
            params![owner.as_slice(), weights_blob, version as i64, now],
        );
    }

    pub async fn has_active_content(&self, owner: &[u8; 32], content: &[u8]) -> bool {
        let compare_content: Vec<u8> = if let Some(ref key) = self.record_key {
            if content.is_empty() {
                content.to_vec()
            } else {
                match encrypt_record_content(key, content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        warn!("[STORAGE] has_active_content encryption failed: {}", e);
                        content.to_vec()
                    }
                }
            }
        } else {
            content.to_vec()
        };
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records WHERE owner = ?1 AND encrypted_content = ?2 AND status = 0 LIMIT 1",
            params![owner.as_slice(), compare_content.as_slice()],
            |row| row.get(0),
        ).unwrap_or(0);
        count > 0
    }

    pub async fn get_embedding_model(&self, record_id: &[u8; 32]) -> Option<String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT embedding_model FROM records WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .unwrap_or(None)
    }

    pub async fn get_overview(&self, owner: &[u8; 32], per_layer_limit: usize) -> OverviewData {
        let conn = self.conn.lock().await;
        let limit = per_layer_limit.min(50).max(1);
        let mut by_layer = HashMap::new();
        let layer_names = [
            (0i64, "identity"),
            (1, "knowledge"),
            (2, "episode"),
            (3, "archive"),
        ];
        for (layer_val, layer_name) in &layer_names {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM records WHERE owner = ?1 AND status = 0 AND layer = ?2",
                    params![owner.as_slice(), layer_val],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            by_layer.insert(layer_name.to_string(), count as u64);
        }
        let mut recent_by_layer = HashMap::new();
        let rk = self.record_key.as_ref().map(|v| &**v);
        for (layer_val, layer_name) in &layer_names {
            let mut stmt = match conn.prepare(
                "SELECT record_id, encrypted_content, topic_tags, timestamp,
                        access_count, positive_feedback, negative_feedback, source_ai
                 FROM records WHERE owner = ?1 AND layer = ?2 AND status = 0
                 ORDER BY timestamp DESC LIMIT ?3",
            ) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let records: Vec<OverviewRecord> = stmt
                .query_map(params![owner.as_slice(), layer_val, limit as i64], |row| {
                    let rid_blob: Vec<u8> = row.get(0)?;
                    let raw_content: Vec<u8> = row.get(1)?;
                    let tags_json: String = row.get(2)?;
                    let timestamp: i64 = row.get(3)?;
                    let access_count: i64 = row.get(4)?;
                    let pos_fb: i64 = row.get(5)?;
                    let neg_fb: i64 = row.get(6)?;
                    let source_ai: String = row.get(7)?;
                    let content = if let Some(key) = rk {
                        if raw_content.len() >= 28 {
                            match decrypt_record_content(key, &raw_content) {
                                Ok(plain) => String::from_utf8_lossy(&plain).to_string(),
                                Err(_) => String::from_utf8_lossy(&raw_content).to_string(),
                            }
                        } else {
                            String::from_utf8_lossy(&raw_content).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&raw_content).to_string()
                    };
                    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                    Ok(OverviewRecord {
                        record_id: hex::encode(&rid_blob),
                        content,
                        topic_tags: tags,
                        timestamp: timestamp as u64,
                        access_count: access_count as u32,
                        positive_feedback: pos_fb as u32,
                        negative_feedback: neg_fb as u32,
                        source_ai,
                    })
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();
            recent_by_layer.insert(layer_name.to_string(), records);
        }
        let last_memory_at: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(timestamp), 0) FROM records WHERE owner = ?1 AND status = 0",
                params![owner.as_slice()],
                |row| row.get(0),
            )
            .unwrap_or(0);
        OverviewData {
            by_layer,
            recent_by_layer,
            last_memory_at: last_memory_at as u64,
        }
    }

    pub async fn count_distinct_owners(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(DISTINCT owner) FROM records", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap_or(0) as usize
    }

    pub async fn owner_exists(&self, owner: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE owner = ?1 LIMIT 1)",
            params![owner.as_slice()],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
    }

    /// Get active records filtered by project_id (context isolation).
    ///
    /// Matches records via TWO paths:
    /// 1. `records.project_id = project_id` — direct tag (set by future /remember context param)
    /// 2. `records.session_id → sessions.project_id = project_id` — via session association
    ///    (set by /log handler when `context` field is provided, stored by upsert_session)
    ///
    /// Records inserted via `/remember` directly without a session use path 1 only.
    /// Records inserted via `/log` with `context` field use path 2.
    ///
    /// ## Usage
    /// Called by `recall_handler.rs` Step 4.1 when `RecallRequest.context` is
    /// a non-"all" value. The caller builds a HashSet of returned record_ids
    /// and filters `scored` with `retain()`.
    ///
    /// ## Performance
    /// LEFT JOIN on session_id. For best performance, ensure an index exists on
    /// `records.session_id` and `sessions.project_id`. With typical MemChain
    /// data sizes (< 100k records per user) this is fast without extra indexes.
    ///
    /// v2.5.3+Isolation
    pub async fn get_active_records_by_context(
        &self,
        owner: &[u8; 32],
        project_id: &str,
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;

        if let Some(l) = layer {
            self.query_rows(
                &conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with,
                        r.blind
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND r.layer = ?2
                   AND (r.project_id = ?3 OR s.project_id = ?3)
                 ORDER BY r.timestamp DESC
                 LIMIT ?4",
                params![owner.as_slice(), l as u8 as i64, project_id, limit as i64],
            )
        } else {
            self.query_rows(
                &conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with,
                        r.blind
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND (r.project_id = ?2 OR s.project_id = ?2)
                 ORDER BY r.timestamp DESC
                 LIMIT ?3",
                params![owner.as_slice(), project_id, limit as i64],
            )
        }
    }
}
