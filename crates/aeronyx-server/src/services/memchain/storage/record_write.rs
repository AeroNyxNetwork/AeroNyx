// [ARCH-SPLIT 2026-10-02]
// Insert, update, revoke, and blind-provenance writes.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl MemoryStorage {
    // ========================================
    // PATCH: Update record content (v2.5.2+Provenance)
    // ========================================

    /// Update a record's content, tags, layer, and/or source_ai (PATCH semantics).
    ///
    /// ## ⚠️ Security: single-lock ownership check (P0 SecAudit)
    /// Ownership check and UPDATE run inside the same `conn.lock()` to eliminate
    /// the TOCTOU window that existed when using two separate lock acquisitions.
    /// The UPDATE itself also carries `AND owner = ?owner AND status = 0` so a
    /// concurrent revoke between the check and the write cannot corrupt state.
    ///
    /// ## ⚠️ Encryption failure returns Err (P1 SecAudit)
    /// If `record_key` is set and encryption fails, the method returns an error
    /// rather than silently storing plaintext.
    ///
    /// ## ⚠️ Critical: embedding is cleared on content change
    /// When `new_content` is Some, the stored embedding is set to NULL so that
    /// the Miner can detect and re-embed the record on its next tick.
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn update_record_content(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        new_content: Option<&str>,
        new_tags: Option<&[String]>,
        new_layer: Option<MemoryLayer>,
        new_source_ai: Option<&str>,
    ) -> Result<bool, String> {
        // Encrypt new content before acquiring the lock (CPU work outside critical section)
        let stored_content: Option<Vec<u8>> = if let Some(text) = new_content {
            let bytes = text.as_bytes().to_vec();
            if let Some(ref key) = self.record_key {
                let key: &[u8; 32] = &**key;
                // P1: encryption failure returns Err — no silent plaintext fallback
                let ct = encrypt_record_content(key, &bytes)
                    .map_err(|e| format!("encrypt new content: {}", e))?;
                Some(ct)
            } else {
                Some(bytes)
            }
        } else {
            None
        };

        let tags_json: Option<String> =
            new_tags.map(|t| serde_json::to_string(t).unwrap_or_else(|_| "[]".to_string()));

        // P0 SecAudit: single lock — ownership check + all UPDATEs in one critical section.
        // Each UPDATE carries AND owner = ?owner (+ AND status = 0 for content) so a
        // concurrent revoke cannot produce an inconsistent write.
        let conn = self.conn.lock().await;

        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM records WHERE record_id = ?1 AND owner = ?2 AND status = 0",
                params![record_id.as_slice(), owner.as_slice()],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !exists {
            return Ok(false);
        }

        if stored_content.is_some() {
            let affected = conn
                .execute(
                    "UPDATE records SET
                    encrypted_content = ?1,
                    embedding = NULL,
                    embedding_model = '',
                    embedding_dim = 0
                 WHERE record_id = ?2 AND owner = ?3 AND status = 0",
                    params![
                        stored_content.as_deref(),
                        record_id.as_slice(),
                        owner.as_slice(),
                    ],
                )
                .map_err(|e| format!("update content: {}", e))?;
            if affected == 0 {
                return Ok(false);
            }
        }

        if let Some(ref tj) = tags_json {
            conn.execute(
                "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![tj, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update tags: {}", e))?;
        }

        if let Some(l) = new_layer {
            conn.execute(
                "UPDATE records SET layer = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![l as u8 as i64, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update layer: {}", e))?;
        }

        if let Some(src) = new_source_ai {
            conn.execute(
                "UPDATE records SET source_ai = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![src, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update source_ai: {}", e))?;
        }

        drop(conn); // release lock before cache write
        self.cache.write().invalidate(record_id);

        debug!(
            record_id = hex::encode(record_id),
            content_changed = new_content.is_some(),
            tags_changed = new_tags.is_some(),
            layer_changed = new_layer.is_some(),
            "[STORAGE] ✅ Record patched"
        );

        Ok(true)
    }

    // ========================================
    // Provenance: set session_id on a record (v2.5.2+Provenance)
    // ========================================

    /// Associate a record with the session it was extracted from.
    ///
    /// ## P0 SecAudit: owner verification added
    /// Without owner verification, any caller that knows a record_id could
    /// tamper with the provenance chain. SQL now carries AND owner = ?3.
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn set_record_session_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        session_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let result = conn.execute(
            "UPDATE records SET session_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![session_id, record_id.as_slice(), owner.as_slice()],
        );
        if let Err(e) = result {
            debug!(
                record_id = hex::encode(record_id),
                session_id = session_id,
                error = %e,
                "[STORAGE] set_record_session_id failed (non-fatal)"
            );
        }
    }

    /// Associate a record with the episode it was extracted from.
    ///
    /// ## P0 SecAudit: owner verification added (same rationale as set_record_session_id)
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn set_record_episode_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        episode_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET episode_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![episode_id, record_id.as_slice(), owner.as_slice()],
        );
    }

    /// Tag a record with a project (node-blind grouping). For blind records the
    /// client supplies `project_id` — a plaintext label or an opaque hash — so it
    /// can archive and recall memories by project via the existing `context`
    /// filter (`get_active_records_by_context`). Owner-scoped.
    pub async fn set_record_project_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        project_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET project_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![project_id, record_id.as_slice(), owner.as_slice()],
        );
    }

    // ========================================
    // Insert
    // ========================================

    pub async fn insert(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        if !record.verify_id() {
            warn!(
                record_id = hex::encode(record.record_id),
                "[STORAGE] ❌ Rejected: hash mismatch"
            );
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let tags_json =
            serde_json::to_string(&record.topic_tags).unwrap_or_else(|_| "[]".to_string());
        let embedding_blob: Option<Vec<u8>> = if record.has_embedding() {
            Some(embedding_to_bytes(&record.embedding))
        } else {
            None
        };
        let embedding_dim = record.embedding_dim() as i64;
        let conflict_with_blob: Option<Vec<u8>> = record.conflict_with.map(|c| c.to_vec());

        let stored_content: Vec<u8> = if record.blind {
            // Node-blind (Brick 1): the client already sealed this content with its
            // own key. Store it verbatim — the node must never encrypt it (it cannot
            // decrypt it back, and doing so would double-wrap the client ciphertext).
            record.encrypted_content.clone()
        } else if let Some(ref key) = self.record_key {
            let key: &[u8; 32] = &**key;
            if record.encrypted_content.is_empty() {
                record.encrypted_content.clone()
            } else {
                // P1 SecAudit: encryption failure returns false (rejects insert)
                // instead of silently storing plaintext.
                match encrypt_record_content(key, &record.encrypted_content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        error!(
                            record_id = hex::encode(record.record_id),
                            error = %e,
                            "[STORAGE] ❌ Record encryption failed — insert rejected"
                        );
                        self.total_rejected.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                }
            }
        } else {
            record.encrypted_content.clone()
        };

        let conn = self.conn.lock().await;
        let result = conn.execute(
            "INSERT OR IGNORE INTO records (
                record_id, owner, timestamp, layer, topic_tags, source_ai,
                status, supersedes, encrypted_content, embedding,
                embedding_model, embedding_dim, signature, access_count, created_at,
                positive_feedback, negative_feedback, conflict_with, blind
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![
                record.record_id.as_slice(),
                record.owner.as_slice(),
                record.timestamp as i64,
                record.layer as u8 as i64,
                tags_json,
                record.source_ai,
                record.status as u8 as i64,
                record.supersedes.as_ref().map(|s| s.as_slice()),
                stored_content.as_slice(),
                embedding_blob.as_deref(),
                embedding_model,
                embedding_dim,
                record.signature.as_slice(),
                record.access_count as i64,
                now,
                record.positive_feedback as i64,
                record.negative_feedback as i64,
                conflict_with_blob.as_deref(),
                record.blind as i64,
            ],
        );

        match result {
            Ok(changes) if changes > 0 => {
                self.total_inserted.fetch_add(1, Ordering::Relaxed);
                self.cache.write().put(record.clone());
                debug!(record_id = hex::encode(record.record_id), layer = %record.layer, "[STORAGE] ✅ Inserted");
                true
            }
            Ok(_) => {
                debug!(
                    record_id = hex::encode(record.record_id),
                    "[STORAGE] Duplicate, skipped"
                );
                false
            }
            Err(e) => {
                error!(record_id = hex::encode(record.record_id), error = %e, "[STORAGE] ❌ Insert failed");
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Store a node-blind record received from a peer as a **verbatim replica**
    /// (replication receive path, Brick 4). The content is opaque to this node,
    /// so it is stored exactly as received — the node **never** re-encrypts it
    /// with its own key (which would double-wrap the origin's ciphertext and make
    /// the replica unreadable). `insert()`'s `verify_id()` still guards content
    /// integrity. The caller MUST have already verified the origin's Ed25519
    /// signature (same as the P2P `BroadcastRecord` path). Returns true if stored.
    pub async fn insert_blind_replica(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        let mut replica = record.clone();
        replica.blind = true;
        self.insert(&replica, embedding_model).await
    }

    // ========================================
    // Lifecycle
    // ========================================

    pub async fn update_status(&self, record_id: &[u8; 32], new_status: RecordStatus) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1 WHERE record_id=?2",
            params![new_status as u8 as i64, record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(record_id=hex::encode(record_id), %new_status, "[STORAGE] ✅ Status updated");
                true
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] Not found for status update"
                );
                false
            }
            Err(e) => {
                error!(error=%e, "[STORAGE] Status update failed");
                false
            }
        }
    }

    pub async fn revoke(&self, record_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1, encrypted_content=x'', embedding=NULL WHERE record_id=?2",
            params![RecordStatus::Revoked as u8 as i64, record_id.as_slice()]) {
            Ok(n) if n > 0 => {
                self.cache.write().invalidate(record_id);
                info!(record_id=hex::encode(record_id), "[STORAGE] 🗑️ Revoked"); true
            }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for revoke"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Revoke failed"); false }
        }
    }

    pub async fn increment_access(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET access_count=access_count+1 WHERE record_id=?1",
            params![record_id.as_slice()],
        );
    }

    /// Store node-blind derived-record provenance (Brick 3d): link a derived
    /// record (e.g. a client/LLM-produced summary) to the source records it was
    /// built from. `sources` are opaque record_id hexes; the node learns only the
    /// provenance DAG shape, never the content. Self-links are ignored.
    pub async fn insert_blind_provenance(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        sources: &[String],
    ) {
        if sources.is_empty() {
            return;
        }
        let rid_hex = hex::encode(record_id);
        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;
        for src in sources {
            if src.is_empty() || src == &rid_hex {
                continue;
            }
            let _ = conn.execute(
                "INSERT OR IGNORE INTO blind_provenance (record_id, source_id, owner_hex)
                 VALUES (?1, ?2, ?3)",
                params![rid_hex, src, owner_hex],
            );
        }
    }
}
