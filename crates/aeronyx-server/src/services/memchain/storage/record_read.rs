// [ARCH-SPLIT 2026-10-02]
// Record lookups, owner and session indexes, embeddings, and row decoding.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::ledger::record::{
    memory_sealed_v2_record_id, memory_sealed_v2_signature_transcript, MemorySealedV2Envelope,
};

// [MEMORY-SEALED-V2 2026-10-02 by Codex] The read model contains only opaque
// envelope bytes and owner-scoped lifecycle data. It is never fed to legacy
// semantic, vector, FTS, or graph queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedV2Row {
    pub record_id: [u8; 32],
    pub owner: [u8; 32],
    pub created_at: u64,
    pub envelope: Vec<u8>,
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedV2Page {
    pub rows: Vec<SealedV2Row>,
    pub next_cursor: Option<[u8; 32]>,
}

fn validate_sealed_v2_row(
    record_id: &[u8; 32],
    owner: &[u8; 32],
    created_at: u64,
    envelope: &[u8],
    signature: &[u8; 64],
) -> Result<(), ()> {
    let envelope_view = MemorySealedV2Envelope::decode(envelope).map_err(|_| ())?;
    if memory_sealed_v2_record_id(owner, created_at, envelope_view.as_bytes()) != *record_id {
        return Err(());
    }
    let transcript = memory_sealed_v2_signature_transcript(
        owner,
        record_id,
        created_at,
        envelope_view.as_bytes(),
    );
    IdentityPublicKey::from_bytes(owner)
        .and_then(|key| key.verify(&transcript, signature))
        .map_err(|_| ())
}

// ============================================
// Embedding helpers
// ============================================

pub(crate) fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            f32::from_le_bytes(buf)
        })
        .collect()
}

impl MemoryStorage {
    /// Node-blind project scope: the set of the owner's active record_ids
    /// archived under `project_id`. Cheap ids-only lookup (served by
    /// `idx_records_project`) used by recall to constrain node-blind FTS / graph
    /// hits to a project, since `MemoryRecord` does not carry `project_id`.
    pub async fn project_record_ids(
        &self,
        owner: &[u8; 32],
        project_id: &str,
    ) -> std::collections::HashSet<[u8; 32]> {
        let mut set = std::collections::HashSet::new();
        let conn = self.conn.lock().await;
        if let Ok(mut stmt) = conn
            .prepare("SELECT record_id FROM records WHERE owner=?1 AND project_id=?2 AND status=0")
        {
            if let Ok(rows) = stmt.query_map(params![owner.as_slice(), project_id], |row| {
                row.get::<_, Vec<u8>>(0)
            }) {
                for r in rows.flatten() {
                    if r.len() == 32 {
                        let mut a = [0u8; 32];
                        a.copy_from_slice(&r);
                        set.insert(a);
                    }
                }
            }
        }
        set
    }

    /// [D6 ATTEST] Node-blind: all of the owner's ACTIVE record_ids (status=0),
    /// returned SORTED for a deterministic storage-root commitment. The node
    /// reads only opaque content-address hashes here (never plaintext), so
    /// signing a root over this set preserves blindness while giving the client
    /// verifiable, tamper-evident proof of exactly which records the node holds.
    pub async fn owner_record_ids(&self, owner: &[u8; 32]) -> Result<Vec<[u8; 32]>, String> {
        let mut ids: Vec<[u8; 32]> = Vec::new();
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT record_id FROM records WHERE owner=?1 AND status=0")
            .map_err(|error| format!("record id query: {error}"))?;
        let rows = stmt
            .query_map(params![owner.as_slice()], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|error| format!("record id rows: {error}"))?;
        for r in rows {
            let r = r.map_err(|error| format!("record id row: {error}"))?;
            let a = r
                .try_into()
                .map_err(|_| "invalid legacy record id".to_string())?;
            ids.push(a);
        }

        let mut stmt = conn
            .prepare(
                "SELECT record_id, owner, created_at, envelope, signature
                 FROM memory_sealed_v2 WHERE owner=?1 AND status=0",
            )
            .map_err(|error| format!("sealed v2 id query: {error}"))?;
        let rows = stmt
            .query_map(params![owner.as_slice()], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                ))
            })
            .map_err(|error| format!("sealed v2 id rows: {error}"))?;
        for row in rows {
            let (record_id, row_owner, created_at, envelope, signature) =
                row.map_err(|error| format!("sealed v2 id row: {error}"))?;
            let record_id: [u8; 32] = record_id
                .try_into()
                .map_err(|_| "invalid sealed v2 record id".to_string())?;
            let row_owner: [u8; 32] = row_owner
                .try_into()
                .map_err(|_| "invalid sealed v2 owner".to_string())?;
            let created_at =
                u64::try_from(created_at).map_err(|_| "invalid sealed v2 timestamp".to_string())?;
            let signature: [u8; 64] = signature
                .try_into()
                .map_err(|_| "invalid sealed v2 signature".to_string())?;
            validate_sealed_v2_row(&record_id, &row_owner, created_at, &envelope, &signature)
                .map_err(|_| "invalid sealed v2 durable row".to_string())?;
            ids.push(record_id);
        }
        ids.sort_unstable();
        Ok(ids)
    }

    pub async fn get_sealed_v2(
        &self,
        owner: &[u8; 32],
        record_id: &[u8; 32],
    ) -> Result<Option<SealedV2Row>, String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT record_id, owner, created_at, envelope, signature
             FROM memory_sealed_v2 WHERE record_id=?1 AND owner=?2 AND status=0",
            params![record_id.as_slice(), owner.as_slice()],
            |row| {
                let rid: Vec<u8> = row.get(0)?;
                let own: Vec<u8> = row.get(1)?;
                let sig: Vec<u8> = row.get(4)?;
                let record_id = rid.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let owner = own.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let signature = sig.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let created_at = u64::try_from(row.get::<_, i64>(2)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                let envelope: Vec<u8> = row.get(3)?;
                validate_sealed_v2_row(&record_id, &owner, created_at, &envelope, &signature)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                Ok(SealedV2Row {
                    record_id,
                    owner,
                    created_at,
                    envelope,
                    signature,
                })
            },
        )
        .optional()
        .map_err(|error| format!("sealed v2 read: {error}"))
    }

    pub async fn list_sealed_v2(
        &self,
        owner: &[u8; 32],
        after: Option<&[u8; 32]>,
        limit: usize,
    ) -> Result<SealedV2Page, String> {
        let limit = limit.clamp(1, 100);
        let conn = self.conn.lock().await;
        let mut rows = Vec::new();
        let mut stmt = match conn.prepare(
            "SELECT record_id, owner, created_at, envelope, signature
             FROM memory_sealed_v2
             WHERE owner=?1 AND status=0 AND (?2 IS NULL OR record_id>?2)
             ORDER BY record_id ASC LIMIT ?3",
        ) {
            Ok(stmt) => stmt,
            Err(error) => return Err(format!("sealed v2 list prepare: {error}")),
        };
        let after_blob = after.map(|v| v.as_slice());
        let mapped = stmt.query_map(
            params![owner.as_slice(), after_blob, (limit + 1) as i64],
            |row| {
                let rid: Vec<u8> = row.get(0)?;
                let own: Vec<u8> = row.get(1)?;
                let sig: Vec<u8> = row.get(4)?;
                let record_id = rid.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let owner = own.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let signature = sig.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
                let created_at = u64::try_from(row.get::<_, i64>(2)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                let envelope: Vec<u8> = row.get(3)?;
                validate_sealed_v2_row(&record_id, &owner, created_at, &envelope, &signature)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                Ok(SealedV2Row {
                    record_id,
                    owner,
                    created_at,
                    envelope,
                    signature,
                })
            },
        );
        let mapped = mapped.map_err(|error| format!("sealed v2 list: {error}"))?;
        for row in mapped {
            rows.push(row.map_err(|error| format!("sealed v2 row: {error}"))?);
        }
        let next_cursor = if rows.len() > limit {
            let cursor = rows[limit - 1].record_id;
            rows.truncate(limit);
            Some(cursor)
        } else {
            None
        };
        Ok(SealedV2Page { rows, next_cursor })
    }

    // ========================================
    // Provenance: find records by session (v2.5.2+Provenance)
    // ========================================

    /// Get all active records extracted from a specific session.
    ///
    /// Used by the `/record/:id/provenance` endpoint and by
    /// find_records_by_content() to scope searches to a session.
    ///
    /// v2.5.2+Provenance
    pub async fn get_records_for_session(
        &self,
        session_id: &str,
        owner: &[u8; 32],
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,blind
             FROM records
             WHERE session_id = ?1 AND owner = ?2 AND status = 0
             ORDER BY timestamp ASC",
            params![session_id, owner.as_slice()],
        )
    }

    /// Find active records whose plaintext content contains the given substring.
    ///
    /// ## P2 SecAudit: pub(crate) + hard limit
    /// Exposed as pub(crate) only — not callable from external crates.
    /// Hard-capped at 100 regardless of caller-supplied limit to prevent
    /// O(n) scan DoS (each record requires a decryption + string match).
    /// For production search, prefer bm25_search (FTS5).
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub(crate) async fn find_records_by_content(
        &self,
        owner: &[u8; 32],
        content_substring: &str,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        if content_substring.trim().is_empty() {
            return Vec::new();
        }
        let all = self.get_active_records(owner, None, limit * 10).await;
        let needle = content_substring.to_lowercase();

        all.into_iter()
            .filter(|r| {
                let text = String::from_utf8_lossy(&r.encrypted_content);
                text.to_lowercase().contains(&needle)
            })
            .take(limit)
            .collect()
    }

    /// Get full provenance chain for a record.
    ///
    /// Returns:
    /// - The record itself
    /// - session_id (from records.session_id column)
    /// - session metadata (title, started_at)
    /// - turn_index hint (best-effort: LENGTH heuristic, may be inaccurate)
    ///
    /// ## ⚠️ turn_index accuracy (BUG FIX v2.5.2+Provenance)
    /// The turn_index lookup uses `ABS(LENGTH(CAST(content AS TEXT)) - ?)` as a
    /// heuristic to find the closest-length turn. This is unreliable:
    ///   - For encrypted raw_logs, CAST gives BLOB hex length, not plaintext length.
    ///   - Multiple turns may have the same content length (e.g. short turns).
    ///   - `content_text` from `record.encrypted_content` is already decrypted
    ///     (see find_records_by_content note), so the length comparison is
    ///     against plaintext vs potentially encrypted BLOB.
    /// Treat turn_index as advisory — it is not guaranteed to be the source turn.
    /// A future v7 improvement: store turn_index directly in records.session_turn_index.
    ///
    /// v2.5.2+Provenance
    pub async fn get_record_provenance(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
    ) -> Option<RecordProvenance> {
        let record = self.get(record_id).await?;
        if record.owner != *owner {
            return None;
        }

        let conn = self.conn.lock().await;

        // Get session_id from the records table (provenance field)
        let session_id: Option<String> = conn
            .query_row(
                "SELECT session_id FROM records WHERE record_id = ?1",
                params![record_id.as_slice()],
                |r| r.get(0),
            )
            .ok()
            .flatten();

        // Get session metadata if session_id is known
        let (session_title, session_started_at) = if let Some(ref sid) = session_id {
            let meta: Option<(Option<String>, i64)> = conn
                .query_row(
                    "SELECT title, started_at FROM sessions WHERE session_id = ?1",
                    params![sid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ok();
            match meta {
                Some((title, started_at)) => (title, Some(started_at)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

        // Best-effort turn_index: find the raw_log turn whose content length is
        // closest to the record's content length (LENGTH heuristic).
        // ⚠️ Advisory only — see method doc comment for accuracy limitations.
        let turn_index: Option<i64> = if let Some(ref sid) = session_id {
            let content_len = record.encrypted_content.len() as i64;
            conn.query_row(
                "SELECT turn_index FROM raw_logs
                 WHERE session_id = ?1
                 ORDER BY ABS(LENGTH(content) - ?2) ASC
                 LIMIT 1",
                params![sid, content_len],
                |r| r.get(0),
            )
            .ok()
        } else {
            None
        };

        Some(RecordProvenance {
            record_id: hex::encode(record_id),
            session_id,
            session_title,
            session_started_at,
            turn_index,
            layer: record.layer.to_string(),
            topic_tags: record.topic_tags.clone(),
            extracted_at: record.timestamp,
            source_ai: record.source_ai.clone(),
        })
    }

    // ========================================
    // Query
    // ========================================

    pub async fn get(&self, record_id: &[u8; 32]) -> Option<MemoryRecord> {
        {
            let mut cache = self.cache.write();
            if let Some(record) = cache.get(record_id) {
                return Some(record.clone());
            }
        }
        let conn = self.conn.lock().await;
        let rk = self.record_key.as_ref().map(|v| &**v);
        let result = conn
            .query_row(
                Self::SELECT_RECORD_COLS,
                params![record_id.as_slice()],
                |row| Self::row_to_record(row, rk),
            )
            .optional()
            .unwrap_or_else(|e| {
                error!(error=%e, "[STORAGE] Query failed");
                None
            });

        if let Some(ref record) = result {
            self.cache.write().put(record.clone());
        }
        result
    }

    pub async fn get_active_records(
        &self,
        owner: &[u8; 32],
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;
        if let Some(l) = layer {
            self.query_rows(
                &conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with,blind
                 FROM records WHERE owner=?1 AND status=0 AND layer=?2
                 ORDER BY timestamp DESC LIMIT ?3",
                params![owner.as_slice(), l as u8 as i64, limit as i64],
            )
        } else {
            self.query_rows(
                &conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with,blind
                 FROM records WHERE owner=?1 AND status=0
                 ORDER BY timestamp DESC LIMIT ?2",
                params![owner.as_slice(), limit as i64],
            )
        }
    }

    pub async fn query_by_owner_after(
        &self,
        owner: &[u8; 32],
        after_timestamp: u64,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,blind
             FROM records WHERE owner=?1 AND timestamp>?2
             ORDER BY timestamp ASC LIMIT ?3",
            params![
                owner.as_slice(),
                after_timestamp as i64,
                DEFAULT_PAGE_SIZE as i64
            ],
        )
    }

    /// Returns active node-blind records not yet represented in the local
    /// commitment chain.
    ///
    /// This is the only source used by the V1 block packer. Node-readable
    /// records are deliberately excluded so block production cannot turn the
    /// synchronised ledger into a plaintext or owner-metadata replication
    /// channel. The caller must still verify each record's content address and
    /// Ed25519 owner signature before building a block.
    pub async fn get_uncommitted_blind_records(&self, limit: usize) -> Vec<MemoryRecord> {
        let limit = limit.clamp(1, MAX_RECORD_COMMITMENTS_PER_BLOCK);
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT r.record_id,r.owner,r.timestamp,r.layer,r.topic_tags,r.source_ai,
                    r.status,r.supersedes,r.encrypted_content,r.embedding,r.signature,r.access_count,
                    r.positive_feedback,r.negative_feedback,r.conflict_with,r.blind
             FROM records r
             LEFT JOIN record_block_commitments c ON c.record_id = r.record_id
             WHERE r.blind=1 AND r.status=0 AND c.record_id IS NULL
             ORDER BY r.record_id ASC LIMIT ?1",
            params![limit as i64],
        )
    }

    pub async fn get_records_with_embedding(
        &self,
        owner: &[u8; 32],
    ) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,embedding_model,blind
             FROM records WHERE owner=?1 AND status=0 AND embedding IS NOT NULL
             ORDER BY timestamp DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map(params![owner.as_slice()], |row| {
            let record = Self::row_to_record(row, rk)?;
            let model: String = row.get(15)?;
            Ok((record, model))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Return every active record that has a persisted embedding.
    ///
    /// This is intentionally separate from `get_records_with_embedding(owner)`:
    /// normal Local-mode nodes must keep the historical single-owner rebuild,
    /// while node-blind/remote storage nodes share one SQLite database across
    /// authenticated owners and therefore need to restore every owner/model
    /// partition after process restart.
    ///
    /// # Security
    /// The returned records remain partitioned by `record.owner` when inserted
    /// into `VectorIndex`; recall still supplies an authenticated owner and can
    /// never search another owner's partition. Blind record content also stays
    /// opaque because `row_to_record` never decrypts rows marked `blind`.
    pub async fn get_all_records_with_embedding(&self) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,embedding_model,blind
             FROM records WHERE status=0 AND embedding IS NOT NULL
             ORDER BY owner ASC, timestamp DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare all-owner embedding rebuild failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map([], |row| {
            let record = Self::row_to_record(row, rk)?;
            let model: String = row.get(15)?;
            Ok((record, model))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Return the source record_id hexes a derived record was built from.
    pub async fn get_blind_provenance(&self, record_id: &[u8; 32]) -> Vec<String> {
        let rid_hex = hex::encode(record_id);
        let conn = self.conn.lock().await;
        let mut stmt =
            match conn.prepare("SELECT source_id FROM blind_provenance WHERE record_id = ?1") {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
        stmt.query_map(params![rid_hex], |row| row.get::<_, String>(0))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn query_rows(
        &self,
        conn: &Connection,
        sql: &str,
        p: impl rusqlite::Params,
    ) -> Vec<MemoryRecord> {
        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map(p, |row| Self::row_to_record(row, rk))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn row_to_record(
        row: &rusqlite::Row<'_>,
        record_key: Option<&[u8; 32]>,
    ) -> rusqlite::Result<MemoryRecord> {
        let record_id_blob: Vec<u8> = row.get(0)?;
        let owner_blob: Vec<u8> = row.get(1)?;
        let timestamp: i64 = row.get(2)?;
        let layer_val: i64 = row.get(3)?;
        let tags_json: String = row.get(4)?;
        let source_ai: String = row.get(5)?;
        let status_val: i64 = row.get(6)?;
        let supersedes_blob: Option<Vec<u8>> = row.get(7)?;
        let encrypted_content: Vec<u8> = row.get(8)?;
        let embedding_blob: Option<Vec<u8>> = row.get(9)?;

        // Node-blind marker (Brick 1). Read BY NAME so it is robust to column
        // position across the various record SELECTs; if the column is not in the
        // result set, default to 0 (sighted).
        let blind: bool = row.get::<&str, i64>("blind").unwrap_or(0) != 0;

        let decrypted_content = if blind {
            // Node-blind record: client-sealed with a key the node does not have.
            // Return the ciphertext verbatim — never attempt to decrypt it.
            encrypted_content
        } else if let Some(key) = record_key {
            if encrypted_content.len() >= 28 {
                match decrypt_record_content(key, &encrypted_content) {
                    Ok(plain) => plain,
                    Err(_) => encrypted_content,
                }
            } else {
                encrypted_content
            }
        } else {
            encrypted_content
        };

        let signature_blob: Vec<u8> = row.get(10)?;
        let access_count: i64 = row.get(11)?;
        let positive_feedback: i64 = row.get(12).unwrap_or(0);
        let negative_feedback: i64 = row.get(13).unwrap_or(0);
        let conflict_with_blob: Option<Vec<u8>> = row.get(14).unwrap_or(None);

        let mut record_id = [0u8; 32];
        if record_id_blob.len() == 32 {
            record_id.copy_from_slice(&record_id_blob);
        } else {
            // P3 SecAudit: return Err instead of silently using zero-ID.
            // A zero-ID record would corrupt the cache (multiple broken records
            // share the same key) and pollute search results.
            return Err(rusqlite::Error::InvalidColumnType(
                0,
                "record_id".into(),
                rusqlite::types::Type::Blob,
            ));
        }
        let mut owner = [0u8; 32];
        if owner_blob.len() == 32 {
            owner.copy_from_slice(&owner_blob);
        } else {
            return Err(rusqlite::Error::InvalidColumnType(
                1,
                "owner".into(),
                rusqlite::types::Type::Blob,
            ));
        }
        let mut signature = [0u8; 64];
        if signature_blob.len() == 64 {
            signature.copy_from_slice(&signature_blob);
        } else {
            return Err(rusqlite::Error::InvalidColumnType(
                10,
                "signature".into(),
                rusqlite::types::Type::Blob,
            ));
        }

        let supersedes = supersedes_blob.and_then(|b| {
            if b.len() == 32 {
                let mut a = [0u8; 32];
                a.copy_from_slice(&b);
                Some(a)
            } else {
                None
            }
        });
        let conflict_with = conflict_with_blob.and_then(|b| {
            if b.len() == 32 {
                let mut a = [0u8; 32];
                a.copy_from_slice(&b);
                Some(a)
            } else {
                None
            }
        });
        let embedding = embedding_blob
            .map(|b| bytes_to_embedding(&b))
            .unwrap_or_default();

        Ok(MemoryRecord {
            record_id,
            owner,
            timestamp: timestamp as u64,
            layer: MemoryLayer::from_u8(layer_val as u8).unwrap_or(MemoryLayer::Episode),
            topic_tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            source_ai,
            status: RecordStatus::from_u8(status_val as u8).unwrap_or(RecordStatus::Active),
            supersedes,
            encrypted_content: decrypted_content,
            embedding,
            signature,
            access_count: access_count as u32,
            positive_feedback: positive_feedback as u32,
            negative_feedback: negative_feedback as u32,
            conflict_with,
            blind,
        })
    }
}
