// [ARCH-SPLIT 2026-10-02]
// Insert, update, revoke, and blind-provenance writes.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::ledger::record::{
    memory_sealed_v2_record_id, memory_sealed_v2_signature_transcript, MemorySealedV2Envelope,
    MEMORY_SEALED_V2_MAX_ENVELOPE_BYTES, MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES,
};

// [MEMORY-SEALED-V2 2026-10-02 by Codex] Durable outcomes are deliberately
// coarse; callers never receive a row or any decrypted material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealedV2InsertOutcome {
    Inserted,
    ExactDuplicate,
    Conflict,
}

impl SealedV2InsertOutcome {
    pub(crate) const fn is_inserted(self) -> bool {
        matches!(self, Self::Inserted)
    }

    pub(crate) const fn is_exact_duplicate(self) -> bool {
        matches!(self, Self::ExactDuplicate)
    }
}

// [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] The owner ceiling is checked
// inside the same IMMEDIATE transaction as the first row for that owner.  All
// lifecycle states count, so revocation cannot release a slot.
fn enforce_owner_slot_tx(
    tx: &rusqlite::Transaction<'_>,
    owner: &[u8; 32],
    policy: OwnerSlotPolicy,
) -> Result<(), OwnerSlotAdmissionError> {
    if policy.max_remote_owners == 0 || *owner == policy.local_owner {
        return Ok(());
    }
    let owner_exists: bool = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM (
                 SELECT owner FROM records
                 UNION
                 SELECT owner FROM memory_sealed_v2
             ) WHERE owner = ?1)",
            params![owner.as_slice()],
            |row| row.get(0),
        )
        .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
    if owner_exists {
        return Ok(());
    }
    let remote_count: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM (
                 SELECT owner FROM records
                 UNION
                 SELECT owner FROM memory_sealed_v2
             ) WHERE owner != ?1",
            params![policy.local_owner.as_slice()],
            |row| row.get(0),
        )
        .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
    let remote_count =
        usize::try_from(remote_count).map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
    if remote_count >= policy.max_remote_owners {
        return Err(OwnerSlotAdmissionError::AtCapacity);
    }
    Ok(())
}

// [MEMORY-SEALED-V2 2026-10-02 by Codex] Classification is shared by the
// read-only admission preflight and the insert transaction.  It scans every
// lifecycle state so tombstones cannot be resurrected by an exact retry.
fn classify_sealed_v2_conn(
    conn: &rusqlite::Connection,
    record_id: &[u8; 32],
    owner: &[u8; 32],
    created_at: u64,
    envelope: &[u8],
    signature: &[u8; 64],
) -> Result<Option<SealedV2InsertOutcome>, String> {
    let existing: Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>, i64)> = conn
        .query_row(
            "SELECT owner, created_at, envelope, signature, status
             FROM memory_sealed_v2 WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()
        .map_err(|e| format!("sealed v2 lookup: {e}"))?;
    let Some((stored_owner, stored_created, stored_envelope, stored_sig, status)) = existing else {
        return Ok(None);
    };
    let stored_owner: [u8; 32] = stored_owner
        .try_into()
        .map_err(|_| "sealed v2 owner corruption".to_string())?;
    let stored_created =
        u64::try_from(stored_created).map_err(|_| "sealed v2 timestamp corruption".to_string())?;
    let stored_signature: [u8; 64] = stored_sig
        .try_into()
        .map_err(|_| "sealed v2 signature corruption".to_string())?;
    let stored_envelope_view = MemorySealedV2Envelope::decode(&stored_envelope)
        .map_err(|_| "sealed v2 envelope corruption".to_string())?;
    if memory_sealed_v2_record_id(
        &stored_owner,
        stored_created,
        stored_envelope_view.as_bytes(),
    ) != *record_id
        || IdentityPublicKey::from_bytes(&stored_owner)
            .and_then(|key| {
                key.verify(
                    &memory_sealed_v2_signature_transcript(
                        &stored_owner,
                        record_id,
                        stored_created,
                        stored_envelope_view.as_bytes(),
                    ),
                    &stored_signature,
                )
            })
            .is_err()
    {
        return Err("sealed v2 row integrity failure".to_string());
    }
    if status == 0
        && stored_owner == *owner
        && stored_created == created_at
        && stored_envelope == envelope
        && stored_signature == *signature
    {
        Ok(Some(SealedV2InsertOutcome::ExactDuplicate))
    } else {
        Ok(Some(SealedV2InsertOutcome::Conflict))
    }
}

impl MemoryStorage {
    // [MEMORY-SEALED-V2 2026-10-02 by Codex] Read-only preflight deliberately
    // releases the SQLite mutex before quota admission can await.
    pub(crate) async fn classify_sealed_v2(
        &self,
        owner: &[u8; 32],
        record_id: &[u8; 32],
        created_at: u64,
        envelope: &[u8],
        signature: &[u8; 64],
    ) -> Result<Option<SealedV2InsertOutcome>, String> {
        let conn = self.conn.lock().await;
        classify_sealed_v2_conn(&conn, record_id, owner, created_at, envelope, signature)
    }

    /// Atomically insert one owner-authenticated opaque V2 envelope.
    /// Exact byte retries are idempotent; a reused id with different bytes is
    /// a conflict. No legacy projections are touched.
    pub async fn insert_sealed_v2(
        &self,
        owner: &[u8; 32],
        record_id: &[u8; 32],
        created_at: u64,
        envelope: &[u8],
        signature: &[u8; 64],
    ) -> Result<SealedV2InsertOutcome, String> {
        if created_at > i64::MAX as u64
            || !(MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES..=MEMORY_SEALED_V2_MAX_ENVELOPE_BYTES)
                .contains(&envelope.len())
            || MemorySealedV2Envelope::decode(envelope).is_err()
        {
            return Err("invalid sealed v2 row".to_string());
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "clock unavailable".to_string())?
            .as_secs();
        let now = i64::try_from(now).map_err(|_| "clock overflow".to_string())?;
        let conn = self.conn.lock().await;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("sealed v2 transaction: {e}"))?;
        let outcome = if let Some(outcome) =
            classify_sealed_v2_conn(&tx, record_id, owner, created_at, envelope, signature)?
        {
            outcome
        } else {
            tx.execute(
                "INSERT INTO memory_sealed_v2
                 (record_id, owner, created_at, envelope, signature, status, inserted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
                params![
                    record_id.as_slice(),
                    owner.as_slice(),
                    created_at as i64,
                    envelope,
                    signature.as_slice(),
                    now,
                ],
            )
            .map_err(|e| format!("sealed v2 insert: {e}"))?;
            SealedV2InsertOutcome::Inserted
        };
        tx.commit().map_err(|e| format!("sealed v2 commit: {e}"))?;
        Ok(outcome)
    }

    // [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] Remote API insertion uses a
    // separate typed entry point so local/replication callers retain their
    // historical behavior.  Duplicate/conflict classification precedes the
    // owner ceiling; only a genuinely new owner can consume a slot.
    pub(crate) async fn insert_sealed_v2_with_owner_slot(
        &self,
        owner: &[u8; 32],
        record_id: &[u8; 32],
        created_at: u64,
        envelope: &[u8],
        signature: &[u8; 64],
        policy: OwnerSlotPolicy,
    ) -> Result<SealedV2InsertOutcome, OwnerSlotAdmissionError> {
        if created_at > i64::MAX as u64
            || !(MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES..=MEMORY_SEALED_V2_MAX_ENVELOPE_BYTES)
                .contains(&envelope.len())
            || MemorySealedV2Envelope::decode(envelope).is_err()
        {
            return Err(OwnerSlotAdmissionError::StorageUnavailable);
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?
            .as_secs();
        let now = i64::try_from(now).map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
        let outcome =
            match classify_sealed_v2_conn(&tx, record_id, owner, created_at, envelope, signature) {
                Ok(Some(outcome)) => outcome,
                Ok(None) => {
                    enforce_owner_slot_tx(&tx, owner, policy)?;
                    tx.execute(
                        "INSERT INTO memory_sealed_v2
                     (record_id, owner, created_at, envelope, signature, status, inserted_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
                        params![
                            record_id.as_slice(),
                            owner.as_slice(),
                            created_at as i64,
                            envelope,
                            signature.as_slice(),
                            now,
                        ],
                    )
                    .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
                    SealedV2InsertOutcome::Inserted
                }
                Err(_) => return Err(OwnerSlotAdmissionError::StorageUnavailable),
            };
        tx.commit()
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
        Ok(outcome)
    }

    /// Owner-scoped V2 revoke/tombstone. The opaque envelope is retained for
    /// audit/restart idempotence, while active listing excludes the row.
    pub async fn revoke_sealed_v2(&self, owner: &[u8; 32], record_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memory_sealed_v2 SET status = 2
             WHERE record_id = ?1 AND owner = ?2 AND status = 0",
            params![record_id.as_slice(), owner.as_slice()],
        )
        .map(|n| n == 1)
        .unwrap_or(false)
    }

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

    // [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] This bounded path is used by
    // remote V1 writers.  Owner discovery, the all-status ceiling check, and
    // the INSERT share one IMMEDIATE transaction; cache/counters advance only
    // after commit.  The legacy unbounded `insert` above remains compatible
    // for local and replication callers until API wiring is integrated.
    pub(crate) async fn insert_with_owner_slot(
        &self,
        record: &MemoryRecord,
        embedding_model: &str,
        policy: OwnerSlotPolicy,
    ) -> Result<bool, OwnerSlotAdmissionError> {
        if !record.verify_id() {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
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
            record.encrypted_content.clone()
        } else if let Some(ref key) = self.record_key {
            let key: &[u8; 32] = &**key;
            if record.encrypted_content.is_empty() {
                record.encrypted_content.clone()
            } else {
                match encrypt_record_content(key, &record.encrypted_content) {
                    Ok(ct) => ct,
                    Err(_) => {
                        self.total_rejected.fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    }
                }
            }
        } else {
            record.encrypted_content.clone()
        };

        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
        enforce_owner_slot_tx(&tx, &record.owner, policy)?;
        let changes = tx
            .execute(
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
            )
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;
        tx.commit()
            .map_err(|_| OwnerSlotAdmissionError::StorageUnavailable)?;

        if changes > 0 {
            self.total_inserted.fetch_add(1, Ordering::Relaxed);
            self.cache.write().put(record.clone());
            Ok(true)
        } else {
            Ok(false)
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

#[cfg(test)]
mod sealed_v2_tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::ledger::record::{
        memory_sealed_v2_record_id, memory_sealed_v2_signature_transcript, MemorySealedV2Envelope,
        MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES,
    };

    fn frame() -> Vec<u8> {
        let mut bytes = vec![0u8; MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES];
        bytes[..4].copy_from_slice(b"AMV2");
        bytes[4] = 1;
        bytes[17..21].copy_from_slice(&(16u32).to_be_bytes());
        assert!(MemorySealedV2Envelope::decode(&bytes).is_ok());
        bytes
    }

    fn frame_with_marker(marker: u8) -> Vec<u8> {
        let mut bytes = frame();
        bytes[21] = marker;
        bytes
    }

    fn v1_record(owner: [u8; 32], marker: u8) -> MemoryRecord {
        MemoryRecord::new(
            owner,
            u64::from(marker),
            MemoryLayer::Knowledge,
            Vec::new(),
            String::new(),
            vec![marker],
            Vec::new(),
        )
    }

    fn signed_v2(key: &IdentityKeyPair, marker: u8) -> ([u8; 32], u64, Vec<u8>, [u8; 64]) {
        let owner = key.public_key_bytes();
        let created_at = u64::from(marker);
        let envelope = frame_with_marker(marker);
        let record_id = memory_sealed_v2_record_id(&owner, created_at, &envelope);
        let signature = key.sign(&memory_sealed_v2_signature_transcript(
            &owner, &record_id, created_at, &envelope,
        ));
        (record_id, created_at, envelope, signature)
    }

    // [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] The race matrix deliberately
    // maps both storage APIs to one coarse outcome so every pair proves the
    // same aggregate invariant without cold-starting a separate test process.
    #[derive(Clone, Copy, Debug)]
    enum OwnerSlotRaceKind {
        V1V1,
        V2V2,
        Mixed,
    }

    enum FirstWrite {
        V1(MemoryRecord),
        V2 {
            owner: [u8; 32],
            record_id: [u8; 32],
            created_at: u64,
            envelope: Vec<u8>,
            signature: [u8; 64],
        },
    }

    async fn run_owner_slot_race(
        kind: OwnerSlotRaceKind,
    ) -> ([Result<bool, OwnerSlotAdmissionError>; 2], usize) {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let policy = OwnerSlotPolicy {
            local_owner: [0xE0; 32],
            max_remote_owners: 1,
        };
        let key_a = IdentityKeyPair::from_bytes(&[0xA1; 32]).unwrap();
        let key_b = IdentityKeyPair::from_bytes(&[0xA2; 32]).unwrap();
        let v1_a = FirstWrite::V1(v1_record(key_a.public_key_bytes(), 1));
        let v1_b = FirstWrite::V1(v1_record(key_b.public_key_bytes(), 2));
        let (v2_id_a, v2_created_a, v2_envelope_a, v2_signature_a) = signed_v2(&key_a, 3);
        let v2_a = FirstWrite::V2 {
            owner: key_a.public_key_bytes(),
            record_id: v2_id_a,
            created_at: v2_created_a,
            envelope: v2_envelope_a,
            signature: v2_signature_a,
        };
        let (v2_id_b, v2_created_b, v2_envelope_b, v2_signature_b) = signed_v2(&key_b, 4);
        let v2_b = FirstWrite::V2 {
            owner: key_b.public_key_bytes(),
            record_id: v2_id_b,
            created_at: v2_created_b,
            envelope: v2_envelope_b,
            signature: v2_signature_b,
        };
        let (first, second) = match kind {
            OwnerSlotRaceKind::V1V1 => (v1_a, v1_b),
            OwnerSlotRaceKind::V2V2 => (v2_a, v2_b),
            OwnerSlotRaceKind::Mixed => (v1_a, v2_b),
        };
        let barrier = Arc::new(tokio::sync::Barrier::new(3));
        let first_storage = Arc::clone(&storage);
        let first_barrier = Arc::clone(&barrier);
        let first_task = tokio::spawn(async move {
            first_barrier.wait().await;
            match first {
                FirstWrite::V1(record) => {
                    first_storage
                        .insert_with_owner_slot(&record, "", policy)
                        .await
                }
                FirstWrite::V2 {
                    owner,
                    record_id,
                    created_at,
                    envelope,
                    signature,
                } => first_storage
                    .insert_sealed_v2_with_owner_slot(
                        &owner, &record_id, created_at, &envelope, &signature, policy,
                    )
                    .await
                    .map(|outcome| outcome == SealedV2InsertOutcome::Inserted),
            }
        });
        let second_storage = Arc::clone(&storage);
        let second_barrier = Arc::clone(&barrier);
        let second_task = tokio::spawn(async move {
            second_barrier.wait().await;
            match second {
                FirstWrite::V1(record) => {
                    second_storage
                        .insert_with_owner_slot(&record, "", policy)
                        .await
                }
                FirstWrite::V2 {
                    owner,
                    record_id,
                    created_at,
                    envelope,
                    signature,
                } => second_storage
                    .insert_sealed_v2_with_owner_slot(
                        &owner, &record_id, created_at, &envelope, &signature, policy,
                    )
                    .await
                    .map(|outcome| outcome == SealedV2InsertOutcome::Inserted),
            }
        });
        barrier.wait().await;
        let outcomes = [first_task.await.unwrap(), second_task.await.unwrap()];
        let owner_count = storage.count_distinct_owners().await;
        (outcomes, owner_count)
    }

    #[tokio::test]
    async fn owner_slot_v1_v1_v2_v2_and_mixed_races_admit_exactly_one() {
        for kind in [
            OwnerSlotRaceKind::V1V1,
            OwnerSlotRaceKind::V2V2,
            OwnerSlotRaceKind::Mixed,
        ] {
            let (outcomes, owner_count) = run_owner_slot_race(kind).await;
            let admitted = outcomes
                .iter()
                .filter(|outcome| matches!(outcome, Ok(true)))
                .count();
            assert_eq!(admitted, 1, "{kind:?} race admitted {admitted} writers");
            assert_eq!(
                owner_count, 1,
                "{kind:?} race persisted {owner_count} owners"
            );
            assert!(
                outcomes
                    .iter()
                    .any(|outcome| matches!(outcome, Err(error) if error.is_at_capacity())),
                "{kind:?} race did not produce the typed capacity rejection"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn owner_slot_independent_disk_connections_mixed_race_and_restart_preserve_cap() {
        // [MEMORY-V2-OWNER-SLOT-CROSS-CONNECTION-TEST 2026-10-03 by Codex]
        // The in-memory matrix above shares one MemoryStorage mutex. This
        // exercise uses two independent WAL connections, treats SQLite busy
        // as StorageUnavailable (not capacity), probes a fresh owner after
        // contention, and proves the persisted union remains capped after
        // reopen.
        let directory = tempfile::tempdir().unwrap();
        let db_path = directory.path().join("owner-slot-cross-connection.db");
        let first_storage = Arc::new(MemoryStorage::open(&db_path, None).unwrap());
        let second_storage = Arc::new(MemoryStorage::open(&db_path, None).unwrap());
        let policy = OwnerSlotPolicy {
            local_owner: [0xE3; 32],
            max_remote_owners: 1,
        };
        let key_a = IdentityKeyPair::from_bytes(&[0xB1; 32]).unwrap();
        let key_b = IdentityKeyPair::from_bytes(&[0xB2; 32]).unwrap();
        let v1 = v1_record(key_a.public_key_bytes(), 71);
        let (v2_id, v2_created_at, v2_envelope, v2_signature) = signed_v2(&key_b, 72);
        let barrier = Arc::new(tokio::sync::Barrier::new(3));
        let first_barrier = Arc::clone(&barrier);
        let first_worker_storage = Arc::clone(&first_storage);
        let first_task = tokio::spawn(async move {
            first_barrier.wait().await;
            first_worker_storage
                .insert_with_owner_slot(&v1, "", policy)
                .await
        });
        let second_barrier = Arc::clone(&barrier);
        let second_worker_storage = Arc::clone(&second_storage);
        let second_task = tokio::spawn(async move {
            second_barrier.wait().await;
            second_worker_storage
                .insert_sealed_v2_with_owner_slot(
                    &key_b.public_key_bytes(),
                    &v2_id,
                    v2_created_at,
                    &v2_envelope,
                    &v2_signature,
                    policy,
                )
                .await
                .map(|outcome| outcome == SealedV2InsertOutcome::Inserted)
        });
        let (first_outcome, second_outcome) =
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                barrier.wait().await;
                (
                    first_task.await.expect("V1 worker panicked"),
                    second_task.await.expect("V2 worker panicked"),
                )
            })
            .await
            .expect("cross-connection owner-slot race timed out");
        let outcomes = [first_outcome, second_outcome];
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, Ok(true)))
                .count(),
            1,
            "independent connections must admit exactly one mixed writer"
        );
        let rejected = outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                Err(error) => Some(error),
                Ok(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(rejected.len(), 1, "one mixed writer must be rejected");
        assert!(matches!(
            rejected[0],
            &OwnerSlotAdmissionError::AtCapacity | &OwnerSlotAdmissionError::StorageUnavailable
        ));

        // With contention gone, a genuinely new owner must report capacity,
        // not inherit a transient SQLITE_BUSY/StorageUnavailable result.
        let third = second_storage
            .insert_with_owner_slot(&v1_record([0xB3; 32], 73), "", policy)
            .await;
        assert!(matches!(
            third,
            Err(error) if error.is_at_capacity()
        ));
        drop(first_storage);
        drop(second_storage);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        assert_eq!(reopened.count_distinct_owners().await, 1);
    }

    #[tokio::test]
    async fn owner_slot_handles_local_bypass_tombstone_exact_retry_and_unlimited() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let local = [0xE1; 32];
        let policy = OwnerSlotPolicy {
            local_owner: local,
            max_remote_owners: 1,
        };
        let remote_key = IdentityKeyPair::from_bytes(&[0xB1; 32]).unwrap();
        let (remote_id, created_at, envelope, signature) = signed_v2(&remote_key, 11);
        assert_eq!(
            storage
                .insert_sealed_v2_with_owner_slot(
                    &remote_key.public_key_bytes(),
                    &remote_id,
                    created_at,
                    &envelope,
                    &signature,
                    policy,
                )
                .await
                .unwrap(),
            SealedV2InsertOutcome::Inserted
        );
        assert_eq!(
            storage
                .insert_sealed_v2_with_owner_slot(
                    &remote_key.public_key_bytes(),
                    &remote_id,
                    created_at,
                    &envelope,
                    &signature,
                    policy,
                )
                .await
                .unwrap(),
            SealedV2InsertOutcome::ExactDuplicate
        );
        assert!(
            storage
                .revoke_sealed_v2(&remote_key.public_key_bytes(), &remote_id)
                .await
        );
        // [MEMORY-V2-OWNER-SLOT-TESTS 2026-10-02 by Codex] A revoked exact
        // retry remains a conflict and cannot resurrect the tombstone.
        assert_eq!(
            storage
                .insert_sealed_v2_with_owner_slot(
                    &remote_key.public_key_bytes(),
                    &remote_id,
                    created_at,
                    &envelope,
                    &signature,
                    policy,
                )
                .await
                .unwrap(),
            SealedV2InsertOutcome::Conflict
        );
        assert!(storage
            .get_sealed_v2(&remote_key.public_key_bytes(), &remote_id)
            .await
            .unwrap()
            .is_none());
        let rejected = storage
            .insert_with_owner_slot(&v1_record([0xB2; 32], 12), "", policy)
            .await
            .unwrap_err();
        assert!(rejected.is_at_capacity());
        assert!(storage
            .insert_with_owner_slot(&v1_record(local, 13), "", policy)
            .await
            .unwrap());

        let unlimited = MemoryStorage::open(":memory:", None).unwrap();
        assert!(unlimited
            .insert_with_owner_slot(
                &v1_record([0xB3; 32], 14),
                "",
                OwnerSlotPolicy {
                    local_owner: [0xEF; 32],
                    max_remote_owners: 0,
                },
            )
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn owner_slot_failed_insert_rolls_back_without_cache_or_slot() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let policy = OwnerSlotPolicy {
            local_owner: [0xE2; 32],
            max_remote_owners: 1,
        };
        {
            let conn = storage.conn_lock().await;
            conn.execute_batch(
                "CREATE TRIGGER reject_owner_slot_insert
                 BEFORE INSERT ON records
                 BEGIN SELECT RAISE(ABORT, 'test failure'); END;",
            )
            .unwrap();
        }
        let failed_record = v1_record([0xC1; 32], 21);
        let failed_id = failed_record.record_id;
        let failed = storage
            .insert_with_owner_slot(&failed_record, "", policy)
            .await
            .unwrap_err();
        assert!(!failed.is_at_capacity());
        assert_eq!(storage.total_inserted(), 0);
        // [MEMORY-V2-OWNER-SLOT-TESTS 2026-10-02 by Codex] A rolled-back
        // write leaves neither a durable row nor a cache entry, so another
        // remote owner may consume the still-free slot.
        assert!(storage.get(&failed_id).await.is_none());
        assert!(storage.cache.write().get(&failed_id).is_none());
        {
            let conn = storage.conn_lock().await;
            conn.execute("DROP TRIGGER reject_owner_slot_insert", [])
                .unwrap();
        }
        assert!(storage
            .insert_with_owner_slot(&v1_record([0xC2; 32], 22), "", policy)
            .await
            .unwrap());
        assert_eq!(storage.total_inserted(), 1);
        assert_eq!(storage.count_distinct_owners().await, 1);
    }

    #[tokio::test]
    async fn sealed_v2_exact_retry_conflict_owner_listing_and_revoke() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let key = IdentityKeyPair::from_bytes(&[7u8; 32]).unwrap();
        let owner = key.public_key_bytes();
        let other = [2u8; 32];
        let envelope = frame();
        let id = memory_sealed_v2_record_id(&owner, 7, &envelope);
        let transcript = memory_sealed_v2_signature_transcript(&owner, &id, 7, &envelope);
        let signature = key.sign(&transcript);
        assert_eq!(
            storage
                .insert_sealed_v2(&owner, &id, 7, &envelope, &signature)
                .await
                .unwrap(),
            SealedV2InsertOutcome::Inserted
        );
        assert_eq!(
            storage
                .insert_sealed_v2(&owner, &id, 7, &envelope, &signature)
                .await
                .unwrap(),
            SealedV2InsertOutcome::ExactDuplicate
        );
        assert_eq!(
            storage
                .classify_sealed_v2(&owner, &id, 7, &envelope, &signature)
                .await
                .unwrap(),
            Some(SealedV2InsertOutcome::ExactDuplicate)
        );
        let mut changed = envelope.clone();
        changed[21] = 9;
        assert_eq!(
            storage
                .insert_sealed_v2(&owner, &id, 7, &changed, &signature)
                .await
                .unwrap(),
            SealedV2InsertOutcome::Conflict
        );
        assert!(storage.get_sealed_v2(&other, &id).await.unwrap().is_none());
        for created_at in [8u64, 9] {
            let row_id = memory_sealed_v2_record_id(&owner, created_at, &envelope);
            let row_transcript =
                memory_sealed_v2_signature_transcript(&owner, &row_id, created_at, &envelope);
            let row_signature = key.sign(&row_transcript);
            assert!(storage
                .insert_sealed_v2(&owner, &row_id, created_at, &envelope, &row_signature)
                .await
                .unwrap()
                .is_inserted());
        }
        let first = storage.list_sealed_v2(&owner, None, 1).await.unwrap();
        assert_eq!(first.rows.len(), 1);
        let second = storage
            .list_sealed_v2(&owner, first.next_cursor.as_ref(), 1)
            .await
            .unwrap();
        assert_eq!(second.rows.len(), 1);
        let third = storage
            .list_sealed_v2(&owner, second.next_cursor.as_ref(), 1)
            .await
            .unwrap();
        assert_eq!(third.rows.len(), 1);
        assert!(third.next_cursor.is_none());
        assert!(storage.revoke_sealed_v2(&owner, &id).await);
        assert_eq!(
            storage
                .classify_sealed_v2(&owner, &id, 7, &envelope, &signature)
                .await
                .unwrap(),
            Some(SealedV2InsertOutcome::Conflict)
        );
        assert!(storage.get_sealed_v2(&owner, &id).await.unwrap().is_none());
        assert_eq!(
            storage
                .list_sealed_v2(&owner, None, 10)
                .await
                .unwrap()
                .rows
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn sealed_v2_reads_fail_closed_on_corrupt_durable_row() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let key = IdentityKeyPair::from_bytes(&[8u8; 32]).unwrap();
        let owner = key.public_key_bytes();
        let record_id = [4u8; 32];
        let corrupt_envelope = vec![0u8; MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES];
        let conn = storage.conn_lock().await;
        conn.execute(
            "INSERT INTO memory_sealed_v2
             (record_id, owner, created_at, envelope, signature, status, inserted_at)
             VALUES (?1, ?2, 7, ?3, ?4, 0, 7)",
            rusqlite::params![
                record_id.as_slice(),
                owner.as_slice(),
                corrupt_envelope,
                [0u8; 64].as_slice()
            ],
        )
        .unwrap();
        drop(conn);

        assert!(storage.get_sealed_v2(&owner, &record_id).await.is_err());
        assert!(storage.list_sealed_v2(&owner, None, 10).await.is_err());
    }

    #[tokio::test]
    async fn sealed_v2_list_prepare_failure_is_not_an_empty_page() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        {
            let conn = storage.conn_lock().await;
            conn.execute("DROP TABLE memory_sealed_v2", []).unwrap();
        }
        assert!(storage.list_sealed_v2(&[3u8; 32], None, 10).await.is_err());
    }

    #[tokio::test]
    async fn owner_capacity_snapshot_unions_legacy_and_sealed_rows() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let legacy_owner = [0x31u8; 32];
        let legacy_id = [0x41u8; 32];
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "INSERT INTO records
                 (record_id, owner, timestamp, layer, signature, created_at)
                 VALUES (?1, ?2, 1, 0, ?3, 1)",
                rusqlite::params![legacy_id.as_slice(), legacy_owner.as_slice(), [0u8; 64]],
            )
            .unwrap();
        }
        let remote_key = IdentityKeyPair::from_bytes(&[9u8; 32]).unwrap();
        let remote_owner = remote_key.public_key_bytes();
        let remote_envelope = frame();
        let remote_id = memory_sealed_v2_record_id(&remote_owner, 2, &remote_envelope);
        let remote_signature = remote_key.sign(&memory_sealed_v2_signature_transcript(
            &remote_owner,
            &remote_id,
            2,
            &remote_envelope,
        ));
        assert!(storage
            .insert_sealed_v2(
                &remote_owner,
                &remote_id,
                2,
                &remote_envelope,
                &remote_signature,
            )
            .await
            .unwrap()
            .is_inserted());
        let local_key = IdentityKeyPair::from_bytes(&[10u8; 32]).unwrap();
        let local_owner = local_key.public_key_bytes();
        let local_envelope = frame();
        let local_id = memory_sealed_v2_record_id(&local_owner, 3, &local_envelope);
        let local_signature = local_key.sign(&memory_sealed_v2_signature_transcript(
            &local_owner,
            &local_id,
            3,
            &local_envelope,
        ));
        assert!(storage
            .insert_sealed_v2(
                &local_owner,
                &local_id,
                3,
                &local_envelope,
                &local_signature,
            )
            .await
            .unwrap()
            .is_inserted());

        assert_eq!(storage.count_distinct_owners().await, 3);
        assert!(storage.owner_exists(&remote_owner).await);
        assert_eq!(
            storage
                .remote_owner_capacity_snapshot(&local_owner, &[0x61u8; 32])
                .await
                .unwrap(),
            (2, false)
        );
        assert_eq!(
            storage
                .remote_owner_capacity_snapshot(&local_owner, &remote_owner)
                .await
                .unwrap(),
            (2, true)
        );
        assert!(storage.revoke_sealed_v2(&remote_owner, &remote_id).await);
        assert_eq!(storage.count_distinct_owners().await, 3);
        assert_eq!(
            storage
                .remote_owner_capacity_snapshot(&local_owner, &remote_owner)
                .await
                .unwrap(),
            (2, true)
        );
    }
}
