// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_feedback.rs
// ============================================
//! Raw-log ingestion and memory-feedback persistence.
//!
//! [MEMCHAIN-STORAGE-FEEDBACK-SPLIT 2026-09-23 by Codex] This module is a
//! behavior-preserving extraction from `storage_ops.rs`. SQL text, ordering,
//! error handling, cache invalidation, and transaction behavior are unchanged.

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, error, warn};

use super::storage::{MemoryStorage, RawLogRow};
use super::storage_crypto::{decrypt_rawlog_content, encrypt_rawlog_content};

// ============================================
// impl MemoryStorage - RawLog Operations
// ============================================

impl MemoryStorage {
    pub async fn insert_raw_log(
        &self,
        session_id: &str,
        turn_index: i64,
        role: &str,
        content: &str,
        source_ai: &str,
        recall_context: Option<&str>,
        extractable: i64,
        feedback_signal: Option<i64>,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Result<i64, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let (stored_content, encrypted_flag): (Vec<u8>, i64) = match rawlog_key {
            Some(key) => match encrypt_rawlog_content(key, content.as_bytes()) {
                Ok(ciphertext) => (ciphertext, 1),
                Err(e) => {
                    warn!(
                        "[STORAGE] RawLog encryption failed, storing plaintext: {}",
                        e
                    );
                    (content.as_bytes().to_vec(), 0)
                }
            },
            None => (content.as_bytes().to_vec(), 0),
        };

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO raw_logs (session_id, turn_index, role, content, source_ai,
                recall_context, extractable, feedback_signal, encrypted, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                session_id,
                turn_index,
                role,
                stored_content,
                source_ai,
                recall_context,
                extractable,
                feedback_signal,
                encrypted_flag,
                now,
            ],
        )
        .map_err(|e| format!("RawLog insert: {}", e))?;

        let log_id = conn.last_insert_rowid();
        Ok(log_id)
    }

    pub async fn read_rawlog_content(
        &self,
        log_id: i64,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Option<String> {
        let conn = self.conn.lock().await;
        let row: Option<(Vec<u8>, i64)> = conn
            .query_row(
                "SELECT content, encrypted FROM raw_logs WHERE log_id = ?1",
                params![log_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .ok()?;

        let (content_bytes, encrypted) = row?;

        if encrypted == 0 {
            String::from_utf8(content_bytes).ok()
        } else if let Some(key) = rawlog_key {
            decrypt_rawlog_content(key, &content_bytes)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
        } else {
            warn!("[STORAGE] Encrypted rawlog but no key provided");
            None
        }
    }

    pub async fn update_rawlog_feedback(&self, log_id: i64, signal: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE raw_logs SET feedback_signal = ?1 WHERE log_id = ?2",
            params![signal, log_id],
        );
    }

    pub async fn get_unprocessed_rawlogs(&self, limit: usize) -> Vec<RawLogRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT log_id, session_id, turn_index, role, content, encrypted,
                    recall_context, extractable, feedback_signal
             FROM raw_logs
             WHERE feedback_signal IS NULL
             ORDER BY session_id, turn_index
             LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![limit as i64], |row| {
            Ok(RawLogRow {
                log_id: row.get(0)?,
                session_id: row.get(1)?,
                turn_index: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                encrypted: row.get(5)?,
                recall_context: row.get(6)?,
                extractable: row.get(7)?,
                feedback_signal: row.get(8)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage - Feedback Operations
// ============================================

impl MemoryStorage {
    pub async fn increment_positive_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET positive_feedback = positive_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] positive_feedback incremented"
                );
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] positive_feedback: record not found"
                );
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ positive_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn increment_negative_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET negative_feedback = negative_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] negative_feedback incremented"
                );
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] negative_feedback: record not found"
                );
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ negative_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn set_conflict_with(&self, record_id: &[u8; 32], conflict_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET conflict_with = ?1 WHERE record_id = ?2",
            params![conflict_id.as_slice(), record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    conflict = hex::encode(conflict_id),
                    "[STORAGE] conflict_with set"
                );
                self.cache.write().invalidate(record_id);
                true
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] conflict_with: record not found"
                );
                false
            }
            Err(e) => {
                error!(error = %e, "[STORAGE] ❌ set_conflict_with failed");
                false
            }
        }
    }

    pub async fn insert_feedback(
        &self,
        owner: &[u8; 32],
        memory_id: &[u8; 32],
        session_id: &str,
        turn_index: i64,
        signal: i64,
        features: Option<&[f32; 9]>,
        prediction: Option<f32>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let features_blob: Option<Vec<u8>> =
            features.map(|f| f.iter().flat_map(|v| v.to_le_bytes()).collect());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO memory_feedback (owner, memory_id, session_id, turn_index,
                signal, features, prediction, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                owner.as_slice(),
                memory_id.as_slice(),
                session_id,
                turn_index,
                signal,
                features_blob.as_deref(),
                prediction,
                now,
            ],
        );
    }

    pub async fn get_recent_feedback(&self, limit: usize) -> Vec<(i64, f32)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT signal, prediction FROM memory_feedback ORDER BY created_at DESC LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1).unwrap_or(0.0)))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}
