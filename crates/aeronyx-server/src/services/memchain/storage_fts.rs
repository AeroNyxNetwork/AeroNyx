// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_fts.rs
// ============================================
//! # Full-Text Search Operations — FTS5 Index + BM25 Search
//!
//! ## Creation Reason (v2.4.0+BM25)
//! Extracted as a new file to keep storage_ops.rs manageable (already 1500+ lines).
//! Provides FTS5-based BM25 keyword search as a complement to vector similarity
//! search, enabling hybrid retrieval via Reciprocal Rank Fusion (RRF).
//!
//! ## Main Functionality
//! - `bm25_search()`: Full-text search across records, entities, sessions
//! - `search_with_snippets()`: FTS5 snippet search with `<mark>` highlights (v2.5.0+Unify)
//! - `group_hits_by_session()`: Group search hits by session with metadata (v2.5.0+Unify)
//! - `fts_index_record()`: Index a memory record into FTS5
//! - `fts_index_entity()`: Index an entity name + description
//! - `fts_index_session()`: Index a session summary
//! - `fts_remove_record()`: Remove a record from FTS5 (on revoke/forget)
//! - `fts_reindex_all()`: Full reindex (startup or migration)
//!
//! ## FTS5 Schema
//! ```sql
//! CREATE VIRTUAL TABLE fts_index USING fts5(
//!     source_type,   -- 'record' | 'entity' | 'session'
//!     source_id,     -- record_id hex / entity_id / session_id
//!     owner_hex,     -- owner public key hex (access control)
//!     content,       -- searchable text
//!     tags,          -- topic_tags or entity_type (for boosting)
//!     tokenize='porter unicode61'
//! );
//! ```
//!
//! ## BM25 Scoring
//! FTS5's built-in `bm25()` returns negative scores (lower = more relevant).
//! We negate them so higher = better, consistent with cosine similarity.
//! The `bm25()` weights parameter `(0, 0, 0, 1, 0)` means:
//! - Only the `content` column (index 3) contributes to scoring
//! - source_type, source_id, owner_hex, tags are not scored
//!
//! ## Integration Points
//! - `storage.rs` create_schema(): Creates the FTS5 virtual table
//! - `storage.rs` maybe_migrate(): Backfills FTS5 from existing records
//! - `recall_handler.rs`: Calls bm25_search() as Step 2a-bis
//! - `mpi_graph_handlers.rs`: Calls search_with_snippets() + group_hits_by_session()
//!   for the human-facing GET /api/mpi/search endpoint
//! - `reflection.rs` Step 7: Calls fts_index_entity() after upsert
//! - `reflection.rs` Step 10: Calls fts_index_session() after summary
//! - `mpi_handlers.rs` remember: Calls fts_index_record() after insert
//! - `mpi_handlers.rs` forget: Calls fts_remove_record() after revoke
//!
//! ## Architecture
//! This file uses `impl MemoryStorage` blocks (same pattern as storage_ops.rs).
//! Rust allows multiple impl blocks across files within the same crate.
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage struct, conn (TokioMutex<Connection>)
//! - rusqlite — FTS5 is included in the `bundled` feature (no extra deps)
//!
//! ⚠️ Important Note for Next Developer:
//! - FTS5 is included in rusqlite's `bundled` feature — no Cargo.toml changes needed.
//! - The `tokenize='porter unicode61'` handles English stemming + Unicode.
//!   Chinese/CJK tokenization uses character bigrams (unicode61 default).
//!   For better CJK support, consider adding a custom tokenizer in Phase C.
//! - bm25() weight array must match column count (5 columns → 5 weights).
//! - FTS5 MATCH syntax: space between words = implicit AND (requires all terms).
//!   Use "word1 OR word2" for explicit OR.
//! - When records are encrypted (record_key is set), FTS5 indexes the
//!   PLAINTEXT content (decrypted at insert time). The FTS index itself
//!   is NOT encrypted — this is acceptable because the DB file is local-only.
//! - fts_reindex_all() is expensive (full table scan + per-row decryption).
//!   Only call at startup or when migration detects missing FTS data.
//! - fts_reindex_all() now decrypts record content before indexing (v2.4.0+BM25-fix).
//!   Previous version indexed raw encrypted_content from DB, which was ciphertext
//!   when record_key was set — causing BM25 searches to return no results.
//! - search_with_snippets() uses FTS5 snippet() function for highlighted results.
//!   The snippet column index (3) must match the content column position in the
//!   FTS5 virtual table definition. If schema changes, update the index.
//! - group_hits_by_session() does synchronous SQLite queries inside the conn lock.
//!   For large result sets this could hold the lock longer than ideal. Consider
//!   batching the metadata lookups if performance becomes an issue.
//!
//! ## Last Modified
//! v2.4.0+BM25 - 🌟 Initial implementation
//! v2.4.0+BM25-fix - 🐛 Fixed: bm25_search debug logging, fts_reindex_all decryption,
//!   impl block brace mismatch
//! v2.5.0+Unify - 🌟 Added search_with_snippets() and group_hits_by_session() methods.
//!   These were called by mpi_graph_handlers.rs::mpi_search() but never implemented,
//!   causing E0599 compilation errors. Also added SearchHit and SessionSearchGroup types.

use std::collections::HashMap;

use rusqlite::params;
use tracing::{debug, info, warn};

use super::storage::MemoryStorage;
use super::storage_crypto::decrypt_record_content;

// ============================================
// Search Hit Types (v2.5.0+Unify)
// ============================================

/// A single FTS5 search hit with snippet highlight.
///
/// Returned by `search_with_snippets()`. The snippet contains `<mark>` tags
/// around matched terms, suitable for direct rendering in HTML UIs.
///
/// ## v2.5.0+Unify
/// Created to support mpi_graph_handlers::mpi_search() which was calling
/// search_with_snippets() before this type and method existed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchHit {
    /// Source type: "record" | "entity" | "session"
    pub source_type: String,
    /// Source identifier (record_id hex / entity_id / session_id)
    pub source_id: String,
    /// FTS5 snippet with `<mark>` highlights around matched terms.
    /// Example: "...用 <mark>token bucket</mark> 实现限流..."
    pub snippet: String,
    /// BM25 relevance score (negated so higher = more relevant).
    pub score: f64,
    /// Session ID associated with this hit (resolved from source).
    /// - For 'session' source_type: the source_id itself.
    /// - For 'record' / 'entity': may be None if not resolvable.
    pub session_id: Option<String>,
}

/// Search results grouped by session, with session metadata.
///
/// Used by the `/api/mpi/search` endpoint response format.
/// Groups are sorted by `best_score` (highest first).
///
/// ## v2.5.0+Unify
/// Created to support mpi_graph_handlers::mpi_search().
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSearchGroup {
    pub session_id: String,
    pub session_title: Option<String>,
    pub project_name: Option<String>,
    pub started_at: Option<i64>,
    pub hits: Vec<SearchHit>,
    pub best_score: f64,
}

// ============================================
// impl MemoryStorage — BM25 Search
// ============================================

impl MemoryStorage {
    /// BM25 full-text search across records, entities, and sessions.
    ///
    /// Returns `Vec<(source_type, source_id, bm25_score)>` sorted by relevance.
    /// The caller (recall handler) joins back to source tables for content.
    ///
    /// ## Arguments
    /// * `query` - Search query (natural language, tokenized by FTS5 porter stemmer)
    /// * `owner` - Owner public key for access control filtering
    /// * `limit` - Maximum results to return
    ///
    /// ## BM25 Score
    /// FTS5 `bm25()` returns negative values (lower = more relevant).
    /// We negate them so higher = better, consistent with cosine similarity.
    /// Weights `(0, 0, 0, 1, 0)` score only the `content` column.
    ///
    /// ## Query Syntax
    /// FTS5 tokenizes the query using the same `porter unicode61` tokenizer.
    /// - "JWT authentication" → matches documents containing both stems (implicit AND)
    /// - "rate OR limiting" → matches documents containing either
    /// - "RS256" → exact keyword match (porter stemmer preserves it)
    pub async fn bm25_search(
        &self,
        query: &str,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<(String, String, f64)> {
        if query.trim().is_empty() {
            return Vec::new();
        }

        let owner_hex = hex::encode(owner);

        // Sanitize query for FTS5: remove special characters that could break syntax
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Vec::new();
        }

        debug!(
            raw_query = %query,
            sanitized = %sanitized,
            owner_hex = %owner_hex,
            limit = limit,
            "[BM25] Search params"
        );

        let conn = self.conn.lock().await;

        // bm25(fts_index, 0, 0, 0, 1, 0) → only score the `content` column (index 3)
        // source_type(0), source_id(1), owner_hex(2) are not scored
        // tags(4) is not scored (could be added later for boosting)
        let mut stmt = match conn.prepare(
            "SELECT source_type, source_id, -bm25(fts_index, 0, 0, 0, 1, 0) as score
             FROM fts_index
             WHERE fts_index MATCH ?1 AND owner_hex = ?2
             ORDER BY score DESC
             LIMIT ?3"
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!("[BM25] Query prepare failed (FTS5 may not be available): {}", e);
                return Vec::new();
            }
        };

        let results: Vec<(String, String, f64)> = stmt.query_map(
            params![sanitized, owner_hex, limit as i64],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        )
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();

        debug!(
            result_count = results.len(),
            "[BM25] Search complete"
        );

        results
    }

    /// FTS5 search with snippet highlights and session resolution.
    ///
    /// Returns search hits with `<mark>` highlighted snippets from FTS5's
    /// `snippet()` function. Used by the human-facing `/api/mpi/search` endpoint.
    ///
    /// ## Snippet Format
    /// FTS5 `snippet(fts_index, 3, '<mark>', '</mark>', '...', 32)`:
    /// - Column 3 = content column
    /// - `<mark>` / `</mark>` = highlight delimiters for frontend rendering
    /// - `...` = ellipsis for truncated context
    /// - 32 = max tokens in snippet window
    ///
    /// ## Session Resolution
    /// For 'session' hits, session_id = source_id (direct).
    /// For 'record' and 'entity' hits, session_id resolution is best-effort:
    /// - Records: attempts to find session via raw_logs or episode linkage
    /// - Entities: may span multiple sessions, returns None
    /// Callers should handle hits with session_id = None gracefully.
    ///
    /// ## v2.5.0+Unify
    /// Created to fix E0599 "method not found" in mpi_graph_handlers.rs.
    /// This method was called by mpi_search() but never implemented.
    pub async fn search_with_snippets(
        &self,
        query: &str,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<SearchHit> {
        if query.trim().is_empty() {
            return Vec::new();
        }

        let owner_hex = hex::encode(owner);
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Vec::new();
        }

        let conn = self.conn.lock().await;

        // FTS5 snippet() produces highlighted text around matched terms.
        // Arguments: (table, column_index, open_mark, close_mark, ellipsis, max_tokens)
        // Column 3 = content (matching the FTS5 schema column order).
        let mut stmt = match conn.prepare(
            "SELECT source_type, source_id,
                    snippet(fts_index, 3, '<mark>', '</mark>', '...', 32) as snip,
                    -bm25(fts_index, 0, 0, 0, 1, 0) as score
             FROM fts_index
             WHERE fts_index MATCH ?1 AND owner_hex = ?2
             ORDER BY score DESC
             LIMIT ?3"
        ) {
            Ok(s) => s,
            Err(e) => {
                debug!("[FTS] search_with_snippets prepare failed: {}", e);
                return Vec::new();
            }
        };

        let raw_hits: Vec<(String, String, String, f64)> = stmt
            .query_map(
                params![sanitized, owner_hex, limit as i64],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                )),
            )
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        // Resolve session_id for each hit based on source_type
        let hits: Vec<SearchHit> = raw_hits
            .into_iter()
            .map(|(source_type, source_id, snippet, score)| {
                let session_id = match source_type.as_str() {
                    // Session hits: the source_id IS the session_id
                    "session" => Some(source_id.clone()),
                    // Record hits: try to resolve session via raw_logs table
                    // (record_id hex → raw_logs.session_id via content match)
                    // This is best-effort; many records don't have a direct session link.
                    "record" => {
                        // Attempt: look up the record_id in raw_logs (if the record was
                        // ingested via /log endpoint, it may have a session association).
                        // For simplicity and to avoid holding the lock too long, we skip
                        // this resolution here. The group_hits_by_session() method will
                        // place these hits in the "(ungrouped)" bucket.
                        None
                    }
                    // Entity hits: entities can span multiple sessions, no single session_id
                    _ => None,
                };
                SearchHit {
                    source_type,
                    source_id,
                    snippet,
                    score,
                    session_id,
                }
            })
            .collect();

        debug!(
            query = %query,
            hits = hits.len(),
            "[FTS] search_with_snippets complete"
        );

        hits
    }

    /// Group search hits by session and enrich with session metadata.
    ///
    /// Takes the flat list of `SearchHit` from `search_with_snippets()` and
    /// organizes them into `SessionSearchGroup` structs, each containing:
    /// - Session metadata (title, project_name, started_at)
    /// - All hits belonging to that session
    /// - The best (highest) score among the group's hits
    ///
    /// Hits without a `session_id` are placed in a synthetic "(ungrouped)" group.
    /// Groups are sorted by `best_score` descending.
    ///
    /// ## v2.5.0+Unify
    /// Created to fix E0599 "method not found" in mpi_graph_handlers.rs.
    /// This method was called by mpi_search() but never implemented.
    pub async fn group_hits_by_session(&self, hits: &[SearchHit]) -> Vec<SessionSearchGroup> {
        if hits.is_empty() {
            return Vec::new();
        }

        // Group hits by session_id
        let mut groups: HashMap<String, Vec<SearchHit>> = HashMap::new();

        for hit in hits {
            let key = hit.session_id.clone().unwrap_or_else(|| "(ungrouped)".to_string());
            groups.entry(key).or_default().push(hit.clone());
        }

        let conn = self.conn.lock().await;

        let mut result: Vec<SessionSearchGroup> = Vec::with_capacity(groups.len());

        for (session_id, group_hits) in &groups {
            let best_score = group_hits
                .iter()
                .map(|h| h.score)
                .fold(0.0_f64, f64::max);

            // Look up session metadata from sessions + projects tables
            let (title, project_name, started_at) = if session_id != "(ungrouped)" {
                let meta: Option<(Option<String>, Option<String>, Option<i64>)> = conn
                    .query_row(
                        "SELECT title, project_id, started_at FROM sessions WHERE session_id = ?1",
                        params![session_id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .ok();

                match meta {
                    Some((title, project_id, started_at)) => {
                        // Resolve project_name from project_id if available
                        let project_name = project_id.and_then(|pid| {
                            conn.query_row(
                                "SELECT name FROM projects WHERE project_id = ?1",
                                params![pid],
                                |row| row.get::<_, String>(0),
                            )
                            .ok()
                        });
                        (title, project_name, started_at)
                    }
                    None => (None, None, None),
                }
            } else {
                (None, None, None)
            };

            result.push(SessionSearchGroup {
                session_id: session_id.clone(),
                session_title: title,
                project_name,
                started_at,
                hits: group_hits.clone(),
                best_score,
            });
        }

        // Sort groups by best_score descending (most relevant session first)
        result.sort_by(|a, b| {
            b.best_score
                .partial_cmp(&a.best_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        result
    }
}

// ============================================
// impl MemoryStorage — FTS5 Index Maintenance
// ============================================

impl MemoryStorage {
    /// Index a memory record into FTS5.
    ///
    /// Called after successful record insertion (remember endpoint + rule engine).
    /// Content should be the PLAINTEXT (pre-encryption) content.
    pub async fn fts_index_record(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        content: &str,
        tags: &str,
    ) {
        if content.trim().is_empty() { return; }

        let rid_hex = hex::encode(record_id);
        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;

        if let Err(e) = conn.execute(
            "INSERT OR REPLACE INTO fts_index (source_type, source_id, owner_hex, content, tags)
             VALUES ('record', ?1, ?2, ?3, ?4)",
            params![rid_hex, owner_hex, content, tags],
        ) {
            debug!("[FTS] Record index failed (non-fatal): {}", e);
        }
    }

    /// Index an entity into FTS5.
    ///
    /// Called by Miner Step 7 after upsert_entity().
    /// Indexes entity name + description for keyword search.
    pub async fn fts_index_entity(
        &self,
        entity_id: &str,
        owner: &[u8; 32],
        name: &str,
        description: Option<&str>,
        entity_type: &str,
    ) {
        let owner_hex = hex::encode(owner);
        let content = match description {
            Some(desc) if !desc.is_empty() => format!("{} {}", name, desc),
            _ => name.to_string(),
        };

        let conn = self.conn.lock().await;
        if let Err(e) = conn.execute(
            "INSERT OR REPLACE INTO fts_index (source_type, source_id, owner_hex, content, tags)
             VALUES ('entity', ?1, ?2, ?3, ?4)",
            params![entity_id, owner_hex, content, entity_type],
        ) {
            debug!("[FTS] Entity index failed (non-fatal): {}", e);
        }
    }

    /// Index a session summary into FTS5.
    ///
    /// Called by Miner Step 10 after update_session_summary().
    pub async fn fts_index_session(
        &self,
        session_id: &str,
        owner: &[u8; 32],
        summary: &str,
    ) {
        if summary.trim().is_empty() { return; }

        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;
        if let Err(e) = conn.execute(
            "INSERT OR REPLACE INTO fts_index (source_type, source_id, owner_hex, content, tags)
             VALUES ('session', ?1, ?2, ?3, 'session_summary')",
            params![session_id, owner_hex, summary],
        ) {
            debug!("[FTS] Session index failed (non-fatal): {}", e);
        }
    }

    /// Remove a record from the FTS5 index.
    ///
    /// Called on revoke/forget to keep FTS in sync.
    pub async fn fts_remove_record(&self, record_id: &[u8; 32]) {
        let rid_hex = hex::encode(record_id);
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "DELETE FROM fts_index WHERE source_type = 'record' AND source_id = ?1",
            params![rid_hex],
        );
    }

    /// Full reindex: populate FTS5 from all active records, entities, and sessions.
    ///
    /// Called at startup if FTS index is empty or after migration.
    /// Expensive operation — scans all three source tables.
    ///
    /// ## v2.4.0+BM25-fix: Decryption support
    /// Records are stored encrypted in DB when record_key is set. Previous version
    /// did a bulk `INSERT ... SELECT encrypted_content` which indexed ciphertext.
    /// Now iterates rows and decrypts each before indexing into FTS5.
    pub async fn fts_reindex_all(&self, owner: &[u8; 32]) {
        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;

        // Clear existing FTS data for this owner
        let _ = conn.execute(
            "DELETE FROM fts_index WHERE owner_hex = ?1",
            params![owner_hex],
        );

        // ── Index records (with decryption) ──
        // Cannot use bulk INSERT...SELECT because encrypted_content may be
        // ciphertext that needs Rust-side ChaCha20 decryption.
        let records_indexed = {
            let mut stmt = match conn.prepare(
                "SELECT record_id, encrypted_content, topic_tags
                 FROM records
                 WHERE owner = ?1 AND status = 0 AND encrypted_content != x''"
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!("[FTS] Reindex: failed to prepare records query: {}", e);
                    return;
                }
            };

            let rk = self.record_key.as_ref().map(|v| &**v);
            let rows: Vec<(Vec<u8>, Vec<u8>, String)> = stmt
                .query_map(params![owner.as_slice()], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .map(|r| r.filter_map(|x| x.ok()).collect())
                .unwrap_or_default();

            let mut count = 0usize;
            for (rid, encrypted, tags) in &rows {
                // Decrypt if record_key is set and content looks like ciphertext
                // (ChaCha20-Poly1305: 12-byte nonce + 16-byte tag = 28 byte overhead minimum)
                let plaintext = if let Some(key) = rk.as_ref() {
                    if encrypted.len() >= 28 {
                        decrypt_record_content(key, encrypted)
                            .unwrap_or_else(|_| encrypted.clone())
                    } else {
                        encrypted.clone()
                    }
                } else {
                    encrypted.clone()
                };

                let text = String::from_utf8_lossy(&plaintext);
                if text.trim().is_empty() { continue; }

                let _ = conn.execute(
                    "INSERT INTO fts_index (source_type, source_id, owner_hex, content, tags)
                     VALUES ('record', ?1, ?2, ?3, ?4)",
                    params![hex::encode(rid), &owner_hex, text.as_ref(), tags],
                );
                count += 1;
            }
            count
        };

        // ── Index entities (plaintext, bulk is fine) ──
        let entities_indexed = conn.execute(
            "INSERT INTO fts_index (source_type, source_id, owner_hex, content, tags)
             SELECT 'entity', entity_id, ?1,
                    CASE WHEN description IS NOT NULL THEN name || ' ' || description ELSE name END,
                    entity_type
             FROM entities
             WHERE owner = ?2",
            params![owner_hex, owner.as_slice()],
        ).unwrap_or(0);

        // ── Index sessions (plaintext summaries, bulk is fine) ──
        let sessions_indexed = conn.execute(
            "INSERT INTO fts_index (source_type, source_id, owner_hex, content, tags)
             SELECT 'session', session_id, ?1, summary, 'session_summary'
             FROM sessions
             WHERE owner = ?2 AND summary IS NOT NULL AND summary != ''",
            params![owner_hex, owner.as_slice()],
        ).unwrap_or(0);

        info!(
            records = records_indexed,
            entities = entities_indexed,
            sessions = sessions_indexed,
            "[FTS] Full reindex complete"
        );
    }
}

// ============================================
// Utility: FTS5 Query Sanitization
// ============================================

/// Sanitize a user query for FTS5 MATCH syntax.
///
/// FTS5 has special syntax characters that can cause parse errors:
/// - Quotes, parentheses, asterisks, carets, etc.
/// - We strip these and keep only alphanumeric + spaces + basic punctuation.
///
/// Also filters out single-character words (noise) and normalizes whitespace.
/// FTS5 with `porter unicode61` tokenizer: space between words = implicit AND
/// (requires all terms present).
fn sanitize_fts_query(query: &str) -> String {
    let cleaned: String = query.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '-' || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect();

    // Split into words and rejoin — this normalizes whitespace
    let words: Vec<&str> = cleaned.split_whitespace()
        .filter(|w| w.len() >= 2) // skip single-char noise
        .collect();

    if words.is_empty() {
        return String::new();
    }

    if words.len() == 1 {
        words[0].to_string()
    } else {
        // FTS5: space between words = implicit AND (requires all terms)
        // Explicit "AND" also works but space is simpler and avoids edge cases
        // where "AND" itself could be misinterpreted by stemmer or locale.
        // Using space (implicit AND) for robustness.
        words.join(" ")
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Query sanitization tests ──

    #[test]
    fn test_sanitize_fts_query_basic() {
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
    }

    #[test]
    fn test_sanitize_fts_query_special_chars() {
        assert_eq!(sanitize_fts_query("JWT (RS256)"), "JWT RS256");
    }

    #[test]
    fn test_sanitize_fts_query_single_word() {
        assert_eq!(sanitize_fts_query("authentication"), "authentication");
    }

    #[test]
    fn test_sanitize_fts_query_empty() {
        assert_eq!(sanitize_fts_query(""), "");
        assert_eq!(sanitize_fts_query("   "), "");
    }

    #[test]
    fn test_sanitize_fts_query_short_words_filtered() {
        assert_eq!(sanitize_fts_query("I am a developer"), "am developer");
    }

    #[test]
    fn test_sanitize_fts_query_unicode() {
        assert_eq!(sanitize_fts_query("认证模块 JWT"), "认证模块 JWT");
    }

    #[test]
    fn test_sanitize_fts_query_hyphenated() {
        assert_eq!(sanitize_fts_query("rate-limiting approach"), "rate-limiting approach");
    }

    // ── BM25 search tests ──

    #[tokio::test]
    async fn test_bm25_search_empty_query() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let results = s.bm25_search("", &[0xAA; 32], 10).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_fts_index_and_search() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let rid = [0xBB; 32];

        // Index a record
        s.fts_index_record(&rid, &owner, "JWT authentication using RS256 algorithm", "technology").await;

        // Search for it
        let results = s.bm25_search("RS256", &owner, 10).await;
        assert!(!results.is_empty(), "BM25 should find RS256");
        assert_eq!(results[0].0, "record");
        assert_eq!(results[0].1, hex::encode(rid));

        // Search with different terms
        let results2 = s.bm25_search("authentication", &owner, 10).await;
        assert!(!results2.is_empty(), "BM25 should find authentication");

        // Search with non-matching terms
        let results3 = s.bm25_search("PostgreSQL", &owner, 10).await;
        assert!(results3.is_empty(), "BM25 should not find PostgreSQL");
    }

    #[tokio::test]
    async fn test_fts_index_entity_and_search() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.fts_index_entity("ent_001", &owner, "Project Alpha", Some("Authentication system"), "project").await;
        s.fts_index_entity("ent_002", &owner, "PostgreSQL", None, "technology").await;

        let results = s.bm25_search("Alpha", &owner, 10).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "entity");
        assert_eq!(results[0].1, "ent_001");
    }

    #[tokio::test]
    async fn test_fts_index_session_and_search() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.fts_index_session("sess_001", &owner, "Implemented JWT auth with RS256 for Project Alpha").await;

        let results = s.bm25_search("RS256", &owner, 10).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "session");
    }

    #[tokio::test]
    async fn test_fts_owner_isolation() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner_a = [0xAA; 32];
        let owner_b = [0xBB; 32];

        s.fts_index_record(&[0x01; 32], &owner_a, "secret data for Alice", "").await;
        s.fts_index_record(&[0x02; 32], &owner_b, "secret data for Bob", "").await;

        let results_a = s.bm25_search("secret", &owner_a, 10).await;
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0].1, hex::encode([0x01; 32]));

        let results_b = s.bm25_search("secret", &owner_b, 10).await;
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].1, hex::encode([0x02; 32]));
    }

    #[tokio::test]
    async fn test_fts_remove_record() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let rid = [0xCC; 32];

        s.fts_index_record(&rid, &owner, "temporary content", "").await;
        assert!(!s.bm25_search("temporary", &owner, 10).await.is_empty());

        s.fts_remove_record(&rid).await;
        assert!(s.bm25_search("temporary", &owner, 10).await.is_empty());
    }

    #[tokio::test]
    async fn test_fts_multi_source_search() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.fts_index_record(&[0x01; 32], &owner, "rate limiting with token bucket", "").await;
        s.fts_index_entity("ent_rl", &owner, "rate limiting", Some("Token bucket algorithm"), "technology").await;
        s.fts_index_session("sess_rl", &owner, "Discussed rate limiting approach using token bucket").await;

        // Space = implicit AND in FTS5, matches all three sources
        let results = s.bm25_search("rate limiting", &owner, 10).await;
        assert_eq!(results.len(), 3, "Should find record + entity + session");

        let types: Vec<&str> = results.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(types.contains(&"record"));
        assert!(types.contains(&"entity"));
        assert!(types.contains(&"session"));
    }

    // ── search_with_snippets tests (v2.5.0+Unify) ──

    #[tokio::test]
    async fn test_search_with_snippets_empty_query() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let results = s.search_with_snippets("", &[0xAA; 32], 10).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_with_snippets_basic() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.fts_index_record(
            &[0x01; 32], &owner,
            "JWT authentication using RS256 algorithm for secure token verification", "auth",
        ).await;

        let hits = s.search_with_snippets("JWT", &owner, 10).await;
        assert!(!hits.is_empty(), "Should find JWT in indexed content");
        assert_eq!(hits[0].source_type, "record");
        assert!(hits[0].snippet.contains("<mark>"), "Snippet should contain <mark> highlight tags");
        assert!(hits[0].score > 0.0, "Score should be positive (negated BM25)");
    }

    #[tokio::test]
    async fn test_search_with_snippets_session_resolves_session_id() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.fts_index_session("sess_test_001", &owner, "Implementing rate limiting with token bucket").await;

        let hits = s.search_with_snippets("token bucket", &owner, 10).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source_type, "session");
        // For session hits, session_id should be the source_id itself
        assert_eq!(hits[0].session_id, Some("sess_test_001".to_string()));
    }

    #[tokio::test]
    async fn test_search_with_snippets_owner_isolation() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner_a = [0xAA; 32];
        let owner_b = [0xBB; 32];

        s.fts_index_record(&[0x01; 32], &owner_a, "secret project for Alice", "").await;
        s.fts_index_record(&[0x02; 32], &owner_b, "secret project for Bob", "").await;

        let hits_a = s.search_with_snippets("secret", &owner_a, 10).await;
        assert_eq!(hits_a.len(), 1, "Owner A should only see their own results");

        let hits_b = s.search_with_snippets("secret", &owner_b, 10).await;
        assert_eq!(hits_b.len(), 1, "Owner B should only see their own results");
    }

    // ── group_hits_by_session tests (v2.5.0+Unify) ──

    #[tokio::test]
    async fn test_group_hits_by_session_empty() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let groups = s.group_hits_by_session(&[]).await;
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn test_group_hits_by_session_groups_correctly() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        let hits = vec![
            SearchHit {
                source_type: "session".into(), source_id: "sess_a".into(),
                snippet: "hit 1".into(), score: 2.5, session_id: Some("sess_a".into()),
            },
            SearchHit {
                source_type: "session".into(), source_id: "sess_a".into(),
                snippet: "hit 2".into(), score: 1.5, session_id: Some("sess_a".into()),
            },
            SearchHit {
                source_type: "record".into(), source_id: "rec_x".into(),
                snippet: "hit 3".into(), score: 3.0, session_id: None,
            },
        ];

        let groups = s.group_hits_by_session(&hits).await;
        assert_eq!(groups.len(), 2, "Should have 2 groups: sess_a and (ungrouped)");

        // Groups sorted by best_score descending
        // (ungrouped) has score 3.0, sess_a has score 2.5
        assert_eq!(groups[0].session_id, "(ungrouped)");
        assert_eq!(groups[0].best_score, 3.0);
        assert_eq!(groups[0].hits.len(), 1);

        assert_eq!(groups[1].session_id, "sess_a");
        assert_eq!(groups[1].best_score, 2.5);
        assert_eq!(groups[1].hits.len(), 2);
    }

    #[tokio::test]
    async fn test_group_hits_by_session_with_metadata() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        // Create a session in the DB so metadata resolution works
        s.upsert_session("sess_meta", &owner, "test-source", 5).await;
        s.update_session_summary("sess_meta", "Test summary", None, Some("Test Title")).await;

        let hits = vec![
            SearchHit {
                source_type: "session".into(), source_id: "sess_meta".into(),
                snippet: "matched".into(), score: 1.0, session_id: Some("sess_meta".into()),
            },
        ];

        let groups = s.group_hits_by_session(&hits).await;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].session_id, "sess_meta");
        assert_eq!(groups[0].session_title, Some("Test Title".to_string()));
    }
}
