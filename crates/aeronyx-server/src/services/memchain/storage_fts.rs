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
//! - FTS5 MATCH syntax: simple words are OR'd by default. Use "word1 word2"
//!   for implicit AND. Use "word1 OR word2" for explicit OR.
//! - When records are encrypted (record_key is set), FTS5 indexes the
//!   PLAINTEXT content (decrypted at insert time). The FTS index itself
//!   is NOT encrypted — this is acceptable because the DB file is local-only.
//! - fts_reindex_all() is expensive (full table scan). Only call at startup
//!   or when migration detects missing FTS data.
//!
//! ## Last Modified
//! v2.4.0+BM25 - 🌟 Initial implementation

use rusqlite::params;
use tracing::{debug, info, warn};

use super::storage::MemoryStorage;

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
    /// - "JWT authentication" → matches documents containing both stems
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

    let mut stmt = match conn.prepare(
        "SELECT source_type, source_id, -bm25(fts_index, 0, 0, 0, 1, 0) as score
         FROM fts_index
         WHERE fts_index MATCH ?1 AND owner_hex = ?2
         ORDER BY score DESC
         LIMIT ?3"
    ) {
        Ok(s) => s,
        Err(e) => {
            debug!("[BM25] Query prepare failed: {}", e);
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
    pub async fn fts_reindex_all(&self, owner: &[u8; 32]) {
        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;

        // Clear existing FTS data for this owner
        let _ = conn.execute(
            "DELETE FROM fts_index WHERE owner_hex = ?1",
            params![owner_hex],
        );

        // Index records
        let records_indexed = conn.execute(
            "INSERT INTO fts_index (source_type, source_id, owner_hex, content, tags)
             SELECT 'record', hex(record_id), ?1, encrypted_content, topic_tags
             FROM records
             WHERE owner = ?2 AND status = 0 AND encrypted_content != x''",
            params![owner_hex, owner.as_slice()],
        ).unwrap_or(0);

        // Index entities
        let entities_indexed = conn.execute(
            "INSERT INTO fts_index (source_type, source_id, owner_hex, content, tags)
             SELECT 'entity', entity_id, ?1,
                    CASE WHEN description IS NOT NULL THEN name || ' ' || description ELSE name END,
                    entity_type
             FROM entities
             WHERE owner = ?2",
            params![owner_hex, owner.as_slice()],
        ).unwrap_or(0);

        // Index sessions
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
/// Also converts the query to implicit AND (FTS5 default is OR for multiple words,
/// but AND gives better precision for recall).
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

    // FTS5 implicit AND: wrap each word in quotes to prevent OR behavior
    // "word1" "word2" → must match both stems
    // For single words, just return as-is (FTS5 handles it)
    if words.len() == 1 {
        words[0].to_string()
    } else {
        // Use AND between words for better precision
        words.join(" AND ")
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_fts_query_basic() {
        assert_eq!(sanitize_fts_query("hello world"), "hello AND world");
    }

    #[test]
    fn test_sanitize_fts_query_special_chars() {
        assert_eq!(sanitize_fts_query("JWT (RS256)"), "JWT AND RS256");
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
        assert_eq!(sanitize_fts_query("I am a developer"), "am AND developer");
    }

    #[test]
    fn test_sanitize_fts_query_unicode() {
        assert_eq!(sanitize_fts_query("认证模块 JWT"), "认证模块 AND JWT");
    }

    #[test]
    fn test_sanitize_fts_query_hyphenated() {
        assert_eq!(sanitize_fts_query("rate-limiting approach"), "rate-limiting AND approach");
    }

    // Integration tests require MemoryStorage with FTS5 — tested via end-to-end

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

        let results = s.bm25_search("rate AND limiting", &owner, 10).await;
        assert_eq!(results.len(), 3, "Should find record + entity + session");

        let types: Vec<&str> = results.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(types.contains(&"record"));
        assert!(types.contains(&"entity"));
        assert!(types.contains(&"session"));
    }
}
