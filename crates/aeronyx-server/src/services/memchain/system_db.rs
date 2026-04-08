// ============================================
// File: crates/aeronyx-server/src/services/memchain/system_db.rs
// ============================================
//! # SystemDb — Global Metadata Database
//!
//! ## Creation Reason
//! Part of the MemChain Multi-Tenant Architecture (v1.0).
//! Provides global metadata storage for the SaaS mode:
//! - Volume assignment mapping (owner → volume_id)
//! - LLM usage logging (cross-user aggregation)
//! - Authentication event logging (audit trail)
//!
//! ## Main Functionality
//! - Stores which volume each user's data lives on
//! - Tracks LLM token usage per owner for billing/quota
//! - Records authentication events for security auditing
//! - Provides active owner list for MinerScheduler scheduling
//!
//! ## Dependencies
//! - Used by VolumeRouter (Task 1a) for persistent volume assignment
//! - Used by StoragePool (Task 1b) for last_active_at updates
//! - Used by MinerScheduler (Task 4) for active owner scheduling
//! - Used by Auth layer (Task 2) for auth event logging
//! - Used by Admin endpoints (Task 5) for usage statistics
//!
//! ## Thread Safety
//! Internal `Arc<Mutex<Connection>>` — all blocking DB ops run inside
//! `tokio::task::spawn_blocking` to avoid blocking the async runtime.
//! The Arc is cloned into each closure so MutexGuard is never Send-crossed.
//!
//! ## Security Notes
//! - owner_pubkey stored as raw 32-byte BLOB (not hex) for efficiency
//! - This DB stores ONLY metadata — no user memory content
//! - auth_events table: only log token_issued and auth_fail, NOT every
//!   request (use update_last_active for request-level activity tracking)
//!
//! ⚠️ Important Note for Next Developer:
//! - spawn_blocking pattern: Arc::clone(&self.conn) INSIDE the async fn,
//!   then lock() INSIDE the spawn_blocking closure. Never hold a MutexGuard
//!   across an await point or pass it across spawn_blocking boundaries.
//! - owner_pubkey BLOB binding: always use owner.as_slice() not &owner[..]
//! - assign_volume is idempotent-safe via INSERT OR IGNORE + existence check
//! - update_last_active silently ignores unknown owners (no error returned)
//! - get_active_owners orders by last_active_at DESC — most recent API
//!   callers get Miner priority (fair scheduling by activity)
//! - log_auth_event: do NOT call for every request, only for significant
//!   auth events (token_issued, auth_fail). High-frequency tracking is
//!   handled by update_last_active() which has no table growth concern.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant - Initial implementation (Task 1a)
// ============================================

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use tokio::task;
use tracing::{debug, error, info, warn};

// ============================================
// Public Types
// ============================================

/// Summary of a recently active owner, used by MinerScheduler.
#[derive(Debug, Clone)]
pub struct ActiveOwner {
    /// Ed25519 public key bytes (32 bytes).
    pub pubkey: [u8; 32],
    /// Volume this owner's data lives on.
    pub volume_id: String,
    /// Unix timestamp of most recent API request.
    pub last_active_at: i64,
}

/// Per-owner LLM usage totals for a time period.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OwnerUsageStats {
    /// Ed25519 public key bytes (32 bytes).
    pub owner_pubkey: [u8; 32],
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub call_count: u64,
}

/// Errors that can arise from SystemDb operations.
#[derive(Debug, thiserror::Error)]
pub enum SystemDbError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Owner already assigned to volume {0}")]
    AlreadyAssigned(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

// ============================================
// SystemDb
// ============================================

/// Global metadata database for the SaaS multi-tenant mode.
///
/// Stores volume assignments, LLM usage logs, and auth audit events.
/// Does NOT store any user memory content — only operational metadata.
///
/// ## Thread Safety
/// `Arc<Mutex<Connection>>` with all DB operations run in `spawn_blocking`.
/// The Arc is cloned per-call so the MutexGuard never crosses async boundaries.
pub struct SystemDb {
    /// Inner SQLite connection, wrapped in Arc<Mutex> so it can be
    /// cloned into spawn_blocking closures without lifetime issues.
    ///
    /// SECURITY: Arc clone, not clone of Connection. MutexGuard is locked
    /// only inside the spawn_blocking closure — never held across await.
    conn: Arc<Mutex<Connection>>,
}

impl SystemDb {
    // ============================================
    // Initialization
    // ============================================

    /// Open or create the system.db file, initializing schema automatically.
    ///
    /// Enables WAL journal mode and sets busy_timeout = 5000ms.
    /// Idempotent: safe to call on an existing database.
    ///
    /// # Errors
    /// Returns `SystemDbError::Db` if the file cannot be opened or the
    /// schema cannot be initialized.
    pub async fn open(path: &Path) -> Result<Arc<Self>, SystemDbError> {
        // Clone the path into an owned value for the spawn_blocking closure.
        let path_owned = path.to_path_buf();

        let conn = task::spawn_blocking(move || -> Result<Connection, rusqlite::Error> {
            // Create parent directories if needed.
            if let Some(parent) = path_owned.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                            Some(format!("create_dir_all: {}", e)),
                        )
                    })?;
                }
            }

            let conn = Connection::open(&path_owned)?;

            // WAL mode: allows concurrent readers alongside a single writer.
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA busy_timeout = 5000;
                 PRAGMA foreign_keys = ON;"
            )?;

            Self::create_schema_sync(&conn)?;

            Ok(conn)
        }).await??;

        info!(
            path = %path.display(),
            "[SYSTEM_DB] ✅ Opened"
        );

        Ok(Arc::new(Self {
            conn: Arc::new(Mutex::new(conn)),
        }))
    }

    /// Create all required tables (idempotent via IF NOT EXISTS).
    fn create_schema_sync(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(
            "-- Volume assignment: owner → volume_id mapping
            CREATE TABLE IF NOT EXISTS volume_assignments (
                owner_pubkey    BLOB NOT NULL PRIMARY KEY,   -- [u8; 32] Ed25519 pubkey
                volume_id       TEXT NOT NULL,
                assigned_at     INTEGER NOT NULL,            -- Unix timestamp
                last_active_at  INTEGER NOT NULL             -- Updated per API request
            );

            CREATE INDEX IF NOT EXISTS idx_va_volume
                ON volume_assignments(volume_id);

            CREATE INDEX IF NOT EXISTS idx_va_active
                ON volume_assignments(last_active_at);

            -- Global LLM usage log (aggregated across users)
            CREATE TABLE IF NOT EXISTS global_llm_usage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_pubkey    BLOB NOT NULL,               -- [u8; 32]
                model           TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL,
                output_tokens   INTEGER NOT NULL,
                task_type       TEXT NOT NULL,               -- reflection | summary | conflict | ...
                created_at      INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_owner
                ON global_llm_usage(owner_pubkey, created_at);

            CREATE INDEX IF NOT EXISTS idx_usage_time
                ON global_llm_usage(created_at);

            -- Authentication audit log
            -- ⚠️ Only log token_issued / auth_fail here, NOT every request.
            --    Per-request activity tracking uses volume_assignments.last_active_at.
            CREATE TABLE IF NOT EXISTS auth_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_pubkey    BLOB NOT NULL,               -- [u8; 32]
                event_type      TEXT NOT NULL,               -- token_issued | auth_fail
                ip_address      TEXT,
                created_at      INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_auth_owner
                ON auth_events(owner_pubkey, created_at);"
        )?;

        debug!("[SYSTEM_DB] Schema initialized");
        Ok(())
    }

    // ============================================
    // Volume Assignment
    // ============================================

    /// Look up which volume a given owner has been assigned to.
    ///
    /// Returns `None` for new users who have not been assigned yet.
    pub async fn get_assignment(
        &self,
        owner: &[u8; 32],
    ) -> Result<Option<String>, SystemDbError> {
        let owner = *owner;
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.query_row(
                "SELECT volume_id FROM volume_assignments WHERE owner_pubkey = ?1",
                params![owner.as_slice()],
                |row| row.get::<_, String>(0),
            ).optional()
        }).await?
        .map_err(SystemDbError::Db)
    }

    /// Persistently assign an owner to a volume.
    ///
    /// # Errors
    /// - `AlreadyAssigned` if the owner already has an assignment.
    ///   Callers should call `get_assignment` first, or use `assign_if_new`.
    ///
    /// # Concurrency
    /// Uses `INSERT OR IGNORE` to be safe against races, then checks
    /// affected rows to distinguish "new insert" from "already existed".
    pub async fn assign_volume(
        &self,
        owner: &[u8; 32],
        volume_id: &str,
    ) -> Result<(), SystemDbError> {
        let owner = *owner;
        let volume_id = volume_id.to_string();
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<(), SystemDbError> {
            let conn = conn.lock().unwrap();
            let now = now_unix();

            // INSERT OR IGNORE: safe against concurrent assign attempts.
            // We check affected rows to distinguish new vs duplicate.
            let affected = conn.execute(
                "INSERT OR IGNORE INTO volume_assignments
                    (owner_pubkey, volume_id, assigned_at, last_active_at)
                 VALUES (?1, ?2, ?3, ?3)",
                params![owner.as_slice(), volume_id, now],
            )?;

            if affected == 0 {
                // Row already existed — retrieve the existing volume_id.
                let existing: String = conn.query_row(
                    "SELECT volume_id FROM volume_assignments WHERE owner_pubkey = ?1",
                    params![owner.as_slice()],
                    |row| row.get(0),
                )?;
                return Err(SystemDbError::AlreadyAssigned(existing));
            }

            Ok(())
        }).await?
    }

    /// Count how many users are assigned to each volume.
    ///
    /// Returns `Vec<(volume_id, user_count)>` sorted by volume_id.
    /// Used by VolumeRouter to select the least-loaded writable volume.
    pub async fn count_users_per_volume(
        &self,
    ) -> Result<Vec<(String, usize)>, SystemDbError> {
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<Vec<(String, usize)>, rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT volume_id, COUNT(*) as cnt
                 FROM volume_assignments
                 GROUP BY volume_id
                 ORDER BY volume_id"
            )?;

            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
            })?;

            rows.collect::<Result<Vec<_>, _>>()
        }).await?
        .map_err(SystemDbError::Db)
    }

    /// Update the last-active timestamp for an owner.
    ///
    /// Called on every authenticated API request.
    /// Silently ignores unknown owners (they may not be assigned yet,
    /// or the assignment may live in a different DB shard in future).
    ///
    /// # Performance
    /// High-frequency write. Uses a direct UPDATE (not INSERT OR REPLACE)
    /// to avoid re-writing the full row on every call. WAL mode ensures
    /// this does not block concurrent readers.
    pub async fn update_last_active(
        &self,
        owner: &[u8; 32],
    ) -> Result<(), SystemDbError> {
        let owner = *owner;
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<(), rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let now = now_unix();
            // Silently ignore if owner not found (0 rows updated is fine).
            conn.execute(
                "UPDATE volume_assignments SET last_active_at = ?1
                 WHERE owner_pubkey = ?2",
                params![now, owner.as_slice()],
            )?;
            Ok(())
        }).await?
        .map_err(SystemDbError::Db)
    }

    /// Retrieve the most recently active owners, ordered by last_active_at DESC.
    ///
    /// Used by MinerScheduler to select which users to process each tick.
    /// `limit` controls how many owners are returned.
    pub async fn get_active_owners(
        &self,
        limit: usize,
    ) -> Result<Vec<ActiveOwner>, SystemDbError> {
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<Vec<ActiveOwner>, rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT owner_pubkey, volume_id, last_active_at
                 FROM volume_assignments
                 ORDER BY last_active_at DESC
                 LIMIT ?1"
            )?;

            let rows = stmt.query_map(params![limit as i64], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let volume_id: String = row.get(1)?;
                let last_active_at: i64 = row.get(2)?;

                // Validate blob length — corrupt rows are skipped.
                if blob.len() != 32 {
                    return Err(rusqlite::Error::InvalidColumnType(
                        0,
                        "owner_pubkey".into(),
                        rusqlite::types::Type::Blob,
                    ));
                }

                let mut pubkey = [0u8; 32];
                pubkey.copy_from_slice(&blob);
                Ok(ActiveOwner { pubkey, volume_id, last_active_at })
            })?;

            // Filter out any rows that failed to parse (corrupt data).
            let owners: Vec<ActiveOwner> = rows
                .filter_map(|r| match r {
                    Ok(o) => Some(o),
                    Err(e) => {
                        warn!("[SYSTEM_DB] Skipping corrupt active_owner row: {}", e);
                        None
                    }
                })
                .collect();

            Ok(owners)
        }).await?
        .map_err(SystemDbError::Db)
    }

    /// Load all owner → volume_id assignments into memory at startup.
    ///
    /// Called once during VolumeRouter initialization to populate the
    /// in-memory DashMap cache from persistent state.
    pub async fn load_all_assignments(
        &self,
    ) -> Result<Vec<([u8; 32], String)>, SystemDbError> {
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<Vec<([u8; 32], String)>, rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT owner_pubkey, volume_id FROM volume_assignments"
            )?;

            let rows = stmt.query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let volume_id: String = row.get(1)?;

                if blob.len() != 32 {
                    return Err(rusqlite::Error::InvalidColumnType(
                        0,
                        "owner_pubkey".into(),
                        rusqlite::types::Type::Blob,
                    ));
                }

                let mut pubkey = [0u8; 32];
                pubkey.copy_from_slice(&blob);
                Ok((pubkey, volume_id))
            })?;

            rows.filter_map(|r| match r {
                Ok(pair) => Some(Ok(pair)),
                Err(e) => {
                    warn!("[SYSTEM_DB] Skipping corrupt assignment row: {}", e);
                    None
                }
            })
            .collect::<Result<Vec<_>, _>>()
        }).await?
        .map_err(SystemDbError::Db)
    }

    // ============================================
    // LLM Usage Logging
    // ============================================

    /// Record a single LLM API call's token usage.
    ///
    /// Called after each successful LLM call by TaskWorker / LlmRouter.
    /// `task_type` matches CognitiveTaskType string representations
    /// (e.g., "session_title", "community_narrative", "conflict_resolution").
    pub async fn log_llm_usage(
        &self,
        owner: &[u8; 32],
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        task_type: &str,
    ) -> Result<(), SystemDbError> {
        let owner = *owner;
        let model = model.to_string();
        let task_type = task_type.to_string();
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<(), rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let now = now_unix();
            conn.execute(
                "INSERT INTO global_llm_usage
                    (owner_pubkey, model, input_tokens, output_tokens, task_type, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    owner.as_slice(),
                    model,
                    input_tokens as i64,
                    output_tokens as i64,
                    task_type,
                    now,
                ],
            )?;
            Ok(())
        }).await?
        .map_err(SystemDbError::Db)
    }

    /// Aggregate LLM usage statistics for a time range.
    ///
    /// Returns per-owner totals for the given `[since, until]` window.
    /// Both timestamps are Unix seconds (inclusive).
    ///
    /// Used by `GET /api/admin/usage` endpoint.
    pub async fn get_usage_stats(
        &self,
        since: i64,
        until: i64,
    ) -> Result<Vec<OwnerUsageStats>, SystemDbError> {
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<Vec<OwnerUsageStats>, rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT owner_pubkey,
                        SUM(input_tokens),
                        SUM(output_tokens),
                        COUNT(*)
                 FROM global_llm_usage
                 WHERE created_at >= ?1 AND created_at <= ?2
                 GROUP BY owner_pubkey
                 ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC"
            )?;

            let rows = stmt.query_map(params![since, until], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let input: i64 = row.get(1)?;
                let output: i64 = row.get(2)?;
                let count: i64 = row.get(3)?;

                if blob.len() != 32 {
                    return Err(rusqlite::Error::InvalidColumnType(
                        0,
                        "owner_pubkey".into(),
                        rusqlite::types::Type::Blob,
                    ));
                }

                let mut owner_pubkey = [0u8; 32];
                owner_pubkey.copy_from_slice(&blob);

                Ok(OwnerUsageStats {
                    owner_pubkey,
                    total_input_tokens: input.unsigned_abs(),
                    total_output_tokens: output.unsigned_abs(),
                    call_count: count.unsigned_abs(),
                })
            })?;

            rows.filter_map(|r| match r {
                Ok(s) => Some(Ok(s)),
                Err(e) => {
                    warn!("[SYSTEM_DB] Skipping corrupt usage row: {}", e);
                    None
                }
            })
            .collect::<Result<Vec<_>, _>>()
        }).await?
        .map_err(SystemDbError::Db)
    }

    // ============================================
    // Auth Event Logging
    // ============================================

    /// Record an authentication event for audit purposes.
    ///
    /// ## When to call
    /// - `event_type = "token_issued"`: after successfully issuing a JWT
    /// - `event_type = "auth_fail"`: after a failed authentication attempt
    ///
    /// ## When NOT to call
    /// Do NOT call for every API request — that would cause unbounded table
    /// growth. Per-request activity is tracked via `update_last_active()`.
    ///
    /// `ip_address` is optional and may be None if not available.
    pub async fn log_auth_event(
        &self,
        owner: &[u8; 32],
        event_type: &str,
        ip_address: Option<&str>,
    ) -> Result<(), SystemDbError> {
        let owner = *owner;
        let event_type = event_type.to_string();
        let ip_address = ip_address.map(|s| s.to_string());
        let conn = Arc::clone(&self.conn);

        task::spawn_blocking(move || -> Result<(), rusqlite::Error> {
            let conn = conn.lock().unwrap();
            let now = now_unix();
            conn.execute(
                "INSERT INTO auth_events
                    (owner_pubkey, event_type, ip_address, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    owner.as_slice(),
                    event_type,
                    ip_address,
                    now,
                ],
            )?;
            Ok(())
        }).await?
        .map_err(SystemDbError::Db)
    }
}

// ============================================
// Private Helpers
// ============================================

/// Current Unix timestamp in seconds.
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a SystemDb in a temp directory.
    async fn open_temp_db() -> (TempDir, Arc<SystemDb>) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("system.db");
        let db = SystemDb::open(&db_path).await.unwrap();
        (dir, db)
    }

    /// Generate a deterministic test owner from a seed byte.
    fn make_owner(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    // ── Schema ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_open_creates_tables() {
        let (_dir, db) = open_temp_db().await;
        let conn = Arc::clone(&db.conn);

        let tables = task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                )
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect::<Vec<_>>()
        })
        .await
        .unwrap();

        assert!(tables.contains(&"volume_assignments".to_string()));
        assert!(tables.contains(&"global_llm_usage".to_string()));
        assert!(tables.contains(&"auth_events".to_string()));
    }

    // ── Volume Assignment ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_assign_and_get() {
        let (_dir, db) = open_temp_db().await;
        let owner = make_owner(0xAA);

        // No assignment yet.
        assert!(db.get_assignment(&owner).await.unwrap().is_none());

        // Assign.
        db.assign_volume(&owner, "vol-001").await.unwrap();

        // Now should return the volume.
        let vol = db.get_assignment(&owner).await.unwrap();
        assert_eq!(vol, Some("vol-001".to_string()));
    }

    #[tokio::test]
    async fn test_assign_duplicate_returns_error() {
        let (_dir, db) = open_temp_db().await;
        let owner = make_owner(0xBB);

        db.assign_volume(&owner, "vol-001").await.unwrap();

        let err = db.assign_volume(&owner, "vol-002").await.unwrap_err();
        match err {
            SystemDbError::AlreadyAssigned(vol) => {
                assert_eq!(vol, "vol-001");
            }
            other => panic!("Expected AlreadyAssigned, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_count_users_per_volume() {
        let (_dir, db) = open_temp_db().await;

        // 3 users on vol-001, 2 on vol-002.
        for i in 0u8..3 {
            db.assign_volume(&make_owner(i), "vol-001").await.unwrap();
        }
        for i in 3u8..5 {
            db.assign_volume(&make_owner(i), "vol-002").await.unwrap();
        }

        let counts = db.count_users_per_volume().await.unwrap();
        let map: std::collections::HashMap<_, _> = counts.into_iter().collect();
        assert_eq!(map["vol-001"], 3);
        assert_eq!(map["vol-002"], 2);
    }

    #[tokio::test]
    async fn test_update_last_active() {
        let (_dir, db) = open_temp_db().await;
        let owner_a = make_owner(0xAA);
        let owner_b = make_owner(0xBB);

        db.assign_volume(&owner_a, "vol-001").await.unwrap();
        db.assign_volume(&owner_b, "vol-001").await.unwrap();

        // Touch owner_b more recently.
        db.update_last_active(&owner_a).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        db.update_last_active(&owner_b).await.unwrap();

        let active = db.get_active_owners(2).await.unwrap();
        assert_eq!(active.len(), 2);
        // Most recent first.
        assert_eq!(active[0].pubkey, owner_b);
        assert_eq!(active[1].pubkey, owner_a);
    }

    #[tokio::test]
    async fn test_update_last_active_nonexistent_owner_is_noop() {
        let (_dir, db) = open_temp_db().await;
        let unknown = make_owner(0xFF);
        // Should not error even if owner is not in DB.
        db.update_last_active(&unknown).await.unwrap();
    }

    #[tokio::test]
    async fn test_load_all_assignments() {
        let (_dir, db) = open_temp_db().await;

        let pairs: Vec<([u8; 32], &str)> = vec![
            (make_owner(0x01), "vol-001"),
            (make_owner(0x02), "vol-001"),
            (make_owner(0x03), "vol-002"),
        ];

        for (owner, vol) in &pairs {
            db.assign_volume(owner, vol).await.unwrap();
        }

        let loaded = db.load_all_assignments().await.unwrap();
        assert_eq!(loaded.len(), 3);

        // Verify all pairs are present (order not guaranteed).
        for (owner, vol) in &pairs {
            assert!(
                loaded.iter().any(|(o, v)| o == owner && v == vol),
                "Missing assignment: {:?} → {}",
                owner,
                vol
            );
        }
    }

    // ── LLM Usage ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_log_llm_usage_and_stats() {
        let (_dir, db) = open_temp_db().await;
        let owner_a = make_owner(0xAA);
        let owner_b = make_owner(0xBB);

        let now = now_unix();

        // owner_a: 2 calls
        db.log_llm_usage(&owner_a, "gpt-4o", 100, 50, "session_title").await.unwrap();
        db.log_llm_usage(&owner_a, "gpt-4o", 200, 80, "community_narrative").await.unwrap();
        // owner_b: 1 call
        db.log_llm_usage(&owner_b, "claude-3", 150, 60, "session_title").await.unwrap();

        let stats = db.get_usage_stats(now - 10, now + 10).await.unwrap();
        assert_eq!(stats.len(), 2);

        let a_stat = stats.iter().find(|s| s.owner_pubkey == owner_a).unwrap();
        assert_eq!(a_stat.total_input_tokens, 300);
        assert_eq!(a_stat.total_output_tokens, 130);
        assert_eq!(a_stat.call_count, 2);

        let b_stat = stats.iter().find(|s| s.owner_pubkey == owner_b).unwrap();
        assert_eq!(b_stat.total_input_tokens, 150);
        assert_eq!(b_stat.total_output_tokens, 60);
        assert_eq!(b_stat.call_count, 1);
    }

    #[tokio::test]
    async fn test_usage_stats_outside_window_returns_empty() {
        let (_dir, db) = open_temp_db().await;
        let owner = make_owner(0xAA);

        db.log_llm_usage(&owner, "gpt-4o", 100, 50, "session_title").await.unwrap();

        // Query a window far in the future.
        let future = now_unix() + 100_000;
        let stats = db.get_usage_stats(future, future + 3600).await.unwrap();
        assert!(stats.is_empty());
    }

    // ── Auth Events ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_log_auth_event_succeeds() {
        let (_dir, db) = open_temp_db().await;
        let owner = make_owner(0xAA);

        // Both event types with and without IP.
        db.log_auth_event(&owner, "token_issued", Some("192.168.1.1")).await.unwrap();
        db.log_auth_event(&owner, "auth_fail", None).await.unwrap();

        // Verify rows were inserted.
        let conn = Arc::clone(&db.conn);
        let count: i64 = task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.query_row(
                "SELECT COUNT(*) FROM auth_events WHERE owner_pubkey = ?1",
                params![owner.as_slice()],
                |row| row.get(0),
            )
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(count, 2);
    }

    // ── get_active_owners limit ────────────────────────────────────────

    #[tokio::test]
    async fn test_get_active_owners_respects_limit() {
        let (_dir, db) = open_temp_db().await;

        for i in 0u8..10 {
            db.assign_volume(&make_owner(i), "vol-001").await.unwrap();
        }

        let active = db.get_active_owners(3).await.unwrap();
        assert_eq!(active.len(), 3);
    }

    // ── Idempotent open ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_open_twice_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("system.db");

        let db1 = SystemDb::open(&db_path).await.unwrap();
        db1.assign_volume(&make_owner(0xAA), "vol-001").await.unwrap();
        drop(db1);

        // Re-open same file — should not lose data.
        let db2 = SystemDb::open(&db_path).await.unwrap();
        let vol = db2.get_assignment(&make_owner(0xAA)).await.unwrap();
        assert_eq!(vol, Some("vol-001".to_string()));
    }
}
