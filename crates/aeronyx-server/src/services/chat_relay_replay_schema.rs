// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_replay_schema.rs
// ============================================
// Version: 1.0.0-ReplaySchemaMigrationDomain
//
// Creation Reason:
//   [CHAT-RELAY-REPLAY-SCHEMA-DOMAIN 2026-08-27 by Codex] Extract the
//   verified-submit and blind-route replay schema migrations from the
//   oversized relay orchestration service.
//
// Main Functionality:
//   - Models immutable feature-version and retention requirements.
//   - Defines a replaceable replay-schema migration capability.
//   - Installs both private replay stores in immediate SQLite transactions.
//   - Migrates legacy ownership/effect state without weakening replay fences.
//   - Validates installation markers, required columns, and every stored row.
//   - Applies bounded startup retention only after migration validation.
//
// Dependencies:
//   - `chat_relay.rs` supplies stable schema versions and retention policy.
//   - `chat_relay_error.rs` supplies typed fail-closed storage failures.
//   - `rusqlite` supplies immediate transactions and schema introspection.
//
// Main Logical Flow:
//   1. Snapshot whether the expected replay tables existed before migration.
//   2. Open one immediate transaction and install the current table shape.
//   3. Validate the durable feature marker before adding legacy columns.
//   4. Normalize old owner/effect state and validate all retained rows.
//   5. Advance the feature marker, prune expired evidence, and commit.
//
// Important Note for Next Developer:
//   - An installed marker with missing durable state is corruption, not setup.
//   - Preserve the exact v1/v2 owner and side-effect migration semantics.
//   - Never persist request IDs, message IDs, routes, peers, or plaintext.
//   - Migration and marker advancement must remain in the same transaction.
//   - Do not widen accepted versions without an explicit tested migration.
//
// Last Modified:
//   v1.0.0-ReplaySchemaMigrationDomain - Initial composed SQLite migrator
// ============================================

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Version contract for one durable private replay feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReplaySchemaVersion {
    feature: &'static str,
    legacy: i64,
    intermediate: i64,
    current: i64,
}

impl ReplaySchemaVersion {
    pub(crate) const fn new(
        feature: &'static str,
        legacy: i64,
        intermediate: i64,
        current: i64,
    ) -> Self {
        Self {
            feature,
            legacy,
            intermediate,
            current,
        }
    }

    fn accepts(self, installed: i64) -> bool {
        installed == self.legacy || installed == self.intermediate || installed == self.current
    }

    fn needs_migration(self, installed: Option<i64>) -> bool {
        installed.is_some_and(|version| version != self.current)
    }
}

/// Immutable schema and retention contract supplied by relay composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReplaySchemaContract {
    verified_submit: ReplaySchemaVersion,
    blind_route: ReplaySchemaVersion,
    verified_submit_ttl_secs: u64,
    blind_route_ttl_secs: u64,
    owner_epoch_bytes: usize,
}

impl ReplaySchemaContract {
    pub(crate) const fn new(
        verified_submit: ReplaySchemaVersion,
        blind_route: ReplaySchemaVersion,
        verified_submit_ttl_secs: u64,
        blind_route_ttl_secs: u64,
        owner_epoch_bytes: usize,
    ) -> Self {
        Self {
            verified_submit,
            blind_route,
            verified_submit_ttl_secs,
            blind_route_ttl_secs,
            owner_epoch_bytes,
        }
    }
}

/// Capability that brings both replay stores to their required durable shape.
pub(crate) trait ChatRelayReplaySchemaMigration {
    fn migrate_verified_submit(&self, connection: &mut Connection, now: u64)
        -> ChatRelayResult<()>;

    fn migrate_blind_route(&self, connection: &mut Connection, now: u64) -> ChatRelayResult<()>;
}

/// Production SQLite implementation of the replay-schema migration boundary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SqliteChatRelayReplaySchemaMigrator {
    contract: ReplaySchemaContract,
}

impl SqliteChatRelayReplaySchemaMigrator {
    pub(crate) const fn new(contract: ReplaySchemaContract) -> Self {
        Self { contract }
    }

    fn migrate_verified_submit_store(
        &self,
        connection: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        // [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex]
        // Completed responses and unfinished reservations contain only
        // node-secret HMACs, sealed bytes, timestamps, and random ownership.
        let response_table_existed = table_exists(connection, "relay_verified_submit_responses")?;
        let reservation_table_existed =
            table_exists(connection, "relay_verified_submit_reservations")?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_verified_submit_responses (
                cache_key            BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                envelope_fingerprint BLOB    NOT NULL CHECK(LENGTH(envelope_fingerprint) = 32),
                response_nonce       BLOB    NOT NULL CHECK(LENGTH(response_nonce) = 24),
                response_ciphertext  BLOB    NOT NULL CHECK(
                    LENGTH(response_ciphertext) > 16
                    AND LENGTH(response_ciphertext) <= 528
                ),
                completed_at         INTEGER NOT NULL CHECK(completed_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_verified_submit_response_retention
                ON relay_verified_submit_responses(completed_at);

            CREATE TABLE IF NOT EXISTS relay_verified_submit_reservations (
                cache_key            BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                envelope_fingerprint BLOB    NOT NULL CHECK(LENGTH(envelope_fingerprint) = 32),
                reserved_at          INTEGER NOT NULL CHECK(reserved_at >= 0),
                owner_epoch          BLOB    NOT NULL CHECK(LENGTH(owner_epoch) = 16),
                owner_acquired_at    INTEGER NOT NULL CHECK(owner_acquired_at >= reserved_at)
            );
            CREATE INDEX IF NOT EXISTS idx_verified_submit_reservation_retention
                ON relay_verified_submit_reservations(reserved_at);
            ",
        )?;

        let schema = self.contract.verified_submit;
        let installed_version = installed_version(&tx, schema.feature)?;
        if installed_version.is_some_and(|version| !schema.accepts(version)) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_installation_version",
            });
        }
        if !response_table_existed && installed_version.is_some() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_table",
            });
        }
        if matches!(installed_version, Some(version) if version == schema.intermediate || version == schema.current)
            && !reservation_table_existed
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_table",
            });
        }

        let owner_epoch_exists =
            column_exists(&tx, "relay_verified_submit_reservations", "owner_epoch")?;
        let owner_acquired_at_exists = column_exists(
            &tx,
            "relay_verified_submit_reservations",
            "owner_acquired_at",
        )?;
        if installed_version == Some(schema.current)
            && (!owner_epoch_exists || !owner_acquired_at_exists)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_columns",
            });
        }
        if !owner_epoch_exists {
            tx.execute_batch(
                "ALTER TABLE relay_verified_submit_reservations
                 ADD COLUMN owner_epoch BLOB",
            )?;
        }
        if !owner_acquired_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_verified_submit_reservations
                 ADD COLUMN owner_acquired_at INTEGER",
            )?;
        }
        if installed_version != Some(schema.current) {
            // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] A pre-v3
            // owner is foreign; immutable reservation age starts its lease.
            tx.execute(
                "UPDATE relay_verified_submit_reservations
                 SET owner_epoch = zeroblob(?1), owner_acquired_at = reserved_at",
                params![sqlite_integer_from_usize(self.contract.owner_epoch_bytes)],
            )?;
        }

        let invalid_responses = tx.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 528
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_responses != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_row_shape",
            });
        }
        let invalid_reservations = tx.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_row_shape",
            });
        }

        install_or_advance_marker(
            &tx,
            schema,
            installed_version,
            now,
            "verified_submit_response_schema_installed_at",
            "verified_submit_response_installation_marker",
            "verified_submit_response_migration_marker",
        )?;
        let cutoff = sqlite_integer(
            now.saturating_sub(self.contract.verified_submit_ttl_secs),
            "verified_submit_response_startup_cutoff",
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn migrate_blind_route_store(
        &self,
        connection: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Installed markers
        // with missing tables are corruption because recreation would erase
        // the route side-effect boundary after an operator accident.
        let response_table_existed = table_exists(connection, "relay_blind_route_responses")?;
        let reservation_table_existed = table_exists(connection, "relay_blind_route_reservations")?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_blind_route_responses (
                cache_key           BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                request_fingerprint BLOB    NOT NULL CHECK(LENGTH(request_fingerprint) = 32),
                response_nonce      BLOB    NOT NULL CHECK(LENGTH(response_nonce) = 24),
                response_ciphertext BLOB    NOT NULL CHECK(
                    LENGTH(response_ciphertext) > 16
                    AND LENGTH(response_ciphertext) <= 2064
                ),
                completed_at        INTEGER NOT NULL CHECK(completed_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_blind_route_response_retention
                ON relay_blind_route_responses(completed_at);

            CREATE TABLE IF NOT EXISTS relay_blind_route_reservations (
                cache_key           BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                request_fingerprint BLOB    NOT NULL CHECK(LENGTH(request_fingerprint) = 32),
                reserved_at         INTEGER NOT NULL CHECK(reserved_at >= 0),
                owner_epoch         BLOB    NOT NULL CHECK(LENGTH(owner_epoch) = 16),
                owner_acquired_at   INTEGER NOT NULL CHECK(owner_acquired_at >= reserved_at),
                effect_started_at   INTEGER CHECK(
                    effect_started_at IS NULL OR effect_started_at >= reserved_at
                )
            );
            CREATE INDEX IF NOT EXISTS idx_blind_route_reservation_retention
                ON relay_blind_route_reservations(reserved_at);
            ",
        )?;

        let schema = self.contract.blind_route;
        let installed_version = installed_version(&tx, schema.feature)?;
        if installed_version.is_some_and(|version| !schema.accepts(version)) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_installation_version",
            });
        }
        if installed_version.is_some() && (!response_table_existed || !reservation_table_existed) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_table",
            });
        }

        let owner_epoch_exists =
            column_exists(&tx, "relay_blind_route_reservations", "owner_epoch")?;
        let owner_acquired_at_exists =
            column_exists(&tx, "relay_blind_route_reservations", "owner_acquired_at")?;
        let effect_started_at_exists =
            column_exists(&tx, "relay_blind_route_reservations", "effect_started_at")?;
        if installed_version == Some(schema.current)
            && (!owner_epoch_exists || !owner_acquired_at_exists || !effect_started_at_exists)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_reservation_columns",
            });
        }
        if !owner_epoch_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN owner_epoch BLOB",
            )?;
        }
        if !effect_started_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN effect_started_at INTEGER",
            )?;
        }
        if !owner_acquired_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN owner_acquired_at INTEGER",
            )?;
        }
        if installed_version == Some(schema.legacy) {
            // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Legacy
            // claims are armed because their external effect is ambiguous.
            tx.execute(
                "UPDATE relay_blind_route_reservations
                 SET owner_epoch = zeroblob(?1), effect_started_at = reserved_at",
                params![sqlite_integer_from_usize(self.contract.owner_epoch_bytes)],
            )?;
        }
        if installed_version != Some(schema.current) {
            // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Ownership gets
            // its own age without extending immutable replay retention.
            tx.execute(
                "UPDATE relay_blind_route_reservations
                 SET owner_acquired_at = reserved_at",
                [],
            )?;
        }

        let invalid_responses = tx.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 2064
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        let invalid_reservations = tx.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at
                OR (effect_started_at IS NOT NULL
                    AND (TYPEOF(effect_started_at) != 'integer'
                         OR effect_started_at < reserved_at))",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_responses != 0 || invalid_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_row_shape",
            });
        }

        install_or_advance_marker(
            &tx,
            schema,
            installed_version,
            now,
            "blind_relay_route_replay_schema_installed_at",
            "blind_relay_route_replay_installation_marker",
            "blind_relay_route_replay_migration_marker",
        )?;
        let cutoff = sqlite_integer(
            now.saturating_sub(self.contract.blind_route_ttl_secs),
            "blind_relay_route_replay_startup_cutoff",
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(())
    }
}

impl ChatRelayReplaySchemaMigration for SqliteChatRelayReplaySchemaMigrator {
    fn migrate_verified_submit(
        &self,
        connection: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        self.migrate_verified_submit_store(connection, now)
    }

    fn migrate_blind_route(&self, connection: &mut Connection, now: u64) -> ChatRelayResult<()> {
        self.migrate_blind_route_store(connection, now)
    }
}

fn table_exists(connection: &Connection, table: &'static str) -> ChatRelayResult<bool> {
    Ok(connection.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
         )",
        params![table],
        |row| row.get::<_, i64>(0),
    )? != 0)
}

fn column_exists(
    tx: &Transaction<'_>,
    table: &'static str,
    expected_column: &'static str,
) -> ChatRelayResult<bool> {
    let query = match table {
        "relay_verified_submit_reservations" => {
            "PRAGMA table_info(relay_verified_submit_reservations)"
        }
        "relay_blind_route_reservations" => "PRAGMA table_info(relay_blind_route_reservations)",
        _ => {
            return Err(ChatRelayError::CorruptStoredData {
                field: "replay_schema_table_contract",
            });
        }
    };
    let mut statement = tx.prepare(query)?;
    let columns = statement.query_map([], |row| row.get::<_, String>(1))?;
    for column in columns {
        if column? == expected_column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn installed_version(tx: &Transaction<'_>, feature: &'static str) -> ChatRelayResult<Option<i64>> {
    Ok(tx
        .query_row(
            "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
            params![feature],
            |row| row.get::<_, i64>(0),
        )
        .optional()?)
}

fn install_or_advance_marker(
    tx: &Transaction<'_>,
    schema: ReplaySchemaVersion,
    installed: Option<i64>,
    now: u64,
    installed_at_field: &'static str,
    installation_error_field: &'static str,
    migration_error_field: &'static str,
) -> ChatRelayResult<()> {
    if installed.is_none()
        && tx.execute(
            "INSERT INTO relay_schema_features (feature, schema_version, installed_at)
             VALUES (?1, ?2, ?3)",
            params![
                schema.feature,
                schema.current,
                sqlite_integer(now, installed_at_field)?
            ],
        )? != 1
    {
        return Err(ChatRelayError::CorruptStoredData {
            field: installation_error_field,
        });
    }
    if schema.needs_migration(installed)
        && tx.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2 AND schema_version IN (?3, ?4)",
            params![
                schema.current,
                schema.feature,
                schema.legacy,
                schema.intermediate,
            ],
        )? != 1
    {
        return Err(ChatRelayError::CorruptStoredData {
            field: migration_error_field,
        });
    }
    Ok(())
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

fn sqlite_integer_from_usize(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}
