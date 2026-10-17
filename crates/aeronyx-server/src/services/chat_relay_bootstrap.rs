// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_bootstrap.rs
// ============================================
// Version: 1.2.0-BlindRouteResourceBound
//
// Creation Reason:
//   [CHAT-BOOTSTRAP-FACADE-DOMAIN 2026-08-28 by Codex] Move fail-closed
//   runtime construction and schema initialization out of the relay
//   composition root while preserving the exact startup order.
//
// Main Functionality:
//   - Creates private durable storage under one process-owned runtime fence.
//   - Verifies SQLite integrity and minimum acknowledgement durability.
//   - Composes the relay's narrowly scoped domain capabilities.
//   - Migrates schemas, reconciles counters, and restores restart state.
//   - Supplies explicit replay history and capacity to schema migration.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns constants, fields, and stable public paths.
//   - Schema migrators own versioned, atomic persistence upgrades.
//   - Runtime fence and SQLite adapters own process and durability boundaries.
//
// Main Logical Flow:
//   1. Prepare the private database directory and acquire the runtime fence.
//   2. Open, permission, verify, and configure the SQLite custody store.
//   3. Construct every domain capability from immutable config and node secret.
//   4. Migrate storage, reconcile accounting, and restore durable circuit state.
//
// Important Note for Next Developer:
//   - Preserve startup order; no migration may run before the runtime fence.
//   - Integrity or durability uncertainty must fail closed before activation.
//   - Never log the database path, node secret, process epoch, or stored data.
//   - New durable domains must install and reconcile before service exposure.
//
// Last Modified:
//   [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex]
//   v1.2.0-BlindRouteResourceBound - Compose v5 byte and WAL boundaries
//   [BLIND-ROUTE-RESPONSE-SCHEMA-V4 2026-08-31 by Codex]
//   v1.1.0-BlindRouteResponseSchemaV4 - Bound v1-v4 migration composition
//   v1.0.0-RelayBootstrap - Initial runtime and schema bootstrap extraction
// ============================================

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a chat relay SQLite statement waits for a competing lock.
///
/// [CHAT-HTTP-503-INVARIANT 2026-10-10 by Claude] Must stay shorter than the
/// HTTP deadline in `api::chat_handlers::CHAT_HTTP_TIMEOUT`: when the HTTP
/// layer gives up it answers 503 but cannot cancel the blocking SQLite worker,
/// so "503 means the mutation did not happen" only holds if the worker has
/// already stopped waiting by then. A compile-time assertion in `chat_handlers`
/// enforces it.
pub(crate) const CHAT_RELAY_SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest single sleep between lock retries.
const CHAT_RELAY_SQLITE_BUSY_STEP: Duration = Duration::from_millis(25);

thread_local! {
    /// When the current statement first found the database locked.
    static CHAT_RELAY_BUSY_SINCE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Busy handler for the chat relay connection: retry until
/// [`CHAT_RELAY_SQLITE_BUSY_TIMEOUT`] of wall-clock time has passed.
///
/// [CHAT-BUSY-WALL-CLOCK 2026-10-17 by Claude] `busy_timeout` counts the
/// delays SQLite asks for, not the time that passed. On a loaded host the
/// sleeps overrun, so a "5 s" wait outlived the 10 s HTTP deadline: the client
/// got 503 while the worker, still holding the connection mutex, took the lock
/// afterwards and committed (the ACK test failed this way on macOS-14 CI in 4
/// runs: 38054116414, 38057377887, 38057872954, 39039754921). Counting real
/// time keeps the invariant the HTTP layer relies on. The connection is used by
/// one thread at a time under its mutex, so a thread-local start time is exact.
fn chat_relay_busy_wait(retry: i32) -> bool {
    let now = Instant::now();
    let since = CHAT_RELAY_BUSY_SINCE.with(|since| {
        let start = if retry == 0 {
            now
        } else {
            since.get().unwrap_or(now)
        };
        since.set(Some(start));
        start
    });
    match busy_wait_step(
        now.duration_since(since),
        retry,
        CHAT_RELAY_SQLITE_BUSY_TIMEOUT,
    ) {
        Some(pause) => {
            std::thread::sleep(pause);
            true
        }
        None => false,
    }
}

/// Next pause for a statement that has been blocked for `elapsed`, or `None`
/// once the wall-clock `bound` is spent. Pauses start short and never sleep
/// past the bound.
fn busy_wait_step(elapsed: Duration, retry: i32, bound: Duration) -> Option<Duration> {
    let remaining = bound.checked_sub(elapsed).filter(|left| !left.is_zero())?;
    let backoff = Duration::from_millis(1u64 << retry.clamp(0, 4) as u32);
    Some(backoff.min(CHAT_RELAY_SQLITE_BUSY_STEP).min(remaining))
}

use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::Connection;
use tracing::info;

use crate::config::ChatRelayConfig;
use crate::services::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use crate::services::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use crate::services::chat_relay_pending_schema::{
    ChatRelayPendingSchemaMigration, SqliteChatRelayPendingSchemaMigrator,
};
use crate::services::chat_relay_replay_schema::{
    ChatRelayReplaySchemaMigration, ReplaySchemaContract, ReplaySchemaVersion,
    SqliteChatRelayReplaySchemaMigrator,
};
#[cfg(unix)]
use crate::services::chat_relay_runtime_fence::ChatRelayRuntimeFence;
use crate::services::chat_relay_storage_schema::{
    ChatRelayStorageSchemaMigration, SqliteChatRelayStorageSchemaMigrator,
};

#[cfg(unix)]
use super::ChatRelayError;
use super::{
    now_secs, BlindRouteCoordinator, BoundedRelayCleanupExecutor, ChatRelayCustodyDurabilityStatus,
    ChatRelayPeerStatus, ChatRelayResult, ChatRelayService, DirectPeerCircuitDomain,
    DurableQuarantineDomain, EncryptedBlobCustodyDomain, ExpiredNotificationDelivery, MessageDedup,
    PeerRelayTelemetryDomain, PendingMessageCustodyDomain, PendingMessageDeliveryDomain,
    RelayMaintenanceTelemetry, SqliteRelayStorageUsageRepository, VerifiedSubmitCoordinator,
    WalletRouteCache, BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS, BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
    BLIND_RELAY_ROUTE_REPLAY_TTL_SECS, CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL,
    REPLAY_PROCESS_EPOCH_BYTES, VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS,
    VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE, VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
    VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION, VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
    VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
};

impl ChatRelayService {
    /// Creates a relay service over a private, verified SQLite custody store.
    ///
    /// # Errors
    ///
    /// Returns a stable fail-closed error when the runtime fence, durable
    /// storage, schema, accounting, or restart-state boundary cannot activate.
    pub fn new(config: ChatRelayConfig, node_secret: [u8; 32]) -> ChatRelayResult<Self> {
        if let Some(parent) = Path::new(&config.db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] A raw IO
                // error can include an operator path. Preserve one stable,
                // path-free failure at this storage trust boundary.
                std::fs::create_dir_all(parent).map_err(|_| {
                    rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                        Some("unable to create relay database directory".to_string()),
                    )
                })?;
            }
        }

        // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Acquire before
        // opening or migrating SQLite. A replacement process can recover an
        // aged reservation only after the kernel has released this guard,
        // proving that its predecessor no longer owns the custody store.
        #[cfg(unix)]
        let runtime_fence = if config.db_path == ":memory:" {
            None
        } else {
            Some(
                ChatRelayRuntimeFence::acquire(Path::new(&config.db_path)).map_err(|error| {
                    ChatRelayError::RuntimeFenceUnavailable {
                        reason: error.as_str(),
                        public_reason_bucket: error.public_reason_bucket(),
                    }
                })?,
            )
        };
        let conn = Connection::open(&config.db_path)?;
        if config.db_path != ":memory:" {
            restrict_private_sqlite_permissions(Path::new(&config.db_path))?;
        }
        // A short bounded wait absorbs transient locks from an operator backup
        // or diagnostic reader without allowing relay requests to hang forever.
        conn.busy_handler(Some(chat_relay_busy_wait))?;
        verify_sqlite_physical_integrity(&conn, "sqlite_startup_integrity")?;
        let synchronous_level =
            configure_full_durability(&conn, CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL)?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let pending_delivery = PendingMessageDeliveryDomain::new(&node_secret)?;
        let pending_custody = PendingMessageCustodyDomain::new(&config);
        let expired_notification_delivery = ExpiredNotificationDelivery::new();
        let blob_custody = EncryptedBlobCustodyDomain::new(node_secret, &config);
        let durable_quarantine = DurableQuarantineDomain::new(&config);
        let cleanup_execution = BoundedRelayCleanupExecutor::new(
            &config,
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
        );
        let verified_submit = VerifiedSubmitCoordinator::new(
            node_secret,
            dedup_capacity,
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS,
        )?;
        let blind_route = BlindRouteCoordinator::new(
            node_secret,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
            BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS,
            config.blind_route_replay_max_bytes_total,
            config.blind_route_wal_backlog_max_bytes,
            config.db_path != ":memory:",
        )?;
        let mut replay_process_epoch = [0_u8; REPLAY_PROCESS_EPOCH_BYTES];
        OsRng.fill_bytes(&mut replay_process_epoch);
        let mut peer_status = ChatRelayPeerStatus::new(relay_enabled);
        peer_status.custody_durability =
            ChatRelayCustodyDurabilityStatus::verified_full(synchronous_level);
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            #[cfg(unix)]
            _runtime_fence: runtime_fence,
            node_secret,
            pending_delivery,
            pending_custody,
            expired_notification_delivery,
            blob_custody,
            durable_quarantine,
            cleanup_execution,
            verified_submit,
            blind_route,
            replay_process_epoch,
            dedup: MessageDedup::new(dedup_capacity),
            storage_usage_repository: SqliteRelayStorageUsageRepository,
            peer_telemetry: PeerRelayTelemetryDomain::new(peer_status),
            direct_peer_relay_circuit: DirectPeerCircuitDomain::default(),
            maintenance_telemetry: RelayMaintenanceTelemetry::default(),
            backup_operations: Mutex::new(()),
            // v1.3.0-Sovereign: initialise empty route cache
            wallet_routes: Arc::new(WalletRouteCache::new()),
        };

        svc.init_schema()?;
        svc.direct_peer_relay_circuit
            .restore(&svc.conn, now_secs())?;
        // [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] The filesystem
        // path is operator-local state and may contain deployment identities.
        // Keep successful activation observable without publishing that path.
        info!("[CHAT_RELAY] Durable service initialized");
        Ok(svc)
    }

    fn init_schema(&self) -> ChatRelayResult<()> {
        let mut conn = self.conn.lock();
        let pending_schema = SqliteChatRelayPendingSchemaMigrator::new();
        pending_schema.migrate(&mut conn)?;
        let storage_schema = SqliteChatRelayStorageSchemaMigrator::new();
        storage_schema.install_custody_tables(&conn)?;
        self.durable_quarantine.init_schema(&conn)?;
        storage_schema.install_usage_accounting(&conn)?;
        self.direct_peer_relay_circuit
            .init_schema(&mut conn, now_secs())?;
        let replay_schema = SqliteChatRelayReplaySchemaMigrator::new(ReplaySchemaContract::new(
            ReplaySchemaVersion::new(
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            ),
            ReplaySchemaVersion::new(
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
            ),
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
            REPLAY_PROCESS_EPOCH_BYTES,
            BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
            self.config.blind_route_replay_max_bytes_total,
        ));
        replay_schema.migrate_verified_submit(&mut conn, now_secs())?;
        replay_schema.migrate_blind_route(&mut conn, now_secs())?;
        storage_schema.reconcile_usage(&conn)?;
        let retained_quarantine_events = self.durable_quarantine.retained_count(&conn)?;
        drop(conn);
        self.maintenance_telemetry
            .set_retained_quarantine_events(retained_quarantine_events);
        Ok(())
    }
}

#[cfg(test)]
mod busy_wait_tests {
    use super::*;

    // [CHAT-BUSY-WALL-CLOCK 2026-10-17 by Claude]
    #[test]
    fn busy_wait_backs_off_and_never_sleeps_past_the_wall_clock_bound() {
        let bound = Duration::from_millis(100);
        assert_eq!(
            busy_wait_step(Duration::ZERO, 0, bound),
            Some(Duration::from_millis(1))
        );
        assert_eq!(
            busy_wait_step(Duration::ZERO, 3, bound),
            Some(Duration::from_millis(8))
        );
        assert_eq!(
            busy_wait_step(Duration::ZERO, 50, bound),
            Some(Duration::from_millis(16))
        );
        assert_eq!(
            busy_wait_step(Duration::from_millis(95), 50, bound),
            Some(Duration::from_millis(5))
        );
        assert_eq!(busy_wait_step(bound, 50, bound), None);
        assert_eq!(
            busy_wait_step(bound + Duration::from_secs(9), 1, bound),
            None
        );
        assert_eq!(
            busy_wait_step(Duration::ZERO, -3, bound),
            Some(Duration::from_millis(1))
        );
    }

    #[test]
    fn locked_chat_relay_write_gives_up_on_wall_clock_time() {
        // A real competing writer: the relay connection must report BUSY close
        // to the bound in real time, never at some later multiple of it.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("busy.sqlite3");
        let relay = Connection::open(&path).expect("relay connection");
        relay
            .busy_handler(Some(chat_relay_busy_wait))
            .expect("handler");
        relay
            .execute_batch("CREATE TABLE t (v INTEGER);")
            .expect("schema");
        let blocker = Connection::open(&path).expect("blocker");
        blocker.execute_batch("BEGIN EXCLUSIVE;").expect("lock");
        let started = Instant::now();
        let error = relay
            .execute("INSERT INTO t VALUES (1)", [])
            .expect_err("must stay locked");
        let waited = started.elapsed();
        assert!(matches!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::DatabaseBusy)
        ));
        assert!(
            waited >= CHAT_RELAY_SQLITE_BUSY_TIMEOUT,
            "gave up early: {waited:?}"
        );
        assert!(
            waited < CHAT_RELAY_SQLITE_BUSY_TIMEOUT + Duration::from_secs(2),
            "outlived the bound: {waited:?}"
        );
        blocker.execute_batch("ROLLBACK;").expect("unlock");
        relay
            .execute("INSERT INTO t VALUES (1)", [])
            .expect("writes once unlocked");
    }
}
