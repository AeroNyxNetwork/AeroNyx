// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_bootstrap.rs
// ============================================
// Version: 1.0.0-RelayBootstrap
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
//   v1.0.0-RelayBootstrap - Initial runtime and schema bootstrap extraction
// ============================================

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

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
    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
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
        conn.busy_timeout(Duration::from_secs(5))?;
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
            ),
            ReplaySchemaVersion::new(
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
            ),
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
            REPLAY_PROCESS_EPOCH_BYTES,
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
