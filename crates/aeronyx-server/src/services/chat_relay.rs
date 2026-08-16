// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay.rs
// ============================================================================
// Version: 3.2.0-CustodyBackupPrune
//
// Modification Reason:
//   v1.3.0-Sovereign — Added WalletRouteCache field to ChatRelayService.
//   The route cache decouples wallet identity from session key, enabling
//   per-message signature-based authentication for all chat operations.
//   Also added dedup_cache as an Arc<Mutex<LruCache>> for the online-path
//   deduplication that will be used by the new handler in server.rs.
//   v1.3.1-Maintenance — Removed stale imports after the chat relay schema and
//   wallet-route integration stabilized. No database schema or API behavior changed.
//   v1.4.0-PeerRelayHealth — Added privacy-safe node-to-node relay health
//   counters for heartbeat/nodeboard diagnostics.
//   v1.5.0-GlobalStorageQuotas — Added transactionally maintained node-wide
//   message/blob usage and hard count/byte ceilings.
//   v1.6.0-OfflineControlReliability — Made ACK and notification batches
//   bounded and atomic, surfaced corrupt durable rows instead of skipping or
//   panicking, and split expiry notifications into transport-safe chunks.
//   v1.7.0-MaintenanceRuntime — Added aggregate cleanup execution evidence
//   and aligned durable pull ordering with the existing message-id cursor.
//   v1.8.0-BoundedMaintenance — Split retention cleanup into bounded SQLite
//   transactions and exposed deferred-backlog evidence.
//   v1.9.0-DurableQuarantine — Added privacy-minimised corrupt-row tombstones,
//   poison-row isolation, complete durable-envelope consistency checks, and
//   atomic concurrent online deduplication.
//   v2.0.0-SnapshotPull — Added a durable monotonic queue sequence and an
//   authenticated opaque cursor for stable ChatPullV2 snapshot pagination.
//   v2.0.1-StartupIntegrity — Removed the configured database path from
//   successful startup logs; server.rs now owns fail-closed activation.
//   v2.0.2-DirectRetryTelemetry — Added aggregate-only direct peer retry
//   recovery, exhaustion, and deterministic-failure telemetry.
//   v2.1.0-DirectRetrySlo — Added a fixed-memory five-minute delivery SLO
//   window with deterministic degradation and outage thresholds.
//   v2.2.0-DirectRelayCircuit — Added a source-blind, generation-safe circuit
//   breaker for target-bound direct relay delivery.
//   v2.3.0-DurableDirectRelayCircuit — Persisted the anonymous circuit safety
//   state so process restarts cannot silently bypass an active outage gate.
//   v2.4.0-DurableSchemaSentinel — Added an atomic installation marker so a
//   deleted circuit checkpoint table cannot be mistaken for a first upgrade.
//   v2.5.0-PowerLossDurability — Requires SQLite FULL durability before the
//   node may acknowledge encrypted custody or persist relay safety state.
//   v2.6.0-CustodyDurabilityStatus — Publishes only the verified aggregate
//   durability mode so operators can audit custody readiness.
//   v2.7.0-PrivateCustodyFile — Restricts the SQLite custody database and its
//   WAL sidecars to the node service account on Unix hosts.
//   v2.8.0-StartupPhysicalIntegrity — Rejects relay activation when SQLite's
//   bounded startup quick-check cannot prove the custody file is intact.
//   v2.9.0-VerifiedCustodyBackup — Added a private, transactionally consistent
//   SQLite backup artifact with pre-publication physical and logical checks.
//   v3.0.0-IdempotentCustodyBackup — Bound audited backup commands to a stable,
//   node-secret HMAC artifact key so restart replay reuses one verified image.
//   v3.1.0-CustodyBackupRetention — Added serialized, oldest-first count/byte
//   retention inspection and an aggregate-only audit operation.
//   v3.2.0-CustodyBackupPrune — Added explicit host-local dry-run/prune with
//   cross-process exclusion, grace-gated partial cleanup, and HMAC audit.
//
// Main Functionality:
//   - ChatRelayService: Central service managing all chat relay state
//   - Message store/pull/ack: SQLite-backed pending message queue
//   - Blob store/get/delete: SQLite-backed encrypted media cache
//   - TTL cleanup: Expires pending messages and blobs, queues notifications
//   - Expired notifications: Queued ChatExpired delivery for offline senders
//   - Online deduplication: LRU cache prevents duplicate delivery (online path)
//   - WalletRouteCache: In-memory wallet → session routing (v1.3.0-Sovereign)
//   - Peer relay health: aggregate outbound/inbound node-to-node relay status
//   - Durable queue quotas: per-receiver and node-wide count/byte ceilings
//   - Maintenance telemetry: aggregate TTL cleanup, batch, and backlog evidence
//   - Durable quarantine: bounded de-identified evidence for corrupt relay rows
//   - Snapshot pull: restart-stable pagination that excludes concurrent inserts
//   - Direct retry telemetry: aggregate ACK-loss recovery evidence without IDs
//   - Direct retry SLO: five fixed minute buckets, no event log or timer task
//   - Direct relay circuit: bounded open/half-open recovery without downgrade
//   - Durable circuit checkpoint: fixed-size anonymous restart protection
//   - Durable schema sentinel: rejects post-install checkpoint table loss
//   - Power-loss durability: WAL + FULL is verified before relay activation
//   - Custody durability status: anonymous verified mode in relay health
//   - Private custody file: Unix database/WAL material is owner-only
//   - Startup physical integrity: corruption fails closed before migrations
//   - Verified custody backup: WAL-aware, private, no-overwrite recovery image
//   - Idempotent custody backup: restart-safe command replay without raw IDs
//   - Custody backup retention: bounded, verified, serialized local audit
//
// Dependencies:
//   - aeronyx-core/src/protocol/chat.rs: ChatEnvelope, encode_envelope, decode_envelope
//   - aeronyx-server/src/config_chat_relay.rs: ChatRelayConfig
//   - crates/aeronyx-server/src/services/chat_relay/wallet_routes.rs: WalletRouteCache
//
// Main Logical Flow:
//   ChatRelayService::new():
//     1. Open/create SQLite database
//     2. Restrict the durable file to the node account on Unix
//     3. Verify bounded SQLite physical integrity without exposing findings
//     4. Set and verify WAL + FULL pragmas
//     5. init_schema() creates tables if missing
//     6. Initialise MessageDedup (online-path LRU)
//     7. Initialise WalletRouteCache (in-memory, empty on startup)
//
// ⚠️ Important Notes for Next Developer:
//   - wallet_routes is Arc<WalletRouteCache> so server.rs can hold a separate
//     Arc clone for the cleanup background task without borrowing ChatRelayService.
//   - All SQLite operations use parking_lot::Mutex<Connection>. Do NOT call
//     SQLite methods while holding another lock.
//   - ack_messages deletes only WHERE receiver = receiver_wallet.
//   - run_cleanup is synchronous — call from spawn_blocking or a sync task.
//   - node_secret is HKDF-derived from Ed25519 private key; stable across restarts.
//   - `relay_storage_usage` is rebuilt from canonical rows at startup, then
//     maintained only by SQLite triggers in the same write transaction.
//   - Logs must remain aggregate-only. Do not log message IDs, wallet prefixes,
//     blob IDs, sender/receiver keys, payload bytes, or endpoint/session IDs.
//   - Offline control batches are protocol-bounded. Do not remove their limits.
//     Malformed rows must be atomically replaced by de-identified quarantine
//     evidence; never silently skip them or copy raw routing metadata.
//   - Pending-message pages are ordered by message_id because the v1 wire
//     cursor contains only message_id. Chronological display belongs client-side.
//   - ChatPullV2 queue_sequence is an internal ordering primitive. Never expose
//     it on the wire or in logs; only issue the AEAD-protected opaque cursor.
//   - Retention cleanup is batch-bounded. Do not replace it with an unbounded
//     SELECT/DELETE or hold the SQLite connection across multiple batches.
//   - [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Do not log the relay
//     database path or raw initialization errors. server.rs exposes only the
//     stable `reason_bucket()` when explicit activation fails.
//   - [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Retry telemetry must
//     remain aggregate-only. Never add peer IDs, message IDs, request
//     commitments, endpoints, wallet keys, or payload-derived dimensions.
//   - [DIRECT-RELAY-SLO 2026-08-15 by Codex] The recent window is process-local
//     and fixed-size. Do not replace it with per-delivery events, labels, or a
//     background timer; snapshots prune stale minute buckets on read.
//   - [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Circuit state is source-blind;
//     permit generations remain process-local while the minimal safety state
//     survives restart. Never add peer, route, message, endpoint, wallet, or
//     payload identifiers to permits, state, telemetry, or health snapshots.
//   - [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Only the anonymous
//     circuit state and aggregate counters may cross a restart. A corrupt
//     checkpoint must reject relay activation; an unavailable runtime write
//     must open the in-memory circuit, deny new admission, and stop fanout.
//     Runtime transitions lock circuit -> SQLite; never acquire the circuit
//     while holding `conn`.
//   - [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The fixed feature
//     marker and checkpoint singleton are one atomic migration. Once installed,
//     a missing checkpoint table is corruption and must never reset to closed.
//   - [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] Signed custody ACKs and
//     direct-relay safety checkpoints share one SQLite connection. It must
//     remain FULL-or-stronger; NORMAL can lose acknowledged writes on power loss.
//   - [CHAT-RELAY-DURABILITY-STATUS 2026-08-16 by Codex] Durability telemetry
//     contains only a fixed state, protection boolean, and SQLite mode number.
//     Never add database paths, row counts, message IDs, or owner dimensions.
//   - [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] Restrict the primary DB
//     before enabling WAL so SQLite derives owner-only permissions for `-wal`
//     and `-shm`. Permission failures reject activation without logging paths.
//   - [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] Run the physical
//     integrity gate before WAL changes and schema migrations. Never log raw
//     SQLite findings; they may disclose schema and storage-layout details.
//   - [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Never copy the live DB
//     file directly: committed custody can still reside in its WAL. Backups
//     must remain owner-only, validate physical integrity, usage counters and
//     anonymous circuit safety state, then publish without replacing a prior
//     artifact. Backup paths and raw SQLite findings are operator-private.
//   - [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] CMS command IDs are
//     untrusted routing metadata. Derive operation artifact names with a
//     domain-separated node-secret HMAC; never persist or expose the raw ID.
//     Existing operation artifacts must be re-verified and reused, never
//     overwritten, including after process restart or concurrent replay.
//   - [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Backup creation and
//     retention inspection share `backup_operations`; never inspect outside
//     that lock. Verify every managed artifact, model the protected/newest
//     image as retained, and report aggregate counts/bytes only.
//   - [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Deletion is never timed or
//     remote. Require the host-local confirmation contract, cross-process lock,
//     pre-delete identity/integrity recheck, durable directory sync, and the
//     private HMAC-chained aggregate audit around every explicit prune.
//   - [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Recovery readiness is
//     non-destructive: verify every managed image, select the newest valid
//     image, and inspect only aggregate active-file/sidecar state. It must not
//     copy, rename, remove, open active storage, or imply restore execution.
//   - [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] A restore plan is a
//     short-lived, node-secret HMAC commitment to one verified image and the
//     active custody boundary observed at issuance. It is not authorization to
//     replace data and must remain host-local.
//   - Quarantine events must remain de-identified. Never persist message IDs,
//     sender/receiver keys, ciphertext, endpoints, or raw durable rows there.
//
// Last Modified:
//   v3.4.0-CustodyRestorePlan — Authenticated, state-bound recovery planning
//   v3.3.0-CustodyRestoreReadiness — Read-only latest-image recovery preflight
//   v3.2.0-CustodyBackupPrune — Confirmation-gated local recovery maintenance
//   v3.1.0-CustodyBackupRetention — Bounded verified recovery-image retention
//   v3.0.0-IdempotentCustodyBackup — Restart-safe audited backup replay
//   v2.9.0-VerifiedCustodyBackup — Atomic validated SQLite recovery artifact
//   v2.8.0-StartupPhysicalIntegrity — Pre-migration SQLite quick-check gate
//   v2.7.0-PrivateCustodyFile — Owner-only SQLite custody files on Unix
//   v2.6.0-CustodyDurabilityStatus — Aggregate verified durability evidence
//   v2.5.0-PowerLossDurability — Verified FULL durability for custody writes
//   v2.4.0-DurableSchemaSentinel — Fail-closed checkpoint installation marker
//   v2.3.0-DurableDirectRelayCircuit — Anonymous restart-safe circuit checkpoint
//   v2.2.0-DirectRelayCircuit — Generation-safe fail-closed delivery circuit
//   v2.1.0-DirectRetrySlo — Fixed-memory five-minute delivery health window
//   v2.0.2-DirectRetryTelemetry — Privacy-safe ACK retry outcome counters
//   v2.0.1-StartupIntegrity — Privacy-safe, fail-closed service activation
//   v2.0.0-SnapshotPull — Monotonic queue sequence, atomic legacy backfill,
//     and wallet-bound XChaCha20-Poly1305 snapshot cursors
//   v1.9.0-DurableQuarantine — Poison-row isolation, private tombstones,
//     durable metadata/signature validation, and atomic live-path deduplication
//   v1.8.0-BoundedMaintenance — Bounded transactions and backlog observability
//   v1.7.0-MaintenanceRuntime — Runtime cleanup evidence and cursor-safe paging
//   v1.6.0-OfflineControlReliability — Atomic bounded ACK/expiry control flow
//   v1.5.0-GlobalStorageQuotas — Durable global quotas, enforced message size,
//     and route-safe logging
//   v1.4.0-PeerRelayHealth — Added node-to-node relay health status snapshot
//   v1.1.0-ChatRelay — Initial implementation
//   v1.3.0-Sovereign — Added wallet_routes: Arc<WalletRouteCache> field
//   v1.3.1-Maintenance — Removed stale imports; behavior unchanged
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use dashmap::{mapref::entry::Entry, DashMap};
use hmac::{Hmac, Mac};
use parking_lot::{Mutex, RwLock};
use rand::{rngs::OsRng, RngCore};
use rusqlite::{
    backup::{Backup, StepResult},
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use tracing::{debug, info, warn};

use aeronyx_core::protocol::chat::{decode_envelope, encode_envelope, ChatEnvelope};

use crate::config::ChatRelayConfig;
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Type aliases
// ============================================

type HmacSha256 = Hmac<Sha256>;

/// Maximum IDs accepted in one authenticated `ChatAck` frame.
pub const MAX_CHAT_ACK_MESSAGE_IDS: usize = 100;
/// Maximum IDs encoded into one `ChatExpired` frame.
const MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION: usize = 32;
/// Maximum notification rows offered during one authenticated pull.
const MAX_EXPIRED_NOTIFICATIONS_PER_PULL: usize = 16;
/// Defensive ceiling for one persisted bincode notification payload.
const MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES: usize = 1024;
/// Maximum expired message rows processed by one `SQLite` transaction.
const CLEANUP_MESSAGE_BATCH_SIZE: usize = 1024;
/// Maximum expired encrypted blobs deleted by one `SQLite` transaction.
const CLEANUP_BLOB_BATCH_SIZE: usize = 128;
/// Maximum delivered/stale notification rows deleted by one transaction.
const CLEANUP_NOTIFICATION_BATCH_SIZE: usize = 1024;
/// Maximum privacy-minimised quarantine events removed by one transaction.
const CLEANUP_QUARANTINE_EVENT_BATCH_SIZE: usize = 1024;
/// Maximum cleanup transactions executed by one scheduled maintenance run.
const CLEANUP_MAX_BATCHES_PER_RUN: usize = 8;
/// Maximum retained de-identified corruption events.
const MAX_QUARANTINE_EVENTS: usize = 4096;
const QUARANTINE_SOURCE_PENDING_MESSAGE: &str = "pending_message";
const QUARANTINE_SOURCE_EXPIRED_NOTIFICATION: &str = "expired_notification";
/// Current binary format for an opaque ChatPullV2 cursor.
const CHAT_PULL_CURSOR_V2_VERSION: u8 = 1;
const CHAT_PULL_CURSOR_V2_NONCE_BYTES: usize = 24;
const CHAT_PULL_CURSOR_V2_PAYLOAD_BYTES: usize = 16;
const CHAT_PULL_CURSOR_V2_TAG_BYTES: usize = 16;
const CHAT_PULL_CURSOR_V2_BYTES: usize = 1
    + CHAT_PULL_CURSOR_V2_NONCE_BYTES
    + CHAT_PULL_CURSOR_V2_PAYLOAD_BYTES
    + CHAT_PULL_CURSOR_V2_TAG_BYTES;
const CHAT_PULL_CURSOR_V2_AAD_DOMAIN: &[u8] = b"AeroNyx-ChatPullCursor-v2";
const CHAT_PULL_CURSOR_V2_HKDF_SALT: &[u8] = b"AeroNyx-ChatPullCursor-v2-key";
const CHAT_PULL_CURSOR_V2_HKDF_INFO: &[u8] = b"XChaCha20-Poly1305";
/// Recent target-bound delivery health uses five fixed one-minute buckets.
const DIRECT_PEER_RETRY_SLO_BUCKET_SECS: u64 = 60;
const DIRECT_PEER_RETRY_SLO_BUCKET_COUNT: usize = 5;
const DIRECT_PEER_RETRY_SLO_WINDOW_SECS: u64 =
    DIRECT_PEER_RETRY_SLO_BUCKET_SECS * DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64;
/// 99.00% target-bound delivery success target, represented in basis points.
const DIRECT_PEER_RETRY_SLO_TARGET_BPS: u16 = 9_900;
/// Require repeated failures before declaring a short-window outage.
const DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES: u64 = 3;
/// At or below 50% delivery success with enough failures is a failed window.
const DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS: u16 = 5_000;
/// Cooldown before one source-blind target-bound relay recovery probe.
const DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS: u64 = 30;
/// Maximum time reserved for one half-open delivery before fail-closed reopen.
const DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS: u64 = 15;
/// Consecutive half-open delivery successes required to close the circuit.
const DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES: u8 = 2;
/// Durable singleton format for source-blind direct relay circuit state.
const DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION: i64 = 1;
/// Fixed schema marker proving the durable circuit checkpoint was installed.
const DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE: &str = "direct_peer_relay_circuit_checkpoint";
/// Minimum SQLite synchronous level permitted for acknowledged relay custody.
const CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
/// Pages copied per online-backup step before SQLite releases its read lock.
const CHAT_RELAY_BACKUP_PAGES_PER_STEP: i32 = 256;
/// Maximum consecutive SQLite busy time before a backup fails closed.
const CHAT_RELAY_BACKUP_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Delay between bounded retries while another process holds a SQLite lock.
const CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY: Duration = Duration::from_millis(10);
/// Maximum UTF-8 bytes accepted from one management-plane operation ID.
const CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES: usize = 128;
/// Domain separation for opaque, node-local backup operation artifact keys.
const CHAT_RELAY_BACKUP_OPERATION_HMAC_DOMAIN: &[u8] = b"AeroNyx-RelayCustodyBackup-Operation-v1";
/// Fixed validity window for a host-local authenticated restore plan.
pub const CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS: u64 = 10 * 60;
/// Domain separation for short-lived restore-plan commitments.
const CHAT_RELAY_RESTORE_PLAN_HMAC_DOMAIN: &[u8] = b"AeroNyx-RelayCustodyRestorePlan-v1";
/// Current restore-plan wire contract.
const CHAT_RELAY_RESTORE_PLAN_VERSION: u8 = 1;
/// Random nonce bytes encoded into each restore plan.
const CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES: usize = 16;
/// Defensive ceiling for one verified backup-directory maintenance scan.
const CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT: usize = 1024;
/// Exact phrase required before a host-local command may delete backup files.
pub const CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION: &str = "PRUNE-VERIFIED-RELAY-BACKUPS";
/// Domain separation for the private append-only maintenance audit HMAC.
const CHAT_RELAY_BACKUP_AUDIT_HMAC_DOMAIN: &[u8] =
    b"AeroNyx-RelayCustodyBackup-MaintenanceAudit-v1";
/// Private sibling file holding aggregate-only maintenance audit records.
const CHAT_RELAY_BACKUP_AUDIT_FILE_NAME: &str = ".aeronyx-relay-backup-maintenance-audit.jsonl";
/// Private sibling file serializing maintenance across server processes.
const CHAT_RELAY_BACKUP_LOCK_FILE_NAME: &str = ".aeronyx-relay-backup-maintenance.lock";
/// Hard ceiling for the append-only audit file before operator intervention.
const CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Hard ceiling for one audit record, including its trailing newline.
const CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES: usize = 4096;
/// Hard ceiling for audit records verified before one append.
const CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS: usize = 65_536;
/// Small tolerated wall-clock adjustment before restart recovery fails closed.
const DIRECT_PEER_RELAY_CIRCUIT_CLOCK_SKEW_SECS: u64 = 5;

// ============================================
// Peer relay health status
// ============================================

/// Aggregate-only health for one outbound encrypted relay route class.
///
/// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Authenticated onion relay
/// and compatibility direct relay can run sequentially for the same opaque
/// envelope. Keeping independent snapshots prevents the fallback result from
/// overwriting the route class an operator or live proof is actually testing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayOutboundRouteStatus {
    /// Total requests attempted through this route class.
    pub attempted_total: u64,
    /// Total requests accepted through this route class.
    pub accepted_total: u64,
    /// Total requests that failed or were rejected through this route class.
    pub failed_total: u64,
    /// Total route-class rounds observed, including failed preflight rounds.
    pub rounds: u64,
    /// Requests attempted in the latest route-class round.
    pub last_attempted: u64,
    /// Requests accepted in the latest route-class round.
    pub last_accepted: u64,
    /// Requests failed in the latest route-class round.
    pub last_failed: u64,
    /// Latest health bucket: healthy, degraded, failed, idle.
    pub last_status: Option<String>,
    /// Privacy-safe reason bucket for the latest failure.
    pub last_failure_reason: Option<String>,
    /// Consecutive route-class rounds with no accepted request.
    pub consecutive_failures: u64,
    /// Timestamp of the latest route-class round with an accepted request.
    pub last_success_at: Option<u64>,
    /// Timestamp of the latest route-class round.
    pub last_at: Option<u64>,
}

/// Fixed-window target-bound direct relay delivery SLO.
///
/// All ratios use basis points (`10_000 == 100%`) so heartbeat consumers get a
/// stable integer contract. The window contains aggregate counts only and is
/// reset when the process restarts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerSloStatus {
    /// Width of the rolling aggregate window.
    pub window_seconds: u64,
    /// Delivery success target in basis points.
    pub slo_target_bps: u16,
    /// Minimum failures required before the window can be classified failed.
    pub failed_min_failures: u64,
    /// Delivery success ceiling for a failed classification, in basis points.
    pub failed_success_ceiling_bps: u16,
    /// Target-bound v3 deliveries observed in the current window.
    pub deliveries_total: u64,
    /// Deliveries that ended with a valid acknowledgement.
    pub delivered_total: u64,
    /// Deliveries that ended without a valid acknowledgement.
    pub failed_total: u64,
    /// Deliveries that entered the exact retry path.
    pub retry_triggered_total: u64,
    /// Exact retries that recovered a valid acknowledgement.
    pub retry_recovered_total: u64,
    /// Exact retries that exhausted their bounded retry budget.
    pub retry_exhausted_total: u64,
    /// Deliveries whose final typed failure was deterministic.
    pub deterministic_failure_total: u64,
    /// Delivery success ratio in basis points, or none when idle.
    pub delivery_success_bps: Option<u16>,
    /// Retry recovery ratio in basis points, or none when no retry occurred.
    pub retry_recovery_bps: Option<u16>,
    /// Whether the current non-empty window meets the configured SLO.
    pub meets_slo: Option<bool>,
    /// Current aggregate bucket: idle, healthy, degraded, or failed.
    pub status: String,
    /// Unix timestamp used to evaluate the rolling window.
    pub evaluated_at: u64,
}

impl Default for ChatRelayDirectPeerSloStatus {
    fn default() -> Self {
        Self {
            window_seconds: DIRECT_PEER_RETRY_SLO_WINDOW_SECS,
            slo_target_bps: DIRECT_PEER_RETRY_SLO_TARGET_BPS,
            failed_min_failures: DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES,
            failed_success_ceiling_bps: DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS,
            deliveries_total: 0,
            delivered_total: 0,
            failed_total: 0,
            retry_triggered_total: 0,
            retry_recovered_total: 0,
            retry_exhausted_total: 0,
            deterministic_failure_total: 0,
            delivery_success_bps: None,
            retry_recovery_bps: None,
            meets_slo: None,
            status: "idle".to_string(),
            evaluated_at: 0,
        }
    }
}

/// Aggregate-only target-bound direct relay retry outcomes.
///
/// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] These counters describe
/// transport reliability without recording which peer, envelope, request
/// commitment, endpoint, or wallet caused an event. Every recorded retry
/// trigger is classified as recovered or exhausted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerRetryStatus {
    /// Deliveries that entered the one-shot exact retry path.
    pub retry_triggered_total: u64,
    /// Retried deliveries whose exact retry produced a valid ACK.
    pub retry_recovered_total: u64,
    /// Retried deliveries that still failed after the retry budget was spent.
    pub retry_exhausted_total: u64,
    /// Deliveries whose final failure was deterministic and not retryable.
    ///
    /// This may overlap `retry_exhausted_total` when an ambiguity retry receives
    /// a deterministic rejection on its second attempt.
    pub deterministic_failure_total: u64,
    /// Latest observed retry outcome: recovered, exhausted, or deterministic_failure.
    pub last_outcome: Option<String>,
    /// Unix timestamp of the latest retry or deterministic-failure observation.
    pub last_at: Option<u64>,
    /// Current fixed-memory target-bound delivery SLO window.
    #[serde(default)]
    pub recent_window: ChatRelayDirectPeerSloStatus,
    /// Source-blind target-bound delivery admission circuit.
    #[serde(default)]
    pub circuit: ChatRelayDirectPeerCircuitStatus,
}

/// Aggregate target-bound direct relay circuit status.
///
/// This contract intentionally exposes no circuit slot, peer identity, route,
/// endpoint, request commitment, message identifier, or payload dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerCircuitStatus {
    /// Current state: closed, open, or half_open.
    pub state: String,
    /// Fixed open-state cooldown in seconds.
    pub cooldown_seconds: u64,
    /// Maximum lease for one in-flight half-open probe.
    pub half_open_lease_seconds: u64,
    /// Consecutive successful probes required to close.
    pub half_open_successes_required: u8,
    /// Successful probes accumulated in the current recovery sequence.
    pub half_open_consecutive_successes: u8,
    /// Number of transitions into open state.
    pub opened_total: u64,
    /// Delivery rounds denied while open or while a probe is in flight.
    pub blocked_total: u64,
    /// Half-open probes admitted by this process.
    pub half_open_attempted_total: u64,
    /// Half-open probes that ended with valid acknowledgement.
    pub half_open_succeeded_total: u64,
    /// Half-open probes that failed or exceeded their lease.
    pub half_open_failed_total: u64,
    /// Half-open recovery sequences that closed the circuit.
    pub recovered_total: u64,
    /// Remaining open cooldown, or none outside a cooling open state.
    pub open_remaining_seconds: Option<u64>,
    /// Unix timestamp of the latest state transition.
    pub last_transition_at: Option<u64>,
    /// Whether the current safety state is protected across process restart.
    pub restart_protected: bool,
    /// Unix timestamp when this process loaded the durable checkpoint.
    pub checkpoint_loaded_at: Option<u64>,
    /// Unix timestamp of the latest successful checkpoint write.
    pub checkpoint_persisted_at: Option<u64>,
    /// Runtime checkpoint writes that failed closed in this process.
    pub checkpoint_failures_total: u64,
    /// Unix timestamp of the latest runtime checkpoint write failure.
    pub last_checkpoint_failure_at: Option<u64>,
}

impl Default for ChatRelayDirectPeerCircuitStatus {
    fn default() -> Self {
        Self {
            state: "closed".to_string(),
            cooldown_seconds: DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS,
            half_open_lease_seconds: DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS,
            half_open_successes_required: DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES,
            half_open_consecutive_successes: 0,
            opened_total: 0,
            blocked_total: 0,
            half_open_attempted_total: 0,
            half_open_succeeded_total: 0,
            half_open_failed_total: 0,
            recovered_total: 0,
            open_remaining_seconds: None,
            last_transition_at: None,
            restart_protected: false,
            checkpoint_loaded_at: None,
            checkpoint_persisted_at: None,
            checkpoint_failures_total: 0,
            last_checkpoint_failure_at: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DirectPeerRetrySloBucket {
    initialized: bool,
    epoch_minute: u64,
    deliveries: u64,
    delivered: u64,
    retry_triggered: u64,
    retry_recovered: u64,
    retry_exhausted: u64,
    deterministic_failure: u64,
}

#[derive(Debug, Default)]
struct DirectPeerRetrySloWindow {
    buckets: [DirectPeerRetrySloBucket; DIRECT_PEER_RETRY_SLO_BUCKET_COUNT],
    latest_epoch_minute: u64,
}

impl DirectPeerRetrySloWindow {
    fn record(
        &mut self,
        now: u64,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) {
        let observed_epoch = now / DIRECT_PEER_RETRY_SLO_BUCKET_SECS;
        let epoch_minute = observed_epoch.max(self.latest_epoch_minute);
        self.latest_epoch_minute = epoch_minute;
        let index = (epoch_minute % DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64) as usize;
        let bucket = &mut self.buckets[index];
        if !bucket.initialized || bucket.epoch_minute != epoch_minute {
            *bucket = DirectPeerRetrySloBucket {
                initialized: true,
                epoch_minute,
                ..DirectPeerRetrySloBucket::default()
            };
        }

        bucket.deliveries = bucket.deliveries.saturating_add(1);
        if delivery_succeeded {
            bucket.delivered = bucket.delivered.saturating_add(1);
        }
        if retry_triggered {
            bucket.retry_triggered = bucket.retry_triggered.saturating_add(1);
            if delivery_succeeded {
                bucket.retry_recovered = bucket.retry_recovered.saturating_add(1);
            } else {
                bucket.retry_exhausted = bucket.retry_exhausted.saturating_add(1);
            }
        }
        if final_failure_deterministic {
            bucket.deterministic_failure = bucket.deterministic_failure.saturating_add(1);
        }
    }

    fn snapshot(&self, now: u64) -> ChatRelayDirectPeerSloStatus {
        let current_epoch = (now / DIRECT_PEER_RETRY_SLO_BUCKET_SECS).max(self.latest_epoch_minute);
        let mut snapshot = ChatRelayDirectPeerSloStatus {
            evaluated_at: now,
            ..ChatRelayDirectPeerSloStatus::default()
        };
        for bucket in &self.buckets {
            if !bucket.initialized
                || bucket.epoch_minute > current_epoch
                || current_epoch.saturating_sub(bucket.epoch_minute)
                    >= DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64
            {
                continue;
            }
            snapshot.deliveries_total = snapshot.deliveries_total.saturating_add(bucket.deliveries);
            snapshot.delivered_total = snapshot.delivered_total.saturating_add(bucket.delivered);
            snapshot.retry_triggered_total = snapshot
                .retry_triggered_total
                .saturating_add(bucket.retry_triggered);
            snapshot.retry_recovered_total = snapshot
                .retry_recovered_total
                .saturating_add(bucket.retry_recovered);
            snapshot.retry_exhausted_total = snapshot
                .retry_exhausted_total
                .saturating_add(bucket.retry_exhausted);
            snapshot.deterministic_failure_total = snapshot
                .deterministic_failure_total
                .saturating_add(bucket.deterministic_failure);
        }
        snapshot.failed_total = snapshot
            .deliveries_total
            .saturating_sub(snapshot.delivered_total);
        snapshot.delivery_success_bps =
            ratio_basis_points(snapshot.delivered_total, snapshot.deliveries_total);
        snapshot.retry_recovery_bps = ratio_basis_points(
            snapshot.retry_recovered_total,
            snapshot.retry_triggered_total,
        );
        snapshot.meets_slo = snapshot
            .delivery_success_bps
            .map(|ratio| ratio >= DIRECT_PEER_RETRY_SLO_TARGET_BPS);
        snapshot.status = if snapshot.deliveries_total == 0 {
            "idle"
        } else if snapshot.failed_total >= DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES
            && snapshot.delivery_success_bps.unwrap_or(0)
                <= DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS
        {
            "failed"
        } else if snapshot.meets_slo == Some(true) {
            "healthy"
        } else {
            "degraded"
        }
        .to_string();
        snapshot
    }
}

fn ratio_basis_points(numerator: u64, denominator: u64) -> Option<u16> {
    if denominator == 0 {
        return None;
    }
    let basis_points = (u128::from(numerator).saturating_mul(10_000)) / u128::from(denominator);
    Some(basis_points.min(10_000) as u16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPeerRelayCircuitState {
    Closed,
    Open {
        retry_at: u64,
    },
    HalfOpenReady {
        successful_probes: u8,
    },
    HalfOpenInFlight {
        successful_probes: u8,
        lease_expires_at: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPeerRelayPermitKind {
    Closed,
    HalfOpen,
}

/// Process-local admission token for one target-bound direct relay attempt.
///
/// [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] The generation prevents a late
/// outcome from an older request from closing or reopening a newer circuit.
/// The token deliberately contains no peer, route, endpoint, message, wallet,
/// commitment, ciphertext, or payload-derived value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatRelayDirectPeerPermit {
    generation: u64,
    kind: DirectPeerRelayPermitKind,
}

impl ChatRelayDirectPeerPermit {
    /// Returns whether this permit is the circuit's single recovery probe.
    #[must_use]
    pub(crate) const fn is_half_open(self) -> bool {
        matches!(self.kind, DirectPeerRelayPermitKind::HalfOpen)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectPeerRelayCircuit {
    state: DirectPeerRelayCircuitState,
    generation: u64,
    opened_total: u64,
    blocked_total: u64,
    half_open_attempted_total: u64,
    half_open_succeeded_total: u64,
    half_open_failed_total: u64,
    recovered_total: u64,
    last_transition_at: Option<u64>,
    restart_protected: bool,
    checkpoint_loaded_at: Option<u64>,
    checkpoint_persisted_at: Option<u64>,
    checkpoint_failures_total: u64,
    last_checkpoint_failure_at: Option<u64>,
}

impl Default for DirectPeerRelayCircuit {
    fn default() -> Self {
        Self {
            state: DirectPeerRelayCircuitState::Closed,
            generation: 0,
            opened_total: 0,
            blocked_total: 0,
            half_open_attempted_total: 0,
            half_open_succeeded_total: 0,
            half_open_failed_total: 0,
            recovered_total: 0,
            last_transition_at: None,
            restart_protected: false,
            checkpoint_loaded_at: None,
            checkpoint_persisted_at: None,
            checkpoint_failures_total: 0,
            last_checkpoint_failure_at: None,
        }
    }
}

impl DirectPeerRelayCircuit {
    fn checkpoint_state(&self) -> (&'static str, u8, Option<u64>) {
        match self.state {
            DirectPeerRelayCircuitState::Closed => ("closed", 0, None),
            DirectPeerRelayCircuitState::Open { retry_at } => ("open", 0, Some(retry_at)),
            DirectPeerRelayCircuitState::HalfOpenReady { successful_probes } => {
                ("half_open_ready", successful_probes, None)
            }
            DirectPeerRelayCircuitState::HalfOpenInFlight {
                successful_probes,
                lease_expires_at,
            } => (
                "half_open_in_flight",
                successful_probes,
                Some(lease_expires_at),
            ),
        }
    }

    fn safety_state_changed(&self, previous: &Self) -> bool {
        self.state != previous.state
            || self.opened_total != previous.opened_total
            || self.half_open_attempted_total != previous.half_open_attempted_total
            || self.half_open_succeeded_total != previous.half_open_succeeded_total
            || self.half_open_failed_total != previous.half_open_failed_total
            || self.recovered_total != previous.recovered_total
            || self.last_transition_at != previous.last_transition_at
    }

    fn mark_checkpoint_loaded(&mut self, now: u64, persisted_at: Option<u64>) {
        self.restart_protected = true;
        self.checkpoint_loaded_at = Some(now);
        self.checkpoint_persisted_at = persisted_at;
    }

    fn mark_checkpoint_persisted(&mut self, now: u64) {
        self.restart_protected = true;
        self.checkpoint_persisted_at = Some(now);
    }

    fn fail_closed_after_checkpoint_error(&mut self, now: u64) {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A runtime SQLite
        // failure must never leave a newly admitted half-open request usable.
        self.checkpoint_failures_total = self.checkpoint_failures_total.saturating_add(1);
        self.last_checkpoint_failure_at = Some(now);
        self.restart_protected = false;
        self.open(now);
    }

    fn accepts_completion(&self, permit: ChatRelayDirectPeerPermit) -> bool {
        if permit.generation != self.generation {
            return false;
        }
        matches!(
            (permit.kind, self.state),
            (
                DirectPeerRelayPermitKind::Closed,
                DirectPeerRelayCircuitState::Closed
            ) | (
                DirectPeerRelayPermitKind::HalfOpen,
                DirectPeerRelayCircuitState::HalfOpenInFlight { .. }
            )
        )
    }

    fn begin(&mut self, now: u64) -> Option<ChatRelayDirectPeerPermit> {
        match self.state {
            DirectPeerRelayCircuitState::Closed => Some(ChatRelayDirectPeerPermit {
                generation: self.generation,
                kind: DirectPeerRelayPermitKind::Closed,
            }),
            DirectPeerRelayCircuitState::Open { retry_at } if now < retry_at => {
                self.blocked_total = self.blocked_total.saturating_add(1);
                None
            }
            DirectPeerRelayCircuitState::Open { .. } => Some(self.begin_half_open(now, 0)),
            DirectPeerRelayCircuitState::HalfOpenReady { successful_probes } => {
                Some(self.begin_half_open(now, successful_probes))
            }
            DirectPeerRelayCircuitState::HalfOpenInFlight {
                lease_expires_at, ..
            } if now < lease_expires_at => {
                self.blocked_total = self.blocked_total.saturating_add(1);
                None
            }
            DirectPeerRelayCircuitState::HalfOpenInFlight { .. } => {
                // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A dropped future
                // cannot permanently strand half-open admission. Expiry is a
                // failed probe and starts a fresh cooldown without a timer.
                self.half_open_failed_total = self.half_open_failed_total.saturating_add(1);
                self.open(now);
                self.blocked_total = self.blocked_total.saturating_add(1);
                None
            }
        }
    }

    fn begin_half_open(&mut self, now: u64, successful_probes: u8) -> ChatRelayDirectPeerPermit {
        self.generation = self.generation.wrapping_add(1);
        self.state = DirectPeerRelayCircuitState::HalfOpenInFlight {
            successful_probes,
            lease_expires_at: now.saturating_add(DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS),
        };
        self.half_open_attempted_total = self.half_open_attempted_total.saturating_add(1);
        self.last_transition_at = Some(now);
        ChatRelayDirectPeerPermit {
            generation: self.generation,
            kind: DirectPeerRelayPermitKind::HalfOpen,
        }
    }

    fn cancel(&mut self, now: u64, permit: ChatRelayDirectPeerPermit) {
        if permit.kind != DirectPeerRelayPermitKind::HalfOpen
            || permit.generation != self.generation
        {
            return;
        }
        let DirectPeerRelayCircuitState::HalfOpenInFlight {
            successful_probes, ..
        } = self.state
        else {
            return;
        };
        self.generation = self.generation.wrapping_add(1);
        self.state = DirectPeerRelayCircuitState::HalfOpenReady { successful_probes };
        self.last_transition_at = Some(now);
    }

    fn complete(
        &mut self,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
        delivery_succeeded: bool,
        slo_failed: bool,
    ) -> bool {
        if !self.accepts_completion(permit) {
            return false;
        }
        match (permit.kind, self.state) {
            (DirectPeerRelayPermitKind::Closed, DirectPeerRelayCircuitState::Closed) => {
                if !delivery_succeeded && slo_failed {
                    self.open(now);
                }
            }
            (
                DirectPeerRelayPermitKind::HalfOpen,
                DirectPeerRelayCircuitState::HalfOpenInFlight {
                    successful_probes, ..
                },
            ) => {
                if !delivery_succeeded {
                    self.half_open_failed_total = self.half_open_failed_total.saturating_add(1);
                    self.open(now);
                    return false;
                }

                self.half_open_succeeded_total = self.half_open_succeeded_total.saturating_add(1);
                let successful_probes = successful_probes.saturating_add(1);
                self.generation = self.generation.wrapping_add(1);
                if successful_probes >= DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES {
                    self.state = DirectPeerRelayCircuitState::Closed;
                    self.recovered_total = self.recovered_total.saturating_add(1);
                } else {
                    self.state = DirectPeerRelayCircuitState::HalfOpenReady { successful_probes };
                }
                self.last_transition_at = Some(now);
            }
            _ => {}
        }
        !matches!(self.state, DirectPeerRelayCircuitState::Open { .. })
    }

    fn open(&mut self, now: u64) {
        self.generation = self.generation.wrapping_add(1);
        self.state = DirectPeerRelayCircuitState::Open {
            retry_at: now.saturating_add(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS),
        };
        self.opened_total = self.opened_total.saturating_add(1);
        self.last_transition_at = Some(now);
    }

    fn snapshot(&self, now: u64) -> ChatRelayDirectPeerCircuitStatus {
        let (state, successful_probes, open_remaining_seconds) = match self.state {
            DirectPeerRelayCircuitState::Closed => ("closed", 0, None),
            DirectPeerRelayCircuitState::Open { retry_at } if now < retry_at => {
                ("open", 0, Some(retry_at.saturating_sub(now)))
            }
            DirectPeerRelayCircuitState::Open { .. } => ("half_open", 0, None),
            DirectPeerRelayCircuitState::HalfOpenReady { successful_probes }
            | DirectPeerRelayCircuitState::HalfOpenInFlight {
                successful_probes, ..
            } => ("half_open", successful_probes, None),
        };
        ChatRelayDirectPeerCircuitStatus {
            state: state.to_string(),
            half_open_consecutive_successes: successful_probes,
            opened_total: self.opened_total,
            blocked_total: self.blocked_total,
            half_open_attempted_total: self.half_open_attempted_total,
            half_open_succeeded_total: self.half_open_succeeded_total,
            half_open_failed_total: self.half_open_failed_total,
            recovered_total: self.recovered_total,
            open_remaining_seconds,
            last_transition_at: self.last_transition_at,
            restart_protected: self.restart_protected,
            checkpoint_loaded_at: self.checkpoint_loaded_at,
            checkpoint_persisted_at: self.checkpoint_persisted_at,
            checkpoint_failures_total: self.checkpoint_failures_total,
            last_checkpoint_failure_at: self.last_checkpoint_failure_at,
            ..ChatRelayDirectPeerCircuitStatus::default()
        }
    }
}

#[derive(Debug)]
struct DirectPeerRelayCircuitCheckpointRow {
    schema_version: i64,
    state: String,
    successful_probes: i64,
    deadline_at: Option<i64>,
    opened_total: i64,
    blocked_total: i64,
    half_open_attempted_total: i64,
    half_open_succeeded_total: i64,
    half_open_failed_total: i64,
    recovered_total: i64,
    last_transition_at: Option<i64>,
    checkpoint_failures_total: i64,
    last_checkpoint_failure_at: Option<i64>,
    updated_at: i64,
}

impl DirectPeerRelayCircuitCheckpointRow {
    fn into_circuit(self, now: u64) -> ChatRelayResult<(DirectPeerRelayCircuit, bool)> {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Decode every
        // persisted scalar through a bounded conversion before admitting relay.
        if self.schema_version != DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_version",
            });
        }
        let successful_probes = u8::try_from(self.successful_probes).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_probe_count",
            }
        })?;
        if successful_probes >= DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_probe_count",
            });
        }
        let deadline_at = optional_nonnegative_sqlite_value(
            self.deadline_at,
            "direct_peer_circuit_checkpoint_deadline",
        )?;
        let last_transition_at = optional_nonnegative_sqlite_value(
            self.last_transition_at,
            "direct_peer_circuit_checkpoint_transition",
        )?;
        let last_checkpoint_failure_at = optional_nonnegative_sqlite_value(
            self.last_checkpoint_failure_at,
            "direct_peer_circuit_checkpoint_failure_time",
        )?;
        let updated_at =
            nonnegative_sqlite_value(self.updated_at, "direct_peer_circuit_checkpoint_updated_at")?;
        if last_transition_at.is_some_and(|value| value > updated_at)
            || last_checkpoint_failure_at.is_some_and(|value| value > updated_at)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_time_order",
            });
        }
        let opened_total = nonnegative_sqlite_value(
            self.opened_total,
            "direct_peer_circuit_checkpoint_opened_total",
        )?;
        let blocked_total = nonnegative_sqlite_value(
            self.blocked_total,
            "direct_peer_circuit_checkpoint_blocked_total",
        )?;
        let half_open_attempted_total = nonnegative_sqlite_value(
            self.half_open_attempted_total,
            "direct_peer_circuit_checkpoint_attempted_total",
        )?;
        let half_open_succeeded_total = nonnegative_sqlite_value(
            self.half_open_succeeded_total,
            "direct_peer_circuit_checkpoint_succeeded_total",
        )?;
        let half_open_failed_total = nonnegative_sqlite_value(
            self.half_open_failed_total,
            "direct_peer_circuit_checkpoint_failed_total",
        )?;
        let recovered_total = nonnegative_sqlite_value(
            self.recovered_total,
            "direct_peer_circuit_checkpoint_recovered_total",
        )?;
        if half_open_succeeded_total.saturating_add(half_open_failed_total)
            > half_open_attempted_total
            || u64::from(successful_probes) > half_open_succeeded_total
            || recovered_total > opened_total
            || recovered_total.saturating_mul(u64::from(DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES))
                > half_open_succeeded_total
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_counter_relation",
            });
        }

        let state = match self.state.as_str() {
            "closed" if successful_probes == 0 && deadline_at.is_none() => {
                DirectPeerRelayCircuitState::Closed
            }
            "open" if successful_probes == 0 && opened_total > 0 => {
                DirectPeerRelayCircuitState::Open {
                    retry_at: deadline_at.ok_or(ChatRelayError::CorruptStoredData {
                        field: "direct_peer_circuit_checkpoint_open_deadline",
                    })?,
                }
            }
            "half_open_ready" if deadline_at.is_none() && opened_total > 0 => {
                DirectPeerRelayCircuitState::HalfOpenReady { successful_probes }
            }
            "half_open_in_flight" if opened_total > 0 => {
                DirectPeerRelayCircuitState::HalfOpenInFlight {
                    successful_probes,
                    lease_expires_at: deadline_at.ok_or(ChatRelayError::CorruptStoredData {
                        field: "direct_peer_circuit_checkpoint_probe_deadline",
                    })?,
                }
            }
            _ => {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "direct_peer_circuit_checkpoint_state",
                });
            }
        };

        let mut circuit = DirectPeerRelayCircuit {
            state,
            generation: 0,
            opened_total,
            blocked_total,
            half_open_attempted_total,
            half_open_succeeded_total,
            half_open_failed_total,
            recovered_total,
            last_transition_at,
            restart_protected: true,
            checkpoint_loaded_at: Some(now),
            checkpoint_persisted_at: Some(updated_at),
            checkpoint_failures_total: nonnegative_sqlite_value(
                self.checkpoint_failures_total,
                "direct_peer_circuit_checkpoint_failures_total",
            )?,
            last_checkpoint_failure_at,
        };

        let clock_rollback =
            updated_at > now.saturating_add(DIRECT_PEER_RELAY_CIRCUIT_CLOCK_SKEW_SECS);
        let interrupted_probe = matches!(
            circuit.state,
            DirectPeerRelayCircuitState::HalfOpenInFlight { .. }
        );
        let needs_rewrite = clock_rollback || interrupted_probe;
        if needs_rewrite {
            if interrupted_probe {
                circuit.half_open_failed_total = circuit.half_open_failed_total.saturating_add(1);
            }
            circuit.open(now);
        }
        Ok((circuit, needs_rewrite))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutboundRouteClass {
    AuthenticatedOnion,
    DirectPeer,
}

/// Aggregate commit-durability evidence for encrypted relay custody.
///
/// [CHAT-RELAY-DURABILITY-STATUS 2026-08-16 by Codex] This is configuration
/// evidence, not traffic telemetry. It intentionally carries no database path,
/// message, wallet, peer, endpoint, payload, or row-count dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayCustodyDurabilityStatus {
    /// Stable state bucket: `unknown` or `full`.
    pub state: String,
    /// Whether activation read back SQLite FULL-or-stronger commit durability.
    pub full_durability_verified: bool,
    /// Effective SQLite synchronous level, when verified by the service.
    pub synchronous_level: Option<u8>,
}

impl Default for ChatRelayCustodyDurabilityStatus {
    fn default() -> Self {
        Self {
            state: "unknown".to_string(),
            full_durability_verified: false,
            synchronous_level: None,
        }
    }
}

impl ChatRelayCustodyDurabilityStatus {
    fn verified_full(synchronous_level: u8) -> Self {
        Self {
            state: "full".to_string(),
            full_durability_verified: true,
            synchronous_level: Some(synchronous_level),
        }
    }
}

/// Privacy-safe node-to-node encrypted chat relay health snapshot.
///
/// This structure intentionally contains only aggregate counters and stable
/// reason buckets. It must not include message IDs, wallet IDs, client IPs,
/// destinations, DNS contents, URLs, chat plaintext, ciphertext, private keys,
/// voucher secrets, or per-user traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayPeerStatus {
    /// Whether chat relay is enabled in local config.
    pub enabled: bool,
    /// Verified aggregate commit durability for encrypted custody.
    #[serde(default)]
    pub custody_durability: ChatRelayCustodyDurabilityStatus,
    /// Total outbound peer relay attempts.
    pub outbound_attempted_total: u64,
    /// Total outbound peer relay requests accepted by peer nodes.
    pub outbound_accepted_total: u64,
    /// Total outbound peer relay requests that failed or were rejected.
    pub outbound_failed_total: u64,
    /// Total outbound fanout rounds observed.
    pub outbound_rounds: u64,
    /// Number of peers attempted in the last outbound fanout round.
    pub last_outbound_attempted: u64,
    /// Number of peers that accepted the last outbound fanout round.
    pub last_outbound_accepted: u64,
    /// Number of peers that failed the last outbound fanout round.
    pub last_outbound_failed: u64,
    /// Last outbound health bucket: healthy, degraded, failed, idle.
    pub last_outbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last outbound relay failure.
    pub last_outbound_failure_reason: Option<String>,
    /// Consecutive outbound rounds with zero accepted peer relays.
    pub consecutive_outbound_failures: u64,
    /// Timestamp of the last outbound round with at least one accepted peer.
    pub last_outbound_success_at: Option<u64>,
    /// Timestamp of the last outbound fanout round.
    pub last_outbound_at: Option<u64>,
    /// Authenticated receipt-verified onion relay health, isolated from fallback.
    #[serde(default)]
    pub authenticated_onion_outbound: ChatRelayOutboundRouteStatus,
    /// Compatibility direct peer relay health, isolated from onion delivery.
    #[serde(default)]
    pub direct_peer_outbound: ChatRelayOutboundRouteStatus,
    /// Aggregate target-bound direct relay retry reliability.
    #[serde(default)]
    pub direct_peer_retry: ChatRelayDirectPeerRetryStatus,
    /// Total inbound peer relay envelopes accepted for local processing.
    pub inbound_accepted_total: u64,
    /// Total inbound duplicate envelopes ignored idempotently.
    pub inbound_duplicate_total: u64,
    /// Total inbound envelopes delivered to online local sessions.
    pub inbound_delivered_online_total: u64,
    /// Total inbound envelopes stored in the local pending queue.
    pub inbound_stored_pending_total: u64,
    /// Total inbound peer relay requests rejected by local validation/storage.
    pub inbound_rejected_total: u64,
    /// Last inbound status bucket: accepted, duplicate, rejected.
    pub last_inbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last inbound rejection.
    pub last_inbound_failure_reason: Option<String>,
    /// Timestamp of the last inbound peer relay request processed.
    pub last_inbound_at: Option<u64>,
}

impl ChatRelayPeerStatus {
    /// Creates an empty aggregate snapshot for one configured runtime state.
    ///
    /// [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] This is crate-visible so
    /// host-local health can publish the exact same typed contract when relay
    /// runtime initialization is unavailable, without duplicating counters or
    /// introducing a second telemetry model.
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            custody_durability: ChatRelayCustodyDurabilityStatus::default(),
            outbound_attempted_total: 0,
            outbound_accepted_total: 0,
            outbound_failed_total: 0,
            outbound_rounds: 0,
            last_outbound_attempted: 0,
            last_outbound_accepted: 0,
            last_outbound_failed: 0,
            last_outbound_status: None,
            last_outbound_failure_reason: None,
            consecutive_outbound_failures: 0,
            last_outbound_success_at: None,
            last_outbound_at: None,
            authenticated_onion_outbound: ChatRelayOutboundRouteStatus::default(),
            direct_peer_outbound: ChatRelayOutboundRouteStatus::default(),
            direct_peer_retry: ChatRelayDirectPeerRetryStatus::default(),
            inbound_accepted_total: 0,
            inbound_duplicate_total: 0,
            inbound_delivered_online_total: 0,
            inbound_stored_pending_total: 0,
            inbound_rejected_total: 0,
            last_inbound_status: None,
            last_inbound_failure_reason: None,
            last_inbound_at: None,
        }
    }
}

// ============================================
// Error type
// ============================================

/// Errors produced by `ChatRelayService`.
#[derive(Debug, thiserror::Error)]
pub enum ChatRelayError {
    /// SQLite schema, query, transaction, or persistence failure.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Envelope or notification serialization failure.
    #[error("Serialization error: {0}")]
    Serialize(#[from] bincode::Error),

    /// One authenticated ACK frame exceeds the protocol processing ceiling.
    #[error("ACK batch too large: {size} message IDs (limit {limit})")]
    AckBatchTooLarge {
        /// Number of IDs supplied by the authenticated caller.
        size: usize,
        /// Protocol-defined processing ceiling.
        limit: usize,
    },

    /// Durable relay data violates a fixed-size or bounded storage invariant.
    #[error("Corrupt stored relay data: {field}")]
    CorruptStoredData {
        /// Stable aggregate-only field bucket; never include stored values.
        field: &'static str,
    },

    /// A client-supplied timestamp cannot be represented by SQLite INTEGER.
    #[error("Message timestamp is outside the supported range")]
    TimestampOutOfRange,

    /// An existing durable row uses the same ID for a different signed envelope.
    #[error("Message ID conflicts with an existing durable envelope")]
    MessageIdConflict,

    /// A ChatPullV2 cursor has an invalid length, version, binding, or tag.
    #[error("Invalid or expired opaque pull cursor")]
    InvalidPullCursor,

    /// The node could not derive or encrypt an opaque ChatPullV2 cursor.
    #[error("Unable to protect opaque pull cursor")]
    PullCursorEncryptionFailed,

    /// The durable monotonic queue sequence reached SQLite INTEGER capacity.
    #[error("Durable relay queue sequence exhausted")]
    QueueSequenceExhausted,

    /// Encrypted message ciphertext exceeds the configured item ceiling.
    #[error("Message too large: {size} bytes (limit {limit})")]
    MessageTooLarge {
        /// Incoming ciphertext bytes.
        size: usize,
        /// Configured ciphertext byte ceiling.
        limit: usize,
    },

    /// One receiver already holds the configured maximum pending messages.
    #[error("Mailbox full: receiver has {current} pending messages (limit {limit})")]
    MailboxFull {
        /// Current pending rows for the receiver.
        current: usize,
        /// Configured per-receiver row ceiling.
        limit: usize,
    },

    /// Node-wide pending message count is at capacity.
    #[error("Pending message queue full: {current} messages (limit {limit})")]
    PendingMessageQueueFull {
        /// Current active pending rows on the node.
        current: usize,
        /// Configured node-wide pending row ceiling.
        limit: usize,
    },

    /// Adding a message would exceed node-wide pending encoded bytes.
    #[error("Pending message byte quota exceeded: {current} + {incoming} bytes (limit {limit})")]
    PendingMessageBytesExceeded {
        /// Current encoded pending bytes.
        current: u64,
        /// Encoded bytes required by the incoming envelope.
        incoming: u64,
        /// Configured node-wide encoded byte ceiling.
        limit: u64,
    },

    /// One receiver already holds the configured maximum encrypted blobs.
    #[error("Blob quota exceeded: receiver has {current} pending blobs (limit {limit})")]
    BlobQuotaExceeded {
        /// Current blob rows for the receiver.
        current: usize,
        /// Configured per-receiver blob ceiling.
        limit: usize,
    },

    /// Node-wide encrypted blob count is at capacity.
    #[error("Pending blob store full: {current} blobs (limit {limit})")]
    PendingBlobStoreFull {
        /// Current retained blob rows on the node.
        current: usize,
        /// Configured node-wide blob row ceiling.
        limit: usize,
    },

    /// Adding an encrypted blob would exceed node-wide retained blob bytes.
    #[error("Pending blob byte quota exceeded: {current} + {incoming} bytes (limit {limit})")]
    PendingBlobBytesExceeded {
        /// Current retained encrypted blob bytes.
        current: u64,
        /// Incoming encrypted blob bytes.
        incoming: u64,
        /// Configured node-wide encrypted blob byte ceiling.
        limit: u64,
    },

    /// One encrypted blob exceeds the configured item ceiling.
    #[error("Blob too large: {size} bytes (limit {limit})")]
    BlobTooLarge {
        /// Incoming encrypted blob bytes.
        size: usize,
        /// Configured encrypted blob byte ceiling.
        limit: usize,
    },

    /// The opaque blob identifier does not resolve to a retained object.
    #[error("Blob not found: {blob_id}")]
    BlobNotFound {
        /// Opaque HMAC-derived identifier supplied by the caller.
        blob_id: String,
    },

    /// The authenticated caller is not allowed to mutate the object.
    #[error("Unauthorized: sender mismatch")]
    Unauthorized,
}

impl ChatRelayError {
    /// Returns a stable aggregate-only diagnostics bucket.
    #[must_use]
    pub const fn reason_bucket(&self) -> &'static str {
        match self {
            Self::Sqlite(_) => "sqlite_error",
            Self::Serialize(_) => "serialization_error",
            Self::AckBatchTooLarge { .. } => "ack_batch_too_large",
            Self::CorruptStoredData { .. } => "corrupt_stored_data",
            Self::TimestampOutOfRange => "timestamp_out_of_range",
            Self::MessageIdConflict => "message_id_conflict",
            Self::InvalidPullCursor => "invalid_pull_cursor",
            Self::PullCursorEncryptionFailed => "pull_cursor_encryption_failed",
            Self::QueueSequenceExhausted => "queue_sequence_exhausted",
            Self::MessageTooLarge { .. } => "message_too_large",
            Self::MailboxFull { .. } => "mailbox_full",
            Self::PendingMessageQueueFull { .. } => "pending_message_count_quota",
            Self::PendingMessageBytesExceeded { .. } => "pending_message_byte_quota",
            Self::BlobQuotaExceeded { .. } => "receiver_blob_quota",
            Self::PendingBlobStoreFull { .. } => "pending_blob_count_quota",
            Self::PendingBlobBytesExceeded { .. } => "pending_blob_byte_quota",
            Self::BlobTooLarge { .. } => "blob_too_large",
            Self::BlobNotFound { .. } => "blob_not_found",
            Self::Unauthorized => "unauthorized",
        }
    }

    /// Whether retrying without queue cleanup or operator action cannot help.
    #[must_use]
    pub const fn is_capacity_exhausted(&self) -> bool {
        matches!(
            self,
            Self::MailboxFull { .. }
                | Self::PendingMessageQueueFull { .. }
                | Self::PendingMessageBytesExceeded { .. }
                | Self::BlobQuotaExceeded { .. }
                | Self::PendingBlobStoreFull { .. }
                | Self::PendingBlobBytesExceeded { .. }
        )
    }
}

pub type ChatRelayResult<T> = Result<T, ChatRelayError>;

/// Aggregate durable relay usage with no user or routing identifiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayStorageUsage {
    /// Active pending message rows.
    pub pending_messages: u64,
    /// Encoded bytes held by active pending messages.
    pub pending_message_bytes: u64,
    /// Pending encrypted blob rows.
    pub pending_blobs: u64,
    /// Encrypted blob bytes retained by the node.
    pub pending_blob_bytes: u64,
}

/// Aggregate TTL maintenance evidence safe for heartbeat and node health APIs.
///
/// This snapshot intentionally excludes message IDs, wallet keys, routes,
/// endpoints, payloads, and per-user counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayMaintenanceStatus {
    /// Total cleanup attempts, including failed transactions.
    pub cleanup_runs_total: u64,
    /// Cleanup attempts that returned an error.
    pub cleanup_failures_total: u64,
    /// Successfully committed bounded cleanup transactions.
    pub cleanup_batches_total: u64,
    /// Runs that reached their transaction budget with work still pending.
    pub cleanup_backlog_deferred_total: u64,
    /// Pending message rows removed by successfully committed batches.
    pub expired_messages_total: u64,
    /// Encrypted blob rows removed by successfully committed batches.
    pub expired_blobs_total: u64,
    /// Delivered or stale expiry-notification rows removed by committed batches.
    pub expired_notifications_removed_total: u64,
    /// Corrupt pending-message rows atomically isolated from active delivery.
    pub quarantined_pending_messages_total: u64,
    /// Corrupt expiry-notification rows atomically isolated from delivery.
    pub quarantined_expired_notifications_total: u64,
    /// De-identified quarantine event rows removed by bounded retention.
    pub quarantine_events_removed_total: u64,
    /// Current durable de-identified quarantine event rows.
    pub quarantine_events_retained: u64,
    /// Unix timestamp of the most recent poison-row isolation.
    pub last_quarantine_at: Option<u64>,
    /// Unix timestamp of the most recent cleanup attempt.
    pub last_cleanup_at: Option<u64>,
    /// Stable state bucket: `succeeded` or `failed`.
    pub last_cleanup_status: Option<String>,
    /// Stable aggregate failure bucket from [`ChatRelayError::reason_bucket`].
    pub last_cleanup_failure_reason: Option<String>,
    /// Number of successfully committed transactions in the latest run.
    pub last_cleanup_batches: u64,
    /// Whether the latest run deferred remaining work to the next timer tick.
    pub last_cleanup_backlog_deferred: bool,
    /// Corrupt pending-message rows isolated by the latest cleanup run.
    pub last_cleanup_quarantined_pending_messages: u64,
}

// ============================================
// Pending message row (returned from pull)
// ============================================

/// A pending offline message retrieved from the store.
#[derive(Debug)]
pub struct PendingMessage {
    /// Opaque client-generated message identifier used for ACK pagination.
    pub message_id: [u8; 16],
    /// Signed end-to-end encrypted envelope; relay code must not inspect its ciphertext.
    pub envelope: ChatEnvelope,
}

/// One stable ChatPullV2 page and the opaque continuation state for it.
#[derive(Debug)]
pub struct PendingMessagePageV2 {
    /// Valid signed envelopes returned to the authenticated receiver.
    pub messages: Vec<PendingMessage>,
    /// AEAD-protected continuation cursor. An empty request cursor starts a new snapshot.
    pub next_cursor: Vec<u8>,
    /// Whether the caller should continue the current snapshot with `next_cursor`.
    pub has_more: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PullCursorV2 {
    position: u64,
    ceiling: u64,
}

#[derive(Debug)]
struct ExpiredMessageRow {
    message_id: [u8; 16],
    sender: [u8; 32],
    receiver: [u8; 32],
}

type ExpiredMessagesBySender = HashMap<[u8; 32], HashMap<[u8; 32], Vec<[u8; 16]>>>;
#[derive(Debug)]
struct StoredExpiredNotificationRow {
    id: i64,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    message_ids_raw: Vec<u8>,
}

#[derive(Debug)]
struct StoredPendingMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
}

#[derive(Debug)]
struct StoredSequencedPendingMessageRow {
    queue_sequence: i64,
    row: StoredPendingMessageRow,
}

#[derive(Debug)]
struct StoredExpiredMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
    queue_sequence: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
struct CorruptDurableRow {
    row_key: i64,
    source_kind: &'static str,
    reason: &'static str,
    encoded_bytes: u64,
}

#[derive(Debug, Default)]
struct ValidatedExpiredMessageBatch {
    valid_rows: Vec<ExpiredMessageRow>,
    corrupt_rows: Vec<CorruptDurableRow>,
    selected_rowids: Vec<i64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CleanupBatchOutcome {
    expired_messages: usize,
    expired_blobs: usize,
    removed_notifications: usize,
    quarantined_pending_messages: usize,
    removed_quarantine_events: usize,
    retained_quarantine_events: usize,
    has_more: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CleanupRunSummary {
    expired_messages: usize,
    expired_blobs: usize,
    removed_notifications: usize,
    quarantined_pending_messages: usize,
    removed_quarantine_events: usize,
    retained_quarantine_events: usize,
    successful_batches: usize,
    backlog_deferred: bool,
}

impl CleanupRunSummary {
    fn absorb(&mut self, batch: CleanupBatchOutcome) {
        self.expired_messages = self.expired_messages.saturating_add(batch.expired_messages);
        self.expired_blobs = self.expired_blobs.saturating_add(batch.expired_blobs);
        self.removed_notifications = self
            .removed_notifications
            .saturating_add(batch.removed_notifications);
        self.quarantined_pending_messages = self
            .quarantined_pending_messages
            .saturating_add(batch.quarantined_pending_messages);
        self.removed_quarantine_events = self
            .removed_quarantine_events
            .saturating_add(batch.removed_quarantine_events);
        self.retained_quarantine_events = batch.retained_quarantine_events;
        self.successful_batches = self.successful_batches.saturating_add(1);
    }

    fn removed_anything(self) -> bool {
        self.expired_messages > 0
            || self.expired_blobs > 0
            || self.removed_notifications > 0
            || self.quarantined_pending_messages > 0
            || self.removed_quarantine_events > 0
    }
}

// ============================================
// Expired notification row
// ============================================

/// A queued `ChatExpired` notification for an offline sender.
#[derive(Debug)]
pub struct ExpiredNotification {
    /// Local notification row identifier.
    pub id: i64,
    /// Original sender public key used only for authenticated delivery lookup.
    pub sender: [u8; 32],
    /// Original receiver public key returned inside the encrypted client flow.
    pub receiver: [u8; 32],
    /// bincode-serialised `Vec<[u8; 16]>`
    pub message_ids_raw: Vec<u8>,
}

/// Aggregate result of one audited, idempotent custody backup operation.
///
/// [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] This intentionally
/// excludes the filesystem path and opaque artifact key. Management callers
/// may report only whether a verified image was created or reused and its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatRelayBackupReceipt {
    /// Size of the verified SQLite recovery image.
    pub(crate) size_bytes: u64,
    /// `true` only when this invocation published the artifact.
    pub(crate) created: bool,
}

/// Aggregate, path-free result of a verified custody-backup retention audit.
///
/// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] No artifact name,
/// operation ID, filesystem timestamp, identity, route, or payload-derived
/// value may cross this boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayBackupRetentionReceipt {
    /// Verified recovery images modeled as retained under the planning target.
    pub retained_count: usize,
    /// Aggregate verified bytes modeled as retained under the planning target.
    pub retained_bytes: u64,
    /// Verified recovery images exceeding the configured retention policy.
    pub excess_count: usize,
    /// Aggregate verified bytes exceeding the configured retention policy.
    pub excess_bytes: u64,
    /// Incomplete private SQLite entries observed after interrupted runs.
    pub partial_count: usize,
    /// Aggregate incomplete bytes observed after interrupted runs.
    pub partial_bytes: u64,
    /// The inventory or newest recovery point exceeds the byte target.
    pub budget_exceeded: bool,
}

/// Explicit host-local backup-prune request.
///
/// `execute=false` is a dry-run. Execution requires the exact public
/// confirmation phrase and an operator assertion that the node process is
/// stopped. These gates supplement, rather than replace, filesystem locking.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatRelayBackupPruneRequest {
    /// Whether eligible artifacts should actually be deleted.
    pub execute: bool,
    /// Exact confirmation phrase required for execution.
    pub confirmation: Option<String>,
    /// Operator assertion that the serving node process has been stopped.
    pub node_stopped_confirmed: bool,
}

/// Aggregate, path-free result of one host-local prune command.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayBackupPruneReceipt {
    /// Whether this invocation performed deletion rather than a dry-run.
    pub executed: bool,
    /// Verified complete recovery images selected by policy.
    pub planned_backup_count: usize,
    /// Aggregate bytes of selected complete recovery images.
    pub planned_backup_bytes: u64,
    /// Grace-expired interrupted files selected by policy.
    pub planned_partial_count: usize,
    /// Aggregate bytes of selected interrupted files.
    pub planned_partial_bytes: u64,
    /// Complete recovery images deleted by this invocation.
    pub deleted_backup_count: usize,
    /// Aggregate complete recovery-image bytes deleted.
    pub deleted_backup_bytes: u64,
    /// Interrupted files deleted by this invocation.
    pub deleted_partial_count: usize,
    /// Aggregate interrupted-file bytes deleted.
    pub deleted_partial_bytes: u64,
    /// Verified post-command retention state.
    pub remaining: ChatRelayBackupRetentionReceipt,
}

/// Aggregate, path-free result of a read-only recovery preflight.
///
/// [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] This contract never
/// identifies an artifact and never replaces or removes active/backup storage.
/// A ready result means an operator may evaluate a separately approved restore
/// flow; it does not mean a restore has happened.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayRestoreReadinessReceipt {
    /// Whether all preflight gates needed by a future restore are satisfied.
    pub ready: bool,
    /// Number of fully verified recovery images in the private boundary.
    pub verified_backup_count: usize,
    /// Size of the newest fully verified recovery image.
    pub selected_backup_bytes: u64,
    /// Whether the configured active main database currently exists.
    pub active_database_present: bool,
    /// Size of the active main database, or zero when absent.
    pub active_database_bytes: u64,
    /// Whether any active SQLite journal/WAL/SHM sidecar exists.
    pub active_sidecars_present: bool,
    /// Stable aggregate blocker code; absent when `ready=true`.
    pub blocker: Option<&'static str>,
}

/// Short-lived, path-free commitment to one verified recovery plan.
///
/// [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] The HMAC binds this public
/// aggregate state to private backup/active-file identity metadata without
/// disclosing those identities. The plan is a stale-state guard only. A future
/// restore must additionally require an explicit stopped-node assertion,
/// confirmation phrase, rollback image, and post-restore verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRelayRestorePlanReceipt {
    /// Restore-plan wire contract version.
    pub version: u8,
    /// Host wall-clock issue time in Unix seconds.
    pub issued_at: u64,
    /// Exclusive expiry time in Unix seconds.
    pub expires_at: u64,
    /// Number of verified recovery images observed when planning.
    pub verified_backup_count: u64,
    /// Size of the selected newest recovery image.
    pub selected_backup_bytes: u64,
    /// Whether the configured active main database existed at issuance.
    pub active_database_present: bool,
    /// Size of the active main database at issuance, or zero when absent.
    pub active_database_bytes: u64,
    /// Per-plan random lowercase hexadecimal nonce.
    pub nonce: String,
    /// Lowercase HMAC-SHA256 commitment over public and private plan state.
    pub commitment: String,
}

/// Private filesystem metadata used only while enforcing backup retention.
#[derive(Clone)]
struct ChatRelayBackupArtifact {
    path: PathBuf,
    file_name: String,
    size_bytes: u64,
    modified_at: SystemTime,
    #[cfg(unix)]
    device_id: u64,
    #[cfg(unix)]
    inode: u64,
}

struct ChatRelayBackupRetentionInspection {
    receipt: ChatRelayBackupRetentionReceipt,
    newest_backup: Option<ChatRelayBackupArtifact>,
    excess_backups: Vec<ChatRelayBackupArtifact>,
    stale_partials: Vec<ChatRelayBackupArtifact>,
}

/// Private active-custody metadata included in restore-plan commitments.
#[derive(Default)]
struct ChatRelayActiveRestoreBoundary {
    present: bool,
    size_bytes: u64,
    sidecars_present: bool,
    modified_at: Option<SystemTime>,
    device_id: u64,
    inode: u64,
}

/// Canonical HMAC input. Private identities never cross the service boundary.
#[derive(Serialize)]
struct ChatRelayRestorePlanSigningState<'a> {
    version: u8,
    issued_at: u64,
    expires_at: u64,
    verified_backup_count: u64,
    selected_backup_bytes: u64,
    active_database_present: bool,
    active_database_bytes: u64,
    nonce: &'a str,
    configured_database_path: &'a str,
    selected_backup_name: &'a str,
    selected_backup_modified_secs: u64,
    selected_backup_modified_nanos: u32,
    active_database_modified_secs: u64,
    active_database_modified_nanos: u32,
    selected_backup_device_id: u64,
    selected_backup_inode: u64,
    active_database_device_id: u64,
    active_database_inode: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRelayBackupMaintenanceAuditRecord {
    version: u8,
    sequence: u64,
    timestamp: u64,
    action: String,
    phase: String,
    planned_backup_count: u64,
    planned_backup_bytes: u64,
    planned_partial_count: u64,
    planned_partial_bytes: u64,
    completed_backup_count: u64,
    completed_backup_bytes: u64,
    completed_partial_count: u64,
    completed_partial_bytes: u64,
    previous_mac: String,
    mac: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChatRelayBackupMaintenanceAuditCounts {
    planned_backup_count: usize,
    planned_backup_bytes: u64,
    planned_partial_count: usize,
    planned_partial_bytes: u64,
    completed_backup_count: usize,
    completed_backup_bytes: u64,
    completed_partial_count: usize,
    completed_partial_bytes: u64,
}

impl ExpiredNotification {
    /// Deserialise the stored message IDs.
    pub fn message_ids(&self) -> ChatRelayResult<Vec<[u8; 16]>> {
        if self.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_payload_size",
            });
        }
        let message_ids: Vec<[u8; 16]> = bincode::deserialize(&self.message_ids_raw)?;
        if message_ids.is_empty() || message_ids.len() > MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_message_count",
            });
        }
        Ok(message_ids)
    }
}

// ============================================
// Minimal LRU for online-path deduplication
// ============================================

/// Fixed-capacity LRU cache for `message_id` deduplication on the online path.
struct MessageDedup {
    map: DashMap<[u8; 16], u64>,
    capacity: usize,
    seq: AtomicU64,
}

impl MessageDedup {
    fn new(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            capacity,
            seq: AtomicU64::new(0),
        }
    }

    /// Returns `true` if the message_id was already seen (duplicate).
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        match self.map.entry(*message_id) {
            Entry::Occupied(_) => return true,
            Entry::Vacant(entry) => {
                entry.insert(seq);
            }
        }

        if self.map.len() > self.capacity {
            let oldest_key = self.map.iter().min_by_key(|e| *e.value()).map(|e| *e.key());
            if let Some(k) = oldest_key {
                self.map.remove(&k);
            }
        }
        false
    }
}

// ============================================
// ChatRelayService
// ============================================

/// Central service for zero-knowledge P2P chat relay.
///
/// ## v1.3.0-Sovereign additions
/// - `wallet_routes`: Arc-wrapped WalletRouteCache for wallet→session routing.
///   Exposed as a public Arc so `server.rs` can hold an independent clone for
///   the background cleanup task and for passing into message handlers.
pub struct ChatRelayService {
    config: ChatRelayConfig,
    conn: Mutex<Connection>,
    node_secret: [u8; 32],
    /// Stable node-local AEAD key for opaque v2 pull cursors.
    pull_cursor_key: [u8; 32],
    dedup: MessageDedup,
    peer_status: RwLock<ChatRelayPeerStatus>,
    direct_peer_retry_slo: Mutex<DirectPeerRetrySloWindow>,
    direct_peer_relay_circuit: Mutex<DirectPeerRelayCircuit>,
    maintenance_status: RwLock<ChatRelayMaintenanceStatus>,
    /// Serializes backup publication, replay verification, and retention.
    backup_operations: Mutex<()>,
    /// In-memory wallet → session routing table.
    ///
    /// Arc so the cleanup task and each handler can hold independent references
    /// without borrowing the whole ChatRelayService.
    pub wallet_routes: Arc<WalletRouteCache>,
}

impl ChatRelayService {
    #[cfg(unix)]
    fn restrict_sqlite_file_permissions(path: &Path) -> ChatRelayResult<()> {
        use std::os::unix::fs::PermissionsExt;

        // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] SQLite creates WAL
        // sidecars using the primary database mode. Tighten the primary file
        // before enabling WAL, and keep the configured path out of the error.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|_| {
            ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_PERM),
                Some("unable to restrict relay database permissions".to_string()),
            ))
        })
    }

    #[cfg(not(unix))]
    fn restrict_sqlite_file_permissions(_path: &Path) -> ChatRelayResult<()> {
        Ok(())
    }

    fn verify_sqlite_integrity(
        conn: &Connection,
        failure_field: &'static str,
    ) -> ChatRelayResult<()> {
        // [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] `quick_check(1)`
        // bounds returned findings while still traversing the database. Both a
        // non-`ok` finding and an SQLite error collapse to one path-free bucket.
        let outcome = conn
            .query_row("PRAGMA quick_check(1)", [], |row| row.get::<_, String>(0))
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: failure_field,
            })?;
        if outcome != "ok" {
            return Err(ChatRelayError::CorruptStoredData {
                field: failure_field,
            });
        }
        Ok(())
    }

    fn configure_sqlite_durability(conn: &Connection) -> ChatRelayResult<u8> {
        // [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] NORMAL protects
        // SQLite consistency across process failure but may lose a recently
        // acknowledged transaction after host power loss. The relay signs
        // custody receipts and persists its outage circuit from this database,
        // so activation requires FULL-or-stronger durability.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        let synchronous_level =
            conn.query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))?;
        if synchronous_level < CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_synchronous_level",
            });
        }
        u8::try_from(synchronous_level).map_err(|_| ChatRelayError::CorruptStoredData {
            field: "sqlite_synchronous_level",
        })
    }

    fn backup_io_error(code: i32, message: &'static str) -> ChatRelayError {
        ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            Some(message.to_string()),
        ))
    }

    #[cfg(unix)]
    fn reserve_private_backup_file(path: &Path) -> ChatRelayResult<()> {
        use std::os::unix::fs::OpenOptionsExt;

        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map(drop)
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to reserve private relay backup file",
                )
            })
    }

    #[cfg(not(unix))]
    fn reserve_private_backup_file(path: &Path) -> ChatRelayResult<()> {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map(drop)
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to reserve private relay backup file",
                )
            })
    }

    #[cfg(unix)]
    fn create_backup_directory(path: &Path) -> std::io::Result<()> {
        use std::os::unix::fs::DirBuilderExt;

        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700).create(path)
    }

    #[cfg(not(unix))]
    fn create_backup_directory(path: &Path) -> std::io::Result<()> {
        std::fs::create_dir(path)
    }

    #[cfg(unix)]
    fn restrict_backup_directory_permissions(path: &Path) -> ChatRelayResult<()> {
        use std::os::unix::fs::PermissionsExt;

        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "unable to restrict relay backup directory permissions",
            )
        })
    }

    #[cfg(not(unix))]
    fn restrict_backup_directory_permissions(_path: &Path) -> ChatRelayResult<()> {
        Ok(())
    }

    fn ensure_private_backup_directory(path: &Path) -> ChatRelayResult<()> {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CANTOPEN,
                        "relay backup boundary is not a private directory",
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // The active database parent already exists. A single-level,
                // owner-private create avoids a world-readable umask window.
                // A concurrent backup may win this create; re-inspection below
                // is the authority for an AlreadyExists race.
                match Self::create_backup_directory(path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(_) => {
                        return Err(Self::backup_io_error(
                            rusqlite::ffi::SQLITE_CANTOPEN,
                            "unable to create private relay backup directory",
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect private relay backup directory",
                ));
            }
        }
        let metadata = std::fs::symlink_metadata(path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect private relay backup directory",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup boundary is not a private directory",
            ));
        }
        Self::restrict_backup_directory_permissions(path)
    }

    fn private_backup_directory_for_config(config: &ChatRelayConfig) -> ChatRelayResult<PathBuf> {
        if config.db_path == ":memory:" {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "in-memory relay storage has no private backup boundary",
            ));
        }
        let source_path = Path::new(&config.db_path);
        let source_parent = source_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let backup_directory = source_parent.join(".aeronyx-relay-backups");
        Self::ensure_private_backup_directory(&backup_directory)?;
        Ok(backup_directory)
    }

    fn private_backup_directory(&self) -> ChatRelayResult<PathBuf> {
        Self::private_backup_directory_for_config(&self.config)
    }

    fn open_private_backup_control_file(path: &Path, append: bool) -> ChatRelayResult<File> {
        #[cfg(unix)]
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        #[cfg(not(unix))]
        if let Ok(metadata) = std::fs::symlink_metadata(path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup control boundary is not a private regular file",
                ));
            }
        }

        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(!append)
            .append(append)
            .create(true);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
        let file = options.open(path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open private relay backup control file",
            )
        })?;
        let metadata = file.metadata().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to inspect private relay backup control file",
            )
        })?;
        if !metadata.is_file() {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "relay backup control boundary is not a private regular file",
            ));
        }
        #[cfg(unix)]
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "relay backup control file is not owner-private",
            ));
        }
        Ok(file)
    }

    fn acquire_backup_filesystem_lock(backup_directory: &Path) -> ChatRelayResult<File> {
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private control parent",
            )
        })?;
        let lock_path = parent.join(CHAT_RELAY_BACKUP_LOCK_FILE_NAME);
        let lock_file = Self::open_private_backup_control_file(&lock_path, false)?;
        lock_file.try_lock().map_err(|error| {
            let code = match error {
                std::fs::TryLockError::WouldBlock => rusqlite::ffi::SQLITE_BUSY,
                std::fs::TryLockError::Error(_) => rusqlite::ffi::SQLITE_IOERR,
            };
            Self::backup_io_error(code, "relay backup maintenance lock is unavailable")
        })?;
        Ok(lock_file)
    }

    fn backup_audit_signing_bytes(
        record: &ChatRelayBackupMaintenanceAuditRecord,
    ) -> ChatRelayResult<Vec<u8>> {
        bincode::serialize(&(
            record.version,
            record.sequence,
            record.timestamp,
            record.action.as_str(),
            record.phase.as_str(),
            record.planned_backup_count,
            record.planned_backup_bytes,
            record.planned_partial_count,
            record.planned_partial_bytes,
            record.completed_backup_count,
            record.completed_backup_bytes,
            record.completed_partial_count,
            record.completed_partial_bytes,
            record.previous_mac.as_str(),
        ))
        .map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay backup maintenance audit",
            )
        })
    }

    fn backup_audit_mac(
        node_secret: &[u8; 32],
        record: &ChatRelayBackupMaintenanceAuditRecord,
    ) -> ChatRelayResult<String> {
        let mut mac = HmacSha256::new_from_slice(node_secret).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to initialize relay backup maintenance audit",
            )
        })?;
        mac.update(CHAT_RELAY_BACKUP_AUDIT_HMAC_DOMAIN);
        mac.update(&Self::backup_audit_signing_bytes(record)?);
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    fn verify_backup_audit_log(
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<(u64, String)> {
        let metadata = file.metadata().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to inspect relay backup maintenance audit",
            )
        })?;
        if metadata.len() > CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit exceeds its bounded size",
            ));
        }
        file.seek(SeekFrom::Start(0)).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to read relay backup maintenance audit",
            )
        })?;

        let mut reader = BufReader::new(file.try_clone().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to read relay backup maintenance audit",
            )
        })?);
        let mut expected_sequence = 1u64;
        let mut previous_mac = "0".repeat(64);
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to read relay backup maintenance audit",
                )
            })?;
            if bytes == 0 {
                break;
            }
            if bytes > CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES || !line.ends_with('\n') {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup maintenance audit record is malformed",
                ));
            }
            if usize::try_from(expected_sequence).unwrap_or(usize::MAX)
                > CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS
            {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup maintenance audit record limit reached",
                ));
            }
            let record: ChatRelayBackupMaintenanceAuditRecord =
                serde_json::from_str(line.trim_end_matches('\n')).map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CORRUPT,
                        "relay backup maintenance audit record is malformed",
                    )
                })?;
            if record.version != 1
                || record.sequence != expected_sequence
                || record.previous_mac != previous_mac
                || !Self::is_lower_hex(&record.previous_mac, 64)
                || !Self::is_lower_hex(&record.mac, 64)
                || record.action != "prune"
                || !matches!(
                    record.phase.as_str(),
                    "dry_run" | "planned" | "completed" | "failed"
                )
                || Self::backup_audit_mac(node_secret, &record)? != record.mac
            {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup maintenance audit verification failed",
                ));
            }
            previous_mac = record.mac;
            expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup maintenance audit sequence overflow",
                )
            })?;
        }
        Ok((expected_sequence - 1, previous_mac))
    }

    fn append_backup_maintenance_audit(
        backup_directory: &Path,
        node_secret: &[u8; 32],
        phase: &str,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        let audit_path = parent.join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        let mut file = Self::open_private_backup_control_file(&audit_path, true)?;
        let (last_sequence, previous_mac) = Self::verify_backup_audit_log(&mut file, node_secret)?;
        if usize::try_from(last_sequence).unwrap_or(usize::MAX)
            >= CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit record limit reached",
            ));
        }
        let mut record = ChatRelayBackupMaintenanceAuditRecord {
            version: 1,
            sequence: last_sequence.checked_add(1).ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup maintenance audit sequence overflow",
                )
            })?,
            timestamp,
            action: "prune".to_string(),
            phase: phase.to_string(),
            planned_backup_count: u64::try_from(counts.planned_backup_count).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_TOOBIG,
                    "relay backup maintenance count exceeds audit format",
                )
            })?,
            planned_backup_bytes: counts.planned_backup_bytes,
            planned_partial_count: u64::try_from(counts.planned_partial_count).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_TOOBIG,
                    "relay backup maintenance count exceeds audit format",
                )
            })?,
            planned_partial_bytes: counts.planned_partial_bytes,
            completed_backup_count: u64::try_from(counts.completed_backup_count).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_TOOBIG,
                    "relay backup maintenance count exceeds audit format",
                )
            })?,
            completed_backup_bytes: counts.completed_backup_bytes,
            completed_partial_count: u64::try_from(counts.completed_partial_count).map_err(
                |_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_TOOBIG,
                        "relay backup maintenance count exceeds audit format",
                    )
                },
            )?,
            completed_partial_bytes: counts.completed_partial_bytes,
            previous_mac,
            mac: String::new(),
        };
        record.mac = Self::backup_audit_mac(node_secret, &record)?;
        let mut encoded = serde_json::to_vec(&record).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay backup maintenance audit",
            )
        })?;
        encoded.push(b'\n');
        if encoded.len() > CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES
            || file
                .metadata()
                .ok()
                .and_then(|metadata| metadata.len().checked_add(encoded.len() as u64))
                .map(|size| size > CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES)
                .unwrap_or(true)
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit capacity exhausted",
            ));
        }
        file.write_all(&encoded).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_WRITE,
                "unable to append relay backup maintenance audit",
            )
        })?;
        file.sync_all().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_FSYNC,
                "unable to durably sync relay backup maintenance audit",
            )
        })
    }

    fn is_lower_hex(value: &str, expected_len: usize) -> bool {
        value.len() == expected_len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn is_managed_backup_file_name(name: &str) -> bool {
        let Some(stem) = name
            .strip_prefix("relay-custody-")
            .and_then(|value| value.strip_suffix(".sqlite"))
        else {
            return false;
        };

        if let Some(operation_key) = stem.strip_prefix("operation-") {
            return Self::is_lower_hex(operation_key, 32);
        }

        let Some((created_at, nonce)) = stem.rsplit_once('-') else {
            return false;
        };
        !created_at.is_empty()
            && created_at.bytes().all(|byte| byte.is_ascii_digit())
            && Self::is_lower_hex(nonce, 16)
    }

    fn is_managed_backup_temporary_name(name: &str) -> bool {
        let base = ["-journal", "-wal", "-shm"]
            .into_iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .unwrap_or(name);
        let Some(stem) = base
            .strip_prefix(".relay-custody-")
            .and_then(|value| value.strip_suffix(".tmp"))
        else {
            return false;
        };
        let Some((created_at, nonce)) = stem.rsplit_once('-') else {
            return false;
        };
        !created_at.is_empty()
            && created_at.bytes().all(|byte| byte.is_ascii_digit())
            && Self::is_lower_hex(nonce, 16)
    }

    fn inspect_private_backup_entry(
        path: PathBuf,
        file_name: String,
    ) -> ChatRelayResult<ChatRelayBackupArtifact> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        let metadata = std::fs::symlink_metadata(&path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect relay backup retention entry",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "relay backup retention entry is not a private regular file",
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup retention entry is not owner-private",
                ));
            }
        }

        Ok(ChatRelayBackupArtifact {
            path,
            file_name,
            size_bytes: metadata.len(),
            modified_at: metadata.modified().map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup retention age",
                )
            })?,
            #[cfg(unix)]
            device_id: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }

    fn inspect_verified_backup_retention(
        config: &ChatRelayConfig,
        backup_directory: &Path,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupRetentionInspection> {
        // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Scan and classify
        // the complete private namespace without deleting anything. Unknown,
        // non-private, corrupt, or racing entries fail closed rather than
        // turning an audit command into an unreliable capacity estimate.
        let mut artifacts = Vec::new();
        let mut partials = Vec::new();
        for (index, entry) in std::fs::read_dir(backup_directory)
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to read private relay backup directory",
                )
            })?
            .enumerate()
        {
            if index >= CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup directory exceeds maintenance scan limit",
                ));
            }
            let entry = entry.map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup directory entry",
                )
            })?;
            let file_name = entry.file_name().into_string().map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_MISMATCH,
                    "relay backup directory contains an unsupported entry name",
                )
            })?;
            let inspected = Self::inspect_private_backup_entry(entry.path(), file_name.clone())?;
            if Self::is_managed_backup_file_name(&file_name) {
                if inspected.size_bytes == 0 {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CORRUPT,
                        "relay backup retention artifact is empty",
                    ));
                }
                artifacts.push(inspected);
            } else if Self::is_managed_backup_temporary_name(&file_name) {
                partials.push(inspected);
            } else {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_MISMATCH,
                    "relay backup directory contains an unmanaged entry",
                ));
            }
        }

        for artifact in &artifacts {
            let verified_size = Self::verify_existing_backup_artifact(&artifact.path)?;
            let rechecked = Self::inspect_private_backup_entry(
                artifact.path.clone(),
                artifact.file_name.clone(),
            )?;
            let identity_changed = verified_size != artifact.size_bytes
                || rechecked.size_bytes != artifact.size_bytes
                || rechecked.modified_at != artifact.modified_at;
            #[cfg(unix)]
            let identity_changed = identity_changed
                || rechecked.device_id != artifact.device_id
                || rechecked.inode != artifact.inode;
            if identity_changed {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup retention artifact changed during verification",
                ));
            }
        }

        artifacts.sort_by(|left, right| {
            right
                .modified_at
                .cmp(&left.modified_at)
                .then_with(|| right.file_name.cmp(&left.file_name))
        });
        let newest_backup = artifacts.first().cloned();
        let mut retained_count = 0usize;
        let mut retained_bytes = 0u64;
        let mut excess_count = 0usize;
        let mut excess_bytes = 0u64;
        let mut excess_backups = Vec::new();
        for artifact in artifacts {
            let next_bytes = retained_bytes.checked_add(artifact.size_bytes);
            let fits = retained_count < config.custody_backup_retention_target_artifacts
                && next_bytes
                    .map(|bytes| bytes <= config.custody_backup_retention_target_bytes)
                    .unwrap_or(false);
            if retained_count == 0 || fits {
                retained_count += 1;
                retained_bytes =
                    retained_bytes
                        .checked_add(artifact.size_bytes)
                        .ok_or_else(|| {
                            Self::backup_io_error(
                                rusqlite::ffi::SQLITE_FULL,
                                "relay backup retained-byte accounting overflow",
                            )
                        })?;
            } else {
                excess_count += 1;
                excess_bytes = excess_bytes
                    .checked_add(artifact.size_bytes)
                    .ok_or_else(|| {
                        Self::backup_io_error(
                            rusqlite::ffi::SQLITE_FULL,
                            "relay backup excess-byte accounting overflow",
                        )
                    })?;
                excess_backups.push(artifact);
            }
        }

        let partial_bytes = partials.iter().try_fold(0u64, |total, partial| {
            total.checked_add(partial.size_bytes).ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup partial-byte accounting overflow",
                )
            })
        })?;

        let partial_count = partials.len();
        let partial_cutoff = UNIX_EPOCH
            .checked_add(Duration::from_secs(
                now_unix_secs.saturating_sub(config.custody_backup_partial_grace_secs),
            ))
            .ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_RANGE,
                    "relay backup partial grace cutoff is out of range",
                )
            })?;
        let stale_partials = partials
            .into_iter()
            .filter(|partial| partial.modified_at <= partial_cutoff)
            .collect();

        Ok(ChatRelayBackupRetentionInspection {
            receipt: ChatRelayBackupRetentionReceipt {
                retained_count,
                retained_bytes,
                excess_count,
                excess_bytes,
                partial_count,
                partial_bytes,
                budget_exceeded: excess_count > 0
                    || retained_count > config.custody_backup_retention_target_artifacts
                    || retained_bytes > config.custody_backup_retention_target_bytes,
            },
            newest_backup,
            excess_backups,
            stale_partials,
        })
    }

    fn inspect_active_restore_boundary(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayActiveRestoreBoundary> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        // [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] This preflight
        // uses metadata only. Opening an active WAL database read-only can
        // still create shared-memory sidecars, which would violate the
        // command's no-mutation contract.
        let active_path = Path::new(&config.db_path);
        let mut boundary = ChatRelayActiveRestoreBoundary::default();
        match std::fs::symlink_metadata(active_path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_PERM,
                        "active relay custody boundary is not a regular file",
                    ));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;

                    if metadata.permissions().mode() & 0o077 != 0 {
                        return Err(Self::backup_io_error(
                            rusqlite::ffi::SQLITE_PERM,
                            "active relay custody file is not owner-private",
                        ));
                    }
                    boundary.device_id = metadata.dev();
                    boundary.inode = metadata.ino();
                }
                boundary.present = true;
                boundary.size_bytes = metadata.len();
                boundary.modified_at = Some(metadata.modified().map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR,
                        "unable to inspect active relay custody age",
                    )
                })?);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect active relay custody boundary",
                ));
            }
        }

        for suffix in ["-journal", "-wal", "-shm"] {
            let mut sidecar = active_path.as_os_str().to_os_string();
            sidecar.push(suffix);
            match std::fs::symlink_metadata(PathBuf::from(sidecar)) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink() || !metadata.is_file() {
                        return Err(Self::backup_io_error(
                            rusqlite::ffi::SQLITE_PERM,
                            "active relay custody sidecar boundary is unsafe",
                        ));
                    }
                    boundary.sidecars_present = true;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CANTOPEN,
                        "unable to inspect active relay custody sidecar boundary",
                    ));
                }
            }
        }

        Ok(boundary)
    }

    fn restore_plan_time_components(time: SystemTime) -> ChatRelayResult<(u64, u32)> {
        let elapsed = time.duration_since(UNIX_EPOCH).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_RANGE,
                "relay restore-plan filesystem time is out of range",
            )
        })?;
        Ok((elapsed.as_secs(), elapsed.subsec_nanos()))
    }

    fn verified_restore_backup_count(
        inspection: &ChatRelayBackupRetentionInspection,
    ) -> ChatRelayResult<usize> {
        inspection
            .receipt
            .retained_count
            .checked_add(inspection.receipt.excess_count)
            .ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay restore-plan backup count overflow",
                )
            })
    }

    fn restore_plan_signing_bytes(
        config: &ChatRelayConfig,
        plan: &ChatRelayRestorePlanReceipt,
        backup: &ChatRelayBackupArtifact,
        active: &ChatRelayActiveRestoreBoundary,
    ) -> ChatRelayResult<Vec<u8>> {
        let (selected_backup_modified_secs, selected_backup_modified_nanos) =
            Self::restore_plan_time_components(backup.modified_at)?;
        let (active_database_modified_secs, active_database_modified_nanos) = active
            .modified_at
            .map(Self::restore_plan_time_components)
            .transpose()?
            .unwrap_or_default();
        let signing_state = ChatRelayRestorePlanSigningState {
            version: plan.version,
            issued_at: plan.issued_at,
            expires_at: plan.expires_at,
            verified_backup_count: plan.verified_backup_count,
            selected_backup_bytes: plan.selected_backup_bytes,
            active_database_present: plan.active_database_present,
            active_database_bytes: plan.active_database_bytes,
            nonce: &plan.nonce,
            configured_database_path: &config.db_path,
            selected_backup_name: &backup.file_name,
            selected_backup_modified_secs,
            selected_backup_modified_nanos,
            active_database_modified_secs,
            active_database_modified_nanos,
            #[cfg(unix)]
            selected_backup_device_id: backup.device_id,
            #[cfg(not(unix))]
            selected_backup_device_id: 0,
            #[cfg(unix)]
            selected_backup_inode: backup.inode,
            #[cfg(not(unix))]
            selected_backup_inode: 0,
            active_database_device_id: active.device_id,
            active_database_inode: active.inode,
        };
        bincode::serialize(&signing_state).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay restore plan",
            )
        })
    }

    fn restore_plan_mac(
        node_secret: &[u8; 32],
        config: &ChatRelayConfig,
        plan: &ChatRelayRestorePlanReceipt,
        backup: &ChatRelayBackupArtifact,
        active: &ChatRelayActiveRestoreBoundary,
    ) -> ChatRelayResult<HmacSha256> {
        let mut mac = HmacSha256::new_from_slice(node_secret).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to initialize relay restore plan",
            )
        })?;
        mac.update(CHAT_RELAY_RESTORE_PLAN_HMAC_DOMAIN);
        mac.update(&Self::restore_plan_signing_bytes(
            config, plan, backup, active,
        )?);
        Ok(mac)
    }

    fn invalid_restore_plan() -> ChatRelayError {
        Self::backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "relay restore plan is invalid, expired, or stale",
        )
    }

    fn backup_artifact_identity_matches(
        expected: &ChatRelayBackupArtifact,
        actual: &ChatRelayBackupArtifact,
    ) -> bool {
        let matches = expected.file_name == actual.file_name
            && expected.size_bytes == actual.size_bytes
            && expected.modified_at == actual.modified_at;
        #[cfg(unix)]
        let matches =
            matches && expected.device_id == actual.device_id && expected.inode == actual.inode;
        matches
    }

    fn reverify_planned_backup_artifact(
        artifact: &ChatRelayBackupArtifact,
        verify_sqlite: bool,
    ) -> ChatRelayResult<()> {
        let before =
            Self::inspect_private_backup_entry(artifact.path.clone(), artifact.file_name.clone())?;
        if !Self::backup_artifact_identity_matches(artifact, &before) {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup prune candidate changed after planning",
            ));
        }
        if verify_sqlite {
            let verified_size = Self::verify_existing_backup_artifact(&artifact.path)?;
            let after = Self::inspect_private_backup_entry(
                artifact.path.clone(),
                artifact.file_name.clone(),
            )?;
            if verified_size != artifact.size_bytes
                || !Self::backup_artifact_identity_matches(artifact, &after)
            {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup prune candidate changed during verification",
                ));
            }
        }
        Ok(())
    }

    fn checked_backup_artifact_bytes(
        artifacts: &[ChatRelayBackupArtifact],
        reason: &'static str,
    ) -> ChatRelayResult<u64> {
        artifacts.iter().try_fold(0u64, |total, artifact| {
            total
                .checked_add(artifact.size_bytes)
                .ok_or_else(|| Self::backup_io_error(rusqlite::ffi::SQLITE_FULL, reason))
        })
    }

    fn sync_backup_directory(backup_directory: &Path) -> ChatRelayResult<()> {
        File::open(backup_directory)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_FSYNC,
                    "unable to durably sync relay backup directory",
                )
            })
    }

    fn prune_verified_backup_retention_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Execution is host
        // local, explicit, and fail-closed. A dry-run may run beside the node;
        // deletion additionally requires the operator to assert it is stopped.
        if request.execute
            && (request.confirmation.as_deref() != Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION)
                || !request.node_stopped_confirmed)
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "relay backup prune confirmation is incomplete",
            ));
        }

        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, now_unix_secs)?;
        let planned_backup_count = inspection.excess_backups.len();
        let planned_backup_bytes = Self::checked_backup_artifact_bytes(
            &inspection.excess_backups,
            "relay backup prune-plan byte accounting overflow",
        )?;
        let planned_partial_count = inspection.stale_partials.len();
        let planned_partial_bytes = Self::checked_backup_artifact_bytes(
            &inspection.stale_partials,
            "relay backup partial-prune byte accounting overflow",
        )?;
        let planned_audit_counts = ChatRelayBackupMaintenanceAuditCounts {
            planned_backup_count,
            planned_backup_bytes,
            planned_partial_count,
            planned_partial_bytes,
            ..Default::default()
        };

        if !request.execute {
            Self::append_backup_maintenance_audit(
                &backup_directory,
                node_secret,
                "dry_run",
                now_unix_secs,
                planned_audit_counts,
            )?;
            return Ok(ChatRelayBackupPruneReceipt {
                executed: false,
                planned_backup_count,
                planned_backup_bytes,
                planned_partial_count,
                planned_partial_bytes,
                remaining: inspection.receipt,
                ..Default::default()
            });
        }

        Self::append_backup_maintenance_audit(
            &backup_directory,
            node_secret,
            "planned",
            now_unix_secs,
            planned_audit_counts,
        )?;

        let mut deleted_backup_count = 0usize;
        let mut deleted_backup_bytes = 0u64;
        let mut deleted_partial_count = 0usize;
        let mut deleted_partial_bytes = 0u64;
        let deletion_result = (|| -> ChatRelayResult<()> {
            for artifact in &inspection.excess_backups {
                Self::reverify_planned_backup_artifact(artifact, true)?;
                std::fs::remove_file(&artifact.path).map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR_DELETE,
                        "unable to remove verified relay backup artifact",
                    )
                })?;
                deleted_backup_count += 1;
                deleted_backup_bytes = deleted_backup_bytes
                    .checked_add(artifact.size_bytes)
                    .ok_or_else(|| {
                        Self::backup_io_error(
                            rusqlite::ffi::SQLITE_FULL,
                            "relay backup deletion byte accounting overflow",
                        )
                    })?;
            }
            for partial in &inspection.stale_partials {
                Self::reverify_planned_backup_artifact(partial, false)?;
                std::fs::remove_file(&partial.path).map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR_DELETE,
                        "unable to remove grace-expired relay backup partial",
                    )
                })?;
                deleted_partial_count += 1;
                deleted_partial_bytes = deleted_partial_bytes
                    .checked_add(partial.size_bytes)
                    .ok_or_else(|| {
                        Self::backup_io_error(
                            rusqlite::ffi::SQLITE_FULL,
                            "relay backup partial deletion byte accounting overflow",
                        )
                    })?;
            }
            Self::sync_backup_directory(&backup_directory)
        })();

        if let Err(error) = deletion_result {
            let _ = Self::append_backup_maintenance_audit(
                &backup_directory,
                node_secret,
                "failed",
                now_unix_secs,
                ChatRelayBackupMaintenanceAuditCounts {
                    completed_backup_count: deleted_backup_count,
                    completed_backup_bytes: deleted_backup_bytes,
                    completed_partial_count: deleted_partial_count,
                    completed_partial_bytes: deleted_partial_bytes,
                    ..planned_audit_counts
                },
            );
            return Err(error);
        }

        let remaining =
            match Self::inspect_verified_backup_retention(config, &backup_directory, now_unix_secs)
            {
                Ok(inspection) => inspection.receipt,
                Err(error) => {
                    let _ = Self::append_backup_maintenance_audit(
                        &backup_directory,
                        node_secret,
                        "failed",
                        now_unix_secs,
                        ChatRelayBackupMaintenanceAuditCounts {
                            completed_backup_count: deleted_backup_count,
                            completed_backup_bytes: deleted_backup_bytes,
                            completed_partial_count: deleted_partial_count,
                            completed_partial_bytes: deleted_partial_bytes,
                            ..planned_audit_counts
                        },
                    );
                    return Err(error);
                }
            };
        Self::append_backup_maintenance_audit(
            &backup_directory,
            node_secret,
            "completed",
            now_unix_secs,
            ChatRelayBackupMaintenanceAuditCounts {
                completed_backup_count: deleted_backup_count,
                completed_backup_bytes: deleted_backup_bytes,
                completed_partial_count: deleted_partial_count,
                completed_partial_bytes: deleted_partial_bytes,
                ..planned_audit_counts
            },
        )?;

        Ok(ChatRelayBackupPruneReceipt {
            executed: true,
            planned_backup_count,
            planned_backup_bytes,
            planned_partial_count,
            planned_partial_bytes,
            deleted_backup_count,
            deleted_backup_bytes,
            deleted_partial_count,
            deleted_partial_bytes,
            remaining,
        })
    }

    fn verify_existing_backup_artifact(path: &Path) -> ChatRelayResult<u64> {
        let metadata = std::fs::symlink_metadata(path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect existing relay backup artifact",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact is not a private regular file",
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "existing relay backup artifact is not owner-private",
                ));
            }
        }

        for suffix in ["-journal", "-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_os_string();
            sidecar.push(suffix);
            match std::fs::symlink_metadata(PathBuf::from(sidecar)) {
                Ok(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CORRUPT,
                        "existing relay backup artifact has mutable sidecar state",
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CANTOPEN,
                        "unable to inspect existing relay backup sidecar state",
                    ));
                }
            }
        }

        // Resolve only the already-verified private directory. macOS temp paths
        // commonly contain a system-managed ancestor symlink (`/var`), while
        // SQLite NOFOLLOW rejects any symlink component. Keeping the filename
        // separate preserves the final-component race defense.
        let parent = path.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact has no private parent",
            )
        })?;
        let file_name = path.file_name().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact has no private name",
            )
        })?;
        let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to resolve private relay backup directory",
            )
        })?;
        let nofollow_path = canonical_parent.join(file_name);

        // NOFOLLOW closes the final-component symlink race between metadata
        // inspection and SQLite open. Read-only verification must never create
        // journal/WAL sidecars next to an immutable recovery image.
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let backup_conn = Connection::open_with_flags(&nofollow_path, flags).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open existing relay backup artifact",
            )
        })?;
        Self::verify_sqlite_backup(&backup_conn)?;
        drop(backup_conn);

        let verified_metadata = std::fs::symlink_metadata(&nofollow_path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "verified relay backup artifact became unavailable",
            )
        })?;
        if verified_metadata.file_type().is_symlink()
            || !verified_metadata.is_file()
            || verified_metadata.len() != metadata.len()
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "verified relay backup artifact changed during inspection",
            ));
        }

        // In a concurrent hard-link publication, this verifier may run before
        // the creator reaches its directory fsync. Syncing here makes either
        // successful receipt sufficient to durably publish the shared name.
        Self::sync_backup_parent(&canonical_parent)?;
        Ok(verified_metadata.len())
    }

    fn create_verified_backup_artifact(
        &self,
        backup_directory: &Path,
        destination: &Path,
        reuse_existing: bool,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        match std::fs::symlink_metadata(destination) {
            Ok(_) if reuse_existing => {
                return Ok(ChatRelayBackupReceipt {
                    size_bytes: Self::verify_existing_backup_artifact(destination)?,
                    created: false,
                });
            }
            Ok(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CONSTRAINT,
                    "relay backup destination already exists",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect relay backup destination",
                ));
            }
        }

        let temporary_nonce = rand::random::<u64>();
        let temporary = backup_directory.join(format!(
            ".relay-custody-{}-{temporary_nonce:016x}.tmp",
            now_secs()
        ));
        Self::reserve_private_backup_file(&temporary)?;

        let mut destination_published = false;
        let outcome = (|| {
            let mut backup_conn = Connection::open(&temporary).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to open private relay backup file",
                )
            })?;
            Self::restrict_sqlite_file_permissions(&temporary)?;
            {
                let source = self.conn.lock();
                Self::copy_sqlite_backup(&source, &mut backup_conn)?;
            }
            Self::normalize_sqlite_backup_journal(&backup_conn)?;
            Self::verify_sqlite_backup(&backup_conn)?;
            drop(backup_conn);

            Self::restrict_sqlite_file_permissions(&temporary)?;
            std::fs::File::open(&temporary)
                .and_then(|file| file.sync_all())
                .map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR,
                        "unable to synchronize relay backup file",
                    )
                })?;

            match std::fs::hard_link(&temporary, destination) {
                Ok(()) => destination_published = true,
                Err(error)
                    if reuse_existing && error.kind() == std::io::ErrorKind::AlreadyExists =>
                {
                    Self::remove_sqlite_artifact(&temporary);
                    return Ok(ChatRelayBackupReceipt {
                        size_bytes: Self::verify_existing_backup_artifact(destination)?,
                        created: false,
                    });
                }
                Err(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CONSTRAINT,
                        "unable to publish relay backup without replacement",
                    ));
                }
            }

            Self::restrict_sqlite_file_permissions(destination)?;
            std::fs::remove_file(&temporary).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to finalize relay backup publication",
                )
            })?;
            Self::sync_backup_parent(backup_directory)?;
            let metadata = std::fs::symlink_metadata(destination).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect published relay backup artifact",
                )
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "published relay backup artifact is invalid",
                ));
            }
            Ok(ChatRelayBackupReceipt {
                size_bytes: metadata.len(),
                created: true,
            })
        })();

        Self::remove_sqlite_artifact(&temporary);
        if outcome.is_err() && destination_published {
            // Never remove a destination created by another replay. Ownership
            // begins only after this invocation's hard-link publication.
            Self::remove_sqlite_artifact(destination);
        }
        outcome
    }

    fn remove_sqlite_artifact(path: &Path) {
        let _ = std::fs::remove_file(path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_os_string();
            sidecar.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(sidecar));
        }
    }

    #[cfg(unix)]
    fn sync_backup_parent(parent: &Path) -> ChatRelayResult<()> {
        std::fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to synchronize relay backup directory",
                )
            })
    }

    #[cfg(not(unix))]
    fn sync_backup_parent(_parent: &Path) -> ChatRelayResult<()> {
        Ok(())
    }

    fn copy_sqlite_backup(
        source: &Connection,
        destination: &mut Connection,
    ) -> ChatRelayResult<()> {
        let backup = Backup::new(source, destination)?;
        let mut busy_since = None;
        loop {
            match backup.step(CHAT_RELAY_BACKUP_PAGES_PER_STEP)? {
                StepResult::Done => return Ok(()),
                StepResult::More => busy_since = None,
                StepResult::Busy | StepResult::Locked => {
                    let started = busy_since.get_or_insert_with(Instant::now);
                    if started.elapsed() >= CHAT_RELAY_BACKUP_BUSY_TIMEOUT {
                        return Err(Self::backup_io_error(
                            rusqlite::ffi::SQLITE_BUSY,
                            "relay backup remained busy",
                        ));
                    }
                    std::thread::sleep(CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY);
                }
                _ => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_ERROR,
                        "unsupported relay backup step result",
                    ));
                }
            }
        }
    }

    fn verify_sqlite_backup_logical_integrity(conn: &Connection) -> ChatRelayResult<()> {
        let installed_version = conn
            .query_row(
                "SELECT schema_version
                 FROM relay_schema_features
                 WHERE feature = ?1",
                params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if installed_version != Some(DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_schema_sentinel",
            });
        }

        let _ = Self::read_direct_peer_relay_circuit_checkpoint(conn, now_secs())?;

        let stored_usage = Self::read_storage_usage(conn)?;
        let canonical_usage = Self::read_canonical_storage_usage(conn)?;
        if stored_usage != canonical_usage {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_storage_usage",
            });
        }

        let (last_sequence, max_sequence, missing_sequences) = conn.query_row(
            "SELECT
                (SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1),
                (SELECT COALESCE(MAX(queue_sequence), 0) FROM pending_messages),
                (SELECT COUNT(*) FROM pending_messages WHERE queue_sequence IS NULL)",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )?;
        if last_sequence < 0
            || max_sequence < 0
            || missing_sequences != 0
            || last_sequence < max_sequence
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_queue_sequence",
            });
        }
        Ok(())
    }

    fn verify_sqlite_backup(conn: &Connection) -> ChatRelayResult<()> {
        Self::verify_sqlite_integrity(conn, "sqlite_backup_integrity")?;
        Self::verify_sqlite_backup_logical_integrity(conn).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_logical_integrity",
            }
        })
    }

    fn normalize_sqlite_backup_journal(conn: &Connection) -> ChatRelayResult<()> {
        // [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] SQLite may copy
        // the source WAL mode into the isolated image. A later read-only open
        // can then create `-wal`/`-shm`, violating immutable artifact replay.
        // Normalize only the offline copy; restored service startup explicitly
        // re-enables WAL + FULL before accepting custody.
        let journal_mode = conn
            .query_row("PRAGMA journal_mode=DELETE", [], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            })?;
        if !journal_mode.eq_ignore_ascii_case("delete") {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            });
        }
        Ok(())
    }

    /// Creates a new `ChatRelayService`, opening (or creating) the SQLite database.
    pub fn new(config: ChatRelayConfig, node_secret: [u8; 32]) -> ChatRelayResult<Self> {
        if let Some(parent) = std::path::Path::new(&config.db_path).parent() {
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

        let conn = Connection::open(&config.db_path)?;
        if config.db_path != ":memory:" {
            Self::restrict_sqlite_file_permissions(Path::new(&config.db_path))?;
        }
        // A short bounded wait absorbs transient locks from an operator backup
        // or diagnostic reader without allowing relay requests to hang forever.
        conn.busy_timeout(Duration::from_secs(5))?;
        Self::verify_sqlite_integrity(&conn, "sqlite_startup_integrity")?;
        let synchronous_level = Self::configure_sqlite_durability(&conn)?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let pull_cursor_key = Self::derive_pull_cursor_key(&node_secret)?;
        let mut peer_status = ChatRelayPeerStatus::new(relay_enabled);
        peer_status.custody_durability =
            ChatRelayCustodyDurabilityStatus::verified_full(synchronous_level);
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            node_secret,
            pull_cursor_key,
            dedup: MessageDedup::new(dedup_capacity),
            peer_status: RwLock::new(peer_status),
            direct_peer_retry_slo: Mutex::new(DirectPeerRetrySloWindow::default()),
            direct_peer_relay_circuit: Mutex::new(DirectPeerRelayCircuit::default()),
            maintenance_status: RwLock::new(ChatRelayMaintenanceStatus::default()),
            backup_operations: Mutex::new(()),
            // v1.3.0-Sovereign: initialise empty route cache
            wallet_routes: Arc::new(WalletRouteCache::new()),
        };

        svc.init_schema()?;
        svc.restore_direct_peer_relay_circuit(now_secs())?;
        // [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] The filesystem
        // path is operator-local state and may contain deployment identities.
        // Keep successful activation observable without publishing that path.
        info!("[CHAT_RELAY] Durable service initialized");
        Ok(svc)
    }

    // ============================================
    // Schema initialisation
    // ============================================

    fn init_schema(&self) -> ChatRelayResult<()> {
        let mut conn = self.conn.lock();
        Self::init_pending_message_schema(&mut conn)?;
        Self::init_blob_and_notification_schema(&conn)?;
        Self::init_quarantine_schema(&conn)?;
        Self::init_usage_schema(&conn)?;
        Self::init_direct_peer_circuit_checkpoint_schema(&mut conn, now_secs())?;
        Self::reconcile_storage_usage(&conn)?;
        let retained_quarantine_events =
            conn.query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
                row.get::<_, i64>(0)
            })?;
        drop(conn);
        self.maintenance_status.write().quarantine_events_retained =
            nonnegative_sqlite_counter(retained_quarantine_events);
        Ok(())
    }

    fn init_direct_peer_circuit_checkpoint_schema(
        conn: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] This singleton is
        // deliberately dimensionless. Do not add peer, route, endpoint, wallet,
        // message, request commitment, ciphertext, or payload columns.
        let table_existed = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'relay_direct_peer_circuit_checkpoint'
             )",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_schema_features (
                feature        TEXT    PRIMARY KEY,
                schema_version INTEGER NOT NULL CHECK(schema_version > 0),
                installed_at   INTEGER NOT NULL CHECK(installed_at >= 0)
            );

            CREATE TABLE IF NOT EXISTS relay_direct_peer_circuit_checkpoint (
                singleton                     INTEGER PRIMARY KEY CHECK(singleton = 1),
                schema_version                INTEGER NOT NULL CHECK(schema_version > 0),
                state                         TEXT    NOT NULL,
                successful_probes             INTEGER NOT NULL CHECK(successful_probes >= 0),
                deadline_at                   INTEGER,
                opened_total                  INTEGER NOT NULL CHECK(opened_total >= 0),
                blocked_total                 INTEGER NOT NULL CHECK(blocked_total >= 0),
                half_open_attempted_total     INTEGER NOT NULL CHECK(half_open_attempted_total >= 0),
                half_open_succeeded_total     INTEGER NOT NULL CHECK(half_open_succeeded_total >= 0),
                half_open_failed_total        INTEGER NOT NULL CHECK(half_open_failed_total >= 0),
                recovered_total               INTEGER NOT NULL CHECK(recovered_total >= 0),
                last_transition_at             INTEGER,
                checkpoint_failures_total     INTEGER NOT NULL CHECK(checkpoint_failures_total >= 0),
                last_checkpoint_failure_at    INTEGER,
                updated_at                    INTEGER NOT NULL CHECK(updated_at >= 0)
            );
            ",
        )?;
        let installed_version = tx
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if installed_version
            .is_some_and(|version| version != DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_installation_version",
            });
        }
        if !table_existed && installed_version.is_some() {
            // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] CREATE TABLE
            // above is transactional. Returning here rolls it back instead of
            // manufacturing a closed checkpoint after an installed table loss.
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_table",
            });
        }
        if table_existed {
            let row_count = tx.query_row(
                "SELECT COUNT(*) FROM relay_direct_peer_circuit_checkpoint",
                [],
                |row| row.get::<_, i64>(0),
            )?;
            if row_count != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "direct_peer_circuit_checkpoint_singleton",
                });
            }
        } else if tx.execute(
            "INSERT INTO relay_direct_peer_circuit_checkpoint (
                singleton, schema_version, state, successful_probes, deadline_at,
                opened_total, blocked_total, half_open_attempted_total,
                half_open_succeeded_total, half_open_failed_total, recovered_total,
                last_transition_at, checkpoint_failures_total,
                last_checkpoint_failure_at, updated_at
             ) VALUES (1, ?1, 'closed', 0, NULL, 0, 0, 0, 0, 0, 0, NULL, 0, NULL, ?2)",
            params![
                DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION,
                sqlite_integer(now, "direct_peer_circuit_checkpoint_init_time")?
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_singleton",
            });
        }
        if installed_version.is_none()
            && tx.execute(
                "INSERT INTO relay_schema_features (feature, schema_version, installed_at)
                 VALUES (?1, ?2, ?3)",
                params![
                    DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE,
                    DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION,
                    sqlite_integer(now, "direct_peer_circuit_schema_installed_at")?
                ],
            )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_installation_marker",
            });
        }
        tx.commit()?;
        Ok(())
    }

    fn read_direct_peer_relay_circuit_checkpoint(
        conn: &Connection,
        now: u64,
    ) -> ChatRelayResult<(DirectPeerRelayCircuit, bool)> {
        // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Startup recovery
        // and backup certification must validate exactly the same anonymous
        // checkpoint semantics. Keep this reader free of in-memory locks.
        let row = conn.query_row(
            "SELECT schema_version, state, successful_probes, deadline_at,
                    opened_total, blocked_total, half_open_attempted_total,
                    half_open_succeeded_total, half_open_failed_total,
                    recovered_total, last_transition_at,
                    checkpoint_failures_total, last_checkpoint_failure_at,
                    updated_at
             FROM relay_direct_peer_circuit_checkpoint
             WHERE singleton = 1",
            [],
            |row| {
                Ok(DirectPeerRelayCircuitCheckpointRow {
                    schema_version: row.get(0)?,
                    state: row.get(1)?,
                    successful_probes: row.get(2)?,
                    deadline_at: row.get(3)?,
                    opened_total: row.get(4)?,
                    blocked_total: row.get(5)?,
                    half_open_attempted_total: row.get(6)?,
                    half_open_succeeded_total: row.get(7)?,
                    half_open_failed_total: row.get(8)?,
                    recovered_total: row.get(9)?,
                    last_transition_at: row.get(10)?,
                    checkpoint_failures_total: row.get(11)?,
                    last_checkpoint_failure_at: row.get(12)?,
                    updated_at: row.get(13)?,
                })
            },
        )?;
        row.into_circuit(now)
    }

    fn restore_direct_peer_relay_circuit(&self, now: u64) -> ChatRelayResult<()> {
        let mut circuit = {
            let conn = self.conn.lock();
            let (mut circuit, needs_rewrite) =
                Self::read_direct_peer_relay_circuit_checkpoint(&conn, now)?;
            if needs_rewrite {
                circuit.mark_checkpoint_persisted(now);
                Self::write_direct_peer_circuit_checkpoint(&conn, &circuit, now)?;
            }
            circuit
        };
        // The loaded timestamp is process-local evidence, while persisted_at
        // identifies the durable safety state that survived the restart.
        let persisted_at = circuit.checkpoint_persisted_at;
        circuit.mark_checkpoint_loaded(now, persisted_at);
        *self.direct_peer_relay_circuit.lock() = circuit;
        Ok(())
    }

    fn write_direct_peer_circuit_checkpoint(
        conn: &Connection,
        circuit: &DirectPeerRelayCircuit,
        now: u64,
    ) -> ChatRelayResult<()> {
        let (state, successful_probes, deadline_at) = circuit.checkpoint_state();
        let updated = conn.execute(
            "UPDATE relay_direct_peer_circuit_checkpoint
             SET schema_version = ?1,
                 state = ?2,
                 successful_probes = ?3,
                 deadline_at = ?4,
                 opened_total = ?5,
                 blocked_total = ?6,
                 half_open_attempted_total = ?7,
                 half_open_succeeded_total = ?8,
                 half_open_failed_total = ?9,
                 recovered_total = ?10,
                 last_transition_at = ?11,
                 checkpoint_failures_total = ?12,
                 last_checkpoint_failure_at = ?13,
                 updated_at = ?14
             WHERE singleton = 1",
            params![
                DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION,
                state,
                i64::from(successful_probes),
                optional_sqlite_integer(deadline_at, "direct_peer_circuit_checkpoint_deadline")?,
                sqlite_integer(
                    circuit.opened_total,
                    "direct_peer_circuit_checkpoint_opened_total"
                )?,
                sqlite_integer(
                    circuit.blocked_total,
                    "direct_peer_circuit_checkpoint_blocked_total"
                )?,
                sqlite_integer(
                    circuit.half_open_attempted_total,
                    "direct_peer_circuit_checkpoint_attempted_total"
                )?,
                sqlite_integer(
                    circuit.half_open_succeeded_total,
                    "direct_peer_circuit_checkpoint_succeeded_total"
                )?,
                sqlite_integer(
                    circuit.half_open_failed_total,
                    "direct_peer_circuit_checkpoint_failed_total"
                )?,
                sqlite_integer(
                    circuit.recovered_total,
                    "direct_peer_circuit_checkpoint_recovered_total"
                )?,
                optional_sqlite_integer(
                    circuit.last_transition_at,
                    "direct_peer_circuit_checkpoint_transition"
                )?,
                sqlite_integer(
                    circuit.checkpoint_failures_total,
                    "direct_peer_circuit_checkpoint_failures_total"
                )?,
                optional_sqlite_integer(
                    circuit.last_checkpoint_failure_at,
                    "direct_peer_circuit_checkpoint_failure_time"
                )?,
                sqlite_integer(now, "direct_peer_circuit_checkpoint_updated_at")?,
            ],
        )?;
        if updated != 1 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_singleton",
            });
        }
        Ok(())
    }

    fn persist_direct_peer_circuit_transition(
        &self,
        circuit: &mut DirectPeerRelayCircuit,
        mut next: DirectPeerRelayCircuit,
        now: u64,
    ) -> bool {
        next.mark_checkpoint_persisted(now);
        let result = {
            let conn = self.conn.lock();
            Self::write_direct_peer_circuit_checkpoint(&conn, &next, now)
        };
        match result {
            Ok(()) => {
                *circuit = next;
                true
            }
            Err(error) => {
                // Log only the stable bucket. Raw SQLite details can expose an
                // operator-local path or database internals.
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Direct relay circuit checkpoint failed closed"
                );
                circuit.fail_closed_after_checkpoint_error(now);
                false
            }
        }
    }

    fn init_pending_message_schema(conn: &mut Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_messages (
                message_id   BLOB(16) PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                timestamp    INTEGER  NOT NULL,
                envelope     BLOB     NOT NULL,
                received_at  INTEGER  NOT NULL,
                status       INTEGER  NOT NULL DEFAULT 0,
                queue_sequence INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status
                ON pending_messages(receiver, status);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status_message_id
                ON pending_messages(receiver, status, message_id);
            CREATE INDEX IF NOT EXISTS idx_pm_received_at
                ON pending_messages(received_at);
            CREATE INDEX IF NOT EXISTS idx_pm_cleanup
                ON pending_messages(status, received_at, message_id);
            ",
        )?;

        // Upgrade legacy queues atomically. Existing positive unique sequence
        // values are stable across restarts; only missing/invalid/duplicate
        // values are assigned above the current maximum. No routing metadata
        // leaves SQLite during this migration.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if !Self::pending_message_column_exists(&tx, "queue_sequence")? {
            tx.execute(
                "ALTER TABLE pending_messages ADD COLUMN queue_sequence INTEGER",
                [],
            )?;
        }
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_queue_sequence (
                singleton     INTEGER PRIMARY KEY CHECK(singleton = 1),
                last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0)
            );
            INSERT OR IGNORE INTO relay_queue_sequence (singleton, last_sequence)
            VALUES (1, 0);
            ",
        )?;

        let mut seen_sequences = HashSet::new();
        let mut max_sequence = 0_i64;
        let mut rowids_to_assign = Vec::new();
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, queue_sequence
                 FROM pending_messages
                 ORDER BY rowid ASC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?))
            })?;
            for row in rows {
                let (rowid, sequence) = row?;
                match sequence {
                    Some(sequence) if sequence > 0 && seen_sequences.insert(sequence) => {
                        max_sequence = max_sequence.max(sequence);
                    }
                    _ => rowids_to_assign.push(rowid),
                }
            }
        }

        let persisted_last = tx.query_row(
            "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if persisted_last < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "relay_queue_sequence_negative",
            });
        }
        max_sequence = max_sequence.max(persisted_last);
        for rowid in rowids_to_assign {
            max_sequence = max_sequence
                .checked_add(1)
                .ok_or(ChatRelayError::QueueSequenceExhausted)?;
            if tx.execute(
                "UPDATE pending_messages SET queue_sequence = ?1 WHERE rowid = ?2",
                params![max_sequence, rowid],
            )? != 1
            {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_sequence_backfill_count",
                });
            }
        }
        tx.execute(
            "UPDATE relay_queue_sequence
             SET last_sequence = ?1
             WHERE singleton = 1",
            params![max_sequence],
        )?;
        tx.execute_batch(
            "
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_queue_sequence
                ON pending_messages(queue_sequence);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_snapshot_v2
                ON pending_messages(receiver, status, queue_sequence, timestamp);
            ",
        )?;
        tx.commit()?;
        Ok(())
    }

    fn pending_message_column_exists(
        tx: &Transaction<'_>,
        expected_column: &str,
    ) -> ChatRelayResult<bool> {
        let mut stmt = tx.prepare("PRAGMA table_info(pending_messages)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for column in columns {
            if column? == expected_column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn allocate_queue_sequence(tx: &Transaction<'_>) -> ChatRelayResult<i64> {
        let updated = tx.execute(
            "UPDATE relay_queue_sequence
             SET last_sequence = last_sequence + 1
             WHERE singleton = 1 AND last_sequence < ?1",
            params![i64::MAX],
        )?;
        if updated != 1 {
            return Err(ChatRelayError::QueueSequenceExhausted);
        }
        let sequence = tx.query_row(
            "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if sequence <= 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "relay_queue_sequence_nonpositive",
            });
        }
        Ok(sequence)
    }

    fn init_blob_and_notification_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_blobs (
                blob_id      TEXT PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                data         BLOB     NOT NULL,
                size         INTEGER  NOT NULL,
                received_at  INTEGER  NOT NULL,
                downloaded   INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_pb_received_at
                ON pending_blobs(received_at);
            CREATE INDEX IF NOT EXISTS idx_pb_receiver
                ON pending_blobs(receiver);

            CREATE TABLE IF NOT EXISTS expired_notifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sender      BLOB(32) NOT NULL,
                receiver    BLOB(32) NOT NULL,
                message_ids BLOB     NOT NULL,
                created_at  INTEGER  NOT NULL,
                pushed      INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_en_sender_pushed
                ON expired_notifications(sender, pushed);
            CREATE INDEX IF NOT EXISTS idx_en_sender_pull_order
                ON expired_notifications(sender, pushed, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_en_cleanup
                ON expired_notifications(pushed, created_at, id);
            ",
        )?;
        Ok(())
    }

    fn init_quarantine_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_quarantine_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_kind     TEXT    NOT NULL,
                reason          TEXT    NOT NULL,
                row_count       INTEGER NOT NULL CHECK(row_count > 0),
                encoded_bytes   INTEGER NOT NULL CHECK(encoded_bytes >= 0),
                quarantined_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rqe_retention
                ON relay_quarantine_events(quarantined_at, id);
            ",
        )?;
        Ok(())
    }

    fn init_usage_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_storage_usage (
                singleton              INTEGER PRIMARY KEY CHECK(singleton = 1),
                pending_message_count  INTEGER NOT NULL CHECK(pending_message_count >= 0),
                pending_message_bytes  INTEGER NOT NULL CHECK(pending_message_bytes >= 0),
                pending_blob_count     INTEGER NOT NULL CHECK(pending_blob_count >= 0),
                pending_blob_bytes     INTEGER NOT NULL CHECK(pending_blob_bytes >= 0)
            );

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_insert
            AFTER INSERT ON pending_messages
            WHEN NEW.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = pending_message_count + 1,
                    pending_message_bytes = pending_message_bytes + LENGTH(NEW.envelope)
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_delete
            AFTER DELETE ON pending_messages
            WHEN OLD.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(0, pending_message_count - 1),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes - LENGTH(OLD.envelope)
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_status
            AFTER UPDATE OF status ON pending_messages
            WHEN OLD.status != NEW.status
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(
                        0,
                        pending_message_count
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0 THEN -1
                            WHEN OLD.status != 0 AND NEW.status = 0 THEN 1
                            ELSE 0
                          END
                    ),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0
                                THEN -LENGTH(OLD.envelope)
                            WHEN OLD.status != 0 AND NEW.status = 0
                                THEN LENGTH(NEW.envelope)
                            ELSE 0
                          END
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_insert
            AFTER INSERT ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = pending_blob_count + 1,
                    pending_blob_bytes = pending_blob_bytes + NEW.size
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_delete
            AFTER DELETE ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = MAX(0, pending_blob_count - 1),
                    pending_blob_bytes = MAX(0, pending_blob_bytes - OLD.size)
                WHERE singleton = 1;
            END;
            ",
        )?;
        Ok(())
    }

    fn reconcile_storage_usage(conn: &Connection) -> ChatRelayResult<()> {
        // Reconcile from canonical rows at every startup. This makes upgrades
        // and restored databases deterministic even if an older process never
        // maintained the aggregate usage row.
        let usage = Self::read_canonical_storage_usage(conn)?;
        conn.execute(
            "INSERT OR REPLACE INTO relay_storage_usage (
                singleton,
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             )
             VALUES (1, ?1, ?2, ?3, ?4)",
            params![
                sqlite_integer(usage.pending_messages, "pending_message_count")?,
                sqlite_integer(usage.pending_message_bytes, "pending_message_bytes")?,
                sqlite_integer(usage.pending_blobs, "pending_blob_count")?,
                sqlite_integer(usage.pending_blob_bytes, "pending_blob_bytes")?,
            ],
        )?;
        Ok(())
    }

    // ============================================
    // Opaque ChatPullV2 cursor protection
    // ============================================

    fn derive_pull_cursor_key(node_secret: &[u8; 32]) -> ChatRelayResult<[u8; 32]> {
        let hkdf = hkdf::Hkdf::<Sha256>::new(Some(CHAT_PULL_CURSOR_V2_HKDF_SALT), node_secret);
        let mut key = [0_u8; 32];
        hkdf.expand(CHAT_PULL_CURSOR_V2_HKDF_INFO, &mut key)
            .map_err(|_| ChatRelayError::PullCursorEncryptionFailed)?;
        Ok(key)
    }

    fn pull_cursor_v2_aad(receiver: &[u8; 32], after_timestamp: u64) -> Vec<u8> {
        let mut aad = Vec::with_capacity(
            CHAT_PULL_CURSOR_V2_AAD_DOMAIN.len() + receiver.len() + std::mem::size_of::<u64>(),
        );
        aad.extend_from_slice(CHAT_PULL_CURSOR_V2_AAD_DOMAIN);
        aad.extend_from_slice(receiver);
        aad.extend_from_slice(&after_timestamp.to_le_bytes());
        aad
    }

    fn encode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: PullCursorV2,
    ) -> ChatRelayResult<Vec<u8>> {
        if cursor.position > cursor.ceiling {
            return Err(ChatRelayError::PullCursorEncryptionFailed);
        }

        let mut plaintext = [0_u8; CHAT_PULL_CURSOR_V2_PAYLOAD_BYTES];
        plaintext[..8].copy_from_slice(&cursor.position.to_le_bytes());
        plaintext[8..].copy_from_slice(&cursor.ceiling.to_le_bytes());

        let mut nonce_bytes = [0_u8; CHAT_PULL_CURSOR_V2_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce_bytes);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.pull_cursor_key));
        let aad = Self::pull_cursor_v2_aad(receiver, after_timestamp);
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce_bytes),
                Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::PullCursorEncryptionFailed)?;
        if ciphertext.len() != CHAT_PULL_CURSOR_V2_PAYLOAD_BYTES + CHAT_PULL_CURSOR_V2_TAG_BYTES {
            return Err(ChatRelayError::PullCursorEncryptionFailed);
        }

        let mut encoded = Vec::with_capacity(CHAT_PULL_CURSOR_V2_BYTES);
        encoded.push(CHAT_PULL_CURSOR_V2_VERSION);
        encoded.extend_from_slice(&nonce_bytes);
        encoded.extend_from_slice(&ciphertext);
        Ok(encoded)
    }

    fn decode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        if encoded.len() != CHAT_PULL_CURSOR_V2_BYTES
            || encoded.first().copied() != Some(CHAT_PULL_CURSOR_V2_VERSION)
        {
            return Err(ChatRelayError::InvalidPullCursor);
        }

        let nonce_start = 1;
        let ciphertext_start = nonce_start + CHAT_PULL_CURSOR_V2_NONCE_BYTES;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.pull_cursor_key));
        let aad = Self::pull_cursor_v2_aad(receiver, after_timestamp);
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(&encoded[nonce_start..ciphertext_start]),
                Payload {
                    msg: &encoded[ciphertext_start..],
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::InvalidPullCursor)?;
        if plaintext.len() != CHAT_PULL_CURSOR_V2_PAYLOAD_BYTES {
            return Err(ChatRelayError::InvalidPullCursor);
        }

        let mut position_bytes = [0_u8; 8];
        position_bytes.copy_from_slice(&plaintext[..8]);
        let mut ceiling_bytes = [0_u8; 8];
        ceiling_bytes.copy_from_slice(&plaintext[8..]);
        let cursor = PullCursorV2 {
            position: u64::from_le_bytes(position_bytes),
            ceiling: u64::from_le_bytes(ceiling_bytes),
        };
        if cursor.position > cursor.ceiling || cursor.ceiling > i64::MAX as u64 {
            return Err(ChatRelayError::InvalidPullCursor);
        }
        Ok(cursor)
    }

    // ============================================
    // Blob ID derivation
    // ============================================

    pub fn compute_blob_id(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        file_hash: &[u8; 32],
    ) -> String {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(sender);
        mac.update(receiver);
        mac.update(file_hash);
        let result = mac.finalize().into_bytes();
        hex::encode(&result[..16])
    }

    // ============================================
    // Online-path deduplication
    // ============================================

    /// Returns `true` if this `message_id` has already been forwarded on the
    /// online path (duplicate detection for live sessions).
    pub fn is_online_duplicate(&self, message_id: &[u8; 16]) -> bool {
        self.dedup.check_and_insert(message_id)
    }

    // ============================================
    // Message store / pull / ack
    // ============================================

    fn read_storage_usage(conn: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        let counters = conn.query_row(
            "SELECT
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             FROM relay_storage_usage
             WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?;
        Ok(ChatRelayStorageUsage {
            pending_messages: nonnegative_sqlite_value(counters.0, "pending_message_count")?,
            pending_message_bytes: nonnegative_sqlite_value(counters.1, "pending_message_bytes")?,
            pending_blobs: nonnegative_sqlite_value(counters.2, "pending_blob_count")?,
            pending_blob_bytes: nonnegative_sqlite_value(counters.3, "pending_blob_bytes")?,
        })
    }

    fn read_canonical_storage_usage(conn: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        let counters = conn.query_row(
            "SELECT
                (SELECT COUNT(*) FROM pending_messages WHERE status = 0),
                (SELECT COALESCE(SUM(LENGTH(envelope)), 0)
                   FROM pending_messages WHERE status = 0),
                (SELECT COUNT(*) FROM pending_blobs),
                (SELECT COALESCE(SUM(size), 0) FROM pending_blobs)",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?;
        Ok(ChatRelayStorageUsage {
            pending_messages: nonnegative_sqlite_value(
                counters.0,
                "canonical_pending_message_count",
            )?,
            pending_message_bytes: nonnegative_sqlite_value(
                counters.1,
                "canonical_pending_message_bytes",
            )?,
            pending_blobs: nonnegative_sqlite_value(counters.2, "canonical_pending_blob_count")?,
            pending_blob_bytes: nonnegative_sqlite_value(
                counters.3,
                "canonical_pending_blob_bytes",
            )?,
        })
    }

    fn quarantine_retention_cutoff(&self, now: i64) -> i64 {
        let ttl = i64::try_from(self.config.expired_notification_ttl_secs).unwrap_or(i64::MAX);
        now.saturating_sub(ttl)
    }

    fn insert_quarantine_events(
        tx: &Transaction<'_>,
        now: i64,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut aggregates: HashMap<(&'static str, &'static str), (u64, u64)> = HashMap::new();
        for row in rows {
            let aggregate = aggregates
                .entry((row.source_kind, row.reason))
                .or_insert((0, 0));
            aggregate.0 = aggregate.0.saturating_add(1);
            aggregate.1 = aggregate.1.saturating_add(row.encoded_bytes);
        }

        let mut stmt = tx.prepare(
            "INSERT INTO relay_quarantine_events
             (source_kind, reason, row_count, encoded_bytes, quarantined_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for ((source_kind, reason), (row_count, encoded_bytes)) in aggregates {
            stmt.execute(params![
                source_kind,
                reason,
                i64::try_from(row_count).unwrap_or(i64::MAX),
                i64::try_from(encoded_bytes).unwrap_or(i64::MAX),
                now,
            ])?;
        }
        Ok(())
    }

    fn delete_pending_rows_by_rowid(
        tx: &Transaction<'_>,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM pending_messages WHERE rowid = ?1")?;
        for row in rows {
            if stmt.execute(params![row.row_key])? != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_quarantine_delete_count",
                });
            }
        }
        Ok(())
    }

    fn delete_notification_rows_by_id(
        tx: &Transaction<'_>,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM expired_notifications WHERE id = ?1")?;
        for row in rows {
            if stmt.execute(params![row.row_key])? != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "expired_notification_quarantine_delete_count",
                });
            }
        }
        Ok(())
    }

    fn trim_quarantine_events(
        tx: &Transaction<'_>,
        retention_cutoff: i64,
    ) -> ChatRelayResult<usize> {
        let cleanup_limit = i64::try_from(CLEANUP_QUARANTINE_EVENT_BATCH_SIZE).unwrap_or(i64::MAX);
        let max_events = i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX);
        let removed_stale = tx.execute(
            "DELETE FROM relay_quarantine_events
             WHERE id IN (
                 SELECT id FROM relay_quarantine_events
                 WHERE quarantined_at < ?1
                 ORDER BY quarantined_at ASC, id ASC
                 LIMIT ?2
             )",
            params![retention_cutoff, cleanup_limit],
        )?;
        let removed_overflow = tx.execute(
            "DELETE FROM relay_quarantine_events
             WHERE id IN (
                 SELECT id FROM relay_quarantine_events
                 ORDER BY quarantined_at DESC, id DESC
                 LIMIT ?1 OFFSET ?2
             )",
            params![cleanup_limit, max_events],
        )?;
        Ok(removed_stale.saturating_add(removed_overflow))
    }

    fn quarantine_event_count(tx: &Transaction<'_>) -> ChatRelayResult<usize> {
        let count = tx.query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(usize::try_from(count.max(0)).unwrap_or(usize::MAX))
    }

    fn record_pull_quarantine(
        &self,
        now: u64,
        pending_messages: usize,
        expired_notifications: usize,
        removed_events: usize,
        retained_events: usize,
    ) {
        let mut status = self.maintenance_status.write();
        status.quarantined_pending_messages_total = status
            .quarantined_pending_messages_total
            .saturating_add(u64::try_from(pending_messages).unwrap_or(u64::MAX));
        status.quarantined_expired_notifications_total = status
            .quarantined_expired_notifications_total
            .saturating_add(u64::try_from(expired_notifications).unwrap_or(u64::MAX));
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(removed_events).unwrap_or(u64::MAX));
        status.quarantine_events_retained = u64::try_from(retained_events).unwrap_or(u64::MAX);
        status.last_quarantine_at = Some(now);
    }

    /// Stores a pending offline message for a receiver that is not currently online.
    ///
    /// # Errors
    ///
    /// Returns an item-size or durable-capacity error before insertion, or a
    /// serialization/SQLite error if encoding or the atomic write fails.
    pub fn store_pending(&self, envelope: &ChatEnvelope) -> ChatRelayResult<()> {
        if envelope.ciphertext.len() > self.config.max_message_size {
            return Err(ChatRelayError::MessageTooLarge {
                size: envelope.ciphertext.len(),
                limit: self.config.max_message_size,
            });
        }

        let now = now_secs();
        let received_at = i64::try_from(now).unwrap_or(i64::MAX);
        let envelope_timestamp =
            i64::try_from(envelope.timestamp).map_err(|_| ChatRelayError::TimestampOutOfRange)?;
        let receiver = envelope.receiver;
        let envelope_bytes = encode_envelope(envelope)?;
        let incoming_bytes = u64::try_from(envelope_bytes.len()).unwrap_or(u64::MAX);

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Idempotence is checked before every quota. A retry of an already
        // durable message must succeed even while the queue is at capacity.
        let existing_envelope = tx
            .query_row(
                "SELECT envelope FROM pending_messages WHERE message_id = ?1",
                params![envelope.message_id.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if let Some(existing_envelope) = existing_envelope {
            if existing_envelope == envelope_bytes {
                tx.commit()?;
                return Ok(());
            }
            return Err(ChatRelayError::MessageIdConflict);
        }

        let usage = Self::read_storage_usage(&tx)?;
        if usage.pending_messages
            >= u64::try_from(self.config.max_pending_messages_total).unwrap_or(u64::MAX)
        {
            return Err(ChatRelayError::PendingMessageQueueFull {
                current: usize::try_from(usage.pending_messages).unwrap_or(usize::MAX),
                limit: self.config.max_pending_messages_total,
            });
        }
        if usage.pending_message_bytes.saturating_add(incoming_bytes)
            > self.config.max_pending_message_bytes_total
        {
            return Err(ChatRelayError::PendingMessageBytesExceeded {
                current: usage.pending_message_bytes,
                incoming: incoming_bytes,
                limit: self.config.max_pending_message_bytes_total,
            });
        }

        let count = tx.query_row(
            "SELECT COUNT(*) FROM pending_messages WHERE receiver = ? AND status = 0",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let count = usize::try_from(count.max(0)).unwrap_or(usize::MAX);

        if count >= self.config.max_pending_per_wallet {
            return Err(ChatRelayError::MailboxFull {
                current: count,
                limit: self.config.max_pending_per_wallet,
            });
        }

        // Allocate only after idempotence and all quotas pass. The sequence
        // update and row insert share this transaction, so failed inserts do
        // not consume observable ordering state.
        let queue_sequence = Self::allocate_queue_sequence(&tx)?;
        tx.execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status,
              queue_sequence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7)",
            params![
                envelope.message_id.as_slice(),
                envelope.sender.as_slice(),
                envelope.receiver.as_slice(),
                envelope_timestamp,
                envelope_bytes,
                received_at,
                queue_sequence,
            ],
        )?;
        tx.commit()?;
        drop(conn);

        debug!(
            encoded_bytes = incoming_bytes,
            "[CHAT_RELAY] Message stored pending"
        );
        Ok(())
    }

    fn validate_pending_message_row(
        row: StoredPendingMessageRow,
        expected_receiver: &[u8; 32],
    ) -> Result<PendingMessage, CorruptDurableRow> {
        let encoded_bytes = u64::try_from(row.envelope.len()).unwrap_or(u64::MAX);
        let corrupt = |reason| CorruptDurableRow {
            row_key: row.rowid,
            source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
            reason,
            encoded_bytes,
        };
        let message_id: [u8; 16] = row
            .message_id
            .try_into()
            .map_err(|_| corrupt("pending_message_id"))?;
        let stored_sender: [u8; 32] = row
            .sender
            .try_into()
            .map_err(|_| corrupt("pending_message_sender"))?;
        let stored_receiver: [u8; 32] = row
            .receiver
            .try_into()
            .map_err(|_| corrupt("pending_message_receiver"))?;
        let stored_timestamp =
            u64::try_from(row.timestamp).map_err(|_| corrupt("pending_message_timestamp"))?;
        if stored_receiver != *expected_receiver {
            return Err(corrupt("pending_message_receiver_mismatch"));
        }
        let envelope =
            decode_envelope(&row.envelope).map_err(|_| corrupt("pending_message_envelope"))?;
        if envelope.message_id != message_id {
            return Err(corrupt("pending_message_id_mismatch"));
        }
        if envelope.receiver != *expected_receiver {
            return Err(corrupt("pending_message_envelope_receiver_mismatch"));
        }
        if envelope.sender != stored_sender {
            return Err(corrupt("pending_message_sender_mismatch"));
        }
        if envelope.timestamp != stored_timestamp {
            return Err(corrupt("pending_message_timestamp_mismatch"));
        }
        envelope
            .verify_signature()
            .map_err(|_| corrupt("pending_message_signature"))?;
        Ok(PendingMessage {
            message_id,
            envelope,
        })
    }

    fn quarantine_pending_pull_rows(
        &self,
        conn: &mut Connection,
        corrupt_rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        if corrupt_rows.is_empty() {
            return Ok(());
        }

        let quarantine_now = now_secs();
        let quarantine_now_i64 = i64::try_from(quarantine_now).unwrap_or(i64::MAX);
        let retention_cutoff = self.quarantine_retention_cutoff(quarantine_now_i64);
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::insert_quarantine_events(&tx, quarantine_now_i64, corrupt_rows)?;
        Self::delete_pending_rows_by_rowid(&tx, corrupt_rows)?;
        let removed_events = Self::trim_quarantine_events(&tx, retention_cutoff)?;
        let retained_events = Self::quarantine_event_count(&tx)?;
        tx.commit()?;

        self.record_pull_quarantine(
            quarantine_now,
            corrupt_rows.len(),
            0,
            removed_events,
            retained_events,
        );
        warn!(
            quarantined_pending_messages = corrupt_rows.len(),
            "[CHAT_RELAY] Corrupt pending rows isolated during pull"
        );
        Ok(())
    }

    /// Retrieves a page of pending messages for the given receiver wallet.
    ///
    /// The v1 wire cursor contains only `message_id`, so rows must be ordered
    /// by that same key. Ordering by timestamp first can permanently skip a
    /// later row whose random ID sorts below the previous page's cursor.
    ///
    /// # Errors
    ///
    /// Corrupt rows are atomically replaced by de-identified quarantine events
    /// so one poison row cannot permanently block a receiver's mailbox.
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<(Vec<PendingMessage>, bool)> {
        let page_limit = usize::try_from(limit.clamp(1, 100)).unwrap_or(100);
        let effective_limit = page_limit.saturating_add(1);
        let query_after_timestamp = i64::try_from(after_timestamp).unwrap_or(i64::MAX);
        let query_limit = i64::try_from(effective_limit).unwrap_or(i64::MAX);

        let mut conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT rowid, message_id, sender, receiver, timestamp, envelope
             FROM pending_messages
             WHERE receiver = ?1
               AND status = 0
               AND timestamp > ?2
               AND message_id > ?3
             ORDER BY message_id ASC
             LIMIT ?4",
        )?;

        let rows: Vec<StoredPendingMessageRow> = stmt
            .query_map(
                params![
                    receiver.as_slice(),
                    query_after_timestamp,
                    cursor.as_slice(),
                    query_limit,
                ],
                |row| {
                    Ok(StoredPendingMessageRow {
                        rowid: row.get(0)?,
                        message_id: row.get(1)?,
                        sender: row.get(2)?,
                        receiver: row.get(3)?,
                        timestamp: row.get(4)?,
                        envelope: row.get(5)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);
        let raw_has_more = rows.len() == effective_limit;
        let mut messages = Vec::with_capacity(rows.len().min(page_limit));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match Self::validate_pending_message_row(row, receiver) {
                Ok(message) => messages.push(message),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        self.quarantine_pending_pull_rows(&mut conn, &corrupt_rows)?;
        drop(conn);

        let has_more = raw_has_more || messages.len() > page_limit;
        messages.truncate(page_limit);
        Ok((messages, has_more))
    }

    /// Retrieves one stable monotonic snapshot page for ChatPullV2.
    ///
    /// An empty cursor captures the current receiver-specific sequence ceiling.
    /// Later inserts receive larger sequences and cannot move into that snapshot,
    /// preventing duplicate/skip behavior while the client paginates. The
    /// sequence and ceiling remain node-internal inside an AEAD-protected cursor.
    ///
    /// # Errors
    ///
    /// Returns [`ChatRelayError::InvalidPullCursor`] for tampered, cross-wallet,
    /// cross-filter, malformed, or foreign-node cursors. Corrupt durable rows are
    /// atomically quarantined using the same path as v1 pulls.
    pub fn pull_pending_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
    ) -> ChatRelayResult<PendingMessagePageV2> {
        let page_limit = usize::try_from(limit.clamp(1, 100)).unwrap_or(100);
        let effective_limit = page_limit.saturating_add(1);
        let query_after_timestamp = i64::try_from(after_timestamp).unwrap_or(i64::MAX);
        let query_limit = i64::try_from(effective_limit).unwrap_or(i64::MAX);

        let mut conn = self.conn.lock();
        let cursor = if encoded_cursor.is_empty() {
            let ceiling = conn.query_row(
                "SELECT COALESCE(MAX(queue_sequence), 0)
                 FROM pending_messages
                 WHERE receiver = ?1
                   AND status = 0
                   AND timestamp > ?2
                   AND queue_sequence > 0",
                params![receiver.as_slice(), query_after_timestamp],
                |row| row.get::<_, i64>(0),
            )?;
            if ceiling < 0 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_snapshot_ceiling",
                });
            }
            PullCursorV2 {
                position: 0,
                ceiling: u64::try_from(ceiling).unwrap_or(0),
            }
        } else {
            self.decode_pull_cursor_v2(receiver, after_timestamp, encoded_cursor)?
        };

        let query_position =
            i64::try_from(cursor.position).map_err(|_| ChatRelayError::InvalidPullCursor)?;
        let query_ceiling =
            i64::try_from(cursor.ceiling).map_err(|_| ChatRelayError::InvalidPullCursor)?;
        let mut stmt = conn.prepare(
            "SELECT queue_sequence, rowid, message_id, sender, receiver, timestamp, envelope
             FROM pending_messages
             WHERE receiver = ?1
               AND status = 0
               AND timestamp > ?2
               AND queue_sequence > ?3
               AND queue_sequence <= ?4
             ORDER BY queue_sequence ASC
             LIMIT ?5",
        )?;
        let rows: Vec<StoredSequencedPendingMessageRow> = stmt
            .query_map(
                params![
                    receiver.as_slice(),
                    query_after_timestamp,
                    query_position,
                    query_ceiling,
                    query_limit,
                ],
                |row| {
                    Ok(StoredSequencedPendingMessageRow {
                        queue_sequence: row.get(0)?,
                        row: StoredPendingMessageRow {
                            rowid: row.get(1)?,
                            message_id: row.get(2)?,
                            sender: row.get(3)?,
                            receiver: row.get(4)?,
                            timestamp: row.get(5)?,
                            envelope: row.get(6)?,
                        },
                    })
                },
            )?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);

        let raw_has_more = rows.len() == effective_limit;
        let raw_max_sequence = rows
            .last()
            .and_then(|row| u64::try_from(row.queue_sequence).ok());
        let mut valid_messages = Vec::with_capacity(rows.len().min(page_limit));
        let mut corrupt_rows = Vec::new();
        for stored in rows {
            let sequence = match u64::try_from(stored.queue_sequence) {
                Ok(sequence) if sequence > 0 => sequence,
                _ => {
                    corrupt_rows.push(CorruptDurableRow {
                        row_key: stored.row.rowid,
                        source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
                        reason: "pending_message_queue_sequence",
                        encoded_bytes: u64::try_from(stored.row.envelope.len()).unwrap_or(u64::MAX),
                    });
                    continue;
                }
            };
            match Self::validate_pending_message_row(stored.row, receiver) {
                Ok(message) => valid_messages.push((sequence, message)),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        self.quarantine_pending_pull_rows(&mut conn, &corrupt_rows)?;
        drop(conn);

        let valid_overflow = valid_messages.len() > page_limit;
        let has_more = raw_has_more || valid_overflow;
        let next_position = if valid_overflow {
            valid_messages
                .get(page_limit.saturating_sub(1))
                .map(|(sequence, _)| *sequence)
                .unwrap_or(cursor.position)
        } else if has_more {
            raw_max_sequence.unwrap_or(cursor.position)
        } else {
            cursor.ceiling
        };
        valid_messages.truncate(page_limit);
        let messages = valid_messages
            .into_iter()
            .map(|(_, message)| message)
            .collect();
        let next_cursor = self.encode_pull_cursor_v2(
            receiver,
            after_timestamp,
            PullCursorV2 {
                position: next_position,
                ceiling: cursor.ceiling,
            },
        )?;

        Ok(PendingMessagePageV2 {
            messages,
            next_cursor,
            has_more,
        })
    }

    /// Acknowledges delivery of a batch of messages, deleting them from the store.
    ///
    /// Only deletes rows where `receiver = receiver_wallet`.
    ///
    /// # Errors
    ///
    /// Returns an oversized-batch or `SQLite` error. The transaction is atomic.
    pub fn ack_messages(
        &self,
        message_ids: &[[u8; 16]],
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        if message_ids.is_empty() {
            return Ok(0);
        }
        if message_ids.len() > MAX_CHAT_ACK_MESSAGE_IDS {
            return Err(ChatRelayError::AckBatchTooLarge {
                size: message_ids.len(),
                limit: MAX_CHAT_ACK_MESSAGE_IDS,
            });
        }

        let unique_ids: HashSet<[u8; 16]> = message_ids.iter().copied().collect();
        let deleted =
            Self::ack_messages_transaction(&mut self.conn.lock(), &unique_ids, receiver_wallet)?;

        debug!(count = deleted, "[CHAT_RELAY] Messages ACKed and deleted");
        Ok(deleted)
    }

    fn ack_messages_transaction(
        conn: &mut Connection,
        unique_ids: &HashSet<[u8; 16]>,
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut deleted = 0usize;

        for mid in unique_ids {
            let n = tx.execute(
                "DELETE FROM pending_messages
                 WHERE message_id = ?1 AND receiver = ?2",
                params![mid.as_slice(), receiver_wallet.as_slice()],
            )?;
            deleted += n;
        }
        tx.commit()?;
        Ok(deleted)
    }

    // ============================================
    // Blob store / get / delete
    // ============================================

    /// Stores one opaque encrypted blob under node-wide and receiver quotas.
    ///
    /// # Errors
    ///
    /// Returns an item-size, capacity, serialization, or `SQLite` error.
    pub fn put_blob(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        data: &[u8],
        file_hash: &[u8; 32],
    ) -> ChatRelayResult<String> {
        if data.len() > self.config.max_blob_size {
            return Err(ChatRelayError::BlobTooLarge {
                size: data.len(),
                limit: self.config.max_blob_size,
            });
        }

        let blob_id = self.compute_blob_id(sender, receiver, file_hash);
        let incoming_bytes = u64::try_from(data.len()).unwrap_or(u64::MAX);
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Return the stable content-derived ID before quota checks when the
        // encrypted object is already present. Retries remain idempotent even
        // while the blob store is full.
        let duplicate = tx
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?1",
                params![&blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if duplicate {
            tx.commit()?;
            return Ok(blob_id);
        }

        let usage = Self::read_storage_usage(&tx)?;
        if usage.pending_blobs
            >= u64::try_from(self.config.max_pending_blobs_total).unwrap_or(u64::MAX)
        {
            return Err(ChatRelayError::PendingBlobStoreFull {
                current: usize::try_from(usage.pending_blobs).unwrap_or(usize::MAX),
                limit: self.config.max_pending_blobs_total,
            });
        }
        if usage.pending_blob_bytes.saturating_add(incoming_bytes)
            > self.config.max_pending_blob_bytes_total
        {
            return Err(ChatRelayError::PendingBlobBytesExceeded {
                current: usage.pending_blob_bytes,
                incoming: incoming_bytes,
                limit: self.config.max_pending_blob_bytes_total,
            });
        }

        let count = tx.query_row(
            "SELECT COUNT(*) FROM pending_blobs WHERE receiver = ?",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let count = usize::try_from(count.max(0)).unwrap_or(usize::MAX);

        if count >= self.config.max_blobs_per_receiver {
            return Err(ChatRelayError::BlobQuotaExceeded {
                current: count,
                limit: self.config.max_blobs_per_receiver,
            });
        }

        let now = now_secs();
        let received_at = i64::try_from(now).unwrap_or(i64::MAX);
        let stored_size = i64::try_from(data.len()).unwrap_or(i64::MAX);

        tx.execute(
            "INSERT OR IGNORE INTO pending_blobs
             (blob_id, sender, receiver, data, size, received_at, downloaded)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![
                &blob_id,
                sender.as_slice(),
                receiver.as_slice(),
                data,
                stored_size,
                received_at,
            ],
        )?;
        tx.commit()?;
        drop(conn);

        info!(size = data.len(), "[CHAT_RELAY] Encrypted blob stored");
        Ok(blob_id)
    }

    /// Retrieves an opaque encrypted blob by its HMAC-derived identifier.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error or [`ChatRelayError::BlobNotFound`].
    pub fn get_blob(&self, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        let conn = self.conn.lock();

        let data: Option<Vec<u8>> = conn
            .query_row(
                "SELECT data FROM pending_blobs WHERE blob_id = ?",
                params![blob_id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;

        match data {
            None => {
                drop(conn);
                Err(ChatRelayError::BlobNotFound {
                    blob_id: blob_id.to_string(),
                })
            }
            Some(bytes) => {
                let _ = conn.execute(
                    "UPDATE pending_blobs SET downloaded = 1 WHERE blob_id = ?",
                    params![blob_id],
                );
                drop(conn);
                debug!(size = bytes.len(), "[CHAT_RELAY] Encrypted blob retrieved");
                Ok(bytes)
            }
        }
    }

    /// Deletes an encrypted blob when requested by its original sender.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite`, not-found, or authorization error.
    pub fn delete_blob(&self, blob_id: &str, requester: &[u8; 32]) -> ChatRelayResult<()> {
        let conn = self.conn.lock();

        let deleted = conn.execute(
            "DELETE FROM pending_blobs WHERE blob_id = ?1 AND sender = ?2",
            params![blob_id, requester.as_slice()],
        )?;

        if deleted == 1 {
            drop(conn);
            info!("[CHAT_RELAY] Encrypted blob deleted by authorized sender");
            return Ok(());
        }

        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?",
                params![blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        drop(conn);

        if exists {
            Err(ChatRelayError::Unauthorized)
        } else {
            Err(ChatRelayError::BlobNotFound {
                blob_id: blob_id.to_string(),
            })
        }
    }

    // ============================================
    // Expired notifications
    // ============================================

    fn validate_expired_notification_row(
        row: StoredExpiredNotificationRow,
        expected_sender: &[u8; 32],
    ) -> Result<ExpiredNotification, CorruptDurableRow> {
        let encoded_bytes = u64::try_from(row.message_ids_raw.len()).unwrap_or(u64::MAX);
        let corrupt = |reason| CorruptDurableRow {
            row_key: row.id,
            source_kind: QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
            reason,
            encoded_bytes,
        };
        if row.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
            return Err(corrupt("expired_notification_payload_size"));
        }
        let stored_sender: [u8; 32] = row
            .sender
            .try_into()
            .map_err(|_| corrupt("expired_notification_sender"))?;
        if stored_sender != *expected_sender {
            return Err(corrupt("expired_notification_sender_mismatch"));
        }
        let receiver: [u8; 32] = row
            .receiver
            .try_into()
            .map_err(|_| corrupt("expired_notification_receiver"))?;
        let notification = ExpiredNotification {
            id: row.id,
            sender: stored_sender,
            receiver,
            message_ids_raw: row.message_ids_raw,
        };
        notification
            .message_ids()
            .map_err(|_| corrupt("expired_notification_message_ids"))?;
        Ok(notification)
    }

    /// Retrieves one bounded page of expiry notifications for a sender.
    ///
    /// The extra row is used only to compute `has_more`; it is never returned.
    /// Invalid durable rows are atomically replaced by de-identified quarantine
    /// evidence so one poison row cannot permanently block sender control flow.
    ///
    /// # Errors
    ///
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<(Vec<ExpiredNotification>, bool)> {
        let effective_limit = MAX_EXPIRED_NOTIFICATIONS_PER_PULL + 1;
        let query_limit = i64::try_from(effective_limit).unwrap_or(i64::MAX);
        let mut conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, sender, receiver, message_ids
             FROM expired_notifications
             WHERE sender = ?1 AND pushed = 0
             ORDER BY created_at ASC, id ASC
             LIMIT ?2",
        )?;

        let rows: Vec<StoredExpiredNotificationRow> = stmt
            .query_map(params![sender.as_slice(), query_limit], |row| {
                Ok(StoredExpiredNotificationRow {
                    id: row.get(0)?,
                    sender: row.get(1)?,
                    receiver: row.get(2)?,
                    message_ids_raw: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);

        let raw_has_more = rows.len() == effective_limit;
        let mut notifications =
            Vec::with_capacity(rows.len().min(MAX_EXPIRED_NOTIFICATIONS_PER_PULL));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match Self::validate_expired_notification_row(row, sender) {
                Ok(notification) => notifications.push(notification),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        if corrupt_rows.is_empty() {
            drop(conn);
        } else {
            let quarantine_now = now_secs();
            let quarantine_now_i64 = i64::try_from(quarantine_now).unwrap_or(i64::MAX);
            let retention_cutoff = self.quarantine_retention_cutoff(quarantine_now_i64);
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            Self::insert_quarantine_events(&tx, quarantine_now_i64, &corrupt_rows)?;
            Self::delete_notification_rows_by_id(&tx, &corrupt_rows)?;
            let removed_events = Self::trim_quarantine_events(&tx, retention_cutoff)?;
            let retained_events = Self::quarantine_event_count(&tx)?;
            tx.commit()?;
            drop(conn);

            self.record_pull_quarantine(
                quarantine_now,
                0,
                corrupt_rows.len(),
                removed_events,
                retained_events,
            );
            warn!(
                quarantined_expired_notifications = corrupt_rows.len(),
                "[CHAT_RELAY] Corrupt expiry notifications isolated during pull"
            );
        }

        let has_more = raw_has_more || notifications.len() > MAX_EXPIRED_NOTIFICATIONS_PER_PULL;
        notifications.truncate(MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
        Ok((notifications, has_more))
    }

    /// Compatibility wrapper for callers that do not consume pagination yet.
    ///
    /// New runtime code should use [`Self::pull_pending_notifications`] and
    /// propagate its `has_more` flag.
    ///
    /// # Errors
    ///
    /// Returns a storage, decoding, or durable-data integrity error.
    pub fn get_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<Vec<ExpiredNotification>> {
        self.pull_pending_notifications(sender)
            .map(|(notifications, _)| notifications)
    }

    /// Atomically marks a successfully written notification page as pushed.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error and rolls back the whole page on failure.
    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let unique_ids: HashSet<i64> = ids.iter().copied().collect();
        Self::mark_notifications_pushed_transaction(&mut self.conn.lock(), &unique_ids)
    }

    fn mark_notifications_pushed_transaction(
        conn: &mut Connection,
        unique_ids: &HashSet<i64>,
    ) -> ChatRelayResult<()> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for id in unique_ids {
            tx.execute(
                "UPDATE expired_notifications SET pushed = 1 WHERE id = ?",
                params![id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    // ============================================
    // TTL cleanup
    // ============================================

    /// Runs one TTL cleanup cycle (synchronous — call from `spawn_blocking`).
    ///
    /// Mutations run in a bounded sequence of `SQLite` IMMEDIATE transactions.
    /// Each committed batch releases the connection before the next begins.
    /// Returns `(expired_messages, expired_blobs)`.
    ///
    /// # Errors
    ///
    /// Returns a storage, serialization, or durable-data integrity error. A
    /// failed batch is rolled back and counted in maintenance evidence. Earlier
    /// committed batches remain durable and are included in aggregate counters.
    pub fn run_cleanup(&self) -> ChatRelayResult<(usize, usize)> {
        self.run_cleanup_with_batch_budget(CLEANUP_MAX_BATCHES_PER_RUN)
    }

    fn run_cleanup_with_batch_budget(&self, max_batches: usize) -> ChatRelayResult<(usize, usize)> {
        let now = now_secs();
        let cleanup_now = i64::try_from(now).unwrap_or(i64::MAX);
        let (summary, failure) = self.run_cleanup_at(cleanup_now, max_batches.max(1));

        self.record_cleanup_run(now, summary, failure.as_ref());

        let Some(error) = failure else {
            return Ok((summary.expired_messages, summary.expired_blobs));
        };
        Err(error)
    }

    fn record_cleanup_run(
        &self,
        now: u64,
        summary: CleanupRunSummary,
        failure: Option<&ChatRelayError>,
    ) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_batches_total = status
            .cleanup_batches_total
            .saturating_add(u64::try_from(summary.successful_batches).unwrap_or(u64::MAX));
        if summary.backlog_deferred {
            status.cleanup_backlog_deferred_total =
                status.cleanup_backlog_deferred_total.saturating_add(1);
        }
        status.expired_messages_total = status
            .expired_messages_total
            .saturating_add(u64::try_from(summary.expired_messages).unwrap_or(u64::MAX));
        status.expired_blobs_total = status
            .expired_blobs_total
            .saturating_add(u64::try_from(summary.expired_blobs).unwrap_or(u64::MAX));
        status.expired_notifications_removed_total = status
            .expired_notifications_removed_total
            .saturating_add(u64::try_from(summary.removed_notifications).unwrap_or(u64::MAX));
        status.quarantined_pending_messages_total =
            status.quarantined_pending_messages_total.saturating_add(
                u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX),
            );
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(summary.removed_quarantine_events).unwrap_or(u64::MAX));
        if summary.successful_batches > 0 {
            status.quarantine_events_retained =
                u64::try_from(summary.retained_quarantine_events).unwrap_or(u64::MAX);
        }
        if summary.quarantined_pending_messages > 0 {
            status.last_quarantine_at = Some(now);
        }
        status.last_cleanup_at = Some(now);
        status.last_cleanup_batches = u64::try_from(summary.successful_batches).unwrap_or(u64::MAX);
        status.last_cleanup_backlog_deferred = summary.backlog_deferred;
        status.last_cleanup_quarantined_pending_messages =
            u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX);
        match failure {
            None => {
                status.last_cleanup_status = Some("succeeded".to_string());
                status.last_cleanup_failure_reason = None;
            }
            Some(error) => {
                status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
                status.last_cleanup_status = Some("failed".to_string());
                status.last_cleanup_failure_reason = Some(error.reason_bucket().to_string());
            }
        }
    }

    fn run_cleanup_at(
        &self,
        now: i64,
        max_batches: usize,
    ) -> (CleanupRunSummary, Option<ChatRelayError>) {
        // Configuration validation rejects values above i64::MAX. These
        // fallbacks keep direct service construction fail-closed as well:
        // an out-of-range TTL retains data instead of expiring fresh rows.
        let ttl = i64::try_from(self.config.offline_ttl_secs).unwrap_or(i64::MAX);
        let notif_ttl =
            i64::try_from(self.config.expired_notification_ttl_secs).unwrap_or(i64::MAX);
        let cutoff = now.saturating_sub(ttl);
        let notif_cutoff = now.saturating_sub(notif_ttl);

        let mut summary = CleanupRunSummary::default();
        let mut failure = None;
        for batch_index in 0..max_batches {
            let batch_result = {
                let mut conn = self.conn.lock();
                Self::run_cleanup_transaction(&mut conn, now, cutoff, notif_cutoff)
            };
            match batch_result {
                Ok(batch) => {
                    let has_more = batch.has_more;
                    summary.absorb(batch);
                    if !has_more {
                        break;
                    }
                    if batch_index + 1 == max_batches {
                        summary.backlog_deferred = true;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        if summary.removed_anything() || summary.backlog_deferred {
            info!(
                expired_messages = summary.expired_messages,
                expired_blobs = summary.expired_blobs,
                removed_notifications = summary.removed_notifications,
                quarantined_pending_messages = summary.quarantined_pending_messages,
                removed_quarantine_events = summary.removed_quarantine_events,
                retained_quarantine_events = summary.retained_quarantine_events,
                committed_batches = summary.successful_batches,
                backlog_deferred = summary.backlog_deferred,
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Bounded cleanup run complete"
            );
        } else {
            debug!(
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Cleanup: nothing to expire"
            );
        }
        if summary.quarantined_pending_messages > 0 {
            warn!(
                quarantined_pending_messages = summary.quarantined_pending_messages,
                "[CHAT_RELAY] Corrupt pending rows isolated during cleanup"
            );
        }

        (summary, failure)
    }

    fn run_cleanup_transaction(
        conn: &mut Connection,
        now: i64,
        cutoff: i64,
        notif_cutoff: i64,
    ) -> ChatRelayResult<CleanupBatchOutcome> {
        let message_limit = i64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(i64::MAX);
        let blob_limit = i64::try_from(CLEANUP_BLOB_BATCH_SIZE).unwrap_or(i64::MAX);
        let notification_limit = i64::try_from(CLEANUP_NOTIFICATION_BATCH_SIZE).unwrap_or(i64::MAX);
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let transaction_result: ChatRelayResult<CleanupBatchOutcome> = (|| {
            let expired_batch = Self::load_expired_message_batch(&tx, cutoff, message_limit)?;
            let expired_message_count = expired_batch.valid_rows.len();
            let quarantined_pending_messages = expired_batch.corrupt_rows.len();
            Self::queue_expiry_notifications(&tx, now, &expired_batch.valid_rows)?;
            Self::insert_quarantine_events(&tx, now, &expired_batch.corrupt_rows)?;
            Self::delete_expired_message_batch(&tx, &expired_batch.selected_rowids)?;
            let expired_blobs = Self::delete_expired_blob_batch(&tx, cutoff, blob_limit)?;
            let removed_notifications =
                Self::delete_stale_notification_batch(&tx, notif_cutoff, notification_limit)?;
            let removed_quarantine_events = Self::trim_quarantine_events(&tx, notif_cutoff)?;
            let retained_quarantine_events = Self::quarantine_event_count(&tx)?;
            let has_more = Self::cleanup_backlog_exists(&tx, cutoff, notif_cutoff)?;

            Ok(CleanupBatchOutcome {
                expired_messages: expired_message_count,
                expired_blobs,
                removed_notifications,
                quarantined_pending_messages,
                removed_quarantine_events,
                retained_quarantine_events,
                has_more,
            })
        })();

        match transaction_result {
            Ok(counts) => {
                tx.commit()?;
                Ok(counts)
            }
            Err(error) => Err(error),
        }
    }

    fn load_expired_message_batch(
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<ValidatedExpiredMessageBatch> {
        let mut stmt = tx.prepare(
            "SELECT rowid, message_id, sender, receiver, timestamp, envelope, queue_sequence
             FROM pending_messages
             WHERE status = 0 AND received_at < ?1
             ORDER BY received_at ASC, message_id ASC
             LIMIT ?2",
        )?;
        let stored_rows: Vec<StoredExpiredMessageRow> = stmt
            .query_map(params![cutoff, limit], |row| {
                Ok(StoredExpiredMessageRow {
                    rowid: row.get(0)?,
                    message_id: row.get(1)?,
                    sender: row.get(2)?,
                    receiver: row.get(3)?,
                    timestamp: row.get(4)?,
                    envelope: row.get(5)?,
                    queue_sequence: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);

        let mut batch = ValidatedExpiredMessageBatch {
            valid_rows: Vec::with_capacity(stored_rows.len()),
            corrupt_rows: Vec::new(),
            selected_rowids: Vec::with_capacity(stored_rows.len()),
        };
        for row in stored_rows {
            batch.selected_rowids.push(row.rowid);
            let encoded_bytes = u64::try_from(row.envelope.len()).unwrap_or(u64::MAX);
            let corrupt = |reason| CorruptDurableRow {
                row_key: row.rowid,
                source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
                reason,
                encoded_bytes,
            };
            let parsed = (|| {
                let message_id: [u8; 16] = row
                    .message_id
                    .try_into()
                    .map_err(|_| corrupt("expired_message_id"))?;
                let sender: [u8; 32] = row
                    .sender
                    .try_into()
                    .map_err(|_| corrupt("expired_message_sender"))?;
                let receiver: [u8; 32] = row
                    .receiver
                    .try_into()
                    .map_err(|_| corrupt("expired_message_receiver"))?;
                let timestamp = u64::try_from(row.timestamp)
                    .map_err(|_| corrupt("expired_message_timestamp"))?;
                let envelope = decode_envelope(&row.envelope)
                    .map_err(|_| corrupt("expired_message_envelope"))?;
                if envelope.message_id != message_id {
                    return Err(corrupt("expired_message_id_mismatch"));
                }
                if envelope.sender != sender {
                    return Err(corrupt("expired_message_sender_mismatch"));
                }
                if envelope.receiver != receiver {
                    return Err(corrupt("expired_message_receiver_mismatch"));
                }
                if envelope.timestamp != timestamp {
                    return Err(corrupt("expired_message_timestamp_mismatch"));
                }
                envelope
                    .verify_signature()
                    .map_err(|_| corrupt("expired_message_signature"))?;
                match row.queue_sequence {
                    Some(sequence) if sequence > 0 => {}
                    _ => return Err(corrupt("expired_message_queue_sequence")),
                }
                Ok::<ExpiredMessageRow, CorruptDurableRow>(ExpiredMessageRow {
                    message_id,
                    sender,
                    receiver,
                })
            })();
            match parsed {
                Ok(valid) => batch.valid_rows.push(valid),
                Err(corrupt) => batch.corrupt_rows.push(corrupt),
            }
        }
        Ok(batch)
    }

    fn queue_expiry_notifications(
        tx: &Transaction<'_>,
        now: i64,
        expired_rows: &[ExpiredMessageRow],
    ) -> ChatRelayResult<()> {
        let mut by_sender = ExpiredMessagesBySender::new();
        for row in expired_rows {
            by_sender
                .entry(row.sender)
                .or_default()
                .entry(row.receiver)
                .or_default()
                .push(row.message_id);
        }

        for (sender, by_receiver) in &by_sender {
            for (receiver, ids) in by_receiver {
                for ids_chunk in ids.chunks(MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION) {
                    let ids_bytes = bincode::serialize(ids_chunk)?;
                    if ids_bytes.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
                        return Err(ChatRelayError::CorruptStoredData {
                            field: "generated_expired_notification_payload_size",
                        });
                    }
                    tx.execute(
                        "INSERT INTO expired_notifications
                         (sender, receiver, message_ids, created_at, pushed)
                         VALUES (?1, ?2, ?3, ?4, 0)",
                        params![sender.as_slice(), receiver.as_slice(), ids_bytes, now],
                    )?;
                }
            }
        }
        Ok(())
    }

    fn delete_expired_message_batch(
        tx: &Transaction<'_>,
        selected_rowids: &[i64],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM pending_messages WHERE rowid = ?1")?;
        let mut deleted = 0usize;
        for rowid in selected_rowids {
            deleted = deleted.saturating_add(stmt.execute(params![rowid])?);
        }
        if deleted != selected_rowids.len() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_message_cleanup_count",
            });
        }
        Ok(())
    }

    fn delete_expired_blob_batch(
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize> {
        Ok(tx.execute(
            "DELETE FROM pending_blobs
             WHERE rowid IN (
                 SELECT rowid FROM pending_blobs
                 WHERE received_at < ?1
                 ORDER BY received_at ASC, rowid ASC
                 LIMIT ?2
             )",
            params![cutoff, limit],
        )?)
    }

    fn delete_stale_notification_batch(
        tx: &Transaction<'_>,
        notif_cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize> {
        Ok(tx.execute(
            "DELETE FROM expired_notifications
             WHERE id IN (
                 SELECT id FROM expired_notifications
                 WHERE pushed = 1 OR created_at < ?1
                 ORDER BY id ASC
                 LIMIT ?2
             )",
            params![notif_cutoff, limit],
        )?)
    }

    fn cleanup_backlog_exists(
        tx: &Transaction<'_>,
        cutoff: i64,
        notif_cutoff: i64,
    ) -> ChatRelayResult<bool> {
        let message_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pending_messages
                 WHERE status = 0 AND received_at < ?1
             )",
            params![cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let blob_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pending_blobs WHERE received_at < ?1
             )",
            params![cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let notification_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM expired_notifications
                 WHERE pushed = 1 OR created_at < ?1
             )",
            params![notif_cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let max_quarantine_events = i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX);
        let quarantine_has_more = tx.query_row(
            "SELECT
                 EXISTS(
                     SELECT 1 FROM relay_quarantine_events
                     WHERE quarantined_at < ?1
                 )
                 OR (SELECT COUNT(*) FROM relay_quarantine_events) > ?2",
            params![notif_cutoff, max_quarantine_events],
            |row| row.get::<_, i64>(0),
        )? != 0;

        Ok(message_has_more || blob_has_more || notification_has_more || quarantine_has_more)
    }

    // ============================================
    // Peer relay health
    // ============================================

    /// Records a compatibility direct node-to-node encrypted relay round.
    ///
    /// The backward-compatible aggregate fields are also advanced.
    /// The failure reason must be a stable bucket such as
    /// `peer_relay_request_timeout` or `peer_relay_http_503`; do not pass peer
    /// URLs, message IDs, wallet IDs, client IPs, or payload-derived data.
    pub fn record_peer_relay_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::DirectPeer,
        );
    }

    /// Begins one target-bound direct relay delivery when the circuit permits.
    ///
    /// A returned permit must be completed after a network outcome or cancelled
    /// if local request construction fails before network I/O. `None` means the
    /// caller must fail closed without falling back to an older relay protocol.
    pub(crate) fn begin_direct_peer_delivery(&self, now: u64) -> Option<ChatRelayDirectPeerPermit> {
        // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Admission is intentionally
        // process-global and source-blind; per-peer quarantine remains owned by
        // PeerStore and must not be duplicated with identity-labelled state here.
        let mut circuit = self.direct_peer_relay_circuit.lock();
        let previous = circuit.clone();
        let mut next = previous.clone();
        let permit = next.begin(now);
        if next.safety_state_changed(&previous) {
            if !self.persist_direct_peer_circuit_transition(&mut circuit, next, now) {
                return None;
            }
        } else {
            *circuit = next;
        }
        permit
    }

    /// Releases an unused half-open permit after a local preflight failure.
    pub(crate) fn cancel_direct_peer_delivery(&self, now: u64, permit: ChatRelayDirectPeerPermit) {
        let mut circuit = self.direct_peer_relay_circuit.lock();
        let previous = circuit.clone();
        let mut next = previous.clone();
        next.cancel(now, permit);
        if next.safety_state_changed(&previous) {
            let _ = self.persist_direct_peer_circuit_transition(&mut circuit, next, now);
        } else {
            *circuit = next;
        }
    }

    /// Completes one aggregate target-bound direct relay delivery observation.
    ///
    /// `retry_triggered` means a second exact request was sent. A triggered
    /// retry is classified as either recovered or exhausted. Independently,
    /// `final_failure_deterministic` records that the final typed failure was
    /// not eligible for another ambiguity retry. Every delivery updates the
    /// recent SLO window, while successful first attempts leave the lifetime
    /// exception counters unchanged.
    ///
    /// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Parameters are
    /// deliberately aggregate booleans. Do not extend this API with peer,
    /// message, commitment, endpoint, wallet, or payload identifiers.
    pub(crate) fn complete_direct_peer_delivery(
        &self,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) -> bool {
        debug_assert!(!(delivery_succeeded && final_failure_deterministic));
        // [DIRECT-RELAY-SLO 2026-08-15 by Codex] Every v3 delivery contributes
        // exactly one aggregate sample, including successful first attempts.
        // The fixed ring retains no event identity or peer dimension.
        let mut circuit = self.direct_peer_relay_circuit.lock();
        if !circuit.accepts_completion(permit) {
            return false;
        }
        let slo_failed = {
            let mut window = self.direct_peer_retry_slo.lock();
            window.record(
                now,
                retry_triggered,
                delivery_succeeded,
                final_failure_deterministic,
            );
            window.snapshot(now).status == "failed"
        };
        let previous = circuit.clone();
        let mut next = previous.clone();
        let mut circuit_allows_more = next.complete(now, permit, delivery_succeeded, slo_failed);
        if next.safety_state_changed(&previous) {
            circuit_allows_more =
                self.persist_direct_peer_circuit_transition(&mut circuit, next, now)
                    && circuit_allows_more;
        } else {
            *circuit = next;
        }
        drop(circuit);
        if !retry_triggered && !final_failure_deterministic {
            return circuit_allows_more;
        }

        let mut status = self.peer_status.write();
        let retry = &mut status.direct_peer_retry;
        if retry_triggered {
            retry.retry_triggered_total = retry.retry_triggered_total.saturating_add(1);
            if delivery_succeeded {
                retry.retry_recovered_total = retry.retry_recovered_total.saturating_add(1);
                retry.last_outcome = Some("recovered".to_string());
            } else {
                retry.retry_exhausted_total = retry.retry_exhausted_total.saturating_add(1);
                retry.last_outcome = Some("exhausted".to_string());
            }
        } else {
            retry.last_outcome = Some("deterministic_failure".to_string());
        }
        if final_failure_deterministic {
            retry.deterministic_failure_total = retry.deterministic_failure_total.saturating_add(1);
        }
        retry.last_at = Some(now);
        circuit_allows_more
    }

    /// Records one authenticated receipt-verified onion relay round.
    ///
    /// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] This advances both the
    /// backward-compatible aggregate and the authenticated-onion snapshot.
    /// A later compatibility fallback may update the aggregate but cannot
    /// erase the authenticated path's latest evidence.
    pub fn record_authenticated_onion_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::AuthenticatedOnion,
        );
    }

    fn record_outbound_round(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
        route_class: OutboundRouteClass,
    ) {
        let failed = attempted.saturating_sub(accepted);
        let status_bucket = if attempted == 0 && failure_reason.is_some() {
            "failed"
        } else if attempted == 0 {
            "idle"
        } else if accepted == attempted {
            "healthy"
        } else if accepted > 0 {
            "degraded"
        } else {
            "failed"
        };
        let failure_reason = if failed > 0 || (attempted == 0 && status_bucket == "failed") {
            Some(failure_reason.unwrap_or_else(|| "unknown".to_string()))
        } else {
            None
        };

        let mut status = self.peer_status.write();
        let route_status = match route_class {
            OutboundRouteClass::AuthenticatedOnion => &mut status.authenticated_onion_outbound,
            OutboundRouteClass::DirectPeer => &mut status.direct_peer_outbound,
        };
        route_status.attempted_total = route_status
            .attempted_total
            .saturating_add(attempted as u64);
        route_status.accepted_total = route_status.accepted_total.saturating_add(accepted as u64);
        route_status.failed_total = route_status.failed_total.saturating_add(failed as u64);
        route_status.rounds = route_status.rounds.saturating_add(1);
        route_status.last_attempted = attempted as u64;
        route_status.last_accepted = accepted as u64;
        route_status.last_failed = failed as u64;
        route_status.last_status = Some(status_bucket.to_string());
        route_status.last_failure_reason = failure_reason.clone();
        route_status.last_at = Some(now);
        if accepted > 0 {
            route_status.consecutive_failures = 0;
            route_status.last_success_at = Some(now);
        } else if attempted > 0 || route_status.last_failure_reason.is_some() {
            route_status.consecutive_failures = route_status.consecutive_failures.saturating_add(1);
        }

        // Preserve the original aggregate contract for existing health and
        // heartbeat consumers while route-specific readers use the fields above.
        status.outbound_attempted_total = status
            .outbound_attempted_total
            .saturating_add(attempted as u64);
        status.outbound_accepted_total = status
            .outbound_accepted_total
            .saturating_add(accepted as u64);
        status.outbound_failed_total = status.outbound_failed_total.saturating_add(failed as u64);
        status.outbound_rounds = status.outbound_rounds.saturating_add(1);
        status.last_outbound_attempted = attempted as u64;
        status.last_outbound_accepted = accepted as u64;
        status.last_outbound_failed = failed as u64;
        status.last_outbound_status = Some(status_bucket.to_string());
        status.last_outbound_failure_reason = failure_reason;
        status.last_outbound_at = Some(now);

        if accepted > 0 {
            status.consecutive_outbound_failures = 0;
            status.last_outbound_success_at = Some(now);
        } else if attempted > 0 || status.last_outbound_failure_reason.is_some() {
            status.consecutive_outbound_failures =
                status.consecutive_outbound_failures.saturating_add(1);
        }
    }

    /// Records an accepted inbound peer relay request.
    pub fn record_peer_relay_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    ) {
        let mut status = self.peer_status.write();
        status.inbound_accepted_total = status.inbound_accepted_total.saturating_add(1);
        if duplicate {
            status.inbound_duplicate_total = status.inbound_duplicate_total.saturating_add(1);
        }
        status.inbound_delivered_online_total = status
            .inbound_delivered_online_total
            .saturating_add(delivered_online as u64);
        if stored_pending {
            status.inbound_stored_pending_total =
                status.inbound_stored_pending_total.saturating_add(1);
        }
        status.last_inbound_status = Some(if duplicate { "duplicate" } else { "accepted" }.into());
        status.last_inbound_failure_reason = None;
        status.last_inbound_at = Some(now);
    }

    /// Records a rejected inbound peer relay request with a stable reason bucket.
    pub fn record_peer_relay_inbound_rejected(&self, now: u64, reason: impl Into<String>) {
        let mut status = self.peer_status.write();
        status.inbound_rejected_total = status.inbound_rejected_total.saturating_add(1);
        status.last_inbound_status = Some("rejected".to_string());
        status.last_inbound_failure_reason = Some(reason.into());
        status.last_inbound_at = Some(now);
    }

    // ============================================
    // Accessors
    // ============================================

    #[must_use]
    pub fn config(&self) -> &ChatRelayConfig {
        &self.config
    }

    /// Returns a privacy-safe node-to-node relay health snapshot.
    #[must_use]
    pub fn peer_status(&self) -> ChatRelayPeerStatus {
        // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Snapshot each process-local
        // aggregate independently; neither guard is held while taking the next.
        // Completion uses circuit -> SLO, so no reverse nested lock exists.
        let now = now_secs();
        let circuit = self.direct_peer_relay_circuit.lock().snapshot(now);
        let recent_window = self.direct_peer_retry_slo.lock().snapshot(now);
        let mut status = self.peer_status.read().clone();
        status.direct_peer_retry.recent_window = recent_window;
        status.direct_peer_retry.circuit = circuit;
        status
    }

    /// Returns aggregate TTL cleanup execution evidence.
    #[must_use]
    pub fn maintenance_status(&self) -> ChatRelayMaintenanceStatus {
        self.maintenance_status.read().clone()
    }

    /// Records a blocking-worker failure that occurred outside `run_cleanup`.
    ///
    /// Tokio join failures are deliberately converted to stable buckets so a
    /// heartbeat never exposes panic payloads or other runtime internals.
    pub(crate) fn record_maintenance_worker_failure(&self, reason: &'static str) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
        status.last_cleanup_at = Some(now_secs());
        status.last_cleanup_status = Some("failed".to_string());
        status.last_cleanup_failure_reason = Some(reason.to_string());
        status.last_cleanup_batches = 0;
        status.last_cleanup_backlog_deferred = false;
    }

    /// Returns aggregate durable queue usage maintained by `SQLite` triggers.
    ///
    /// The result contains no message, wallet, sender, receiver, route, or
    /// payload identifiers and is safe for operator-capacity telemetry.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error if the reconciled singleton usage row cannot be
    /// read. Callers must treat that as unavailable telemetry, not zero usage.
    pub fn storage_usage(&self) -> ChatRelayResult<ChatRelayStorageUsage> {
        let conn = self.conn.lock();
        Self::read_storage_usage(&conn)
    }

    /// Creates an owner-private, verified recovery image of relay custody.
    ///
    /// The artifact is confined to `.aeronyx-relay-backups` beside the active
    /// database; callers cannot redirect encrypted custody to an arbitrary
    /// path. `SQLite`'s online-backup API includes committed WAL pages. Before
    /// publication, the isolated image must pass physical integrity, schema
    /// sentinel, canonical usage, queue sequence, and anonymous circuit-state
    /// validation. Existing artifacts are never replaced.
    ///
    /// [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] The source connection
    /// mutex remains held only during page copying. Validation and publication
    /// operate on the isolated artifact and never inspect E2E ciphertext,
    /// wallet identities, routes, or message identifiers. This is synchronous
    /// storage work; async operator surfaces must invoke it through
    /// `spawn_blocking`, never from an Axum request executor directly.
    ///
    /// # Errors
    ///
    /// Returns a path-free SQLite/storage error if the service uses an
    /// in-memory database, the private backup area cannot be secured, the
    /// source remains locked, validation fails, or durable publication fails.
    pub fn create_verified_backup(&self) -> ChatRelayResult<PathBuf> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let created_at = now_secs();
        let nonce = rand::random::<u64>();
        let destination =
            backup_directory.join(format!("relay-custody-{created_at}-{nonce:016x}.sqlite"));
        self.create_verified_backup_artifact(&backup_directory, &destination, false)?;
        Ok(destination)
    }

    /// Creates or reuses one verified backup for an audited operation ID.
    ///
    /// The raw ID is never persisted. A domain-separated HMAC under the stable
    /// node secret selects one private destination, so CMS replay after ACK
    /// loss or process restart cannot create a second recovery image. An
    /// existing artifact is opened read-only with `NOFOLLOW` and fully verified
    /// before it is accepted. Corrupt or non-private artifacts fail closed.
    ///
    /// [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] This method returns
    /// aggregate receipt data only; callers cannot learn the artifact key or
    /// path. It is synchronous and must run through `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the operation ID is empty or too
    /// large, the private boundary is unavailable, snapshot certification or
    /// publication fails, or a replay artifact cannot be re-verified.
    pub(crate) fn create_verified_backup_for_operation(
        &self,
        operation_id: &str,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        if operation_id.is_empty() || operation_id.len() > CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_MISUSE,
                "invalid relay backup operation identifier",
            ));
        }

        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(CHAT_RELAY_BACKUP_OPERATION_HMAC_DOMAIN);
        mac.update(&(operation_id.len() as u64).to_be_bytes());
        mac.update(operation_id.as_bytes());
        let digest = mac.finalize().into_bytes();
        let opaque_key = hex::encode(&digest[..16]);

        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let destination =
            backup_directory.join(format!("relay-custody-operation-{opaque_key}.sqlite"));
        self.create_verified_backup_artifact(&backup_directory, &destination, true)
    }

    /// Audits the configured retention policy without creating or deleting
    /// recovery images.
    ///
    /// Every managed recovery image is fully re-verified before the first
    /// capacity calculation. Incomplete private SQLite files left by an
    /// interrupted backup are reported, while unknown entries, symlinks,
    /// permission drift, or corrupt recovery images fail closed. At least the
    /// newest verified image is counted as retained even if it alone exceeds
    /// the configured byte budget.
    ///
    /// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] This operation shares
    /// one lock with publication and replay, so the inspection cannot race a
    /// backup that has not yet produced its audit receipt. This method has no
    /// filesystem deletion path.
    /// This is synchronous filesystem/SQLite work; async callers must use a
    /// blocking worker rather than an application request executor.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary cannot be
    /// inspected, its entry count is unbounded, an entry is unmanaged or not
    /// owner-private, or a recovery image fails full SQLite verification.
    pub fn audit_verified_backup_retention(
        &self,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(
            Self::inspect_verified_backup_retention(&self.config, &backup_directory, now_secs())?
                .receipt,
        )
    }

    /// Audits one configured private backup boundary without opening the live
    /// relay database.
    ///
    /// This host-local entry point exists for the CLI. It takes the same
    /// cross-process lock as online backup creation and returns aggregate-only
    /// state without writing an audit record or deleting an artifact.
    pub fn audit_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(Self::inspect_verified_backup_retention(config, &backup_directory, now_secs())?.receipt)
    }

    /// Verifies whether the newest private recovery image is ready for a
    /// separately approved host-local restore operation.
    ///
    /// The command fully verifies every managed backup under the same
    /// cross-process lock used by publication and prune, then inspects the
    /// configured active main file and sidecars through metadata only. It never
    /// opens active storage, writes a maintenance audit record, or changes
    /// custody/backup artifacts. The shared control lock may be initialized.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary is unsafe,
    /// any managed image is corrupt, or active-file metadata cannot be trusted.
    pub fn audit_latest_restore_readiness_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayRestoreReadinessReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, now_secs())?;
        let verified_backup_count = Self::verified_restore_backup_count(&inspection)?;
        let selected_backup_bytes = inspection
            .newest_backup
            .as_ref()
            .map(|artifact| artifact.size_bytes)
            .unwrap_or_default();
        let active = Self::inspect_active_restore_boundary(config)?;
        let blocker = if inspection.newest_backup.is_none() {
            Some("no_verified_backup")
        } else if active.sidecars_present {
            Some("active_sqlite_sidecars_present")
        } else {
            None
        };

        Ok(ChatRelayRestoreReadinessReceipt {
            ready: blocker.is_none(),
            verified_backup_count,
            selected_backup_bytes,
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
            active_sidecars_present: active.sidecars_present,
            blocker,
        })
    }

    /// Creates a short-lived authenticated plan for the newest verified image.
    ///
    /// The plan commits to private artifact identity, active database identity,
    /// aggregate recovery state, a random nonce, and a fixed ten-minute expiry.
    /// It does not copy, replace, remove, or open active custody. The plan is
    /// not sufficient authorization for a future restore operation.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when no verified image exists, active
    /// `SQLite` sidecars remain, private metadata is unsafe, or the commitment
    /// cannot be produced.
    pub fn create_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        Self::create_latest_restore_plan_at(config, node_secret, now_secs())
    }

    fn create_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        issued_at: u64,
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, issued_at)?;
        let backup = inspection.newest_backup.as_ref().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay restore plan requires a verified backup",
            )
        })?;
        let active = Self::inspect_active_restore_boundary(config)?;
        if active.sidecars_present {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay restore plan requires an inactive SQLite boundary",
            ));
        }

        let expires_at = issued_at
            .checked_add(CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS)
            .ok_or_else(|| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_RANGE,
                    "relay restore-plan expiry is out of range",
                )
            })?;
        let mut nonce = [0u8; CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let mut plan = ChatRelayRestorePlanReceipt {
            version: CHAT_RELAY_RESTORE_PLAN_VERSION,
            issued_at,
            expires_at,
            verified_backup_count: u64::try_from(Self::verified_restore_backup_count(&inspection)?)
                .map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_FULL,
                        "relay restore-plan backup count exceeds wire format",
                    )
                })?,
            selected_backup_bytes: backup.size_bytes,
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
            nonce: hex::encode(nonce),
            commitment: String::new(),
        };
        plan.commitment = hex::encode(
            Self::restore_plan_mac(node_secret, config, &plan, backup, &active)?
                .finalize()
                .into_bytes(),
        );
        Ok(plan)
    }

    /// Re-verifies that an authenticated restore plan remains current.
    ///
    /// Verification repeats full backup integrity checks and active-boundary
    /// metadata inspection under the shared filesystem lock. Any expiry,
    /// tampering, backup rotation, config drift, or active database change
    /// invalidates the plan. This remains a read-only check and does not grant
    /// permission to restore. The shared lock is released on return; a future
    /// executor must re-verify and replace storage inside one uninterrupted
    /// lock scope rather than treating this method as TOCTOU-safe authority.
    ///
    /// # Errors
    ///
    /// Returns a generic path-free authentication error for an invalid,
    /// expired, or stale plan, and a storage error when the private boundary
    /// itself cannot be inspected safely.
    pub fn verify_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
    ) -> ChatRelayResult<()> {
        Self::verify_latest_restore_plan_at(config, node_secret, plan, now_secs())
    }

    fn verify_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> ChatRelayResult<()> {
        let valid_time = plan.version == CHAT_RELAY_RESTORE_PLAN_VERSION
            && plan
                .issued_at
                .checked_add(CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS)
                == Some(plan.expires_at)
            && plan.issued_at <= now_unix_secs
            && now_unix_secs < plan.expires_at;
        if !valid_time
            || !Self::is_lower_hex(&plan.nonce, CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES * 2)
            || !Self::is_lower_hex(&plan.commitment, 64)
        {
            return Err(Self::invalid_restore_plan());
        }

        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, now_unix_secs)?;
        let backup = inspection
            .newest_backup
            .as_ref()
            .ok_or_else(Self::invalid_restore_plan)?;
        let active = Self::inspect_active_restore_boundary(config)?;
        let current_count = u64::try_from(Self::verified_restore_backup_count(&inspection)?)
            .map_err(|_| Self::invalid_restore_plan())?;
        if active.sidecars_present
            || plan.verified_backup_count != current_count
            || plan.selected_backup_bytes != backup.size_bytes
            || plan.active_database_present != active.present
            || plan.active_database_bytes != active.size_bytes
        {
            return Err(Self::invalid_restore_plan());
        }

        let commitment = hex::decode(&plan.commitment).map_err(|_| Self::invalid_restore_plan())?;
        Self::restore_plan_mac(node_secret, config, plan, backup, &active)?
            .verify_slice(&commitment)
            .map_err(|_| Self::invalid_restore_plan())
    }

    /// Runs a host-local retention dry-run or explicitly-confirmed prune.
    ///
    /// Dry-run is the default request state. Deletion requires the exact
    /// confirmation phrase and an operator assertion that the node is stopped;
    /// every mutation is bracketed by a private HMAC-chained aggregate audit.
    pub fn prune_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        Self::prune_verified_backup_retention_at(config, node_secret, request, now_secs())
    }

    /// Runs the same prune contract through an initialized relay service.
    pub fn prune_verified_backup_retention(
        &self,
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        let _operation = self.backup_operations.lock();
        Self::prune_verified_backup_retention_at(
            &self.config,
            &self.node_secret,
            request,
            now_secs(),
        )
    }
}

fn nonnegative_sqlite_counter(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

fn nonnegative_sqlite_value(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

fn optional_nonnegative_sqlite_value(
    value: Option<i64>,
    field: &'static str,
) -> ChatRelayResult<Option<u64>> {
    value
        .map(|value| nonnegative_sqlite_value(value, field))
        .transpose()
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

fn optional_sqlite_integer(
    value: Option<u64>,
    field: &'static str,
) -> ChatRelayResult<Option<i64>> {
    value.map(|value| sqlite_integer(value, field)).transpose()
}

// ============================================
// Helpers
// ============================================

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Derives a stable 32-byte node secret from the node's Ed25519 private key.
pub fn derive_node_secret(ed25519_sk_bytes: &[u8; 32]) -> [u8; 32] {
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(Some(b"aeronyx-chat-relay-v1"), ed25519_sk_bytes);
    let mut okm = [0u8; 32];
    hk.expand(b"", &mut okm)
        .expect("HKDF expand with 32-byte output always succeeds");
    okm
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_common::types::SessionId;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;
    use sha2::{Digest, Sha256};
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::sync::Barrier;

    fn test_config() -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: ":memory:".to_string(),
            offline_ttl_secs: 259_200,
            max_pending_per_wallet: 5,
            max_pending_messages_total: 100,
            max_pending_message_bytes_total: 1024 * 1024,
            max_message_size: 65_536,
            max_blob_size: 1_024,
            max_blobs_per_receiver: 3,
            max_pending_blobs_total: 10,
            max_pending_blob_bytes_total: 10 * 1024,
            cleanup_interval_secs: 60,
            dedup_lru_capacity: 10,
            expired_notification_ttl_secs: 604_800,
            peer_relay_requests_per_minute: 1_200,
            peer_relay_authenticated_requests_per_minute: 240,
            // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Service tests
            // inherit recovery-planning defaults unless a case overrides them.
            ..ChatRelayConfig::default()
        }
    }

    fn make_service() -> ChatRelayService {
        make_service_with_config(test_config())
    }

    fn make_service_with_config(config: ChatRelayConfig) -> ChatRelayService {
        let secret = derive_node_secret(&[0x42u8; 32]);
        ChatRelayService::new(config, secret).expect("init")
    }

    fn complete_direct_peer_test_delivery(
        svc: &ChatRelayService,
        now: u64,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) {
        let permit = svc
            .begin_direct_peer_delivery(now)
            .expect("test delivery should be admitted");
        svc.complete_direct_peer_delivery(
            now,
            permit,
            retry_triggered,
            delivery_succeeded,
            final_failure_deterministic,
        );
    }

    fn unique_test_db_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "aeronyx-chat-relay-{}-{}-{}.sqlite",
            label,
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    fn remove_test_db(path: &Path) {
        let _ = std::fs::remove_file(path);
        remove_test_db_sidecars(path);
    }

    fn remove_test_db_sidecars(path: &Path) {
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        let _ = std::fs::remove_file(format!("{}-journal", path.display()));
    }

    fn backup_directory_snapshot(path: &Path) -> Vec<(String, Vec<u8>)> {
        let mut snapshot = std::fs::read_dir(path)
            .expect("read backup directory snapshot")
            .map(|entry| {
                let entry = entry.expect("read backup directory entry");
                let name = entry
                    .file_name()
                    .into_string()
                    .expect("test backup name is UTF-8");
                let bytes = std::fs::read(entry.path()).expect("read backup artifact");
                (name, bytes)
            })
            .collect::<Vec<_>>();
        snapshot.sort_by(|left, right| left.0.cmp(&right.0));
        snapshot
    }

    fn insert_expired_pending_rows(svc: &ChatRelayService, count: usize, prefix: u8) {
        let identity = IdentityKeyPair::generate();
        let mut conn = svc.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .expect("start bulk pending insert");
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO pending_messages
                     (message_id, sender, receiver, timestamp, envelope, received_at, status,
                      queue_sequence)
                     VALUES (?1, ?2, ?3, 0, ?4, 0, 0, ?5)",
                )
                .expect("prepare bulk pending insert");
            for sequence in 0..count {
                let mut message_id = [0u8; 16];
                message_id[0] = prefix;
                message_id[8..]
                    .copy_from_slice(&u64::try_from(sequence).unwrap_or(u64::MAX).to_be_bytes());
                let mut envelope = ChatEnvelope {
                    message_id,
                    sender: identity.public_key_bytes(),
                    receiver: [0xA3u8; 32],
                    timestamp: 0,
                    ciphertext: vec![0xA4],
                    nonce: [0u8; 24],
                    content_type: ChatContentType::System,
                    signature: [0u8; 64],
                };
                envelope.signature = identity.sign(&envelope.sign_data());
                let encoded_envelope = encode_envelope(&envelope).expect("encode expired envelope");
                let queue_sequence = ChatRelayService::allocate_queue_sequence(&tx)
                    .expect("allocate test queue sequence");
                stmt.execute(params![
                    message_id.as_slice(),
                    envelope.sender.as_slice(),
                    envelope.receiver.as_slice(),
                    encoded_envelope,
                    queue_sequence,
                ])
                .expect("insert expired pending row");
            }
        }
        tx.commit().expect("commit bulk pending insert");
    }

    #[test]
    fn chat_relay_custody_uses_full_sqlite_durability() {
        // [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] A successful
        // service construction is also the activation gate for signed custody
        // receipts, so the effective connection mode must be FULL or EXTRA.
        let svc = make_service();
        let synchronous_level = svc
            .conn
            .lock()
            .query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))
            .expect("read effective relay durability");
        assert!(synchronous_level >= CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL);
        let durability = svc.peer_status().custody_durability;
        assert_eq!(durability.state, "full");
        assert!(durability.full_durability_verified);
        assert_eq!(
            durability.synchronous_level,
            Some(u8::try_from(synchronous_level).expect("SQLite level fits u8"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn chat_relay_custody_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] Cover both an existing
        // permissive database and the WAL/SHM files SQLite creates after the
        // primary mode is tightened. This test never changes process umask.
        let db_path = unique_test_db_path("private-custody-file");
        std::fs::write(&db_path, []).expect("create permissive relay database");
        std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o666))
            .expect("make relay database permissive");

        let mut config = test_config();
        config.db_path = db_path.to_string_lossy().into_owned();
        let service = make_service_with_config(config);

        for path in [
            db_path.clone(),
            PathBuf::from(format!("{}-wal", db_path.display())),
            PathBuf::from(format!("{}-shm", db_path.display())),
        ] {
            let mode = std::fs::metadata(&path)
                .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()))
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "{} must be owner-only", path.display());
        }

        drop(service);
        remove_test_db(&db_path);
    }

    #[cfg(unix)]
    #[test]
    fn chat_relay_permission_failure_is_path_private_and_fail_closed() {
        // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] A missing target
        // deterministically exercises the activation error without relying on
        // process-global umask, root privileges, or platform ACL behavior.
        let db_path = unique_test_db_path("missing-private-custody-file");
        remove_test_db(&db_path);

        let error = ChatRelayService::restrict_sqlite_file_permissions(&db_path)
            .expect_err("unrestrictable relay database must fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        let rendered = error.to_string();
        assert!(rendered.contains("unable to restrict relay database permissions"));
        assert!(!rendered.contains(db_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn chat_relay_startup_integrity_rejects_corrupt_schema_before_activation() {
        // [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] Build a valid
        // production-shaped store, corrupt one schema root page through
        // SQLite's test-only writable schema, then exercise a real restart.
        let db_path = unique_test_db_path("startup-integrity-corrupt-schema");
        let mut config = test_config();
        config.db_path = db_path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x42u8; 32]);

        let service = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        drop(service);

        let corruptor = Connection::open(&db_path).expect("open relay store for corruption drill");
        corruptor
            .execute_batch(
                "PRAGMA writable_schema=ON;
                 UPDATE sqlite_schema
                 SET rootpage=2147483647
                 WHERE type='table' AND name='pending_messages';
                 PRAGMA writable_schema=OFF;",
            )
            .expect("install malformed root page fixture");
        drop(corruptor);

        let error = match ChatRelayService::new(config, secret) {
            Ok(_) => panic!("corrupt custody database must not activate"),
            Err(error) => error,
        };
        assert_eq!(error.reason_bucket(), "corrupt_stored_data");
        let rendered = error.to_string();
        assert!(rendered.contains("sqlite_startup_integrity"));
        assert!(!rendered.contains(db_path.to_string_lossy().as_ref()));

        remove_test_db(&db_path);
    }

    #[test]
    fn verified_backup_restores_committed_custody_and_circuit_state() {
        // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Disable automatic
        // checkpoints so this drill proves the backup API reads committed WAL
        // pages rather than merely copying the primary database file.
        let directory = tempfile::tempdir().expect("verified backup directory");
        let source_path = directory.path().join("source.sqlite");
        let mut source_config = test_config();
        source_config.db_path = source_path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x8Au8; 32]);
        let identity = IdentityKeyPair::generate();
        let receiver = [0x8Bu8; 32];
        let first = make_envelope(&identity, receiver);
        let second = make_envelope(&identity, receiver);
        let blob_data = b"opaque-encrypted-backup-blob";
        let blob_hash = Sha256::digest(blob_data);
        let mut blob_hash_array = [0u8; 32];
        blob_hash_array.copy_from_slice(&blob_hash);
        let circuit_started_at = now_secs();

        let source = ChatRelayService::new(source_config.clone(), secret)
            .expect("create source relay store");
        source
            .conn
            .lock()
            .execute_batch("PRAGMA wal_autocheckpoint=0")
            .expect("keep committed custody in WAL");
        source.store_pending(&first).expect("store pending message");
        let blob_id = source
            .put_blob(
                &identity.public_key_bytes(),
                &receiver,
                blob_data,
                &blob_hash_array,
            )
            .expect("store encrypted blob");
        for offset in 0..3 {
            complete_direct_peer_test_delivery(
                &source,
                circuit_started_at.saturating_add(offset),
                false,
                false,
                true,
            );
        }
        assert_eq!(source.peer_status().direct_peer_retry.circuit.state, "open");

        let backup_path = source
            .create_verified_backup()
            .expect("create verified backup");
        source
            .store_pending(&second)
            .expect("store post-snapshot message");
        drop(source);

        let mut restored_config = source_config.clone();
        restored_config.db_path = backup_path.to_string_lossy().into_owned();
        let restored = ChatRelayService::new(restored_config, secret)
            .expect("activate verified recovery image");
        let (messages, has_more) = restored
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("read restored custody");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, first.message_id);
        assert_eq!(messages[0].envelope.ciphertext, first.ciphertext);
        assert_eq!(restored.get_blob(&blob_id).unwrap(), blob_data);
        assert_eq!(
            restored.storage_usage().unwrap(),
            ChatRelayStorageUsage {
                pending_messages: 1,
                pending_message_bytes: encode_envelope(&first).unwrap().len() as u64,
                pending_blobs: 1,
                pending_blob_bytes: blob_data.len() as u64,
            }
        );
        let circuit = restored.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "open");
        assert!(circuit.restart_protected);
        assert_eq!(circuit.opened_total, 1);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let backup_mode = std::fs::metadata(&backup_path)
                .expect("inspect backup file")
                .permissions()
                .mode()
                & 0o777;
            let backup_directory_mode = std::fs::metadata(backup_path.parent().unwrap())
                .expect("inspect backup directory")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(backup_mode, 0o600);
            assert_eq!(backup_directory_mode, 0o700);
        }

        drop(restored);
        remove_test_db(&backup_path);
        remove_test_db(&source_path);
    }

    #[test]
    fn restore_readiness_selects_verified_backup_without_mutating_artifacts() {
        // [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] A positive
        // preflight proves the newest image is fully usable while preserving
        // both active custody and every recovery artifact byte-for-byte.
        let directory = tempfile::tempdir().expect("restore readiness directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        source
            .create_verified_backup_for_operation("restore-readiness")
            .expect("create verified readiness image");
        drop(source);
        remove_test_db_sidecars(&source_path);

        let backup_directory = directory.path().join(".aeronyx-relay-backups");
        let before = backup_directory_snapshot(&backup_directory);
        let active_before = std::fs::read(&source_path).expect("read active custody before audit");
        let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
            .expect("audit restore readiness");
        let after = backup_directory_snapshot(&backup_directory);

        assert!(receipt.ready);
        assert_eq!(receipt.verified_backup_count, 1);
        assert!(receipt.selected_backup_bytes > 0);
        assert!(receipt.active_database_present);
        assert_eq!(receipt.active_database_bytes, active_before.len() as u64);
        assert!(!receipt.active_sidecars_present);
        assert_eq!(receipt.blocker, None);
        assert_eq!(before, after);
        assert_eq!(
            std::fs::read(&source_path).expect("read active custody after audit"),
            active_before
        );

        // [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Pin the public
        // aggregate contract: operators may automate against these fields,
        // while custody and recovery paths must remain private.
        let json = serde_json::to_value(&receipt).expect("serialize restore readiness receipt");
        let object = json.as_object().expect("readiness JSON is an object");
        assert_eq!(object.len(), 7);
        for field in [
            "ready",
            "verified_backup_count",
            "selected_backup_bytes",
            "active_database_present",
            "active_database_bytes",
            "active_sidecars_present",
            "blocker",
        ] {
            assert!(object.contains_key(field), "missing JSON field: {field}");
        }
        let encoded = serde_json::to_string(&receipt).expect("encode readiness JSON");
        assert!(!encoded.contains(source_path.to_string_lossy().as_ref()));
        assert!(!encoded.contains(".aeronyx-relay-backups"));
    }

    #[test]
    fn restore_readiness_reports_missing_verified_backup_without_execution() {
        let directory = tempfile::tempdir().expect("missing restore image directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        drop(source);
        remove_test_db_sidecars(&source_path);

        let active_before = std::fs::read(&source_path).expect("read active custody");
        let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
            .expect("report missing verified backup");

        assert!(!receipt.ready);
        assert_eq!(receipt.verified_backup_count, 0);
        assert_eq!(receipt.selected_backup_bytes, 0);
        assert_eq!(receipt.blocker, Some("no_verified_backup"));
        assert_eq!(
            std::fs::read(&source_path).expect("read unchanged active custody"),
            active_before
        );
    }

    #[test]
    fn restore_readiness_fails_closed_while_active_sidecar_exists() {
        let directory = tempfile::tempdir().expect("restore sidecar directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        source
            .create_verified_backup_for_operation("restore-sidecar")
            .expect("create verified sidecar image");
        drop(source);
        remove_test_db_sidecars(&source_path);
        let wal_path = PathBuf::from(format!("{}-wal", source_path.display()));
        std::fs::write(&wal_path, b"stopped-state-sidecar-marker").expect("create sidecar marker");

        let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
            .expect("report active sidecar blocker");
        assert!(!receipt.ready);
        assert!(receipt.active_sidecars_present);
        assert_eq!(receipt.blocker, Some("active_sqlite_sidecars_present"));
        assert!(
            wal_path.exists(),
            "readiness must not remove active sidecars"
        );
    }

    #[test]
    fn restore_plan_is_path_free_unique_and_verifiable() {
        // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] The public plan binds
        // private artifact identity without exposing a filename or path.
        let directory = tempfile::tempdir().expect("restore plan directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        source
            .create_verified_backup_for_operation("restore-plan")
            .expect("create verified restore-plan image");
        drop(source);
        remove_test_db_sidecars(&source_path);

        let secret = derive_node_secret(&[0x42u8; 32]);
        let issued_at = now_secs();
        let first = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
            .expect("create first authenticated plan");
        let second = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
            .expect("create second authenticated plan");

        assert_eq!(first.version, CHAT_RELAY_RESTORE_PLAN_VERSION);
        assert_eq!(
            first.expires_at - first.issued_at,
            CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS
        );
        assert_ne!(first.nonce, second.nonce);
        assert_ne!(first.commitment, second.commitment);
        assert!(ChatRelayService::is_lower_hex(&first.nonce, 32));
        assert!(ChatRelayService::is_lower_hex(&first.commitment, 64));
        ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &first, issued_at)
            .expect("verify fresh restore plan");

        let encoded = serde_json::to_string(&first).expect("encode restore plan");
        assert!(!encoded.contains(source_path.to_string_lossy().as_ref()));
        assert!(!encoded.contains(".aeronyx-relay-backups"));
        assert!(!encoded.contains("relay-custody-operation"));
        let json = serde_json::to_value(&first).expect("serialize restore-plan contract");
        let object = json.as_object().expect("restore plan JSON object");
        assert_eq!(object.len(), 9);
        for field in [
            "version",
            "issued_at",
            "expires_at",
            "verified_backup_count",
            "selected_backup_bytes",
            "active_database_present",
            "active_database_bytes",
            "nonce",
            "commitment",
        ] {
            assert!(object.contains_key(field), "missing plan field: {field}");
        }
    }

    #[test]
    fn restore_plan_rejects_tampering_wrong_key_and_expiry() {
        let directory = tempfile::tempdir().expect("restore plan auth directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        source
            .create_verified_backup_for_operation("restore-plan-auth")
            .expect("create verified restore-plan auth image");
        drop(source);
        remove_test_db_sidecars(&source_path);

        let secret = derive_node_secret(&[0x42u8; 32]);
        let issued_at = now_secs();
        let plan = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
            .expect("create authenticated plan");

        let mut tampered = plan.clone();
        tampered.selected_backup_bytes = tampered.selected_backup_bytes.saturating_add(1);
        ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &tampered, issued_at)
            .expect_err("aggregate tampering must fail closed");
        ChatRelayService::verify_latest_restore_plan_at(&config, &[0xA5u8; 32], &plan, issued_at)
            .expect_err("wrong node secret must fail closed");
        ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &plan, plan.expires_at)
            .expect_err("expired plan must fail closed");
    }

    #[test]
    fn restore_plan_rejects_private_state_drift_even_when_size_is_unchanged() {
        let directory = tempfile::tempdir().expect("restore plan drift directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config.clone());
        source
            .create_verified_backup_for_operation("restore-plan-drift")
            .expect("create verified restore-plan drift image");
        drop(source);
        remove_test_db_sidecars(&source_path);

        let secret = derive_node_secret(&[0x42u8; 32]);
        let issued_at = now_secs();
        let plan = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
            .expect("create state-bound plan");
        let active_bytes = std::fs::read(&source_path).expect("read active custody bytes");
        std::thread::sleep(Duration::from_millis(10));
        std::fs::write(&source_path, &active_bytes).expect("rewrite same active custody bytes");
        assert_eq!(
            std::fs::metadata(&source_path)
                .expect("inspect rewritten custody")
                .len(),
            plan.active_database_bytes
        );

        ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &plan, issued_at)
            .expect_err("private metadata drift must invalidate the plan");
    }

    #[test]
    fn verified_backup_rejects_inconsistent_usage_without_partial_artifact() {
        let directory = tempfile::tempdir().expect("counter mismatch backup directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config);
        let identity = IdentityKeyPair::generate();
        source
            .store_pending(&make_envelope(&identity, [0x8Cu8; 32]))
            .expect("store canonical custody row");
        source
            .conn
            .lock()
            .execute(
                "UPDATE relay_storage_usage
                 SET pending_message_count = 0, pending_message_bytes = 0
                 WHERE singleton = 1",
                [],
            )
            .expect("tamper derived usage counters");

        let error = source
            .create_verified_backup()
            .expect_err("inconsistent backup must fail closed");
        assert!(matches!(
            error,
            ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_logical_integrity"
            }
        ));
        let backup_directory = source_path.parent().unwrap().join(".aeronyx-relay-backups");
        assert_eq!(
            std::fs::read_dir(&backup_directory)
                .expect("inspect private backup directory")
                .count(),
            0,
            "failed certification must remove all partial artifacts"
        );

        drop(source);
        remove_test_db(&source_path);
    }

    #[test]
    fn verified_backup_rejects_in_memory_storage() {
        let error = make_service()
            .create_verified_backup()
            .expect_err("in-memory relay must not escape its storage boundary");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        assert!(error
            .to_string()
            .contains("in-memory relay storage has no private backup boundary"));
    }

    #[test]
    fn audited_backup_replay_reuses_one_verified_artifact_across_restart() {
        // [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] The durable
        // artifact key depends only on the stable node secret and opaque
        // operation ID. Process-local command deduplication is not involved.
        let directory = tempfile::tempdir().expect("idempotent backup directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let secret = [0x91u8; 32];

        let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        let first = source
            .create_verified_backup_for_operation("cms-command-42")
            .expect("create audited backup");
        assert!(first.created);
        assert!(first.size_bytes > 0);
        drop(source);

        let restarted = ChatRelayService::new(config, secret).expect("restart relay store");
        let replay = restarted
            .create_verified_backup_for_operation("cms-command-42")
            .expect("reuse verified backup after restart");
        assert!(!replay.created);
        assert_eq!(replay.size_bytes, first.size_bytes);

        let second = restarted
            .create_verified_backup_for_operation("cms-command-43")
            .expect("different operation creates a distinct backup");
        assert!(second.created);
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect operation artifacts")
                .count(),
            2
        );
    }

    #[test]
    fn concurrent_audited_backup_replay_publishes_exactly_once() {
        let directory = tempfile::tempdir().expect("concurrent backup directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = Arc::new(
            ChatRelayService::new(config, [0x93u8; 32]).expect("create concurrent relay store"),
        );
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let source = Arc::clone(&source);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    source
                        .create_verified_backup_for_operation("cms-command-concurrent")
                        .expect("complete concurrent audited backup")
                })
            })
            .collect();
        barrier.wait();
        let receipts: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("join backup worker"))
            .collect();

        assert_eq!(receipts.iter().filter(|receipt| receipt.created).count(), 1);
        assert_eq!(receipts[0].size_bytes, receipts[1].size_bytes);
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect concurrent operation artifacts")
                .count(),
            1
        );
    }

    #[test]
    fn audited_backup_replay_rejects_corrupt_existing_artifact_without_overwrite() {
        let directory = tempfile::tempdir().expect("corrupt replay directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = ChatRelayService::new(config, [0x92u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("cms-command-corrupt")
            .expect("create audited backup");
        let backup_directory = directory.path().join(".aeronyx-relay-backups");
        let artifact = std::fs::read_dir(&backup_directory)
            .expect("read audited backup directory")
            .next()
            .expect("one audited backup")
            .expect("valid directory entry")
            .path();
        std::fs::write(&artifact, b"corrupt-replay-fixture").expect("corrupt backup fixture");

        let error = source
            .create_verified_backup_for_operation("cms-command-corrupt")
            .expect_err("corrupt replay artifact must fail closed");
        assert_eq!(error.reason_bucket(), "corrupt_stored_data");
        assert_eq!(
            std::fs::read(&artifact).expect("read preserved corrupt artifact"),
            b"corrupt-replay-fixture"
        );
        assert_eq!(
            std::fs::read_dir(&backup_directory)
                .expect("inspect preserved backup directory")
                .count(),
            1,
            "replay must not replace or duplicate a corrupt artifact"
        );
    }

    #[test]
    fn audited_backup_replay_rejects_mutable_sidecar_state() {
        let directory = tempfile::tempdir().expect("sidecar replay directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = ChatRelayService::new(config, [0x94u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("cms-command-sidecar")
            .expect("create audited backup");
        let backup_directory = directory.path().join(".aeronyx-relay-backups");
        let artifact = std::fs::read_dir(&backup_directory)
            .expect("read audited backup directory")
            .next()
            .expect("one audited backup")
            .expect("valid directory entry")
            .path();
        let mut wal_path = artifact.as_os_str().to_os_string();
        wal_path.push("-wal");
        std::fs::write(PathBuf::from(wal_path), b"mutable-sidecar-fixture")
            .expect("install mutable sidecar fixture");

        let error = source
            .create_verified_backup_for_operation("cms-command-sidecar")
            .expect_err("mutable sidecar state must fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        assert!(error.to_string().contains("mutable sidecar state"));
    }

    #[test]
    fn audited_backup_rejects_unbounded_operation_ids_before_storage_access() {
        let source = make_service();
        for operation_id in ["".to_string(), "x".repeat(129)] {
            let error = source
                .create_verified_backup_for_operation(&operation_id)
                .expect_err("invalid operation ID must fail closed");
            assert_eq!(error.reason_bucket(), "sqlite_error");
            if !operation_id.is_empty() {
                assert!(!error.to_string().contains(&operation_id));
            }
        }
    }

    #[test]
    fn backup_retention_audit_reports_excess_without_deleting_artifacts() {
        // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Retention is a
        // local read-only decision aid until an explicitly-authorized deletion
        // command exists. The audit must never make policy irreversible.
        let directory = tempfile::tempdir().expect("retention audit directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 2;
        config.custody_backup_retention_target_bytes = u64::MAX;
        let source = ChatRelayService::new(config, [0x95u8; 32]).expect("create relay store");
        for operation in ["retention-1", "retention-2", "retention-3"] {
            source
                .create_verified_backup_for_operation(operation)
                .expect("create audited recovery image");
        }

        let receipt = source
            .audit_verified_backup_retention()
            .expect("audit verified backup retention");
        assert_eq!(receipt.retained_count, 2);
        assert_eq!(receipt.excess_count, 1);
        assert!(receipt.retained_bytes > 0);
        assert!(receipt.excess_bytes > 0);
        assert!(receipt.budget_exceeded);
        assert_eq!(receipt.partial_count, 0);
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect untouched recovery images")
                .count(),
            3,
            "read-only retention audit must not remove excess artifacts"
        );
    }

    #[test]
    fn backup_retention_audit_keeps_one_recovery_point_over_byte_budget() {
        let directory = tempfile::tempdir().expect("retention byte budget directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 8;
        config.custody_backup_retention_target_bytes = 1;
        let source = ChatRelayService::new(config, [0x96u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("retention-byte-1")
            .expect("create first recovery image");
        source
            .create_verified_backup_for_operation("retention-byte-2")
            .expect("create second recovery image");

        let receipt = source
            .audit_verified_backup_retention()
            .expect("audit byte-limited retention");
        assert_eq!(receipt.retained_count, 1);
        assert_eq!(receipt.excess_count, 1);
        assert!(receipt.retained_bytes > 1);
        assert!(receipt.budget_exceeded);
    }

    #[test]
    fn backup_retention_audit_reports_private_partial_without_removing_it() {
        let directory = tempfile::tempdir().expect("retention partial directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = ChatRelayService::new(config, [0x97u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("retention-partial")
            .expect("create recovery image");
        let partial = directory
            .path()
            .join(".aeronyx-relay-backups")
            .join(".relay-custody-1800000000-0123456789abcdef.tmp");
        std::fs::write(&partial, b"interrupted-private-snapshot")
            .expect("install interrupted snapshot fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600))
                .expect("restrict partial fixture");
        }

        let receipt = source
            .audit_verified_backup_retention()
            .expect("audit interrupted private snapshot");
        assert_eq!(receipt.partial_count, 1);
        assert_eq!(receipt.partial_bytes, 28);
        assert!(partial.exists(), "read-only audit must preserve partials");
    }

    #[test]
    fn backup_retention_audit_rejects_unmanaged_entries_without_side_effects() {
        let directory = tempfile::tempdir().expect("retention unmanaged directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = ChatRelayService::new(config, [0x98u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("retention-unmanaged")
            .expect("create recovery image");
        let unmanaged = directory
            .path()
            .join(".aeronyx-relay-backups")
            .join("operator-note.txt");
        std::fs::write(&unmanaged, b"do not interpret").expect("install unmanaged fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&unmanaged, std::fs::Permissions::from_mode(0o600))
                .expect("restrict unmanaged fixture");
        }

        let error = source
            .audit_verified_backup_retention()
            .expect_err("unmanaged entry must make audit fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        assert!(unmanaged.exists());
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect preserved directory")
                .count(),
            2
        );
    }

    #[test]
    fn backup_retention_audit_rejects_corrupt_managed_artifact_without_mutation() {
        let directory = tempfile::tempdir().expect("retention corrupt directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = ChatRelayService::new(config, [0x99u8; 32]).expect("create relay store");
        source
            .create_verified_backup_for_operation("retention-corrupt")
            .expect("create recovery image");
        let backup_directory = directory.path().join(".aeronyx-relay-backups");
        let artifact = std::fs::read_dir(&backup_directory)
            .expect("read recovery directory")
            .next()
            .expect("one recovery image")
            .expect("valid recovery entry")
            .path();
        std::fs::write(&artifact, b"corrupt-retention-fixture")
            .expect("corrupt managed recovery fixture");

        let error = source
            .audit_verified_backup_retention()
            .expect_err("corrupt managed recovery image must fail audit");
        assert_eq!(error.reason_bucket(), "corrupt_stored_data");
        assert_eq!(
            std::fs::read(&artifact).expect("read preserved corrupt fixture"),
            b"corrupt-retention-fixture"
        );
        assert_eq!(
            std::fs::read_dir(&backup_directory)
                .expect("inspect preserved corrupt directory")
                .count(),
            1
        );
    }

    #[test]
    fn backup_prune_dry_run_plans_without_deleting_and_writes_private_audit() {
        // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Dry-run is the default
        // operator experience and must leave every recovery artifact intact.
        let directory = tempfile::tempdir().expect("backup prune dry-run directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 2;
        config.custody_backup_retention_target_bytes = u64::MAX;
        let secret = [0xa1u8; 32];
        let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        for operation in ["prune-dry-1", "prune-dry-2", "prune-dry-3"] {
            source
                .create_verified_backup_for_operation(operation)
                .expect("create recovery image");
        }
        let partial = directory
            .path()
            .join(".aeronyx-relay-backups")
            .join(".relay-custody-1800000000-0123456789abcdef.tmp");
        std::fs::write(&partial, b"interrupted-private-snapshot")
            .expect("install interrupted snapshot fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600))
                .expect("restrict partial fixture");
        }

        let receipt = ChatRelayService::prune_verified_backup_retention_at(
            &config,
            &secret,
            &ChatRelayBackupPruneRequest::default(),
            now_secs() + config.custody_backup_partial_grace_secs + 1,
        )
        .expect("complete retention dry-run");
        assert!(!receipt.executed);
        assert_eq!(receipt.planned_backup_count, 1);
        assert_eq!(receipt.planned_partial_count, 1);
        assert_eq!(receipt.deleted_backup_count, 0);
        assert_eq!(receipt.deleted_partial_count, 0);
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect dry-run artifacts")
                .count(),
            4
        );
        let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        let mut audit = ChatRelayService::open_private_backup_control_file(&audit_path, true)
            .expect("open private maintenance audit");
        assert_eq!(
            ChatRelayService::verify_backup_audit_log(&mut audit, &secret)
                .expect("verify maintenance audit")
                .0,
            1
        );
    }

    #[test]
    fn backup_prune_execution_requires_both_confirmations_without_side_effects() {
        let directory = tempfile::tempdir().expect("backup prune confirmation directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 1;
        let secret = [0xa2u8; 32];
        let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        for operation in ["prune-confirm-1", "prune-confirm-2"] {
            source
                .create_verified_backup_for_operation(operation)
                .expect("create recovery image");
        }

        for request in [
            ChatRelayBackupPruneRequest {
                execute: true,
                confirmation: None,
                node_stopped_confirmed: true,
            },
            ChatRelayBackupPruneRequest {
                execute: true,
                confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
                node_stopped_confirmed: false,
            },
        ] {
            let error = ChatRelayService::prune_verified_backup_retention_at(
                &config,
                &secret,
                &request,
                now_secs(),
            )
            .expect_err("incomplete confirmation must fail closed");
            assert_eq!(error.reason_bucket(), "sqlite_error");
        }
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect confirmed artifacts")
                .count(),
            2
        );
        assert!(!directory
            .path()
            .join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME)
            .exists());
    }

    #[test]
    fn backup_prune_deletes_only_excess_and_grace_expired_artifacts() {
        let directory = tempfile::tempdir().expect("backup prune execution directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 2;
        config.custody_backup_retention_target_bytes = u64::MAX;
        let secret = [0xa3u8; 32];
        let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        for operation in ["prune-execute-1", "prune-execute-2", "prune-execute-3"] {
            source
                .create_verified_backup_for_operation(operation)
                .expect("create recovery image");
        }
        let backup_directory = directory.path().join(".aeronyx-relay-backups");
        let stale_partial = backup_directory.join(".relay-custody-1-0123456789abcdef.tmp");
        let fresh_partial = backup_directory.join(".relay-custody-2-fedcba9876543210.tmp");
        for partial in [&stale_partial, &fresh_partial] {
            std::fs::write(partial, b"private-partial").expect("install partial fixture");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(partial, std::fs::Permissions::from_mode(0o600))
                    .expect("restrict partial fixture");
            }
        }
        let now = now_secs();
        let stale_now = now + config.custody_backup_partial_grace_secs + 1;
        let request = ChatRelayBackupPruneRequest {
            execute: true,
            confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
            node_stopped_confirmed: true,
        };

        // Both partial fixtures have the same current mtime, so first prove a
        // normal execution treats neither as stale.
        let fresh_receipt =
            ChatRelayService::prune_verified_backup_retention_at(&config, &secret, &request, now)
                .expect("prune excess complete backup only");
        assert_eq!(fresh_receipt.deleted_backup_count, 1);
        assert_eq!(fresh_receipt.deleted_partial_count, 0);
        assert!(stale_partial.exists());
        assert!(fresh_partial.exists());

        // Advance the internal test clock beyond the mandatory grace period.
        let stale_receipt = ChatRelayService::prune_verified_backup_retention_at(
            &config, &secret, &request, stale_now,
        )
        .expect("prune grace-expired partials");
        assert_eq!(stale_receipt.deleted_backup_count, 0);
        assert_eq!(stale_receipt.deleted_partial_count, 2);
        assert_eq!(stale_receipt.remaining.retained_count, 2);
        assert_eq!(stale_receipt.remaining.excess_count, 0);
        assert!(!stale_partial.exists());
        assert!(!fresh_partial.exists());

        let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        let mut audit = ChatRelayService::open_private_backup_control_file(&audit_path, true)
            .expect("open private maintenance audit");
        assert_eq!(
            ChatRelayService::verify_backup_audit_log(&mut audit, &secret)
                .expect("verify maintenance audit")
                .0,
            4,
            "two successful executions write planned and completed records"
        );
    }

    #[test]
    fn backup_prune_rejects_tampered_audit_before_deletion() {
        let directory = tempfile::tempdir().expect("backup prune tamper directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        config.custody_backup_retention_target_artifacts = 1;
        let secret = [0xa4u8; 32];
        let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        for operation in ["prune-tamper-1", "prune-tamper-2"] {
            source
                .create_verified_backup_for_operation(operation)
                .expect("create recovery image");
        }
        ChatRelayService::prune_verified_backup_retention_at(
            &config,
            &secret,
            &ChatRelayBackupPruneRequest::default(),
            now_secs(),
        )
        .expect("write dry-run audit");
        let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        let encoded = std::fs::read_to_string(&audit_path).expect("read audit fixture");
        std::fs::write(&audit_path, encoded.replace("dry_run", "completed"))
            .expect("tamper audit fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&audit_path, std::fs::Permissions::from_mode(0o600))
                .expect("restore private audit permissions");
        }
        let request = ChatRelayBackupPruneRequest {
            execute: true,
            confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
            node_stopped_confirmed: true,
        };
        let error = ChatRelayService::prune_verified_backup_retention_at(
            &config,
            &secret,
            &request,
            now_secs(),
        )
        .expect_err("tampered audit must block deletion");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        assert_eq!(
            std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
                .expect("inspect preserved recovery images")
                .count(),
            2
        );
    }

    #[test]
    fn backup_prune_fails_closed_while_cross_process_lock_is_held() {
        let directory = tempfile::tempdir().expect("backup prune lock directory");
        let source_path = directory.path().join("source.sqlite");
        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let secret = [0xa5u8; 32];
        let _source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
        let backup_directory = ChatRelayService::private_backup_directory_for_config(&config)
            .expect("backup boundary");
        let _held = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold maintenance lock");

        let error = ChatRelayService::prune_verified_backup_retention_at(
            &config,
            &secret,
            &ChatRelayBackupPruneRequest::default(),
            now_secs(),
        )
        .expect_err("held lock must block concurrent maintenance");
        assert_eq!(error.reason_bucket(), "sqlite_error");
    }

    #[cfg(unix)]
    #[test]
    fn verified_backup_rejects_symlinked_storage_boundary() {
        use std::os::unix::fs::symlink;

        // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] A process with
        // access to the database parent must not redirect ciphertext custody
        // through a pre-created symlink before an operator backup runs.
        let directory = tempfile::tempdir().expect("symlink backup boundary directory");
        let source_path = directory.path().join("source.sqlite");
        let outside = directory.path().join("redirect-target");
        std::fs::create_dir(&outside).expect("create redirect target");
        symlink(&outside, directory.path().join(".aeronyx-relay-backups"))
            .expect("install backup boundary symlink");

        let mut config = test_config();
        config.db_path = source_path.to_string_lossy().into_owned();
        let source = make_service_with_config(config);
        let error = source
            .create_verified_backup()
            .expect_err("symlinked backup boundary must fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        assert_eq!(
            std::fs::read_dir(&outside)
                .expect("inspect redirect target")
                .count(),
            0
        );
    }

    #[test]
    fn test_peer_relay_outbound_health_tracks_failure_and_recovery() {
        let svc = make_service();

        svc.record_peer_relay_outbound(
            1_800_000_010,
            2,
            1,
            Some("peer_relay_request_timeout".to_string()),
        );
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("degraded"));
        assert_eq!(status.last_outbound_attempted, 2);
        assert_eq!(status.last_outbound_accepted, 1);
        assert_eq!(status.last_outbound_failed, 1);
        assert_eq!(status.consecutive_outbound_failures, 0);
        assert_eq!(status.last_outbound_success_at, Some(1_800_000_010));

        svc.record_peer_relay_outbound(
            1_800_000_020,
            1,
            0,
            Some("peer_relay_http_503".to_string()),
        );
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("failed"));
        assert_eq!(
            status.last_outbound_failure_reason.as_deref(),
            Some("peer_relay_http_503")
        );
        assert_eq!(status.consecutive_outbound_failures, 1);

        svc.record_peer_relay_outbound(1_800_000_030, 1, 1, None);
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("healthy"));
        assert_eq!(status.last_outbound_failure_reason, None);
        assert_eq!(status.consecutive_outbound_failures, 0);
        assert_eq!(status.last_outbound_success_at, Some(1_800_000_030));
        assert_eq!(status.direct_peer_outbound.rounds, 3);
        assert_eq!(
            status.direct_peer_outbound.last_status.as_deref(),
            Some("healthy")
        );
        assert_eq!(status.authenticated_onion_outbound.rounds, 0);
    }

    #[test]
    fn authenticated_onion_health_survives_direct_fallback_result() {
        let svc = make_service();

        // [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] This reproduces the
        // production order: receipt-verified onion fails, then compatibility
        // direct relay succeeds for availability. Aggregate remains backward
        // compatible while the authenticated proof keeps its true result.
        svc.record_authenticated_onion_outbound(
            1_800_000_040,
            1,
            0,
            Some("onion_delivery_receipt_rejected".to_string()),
        );
        svc.record_peer_relay_outbound(1_800_000_041, 1, 1, None);

        let status = svc.peer_status();
        assert_eq!(status.outbound_rounds, 2);
        assert_eq!(status.last_outbound_status.as_deref(), Some("healthy"));
        assert_eq!(status.authenticated_onion_outbound.rounds, 1);
        assert_eq!(
            status.authenticated_onion_outbound.last_status.as_deref(),
            Some("failed")
        );
        assert_eq!(
            status
                .authenticated_onion_outbound
                .last_failure_reason
                .as_deref(),
            Some("onion_delivery_receipt_rejected")
        );
        assert_eq!(status.direct_peer_outbound.rounds, 1);
        assert_eq!(
            status.direct_peer_outbound.last_status.as_deref(),
            Some("healthy")
        );
    }

    #[test]
    fn peer_health_deserializes_pre_route_class_snapshot() {
        let svc = make_service();
        svc.record_peer_relay_outbound(1_800_000_050, 1, 1, None);
        let mut encoded = serde_json::to_value(svc.peer_status()).expect("serialize peer status");
        let object = encoded.as_object_mut().expect("peer status JSON object");
        object.remove("custody_durability");
        object.remove("authenticated_onion_outbound");
        object.remove("direct_peer_outbound");
        object.remove("direct_peer_retry");

        // [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Additive health
        // fields must not invalidate status cached or forwarded by an older
        // node during a rolling upgrade.
        let decoded: ChatRelayPeerStatus =
            serde_json::from_value(encoded).expect("deserialize legacy peer status");
        assert_eq!(decoded.outbound_rounds, 1);
        assert_eq!(decoded.last_outbound_status.as_deref(), Some("healthy"));
        assert_eq!(
            decoded.custody_durability,
            ChatRelayCustodyDurabilityStatus::default()
        );
        assert_eq!(decoded.authenticated_onion_outbound.rounds, 0);
        assert_eq!(decoded.direct_peer_outbound.rounds, 0);
        assert_eq!(
            decoded.direct_peer_retry,
            ChatRelayDirectPeerRetryStatus::default()
        );
    }

    #[test]
    fn direct_peer_retry_health_preserves_nested_rolling_compatibility() {
        let svc = make_service();
        complete_direct_peer_test_delivery(&svc, 1_800_000_051, true, true, false);
        let mut encoded = serde_json::to_value(svc.peer_status()).expect("serialize peer status");
        let retry = encoded
            .get_mut("direct_peer_retry")
            .and_then(serde_json::Value::as_object_mut)
            .expect("direct peer retry JSON object");
        let recent = retry
            .get("recent_window")
            .and_then(serde_json::Value::as_object)
            .expect("recent delivery SLO JSON object");
        assert_eq!(recent.get("window_seconds"), Some(&serde_json::json!(300)));
        assert_eq!(recent.get("deliveries_total"), Some(&serde_json::json!(1)));
        assert_eq!(recent.get("status"), Some(&serde_json::json!("healthy")));
        let circuit = retry
            .get("circuit")
            .and_then(serde_json::Value::as_object)
            .expect("direct relay circuit JSON object");
        assert_eq!(circuit.get("state"), Some(&serde_json::json!("closed")));
        assert_eq!(
            circuit.get("restart_protected"),
            Some(&serde_json::json!(true))
        );
        retry.remove("recent_window");
        retry.remove("circuit");

        // [DIRECT-RELAY-RETRY-SLO 2026-08-15 by Codex] Nodes may first learn
        // the lifetime retry counters and only later learn the rolling SLO.
        // Missing additive nested health must therefore deserialize as idle.
        let decoded: ChatRelayPeerStatus =
            serde_json::from_value(encoded).expect("deserialize pre-SLO peer status");
        assert_eq!(decoded.direct_peer_retry.retry_triggered_total, 1);
        assert_eq!(
            decoded.direct_peer_retry.recent_window,
            ChatRelayDirectPeerSloStatus::default()
        );
        assert_eq!(
            decoded.direct_peer_retry.circuit,
            ChatRelayDirectPeerCircuitStatus::default()
        );
    }

    #[test]
    fn direct_peer_retry_health_tracks_recovery_exhaustion_and_determinism() {
        let svc = make_service();

        // [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] A normal first
        // attempt is invisible to lifetime exception counters but remains in
        // the recent delivery SLO denominator.
        complete_direct_peer_test_delivery(&svc, 1_800_000_001, false, true, false);
        let retry = svc.peer_status().direct_peer_retry;
        assert_eq!(retry.retry_triggered_total, 0);
        assert_eq!(retry.retry_recovered_total, 0);
        assert_eq!(retry.retry_exhausted_total, 0);
        assert_eq!(retry.deterministic_failure_total, 0);
        assert_eq!(retry.recent_window.deliveries_total, 1);
        assert_eq!(retry.recent_window.delivered_total, 1);
        assert_eq!(retry.recent_window.delivery_success_bps, Some(10_000));
        assert_eq!(retry.recent_window.status, "healthy");

        complete_direct_peer_test_delivery(&svc, 1_800_000_010, true, true, false);
        complete_direct_peer_test_delivery(&svc, 1_800_000_020, true, false, true);
        complete_direct_peer_test_delivery(&svc, 1_800_000_030, false, false, true);

        let retry = svc.peer_status().direct_peer_retry;
        assert_eq!(retry.retry_triggered_total, 2);
        assert_eq!(retry.retry_recovered_total, 1);
        assert_eq!(retry.retry_exhausted_total, 1);
        assert_eq!(retry.deterministic_failure_total, 2);
        assert_eq!(
            retry.retry_recovered_total + retry.retry_exhausted_total,
            retry.retry_triggered_total
        );
        assert_eq!(retry.last_outcome.as_deref(), Some("deterministic_failure"));
        assert_eq!(retry.last_at, Some(1_800_000_030));
        assert_eq!(retry.recent_window.deliveries_total, 4);
        assert_eq!(retry.recent_window.delivered_total, 2);
        assert_eq!(retry.recent_window.failed_total, 2);
        assert_eq!(retry.recent_window.delivery_success_bps, Some(5_000));
        assert_eq!(retry.recent_window.retry_recovery_bps, Some(5_000));
        assert_eq!(retry.recent_window.meets_slo, Some(false));
        assert_eq!(retry.recent_window.status, "degraded");
    }

    #[test]
    fn direct_peer_retry_slo_window_expires_and_requires_repeated_failure() {
        let mut window = DirectPeerRetrySloWindow::default();
        let base = 1_800_000_000;

        window.record(base, false, true, false);
        window.record(base + 1, true, true, false);
        let healthy = window.snapshot(base + 2);
        assert_eq!(healthy.deliveries_total, 2);
        assert_eq!(healthy.delivery_success_bps, Some(10_000));
        assert_eq!(healthy.retry_recovery_bps, Some(10_000));
        assert_eq!(healthy.meets_slo, Some(true));
        assert_eq!(healthy.status, "healthy");

        window.record(base + 61, false, false, true);
        let one_failure = window.snapshot(base + 61);
        assert_eq!(one_failure.failed_total, 1);
        assert_eq!(one_failure.status, "degraded");

        window.record(base + 62, true, false, false);
        window.record(base + 63, true, false, true);
        let failed = window.snapshot(base + 64);
        assert_eq!(failed.deliveries_total, 5);
        assert_eq!(failed.delivered_total, 2);
        assert_eq!(failed.failed_total, 3);
        assert_eq!(failed.delivery_success_bps, Some(4_000));
        assert_eq!(failed.retry_triggered_total, 3);
        assert_eq!(failed.retry_recovered_total, 1);
        assert_eq!(failed.retry_exhausted_total, 2);
        assert_eq!(failed.deterministic_failure_total, 2);
        assert_eq!(failed.status, "failed");

        let expired = window.snapshot(base + DIRECT_PEER_RETRY_SLO_WINDOW_SECS + 61);
        assert_eq!(expired.deliveries_total, 0);
        assert_eq!(expired.delivery_success_bps, None);
        assert_eq!(expired.meets_slo, None);
        assert_eq!(expired.status, "idle");
    }

    #[test]
    fn direct_peer_circuit_opens_half_opens_and_requires_two_successes() {
        let svc = make_service();
        let base = 1_800_000_100;

        for offset in 0..3 {
            complete_direct_peer_test_delivery(&svc, base + offset, false, false, true);
        }
        let opened = svc.direct_peer_relay_circuit.lock().snapshot(base + 2);
        assert_eq!(opened.state, "open");
        assert_eq!(opened.opened_total, 1);
        assert_eq!(
            opened.open_remaining_seconds,
            Some(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
        );
        assert!(svc.begin_direct_peer_delivery(base + 3).is_none());

        let first_probe = svc
            .begin_direct_peer_delivery(base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
            .expect("cooldown expiry should admit one half-open probe");
        assert!(first_probe.is_half_open());
        assert!(svc
            .begin_direct_peer_delivery(base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
            .is_none());
        svc.complete_direct_peer_delivery(base + 33, first_probe, false, true, false);
        let recovering = svc.direct_peer_relay_circuit.lock().snapshot(base + 33);
        assert_eq!(recovering.state, "half_open");
        assert_eq!(recovering.half_open_consecutive_successes, 1);

        // A duplicate late outcome for the consumed permit cannot overwrite
        // the newer generation or count as a failed recovery probe.
        svc.complete_direct_peer_delivery(base + 34, first_probe, false, false, true);
        let after_stale = svc.direct_peer_relay_circuit.lock().snapshot(base + 34);
        assert_eq!(after_stale.state, "half_open");
        assert_eq!(after_stale.half_open_failed_total, 0);
        assert_eq!(
            svc.direct_peer_retry_slo
                .lock()
                .snapshot(base + 34)
                .deliveries_total,
            4
        );

        let second_probe = svc
            .begin_direct_peer_delivery(base + 34)
            .expect("second recovery proof should be admitted serially");
        assert!(second_probe.is_half_open());
        svc.complete_direct_peer_delivery(base + 34, second_probe, false, true, false);
        let recovered = svc.direct_peer_relay_circuit.lock().snapshot(base + 34);
        assert_eq!(recovered.state, "closed");
        assert_eq!(recovered.half_open_attempted_total, 2);
        assert_eq!(recovered.half_open_succeeded_total, 2);
        assert_eq!(recovered.recovered_total, 1);
        assert_eq!(recovered.blocked_total, 2);

        // The previous failed SLO is not enough to reopen by itself, but one
        // new failed delivery while that window remains failed is.
        complete_direct_peer_test_delivery(&svc, base + 35, false, false, true);
        let reopened = svc.direct_peer_relay_circuit.lock().snapshot(base + 35);
        assert_eq!(reopened.state, "open");
        assert_eq!(reopened.opened_total, 2);
    }

    #[test]
    fn direct_peer_circuit_recovers_abandoned_half_open_permit_fail_closed() {
        let svc = make_service();
        let base = 1_800_000_200;
        for offset in 0..3 {
            complete_direct_peer_test_delivery(&svc, base + offset, false, false, true);
        }

        let first_probe_at = base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS;
        let abandoned = svc
            .begin_direct_peer_delivery(first_probe_at)
            .expect("expired open circuit should admit a half-open probe");
        assert!(abandoned.is_half_open());
        assert!(svc
            .begin_direct_peer_delivery(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS)
            .is_none());
        let timed_out = svc
            .direct_peer_relay_circuit
            .lock()
            .snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS);
        assert_eq!(timed_out.state, "open");
        assert_eq!(timed_out.opened_total, 2);
        assert_eq!(timed_out.half_open_failed_total, 1);

        // A later completion from the expired lease is stale and cannot close
        // the newly opened generation.
        svc.complete_direct_peer_delivery(
            first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1,
            abandoned,
            false,
            true,
            false,
        );
        assert_eq!(
            svc.direct_peer_relay_circuit
                .lock()
                .snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1)
                .state,
            "open"
        );
        assert_eq!(
            svc.direct_peer_retry_slo
                .lock()
                .snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1)
                .deliveries_total,
            3
        );
    }

    #[test]
    fn direct_peer_circuit_open_state_survives_restart() {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Restarting a node
        // during an outage must not reset target-bound admission to closed.
        let path = unique_test_db_path("direct-circuit-restart");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x73; 32]);
        let base = now_secs();
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            for offset in 0..3 {
                complete_direct_peer_test_delivery(
                    &svc,
                    base.saturating_add(offset),
                    false,
                    false,
                    true,
                );
            }
            let status = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(status.state, "open");
            assert!(status.restart_protected);
            assert_eq!(status.opened_total, 1);
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("restart relay");
            let status = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(status.state, "open");
            assert!(status.restart_protected);
            assert_eq!(status.opened_total, 1);
            assert!(status.checkpoint_loaded_at.is_some());
            assert!(svc
                .begin_direct_peer_delivery(base.saturating_add(3))
                .is_none());
        }
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_interrupted_half_open_probe_reopens_on_restart() {
        // A persisted in-flight probe has an unknowable outcome after process
        // loss, so startup classifies it failed and starts a fresh cooldown.
        let path = unique_test_db_path("direct-circuit-interrupted-probe");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x74; 32]);
        let now = now_secs();
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "UPDATE relay_direct_peer_circuit_checkpoint
                     SET state = 'half_open_in_flight',
                         successful_probes = 1,
                         deadline_at = ?1,
                         opened_total = 1,
                         half_open_attempted_total = 1,
                         half_open_succeeded_total = 1,
                         last_transition_at = ?2,
                         updated_at = ?2
                     WHERE singleton = 1",
                    params![
                        sqlite_integer(
                            now.saturating_add(DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS),
                            "test_probe_deadline"
                        )
                        .unwrap(),
                        sqlite_integer(now, "test_probe_updated_at").unwrap()
                    ],
                )
                .expect("seed interrupted probe checkpoint");
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("restart relay");
            let status = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(status.state, "open");
            assert!(status.restart_protected);
            assert_eq!(status.half_open_failed_total, 1);
            assert_eq!(status.opened_total, 2);
            assert_eq!(
                status.open_remaining_seconds,
                Some(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
            );
        }
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_half_open_progress_completes_across_restart() {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A completed probe
        // is safe to retain; restart may resume with one new serial probe but
        // may never treat an in-flight probe as completed.
        let path = unique_test_db_path("direct-circuit-half-open-progress");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x78; 32]);
        let now = now_secs();
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "UPDATE relay_direct_peer_circuit_checkpoint
                     SET state = 'half_open_ready',
                         successful_probes = 1,
                         opened_total = 1,
                         half_open_attempted_total = 1,
                         half_open_succeeded_total = 1,
                         last_transition_at = ?1,
                         updated_at = ?1
                     WHERE singleton = 1",
                    params![sqlite_integer(now, "test_half_open_ready_time").unwrap()],
                )
                .expect("seed completed first probe");
        }
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("restart relay");
            let restored = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(restored.state, "half_open");
            assert_eq!(restored.half_open_consecutive_successes, 1);
            let permit = svc
                .begin_direct_peer_delivery(now)
                .expect("restored circuit should admit the second probe");
            assert!(permit.is_half_open());
            assert!(svc.complete_direct_peer_delivery(now, permit, false, true, false));
            let closed = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(closed.state, "closed");
            assert_eq!(closed.recovered_total, 1);
            assert!(closed.restart_protected);
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("verify closed restart");
            let closed = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(closed.state, "closed");
            assert_eq!(closed.recovered_total, 1);
            assert!(closed.restart_protected);
        }
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_clock_rollback_recovers_fail_closed() {
        let path = unique_test_db_path("direct-circuit-clock-rollback");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x75; 32]);
        let now = now_secs();
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "UPDATE relay_direct_peer_circuit_checkpoint
                     SET updated_at = ?1
                     WHERE singleton = 1",
                    params![sqlite_integer(now.saturating_add(120), "test_future_time").unwrap()],
                )
                .expect("seed future checkpoint timestamp");
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("restart relay");
            let status = svc.peer_status().direct_peer_retry.circuit;
            assert_eq!(status.state, "open");
            assert!(status.restart_protected);
            assert_eq!(status.opened_total, 1);
            assert!(status
                .checkpoint_persisted_at
                .is_some_and(|value| value <= now_secs()));
        }
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_corrupt_checkpoint_rejects_relay_restart() {
        let path = unique_test_db_path("direct-circuit-corrupt");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x76; 32]);
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "UPDATE relay_direct_peer_circuit_checkpoint
                     SET state = 'closed', successful_probes = 1,
                         opened_total = 1,
                         half_open_attempted_total = 1,
                         half_open_succeeded_total = 1
                     WHERE singleton = 1",
                    [],
                )
                .expect("seed semantically corrupt checkpoint");
        }
        assert!(matches!(
            ChatRelayService::new(config, secret),
            Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_state"
            })
        ));
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_missing_checkpoint_rejects_existing_schema() {
        // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Only a first-time
        // schema upgrade may create the singleton. Its later disappearance is
        // corruption, not permission to reset an unknown circuit to closed.
        let path = unique_test_db_path("direct-circuit-missing");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x77; 32]);
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "DELETE FROM relay_direct_peer_circuit_checkpoint WHERE singleton = 1",
                    [],
                )
                .expect("delete checkpoint singleton");
        }
        assert!(matches!(
            ChatRelayService::new(config, secret),
            Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_singleton"
            })
        ));
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_missing_checkpoint_table_rejects_installed_schema() {
        // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The installation
        // marker distinguishes destructive table loss from a first upgrade.
        // Restart must not manufacture a closed checkpoint after that loss.
        let path = unique_test_db_path("direct-circuit-missing-table");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x78; 32]);
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
                .expect("remove installed checkpoint table");
        }
        assert!(matches!(
            ChatRelayService::new(config, secret),
            Err(ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_table"
            })
        ));
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_existing_checkpoint_installs_missing_schema_sentinel() {
        // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Deployed v2.3
        // databases already have a validated checkpoint but no feature marker.
        // Their first v2.4 startup installs the marker in the same transaction.
        let path = unique_test_db_path("direct-circuit-marker-upgrade");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x79; 32]);
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            svc.conn
                .lock()
                .execute(
                    "DELETE FROM relay_schema_features WHERE feature = ?1",
                    params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                )
                .expect("simulate pre-sentinel database");
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("upgrade existing relay");
            let installed_version = svc
                .conn
                .lock()
                .query_row(
                    "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                    params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                    |row| row.get::<_, i64>(0),
                )
                .expect("read installed schema sentinel");
            assert_eq!(
                installed_version,
                DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION
            );
        }
        remove_test_db(&path);
    }

    #[test]
    fn direct_peer_circuit_runtime_checkpoint_failure_denies_delivery() {
        let svc = make_service();
        let base = now_secs();
        svc.conn
            .lock()
            .execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
            .expect("remove checkpoint table");

        for offset in 0..3 {
            complete_direct_peer_test_delivery(
                &svc,
                base.saturating_add(offset),
                false,
                false,
                true,
            );
        }
        let status = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(!status.restart_protected);
        assert_eq!(status.checkpoint_failures_total, 1);
        assert_eq!(status.last_checkpoint_failure_at, Some(base + 2));
        assert!(svc
            .begin_direct_peer_delivery(
                base.saturating_add(2)
                    .saturating_add(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
            )
            .is_none());
        let blocked = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(blocked.state, "open");
        assert!(!blocked.restart_protected);
        assert_eq!(blocked.checkpoint_failures_total, 2);
    }

    #[test]
    fn test_peer_relay_inbound_health_tracks_accept_and_reject() {
        let svc = make_service();

        svc.record_peer_relay_inbound_accepted(1_800_000_010, false, 0, true);
        let status = svc.peer_status();
        assert_eq!(status.inbound_accepted_total, 1);
        assert_eq!(status.inbound_stored_pending_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("accepted"));
        assert_eq!(status.last_inbound_failure_reason, None);

        svc.record_peer_relay_inbound_accepted(1_800_000_020, true, 0, false);
        let status = svc.peer_status();
        assert_eq!(status.inbound_accepted_total, 2);
        assert_eq!(status.inbound_duplicate_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("duplicate"));

        svc.record_peer_relay_inbound_rejected(1_800_000_030, "invalid_signature");
        let status = svc.peer_status();
        assert_eq!(status.inbound_rejected_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("rejected"));
        assert_eq!(
            status.last_inbound_failure_reason.as_deref(),
            Some("invalid_signature")
        );
    }

    fn make_envelope(kp: &IdentityKeyPair, receiver: [u8; 32]) -> ChatEnvelope {
        let mut env = ChatEnvelope {
            message_id: rand::random(),
            sender: kp.public_key_bytes(),
            receiver,
            timestamp: now_secs(),
            ciphertext: b"encrypted".to_vec(),
            nonce: [0x02; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        let data = env.sign_data();
        env.signature = kp.sign(&data);
        env
    }

    fn make_session() -> SessionId {
        SessionId::from_bytes(&rand::random::<[u8; 16]>())
            .expect("random bytes form valid SessionId")
    }

    fn make_addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{}", port).parse().unwrap()
    }

    // ── Schema init ──────────────────────────────────────────────────────

    #[test]
    fn test_service_init() {
        let svc = make_service();
        let (m, b) = svc.run_cleanup().expect("cleanup");
        assert_eq!(m, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_legacy_queue_sequence_migration_is_atomic_and_restart_stable() {
        let path = unique_test_db_path("sequence-migration");
        {
            let conn = Connection::open(&path).expect("open legacy database");
            conn.execute_batch(
                "CREATE TABLE pending_messages (
                    message_id  BLOB(16) PRIMARY KEY,
                    sender      BLOB(32) NOT NULL,
                    receiver    BLOB(32) NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    envelope    BLOB NOT NULL,
                    received_at INTEGER NOT NULL,
                    status      INTEGER NOT NULL DEFAULT 0
                );",
            )
            .expect("create legacy pending schema");
            for marker in [0x11_u8, 0x22] {
                conn.execute(
                    "INSERT INTO pending_messages
                     (message_id, sender, receiver, timestamp, envelope, received_at, status)
                     VALUES (?1, ?2, ?3, 1, ?4, 1, 0)",
                    params![
                        [marker; 16].as_slice(),
                        [0x31_u8; 32].as_slice(),
                        [0x41_u8; 32].as_slice(),
                        [0x51_u8].as_slice(),
                    ],
                )
                .expect("insert legacy row");
            }
        }

        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x42; 32]);
        {
            let svc = ChatRelayService::new(config.clone(), secret).expect("migrate legacy queue");
            let conn = svc.conn.lock();
            let mut stmt = conn
                .prepare("SELECT queue_sequence FROM pending_messages ORDER BY rowid")
                .expect("prepare migrated sequence query");
            let sequences: Vec<i64> = stmt
                .query_map([], |row| row.get(0))
                .expect("query migrated sequences")
                .collect::<Result<Vec<_>, _>>()
                .expect("collect migrated sequences");
            assert_eq!(sequences, vec![1, 2]);
            let last: i64 = conn
                .query_row(
                    "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
                    [],
                    |row| row.get(0),
                )
                .expect("read migrated high-water mark");
            assert_eq!(last, 2);
        }
        {
            let svc = ChatRelayService::new(config, secret).expect("reopen migrated queue");
            let conn = svc.conn.lock();
            let mut stmt = conn
                .prepare("SELECT queue_sequence FROM pending_messages ORDER BY rowid")
                .expect("prepare restart sequence query");
            let sequences: Vec<i64> = stmt
                .query_map([], |row| row.get(0))
                .expect("query restart sequences")
                .collect::<Result<Vec<_>, _>>()
                .expect("collect restart sequences");
            assert_eq!(sequences, vec![1, 2]);
        }
        remove_test_db(&path);
    }

    #[test]
    fn chat_relay_logs_stay_free_of_routing_identifiers() {
        let source = include_str!("chat_relay.rs");
        let message_log = concat!("id = %hex::", "encode(envelope.message_id)");
        let receiver_log = concat!("receiver = %hex::", "encode");
        let sender_log = concat!("sender = %hex::", "encode");
        let blob_log = concat!("blob_id", " = %");

        for forbidden in [message_log, receiver_log, sender_log, blob_log] {
            assert!(
                !source.contains(forbidden),
                "relay logs must not expose stable routing identifiers"
            );
        }
    }

    // ── v1.3.0: wallet_routes field accessible ───────────────────────────

    #[test]
    fn test_wallet_routes_field_accessible() {
        let svc = make_service();
        let wallet = [0xAAu8; 32];
        let sid = make_session();
        let addr = make_addr(9000);

        // announce via the public field
        svc.wallet_routes.announce(&wallet, sid.clone(), addr);

        let results = svc.wallet_routes.lookup(&wallet);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, sid);
    }

    #[test]
    fn test_wallet_routes_arc_clone_shares_state() {
        let svc = make_service();
        let routes_clone = Arc::clone(&svc.wallet_routes);

        let wallet = [0xBBu8; 32];
        let sid = make_session();

        // announce via original
        svc.wallet_routes
            .announce(&wallet, sid.clone(), make_addr(9001));

        // lookup via clone — must see the same entry
        let results = routes_clone.lookup(&wallet);
        assert_eq!(
            results.len(),
            1,
            "Arc clone must share the same underlying cache"
        );
    }

    // ── store → pull → ack (preserved) ───────────────────────────────────

    #[test]
    fn test_store_pull_ack_roundtrip() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);
        let mid = env.message_id;

        svc.store_pending(&env).expect("store");
        let usage = svc.storage_usage().expect("usage after store");
        assert_eq!(usage.pending_messages, 1);
        assert!(usage.pending_message_bytes > 0);

        let (msgs, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
        assert!(!has_more);
        assert_eq!(msgs[0].message_id, mid);

        let deleted = svc.ack_messages(&[mid], &receiver).expect("ack");
        assert_eq!(deleted, 1);
        let usage = svc.storage_usage().expect("usage after ack");
        assert_eq!(usage.pending_messages, 0);
        assert_eq!(usage.pending_message_bytes, 0);

        let (msgs2, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull2");
        assert!(msgs2.is_empty());
    }

    #[test]
    fn test_pull_isolates_malformed_row_and_delivers_valid_message() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBEu8; 32];
        let envelope = make_envelope(&kp, receiver);
        let expected_message_id = envelope.message_id;
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 1, ?4, 1, 0)",
                params![
                    [0x01u8; 15].as_slice(),
                    kp.public_key_bytes().as_slice(),
                    receiver.as_slice(),
                    [0xFFu8].as_slice(),
                ],
            )
            .expect("insert malformed pending row");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull with poison-row isolation");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
        assert_eq!(messages[0].message_id, expected_message_id);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

        let status = svc.maintenance_status();
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        let event: (String, String, i64) = svc
            .conn
            .lock()
            .query_row(
                "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read pending quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
        assert_eq!(event.1, "pending_message_id");
        assert_eq!(event.2, 1);
    }

    #[test]
    fn test_pull_quarantines_message_id_envelope_mismatch() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBFu8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET message_id = ?1 WHERE message_id = ?2",
                params![[0xFEu8; 16].as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper durable message id");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull mismatched durable row");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read mismatch reason");
        assert_eq!(reason, "pending_message_id_mismatch");
    }

    #[test]
    fn test_pull_quarantines_stored_sender_mismatch_before_delivery() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC2u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = ?1 WHERE message_id = ?2",
                params![[0xF1u8; 32].as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper durable sender");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull mismatched durable sender");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read mismatch reason");
        assert_eq!(reason, "pending_message_sender_mismatch");
    }

    #[test]
    fn test_pull_quarantines_invalid_durable_signature() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC3u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        let mut tampered = envelope.clone();
        tampered.signature[0] ^= 0xFF;
        let tampered_bytes = encode_envelope(&tampered).expect("encode tampered envelope");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET envelope = ?1 WHERE message_id = ?2",
                params![tampered_bytes, envelope.message_id.as_slice()],
            )
            .expect("tamper durable signature");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull invalid durable signature");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read signature reason");
        assert_eq!(reason, "pending_message_signature");
    }

    #[test]
    fn test_store_rejects_timestamp_outside_sqlite_domain() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBCu8; 32];
        let mut envelope = make_envelope(&kp, receiver);
        envelope.timestamp = u64::MAX;
        envelope.signature = kp.sign(&envelope.sign_data());

        let error = svc
            .store_pending(&envelope)
            .expect_err("out-of-range timestamp must be rejected");
        assert!(matches!(error, ChatRelayError::TimestampOutOfRange));
        assert_eq!(error.reason_bucket(), "timestamp_out_of_range");
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);
    }

    #[test]
    fn test_pull_out_of_range_timestamp_fails_closed() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBDu8; 32];
        let mut envelope = make_envelope(&kp, receiver);
        envelope.timestamp = 1;
        envelope.signature = kp.sign(&envelope.sign_data());
        svc.store_pending(&envelope).expect("store pending message");

        let (messages, has_more) = svc
            .pull_pending(&receiver, u64::MAX, &[0u8; 16], 50)
            .expect("bounded pull");
        assert!(messages.is_empty());
        assert!(!has_more);
    }

    #[test]
    fn test_store_duplicate_ignored() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);

        svc.store_pending(&env).expect("first store");
        svc.store_pending(&env)
            .expect("duplicate store — should not error");

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_store_rejects_message_id_conflict_without_replacing_original() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC0u8; 32];
        let original = make_envelope(&kp, receiver);
        let mut conflict = make_envelope(&kp, receiver);
        conflict.message_id = original.message_id;
        conflict.ciphertext = b"different ciphertext".to_vec();
        conflict.signature = kp.sign(&conflict.sign_data());

        svc.store_pending(&original).expect("store original");
        let error = svc
            .store_pending(&conflict)
            .expect_err("conflicting message id must fail");
        assert!(matches!(error, ChatRelayError::MessageIdConflict));
        assert_eq!(error.reason_bucket(), "message_id_conflict");
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull original");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
        assert_eq!(messages[0].envelope.ciphertext, original.ciphertext);
    }

    #[test]
    fn test_pull_zero_limit_makes_progress_with_minimum_page() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC1u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store message");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 0)
            .expect("zero limit pull");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
    }

    #[test]
    fn test_store_enforces_configured_ciphertext_size_limit() {
        let mut config = test_config();
        config.max_message_size = 4;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let envelope = make_envelope(&kp, [0x10; 32]);

        assert!(matches!(
            svc.store_pending(&envelope),
            Err(ChatRelayError::MessageTooLarge { size: 9, limit: 4 })
        ));
        assert_eq!(
            svc.storage_usage().unwrap(),
            ChatRelayStorageUsage::default()
        );
    }

    #[test]
    fn test_global_message_count_quota_preserves_duplicate_idempotence() {
        let mut config = test_config();
        config.max_pending_messages_total = 1;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let first = make_envelope(&kp, [0x11; 32]);

        svc.store_pending(&first).expect("first store");
        svc.store_pending(&first)
            .expect("duplicate remains successful at global capacity");

        let second = make_envelope(&kp, [0x22; 32]);
        assert!(matches!(
            svc.store_pending(&second),
            Err(ChatRelayError::PendingMessageQueueFull { .. })
        ));
        assert_eq!(svc.storage_usage().unwrap().pending_messages, 1);
    }

    #[test]
    fn test_global_message_byte_quota_spans_distinct_receivers() {
        let kp = IdentityKeyPair::generate();
        let first = make_envelope(&kp, [0x31; 32]);
        let encoded_bytes = encode_envelope(&first).unwrap().len() as u64;
        let mut config = test_config();
        config.max_pending_message_bytes_total = encoded_bytes;
        let svc = make_service_with_config(config);

        svc.store_pending(&first).expect("first store");
        let second = make_envelope(&kp, [0x32; 32]);
        assert!(matches!(
            svc.store_pending(&second),
            Err(ChatRelayError::PendingMessageBytesExceeded { .. })
        ));
    }

    #[test]
    fn test_storage_usage_reconciles_from_canonical_rows_on_restart() {
        let path = std::env::temp_dir().join(format!(
            "aeronyx-chat-relay-usage-{}.db",
            rand::random::<u64>()
        ));
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let kp = IdentityKeyPair::generate();
        let envelope = make_envelope(&kp, [0x61; 32]);

        {
            let svc = make_service_with_config(config.clone());
            svc.store_pending(&envelope).expect("store before restart");
        }
        {
            let conn = Connection::open(&path).expect("open usage database");
            conn.execute(
                "UPDATE relay_storage_usage
                 SET pending_message_count = 0, pending_message_bytes = 0
                 WHERE singleton = 1",
                [],
            )
            .expect("tamper derived usage row");
        }

        let restarted = make_service_with_config(config);
        let usage = restarted.storage_usage().expect("reconciled usage");
        assert_eq!(usage.pending_messages, 1);
        assert_eq!(
            usage.pending_message_bytes,
            encode_envelope(&envelope).unwrap().len() as u64
        );
        drop(restarted);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    #[test]
    fn storage_usage_rejects_negative_durable_counter() {
        // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] The previous
        // conversion silently mapped a negative tampered counter to zero,
        // which could disguise unavailable quota state as spare capacity.
        let svc = make_service();
        svc.conn
            .lock()
            .execute_batch(
                "PRAGMA ignore_check_constraints=ON;
                 UPDATE relay_storage_usage
                 SET pending_blob_bytes = -1
                 WHERE singleton = 1;",
            )
            .expect("install negative usage fixture");
        assert!(matches!(
            svc.storage_usage(),
            Err(ChatRelayError::CorruptStoredData {
                field: "pending_blob_bytes"
            })
        ));
    }

    #[test]
    fn test_mailbox_full_rejected() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];

        for _ in 0..5 {
            let env = make_envelope(&kp, receiver);
            svc.store_pending(&env).expect("store");
        }

        let env6 = make_envelope(&kp, receiver);
        let result = svc.store_pending(&env6);
        assert!(matches!(result, Err(ChatRelayError::MailboxFull { .. })));
    }

    #[test]
    fn test_ack_wrong_receiver_cannot_delete() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);
        let mid = env.message_id;

        svc.store_pending(&env).expect("store");

        let wrong_receiver = [0xCCu8; 32];
        let deleted = svc.ack_messages(&[mid], &wrong_receiver).expect("ack");
        assert_eq!(deleted, 0, "Wrong receiver must not delete messages");

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_ack_batch_is_atomic_and_deduplicated() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBD; 32];
        let first = make_envelope(&kp, receiver);
        let second = make_envelope(&kp, receiver);

        svc.store_pending(&first).expect("store first");
        svc.store_pending(&second).expect("store second");
        let deleted = svc
            .ack_messages(
                &[first.message_id, first.message_id, second.message_id],
                &receiver,
            )
            .expect("deduplicated ACK");

        assert_eq!(deleted, 2);
        assert_eq!(
            svc.storage_usage().expect("usage after ACK"),
            ChatRelayStorageUsage::default()
        );
    }

    #[test]
    fn test_ack_batch_above_protocol_ceiling_is_rejected() {
        let svc = make_service();
        let ids = vec![[0x11; 16]; MAX_CHAT_ACK_MESSAGE_IDS + 1];

        assert!(matches!(
            svc.ack_messages(&ids, &[0xBE; 32]),
            Err(ChatRelayError::AckBatchTooLarge {
                size,
                limit: MAX_CHAT_ACK_MESSAGE_IDS,
            }) if size == MAX_CHAT_ACK_MESSAGE_IDS + 1
        ));
    }

    // ── Pagination (preserved) ───────────────────────────────────────────

    #[test]
    fn test_pull_pagination() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];

        for _ in 0..5 {
            let env = make_envelope(&kp, receiver);
            svc.store_pending(&env).expect("store");
        }

        let (page1, has_more1) = svc.pull_pending(&receiver, 0, &[0u8; 16], 3).expect("p1");
        assert_eq!(page1.len(), 3);
        assert!(has_more1);

        let cursor = page1.last().unwrap().message_id;
        let (page2, has_more2) = svc.pull_pending(&receiver, 0, &cursor, 3).expect("p2");
        assert_eq!(page2.len(), 2);
        assert!(!has_more2);
    }

    #[test]
    fn test_pull_cursor_does_not_skip_rows_across_timestamps() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBC; 32];
        let fixtures = [
            (100, [0xF0; 16]),
            (200, [0xE0; 16]),
            (300, [0xD0; 16]),
            (400, [0xC0; 16]),
        ];

        for (timestamp, message_id) in fixtures {
            let mut envelope = make_envelope(&kp, receiver);
            envelope.timestamp = timestamp;
            envelope.message_id = message_id;
            envelope.signature = kp.sign(&envelope.sign_data());
            svc.store_pending(&envelope).expect("store ordered fixture");
        }

        let (first_page, first_has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 2)
            .expect("first cursor page");
        assert_eq!(first_page.len(), 2);
        assert!(first_has_more);

        let cursor = first_page.last().expect("first page cursor").message_id;
        let (second_page, second_has_more) = svc
            .pull_pending(&receiver, 0, &cursor, 2)
            .expect("second cursor page");
        assert_eq!(second_page.len(), 2);
        assert!(!second_has_more);

        let actual: HashSet<[u8; 16]> = first_page
            .iter()
            .chain(&second_page)
            .map(|message| message.message_id)
            .collect();
        let expected: HashSet<[u8; 16]> = fixtures
            .into_iter()
            .map(|(_, message_id)| message_id)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_queue_sequence_is_monotonic_and_idempotent_retries_do_not_consume_it() {
        let svc = make_service();
        let identity = IdentityKeyPair::generate();
        let receiver = [0xC4; 32];
        let first = make_envelope(&identity, receiver);
        let second = make_envelope(&identity, receiver);
        let third = make_envelope(&identity, receiver);

        svc.store_pending(&first).expect("store first");
        svc.store_pending(&second).expect("store second");
        svc.store_pending(&first).expect("retry first idempotently");
        svc.store_pending(&third).expect("store third");

        let conn = svc.conn.lock();
        let sequences: Vec<i64> = conn
            .prepare("SELECT queue_sequence FROM pending_messages ORDER BY queue_sequence")
            .expect("prepare sequence query")
            .query_map([], |row| row.get(0))
            .expect("query sequences")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect sequences");
        assert_eq!(sequences, vec![1, 2, 3]);
        let last: i64 = conn
            .query_row(
                "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .expect("read sequence high-water mark");
        assert_eq!(last, 3);
    }

    #[test]
    fn test_pull_v2_snapshot_excludes_concurrent_inserts_without_skipping() {
        let svc = make_service();
        let identity = IdentityKeyPair::generate();
        let receiver = [0xC5; 32];
        let initial: Vec<ChatEnvelope> =
            (0..3).map(|_| make_envelope(&identity, receiver)).collect();
        for envelope in &initial {
            svc.store_pending(envelope).expect("store snapshot fixture");
        }

        let first_page = svc
            .pull_pending_v2(&receiver, 0, &[], 2)
            .expect("first v2 snapshot page");
        assert_eq!(first_page.messages.len(), 2);
        assert!(first_page.has_more);
        assert_eq!(first_page.next_cursor.len(), CHAT_PULL_CURSOR_V2_BYTES);

        let concurrent = make_envelope(&identity, receiver);
        svc.store_pending(&concurrent)
            .expect("store concurrent post-snapshot message");
        let second_page = svc
            .pull_pending_v2(&receiver, 0, &first_page.next_cursor, 2)
            .expect("second v2 snapshot page");
        assert_eq!(second_page.messages.len(), 1);
        assert!(!second_page.has_more);

        let delivered: HashSet<[u8; 16]> = first_page
            .messages
            .iter()
            .chain(&second_page.messages)
            .map(|message| message.message_id)
            .collect();
        let expected: HashSet<[u8; 16]> =
            initial.iter().map(|envelope| envelope.message_id).collect();
        assert_eq!(delivered, expected);
        assert!(!delivered.contains(&concurrent.message_id));

        let delivered_ids: Vec<[u8; 16]> = delivered.into_iter().collect();
        svc.ack_messages(&delivered_ids, &receiver)
            .expect("ack completed snapshot");
        let fresh_snapshot = svc
            .pull_pending_v2(&receiver, 0, &[], 10)
            .expect("start fresh snapshot");
        assert_eq!(fresh_snapshot.messages.len(), 1);
        assert_eq!(fresh_snapshot.messages[0].message_id, concurrent.message_id);
    }

    #[test]
    fn test_pull_v2_cursor_rejects_tampering_and_binding_replay() {
        let svc = make_service();
        let identity = IdentityKeyPair::generate();
        let receiver = [0xC6; 32];
        for _ in 0..2 {
            svc.store_pending(&make_envelope(&identity, receiver))
                .expect("store cursor fixture");
        }
        let page = svc
            .pull_pending_v2(&receiver, 0, &[], 1)
            .expect("issue cursor");
        let decoded = svc
            .decode_pull_cursor_v2(&receiver, 0, &page.next_cursor)
            .expect("decode server-owned cursor in test");
        assert_eq!(
            decoded,
            PullCursorV2 {
                position: 1,
                ceiling: 2
            }
        );
        assert!(!page
            .next_cursor
            .windows(8)
            .any(|window| window == decoded.position.to_le_bytes()));
        assert!(!page
            .next_cursor
            .windows(8)
            .any(|window| window == decoded.ceiling.to_le_bytes()));

        let mut tampered = page.next_cursor.clone();
        let last = tampered.last_mut().expect("non-empty cursor");
        *last ^= 0x01;
        assert!(matches!(
            svc.pull_pending_v2(&receiver, 0, &tampered, 1),
            Err(ChatRelayError::InvalidPullCursor)
        ));
        assert!(matches!(
            svc.pull_pending_v2(&[0xC7; 32], 0, &page.next_cursor, 1),
            Err(ChatRelayError::InvalidPullCursor)
        ));
        assert!(matches!(
            svc.pull_pending_v2(&receiver, 1, &page.next_cursor, 1),
            Err(ChatRelayError::InvalidPullCursor)
        ));

        let foreign =
            ChatRelayService::new(test_config(), [0x91; 32]).expect("create foreign-key relay");
        assert!(matches!(
            foreign.pull_pending_v2(&receiver, 0, &page.next_cursor, 1),
            Err(ChatRelayError::InvalidPullCursor)
        ));
    }

    #[test]
    fn test_pull_v2_cursor_survives_restart_with_same_node_secret() {
        let path = unique_test_db_path("cursor-restart");
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let secret = derive_node_secret(&[0x62; 32]);
        let identity = IdentityKeyPair::generate();
        let receiver = [0xC8; 32];
        let cursor = {
            let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
            for _ in 0..3 {
                svc.store_pending(&make_envelope(&identity, receiver))
                    .expect("store restart fixture");
            }
            let page = svc
                .pull_pending_v2(&receiver, 0, &[], 2)
                .expect("issue pre-restart cursor");
            assert!(page.has_more);
            page.next_cursor
        };
        {
            let svc = ChatRelayService::new(config, secret).expect("restart relay");
            let page = svc
                .pull_pending_v2(&receiver, 0, &cursor, 2)
                .expect("resume cursor after restart");
            assert_eq!(page.messages.len(), 1);
            assert!(!page.has_more);
        }
        remove_test_db(&path);
    }

    #[test]
    fn test_pull_v2_quarantines_poison_row_and_advances_snapshot() {
        let svc = make_service();
        let identity = IdentityKeyPair::generate();
        let receiver = [0xC9; 32];
        let poison = make_envelope(&identity, receiver);
        let valid = make_envelope(&identity, receiver);
        svc.store_pending(&poison).expect("store poison fixture");
        svc.store_pending(&valid).expect("store valid fixture");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET envelope = ?1 WHERE message_id = ?2",
                params![[0xFF_u8].as_slice(), poison.message_id.as_slice()],
            )
            .expect("corrupt first sequence row");

        let first = svc
            .pull_pending_v2(&receiver, 0, &[], 1)
            .expect("pull through poison row");
        assert_eq!(first.messages.len(), 1);
        assert_eq!(first.messages[0].message_id, valid.message_id);
        assert!(first.has_more);
        let second = svc
            .pull_pending_v2(&receiver, 0, &first.next_cursor, 1)
            .expect("complete snapshot after poison quarantine");
        assert!(second.messages.is_empty());
        assert!(!second.has_more);
        assert_eq!(
            svc.maintenance_status().quarantined_pending_messages_total,
            1
        );
    }

    #[test]
    fn test_queue_sequence_exhaustion_rolls_back_message_insert() {
        let svc = make_service();
        svc.conn
            .lock()
            .execute(
                "UPDATE relay_queue_sequence SET last_sequence = ?1 WHERE singleton = 1",
                params![i64::MAX],
            )
            .expect("force sequence exhaustion");
        let identity = IdentityKeyPair::generate();
        let envelope = make_envelope(&identity, [0xCA; 32]);
        assert!(matches!(
            svc.store_pending(&envelope),
            Err(ChatRelayError::QueueSequenceExhausted)
        ));
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);
    }

    // ── Blob (preserved) ─────────────────────────────────────────────────

    #[test]
    fn test_blob_put_get_delete() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"encrypted_image_bytes";
        let file_hash: [u8; 32] = Sha256::digest(data).into();

        let blob_id = svc
            .put_blob(&sender, &receiver, data, &file_hash)
            .expect("put");
        assert_eq!(blob_id.len(), 32);
        let usage = svc.storage_usage().expect("usage after blob put");
        assert_eq!(usage.pending_blobs, 1);
        assert_eq!(usage.pending_blob_bytes, data.len() as u64);

        let fetched = svc.get_blob(&blob_id).expect("get");
        assert_eq!(fetched, data);

        svc.delete_blob(&blob_id, &sender).expect("delete");
        let usage = svc.storage_usage().expect("usage after blob delete");
        assert_eq!(usage.pending_blobs, 0);
        assert_eq!(usage.pending_blob_bytes, 0);
        assert!(matches!(
            svc.get_blob(&blob_id),
            Err(ChatRelayError::BlobNotFound { .. })
        ));
    }

    #[test]
    fn test_blob_too_large_rejected() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = vec![0u8; 2048];
        let file_hash: [u8; 32] = Sha256::digest(&data).into();

        let result = svc.put_blob(&sender, &receiver, &data, &file_hash);
        assert!(matches!(result, Err(ChatRelayError::BlobTooLarge { .. })));
    }

    #[test]
    fn test_global_blob_count_quota_preserves_duplicate_idempotence() {
        let mut config = test_config();
        config.max_pending_blobs_total = 1;
        let svc = make_service_with_config(config);
        let sender = [0x41; 32];
        let first_receiver = [0x42; 32];
        let first_data = b"first encrypted blob";
        let first_hash: [u8; 32] = Sha256::digest(first_data).into();

        let first_id = svc
            .put_blob(&sender, &first_receiver, first_data, &first_hash)
            .expect("first put");
        let duplicate_id = svc
            .put_blob(&sender, &first_receiver, first_data, &first_hash)
            .expect("duplicate remains successful at global capacity");
        assert_eq!(duplicate_id, first_id);

        let second_data = b"second encrypted blob";
        let second_hash: [u8; 32] = Sha256::digest(second_data).into();
        assert!(matches!(
            svc.put_blob(&sender, &[0x43; 32], second_data, &second_hash),
            Err(ChatRelayError::PendingBlobStoreFull { .. })
        ));
        assert_eq!(svc.storage_usage().unwrap().pending_blobs, 1);
    }

    #[test]
    fn test_global_blob_byte_quota_spans_distinct_receivers() {
        let data = b"bounded encrypted blob";
        let mut config = test_config();
        config.max_pending_blob_bytes_total = data.len() as u64;
        let svc = make_service_with_config(config);
        let sender = [0x51; 32];
        let first_hash: [u8; 32] = Sha256::digest(data).into();

        svc.put_blob(&sender, &[0x52; 32], data, &first_hash)
            .expect("first put");
        let second_hash: [u8; 32] = Sha256::digest(b"different hash").into();
        assert!(matches!(
            svc.put_blob(&sender, &[0x53; 32], data, &second_hash),
            Err(ChatRelayError::PendingBlobBytesExceeded { .. })
        ));
    }

    #[test]
    fn test_blob_delete_wrong_sender_rejected() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"file";
        let file_hash: [u8; 32] = Sha256::digest(data).into();

        let blob_id = svc
            .put_blob(&sender, &receiver, data, &file_hash)
            .expect("put");
        let wrong = [0xCCu8; 32];
        assert!(matches!(
            svc.delete_blob(&blob_id, &wrong),
            Err(ChatRelayError::Unauthorized)
        ));
    }

    // ── Online dedup (preserved) ─────────────────────────────────────────

    #[test]
    fn test_online_dedup() {
        let svc = make_service();
        let id = [0x01u8; 16];
        assert!(!svc.is_online_duplicate(&id));
        assert!(svc.is_online_duplicate(&id));
    }

    #[test]
    fn test_online_dedup_is_atomic_under_concurrency() {
        const WORKERS: usize = 16;
        let dedup = Arc::new(MessageDedup::new(32));
        let barrier = Arc::new(Barrier::new(WORKERS));
        let id = [0x02u8; 16];
        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let dedup = Arc::clone(&dedup);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    dedup.check_and_insert(&id)
                })
            })
            .collect();

        let duplicate_results = handles
            .into_iter()
            .map(|handle| handle.join().expect("dedup worker must not panic"))
            .collect::<Vec<_>>();
        assert_eq!(
            duplicate_results
                .iter()
                .filter(|is_duplicate| !**is_duplicate)
                .count(),
            1,
            "exactly one concurrent caller must win first delivery"
        );
    }

    // ── TTL cleanup (preserved) ──────────────────────────────────────────

    #[test]
    fn test_cleanup_does_not_touch_fresh_messages() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);

        svc.store_pending(&env).expect("store");
        let (expired, blobs) = svc.run_cleanup().expect("cleanup");
        assert_eq!(expired, 0);
        assert_eq!(blobs, 0);

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_cleanup_chunks_expiry_notifications_and_reconciles_usage() {
        let mut config = test_config();
        config.max_pending_per_wallet = 100;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let sender = kp.public_key_bytes();
        let receiver = [0xC1; 32];
        let mut expected_ids = HashSet::new();

        for _ in 0..70 {
            let envelope = make_envelope(&kp, receiver);
            expected_ids.insert(envelope.message_id);
            svc.store_pending(&envelope)
                .expect("store expiring message");
        }
        svc.conn
            .lock()
            .execute("UPDATE pending_messages SET received_at = 0", [])
            .expect("age pending messages");

        let (expired, blobs) = svc.run_cleanup().expect("cleanup expired messages");
        assert_eq!(expired, 70);
        assert_eq!(blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("usage after cleanup"),
            ChatRelayStorageUsage::default()
        );

        let (notifications, has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("pull expiry notifications");
        assert!(!has_more);
        assert_eq!(notifications.len(), 3);

        let mut chunk_lengths = Vec::new();
        let mut actual_ids = HashSet::new();
        for notification in &notifications {
            assert_eq!(notification.sender, sender);
            assert_eq!(notification.receiver, receiver);
            let ids = notification.message_ids().expect("decode notification");
            chunk_lengths.push(ids.len());
            actual_ids.extend(ids);
        }
        chunk_lengths.sort_unstable();
        assert_eq!(chunk_lengths, vec![6, 32, 32]);
        assert_eq!(actual_ids, expected_ids);

        let pending_rows: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count pending rows");
        assert_eq!(pending_rows, 0);
    }

    #[test]
    fn test_cleanup_quarantines_malformed_row_without_blocking() {
        let svc = make_service();
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xA1u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed durable row");

        assert_eq!(svc.run_cleanup().expect("quarantine cleanup"), (0, 0));
        assert_eq!(
            svc.storage_usage().expect("usage after quarantine"),
            ChatRelayStorageUsage::default()
        );

        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_runs_total, 1);
        assert_eq!(status.cleanup_failures_total, 0);
        assert_eq!(status.cleanup_batches_total, 1);
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        assert_eq!(status.last_cleanup_quarantined_pending_messages, 1);
        assert!(status.last_quarantine_at.is_some());
        assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));

        let conn = svc.conn.lock();
        let pending_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count pending rows");
        assert_eq!(pending_rows, 0);
        let event: (String, String, i64, i64) = conn
            .query_row(
                "SELECT source_kind, reason, row_count, encoded_bytes
                 FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("read de-identified quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
        assert_eq!(event.1, "expired_message_id");
        assert_eq!(event.2, 1);
        assert!(event.3 > 0);

        let mut schema_stmt = conn
            .prepare("PRAGMA table_info(relay_quarantine_events)")
            .expect("prepare quarantine schema query");
        let columns: Vec<String> = schema_stmt
            .query_map([], |row| row.get(1))
            .expect("query quarantine columns")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect quarantine columns");
        for forbidden in ["message_id", "sender", "receiver", "envelope", "ciphertext"] {
            assert!(!columns.iter().any(|column| column == forbidden));
        }
    }

    #[test]
    fn test_cleanup_does_not_notify_tampered_stored_sender() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xD4u8; 32];
        let envelope = make_envelope(&kp, receiver);
        let tampered_sender = [0xD5u8; 32];
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages
                 SET sender = ?1, received_at = 0
                 WHERE message_id = ?2",
                params![tampered_sender.as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper expired message sender");

        assert_eq!(svc.run_cleanup().expect("cleanup tampered sender"), (0, 0));
        let (notifications, has_more) = svc
            .pull_pending_notifications(&tampered_sender)
            .expect("pull attacker notifications");
        assert!(notifications.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read cleanup mismatch reason");
        assert_eq!(reason, "expired_message_sender_mismatch");
    }

    #[test]
    fn test_cleanup_defers_backlog_at_batch_budget_and_recovers_next_run() {
        let svc = make_service();
        insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE + 1, 0x10);

        let (first_expired, first_blobs) = svc
            .run_cleanup_with_batch_budget(1)
            .expect("first bounded cleanup");
        assert_eq!(first_expired, CLEANUP_MESSAGE_BATCH_SIZE);
        assert_eq!(first_blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("first usage").pending_messages,
            1
        );

        let deferred = svc.maintenance_status();
        assert_eq!(deferred.cleanup_runs_total, 1);
        assert_eq!(deferred.cleanup_batches_total, 1);
        assert_eq!(deferred.cleanup_backlog_deferred_total, 1);
        assert_eq!(
            deferred.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
        );
        assert_eq!(deferred.last_cleanup_batches, 1);
        assert!(deferred.last_cleanup_backlog_deferred);

        let (second_expired, second_blobs) = svc
            .run_cleanup_with_batch_budget(1)
            .expect("second bounded cleanup");
        assert_eq!(second_expired, 1);
        assert_eq!(second_blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("second usage").pending_messages,
            0
        );

        let recovered = svc.maintenance_status();
        assert_eq!(recovered.cleanup_runs_total, 2);
        assert_eq!(recovered.cleanup_batches_total, 2);
        assert_eq!(recovered.cleanup_backlog_deferred_total, 1);
        assert_eq!(
            recovered.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE + 1).unwrap_or(u64::MAX)
        );
        assert!(!recovered.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_cleanup_isolates_trailing_poison_row_after_committed_batch() {
        let svc = make_service();
        insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE, 0x10);
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xF0u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed trailing row");

        let (expired, blobs) = svc
            .run_cleanup_with_batch_budget(2)
            .expect("bounded cleanup with poison-row isolation");
        assert_eq!(expired, CLEANUP_MESSAGE_BATCH_SIZE);
        assert_eq!(blobs, 0);

        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_runs_total, 1);
        assert_eq!(status.cleanup_failures_total, 0);
        assert_eq!(status.cleanup_batches_total, 2);
        assert_eq!(
            status.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
        );
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.last_cleanup_batches, 2);
        assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));
        assert_eq!(
            svc.storage_usage()
                .expect("remaining usage")
                .pending_messages,
            0
        );
    }

    #[test]
    fn test_quarantine_persistence_failure_rolls_back_source_deletion() {
        let svc = make_service();
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xA1u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed durable row");
        svc.conn
            .lock()
            .execute("DROP TABLE relay_quarantine_events", [])
            .expect("simulate quarantine persistence failure");

        assert!(matches!(svc.run_cleanup(), Err(ChatRelayError::Sqlite(_))));
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);
        let pending_rows: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count retained source rows");
        assert_eq!(pending_rows, 1);
        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_failures_total, 1);
        assert_eq!(status.quarantined_pending_messages_total, 0);
    }

    #[test]
    fn test_quarantine_event_store_enforces_hard_retention_cap() {
        let svc = make_service();
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start quarantine event insert");
            let mut stmt = tx
                .prepare(
                    "INSERT INTO relay_quarantine_events
                     (source_kind, reason, row_count, encoded_bytes, quarantined_at)
                     VALUES (?1, 'test_reason', 1, 1, ?2)",
                )
                .expect("prepare quarantine event insert");
            for _ in 0..=MAX_QUARANTINE_EVENTS {
                stmt.execute(params![QUARANTINE_SOURCE_PENDING_MESSAGE, i64::MAX])
                    .expect("insert quarantine event");
            }
            drop(stmt);
            tx.commit().expect("commit quarantine events");
        }

        svc.run_cleanup_with_batch_budget(1)
            .expect("trim quarantine event overflow");
        let retained: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("count bounded quarantine events");
        assert_eq!(
            retained,
            i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX)
        );
        let status = svc.maintenance_status();
        assert_eq!(status.quarantine_events_removed_total, 1);
        assert_eq!(
            status.quarantine_events_retained,
            u64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(u64::MAX)
        );
        assert!(!status.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_cleanup_out_of_range_ttl_fails_closed() {
        let mut config = test_config();
        config.offline_ttl_secs = u64::MAX;
        let svc = ChatRelayService::new(config, [0x42; 32]).expect("service");
        let kp = IdentityKeyPair::generate();
        let receiver = [0xB4; 32];
        let envelope = make_envelope(&kp, receiver);

        svc.store_pending(&envelope).expect("store pending message");
        let (expired, _) = svc.run_cleanup().expect("cleanup");

        assert_eq!(expired, 0);
        assert_eq!(
            svc.storage_usage().expect("storage usage").pending_messages,
            1
        );
    }

    #[test]
    fn test_maintenance_status_deserializes_older_snapshot() {
        let status: ChatRelayMaintenanceStatus = serde_json::from_value(serde_json::json!({
            "cleanup_runs_total": 7,
            "last_cleanup_status": "succeeded"
        }))
        .expect("deserialize backward-compatible maintenance snapshot");

        assert_eq!(status.cleanup_runs_total, 7);
        assert_eq!(status.cleanup_batches_total, 0);
        assert_eq!(status.quarantined_pending_messages_total, 0);
        assert_eq!(status.quarantine_events_retained, 0);
        assert!(!status.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_expiry_notification_pull_is_bounded_and_pageable() {
        let svc = make_service();
        let sender = [0xD1; 32];
        let receiver = [0xD2u8; 32];
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start notification insert");
            for created_at in 0..17i64 {
                let ids = bincode::serialize(&vec![[created_at as u8; 16]])
                    .expect("serialize notification");
                tx.execute(
                    "INSERT INTO expired_notifications
                     (sender, receiver, message_ids, created_at, pushed)
                     VALUES (?1, ?2, ?3, ?4, 0)",
                    params![sender.as_slice(), receiver.as_slice(), ids, created_at],
                )
                .expect("insert notification");
            }
            tx.commit().expect("commit notifications");
        }

        let (first_page, first_has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("first notification page");
        assert_eq!(first_page.len(), MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
        assert!(first_has_more);
        let first_ids: Vec<i64> = first_page
            .iter()
            .map(|notification| notification.id)
            .collect();
        svc.mark_notifications_pushed(&first_ids)
            .expect("mark first page");

        let (second_page, second_has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("second notification page");
        assert_eq!(second_page.len(), 1);
        assert!(!second_has_more);
    }

    #[test]
    fn test_malformed_expiry_notification_isolated_without_blocking_valid_rows() {
        let svc = make_service();
        let sender = [0xE1; 32];
        let valid_receiver = [0xE4; 32];
        let ids = bincode::serialize(&vec![[0xE2; 16]]).expect("serialize notification");
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start mixed notification transaction");
            tx.execute(
                "INSERT INTO expired_notifications
                 (sender, receiver, message_ids, created_at, pushed)
                 VALUES (?1, ?2, ?3, 0, 0)",
                params![sender.as_slice(), [0xE3u8; 31].as_slice(), &ids],
            )
            .expect("insert malformed notification");
            tx.execute(
                "INSERT INTO expired_notifications
                 (sender, receiver, message_ids, created_at, pushed)
                 VALUES (?1, ?2, ?3, 1, 0)",
                params![sender.as_slice(), valid_receiver.as_slice(), ids],
            )
            .expect("insert valid notification");
            tx.commit().expect("commit mixed notifications");
        }

        let (notifications, has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("pull must isolate poison row");
        assert_eq!(notifications.len(), 1);
        assert!(!has_more);
        assert_eq!(notifications[0].receiver, valid_receiver);

        let status = svc.maintenance_status();
        assert_eq!(status.quarantined_expired_notifications_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        assert!(status.last_quarantine_at.is_some());

        let conn = svc.conn.lock();
        let remaining: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM expired_notifications WHERE pushed = 0",
                [],
                |row| row.get(0),
            )
            .expect("count valid notification");
        assert_eq!(remaining, 1);
        let event: (String, String, i64) = conn
            .query_row(
                "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read notification quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION);
        assert_eq!(event.1, "expired_notification_receiver");
        assert_eq!(event.2, 1);
    }

    #[test]
    fn expired_notifications_are_wired_to_authenticated_chat_pull() {
        let source = include_str!("../server.rs");
        assert!(source.contains("relay.pull_pending_notifications(&wallet)"));
        assert!(source.contains("Self::push_expired_notifications("));
        assert!(source.contains("has_more |= notification_has_more || !delivery_complete"));
        assert!(source.contains("self.spawn_chat_relay_cleanup_task(Arc::clone(relay))"));
        assert!(source.contains("tokio::task::spawn_blocking(move || cleanup_relay.run_cleanup())"));
        assert!(source.contains("tokio::time::MissedTickBehavior::Skip"));
        assert!(source.contains("relay.record_maintenance_worker_failure(reason)"));
        assert!(source.contains("\"maintenance\": relay.maintenance_status()"));
    }

    // ── node_secret derivation (preserved) ───────────────────────────────

    #[test]
    fn test_derive_node_secret_deterministic() {
        let sk = [0x42u8; 32];
        let s1 = derive_node_secret(&sk);
        let s2 = derive_node_secret(&sk);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_derive_node_secret_different_keys() {
        let s1 = derive_node_secret(&[0x01u8; 32]);
        let s2 = derive_node_secret(&[0x02u8; 32]);
        assert_ne!(s1, s2);
    }
}
