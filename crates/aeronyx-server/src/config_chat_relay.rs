// ============================================
// File: crates/aeronyx-server/src/config_chat_relay.rs
// ============================================
//! # Chat Relay Configuration
//!
//! ## Creation Reason
//! New subsystem added in v1.1.0-ChatRelay. Placed in a dedicated file
//! from the start to avoid repeating the config.rs bloat pattern.
//!
//! ## Modification Reason
//! v1.1.0-ChatRelay — 🌟 Initial implementation.
//! v1.2.0-GlobalStorageQuotas — Added node-wide message/blob count and byte
//! ceilings so many synthetic receivers cannot bypass per-mailbox limits.
//! v1.3.0-MaintenanceBounds — Rejected TTL values that cannot be represented
//! safely by SQLite's signed timestamp domain.
//! v1.4.0-PeerRelayAdmission — Added a backward-compatible global admission
//! ceiling for the legacy direct peer-relay endpoint.
//! v1.5.0-AuthenticatedPeerFairness — Added a bounded per-authenticated-node
//! ceiling behind direct peer-relay v2 signature verification.
//! v1.6.0-DurableCustody — Documented the non-configurable FULL SQLite
//! durability boundary required by signed custody acknowledgements.
//! v1.7.0-StartupCustodyIntegrity — Documented owner-only Unix storage and the
//! pre-migration SQLite physical-integrity activation gate.
//! v1.8.0-VerifiedCustodyBackup — Documented the WAL-aware, owner-private
//! recovery artifact boundary exposed by the relay service.
//! v1.9.0-IdempotentCustodyBackup — Documented restart-safe audited command
//! replay and immutable re-verification of existing recovery artifacts.
//! v1.10.0-CustodyBackupRetention — Added bounded count/byte retention targets
//! and the non-destructive audit command contract for private recovery images.
//! v1.11.0-CustodyBackupPrune — Added the minimum interrupted-backup grace
//! period used only by explicit host-local dry-run/prune commands.
//! v1.12.0-CrashSafeVerifiedSubmitAdmission — Documented that the node-wide
//! deduplication capacity also bounds durable verified-submit replay evidence.
//!
//! ## Main Functionality
//! - `ChatRelayConfig` — all knobs for the zero-knowledge P2P chat relay
//! - `validate()` — gated by `enabled`; skipped entirely when disabled
//! - All fields have sane defaults for backward-compatible TOML loading
//!
//! ## Dependencies
//! - `config_memchain.rs` — embeds `ChatRelayConfig` as `chat_relay` field
//! - `services/chat_relay.rs` — consumes this config at startup
//! - `api/chat.rs`           — reads size limits at request time
//!
//! ## Main Logical Flow
//! 1. TOML `[memchain.chat_relay]` deserializes into `ChatRelayConfig`
//! 2. `MemChainConfig::validate()` calls `self.chat_relay.validate()`
//! 3. When `enabled = false` the entire validate body is a no-op
//! 4. When `enabled = true` all non-zero / non-empty invariants are checked
//!
//! ⚠️ Important Note for Next Developer:
//! - `db_path` is relative to CWD unless absolute. The service layer must
//!   create parent directories before opening (see `ChatRelayService::new()`).
//! - `max_pending_per_wallet` is enforced at write time — a "mailbox full"
//!   error is sent back to the sender when the limit is reached.
//! - Node-wide message/blob count and byte ceilings are mandatory when the
//!   relay is enabled. Keep them at least as large as their per-wallet/single
//!   item limits so one valid mailbox or object remains usable.
//! - [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex]
//!   `dedup_lru_capacity` is node-wide (not per-wallet). It bounds both the
//!   in-memory online-delivery LRU and short-lived durable verified-submit
//!   responses/reservations. When the durable bound is full, new verified
//!   submits fail before route or custody side effects; retained evidence is
//!   never evicted early. Keep operational headroom for the configured TTL.
//! - `max_message_size`: values > 64 KB emit a warn (UDP fragmentation risk)
//!   and are hard-rejected above `MAX_MESSAGE_SIZE_HARD_LIMIT` (1 MB).
//!   `ChatRelayService::store_pending()` enforces the configured ciphertext
//!   ceiling before any durable write.
//!   Rationale: text chat envelopes should never approach 1 MB; if they do,
//!   it indicates a misconfiguration rather than a legitimate use-case.
//! - [PEER-RELAY-ADMISSION 2026-08-15 by Codex] The direct peer legacy
//!   endpoint has no authenticated previous-hop identity. The parser-front
//!   `peer_relay_requests_per_minute` guard is therefore deliberately global
//!   across v1/v2; never derive it from user/sender/receiver keys or IPs.
//! - [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] Direct relay v2 may
//!   additionally use `peer_relay_authenticated_requests_per_minute` only
//!   after node-signature verification. This node-id bucket is bounded and is
//!   not a substitute for the global guard because permissionless identities
//!   remain Sybil-able.
//! - [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] `ChatRelayService`
//!   verifies SQLite FULL-or-stronger durability before activation. Do not add
//!   a NORMAL/OFF operator override while the protocol issues signed custody
//!   acknowledgements from successful durable writes.
//! - [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] The service restricts
//!   Unix custody files to the node account and runs a bounded SQLite physical
//!   integrity check before WAL changes or migrations. Failure disables relay;
//!   raw findings and configured paths must not enter logs or public health.
//! - [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Verified backups are
//!   created only inside the owner-private `.aeronyx-relay-backups` directory
//!   beside `db_path`; callers cannot supply an arbitrary destination. The
//!   core service does not schedule backups or expose them over HTTP. The CMS
//!   operator command is HTTPS-only, confirmation-gated, audited, and runs via
//!   `spawn_blocking`. Its command ID is HMAC-derived into a private artifact
//!   key so retries across restart re-verify and reuse one immutable image.
//! - [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Verified recovery
//!   images have count and aggregate-byte planning targets. The local service
//!   audit verifies all images and reports excess capacity without deleting
//!   it or publishing it over HTTP/CMS. Explicit pruning is host-local,
//!   confirmation-gated, and never automatic; restore, listing, and download
//!   remain separate local operator concerns.
//! - `expired_notification_ttl_secs`: after this TTL, undelivered expiry
//!   notifications are silently discarded. Flutter client local timeout is
//!   the fallback.
//! - chat_relay.db_path and saas.data_root share the `"data/"` prefix by
//!   convention but are NOT linked. If you change `saas.data_root`, also
//!   update `chat_relay.db_path` explicitly in your config file.
//!
//! ## Last Modified
//! v1.12.0-CrashSafeVerifiedSubmitAdmission — Documented the shared volatile
//! and durable admission capacity boundary for verified submits.
//! v1.11.0-CustodyBackupPrune — Host-local, confirmation-gated prune policy.
//! v1.10.0-CustodyBackupRetention — Bounded private backup retention.
//! v1.9.0-IdempotentCustodyBackup — Restart-safe audited backup replay.
//! v1.8.0-VerifiedCustodyBackup — Declared the private recovery boundary.
//! v1.7.0-StartupCustodyIntegrity — Fail-closed physical storage activation.
//! v1.6.0-DurableCustody — Declared FULL durability a custody invariant.
//! v1.5.0-AuthenticatedPeerFairness — Added configurable post-signature
//! per-node admission for direct peer-relay v2.
//! v1.4.0-PeerRelayAdmission — Added configurable parser-front admission for
//! the legacy direct peer-relay compatibility path.
//! v1.3.0-MaintenanceBounds — Added signed timestamp boundary validation.
//! v1.2.0-GlobalStorageQuotas — Added backward-compatible global disk ceilings.
//! v1.1.0-ChatRelay — Initial implementation.

use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

/// Hard upper bound for `max_message_size`.
///
/// Text chat envelopes should never legitimately exceed 1 MB.
/// Values above this indicate misconfiguration and are rejected.
const MAX_MESSAGE_SIZE_HARD_LIMIT: usize = 1_048_576; // 1 MB

/// SQLite INTEGER and cleanup timestamp arithmetic use signed 64-bit values.
const MAX_SQLITE_TTL_SECS: u64 = i64::MAX as u64;

/// Default node-global admission ceiling for legacy direct peer relay.
///
/// The onion path has authenticated per-hop limiting. Direct relay predates
/// that contract, so this coarse ceiling bounds parser/storage pressure without
/// creating buckets keyed by user-visible envelope metadata.
pub const DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE: u32 = 1_200;

/// Default per-authenticated-node admission ceiling for direct relay v2.
///
/// This fairness limit runs only after Ed25519 node authentication. The global
/// parser-front limit remains mandatory because permissionless node identities
/// can be rotated and therefore cannot provide Sybil resistance by themselves.
pub const DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE: u32 = 240;

/// Default planning target for verified relay-custody recovery-image count.
pub const DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS: usize = 8;

/// Default planning target for aggregate verified recovery-image bytes.
pub const DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Defensive ceiling for the count planning target.
pub const MAX_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS: usize = 64;

/// Default grace period before an interrupted private backup may be pruned.
pub const DEFAULT_CUSTODY_BACKUP_PARTIAL_GRACE_SECS: u64 = 24 * 60 * 60;

/// Hard minimum grace period for interrupted private backup files.
pub const MIN_CUSTODY_BACKUP_PARTIAL_GRACE_SECS: u64 = 24 * 60 * 60;

// ============================================
// ChatRelayConfig
// ============================================

/// Zero-knowledge P2P chat relay configuration.
///
/// ## Design
/// The node acts as a blind relay — it stores and forwards E2E-encrypted
/// envelopes without being able to read message content. All cryptographic
/// operations (encryption, decryption, key derivation) happen on the
/// Flutter client.
///
/// ## Activation
/// Set `enabled = true` to activate. All other fields have safe defaults.
/// Existing deployments upgrading to v1.1.0-ChatRelay see zero behavior
/// change until `enabled = true` is explicitly set.
///
/// ## Storage
/// Chat data is stored in a separate SQLite file (`db_path`) isolated from
/// the main MemChain database. This ensures chat relay failures cannot
/// corrupt MemChain state and simplifies backup/purge. The service enforces
/// WAL + FULL durability before it can acknowledge encrypted custody; this is
/// intentionally not an operator-tunable downgrade. On Unix, the database and
/// WAL sidecars are owner-only; startup physical-integrity failure rejects
/// activation before migrations or custody receipts. Verified recovery images
/// are WAL-aware and remain confined beside this database in an owner-private
/// directory; no automatic schedule or remote download interface is enabled.
///
/// ## Configuration Example
/// ```toml
/// [memchain.chat_relay]
/// enabled = true
/// offline_ttl_secs = 259200       # 72 hours
/// max_pending_per_wallet = 500
/// max_pending_messages_total = 100000
/// max_pending_message_bytes_total = 536870912  # 512 MiB
/// db_path = "data/chat_pending.db"
/// max_message_size = 65536        # 64 KB (text envelope)
/// max_blob_size = 10485760        # 10 MB (encrypted media)
/// max_blobs_per_receiver = 50
/// max_pending_blobs_total = 5000
/// max_pending_blob_bytes_total = 2147483648    # 2 GiB
/// cleanup_interval_secs = 60
/// dedup_lru_capacity = 10000
/// expired_notification_ttl_secs = 604800  # 7 days
/// peer_relay_requests_per_minute = 1200
/// peer_relay_authenticated_requests_per_minute = 240
/// custody_backup_retention_target_artifacts = 8
/// custody_backup_retention_target_bytes = 8589934592  # 8 GiB
/// custody_backup_partial_grace_secs = 86400             # 24 hours
/// ```
///
/// ## Last Modified
/// v1.10.0-CustodyBackupRetention — Bounded private recovery-image retention.
/// v1.8.0-VerifiedCustodyBackup — Private WAL-aware recovery artifacts.
/// v1.7.0-StartupCustodyIntegrity — Owner-only files and startup quick-check.
/// v1.2.0-GlobalStorageQuotas — Added node-wide durable queue ceilings.
/// v1.1.0-ChatRelay — Initial implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRelayConfig {
    /// Enable the chat relay subsystem.
    ///
    /// When `false` (default), `ChatRelay` / `ChatPull` / `ChatAck` /
    /// `ChatExpired` MemChain messages are silently ignored by the node.
    #[serde(default)]
    pub enabled: bool,

    /// Offline message TTL in seconds (default: 259 200 = 72 hours).
    ///
    /// Messages not acknowledged within this window are marked `Expired`,
    /// and a `ChatExpired` notification is queued for the sender.
    #[serde(default = "default_chat_ttl")]
    pub offline_ttl_secs: u64,

    /// Maximum number of pending (unacknowledged) messages per receiver wallet.
    ///
    /// When this limit is reached, new messages addressed to that wallet are
    /// rejected with a "mailbox full" response to the sender.
    /// Default: 500.
    #[serde(default = "default_max_pending_per_wallet")]
    pub max_pending_per_wallet: usize,

    /// Maximum pending messages across every receiver on this node.
    ///
    /// This closes the synthetic-receiver bypass of the per-wallet mailbox
    /// limit. Default: 100 000.
    #[serde(default = "default_max_pending_messages_total")]
    pub max_pending_messages_total: usize,

    /// Maximum encoded pending-message bytes across the node.
    ///
    /// Count limits alone do not bound disk use because encrypted envelopes
    /// vary in size. Default: 536 870 912 (512 MiB).
    #[serde(default = "default_max_pending_message_bytes_total")]
    pub max_pending_message_bytes_total: u64,

    /// Path to the SQLite database file for chat relay storage.
    ///
    /// Stores `pending_messages`, `pending_blobs`, and
    /// `expired_notifications` tables. Unix deployments restrict this file
    /// and SQLite sidecars to the node account and fail closed when startup
    /// physical-integrity verification does not return `ok`.
    ///
    /// ⚠️ Not linked to `saas.data_root` — must be updated independently.
    /// Default: `"data/chat_pending.db"`.
    #[serde(default = "default_chat_db_path")]
    pub db_path: String,

    /// Maximum size in bytes for a text `ChatEnvelope.ciphertext`.
    ///
    /// Text envelopes travel over UDP; enforcing this limit prevents
    /// accidental MTU violations and abuse.
    ///
    /// - Values > 65 536 (64 KB) emit a warning (UDP fragmentation risk).
    /// - Values > 1 048 576 (1 MB) are hard-rejected at validate time.
    ///
    /// Default: 65 536 (64 KB).
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,

    /// Maximum size in bytes for an encrypted blob (image / file).
    ///
    /// Blobs are uploaded via `POST /api/chat/blob` and stored in
    /// `pending_blobs`. Uploads exceeding this limit are rejected with
    /// HTTP 413.
    /// Default: 10 485 760 (10 MB).
    #[serde(default = "default_max_blob_size")]
    pub max_blob_size: usize,

    /// Maximum number of pending blobs per receiver wallet.
    ///
    /// Prevents a single sender from filling the node's disk by uploading
    /// large files to an offline receiver.
    /// Default: 50.
    #[serde(default = "default_max_blobs_per_receiver")]
    pub max_blobs_per_receiver: usize,

    /// Maximum encrypted blobs retained across every receiver.
    /// Default: 5 000.
    #[serde(default = "default_max_pending_blobs_total")]
    pub max_pending_blobs_total: usize,

    /// Maximum encrypted blob bytes retained across the node.
    /// Default: 2 147 483 648 (2 GiB).
    #[serde(default = "default_max_pending_blob_bytes_total")]
    pub max_pending_blob_bytes_total: u64,

    /// Interval in seconds between TTL cleanup runs.
    ///
    /// The cleanup task scans `pending_messages` and `pending_blobs` for
    /// expired entries, queues `ChatExpired` notifications, and deletes
    /// delivered/expired rows.
    /// Default: 60.
    #[serde(default = "default_cleanup_interval")]
    pub cleanup_interval_secs: u64,

    /// Node-wide capacity for online-delivery deduplication and durable
    /// verified-submit replay evidence.
    ///
    /// When a receiver is online, messages are forwarded directly without
    /// hitting SQLite. The LRU cache prevents duplicate delivery if the
    /// sender retransmits before the first ACK arrives.
    ///
    /// [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] The same
    /// value bounds short-lived durable response/reservation rows. Saturation
    /// rejects a new verified submit before any route or custody side effect;
    /// unexpired replay evidence is not evicted to make room.
    ///
    /// In-memory entries are ~64 bytes each (16-byte message_id + LRU
    /// overhead). At capacity 10 000: ~640 KB, excluding SQLite rows.
    /// Default: 10 000.
    #[serde(default = "default_dedup_lru_capacity")]
    pub dedup_lru_capacity: usize,

    /// How long (seconds) to retain undelivered `ChatExpired` notifications
    /// for offline senders.
    ///
    /// If Alice is offline when her message expires (72 h TTL), the node
    /// queues a `ChatExpired` notification in `expired_notifications`.
    /// This field controls how long that queued notification is kept before
    /// being discarded (Alice's Flutter client uses a local timeout as
    /// fallback).
    /// Default: 604 800 (7 days).
    #[serde(default = "default_expired_notification_ttl")]
    pub expired_notification_ttl_secs: u64,

    /// Maximum direct peer-relay HTTP requests admitted per minute.
    ///
    /// This global guard covers direct relay v1/v2 before JSON deserialization.
    /// Because legacy v1 cannot authenticate a previous-hop node, and because
    /// permissionless node identities are rotatable, it intentionally does not
    /// key on node, sender, receiver, IP, or other user-adjacent metadata.
    /// Default: 1 200.
    #[serde(default = "default_peer_relay_requests_per_minute")]
    pub peer_relay_requests_per_minute: u32,

    /// Maximum direct relay v2 requests admitted per authenticated node/minute.
    ///
    /// The bucket is created only after the outer Ed25519 node signature has
    /// verified. It contains no user, receiver, IP, endpoint, or payload data.
    /// Default: 240.
    #[serde(default = "default_authenticated_peer_relay_requests_per_minute")]
    pub peer_relay_authenticated_requests_per_minute: u32,

    /// Planning target for verified relay-custody recovery-image count.
    ///
    /// The audited retention inspection models an oldest-first policy. The
    /// newest image is always counted as retained even when its size alone
    /// exceeds the byte budget. The inspection never deletes files.
    /// Default: 8; hard maximum: 64.
    ///
    /// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] This field does not
    /// delete or reject backups; it only defines the local audit comparison.
    #[serde(default = "default_custody_backup_retention_target_artifacts")]
    pub custody_backup_retention_target_artifacts: usize,

    /// Planning target for aggregate verified recovery-image bytes.
    ///
    /// The newest verified image is always modeled as retained even when it
    /// alone exceeds this planning target. Such a condition is reported as
    /// `budget_exceeded`; no image is removed. Default: 8 GiB.
    #[serde(default = "default_custody_backup_retention_target_bytes")]
    pub custody_backup_retention_target_bytes: u64,

    /// Minimum age before an interrupted private backup file becomes eligible
    /// for an explicitly-confirmed host-local prune command.
    ///
    /// No timer is created by this setting. Dry-run and prune evaluate it only
    /// when an operator invokes the local command. Default/minimum: 24 hours.
    ///
    /// [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] A lower value is rejected
    /// so a clock adjustment or slow backup cannot make a live temporary file
    /// immediately eligible for deletion.
    #[serde(default = "default_custody_backup_partial_grace_secs")]
    pub custody_backup_partial_grace_secs: u64,
}

// ── Default functions ──

fn default_chat_ttl() -> u64 {
    259_200
} // 72 hours
fn default_max_pending_per_wallet() -> usize {
    500
}
fn default_max_pending_messages_total() -> usize {
    100_000
}
fn default_max_pending_message_bytes_total() -> u64 {
    512 * 1024 * 1024
}
fn default_chat_db_path() -> String {
    "data/chat_pending.db".into()
}
fn default_max_message_size() -> usize {
    65_536
} // 64 KB
fn default_max_blob_size() -> usize {
    10_485_760
} // 10 MB
fn default_max_blobs_per_receiver() -> usize {
    50
}
fn default_max_pending_blobs_total() -> usize {
    5_000
}
fn default_max_pending_blob_bytes_total() -> u64 {
    2 * 1024 * 1024 * 1024
}
fn default_cleanup_interval() -> u64 {
    60
}
fn default_dedup_lru_capacity() -> usize {
    10_000
}
fn default_expired_notification_ttl() -> u64 {
    604_800
} // 7 days
fn default_peer_relay_requests_per_minute() -> u32 {
    DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE
}
fn default_authenticated_peer_relay_requests_per_minute() -> u32 {
    DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE
}
fn default_custody_backup_retention_target_artifacts() -> usize {
    DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS
}
fn default_custody_backup_retention_target_bytes() -> u64 {
    DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_BYTES
}
fn default_custody_backup_partial_grace_secs() -> u64 {
    DEFAULT_CUSTODY_BACKUP_PARTIAL_GRACE_SECS
}

impl Default for ChatRelayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            offline_ttl_secs: default_chat_ttl(),
            max_pending_per_wallet: default_max_pending_per_wallet(),
            max_pending_messages_total: default_max_pending_messages_total(),
            max_pending_message_bytes_total: default_max_pending_message_bytes_total(),
            db_path: default_chat_db_path(),
            max_message_size: default_max_message_size(),
            max_blob_size: default_max_blob_size(),
            max_blobs_per_receiver: default_max_blobs_per_receiver(),
            max_pending_blobs_total: default_max_pending_blobs_total(),
            max_pending_blob_bytes_total: default_max_pending_blob_bytes_total(),
            cleanup_interval_secs: default_cleanup_interval(),
            dedup_lru_capacity: default_dedup_lru_capacity(),
            expired_notification_ttl_secs: default_expired_notification_ttl(),
            peer_relay_requests_per_minute: default_peer_relay_requests_per_minute(),
            peer_relay_authenticated_requests_per_minute:
                default_authenticated_peer_relay_requests_per_minute(),
            custody_backup_retention_target_artifacts:
                default_custody_backup_retention_target_artifacts(),
            custody_backup_retention_target_bytes: default_custody_backup_retention_target_bytes(),
            custody_backup_partial_grace_secs: default_custody_backup_partial_grace_secs(),
        }
    }
}

impl ChatRelayConfig {
    /// Validates chat relay configuration.
    ///
    /// When `enabled = false`, all validation is skipped (safe defaults
    /// guaranteed by `Default` impl). This ensures existing deployments
    /// upgrading to v1.1.0-ChatRelay see zero behavior change.
    ///
    /// # Errors
    /// Returns `ServerError::ConfigInvalid` if any enabled constraint is violated.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if self.offline_ttl_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.offline_ttl_secs",
                "must be > 0",
            ));
        }

        if self.offline_ttl_secs > MAX_SQLITE_TTL_SECS {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.offline_ttl_secs",
                "must fit SQLite's signed 64-bit timestamp domain",
            ));
        }

        if self.max_pending_per_wallet == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_per_wallet",
                "must be > 0",
            ));
        }

        if self.max_pending_messages_total < self.max_pending_per_wallet {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_messages_total",
                "must be >= max_pending_per_wallet",
            ));
        }

        if self.max_pending_message_bytes_total < self.max_message_size as u64 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_message_bytes_total",
                "must be >= max_message_size",
            ));
        }

        if self.db_path.is_empty() {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.db_path",
                "cannot be empty when chat_relay.enabled = true",
            ));
        }

        if self.max_message_size == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_message_size",
                "must be > 0",
            ));
        }

        // Hard upper limit: text envelopes > 1 MB indicate misconfiguration.
        if self.max_message_size > MAX_MESSAGE_SIZE_HARD_LIMIT {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_message_size",
                format!(
                    "must be <= {} bytes (1 MB hard limit), got {}",
                    MAX_MESSAGE_SIZE_HARD_LIMIT, self.max_message_size
                ),
            ));
        }

        // Soft warning: values > 64 KB risk UDP fragmentation.
        if self.max_message_size > 65_536 {
            tracing::warn!(
                max = self.max_message_size,
                "[CHAT_RELAY] max_message_size > 64 KB may cause UDP fragmentation"
            );
        }

        if self.max_blob_size == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_blob_size",
                "must be > 0",
            ));
        }

        if self.max_blobs_per_receiver == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_blobs_per_receiver",
                "must be > 0",
            ));
        }

        if self.max_pending_blobs_total < self.max_blobs_per_receiver {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_blobs_total",
                "must be >= max_blobs_per_receiver",
            ));
        }

        if self.max_pending_blob_bytes_total < self.max_blob_size as u64 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_blob_bytes_total",
                "must be >= max_blob_size",
            ));
        }

        if self.cleanup_interval_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.cleanup_interval_secs",
                "must be > 0",
            ));
        }

        if self.dedup_lru_capacity == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.dedup_lru_capacity",
                "must be > 0",
            ));
        }

        if self.expired_notification_ttl_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.expired_notification_ttl_secs",
                "must be > 0",
            ));
        }

        if self.expired_notification_ttl_secs > MAX_SQLITE_TTL_SECS {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.expired_notification_ttl_secs",
                "must fit SQLite's signed 64-bit timestamp domain",
            ));
        }

        if self.peer_relay_requests_per_minute == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.peer_relay_requests_per_minute",
                "must be > 0",
            ));
        }

        if self.peer_relay_authenticated_requests_per_minute == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.peer_relay_authenticated_requests_per_minute",
                "must be > 0",
            ));
        }

        if self.custody_backup_retention_target_artifacts == 0
            || self.custody_backup_retention_target_artifacts
                > MAX_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS
        {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.custody_backup_retention_target_artifacts",
                format!("must be between 1 and {MAX_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS}"),
            ));
        }

        if self.custody_backup_retention_target_bytes == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.custody_backup_retention_target_bytes",
                "must be > 0",
            ));
        }

        if self.custody_backup_partial_grace_secs < MIN_CUSTODY_BACKUP_PARTIAL_GRACE_SECS {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.custody_backup_partial_grace_secs",
                format!("must be >= {MIN_CUSTODY_BACKUP_PARTIAL_GRACE_SECS}"),
            ));
        }

        Ok(())
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_relay_disabled_by_default() {
        let cr = ChatRelayConfig::default();
        assert!(!cr.enabled);
    }

    #[test]
    fn test_chat_relay_default_values() {
        let cr = ChatRelayConfig::default();
        assert_eq!(cr.offline_ttl_secs, 259_200);
        assert_eq!(cr.max_pending_per_wallet, 500);
        assert_eq!(cr.max_pending_messages_total, 100_000);
        assert_eq!(cr.max_pending_message_bytes_total, 536_870_912);
        assert_eq!(cr.db_path, "data/chat_pending.db");
        assert_eq!(cr.max_message_size, 65_536);
        assert_eq!(cr.max_blob_size, 10_485_760);
        assert_eq!(cr.max_blobs_per_receiver, 50);
        assert_eq!(cr.max_pending_blobs_total, 5_000);
        assert_eq!(cr.max_pending_blob_bytes_total, 2_147_483_648);
        assert_eq!(cr.cleanup_interval_secs, 60);
        assert_eq!(cr.dedup_lru_capacity, 10_000);
        assert_eq!(cr.expired_notification_ttl_secs, 604_800);
        assert_eq!(
            cr.peer_relay_requests_per_minute,
            DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE
        );
        assert_eq!(
            cr.peer_relay_authenticated_requests_per_minute,
            DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE
        );
        assert_eq!(
            cr.custody_backup_retention_target_artifacts,
            DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS
        );
        assert_eq!(
            cr.custody_backup_retention_target_bytes,
            DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_BYTES
        );
        assert_eq!(
            cr.custody_backup_partial_grace_secs,
            DEFAULT_CUSTODY_BACKUP_PARTIAL_GRACE_SECS
        );
    }

    #[test]
    fn test_chat_relay_disabled_skips_validation() {
        // All invalid values — must pass because enabled = false
        let cr = ChatRelayConfig {
            enabled: false,
            offline_ttl_secs: 0,
            max_pending_per_wallet: 0,
            max_pending_messages_total: 0,
            max_pending_message_bytes_total: 0,
            db_path: String::new(),
            max_message_size: 0,
            max_blob_size: 0,
            max_blobs_per_receiver: 0,
            max_pending_blobs_total: 0,
            max_pending_blob_bytes_total: 0,
            cleanup_interval_secs: 0,
            dedup_lru_capacity: 0,
            expired_notification_ttl_secs: 0,
            peer_relay_requests_per_minute: 0,
            peer_relay_authenticated_requests_per_minute: 0,
            custody_backup_retention_target_artifacts: 0,
            custody_backup_retention_target_bytes: 0,
            custody_backup_partial_grace_secs: 0,
        };
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_enabled_default_valid() {
        let cr = ChatRelayConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_ttl_zero_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            offline_ttl_secs: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_ttl_above_sqlite_integer_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            offline_ttl_secs: u64::MAX,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_empty_db_path_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            db_path: String::new(),
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_global_message_count_below_mailbox_limit_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_pending_per_wallet: 501,
            max_pending_messages_total: 500,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_global_message_bytes_below_single_message_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_message_size: 65_536,
            max_pending_message_bytes_total: 65_535,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_message_size_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_message_size: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_message_size_over_hard_limit_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_message_size: MAX_MESSAGE_SIZE_HARD_LIMIT + 1,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_message_size_at_hard_limit_accepted() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_message_size: MAX_MESSAGE_SIZE_HARD_LIMIT,
            ..Default::default()
        };
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_zero_blob_size_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_blob_size: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_blobs_per_receiver_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_blobs_per_receiver: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_global_blob_count_below_receiver_limit_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_blobs_per_receiver: 51,
            max_pending_blobs_total: 50,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_global_blob_bytes_below_single_blob_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            max_blob_size: 1_024,
            max_pending_blob_bytes_total: 1_023,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_cleanup_interval_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            cleanup_interval_secs: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_lru_capacity_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            dedup_lru_capacity: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_expired_ttl_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            expired_notification_ttl_secs: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_expired_notification_ttl_above_sqlite_integer_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            expired_notification_ttl_secs: u64::MAX,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_peer_relay_rate_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            peer_relay_requests_per_minute: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_authenticated_peer_relay_rate_rejected() {
        let cr = ChatRelayConfig {
            enabled: true,
            peer_relay_authenticated_requests_per_minute: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_backup_retention_bounds_rejected() {
        // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Keep both the
        // planning target sane; the service separately hard-bounds directory
        // traversal so an operator value cannot weaken scan limits.
        for target_artifacts in [0, MAX_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS + 1] {
            let cr = ChatRelayConfig {
                enabled: true,
                custody_backup_retention_target_artifacts: target_artifacts,
                ..Default::default()
            };
            assert!(cr.validate().is_err());
        }

        let cr = ChatRelayConfig {
            enabled: true,
            custody_backup_retention_target_bytes: 0,
            ..Default::default()
        };
        assert!(cr.validate().is_err());

        let cr = ChatRelayConfig {
            enabled: true,
            custody_backup_partial_grace_secs: MIN_CUSTODY_BACKUP_PARTIAL_GRACE_SECS - 1,
            ..Default::default()
        };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_toml_parsing() {
        // We can't call `toml::from_str::<ServerConfig>` here (circular dep),
        // but we can test raw TOML → ChatRelayConfig deserialization.
        let toml_str = r#"
enabled = true
offline_ttl_secs = 86400
max_pending_per_wallet = 200
max_pending_messages_total = 40000
max_pending_message_bytes_total = 268435456
db_path = "data/chat_test.db"
max_message_size = 32768
max_blob_size = 5242880
max_blobs_per_receiver = 20
max_pending_blobs_total = 2000
max_pending_blob_bytes_total = 1073741824
cleanup_interval_secs = 30
dedup_lru_capacity = 5000
expired_notification_ttl_secs = 172800
peer_relay_requests_per_minute = 2400
peer_relay_authenticated_requests_per_minute = 480
custody_backup_retention_target_artifacts = 4
custody_backup_retention_target_bytes = 4294967296
custody_backup_partial_grace_secs = 172800
"#;
        let cr: ChatRelayConfig = toml::from_str(toml_str).unwrap();
        assert!(cr.enabled);
        assert_eq!(cr.offline_ttl_secs, 86_400);
        assert_eq!(cr.max_pending_per_wallet, 200);
        assert_eq!(cr.max_pending_messages_total, 40_000);
        assert_eq!(cr.max_pending_message_bytes_total, 268_435_456);
        assert_eq!(cr.db_path, "data/chat_test.db");
        assert_eq!(cr.max_message_size, 32_768);
        assert_eq!(cr.max_blob_size, 5_242_880);
        assert_eq!(cr.max_blobs_per_receiver, 20);
        assert_eq!(cr.max_pending_blobs_total, 2_000);
        assert_eq!(cr.max_pending_blob_bytes_total, 1_073_741_824);
        assert_eq!(cr.cleanup_interval_secs, 30);
        assert_eq!(cr.dedup_lru_capacity, 5_000);
        assert_eq!(cr.expired_notification_ttl_secs, 172_800);
        assert_eq!(cr.peer_relay_requests_per_minute, 2_400);
        assert_eq!(cr.peer_relay_authenticated_requests_per_minute, 480);
        assert_eq!(cr.custody_backup_retention_target_artifacts, 4);
        assert_eq!(cr.custody_backup_retention_target_bytes, 4_294_967_296);
        assert_eq!(cr.custody_backup_partial_grace_secs, 172_800);
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_toml_backward_compat_empty_section() {
        // Missing fields → all defaults applied
        let cr: ChatRelayConfig = toml::from_str("").unwrap();
        assert!(!cr.enabled);
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_toml_backward_compat_defaults_authenticated_rate() {
        // [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] A pre-v1.5
        // operator file remains valid and receives the conservative default.
        let cr: ChatRelayConfig = toml::from_str(
            r#"
enabled = true
peer_relay_requests_per_minute = 1200
"#,
        )
        .unwrap();
        assert_eq!(
            cr.peer_relay_authenticated_requests_per_minute,
            DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE
        );
        assert_eq!(
            cr.custody_backup_retention_target_artifacts,
            DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_ARTIFACTS
        );
        assert_eq!(
            cr.custody_backup_retention_target_bytes,
            DEFAULT_CUSTODY_BACKUP_RETENTION_TARGET_BYTES
        );
        assert_eq!(
            cr.custody_backup_partial_grace_secs,
            DEFAULT_CUSTODY_BACKUP_PARTIAL_GRACE_SECS
        );
        assert!(cr.validate().is_ok());
    }
}
