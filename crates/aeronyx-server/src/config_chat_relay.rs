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
//!
//! ## Main Functionality
//! - `ChatRelayConfig` — all knobs for the zero-knowledge P2P chat relay
//! - `validate()` — gated by `enabled`; skipped entirely when disabled
//! - All fields have sane defaults for backward-compatible TOML loading
//!
//! ## Dependencies
//! - `config_memchain.rs` — embeds `ChatRelayConfig` as `chat_relay` field
//! - `chat_relay_service.rs` — consumes this config at startup
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
//! - `dedup_lru_capacity` is node-wide (not per-wallet). ~64 bytes/entry;
//!   10 000 entries ≈ 640 KB RAM.
//! - `max_message_size`: values > 64 KB emit a warn (UDP fragmentation risk)
//!   AND are now hard-rejected (> MAX_MESSAGE_SIZE_HARD_LIMIT = 1 MB).
//!   Rationale: text chat envelopes should never approach 1 MB; if they do,
//!   it indicates a misconfiguration rather than a legitimate use-case.
//! - `expired_notification_ttl_secs`: after this TTL, undelivered expiry
//!   notifications are silently discarded. Flutter client local timeout is
//!   the fallback.
//! - chat_relay.db_path and saas.data_root share the `"data/"` prefix by
//!   convention but are NOT linked. If you change `saas.data_root`, also
//!   update `chat_relay.db_path` explicitly in your config file.
//!
//! ## Last Modified
//! v1.1.0-ChatRelay — Initial implementation.

use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

/// Hard upper bound for `max_message_size`.
///
/// Text chat envelopes should never legitimately exceed 1 MB.
/// Values above this indicate misconfiguration and are rejected.
const MAX_MESSAGE_SIZE_HARD_LIMIT: usize = 1_048_576; // 1 MB

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
/// corrupt MemChain state and simplifies backup/purge.
///
/// ## Configuration Example
/// ```toml
/// [memchain.chat_relay]
/// enabled = true
/// offline_ttl_secs = 259200       # 72 hours
/// max_pending_per_wallet = 500
/// db_path = "data/chat_pending.db"
/// max_message_size = 65536        # 64 KB (text envelope)
/// max_blob_size = 10485760        # 10 MB (encrypted media)
/// max_blobs_per_receiver = 50
/// cleanup_interval_secs = 60
/// dedup_lru_capacity = 10000
/// expired_notification_ttl_secs = 604800  # 7 days
/// ```
///
/// ## Last Modified
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

    /// Path to the SQLite database file for chat relay storage.
    ///
    /// Stores `pending_messages`, `pending_blobs`, and
    /// `expired_notifications` tables.
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

    /// Interval in seconds between TTL cleanup runs.
    ///
    /// The cleanup task scans `pending_messages` and `pending_blobs` for
    /// expired entries, queues `ChatExpired` notifications, and deletes
    /// delivered/expired rows.
    /// Default: 60.
    #[serde(default = "default_cleanup_interval")]
    pub cleanup_interval_secs: u64,

    /// Capacity of the in-memory LRU deduplication cache for online delivery.
    ///
    /// When a receiver is online, messages are forwarded directly without
    /// hitting SQLite. The LRU cache prevents duplicate delivery if the
    /// sender retransmits before the first ACK arrives.
    ///
    /// Each entry is ~64 bytes (16-byte message_id + LRU overhead).
    /// At capacity 10 000: ~640 KB.
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
}

// ── Default functions ──

fn default_chat_ttl() -> u64 { 259_200 }                // 72 hours
fn default_max_pending_per_wallet() -> usize { 500 }
fn default_chat_db_path() -> String { "data/chat_pending.db".into() }
fn default_max_message_size() -> usize { 65_536 }        // 64 KB
fn default_max_blob_size() -> usize { 10_485_760 }       // 10 MB
fn default_max_blobs_per_receiver() -> usize { 50 }
fn default_cleanup_interval() -> u64 { 60 }
fn default_dedup_lru_capacity() -> usize { 10_000 }
fn default_expired_notification_ttl() -> u64 { 604_800 } // 7 days

impl Default for ChatRelayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            offline_ttl_secs: default_chat_ttl(),
            max_pending_per_wallet: default_max_pending_per_wallet(),
            db_path: default_chat_db_path(),
            max_message_size: default_max_message_size(),
            max_blob_size: default_max_blob_size(),
            max_blobs_per_receiver: default_max_blobs_per_receiver(),
            cleanup_interval_secs: default_cleanup_interval(),
            dedup_lru_capacity: default_dedup_lru_capacity(),
            expired_notification_ttl_secs: default_expired_notification_ttl(),
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

        if self.max_pending_per_wallet == 0 {
            return Err(ServerError::config_invalid(
                "memchain.chat_relay.max_pending_per_wallet",
                "must be > 0",
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
        assert_eq!(cr.db_path, "data/chat_pending.db");
        assert_eq!(cr.max_message_size, 65_536);
        assert_eq!(cr.max_blob_size, 10_485_760);
        assert_eq!(cr.max_blobs_per_receiver, 50);
        assert_eq!(cr.cleanup_interval_secs, 60);
        assert_eq!(cr.dedup_lru_capacity, 10_000);
        assert_eq!(cr.expired_notification_ttl_secs, 604_800);
    }

    #[test]
    fn test_chat_relay_disabled_skips_validation() {
        // All invalid values — must pass because enabled = false
        let cr = ChatRelayConfig {
            enabled: false,
            offline_ttl_secs: 0,
            max_pending_per_wallet: 0,
            db_path: String::new(),
            max_message_size: 0,
            max_blob_size: 0,
            max_blobs_per_receiver: 0,
            cleanup_interval_secs: 0,
            dedup_lru_capacity: 0,
            expired_notification_ttl_secs: 0,
        };
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_enabled_default_valid() {
        let cr = ChatRelayConfig { enabled: true, ..Default::default() };
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_ttl_zero_rejected() {
        let cr = ChatRelayConfig { enabled: true, offline_ttl_secs: 0, ..Default::default() };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_empty_db_path_rejected() {
        let cr = ChatRelayConfig { enabled: true, db_path: String::new(), ..Default::default() };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_message_size_rejected() {
        let cr = ChatRelayConfig { enabled: true, max_message_size: 0, ..Default::default() };
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
        let cr = ChatRelayConfig { enabled: true, max_blob_size: 0, ..Default::default() };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_blobs_per_receiver_rejected() {
        let cr = ChatRelayConfig { enabled: true, max_blobs_per_receiver: 0, ..Default::default() };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_cleanup_interval_rejected() {
        let cr = ChatRelayConfig { enabled: true, cleanup_interval_secs: 0, ..Default::default() };
        assert!(cr.validate().is_err());
    }

    #[test]
    fn test_chat_relay_zero_lru_capacity_rejected() {
        let cr = ChatRelayConfig { enabled: true, dedup_lru_capacity: 0, ..Default::default() };
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
    fn test_chat_relay_toml_parsing() {
        // We can't call `toml::from_str::<ServerConfig>` here (circular dep),
        // but we can test raw TOML → ChatRelayConfig deserialization.
        let toml_str = r#"
enabled = true
offline_ttl_secs = 86400
max_pending_per_wallet = 200
db_path = "data/chat_test.db"
max_message_size = 32768
max_blob_size = 5242880
max_blobs_per_receiver = 20
cleanup_interval_secs = 30
dedup_lru_capacity = 5000
expired_notification_ttl_secs = 172800
"#;
        let cr: ChatRelayConfig = toml::from_str(toml_str).unwrap();
        assert!(cr.enabled);
        assert_eq!(cr.offline_ttl_secs, 86_400);
        assert_eq!(cr.max_pending_per_wallet, 200);
        assert_eq!(cr.db_path, "data/chat_test.db");
        assert_eq!(cr.max_message_size, 32_768);
        assert_eq!(cr.max_blob_size, 5_242_880);
        assert_eq!(cr.max_blobs_per_receiver, 20);
        assert_eq!(cr.cleanup_interval_secs, 30);
        assert_eq!(cr.dedup_lru_capacity, 5_000);
        assert_eq!(cr.expired_notification_ttl_secs, 172_800);
        assert!(cr.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_toml_backward_compat_empty_section() {
        // Missing fields → all defaults applied
        let cr: ChatRelayConfig = toml::from_str("").unwrap();
        assert!(!cr.enabled);
        assert!(cr.validate().is_ok());
    }
}
