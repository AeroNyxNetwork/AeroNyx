// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_error.rs
// ============================================
// Version: 1.0.0-RelayErrorDomain
//
// Creation Reason:
//   [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Extract the stable,
//   typed relay failure contract from the oversized orchestration service so
//   composed storage and protocol domains no longer depend on its facade.
//
// Main Functionality:
//   - Defines every public chat relay failure as a closed enum.
//   - Preserves stable aggregate-only reason buckets for diagnostics.
//   - Classifies capacity failures that require cleanup or operator action.
//   - Exposes the shared `ChatRelayResult<T>` alias.
//
// Dependencies:
//   - `thiserror` provides typed display and source conversion.
//   - `rusqlite` and `bincode` remain transparent source error boundaries.
//
// Main Logical Flow:
//   1. A domain returns a typed failure instead of an unbounded string.
//   2. API callers retain the full typed error for local control flow.
//   3. Telemetry maps it to a stable, privacy-safe reason bucket.
//   4. Capacity callers can fail closed without unsafe blind retries.
//
// Important Note for Next Developer:
//   - Preserve existing variants, field types, display text, and reason buckets.
//   - Never add payloads, identities, endpoints, routes, or raw stored values.
//   - Add new failures as explicit variants with stable aggregate diagnostics.
//   - Keep this module free of storage, network, clock, and process side effects.
//
// Last Modified:
//   v1.0.0-RelayErrorDomain - Initial typed error extraction
// ============================================

/// Errors produced by `ChatRelayService`.
#[derive(Debug, thiserror::Error)]
pub enum ChatRelayError {
    /// `SQLite` schema, query, transaction, or persistence failure.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Another live process owns this custody database, or its fence is unsafe.
    #[error("Chat relay runtime fence unavailable: {reason}")]
    RuntimeFenceUnavailable {
        /// Stable path-free reason suitable for operator diagnostics.
        reason: &'static str,
        /// Existing or fence-specific public diagnostic class.
        public_reason_bucket: &'static str,
    },

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

    /// A client-supplied timestamp cannot be represented by `SQLite INTEGER`.
    #[error("Message timestamp is outside the supported range")]
    TimestampOutOfRange,

    /// An existing durable row uses the same ID for a different signed envelope.
    #[error("Message ID conflicts with an existing durable envelope")]
    MessageIdConflict,

    /// A `ChatPullV2` cursor has an invalid length, version, binding, or tag.
    #[error("Invalid or expired opaque pull cursor")]
    InvalidPullCursor,

    /// The node could not derive or encrypt an opaque `ChatPullV2` cursor.
    #[error("Unable to protect opaque pull cursor")]
    PullCursorEncryptionFailed,

    /// The node could not protect or recover one private verified-submit row.
    #[error("Unable to protect verified submit response")]
    VerifiedSubmitProtectionFailed,

    /// The node could not protect or recover one private blind-route replay row.
    #[error("Unable to protect blind relay replay response")]
    BlindRelayReplayProtectionFailed,

    /// The durable monotonic queue sequence reached `SQLite INTEGER` capacity.
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
            // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] The typed fence
            // maps its closed failure enum before entering this public error.
            // This keeps the existing const reason-bucket API unchanged.
            Self::RuntimeFenceUnavailable {
                public_reason_bucket,
                ..
            } => public_reason_bucket,
            Self::Serialize(_) => "serialization_error",
            Self::AckBatchTooLarge { .. } => "ack_batch_too_large",
            Self::CorruptStoredData { .. } => "corrupt_stored_data",
            Self::TimestampOutOfRange => "timestamp_out_of_range",
            Self::MessageIdConflict => "message_id_conflict",
            Self::InvalidPullCursor => "invalid_pull_cursor",
            Self::PullCursorEncryptionFailed => "pull_cursor_encryption_failed",
            Self::VerifiedSubmitProtectionFailed => "verified_submit_protection_failed",
            Self::BlindRelayReplayProtectionFailed => "blind_relay_replay_protection_failed",
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

/// Shared result boundary for chat relay storage and protocol domains.
pub type ChatRelayResult<T> = Result<T, ChatRelayError>;

#[cfg(test)]
mod tests {
    use super::ChatRelayError;

    type ErrorCase = (ChatRelayError, &'static str, bool);

    fn infrastructure_error_cases() -> Vec<ErrorCase> {
        vec![
            (
                ChatRelayError::Sqlite(rusqlite::Error::InvalidQuery),
                "sqlite_error",
                false,
            ),
            (
                ChatRelayError::RuntimeFenceUnavailable {
                    reason: "test",
                    public_reason_bucket: "runtime_fence_unavailable",
                },
                "runtime_fence_unavailable",
                false,
            ),
            (
                ChatRelayError::Serialize(Box::new(bincode::ErrorKind::Custom("test".to_string()))),
                "serialization_error",
                false,
            ),
            (
                ChatRelayError::AckBatchTooLarge { size: 2, limit: 1 },
                "ack_batch_too_large",
                false,
            ),
            (
                ChatRelayError::CorruptStoredData { field: "test" },
                "corrupt_stored_data",
                false,
            ),
            (
                ChatRelayError::TimestampOutOfRange,
                "timestamp_out_of_range",
                false,
            ),
            (
                ChatRelayError::MessageIdConflict,
                "message_id_conflict",
                false,
            ),
            (
                ChatRelayError::InvalidPullCursor,
                "invalid_pull_cursor",
                false,
            ),
            (
                ChatRelayError::PullCursorEncryptionFailed,
                "pull_cursor_encryption_failed",
                false,
            ),
            (
                ChatRelayError::VerifiedSubmitProtectionFailed,
                "verified_submit_protection_failed",
                false,
            ),
            (
                ChatRelayError::BlindRelayReplayProtectionFailed,
                "blind_relay_replay_protection_failed",
                false,
            ),
            (
                ChatRelayError::QueueSequenceExhausted,
                "queue_sequence_exhausted",
                false,
            ),
        ]
    }

    fn item_error_cases() -> Vec<ErrorCase> {
        vec![
            (
                ChatRelayError::MessageTooLarge { size: 2, limit: 1 },
                "message_too_large",
                false,
            ),
            (
                ChatRelayError::BlobTooLarge { size: 2, limit: 1 },
                "blob_too_large",
                false,
            ),
            (
                ChatRelayError::BlobNotFound {
                    blob_id: "opaque".to_string(),
                },
                "blob_not_found",
                false,
            ),
            (ChatRelayError::Unauthorized, "unauthorized", false),
        ]
    }

    fn capacity_error_cases() -> Vec<ErrorCase> {
        vec![
            (
                ChatRelayError::MailboxFull {
                    current: 1,
                    limit: 1,
                },
                "mailbox_full",
                true,
            ),
            (
                ChatRelayError::PendingMessageQueueFull {
                    current: 1,
                    limit: 1,
                },
                "pending_message_count_quota",
                true,
            ),
            (
                ChatRelayError::PendingMessageBytesExceeded {
                    current: 1,
                    incoming: 1,
                    limit: 1,
                },
                "pending_message_byte_quota",
                true,
            ),
            (
                ChatRelayError::BlobQuotaExceeded {
                    current: 1,
                    limit: 1,
                },
                "receiver_blob_quota",
                true,
            ),
            (
                ChatRelayError::PendingBlobStoreFull {
                    current: 1,
                    limit: 1,
                },
                "pending_blob_count_quota",
                true,
            ),
            (
                ChatRelayError::PendingBlobBytesExceeded {
                    current: 1,
                    incoming: 1,
                    limit: 1,
                },
                "pending_blob_byte_quota",
                true,
            ),
        ]
    }

    #[test]
    fn every_error_has_stable_diagnostics_and_explicit_capacity_semantics() {
        // [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Keep retry admission
        // closed: every current variant is intentionally classified here.
        let cases = infrastructure_error_cases()
            .into_iter()
            .chain(item_error_cases())
            .chain(capacity_error_cases());
        for (error, expected_bucket, expected_capacity) in cases {
            assert_eq!(error.reason_bucket(), expected_bucket);
            assert_eq!(error.is_capacity_exhausted(), expected_capacity);
        }
    }
}
