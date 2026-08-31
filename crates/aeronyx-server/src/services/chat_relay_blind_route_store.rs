// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blind_route_store.rs
// ============================================
// Version: 1.2.0-ResourceBound
//
// Creation Reason:
//   [BLIND-ROUTE-DURABLE-STORE-DOMAIN 2026-08-27 by Codex] Extract blind-route
//   SQLite replay admission, effect arming, lease takeover, and completion
//   from the oversized relay orchestration service.
//
// Main Functionality:
//   - Defines a replaceable durable blind-route repository capability.
//   - Admits bounded route claims in one immediate SQLite transaction.
//   - Distinguishes unarmed claims from recoverable armed effects.
//   - Fences restart ownership through an exact owner-epoch CAS.
//   - Atomically replaces an owned claim with a sealed exact response.
//   - Rejects impossible stored AEAD shapes before materializing response BLOBs.
//   - Accounts worst-case live opaque bytes in the reservation transaction.
//   - Pauses only new writes when SQLite cannot checkpoint a bounded WAL.
//
// Dependencies:
//   - `chat_relay_blind_route.rs` supplies sealed response data.
//   - `chat_relay_error.rs` supplies stable fail-closed storage failures.
//   - `parking_lot` and `rusqlite` supply locking and transactions.
//
// Main Logical Flow:
//   1. Prune expired private replay evidence before capacity admission.
//   2. Return completed ciphertext, pending ownership, conflict, or capacity.
//   3. Reclaim an eligible foreign-process lease with an exact SQLite CAS.
//   4. Arm the owned claim immediately before an external effect begins.
//   5. Persist a sealed response and remove the exact owner claim atomically.
//
// Important Note for Next Developer:
//   - Never persist raw route ids, peer ids, endpoints, or request plaintext.
//   - Unexpired replay evidence must never be evicted to admit new work.
//   - Only an armed foreign-process claim is eligible for effect recovery.
//   - Completion and release must match the current process owner epoch.
//   - Keep every ownership transition fail-closed and transactionally exact.
//   - Validate BLOB storage classes and lengths before reading their contents.
//   - WAL admission is a recoverable checkpoint-backlog bound, not a disk quota.
//
// Last Modified:
//   v1.2.0-ResourceBound - Added transactional logical bytes and WAL admission
//   v1.1.0-StoredResponseBounds - Bound response BLOB reads and writes
//   v1.0.0-BlindRouteDurableStore - Initial composed SQLite repository
// ============================================

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use super::chat_relay_blind_route::{
    is_valid_protected_blind_route_response_shape, ProtectedBlindRouteResponse,
    MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES, RESPONSE_NONCE_BYTES,
};
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Sealed completed route response returned to the cryptographic domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DurableBlindRouteResponse {
    pub(crate) nonce: Vec<u8>,
    pub(crate) ciphertext: Vec<u8>,
    pub(crate) completed_at: u64,
}

/// Allocation-free metadata inspected before materializing stored BLOBs.
///
/// [BLIND-ROUTE-STORED-RESPONSE-BOUNDS 2026-08-31 by Codex] SQLite can report
/// storage classes and byte lengths without copying the response into a Rust
/// `Vec`, keeping corrupt oversized rows outside the cryptographic boundary.
struct DurableBlindRouteResponseMetadata {
    request_fingerprint: Vec<u8>,
    nonce_storage_class: String,
    nonce_len: Option<i64>,
    ciphertext_storage_class: String,
    ciphertext_len: Option<i64>,
    completed_at: i64,
}

/// Durable admission result before exact response authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DurableBlindRouteAdmission {
    Reserved,
    ReservedForRecovery,
    Pending,
    Conflict,
    Completed(DurableBlindRouteResponse),
    CapacityExhausted,
}

/// Logical live-data and recoverable WAL write-admission boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlindRouteResourceBudget {
    live_opaque_bytes: u64,
    wal_backlog_bytes: u64,
}

/// Cache key, request fingerprint, response nonce, and maximum ciphertext.
pub(crate) const MAX_DURABLE_RESPONSE_OPAQUE_BYTES: u64 =
    32 + 32 + RESPONSE_NONCE_BYTES as u64 + MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES as u64;

/// Conservative allowance for one response write and its B-tree/index pages.
const WAL_WRITE_ADMISSION_HEADROOM_BYTES: u64 = 1024 * 1024;

const SQLITE_WAL_HEADER_BYTES: u64 = 32;
const SQLITE_WAL_FRAME_HEADER_BYTES: u64 = 24;

impl BlindRouteResourceBudget {
    pub(crate) fn new(live_opaque_bytes: u64, wal_backlog_bytes: u64) -> ChatRelayResult<Self> {
        // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] A reservation
        // charges one maximum protected response. Reject a configuration that
        // cannot admit even one legal route or one conservatively-sized write.
        if live_opaque_bytes < MAX_DURABLE_RESPONSE_OPAQUE_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_live_byte_budget",
            });
        }
        if wal_backlog_bytes < WAL_WRITE_ADMISSION_HEADROOM_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_byte_budget",
            });
        }
        Ok(Self {
            live_opaque_bytes,
            wal_backlog_bytes,
        })
    }
}

/// Durable reservation and completion capability for blind relay routes.
pub(crate) trait BlindRouteDurableRepository {
    fn reserve(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<DurableBlindRouteAdmission>;

    fn arm_effect(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        started_at: u64,
    ) -> ChatRelayResult<()>;

    fn release_unarmed(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
    ) -> ChatRelayResult<bool>;

    fn complete(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        protected: ProtectedBlindRouteResponse,
        completed_at: u64,
    ) -> ChatRelayResult<()>;
}

/// Production SQLite repository for blind-route replay state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SqliteBlindRouteDurableStore {
    replay_ttl_secs: u64,
    capacity: usize,
    owner_takeover_grace_secs: u64,
    resource_budget: BlindRouteResourceBudget,
    enforce_wal_guard: bool,
}

impl SqliteBlindRouteDurableStore {
    pub(crate) fn new(
        replay_ttl_secs: u64,
        capacity: usize,
        owner_takeover_grace_secs: u64,
        resource_budget: BlindRouteResourceBudget,
        enforce_wal_guard: bool,
    ) -> Self {
        Self {
            replay_ttl_secs,
            capacity: if capacity == 0 { 1 } else { capacity },
            owner_takeover_grace_secs,
            resource_budget,
            enforce_wal_guard,
        }
    }

    fn ttl_as_sqlite_integer(self) -> i64 {
        i64::try_from(self.replay_ttl_secs).unwrap_or(i64::MAX)
    }

    fn enforce_wal_write_admission(self, connection: &Connection) -> ChatRelayResult<()> {
        if !self.enforce_wal_guard {
            return Ok(());
        }

        let journal_mode =
            connection.query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))?;
        if !journal_mode.eq_ignore_ascii_case("wal") {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_journal_mode",
            });
        }
        // PASSIVE never waits for readers; it also makes the next decision
        // recover immediately after a pinned snapshot is released.
        let (busy, log_frames, checkpointed_frames) =
            connection.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
        if !(0..=1).contains(&busy)
            || log_frames < 0
            || checkpointed_frames < 0
            || checkpointed_frames > log_frames
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_checkpoint_metadata",
            });
        }
        let page_size = connection.query_row("PRAGMA page_size", [], |row| row.get::<_, i64>(0))?;
        if !(512..=65_536).contains(&page_size) || page_size & (page_size - 1) != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_page_size",
            });
        }
        let uncheckpointed_frames =
            u64::try_from(log_frames - checkpointed_frames).map_err(|_| {
                ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_wal_frame_count",
                }
            })?;
        let frame_bytes = u64::try_from(page_size)
            .ok()
            .and_then(|bytes| bytes.checked_add(SQLITE_WAL_FRAME_HEADER_BYTES))
            .ok_or(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_frame_size",
            })?;
        let backlog_bytes = if uncheckpointed_frames == 0 {
            0
        } else {
            uncheckpointed_frames
                .checked_mul(frame_bytes)
                .and_then(|bytes| bytes.checked_add(SQLITE_WAL_HEADER_BYTES))
                .ok_or(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_wal_backlog_size",
                })?
        };
        let projected_bytes = backlog_bytes
            .checked_add(WAL_WRITE_ADMISSION_HEADROOM_BYTES)
            .ok_or(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_wal_projected_size",
            })?;
        if projected_bytes > self.resource_budget.wal_backlog_bytes {
            return Err(ChatRelayError::BlindRelayWalBacklog {
                current_bytes: projected_bytes,
                limit_bytes: self.resource_budget.wal_backlog_bytes,
            });
        }
        Ok(())
    }

    #[cfg(test)]
    fn without_wal_guard(mut self) -> Self {
        self.enforce_wal_guard = false;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetainedLogicalUsage {
    retained_count: i64,
    opaque_bytes: u64,
}

fn retained_logical_usage(connection: &Connection) -> ChatRelayResult<RetainedLogicalUsage> {
    // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] Inspect storage classes
    // and lengths before arithmetic so corrupt rows cannot reduce accounting.
    let invalid_rows = connection.query_row(
        "SELECT
            (SELECT COUNT(*) FROM relay_blind_route_responses
             WHERE TYPEOF(cache_key) != 'blob' OR LENGTH(cache_key) != 32
                OR TYPEOF(request_fingerprint) != 'blob'
                OR LENGTH(request_fingerprint) != 32
                OR TYPEOF(response_nonce) != 'blob' OR LENGTH(response_nonce) != 24
                OR TYPEOF(response_ciphertext) != 'blob'
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > ?1)
          + (SELECT COUNT(*) FROM relay_blind_route_reservations
             WHERE TYPEOF(cache_key) != 'blob' OR LENGTH(cache_key) != 32
                OR TYPEOF(request_fingerprint) != 'blob'
                OR LENGTH(request_fingerprint) != 32
                OR TYPEOF(owner_epoch) != 'blob' OR LENGTH(owner_epoch) != 16)",
        params![i64::try_from(MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES).unwrap_or(i64::MAX)],
        |row| row.get::<_, i64>(0),
    )?;
    if invalid_rows != 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_resource_row_shape",
        });
    }
    let (response_count, response_bytes, reservation_count) = connection.query_row(
        "SELECT
            (SELECT COUNT(*) FROM relay_blind_route_responses),
            (SELECT COALESCE(SUM(
                LENGTH(cache_key) + LENGTH(request_fingerprint)
                + LENGTH(response_nonce) + LENGTH(response_ciphertext)
             ), 0) FROM relay_blind_route_responses),
            (SELECT COUNT(*) FROM relay_blind_route_reservations)",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        },
    )?;
    if response_count < 0 || response_bytes < 0 || reservation_count < 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_resource_accounting",
        });
    }
    let retained_count =
        response_count
            .checked_add(reservation_count)
            .ok_or(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_retained_count",
            })?;
    let reservation_bytes = u64::try_from(reservation_count)
        .ok()
        .and_then(|count| count.checked_mul(MAX_DURABLE_RESPONSE_OPAQUE_BYTES))
        .ok_or(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_reservation_byte_accounting",
        })?;
    let opaque_bytes = u64::try_from(response_bytes)
        .ok()
        .and_then(|bytes| bytes.checked_add(reservation_bytes))
        .ok_or(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_live_byte_accounting",
        })?;
    Ok(RetainedLogicalUsage {
        retained_count,
        opaque_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn probe_non_mutating_admission(
    connection: &Connection,
    cache_key: &[u8; 32],
    request_fingerprint: &[u8; 32],
    process_epoch: &[u8],
    now: i64,
    cutoff: i64,
    owner_takeover_grace_secs: u64,
) -> ChatRelayResult<Option<DurableBlindRouteAdmission>> {
    let completed = connection
        .query_row(
            "SELECT request_fingerprint,
                    typeof(response_nonce), length(response_nonce),
                    typeof(response_ciphertext), length(response_ciphertext),
                    completed_at
             FROM relay_blind_route_responses WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| {
                Ok(DurableBlindRouteResponseMetadata {
                    request_fingerprint: row.get(0)?,
                    nonce_storage_class: row.get(1)?,
                    nonce_len: row.get(2)?,
                    ciphertext_storage_class: row.get(3)?,
                    ciphertext_len: row.get(4)?,
                    completed_at: row.get(5)?,
                })
            },
        )
        .optional()?;
    if let Some(completed) = completed {
        if completed.completed_at < cutoff {
            return Ok(None);
        }
        let stored_fingerprint: [u8; 32] = completed
            .request_fingerprint
            .as_slice()
            .try_into()
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_fingerprint",
            })?;
        if stored_fingerprint != *request_fingerprint {
            return Ok(Some(DurableBlindRouteAdmission::Conflict));
        }
        validate_stored_response_shape(&completed)?;
        let completed_at = u64::try_from(completed.completed_at).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_completed_at",
            }
        })?;
        let (nonce, ciphertext) = connection.query_row(
            "SELECT response_nonce, response_ciphertext
             FROM relay_blind_route_responses WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )?;
        if !is_valid_protected_blind_route_response_shape(nonce.len(), ciphertext.len()) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_materialized_shape",
            });
        }
        return Ok(Some(DurableBlindRouteAdmission::Completed(
            DurableBlindRouteResponse {
                nonce,
                ciphertext,
                completed_at,
            },
        )));
    }

    let reservation = connection
        .query_row(
            "SELECT request_fingerprint, reserved_at, owner_epoch,
                    owner_acquired_at, effect_started_at
             FROM relay_blind_route_reservations WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                ))
            },
        )
        .optional()?;
    let Some((stored_fingerprint, reserved_at, owner_epoch, owner_acquired_at, effect_started_at)) =
        reservation
    else {
        return Ok(None);
    };
    if reserved_at < cutoff {
        return Ok(None);
    }
    let stored_fingerprint: [u8; 32] =
        stored_fingerprint
            .try_into()
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_reservation_fingerprint",
            })?;
    if stored_fingerprint != *request_fingerprint {
        return Ok(Some(DurableBlindRouteAdmission::Conflict));
    }
    if owner_epoch.len() != process_epoch.len()
        || reserved_at < 0
        || owner_acquired_at < reserved_at
        || effect_started_at.is_some_and(|started_at| started_at < reserved_at)
    {
        return Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_reservation_state",
        });
    }
    let reclaim_at = owner_acquired_at
        .saturating_add(i64::try_from(owner_takeover_grace_secs).unwrap_or(i64::MAX));
    if owner_epoch.as_slice() != process_epoch && now >= reclaim_at {
        return Ok(None);
    }
    Ok(Some(DurableBlindRouteAdmission::Pending))
}

impl BlindRouteDurableRepository for SqliteBlindRouteDurableStore {
    fn reserve(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<DurableBlindRouteAdmission> {
        // [BLIND-ROUTE-DURABLE-STORE-DOMAIN 2026-08-27 by Codex] Cleanup,
        // capacity, ownership, and recovery takeover are one transaction.
        let reserved_at = sqlite_integer(now, "blind_relay_route_reserved_at")?;
        let cutoff = reserved_at.saturating_sub(self.ttl_as_sqlite_integer());
        let capacity = i64::try_from(self.capacity).unwrap_or(i64::MAX);
        let mut connection = connection.lock();
        // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] Exact completed,
        // conflict, and still-pending reads do not grow WAL and therefore stay
        // available while a pinned reader pauses new writes. A missing,
        // expired, or reclaimable claim proceeds to guarded write admission.
        if let Some(existing) = probe_non_mutating_admission(
            &connection,
            cache_key,
            request_fingerprint,
            process_epoch,
            reserved_at,
            cutoff,
            self.owner_takeover_grace_secs,
        )? {
            return Ok(existing);
        }
        self.enforce_wal_write_admission(&connection)?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM relay_blind_route_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;

        let completed = tx
            .query_row(
                "SELECT request_fingerprint,
                        typeof(response_nonce), length(response_nonce),
                        typeof(response_ciphertext), length(response_ciphertext),
                        completed_at
                 FROM relay_blind_route_responses WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok(DurableBlindRouteResponseMetadata {
                        request_fingerprint: row.get(0)?,
                        nonce_storage_class: row.get(1)?,
                        nonce_len: row.get(2)?,
                        ciphertext_storage_class: row.get(3)?,
                        ciphertext_len: row.get(4)?,
                        completed_at: row.get(5)?,
                    })
                },
            )
            .optional()?;
        if let Some(completed) = completed {
            let stored_fingerprint: [u8; 32] = completed
                .request_fingerprint
                .as_slice()
                .try_into()
                .map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_fingerprint",
                })?;
            if stored_fingerprint != *request_fingerprint {
                tx.commit()?;
                return Ok(DurableBlindRouteAdmission::Conflict);
            }
            let completed_at = u64::try_from(completed.completed_at).map_err(|_| {
                ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_completed_at",
                }
            })?;
            validate_stored_response_shape(&completed)?;
            let (nonce, ciphertext) = tx.query_row(
                "SELECT response_nonce, response_ciphertext
                 FROM relay_blind_route_responses WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )?;
            if !is_valid_protected_blind_route_response_shape(nonce.len(), ciphertext.len()) {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_materialized_shape",
                });
            }
            tx.commit()?;
            return Ok(DurableBlindRouteAdmission::Completed(
                DurableBlindRouteResponse {
                    nonce,
                    ciphertext,
                    completed_at,
                },
            ));
        }

        let reservation = tx
            .query_row(
                "SELECT request_fingerprint, reserved_at, owner_epoch,
                        owner_acquired_at, effect_started_at
                 FROM relay_blind_route_reservations
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<i64>>(4)?,
                    ))
                },
            )
            .optional()?;
        if let Some((
            stored_fingerprint,
            stored_at,
            owner_epoch,
            owner_acquired_at,
            effect_started_at,
        )) = reservation
        {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_reservation_fingerprint",
                    })?;
            if stored_fingerprint != *request_fingerprint {
                tx.commit()?;
                return Ok(DurableBlindRouteAdmission::Conflict);
            }
            if owner_epoch.len() != process_epoch.len() {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_reservation_owner_epoch",
                });
            }
            if stored_at < 0
                || owner_acquired_at < stored_at
                || effect_started_at.is_some_and(|started_at| started_at < stored_at)
            {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_reservation_state",
                });
            }
            let reclaim_at = owner_acquired_at
                .saturating_add(i64::try_from(self.owner_takeover_grace_secs).unwrap_or(i64::MAX));
            if owner_epoch.as_slice() != process_epoch && reserved_at >= reclaim_at {
                if tx.execute(
                    "UPDATE relay_blind_route_reservations
                     SET owner_epoch = ?1, owner_acquired_at = ?2
                     WHERE cache_key = ?3
                       AND request_fingerprint = ?4
                       AND reserved_at = ?5
                       AND owner_epoch = ?6
                       AND owner_acquired_at = ?7",
                    params![
                        process_epoch,
                        reserved_at,
                        cache_key.as_slice(),
                        request_fingerprint.as_slice(),
                        stored_at,
                        owner_epoch.as_slice(),
                        owner_acquired_at,
                    ],
                )? != 1
                {
                    return Err(ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_reservation_takeover",
                    });
                }
                let admission = if effect_started_at.is_some() {
                    DurableBlindRouteAdmission::ReservedForRecovery
                } else {
                    DurableBlindRouteAdmission::Reserved
                };
                tx.commit()?;
                return Ok(admission);
            }
            tx.commit()?;
            return Ok(DurableBlindRouteAdmission::Pending);
        }

        let usage = retained_logical_usage(&tx)?;
        if usage.retained_count >= capacity
            || usage
                .opaque_bytes
                .checked_add(MAX_DURABLE_RESPONSE_OPAQUE_BYTES)
                .is_none_or(|next| next > self.resource_budget.live_opaque_bytes)
        {
            tx.commit()?;
            return Ok(DurableBlindRouteAdmission::CapacityExhausted);
        }
        if tx.execute(
            "INSERT INTO relay_blind_route_reservations (
                cache_key, request_fingerprint, reserved_at, owner_epoch,
                owner_acquired_at, effect_started_at
             ) VALUES (?1, ?2, ?3, ?4, ?3, NULL)",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                reserved_at,
                process_epoch,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_reservation_insert",
            });
        }
        tx.commit()?;
        Ok(DurableBlindRouteAdmission::Reserved)
    }

    fn arm_effect(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        started_at: u64,
    ) -> ChatRelayResult<()> {
        let started_at = sqlite_integer(started_at, "blind_relay_route_effect_started_at")?;
        let connection = connection.lock();
        self.enforce_wal_write_admission(&connection)?;
        if connection.execute(
            "UPDATE relay_blind_route_reservations
             SET effect_started_at = MAX(?1, reserved_at)
             WHERE cache_key = ?2
               AND request_fingerprint = ?3
               AND owner_epoch = ?4
               AND effect_started_at IS NULL",
            params![
                started_at,
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                process_epoch,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_effect_admission",
            });
        }
        Ok(())
    }

    fn release_unarmed(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
    ) -> ChatRelayResult<bool> {
        let connection = connection.lock();
        Ok(connection.execute(
            "DELETE FROM relay_blind_route_reservations
             WHERE cache_key = ?1
               AND request_fingerprint = ?2
               AND owner_epoch = ?3
               AND effect_started_at IS NULL",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                process_epoch,
            ],
        )? == 1)
    }

    fn complete(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        protected: ProtectedBlindRouteResponse,
        completed_at: u64,
    ) -> ChatRelayResult<()> {
        // [BLIND-ROUTE-STORED-RESPONSE-BOUNDS 2026-08-31 by Codex] Reject an
        // impossible protector implementation before opening a write
        // transaction or moving its ciphertext into rusqlite.
        if !is_valid_protected_blind_route_response_shape(
            protected.nonce.len(),
            protected.ciphertext.len(),
        ) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_protected_response_shape",
            });
        }
        let completed_at = sqlite_integer(completed_at, "blind_relay_route_completed_at")?;
        let mut connection = connection.lock();
        self.enforce_wal_write_admission(&connection)?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // [BLIND-ROUTE-STRICT-COMPLETION 2026-08-31 by Codex] A pre-existing
        // response is an impossible state while this exact reservation is
        // owned. Do not ignore that conflict and then delete the only owner
        // fence; let the transaction fail and preserve both rows for recovery.
        if tx.execute(
            "INSERT INTO relay_blind_route_responses (
                cache_key, request_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                protected.nonce.as_slice(),
                protected.ciphertext,
                completed_at,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_insert",
            });
        }
        let stored_fingerprint = tx.query_row(
            "SELECT request_fingerprint FROM relay_blind_route_responses
             WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )?;
        if stored_fingerprint.as_slice() != request_fingerprint.as_slice() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_insert_conflict",
            });
        }
        if tx.execute(
            "DELETE FROM relay_blind_route_reservations
             WHERE cache_key = ?1
               AND request_fingerprint = ?2
               AND owner_epoch = ?3",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                process_epoch,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_reservation_completion",
            });
        }
        let usage = retained_logical_usage(&tx)?;
        if usage.retained_count > i64::try_from(self.capacity).unwrap_or(i64::MAX)
            || usage.opaque_bytes > self.resource_budget.live_opaque_bytes
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_completion_resource_budget",
            });
        }
        tx.commit()?;
        Ok(())
    }
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

fn validate_stored_response_shape(
    metadata: &DurableBlindRouteResponseMetadata,
) -> ChatRelayResult<()> {
    if metadata.nonce_storage_class != "blob" || metadata.ciphertext_storage_class != "blob" {
        return Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_response_storage_class",
        });
    }
    let nonce_len = sqlite_blob_length(
        metadata.nonce_len,
        "blind_relay_route_response_nonce_length",
    )?;
    let ciphertext_len = sqlite_blob_length(
        metadata.ciphertext_len,
        "blind_relay_route_response_ciphertext_length",
    )?;
    if !is_valid_protected_blind_route_response_shape(nonce_len, ciphertext_len) {
        return Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_response_stored_shape",
        });
    }
    Ok(())
}

fn sqlite_blob_length(length: Option<i64>, field: &'static str) -> ChatRelayResult<usize> {
    let length = length.ok_or(ChatRelayError::CorruptStoredData { field })?;
    usize::try_from(length).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::chat_relay_blind_route::{
        MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES, RESPONSE_NONCE_BYTES,
    };

    const TEST_NOW: u64 = 1_800_000_000;
    const TEST_PROCESS_EPOCH: &[u8; 16] = b"0123456789abcdef";
    const TEST_NEXT_PROCESS_EPOCH: &[u8; 16] = b"fedcba9876543210";

    fn guarded_store(live_bytes: u64, wal_bytes: u64) -> SqliteBlindRouteDurableStore {
        SqliteBlindRouteDurableStore::new(
            90,
            8,
            30,
            BlindRouteResourceBudget::new(live_bytes, wal_bytes)
                .expect("valid guarded test resource budget"),
            true,
        )
    }

    fn test_connection() -> Mutex<Connection> {
        let connection = Connection::open_in_memory().expect("open durable-store test database");
        connection
            .execute_batch(
                "CREATE TABLE relay_blind_route_responses (
                    cache_key           BLOB PRIMARY KEY,
                    request_fingerprint BLOB NOT NULL,
                    response_nonce      BLOB NOT NULL,
                    response_ciphertext BLOB NOT NULL,
                    completed_at        INTEGER NOT NULL
                 );
                 CREATE TABLE relay_blind_route_reservations (
                    cache_key           BLOB PRIMARY KEY,
                    request_fingerprint BLOB NOT NULL,
                    reserved_at         INTEGER NOT NULL,
                    owner_epoch         BLOB NOT NULL,
                    owner_acquired_at   INTEGER NOT NULL,
                    effect_started_at   INTEGER
                 );",
            )
            .expect("create durable-store test tables");
        Mutex::new(connection)
    }

    fn store() -> SqliteBlindRouteDurableStore {
        SqliteBlindRouteDurableStore::new(
            90,
            8,
            30,
            BlindRouteResourceBudget::new(8 * MAX_DURABLE_RESPONSE_OPAQUE_BYTES, 8 * 1024 * 1024)
                .expect("valid test resource budget"),
            true,
        )
        .without_wal_guard()
    }

    fn insert_owned_reservation(
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        process_epoch: &[u8],
    ) {
        connection
            .lock()
            .execute(
                "INSERT INTO relay_blind_route_reservations (
                    cache_key, request_fingerprint, reserved_at, owner_epoch,
                    owner_acquired_at, effect_started_at
                 ) VALUES (?1, ?2, ?3, ?4, ?3, NULL)",
                params![
                    cache_key.as_slice(),
                    request_fingerprint.as_slice(),
                    TEST_NOW as i64,
                    process_epoch,
                ],
            )
            .expect("insert owned test reservation");
    }

    #[test]
    fn reserve_rejects_invalid_blob_shapes_before_materialization() {
        // [BLIND-ROUTE-STORED-RESPONSE-BOUNDS 2026-08-31 by Codex] `zeroblob`
        // creates corrupt rows without constructing their contents in Rust.
        for (nonce_len, ciphertext_len) in [
            (RESPONSE_NONCE_BYTES + 1, 17),
            (
                RESPONSE_NONCE_BYTES,
                MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES + 1,
            ),
        ] {
            let connection = test_connection();
            let cache_key = [0xA1; 32];
            let request_fingerprint = [0xA2; 32];
            connection
                .lock()
                .execute(
                    "INSERT INTO relay_blind_route_responses (
                        cache_key, request_fingerprint, response_nonce,
                        response_ciphertext, completed_at
                     ) VALUES (?1, ?2, zeroblob(?3), zeroblob(?4), ?5)",
                    params![
                        cache_key.as_slice(),
                        request_fingerprint.as_slice(),
                        i64::try_from(nonce_len).expect("nonce test length fits SQLite"),
                        i64::try_from(ciphertext_len).expect("ciphertext test length fits SQLite"),
                        TEST_NOW as i64,
                    ],
                )
                .expect("insert corrupt response shape");

            assert!(matches!(
                store().reserve(
                    &connection,
                    &cache_key,
                    &request_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                ),
                Err(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_stored_shape"
                })
            ));
        }
    }

    #[test]
    fn reserve_rejects_non_blob_response_storage() {
        let connection = test_connection();
        let cache_key = [0xB1; 32];
        let request_fingerprint = [0xB2; 32];
        connection
            .lock()
            .execute(
                "INSERT INTO relay_blind_route_responses (
                    cache_key, request_fingerprint, response_nonce,
                    response_ciphertext, completed_at
                 ) VALUES (?1, ?2, 'not-a-blob', zeroblob(17), ?3)",
                params![
                    cache_key.as_slice(),
                    request_fingerprint.as_slice(),
                    TEST_NOW as i64,
                ],
            )
            .expect("insert non-BLOB response nonce");

        assert!(matches!(
            store().reserve(
                &connection,
                &cache_key,
                &request_fingerprint,
                TEST_PROCESS_EPOCH,
                TEST_NOW,
            ),
            Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_storage_class"
            })
        ));
    }

    #[test]
    fn completion_rejects_impossible_shape_without_mutating_reservation() {
        let connection = test_connection();
        let cache_key = [0xC1; 32];
        let request_fingerprint = [0xC2; 32];
        let process_epoch = TEST_PROCESS_EPOCH;
        insert_owned_reservation(&connection, &cache_key, &request_fingerprint, process_epoch);
        let protected = ProtectedBlindRouteResponse {
            nonce: [0; RESPONSE_NONCE_BYTES],
            ciphertext: vec![0; MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES + 1],
        };

        assert!(matches!(
            store().complete(
                &connection,
                &cache_key,
                &request_fingerprint,
                process_epoch,
                protected,
                TEST_NOW,
            ),
            Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_protected_response_shape"
            })
        ));
        let connection = connection.lock();
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM relay_blind_route_reservations",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .expect("count retained reservation"),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM relay_blind_route_responses",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .expect("count absent response"),
            0
        );
    }

    #[test]
    fn maximum_valid_completion_round_trips_through_bounded_read() {
        let connection = test_connection();
        let cache_key = [0xD1; 32];
        let request_fingerprint = [0xD2; 32];
        let process_epoch = TEST_PROCESS_EPOCH;
        insert_owned_reservation(&connection, &cache_key, &request_fingerprint, process_epoch);
        let protected = ProtectedBlindRouteResponse {
            nonce: [0xD3; RESPONSE_NONCE_BYTES],
            ciphertext: vec![0xD4; MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES],
        };
        store()
            .complete(
                &connection,
                &cache_key,
                &request_fingerprint,
                process_epoch,
                protected,
                TEST_NOW,
            )
            .expect("complete bounded durable response");

        let DurableBlindRouteAdmission::Completed(completed) = store()
            .reserve(
                &connection,
                &cache_key,
                &request_fingerprint,
                TEST_NEXT_PROCESS_EPOCH,
                TEST_NOW,
            )
            .expect("read bounded durable response")
        else {
            panic!("completed response must remain restart-replayable");
        };
        assert_eq!(completed.nonce, vec![0xD3; RESPONSE_NONCE_BYTES]);
        assert_eq!(
            completed.ciphertext,
            vec![0xD4; MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES]
        );
        assert_eq!(completed.completed_at, TEST_NOW);
    }

    #[test]
    fn completion_does_not_ignore_an_existing_response() {
        let connection = test_connection();
        let cache_key = [0xE1; 32];
        let request_fingerprint = [0xE2; 32];
        let process_epoch = TEST_PROCESS_EPOCH;
        insert_owned_reservation(&connection, &cache_key, &request_fingerprint, process_epoch);
        connection
            .lock()
            .execute(
                "INSERT INTO relay_blind_route_responses (
                    cache_key, request_fingerprint, response_nonce,
                    response_ciphertext, completed_at
                 ) VALUES (?1, ?2, zeroblob(?3), ?4, ?5)",
                params![
                    cache_key.as_slice(),
                    request_fingerprint.as_slice(),
                    RESPONSE_NONCE_BYTES as i64,
                    vec![0xE3_u8; 17],
                    TEST_NOW as i64,
                ],
            )
            .expect("insert existing response");
        let protected = ProtectedBlindRouteResponse {
            nonce: [0xE4; RESPONSE_NONCE_BYTES],
            ciphertext: vec![0xE5; 17],
        };

        assert!(store()
            .complete(
                &connection,
                &cache_key,
                &request_fingerprint,
                process_epoch,
                protected,
                TEST_NOW,
            )
            .is_err());
        let connection = connection.lock();
        let stored_ciphertext = connection
            .query_row(
                "SELECT response_ciphertext FROM relay_blind_route_responses",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .expect("read preserved existing response");
        assert_eq!(stored_ciphertext, vec![0xE3; 17]);
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM relay_blind_route_reservations",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .expect("count preserved owner reservation"),
            1
        );
    }

    #[test]
    fn logical_byte_reservation_is_atomic_and_ttl_releases_capacity() {
        // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] One reservation
        // consumes one maximum completed response, so completion cannot grow
        // beyond admitted bytes. Expiry is part of the next admission tx.
        let connection = test_connection();
        let store = SqliteBlindRouteDurableStore::new(
            90,
            8,
            30,
            BlindRouteResourceBudget::new(
                MAX_DURABLE_RESPONSE_OPAQUE_BYTES,
                WAL_WRITE_ADMISSION_HEADROOM_BYTES,
            )
            .expect("one-row resource budget"),
            true,
        )
        .without_wal_guard();
        let first_key = [0xF1; 32];
        let first_fingerprint = [0xF2; 32];
        let second_key = [0xF3; 32];
        let second_fingerprint = [0xF4; 32];

        assert_eq!(
            store
                .reserve(
                    &connection,
                    &first_key,
                    &first_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                )
                .expect("reserve first logical row"),
            DurableBlindRouteAdmission::Reserved
        );
        assert_eq!(
            store
                .reserve(
                    &connection,
                    &second_key,
                    &second_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                )
                .expect("reject second logical row"),
            DurableBlindRouteAdmission::CapacityExhausted
        );
        assert_eq!(
            store
                .reserve(
                    &connection,
                    &first_key,
                    &first_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                )
                .expect("preserve exact pending retry"),
            DurableBlindRouteAdmission::Pending
        );
        assert_eq!(
            store
                .reserve(
                    &connection,
                    &second_key,
                    &second_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW + 91,
                )
                .expect("release expired logical bytes"),
            DurableBlindRouteAdmission::Reserved
        );
    }

    #[test]
    fn concurrent_logical_byte_admission_has_one_winner() {
        use std::sync::{Arc, Barrier};

        let connection = Arc::new(test_connection());
        let store = SqliteBlindRouteDurableStore::new(
            90,
            8,
            30,
            BlindRouteResourceBudget::new(
                MAX_DURABLE_RESPONSE_OPAQUE_BYTES,
                WAL_WRITE_ADMISSION_HEADROOM_BYTES,
            )
            .expect("one-row concurrent budget"),
            true,
        )
        .without_wal_guard();
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for discriminator in [0xA5_u8, 0xB5_u8] {
            let connection = Arc::clone(&connection);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                let key = [discriminator; 32];
                let fingerprint = [discriminator.wrapping_add(1); 32];
                barrier.wait();
                store
                    .reserve(
                        &connection,
                        &key,
                        &fingerprint,
                        TEST_PROCESS_EPOCH,
                        TEST_NOW,
                    )
                    .expect("concurrent resource admission")
            }));
        }
        barrier.wait();
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().expect("join admission thread"))
            .collect();
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == DurableBlindRouteAdmission::Reserved)
                .count(),
            1
        );
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == DurableBlindRouteAdmission::CapacityExhausted)
                .count(),
            1
        );
    }

    #[test]
    fn pinned_reader_stops_new_writes_but_retry_delete_and_recovery_continue() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "aeronyx-blind-route-wal-{}-{unique}.sqlite",
            std::process::id()
        ));
        let writer = Connection::open(&path).expect("open WAL writer");
        writer
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=FULL;
                 PRAGMA wal_autocheckpoint=0;
                 CREATE TABLE relay_blind_route_responses (
                    cache_key BLOB PRIMARY KEY, request_fingerprint BLOB NOT NULL,
                    response_nonce BLOB NOT NULL, response_ciphertext BLOB NOT NULL,
                    completed_at INTEGER NOT NULL
                 );
                 CREATE TABLE relay_blind_route_reservations (
                    cache_key BLOB PRIMARY KEY, request_fingerprint BLOB NOT NULL,
                    reserved_at INTEGER NOT NULL, owner_epoch BLOB NOT NULL,
                    owner_acquired_at INTEGER NOT NULL, effect_started_at INTEGER
                 );",
            )
            .expect("create WAL test schema");
        let retry_key = [0x71; 32];
        let retry_fingerprint = [0x72; 32];
        let release_key = [0x73; 32];
        let release_fingerprint = [0x74; 32];
        writer
            .execute(
                "INSERT INTO relay_blind_route_responses VALUES (?1, ?2, zeroblob(24), zeroblob(17), ?3)",
                params![retry_key.as_slice(), retry_fingerprint.as_slice(), TEST_NOW as i64],
            )
            .expect("seed exact retry response");
        writer
            .execute(
                "INSERT INTO relay_blind_route_reservations VALUES (?1, ?2, ?3, ?4, ?3, NULL)",
                params![
                    release_key.as_slice(),
                    release_fingerprint.as_slice(),
                    TEST_NOW as i64,
                    TEST_PROCESS_EPOCH.as_slice()
                ],
            )
            .expect("seed releasable reservation");
        writer
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .expect("reset WAL before pinned snapshot");

        let reader = Connection::open(&path).expect("open pinned reader");
        reader.execute_batch("BEGIN").expect("begin pinned read");
        reader
            .query_row(
                "SELECT COUNT(*) FROM relay_blind_route_reservations",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("establish pinned snapshot");
        for discriminator in 0_u8..8 {
            let churn_key = [discriminator; 32];
            writer
                .execute(
                    "INSERT INTO relay_blind_route_reservations VALUES (?1, zeroblob(32), ?2, zeroblob(16), ?2, NULL)",
                    params![churn_key.as_slice(), TEST_NOW as i64],
                )
                .expect("append pinned WAL frame");
            writer
                .execute(
                    "DELETE FROM relay_blind_route_reservations WHERE cache_key = ?1",
                    params![churn_key.as_slice()],
                )
                .expect("append pinned WAL delete frame");
        }

        let connection = Mutex::new(writer);
        let store = guarded_store(
            8 * MAX_DURABLE_RESPONSE_OPAQUE_BYTES,
            WAL_WRITE_ADMISSION_HEADROOM_BYTES,
        );
        let new_key = [0x75; 32];
        let new_fingerprint = [0x76; 32];
        assert!(matches!(
            store.reserve(
                &connection,
                &new_key,
                &new_fingerprint,
                TEST_PROCESS_EPOCH,
                TEST_NOW,
            ),
            Err(ChatRelayError::BlindRelayWalBacklog { .. })
        ));
        assert!(matches!(
            store
                .reserve(
                    &connection,
                    &retry_key,
                    &retry_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                )
                .expect("exact retry remains readable"),
            DurableBlindRouteAdmission::Completed(_)
        ));
        assert!(store
            .release_unarmed(
                &connection,
                &release_key,
                &release_fingerprint,
                TEST_PROCESS_EPOCH,
            )
            .expect("delete remains available over WAL boundary"));

        reader
            .execute_batch("COMMIT")
            .expect("release pinned snapshot");
        drop(reader);
        assert_eq!(
            store
                .reserve(
                    &connection,
                    &new_key,
                    &new_fingerprint,
                    TEST_PROCESS_EPOCH,
                    TEST_NOW,
                )
                .expect("recover after checkpoint progress"),
            DurableBlindRouteAdmission::Reserved
        );
        drop(connection);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
    }
}
