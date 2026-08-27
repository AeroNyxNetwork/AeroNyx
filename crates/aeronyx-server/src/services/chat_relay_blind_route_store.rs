// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blind_route_store.rs
// ============================================
// Version: 1.0.0-BlindRouteDurableStore
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
//
// Last Modified:
//   v1.0.0-BlindRouteDurableStore - Initial composed SQLite repository
// ============================================

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use super::chat_relay_blind_route::ProtectedBlindRouteResponse;
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Sealed completed route response returned to the cryptographic domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DurableBlindRouteResponse {
    pub(crate) nonce: Vec<u8>,
    pub(crate) ciphertext: Vec<u8>,
    pub(crate) completed_at: u64,
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
}

impl SqliteBlindRouteDurableStore {
    pub(crate) const fn new(
        replay_ttl_secs: u64,
        capacity: usize,
        owner_takeover_grace_secs: u64,
    ) -> Self {
        Self {
            replay_ttl_secs,
            capacity: if capacity == 0 { 1 } else { capacity },
            owner_takeover_grace_secs,
        }
    }

    fn ttl_as_sqlite_integer(self) -> i64 {
        i64::try_from(self.replay_ttl_secs).unwrap_or(i64::MAX)
    }
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
                "SELECT request_fingerprint, response_nonce,
                        response_ciphertext, completed_at
                 FROM relay_blind_route_responses WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        if let Some((stored_fingerprint, nonce, ciphertext, completed_at)) = completed {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_response_fingerprint",
                    })?;
            if stored_fingerprint != *request_fingerprint {
                tx.commit()?;
                return Ok(DurableBlindRouteAdmission::Conflict);
            }
            let completed_at =
                u64::try_from(completed_at).map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_completed_at",
                })?;
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

        let retained = tx.query_row(
            "SELECT
                (SELECT COUNT(*) FROM relay_blind_route_responses)
              + (SELECT COUNT(*) FROM relay_blind_route_reservations)",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if retained < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_retained_count",
            });
        }
        if retained >= capacity {
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
        let completed_at = sqlite_integer(completed_at, "blind_relay_route_completed_at")?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if tx.execute(
            "INSERT OR IGNORE INTO relay_blind_route_responses (
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
        )? > 1
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
        tx.commit()?;
        Ok(())
    }
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}
