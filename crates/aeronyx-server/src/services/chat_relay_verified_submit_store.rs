// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_verified_submit_store.rs
// ============================================
// Version: 1.0.0-VerifiedSubmitDurableStore
//
// Creation Reason:
//   [VERIFIED-SUBMIT-DURABLE-STORE-DOMAIN 2026-08-27 by Codex] Extract the
//   verified-submit SQLite lookup, reservation, lease takeover, capacity, and
//   completion state machine from the oversized relay orchestration service.
//
// Main Functionality:
//   - Defines a replaceable durable verified-submit repository capability.
//   - Reads completed sealed responses and unfinished private reservations.
//   - Expires stale evidence with compare-and-delete semantics.
//   - Reserves bounded replay capacity in one immediate transaction.
//   - Fences restart recovery through an owner-epoch compare-and-swap.
//   - Atomically replaces an owned reservation with a completed response.
//
// Dependencies:
//   - `chat_relay_verified_submit.rs` supplies typed admission and sealed data.
//   - `chat_relay_error.rs` supplies stable fail-closed storage failures.
//   - `parking_lot` and `rusqlite` supply locking and durable transactions.
//
// Main Logical Flow:
//   1. Receive only node-private fingerprints after request authentication.
//   2. Return a fresh completed response, pending claim, conflict, or miss.
//   3. Prune expired evidence before capacity admission.
//   4. Reclaim an eligible foreign-process lease with an exact SQLite CAS.
//   5. Persist a sealed result and delete the exact owned reservation atomically.
//
// Important Note for Next Developer:
//   - Never accept or persist raw request ids, message ids, routes, or peers.
//   - Unexpired replay evidence must never be evicted to admit new work.
//   - Recovery may repeat only the caller-approved idempotent entry effect.
//   - Reservation completion must verify the current process owner epoch.
//   - Keep lookup expiry deletion bound to both private key and timestamp.
//
// Last Modified:
//   v1.0.0-VerifiedSubmitDurableStore - Initial composed SQLite repository
// ============================================

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_verified_submit::{ProtectedVerifiedSubmitResponse, VerifiedSubmitAdmission};

/// Sealed completed response returned without parsing protocol plaintext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DurableVerifiedSubmitResponse {
    pub(crate) nonce: Vec<u8>,
    pub(crate) ciphertext: Vec<u8>,
}

/// Durable lookup result before response authentication and deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DurableVerifiedSubmitLookup {
    Miss,
    Conflict,
    Pending,
    Completed(DurableVerifiedSubmitResponse),
}

/// Durable reservation and completion capability for verified submissions.
pub(crate) trait VerifiedSubmitDurableRepository {
    fn lookup(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        now: u64,
    ) -> ChatRelayResult<DurableVerifiedSubmitLookup>;

    fn reserve(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<VerifiedSubmitAdmission>;

    fn complete(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        protected: ProtectedVerifiedSubmitResponse,
        now: u64,
    ) -> ChatRelayResult<()>;
}

/// Production SQLite verified-submit repository.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SqliteVerifiedSubmitDurableStore {
    response_ttl_secs: u64,
    capacity: usize,
    owner_takeover_grace_secs: u64,
}

impl SqliteVerifiedSubmitDurableStore {
    pub(crate) const fn new(
        response_ttl_secs: u64,
        capacity: usize,
        owner_takeover_grace_secs: u64,
    ) -> Self {
        Self {
            response_ttl_secs,
            capacity: if capacity == 0 { 1 } else { capacity },
            owner_takeover_grace_secs,
        }
    }

    fn ttl_as_sqlite_integer(self) -> i64 {
        i64::try_from(self.response_ttl_secs).unwrap_or(i64::MAX)
    }
}

impl VerifiedSubmitDurableRepository for SqliteVerifiedSubmitDurableStore {
    fn lookup(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        now: u64,
    ) -> ChatRelayResult<DurableVerifiedSubmitLookup> {
        let now = sqlite_integer(now, "verified_submit_response_lookup_time")?;
        let durable_row = {
            let connection = connection.lock();
            connection
                .query_row(
                    "SELECT envelope_fingerprint, response_nonce,
                            response_ciphertext, completed_at
                     FROM relay_verified_submit_responses
                     WHERE cache_key = ?1",
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
                .optional()?
        };
        let Some((stored_fingerprint, nonce, ciphertext, completed_at)) = durable_row else {
            let reservation = {
                let connection = connection.lock();
                connection
                    .query_row(
                        "SELECT envelope_fingerprint, reserved_at
                         FROM relay_verified_submit_reservations
                         WHERE cache_key = ?1",
                        params![cache_key.as_slice()],
                        |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .optional()?
            };
            let Some((stored_fingerprint, reserved_at)) = reservation else {
                return Ok(DurableVerifiedSubmitLookup::Miss);
            };
            if reserved_at < 0 || now.saturating_sub(reserved_at) > self.ttl_as_sqlite_integer() {
                let connection = connection.lock();
                connection.execute(
                    "DELETE FROM relay_verified_submit_reservations
                     WHERE cache_key = ?1 AND reserved_at = ?2",
                    params![cache_key.as_slice(), reserved_at],
                )?;
                return Ok(DurableVerifiedSubmitLookup::Miss);
            }
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_envelope_fingerprint",
                    })?;
            return if stored_fingerprint == *envelope_fingerprint {
                Ok(DurableVerifiedSubmitLookup::Pending)
            } else {
                Ok(DurableVerifiedSubmitLookup::Conflict)
            };
        };

        if completed_at < 0 || now.saturating_sub(completed_at) > self.ttl_as_sqlite_integer() {
            let connection = connection.lock();
            connection.execute(
                "DELETE FROM relay_verified_submit_responses
                 WHERE cache_key = ?1 AND completed_at = ?2",
                params![cache_key.as_slice(), completed_at],
            )?;
            return Ok(DurableVerifiedSubmitLookup::Miss);
        }
        let stored_fingerprint: [u8; 32] =
            stored_fingerprint
                .try_into()
                .map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "verified_submit_response_envelope_fingerprint",
                })?;
        if stored_fingerprint != *envelope_fingerprint {
            return Ok(DurableVerifiedSubmitLookup::Conflict);
        }
        Ok(DurableVerifiedSubmitLookup::Completed(
            DurableVerifiedSubmitResponse { nonce, ciphertext },
        ))
    }

    fn reserve(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<VerifiedSubmitAdmission> {
        // [VERIFIED-SUBMIT-DURABLE-STORE-DOMAIN 2026-08-27 by Codex]
        // Capacity, ownership, and stale-evidence cleanup are one transaction.
        let reserved_at = sqlite_integer(now, "verified_submit_reservation_time")?;
        let cutoff = reserved_at.saturating_sub(self.ttl_as_sqlite_integer());
        let capacity = sqlite_integer(
            u64::try_from(self.capacity).unwrap_or(u64::MAX),
            "verified_submit_reservation_capacity",
        )?;

        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM relay_verified_submit_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;

        let completed_fingerprint = tx
            .query_row(
                "SELECT envelope_fingerprint FROM relay_verified_submit_responses
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if let Some(stored_fingerprint) = completed_fingerprint {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_response_envelope_fingerprint",
                    })?;
            let outcome = if stored_fingerprint == *envelope_fingerprint {
                VerifiedSubmitAdmission::Completed
            } else {
                VerifiedSubmitAdmission::Conflict
            };
            tx.commit()?;
            return Ok(outcome);
        }

        let existing_reservation = tx
            .query_row(
                "SELECT envelope_fingerprint, reserved_at, owner_epoch,
                        owner_acquired_at
                 FROM relay_verified_submit_reservations
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        if let Some((stored_fingerprint, stored_at, owner_epoch, owner_acquired_at)) =
            existing_reservation
        {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_envelope_fingerprint",
                    })?;
            if stored_fingerprint != *envelope_fingerprint {
                tx.commit()?;
                return Ok(VerifiedSubmitAdmission::Conflict);
            }
            if owner_epoch.len() != process_epoch.len() {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "verified_submit_reservation_owner_epoch",
                });
            }
            if stored_at < 0 || owner_acquired_at < stored_at {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "verified_submit_reservation_state",
                });
            }
            let reclaim_at = owner_acquired_at
                .saturating_add(i64::try_from(self.owner_takeover_grace_secs).unwrap_or(i64::MAX));
            let outcome = if owner_epoch.as_slice() != process_epoch && reserved_at >= reclaim_at {
                if tx.execute(
                    "UPDATE relay_verified_submit_reservations
                     SET owner_epoch = ?1, owner_acquired_at = ?2
                     WHERE cache_key = ?3
                       AND envelope_fingerprint = ?4
                       AND reserved_at = ?5
                       AND owner_epoch = ?6
                       AND owner_acquired_at = ?7",
                    params![
                        process_epoch,
                        reserved_at,
                        cache_key.as_slice(),
                        envelope_fingerprint.as_slice(),
                        stored_at,
                        owner_epoch.as_slice(),
                        owner_acquired_at,
                    ],
                )? != 1
                {
                    return Err(ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_takeover",
                    });
                }
                VerifiedSubmitAdmission::ReservedForEntryRecovery
            } else {
                VerifiedSubmitAdmission::Pending
            };
            tx.commit()?;
            return Ok(outcome);
        }

        let retained = tx.query_row(
            "SELECT
                (SELECT COUNT(*) FROM relay_verified_submit_responses)
              + (SELECT COUNT(*) FROM relay_verified_submit_reservations)",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if retained < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_retained_count",
            });
        }
        if retained >= capacity {
            tx.commit()?;
            return Ok(VerifiedSubmitAdmission::CapacityExhausted);
        }
        if tx.execute(
            "INSERT INTO relay_verified_submit_reservations (
                cache_key, envelope_fingerprint, reserved_at,
                owner_epoch, owner_acquired_at
             ) VALUES (?1, ?2, ?3, ?4, ?3)",
            params![
                cache_key.as_slice(),
                envelope_fingerprint.as_slice(),
                reserved_at,
                process_epoch,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_insert",
            });
        }
        tx.commit()?;
        Ok(VerifiedSubmitAdmission::Reserved)
    }

    fn complete(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        process_epoch: &[u8],
        protected: ProtectedVerifiedSubmitResponse,
        now: u64,
    ) -> ChatRelayResult<()> {
        let completed_at = sqlite_integer(now, "verified_submit_response_completed_at")?;
        let mut connection = connection.lock();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT OR IGNORE INTO relay_verified_submit_responses (
                cache_key, envelope_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                cache_key.as_slice(),
                envelope_fingerprint.as_slice(),
                protected.nonce.as_slice(),
                protected.ciphertext,
                completed_at,
            ],
        )?;
        let stored_fingerprint = tx.query_row(
            "SELECT envelope_fingerprint
             FROM relay_verified_submit_responses
             WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )?;
        if stored_fingerprint.as_slice() != envelope_fingerprint.as_slice() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_insert_conflict",
            });
        }
        if tx.execute(
            "DELETE FROM relay_verified_submit_reservations
             WHERE cache_key = ?1
               AND envelope_fingerprint = ?2
               AND owner_epoch = ?3",
            params![
                cache_key.as_slice(),
                envelope_fingerprint.as_slice(),
                process_epoch,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_completion",
            });
        }
        tx.commit()?;
        Ok(())
    }
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}
