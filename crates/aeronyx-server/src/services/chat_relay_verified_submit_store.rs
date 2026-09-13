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
//   [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Read retained completed
//   results without pruning, renewing, or touching unfinished reservations.
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
    /// Read-only stale retry: only unexpired completed evidence is eligible.
    fn lookup_completed_readonly(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        now: u64,
    ) -> ChatRelayResult<Option<DurableVerifiedSubmitResponse>>;

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
    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] One SELECT, no cache
    // fill, cleanup, reservation takeover, or sliding TTL. Bounds mirror the
    // existing v1 response schema and are checked before Rust BLOB allocation.
    fn lookup_completed_readonly(
        &self,
        connection: &Mutex<Connection>,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        now: u64,
    ) -> ChatRelayResult<Option<DurableVerifiedSubmitResponse>> {
        sqlite_integer(now, "verified_submit_readonly_time")?;
        let connection = connection.lock();
        let mut statement = connection.prepare(
            "SELECT completed_at, length(envelope_fingerprint), length(response_nonce),
                    length(response_ciphertext), envelope_fingerprint,
                    response_nonce, response_ciphertext
             FROM relay_verified_submit_responses WHERE cache_key = ?1",
        )?;
        let mut rows = statement.query(params![cache_key.as_slice()])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let completed_at = u64::try_from(row.get::<_, i64>(0)?).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "verified_submit_completed_time",
            }
        })?;
        let age = now
            .checked_sub(completed_at)
            .ok_or(ChatRelayError::CorruptStoredData {
                field: "verified_submit_completed_time",
            })?;
        if age > self.response_ttl_secs {
            return Ok(None);
        }
        if row.get::<_, i64>(1)? != 32
            || row.get::<_, i64>(2)? != 24
            || !(17..=528).contains(&row.get::<_, i64>(3)?)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_readonly_shape",
            });
        }
        if row.get::<_, Vec<u8>>(4)?.as_slice() != envelope_fingerprint.as_slice() {
            return Ok(None);
        }
        Ok(Some(DurableVerifiedSubmitResponse {
            nonce: row.get(5)?,
            ciphertext: row.get(6)?,
        }))
    }

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

#[cfg(test)]
mod stale_replay_tests {
    use super::*;

    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Real file-backed
    // SQLite, synthetic opaque rows. No network, global time, or real identity.
    #[test]
    fn verified_submit_stale_completed_replay_retention_reopen_and_zero_writes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("completed.sqlite3");
        let connection = Mutex::new(Connection::open(&path).unwrap());
        connection
            .lock()
            .execute_batch(
                "CREATE TABLE relay_verified_submit_responses (
               cache_key BLOB PRIMARY KEY, envelope_fingerprint BLOB,
               response_nonce BLOB, response_ciphertext BLOB, completed_at INTEGER);
             CREATE TABLE relay_verified_submit_reservations (cache_key BLOB PRIMARY KEY);
             INSERT INTO relay_verified_submit_reservations VALUES (zeroblob(32));",
            )
            .unwrap();
        connection
            .lock()
            .execute(
                "INSERT INTO relay_verified_submit_responses VALUES (?1, ?2, ?3, ?4, 1000)",
                params![
                    [1_u8; 32].as_slice(),
                    [2_u8; 32].as_slice(),
                    [3_u8; 24].as_slice(),
                    vec![4_u8; 32]
                ],
            )
            .unwrap();
        let store = SqliteVerifiedSubmitDurableStore::new(121, 8, 5);
        let before: i64 = connection
            .lock()
            .query_row("SELECT total_changes()", [], |r| r.get(0))
            .unwrap();
        for now in [1000, 1060, 1061, 1121] {
            assert!(store
                .lookup_completed_readonly(&connection, &[1; 32], &[2; 32], now)
                .unwrap()
                .is_some());
        }
        assert!(store
            .lookup_completed_readonly(&connection, &[1; 32], &[2; 32], 1122)
            .unwrap()
            .is_none());
        assert!(store
            .lookup_completed_readonly(&connection, &[1; 32], &[9; 32], 1061)
            .unwrap()
            .is_none());
        for key in [[0; 32], [9; 32]] {
            // Pending and miss never touch reservations.
            assert!(store
                .lookup_completed_readonly(&connection, &key, &[2; 32], 1061)
                .unwrap()
                .is_none());
        }
        assert!(store
            .lookup_completed_readonly(&connection, &[1; 32], &[2; 32], 999)
            .is_err());
        assert!(store
            .lookup_completed_readonly(&connection, &[1; 32], &[2; 32], u64::MAX)
            .is_err());
        let after: i64 = connection
            .lock()
            .query_row("SELECT total_changes()", [], |r| r.get(0))
            .unwrap();
        assert_eq!(before, after);
        assert_eq!(
            connection
                .lock()
                .query_row(
                    "SELECT completed_at FROM relay_verified_submit_responses",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1000
        );
        drop(connection);
        let reopened = Mutex::new(Connection::open(path).unwrap());
        assert!(store
            .lookup_completed_readonly(&reopened, &[1; 32], &[2; 32], 1121)
            .unwrap()
            .is_some());
        assert!(store
            .lookup_completed_readonly(&reopened, &[1; 32], &[2; 32], 1122)
            .unwrap()
            .is_none());
        assert_eq!(
            reopened
                .lock()
                .query_row("SELECT total_changes()", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            reopened
                .lock()
                .query_row(
                    "SELECT count(*) FROM relay_verified_submit_reservations",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn verified_submit_stale_completed_replay_rejects_corruption_before_materialization() {
        let connection = Mutex::new(Connection::open_in_memory().unwrap());
        connection
            .lock()
            .execute_batch(
                "CREATE TABLE relay_verified_submit_responses (
               cache_key BLOB PRIMARY KEY, envelope_fingerprint BLOB,
               response_nonce BLOB, response_ciphertext BLOB, completed_at INTEGER);
             INSERT INTO relay_verified_submit_responses VALUES
               (zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(32), 1000);",
            )
            .unwrap();
        let store = SqliteVerifiedSubmitDurableStore::new(121, 8, 5);
        for mutation in [
            "completed_at = -1",
            "completed_at = 2000",
            "envelope_fingerprint = zeroblob(31)",
            "response_nonce = zeroblob(25)",
            "response_ciphertext = zeroblob(16)",
            "response_ciphertext = zeroblob(529)",
            "response_ciphertext = zeroblob(1048576)",
        ] {
            connection
                .lock()
                .execute_batch(
                    "UPDATE relay_verified_submit_responses SET completed_at=1000,
                   envelope_fingerprint=zeroblob(32), response_nonce=zeroblob(24),
                   response_ciphertext=zeroblob(32)",
                )
                .unwrap();
            connection
                .lock()
                .execute(
                    &format!("UPDATE relay_verified_submit_responses SET {mutation}"),
                    [],
                )
                .unwrap();
            let before: i64 = connection
                .lock()
                .query_row("SELECT total_changes()", [], |r| r.get(0))
                .unwrap();
            assert!(store
                .lookup_completed_readonly(&connection, &[0; 32], &[0; 32], 1061)
                .is_err());
            assert_eq!(
                connection
                    .lock()
                    .query_row("SELECT total_changes()", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                before
            );
        }
    }
}
