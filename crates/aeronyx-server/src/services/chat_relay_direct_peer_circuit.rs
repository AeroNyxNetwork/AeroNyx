// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_direct_peer_circuit.rs
// ============================================
// Version: 1.1.0-StatusContractDependency
//
// Creation Reason:
//   [CHAT-DIRECT-PEER-CIRCUIT-DOMAIN 2026-08-25 by Codex] Extract the
//   source-blind direct-peer admission state machine and its durable SQLite
//   checkpoint from the oversized chat relay service.
//
// Modification Reason:
//   [CHAT-RELAY-STATUS-CONTRACT-DOMAIN 2026-08-27 by Codex] Depend directly
//   on the focused status contract domain instead of the orchestration facade.
//
// Main Functionality:
//   - Models closed, open, and leased half-open states as a closed enum.
//   - Issues generation-bound permits that reject stale completions.
//   - Defines a replaceable checkpoint repository trait.
//   - Persists every safety transition before exposing it to callers.
//   - Restores restart safety and fails closed on corrupt or missing state.
//
// Dependencies:
//   - `chat_relay_status.rs` owns the serialized status and policy defaults.
//   - `chat_relay.rs` owns the public compatibility facade and service.
//   - `rusqlite` supplies the production checkpoint repository.
//   - `parking_lot` supplies process-local state and connection guards.
//
// Main Logical Flow:
//   1. Initialise and validate the anonymous singleton checkpoint schema.
//   2. Restore and validate the checkpoint before relay activation.
//   3. Admit one closed-state attempt or one leased half-open probe.
//   4. Persist safety-changing transitions while holding the circuit lock.
//   5. Reject delivery when persistence cannot prove restart-safe state.
//
// Important Note for Next Developer:
//   - Preserve lock order: circuit state first, SQLite connection second.
//   - Never add peer, route, endpoint, wallet, message, or payload dimensions.
//   - Never treat a missing installed checkpoint as a fresh closed circuit.
//   - A runtime checkpoint failure must continue to fail closed.
//
// Last Modified:
//   v1.1.0-StatusContractDependency - Consumed shared status contracts directly
//   v1.0.0-DirectPeerCircuitDomain - Initial state/repository composition
// ============================================

use parking_lot::Mutex;
#[cfg(test)]
use parking_lot::MutexGuard;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use tracing::warn;

use super::chat_relay::{ChatRelayError, ChatRelayResult};
use super::chat_relay_status::{
    ChatRelayDirectPeerCircuitStatus, DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS,
    DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS, DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES,
};
/// Durable singleton format for source-blind direct relay circuit state.
pub(super) const DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION: i64 = 1;
/// Fixed schema marker proving the durable circuit checkpoint was installed.
pub(super) const DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE: &str =
    "direct_peer_relay_circuit_checkpoint";
/// Small tolerated wall-clock adjustment before restart recovery fails closed.
const DIRECT_PEER_RELAY_CIRCUIT_CLOCK_SKEW_SECS: u64 = 5;

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
/// The generation prevents a late outcome from an older request from closing
/// or reopening a newer circuit. The token contains no routing identity.
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
pub(super) struct DirectPeerRelayCircuit {
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

    pub(super) fn snapshot(&self, now: u64) -> ChatRelayDirectPeerCircuitStatus {
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
        if self.schema_version != DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION {
            return Err(corrupt("direct_peer_circuit_checkpoint_version"));
        }
        let successful_probes = u8::try_from(self.successful_probes)
            .map_err(|_| corrupt("direct_peer_circuit_checkpoint_probe_count"))?;
        if successful_probes >= DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES {
            return Err(corrupt("direct_peer_circuit_checkpoint_probe_count"));
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
            return Err(corrupt("direct_peer_circuit_checkpoint_time_order"));
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
            return Err(corrupt("direct_peer_circuit_checkpoint_counter_relation"));
        }

        let state = match self.state.as_str() {
            "closed" if successful_probes == 0 && deadline_at.is_none() => {
                DirectPeerRelayCircuitState::Closed
            }
            "open" if successful_probes == 0 && opened_total > 0 => {
                DirectPeerRelayCircuitState::Open {
                    retry_at: deadline_at
                        .ok_or_else(|| corrupt("direct_peer_circuit_checkpoint_open_deadline"))?,
                }
            }
            "half_open_ready" if deadline_at.is_none() && opened_total > 0 => {
                DirectPeerRelayCircuitState::HalfOpenReady { successful_probes }
            }
            "half_open_in_flight" if opened_total > 0 => {
                DirectPeerRelayCircuitState::HalfOpenInFlight {
                    successful_probes,
                    lease_expires_at: deadline_at
                        .ok_or_else(|| corrupt("direct_peer_circuit_checkpoint_probe_deadline"))?,
                }
            }
            _ => return Err(corrupt("direct_peer_circuit_checkpoint_state")),
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

/// Replaceable persistence boundary for anonymous direct-peer safety state.
pub(crate) trait DirectPeerCircuitRepository: Send + Sync {
    fn init_schema(&self, conn: &mut Connection, now: u64) -> ChatRelayResult<()>;
    fn read(&self, conn: &Connection, now: u64) -> ChatRelayResult<(DirectPeerRelayCircuit, bool)>;
    fn write(
        &self,
        conn: &Connection,
        circuit: &DirectPeerRelayCircuit,
        now: u64,
    ) -> ChatRelayResult<()>;
}

/// SQLite implementation of the anonymous direct-peer checkpoint contract.
#[derive(Debug, Default)]
pub(crate) struct SqliteDirectPeerCircuitRepository;

impl DirectPeerCircuitRepository for SqliteDirectPeerCircuitRepository {
    fn init_schema(&self, conn: &mut Connection, now: u64) -> ChatRelayResult<()> {
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
            return Err(corrupt(
                "direct_peer_circuit_checkpoint_installation_version",
            ));
        }
        if !table_existed && installed_version.is_some() {
            return Err(corrupt("direct_peer_circuit_checkpoint_table"));
        }
        if table_existed {
            let row_count = tx.query_row(
                "SELECT COUNT(*) FROM relay_direct_peer_circuit_checkpoint",
                [],
                |row| row.get::<_, i64>(0),
            )?;
            if row_count != 1 {
                return Err(corrupt("direct_peer_circuit_checkpoint_singleton"));
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
            return Err(corrupt("direct_peer_circuit_checkpoint_singleton"));
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
            return Err(corrupt(
                "direct_peer_circuit_checkpoint_installation_marker",
            ));
        }
        tx.commit()?;
        Ok(())
    }

    fn read(&self, conn: &Connection, now: u64) -> ChatRelayResult<(DirectPeerRelayCircuit, bool)> {
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

    fn write(
        &self,
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
            return Err(corrupt("direct_peer_circuit_checkpoint_singleton"));
        }
        Ok(())
    }
}

/// Composed process-local state and durable restart-safety boundary.
#[derive(Debug, Default)]
pub(crate) struct DirectPeerCircuitDomain<R = SqliteDirectPeerCircuitRepository> {
    state: Mutex<DirectPeerRelayCircuit>,
    repository: R,
}

impl<R: DirectPeerCircuitRepository> DirectPeerCircuitDomain<R> {
    pub(crate) fn init_schema(&self, conn: &mut Connection, now: u64) -> ChatRelayResult<()> {
        self.repository.init_schema(conn, now)
    }

    pub(crate) fn restore(&self, conn: &Mutex<Connection>, now: u64) -> ChatRelayResult<()> {
        let mut circuit = {
            let conn = conn.lock();
            let (mut circuit, needs_rewrite) = self.repository.read(&conn, now)?;
            if needs_rewrite {
                circuit.mark_checkpoint_persisted(now);
                self.repository.write(&conn, &circuit, now)?;
            }
            circuit
        };
        let persisted_at = circuit.checkpoint_persisted_at;
        circuit.mark_checkpoint_loaded(now, persisted_at);
        *self.state.lock() = circuit;
        Ok(())
    }

    pub(crate) fn validate_checkpoint(conn: &Connection, now: u64) -> ChatRelayResult<()>
    where
        R: Default,
    {
        let _ = R::default().read(conn, now)?;
        Ok(())
    }

    pub(crate) fn begin(
        &self,
        conn: &Mutex<Connection>,
        now: u64,
    ) -> Option<ChatRelayDirectPeerPermit> {
        let mut circuit = self.state.lock();
        let previous = circuit.clone();
        let mut next = previous.clone();
        let permit = next.begin(now);
        if next.safety_state_changed(&previous) {
            if !self.persist_transition(conn, &mut circuit, next, now) {
                return None;
            }
        } else {
            *circuit = next;
        }
        permit
    }

    pub(crate) fn cancel(
        &self,
        conn: &Mutex<Connection>,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
    ) {
        let mut circuit = self.state.lock();
        let previous = circuit.clone();
        let mut next = previous.clone();
        next.cancel(now, permit);
        if next.safety_state_changed(&previous) {
            let _ = self.persist_transition(conn, &mut circuit, next, now);
        } else {
            *circuit = next;
        }
    }

    pub(crate) fn complete<F>(
        &self,
        conn: &Mutex<Connection>,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
        delivery_succeeded: bool,
        observe_slo_failed: F,
    ) -> bool
    where
        F: FnOnce() -> bool,
    {
        let mut circuit = self.state.lock();
        if !circuit.accepts_completion(permit) {
            return false;
        }
        let slo_failed = observe_slo_failed();
        let previous = circuit.clone();
        let mut next = previous.clone();
        let mut allows_more = next.complete(now, permit, delivery_succeeded, slo_failed);
        if next.safety_state_changed(&previous) {
            allows_more = self.persist_transition(conn, &mut circuit, next, now) && allows_more;
        } else {
            *circuit = next;
        }
        allows_more
    }

    #[must_use]
    pub(crate) fn snapshot(&self, now: u64) -> ChatRelayDirectPeerCircuitStatus {
        self.state.lock().snapshot(now)
    }

    #[cfg(test)]
    pub(super) fn lock(&self) -> MutexGuard<'_, DirectPeerRelayCircuit> {
        self.state.lock()
    }

    fn persist_transition(
        &self,
        conn: &Mutex<Connection>,
        circuit: &mut DirectPeerRelayCircuit,
        mut next: DirectPeerRelayCircuit,
        now: u64,
    ) -> bool {
        next.mark_checkpoint_persisted(now);
        let result = {
            let conn = conn.lock();
            self.repository.write(&conn, &next, now)
        };
        match result {
            Ok(()) => {
                *circuit = next;
                true
            }
            Err(error) => {
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Direct relay circuit checkpoint failed closed"
                );
                circuit.fail_closed_after_checkpoint_error(now);
                false
            }
        }
    }
}

fn corrupt(field: &'static str) -> ChatRelayError {
    ChatRelayError::CorruptStoredData { field }
}

fn nonnegative_sqlite_value(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| corrupt(field))
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
    i64::try_from(value).map_err(|_| corrupt(field))
}

fn optional_sqlite_integer(
    value: Option<u64>,
    field: &'static str,
) -> ChatRelayResult<Option<i64>> {
    value.map(|value| sqlite_integer(value, field)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_generation_cannot_complete_new_recovery_probe() {
        let mut circuit = DirectPeerRelayCircuit::default();
        let closed = circuit.begin(100).expect("closed permit");
        assert!(!circuit.complete(100, closed, false, true));
        let probe = circuit
            .begin(100 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
            .expect("half-open permit");
        circuit.cancel(131, probe);
        assert!(!circuit.accepts_completion(probe));
    }

    #[test]
    fn interrupted_probe_recovers_as_open_checkpoint() {
        let row = DirectPeerRelayCircuitCheckpointRow {
            schema_version: DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION,
            state: "half_open_in_flight".to_string(),
            successful_probes: 0,
            deadline_at: Some(120),
            opened_total: 1,
            blocked_total: 0,
            half_open_attempted_total: 1,
            half_open_succeeded_total: 0,
            half_open_failed_total: 0,
            recovered_total: 0,
            last_transition_at: Some(100),
            checkpoint_failures_total: 0,
            last_checkpoint_failure_at: None,
            updated_at: 100,
        };
        let (circuit, rewrite) = row.into_circuit(101).expect("valid checkpoint");
        assert!(rewrite);
        assert_eq!(circuit.half_open_failed_total, 1);
        assert_eq!(circuit.snapshot(101).state, "open");
    }
}
