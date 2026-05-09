// ============================================
// File: crates/aeronyx-server/src/management/reporter.rs
// ============================================
//! # Background Reporters
//!
//! Async tasks for periodic heartbeat and session event reporting.
//!
//! ## Modification Reason (v2.3.0+RemoteStorage)
//! - HeartbeatReporter now reports `memchain_status` in heartbeats
//! - Added `memchain_status_fn` callback for lazy MemChain status collection
//! - `send_heartbeat()` signature extended with `memchain_status` parameter
//! - CMS uses this data to update Node's remote storage fields
//!
//! ## Modification Reason (v1.0.0-TrafficAccounting)
//! - Added `SessionTrafficSnapshot` event for periodic mid-session reporting.
//!   Long-lived sessions previously had zero traffic visibility until disconnect.
//!   Now `session_traffic_snapshot()` sends cumulative bytes every 5 minutes.
//! - `SessionEvent::snapshot()` constructor added.
//! - `SessionEvent::to_report()` now sets `is_final` correctly:
//!   true for SessionEnded, false for all others.
//! - `SessionEventSender::session_traffic_snapshot()` public method added.
//!   Called by `server.rs::spawn_traffic_snapshot_task()` every 5 minutes.
//!
//! ## Previous Modifications
//! v1.3.0 - HeartbeatReporter parses commands from response, dynamic interval
//! v1.3.1 - Added agent_manager for status reporting
//!
//! Main Components:
//!   - HeartbeatReporter: Sends periodic heartbeats to CMS
//!   - SessionReporter: Reports session events (create/end/snapshot) to CMS
//!   - SessionEventSender: Thread-safe sender for session events
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatReporter runs on a fixed interval (default 30s)
//!   - SessionReporter processes events from a channel (capacity 1000)
//!   - Both respect shutdown signals for graceful termination
//!   - The command channel is Optional — if None, commands from CMS are
//!     logged but not dispatched (graceful degradation)
//!   - Dynamic interval adjustment is clamped to [10, 300] seconds
//!   - memchain_status_fn is Optional — if None, no memchain_status in heartbeat
//!   - SessionTrafficSnapshot bytes are CUMULATIVE totals, never deltas.
//!     Backend must upsert, not accumulate. is_final=false for snapshots.
//!   - session_traffic_snapshot() silently drops if the channel is full —
//!     non-critical, next tick will send an updated snapshot anyway.
//!
//! Last Modified:
//!   v1.0.0 - Initial implementation
//!   v1.2.0 - Fixed SessionEventType import and client_wallet type
//!   v1.3.0 - Added command forwarding from heartbeat response
//!   v2.3.0+RemoteStorage - Added memchain_status reporting in heartbeat
//!   v1.0.0-TrafficAccounting - Added SessionTrafficSnapshot event type,
//!     session_traffic_snapshot() sender method, is_final flag in to_report().

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use super::client::{ManagementClient, MemChainHeartbeatStatus};
use super::models::{Command, SessionEventReport, SessionEventType};
use crate::services::AgentManager;

// ============================================
// Constants
// ============================================

/// Minimum allowed heartbeat interval (seconds).
const MIN_HEARTBEAT_INTERVAL_SECS: u64 = 10;

/// Maximum allowed heartbeat interval (seconds).
const MAX_HEARTBEAT_INTERVAL_SECS: u64 = 300;

/// Number of initial heartbeats that get extra tolerance during cold-start.
const COLD_START_GRACE_BEATS: u32 = 5;

/// Extended timeout for cold-start heartbeats (seconds).
const COLD_START_TIMEOUT_SECS: u64 = 30;

// ============================================
// MemChainStatusFn type alias (v2.3.0)
// ============================================

/// Callback type for collecting MemChain status at heartbeat time.
pub type MemChainStatusFn = Box<dyn Fn() -> Option<MemChainHeartbeatStatus> + Send + Sync>;

// ============================================
// Session Events
// ============================================

/// Session event for internal use before converting to API report.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    pub event_type: SessionEventType,
    pub session_id: String,
    pub client_wallet: Option<String>,
    /// Cumulative bytes received from client since session start.
    pub bytes_in: u64,
    /// Cumulative bytes sent to client since session start.
    pub bytes_out: u64,
    pub timestamp: u64,
}

impl SessionEvent {
    /// Creates a new session_created event.
    pub fn created(session_id: String, client_wallet: Option<String>) -> Self {
        Self {
            event_type: SessionEventType::SessionCreated,
            session_id,
            client_wallet,
            bytes_in: 0,
            bytes_out: 0,
            timestamp: now_unix(),
        }
    }

    /// Creates a new session_ended event with final traffic statistics.
    ///
    /// `bytes_in` and `bytes_out` are cumulative totals since session start.
    /// `is_final = true` in the generated report — backend closes billing period.
    pub fn ended(
        session_id: String,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionEnded,
            session_id,
            client_wallet,
            bytes_in,
            bytes_out,
            timestamp: now_unix(),
        }
    }

    /// Creates a periodic traffic snapshot for a live session.
    ///
    /// `bytes_in` and `bytes_out` are cumulative totals since session start,
    /// NOT deltas. Backend must upsert these values.
    /// `is_final = false` in the generated report — billing period stays open.
    pub fn snapshot(
        session_id: String,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionTrafficSnapshot,
            session_id,
            client_wallet,
            bytes_in,
            bytes_out,
            timestamp: now_unix(),
        }
    }

    /// Converts internal event to API report format.
    ///
    /// Sets `is_final = true` only for `SessionEnded`.
    /// All other event types (including snapshots) use `is_final = false`.
    fn to_report(&self) -> SessionEventReport {
        SessionEventReport {
            event_type: self.event_type,
            session_id: self.session_id.clone(),
            client_wallet: self.client_wallet.clone(),
            client_ip: None,
            bytes_in: self.bytes_in,
            bytes_out: self.bytes_out,
            timestamp: self.timestamp,
            is_final: matches!(self.event_type, SessionEventType::SessionEnded),
        }
    }
}

/// Returns the current unix timestamp in seconds.
#[inline]
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// HeartbeatReporter
// ============================================

/// Background task for sending periodic heartbeats to CMS.
///
/// v1.3.0: Parses `commands` from `HeartbeatResponse` and forwards them.
/// v2.3.0: Reports `memchain_status` for remote storage field sync.
pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
    command_tx: Option<mpsc::Sender<Command>>,
    agent_manager: Option<Arc<AgentManager>>,
    memchain_status_fn: Option<MemChainStatusFn>,
}

impl HeartbeatReporter {
    pub fn new(client: Arc<ManagementClient>, public_ip: String) -> Self {
        let interval = Duration::from_secs(client.config().heartbeat_interval_secs);
        Self {
            client,
            interval,
            public_ip,
            command_tx: None,
            agent_manager: None,
            memchain_status_fn: None,
        }
    }

    pub fn with_command_sender(mut self, tx: mpsc::Sender<Command>) -> Self {
        self.command_tx = Some(tx);
        self
    }

    pub fn with_agent_manager(mut self, am: Arc<AgentManager>) -> Self {
        self.agent_manager = Some(am);
        self
    }

    pub fn with_memchain_status(mut self, f: MemChainStatusFn) -> Self {
        self.memchain_status_fn = Some(f);
        self
    }

    /// Runs the heartbeat loop until shutdown signal received.
    pub async fn run<F>(
        self,
        session_count_fn: F,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    )
    where
        F: Fn() -> u32 + Send + 'static,
    {
        info!(
            interval_secs = self.interval.as_secs(),
            has_command_channel = self.command_tx.is_some(),
            has_memchain_status = self.memchain_status_fn.is_some(),
            "[HEARTBEAT] Reporter started"
        );

        let mut interval = tokio::time::interval(self.interval);
        let mut failures = 0u32;
        let mut total_beats = 0u32;

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[HEARTBEAT] Stopping");
                    break;
                }
                _ = interval.tick() => {
                    total_beats += 1;
                    let in_cold_start = total_beats <= COLD_START_GRACE_BEATS;

                    let agent_status = if let Some(ref am) = self.agent_manager {
                        Some(am.status().await)
                    } else {
                        None
                    };

                    let memchain_status = self.memchain_status_fn.as_ref()
                        .and_then(|f| f());

                    let timeout_secs = if in_cold_start {
                        COLD_START_TIMEOUT_SECS
                    } else {
                        60
                    };

                    let result = tokio::time::timeout(
                        Duration::from_secs(timeout_secs),
                        self.client.send_heartbeat(
                            &self.public_ip,
                            session_count_fn(),
                            agent_status,
                            memchain_status,
                        ),
                    ).await;

                    match result {
                        Err(_elapsed) => {
                            failures += 1;
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    "[HEARTBEAT] Timeout during cold-start (CMS may still be loading)"
                                );
                            } else {
                                warn!(failures, "[HEARTBEAT] Heartbeat outer timeout");
                            }
                        }
                        Ok(Ok(response)) => {
                            if failures > 0 {
                                info!(
                                    previous_failures = failures,
                                    "[HEARTBEAT] Recovered after {} failure(s)", failures
                                );
                            }
                            failures = 0;

                            if let Some(commands) = response.commands {
                                if !commands.is_empty() {
                                    info!(
                                        count = commands.len(),
                                        "[HEARTBEAT] Received {} command(s) from CMS",
                                        commands.len()
                                    );
                                    self.forward_commands(commands).await;
                                }
                            }

                            if let Some(next_in) = response.next_heartbeat_in {
                                let clamped = next_in
                                    .max(MIN_HEARTBEAT_INTERVAL_SECS)
                                    .min(MAX_HEARTBEAT_INTERVAL_SECS);
                                if clamped != self.interval.as_secs() {
                                    debug!(
                                        current = self.interval.as_secs(),
                                        requested = next_in,
                                        applied = clamped,
                                        "[HEARTBEAT] CMS requested interval change"
                                    );
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            failures += 1;
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    failures,
                                    "[HEARTBEAT] Failed during cold-start (expected): {}", e
                                );
                            } else if failures >= 5 {
                                error!(failures, "[HEARTBEAT] Persistent failure: {}", e);
                            } else if failures >= 3 {
                                warn!(failures, "[HEARTBEAT] Failed: {}", e);
                            } else {
                                debug!(failures, "[HEARTBEAT] Failed (transient): {}", e);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn forward_commands(&self, commands: Vec<Command>) {
        let Some(ref tx) = self.command_tx else {
            warn!(
                count = commands.len(),
                "[HEARTBEAT] Received commands but no command channel configured — discarding"
            );
            return;
        };

        for cmd in commands {
            debug!(
                command_id = %cmd.id,
                action = %cmd.action,
                priority = cmd.priority,
                "[HEARTBEAT] Forwarding command to handler"
            );
            if let Err(e) = tx.try_send(cmd) {
                match e {
                    mpsc::error::TrySendError::Full(cmd) => {
                        warn!(
                            command_id = %cmd.id,
                            action = %cmd.action,
                            "[HEARTBEAT] Command channel full — dropping command"
                        );
                    }
                    mpsc::error::TrySendError::Closed(cmd) => {
                        error!(
                            command_id = %cmd.id,
                            action = %cmd.action,
                            "[HEARTBEAT] Command channel closed — handler may have crashed"
                        );
                    }
                }
            }
        }
    }
}

// ============================================
// SessionReporter
// ============================================

/// Background task for reporting session events to CMS.
///
/// Processes events from the channel sequentially. Each event is sent
/// to CMS via `ManagementClient::report_session_event()`.
///
/// Handles three event types:
/// - `SessionCreated` — initial registration with CMS
/// - `SessionEnded` — final billing report (`is_final=true`)
/// - `SessionTrafficSnapshot` — periodic live traffic upsert (`is_final=false`)
pub struct SessionReporter {
    client: Arc<ManagementClient>,
    event_rx: mpsc::Receiver<SessionEvent>,
}

impl SessionReporter {
    /// Creates a new SessionReporter and returns the event sender.
    pub fn new(client: Arc<ManagementClient>) -> (Self, mpsc::Sender<SessionEvent>) {
        let (tx, rx) = mpsc::channel(1000);
        (Self { client, event_rx: rx }, tx)
    }

    /// Runs the session reporter loop until shutdown signal received.
    pub async fn run(mut self, mut shutdown: tokio::sync::broadcast::Receiver<()>) {
        info!("[SESSION_REPORTER] Started");
        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[SESSION_REPORTER] Stopping");
                    break;
                }
                Some(event) = self.event_rx.recv() => {
                    let event_type = format!("{:?}", event.event_type);
                    if let Err(e) = self.client.report_session_event(event.to_report()).await {
                        warn!(
                            event_type = %event_type,
                            "[SESSION_REPORTER] Report failed: {}", e
                        );
                    } else {
                        debug!(
                            event_type = %event_type,
                            "[SESSION_REPORTER] Report sent"
                        );
                    }
                }
            }
        }
    }
}

// ============================================
// SessionEventSender
// ============================================

/// Thread-safe sender for session events.
///
/// Cloneable — passed to both the UDP task and the cleanup task.
/// All send operations are non-blocking (`try_send`). Dropped events
/// are logged at debug level; the channel has capacity 1000 so drops
/// are extremely rare under normal load.
#[derive(Clone)]
pub struct SessionEventSender {
    tx: Option<mpsc::Sender<SessionEvent>>,
}

impl SessionEventSender {
    pub fn new(tx: mpsc::Sender<SessionEvent>) -> Self {
        Self { tx: Some(tx) }
    }

    pub fn disabled() -> Self {
        Self { tx: None }
    }

    /// Reports a new session creation to CMS.
    pub fn session_created(&self, session_id: &str, client_wallet: Option<String>) {
        self.try_send(SessionEvent::created(session_id.to_string(), client_wallet));
    }

    /// Reports a session end to CMS with final cumulative traffic totals.
    ///
    /// `bytes_in` and `bytes_out` are cumulative since session start.
    /// Generates `is_final = true` — backend closes billing period.
    pub fn session_ended(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) {
        self.try_send(SessionEvent::ended(
            session_id.to_string(),
            client_wallet,
            bytes_in,
            bytes_out,
        ));
    }

    /// Reports a periodic traffic snapshot for a live session.
    ///
    /// Called every 5 minutes by `server.rs::spawn_traffic_snapshot_task()`.
    /// `bytes_in` and `bytes_out` are cumulative totals since session start.
    /// Generates `is_final = false` — backend upserts, does NOT close billing.
    ///
    /// Silently drops if the channel is full — non-critical, the next tick
    /// will send an updated snapshot with the latest cumulative values.
    pub fn session_traffic_snapshot(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) {
        self.try_send(SessionEvent::snapshot(
            session_id.to_string(),
            client_wallet,
            bytes_in,
            bytes_out,
        ));
    }

    /// Internal: non-blocking send with debug logging on drop.
    fn try_send(&self, event: SessionEvent) {
        if let Some(ref tx) = self.tx {
            if tx.try_send(event).is_err() {
                debug!("[SESSION_REPORTER] Event channel full or closed — event dropped");
            }
        }
    }
}
