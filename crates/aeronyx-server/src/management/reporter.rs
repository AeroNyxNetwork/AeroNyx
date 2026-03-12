//! ============================================
//! File: crates/aeronyx-server/src/management/reporter.rs
//! ============================================
//! # Background Reporters
//!
//! Async tasks for periodic heartbeat and session event reporting.
//!
//! ## Modification Reason (v2.3.0+RemoteStorage)
//! - 🌟 HeartbeatReporter now reports `memchain_status` in heartbeats
//! - Added `memchain_status_fn` callback for lazy MemChain status collection
//! - `send_heartbeat()` signature extended with `memchain_status` parameter
//! - CMS uses this data to update Node's remote storage fields
//!
//! ## Previous Modifications
//! v1.3.0 - HeartbeatReporter parses commands from response, dynamic interval
//! v1.3.1 - Added agent_manager for status reporting
//!
//! Main Components:
//!   - HeartbeatReporter: Sends periodic heartbeats to CMS
//!   - SessionReporter: Reports session events (create/end) to CMS
//!   - SessionEventSender: Thread-safe sender for session events
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatReporter runs on a fixed interval (default 30s)
//!   - SessionReporter processes events from a channel
//!   - Both respect shutdown signals for graceful termination
//!   - The command channel is Optional — if `None`, commands from
//!     CMS are logged but not dispatched (graceful degradation)
//!   - Dynamic interval adjustment is clamped to [10, 300] seconds
//!   - memchain_status_fn is Optional — if None, no memchain_status in heartbeat
//!
//! Last Modified:
//!   v1.0.0 - Initial implementation
//!   v1.2.0 - Fixed SessionEventType import and client_wallet type
//!   v1.3.0 - Added command forwarding from heartbeat response
//!   v2.3.0+RemoteStorage - 🌟 Added memchain_status reporting in heartbeat
//! ============================================

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
/// Prevents CMS from setting a dangerously aggressive poll rate.
const MIN_HEARTBEAT_INTERVAL_SECS: u64 = 10;

/// Maximum allowed heartbeat interval (seconds).
/// Prevents CMS from making the node appear offline for too long.
const MAX_HEARTBEAT_INTERVAL_SECS: u64 = 300;

/// Number of initial heartbeats that get extra tolerance.
/// During server cold-start, CMS (Gunicorn) may still be loading
/// Django, DB connections, etc. We suppress error-level logging
/// and use a longer timeout for these first few beats.
const COLD_START_GRACE_BEATS: u32 = 5;

/// Extended timeout for cold-start heartbeats (seconds).
/// Normal timeout is from config (default 10s), but during
/// the grace period we allow up to 30s for CMS to warm up.
const COLD_START_TIMEOUT_SECS: u64 = 30;

// ============================================
// MemChainStatusFn type alias (v2.3.0)
// ============================================

/// Callback type for collecting MemChain status at heartbeat time.
///
/// Returns `Option<MemChainHeartbeatStatus>` — None if MemChain is disabled.
/// The callback is invoked on each heartbeat tick, so it must be cheap.
///
/// Typically constructed as a closure capturing `Arc<MemoryStorage>` and
/// config values, called from `server.rs`.
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
    pub bytes_in: u64,
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
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
        }
    }

    /// Creates a new session_ended event with traffic statistics.
    pub fn ended(session_id: String, client_wallet: Option<String>, bytes_in: u64, bytes_out: u64) -> Self {
        Self {
            event_type: SessionEventType::SessionEnded,
            session_id,
            client_wallet,
            bytes_in,
            bytes_out,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
        }
    }

    /// Converts internal event to API report format.
    fn to_report(&self) -> SessionEventReport {
        SessionEventReport {
            event_type: self.event_type,
            session_id: self.session_id.clone(),
            client_wallet: self.client_wallet.clone(),
            client_ip: None,
            bytes_in: self.bytes_in,
            bytes_out: self.bytes_out,
            timestamp: self.timestamp,
        }
    }
}

// ============================================
// HeartbeatReporter
// ============================================

/// Background task for sending periodic heartbeats to CMS.
///
/// 🌟 v1.3.0: Parses `commands` from `HeartbeatResponse` and forwards them.
/// 🌟 v2.3.0: Reports `memchain_status` for remote storage field sync.
pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
    /// Optional channel to forward commands to CommandHandler.
    command_tx: Option<mpsc::Sender<Command>>,
    /// v1.3.1: Optional reference to AgentManager for status reporting.
    agent_manager: Option<Arc<AgentManager>>,
    /// v2.3.0: Optional callback to collect MemChain status for heartbeat.
    /// Called on each heartbeat tick. Returns None if MemChain is disabled.
    memchain_status_fn: Option<MemChainStatusFn>,
}

impl HeartbeatReporter {
    /// Creates a new HeartbeatReporter.
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

    /// Attaches a command sender channel for forwarding CMS commands.
    pub fn with_command_sender(mut self, tx: mpsc::Sender<Command>) -> Self {
        self.command_tx = Some(tx);
        self
    }

    /// v1.3.1: Attaches an AgentManager for status reporting in heartbeat.
    pub fn with_agent_manager(mut self, am: Arc<AgentManager>) -> Self {
        self.agent_manager = Some(am);
        self
    }

    /// v2.3.0: Attaches a MemChain status callback for heartbeat reporting.
    ///
    /// The callback is invoked on each heartbeat tick to collect current
    /// MemChain state (allow_remote_storage, owner counts, etc.).
    /// CMS uses this data to update Node's remote storage fields.
    pub fn with_memchain_status(mut self, f: MemChainStatusFn) -> Self {
        self.memchain_status_fn = Some(f);
        self
    }

    /// Runs the heartbeat loop until shutdown signal received.
    pub async fn run<F>(self, session_count_fn: F, mut shutdown: tokio::sync::broadcast::Receiver<()>)
    where F: Fn() -> u32 + Send + 'static
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

                    // v1.3.1: Fetch agent status if AgentManager is available
                    let agent_status = if let Some(ref am) = self.agent_manager {
                        Some(am.status().await)
                    } else {
                        None
                    };

                    // v2.3.0: Collect MemChain status if callback is configured
                    let memchain_status = self.memchain_status_fn.as_ref()
                        .and_then(|f| f());

                    // Use extended timeout during cold-start grace period
                    let result = if in_cold_start {
                        debug!(
                            beat = total_beats,
                            grace_remaining = COLD_START_GRACE_BEATS - total_beats,
                            "[HEARTBEAT] Cold-start grace period (extended timeout {}s)",
                            COLD_START_TIMEOUT_SECS
                        );
                        tokio::time::timeout(
                            std::time::Duration::from_secs(COLD_START_TIMEOUT_SECS),
                            self.client.send_heartbeat(
                                &self.public_ip,
                                session_count_fn(),
                                agent_status,
                                memchain_status,
                            ),
                        ).await
                    } else {
                        tokio::time::timeout(
                            std::time::Duration::from_secs(60),
                            self.client.send_heartbeat(
                                &self.public_ip,
                                session_count_fn(),
                                agent_status,
                                memchain_status,
                            ),
                        ).await
                    };

                    match result {
                        // Outer timeout expired
                        Err(_elapsed) => {
                            failures += 1;
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    "[HEARTBEAT] ⏳ Timeout during cold-start (CMS may still be loading)"
                                );
                            } else {
                                warn!(
                                    failures = failures,
                                    "[HEARTBEAT] ⚠️ Heartbeat outer timeout"
                                );
                            }
                        }
                        // Inner result
                        Ok(Ok(response)) => {
                            if failures > 0 {
                                info!(
                                    previous_failures = failures,
                                    "[HEARTBEAT] ✅ Recovered after {} failure(s)",
                                    failures
                                );
                            }
                            failures = 0;

                            // Process commands from CMS
                            if let Some(commands) = response.commands {
                                if !commands.is_empty() {
                                    info!(
                                        count = commands.len(),
                                        "[HEARTBEAT] 📨 Received {} command(s) from CMS",
                                        commands.len()
                                    );

                                    self.forward_commands(commands).await;
                                }
                            }

                            // Dynamic interval adjustment
                            if let Some(next_in) = response.next_heartbeat_in {
                                let clamped = next_in
                                    .max(MIN_HEARTBEAT_INTERVAL_SECS)
                                    .min(MAX_HEARTBEAT_INTERVAL_SECS);

                                if clamped != self.interval.as_secs() {
                                    debug!(
                                        current = self.interval.as_secs(),
                                        requested = next_in,
                                        applied = clamped,
                                        "[HEARTBEAT] 🔄 CMS requested interval change"
                                    );
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            failures += 1;

                            // During cold-start: only info level, not warn/error
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    failures = failures,
                                    "[HEARTBEAT] ⏳ Failed during cold-start (expected): {}",
                                    e
                                );
                            } else if failures >= 5 {
                                error!(
                                    failures = failures,
                                    "[HEARTBEAT] ❌ Persistent failure: {}",
                                    e
                                );
                            } else if failures >= 3 {
                                warn!(
                                    failures = failures,
                                    "[HEARTBEAT] ⚠️ Failed: {}",
                                    e
                                );
                            } else {
                                debug!(
                                    failures = failures,
                                    "[HEARTBEAT] Failed (transient): {}",
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Forwards received commands to the CommandHandler channel.
    async fn forward_commands(&self, commands: Vec<Command>) {
        let Some(ref tx) = self.command_tx else {
            warn!(
                count = commands.len(),
                "[HEARTBEAT] ⚠️ Received commands but no command channel configured — discarding"
            );
            return;
        };

        for cmd in commands {
            debug!(
                command_id = %cmd.id,
                action = %cmd.action,
                priority = cmd.priority,
                "[HEARTBEAT] 📤 Forwarding command to handler"
            );

            if let Err(e) = tx.try_send(cmd) {
                match e {
                    mpsc::error::TrySendError::Full(cmd) => {
                        warn!(
                            command_id = %cmd.id,
                            action = %cmd.action,
                            "[HEARTBEAT] ⚠️ Command channel full — dropping command"
                        );
                    }
                    mpsc::error::TrySendError::Closed(cmd) => {
                        error!(
                            command_id = %cmd.id,
                            action = %cmd.action,
                            "[HEARTBEAT] ❌ Command channel closed — handler may have crashed"
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
                    if let Err(e) = self.client.report_session_event(event.to_report()).await {
                        warn!("[SESSION_REPORTER] Report failed: {}", e);
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
#[derive(Clone)]
pub struct SessionEventSender {
    tx: Option<mpsc::Sender<SessionEvent>>,
}

impl SessionEventSender {
    pub fn new(tx: mpsc::Sender<SessionEvent>) -> Self { Self { tx: Some(tx) } }
    pub fn disabled() -> Self { Self { tx: None } }

    pub fn session_created(&self, session_id: &str, client_wallet: Option<String>) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(SessionEvent::created(session_id.to_string(), client_wallet));
        }
    }

    pub fn session_ended(&self, session_id: &str, client_wallet: Option<String>, bytes_in: u64, bytes_out: u64) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(SessionEvent::ended(session_id.to_string(), client_wallet, bytes_in, bytes_out));
        }
    }
}
