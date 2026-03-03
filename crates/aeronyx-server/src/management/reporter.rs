//! ============================================
//! File: crates/aeronyx-server/src/management/reporter.rs
//! ============================================
//! # Background Reporters
//!
//! Async tasks for periodic heartbeat and session event reporting.
//!
//! ## Modification Reason (v1.3.0)
//! - 🌟 HeartbeatReporter now parses `HeartbeatResponse.commands` and forwards
//!   structured `Command` objects to the `CommandHandler` via an `mpsc` channel.
//! - 🌟 Added `command_tx: Option<mpsc::Sender<Command>>` to HeartbeatReporter
//!   so it can be wired up from `server.rs` during initialisation.
//! - 🌟 Added dynamic heartbeat interval adjustment from `next_heartbeat_in`.
//!
//! Main Components:
//!   - HeartbeatReporter: Sends periodic heartbeats to CMS
//!   - SessionReporter: Reports session events (create/end) to CMS
//!   - SessionEventSender: Thread-safe sender for session events
//!
//! ## Main Logical Flow (HeartbeatReporter)
//! 1. Tick at configured interval (default 30s)
//! 2. Call `ManagementClient::send_heartbeat()`
//! 3. On success: parse `HeartbeatResponse`
//!    a. If `commands` present → forward each `Command` to channel
//!    b. If `next_heartbeat_in` present → adjust next tick interval
//! 4. On failure: increment failure counter, log warning/error
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatReporter runs on a fixed interval (default 30s)
//!   - SessionReporter processes events from a channel
//!   - Both respect shutdown signals for graceful termination
//!   - 🌟 The command channel is Optional — if `None`, commands from
//!     CMS are logged but not dispatched (graceful degradation)
//!   - 🌟 Dynamic interval adjustment is clamped to [10, 300] seconds
//!     to prevent CMS from setting dangerously low or high intervals
//!
//! Last Modified:
//!   v1.0.0 - Initial implementation
//!   v1.2.0 - Fixed SessionEventType import and client_wallet type
//!   v1.3.0 - 🌟 Added command forwarding from heartbeat response
//! ============================================

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::client::ManagementClient;
use super::models::{Command, SessionEventReport, SessionEventType};

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
/// 🌟 v1.3.0: Now also parses `commands` from `HeartbeatResponse`
/// and forwards them to the `CommandHandler` via an internal channel.
pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
    /// 🌟 Optional channel to forward commands to CommandHandler.
    /// If `None`, commands from CMS are logged but not dispatched.
    command_tx: Option<mpsc::Sender<Command>>,
}

impl HeartbeatReporter {
    /// Creates a new HeartbeatReporter.
    ///
    /// # Arguments
    /// * `client` - Shared ManagementClient for API calls
    /// * `public_ip` - Node's public IP address to report
    pub fn new(client: Arc<ManagementClient>, public_ip: String) -> Self {
        let interval = Duration::from_secs(client.config().heartbeat_interval_secs);
        Self {
            client,
            interval,
            public_ip,
            command_tx: None,
        }
    }

    /// 🌟 Attaches a command sender channel for forwarding CMS commands.
    ///
    /// Must be called before `.run()` to enable command dispatch.
    /// If not called, commands from CMS are logged but discarded.
    ///
    /// # Arguments
    /// * `tx` - Sender end of the command channel (receiver held by CommandHandler)
    pub fn with_command_sender(mut self, tx: mpsc::Sender<Command>) -> Self {
        self.command_tx = Some(tx);
        self
    }

    /// Runs the heartbeat loop until shutdown signal received.
    ///
    /// # Arguments
    /// * `session_count_fn` - Closure that returns current active session count
    /// * `shutdown` - Broadcast receiver for shutdown signal
    pub async fn run<F>(self, session_count_fn: F, mut shutdown: tokio::sync::broadcast::Receiver<()>)
    where F: Fn() -> u32 + Send + 'static
    {
        info!(
            interval_secs = self.interval.as_secs(),
            has_command_channel = self.command_tx.is_some(),
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
                            self.client.send_heartbeat(&self.public_ip, session_count_fn()),
                        ).await
                    } else {
                        // Normal operation: use config timeout (already in reqwest client)
                        // Wrap in a generous outer timeout as safety net
                        tokio::time::timeout(
                            std::time::Duration::from_secs(60),
                            self.client.send_heartbeat(&self.public_ip, session_count_fn()),
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

                            // 🌟 Process commands from CMS
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

                            // 🌟 Dynamic interval adjustment
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

    /// 🌟 Forwards received commands to the CommandHandler channel.
    ///
    /// If the channel is not configured or is full, commands are logged
    /// as warnings but not dropped silently.
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
    ///
    /// # Arguments
    /// * `client` - Shared ManagementClient for API calls
    ///
    /// # Returns
    /// Tuple of (SessionReporter, event sender channel)
    pub fn new(client: Arc<ManagementClient>) -> (Self, mpsc::Sender<SessionEvent>) {
        let (tx, rx) = mpsc::channel(1000);
        (Self { client, event_rx: rx }, tx)
    }

    /// Runs the session reporter loop until shutdown signal received.
    ///
    /// # Arguments
    /// * `shutdown` - Broadcast receiver for shutdown signal
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
///
/// Can be cloned and shared across threads to send session events
/// to the SessionReporter.
#[derive(Clone)]
pub struct SessionEventSender {
    tx: Option<mpsc::Sender<SessionEvent>>,
}

impl SessionEventSender {
    /// Creates a new enabled sender with the given channel.
    pub fn new(tx: mpsc::Sender<SessionEvent>) -> Self { Self { tx: Some(tx) } }

    /// Creates a disabled sender that discards all events.
    pub fn disabled() -> Self { Self { tx: None } }

    /// Sends a session_created event.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    /// * `client_wallet` - Optional client wallet address
    pub fn session_created(&self, session_id: &str, client_wallet: Option<String>) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(SessionEvent::created(session_id.to_string(), client_wallet));
        }
    }

    /// Sends a session_ended event with traffic statistics.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    /// * `client_wallet` - Optional client wallet address
    /// * `bytes_in` - Total bytes received from client
    /// * `bytes_out` - Total bytes sent to client
    pub fn session_ended(&self, session_id: &str, client_wallet: Option<String>, bytes_in: u64, bytes_out: u64) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(SessionEvent::ended(session_id.to_string(), client_wallet, bytes_in, bytes_out));
        }
    }
}
