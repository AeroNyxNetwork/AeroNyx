// ============================================
// File: crates/aeronyx-server/src/management/reporter.rs
// ============================================
//! # Background Reporters
//!
//! Async tasks for periodic heartbeat and session event reporting.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::client::ManagementClient;
use super::models::{SessionEventReport, SessionEventType};

// ============================================
// Session Event
// ============================================

/// Session event for reporting queue.
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
    /// Creates a session created event.
    pub fn created(session_id: String, client_wallet: Option<String>) -> Self {
        Self {
            event_type: SessionEventType::SessionCreated,
            session_id,
            client_wallet,
            bytes_in: 0,
            bytes_out: 0,
            timestamp: current_timestamp(),
        }
    }

    /// Creates a session updated event.
    pub fn updated(
        session_id: String,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionUpdated,
            session_id,
            client_wallet,
            bytes_in,
            bytes_out,
            timestamp: current_timestamp(),
        }
    }

    /// Creates a session ended event.
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
            timestamp: current_timestamp(),
        }
    }

    /// Converts to API report format.
    fn to_report(&self) -> SessionEventReport {
        SessionEventReport {
            event_type: self.event_type,
            session_id: self.session_id.clone(),
            client_wallet: self.client_wallet.clone(),
            client_ip: None, // Privacy: don't report client IPs
            bytes_in: self.bytes_in,
            bytes_out: self.bytes_out,
            timestamp: self.timestamp,
        }
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// Heartbeat Reporter
// ============================================

/// Background task for periodic heartbeat reporting.
pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
}

impl HeartbeatReporter {
    /// Creates a new heartbeat reporter.
    pub fn new(client: Arc<ManagementClient>, public_ip: String) -> Self {
        let interval = Duration::from_secs(client.config().heartbeat_interval_secs);
        Self {
            client,
            interval,
            public_ip,
        }
    }

    /// Runs the heartbeat reporter loop.
    ///
    /// # Arguments
    /// * `session_count_fn` - Function to get current session count
    /// * `shutdown` - Shutdown signal receiver
    pub async fn run<F>(
        self,
        session_count_fn: F,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    ) where
        F: Fn() -> u32 + Send + 'static,
    {
        info!(
            "Heartbeat reporter started (interval: {}s)",
            self.interval.as_secs()
        );

        let mut interval = tokio::time::interval(self.interval);
        let mut consecutive_failures = 0u32;
        let max_failures = self.client.config().max_retries;

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("Heartbeat reporter shutting down");
                    break;
                }
                _ = interval.tick() => {
                    let active_sessions = session_count_fn();
                    
                    match self.client.send_heartbeat(&self.public_ip, active_sessions).await {
                        Ok(response) => {
                            consecutive_failures = 0;
                            
                            // Adjust interval if CMS requests different timing
                            if let Some(next_in) = response.next_heartbeat_in {
                                if next_in > 0 && next_in != self.interval.as_secs() {
                                    debug!("CMS requested next heartbeat in {}s", next_in);
                                }
                            }
                            
                            // Handle any commands from CMS
                            if let Some(commands) = response.commands {
                                for cmd in commands {
                                    debug!("Received command from CMS: {}", cmd);
                                    // Future: handle commands like "update", "restart", etc.
                                }
                            }
                        }
                        Err(e) => {
                            consecutive_failures += 1;
                            
                            if consecutive_failures >= max_failures {
                                error!(
                                    "Heartbeat failed {} consecutive times: {}",
                                    consecutive_failures, e
                                );
                            } else {
                                warn!(
                                    "Heartbeat failed ({}/{}): {}",
                                    consecutive_failures, max_failures, e
                                );
                            }
                        }
                    }
                }
            }
        }

        debug!("Heartbeat reporter stopped");
    }
}

// ============================================
// Session Reporter
// ============================================

/// Background task for session event reporting.
pub struct SessionReporter {
    client: Arc<ManagementClient>,
    event_rx: mpsc::Receiver<SessionEvent>,
}

impl SessionReporter {
    /// Creates a new session reporter and returns the event sender.
    pub fn new(client: Arc<ManagementClient>) -> (Self, mpsc::Sender<SessionEvent>) {
        // Buffer up to 1000 events
        let (tx, rx) = mpsc::channel(1000);
        
        let reporter = Self {
            client,
            event_rx: rx,
        };
        
        (reporter, tx)
    }

    /// Runs the session reporter loop.
    pub async fn run(
        mut self,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    ) {
        info!("Session reporter started");

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("Session reporter shutting down");
                    // Drain remaining events
                    self.drain_events().await;
                    break;
                }
                Some(event) = self.event_rx.recv() => {
                    self.report_event(event).await;
                }
            }
        }

        debug!("Session reporter stopped");
    }

    /// Reports a single event.
    async fn report_event(&self, event: SessionEvent) {
        let report = event.to_report();
        
        match self.client.report_session_event(report).await {
            Ok(response) => {
                if !response.success {
                    if let Some(err) = response.error {
                        warn!("Session event report rejected: {}", err);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to report session event: {}", e);
                // Events are dropped on failure (could implement retry queue)
            }
        }
    }

    /// Drains remaining events during shutdown.
    async fn drain_events(&mut self) {
        let mut count = 0;
        
        while let Ok(event) = self.event_rx.try_recv() {
            // Only report session_ended events during shutdown
            if matches!(event.event_type, SessionEventType::SessionEnded) {
                self.report_event(event).await;
                count += 1;
            }
        }
        
        if count > 0 {
            debug!("Drained {} session end events during shutdown", count);
        }
    }
}

// ============================================
// Session Event Sender Helper
// ============================================

/// Helper for sending session events.
#[derive(Clone)]
pub struct SessionEventSender {
    tx: Option<mpsc::Sender<SessionEvent>>,
}

impl SessionEventSender {
    /// Creates a new sender (enabled).
    pub fn new(tx: mpsc::Sender<SessionEvent>) -> Self {
        Self { tx: Some(tx) }
    }

    /// Creates a disabled sender (no-op).
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    /// Sends a session created event.
    pub fn session_created(&self, session_id: &str, client_wallet: Option<String>) {
        if let Some(ref tx) = self.tx {
            let event = SessionEvent::created(session_id.to_string(), client_wallet);
            let _ = tx.try_send(event);
        }
    }

    /// Sends a session updated event.
    pub fn session_updated(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) {
        if let Some(ref tx) = self.tx {
            let event = SessionEvent::updated(
                session_id.to_string(),
                client_wallet,
                bytes_in,
                bytes_out,
            );
            let _ = tx.try_send(event);
        }
    }

    /// Sends a session ended event.
    pub fn session_ended(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
    ) {
        if let Some(ref tx) = self.tx {
            let event = SessionEvent::ended(
                session_id.to_string(),
                client_wallet,
                bytes_in,
                bytes_out,
            );
            let _ = tx.try_send(event);
        }
    }

    /// Returns whether reporting is enabled.
    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }
}
