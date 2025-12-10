// ============================================
// File: crates/aeronyx-server/src/management/reporter.rs
// ============================================
//! # Background Reporters
//!
//! Async tasks for periodic heartbeat and session event reporting.
//! # Background Reporters

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::client::ManagementClient;
use super::models::{SessionEventReport, SessionEventType};

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

pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
}

impl HeartbeatReporter {
    pub fn new(client: Arc<ManagementClient>, public_ip: String) -> Self {
        let interval = Duration::from_secs(client.config().heartbeat_interval_secs);
        Self { client, interval, public_ip }
    }

    pub async fn run<F>(self, session_count_fn: F, mut shutdown: tokio::sync::broadcast::Receiver<()>)
    where F: Fn() -> u32 + Send + 'static
    {
        info!("Heartbeat reporter started ({}s)", self.interval.as_secs());
        let mut interval = tokio::time::interval(self.interval);
        let mut failures = 0u32;

        loop {
            tokio::select! {
                _ = shutdown.recv() => { info!("Heartbeat stopping"); break; }
                _ = interval.tick() => {
                    match self.client.send_heartbeat(&self.public_ip, session_count_fn()).await {
                        Ok(_) => { failures = 0; }
                        Err(e) => {
                            failures += 1;
                            if failures >= 3 { error!("Heartbeat failed: {}", e); }
                            else { warn!("Heartbeat failed: {}", e); }
                        }
                    }
                }
            }
        }
    }
}

pub struct SessionReporter {
    client: Arc<ManagementClient>,
    event_rx: mpsc::Receiver<SessionEvent>,
}

impl SessionReporter {
    pub fn new(client: Arc<ManagementClient>) -> (Self, mpsc::Sender<SessionEvent>) {
        let (tx, rx) = mpsc::channel(1000);
        (Self { client, event_rx: rx }, tx)
    }

    pub async fn run(mut self, mut shutdown: tokio::sync::broadcast::Receiver<()>) {
        info!("Session reporter started");
        loop {
            tokio::select! {
                _ = shutdown.recv() => { info!("Session reporter stopping"); break; }
                Some(event) = self.event_rx.recv() => {
                    if let Err(e) = self.client.report_session_event(event.to_report()).await {
                        warn!("Session report failed: {}", e);
                    }
                }
            }
        }
    }
}

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
