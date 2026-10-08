// ============================================
// File: crates/aeronyx-server/src/server/runtime_supervision.rs
// ============================================
// [SERVER-DECOMPOSITION-PHASE1 2026-09-14 by Codex] Isolate typed runtime
// supervision and shutdown policy without changing task names, timing, logs,
// startup ordering, failure buckets, or the public Server surface.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::{
    sync::{broadcast, mpsc},
    task::{JoinHandle, JoinSet},
};
use tracing::{debug, error, info, warn};

use crate::error::{Result, RuntimeTaskJoinFailureKind, ServerError};
use futures::FutureExt;

use super::{
    CustodyWitnessReadinessBlockReason, Server, DATA_PLANE_RECV_FAILURE_LIMIT,
    DATA_PLANE_RECV_RETRY_BASE_MILLIS, DATA_PLANE_RECV_RETRY_MAX_MILLIS,
};

/// Failure emitted when a required runtime surface disappears after startup.
///
/// [RUNTIME-SUPERVISION 2026-07-29 by Codex] Keep this message local and
/// operational. It may identify a listener role and bind address, but must
/// never contain client identities, routes, payloads, or traffic metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CriticalRuntimeFailure {
    pub(super) task: &'static str,
    pub(super) reason: String,
}

pub(super) fn custody_witness_runtime_failure(
    reason: CustodyWitnessReadinessBlockReason,
) -> CriticalRuntimeFailure {
    // [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Keep supervisor
    // output constrained to stable local policy buckets. Do not pass through
    // storage, cryptographic, path, witness, or anchor error values here.
    CriticalRuntimeFailure {
        task: "custody-witness-runtime",
        reason: reason.as_str().to_string(),
    }
}

pub(super) fn required_runtime_supervisor_channel_closed() -> CriticalRuntimeFailure {
    CriticalRuntimeFailure {
        task: "required_runtime_task_group",
        reason: "critical runtime supervisor channel closed unexpectedly".to_string(),
    }
}

// [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] Keep a required
// recipient worker failure actionable without forwarding panic/error details.
pub(super) fn reverse_onion_recipient_worker_exited() -> CriticalRuntimeFailure {
    CriticalRuntimeFailure {
        task: "reverse-onion-recipient",
        reason: "required recipient worker exited unexpectedly".to_string(),
    }
}

// [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Keep the source
// failure bucket stable and source-blind, like recipient worker supervision.
pub(super) fn reverse_onion_source_failed() -> CriticalRuntimeFailure {
    CriticalRuntimeFailure {
        task: "reverse-onion-source",
        reason: "required source owner failed".to_string(),
    }
}

// [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex] A normal
// shutdown branch is not proof that accepted work stayed healthy. Re-read
// sticky owners after select and after drain; retain an already chosen fault.
// Disabled roles are inert. Normal stopped/drained owners are not failures.
pub(super) fn reconcile_reverse_onion_runtime_failure(
    selected: Option<CriticalRuntimeFailure>,
    source: Option<&super::reverse_onion_source_runtime::ReverseOnionSourceLifecycle>,
    recipient: Option<&super::reverse_onion_runtime::RecipientServerLifecycle>,
) -> Option<CriticalRuntimeFailure> {
    selected.or_else(|| {
        if source.is_some_and(|owner| owner.has_failed()) {
            Some(reverse_onion_source_failed())
        } else if recipient.is_some_and(|owner| owner.has_failed()) {
            Some(reverse_onion_recipient_worker_exited())
        } else {
            None
        }
    })
}

// [PHALA-READY-PUBLICATION 2026-10-07 by Codex] Fault-only fixture helper;
// the actual publication gate below also rejects normal stopped owners.
#[cfg(test)]
pub(super) fn pre_ready_reverse_onion_source_failure(
    source: Option<&super::reverse_onion_source_runtime::ReverseOnionSourceLifecycle>,
) -> Option<CriticalRuntimeFailure> {
    source.filter(|source| source.has_failed()).map(|_| reverse_onion_source_failed())
}

// [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] A disabled role
// is pending, not a failed/missing required owner. Select owns this future;
// dropping a losing branch neither aborts source work nor loses sticky faults.
pub(super) async fn wait_for_reverse_onion_source_failure(
    source: Option<&super::reverse_onion_source_runtime::ReverseOnionSourceLifecycle>,
) -> CriticalRuntimeFailure {
    if let Some(source) = source {
        source.wait_for_failure().await;
        reverse_onion_source_failed()
    } else {
        std::future::pending().await
    }
}

/// Consumes a failure already known before the process advertises readiness.
///
/// [PRE-READY-RUNTIME-GATE 2026-07-30 by Codex] An empty channel means all
/// registered supervisors are still eligible to report. A disconnected
/// channel means every supervisor disappeared and is itself a critical
/// failure; it must not be treated like an empty startup queue.
pub(super) fn take_pre_ready_runtime_failure(
    critical_failure_rx: &mut mpsc::Receiver<CriticalRuntimeFailure>,
) -> Option<CriticalRuntimeFailure> {
    match critical_failure_rx.try_recv() {
        Ok(failure) => Some(failure),
        Err(mpsc::error::TryRecvError::Empty) => None,
        Err(mpsc::error::TryRecvError::Disconnected) => {
            Some(required_runtime_supervisor_channel_closed())
        }
    }
}

// [PHALA-READY-PUBLICATION 2026-10-07 by Codex] No await may separate this
// final decision from READY. This is a fresh local snapshot, not an atomic
// promise about future faults, network delivery or Phala attestation.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PreReadyRuntimeDecision {
    Ready,
    Stopped,
    Failed(CriticalRuntimeFailure),
}

// [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] A stop request must wake
// startup/runtime without telling dependency tasks that reverse drain ended.
// watch retains the request across absent, cancelled and replacement waiters.
#[derive(Clone)]
pub(super) struct ReverseRunStop {
    requested: tokio::sync::watch::Sender<bool>,
}

impl Default for ReverseRunStop {
    fn default() -> Self {
        let (requested, _) = tokio::sync::watch::channel(false);
        Self { requested }
    }
}

impl ReverseRunStop {
    pub(super) fn request_stop(&self) { self.requested.send_replace(true); }

    pub(super) fn is_requested(&self) -> bool { *self.requested.borrow() }

    pub(super) async fn cancelled(&self) {
        let mut receiver = self.requested.subscribe();
        let _ = receiver.wait_for(|requested| *requested).await;
    }

    pub(super) fn before_ready(&self, decision: PreReadyRuntimeDecision) -> PreReadyRuntimeDecision {
        if matches!(decision, PreReadyRuntimeDecision::Ready) && self.is_requested() {
            PreReadyRuntimeDecision::Stopped
        } else { decision }
    }
}

// [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex] Register before
// startup can yield. Reuse one listener future across the READY boundary and
// retain its first observation through drain. Cancelling a select waiter must
// not consume or replace the stop event.
pub(super) struct ProcessShutdownSignals {
    waiter: std::pin::Pin<Box<dyn std::future::Future<Output = Option<CriticalRuntimeFailure>> + Send>>,
    observed: Option<Option<CriticalRuntimeFailure>>,
}

impl ProcessShutdownSignals {
    pub(super) fn install() -> std::result::Result<Self, CriticalRuntimeFailure> {
        #[cfg(unix)]
        let mut signals = {
            use tokio::signal::unix::{signal, SignalKind};
            let mut terminate = signal(SignalKind::terminate())
                .map_err(|error| Server::shutdown_signal_failure("SIGTERM", error))?;
            let mut interrupt = signal(SignalKind::interrupt())
                .map_err(|error| Server::shutdown_signal_failure("SIGINT", error))?;
            Self::from_waiter(async move {
                let (name, received) = tokio::select! {
                    received = terminate.recv() => ("SIGTERM", received),
                    received = interrupt.recv() => ("SIGINT", received),
                };
                match received {
                    Some(()) => {
                        info!(signal = name, "Shutdown signal received");
                        None
                    }
                    None => Some(Server::shutdown_signal_failure(name, "signal stream closed unexpectedly")),
                }
            })
        };
        #[cfg(not(unix))]
        let mut signals = {
            // Poll once in from_waiter so CTRL_C registration is not delayed
            // until the post-READY shutdown select on non-Unix platforms.
            Self::from_waiter(async {
                match tokio::signal::ctrl_c().await {
                    Ok(()) => {
                        info!(signal = "CTRL_C", "Shutdown signal received");
                        None
                    }
                    Err(error) => Some(Server::shutdown_signal_failure("CTRL_C", error)),
                }
            })
        };
        if let Some(Some(failure)) = signals.take_pending() {
            return Err(failure);
        }
        Ok(signals)
    }

    pub(super) fn from_waiter(
        waiter: impl std::future::Future<Output = Option<CriticalRuntimeFailure>> + Send + 'static,
    ) -> Self {
        let mut signals = Self { waiter: Box::pin(waiter), observed: None };
        let _ = signals.take_pending();
        signals
    }

    pub(super) fn take_pending(&mut self) -> Option<Option<CriticalRuntimeFailure>> {
        if self.observed.is_none() {
            self.observed = self.waiter.as_mut().now_or_never();
        }
        self.observed.clone()
    }

    pub(super) async fn wait(&mut self) -> Option<CriticalRuntimeFailure> {
        if let Some(result) = &self.observed { return result.clone(); }
        let result = self.waiter.as_mut().await;
        self.observed = Some(result.clone());
        result
    }

    pub(super) fn before_ready(&mut self, decision: PreReadyRuntimeDecision) -> PreReadyRuntimeDecision {
        // Known runtime failure outranks a racing ordinary signal. Neither
        // case publishes global shutdown before reverse owners have drained.
        if matches!(decision, PreReadyRuntimeDecision::Failed(_)) { return decision; }
        match self.take_pending() {
            Some(Some(failure)) => PreReadyRuntimeDecision::Failed(failure),
            Some(None) => PreReadyRuntimeDecision::Stopped,
            None => decision,
        }
    }
}

pub(super) fn pre_ready_runtime_decision(
    critical_failure_rx: &mut mpsc::Receiver<CriticalRuntimeFailure>,
    shutdown_requested: &AtomicBool,
    source: Option<&super::reverse_onion_source_runtime::ReverseOnionSourceLifecycle>,
    recipient: Option<&super::reverse_onion_runtime::RecipientServerLifecycle>,
    recipient_wait_failed: bool,
) -> PreReadyRuntimeDecision {
    use super::reverse_onion_source_runtime::SourceRuntimeError;

    if let Some(failure) = take_pre_ready_runtime_failure(critical_failure_rx) {
        return PreReadyRuntimeDecision::Failed(failure);
    }
    let source_readiness = source.map(|owner| owner.verify_ready_now());
    if matches!(source_readiness, Some(Err(error)) if error != SourceRuntimeError::Stopped) {
        return PreReadyRuntimeDecision::Failed(reverse_onion_source_failed());
    }
    if let Some(owner) = recipient {
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] An
        // already published local fault outranks a racing normal cancellation.
        if owner.has_failed() || (!owner.is_cancelled() && (recipient_wait_failed || owner.verify_ready_now().is_err())) {
            return PreReadyRuntimeDecision::Failed(reverse_onion_recipient_worker_exited());
        }
    }
    if shutdown_requested.load(Ordering::Acquire)
        || matches!(source_readiness, Some(Err(SourceRuntimeError::Stopped)))
        || recipient.is_some_and(|owner| owner.is_cancelled())
    {
        return PreReadyRuntimeDecision::Stopped;
    }
    PreReadyRuntimeDecision::Ready
}

/// Action after one consecutive required data-plane receive failure.
///
/// [DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex] Keep the policy typed and
/// source-blind. Runtime logs may include the local OS error, but process
/// supervision receives only the required task name and a fixed reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DataPlaneReceiveFailureAction {
    RetryAfter(Duration),
    Stop,
}

pub(super) fn data_plane_receive_failure_action(
    consecutive_failures: u32,
) -> DataPlaneReceiveFailureAction {
    if consecutive_failures >= DATA_PLANE_RECV_FAILURE_LIMIT {
        return DataPlaneReceiveFailureAction::Stop;
    }

    let exponent = consecutive_failures.saturating_sub(1).min(6);
    let delay_millis = DATA_PLANE_RECV_RETRY_BASE_MILLIS
        .saturating_mul(1u64 << exponent)
        .min(DATA_PLANE_RECV_RETRY_MAX_MILLIS);
    DataPlaneReceiveFailureAction::RetryAfter(Duration::from_millis(delay_millis))
}

/// Applies one source-blind receive failure to a required data-plane task.
///
/// Returns `true` after the bounded retry delay, or `false` when global
/// shutdown or the consecutive-failure limit requires the caller to stop.
pub(super) async fn retry_required_data_plane_receive<E: std::fmt::Display>(
    task: &'static str,
    receive_error: &E,
    consecutive_failures: &mut u32,
    shutdown_requested: &AtomicBool,
    shutdown_rx: &mut broadcast::Receiver<()>,
) -> bool {
    if shutdown_requested.load(Ordering::Acquire) {
        return false;
    }

    *consecutive_failures = consecutive_failures.saturating_add(1);
    match data_plane_receive_failure_action(*consecutive_failures) {
        DataPlaneReceiveFailureAction::Stop => {
            error!(
                task,
                consecutive_failures = *consecutive_failures,
                error = %receive_error,
                "[DATA_PLANE] Receive failure limit reached"
            );
            false
        }
        DataPlaneReceiveFailureAction::RetryAfter(delay) => {
            // [DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex] Log the first
            // and power-of-two failures only, preventing a broken descriptor
            // from flooding local operations logs before process recovery.
            if *consecutive_failures == 1 || consecutive_failures.is_power_of_two() {
                warn!(
                    task,
                    consecutive_failures = *consecutive_failures,
                    retry_delay_millis = delay.as_millis() as u64,
                    error = %receive_error,
                    "[DATA_PLANE] Receive failed; retrying"
                );
            }
            tokio::select! {
                biased;
                _ = shutdown_rx.recv() => false,
                _ = tokio::time::sleep(delay) => true,
            }
        }
    }
}

/// Owns an inner required task without inheriting Tokio's detach-on-drop
/// behavior.
///
/// [REQUIRED-TASK-OWNERSHIP 2026-07-30 by Codex] Aborting the outer
/// supervisor during bounded shutdown must also abort its inner task. A plain
/// dropped `JoinHandle` would detach the inner future and leak it past the
/// shutdown report.
struct RequiredRuntimeTaskJoinGuard {
    task: JoinHandle<()>,
}

impl RequiredRuntimeTaskJoinGuard {
    fn new(task: JoinHandle<()>) -> Self {
        Self { task }
    }

    async fn join(&mut self) -> std::result::Result<(), tokio::task::JoinError> {
        (&mut self.task).await
    }
}

impl Drop for RequiredRuntimeTaskJoinGuard {
    fn drop(&mut self) {
        if !self.task.is_finished() {
            self.task.abort();
        }
    }
}

/// Owns all process-lifetime tasks across startup and normal operation.
///
/// [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Tokio detaches a task when its
/// `JoinHandle` is dropped. Keeping every handle here makes startup
/// transactional: any early return aborts registered work, while the explicit
/// handoff preserves the existing bounded graceful-shutdown path.
#[derive(Default)]
pub(super) struct RuntimeTaskRegistry {
    tasks: Vec<(&'static str, JoinHandle<()>)>,
}

impl RuntimeTaskRegistry {
    pub(super) fn push(&mut self, task: (&'static str, JoinHandle<()>)) {
        self.tasks.push(task);
    }

    // [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] The outer
    // owner survives an inner unwind; taking tasks leaves it safely empty.
    pub(super) fn take_for_shutdown(&mut self) -> Vec<(&'static str, JoinHandle<()>)> {
        std::mem::take(&mut self.tasks)
    }
}

// [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] Keep handles and
// queue permits outside the fallible startup/runtime future. Tasks and the
// accepted worker/router clones retain their Arc dependencies through drain.
// This protects Rust unwinding, not panic=abort, process kill or runtime loss.
#[derive(Default)]
pub(super) struct ReverseRuntimeDependencies {
    pub(super) tasks: RuntimeTaskRegistry,
    pub(super) queue: Option<super::ReverseOnionQueueRuntime>,
}

impl ReverseRuntimeDependencies {
    pub(super) async fn drain_reverse_owners(
        &self,
        source: Option<&super::reverse_onion_source_runtime::ReverseOnionSourceLifecycle>,
        recipient: Option<&super::reverse_onion_runtime::RecipientServerLifecycle>,
    ) -> Result<()> {
        // Stop every gate before the first await. A returned drain error must
        // not skip another owner's accepted work or release generic handles.
        if let Some(recipient) = recipient { recipient.request_stop(); }
        if let Some(source) = source { source.request_stop(); }
        if let Some(queue) = &self.queue { queue.request_stop(); }
        let recipient_drain = if let Some(recipient) = recipient {
            recipient.drain().await.map_err(|_| ServerError::startup_failed("Recipient drain failed"))
        } else { Ok(()) };
        let source_drain = if let Some(source) = source {
            source.shutdown_and_drain().await.map_err(|_| ServerError::startup_failed("Source drain failed"))
        } else { Ok(()) };
        if let Some(queue) = &self.queue { queue.shutdown_and_drain().await; }
        recipient_drain?;
        source_drain
    }
}

// [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] Only the inner
// future is unwound. Its caller must keep ReverseRuntimeDependencies alive
// until reverse drain and generic task shutdown complete. Never expose payload.
pub(super) async fn catch_reverse_runtime_unwind(
    runtime: impl std::future::Future<Output = Result<()>>,
) -> Result<()> {
    // [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] Refuse before
    // polling the inner future if this executable would abort instead of drain.
    crate::config_reverse_onion::require_reverse_onion_unwind_support()?;
    std::panic::AssertUnwindSafe(runtime).catch_unwind().await
        .unwrap_or_else(|_| Err(ServerError::runtime_failed(
            "reverse-owned-server", "required reverse server owner unwound",
        )))
}

impl Drop for RuntimeTaskRegistry {
    fn drop(&mut self) {
        for (name, task) in &self.tasks {
            if !task.is_finished() {
                task.abort();
                debug!(
                    task = *name,
                    "[STARTUP] Aborted owned runtime task while unwinding startup"
                );
            }
        }
    }
}

/// Result of bringing one long-lived runtime task to a terminal state.
///
/// [TASK-SHUTDOWN 2026-07-29 by Codex] Keep shutdown outcomes typed so timeout
/// cancellation cannot be confused with a task that completed cooperatively.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum RuntimeTaskShutdownOutcome {
    Completed,
    JoinFailed(RuntimeTaskJoinFailureKind),
    CancelledAfterTimeout,
    CompletedAfterTimeout,
    CancellationUnconfirmed,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct RuntimeTaskShutdownReport {
    pub(super) name: &'static str,
    pub(super) outcome: RuntimeTaskShutdownOutcome,
}

/// Terminal result from one required API listener.
#[derive(Debug)]
pub(super) struct RequiredApiListenerExit {
    pub(super) role: &'static str,
    pub(super) address: SocketAddr,
    pub(super) result: std::io::Result<()>,
}

impl RequiredApiListenerExit {
    fn into_failure(self) -> CriticalRuntimeFailure {
        let reason = match self.result {
            Ok(()) => format!(
                "required listener {} exited unexpectedly without an I/O error",
                self.address
            ),
            // [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] The listener already
            // records its local I/O diagnostic at the failure site. Keep the
            // process-health message fixed so it cannot become an accidental
            // carrier for implementation-specific or request-derived text.
            Err(_) => format!("required listener {} failed", self.address),
        };
        CriticalRuntimeFailure {
            task: self.role,
            reason,
        }
    }
}

impl Server {
    /// Propagates the first unexpected required-listener exit to `run()`.
    pub(super) async fn supervise_required_api_listeners(
        mut listeners: JoinSet<RequiredApiListenerExit>,
        shutdown_requested: Arc<AtomicBool>,
        mut shutdown_rx: broadcast::Receiver<()>,
        critical_failure_tx: mpsc::Sender<CriticalRuntimeFailure>,
    ) {
        // [RUNTIME-SUPERVISION 2026-07-29 by Codex] A required listener is a
        // process-health boundary. Expected exits happen only after the main
        // task marks shutdown before broadcasting the stop signal.
        let first_exit = listeners.join_next().await;
        if !shutdown_requested.load(Ordering::Acquire) {
            let failure = match first_exit {
                Some(Ok(listener_exit)) => listener_exit.into_failure(),
                Some(Err(error)) => CriticalRuntimeFailure {
                    task: "required_api_listener_group",
                    reason: RuntimeTaskJoinFailureKind::classify(&error)
                        .required_api_listener_reason()
                        .to_string(),
                },
                None => CriticalRuntimeFailure {
                    task: "required_api_listener_group",
                    reason: "listener supervisor had no registered tasks".to_string(),
                },
            };
            error!(
                task = failure.task,
                reason = %failure.reason,
                "[RUNTIME] Required API listener exited unexpectedly"
            );
            if critical_failure_tx.send(failure).await.is_err() {
                error!(
                    "[RUNTIME] Main task dropped the critical failure receiver; aborting listener group"
                );
                return;
            }

            // Let the main task publish STOPPING, mark shutdown, and broadcast
            // one shared graceful-stop signal. The bound prevents a secondary
            // control-path failure from pinning this supervisor forever.
            let _ = tokio::time::timeout(Duration::from_secs(5), shutdown_rx.recv()).await;
        }

        while let Some(result) = listeners.join_next().await {
            match result {
                Ok(listener_exit) => debug!(
                    listener_role = listener_exit.role,
                    address = %listener_exit.address,
                    "[RUNTIME] Required listener joined during shutdown"
                ),
                Err(error) => warn!(
                    failure = ?RuntimeTaskJoinFailureKind::classify(&error),
                    "[RUNTIME] Required listener join failed during shutdown"
                ),
            }
        }
    }

    // ============================================
    // Shutdown
    // ============================================

    /// Wraps one already-spawned required task with process-level supervision.
    ///
    /// [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] The wrapper owns the
    /// `JoinHandle`, so normal unexpected return, panic, and cancellation all
    /// reach the main failure channel. Reasons are fixed strings: a panic
    /// payload, endpoint, peer, route, or user value can never enter status or
    /// logs through this boundary.
    pub(super) fn supervise_required_runtime_task(
        name: &'static str,
        task: JoinHandle<()>,
        shutdown_requested: Arc<AtomicBool>,
        critical_failure_tx: mpsc::Sender<CriticalRuntimeFailure>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut task = RequiredRuntimeTaskJoinGuard::new(task);
            let result = task.join().await;
            if shutdown_requested.load(Ordering::Acquire) {
                debug!(task = name, "Required runtime task joined during shutdown");
                return;
            }

            let reason = match result {
                Ok(()) => "required runtime task exited unexpectedly",
                Err(error) => RuntimeTaskJoinFailureKind::classify(&error).required_task_reason(),
            };
            let failure = CriticalRuntimeFailure {
                task: name,
                reason: reason.to_string(),
            };
            error!(
                task = failure.task,
                reason = %failure.reason,
                "[RUNTIME] Required runtime task disappeared"
            );
            if critical_failure_tx.send(failure).await.is_err() {
                error!(
                    task = name,
                    "[RUNTIME] Main task dropped the critical failure receiver"
                );
            }
        })
    }

    pub(super) fn runtime_task_shutdown_grace(name: &str) -> Duration {
        match name {
            // The coordinator lease task may finish one bounded renewal and
            // one bounded release round before exit.
            "memchain-coordinator-lease" => Duration::from_secs(12),
            // One sequential custody+source cycle may observe both stores'
            // bounded SQLite busy timeouts before it can receive shutdown.
            // Keep the join below the service manager's 30-second ceiling.
            "anonymous-mailbox-cleanup" => Duration::from_secs(15),
            // [DIRECTORY-SHUTDOWN-DURABILITY 2026-08-12 by Codex] The final
            // append performs a complete signed-prefix audit in `spawn_blocking`.
            // Tokio cannot cancel a running blocking closure, so aborting its
            // async owner after the generic five seconds only detaches active
            // SQLite work. Twenty seconds covers the measured large-node audit
            // while retaining room beneath the deployed 30-second systemd stop
            // ceiling for listener and transport cleanup.
            "directory-chain-persistence" => Duration::from_secs(20),
            // One in-progress Directory proof audit cannot be cancelled by
            // Tokio after entering the blocking pool. Shutdown checkpoints
            // prevent the round from starting another bounded network/probe
            // stage after that audit completes.
            "discovery-gossip" => Duration::from_secs(24),
            _ => Duration::from_secs(5),
        }
    }

    pub(super) async fn runtime_delay_interrupted_by_shutdown(
        shutdown_rx: &mut broadcast::Receiver<()>,
        delay: Duration,
    ) -> bool {
        tokio::select! {
            _ = shutdown_rx.recv() => true,
            _ = tokio::time::sleep(delay) => false,
        }
    }

    pub(super) async fn join_runtime_task(
        name: &'static str,
        mut task: JoinHandle<()>,
        grace: Duration,
        abort_confirmation: Duration,
    ) -> RuntimeTaskShutdownReport {
        // [TASK-SHUTDOWN 2026-07-29 by Codex] Passing `&mut JoinHandle` keeps
        // ownership after timeout. Dropping the old timeout future detached the
        // still-running task; retaining the handle lets us request and verify
        // cancellation explicitly.
        let outcome = match tokio::time::timeout(grace, &mut task).await {
            Ok(Ok(())) => RuntimeTaskShutdownOutcome::Completed,
            Ok(Err(error)) => {
                RuntimeTaskShutdownOutcome::JoinFailed(RuntimeTaskJoinFailureKind::classify(&error))
            }
            Err(_) => {
                task.abort();
                match tokio::time::timeout(abort_confirmation, &mut task).await {
                    Ok(Ok(())) => RuntimeTaskShutdownOutcome::CompletedAfterTimeout,
                    Ok(Err(error)) if error.is_cancelled() => {
                        RuntimeTaskShutdownOutcome::CancelledAfterTimeout
                    }
                    Ok(Err(error)) => RuntimeTaskShutdownOutcome::JoinFailed(
                        RuntimeTaskJoinFailureKind::classify(&error),
                    ),
                    Err(_) => RuntimeTaskShutdownOutcome::CancellationUnconfirmed,
                }
            }
        };
        RuntimeTaskShutdownReport { name, outcome }
    }

    pub(super) async fn shutdown_runtime_tasks(
        tasks: Vec<(&'static str, JoinHandle<()>)>,
    ) -> Vec<RuntimeTaskShutdownReport> {
        // Poll all graceful joins together. `join_all` preserves registration
        // order in the reports without serializing independent deadlines.
        futures::future::join_all(tasks.into_iter().map(|(name, task)| {
            Self::join_runtime_task(
                name,
                task,
                Self::runtime_task_shutdown_grace(name),
                Duration::from_secs(1),
            )
        }))
        .await
    }

    // [PHALA-EARLY-SHUTDOWN-SIGNALS 2026-10-07 by Codex] Reuse the
    // pre-startup listener. A losing select leaves its sticky observation
    // available to the READY gate and subsequent shutdown waiter.
    pub(super) async fn wait_for_shutdown(
        &self,
        critical_failure_rx: &mut mpsc::Receiver<CriticalRuntimeFailure>,
        signals: &mut ProcessShutdownSignals,
    ) -> Option<CriticalRuntimeFailure> {
        // Subscribe before the flag sample so programmatic stop is not lost.
        let mut programmatic_shutdown_rx = self.shutdown_tx.subscribe();
        if self.shutdown.load(Ordering::Acquire) {
            info!(signal = "PROGRAMMATIC", "Shutdown request already pending");
            return None;
        }
        // [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] Observe a sticky
        // reverse stop without closing dependencies. Known required failures
        // retain priority; reverse owner faults are reconciled after drain.
        if self.reverse_run_stop.is_requested() {
            return take_pre_ready_runtime_failure(critical_failure_rx);
        }
        tokio::select! {
            result = signals.wait() => result,
            _ = self.reverse_run_stop.cancelled() => {
                take_pre_ready_runtime_failure(critical_failure_rx)
            }
            _ = programmatic_shutdown_rx.recv() => {
                info!(signal = "PROGRAMMATIC", "Shutdown signal received");
                None
            }
            failure = critical_failure_rx.recv() => {
                if self.shutdown.load(Ordering::Acquire) {
                    None
                } else {
                    Some(failure.unwrap_or_else(required_runtime_supervisor_channel_closed))
                }
            }
        }
    }

    pub(super) fn shutdown_signal_failure(
        signal: &'static str,
        reason: impl std::fmt::Display,
    ) -> CriticalRuntimeFailure {
        // [FAIL-CLOSED-SHUTDOWN-SIGNALS 2026-08-12 by Codex] Signal setup is
        // part of the required runtime surface. Report a typed local failure so
        // `run()` executes the same persistence and task-join path used for any
        // other required worker loss instead of unwinding the process.
        CriticalRuntimeFailure {
            task: "shutdown_signal_listener",
            reason: format!("{signal} shutdown signal listener failed: {reason}"),
        }
    }

    pub fn shutdown(&self) {
        // [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] Reverse-enabled
        // servers publish global stop only after accepted work has drained.
        // Preserve immediate shutdown for unrelated legacy server roles.
        if self.config.reverse_onion.queue.enabled || self.reverse_source.is_some()
            || self.reverse_recipient.is_some()
        {
            if let Some(source) = &self.reverse_source { source.request_stop(); }
            if let Some(recipient) = &self.reverse_recipient { recipient.request_stop(); }
            self.reverse_run_stop.request_stop();
            return;
        }
        self.publish_dependency_shutdown();
    }

    // [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] Called by the retained
    // reverse owner only after all drain attempts, including startup failure.
    pub(super) fn publish_dependency_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());
    }
}
