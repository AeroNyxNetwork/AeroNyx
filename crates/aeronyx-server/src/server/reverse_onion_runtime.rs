// ============================================
// File: crates/aeronyx-server/src/server/reverse_onion_runtime.rs
// ============================================
//! Bounded outbound carrier for private-recipient reverse onion delivery.
//!
//! [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] The carrier remains
//! transport-only. The separately owned worker journals before send/dispatch,
//! authenticates relay execution authority, and drains before releasing DB.
//! Startup callers must await readiness and retain/drain the worker at shutdown.
//! No redirects, inherited proxies, DNS endpoints, or automatic retries.

use std::time::Duration;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use aeronyx_core::crypto::keys::IdentityKeyPair;
use futures::FutureExt;
use rand::RngCore;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionKindV1, MAX_REVERSE_ONION_FRAME_BYTES,
};

use crate::config_reverse_onion::ReverseOnionConfig;
use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
use crate::services::reverse_onion_recipient::{
    RecipientJournalError, RecipientJournalLimits, RecipientRecovery,
    ReverseOnionRecipientJournal,
};

/// Coarse preflight errors have no network effects or identifying details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionPreflightError {
    Disabled,
    Configuration,
    Frame,
}

/// A response is transport evidence only, never a custody/execution proof.
/// Intentionally omits Debug so opaque bodies cannot enter routine logs.
pub(crate) enum ReverseOnionExchange {
    /// An HTTP response arrived within both the time and body size bounds.
    Response { status: u16, body: Vec<u8> },
    /// A POST may have reached the relay. Preserve the exact durable request;
    /// this must never cause a new claim, lease, target or source reply key.
    Ambiguous,
}

/// Fixed adjacent relay and recipient identity for one outbound worker.
pub(crate) struct ReverseOnionHttpCarrier {
    client: reqwest::Client,
    relay: [u8; 32],
    recipient: [u8; 32],
    claim_url: reqwest::Url,
    result_url: reqwest::Url,
    timeout: Duration,
}

impl ReverseOnionHttpCarrier {
    /// Construct without opening sockets. Endpoint safety is checked before
    /// canonical route composition, including credentials and query rejection.
    pub(crate) fn new(
        config: &ReverseOnionConfig,
        recipient: [u8; 32],
    ) -> Result<Self, ReverseOnionPreflightError> {
        if !config.recipient.enabled {
            return Err(ReverseOnionPreflightError::Disabled);
        }
        config.validate().map_err(|_| ReverseOnionPreflightError::Configuration)?;
        let mut relay = [0u8; 32];
        hex::decode_to_slice(&config.recipient.relay_node_id, &mut relay)
            .map_err(|_| ReverseOnionPreflightError::Configuration)?;
        if recipient == [0; 32] || recipient == relay {
            return Err(ReverseOnionPreflightError::Configuration);
        }
        let timeout = Duration::from_secs(config.recipient.request_timeout_secs);
        let client = crate::api::privacy_safe_peer_http_client_builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|_| ReverseOnionPreflightError::Configuration)?;
        let endpoint = &config.recipient.relay_endpoint;
        Ok(Self {
            client,
            relay,
            recipient,
            claim_url: crate::api::canonical_peer_http_url(
                endpoint, "/api/chat/peer/reverse-onion/claim",
            ).map_err(|_| ReverseOnionPreflightError::Configuration)?,
            result_url: crate::api::canonical_peer_http_url(
                endpoint, "/api/chat/peer/reverse-onion/result",
            ).map_err(|_| ReverseOnionPreflightError::Configuration)?,
            timeout,
        })
    }

    /// Submit exactly one already-persisted frame. No retry is performed here.
    /// Even a non-2xx HTTP response requires protocol-specific recovery policy;
    /// a successful status without a verified bound receipt proves nothing.
    pub(crate) async fn exchange(
        &self,
        frame: &ReverseOnionFrameV1,
    ) -> Result<ReverseOnionExchange, ReverseOnionPreflightError> {
        if frame.relay() != self.relay || frame.immediate_recipient() != self.recipient {
            return Err(ReverseOnionPreflightError::Frame);
        }
        let url = match frame.kind() {
            ReverseOnionKindV1::Claim => &self.claim_url,
            ReverseOnionKindV1::Result => &self.result_url,
            ReverseOnionKindV1::Lease => return Err(ReverseOnionPreflightError::Frame),
        };
        let bytes = frame.encode();
        if bytes.len() > MAX_REVERSE_ONION_FRAME_BYTES {
            return Err(ReverseOnionPreflightError::Frame);
        }
        // One absolute timeout covers headers and streaming the complete body.
        // Never retain raw reqwest errors: they may contain the relay URL.
        let attempt = async {
            let mut response = self.client.post(url.clone())
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(bytes)
                .send().await.map_err(|_| ())?;
            if response.content_length().is_some_and(|n| n > MAX_REVERSE_ONION_FRAME_BYTES as u64) {
                return Err(());
            }
            let status = response.status().as_u16();
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
                if chunk.len() > MAX_REVERSE_ONION_FRAME_BYTES.saturating_sub(body.len()) {
                    return Err(());
                }
                body.extend_from_slice(&chunk);
            }
            Ok::<_, ()>(ReverseOnionExchange::Response { status, body })
        };
        Ok(match tokio::time::timeout(self.timeout, attempt).await {
            Ok(Ok(response)) => response,
            _ => ReverseOnionExchange::Ambiguous,
        })
    }
}

// [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] Single owned task,
// not detached per-tick effects. Limits are independent of network success.
const RECIPIENT_PAGE: usize = 64;
const RECIPIENT_LOCAL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecipientWorkerError { Rejected, Unavailable, Ambiguous }

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerReady { Starting, Ready, Failed }

struct WorkerStop { closed: AtomicBool, wake: Notify }

/// No Debug or abort API. Dropping a shutdown waiter never loses its task handle.
/// The composition owner must retain this object until shutdown_and_drain ends;
/// Drop requests stop, but cannot synchronously promise completed async drain.
pub(crate) struct ReverseOnionRecipientWorker {
    stop: Arc<WorkerStop>,
    ready: watch::Receiver<WorkerReady>,
    task: Mutex<Option<JoinHandle<Result<(), RecipientWorkerError>>>>,
    drain_waiter: tokio::sync::Mutex<()>,
    finished: Mutex<Option<Result<(), RecipientWorkerError>>>,
}

impl ReverseOnionRecipientWorker {
    /// Disabled/configuration rejection happens before task or DB creation.
    /// `adapter` must be the reviewed local terminal adapter, never a network
    /// forwarder. No router registration or public endpoint is added here.
    pub(crate) fn start(config: &ReverseOnionConfig, identity: Arc<IdentityKeyPair>,
        adapter: Arc<ReverseOnionTerminalAdapter>) -> Result<Self, ReverseOnionPreflightError> {
        let carrier = ReverseOnionHttpCarrier::new(config, identity.public_key_bytes())?;
        let path = std::path::PathBuf::from(&config.recipient.state_db_path);
        let entries = config.recipient.max_pending_items as usize;
        let delay = Duration::from_millis(config.recipient.poll_interval_ms);
        let stop = Arc::new(WorkerStop { closed: AtomicBool::new(false), wake: Notify::new() });
        let task_stop = Arc::clone(&stop);
        let (ready_tx, ready) = watch::channel(WorkerReady::Starting);
        // Register synchronously: no cancellation point between spawn and owner.
        let task = tokio::spawn(async move {
            let relay = carrier.relay;
            let recipient = carrier.recipient;
            let opened = tokio::task::spawn_blocking(move || {
                ReverseOnionRecipientJournal::open(&path, relay, recipient,
                    RecipientJournalLimits { max_entries: entries, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now()?)
                    .map(Arc::new).map_err(|_| RecipientWorkerError::Unavailable)
            }).await;
            let journal = match opened {
                Ok(Ok(journal)) => journal,
                _ => {
                    let _ = ready_tx.send(WorkerReady::Failed);
                    let _ = adapter.shutdown_and_drain().await;
                    return Err(RecipientWorkerError::Unavailable);
                }
            };
            let _ = ready_tx.send(WorkerReady::Ready);
            let outcome = std::panic::AssertUnwindSafe(recipient_loop(
                &carrier, &journal, &adapter, &identity, &task_stop, delay))
                .catch_unwind().await.unwrap_or(Err(RecipientWorkerError::Ambiguous));
            task_stop.closed.store(true, Ordering::SeqCst);
            // No worker future is selected away at a DB/terminal await. Even
            // adapter timeout leaves owned work which must finish before DBdrop.
            let drained = adapter.shutdown_and_drain().await
                .map_err(|_| RecipientWorkerError::Ambiguous);
            drop(journal);
            outcome.and(drained)
        });
        Ok(Self { stop, ready, task: Mutex::new(Some(task)),
            drain_waiter: tokio::sync::Mutex::new(()), finished: Mutex::new(None) })
    }

    pub(crate) async fn wait_ready(&self) -> Result<(), RecipientWorkerError> {
        let mut ready = self.ready.clone();
        loop {
            let state = *ready.borrow_and_update();
            match state {
                WorkerReady::Ready if !self.stop.closed.load(Ordering::SeqCst) => return Ok(()),
                WorkerReady::Ready | WorkerReady::Failed => return Err(RecipientWorkerError::Unavailable),
                WorkerReady::Starting => {}
            }
            ready.changed().await.map_err(|_| RecipientWorkerError::Unavailable)?;
        }
    }

    pub(crate) async fn shutdown_and_drain(&self) -> Result<(), RecipientWorkerError> {
        self.request_stop();
        self.wait_completion().await
    }

    // [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Supervision may wait
    // without closing intake; shutdown closes it BEFORE waiting on the joiner.
    pub(crate) fn request_stop(&self) {
        self.stop.closed.store(true, Ordering::SeqCst);
        self.stop.wake.notify_one();
    }

    pub(crate) async fn wait_completion(&self) -> Result<(), RecipientWorkerError> {
        let _waiter = self.drain_waiter.lock().await;
        if let Some(result) = *self.finished.lock().map_err(|_| RecipientWorkerError::Unavailable)? {
            return result;
        }
        futures::future::poll_fn(|cx| {
            let mut slot = match self.task.lock() { Ok(s) => s, Err(_) => return Poll::Ready(Err(RecipientWorkerError::Unavailable)) };
            let Some(task) = slot.as_mut() else { return Poll::Ready(Err(RecipientWorkerError::Unavailable)); };
            match Pin::new(task).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(joined) => {
                    let result = joined.unwrap_or(Err(RecipientWorkerError::Ambiguous));
                    match self.finished.lock() {
                        Ok(mut finished) => *finished = Some(result),
                        Err(_) => return Poll::Ready(Err(RecipientWorkerError::Unavailable)),
                    }
                    *slot = None;
                    Poll::Ready(result)
                }
            }
        }).await
    }
}

// [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Dedicated ownership outside
// RuntimeTaskRegistry's abort-on-drop policy. No DB/task is created by new().
pub(super) struct RecipientServerLifecycle {
    worker: Mutex<Option<Arc<ReverseOnionRecipientWorker>>>,
    pub(super) active: AtomicBool,
    cancelled: AtomicBool,
    wake: Notify,
}

impl RecipientServerLifecycle {
    pub(super) fn new() -> Self {
        Self { worker: Mutex::new(None), active: AtomicBool::new(false),
            cancelled: AtomicBool::new(false), wake: Notify::new() }
    }

    pub(super) fn install(&self, worker: Arc<ReverseOnionRecipientWorker>) -> Result<(), RecipientWorkerError> {
        let mut slot = self.worker.lock().map_err(|_| RecipientWorkerError::Unavailable)?;
        if slot.is_some() || !self.active.load(Ordering::SeqCst) {
            return Err(RecipientWorkerError::Rejected);
        }
        if self.cancelled.load(Ordering::SeqCst) { worker.request_stop(); }
        *slot = Some(worker);
        Ok(())
    }

    pub(super) fn request_stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.worker.lock() {
            if let Some(worker) = slot.as_ref() { worker.request_stop(); }
        }
        self.wake.notify_one();
    }

    pub(super) fn is_cancelled(&self) -> bool { self.cancelled.load(Ordering::SeqCst) }

    pub(super) async fn cancelled(&self) {
        loop {
            let wake = self.wake.notified();
            if self.is_cancelled() { return; }
            wake.await;
        }
    }

    pub(super) async fn drain(&self) -> Result<(), RecipientWorkerError> {
        self.request_stop();
        let worker = self.worker.lock().map_err(|_| RecipientWorkerError::Unavailable)?.clone();
        if let Some(worker) = worker { worker.shutdown_and_drain().await?; }
        Ok(())
    }

    pub(super) async fn verify_ready(&self) -> Result<(), RecipientWorkerError> {
        let worker = self.worker.lock().map_err(|_| RecipientWorkerError::Unavailable)?.clone()
            .ok_or(RecipientWorkerError::Unavailable)?;
        worker.wait_ready().await
    }
}

pub(super) struct RecipientRunCancellation(pub(super) Arc<RecipientServerLifecycle>);
impl Drop for RecipientRunCancellation {
    fn drop(&mut self) { self.0.request_stop(); }
}

impl Drop for ReverseOnionRecipientWorker {
    fn drop(&mut self) {
        self.stop.closed.store(true, Ordering::SeqCst);
        self.stop.wake.notify_one();
        // Do not abort: the owned task still retains journal/adapter and drains.
    }
}

fn worker_now() -> Result<u64, RecipientWorkerError> {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).map_err(|_| RecipientWorkerError::Unavailable)
}

async fn recipient_db<T: Send + 'static>(journal: &Arc<ReverseOnionRecipientJournal>,
    action: impl FnOnce(&ReverseOnionRecipientJournal, u64) -> Result<T, RecipientJournalError> + Send + 'static,
) -> Result<T, RecipientWorkerError> {
    let journal = Arc::clone(journal);
    tokio::task::spawn_blocking(move || action(&journal, worker_now()?)
        .map_err(|_| RecipientWorkerError::Unavailable))
        .await.map_err(|_| RecipientWorkerError::Ambiguous)?
}

async fn recipient_loop(carrier: &ReverseOnionHttpCarrier, journal: &Arc<ReverseOnionRecipientJournal>,
    adapter: &Arc<ReverseOnionTerminalAdapter>, identity: &Arc<IdentityKeyPair>,
    stop: &WorkerStop, delay: Duration) -> Result<(), RecipientWorkerError> {
    let mut after = None;
    let mut pass_has_work = false;
    while !stop.closed.load(Ordering::SeqCst) {
        recipient_db(journal, |j, now| j.cleanup(RECIPIENT_PAGE, now)).await?;
        let page = recipient_db(journal, move |j, now| j.resume(after, RECIPIENT_PAGE, now)).await?;
        pass_has_work |= !page.items.is_empty();
        after = page.next_after;
        for item in page.items {
            if stop.closed.load(Ordering::SeqCst) { break; }
            match item {
                RecipientRecovery::Poll { exact_bytes, .. } => {
                    exchange_poll(carrier, journal, exact_bytes).await?;
                }
                RecipientRecovery::LeaseReady { claim_id } => {
                    let armed = recipient_db(journal, move |j, now| j.arm(claim_id, now)).await?;
                    // Stop racing after durable arm leaves ambiguous, never undo.
                    if stop.closed.load(Ordering::SeqCst) { break; }
                    if let Ok(result) = adapter.dispatch(armed).await {
                        let bytes = recipient_db(journal, move |j, now| {
                            j.record_result(claim_id, &result, now)?;
                            Ok(result.encode())
                        }).await?;
                        if !stop.closed.load(Ordering::SeqCst) {
                            exchange_result(carrier, bytes).await?;
                        }
                    }
                }
                RecipientRecovery::Result { exact_bytes, .. } => { exchange_result(carrier, exact_bytes).await?; }
                RecipientRecovery::Ambiguous { .. } => {} // Never re-execute.
            }
        }
        if after.is_none() {
            // Historical Poll remains visible through its full evidence
            // horizon, not merely Claim freshness. Only a COMPLETE pass with
            // no unresolved jobs permits new polling, not old-route reissue.
            if can_start_poll(after, pass_has_work, stop.closed.load(Ordering::SeqCst)) {
                let identity = Arc::clone(identity);
                let relay = carrier.relay;
                let bytes = recipient_db(journal, move |j, now| {
                    let mut id = [0; 16];
                    rand::rngs::OsRng.try_fill_bytes(&mut id).map_err(|_| RecipientJournalError::Unavailable)?;
                    if id == [0; 16] { return Err(RecipientJournalError::Unavailable); }
                    let expiry = now.checked_add(aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_CLAIM_LIFETIME_SECS)
                        .ok_or(RecipientJournalError::Rejected)?;
                    let claim = ReverseOnionFrameV1::claim(relay, id, now, expiry, &identity)
                        .map_err(|_| RecipientJournalError::Rejected)?;
                    j.prepare_poll(&claim, now)
                }).await?;
                if !stop.closed.load(Ordering::SeqCst) { exchange_poll(carrier, journal, bytes).await?; }
            }
            pass_has_work = false;
        }
        // Cancellation only interrupts idle waiting, never effectful futures.
        if !stop.closed.load(Ordering::SeqCst) {
            tokio::select! { _ = tokio::time::sleep(delay) => {}, _ = stop.wake.notified() => {} }
        }
    }
    Ok(())
}

fn can_start_poll(next_after: Option<[u8; 16]>, pass_has_work: bool, stopped: bool) -> bool {
    next_after.is_none() && !pass_has_work && !stopped
}

// [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] Only canonical,
// locally persisted frames can reach carrier; no fresh timestamp/re-signing.
fn persisted_frame(bytes: &[u8], kind: ReverseOnionKindV1, now: u64)
    -> Result<ReverseOnionFrameV1, RecipientWorkerError> {
    let frame = if kind == ReverseOnionKindV1::Claim {
        let claim = ReverseOnionFrameV1::decode_for_recovery(bytes).map_err(|_| RecipientWorkerError::Rejected)?;
        claim.verify_claim(claim.relay(), claim.immediate_recipient(), claim.issued_at())
            .map_err(|_| RecipientWorkerError::Rejected)?;
        let horizon = claim.expires_at().checked_add(
            aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_ENVELOPE_LIFETIME_SECS
            + aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_RESULT_RETENTION_SECS)
            .ok_or(RecipientWorkerError::Rejected)?;
        if now < claim.issued_at() || now >= horizon { return Err(RecipientWorkerError::Rejected); }
        claim
    } else {
        ReverseOnionFrameV1::decode(bytes, now).map_err(|_| RecipientWorkerError::Rejected)?
    };
    if frame.kind() != kind || frame.encode() != bytes { return Err(RecipientWorkerError::Rejected); }
    Ok(frame)
}

async fn exchange_poll(carrier: &ReverseOnionHttpCarrier, journal: &Arc<ReverseOnionRecipientJournal>, bytes: Vec<u8>)
    -> Result<(), RecipientWorkerError> {
    let decoded = tokio::task::spawn_blocking(move || persisted_frame(&bytes, ReverseOnionKindV1::Claim, worker_now()?))
        .await.map_err(|_| RecipientWorkerError::Ambiguous)?;
    let Ok(claim) = decoded else { return Ok(()); }; // Beyond evidence horizon: no network.
    let response = carrier.exchange(&claim).await.map_err(|_| RecipientWorkerError::Rejected)?;
    let ReverseOnionExchange::Response { status: 200, body } = response else { return Ok(()); };
    let relay = carrier.relay; let recipient = carrier.recipient;
    recipient_db(journal, move |j, now| {
        let Ok(lease) = ReverseOnionFrameV1::decode(&body, now) else { return Ok(()); };
        if lease.encode() != body { return Ok(()); }
        let Ok(proof) = lease.verify_recipient_lease(&claim, relay, recipient, now) else { return Ok(()); };
        j.record_relay_lease(proof, now)
    }).await
}

async fn exchange_result(carrier: &ReverseOnionHttpCarrier, bytes: Vec<u8>) -> Result<(), RecipientWorkerError> {
    let decoded = tokio::task::spawn_blocking(move || persisted_frame(&bytes, ReverseOnionKindV1::Result, worker_now()?))
        .await.map_err(|_| RecipientWorkerError::Ambiguous)?;
    let Ok(result) = decoded else { return Ok(()); };
    let response = carrier.exchange(&result).await.map_err(|_| RecipientWorkerError::Rejected)?;
    if let ReverseOnionExchange::Response { status: 200, body } = response {
        // Exact stored Result echo is custody transport evidence only. Even a
        // valid echo cannot delete journal evidence or attest business success.
        let _exact_echo = tokio::task::spawn_blocking(move || {
            ReverseOnionFrameV1::decode_for_recovery(&body)
                .is_ok_and(|echo| echo.encode() == body && result.require_exact_retry(&echo).is_ok())
        }).await.map_err(|_| RecipientWorkerError::Ambiguous)?;
    }
    Ok(())
}

#[cfg(test)]
mod recipient_worker_tests {
    use super::*;
    const NOW: u64 = 1_800_000_000;

    // [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Cancelling a startup
    // waiter closes intake but leaves the owned dependency stack to drain.
    #[tokio::test]
    async fn startup_waiter_cancellation_keeps_owned_stack_until_drain() {
        let owner = Arc::new(RecipientServerLifecycle::new());
        let held = Arc::new(AtomicBool::new(true));
        struct Dependency(Arc<AtomicBool>);
        impl Drop for Dependency { fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); } }
        let (opened_tx, opened_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let task_owner = Arc::clone(&owner);
        let dependency = Dependency(Arc::clone(&held));
        let task = tokio::spawn(async move {
            let _dependency = dependency;
            let _ = opened_tx.send(());
            task_owner.cancelled().await;
            release_rx.await.unwrap();
            task_owner.drain().await.unwrap();
        });
        opened_rx.await.unwrap();
        drop(RecipientRunCancellation(Arc::clone(&owner)));
        assert!(owner.is_cancelled());
        assert!(held.load(Ordering::SeqCst));
        release_tx.send(()).unwrap();
        task.await.unwrap();
        assert!(!held.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn lifecycle_without_opened_worker_cannot_claim_ready() {
        let owner = RecipientServerLifecycle::new();
        assert_eq!(owner.verify_ready().await, Err(RecipientWorkerError::Unavailable));
        owner.request_stop();
        owner.drain().await.unwrap();
        assert!(owner.is_cancelled());
    }

    // [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] Authored only;
    // no sockets and no manufactured receipt can establish business success.
    #[test]
    fn historical_claim_is_exact_and_bounded_by_full_evidence_horizon() {
        let relay = IdentityKeyPair::from_bytes(&[41; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [1; 16],
            NOW, NOW + 30, &recipient).unwrap();
        let bytes = claim.encode();
        for now in [NOW, NOW + 31, NOW + 929] {
            let restored = persisted_frame(&bytes, ReverseOnionKindV1::Claim, now).unwrap();
            assert_eq!(restored.encode(), bytes);
            assert_eq!(restored.relay(), relay.public_key_bytes());
            assert_eq!(restored.immediate_recipient(), recipient.public_key_bytes());
        }
        assert!(persisted_frame(&bytes, ReverseOnionKindV1::Claim, NOW + 930).is_err());
        assert!(persisted_frame(&bytes, ReverseOnionKindV1::Claim, NOW - 1).is_err());
        assert!(persisted_frame(&bytes, ReverseOnionKindV1::Result, NOW).is_err());
        let mut tampered = bytes.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(persisted_frame(&tampered, ReverseOnionKindV1::Claim, NOW + 31).is_err());
        let mut trailing = bytes; trailing.push(0);
        assert!(persisted_frame(&trailing, ReverseOnionKindV1::Claim, NOW).is_err());
    }

    #[test]
    fn pagination_and_unresolved_work_never_authorize_replacement_poll() {
        assert!(!can_start_poll(Some([1; 16]), false, false));
        assert!(!can_start_poll(None, true, false));
        assert!(!can_start_poll(None, false, true));
        assert!(can_start_poll(None, false, false));
        // Empty later page cannot erase unresolved work found on an earlier one.
        let mut pass_has_work = true;
        pass_has_work |= false;
        assert!(!can_start_poll(None, pass_has_work, false));
    }

    #[test]
    fn disabled_carrier_rejects_before_runtime_task_or_database_creation() {
        assert!(matches!(ReverseOnionHttpCarrier::new(&ReverseOnionConfig::default(), [42; 32]),
            Err(ReverseOnionPreflightError::Disabled)));
    }

    #[tokio::test]
    async fn cancelled_shutdown_retains_handle_and_waits_for_blocking_completion() {
        let stop = Arc::new(WorkerStop { closed: AtomicBool::new(false), wake: Notify::new() });
        let (ready_tx, ready) = watch::channel(WorkerReady::Ready);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let completed = Arc::new(AtomicBool::new(false));
        let task_completed = Arc::clone(&completed);
        let task = tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                let _ = started_tx.send(());
                release_rx.recv().unwrap();
                task_completed.store(true, Ordering::SeqCst);
            }).await.map_err(|_| RecipientWorkerError::Ambiguous)?;
            Ok(())
        });
        let worker = ReverseOnionRecipientWorker { stop, ready,
            task: Mutex::new(Some(task)), drain_waiter: tokio::sync::Mutex::new(()),
            finished: Mutex::new(None) };
        worker.wait_ready().await.unwrap();
        started_rx.await.unwrap();
        // A critical-task observer must not stop intake just by waiting.
        {
            let completion = worker.wait_completion();
            tokio::pin!(completion);
            assert!(futures::poll!(&mut completion).is_pending());
            assert!(!worker.stop.closed.load(Ordering::SeqCst));
        }
        {
            let shutdown = worker.shutdown_and_drain();
            tokio::pin!(shutdown);
            assert!(futures::poll!(&mut shutdown).is_pending());
        }
        assert!(worker.stop.closed.load(Ordering::SeqCst));
        assert!(worker.task.lock().unwrap().is_some());
        assert!(!completed.load(Ordering::SeqCst));
        release_tx.send(()).unwrap();
        worker.shutdown_and_drain().await.unwrap();
        assert!(completed.load(Ordering::SeqCst));
        assert!(worker.task.lock().unwrap().is_none());
        worker.shutdown_and_drain().await.unwrap();
        drop(ready_tx);
    }
}
