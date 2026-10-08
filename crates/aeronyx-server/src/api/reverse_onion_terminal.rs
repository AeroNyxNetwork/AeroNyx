// ============================================
// File: crates/aeronyx-server/src/api/reverse_onion_terminal.rs
// ============================================
//! In-process, terminal-only adapter for a durably Armed reverse delivery.
//!
//! [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] The supplied Router must be a
//! clone of the existing production peer router, with its original middleware,
//! shared limits, previous-hop admission and durable replay store intact. Never
//! construct a fresh router per request or call private handler helpers here.
//! The caller must await journal.arm first. This adapter persists the exact
//! Result inside its tracked operation before publishing completion, including
//! when the original waiter times out. Returning a Result authenticates opaque
//! custody, not source execution proof.
//! [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Runtime owners must
//! close admission and await `shutdown_and_drain` BEFORE dropping this adapter
//! or its Tokio runtime. A wait timeout never aborts the tracked operation.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::encode_blind_relay_envelope;
use aeronyx_core::protocol::onion::{is_onion_blob, try_open_onion_layer};
use aeronyx_core::protocol::onion::reverse_delivery::ReverseOnionFrameV1;
use aeronyx_core::protocol::onion_reply::{
    decode_onion_reply_request, decode_onion_sealed_response, OnionReplyProofMode,
    MAX_ONION_SEALED_RESPONSE_BASE64_BYTES,
};
use aeronyx_core::protocol::OnionRoutePurpose;
use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use axum::Router;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::FutureExt;
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower::ServiceExt;
use zeroize::{Zeroize, Zeroizing};

use super::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};
use crate::services::reverse_onion_recipient::{
    RecipientDispatch, RecipientJournalError, ReverseOnionRecipientJournal,
};

const MAX_LOCAL_DISPATCHES: usize = 4;
// Existing peer router body limit, not an increase to its public admission.
const MAX_REQUEST_JSON_BYTES: usize = 2 * 1024 * 1024;
// Fixed-class opaque base64 plus bounded immediate-hop JSON receipt metadata.
const MAX_RESPONSE_JSON_BYTES: usize = MAX_ONION_SEALED_RESPONSE_BASE64_BYTES + 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ReverseOnionTerminalError {
    #[error("reverse terminal rejected")]
    Rejected,
    // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] This variant is
    // emitted only before `router.oneshot`; it authorizes retrying the exact
    // already-signed lease, never rebuilding or changing its payload.
    #[error("reverse terminal did not enter local dispatch")]
    ZeroDispatch,
    // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] Busy is returned
    // only before the local peer router is entered; callers may retain the
    // exact Lease for a later attempt. Every post-entry uncertainty is Ambiguous.
    #[error("reverse terminal busy")]
    Busy,
    #[error("reverse terminal ambiguous")]
    Ambiguous,
    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Local
    // clock failure is not a recoverable zero-dispatch proof or remote error.
    #[error("reverse terminal clock unavailable")]
    Clock,
}

type Result<T> = std::result::Result<T, ReverseOnionTerminalError>;

// [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Production is
// a zero-sized real-clock source. Tests may inject at most the four owned
// phase samples; no production callback, persisted clock or remote input.
#[derive(Clone, Default)]
struct TerminalClockSource {
    #[cfg(test)]
    samples: Option<Arc<Mutex<std::collections::VecDeque<std::result::Result<u64, ()>>>>>,
}

impl TerminalClockSource {
    fn now(&self) -> Result<u64> {
        #[cfg(test)]
        if let Some(samples) = &self.samples {
            return samples.lock().map_err(|_| ReverseOnionTerminalError::Clock)?
                .pop_front().ok_or(ReverseOnionTerminalError::Clock)?
                .map_err(|_| ReverseOnionTerminalError::Clock);
        }
        now_secs().map_err(|_| ReverseOnionTerminalError::Clock)
    }
}

/// No Debug: identities, frames, response bodies and router state stay private.
pub(crate) struct ReverseOnionTerminalAdapter {
    router: Router,
    identity: Arc<IdentityKeyPair>,
    relay: [u8; 32],
    timeout: Duration,
    permits: Arc<Semaphore>,
    // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] Shared with the
    // worker: stop must fence queued preparation, not only the polling loop.
    intake_stopped: Arc<AtomicBool>,
    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Sticky
    // source-blind fault survives a timed-out/cancelled terminal waiter.
    failure: watch::Sender<bool>,
    tracked: Mutex<TrackedOperations>,
    drain_waiter: tokio::sync::Mutex<()>,
    clock: TerminalClockSource,
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Test-only,
    // bounded phase observation; no production callback or opaque data.
    #[cfg(test)]
    result_ready: Option<tokio::sync::mpsc::Sender<()>>,
    // [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] Tests can hold
    // the real owned operation between result crypto and journal persistence.
    #[cfg(test)]
    result_persistence_gate: Option<Arc<Semaphore>>,
    #[cfg(test)]
    prepared_gate: Option<(tokio::sync::mpsc::Sender<()>, Arc<Semaphore>)>,
}

// [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Live handles stay here
// until real completion, including after a caller's deadline/cancellation.
struct TrackedOperations {
    closed: bool,
    panicked: bool,
    handles: Vec<JoinHandle<()>>,
}

impl TrackedOperations {
    fn reap_finished(&mut self) {
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        let mut index = 0;
        while index < self.handles.len() {
            if self.handles[index].is_finished() {
                if let Poll::Ready(result) = Pin::new(&mut self.handles[index]).poll(&mut cx) {
                    let _completed = self.handles.swap_remove(index);
                    if result.is_err() { self.panicked = true; }
                    continue;
                }
            }
            index += 1;
        }
    }
}

struct PreparedDispatch {
    armed: RecipientDispatch,
    json: Vec<u8>,
    response_class: usize,
    started_at: u64,
    // Owned by blocking work while crypto runs, including after cancellation.
    _permit: OwnedSemaphorePermit,
}

impl ReverseOnionTerminalAdapter {
    /// Construction opens no socket/DB and performs no execution. Registration,
    /// disabled startup, router provenance and shutdown joining belong to runtime.
    pub(crate) fn new(
        peer_router: Router,
        local_identity: Arc<IdentityKeyPair>,
        pinned_relay: [u8; 32],
        timeout: Duration,
    ) -> Result<Self> {
        if pinned_relay == [0; 32] || pinned_relay == local_identity.public_key_bytes()
            || timeout.is_zero() || timeout > Duration::from_secs(30)
        {
            return Err(ReverseOnionTerminalError::Rejected);
        }
        Ok(Self { router: peer_router, identity: local_identity, relay: pinned_relay,
            timeout, permits: Arc::new(Semaphore::new(MAX_LOCAL_DISPATCHES)),
            intake_stopped: Arc::new(AtomicBool::new(false)),
            failure: watch::channel(false).0,
            tracked: Mutex::new(TrackedOperations {
                closed: false, panicked: false, handles: Vec::new(),
            }),
            drain_waiter: tokio::sync::Mutex::new(()),
            clock: TerminalClockSource::default(),
            #[cfg(test)]
            result_ready: None,
            #[cfg(test)]
            result_persistence_gate: None,
            #[cfg(test)]
            prepared_gate: None,
        })
    }

    // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] The composition
    // owner only closes this process-lifetime gate; it never replaces/reopens
    // it during drain. Completion and journal permits remain independently live.
    pub(crate) fn intake_stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.intake_stopped)
    }

    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Bounded
    // deterministic samples exercise the actual tracked/cancel-surviving owner.
    #[cfg(test)]
    pub(crate) fn with_clock_samples(mut self, samples: &[std::result::Result<u64, ()>]) -> Self {
        assert!((1..=4).contains(&samples.len()));
        self.clock.samples = Some(Arc::new(Mutex::new(samples.iter().copied().collect())));
        self
    }

    // [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] Bounded test
    // observation only. Receiving this signal is NOT a persistence ACK;
    // the real worker/adapter drain must finish before reading the journal.
    #[cfg(test)]
    pub(crate) fn with_result_persistence_gate(mut self, observer: tokio::sync::mpsc::Sender<()>,
        gate: Arc<Semaphore>) -> Self {
        self.result_ready = Some(observer);
        self.result_persistence_gate = Some(gate);
        self
    }

    pub(crate) fn request_stop(&self) {
        self.intake_stopped.store(true, Ordering::SeqCst);
    }

    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Normal
    // stop never publishes a fault. Worker idle waits may be cancelled freely;
    // no route data, DB handle or task ownership crosses this signal.
    pub(crate) fn has_failed(&self) -> bool { *self.failure.borrow() }

    pub(crate) async fn wait_for_failure(&self) {
        let mut failure = self.failure.subscribe();
        loop {
            if *failure.borrow_and_update() { return; }
            if failure.changed().await.is_err() { return; }
        }
    }

    /// Consumes one post-Armed projection. Only `ZeroDispatch` proves the
    /// local router was never entered and permits the worker to restore this
    /// exact lease; all post-entry uncertainty remains non-reexecutable.
    /// Caller timeout/cancellation drops only its result receiver, not the
    /// operation. A late verified result is durably stored before task exit.
    pub(crate) async fn dispatch(
        &self,
        armed: RecipientDispatch,
        journal: Arc<ReverseOnionRecipientJournal>,
    ) -> Result<ReverseOnionFrameV1> {
        if self.intake_stopped.load(Ordering::SeqCst) {
            return Err(ReverseOnionTerminalError::ZeroDispatch);
        }
        let permit = Arc::clone(&self.permits).try_acquire_owned()
            .map_err(|_| ReverseOnionTerminalError::Busy)?;
        let claim_id = armed.claim.claim_id();
        let relay = self.relay;
        let local = self.identity.public_key_bytes();
        let router = self.router.clone();
        let identity = Arc::clone(&self.identity);
        let intake_stopped = Arc::clone(&self.intake_stopped);
        let failure = self.failure.clone();
        let panic_stop = Arc::clone(&self.intake_stopped);
        let panic_failure = self.failure.clone();
        let clock = self.clock.clone();
        #[cfg(test)]
        let result_ready = self.result_ready.clone();
        #[cfg(test)]
        let result_persistence_gate = self.result_persistence_gate.clone();
        #[cfg(test)]
        let prepared_gate = self.prepared_gate.clone();
        let operation = async move {
            let prepare_clock = clock.clone();
            let prepared = tokio::task::spawn_blocking(move || prepare(armed, relay, local, permit, &prepare_clock))
                .await.map_err(|_| ReverseOnionTerminalError::ZeroDispatch)?
                .map_err(|error| if error == ReverseOnionTerminalError::Clock { error }
                    else { ReverseOnionTerminalError::ZeroDispatch })?;
            let PreparedDispatch { armed, json, response_class, started_at, _permit } = prepared;
            let request = Request::builder().method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json)).map_err(|_| ReverseOnionTerminalError::ZeroDispatch)?;

            // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] A bounded
            // test-only pause observes the real post-crypto entry boundary.
            #[cfg(test)]
            if let Some((ready, gate)) = prepared_gate {
                let _ = ready.try_send(());
                let _released = gate.acquire_owned().await
                    .map_err(|_| ReverseOnionTerminalError::ZeroDispatch)?;
            }
            // Final entry authorization linearizes here, after all queued
            // preparation. A prior stop proves zero dispatch; a later stop
            // drains this admitted operation and may not discard its result.
            if intake_stopped.load(Ordering::SeqCst) {
                return Err(ReverseOnionTerminalError::ZeroDispatch);
            }
            // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Entry
            // time belongs here, after request construction and queued gates.
            // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex]
            // Keep this newer entry floor through response crypto and storage.
            let entered_at = require_terminal_time(started_at, clock.now())?;
            require_fresh_entry(started_at, armed.lease.expires_at(), entered_at)?;

            // [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] Actual middleware
            // path, not HTTP loopback, fake auth extensions, or direct execution.
            // The preflight rejected forwarding onions, so this route cannot
            // legitimately choose an onward peer for this immutable envelope.
            let response = router.oneshot(request).await
                .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
            if response.status() != StatusCode::OK
                || response.headers().get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok()) != Some("application/json")
            {
                return Err(ReverseOnionTerminalError::Ambiguous);
            }
            let body = to_bytes(response.into_body(), MAX_RESPONSE_JSON_BYTES).await
                .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
            // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Result
            // crypto returns ownership of the execution permit. It must cover
            // the later DB-lane wait and durability fence, not just routing.
            let finish_clock = clock.clone();
            let (result, permit) = tokio::task::spawn_blocking(move || {
                let finished_at = require_terminal_time(entered_at, finish_clock.now())?;
                let response: PeerBlindRelayResponse = serde_json::from_slice(&body)
                    .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
                let sealed = verify_response(&armed, &response, local, response_class, entered_at, finished_at)?;
                let result = ReverseOnionFrameV1::result(&armed.claim, &armed.lease, &sealed,
                    armed.route_deadline, finished_at, identity.as_ref())
                    .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
                Ok::<_, ReverseOnionTerminalError>((result, _permit))
            }).await.map_err(|_| {
                report_terminal_failure(&intake_stopped, &failure);
                ReverseOnionTerminalError::Ambiguous
            })??;
            #[cfg(test)]
            if let Some(ready) = result_ready { let _ = ready.try_send(()); }
            #[cfg(test)]
            if let Some(gate) = result_persistence_gate {
                let _released = gate.acquire_owned().await
                    .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
            }
            let persisted = persist_exact_result(journal, claim_id, result, clock).await;
            // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex]
            // No successful Result escaped the fence. Stop new work, retain
            // the existing durable state and let authenticated restart audit
            // decide whether exact Result bytes or ambiguity survived.
            if persisted.as_ref().err().is_some_and(|error| result_persistence_is_fatal(*error)) {
                report_terminal_failure(&intake_stopped, &failure);
            }
            drop(permit);
            persisted.map_err(|_| ReverseOnionTerminalError::Ambiguous)
        };
        // [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Register under
        // the same lock as admission closure. No live handle is detached or
        // removed by timeout; the permit belongs to operation, not its waiter.
        let receive = {
            let mut tracked = self.tracked.lock().map_err(|_| ReverseOnionTerminalError::Rejected)?;
            if tracked.closed || self.intake_stopped.load(Ordering::SeqCst) {
                return Err(ReverseOnionTerminalError::ZeroDispatch);
            }
            // Completed tasks own no continuing work; pending handles stay put.
            tracked.reap_finished();
            if tracked.panicked {
                tracked.closed = true;
                report_terminal_failure(&self.intake_stopped, &self.failure);
                return Err(ReverseOnionTerminalError::Ambiguous);
            }
            if tracked.handles.len() >= MAX_LOCAL_DISPATCHES {
                return Err(ReverseOnionTerminalError::Busy);
            }
            let (send, receive) = oneshot::channel();
            tracked.handles.push(tokio::spawn(async move {
                // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex]
                // An owned unwind must wake the worker even when the original
                // dispatch receiver no longer exists. Never restore a lease.
                let result = match std::panic::AssertUnwindSafe(operation).catch_unwind().await {
                    Ok(result) => {
                        // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex]
                        // This owner outlives its waiter. Even a pre-entry clock
                        // regression closes intake and must not restore a lease.
                        if matches!(&result, Err(ReverseOnionTerminalError::Clock)) {
                            report_terminal_failure(&panic_stop, &panic_failure);
                        }
                        result
                    }
                    Err(_) => {
                        report_terminal_failure(&panic_stop, &panic_failure);
                        Err(ReverseOnionTerminalError::Ambiguous)
                    }
                };
                let _ = send.send(result);
            }));
            receive
        };
        tokio::time::timeout(self.timeout, receive).await
            .map_err(|_| ReverseOnionTerminalError::Ambiguous)?
            .map_err(|_| ReverseOnionTerminalError::Ambiguous)?
    }

    /// Stop admission and join real completion, without aborting effects.
    /// Cancellation of this wait leaves every unfinished handle in the registry;
    /// retain the adapter and call again. There is intentionally no forced-drop
    /// or abort fallback: an indefinitely blocked router means drain is pending.
    pub(crate) async fn shutdown_and_drain(&self) -> Result<()> {
        self.request_stop();
        {
            let mut tracked = self.tracked.lock().map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
            tracked.closed = true;
        }
        // One join poller at a time: concurrent drains cannot overwrite one
        // another's JoinHandle wakers and leave an earlier waiter asleep.
        let _waiter = self.drain_waiter.lock().await;
        futures::future::poll_fn(|cx| {
            let mut tracked = match self.tracked.lock() {
                Ok(tracked) => tracked,
                Err(_) => return Poll::Ready(Err(ReverseOnionTerminalError::Ambiguous)),
            };
            let mut index = 0;
            while index < tracked.handles.len() {
                match Pin::new(&mut tracked.handles[index]).poll(cx) {
                    Poll::Ready(result) => {
                        let _completed = tracked.handles.swap_remove(index);
                        if result.is_err() {
                            tracked.panicked = true;
                            report_terminal_failure(&self.intake_stopped, &self.failure);
                        }
                    }
                    Poll::Pending => index += 1,
                }
            }
            if tracked.handles.is_empty() {
                Poll::Ready(if tracked.panicked || self.has_failed() {
                    Err(ReverseOnionTerminalError::Ambiguous)
                } else { Ok(()) })
            } else {
                Poll::Pending
            }
        }).await
    }
}

// [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Shared by
// the owned operation and its unwind boundary, without retaining the adapter
// (which owns that same task handle). Publish only after closing intake.
fn report_terminal_failure(stopped: &AtomicBool, failure: &watch::Sender<bool>) {
    stopped.store(true, Ordering::SeqCst);
    failure.send_replace(true);
}

// [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Preserve
// per-job uncertainty/backpressure without promoting expiry or rejection to
// a process fault. A failed/uncertain DB owner cannot safely keep polling.
fn result_persistence_is_fatal(error: RecipientJournalError) -> bool {
    matches!(error, RecipientJournalError::Corrupt | RecipientJournalError::Unavailable
        | RecipientJournalError::Ambiguous)
}

// [REVERSE-ONION-RESULT-DURABILITY 2026-10-05 by Codex] This runs within the
// tracked operation, not its timeout-bound waiter, so a late local-router
// completion still commits exact retry bytes before the operation is reaped.
// [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Result creation
// is a later trusted sample than entry. Carry it through the journal lane/SQL
// wait; failure retains Armed and uses the existing fatal persistence signal.
async fn persist_exact_result(
    journal: Arc<ReverseOnionRecipientJournal>,
    claim_id: [u8; 16],
    result: ReverseOnionFrameV1,
    clock: TerminalClockSource,
) -> std::result::Result<ReverseOnionFrameV1, RecipientJournalError> {
    let finished_at = result.issued_at();
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] This await is
    // inside the tracked operation, never the caller's timeout. Do not drop
    // a finished result just because the worker is using the same journal.
    journal.run_blocking(move |journal| {
        // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] Sampling inside
        // the journal also covers SQLite waits after this owned lane starts.
        journal.record_result_at(claim_id, &result, || {
            require_terminal_time(finished_at, clock.now())
                .map_err(|_| RecipientJournalError::Unavailable)
        })?;
        Ok(result)
    })
    .await
}

// [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] Preflight performs no workload
// or storage effect and rejects a forward hop before router dispatch. It reads
// the existing process onion-key manager; normal startup must initialize that
// manager. Existing router admission still runs independently afterward.
fn prepare(armed: RecipientDispatch, relay: [u8; 32], local: [u8; 32], permit: OwnedSemaphorePermit,
    clock: &TerminalClockSource)
    -> Result<PreparedDispatch> {
    // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] Protocol/capacity
    // failures in preparation prove zero router entry and permit only an exact
    // lease retry. Local clock failure instead closes the owned worker.
    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] A rollback
    // after the durable Armed commit is local failure, never safe retry proof.
    let now = require_terminal_time(armed.armed_at, clock.now())?;
    armed.claim.verify_claim(relay, local, armed.claim.issued_at())
        .map_err(|_| ReverseOnionTerminalError::Rejected)?;
    let envelope = armed.lease.verify_lease(&armed.claim, armed.route_deadline, now)
        .map_err(|_| ReverseOnionTerminalError::Rejected)?;
    if !is_onion_blob(&envelope.encrypted_blob)
        || encode_blind_relay_envelope(&envelope).map_err(|_| ReverseOnionTerminalError::Rejected)?
            != encode_blind_relay_envelope(&armed.envelope).map_err(|_| ReverseOnionTerminalError::Rejected)?
    {
        return Err(ReverseOnionTerminalError::Rejected);
    }
    let secrets = crate::services::onion_keys::peel_secrets(now);
    let peel = try_open_onion_layer(&envelope.encrypted_blob, &secrets)
        .map_err(|_| ReverseOnionTerminalError::Rejected)?;
    let inner = Zeroizing::new(peel.inner);
    if peel.next_hop.is_some() { return Err(ReverseOnionTerminalError::Rejected); }
    // Only the existing fixed-class source-sealed reply workload is admitted.
    // Anonymous mailbox AMST/AMSR uses a different compact carrier; reject it
    // before effects rather than pretending it is an OnionReply response.
    let mut reply = decode_onion_reply_request(&inner)
        .map_err(|_| ReverseOnionTerminalError::Rejected)?;
    let proof_mode = reply.proof_mode();
    let response_class = reply.response_size_class as usize;
    // [PHALA-TERMINAL-CARRIER-REPAIR 2026-10-08 by Codex] The shared
    // preparer decodes the entire reply carrier, not the already-unwrapped frame.
    let supported_payload = validate_private_pull_payload(&inner);
    reply.payload.zeroize();
    if proof_mode != OnionReplyProofMode::SourceSealedTerminalProof {
        return Err(ReverseOnionTerminalError::Rejected);
    }
    supported_payload?;
    let request = PeerBlindRelayRequest {
        envelope, previous_hop_node_id: relay,
        onward_envelope: None, onward_descriptor_hint: None,
    };
    let json = serde_json::to_vec(&request).map_err(|_| ReverseOnionTerminalError::Rejected)?;
    if json.len() > MAX_REQUEST_JSON_BYTES { return Err(ReverseOnionTerminalError::Rejected); }
    Ok(PreparedDispatch { armed, json, response_class, started_at: now, _permit: permit })
}

// [REVERSE-ONION-PULL-ONLY 2026-10-06 by Codex] The private route grant is
// specifically BlindVaultPull. Do not let the shared peer router turn that
// queue capability into source-selected write/admin operations.
fn validate_private_pull_payload(encoded_request: &[u8]) -> Result<()> {
    let prepared = super::chat_peer_terminal_reply::prepare_blind_vault_inline_reply(encoded_request)
        .map_err(|_| ReverseOnionTerminalError::Rejected)?;
    if prepared.purpose() != OnionRoutePurpose::BlindVaultPull
        || prepared.effect().requires_durable_guard()
    {
        return Err(ReverseOnionTerminalError::Rejected);
    }
    Ok(())
}

fn verify_response(
    armed: &RecipientDispatch,
    response: &PeerBlindRelayResponse,
    local: [u8; 32],
    expected_class: usize,
    started_at: u64,
    finished_at: u64,
) -> Result<Vec<u8>> {
    if !response.accepted || !response.terminal || response.forwarded
        || response.ttl_remaining != armed.envelope.ttl
        || response.reason.as_deref() != Some("onion_terminal_delivered")
        || response.delivery_receipt.is_some() || response.failure_receipt.is_some()
    {
        return Err(ReverseOnionTerminalError::Ambiguous);
    }
    let opaque = response.opaque_terminal_response_b64.as_deref()
        .ok_or(ReverseOnionTerminalError::Ambiguous)?;
    if opaque.len() > MAX_ONION_SEALED_RESPONSE_BASE64_BYTES {
        return Err(ReverseOnionTerminalError::Ambiguous);
    }
    let receipt = response.success_receipt.as_ref().ok_or(ReverseOnionTerminalError::Ambiguous)?;
    // The existing durable replay layer may return an earlier exact receipt.
    // Its timestamp need not be this adapter invocation's start; binding and
    // envelope freshness remain authoritative, without authorizing reexecution.
    if finished_at < started_at || receipt.accepted_at < armed.envelope.timestamp
        || receipt.accepted_at > finished_at
    {
        return Err(ReverseOnionTerminalError::Ambiguous);
    }
    receipt.verify_expected(&armed.envelope, true, false, response.ttl_remaining,
        response.reason.as_deref(), None, Some(opaque.as_bytes()), &local)
        .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
    let decoded = STANDARD.decode(opaque).map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
    if STANDARD.encode(&decoded) != opaque { return Err(ReverseOnionTerminalError::Ambiguous); }
    let sealed = decode_onion_sealed_response(&decoded).map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
    if sealed.validate().map_err(|_| ReverseOnionTerminalError::Ambiguous)? != expected_class {
        return Err(ReverseOnionTerminalError::Ambiguous);
    }
    Ok(decoded)
}

fn now_secs() -> Result<u64> {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs())
        .map_err(|_| ReverseOnionTerminalError::Rejected)
}

// [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Each owned
// phase may advance its predecessor's clock but may never erase its floor.
fn require_terminal_time(floor: u64, sample: Result<u64>) -> Result<u64> {
    let now = sample.map_err(|_| ReverseOnionTerminalError::Clock)?;
    if now < floor { Err(ReverseOnionTerminalError::Clock) } else { Ok(now) }
}

// [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] This is a
// zero-dispatch check only; it cannot authorize restoring an expired lease.
fn require_fresh_entry(started_at: u64, execution_deadline: u64, now: u64) -> Result<()> {
    require_terminal_time(started_at, Ok(now))?;
    if now >= execution_deadline {
        Err(ReverseOnionTerminalError::ZeroDispatch)
    } else {
        Ok(())
    }
}

// [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] Authored, unexecuted pure
// response-boundary cases. They do NOT claim middleware or journal integration.
#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::protocol::chat::BlindRelaySuccessReceipt;
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion_reply::{
        encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
        ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };
    use aeronyx_core::protocol::{
        encode_blind_vault_frame, BlindVaultFrame, BlindVaultLeaseStatusRequest,
        BlindVaultPullRequest, BlindVaultPutRequest, BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES,
        BLIND_VAULT_PROTOCOL_VERSION,
    };
    use crate::services::reverse_onion_recipient::{
        RecipientJournalLimits, RecipientRecovery,
    };

    const NOW: u64 = 1_800_000_000;

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authored,
    // not run: this is the exact final-entry guard, not a duplicate assertion.
    #[test]
    fn final_router_entry_requires_the_original_live_execution_window() {
        for now in [NOW, NOW + 9] {
            assert_eq!(require_fresh_entry(NOW, NOW + 10, now), Ok(()));
        }
        for now in [NOW + 10, NOW + 11] {
            assert_eq!(require_fresh_entry(NOW, NOW + 10, now),
                Err(ReverseOnionTerminalError::ZeroDispatch));
        }
        assert_eq!(require_fresh_entry(NOW, NOW + 10, NOW - 1),
            Err(ReverseOnionTerminalError::Clock));
    }

    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Authored,
    // not run: completion may exceed preparation yet regress below real entry.
    #[test]
    fn terminal_phases_cannot_erase_a_newer_clock_floor() {
        let prepared = require_terminal_time(NOW, Ok(NOW + 1)).unwrap();
        let entered = require_terminal_time(prepared, Ok(NOW + 20)).unwrap();
        assert!(NOW + 10 >= prepared, "calibrate the previous weaker check");
        assert_eq!(require_terminal_time(entered, Ok(NOW + 10)), Err(ReverseOnionTerminalError::Clock));
        assert_eq!(require_terminal_time(entered, Err(ReverseOnionTerminalError::Rejected)), Err(ReverseOnionTerminalError::Clock));
        assert_eq!(require_terminal_time(entered, Ok(entered)), Ok(entered));
    }

    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
    // not run: protocol expiry/capacity must not become a remote shutdown lever.
    #[test]
    fn only_local_persistence_owner_faults_fail_recipient_health() {
        for error in [RecipientJournalError::Corrupt, RecipientJournalError::Unavailable,
            RecipientJournalError::Ambiguous] {
            assert!(result_persistence_is_fatal(error));
        }
        for error in [RecipientJournalError::Rejected, RecipientJournalError::Busy,
            RecipientJournalError::Conflict, RecipientJournalError::Capacity, RecipientJournalError::Expired,
            RecipientJournalError::IntakeClosed] {
            assert!(!result_persistence_is_fatal(error));
        }
    }

    // [REVERSE-ONION-PULL-ONLY 2026-10-06 by Codex] Exercise accepted and
    // rejected workload classes before any local-router entry.
    #[test]
    fn private_queue_adapter_accepts_only_read_only_pull_work() {
        // [PHALA-PULL-FIXTURE-REPAIR 2026-10-08 by Codex] The adapter
        // receives the decrypted reply-request wrapper, not a bare workload.
        let wrap = |frame: BlindVaultFrame| {
            let identity = IdentityKeyPair::from_bytes(&[52; 32]).unwrap();
            let (request, _session) = OnionReplySession::prepare_source_sealed(
                [3; 16], identity.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
                encode_blind_vault_frame(&frame).unwrap(),
            ).unwrap();
            aeronyx_core::protocol::onion_reply::encode_onion_reply_request(&request).unwrap()
        };
        let pull = BlindVaultFrame::PullRequest(BlindVaultPullRequest {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: [1; 32],
            read_capability: [2; 32],
            continuation_cursor: Vec::new(),
            limit: 1,
        });
        let pull = wrap(pull);
        assert!(validate_private_pull_payload(&pull).is_ok());

        let status = BlindVaultFrame::LeaseStatus(BlindVaultLeaseStatusRequest::new(
            [3; 32], [4; 16], NOW * 1_000,
        ));
        let status = wrap(status);
        assert_eq!(validate_private_pull_payload(&status).err(),
            Some(ReverseOnionTerminalError::Rejected));

        let put = BlindVaultFrame::Put(BlindVaultPutRequest::new(
            [5; 32], [6; 32], [7; 16],
            vec![0; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0]], NOW * 1_000 + 60_000,
        ));
        let put = wrap(put);
        assert_eq!(validate_private_pull_payload(&put).err(),
            Some(ReverseOnionTerminalError::Rejected));

        let multi_page_pull = BlindVaultFrame::PullRequest(BlindVaultPullRequest {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: [8; 32],
            read_capability: [9; 32],
            continuation_cursor: Vec::new(),
            limit: 2,
        });
        let multi_page_pull = wrap(multi_page_pull);
        assert_eq!(validate_private_pull_payload(&multi_page_pull).err(),
            Some(ReverseOnionTerminalError::Rejected));
    }

    fn fixture() -> (RecipientDispatch, PeerBlindRelayResponse, IdentityKeyPair) {
        let relay = IdentityKeyPair::from_bytes(&[51; 32]).unwrap();
        let local = IdentityKeyPair::from_bytes(&[52; 32]).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &local).unwrap();
        let (_, kem) = local.to_x25519();
        let envelope = build_onion_envelope(
            &[OnionHop { node_id: local.public_key_bytes(), kem_pub: kem.to_bytes() }],
            b"opaque fixture", [2; 16], 1, NOW, &relay,
        ).unwrap();
        let lease = ReverseOnionFrameV1::lease(&claim, &envelope, [3; 16], NOW + 600, NOW, &relay).unwrap();
        let (request, _session) = OnionReplySession::prepare_source_sealed(
            envelope.route_id, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            b"operation".to_vec(),
        ).unwrap();
        let sealed = seal_onion_reply(envelope.route_id, &request, b"opaque result", &local).unwrap();
        let opaque = STANDARD.encode(encode_onion_sealed_response(&sealed).unwrap());
        let receipt = BlindRelaySuccessReceipt::terminal(&envelope, envelope.ttl,
            Some("onion_terminal_delivered"), None, Some(opaque.as_bytes()), NOW, &local);
        let response = PeerBlindRelayResponse {
            accepted: true, terminal: true, forwarded: false, ttl_remaining: envelope.ttl,
            reason: Some("onion_terminal_delivered".into()), delivery_receipt: None,
            success_receipt: Some(receipt), failure_receipt: None,
            opaque_terminal_response_b64: Some(opaque),
        };
        (RecipientDispatch {
            envelope, claim, lease, route_deadline: NOW + 600,
            route_origin_commitment: Some([9; 32]),
            armed_at: NOW,
        }, response, local)
    }

    #[test]
    fn exact_local_hop_receipt_accepts_retained_reply_without_claiming_execution() {
        let (armed, response, local) = fixture();
        let decoded = verify_response(&armed, &response, local.public_key_bytes(),
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0], NOW + 2, NOW + 3).unwrap();
        let result = ReverseOnionFrameV1::result(&armed.claim, &armed.lease, &decoded,
            armed.route_deadline, NOW + 3, &local).unwrap();
        assert!(result.verify_result(&armed.claim, &armed.lease, armed.route_deadline, NOW + 3).is_ok());
    }

    #[test]
    fn valid_signature_does_not_allow_wrong_request_responder_or_response_class() {
        let (mut armed, response, local) = fixture();
        assert!(verify_response(&armed, &response, [99; 32], ONION_REPLY_RESPONSE_SIZE_CLASSES[0], NOW, NOW + 1).is_err());
        assert!(verify_response(&armed, &response, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[1], NOW, NOW + 1).is_err());
        armed.envelope.route_id[0] ^= 1;
        assert!(verify_response(&armed, &response, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], NOW, NOW + 1).is_err());
    }

    #[test]
    fn forwarded_missing_proof_and_changed_opaque_reply_are_ambiguous() {
        let (armed, response, local) = fixture();
        let verify = |candidate: &PeerBlindRelayResponse| verify_response(&armed, candidate,
            local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], NOW, NOW + 1);
        let mut forwarded = response.clone();
        forwarded.terminal = false;
        forwarded.forwarded = true;
        assert_eq!(verify(&forwarded).err(), Some(ReverseOnionTerminalError::Ambiguous));
        let mut missing = response.clone();
        missing.success_receipt = None;
        assert_eq!(verify(&missing).err(), Some(ReverseOnionTerminalError::Ambiguous));
        let mut changed = response;
        changed.opaque_terminal_response_b64 = Some("AA==".into());
        assert_eq!(verify(&changed).err(), Some(ReverseOnionTerminalError::Ambiguous));
    }

    // [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Authored only.
    // A real Router waits for a real blocking worker, but opens no socket/DB.
    // The release guard prevents a failed assertion from stranding test workers.
    struct BlockingGate {
        open: Mutex<bool>,
        condition: std::sync::Condvar,
    }

    impl BlockingGate {
        fn wait(&self) {
            let mut open = self.open.lock().unwrap();
            while !*open { open = self.condition.wait(open).unwrap(); }
        }

        fn release(&self) {
            *self.open.lock().unwrap() = true;
            self.condition.notify_all();
        }
    }

    struct ReleaseOnDrop(Arc<BlockingGate>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) { self.0.release(); }
    }

    // Keep paused Tokio time from auto-advancing while real blocking workers
    // enter. Once their deterministic entry signals arrive, release this guard
    // and explicitly advance time (or cancel the caller) under test control.
    struct HoldPausedClock(JoinHandle<()>);
    impl HoldPausedClock {
        fn new() -> Self {
            Self(tokio::spawn(async {
                loop { tokio::task::yield_now().await; }
            }))
        }
    }
    impl Drop for HoldPausedClock {
        fn drop(&mut self) { self.0.abort(); }
    }

    fn blocked_router(
        gate: Arc<BlockingGate>,
        entered: tokio::sync::mpsc::UnboundedSender<()>,
        count: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Router {
        Router::new().route("/api/chat/peer/blind-relay", axum::routing::post(move || {
            let gate = Arc::clone(&gate);
            let entered = entered.clone();
            let count = Arc::clone(&count);
            async move {
                tokio::task::spawn_blocking(move || {
                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let _ = entered.send(());
                    gate.wait();
                }).await.unwrap();
                StatusCode::SERVICE_UNAVAILABLE
            }
        }))
    }

    // [REVERSE-ONION-PULL-ONLY 2026-10-06 by Codex] The exercised route now
    // carries the exact bounded task accepted by the production adapter.
    fn live_dispatch(relay: &IdentityKeyPair, local: &IdentityKeyPair, id: u8) -> RecipientDispatch {
        let now = now_secs().unwrap();
        let route = [id; 16];
        let payload = encode_blind_vault_frame(&BlindVaultFrame::PullRequest(
            BlindVaultPullRequest {
                version: BLIND_VAULT_PROTOCOL_VERSION,
                lease_id: [id; 32],
                read_capability: [id.wrapping_add(1); 32],
                continuation_cursor: Vec::new(),
                limit: 1,
            },
        )).unwrap();
        let (request, _session) = OnionReplySession::prepare_source_sealed(
            route, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], payload,
        ).unwrap();
        let inner = aeronyx_core::protocol::onion_reply::encode_onion_reply_request(&request).unwrap();
        let envelope = build_onion_envelope(
            &[OnionHop { node_id: local.public_key_bytes(),
                kem_pub: crate::services::onion_keys::current_public_key() }],
            &inner, route, 1, now, relay,
        ).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), route, now, now + 30, local).unwrap();
        let lease = ReverseOnionFrameV1::lease(&claim, &envelope, route, now + 600, now, relay).unwrap();
        RecipientDispatch {
            envelope, claim, lease, route_deadline: now + 600,
            route_origin_commitment: Some([9; 32]),
            armed_at: now,
        }
    }

    fn open_test_journal(
        relay: &IdentityKeyPair,
        local: &IdentityKeyPair,
        now: u64,
    ) -> (tempfile::TempDir, Arc<ReverseOnionRecipientJournal>) {
        let directory = tempfile::Builder::new()
            .prefix("reverse-terminal-journal-")
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp")
            .unwrap();
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex] Match the
        // production private-parent contract without weakening its preflight.
        #[cfg(unix)]
        std::fs::set_permissions(directory.path(),
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700)).unwrap();
        let journal = ReverseOnionRecipientJournal::open(
            &directory.path().join("recipient.sqlite"),
            relay.public_key_bytes(),
            local.public_key_bytes(),
            RecipientJournalLimits { max_entries: 16, max_bytes: 64 * 1024 * 1024 },
            now,
        )
        .unwrap();
        (directory, Arc::new(journal))
    }

    fn arm_in_journal(
        journal: &ReverseOnionRecipientJournal,
        dispatch: RecipientDispatch,
        now: u64,
    ) -> RecipientDispatch {
        let claim_id = dispatch.claim.claim_id();
        journal.prepare_poll(
            &dispatch.claim, dispatch.route_origin_commitment.expect("new test dispatch is pinned"), now,
        ).unwrap();
        journal.record_lease(claim_id, &dispatch.lease, dispatch.route_deadline, now).unwrap();
        journal.arm(claim_id, now).unwrap()
    }

    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Journal fixtures
    // need a canonical opaque reply carrier, not arbitrary bytes that the
    // core Result constructor correctly rejects before any persistence test.
    fn result_for_journal(armed: &RecipientDispatch, local: &IdentityKeyPair, now: u64) -> ReverseOnionFrameV1 {
        let (request, _) = OnionReplySession::prepare_source_sealed(
            armed.lease.route_id(), local.public_key_bytes(),
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"fixture operation".to_vec(),
        ).unwrap();
        let sealed = seal_onion_reply(armed.lease.route_id(), &request, b"fixture opaque result", local).unwrap();
        ReverseOnionFrameV1::result(
            &armed.claim, &armed.lease, &encode_onion_sealed_response(&sealed).unwrap(),
            armed.route_deadline, now, local,
        ).unwrap()
    }

    // [REVERSE-ONION-RESULT-DURABILITY 2026-10-05 by Codex] Authored, not
    // executed. Result persistence returns only after the exact canonical
    // bytes are recoverable from the recipient journal.
    #[tokio::test]
    async fn persisted_terminal_result_is_recoverable_as_exact_retry_bytes() {
        let relay = IdentityKeyPair::from_bytes(&[65; 32]).unwrap();
        let local = IdentityKeyPair::from_bytes(&[66; 32]).unwrap();
        let dispatch = live_dispatch(&relay, &local, 9);
        let now = dispatch.claim.issued_at();
        let (_directory, journal) = open_test_journal(&relay, &local, now);
        let armed = arm_in_journal(&journal, dispatch, now);
        let claim_id = armed.claim.claim_id();
        let result = result_for_journal(&armed, &local, now);
        let exact = result.encode();

        let returned = persist_exact_result(Arc::clone(&journal), claim_id, result,
            TerminalClockSource::default()).await.unwrap();
        assert_eq!(returned.encode(), exact);
        let page = journal.resume(None, 64, now_secs().unwrap()).unwrap();
        let recovered = match page.items.into_iter().next().unwrap() {
            RecipientRecovery::Result { exact_bytes, .. } => exact_bytes,
            _ => panic!("persisted result must be recoverable for exact retry"),
        };
        assert_eq!(recovered, exact);
    }

    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Authored
    // only: the actual tracked pre-entry path must publish a local fault,
    // never return ZeroDispatch and let the worker reopen this Armed lease.
    #[tokio::test]
    async fn rollback_below_armed_clock_stops_intake_before_router_entry() {
        let relay = IdentityKeyPair::from_bytes(&[85; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[86; 32]).unwrap());
        let dispatch = live_dispatch(&relay, &local, 85);
        let now = dispatch.claim.issued_at();
        let (_directory, journal) = open_test_journal(&relay, &local, now);
        let armed = arm_in_journal(&journal, dispatch, now);
        let floor = armed.armed_at;
        let entries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let entered = entries.clone();
        let router = Router::new().route("/api/chat/peer/blind-relay", axum::routing::post(move || {
            entered.fetch_add(1, Ordering::SeqCst);
            async { StatusCode::SERVICE_UNAVAILABLE }
        }));
        let adapter = ReverseOnionTerminalAdapter::new(router, local,
            relay.public_key_bytes(), Duration::from_secs(30)).unwrap()
            .with_clock_samples(&[Ok(floor - 1)]);
        assert_eq!(adapter.dispatch(armed, journal.clone()).await.err(), Some(ReverseOnionTerminalError::Clock));
        assert_eq!(entries.load(Ordering::SeqCst), 0);
        assert!(adapter.has_failed());
        assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
        assert_eq!(adapter.shutdown_and_drain().await, Err(ReverseOnionTerminalError::Ambiguous));
        assert!(matches!(journal.resume(None, 64, floor).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
    }

    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Authored
    // only: use the real lane/transaction and authenticated existing-only reopen.
    #[tokio::test]
    async fn result_clock_rollback_never_persists_or_reopens_execution_after_restart() {
        let relay = IdentityKeyPair::from_bytes(&[87; 32]).unwrap();
        let local = IdentityKeyPair::from_bytes(&[88; 32]).unwrap();
        let dispatch = live_dispatch(&relay, &local, 87);
        let now = dispatch.claim.issued_at();
        let (directory, journal) = open_test_journal(&relay, &local, now);
        let armed = arm_in_journal(&journal, dispatch, now);
        let floor = armed.armed_at;
        let claim_id = armed.claim.claim_id();
        let result = result_for_journal(&armed, &local, floor);
        let clock = TerminalClockSource {
            samples: Some(Arc::new(Mutex::new(std::collections::VecDeque::from([Ok(floor - 1)])))),
        };
        assert_eq!(persist_exact_result(journal.clone(), claim_id, result, clock).await.err(),
            Some(RecipientJournalError::Unavailable));
        assert_eq!(journal.resume(None, 64, floor).err(), Some(RecipientJournalError::Unavailable));
        drop(journal);
        let reopened = ReverseOnionRecipientJournal::open_existing(
            &directory.path().join("recipient.sqlite"), relay.public_key_bytes(), local.public_key_bytes(),
            RecipientJournalLimits { max_entries: 16, max_bytes: 64 * 1024 * 1024 }, floor,
        ).unwrap();
        assert!(matches!(reopened.resume(None, 64, floor).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
        assert!(reopened.arm(claim_id, floor).is_err());
    }

    // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] Authored, not run:
    // stop at the real post-prepare boundary must prove zero router entry.
    // Only that proof permits restoring the exact durable lease.
    #[tokio::test]
    async fn stop_after_preparation_preserves_zero_dispatch_and_exact_lease_retry() {
        let relay = IdentityKeyPair::from_bytes(&[75; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[76; 32]).unwrap());
        let dispatch = live_dispatch(&relay, &local, 11);
        let now = dispatch.claim.issued_at();
        let (_directory, journal) = open_test_journal(&relay, &local, now);
        let armed = arm_in_journal(&journal, dispatch, now);
        let claim = armed.claim.clone();
        let lease = armed.lease.clone();
        let deadline = armed.route_deadline;
        let claim_id = claim.claim_id();
        let entries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&entries);
        let router = Router::new().route("/api/chat/peer/blind-relay", axum::routing::post(
            move || {
                observed.fetch_add(1, Ordering::SeqCst);
                async { StatusCode::SERVICE_UNAVAILABLE }
            },
        ));
        let (prepared, mut preparations) = tokio::sync::mpsc::channel(1);
        let gate = Arc::new(Semaphore::new(0));
        let mut adapter = ReverseOnionTerminalAdapter::new(
            router, local, relay.public_key_bytes(), Duration::from_secs(30),
        ).unwrap();
        adapter.prepared_gate = Some((prepared, Arc::clone(&gate)));
        let adapter = Arc::new(adapter);
        let shared_stop = adapter.intake_stop_flag();
        assert!(Arc::ptr_eq(&shared_stop, &adapter.intake_stop_flag()));
        let caller_adapter = Arc::clone(&adapter);
        let caller_journal = Arc::clone(&journal);
        let mut caller = tokio::spawn(async move { caller_adapter.dispatch(armed, caller_journal).await });
        tokio::select! {
            ready = preparations.recv() => { ready.unwrap(); }
            _ = &mut caller => panic!("caller exited before the post-prepare boundary"),
        }
        // Same atomic gate as WorkerStop::closed, not drain's separate registry.
        shared_stop.store(true, Ordering::SeqCst);
        gate.add_permits(1);
        assert_eq!(caller.await.unwrap().err(), Some(ReverseOnionTerminalError::ZeroDispatch));
        assert_eq!(entries.load(Ordering::SeqCst), 0);
        assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES);
        let at = now_secs().unwrap();
        assert!(matches!(journal.resume(None, 64, at).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
        journal.restore_lease_after_zero_dispatch(claim_id, &claim, &lease, deadline, at).unwrap();
        assert!(matches!(journal.resume(None, 64, at).unwrap().items.as_slice(),
            [RecipientRecovery::LeaseReady { .. }]));
        adapter.shutdown_and_drain().await.unwrap();
        assert!(shared_stop.load(Ordering::SeqCst));
        assert_eq!(entries.load(Ordering::SeqCst), 0);
    }

    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
    // not run: include a late failed fence after the caller has timed out.
    // The sticky failure and repeated drain cannot report successful custody.
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Authored, not
    // run: a synthetic terminal router isolates DB ownership after response.
    // This is not evidence of production middleware or business completion.
    #[tokio::test(start_paused = true)]
    async fn late_router_result_waits_for_journal_lane_and_drains_after_caller_timeout() {
        for fail_fence in [false, true] {
            let clock = HoldPausedClock::new();
            let relay = IdentityKeyPair::from_bytes(&[67; 32]).unwrap();
            let local = Arc::new(IdentityKeyPair::from_bytes(&[68; 32]).unwrap());
            let dispatch = live_dispatch(&relay, local.as_ref(), 10);
            let now = dispatch.claim.issued_at();
            let (_directory, journal) = open_test_journal(&relay, local.as_ref(), now);
            let armed = arm_in_journal(&journal, dispatch, now);
            let claim = armed.claim.clone();
            let lease = armed.lease.clone();
            let deadline = armed.route_deadline;
            let claim_id = claim.claim_id();
            let lane = journal.blocking_operation_lane();
            let held = Arc::clone(&lane).acquire_owned().await.unwrap();
            let (entered, mut entries) = tokio::sync::mpsc::channel(1);
            let router_identity = Arc::clone(&local);
            let router = Router::new().route("/api/chat/peer/blind-relay", axum::routing::post(
                move |axum::Json(request): axum::Json<PeerBlindRelayRequest>| {
                    let local = Arc::clone(&router_identity);
                    let entered = entered.clone();
                    async move {
                        let at = now_secs().unwrap();
                        let secrets = crate::services::onion_keys::peel_secrets(at);
                        let peeled = try_open_onion_layer(&request.envelope.encrypted_blob, &secrets).unwrap();
                        let reply = decode_onion_reply_request(&peeled.inner).unwrap();
                        let sealed = seal_onion_reply(request.envelope.route_id, &reply, b"fixture result", &local).unwrap();
                        let opaque = STANDARD.encode(encode_onion_sealed_response(&sealed).unwrap());
                        let receipt = BlindRelaySuccessReceipt::terminal(
                            &request.envelope, request.envelope.ttl, Some("onion_terminal_delivered"),
                            None, Some(opaque.as_bytes()), at, &local,
                        );
                        entered.try_send(()).unwrap();
                        axum::Json(PeerBlindRelayResponse {
                            accepted: true, terminal: true, forwarded: false,
                            ttl_remaining: request.envelope.ttl,
                            reason: Some("onion_terminal_delivered".into()), delivery_receipt: None,
                            success_receipt: Some(receipt), failure_receipt: None,
                            opaque_terminal_response_b64: Some(opaque),
                        })
                    }
                },
            ));
            let (result_ready, mut ready_results) = tokio::sync::mpsc::channel(1);
            let mut adapter = ReverseOnionTerminalAdapter::new(
                router, Arc::clone(&local), relay.public_key_bytes(), Duration::from_secs(1),
            ).unwrap();
            adapter.result_ready = Some(result_ready);
            let adapter = Arc::new(adapter);
            let caller_adapter = Arc::clone(&adapter);
            let caller_journal = Arc::clone(&journal);
            let mut caller = tokio::spawn(async move { caller_adapter.dispatch(armed, caller_journal).await });
            tokio::select! {
                entry = entries.recv() => { entry.unwrap(); }
                _ = &mut caller => panic!("caller completed before terminal response"),
            }
            // The result signal is after crypto returns. The held DB lane then
            // makes the next await deterministic: releasing the execution permit
            // before persistence would fail the availability assertion below.
            tokio::select! {
                ready = ready_results.recv() => { ready.unwrap(); }
                _ = &mut caller => panic!("caller completed before result-lane wait"),
            }
            // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] Closing intake
            // after router completion must not close the late-result DB lane.
            adapter.request_stop();
            // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] The
            // shared stop gate rejects a new poll, but cannot cancel this
            // already-entered Result writer waiting on the same journal lane.
            let stop_at = now_secs().unwrap();
            let stopped = adapter.intake_stop_flag();
            assert!(stopped.load(Ordering::SeqCst));
            let fresh = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [0xf1; 16],
                stop_at, stop_at + 30, &local).unwrap();
            assert_eq!(journal.prepare_poll_with_admission_at(&fresh, [9; 32], || Ok(stop_at), || {
                if stopped.load(Ordering::SeqCst) { Err(RecipientJournalError::IntakeClosed) }
                else { Ok(()) }
            }).err(), Some(RecipientJournalError::IntakeClosed));
            tokio::time::advance(Duration::from_secs(2)).await;
            drop(clock);
            assert_eq!(caller.await.unwrap().err(), Some(ReverseOnionTerminalError::Ambiguous));
            assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES - 1);
            let mut drain = Box::pin(adapter.shutdown_and_drain());
            assert!(futures::poll!(drain.as_mut()).is_pending());
            drop(drain);
            assert_eq!(adapter.tracked.lock().unwrap().handles.len(), 1);
            let mut fault = Box::pin(adapter.wait_for_failure());
            assert!(futures::poll!(fault.as_mut()).is_pending());
            assert!(!adapter.has_failed());
            if fail_fence { journal.fail_next_commit_fence(); }
            drop(held);
            if fail_fence {
                assert_eq!(adapter.shutdown_and_drain().await, Err(ReverseOnionTerminalError::Ambiguous));
                assert!(adapter.has_failed());
                assert!(futures::poll!(fault.as_mut()).is_ready());
                drop(fault);
                let mut late = Box::pin(adapter.wait_for_failure());
                assert!(futures::poll!(late.as_mut()).is_ready());
                assert_eq!(adapter.shutdown_and_drain().await, Err(ReverseOnionTerminalError::Ambiguous));
                assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES);
                assert_eq!(journal.resume(None, 64, now_secs().unwrap()).err(), Some(RecipientJournalError::Unavailable));
                continue;
            }
            adapter.shutdown_and_drain().await.unwrap();
            assert!(!adapter.has_failed());
            assert!(futures::poll!(fault.as_mut()).is_pending());
            drop(fault);
            assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES);
            assert!(Arc::clone(&lane).try_acquire_owned().is_ok());
            let at = now_secs().unwrap();
            let page = journal.resume(None, 64, at).unwrap();
            let [RecipientRecovery::Result { exact_bytes, .. }] = page.items.as_slice() else {
                panic!("late completion must survive for exact retry");
            };
            let result = ReverseOnionFrameV1::decode(exact_bytes, at).unwrap();
            assert!(result.verify_result(&claim, &lease, deadline, at).is_ok());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn timed_out_router_work_keeps_four_permits_and_drain_cancellation_keeps_handles() {
        let clock = HoldPausedClock::new();
        let relay = IdentityKeyPair::from_bytes(&[61; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[62; 32]).unwrap());
        let (_journal_directory, journal) = open_test_journal(&relay, &local, now_secs().unwrap());
        let gate = Arc::new(BlockingGate { open: Mutex::new(false), condition: std::sync::Condvar::new() });
        let _release = ReleaseOnDrop(Arc::clone(&gate));
        let (entered, mut entries) = tokio::sync::mpsc::unbounded_channel();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let adapter = Arc::new(ReverseOnionTerminalAdapter::new(
            blocked_router(Arc::clone(&gate), entered, Arc::clone(&count)), Arc::clone(&local),
            relay.public_key_bytes(), Duration::from_secs(1),
        ).unwrap());
        let mut callers = tokio::task::JoinSet::new();
        for id in 1..=4 {
            let dispatch = live_dispatch(&relay, local.as_ref(), id);
            let adapter = Arc::clone(&adapter);
            let dispatch_journal = Arc::clone(&journal);
            callers.spawn(async move { adapter.dispatch(dispatch, dispatch_journal).await });
        }
        // Entry signals originate inside the blocked spawn_blocking workers.
        for _ in 0..4 {
            tokio::select! {
                entry = entries.recv() => { entry.unwrap(); }
                _ = callers.join_next() => panic!("caller completed before blocked router entry"),
            }
        }
        tokio::time::advance(Duration::from_secs(2)).await;
        drop(clock);
        while let Some(caller) = callers.join_next().await {
            assert_eq!(caller.unwrap().err(), Some(ReverseOnionTerminalError::Ambiguous));
        }
        assert_eq!(adapter.permits.available_permits(), 0);
        assert_eq!(adapter.dispatch(live_dispatch(&relay, local.as_ref(), 5), Arc::clone(&journal)).await.err(),
            Some(ReverseOnionTerminalError::Busy));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 4);
        let mut drain = Box::pin(adapter.shutdown_and_drain());
        assert!(matches!(futures::poll!(drain.as_mut()), Poll::Pending));
        drop(drain); // Cancel the wait, not the tracked jobs.
        assert_eq!(adapter.tracked.lock().unwrap().handles.len(), 4);
        gate.release();
        adapter.shutdown_and_drain().await.unwrap();
        assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES);
        assert!(adapter.tracked.lock().unwrap().handles.is_empty());
        assert_eq!(adapter.dispatch(live_dispatch(&relay, local.as_ref(), 6), journal).await.err(),
            Some(ReverseOnionTerminalError::ZeroDispatch));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_caller_does_not_abort_owned_router_operation() {
        let clock = HoldPausedClock::new();
        let relay = IdentityKeyPair::from_bytes(&[63; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[64; 32]).unwrap());
        let (_journal_directory, journal) = open_test_journal(&relay, &local, now_secs().unwrap());
        let gate = Arc::new(BlockingGate { open: Mutex::new(false), condition: std::sync::Condvar::new() });
        let _release = ReleaseOnDrop(Arc::clone(&gate));
        let (entered, mut entries) = tokio::sync::mpsc::unbounded_channel();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let adapter = Arc::new(ReverseOnionTerminalAdapter::new(
            blocked_router(Arc::clone(&gate), entered, count), Arc::clone(&local),
            relay.public_key_bytes(), Duration::from_secs(30),
        ).unwrap());
        let dispatch = live_dispatch(&relay, local.as_ref(), 7);
        let caller_adapter = Arc::clone(&adapter);
        let mut caller = tokio::spawn(async move {
            caller_adapter.dispatch(dispatch, journal).await
        });
        tokio::select! {
            entry = entries.recv() => { entry.unwrap(); }
            _ = &mut caller => panic!("caller completed before blocked router entry"),
        }
        assert!(!caller.is_finished());
        caller.abort(); // Abort only the WAITING caller, never the tracked job.
        let _ = caller.await;
        drop(clock);
        assert_eq!(adapter.permits.available_permits(), MAX_LOCAL_DISPATCHES - 1);
        assert_eq!(adapter.tracked.lock().unwrap().handles.len(), 1);
        gate.release();
        adapter.shutdown_and_drain().await.unwrap();
        assert!(adapter.tracked.lock().unwrap().handles.is_empty());
    }
}
