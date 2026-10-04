// ============================================
// File: crates/aeronyx-server/src/api/reverse_onion_terminal.rs
// ============================================
//! In-process, terminal-only adapter for a durably Armed reverse delivery.
//!
//! [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] The supplied Router must be a
//! clone of the existing production peer router, with its original middleware,
//! shared limits, previous-hop admission and durable replay store intact. Never
//! construct a fresh router per request or call private handler helpers here.
//! This adapter neither arms the journal nor persists results. Its caller must
//! await journal.arm first, then persist the returned exact Result BEFORE POST.
//! Returning a Result authenticates opaque custody, not source execution proof.
//! [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Runtime owners must
//! close admission and await `shutdown_and_drain` BEFORE dropping this adapter
//! or its Tokio runtime. A wait timeout never aborts the tracked operation.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
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
use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use axum::Router;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower::ServiceExt;
use zeroize::{Zeroize, Zeroizing};

use super::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};
use crate::services::reverse_onion_recipient::RecipientDispatch;

const MAX_LOCAL_DISPATCHES: usize = 4;
// Existing peer router body limit, not an increase to its public admission.
const MAX_REQUEST_JSON_BYTES: usize = 2 * 1024 * 1024;
// Fixed-class opaque base64 plus bounded immediate-hop JSON receipt metadata.
const MAX_RESPONSE_JSON_BYTES: usize = MAX_ONION_SEALED_RESPONSE_BASE64_BYTES + 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ReverseOnionTerminalError {
    #[error("reverse terminal rejected")]
    Rejected,
    #[error("reverse terminal busy")]
    Busy,
    #[error("reverse terminal ambiguous")]
    Ambiguous,
}

type Result<T> = std::result::Result<T, ReverseOnionTerminalError>;

/// No Debug: identities, frames, response bodies and router state stay private.
pub(crate) struct ReverseOnionTerminalAdapter {
    router: Router,
    identity: Arc<IdentityKeyPair>,
    relay: [u8; 32],
    timeout: Duration,
    permits: Arc<Semaphore>,
    tracked: Mutex<TrackedOperations>,
    drain_waiter: tokio::sync::Mutex<()>,
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
            tracked: Mutex::new(TrackedOperations {
                closed: false, panicked: false, handles: Vec::new(),
            }),
            drain_waiter: tokio::sync::Mutex::new(()),
        })
    }

    /// Consumes one post-Armed projection. Even preflight rejection does NOT
    /// authorize resetting the journal or re-executing this lease. Caller
    /// timeout/cancellation drops only its result receiver, not the operation.
    /// Late results are not automatically journaled; never dispatch again.
    pub(crate) async fn dispatch(&self, armed: RecipientDispatch) -> Result<ReverseOnionFrameV1> {
        let permit = Arc::clone(&self.permits).try_acquire_owned()
            .map_err(|_| ReverseOnionTerminalError::Busy)?;
        let relay = self.relay;
        let local = self.identity.public_key_bytes();
        let router = self.router.clone();
        let identity = Arc::clone(&self.identity);
        let operation = async move {
            let prepared = tokio::task::spawn_blocking(move || prepare(armed, relay, local, permit))
                .await.map_err(|_| ReverseOnionTerminalError::Ambiguous)??;
            let before_dispatch = now_secs()?;
            if before_dispatch < prepared.started_at
                || before_dispatch >= prepared.armed.lease.expires_at()
            {
                return Err(ReverseOnionTerminalError::Rejected);
            }
            let PreparedDispatch { armed, json, response_class, started_at, _permit } = prepared;
            let request = Request::builder().method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json)).map_err(|_| ReverseOnionTerminalError::Rejected)?;

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
            tokio::task::spawn_blocking(move || {
                let _permit = _permit;
                let finished_at = now_secs().map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
                if finished_at < started_at { return Err(ReverseOnionTerminalError::Ambiguous); }
                let response: PeerBlindRelayResponse = serde_json::from_slice(&body)
                    .map_err(|_| ReverseOnionTerminalError::Ambiguous)?;
                let sealed = verify_response(&armed, &response, local, response_class, started_at, finished_at)?;
                ReverseOnionFrameV1::result(&armed.claim, &armed.lease, &sealed,
                    armed.route_deadline, finished_at, identity.as_ref())
                    .map_err(|_| ReverseOnionTerminalError::Ambiguous)
            }).await.map_err(|_| ReverseOnionTerminalError::Ambiguous)?
        };
        // [REVERSE-ONION-TRACKED-DISPATCH 2026-10-04 by Codex] Register under
        // the same lock as admission closure. No live handle is detached or
        // removed by timeout; the permit belongs to operation, not its waiter.
        let receive = {
            let mut tracked = self.tracked.lock().map_err(|_| ReverseOnionTerminalError::Rejected)?;
            if tracked.closed { return Err(ReverseOnionTerminalError::Rejected); }
            // Completed tasks own no continuing work; pending handles stay put.
            tracked.reap_finished();
            if tracked.panicked {
                tracked.closed = true;
                return Err(ReverseOnionTerminalError::Ambiguous);
            }
            if tracked.handles.len() >= MAX_LOCAL_DISPATCHES {
                return Err(ReverseOnionTerminalError::Busy);
            }
            let (send, receive) = oneshot::channel();
            tracked.handles.push(tokio::spawn(async move {
                let result = operation.await;
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
                        if result.is_err() { tracked.panicked = true; }
                    }
                    Poll::Pending => index += 1,
                }
            }
            if tracked.handles.is_empty() {
                Poll::Ready(if tracked.panicked { Err(ReverseOnionTerminalError::Ambiguous) } else { Ok(()) })
            } else {
                Poll::Pending
            }
        }).await
    }
}

// [REVERSE-ONION-TERMINAL 2026-10-04 by Codex] Preflight performs no workload
// or storage effect and rejects a forward hop before router dispatch. It reads
// the existing process onion-key manager; normal startup must initialize that
// manager. Existing router admission still runs independently afterward.
fn prepare(armed: RecipientDispatch, relay: [u8; 32], local: [u8; 32], permit: OwnedSemaphorePermit)
    -> Result<PreparedDispatch> {
    let now = now_secs()?;
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
    reply.payload.zeroize();
    if proof_mode != OnionReplyProofMode::SourceSealedTerminalProof {
        return Err(ReverseOnionTerminalError::Rejected);
    }
    let request = PeerBlindRelayRequest {
        envelope, previous_hop_node_id: relay,
        onward_envelope: None, onward_descriptor_hint: None,
    };
    let json = serde_json::to_vec(&request).map_err(|_| ReverseOnionTerminalError::Rejected)?;
    if json.len() > MAX_REQUEST_JSON_BYTES { return Err(ReverseOnionTerminalError::Rejected); }
    Ok(PreparedDispatch { armed, json, response_class, started_at: now, _permit: permit })
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

    const NOW: u64 = 1_800_000_000;

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
        (RecipientDispatch { envelope, claim, lease, route_deadline: NOW + 600 }, response, local)
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

    fn live_dispatch(relay: &IdentityKeyPair, local: &IdentityKeyPair, id: u8) -> RecipientDispatch {
        let now = now_secs().unwrap();
        let route = [id; 16];
        let (request, _session) = OnionReplySession::prepare_source_sealed(
            route, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"operation".to_vec(),
        ).unwrap();
        let inner = aeronyx_core::protocol::onion_reply::encode_onion_reply_request(&request).unwrap();
        let envelope = build_onion_envelope(
            &[OnionHop { node_id: local.public_key_bytes(),
                kem_pub: crate::services::onion_keys::current_public_key() }],
            &inner, route, 1, now, relay,
        ).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), route, now, now + 30, local).unwrap();
        let lease = ReverseOnionFrameV1::lease(&claim, &envelope, route, now + 600, now, relay).unwrap();
        RecipientDispatch { envelope, claim, lease, route_deadline: now + 600 }
    }

    #[tokio::test(start_paused = true)]
    async fn timed_out_router_work_keeps_four_permits_and_drain_cancellation_keeps_handles() {
        let clock = HoldPausedClock::new();
        let relay = IdentityKeyPair::from_bytes(&[61; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[62; 32]).unwrap());
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
            callers.spawn(async move { adapter.dispatch(dispatch).await });
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
        assert_eq!(adapter.dispatch(live_dispatch(&relay, local.as_ref(), 5)).await.err(),
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
        assert_eq!(adapter.dispatch(live_dispatch(&relay, local.as_ref(), 6)).await.err(),
            Some(ReverseOnionTerminalError::Rejected));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_caller_does_not_abort_owned_router_operation() {
        let clock = HoldPausedClock::new();
        let relay = IdentityKeyPair::from_bytes(&[63; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[64; 32]).unwrap());
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
        let mut caller = tokio::spawn(async move { caller_adapter.dispatch(dispatch).await });
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
