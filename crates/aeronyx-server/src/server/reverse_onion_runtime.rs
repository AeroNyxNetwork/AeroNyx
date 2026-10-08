// ============================================
// File: crates/aeronyx-server/src/server/reverse_onion_runtime.rs
// ============================================
//! Bounded outbound carrier for private-recipient reverse onion delivery.
//!
//! [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] The carrier remains
//! transport-only. The separately owned worker journals before send/dispatch,
//! authenticates relay execution authority, and drains before releasing DB.
//! Startup callers must await readiness and retain/drain the worker at shutdown.
//! No redirects, inherited proxies, unpinned DNS endpoints, or automatic retries.

use std::time::Duration;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use futures::FutureExt;
use rand::RngCore;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionError, ReverseOnionFrameV1, ReverseOnionKindV1, ReverseOnionNoWorkReceiptV1, MAX_REVERSE_ONION_FRAME_BYTES,
};

use crate::config_reverse_onion::ReverseOnionConfig;
use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
use crate::api::reverse_onion_terminal::ReverseOnionTerminalError;
use crate::services::reverse_onion_recipient::{
    RecipientJournalError, RecipientJournalLimits, RecipientRecovery,
    RecipientDispatch, ReverseOnionRecipientJournal,
};
use crate::services::peer_store::{PeerStore, PrivateOnionPullAuthoritySnapshot};

/// Coarse preflight errors have no network effects or identifying details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionPreflightError {
    Disabled,
    Configuration,
    Frame,
    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Local
    // clock failures are not missing authority or a remote transport outcome.
    Clock,
}

// [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] One attempt-local
// clock floor survives cancellation of its HTTP future. This is neither a
// global policy clock nor persisted authority; a failed sample stays failed.
pub(super) struct ReverseOnionLocalObservation {
    latest: AtomicU64,
    failed: AtomicBool,
}

impl ReverseOnionLocalObservation {
    pub(super) fn new(floor: u64) -> Self {
        Self { latest: AtomicU64::new(floor), failed: AtomicBool::new(false) }
    }

    pub(super) fn floor(&self) -> Result<u64, ()> {
        if self.failed.load(Ordering::Acquire) { return Err(()); }
        let floor = self.latest.load(Ordering::Acquire);
        if self.failed.load(Ordering::Acquire) { return Err(()); }
        Ok(floor)
    }

    pub(super) fn observe(&self, sample: Result<u64, ()>) -> Result<u64, ()> {
        let accepted = sample.and_then(|now| {
            self.floor()?;
            self.latest.fetch_update(Ordering::AcqRel, Ordering::Acquire,
                |floor| (now >= floor).then_some(now)).map_err(|_| ())?;
            self.floor().map(|_| now)
        });
        if accepted.is_err() { self.failed.store(true, Ordering::Release); }
        accepted
    }
}

/// A response is transport evidence only, never a custody/execution proof.
/// Intentionally omits Debug so opaque bodies cannot enter routine logs.
pub(crate) enum ReverseOnionExchange {
    /// An HTTP response arrived within both the time and body size bounds.
    Response { status: u16, body: Vec<u8>, authenticated_https: bool, observed_at: u64 },
    /// No request was sent because the route is unavailable or the exact
    /// frame crossed its immutable retry deadline during preflight.
    /// Keep the exact journal frame pending until discovery refreshes it.
    Deferred,
    /// A POST may have reached the relay. Preserve the exact durable request;
    /// this must never cause a new claim, lease, target or source reply key.
    Ambiguous,
}

// [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Keep the route
// selected before DNS through HTTP entry. Recovery intentionally needs no
// fresh P grant, but it still cannot switch descriptors inside one attempt.
enum RecipientRouteSnapshot {
    Pull(PrivateOnionPullAuthoritySnapshot),
    Recovery(SignedNodeDescriptor),
}

// [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Carry the last
// checked preflight time with the exact selected route, not a second lookup.
struct RecipientRouteTarget {
    target: crate::api::PinnedPeerHttpTarget,
    origin: [u8; 32],
    snapshot: RecipientRouteSnapshot,
    observed_at: u64,
}

impl RecipientRouteSnapshot {
    fn is_current_under_authority_guard(&self, peers: &PeerStore, now: u64) -> bool {
        match self {
            Self::Pull(expected) => peers.current_private_onion_pull_authority_snapshot_under_guard(
                &expected.relay.node_id(), &expected.recipient.node_id(), now,
            ).as_ref() == Some(expected),
            Self::Recovery(expected) => peers.current_private_onion_relay_descriptor_under_guard(
                &expected.node_id(), now,
            ).as_ref() == Some(expected),
        }
    }
}

/// Fixed adjacent relay and recipient identity for one outbound worker.
pub(crate) struct ReverseOnionHttpCarrier {
    relay: [u8; 32],
    recipient: [u8; 32],
    relay_origin_commitment: [u8; 32],
    peers: Arc<PeerStore>,
    timeout: Duration,
    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] The carrier shares
    // the actual worker/terminal stop gate, not a copied startup boolean.
    stopped: Arc<AtomicBool>,
}

// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] The relay descriptor may
// carry a hostname only because this carrier is bound to one configured relay
// identity and pins each fresh public DNS answer set before sending.

impl ReverseOnionHttpCarrier {
    /// Construct without opening sockets. Each request resolves a canonical
    /// URL from the current signed descriptor for the configured relay ID.
    pub(crate) fn new(
        config: &ReverseOnionConfig,
        recipient: [u8; 32],
        peers: Arc<PeerStore>,
        stopped: Arc<AtomicBool>,
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
        let relay_origin_commitment = crate::api::reverse_onion_origin_commitment(
            &config.recipient.relay_endpoint,
        )
        .map_err(|_| ReverseOnionPreflightError::Configuration)?;
        let timeout = Duration::from_secs(config.recipient.request_timeout_secs);
        Ok(Self {
            relay,
            recipient,
            relay_origin_commitment,
            peers,
            timeout,
            stopped,
        })
    }

    // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
    // Canonical bytes form one immutable R/P/grant epoch across DNS awaits.
    // No authority material is logged or persisted in the recipient journal.
    fn current_pull_authority_epoch(&self, now: u64) -> Option<PrivateOnionPullAuthoritySnapshot> {
        self.peers.current_private_onion_pull_authority_snapshot(
            &self.relay, &self.recipient, now,
        )
    }

    fn has_current_pull_authority(&self) -> Result<bool, RecipientWorkerError> {
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Do not
        // turn a failed local observation into ordinary discovery absence.
        Ok(self.current_pull_authority_epoch(worker_now()?).is_some())
    }

    // [REVERSE-ONION-RECIPIENT-ROUTE-REFRESH 2026-10-05 by Codex]
    // Missing/expired route state is a zero-send pause. The caller retains
    // and retries the exact journal frame after discovery refresh.
    async fn current_target(
        &self,
        path: &str,
        require_live_pull_authority: bool,
        expected_origin: Option<[u8; 32]>,
        retry_deadline: Option<u64>,
        clock_floor: u64,
    ) -> Result<Option<RecipientRouteTarget>, ReverseOnionPreflightError> {
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Missing
        // routes and DNS failures remain zero-send deferral. Every clock
        // observation after this operation began retains its fatal category.
        if self.stopped.load(Ordering::SeqCst) { return Ok(None); }
        let now = recipient_preflight_now(clock_floor, worker_now())?;
        let authority_before = if require_live_pull_authority {
            let Some(authority) = self.current_pull_authority_epoch(now) else { return Ok(None); };
            Some(authority)
        } else {
            None
        };
        let descriptor = match &authority_before {
            Some(epoch) => epoch.relay.clone(),
            None => {
                let Some(descriptor) = self.peers.current_private_onion_relay_descriptor(&self.relay, now)
                    else { return Ok(None); };
                descriptor
            }
        };
        if descriptor.node_id() != self.relay { return Ok(None); }
        let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else { return Ok(None); };
        // [PHALA-RECIPIENT-RELAY-ORIGIN-PIN 2026-10-06 by Codex] The fixed
        // bootstrap origin is also an immutable transport pin. Signed endpoint
        // rotation requires operator config change, not descriptor gossip alone.
        // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] The same
        // origin validator fences discovery and exact journal retransmission.
        let Ok(origin) = crate::api::reverse_onion_pinned_origin(
            endpoint, self.relay_origin_commitment,
        ) else { return Ok(None); };
        if expected_origin.is_some_and(|expected| expected != origin) {
            return Ok(None);
        }
        let Ok(url) = crate::api::canonical_peer_http_url(endpoint, path) else { return Ok(None); };
        // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] DNS may wait
        // only within this frame's retry window. A route/grant expiry never
        // substitutes for or extends the immutable journal frame deadline.
        let resolution_at = recipient_preflight_now(now, worker_now())?;
        let resolution_timeout = match retry_deadline {
            Some(deadline) if deadline <= resolution_at => return Ok(None),
            Some(deadline) => self.timeout.min(Duration::from_secs(deadline - resolution_at)),
            None => self.timeout,
        };
        let resolved = crate::api::resolve_pinned_peer_http_target(url, resolution_timeout).await;
        // [REVERSE-ONION-RECIPIENT-AUTHORITY-RECHECK 2026-10-05 by Codex]
        // DNS resolution is an await boundary. If discovery rotated the relay
        // meanwhile, keep the exact journal frame pending and send nothing.
        let checked_at = recipient_preflight_now(resolution_at, worker_now())?;
        if self.stopped.load(Ordering::SeqCst) { return Ok(None); }
        let Ok(target) = resolved else { return Ok(None); };
        let current = if authority_before.is_some() {
            let Some(after) = self.current_pull_authority_epoch(checked_at) else { return Ok(None); };
            if authority_before.as_ref() != Some(&after) { return Ok(None); }
            after.relay
        } else {
            let Some(current) = self.peers.current_private_onion_relay_descriptor(&self.relay, checked_at)
                else { return Ok(None); };
            current
        };
        let (Ok(current_bytes), Ok(selected_bytes)) = (current.encode_canonical(), descriptor.encode_canonical())
            else { return Ok(None); };
        if current_bytes != selected_bytes {
            return Ok(None);
        }
        // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Export only
        // the original selection, not a newly loaded post-DNS authority.
        let snapshot = match authority_before {
            Some(epoch) => RecipientRouteSnapshot::Pull(epoch),
            None => RecipientRouteSnapshot::Recovery(descriptor),
        };
        Ok(Some(RecipientRouteTarget { target, origin, snapshot, observed_at: checked_at }))
    }

    // [REVERSE-ONION-ORIGIN-BINDING 2026-10-06 by Codex] Resolve the current
    // signed HTTPS origin before arming a new exact Claim in the local journal.
    async fn prepare_new_claim_origin(&self)
        -> Result<Option<([u8; 32], u64)>, ReverseOnionPreflightError> {
        let floor = recipient_preflight_now(0, worker_now())?;
        self.current_target(
            "/api/chat/peer/reverse-onion/claim", true, None, None, floor,
        ).await.map(|target| target.map(|target| (target.origin, target.observed_at)))
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Sample the final
    // clock after acquiring authority, then re-admit the same selection.
    // This is an admission point, not a lease against later route revocation.
    fn final_send_admission_at(
        &self, frame: &ReverseOnionFrameV1, observation: &ReverseOnionLocalObservation,
        snapshot: &RecipientRouteSnapshot,
        clock: impl FnOnce() -> Result<u64, RecipientWorkerError>,
    ) -> Result<Option<Duration>, ReverseOnionPreflightError> {
        let Some(_authority_epoch) = self.peers.try_private_onion_authority_read_guard() else {
            return Ok(None);
        };
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Store even
        // a sample followed by zero-send rejection; a timeout cannot erase it.
        let now = observation.observe(clock().map_err(|_| ()))
            .map_err(|_| ReverseOnionPreflightError::Clock)?;
        let timeout = recipient_send_admission(&self.stopped, frame, now, Ok(now), self.timeout)?;
        if timeout.is_some() && snapshot.is_current_under_authority_guard(&self.peers, now) {
            Ok(timeout)
        } else {
            Ok(None)
        }
    }

    /// Submit exactly one already-persisted frame. No retry is performed here.
    /// Even a non-2xx HTTP response requires protocol-specific recovery policy;
    /// a successful status without a verified bound receipt proves nothing.
    pub(crate) async fn exchange(
        &self,
        frame: &ReverseOnionFrameV1,
        require_live_pull_authority: bool,
        expected_origin: [u8; 32],
    ) -> Result<ReverseOnionExchange, ReverseOnionPreflightError> {
        if frame.relay() != self.relay || frame.immediate_recipient() != self.recipient {
            return Err(ReverseOnionPreflightError::Frame);
        }
        let path = match frame.kind() {
            ReverseOnionKindV1::Claim => "/api/chat/peer/reverse-onion/claim",
            ReverseOnionKindV1::Result => "/api/chat/peer/reverse-onion/result",
            ReverseOnionKindV1::Lease => return Err(ReverseOnionPreflightError::Frame),
        };
        if self.stopped.load(Ordering::SeqCst) { return Ok(ReverseOnionExchange::Deferred); }
        // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Validate
        // before DNS and again afterward; expiry sends nothing and leaves
        // normal journal retention/cleanup responsible for the durable row.
        let preflight_at = recipient_preflight_now(0, worker_now())?;
        if recipient_exchange_timeout(frame, preflight_at, self.timeout)?.is_none() {
            return Ok(ReverseOnionExchange::Deferred);
        }
        let retry_deadline = frame.recipient_retry_deadline()
            .map_err(|_| ReverseOnionPreflightError::Frame)?;
        let Some(RecipientRouteTarget {
            target, snapshot: route_snapshot, observed_at: route_at, ..
        }) = self.current_target(
            path, require_live_pull_authority, Some(expected_origin), Some(retry_deadline), preflight_at,
        ).await? else {
            return Ok(ReverseOnionExchange::Deferred);
        };
        let authenticated_https = target.url.scheme() == "https";
        let bytes = frame.encode();
        if bytes.len() > MAX_REVERSE_ONION_FRAME_BYTES {
            return Err(ReverseOnionPreflightError::Frame);
        }
        let now = recipient_preflight_now(route_at, worker_now())?;
        let Some(timeout) = recipient_exchange_timeout(frame, now, self.timeout)? else {
            return Ok(ReverseOnionExchange::Deferred);
        };
        let observation = ReverseOnionLocalObservation::new(now);
        // One absolute timeout covers headers and streaming the complete body.
        // Never retain raw reqwest errors: they may contain the relay URL.
        let attempt = async {
            // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Finish local
            // construction before checking the shared stop/deadline gate.
            // After execute starts, errors never imply zero acceptance.
            let request = target.client.post(target.url)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(bytes)
                .build().map_err(|_| ReverseOnionPreflightError::Frame)?;
            let Some(remaining) = self.final_send_admission_at(
                frame, &observation, &route_snapshot, worker_now,
            )? else {
                return Ok(ReverseOnionExchange::Deferred);
            };
            // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Local gate faults
            // reach worker supervision; only entered HTTP failures are ambiguous.
            // Reclip the entire response wait to the final frame-time sample.
            let http = async {
                let mut response = target.client.execute(request).await.map_err(|_| ())?;
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
                Ok::<_, ()>((status, body, authenticated_https))
            };
            Ok::<_, ReverseOnionPreflightError>(match tokio::time::timeout(remaining, http).await {
                Ok(Ok((status, body, authenticated_https))) => ReverseOnionExchange::Response {
                    status, body, authenticated_https,
                    observed_at: observation.floor().map_err(|_| ReverseOnionPreflightError::Clock)?,
                },
                _ => ReverseOnionExchange::Ambiguous,
            })
        };
        let outcome = match tokio::time::timeout(timeout, attempt).await {
            Ok(outcome) => outcome,
            Err(_) => Ok(ReverseOnionExchange::Ambiguous),
        };
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Check after
        // dropping the timed-out future too, before any response is accepted.
        let observed_at = observation.observe(worker_now().map_err(|_| ()))
            .map_err(|_| ReverseOnionPreflightError::Clock)?;
        outcome.map(|response| match response {
            ReverseOnionExchange::Response { status, body, authenticated_https, .. } =>
                ReverseOnionExchange::Response { status, body, authenticated_https, observed_at },
            other => other,
        })
    }
}

// [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Normal forward expiry
// is a zero-send outcome, unlike signature failure, wrong frame kind, or a
// clock preceding issuance. Never infer no acceptance after a POST timeout.
fn recipient_exchange_timeout(
    frame: &ReverseOnionFrameV1, now: u64, configured: Duration,
) -> Result<Option<Duration>, ReverseOnionPreflightError> {
    match frame.verify_recipient_retry(now) {
        Ok(()) => {
            let deadline = frame.recipient_retry_deadline()
                .map_err(|_| ReverseOnionPreflightError::Frame)?;
            Ok(Some(configured.min(Duration::from_secs(deadline - now))))
        }
        Err(ReverseOnionError::Expired) => Ok(None),
        Err(_) => Err(ReverseOnionPreflightError::Frame),
    }
}

// [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] This exact gate runs after
// request construction, not just before DNS. Clock/frame faults remain errors
// even if stop races them; normal stop/forward expiry retain the exact frame.
fn recipient_send_admission(
    stopped: &AtomicBool, frame: &ReverseOnionFrameV1, floor: u64,
    now: Result<u64, RecipientWorkerError>, configured: Duration,
) -> Result<Option<Duration>, ReverseOnionPreflightError> {
    let now = recipient_preflight_now(floor, now)?;
    let timeout = recipient_exchange_timeout(frame, now, configured)?;
    if stopped.load(Ordering::SeqCst) { Ok(None) } else { Ok(timeout) }
}

// [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] All carrier
// observations share this checked floor, including failed DNS resolution and
// final request entry. Forward expiry is handled separately as zero-send.
fn recipient_preflight_now(floor: u64, now: Result<u64, RecipientWorkerError>)
    -> Result<u64, ReverseOnionPreflightError> {
    let now = now.map_err(|_| ReverseOnionPreflightError::Clock)?;
    if now < floor { return Err(ReverseOnionPreflightError::Clock); }
    Ok(now)
}

// [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] Single owned task,
// not detached per-tick effects. Limits are independent of network success.
const RECIPIENT_PAGE: usize = 64;
const RECIPIENT_LOCAL_BYTES: u64 = 64 * 1024 * 1024;
// [REVERSE-ONION-RETRY-BACKOFF 2026-10-05 by Codex] Cap transient recovery
// traffic independently of the operator's configured minimum poll interval.
const MAX_RECIPIENT_RETRY_BACKOFF: Duration = Duration::from_secs(5);

// [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Scheduling owns
// journal/terminal effects regardless of transport. Production startup always
// constructs the pinned HTTP carrier; only cfg(test) exposes injection.
#[async_trait::async_trait]
pub(super) trait RecipientOutboundCarrier: Send + Sync {
    fn binding(&self) -> ([u8; 32], [u8; 32], Duration);
    fn has_current_pull_authority(&self) -> Result<bool, RecipientWorkerError>;
    async fn prepare_new_claim_origin(&self)
        -> Result<Option<([u8; 32], u64)>, ReverseOnionPreflightError>;
    async fn exchange(&self, frame: &ReverseOnionFrameV1, require_live_pull_authority: bool,
        expected_origin: [u8; 32]) -> Result<ReverseOnionExchange, ReverseOnionPreflightError>;
}

#[async_trait::async_trait]
impl RecipientOutboundCarrier for ReverseOnionHttpCarrier {
    fn binding(&self) -> ([u8; 32], [u8; 32], Duration) {
        (self.relay, self.recipient, self.timeout)
    }
    fn has_current_pull_authority(&self) -> Result<bool, RecipientWorkerError> {
        ReverseOnionHttpCarrier::has_current_pull_authority(self)
    }
    async fn prepare_new_claim_origin(&self)
        -> Result<Option<([u8; 32], u64)>, ReverseOnionPreflightError> {
        ReverseOnionHttpCarrier::prepare_new_claim_origin(self).await
    }
    async fn exchange(&self, frame: &ReverseOnionFrameV1, require_live_pull_authority: bool,
        expected_origin: [u8; 32]) -> Result<ReverseOnionExchange, ReverseOnionPreflightError> {
        ReverseOnionHttpCarrier::exchange(self, frame, require_live_pull_authority, expected_origin).await
    }
}

// [REVERSE-ONION-RETRY-BACKOFF 2026-10-05 by Codex] Distinguish completed
// empty polls, newly persisted leases, and work that still needs exact retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PollAttempt {
    Resolved,
    Progressed,
    Pending,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecipientWorkerError { Rejected, Unavailable, Ambiguous }

// [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Preserve the
// source-blind local fault classification at the owned worker boundary.
impl From<ReverseOnionPreflightError> for RecipientWorkerError {
    fn from(error: ReverseOnionPreflightError) -> Self {
        match error {
            ReverseOnionPreflightError::Clock => Self::Unavailable,
            _ => Self::Rejected,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerReady { Starting, Ready, Failed }

// [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] One stop gate covers
// polling, new DB mutations and terminal pre-router entry. Never reopen it.
struct WorkerStop { closed: Arc<AtomicBool>, wake: Notify }

// [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Publish fatal
// ownership loss before awaiting accepted completion work. The watcher is
// separate from task exit so supervision can stop other intake during drain.
fn report_recipient_worker_failure(stop: &WorkerStop, failure: &watch::Sender<bool>) {
    stop.closed.store(true, Ordering::SeqCst);
    stop.wake.notify_one();
    failure.send_replace(true);
}

/// No Debug or abort API. Dropping a shutdown waiter never loses its task handle.
/// The composition owner must retain this object until shutdown_and_drain ends;
/// Drop requests stop, but cannot synchronously promise completed async drain.
pub(crate) struct ReverseOnionRecipientWorker {
    stop: Arc<WorkerStop>,
    ready: watch::Receiver<WorkerReady>,
    // [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] Channel closure
    // is an independent completion signal, including panic/unwind paths.
    completed: watch::Receiver<()>,
    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Sticky local
    // failure is observable while the same owned task is still draining.
    failure: watch::Receiver<bool>,
    task: Mutex<Option<JoinHandle<Result<(), RecipientWorkerError>>>>,
    drain_waiter: tokio::sync::Mutex<()>,
    finished: Mutex<Option<Result<(), RecipientWorkerError>>>,
}

impl ReverseOnionRecipientWorker {
    /// Disabled/configuration rejection happens before task or DB creation.
    /// `adapter` must be the reviewed local terminal adapter, never a network
    /// forwarder. No router registration or public endpoint is added here.
    pub(crate) fn start(config: &ReverseOnionConfig, identity: Arc<IdentityKeyPair>,
        peers: Arc<PeerStore>,
        adapter: Arc<ReverseOnionTerminalAdapter>) -> Result<Self, ReverseOnionPreflightError> {
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Bind the same
        // irreversible intake gate before either carrier or task is published.
        let stop = Arc::new(WorkerStop { closed: adapter.intake_stop_flag(), wake: Notify::new() });
        let carrier = ReverseOnionHttpCarrier::new(config, identity.public_key_bytes(), peers,
            Arc::clone(&stop.closed))?;
        Self::start_owned(config, identity, adapter, stop, carrier)
    }

    // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Reuse the actual
    // config/binding checks, DB opener, worker supervision and irreversible
    // stop gate. This constructor is absent from non-test builds.
    #[cfg(test)]
    pub(super) fn start_with_test_carrier<C: RecipientOutboundCarrier + 'static>(
        config: &ReverseOnionConfig, identity: Arc<IdentityKeyPair>,
        adapter: Arc<ReverseOnionTerminalAdapter>,
        make_carrier: impl FnOnce(Arc<AtomicBool>) -> C,
    ) -> Result<Self, ReverseOnionPreflightError> {
        let stop = Arc::new(WorkerStop { closed: adapter.intake_stop_flag(), wake: Notify::new() });
        let checked = ReverseOnionHttpCarrier::new(config, identity.public_key_bytes(),
            Arc::new(PeerStore::new()), Arc::clone(&stop.closed))?;
        let carrier = make_carrier(Arc::clone(&stop.closed));
        if carrier.binding() != checked.binding() {
            return Err(ReverseOnionPreflightError::Configuration);
        }
        Self::start_owned(config, identity, adapter, stop, carrier)
    }

    fn start_owned<C: RecipientOutboundCarrier + 'static>(config: &ReverseOnionConfig,
        identity: Arc<IdentityKeyPair>, adapter: Arc<ReverseOnionTerminalAdapter>,
        stop: Arc<WorkerStop>, carrier: C) -> Result<Self, ReverseOnionPreflightError> {
        let path = std::path::PathBuf::from(&config.recipient.state_db_path);
        let entries = config.recipient.max_pending_items as usize;
        let recovery_only = config.recipient.recovery_only;
        let permits_new_claims = config.recipient.permits_new_claims();
        let delay = Duration::from_millis(config.recipient.poll_interval_ms);
        let task_stop = Arc::clone(&stop);
        let (ready_tx, ready) = watch::channel(WorkerReady::Starting);
        let (completion_tx, completed) = watch::channel(());
        let (failure_tx, failure) = watch::channel(false);
        // Register synchronously: no cancellation point between spawn and owner.
        let task = tokio::spawn(async move {
            let _completion_guard = completion_tx;
            let (relay, recipient, _) = carrier.binding();
            let opened = tokio::task::spawn_blocking(move || {
                // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] The mode is
                // fixed for this worker's lifetime, including its DB opener.
                let open = if recovery_only { ReverseOnionRecipientJournal::open_existing }
                    else { ReverseOnionRecipientJournal::open };
                open(&path, relay, recipient,
                    RecipientJournalLimits { max_entries: entries, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now()?)
                    .map(Arc::new).map_err(|_| RecipientWorkerError::Unavailable)
            }).await;
            let journal = match opened {
                Ok(Ok(journal)) => journal,
                _ => {
                    report_recipient_worker_failure(&task_stop, &failure_tx);
                    let _ = ready_tx.send(WorkerReady::Failed);
                    let _ = adapter.shutdown_and_drain().await;
                    return Err(RecipientWorkerError::Unavailable);
                }
            };
            let _ = ready_tx.send(WorkerReady::Ready);
            let outcome = std::panic::AssertUnwindSafe(recipient_loop(
                &carrier, &journal, &adapter, &identity, &task_stop, delay, permits_new_claims))
                .catch_unwind().await.unwrap_or(Err(RecipientWorkerError::Ambiguous));
            if outcome.is_err() { report_recipient_worker_failure(&task_stop, &failure_tx); }
            task_stop.closed.store(true, Ordering::SeqCst);
            // No worker future is selected away at a DB/terminal await. Even
            // adapter timeout leaves owned work which must finish before DBdrop.
            let drained = adapter.shutdown_and_drain().await
                .map_err(|_| RecipientWorkerError::Ambiguous);
            if drained.is_err() { report_recipient_worker_failure(&task_stop, &failure_tx); }
            drop(journal);
            outcome.and(drained)
        });
        Ok(Self { stop, ready, completed, failure, task: Mutex::new(Some(task)),
            drain_waiter: tokio::sync::Mutex::new(()), finished: Mutex::new(None) })
    }

    pub(super) async fn wait_for_task_exit(&self) {
        wait_for_worker_task_exit(self.completed.clone()).await;
    }

    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Dropping a
    // supervisor waiter cannot consume or clear this failure observation.
    fn has_failed(&self) -> bool {
        *self.failure.borrow() || self.finished.lock()
            .map(|finished| matches!(*finished, Some(Err(_)))).unwrap_or(true)
    }

    async fn wait_for_failure_or_exit(&self) {
        let mut failure = self.failure.clone();
        if *failure.borrow_and_update() { return; }
        tokio::select! {
            _ = failure.changed() => {},
            _ = self.wait_for_task_exit() => {},
        }
    }

    // [PHALA-READY-PUBLICATION 2026-10-07 by Codex] A prior successful
    // await is not a publication permit. Re-read the actual worker signals
    // without yielding; a closed readiness sender also fails closed.
    fn verify_ready_now(&self) -> Result<(), RecipientWorkerError> {
        let state = *self.ready.borrow();
        if state != WorkerReady::Ready
            || self.has_failed()
            || self.ready.has_changed().is_err()
            || worker_task_has_exited(&self.completed)
            || self.stop.closed.load(Ordering::SeqCst)
        {
            return Err(RecipientWorkerError::Unavailable);
        }
        Ok(())
    }

    pub(crate) async fn wait_ready(&self) -> Result<(), RecipientWorkerError> {
        let mut ready = self.ready.clone();
        loop {
            let state = *ready.borrow_and_update();
            match state {
                WorkerReady::Ready => return self.verify_ready_now(),
                WorkerReady::Failed => return Err(RecipientWorkerError::Unavailable),
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

    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] A known
    // local fault remains failed even if normal shutdown races its observer.
    pub(super) fn has_failed(&self) -> bool {
        self.worker.lock().map(|slot| slot.as_ref().is_some_and(|worker| worker.has_failed()))
            .unwrap_or(true)
    }

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

    // [PHALA-READY-PUBLICATION 2026-10-07 by Codex] The installed owner
    // and its worker must both still be live at the final server READY gate.
    pub(super) fn verify_ready_now(&self) -> Result<(), RecipientWorkerError> {
        let worker = self.worker.lock().map_err(|_| RecipientWorkerError::Unavailable)?.clone()
            .ok_or(RecipientWorkerError::Unavailable)?;
        worker.verify_ready_now()?;
        if !self.active.load(Ordering::SeqCst) || self.is_cancelled() {
            return Err(RecipientWorkerError::Unavailable);
        }
        Ok(())
    }

    // [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] Unexpected
    // worker exit is fatal. [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex]
    // A known local failure is observable before drain and outranks cancellation.
    pub(super) async fn wait_for_unexpected_worker_failure_or_exit(&self) -> bool {
        let worker = match self.worker.lock() {
            Ok(slot) => slot.clone(),
            Err(_) => return true,
        };
        let Some(worker) = worker else { return true; };
        worker.wait_for_failure_or_exit().await;
        worker.has_failed() || !self.cancelled.load(Ordering::Acquire)
    }
}

// [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] Exposed within the
// server module so the completion-channel contract has a focused unrun test.
pub(super) async fn wait_for_worker_task_exit(mut completed: watch::Receiver<()>) {
    let _ = completed.changed().await;
}

// [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] This worker never
// sends a completion value; a closed channel therefore means the task ended.
pub(super) fn worker_task_has_exited(completed: &watch::Receiver<()>) -> bool {
    completed.has_changed().is_err()
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

// [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Sample the
// worker's actual clock under the journal transaction, including pre-call
// signing/decoding time. A backward move below this operation's floor is fatal.
fn worker_journal_now(floor: u64) -> Result<u64, RecipientJournalError> {
    let now = worker_now().map_err(|_| RecipientJournalError::Unavailable)?;
    if now < floor { return Err(RecipientJournalError::Rejected); }
    Ok(now)
}

// [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] This exact shared
// adapter/worker flag is checked inside fresh Claim/Armed transactions, not
// only before their lane waits. Completion writes deliberately do not use it.
pub(super) fn recipient_intake_open(stopped: &AtomicBool) -> Result<(), RecipientJournalError> {
    if stopped.load(Ordering::SeqCst) { Err(RecipientJournalError::IntakeClosed) }
    else { Ok(()) }
}

async fn recipient_db<T: Send + 'static>(journal: &Arc<ReverseOnionRecipientJournal>,
    action: impl FnOnce(&ReverseOnionRecipientJournal, u64) -> Result<T, RecipientJournalError> + Send + 'static,
) -> Result<T, RecipientWorkerError> {
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] The journal
    // owns serialization/cancellation safety; refresh time after the lane wait.
    journal.run_blocking(move |journal| {
        let now = worker_now().map_err(|_| RecipientJournalError::Unavailable)?;
        action(journal, now)
    }).await.map_err(|error| match error {
        RecipientJournalError::Ambiguous => RecipientWorkerError::Ambiguous,
        _ => RecipientWorkerError::Unavailable,
    })
}

// [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] A page's LeaseReady
// projection is not an execution permit. The transaction rechecks time and
// leaves an expired lease untouched; every other failure remains fail-closed.
// [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Ungated adapters
// are fixture-only; production always supplies the shared final intake gate.
#[cfg(test)]
pub(super) fn arm_recovery_lease(
    journal: &ReverseOnionRecipientJournal, claim_id: [u8; 16], now: u64,
) -> Result<Option<RecipientDispatch>, RecipientJournalError> {
    let started = std::time::Instant::now();
    arm_recovery_lease_at(journal, claim_id, || {
        now.checked_add(started.elapsed().as_secs()).ok_or(RecipientJournalError::Unavailable)
    })
}

// [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] The worker's
// actual arm path samples under SQL lock; only forward expiry is skippable.
#[cfg(test)]
pub(super) fn arm_recovery_lease_at(
    journal: &ReverseOnionRecipientJournal, claim_id: [u8; 16],
    refresh_now: impl FnOnce() -> Result<u64, RecipientJournalError>,
) -> Result<Option<RecipientDispatch>, RecipientJournalError> {
    arm_recovery_lease_with_admission_at(journal, claim_id, refresh_now, || Ok(()))
}

// [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Only the explicit
// closed-intake outcome is a normal stop. A concurrent stop cannot downgrade
// corrupt leases, clock rollback or storage faults into a successful drain.
pub(super) fn arm_recovery_lease_with_admission_at(
    journal: &ReverseOnionRecipientJournal, claim_id: [u8; 16],
    refresh_now: impl FnOnce() -> Result<u64, RecipientJournalError>,
    admit: impl FnOnce() -> Result<(), RecipientJournalError>,
) -> Result<Option<RecipientDispatch>, RecipientJournalError> {
    match journal.arm_with_admission_at(claim_id, refresh_now, admit) {
        Ok(armed) => Ok(Some(armed)),
        Err(RecipientJournalError::Expired | RecipientJournalError::IntakeClosed) => Ok(None),
        Err(error) => Err(error),
    }
}

// [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] The same scheduler
// creates/persists fresh Claims and retries exact frames in connected fixtures.
async fn recipient_loop<C: RecipientOutboundCarrier>(carrier: &C, journal: &Arc<ReverseOnionRecipientJournal>,
    adapter: &Arc<ReverseOnionTerminalAdapter>, identity: &Arc<IdentityKeyPair>,
    stop: &Arc<WorkerStop>, delay: Duration, permits_new_claims: bool) -> Result<(), RecipientWorkerError> {
    let mut after = None;
    let mut pass_has_work = false;
    // [REVERSE-ONION-RETRY-BACKOFF 2026-10-05 by Codex] Back off only after
    // unresolved work; durable progress resets the retry streak.
    let mut pass_needs_retry = false;
    let mut retry_streak = 0u8;
    let mut wait = delay;
    // [REVERSE-ONION-RESULT-CUSTODY-ACK 2026-10-05 by Codex] A process-local
    // cache is sufficient: on restart the exact durable Result is resent and
    // must receive the same idempotent relay echo before polling advances.
    let mut acknowledged_results = HashSet::new();
    while !stop.closed.load(Ordering::SeqCst) {
        // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex]
        // Terminal completion can fail after its waiter timed out. Observe it
        // between owned effects; never select away a DB/router/network effect.
        terminal_owner_health(adapter)?;
        let removed = recipient_db(journal, |j, now| j.cleanup(RECIPIENT_PAGE, now)).await?;
        if removed != 0 { acknowledged_results.clear(); }
        let page = recipient_db(journal, move |j, now| j.resume(after, RECIPIENT_PAGE, now)).await?;
        after = page.next_after;
        for item in page.items {
            terminal_owner_health(adapter)?;
            if stop.closed.load(Ordering::SeqCst) { break; }
            match item {
                RecipientRecovery::Poll { exact_bytes, route_origin_commitment, .. } => {
                    match exchange_poll(
                        carrier, journal, exact_bytes, route_origin_commitment, false,
                    ).await? {
                        PollAttempt::Resolved => {}
                        PollAttempt::Progressed => pass_has_work = true,
                        PollAttempt::Pending => {
                            pass_has_work = true;
                            pass_needs_retry = true;
                        }
                    }
                }
                RecipientRecovery::LeaseReady { claim_id } => {
                    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex]
                    // Stop may race the async lane wait. Recheck before the
                    // Armed mutation; the existing post-arm check still fences
                    // a later stop without undoing a committed Armed barrier.
                    let arm_stop = Arc::clone(stop);
                    let armed = recipient_db(journal, move |j, now| {
                        if arm_stop.closed.load(Ordering::SeqCst) { return Ok(None); }
                        arm_recovery_lease_with_admission_at(j, claim_id, || worker_journal_now(now),
                            || recipient_intake_open(&arm_stop.closed))
                    }).await?;
                    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex]
                    // A stopped/expired projection did not resolve custody.
                    // Retain the recovery barrier for this entire pass; the
                    // next scan classifies the durable row before fresh polls.
                    let Some(armed) = armed else {
                        pass_has_work = true;
                        pass_needs_retry = true;
                        continue;
                    };
                    // Stop racing after durable arm leaves ambiguous, never undo.
                    if stop.closed.load(Ordering::SeqCst) { break; }
                    // [REVERSE-ONION-RESULT-DURABILITY 2026-10-05 by Codex]
                    // Adapter owns persistence so timeout cannot strand a late result.
                    let claim = armed.claim.clone();
                    let lease = armed.lease.clone();
                    let deadline = armed.route_deadline;
                    let route_origin_commitment = armed.route_origin_commitment;
                    let dispatched = adapter.dispatch(armed, Arc::clone(journal)).await;
                    terminal_owner_health(adapter)?;
                    match dispatched {
                        Ok(result) => {
                            let bytes = result.encode();
                            if stop.closed.load(Ordering::SeqCst)
                                || !exchange_result(
                                    carrier, bytes, route_origin_commitment,
                                ).await?
                            {
                                pass_has_work = true;
                                pass_needs_retry = true;
                            } else {
                                acknowledged_results.insert(claim.claim_id());
                            }
                        }
                        // Only capacity rejection or proven pre-router
                        // failure can re-arm the SAME signed lease. Every
                        // router-entered/uncertain outcome remains Armed.
                        Err(error) if may_restore_after_zero_dispatch(error) => {
                            let claim_id = claim.claim_id();
                            recipient_db(journal, move |j, now| {
                                j.restore_lease_after_zero_dispatch_at(
                                    claim_id, &claim, &lease, deadline, || worker_journal_now(now),
                                )
                                // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex]
                                // If the exact lease expired while queued, keep
                                // Armed/non-executable instead of reviving it.
                                .or_else(|error| match error {
                                    RecipientJournalError::Expired => Ok(()),
                                    _ => Err(error),
                                })
                            }).await?;
                            pass_has_work = true;
                            pass_needs_retry = true;
                        }
                        Err(_) => {
                            pass_has_work = true;
                            pass_needs_retry = true;
                        }
                    }
                }
                RecipientRecovery::Result { claim_id, exact_bytes, route_origin_commitment } => {
                    if acknowledged_results.contains(&claim_id) { continue; }
                    if exchange_result(carrier, exact_bytes, route_origin_commitment).await? {
                        acknowledged_results.insert(claim_id);
                    } else {
                        pass_has_work = true;
                        pass_needs_retry = true;
                    }
                }
                RecipientRecovery::Ambiguous { .. } => {
                    pass_has_work = true;
                    pass_needs_retry = true;
                } // Never re-execute.
            }
        }
        if after.is_none() {
            // Historical Poll remains visible through its full evidence
            // horizon, not merely Claim freshness. Only a COMPLETE pass with
            // no unresolved jobs permits new polling, not old-route reissue.
            // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex]
            // Route absence retries; clock/owner errors leave this owned loop.
            if can_start_poll(
                after,
                pass_has_work,
                stop.closed.load(Ordering::SeqCst),
                permits_new_claims,
                carrier.has_current_pull_authority()?,
            ) {
                if let Some((origin, observed_at)) = carrier.prepare_new_claim_origin().await
                    .map_err(RecipientWorkerError::from)? {
                    if let Some(bytes) = prepare_new_claim(
                        journal, carrier.binding().0, identity, origin, observed_at, Arc::clone(stop),
                    ).await? {
                        if !stop.closed.load(Ordering::SeqCst) {
                            match exchange_poll(carrier, journal, bytes, origin, true).await? {
                            PollAttempt::Resolved => {}
                            PollAttempt::Progressed => pass_has_work = true,
                            PollAttempt::Pending => {
                                pass_has_work = true;
                                pass_needs_retry = true;
                            }
                            }
                        }
                    } else {
                        pass_needs_retry = true;
                    }
                } else {
                    pass_needs_retry = true;
                }
            }
            retry_streak = if pass_needs_retry {
                retry_streak.saturating_add(1)
            } else {
                0
            };
            wait = recipient_retry_delay(delay, retry_streak);
            pass_has_work = false;
            pass_needs_retry = false;
        }
        // Cancellation only interrupts idle waiting, never effectful futures.
        if !stop.closed.load(Ordering::SeqCst) {
            tokio::select! {
                _ = tokio::time::sleep(wait) => {},
                _ = stop.wake.notified() => {},
                _ = adapter.wait_for_failure() => {},
            }
        }
    }
    terminal_owner_health(adapter)
}

// [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Fatal local
// completion faults reach the existing required-worker supervisor/drain path.
// Normal stop remains Ok; protocol ambiguity alone is not an owner failure.
pub(super) fn terminal_owner_health(adapter: &ReverseOnionTerminalAdapter) -> Result<(), RecipientWorkerError> {
    if adapter.has_failed() { Err(RecipientWorkerError::Unavailable) } else { Ok(()) }
}

// [REVERSE-ONION-QUEUE-BACKPRESSURE 2026-10-05 by Codex] A full bounded
// recipient journal is temporary backpressure, not a fatal worker failure.
async fn prepare_new_claim(
    journal: &Arc<ReverseOnionRecipientJournal>,
    relay: [u8; 32],
    identity: &Arc<IdentityKeyPair>,
    route_origin_commitment: [u8; 32],
    route_observed_at: u64,
    stop: Arc<WorkerStop>,
) -> Result<Option<Vec<u8>>, RecipientWorkerError> {
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Claim creation
    // uses the same lane and samples time only after waiting for completion
    // writers, so a queued scan cannot move the durable clock past this claim.
    let identity = Arc::clone(identity);
    journal.run_blocking(move |journal| {
        if stop.closed.load(Ordering::SeqCst) { return Ok(None); }
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] The
        // journal lane wait cannot hide rollback below DNS/authority preflight.
        let now = recipient_preflight_now(route_observed_at, worker_now())
            .map_err(|_| RecipientJournalError::Unavailable)?;
        let mut id = [0; 16];
        rand::rngs::OsRng.try_fill_bytes(&mut id)
            .map_err(|_| RecipientJournalError::Unavailable)?;
        if id == [0; 16] { return Err(RecipientJournalError::Unavailable); }
        let expiry = now.checked_add(
            aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_CLAIM_LIFETIME_SECS,
        ).ok_or(RecipientJournalError::Unavailable)?;
        let claim = ReverseOnionFrameV1::claim(relay, id, now, expiry, &identity)
            .map_err(|_| RecipientJournalError::Rejected)?;
        if stop.closed.load(Ordering::SeqCst) { return Ok(None); }
        match journal.prepare_poll_with_admission_at(&claim, route_origin_commitment,
            || worker_journal_now(now), || recipient_intake_open(&stop.closed)) {
            Ok(bytes) => Ok(Some(bytes)),
            // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] A
            // signed Claim can age during SQL wait without becoming a fault.
            Err(RecipientJournalError::Capacity | RecipientJournalError::Busy
                | RecipientJournalError::Expired | RecipientJournalError::IntakeClosed) => Ok(None),
            Err(error) => Err(error),
        }
    }).await.map_err(|error| match error {
        RecipientJournalError::Ambiguous => RecipientWorkerError::Ambiguous,
        RecipientJournalError::Rejected => RecipientWorkerError::Rejected,
        _ => RecipientWorkerError::Unavailable,
    })
}

// [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex] A fresh poll
// requires both active operator mode and a current signed authority snapshot.
pub(super) fn can_start_poll(
    next_after: Option<[u8; 16]>,
    pass_has_work: bool,
    stopped: bool,
    permits_new_claims: bool,
    current_authority: bool,
) -> bool {
    next_after.is_none() && !pass_has_work && !stopped && permits_new_claims && current_authority
}

// [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] Only capacity rejection
// or an adapter proof that router entry never occurred may restore the same
// immutable lease. Ambiguous/post-entry outcomes stay non-reexecutable.
pub(super) fn may_restore_after_zero_dispatch(error: ReverseOnionTerminalError) -> bool {
    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Clock is
    // local owner failure even before router entry, not retry authorization.
    match error {
        ReverseOnionTerminalError::Busy | ReverseOnionTerminalError::ZeroDispatch => true,
        ReverseOnionTerminalError::Clock | ReverseOnionTerminalError::Rejected
            | ReverseOnionTerminalError::Ambiguous => false,
    }
}

// [REVERSE-ONION-RETRY-BACKOFF 2026-10-05 by Codex] Persistent ambiguity
// retries the same durable frame. The first failed pass waits the configured
// interval; later failures grow exponentially up to the bounded cap.
fn recipient_retry_delay(base: Duration, retry_streak: u8) -> Duration {
    let base_ms = u64::try_from(base.as_millis()).unwrap_or(u64::MAX);
    let cap_ms = u64::try_from(MAX_RECIPIENT_RETRY_BACKOFF.as_millis())
        .unwrap_or(u64::MAX)
        .max(base_ms);
    let shift = u32::from(retry_streak.saturating_sub(1).min(63));
    let factor = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
    Duration::from_millis(base_ms.saturating_mul(factor).min(cap_ms))
}

// [REVERSE-ONION-RECIPIENT-WORKER 2026-10-04 by Codex] Only canonical,
// locally persisted frames can reach carrier; no fresh timestamp/re-signing.
fn persisted_frame(bytes: &[u8], kind: ReverseOnionKindV1, now: u64)
    -> Result<Option<ReverseOnionFrameV1>, RecipientWorkerError> {
    // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Authenticate
    // historical bytes first, then distinguish normal expiry from corruption.
    let frame = ReverseOnionFrameV1::decode_for_recovery(bytes)
        .map_err(|_| RecipientWorkerError::Rejected)?;
    if frame.kind() != kind || frame.encode() != bytes { return Err(RecipientWorkerError::Rejected); }
    if kind == ReverseOnionKindV1::Claim {
        frame.verify_claim(frame.relay(), frame.immediate_recipient(), frame.issued_at())
            .map_err(|_| RecipientWorkerError::Rejected)?;
    }
    match frame.verify_recipient_retry(now) {
        Ok(()) => Ok(Some(frame)),
        Err(ReverseOnionError::Expired) => Ok(None),
        Err(_) => Err(RecipientWorkerError::Rejected),
    }
}

async fn exchange_poll<C: RecipientOutboundCarrier>(
    carrier: &C,
    journal: &Arc<ReverseOnionRecipientJournal>,
    bytes: Vec<u8>,
    route_origin_commitment: [u8; 32],
    require_live_pull_authority: bool,
)
    -> Result<PollAttempt, RecipientWorkerError> {
    let saved_bytes = bytes.clone();
    let decoded = tokio::task::spawn_blocking(move || persisted_frame(&bytes, ReverseOnionKindV1::Claim, worker_now()?))
        .await.map_err(|_| RecipientWorkerError::Ambiguous)?;
    // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] A bounded DB or
    // scheduler wait can age a valid page. Do not fail the worker, delete the
    // row, or issue a replacement poll in this pass; cleanup owns retirement.
    let Some(claim) = decoded? else { return Ok(PollAttempt::Pending); };
    let response = carrier.exchange(&claim, require_live_pull_authority, route_origin_commitment)
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex]
        .await.map_err(RecipientWorkerError::from)?;
    let (body, observed_at) = match response {
        ReverseOnionExchange::Response { status: 200, body, observed_at, .. } => (body, observed_at),
        _ => return Ok(PollAttempt::Pending),
    };
    let (relay, recipient, _) = carrier.binding();
    accept_poll_response(journal, saved_bytes, claim, relay, recipient, body, observed_at).await
}

// [PHALA-CONNECTED-REVERSE-LOOP 2026-10-07 by Codex] Share the worker's
// exact authenticated response-to-journal boundary with connected regression
// source. Transport still supplies only bounded HTTP-200 bytes, never a lease
// verification flag, route payload, deadline, or replacement Claim.
pub(super) async fn accept_poll_response(
    journal: &Arc<ReverseOnionRecipientJournal>, saved_bytes: Vec<u8>,
    claim: ReverseOnionFrameV1, relay: [u8; 32], recipient: [u8; 32], body: Vec<u8>,
    observed_at: u64,
) -> Result<PollAttempt, RecipientWorkerError> {
    let claim_id = claim.claim_id();
    let claim_for_receipt = claim;
    let claim_for_lease = claim_for_receipt.clone();
    recipient_db(journal, move |j, now| {
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] The DB lane
        // must retain HTTP completion's floor before verifying or mutating.
        let now = recipient_preflight_now(observed_at, Ok(now))
            .map_err(|_| RecipientJournalError::Unavailable)?;
        if ReverseOnionNoWorkReceiptV1::decode_for_claim(
            &body, &claim_for_receipt, relay, recipient, now,
        ).is_ok() {
            return poll_persistence_outcome(
                j.complete_no_work_poll_at(claim_id, &saved_bytes, &body, now,
                    || worker_journal_now(now)),
                PollAttempt::Resolved,
            );
        }
        let Ok(lease) = ReverseOnionFrameV1::decode(&body, now) else {
            return Ok(PollAttempt::Pending);
        };
        if lease.encode() != body { return Ok(PollAttempt::Pending); }
        let Ok(proof) = lease.verify_recipient_lease(&claim_for_lease, relay, recipient, now) else {
            return Ok(PollAttempt::Pending);
        };
        poll_persistence_outcome(j.record_relay_lease_at(proof,
            || worker_journal_now(now)), PollAttempt::Progressed)
    }).await
}

// [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Preserve the
// exact poll when an authenticated response expires during SQL wait. Clock
// rollback, altered bindings and local storage faults are never downgraded.
fn poll_persistence_outcome(
    result: Result<(), RecipientJournalError>, success: PollAttempt,
) -> Result<PollAttempt, RecipientJournalError> {
    match result {
        Ok(()) => Ok(success),
        Err(RecipientJournalError::Expired) => Ok(PollAttempt::Pending),
        Err(error) => Err(error),
    }
}

async fn exchange_result<C: RecipientOutboundCarrier>(
    carrier: &C,
    bytes: Vec<u8>,
    route_origin_commitment: Option<[u8; 32]>,
) -> Result<bool, RecipientWorkerError> {
    let Some(route_origin_commitment) = route_origin_commitment else {
        return Ok(false);
    };
    let decoded = tokio::task::spawn_blocking(move || persisted_frame(&bytes, ReverseOnionKindV1::Result, worker_now()?))
        .await.map_err(|_| RecipientWorkerError::Ambiguous)?;
    let Some(result) = decoded? else { return Ok(false); };
    let response = carrier.exchange(&result, false, route_origin_commitment)
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex]
        .await.map_err(RecipientWorkerError::from)?;
    let ReverseOnionExchange::Response { status, body, authenticated_https, observed_at } = response
        else { return Ok(false); };
    // [REVERSE-ONION-RESULT-CUSTODY-ACK 2026-10-05 by Codex] An exact echo
    // counts only over certificate-verified HTTPS. Plain HTTP can be echoed
    // by an intermediary and must leave the exact Result eligible for retry.
    let timeout = carrier.binding().2;
    tokio::task::spawn_blocking(move || {
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Scheduling
        // after HTTPS cannot erase the completion floor or extend Result life.
        let now = recipient_preflight_now(observed_at, worker_now())
            .map_err(RecipientWorkerError::from)?;
        if recipient_exchange_timeout(&result, now, timeout)
            .map_err(RecipientWorkerError::from)?.is_none() { return Ok(false); }
        Ok(authenticated_result_ack(&result, status, authenticated_https, &body))
    })
        .await.map_err(|_| RecipientWorkerError::Unavailable)?
}

fn authenticated_result_ack(
    result: &ReverseOnionFrameV1,
    status: u16,
    authenticated_https: bool,
    body: &[u8],
) -> bool {
    status == reqwest::StatusCode::OK.as_u16()
        && authenticated_https
        && exact_result_echo(result, body)
}

fn exact_result_echo(result: &ReverseOnionFrameV1, body: &[u8]) -> bool {
    ReverseOnionFrameV1::decode_for_recovery(body).is_ok_and(|echo| {
        echo.encode().as_slice() == body && result.require_exact_retry(&echo).is_ok()
    })
}

#[cfg(test)]
mod recipient_worker_tests {
    use super::*;

    // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Authored,
    // not run: later success cannot heal a failed local sample in one attempt.
    #[test]
    fn attempt_observations_keep_latest_floor_and_sticky_local_failure() {
        let observation = ReverseOnionLocalObservation::new(100);
        assert_eq!(observation.observe(Ok(110)), Ok(110));
        assert_eq!(observation.floor(), Ok(110));
        assert_eq!(observation.observe(Ok(105)), Err(()));
        assert_eq!(observation.observe(Ok(120)), Err(()));
        assert_eq!(observation.floor(), Err(()));
        let failed = ReverseOnionLocalObservation::new(100);
        assert_eq!(failed.observe(Err(())), Err(()));
        assert_eq!(failed.observe(Ok(100)), Err(()));
        let independent = ReverseOnionLocalObservation::new(100);
        assert_eq!(independent.observe(Ok(100)), Ok(100));
    }

    // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Authored,
    // not run: the real poll completion lane rejects an impossible HTTP floor
    // without retiring/replacing exact custody; a healthy receipt resolves it.
    #[cfg(unix)]
    #[tokio::test]
    async fn poll_completion_keeps_http_clock_floor_through_lane_wait_and_restart() {
        for rollback in [false, true] {
            let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
            let recipient = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
            let directory = tempfile::Builder::new().prefix("phala-poll-completion-clock-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("recipient.sqlite");
            let at = worker_now().unwrap();
            let journal = Arc::new(ReverseOnionRecipientJournal::open(&path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: RECIPIENT_LOCAL_BYTES }, at).unwrap());
            let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [11; 16], at, at + 30, &recipient).unwrap();
            let exact = journal.prepare_poll(&claim, [9; 32], at).unwrap();
            let receipt = ReverseOnionNoWorkReceiptV1::issue_no_work(&claim, at, &relay).unwrap().encode();
            let held = journal.blocking_operation_lane().acquire_owned().await.unwrap();
            let mut completion = Box::pin(accept_poll_response(&journal, exact.clone(), claim,
                relay.public_key_bytes(), recipient.public_key_bytes(), receipt,
                if rollback { u64::MAX } else { at }));
            assert!(futures::poll!(completion.as_mut()).is_pending());
            drop(held);
            assert_eq!(completion.await,
                if rollback { Err(RecipientWorkerError::Unavailable) } else { Ok(PollAttempt::Resolved) });
            drop(journal);
            let reopened = ReverseOnionRecipientJournal::open_existing(&path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now().unwrap()).unwrap();
            let page = reopened.resume(None, 64, worker_now().unwrap()).unwrap();
            if rollback {
                assert_eq!(page.items.len(), 1);
                assert!(matches!(&page.items[0], RecipientRecovery::Poll { exact_bytes, .. } if exact_bytes == &exact));
            } else { assert!(page.items.is_empty()); }
        }
    }
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion_reply::{
        encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
        ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authored,
    // not run: response expiry preserves retry, all other error classes survive.
    #[test]
    fn only_forward_expiry_downgrades_poll_persistence_to_pending() {
        for success in [PollAttempt::Resolved, PollAttempt::Progressed] {
            assert_eq!(poll_persistence_outcome(Ok(()), success), Ok(success));
            assert_eq!(poll_persistence_outcome(Err(RecipientJournalError::Expired), success),
                Ok(PollAttempt::Pending));
            for error in [RecipientJournalError::Rejected, RecipientJournalError::Conflict,
                RecipientJournalError::Busy, RecipientJournalError::Capacity,
                RecipientJournalError::Ambiguous, RecipientJournalError::Corrupt,
                RecipientJournalError::Unavailable, RecipientJournalError::IntakeClosed] {
                assert_eq!(poll_persistence_outcome(Err(error), success), Err(error));
            }
        }
    }
    const NOW: u64 = 1_800_000_000;

    // [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] Authored, not
    // run: the actual startup owner must reject absent/unowned recovery state,
    // publish failure rather than READY, and drain its adapter without I/O.
    #[cfg(unix)]
    #[tokio::test]
    async fn actual_recovery_start_rejects_missing_and_unowned_history() {
        use std::os::unix::fs::PermissionsExt;
        for unowned in [false, true] {
            let directory = tempfile::Builder::new().prefix("worker-recovery-boot-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = if unowned { directory.path().join("recipient.sqlite") }
                else { directory.path().join("missing/recipient.sqlite") };
            let before = if unowned {
                let db = rusqlite::Connection::open(&path).unwrap();
                db.execute_batch("CREATE TABLE removed(value BLOB); DROP TABLE removed;").unwrap();
                drop(db);
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
                Some(std::fs::read(&path).unwrap())
            } else { None };
            let recipient = Arc::new(IdentityKeyPair::from_bytes(&[77; 32]).unwrap());
            let relay = IdentityKeyPair::from_bytes(&[78; 32]).unwrap();
            let mut config = ReverseOnionConfig::default();
            config.recipient.enabled = true;
            config.recipient.recovery_only = true;
            config.recipient.relay_node_id = hex::encode(relay.public_key_bytes());
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex] Pass
            // origin syntax policy to reach the intended disk-open failure.
            // Startup fails before DNS or any transport is attempted.
            config.recipient.relay_endpoint = "https://relay.example.net".into();
            config.recipient.state_db_path = path.display().to_string();
            let adapter = Arc::new(ReverseOnionTerminalAdapter::new(axum::Router::new(),
                recipient.clone(), relay.public_key_bytes(), Duration::from_secs(1)).unwrap());
            let worker = ReverseOnionRecipientWorker::start(&config, recipient,
                Arc::new(PeerStore::new()), adapter.clone()).unwrap();
            assert_eq!(tokio::time::timeout(Duration::from_secs(30), worker.wait_ready()).await.unwrap(),
                Err(RecipientWorkerError::Unavailable));
            assert_eq!(tokio::time::timeout(Duration::from_secs(30), worker.wait_completion()).await.unwrap(),
                Err(RecipientWorkerError::Unavailable));
            assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
            assert!(!adapter.has_failed());
            if let Some(before) = before { assert_eq!(std::fs::read(&path).unwrap(), before); }
            else {
                assert!(!path.exists());
                assert!(!path.parent().unwrap().exists());
            }
        }
    }

    // [REVERSE-ONION-RESULT-CUSTODY-ACK 2026-10-05 by Codex] Authored, not run.
    #[test]
    fn custody_ack_requires_exact_result_echo_and_verified_https() {
        let relay = IdentityKeyPair::from_bytes(&[41; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &recipient,
        ).unwrap();
        let (_, kem) = recipient.to_x25519();
        let envelope = build_onion_envelope(
            &[OnionHop { node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes() }],
            b"opaque request", [2; 16], 1, NOW, &relay,
        ).unwrap();
        let lease = ReverseOnionFrameV1::lease(
            &claim, &envelope, [3; 16], NOW + 120, NOW, &relay,
        ).unwrap();
        let (request, _) = OnionReplySession::prepare_source_sealed(
            lease.route_id(), recipient.public_key_bytes(),
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"operation".to_vec(),
        ).unwrap();
        let sealed = seal_onion_reply(lease.route_id(), &request, b"opaque result", &recipient).unwrap();
        let result = ReverseOnionFrameV1::result(
            &claim, &lease, &encode_onion_sealed_response(&sealed).unwrap(),
            NOW + 120, NOW + 1, &recipient,
        ).unwrap();
        let exact = result.encode();
        // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] A late
        // persisted Result is retained locally but never posted after grace.
        assert!(persisted_frame(&exact, ReverseOnionKindV1::Result, NOW + 419).unwrap().is_some());
        assert!(persisted_frame(&exact, ReverseOnionKindV1::Result, NOW + 420).unwrap().is_none());
        assert_eq!(recipient_exchange_timeout(&result, NOW + 419, Duration::from_secs(10)),
            Ok(Some(Duration::from_secs(1))));
        assert_eq!(recipient_exchange_timeout(&result, NOW + 420, Duration::from_secs(10)), Ok(None));
        assert!(exact_result_echo(&result, &exact));
        assert!(authenticated_result_ack(&result, 200, true, &exact));
        assert!(!authenticated_result_ack(&result, 200, false, &exact));
        assert!(!authenticated_result_ack(&result, 201, true, &exact));
        let mut altered = exact.clone();
        *altered.last_mut().unwrap() ^= 1;
        assert!(!exact_result_echo(&result, &altered));
        assert!(!exact_result_echo(&result, &[0; 8]));
        assert!(!exact_result_echo(&result, b"not the stored frame"));
    }

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
            let restored = persisted_frame(&bytes, ReverseOnionKindV1::Claim, now).unwrap().unwrap();
            assert_eq!(restored.encode(), bytes);
            assert_eq!(restored.relay(), relay.public_key_bytes());
            assert_eq!(restored.immediate_recipient(), recipient.public_key_bytes());
        }
        // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Model a DNS
        // wait crossing the deadline: preflight was valid, final send is not.
        assert_eq!(recipient_exchange_timeout(&claim, NOW + 929, Duration::from_secs(10)),
            Ok(Some(Duration::from_secs(1))));
        assert_eq!(recipient_exchange_timeout(&claim, NOW + 930, Duration::from_secs(10)), Ok(None));
        assert_eq!(recipient_exchange_timeout(&claim, NOW - 1, Duration::from_secs(10)),
            Err(ReverseOnionPreflightError::Frame));
        assert!(persisted_frame(&bytes, ReverseOnionKindV1::Claim, NOW + 930).unwrap().is_none());
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
        assert!(!can_start_poll(Some([1; 16]), false, false, true, true));
        assert!(!can_start_poll(None, true, false, true, true));
        assert!(!can_start_poll(None, false, true, true, true));
        assert!(can_start_poll(None, false, false, true, true));
        assert!(!can_start_poll(None, false, false, true, false));
        assert!(!can_start_poll(None, false, false, false, true));
        // Empty later page cannot erase unresolved work found on an earlier one.
        let mut pass_has_work = true;
        pass_has_work |= false;
        assert!(!can_start_poll(None, pass_has_work, false, true, true));
    }

    // [REVERSE-ONION-RETRY-BACKOFF 2026-10-05 by Codex] Authored, not run.
    #[test]
    fn unresolved_retries_back_off_with_a_bounded_cap() {
        let base = Duration::from_millis(250);
        assert_eq!(recipient_retry_delay(base, 0), base);
        assert_eq!(recipient_retry_delay(base, 1), base);
        assert_eq!(recipient_retry_delay(base, 2), Duration::from_millis(500));
        assert_eq!(recipient_retry_delay(base, 4), Duration::from_secs(2));
        assert_eq!(recipient_retry_delay(base, u8::MAX), MAX_RECIPIENT_RETRY_BACKOFF);
        let operator_minimum = Duration::from_secs(60);
        assert_eq!(recipient_retry_delay(operator_minimum, u8::MAX), operator_minimum);
    }

    #[test]
    fn disabled_carrier_rejects_before_runtime_task_or_database_creation() {
        assert!(matches!(ReverseOnionHttpCarrier::new(
            &ReverseOnionConfig::default(), [42; 32], Arc::new(PeerStore::new()),
            Arc::new(AtomicBool::new(false))),
            Err(ReverseOnionPreflightError::Disabled)));
    }

    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
    // this is the production post-construction gate, not a second predicate.
    #[test]
    fn recipient_final_send_gate_observes_stop_expiry_and_clock_faults() {
        let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [18; 16], 100, 130, &recipient).unwrap();
        let stopped = AtomicBool::new(false);
        let timeout = Duration::from_secs(5);
        assert_eq!(recipient_send_admission(&stopped, &claim, 100, Ok(101), timeout), Ok(Some(timeout)));
        let deadline = claim.recipient_retry_deadline().unwrap();
        assert_eq!(recipient_send_admission(&stopped, &claim, 100, Ok(deadline - 1), timeout),
            Ok(Some(Duration::from_secs(1))));
        assert_eq!(recipient_send_admission(&stopped, &claim, 100,
            Ok(deadline), timeout), Ok(None));
        stopped.store(true, Ordering::SeqCst);
        assert_eq!(recipient_send_admission(&stopped, &claim, 100, Ok(101), timeout), Ok(None));
        assert_eq!(recipient_send_admission(&stopped, &claim, 100, Ok(99), timeout),
            Err(ReverseOnionPreflightError::Clock));
        assert_eq!(recipient_send_admission(&stopped, &claim, 100, Err(RecipientWorkerError::Unavailable), timeout),
            Err(ReverseOnionPreflightError::Clock));
    }

    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Authored,
    // not run: exercise the production observation/floor check with known
    // local failures, separately from ordinary route absence and frame expiry.
    #[tokio::test]
    async fn recipient_route_preflight_does_not_downgrade_local_clock_faults() {
        assert_eq!(recipient_preflight_now(100, Ok(100)), Ok(100));
        assert_eq!(recipient_preflight_now(100, Ok(99)), Err(ReverseOnionPreflightError::Clock));
        assert_eq!(recipient_preflight_now(100, Err(RecipientWorkerError::Unavailable)),
            Err(ReverseOnionPreflightError::Clock));
        assert_eq!(RecipientWorkerError::from(ReverseOnionPreflightError::Clock), RecipientWorkerError::Unavailable);
        assert_eq!(RecipientWorkerError::from(ReverseOnionPreflightError::Frame), RecipientWorkerError::Rejected);
        let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
        let carrier = ReverseOnionHttpCarrier {
            relay: relay.public_key_bytes(), recipient: recipient.public_key_bytes(),
            relay_origin_commitment: [7; 32], peers: Arc::new(PeerStore::new()),
            timeout: Duration::from_secs(1), stopped: Arc::new(AtomicBool::new(false)),
        };
        for live in [false, true] {
            // Empty PeerStore is a nonfatal zero-send pause. A known failing
            // clock floor must reject before that absence can hide the fault.
            assert!(carrier.current_target("/api/chat/peer/reverse-onion/claim", live, None, None, 0)
                .await.unwrap().is_none());
            assert!(matches!(carrier.current_target("/api/chat/peer/reverse-onion/claim", live, None, None, u64::MAX)
                .await, Err(ReverseOnionPreflightError::Clock)));
        }
        assert_eq!(carrier.has_current_pull_authority(), Ok(false));
        assert!(!carrier.stopped.load(Ordering::SeqCst));
    }

    // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
    // constructing an exchange future does not freeze a successful stop check.
    #[tokio::test]
    async fn recipient_exchange_first_poll_uses_the_shared_stop_gate() {
        let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
        let at = worker_now().unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [19; 16], at, at + 30, &recipient).unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let carrier = ReverseOnionHttpCarrier {
            relay: relay.public_key_bytes(), recipient: recipient.public_key_bytes(),
            relay_origin_commitment: [7; 32], peers: Arc::new(PeerStore::new()),
            timeout: Duration::from_secs(1), stopped: stopped.clone(),
        };
        let attempt = carrier.exchange(&claim, true, [7; 32]);
        stopped.store(true, Ordering::SeqCst);
        assert!(matches!(attempt.await, Ok(ReverseOnionExchange::Deferred)));
    }

    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Authored, not
    // run: call the production final gate with a controlled clock. Appraisal
    // expiry, grant replacement and descriptor renewal cannot silently replace
    // the pre-DNS selection. No DNS/HTTP or recipient execution is involved.
    #[cfg(unix)]
    #[test]
    fn recipient_final_route_gate_preserves_selected_authority_and_recovery_mode() {
        use aeronyx_core::protocol::{NodeProtocolFeature, onion::OnionRoutePurpose};
        use aeronyx_core::protocol::discovery::SignedPrivateOnionRecipientAuthorizationV1;
        let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
        let (relay, recipient, _) = fixture.policy_parts();
        let now = fixture.now();
        let relay_key = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
        let recipient_key = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
        // [PHALA-PULL-ROLE-REPAIR 2026-10-08 by Codex] Private Pull must
        // reach the final route gate without advertising public replica service.
        let mut recipient_body = recipient.descriptor;
        recipient_body.capabilities.retain(|capability|
            *capability != aeronyx_core::protocol::NodeCapability::BlindVaultReplica);
        let recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
        let relay = SignedNodeDescriptor::sign(relay.descriptor.with_protocol_features(
            [NodeProtocolFeature::PhalaNodeAttestationV1],
        ), &relay_key).unwrap();
        let grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now, now + 600, &recipient_key,
        ).unwrap();
        let peers = Arc::new(PeerStore::new());
        peers.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
        peers.upsert_verified_from_source(recipient.clone(), now, "test_pin").unwrap();
        peers.remember_issued_private_onion_authorization(grant, recipient.node_id(), now).unwrap();
        peers.configure_phala_attested_peer_routes(true, 1);
        assert!(peers.record_phala_peer_attestation(&relay, now));
        let selected = RecipientRouteSnapshot::Pull(peers.current_private_onion_pull_authority_snapshot(
            &relay.node_id(), &recipient.node_id(), now,
        ).unwrap());
        let recovery = RecipientRouteSnapshot::Recovery(relay.clone());
        let stopped = Arc::new(AtomicBool::new(false));
        let carrier = ReverseOnionHttpCarrier {
            relay: relay.node_id(), recipient: recipient.node_id(),
            relay_origin_commitment: crate::api::reverse_onion_origin_commitment(
                relay.descriptor.public_endpoint.as_deref().unwrap(),
            ).unwrap(),
            peers: peers.clone(), timeout: Duration::from_secs(5), stopped: stopped.clone(),
        };
        let claim = ReverseOnionFrameV1::claim(relay.node_id(), [20; 16], now, now + 30, &recipient_key).unwrap();
        // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] Keep each
        // observed sample even when authority rejects before HTTP entry.
        let observation = ReverseOnionLocalObservation::new(now);
        assert!(carrier.final_send_admission_at(&claim, &observation, &selected, || Ok(now)).unwrap().is_some());
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &selected, || Ok(now + 2)), Ok(None));
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &recovery, || Ok(now + 2)), Ok(None));
        assert!(peers.record_phala_peer_attestation(&relay, now + 2));
        assert!(carrier.final_send_admission_at(&claim, &observation, &selected, || Ok(now + 2)).unwrap().is_some());
        let renewed_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now + 3, now + 603, &recipient_key,
        ).unwrap();
        peers.remember_issued_private_onion_authorization(renewed_grant, recipient.node_id(), now + 3).unwrap();
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &selected, || Ok(now + 3)), Ok(None));
        assert!(carrier.final_send_admission_at(&claim, &observation, &recovery, || Ok(now + 3)).unwrap().is_some());

        let mut renewed_body = relay.descriptor.clone();
        renewed_body.sequence += 1;
        renewed_body.issued_at = now + 4;
        let renewed_relay = SignedNodeDescriptor::sign(renewed_body, &relay_key).unwrap();
        peers.upsert_verified_from_source(renewed_relay.clone(), now + 4, "test_pin").unwrap();
        assert!(!peers.record_phala_peer_attestation(&relay, now + 4));
        assert!(peers.record_phala_peer_attestation(&renewed_relay, now + 4));
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &recovery, || Ok(now + 4)), Ok(None));
        let renewed_recovery = RecipientRouteSnapshot::Recovery(renewed_relay);
        assert!(carrier.final_send_admission_at(&claim, &observation, &renewed_recovery, || Ok(now + 4)).unwrap().is_some());
        stopped.store(true, Ordering::SeqCst);
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &renewed_recovery, || Ok(now + 4)), Ok(None));
        assert_eq!(carrier.final_send_admission_at(&claim, &observation, &renewed_recovery, || Ok(now - 1)),
            Err(ReverseOnionPreflightError::Clock));
    }

    // [PHALA-ROTATED-CUSTODY-RECOVERY 2026-10-08 by Codex] Authored,
    // not run: Result retry may use the same pinned relay's new descriptor
    // without a new P grant, but an old preflight snapshot stays revoked.
    #[cfg(unix)]
    #[test]
    fn result_retry_after_renewal_does_not_require_fresh_pull_grant() {
        let f = crate::services::reverse_onion_source::tests::Fixture::new();
        let now = f.now();
        let journal = f.open(now);
        let result = f.ready(&journal);
        let exact_result = result.encode();
        let (relay, recipient, grant) = f.policy_parts();
        let peers = Arc::new(PeerStore::new());
        peers.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
        peers.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
        peers.upsert_verified_from_source(recipient.clone(), now, "test_pin").unwrap();
        peers.remember_issued_private_onion_authorization(grant, recipient.node_id(), now).unwrap();
        let original_pull = RecipientRouteSnapshot::Pull(
            peers.current_private_onion_pull_authority_snapshot(
                &relay.node_id(), &recipient.node_id(), now,
            ).unwrap(),
        );
        let original_recovery = RecipientRouteSnapshot::Recovery(relay.clone());
        let carrier = ReverseOnionHttpCarrier {
            relay: relay.node_id(), recipient: recipient.node_id(),
            relay_origin_commitment: crate::api::reverse_onion_origin_commitment(
                relay.descriptor.public_endpoint.as_deref().unwrap(),
            ).unwrap(),
            peers: Arc::clone(&peers), timeout: Duration::from_secs(5),
            stopped: Arc::new(AtomicBool::new(false)),
        };
        let (renewed_relay, renewed_recipient, renewed_grant) = f.renewed_policy_parts(now + 31);
        peers.upsert_verified_from_source(renewed_relay.clone(), now + 31, "test_pin").unwrap();
        peers.upsert_verified_from_source(renewed_recipient, now + 31, "test_pin").unwrap();
        assert!(carrier.current_pull_authority_epoch(now + 31).is_none());
        let observation = ReverseOnionLocalObservation::new(now + 31);
        assert_eq!(carrier.final_send_admission_at(&result, &observation,
            &original_pull, || Ok(now + 31)), Ok(None));
        assert_eq!(carrier.final_send_admission_at(&result, &observation,
            &original_recovery, || Ok(now + 31)), Ok(None));
        let renewed_recovery = RecipientRouteSnapshot::Recovery(renewed_relay);
        assert!(carrier.final_send_admission_at(&result, &observation,
            &renewed_recovery, || Ok(now + 31)).unwrap().is_some());
        // Positive control for fresh intake: only the matching new grant
        // restores Pull authority, with no alteration of the stored Result.
        peers.remember_issued_private_onion_authorization(renewed_grant,
            recipient.node_id(), now + 32).unwrap();
        assert!(carrier.current_pull_authority_epoch(now + 32).is_some());
        assert_eq!(result.encode(), exact_result);
        carrier.stopped.store(true, Ordering::SeqCst);
        assert_eq!(carrier.final_send_admission_at(&result, &observation,
            &renewed_recovery, || Ok(now + 32)), Ok(None));
    }

    // [PHALA-RECIPIENT-RELAY-ORIGIN-PIN 2026-10-06 by Codex] Authored only;
    // no build or test execution is authorized in the current development phase.
    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Exercise the shared
    // transport gate rather than a separate commitment equality helper.
    #[test]
    fn recipient_transport_origin_requires_the_configured_pin() {
        let configured = crate::api::reverse_onion_origin_commitment(
            "https://relay.example.net",
        )
        .unwrap();
        assert_eq!(crate::api::reverse_onion_pinned_origin(
            "https://RELAY.example.net:443/api/discovery/gossip", configured,
        ).unwrap(), configured);
        for endpoint in ["https://other.example.net", "https://relay.example.net:8443",
            "http://relay.example.net", "https://relay.internal"] {
            assert!(crate::api::reverse_onion_pinned_origin(endpoint, configured).is_err());
        }
    }

    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Authored, not
    // run: a queued fresh poll must observe stop after the shared-lane wait.
    #[cfg(unix)]
    #[tokio::test]
    async fn stopped_worker_does_not_prepare_fresh_claim_after_journal_lane_wait() {
        let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
        let recipient = Arc::new(IdentityKeyPair::from_bytes(&[74; 32]).unwrap());
        let directory = tempfile::Builder::new().prefix("phala-claim-stop-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let journal = Arc::new(ReverseOnionRecipientJournal::open(
            &directory.path().join("recipient.sqlite"), relay.public_key_bytes(), recipient.public_key_bytes(),
            RecipientJournalLimits { max_entries: 8, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now().unwrap(),
        ).unwrap());
        let stop = Arc::new(WorkerStop { closed: Arc::new(AtomicBool::new(false)), wake: Notify::new() });
        let held = journal.blocking_operation_lane().acquire_owned().await.unwrap();
        let mut preparing = Box::pin(prepare_new_claim(
            &journal, relay.public_key_bytes(), &recipient, [9; 32], worker_now().unwrap(), Arc::clone(&stop),
        ));
        assert!(futures::poll!(preparing.as_mut()).is_pending());
        stop.closed.store(true, Ordering::SeqCst);
        drop(held);
        assert!(preparing.await.unwrap().is_none());
        let page = recipient_db(&journal, |journal, now| journal.resume(None, 64, now)).await.unwrap();
        assert!(page.items.is_empty());
    }

    // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
    // Authored, not run: the actual Claim producer returns backpressure, not
    // outbound bytes or a worker fault, unless eventual Result custody fits.
    #[cfg(unix)]
    #[tokio::test]
    async fn fresh_claim_producer_reserves_result_capacity_before_returning_bytes() {
        use crate::services::reverse_onion_recipient::RECIPIENT_RESERVED_JOB_BYTES;
        for funded in [false, true] {
            let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
            let recipient = Arc::new(IdentityKeyPair::from_bytes(&[74; 32]).unwrap());
            let directory = tempfile::Builder::new().prefix("phala-claim-capacity-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("recipient.sqlite");
            let max_bytes = RECIPIENT_RESERVED_JOB_BYTES - if funded { 0 } else { 1 };
            let journal = Arc::new(ReverseOnionRecipientJournal::open(&path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes }, worker_now().unwrap()).unwrap());
            let stop = Arc::new(WorkerStop { closed: Arc::new(AtomicBool::new(false)), wake: Notify::new() });
            let exact = prepare_new_claim(&journal, relay.public_key_bytes(), &recipient,
                [9; 32], worker_now().unwrap(), Arc::clone(&stop)).await.unwrap();
            assert_eq!(exact.is_some(), funded);
            assert!(!stop.closed.load(Ordering::SeqCst));
            if funded {
                assert!(prepare_new_claim(&journal, relay.public_key_bytes(), &recipient,
                    [9; 32], worker_now().unwrap(), Arc::clone(&stop)).await.unwrap().is_none());
            }
            drop(journal);
            let recovered = ReverseOnionRecipientJournal::open_existing(&path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes }, worker_now().unwrap()).unwrap();
            let page = recovered.resume(None, 64, worker_now().unwrap()).unwrap();
            if let Some(exact) = exact {
                assert!(matches!(page.items.as_slice(),
                    [RecipientRecovery::Poll { exact_bytes, .. }] if exact_bytes == &exact));
            } else { assert!(page.items.is_empty()); }
        }
    }

    // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Authored,
    // not run: a route time floor survives the actual journal lane wait.
    // A deliberately impossible floor rejects before signing or SQL mutation;
    // the healthy control persists exactly one Claim through restart audit.
    #[cfg(unix)]
    #[tokio::test]
    async fn fresh_claim_keeps_route_clock_floor_through_journal_admission_and_restart() {
        for rollback in [false, true] {
            let relay = IdentityKeyPair::from_bytes(&[73; 32]).unwrap();
            let recipient = Arc::new(IdentityKeyPair::from_bytes(&[74; 32]).unwrap());
            let directory = tempfile::Builder::new().prefix("phala-claim-clock-floor-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("recipient.sqlite");
            let journal = Arc::new(ReverseOnionRecipientJournal::open(
                &path, relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now().unwrap(),
            ).unwrap());
            let stop = Arc::new(WorkerStop { closed: Arc::new(AtomicBool::new(false)), wake: Notify::new() });
            let floor = if rollback { u64::MAX } else { worker_now().unwrap() };
            let held = journal.blocking_operation_lane().acquire_owned().await.unwrap();
            let mut preparing = Box::pin(prepare_new_claim(
                &journal, relay.public_key_bytes(), &recipient, [9; 32], floor, Arc::clone(&stop),
            ));
            assert!(futures::poll!(preparing.as_mut()).is_pending());
            drop(held);
            let outcome = preparing.await;
            let exact_claim = if rollback {
                assert_eq!(outcome.err(), Some(RecipientWorkerError::Unavailable));
                None
            } else {
                Some(outcome.unwrap().unwrap())
            };
            // The worker, not this journal helper, owns fault publication.
            assert!(!stop.closed.load(Ordering::SeqCst));
            drop(journal);
            let reopened = ReverseOnionRecipientJournal::open_existing(
                &path, relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: RECIPIENT_LOCAL_BYTES }, worker_now().unwrap(),
            ).unwrap();
            let page = reopened.resume(None, 64, worker_now().unwrap()).unwrap();
            match exact_claim {
                Some(expected) => {
                    assert_eq!(page.items.len(), 1);
                    assert!(matches!(&page.items[0], RecipientRecovery::Poll {
                        exact_bytes, route_origin_commitment, ..
                    } if exact_bytes == &expected && route_origin_commitment == &[9; 32]));
                }
                None => assert!(page.items.is_empty()),
            }
        }
    }

    #[tokio::test]
    async fn final_ready_snapshot_rejects_stale_successful_recipient_wait() {
        // [PHALA-READY-PUBLICATION 2026-10-07 by Codex] Authored, not run.
        // A controlled owned task isolates completion/readiness signals; it
        // does not stand in for journal audit or real network startup.
        use super::super::runtime_supervision::{self, PreReadyRuntimeDecision as D};
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Known
        // failure must be observable with the accepted task still blocked.
        for mode in 0..6 {
            let (ready_tx, ready) = watch::channel(WorkerReady::Ready);
            let mut ready_tx = Some(ready_tx);
            let (completed_tx, completed) = watch::channel(());
            let (failed_tx, failure) = watch::channel(false);
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let mut release_tx = Some(release_tx);
            let task = tokio::spawn(async move {
                let _completion_guard = completed_tx;
                let _ = release_rx.await;
                Ok(())
            });
            let worker = Arc::new(ReverseOnionRecipientWorker {
                stop: Arc::new(WorkerStop { closed: Arc::new(AtomicBool::new(false)), wake: Notify::new() }),
                ready, completed, failure, task: Mutex::new(Some(task)),
                drain_waiter: tokio::sync::Mutex::new(()), finished: Mutex::new(None),
            });
            let owner = RecipientServerLifecycle::new();
            owner.active.store(true, Ordering::SeqCst);
            owner.install(worker.clone()).unwrap();
            owner.verify_ready().await.unwrap();
            let (_failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
            let shutdown = AtomicBool::new(false);
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut failure_rx, &shutdown, None, Some(&owner), false,
            ), D::Ready);
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut failure_rx, &shutdown, None, Some(&owner), true,
            ), D::Failed(runtime_supervision::reverse_onion_recipient_worker_exited()));
            match mode {
                0 => owner.request_stop(),
                1 => worker.request_stop(),
                2 => {
                    release_tx.take().unwrap().send(()).unwrap();
                    worker.wait_for_task_exit().await;
                }
                3 => drop(ready_tx.take()),
                _ => {
                    report_recipient_worker_failure(&worker.stop, &failed_tx);
                    if mode == 5 { owner.request_stop(); }
                    assert!(owner.has_failed());
                    assert!(!worker_task_has_exited(&worker.completed));
                    assert!(owner.wait_for_unexpected_worker_failure_or_exit().await);
                    let mut repeated = Box::pin(owner.wait_for_unexpected_worker_failure_or_exit());
                    assert_eq!(futures::poll!(repeated.as_mut()), Poll::Ready(true));
                    drop(repeated);
                    let mut drain = Box::pin(owner.drain());
                    assert!(futures::poll!(drain.as_mut()).is_pending());
                    drop(drain);
                }
            }
            assert!(owner.verify_ready_now().is_err());
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut failure_rx, &shutdown, None, Some(&owner), false,
            ), if mode == 0 { D::Stopped }
                else { D::Failed(runtime_supervision::reverse_onion_recipient_worker_exited()) });
            if let Some(release) = release_tx.take() { let _ = release.send(()); }
            owner.drain().await.unwrap();
        }
    }

    #[tokio::test]
    async fn cancelled_shutdown_retains_handle_and_waits_for_blocking_completion() {
        // [PHALA-RECIPIENT-ENTRY-STOP 2026-10-07 by Codex] Exercise the
        // worker shutdown method against the real adapter's shared entry gate.
        let identity = Arc::new(IdentityKeyPair::from_bytes(&[77; 32]).unwrap());
        let relay = IdentityKeyPair::from_bytes(&[78; 32]).unwrap();
        let adapter = ReverseOnionTerminalAdapter::new(
            axum::Router::new(), identity, relay.public_key_bytes(), Duration::from_secs(1),
        ).unwrap();
        let stop = Arc::new(WorkerStop { closed: adapter.intake_stop_flag(), wake: Notify::new() });
        let (ready_tx, ready) = watch::channel(WorkerReady::Ready);
        // [REVERSE-ONION-WORKER-COMPLETION 2026-10-06 by Codex]
        let (completed_tx, completed) = watch::channel(());
        // [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] A
        // normal cancellation/drain has no local fault publication.
        let (_failed_tx, failure) = watch::channel(false);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let work_completed = Arc::new(AtomicBool::new(false));
        let task_completed = Arc::clone(&work_completed);
        let task = tokio::spawn(async move {
            let _completion_guard = completed_tx;
            tokio::task::spawn_blocking(move || {
                let _ = started_tx.send(());
                release_rx.recv().unwrap();
                task_completed.store(true, Ordering::SeqCst);
            }).await.map_err(|_| RecipientWorkerError::Ambiguous)?;
            Ok(())
        });
        let worker = ReverseOnionRecipientWorker { stop, ready, completed, failure,
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
        assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
        assert!(worker.task.lock().unwrap().is_some());
        assert!(!work_completed.load(Ordering::SeqCst));
        release_tx.send(()).unwrap();
        worker.shutdown_and_drain().await.unwrap();
        assert!(work_completed.load(Ordering::SeqCst));
        assert!(worker.task.lock().unwrap().is_none());
        worker.shutdown_and_drain().await.unwrap();
        drop(ready_tx);
    }
}
