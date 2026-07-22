// ============================================
// File: crates/aeronyx-blind-issuer/src/api.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] Identity-free, bounded loopback protocol
// with authorization and pressure controls ahead of body extraction.
//! # Internal Blind-Signing API
//!
//! ## Creation Reason
//! The entitlement backend needs a narrow local channel to request an RFC 9474
//! operation without exposing account context to the signer.
//!
//! ## Main Functionality
//! - Authenticates a backend bearer token before buffering the request body.
//! - Enforces private-operation concurrency and per-second pressure ceilings.
//! - Encodes a fixed, allocation-bounded binary request/response contract.
//! - Publishes authenticated, public-only key epochs for safe rotation.
//! - Runs private RSA operations on Tokio's blocking pool.
//! - Holds concurrency capacity until custody work ends, even after cancellation.
//! - Exposes authenticated aggregate health without request/user identifiers.
//! - Opens a monotonic circuit breaker after repeated custody failures.
//! - Audits live-reload outcomes as coarse process-wide counters.
//! - Uses a monotonic bounded token bucket for signing admission.
//!
//! ## Calling Relationships
//! `main.rs` binds this router to loopback; `signer.rs` performs the private
//! operation; the upstream backend may implement the tiny binary codec.
//!
//! ## Privacy Invariant
//! No request/response tracing middleware is installed. Frames contain only a
//! public key fingerprint and blinded RSA bytes.
//!
//! ## Next Developer Guide
//! Keep authentication and pressure middleware outside the `Bytes` extractor.
//! The in-flight guard must move into the blocking operation so client
//! cancellation cannot release HSM capacity early. Never add wallet/account
//! headers or request-body diagnostics.
//!
//! Last Modified: v0.8.0-BlindIssuerApi - Replaced wall-clock fixed-window
//! admission with a monotonic constant-memory token bucket.
//! ============================================

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindIssuerEpoch, BLIND_VAULT_BLIND_ADMISSION_VERSION,
    MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES, MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS,
    MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS, MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
    MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;
use zeroize::Zeroizing;

use crate::config::{
    DEFAULT_CIRCUIT_COOLDOWN_MS, DEFAULT_CIRCUIT_FAILURE_THRESHOLD, DEFAULT_SIGNING_TIMEOUT_MS,
};
use crate::signer::{BlindSignError, BlindSignRequest, BlindSignResponse, BlindSigner};

const REQUEST_MAGIC: [u8; 4] = *b"ANBI";
const RESPONSE_MAGIC: [u8; 4] = *b"ANBS";
const EPOCH_RESPONSE_MAGIC: [u8; 4] = *b"ANBE";
const INTERNAL_WIRE_VERSION: u16 = 1;
const FRAME_HEADER_BYTES: usize = 38;
const EPOCH_RESPONSE_HEADER_BYTES: usize = 16;
const EPOCH_ENTRY_FIXED_BYTES: usize = 60;
const MAX_SIGN_REQUEST_BYTES: usize = FRAME_HEADER_BYTES + MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES;
const MAX_EPOCH_RESPONSE_BYTES: usize = EPOCH_RESPONSE_HEADER_BYTES
    + MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS
        * (EPOCH_ENTRY_FIXED_BYTES + MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES);
const AUTHORIZATION_PREFIX: &[u8] = b"Bearer ";
const CACHE_CONTROL_NO_STORE: &str = "no-store";
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Content type for the local blind-issuer binary contract.
pub const BLIND_ISSUER_CONTENT_TYPE: &str = "application/vnd.aeronyx.blind-issuer-v1";
/// Content type for authenticated public-key epoch snapshots.
pub const BLIND_ISSUER_EPOCH_CONTENT_TYPE: &str = "application/vnd.aeronyx.blind-issuer-epochs-v1";

/// Immutable runtime limits for the local authenticated API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindIssuerApiPolicy {
    /// Maximum signing requests accepted in one wall-clock second.
    pub max_requests_per_second: u64,
    /// Maximum private operations executing concurrently.
    pub max_in_flight: usize,
    /// Maximum HTTP wait for one non-cancellable private operation.
    pub signing_timeout: Duration,
    /// Consecutive backend failures/timeouts required to open the circuit.
    pub circuit_failure_threshold: u64,
    /// Monotonic recovery delay before requests may probe again.
    pub circuit_cooldown: Duration,
}

impl BlindIssuerApiPolicy {
    /// Builds policy with production timeout and circuit-breaker defaults.
    #[must_use]
    pub const fn new(max_requests_per_second: u64, max_in_flight: usize) -> Self {
        Self {
            max_requests_per_second,
            max_in_flight,
            signing_timeout: Duration::from_millis(DEFAULT_SIGNING_TIMEOUT_MS),
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown: Duration::from_millis(DEFAULT_CIRCUIT_COOLDOWN_MS),
        }
    }

    /// Replaces the caller wait bound while retaining all other limits.
    #[must_use]
    pub const fn with_signing_timeout(mut self, signing_timeout: Duration) -> Self {
        self.signing_timeout = signing_timeout;
        self
    }

    /// Replaces circuit failure and monotonic recovery policy.
    #[must_use]
    pub const fn with_circuit_breaker(
        mut self,
        failure_threshold: u64,
        cooldown: Duration,
    ) -> Self {
        self.circuit_failure_threshold = failure_threshold;
        self.circuit_cooldown = cooldown;
        self
    }
}

/// Authenticated aggregate runtime status with no issuance/request dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindIssuerOperationalSnapshot {
    /// Snapshot creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Whether at least one configured key is currently active.
    pub active_key: bool,
    /// Number of public/private custody epochs in the current signer generation.
    pub key_count: usize,
    /// Monotonic in-process signer snapshot generation, starting at one.
    #[serde(default)]
    pub signer_generation: u64,
    /// Key reload candidates evaluated since process startup.
    #[serde(default)]
    pub reload_attempted: u64,
    /// Key reloads atomically installed since process startup.
    #[serde(default)]
    pub reload_succeeded: u64,
    /// Key reloads rejected before changing the active signer.
    #[serde(default)]
    pub reload_rejected: u64,
    /// Unix time of the last evaluated reload, or zero before the first attempt.
    #[serde(default)]
    pub last_reload_attempt_at_ms: u64,
    /// Unix time of the last installed reload, or zero before the first success.
    #[serde(default)]
    pub last_reload_success_at_ms: u64,
    /// Private operations that have not yet returned from custody.
    pub in_flight: usize,
    /// Configured private-operation concurrency ceiling.
    pub max_in_flight: usize,
    /// Whether custody remains unverified after the circuit opened.
    pub circuit_open: bool,
    /// Whether the one allowed post-cooldown recovery probe is in flight.
    #[serde(default)]
    pub circuit_half_open: bool,
    /// Approximate Unix recovery time, or zero when closed/probe-eligible.
    pub circuit_open_until_ms: u64,
    /// Consecutive backend failures/timeouts since the last success.
    pub consecutive_backend_failures: u64,
    /// Signing operations that returned a valid bounded response.
    pub signing_succeeded: u64,
    /// Backend or blocking-task failures returned to callers.
    pub backend_failed: u64,
    /// Caller deadlines exceeded while custody continued in the background.
    pub signing_timed_out: u64,
    /// Requests rejected because every private-operation slot was occupied.
    pub capacity_rejected: u64,
    /// Requests rejected by the per-second process ceiling.
    pub rate_rejected: u64,
    /// Requests rejected while the backend circuit was open.
    pub circuit_rejected: u64,
    /// Requests rejected before body extraction due to bad authorization.
    pub authorization_rejected: u64,
}

/// Public-only issuer key material used by the backend to coordinate rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindIssuerEpochSnapshot {
    /// Snapshot creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Strictly key-ID-sorted, non-expired issuer epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
}

/// Bounded internal wire-codec failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindIssuerWireError {
    /// Frame magic, version, key ID, or length was invalid.
    #[error("blind issuer frame is invalid")]
    InvalidFrame,
    /// Frame exceeded the fixed local protocol limit.
    #[error("blind issuer frame is too large")]
    FrameTooLarge,
}

/// Coarse fail-closed reasons for rejecting a live signer replacement.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BlindIssuerReloadError {
    /// The candidate has no key that can sign at the replacement time.
    #[error("blind issuer reload has no active signer")]
    NoActiveIssuer,
    /// The candidate removed or changed a still-valid published epoch.
    #[error("blind issuer reload breaks epoch continuity")]
    EpochContinuity,
    /// The process exhausted its monotonic signer-generation space.
    #[error("blind issuer signer generation exhausted")]
    GenerationExhausted,
}

struct BlindIssuerRuntimeInner {
    state: RwLock<BlindIssuerRuntimeState>,
}

struct BlindIssuerRuntimeState {
    signer: Arc<BlindSigner>,
    generation: u64,
    reload_attempted: u64,
    reload_succeeded: u64,
    reload_rejected: u64,
    last_reload_attempt_at_ms: u64,
    last_reload_success_at_ms: u64,
}

// [BLIND-ISSUER-RELOAD-AUDIT 2026-07-23 by Codex] Return one coherent
// signer/audit view instead of extending an increasingly fragile tuple.
struct BlindIssuerRuntimeSnapshot {
    signer: Arc<BlindSigner>,
    signer_generation: u64,
    reload_attempted: u64,
    reload_succeeded: u64,
    reload_rejected: u64,
    last_reload_attempt_at_ms: u64,
    last_reload_success_at_ms: u64,
}

impl BlindIssuerRuntimeState {
    fn record_reload_attempt(&mut self, now_ms: u64) {
        self.reload_attempted = self.reload_attempted.saturating_add(1);
        self.last_reload_attempt_at_ms = now_ms;
    }

    fn record_reload_rejection(&mut self) {
        self.reload_rejected = self.reload_rejected.saturating_add(1);
    }

    fn record_reload_success(&mut self, now_ms: u64) {
        self.reload_succeeded = self.reload_succeeded.saturating_add(1);
        self.last_reload_success_at_ms = now_ms;
    }
}

/// Reloadable signer handle shared by HTTP state and the process signal loop.
///
/// Reads hold the lock only long enough to clone an `Arc`; private operations
/// never execute under the lock. A replaced signer therefore remains alive for
/// every request that already selected it.
#[derive(Clone)]
pub struct BlindIssuerRuntime {
    inner: Arc<BlindIssuerRuntimeInner>,
}

impl BlindIssuerRuntime {
    /// Creates generation one from one fully validated signer.
    #[must_use]
    pub fn new(signer: Arc<BlindSigner>) -> Self {
        Self {
            inner: Arc::new(BlindIssuerRuntimeInner {
                state: RwLock::new(BlindIssuerRuntimeState {
                    signer,
                    generation: 1,
                    reload_attempted: 0,
                    reload_succeeded: 0,
                    reload_rejected: 0,
                    last_reload_attempt_at_ms: 0,
                    last_reload_success_at_ms: 0,
                }),
            }),
        }
    }

    /// Atomically installs a candidate that preserves all unexpired epochs.
    ///
    /// # Errors
    /// Returns a coarse error when the candidate is inactive or would remove or
    /// mutate public material that clients may still hold.
    pub fn replace_signer(
        &self,
        candidate: Arc<BlindSigner>,
        now_ms: u64,
    ) -> Result<u64, BlindIssuerReloadError> {
        let candidate_has_active_key = candidate.has_active_key(now_ms);
        let candidate_epochs = candidate.public_epochs(now_ms);
        let mut state = self.inner.state.write();
        state.record_reload_attempt(now_ms);
        if !candidate_has_active_key {
            state.record_reload_rejection();
            return Err(BlindIssuerReloadError::NoActiveIssuer);
        }
        if state
            .signer
            .public_epochs(now_ms)
            .iter()
            .any(|epoch| !candidate_epochs.contains(epoch))
        {
            state.record_reload_rejection();
            return Err(BlindIssuerReloadError::EpochContinuity);
        }
        let Some(next_generation) = state.generation.checked_add(1) else {
            state.record_reload_rejection();
            return Err(BlindIssuerReloadError::GenerationExhausted);
        };
        state.signer = candidate;
        state.generation = next_generation;
        state.record_reload_success(now_ms);
        drop(state);
        Ok(next_generation)
    }

    /// Records a candidate that failed before a signer could be constructed.
    ///
    /// This deliberately stores no failure reason, path, provider, or key ID.
    pub fn record_reload_rejection(&self, now_ms: u64) {
        let mut state = self.inner.state.write();
        state.record_reload_attempt(now_ms);
        state.record_reload_rejection();
    }

    fn snapshot(&self) -> BlindIssuerRuntimeSnapshot {
        let state = self.inner.state.read();
        BlindIssuerRuntimeSnapshot {
            signer: Arc::clone(&state.signer),
            signer_generation: state.generation,
            reload_attempted: state.reload_attempted,
            reload_succeeded: state.reload_succeeded,
            reload_rejected: state.reload_rejected,
            last_reload_attempt_at_ms: state.last_reload_attempt_at_ms,
            last_reload_success_at_ms: state.last_reload_success_at_ms,
        }
    }
}

#[derive(Clone)]
struct ApiState {
    runtime: BlindIssuerRuntime,
    auth_token: Arc<Zeroizing<Vec<u8>>>,
    in_flight: Arc<AtomicUsize>,
    rate: Arc<Mutex<TokenBucket>>,
    policy: BlindIssuerApiPolicy,
    breaker: Arc<CircuitBreaker>,
    metrics: Arc<RuntimeMetrics>,
}

#[derive(Debug, Default)]
struct RuntimeMetrics {
    signing_succeeded: AtomicU64,
    backend_failed: AtomicU64,
    signing_timed_out: AtomicU64,
    capacity_rejected: AtomicU64,
    rate_rejected: AtomicU64,
    circuit_rejected: AtomicU64,
    authorization_rejected: AtomicU64,
}

#[derive(Debug, Default)]
struct CircuitState {
    consecutive_failures: u64,
    open_until: Option<Instant>,
    open_until_ms: u64,
    probe_in_flight: bool,
    generation: u64,
}

#[derive(Debug)]
struct CircuitBreaker {
    failure_threshold: u64,
    cooldown: Duration,
    state: Mutex<CircuitState>,
}

impl CircuitBreaker {
    fn new(failure_threshold: u64, cooldown: Duration) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            cooldown,
            state: Mutex::new(CircuitState::default()),
        }
    }

    fn try_admit(self: &Arc<Self>) -> Option<Arc<CircuitPermit>> {
        let mut state = self.state.lock();
        let is_probe = match state.open_until {
            None => false,
            Some(deadline) if Instant::now() < deadline || state.probe_in_flight => return None,
            Some(_) => {
                state.probe_in_flight = true;
                true
            }
        };
        Some(Arc::new(CircuitPermit {
            breaker: Arc::clone(self),
            generation: state.generation,
            is_probe,
            resolved: AtomicBool::new(false),
        }))
    }

    fn is_healthy(&self) -> bool {
        self.state.lock().open_until.is_none()
    }

    fn record_success(&self, generation: u64) {
        let mut state = self.state.lock();
        if state.generation != generation {
            return;
        }
        let recovered_from_open = state.open_until.is_some();
        state.consecutive_failures = 0;
        state.open_until = None;
        state.open_until_ms = 0;
        state.probe_in_flight = false;
        if recovered_from_open {
            state.generation = state.generation.saturating_add(1);
        }
    }

    fn record_failure(&self, generation: u64, now_ms: u64) {
        let mut state = self.state.lock();
        if state.generation != generation {
            return;
        }
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        if state.probe_in_flight || state.consecutive_failures >= self.failure_threshold {
            state.open_until = Instant::now().checked_add(self.cooldown);
            state.open_until_ms =
                now_ms.saturating_add(self.cooldown.as_millis().try_into().unwrap_or(u64::MAX));
            state.probe_in_flight = false;
            state.generation = state.generation.saturating_add(1);
        }
    }

    fn record_neutral(&self, generation: u64, is_probe: bool) {
        if !is_probe {
            return;
        }
        let mut state = self.state.lock();
        if state.generation == generation {
            state.probe_in_flight = false;
        }
    }

    fn snapshot(&self) -> (bool, bool, u64, u64) {
        let state = self.state.lock();
        let now = Instant::now();
        let circuit_open = state.open_until.is_some();
        (
            circuit_open,
            state.probe_in_flight,
            if state.open_until.is_some_and(|deadline| now < deadline) {
                state.open_until_ms
            } else {
                0
            },
            state.consecutive_failures,
        )
    }
}

/// One admission decision whose first terminal outcome updates the breaker.
///
/// [BLIND-ISSUER-HALF-OPEN 2026-07-23 by Codex] The permit is shared with the
/// blocking custody call. HTTP cancellation therefore cannot release a probe
/// while its private operation is still executing, and timeout/completion races
/// can update the circuit at most once.
#[derive(Debug)]
struct CircuitPermit {
    breaker: Arc<CircuitBreaker>,
    generation: u64,
    is_probe: bool,
    resolved: AtomicBool,
}

impl CircuitPermit {
    fn resolve_success(&self) {
        if !self.resolved.swap(true, Ordering::AcqRel) {
            self.breaker.record_success(self.generation);
        }
    }

    fn resolve_failure(&self, now_ms: u64) {
        if !self.resolved.swap(true, Ordering::AcqRel) {
            self.breaker.record_failure(self.generation, now_ms);
        }
    }

    fn resolve_neutral(&self) {
        if !self.resolved.swap(true, Ordering::AcqRel) {
            self.breaker.record_neutral(self.generation, self.is_probe);
        }
    }
}

impl Drop for CircuitPermit {
    fn drop(&mut self) {
        if !self.resolved.swap(true, Ordering::AcqRel) {
            self.breaker.record_neutral(self.generation, self.is_probe);
        }
    }
}

// [BLIND-ISSUER-RATE 2026-07-23 by Codex] Wall time can jump during NTP
// correction. Keep admission on `Instant` and retain constant memory even when
// callers configure the public router constructor directly.
#[derive(Debug)]
struct TokenBucket {
    capacity: u64,
    available: u64,
    last_refill: Instant,
    refill_remainder_nanos: u128,
}

impl TokenBucket {
    const fn new(capacity: u64, now: Instant) -> Self {
        Self {
            capacity,
            available: capacity,
            last_refill: now,
            refill_remainder_nanos: 0,
        }
    }

    fn try_take(&mut self, now: Instant) -> bool {
        self.refill(now);
        if self.available == 0 {
            return false;
        }
        self.available -= 1;
        true
    }

    fn refill(&mut self, now: Instant) {
        let Some(elapsed) = now.checked_duration_since(self.last_refill) else {
            return;
        };
        self.last_refill = now;
        if self.capacity == 0 {
            self.refill_remainder_nanos = 0;
            return;
        }
        let generated_numerator = elapsed
            .as_nanos()
            .saturating_mul(u128::from(self.capacity))
            .saturating_add(self.refill_remainder_nanos);
        let generated = u64::try_from(generated_numerator / NANOS_PER_SECOND).unwrap_or(u64::MAX);
        self.available = self.capacity.min(self.available.saturating_add(generated));
        self.refill_remainder_nanos = if self.available == self.capacity {
            0
        } else {
            generated_numerator % NANOS_PER_SECOND
        };
    }
}

struct InFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl InFlightGuard {
    fn try_acquire(counter: &Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Self {
                        counter: Arc::clone(counter),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Builds the local signer router with authentication and pressure policy.
pub fn build_router(
    signer: Arc<BlindSigner>,
    auth_token: Zeroizing<Vec<u8>>,
    max_requests_per_second: u64,
    max_in_flight: usize,
) -> Router {
    build_router_with_policy(
        signer,
        auth_token,
        BlindIssuerApiPolicy::new(max_requests_per_second, max_in_flight),
    )
}

/// Builds the local signer router with an explicit custody response deadline.
///
/// A timeout ends only the HTTP wait. The private operation keeps its in-flight
/// permit until the software/HSM/KMS backend actually returns.
pub fn build_router_with_timeout(
    signer: Arc<BlindSigner>,
    auth_token: Zeroizing<Vec<u8>>,
    max_requests_per_second: u64,
    max_in_flight: usize,
    signing_timeout: Duration,
) -> Router {
    build_router_with_policy(
        signer,
        auth_token,
        BlindIssuerApiPolicy::new(max_requests_per_second, max_in_flight)
            .with_signing_timeout(signing_timeout),
    )
}

/// Builds the local signer router from one immutable validated runtime policy.
pub fn build_router_with_policy(
    signer: Arc<BlindSigner>,
    auth_token: Zeroizing<Vec<u8>>,
    policy: BlindIssuerApiPolicy,
) -> Router {
    build_router_with_runtime(BlindIssuerRuntime::new(signer), auth_token, policy)
}

/// Builds the local signer router around a reloadable validated signer handle.
pub fn build_router_with_runtime(
    runtime: BlindIssuerRuntime,
    auth_token: Zeroizing<Vec<u8>>,
    policy: BlindIssuerApiPolicy,
) -> Router {
    // [BLIND-ISSUER-OPS 2026-07-23 by Codex] Metrics are process-wide,
    // aggregate-only, and intentionally contain no key/request dimensions.
    let breaker = Arc::new(CircuitBreaker::new(
        policy.circuit_failure_threshold,
        policy.circuit_cooldown,
    ));
    let state = ApiState {
        runtime,
        auth_token: Arc::new(auth_token),
        in_flight: Arc::new(AtomicUsize::new(0)),
        rate: Arc::new(Mutex::new(TokenBucket::new(
            policy.max_requests_per_second,
            Instant::now(),
        ))),
        policy,
        breaker,
        metrics: Arc::new(RuntimeMetrics::default()),
    };
    let signing = Router::new()
        .route("/internal/v1/blind-sign", post(sign_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            signing_pressure_gate,
        ))
        .layer(DefaultBodyLimit::max(MAX_SIGN_REQUEST_BYTES));
    let authenticated = signing
        .merge(
            Router::new()
                .route("/internal/v1/issuer-epochs", get(epoch_handler))
                .route("/internal/v1/status", get(status_handler)),
        )
        // [BLIND-ISSUER-EPOCHS 2026-07-23 by Codex] Authorization wraps all
        // internal data routes; only private RSA consumes signing capacity.
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            authorization_gate,
        ));
    let health = Router::new().route("/internal/v1/health", get(health_handler));
    authenticated.merge(health).with_state(state)
}

async fn authorization_gate(
    State(state): State<ApiState>,
    request: Request,
    next: Next,
) -> Response {
    if !is_authorized(request.headers(), state.auth_token.as_ref().as_slice()) {
        state
            .metrics
            .authorization_rejected
            .fetch_add(1, Ordering::Relaxed);
        return empty_response(StatusCode::UNAUTHORIZED);
    }
    next.run(request).await
}

async fn signing_pressure_gate(
    State(state): State<ApiState>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(circuit_permit) = state.breaker.try_admit() else {
        state
            .metrics
            .circuit_rejected
            .fetch_add(1, Ordering::Relaxed);
        return empty_response(StatusCode::SERVICE_UNAVAILABLE);
    };
    let Some(guard) = InFlightGuard::try_acquire(&state.in_flight, state.policy.max_in_flight)
    else {
        state
            .metrics
            .capacity_rejected
            .fetch_add(1, Ordering::Relaxed);
        return empty_response(StatusCode::TOO_MANY_REQUESTS);
    };
    if !state.rate.lock().try_take(Instant::now()) {
        state.metrics.rate_rejected.fetch_add(1, Ordering::Relaxed);
        return empty_response(StatusCode::TOO_MANY_REQUESTS);
    }
    // [BLIND-ISSUER-CANCEL 2026-07-23 by Codex] The handler clones this guard
    // into `spawn_blocking`. Dropping a disconnected HTTP request therefore
    // cannot advertise capacity while its non-cancellable HSM call still runs.
    request.extensions_mut().insert(Arc::new(guard));
    request.extensions_mut().insert(circuit_permit);
    next.run(request).await
}

async fn sign_handler(
    State(state): State<ApiState>,
    Extension(operation_guard): Extension<Arc<InFlightGuard>>,
    Extension(circuit_permit): Extension<Arc<CircuitPermit>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if headers.get(header::CONTENT_TYPE).map(HeaderValue::as_bytes)
        != Some(BLIND_ISSUER_CONTENT_TYPE.as_bytes())
    {
        return empty_response(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }
    let Ok(request) = decode_sign_request(&body) else {
        return empty_response(StatusCode::BAD_REQUEST);
    };
    let runtime = state.runtime.snapshot();
    let signer = runtime.signer;
    let backend_circuit_permit = Arc::clone(&circuit_permit);
    let operation = tokio::task::spawn_blocking(move || {
        let result = signer.sign(&request, now_millis());
        match &result {
            Ok(_) => backend_circuit_permit.resolve_success(),
            Err(BlindSignError::SigningFailed) => {
                backend_circuit_permit.resolve_failure(now_millis());
            }
            Err(
                BlindSignError::InvalidRequest
                | BlindSignError::UnknownIssuer
                | BlindSignError::InactiveIssuer,
            ) => backend_circuit_permit.resolve_neutral(),
        }
        drop(operation_guard);
        result
    });
    match tokio::time::timeout(state.policy.signing_timeout, operation).await {
        Ok(Ok(Ok(response))) => {
            state
                .metrics
                .signing_succeeded
                .fetch_add(1, Ordering::Relaxed);
            binary_response(&response)
        }
        Ok(Ok(Err(BlindSignError::InvalidRequest))) => empty_response(StatusCode::BAD_REQUEST),
        Ok(Ok(Err(BlindSignError::UnknownIssuer | BlindSignError::InactiveIssuer))) => {
            empty_response(StatusCode::FORBIDDEN)
        }
        Ok(Ok(Err(BlindSignError::SigningFailed)) | Err(_)) => {
            circuit_permit.resolve_failure(now_millis());
            state.metrics.backend_failed.fetch_add(1, Ordering::Relaxed);
            empty_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(_) => {
            circuit_permit.resolve_failure(now_millis());
            state
                .metrics
                .signing_timed_out
                .fetch_add(1, Ordering::Relaxed);
            empty_response(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

async fn health_handler(State(state): State<ApiState>) -> Response {
    let runtime = state.runtime.snapshot();
    if runtime.signer.has_active_key(now_millis()) && state.breaker.is_healthy() {
        empty_response(StatusCode::NO_CONTENT)
    } else {
        empty_response(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn status_handler(State(state): State<ApiState>) -> Response {
    let generated_at_ms = now_millis();
    let runtime = state.runtime.snapshot();
    let (circuit_open, circuit_half_open, circuit_open_until_ms, consecutive_backend_failures) =
        state.breaker.snapshot();
    let snapshot = BlindIssuerOperationalSnapshot {
        generated_at_ms,
        active_key: runtime.signer.has_active_key(generated_at_ms),
        key_count: runtime.signer.key_count(),
        signer_generation: runtime.signer_generation,
        reload_attempted: runtime.reload_attempted,
        reload_succeeded: runtime.reload_succeeded,
        reload_rejected: runtime.reload_rejected,
        last_reload_attempt_at_ms: runtime.last_reload_attempt_at_ms,
        last_reload_success_at_ms: runtime.last_reload_success_at_ms,
        in_flight: state.in_flight.load(Ordering::Acquire),
        max_in_flight: state.policy.max_in_flight,
        circuit_open,
        circuit_half_open,
        circuit_open_until_ms,
        consecutive_backend_failures,
        signing_succeeded: state.metrics.signing_succeeded.load(Ordering::Relaxed),
        backend_failed: state.metrics.backend_failed.load(Ordering::Relaxed),
        signing_timed_out: state.metrics.signing_timed_out.load(Ordering::Relaxed),
        capacity_rejected: state.metrics.capacity_rejected.load(Ordering::Relaxed),
        rate_rejected: state.metrics.rate_rejected.load(Ordering::Relaxed),
        circuit_rejected: state.metrics.circuit_rejected.load(Ordering::Relaxed),
        authorization_rejected: state.metrics.authorization_rejected.load(Ordering::Relaxed),
    };
    (
        [(header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE)],
        Json(snapshot),
    )
        .into_response()
}

async fn epoch_handler(State(state): State<ApiState>) -> Response {
    let generated_at_ms = now_millis();
    let runtime = state.runtime.snapshot();
    let snapshot = BlindIssuerEpochSnapshot {
        generated_at_ms,
        epochs: runtime.signer.public_epochs(generated_at_ms),
    };
    encode_epoch_snapshot(&snapshot).map_or_else(
        |_| empty_response(StatusCode::INTERNAL_SERVER_ERROR),
        |bytes| {
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, BLIND_ISSUER_EPOCH_CONTENT_TYPE),
                    (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
                ],
                bytes,
            )
                .into_response()
        },
    )
}

fn is_authorized(headers: &HeaderMap, expected: &[u8]) -> bool {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return false;
    };
    let bytes = value.as_bytes();
    let Some(candidate) = bytes.strip_prefix(AUTHORIZATION_PREFIX) else {
        return false;
    };
    candidate.len() == expected.len() && bool::from(candidate.ct_eq(expected))
}

/// Encodes one backend-to-signer request.
///
/// # Errors
/// Returns a wire error for unsupported versions, zero key IDs, or bad bounds.
pub fn encode_sign_request(request: &BlindSignRequest) -> Result<Vec<u8>, BlindIssuerWireError> {
    validate_request(request)?;
    let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + request.blinded_message.len());
    bytes.extend_from_slice(&REQUEST_MAGIC);
    bytes.extend_from_slice(&request.version.to_be_bytes());
    bytes.extend_from_slice(&request.issuer_key_id);
    bytes.extend_from_slice(&request.blinded_message);
    Ok(bytes)
}

fn decode_sign_request(bytes: &[u8]) -> Result<BlindSignRequest, BlindIssuerWireError> {
    if bytes.len() > MAX_SIGN_REQUEST_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < FRAME_HEADER_BYTES + MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES
        || bytes[..4] != REQUEST_MAGIC
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    let mut issuer_key_id = [0; 32];
    issuer_key_id.copy_from_slice(&bytes[6..FRAME_HEADER_BYTES]);
    let request = BlindSignRequest {
        version,
        issuer_key_id,
        blinded_message: bytes[FRAME_HEADER_BYTES..].to_vec(),
    };
    validate_request(&request)?;
    Ok(request)
}

fn validate_request(request: &BlindSignRequest) -> Result<(), BlindIssuerWireError> {
    if request.version != BLIND_VAULT_BLIND_ADMISSION_VERSION
        || request.issuer_key_id.iter().all(|byte| *byte == 0)
        || !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
            .contains(&request.blinded_message.len())
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    Ok(())
}

fn binary_response(response: &BlindSignResponse) -> Response {
    let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + response.blind_signature.len());
    bytes.extend_from_slice(&RESPONSE_MAGIC);
    bytes.extend_from_slice(&response.version.to_be_bytes());
    bytes.extend_from_slice(&response.issuer_key_id);
    bytes.extend_from_slice(&response.blind_signature);
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE),
            (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
        ],
        bytes,
    )
        .into_response()
}

/// Decodes one signer-to-backend response.
///
/// # Errors
/// Returns a wire error for malformed, oversized, or unsupported frames.
pub fn decode_sign_response(bytes: &[u8]) -> Result<BlindSignResponse, BlindIssuerWireError> {
    if bytes.len() > MAX_SIGN_REQUEST_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < FRAME_HEADER_BYTES + MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES
        || bytes[..4] != RESPONSE_MAGIC
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    let mut issuer_key_id = [0; 32];
    issuer_key_id.copy_from_slice(&bytes[6..FRAME_HEADER_BYTES]);
    let blind_signature = bytes[FRAME_HEADER_BYTES..].to_vec();
    if version != BLIND_VAULT_BLIND_ADMISSION_VERSION
        || issuer_key_id.iter().all(|byte| *byte == 0)
        || !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
            .contains(&blind_signature.len())
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    Ok(BlindSignResponse {
        version,
        issuer_key_id,
        blind_signature,
    })
}

/// Encodes a deterministic, allocation-bounded public epoch snapshot.
///
/// # Errors
/// Returns a wire error when an epoch violates key, ordering, or time bounds.
pub fn encode_epoch_snapshot(
    snapshot: &BlindIssuerEpochSnapshot,
) -> Result<Vec<u8>, BlindIssuerWireError> {
    validate_epoch_snapshot(snapshot)?;
    let capacity = EPOCH_RESPONSE_HEADER_BYTES
        + snapshot
            .epochs
            .iter()
            .map(|epoch| EPOCH_ENTRY_FIXED_BYTES + epoch.public_key_der.len())
            .sum::<usize>();
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&EPOCH_RESPONSE_MAGIC);
    bytes.extend_from_slice(&INTERNAL_WIRE_VERSION.to_be_bytes());
    bytes.extend_from_slice(&snapshot.generated_at_ms.to_be_bytes());
    let epoch_count =
        u16::try_from(snapshot.epochs.len()).map_err(|_| BlindIssuerWireError::InvalidFrame)?;
    bytes.extend_from_slice(&epoch_count.to_be_bytes());
    for epoch in &snapshot.epochs {
        bytes.extend_from_slice(&epoch.admission_version.to_be_bytes());
        bytes.extend_from_slice(&epoch.issuer_key_id);
        bytes.extend_from_slice(&epoch.not_before_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.max_lease_ttl_ms.to_be_bytes());
        let der_length = u16::try_from(epoch.public_key_der.len())
            .map_err(|_| BlindIssuerWireError::InvalidFrame)?;
        bytes.extend_from_slice(&der_length.to_be_bytes());
        bytes.extend_from_slice(&epoch.public_key_der);
    }
    Ok(bytes)
}

/// Decodes and validates one signer public-key epoch snapshot.
///
/// # Errors
/// Returns a wire error for malformed, oversized, unsorted, or stale epochs.
pub fn decode_epoch_snapshot(
    bytes: &[u8],
) -> Result<BlindIssuerEpochSnapshot, BlindIssuerWireError> {
    if bytes.len() > MAX_EPOCH_RESPONSE_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < EPOCH_RESPONSE_HEADER_BYTES || bytes[..4] != EPOCH_RESPONSE_MAGIC {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let mut cursor = 4;
    let wire_version = u16::from_be_bytes(take_array(bytes, &mut cursor)?);
    if wire_version != INTERNAL_WIRE_VERSION {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let generated_at_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
    let epoch_count = usize::from(u16::from_be_bytes(take_array(bytes, &mut cursor)?));
    if epoch_count > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let mut epochs = Vec::with_capacity(epoch_count);
    for _ in 0..epoch_count {
        let admission_version = u16::from_be_bytes(take_array(bytes, &mut cursor)?);
        let issuer_key_id = take_array(bytes, &mut cursor)?;
        let not_before_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let expires_at_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let max_lease_ttl_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let der_length = usize::from(u16::from_be_bytes(take_array(bytes, &mut cursor)?));
        let der_end = cursor
            .checked_add(der_length)
            .ok_or(BlindIssuerWireError::InvalidFrame)?;
        let public_key_der = bytes
            .get(cursor..der_end)
            .ok_or(BlindIssuerWireError::InvalidFrame)?
            .to_vec();
        cursor = der_end;
        epochs.push(BlindVaultBlindIssuerEpoch {
            admission_version,
            issuer_key_id,
            public_key_der,
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        });
    }
    if cursor != bytes.len() {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let snapshot = BlindIssuerEpochSnapshot {
        generated_at_ms,
        epochs,
    };
    validate_epoch_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn validate_epoch_snapshot(
    snapshot: &BlindIssuerEpochSnapshot,
) -> Result<(), BlindIssuerWireError> {
    if snapshot.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS
        || snapshot
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    for epoch in &snapshot.epochs {
        let lifetime = epoch
            .expires_at_ms
            .checked_sub(epoch.not_before_ms)
            .ok_or(BlindIssuerWireError::InvalidFrame)?;
        let derived_key_id: [u8; 32] = Sha256::digest(&epoch.public_key_der).into();
        if epoch.admission_version != BLIND_VAULT_BLIND_ADMISSION_VERSION
            || epoch.issuer_key_id.iter().all(|byte| *byte == 0)
            || epoch.issuer_key_id != derived_key_id
            || epoch.public_key_der.is_empty()
            || epoch.public_key_der.len() > MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES
            || lifetime == 0
            || lifetime > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS
            || epoch.expires_at_ms <= snapshot.generated_at_ms
            || epoch.max_lease_ttl_ms == 0
        {
            return Err(BlindIssuerWireError::InvalidFrame);
        }
    }
    Ok(())
}

fn take_array<const LENGTH: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; LENGTH], BlindIssuerWireError> {
    let end = cursor
        .checked_add(LENGTH)
        .ok_or(BlindIssuerWireError::InvalidFrame)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(BlindIssuerWireError::InvalidFrame)?;
    let mut value = [0; LENGTH];
    value.copy_from_slice(slice);
    *cursor = end;
    Ok(value)
}

fn empty_response(status: StatusCode) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
            (header::CONTENT_LENGTH, "0"),
        ],
        Body::empty(),
    )
        .into_response()
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(all(test, unix))]
mod tests {
    // Panicking on fixture/setup errors is intentional in tests; production
    // paths remain fully fallible and return privacy-safe errors.
    #![allow(clippy::expect_used, clippy::similar_names, clippy::too_many_lines)]

    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    use std::sync::{Condvar, Mutex as StdMutex};
    use std::time::Duration;

    use aeronyx_core::protocol::blind_vault::BlindVaultBlindAdmissionToken;
    use axum::body::to_bytes;
    use axum::http::Request as HttpRequest;
    use blind_rsa_signatures::{BlindSignature, DefaultRng, KeyPairSha384PSSRandomized};
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;

    use crate::config::{BlindIssuerConfig, BlindIssuerKeyConfig};
    use crate::signer::{BlindSigningBackend, BlindSigningBackendError, BlindSigningKey};

    const BACKEND_TOKEN: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFG";

    fn write_secret(path: &std::path::Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .expect("create secret");
        file.write_all(bytes).expect("write secret");
        file.sync_all().expect("sync secret");
    }

    struct BlockingBackend {
        release: Arc<(StdMutex<bool>, Condvar)>,
        started: tokio::sync::mpsc::UnboundedSender<()>,
        completed: tokio::sync::mpsc::UnboundedSender<()>,
    }

    impl BlindSigningBackend for BlockingBackend {
        fn sign_blinded(
            &self,
            _blinded_message: &[u8],
        ) -> Result<Vec<u8>, BlindSigningBackendError> {
            self.started
                .send(())
                .map_err(|_| BlindSigningBackendError)?;
            let (lock, wake) = self.release.as_ref();
            let mut released = lock.lock().map_err(|_| BlindSigningBackendError)?;
            while !*released {
                released = wake.wait(released).map_err(|_| BlindSigningBackendError)?;
            }
            drop(released);
            self.completed
                .send(())
                .map_err(|_| BlindSigningBackendError)?;
            Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES])
        }
    }

    struct SwitchableBackend {
        failing: Arc<AtomicBool>,
    }

    impl BlindSigningBackend for SwitchableBackend {
        fn sign_blinded(
            &self,
            _blinded_message: &[u8],
        ) -> Result<Vec<u8>, BlindSigningBackendError> {
            if self.failing.load(Ordering::Acquire) {
                Err(BlindSigningBackendError)
            } else {
                Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES])
            }
        }
    }

    fn signer_from_epochs(epochs: Vec<BlindVaultBlindIssuerEpoch>) -> Arc<BlindSigner> {
        let signing_keys = epochs
            .into_iter()
            .map(|epoch| {
                BlindSigningKey::new(
                    epoch,
                    Box::new(SwitchableBackend {
                        failing: Arc::new(AtomicBool::new(false)),
                    }),
                )
                .expect("test signing key")
            })
            .collect();
        Arc::new(BlindSigner::from_signing_keys(signing_keys).expect("test signer"))
    }

    fn authorized_sign_http_request(body: Vec<u8>, authorization: &str) -> HttpRequest<Body> {
        HttpRequest::builder()
            .method("POST")
            .uri("/internal/v1/blind-sign")
            .header(header::AUTHORIZATION, authorization)
            .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
            .body(Body::from(body))
            .expect("authorized signing request")
    }

    fn status_http_request(authorization: Option<&str>) -> HttpRequest<Body> {
        let mut request = HttpRequest::builder().uri("/internal/v1/status");
        if let Some(authorization) = authorization {
            request = request.header(header::AUTHORIZATION, authorization);
        }
        request.body(Body::empty()).expect("status request")
    }

    async fn operational_snapshot(
        router: &Router,
        authorization: &str,
    ) -> BlindIssuerOperationalSnapshot {
        let response = router
            .clone()
            .oneshot(status_http_request(Some(authorization)))
            .await
            .expect("status response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = to_bytes(response.into_body(), 8 * 1024)
            .await
            .expect("status body");
        serde_json::from_slice(&bytes).expect("status snapshot")
    }

    async fn health_status(router: &Router) -> StatusCode {
        router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/internal/v1/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response")
            .status()
    }

    struct BlockingFixture {
        router: Router,
        body: Vec<u8>,
        authorization: String,
        release: Arc<(StdMutex<bool>, Condvar)>,
        started: tokio::sync::mpsc::UnboundedReceiver<()>,
        completed: tokio::sync::mpsc::UnboundedReceiver<()>,
    }

    impl BlockingFixture {
        fn new(signing_timeout: Duration) -> Self {
            let now_ms = now_millis();
            let key_pair =
                KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
            let public_epoch = BlindVaultBlindIssuerEpoch::new(
                key_pair.pk.to_der().expect("public DER"),
                now_ms - 60_000,
                now_ms + 24 * 60 * 60 * 1_000,
                7 * 24 * 60 * 60 * 1_000,
            );
            let issuer_key_id = public_epoch.issuer_key_id;
            let release = Arc::new((StdMutex::new(false), Condvar::new()));
            let (started_tx, started) = tokio::sync::mpsc::unbounded_channel();
            let (completed_tx, completed) = tokio::sync::mpsc::unbounded_channel();
            let signer = Arc::new(
                BlindSigner::from_signing_keys(vec![BlindSigningKey::new(
                    public_epoch,
                    Box::new(BlockingBackend {
                        release: Arc::clone(&release),
                        started: started_tx,
                        completed: completed_tx,
                    }),
                )
                .expect("blocking signing key")])
                .expect("blocking signer"),
            );
            let router = build_router_with_timeout(
                signer,
                Zeroizing::new(BACKEND_TOKEN.to_vec()),
                128,
                1,
                signing_timeout,
            );
            let request = BlindSignRequest {
                version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
                issuer_key_id,
                blinded_message: vec![3; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
            };
            let authorization = format!(
                "Bearer {}",
                std::str::from_utf8(BACKEND_TOKEN).expect("ASCII backend token")
            );
            Self {
                router,
                body: encode_sign_request(&request).expect("encode request"),
                authorization,
                release,
                started,
                completed,
            }
        }

        async fn wait_started(&mut self) {
            tokio::time::timeout(Duration::from_secs(2), self.started.recv())
                .await
                .expect("backend start should not time out")
                .expect("backend operation started");
        }

        async fn release_and_wait(&mut self) {
            let (lock, wake) = self.release.as_ref();
            *lock.lock().expect("release lock") = true;
            wake.notify_all();
            tokio::time::timeout(Duration::from_secs(2), self.completed.recv())
                .await
                .expect("backend completion should not time out")
                .expect("backend operation completed");
        }
    }

    #[test]
    fn token_bucket_refills_monotonically_and_caps_idle_burst() {
        let started_at = Instant::now();
        let mut bucket = TokenBucket::new(2, started_at);

        assert!(bucket.try_take(started_at));
        assert!(bucket.try_take(started_at));
        assert!(!bucket.try_take(started_at));
        assert!(!bucket.try_take(started_at + Duration::from_millis(499)));
        assert!(bucket.try_take(started_at + Duration::from_millis(500)));
        assert!(!bucket.try_take(started_at + Duration::from_millis(500)));
        assert!(bucket.try_take(started_at + Duration::from_secs(1)));
        assert!(!bucket.try_take(started_at + Duration::from_secs(1)));

        let after_idle = started_at + Duration::from_secs(10);
        assert!(bucket.try_take(after_idle));
        assert!(bucket.try_take(after_idle));
        assert!(!bucket.try_take(after_idle));

        let mut closed = TokenBucket::new(0, started_at);
        assert!(!closed.try_take(started_at + Duration::from_secs(1)));
    }

    #[tokio::test]
    async fn cancelled_requests_hold_capacity_until_backend_completion() {
        let mut fixture = BlockingFixture::new(Duration::from_secs(10));

        let first_router = fixture.router.clone();
        let first_body = fixture.body.clone();
        let first_authorization = fixture.authorization.clone();
        let first = tokio::spawn(async move {
            first_router
                .oneshot(authorized_sign_http_request(
                    first_body,
                    &first_authorization,
                ))
                .await
        });
        fixture.wait_started().await;
        first.abort();
        assert!(first
            .await
            .expect_err("request should be cancelled")
            .is_cancelled());

        let second = tokio::time::timeout(
            Duration::from_secs(2),
            fixture.router.clone().oneshot(authorized_sign_http_request(
                fixture.body.clone(),
                &fixture.authorization,
            )),
        )
        .await;

        fixture.release_and_wait().await;

        let second_status = second
            .expect("capacity rejection should not block")
            .expect("capacity response")
            .status();
        assert_eq!(second_status, StatusCode::TOO_MANY_REQUESTS);

        let recovered = fixture
            .router
            .oneshot(authorized_sign_http_request(
                fixture.body,
                &fixture.authorization,
            ))
            .await
            .expect("recovered response");
        assert_eq!(recovered.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn signing_timeout_is_retryable_without_releasing_capacity() {
        let mut fixture = BlockingFixture::new(Duration::from_millis(50));
        let first_router = fixture.router.clone();
        let first_body = fixture.body.clone();
        let first_authorization = fixture.authorization.clone();
        let first = tokio::spawn(async move {
            first_router
                .oneshot(authorized_sign_http_request(
                    first_body,
                    &first_authorization,
                ))
                .await
        });
        fixture.wait_started().await;
        let timed_out = tokio::time::timeout(Duration::from_secs(2), first)
            .await
            .expect("HTTP timeout response")
            .expect("request task")
            .expect("timeout response");
        assert_eq!(timed_out.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(timed_out.headers()[header::CACHE_CONTROL], "no-store");

        let saturated = fixture
            .router
            .clone()
            .oneshot(authorized_sign_http_request(
                fixture.body.clone(),
                &fixture.authorization,
            ))
            .await
            .expect("saturated response");
        assert_eq!(saturated.status(), StatusCode::TOO_MANY_REQUESTS);

        fixture.release_and_wait().await;
        let recovered = fixture
            .router
            .oneshot(authorized_sign_http_request(
                fixture.body,
                &fixture.authorization,
            ))
            .await
            .expect("recovered response");
        assert_eq!(recovered.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn circuit_breaker_recovers_and_status_stays_aggregate_only() {
        let now_ms = now_millis();
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        let public_epoch = BlindVaultBlindIssuerEpoch::new(
            key_pair.pk.to_der().expect("public DER"),
            now_ms - 60_000,
            now_ms + 24 * 60 * 60 * 1_000,
            7 * 24 * 60 * 60 * 1_000,
        );
        let issuer_key_id = public_epoch.issuer_key_id;
        let failing = Arc::new(AtomicBool::new(true));
        let signer = Arc::new(
            BlindSigner::from_signing_keys(vec![BlindSigningKey::new(
                public_epoch,
                Box::new(SwitchableBackend {
                    failing: Arc::clone(&failing),
                }),
            )
            .expect("switchable signing key")])
            .expect("switchable signer"),
        );
        let policy =
            BlindIssuerApiPolicy::new(128, 1).with_circuit_breaker(2, Duration::from_millis(100));
        let router =
            build_router_with_policy(signer, Zeroizing::new(BACKEND_TOKEN.to_vec()), policy);
        let body = encode_sign_request(&BlindSignRequest {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            blinded_message: vec![9; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
        })
        .expect("encode request");
        let authorization = format!(
            "Bearer {}",
            std::str::from_utf8(BACKEND_TOKEN).expect("ASCII backend token")
        );

        let unauthorized = router
            .clone()
            .oneshot(status_http_request(None))
            .await
            .expect("unauthorized status response");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        for _ in 0..2 {
            let failed = router
                .clone()
                .oneshot(authorized_sign_http_request(body.clone(), &authorization))
                .await
                .expect("backend failure response");
            assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);
        }
        let rejected = router
            .clone()
            .oneshot(authorized_sign_http_request(body.clone(), &authorization))
            .await
            .expect("open circuit response");
        assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);

        assert_eq!(
            health_status(&router).await,
            StatusCode::SERVICE_UNAVAILABLE
        );

        let open = operational_snapshot(&router, &authorization).await;
        assert!(open.active_key);
        assert_eq!(open.key_count, 1);
        assert_eq!(open.signer_generation, 1);
        assert_eq!(open.reload_attempted, 0);
        assert_eq!(open.reload_succeeded, 0);
        assert_eq!(open.reload_rejected, 0);
        assert_eq!(open.last_reload_attempt_at_ms, 0);
        assert_eq!(open.last_reload_success_at_ms, 0);
        assert_eq!(open.in_flight, 0);
        assert_eq!(open.max_in_flight, 1);
        assert!(open.circuit_open);
        assert!(!open.circuit_half_open);
        assert!(open.circuit_open_until_ms >= open.generated_at_ms);
        assert_eq!(open.consecutive_backend_failures, 2);
        assert_eq!(open.backend_failed, 2);
        assert_eq!(open.signing_timed_out, 0);
        assert_eq!(open.circuit_rejected, 1);
        assert_eq!(open.authorization_rejected, 1);

        failing.store(false, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(
            health_status(&router).await,
            StatusCode::SERVICE_UNAVAILABLE,
            "cooldown alone must not restore custody health"
        );
        let recovered = router
            .clone()
            .oneshot(authorized_sign_http_request(body, &authorization))
            .await
            .expect("recovered signing response");
        assert_eq!(recovered.status(), StatusCode::OK);
        let closed = operational_snapshot(&router, &authorization).await;
        assert!(!closed.circuit_open);
        assert!(!closed.circuit_half_open);
        assert_eq!(closed.circuit_open_until_ms, 0);
        assert_eq!(closed.consecutive_backend_failures, 0);
        assert_eq!(closed.signing_succeeded, 1);
    }

    #[tokio::test]
    async fn half_open_allows_one_probe_and_ignores_stale_success() {
        let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_millis(20)));
        let failing = breaker.try_admit().expect("initial admission");
        let stale_success = breaker.try_admit().expect("concurrent admission");

        failing.resolve_failure(now_millis());
        stale_success.resolve_success();
        assert!(
            !breaker.is_healthy(),
            "stale success must not close circuit"
        );
        assert!(breaker.try_admit().is_none());

        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!breaker.is_healthy());
        let abandoned_probe = breaker.try_admit().expect("half-open probe");
        assert!(breaker.try_admit().is_none(), "only one probe may run");
        let snapshot = breaker.snapshot();
        assert!(snapshot.0);
        assert!(snapshot.1);
        assert_eq!(snapshot.2, 0);
        drop(abandoned_probe);

        let request_probe = breaker.try_admit().expect("replacement probe");
        let backend_probe = Arc::clone(&request_probe);
        drop(request_probe);
        assert!(
            breaker.try_admit().is_none(),
            "request cancellation must not release a running backend probe"
        );
        backend_probe.resolve_failure(now_millis());
        assert!(breaker.try_admit().is_none());

        tokio::time::sleep(Duration::from_millis(30)).await;
        let successful_probe = breaker.try_admit().expect("recovery probe");
        successful_probe.resolve_success();
        assert!(breaker.is_healthy());
        assert!(breaker.try_admit().is_some());
    }

    #[tokio::test]
    async fn signer_reload_is_atomic_active_and_continuity_safe() {
        let now_ms = now_millis();
        let old_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("old RSA key");
        let new_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("new RSA key");
        let old_epoch = BlindVaultBlindIssuerEpoch::new(
            old_pair.pk.to_der().expect("old public DER"),
            now_ms - 60_000,
            now_ms + 60 * 60 * 1_000,
            7 * 24 * 60 * 60 * 1_000,
        );
        let new_epoch = BlindVaultBlindIssuerEpoch::new(
            new_pair.pk.to_der().expect("new public DER"),
            now_ms - 30_000,
            now_ms + 2 * 60 * 60 * 1_000,
            7 * 24 * 60 * 60 * 1_000,
        );
        let future_epoch = BlindVaultBlindIssuerEpoch {
            not_before_ms: now_ms + 60_000,
            expires_at_ms: now_ms + 3 * 60 * 60 * 1_000,
            ..new_epoch.clone()
        };
        let runtime = BlindIssuerRuntime::new(signer_from_epochs(vec![old_epoch.clone()]));
        let held = runtime.snapshot();
        assert_eq!(held.signer_generation, 1);

        runtime.record_reload_rejection(now_ms - 3);

        assert_eq!(
            runtime.replace_signer(signer_from_epochs(vec![future_epoch]), now_ms - 2),
            Err(BlindIssuerReloadError::NoActiveIssuer)
        );
        assert_eq!(
            runtime.replace_signer(signer_from_epochs(vec![new_epoch.clone()]), now_ms - 1),
            Err(BlindIssuerReloadError::EpochContinuity)
        );
        assert_eq!(
            runtime.replace_signer(signer_from_epochs(vec![old_epoch, new_epoch]), now_ms,),
            Ok(2)
        );

        let current = runtime.snapshot();
        assert_eq!(current.signer_generation, 2);
        assert_eq!(current.signer.key_count(), 2);
        assert_eq!(current.reload_attempted, 4);
        assert_eq!(current.reload_succeeded, 1);
        assert_eq!(current.reload_rejected, 3);
        assert_eq!(current.last_reload_attempt_at_ms, now_ms);
        assert_eq!(current.last_reload_success_at_ms, now_ms);
        assert_eq!(held.signer.key_count(), 1, "in-flight snapshot stays alive");

        let router = build_router_with_runtime(
            runtime.clone(),
            Zeroizing::new(BACKEND_TOKEN.to_vec()),
            BlindIssuerApiPolicy::new(128, 1),
        );
        let authorization = format!(
            "Bearer {}",
            std::str::from_utf8(BACKEND_TOKEN).expect("ASCII backend token")
        );
        let reported = operational_snapshot(&router, &authorization).await;
        assert_eq!(reported.signer_generation, 2);
        assert_eq!(reported.reload_attempted, 4);
        assert_eq!(reported.reload_succeeded, 1);
        assert_eq!(reported.reload_rejected, 3);
        assert_eq!(reported.last_reload_attempt_at_ms, now_ms);
        assert_eq!(reported.last_reload_success_at_ms, now_ms);

        runtime.inner.state.write().generation = u64::MAX;
        assert_eq!(
            runtime.replace_signer(Arc::clone(&current.signer), now_ms + 1),
            Err(BlindIssuerReloadError::GenerationExhausted)
        );
        let exhausted = runtime.snapshot();
        assert_eq!(exhausted.signer_generation, u64::MAX);
        assert_eq!(exhausted.reload_attempted, 5);
        assert_eq!(exhausted.reload_succeeded, 1);
        assert_eq!(exhausted.reload_rejected, 4);
    }

    #[tokio::test]
    async fn authenticated_http_signing_finalizes_and_verifies() {
        let now_ms = now_millis();
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        let private_der = key_pair.sk.to_der().expect("private DER");
        let public_der = key_pair.pk.to_der().expect("public DER");
        let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
        let directory = tempfile::tempdir().expect("temp directory");
        let private_path = directory.path().join("issuer.der");
        write_secret(&private_path, &private_der);
        let config = BlindIssuerConfig {
            listen_addr: "127.0.0.1:9191".to_owned(),
            auth_token_file: directory.path().join("backend.token"),
            max_requests_per_second: 128,
            max_in_flight: 8,
            signing_timeout_ms: DEFAULT_SIGNING_TIMEOUT_MS,
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_ms: DEFAULT_CIRCUIT_COOLDOWN_MS,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: private_path,
                not_before_unix_secs: now_ms / 1_000 - 60,
                expires_at_unix_secs: now_ms / 1_000 + 24 * 60 * 60,
                max_lease_ttl_secs: 7 * 24 * 60 * 60,
            }],
        };
        let signer = Arc::new(BlindSigner::from_config(&config).expect("signer"));
        let router = build_router(
            Arc::clone(&signer),
            Zeroizing::new(BACKEND_TOKEN.to_vec()),
            2,
            8,
        );

        let unsigned =
            BlindVaultBlindAdmissionToken::new(issuer_key_id, [7; 32], [1; 32], vec![0; 256]);
        let message = unsigned.message_bytes();
        let blinding = key_pair
            .pk
            .blind(&mut DefaultRng, &message)
            .expect("blind message");
        let request = BlindSignRequest {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            blinded_message: blinding.blind_message.0.clone(),
        };
        let body = encode_sign_request(&request).expect("encode request");

        let unauthorized = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body.clone()))
                    .expect("unauthorized request"),
            )
            .await
            .expect("unauthorized response");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorization = format!(
            "Bearer {}",
            std::str::from_utf8(BACKEND_TOKEN).expect("ASCII backend token")
        );
        let epoch_response = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/internal/v1/issuer-epochs")
                    .header(header::AUTHORIZATION, &authorization)
                    .body(Body::empty())
                    .expect("epoch request"),
            )
            .await
            .expect("epoch response");
        assert_eq!(epoch_response.status(), StatusCode::OK);
        assert_eq!(
            epoch_response.headers()[header::CONTENT_TYPE],
            BLIND_ISSUER_EPOCH_CONTENT_TYPE
        );
        let epoch_body = to_bytes(epoch_response.into_body(), MAX_EPOCH_RESPONSE_BYTES)
            .await
            .expect("epoch body");
        let epoch_snapshot = decode_epoch_snapshot(&epoch_body).expect("epoch snapshot");
        assert_eq!(epoch_snapshot.epochs.len(), 1);
        assert_eq!(epoch_snapshot.epochs[0].issuer_key_id, issuer_key_id);
        let mut tampered_epoch_body = epoch_body.to_vec();
        *tampered_epoch_body.last_mut().expect("epoch DER byte") ^= 1;
        assert_eq!(
            decode_epoch_snapshot(&tampered_epoch_body),
            Err(BlindIssuerWireError::InvalidFrame)
        );
        let mut trailing_epoch_body = epoch_body.to_vec();
        trailing_epoch_body.push(0);
        assert_eq!(
            decode_epoch_snapshot(&trailing_epoch_body),
            Err(BlindIssuerWireError::InvalidFrame)
        );

        let response = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::AUTHORIZATION, &authorization)
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body.clone()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let response_body = to_bytes(response.into_body(), 1_024)
            .await
            .expect("response body");
        let signed_response =
            decode_sign_response(&response_body).expect("blind signature response");
        assert_eq!(signed_response.issuer_key_id, issuer_key_id);
        let blind_signature = BlindSignature::new(signed_response.blind_signature.clone());
        let finalized = key_pair
            .pk
            .finalize(&blind_signature, &blinding, &message)
            .expect("finalize signature");
        key_pair
            .pk
            .verify(&finalized, blinding.msg_randomizer, &message)
            .expect("verify finalized signature");

        let retry = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::AUTHORIZATION, &authorization)
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body))
                    .expect("retry request"),
            )
            .await
            .expect("retry response");
        let retry_body = to_bytes(retry.into_body(), 1_024)
            .await
            .expect("retry body");
        assert_eq!(
            decode_sign_response(&retry_body)
                .expect("retry signature")
                .blind_signature,
            signed_response.blind_signature
        );

        let health = router
            .oneshot(
                HttpRequest::builder()
                    .uri("/internal/v1/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response");
        assert_eq!(health.status(), StatusCode::NO_CONTENT);
        assert_eq!(signer.public_epochs(now_ms).len(), 1);
        assert!(signer.has_active_key(now_ms));
        assert!(!signer.has_active_key(now_ms + 2 * 24 * 60 * 60 * 1_000));
    }
}
