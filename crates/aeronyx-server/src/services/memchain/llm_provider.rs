// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_provider.rs
// ============================================
//! # LLM Provider — Trait + Shared Types
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Defines the `LlmProvider` async trait and all shared request/response types
//! used by the two provider implementations (OpenAI-compatible, Anthropic) and
//! the router (`LlmRouter`).
//!
//! ## Main Types
//! - `LlmProvider` trait — single method: `chat(req) → Result<ChatResponse>`
//! - `ChatRequest` — messages + parameters sent to any provider
//! - `ChatResponse` — model output + token usage
//! - `ChatMessage` — role + content pair
//! - `TokenUsage` — input/output/cached token counts
//! - `LlmError` — structured error type for provider failures
//!
//! ## CognitiveTaskType — RE-EXPORTED from config_supernode.rs
//! ⚠️ CognitiveTaskType is defined in config_supernode.rs (the single source of truth)
//! and re-exported here for backward compatibility with code that imports from
//! `llm_provider::CognitiveTaskType`. Do NOT define CognitiveTaskType in this file.
//!
//! ## Design Decisions
//! - `LlmProvider` is object-safe (`async_trait` macro expands to boxed futures)
//! - `TokenUsage` intentionally omits `cost_usd` — fee rates change; compute at query time
//! - All types derive `serde::Serialize/Deserialize` for JSON storage in DB
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding a new task type, add the variant to config_supernode::CognitiveTaskType,
//!   NOT here. This file only re-exports it.
//! - `ChatRequest::system` is optional. For task types that don't need a system
//!   prompt (simple completion), leave it None.
//! - `LlmProvider::chat()` must be cancel-safe — the caller may drop the future
//!   if the task is cancelled.
//! - Provider response bodies must be consumed through
//!   [`read_bounded_llm_response`]. Never call unbounded `Response::text/json`.
//! - Provider construction must use [`normalize_llm_api_base`] and
//!   [`resolve_llm_api_key`], then [`build_llm_http_client`], so explicitly
//!   configured runtimes fail closed without copying endpoints,
//!   environment-variable names, or secrets into process-health diagnostics.
//!
//! ## Last Modified
//! v2.5.5-FailureBoundary - [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex]
//!   Added stable, privacy-safe runtime reason codes for persistence and logs.
//! v2.5.4-StartupIntegrity - [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex]
//!   Added typed, privacy-safe provider initialization errors plus shared API
//!   endpoint and environment-backed key validation.
//! v2.5.3-ResponseBoundary - [LLM-RESPONSE-BOUNDARY 2026-07-30 by Codex]
//!   Added a shared bounded response reader, UTF-8-safe error formatting, and
//!   a monotonic provider cooldown that recovers without a background timer.
//! v2.5.0+SuperNode - 🌟 Created.
//! v2.5.0+Unify     - 🔧 [BUG FIX] Removed duplicate CognitiveTaskType definition.
//!   CognitiveTaskType is now defined ONLY in config_supernode.rs and re-exported
//!   here. The old definition had different variant names (CommunitySummary vs
//!   CommunityNarrative, NaturalSummary vs RecallSynthesis, CustomPrompt vs
//!   ConflictResolution/CodeAnalysis) which caused compilation errors across
//!   task_worker.rs, llm_router.rs, and mod.rs re-exports.

use std::fmt;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    CertificateError, ClientConfig, DigitallySignedStruct, Error as RustlsError, RootCertStore,
    SignatureScheme,
};
use sha2::{Digest, Sha256};

/// Maximum successful LLM response body retained before JSON parsing.
///
/// This is intentionally much larger than normal cognitive-task output while
/// still preventing a custom or compromised provider from exhausting memory.
pub(super) const MAX_LLM_SUCCESS_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Maximum non-success response body retained for diagnostics.
pub(super) const MAX_LLM_ERROR_BODY_BYTES: usize = 64 * 1024;

// [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] The cap applies to
// exact JSON bytes, including escaping, before attestation or inference IO.
pub(super) const MAX_ACI_REQUEST_BODY_BYTES: usize = 1024 * 1024;

pub(super) fn serialize_bounded_aci_request<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, LlmError> {
    struct BoundedJson(Vec<u8>);
    impl std::io::Write for BoundedJson {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let required = self.0.len().checked_add(bytes.len())
                .filter(|size| *size <= MAX_ACI_REQUEST_BODY_BYTES)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "ACI request limit"))?;
            if required > self.0.capacity() {
                let capacity = required.max(self.0.capacity().saturating_mul(2)).max(1024)
                    .min(MAX_ACI_REQUEST_BODY_BYTES);
                self.0.reserve_exact(capacity - self.0.len());
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut writer = BoundedJson(Vec::new());
    serde_json::to_writer(&mut writer, value).map_err(|_| LlmError::AciRequestRejected)?;
    Ok(writer.0)
}

pub(super) fn validate_aci_request_options(
    model: &str, max_tokens: Option<u32>, temperature: Option<f32>,
) -> Result<(), LlmError> {
    if model.trim().is_empty() || model.len() > 256 || max_tokens == Some(0)
        || temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
    {
        return Err(LlmError::AciRequestRejected);
    }
    Ok(())
}

/// Maximum API-error bytes rendered through `Display`.
const MAX_LLM_ERROR_DISPLAY_BYTES: usize = 200;

/// Default cooldown after transport, parsing, or provider API failure.
const DEFAULT_LLM_PROVIDER_COOLDOWN: Duration = Duration::from_secs(30);

/// Maximum provider-requested cooldown accepted from `Retry-After`.
const MAX_LLM_PROVIDER_COOLDOWN_SECS: u64 = 5 * 60;

/// Fixed provider request deadline until per-provider limits are introduced.
const LLM_PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

// [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Local lifetime
// ceiling, not a replacement for the attested report/keyset validity bounds.
const PHALA_ACI_CONTEXT_MAX_AGE: Duration = Duration::from_secs(5 * 60);

pub(super) struct PhalaAciObservation {
    started: Instant,
    floor: AtomicU64,
}

impl PhalaAciObservation {
    pub(super) fn begin() -> Result<Self, LlmError> {
        let started = Instant::now();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| LlmError::AciVerifierUnavailable)?.as_secs();
        Ok(Self { started, floor: AtomicU64::new(now) })
    }

    pub(super) fn started(&self) -> Instant { self.started }

    pub(super) fn observe(&self) -> Result<u64, LlmError> {
        self.observe_at(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|time| time.as_secs()).map_err(|_| LlmError::AciVerifierUnavailable))
    }

    fn observe_at(&self, now: Result<u64, LlmError>) -> Result<u64, LlmError> {
        let now = now?;
        if self.started.elapsed() > PHALA_ACI_CONTEXT_MAX_AGE {
            return Err(LlmError::AciVerifierUnavailable);
        }
        self.floor.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |previous| (now >= previous).then_some(now))
            .map_err(|_| LlmError::AciVerifierUnavailable)?;
        Ok(now)
    }
}

// [MEMCHAIN-PHALA-ENDPOINT-CONTRACT 2026-10-05 by Codex]
pub(crate) const PHALA_ACI_API_BASE_DEFAULT: &str = "https://inference.phala.com/v1";

// [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] ACI/1 section 6.1
// defines this header independently of a product's provider routing block.
// Replace any previous values rather than appending a conflicting `none`.
pub(super) fn with_required_aci_upstream(
    request: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::HeaderName::from_static("x-upstream-verification"),
        reqwest::header::HeaderValue::from_static("required"),
    );
    request.headers(headers)
}

// [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] This plaintext provider
// API has no source-owned E2EE request or response key. Gateway appraisal and
// provider.aci_verified do not change that fact. Keep router and direct
// adapters on the same boundary; client E2EE must not enable this API.
pub(super) const fn node_has_source_e2ee_transport() -> bool {
    false
}

pub(super) fn require_source_e2ee_transport() -> Result<(), LlmError> {
    if node_has_source_e2ee_transport() {
        Ok(())
    } else {
        Err(LlmError::ConfidentialE2eeTransportUnavailable)
    }
}

// ============================================
// Re-export CognitiveTaskType from canonical location
// ============================================

/// Re-exported from config_supernode.rs — the SINGLE SOURCE OF TRUTH.
/// All code that previously imported `llm_provider::CognitiveTaskType`
/// will continue to work without changes.
pub use crate::config_supernode::CognitiveTaskType;

// ============================================
// Error Type
// ============================================

/// Privacy-safe reason for rejecting one configured LLM provider at startup.
///
/// [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] These variants are the
/// complete diagnostics boundary used by process health. They intentionally
/// retain no endpoint, provider name, environment-variable name, or secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProviderInitError {
    InvalidApiBase,
    ProviderSecretUnavailable,
    ProviderSecretRequired,
    HttpClientInitializationFailed,
}

impl LlmProviderInitError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::InvalidApiBase => "invalid_api_base",
            Self::ProviderSecretUnavailable => "provider_secret_unavailable",
            Self::ProviderSecretRequired => "provider_secret_required",
            Self::HttpClientInitializationFailed => "http_client_initialization_failed",
        }
    }
}

impl fmt::Display for LlmProviderInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LLM provider initialization failed ({})",
            self.reason_code()
        )
    }
}

impl std::error::Error for LlmProviderInitError {}

/// Canonicalizes and validates a configured provider endpoint.
///
/// Only HTTP(S) hierarchical URLs without embedded credentials, query, or
/// fragment state are accepted. Provider adapters append their fixed API path
/// after this boundary, so accepting those components would make routing
/// ambiguous and could leak credentials through ordinary URL diagnostics.
pub(super) fn normalize_llm_api_base(raw_api_base: &str) -> Result<String, LlmProviderInitError> {
    let normalized = raw_api_base.trim().trim_end_matches('/');
    if normalized.is_empty() {
        return Err(LlmProviderInitError::InvalidApiBase);
    }

    let parsed =
        reqwest::Url::parse(normalized).map_err(|_| LlmProviderInitError::InvalidApiBase)?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.cannot_be_a_base()
    {
        return Err(LlmProviderInitError::InvalidApiBase);
    }

    Ok(normalized.to_owned())
}

// [MEMCHAIN-PHALA-ENDPOINT-CONTRACT 2026-10-05 by Codex]
/// Validates and canonicalizes the sole supported Phala ACI API origin.
pub(crate) fn validate_phala_aci_api_base(
    raw_api_base: &str,
) -> Result<String, LlmProviderInitError> {
    let normalized = normalize_llm_api_base(raw_api_base)?;
    let parsed = reqwest::Url::parse(&normalized)
        .map_err(|_| LlmProviderInitError::InvalidApiBase)?;
    if parsed.scheme() != "https"
        || parsed.host_str() != Some("inference.phala.com")
        || parsed.port().is_some_and(|port| port != 443)
        || !matches!(parsed.path(), "" | "/" | "/v1")
    {
        return Err(LlmProviderInitError::InvalidApiBase);
    }
    Ok(normalized)
}

/// Resolves an optional `$ENV_VAR` provider key at the startup boundary.
///
/// A missing explicitly referenced environment variable is configuration
/// drift and therefore an error even for keyless OpenAI-compatible endpoints.
/// A literal empty key remains valid only when `required` is false.
pub(super) fn resolve_llm_api_key(
    raw_api_key: &str,
    required: bool,
) -> Result<String, LlmProviderInitError> {
    let api_key = if let Some(var_name) = raw_api_key.strip_prefix('$') {
        if var_name.is_empty() {
            return Err(LlmProviderInitError::ProviderSecretUnavailable);
        }
        std::env::var(var_name).map_err(|_| LlmProviderInitError::ProviderSecretUnavailable)?
    } else {
        raw_api_key.to_owned()
    };

    if required && api_key.trim().is_empty() {
        return Err(LlmProviderInitError::ProviderSecretRequired);
    }

    Ok(api_key)
}

/// Builds the shared provider HTTP transport without implicit host proxies.
///
/// [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Provider routing is an
/// explicit node configuration decision. Consulting OS or environment proxy
/// state could redirect private cognitive traffic to an undeclared endpoint;
/// on some macOS hosts the system proxy adapter can also panic during client
/// construction. Explicit future proxy support must be validated configuration.
pub(super) fn build_llm_http_client() -> Result<reqwest::Client, LlmProviderInitError> {
    build_llm_http_client_with_redirects(true)
}

// [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Do not forward the Phala API
// credential across a redirect to a different authority.
pub(super) fn build_confidential_llm_http_client() -> Result<reqwest::Client, LlmProviderInitError> {
    build_llm_http_client_with_redirects(false)
}

// [MEMCHAIN-PHALA-TLS-SPKI-PIN 2026-10-05 by Codex] Keep WebPKI chain and
// hostname validation, then additionally require the certificate SPKI that
// the nonce-bound ACI keyset declared. ACI pins never replace normal TLS.
pub(super) fn build_phala_pinned_http_client(
    pins: &[String],
) -> Result<reqwest::Client, LlmProviderInitError> {
    let pins: HashSet<String> = pins
        .iter()
        .map(|pin| pin.to_ascii_lowercase())
        .filter(|pin| pin.len() == 64 && pin.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .collect();
    if pins.is_empty() {
        return Err(LlmProviderInitError::HttpClientInitializationFailed);
    }

    let native = rustls_native_certs::load_native_certs();
    let mut roots = RootCertStore::empty();
    let mut added = 0usize;
    for certificate in native.certs {
        if roots.add(certificate).is_ok() {
            added = added.saturating_add(1);
        }
    }
    if added == 0 {
        return Err(LlmProviderInitError::HttpClientInitializationFailed);
    }

    let chain_verifier = rustls::client::WebPkiServerVerifier::builder(Arc::new(roots.clone()))
        .build()
        .map_err(|_| LlmProviderInitError::HttpClientInitializationFailed)?;
    let mut tls = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    // [MEMCHAIN-PHALA-TLS-API 2026-10-06 by Codex] rustls 0.23 installs the
    // additive verifier through its mutating setter; default PKI validation
    // remains explicitly delegated to `chain_verifier` above.
    tls.dangerous().set_certificate_verifier(Arc::new(
        PhalaPinnedServerCertVerifier {
            chain_verifier,
            pins,
        },
    ));

    reqwest::Client::builder()
        .no_proxy()
        .timeout(LLM_PROVIDER_REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .use_preconfigured_tls(tls)
        .build()
        .map_err(|_| LlmProviderInitError::HttpClientInitializationFailed)
}

// [MEMCHAIN-PHALA-TLS-SPKI-PIN 2026-10-05 by Codex]
#[derive(Debug)]
struct PhalaPinnedServerCertVerifier {
    chain_verifier: Arc<rustls::client::WebPkiServerVerifier>,
    pins: HashSet<String>,
}

impl ServerCertVerifier for PhalaPinnedServerCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        self.chain_verifier.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        )?;
        let (_, certificate) = x509_parser::parse_x509_certificate(end_entity.as_ref())
            .map_err(|_| RustlsError::InvalidCertificate(CertificateError::BadEncoding))?;
        let digest = hex::encode(Sha256::digest(certificate.tbs_certificate.subject_pki.raw));
        if !self.pins.contains(&digest) {
            return Err(RustlsError::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        self.chain_verifier
            .verify_tls12_signature(message, certificate, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        self.chain_verifier
            .verify_tls13_signature(message, certificate, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.chain_verifier.supported_verify_schemes()
    }
}

fn build_llm_http_client_with_redirects(
    allow_redirects: bool,
) -> Result<reqwest::Client, LlmProviderInitError> {
    let builder = reqwest::Client::builder()
        .no_proxy()
        .timeout(LLM_PROVIDER_REQUEST_TIMEOUT);
    let builder = if allow_redirects {
        builder
    } else {
        builder.redirect(reqwest::redirect::Policy::none())
    };
    builder
        .build()
        .map_err(|_| LlmProviderInitError::HttpClientInitializationFailed)
}

/// Structured error returned by LLM provider calls.
#[derive(Debug, Clone)]
pub enum LlmError {
    /// HTTP transport error (connection refused, timeout, etc.)
    Transport(String),
    /// Provider returned a non-2xx HTTP status
    ApiError { status: u16, body: String },
    /// Response body could not be parsed
    ParseError(String),
    /// Model returned an empty or unusable response
    EmptyResponse,
    /// Rate limit hit (HTTP 429)
    RateLimit { retry_after_secs: Option<u64> },
    /// Context too long for this model
    ContextTooLong,
    /// Provider is not configured
    NotConfigured(String),
    /// The request requires confidential serving but the provider cannot enforce it.
    ConfidentialServingRequired,
    /// The provider sends ACI constraints but has no cryptographic verifier wired in.
    // [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
    AciVerifierUnavailable,
    /// The node has no client-to-attested-workload E2EE transport for model payloads.
    // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex]
    ConfidentialE2eeTransportUnavailable,
    /// A provider did not return the minimum ACI/1 response-header contract.
    // [MEMCHAIN-PHALA-ACI-HEADERS 2026-10-05 by Codex]
    AciResponseContractViolation,
    /// Invalid or oversized local input; no provider IO is authorized.
    // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex]
    AciRequestRejected,
}

impl LlmError {
    /// Stable failure category safe for operator logs and task persistence.
    ///
    /// [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex] `Display` retains
    /// bounded provider diagnostics for direct callers, but those diagnostics
    /// may include a provider-controlled response or transport endpoint. Queue
    /// state and routine logs must use this method instead.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "llm_transport_error",
            Self::ApiError { .. } => "llm_api_error",
            Self::ParseError(_) => "llm_response_parse_error",
            Self::EmptyResponse => "llm_empty_response",
            Self::RateLimit { .. } => "llm_rate_limited",
            Self::ContextTooLong => "llm_context_too_long",
            Self::NotConfigured(_) => "llm_provider_not_configured",
            Self::ConfidentialServingRequired => "llm_confidential_serving_required",
            Self::AciVerifierUnavailable => "llm_aci_verifier_unavailable",
            Self::ConfidentialE2eeTransportUnavailable => "llm_client_to_tee_e2ee_unavailable",
            Self::AciResponseContractViolation => "llm_aci_response_contract_violation",
            Self::AciRequestRejected => "llm_aci_request_rejected",
        }
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "transport error: {}", e),
            Self::ApiError { status, body } => {
                write!(
                    f,
                    "API error {}: {}",
                    status,
                    utf8_prefix(body, MAX_LLM_ERROR_DISPLAY_BYTES)
                )
            }
            Self::ParseError(e) => write!(f, "parse error: {}", e),
            Self::EmptyResponse => write!(f, "empty response from model"),
            Self::RateLimit { retry_after_secs } => {
                write!(f, "rate limit hit (retry after: {:?}s)", retry_after_secs)
            }
            Self::ContextTooLong => write!(f, "context too long for this model"),
            Self::NotConfigured(name) => write!(f, "provider '{}' not configured", name),
            Self::ConfidentialServingRequired => write!(f, "confidential serving is required"),
            Self::AciVerifierUnavailable => write!(f, "ACI verification is unavailable"),
            Self::ConfidentialE2eeTransportUnavailable => {
                write!(f, "client-to-attested-workload E2EE transport is unavailable")
            }
            Self::AciResponseContractViolation => write!(f, "ACI response contract was not satisfied"),
            Self::AciRequestRejected => write!(f, "ACI request was rejected before dispatch"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Monotonic provider availability state shared by all HTTP implementations.
///
/// [LLM-PROVIDER-COOLDOWN 2026-07-30 by Codex] A permanent boolean creates a
/// one-way failure latch: once the router skips a provider, no future success
/// can make it healthy. A monotonic deadline permits bounded automatic retry
/// without wall-clock jumps or a background timer.
#[derive(Debug, Default)]
pub(super) struct ProviderHealth {
    unhealthy_until_ms: AtomicU64,
}

impl ProviderHealth {
    pub(super) fn mark_unhealthy(&self) {
        self.mark_unhealthy_for(DEFAULT_LLM_PROVIDER_COOLDOWN);
    }

    pub(super) fn mark_rate_limited(&self, retry_after_secs: Option<u64>) {
        self.mark_unhealthy_for(rate_limit_cooldown(retry_after_secs));
    }

    pub(super) fn mark_healthy(&self) {
        self.mark_healthy_at(monotonic_millis());
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.is_healthy_at(monotonic_millis())
    }

    fn mark_unhealthy_for(&self, duration: Duration) {
        self.mark_unhealthy_at(monotonic_millis(), duration_to_millis(duration));
    }

    fn mark_unhealthy_at(&self, now_ms: u64, duration_ms: u64) {
        let deadline = now_ms.saturating_add(duration_ms.max(1));
        self.unhealthy_until_ms
            .fetch_max(deadline, Ordering::Relaxed);
    }

    fn mark_healthy_at(&self, now_ms: u64) {
        // [LLM-PROVIDER-COOLDOWN 2026-07-30 by Codex] A successful request may
        // have started before a concurrent request recorded a newer failure.
        // Only clear an already-expired deadline so that stale success cannot
        // erase a live cooldown.
        let _ = self.unhealthy_until_ms.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |deadline| (deadline <= now_ms).then_some(0),
        );
    }

    fn is_healthy_at(&self, now_ms: u64) -> bool {
        now_ms >= self.unhealthy_until_ms.load(Ordering::Relaxed)
    }
}

fn monotonic_millis() -> u64 {
    static PROCESS_EPOCH: OnceLock<Instant> = OnceLock::new();
    let elapsed = PROCESS_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis();
    u64::try_from(elapsed).unwrap_or(u64::MAX)
}

fn duration_to_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn rate_limit_cooldown(retry_after_secs: Option<u64>) -> Duration {
    Duration::from_secs(
        retry_after_secs
            .unwrap_or(DEFAULT_LLM_PROVIDER_COOLDOWN.as_secs())
            .clamp(1, MAX_LLM_PROVIDER_COOLDOWN_SECS),
    )
}

/// Size-checked accumulator shared by fixed-length and chunked responses.
struct BoundedLlmBody {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl BoundedLlmBody {
    fn new(max_bytes: usize, content_length: Option<u64>) -> Result<Self, LlmError> {
        if content_length
            .is_some_and(|length| length > u64::try_from(max_bytes).unwrap_or(u64::MAX))
        {
            return Err(response_body_too_large(max_bytes));
        }
        let initial_capacity = content_length
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0)
            .min(max_bytes);

        Ok(Self {
            bytes: Vec::with_capacity(initial_capacity),
            max_bytes,
        })
    }

    fn push(&mut self, chunk: &[u8]) -> Result<(), LlmError> {
        let next_length = self
            .bytes
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| response_body_too_large(self.max_bytes))?;
        if next_length > self.max_bytes {
            return Err(response_body_too_large(self.max_bytes));
        }
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Consume one HTTP response under an explicit byte ceiling.
///
/// [LLM-RESPONSE-BOUNDARY 2026-07-30 by Codex] `Content-Length` is only an
/// early rejection hint because chunked responses can omit or falsify it.
/// Every received chunk is checked again before extending the accumulator.
pub(super) async fn read_bounded_llm_response(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, LlmError> {
    let mut body = BoundedLlmBody::new(max_bytes, response.content_length())?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| LlmError::Transport(error.to_string()))?
    {
        body.push(&chunk)?;
    }

    Ok(body.into_bytes())
}

fn response_body_too_large(max_bytes: usize) -> LlmError {
    LlmError::ParseError(format!("LLM response body exceeds {max_bytes} byte limit"))
}

fn utf8_prefix(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

// ============================================
// Chat Types
// ============================================

/// A single message in a conversation (role + content).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Request sent to an LLM provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    /// Conversation messages (system + user + optional assistant)
    pub messages: Vec<ChatMessage>,
    /// Optional override for model (if None, provider uses its configured default)
    pub model_override: Option<String>,
    /// Maximum tokens to generate (None = provider default)
    pub max_tokens: Option<u32>,
    /// Temperature 0.0-2.0 (None = provider default)
    pub temperature: Option<f32>,
    /// Stop sequences (None = no stop sequences)
    pub stop: Option<Vec<String>>,
    /// Require Phala's ACI-verified upstream routing for this request.
    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    #[serde(default)]
    pub require_aci_verified: bool,
}

impl ChatRequest {
    // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Bound iteration
    // and raw fields before constructing the provider's exact wire payload.
    pub(super) fn validate_aci_bounds(&self) -> Result<(), LlmError> {
        if self.messages.is_empty() || self.messages.len() > 128
            || self.stop.as_ref().is_some_and(|values| values.len() > 16
                || values.iter().any(|value| value.len() > 1024))
        {
            return Err(LlmError::AciRequestRejected);
        }
        let mut total = 0usize;
        for message in &self.messages {
            if message.role.is_empty() || message.role.len() > 64 {
                return Err(LlmError::AciRequestRejected);
            }
            total = total.checked_add(message.role.len())
                .and_then(|size| size.checked_add(message.content.len()))
                .filter(|size| *size <= MAX_ACI_REQUEST_BODY_BYTES)
                .ok_or(LlmError::AciRequestRejected)?;
        }
        validate_aci_request_options(self.model_override.as_deref().unwrap_or("configured-model"),
            self.max_tokens, self.temperature)
    }

    /// Convenience constructor: single user message, no system prompt.
    pub fn simple(user_content: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::user(user_content)],
            model_override: None,
            max_tokens: None,
            temperature: None,
            stop: None,
            require_aci_verified: false,
        }
    }

    /// Convenience constructor: system + user message.
    pub fn with_system(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
            model_override: None,
            max_tokens: None,
            temperature: None,
            stop: None,
            require_aci_verified: false,
        }
    }
}

/// Token usage for a single LLM call.
///
/// ## Note on cost_usd
/// Cost is intentionally NOT stored here. Fee rates change frequently and vary
/// by context (cached vs. uncached, batch vs. real-time). Compute at query time
/// using rate tables in `LlmRouter::estimate_cost()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens served from prompt cache (subset of input_tokens)
    pub cached_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    pub fn billable_input(&self) -> u32 {
        self.input_tokens.saturating_sub(self.cached_tokens)
    }
}

/// Response from an LLM provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatResponse {
    /// The model's text output (first choice, trimmed)
    pub content: String,
    /// Token usage for this call
    pub usage: TokenUsage,
    /// The model identifier actually used (may differ from request if overridden)
    pub model_used: String,
    /// Provider name (for logging and writeback)
    pub provider_name: String,
    /// Wall-clock latency in milliseconds
    pub latency_ms: u64,
    /// Unauthenticated ACI response-header hints; these are not a verified quote or receipt.
    // [MEMCHAIN-PHALA-ACI-HEADERS 2026-10-05 by Codex]
    #[serde(default)]
    pub aci_response_hints: Option<AciResponseHints>,
    /// Quote-bound report and signed receipt, present only after ACI verification.
    // [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
    #[serde(default)]
    pub aci_verification: Option<AciVerificationEvidence>,
}

/// Batch embedding input. Confidential semantic text is accepted only by the
/// explicitly configured Phala ACI route.
// [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// Verified embeddings plus the exact ACI evidence used to authorize storage.
// [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model_used: String,
    pub aci_response_hints: Option<AciResponseHints>,
    pub aci_verification: Option<AciVerificationEvidence>,
}

/// Evidence retained with a staged task so restart recovery can audit the proof.
// [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AciVerificationEvidence {
    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Persist the exact wire
    // profile; missing historical tags are readable but not current authority.
    #[serde(default)]
    pub wire_profile: String,
    #[serde(default)]
    pub key_custody_scope: String,
    /// Stable hash of the workload identity public-key object, NOT its keyset.
    #[serde(default)]
    pub workload_id: String,
    /// Digest of the currently attested operational keyset.
    pub keyset_digest: String,
    /// Phala app-compose measurement recovered from the verified TDX event log.
    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    #[serde(default)]
    pub compose_hash: String,
    /// Compressed SEC1 Dstack KMS root recovered from the identity-key chain.
    // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex]
    #[serde(default)]
    pub kms_root_public_key: String,
    pub receipt_id: String,
    #[serde(default)]
    pub upstream_session_id: String,
    #[serde(default)]
    pub upstream_session: serde_json::Value,
    /// ACI receipt attests what this gateway asserts; it is not an independent
    /// verification of the upstream TEE evidence by this client.
    // [MEMCHAIN-PHALA-UPSTREAM-SCOPE 2026-10-06 by Codex]
    #[serde(default)]
    pub upstream_claim_scope: String,
    pub request_body_sha256: String,
    pub response_body_sha256: String,
    pub attestation_report: serde_json::Value,
    pub receipt: serde_json::Value,
    pub upstream_verified_required: bool,
}

impl AciVerificationEvidence {
    // This structural guard does not itself verify the quote or receipt signature.
    // [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
    pub(crate) fn has_complete_shape(&self) -> bool {
        let digest = |value: &str| {
            value.strip_prefix("sha256:").is_some_and(|hex| {
                hex.len() == 64
                    && hex.bytes().all(|byte| {
                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                    })
            })
        };
        self.wire_profile == PHALA_ACI_WIRE_PROFILE
            && self.key_custody_scope == "identity_kms_operational_measured_code"
            && digest(&self.keyset_digest)
            && digest(&self.workload_id)
            && self.attestation_report.get("workload_id").and_then(serde_json::Value::as_str)
                == Some(self.workload_id.as_str())
            && self.attestation_report.get("workload_keyset_digest").and_then(serde_json::Value::as_str)
                == Some(self.keyset_digest.as_str())
            && self.receipt.get("workload_id").and_then(serde_json::Value::as_str)
                == Some(self.workload_id.as_str())
            && digest(&self.compose_hash)
            && self.kms_root_public_key.strip_prefix("0x").is_some_and(|key| {
                key.len() == 66
                    && (key.starts_with("02") || key.starts_with("03"))
                    && key.bytes().all(|byte| {
                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                    })
            })
            && digest(&self.request_body_sha256)
            && digest(&self.response_body_sha256)
            && !self.receipt_id.is_empty()
            && self.receipt_id.len() <= 512
            && self.receipt.get("workload_keyset_digest").and_then(serde_json::Value::as_str)
                == Some(self.keyset_digest.as_str())
            // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
            // ACI session content addresses use `as_` plus lowercase SHA-256 hex.
            && self.upstream_session_id.strip_prefix("as_").is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
            && self.upstream_claim_scope == "gateway_assertion"
            && self.upstream_session.is_object()
            && self.attestation_report.is_object()
            && self.receipt.is_object()
            && self.upstream_verified_required
    }
}

// [PHALA-133171-PROFILE 2026-10-08 by Codex] Internal discriminator, not
// a negotiated HTTP version and not a claim about any deployed gateway.
pub const PHALA_ACI_WIRE_PROFILE: &str = "aci/1@133171efb115bb0437f69a4679b5522c5e36139e";

/// Headers are unauthenticated lookup hints. This profile carries separate
/// identity and keyset digests, both later bound to the report and receipt.
// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AciResponseHints {
    pub version: String,
    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Never synthesize this
    // from keyset_digest, even for legacy serialized lookup hints.
    #[serde(default)]
    pub workload_id: String,
    pub keyset_digest: String,
    pub receipt_id: String,
}

impl AciResponseHints {
    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
    pub(super) fn has_valid_shape(&self) -> bool {
        let is_sha256_digest = |value: &str| {
            value.strip_prefix("sha256:").is_some_and(|hex| {
                hex.len() == 64
                    && hex.bytes().all(|byte| {
                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                    })
            })
        };
        self.version == "aci/1"
            && is_sha256_digest(&self.keyset_digest)
            && is_sha256_digest(&self.workload_id)
            && !self.receipt_id.is_empty()
            && self.receipt_id.len() <= 512
            && self
                .receipt_id
                .bytes()
                .all(|byte| (0x21..=0x7e).contains(&byte))
    }

    pub(super) fn from_headers(
        headers: &reqwest::header::HeaderMap,
    ) -> Result<Self, LlmError> {
        let value = |name: &'static str| -> Result<String, LlmError> {
            let mut values = headers.get_all(name).iter();
            let value = values
                .next()
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty() && value.len() <= 512)
                .ok_or(LlmError::AciResponseContractViolation)?;
            if values.next().is_some() {
                return Err(LlmError::AciResponseContractViolation);
            }
            Ok(value.to_owned())
        };
        let version = value("x-aci-version")?;
        let workload_id = value("x-aci-identity")?;
        let keyset_digest = value("x-aci-keyset-digest")?;
        let receipt_id = value("x-receipt-id")?;
        let hints = Self {
            version,
            workload_id,
            keyset_digest,
            receipt_id,
        };
        if !hints.has_valid_shape() {
            return Err(LlmError::AciResponseContractViolation);
        }
        Ok(hints)
    }
}

// ============================================
// LlmProvider Trait
// ============================================

/// Async trait for a single LLM provider backend.
///
/// Implementations: `OpenAiCompatProvider` (covers OpenAI/DeepSeek/Groq/Ollama),
/// `AnthropicProvider` (Anthropic Messages API).
///
/// Each implementation handles its own:
/// - HTTP transport (reqwest)
/// - Authentication header format
/// - Request/response JSON shape
/// - Rate limit detection
/// - Timeout (from its config)
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Legacy plaintext chat API. Production adapters must reject before
    /// appraisal or inference IO when source-owned E2EE is unavailable.
    // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex]
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Embed a bounded batch. Generic providers deliberately have no implicit
    /// embedding implementation; confidential indexing requires explicit ACI.
    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    async fn embed(&self, _req: &EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ConfidentialServingRequired)
    }

    /// Provider name for logging and writeback (e.g. "deepseek", "anthropic").
    fn name(&self) -> &str;

    /// Default model identifier for this provider (e.g. "deepseek-chat").
    fn default_model(&self) -> &str;

    /// Whether this provider is currently healthy (not rate-limited, not in backoff).
    /// Default: always healthy. Providers can override to implement circuit breaking.
    fn is_healthy(&self) -> bool {
        true
    }

    /// True only when this provider attaches the ACI-verified serving constraint.
    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex]
    fn supports_aci_verified(&self) -> bool {
        false
    }

    /// True only when ACI attestation and per-response receipts are verified cryptographically.
    // [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
    fn has_cryptographic_aci_verifier(&self) -> bool {
        false
    }

    /// Whether this provider's explicit relying-party policy accepts the
    /// cryptographically verified workload measurement in this proof.
    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    fn accepts_aci_compose_hash(&self, _compose_hash: &str) -> bool {
        false
    }

    /// Whether relying-party policy accepts the KMS root bound to the receipt signer.
    // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex]
    fn accepts_aci_kms_root(&self, _root_public_key: &str) -> bool {
        false
    }

    /// Whether relying-party policy binds the report's source provenance to
    /// this quote-measured compose hash.
    // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
    fn accepts_aci_source_provenance(
        &self,
        _compose_hash: &str,
        _provenance: &serde_json::Value,
    ) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] Authored only;
    // building a request checks its wire metadata without sending it.
    #[test]
    fn aci_upstream_requirement_replaces_conflicting_values_without_rewriting_body() {
        let client = reqwest::Client::new();
        let body = br#"{"provider":{"aci_verified":true},"model":"synthetic"}"#.to_vec();
        for previous in [None, Some("none"), Some("required")] {
            let mut request = client.post("https://inference.phala.com/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer synthetic-only")
                .body(body.clone());
            if let Some(previous) = previous {
                request = request.header("x-upstream-verification", previous)
                    .header("x-upstream-verification", "none");
            }
            let request = with_required_aci_upstream(request).build().unwrap();
            assert_eq!(request.headers().get_all("x-upstream-verification").iter().count(), 1);
            assert_eq!(request.headers()["x-upstream-verification"], "required");
            assert_eq!(request.headers()["authorization"], "Bearer synthetic-only");
            assert_eq!(request.headers()["content-type"], "application/json");
            assert_eq!(request.body().and_then(reqwest::Body::as_bytes), Some(body.as_slice()));
        }
        assert!(!node_has_source_e2ee_transport());
        assert!(matches!(require_source_e2ee_transport(),
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
    }

    // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Authored only: a
    // receipt verifier is not a capability to release source plaintext.
    #[test]
    fn node_plaintext_provider_contract_has_no_source_e2ee_capability() {
        assert!(!node_has_source_e2ee_transport());
        assert!(matches!(require_source_e2ee_transport(),
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
    }

    // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Source-only
    // regressions: exact bytes, inclusive cap and escaping-induced overflow.
    #[test]
    fn aci_json_writer_preserves_bytes_and_enforces_exact_wire_cap() {
        let small = serde_json::json!({"text": "quotes\" backslash\\ control\u{0} unicode\u{4e2d}"});
        assert_eq!(serialize_bounded_aci_request(&small).unwrap(), serde_json::to_vec(&small).unwrap());
        let exact = "x".repeat(MAX_ACI_REQUEST_BODY_BYTES - 2);
        assert_eq!(serialize_bounded_aci_request(&exact).unwrap().len(), MAX_ACI_REQUEST_BODY_BYTES);
        assert!(matches!(serialize_bounded_aci_request(&(exact + "x")), Err(LlmError::AciRequestRejected)));
        let escaped = "\u{0}".repeat(MAX_ACI_REQUEST_BODY_BYTES / 6 + 1);
        assert!(escaped.len() < MAX_ACI_REQUEST_BODY_BYTES);
        assert!(matches!(serialize_bounded_aci_request(&escaped), Err(LlmError::AciRequestRejected)));
    }

    // [PHALA-ACI-SERDE-BYTE-VECTORS 2026-10-08 by Codex] Authored only.
    // Literal byte expectations avoid using the serializer under test as its
    // own oracle. These local vectors are not a deployed-gateway profile pin.
    #[test]
    fn aci_json_writer_matches_explicit_value_roundtrip_bytes() {
        let vectors = [
            (
                r#" { "z": 1, "a": 2, "nested": {"b":true,"a":null} } "#,
                r#"{"z":1,"a":2,"nested":{"b":true,"a":null}}"#,
            ),
            (
                r#"{"integer":1,"float":1.0,"exponent":1e0,"negative_zero":-0}"#,
                r#"{"integer":1,"float":1.0,"exponent":1.0,"negative_zero":-0.0}"#,
            ),
            (
                r#"{"u64":18446744073709551615,"i64":-9223372036854775808}"#,
                r#"{"u64":18446744073709551615,"i64":-9223372036854775808}"#,
            ),
            (
                r#"{"s":"\u4e2d\u6587\uD83D\uDE00\/\u2028\u2029\u003c\u003e\u0026"}"#,
                "{\"s\":\"\u{4e2d}\u{6587}\u{1f600}/\u{2028}\u{2029}<>&\"}",
            ),
            (
                r#"{"s":"\u0000\u0001\u0008\u0009\u000A\u000C\u000D\u001F\"\\"}"#,
                r#"{"s":"\u0000\u0001\b\t\n\f\r\u001f\"\\"}"#,
            ),
            (
                r#"{"messages":[{"content":[{"type":"text","text":"[1,2]"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"text","text":"[1,2]"}]}]}"#,
            ),
        ];
        for (input, expected) in vectors {
            let value: serde_json::Value = serde_json::from_slice(input.as_bytes()).unwrap();
            assert_eq!(serialize_bounded_aci_request(&value).unwrap(), expected.as_bytes());
        }

        // A source-side reconstruction must not collapse these byte-distinct
        // representations just because their numeric/string values look alike.
        for (left, right) in [
            (r#"{"n":1}"#, r#"{"n":1.0}"#),
            (r#"{"n":0}"#, r#"{"n":-0}"#),
            (r#"{"z":1,"a":2}"#, r#"{"a":2,"z":1}"#),
            (r#"{"content":"[1,2]"}"#, r#"{"content":[1,2]}"#),
        ] {
            let left: serde_json::Value = serde_json::from_slice(left.as_bytes()).unwrap();
            let right: serde_json::Value = serde_json::from_slice(right.as_bytes()).unwrap();
            assert_ne!(serialize_bounded_aci_request(&left).unwrap(),
                serialize_bounded_aci_request(&right).unwrap());
        }
    }

    // [PHALA-ACI-NUMERIC-PROFILE 2026-10-08 by Codex] Authored only: this
    // sentinel distinguishes the pinned reference's default decimal parser
    // from float_roundtrip/arbitrary_precision feature unification. It is not
    // a deployed-gateway capability, nor permission to round source intent.
    #[test]
    fn aci_candidate_numeric_profile_detects_parser_feature_drift() {
        let value: serde_json::Value =
            serde_json::from_slice(br#"{"n":51.248178375505404}"#).unwrap();
        let number = value.get("n").unwrap().as_number().unwrap();
        assert!(number.is_f64());
        assert_eq!(number.as_f64().unwrap().to_bits(), 51.24817837550541_f64.to_bits());
        assert_ne!(number.as_f64().unwrap().to_bits(), 51.248178375505404_f64.to_bits());
        assert_eq!(serialize_bounded_aci_request(&value).unwrap(),
            br#"{"n":51.24817837550541}"#);

        // An arbitrary-precision Value would retain this out-of-f64 exponent.
        // Ordinary numeric representation rejects it instead of retaining a
        // number that the candidate received-body reconstruction cannot hold.
        assert!(serde_json::from_slice::<serde_json::Value>(br#"{"n":1e400}"#).is_err());
    }

    #[test]
    fn aci_preflight_bounds_fields_iteration_and_numeric_options() {
        let mut request = ChatRequest::simple("synthetic");
        request.require_aci_verified = true;
        assert!(request.validate_aci_bounds().is_ok());
        request.messages = vec![ChatMessage::user(""); 128];
        assert!(request.validate_aci_bounds().is_ok());
        request.messages.push(ChatMessage::user(""));
        assert!(matches!(request.validate_aci_bounds(), Err(LlmError::AciRequestRejected)));
        request.messages = vec![ChatMessage { role: "r".repeat(64), content: String::new() }];
        assert!(request.validate_aci_bounds().is_ok());
        request.messages[0].role.push('r');
        assert!(request.validate_aci_bounds().is_err());
        request.messages = vec![ChatMessage::user("x".repeat(MAX_ACI_REQUEST_BODY_BYTES / 2)); 2];
        assert!(request.validate_aci_bounds().is_err());
        request.messages = vec![ChatMessage::user("synthetic")];
        request.stop = Some(vec!["x".repeat(1024); 16]);
        assert!(request.validate_aci_bounds().is_ok());
        request.stop.as_mut().unwrap().push(String::new());
        assert!(request.validate_aci_bounds().is_err());
        request.stop = Some(vec!["x".repeat(1025)]);
        assert!(request.validate_aci_bounds().is_err());
        for temperature in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 2.1] {
            assert!(validate_aci_request_options("model", None, Some(temperature)).is_err());
        }
        for temperature in [0.0, 2.0] {
            assert!(validate_aci_request_options("model", Some(1), Some(temperature)).is_ok());
        }
        assert!(validate_aci_request_options(" ", None, None).is_err());
        assert!(validate_aci_request_options(&"m".repeat(256), None, None).is_ok());
        assert!(validate_aci_request_options(&"m".repeat(257), None, None).is_err());
        assert!(validate_aci_request_options("model", Some(0), None).is_err());
        request = ChatRequest::simple("synthetic");
        request.model_override = Some("m".repeat(257));
        assert!(request.validate_aci_bounds().is_err());
        request.model_override = None;
        request.max_tokens = Some(0);
        assert!(request.validate_aci_bounds().is_err());
        request.max_tokens = None;
        request.temperature = Some(f32::NAN);
        assert!(request.validate_aci_bounds().is_err());
        request.temperature = None;
        request.messages.clear();
        assert!(request.validate_aci_bounds().is_err());
    }

    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Authored,
    // not run: deterministic floor/error/age checks need no clock sleeps.
    #[test]
    fn phala_observation_preserves_floor_and_rejects_monotonic_expiry() {
        let mut observation = PhalaAciObservation {
            started: Instant::now(), floor: AtomicU64::new(100),
        };
        assert_eq!(observation.observe_at(Ok(100)).unwrap(), 100);
        assert_eq!(observation.observe_at(Ok(101)).unwrap(), 101);
        assert!(matches!(observation.observe_at(Ok(100)), Err(LlmError::AciVerifierUnavailable)));
        assert_eq!(observation.floor.load(Ordering::Acquire), 101);
        assert!(matches!(observation.observe_at(Err(LlmError::AciVerifierUnavailable)),
            Err(LlmError::AciVerifierUnavailable)));
        assert_eq!(observation.floor.load(Ordering::Acquire), 101);
        assert_eq!(observation.observe_at(Ok(101)).unwrap(), 101);
        observation.started = Instant::now().checked_sub(PHALA_ACI_CONTEXT_MAX_AGE + Duration::from_secs(1)).unwrap();
        assert!(matches!(observation.observe_at(Ok(102)), Err(LlmError::AciVerifierUnavailable)));
        assert_eq!(observation.floor.load(Ordering::Acquire), 101);
    }

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Authored, not executed:
    // distinct digests expose accidental identity/keyset aliasing.
    #[test]
    fn aci_response_hints_match_pinned_aci1_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-aci-version", "aci/1".parse().unwrap());
        headers.insert("x-aci-identity", format!("sha256:{}", "b".repeat(64)).parse().unwrap());
        headers.insert(
            "x-aci-keyset-digest",
            format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        );
        headers.insert("x-receipt-id", "receipt-123".parse().unwrap());

        let hints = AciResponseHints::from_headers(&headers).unwrap();
        assert_eq!(hints.version, "aci/1");
        assert!(hints.has_valid_shape());
        assert_eq!(hints.workload_id, format!("sha256:{}", "b".repeat(64)));
        assert_ne!(hints.workload_id, hints.keyset_digest);
        assert_eq!(hints.keyset_digest, format!("sha256:{}", "a".repeat(64)));
        assert_eq!(hints.receipt_id, "receipt-123");

        headers.insert("x-aci-identity", "not-a-digest".parse().unwrap());
        assert!(AciResponseHints::from_headers(&headers).is_err());
        headers.insert("x-aci-identity", format!("sha256:{}", "b".repeat(64)).parse().unwrap());
        for name in ["x-aci-identity", "x-aci-keyset-digest", "x-aci-version", "x-receipt-id"] {
            let saved = headers.remove(name).unwrap();
            assert!(matches!(AciResponseHints::from_headers(&headers),
                Err(LlmError::AciResponseContractViolation)), "missing {name}");
            headers.insert(name, saved);
        }
    }

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Authored, not executed.
    #[test]
    fn aci_response_hints_reject_duplicate_security_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-aci-version", "aci/1".parse().unwrap());
        headers.insert("x-aci-identity", format!("sha256:{}", "b".repeat(64)).parse().unwrap());
        headers.insert(
            "x-aci-keyset-digest",
            format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        );
        headers.insert("x-receipt-id", "receipt-123".parse().unwrap());

        assert!(AciResponseHints::from_headers(&headers).is_ok());
        for name in ["x-aci-identity", "x-aci-keyset-digest", "x-aci-version", "x-receipt-id"] {
            let mut duplicate = headers.clone();
            duplicate.append(name, headers.get(name).unwrap().clone());
            assert!(matches!(AciResponseHints::from_headers(&duplicate),
                Err(LlmError::AciResponseContractViolation)), "duplicate {name}");
        }
    }

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Authored, not executed.
    #[test]
    fn aci_response_hints_reject_wrong_version_and_malformed_digest() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-aci-version", "aci/2".parse().unwrap());
        headers.insert("x-aci-identity", format!("sha256:{}", "b".repeat(64)).parse().unwrap());
        headers.insert(
            "x-aci-keyset-digest",
            format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        );
        headers.insert("x-receipt-id", "receipt-123".parse().unwrap());
        assert!(matches!(
            AciResponseHints::from_headers(&headers),
            Err(LlmError::AciResponseContractViolation)
        ));

        headers.insert("x-aci-version", "aci/1".parse().unwrap());
        headers.insert("x-aci-keyset-digest", "sha256:xyz".parse().unwrap());
        assert!(matches!(
            AciResponseHints::from_headers(&headers),
            Err(LlmError::AciResponseContractViolation)
        ));

        headers.insert(
            "x-aci-keyset-digest",
            format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        );
        headers.insert("x-receipt-id", "invalid receipt id".parse().unwrap());
        assert!(matches!(
            AciResponseHints::from_headers(&headers),
            Err(LlmError::AciResponseContractViolation)
        ));
    }

    #[test]
    fn api_error_display_truncates_on_utf8_boundary() {
        let error = LlmError::ApiError {
            status: 500,
            body: "界".repeat(100),
        };
        let rendered = error.to_string();

        assert!(rendered.starts_with("API error 500: "));
        assert!(rendered.is_char_boundary(rendered.len()));
        assert!(!rendered.contains('\u{fffd}'));
        assert!(rendered.len() <= "API error 500: ".len() + MAX_LLM_ERROR_DISPLAY_BYTES);
    }

    #[test]
    fn utf8_prefix_handles_ascii_unicode_and_zero_budget() {
        assert_eq!(utf8_prefix("abcdef", 3), "abc");
        assert_eq!(utf8_prefix("认证模块", 7), "认证");
        assert_eq!(utf8_prefix("认证模块", 0), "");
        assert_eq!(utf8_prefix("short", usize::MAX), "short");
    }

    #[test]
    fn oversized_response_error_is_bounded_and_stable() {
        let error = response_body_too_large(MAX_LLM_SUCCESS_BODY_BYTES);
        assert!(matches!(error, LlmError::ParseError(_)));
        assert!(error.to_string().contains("exceeds"));
        assert!(error.to_string().contains("8388608"));
    }

    #[test]
    fn bounded_body_accepts_exact_limit_and_rejects_all_overflow_paths() {
        let mut body = BoundedLlmBody::new(5, Some(5)).unwrap();
        body.push(b"ab").unwrap();
        body.push(b"cde").unwrap();
        assert_eq!(body.into_bytes(), b"abcde");

        assert!(BoundedLlmBody::new(5, Some(6)).is_err());

        let mut chunked = BoundedLlmBody::new(5, None).unwrap();
        chunked.push(b"abc").unwrap();
        assert!(chunked.push(b"def").is_err());
        assert_eq!(chunked.into_bytes(), b"abc");
    }

    #[test]
    fn provider_health_recovers_after_monotonic_cooldown() {
        let health = ProviderHealth::default();
        assert!(health.is_healthy_at(100));

        health.mark_unhealthy_at(100, 30_000);
        assert!(!health.is_healthy_at(100));
        assert!(!health.is_healthy_at(30_099));
        assert!(health.is_healthy_at(30_100));

        health.mark_unhealthy_at(200, 1_000);
        assert!(!health.is_healthy_at(30_099));
        health.mark_healthy_at(200);
        assert!(!health.is_healthy_at(200));
        health.mark_healthy_at(30_100);
        assert!(health.is_healthy_at(200));
    }

    #[test]
    fn rate_limit_cooldown_is_clamped_to_operational_bounds() {
        assert_eq!(rate_limit_cooldown(Some(0)), Duration::from_secs(1));
        assert_eq!(rate_limit_cooldown(None), DEFAULT_LLM_PROVIDER_COOLDOWN);
        assert_eq!(
            rate_limit_cooldown(Some(MAX_LLM_PROVIDER_COOLDOWN_SECS + 1)),
            Duration::from_secs(MAX_LLM_PROVIDER_COOLDOWN_SECS)
        );
    }

    #[test]
    fn provider_api_base_rejects_ambiguous_or_secret_bearing_urls() {
        assert_eq!(
            normalize_llm_api_base(" https://api.example.com/v1/// ").unwrap(),
            "https://api.example.com/v1"
        );
        for invalid in [
            "",
            "api.example.com",
            "ftp://api.example.com",
            "https://user:secret@api.example.com",
            "https://api.example.com/v1?token=secret",
            "https://api.example.com/v1#fragment",
        ] {
            assert_eq!(
                normalize_llm_api_base(invalid),
                Err(LlmProviderInitError::InvalidApiBase)
            );
        }
    }

    #[test]
    fn provider_key_resolution_distinguishes_keyless_and_required_modes() {
        assert_eq!(resolve_llm_api_key("", false).unwrap(), "");
        assert_eq!(
            resolve_llm_api_key("", true),
            Err(LlmProviderInitError::ProviderSecretRequired)
        );
        assert_eq!(
            resolve_llm_api_key(
                "$AERONYX_TEST_SUPERNODE_SECRET_MUST_NOT_EXIST_20260814",
                false
            ),
            Err(LlmProviderInitError::ProviderSecretUnavailable)
        );
    }

    #[test]
    fn runtime_reason_codes_do_not_expose_provider_diagnostics() {
        let cases = [
            (
                LlmError::Transport("https://secret.invalid".into()),
                "llm_transport_error",
            ),
            (
                LlmError::ApiError {
                    status: 500,
                    body: "private provider response".into(),
                },
                "llm_api_error",
            ),
            (
                LlmError::ParseError("private response fragment".into()),
                "llm_response_parse_error",
            ),
            (LlmError::EmptyResponse, "llm_empty_response"),
            (
                LlmError::RateLimit {
                    retry_after_secs: Some(10),
                },
                "llm_rate_limited",
            ),
            (LlmError::ContextTooLong, "llm_context_too_long"),
            (LlmError::AciRequestRejected, "llm_aci_request_rejected"),
            (
                LlmError::NotConfigured("private-provider-name".into()),
                "llm_provider_not_configured",
            ),
            (
                LlmError::ConfidentialServingRequired,
                "llm_confidential_serving_required",
            ),
            (
                LlmError::ConfidentialE2eeTransportUnavailable,
                "llm_client_to_tee_e2ee_unavailable",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.reason_code(), expected);
            assert!(!error.reason_code().contains("private"));
            assert!(!error.reason_code().contains("secret"));
        }
    }
}
