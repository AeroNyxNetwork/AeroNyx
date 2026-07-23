// ============================================
// File: crates/aeronyx-server/src/api/auth.rs
// ============================================
//! # Auth — JWT Token Issuance for SaaS Mode
//!
//! ## Modification History
//! v2.0.0-ServerNonce - Add a stateless, HMAC-authenticated server nonce flow
//!                      for fresh Ed25519 token challenges while retaining the
//!                      v1 timestamp endpoint for backward compatibility.
//! v1.2.0-ReplayGuard - Reject repeated use of an already verified v1 token
//!                      challenge through a bounded, expiring replay cache.
//! v1.1.2-PortablePathTest - Restrict the raw non-UTF-8 filename test to
//!                      non-Apple Unix targets whose filesystems accept it.
//! v1.1.1-CrashSafe   - Replace secret-bearing configuration through a synced
//!                      same-directory temporary file and atomic rename while
//!                      preserving permissions and valid platform file names.
//! v1.1.0-SectionSafe - Persist generated secrets under `[memchain]`, migrate
//!                      compatible legacy `[auth]` values, and validate the
//!                      resulting TOML before replacing the configuration.
//! v1.0.0-MultiTenant - Initial implementation (Task 2)
//! v1.0.1-Fix         - write_secret_to_config promoted to `pub`.
//!                      Added `write_secret_to_config_pub` as an explicit
//!                      public alias used by server.rs for api_secret
//!                      persistence (ensure_api_secret_on_disk).
//!                      Comment-line guard added to write_secret_to_config
//!                      to prevent false replacement of commented-out keys.
//!
//! ## Main Functionality
//! - `POST /api/auth/challenge`: Issue a stateless v2 server nonce challenge
//! - `POST /api/auth/token/v2`: Verify and consume a signed v2 challenge
//! - `POST /api/auth/token`: Verify legacy v1 timestamp challenge, issue JWT
//! - `build_auth_router()`: Own all anonymous auth routes and body limits
//! - `issue_jwt()`: Sign a JWT with HS256 using the configured secret
//! - `verify_jwt()`: Validate JWT signature + expiry, extract Claims
//! - `TokenReplayGuard`: Bound and expire accepted v1/v2 token challenges
//! - `ensure_jwt_secret()`: Auto-generate and persist the JWT secret
//! - `ensure_api_secret()`: Resolve, migrate, or generate the MPI admin secret
//! - `write_secret_to_config_pub()`: Section-safe config persistence alias
//!
//! ## Authentication Flow (v2)
//! 1. Client posts its Ed25519 public key to `/api/auth/challenge`.
//! 2. Server returns an OS-random nonce, canonical message, timestamps, and a
//!    domain-separated HMAC that binds every challenge field.
//! 3. Client signs the canonical message and posts the signed fields to
//!    `/api/auth/token/v2`.
//! 4. Server verifies HMAC, validity window, Ed25519 signature, and one-time
//!    replay claim before issuing a JWT where `sub = pubkey_hex`.
//! 5. Subsequent MPI requests include `Authorization: Bearer <jwt>`.
//!
//! ## JWT Format
//! - Algorithm: HS256
//! - `sub`:  64-char hex Ed25519 public key
//! - `iat`:  issue time (Unix seconds)
//! - `exp`:  expiry (iat + token_ttl_secs, default 86400 = 24h)
//! - `iss`:  "memchain"
//!
//! ## Security Notes
//! - `sub` is the SINGLE SOURCE OF TRUTH for owner identity in SaaS mode.
//!   Handlers extract owner ONLY from verified JWT claims, never from
//!   request headers or body fields.
//! - JWT secret auto-generation: 64-char alphanumeric random string,
//!   written back to config.toml on first SaaS startup.
//! - Generated API and JWT secrets belong to `[memchain]`. Legacy values under
//!   `[auth]` are migrated once so upgrades do not rotate working credentials.
//! - Timestamp window is +/-60 seconds (not 300s like the remote storage path)
//!   because token issuance is more sensitive than request signing.
//! - The v1 replay guard blocks reuse after the first accepted request and is
//!   capacity-bounded to resist memory exhaustion. It cannot prevent a stolen
//!   signature from winning a first-use race; a server-issued nonce protocol
//!   remains the required v2 design for complete challenge freshness.
//! - v2 challenge issuance is stateless. The node does not retain unverified
//!   nonces, so anonymous challenge requests cannot grow server memory.
//! - v2 HMAC uses a domain-separated key derived from the JWT secret. This
//!   prevents challenge MACs from becoming valid JWT signatures or vice versa.
//! - Successful challenge and token responses set `Cache-Control: no-store`.
//! - One-time replay claims are process-local. A future horizontally scaled
//!   SaaS deployment that shares one JWT secret across instances MUST use a
//!   shared atomic replay store (or issuer-affine routing) before enabling v2.
//!
//! ## Dependencies
//! - `jsonwebtoken` crate for HS256 JWT sign/verify
//! - `ed25519-dalek` via `aeronyx_core::crypto` for signature verification
//! - Used by `mpi.rs` router (SaaS mode only, excluded from local mode)
//! - `ensure_jwt_secret()` called from `server.rs` during SaaS init
//! - `ensure_api_secret()` called from `main.rs` before Server construction
//!
//! ⚠️ Important Notes for Next Developer:
//! - This endpoint is EXCLUDED from `unified_auth_middleware` — it must be
//!   registered before the middleware layer in `build_mpi_router()`.
//! - `ed25519-dalek` v2.x changed the Verifier trait API. This code uses
//!   `IdentityPublicKey::verify()` from aeronyx-core, which abstracts the
//!   version difference — do NOT call dalek directly.
//! - JWT secret minimum length: 32 bytes. Auto-generated secrets are 64 chars.
//! - The v2 canonical message is a protocol surface. Keep field order, labels,
//!   newline separators, URL-safe unpadded base64, and domain text stable.
//! - Do not move anonymous auth routes behind a shared cluster load balancer
//!   without preserving the process-local one-time claim invariant above.
//! - `ensure_jwt_secret()` returns Err if the config file cannot be written.
//!   Server startup should fail loudly in this case.
//! - `write_secret_to_config` is `pub` for compatibility. It now writes only
//!   inside `[memchain]`; never reintroduce global first-key replacement.
//! - Secret persistence must remain a same-directory, synced atomic replace.
//!   Directly truncating the live TOML can make the node unbootable after a
//!   crash and can expose a partially written credential.
//! - Comment-line guard: lines whose first non-whitespace char is '#' are
//!   never modified, preventing false replacement of commented-out examples.
//!
//! ## Last Modified
//! v2.0.0-ServerNonce - Added stateless signed challenges and v2 token issuance.
//! v1.2.0-ReplayGuard - Added bounded one-time claims for verified v1 requests.
//! v1.1.1-CrashSafe - Made secret configuration replacement durable and atomic.
//! v1.1.0-SectionSafe - Fixed cross-section secret replacement and first-start
//!                      runtime authentication gap; added legacy migration.
//! v1.0.1-Fix - Promoted write_secret_to_config to pub; added _pub alias;
//!              added comment-line guard in write_secret_to_config.
// ============================================

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Error as IoError, ErrorKind as IoErrorKind, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::DefaultBodyLimit;
use axum::http::header::{CACHE_CONTROL, PRAGMA};
use axum::http::HeaderValue;
use axum::response::Response;
use axum::routing::post;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json, Router};
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64URL};
use base64::Engine as _;
use hmac::{Hmac, Mac};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use parking_lot::Mutex;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use aeronyx_core::crypto::keys::IdentityPublicKey;

// ============================================
// Constants
// ============================================

/// Issuer claim embedded in all issued JWTs.
pub const JWT_ISSUER: &str = "memchain";

/// Maximum allowed clock skew for token request timestamps (seconds).
const TIMESTAMP_TOLERANCE_SECS: u64 = 60;

/// Server nonce lifetime. Client wall-clock time is not part of v2 validity.
const TOKEN_CHALLENGE_V2_TTL_SECS: u64 = 120;

/// Tolerate a small backward clock adjustment between challenge and exchange.
const TOKEN_CHALLENGE_V2_CLOCK_SKEW_SECS: u64 = 5;

/// OS-random nonce size for v2 challenges.
const TOKEN_CHALLENGE_V2_NONCE_BYTES: usize = 32;

/// Stable protocol domain used in the exact client-signed v2 message.
const TOKEN_CHALLENGE_V2_DOMAIN: &str = "AeroNyx Auth Token v2";

/// Domain used when deriving the challenge HMAC subkey from the JWT secret.
const TOKEN_CHALLENGE_V2_KEY_DOMAIN: &[u8] = b"aeronyx-auth-token-v2-hmac-key\0";

/// Maximum verified challenges retained during the replay window.
const TOKEN_REPLAY_CACHE_MAX_ENTRIES: usize = 65_536;

/// Avoid an O(n) expiry scan on every token request.
const TOKEN_REPLAY_CLEANUP_INTERVAL_SECS: u64 = 10;

/// Canonical TOML section for MemChain API and JWT secrets.
const MEMCHAIN_CONFIG_SECTION: &str = "memchain";

/// Historical section used by the pre-v1.1.0 text writer.
const LEGACY_AUTH_CONFIG_SECTION: &str = "auth";

/// Process-wide guard shared by every SaaS auth router in this node process.
static TOKEN_REPLAY_GUARD: OnceLock<TokenReplayGuard> = OnceLock::new();

type HmacSha256 = Hmac<Sha256>;

// ============================================
// Token Replay Guard
// ============================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenReplayDecision {
    Accepted,
    Duplicate,
    Saturated,
}

#[derive(Debug, Default)]
struct TokenReplayCache {
    entries: HashMap<[u8; 32], u64>,
    next_cleanup_at: u64,
}

/// Bounded one-time claim store for verified token challenges.
///
/// The cache stores only a domain-separated digest of the challenge identity,
/// never the signature, nonce, canonical message, or an issued token. Expiry
/// is tied to the corresponding endpoint's acceptance window. Capacity
/// exhaustion fails closed.
#[derive(Debug)]
struct TokenReplayGuard {
    cache: Mutex<TokenReplayCache>,
    capacity: usize,
}

impl TokenReplayGuard {
    fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(TokenReplayCache::default()),
            capacity,
        }
    }

    fn claim(&self, fingerprint: [u8; 32], expires_at: u64, now: u64) -> TokenReplayDecision {
        let mut cache = self.cache.lock();
        if now >= cache.next_cleanup_at {
            cache.entries.retain(|_, expiry| *expiry >= now);
            cache.next_cleanup_at = now.saturating_add(TOKEN_REPLAY_CLEANUP_INTERVAL_SECS);
        }

        if cache.entries.contains_key(&fingerprint) {
            return TokenReplayDecision::Duplicate;
        }
        if cache.entries.len() >= self.capacity {
            return TokenReplayDecision::Saturated;
        }

        cache.entries.insert(fingerprint, expires_at);
        TokenReplayDecision::Accepted
    }
}

fn token_replay_guard() -> &'static TokenReplayGuard {
    TOKEN_REPLAY_GUARD.get_or_init(|| TokenReplayGuard::new(TOKEN_REPLAY_CACHE_MAX_ENTRIES))
}

fn token_challenge_fingerprint(pubkey: &[u8; 32], timestamp: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"aeronyx-auth-token-v1\0");
    digest.update(pubkey);
    digest.update(timestamp.to_be_bytes());
    digest.finalize().into()
}

fn token_challenge_v2_fingerprint(challenge: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"aeronyx-auth-token-v2-replay\0");
    digest.update(challenge.as_bytes());
    digest.finalize().into()
}

/// Build the byte-exact, domain-separated message that v2 clients sign.
fn token_challenge_v2_message(
    canonical_pubkey: &str,
    nonce_b64: &str,
    issued_at: u64,
    expires_at: u64,
) -> String {
    format!(
        "{}\npubkey={}\nnonce={}\nissued_at={}\nexpires_at={}",
        TOKEN_CHALLENGE_V2_DOMAIN, canonical_pubkey, nonce_b64, issued_at, expires_at
    )
}

fn derive_token_challenge_v2_key(jwt_secret: &str) -> Result<[u8; 32], &'static str> {
    let mut derivation = HmacSha256::new_from_slice(jwt_secret.as_bytes())
        .map_err(|_| "invalid JWT secret for challenge key derivation")?;
    derivation.update(TOKEN_CHALLENGE_V2_KEY_DOMAIN);
    Ok(derivation.finalize().into_bytes().into())
}

fn token_challenge_v2_mac(jwt_secret: &str, challenge: &str) -> Result<[u8; 32], &'static str> {
    let derived_key = derive_token_challenge_v2_key(jwt_secret)?;
    let mut mac =
        HmacSha256::new_from_slice(&derived_key).map_err(|_| "invalid derived challenge key")?;
    mac.update(challenge.as_bytes());
    Ok(mac.finalize().into_bytes().into())
}

fn verify_token_challenge_v2_mac(
    jwt_secret: &str,
    challenge: &str,
    expected_mac: &[u8; 32],
) -> bool {
    let Ok(derived_key) = derive_token_challenge_v2_key(jwt_secret) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(&derived_key) else {
        return false;
    };
    mac.update(challenge.as_bytes());
    mac.verify_slice(expected_mac).is_ok()
}

fn generate_token_challenge_v2_nonce() -> Result<[u8; TOKEN_CHALLENGE_V2_NONCE_BYTES], String> {
    let mut nonce = [0u8; TOKEN_CHALLENGE_V2_NONCE_BYTES];
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|error| format!("operating-system randomness unavailable: {}", error))?;
    Ok(nonce)
}

fn parse_base64url_32(value: &str, label: &str) -> Result<[u8; 32], String> {
    let decoded = BASE64URL
        .decode(value)
        .map_err(|_| format!("{} is not valid URL-safe unpadded base64", label))?;
    if decoded.len() != 32 {
        return Err(format!("{} must decode to exactly 32 bytes", label));
    }
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&decoded);
    Ok(bytes)
}

fn auth_response_no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
        .headers_mut()
        .insert(PRAGMA, HeaderValue::from_static("no-cache"));
    response
}

fn auth_error_response(status: StatusCode, error: &str, code: &str) -> Response {
    auth_response_no_store(
        (
            status,
            Json(serde_json::json!({
                "error": error,
                "code": code,
            })),
        )
            .into_response(),
    )
}

/// Preserve the legacy v1 JSON contract while applying the shared cache policy.
///
/// v1 clients historically receive only `{ "error": ... }`; adding the v2
/// machine-readable `code` field here could break strict response decoders.
fn legacy_auth_error_response(status: StatusCode, error: &str) -> Response {
    auth_response_no_store((status, Json(serde_json::json!({ "error": error }))).into_response())
}

// ============================================
// Public Types
// ============================================

/// JWT claims payload.
///
/// `sub` is the 64-char hex Ed25519 public key — the single source of truth
/// for owner identity in SaaS mode. All other fields are standard JWT claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Ed25519 public key (64 hex chars) — the authenticated owner.
    pub sub: String,
    /// Issued-at time (Unix seconds).
    pub iat: u64,
    /// Expiry time (Unix seconds). Validated by verify_jwt().
    pub exp: u64,
    /// Issuer — always "memchain".
    pub iss: String,
}

/// Request body for `POST /api/auth/token`.
#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    /// Ed25519 public key as 64 hex chars.
    pub pubkey: String,
    /// Unix timestamp (seconds) at time of signing.
    pub timestamp: u64,
    /// Ed25519 signature of `"{pubkey}:{timestamp}"` as base64.
    pub signature: String,
}

/// Request body for `POST /api/auth/challenge`.
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenChallengeV2Request {
    /// Ed25519 public key as 64 hex chars. The response canonicalizes case.
    pub pubkey: String,
}

/// Stateless challenge returned by `POST /api/auth/challenge`.
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenChallengeV2Response {
    /// Protocol version for explicit client dispatch.
    pub version: u8,
    /// Canonical lower-case Ed25519 public key.
    pub pubkey: String,
    /// 32-byte nonce encoded as URL-safe unpadded base64.
    pub nonce: String,
    /// Server issue time in Unix seconds.
    pub issued_at: u64,
    /// Server expiry time in Unix seconds.
    pub expires_at: u64,
    /// Exact UTF-8 message the client must sign.
    pub challenge: String,
    /// HMAC-SHA256 binding all canonical challenge fields.
    pub challenge_mac: String,
}

/// Request body for `POST /api/auth/token/v2`.
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenV2Request {
    /// Ed25519 public key as 64 hex chars.
    pub pubkey: String,
    /// Nonce copied from the server challenge response.
    pub nonce: String,
    /// Server issue time copied from the challenge response.
    pub issued_at: u64,
    /// Server expiry time copied from the challenge response.
    pub expires_at: u64,
    /// Server HMAC copied from the challenge response.
    pub challenge_mac: String,
    /// Ed25519 signature of the exact canonical challenge, standard base64.
    pub signature: String,
}

/// Successful response from `POST /api/auth/token`.
#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub token: String,
    pub expires_at: u64,
}

// ============================================
// Shared State for Auth Endpoint
// ============================================

/// Minimal state required by the auth endpoint.
///
/// Extracted from MpiState so the endpoint can be registered outside
/// the unified_auth_middleware layer. Passed as Axum State.
#[derive(Clone)]
pub struct AuthState {
    /// HS256 signing secret for JWT issuance.
    pub jwt_secret: String,
    /// Token validity duration in seconds (default 86400 = 24h).
    pub token_ttl_secs: u64,
}

/// Build the complete anonymous SaaS authentication router.
///
/// Keeping route registration and request-size policy beside the handlers
/// prevents callers from accidentally exposing an auth endpoint without the
/// same 16 KiB body limit. The returned router is merged into MPI only in SaaS
/// mode; Local mode never registers these routes.
pub fn build_auth_router(state: AuthState) -> Router {
    Router::new()
        .route("/api/auth/challenge", post(issue_token_challenge_v2))
        .route("/api/auth/token/v2", post(issue_token_v2))
        .route("/api/auth/token", post(issue_token))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(state)
}

// ============================================
// Endpoint Handler
// ============================================

/// `POST /api/auth/challenge`
///
/// Issues a stateless v2 challenge bound to the requested public key. The
/// response HMAC allows `/api/auth/token/v2` to authenticate every challenge
/// field without retaining anonymous pending requests in server memory.
pub async fn issue_token_challenge_v2(
    State(state): State<AuthState>,
    Json(req): Json<TokenChallengeV2Request>,
) -> Response {
    let pubkey_bytes = match parse_pubkey_hex(&req.pubkey) {
        Ok(bytes) => bytes,
        Err(message) => {
            return auth_error_response(StatusCode::BAD_REQUEST, message, "invalid_public_key");
        }
    };
    if state.jwt_secret.len() < 32 {
        tracing::error!("[AUTH_V2] Refusing challenge issuance with invalid JWT secret");
        return auth_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "authentication service is not configured",
            "auth_service_misconfigured",
        );
    }

    let issued_at = now_secs();
    let Some(expires_at) = issued_at.checked_add(TOKEN_CHALLENGE_V2_TTL_SECS) else {
        tracing::error!("[AUTH_V2] Challenge expiry overflow");
        return auth_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "authentication service clock is invalid",
            "auth_clock_invalid",
        );
    };
    let nonce = match generate_token_challenge_v2_nonce() {
        Ok(nonce) => nonce,
        Err(error) => {
            tracing::error!(error = %error, "[AUTH_V2] Secure nonce generation failed");
            return auth_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "secure randomness is unavailable",
                "auth_randomness_unavailable",
            );
        }
    };

    let canonical_pubkey = hex::encode(pubkey_bytes);
    let nonce_b64 = BASE64URL.encode(nonce);
    let challenge =
        token_challenge_v2_message(&canonical_pubkey, &nonce_b64, issued_at, expires_at);
    let challenge_mac = match token_challenge_v2_mac(&state.jwt_secret, &challenge) {
        Ok(mac) => BASE64URL.encode(mac),
        Err(error) => {
            tracing::error!(error = %error, "[AUTH_V2] Challenge MAC generation failed");
            return auth_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "authentication service is not configured",
                "auth_service_misconfigured",
            );
        }
    };

    debug!(
        pubkey = &canonical_pubkey[..8],
        expires_at, "[AUTH_V2] Stateless token challenge issued"
    );
    auth_response_no_store(
        (
            StatusCode::OK,
            Json(TokenChallengeV2Response {
                version: 2,
                pubkey: canonical_pubkey,
                nonce: nonce_b64,
                issued_at,
                expires_at,
                challenge,
                challenge_mac,
            }),
        )
            .into_response(),
    )
}

/// `POST /api/auth/token/v2`
///
/// Verifies a server-issued stateless challenge and consumes it once before
/// issuing a JWT. HMAC verification happens before Ed25519 verification so
/// arbitrary attacker-controlled requests cannot force expensive public-key
/// work without first presenting a challenge created by this node.
pub async fn issue_token_v2(
    State(state): State<AuthState>,
    Json(req): Json<TokenV2Request>,
) -> Response {
    let pubkey_bytes = match parse_pubkey_hex(&req.pubkey) {
        Ok(bytes) => bytes,
        Err(message) => {
            return auth_error_response(StatusCode::BAD_REQUEST, message, "invalid_public_key");
        }
    };
    let nonce = match parse_base64url_32(&req.nonce, "nonce") {
        Ok(bytes) => bytes,
        Err(message) => {
            return auth_error_response(StatusCode::BAD_REQUEST, &message, "invalid_nonce");
        }
    };
    let challenge_mac = match parse_base64url_32(&req.challenge_mac, "challenge_mac") {
        Ok(bytes) => bytes,
        Err(message) => {
            return auth_error_response(StatusCode::BAD_REQUEST, &message, "invalid_challenge_mac");
        }
    };

    let now = now_secs();
    let valid_duration = req
        .issued_at
        .checked_add(TOKEN_CHALLENGE_V2_TTL_SECS)
        .is_some_and(|expected| expected == req.expires_at);
    let issued_too_far_ahead =
        req.issued_at > now.saturating_add(TOKEN_CHALLENGE_V2_CLOCK_SKEW_SECS);
    if !valid_duration || issued_too_far_ahead || now > req.expires_at {
        debug!(
            pubkey = &req.pubkey[..8],
            issued_at = req.issued_at,
            expires_at = req.expires_at,
            "[AUTH_V2] Token request rejected: challenge timing invalid"
        );
        return auth_error_response(
            StatusCode::UNAUTHORIZED,
            "token challenge is invalid or expired",
            "token_challenge_expired",
        );
    }

    let canonical_pubkey = hex::encode(pubkey_bytes);
    let nonce_b64 = BASE64URL.encode(nonce);
    let challenge =
        token_challenge_v2_message(&canonical_pubkey, &nonce_b64, req.issued_at, req.expires_at);
    if !verify_token_challenge_v2_mac(&state.jwt_secret, &challenge, &challenge_mac) {
        debug!(
            pubkey = &canonical_pubkey[..8],
            "[AUTH_V2] Token request rejected: challenge MAC invalid"
        );
        return auth_error_response(
            StatusCode::UNAUTHORIZED,
            "token challenge authentication failed",
            "invalid_token_challenge",
        );
    }

    let signature = match parse_signature_base64(&req.signature) {
        Ok(signature) => signature,
        Err(message) => {
            return auth_error_response(
                StatusCode::BAD_REQUEST,
                message,
                "invalid_signature_encoding",
            );
        }
    };
    let identity = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(identity) => identity,
        Err(_) => {
            return auth_error_response(
                StatusCode::BAD_REQUEST,
                "invalid Ed25519 public key",
                "invalid_public_key",
            );
        }
    };
    if identity.verify(challenge.as_bytes(), &signature).is_err() {
        debug!(
            pubkey = &canonical_pubkey[..8],
            "[AUTH_V2] Token request rejected: signature verification failed"
        );
        return auth_error_response(
            StatusCode::UNAUTHORIZED,
            "invalid signature",
            "invalid_signature",
        );
    }

    let jwt_expires_at = match checked_jwt_expiry(&state, now) {
        Ok(expires_at) => expires_at,
        Err(response) => return response,
    };
    if let Err(response) = claim_token_challenge(
        token_challenge_v2_fingerprint(&challenge),
        req.expires_at,
        now,
        &canonical_pubkey,
        "v2",
    ) {
        return response;
    }

    issue_jwt_response(
        &canonical_pubkey,
        now,
        jwt_expires_at,
        &state.jwt_secret,
        "v2",
    )
}

/// `POST /api/auth/token`
///
/// Issues a JWT to a client that proves ownership of an Ed25519 keypair.
///
/// Challenge format: UTF-8 bytes of `"{pubkey_hex}:{timestamp}"`
/// The client signs this string with their Ed25519 private key.
///
/// ## Error Responses
/// - 400: malformed pubkey or signature
/// - 401: invalid signature or expired timestamp
/// - 500: JWT issuance failed (internal error)
pub async fn issue_token(
    State(state): State<AuthState>,
    Json(req): Json<TokenRequest>,
) -> impl IntoResponse {
    // Validate pubkey format.
    let pubkey_bytes = match parse_pubkey_hex(&req.pubkey) {
        Ok(b) => b,
        Err(msg) => return legacy_auth_error_response(StatusCode::BAD_REQUEST, msg),
    };

    // Validate timestamp (replay protection).
    let now = now_secs();
    let drift = if now > req.timestamp {
        now - req.timestamp
    } else {
        req.timestamp - now
    };

    if drift > TIMESTAMP_TOLERANCE_SECS {
        debug!(
            drift_secs = drift,
            pubkey = &req.pubkey[..8],
            "[AUTH] Token request rejected: timestamp drift too large"
        );
        return legacy_auth_error_response(
            StatusCode::UNAUTHORIZED,
            "timestamp expired or clock drift too large",
        );
    }

    // Parse Ed25519 signature.
    let sig_bytes = match parse_signature_base64(&req.signature) {
        Ok(b) => b,
        Err(msg) => return legacy_auth_error_response(StatusCode::BAD_REQUEST, msg),
    };

    // Verify Ed25519 signature.
    // Challenge = "{pubkey_hex}:{timestamp}" as UTF-8 bytes.
    let challenge = format!("{}:{}", req.pubkey, req.timestamp);
    let identity_pubkey = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => {
            return legacy_auth_error_response(
                StatusCode::BAD_REQUEST,
                "invalid Ed25519 public key",
            );
        }
    };

    if identity_pubkey
        .verify(challenge.as_bytes(), &sig_bytes)
        .is_err()
    {
        debug!(
            pubkey = &req.pubkey[..8],
            "[AUTH] Token request rejected: signature verification failed"
        );
        return legacy_auth_error_response(StatusCode::UNAUTHORIZED, "invalid signature");
    }

    let jwt_expires_at = match checked_jwt_expiry(&state, now) {
        Ok(expires_at) => expires_at,
        Err(response) => return response,
    };
    let replay_expires_at = req.timestamp.saturating_add(TIMESTAMP_TOLERANCE_SECS);
    if let Err(response) = claim_token_challenge(
        token_challenge_fingerprint(&pubkey_bytes, req.timestamp),
        replay_expires_at,
        now,
        &req.pubkey,
        "v1",
    ) {
        return response;
    }

    issue_jwt_response(&req.pubkey, now, jwt_expires_at, &state.jwt_secret, "v1")
}

fn checked_jwt_expiry(state: &AuthState, now: u64) -> Result<u64, Response> {
    if state.jwt_secret.len() < 32 {
        tracing::error!("[AUTH] Refusing JWT issuance with invalid signing secret");
        return Err(auth_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "authentication service is not configured",
            "auth_service_misconfigured",
        ));
    }
    now.checked_add(state.token_ttl_secs).ok_or_else(|| {
        tracing::error!(
            ttl_secs = state.token_ttl_secs,
            "[AUTH] JWT expiry overflow"
        );
        auth_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "authentication token lifetime is invalid",
            "auth_token_lifetime_invalid",
        )
    })
}

fn claim_token_challenge(
    fingerprint: [u8; 32],
    challenge_expires_at: u64,
    now: u64,
    canonical_pubkey: &str,
    protocol: &'static str,
) -> Result<(), Response> {
    match token_replay_guard().claim(fingerprint, challenge_expires_at, now) {
        TokenReplayDecision::Accepted => Ok(()),
        TokenReplayDecision::Duplicate => {
            warn!(
                pubkey = &canonical_pubkey[..8],
                protocol, "[AUTH] Token request rejected: verified challenge replayed"
            );
            Err(auth_error_response(
                StatusCode::CONFLICT,
                "token challenge has already been used",
                "token_challenge_replayed",
            ))
        }
        TokenReplayDecision::Saturated => {
            tracing::error!(
                capacity = TOKEN_REPLAY_CACHE_MAX_ENTRIES,
                protocol,
                "[AUTH] Token request rejected: replay guard saturated"
            );
            Err(auth_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "token service is temporarily busy",
                "token_replay_guard_saturated",
            ))
        }
    }
}

fn issue_jwt_response(
    pubkey: &str,
    issued_at: u64,
    expires_at: u64,
    jwt_secret: &str,
    protocol: &'static str,
) -> Response {
    match issue_jwt(pubkey, issued_at, expires_at, jwt_secret) {
        Ok(token) => {
            info!(
                pubkey = &pubkey[..8],
                expires_at, protocol, "[AUTH] JWT issued"
            );
            auth_response_no_store(
                (StatusCode::OK, Json(TokenResponse { token, expires_at })).into_response(),
            )
        }
        Err(error) => {
            // JWT signing failure is an internal error — do not expose details.
            tracing::error!(error = %error, protocol, "[AUTH] JWT signing failed");
            auth_response_no_store(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "internal error" })),
                )
                    .into_response(),
            )
        }
    }
}

// ============================================
// JWT Utility Functions
// ============================================

/// Sign and encode a JWT with HS256.
///
/// # Arguments
/// - `pubkey_hex`: 64-char hex Ed25519 public key (becomes `sub`)
/// - `iat`: issued-at Unix timestamp
/// - `exp`: expiry Unix timestamp
/// - `secret`: HS256 signing secret (minimum 32 bytes)
pub fn issue_jwt(
    pubkey_hex: &str,
    iat: u64,
    exp: u64,
    secret: &str,
) -> Result<String, jsonwebtoken::errors::Error> {
    let claims = Claims {
        sub: pubkey_hex.to_string(),
        iat,
        exp,
        iss: JWT_ISSUER.to_string(),
    };
    let header = Header::new(Algorithm::HS256);
    let key = EncodingKey::from_secret(secret.as_bytes());
    encode(&header, &claims, &key)
}

/// Verify a JWT and extract its claims.
///
/// Validates HS256 signature, `exp` (not expired), and `iss` ("memchain").
///
/// # Errors
/// Returns `jsonwebtoken::errors::Error` for any validation failure.
pub fn verify_jwt(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let key = DecodingKey::from_secret(secret.as_bytes());
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_issuer(&[JWT_ISSUER]);
    // exp validation is enabled by default. leeway=0: clocks must be in sync.
    validation.leeway = 0;
    let data = decode::<Claims>(token, &key, &validation)?;
    Ok(data.claims)
}

// ============================================
// JWT Secret Management
// ============================================

/// Ensure an MPI API secret exists in memory and in `[memchain]` on disk.
///
/// A valid legacy `[auth].api_secret` is migrated without rotation. When no
/// reusable value exists, a new 64-character secret is generated. The returned
/// value must be injected into the runtime configuration before listeners open.
pub fn ensure_api_secret(
    current_secret: Option<&str>,
    config_path: Option<&Path>,
) -> Result<String, String> {
    ensure_memchain_secret(current_secret, config_path, "api_secret", 16)
}

/// Ensure a JWT secret exists, generating and persisting one if needed.
///
/// If `current_secret` is non-empty and >= 32 chars, it is returned as-is.
/// Otherwise a 64-char random alphanumeric secret is generated and written
/// to `config_path` via `write_secret_to_config`.
///
/// # Errors
/// Returns Err if the secret is too short, or if the config file cannot
/// be written. Server startup should fail loudly on Err.
pub fn ensure_jwt_secret(
    current_secret: Option<&str>,
    config_path: Option<&Path>,
) -> Result<String, String> {
    ensure_memchain_secret(current_secret, config_path, "jwt_secret", 32)
}

/// Resolve one secret through existing config, legacy migration, or generation.
fn ensure_memchain_secret(
    current_secret: Option<&str>,
    config_path: Option<&Path>,
    key: &str,
    minimum_len: usize,
) -> Result<String, String> {
    if let Some(secret) = current_secret.filter(|secret| !secret.is_empty()) {
        if secret.len() < minimum_len {
            return Err(format!(
                "{} is too short ({} chars, minimum {})",
                key,
                secret.len(),
                minimum_len
            ));
        }
        return Ok(secret.to_string());
    }

    if let Some(path) = config_path {
        let legacy = read_secret_from_section(path, LEGACY_AUTH_CONFIG_SECTION, key)
            .map_err(|error| format!("Failed to inspect {}: {}", path.display(), error))?;
        if let Some(secret) = legacy.filter(|secret| secret.len() >= minimum_len) {
            write_secret_to_config(path, key, &secret).map_err(|error| {
                format!("Failed to migrate {} in {}: {}", key, path.display(), error)
            })?;
            info!(
                path = %path.display(),
                key,
                "[AUTH] Migrated legacy secret into the canonical memchain section"
            );
            return Ok(secret);
        }
    }

    let secret = generate_secret();
    info!(key, "[AUTH] Generated new secret (64 chars alphanumeric)");

    if let Some(path) = config_path {
        write_secret_to_config(path, key, &secret).map_err(|error| {
            format!("Failed to persist {} to {}: {}", key, path.display(), error)
        })?;
        info!(path = %path.display(), key, "[AUTH] Secret written to config file");
    }

    Ok(secret)
}

/// Generate a 64-char random alphanumeric secret.
pub fn generate_secret() -> String {
    // Use thread_rng from whichever rand version is actually linked,
    // accessed via the Rng trait. We avoid gen_range(Range) because its
    // signature changed between rand 0.7 (2 args) and 0.8+ (1 arg).
    // Instead, generate a raw u64 and take modulo — works on all versions.
    use rand::RngCore;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();
    (0..64)
        .map(|_| {
            let idx = (rng.next_u64() as usize) % CHARSET.len();
            CHARSET[idx] as char
        })
        .collect()
}

// ============================================
// Parsing Helpers
// ============================================

/// Parse a 64-char hex pubkey string into a 32-byte array.
pub fn parse_pubkey_hex(hex_str: &str) -> Result<[u8; 32], &'static str> {
    if hex_str.len() != 64 {
        return Err("pubkey must be exactly 64 hex characters (32 bytes)");
    }
    let bytes = hex::decode(hex_str).map_err(|_| "pubkey contains invalid hex characters")?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Parse a base64-encoded Ed25519 signature into a 64-byte array.
pub fn parse_signature_base64(b64: &str) -> Result<[u8; 64], &'static str> {
    let bytes = BASE64
        .decode(b64)
        .map_err(|_| "signature is not valid base64")?;
    if bytes.len() != 64 {
        return Err("signature must be exactly 64 bytes");
    }
    let mut arr = [0u8; 64];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

// ============================================
// Config File Write Helpers
// ============================================

/// Write a `key = "value"` line into `[memchain]` in an existing TOML file.
///
/// The writer is deliberately section-aware. A same-named key in `[auth]` or
/// another section is never replaced. This preserves backward compatibility
/// with comments and formatting without reserializing the complete document.
///
/// The complete TOML document is parsed before and after editing, so malformed
/// input or an invalid generated document fails before disk replacement.
///
/// # Comment-line guard
/// Lines whose first non-whitespace character is `#` are never replaced.
/// This prevents false matches against commented-out example values like:
///   `# jwt_secret = "example_do_not_use"`
///
/// `pub` since v1.0.1; retained as a section-safe compatibility surface.
pub fn write_secret_to_config(path: &Path, key: &str, value: &str) -> std::io::Result<()> {
    write_secret_to_config_section(path, MEMCHAIN_CONFIG_SECTION, key, value)
}

fn write_secret_to_config_section(
    path: &Path,
    section: &str,
    key: &str,
    value: &str,
) -> std::io::Result<()> {
    validate_toml_identifier(section, "section")?;
    validate_toml_identifier(key, "key")?;

    let content = std::fs::read_to_string(path)?;
    parse_toml_document(&content)?;
    let encoded_value = toml::Value::String(value.to_string()).to_string();
    let new_line = format!("{} = {}", key, encoded_value);

    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let section_header = format!("[{}]", section);
    let descendant_prefix = format!("[{}.", section);
    let section_start = lines.iter().position(|line| line.trim() == section_header);

    if let Some(section_start) = section_start {
        let section_end = lines
            .iter()
            .enumerate()
            .skip(section_start + 1)
            .find_map(|(index, line)| line.trim().starts_with('[').then_some(index))
            .unwrap_or(lines.len());
        let key_line = lines
            .iter()
            .enumerate()
            .take(section_end)
            .skip(section_start + 1)
            .find_map(|(index, line)| line_assigns_key(line, key).then_some(index));

        if let Some(key_line) = key_line {
            lines[key_line] = new_line;
        } else {
            lines.insert(section_end, new_line);
        }
    } else if let Some(first_descendant) = lines
        .iter()
        .position(|line| line.trim().starts_with(&descendant_prefix))
    {
        lines.splice(
            first_descendant..first_descendant,
            [section_header, new_line, String::new()],
        );
    } else {
        if lines.last().is_some_and(|line| !line.is_empty()) {
            lines.push(String::new());
        }
        lines.push(section_header);
        lines.push(new_line);
    }

    let mut output = lines.join("\n");
    if content.ends_with('\n') {
        output.push('\n');
    }
    parse_toml_document(&output)?;
    atomic_replace_preserving_permissions(path, output.as_bytes())
}

/// Durably replace a configuration file without exposing a truncated state.
///
/// The temporary file lives beside the destination so the final rename stays
/// on one filesystem. Existing permissions are copied before the rename; the
/// file and parent directory are synced to make the replacement crash-safe on
/// filesystems that honor `fsync` durability guarantees.
fn atomic_replace_preserving_permissions(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = normalized_parent(path);
    let file_name = path.file_name().ok_or_else(|| {
        IoError::new(
            IoErrorKind::InvalidInput,
            "configuration path has no file name",
        )
    })?;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut temporary_name = OsString::from(".");
    temporary_name.push(file_name);
    temporary_name.push(format!(".aeronyx.{}.{}.tmp", std::process::id(), unique));
    let temporary_path = parent.join(temporary_name);
    let permissions = std::fs::metadata(path)?.permissions();

    let result = (|| {
        let mut temporary = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        temporary.set_permissions(permissions)?;
        temporary.write_all(bytes)?;
        temporary.sync_all()?;
        drop(temporary);

        std::fs::rename(&temporary_path, path)?;
        #[cfg(unix)]
        {
            let directory = std::fs::File::open(parent)?;
            directory.sync_all()?;
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

/// Return a directory suitable for sibling temporary files and `fsync`.
///
/// `Path::parent()` returns an empty path for a relative file such as
/// `server.toml`; treating that as the current directory keeps relative
/// command-line configuration paths fully supported.
fn normalized_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn line_assigns_key(line: &str, key: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return false;
    }
    trimmed
        .split_once('=')
        .is_some_and(|(candidate, _)| candidate.trim() == key)
}

fn validate_toml_identifier(value: &str, label: &str) -> std::io::Result<()> {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Ok(());
    }
    Err(IoError::new(
        IoErrorKind::InvalidInput,
        format!("invalid TOML {} identifier", label),
    ))
}

fn parse_toml_document(content: &str) -> std::io::Result<toml::Value> {
    toml::from_str(content)
        .map_err(|error| IoError::new(IoErrorKind::InvalidData, error.to_string()))
}

fn read_secret_from_section(
    path: &Path,
    section: &str,
    key: &str,
) -> std::io::Result<Option<String>> {
    let content = std::fs::read_to_string(path)?;
    let document = parse_toml_document(&content)?;
    Ok(document
        .get(section)
        .and_then(toml::Value::as_table)
        .and_then(|table| table.get(key))
        .and_then(toml::Value::as_str)
        .map(str::to_string))
}

/// Public alias for `write_secret_to_config`.
///
/// Retained for source compatibility with older internal call sites. New code
/// should normally use `ensure_api_secret()` or `ensure_jwt_secret()` so the
/// persisted value is also returned for immediate runtime injection.
#[inline(always)]
pub fn write_secret_to_config_pub(path: &Path, key: &str, value: &str) -> std::io::Result<()> {
    write_secret_to_config(path, key, value)
}

// ============================================
// Private Helpers
// ============================================

/// Current Unix timestamp in seconds.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::keys::IdentityKeyPair;
    use tempfile::TempDir;

    const TEST_SECRET: &str = "test-secret-that-is-at-least-32-chars-long-for-safety";

    fn make_test_secret() -> String {
        TEST_SECRET.to_string()
    }

    fn make_auth_state() -> AuthState {
        AuthState {
            jwt_secret: make_test_secret(),
            token_ttl_secs: 3_600,
        }
    }

    async fn request_v2_challenge(
        identity: &IdentityKeyPair,
        state: &AuthState,
    ) -> TokenChallengeV2Response {
        let response = issue_token_challenge_v2(
            State(state.clone()),
            Json(TokenChallengeV2Request {
                pubkey: hex::encode(identity.public_key_bytes()),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CACHE_CONTROL).unwrap(),
            HeaderValue::from_static("no-store")
        );
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn signed_v2_request(
        identity: &IdentityKeyPair,
        challenge: &TokenChallengeV2Response,
    ) -> TokenV2Request {
        TokenV2Request {
            pubkey: challenge.pubkey.clone(),
            nonce: challenge.nonce.clone(),
            issued_at: challenge.issued_at,
            expires_at: challenge.expires_at,
            challenge_mac: challenge.challenge_mac.clone(),
            signature: BASE64.encode(identity.sign(challenge.challenge.as_bytes())),
        }
    }

    // -- JWT round-trip ---------------------------------------------------

    #[test]
    fn test_issue_and_verify_jwt() {
        let pubkey_hex = hex::encode([0xAAu8; 32]);
        let now = now_secs();
        let exp = now + 86400;

        let token = issue_jwt(&pubkey_hex, now, exp, &make_test_secret()).unwrap();
        assert!(!token.is_empty());

        let claims = verify_jwt(&token, &make_test_secret()).unwrap();
        assert_eq!(claims.sub, pubkey_hex);
        assert_eq!(claims.iss, JWT_ISSUER);
        assert_eq!(claims.exp, exp);
    }

    #[test]
    fn test_verify_wrong_secret_fails() {
        let pubkey_hex = hex::encode([0xAAu8; 32]);
        let now = now_secs();
        let token = issue_jwt(&pubkey_hex, now, now + 3600, &make_test_secret()).unwrap();
        let result = verify_jwt(&token, "wrong-secret-at-least-32-chars-xxxxxxxxxxxx");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_expired_token_fails() {
        let pubkey_hex = hex::encode([0xAAu8; 32]);
        let past = now_secs() - 7200;
        let token = issue_jwt(&pubkey_hex, past - 3600, past, &make_test_secret()).unwrap();
        assert!(verify_jwt(&token, &make_test_secret()).is_err());
    }

    #[test]
    fn test_verify_wrong_issuer_fails() {
        let now = now_secs();
        #[derive(Serialize)]
        struct BadClaims {
            sub: String,
            iat: u64,
            exp: u64,
            iss: String,
        }
        let bad = BadClaims {
            sub: hex::encode([0xAAu8; 32]),
            iat: now,
            exp: now + 3600,
            iss: "evil".to_string(),
        };
        let key = jsonwebtoken::EncodingKey::from_secret(TEST_SECRET.as_bytes());
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &bad,
            &key,
        )
        .unwrap();
        assert!(verify_jwt(&token, &make_test_secret()).is_err());
    }

    // -- parse_pubkey_hex -------------------------------------------------

    #[test]
    fn test_parse_pubkey_hex_valid() {
        let hex = hex::encode([0xABu8; 32]);
        assert_eq!(parse_pubkey_hex(&hex).unwrap(), [0xABu8; 32]);
    }

    #[test]
    fn test_parse_pubkey_hex_wrong_length() {
        assert!(parse_pubkey_hex("aabb").is_err());
        assert!(parse_pubkey_hex(&"a".repeat(128)).is_err());
    }

    #[test]
    fn test_parse_pubkey_hex_invalid_chars() {
        let mut hex = hex::encode([0u8; 32]);
        hex.replace_range(0..2, "ZZ");
        assert!(parse_pubkey_hex(&hex).is_err());
    }

    // -- parse_signature_base64 -------------------------------------------

    #[test]
    fn test_parse_signature_base64_valid() {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let sig = [0x42u8; 64];
        let b64 = BASE64.encode(sig);
        assert_eq!(parse_signature_base64(&b64).unwrap(), sig);
    }

    #[test]
    fn test_parse_signature_base64_wrong_size() {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let short = BASE64.encode([0u8; 32]);
        assert!(parse_signature_base64(&short).is_err());
    }

    // -- ensure_jwt_secret ------------------------------------------------

    #[test]
    fn test_ensure_jwt_secret_uses_existing() {
        let existing = "a".repeat(32);
        assert_eq!(ensure_jwt_secret(Some(&existing), None).unwrap(), existing);
    }

    #[test]
    fn test_ensure_jwt_secret_too_short_fails() {
        assert!(ensure_jwt_secret(Some("too-short"), None).is_err());
    }

    #[test]
    fn test_ensure_jwt_secret_generates_when_empty() {
        let s = ensure_jwt_secret(Some(""), None).unwrap();
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c.is_alphanumeric()));
    }

    #[test]
    fn test_ensure_jwt_secret_generates_when_none() {
        let s = ensure_jwt_secret(None, None).unwrap();
        assert_eq!(s.len(), 64);
    }

    #[test]
    fn test_ensure_jwt_secret_writes_to_config() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[memchain]\nmode = \"saas\"\n\n[auth]\ntoken_ttl_secs = 86400\n",
        )
        .unwrap();

        let secret = ensure_jwt_secret(None, Some(&config_path)).unwrap();
        assert_eq!(secret.len(), 64);

        let content = std::fs::read_to_string(&config_path).unwrap();
        assert!(content.contains(&secret));
        assert!(content.contains("jwt_secret ="));
        let parsed: toml::Value = toml::from_str(&content).unwrap();
        assert_eq!(
            parsed["memchain"]["jwt_secret"].as_str(),
            Some(secret.as_str())
        );
        assert!(parsed["auth"].get("jwt_secret").is_none());
    }

    #[test]
    fn test_ensure_jwt_secret_uses_memchain_section() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[memchain]\nmode = \"saas\"\n").unwrap();

        let secret = ensure_jwt_secret(None, Some(&config_path)).unwrap();
        let content = std::fs::read_to_string(&config_path).unwrap();
        assert!(!content.contains("[auth]"));
        assert!(content.contains(&secret));
    }

    #[test]
    fn test_ensure_api_secret_migrates_legacy_auth_value_without_rotation() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        let legacy = "legacy-api-secret-that-clients-already-use";
        std::fs::write(
            &config_path,
            format!(
                "[memchain]\nmode = \"local\"\n\n[auth]\napi_secret = \"{}\"\n",
                legacy
            ),
        )
        .unwrap();

        let resolved = ensure_api_secret(None, Some(&config_path)).unwrap();
        assert_eq!(resolved, legacy);

        let content = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value = toml::from_str(&content).unwrap();
        assert_eq!(parsed["memchain"]["api_secret"].as_str(), Some(legacy));
        assert_eq!(parsed["auth"]["api_secret"].as_str(), Some(legacy));
    }

    #[test]
    fn test_ensure_api_secret_generates_and_persists_first_start_value() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[memchain]\nmode = \"local\"\n").unwrap();

        let resolved = ensure_api_secret(None, Some(&config_path)).unwrap();
        assert_eq!(resolved.len(), 64);

        let content = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value = toml::from_str(&content).unwrap();
        assert_eq!(
            parsed["memchain"]["api_secret"].as_str(),
            Some(resolved.as_str())
        );
    }

    // -- write_secret_to_config comment-line guard ------------------------

    #[test]
    fn test_write_secret_skips_comment_lines() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[memchain]\n# jwt_secret = \"example_do_not_use\"\njwt_secret = \"\"\n",
        )
        .unwrap();

        write_secret_to_config(&config_path, "jwt_secret", "real_value").unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        // Comment line must be preserved unchanged.
        assert!(content.contains("# jwt_secret = \"example_do_not_use\""));
        // Real key line must be replaced.
        assert!(content.contains("jwt_secret = \"real_value\""));
    }

    #[test]
    fn test_write_secret_never_replaces_same_key_in_another_section() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[memchain]\nmode = \"local\"\n\n[auth]\napi_secret = \"legacy_value\"\n",
        )
        .unwrap();

        write_secret_to_config(&config_path, "api_secret", "canonical_value").unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value = toml::from_str(&content).unwrap();
        assert_eq!(
            parsed["memchain"]["api_secret"].as_str(),
            Some("canonical_value")
        );
        assert_eq!(parsed["auth"]["api_secret"].as_str(), Some("legacy_value"));
    }

    #[test]
    fn test_write_secret_inserts_parent_before_existing_child_section() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[server]\nname = \"node\"\n\n[memchain.saas]\ndata_root = \"data\"\n",
        )
        .unwrap();

        write_secret_to_config(&config_path, "api_secret", "canonical_value").unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        let parent = content.find("[memchain]\n").unwrap();
        let child = content.find("[memchain.saas]").unwrap();
        assert!(parent < child);
        let parsed: toml::Value = toml::from_str(&content).unwrap();
        assert_eq!(
            parsed["memchain"]["api_secret"].as_str(),
            Some("canonical_value")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_write_secret_atomic_replace_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[memchain]\nmode = \"local\"\n").unwrap();
        std::fs::set_permissions(&config_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        write_secret_to_config(&config_path, "api_secret", "canonical_value").unwrap();

        let mode = std::fs::metadata(&config_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
        let temporary_files = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".aeronyx."))
            .count();
        assert_eq!(temporary_files, 0);
    }

    #[test]
    fn test_atomic_replace_normalizes_relative_parent() {
        assert_eq!(normalized_parent(Path::new("server.toml")), Path::new("."));
        assert_eq!(
            normalized_parent(Path::new("config/server.toml")),
            Path::new("config")
        );
    }

    // [PORTABLE-PATH-TEST 2026-07-23 by Codex] Darwin rejects this byte sequence
    // before the atomic replacement code runs, so only test it where it is a
    // representable filesystem path.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    #[test]
    fn test_atomic_replace_accepts_non_utf8_file_name() {
        use std::os::unix::ffi::OsStringExt;

        let dir = TempDir::new().unwrap();
        let mut name = b"server-".to_vec();
        name.push(0xFF);
        name.extend_from_slice(b".toml");
        let config_path = dir.path().join(OsString::from_vec(name));
        std::fs::write(&config_path, "[memchain]\nmode = \"local\"\n").unwrap();

        write_secret_to_config(&config_path, "api_secret", "canonical_value").unwrap();

        let parsed: toml::Value = toml::from_str(
            &std::fs::read_to_string(&config_path).expect("updated config remains readable"),
        )
        .unwrap();
        assert_eq!(
            parsed["memchain"]["api_secret"].as_str(),
            Some("canonical_value")
        );
    }

    // -- write_secret_to_config_pub alias ---------------------------------

    #[test]
    fn test_write_secret_to_config_pub_alias() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[memchain]\napi_secret = \"\"\n").unwrap();

        write_secret_to_config_pub(&config_path, "api_secret", "generated_secret").unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        assert!(content.contains("api_secret = \"generated_secret\""));
    }

    // -- generate_secret --------------------------------------------------

    #[test]
    fn test_generate_secret_length_and_charset() {
        for _ in 0..10 {
            let s = generate_secret();
            assert_eq!(s.len(), 64);
            assert!(s.chars().all(|c| c.is_alphanumeric()));
        }
    }

    #[test]
    fn test_generate_secret_is_random() {
        assert_ne!(generate_secret(), generate_secret());
    }

    #[tokio::test]
    async fn test_issue_token_rejects_replayed_verified_challenge() {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

        let identity = IdentityKeyPair::generate();
        let pubkey = hex::encode(identity.public_key_bytes());
        let timestamp = now_secs();
        let challenge = format!("{}:{}", pubkey, timestamp);
        let signature = BASE64.encode(identity.sign(challenge.as_bytes()));
        let state = AuthState {
            jwt_secret: make_test_secret(),
            token_ttl_secs: 3_600,
        };
        let request = || TokenRequest {
            pubkey: pubkey.clone(),
            timestamp,
            signature: signature.clone(),
        };

        let first = issue_token(State(state.clone()), Json(request()))
            .await
            .into_response();
        assert_eq!(first.status(), StatusCode::OK);

        let replay = issue_token(State(state), Json(request()))
            .await
            .into_response();
        assert_eq!(replay.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_v1_error_preserves_contract_and_disables_caching() {
        let identity = IdentityKeyPair::generate();
        let attacker = IdentityKeyPair::generate();
        let pubkey = hex::encode(identity.public_key_bytes());
        let timestamp = now_secs();
        let challenge = format!("{}:{}", pubkey, timestamp);
        let response = issue_token(
            State(make_auth_state()),
            Json(TokenRequest {
                pubkey,
                timestamp,
                signature: BASE64.encode(attacker.sign(challenge.as_bytes())),
            }),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(CACHE_CONTROL).unwrap(),
            HeaderValue::from_static("no-store")
        );
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({ "error": "invalid signature" })
        );
    }

    #[test]
    fn test_v2_canonical_challenge_format_is_stable() {
        assert_eq!(
            token_challenge_v2_message("aabb", "bm9uY2U", 100, 220),
            "AeroNyx Auth Token v2\npubkey=aabb\nnonce=bm9uY2U\nissued_at=100\nexpires_at=220"
        );
    }

    #[test]
    fn test_v2_challenge_mac_rejects_tampering() {
        let challenge = token_challenge_v2_message("aabb", "bm9uY2U", 100, 220);
        let mac = token_challenge_v2_mac(TEST_SECRET, &challenge).unwrap();
        assert!(verify_token_challenge_v2_mac(TEST_SECRET, &challenge, &mac));
        assert!(!verify_token_challenge_v2_mac(
            TEST_SECRET,
            &format!("{}x", challenge),
            &mac
        ));
    }

    #[tokio::test]
    async fn test_v2_token_round_trip_and_replay_rejection() {
        let identity = IdentityKeyPair::generate();
        let state = make_auth_state();
        let challenge = request_v2_challenge(&identity, &state).await;
        let first = issue_token_v2(
            State(state.clone()),
            Json(signed_v2_request(&identity, &challenge)),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(
            first.headers().get(CACHE_CONTROL).unwrap(),
            HeaderValue::from_static("no-store")
        );

        let replay =
            issue_token_v2(State(state), Json(signed_v2_request(&identity, &challenge))).await;
        assert_eq!(replay.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_auth_router_wires_v2_and_enforces_body_limit() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let identity = IdentityKeyPair::generate();
        let router = build_auth_router(make_auth_state());
        let challenge_body = serde_json::to_vec(&TokenChallengeV2Request {
            pubkey: hex::encode(identity.public_key_bytes()),
        })
        .unwrap();
        let challenge_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/challenge")
                    .header("content-type", "application/json")
                    .body(Body::from(challenge_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(challenge_response.status(), StatusCode::OK);
        let challenge_bytes = axum::body::to_bytes(challenge_response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let challenge: TokenChallengeV2Response = serde_json::from_slice(&challenge_bytes).unwrap();

        let token_body = serde_json::to_vec(&signed_v2_request(&identity, &challenge)).unwrap();
        let token_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/token/v2")
                    .header("content-type", "application/json")
                    .body(Body::from(token_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(token_response.status(), StatusCode::OK);

        let oversized = serde_json::json!({ "pubkey": "a".repeat(17 * 1024) }).to_string();
        let oversized_response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/challenge")
                    .header("content-type", "application/json")
                    .body(Body::from(oversized))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversized_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_v2_wrong_signature_does_not_consume_challenge() {
        let identity = IdentityKeyPair::generate();
        let attacker = IdentityKeyPair::generate();
        let state = make_auth_state();
        let challenge = request_v2_challenge(&identity, &state).await;

        let invalid = issue_token_v2(
            State(state.clone()),
            Json(signed_v2_request(&attacker, &challenge)),
        )
        .await;
        assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);

        let valid =
            issue_token_v2(State(state), Json(signed_v2_request(&identity, &challenge))).await;
        assert_eq!(valid.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_v2_tampered_nonce_is_rejected_before_replay_claim() {
        let identity = IdentityKeyPair::generate();
        let state = make_auth_state();
        let challenge = request_v2_challenge(&identity, &state).await;
        let mut tampered_nonce = parse_base64url_32(&challenge.nonce, "nonce").unwrap();
        tampered_nonce[0] ^= 0x80;
        let tampered_nonce_b64 = BASE64URL.encode(tampered_nonce);
        let tampered_message = token_challenge_v2_message(
            &challenge.pubkey,
            &tampered_nonce_b64,
            challenge.issued_at,
            challenge.expires_at,
        );
        let tampered = TokenV2Request {
            pubkey: challenge.pubkey.clone(),
            nonce: tampered_nonce_b64,
            issued_at: challenge.issued_at,
            expires_at: challenge.expires_at,
            challenge_mac: challenge.challenge_mac.clone(),
            signature: BASE64.encode(identity.sign(tampered_message.as_bytes())),
        };

        let invalid = issue_token_v2(State(state.clone()), Json(tampered)).await;
        assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);

        let valid =
            issue_token_v2(State(state), Json(signed_v2_request(&identity, &challenge))).await;
        assert_eq!(valid.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_v2_expired_challenge_is_rejected() {
        let identity = IdentityKeyPair::generate();
        let state = make_auth_state();
        let now = now_secs();
        let issued_at = now - TOKEN_CHALLENGE_V2_TTL_SECS - 1;
        let expires_at = issued_at + TOKEN_CHALLENGE_V2_TTL_SECS;
        let nonce = [0x42; TOKEN_CHALLENGE_V2_NONCE_BYTES];
        let nonce_b64 = BASE64URL.encode(nonce);
        let pubkey = hex::encode(identity.public_key_bytes());
        let challenge = token_challenge_v2_message(&pubkey, &nonce_b64, issued_at, expires_at);
        let request = TokenV2Request {
            pubkey,
            nonce: nonce_b64,
            issued_at,
            expires_at,
            challenge_mac: BASE64URL
                .encode(token_challenge_v2_mac(&state.jwt_secret, &challenge).unwrap()),
            signature: BASE64.encode(identity.sign(challenge.as_bytes())),
        };

        let response = issue_token_v2(State(state), Json(request)).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_checked_jwt_expiry_rejects_overflow() {
        let state = AuthState {
            jwt_secret: make_test_secret(),
            token_ttl_secs: u64::MAX,
        };
        assert!(checked_jwt_expiry(&state, 1).is_err());
    }

    #[test]
    fn test_token_replay_guard_rejects_duplicate_challenge() {
        let guard = TokenReplayGuard::new(8);
        let fingerprint = token_challenge_fingerprint(&[0x11; 32], 1_000);

        assert_eq!(
            guard.claim(fingerprint, 1_060, 1_000),
            TokenReplayDecision::Accepted
        );
        assert_eq!(
            guard.claim(fingerprint, 1_060, 1_001),
            TokenReplayDecision::Duplicate
        );
    }

    #[test]
    fn test_token_replay_guard_expires_old_claims() {
        let guard = TokenReplayGuard::new(1);
        let first = token_challenge_fingerprint(&[0x11; 32], 1_000);
        let second = token_challenge_fingerprint(&[0x22; 32], 1_100);

        assert_eq!(
            guard.claim(first, 1_060, 1_000),
            TokenReplayDecision::Accepted
        );
        assert_eq!(
            guard.claim(second, 1_160, 1_061),
            TokenReplayDecision::Accepted
        );
    }

    #[test]
    fn test_token_replay_guard_fails_closed_at_capacity() {
        let guard = TokenReplayGuard::new(1);
        let first = token_challenge_fingerprint(&[0x11; 32], 1_000);
        let second = token_challenge_fingerprint(&[0x22; 32], 1_001);

        assert_eq!(
            guard.claim(first, 1_060, 1_000),
            TokenReplayDecision::Accepted
        );
        assert_eq!(
            guard.claim(second, 1_061, 1_001),
            TokenReplayDecision::Saturated
        );
    }
}
