// ============================================
// File: crates/aeronyx-server/src/api/auth.rs
// ============================================
//! # Auth — JWT Token Issuance for SaaS Mode
//!
//! ## Modification History
//! v1.0.0-MultiTenant - Initial implementation (Task 2)
//! v1.0.1-Fix         - write_secret_to_config promoted to `pub`.
//!                      Added `write_secret_to_config_pub` as an explicit
//!                      public alias used by server.rs for api_secret
//!                      persistence (ensure_api_secret_on_disk).
//!                      Comment-line guard added to write_secret_to_config
//!                      to prevent false replacement of commented-out keys.
//!
//! ## Main Functionality
//! - `POST /api/auth/token`: Verify Ed25519 challenge signature, issue JWT
//! - `issue_jwt()`: Sign a JWT with HS256 using the configured secret
//! - `verify_jwt()`: Validate JWT signature + expiry, extract Claims
//! - `ensure_jwt_secret()`: Auto-generate and persist the JWT secret
//! - `write_secret_to_config_pub()`: Public alias for config file writes
//!
//! ## Authentication Flow
//! 1. Client signs `"{pubkey_hex}:{unix_timestamp}"` with their Ed25519 private key
//! 2. Server verifies signature using the pubkey in the request body
//! 3. Timestamp must be within +/-60 seconds of server time
//! 4. On success, server issues a JWT where `sub = pubkey_hex`
//! 5. All subsequent API requests include `Authorization: Bearer <jwt>`
//! 6. `unified_auth_middleware` (in mpi.rs) verifies JWT + extracts owner
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
//! - Timestamp window is +/-60 seconds (not 300s like the remote storage path)
//!   because token issuance is more sensitive than request signing.
//!
//! ## Dependencies
//! - `jsonwebtoken` crate for HS256 JWT sign/verify
//! - `ed25519-dalek` via `aeronyx_core::crypto` for signature verification
//! - Used by `mpi.rs` router (SaaS mode only, excluded from local mode)
//! - `ensure_jwt_secret()` called from `server.rs` during SaaS init
//! - `write_secret_to_config_pub()` called from `server.rs` for api_secret
//!
//! ⚠️ Important Notes for Next Developer:
//! - This endpoint is EXCLUDED from `unified_auth_middleware` — it must be
//!   registered before the middleware layer in `build_mpi_router()`.
//! - `ed25519-dalek` v2.x changed the Verifier trait API. This code uses
//!   `IdentityPublicKey::verify()` from aeronyx-core, which abstracts the
//!   version difference — do NOT call dalek directly.
//! - JWT secret minimum length: 32 bytes. Auto-generated secrets are 64 chars.
//! - `ensure_jwt_secret()` returns Err if the config file cannot be written.
//!   Server startup should fail loudly in this case.
//! - `write_secret_to_config` is `pub` (v1.0.1 fix) so that `server.rs` can
//!   call it for both api_secret and jwt_secret persistence. Use the
//!   `write_secret_to_config_pub` alias at call sites for clarity.
//! - Comment-line guard: lines whose first non-whitespace char is '#' are
//!   never modified, preventing false replacement of commented-out examples.
//!
//! ## Last Modified
//! v1.0.1-Fix - Promoted write_secret_to_config to pub; added _pub alias;
//!              added comment-line guard in write_secret_to_config.
// ============================================

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use aeronyx_core::crypto::keys::IdentityPublicKey;

// ============================================
// Constants
// ============================================

/// Issuer claim embedded in all issued JWTs.
pub const JWT_ISSUER: &str = "memchain";

/// Maximum allowed clock skew for token request timestamps (seconds).
const TIMESTAMP_TOLERANCE_SECS: u64 = 60;

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

// ============================================
// Endpoint Handler
// ============================================

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
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response();
        }
    };

    // Validate timestamp (replay protection).
    let now = now_secs();
    let drift = if now > req.timestamp {
        now - req.timestamp
    } else {
        req.timestamp - now
    };

    if drift > TIMESTAMP_TOLERANCE_SECS {
        warn!(
            drift_secs = drift,
            pubkey = &req.pubkey[..8],
            "[AUTH] Token request rejected: timestamp drift too large"
        );
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "timestamp expired or clock drift too large" })),
        )
            .into_response();
    }

    // Parse Ed25519 signature.
    let sig_bytes = match parse_signature_base64(&req.signature) {
        Ok(b) => b,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response();
        }
    };

    // Verify Ed25519 signature.
    // Challenge = "{pubkey_hex}:{timestamp}" as UTF-8 bytes.
    let challenge = format!("{}:{}", req.pubkey, req.timestamp);
    let identity_pubkey = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid Ed25519 public key" })),
            )
                .into_response();
        }
    };

    if identity_pubkey
        .verify(challenge.as_bytes(), &sig_bytes)
        .is_err()
    {
        warn!(
            pubkey = &req.pubkey[..8],
            "[AUTH] Token request rejected: signature verification failed"
        );
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid signature" })),
        )
            .into_response();
    }

    // Issue JWT.
    let expires_at = now + state.token_ttl_secs;
    match issue_jwt(&req.pubkey, now, expires_at, &state.jwt_secret) {
        Ok(token) => {
            info!(
                pubkey = &req.pubkey[..8],
                expires_at,
                "[AUTH] JWT issued"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "token": token,
                    "expires_at": expires_at,
                })),
            )
                .into_response()
        }
        Err(e) => {
            // JWT signing failure is an internal error — do not expose details.
            tracing::error!(error = %e, "[AUTH] JWT signing failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response()
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
pub fn verify_jwt(
    token: &str,
    secret: &str,
) -> Result<Claims, jsonwebtoken::errors::Error> {
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
    if let Some(s) = current_secret {
        if !s.is_empty() {
            if s.len() < 32 {
                return Err(format!(
                    "jwt_secret is too short ({} chars, minimum 32)",
                    s.len()
                ));
            }
            return Ok(s.to_string());
        }
    }

    let secret = generate_secret();
    info!("[AUTH] Generated new JWT secret (64 chars alphanumeric)");

    if let Some(path) = config_path {
        if let Err(e) = write_secret_to_config(path, "jwt_secret", &secret) {
            return Err(format!(
                "Failed to persist jwt_secret to {}: {}",
                path.display(),
                e
            ));
        }
        info!(path = %path.display(), "[AUTH] JWT secret written to config file");
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
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
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

/// Write a `key = "value"` line into an existing TOML config file.
///
/// Scans the file line by line and replaces the first non-comment line
/// that starts with `key =` or `key=`. If no such line is found, appends
/// the entry under the `[auth]` section (creating the section if absent).
///
/// This is a best-effort text replacement — it does NOT parse/re-serialize
/// the full TOML document. Suitable for the simple flat configs used here.
///
/// # Comment-line guard
/// Lines whose first non-whitespace character is `#` are never replaced.
/// This prevents false matches against commented-out example values like:
///   `# jwt_secret = "example_do_not_use"`
///
/// `pub` since v1.0.1: required by server.rs for api_secret persistence.
pub fn write_secret_to_config(path: &Path, key: &str, value: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let new_line = format!("{} = \"{}\"", key, value);

    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let mut replaced = false;

    for line in lines.iter_mut() {
        let trimmed = line.trim_start();
        // Skip comment lines — never replace `# key = "example"`.
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with(&format!("{} =", key))
            || trimmed.starts_with(&format!("{}=", key))
        {
            *line = new_line.clone();
            replaced = true;
            break;
        }
    }

    if !replaced {
        // Append under [auth] section, or at end of file if section absent.
        let mut in_auth = false;
        let mut insert_pos = lines.len();

        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t == "[auth]" {
                in_auth = true;
                continue;
            }
            if in_auth {
                if t.starts_with('[') {
                    insert_pos = i;
                    break;
                }
                if !t.is_empty() {
                    insert_pos = i + 1;
                }
            }
        }

        if !in_auth {
            lines.push(String::new());
            lines.push("[auth]".to_string());
            lines.push(new_line);
        } else {
            lines.insert(insert_pos, new_line);
        }
    }

    let mut output = lines.join("\n");
    if content.ends_with('\n') {
        output.push('\n');
    }
    std::fs::write(path, output)
}

/// Public alias for `write_secret_to_config`.
///
/// Used by `server.rs::ensure_api_secret_on_disk()` to persist the
/// auto-generated api_secret. The `_pub` suffix makes the cross-module
/// call site self-documenting without changing the underlying function.
#[inline(always)]
pub fn write_secret_to_config_pub(
    path: &Path,
    key: &str,
    value: &str,
) -> std::io::Result<()> {
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
    use tempfile::TempDir;

    const TEST_SECRET: &str = "test-secret-that-is-at-least-32-chars-long-for-safety";

    fn make_test_secret() -> String { TEST_SECRET.to_string() }

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
        struct BadClaims { sub: String, iat: u64, exp: u64, iss: String }
        let bad = BadClaims {
            sub: hex::encode([0xAAu8; 32]),
            iat: now, exp: now + 3600,
            iss: "evil".to_string(),
        };
        let key = jsonwebtoken::EncodingKey::from_secret(TEST_SECRET.as_bytes());
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &bad, &key,
        ).unwrap();
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
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
        let sig = [0x42u8; 64];
        let b64 = BASE64.encode(sig);
        assert_eq!(parse_signature_base64(&b64).unwrap(), sig);
    }

    #[test]
    fn test_parse_signature_base64_wrong_size() {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
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
            "[memchain]\nmode = \"saas\"\n\n[auth]\njwt_secret = \"\"\ntoken_ttl_secs = 86400\n",
        ).unwrap();

        let secret = ensure_jwt_secret(None, Some(&config_path)).unwrap();
        assert_eq!(secret.len(), 64);

        let content = std::fs::read_to_string(&config_path).unwrap();
        assert!(content.contains(&secret));
        assert!(content.contains("jwt_secret ="));
    }

    #[test]
    fn test_ensure_jwt_secret_creates_auth_section_if_missing() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[memchain]\nmode = \"saas\"\n").unwrap();

        let secret = ensure_jwt_secret(None, Some(&config_path)).unwrap();
        let content = std::fs::read_to_string(&config_path).unwrap();
        assert!(content.contains("[auth]"));
        assert!(content.contains(&secret));
    }

    // -- write_secret_to_config comment-line guard ------------------------

    #[test]
    fn test_write_secret_skips_comment_lines() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[auth]\n# jwt_secret = \"example_do_not_use\"\njwt_secret = \"\"\n",
        ).unwrap();

        write_secret_to_config(&config_path, "jwt_secret", "real_value").unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        // Comment line must be preserved unchanged.
        assert!(content.contains("# jwt_secret = \"example_do_not_use\""));
        // Real key line must be replaced.
        assert!(content.contains("jwt_secret = \"real_value\""));
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
}
