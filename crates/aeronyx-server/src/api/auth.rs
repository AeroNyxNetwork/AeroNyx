// ============================================
// File: crates/aeronyx-server/src/api/auth.rs
// ============================================
//! # Auth — JWT Token Issuance for SaaS Mode
//!
//! ## Modification History
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
//! - `POST /api/auth/token`: Verify Ed25519 challenge signature, issue JWT
//! - `issue_jwt()`: Sign a JWT with HS256 using the configured secret
//! - `verify_jwt()`: Validate JWT signature + expiry, extract Claims
//! - `ensure_jwt_secret()`: Auto-generate and persist the JWT secret
//! - `ensure_api_secret()`: Resolve, migrate, or generate the MPI admin secret
//! - `write_secret_to_config_pub()`: Section-safe config persistence alias
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
//! - Generated API and JWT secrets belong to `[memchain]`. Legacy values under
//!   `[auth]` are migrated once so upgrades do not rotate working credentials.
//! - Timestamp window is +/-60 seconds (not 300s like the remote storage path)
//!   because token issuance is more sensitive than request signing.
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
//! v1.1.1-CrashSafe - Made secret configuration replacement durable and atomic.
//! v1.1.0-SectionSafe - Fixed cross-section secret replacement and first-start
//!                      runtime authentication gap; added legacy migration.
//! v1.0.1-Fix - Promoted write_secret_to_config to pub; added _pub alias;
//!              added comment-line guard in write_secret_to_config.
// ============================================

use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Error as IoError, ErrorKind as IoErrorKind, Write};
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

/// Canonical TOML section for MemChain API and JWT secrets.
const MEMCHAIN_CONFIG_SECTION: &str = "memchain";

/// Historical section used by the pre-v1.1.0 text writer.
const LEGACY_AUTH_CONFIG_SECTION: &str = "auth";

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
            info!(pubkey = &req.pubkey[..8], expires_at, "[AUTH] JWT issued");
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
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
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
    use tempfile::TempDir;

    const TEST_SECRET: &str = "test-secret-that-is-at-least-32-chars-long-for-safety";

    fn make_test_secret() -> String {
        TEST_SECRET.to_string()
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

    #[cfg(unix)]
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
}
