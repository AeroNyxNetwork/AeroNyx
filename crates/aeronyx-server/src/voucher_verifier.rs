// ============================================
// File: crates/aeronyx-server/src/voucher_verifier.rs
// ============================================
//! VPN voucher verifier.
//!
//! Phase 2 rollout policy: missing vouchers are still allowed for compatibility,
//! but malformed or cryptographically invalid vouchers are rejected before the
//! handshake is accepted. Phase 3 can switch missing vouchers to reject once
//! production metrics show old clients have drained out.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use blind_rsa_signatures::{MessageRandomizer, PublicKeySha384PSSRandomized, Signature};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Magic marker after the fixed 138-byte ClientHello.
pub const VOUCHER_EXTENSION_MAGIC: &[u8; 4] = b"AVCH";

const VOUCHER_EXTENSION_HEADER_SIZE: usize = 6;
const VOUCHER_EXTENSION_MAX_SIZE: usize = 2048;
const ISSUER_KEY_CACHE_TTL: Duration = Duration::from_secs(3600);
const DEFAULT_ISSUER_KEYS_URL: &str = "https://api.aeronyx.network/api/voucher/issuer/keys/";

#[derive(Debug, Clone, Deserialize)]
struct VoucherWire {
    token: String,
    #[serde(alias = "sig")]
    signature: String,
    #[serde(alias = "rand")]
    msg_randomizer: String,
    epoch: String,
}

#[derive(Debug, Deserialize)]
struct IssuerKeysResponse {
    success: bool,
    keys: Vec<IssuerKeyWire>,
}

#[derive(Debug, Deserialize)]
struct IssuerKeyWire {
    epoch: String,
    issuer_pk: String,
}

#[derive(Debug, Default)]
struct IssuerKeyCache {
    keys_by_epoch: HashMap<String, String>,
    fetched_at: Option<Instant>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoucherMetricsSnapshot {
    pub mode: &'static str,
    pub valid: u64,
    pub invalid: u64,
    pub missing: u64,
    pub malformed: u64,
    pub total: u64,
    pub valid_ratio: f64,
    pub invalid_ratio: f64,
    pub missing_ratio: f64,
    pub malformed_ratio: f64,
    pub last_observation: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Default)]
struct VoucherMetrics {
    valid: AtomicU64,
    invalid: AtomicU64,
    missing: AtomicU64,
    malformed: AtomicU64,
    last_observation: Mutex<Option<String>>,
    last_error: Mutex<Option<String>>,
}

impl VoucherMetrics {
    fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.valid.load(Ordering::Relaxed),
            self.invalid.load(Ordering::Relaxed),
            self.missing.load(Ordering::Relaxed),
            self.malformed.load(Ordering::Relaxed),
        )
    }

    fn total(&self) -> u64 {
        let (valid, invalid, missing, malformed) = self.snapshot();
        valid + invalid + missing + malformed
    }

    fn last(&self) -> (Option<String>, Option<String>) {
        let observation = self
            .last_observation
            .lock()
            .ok()
            .and_then(|value| value.clone());
        let error = self.last_error.lock().ok().and_then(|value| value.clone());
        (observation, error)
    }

    fn set_last(&self, observation: String, error: Option<String>) {
        if let Ok(mut value) = self.last_observation.lock() {
            *value = Some(observation);
        }
        if let Ok(mut value) = self.last_error.lock() {
            *value = error;
        }
    }
}

/// Voucher verification result for observation.
#[derive(Debug)]
enum VoucherObservation {
    Missing,
    Valid { epoch: String },
    Invalid { epoch: String, reason: String },
    Malformed { reason: String },
}

/// VPN voucher verifier.
#[derive(Clone)]
pub struct VoucherVerifier {
    issuer_keys_url: String,
    http: reqwest::Client,
    cache: Arc<RwLock<IssuerKeyCache>>,
    metrics: Arc<VoucherMetrics>,
}

impl VoucherVerifier {
    /// Creates a verifier using the production issuer key discovery endpoint.
    #[must_use]
    pub fn new() -> Self {
        Self::with_issuer_keys_url(DEFAULT_ISSUER_KEYS_URL.to_string())
    }

    /// Creates a verifier with an explicit issuer key discovery endpoint.
    #[must_use]
    pub fn with_issuer_keys_url(issuer_keys_url: String) -> Self {
        let http = reqwest::Client::builder()
            .user_agent("AeroNyx-VPN-Verifier/0.1")
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            issuer_keys_url,
            http,
            cache: Arc::new(RwLock::new(IssuerKeyCache::default())),
            metrics: Arc::new(VoucherMetrics::default()),
        }
    }

    /// Verify a ClientHello trailing extension and return whether to accept it.
    ///
    /// Phase 2 policy keeps missing vouchers compatible, but rejects malformed
    /// or invalid vouchers. Every path is still recorded in metrics.
    pub async fn accept_client_hello_extension(&self, extension: Vec<u8>) -> bool {
        let observation = self.verify_extension(&extension).await;
        let accept = matches!(
            observation,
            VoucherObservation::Missing | VoucherObservation::Valid { .. }
        );
        self.record(observation);
        accept
    }

    /// Observe a ClientHello trailing extension without affecting handshake.
    #[allow(dead_code)]
    pub async fn observe_client_hello_extension(&self, extension: Vec<u8>) {
        let observation = self.verify_extension(&extension).await;
        self.record(observation);
    }

    /// Returns current voucher counters for node health checks.
    #[must_use]
    pub fn metrics_snapshot(&self) -> VoucherMetricsSnapshot {
        let (valid, invalid, missing, malformed) = self.metrics.snapshot();
        let (last_observation, last_error) = self.metrics.last();
        let total = valid + invalid + missing + malformed;
        let ratio = |count: u64| {
            if total == 0 {
                0.0
            } else {
                count as f64 / total as f64
            }
        };
        VoucherMetricsSnapshot {
            mode: "reject_invalid",
            valid,
            invalid,
            missing,
            malformed,
            total,
            valid_ratio: ratio(valid),
            invalid_ratio: ratio(invalid),
            missing_ratio: ratio(missing),
            malformed_ratio: ratio(malformed),
            last_observation,
            last_error,
        }
    }

    async fn verify_extension(&self, extension: &[u8]) -> VoucherObservation {
        let voucher_json = match extract_voucher_json(extension) {
            Ok(Some(v)) => v,
            Ok(None) => return VoucherObservation::Missing,
            Err(reason) => return VoucherObservation::Malformed { reason },
        };

        let voucher: VoucherWire = match serde_json::from_str(&voucher_json) {
            Ok(v) => v,
            Err(e) => {
                return VoucherObservation::Malformed {
                    reason: format!("invalid_json: {e}"),
                };
            }
        };

        match self.verify_voucher(&voucher).await {
            Ok(true) => VoucherObservation::Valid {
                epoch: voucher.epoch,
            },
            Ok(false) => VoucherObservation::Invalid {
                epoch: voucher.epoch,
                reason: "signature_invalid".to_string(),
            },
            Err(reason) => VoucherObservation::Invalid {
                epoch: voucher.epoch,
                reason,
            },
        }
    }

    async fn verify_voucher(&self, voucher: &VoucherWire) -> Result<bool, String> {
        let issuer_pk_b64 = self
            .issuer_pk_for_epoch(&voucher.epoch)
            .await
            .ok_or_else(|| "issuer_key_missing".to_string())?;

        let pk_der = b64d(&issuer_pk_b64)?;
        let pk = PublicKeySha384PSSRandomized::from_der(&pk_der)
            .map_err(|e| format!("issuer_pk_parse_failed: {e}"))?;
        let token = b64d(&voucher.token)?;
        let sig = Signature::new(b64d(&voucher.signature)?);
        let randomizer = parse_randomizer(&voucher.msg_randomizer)?;

        Ok(pk.verify(&sig, Some(randomizer), &token).is_ok())
    }

    async fn issuer_pk_for_epoch(&self, epoch: &str) -> Option<String> {
        {
            let cache = self.cache.read().await;
            if cache
                .fetched_at
                .map(|t| t.elapsed() < ISSUER_KEY_CACHE_TTL)
                .unwrap_or(false)
            {
                if let Some(pk) = cache.keys_by_epoch.get(epoch) {
                    return Some(pk.clone());
                }
            }
        }

        if let Err(e) = self.refresh_issuer_keys().await {
            warn!(error = %e, "[VOUCHER] issuer key refresh failed");
        }

        let cache = self.cache.read().await;
        cache.keys_by_epoch.get(epoch).cloned()
    }

    async fn refresh_issuer_keys(&self) -> Result<(), String> {
        let resp = self
            .http
            .get(&self.issuer_keys_url)
            .send()
            .await
            .map_err(|e| format!("http: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("http_status: {}", resp.status()));
        }
        let body = resp
            .json::<IssuerKeysResponse>()
            .await
            .map_err(|e| format!("json: {e}"))?;
        if !body.success {
            return Err("issuer_response_not_success".to_string());
        }

        let mut next = HashMap::new();
        for key in body.keys {
            if !key.epoch.is_empty() && !key.issuer_pk.is_empty() {
                next.insert(key.epoch, key.issuer_pk);
            }
        }
        let count = next.len();
        let mut cache = self.cache.write().await;
        cache.keys_by_epoch = next;
        cache.fetched_at = Some(Instant::now());
        debug!(count, "[VOUCHER] issuer keys refreshed");
        Ok(())
    }

    fn record(&self, observation: VoucherObservation) {
        match observation {
            VoucherObservation::Missing => {
                self.metrics.missing.fetch_add(1, Ordering::Relaxed);
                self.metrics.set_last("missing".to_string(), None);
                debug!("[VOUCHER] observe missing voucher");
            }
            VoucherObservation::Valid { epoch } => {
                self.metrics.valid.fetch_add(1, Ordering::Relaxed);
                self.metrics.set_last(format!("valid:{epoch}"), None);
                debug!(epoch = %epoch, "[VOUCHER] observe valid voucher");
            }
            VoucherObservation::Invalid { epoch, reason } => {
                self.metrics.invalid.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .set_last(format!("invalid:{epoch}"), Some(reason.clone()));
                warn!(epoch = %epoch, reason = %reason, "[VOUCHER] observe invalid voucher");
            }
            VoucherObservation::Malformed { reason } => {
                self.metrics.malformed.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .set_last("malformed".to_string(), Some(reason.clone()));
                warn!(reason = %reason, "[VOUCHER] observe malformed voucher extension");
            }
        }

        let total = self.metrics.total();
        if total <= 20 || total % 100 == 0 {
            let (valid, invalid, missing, malformed) = self.metrics.snapshot();
            info!(
                valid,
                invalid,
                missing,
                malformed,
                total,
                mode = "reject_invalid",
                "[VOUCHER] observe-only metrics"
            );
        }
    }
}

impl Default for VoucherVerifier {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_voucher_json(extension: &[u8]) -> Result<Option<String>, String> {
    if extension.is_empty() {
        return Ok(None);
    }
    if extension.len() < VOUCHER_EXTENSION_HEADER_SIZE {
        return Err(format!("extension_too_short: len={}", extension.len()));
    }
    if &extension[..4] != VOUCHER_EXTENSION_MAGIC {
        return Err(format!(
            "unknown_extension_magic: len={} prefix={}",
            extension.len(),
            hex::encode(&extension[..extension.len().min(8)])
        ));
    }

    let len = u16::from_le_bytes([extension[4], extension[5]]) as usize;
    if len == 0 {
        return Err(format!("empty_voucher: ext_len={}", extension.len()));
    }
    if len > VOUCHER_EXTENSION_MAX_SIZE {
        return Err(format!("voucher_too_large: len={}", len));
    }
    if extension.len() < VOUCHER_EXTENSION_HEADER_SIZE + len {
        return Err(format!(
            "voucher_truncated: ext_len={} declared_len={}",
            extension.len(),
            len
        ));
    }

    let raw = &extension[VOUCHER_EXTENSION_HEADER_SIZE..VOUCHER_EXTENSION_HEADER_SIZE + len];
    String::from_utf8(raw.to_vec())
        .map(Some)
        .map_err(|e| format!("utf8: {e}"))
}

fn b64d(s: &str) -> Result<Vec<u8>, String> {
    BASE64
        .decode(s)
        .map_err(|e| format!("base64_decode_failed: {e}"))
}

fn parse_randomizer(s: &str) -> Result<MessageRandomizer, String> {
    let bytes = b64d(s)?;
    if bytes.len() != 32 {
        return Err(format!("randomizer_size: {}", bytes.len()));
    }
    let mut arr = [0_u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(MessageRandomizer::new(arr))
}
