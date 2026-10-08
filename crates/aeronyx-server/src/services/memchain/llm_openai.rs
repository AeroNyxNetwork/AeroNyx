// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_openai.rs
// ============================================
//! # OpenAI-Compatible Provider
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Implements `LlmProvider` for the OpenAI Chat Completions API format.
//! [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Enabled MemChain inference
//! selects only the dedicated Phala ACI constructor; generic adapters remain
//! for compatibility and cannot serve confidential requests.
//! [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Direct calls also refuse
//! plaintext inference, including the Phala ACI adapter. The retained wire
//! and evidence code below does not implement source-owned field encryption.
//!
//! ## Configuration
//! ```toml
//! [[memchain.supernode.providers]]
//! name = "phala"
//! type = "phala_aci"
//! api_key = "$PHALA_API_KEY"
//! model = "<Phala-supported-model>"
//! ```
//!
//! ## Request Format
//! POST the normalized endpoint with `X-Upstream-Verification: required`
//! for ACI calls. The retained `provider.aci_verified` body field is a product
//! routing extension, not the ACI header or proof of upstream verification.
//! [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] Neither constraint
//! enables the node's disabled plaintext inference transport.
//! - `{api_base}/chat/completions` when `api_base` already ends in `/v1`
//! - `{api_base}/v1/chat/completions` otherwise
//! ```json
//! {
//!   "model": "<Phala-supported-model>",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "max_tokens": 1000,
//!   "temperature": 0.3,
//!   "provider": {"aci_verified": true}
//! }
//! ```
//!
//! ## Response Parsing
//! Extracts `choices[0].message.content` and `usage.{prompt,completion}_tokens`.
//! Anthropic-style `cached_tokens` is extracted from `usage.prompt_tokens_details.cached_tokens`
//! if present (OpenAI o-series models).
//! [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Confidential
//! calls require ACI/1 version, keyset-digest, and receipt-id headers, then
//! bind them to the fresh attestation and signed receipt.
//! [PHALA-133171-PROFILE 2026-10-08 by Codex] X-ACI-Identity is also
//! required; its stable identity digest is not the rotating keyset digest.
//!
//! ⚠️ Important Note for Next Developer:
//! - $ENV_VAR api_key resolution happens at construction time (in `new()`).
//!   A missing explicit environment reference is a startup error.
//! - Ollama does not require Authorization header — empty api_key → no header sent.
//! - Timeout is fixed at 60s for now. TODO: make configurable per provider.
//! - The provider does NOT retry on failure — LlmRouter handles retry policy.
//! - All response bodies use the shared bounded reader; provider-controlled
//!   `Content-Length` is never trusted as the only memory guard.
//! - API base and `$ENV_VAR` resolution use the shared typed startup boundary;
//!   a configured missing environment variable is no longer treated as an
//!   implicit keyless provider.
//! - Provider traffic never inherits undeclared host proxy state.
//!
//! ## Last Modified
//! v2.5.4-StartupIntegrity - [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex]
//!   Fails closed on invalid API bases and unavailable configured secrets.
//! v2.5.3-ResponseBoundary - [LLM-RESPONSE-BOUNDARY 2026-07-30 by Codex]
//!   Bounded success/error bodies, added recoverable provider cooldowns, and
//!   normalized `/v1` API bases so configured endpoints are not duplicated.
//! v2.5.0+SuperNode - 🌟 Created.

use std::collections::HashSet;
use std::time::Instant;

use aci_protocol::types::WorkloadKeyset;
use rand::RngCore;
// [PHALA-ACI-RECEIPT-HASH 2026-10-06 by Codex]
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use super::llm_provider::{
    build_confidential_llm_http_client, build_llm_http_client,
    build_phala_pinned_http_client, normalize_llm_api_base,
    read_bounded_llm_response, resolve_llm_api_key, validate_phala_aci_api_base,
    serialize_bounded_aci_request, validate_aci_request_options,
    require_source_e2ee_transport, with_required_aci_upstream,
    ChatRequest, ChatResponse, LlmError,
    EmbeddingRequest, EmbeddingResponse,
    AciResponseHints, LlmProvider, LlmProviderInitError, ProviderHealth, TokenUsage, MAX_LLM_ERROR_BODY_BYTES,
    MAX_LLM_SUCCESS_BODY_BYTES, PhalaAciObservation,
};
use super::aci_receipt_signature::{
    verify_bounded_aci_receipt_signature, verify_aci_identity_endorsement,
    recover_aci_identity_kms_root,
};
use crate::config_supernode::AcceptedAciSourceProvenance;

// ============================================
// Request / Response wire types
// ============================================

#[derive(serde::Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<OpenAiProviderConstraints>,
}

#[derive(serde::Serialize)]
struct OpenAiProviderConstraints {
    aci_verified: bool,
}

#[derive(serde::Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    model: Option<String>,
}

// [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
#[derive(serde::Serialize)]
struct OpenAiEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    provider: OpenAiProviderConstraints,
}

#[derive(serde::Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
    model: String,
}

#[derive(serde::Deserialize)]
struct OpenAiEmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(serde::Deserialize)]
struct OpenAiChoiceMessage {
    content: String,
}

#[derive(serde::Deserialize, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    /// OpenAI o-series / cached prompt tokens
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiTokenDetails>,
}

#[derive(serde::Deserialize)]
struct OpenAiTokenDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(serde::Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorBody,
}

#[derive(serde::Deserialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type", default)]
    error_type: String,
}

// ============================================
// OpenAiCompatProvider
// ============================================

/// LLM provider for any OpenAI Chat Completions-compatible endpoint.
/// Covers: OpenAI, DeepSeek, Groq, Ollama, Together AI, Fireworks, etc.
pub struct OpenAiCompatProvider {
    /// Provider name (e.g. "deepseek", "ollama") — for logging and writeback.
    name: String,
    /// Full API base URL (e.g. "https://api.deepseek.com").
    api_base: String,
    /// API key (empty string = no Authorization header, e.g. for Ollama).
    api_key: String,
    /// Default model identifier (e.g. "deepseek-chat").
    model: String,
    /// Optional max_tokens override for all requests.
    max_tokens: Option<u32>,
    /// Optional temperature override for all requests.
    temperature: Option<f32>,
    /// Shared HTTP client (keep-alive connection pool).
    client: reqwest::Client,
    /// Monotonic cooldown state for router availability.
    health: ProviderHealth,
    /// Construction-time capability for Phala's ACI serving constraint.
    require_aci_verified: bool,
    /// Local relying-party allowlist for verifier-produced compose measurements.
    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    accepted_compose_hashes: HashSet<String>,
    /// Operator-reviewed provenance mappings tied to the same measured compose.
    // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
    accepted_source_provenance: HashSet<AcceptedAciSourceProvenance>,
    /// Locally trusted Dstack roots for receipt-signing-key custody.
    // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex]
    accepted_kms_root_public_keys: HashSet<String>,
}

impl OpenAiCompatProvider {
    /// Construct a new provider.
    ///
    /// `api_key` supports `$ENV_VAR` syntax — resolved at construction time.
    /// Empty string = no Authorization header (for Ollama and local endpoints).
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self, LlmProviderInitError> {
        Self::new_with_aci_policy(
            name, api_base, api_key, model, max_tokens, temperature, false, &[], &[], &[],
        )
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] This constructor is the
    // only path that marks an OpenAI-compatible provider as confidential.
    pub fn new_phala_aci(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        accepted_compose_hashes: &[String],
        accepted_source_provenance: &[AcceptedAciSourceProvenance],
        accepted_kms_root_public_keys: &[String],
    ) -> Result<Self, LlmProviderInitError> {
        Self::new_with_aci_policy(
            name,
            api_base,
            api_key,
            model,
            max_tokens,
            temperature,
            true,
            accepted_compose_hashes,
            accepted_source_provenance,
            accepted_kms_root_public_keys,
        )
    }

    fn new_with_aci_policy(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        require_aci_verified: bool,
        accepted_compose_hashes: &[String],
        accepted_source_provenance: &[AcceptedAciSourceProvenance],
        accepted_kms_root_public_keys: &[String],
    ) -> Result<Self, LlmProviderInitError> {
        // [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] An explicit
        // environment reference is a required configuration dependency. An
        // absent api_key remains valid for intentionally keyless local APIs.
        let api_key = resolve_llm_api_key(&api_key.into(), require_aci_verified)?;
        let raw_api_base = api_base.into();
        let api_base = if require_aci_verified {
            validate_phala_aci_api_base(&raw_api_base)?
        } else {
            normalize_llm_api_base(&raw_api_base)?
        };

        let client = if require_aci_verified {
            build_confidential_llm_http_client()?
        } else {
            build_llm_http_client()?
        };

        Ok(Self {
            name: name.into(),
            api_base,
            api_key,
            model: model.into(),
            max_tokens,
            temperature,
            client,
            health: ProviderHealth::default(),
            require_aci_verified,
            accepted_compose_hashes: accepted_compose_hashes.iter().cloned().collect(),
            accepted_source_provenance: accepted_source_provenance.iter().cloned().collect(),
            accepted_kms_root_public_keys: accepted_kms_root_public_keys
                .iter()
                .cloned()
                .collect(),
        })
    }
}

// [PHALA-133171-PROFILE 2026-10-08 by Codex] Explicit wire profile:
// Phala-Network/private-ai-gateway-with-vllm-router-as-middleware at
// 133171efb115bb0437f69a4679b5522c5e36139e. Do not deserialize this into
// a991's report (same api_version, different identity/binding semantics).
#[derive(serde::Serialize, serde::Deserialize)]
struct AttestationReport {
    api_version: String,
    workload_id: String,
    workload_keyset_digest: String,
    attestation: PhalaAttestationEnvelope,
    #[serde(default)]
    service_capabilities: aci_protocol::types::ServiceCapabilities,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PhalaAttestationEnvelope {
    tee_type: String,
    workload_keyset: serde_json::Value,
    #[serde(rename = "report_data")]
    report_data_hex: String,
    keyset_endorsement: serde_json::Value,
    #[serde(default)]
    source_provenance: aci_protocol::types::SourceProvenance,
    #[serde(default)]
    evidence: serde_json::Value,
}

struct PhalaReportBinding {
    // Projection for role/expiry/TLS helpers ONLY. It is never serialized or
    // used to compute the 133171 keyset digest or attestation statement.
    keyset: WorkloadKeyset,
    report_data: [u8; 32],
    identity_algorithm: String,
    identity_public_key: String,
}

// [PHALA-133171-PROFILE 2026-10-08 by Codex] Pure candidate binding, not
// hardware appraisal or endorsement verification. Always hash the served
// closed-schema object; never a lossy projection into a991 wire types.
fn bind_phala_report(
    report: &AttestationReport, nonce: &str, now: u64,
) -> Result<PhalaReportBinding, LlmError> {
    let invalid = || LlmError::AciResponseContractViolation;
    if report.api_version != "aci/1" || nonce.len() != 64 || !is_lower_hex_digest(nonce) {
        return Err(invalid());
    }
    validate_aci_keyset_resource_bounds(report)?;
    let only = |value: &serde_json::Value, names: &[&str]| {
        value.as_object().is_some_and(|object| object.keys().all(|key| names.contains(&key.as_str())))
    };
    let raw = &report.attestation.workload_keyset;
    if !only(raw, &["workload_identity", "keyset_epoch", "receipt_signing_keys", "e2ee_public_keys", "tls_public_keys"])
        || !only(&raw["workload_identity"], &["public_key", "subject"])
        || !only(&raw["workload_identity"]["public_key"], &["algo", "public_key"])
        || !only(&raw["keyset_epoch"], &["version", "not_after"])
    { return Err(invalid()); }
    let identity = &raw["workload_identity"]["public_key"];
    let algorithm = identity["algo"].as_str().ok_or_else(invalid)?;
    let public_key = identity["public_key"].as_str().ok_or_else(invalid)?;
    if !(match algorithm {
        "ed25519" => public_key.len() == 64 && is_lower_hex_digest(public_key),
        "ecdsa-secp256k1" => matches!(public_key.len(), 66 | 130)
            && public_key.bytes().all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v)),
        _ => false,
    }) { return Err(invalid()); }
    let subject = match &raw["workload_identity"]["subject"] {
        serde_json::Value::Null => None,
        serde_json::Value::String(subject) => Some(subject.clone()),
        _ => return Err(invalid()),
    };
    const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
    let version = raw["keyset_epoch"]["version"].as_u64().ok_or_else(invalid)?;
    let not_after = raw["keyset_epoch"]["not_after"].as_u64().ok_or_else(invalid)?;
    if version > MAX_SAFE_INTEGER || not_after > MAX_SAFE_INTEGER || now >= not_after {
        return Err(invalid());
    }
    for role in ["receipt_signing_keys", "e2ee_public_keys", "tls_public_keys"] {
        if role == "tls_public_keys" && raw.get(role).is_none() { continue; }
        let entries = raw[role].as_array().ok_or_else(invalid)?;
        let mut ids = HashSet::new();
        for entry in entries {
            let fields: &[&str] = if role == "tls_public_keys" { &["spki_sha256", "domain"] }
                else { &["key_id", "algo", "public_key"] };
            if !only(entry, fields) { return Err(invalid()); }
            if role != "tls_public_keys" {
                let id = entry["key_id"].as_str().filter(|id| !id.is_empty() && id.len() <= 256).ok_or_else(invalid)?;
                if !ids.insert(id) { return Err(invalid()); }
            }
        }
    }
    let digest = |value: &serde_json::Value| -> Result<String, LlmError> {
        let bytes = aci_protocol::digest::jcs_bytes(value).map_err(|_| invalid())?;
        Ok(format!("sha256:{}", hex::encode(Sha256::digest(bytes))))
    };
    if digest(identity)? != report.workload_id || digest(raw)? != report.workload_keyset_digest {
        return Err(invalid());
    }
    let statement = serde_json::json!({
        "purpose": "aci.report_data.v1", "nonce": nonce,
        "workload_id": report.workload_id,
        "workload_keyset_digest": report.workload_keyset_digest,
    });
    let report_data: [u8; 32] = Sha256::digest(
        aci_protocol::digest::jcs_bytes(&statement).map_err(|_| invalid())?
    ).into();
    if report.attestation.report_data_hex != hex::encode(report_data) { return Err(invalid()); }
    let keyset = WorkloadKeyset {
        subject, not_after,
        receipt_signing_keys: serde_json::from_value(raw["receipt_signing_keys"].clone()).map_err(|_| invalid())?,
        e2ee_public_keys: serde_json::from_value(raw["e2ee_public_keys"].clone()).map_err(|_| invalid())?,
        tls_public_keys: match raw.get("tls_public_keys") {
            Some(value) => serde_json::from_value(value.clone()).map_err(|_| invalid())?,
            None => Vec::new(),
        },
    };
    if keyset.receipt_signing_keys.iter().any(|signing| keyset.e2ee_public_keys.iter()
        .any(|e2ee| e2ee.public_key_hex == signing.public_key_hex))
    { return Err(invalid()); }
    Ok(PhalaReportBinding { keyset, report_data, identity_algorithm: algorithm.to_owned(), identity_public_key: public_key.to_owned() })
}

// [MEMCHAIN-PHALA-ACI-REPORT-VERIFY 2026-10-05 by Codex]
struct VerifiedPhalaIdentity {
    report: AttestationReport,
    keyset: WorkloadKeyset,
    report_json: serde_json::Value,
    compose_hash: String,
    kms_root_public_key: String,
    tls_pins: Vec<String>,
}

// [MEMCHAIN-PHALA-ACI-SESSION 2026-10-05 by Codex]
struct VerifiedPhalaContext {
    identity: VerifiedPhalaIdentity,
    client: reqwest::Client,
    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Only the
    // current request owns this nonce/report and its monotonic clock floor.
    observation: PhalaAciObservation,
    nonce: String,
}

impl VerifiedPhalaContext {
    fn current_time(&self) -> Result<u64, LlmError> {
        let now = self.observation.observe()?;
        validate_aci_keyset_resource_bounds(&self.identity.report)?;
        bind_phala_report(&self.identity.report, &self.nonce, now)?;
        if !aci_keyset_accepts_request(&self.identity.keyset, now) {
            return Err(LlmError::AciResponseContractViolation);
        }
        Ok(now)
    }
}

// [PHALA-133171-PROFILE 2026-10-08 by Codex] Bound each role before
// cross-role comparisons, including repeated final acceptance checks.
fn validate_aci_keyset_resource_bounds(report: &AttestationReport) -> Result<(), LlmError> {
    const MAX_KEYS_PER_ROLE: usize = 64;
    let object = report.attestation.workload_keyset.as_object()
        .ok_or(LlmError::AciResponseContractViolation)?;
    for role in ["receipt_signing_keys", "e2ee_public_keys", "tls_public_keys"] {
        let keys = match object.get(role) {
            None if role == "tls_public_keys" => continue,
            Some(value) => value.as_array().ok_or(LlmError::AciResponseContractViolation)?,
            None => return Err(LlmError::AciResponseContractViolation),
        };
        if keys.len() > MAX_KEYS_PER_ROLE { return Err(LlmError::AciResponseContractViolation); }
    }
    Ok(())
}

// [MEMCHAIN-PHALA-KEYSET-EXPIRY 2026-10-06 by Codex]
fn aci_keyset_accepts_request(keyset: &WorkloadKeyset, now: u64) -> bool {
    !keyset.is_expired_at(now)
}

// [MEMCHAIN-PHALA-KEYSET-EXPIRY 2026-10-07 by Codex] ACI `not_after` is an
// exclusive acceptance boundary for both the signed serving time and the
// verifier's current time; a slow receipt/session lookup cannot revive it.
fn aci_receipt_is_current_for_keyset(keyset: &WorkloadKeyset, served_at: u64, now: u64) -> bool {
    aci_keyset_accepts_request(keyset, served_at)
        && aci_keyset_accepts_request(keyset, now)
}

// [MEMCHAIN-PHALA-ACI-REPORT-VERIFY 2026-10-05 by Codex]
async fn establish_phala_identity(
    client: &reqwest::Client,
    api_base: &str,
    accepted_compose_hashes: &HashSet<String>,
    accepted_source_provenance: &HashSet<AcceptedAciSourceProvenance>,
    accepted_kms_roots: &HashSet<String>,
) -> Result<VerifiedPhalaContext, LlmError> {
    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] One budget
    // spans nonce/report IO, fixed-origin collateral and blocking QVL/ACI.
    // Cancellation never releases a permit while its crypto child is alive.
    let observation = PhalaAciObservation::begin()?;
    let (identity, observation, nonce) = crate::api::discovery::with_phala_appraisal_budget(|permit| async move {
        let mut nonce_bytes = [0_u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
        if nonce_bytes == [0; 32] { return Err(LlmError::AciVerifierUnavailable); }
        let nonce = hex::encode(nonce_bytes);
        let mut endpoint = aci_v1_resource_url(api_base, &["attestation"])?;
        endpoint.query_pairs_mut().append_pair("nonce", &nonce);
        let response = client.get(endpoint).send().await
            .map_err(|_| LlmError::AciVerifierUnavailable)?;
        if !response.status().is_success() { return Err(LlmError::AciVerifierUnavailable); }
        const MAX_ACI_REPORT_BYTES: usize = 1024 * 1024;
        let report_bytes = read_bounded_llm_response(response, MAX_ACI_REPORT_BYTES).await?;
        observation.observe()?;
        // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Reject
        // ambiguous keysets/evidence before typed or Value parsing loses them.
        super::validate_phala_json(&report_bytes, MAX_ACI_REPORT_BYTES)
            .map_err(|_| LlmError::AciResponseContractViolation)?;
        let report: AttestationReport = serde_json::from_slice(&report_bytes)
            .map_err(|_| LlmError::AciResponseContractViolation)?;
        let quote = aci_verify::quote::quote_bytes(&report.attestation.evidence)
            .map_err(|_| LlmError::AciResponseContractViolation)?;
        let collateral = crate::api::discovery::fetch_bounded_phala_collateral(&quote).await
            // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Invalid
            // or unsupported evidence is a contract stop, not an IO outage.
            .map_err(|error| match error {
                "peer_collateral_malformed" | "peer_collateral_unsupported_form" | "peer_collateral_oversized" => LlmError::AciResponseContractViolation,
                _ => LlmError::AciVerifierUnavailable,
            })?;
        let accepted_compose_hashes = HashSet::clone(accepted_compose_hashes);
        let accepted_source_provenance = HashSet::clone(accepted_source_provenance);
        let accepted_kms_roots = HashSet::clone(accepted_kms_roots);
        crate::api::discovery::run_phala_appraisal_crypto(permit, move || {
            let now = observation.observe()?;
            if observation.started().elapsed() > crate::api::discovery::PHALA_APPRAISAL_TIMEOUT {
                return Err(LlmError::AciVerifierUnavailable);
            }
            let identity = verify_phala_identity_with_collateral(
                &report_bytes, &nonce, &accepted_compose_hashes,
                &accepted_source_provenance, &accepted_kms_roots, &collateral, now,
            )?;
            let after_crypto = observation.observe()?;
            bind_phala_report(&identity.report, &nonce, after_crypto)?;
            if !aci_keyset_accepts_request(&identity.keyset, after_crypto)
                || observation.started().elapsed() > crate::api::discovery::PHALA_APPRAISAL_TIMEOUT
            {
                return Err(LlmError::AciVerifierUnavailable);
            }
            Ok((identity, observation, nonce))
        }).await.map_err(|_| LlmError::AciVerifierUnavailable)?
    }).await.map_err(|_| LlmError::AciVerifierUnavailable)??;
    let client = build_phala_pinned_http_client(&identity.tls_pins)
        .map_err(|_| LlmError::AciVerifierUnavailable)?;
    let context = VerifiedPhalaContext { identity, client, observation, nonce };
    context.current_time()?;
    Ok(context)
}

// [MEMCHAIN-PHALA-OFFLINE-COLLATERAL 2026-10-06 by Codex]
fn verify_phala_identity_with_collateral(
    report_bytes: &[u8],
    nonce: &str,
    accepted_compose_hashes: &HashSet<String>,
    accepted_source_provenance: &HashSet<AcceptedAciSourceProvenance>,
    accepted_kms_roots: &HashSet<String>,
    collateral: &dcap_qvl::QuoteCollateralV3,
    now: u64,
) -> Result<VerifiedPhalaIdentity, LlmError> {
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] The offline
    // verification boundary also checks raw input, not only its HTTP caller.
    super::validate_phala_json(report_bytes, 1024 * 1024)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Injected
    // collateral uses the same signed-subtree profile as PCCS ingestion.
    // ACI report/keyset and receipt JCS remain outside this restriction.
    crate::api::discovery::validate_phala_collateral_signed_inputs(collateral)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let report_json: serde_json::Value = serde_json::from_slice(report_bytes)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let report: AttestationReport = serde_json::from_value(report_json.clone())
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    if report.attestation.tee_type != "tdx"
        || report.service_capabilities.serving != "aggregator"
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    validate_aci_keyset_resource_bounds(&report)?;
    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Hardware appraisal
    // remains inside the bounded crypto child. Endorsement is required in
    // addition to, never instead of, quote binding and accepted provenance.
    let bound = bind_phala_report(&report, nonce, now)?;
    // This appraisal targets an ACI service, not a receipt-only fixture. The
    // profile requires at least one client-facing E2EE key; capabilities alone
    // cannot supply one or authorize a source-owned encrypted request.
    if !bound.keyset.e2ee_public_keys.iter().any(|key| match key.algo.as_str() {
        "x25519-aes-256-gcm-hkdf-sha256" => is_lower_hex_digest(&key.public_key_hex),
        "secp256k1-aes-256-gcm-hkdf-sha256" => matches!(key.public_key_hex.len(), 66 | 130)
            && key.public_key_hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        _ => false,
    }) { return Err(LlmError::AciResponseContractViolation); }
    let endorsement = &report.attestation.keyset_endorsement;
    let endorsement_payload = aci_protocol::digest::jcs_bytes(&serde_json::json!({
        "purpose": "aci.keyset.endorsement.v1",
        "workload_keyset_digest": report.workload_keyset_digest,
    })).map_err(|_| LlmError::AciResponseContractViolation)?;
    if endorsement.get("algo").and_then(|v| v.as_str()) != Some(bound.identity_algorithm.as_str())
        || !endorsement.get("value").and_then(|v| v.as_str()).is_some_and(|signature|
            verify_aci_identity_endorsement(&bound.identity_algorithm, &bound.identity_public_key, signature, &endorsement_payload))
    { return Err(LlmError::AciResponseContractViolation); }
    // [MEMCHAIN-PHALA-KEYSET-EXPIRY 2026-10-06 by Codex] A fresh hardware
    // quote does not revive an expired operational keyset.
    if !aci_keyset_accepts_request(&bound.keyset, now) {
        return Err(LlmError::AciResponseContractViolation);
    }

    let quote = aci_verify::quote::quote_bytes(&report.attestation.evidence)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    // The offline/injected boundary must enforce the same allocation guard.
    crate::api::discovery::guard_phala_quote_allocations(&quote)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let quote_report = dcap_qvl::verify::rustcrypto::verify(&quote, collateral, now)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    if quote_report.status != "UpToDate" {
        return Err(LlmError::AciResponseContractViolation);
    }
    let td_report = quote_report
        .report
        .as_td10()
        .ok_or(LlmError::AciResponseContractViolation)?;
    aci_verify::quote::quote_binds_report_data(
        &report.attestation.evidence,
        &td_report.report_data,
        bound.report_data,
    )
    .map_err(|_| LlmError::AciResponseContractViolation)?;
    let event_log = aci_verify::dstack::verify_dstack_event_log(
        &report.attestation.evidence,
        Some(&td_report.rt_mr3),
    )
    .map_err(|_| LlmError::AciResponseContractViolation)?;
    let compose_hash = aci_verify::dstack::verify_dstack_compose_measurement(
        &report.attestation.evidence,
        &event_log,
    )
    .map_err(|_| LlmError::AciResponseContractViolation)?;
    let compose_hash = format!("sha256:{compose_hash}");
    if !accepted_compose_hashes.contains(&compose_hash)
        || !has_acceptable_aci_source_provenance(
            &report.attestation.source_provenance,
            &compose_hash,
            accepted_source_provenance,
        )
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let app_id = aci_verify::dstack::dstack_app_id(&event_log)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    if bound.identity_algorithm != "ecdsa-secp256k1" {
        return Err(LlmError::AciResponseContractViolation);
    }
    let kms_root_public_key = recover_aci_identity_kms_root(
        &report.attestation.evidence, &bound.identity_public_key, &app_id,
    ).ok_or(LlmError::AciResponseContractViolation)?;
    if !accepted_kms_roots.contains(&kms_root_public_key) {
        return Err(LlmError::AciResponseContractViolation);
    }
    let tls_pins = aci_verify::channel::declared_tls_pins(
        &bound.keyset,
        &report.attestation.evidence,
        "https://inference.phala.com",
    )
    .map_err(|_| LlmError::AciResponseContractViolation)?;
    if tls_pins.is_empty() {
        return Err(LlmError::AciResponseContractViolation);
    }

    Ok(VerifiedPhalaIdentity {
        compose_hash,
        kms_root_public_key,
        report,
        keyset: bound.keyset,
        report_json,
        tls_pins,
    })
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn has_acceptable_aci_source_provenance(
    provenance: &aci_protocol::types::SourceProvenance,
    measured_compose_hash: &str,
    accepted: &HashSet<AcceptedAciSourceProvenance>,
) -> bool {
    let repo_revision_is_valid = match (
        provenance.repo_url.as_deref(),
        provenance.repo_commit.as_deref(),
    ) {
        (Some(repo_url), Some(revision)) => {
            let valid_url = reqwest::Url::parse(repo_url).is_ok_and(|url| {
                url.scheme() == "https"
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none()
            });
            valid_url
                && matches!(revision.len(), 40 | 64)
                && revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }
        (None, None) => false,
        _ => false,
    };
    let image_digest_is_valid = match provenance.image_digest.as_deref() {
        Some(digest) => digest
            .strip_prefix("sha256:")
            .is_some_and(is_lower_hex_digest),
        None => false,
    };
    let repo_fields_well_formed = match (
        provenance.repo_url.as_deref(),
        provenance.repo_commit.as_deref(),
    ) {
        (None, None) => true,
        (Some(_), Some(_)) => repo_revision_is_valid,
        _ => false,
    };
    let image_field_well_formed = provenance.image_digest.is_none() || image_digest_is_valid;

    repo_fields_well_formed
        && image_field_well_formed
        && (repo_revision_is_valid || image_digest_is_valid)
        && provenance.image_provenance.is_none()
        && accepted.iter().any(|mapping| {
            mapping.compose_hash == measured_compose_hash
                && mapping.repo_url.as_deref() == provenance.repo_url.as_deref()
                && mapping.repo_commit.as_deref() == provenance.repo_commit.as_deref()
                && mapping.image_digest.as_deref() == provenance.image_digest.as_deref()
        })
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn parse_aci_https_origin(value: &str) -> Option<reqwest::Url> {
    let url = reqwest::Url::parse(value).ok()?;
    (url.scheme() == "https"
        && url.host_str().is_some()
        && url.path() == "/"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none())
    .then_some(url)
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn is_aci_endpoint(value: &serde_json::Value) -> bool {
    value.is_null()
        || value
            .as_str()
            .and_then(parse_aci_https_origin)
            .is_some()
}

// [MEMCHAIN-PHALA-ACI-UPSTREAM-BINDING 2026-10-06 by Codex]
fn has_enforceable_aci_channel_binding(bindings: &serde_json::Value) -> bool {
    bindings.as_array().is_some_and(|bindings| {
        bindings.iter().any(|binding| {
            let Some(kind) = binding.get("type").and_then(serde_json::Value::as_str) else {
                return false;
            };
            match kind {
                "tls_spki_sha256" => {
                    binding.get("origin").and_then(serde_json::Value::as_str)
                        .and_then(parse_aci_https_origin).is_some()
                        && binding.get("spki_sha256").and_then(serde_json::Value::as_str)
                            .is_some_and(is_lower_hex_digest)
                }
                "tls_certificate_sha256" => {
                    binding.get("origin").and_then(serde_json::Value::as_str)
                        .and_then(parse_aci_https_origin).is_some()
                        && binding.get("certificate_sha256").and_then(serde_json::Value::as_str)
                            .is_some_and(is_lower_hex_digest)
                }
                "e2ee_public_key_sha256" => {
                    binding.get("provider").and_then(serde_json::Value::as_str)
                        .is_some_and(|provider| !provider.trim().is_empty())
                        && binding.get("algorithm").and_then(serde_json::Value::as_str)
                            .is_some_and(|algorithm| !algorithm.trim().is_empty())
                        && binding.get("public_key_sha256").and_then(serde_json::Value::as_str)
                            .is_some_and(is_lower_hex_digest)
                }
                _ => false,
            }
        })
    })
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn verify_aci_session(
    session: &serde_json::Value,
    expected_id: &str,
    receipt_served_at: u64,
) -> bool {
    let Some(object) = session.as_object() else {
        return false;
    };
    if object.get("api_version").and_then(serde_json::Value::as_str) != Some("aci/1")
        || !is_aci_session_id(expected_id)
        || object
            .get("upstream_name")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
        // `verifier_id` is required by the pinned ACI v1 session record.
        || object
            .get("verifier_id")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        || object
            .get("endpoint")
            .is_some_and(|endpoint| !is_aci_endpoint(endpoint))
        || object.get("channel_binding")
            .is_none_or(|bindings| !has_enforceable_aci_channel_binding(bindings))
        || object.get("claims").is_none_or(|claims| !claims.is_object())
        || object
            .get("identity")
            .is_some_and(|identity| !identity.is_null() && !identity.is_object())
    {
        return false;
    }
    let Some(established_at) = object.get("established_at").and_then(serde_json::Value::as_u64)
    else {
        return false;
    };
    let Some(expires_at) = object.get("expires_at").and_then(serde_json::Value::as_u64) else {
        return false;
    };
    // [MEMCHAIN-PHALA-ACI-SESSION-RETENTION 2026-10-06 by Codex] ACI's
    // expires_at is a retention deadline, not an upstream-validity claim. The
    // receipt must have been served after establishment and before retention.
    if established_at > receipt_served_at
        || expires_at <= established_at
        || expires_at <= receipt_served_at
    {
        return false;
    }
    let Some(evidence) = object.get("evidence") else {
        return false;
    };
    let (Some(digest), Some(data)) = (
        evidence.get("digest").and_then(serde_json::Value::as_str),
        evidence.get("data").and_then(serde_json::Value::as_str),
    ) else {
        return false;
    };
    let Some((metadata, encoded)) = data.split_once(',') else {
        return false;
    };
    if !metadata.starts_with("data:") || !metadata.ends_with(";base64") {
        return false;
    }
    use base64::Engine as _;
    let Ok(evidence_bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        return false;
    };
    let actual_digest = format!("sha256:{}", hex::encode(Sha256::digest(&evidence_bytes)));
    if digest != actual_digest {
        return false;
    }
    if object.get("session_id").and_then(serde_json::Value::as_str) != Some(expected_id) {
        return false;
    }
    // [MEMCHAIN-PHALA-ACI-SESSION-ID 2026-10-06 by Codex] ACI hashes only the
    // immutable session material; timestamps and fetched evidence bytes are
    // deliberately excluded, while omitted optional fields become null.
    let Some(material) = aci_session_id_material(session) else {
        return false;
    };
    let Ok(canonical) = aci_protocol::digest::jcs_bytes(&material) else {
        return false;
    };
    let actual_id = format!("as_{}", hex::encode(Sha256::digest(&canonical)));
    if actual_id != expected_id {
        return false;
    }
    let tee_claim = object
        .get("claims")
        .and_then(|claims| claims.get("tee_attested"));
    let tcb_claim = object
        .get("claims")
        .and_then(|claims| claims.get("tcb_up_to_date"));
    let claim_is_hardware_asserted = |claim: Option<&serde_json::Value>| {
        claim.is_some_and(|claim| {
            claim.get("status").and_then(serde_json::Value::as_str) == Some("asserted")
                && claim.get("source").and_then(serde_json::Value::as_str)
                    == Some("hardware_proven")
        })
    };
    let tcb_is_current = tcb_claim.is_some_and(|claim| {
        claim.get("status").and_then(serde_json::Value::as_str) == Some("asserted")
            && matches!(
                claim.get("source").and_then(serde_json::Value::as_str),
                Some("hardware_proven" | "verifier_derived")
            )
    });
    if !claim_is_hardware_asserted(tee_claim) || !tcb_is_current {
        return false;
    }
    let Some(bindings) = object
        .get("channel_binding")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    let endpoint = object
        .get("endpoint")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_aci_https_origin);
    let has_bound_channel = bindings.iter().any(|binding| {
        match binding.get("type").and_then(serde_json::Value::as_str) {
            Some("tls_spki_sha256" | "tls_certificate_sha256") => {
                let origin = binding
                    .get("origin")
                    .and_then(serde_json::Value::as_str)
                    .and_then(parse_aci_https_origin);
                let endpoint_matches = endpoint.as_ref().is_some_and(|endpoint| {
                    origin
                        .as_ref()
                        .is_some_and(|origin| endpoint.origin() == origin.origin())
                });
                let pin = match binding.get("type").and_then(serde_json::Value::as_str) {
                    Some("tls_spki_sha256") => binding
                        .get("spki_sha256")
                        .and_then(serde_json::Value::as_str),
                    Some("tls_certificate_sha256") => binding
                        .get("certificate_sha256")
                        .and_then(serde_json::Value::as_str),
                    _ => None,
                };
                endpoint_matches && pin.is_some_and(is_lower_hex_digest)
            }
            Some("e2ee_public_key_sha256") => {
                binding
                    .get("provider")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|provider| !provider.is_empty())
                    && binding
                        .get("algorithm")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|algorithm| !algorithm.is_empty())
                    && binding
                        .get("public_key_sha256")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(is_lower_hex_digest)
            }
            _ => false,
        }
    });
    has_bound_channel
}

// [MEMCHAIN-PHALA-ACI-SESSION 2026-10-05 by Codex]
fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn is_aci_session_id(value: &str) -> bool {
    value.strip_prefix("as_").is_some_and(is_lower_hex_digest)
}

// [MEMCHAIN-PHALA-ACI-SESSION-ID 2026-10-06 by Codex]
fn aci_session_id_material(session: &serde_json::Value) -> Option<serde_json::Value> {
    let object = session.as_object()?;
    let upstream_name = object.get("upstream_name")?.as_str()?;
    let verifier_id = object.get("verifier_id")?.as_str()?;
    let channel_binding = object.get("channel_binding")?;
    let claims = object.get("claims")?;
    let evidence_digest = object.get("evidence")?.get("digest")?.clone();
    Some(serde_json::json!({
        "upstream_name": upstream_name,
        "endpoint": object.get("endpoint").cloned().unwrap_or(serde_json::Value::Null),
        "verifier_id": verifier_id,
        "identity": object.get("identity").cloned().unwrap_or(serde_json::Value::Null),
        "channel_binding": channel_binding,
        "claims": claims,
        "evidence_digest": evidence_digest,
    }))
}

// [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex]
fn aci_receipt_signing_bytes(receipt: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
    let mut signing_object = receipt.clone();
    let signature = signing_object
        .as_object_mut()
        .and_then(|object| object.get_mut("signature"))
        .ok_or(LlmError::AciResponseContractViolation)?;
    signature.as_object_mut()
        .and_then(|signature| signature.remove("value"))
        .filter(serde_json::Value::is_string)
        .ok_or(LlmError::AciResponseContractViolation)?;
    aci_protocol::digest::jcs_bytes(&signing_object)
        .map_err(|_| LlmError::AciResponseContractViolation)
}

// [MEMCHAIN-PHALA-ACI-RESPONSE-TRANSPARENCY 2026-10-06 by Codex]
fn response_transparency_is_consistent(
    events: &[serde_json::Value],
    returned_hash: &str,
) -> bool {
    // [MEMCHAIN-PHALA-RESPONSE-ORDER 2026-10-06 by Codex]
    // A signed marker after response.returned cannot explain a transformation
    // that already crossed the gateway's response boundary.
    let mut returned_responses = events.iter().filter(|event| {
        event.get("type").and_then(serde_json::Value::as_str)
            == Some("response.returned")
    });
    let Some(returned_response) = returned_responses.next() else {
        return false;
    };
    if returned_responses.next().is_some() {
        return false;
    }
    let Some(returned_seq) = returned_response
        .get("seq")
        .and_then(serde_json::Value::as_u64)
    else {
        return false;
    };

    let received_responses = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(serde_json::Value::as_str)
                == Some("response.received")
        })
        .collect::<Vec<_>>();
    let modification_events = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(serde_json::Value::as_str)
                == Some("transparency.response_modified")
        })
        .collect::<Vec<_>>();
    if received_responses.is_empty() {
        return modification_events.is_empty();
    }
    if received_responses.len() != 1 {
        return false;
    }
    let received_response = received_responses[0];
    let Some(received_seq) = received_response
        .get("seq")
        .and_then(serde_json::Value::as_u64)
    else {
        return false;
    };
    if received_seq >= returned_seq {
        return false;
    }
    let Some(received_hash) = received_response
        .get("cleartext_hash")
        .and_then(serde_json::Value::as_str)
        .filter(|hash| hash.strip_prefix("sha256:").is_some_and(is_lower_hex_digest))
    else {
        return false;
    };
    if received_hash == returned_hash {
        return modification_events.is_empty();
    }
    if modification_events.len() != 1 {
        return false;
    }
    modification_events[0]
        .get("seq")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|modified_seq| received_seq < modified_seq && modified_seq < returned_seq)
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn aci_v1_resource_url(api_base: &str, resource: &[&str]) -> Result<reqwest::Url, LlmError> {
    let mut url = reqwest::Url::parse(api_base)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let has_v1_prefix = url.path().trim_end_matches('/').ends_with("/v1");
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    segments.pop_if_empty();
    if !has_v1_prefix {
        segments.push("v1");
    }
    segments.push("aci");
    for part in resource {
        segments.push(part);
    }
    drop(segments);
    Ok(url)
}

// [MEMCHAIN-PHALA-ACI-RECEIPT 2026-10-06 by Codex]
async fn fetch_and_verify_aci_receipt(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    expected_hints: &AciResponseHints,
    context: &VerifiedPhalaContext,
    expected_endpoint: &str,
    model: &str,
    request_bytes: &[u8],
    response_bytes: &[u8],
) -> Result<(serde_json::Value, String, serde_json::Value), LlmError> {
    const MAX_RECEIPT_AGE_SECS: u64 = 5 * 60;
    const MAX_RECEIPT_FUTURE_SKEW_SECS: u64 = 30;
    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Preserve one
    // observation floor from nonce issuance through receipt/session acceptance.
    context.current_time()?;
    let identity = &context.identity;
    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Unauthenticated lookup
    // hints must match BOTH the stable identity and the attested keyset.
    if expected_hints.keyset_digest != identity.report.workload_keyset_digest
        || expected_hints.workload_id != identity.report.workload_id
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let receipt_url = aci_v1_resource_url(
        api_base,
        &["receipts", expected_hints.receipt_id.as_str()],
    )?;
    let mut request = client.get(receipt_url);
    if !api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {api_key}"));
    }
    let response = request
        .send()
        .await
        .map_err(|error| LlmError::Transport(error.to_string()))?;
    if !response.status().is_success() {
        return Err(LlmError::AciResponseContractViolation);
    }
    const MAX_ACI_RECEIPT_BYTES: usize = 1024 * 1024;
    let receipt_bytes = read_bounded_llm_response(response, MAX_ACI_RECEIPT_BYTES).await?;
    context.current_time()?;
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Receipt JCS
    // must not canonicalize a last-member-wins interpretation of the wire.
    super::validate_phala_json(&receipt_bytes, MAX_ACI_RECEIPT_BYTES)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    let object = receipt
        .as_object()
        .ok_or(LlmError::AciResponseContractViolation)?;
    if object.get("api_version").and_then(serde_json::Value::as_str) != Some("aci/1")
        || object.get("workload_id").and_then(serde_json::Value::as_str)
            != Some(identity.report.workload_id.as_str())
        || object.get("receipt_id").and_then(serde_json::Value::as_str)
            != Some(expected_hints.receipt_id.as_str())
        || object.get("workload_keyset_digest").and_then(serde_json::Value::as_str)
            != Some(identity.report.workload_keyset_digest.as_str())
        || object.get("method").and_then(serde_json::Value::as_str) != Some("POST")
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let endpoint = object
        .get("endpoint")
        .and_then(serde_json::Value::as_str)
        .ok_or(LlmError::AciResponseContractViolation)?;
    if endpoint != expected_endpoint {
        return Err(LlmError::AciResponseContractViolation);
    }
    if object.get("model").and_then(serde_json::Value::as_str) != Some(model) {
        return Err(LlmError::AciResponseContractViolation);
    }
    let signature_object = object
        .get("signature")
        .and_then(serde_json::Value::as_object)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let algorithm = signature_object
        .get("algo")
        .and_then(serde_json::Value::as_str)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let key_id = signature_object
        .get("key_id")
        .and_then(serde_json::Value::as_str)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let key = identity
        .keyset
        .receipt_signing_keys
        .iter()
        .find(|key| key.key_id == key_id)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let signature_hex = signature_object
        .get("value")
        .and_then(serde_json::Value::as_str)
        .ok_or(LlmError::AciResponseContractViolation)?;
    if key.algo != algorithm {
        return Err(LlmError::AciResponseContractViolation);
    }
    let signing_input = aci_receipt_signing_bytes(&receipt)?;
    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] ACI algorithm,
    // key id, and key bytes all come from the already-attested keyset entry.
    if !verify_bounded_aci_receipt_signature(
        key.algo.clone(), signature_hex.to_owned(), key.public_key_hex.clone(), signing_input,
    ).await? {
        return Err(LlmError::AciResponseContractViolation);
    }
    context.current_time()?;

    let events = object
        .get("event_log")
        .and_then(serde_json::Value::as_array)
        .ok_or(LlmError::AciResponseContractViolation)?;
    if events.is_empty() || events.len() > 256 {
        return Err(LlmError::AciResponseContractViolation);
    }
    if events.iter().enumerate().any(|(index, event)| {
        event.get("seq").and_then(serde_json::Value::as_u64) != Some(index as u64)
    }) {
        return Err(LlmError::AciResponseContractViolation);
    }
    let one_event = |event_type: &str| -> Result<&serde_json::Value, LlmError> {
        let mut matches = events.iter().filter(|event| {
            event.get("type").and_then(serde_json::Value::as_str) == Some(event_type)
        });
        let event = matches.next().ok_or(LlmError::AciResponseContractViolation)?;
        if matches.next().is_some() {
            return Err(LlmError::AciResponseContractViolation);
        }
        Ok(event)
    };
    let received = one_event("request.received")?;
    let forwarded = one_event("request.forwarded")?;
    let returned = one_event("response.returned")?;
    if events.first().and_then(|event| event.get("type")).and_then(serde_json::Value::as_str)
        != Some("request.received")
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let expected_request_hash = format!("sha256:{}", hex::encode(Sha256::digest(request_bytes)));
    let expected_response_hash = format!("sha256:{}", hex::encode(Sha256::digest(response_bytes)));
    let forwarded_hash = forwarded
        .get("body_hash")
        .and_then(serde_json::Value::as_str)
        .ok_or(LlmError::AciResponseContractViolation)?;
    if received.get("body_hash").and_then(serde_json::Value::as_str)
        != Some(expected_request_hash.as_str())
        || !forwarded_hash.strip_prefix("sha256:").is_some_and(is_lower_hex_digest)
        || returned.get("wire_hash").and_then(serde_json::Value::as_str)
            != Some(expected_response_hash.as_str())
        || returned.get("cleartext_hash").and_then(serde_json::Value::as_str)
            != Some(expected_response_hash.as_str())
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    if forwarded_hash != expected_request_hash
        && !events.iter().any(|event| {
            event.get("type").and_then(serde_json::Value::as_str)
                == Some("transparency.request_modified")
        })
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    // ACI requires a signed marker when the aggregator transforms a received
    // upstream response before returning it to the client.
    if !response_transparency_is_consistent(events, &expected_response_hash) {
        return Err(LlmError::AciResponseContractViolation);
    }
    // ACI reports service-side request rewriting by the two signed hashes;
    // custom marker event types are optional and are not part of this contract.
    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Accept failed
    // required verification attempts, but exactly one required route may serve.
    let (session_id, receipt_channel_bindings, receipt_claims) =
        required_verified_upstream_session(events)?;
    let served_at = object
        .get("served_at")
        .and_then(serde_json::Value::as_u64)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let now = context.current_time()?;
    if served_at > now.saturating_add(MAX_RECEIPT_FUTURE_SKEW_SECS)
        || now.saturating_sub(served_at) > MAX_RECEIPT_AGE_SECS
        || !aci_receipt_is_current_for_keyset(&identity.keyset, served_at, now)
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let session = fetch_aci_session_by_id(client, api_base, session_id).await?;
    let session_channel_bindings = session
        .get("channel_binding")
        .ok_or(LlmError::AciResponseContractViolation)?;
    let session_expires = session
        .get("expires_at")
        .and_then(serde_json::Value::as_u64)
        .ok_or(LlmError::AciResponseContractViolation)?;
    let session_established = session
        .get("established_at")
        .and_then(serde_json::Value::as_u64)
        .ok_or(LlmError::AciResponseContractViolation)?;
    if !(session_established <= served_at && served_at < session_expires)
        || receipt_channel_bindings != session_channel_bindings
        || session.get("claims") != Some(receipt_claims)
        || !verify_aci_session(&session, session_id, served_at)
    {
        return Err(LlmError::AciResponseContractViolation);
    }
    let after_session_fetch = context.current_time()?;
    if !aci_keyset_accepts_request(&identity.keyset, after_session_fetch) {
        return Err(LlmError::AciResponseContractViolation);
    }
    // [PHALA-ACI-SESSION-OWNERSHIP 2026-10-06 by Codex]
    let session_id = session_id.to_owned();
    Ok((receipt, session_id, session))
}

// [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex]
fn required_verified_upstream_session(
    events: &[serde_json::Value],
) -> Result<(&str, &serde_json::Value, &serde_json::Value), LlmError> {
    let mut serving_session = None;
    for event in events.iter().filter(|event| {
        event.get("type").and_then(serde_json::Value::as_str) == Some("upstream.verified")
    }) {
        if event.get("required").and_then(serde_json::Value::as_bool) != Some(true)
            || event
                .get("model_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(LlmError::AciResponseContractViolation);
        }
        match event.get("result").and_then(serde_json::Value::as_str) {
            Some("verified") => {
                let session_id = event
                    .get("session_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| is_aci_session_id(id))
                    .ok_or(LlmError::AciResponseContractViolation)?;
                let bindings = event
                    .get("channel_bindings")
                    .filter(|bindings| has_enforceable_aci_channel_binding(bindings))
                    .ok_or(LlmError::AciResponseContractViolation)?;
                let claims = event
                    .get("claims")
                    .filter(|claims| claims.is_object())
                    .ok_or(LlmError::AciResponseContractViolation)?;
                if serving_session
                    .replace((session_id, bindings, claims))
                    .is_some()
                {
                    return Err(LlmError::AciResponseContractViolation);
                }
            }
            Some("failed") => {
                if event
                    .get("reason")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                    || event.get("session_id").is_some()
                {
                    return Err(LlmError::AciResponseContractViolation);
                }
            }
            _ => return Err(LlmError::AciResponseContractViolation),
        }
    }
    serving_session.ok_or(LlmError::AciResponseContractViolation)
}

// [MEMCHAIN-PHALA-ACI-UPSTREAM-BINDING 2026-10-06 by Codex]
#[cfg(test)]
fn required_verified_upstream_session_id(
    events: &[serde_json::Value],
) -> Result<&str, LlmError> {
    required_verified_upstream_session(events).map(|(session_id, _, _)| session_id)
}

// [MEMCHAIN-PHALA-ACI-SESSION 2026-10-05 by Codex]
async fn fetch_aci_session_by_id(
    client: &reqwest::Client,
    api_base: &str,
    session_id: &str,
) -> Result<serde_json::Value, LlmError> {
    if !is_aci_session_id(session_id) {
        return Err(LlmError::AciResponseContractViolation);
    }
    let session_url = aci_v1_resource_url(api_base, &["sessions", session_id])?;
    let response = client
        .get(session_url)
        .send()
        .await
        .map_err(|error| LlmError::Transport(error.to_string()))?;
    if !response.status().is_success() {
        return Err(LlmError::AciResponseContractViolation);
    }
    let bytes = read_bounded_llm_response(response, 1024 * 1024).await?;
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Session claims
    // must have one interpretation before matching the signed receipt events.
    super::validate_phala_json(&bytes, 1024 * 1024)
        .map_err(|_| LlmError::AciResponseContractViolation)?;
    serde_json::from_slice(&bytes).map_err(|_| LlmError::AciResponseContractViolation)
}

/// Resolve root, versioned-base, and full-endpoint configurations uniformly.
///
/// [LLM-OPENAI-ENDPOINT 2026-07-30 by Codex] Project examples historically
/// use both `https://host` and `https://host/v1`. Appending `/v1` blindly made
/// the latter call `/v1/v1/chat/completions`.
fn chat_completions_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with("/chat/completions") {
        api_base.to_owned()
    } else if api_base.ends_with("/v1") {
        format!("{api_base}/chat/completions")
    } else {
        format!("{api_base}/v1/chat/completions")
    }
}

// [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
fn embeddings_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with("/embeddings") {
        api_base.to_owned()
    } else if api_base.ends_with("/v1") {
        format!("{api_base}/embeddings")
    } else {
        format!("{api_base}/v1/embeddings")
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiCompatProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        if req.require_aci_verified && !self.require_aci_verified {
            return Err(LlmError::ConfidentialServingRequired);
        }
        let aci_required = self.require_aci_verified || req.require_aci_verified;

        // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Generic calls
        // have no ACI preflight to preserve; reject before copying/serializing
        // their otherwise unbounded legacy messages.
        if !aci_required {
            require_source_e2ee_transport()?;
        }

        let model = req.model_override.as_deref().unwrap_or(&self.model);
        let max_tokens = req.max_tokens.or(self.max_tokens);
        let temperature = req.temperature.or(self.temperature);

        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] A direct
        // provider caller gets the same preflight as the router/worker.
        if aci_required {
            req.validate_aci_bounds()?;
            // Request overrides were checked above. Remaining invalid values
            // are operator defaults, not grounds to permanently reject a task.
            validate_aci_request_options(model, max_tokens, temperature)
                .map_err(|_| LlmError::AciVerifierUnavailable)?;
        }

        let messages: Vec<OpenAiMessage> = req
            .messages
            .iter()
            .map(|m| OpenAiMessage {
                role: &m.role,
                content: &m.content,
            })
            .collect();

        let body = OpenAiRequest {
            model,
            messages,
            max_tokens,
            temperature,
            stop: req.stop.as_deref(),
            provider: aci_required.then_some(OpenAiProviderConstraints {
                aci_verified: true,
            }),
        };

        let url = chat_completions_url(&self.api_base);
        let request_body_bytes = if aci_required {
            serialize_bounded_aci_request(&body)?
        } else {
            serde_json::to_vec(&body).map_err(|_| LlmError::AciResponseContractViolation)?
        };

        // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] A direct trait
        // caller must not bypass the router's hold. Retain bounded preflight
        // diagnostics, but acquire no appraisal permit and perform no IO.
        require_source_e2ee_transport()?;

        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Only a fully
        // bounded payload may acquire appraisal capacity or start network IO.
        // Upstream verification remains the attested gateway's assertion.
        let phala = if aci_required {
            Some(establish_phala_identity(
                &self.client, &self.api_base, &self.accepted_compose_hashes,
                &self.accepted_source_provenance, &self.accepted_kms_root_public_keys,
            ).await?)
        } else {
            None
        };
        let start = Instant::now();

        if let Some(context) = phala.as_ref() {
            context.current_time()?;
        }

        let client = phala
            .as_ref()
            .map_or(&self.client, |context| &context.client);
        let mut request_builder = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(request_body_bytes.clone());
        // [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] Do not rely
        // on a gateway default or product-specific body routing semantics.
        if aci_required {
            request_builder = with_required_aci_upstream(request_builder);
        }

        // Only add Authorization if api_key is non-empty
        if !self.api_key.is_empty() {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let resp = request_builder.send().await.map_err(|e| {
            self.health.mark_unhealthy();
            LlmError::Transport(e.to_string())
        })?;

        let status = resp.status().as_u16();
        let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        if status == 429 {
            // Rate limit — extract Retry-After header if present
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            self.health.mark_rate_limited(retry_after);
            warn!(provider = %self.name, retry_after = ?retry_after, "[LLM_OPENAI] Rate limited");
            return Err(LlmError::RateLimit {
                retry_after_secs: retry_after,
            });
        }

        if !resp.status().is_success() {
            let body_bytes = read_bounded_llm_response(resp, MAX_LLM_ERROR_BODY_BYTES)
                .await
                .map_err(|error| {
                    self.health.mark_unhealthy();
                    error
                })?;
            // Try to parse structured error
            let msg = serde_json::from_slice::<OpenAiErrorResponse>(&body_bytes).map_or_else(
                |_| String::from_utf8_lossy(&body_bytes).into_owned(),
                |response| response.error.message,
            );
            self.health.mark_unhealthy();
            return Err(LlmError::ApiError { status, body: msg });
        }

        // [MEMCHAIN-PHALA-ACI-HEADERS 2026-10-05 by Codex] Enforce the ACI/1
        // header contract, but retain these values only as unauthenticated hints.
        let aci_response_hints = if aci_required {
            match AciResponseHints::from_headers(resp.headers()) {
                Ok(hints) => Some(hints),
                Err(error) => {
                    self.health.mark_unhealthy();
                    return Err(error);
                }
            }
        } else {
            None
        };

        let body_bytes = read_bounded_llm_response(resp, MAX_LLM_SUCCESS_BODY_BYTES)
            .await
            .map_err(|error| {
                self.health.mark_unhealthy();
                error
            })?;
        let aci_verification = if let (Some(context), Some(hints)) =
            (phala.as_ref(), aci_response_hints.as_ref())
        {
            let (receipt, upstream_session_id, upstream_session) = fetch_and_verify_aci_receipt(
                &context.client,
                &self.api_base,
                &self.api_key,
                hints,
                context,
                reqwest::Url::parse(&url)
                    .map_err(|_| LlmError::AciResponseContractViolation)?
                    .path(),
                model,
                &request_body_bytes,
                &body_bytes,
            )
            .await?;
            Some(super::llm_provider::AciVerificationEvidence {
                // [PHALA-133171-PROFILE 2026-10-08 by Codex]
                wire_profile: super::llm_provider::PHALA_ACI_WIRE_PROFILE.into(),
                key_custody_scope: "identity_kms_operational_measured_code".into(),
                // [PHALA-133171-PROFILE 2026-10-08 by Codex] Stable ID.
                workload_id: context.identity.report.workload_id.clone(),
                keyset_digest: context.identity.report.workload_keyset_digest.clone(),
                compose_hash: context.identity.compose_hash.clone(),
                kms_root_public_key: context.identity.kms_root_public_key.clone(),
                receipt_id: hints.receipt_id.clone(),
                upstream_session_id,
                upstream_session,
                // The client verifies the gateway's attested identity and
                // signed assertion, but does not independently re-run the
                // upstream provider's TEE evidence verifier.
                // [MEMCHAIN-PHALA-UPSTREAM-SCOPE 2026-10-06 by Codex]
                upstream_claim_scope: "gateway_assertion".into(),
                request_body_sha256: format!(
                    "sha256:{}",
                    hex::encode(Sha256::digest(&request_body_bytes))
                ),
                response_body_sha256: format!(
                    "sha256:{}",
                    hex::encode(Sha256::digest(&body_bytes))
                ),
                attestation_report: context.identity.report_json.clone(),
                receipt,
                upstream_verified_required: true,
            })
        } else {
            None
        };
        let resp_json: OpenAiResponse = serde_json::from_slice(&body_bytes).map_err(|error| {
            self.health.mark_unhealthy();
            LlmError::ParseError(error.to_string())
        })?;

        let content = resp_json
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .unwrap_or_default();

        if content.is_empty() {
            self.health.mark_unhealthy();
            return Err(LlmError::EmptyResponse);
        }

        let usage = resp_json.usage.unwrap_or_default();
        let cached = usage
            .prompt_tokens_details
            .map_or(0, |details| details.cached_tokens);

        let model_used = resp_json.model.unwrap_or_else(|| model.to_string());

        debug!(
            provider = %self.name,
            model = %model_used,
            input = usage.prompt_tokens,
            output = usage.completion_tokens,
            latency_ms = latency_ms,
            "[LLM_OPENAI] Call complete"
        );

        // Reset health on successful call
        if let Some(context) = phala.as_ref() { context.current_time()?; }
        self.health.mark_healthy();

        Ok(ChatResponse {
            content,
            usage: TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cached_tokens: cached,
            },
            model_used,
            provider_name: self.name.clone(),
            latency_ms,
            aci_response_hints,
            aci_verification,
        })
    }

    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    async fn embed(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        const MAX_INPUTS: usize = 32;
        const MAX_INPUT_BYTES: usize = 16 * 1024;
        const MAX_VECTOR_DIMENSIONS: usize = 16_384;
        if !self.require_aci_verified
            || req.input.is_empty()
            || req.input.len() > MAX_INPUTS
            || req.model.trim().is_empty()
            || req.model.len() > 256
            || req.input.iter().any(|text| text.is_empty() || text.len() > MAX_INPUT_BYTES)
        {
            return Err(LlmError::ConfidentialServingRequired);
        }
        let body = OpenAiEmbeddingRequest {
            model: &req.model,
            input: &req.input,
            provider: OpenAiProviderConstraints { aci_verified: true },
        };
        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Count exact
        // escaped JSON, not only the already bounded raw embedding strings.
        let request_body_bytes = serialize_bounded_aci_request(&body)?;
        // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Embedding text
        // and the returned semantic vectors belong to the source, not a node.
        require_source_e2ee_transport()?;
        let context = establish_phala_identity(
            &self.client, &self.api_base, &self.accepted_compose_hashes,
            &self.accepted_source_provenance, &self.accepted_kms_root_public_keys,
        ).await?;
        let client = &context.client;
        let identity = &context.identity;
        context.current_time()?;
        let url = embeddings_url(&self.api_base);
        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(request_body_bytes.clone());
        // [PHALA-ACI-UPSTREAM-HEADER 2026-10-08 by Codex] Embeddings
        // have the same required upstream policy as confidential chat.
        request = with_required_aci_upstream(request);
        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }
        let response = request.send().await.map_err(|error| {
            self.health.mark_unhealthy();
            LlmError::Transport(error.to_string())
        })?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            if status == 429 {
                let retry_after = response.headers().get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok());
                self.health.mark_rate_limited(retry_after);
                return Err(LlmError::RateLimit { retry_after_secs: retry_after });
            }
            let bytes = read_bounded_llm_response(response, MAX_LLM_ERROR_BODY_BYTES).await?;
            self.health.mark_unhealthy();
            return Err(LlmError::ApiError {
                status,
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        let hints = AciResponseHints::from_headers(response.headers())?;
        let response_bytes = read_bounded_llm_response(response, MAX_LLM_SUCCESS_BODY_BYTES).await?;
        let endpoint = reqwest::Url::parse(&url)
            .map_err(|_| LlmError::AciResponseContractViolation)?
            .path()
            .to_owned();
        let (receipt, upstream_session_id, upstream_session) = fetch_and_verify_aci_receipt(
            client,
            &self.api_base,
            &self.api_key,
            &hints,
            &context,
            &endpoint,
            &req.model,
            &request_body_bytes,
            &response_bytes,
        )
        .await?;
        let parsed: OpenAiEmbeddingResponse = serde_json::from_slice(&response_bytes)
            .map_err(|_| LlmError::AciResponseContractViolation)?;
        if parsed.model != req.model || parsed.data.len() != req.input.len() {
            return Err(LlmError::AciResponseContractViolation);
        }
        let mut indexed: Vec<Option<Vec<f32>>> = vec![None; req.input.len()];
        for item in parsed.data {
            if item.index >= indexed.len()
                || item.embedding.is_empty()
                || item.embedding.len() > MAX_VECTOR_DIMENSIONS
                || item.embedding.iter().any(|value| !value.is_finite())
                || indexed[item.index].replace(item.embedding).is_some()
            {
                return Err(LlmError::AciResponseContractViolation);
            }
        }
        let embeddings = indexed
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(LlmError::AciResponseContractViolation)?;
        let dimensions = embeddings.first().map(Vec::len)
            .ok_or(LlmError::AciResponseContractViolation)?;
        if embeddings.iter().any(|embedding| embedding.len() != dimensions) {
            return Err(LlmError::AciResponseContractViolation);
        }
        context.current_time()?;
        self.health.mark_healthy();
        Ok(EmbeddingResponse {
            embeddings,
            model_used: parsed.model,
            aci_response_hints: Some(hints),
            aci_verification: Some(super::llm_provider::AciVerificationEvidence {
                // [PHALA-133171-PROFILE 2026-10-08 by Codex]
                wire_profile: super::llm_provider::PHALA_ACI_WIRE_PROFILE.into(),
                key_custody_scope: "identity_kms_operational_measured_code".into(),
                // [PHALA-133171-PROFILE 2026-10-08 by Codex] Stable ID.
                workload_id: identity.report.workload_id.clone(),
                keyset_digest: identity.report.workload_keyset_digest.clone(),
                compose_hash: identity.compose_hash.clone(),
                kms_root_public_key: identity.kms_root_public_key.clone(),
                receipt_id: receipt["receipt_id"].as_str().unwrap_or_default().to_owned(),
                upstream_session_id,
                upstream_session,
                upstream_claim_scope: "gateway_assertion".into(),
                request_body_sha256: format!("sha256:{}", hex::encode(Sha256::digest(&request_body_bytes))),
                response_body_sha256: format!("sha256:{}", hex::encode(Sha256::digest(&response_bytes))),
                attestation_report: identity.report_json.clone(),
                receipt,
                upstream_verified_required: true,
            }),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn is_healthy(&self) -> bool {
        self.health.is_healthy()
    }

    fn supports_aci_verified(&self) -> bool {
        self.require_aci_verified
    }

    fn accepts_aci_compose_hash(&self, compose_hash: &str) -> bool {
        self.require_aci_verified && self.accepted_compose_hashes.contains(compose_hash)
    }

    // [MEMCHAIN-PHALA-KMS-POLICY 2026-10-05 by Codex]
    fn accepts_aci_kms_root(&self, root_public_key: &str) -> bool {
        self.require_aci_verified && self.accepted_kms_root_public_keys.contains(root_public_key)
    }

    // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
    fn accepts_aci_source_provenance(
        &self,
        compose_hash: &str,
        provenance: &serde_json::Value,
    ) -> bool {
        if !self.require_aci_verified {
            return false;
        }
        let Ok(provenance) = serde_json::from_value::<aci_protocol::types::SourceProvenance>(
            provenance.clone(),
        ) else {
            return false;
        };
        has_acceptable_aci_source_provenance(
            &provenance,
            compose_hash,
            &self.accepted_source_provenance,
        )
    }

    // [MEMCHAIN-PHALA-ACI-REPORT-VERIFY 2026-10-05 by Codex]
    fn has_cryptographic_aci_verifier(&self) -> bool {
        self.require_aci_verified
            && !self.accepted_compose_hashes.is_empty()
            && !self.accepted_source_provenance.is_empty()
            && !self.accepted_kms_root_public_keys.is_empty()
    }
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("name", &self.name)
            .field("api_base", &self.api_base)
            .field("model", &self.model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Synthetic
    // binding fixtures only, not successful hardware/appraisal evidence.
    fn synthetic_aci_context(not_after: u64) -> VerifiedPhalaContext {
        // [PHALA-133171-PROFILE 2026-10-08 by Codex] Binding-only fixture;
        // the dummy endorsement never passes the hardware appraisal boundary.
        let nonce = "1a".repeat(32);
        let keyset = serde_json::json!({
            "workload_identity": {"public_key": {"algo": "ed25519", "public_key": "cd".repeat(32)}, "subject": null},
            "keyset_epoch": {"version": 1, "not_after": not_after},
            "receipt_signing_keys": [{"key_id": "test", "algo": "ed25519", "public_key": "ab".repeat(32)}],
            "e2ee_public_keys": [], "tls_public_keys": [],
        });
        let hash = |value: &serde_json::Value| format!("sha256:{}", hex::encode(Sha256::digest(aci_protocol::digest::jcs_bytes(value).unwrap())));
        let digest = hash(&keyset);
        let workload_id = hash(&keyset["workload_identity"]["public_key"]);
        let statement = serde_json::json!({"purpose": "aci.report_data.v1", "nonce": nonce,
            "workload_id": workload_id, "workload_keyset_digest": digest});
        let report = AttestationReport {
            api_version: "aci/1".into(), workload_id, workload_keyset_digest: digest,
            attestation: PhalaAttestationEnvelope {
                tee_type: "tdx".into(), workload_keyset: keyset.clone(),
                report_data_hex: hex::encode(Sha256::digest(aci_protocol::digest::jcs_bytes(&statement).unwrap())),
                keyset_endorsement: serde_json::json!({"algo": "ed25519", "value": "00".repeat(64)}),
                source_provenance: Default::default(), evidence: serde_json::json!({}),
            },
            service_capabilities: Default::default(),
        };
        VerifiedPhalaContext {
            identity: VerifiedPhalaIdentity {
                report_json: serde_json::to_value(&report).unwrap(), report,
                keyset: WorkloadKeyset { subject: None, not_after,
                    receipt_signing_keys: serde_json::from_value(keyset["receipt_signing_keys"].clone()).unwrap(),
                    e2ee_public_keys: Vec::new(), tls_public_keys: Vec::new() },
                compose_hash: String::new(), kms_root_public_key: String::new(), tls_pins: Vec::new(),
            },
            client: reqwest::Client::new(), observation: PhalaAciObservation::begin().unwrap(), nonce,
        }
    }

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Recompute only public
    // candidate hashes for mutations; never sign, appraise, or mint authority.
    fn refresh_synthetic_report_binding(report: &mut AttestationReport, nonce: &str) {
        let hash = |value: &serde_json::Value| format!("sha256:{}",
            hex::encode(Sha256::digest(aci_protocol::digest::jcs_bytes(value).unwrap())));
        report.workload_id = hash(&report.attestation.workload_keyset["workload_identity"]["public_key"]);
        report.workload_keyset_digest = hash(&report.attestation.workload_keyset);
        report.attestation.report_data_hex = hex::encode(Sha256::digest(aci_protocol::digest::jcs_bytes(
            &serde_json::json!({"purpose": "aci.report_data.v1", "nonce": nonce,
                "workload_id": report.workload_id, "workload_keyset_digest": report.workload_keyset_digest})
        ).unwrap()));
    }

    #[test]
    fn profile_133171_binding_preserves_identity_across_epoch_change_and_refuses_a991_shape() {
        let context = synthetic_aci_context(100);
        let mut rotated = synthetic_aci_context(200);
        rotated.identity.report.attestation.workload_keyset["keyset_epoch"]["version"] = serde_json::json!(2);
        refresh_synthetic_report_binding(&mut rotated.identity.report, &rotated.nonce);
        assert!(bind_phala_report(&context.identity.report, &context.nonce, 1).is_ok());
        assert!(bind_phala_report(&rotated.identity.report, &rotated.nonce, 1).is_ok());
        assert_ne!(context.identity.report.workload_id, context.identity.report.workload_keyset_digest);
        assert_eq!(context.identity.report.workload_id, rotated.identity.report.workload_id);
        assert_ne!(context.identity.report.workload_keyset_digest, rotated.identity.report.workload_keyset_digest);

        let mut old_statement = synthetic_aci_context(100);
        let statement = serde_json::json!({"purpose": "aci.report_data.v1", "nonce": old_statement.nonce,
            "keyset_digest": old_statement.identity.report.workload_keyset_digest});
        old_statement.identity.report.attestation.report_data_hex = hex::encode(Sha256::digest(
            aci_protocol::digest::jcs_bytes(&statement).unwrap()));
        assert!(bind_phala_report(&old_statement.identity.report, &old_statement.nonce, 1).is_err());
        let mut alias = synthetic_aci_context(100);
        alias.identity.report.workload_id = alias.identity.report.workload_keyset_digest.clone();
        assert!(bind_phala_report(&alias.identity.report, &alias.nonce, 1).is_err());
        let mut old_keyset = synthetic_aci_context(100);
        old_keyset.identity.report.attestation.workload_keyset = serde_json::to_value(&old_keyset.identity.keyset).unwrap();
        refresh_synthetic_report_binding(&mut old_keyset.identity.report, &old_keyset.nonce);
        assert!(bind_phala_report(&old_keyset.identity.report, &old_keyset.nonce, 1).is_err());
        for missing in ["workload_id", "keyset_endorsement"] {
            let mut report = context.identity.report_json.clone();
            if missing == "workload_id" { report.as_object_mut().unwrap().remove(missing); }
            else { report["attestation"].as_object_mut().unwrap().remove(missing); }
            assert!(serde_json::from_value::<AttestationReport>(report).is_err());
        }
    }

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Exact public vectors
    // from spec/test-vectors.md at 133171ef..., authored but not executed.
    // Placeholder E2EE/TLS entries are digest inputs, not hardware evidence.
    #[test]
    fn profile_133171_published_identity_keyset_statement_and_endorsement_vectors() {
        let mut context = synthetic_aci_context(1_800_000_000);
        let identity_public = "8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c";
        context.identity.report.attestation.workload_keyset = serde_json::json!({
            "workload_identity": {"public_key": {"algo": "ed25519", "public_key": identity_public}, "subject": null},
            "keyset_epoch": {"version": 1, "not_after": 1_800_000_000},
            "receipt_signing_keys": [{"key_id": "receipt-1", "algo": "ed25519",
                "public_key": "8139770ea87d175f56a35466c34c7ecccb8d8a91b4ee37a25df60f5b8fc9b394"}],
            "e2ee_public_keys": [{"key_id": "e2ee-1", "algo": "x25519-aes-256-gcm-hkdf-sha256", "public_key": "ab".repeat(32)}],
            "tls_public_keys": [{"spki_sha256": "c0".repeat(32), "domain": "api.example.com"}],
        });
        refresh_synthetic_report_binding(&mut context.identity.report, "test-nonce");
        assert_eq!(context.identity.report.workload_id, "sha256:57c2c8fa98bcf11441f1eff9ef087db67a5560a026082e96903e15365677b8c0");
        assert_eq!(context.identity.report.workload_keyset_digest, "sha256:f2fba7e1b1451e0c0231df624f293407692ef939d3e0e55bca723131bea3f1ff");
        assert_eq!(context.identity.report.attestation.report_data_hex, "0b8cc28d7e989a88b1e969af20aa2b224afdc2c99f24c97c31a4af330c964ecf");
        let null_statement = serde_json::json!({"purpose": "aci.report_data.v1", "nonce": null,
            "workload_id": context.identity.report.workload_id,
            "workload_keyset_digest": context.identity.report.workload_keyset_digest});
        assert_eq!(hex::encode(Sha256::digest(aci_protocol::digest::jcs_bytes(&null_statement).unwrap())),
            "e1818eadad3c28375c625e2fa2d2ffd983d2760c84ce17f8527ddcac884c21b9");
        // The general spec vector uses a textual nonce. Our live profile only
        // accepts its own 32-byte random hex challenge, never null/replay mode.
        assert!(bind_phala_report(&context.identity.report, "test-nonce", 1).is_err());
        refresh_synthetic_report_binding(&mut context.identity.report, &context.nonce);
        assert!(bind_phala_report(&context.identity.report, &context.nonce, 1).is_ok());
        let payload = aci_protocol::digest::jcs_bytes(&serde_json::json!({
            "purpose": "aci.keyset.endorsement.v1", "workload_keyset_digest": context.identity.report.workload_keyset_digest,
        })).unwrap();
        let signature = "64e0a4f5d7af28dfdacc102d14c13470b4ddbd90708e190fc0e787f07b36f20eda0ef1f42ea96b8a7f290eb64a918574dc914ce06b6ea023d2153275f06fd201";
        assert!(verify_aci_identity_endorsement("ed25519", identity_public, signature, &payload));
        let revoked = aci_protocol::digest::jcs_bytes(&serde_json::json!({
            "purpose": "aci.keyset.revocation.v1", "workload_keyset_digest": context.identity.report.workload_keyset_digest,
        })).unwrap();
        assert!(!verify_aci_identity_endorsement("ed25519", identity_public, signature, &revoked));
    }

    #[test]
    fn profile_133171_candidate_rejects_closed_schema_role_and_epoch_mutations() {
        for (pointer, value) in [
            ("/workload_identity/public_key/algo", serde_json::json!("ecdsa")),
            ("/workload_identity/subject", serde_json::json!(1)),
            ("/keyset_epoch/version", serde_json::json!(-1)),
            ("/keyset_epoch/version", serde_json::json!(9_007_199_254_740_992u64)),
            ("/keyset_epoch/not_after", serde_json::json!(9_007_199_254_740_992u64)),
            ("/receipt_signing_keys/0/key_id", serde_json::json!("")),
            ("/receipt_signing_keys/0/public_key", serde_json::json!(null)),
        ] {
            let mut context = synthetic_aci_context(100);
            *context.identity.report.attestation.workload_keyset.pointer_mut(pointer).unwrap() = value;
            refresh_synthetic_report_binding(&mut context.identity.report, &context.nonce);
            assert!(bind_phala_report(&context.identity.report, &context.nonce, 1).is_err(), "{pointer}");
        }
        for pointer in ["", "/workload_identity", "/workload_identity/public_key", "/keyset_epoch", "/receipt_signing_keys/0"] {
            let mut context = synthetic_aci_context(100);
            context.identity.report.attestation.workload_keyset.pointer_mut(pointer).unwrap()
                .as_object_mut().unwrap().insert("unknown".into(), serde_json::json!(true));
            refresh_synthetic_report_binding(&mut context.identity.report, &context.nonce);
            assert!(bind_phala_report(&context.identity.report, &context.nonce, 1).is_err(), "{pointer}");
        }
        let mut duplicate = synthetic_aci_context(100);
        let entry = duplicate.identity.report.attestation.workload_keyset["receipt_signing_keys"][0].clone();
        duplicate.identity.report.attestation.workload_keyset["receipt_signing_keys"].as_array_mut().unwrap().push(entry.clone());
        refresh_synthetic_report_binding(&mut duplicate.identity.report, &duplicate.nonce);
        assert!(bind_phala_report(&duplicate.identity.report, &duplicate.nonce, 1).is_err());
        let mut cross_role = synthetic_aci_context(100);
        cross_role.identity.report.attestation.workload_keyset["e2ee_public_keys"] = serde_json::json!([entry]);
        refresh_synthetic_report_binding(&mut cross_role.identity.report, &cross_role.nonce);
        assert!(bind_phala_report(&cross_role.identity.report, &cross_role.nonce, 1).is_err());
        let valid = synthetic_aci_context(100);
        assert!(bind_phala_report(&valid.identity.report, &valid.nonce, 99).is_ok());
        assert!(bind_phala_report(&valid.identity.report, &valid.nonce, 100).is_err());
        assert!(bind_phala_report(&valid.identity.report, &valid.nonce, 101).is_err());
        for nonce in ["", "null", "test-nonce"] {
            assert!(bind_phala_report(&valid.identity.report, nonce, 1).is_err());
        }
    }

    #[test]
    fn aci_context_rechecks_original_nonce_and_keyset_at_acceptance() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let mut valid = synthetic_aci_context(now + 3_600);
        assert!(valid.current_time().is_ok());
        valid.nonce = "2b".repeat(32);
        assert!(matches!(valid.current_time(), Err(LlmError::AciResponseContractViolation)));
        let expired = synthetic_aci_context(now);
        assert!(matches!(expired.current_time(), Err(LlmError::AciResponseContractViolation)));
        let mut digest_mismatch = synthetic_aci_context(now + 3_600);
        digest_mismatch.identity.report.workload_keyset_digest = format!("sha256:{}", "00".repeat(32));
        assert!(matches!(digest_mismatch.current_time(), Err(LlmError::AciResponseContractViolation)));
    }

    #[test]
    fn aci_keyset_bounds_precede_cross_role_comparisons() {
        let mut context = synthetic_aci_context(9_007_199_254_740_991);
        assert!(validate_aci_keyset_resource_bounds(&context.identity.report).is_ok());
        for role in ["receipt_signing_keys", "e2ee_public_keys", "tls_public_keys"] {
            let saved = context.identity.report.attestation.workload_keyset[role].clone();
            context.identity.report.attestation.workload_keyset[role] = serde_json::json!(vec![serde_json::json!({}); 64]);
            assert!(validate_aci_keyset_resource_bounds(&context.identity.report).is_ok());
            context.identity.report.attestation.workload_keyset[role] = serde_json::json!(vec![serde_json::json!({}); 65]);
            assert!(validate_aci_keyset_resource_bounds(&context.identity.report).is_err());
            context.identity.report.attestation.workload_keyset[role] = serde_json::Value::Null;
            assert!(validate_aci_keyset_resource_bounds(&context.identity.report).is_err());
            context.identity.report.attestation.workload_keyset[role] = saved;
        }
        context.identity.report.attestation.workload_keyset.as_object_mut().unwrap().remove("tls_public_keys");
        assert!(validate_aci_keyset_resource_bounds(&context.identity.report).is_ok());
    }

    // [MEMCHAIN-PHALA-OFFLINE-COLLATERAL 2026-10-06 by Codex]
    // Authored to exercise the no-network verifier boundary; not executed.
    #[test]
    fn collateral_injected_verifier_checks_signed_subtrees_and_malformed_reports() {
        let collateral = dcap_qvl::QuoteCollateralV3 {
            pck_crl_issuer_chain: String::new(),
            root_ca_crl: Vec::new(),
            pck_crl: Vec::new(),
            tcb_info_issuer_chain: String::new(),
            tcb_info: "{}".to_owned(),
            tcb_info_signature: vec![0; 64],
            qe_identity_issuer_chain: String::new(),
            qe_identity: "{}".to_owned(),
            qe_identity_signature: vec![0; 64],
            pck_certificate_chain: None,
        };
        // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] This minimal
        // fixture passes framing only; zero signatures/empty chains are not
        // trusted. Check both injected strings, not just the network parser.
        assert!(crate::api::discovery::validate_phala_collateral_signed_inputs(&collateral).is_ok());
        for tcb in [true, false] {
            let mut invalid = collateral.clone();
            if tcb { invalid.tcb_info = r#"{"x":"\u0008"}"#.to_owned(); }
            else { invalid.qe_identity = r#"{"x":1e2}"#.to_owned(); }
            assert_eq!(crate::api::discovery::validate_phala_collateral_signed_inputs(&invalid), Err("peer_collateral_unsupported_form"));
            let mut invalid = collateral.clone();
            if tcb { invalid.tcb_info_signature.pop(); }
            else { invalid.qe_identity_signature.push(0); }
            assert_eq!(crate::api::discovery::validate_phala_collateral_signed_inputs(&invalid), Err("peer_collateral_malformed"));
        }
        let empty_policy = HashSet::new();

        assert!(matches!(
            verify_phala_identity_with_collateral(
                b"not-json",
                &"00".repeat(32),
                &empty_policy,
                &HashSet::new(),
                &empty_policy,
                &collateral,
                1_760_000_000,
            ),
            Err(LlmError::AciResponseContractViolation)
        ));
    }

    // [MEMCHAIN-PHALA-KEYSET-EXPIRY 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_keyset_expiry_is_exclusive_at_not_after() {
        let keyset = WorkloadKeyset {
            subject: None,
            not_after: 100,
            receipt_signing_keys: Vec::new(),
            e2ee_public_keys: Vec::new(),
            tls_public_keys: Vec::new(),
        };
        assert!(aci_keyset_accepts_request(&keyset, 99));
        assert!(!aci_keyset_accepts_request(&keyset, 100));
        assert!(!aci_keyset_accepts_request(&keyset, 101));
    }

    // [MEMCHAIN-PHALA-KEYSET-EXPIRY 2026-10-07 by Codex] Authored, not
    // executed: reject a receipt if either its serving time or verification
    // time reaches the keyset's exclusive expiry boundary.
    #[test]
    fn aci_receipt_and_verification_must_precede_keyset_expiry() {
        let keyset = WorkloadKeyset {
            subject: None,
            not_after: 100,
            receipt_signing_keys: Vec::new(),
            e2ee_public_keys: Vec::new(),
            tls_public_keys: Vec::new(),
        };
        assert!(aci_receipt_is_current_for_keyset(&keyset, 99, 99));
        assert!(!aci_receipt_is_current_for_keyset(&keyset, 100, 99));
        assert!(!aci_receipt_is_current_for_keyset(&keyset, 99, 100));
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_session_endpoint_is_null_or_an_https_origin() {
        assert!(is_aci_endpoint(&serde_json::Value::Null));
        assert!(is_aci_endpoint(&serde_json::json!("https://upstream.example.com")));
        assert!(!is_aci_endpoint(&serde_json::json!({
            "origin": "https://upstream.example.com"
        })));
        assert!(!is_aci_endpoint(&serde_json::json!("https://upstream.example.com/path")));
        assert!(!is_aci_endpoint(&serde_json::json!("http://upstream.example.com")));
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
    #[test]
    fn aci_source_provenance_requires_a_complete_repo_or_image_identity() {
        let compose_hash = format!("sha256:{}", "c".repeat(64));
        let accepted = HashSet::from([AcceptedAciSourceProvenance {
            compose_hash: compose_hash.clone(),
            repo_url: Some("https://github.com/example/private-ai-gateway.git".into()),
            repo_commit: Some("a".repeat(40)),
            image_digest: None,
        }]);
        let repo_revision = aci_protocol::types::SourceProvenance {
            repo_url: Some("https://github.com/example/private-ai-gateway.git".into()),
            repo_commit: Some("a".repeat(40)),
            image_digest: None,
            image_provenance: None,
        };
        assert!(has_acceptable_aci_source_provenance(
            &repo_revision,
            &compose_hash,
            &accepted,
        ));

        let repo_without_revision = aci_protocol::types::SourceProvenance {
            repo_commit: None,
            ..repo_revision.clone()
        };
        assert!(!has_acceptable_aci_source_provenance(
            &repo_without_revision,
            &compose_hash,
            &accepted,
        ));
        let different_compose = format!("sha256:{}", "d".repeat(64));
        assert!(!has_acceptable_aci_source_provenance(
            &repo_revision,
            &different_compose,
            &accepted,
        ));
        let unreviewed_revision = aci_protocol::types::SourceProvenance {
            repo_commit: Some("b".repeat(40)),
            ..repo_revision.clone()
        };
        assert!(!has_acceptable_aci_source_provenance(
            &unreviewed_revision,
            &compose_hash,
            &accepted,
        ));

        let image_digest = aci_protocol::types::SourceProvenance {
            repo_url: None,
            repo_commit: None,
            image_digest: Some(format!("sha256:{}", "b".repeat(64))),
            image_provenance: None,
        };
        let accepted_image = HashSet::from([AcceptedAciSourceProvenance {
            compose_hash: compose_hash.clone(),
            repo_url: None,
            repo_commit: None,
            image_digest: Some(format!("sha256:{}", "b".repeat(64))),
        }]);
        assert!(has_acceptable_aci_source_provenance(
            &image_digest,
            &compose_hash,
            &accepted_image,
        ));
        assert!(!has_acceptable_aci_source_provenance(
            &image_digest,
            &compose_hash,
            &accepted,
        ));

        let uncorroborated_extension = aci_protocol::types::SourceProvenance {
            repo_url: None,
            repo_commit: None,
            image_digest: None,
            image_provenance: Some(serde_json::json!({"source": "unbound"})),
        };
        assert!(!has_acceptable_aci_source_provenance(
            &uncorroborated_extension,
            &compose_hash,
            &accepted,
        ));
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_session_id_matches_phala_published_vector() {
        let session = serde_json::json!({
            "upstream_name": "demo-upstream",
            "endpoint": "https://upstream.example.com",
            "verifier_id": "example/1",
            "channel_binding": [{
                "type": "tls_spki_sha256",
                "origin": "https://upstream.example.com",
                "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
            }],
            "claims": {
                "tee_attested": {
                    "status": "asserted",
                    "source": "hardware_proven",
                    "reason": "example quote verified"
                },
                "tcb_up_to_date": {"status": "unknown"},
                "gpu_attested": {"status": "unknown"},
                "model_weights_provenance": {"status": "unknown"},
                "os_known_good": {"status": "unknown"},
                "serving_software_known_good": {"status": "unknown"}
            },
            "evidence": {
                "digest": "sha256:80d70e44d0ae1e829fd5f37c3ee4a60dfbea8d3aa18407ea3f34cf7ec91da34d"
            }
        });
        let material = aci_session_id_material(&session).unwrap();
        let canonical = aci_protocol::digest::jcs_bytes(&material).unwrap();
        assert_eq!(
            format!("as_{}", hex::encode(Sha256::digest(canonical))),
            "as_2e9011abafe00fc2902aaa5dedf8373f14e2c4f1a456b854ddb475413547188e"
        );
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_session_id_hashes_only_specified_immutable_material() {
        let evidence_hash = format!(
            "sha256:{}",
            hex::encode(Sha256::digest(b"example-evidence"))
        );
        let mut session = serde_json::json!({
            "api_version": "aci/1",
            "upstream_name": "demo-upstream",
            "endpoint": "https://upstream.example.com",
            "verifier_id": "example/1",
            "established_at": 1750000000,
            "expires_at": 1750003600,
            "identity": {},
            "channel_binding": [{
                "type": "tls_spki_sha256",
                "origin": "https://upstream.example.com",
                "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
            }],
            "claims": {
                "tee_attested": {"status":"asserted", "source":"hardware_proven", "reason":"example quote verified"},
                "tcb_up_to_date": {"status":"asserted", "source":"verifier_derived"},
                "gpu_attested": {"status":"unknown"},
                "model_weights_provenance": {"status":"unknown"},
                "os_known_good": {"status":"unknown"},
                "serving_software_known_good": {"status":"unknown"}
            },
            "evidence": {
                "digest": evidence_hash,
                "data": "data:application/octet-stream;base64,ZXhhbXBsZS1ldmlkZW5jZQ=="
            }
        });
        let material = serde_json::json!({
            "upstream_name": session["upstream_name"],
            "endpoint": session["endpoint"],
            "verifier_id": session["verifier_id"],
            "identity": session["identity"],
            "channel_binding": session["channel_binding"],
            "claims": session["claims"],
            "evidence_digest": session["evidence"]["digest"],
        });
        let canonical = aci_protocol::digest::jcs_bytes(&material).unwrap();
        let session_id = format!("as_{}", hex::encode(Sha256::digest(canonical)));
        assert!(is_aci_session_id(&session_id));
        assert!(!is_aci_session_id(&session_id[3..]));
        assert!(!is_aci_session_id(&format!("as_{}", "A".repeat(64))));
        session["session_id"] = serde_json::Value::String(session_id.clone());
        assert!(verify_aci_session(&session, &session_id, 1_750_000_100));

        // ACI expires_at retains the session artifact for referencing receipts.
        session["expires_at"] = serde_json::json!(1_750_000_100);
        assert!(!verify_aci_session(&session, &session_id, 1_750_000_100));
        session["expires_at"] = serde_json::json!(1_750_000_099);
        assert!(!verify_aci_session(&session, &session_id, 1_750_000_100));
        session["expires_at"] = serde_json::json!(1_750_000_360);

        session["established_at"] = serde_json::json!(1_750_000_010);
        session["expires_at"] = serde_json::json!(1_750_004_000);
        session["evidence"]["data"] = serde_json::json!(
            "data:application/x-octet-stream;base64,ZXhhbXBsZS1ldmlkZW5jZQ=="
        );
        assert!(verify_aci_session(&session, &session_id, 1_750_000_100));

        session["channel_binding"][0]["spki_sha256"] = serde_json::json!("e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2");
        assert!(!verify_aci_session(&session, &session_id, 1_750_000_100));

        let mut certificate_session = session.clone();
        let certificate_hash = "c4".repeat(32);
        certificate_session["channel_binding"][0] = serde_json::json!({
            "type": "tls_certificate_sha256",
            "origin": "https://upstream.example.com",
            "certificate_sha256": certificate_hash
        });
        let certificate_material = serde_json::json!({
            "upstream_name": certificate_session["upstream_name"],
            "endpoint": certificate_session["endpoint"],
            "verifier_id": certificate_session["verifier_id"],
            "identity": certificate_session["identity"],
            "channel_binding": certificate_session["channel_binding"],
            "claims": certificate_session["claims"],
            "evidence_digest": certificate_session["evidence"]["digest"],
        });
        let canonical = aci_protocol::digest::jcs_bytes(&certificate_material).unwrap();
        let certificate_session_id = format!("as_{}", hex::encode(Sha256::digest(canonical)));
        certificate_session["session_id"] = serde_json::json!(certificate_session_id);
        assert!(verify_aci_session(
            &certificate_session,
            certificate_session["session_id"].as_str().unwrap(),
            1_750_000_100,
        ));

        let mut optional_fields_session = session.clone();
        optional_fields_session.as_object_mut().unwrap().remove("endpoint");
        optional_fields_session.as_object_mut().unwrap().remove("identity");
        optional_fields_session["channel_binding"] = serde_json::json!([{
            "type": "e2ee_public_key_sha256",
            "provider": "upstream",
            "algorithm": "x25519-aes-256-gcm-hkdf-sha256",
            "public_key_sha256": "f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3"
        }]);
        let optional_material = serde_json::json!({
            "upstream_name": optional_fields_session["upstream_name"],
            "endpoint": serde_json::Value::Null,
            "verifier_id": optional_fields_session["verifier_id"],
            "identity": serde_json::Value::Null,
            "channel_binding": optional_fields_session["channel_binding"],
            "claims": optional_fields_session["claims"],
            "evidence_digest": optional_fields_session["evidence"]["digest"],
        });
        let canonical = aci_protocol::digest::jcs_bytes(&optional_material).unwrap();
        let optional_session_id = format!("as_{}", hex::encode(Sha256::digest(canonical)));
        optional_fields_session["session_id"] = serde_json::json!(optional_session_id);
        assert!(verify_aci_session(
            &optional_fields_session,
            optional_fields_session["session_id"].as_str().unwrap(),
            1_750_000_100,
        ));
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_session_requires_a_nonempty_verifier_identifier() {
        let evidence_hash = format!(
            "sha256:{}",
            hex::encode(Sha256::digest(b"example-evidence"))
        );
        let mut session = serde_json::json!({
            "api_version": "aci/1",
            "upstream_name": "demo-upstream",
            "endpoint": "https://upstream.example.com",
            "verifier_id": "example/1",
            "established_at": 1750000000,
            "expires_at": 1750003600,
            "identity": {},
            "channel_binding": [{
                "type": "tls_spki_sha256",
                "origin": "https://upstream.example.com",
                "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
            }],
            "claims": {
                "tee_attested": {"status":"asserted", "source":"hardware_proven"},
                "tcb_up_to_date": {"status":"asserted", "source":"verifier_derived"}
            },
            "evidence": {
                "digest": evidence_hash,
                "data": "data:application/octet-stream;base64,ZXhhbXBsZS1ldmlkZW5jZQ=="
            }
        });
        let canonical = aci_protocol::digest::jcs_bytes(&session).unwrap();
        let session_id = format!("as_{}", hex::encode(Sha256::digest(canonical)));
        assert!(verify_aci_session(&session, &session_id, 1_750_000_100));

        session["verifier_id"] = serde_json::Value::String(String::new());
        let canonical = aci_protocol::digest::jcs_bytes(&session).unwrap();
        let session_id = format!("as_{}", hex::encode(Sha256::digest(canonical)));
        assert!(!verify_aci_session(&session, &session_id, 1_750_000_100));
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn receipt_signing_projection_removes_only_signature_value() {
        let receipt = serde_json::json!({
            "api_version": "aci/1",
            "signature": {"algo": "ed25519", "key_id": "receipt-1", "value": "00"}
        });
        assert_eq!(
            aci_receipt_signing_bytes(&receipt).unwrap().as_slice(),
            br#"{"api_version":"aci/1","signature":{"algo":"ed25519","key_id":"receipt-1"}}"#
        );
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn receipt_signing_projection_matches_phala_published_vector() {
        let request_hash = "sha256:94d809bf47380d8a2eab0eb6e126d4dda9364b0b4725cdf7ead52dd70b2aa87b";
        let response_hash = "sha256:dedfffe5b14d031b8e2c01996d021a15293cb7c63b56be7e4be9e89b6f0a5f61";
        let session_id = "as_2e9011abafe00fc2902aaa5dedf8373f14e2c4f1a456b854ddb475413547188e";
        let receipt = serde_json::json!({
            "api_version": "aci/1",
            "chat_id": "chatcmpl-123",
            "endpoint": "/v1/chat/completions",
            "event_log": [
                {"body_hash": request_hash, "seq": 0, "type": "request.received"},
                {"body_hash": request_hash, "seq": 1, "type": "request.forwarded"},
                {
                    "channel_bindings": [{
                        "origin": "https://upstream.example.com",
                        "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1",
                        "type": "tls_spki_sha256"
                    }],
                    "claims": {
                        "gpu_attested": {"status": "unknown"},
                        "model_weights_provenance": {"status": "unknown"},
                        "os_known_good": {"status": "unknown"},
                        "serving_software_known_good": {"status": "unknown"},
                        "tcb_up_to_date": {"status": "unknown"},
                        "tee_attested": {
                            "reason": "example quote verified",
                            "source": "hardware_proven",
                            "status": "asserted"
                        }
                    },
                    "model_id": "demo-model",
                    "provider_claims": null,
                    "provider_type": null,
                    "reason": null,
                    "required": true,
                    "result": "verified",
                    "seq": 2,
                    "session_id": session_id,
                    "type": "upstream.verified",
                    "upstream_name": "demo-upstream",
                    "url_origin": "https://upstream.example.com",
                    "verifier_id": "example/1"
                },
                {
                    "cleartext_hash": response_hash,
                    "seq": 3,
                    "type": "response.returned",
                    "wire_hash": response_hash
                }
            ],
            "method": "POST",
            "model": "demo-model",
            "receipt_id": "rcpt-0001",
            "served_at": 1750000000,
            "signature": {
                "algo": "ed25519",
                "key_id": "receipt-1",
                "value": "ignored-by-projection"
            },
            "workload_id": "sha256:57c2c8fa98bcf11441f1eff9ef087db67a5560a026082e96903e15365677b8c0",
            "workload_keyset_digest": "sha256:f2fba7e1b1451e0c0231df624f293407692ef939d3e0e55bca723131bea3f1ff"
        });
        let canonical = aci_receipt_signing_bytes(&receipt).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(canonical)),
            "1cd5a27c330ac3a5a82ac30176ba67349b42a64b00868d75ffd23600bf7e7b7c"
        );
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex]
    #[test]
    fn receipt_signing_projection_rejects_missing_or_non_string_value() {
        for receipt in [
            serde_json::json!({"signature": {"algo": "ed25519", "key_id": "k"}}),
            serde_json::json!({"signature": {"algo": "ed25519", "key_id": "k", "value": 1}}),
        ] {
            assert!(aci_receipt_signing_bytes(&receipt).is_err());
        }
    }

    // [MEMCHAIN-PHALA-ACI-RESPONSE-TRANSPARENCY 2026-10-06 by Codex]
    // Test-only consistency checks; authored but not executed in this phase.
    #[test]
    fn response_rewrite_requires_transparency_event() {
        let returned = format!("sha256:{}", "a".repeat(64));
        let received = format!("sha256:{}", "b".repeat(64));
        let event = serde_json::json!({
            "seq": 1,
            "type": "response.received",
            "cleartext_hash": received
        });
        let events = vec![
            event.clone(),
            serde_json::json!({"seq": 2, "type": "response.returned"}),
        ];
        assert!(!response_transparency_is_consistent(&events, &returned));

        let events = vec![
            event,
            serde_json::json!({"seq": 2, "type": "transparency.response_modified"}),
            serde_json::json!({"seq": 3, "type": "response.returned"}),
        ];
        assert!(response_transparency_is_consistent(&events, &returned));
    }

    // [MEMCHAIN-PHALA-RESPONSE-ORDER 2026-10-06 by Codex]
    #[test]
    fn response_rewrite_marker_must_precede_return_and_be_unique() {
        let returned = format!("sha256:{}", "a".repeat(64));
        let received = format!("sha256:{}", "b".repeat(64));
        let received_event = serde_json::json!({
            "seq": 1,
            "type": "response.received",
            "cleartext_hash": received,
        });
        let returned_event = serde_json::json!({
            "seq": 3,
            "type": "response.returned",
        });

        assert!(!response_transparency_is_consistent(
            &[
                received_event.clone(),
                returned_event.clone(),
                serde_json::json!({"seq": 4, "type": "transparency.response_modified"}),
            ],
            &returned,
        ));
        assert!(!response_transparency_is_consistent(
            &[
                received_event.clone(),
                serde_json::json!({"seq": 2, "type": "transparency.response_modified"}),
                serde_json::json!({"seq": 2, "type": "transparency.response_modified"}),
                returned_event.clone(),
            ],
            &returned,
        ));
        assert!(!response_transparency_is_consistent(
            &[
                received_event,
                serde_json::json!({"seq": 2, "type": "transparency.response_modified"}),
                serde_json::json!({"seq": 3, "type": "transparency.response_modified"}),
                returned_event,
            ],
            &returned,
        ));
    }

    // [MEMCHAIN-PHALA-ACI-RESPONSE-TRANSPARENCY 2026-10-06 by Codex]
    #[test]
    fn unchanged_or_absent_received_response_needs_no_marker() {
        let returned = format!("sha256:{}", "a".repeat(64));
        assert!(response_transparency_is_consistent(&[
            serde_json::json!({"seq": 1, "type": "response.returned"}),
        ], &returned));
        assert!(response_transparency_is_consistent(&[
            serde_json::json!({"seq": 1, "type": "response.received", "cleartext_hash": returned}),
            serde_json::json!({"seq": 2, "type": "response.returned"}),
        ], &returned));
        assert!(!response_transparency_is_consistent(&[
            serde_json::json!({"seq": 1, "type": "transparency.response_modified"}),
            serde_json::json!({"seq": 2, "type": "response.returned"}),
        ], &returned));
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn required_route_accepts_failed_attempts_but_only_one_serving_session() {
        let events = vec![
            serde_json::json!({
                "type": "upstream.verified",
                "result": "failed",
                "required": true,
                "model_id": "model-a",
                "reason": "attestation unavailable"
            }),
            serde_json::json!({
                "type": "upstream.verified",
                "result": "verified",
                "required": true,
                "model_id": "model-b",
                "claims": {},
                "session_id": format!("as_{}", "a".repeat(64)),
                "channel_bindings": [{
                    "type": "tls_spki_sha256",
                    "origin": "https://upstream.example.com",
                    "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
                }]
            }),
        ];
        assert_eq!(
            required_verified_upstream_session_id(&events).unwrap(),
            format!("as_{}", "a".repeat(64))
        );
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn required_route_rejects_incomplete_or_ambiguous_upstream_events() {
        let missing_model = vec![serde_json::json!({
            "type": "upstream.verified",
            "result": "verified",
            "required": true,
            "session_id": format!("as_{}", "a".repeat(64))
        })];
        assert!(required_verified_upstream_session_id(&missing_model).is_err());

        let duplicate_serving = vec![
            serde_json::json!({
                "type": "upstream.verified",
                "result": "verified",
                "required": true,
                "model_id": "model-a",
                "claims": {},
                "session_id": format!("as_{}", "a".repeat(64)),
                "channel_bindings": [{
                    "type": "tls_spki_sha256",
                    "origin": "https://upstream.example.com",
                    "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
                }]
            }),
            serde_json::json!({
                "type": "upstream.verified",
                "result": "verified",
                "required": true,
                "model_id": "model-b",
                "claims": {},
                "session_id": format!("as_{}", "b".repeat(64)),
                "channel_bindings": [{
                    "type": "tls_spki_sha256",
                    "origin": "https://upstream.example.com",
                    "spki_sha256": "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1"
                }]
            }),
        ];
        assert!(required_verified_upstream_session_id(&duplicate_serving).is_err());
    }

    // [MEMCHAIN-PHALA-ACI-UPSTREAM-BINDING 2026-10-06 by Codex]
    // Authored to reject assertion-only success; not executed in this phase.
    #[test]
    fn required_upstream_verification_needs_a_supported_channel_binding() {
        let events = vec![serde_json::json!({
            "type": "upstream.verified",
            "result": "verified",
            "required": true,
            "model_id": "model-a",
            "claims": {},
            "session_id": format!("as_{}", "a".repeat(64)),
            "channel_bindings": [{"type": "unknown", "anything": "ignored"}]
        })];
        assert!(required_verified_upstream_session_id(&events).is_err());
    }

    // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] Authored, not executed.
    #[test]
    fn aci_resources_use_versioned_paths_with_or_without_v1_api_base() {
        assert_eq!(
            aci_v1_resource_url("https://inference.phala.com", &["attestation"])
                .unwrap()
                .as_str(),
            "https://inference.phala.com/v1/aci/attestation"
        );
        let session_id = format!("as_{}", "a".repeat(64));
        assert_eq!(
            aci_v1_resource_url(
                "https://inference.phala.com/v1",
                &["sessions", session_id.as_str()],
            )
            .unwrap()
            .as_str(),
            format!("https://inference.phala.com/v1/aci/sessions/{session_id}")
        );
    }

    #[test]
    fn chat_endpoint_accepts_root_versioned_and_complete_bases() {
        assert_eq!(
            chat_completions_url("https://api.example.com"),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.example.com/v1"),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn endpoint_resolution_removes_all_trailing_slashes_without_network_state() {
        assert_eq!(
            chat_completions_url("http://localhost:11434/v1///"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-06 by Codex] Keep generic provider
    // constructor coverage aligned with its public non-ACI API.
    #[test]
    fn constructor_rejects_invalid_base_and_missing_explicit_secret() {
        assert!(matches!(
            OpenAiCompatProvider::new("local", "not-a-url", "", "model", None, None),
            Err(LlmProviderInitError::InvalidApiBase)
        ));
        assert!(matches!(
            OpenAiCompatProvider::new(
                "local",
                "http://localhost:11434/v1",
                "$AERONYX_TEST_SUPERNODE_SECRET_MUST_NOT_EXIST_20260814",
                "model",
                None,
                None,
            ),
            Err(LlmProviderInitError::ProviderSecretUnavailable)
        ));
        assert!(OpenAiCompatProvider::new(
            "local",
            "http://localhost:11434/v1",
            "",
            "model",
            None,
            None,
        )
        .is_ok());
    }

    // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Authored only:
    // an invalid local resource URL is a tripwire if appraisal is reached.
    #[tokio::test]
    async fn aci_chat_and_embeddings_reject_escaped_wire_overflow_before_appraisal() {
        let mut provider = OpenAiCompatProvider::new_phala_aci(
            "phala", "https://inference.phala.com/v1", "synthetic-key", "model",
            None, None, &[], &[], &[],
        ).unwrap();
        provider.api_base = "://invalid-before-network".into();
        let request = ChatRequest::simple("\u{0}".repeat(180_000));
        assert!(request.validate_aci_bounds().is_ok());
        assert!(matches!(provider.chat(&request).await, Err(LlmError::AciRequestRejected)));
        let request = EmbeddingRequest {
            model: "model".into(), input: vec!["\u{0}".repeat(16 * 1024); 32],
        };
        assert!(matches!(provider.embed(&request).await, Err(LlmError::AciRequestRejected)));
        provider.model = "m".repeat(257);
        assert!(matches!(provider.chat(&ChatRequest::simple("small")).await, Err(LlmError::AciVerifierUnavailable)));
        provider.model = "model".into();
        provider.temperature = Some(f32::NAN);
        assert!(matches!(provider.chat(&ChatRequest::simple("small")).await, Err(LlmError::AciVerifierUnavailable)));
        provider.temperature = None;
        provider.max_tokens = Some(0);
        assert!(matches!(provider.chat(&ChatRequest::simple("small")).await, Err(LlmError::AciVerifierUnavailable)));
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn phala_constructor_rejects_non_phala_authority() {
        assert!(matches!(
            OpenAiCompatProvider::new_phala_aci(
                "phala",
                "https://api.example.invalid/v1",
                "test-key",
                "model",
                None,
                None,
                &[],
                &[],
                &[],
            ),
            Err(LlmProviderInitError::InvalidApiBase)
        ));
    }

    // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Authored, not executed.
    #[tokio::test]
    async fn generic_provider_refuses_confidential_request_before_network() {
        let provider = OpenAiCompatProvider::new(
            "generic",
            "http://127.0.0.1:1/v1",
            "",
            "model",
            None,
            None,
        )
        .unwrap();
        let mut request = ChatRequest::simple("");
        request.require_aci_verified = true;

        assert!(matches!(
            provider.chat(&request).await,
            Err(LlmError::ConfidentialServingRequired)
        ));
    }

    // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Authored, not run.
    // An invalid URL is a tripwire: any attempted send/appraisal returns a
    // different error. Holds must not poison provider health or consume data.
    #[tokio::test]
    async fn direct_generic_and_phala_calls_cannot_release_source_plaintext() {
        let mut generic = OpenAiCompatProvider::new(
            "generic", "http://127.0.0.1:1/v1", "", "model", None, None,
        ).unwrap();
        generic.api_base = "://invalid-before-network".into();
        let mut phala = OpenAiCompatProvider::new_phala_aci(
            "phala", "https://inference.phala.com/v1", "synthetic-key", "model",
            None, None, &[], &[], &[],
        ).unwrap();
        phala.api_base = "://invalid-before-network".into();
        let mut request = ChatRequest::simple("source-private-prompt");
        for provider in [&generic, &phala] {
            assert!(matches!(provider.chat(&request).await,
                Err(LlmError::ConfidentialE2eeTransportUnavailable)));
            assert!(provider.is_healthy());
        }
        request.require_aci_verified = true;
        assert!(matches!(phala.chat(&request).await,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
        let embeddings = EmbeddingRequest {
            model: "model".into(), input: vec!["source-private-embedding".into()],
        };
        assert!(matches!(phala.embed(&embeddings).await,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
        assert!(phala.is_healthy());
        assert_eq!(request.messages[0].content, "source-private-prompt");
        assert_eq!(embeddings.input, vec!["source-private-embedding"]);
    }
}
