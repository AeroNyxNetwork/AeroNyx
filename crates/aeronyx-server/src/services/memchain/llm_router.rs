// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_router.rs
// ============================================
//! # LLM Router — Task Type → Provider Routing + Fallback
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Routes `CognitiveTaskType` requests to the appropriate configured provider,
//! with fallback to other providers if the primary is unhealthy or errors.
//! Also provides token cost estimation.
//!
//! ## Routing Strategy
//! 1. Look up the provider name configured for this `CognitiveTaskType` in the
//!    `routing` config section.
//! 2. If that provider is healthy, use it.
//! 3. If unhealthy (rate limited), try providers in order until one works.
//! 4. If all fail, return the last error.
//!
//! ## Cost Estimation
//! `estimate_cost()` applies approximate fee rates (USD per 1M tokens).
//! Rates are APPROXIMATE and may be stale — use for budgeting, not billing.
//! Update `cost_table()` when provider pricing changes.
//!
//! ## Prompt Builders
//! Prompt construction lives entirely in `prompts.rs`. `LlmRouter` does NOT
//! contain prompt-building logic.
//!
//! ⚠️ Important Note for Next Developer:
//! - CognitiveTaskType is defined in config_supernode.rs (single source of truth).
//!   This file imports it via llm_provider.rs re-export.
//! - `new()` accepts `TaskRoutingConfig` (from config_supernode.rs).
//! - Routing table keys use as_str() which returns the DB string form.
//! - `COST_TABLE` rates will go stale. Consider making them config-driven (Phase C).
//! - Prompt builders belong in `prompts.rs`. Do NOT add them here.
//!
//! ## Last Modified
//! v2.5.5-FailureBoundary - [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex]
//!   Runtime fallback logs now emit privacy-safe reason codes only.
//! v2.5.0+SuperNode - 🌟 Created.
//! v2.5.0+Audit Fix - 🔧 Various fixes (see previous doc).
//! v2.5.0+Unify     - 🔧 [BUG FIX] Unified CognitiveTaskType from config_supernode.rs.
//!   route() now uses as_str() consistently (as_str == task_type_str, they're aliases).
//!   Routing table built using config_supernode::CognitiveTaskType::ALL which has
//!   all 6 variants (SessionTitle, CommunityNarrative, ConflictResolution,
//!   RecallSynthesis, CodeAnalysis, EntityDescription).

use std::collections::HashMap;
use std::sync::Arc;
use sha2::Digest;

use tracing::{debug, info, warn};

use super::llm_provider::{ChatRequest, ChatResponse, CognitiveTaskType, EmbeddingRequest, LlmError, LlmProvider};
use crate::config_supernode::TaskRoutingConfig;

// [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
#[derive(serde::Serialize)]
struct EmbeddingRequestPayload<'a> {
    model: &'a str,
    input: &'a [String],
    provider: EmbeddingProviderConstraint,
}

#[derive(serde::Serialize)]
struct EmbeddingProviderConstraint {
    aci_verified: bool,
}

// [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex]
fn has_verified_aci_response(provider: &dyn LlmProvider, response: &ChatResponse) -> bool {
    let (Some(hints), Some(proof)) = (
        response.aci_response_hints.as_ref(),
        response.aci_verification.as_ref(),
    ) else {
        return false;
    };
    hints.has_valid_shape()
        && proof.has_complete_shape()
        && hints.workload_id == proof.workload_id
        && hints.keyset_digest == proof.keyset_digest
        && hints.receipt_id == proof.receipt_id
        // [PHALA-133171-PROFILE 2026-10-08 by Codex] Stable identity and
        // rotating keyset are separately checked by has_complete_shape.
        && proof.attestation_report.get("workload_keyset_digest").and_then(serde_json::Value::as_str)
            == Some(proof.keyset_digest.as_str())
        && provider.accepts_aci_compose_hash(&proof.compose_hash)
        && provider.accepts_aci_kms_root(&proof.kms_root_public_key)
        // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex] The report's
        // self-asserted source fields must match local policy for this quote.
        && proof
            .attestation_report
            .get("attestation")
            .and_then(|attestation| attestation.get("source_provenance"))
            .is_some_and(|provenance| {
                provider.accepts_aci_source_provenance(&proof.compose_hash, provenance)
            })
        // [MEMCHAIN-PHALA-ACI-PINNED-CONTRACT 2026-10-06 by Codex] ACI v1
        // binds service identity through this digest, not an identity header.
        && hints.keyset_digest == proof.keyset_digest
        && hints.receipt_id == proof.receipt_id
}

// ============================================
// Cost Table (approximate, USD per 1M tokens)
// ============================================

struct CostRate {
    input_per_m: f64,
    output_per_m: f64,
    cached_input_per_m: f64,
}

fn cost_table() -> HashMap<&'static str, CostRate> {
    let mut m = HashMap::new();
    m.insert(
        "gpt-4o-mini",
        CostRate {
            input_per_m: 0.15,
            output_per_m: 0.60,
            cached_input_per_m: 0.075,
        },
    );
    m.insert(
        "gpt-4o",
        CostRate {
            input_per_m: 2.50,
            output_per_m: 10.0,
            cached_input_per_m: 1.25,
        },
    );
    m.insert(
        "deepseek-chat",
        CostRate {
            input_per_m: 0.07,
            output_per_m: 1.10,
            cached_input_per_m: 0.014,
        },
    );
    m.insert(
        "deepseek-reasoner",
        CostRate {
            input_per_m: 0.55,
            output_per_m: 2.19,
            cached_input_per_m: 0.14,
        },
    );
    m.insert(
        "claude-haiku-4-5-20251001",
        CostRate {
            input_per_m: 0.80,
            output_per_m: 4.00,
            cached_input_per_m: 0.08,
        },
    );
    m.insert(
        "claude-sonnet-4-6",
        CostRate {
            input_per_m: 3.00,
            output_per_m: 15.0,
            cached_input_per_m: 0.30,
        },
    );
    m.insert(
        "llama-3.3-70b-versatile",
        CostRate {
            input_per_m: 0.59,
            output_per_m: 0.79,
            cached_input_per_m: 0.0,
        },
    );
    m.insert(
        "llama3.2",
        CostRate {
            input_per_m: 0.0,
            output_per_m: 0.0,
            cached_input_per_m: 0.0,
        },
    );
    m
}

// ============================================
// Provider metadata (for health checks)
// ============================================

struct ProviderMeta {
    api_base: String,
    model: String,
}

// ============================================
// LlmRouter
// ============================================

/// Routes cognitive tasks to the appropriate LLM provider.
/// Thread-safe (Arc<dyn LlmProvider> + immutable after construction).
pub struct LlmRouter {
    /// All configured providers, keyed by name.
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    /// Provider metadata for health checks (name → meta).
    provider_meta: HashMap<String, ProviderMeta>,
    /// task_type as_str() → provider name.
    routing: HashMap<String, String>,
    /// Ordered list of provider names for fallback traversal.
    provider_order: Vec<String>,
    /// Dedicated, explicitly configured provider for confidential embeddings.
    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    embedding_provider: Option<String>,
    embedding_model: Option<String>,
}

impl LlmRouter {
    /// Construct a new router from providers and routing config.
    ///
    /// ## Parameters
    /// - `providers`: Vec of `(name, api_base, model, Arc<dyn LlmProvider>)`.
    /// - `routing`: `TaskRoutingConfig` from `config_supernode.rs`.
    pub fn new(
        providers: Vec<(String, String, String, Arc<dyn LlmProvider>)>,
        routing: TaskRoutingConfig,
    ) -> Self {
        let provider_order: Vec<String> = providers.iter().map(|(n, _, _, _)| n.clone()).collect();
        let mut provider_map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        let mut meta_map: HashMap<String, ProviderMeta> = HashMap::new();

        for (name, api_base, model, p) in providers {
            meta_map.insert(name.clone(), ProviderMeta { api_base, model });
            provider_map.insert(name, p);
        }

        // Build routing table from TaskRoutingConfig.
        // v2.5.0+Unify: CognitiveTaskType::ALL is from config_supernode (canonical).
        // as_str() returns the DB string form used as routing key.
        let mut routing_table: HashMap<String, String> = HashMap::new();
        for task_type in crate::config_supernode::CognitiveTaskType::ALL {
            if let Some(provider_name) = routing.provider_for(*task_type) {
                if provider_map.contains_key(provider_name) {
                    routing_table.insert(task_type.as_str().to_string(), provider_name.to_string());
                } else {
                    warn!(
                        task_type = task_type.as_str(),
                        provider = provider_name,
                        "[LLM_ROUTER] Routing references unknown provider — using fallback"
                    );
                }
            }
        }

        info!(
            providers = provider_map.len(),
            routes = routing_table.len(),
            "[LLM_ROUTER] Initialized"
        );

        Self {
            providers: provider_map,
            provider_meta: meta_map,
            routing: routing_table,
            provider_order,
            embedding_provider: None,
            embedding_model: None,
        }
    }

    /// Configure embeddings on one named ACI-verifying provider only.
    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    pub fn with_embedding_route(
        mut self,
        provider_name: &str,
        model: &str,
    ) -> Result<Self, LlmError> {
        let provider = self.providers.get(provider_name)
            .ok_or_else(|| LlmError::NotConfigured("embedding provider".into()))?;
        if model.trim().is_empty()
            || model.len() > 256
            || !provider.supports_aci_verified()
            || !provider.has_cryptographic_aci_verifier()
        {
            return Err(LlmError::ConfidentialServingRequired);
        }
        self.embedding_provider = Some(provider_name.to_owned());
        self.embedding_model = Some(model.to_owned());
        Ok(self)
    }

    pub fn has_embedding_route(&self) -> bool {
        self.embedding_provider.is_some() && self.embedding_model.is_some()
    }

    pub fn embedding_model(&self) -> Option<&str> {
        self.embedding_model.as_deref()
    }

    // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex] ACI attestation and
    // receipts do not encrypt request/response fields for the source client.
    // This node has no ACI E2EE field-encryption implementation or source key.
    #[must_use]
    pub const fn has_client_to_tee_e2ee_transport(&self) -> bool {
        // [PHALA-NODE-PLAINTEXT-GATE 2026-10-07 by Codex] Direct provider
        // calls and worker admission use this same source-ownership boundary.
        super::llm_provider::node_has_source_e2ee_transport()
    }

    /// Embeddings never traverse generic-provider fallback. Both the provider
    /// and model are pinned by explicit operator configuration, and evidence is
    /// structurally and policy checked before vectors leave this method.
    // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex]
    pub async fn embed_batch(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LlmError> {
        if !self.has_client_to_tee_e2ee_transport() {
            return Err(LlmError::ConfidentialE2eeTransportUnavailable);
        }
        const MAX_INPUTS: usize = 32;
        const MAX_INPUT_BYTES: usize = 16 * 1024;
        const MAX_DIMENSIONS: usize = 16_384;
        if input.is_empty() || input.len() > MAX_INPUTS
            || input.iter().any(|text| text.is_empty() || text.len() > MAX_INPUT_BYTES)
        {
            return Err(LlmError::AciResponseContractViolation);
        }
        let provider_name = self.embedding_provider.as_deref()
            .ok_or_else(|| LlmError::NotConfigured("Phala embedding route".into()))?;
        let model = self.embedding_model.as_deref()
            .ok_or_else(|| LlmError::NotConfigured("Phala embedding model".into()))?;
        let provider = self.providers.get(provider_name)
            .ok_or_else(|| LlmError::NotConfigured("Phala embedding provider".into()))?;
        if !provider.is_healthy() {
            return Err(LlmError::AciVerifierUnavailable);
        }
        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Reject the
        // exact oversized payload before invoking even a custom provider.
        let request_body = super::llm_provider::serialize_bounded_aci_request(&EmbeddingRequestPayload {
            model,
            input: &input,
            provider: EmbeddingProviderConstraint { aci_verified: true },
        })?;
        let request_body_sha256 = format!("sha256:{}", hex::encode(sha2::Sha256::digest(request_body)));
        let response = provider.embed(&EmbeddingRequest {
            model: model.to_owned(),
            input: input.clone(),
        }).await?;
        let (Some(hints), Some(proof)) = (
            response.aci_response_hints.as_ref(), response.aci_verification.as_ref()
        ) else {
            return Err(LlmError::AciResponseContractViolation);
        };
        let dimensions = response.embeddings.first().map(Vec::len).unwrap_or(0);
        if !hints.has_valid_shape()
            || !proof.has_complete_shape()
            || hints.workload_id != proof.workload_id
            || hints.keyset_digest != proof.keyset_digest
            || hints.receipt_id != proof.receipt_id
            || !provider.accepts_aci_compose_hash(&proof.compose_hash)
            || !provider.accepts_aci_kms_root(&proof.kms_root_public_key)
            || !proof.attestation_report.get("attestation")
                .and_then(|attestation| attestation.get("source_provenance"))
                .is_some_and(|provenance| provider.accepts_aci_source_provenance(&proof.compose_hash, provenance))
            || proof.attestation_report.get("workload_keyset_digest").and_then(serde_json::Value::as_str)
                != Some(proof.keyset_digest.as_str())
            || proof.request_body_sha256 != request_body_sha256
            || response.model_used != model
            || response.embeddings.len() != input.len()
            || dimensions == 0
            || response.embeddings.iter().any(|vector| vector.is_empty()
                || vector.len() > MAX_DIMENSIONS
                || vector.len() != dimensions
                || vector.iter().any(|value| !value.is_finite()))
        {
            return Err(LlmError::AciResponseContractViolation);
        }
        Ok(response.embeddings)
    }

    /// Route a request to the configured provider for this task type.
    /// Falls back to other healthy providers if the primary fails.
    ///
    /// v2.5.0+Unify: Uses as_str() which is the canonical string method.
    /// (as_str and task_type_str are aliases — both return the same string.)
    pub async fn route(
        &self,
        task_type: &CognitiveTaskType,
        req: &ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        if !self.has_client_to_tee_e2ee_transport() {
            return Err(LlmError::ConfidentialE2eeTransportUnavailable);
        }
        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] A malformed
        // confidential input must not trigger provider selection or fallback.
        if req.require_aci_verified { req.validate_aci_bounds()?; }
        let task_str = task_type.as_str();

        // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Confidential requests
        // may only use a provider that sends the ACI verified-route constraint.
        let configured_primary = self.routing.get(task_str);
        let eligible = |name: &String| {
            self.providers
                .get(name)
                .is_some_and(|provider| {
                    !req.require_aci_verified
                        || (provider.supports_aci_verified()
                            && provider.has_cryptographic_aci_verifier())
                })
        };
        let primary_name = configured_primary
            .filter(|name| eligible(name))
            .or_else(|| self.provider_order.iter().find(|name| eligible(name)));

        let Some(primary_name) = primary_name else {
            if req.require_aci_verified {
                if self.provider_order.iter().any(|name| {
                    self.providers
                        .get(name)
                        .is_some_and(|provider| provider.supports_aci_verified())
                }) {
                    return Err(LlmError::AciVerifierUnavailable);
                }
                return Err(LlmError::ConfidentialServingRequired);
            }
            return Err(LlmError::NotConfigured(format!(
                "no provider configured for task_type={}",
                task_str
            )));
        };

        // Try primary provider first
        if let Some(primary) = self.providers.get(primary_name) {
            if primary.is_healthy() {
                match primary.chat(req).await {
                    Ok(resp) => {
                        // [MEMCHAIN-PHALA-ACI-IDENTITY 2026-10-05 by Codex]
                        if req.require_aci_verified
                            && !has_verified_aci_response(primary.as_ref(), &resp)
                        {
                            // [MEMCHAIN-PHALA-ACI-HEADERS 2026-10-05 by Codex]
                            // A custom provider cannot bypass the response gate.
                            return Err(LlmError::AciResponseContractViolation);
                        }
                        debug!(
                            provider = %primary_name, task = task_str,
                            "[LLM_ROUTER] Routed to primary"
                        );
                        return Ok(resp);
                    }
                    Err(LlmError::RateLimit { .. }) => {
                        warn!(provider = %primary_name, "[LLM_ROUTER] Primary rate limited, trying fallback");
                    }
                    Err(error @ LlmError::AciResponseContractViolation) => return Err(error),
                    Err(error @ LlmError::AciVerifierUnavailable) => return Err(error),
                    Err(error @ LlmError::AciRequestRejected) => return Err(error),
                    Err(e) => {
                        // [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex]
                        // Provider diagnostics can contain response text or a
                        // transport endpoint; routine operator logs only need
                        // the stable category for fallback observability.
                        warn!(
                            provider = %primary_name,
                            reason = e.reason_code(),
                            "[LLM_ROUTER] Primary failed, trying fallback"
                        );
                    }
                }
            } else {
                debug!(provider = %primary_name, "[LLM_ROUTER] Primary unhealthy, skipping");
            }
        }

        // Fallback: try remaining providers in declaration order
        let mut last_error =
            LlmError::NotConfigured(format!("all providers failed for task_type={}", task_str));
        for name in &self.provider_order {
            if name == primary_name {
                continue;
            }
            if let Some(provider) = self.providers.get(name) {
                if req.require_aci_verified
                    && !(provider.supports_aci_verified()
                        && provider.has_cryptographic_aci_verifier())
                {
                    continue;
                }
                if !provider.is_healthy() {
                    continue;
                }
                match provider.chat(req).await {
                    Ok(resp) => {
                        // [MEMCHAIN-PHALA-ACI-IDENTITY 2026-10-05 by Codex]
                        if req.require_aci_verified
                            && !has_verified_aci_response(provider.as_ref(), &resp)
                        {
                            return Err(LlmError::AciResponseContractViolation);
                        }
                        info!(
                            provider = %name, task = task_str,
                            "[LLM_ROUTER] Fallback provider succeeded"
                        );
                        return Ok(resp);
                    }
                    Err(e) => {
                        if matches!(
                            &e,
                            LlmError::AciResponseContractViolation
                                | LlmError::AciVerifierUnavailable
                                | LlmError::AciRequestRejected
                        ) {
                            return Err(e);
                        }
                        warn!(
                            provider = %name,
                            reason = e.reason_code(),
                            "[LLM_ROUTER] Fallback failed"
                        );
                        last_error = e;
                    }
                }
            }
        }

        Err(last_error)
    }

    // [MEMCHAIN-PHALA-ACI-PROOF-SCOPE 2026-10-06 by Codex]
    pub(crate) fn response_has_accepted_aci_evidence(&self, response: &ChatResponse) -> bool {
        self.providers
            .get(&response.provider_name)
            .is_some_and(|provider| has_verified_aci_response(provider.as_ref(), response))
    }

    /// Estimate cost in USD for a given model + token counts.
    pub fn estimate_cost(
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        cached_tokens: u32,
    ) -> f64 {
        let table = cost_table();
        let rate = table.get(model).or_else(|| {
            table
                .iter()
                .find(|(k, _)| model.starts_with(*k))
                .map(|(_, v)| v)
        });

        match rate {
            Some(r) => {
                let billable_input = (input_tokens.saturating_sub(cached_tokens)) as f64;
                let cached = cached_tokens as f64;
                let output = output_tokens as f64;
                (billable_input * r.input_per_m
                    + cached * r.cached_input_per_m
                    + output * r.output_per_m)
                    / 1_000_000.0
            }
            None => 0.0,
        }
    }

    /// Number of configured providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// List of provider names in declaration order.
    pub fn provider_names(&self) -> Vec<&str> {
        self.provider_order.iter().map(|s| s.as_str()).collect()
    }

    /// Provider configs for health check pings.
    /// Returns `(name, api_base, model)` tuples.
    pub fn provider_configs(&self) -> Vec<(String, String, String)> {
        self.provider_order
            .iter()
            .filter_map(|name| {
                self.provider_meta
                    .get(name)
                    .map(|meta| (name.clone(), meta.api_base.clone(), meta.model.clone()))
            })
            .collect()
    }

    /// Check if at least one provider is configured and healthy.
    pub fn any_healthy(&self) -> bool {
        self.providers.values().any(|p| p.is_healthy())
    }
}

impl std::fmt::Debug for LlmRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmRouter")
            .field("providers", &self.provider_order)
            .field("routes", &self.routing.len())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_supernode::TaskRoutingConfig;

    // [PHALA-POPULATED-ROUTER-GATE 2026-10-08 by Codex] Returning the same
    // error as the outer gate makes counters necessary: error-only assertions
    // would miss an accidental call into a configured primary or fallback.
    struct SourceGateProbe {
        name: &'static str,
        healthy: bool,
        verified: bool,
        health_calls: std::sync::atomic::AtomicUsize,
        chat_calls: std::sync::atomic::AtomicUsize,
        embedding_calls: std::sync::atomic::AtomicUsize,
    }

    impl SourceGateProbe {
        fn new(name: &'static str, healthy: bool, verified: bool) -> Self {
            Self {
                name, healthy, verified,
                health_calls: std::sync::atomic::AtomicUsize::new(0),
                chat_calls: std::sync::atomic::AtomicUsize::new(0),
                embedding_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> [usize; 3] {
            use std::sync::atomic::Ordering::SeqCst;
            [self.health_calls.load(SeqCst), self.chat_calls.load(SeqCst),
                self.embedding_calls.load(SeqCst)]
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for SourceGateProbe {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            self.chat_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(LlmError::ConfidentialE2eeTransportUnavailable)
        }

        async fn embed(&self, _req: &EmbeddingRequest)
            -> Result<super::super::llm_provider::EmbeddingResponse, LlmError> {
            self.embedding_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(LlmError::ConfidentialE2eeTransportUnavailable)
        }

        fn name(&self) -> &str { self.name }
        fn default_model(&self) -> &str { "synthetic-only" }
        fn is_healthy(&self) -> bool {
            self.health_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.healthy
        }
        fn supports_aci_verified(&self) -> bool { self.verified }
        fn has_cryptographic_aci_verifier(&self) -> bool { self.verified }
    }

    // [PHALA-POPULATED-ROUTER-GATE 2026-10-08 by Codex] Authored, not run.
    // Calibrate the spy through its real trait entrypoints, with no transport.
    #[tokio::test]
    async fn source_gate_probe_detects_calls_even_when_errors_match() {
        let probe = SourceGateProbe::new("probe", true, true);
        assert_eq!(probe.calls(), [0, 0, 0]);
        assert!(probe.is_healthy());
        assert!(matches!(probe.chat(&ChatRequest::simple("synthetic")).await,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
        assert!(matches!(probe.embed(&EmbeddingRequest {
            model: "synthetic-only".into(), input: vec!["synthetic".into()],
        }).await, Err(LlmError::ConfidentialE2eeTransportUnavailable)));
        assert_eq!(probe.calls(), [1, 1, 1]);
    }

    #[tokio::test]
    async fn configured_primary_fallback_and_embeddings_never_bypass_source_gate() {
        for primary_healthy in [false, true] {
            for primary_verified in [false, true] {
                let primary = Arc::new(SourceGateProbe::new(
                    "primary", primary_healthy, primary_verified));
                let fallback = Arc::new(SourceGateProbe::new("fallback", true, true));
                let providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = vec![
                    ("primary".into(), "https://primary.invalid".into(),
                        "synthetic-only".into(), primary.clone()),
                    ("fallback".into(), "https://fallback.invalid".into(),
                        "synthetic-only".into(), fallback.clone()),
                ];
                let mut router = LlmRouter::new(providers, TaskRoutingConfig::default())
                    .with_embedding_route("fallback", "synthetic-only").unwrap();
                // Pin every real task type to the first provider, independent
                // of operator defaults; verified fallback is still available.
                for task in CognitiveTaskType::ALL {
                    router.routing.insert(task.as_str().to_owned(), "primary".into());
                }
                assert_eq!(router.provider_count(), 2);
                assert!(router.has_embedding_route());
                for task in CognitiveTaskType::ALL {
                    for require_aci_verified in [false, true] {
                        for text in [String::new(), "source-private-prompt".into(),
                            "x".repeat(1024 * 1024 + 1)] {
                            let mut request = ChatRequest::simple(&text);
                            request.require_aci_verified = require_aci_verified;
                            assert!(matches!(router.route(task, &request).await,
                                Err(LlmError::ConfidentialE2eeTransportUnavailable)));
                            assert_eq!(request.messages[0].content, text);
                            assert_eq!(primary.calls(), [0, 0, 0]);
                            assert_eq!(fallback.calls(), [0, 0, 0]);
                        }
                    }
                }
                for input in [Vec::new(), vec!["source-private-input".into()],
                    vec!["x".repeat(16 * 1024 + 1)], vec!["synthetic".into(); 33]] {
                    assert!(matches!(router.embed_batch(input).await,
                        Err(LlmError::ConfidentialE2eeTransportUnavailable)));
                    assert_eq!(primary.calls(), [0, 0, 0]);
                    assert_eq!(fallback.calls(), [0, 0, 0]);
                }
            }
        }
    }

    // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex]
    #[tokio::test]
    async fn server_model_calls_fail_closed_before_provider_io() {
        let router = LlmRouter::new(Vec::new(), TaskRoutingConfig::default());
        let chat = router
            .route(
                &CognitiveTaskType::SessionTitle,
                &ChatRequest::simple("private input"),
            )
            .await;
        assert!(matches!(
            chat,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)
        ));

        let embedding = router.embed_batch(vec!["private input".into()]).await;
        assert!(matches!(
            embedding,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)
        ));

        // [PHALA-ACI-REQUEST-BOUNDARY 2026-10-07 by Codex] Resource
        // preflight never substitutes for, or opens, the source E2EE gate.
        let mut oversized = ChatRequest::simple("x".repeat(1024 * 1024 + 1));
        oversized.require_aci_verified = true;
        assert!(matches!(router.route(&CognitiveTaskType::SessionTitle, &oversized).await,
            Err(LlmError::ConfidentialE2eeTransportUnavailable)));
    }

    // [MEMCHAIN-PHALA-MEASUREMENT-POLICY 2026-10-05 by Codex]
    #[test]
    fn verified_response_requires_explicitly_allowed_compose_measurement() {
        struct AllowlistedProvider;
        #[async_trait::async_trait]
        impl LlmProvider for AllowlistedProvider {
            async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
                unreachable!("the response gate is checked without network I/O")
            }
            fn name(&self) -> &str { "allowlisted" }
            fn default_model(&self) -> &str { "test" }
            fn supports_aci_verified(&self) -> bool { true }
            fn has_cryptographic_aci_verifier(&self) -> bool { true }
            fn accepts_aci_compose_hash(&self, hash: &str) -> bool {
                hash == format!("sha256:{}", "e".repeat(64))
            }
            fn accepts_aci_kms_root(&self, root: &str) -> bool {
                root == format!("0x02{}", "f".repeat(64))
            }
            // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
            fn accepts_aci_source_provenance(
                &self,
                compose_hash: &str,
                provenance: &serde_json::Value,
            ) -> bool {
                compose_hash == format!("sha256:{}", "e".repeat(64))
                    && provenance["repo_url"] == "https://example.invalid/repo"
                    && provenance["repo_commit"]
                        == "0123456789abcdef0123456789abcdef01234567"
            }
        }

        let provider = AllowlistedProvider;
        let evidence = super::super::llm_provider::AciVerificationEvidence {
            // [PHALA-133171-PROFILE 2026-10-08 by Codex] Distinct IDs
            // catch accidental resurrection of the keyset-as-identity alias.
            wire_profile: super::super::llm_provider::PHALA_ACI_WIRE_PROFILE.into(),
            key_custody_scope: "identity_kms_operational_measured_code".into(),
            workload_id: format!("sha256:{}", "b".repeat(64)),
            keyset_digest: format!("sha256:{}", "a".repeat(64)),
            compose_hash: format!("sha256:{}", "e".repeat(64)),
            kms_root_public_key: format!("0x02{}", "f".repeat(64)),
            receipt_id: "receipt-1".into(),
            upstream_session_id: format!("as_{}", "a".repeat(64)),
            upstream_session: serde_json::json!({"test_only": true}),
            upstream_claim_scope: "gateway_assertion".into(),
            request_body_sha256: format!("sha256:{}", "b".repeat(64)),
            response_body_sha256: format!("sha256:{}", "d".repeat(64)),
            attestation_report: serde_json::json!({
                "workload_id": format!("sha256:{}", "b".repeat(64)),
                "workload_keyset_digest": format!("sha256:{}", "a".repeat(64)),
                "attestation": {"source_provenance": {
                    "repo_url": "https://example.invalid/repo",
                    "repo_commit": "0123456789abcdef0123456789abcdef01234567"
                }}
            }),
            receipt: serde_json::json!({
                "workload_id": format!("sha256:{}", "b".repeat(64)),
                "workload_keyset_digest": format!("sha256:{}", "a".repeat(64)),
                "signature_verified": true
            }),
            upstream_verified_required: true,
        };
        let response = ChatResponse {
            content: "synthetic".into(),
            usage: Default::default(),
            model_used: "test".into(),
            provider_name: "allowlisted".into(),
            latency_ms: 0,
            aci_response_hints: Some(super::super::llm_provider::AciResponseHints {
                version: "aci/1".into(),
                workload_id: evidence.workload_id.clone(),
                keyset_digest: evidence.keyset_digest.clone(),
                receipt_id: evidence.receipt_id.clone(),
            }),
            aci_verification: Some(evidence.clone()),
        };
        assert!(super::has_verified_aci_response(&provider, &response));

        // [PHALA-133171-PROFILE 2026-10-08 by Codex] Structural policy
        // tests only: synthetic providers cannot establish hardware evidence.
        for (field, replacement) in [
            ("wire_profile", serde_json::json!("")),
            ("wire_profile", serde_json::json!("aci/1@a991c08553cbd0638199abfd6eace9c83ee0a891")),
            ("key_custody_scope", serde_json::json!("all_keys_independently_proven")),
        ] {
            let mut invalid = response.clone();
            let mut value = serde_json::to_value(invalid.aci_verification.as_ref().unwrap()).unwrap();
            value[field] = replacement;
            invalid.aci_verification = Some(serde_json::from_value(value).unwrap());
            assert!(!super::has_verified_aci_response(&provider, &invalid), "{field}");
        }
        let mut historical = serde_json::to_value(&evidence).unwrap();
        historical.as_object_mut().unwrap().remove("wire_profile");
        historical.as_object_mut().unwrap().remove("key_custody_scope");
        let historical: super::super::llm_provider::AciVerificationEvidence = serde_json::from_value(historical).unwrap();
        assert!(!historical.has_complete_shape(), "historical records remain readable, not current authority");
        for (report, field) in [(true, "workload_id"), (true, "workload_keyset_digest"), (false, "workload_id")] {
            let mut invalid = response.clone();
            let proof = invalid.aci_verification.as_mut().unwrap();
            let value = if report { &mut proof.attestation_report } else { &mut proof.receipt };
            value[field] = serde_json::json!(format!("sha256:{}", "c".repeat(64)));
            assert!(!super::has_verified_aci_response(&provider, &invalid), "{report}/{field}");
        }

        let mut mismatched_keyset = response.clone();
        mismatched_keyset
            .aci_response_hints
            .as_mut()
            .unwrap()
            .keyset_digest = format!("sha256:{}", "d".repeat(64));
        assert!(!super::has_verified_aci_response(&provider, &mismatched_keyset));

        let mut mismatched_identity = response.clone();
        // [MEMCHAIN-PHALA-ACI-IDENTITY-HEADER 2026-10-06 by Codex]
        mismatched_identity
            .aci_response_hints
            .as_mut()
            .unwrap()
            .workload_id = format!("sha256:{}", "e".repeat(64));
        assert!(!super::has_verified_aci_response(&provider, &mismatched_identity));

        let mut mismatched_receipt = response.clone();
        mismatched_receipt
            .aci_response_hints
            .as_mut()
            .unwrap()
            .receipt_id = "different-receipt".into();
        assert!(!super::has_verified_aci_response(&provider, &mismatched_receipt));

        let mut unapproved = evidence;
        unapproved.compose_hash = format!("sha256:{}", "f".repeat(64));
        let response = ChatResponse {
            aci_verification: Some(unapproved),
            ..response
        };
        assert!(!super::has_verified_aci_response(&provider, &response));
    }

    #[test]
    fn test_estimate_cost_known_model() {
        let cost = LlmRouter::estimate_cost("deepseek-chat", 1_000_000, 1_000_000, 0);
        let expected = (0.07 + 1.10) / 1.0;
        assert!(
            (cost - expected).abs() < 0.001,
            "cost={} expected={}",
            cost,
            expected
        );
    }

    #[test]
    fn test_estimate_cost_prefix_match() {
        let cost_exact = LlmRouter::estimate_cost("deepseek-chat", 100, 50, 0);
        let cost_prefix = LlmRouter::estimate_cost("deepseek-chat-v3", 100, 50, 0);
        assert_eq!(cost_exact, cost_prefix);
    }

    #[test]
    fn test_estimate_cost_unknown_model() {
        let cost = LlmRouter::estimate_cost("unknown-model-xyz", 1000, 500, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_estimate_cost_with_cached_tokens() {
        let cost = LlmRouter::estimate_cost("deepseek-reasoner", 1000, 200, 500);
        let expected = (500.0 * 0.55 + 500.0 * 0.14 + 200.0 * 2.19) / 1_000_000.0;
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_cost_local_model_is_zero() {
        let cost = LlmRouter::estimate_cost("llama3.2", 10_000, 5_000, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_routing_table_built_from_task_routing_config() {
        struct StubProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StubProvider {
            fn name(&self) -> &str {
                "stub"
            }
            fn default_model(&self) -> &str {
                "test-model"
            }
            fn is_healthy(&self) -> bool {
                true
            }
            async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
                Err(LlmError::NotConfigured("stub".into()))
            }
        }

        let routing = TaskRoutingConfig {
            session_title: Some("stub".into()),
            community_narrative: Some("stub".into()),
            conflict_resolution: None,
            recall_synthesis: None,
            code_analysis: None,
            entity_description: None,
            entity_extraction: None,
            fallback: Some("stub".into()),
        };

        let router = LlmRouter::new(
            vec![(
                "stub".into(),
                "http://localhost".into(),
                "test-model".into(),
                Arc::new(StubProvider),
            )],
            routing,
        );

        assert_eq!(router.provider_count(), 1);
        assert_eq!(
            router.routing.get("session_title").map(|s| s.as_str()),
            Some("stub")
        );
        // conflict_resolution has no explicit route but fallback is "stub"
        assert_eq!(
            router
                .routing
                .get("conflict_resolution")
                .map(|s| s.as_str()),
            Some("stub")
        );
    }

    #[test]
    fn test_provider_configs_returns_all() {
        struct StubProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StubProvider {
            fn name(&self) -> &str {
                "stub"
            }
            fn default_model(&self) -> &str {
                "test-model"
            }
            fn is_healthy(&self) -> bool {
                true
            }
            async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
                Err(LlmError::NotConfigured("stub".into()))
            }
        }

        let router = LlmRouter::new(
            vec![
                (
                    "deepseek".into(),
                    "https://api.deepseek.com/v1".into(),
                    "deepseek-chat".into(),
                    Arc::new(StubProvider),
                ),
                (
                    "ollama".into(),
                    "http://localhost:11434/v1".into(),
                    "llama3.2".into(),
                    Arc::new(StubProvider),
                ),
            ],
            TaskRoutingConfig::default(),
        );

        let configs = router.provider_configs();
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].0, "deepseek");
        assert_eq!(configs[0].1, "https://api.deepseek.com/v1");
        assert_eq!(configs[0].2, "deepseek-chat");
        assert_eq!(configs[1].0, "ollama");
    }
}
