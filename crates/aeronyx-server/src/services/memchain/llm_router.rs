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
//! contain prompt-building logic (the original static builder methods were removed
//! in v2.5.0+Audit Fix to eliminate duplication with prompts.rs).
//!
//! ⚠️ Important Note for Next Developer:
//! - `new()` accepts `TaskRoutingConfig` (from config_supernode.rs), not a raw HashMap.
//!   It calls `provider_for()` for each known task type to build the routing table.
//! - `provider_configs()` returns (name, api_base, model) for health-check pings.
//!   The api_base is stored at construction time from `ProviderConfig.api_base`.
//! - `COST_TABLE` rates will go stale. Consider making them config-driven (Phase C).
//! - Fallback order: routing-config provider first, then remaining providers in
//!   declaration order.
//! - Prompt builders belong in `prompts.rs`. Do NOT add them back here.
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.
//! v2.5.0+Audit Fix - 🔧 [BUG FIX] new() now accepts TaskRoutingConfig instead of
//!   HashMap+Option — matches server.rs call site.
//!                  - 🔧 [BUG FIX] route() uses task_type.as_str() (not task_type_str()).
//!                  - 🔧 [BUG FIX] Routing pre-fill uses correct task type string names
//!   (community_narrative, recall_synthesis, entity_description).
//!                  - 🌟 Added provider_configs() → (name, api_base, model) for health pings.
//!                  - 🗑️ Removed duplicate prompt builder methods (build_session_title_prompt
//!   etc.) — prompts.rs is the single source of truth for prompt construction.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{debug, info, warn};

use super::llm_provider::{ChatRequest, ChatResponse, CognitiveTaskType, LlmError, LlmProvider};
// config_supernode is declared at crate root in lib.rs (not under config/)
use crate::config_supernode::TaskRoutingConfig;

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
    m.insert("gpt-4o-mini",               CostRate { input_per_m: 0.15,  output_per_m: 0.60,  cached_input_per_m: 0.075 });
    m.insert("gpt-4o",                    CostRate { input_per_m: 2.50,  output_per_m: 10.0,  cached_input_per_m: 1.25  });
    m.insert("deepseek-chat",             CostRate { input_per_m: 0.07,  output_per_m: 1.10,  cached_input_per_m: 0.014 });
    m.insert("deepseek-reasoner",         CostRate { input_per_m: 0.55,  output_per_m: 2.19,  cached_input_per_m: 0.14  });
    m.insert("claude-haiku-4-5-20251001", CostRate { input_per_m: 0.80,  output_per_m: 4.00,  cached_input_per_m: 0.08  });
    m.insert("claude-sonnet-4-6",         CostRate { input_per_m: 3.00,  output_per_m: 15.0,  cached_input_per_m: 0.30  });
    m.insert("llama-3.3-70b-versatile",   CostRate { input_per_m: 0.59,  output_per_m: 0.79,  cached_input_per_m: 0.0   });
    m.insert("llama3.2",                  CostRate { input_per_m: 0.0,   output_per_m: 0.0,   cached_input_per_m: 0.0   });
    m
}

// ============================================
// Provider metadata (for health checks)
// ============================================

/// Per-provider metadata stored at construction time for health check pings.
/// Separate from LlmProvider trait to avoid exposing config details there.
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
    /// task_type_str → provider name.
    routing: HashMap<String, String>,
    /// Ordered list of provider names for fallback traversal.
    provider_order: Vec<String>,
}

impl LlmRouter {
    /// Construct a new router from providers and routing config.
    ///
    /// ## Parameters
    /// - `providers`: Vec of `(name, api_base, model, Arc<dyn LlmProvider>)`.
    ///   `api_base` and `model` are stored for health check pings.
    /// - `routing`: `TaskRoutingConfig` from `config_supernode.rs`.
    ///   The router calls `provider_for(task_type)` for each known task type
    ///   to build the internal routing table.
    ///
    /// ## v2.5.0+Audit Fix
    /// Previously accepted `HashMap<String, String> + Option<String>` which
    /// didn't match the `server.rs` call site (which passes `TaskRoutingConfig`).
    /// Now accepts `TaskRoutingConfig` directly and builds the routing table
    /// by calling `provider_for()` for each known `CognitiveTaskType`.
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
        // CognitiveTaskType::ALL comes from config_supernode (our canonical list).
        // task_type_str() is the method on llm_provider::CognitiveTaskType.
        // We use config_supernode::CognitiveTaskType here because it drives routing config.
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
        }
    }

    /// Route a request to the configured provider for this task type.
    /// Falls back to other healthy providers if the primary fails.
    ///
    /// Returns `LlmError::NotConfigured` if no provider is available.
    ///
    /// ## v2.5.0+Audit Fix
    /// Changed `task_type.task_type_str()` → `task_type.as_str()` to match
    /// the actual method name on `CognitiveTaskType`.
    pub async fn route(
        &self,
        task_type: &CognitiveTaskType,
        req: &ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        // Use task_type_str() — this is the method on llm_provider::CognitiveTaskType
        let task_str = task_type.task_type_str();

        // Determine primary provider name from routing table
        let primary_name = self.routing.get(task_str)
            .or_else(|| self.provider_order.first());

        let Some(primary_name) = primary_name else {
            return Err(LlmError::NotConfigured(
                format!("no provider configured for task_type={}", task_str)
            ));
        };

        // Try primary provider first
        if let Some(primary) = self.providers.get(primary_name) {
            if primary.is_healthy() {
                match primary.chat(req).await {
                    Ok(resp) => {
                        debug!(
                            provider = %primary_name, task = task_str,
                            "[LLM_ROUTER] Routed to primary"
                        );
                        return Ok(resp);
                    }
                    Err(LlmError::RateLimit { .. }) => {
                        warn!(provider = %primary_name, "[LLM_ROUTER] Primary rate limited, trying fallback");
                    }
                    Err(e) => {
                        warn!(provider = %primary_name, error = %e, "[LLM_ROUTER] Primary failed, trying fallback");
                    }
                }
            } else {
                debug!(provider = %primary_name, "[LLM_ROUTER] Primary unhealthy, skipping");
            }
        }

        // Fallback: try remaining providers in declaration order
        let mut last_error = LlmError::NotConfigured(
            format!("all providers failed for task_type={}", task_str)
        );
        for name in &self.provider_order {
            if name == primary_name { continue; }
            if let Some(provider) = self.providers.get(name) {
                if !provider.is_healthy() { continue; }
                match provider.chat(req).await {
                    Ok(resp) => {
                        info!(
                            provider = %name, task = task_str,
                            "[LLM_ROUTER] Fallback provider succeeded"
                        );
                        return Ok(resp);
                    }
                    Err(e) => {
                        warn!(provider = %name, error = %e, "[LLM_ROUTER] Fallback failed");
                        last_error = e;
                    }
                }
            }
        }

        Err(last_error)
    }

    /// Estimate cost in USD for a given model + token counts.
    ///
    /// Uses approximate rates from `cost_table()`. Returns 0.0 for unknown models.
    /// ⚠️ Approximate — use for budgeting, not billing.
    pub fn estimate_cost(model: &str, input_tokens: u32, output_tokens: u32, cached_tokens: u32) -> f64 {
        let table = cost_table();
        let rate = table.get(model)
            .or_else(|| {
                table.iter()
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
                    + output * r.output_per_m) / 1_000_000.0
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
    ///
    /// Returns `(name, api_base, model)` tuples for all configured providers.
    /// Used by `supernode_handlers::supernode_health()` to issue HTTP HEAD pings
    /// without consuming LLM API quota.
    ///
    /// ## v2.5.0+Audit Fix 2
    /// Added to replace the invalid `CognitiveTaskType::CustomPrompt` approach
    /// in the health handler, which referenced a non-existent enum variant.
    pub fn provider_configs(&self) -> Vec<(String, String, String)> {
        self.provider_order.iter()
            .filter_map(|name| {
                self.provider_meta.get(name).map(|meta| {
                    (name.clone(), meta.api_base.clone(), meta.model.clone())
                })
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
    use crate::config_supernode::{CognitiveTaskType, TaskRoutingConfig};

    #[test]
    fn test_estimate_cost_known_model() {
        // deepseek-chat: $0.07/M input, $1.10/M output
        let cost = LlmRouter::estimate_cost("deepseek-chat", 1_000_000, 1_000_000, 0);
        let expected = (0.07 + 1.10) / 1.0; // per 1M tokens
        assert!((cost - expected).abs() < 0.001, "cost={} expected={}", cost, expected);
    }

    #[test]
    fn test_estimate_cost_prefix_match() {
        // "deepseek-chat-v3" should prefix-match "deepseek-chat"
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
        // 1000 input (500 cached): billable_input = 500, cached = 500
        // deepseek-reasoner: $0.55/M input, $0.14/M cached, $2.19/M output
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
        use crate::services::memchain::{OpenAiCompatProvider, LlmProvider};

        // Minimal stub provider for testing
        struct StubProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StubProvider {
            fn name(&self) -> &str { "stub" }
            fn is_healthy(&self) -> bool { true }
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
            fallback: Some("stub".into()),
        };

        let router = LlmRouter::new(
            vec![("stub".into(), "http://localhost".into(), "test-model".into(), Arc::new(StubProvider))],
            routing,
        );

        assert_eq!(router.provider_count(), 1);
        // session_title should be routed to "stub"
        assert_eq!(router.routing.get("session_title").map(|s| s.as_str()), Some("stub"));
        // conflict_resolution falls back to fallback provider → also "stub"
        assert_eq!(router.routing.get("conflict_resolution").map(|s| s.as_str()), Some("stub"));
    }

    #[test]
    fn test_provider_configs_returns_all() {
        struct StubProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StubProvider {
            fn name(&self) -> &str { "stub" }
            fn is_healthy(&self) -> bool { true }
            async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
                Err(LlmError::NotConfigured("stub".into()))
            }
        }

        let router = LlmRouter::new(
            vec![
                ("deepseek".into(), "https://api.deepseek.com/v1".into(), "deepseek-chat".into(), Arc::new(StubProvider)),
                ("ollama".into(), "http://localhost:11434/v1".into(), "llama3.2".into(), Arc::new(StubProvider)),
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
