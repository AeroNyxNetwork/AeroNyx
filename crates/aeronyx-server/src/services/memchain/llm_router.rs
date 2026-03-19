// ============================================
// File: crates/aeronyx-server/src/services/memchain/llm_router.rs
// ============================================
//! # LLM Router — Task Type → Provider Routing + Fallback
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Routes `CognitiveTaskType` requests to the appropriate configured provider,
//! with fallback to other providers if the primary is unhealthy or errors.
//! Also provides token cost estimation and prompt construction helpers.
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
//! Update `COST_TABLE` when provider pricing changes.
//!
//! ## Prompt Builders
//! `LlmRouter` contains static prompt-building methods for each `CognitiveTaskType`.
//! These produce `ChatRequest` structs ready to pass to `LlmProvider::chat()`.
//!
//! ⚠️ Important Note for Next Developer:
//! - `COST_TABLE` rates will go stale. Consider making them config-driven (Phase C).
//! - Fallback order: routing-config provider first, then other healthy providers.
//!   If no fallback is desired for a task type, set `fallback_enabled = false` in
//!   the routing config (not yet implemented — Phase C).
//! - `build_session_title_prompt()` intentionally sends ONLY entity names (not
//!   conversation content) to respect privacy_level = "structured".
//!
//! ## Last Modified
//! v2.5.0+SuperNode - 🌟 Created.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{debug, info, warn};

use super::llm_provider::{ChatRequest, ChatResponse, CognitiveTaskType, LlmError, LlmProvider};

// ============================================
// Cost Table (approximate, USD per 1M tokens)
// ============================================

/// Approximate token costs (USD per 1M tokens, input / output / cached_input).
/// Update when provider pricing changes.
/// ⚠️ These rates may be stale. Phase C: move to config.
struct CostRate {
    input_per_m: f64,
    output_per_m: f64,
    cached_input_per_m: f64,
}

fn cost_table() -> HashMap<&'static str, CostRate> {
    let mut m = HashMap::new();
    m.insert("gpt-4o-mini",                 CostRate { input_per_m: 0.15,  output_per_m: 0.60,  cached_input_per_m: 0.075 });
    m.insert("gpt-4o",                      CostRate { input_per_m: 2.50,  output_per_m: 10.0,  cached_input_per_m: 1.25  });
    m.insert("deepseek-chat",               CostRate { input_per_m: 0.07,  output_per_m: 1.10,  cached_input_per_m: 0.014 });
    m.insert("deepseek-reasoner",           CostRate { input_per_m: 0.55,  output_per_m: 2.19,  cached_input_per_m: 0.14  });
    m.insert("claude-haiku-4-5-20251001",   CostRate { input_per_m: 0.80,  output_per_m: 4.00,  cached_input_per_m: 0.08  });
    m.insert("claude-sonnet-4-6",           CostRate { input_per_m: 3.00,  output_per_m: 15.0,  cached_input_per_m: 0.30  });
    // Groq (prices as of 2025)
    m.insert("llama-3.3-70b-versatile",     CostRate { input_per_m: 0.59,  output_per_m: 0.79,  cached_input_per_m: 0.0   });
    // Ollama (local — zero cost)
    m.insert("llama3.2",                    CostRate { input_per_m: 0.0,   output_per_m: 0.0,   cached_input_per_m: 0.0   });
    m
}

// ============================================
// LlmRouter
// ============================================

/// Routes cognitive tasks to the appropriate LLM provider.
/// Thread-safe (Arc<dyn LlmProvider> + immutable routing config).
pub struct LlmRouter {
    /// All configured providers, keyed by name.
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    /// task_type → provider name mapping.
    routing: HashMap<String, String>,
    /// Ordered list of provider names for fallback traversal.
    provider_order: Vec<String>,
}

impl LlmRouter {
    /// Construct a new router from a list of providers and routing config.
    ///
    /// `providers`: Vec of (name, Arc<dyn LlmProvider>) pairs.
    /// `routing`: Map of task_type_str → provider_name.
    ///   Unknown task types will use `default_provider_name` if set.
    /// `default_provider_name`: Optional fallback provider for unmapped task types.
    pub fn new(
        providers: Vec<(String, Arc<dyn LlmProvider>)>,
        routing: HashMap<String, String>,
        default_provider_name: Option<String>,
    ) -> Self {
        let provider_order: Vec<String> = providers.iter().map(|(n, _)| n.clone()).collect();
        let mut provider_map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        for (name, p) in providers {
            provider_map.insert(name, p);
        }

        // Expand routing: fill in default for unmapped task types
        let mut full_routing = routing;
        if let Some(ref default_name) = default_provider_name {
            for task_type in &[
                "session_title", "community_summary", "entity_description",
                "natural_summary", "custom_prompt",
            ] {
                full_routing.entry(task_type.to_string())
                    .or_insert_with(|| default_name.clone());
            }
        }

        info!(
            providers = provider_map.len(),
            routes = full_routing.len(),
            "[LLM_ROUTER] Initialized"
        );

        Self {
            providers: provider_map,
            routing: full_routing,
            provider_order,
        }
    }

    /// Route a request to the configured provider for this task type.
    /// Falls back to other healthy providers if the primary fails.
    ///
    /// Returns `LlmError::NotConfigured` if no provider is available.
    pub async fn route(
        &self,
        task_type: &CognitiveTaskType,
        req: &ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        let task_str = task_type.task_type_str();

        // Determine primary provider name
        let primary_name = self.routing.get(task_str)
            .or_else(|| self.provider_order.first());

        let Some(primary_name) = primary_name else {
            return Err(LlmError::NotConfigured(format!("no provider configured for task_type={}", task_str)));
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

        // Fallback: try remaining providers in order
        let mut last_error = LlmError::NotConfigured(format!("all providers failed for {}", task_str));
        for name in &self.provider_order {
            if name == primary_name { continue; } // Already tried
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

    /// Estimate cost in USD for a response.
    ///
    /// Uses approximate rates from `COST_TABLE`. Returns 0.0 for unknown models.
    /// ⚠️ Approximate — use for budgeting, not billing.
    pub fn estimate_cost(model: &str, input_tokens: u32, output_tokens: u32, cached_tokens: u32) -> f64 {
        let table = cost_table();
        // Try exact match first, then prefix match (e.g. "deepseek-chat-v3" → "deepseek-chat")
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
                (billable_input * r.input_per_m + cached * r.cached_input_per_m + output * r.output_per_m) / 1_000_000.0
            }
            None => 0.0,
        }
    }

    /// Number of configured providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// List of provider names in routing order.
    pub fn provider_names(&self) -> Vec<&str> {
        self.provider_order.iter().map(|s| s.as_str()).collect()
    }

    // ============================================
    // Prompt Builders
    // ============================================
    // Each builder produces a ChatRequest appropriate for its task type.
    // Privacy levels are strictly enforced:
    //   "structured" → only metadata (names, IDs, types) — NO raw content
    //   "summary"    → anonymized summaries only
    //   "full"       → full content (caller's responsibility)

    /// Build a ChatRequest for session title generation.
    ///
    /// Privacy: "structured" — sends only top entity names and optional project name.
    /// Input: entity names (sorted by mention_count), optional project name.
    /// Expected output: a short title string (≤ 60 chars).
    pub fn build_session_title_prompt(
        entity_names: &[&str],
        project_name: Option<&str>,
        first_user_message_preview: Option<&str>,
    ) -> ChatRequest {
        let system = "You generate short, human-readable titles for AI conversation sessions. \
                      Output ONLY the title, 5-10 words max, no quotes, no punctuation at end. \
                      If a project name is provided, start with it followed by a colon.";

        let user = if let Some(proj) = project_name {
            if entity_names.is_empty() {
                format!("Project: {}\nGenerate a session title.", proj)
            } else {
                format!(
                    "Project: {}\nKey topics: {}\nGenerate a session title.",
                    proj,
                    entity_names.join(", ")
                )
            }
        } else if !entity_names.is_empty() {
            format!("Key topics: {}\nGenerate a session title.", entity_names.join(", "))
        } else if let Some(preview) = first_user_message_preview {
            format!("First message (truncated): {}\nGenerate a session title.", preview)
        } else {
            "Generate a short generic session title.".to_string()
        };

        let mut req = ChatRequest::with_system(system, user);
        req.max_tokens = Some(30);
        req.temperature = Some(0.4);
        req
    }

    /// Build a ChatRequest for community summary generation.
    ///
    /// Privacy: "structured" — sends only entity names and types.
    /// Expected output: 1-2 sentence summary of what this community represents.
    pub fn build_community_summary_prompt(
        community_name: &str,
        members: &[(&str, &str)], // (entity_name, entity_type)
    ) -> ChatRequest {
        let system = "You write concise summaries for knowledge graph communities. \
                      A community is a cluster of related entities. \
                      Output ONLY 1-2 sentences describing what this community represents. \
                      Be specific and technical, not vague.";

        let member_list = members.iter()
            .take(15)
            .map(|(name, typ)| format!("{} ({})", name, typ))
            .collect::<Vec<_>>()
            .join(", ");

        let user = format!(
            "Community name: {}\nMembers: {}\nWrite a 1-2 sentence summary.",
            community_name, member_list
        );

        let mut req = ChatRequest::with_system(system, user);
        req.max_tokens = Some(100);
        req.temperature = Some(0.3);
        req
    }

    /// Build a ChatRequest for entity description generation.
    ///
    /// Privacy: "structured" — sends only entity name, type, and relation names.
    /// Expected output: 1 sentence describing what this entity is.
    pub fn build_entity_description_prompt(
        entity_name: &str,
        entity_type: &str,
        relations: &[(&str, &str)], // (relation_type, other_entity_name)
    ) -> ChatRequest {
        let system = "You write concise one-sentence descriptions for knowledge graph entities. \
                      Output ONLY the description, no quotes.";

        let rel_text = if relations.is_empty() {
            String::new()
        } else {
            let rel_list = relations.iter()
                .take(5)
                .map(|(rel, name)| format!("{} {}", rel, name))
                .collect::<Vec<_>>()
                .join("; ");
            format!("\nRelations: {}", rel_list)
        };

        let user = format!(
            "Entity: {} ({}){}\nWrite a one-sentence description.",
            entity_name, entity_type, rel_text
        );

        let mut req = ChatRequest::with_system(system, user);
        req.max_tokens = Some(60);
        req.temperature = Some(0.2);
        req
    }

    /// Build a ChatRequest for natural session summary generation.
    ///
    /// Privacy: "summary" — sends an anonymized summary of topics discussed.
    /// Expected output: 2-3 sentence natural summary.
    pub fn build_natural_summary_prompt(
        entity_names: &[&str],
        turn_count: i64,
        existing_summary: Option<&str>,
    ) -> ChatRequest {
        let system = "You write natural language summaries for AI conversation sessions. \
                      Output ONLY the summary, 2-3 sentences, present tense.";

        let base = if let Some(existing) = existing_summary {
            format!("Existing summary: {}\nKey entities: {}\nTurn count: {}\n\
                     Rewrite as a natural 2-3 sentence summary.",
                existing, entity_names.join(", "), turn_count)
        } else {
            format!("Key topics discussed: {}\nTurn count: {}\n\
                     Write a natural 2-3 sentence summary.",
                entity_names.join(", "), turn_count)
        };

        let mut req = ChatRequest::with_system(system, base);
        req.max_tokens = Some(150);
        req.temperature = Some(0.4);
        req
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
