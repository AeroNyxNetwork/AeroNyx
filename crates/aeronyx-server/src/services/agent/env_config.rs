//! ============================================
//! File: crates/aeronyx-server/src/services/agent/env_config.rs
//! Path: aeronyx-server/src/services/agent/env_config.rs
//! ============================================
//!
//! ## Creation Reason
//! Extracted from `agent_manager.rs` to isolate environment variable
//! and OpenClaw configuration management into a focused module.
//!
//! ## Main Functionality
//! - `EnvConfig::load_env_file()`: Parse ~/.openclaw/.env into HashMap
//! - `EnvConfig::ensure_http_api_enabled()`: Enable /v1/chat/completions
//! - `EnvConfig::set_config()`: Set arbitrary openclaw config values
//! - `EnvConfig::write_env_var()`: Write/update a key in .env file
//!
//! ## .env File Format
//! Standard dotenv format:
//! ```text
//! # API Keys for LLM providers
//! XAI_API_KEY=xai-abc123...
//! OPENAI_API_KEY=sk-...
//! ANTHROPIC_API_KEY=sk-ant-...
//! ```
//! - Lines starting with `#` are comments
//! - Empty lines are skipped
//! - Values may be quoted (single or double quotes, stripped on read)
//! - No shell expansion ($VAR substitution)
//!
//! ## ⚠️ Important Note for Next Developer
//! - The .env file is owned by the `openclaw` user — write operations
//!   must use `sudo -u openclaw` or write as the correct user
//! - `ensure_http_api_enabled()` is idempotent — safe to call repeatedly
//! - OpenClaw's `config set` command validates the key path against its
//!   schema. If you try to set an unknown key, it will error.
//! - The HTTP API endpoint MUST be enabled for ws_client.rs to work.
//!   This is the single most common misconfiguration issue.
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! ============================================

use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::{CommandRunner, OPENCLAW_ENV_PATH, OPENCLAW_USER};

// ============================================
// EnvConfig
// ============================================

/// Manages OpenClaw environment variables and configuration.
pub struct EnvConfig;

impl EnvConfig {
    // ============================================
    // .env File Management
    // ============================================

    /// Reads key=value pairs from the OpenClaw .env file.
    ///
    /// Returns a HashMap of environment variable name → value.
    /// Returns an empty map if the file doesn't exist or can't be read.
    ///
    /// ## Parsing Rules
    /// - Lines starting with `#` are comments
    /// - Empty lines are skipped
    /// - Split on first `=` (value may contain `=`)
    /// - Optional quotes around values are stripped
    pub async fn load_env_file() -> HashMap<String, String> {
        let content = match tokio::fs::read_to_string(OPENCLAW_ENV_PATH).await {
            Ok(c) => c,
            Err(_) => {
                debug!("[ENV] No .env file at {} — skipping", OPENCLAW_ENV_PATH);
                return HashMap::new();
            }
        };

        let mut env_vars = HashMap::new();

        for line in content.lines() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim().to_string();
                let value = line[eq_pos + 1..].trim().to_string();

                // Strip optional surrounding quotes
                let value = if (value.starts_with('"') && value.ends_with('"'))
                    || (value.starts_with('\'') && value.ends_with('\''))
                {
                    if value.len() >= 2 {
                        value[1..value.len() - 1].to_string()
                    } else {
                        value
                    }
                } else {
                    value
                };

                if !key.is_empty() {
                    env_vars.insert(key, value);
                }
            }
        }

        if !env_vars.is_empty() {
            debug!(
                count = env_vars.len(),
                "[ENV] Loaded {} env vars from .env",
                env_vars.len()
            );
        }

        env_vars
    }

    /// Writes or updates a key=value pair in the .env file.
    ///
    /// If the key already exists, its value is updated in-place.
    /// If the key doesn't exist, it's appended to the end.
    /// If the .env file doesn't exist, it's created.
    ///
    /// The file is written with mode 600 (owner-only read/write)
    /// for security (API keys are sensitive).
    pub async fn write_env_var(key: &str, value: &str) -> Result<(), String> {
        let mut lines: Vec<String> = match tokio::fs::read_to_string(OPENCLAW_ENV_PATH).await {
            Ok(content) => content.lines().map(|l| l.to_string()).collect(),
            Err(_) => Vec::new(),
        };

        let target_prefix = format!("{}=", key);
        let new_line = format!("{}={}", key, value);

        let mut found = false;
        for line in &mut lines {
            if line.starts_with(&target_prefix) {
                *line = new_line.clone();
                found = true;
                break;
            }
        }

        if !found {
            lines.push(new_line);
        }

        let content = lines.join("\n") + "\n";

        // Write via bash as the openclaw user to ensure correct ownership.
        // We use run_as_user (which already wraps sudo -u openclaw).
        let write_cmd = format!(
            "printf '%s' '{}' > {} && chmod 600 {}",
            content.replace('\'', "'\\''"),  // escape single quotes in content
            OPENCLAW_ENV_PATH,
            OPENCLAW_ENV_PATH
        );

        CommandRunner::run_as_user("bash", &["-c", &write_cmd])
            .await
            .map_err(|e| format!("Failed to write .env file: {}", e))?;

        debug!(key = %key, "[ENV] Written env var to .env");
        Ok(())
    }

    // ============================================
    // OpenClaw Config Management
    // ============================================

    /// Ensures the HTTP Chat Completions API endpoint is enabled.
    ///
    /// This is REQUIRED for `ws_client.rs` to function. The endpoint
    /// is disabled by default in OpenClaw. We enable it via
    /// `openclaw config set` which is idempotent.
    ///
    /// Sets: `gateway.http.endpoints.chatCompletions.enabled = true`
    pub async fn ensure_http_api_enabled() -> Result<(), String> {
        info!("[CONFIG] Ensuring HTTP Chat Completions API is enabled...");

        CommandRunner::run_as_user(
            "openclaw",
            &["config", "set", "gateway.http.endpoints.chatCompletions.enabled", "true"],
        ).await.map_err(|e| format!("Failed to enable HTTP API: {}", e))?;

        info!("[CONFIG] ✅ HTTP Chat Completions API enabled");
        Ok(())
    }

    /// Sets an arbitrary OpenClaw config value via CLI.
    ///
    /// Uses `openclaw config set <key> <value>` which validates the key
    /// against the OpenClaw config schema.
    ///
    /// ## Example
    /// ```ignore
    /// EnvConfig::set_config("agents.defaults.model", "xai/grok-3-mini").await?;
    /// ```
    pub async fn set_config(key: &str, value: &str) -> Result<(), String> {
        CommandRunner::run_as_user(
            "openclaw",
            &["config", "set", key, value],
        ).await.map_err(|e| format!("Failed to set config {}: {}", key, e))
    }

    /// Gets an OpenClaw config value via CLI.
    ///
    /// Returns the value as a string, or None if the key doesn't exist
    /// or the command fails.
    pub async fn get_config(key: &str) -> Option<String> {
        CommandRunner::run_as_user_output(
            "openclaw",
            &["config", "get", key],
        ).await.ok()
    }

    /// Disables memory search (which requires an embedding provider).
    ///
    /// OpenClaw's memory search feature requires an embedding API key
    /// (OpenAI, Gemini, Voyage, or Mistral). If none is configured,
    /// `openclaw doctor` warns about it. Disabling it silences the
    /// warning and prevents errors during agent operation.
    pub async fn disable_memory_search() -> Result<(), String> {
        Self::set_config("agents.defaults.memorySearch.enabled", "false").await
    }
}
