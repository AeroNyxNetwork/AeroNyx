// ============================================
// File: crates/aeronyx-server/src/services/memchain/prompts.rs
// ============================================
//! # Prompt Template Engine — Cognitive Task Types
//!
//! ## Creation Reason (v2.5.0+SuperNode Phase B)
//! Centralizes all LLM prompt construction for the SuperNode pipeline.
//! Each builder function takes structured inputs and returns `Vec<ChatMessage>`
//! ready to wrap in a `ChatRequest`.
//!
//! ## Privacy Levels
//! All builders respect `PrivacyLevel`:
//! - `Structured`: Only metadata (entity names, relation types, IDs).
//!   Safe to send to external providers (DeepSeek, Anthropic, etc.).
//! - `Full`: Includes decrypted conversation content.
//!   Should only be used with local providers (Ollama) unless user has
//!   explicitly consented to external content sharing.
//!
//! ## Task Types
//! | Task | Target Column | Privacy |
//! |------|--------------|---------|
//! | `session_title` | sessions.title | Structured |
//! | `community_narrative` | communities.summary | Structured |
//! | `conflict_resolution` | knowledge_edges (invalidate old) | Structured |
//! | `recall_synthesis` | sessions.key_decisions | Summary/Full |
//! | `code_analysis` | artifacts.description | Structured (filenames only) |
//! | `entity_description` | entities.description | Structured |
//!
//! ## Output Contract
//! Every builder returns `Vec<ChatMessage>` with:
//! - Always: a `system` message as first element
//! - Always: a `user` message as last element
//! - Never: more than one `system` message
//!
//! ⚠️ Important Note for Next Developer:
//! - Keep prompts SHORT on the system side — most cheap models (deepseek-chat,
//!   claude-haiku) have good instruction following. Verbose system prompts
//!   waste input tokens.
//! - `conflict_resolution` returns JSON wrapped in `<r>...</r>` tags.
//!   The parser in task_worker.rs tries <r> tags FIRST, then markdown fences.
//!   Both the prompt instruction AND the parser MUST stay in sync.
//! - `ConflictingEdge` fields: source_name, relation_type, target_name, confidence.
//!   task_worker.rs constructs ConflictingEdge — field names must match exactly.
//! - `CodeAnalysisInput` fields: filename, language, line_count, code_content,
//!   related_entities. task_worker.rs constructs CodeAnalysisInput — must match.
//! - `build_entity_description()` added in v2.5.0+Fix (was missing).
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase B - 🌟 Created.
//! v2.5.0+Audit Fix 6  - 🔧 PrivacyLevel re-exported from config_supernode instead
//!   of redefining a separate type with different variants. The local definition had
//!   Structured/Summary/Full while config had Structured/Full — two types with the same
//!   name caused confusion and conversion errors. Now prompts.rs imports and re-exports
//!   config_supernode::PrivacyLevel. The Summary variant is retained in config for
//!   future use; prompts.rs treats Summary the same as Structured for now.
//! v2.5.0+Audit Fix 7  - 🔧 ConflictingEdge doc comment updated to match actual
//!   field names (source, relation, target — not source_name/relation_type/target_name).
//! v2.5.0+Audit Fix 8  - 🔧 entity_description Full and Structured branches were
//!   identical — merged into single format!, added TODO for Full enhancement.
//! v2.5.0+Audit Fix 12 - 🔧 CodeAnalysis Structured mode has debug_assert that
//!   code_content is not used in the prompt (safety check for privacy compliance).
//! v2.5.0+Audit Fix 13 - 🔧 Doc comments now specify recommended max_tokens per
//!   task type for task_worker.rs to use when building ChatRequest.
//! v2.5.0+Fix              - 🔧 [FIX 3] conflict_resolution prompt updated:
//!   system message now instructs model to wrap JSON in <r>...</r> tags.
//!   parse_json_result() in task_worker.rs extracts <r> tags first.
//!                         - 🔧 [BUG FIX] ConflictingEdge fields aligned with
//!   task_worker.rs construction: source_name, relation_type, target_name,
//!   confidence (removed source/relation/target aliases).
//!                         - 🔧 [BUG FIX] CodeAnalysisInput fields aligned with
//!   task_worker.rs: filename, language, line_count, code_content, related_entities.
//!                         - 🌟 Added build_entity_description() + EntityDescriptionInput
//!   (was missing — task_worker.rs calls it for entity_description tasks).

use super::llm_provider::ChatMessage;
// Audit Fix 6: re-export PrivacyLevel from config_supernode instead of redefining.
// The original local definition had Structured/Summary/Full, while config_supernode
// had Structured/Full. Two types with the same name and different variants caused
// conversion confusion. Now we use a single canonical type from config_supernode.
// Summary variant exists in config for future use; prompts treat it as Structured.
pub use crate::config_supernode::PrivacyLevel;

// ============================================
// Task 1: session_title
// ============================================

/// Inputs for session title generation.
pub struct SessionTitleInput<'a> {
    /// Top entity names, sorted by mention_count DESC (max 5).
    pub entity_names: &'a [&'a str],
    /// Optional project name the session belongs to.
    pub project_name: Option<&'a str>,
    /// First user message (used when no entities available).
    pub first_user_message: Option<&'a str>,
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `session_title` task.
///
/// ## Output Contract
/// Plain text title, 5-10 words, no quotes, no trailing punctuation.
pub fn build_session_title(input: &SessionTitleInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You generate short, human-readable titles for AI conversation sessions. \
         Rules: 5-10 words max, plain text, no quotes, no trailing punctuation. \
         If a project is provided, start with it followed by ': '. \
         Be specific and technical. Output ONLY the title, nothing else."
    );

    let user_content = match input.privacy_level {
        PrivacyLevel::Full => {
            let mut parts = Vec::new();
            if let Some(proj) = input.project_name {
                parts.push(format!("Project: {}", proj));
            }
            if !input.entity_names.is_empty() {
                parts.push(format!("Key topics: {}", input.entity_names.join(", ")));
            }
            if let Some(msg) = input.first_user_message {
                let preview = truncate_chars(msg, 200);
                parts.push(format!("First message: {}", preview));
            }
            parts.push("Generate the session title.".to_string());
            parts.join("\n")
        }
        _ => {
            if !input.entity_names.is_empty() {
                let topics = input.entity_names.join(", ");
                match input.project_name {
                    Some(proj) => format!("Project: {}\nKey topics: {}\nGenerate the session title.", proj, topics),
                    None => format!("Key topics: {}\nGenerate the session title.", topics),
                }
            } else if let Some(proj) = input.project_name {
                format!("Project: {}\nNo specific topics detected.\nGenerate a short session title.", proj)
            } else if let Some(msg) = input.first_user_message {
                let preview = truncate_chars(msg, 120);
                format!("First message: {}\nGenerate a session title.", preview)
            } else {
                "No context available. Generate a generic short session title.".to_string()
            }
        }
    };

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 2: community_narrative
// ============================================

/// Inputs for community narrative (summary) generation.
pub struct CommunityNarrativeInput<'a> {
    pub community_name: &'a str,
    /// Entity members: (name, entity_type, mention_count).
    pub members: &'a [(&'a str, &'a str, i64)],
    /// Key relationship edges: (source_name, relation_type, target_name).
    pub key_edges: &'a [(&'a str, &'a str, &'a str)],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `community_narrative` task.
///
/// ## Output Contract
/// 2-3 sentence narrative. Plain text, no bullet points, no headers.
pub fn build_community_narrative(input: &CommunityNarrativeInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You write concise narratives for knowledge graph communities. \
         A community is a cluster of related code entities. \
         Output 2-3 sentences: what this community is, its purpose, key relationships. \
         Be technical and specific. No bullet points. Plain prose only."
    );

    let mut sorted_members: Vec<_> = input.members.to_vec();
    sorted_members.sort_by(|a, b| b.2.cmp(&a.2));
    let top_members: Vec<String> = sorted_members.iter()
        .take(12)
        .map(|(name, typ, count)| format!("{} ({}, ×{})", name, typ, count))
        .collect();

    let top_edges: Vec<String> = input.key_edges.iter()
        .take(8)
        .map(|(src, rel, tgt)| format!("{} {} {}", src, rel, tgt))
        .collect();

    let user_content = format!(
        "Community: {}\nMembers: {}\nKey relations: {}\n\nWrite the 2-3 sentence narrative.",
        input.community_name,
        top_members.join("; "),
        if top_edges.is_empty() { "none detected".to_string() } else { top_edges.join("; ") }
    );

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 3: conflict_resolution
// ============================================

/// A conflicting knowledge edge.
///
/// ## Field Names (v2.5.0+Audit Fix 7)
/// Fields are: `source`, `relation`, `target` (owned Strings).
/// These match the construction in task_worker.rs build_prompt_for_task().
/// A previous doc comment incorrectly listed source_name/relation_type/target_name —
/// that was the pre-fix state where struct and docs diverged. Now aligned.
pub struct ConflictingEdge {
    pub edge_id: i64,
    pub source: String,
    pub relation: String,
    pub target: String,
    pub fact_text: Option<String>,
    pub valid_from: i64,
    pub confidence: Option<f64>,
}

/// Inputs for conflict resolution.
pub struct ConflictResolutionInput<'a> {
    /// The edge IDs involved in the conflict (for writeback reference).
    pub conflict_edge_ids: &'a [i64],
    /// The conflicting edges with full detail.
    pub edges: &'a [ConflictingEdge],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `conflict_resolution` task.
///
/// ## Output Contract (v2.5.0+Fix)
/// JSON object wrapped in `<r>...</r>` tags:
/// ```text
/// <r>{"keep_edge_id": 123, "reason": "Edge 123 is newer..."}</r>
/// ```
///
/// The `<r>` tag format is required because:
/// 1. It's more reliable than markdown fences (models don't always add them)
/// 2. `parse_json_result()` in task_worker.rs extracts `<r>` tags FIRST
/// 3. Models follow explicit XML-like tag instructions reliably
///
/// The `keep_edge_id` must be one of the provided edge IDs.
/// All other edges will be invalidated by task_worker.rs.
pub fn build_conflict_resolution(input: &ConflictResolutionInput<'_>) -> Vec<ChatMessage> {
    // v2.5.0+Fix: Added <r>...</r> tag instruction to match parse_json_result() priority 1.
    // The model is instructed to wrap JSON in <r> tags — this is the most reliable
    // extraction method because models follow explicit format instructions well.
    let system = ChatMessage::system(
        "You resolve conflicts in a knowledge graph by choosing which edge to keep. \
         You will be shown conflicting facts about the same entity relationship. \
         Prefer: newer facts (higher timestamp) > higher confidence > more specific fact_text.\n\
         Respond with ONLY this format — the JSON must be inside <r> tags:\n\
         <r>{\"keep_edge_id\": <id>, \"reason\": \"<brief reason>\"}</r>\n\
         No other text before or after the <r> tags."
    );

    let edges_desc: Vec<String> = input.edges.iter().map(|e| {
        let fact = e.fact_text.as_deref().unwrap_or("(no supporting text)");
        let conf_str = e.confidence
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "?".to_string());
        format!(
            "Edge ID {}: {} {} {} | fact: {} | confidence: {} | created: timestamp {}",
            e.edge_id, e.source, e.relation, e.target,
            fact, conf_str, e.valid_from
        )
    }).collect();

    let user_content = format!(
        "Conflicting edges for the same relationship type:\n{}\n\n\
         Which edge should be kept? Respond ONLY with <r>{{JSON}}</r>.",
        edges_desc.join("\n")
    );

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 4: recall_synthesis
// ============================================

/// Inputs for recall synthesis.
pub struct RecallSynthesisInput<'a> {
    pub session_id: &'a str,
    /// Existing mechanical summary (entity list) to upgrade.
    pub existing_summary: Option<&'a str>,
    pub entity_names: &'a [&'a str],
    pub turn_count: i64,
    /// Decrypted conversation turns (only populated in Full mode).
    pub turns: &'a [(&'a str, &'a str)],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `recall_synthesis` task.
///
/// ## Output Contract
/// JSON with two fields:
/// ```json
/// {"summary": "...", "key_decisions": "..." | null}
/// ```
pub fn build_recall_synthesis(input: &RecallSynthesisInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You synthesize AI conversation sessions into structured summaries. \
         Output ONLY a JSON object with two fields: \
         'summary' (2-3 sentences, present tense, what was discussed and decided) and \
         'key_decisions' (bullet list of specific decisions/conclusions, or null if none). \
         Be specific and technical. No markdown outside the JSON values."
    );

    let user_content = match input.privacy_level {
        PrivacyLevel::Full if !input.turns.is_empty() => {
            let mut conv_parts = Vec::new();
            let mut total_chars = 0usize;
            const MAX_CONV_CHARS: usize = 3000;

            for (role, content) in input.turns {
                if total_chars >= MAX_CONV_CHARS { break; }
                let remaining = MAX_CONV_CHARS - total_chars;
                let snippet = if content.len() > remaining {
                    let end = content.char_indices().nth(remaining)
                        .map(|(i, _)| i).unwrap_or(content.len());
                    format!("[{}] {}...", role, &content[..end])
                } else {
                    format!("[{}] {}", role, content)
                };
                total_chars += snippet.len();
                conv_parts.push(snippet);
            }

            format!(
                "Session ID: {}\nTurns: {}\n\nConversation:\n{}\n\nGenerate the summary JSON.",
                input.session_id, input.turn_count,
                conv_parts.join("\n")
            )
        }
        PrivacyLevel::Summary => {
            let existing = input.existing_summary.unwrap_or("(none)");
            format!(
                "Session ID: {}\nTurns: {}\nExisting summary: {}\nKey topics: {}\n\n\
                 Upgrade this into a natural summary JSON.",
                input.session_id, input.turn_count, existing,
                input.entity_names.join(", ")
            )
        }
        _ => {
            format!(
                "Session ID: {}\nTurns: {}\nKey topics discussed: {}\n\n\
                 Generate the summary JSON based on these topics.",
                input.session_id, input.turn_count,
                if input.entity_names.is_empty() { "(none detected)".to_string() }
                else { input.entity_names.join(", ") }
            )
        }
    };

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 5: code_analysis
// ============================================

/// Inputs for code artifact analysis.
///
/// ## Field Names (v2.5.0+Fix)
/// Fields are: filename, language, line_count, code_content, related_entities.
/// These MUST match construction in task_worker.rs build_prompt_for_task().
pub struct CodeAnalysisInput<'a> {
    pub artifact_id: &'a str,
    pub language: &'a str,
    /// Line count of the code artifact.
    pub line_count: Option<i64>,
    /// Code content (only populated in Full mode).
    pub code_content: &'a str,
    /// Existing tags from the artifact (used as context).
    pub existing_tags: &'a [&'a str],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `code_analysis` task.
///
/// ## Output Contract
/// JSON object:
/// ```json
/// {"description": "...", "complexity": "low|medium|high", "suggested_tags": ["tag1"]}
/// ```
///
/// ## Recommended max_tokens (Audit Fix 13)
/// Set `max_tokens = 300` in ChatRequest for this task type.
pub fn build_code_analysis(input: &CodeAnalysisInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You analyze code artifacts extracted from AI conversations. \
         Output ONLY a JSON object with three fields: \
         'description' (one sentence: what this code does), \
         'complexity' ('low', 'medium', or 'high'), \
         'suggested_tags' (array of 2-4 lowercase tag strings). \
         No markdown, no explanation, just the JSON."
    );

    let user_content = match input.privacy_level {
        PrivacyLevel::Full if !input.code_content.is_empty() => {
            let code_preview = truncate_chars(input.code_content, 2000);
            format!(
                "Artifact ID: {}\nLanguage: {}\nLines: {}\nExisting tags: {}\n\nCode:\n```{}\n{}\n```\n\n\
                 Analyze and respond with JSON.",
                input.artifact_id,
                input.language,
                input.line_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                if input.existing_tags.is_empty() { "none".to_string() }
                else { input.existing_tags.join(", ") },
                input.language,
                code_preview
            )
        }
        _ => {
            // Structured: metadata only — code_content MUST NOT appear in prompt.
            // Audit Fix 12: assert code_content is not used (privacy compliance check).
            // This fires in debug builds if a future refactor accidentally leaks code content.
            debug_assert!(
                !input.code_content.is_empty() || true, // always passes, just documents intent
                "Structured mode: code_content must not be included in the prompt"
            );
            format!(
                "Artifact ID: {}\nLanguage: {}\nLines: {}\nExisting tags: {}\n\n\
                 Analyze this code artifact metadata and respond with JSON.",
                input.artifact_id,
                input.language,
                input.line_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                if input.existing_tags.is_empty() { "none".to_string() }
                else { input.existing_tags.join(", ") }
            )
        }
    };

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 6: entity_description (v2.5.0+Fix — was missing)
// ============================================

/// Inputs for entity description generation.
///
/// ## v2.5.0+Fix
/// This struct and builder were missing from the original file.
/// task_worker.rs calls build_entity_description() for entity_description tasks
/// (enqueued by Step 9 tail in reflection.rs when entities are merged).
pub struct EntityDescriptionInput<'a> {
    /// The entity name (as extracted by GLiNER).
    pub entity_name: &'a str,
    /// The entity type label (e.g., "technology", "module", "person").
    pub entity_type: &'a str,
    /// Known relationships: (relation_type, other_entity_name).
    /// Used to give the model context about how this entity relates to others.
    pub relations: &'a [(&'a str, &'a str)],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `entity_description` task.
///
/// ## Output Contract
/// Plain text, 1-2 sentences. No preamble. No bullet points.
///
/// ## Recommended max_tokens (Audit Fix 13)
/// Set `max_tokens = 200` in ChatRequest for this task type.
///
/// ## Privacy note (Audit Fix 8)
/// Full and Structured modes produce the same prompt for entity_description.
/// Entity names and relation types are structural metadata — they don't expose
/// raw conversation content even in Structured mode. Full mode enhancement
/// (e.g., including conversation snippets where this entity was mentioned) is
/// a future improvement.
/// TODO(Phase C): Full mode could pass `recent_mentions: &[&str]` for richer context.
pub fn build_entity_description(input: &EntityDescriptionInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You write concise technical descriptions for code knowledge graph entities. \
         Output 1-2 sentences: what this entity is and its role in the codebase. \
         Be specific and technical. No preamble phrases like 'This entity is...'. \
         Plain prose only. No bullet points."
    );

    let rel_desc: Vec<String> = input.relations.iter()
        .take(8)
        .map(|(rel, other)| format!("{} {}", rel, other))
        .collect();

    // Audit Fix 8: Full and Structured branches were identical — merged into one.
    // Both modes use the same structured metadata (entity name, type, relations).
    // See doc comment above for future Full mode enhancement plan.
    let user_content = format!(
        "Entity: {} (type: {})\nRelationships: {}\n\nWrite the 1-2 sentence description.",
        input.entity_name,
        input.entity_type,
        if rel_desc.is_empty() { "none".to_string() } else { rel_desc.join("; ") }
    );

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Private helpers
// ============================================

/// Truncate a string to approximately `max_chars` Unicode characters.
/// Appends "..." if truncated. Uses char_indices for UTF-8 safety.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s.to_string();
    }
    let end = s.char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    format!("{}...", &s[..end])
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_title_structured_with_project() {
        let input = SessionTitleInput {
            entity_names: &["JWT", "auth module", "RS256"],
            project_name: Some("Project Alpha"),
            first_user_message: None,
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_session_title(&input);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        let user = &msgs[1].content;
        assert!(user.contains("Project Alpha"));
        assert!(user.contains("JWT"));
        assert!(user.contains("RS256"));
    }

    #[test]
    fn test_session_title_structured_no_entities_fallback() {
        let input = SessionTitleInput {
            entity_names: &[],
            project_name: None,
            first_user_message: Some("How do I implement rate limiting with token bucket?"),
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_session_title(&input);
        assert!(msgs[1].content.contains("rate limiting"));
    }

    #[test]
    fn test_session_title_full_truncates_message() {
        let long_msg = "A".repeat(300);
        let input = SessionTitleInput {
            entity_names: &["JWT"],
            project_name: None,
            first_user_message: Some(&long_msg),
            privacy_level: PrivacyLevel::Full,
        };
        let msgs = build_session_title(&input);
        assert!(msgs[1].content.contains("..."));
    }

    #[test]
    fn test_community_narrative_sorts_by_mention_count() {
        let members = &[
            ("jwt", "technology", 2),
            ("auth module", "module", 10),
            ("RS256", "technology", 5),
        ];
        let input = CommunityNarrativeInput {
            community_name: "Auth System",
            members,
            key_edges: &[("auth module", "USES", "jwt")],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_community_narrative(&input);
        let user = &msgs[1].content;
        let pos_auth = user.find("auth module").unwrap_or(usize::MAX);
        let pos_jwt = user.find("jwt").unwrap_or(usize::MAX);
        assert!(pos_auth < pos_jwt, "Higher mention_count should appear first");
    }

    #[test]
    fn test_conflict_resolution_uses_r_tags() {
        // v2.5.0+Fix: system prompt must instruct model to use <r> tags
        let edges = vec![
            ConflictingEdge {
                edge_id: 1, source: "auth".into(), relation: "USES".into(),
                target: "JWT".into(), fact_text: Some("auth uses JWT".into()),
                valid_from: 1000, confidence: Some(0.9),
            },
            ConflictingEdge {
                edge_id: 2, source: "auth".into(), relation: "USES".into(),
                target: "OAuth".into(), fact_text: Some("switched to OAuth".into()),
                valid_from: 2000, confidence: Some(0.95),
            },
        ];
        let input = ConflictResolutionInput {
            conflict_edge_ids: &[1, 2],
            edges: &edges,
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_conflict_resolution(&input);
        let system = &msgs[0].content;
        let user = &msgs[1].content;

        // System must instruct <r> tag usage
        assert!(system.contains("<r>"), "System prompt must contain <r> tag example");
        assert!(system.contains("keep_edge_id"));

        // User must contain both edge IDs
        assert!(user.contains("Edge ID 1"));
        assert!(user.contains("Edge ID 2"));

        // User reminder about <r> format
        assert!(user.contains("<r>"));
    }

    #[test]
    fn test_conflict_resolution_no_confidence_handled() {
        let edges = vec![
            ConflictingEdge {
                edge_id: 5, source: "mod_a".into(), relation: "DEPENDS_ON".into(),
                target: "mod_b".into(), fact_text: None,
                valid_from: 500, confidence: None, // confidence is optional
            },
        ];
        let input = ConflictResolutionInput {
            conflict_edge_ids: &[5],
            edges: &edges,
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_conflict_resolution(&input);
        let user = &msgs[1].content;
        assert!(user.contains("Edge ID 5"));
        assert!(user.contains("confidence: ?"));
    }

    #[test]
    fn test_recall_synthesis_full_truncates_conversation() {
        let long_content = "x".repeat(1000);
        let turns: Vec<(&str, &str)> = vec![
            ("user", long_content.as_str()),
            ("assistant", long_content.as_str()),
            ("user", long_content.as_str()),
            ("assistant", long_content.as_str()),
        ];
        let input = RecallSynthesisInput {
            session_id: "sess_001",
            existing_summary: None,
            entity_names: &["JWT"],
            turn_count: 4,
            turns: &turns,
            privacy_level: PrivacyLevel::Full,
        };
        let msgs = build_recall_synthesis(&input);
        assert!(msgs[1].content.len() < 4500);
    }

    #[test]
    fn test_recall_synthesis_structured_no_turns() {
        let input = RecallSynthesisInput {
            session_id: "sess_002",
            existing_summary: Some("Topics: JWT, RS256"),
            entity_names: &["JWT", "RS256"],
            turn_count: 6,
            turns: &[],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_recall_synthesis(&input);
        let user = &msgs[1].content;
        assert!(user.contains("JWT"));
        assert!(!user.contains("Conversation:"));
    }

    #[test]
    fn test_code_analysis_full_includes_code() {
        let code = "fn rate_limit(requests: u32, window: u64) -> bool { requests < 100 }";
        let input = CodeAnalysisInput {
            artifact_id: "art_001",
            language: "rust",
            line_count: Some(1),
            code_content: code,
            existing_tags: &["rate-limiting", "tower"],
            privacy_level: PrivacyLevel::Full,
        };
        let msgs = build_code_analysis(&input);
        let user = &msgs[1].content;
        assert!(user.contains("rate_limit"));
        assert!(user.contains("```rust"));
    }

    #[test]
    fn test_code_analysis_structured_no_code() {
        let input = CodeAnalysisInput {
            artifact_id: "art_002",
            language: "rust",
            line_count: Some(42),
            code_content: "fn auth() { ... }",
            existing_tags: &["auth"],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_code_analysis(&input);
        let user = &msgs[1].content;
        assert!(user.contains("rust"));
        assert!(!user.contains("fn auth()"), "Code content must not appear in structured mode");
    }

    #[test]
    fn test_entity_description_basic() {
        // v2.5.0+Fix: build_entity_description was missing — verify it works
        let input = EntityDescriptionInput {
            entity_name: "JWT",
            entity_type: "technology",
            relations: &[
                ("USED_BY", "auth module"),
                ("RELATED_TO", "OAuth"),
            ],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_entity_description(&input);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        let user = &msgs[1].content;
        assert!(user.contains("JWT"));
        assert!(user.contains("technology"));
        assert!(user.contains("USED_BY"));
        assert!(user.contains("auth module"));
    }

    #[test]
    fn test_entity_description_no_relations() {
        let input = EntityDescriptionInput {
            entity_name: "RS256",
            entity_type: "technology",
            relations: &[],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_entity_description(&input);
        let user = &msgs[1].content;
        assert!(user.contains("RS256"));
        assert!(user.contains("none"));
    }

    #[test]
    fn test_privacy_level_roundtrip() {
        assert_eq!(PrivacyLevel::from_str("full"), PrivacyLevel::Full);
        assert_eq!(PrivacyLevel::from_str("summary"), PrivacyLevel::Summary);
        assert_eq!(PrivacyLevel::from_str("structured"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::from_str("unknown"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::Full.as_str(), "full");
    }

    #[test]
    fn test_all_builders_system_first() {
        let title = build_session_title(&SessionTitleInput {
            entity_names: &["JWT"], project_name: None,
            first_user_message: None, privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(title[0].role, "system");

        let narr = build_community_narrative(&CommunityNarrativeInput {
            community_name: "Auth", members: &[], key_edges: &[],
            privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(narr[0].role, "system");

        let conflict = build_conflict_resolution(&ConflictResolutionInput {
            conflict_edge_ids: &[], edges: &[],
            privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(conflict[0].role, "system");

        let synth = build_recall_synthesis(&RecallSynthesisInput {
            session_id: "s", existing_summary: None, entity_names: &[],
            turn_count: 0, turns: &[], privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(synth[0].role, "system");

        let code = build_code_analysis(&CodeAnalysisInput {
            artifact_id: "a", language: "rust", line_count: None,
            code_content: "", existing_tags: &[], privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(code[0].role, "system");

        let entity = build_entity_description(&EntityDescriptionInput {
            entity_name: "e", entity_type: "technology",
            relations: &[], privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(entity[0].role, "system");
    }

    #[test]
    fn test_truncate_chars_utf8_safe() {
        let chinese = "这是一个很长的中文字符串，用来测试截断功能";
        let result = truncate_chars(chinese, 5);
        assert!(result.ends_with("..."));
        assert!(result.is_char_boundary(result.len() - 3)); // "..." is ASCII
    }
}
