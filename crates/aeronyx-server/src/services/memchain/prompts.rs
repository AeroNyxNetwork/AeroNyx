// ============================================
// File: crates/aeronyx-server/src/services/memchain/prompts.rs
// ============================================
//! # Prompt Template Engine — 5 Cognitive Task Types
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
//! ## Five Task Types
//! | Task | Target Column | Privacy |
//! |------|--------------|---------|
//! | `session_title` | sessions.title | Structured |
//! | `community_narrative` | communities.summary | Structured |
//! | `conflict_resolution` | knowledge_edges (invalidate old) | Structured |
//! | `recall_synthesis` | sessions.key_decisions | Summary/Full |
//! | `code_analysis` | artifacts.description | Structured (filenames only) |
//!
//! ## Output Contract
//! Every builder returns `Vec<ChatMessage>` with:
//! - Always: a `system` message as first element
//! - Always: a `user` message as last element
//! - Never: more than one `system` message
//!
//! The LLM output is expected to be plain text (not JSON) unless the builder
//! doc says otherwise. Parsers in `task_worker.rs` handle extraction.
//!
//! ⚠️ Important Note for Next Developer:
//! - Keep prompts SHORT on the system side — most cheap models (deepseek-chat,
//!   claude-haiku) have good instruction following. Verbose system prompts
//!   waste input tokens.
//! - `conflict_resolution` returns JSON — see its doc for the expected shape.
//!   The parser in task_worker.rs must stay in sync with this contract.
//! - `PrivacyLevel::Full` builders must NEVER be called without checking
//!   `config.memchain.supernode.privacy.allow_full_content` first.
//!   This check happens in reflection.rs before enqueuing.
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase B - 🌟 Created.

use super::llm_provider::ChatMessage;

// ============================================
// Privacy Level
// ============================================

/// Privacy level for LLM prompt construction.
///
/// Determines how much content is included in the prompt sent to the LLM.
/// Check `config.memchain.supernode.privacy` before using `Full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyLevel {
    /// Only metadata: entity names, relation types, IDs.
    /// Safe for external providers.
    Structured,
    /// Anonymized summary text (no raw user content).
    /// Suitable for most cloud providers.
    Summary,
    /// Full decrypted conversation content.
    /// Use ONLY with local providers (Ollama) unless user explicitly consented.
    Full,
}

impl PrivacyLevel {
    pub fn from_str(s: &str) -> Self {
        match s {
            "full" => Self::Full,
            "summary" => Self::Summary,
            _ => Self::Structured,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Structured => "structured",
            Self::Summary => "summary",
            Self::Full => "full",
        }
    }
}

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
    /// In Structured mode: only passed if entity_names is empty.
    /// In Full mode: always passed for richer context.
    pub first_user_message: Option<&'a str>,
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `session_title` task.
///
/// ## Output Contract
/// Plain text title, 5-10 words, no quotes, no trailing punctuation.
/// If project_name is provided, start with "{project_name}: ".
///
/// ## Examples
/// - "Project Alpha: JWT auth, RS256 migration"
/// - "Rate limiting with token bucket algorithm"
/// - "React component refactor and test coverage"
pub fn build_session_title(input: &SessionTitleInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You generate short, human-readable titles for AI conversation sessions. \
         Rules: 5-10 words max, plain text, no quotes, no trailing punctuation. \
         If a project is provided, start with it followed by ': '. \
         Be specific and technical."
    );

    let user_content = match input.privacy_level {
        PrivacyLevel::Full => {
            // Full mode: use first message for richest context
            let mut parts = Vec::new();
            if let Some(proj) = input.project_name {
                parts.push(format!("Project: {}", proj));
            }
            if !input.entity_names.is_empty() {
                parts.push(format!("Key topics: {}", input.entity_names.join(", ")));
            }
            if let Some(msg) = input.first_user_message {
                // Truncate to 200 chars for token budget
                let preview = if msg.len() > 200 {
                    let end = msg.char_indices().nth(200).map(|(i, _)| i).unwrap_or(msg.len());
                    format!("{}...", &msg[..end])
                } else {
                    msg.to_string()
                };
                parts.push(format!("First message: {}", preview));
            }
            parts.push("Generate the session title.".to_string());
            parts.join("\n")
        }
        _ => {
            // Structured/Summary: metadata only
            if !input.entity_names.is_empty() {
                let topics = input.entity_names.join(", ");
                match input.project_name {
                    Some(proj) => format!("Project: {}\nKey topics: {}\nGenerate the session title.", proj, topics),
                    None => format!("Key topics: {}\nGenerate the session title.", topics),
                }
            } else if let Some(proj) = input.project_name {
                format!("Project: {}\nNo specific topics detected.\nGenerate a short session title.", proj)
            } else if let Some(msg) = input.first_user_message {
                // Fallback: truncated first message (structured — already stripped of no-mem)
                let preview = if msg.len() > 120 {
                    let end = msg.char_indices().nth(120).map(|(i, _)| i).unwrap_or(msg.len());
                    format!("{}...", &msg[..end])
                } else {
                    msg.to_string()
                };
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
/// 2-3 sentence narrative describing what this community represents,
/// its purpose, and the relationships between its key members.
/// Plain text, no bullet points, no headers.
pub fn build_community_narrative(input: &CommunityNarrativeInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You write concise narratives for knowledge graph communities. \
         A community is a cluster of related code entities. \
         Output 2-3 sentences: what this community is, its purpose, key relationships. \
         Be technical and specific. No bullet points. Plain prose only."
    );

    // Sort members by mention_count DESC, take top 12
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

/// A conflicting knowledge edge pair.
pub struct ConflictingEdge<'a> {
    pub edge_id: i64,
    pub source_name: &'a str,
    pub relation_type: &'a str,
    pub target_name: &'a str,
    pub fact_text: Option<&'a str>,
    /// Unix timestamp of when this edge was created.
    pub valid_from: i64,
    pub confidence: f64,
}

/// Inputs for conflict resolution.
pub struct ConflictResolutionInput<'a> {
    /// The two (or more) conflicting edges for the same (source, relation_type) pair.
    pub edges: &'a [ConflictingEdge<'a>],
    /// Context about the source entity.
    pub source_entity: &'a str,
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `conflict_resolution` task.
///
/// ## Output Contract
/// JSON object (the ONLY task that returns JSON):
/// ```json
/// {
///   "keep_edge_id": 123,
///   "reason": "Edge 123 is newer and supported by higher-confidence evidence."
/// }
/// ```
/// The `keep_edge_id` must be one of the provided edge IDs.
/// All other edges will be invalidated by `task_worker.rs`.
pub fn build_conflict_resolution(input: &ConflictResolutionInput<'_>) -> Vec<ChatMessage> {
    let system = ChatMessage::system(
        "You resolve conflicts in a knowledge graph by choosing which edge to keep. \
         You will be shown conflicting facts about the same entity relationship. \
         Respond with ONLY a JSON object: {\"keep_edge_id\": <id>, \"reason\": \"<brief reason>\"}. \
         Prefer: newer facts > higher confidence > more specific fact_text. \
         No other text, no markdown, just the JSON object."
    );

    let edges_desc: Vec<String> = input.edges.iter().map(|e| {
        let fact = e.fact_text.unwrap_or("(no supporting text)");
        format!(
            "Edge ID {}: {} {} {} | fact: {} | confidence: {:.2} | created: timestamp {}",
            e.edge_id, e.source_name, e.relation_type, e.target_name,
            fact, e.confidence, e.valid_from
        )
    }).collect();

    let user_content = format!(
        "Entity: {}\nConflicting edges for the same relationship type:\n{}\n\n\
         Which edge should be kept? Respond with JSON only.",
        input.source_entity,
        edges_desc.join("\n")
    );

    vec![system, ChatMessage::user(user_content)]
}

// ============================================
// Task 4: recall_synthesis
// ============================================

/// Inputs for recall synthesis (key decisions / natural summary).
pub struct RecallSynthesisInput<'a> {
    pub session_id: &'a str,
    /// Existing mechanical summary (entity list) to upgrade.
    pub existing_summary: Option<&'a str>,
    pub entity_names: &'a [&'a str],
    pub turn_count: i64,
    /// Decrypted conversation turns (only populated in Full mode).
    /// Format: Vec<(role, content)>
    pub turns: &'a [(&'a str, &'a str)],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `recall_synthesis` task.
///
/// ## Output Contract
/// JSON object with two fields:
/// ```json
/// {
///   "summary": "2-3 sentence natural summary of what was discussed",
///   "key_decisions": "bullet list of decisions/conclusions reached, or null if none"
/// }
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
            // Full mode: include actual conversation (truncated for token budget)
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
                "Session ID: {}\nTurns: {}\n\nConversation:\n{}\n\n\
                 Generate the summary JSON.",
                input.session_id, input.turn_count,
                conv_parts.join("\n")
            )
        }
        PrivacyLevel::Summary => {
            // Summary mode: existing summary + entity names
            let existing = input.existing_summary.unwrap_or("(none)");
            format!(
                "Session ID: {}\nTurns: {}\nExisting summary: {}\nKey topics: {}\n\n\
                 Upgrade this into a natural summary JSON.",
                input.session_id, input.turn_count, existing,
                input.entity_names.join(", ")
            )
        }
        _ => {
            // Structured: entity names only
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
pub struct CodeAnalysisInput<'a> {
    pub artifact_id: &'a str,
    pub filename: Option<&'a str>,
    pub language: Option<&'a str>,
    pub line_count: Option<i64>,
    /// Code content (only populated in Full mode — may be large).
    pub code_content: Option<&'a str>,
    /// Related entity names (module, technology) from the session.
    pub related_entities: &'a [&'a str],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `code_analysis` task.
///
/// ## Output Contract
/// JSON object:
/// ```json
/// {
///   "description": "One sentence: what this code does",
///   "complexity": "low|medium|high",
///   "suggested_tags": ["tag1", "tag2"]
/// }
/// ```
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
        PrivacyLevel::Full if input.code_content.is_some() => {
            let code = input.code_content.unwrap();
            // Truncate to ~2000 chars for token budget
            let code_preview = if code.len() > 2000 {
                let end = code.char_indices().nth(2000)
                    .map(|(i, _)| i).unwrap_or(code.len());
                format!("{}... (truncated)", &code[..end])
            } else {
                code.to_string()
            };

            format!(
                "Artifact ID: {}\nFile: {}\nLanguage: {}\nLines: {}\n\nCode:\n```{}\n{}\n```\n\n\
                 Analyze and respond with JSON.",
                input.artifact_id,
                input.filename.unwrap_or("unknown"),
                input.language.unwrap_or("unknown"),
                input.line_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                input.language.unwrap_or(""),
                code_preview
            )
        }
        _ => {
            // Structured: filename, language, related entities only (no raw code)
            format!(
                "Artifact ID: {}\nFile: {}\nLanguage: {}\nLines: {}\nRelated entities: {}\n\n\
                 Analyze this code artifact metadata and respond with JSON.",
                input.artifact_id,
                input.filename.unwrap_or("unknown"),
                input.language.unwrap_or("unknown"),
                input.line_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                if input.related_entities.is_empty() { "none".to_string() }
                else { input.related_entities.join(", ") }
            )
        }
    };

    vec![system, ChatMessage::user(user_content)]
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
    fn test_session_title_structured_no_entities_fallback_to_message() {
        let input = SessionTitleInput {
            entity_names: &[],
            project_name: None,
            first_user_message: Some("How do I implement rate limiting with token bucket?"),
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_session_title(&input);
        let user = &msgs[1].content;
        assert!(user.contains("rate limiting"));
    }

    #[test]
    fn test_session_title_full_includes_first_message() {
        let long_msg = "A".repeat(300);
        let input = SessionTitleInput {
            entity_names: &["JWT"],
            project_name: None,
            first_user_message: Some(&long_msg),
            privacy_level: PrivacyLevel::Full,
        };
        let msgs = build_session_title(&input);
        let user = &msgs[1].content;
        // Should be truncated at 200 chars
        assert!(user.contains("..."));
        assert!(!user.contains(&long_msg)); // truncated
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
        // auth module (×10) should appear before jwt (×2) in the prompt
        let pos_auth = user.find("auth module").unwrap_or(usize::MAX);
        let pos_jwt = user.find("jwt").unwrap_or(usize::MAX);
        assert!(pos_auth < pos_jwt, "Higher mention_count should appear first");
    }

    #[test]
    fn test_conflict_resolution_json_contract() {
        let edges = &[
            ConflictingEdge {
                edge_id: 1, source_name: "auth", relation_type: "USES",
                target_name: "JWT", fact_text: Some("auth uses JWT"),
                valid_from: 1000, confidence: 0.9,
            },
            ConflictingEdge {
                edge_id: 2, source_name: "auth", relation_type: "USES",
                target_name: "OAuth", fact_text: Some("switched to OAuth"),
                valid_from: 2000, confidence: 0.95,
            },
        ];
        let input = ConflictResolutionInput {
            edges,
            source_entity: "auth module",
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_conflict_resolution(&input);
        let system = &msgs[0].content;
        let user = &msgs[1].content;
        // System must specify JSON-only output
        assert!(system.contains("JSON object"));
        assert!(system.contains("keep_edge_id"));
        // User must contain both edge IDs
        assert!(user.contains("Edge ID 1"));
        assert!(user.contains("Edge ID 2"));
    }

    #[test]
    fn test_recall_synthesis_full_truncates_conversation() {
        // Build a long conversation
        let long_content = "x".repeat(1000);
        let turns: Vec<(&str, &str)> = vec![
            ("user", long_content.as_str()),
            ("assistant", long_content.as_str()),
            ("user", long_content.as_str()),
            ("assistant", long_content.as_str()),
        ];
        let turn_refs: Vec<(&str, &str)> = turns.iter().map(|(r, c)| (*r, *c)).collect();
        let input = RecallSynthesisInput {
            session_id: "sess_001",
            existing_summary: None,
            entity_names: &["JWT"],
            turn_count: 4,
            turns: &turn_refs,
            privacy_level: PrivacyLevel::Full,
        };
        let msgs = build_recall_synthesis(&input);
        let user = &msgs[1].content;
        // Truncation at 3000 chars total means not all 4000 chars of content appear
        assert!(user.len() < 4500, "Conversation should be truncated");
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
        assert!(user.contains("RS256"));
        assert!(!user.contains("Conversation:")); // no conversation in structured mode
    }

    #[test]
    fn test_code_analysis_full_includes_code() {
        let code = "fn rate_limit(requests: u32, window: u64) -> bool { requests < 100 }";
        let input = CodeAnalysisInput {
            artifact_id: "art_001",
            filename: Some("rate_limit.rs"),
            language: Some("rust"),
            line_count: Some(1),
            code_content: Some(code),
            related_entities: &["token bucket", "tower middleware"],
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
            filename: Some("auth.rs"),
            language: Some("rust"),
            line_count: Some(42),
            code_content: Some("fn auth() { ... }"),
            related_entities: &["JWT", "auth module"],
            privacy_level: PrivacyLevel::Structured,
        };
        let msgs = build_code_analysis(&input);
        let user = &msgs[1].content;
        // Structured: should include filename/language but NOT code content
        assert!(user.contains("auth.rs"));
        assert!(user.contains("rust"));
        assert!(!user.contains("fn auth()"), "Code content must not appear in structured mode");
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
    fn test_all_builders_have_system_first() {
        // Invariant: system message is always index 0
        let title_msgs = build_session_title(&SessionTitleInput {
            entity_names: &["JWT"], project_name: None,
            first_user_message: None, privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(title_msgs[0].role, "system");

        let narr_msgs = build_community_narrative(&CommunityNarrativeInput {
            community_name: "Auth", members: &[], key_edges: &[],
            privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(narr_msgs[0].role, "system");

        let conflict_msgs = build_conflict_resolution(&ConflictResolutionInput {
            edges: &[], source_entity: "auth", privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(conflict_msgs[0].role, "system");

        let synth_msgs = build_recall_synthesis(&RecallSynthesisInput {
            session_id: "s", existing_summary: None, entity_names: &[],
            turn_count: 0, turns: &[], privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(synth_msgs[0].role, "system");

        let code_msgs = build_code_analysis(&CodeAnalysisInput {
            artifact_id: "a", filename: None, language: None, line_count: None,
            code_content: None, related_entities: &[], privacy_level: PrivacyLevel::Structured,
        });
        assert_eq!(code_msgs[0].role, "system");
    }
}
