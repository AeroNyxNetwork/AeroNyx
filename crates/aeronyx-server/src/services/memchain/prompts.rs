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
//! All builders respect `PrivacyLevel` (defined in config_supernode.rs):
//! - `Structured`: Only metadata (entity names, relation types, IDs).
//!   Safe to send to external providers (DeepSeek, Anthropic, etc.).
//! - `Summary`: Anonymized summary text. Currently treated same as Structured
//!   by most builders — future enhancement will differentiate.
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
//! - PrivacyLevel is defined in config_supernode.rs and re-exported here.
//!   Do NOT define a second PrivacyLevel in this file.
//! - `conflict_resolution` returns JSON wrapped in `<r>...</r>` tags.
//!   The parser in task_worker.rs tries <r> tags FIRST, then markdown fences.
//! - Keep prompts SHORT — most cheap models have good instruction following.
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase B - 🌟 Created.
//! v2.5.0+Audit Fix 6-13   - 🔧 Various fixes (see previous doc comments).
//! v2.5.0+Fix               - 🔧 Added build_entity_description + EntityDescriptionInput.
//! v2.5.0+Unify             - 🔧 [BUG FIX] PrivacyLevel re-exported from
//!   config_supernode.rs which now has all 3 variants (Structured/Summary/Full).
//!   Removed any local PrivacyLevel definition. recall_synthesis builder now
//!   correctly matches PrivacyLevel::Summary variant.

use super::llm_provider::ChatMessage;
// PrivacyLevel is defined in config_supernode.rs — re-export it here.
// config_supernode.rs now has Structured / Summary / Full variants.
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
        // Structured and Summary use the same prompt for session_title
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
    pub conflict_edge_ids: &'a [i64],
    pub edges: &'a [ConflictingEdge],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `conflict_resolution` task.
///
/// ## Output Contract
/// JSON object wrapped in `<r>...</r>` tags:
/// ```text
/// <r>{"keep_edge_id": 123, "reason": "Edge 123 is newer..."}</r>
/// ```
pub fn build_conflict_resolution(input: &ConflictResolutionInput<'_>) -> Vec<ChatMessage> {
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
/// JSON: `{"summary": "...", "key_decisions": "..." | null}`
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
            // Structured (and Full with no turns)
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
    pub language: &'a str,
    pub line_count: Option<i64>,
    /// Code content (only populated in Full mode).
    pub code_content: &'a str,
    pub existing_tags: &'a [&'a str],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `code_analysis` task.
///
/// ## Output Contract
/// JSON: `{"description": "...", "complexity": "low|medium|high", "suggested_tags": ["tag1"]}`
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
            // Structured/Summary: metadata only — code_content MUST NOT appear in prompt.
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
// Task 6: entity_description
// ============================================

/// Inputs for entity description generation.
pub struct EntityDescriptionInput<'a> {
    pub entity_name: &'a str,
    pub entity_type: &'a str,
    /// Known relationships: (relation_type, other_entity_name).
    pub relations: &'a [(&'a str, &'a str)],
    pub privacy_level: PrivacyLevel,
}

/// Build prompt for `entity_description` task.
///
/// ## Output Contract
/// Plain text, 1-2 sentences. No preamble. No bullet points.
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

        assert!(system.contains("<r>"));
        assert!(system.contains("keep_edge_id"));
        assert!(user.contains("Edge ID 1"));
        assert!(user.contains("Edge ID 2"));
        assert!(user.contains("<r>"));
    }

    #[test]
    fn test_conflict_resolution_no_confidence_handled() {
        let edges = vec![
            ConflictingEdge {
                edge_id: 5, source: "mod_a".into(), relation: "DEPENDS_ON".into(),
                target: "mod_b".into(), fact_text: None,
                valid_from: 500, confidence: None,
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
    fn test_recall_synthesis_summary_mode() {
        // v2.5.0+Unify: Summary variant now exists in PrivacyLevel
        let input = RecallSynthesisInput {
            session_id: "sess_003",
            existing_summary: Some("Topics: Docker, Kubernetes"),
            entity_names: &["Docker", "Kubernetes"],
            turn_count: 8,
            turns: &[],
            privacy_level: PrivacyLevel::Summary,
        };
        let msgs = build_recall_synthesis(&input);
        let user = &msgs[1].content;
        assert!(user.contains("Existing summary"));
        assert!(user.contains("Docker"));
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
        // v2.5.0+Unify: PrivacyLevel from_str/as_str defined in config_supernode
        assert_eq!(PrivacyLevel::from_str("full"), PrivacyLevel::Full);
        assert_eq!(PrivacyLevel::from_str("summary"), PrivacyLevel::Summary);
        assert_eq!(PrivacyLevel::from_str("structured"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::from_str("unknown"), PrivacyLevel::Structured);
        assert_eq!(PrivacyLevel::Full.as_str(), "full");
        assert_eq!(PrivacyLevel::Summary.as_str(), "summary");
        assert_eq!(PrivacyLevel::Structured.as_str(), "structured");
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
        assert!(result.is_char_boundary(result.len() - 3));
    }
}
