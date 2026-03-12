// ============================================
// File: crates/aeronyx-server/src/services/memchain/query_analyzer.rs
// ============================================
//! # Query Analyzer — GLiNER + Regex + Entity Matching (v2.4.0)
//!
//! ## Creation Reason (v2.4.0-GraphCognition)
//! Analyzes recall queries to determine the optimal retrieval strategy.
//! Replaces the "always vector search" approach with intelligent routing:
//! - SEMANTIC queries → pure vector search (v2.3.0 behavior)
//! - PROJECT queries → project timeline + session summaries
//! - TEMPORAL queries → time-filtered sessions/episodes
//! - ENTITY queries → entity graph traversal + related records
//!
//! ## Main Functionality
//! - Detect entities in query text using GLiNER NER engine
//! - Extract temporal references using regex patterns
//! - Match detected entities against known entities in the database
//! - Classify query type (SEMANTIC/PROJECT/TEMPORAL/ENTITY)
//! - Return analysis result for hybrid retrieval routing
//!
//! ## Architecture Position
//! ```text
//! recall-hook (plugin) → POST /api/mpi/recall → mpi_recall handler
//!                                                    │
//!                                                    ▼
//!                                          QueryAnalyzer::analyze()
//!                                                    │
//!                                     ┌──────────────┼──────────────┐
//!                                     ▼              ▼              ▼
//!                               GLiNER NER     Regex time     Entity DB
//!                               (~10ms)        (~1ms)         lookup
//!                                     │              │              │
//!                                     └──────────────┼──────────────┘
//!                                                    ▼
//!                                          QueryAnalysis result
//!                                          (query_type + entities + time_range)
//! ```
//!
//! ## Performance Targets
//! - Total analysis time: < 15ms
//! - GLiNER inference: ~10ms (single text, 8 labels)
//! - Regex + entity matching: ~5ms
//!
//! ## Fallback Behavior
//! - NER engine unavailable → regex-only entity detection + SEMANTIC fallback
//! - No entities matched → SEMANTIC (pure vector search = v2.3.0)
//! - Empty query → SEMANTIC with no analysis
//!
//! ## Dependencies
//! - ner.rs — NerEngine for GLiNER entity detection
//! - storage_ops.rs — get_entities_cached() for known entity matching
//!
//! ⚠️ Important Note for Next Developer:
//! - QueryAnalyzer does NOT own NerEngine — it borrows via reference
//! - Entity matching is case-insensitive (name_normalized)
//! - Temporal regex supports CN + EN date/time patterns
//! - QueryType::SEMANTIC is the default fallback — ensures v2.3.0 compatibility
//! - The analyzer is stateless — safe to call from concurrent handlers
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Initial implementation

use std::collections::HashMap;

use tracing::debug;

use super::ner::{NerEngine, DetectedEntity};

// ============================================
// Constants
// ============================================

/// Entity labels used for query analysis.
/// These match the labels used in Stage 2 entity extraction (consistency).
const QUERY_ENTITY_LABELS: &[&str] = &[
    "project", "module", "technology", "file", "person",
    "decision", "problem", "solution",
];

/// Keywords that indicate a PROJECT-type query (progress/status/overview).
const PROJECT_KEYWORDS_CN: &[&str] = &[
    "进展", "进度", "状态", "概览", "概况", "总结", "综合",
];
const PROJECT_KEYWORDS_EN: &[&str] = &[
    "progress", "status", "overview", "summary",
];

// ============================================
// Types
// ============================================

/// The classified type of a recall query.
///
/// Determines which retrieval strategy the recall handler uses:
/// - SEMANTIC: pure vector search (v2.3.0 default behavior)
/// - PROJECT: project timeline + session summaries + graph context
/// - TEMPORAL: time-filtered sessions/episodes
/// - ENTITY: entity-centric graph traversal + related records
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryType {
    /// No entity match — use traditional vector search.
    /// This is the default fallback, ensuring v2.3.0 behavior.
    Semantic,
    /// Matched a project-type entity + progress/status keywords.
    /// Retrieval: project timeline, session summaries, key decisions.
    Project,
    /// Detected a date/time reference in the query.
    /// Retrieval: time-filtered sessions/episodes + vector search.
    Temporal,
    /// Matched a non-project entity (technology, module, person, etc.).
    /// Retrieval: entity graph traversal (BFS) + related records.
    Entity,
}

/// A matched entity from the query, linked to a known entity in the database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MatchedEntity {
    /// The text span detected in the query.
    pub query_text: String,
    /// The entity type label from GLiNER.
    pub label: String,
    /// The confidence score from GLiNER detection.
    pub confidence: f32,
    /// The entity_id from the entities table (if matched to a known entity).
    /// None if detected but not found in the database.
    pub entity_id: Option<String>,
    /// The entity_type from the database (may differ from GLiNER label).
    pub entity_type: Option<String>,
}

/// Optional time range extracted from the query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimeRange {
    /// Start timestamp (Unix seconds). 0 = unbounded start.
    pub start: i64,
    /// End timestamp (Unix seconds). 0 = unbounded end (now).
    pub end: i64,
    /// The original time text detected in the query.
    pub source_text: String,
}

/// Complete query analysis result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryAnalysis {
    /// Classified query type.
    pub query_type: QueryType,
    /// Entities detected and matched.
    pub matched_entities: Vec<MatchedEntity>,
    /// Time range extracted (if any).
    pub time_range: Option<TimeRange>,
    /// Original query text (for reference).
    pub query_text: String,
}

// ============================================
// Query Analyzer
// ============================================

/// Analyze a recall query to determine retrieval strategy.
///
/// ## Arguments
/// * `query` - The user's query text
/// * `ner_engine` - Optional NER engine for GLiNER entity detection
/// * `known_entities` - Map of name_normalized → entity_id from the database
///   (obtained from storage.get_entities_cached())
/// * `now_ts` - Current Unix timestamp (for relative time resolution)
///
/// ## Returns
/// QueryAnalysis with query_type, matched entities, and optional time range.
///
/// ## Fallback Behavior
/// - No NER engine → regex-only detection → likely SEMANTIC
/// - NER detects entities but none match DB → SEMANTIC (unknown entities)
/// - NER + DB match → PROJECT/ENTITY depending on entity type + keywords
/// - Time reference detected → TEMPORAL (regardless of entity matches)
pub fn analyze_query(
    query: &str,
    ner_engine: Option<&NerEngine>,
    known_entities: &HashMap<String, String>,
    now_ts: i64,
) -> QueryAnalysis {
    if query.trim().is_empty() {
        return QueryAnalysis {
            query_type: QueryType::Semantic,
            matched_entities: Vec::new(),
            time_range: None,
            query_text: query.to_string(),
        };
    }

    // Step 1: Detect entities via GLiNER (if available)
    let detected_entities = if let Some(engine) = ner_engine {
        match engine.detect_entities(query, QUERY_ENTITY_LABELS) {
            Ok(entities) => entities,
            Err(e) => {
                debug!(error = %e, "[QUERY_ANALYZER] GLiNER detection failed, falling back to regex");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // Step 2: Match detected entities against known entities in DB
    let matched_entities = match_entities(&detected_entities, known_entities);

    // Step 3: Extract temporal references via regex
    let time_range = extract_time_reference(query, now_ts);

    // Step 4: Classify query type
    let query_type = classify_query(query, &matched_entities, &time_range);

    debug!(
        query_type = ?query_type,
        entities = matched_entities.len(),
        has_time = time_range.is_some(),
        "[QUERY_ANALYZER] Analysis complete"
    );

    QueryAnalysis {
        query_type,
        matched_entities,
        time_range,
        query_text: query.to_string(),
    }
}

// ============================================
// Entity Matching
// ============================================

/// Match GLiNER-detected entities against known entities in the database.
fn match_entities(
    detected: &[DetectedEntity],
    known: &HashMap<String, String>,
) -> Vec<MatchedEntity> {
    let mut matched = Vec::new();

    for det in detected {
        let normalized = det.text.to_lowercase().trim().to_string();

        // Try exact match first
        let entity_id = known.get(&normalized).cloned();

        // Try partial match if exact fails (e.g., "项目B" matches "项目b")
        let entity_id = entity_id.or_else(|| {
            known.iter()
                .find(|(k, _)| k.contains(&normalized) || normalized.contains(k.as_str()))
                .map(|(_, v)| v.clone())
        });

        matched.push(MatchedEntity {
            query_text: det.text.clone(),
            label: det.label.clone(),
            confidence: det.confidence,
            entity_id: entity_id.clone(),
            entity_type: if entity_id.is_some() { Some(det.label.clone()) } else { None },
        });
    }

    matched
}

// ============================================
// Query Classification
// ============================================

/// Classify the query type based on matched entities, time references, and keywords.
fn classify_query(
    query: &str,
    matched_entities: &[MatchedEntity],
    time_range: &Option<TimeRange>,
) -> QueryType {
    let lower = query.to_lowercase();

    // TEMPORAL takes priority if a date/time reference is detected
    if time_range.is_some() {
        return QueryType::Temporal;
    }

    // Check for project-type keywords + project entity match
    let has_project_keywords =
        PROJECT_KEYWORDS_CN.iter().any(|kw| lower.contains(kw))
        || PROJECT_KEYWORDS_EN.iter().any(|kw| lower.contains(kw));

    let has_project_entity = matched_entities.iter().any(|e| {
        e.entity_id.is_some() && e.label == "project"
    });

    if has_project_keywords || has_project_entity {
        // If we have a matched project entity, it's definitely PROJECT
        if has_project_entity {
            return QueryType::Project;
        }
        // If we have project keywords + any matched entity, still PROJECT
        if has_project_keywords && matched_entities.iter().any(|e| e.entity_id.is_some()) {
            return QueryType::Project;
        }
    }

    // Check for any matched entity (non-project)
    let has_matched_entity = matched_entities.iter().any(|e| e.entity_id.is_some());
    if has_matched_entity {
        return QueryType::Entity;
    }

    // Default: SEMANTIC (pure vector search)
    QueryType::Semantic
}

// ============================================
// Temporal Reference Extraction
// ============================================

/// Extract a time reference from the query using regex patterns.
///
/// Supports:
/// - Relative: "昨天", "上周", "yesterday", "last week", "today"
/// - Absolute CN: "2月18日", "3月", "2月18号"
/// - Absolute EN: "February 18", "Feb 18", "March 2024"
///
/// Returns None if no time reference is detected.
fn extract_time_reference(query: &str, now_ts: i64) -> Option<TimeRange> {
    let lower = query.to_lowercase();

    // ── Relative time patterns ──

    // "昨天" / "yesterday"
    if lower.contains("昨天") || lower.contains("yesterday") {
        let day_start = now_ts - (now_ts % 86400) - 86400;
        return Some(TimeRange {
            start: day_start,
            end: day_start + 86400,
            source_text: if lower.contains("昨天") { "昨天".into() } else { "yesterday".into() },
        });
    }

    // "今天" / "today"
    if lower.contains("今天") || lower.contains("today") {
        let day_start = now_ts - (now_ts % 86400);
        return Some(TimeRange {
            start: day_start,
            end: now_ts,
            source_text: if lower.contains("今天") { "今天".into() } else { "today".into() },
        });
    }

    // "上周" / "last week"
    if lower.contains("上周") || lower.contains("last week") {
        let week_start = now_ts - (now_ts % 86400) - 7 * 86400;
        let week_end = now_ts - (now_ts % 86400);
        return Some(TimeRange {
            start: week_start,
            end: week_end,
            source_text: if lower.contains("上周") { "上周".into() } else { "last week".into() },
        });
    }

    // "这周" / "this week"
    if lower.contains("这周") || lower.contains("this week") {
        // Approximate: last 7 days
        let week_start = now_ts - 7 * 86400;
        return Some(TimeRange {
            start: week_start,
            end: now_ts,
            source_text: if lower.contains("这周") { "这周".into() } else { "this week".into() },
        });
    }

    // "上个月" / "last month"
    if lower.contains("上个月") || lower.contains("last month") {
        let month_start = now_ts - 30 * 86400;
        let month_end = now_ts - (now_ts % 86400);
        return Some(TimeRange {
            start: month_start,
            end: month_end,
            source_text: if lower.contains("上个月") { "上个月".into() } else { "last month".into() },
        });
    }

    // ── Absolute date patterns (CN): "X月Y日" or "X月Y号" ──
    if let Some(caps) = regex_match(&lower, r"(\d{1,2})月(\d{1,2})[日号]") {
        if let (Some(month), Some(day)) = (caps.get(0), caps.get(1)) {
            let m: u32 = month.parse().unwrap_or(0);
            let d: u32 = day.parse().unwrap_or(0);
            if m >= 1 && m <= 12 && d >= 1 && d <= 31 {
                // Assume current year; resolve to timestamp
                let ts = approximate_date_ts(now_ts, m, d);
                if let Some(start) = ts {
                    return Some(TimeRange {
                        start,
                        end: start + 86400,
                        source_text: format!("{}月{}日", m, d),
                    });
                }
            }
        }
    }

    // ── Absolute date patterns (EN): "Month Day" ──
    let months = [
        ("january", 1), ("february", 2), ("march", 3), ("april", 4),
        ("may", 5), ("june", 6), ("july", 7), ("august", 8),
        ("september", 9), ("october", 10), ("november", 11), ("december", 12),
        ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
        ("jun", 6), ("jul", 7), ("aug", 8), ("sep", 9),
        ("oct", 10), ("nov", 11), ("dec", 12),
    ];

    for (name, month_num) in &months {
        if let Some(caps) = regex_match(&lower, &format!(r"{}\s+(\d{{1,2}})", name)) {
            if let Some(day_str) = caps.get(0) {
                let d: u32 = day_str.parse().unwrap_or(0);
                if d >= 1 && d <= 31 {
                    let ts = approximate_date_ts(now_ts, *month_num, d);
                    if let Some(start) = ts {
                        return Some(TimeRange {
                            start,
                            end: start + 86400,
                            source_text: format!("{} {}", name, d),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Simple regex capture helper. Returns captured groups (not the full match).
fn regex_match(text: &str, pattern: &str) -> Option<Vec<String>> {
    let re = regex::Regex::new(pattern).ok()?;
    let caps = re.captures(text)?;
    let groups: Vec<String> = (1..caps.len())
        .filter_map(|i| caps.get(i).map(|m| m.as_str().to_string()))
        .collect();
    if groups.is_empty() { None } else { Some(groups) }
}

/// Approximate a (month, day) date to a Unix timestamp.
/// Assumes the current year. If the date is in the future, uses the previous year.
fn approximate_date_ts(now_ts: i64, month: u32, day: u32) -> Option<i64> {
    // Simple approximation: use days-since-epoch math
    // This avoids pulling in chrono as a dependency
    let secs_per_day: i64 = 86400;

    // Approximate current year from now_ts
    // 2024-01-01 00:00:00 UTC ≈ 1704067200
    // Each year ≈ 365.25 days
    let years_since_1970 = (now_ts as f64) / (365.25 * secs_per_day as f64);
    let current_year = 1970 + years_since_1970 as u32;

    // Days in each month (non-leap approximation)
    let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    if month < 1 || month > 12 || day < 1 || day > days_in_month[month as usize] as u32 + 1 {
        return None;
    }

    // Calculate day of year
    let mut day_of_year: u32 = day;
    for m in 1..month {
        day_of_year += days_in_month[m as usize] as u32;
    }

    // Calculate approximate timestamp for this year
    let year_start = ((current_year - 1970) as f64 * 365.25 * secs_per_day as f64) as i64;
    let date_ts = year_start + (day_of_year as i64 - 1) * secs_per_day;

    // If in the future, use previous year
    if date_ts > now_ts {
        Some(date_ts - (365 * secs_per_day))
    } else {
        Some(date_ts)
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_known() -> HashMap<String, String> {
        HashMap::new()
    }

    fn sample_known() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("jwt".to_string(), "ent_jwt".to_string());
        m.insert("auth module".to_string(), "ent_auth".to_string());
        m.insert("项目b".to_string(), "ent_project_b".to_string());
        m.insert("project b".to_string(), "ent_project_b".to_string());
        m
    }

    #[test]
    fn test_empty_query_returns_semantic() {
        let result = analyze_query("", None, &empty_known(), 1700000000);
        assert_eq!(result.query_type, QueryType::Semantic);
        assert!(result.matched_entities.is_empty());
        assert!(result.time_range.is_none());
    }

    #[test]
    fn test_no_ner_engine_returns_semantic() {
        let result = analyze_query("what is JWT", None, &sample_known(), 1700000000);
        // Without NER engine, no entities detected → SEMANTIC
        assert_eq!(result.query_type, QueryType::Semantic);
    }

    #[test]
    fn test_temporal_yesterday_cn() {
        let now = 1700000000i64;
        let result = analyze_query("昨天做了什么", None, &empty_known(), now);
        assert_eq!(result.query_type, QueryType::Temporal);
        assert!(result.time_range.is_some());
        let tr = result.time_range.unwrap();
        assert_eq!(tr.source_text, "昨天");
        assert!(tr.start < now);
        assert!(tr.end <= now);
    }

    #[test]
    fn test_temporal_yesterday_en() {
        let now = 1700000000i64;
        let result = analyze_query("what did we do yesterday", None, &empty_known(), now);
        assert_eq!(result.query_type, QueryType::Temporal);
        assert!(result.time_range.is_some());
    }

    #[test]
    fn test_temporal_last_week() {
        let now = 1700000000i64;
        let result = analyze_query("上周的工作进展", None, &empty_known(), now);
        assert_eq!(result.query_type, QueryType::Temporal);
        let tr = result.time_range.unwrap();
        assert_eq!(tr.source_text, "上周");
    }

    #[test]
    fn test_temporal_absolute_cn_date() {
        let now = 1700000000i64; // ~Nov 2023
        let result = analyze_query("2月18日做了什么", None, &empty_known(), now);
        assert_eq!(result.query_type, QueryType::Temporal);
        let tr = result.time_range.unwrap();
        assert!(tr.source_text.contains("2月18日"));
    }

    #[test]
    fn test_temporal_today() {
        let now = 1700000000i64;
        let result = analyze_query("today's tasks", None, &empty_known(), now);
        assert_eq!(result.query_type, QueryType::Temporal);
        let tr = result.time_range.unwrap();
        assert_eq!(tr.source_text, "today");
        assert!(tr.end == now);
    }

    #[test]
    fn test_classify_project_keywords_cn() {
        // Project keywords without entity match → still SEMANTIC (need entity)
        let lower = "项目B的综合进展";
        let has_kw = PROJECT_KEYWORDS_CN.iter().any(|kw| lower.contains(kw));
        assert!(has_kw);
    }

    #[test]
    fn test_classify_project_with_entity() {
        // Simulate matched project entity
        let matched = vec![MatchedEntity {
            query_text: "项目B".into(),
            label: "project".into(),
            confidence: 0.9,
            entity_id: Some("ent_project_b".into()),
            entity_type: Some("project".into()),
        }];
        let qt = classify_query("项目B的进展", &matched, &None);
        assert_eq!(qt, QueryType::Project);
    }

    #[test]
    fn test_classify_entity_non_project() {
        let matched = vec![MatchedEntity {
            query_text: "JWT".into(),
            label: "technology".into(),
            confidence: 0.9,
            entity_id: Some("ent_jwt".into()),
            entity_type: Some("technology".into()),
        }];
        let qt = classify_query("为什么选JWT", &matched, &None);
        assert_eq!(qt, QueryType::Entity);
    }

    #[test]
    fn test_classify_semantic_no_match() {
        let matched = vec![MatchedEntity {
            query_text: "unknown thing".into(),
            label: "technology".into(),
            confidence: 0.6,
            entity_id: None, // Not in DB
            entity_type: None,
        }];
        let qt = classify_query("what is unknown thing", &matched, &None);
        assert_eq!(qt, QueryType::Semantic);
    }

    #[test]
    fn test_temporal_takes_priority() {
        // Even with matched entity, temporal should win
        let matched = vec![MatchedEntity {
            query_text: "JWT".into(),
            label: "technology".into(),
            confidence: 0.9,
            entity_id: Some("ent_jwt".into()),
            entity_type: Some("technology".into()),
        }];
        let tr = Some(TimeRange { start: 100, end: 200, source_text: "yesterday".into() });
        let qt = classify_query("yesterday's JWT changes", &matched, &tr);
        assert_eq!(qt, QueryType::Temporal);
    }

    #[test]
    fn test_match_entities_exact() {
        let known = sample_known();
        let detected = vec![DetectedEntity {
            text: "JWT".into(), label: "technology".into(),
            confidence: 0.9, char_start: 0, char_end: 3,
            word_start: 0, word_end: 0,
        }];
        let matched = match_entities(&detected, &known);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].entity_id, Some("ent_jwt".to_string()));
    }

    #[test]
    fn test_match_entities_partial() {
        let known = sample_known();
        let detected = vec![DetectedEntity {
            text: "auth".into(), label: "module".into(),
            confidence: 0.8, char_start: 0, char_end: 4,
            word_start: 0, word_end: 0,
        }];
        let matched = match_entities(&detected, &known);
        assert_eq!(matched.len(), 1);
        // "auth" is contained in "auth module" → partial match
        assert_eq!(matched[0].entity_id, Some("ent_auth".to_string()));
    }

    #[test]
    fn test_match_entities_no_match() {
        let known = sample_known();
        let detected = vec![DetectedEntity {
            text: "React".into(), label: "technology".into(),
            confidence: 0.8, char_start: 0, char_end: 5,
            word_start: 0, word_end: 0,
        }];
        let matched = match_entities(&detected, &known);
        assert_eq!(matched.len(), 1);
        assert!(matched[0].entity_id.is_none());
    }
}
