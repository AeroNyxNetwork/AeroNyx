// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/recent_errors/tests.rs
// ============================================
//! # Tests: recent error telemetry
//!
//! Unit tests for recent error telemetry, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

#[test]
fn recent_error_sanitizer_redacts_sensitive_tokens() {
    let sanitized = sanitize_operational_log_message(
        "failed peer 203.0.113.10 endpoint=198.51.100.8:443 ipv6=2001:db8::1 host=api.example.com url=https://example.com/path db=/root/private/operator.sqlite win=C:\\AeroNyx\\private.db key=0123456789abcdef0123456789abcdef",
    );

    assert!(sanitized.contains("[ip]"));
    assert!(sanitized.contains("endpoint=[ip]"));
    assert!(sanitized.contains("ipv6=[ip]"));
    assert!(sanitized.contains("host=[ip]"));
    assert!(sanitized.contains("url=[url]") || sanitized.contains("[url]"));
    assert!(sanitized.contains("db=[path]"));
    assert!(sanitized.contains("win=[path]"));
    assert!(sanitized.contains("key=[redacted]") || sanitized.contains("[redacted]"));
    assert!(!sanitized.contains("203.0.113.10"));
    assert!(!sanitized.contains("198.51.100.8"));
    assert!(!sanitized.contains("2001:db8::1"));
    assert!(!sanitized.contains("api.example.com"));
    assert!(!sanitized.contains("/root/private/operator.sqlite"));
    assert!(!sanitized.contains("C:\\AeroNyx\\private.db"));
    assert!(!sanitized.contains("0123456789abcdef0123456789abcdef"));
}

#[test]
fn recent_error_parser_classifies_severity_without_leaking_sensitive_tokens() {
    let critical = parse_recent_error_line(
        "2026-06-18T12:00:00Z host aeronyx-server[42]: ERROR failed endpoint=203.0.113.10:443 key=0123456789abcdef0123456789abcdef",
    )
    .expect("critical event");
    assert_eq!(critical.severity, "critical");
    assert!(critical.message.contains("endpoint=[ip]"));
    assert!(critical.message.contains("key=[redacted]"));
    assert!(!critical.message.contains("203.0.113.10"));
    assert!(!critical
        .message
        .contains("0123456789abcdef0123456789abcdef"));

    let warning = parse_recent_error_line(
        "2026-06-18T12:00:01Z host aeronyx-server[42]: warning capacity threshold near limit",
    )
    .expect("warning event");
    assert_eq!(warning.severity, "warning");

    let info = parse_recent_error_line(
        "2026-06-18T12:00:02Z host aeronyx-server[42]: notice service recovered after restart",
    )
    .expect("info event");
    assert_eq!(info.severity, "info");
}

#[test]
fn recent_error_cache_bounds_refresh_and_stale_windows() {
    assert!(recent_error_cache_needs_refresh(0, 100));
    assert!(!recent_error_cache_needs_refresh(
        100,
        100 + RECENT_ERROR_CACHE_TTL_SECS - 1
    ));
    assert!(recent_error_cache_needs_refresh(
        100,
        100 + RECENT_ERROR_CACHE_TTL_SECS
    ));

    let cache = RecentErrorCache::default();
    cache.last_success_at.store(100, Ordering::Release);
    cache.entries.write().expect("cache write").push(
        parse_recent_error_line(
            "2026-08-12T00:00:00Z host aeronyx-server[42]: warning bounded cache test",
        )
        .expect("test event"),
    );
    assert_eq!(
        recent_error_cache_snapshot(&cache, 100 + RECENT_ERROR_CACHE_MAX_STALE_SECS).len(),
        1
    );
    assert!(
        recent_error_cache_snapshot(&cache, 101 + RECENT_ERROR_CACHE_MAX_STALE_SECS).is_empty()
    );
}
