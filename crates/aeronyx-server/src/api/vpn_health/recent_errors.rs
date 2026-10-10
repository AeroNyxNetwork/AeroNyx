// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/recent_errors.rs
// ============================================
//! # Recent error telemetry
//!
//! Owns the bounded, background-refreshed journal error cache, journal
//! line parsing and severity mapping, and the operational log sanitizer
//! that redacts URLs, paths, network destinations, and key material.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::Ordering;

use super::{
    run_command, unix_now_secs, RecentErrorCache, RecentErrorEvent, RECENT_ERROR_CACHE,
    RECENT_ERROR_CACHE_MAX_STALE_SECS, RECENT_ERROR_CACHE_TTL_SECS, RECENT_ERROR_COMMAND_TIMEOUT,
};

pub(super) async fn collect_recent_error_events(service_name: &str) -> Vec<RecentErrorEvent> {
    let cache = RECENT_ERROR_CACHE.get_or_init(RecentErrorCache::default);
    let now = unix_now_secs();
    let snapshot = recent_error_cache_snapshot(cache, now);
    if recent_error_cache_needs_refresh(cache.last_attempt_at.load(Ordering::Acquire), now)
        && cache
            .refresh_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    {
        cache.last_attempt_at.store(now, Ordering::Release);
        let service_name = service_name.to_string();
        tokio::spawn(async move {
            if let Some(entries) = query_recent_error_events(&service_name).await {
                if let Ok(mut current) = cache.entries.write() {
                    *current = entries;
                    cache
                        .last_success_at
                        .store(unix_now_secs(), Ordering::Release);
                }
            }
            cache.refresh_in_flight.store(false, Ordering::Release);
        });
    }
    snapshot
}

fn recent_error_cache_needs_refresh(last_attempt_at: u64, now: u64) -> bool {
    last_attempt_at == 0 || now.saturating_sub(last_attempt_at) >= RECENT_ERROR_CACHE_TTL_SECS
}

fn recent_error_cache_snapshot(cache: &RecentErrorCache, now: u64) -> Vec<RecentErrorEvent> {
    let last_success_at = cache.last_success_at.load(Ordering::Acquire);
    if last_success_at == 0
        || now.saturating_sub(last_success_at) > RECENT_ERROR_CACHE_MAX_STALE_SECS
    {
        return Vec::new();
    }
    cache
        .entries
        .read()
        .map(|entries| entries.clone())
        .unwrap_or_default()
}

async fn query_recent_error_events(service_name: &str) -> Option<Vec<RecentErrorEvent>> {
    let output = match run_command(
        "journalctl",
        &[
            "-u",
            service_name,
            "-p",
            "warning..alert",
            "--since",
            "-24 hours",
            "--no-pager",
            "--output=short-iso",
            "--lines=20",
        ],
        RECENT_ERROR_COMMAND_TIMEOUT,
    )
    .await
    {
        Ok(output) => output,
        Err(_) => return None,
    };

    Some(
        output
            .lines()
            .rev()
            .filter_map(parse_recent_error_line)
            .take(5)
            .collect(),
    )
}

fn parse_recent_error_line(line: &str) -> Option<RecentErrorEvent> {
    let line = line.trim();
    if line.is_empty() || line == "-- No entries --" {
        return None;
    }

    let mut parts = line.split_whitespace();
    let timestamp = parts.next().map(|value| value.to_string());
    let _host = parts.next();
    let _unit = parts.next();
    let raw_message = parts.collect::<Vec<_>>().join(" ");
    let severity = classify_recent_error_severity(&raw_message);
    let message = raw_message;
    let message = sanitize_operational_log_message(&message);
    if message.is_empty() {
        return None;
    }

    Some(RecentErrorEvent {
        timestamp,
        severity,
        source: "systemd_journal_aeronyx_server_warning_alert",
        message,
        privacy_boundary: concat!(
            "sanitized node service log summary only; client public IPs, URLs, ",
            "long key-like values, DNS contents, destinations, packet payloads, ",
            "voucher secrets, chat plaintext, and wallet-level traffic are redacted"
        ),
    })
}

fn classify_recent_error_severity(message: &str) -> &'static str {
    let normalized = message.to_ascii_lowercase();
    if normalized.contains("emerg")
        || normalized.contains("emergency")
        || normalized.contains("alert")
        || normalized.contains("critical")
        || normalized.contains("crit")
        || normalized.contains("panic")
        || normalized.contains("fatal")
    {
        return "critical";
    }

    if normalized.contains("error")
        || normalized.contains("failed")
        || normalized.contains("failure")
        || normalized.contains("timed out")
        || normalized.contains("timeout")
    {
        return "critical";
    }

    if normalized.contains("notice") || normalized.contains("info") {
        return "info";
    }

    "warning"
}

fn sanitize_operational_log_message(input: &str) -> String {
    let mut out = Vec::new();
    for token in input.split_whitespace() {
        let trimmed = token
            .trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '[' | ']' | '"' | '\''));
        let (prefix, value) = trimmed
            .split_once('=')
            .map(|(key, value)| (Some(key), value))
            .unwrap_or((None, trimmed));
        let lower = value.to_ascii_lowercase();
        let replacement = if lower.starts_with("http://") || lower.starts_with("https://") {
            Some("[url]")
        } else if looks_like_filesystem_path(value) {
            Some("[path]")
        } else if looks_like_network_destination(value) {
            Some("[ip]")
        } else if looks_like_key_material(value) {
            Some("[redacted]")
        } else {
            None
        };

        if let Some(replacement) = replacement {
            if let Some(prefix) = prefix {
                out.push(format!("{}={}", prefix, replacement));
            } else {
                out.push(replacement.to_string());
            }
        } else {
            out.push(token.to_string());
        }
    }

    let mut message = out.join(" ").replace('\0', "");
    const MAX_LEN: usize = 240;
    if message.len() > MAX_LEN {
        message.truncate(MAX_LEN);
        message.push_str("...");
    }
    message
}

/// Recognizes host-local path tokens without treating ordinary protocol
/// labels containing a slash as private filesystem state.
fn looks_like_filesystem_path(value: &str) -> bool {
    let value = value
        .trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '[' | ']' | '"' | '\'' | ':'));
    if value.starts_with('/') || value.starts_with("~/") || value.starts_with("./") {
        return value.len() > 1;
    }
    if value.starts_with("../") || value.starts_with("file://") {
        return true;
    }

    let bytes = value.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
}

fn looks_like_network_destination(value: &str) -> bool {
    let value = value
        .trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '[' | ']' | '"' | '\'' | '.'));

    if value.parse::<Ipv4Addr>().is_ok()
        || value.parse::<Ipv6Addr>().is_ok()
        || value.parse::<SocketAddr>().is_ok()
    {
        return true;
    }

    if let Some((host, port)) = value.rsplit_once(':') {
        if !host.contains(':')
            && port.parse::<u16>().is_ok()
            && (host.parse::<Ipv4Addr>().is_ok() || looks_like_domain_name(host))
        {
            return true;
        }
    }

    looks_like_domain_name(value)
}

fn looks_like_domain_name(value: &str) -> bool {
    let value = value.trim_end_matches('.');
    if value.len() > 253 || !value.contains('.') {
        return false;
    }

    let mut labels = value.split('.').peekable();
    let mut label_count = 0usize;
    while let Some(label) = labels.next() {
        label_count += 1;
        if label.is_empty()
            || label.len() > 63
            || label.starts_with('-')
            || label.ends_with('-')
            || !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return false;
        }

        if labels.peek().is_none()
            && (label.len() < 2 || !label.chars().all(|c| c.is_ascii_alphabetic()))
        {
            return false;
        }
    }

    label_count >= 2
}

fn looks_like_key_material(value: &str) -> bool {
    if value.len() >= 32 && value.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    if value.len() >= 24
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '=' | '/' | '+'))
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests;
