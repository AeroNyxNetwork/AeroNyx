// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/crash_drill.rs
// ============================================
//! # Test-only source journal crash hook
//!
//! Owns the libtest-only post-commit process-exit hook used by the bounded
//! source-journal crash drill. It is compiled only under `cfg(test)`.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::AnonymousMailboxSourcePhase;

#[cfg(test)]
pub(super) const SOURCE_JOURNAL_CRASH_PHASE_ENV: &str =
    "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_PHASE";
#[cfg(test)]
pub(super) const SOURCE_JOURNAL_CRASH_BARRIER_ENV: &str =
    "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_CRASH_BARRIER";
#[cfg(test)]
pub(super) const SOURCE_JOURNAL_CRASH_EXIT_CODE: i32 = 79;

#[cfg(test)]
pub(super) fn crash_after_source_journal_commit(phase: AnonymousMailboxSourcePhase) {
    use std::io::Write;

    if std::env::var(SOURCE_JOURNAL_CRASH_PHASE_ENV)
        .ok()
        .as_deref()
        != Some(match phase {
            AnonymousMailboxSourcePhase::Prepared => "prepared",
            AnonymousMailboxSourcePhase::Armed => "armed",
            AnonymousMailboxSourcePhase::Completed => "completed",
            AnonymousMailboxSourcePhase::Ambiguous => "ambiguous",
            AnonymousMailboxSourcePhase::Rejected => "rejected",
        })
    {
        return;
    }
    let barrier = std::env::var_os(SOURCE_JOURNAL_CRASH_BARRIER_ENV)
        .expect("source crash drill barrier path");
    let mut marker = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(barrier)
        .expect("create source crash drill barrier");
    marker
        .write_all(b"phase-commit-observed")
        .expect("write source crash drill barrier");
    marker.sync_all().expect("sync source crash drill barrier");

    // [ANONYMOUS-MAILBOX-SOURCE-CRASH-DRILL 2026-09-05 by Codex] This hook
    // exists only in the libtest build. It crosses a real process boundary
    // after SQLite commit but before the journal method can return, without
    // signalling or otherwise interacting with any production process.
    std::process::exit(SOURCE_JOURNAL_CRASH_EXIT_CODE);
}
