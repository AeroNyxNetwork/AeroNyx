// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source.rs
// ============================================
//! # Exact-target anonymous-mailbox source coordinator
//!
//! This default-off building block prepares one immutable blind-relay request
//! for one receiver-provided, descriptor-pinned custody node. It does not own
//! HTTP, server startup, client APIs, discovery, or any participant identity.
//! Journal methods are synchronous: callers use an explicit blocking boundary.
//! The shared execution registry admits bounded asynchronous per-route work.
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.
//! This root keeps the module docs, the persisted-state size bounds shared by
//! the journal, storage and codec children, and the explicit re-exports, so
//! every `services::chat_relay_anonymous_mailbox_source::X` path is unchanged.
//! The crash-drill tests also stay here (`tests.rs`): their subprocess worker
//! is selected by its libtest path under this module.
//! - `chat_relay_anonymous_mailbox_source/types.rs`: errors, phases, exact
//!   target pin and resolver boundary, prepared/result/outbound values
//! - `chat_relay_anonymous_mailbox_source/execution.rs`: bounded per-route
//!   execution registry and owned permits
//! - `chat_relay_anonymous_mailbox_source/journal.rs`: encrypted `SQLite`
//!   journal, phase CAS transitions, retention cleanup
//! - `chat_relay_anonymous_mailbox_source/storage.rs`: schema bootstrap and
//!   migration, aggregate accounting, private `SQLite` shims
//! - `chat_relay_anonymous_mailbox_source/state_codec.rs`: persisted-state
//!   envelope, AEAD associated data, commitments, projection check
//! - `chat_relay_anonymous_mailbox_source/frame_validation.rs`: request-frame
//!   admission and response verification
//! - `chat_relay_anonymous_mailbox_source/coordinator.rs`: exact-target
//!   coordinator composition
//! - `chat_relay_anonymous_mailbox_source/crash_drill.rs`: test-only crash hook
//! - Tests sit beside the code they exercise (`<child>/tests.rs`), with shared
//!   fixtures in `test_support.rs`.
//!
//! ## Last Modified
//! v1.0.1-TicketTargetGuard — Reject an inner TicketIssue target mismatch
//! before journal admission and on durable record recovery.
//! [MAILBOX-SOURCE-COALESCING 2026-10-01 by Codex] Bound and serialize
//! process-local exact-route executions without changing durable replay state.

use aeronyx_core::protocol::anonymous_mailbox::MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES;

mod coordinator;
#[cfg(test)]
mod crash_drill;
mod execution;
mod frame_validation;
mod journal;
mod state_codec;
mod storage;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
mod types;

pub(crate) use coordinator::AnonymousMailboxSourceCoordinator;
pub(crate) use execution::SourceExecutionPermit;
pub(crate) use journal::{
    AnonymousMailboxSourceCleanupReport, SqliteAnonymousMailboxSourceJournal,
};
pub(crate) use types::{
    AnonymousMailboxSourceError, AnonymousMailboxSourceOutbound, AnonymousMailboxSourcePhase,
    AnonymousMailboxSourcePrepared, AnonymousMailboxSourceResult, ExactAnonymousMailboxTargetPin,
    ExactAnonymousMailboxTargetResolver,
};

const JOURNAL_AEAD_TAG_BYTES: usize = 16;
const MAX_JOURNAL_BODY_BYTES: usize = 2 * 1024 * 1024;
// [ANONYMOUS-MAILBOX-SOURCE-BOUNDS 2026-09-03 by Codex] Keep persisted
// clear/protected state bounded by protocol frames, not operator-configurable
// aggregate storage. The restart allowance intentionally requires an explicit
// source-journal update if the core restart ABI ever grows beyond 256 bytes.
const MAX_JOURNAL_RESTART_STATE_BYTES: usize = 256;
const JOURNAL_STATE_BODY_COMMITMENT_BYTES: usize = 32;
const JOURNAL_STATE_ENVELOPE_BYTES: usize = 1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES + 4 + 2 + 4;
const MAX_JOURNAL_CLEAR_STATE_BYTES: usize = JOURNAL_STATE_ENVELOPE_BYTES
    + (2 * MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES)
    + MAX_JOURNAL_RESTART_STATE_BYTES;
const MAX_JOURNAL_PROTECTED_STATE_BYTES: usize =
    MAX_JOURNAL_CLEAR_STATE_BYTES + JOURNAL_AEAD_TAG_BYTES;
