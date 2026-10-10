// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/execution/tests.rs
// ============================================
//! # Tests: source execution admission
//!
//! Unit tests for the bounded per-route execution registry, moved from the
//! former inline `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::*;

use std::sync::atomic::Ordering;

// [MAILBOX-SOURCE-COALESCING 2026-10-01 by Codex] Active work and
// queued duplicates share one fixed budget; cancellation reclaims it.
#[tokio::test]
async fn source_coalescing_bounds_waiters_and_reclaims_idle_lanes() {
    let registry = SourceExecutionRegistry::new(3);
    let first = registry.acquire([1; 16]).await.unwrap();
    let mut duplicate = Box::pin(registry.acquire([1; 16]));
    assert!(futures::poll!(duplicate.as_mut()).is_pending());
    assert_eq!(registry.waiters.load(Ordering::Relaxed), 1);
    let other = registry.acquire([2; 16]).await.unwrap();
    assert!(registry.acquire([3; 16]).await.is_none());
    assert_eq!(registry.lanes.lock().len(), 2);
    drop(duplicate);
    assert_eq!(registry.waiters.load(Ordering::Relaxed), 0);
    let third = registry.acquire([3; 16]).await.unwrap();
    drop((first, other, third));
    for id in 4..20 {
        let permit = registry.acquire([id; 16]).await.unwrap();
        assert!(registry.lanes.lock().len() <= 3);
        drop(permit);
    }
    assert_eq!(registry.capacity.available_permits(), 3);
}
