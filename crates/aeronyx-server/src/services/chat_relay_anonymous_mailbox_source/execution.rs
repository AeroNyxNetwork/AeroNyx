// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/execution.rs
// ============================================
//! # Source execution admission
//!
//! Owns the process-local bounded admission registry for exact-route source
//! executions: one serialized lane per route ID under a fixed capacity budget,
//! and the owned permits that represent an admitted execution.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

pub(super) struct SourceExecutionRegistry {
    lanes: Mutex<HashMap<[u8; 16], Weak<tokio::sync::Semaphore>>>,
    capacity: Arc<tokio::sync::Semaphore>,
    limit: usize,
    #[cfg(test)]
    pub(super) waiters: std::sync::atomic::AtomicUsize,
}

/// Owned asynchronous permits, never a synchronous mutex guard. Intentionally
/// not Debug: its lifetime represents private request execution state.
pub struct SourceExecutionPermit {
    _route: tokio::sync::OwnedSemaphorePermit,
    _capacity: tokio::sync::OwnedSemaphorePermit,
}

#[cfg(test)]
struct SourceExecutionWaiter<'a>(&'a std::sync::atomic::AtomicUsize);

#[cfg(test)]
impl Drop for SourceExecutionWaiter<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl SourceExecutionRegistry {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            lanes: Mutex::new(HashMap::new()),
            capacity: Arc::new(tokio::sync::Semaphore::new(limit)),
            limit,
            #[cfg(test)]
            waiters: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(super) async fn acquire(&self, route_id: [u8; 16]) -> Option<SourceExecutionPermit> {
        let capacity = self.capacity.clone().try_acquire_owned().ok()?;
        let lane = {
            // This guard is released BEFORE the only await. Different route
            // IDs never share an execution lane, including hash collisions.
            let mut lanes = self.lanes.lock();
            if let Some(lane) = lanes.get(&route_id).and_then(Weak::upgrade) {
                lane
            } else {
                if lanes.len() >= self.limit {
                    lanes.retain(|_, lane| lane.strong_count() != 0);
                }
                if lanes.len() >= self.limit {
                    return None;
                }
                let lane = Arc::new(tokio::sync::Semaphore::new(1));
                lanes.insert(route_id, Arc::downgrade(&lane));
                lane
            }
        };
        #[cfg(test)]
        let _waiting = {
            self.waiters
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            SourceExecutionWaiter(&self.waiters)
        };
        let route = lane.acquire_owned().await.ok()?;
        Some(SourceExecutionPermit {
            _route: route,
            _capacity: capacity,
        })
    }
}

#[cfg(test)]
mod tests;
