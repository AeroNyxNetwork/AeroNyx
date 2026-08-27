// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_copy.rs
// ============================================
// Version: 1.1.0-SqliteAdapterComposition
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-COPY-RETRY-DOMAIN 2026-08-27 by Codex] Isolate the
//   bounded SQLite online-backup retry state machine from service-owned I/O.
//
// Modification Reason:
//   [CHAT-BACKUP-SQLITE-DOMAIN 2026-08-28 by Codex] Updated composition after
//   SQLite step mapping and sleeping moved out of the central relay service.
//
// Main Functionality:
//   - Models SQLite backup progress through a closed enum.
//   - Models complete, continue, and delayed-retry actions explicitly.
//   - Tracks only consecutive busy/locked time and resets after progress.
//   - Fails closed on timeout, unsupported progress, or regressed time.
//
// Dependencies:
//   - Uses only `std::time`; it does not depend on SQLite or the filesystem.
//   - `chat_relay_backup_sqlite.rs` maps SQLite progress and executes actions.
//
// Main Logical Flow:
//   1. The SQLite adapter performs one bounded online-backup step.
//   2. It maps the result into `BackupCopyProgress` with an observation time.
//   3. The policy transitions consecutive-busy state and returns one action.
//   4. The adapter completes, steps immediately, sleeps, or returns an error.
//
// Important Note for Next Developer:
//   - Only consecutive Busy/Locked observations consume the timeout budget.
//   - Any More observation must reset the busy window exactly as v1 did.
//   - The timeout boundary is inclusive: elapsed >= timeout fails closed.
//   - This module must remain side-effect free; never sleep or access SQLite.
//
// Last Modified:
//   v1.1.0-SqliteAdapterComposition - Documented adapter ownership
//   v1.0.0-BackupCopyRetryDomain - Initial bounded retry state machine
// ============================================

use std::time::{Duration, Instant};

/// Closed SQLite-independent progress vocabulary for one backup step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupCopyProgress {
    Complete,
    More,
    Busy,
    Locked,
    Unsupported,
}

/// Closed action vocabulary returned to the SQLite-owning service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupCopyAction {
    Complete,
    Continue,
    RetryAfter(Duration),
}

/// Closed failure vocabulary for bounded backup-copy retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupCopyPolicyError {
    BusyTimeout,
    ObservationTimeRegressed,
    UnsupportedProgress,
}

/// Mutable state for one SQLite online-backup copy attempt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BackupCopyRetryState {
    busy_since: Option<Instant>,
}

/// Replaceable policy boundary for SQLite-independent retry decisions.
pub(crate) trait BackupCopyRetryPolicy {
    fn transition(
        &self,
        state: &mut BackupCopyRetryState,
        progress: BackupCopyProgress,
        observed_at: Instant,
    ) -> Result<BackupCopyAction, BackupCopyPolicyError>;
}

/// Production retry policy preserving the v1 timeout and reset semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoundedBackupCopyRetryPolicy {
    busy_timeout: Duration,
    retry_delay: Duration,
}

impl BoundedBackupCopyRetryPolicy {
    pub(crate) const fn new(busy_timeout: Duration, retry_delay: Duration) -> Self {
        Self {
            busy_timeout,
            retry_delay,
        }
    }

    fn busy_action(
        &self,
        state: &mut BackupCopyRetryState,
        observed_at: Instant,
    ) -> Result<BackupCopyAction, BackupCopyPolicyError> {
        let started_at = match state.busy_since {
            Some(started_at) => started_at,
            None => {
                state.busy_since = Some(observed_at);
                observed_at
            }
        };
        let elapsed = observed_at
            .checked_duration_since(started_at)
            .ok_or(BackupCopyPolicyError::ObservationTimeRegressed)?;
        if elapsed >= self.busy_timeout {
            return Err(BackupCopyPolicyError::BusyTimeout);
        }
        Ok(BackupCopyAction::RetryAfter(self.retry_delay))
    }
}

impl BackupCopyRetryPolicy for BoundedBackupCopyRetryPolicy {
    fn transition(
        &self,
        state: &mut BackupCopyRetryState,
        progress: BackupCopyProgress,
        observed_at: Instant,
    ) -> Result<BackupCopyAction, BackupCopyPolicyError> {
        match progress {
            BackupCopyProgress::Complete => {
                state.busy_since = None;
                Ok(BackupCopyAction::Complete)
            }
            BackupCopyProgress::More => {
                state.busy_since = None;
                Ok(BackupCopyAction::Continue)
            }
            BackupCopyProgress::Busy | BackupCopyProgress::Locked => {
                self.busy_action(state, observed_at)
            }
            BackupCopyProgress::Unsupported => Err(BackupCopyPolicyError::UnsupportedProgress),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> BoundedBackupCopyRetryPolicy {
        BoundedBackupCopyRetryPolicy::new(Duration::from_secs(5), Duration::from_millis(10))
    }

    #[test]
    fn complete_and_more_return_distinct_actions() {
        let policy = policy();
        let observed_at = Instant::now();
        let mut state = BackupCopyRetryState::default();

        assert_eq!(
            policy.transition(&mut state, BackupCopyProgress::More, observed_at),
            Ok(BackupCopyAction::Continue)
        );
        assert_eq!(
            policy.transition(&mut state, BackupCopyProgress::Complete, observed_at),
            Ok(BackupCopyAction::Complete)
        );
    }

    #[test]
    fn busy_and_locked_share_the_same_bounded_retry_action() {
        let policy = policy();
        let observed_at = Instant::now();
        for progress in [BackupCopyProgress::Busy, BackupCopyProgress::Locked] {
            let mut state = BackupCopyRetryState::default();
            assert_eq!(
                policy.transition(&mut state, progress, observed_at),
                Ok(BackupCopyAction::RetryAfter(Duration::from_millis(10)))
            );
        }
    }

    #[test]
    fn forward_progress_resets_the_consecutive_busy_window() {
        let policy = policy();
        let started_at = Instant::now();
        let mut state = BackupCopyRetryState::default();

        assert!(policy
            .transition(&mut state, BackupCopyProgress::Busy, started_at)
            .is_ok());
        assert_eq!(
            policy.transition(
                &mut state,
                BackupCopyProgress::More,
                started_at + Duration::from_secs(4),
            ),
            Ok(BackupCopyAction::Continue)
        );
        assert_eq!(
            policy.transition(
                &mut state,
                BackupCopyProgress::Locked,
                started_at + Duration::from_secs(8),
            ),
            Ok(BackupCopyAction::RetryAfter(Duration::from_millis(10)))
        );
    }

    #[test]
    fn timeout_boundary_is_inclusive() {
        let policy = policy();
        let started_at = Instant::now();
        let mut state = BackupCopyRetryState::default();

        assert!(policy
            .transition(&mut state, BackupCopyProgress::Busy, started_at)
            .is_ok());
        assert_eq!(
            policy.transition(
                &mut state,
                BackupCopyProgress::Busy,
                started_at + Duration::from_secs(5),
            ),
            Err(BackupCopyPolicyError::BusyTimeout)
        );
    }

    #[test]
    fn regressed_observation_time_fails_closed() {
        let policy = policy();
        let started_at = Instant::now() + Duration::from_secs(1);
        let mut state = BackupCopyRetryState::default();

        assert!(policy
            .transition(&mut state, BackupCopyProgress::Busy, started_at)
            .is_ok());
        assert_eq!(
            policy.transition(
                &mut state,
                BackupCopyProgress::Locked,
                started_at - Duration::from_millis(1),
            ),
            Err(BackupCopyPolicyError::ObservationTimeRegressed)
        );
    }

    #[test]
    fn unsupported_progress_fails_without_mutating_retry_state() {
        let policy = policy();
        let observed_at = Instant::now();
        let mut state = BackupCopyRetryState::default();
        let before = state;

        assert_eq!(
            policy.transition(&mut state, BackupCopyProgress::Unsupported, observed_at,),
            Err(BackupCopyPolicyError::UnsupportedProgress)
        );
        assert_eq!(state, before);
    }
}
