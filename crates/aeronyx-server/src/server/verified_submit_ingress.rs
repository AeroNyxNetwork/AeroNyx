// File: crates/aeronyx-server/src/server/verified_submit_ingress.rs
// Purpose: Authenticated submit admission, exact replay, and custody completion.
// Dependencies: relay replay facade, owned Tokio lanes/permits, blocking workers.
// Flow: authenticate -> bounded lane -> DB admission -> onion -> DB completion.
// Boundary: no wire/schema changes; cancelled DB work retains its lane and permit.
// [ARCH-SPLIT 2026-10-02] Private entry points remain available to the parent.
// [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] No SQLite on async workers.
// Last Modified: 2026-10-04.
use super::*;

use std::sync::OnceLock;

use crate::services::chat_relay::{ChatRelayError, ChatRelayResult};
use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

// [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] One process-wide budget
// bounds queued/running DB work AND lane waiters. Do not release a permit on
// cancellation while its non-cancellable blocking closure is still running.
const VERIFIED_SUBMIT_MAX_IN_FLIGHT: usize = 32;

fn verified_submit_admission() -> Arc<Semaphore> {
    static ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();
    Arc::clone(ADMISSION.get_or_init(|| Arc::new(Semaphore::new(VERIFIED_SUBMIT_MAX_IN_FLIGHT))))
}

struct VerifiedSubmitExecutionLease {
    _lane: OwnedMutexGuard<()>,
    _permit: OwnedSemaphorePermit,
}

enum VerifiedSubmitDbFailure {
    Storage(ChatRelayError),
    WorkerUnavailable,
}

impl VerifiedSubmitDbFailure {
    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::Storage(error) => error.reason_bucket(),
            Self::WorkerUnavailable => "verified_submit_worker_unavailable",
        }
    }
}

// The async owner has no lease while the worker owns it. A cancelled JoinHandle
// detaches work; its eventual result drops the lease instead of reopening the
// lane early. Worker loss leaves this executor unusable, never reacquiring it.
struct VerifiedSubmitExecution {
    lease: Option<VerifiedSubmitExecutionLease>,
}

impl VerifiedSubmitExecution {
    async fn run<T, F>(
        &mut self,
        relay: &Arc<ChatRelayService>,
        request: &ChatRelayVerifiedSubmitRequestV1,
        operation: F,
    ) -> std::result::Result<T, VerifiedSubmitDbFailure>
    where
        T: Send + 'static,
        F: FnOnce(&ChatRelayService, &ChatRelayVerifiedSubmitRequestV1) -> ChatRelayResult<T>
            + Send
            + 'static,
    {
        let relay = Arc::clone(relay);
        let request = request.clone();
        let lease = self
            .lease
            .take()
            .ok_or(VerifiedSubmitDbFailure::WorkerUnavailable)?;
        let (lease, result) = tokio::task::spawn_blocking(move || {
            let result = operation(relay.as_ref(), &request);
            (lease, result)
        })
        .await
        .map_err(|_| VerifiedSubmitDbFailure::WorkerUnavailable)?;
        self.lease = Some(lease);
        result.map_err(VerifiedSubmitDbFailure::Storage)
    }
}

impl Server {
    pub(super) async fn handle_verified_chat_submit(
        request: ChatRelayVerifiedSubmitRequestV1,
        session: &Arc<crate::services::Session>,
        chat_relay: &Option<Arc<ChatRelayService>>,
        peer_store: &PeerStore,
        self_node_id: &[u8; 32],
        node_identity: &IdentityKeyPair,
        chat_peer_client: Option<&reqwest::Client>,
    ) -> ChatRelayVerifiedSubmitResponseV1 {
        Self::handle_verified_chat_submit_with_clock(
            request,
            session,
            chat_relay,
            peer_store,
            self_node_id,
            node_identity,
            chat_peer_client,
            unix_now_secs,
        )
        .await
    }

    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Read the clock after
    // the single-flight wait so admission cannot reuse a pre-wait timestamp.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_verified_chat_submit_with_clock(
        request: ChatRelayVerifiedSubmitRequestV1,
        session: &Arc<crate::services::Session>,
        chat_relay: &Option<Arc<ChatRelayService>>,
        peer_store: &PeerStore,
        self_node_id: &[u8; 32],
        node_identity: &IdentityKeyPair,
        chat_peer_client: Option<&reqwest::Client>,
        clock: impl Fn() -> u64 + Send + Sync,
    ) -> ChatRelayVerifiedSubmitResponseV1 {
        let rejected = || {
            ChatRelayVerifiedSubmitResponseV1::rejected(
                request.request_id,
                request.envelope.message_id,
            )
        };

        if !request
            .envelope
            .sender_matches_authenticated_identity(&session.client_public_key.to_bytes())
            || request.verify_signatures_for_replay().is_err()
        {
            let response = rejected();
            if let Some(relay) = chat_relay.as_ref() {
                // [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] Count
                // rejected explicit submissions only as an aggregate result.
                relay.record_verified_submit_result(unix_now_secs(), response.result);
            }
            warn!(
                reason = "verified_submit_authentication_failed",
                "[CHAT_RELAY] Verified submit rejected"
            );
            return response;
        }

        let Some(relay) = chat_relay.as_ref() else {
            warn!(
                reason = "relay_unavailable",
                "[CHAT_RELAY] Verified submit rejected"
            );
            return rejected();
        };

        // [CHAT-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-23 by Codex] The
        // sender/request-id private key is single-flight across one fixed lock
        // lane. Exact retries replay the first response without repeating
        // onion relay or entry custody. Reuse for another envelope fails
        // closed before route, wallet-route, or durable-state mutation.
        // [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] Admit before waiting
        // for a lane; waiting requests cannot grow a detached worker backlog.
        let permit = match verified_submit_admission().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                let response = rejected();
                relay.record_verified_submit_result(unix_now_secs(), response.result);
                warn!(
                    reason = "verified_submit_backpressure",
                    "[CHAT_RELAY] Verified submit rejected"
                );
                return response;
            }
        };
        let lane = relay.lock_verified_submit(&request).await;
        let mut execution = VerifiedSubmitExecution {
            lease: Some(VerifiedSubmitExecutionLease {
                _lane: lane,
                _permit: permit,
            }),
        };
        let now = clock();
        // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Preserve the
        // existing symmetric freshness window for new effects. Only too-old
        // requests may take this durable completed-only branch; future-dated
        // requests never gain replay authority.
        if request.request_timestamp.abs_diff(now)
            > aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS
        {
            if request.request_timestamp < now {
                match execution
                    .run(relay, &request, move |relay, request| {
                        relay.verified_submit_completed_readonly(request, now)
                    })
                    .await
                {
                    Ok(Some(response)) => {
                        relay.record_verified_submit_replay(now, response.result);
                        return response;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        warn!(
                            reason = error.reason_bucket(),
                            "[CHAT_RELAY] Retained verified submit unavailable"
                        );
                    }
                }
            }
            let response = rejected();
            relay.record_verified_submit_result(now, response.result);
            return response;
        }
        match execution
            .run(relay, &request, |relay, request| {
                relay.verified_submit_cache_lookup(request)
            })
            .await
        {
            Ok(VerifiedSubmitCacheLookup::Exact(response)) => {
                relay.record_verified_submit_replay(unix_now_secs(), response.result);
                return response;
            }
            Ok(VerifiedSubmitCacheLookup::Conflict) => {
                let response = rejected();
                relay.record_verified_submit_conflict(unix_now_secs(), response.result);
                warn!(
                    reason = "verified_submit_request_conflict",
                    "[CHAT_RELAY] Verified submit rejected"
                );
                return response;
            }
            Ok(VerifiedSubmitCacheLookup::Pending) => {
                // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] The
                // transactional admission gate below decides whether this is
                // still live work or exact abandoned custody from an older
                // process. Lookup alone cannot safely make that distinction.
            }
            Ok(VerifiedSubmitCacheLookup::Miss) => {}
            Err(error) => {
                let response = rejected();
                relay.record_verified_submit_result(unix_now_secs(), response.result);
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Verified submit durable replay check failed"
                );
                return response;
            }
        }

        // [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] Reserve
        // bounded durable replay capacity before wallet-route, network, or
        // custody mutation. Saturation and crash-left pending work reject here;
        // unexpired evidence is never evicted to make room for new effects.
        let entry_recovery = match execution
            .run(relay, &request, |relay, request| {
                relay.reserve_verified_submit(request)
            })
            .await
        {
            Ok(VerifiedSubmitAdmission::Reserved) => false,
            Ok(VerifiedSubmitAdmission::ReservedForEntryRecovery) => true,
            Ok(VerifiedSubmitAdmission::Pending) => {
                let response = rejected();
                relay.record_verified_submit_pending_rejection(unix_now_secs(), response.result);
                warn!(
                    reason = "verified_submit_request_pending",
                    "[CHAT_RELAY] Verified submit rejected"
                );
                return response;
            }
            Ok(VerifiedSubmitAdmission::Conflict) => {
                let response = rejected();
                relay.record_verified_submit_conflict(unix_now_secs(), response.result);
                warn!(
                    reason = "verified_submit_request_conflict",
                    "[CHAT_RELAY] Verified submit rejected"
                );
                return response;
            }
            Ok(VerifiedSubmitAdmission::CapacityExhausted) => {
                let response = rejected();
                relay.record_verified_submit_capacity_rejection(unix_now_secs(), response.result);
                warn!(
                    reason = "verified_submit_replay_capacity",
                    "[CHAT_RELAY] Verified submit rejected"
                );
                return response;
            }
            Ok(VerifiedSubmitAdmission::Completed) => {
                match execution
                    .run(relay, &request, |relay, request| {
                        relay.verified_submit_cache_lookup(request)
                    })
                    .await
                {
                    Ok(VerifiedSubmitCacheLookup::Exact(response)) => {
                        relay.record_verified_submit_replay(unix_now_secs(), response.result);
                        return response;
                    }
                    Ok(_) | Err(_) => {
                        let response = rejected();
                        relay.record_verified_submit_result(unix_now_secs(), response.result);
                        warn!(
                            reason = "verified_submit_admission_race",
                            "[CHAT_RELAY] Verified submit rejected"
                        );
                        return response;
                    }
                }
            }
            Err(error) => {
                let response = rejected();
                relay.record_verified_submit_result(unix_now_secs(), response.result);
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Verified submit durable admission failed"
                );
                return response;
            }
        };

        // [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] DB queue/lock waits
        // must not lend stale authentication to new route or custody effects.
        // Keep any reserved slot fail-closed; do not release it for a new leader.
        if request.request_timestamp.abs_diff(clock())
            > aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS
        {
            let response = rejected();
            relay.record_verified_submit_result(unix_now_secs(), response.result);
            warn!(
                reason = "verified_submit_expired_before_effect",
                "[CHAT_RELAY] Verified submit rejected"
            );
            return response;
        }

        let (onion_delivered, terminal_receipt) = if entry_recovery {
            // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Recovery
            // repeats only idempotent local entry custody. Re-announcing a
            // wallet route or selecting a fresh onion path could duplicate a
            // pre-crash network effect whose terminal ACK was never retained.
            (false, None)
        } else {
            relay.wallet_routes.announce(
                &request.envelope.sender,
                session.id.clone(),
                session.client_endpoint,
            );
            let onion_outcome = Self::relay_authenticated_chat_over_onion_paths(
                chat_peer_client,
                Some(relay.as_ref()),
                peer_store,
                node_identity,
                self_node_id,
                &request.envelope,
                Some(&request.request_id),
            )
            .await;
            (
                onion_outcome.delivered(),
                onion_outcome.first_terminal_receipt,
            )
        };
        let verified_onion = terminal_receipt.is_some();
        if onion_delivered != verified_onion {
            // This is an internal invariant failure, not a remote route fault.
            // Fail closed rather than claim terminal evidence without bytes
            // the client can independently verify.
            error!(
                reason = "verified_submit_receipt_invariant_failed",
                "[CHAT_RELAY] Verified submit evidence rejected"
            );
        }

        // [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] Custody and response
        // persistence stay in ONE closure, with no newly cancellable gap. If
        // the worker is lost, retain only independently verified remote proof;
        // unknown local custody must never be reported as established.
        let worker_failure_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            request.envelope.message_id,
            verified_onion && onion_delivered,
            false,
            terminal_receipt.clone(),
        );
        match execution
            .run(relay, &request, move |relay, request| {
                Ok(Self::complete_verified_submit_custody(
                    relay,
                    request,
                    verified_onion && onion_delivered,
                    terminal_receipt,
                    entry_recovery,
                ))
            })
            .await
        {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Verified submit completion unavailable"
                );
                worker_failure_response
            }
        }
    }

    // [VERIFIED-SUBMIT-BLOCKING 2026-10-04 by Codex] Called only inside the
    // owned blocking executor. Preserve custody -> evidence -> replay ordering
    // and the existing result when durable replay persistence returns an error.
    fn complete_verified_submit_custody(
        relay: &ChatRelayService,
        request: &ChatRelayVerifiedSubmitRequestV1,
        verified_onion: bool,
        terminal_receipt: Option<BlindRelayDeliveryReceipt>,
        entry_recovery: bool,
    ) -> ChatRelayVerifiedSubmitResponseV1 {
        let entry_custody = match relay.store_pending(&request.envelope) {
            Ok(()) => true,
            Err(error) => {
                warn!(
                    reason = error.reason_bucket(),
                    "[CHAT_RELAY] Verified submit entry custody failed"
                );
                false
            }
        };
        let response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            request.envelope.message_id,
            verified_onion,
            entry_custody,
            terminal_receipt,
        );
        // [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] The relay
        // status records only the closed result bucket, never identifiers.
        let completed_at = unix_now_secs();
        relay.record_verified_submit_result(completed_at, response.result);
        let persistence = relay.remember_verified_submit_response(request, &response);
        if entry_recovery {
            // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Recovery
            // closes exactly once: successful custody plus durable replay is
            // completed; a durable no-custody result is failed; persistence
            // failure remains deferred for a later replacement process.
            let recovery_outcome =
                VerifiedSubmitRecoveryOutcome::from_results(entry_custody, persistence.is_ok());
            relay.record_verified_submit_recovery_outcome(completed_at, recovery_outcome);
        }
        if let Err(error) = persistence {
            // [DURABLE-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-24 by Codex]
            // Delivery/custody has already completed, so changing the result
            // would misreport evidence and encourage another submission. Keep
            // same-process replay active and expose only the fixed error bucket.
            warn!(
                reason = error.reason_bucket(),
                "[CHAT_RELAY] Verified submit durable replay persistence failed"
            );
        }
        response
    }
}
