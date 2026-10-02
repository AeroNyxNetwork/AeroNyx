// [ARCH-SPLIT 2026-10-02]
// Authenticated chat submit entry. Replay and storage gates stay in this path.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

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
        let _single_flight = relay.lock_verified_submit(&request).await;
        let now = clock();
        // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Preserve the
        // existing symmetric freshness window for new effects. Only too-old
        // requests may take this durable completed-only branch; future-dated
        // requests never gain replay authority.
        if request.request_timestamp.abs_diff(now)
            > aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS
        {
            if request.request_timestamp < now {
                match relay.verified_submit_completed_readonly(&request, now) {
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
        match relay.verified_submit_cache_lookup(&request) {
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
        let entry_recovery = match relay.reserve_verified_submit(&request) {
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
                match relay.verified_submit_cache_lookup(&request) {
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
            verified_onion && onion_delivered,
            entry_custody,
            terminal_receipt,
        );
        // [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] The relay
        // status records only the closed result bucket, never identifiers.
        let completed_at = unix_now_secs();
        relay.record_verified_submit_result(completed_at, response.result);
        let persistence = relay.remember_verified_submit_response(&request, &response);
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
