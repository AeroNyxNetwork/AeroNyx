// ============================================
// File: crates/aeronyx-server/src/server/chat_outbound_runtime.rs
// ============================================
// [CHAT-OUTBOUND-RUNTIME 2026-09-25 by Codex] Own signed-descriptor outbound
// chat route selection, bounded onion/direct relay, receipt validation, and
// ambiguity-safe retry. Caller startup/shutdown and public wire stay unchanged.
use super::*;

const CHAT_PEER_RELAY_FANOUT_LIMIT: usize = 3;
/// Target-bound direct relay permits one exact retry after transport ambiguity.
const DIRECT_PEER_RELAY_V3_MAX_ATTEMPTS: usize = 2;
/// Small delay gives an in-flight first attempt time to publish its custody ACK.
const DIRECT_PEER_RELAY_V3_RETRY_DELAY_MILLIS: u64 = 50;
/// Numeric form keeps compatibility with the workspace's pinned http crate.
pub(super) const HTTP_TOO_EARLY_STATUS_CODE: u16 = 425;

/// Typed validation failures for one bounded direct-relay acknowledgement.
///
/// [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] Keep this typed until the route
/// health boundary so retry policy cannot depend on strings or peer-controlled
/// response content. Body-read/JSON truncation is retryable because delivery
/// remains ambiguous. Remote rejection, invalid cryptographic evidence, and a
/// local verifier failure are terminal for this network attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DirectPeerRelayAckFailure {
    Bounded(BoundedHttpResponseError),
    Rejected,
    ReceiptRequestMissing,
    ReceiptMissing,
    ReceiptInvalid(&'static str),
    VerificationUnavailable,
}

impl DirectPeerRelayAckFailure {
    pub(super) const fn retryable_after_ambiguous_delivery(self) -> bool {
        matches!(
            self,
            Self::Bounded(
                BoundedHttpResponseError::BodyRead | BoundedHttpResponseError::JsonDecode
            )
        )
    }

    pub(super) const fn is_local_runtime_failure(self) -> bool {
        matches!(self, Self::VerificationUnavailable)
    }

    pub(super) fn privacy_safe_reason(self) -> String {
        match self {
            Self::Bounded(error) => format!("peer_relay_ack_{}", error.as_str()),
            Self::Rejected => "peer_relay_ack_rejected".to_string(),
            Self::ReceiptRequestMissing => "peer_relay_receipt_request_missing".to_string(),
            Self::ReceiptMissing => "peer_relay_receipt_missing".to_string(),
            Self::ReceiptInvalid(reason) => format!("peer_relay_{reason}"),
            // Preserve the existing closed aggregate vocabulary. Attribution
            // remains typed, so these local faults never affect peer health.
            Self::VerificationUnavailable => "peer_relay_request_unknown".to_string(),
        }
    }
}

/// Whether a direct delivery failure is evidence about the selected peer.
///
/// [DIRECT-RELAY-LOCAL-FAULT-ATTRIBUTION 2026-08-31 by Codex] Keep this typed
/// until both route reputation and circuit state have consumed it. Strings
/// cannot safely distinguish local scheduler pressure from remote failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DirectPeerRelayFailureAttribution {
    LocalRuntime,
    SelectedPeer,
}

#[derive(Debug)]
pub(super) struct DirectPeerRelayDeliveryFailure {
    pub(super) reason: String,
    pub(super) attribution: DirectPeerRelayFailureAttribution,
}

impl DirectPeerRelayDeliveryFailure {
    pub(super) fn selected_peer(reason: String) -> Self {
        Self {
            reason,
            attribution: DirectPeerRelayFailureAttribution::SelectedPeer,
        }
    }

    pub(super) fn from_ack(error: DirectPeerRelayAckFailure) -> Self {
        let attribution = if error.is_local_runtime_failure() {
            DirectPeerRelayFailureAttribution::LocalRuntime
        } else {
            DirectPeerRelayFailureAttribution::SelectedPeer
        };
        Self {
            reason: error.privacy_safe_reason(),
            attribution,
        }
    }

    pub(super) const fn affects_peer_reputation(&self) -> bool {
        matches!(
            self.attribution,
            DirectPeerRelayFailureAttribution::SelectedPeer
        )
    }
}

/// Failure of one complete target-bound v3 delivery attempt.
#[derive(Debug)]
pub(super) enum TargetBoundPeerRelayFailure {
    Transport(reqwest::Error),
    Http(reqwest::StatusCode),
    Ack(DirectPeerRelayAckFailure),
}

impl TargetBoundPeerRelayFailure {
    pub(super) fn retryable_after_ambiguous_delivery(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Http(status) => status.as_u16() == HTTP_TOO_EARLY_STATUS_CODE,
            Self::Ack(error) => error.retryable_after_ambiguous_delivery(),
        }
    }

    pub(super) fn is_local_runtime_failure(&self) -> bool {
        matches!(self, Self::Ack(error) if error.is_local_runtime_failure())
    }
}

/// Result of one bounded target-bound delivery, including aggregate retry state.
///
/// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Attempt count stays
/// process-local and is reduced to aggregate counters before health publication.
/// Never add peer, message, commitment, endpoint, wallet, or payload fields.
#[derive(Debug)]
pub(super) struct TargetBoundPeerRelayDeliveryOutcome {
    pub(super) result: std::result::Result<(), TargetBoundPeerRelayFailure>,
    pub(super) attempts: usize,
}

impl TargetBoundPeerRelayDeliveryOutcome {
    pub(super) fn retry_triggered(&self) -> bool {
        self.attempts > 1
    }

    pub(super) fn delivery_succeeded(&self) -> bool {
        self.result.is_ok()
    }

    pub(super) fn final_failure_deterministic(&self) -> bool {
        self.result
            .as_ref()
            .err()
            .is_some_and(|error| !error.retryable_after_ambiguous_delivery())
    }

    pub(super) fn local_runtime_failure(&self) -> bool {
        self.result
            .as_ref()
            .err()
            .is_some_and(TargetBoundPeerRelayFailure::is_local_runtime_failure)
    }
}

/// Aggregate-only result of one authenticated App/client onion relay round.
///
/// [CLIENT-ONION-PARTIAL-SUCCESS 2026-07-31 by Codex] A valid terminal-signed
/// receipt is a hard delivery boundary even when another independently chosen
/// replica fails. The caller may retry replica durability later, but it must
/// not expose the same envelope to the legacy direct peer relay after any
/// terminal has already accepted the exact opaque payload.
///
/// Never add selected hops, route ids, message ids, sender/receiver keys,
/// endpoints, payload commitments, ciphertext, or client metadata here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct AuthenticatedChatOnionRelayOutcome {
    pub(super) attempted_paths: usize,
    pub(super) verified_receipts: usize,
    /// First exact terminal receipt retained only for the requesting client.
    /// Never persist or export it through aggregate node health.
    pub(super) first_terminal_receipt: Option<BlindRelayDeliveryReceipt>,
}

/// Whether an onion-route failure can safely affect first-hop reputation.
///
/// [ONION-FAILURE-ATTRIBUTION 2026-08-11 by Codex] HTTP transport, status,
/// and malformed first-hop responses are observable at the selected endpoint.
/// A missing or invalid terminal receipt is different: the source cannot tell
/// whether the middle, a later hop, or the terminal caused it. Treating that
/// ambiguous evidence as a first-hop fault lets a malicious downstream node
/// degrade or quarantine an honest middle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OnionRouteFailureAttribution {
    FirstHop,
    EndToEnd,
}

/// Bounded local failure from authenticated onion request construction.
///
/// [ONION-REQUEST-BUILD-ERROR 2026-08-29 by Codex] This type preserves route
/// admission semantics until the aggregate operator boundary. It never carries
/// payload bytes, route ids, descriptors, endpoints, or node identities.
#[derive(Debug)]
pub(super) enum OnionRequestBuildError {
    PayloadEncoding,
    RoutePlan(OnionRoutePlanError),
}

impl OnionRequestBuildError {
    #[must_use]
    pub(super) const fn reason_bucket(&self) -> &'static str {
        match self {
            Self::PayloadEncoding => "onion_payload_encoding_failed",
            Self::RoutePlan(error) => match error.disposition() {
                OnionRouteFailureDisposition::RefreshRoute => "onion_route_refresh_required",
                OnionRouteFailureDisposition::PolicyRejected => "onion_route_policy_rejected",
                OnionRouteFailureDisposition::LocalConstructionFailed => {
                    "onion_route_local_construction_failed"
                }
            },
        }
    }
}

impl AuthenticatedChatOnionRelayOutcome {
    #[must_use]
    pub(super) fn delivered(&self) -> bool {
        self.verified_receipts > 0
    }

    #[must_use]
    pub(super) fn fully_replicated(&self) -> bool {
        self.attempted_paths > 0 && self.attempted_paths == self.verified_receipts
    }

    /// Whether compatibility direct relay can run without widening a route
    /// surface that already observed this exact opaque envelope.
    ///
    /// [ONION-FALLBACK-PRIVACY 2026-08-15 by Codex] A lost or invalid receipt
    /// does not prove the attempted middle/terminal failed to observe or store
    /// the payload. Only a preflight outcome with zero network attempts may
    /// safely enter the compatibility path; otherwise local pending storage
    /// preserves retry availability without immediate correlation expansion.
    #[must_use]
    pub(super) fn compatibility_direct_fallback_allowed(&self) -> bool {
        self.attempted_paths == 0
    }
}

impl Server {
    pub(super) fn build_two_hop_onion_request(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        chat_envelope: &ChatEnvelope,
        route_id: [u8; 16],
        now: u64,
    ) -> std::result::Result<(PeerBlindRelayRequest, [u8; 32]), OnionRequestBuildError> {
        Self::build_onion_request(
            identity,
            self_node_id,
            &[middle, terminal],
            chat_envelope,
            route_id,
            now,
        )
    }

    /// Builds one onion request from a descriptor-authenticated route plan.
    ///
    /// [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] The builder is shared by
    /// two-hop and three-hop probes so descriptor verification, node
    /// uniqueness, signed capabilities/features, exact TTL, payload commitment,
    /// KEM admission, and outer-envelope signing stay identical. Callers remain
    /// responsible for liveness and network/operator anti-affinity policy.
    pub(super) fn build_onion_request(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        path_descriptors: &[&SignedNodeDescriptor],
        chat_envelope: &ChatEnvelope,
        route_id: [u8; 16],
        now: u64,
    ) -> std::result::Result<(PeerBlindRelayRequest, [u8; 32]), OnionRequestBuildError> {
        let encoded_chat =
            encode_envelope(chat_envelope).map_err(|_| OnionRequestBuildError::PayloadEncoding)?;
        // [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] The source computes the
        // same opaque v2 commitment as the terminal. The purpose is not sent as
        // relay metadata, and a storage receipt cannot satisfy this message
        // route even when route identifiers are accidentally reused.
        let payload_commitment = BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
            &encoded_chat,
            OnionRoutePurpose::MessageRelay,
        );
        // [VERIFIED-ONION-ROUTE 2026-08-29 by Codex] Never derive raw hops
        // directly from candidate projections. The core plan authenticates the
        // original signed descriptors and derives the only admissible TTL.
        let route = VerifiedOnionRoute::from_signed_descriptors(
            *self_node_id,
            path_descriptors.iter().copied(),
            OnionRoutePurpose::MessageRelay,
            now,
        )
        .map_err(OnionRequestBuildError::RoutePlan)?;
        let envelope = route
            .build_envelope(&encoded_chat, route_id, now, identity)
            .map_err(OnionRequestBuildError::RoutePlan)?;

        Ok((
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: *self_node_id,
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
            payload_commitment,
        ))
    }

    #[cfg(test)]
    pub(super) fn verified_delivery_receipt(
        receipt: Option<&BlindRelayDeliveryReceipt>,
        route_id: &[u8; 16],
        payload_commitment: &[u8; 32],
        terminal_node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        receipt.is_some_and(|receipt| {
            blind_relay_delivery_receipt_is_valid(
                receipt,
                route_id,
                payload_commitment,
                terminal_node_id,
                now,
            )
        })
    }

    /// Applies route-health penalties only when the source observed evidence
    /// attributable to the selected first-hop endpoint.
    ///
    /// Aggregate probe/chat outcomes are recorded by the caller in both cases;
    /// this helper controls only node-specific reputation. Reasons must remain
    /// stable privacy-safe buckets and must never contain route or user data.
    #[must_use]
    pub(super) fn record_onion_route_failure(
        peer_store: &PeerStore,
        first_hop: &SignedNodeDescriptor,
        observed_at: u64,
        reason: impl Into<String>,
        attribution: OnionRouteFailureAttribution,
    ) -> bool {
        let reason = reason.into();
        match attribution {
            OnionRouteFailureAttribution::FirstHop => peer_store
                .record_route_forward_failure_for_descriptor(first_hop, observed_at, reason),
            OnionRouteFailureAttribution::EndToEnd => {
                debug!(
                    reason = %reason,
                    "[CHAT_RELAY] Onion route failure left unattributed"
                );
                false
            }
        }
    }

    pub(super) fn blind_relay_probe_url(endpoint: &str) -> Option<String> {
        Self::permissionless_peer_transport_url(endpoint, "/api/chat/peer/blind-relay")
    }

    /// Attempts authenticated client traffic over receipt-capable two-hop
    /// onion paths and returns aggregate delivery/replication evidence.
    ///
    /// At least one verified receipt proves terminal delivery. Any onion
    /// network attempt suppresses legacy direct relay because a lost receipt
    /// cannot prove the selected hops never observed the opaque payload.
    /// `fully_replicated()` additionally means every attempted independent
    /// terminal replica returned a fresh signature bound to that payload.
    /// Mixed-version meshes return an empty outcome before sending, preserving
    /// the compatibility availability fallback without widening route exposure.
    pub(super) async fn relay_authenticated_chat_over_onion_paths(
        client: Option<&reqwest::Client>,
        relay: Option<&ChatRelayService>,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        envelope: &ChatEnvelope,
        client_request_id: Option<&[u8; 16]>,
    ) -> AuthenticatedChatOnionRelayOutcome {
        let now = unix_now_secs();
        // [RELAY-SELECTION-DIAGNOSTICS 2026-08-15 by Codex] Every pre-attempt
        // return must advance the aggregate status with a stable reason bucket;
        // otherwise a real routing failure is indistinguishable from idle.
        let Some(client) = client else {
            if let Some(relay) = relay {
                relay.record_authenticated_onion_outbound_typed(
                    now,
                    0,
                    0,
                    Some(ChatRelayOutboundFailureReason::from_bucket(
                        "peer_http_client_unavailable",
                    )),
                );
            }
            return AuthenticatedChatOnionRelayOutcome::default();
        };
        // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Fail over
        // before allocating route material when no current receipt-capable,
        // network-diverse pair satisfies the same production selector. The
        // coarse reason is safe for operator logs and contains no route data.
        let path_readiness =
            peer_store.authenticated_delivery_path_readiness_excluding(now, &[*self_node_id]);
        if !path_readiness.ready {
            debug!(
                reason = path_readiness.reason,
                "[CHAT_RELAY] Authenticated onion path is not currently ready"
            );
            if let Some(relay) = relay {
                relay.record_authenticated_onion_outbound_typed(
                    now,
                    0,
                    0,
                    Some(ChatRelayOutboundFailureReason::from_bucket(
                        path_readiness.reason,
                    )),
                );
            }
            return AuthenticatedChatOnionRelayOutcome::default();
        }
        let terminal_candidates = peer_store
            .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                NodeCapability::ChatRelay,
                now,
                AUTHENTICATED_CHAT_TERMINAL_FANOUT_LIMIT,
                &[*self_node_id],
            );
        if terminal_candidates.is_empty() {
            // [RELAY-SELECTION-DIAGNOSTICS 2026-08-15 by Codex] Peer state can
            // change between readiness and selection. Record only a stable
            // aggregate bucket, never the missing descriptor or endpoint.
            if let Some(relay) = relay {
                relay.record_authenticated_onion_outbound_typed(
                    now,
                    0,
                    0,
                    Some(ChatRelayOutboundFailureReason::from_bucket(
                        "onion_terminal_selection_changed",
                    )),
                );
            }
            return AuthenticatedChatOnionRelayOutcome::default();
        }

        let mut attempted = 0usize;
        let mut accepted = 0usize;
        let mut first_terminal_receipt = None;
        let mut last_failure_reason = None;
        let mut used_hop_node_ids = Vec::new();
        let mut used_hops = Vec::new();
        for terminal in terminal_candidates {
            let terminal_node_id = terminal.node_id();
            if used_hop_node_ids.contains(&terminal_node_id)
                || !PeerStore::route_endpoint_is_network_diverse_from_all(&terminal, &used_hops)
            {
                last_failure_reason = Some("onion_terminal_diversity_exhausted".to_string());
                continue;
            }
            let mut excluded_node_ids = Vec::with_capacity(used_hop_node_ids.len() + 2);
            excluded_node_ids.push(*self_node_id);
            excluded_node_ids.push(terminal_node_id);
            excluded_node_ids.extend(used_hop_node_ids.iter().copied());
            let Some(middle) = peer_store
                .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                    NodeCapability::OnionMiddle,
                    now,
                    AUTHENTICATED_CHAT_MIDDLE_CANDIDATE_LIMIT,
                    &excluded_node_ids,
                )
                .into_iter()
                .find(|middle| {
                    PeerStore::route_endpoints_are_network_diverse(middle, &terminal)
                        && PeerStore::route_endpoint_is_network_diverse_from_all(middle, &used_hops)
                })
            else {
                last_failure_reason = Some("onion_middle_candidate_unavailable".to_string());
                continue;
            };
            let middle_node_id = middle.node_id();
            let Some(endpoint) = middle.descriptor.public_endpoint.as_deref() else {
                last_failure_reason = Some("onion_middle_endpoint_missing".to_string());
                continue;
            };
            let Some(url) = Self::blind_relay_probe_url(endpoint) else {
                last_failure_reason = Some("onion_middle_endpoint_invalid".to_string());
                continue;
            };

            let route_id = if let Some(request_id) = client_request_id {
                // [CHAT-VERIFIED-SUBMIT 2026-08-22 by Codex] An explicit
                // client request gets a deterministic route id per selected
                // route. Retrying the same signed request therefore reaches
                // the blind-relay replay cache instead of multiplying custody
                // work, while different hop surfaces remain unlinkable.
                chat_verified_submit_route_id(
                    request_id,
                    self_node_id,
                    &middle_node_id,
                    &terminal_node_id,
                )
            } else {
                let mut route_id = [0u8; 16];
                rand::thread_rng().fill_bytes(&mut route_id);
                route_id
            };
            let preparation_identity = (*identity).clone();
            let preparation_self_node_id = *self_node_id;
            let preparation_middle = middle.clone();
            let preparation_terminal = terminal.clone();
            let preparation_envelope = (*envelope).clone();
            let (request, payload_commitment) =
                match prepare_peer_blind_relay_http_request_with(move || {
                    Self::build_two_hop_onion_request(
                        &preparation_identity,
                        &preparation_self_node_id,
                        &preparation_middle,
                        &preparation_terminal,
                        &preparation_envelope,
                        route_id,
                        now,
                    )
                })
                .await
                {
                    Ok(request) => request,
                    Err(BlindRelayRequestPreparationError::Build(error)) => {
                        last_failure_reason = Some(error.reason_bucket().to_string());
                        continue;
                    }
                    Err(BlindRelayRequestPreparationError::Local(error)) => {
                        // [ATOMIC-OUTBOUND-BLIND-PREPARATION 2026-08-31 by Codex]
                        // No hop observed locally rejected route material.
                        last_failure_reason = Some(error.reason_bucket().to_string());
                        continue;
                    }
                };

            // Once sent, both hops are considered exposed for this envelope
            // even if the ACK is lost. Reusing either node or endpoint network
            // for another replica would weaken unlinkability under ambiguity.
            used_hop_node_ids.push(middle_node_id);
            used_hop_node_ids.push(terminal_node_id);
            used_hops.push(middle.clone());
            used_hops.push(terminal.clone());
            attempted = attempted.saturating_add(1);
            match client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(request.body())
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    match decode_bounded_json_response::<PeerBlindRelayResponse>(
                        response,
                        PEER_ACK_RESPONSE_MAX_BYTES,
                    )
                    .await
                    {
                        Ok(ack) if ack.accepted && ack.forwarded => {
                            // [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex]
                            // Validate freshness at response observation time,
                            // not request-selection time, then commit both hop
                            // surfaces and the aggregate delivery as one state
                            // transition. A concurrent descriptor rotation is
                            // a conservative retry/fallback signal, not route
                            // failure evidence against either replacement.
                            let observed_at = unix_now_secs();
                            match verify_blind_relay_delivery_receipt(
                                ack.delivery_receipt,
                                route_id,
                                payload_commitment,
                                terminal_node_id,
                                observed_at,
                            )
                            .await
                            {
                                Ok(receipt)
                                    if peer_store.record_verified_client_onion_route_delivery(
                                        &middle,
                                        &terminal,
                                        observed_at,
                                    ) =>
                                {
                                    accepted = accepted.saturating_add(1);
                                    if first_terminal_receipt.is_none() {
                                        first_terminal_receipt = Some(receipt);
                                    }
                                }
                                Ok(_) => {
                                    last_failure_reason =
                                        Some("onion_delivery_route_surface_changed".to_string());
                                }
                                Err(
                                    BlindRelayDeliveryReceiptVerificationFailure::Missing
                                    | BlindRelayDeliveryReceiptVerificationFailure::Invalid,
                                ) => {
                                    last_failure_reason =
                                        Some("onion_delivery_receipt_rejected".to_string());
                                    let _ = Self::record_onion_route_failure(
                                        peer_store,
                                        &middle,
                                        observed_at,
                                        "delivery_receipt_rejected",
                                        OnionRouteFailureAttribution::EndToEnd,
                                    );
                                }
                                Err(BlindRelayDeliveryReceiptVerificationFailure::Unavailable) => {
                                    // The route was exposed, so direct fallback remains
                                    // forbidden, but local verifier loss is not peer fault.
                                    last_failure_reason = Some(
                                        "onion_delivery_receipt_verifier_unavailable".to_string(),
                                    );
                                }
                            }
                        }
                        Ok(_) => {
                            let observed_at = unix_now_secs();
                            last_failure_reason =
                                Some("onion_delivery_receipt_rejected".to_string());
                            let _ = Self::record_onion_route_failure(
                                peer_store,
                                &middle,
                                observed_at,
                                "delivery_receipt_rejected",
                                OnionRouteFailureAttribution::FirstHop,
                            );
                        }
                        Err(error) => {
                            let observed_at = unix_now_secs();
                            let reason = format!("onion_delivery_ack_{}", error.as_str());
                            last_failure_reason = Some(reason.clone());
                            let _ = Self::record_onion_route_failure(
                                peer_store,
                                &middle,
                                observed_at,
                                reason,
                                OnionRouteFailureAttribution::FirstHop,
                            );
                        }
                    }
                }
                Ok(response) => {
                    let observed_at = unix_now_secs();
                    let reason = format!("onion_delivery_http_{}", response.status().as_u16());
                    last_failure_reason = Some(reason.clone());
                    let _ = Self::record_onion_route_failure(
                        peer_store,
                        &middle,
                        observed_at,
                        reason,
                        OnionRouteFailureAttribution::FirstHop,
                    );
                }
                Err(error) => {
                    let observed_at = unix_now_secs();
                    let reason = Self::classify_reqwest_error("onion_delivery_request", &error);
                    last_failure_reason = Some(reason.clone());
                    let _ = Self::record_onion_route_failure(
                        peer_store,
                        &middle,
                        observed_at,
                        reason,
                        OnionRouteFailureAttribution::FirstHop,
                    );
                }
            }
        }

        if let Some(relay) = relay {
            // [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Keep verified
            // onion evidence independent from the compatibility direct-relay
            // fallback that may run immediately after this function returns.
            // [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] PeerStore
            // keeps its internal route diagnostic, while heartbeat receives
            // only the closed aggregate reason vocabulary.
            relay.record_authenticated_onion_outbound_typed(
                now,
                attempted,
                accepted,
                last_failure_reason
                    .as_deref()
                    .map(ChatRelayOutboundFailureReason::from_bucket),
            );
        }
        let outcome = AuthenticatedChatOnionRelayOutcome {
            attempted_paths: attempted,
            verified_receipts: accepted,
            first_terminal_receipt,
        };
        if outcome.attempted_paths > 0 {
            debug!(
                attempted_paths = outcome.attempted_paths,
                verified_receipts = outcome.verified_receipts,
                delivered = outcome.delivered(),
                fully_replicated = outcome.fully_replicated(),
                "[CHAT_RELAY] Authenticated onion relay round complete"
            );
        }
        outcome
    }

    /// Validates durable peer custody before route health records success.
    ///
    /// [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Signed receipts are
    /// required only when the selected descriptor advertises that contract;
    /// rolling auth-only v2 and legacy v1 peers retain their response shape.
    /// The typed result preserves local verifier faults until attribution.
    pub(super) async fn validate_direct_peer_relay_ack_typed(
        response: reqwest::Response,
        expected_request_commitment: Option<&[u8; 32]>,
        expected_node_id: &[u8; 32],
        require_signed_receipt: bool,
        observed_at: u64,
    ) -> std::result::Result<(), DirectPeerRelayAckFailure> {
        if require_signed_receipt {
            let request_commitment = expected_request_commitment
                .ok_or(DirectPeerRelayAckFailure::ReceiptRequestMissing)?;
            return Self::validate_signed_direct_peer_relay_ack(
                response,
                request_commitment,
                expected_node_id,
                observed_at,
            )
            .await;
        }
        let response = decode_bounded_json_response::<PeerChatRelayResponse>(
            response,
            PEER_ACK_RESPONSE_MAX_BYTES,
        )
        .await
        .map_err(DirectPeerRelayAckFailure::Bounded)?;
        if response.accepted && response.stored_pending {
            Ok(())
        } else {
            Err(DirectPeerRelayAckFailure::Rejected)
        }
    }

    /// Validates one bounded signed durable-custody acknowledgement.
    pub(super) async fn validate_signed_direct_peer_relay_ack(
        response: reqwest::Response,
        expected_request_commitment: &[u8; 32],
        expected_node_id: &[u8; 32],
        observed_at: u64,
    ) -> std::result::Result<(), DirectPeerRelayAckFailure> {
        let response = decode_bounded_json_response::<PeerChatRelayResponseV2>(
            response,
            PEER_ACK_RESPONSE_MAX_BYTES,
        )
        .await
        .map_err(DirectPeerRelayAckFailure::Bounded)?;
        if !response.relay.accepted || !response.relay.stored_pending {
            return Err(DirectPeerRelayAckFailure::Rejected);
        }
        let receipt = response
            .receipt
            .ok_or(DirectPeerRelayAckFailure::ReceiptMissing)?;
        verify_peer_chat_relay_receipt(
            receipt,
            *expected_request_commitment,
            *expected_node_id,
            observed_at,
        )
        .await
        .map_err(|error| match error {
            DirectRelayReceiptVerificationFailure::Invalid(reason) => {
                DirectPeerRelayAckFailure::ReceiptInvalid(reason)
            }
            DirectRelayReceiptVerificationFailure::Unavailable => {
                DirectPeerRelayAckFailure::VerificationUnavailable
            }
        })
    }

    /// Sends and validates one target-bound request with one exact ambiguity retry.
    ///
    /// [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] v3 binds the exact request
    /// to one target and retains its signed ACK by opaque commitment. One
    /// attempt therefore spans send, status, bounded body read, and receipt
    /// verification. Retry remains limited to transport ambiguity, HTTP 425,
    /// or an incomplete/undecodable bounded ACK body. A verifier worker loss
    /// is local and cannot be repaired by repeating remote custody I/O, so it
    /// stops fail-closed without penalizing the selected peer.
    pub(super) async fn send_and_validate_target_bound_peer_relay(
        client: &reqwest::Client,
        url: &str,
        request: &PreparedAuthenticatedPeerChatRelayHttpRequest,
        expected_request_commitment: &[u8; 32],
        expected_node_id: &[u8; 32],
        require_signed_receipt: bool,
    ) -> TargetBoundPeerRelayDeliveryOutcome {
        let mut attempt = 1usize;
        loop {
            let outcome = match client
                .post(url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(request.body())
                .send()
                .await
            {
                Err(error) => Err(TargetBoundPeerRelayFailure::Transport(error)),
                Ok(response) if !response.status().is_success() => {
                    Err(TargetBoundPeerRelayFailure::Http(response.status()))
                }
                Ok(response) => Self::validate_direct_peer_relay_ack_typed(
                    response,
                    Some(expected_request_commitment),
                    expected_node_id,
                    require_signed_receipt,
                    unix_now_secs(),
                )
                .await
                .map_err(TargetBoundPeerRelayFailure::Ack),
            };
            let should_retry = outcome
                .as_ref()
                .err()
                .is_some_and(TargetBoundPeerRelayFailure::retryable_after_ambiguous_delivery);
            if !should_retry || attempt >= DIRECT_PEER_RELAY_V3_MAX_ATTEMPTS {
                return TargetBoundPeerRelayDeliveryOutcome {
                    result: outcome,
                    attempts: attempt,
                };
            }

            attempt = attempt.saturating_add(1);
            tokio::time::sleep(Duration::from_millis(
                DIRECT_PEER_RELAY_V3_RETRY_DELAY_MILLIS,
            ))
            .await;
        }
    }

    pub(super) async fn relay_chat_envelope_to_discovered_peers(
        client: Option<&reqwest::Client>,
        relay: Option<&ChatRelayService>,
        peer_store: &PeerStore,
        node_identity: &IdentityKeyPair,
        envelope: &ChatEnvelope,
    ) -> usize {
        let now = unix_now_secs();
        let Some(client) = client else {
            if let Some(relay) = relay {
                relay.record_peer_relay_outbound_typed(
                    now,
                    0,
                    0,
                    Some(ChatRelayOutboundFailureReason::from_bucket(
                        "peer_http_client_unavailable",
                    )),
                );
            }
            return 0;
        };

        let mut attempted = 0usize;
        let mut accepted = 0usize;
        let mut last_failure_reason: Option<String> = None;

        let excluded_node_ids = [node_identity.public_key_bytes()];
        let mut candidates = peer_store
            .route_candidates_with_capability_excluding(
                NodeCapability::ChatRelay,
                now,
                CHAT_PEER_RELAY_FANOUT_LIMIT,
                &excluded_node_ids,
            )
            .into_iter()
            .filter(|peer| peer_store.is_routeable_now(&peer.node_id(), now))
            .collect::<Vec<_>>();
        let has_target_bound_candidate = candidates.iter().any(|peer| {
            peer.descriptor
                .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayTargetBindingV3)
        });
        let has_authenticated_v2_candidate = candidates.iter().any(|peer| {
            !peer
                .descriptor
                .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayTargetBindingV3)
                && peer
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayAuthV2)
        });
        let has_legacy_candidate = candidates.iter().any(|peer| {
            !peer
                .descriptor
                .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayTargetBindingV3)
                && !peer
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayAuthV2)
        });
        // [OUTBOUND-DIRECT-REQUEST-PREPARATION 2026-08-31 by Codex] Clone the
        // process identity once for the bounded blocking domain, and skip all
        // signing work when the selected candidates support only legacy v1.
        let signing_identity = (has_target_bound_candidate || has_authenticated_v2_candidate)
            .then(|| Arc::new(node_identity.clone()));
        let authenticated_request =
            match (has_authenticated_v2_candidate, signing_identity.as_ref()) {
                (true, Some(identity)) => Some(
                    prepare_peer_chat_relay_request_v2(envelope.clone(), Arc::clone(identity))
                        .await,
                ),
                _ => None,
            };
        let legacy_request = if has_legacy_candidate {
            Some(prepare_peer_chat_relay_request_v1(envelope.clone()).await)
        } else {
            None
        };
        let mut first_target_bound_permit = None;
        if has_target_bound_candidate {
            if let Some(relay) = relay {
                let Some(permit) = relay.begin_direct_peer_delivery(now) else {
                    relay.record_peer_relay_outbound_typed(
                        now,
                        0,
                        0,
                        Some(ChatRelayOutboundFailureReason::from_bucket(
                            "peer_relay_circuit_open",
                        )),
                    );
                    return 0;
                };
                if permit.is_half_open() {
                    // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A half-open
                    // round must probe v3 first; trying compatibility peers
                    // before the reserved probe would be a protocol downgrade.
                    candidates.sort_by_key(|peer| {
                        !peer.descriptor.advertises_protocol_feature(
                            NodeProtocolFeature::DirectPeerRelayTargetBindingV3,
                        )
                    });
                }
                first_target_bound_permit = Some(permit);
            }
        }

        for peer in candidates {
            let use_target_bound_relay = peer
                .descriptor
                .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayTargetBindingV3);
            if !use_target_bound_relay
                && first_target_bound_permit.is_some_and(|permit| permit.is_half_open())
            {
                // Every v3 candidate failed local preflight before the reserved
                // probe could run. Do not spend that recovery edge on v2/v1.
                if let (Some(relay), Some(permit)) = (relay, first_target_bound_permit.take()) {
                    relay.cancel_direct_peer_delivery(unix_now_secs(), permit);
                }
                last_failure_reason = Some("peer_relay_half_open_probe_unavailable".to_string());
                break;
            }
            let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
                let _ = peer_store.record_route_forward_failure_for_descriptor(
                    &peer,
                    now,
                    "missing_endpoint",
                );
                continue;
            };
            // [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] Select the strict
            // endpoint only from the target's verified signed descriptor.
            // Legacy peers retain v1 during rolling upgrades; an unsigned HTTP
            // response can never upgrade or downgrade this decision.
            let use_authenticated_relay = use_target_bound_relay
                || peer
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayAuthV2);
            let require_signed_receipt = use_authenticated_relay
                && peer
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayReceiptV2);
            let url = if use_target_bound_relay {
                Self::chat_peer_relay_v3_url(endpoint)
            } else if use_authenticated_relay {
                Self::chat_peer_relay_v2_url(endpoint)
            } else {
                Self::chat_peer_relay_url(endpoint)
            };
            let Some(url) = url else {
                let _ = peer_store.record_route_forward_failure_for_descriptor(
                    &peer,
                    now,
                    "invalid_endpoint",
                );
                continue;
            };

            let delivery_permit = if use_target_bound_relay {
                if let Some(relay) = relay {
                    let permit = first_target_bound_permit
                        .take()
                        .or_else(|| relay.begin_direct_peer_delivery(unix_now_secs()));
                    let Some(permit) = permit else {
                        last_failure_reason = Some("peer_relay_circuit_open".to_string());
                        break;
                    };
                    Some(permit)
                } else {
                    None
                }
            } else {
                None
            };
            let stop_after_delivery = delivery_permit.is_some_and(|permit| permit.is_half_open());
            let mut circuit_allows_more = true;
            let delivery_result =
                if use_target_bound_relay {
                    // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Build
                    // inside the peer loop because the signed target differs for
                    // every candidate. Reusing one request would recreate the v2
                    // cross-node replay boundary this contract closes.
                    let Some(signing_identity) = signing_identity.as_ref() else {
                        if let (Some(relay), Some(permit)) = (relay, delivery_permit) {
                            relay.cancel_direct_peer_delivery(unix_now_secs(), permit);
                        }
                        last_failure_reason = Some("peer_relay_auth_encode_failed".to_string());
                        break;
                    };
                    let prepared_request = prepare_peer_chat_relay_request_v3(
                        envelope.clone(),
                        peer.node_id(),
                        Arc::clone(signing_identity),
                    )
                    .await;
                    let request = match prepared_request {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            if let (Some(relay), Some(permit)) = (relay, delivery_permit) {
                                relay.cancel_direct_peer_delivery(unix_now_secs(), permit);
                            }
                            let reason = error.reason_bucket().to_string();
                            last_failure_reason = Some(reason);
                            break;
                        }
                    };
                    let request_commitment = request.request_commitment();
                    // [DIRECT-RELAY-ATTEMPT-BOUNDARY 2026-08-31 by Codex] Local
                    // request preparation cannot affect peer reputation. Count an
                    // attempt only after the exact signed request is ready and the
                    // HTTP transport is about to observe it.
                    attempted += 1;
                    let outcome = Self::send_and_validate_target_bound_peer_relay(
                        client,
                        &url,
                        &request,
                        &request_commitment,
                        &peer.node_id(),
                        require_signed_receipt,
                    )
                    .await;
                    if let (Some(relay), Some(permit)) = (relay, delivery_permit) {
                        if outcome.local_runtime_failure() {
                            // [DIRECT-RELAY-LOCAL-FAULT-ATTRIBUTION 2026-08-31 by
                            // Codex] Local verifier capacity says nothing about the
                            // selected peer. Release half-open ownership without
                            // advancing outage evidence.
                            relay.cancel_direct_peer_delivery(unix_now_secs(), permit);
                        } else {
                            circuit_allows_more = relay.complete_direct_peer_delivery(
                                unix_now_secs(),
                                permit,
                                outcome.retry_triggered(),
                                outcome.delivery_succeeded(),
                                outcome.final_failure_deterministic(),
                            );
                        }
                    }
                    outcome.result.map_err(|error| match error {
                        TargetBoundPeerRelayFailure::Transport(error) => {
                            DirectPeerRelayDeliveryFailure::selected_peer(
                                Self::classify_reqwest_error("peer_relay_request", &error),
                            )
                        }
                        TargetBoundPeerRelayFailure::Http(status) => {
                            DirectPeerRelayDeliveryFailure::selected_peer(format!(
                                "peer_relay_http_{}",
                                status.as_u16()
                            ))
                        }
                        TargetBoundPeerRelayFailure::Ack(error) => {
                            DirectPeerRelayDeliveryFailure::from_ack(error)
                        }
                    })
                } else {
                    let (response, expected_request_commitment) = if use_authenticated_relay {
                        let Some(prepared) = authenticated_request.as_ref() else {
                            let reason = "peer_relay_auth_encode_failed".to_string();
                            last_failure_reason = Some(reason);
                            continue;
                        };
                        let request = match prepared {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let reason = error.reason_bucket().to_string();
                                last_failure_reason = Some(reason);
                                continue;
                            }
                        };
                        let request_commitment = request.request_commitment();
                        attempted += 1;
                        (
                            client
                                .post(&url)
                                .header(reqwest::header::CONTENT_TYPE, "application/json")
                                .body(request.body())
                                .send()
                                .await,
                            Some(request_commitment),
                        )
                    } else {
                        let Some(prepared) = legacy_request.as_ref() else {
                            last_failure_reason = Some("peer_relay_auth_encode_failed".to_string());
                            continue;
                        };
                        let request = match prepared {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                last_failure_reason = Some(error.reason_bucket().to_string());
                                continue;
                            }
                        };
                        attempted += 1;
                        (
                            client
                                .post(&url)
                                .header(reqwest::header::CONTENT_TYPE, "application/json")
                                .body(request.body())
                                .send()
                                .await,
                            None,
                        )
                    };
                    match response {
                        Ok(response) if response.status().is_success() => {
                            Self::validate_direct_peer_relay_ack_typed(
                                response,
                                expected_request_commitment.as_ref(),
                                &peer.node_id(),
                                require_signed_receipt,
                                unix_now_secs(),
                            )
                            .await
                            .map_err(DirectPeerRelayDeliveryFailure::from_ack)
                        }
                        Ok(response) => Err(DirectPeerRelayDeliveryFailure::selected_peer(
                            format!("peer_relay_http_{}", response.status().as_u16()),
                        )),
                        Err(error) => Err(DirectPeerRelayDeliveryFailure::selected_peer(
                            Self::classify_reqwest_error("peer_relay_request", &error),
                        )),
                    }
                };

            let observed_at = unix_now_secs();
            match delivery_result {
                Ok(()) => {
                    accepted += 1;
                    let _ =
                        peer_store.record_route_forward_success_for_descriptor(&peer, observed_at);
                }
                Err(failure) => {
                    if failure.affects_peer_reputation() {
                        let _ = peer_store.record_route_forward_failure_for_descriptor(
                            &peer,
                            observed_at,
                            failure.reason.clone(),
                        );
                    }
                    last_failure_reason = Some(failure.reason.clone());
                    debug!(
                        reason = failure.reason,
                        "[CHAT_RELAY] Peer relay delivery failed"
                    );
                }
            }
            if stop_after_delivery || !circuit_allows_more {
                break;
            }
        }

        if let (Some(relay), Some(permit)) = (relay, first_target_bound_permit.take()) {
            if permit.is_half_open() && last_failure_reason.is_none() {
                last_failure_reason = Some("peer_relay_half_open_probe_unavailable".to_string());
            }
            relay.cancel_direct_peer_delivery(unix_now_secs(), permit);
        }

        if let Some(relay) = relay {
            // [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Never send
            // raw transport or peer-returned text into heartbeat telemetry.
            relay.record_peer_relay_outbound_typed(
                now,
                attempted,
                accepted,
                last_failure_reason
                    .as_deref()
                    .map(ChatRelayOutboundFailureReason::from_bucket),
            );
        }

        if attempted > 0 {
            debug!(
                attempted,
                accepted, "[CHAT_RELAY] Peer relay fanout complete"
            );
        }

        accepted
    }

    pub(super) fn chat_peer_relay_url(endpoint: &str) -> Option<String> {
        Self::permissionless_peer_transport_url(endpoint, "/api/chat/peer/relay")
    }

    pub(super) fn chat_peer_relay_v2_url(endpoint: &str) -> Option<String> {
        Self::permissionless_peer_transport_url(endpoint, "/api/chat/peer/relay-v2")
    }

    pub(super) fn chat_peer_relay_v3_url(endpoint: &str) -> Option<String> {
        Self::permissionless_peer_transport_url(endpoint, "/api/chat/peer/relay-v3")
    }

    /// Derives an outbound route only for a safe permissionless descriptor.
    pub(super) fn permissionless_peer_transport_url(endpoint: &str, path: &str) -> Option<String> {
        // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] Keep real chat movement,
        // blind probes, and onion forwarding on the same endpoint boundary as
        // discovery and MemChain. Localhost is a test-only transport seam.
        if !peer_endpoint_is_public_ip(endpoint) {
            #[cfg(not(test))]
            return None;
            #[cfg(test)]
            if !crate::api::peer_endpoint_is_loopback_ip(endpoint) {
                return None;
            }
        }
        canonical_peer_http_url(endpoint, path)
            .ok()
            .map(|url| url.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_bound_retry_policy_excludes_deterministic_protocol_failures() {
        // [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] A bounded exact retry
        // may recover ambiguous delivery, but it must not amplify explicit
        // rejection, oversized bodies, or cryptographic contract failures.
        for error in [
            BoundedHttpResponseError::BodyRead,
            BoundedHttpResponseError::JsonDecode,
        ] {
            assert!(
                TargetBoundPeerRelayFailure::Ack(DirectPeerRelayAckFailure::Bounded(error))
                    .retryable_after_ambiguous_delivery()
            );
        }
        for error in [
            DirectPeerRelayAckFailure::Bounded(BoundedHttpResponseError::TooLarge),
            DirectPeerRelayAckFailure::Rejected,
            DirectPeerRelayAckFailure::ReceiptRequestMissing,
            DirectPeerRelayAckFailure::ReceiptMissing,
            DirectPeerRelayAckFailure::ReceiptInvalid("receipt_signature_invalid"),
            // [DIRECT-RECEIPT-LOCAL-FAILURE 2026-08-31 by Codex] A complete
            // signed response already crossed the network boundary. Repeating
            // the request cannot restore a failed local verifier worker.
            DirectPeerRelayAckFailure::VerificationUnavailable,
        ] {
            assert!(!TargetBoundPeerRelayFailure::Ack(error).retryable_after_ambiguous_delivery());
        }
        assert!(TargetBoundPeerRelayFailure::Http(
            reqwest::StatusCode::from_u16(HTTP_TOO_EARLY_STATUS_CODE).unwrap()
        )
        .retryable_after_ambiguous_delivery());
        assert!(
            !TargetBoundPeerRelayFailure::Http(reqwest::StatusCode::TOO_MANY_REQUESTS)
                .retryable_after_ambiguous_delivery()
        );
    }

    #[test]
    fn authenticated_chat_partial_replica_success_is_terminal_delivery() {
        // [CLIENT-ONION-PARTIAL-SUCCESS 2026-07-31 by Codex] Once any
        // independently selected terminal signs acceptance of the exact
        // opaque payload, direct-relay fallback would only duplicate delivery
        // and expose additional routing metadata. Replica completeness remains
        // a separate durability signal.
        let partial = super::AuthenticatedChatOnionRelayOutcome {
            attempted_paths: 2,
            verified_receipts: 1,
            first_terminal_receipt: None,
        };
        assert!(partial.delivered());
        assert!(!partial.fully_replicated());

        let failed = super::AuthenticatedChatOnionRelayOutcome {
            attempted_paths: 2,
            verified_receipts: 0,
            first_terminal_receipt: None,
        };
        assert!(!failed.delivered());
        assert!(!failed.fully_replicated());
        assert!(!failed.compatibility_direct_fallback_allowed());

        // [ONION-FALLBACK-PRIVACY 2026-08-15 by Codex] Mixed-version or cold
        // meshes may use compatibility relay only when onion preflight sent
        // the envelope to no peer at all.
        let preflight_only = super::AuthenticatedChatOnionRelayOutcome::default();
        assert!(preflight_only.compatibility_direct_fallback_allowed());
    }

    #[test]
    fn chat_peer_relay_url_normalizes_endpoint_forms() {
        assert_eq!(
            Server::chat_peer_relay_url("8.8.8.8:8421/ignored?secret=no").as_deref(),
            Some("http://8.8.8.8:8421/api/chat/peer/relay")
        );
        assert_eq!(
            Server::blind_relay_probe_url("https://[2606:4700:4700::1111]:8421/").as_deref(),
            Some("https://[2606:4700:4700::1111]:8421/api/chat/peer/blind-relay")
        );
        assert!(Server::chat_peer_relay_url("http://127.0.0.1:8421").is_some());
        for endpoint in [
            "https://node.example.com/",
            "http://10.0.0.1:8421",
            "http://169.254.169.254/latest/meta-data",
            "http://203.0.113.1:8421",
        ] {
            assert_eq!(
                Server::chat_peer_relay_url(endpoint),
                None,
                "unexpectedly accepted {endpoint}"
            );
        }
        assert_eq!(Server::chat_peer_relay_url("   "), None);
    }
}
