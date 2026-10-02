// [ARCH-SPLIT 2026-10-02]
// Join, snapshot, gossip, and route-domain certificate ingress.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

// ============================================
// Handlers
// ============================================

pub(super) async fn open_node_admission_handler(
    State(state): State<DiscoveryApiState>,
    body: Bytes,
) -> axum::response::Response {
    let now = now_secs();
    if !state
        .node_admission_rate_limit
        .lock()
        .allow(now, state.policy.gossip_rate_limit_per_minute)
    {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(OpenNodeAdmissionResponse::new(false, "rate_limited")),
        )
            .into_response();
    }

    // [OPEN-NODE-ADMISSION 2026-09-24 by Codex] Decode the exact bounded wire
    // bytes before policy or store mutation. This endpoint does not accept a
    // JSON projection whose ignored/duplicate fields could obscure the signed
    // canonical transcript.
    let descriptor = match SignedNodeDescriptor::decode_canonical(&body) {
        Ok(descriptor) => descriptor,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenNodeAdmissionResponse::new(false, "rejected")),
            )
                .into_response();
        }
    };
    if state.policy.node_denied(&descriptor.node_id()) {
        return (
            StatusCode::FORBIDDEN,
            Json(OpenNodeAdmissionResponse::new(false, "rejected")),
        )
            .into_response();
    }

    let outcome = state
        .peer_store
        .admit_permissionless_descriptor(descriptor, now);
    let (status_code, response) = match outcome {
        PermissionlessNodeAdmissionOutcome::Admitted => (
            StatusCode::OK,
            OpenNodeAdmissionResponse::new(true, "candidate_admitted"),
        ),
        PermissionlessNodeAdmissionOutcome::ExactReplay => (
            StatusCode::OK,
            OpenNodeAdmissionResponse::new(true, "exact_replay"),
        ),
        PermissionlessNodeAdmissionOutcome::Stale => (
            StatusCode::CONFLICT,
            OpenNodeAdmissionResponse::new(false, "stale"),
        ),
        PermissionlessNodeAdmissionOutcome::Conflict => (
            StatusCode::CONFLICT,
            OpenNodeAdmissionResponse::new(false, "conflict"),
        ),
        PermissionlessNodeAdmissionOutcome::Saturated => (
            StatusCode::SERVICE_UNAVAILABLE,
            OpenNodeAdmissionResponse::new(false, "capacity_reached"),
        ),
        PermissionlessNodeAdmissionOutcome::Rejected => (
            StatusCode::BAD_REQUEST,
            OpenNodeAdmissionResponse::new(false, "rejected"),
        ),
    };
    (status_code, Json(response)).into_response()
}

pub(super) async fn snapshot_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<SnapshotQuery>,
) -> Json<NodeBootstrapSnapshot> {
    let now = now_secs();
    let limit = state.policy.snapshot_limit(query.limit);
    let _requested_public_only = query.public_only;
    // [PUBLIC-DISCOVERY-PROJECTION 2026-09-01 by Codex] Keep accepting the
    // legacy `public_only` query field, but never let an unauthenticated public
    // request downgrade the descriptor-visibility boundary. Host-local cache
    // persistence continues to call `PeerStore` directly when it needs the
    // complete private descriptor set.
    Json(
        state
            .peer_store
            .export_bootstrap_snapshot(now, now, true, Some(limit)),
    )
}

/// Imports one portable route-domain certificate under exact host-local pins.
///
/// [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] The transport sender
/// is intentionally not authority: any peer may carry the bounded frame, while
/// only signatures from locally pinned independent attestors count. Responses
/// and audit events remain identity-blind.
pub(super) async fn route_domain_certificate_handler(
    State(state): State<DiscoveryApiState>,
    body: Bytes,
) -> impl IntoResponse {
    let now = now_secs();
    let rate_limit = state
        .policy
        .route_domain_certificate_rate_limit_per_minute();
    if !state
        .route_domain_certificate_rate_limit
        .lock()
        .allow(now, rate_limit)
    {
        state.peer_store.record_audit_event(
            now,
            "route_domain_certificate_import",
            "rate_limited",
            "reason=bounded_ingress_limit",
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(RouteDomainCertificateImportResponse {
                accepted: false,
                stored: false,
                status: "rate_limited",
            }),
        )
            .into_response();
    }

    let certificate = match decode_route_domain_attestation_certificate(&body) {
        Ok(certificate) => certificate,
        Err(_) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=malformed_frame",
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "malformed_certificate",
                }),
            )
                .into_response();
        }
    };

    match state
        .peer_store
        .import_route_domain_attestation_certificate(certificate, now)
    {
        Ok(stored) => {
            let status = if stored { "stored" } else { "already_present" };
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "accepted",
                format!("result={status}"),
            );
            (
                StatusCode::OK,
                Json(RouteDomainCertificateImportResponse {
                    accepted: true,
                    stored,
                    status,
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::Rejected) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=local_policy_verification_failed",
            );
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "certificate_rejected",
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::Stale) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=stale_evidence",
            );
            (
                StatusCode::CONFLICT,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "stale_certificate",
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::CapacityExceeded) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=bounded_cache_capacity",
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "certificate_capacity_reached",
                }),
            )
                .into_response()
        }
    }
}

pub(super) async fn gossip_handler(
    State(state): State<DiscoveryApiState>,
    Json(message): Json<NodeDiscoveryMessage>,
) -> impl IntoResponse {
    let now = now_secs();
    let is_endpoint_attestation = matches!(
        &message,
        NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. }
    );
    if !state
        .rate_limit
        .lock()
        .allow(now, state.policy.gossip_rate_limit_per_minute)
    {
        if !is_endpoint_attestation {
            state.peer_store.record_rate_limited(
                now,
                format!(
                    "global_limit_per_minute={}",
                    state.policy.gossip_rate_limit_per_minute
                ),
            );
        }
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    if let NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { attestation_frame } = &message {
        return persist_or_discard_endpoint_attestation(&state, attestation_frame, now).await;
    }

    if !state.policy.message_allowed(&message) {
        state.peer_store.record_policy_rejected(
            now,
            format!(
                "allow_list_enabled={} allowed_peer_count={} denied_peer_count={}",
                !state.policy.allowed_peer_ids.is_empty(),
                state.policy.allowed_peer_ids.len(),
                state.policy.denied_peer_ids.len()
            ),
        );
        return (
            StatusCode::FORBIDDEN,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    let (admission_status, applied) = apply_gossip_message(&state, &message, now);
    state.peer_store.mark_gossip_at(now);
    if admission_status != StatusCode::OK {
        return (
            admission_status,
            Json(GossipResponse {
                applied,
                response: None,
            }),
        )
            .into_response();
    }
    let response = match message {
        NodeDiscoveryMessage::SnapshotRequest { limit, .. } => {
            Some(state.peer_store.build_snapshot_response(
                now,
                now,
                true,
                Some(state.policy.snapshot_limit(limit.map(usize::from))),
            ))
        }
        NodeDiscoveryMessage::SnapshotResponse { .. }
        | NodeDiscoveryMessage::DescriptorAnnounce { .. }
        | NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. }
        | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => None,
    };

    (StatusCode::OK, Json(GossipResponse { applied, response })).into_response()
}

// [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX-COMPOSITION 2026-09-24 by Codex]
// All policy locks are released before canonical verification and blocking
// storage. Every outcome returns before any PeerStore mutation.
pub(super) async fn persist_or_discard_endpoint_attestation(
    state: &DiscoveryApiState,
    frame: &[u8],
    now: u64,
) -> axum::response::Response {
    let empty = || GossipResponse {
        applied: PeerStoreImportReport::empty(),
        response: None,
    };
    let decoded = match DiscoveryEndpointEvidenceAttestationV1::decode(frame) {
        Ok(decoded) => decoded,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(empty())).into_response(),
    };
    if !state.policy.node_allowed(&decoded.subject_node_id()) {
        return (StatusCode::FORBIDDEN, Json(empty())).into_response();
    }
    let context = public_endpoint_flow_context(decoded.observer_node_id());
    let verified = match VerifiedDiscoveryEndpointAttestationV1::verify(frame, now, context) {
        Ok(verified) => verified,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(empty())).into_response(),
    };
    let Some(inbox) = state.endpoint_attestation_inbox.clone() else {
        return (StatusCode::OK, Json(empty())).into_response();
    };
    let outcome =
        tokio::task::spawn_blocking(move || inbox.record_verified_at(&verified, now)).await;
    let status = match outcome {
        Ok(Ok(
            DiscoveryEndpointAttestationRecordOutcome::Inserted
            | DiscoveryEndpointAttestationRecordOutcome::Existing,
        )) => StatusCode::OK,
        Ok(Ok(DiscoveryEndpointAttestationRecordOutcome::Conflict)) => StatusCode::CONFLICT,
        Ok(Ok(DiscoveryEndpointAttestationRecordOutcome::AtCapacity))
        | Ok(Err(
            DiscoveryEndpointAttestationInboxError::UnsupportedSchema
            | DiscoveryEndpointAttestationInboxError::Corrupt
            | DiscoveryEndpointAttestationInboxError::Unavailable,
        ))
        | Err(_) => StatusCode::SERVICE_UNAVAILABLE,
        Ok(Err(DiscoveryEndpointAttestationInboxError::Rejected)) => StatusCode::BAD_REQUEST,
    };
    (status, Json(empty())).into_response()
}

pub(super) fn apply_gossip_message(
    state: &DiscoveryApiState,
    message: &NodeDiscoveryMessage,
    now: u64,
) -> (StatusCode, PeerStoreImportReport) {
    let NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
        producer,
        block_hash,
        descriptor_hash,
        proof,
    } = message
    else {
        return (
            StatusCode::OK,
            state.peer_store.apply_discovery_message(message, now),
        );
    };

    // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] A node without an
    // audited replica cannot authenticate this stronger gossip contract. Do
    // not silently downgrade it to signature-only DescriptorAnnounce.
    let Some(replica_store) = state.directory_replica_store.as_deref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            state.peer_store.record_rejected_directory_proof_import(now),
        );
    };
    match admit_directory_gossip_descriptor(
        replica_store,
        &state.peer_store,
        proof,
        producer,
        block_hash,
        descriptor_hash,
        now,
    ) {
        Ok(report) => (StatusCode::OK, report),
        Err(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            state.peer_store.record_rejected_directory_proof_import(now),
        ),
    }
}
