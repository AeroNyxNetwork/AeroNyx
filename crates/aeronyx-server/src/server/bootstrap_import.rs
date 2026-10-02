// [ARCH-SPLIT 2026-10-02]
// Signed peer-cache snapshot restore. Startup calls this before gossip tasks.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    pub(super) fn import_bootstrap_snapshot_bytes(
        peer_store: &PeerStore,
        source_kind: &'static str,
        source: &str,
        bytes: &[u8],
        now: u64,
        cache_identity: Option<&IdentityKeyPair>,
    ) -> bool {
        Self::import_bootstrap_snapshot_bytes_with_anchor(
            peer_store,
            source_kind,
            source,
            bytes,
            now,
            cache_identity,
            &PeerStoreVerifiedClientDeliveryAnchorState::NotChecked,
        )
    }

    pub(super) fn import_bootstrap_snapshot_bytes_with_anchor(
        peer_store: &PeerStore,
        source_kind: &'static str,
        source: &str,
        bytes: &[u8],
        now: u64,
        cache_identity: Option<&IdentityKeyPair>,
        client_delivery_anchor: &PeerStoreVerifiedClientDeliveryAnchorState,
    ) -> bool {
        let is_peer_cache = matches!(source_kind, "cache" | "cache_backup");
        let parsed = if is_peer_cache {
            PeerStoreCacheDocument::from_json_bytes(bytes).map(|document| {
                let routeability_authentication =
                    if document.routeability_evidence_schema_version == 0 {
                        "legacy_descriptor_only"
                    } else if cache_identity.is_some_and(|identity| {
                        document
                            .verify_routeability_evidence_signature(identity)
                            .is_ok()
                    }) {
                        "verified"
                    } else if cache_identity.is_some() {
                        "signature_invalid"
                    } else {
                        "identity_unavailable"
                    };
                let two_hop_proof_authentication =
                    if document.two_hop_path_proof_schema_version == 0 {
                        "legacy_descriptor_only"
                    } else if cache_identity.is_some_and(|identity| {
                        document
                            .verify_two_hop_path_proof_signature(identity)
                            .is_ok()
                    }) {
                        "verified"
                    } else if cache_identity.is_some() {
                        "signature_invalid"
                    } else {
                        "identity_unavailable"
                    };
                let three_hop_proof_authentication =
                    if document.three_hop_path_proof_schema_version == 0 {
                        "legacy_descriptor_only"
                    } else if cache_identity.is_some_and(|identity| {
                        document
                            .verify_three_hop_path_proof_signature(identity)
                            .is_ok()
                    }) {
                        "verified"
                    } else if cache_identity.is_some() {
                        "signature_invalid"
                    } else {
                        "identity_unavailable"
                    };
                let client_delivery_authentication =
                    if document.verified_client_delivery_schema_version == 0 {
                        "legacy_descriptor_only"
                    } else if cache_identity.is_some_and(|identity| {
                        document
                            .verify_verified_client_delivery_signature(identity)
                            .is_ok()
                    }) {
                        "verified"
                    } else if cache_identity.is_some() {
                        "signature_invalid"
                    } else {
                        "identity_unavailable"
                    };
                let client_delivery_generation = document.verified_client_delivery_generation;
                let client_delivery_rollback_protection =
                    if client_delivery_authentication == "verified" {
                        client_delivery_anchor.protection_for(&document)
                    } else {
                        "not_checked"
                    };
                // [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] The same
                // monotonic generation now protects the independently signed
                // routeability/quarantine section. Authentication and rollback
                // remain separate decisions so diagnostics stay precise.
                let routeability_rollback_protection = if routeability_authentication == "verified"
                {
                    client_delivery_anchor.route_state_protection_for(&document)
                } else {
                    "not_checked"
                };
                // [PATH-PROOF-ROLLBACK-ANCHOR 2026-08-02 by Codex] The same
                // monotonic recovery anchor commits to opaque digests of both
                // proof sections. This remains an independent failure domain:
                // descriptors and aggregate delivery evidence keep their own
                // authentication and restore decisions.
                let two_hop_proof_rollback_protection =
                    client_delivery_anchor.two_hop_path_proof_protection_for(&document);
                let three_hop_proof_rollback_protection =
                    client_delivery_anchor.three_hop_path_proof_protection_for(&document);
                let route_domain_certificates = (document.route_domain_certificate_schema_version
                    != 0)
                    .then_some(document.route_domain_certificates);
                let route_quarantine_evidence = (document.routeability_evidence_schema_version
                    == ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION)
                    .then_some(document.route_quarantine_evidence);
                (
                    document.descriptor_snapshot,
                    Some(document.routeability_evidence),
                    route_quarantine_evidence,
                    routeability_authentication,
                    routeability_rollback_protection,
                    route_domain_certificates,
                    Some(document.two_hop_path_proof_events),
                    two_hop_proof_authentication,
                    Some(document.three_hop_path_proof_events),
                    three_hop_proof_authentication,
                    document.verified_client_delivery_evidence,
                    client_delivery_authentication,
                    client_delivery_generation,
                    client_delivery_rollback_protection,
                    two_hop_proof_rollback_protection,
                    three_hop_proof_rollback_protection,
                )
            })
        } else {
            NodeBootstrapSnapshot::from_json_bytes(bytes)
                .map(|snapshot| {
                    (
                        snapshot,
                        None,
                        None,
                        "not_applicable",
                        "not_applicable",
                        None,
                        None,
                        "not_applicable",
                        None,
                        "not_applicable",
                        None,
                        "not_applicable",
                        0,
                        "not_applicable",
                        "not_applicable",
                        "not_applicable",
                    )
                })
                .map_err(|error| error.to_string())
        };
        let (
            snapshot,
            routeability_evidence,
            route_quarantine_evidence,
            routeability_authentication,
            routeability_rollback_protection,
            route_domain_certificates,
            two_hop_proof_events,
            two_hop_proof_authentication,
            three_hop_proof_events,
            three_hop_proof_authentication,
            client_delivery_evidence,
            client_delivery_authentication,
            client_delivery_generation,
            client_delivery_rollback_protection,
            two_hop_proof_rollback_protection,
            three_hop_proof_rollback_protection,
        ) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                peer_store.record_bootstrap_source(now, source_kind, "failed", "json_rejected");
                warn!(
                    source_kind,
                    source = %source,
                    error = %error,
                    "[DISCOVERY] Bootstrap snapshot rejected"
                );
                return false;
            }
        };

        if is_peer_cache {
            peer_store.record_routeability_cache_rollback_protection(
                now,
                client_delivery_generation,
                routeability_rollback_protection,
            );
            peer_store.record_two_hop_proof_cache_authentication(now, two_hop_proof_authentication);
            peer_store
                .record_three_hop_proof_cache_authentication(now, three_hop_proof_authentication);
            peer_store
                .record_client_delivery_cache_authentication(now, client_delivery_authentication);
            peer_store.record_client_delivery_cache_rollback_protection(
                now,
                client_delivery_generation,
                client_delivery_rollback_protection,
            );
            peer_store.record_two_hop_proof_cache_rollback_protection(
                now,
                client_delivery_generation,
                two_hop_proof_rollback_protection,
            );
            peer_store.record_three_hop_proof_cache_rollback_protection(
                now,
                client_delivery_generation,
                three_hop_proof_rollback_protection,
            );
        }

        let report = if is_peer_cache {
            peer_store.load_peer_cache_snapshot_from_source(&snapshot, now, source_kind)
        } else {
            peer_store.load_bootstrap_snapshot_from_source(&snapshot, now, source_kind)
        };
        // [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] Restore is
        // deliberately independent from descriptor/proof sections. Every
        // record is rechecked against the policy installed before cache load.
        let route_domain_certificate_report =
            route_domain_certificates.as_deref().map(|certificates| {
                peer_store.restore_route_domain_attestation_certificates(certificates, now)
            });
        let route_domain_certificates_restored = route_domain_certificate_report
            .map(|value| value.restored)
            .unwrap_or(0);
        let route_domain_certificates_unchanged = route_domain_certificate_report
            .map(|value| value.unchanged)
            .unwrap_or(0);
        let route_domain_certificates_rejected = route_domain_certificate_report
            .map(|value| value.rejected)
            .unwrap_or(0);
        let route_report = match (
            routeability_authentication,
            routeability_rollback_protection,
            routeability_evidence.as_deref(),
        ) {
            ("signature_invalid", _, Some(records)) => {
                Some(peer_store.reject_routeability_cache_evidence(
                    records.len(),
                    now,
                    "signature_invalid",
                ))
            }
            ("identity_unavailable", _, Some(records)) => {
                Some(peer_store.reject_routeability_cache_evidence(
                    records.len(),
                    now,
                    "identity_unavailable",
                ))
            }
            (_, protection, Some(records))
                if !matches!(protection, "anchored" | "cache_ahead" | "not_checked") =>
            {
                Some(peer_store.reject_routeability_cache_evidence(records.len(), now, protection))
            }
            (_, _, Some(records)) => {
                Some(peer_store.restore_routeability_cache_evidence(records, now))
            }
            (_, _, None) => None,
        };
        // [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Cache v2 signs
        // positive routeability and active quarantine as one policy snapshot.
        // Authentication failure rejects both sections; an older signed v1
        // cache has no quarantine section and remains source-compatible.
        let route_quarantine_report = match (
            routeability_authentication,
            routeability_rollback_protection,
            route_quarantine_evidence.as_deref(),
        ) {
            ("signature_invalid", _, Some(records)) => {
                Some(peer_store.reject_route_quarantine_cache_evidence(
                    records.len(),
                    now,
                    "signature_invalid",
                ))
            }
            ("identity_unavailable", _, Some(records)) => {
                Some(peer_store.reject_route_quarantine_cache_evidence(
                    records.len(),
                    now,
                    "identity_unavailable",
                ))
            }
            (_, protection, Some(records))
                if !matches!(protection, "anchored" | "cache_ahead" | "not_checked") =>
            {
                Some(peer_store.reject_route_quarantine_cache_evidence(
                    records.len(),
                    now,
                    protection,
                ))
            }
            (_, _, Some(records)) => {
                Some(peer_store.restore_route_quarantine_cache_evidence(records, now))
            }
            (_, _, None) => None,
        };
        let route_restored = route_report.map(|value| value.restored).unwrap_or(0);
        let route_rejected = route_report.map(|value| value.rejected).unwrap_or(0);
        let route_quarantine_restored = route_quarantine_report
            .map(|value| value.restored)
            .unwrap_or(0);
        let route_quarantine_rejected = route_quarantine_report
            .map(|value| value.rejected)
            .unwrap_or(0);
        let route_authentication_rejected = matches!(
            routeability_authentication,
            "signature_invalid" | "identity_unavailable"
        );
        let route_rollback_rejected = !matches!(
            routeability_rollback_protection,
            "anchored" | "cache_ahead" | "not_checked" | "not_applicable"
        );
        // Routeability must be restored first: proof history is accepted only
        // when the current signed descriptors still form a complete distinct
        // middle/terminal path.
        let proof_report = match (
            two_hop_proof_authentication,
            two_hop_proof_events.as_deref(),
        ) {
            ("signature_invalid", Some(records)) => {
                Some(peer_store.reject_two_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    "signature_invalid",
                ))
            }
            ("identity_unavailable", Some(records)) => {
                Some(peer_store.reject_two_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    "identity_unavailable",
                ))
            }
            (_, Some(records)) if records.is_empty() => {
                Some(peer_store.restore_two_hop_path_proof_cache_events(records, now))
            }
            (_, Some(records))
                if !matches!(
                    two_hop_proof_rollback_protection,
                    "anchored" | "cache_ahead"
                ) =>
            {
                Some(peer_store.reject_two_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    two_hop_proof_rollback_protection,
                ))
            }
            (_, Some(records)) => {
                Some(peer_store.restore_two_hop_path_proof_cache_events(records, now))
            }
            (_, None) => None,
        };
        let proof_restored = proof_report.map(|value| value.restored).unwrap_or(0);
        let proof_rejected = proof_report.map(|value| value.rejected).unwrap_or(0);
        let proof_authentication_rejected = matches!(
            two_hop_proof_authentication,
            "signature_invalid" | "identity_unavailable"
        );
        let three_hop_proof_report = match (
            three_hop_proof_authentication,
            three_hop_proof_events.as_deref(),
        ) {
            ("signature_invalid", Some(records)) => {
                Some(peer_store.reject_three_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    "signature_invalid",
                ))
            }
            ("identity_unavailable", Some(records)) => {
                Some(peer_store.reject_three_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    "identity_unavailable",
                ))
            }
            (_, Some(records)) if records.is_empty() => {
                Some(peer_store.restore_three_hop_path_proof_cache_events(records, now))
            }
            (_, Some(records))
                if !matches!(
                    three_hop_proof_rollback_protection,
                    "anchored" | "cache_ahead"
                ) =>
            {
                Some(peer_store.reject_three_hop_path_proof_cache_events(
                    records.len(),
                    now,
                    three_hop_proof_rollback_protection,
                ))
            }
            (_, Some(records)) => {
                Some(peer_store.restore_three_hop_path_proof_cache_events(records, now))
            }
            (_, None) => None,
        };
        let three_hop_proof_restored = three_hop_proof_report
            .map(|value| value.restored)
            .unwrap_or(0);
        let three_hop_proof_rejected = three_hop_proof_report
            .map(|value| value.rejected)
            .unwrap_or(0);
        let three_hop_proof_authentication_rejected = matches!(
            three_hop_proof_authentication,
            "signature_invalid" | "identity_unavailable"
        );
        let two_hop_proof_rollback_rejected = proof_report.is_some_and(|report| report.total > 0)
            && matches!(
                two_hop_proof_rollback_protection,
                "legacy_unanchored"
                    | "anchor_missing"
                    | "anchor_invalid"
                    | "anchor_conflict"
                    | "rollback_detected"
            );
        let three_hop_proof_rollback_rejected = three_hop_proof_report
            .is_some_and(|report| report.total > 0)
            && matches!(
                three_hop_proof_rollback_protection,
                "legacy_unanchored"
                    | "anchor_missing"
                    | "anchor_invalid"
                    | "anchor_conflict"
                    | "rollback_detected"
            );
        let client_delivery_report = if is_peer_cache {
            Some(
                match (
                    client_delivery_authentication,
                    client_delivery_rollback_protection,
                ) {
                    ("signature_invalid", _) => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "signature_invalid",
                        ),
                    ("identity_unavailable", _) => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "identity_unavailable",
                        ),
                    (_, "anchor_missing") => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "anchor_missing",
                        ),
                    (_, "anchor_invalid") => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "anchor_invalid",
                        ),
                    (_, "anchor_conflict") => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "anchor_conflict",
                        ),
                    (_, "rollback_detected") => peer_store
                        .reject_verified_client_delivery_cache_evidence(
                            client_delivery_evidence.is_some(),
                            now,
                            "rollback_detected",
                        ),
                    _ => peer_store.restore_verified_client_delivery_cache_evidence(
                        client_delivery_evidence.as_ref(),
                        now,
                    ),
                },
            )
        } else {
            None
        };
        let client_delivery_restored = client_delivery_report
            .map(|report| report.restored_deliveries)
            .unwrap_or(0);
        let client_delivery_rejected =
            client_delivery_report.is_some_and(|report| report.present && !report.restored);
        let client_delivery_authentication_rejected = matches!(
            client_delivery_authentication,
            "signature_invalid" | "identity_unavailable"
        );
        let client_delivery_rollback_rejected = matches!(
            client_delivery_rollback_protection,
            "anchor_missing" | "anchor_invalid" | "anchor_conflict" | "rollback_detected"
        );
        peer_store.record_bootstrap_source(
            now,
            source_kind,
            if report.rejected > 0
                || route_rejected > 0
                || route_quarantine_rejected > 0
                || route_authentication_rejected
                || route_rollback_rejected
                || route_domain_certificates_rejected > 0
                || proof_rejected > 0
                || proof_authentication_rejected
                || three_hop_proof_rejected > 0
                || three_hop_proof_authentication_rejected
                || two_hop_proof_rollback_rejected
                || three_hop_proof_rollback_rejected
                || client_delivery_rejected
                || client_delivery_authentication_rejected
                || client_delivery_rollback_rejected
            {
                "warning"
            } else {
                "success"
            },
            format!(
                "total={} inserted={} unchanged={} stale={} rejected={} routeability_authentication={} routeability_rollback_protection={} routeability_restored={} routeability_rejected={} route_quarantine_restored={} route_quarantine_rejected={} route_domain_certificates_restored={} route_domain_certificates_unchanged={} route_domain_certificates_rejected={} two_hop_proof_authentication={} two_hop_proof_restored={} two_hop_proof_rejected={} two_hop_proof_rollback_protection={} three_hop_proof_authentication={} three_hop_proof_restored={} three_hop_proof_rejected={} three_hop_proof_rollback_protection={} client_delivery_authentication={} client_delivery_generation={} client_delivery_rollback_protection={} client_delivery_restored={} client_delivery_rejected={}",
                report.total,
                report.inserted,
                report.unchanged,
                report.stale,
                report.rejected,
                routeability_authentication,
                routeability_rollback_protection,
                route_restored,
                route_rejected,
                route_quarantine_restored,
                route_quarantine_rejected,
                route_domain_certificates_restored,
                route_domain_certificates_unchanged,
                route_domain_certificates_rejected,
                two_hop_proof_authentication,
                proof_restored,
                proof_rejected,
                two_hop_proof_rollback_protection,
                three_hop_proof_authentication,
                three_hop_proof_restored,
                three_hop_proof_rejected,
                three_hop_proof_rollback_protection,
                client_delivery_authentication,
                client_delivery_generation,
                client_delivery_rollback_protection,
                client_delivery_restored,
                client_delivery_rejected
            ),
        );
        info!(
            source_kind,
            source = %source,
            total = report.total,
            inserted = report.inserted,
            unchanged = report.unchanged,
            stale = report.stale,
            rejected = report.rejected,
            routeability_authentication,
            routeability_rollback_protection,
            routeability_restored = route_restored,
            routeability_rejected = route_rejected,
            route_quarantine_restored,
            route_quarantine_rejected,
            route_domain_certificates_restored,
            route_domain_certificates_unchanged,
            route_domain_certificates_rejected,
            two_hop_proof_authentication,
            two_hop_proof_restored = proof_restored,
            two_hop_proof_rejected = proof_rejected,
            two_hop_proof_rollback_protection,
            three_hop_proof_authentication,
            three_hop_proof_restored,
            three_hop_proof_rejected,
            three_hop_proof_rollback_protection,
            client_delivery_authentication,
            client_delivery_generation,
            client_delivery_rollback_protection,
            client_delivery_restored,
            client_delivery_rejected,
            "[DISCOVERY] Bootstrap snapshot imported"
        );
        report.inserted > 0 || report.unchanged > 0
    }
}
