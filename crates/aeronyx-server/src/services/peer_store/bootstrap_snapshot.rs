// [ARCH-SPLIT 2026-10-02]
// Bootstrap snapshot export and expiry cleanup.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Exports valid descriptors as a bootstrap snapshot for gossip response.
    ///
    /// When `public_only` is true, descriptors with `public_discovery=false`
    /// are excluded. `limit` caps the number of descriptors returned.
    #[must_use]
    pub fn export_bootstrap_snapshot(
        &self,
        generated_at: u64,
        now: u64,
        public_only: bool,
        limit: Option<usize>,
    ) -> NodeBootstrapSnapshot {
        self.counters
            .last_snapshot_at
            .store(generated_at, Ordering::Relaxed);
        let mut descriptors: Vec<SignedNodeDescriptor> = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, now, true))
            .filter(|descriptor| !public_only || descriptor.descriptor.policy.public_discovery)
            .cloned()
            .collect();

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        if let Some(limit) = limit {
            descriptors.truncate(limit);
        }

        self.record_audit_event(
            generated_at,
            "snapshot_export",
            "accepted",
            format!(
                "public_only={} limit={} exported={}",
                public_only,
                limit.map_or_else(|| "none".to_string(), |value| value.to_string()),
                descriptors.len()
            ),
        );

        NodeBootstrapSnapshot::new(generated_at, descriptors)
    }

    /// Exports a bounded verifiable peer-record snapshot for heartbeat.
    ///
    /// Heartbeat consumers should verify every descriptor in `records` using
    /// `SignedNodeDescriptor::verify_at(generated_at)` before accepting peer
    /// claims. This method deliberately exports signed node-level discovery
    /// metadata, not client/session traffic. It filters out expired or invalid
    /// descriptors and does not include retained expired cache history.
    #[must_use]
    pub fn export_signed_peer_records_for_heartbeat(
        &self,
        generated_at: u64,
        limit: Option<usize>,
    ) -> PeerStoreSignedPeerRecordsStatus {
        let peers = self.peers.read();
        let total_retained_records = peers.len();
        let mut descriptors: Vec<SignedNodeDescriptor> = peers
            .values()
            .filter(|descriptor| descriptor.verify_at(generated_at).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, generated_at, true))
            .cloned()
            .collect();
        drop(peers);

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        let valid_signed_records = descriptors.len();
        if let Some(limit) = limit {
            descriptors.truncate(limit);
        }
        let exported_signed_records = descriptors.len();

        self.record_audit_event(
            generated_at,
            "heartbeat_signed_peer_records_export",
            "accepted",
            format!(
                "retained={} valid={} exported={} limit={}",
                total_retained_records,
                valid_signed_records,
                exported_signed_records,
                limit.map_or_else(|| "none".to_string(), |value| value.to_string())
            ),
        );

        PeerStoreSignedPeerRecordsStatus {
            generated_at,
            source: "rust_peer_store_signed_descriptors".to_string(),
            total_retained_records,
            valid_signed_records,
            exported_signed_records,
            limit,
            records: NodeBootstrapSnapshot::new(generated_at, descriptors),
            verification_rule:
                "verify each records.peers[] with SignedNodeDescriptor::verify_at(generated_at)"
                    .to_string(),
            privacy_boundary: "signed node discovery descriptors only; may include node public keys and public endpoints, but no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    /// Builds a discovery snapshot response from current valid peers.
    #[must_use]
    pub fn build_snapshot_response(
        &self,
        generated_at: u64,
        now: u64,
        public_only: bool,
        limit: Option<usize>,
    ) -> NodeDiscoveryMessage {
        NodeDiscoveryMessage::SnapshotResponse {
            snapshot: self.export_bootstrap_snapshot(generated_at, now, public_only, limit),
        }
    }

    /// Downgrades descriptors that are no longer valid at `now`.
    ///
    /// Ordinary expired peers are retained as signed, non-routeable local
    /// history. Permissionless promotions are different: their expiring deny
    /// gate and exact live descriptor are removed together, so dropping the
    /// gate can never reveal an otherwise routeable historic descriptor.
    pub fn cleanup_expired(&self, now: u64) -> usize {
        let promotion_removed = self.prune_expired_permissionless_promotions(now);
        let expired_candidates = {
            let mut candidates = self.untrusted_discovery_candidates.write();
            let before = candidates.candidates.len();
            Self::prune_untrusted_candidate_state(&mut candidates, now);
            before.saturating_sub(candidates.candidates.len())
        };
        let expired: Vec<([u8; 32], u64)> = self
            .peers
            .read()
            .iter()
            .filter(|(_, descriptor)| descriptor.verify_at(now).is_err())
            .map(|(node_id, descriptor)| (*node_id, descriptor.sequence()))
            .collect();

        let mut newly_degraded = Vec::new();
        if !expired.is_empty() {
            let mut metadata = self.peer_runtime.write();
            for (node_id, sequence) in expired {
                let entry = metadata
                    .entry(node_id)
                    .or_insert_with(|| PeerRuntimeMetadata {
                        source: "unknown".to_string(),
                        first_seen_at: now,
                        last_seen_at: now,
                        last_sequence: sequence,
                        imported_count: 0,
                        expired_degraded_at: None,
                    });
                if entry.expired_degraded_at.is_none() {
                    entry.expired_degraded_at = Some(now);
                    newly_degraded.push((node_id, sequence));
                }
            }
        }

        let degraded_count = newly_degraded.len();
        if degraded_count > 0 || expired_candidates > 0 || promotion_removed > 0 {
            self.counters
                .expired_degraded
                .fetch_add(degraded_count as u64, Ordering::Relaxed);
            self.counters.last_cleanup_at.store(now, Ordering::Relaxed);
            for (node_id, sequence) in &newly_degraded {
                self.record_peer_event(
                    now,
                    "peer_expired",
                    "degraded",
                    "cleanup",
                    node_id,
                    Some(*sequence),
                    Some("descriptor_expired_retained"),
                );
            }
            self.record_audit_event(
                now,
                "expired_peer_cleanup",
                "accepted",
                format!(
                    "degraded={degraded_count} candidate_released={expired_candidates} retained_total={} removed={promotion_removed}",
                    self.peers.read().len()
                ),
            );
        }
        degraded_count
            .saturating_add(expired_candidates)
            .saturating_add(promotion_removed)
    }
}
